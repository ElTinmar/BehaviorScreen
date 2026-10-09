#!/usr/bin/env python3
"""Run Megabouts and create a stimulus-augmented bout table."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from megabouts.classification.classification import TailBouts
from megabouts.pipeline import FullTrackingPipeline
from megabouts.pipeline.freely_swimming_pipeline import (
    EthogramFullTracking,
)
from megabouts.preprocessing.tail_preprocessing import (
    TailPreprocessingResult,
)
from megabouts.preprocessing.traj_preprocessing import (
    TrajPreprocessingResult,
)
from megabouts.segmentation.segmentation import (
    SegmentationResult,
)
from megabouts.tracking_data import (
    FullTrackingData,
    TrackingConfig,
)

from BehaviorScreen.event_context import RecordingContext
from BehaviorScreen.load import (
    BehaviorData,
    BehaviorFiles,
    Directories,
    find_files,
    load_data,
)
from BehaviorScreen.protocol import bout_laterality


class MegaboutResults(NamedTuple):
    """Outputs returned by the Megabouts full-tracking pipeline."""

    timestamp: np.ndarray
    ethogram: EthogramFullTracking
    bouts: TailBouts
    segments: SegmentationResult
    tail: TailPreprocessingResult
    traj: TrajPreprocessingResult


def full_tracking_data_from_lp(
    dataframe: pd.DataFrame,
    millimeters_per_pixel: float,
) -> FullTrackingData:
    """Convert Lightning Pose keypoints to Megabouts tracking data."""
    head_x = dataframe.Head.x.to_numpy() * millimeters_per_pixel
    head_y = dataframe.Head.y.to_numpy() * millimeters_per_pixel

    tail_parts = [f"Tail_{index}" for index in range(9)]

    tail_x = dataframe.loc[:, (tail_parts, "x")].to_numpy() * millimeters_per_pixel
    tail_y = dataframe.loc[:, (tail_parts, "y")].to_numpy() * millimeters_per_pixel

    return FullTrackingData.from_keypoints(
        head_x=head_x,
        head_y=head_y,
        tail_x=tail_x,
        tail_y=tail_y,
    )


def megabout_fulltracking_pipeline(
    behavior_data: BehaviorData,
    min_bout_duration_ms: int = 70,
    segmentation_threshold: float = 7.5,
    tail_speed_boxcar_filter_ms: int = 30,
    savgol_window_ms: int = 20,
) -> MegaboutResults:
    """Run the Megabouts full-tracking pipeline."""
    pixels_per_mm = float(behavior_data.metadata["calibration"]["pix_per_mm"])
    millimeters_per_pixel = 1.0 / pixels_per_mm

    frames_per_second = float(behavior_data.metadata["camera"]["framerate_value"])

    timestamps = behavior_data.tracking["timestamp"].to_numpy()

    tracking_config = TrackingConfig(
        fps=frames_per_second,
        tracking="full_tracking",
    )

    pipeline = FullTrackingPipeline(
        tracking_config,
        exclude_CS=True,
    )

    pipeline.segmentation_cfg.min_bout_duration_ms = min_bout_duration_ms
    pipeline.segmentation_cfg.threshold = segmentation_threshold
    pipeline.tail_preprocessing_cfg.tail_speed_boxcar_filter_ms = (
        tail_speed_boxcar_filter_ms
    )
    pipeline.tail_preprocessing_cfg.savgol_window_ms = savgol_window_ms

    tracking_data = full_tracking_data_from_lp(
        behavior_data.full_tracking,
        millimeters_per_pixel,
    )

    (
        ethogram,
        bouts,
        segments,
        tail,
        trajectory,
    ) = pipeline.run(tracking_data)

    return MegaboutResults(
        timestamp=timestamps,
        ethogram=ethogram,
        bouts=bouts,
        segments=segments,
        tail=tail,
        traj=trajectory,
    )


def normalize_rows(
    vectors: np.ndarray,
) -> np.ndarray:
    """Normalize two-dimensional vectors row by row."""
    vectors = np.asarray(vectors, dtype=float)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    return np.divide(
        vectors,
        norms,
        out=np.full_like(vectors, np.nan),
        where=norms > 0,
    )


def calculate_tracking_quality(
    online_centroid: np.ndarray,
    posthoc_centroid: np.ndarray,
    online_heading: np.ndarray,
    posthoc_heading: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, Any]:
    """Calculate online-versus-posthoc tracking mismatch metrics."""
    stop = min(
        stop,
        len(online_centroid),
        len(posthoc_centroid),
        len(online_heading),
        len(posthoc_heading),
    )
    start = max(0, start)

    if stop <= start:
        return {
            "centroid_mismatch_avg": np.nan,
            "centroid_mismatch_max": np.nan,
            "heading_mismatch_avg": np.nan,
            "heading_mismatch_max": np.nan,
            "heading_flip": False,
        }

    centroid_distance = np.linalg.norm(
        online_centroid[start:stop] - posthoc_centroid[start:stop],
        axis=1,
    )

    dot_product = np.sum(
        online_heading[start:stop] * posthoc_heading[start:stop],
        axis=1,
    )
    dot_product = np.clip(
        dot_product,
        -1.0,
        1.0,
    )

    angular_distance = np.rad2deg(np.arccos(dot_product))

    return {
        "centroid_mismatch_avg": float(np.nanmean(centroid_distance)),
        "centroid_mismatch_max": float(np.nanmax(centroid_distance)),
        "heading_mismatch_avg": float(np.nanmean(angular_distance)),
        "heading_mismatch_max": float(np.nanmax(angular_distance)),
        "heading_flip": bool(np.any(angular_distance > 160.0)),
    }


def get_peak_signed_value(
    values: np.ndarray,
) -> float:
    """Return the value with the largest absolute magnitude."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0 or not np.isfinite(values).any():
        return np.nan

    finite_indices = np.flatnonzero(np.isfinite(values))
    local_index = np.argmax(np.abs(values[finite_indices]))

    return float(values[finite_indices[local_index]])


def get_bout_metrics(
    directories: Directories,
    behavior_data: BehaviorData,
    behavior_files: BehaviorFiles,
    megabout: MegaboutResults,
    rollover_time_s: int = 3600,
) -> list[dict[str, Any]]:
    """Create one augmented row per detected swimming bout."""
    recording = RecordingContext(
        directories=directories,
        behavior_files=behavior_files,
        behavior_data=behavior_data,
        rollover_time_s=rollover_time_s,
    )

    frames_per_second = float(behavior_data.metadata["camera"]["framerate_value"])

    bout_onsets = np.asarray(
        megabout.bouts.onset,
        dtype=int,
    )
    bout_offsets = np.asarray(
        megabout.bouts.offset,
        dtype=int,
    )
    bout_categories = np.asarray(
        megabout.bouts.category,
    )
    bout_signs = np.asarray(
        megabout.bouts.sign,
    )
    bout_probabilities = np.asarray(
        megabout.bouts.proba,
    )

    number_of_bouts = min(
        len(bout_onsets),
        len(bout_offsets),
        len(bout_categories),
        len(bout_signs),
        len(bout_probabilities),
    )

    bout_onsets = bout_onsets[:number_of_bouts]
    bout_offsets = bout_offsets[:number_of_bouts]
    bout_categories = bout_categories[:number_of_bouts]
    bout_signs = bout_signs[:number_of_bouts]
    bout_probabilities = bout_probabilities[:number_of_bouts]

    valid_bout_indices = (
        (bout_onsets >= 0)
        & (bout_offsets > bout_onsets)
        & (bout_onsets < len(megabout.timestamp))
        & (bout_offsets < len(megabout.timestamp))
    )

    bout_start_timestamps = np.full(
        number_of_bouts,
        -1,
        dtype=np.int64,
    )
    bout_stop_timestamps = np.full(
        number_of_bouts,
        -1,
        dtype=np.int64,
    )

    bout_start_timestamps[valid_bout_indices] = np.asarray(megabout.timestamp)[
        bout_onsets[valid_bout_indices]
    ].astype(np.int64)
    bout_stop_timestamps[valid_bout_indices] = np.asarray(megabout.timestamp)[
        bout_offsets[valid_bout_indices]
    ].astype(np.int64)

    trial_indices = np.full(
        number_of_bouts,
        -1,
        dtype=int,
    )
    trial_indices[valid_bout_indices] = recording.trial_indices(
        bout_start_timestamps[valid_bout_indices]
    )

    online_centroid = behavior_data.tracking[["centroid_x", "centroid_y"]].to_numpy(
        dtype=float
    )

    posthoc_centroid = behavior_data.full_tracking.Swim_Bladder[["x", "y"]].to_numpy(
        dtype=float
    )

    online_heading = behavior_data.tracking[["pc1_x", "pc1_y"]].to_numpy(dtype=float)

    posthoc_heading_vector = (
        behavior_data.full_tracking.Head[["x", "y"]].to_numpy(dtype=float)
        - posthoc_centroid
    )
    posthoc_heading = normalize_rows(posthoc_heading_vector)

    rows: list[dict[str, Any]] = []

    previous_offset: int | None = None

    for bout_index in range(number_of_bouts):
        if not valid_bout_indices[bout_index]:
            continue

        onset = int(bout_onsets[bout_index])
        offset = int(bout_offsets[bout_index])
        trial_index = int(trial_indices[bout_index])
        last_offset, previous_offset = previous_offset, offset

        if trial_index < 0:
            continue

        event_timestamp = int(bout_start_timestamps[bout_index])

        x_mm = float(megabout.traj.x_smooth[onset])
        y_mm = float(megabout.traj.y_smooth[onset])
        heading = float(megabout.traj.yaw_smooth[onset])

        common_context = recording.event_context(
            event_timestamp=event_timestamp,
            trial_index=trial_index,
            x_mm=x_mm,
            y_mm=y_mm,
            heading=heading,
        )

        heading_change = float(
            megabout.traj.yaw_smooth[offset] - megabout.traj.yaw_smooth[onset]
        )

        delta_x = np.diff(megabout.traj.x_smooth[onset:offset])
        delta_y = np.diff(megabout.traj.y_smooth[onset:offset])
        distance = float(np.sum(np.hypot(delta_x, delta_y)))

        bout_duration = (offset - onset) / frames_per_second

        interbout_duration = np.nan if last_offset is None else (onset - last_offset) / frames_per_second

        peak_axial_speed = get_peak_signed_value(
            megabout.traj.axial_speed[onset:offset]
        )
        peak_yaw_speed = get_peak_signed_value(megabout.traj.yaw_speed[onset:offset])

        start_time = (event_timestamp - int(megabout.timestamp[0])) * 1e-9
        stop_time = (
            int(bout_stop_timestamps[bout_index]) - int(megabout.timestamp[0])
        ) * 1e-9

        quality = calculate_tracking_quality(
            online_centroid=online_centroid,
            posthoc_centroid=posthoc_centroid,
            online_heading=online_heading,
            posthoc_heading=posthoc_heading,
            start=onset if last_offset is None else last_offset,
            stop=offset,
        )

        sign = bout_signs[bout_index]
        trial = recording.trials.iloc[trial_index]
        laterality = bout_laterality(
            epoch_name=trial.epoch_name,
            sign=sign,
        )

        bout_specific = {
            "bout_index": bout_index,
            "frame_start": onset,
            "frame_stop": offset,
            "time_start": float(start_time),
            "time_stop": float(stop_time),
            "heading_change": heading_change,
            "distance": distance,
            "bout_duration": float(bout_duration),
            "interbout_duration": float(interbout_duration),
            "peak_axial_speed": peak_axial_speed,
            "peak_yaw_speed": peak_yaw_speed,
            "category": int(bout_categories[bout_index]),
            "within_trial": bool(bout_stop_timestamps[bout_index] < recording.trial_stops[trial_index]),
            "sign": int(sign),
            "proba": float(bout_probabilities[bout_index]),
            "laterality": laterality,
        }

        rows.append(
            {
                **common_context,
                **bout_specific,
                **quality,
            }
        )

    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Megabouts pipeline and create an augmented " "bout table."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder.",
    )
    parser.add_argument(
        "--bouts-csv",
        default="bouts.csv",
        help=("Output CSV filename relative to root. " "Default: bouts.csv."),
    )
    parser.add_argument(
        "--rollover-time-s",
        type=int,
        default=3600,
    )

    parser.add_argument("--metadata", default="results")
    parser.add_argument("--stimuli", default="results")
    parser.add_argument("--tracking", default="results")
    parser.add_argument(
        "--lightning-pose",
        default="lightning_pose",
    )
    parser.add_argument("--temperature", default="results")
    parser.add_argument("--video", default="results")
    parser.add_argument(
        "--video-timestamp",
        default="results",
    )
    parser.add_argument("--results", default="results")
    parser.add_argument("--plots", default="plots")

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force Megabouts to run without CUDA.",
    )

    return parser


def run_megabouts(
    root: Path,
    output_csv: str,
    metadata: str,
    stimuli: str,
    tracking: str,
    lightning_pose: str,
    temperature: str,
    video: str,
    video_timestamp: str,
    results: str,
    plots: str,
    cpu: bool,
    rollover_time_s: int,
) -> None:
    """Run Megabouts for all experiments and write the bout table."""
    if cpu:
        import torch

        torch.cuda.is_available = lambda: False

    directories = Directories(
        root,
        metadata=metadata,
        stimuli=stimuli,
        tracking=tracking,
        full_tracking=lightning_pose,
        temperature=temperature,
        video=video,
        video_timestamp=video_timestamp,
        results=results,
        plots=plots,
    )

    behavior_files = find_files(directories)
    bout_rows: list[dict[str, Any]] = []

    for behavior_file in tqdm(
        behavior_files,
        desc="Megabouts",
    ):
        behavior_data = load_data(behavior_file)

        if behavior_data.full_tracking.empty:
            print(f"[skip] {behavior_file.metadata.stem}: " "no full tracking")
            continue

        megabout = megabout_fulltracking_pipeline(behavior_data)

        pickle_path = behavior_file.metadata.with_suffix(".pkl")

        with pickle_path.open("wb") as output_file:
            pickle.dump(megabout, output_file)

        fish_rows = get_bout_metrics(
            directories=directories,
            behavior_data=behavior_data,
            behavior_files=behavior_file,
            megabout=megabout,
            rollover_time_s=rollover_time_s,
        )

        bout_rows.extend(fish_rows)

        print(f"[ok] {behavior_file.metadata.stem}: " f"{len(fish_rows):,} bouts")

    bouts = pd.DataFrame.from_records(bout_rows)
    output_path = root / output_csv

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    bouts.to_csv(
        output_path,
        header=True,
        index=False,
        float_format="%.10g",
    )

    print(f"Saved {len(bouts):,} bouts to {output_path}")


def main() -> None:
    """Run the Megabouts command-line application."""
    args = build_parser().parse_args()

    run_megabouts(
        root=args.root,
        output_csv=args.bouts_csv,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        lightning_pose=args.lightning_pose,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
        cpu=args.cpu,
        rollover_time_s=args.rollover_time_s,
    )


if __name__ == "__main__":
    main()
