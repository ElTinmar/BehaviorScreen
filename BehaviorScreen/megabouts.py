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

from BehaviorScreen.event_context import event_context, trial_indices
from BehaviorScreen.load import Directories, file_younger_than, find_files
from BehaviorScreen.protocol import bout_laterality
from BehaviorScreen.recording import Recording


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
    rec: Recording,
    min_bout_duration_ms: int = 70,
    segmentation_threshold: float = 7.5,
    tail_speed_boxcar_filter_ms: int = 30,
    savgol_window_ms: int = 20,
) -> MegaboutResults:
    """Run the Megabouts full-tracking pipeline."""
    tracking_config = TrackingConfig(
        fps=rec.fps,
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

    tracking_data = full_tracking_data_from_lp(rec.pose, 1.0 / rec.pix_per_mm)

    (
        ethogram,
        bouts,
        segments,
        tail,
        trajectory,
    ) = pipeline.run(tracking_data)

    return MegaboutResults(
        timestamp=rec.frame_time_ns[: len(rec.pose)],
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


def get_bout_metrics(rec: Recording, megabout: MegaboutResults) -> list[dict[str, Any]]:
    """One augmented row per detected swimming bout that starts inside a trial."""
    bouts = megabout.bouts
    n_bouts = min(
        len(bouts.onset), len(bouts.offset), len(bouts.category), len(bouts.sign), len(bouts.proba)
    )
    onsets = np.asarray(bouts.onset, dtype=int)[:n_bouts]
    offsets = np.asarray(bouts.offset, dtype=int)[:n_bouts]
    frame_time = np.asarray(megabout.timestamp, dtype=np.int64)

    valid = (onsets >= 0) & (offsets > onsets) & (offsets < len(frame_time))
    start_ts = np.where(valid, frame_time[np.clip(onsets, 0, len(frame_time) - 1)], -1)
    stop_ts = np.where(valid, frame_time[np.clip(offsets, 0, len(frame_time) - 1)], -1)
    trial_index = np.where(valid, trial_indices(rec, start_ts), -1)
    trial_stops = rec.trials["stop_timestamp"].to_numpy(np.int64)

    posthoc_centroid = rec.pose.Swim_Bladder[["x", "y"]].to_numpy(float)
    online = rec.online.iloc[: len(posthoc_centroid)]
    online_centroid = online[["centroid_x", "centroid_y"]].to_numpy(float)
    online_heading = online[["pc1_x", "pc1_y"]].to_numpy(float)
    posthoc_heading = normalize_rows(
        rec.pose.Head[["x", "y"]].to_numpy(float) - posthoc_centroid
    )
    traj = megabout.traj

    rows: list[dict[str, Any]] = []
    previous_offset: int | None = None

    for i in np.flatnonzero(valid):
        onset, offset, trial = int(onsets[i]), int(offsets[i]), int(trial_index[i])
        last_offset, previous_offset = previous_offset, offset

        if trial < 0:
            continue

        sign = int(bouts.sign[i])
        bout = {
            "bout_index": int(i),
            "frame_start": onset,
            "frame_stop": offset,
            "time_start": (start_ts[i] - frame_time[0]) * 1e-9,
            "time_stop": (stop_ts[i] - frame_time[0]) * 1e-9,
            "heading_change": float(traj.yaw_smooth[offset] - traj.yaw_smooth[onset]),
            "distance": float(
                np.sum(np.hypot(np.diff(traj.x_smooth[onset:offset]), np.diff(traj.y_smooth[onset:offset])))
            ),
            "bout_duration": (stop_ts[i] - start_ts[i]) * 1e-9,
            "interbout_duration": np.nan
            if last_offset is None
            else (start_ts[i] - frame_time[last_offset]) * 1e-9,
            "peak_axial_speed": get_peak_signed_value(traj.axial_speed[onset:offset]),
            "peak_yaw_speed": get_peak_signed_value(traj.yaw_speed[onset:offset]),
            "category": int(bouts.category[i]),
            "within_trial": bool(stop_ts[i] < trial_stops[trial]),
            "sign": sign,
            "proba": float(bouts.proba[i]),
            "laterality": bout_laterality(rec.trials.epoch_name.iloc[trial], sign),
        }
        quality = calculate_tracking_quality(
            online_centroid,
            posthoc_centroid,
            online_heading,
            posthoc_heading,
            start=onset if last_offset is None else last_offset,
            stop=offset,
        )
        context = event_context(
            rec,
            int(start_ts[i]),
            trial,
            float(traj.x_smooth[onset]),
            float(traj.y_smooth[onset]),
            float(traj.yaw_smooth[onset]),
        )
        rows.append(context | bout | quality)

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
    overwrite_mtime: float = 0,
) -> None:
    """Run Megabouts for all experiments and write the bout table."""
    if cpu:
        import torch

        torch.cuda.is_available = lambda: False

    output_path = root / output_csv
    if file_younger_than(output_path, overwrite_mtime):
        print(f"{output_path} is younger than {overwrite_mtime}h, skipping...")
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    bout_rows: list[dict[str, Any]] = []

    for behavior_file in tqdm(find_files(directories), desc="Megabouts"):
        rec = Recording(behavior_file)

        if rec.pose.empty:
            print(f"[skip] {rec.name}: no full tracking")
            continue

        megabout = megabout_fulltracking_pipeline(rec)

        with behavior_file.metadata.with_suffix(".pkl").open("wb") as output_file:
            pickle.dump(megabout, output_file)

        fish_rows = get_bout_metrics(rec, megabout)
        bout_rows.extend(fish_rows)
        print(f"[ok] {rec.name}: {len(fish_rows):,} bouts")

    bouts = pd.DataFrame.from_records(bout_rows)
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
    )


if __name__ == "__main__":
    main()
