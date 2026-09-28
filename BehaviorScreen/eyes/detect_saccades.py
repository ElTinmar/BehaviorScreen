#!/usr/bin/env python3
"""
Detect saccades and extract the nine Dowell et al. oculomotor metrics.

This stage deliberately does not perform UMAP or classification.

Outputs
-------
events.csv
    One row per event with physical-unit metrics.

events.npz
    Row-aligned raw and smoothed event snippets.

Example
-------
python -m BehaviorScreen.eyes.detect_saccades \
    /media/martin/DATA_18TB/Screen/WT/vehicle \
    --mode freeswim \
    --output saccades.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.load import (
    BehaviorData,
    Directories,
    find_files,
    load_data,
)
from BehaviorScreen.process import compute_angle_between_vectors

import BehaviorScreen.eyes.saccade_pipeline as sp


SAMPLING_RATE = 500.0


def extract_raw_eye_angles(
    behavior_data: BehaviorData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract timestamps and left/right eye angles from tracking data."""
    left_vector = (
        behavior_data.eyes_tracking.eye_left_back[
            ["x", "y"]
        ].to_numpy()
        - behavior_data.eyes_tracking.eye_left_front[
            ["x", "y"]
        ].to_numpy()
    )

    right_vector = (
        behavior_data.eyes_tracking.eye_right_back[
            ["x", "y"]
        ].to_numpy()
        - behavior_data.eyes_tracking.eye_right_front[
            ["x", "y"]
        ].to_numpy()
    )

    left_angle = np.rad2deg(
        compute_angle_between_vectors(
            left_vector,
            np.array([0, 1]),
        )
    )
    right_angle = np.rad2deg(
        compute_angle_between_vectors(
            right_vector,
            np.array([0, 1]),
        )
    )

    timestamps_ns = (
        behavior_data.video_timestamps.timestamp.to_numpy()
    )

    number_of_samples = min(
        len(timestamps_ns),
        len(left_angle),
        len(right_angle),
    )

    timestamps_ns = timestamps_ns[:number_of_samples]
    start_time_ns = timestamps_ns[0]

    time_seconds = (
        timestamps_ns.astype(np.float64) - start_time_ns
    ) * 1e-9

    return (
        time_seconds,
        left_angle[:number_of_samples],
        right_angle[:number_of_samples],
    )


def extract_event_snippet(
    trace: np.ndarray,
    onset_index: int,
    sampling_rate: float,
    pre_ms: float,
    post_ms: float,
) -> np.ndarray:
    """Extract a fixed event window, padding recording edges with NaN."""
    pre_samples = int(
        round(pre_ms / 1000.0 * sampling_rate)
    )
    post_samples = int(
        round(post_ms / 1000.0 * sampling_rate)
    )

    snippet = np.full(
        pre_samples + post_samples,
        np.nan,
        dtype=np.float32,
    )

    source_start = max(0, onset_index - pre_samples)
    source_stop = min(
        len(trace),
        onset_index + post_samples,
    )

    destination_start = source_start - (
        onset_index - pre_samples
    )
    destination_stop = (
        destination_start
        + source_stop
        - source_start
    )

    snippet[destination_start:destination_stop] = trace[
        source_start:source_stop
    ]

    return snippet


def empty_result(
    snippet_length: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Create an empty per-fish result."""
    empty_traces = np.empty(
        (0, snippet_length),
        dtype=np.float32,
    )

    return (
        pd.DataFrame(),
        {
            "L_raw": empty_traces.copy(),
            "R_raw": empty_traces.copy(),
            "L_smooth": empty_traces.copy(),
            "R_smooth": empty_traces.copy(),
        },
    )


def process_fish(
    fish_label: str,
    time_seconds: np.ndarray,
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    mode: str,
    snippet_pre_ms: float,
    snippet_post_ms: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Detect and measure saccades for one fish."""
    pre_samples = int(
        round(snippet_pre_ms / 1000.0 * SAMPLING_RATE)
    )
    post_samples = int(
        round(snippet_post_ms / 1000.0 * SAMPLING_RATE)
    )
    snippet_length = pre_samples + post_samples

    coarse = sp.coarse_detect_events(
        time_seconds,
        left_raw,
        right_raw,
        fs=100.0,
    )

    event_times, has_left, has_right = (
        sp.pair_binocular_events(
            coarse["L_events"]["t"],
            coarse["R_events"]["t"],
        )
    )

    # Keep the provenance arrays aligned with the refractory filter.
    order = np.argsort(event_times)
    event_times = event_times[order]
    has_left = has_left[order]
    has_right = has_right[order]

    retained = np.zeros(len(event_times), dtype=bool)
    previous_retained_time = -np.inf

    for index, event_time in enumerate(event_times):
        if event_time - previous_retained_time >= 0.3:
            retained[index] = True
            previous_retained_time = event_time

    event_times = event_times[retained]
    has_left = has_left[retained]
    has_right = has_right[retained]

    if len(event_times) == 0:
        return empty_result(snippet_length)

    time_500, left_500 = sp.interp_to_rate(
        time_seconds,
        left_raw,
        fs=SAMPLING_RATE,
    )
    _, right_500 = sp.interp_to_rate(
        time_seconds,
        right_raw,
        fs=SAMPLING_RATE,
    )

    left_smooth = sp.smooth_trace_for_metrics(
        left_500,
        mode=mode,
    )
    right_smooth = sp.smooth_trace_for_metrics(
        right_500,
        mode=mode,
    )

    records: list[dict[str, object]] = []
    trace_lists = {
        "L_raw": [],
        "R_raw": [],
        "L_smooth": [],
        "R_smooth": [],
    }

    for event_number, event_time in enumerate(event_times):
        coarse_index = int(
            np.clip(
                round(event_time * SAMPLING_RATE),
                0,
                len(left_smooth) - 1,
            )
        )

        # Preserve the existing pipeline behavior. This currently refines
        # the shared event onset using the left-eye trace.
        onset_index = sp.refine_onset_time(
            left_smooth,
            SAMPLING_RATE,
            coarse_index,
        )

        left_metrics = sp.event_position_velocity_metrics(
            left_smooth,
            SAMPLING_RATE,
            onset_index,
        )
        right_metrics = sp.event_position_velocity_metrics(
            right_smooth,
            SAMPLING_RATE,
            onset_index,
        )

        feature_values = sp.compute_9_metrics(
            left_metrics,
            right_metrics,
        )

        is_bconv, details = sp.detect_biphasic_convergent(
            left_smooth,
            right_smooth,
            SAMPLING_RATE,
            onset_index,
            is_tethered=(mode == "tethered"),
        )

        bconv_side = None

        if is_bconv:
            left_velocity = details["L"]["max_vel"]
            right_velocity = details["R"]["max_vel"]

            if left_velocity is None or not np.isfinite(
                left_velocity
            ):
                left_velocity = -np.inf

            if right_velocity is None or not np.isfinite(
                right_velocity
            ):
                right_velocity = -np.inf

            bconv_side = (
                "L"
                if left_velocity >= right_velocity
                else "R"
            )

        record: dict[str, object] = {
            "event_id": (
                f"{fish_label}__{onset_index:010d}"
                f"__{event_number:06d}"
            ),
            "fish": fish_label,
            "coarse_time_s": float(event_time),
            "onset_time_s": float(
                time_500[onset_index]
            ),
            "onset_sample_500hz": onset_index,
            "has_left_detection": bool(
                has_left[event_number]
            ),
            "has_right_detection": bool(
                has_right[event_number]
            ),
            "bconv_flag": bool(is_bconv),
            "bconv_side": bconv_side,
        }

        record.update(
            zip(sp.METRIC_NAMES, feature_values)
        )
        records.append(record)

        trace_lists["L_raw"].append(
            extract_event_snippet(
                left_500,
                onset_index,
                SAMPLING_RATE,
                snippet_pre_ms,
                snippet_post_ms,
            )
        )
        trace_lists["R_raw"].append(
            extract_event_snippet(
                right_500,
                onset_index,
                SAMPLING_RATE,
                snippet_pre_ms,
                snippet_post_ms,
            )
        )
        trace_lists["L_smooth"].append(
            extract_event_snippet(
                left_smooth,
                onset_index,
                SAMPLING_RATE,
                snippet_pre_ms,
                snippet_post_ms,
            )
        )
        trace_lists["R_smooth"].append(
            extract_event_snippet(
                right_smooth,
                onset_index,
                SAMPLING_RATE,
                snippet_pre_ms,
                snippet_post_ms,
            )
        )

    events = pd.DataFrame.from_records(records)

    traces = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in trace_lists.items()
    }

    return events, traces


def collect_events(
    directories: Directories,
    mode: str,
    snippet_pre_ms: float,
    snippet_post_ms: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    """Detect events across all experiments."""
    behavior_files = find_files(directories)
    print(f"Found {len(behavior_files)} experiments")

    event_tables: list[pd.DataFrame] = []
    trace_groups = {
        "L_raw": [],
        "R_raw": [],
        "L_smooth": [],
        "R_smooth": [],
    }

    for files in tqdm(behavior_files, desc="Fish"):
        fish_label = files.metadata.stem
        behavior_data = load_data(files)

        if (
            behavior_data.eyes_tracking.empty
            or behavior_data.video_timestamps.empty
        ):
            print(
                f"[skip] {fish_label}: no eye tracking "
                "or timestamps"
            )
            continue

        (
            time_seconds,
            left_raw,
            right_raw,
        ) = extract_raw_eye_angles(behavior_data)

        events, traces = process_fish(
            fish_label=fish_label,
            time_seconds=time_seconds,
            left_raw=left_raw,
            right_raw=right_raw,
            mode=mode,
            snippet_pre_ms=snippet_pre_ms,
            snippet_post_ms=snippet_post_ms,
        )

        if events.empty:
            print(f"[skip] {fish_label}: no events")
            continue

        event_tables.append(events)

        for name in trace_groups:
            trace_groups[name].append(traces[name])

        print(f"[ok] {fish_label}: {len(events):,} events")

    if not event_tables:
        return pd.DataFrame(), None

    events = pd.concat(
        event_tables,
        ignore_index=True,
    )

    pre_samples = int(
        round(snippet_pre_ms / 1000.0 * SAMPLING_RATE)
    )
    post_samples = int(
        round(snippet_post_ms / 1000.0 * SAMPLING_RATE)
    )

    time_axis_ms = (
        np.arange(-pre_samples, post_samples)
        / SAMPLING_RATE
        * 1000.0
    ).astype(np.float32)

    combined_traces = {
        name: np.concatenate(arrays, axis=0)
        for name, arrays in trace_groups.items()
    }
    combined_traces["time_axis_ms"] = time_axis_ms

    return events, combined_traces


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect saccades and extract event metrics. "
            "Clustering is performed separately."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("saccades.csv"),
        help="Output event CSV.",
    )
    parser.add_argument(
        "--mode",
        choices=("tethered", "freeswim"),
        default="freeswim",
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
        "--snippet-pre-ms",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--snippet-post-ms",
        type=float,
        default=600.0,
    )

    return parser


def main() -> None:
    """Run event detection."""
    args = build_parser().parse_args()

    directories = Directories(
        args.root,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        full_tracking=args.lightning_pose,
        eyes_tracking=args.lightning_pose,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
    )

    events, traces = collect_events(
        directories=directories,
        mode=args.mode,
        snippet_pre_ms=args.snippet_pre_ms,
        snippet_post_ms=args.snippet_post_ms,
    )

    if events.empty or traces is None:
        print("No events were detected.")
        return

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_csv(
        args.output,
        index=False,
        float_format="%.10g",
    )

    trace_path = args.output.with_suffix(".npz")
    np.savez_compressed(trace_path, **traces)

    print(
        f"Saved {len(events):,} events from "
        f"{events['fish'].nunique():,} fish to {args.output}"
    )
    print(f"Saved row-aligned traces to {trace_path}")


if __name__ == "__main__":
    main()