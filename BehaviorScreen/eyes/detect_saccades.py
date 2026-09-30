#!/usr/bin/env python3
"""
Detect saccades and extract the nine Dowell et al. oculomotor metrics.

This stage performs:

1. Extraction of left/right eye angles from Lightning Pose landmarks.
2. Combination of front/back landmark likelihoods for each eye.
3. Removal of low-likelihood samples.
4. Short-gap interpolation without bridging long tracking gaps.
5. Coarse saccade detection at 100 Hz.
6. Binocular event pairing within 100 ms.
7. Removal of events within 300 ms of a preceding retained event.
8. Resampling and smoothing at 500 Hz.
9. Refined onset estimation.
10. Extraction of the nine oculomotor metrics.
11. Detection of candidate biphasic convergent events.

This stage deliberately does not perform UMAP or final classification.

Outputs
-------
<output>.csv
    One row per valid event with physical-unit metrics.

<output>.npz
    Row-aligned likelihood-masked/resampled and smoothed event snippets.

Notes
-----
The arrays named L_raw and R_raw in the NPZ file are not the original
camera-rate traces. They are likelihood-masked traces resampled to 500 Hz,
with only short missing-data gaps interpolated. Long low-confidence gaps
remain NaN.

Example
-------
python -m BehaviorScreen.eyes.detect_saccades \
    /media/martin/DATA_18TB/Screen/WT/vehicle \
    --mode freeswim \
    --likelihood-threshold 0.9 \
    --detection-max-gap-ms 40 \
    --metric-max-gap-ms 20 \
    --output saccades.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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

# ============================================================================
# Defaults
# ============================================================================

SAMPLING_RATE = 500.0

MIN_PROMINENCE_FREESWIM = 1.1
MIN_PROMINENCE_TETHERED = 0.6

DEFAULT_LIKELIHOOD_THRESHOLD = 0.9

# Short gaps may be interpolated for coarse event detection.
DEFAULT_DETECTION_MAX_GAP_MS = 40.0

# Use a stricter interpolation limit for position/velocity metrics.
DEFAULT_METRIC_MAX_GAP_MS = 20.0

DEFAULT_SNIPPET_PRE_MS = 300.0
DEFAULT_SNIPPET_POST_MS = 600.0


# ============================================================================
# Lightning Pose data extraction
# ============================================================================


def extract_landmark_likelihood(
    landmark: pd.DataFrame,
    landmark_name: str,
) -> np.ndarray:
    """
    Extract the likelihood/confidence column for one Lightning Pose point.

    Lightning Pose commonly uses ``likelihood``. Alternative names are
    accepted to accommodate different loading conventions.

    Parameters
    ----------
    landmark
        DataFrame containing at least x, y, and a confidence column.
    landmark_name
        Name used in error messages.

    Returns
    -------
    likelihood
        One-dimensional floating-point likelihood array.
    """
    candidate_columns = (
        "likelihood",
        "confidence",
        "score",
        "probability",
    )

    for column in candidate_columns:
        if column in landmark.columns:
            return landmark[column].to_numpy(dtype=float)

    raise KeyError(
        f"No likelihood/confidence column was found for "
        f"{landmark_name}. Available columns are: "
        f"{list(landmark.columns)}"
    )


def extract_raw_eye_angles(
    behavior_data: BehaviorData,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Extract timestamps, eye angles, and eye-level likelihoods.

    Each eye angle is defined by two landmarks. The eye-level likelihood is
    therefore the minimum likelihood of the front and back landmarks.

    Returns
    -------
    time_seconds
        Timestamps relative to the first finite timestamp.
    left_angle
        Left-eye angle in degrees.
    right_angle
        Right-eye angle in degrees.
    left_likelihood
        Minimum of left-front and left-back likelihood.
    right_likelihood
        Minimum of right-front and right-back likelihood.
    """
    left_front = behavior_data.eyes_tracking.eye_left_front
    left_back = behavior_data.eyes_tracking.eye_left_back
    right_front = behavior_data.eyes_tracking.eye_right_front
    right_back = behavior_data.eyes_tracking.eye_right_back

    left_vector = left_back[["x", "y"]].to_numpy(dtype=float) - left_front[
        ["x", "y"]
    ].to_numpy(dtype=float)

    right_vector = right_back[["x", "y"]].to_numpy(dtype=float) - right_front[
        ["x", "y"]
    ].to_numpy(dtype=float)

    reference_vector = np.array([0.0, 1.0], dtype=float)

    left_angle = np.rad2deg(
        compute_angle_between_vectors(
            left_vector,
            reference_vector,
        )
    )

    right_angle = np.rad2deg(
        compute_angle_between_vectors(
            right_vector,
            reference_vector,
        )
    )

    left_front_likelihood = extract_landmark_likelihood(
        left_front,
        "eye_left_front",
    )
    left_back_likelihood = extract_landmark_likelihood(
        left_back,
        "eye_left_back",
    )
    right_front_likelihood = extract_landmark_likelihood(
        right_front,
        "eye_right_front",
    )
    right_back_likelihood = extract_landmark_likelihood(
        right_back,
        "eye_right_back",
    )

    # Both landmarks are required to define an eye vector. The minimum is
    # therefore the most conservative eye-level confidence measure.
    left_likelihood = sp.combine_likelihoods(
        left_front_likelihood,
        left_back_likelihood,
        method="min",
    )

    right_likelihood = sp.combine_likelihoods(
        right_front_likelihood,
        right_back_likelihood,
        method="min",
    )

    timestamps_ns = behavior_data.video_timestamps.timestamp.to_numpy()

    number_of_samples = min(
        len(timestamps_ns),
        len(left_angle),
        len(right_angle),
        len(left_likelihood),
        len(right_likelihood),
    )

    if number_of_samples < 2:
        raise ValueError("Fewer than two aligned eye-tracking samples were found")

    timestamps_ns = np.asarray(
        timestamps_ns[:number_of_samples],
        dtype=np.float64,
    )
    left_angle = np.asarray(
        left_angle[:number_of_samples],
        dtype=float,
    )
    right_angle = np.asarray(
        right_angle[:number_of_samples],
        dtype=float,
    )
    left_likelihood = np.asarray(
        left_likelihood[:number_of_samples],
        dtype=float,
    )
    right_likelihood = np.asarray(
        right_likelihood[:number_of_samples],
        dtype=float,
    )

    finite_timestamp_indices = np.flatnonzero(np.isfinite(timestamps_ns))

    if finite_timestamp_indices.size < 2:
        raise ValueError("Fewer than two finite video timestamps were found")

    start_time_ns = timestamps_ns[finite_timestamp_indices[0]]

    time_seconds = (timestamps_ns - start_time_ns) * 1e-9

    return (
        time_seconds,
        left_angle,
        right_angle,
        left_likelihood,
        right_likelihood,
    )


# ============================================================================
# Event snippets
# ============================================================================


def extract_event_snippet(
    trace: np.ndarray,
    onset_index: int,
    sampling_rate: float,
    pre_ms: float,
    post_ms: float,
) -> np.ndarray:
    """
    Extract a fixed event window.

    Samples outside the recording are padded with NaN. Existing NaNs caused
    by low-confidence tracking remain NaN.
    """
    trace = np.asarray(trace, dtype=float).ravel()

    pre_samples = int(round(pre_ms / 1000.0 * sampling_rate))
    post_samples = int(round(post_ms / 1000.0 * sampling_rate))

    snippet_length = pre_samples + post_samples

    snippet = np.full(
        snippet_length,
        np.nan,
        dtype=np.float32,
    )

    source_start = max(
        0,
        onset_index - pre_samples,
    )
    source_stop = min(
        len(trace),
        onset_index + post_samples,
    )

    destination_start = source_start - (onset_index - pre_samples)
    destination_stop = destination_start + source_stop - source_start

    if source_stop > source_start:
        snippet[destination_start:destination_stop] = trace[source_start:source_stop]

    return snippet


def empty_result(
    snippet_length: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Create an empty result with correctly shaped trace arrays."""
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


# ============================================================================
# Per-fish processing
# ============================================================================


def process_fish(
    fish_label: str,
    time_seconds: np.ndarray,
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    left_likelihood: np.ndarray,
    right_likelihood: np.ndarray,
    mode: str,
    snippet_pre_ms: float,
    snippet_post_ms: float,
    likelihood_threshold: float,
    detection_max_gap_ms: float,
    metric_max_gap_ms: float,
    onset_threshold_fraction: float = 0.5,
    lowess_delta_threshold: float = 0.5,
    lowess_anneal_samples: int = 50,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Detect and measure saccades for one fish.

    Only events with valid metric windows for both eyes are written to the
    output. Consequently, rows in the CSV remain aligned with rows in the
    NPZ trace arrays.
    """
    pre_samples = int(round(snippet_pre_ms / 1000.0 * SAMPLING_RATE))
    post_samples = int(round(snippet_post_ms / 1000.0 * SAMPLING_RATE))
    snippet_length = pre_samples + post_samples

    if mode == "tethered":
        minimum_prominence = MIN_PROMINENCE_TETHERED
    else:
        minimum_prominence = MIN_PROMINENCE_FREESWIM

    detection_config = sp.DetectionConfig(
        fs=100.0,
        lowpass_cutoff_hz=1.0,
        lowpass_order=2,
        step_width_ms=160.0,
        min_prominence=minimum_prominence,
        likelihood_threshold=likelihood_threshold,
        max_interp_gap_s=(detection_max_gap_ms / 1000.0),
        min_valid_block_s=0.5,
        gap_guard_ms=100.0,
        pairing_window_s=0.100,
        refractory_s=0.300,
    )

    metric_config = sp.MetricConfig(
        fs=SAMPLING_RATE,
        likelihood_threshold=likelihood_threshold,
        max_interp_gap_s=(metric_max_gap_ms / 1000.0),
        pre_window_ms=200.0,
        post_window_ms=200.0,
        velocity_window_ms=150.0,
        onset_search_window_ms=400.0,
        onset_wide_step_ms=100.0,
        onset_narrow_step_ms=40.0,
        # The precise threshold was not reported in the methods.
        onset_threshold_fraction=(onset_threshold_fraction),
        # Reject an event if a retained long gap intersects any metric
        # window.
        require_complete_metric_windows=True,
        minimum_valid_fraction=0.95,
        # These parameters were not fully reported for the classification
        # pipeline and should be validated on representative recordings.
        lowess_delta_threshold=(lowess_delta_threshold),
        lowess_anneal_samples=(lowess_anneal_samples),
        lowess_conv_window_samples=None,
        lowess_sigma_samples=None,
        lowess_min_block_s=0.25,
    )

    result = sp.process_trial(
        time=time_seconds,
        left_position=left_raw,
        right_position=right_raw,
        left_likelihood=left_likelihood,
        right_likelihood=right_likelihood,
        mode=mode,
        detection_config=detection_config,
        metric_config=metric_config,
    )

    if len(result["retained_events"]) == 0:
        return empty_result(snippet_length)

    time_500 = result["timebase_500"]

    # These traces are likelihood-masked and resampled to 500 Hz.
    # Short gaps have been interpolated, whereas long gaps remain NaN.
    left_500 = result["left_position_500"]
    right_500 = result["right_position_500"]

    left_smooth = result["left_smoothed_500"]
    right_smooth = result["right_smoothed_500"]

    records: list[dict[str, object]] = []

    trace_lists: dict[str, list[np.ndarray]] = {
        "L_raw": [],
        "R_raw": [],
        "L_smooth": [],
        "R_smooth": [],
    }

    for event_number, event_record in enumerate(result["event_records"]):
        # Events that intersect long missing-data periods or recording
        # boundaries are not suitable for metric extraction.
        if not event_record["valid"]:
            continue

        event = event_record["event"]

        left_onset_index = event_record["left_onset_index"]
        right_onset_index = event_record["right_onset_index"]

        if left_onset_index is None or right_onset_index is None:
            continue

        feature_values = event_record["features"]

        if feature_values is None:
            continue

        feature_values = np.asarray(
            feature_values,
            dtype=float,
        )

        if feature_values.shape != (len(sp.METRIC_NAMES),) or not np.all(
            np.isfinite(feature_values)
        ):
            continue

        # The eye-specific onset indices are used for their respective
        # metric calculations. A shared reference index is used only to
        # align the saved binocular snippets and run the candidate BConv
        # detector.
        onset_index = int(round(0.5 * (int(left_onset_index) + int(right_onset_index))))

        onset_index = int(
            np.clip(
                onset_index,
                0,
                len(time_500) - 1,
            )
        )

        (
            is_bconv,
            bconv_side,
            bconv_details,
        ) = sp.detect_biphasic_convergent(
            left_smooth,
            right_smooth,
            fs=SAMPLING_RATE,
            onset_index=onset_index,
            is_tethered=(mode == "tethered"),
        )

        record: dict[str, object] = {
            "event_id": (
                f"{fish_label}" f"__{onset_index:010d}" f"__{event_number:06d}"
            ),
            "fish": fish_label,
            "mode": mode,
            # Coarse event times
            "coarse_time_s": float(event.reference_time),
            "left_coarse_time_s": (
                float(event.left_time) if event.has_left else np.nan
            ),
            "right_coarse_time_s": (
                float(event.right_time) if event.has_right else np.nan
            ),
            # Refined event times
            "onset_time_s": float(time_500[onset_index]),
            "left_onset_time_s": float(time_500[left_onset_index]),
            "right_onset_time_s": float(time_500[right_onset_index]),
            # Refined sample indices
            "onset_sample_500hz": onset_index,
            "left_onset_sample_500hz": int(left_onset_index),
            "right_onset_sample_500hz": int(right_onset_index),
            # Coarse detection provenance
            "has_left_detection": bool(event.has_left),
            "has_right_detection": bool(event.has_right),
            # Likelihood-filter configuration
            "likelihood_threshold": float(likelihood_threshold),
            "detection_max_gap_ms": float(detection_max_gap_ms),
            "metric_max_gap_ms": float(metric_max_gap_ms),
            # BConv here is a candidate flag. Final reassignment should
            # happen only after clustering identifies the Conv cluster.
            "bconv_candidate": bool(is_bconv),
            "bconv_flag": bool(is_bconv),
            "bconv_side": bconv_side,
        }

        record.update(
            {
                "bconv_left_candidate": bool(
                    bconv_details.get(
                        "left_candidate",
                        False,
                    )
                ),
                "bconv_right_candidate": bool(
                    bconv_details.get(
                        "right_candidate",
                        False,
                    )
                ),
                "bconv_left_min_velocity": float(
                    bconv_details.get(
                        "left_minimum_velocity",
                        np.nan,
                    )
                ),
                "bconv_right_max_velocity": float(
                    bconv_details.get(
                        "right_maximum_velocity",
                        np.nan,
                    )
                ),
                "bconv_left_baseline_std": float(
                    bconv_details.get(
                        "left_baseline_std",
                        np.nan,
                    )
                ),
                "bconv_right_baseline_std": float(
                    bconv_details.get(
                        "right_baseline_std",
                        np.nan,
                    )
                ),
            }
        )

        record.update(
            {
                metric_name: float(metric_value)
                for metric_name, metric_value in zip(
                    sp.METRIC_NAMES,
                    feature_values,
                )
            }
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

    if not records:
        return empty_result(snippet_length)

    events = pd.DataFrame.from_records(records)

    traces = {
        name: np.stack(values).astype(
            np.float32,
            copy=False,
        )
        for name, values in trace_lists.items()
    }

    if any(len(array) != len(events) for array in traces.values()):
        raise RuntimeError("Event table and trace arrays are not row-aligned")

    return events, traces


# ============================================================================
# Multi-fish collection
# ============================================================================


def collect_events(
    directories: Directories,
    mode: str,
    snippet_pre_ms: float,
    snippet_post_ms: float,
    likelihood_threshold: float,
    detection_max_gap_ms: float,
    metric_max_gap_ms: float,
    onset_threshold_fraction: float,
    lowess_delta_threshold: float,
    lowess_anneal_samples: int,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray] | None,
]:
    """Detect events across all available experiments."""
    behavior_files = find_files(directories)
    print(f"Found {len(behavior_files)} experiments")

    event_tables: list[pd.DataFrame] = []

    trace_groups: dict[str, list[np.ndarray]] = {
        "L_raw": [],
        "R_raw": [],
        "L_smooth": [],
        "R_smooth": [],
    }

    for files in tqdm(
        behavior_files,
        desc="Fish",
    ):
        fish_label = files.metadata.stem

        try:
            behavior_data = load_data(files)
        except Exception as error:
            print(f"[skip] {fish_label}: failed to load data: " f"{error}")
            continue

        if behavior_data.eyes_tracking.empty:
            print(f"[skip] {fish_label}: no eye tracking")
            continue

        if behavior_data.video_timestamps.empty:
            print(f"[skip] {fish_label}: no video timestamps")
            continue

        try:
            (
                time_seconds,
                left_raw,
                right_raw,
                left_likelihood,
                right_likelihood,
            ) = extract_raw_eye_angles(behavior_data)
        except Exception as error:
            print(f"[skip] {fish_label}: failed to extract " f"eye data: {error}")
            continue

        left_low_confidence = (
            ~np.isfinite(left_raw)
            | ~np.isfinite(left_likelihood)
            | (left_likelihood < likelihood_threshold)
        )

        right_low_confidence = (
            ~np.isfinite(right_raw)
            | ~np.isfinite(right_likelihood)
            | (right_likelihood < likelihood_threshold)
        )

        print(
            f"[tracking] {fish_label}: "
            f"L invalid={np.mean(left_low_confidence):.1%}, "
            f"R invalid={np.mean(right_low_confidence):.1%}"
        )

        try:
            events, traces = process_fish(
                fish_label=fish_label,
                time_seconds=time_seconds,
                left_raw=left_raw,
                right_raw=right_raw,
                left_likelihood=left_likelihood,
                right_likelihood=right_likelihood,
                mode=mode,
                snippet_pre_ms=snippet_pre_ms,
                snippet_post_ms=snippet_post_ms,
                likelihood_threshold=(likelihood_threshold),
                detection_max_gap_ms=(detection_max_gap_ms),
                metric_max_gap_ms=(metric_max_gap_ms),
                onset_threshold_fraction=(onset_threshold_fraction),
                lowess_delta_threshold=(lowess_delta_threshold),
                lowess_anneal_samples=(lowess_anneal_samples),
            )
        except Exception as error:
            print(f"[skip] {fish_label}: saccade processing " f"failed: {error}")
            continue

        if events.empty:
            print(f"[skip] {fish_label}: no valid events")
            continue

        event_tables.append(events)

        for name in trace_groups:
            trace_groups[name].append(traces[name])

        print(f"[ok] {fish_label}: " f"{len(events):,} valid events")

    if not event_tables:
        return pd.DataFrame(), None

    events = pd.concat(
        event_tables,
        ignore_index=True,
    )

    pre_samples = int(round(snippet_pre_ms / 1000.0 * SAMPLING_RATE))
    post_samples = int(round(snippet_post_ms / 1000.0 * SAMPLING_RATE))

    time_axis_ms = (
        np.arange(
            -pre_samples,
            post_samples,
        )
        / SAMPLING_RATE
        * 1000.0
    ).astype(np.float32)

    combined_traces = {
        name: np.concatenate(
            arrays,
            axis=0,
        )
        for name, arrays in trace_groups.items()
    }

    combined_traces["time_axis_ms"] = time_axis_ms

    number_of_events = len(events)

    for name in (
        "L_raw",
        "R_raw",
        "L_smooth",
        "R_smooth",
    ):
        if len(combined_traces[name]) != number_of_events:
            raise RuntimeError(
                f"{name} contains "
                f"{len(combined_traces[name])} rows, but "
                f"the event table contains "
                f"{number_of_events} rows"
            )

    return events, combined_traces


# ============================================================================
# Command-line interface
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect saccades, filter eye tracking using "
            "Lightning Pose likelihood, and extract event "
            "metrics. Clustering is performed separately."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="saccades.csv",
        help=(
            "Output event CSV, relative to the root directory "
            "unless an absolute path is supplied."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=("tethered", "freeswim"),
        default="freeswim",
        help="Recording mode. Default: %(default)s",
    )

    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=DEFAULT_LIKELIHOOD_THRESHOLD,
        help=(
            "Lightning Pose samples below this likelihood "
            "are treated as missing. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--detection-max-gap-ms",
        type=float,
        default=DEFAULT_DETECTION_MAX_GAP_MS,
        help=(
            "Maximum internal low-confidence gap interpolated "
            "during coarse detection. Default: %(default)s ms"
        ),
    )

    parser.add_argument(
        "--metric-max-gap-ms",
        type=float,
        default=DEFAULT_METRIC_MAX_GAP_MS,
        help=(
            "Maximum internal low-confidence gap interpolated "
            "during metric extraction. Default: %(default)s ms"
        ),
    )

    parser.add_argument(
        "--snippet-pre-ms",
        type=float,
        default=DEFAULT_SNIPPET_PRE_MS,
        help=(
            "Snippet duration before the shared event onset. " "Default: %(default)s ms"
        ),
    )

    parser.add_argument(
        "--snippet-post-ms",
        type=float,
        default=DEFAULT_SNIPPET_POST_MS,
        help=(
            "Snippet duration after the shared event onset. " "Default: %(default)s ms"
        ),
    )

    parser.add_argument(
        "--onset-threshold-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of the local onset-product maximum used "
            "for refined onset thresholding. This value was "
            "not reported in the methods. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--lowess-delta-threshold",
        type=float,
        default=0.5,
        help=(
            "Step-response threshold for custom LOWESS. This "
            "value was not fully reported. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--lowess-anneal-samples",
        type=int,
        default=50,
        help=(
            "Annealing distance for custom LOWESS, in 500 Hz "
            "samples. Default: %(default)s"
        ),
    )

    # BehaviorScreen directory names.
    parser.add_argument(
        "--metadata",
        default="results",
    )
    parser.add_argument(
        "--stimuli",
        default="results",
    )
    parser.add_argument(
        "--tracking",
        default="results",
    )
    parser.add_argument(
        "--lightning-pose",
        default="lightning_pose",
    )
    parser.add_argument(
        "--temperature",
        default="results",
    )
    parser.add_argument(
        "--video",
        default="results",
    )
    parser.add_argument(
        "--video-timestamp",
        default="results",
    )
    parser.add_argument(
        "--results",
        default="results",
    )
    parser.add_argument(
        "--plots",
        default="plots",
    )

    return parser


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate command-line parameter ranges."""
    if not 0.0 <= args.likelihood_threshold <= 1.0:
        raise ValueError("--likelihood-threshold must be between 0 and 1")

    if args.detection_max_gap_ms < 0:
        raise ValueError("--detection-max-gap-ms must be non-negative")

    if args.metric_max_gap_ms < 0:
        raise ValueError("--metric-max-gap-ms must be non-negative")

    if args.snippet_pre_ms < 0:
        raise ValueError("--snippet-pre-ms must be non-negative")

    if args.snippet_post_ms <= 0:
        raise ValueError("--snippet-post-ms must be positive")

    if not 0.0 < args.onset_threshold_fraction <= 1.0:
        raise ValueError("--onset-threshold-fraction must be in (0, 1]")

    if args.lowess_delta_threshold < 0:
        raise ValueError("--lowess-delta-threshold must be non-negative")

    if args.lowess_anneal_samples < 1:
        raise ValueError("--lowess-anneal-samples must be at least 1")

    if args.metric_max_gap_ms > args.detection_max_gap_ms:
        print(
            "[warning] --metric-max-gap-ms is greater than "
            "--detection-max-gap-ms. Metric interpolation is "
            "normally set equal to or stricter than detection "
            "interpolation."
        )


def resolve_output_path(
    root: Path,
    output_argument: str,
) -> Path:
    """
    Resolve the output CSV path.

    Relative output paths are interpreted relative to the experiment root.
    """
    output = Path(output_argument)

    if not output.is_absolute():
        output = root / output

    if output.suffix.lower() != ".csv":
        output = output.with_suffix(".csv")

    return output


def main() -> None:
    """Run likelihood-aware saccade detection."""
    args = build_parser().parse_args()
    validate_arguments(args)

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

    print("Saccade-detection configuration")
    print(f"  mode: {args.mode}")
    print("  likelihood threshold: " f"{args.likelihood_threshold}")
    print("  detection interpolation limit: " f"{args.detection_max_gap_ms} ms")
    print("  metric interpolation limit: " f"{args.metric_max_gap_ms} ms")
    print("  onset threshold fraction: " f"{args.onset_threshold_fraction}")
    print("  LOWESS delta threshold: " f"{args.lowess_delta_threshold}")
    print("  LOWESS annealing distance: " f"{args.lowess_anneal_samples} samples")

    events, traces = collect_events(
        directories=directories,
        mode=args.mode,
        snippet_pre_ms=args.snippet_pre_ms,
        snippet_post_ms=args.snippet_post_ms,
        likelihood_threshold=(args.likelihood_threshold),
        detection_max_gap_ms=(args.detection_max_gap_ms),
        metric_max_gap_ms=(args.metric_max_gap_ms),
        onset_threshold_fraction=(args.onset_threshold_fraction),
        lowess_delta_threshold=(args.lowess_delta_threshold),
        lowess_anneal_samples=(args.lowess_anneal_samples),
    )

    if events.empty or traces is None:
        print("No valid events were detected.")
        return

    output_path = resolve_output_path(
        args.root,
        args.output,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_csv(
        output_path,
        index=False,
        float_format="%.10g",
    )

    trace_path = output_path.with_suffix(".npz")

    np.savez_compressed(
        trace_path,
        **traces,
    )

    print()
    print(
        f"Saved {len(events):,} valid events from "
        f"{events['fish'].nunique():,} fish to "
        f"{output_path}"
    )
    print(f"Saved row-aligned event traces to " f"{trace_path}")

    print()
    print("Event summary")

    if "has_left_detection" in events:
        print("  left coarse detections: " f"{events['has_left_detection'].sum():,}")

    if "has_right_detection" in events:
        print("  right coarse detections: " f"{events['has_right_detection'].sum():,}")

    binocular = events["has_left_detection"] & events["has_right_detection"]

    print("  binocular coarse detections: " f"{binocular.sum():,}")

    print("  BConv candidates: " f"{events['bconv_candidate'].sum():,}")

    print()
    print(
        "Reminder: BConv flags are candidates. Final BConv "
        "reassignment should be applied only to events initially "
        "assigned to the convergent cluster."
    )


if __name__ == "__main__":
    main()
