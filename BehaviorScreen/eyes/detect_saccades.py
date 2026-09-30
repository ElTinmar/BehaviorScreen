"""
detect_saccades.py

Saccade detection and classification adapted from:
    doi: 10.1016/j.cub.2024.08.008

This implementation supports Lightning Pose likelihood masking.

Pipeline
--------
1. Mask low-likelihood eye-position samples.
2. Interpolate only short internal tracking gaps.
3. Resample to 100 Hz.
4. Low-pass filter finite blocks independently.
5. Detect rapid eye movements using a 160 ms step kernel.
6. Pair left/right events within 100 ms.
7. Discard events within 300 ms of a preceding retained event.
8. Resample to 500 Hz and apply custom LOWESS smoothing.
9. Refine left/right onset estimates.
10. Calculate nine oculomotor metrics.
11. Winsorize/z-score by fish.
12. UMAP + DBSCAN classification.
13. Optionally reassign biphasic convergent events.

Important
---------
Several parameters were not reported in the paper and were not present in
the recovered MATLAB code, including the exact refined-onset threshold and
the parameters passed to speciallowess4 during classification. They remain
configurable below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    umap = None


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DetectionConfig:
    """Configuration for coarse 100 Hz event detection."""

    fs: float = 100.0
    lowpass_cutoff_hz: float = 1.0
    lowpass_order: int = 2
    step_width_ms: float = 160.0

    # MATLAB uses 1.1 when gmb.fs exists and 0.6 otherwise.
    min_prominence: float = 1.1

    likelihood_threshold: float = 0.9

    # Short low-confidence gaps may be interpolated.
    max_interp_gap_s: float = 0.04

    # Finite blocks shorter than this are not filtered/detected.
    min_valid_block_s: float = 0.50

    # Peaks this close to a long missing-data boundary are rejected.
    gap_guard_ms: float = 100.0

    pairing_window_s: float = 0.100
    refractory_s: float = 0.300


@dataclass
class MetricConfig:
    """Configuration for 500 Hz smoothing and metric extraction."""

    fs: float = 500.0
    likelihood_threshold: float = 0.9

    # Use a stricter interpolation limit for velocity estimation.
    max_interp_gap_s: float = 0.020

    pre_window_ms: float = 200.0
    post_window_ms: float = 200.0
    velocity_window_ms: float = 150.0

    onset_search_window_ms: float = 400.0
    onset_wide_step_ms: float = 100.0
    onset_narrow_step_ms: float = 40.0

    # Not stated in the publication.
    onset_threshold_fraction: float = 0.5

    # Reject metrics when a retained long gap enters any required window.
    require_complete_metric_windows: bool = True
    minimum_valid_fraction: float = 0.95

    # Parameters for speciallowess4. The spans are selected by mode below.
    # These values were not all reported for the classification pipeline.
    lowess_delta_threshold: float = 0.5
    lowess_anneal_samples: int = 50
    lowess_conv_window_samples: Optional[int] = None
    lowess_sigma_samples: Optional[float] = None
    lowess_min_block_s: float = 0.25


@dataclass
class ClusteringConfig:
    """Configuration for UMAP and DBSCAN."""

    n_neighbors: int = 199
    min_dist: float = 0.11
    n_components: int = 2
    metric: str = "euclidean"
    random_state: Optional[int] = 0

    dbscan_eps: float = 0.34

    # MATLAB dbscanCD uses:
    #     sum(distance < epsilon) > 570
    # and includes the point itself. The closest sklearn value is 571.
    dbscan_min_samples: int = 571

    border_assignment_radius: float = 3.0

    # These were not reported in the paper.
    border_density_bins: int = 10
    border_density_radius: float = 0.15

    heldout_neighbors: int = 100
    heldout_max_median_distance: float = 0.3


@dataclass
class PairedEvent:
    """
    A paired or unpaired rapid-eye-movement event.

    Left and right detection times are retained separately because the
    MATLAB data structures appear to retain eye-specific times.
    """

    reference_time: float
    left_time: float = np.nan
    right_time: float = np.nan
    left_index_100: Optional[int] = None
    right_index_100: Optional[int] = None

    @property
    def has_left(self) -> bool:
        return np.isfinite(self.left_time)

    @property
    def has_right(self) -> bool:
        return np.isfinite(self.right_time)


METRIC_NAMES = [
    "Amp_L",
    "Amp_R",
    "MaxMedAmp_L",
    "MaxMedAmp_R",
    "Vel_cw_L",
    "Vel_ccw_L",
    "Vel_cw_R",
    "Vel_ccw_R",
    "Vergence",
]


# ============================================================================
# Generic array utilities
# ============================================================================


def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous True blocks.

    Returns
    -------
    runs
        List of ``(start, stop)`` pairs with Python-exclusive stops.
    """
    mask = np.asarray(mask, dtype=bool).ravel()

    if mask.size == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))

    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)

    return list(zip(starts, stops))


def find_blocks(
    binary_vector: np.ndarray,
    order: str = "size",
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python equivalent of the recovered MATLAB ``findblocks.m``.

    Parameters
    ----------
    binary_vector
        One-dimensional Boolean or 0/1 array.
    order
        ``"size"`` reproduces the default MATLAB behavior: descending
        block size. ``"run"`` returns chronological order.
    """
    x = np.asarray(binary_vector).ravel()

    if np.any(~np.isin(x, [0, 1, False, True])):
        raise ValueError("find_blocks accepts only Boolean/0/1 data")

    runs = contiguous_true_runs(x.astype(bool))

    if not runs:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    starts = np.asarray([run[0] for run in runs], dtype=int)
    sizes = np.asarray([run[1] - run[0] for run in runs], dtype=int)

    if min_size is not None:
        keep = sizes >= int(min_size)
        starts = starts[keep]
        sizes = sizes[keep]

    if max_size is not None:
        keep = sizes <= int(max_size)
        starts = starts[keep]
        sizes = sizes[keep]

    if order == "size":
        sort_order = np.argsort(-sizes, kind="stable")
    elif order == "run":
        sort_order = np.argsort(starts, kind="stable")
    else:
        raise ValueError("order must be 'size' or 'run'")

    return starts[sort_order], sizes[sort_order]


def step_kernel(width_samples: int, normalized: bool = True) -> np.ndarray:
    """
    Construct the step kernel used for coarse event detection.

    For an even width this is:

        [-1, ..., -1, +1, ..., +1]

    MATLAB's coarse-detection kernel is divided by its total width.
    """
    width_samples = int(round(width_samples))

    if width_samples < 2:
        raise ValueError("Step-kernel width must be at least two samples")

    negative_n = width_samples // 2
    positive_n = width_samples - negative_n

    kernel = np.concatenate(
        (
            -np.ones(negative_n, dtype=float),
            np.ones(positive_n, dtype=float),
        )
    )

    if normalized:
        kernel /= float(width_samples)

    return kernel


def speciallowess_step_kernel(width_samples: int) -> np.ndarray:
    """
    Reproduce ``cn`` from speciallowess4.m:

        [ones(floor(width/2));
         -ones(ceil(width/2))]
    """
    width_samples = max(2, int(round(width_samples)))

    positive_n = int(np.floor(width_samples / 2))
    negative_n = int(np.ceil(width_samples / 2))

    return np.concatenate(
        (
            np.ones(positive_n),
            -np.ones(negative_n),
        )
    )


def gaussian_membership(
    x: np.ndarray,
    sigma: float,
    center: float,
) -> np.ndarray:
    """Equivalent to MATLAB ``gaussmf(x, [sigma center])``."""
    x = np.asarray(x, dtype=float)

    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")

    return np.exp(-((x - center) ** 2) / (2.0 * sigma**2))


def nearest_index(timebase: np.ndarray, target_time: float) -> int:
    """Return the index of the sample nearest to target_time."""
    timebase = np.asarray(timebase, dtype=float)

    if timebase.size == 0:
        raise ValueError("timebase is empty")

    return int(np.nanargmin(np.abs(timebase - target_time)))


# ============================================================================
# Likelihood masking and short-gap interpolation
# ============================================================================


def combine_likelihoods(
    *likelihoods: np.ndarray,
    method: str = "min",
) -> np.ndarray:
    """
    Combine multiple Lightning Pose likelihoods for one derived signal.

    For an eye angle calculated from multiple landmarks, ``method="min"``
    is conservative and marks the angle low-confidence if any required
    landmark is low-confidence.
    """
    if not likelihoods:
        raise ValueError("At least one likelihood array is required")

    arrays = [np.asarray(x, dtype=float) for x in likelihoods]

    if any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("All likelihood arrays must have the same shape")

    stacked = np.stack(arrays, axis=0)

    if method == "min":
        return np.nanmin(stacked, axis=0)
    if method == "mean":
        return np.nanmean(stacked, axis=0)
    if method == "product":
        return np.nanprod(stacked, axis=0)

    raise ValueError("method must be 'min', 'mean', or 'product'")


def apply_likelihood_mask(
    position: np.ndarray,
    likelihood: Optional[np.ndarray],
    threshold: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace low-confidence position samples with NaN.

    If likelihood is None, every finite position sample is considered valid.
    """
    position = np.asarray(position, dtype=float).ravel()

    if likelihood is None:
        likelihood = np.ones(position.shape, dtype=float)
    else:
        likelihood = np.asarray(likelihood, dtype=float).ravel()

    if position.shape != likelihood.shape:
        raise ValueError("position and likelihood must have the same shape")

    valid = np.isfinite(position) & np.isfinite(likelihood) & (likelihood >= threshold)

    masked = position.copy()
    masked[~valid] = np.nan

    return masked, valid


def interpolate_short_gaps_irregular(
    time: np.ndarray,
    values: np.ndarray,
    max_gap_s: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate short internal NaN gaps in irregularly sampled data.

    Long gaps and gaps touching the start/end of the recording remain NaN.

    Returns
    -------
    filled
        Values after short-gap interpolation.
    interpolated
        Boolean mask identifying samples filled by interpolation.
    """
    time = np.asarray(time, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel().copy()

    if time.shape != values.shape:
        raise ValueError("time and values must have the same shape")

    interpolated = np.zeros(values.shape, dtype=bool)

    if max_gap_s is None or max_gap_s <= 0:
        return values, interpolated

    finite = np.isfinite(time) & np.isfinite(values)

    for start, stop in contiguous_true_runs(~finite):
        left = start - 1
        right = stop

        # Only interpolate internal gaps.
        if left < 0 or right >= len(values):
            continue

        if not finite[left] or not finite[right]:
            continue

        gap_duration = time[right] - time[left]

        if not np.isfinite(gap_duration) or gap_duration <= 0:
            continue

        if gap_duration <= max_gap_s:
            values[start:stop] = np.interp(
                time[start:stop],
                [time[left], time[right]],
                [values[left], values[right]],
            )
            interpolated[start:stop] = True
            finite[start:stop] = True

    return values, interpolated


def _sort_and_deduplicate_samples(
    time: np.ndarray,
    values: np.ndarray,
    likelihood: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort samples by time and remove duplicate timestamps."""
    finite_time = np.isfinite(time)

    time = time[finite_time]
    values = values[finite_time]
    likelihood = likelihood[finite_time]

    if len(time) < 2:
        raise ValueError("At least two finite timestamps are required")

    order = np.argsort(time, kind="stable")
    time = time[order]
    values = values[order]
    likelihood = likelihood[order]

    # Retain the last occurrence of each duplicate timestamp.
    _, reverse_indices = np.unique(time[::-1], return_index=True)
    unique_indices = len(time) - 1 - reverse_indices
    unique_indices.sort()

    return (
        time[unique_indices],
        values[unique_indices],
        likelihood[unique_indices],
    )


def make_regular_timebase(
    start_time: float,
    end_time: float,
    fs: float,
) -> np.ndarray:
    """Construct a regular time base with spacing exactly 1/fs."""
    if fs <= 0:
        raise ValueError("fs must be positive")

    if end_time < start_time:
        raise ValueError("end_time must not precede start_time")

    count = int(np.floor((end_time - start_time) * fs)) + 1
    return start_time + np.arange(count, dtype=float) / fs


def resample_likelihood_masked(
    time: np.ndarray,
    position: np.ndarray,
    likelihood: Optional[np.ndarray],
    fs: float,
    likelihood_threshold: float = 0.9,
    max_interp_gap_s: Optional[float] = 0.04,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Mask low-confidence samples and resample without bridging long gaps.

    Interpolation onto the regular time base is performed separately within
    each contiguous valid source block.
    """
    time = np.asarray(time, dtype=float).ravel()
    position = np.asarray(position, dtype=float).ravel()

    if likelihood is None:
        likelihood = np.ones(position.shape, dtype=float)
    else:
        likelihood = np.asarray(likelihood, dtype=float).ravel()

    if not (len(time) == len(position) == len(likelihood)):
        raise ValueError("time, position, and likelihood must have equal lengths")

    time, position, likelihood = _sort_and_deduplicate_samples(
        time,
        position,
        likelihood,
    )

    masked, original_valid = apply_likelihood_mask(
        position,
        likelihood,
        threshold=likelihood_threshold,
    )

    filled, interpolated_source = interpolate_short_gaps_irregular(
        time,
        masked,
        max_gap_s=max_interp_gap_s,
    )

    if start_time is None:
        start_time = float(time[0])

    if end_time is None:
        end_time = float(time[-1])

    regular_time = make_regular_timebase(
        start_time=start_time,
        end_time=end_time,
        fs=fs,
    )

    regular_position = np.full(regular_time.shape, np.nan, dtype=float)

    # True means that the regular sample falls inside a source block that
    # remained valid after short-gap interpolation.
    finite_source = np.isfinite(filled)

    for start, stop in contiguous_true_runs(finite_source):
        block_time = time[start:stop]
        block_values = filled[start:stop]

        if len(block_time) == 1:
            index = nearest_index(regular_time, block_time[0])
            if abs(regular_time[index] - block_time[0]) <= 0.5 / fs:
                regular_position[index] = block_values[0]
            continue

        target = (regular_time >= block_time[0]) & (regular_time <= block_time[-1])

        if np.any(target):
            regular_position[target] = np.interp(
                regular_time[target],
                block_time,
                block_values,
            )

    return {
        "time": regular_time,
        "position": regular_position,
        "valid": np.isfinite(regular_position),
        "source_time": time,
        "source_position_masked": masked,
        "source_position_filled": filled,
        "source_original_valid": original_valid,
        "source_interpolated": interpolated_source,
    }


# ============================================================================
# NaN-safe filtering and convolution
# ============================================================================


def lowpass_filter_nan(
    values: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 2,
    endpoint_padding: int = 50,
    min_block_s: float = 0.5,
) -> np.ndarray:
    """
    Low-pass filter contiguous finite blocks independently.

    Notes
    -----
    MATLAB ``lowpass`` is not necessarily identical to this Butterworth
    implementation. This is a stable Python approximation that preserves
    long missing-data gaps.
    """
    values = np.asarray(values, dtype=float).ravel()
    output = np.full(values.shape, np.nan, dtype=float)

    sos = signal.butter(
        order,
        cutoff_hz,
        btype="low",
        fs=fs,
        output="sos",
    )

    min_samples = max(2, int(round(min_block_s * fs)))

    for start, stop in contiguous_true_runs(np.isfinite(values)):
        block = values[start:stop]

        if len(block) < min_samples:
            continue

        pad = min(int(endpoint_padding), max(1, len(block)))

        padded = np.concatenate(
            (
                np.full(pad, block[0]),
                block,
                np.full(pad, block[-1]),
            )
        )

        try:
            filtered = signal.sosfiltfilt(
                sos,
                padded,
                padtype=None,
            )
        except ValueError:
            continue

        output[start:stop] = filtered[pad : pad + len(block)]

    return output


def convolve_finite_blocks(
    values: np.ndarray,
    kernel: np.ndarray,
    endpoint_padding: int = 50,
    minimum_block_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Convolve contiguous finite blocks independently.

    This prevents long missing-data gaps from generating artificial step
    responses.
    """
    values = np.asarray(values, dtype=float).ravel()
    kernel = np.asarray(kernel, dtype=float).ravel()

    output = np.full(values.shape, np.nan, dtype=float)

    if minimum_block_samples is None:
        minimum_block_samples = len(kernel)

    for start, stop in contiguous_true_runs(np.isfinite(values)):
        block = values[start:stop]

        if len(block) < minimum_block_samples:
            continue

        pad = min(int(endpoint_padding), max(1, len(block)))

        padded = np.concatenate(
            (
                np.full(pad, block[0]),
                block,
                np.full(pad, block[-1]),
            )
        )

        convolved = np.convolve(
            padded,
            kernel,
            mode="same",
        )

        output[start:stop] = convolved[pad : pad + len(block)]

    return output


# ============================================================================
# Coarse rapid-eye-movement detection at 100 Hz
# ============================================================================


def _find_peaks_in_finite_blocks(
    filtered: np.ndarray,
    timebase: np.ndarray,
    prominence: float,
    guard_samples: int,
) -> Dict[str, np.ndarray]:
    """Find peaks independently within finite convolution blocks."""
    locations: List[np.ndarray] = []
    prominences: List[np.ndarray] = []

    for start, stop in contiguous_true_runs(np.isfinite(filtered)):
        block = filtered[start:stop]

        if len(block) <= 2 * guard_samples + 2:
            continue

        local_locations, properties = signal.find_peaks(
            np.abs(block),
            prominence=prominence,
        )

        if len(local_locations) == 0:
            continue

        safe = (local_locations >= guard_samples) & (
            local_locations < len(block) - guard_samples
        )

        local_locations = local_locations[safe]
        local_prominences = properties["prominences"][safe]

        if len(local_locations):
            locations.append(local_locations + start)
            prominences.append(local_prominences)

    if not locations:
        return {
            "loc": np.array([], dtype=int),
            "t": np.array([], dtype=float),
            "sign": np.array([], dtype=float),
            "peak": np.array([], dtype=float),
            "prominence": np.array([], dtype=float),
        }

    locations_array = np.concatenate(locations)
    prominence_array = np.concatenate(prominences)

    order = np.argsort(locations_array)
    locations_array = locations_array[order]
    prominence_array = prominence_array[order]

    return {
        "loc": locations_array,
        "t": timebase[locations_array],
        "sign": np.sign(filtered[locations_array]),
        "peak": np.abs(filtered[locations_array]),
        "prominence": prominence_array,
    }


def coarse_detect_events(
    time: np.ndarray,
    left_position: np.ndarray,
    right_position: np.ndarray,
    left_likelihood: Optional[np.ndarray] = None,
    right_likelihood: Optional[np.ndarray] = None,
    config: Optional[DetectionConfig] = None,
) -> Dict[str, Any]:
    """
    Detect coarse left- and right-eye rapid movement events.

    Low-likelihood samples become NaN. Short gaps may be interpolated;
    longer gaps remain NaN throughout filtering and convolution.
    """
    if config is None:
        config = DetectionConfig()

    time = np.asarray(time, dtype=float).ravel()

    finite_time = time[np.isfinite(time)]
    if finite_time.size < 2:
        raise ValueError("At least two finite timestamps are required")

    start_time = float(np.min(finite_time))
    end_time = float(np.max(finite_time))

    left = resample_likelihood_masked(
        time,
        left_position,
        left_likelihood,
        fs=config.fs,
        likelihood_threshold=config.likelihood_threshold,
        max_interp_gap_s=config.max_interp_gap_s,
        start_time=start_time,
        end_time=end_time,
    )

    right = resample_likelihood_masked(
        time,
        right_position,
        right_likelihood,
        fs=config.fs,
        likelihood_threshold=config.likelihood_threshold,
        max_interp_gap_s=config.max_interp_gap_s,
        start_time=start_time,
        end_time=end_time,
    )

    timebase = left["time"]

    if len(right["time"]) != len(timebase) or not np.allclose(right["time"], timebase):
        raise RuntimeError("Left and right regular time bases differ")

    left_pass = lowpass_filter_nan(
        left["position"],
        cutoff_hz=config.lowpass_cutoff_hz,
        fs=config.fs,
        order=config.lowpass_order,
        endpoint_padding=50,
        min_block_s=config.min_valid_block_s,
    )

    right_pass = lowpass_filter_nan(
        right["position"],
        cutoff_hz=config.lowpass_cutoff_hz,
        fs=config.fs,
        order=config.lowpass_order,
        endpoint_padding=50,
        min_block_s=config.min_valid_block_s,
    )

    step_samples = int(round(config.step_width_ms / 1000.0 * config.fs))
    kernel = step_kernel(step_samples, normalized=True)

    left_convolution = convolve_finite_blocks(
        left_pass,
        kernel,
        endpoint_padding=50,
        minimum_block_samples=step_samples,
    )

    right_convolution = convolve_finite_blocks(
        right_pass,
        kernel,
        endpoint_padding=50,
        minimum_block_samples=step_samples,
    )

    guard_samples = max(
        step_samples // 2,
        int(round(config.gap_guard_ms / 1000.0 * config.fs)),
    )

    left_events = _find_peaks_in_finite_blocks(
        left_convolution,
        timebase,
        prominence=config.min_prominence,
        guard_samples=guard_samples,
    )

    right_events = _find_peaks_in_finite_blocks(
        right_convolution,
        timebase,
        prominence=config.min_prominence,
        guard_samples=guard_samples,
    )

    return {
        "timebase": timebase,
        "left_position_100": left["position"],
        "right_position_100": right["position"],
        "left_valid_100": left["valid"],
        "right_valid_100": right["valid"],
        "left_lowpass": left_pass,
        "right_lowpass": right_pass,
        "left_convolution": left_convolution,
        "right_convolution": right_convolution,
        "left_events": left_events,
        "right_events": right_events,
        "left_resampling": left,
        "right_resampling": right,
        "config": config,
    }


# ============================================================================
# Binocular pairing and refractory filtering
# ============================================================================


def pair_binocular_events(
    left_times: np.ndarray,
    right_times: np.ndarray,
    left_indices: Optional[np.ndarray] = None,
    right_indices: Optional[np.ndarray] = None,
    pair_window_s: float = 0.100,
) -> List[PairedEvent]:
    """
    Pair left/right detections using greedy nearest-neighbor matching.

    Unpaired detections are retained. Eye-specific times are preserved.
    The reference time is the mean for paired detections and the available
    eye time for unpaired detections.
    """
    left_times = np.asarray(left_times, dtype=float)
    right_times = np.asarray(right_times, dtype=float)

    if left_indices is None:
        left_indices = np.arange(len(left_times))
    else:
        left_indices = np.asarray(left_indices, dtype=int)

    if right_indices is None:
        right_indices = np.arange(len(right_times))
    else:
        right_indices = np.asarray(right_indices, dtype=int)

    left_order = np.argsort(left_times)
    right_order = np.argsort(right_times)

    left_times = left_times[left_order]
    right_times = right_times[right_order]
    left_indices = left_indices[left_order]
    right_indices = right_indices[right_order]

    used_right = np.zeros(len(right_times), dtype=bool)
    events: List[PairedEvent] = []

    for left_time, left_index in zip(left_times, left_indices):
        available = np.flatnonzero(~used_right)

        if available.size:
            distances = np.abs(right_times[available] - left_time)
            nearest_local = int(np.argmin(distances))
            right_array_index = int(available[nearest_local])

            if distances[nearest_local] <= pair_window_s:
                right_time = float(right_times[right_array_index])
                right_index = int(right_indices[right_array_index])
                used_right[right_array_index] = True

                events.append(
                    PairedEvent(
                        reference_time=0.5 * (left_time + right_time),
                        left_time=float(left_time),
                        right_time=right_time,
                        left_index_100=int(left_index),
                        right_index_100=right_index,
                    )
                )
                continue

        events.append(
            PairedEvent(
                reference_time=float(left_time),
                left_time=float(left_time),
                left_index_100=int(left_index),
            )
        )

    for right_array_index in np.flatnonzero(~used_right):
        events.append(
            PairedEvent(
                reference_time=float(right_times[right_array_index]),
                right_time=float(right_times[right_array_index]),
                right_index_100=int(right_indices[right_array_index]),
            )
        )

    events.sort(key=lambda event: event.reference_time)
    return events


def discard_overlapping_events(
    events: Sequence[PairedEvent],
    refractory_s: float = 0.300,
) -> List[PairedEvent]:
    """
    Retain events separated from the preceding retained event by at least
    refractory_s.
    """
    sorted_events = sorted(events, key=lambda event: event.reference_time)

    retained: List[PairedEvent] = []
    last_retained_time = -np.inf

    for event in sorted_events:
        if event.reference_time - last_retained_time >= refractory_s:
            retained.append(event)
            last_retained_time = event.reference_time

    return retained


# ============================================================================
# LOWESS and speciallowess4
# ============================================================================


def _local_linear_fit_at_zero(
    offsets: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted local-linear prediction at offset zero."""
    valid = (
        np.isfinite(offsets)
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    offsets = offsets[valid]
    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return np.nan

    if values.size == 1:
        return float(values[0])

    s0 = np.sum(weights)
    s1 = np.sum(weights * offsets)
    s2 = np.sum(weights * offsets * offsets)

    t0 = np.sum(weights * values)
    t1 = np.sum(weights * offsets * values)

    denominator = s0 * s2 - s1 * s1

    if denominator <= np.finfo(float).eps:
        return float(t0 / s0)

    intercept = (t0 * s2 - t1 * s1) / denominator
    return float(intercept)


def regular_lowess(values: np.ndarray, span_samples: float) -> np.ndarray:
    """
    Fast local-linear LOWESS for uniformly sampled finite data.

    The interior estimate uses the fact that local-linear LOWESS on a
    symmetric regular grid reduces to a normalized tricube convolution.
    Boundary samples are calculated explicitly with local-linear fits.

    This is closer to MATLAB ``smooth(..., 'lowess')`` than a generic
    moving average, although exact version-specific agreement is not
    guaranteed.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)

    if n == 0:
        return values.copy()

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "regular_lowess expects a finite block; use "
            "speciallowess4_nan for data containing NaNs"
        )

    span = int(round(span_samples))
    span = max(1, min(span, n))

    # Use an odd window for a symmetric center.
    if span % 2 == 0:
        if span < n:
            span += 1
        else:
            span -= 1

    if span <= 1:
        return values.copy()

    half = span // 2
    offsets = np.arange(-half, half + 1, dtype=float)

    scale = max(float(half), 1.0)
    distance = np.abs(offsets) / scale
    weights = np.clip(1.0 - distance**3, 0.0, None) ** 3

    if np.sum(weights) == 0:
        return values.copy()

    normalized_weights = weights / np.sum(weights)

    # convolve1d performs correlation. The kernel is symmetric here.
    output = convolve1d(
        values,
        normalized_weights,
        mode="nearest",
    )

    # Explicit local-linear boundary estimates.
    for index in list(range(half)) + list(range(max(half, n - half), n)):
        start = max(0, min(index - half, n - span))
        stop = min(n, start + span)

        local_indices = np.arange(start, stop)
        local_offsets = local_indices.astype(float) - float(index)

        max_distance = np.max(np.abs(local_offsets))
        if max_distance == 0:
            output[index] = values[index]
            continue

        local_distance = np.abs(local_offsets) / max_distance
        local_weights = (
            np.clip(
                1.0 - local_distance**3,
                0.0,
                None,
            )
            ** 3
        )

        output[index] = _local_linear_fit_at_zero(
            local_offsets,
            values[start:stop],
            local_weights,
        )

    return output


def speciallowess4(
    data: np.ndarray,
    wide_window: float,
    narrow_window: float,
    delta_threshold: float,
    anneal_window: int,
    convolution_window: Optional[float] = None,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Port of the recovered MATLAB speciallowess4.m for a finite data block.

    The apparent MATLAB behavior

        thisoff = set(i,1)

    is retained: blending is centered around each detected block onset,
    rather than spanning the complete threshold-exceedance block.
    """
    data = np.asarray(data, dtype=float).ravel()

    if data.size == 0:
        return data.copy()

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "speciallowess4 expects a finite block; use "
            "speciallowess4_nan for traces containing NaNs"
        )

    if convolution_window is None:
        convolution_window = wide_window

    if sigma is None:
        sigma = anneal_window / 4.0

    pad = 50
    padded = np.concatenate(
        (
            np.full(pad, data[0]),
            data,
            np.full(pad, data[-1]),
        )
    )

    wide = regular_lowess(padded, wide_window)
    output = wide[pad : pad + len(data)].copy()

    if narrow_window == 0:
        narrow = data.copy()
    else:
        narrow = regular_lowess(data, narrow_window)

    kernel = speciallowess_step_kernel(int(round(convolution_window)))

    step_response = np.convolve(
        narrow,
        kernel,
        mode="same",
    )

    thresholded = np.abs(step_response) > delta_threshold

    # Recovered MATLAB findblocks defaults to descending block size.
    onsets, block_sizes = find_blocks(
        thresholded,
        order="size",
    )

    if len(onsets) == 0:
        return output

    search_distance = int(round(anneal_window))

    for onset, _block_size in zip(onsets, block_sizes):
        # Reproduce the recovered MATLAB assignment:
        # thison = set(i,1); thisoff = set(i,1)
        center = int(onset)

        start = max(0, center - search_distance)
        stop = min(len(output) - 1, center + search_distance)

        indices = np.arange(start, stop + 1)
        gaussian = gaussian_membership(
            indices,
            sigma=float(sigma),
            center=float(center),
        )

        left_indices = np.arange(start, center + 1)
        left_weights = gaussian[: len(left_indices)]

        output[left_indices] = narrow[left_indices] * left_weights + output[
            left_indices
        ] * (1.0 - left_weights)

        right_indices = np.arange(center + 1, stop + 1)

        if right_indices.size:
            # This follows the MATLAB fliplr weighting expression.
            right_weights = gaussian[1 : 1 + len(right_indices)][::-1]

            output[right_indices] = narrow[right_indices] * right_weights + output[
                right_indices
            ] * (1.0 - right_weights)

    return output


def speciallowess4_nan(
    data: np.ndarray,
    wide_window: float,
    narrow_window: float,
    delta_threshold: float,
    anneal_window: int,
    convolution_window: Optional[float] = None,
    sigma: Optional[float] = None,
    minimum_block_samples: Optional[int] = None,
) -> np.ndarray:
    """Apply speciallowess4 independently to contiguous finite blocks."""
    data = np.asarray(data, dtype=float).ravel()
    output = np.full(data.shape, np.nan, dtype=float)

    if minimum_block_samples is None:
        minimum_block_samples = max(
            int(np.ceil(wide_window)) + 2,
            3,
        )

    for start, stop in contiguous_true_runs(np.isfinite(data)):
        block = data[start:stop]

        if len(block) < minimum_block_samples:
            continue

        output[start:stop] = speciallowess4(
            block,
            wide_window=wide_window,
            narrow_window=narrow_window,
            delta_threshold=delta_threshold,
            anneal_window=anneal_window,
            convolution_window=convolution_window,
            sigma=sigma,
        )

    return output


def smooth_trace_for_metrics(
    position_500: np.ndarray,
    mode: str,
    config: Optional[MetricConfig] = None,
) -> np.ndarray:
    """
    Apply the paper's LOWESS span choices to a 500 Hz trace.

    Tethered
    --------
    Wide span: 33 ms
    During stepwise changes: no smoothing

    Free-swimming
    -------------
    Wide span: 133 ms
    During stepwise changes: 80 ms
    """
    if config is None:
        config = MetricConfig()

    if mode == "tethered":
        wide_ms = 33.0
        narrow_ms = 0.0
    elif mode in {"free", "freeswim", "free-swimming"}:
        wide_ms = 133.0
        narrow_ms = 80.0
    else:
        raise ValueError(
            "mode must be 'tethered', 'free', 'freeswim', " "or 'free-swimming'"
        )

    wide_samples = wide_ms / 1000.0 * config.fs
    narrow_samples = 0.0 if narrow_ms == 0 else narrow_ms / 1000.0 * config.fs

    convolution_window = config.lowess_conv_window_samples
    if convolution_window is None:
        convolution_window = wide_samples

    minimum_block_samples = max(
        int(round(config.lowess_min_block_s * config.fs)),
        int(np.ceil(wide_samples)) + 2,
    )

    return speciallowess4_nan(
        position_500,
        wide_window=wide_samples,
        narrow_window=narrow_samples,
        delta_threshold=config.lowess_delta_threshold,
        anneal_window=config.lowess_anneal_samples,
        convolution_window=convolution_window,
        sigma=config.lowess_sigma_samples,
        minimum_block_samples=minimum_block_samples,
    )


def prepare_metric_trace(
    time: np.ndarray,
    position: np.ndarray,
    likelihood: Optional[np.ndarray],
    mode: str,
    config: Optional[MetricConfig] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Mask, resample to 500 Hz, and smooth one eye-position trace."""
    if config is None:
        config = MetricConfig()

    resampled = resample_likelihood_masked(
        time,
        position,
        likelihood,
        fs=config.fs,
        likelihood_threshold=config.likelihood_threshold,
        max_interp_gap_s=config.max_interp_gap_s,
        start_time=start_time,
        end_time=end_time,
    )

    smoothed = smooth_trace_for_metrics(
        resampled["position"],
        mode=mode,
        config=config,
    )

    return {
        **resampled,
        "smoothed": smoothed,
        "smoothed_valid": np.isfinite(smoothed),
    }


# ============================================================================
# Refined onset estimation
# ============================================================================


def _valid_same_convolution(
    values: np.ndarray,
    finite: np.ndarray,
    kernel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convolve values and identify samples whose full kernel support is valid.
    """
    filled = np.where(finite, values, 0.0)

    response = np.convolve(
        filled,
        kernel,
        mode="same",
    )

    support = np.convolve(
        finite.astype(float),
        np.ones(len(kernel), dtype=float),
        mode="same",
    )

    valid_response = support >= len(kernel)
    response[~valid_response] = np.nan

    return response, valid_response


def refine_onset_time(
    smoothed_position: np.ndarray,
    fs: float,
    coarse_index: int,
    window_ms: float = 400.0,
    wide_step_ms: float = 100.0,
    narrow_step_ms: float = 40.0,
    threshold_fraction: float = 0.5,
) -> Optional[int]:
    """
    Refine an onset using the product of 100 and 40 ms step convolutions.

    The exact threshold and threshold-crossing rule were not recovered.
    This implementation chooses the above-threshold sample nearest to the
    coarse estimate.
    """
    position = np.asarray(smoothed_position, dtype=float).ravel()
    n = len(position)

    if coarse_index < 0 or coarse_index >= n:
        return None

    half_window = int(round(window_ms / 1000.0 * fs / 2.0))

    start = max(0, coarse_index - half_window)
    stop = min(n, coarse_index + half_window + 1)

    segment = position[start:stop]
    finite = np.isfinite(segment)

    wide_samples = int(round(wide_step_ms / 1000.0 * fs))
    narrow_samples = int(round(narrow_step_ms / 1000.0 * fs))

    wide_kernel = step_kernel(wide_samples, normalized=True)
    narrow_kernel = step_kernel(narrow_samples, normalized=True)

    wide_response, wide_valid = _valid_same_convolution(
        segment,
        finite,
        wide_kernel,
    )

    narrow_response, narrow_valid = _valid_same_convolution(
        segment,
        finite,
        narrow_kernel,
    )

    product = wide_response * narrow_response
    product[~(wide_valid & narrow_valid)] = np.nan

    strength = np.abs(product)

    if not np.any(np.isfinite(strength)):
        return None

    maximum = np.nanmax(strength)

    if not np.isfinite(maximum) or maximum <= 0:
        return None

    candidates = np.flatnonzero(
        np.isfinite(strength) & (strength >= threshold_fraction * maximum)
    )

    if candidates.size == 0:
        return None

    local_coarse_index = coarse_index - start

    selected = candidates[np.argmin(np.abs(candidates - local_coarse_index))]

    return int(start + selected)


def refine_event_onsets(
    event: PairedEvent,
    timebase_500: np.ndarray,
    left_smoothed: np.ndarray,
    right_smoothed: np.ndarray,
    config: Optional[MetricConfig] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Refine left and right onset estimates for a paired event.

    If one eye did not generate a coarse detection, the binocular reference
    time is used as that eye's initial estimate.
    """
    if config is None:
        config = MetricConfig()

    left_coarse_time = event.left_time if event.has_left else event.reference_time
    right_coarse_time = event.right_time if event.has_right else event.reference_time

    left_coarse_index = nearest_index(timebase_500, left_coarse_time)
    right_coarse_index = nearest_index(timebase_500, right_coarse_time)

    left_onset = refine_onset_time(
        left_smoothed,
        fs=config.fs,
        coarse_index=left_coarse_index,
        window_ms=config.onset_search_window_ms,
        wide_step_ms=config.onset_wide_step_ms,
        narrow_step_ms=config.onset_narrow_step_ms,
        threshold_fraction=config.onset_threshold_fraction,
    )

    right_onset = refine_onset_time(
        right_smoothed,
        fs=config.fs,
        coarse_index=right_coarse_index,
        window_ms=config.onset_search_window_ms,
        wide_step_ms=config.onset_wide_step_ms,
        narrow_step_ms=config.onset_narrow_step_ms,
        threshold_fraction=config.onset_threshold_fraction,
    )

    return left_onset, right_onset


# ============================================================================
# Event metrics
# ============================================================================


def empty_eye_metrics() -> Dict[str, Any]:
    """Return an invalid per-eye metric record."""
    return {
        "pre_pos": np.nan,
        "max_post_pos": np.nan,
        "max_post_idx": None,
        "median_post_pos": np.nan,
        "vel_cw": np.nan,
        "vel_ccw": np.nan,
        "valid": False,
        "invalid_reason": None,
    }


def _window_is_valid(
    values: np.ndarray,
    require_complete: bool,
    minimum_valid_fraction: float,
) -> bool:
    if values.size == 0:
        return False

    finite_fraction = np.mean(np.isfinite(values))

    if require_complete:
        return finite_fraction == 1.0

    return finite_fraction >= minimum_valid_fraction


def event_position_velocity_metrics(
    position: np.ndarray,
    fs: float,
    onset_index: Optional[int],
    pre_window_ms: float = 200.0,
    post_window_ms: float = 200.0,
    velocity_window_ms: float = 150.0,
    require_complete_windows: bool = True,
    minimum_valid_fraction: float = 0.95,
) -> Dict[str, Any]:
    """
    Calculate the position and velocity metrics for one eye.

    Events are rejected if retained long tracking gaps overlap required
    windows, unless partial windows are explicitly allowed.
    """
    output = empty_eye_metrics()

    position = np.asarray(position, dtype=float).ravel()
    n = len(position)

    if onset_index is None:
        output["invalid_reason"] = "missing_onset"
        return output

    if onset_index < 0 or onset_index >= n:
        output["invalid_reason"] = "onset_out_of_bounds"
        return output

    pre_samples = int(round(pre_window_ms / 1000.0 * fs))
    post_samples = int(round(post_window_ms / 1000.0 * fs))
    velocity_samples = int(round(velocity_window_ms / 1000.0 * fs))

    velocity_left = velocity_samples // 2
    velocity_right = velocity_samples - velocity_left

    pre_start = onset_index - pre_samples
    first_post_stop = onset_index + post_samples
    velocity_start = onset_index - velocity_left
    velocity_stop = onset_index + velocity_right

    if pre_start < 0 or first_post_stop > n or velocity_start < 0 or velocity_stop > n:
        output["invalid_reason"] = "recording_boundary"
        return output

    pre_segment = position[pre_start:onset_index]
    first_post_segment = position[onset_index:first_post_stop]
    velocity_segment = position[velocity_start:velocity_stop]

    for name, segment in (
        ("pre_window", pre_segment),
        ("first_post_window", first_post_segment),
        ("velocity_window", velocity_segment),
    ):
        if not _window_is_valid(
            segment,
            require_complete_windows,
            minimum_valid_fraction,
        ):
            output["invalid_reason"] = f"invalid_{name}"
            return output

    if not np.isfinite(position[onset_index]):
        output["invalid_reason"] = "invalid_onset_sample"
        return output

    pre_position = float(np.nanmedian(pre_segment))

    deviation = np.abs(first_post_segment - position[onset_index])

    if not np.any(np.isfinite(deviation)):
        output["invalid_reason"] = "invalid_post_deviation"
        return output

    local_maximum_index = int(np.nanargmax(deviation))
    maximum_post_index = onset_index + local_maximum_index
    maximum_post_position = float(position[maximum_post_index])

    median_post_stop = maximum_post_index + post_samples

    if median_post_stop > n:
        output["invalid_reason"] = "median_post_boundary"
        return output

    median_post_segment = position[maximum_post_index:median_post_stop]

    if not _window_is_valid(
        median_post_segment,
        require_complete_windows,
        minimum_valid_fraction,
    ):
        output["invalid_reason"] = "invalid_median_post_window"
        return output

    median_post_position = float(np.nanmedian(median_post_segment))

    # Interpolate any residual NaNs only when partial windows were allowed.
    if not np.all(np.isfinite(velocity_segment)):
        finite = np.isfinite(velocity_segment)

        if np.sum(finite) < 2:
            output["invalid_reason"] = "insufficient_velocity_samples"
            return output

        velocity_input = np.interp(
            np.arange(len(velocity_segment)),
            np.flatnonzero(finite),
            velocity_segment[finite],
        )
    else:
        velocity_input = velocity_segment

    velocity = np.gradient(
        velocity_input,
        1.0 / fs,
    )

    output.update(
        {
            "pre_pos": pre_position,
            "max_post_pos": maximum_post_position,
            "max_post_idx": maximum_post_index,
            "median_post_pos": median_post_position,
            "vel_cw": float(np.nanmax(velocity)),
            "vel_ccw": float(np.nanmin(velocity)),
            "valid": True,
            "invalid_reason": None,
        }
    )

    return output


def compute_9_metrics(
    left_metrics: Dict[str, Any],
    right_metrics: Dict[str, Any],
) -> np.ndarray:
    """Combine valid left/right eye metrics into the nine feature values."""
    if not left_metrics.get("valid", False):
        raise ValueError("left_metrics are invalid")

    if not right_metrics.get("valid", False):
        raise ValueError("right_metrics are invalid")

    amplitude_left = left_metrics["median_post_pos"] - left_metrics["pre_pos"]
    amplitude_right = right_metrics["median_post_pos"] - right_metrics["pre_pos"]

    maximum_median_left = left_metrics["max_post_pos"] - left_metrics["median_post_pos"]
    maximum_median_right = (
        right_metrics["max_post_pos"] - right_metrics["median_post_pos"]
    )

    vergence = left_metrics["median_post_pos"] - right_metrics["median_post_pos"]

    return np.asarray(
        [
            amplitude_left,
            amplitude_right,
            maximum_median_left,
            maximum_median_right,
            left_metrics["vel_cw"],
            left_metrics["vel_ccw"],
            right_metrics["vel_cw"],
            right_metrics["vel_ccw"],
            vergence,
        ],
        dtype=float,
    )


# ============================================================================
# Complete single-trial processing
# ============================================================================


def process_trial(
    time: np.ndarray,
    left_position: np.ndarray,
    right_position: np.ndarray,
    left_likelihood: Optional[np.ndarray] = None,
    right_likelihood: Optional[np.ndarray] = None,
    mode: str = "tethered",
    detection_config: Optional[DetectionConfig] = None,
    metric_config: Optional[MetricConfig] = None,
) -> Dict[str, Any]:
    """
    Run detection and metric extraction for a single trial.

    Returns
    -------
    result
        Dictionary containing traces, events, all event records, and an
        ``N_valid_events x 9`` feature matrix.
    """
    if detection_config is None:
        detection_config = DetectionConfig()

    if metric_config is None:
        metric_config = MetricConfig()

    coarse = coarse_detect_events(
        time,
        left_position,
        right_position,
        left_likelihood=left_likelihood,
        right_likelihood=right_likelihood,
        config=detection_config,
    )

    paired_events = pair_binocular_events(
        coarse["left_events"]["t"],
        coarse["right_events"]["t"],
        left_indices=coarse["left_events"]["loc"],
        right_indices=coarse["right_events"]["loc"],
        pair_window_s=detection_config.pairing_window_s,
    )

    retained_events = discard_overlapping_events(
        paired_events,
        refractory_s=detection_config.refractory_s,
    )

    finite_time = np.asarray(time, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]

    start_time = float(np.min(finite_time))
    end_time = float(np.max(finite_time))

    left_metric_trace = prepare_metric_trace(
        time,
        left_position,
        left_likelihood,
        mode=mode,
        config=metric_config,
        start_time=start_time,
        end_time=end_time,
    )

    right_metric_trace = prepare_metric_trace(
        time,
        right_position,
        right_likelihood,
        mode=mode,
        config=metric_config,
        start_time=start_time,
        end_time=end_time,
    )

    timebase_500 = left_metric_trace["time"]

    if len(right_metric_trace["time"]) != len(timebase_500) or not np.allclose(
        right_metric_trace["time"],
        timebase_500,
    ):
        raise RuntimeError("Left and right 500 Hz time bases differ")

    event_records: List[Dict[str, Any]] = []
    valid_features: List[np.ndarray] = []
    valid_event_indices: List[int] = []

    for event_index, event in enumerate(retained_events):
        left_onset, right_onset = refine_event_onsets(
            event,
            timebase_500,
            left_metric_trace["smoothed"],
            right_metric_trace["smoothed"],
            config=metric_config,
        )

        left_metrics = event_position_velocity_metrics(
            left_metric_trace["smoothed"],
            fs=metric_config.fs,
            onset_index=left_onset,
            pre_window_ms=metric_config.pre_window_ms,
            post_window_ms=metric_config.post_window_ms,
            velocity_window_ms=metric_config.velocity_window_ms,
            require_complete_windows=(metric_config.require_complete_metric_windows),
            minimum_valid_fraction=(metric_config.minimum_valid_fraction),
        )

        right_metrics = event_position_velocity_metrics(
            right_metric_trace["smoothed"],
            fs=metric_config.fs,
            onset_index=right_onset,
            pre_window_ms=metric_config.pre_window_ms,
            post_window_ms=metric_config.post_window_ms,
            velocity_window_ms=metric_config.velocity_window_ms,
            require_complete_windows=(metric_config.require_complete_metric_windows),
            minimum_valid_fraction=(metric_config.minimum_valid_fraction),
        )

        valid = left_metrics["valid"] and right_metrics["valid"]

        features = None

        if valid:
            features = compute_9_metrics(
                left_metrics,
                right_metrics,
            )
            valid_features.append(features)
            valid_event_indices.append(event_index)

        event_records.append(
            {
                "event": event,
                "left_onset_index": left_onset,
                "right_onset_index": right_onset,
                "left_onset_time": (
                    np.nan if left_onset is None else timebase_500[left_onset]
                ),
                "right_onset_time": (
                    np.nan if right_onset is None else timebase_500[right_onset]
                ),
                "left_metrics": left_metrics,
                "right_metrics": right_metrics,
                "features": features,
                "valid": valid,
            }
        )

    if valid_features:
        feature_matrix = np.vstack(valid_features)
    else:
        feature_matrix = np.empty((0, len(METRIC_NAMES)))

    return {
        "coarse": coarse,
        "paired_events": paired_events,
        "retained_events": retained_events,
        "timebase_500": timebase_500,
        "left_position_500": left_metric_trace["position"],
        "right_position_500": right_metric_trace["position"],
        "left_smoothed_500": left_metric_trace["smoothed"],
        "right_smoothed_500": right_metric_trace["smoothed"],
        "left_metric_trace": left_metric_trace,
        "right_metric_trace": right_metric_trace,
        "event_records": event_records,
        "valid_event_indices": np.asarray(
            valid_event_indices,
            dtype=int,
        ),
        "features": feature_matrix,
        "metric_names": METRIC_NAMES.copy(),
    }


# ============================================================================
# Winsorization and standardization
# ============================================================================


def winsorize(
    values: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Winsorize each feature column."""
    values = np.asarray(values, dtype=float)

    lower, upper = np.nanpercentile(
        values,
        [lower_percentile, upper_percentile],
        axis=0,
    )

    return np.clip(values, lower, upper)


def winsorize_zscore_per_fish(
    features: np.ndarray,
    fish_ids: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    ddof: int = 1,
) -> np.ndarray:
    """
    Winsorize and z-score each animal independently.

    ``ddof=1`` is closest to MATLAB's usual sample-standard-deviation
    convention.
    """
    features = np.asarray(features, dtype=float)
    fish_ids = np.asarray(fish_ids)

    if len(features) != len(fish_ids):
        raise ValueError("features and fish_ids must have equal lengths")

    output = np.full(features.shape, np.nan, dtype=float)

    for fish_id in np.unique(fish_ids):
        selected = fish_ids == fish_id
        subset = features[selected]

        subset = winsorize(
            subset,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

        mean = np.nanmean(subset, axis=0)
        std = np.nanstd(subset, axis=0, ddof=ddof)

        std[~np.isfinite(std) | (std == 0)] = 1.0

        output[selected] = (subset - mean) / std

    return output


# ============================================================================
# UMAP and DBSCAN
# ============================================================================


def fit_umap_embedding(
    standardized_features: np.ndarray,
    config: Optional[ClusteringConfig] = None,
):
    """Fit a two-dimensional UMAP model."""
    if config is None:
        config = ClusteringConfig()

    if umap is None:
        raise ImportError(
            "UMAP is not installed. Install it with:\n" "    pip install umap-learn"
        )

    standardized_features = np.asarray(
        standardized_features,
        dtype=float,
    )

    if not np.all(np.isfinite(standardized_features)):
        raise ValueError("UMAP input contains NaN or infinite values")

    reducer = umap.UMAP(
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        n_components=config.n_components,
        metric=config.metric,
        random_state=config.random_state,
    )

    embedding = reducer.fit_transform(standardized_features)
    return reducer, embedding


def run_dbscan(
    embedding: np.ndarray,
    config: Optional[ClusteringConfig] = None,
) -> np.ndarray:
    """
    Run sklearn DBSCAN using settings close to dbscanCD.m.

    MATLAB uses distance < epsilon, whereas sklearn generally uses
    distance <= epsilon. Exact boundary behavior can therefore differ.
    """
    if config is None:
        config = ClusteringConfig()

    model = DBSCAN(
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
        metric="euclidean",
    )

    return model.fit_predict(embedding)


def number_of_clusters(labels: np.ndarray) -> int:
    """Count non-noise cluster labels."""
    labels = np.asarray(labels)
    return len(np.unique(labels[labels >= 0]))


# ============================================================================
# Reassignment of unclustered UMAP points
# ============================================================================


def _density_along_line(
    all_points: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    bins: int,
    radius: float,
) -> np.ndarray:
    """Estimate local point density along a line in UMAP space."""
    fractions = np.linspace(0.0, 1.0, bins)
    sample_points = start[None, :] + fractions[:, None] * (stop - start)[None, :]

    density = np.zeros(bins, dtype=float)

    for index, point in enumerate(sample_points):
        distances = np.linalg.norm(
            all_points - point[None, :],
            axis=1,
        )
        density[index] = np.sum(distances <= radius)

    return density


def _longest_successive_increase(
    density: np.ndarray,
) -> int:
    """Length of the longest run of increasing density values."""
    differences = np.diff(density)

    longest = 0
    current = 0

    for difference in differences:
        if difference > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def reassign_border_points(
    embedding: np.ndarray,
    labels: np.ndarray,
    config: Optional[ClusteringConfig] = None,
) -> np.ndarray:
    """
    Reassign unclustered points near a cluster edge.

    The paper does not report bin count or density radius, so this remains
    an approximate reconstruction.
    """
    if config is None:
        config = ClusteringConfig()

    embedding = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels, dtype=int).copy()

    unclustered = np.flatnonzero(labels < 0)
    cluster_ids = np.unique(labels[labels >= 0])

    if len(unclustered) == 0 or len(cluster_ids) == 0:
        return labels

    clustered_mask = labels >= 0
    clustered_points = embedding[clustered_mask]
    clustered_labels = labels[clustered_mask]

    centroids = {
        cluster_id: np.mean(
            embedding[labels == cluster_id],
            axis=0,
        )
        for cluster_id in cluster_ids
    }

    nearest_clustered = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    ).fit(clustered_points)

    for point_index in unclustered:
        point = embedding[point_index]

        distance, _ = nearest_clustered.kneighbors(point[None, :])

        if distance[0, 0] > config.border_assignment_radius:
            continue

        all_distances = np.linalg.norm(
            clustered_points - point[None, :],
            axis=1,
        )

        candidates = np.unique(
            clustered_labels[all_distances <= config.border_assignment_radius]
        )

        best_cluster = None
        best_score = -1

        for cluster_id in candidates:
            profile = _density_along_line(
                embedding,
                point,
                centroids[cluster_id],
                bins=config.border_density_bins,
                radius=config.border_density_radius,
            )

            score = _longest_successive_increase(profile)

            if score > best_score:
                best_score = score
                best_cluster = int(cluster_id)

        if best_cluster is not None:
            labels[point_index] = best_cluster

    return labels


# ============================================================================
# Held-out UMAP transformation and nearest-neighbor assignment
# ============================================================================


def assign_heldout_events(
    reducer,
    training_embedding: np.ndarray,
    training_labels: np.ndarray,
    new_standardized_features: np.ndarray,
    config: Optional[ClusteringConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform held-out events into the fitted UMAP space and assign labels
    from the 100 nearest training neighbors.
    """
    if config is None:
        config = ClusteringConfig()

    new_embedding = reducer.transform(new_standardized_features)

    n_neighbors = min(
        config.heldout_neighbors,
        len(training_embedding),
    )

    nearest_neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    ).fit(training_embedding)

    distances, indices = nearest_neighbors.kneighbors(new_embedding)

    assigned = np.full(
        len(new_embedding),
        -1,
        dtype=int,
    )

    for event_index in range(len(new_embedding)):
        median_distance = np.median(distances[event_index])

        if median_distance > config.heldout_max_median_distance:
            continue

        neighbor_labels = training_labels[indices[event_index]]
        neighbor_labels = neighbor_labels[neighbor_labels >= 0]

        if neighbor_labels.size == 0:
            continue

        values, counts = np.unique(
            neighbor_labels,
            return_counts=True,
        )

        assigned[event_index] = int(values[np.argmax(counts)])

    return assigned, new_embedding


# ============================================================================
# Biphasic convergent event detection
# ============================================================================


def _sample_std(values: np.ndarray) -> float:
    """MATLAB-like sample standard deviation with NaN omission."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if len(finite) < 2:
        return np.nan

    return float(np.std(finite, ddof=1))


def detect_biphasic_convergent(
    left_position: np.ndarray,
    right_position: np.ndarray,
    fs: float,
    onset_index: int,
    is_tethered: bool = True,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Detect a BConv candidate following FindConvConj.m.

    The recovered coordinate convention is:

    - Left-eye temporal movement: negative velocity
    - Right-eye temporal movement: positive velocity

    The temporal velocity search covers the 100 ms preceding onset.
    The displacement baseline covers 150 ms ending 100 ms before onset.
    """
    left_position = np.asarray(
        left_position,
        dtype=float,
    ).ravel()

    right_position = np.asarray(
        right_position,
        dtype=float,
    ).ravel()

    if left_position.shape != right_position.shape:
        raise ValueError("left_position and right_position must have equal shapes")

    velocity_threshold = 60.0 if is_tethered else 40.0

    baseline_samples = int(round(0.150 * fs))
    gap_samples = int(round(0.100 * fs))
    search_samples = int(round(0.100 * fs))

    baseline_stop = onset_index - gap_samples
    baseline_start = baseline_stop - baseline_samples

    search_start = onset_index - search_samples
    search_stop = onset_index + 1

    details: Dict[str, Any] = {
        "velocity_threshold": velocity_threshold,
        "left_candidate": False,
        "right_candidate": False,
    }

    if baseline_start < 0 or search_start < 0 or search_stop > len(left_position):
        details["invalid_reason"] = "recording_boundary"
        return False, None, details

    left_baseline = left_position[baseline_start:baseline_stop]
    right_baseline = right_position[baseline_start:baseline_stop]

    if not np.all(np.isfinite(left_baseline)) or not np.all(
        np.isfinite(right_baseline)
    ):
        details["invalid_reason"] = "invalid_baseline"
        return False, None, details

    left_search_position = left_position[search_start:search_stop]
    right_search_position = right_position[search_start:search_stop]

    if not np.all(np.isfinite(left_search_position)) or not np.all(
        np.isfinite(right_search_position)
    ):
        details["invalid_reason"] = "invalid_search_window"
        return False, None, details

    left_velocity = np.gradient(
        left_position,
        1.0 / fs,
    )
    right_velocity = np.gradient(
        right_position,
        1.0 / fs,
    )

    left_search_velocity = left_velocity[search_start:search_stop]
    right_search_velocity = right_velocity[search_start:search_stop]

    left_velocity_local_index = int(np.argmin(left_search_velocity))
    right_velocity_local_index = int(np.argmax(right_search_velocity))

    left_velocity_index = search_start + left_velocity_local_index
    right_velocity_index = search_start + right_velocity_local_index

    left_baseline_mean = float(np.mean(left_baseline))
    right_baseline_mean = float(np.mean(right_baseline))

    left_baseline_std = _sample_std(left_baseline)
    right_baseline_std = _sample_std(right_baseline)

    left_position_threshold = left_baseline_mean - left_baseline_std
    right_position_threshold = right_baseline_mean + right_baseline_std

    left_minimum_after_velocity_peak = float(
        np.min(left_position[left_velocity_index:search_stop])
    )

    right_maximum_after_velocity_peak = float(
        np.max(right_position[right_velocity_index:search_stop])
    )

    left_minimum_velocity = float(left_search_velocity[left_velocity_local_index])
    right_maximum_velocity = float(right_search_velocity[right_velocity_local_index])

    left_candidate = (
        left_minimum_velocity < -velocity_threshold
        and left_minimum_after_velocity_peak <= left_position_threshold
    )

    right_candidate = (
        right_maximum_velocity > velocity_threshold
        and right_maximum_after_velocity_peak >= right_position_threshold
    )

    details.update(
        {
            "left_candidate": bool(left_candidate),
            "right_candidate": bool(right_candidate),
            "left_minimum_velocity": left_minimum_velocity,
            "right_maximum_velocity": right_maximum_velocity,
            "left_baseline_mean": left_baseline_mean,
            "right_baseline_mean": right_baseline_mean,
            "left_baseline_std": left_baseline_std,
            "right_baseline_std": right_baseline_std,
            "left_position_threshold": left_position_threshold,
            "right_position_threshold": right_position_threshold,
            "left_minimum_after_velocity_peak": (left_minimum_after_velocity_peak),
            "right_maximum_after_velocity_peak": (right_maximum_after_velocity_peak),
            "invalid_reason": None,
        }
    )

    if left_candidate and not right_candidate:
        return True, "L", details

    if right_candidate and not left_candidate:
        return True, "R", details

    if left_candidate and right_candidate:
        return True, "both", details

    return False, None, details


def reassign_conv_to_bconv(
    labels: np.ndarray,
    conv_cluster_id: int,
    left_bconv_cluster_id: int,
    right_bconv_cluster_id: int,
    bconv_flags: Sequence[bool],
    bconv_sides: Sequence[Optional[str]],
) -> np.ndarray:
    """Reassign flagged Conv events to left/right BConv clusters."""
    labels = np.asarray(labels, dtype=int).copy()

    if not (len(labels) == len(bconv_flags) == len(bconv_sides)):
        raise ValueError("labels, bconv_flags, and bconv_sides must have equal lengths")

    for index, (flag, side) in enumerate(zip(bconv_flags, bconv_sides)):
        if not flag or labels[index] != conv_cluster_id:
            continue

        if side == "L":
            labels[index] = left_bconv_cluster_id
        elif side == "R":
            labels[index] = right_bconv_cluster_id
        elif side == "both":
            # The recovered MATLAB removes overlapping left/right
            # candidates. Leave this ambiguous event unchanged.
            continue

    return labels


# ============================================================================
# Basic quality-control summaries
# ============================================================================


def summarize_trial_result(
    result: Dict[str, Any],
) -> Dict[str, int]:
    """Return event and missing-data quality-control counts."""
    event_records = result["event_records"]

    invalid_left = sum(not record["left_metrics"]["valid"] for record in event_records)
    invalid_right = sum(
        not record["right_metrics"]["valid"] for record in event_records
    )

    return {
        "left_coarse_events": len(result["coarse"]["left_events"]["t"]),
        "right_coarse_events": len(result["coarse"]["right_events"]["t"]),
        "paired_or_monocular_events": len(result["paired_events"]),
        "events_after_refractory_filter": len(result["retained_events"]),
        "valid_metric_events": len(result["features"]),
        "events_invalid_left_metrics": invalid_left,
        "events_invalid_right_metrics": invalid_right,
    }

