"""Recording, trial, stimulus, and position context shared by bouts and saccades."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from BehaviorScreen.core import ROLLOVER_TIME_S, Stim
from BehaviorScreen.load import fish_metadata
from BehaviorScreen.recording import Recording, nearest_indices
from BehaviorScreen.stimulus import (
    looming_constant_velocity_approach,
    prey_capture_arc_stimulus_cosine,
)

TRIAL_COLUMNS_TO_EXCLUDE = {
    "start_timestamp",
    "stop_timestamp",
    "start_frame",
    "stop_frame",
    "trial_duration_s",
    "epoch_name",
    "epoch_idx",
    "trial_num",
    "stim_select",
}


def scalar_for_csv(value: Any) -> Any:
    """Convert a value into a scalar suitable for a CSV table."""
    if value is None:
        return np.nan
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        missing = pd.isna(value)
        if np.isscalar(missing) and missing:
            return np.nan
    except (TypeError, ValueError):
        pass
    if hasattr(value, "value"):
        return value.value
    return str(value)


def trial_indices(rec: Recording, event_timestamps: np.ndarray) -> np.ndarray:
    """Row of rec.trials containing each event (start <= t < stop), else -1."""
    starts = rec.trials["start_timestamp"].to_numpy(np.int64)
    stops = rec.trials["stop_timestamp"].to_numpy(np.int64)
    t = np.asarray(event_timestamps, dtype=np.int64)
    if len(starts) == 0:
        return np.full(len(t), -1)
    indices = np.searchsorted(starts, t, side="right") - 1
    inside = (indices >= 0) & (t < stops[np.clip(indices, 0, None)])
    return np.where(inside, indices, -1)


def calculate_stimulus_values(trial: pd.Series, trial_time_s: float) -> dict[str, float]:
    """Time-dependent stimulus values at an event."""
    result = {"stim_phase": np.nan, "looming_radius": np.nan}
    stimulus = trial.get("stim_select", None)

    try:
        if stimulus == Stim.PREY_CAPTURE:
            result["stim_phase"] = float(
                prey_capture_arc_stimulus_cosine(
                    trial.start_time_sec,
                    trial_time_s,
                    ROLLOVER_TIME_S,
                    trial.prey_arc_start_deg,
                    trial.prey_arc_stop_deg,
                    trial.prey_speed_deg_s,
                )
            )
        elif stimulus == Stim.LOOMING:
            result["looming_radius"] = float(
                looming_constant_velocity_approach(
                    trial.start_time_sec,
                    trial_time_s,
                    ROLLOVER_TIME_S,
                    trial.looming_angle_start_deg,
                    trial.looming_angle_stop_deg,
                    trial.looming_size_to_speed_ratio_ms,
                    trial.looming_distance_to_screen_mm,
                )
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        pass

    return result


def trial_context(rec: Recording, event_timestamp: int, trial_index: int) -> dict[str, Any]:
    """Trial and stimulus context for one event."""
    context: dict[str, Any] = {
        "event_timestamp": int(event_timestamp),
        "in_stimulus_trial": trial_index >= 0,
        "stim": np.nan,
        "epoch_name": np.nan,
        "epoch_idx": np.nan,
        "trial_num": np.nan,
        "trial_time": np.nan,
        "stim_start_time": np.nan,
        "stim_phase": np.nan,
        "looming_radius": np.nan,
    }
    if trial_index < 0:
        return context

    trial = rec.trials.iloc[int(trial_index)]
    trial_time_s = (event_timestamp - int(trial.start_timestamp)) * 1e-9
    first_start = int(rec.trials["start_timestamp"].iloc[0])

    context.update(
        {
            "stim": scalar_for_csv(trial.stim_select),
            "epoch_name": scalar_for_csv(trial.epoch_name),
            "epoch_idx": scalar_for_csv(trial.epoch_idx),
            "trial_num": int(trial.trial_num),
            "trial_time": float(trial_time_s),
            "stim_start_time": (int(trial.start_timestamp) - first_start) * 1e-9,
        }
    )
    for column, value in trial.items():
        if column not in TRIAL_COLUMNS_TO_EXCLUDE and column not in context:
            context[column] = scalar_for_csv(value)

    context.update(calculate_stimulus_values(trial, trial_time_s))
    return context


def position_context(
    rec: Recording, x_mm: float, y_mm: float, heading: float = np.nan
) -> dict[str, float]:
    """Event position relative to the well center (NaN if no well was found)."""
    well = rec.well
    x_start = x_mm - (np.nan if well is None else well.x_mm)
    y_start = y_mm - (np.nan if well is None else well.y_mm)
    return {
        "x_start": float(x_start),
        "y_start": float(y_start),
        "heading_start": float(heading),
        "distance_center": float(np.hypot(x_start, y_start)),
    }


def event_context(
    rec: Recording,
    event_timestamp: int,
    trial_index: int,
    x_mm: float,
    y_mm: float,
    heading: float = np.nan,
) -> dict[str, Any]:
    """Recording, trial, stimulus, and position context for one event."""
    return (
        fish_metadata(rec.name)
        | trial_context(rec, event_timestamp, trial_index)
        | position_context(rec, x_mm, y_mm, heading)
    )


def posthoc_positions_at(rec: Recording, event_timestamps: np.ndarray) -> dict[str, np.ndarray]:
    """
    Pose-estimation position (Swim_Bladder, mm) and heading (Swim_Bladder ->
    Head) at the nearest frame with valid keypoints, for each event.
    """
    event_timestamps = np.asarray(event_timestamps, dtype=np.int64)
    n = len(event_timestamps)
    result = {
        "tracking_frame": np.full(n, -1),
        "tracking_time_error_ms": np.full(n, np.nan),
        "x_mm": np.full(n, np.nan),
        "y_mm": np.full(n, np.nan),
        "heading": np.full(n, np.nan),
    }
    if rec.pose.empty:
        return result

    swim_bladder = rec.pose.Swim_Bladder[["x", "y"]].to_numpy(float)
    head = rec.pose.Head[["x", "y"]].to_numpy(float)
    valid = np.flatnonzero(
        np.isfinite(swim_bladder).all(axis=1) & np.isfinite(head).all(axis=1)
    )
    if len(valid) == 0:
        return result

    frame_time = rec.frame_time_ns[valid]
    frames = valid[nearest_indices(frame_time, event_timestamps)]
    heading = head[frames] - swim_bladder[frames]

    result["tracking_frame"] = frames
    result["tracking_time_error_ms"] = (rec.frame_time_ns[frames] - event_timestamps) * 1e-6
    result["x_mm"], result["y_mm"] = (swim_bladder[frames] / rec.pix_per_mm).T
    result["heading"] = np.arctan2(heading[:, 1], heading[:, 0])
    return result
