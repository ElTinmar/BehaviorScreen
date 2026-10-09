"""Shared recording, trial, stimulus, and position context for events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from BehaviorScreen.core import Stim
from BehaviorScreen.load import (
    BehaviorData,
    BehaviorFiles,
    Directories,
    encode_time_of_day,
    parse_fish,
)
from BehaviorScreen.process import get_trials, get_well_coords_mm
from BehaviorScreen.stimulus import (
    looming_constant_velocity_approach,
    prey_capture_arc_stimulus_cosine,
)

TRIAL_COLUMNS_TO_EXCLUDE = {
    "start_timestamp",
    "stop_timestamp",
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


def prepare_trial_table(
    behavior_data: BehaviorData,
) -> pd.DataFrame:
    """
    Prepare stimulus trials with epoch-local trial numbering.

    ``trial_num`` is zero-based and restarts within each raw
    ``epoch_name``. This matches the convention used in the bout and
    saccade analysis tables.
    """
    trials = get_trials(behavior_data)

    if trials.empty:
        return trials.copy()

    records: list[dict[str, Any]] = []

    for epoch_name, epoch_trials in trials.groupby(
        "epoch_name",
        sort=False,
    ):
        for trial_num, (epoch_index, trial) in enumerate(epoch_trials.iterrows()):
            record = trial.to_dict()
            record["epoch_name"] = epoch_name
            record["epoch_idx"] = epoch_index
            record["trial_num"] = trial_num
            records.append(record)

    return (
        pd.DataFrame.from_records(records)
        .sort_values("start_timestamp", kind="stable")
        .reset_index(drop=True)
    )


def find_trial_indices(
    event_timestamps: np.ndarray,
    trial_starts: np.ndarray,
    trial_stops: np.ndarray,
) -> np.ndarray:
    """
    Find the trial containing each event timestamp.

    Trial intervals are treated as:

        start_timestamp <= event_timestamp < stop_timestamp

    Events outside all trials receive index ``-1``.
    """
    event_timestamps = np.asarray(
        event_timestamps,
        dtype=np.int64,
    )
    trial_starts = np.asarray(
        trial_starts,
        dtype=np.int64,
    )
    trial_stops = np.asarray(
        trial_stops,
        dtype=np.int64,
    )

    if trial_starts.shape != trial_stops.shape:
        raise ValueError("trial_starts and trial_stops must have the same shape.")

    if len(trial_starts) == 0:
        return np.full(
            len(event_timestamps),
            -1,
            dtype=int,
        )

    if np.any(np.diff(trial_starts) < 0):
        raise ValueError("Trial start timestamps must be sorted in ascending order.")

    indices = (
        np.searchsorted(
            trial_starts,
            event_timestamps,
            side="right",
        )
        - 1
    )

    valid = (indices >= 0) & (indices < len(trial_starts))

    safe_indices = np.clip(
        indices,
        0,
        len(trial_starts) - 1,
    )

    valid &= event_timestamps < trial_stops[safe_indices]

    indices[~valid] = -1

    return indices


def nearest_indices(
    reference: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Find the nearest ascending reference value for every target."""
    reference = np.asarray(reference)
    targets = np.asarray(targets)

    if reference.ndim != 1:
        raise ValueError("reference must be one-dimensional.")

    if len(reference) == 0:
        raise ValueError("reference cannot be empty.")

    if np.any(np.diff(reference) < 0):
        raise ValueError("reference must be sorted in ascending order.")

    insertion_indices = np.searchsorted(
        reference,
        targets,
    )

    right_indices = np.clip(
        insertion_indices,
        0,
        len(reference) - 1,
    )
    left_indices = np.clip(
        insertion_indices - 1,
        0,
        len(reference) - 1,
    )

    left_distances = np.abs(targets - reference[left_indices])
    right_distances = np.abs(reference[right_indices] - targets)

    return np.where(
        right_distances < left_distances,
        right_indices,
        left_indices,
    )


def calculate_stimulus_values(
    trial: pd.Series,
    trial_time_s: float,
    rollover_time_s: int,
) -> dict[str, float]:
    """Calculate time-dependent stimulus values at an event."""
    result = {
        "stim_phase": np.nan,
        "looming_radius": np.nan,
    }

    stimulus = trial.get("stim_select", None)

    try:
        if stimulus == Stim.PREY_CAPTURE:
            result["stim_phase"] = float(
                prey_capture_arc_stimulus_cosine(
                    trial.start_time_sec,
                    trial_time_s,
                    rollover_time_s,
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
                    rollover_time_s,
                    trial.looming_angle_start_deg,
                    trial.looming_angle_stop_deg,
                    trial.looming_size_to_speed_ratio_ms,
                    trial.looming_distance_to_screen_mm,
                )
            )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        pass

    return result


@dataclass
class RecordingContext:
    """Reusable recording-level context for behavioral events."""

    directories: Directories
    behavior_files: BehaviorFiles
    behavior_data: BehaviorData
    rollover_time_s: int = 3600

    file: str = field(init=False)
    file_info: Any = field(init=False)
    cos_daytime: float = field(init=False)
    sin_daytime: float = field(init=False)
    trials: pd.DataFrame = field(init=False)
    trial_starts: np.ndarray = field(init=False)
    trial_stops: np.ndarray = field(init=False)
    first_trial_start: int | None = field(init=False)
    recording_start_timestamp: int = field(init=False)
    well_center_x_mm: float = field(init=False)
    well_center_y_mm: float = field(init=False)

    def __post_init__(self) -> None:
        self.file = self.behavior_files.metadata.stem
        self.file_info = parse_fish(self.file)

        daytime = encode_time_of_day(self.file_info)
        self.cos_daytime = float(daytime[0])
        self.sin_daytime = float(daytime[1])

        self.trials = prepare_trial_table(self.behavior_data)

        if self.trials.empty:
            self.trial_starts = np.array(
                [],
                dtype=np.int64,
            )
            self.trial_stops = np.array(
                [],
                dtype=np.int64,
            )
            self.first_trial_start = None
        else:
            self.trial_starts = self.trials["start_timestamp"].to_numpy(dtype=np.int64)
            self.trial_stops = self.trials["stop_timestamp"].to_numpy(dtype=np.int64)
            self.first_trial_start = int(self.trial_starts.min())

        video_timestamps = self.behavior_data.video_timestamps.timestamp.to_numpy()

        if len(video_timestamps) == 0:
            raise ValueError(f"No video timestamps were found for {self.file}.")

        self.recording_start_timestamp = int(video_timestamps[0])

        try:
            (
                self.well_center_x_mm,
                self.well_center_y_mm,
                _,
            ) = get_well_coords_mm(
                self.directories,
                self.behavior_files,
                self.behavior_data,
            )
        except Exception as error:
            print(
                f"[warn] Could not determine well center for " f"{self.file}: {error}"
            )
            self.well_center_x_mm = np.nan
            self.well_center_y_mm = np.nan

    def recording_metadata(self) -> dict[str, Any]:
        """Return metadata constant within one recording."""
        return {
            "file": self.file,
            "dpf": self.file_info.age,
            "day": (
                f"{self.file_info.day}."
                f"{self.file_info.month}."
                f"{self.file_info.year}"
            ),
            "cos_daytime": self.cos_daytime,
            "sin_daytime": self.sin_daytime,
        }

    def relative_seconds_to_timestamps(
        self,
        relative_seconds: np.ndarray,
    ) -> np.ndarray:
        """Convert seconds from recording start to absolute nanoseconds."""
        relative_seconds = np.asarray(
            relative_seconds,
            dtype=float,
        )

        if not np.isfinite(relative_seconds).all():
            raise ValueError("Relative event times contain NaN or infinity.")

        offsets_ns = np.rint(relative_seconds * 1e9).astype(np.int64)

        return self.recording_start_timestamp + offsets_ns

    def trial_indices(
        self,
        event_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Find the containing trial for each event."""
        return find_trial_indices(
            event_timestamps=event_timestamps,
            trial_starts=self.trial_starts,
            trial_stops=self.trial_stops,
        )

    def position_context(
        self,
        x_mm: float,
        y_mm: float,
        heading: float = np.nan,
    ) -> dict[str, float]:
        """Express an event position relative to the well center."""
        x_start = x_mm - self.well_center_x_mm
        y_start = y_mm - self.well_center_y_mm

        return {
            "x_start": float(x_start),
            "y_start": float(y_start),
            "heading_start": float(heading),
            "distance_center": float(np.hypot(x_start, y_start)),
        }

    def trial_context(
        self,
        event_timestamp: int,
        trial_index: int,
    ) -> dict[str, Any]:
        """Return trial and stimulus context for one event."""
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

        trial = self.trials.iloc[int(trial_index)]

        trial_time_s = (event_timestamp - int(trial.start_timestamp)) * 1e-9

        if self.first_trial_start is None:
            stimulus_start_time = np.nan
        else:
            stimulus_start_time = (
                int(trial.start_timestamp) - self.first_trial_start
            ) * 1e-9

        context.update(
            {
                "stim": scalar_for_csv(trial.get("stim_select", np.nan)),
                "epoch_name": scalar_for_csv(trial.epoch_name),
                "epoch_idx": scalar_for_csv(trial.epoch_idx),
                "trial_num": int(trial.trial_num),
                "trial_time": float(trial_time_s),
                "stim_start_time": float(stimulus_start_time),
            }
        )

        for column, value in trial.items():
            if column in TRIAL_COLUMNS_TO_EXCLUDE:
                continue

            if column not in context:
                context[column] = scalar_for_csv(value)

        context.update(
            calculate_stimulus_values(
                trial=trial,
                trial_time_s=trial_time_s,
                rollover_time_s=self.rollover_time_s,
            )
        )

        return context

    def event_context(
        self,
        event_timestamp: int,
        trial_index: int,
        x_mm: float,
        y_mm: float,
        heading: float = np.nan,
    ) -> dict[str, Any]:
        """Return recording, trial, stimulus, and position context."""
        context = self.recording_metadata()

        context.update(
            self.trial_context(
                event_timestamp=event_timestamp,
                trial_index=trial_index,
            )
        )

        context.update(
            self.position_context(
                x_mm=x_mm,
                y_mm=y_mm,
                heading=heading,
            )
        )

        return context

    def posthoc_positions_at(
        self,
        event_timestamps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Get post-hoc position and heading nearest each event timestamp.

        Position is taken from the Lightning Pose ``Swim_Bladder``
        keypoint. Heading is calculated from ``Swim_Bladder`` to ``Head``.
        """
        event_timestamps = np.asarray(
            event_timestamps,
            dtype=np.int64,
        )
        number_of_events = len(event_timestamps)

        result = {
            "tracking_frame": np.full(
                number_of_events,
                -1,
                dtype=int,
            ),
            "tracking_time_error_ms": np.full(
                number_of_events,
                np.nan,
            ),
            "x_mm": np.full(
                number_of_events,
                np.nan,
            ),
            "y_mm": np.full(
                number_of_events,
                np.nan,
            ),
            "heading": np.full(
                number_of_events,
                np.nan,
            ),
        }

        tracking = self.behavior_data.tracking
        full_tracking = self.behavior_data.full_tracking

        if tracking.empty or full_tracking.empty:
            return result

        if "timestamp" not in tracking.columns:
            return result

        try:
            tracking_timestamps_raw = tracking["timestamp"].to_numpy()

            swim_bladder_px = full_tracking.Swim_Bladder[["x", "y"]].to_numpy(
                dtype=float
            )

            head_px = full_tracking.Head[["x", "y"]].to_numpy(dtype=float)
        except (AttributeError, KeyError):
            return result

        number_of_frames = min(
            len(tracking_timestamps_raw),
            len(swim_bladder_px),
            len(head_px),
        )

        if number_of_frames == 0:
            return result

        tracking_timestamps_raw = tracking_timestamps_raw[:number_of_frames]
        swim_bladder_px = swim_bladder_px[:number_of_frames]
        head_px = head_px[:number_of_frames]

        finite = (
            np.isfinite(tracking_timestamps_raw)
            & np.isfinite(swim_bladder_px).all(axis=1)
            & np.isfinite(head_px).all(axis=1)
        )

        if not finite.any():
            return result

        valid_frame_indices = np.flatnonzero(finite)
        valid_timestamps = tracking_timestamps_raw[finite].astype(np.int64)

        order = np.argsort(valid_timestamps)
        valid_timestamps = valid_timestamps[order]
        valid_frame_indices = valid_frame_indices[order]

        nearest_local_indices = nearest_indices(
            reference=valid_timestamps,
            targets=event_timestamps,
        )

        nearest_frames = valid_frame_indices[nearest_local_indices]

        pixels_per_mm = float(self.behavior_data.metadata["calibration"]["pix_per_mm"])

        positions_mm = swim_bladder_px[nearest_frames] / pixels_per_mm

        heading_vectors = head_px[nearest_frames] - swim_bladder_px[nearest_frames]

        headings = np.arctan2(
            heading_vectors[:, 1],
            heading_vectors[:, 0],
        )

        selected_timestamps = tracking_timestamps_raw[nearest_frames].astype(np.int64)

        result["tracking_frame"] = nearest_frames
        result["tracking_time_error_ms"] = (
            selected_timestamps - event_timestamps
        ) * 1e-6
        result["x_mm"] = positions_mm[:, 0]
        result["y_mm"] = positions_mm[:, 1]
        result["heading"] = headings

        return result
