"""One recorded fish: lazily loaded data, frame clock, trials and well."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
from video_tools import OpenCV_VideoReader

from BehaviorScreen.core import STIM_PARAMETERS
from BehaviorScreen.load import (
    BehaviorFiles,
    load_lightning_pose,
    load_metadata,
    load_stimuli,
    load_temperature,
    load_tracking,
    load_video,
    load_video_timestamps,
)
from BehaviorScreen.process import Well


def nearest_indices(reference: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the nearest value in sorted `reference` for each target (ties: left)."""
    right = np.clip(np.searchsorted(reference, targets), 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    take_right = np.abs(reference[right] - targets) < np.abs(targets - reference[left])
    return np.where(take_right, right, left)


@dataclass
class Recording:
    """
    Frames are positions in time order. They equal video frames (rows of the
    video timestamps file and of Lightning Pose outputs) unless the
    acquisition wrote frames out of order; `video_frames` maps back. Camera
    frame counters (`index` columns) only place online tracking on this clock.
    """

    files: BehaviorFiles

    @property
    def name(self) -> str:
        return self.files.metadata.stem

    @cached_property
    def metadata(self) -> dict:
        return load_metadata(self.files.metadata)

    @cached_property
    def stimuli(self) -> list[dict]:
        return load_stimuli(self.files.stimuli)

    @cached_property
    def tracking(self) -> pd.DataFrame:
        return load_tracking(self.files.tracking)

    @cached_property
    def pose(self) -> pd.DataFrame:
        return self._in_time_order(load_lightning_pose(self.files.full_tracking, self.n_frames))

    @cached_property
    def eyes(self) -> pd.DataFrame:
        return self._in_time_order(load_lightning_pose(self.files.eyes_tracking, self.n_frames))

    def _in_time_order(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Video-order rows (one per video frame) reordered like the frame clock."""
        order = self.video_frames[: len(rows)]
        if np.array_equal(order, np.arange(len(rows))):
            return rows
        if order.max() >= len(rows):
            raise ValueError(f"{self.name}: {len(rows)} rows do not cover the reordered frames")
        return rows.iloc[order].reset_index(drop=True)

    @cached_property
    def video(self) -> OpenCV_VideoReader:
        return load_video(self.files.video)

    @cached_property
    def video_timestamps(self) -> pd.DataFrame:
        return load_video_timestamps(self.files.video_timestamps)  # type: ignore

    @cached_property
    def temperature(self) -> pd.DataFrame:
        return load_temperature(self.files.temperature)

    @property
    def pix_per_mm(self) -> float:
        return float(self.metadata["calibration"]["pix_per_mm"])

    @property
    def fps(self) -> float:
        return float(self.metadata["camera"]["framerate_value"])

    @cached_property
    def video_frames(self) -> np.ndarray:
        """Video frame at each frame position: video frames sorted by timestamp."""
        return np.argsort(self.video_timestamps["timestamp"].to_numpy(np.int64), kind="stable")

    @cached_property
    def frame_time_ns(self) -> np.ndarray:
        return self.video_timestamps["timestamp"].to_numpy(np.int64)[self.video_frames]

    @property
    def n_frames(self) -> int:
        return len(self.frame_time_ns)

    def frames_at(self, timestamps_ns: np.ndarray) -> np.ndarray:
        """Nearest frame for each timestamp."""
        return nearest_indices(self.frame_time_ns, np.asarray(timestamps_ns))

    @cached_property
    def online(self) -> pd.DataFrame:
        """Online tracking, one row per frame; NaN where the tracker missed."""
        counter = self.video_timestamps["index"].to_numpy()[self.video_frames]
        frame = pd.Series(np.arange(self.n_frames), index=counter)
        tracking = self.tracking.assign(frame=self.tracking["index"].map(frame))
        return (
            tracking.dropna(subset=["frame"])
            .astype({"frame": int})
            .set_index("frame")
            .reindex(range(self.n_frames))
        )

    @cached_property
    def trials(self) -> pd.DataFrame:
        return build_trials(self)

    @property
    def well(self) -> Optional[Well]:
        if "well" not in self.metadata:
            raise KeyError(
                f"{self.files.metadata} has no well; re-export metadata with "
                "`python -m BehaviorScreen.export ROOT --overwrite --no-tracking "
                "--no-timestamps --no-stimuli --no-videos`"
            )
        well = self.metadata["well"]
        return None if well is None else Well(**well)


def build_trials(rec: Recording) -> pd.DataFrame:
    """
    The single trial table: one row per logged stimulus epoch, in log order.
    A trial lasts until the next epoch starts (the last one until the end of
    the recording). `trial_num` counts occurrences of each epoch_name.
    """
    start = np.array([s["timestamp"] for s in rec.stimuli], dtype=np.int64)
    if np.any(np.diff(start) < 0):
        raise ValueError(f"{rec.name}: stimulus log is not in time order")
    end = max(int(rec.tracking["timestamp"].max()), int(rec.frame_time_ns.max()))
    stop = np.append(start[1:], end)

    trials = pd.DataFrame(
        [
            {
                "stim_select": int(s["stim_select"]),
                "epoch_name": s.get("name", ""),
                "looming_center_mm_x": s.get("looming_center_mm", [pd.NA, pd.NA])[0],
                "looming_center_mm_y": s.get("looming_center_mm", [pd.NA, pd.NA])[1],
                "foreground_color": str(s["foreground_color"]),
                "background_color": str(s["background_color"]),
                **{p: s.get(p, pd.NA) for p in STIM_PARAMETERS},
            }
            for s in rec.stimuli
        ]
    )
    trials.insert(2, "start_timestamp", start)
    trials.insert(3, "stop_timestamp", stop)
    trials["start_frame"] = rec.frames_at(start)
    trials["stop_frame"] = rec.frames_at(stop)
    trials["trial_duration_s"] = (stop - start) * 1e-9
    trials["epoch_idx"] = np.arange(len(trials))
    trials["trial_num"] = trials.groupby("epoch_name", sort=False).cumcount()
    return trials
