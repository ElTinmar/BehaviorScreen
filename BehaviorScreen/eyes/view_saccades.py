#!/usr/bin/env python3
"""
Interactively display a Lightning Pose labeled video synchronized with
long eye-position traces and classified saccades.

The classified event CSV must contain:
    fish
    onset_time_s
    a classification column, such as class_label or cluster_id

Display
-------
- Labeled video
- Native camera-rate eye traces
- Low-likelihood raw samples in gray
- Likelihood-filtered/resampled 500 Hz traces
- Full 500 Hz smoothed traces
- Smoothed event snippets colored by saccade category
- Category labels and onset markers
- A black cursor showing the current video time

Controls
--------
Space
    Play or pause.

J or ,
    Previous frame.

L or .
    Next frame.

Shift+J
    Move backward 10 frames.

Shift+L
    Move forward 10 frames.

Left or A
    Pan the trace backward by half a window.

Right or D
    Pan the trace forward by half a window.

+ or =
    Zoom in.

- or _
    Zoom out.

Home
    Jump to the beginning.

End
    Jump to the end.

1
    Playback at 0.25x.

2
    Playback at 0.5x.

3
    Playback at 1x.

4
    Playback at 2x.

Mouse wheel
    Pan the trace horizontally.

Click on either trace
    Seek the video to the selected time.

Double-click on either trace
    Seek and print information about the nearest classified event.

Q or Escape
    Close the viewer.

Example
-------
python -m BehaviorScreen.eyes.view_saccade_trace \
    /media/martin/DATA_18TB/Screen/WT/vehicle \
    /path/to/classified_saccades.csv \
    --fish EXPERIMENT_NAME \
    --label-column class_label \
    --labeled-video /path/to/EXPERIMENT_NAME_labeled.mp4 \
    --mode freeswim \
    --window-s 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory

from BehaviorScreen.load import (
    BehaviorData,
    Directories,
    find_files,
    load_data,
)
from BehaviorScreen.process import compute_angle_between_vectors

import BehaviorScreen.eyes.saccade_pipeline as sp

# ============================================================================
# Constants
# ============================================================================

SAMPLING_RATE = 500.0

LEFT_RAW_COLOR = np.array([0.30, 0.30, 1.00])
RIGHT_RAW_COLOR = np.array([1.00, 0.30, 0.30])

LEFT_RESAMPLED_COLOR = np.array([0.20, 0.20, 0.85])
RIGHT_RESAMPLED_COLOR = np.array([0.85, 0.20, 0.20])

LEFT_SMOOTH_COLOR = np.array([0.00, 0.00, 1.00])
RIGHT_SMOOTH_COLOR = np.array([1.00, 0.00, 0.00])

INVALID_RAW_COLOR = np.array([0.65, 0.65, 0.65])


# ============================================================================
# Saccade-category colors
# ============================================================================

# Reconstructed from charlieColours('sactype6') and the category ordering
# used by the MATLAB plotting scripts.
CLASS_COLORS = {
    "LConj": np.array([0, 190, 65]) / 255.0,
    "RConj": np.array([204, 136, 197]) / 255.0,
    "Conv": np.array([0, 50, 211]) / 255.0,
    "BConvL": np.array([0, 202, 211]) / 255.0,
    "BConvR": np.array([103, 126, 216]) / 255.0,
    "ConvMini": np.array([195, 93, 46]) / 255.0,
    "Div": np.array([100, 53, 0]) / 255.0,
    "NonSac": np.array([0.50, 0.50, 0.50]),
    "Unclassified": np.array([0.20, 0.20, 0.20]),
}


CLASS_NAMES_BY_ID = {
    0: "Unclassified",
    1: "LConj",
    2: "RConj",
    3: "ConvMini",
    4: "Conv",
    5: "NonSac",
    6: "Div",
    7: "BConvR",
    8: "BConvL",
}


LABEL_ALIASES = {
    "unclassified": "Unclassified",
    "unclust": "Unclassified",
    "unclustered": "Unclassified",
    "noise": "Unclassified",
    "lconj": "LConj",
    "conjl": "LConj",
    "leftconj": "LConj",
    "leftconjugate": "LConj",
    "rconj": "RConj",
    "conjr": "RConj",
    "rightconj": "RConj",
    "rightconjugate": "RConj",
    "conv": "Conv",
    "convergent": "Conv",
    "regularconv": "Conv",
    "bconvl": "BConvL",
    "leftbconv": "BConvL",
    "biphasicconvl": "BConvL",
    "bconvr": "BConvR",
    "rightbconv": "BConvR",
    "biphasicconvr": "BConvR",
    "convmini": "ConvMini",
    "miniconv": "ConvMini",
    "smallconv": "ConvMini",
    "div": "Div",
    "divergent": "Div",
    "nonsac": "NonSac",
    "nonsaccadic": "NonSac",
    "nonsaccade": "NonSac",
}


def normalize_label(label: Any) -> str:
    """Convert numeric or text labels to standard category names."""
    if pd.isna(label):
        return "Unclassified"

    if isinstance(label, (int, np.integer)):
        label_id = int(label)
        return CLASS_NAMES_BY_ID.get(
            label_id,
            f"Class {label_id}",
        )

    if isinstance(label, (float, np.floating)):
        if np.isfinite(label) and float(label).is_integer():
            label_id = int(label)
            return CLASS_NAMES_BY_ID.get(
                label_id,
                f"Class {label_id}",
            )

        return str(label)

    text = str(label).strip()

    try:
        numeric_label = float(text)

        if np.isfinite(numeric_label) and numeric_label.is_integer():
            label_id = int(numeric_label)
            return CLASS_NAMES_BY_ID.get(
                label_id,
                f"Class {label_id}",
            )
    except ValueError:
        pass

    lookup_key = text.lower().replace(" ", "").replace("-", "").replace("_", "")

    return LABEL_ALIASES.get(lookup_key, text)


def class_color(label: Any) -> np.ndarray:
    """Return the category color associated with a class label."""
    class_name = normalize_label(label)

    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]

    # Stable fallback color for unknown labels.
    colors = plt.get_cmap("tab20").colors
    color_index = abs(hash(class_name)) % len(colors)

    return np.asarray(
        colors[color_index],
        dtype=float,
    )


def infer_label_column(events: pd.DataFrame) -> str:
    """Infer the classification column from common names."""
    candidates = (
        "class_label",
        "classified_label",
        "saccade_type",
        "class_name",
        "cluster_name",
        "cluster_label",
        "cluster_id",
        "label",
        "Idx2",
        "Idx",
    )

    for column in candidates:
        if column in events.columns:
            return column

    raise KeyError(
        "Could not infer the classification column. "
        "Specify it with --label-column.\n"
        f"Available columns: {list(events.columns)}"
    )


# ============================================================================
# Eye angle and likelihood extraction
# ============================================================================


def extract_landmark_likelihood(
    landmark: pd.DataFrame,
    landmark_name: str,
) -> np.ndarray:
    """Extract a Lightning Pose confidence column."""
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
        f"No likelihood column was found for {landmark_name}. "
        f"Available columns: {list(landmark.columns)}"
    )


def extract_eye_data(
    behavior_data: BehaviorData,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Extract native timestamps, eye angles, and eye-level likelihoods.

    The angle for each eye requires a front and back landmark. Therefore,
    the eye-level likelihood is the minimum likelihood of the two points.
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

    reference_vector = np.array(
        [0.0, 1.0],
        dtype=float,
    )

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

    left_likelihood = sp.combine_likelihoods(
        extract_landmark_likelihood(
            left_front,
            "eye_left_front",
        ),
        extract_landmark_likelihood(
            left_back,
            "eye_left_back",
        ),
        method="min",
    )

    right_likelihood = sp.combine_likelihoods(
        extract_landmark_likelihood(
            right_front,
            "eye_right_front",
        ),
        extract_landmark_likelihood(
            right_back,
            "eye_right_back",
        ),
        method="min",
    )

    timestamps_ns = behavior_data.video_timestamps.timestamp.to_numpy(dtype=np.float64)

    number_of_samples = min(
        len(timestamps_ns),
        len(left_angle),
        len(right_angle),
        len(left_likelihood),
        len(right_likelihood),
    )

    if number_of_samples < 2:
        raise ValueError("Not enough aligned eye-tracking samples")

    timestamps_ns = timestamps_ns[:number_of_samples]

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
        raise ValueError("Not enough finite video timestamps")

    first_timestamp_ns = timestamps_ns[finite_timestamp_indices[0]]

    time_seconds = (timestamps_ns - first_timestamp_ns) * 1e-9

    return (
        time_seconds,
        left_angle,
        right_angle,
        left_likelihood,
        right_likelihood,
    )


# ============================================================================
# Classified event loading
# ============================================================================


def load_classified_events(
    classified_events_path: Path,
    fish_label: str,
    label_column: str | None,
) -> tuple[pd.DataFrame, str]:
    """
    Load a complete classified event table for one experiment.

    Required columns
    ----------------
    fish
    onset_time_s
    classification column
    """
    classified_events_path = classified_events_path.expanduser().resolve()

    if not classified_events_path.is_file():
        raise FileNotFoundError(
            "Classified event CSV not found: " f"{classified_events_path}"
        )

    events = pd.read_csv(classified_events_path)

    required_columns = {
        "fish",
        "onset_time_s",
    }

    missing_columns = required_columns.difference(events.columns)

    if missing_columns:
        raise KeyError(
            "The classified event table is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    if label_column is None:
        label_column = infer_label_column(events)

    if label_column not in events.columns:
        raise KeyError(
            f"Label column {label_column!r} was not found. "
            f"Available columns: {list(events.columns)}"
        )

    events = events.loc[events["fish"].astype(str) == str(fish_label)].copy()

    if events.empty:
        raise ValueError(f"No classified events were found for fish " f"{fish_label!r}")

    events["onset_time_s"] = pd.to_numeric(
        events["onset_time_s"],
        errors="coerce",
    )

    events = events.loc[np.isfinite(events["onset_time_s"])].copy()

    if events.empty:
        raise ValueError(f"No events for {fish_label!r} have a finite " "onset_time_s")

    events["_class_name"] = events[label_column].map(normalize_label)

    events = events.sort_values(
        "onset_time_s",
        kind="stable",
    ).reset_index(drop=True)

    return events, label_column


# ============================================================================
# Full 500 Hz trace preparation
# ============================================================================


def prepare_long_traces(
    time_seconds: np.ndarray,
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    left_likelihood: np.ndarray,
    right_likelihood: np.ndarray,
    mode: str,
    likelihood_threshold: float,
    metric_max_gap_ms: float,
    lowess_delta_threshold: float,
    lowess_anneal_samples: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Produce likelihood-masked and smoothed 500 Hz eye traces.
    """
    metric_config = sp.MetricConfig(
        fs=SAMPLING_RATE,
        likelihood_threshold=likelihood_threshold,
        max_interp_gap_s=metric_max_gap_ms / 1000.0,
        lowess_delta_threshold=lowess_delta_threshold,
        lowess_anneal_samples=lowess_anneal_samples,
    )

    finite_time = time_seconds[np.isfinite(time_seconds)]

    if finite_time.size < 2:
        raise ValueError("Not enough finite timestamps to prepare traces")

    start_time = float(np.min(finite_time))
    end_time = float(np.max(finite_time))

    left = sp.prepare_metric_trace(
        time=time_seconds,
        position=left_raw,
        likelihood=left_likelihood,
        mode=mode,
        config=metric_config,
        start_time=start_time,
        end_time=end_time,
    )

    right = sp.prepare_metric_trace(
        time=time_seconds,
        position=right_raw,
        likelihood=right_likelihood,
        mode=mode,
        config=metric_config,
        start_time=start_time,
        end_time=end_time,
    )

    if len(left["time"]) != len(right["time"]) or not np.allclose(
        left["time"],
        right["time"],
    ):
        raise RuntimeError("Left and right 500 Hz time bases do not match")

    return (
        left["time"],
        left["position"],
        right["position"],
        left["smoothed"],
        right["smoothed"],
    )


def find_behavior_file(
    directories: Directories,
    fish_label: str,
):
    """Find the experiment matching the requested fish label."""
    all_files = find_files(directories)

    exact_matches = [files for files in all_files if files.metadata.stem == fish_label]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        raise RuntimeError(f"Multiple experiments matched fish " f"{fish_label!r}")

    available = [files.metadata.stem for files in all_files]

    raise FileNotFoundError(
        f"No experiment matched fish {fish_label!r}. "
        f"Available labels include: {available[:20]}"
    )


# ============================================================================
# Timestamp-aware video reader
# ============================================================================


class TimestampedVideo:
    """
    Read a video using externally recorded frame timestamps.

    Frame times are expressed relative to the first finite timestamp,
    matching the eye trace and event time bases.
    """

    def __init__(
        self,
        video_path: Path,
        timestamps_ns: np.ndarray,
    ) -> None:
        self.video_path = Path(video_path).expanduser().resolve()

        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.capture = cv2.VideoCapture(str(self.video_path))

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.video_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.nominal_fps = float(self.capture.get(cv2.CAP_PROP_FPS))

        timestamps_ns = np.asarray(
            timestamps_ns,
            dtype=np.float64,
        ).ravel()

        finite_indices = np.flatnonzero(np.isfinite(timestamps_ns))

        if finite_indices.size < 2:
            self.close()
            raise ValueError("At least two finite frame timestamps are required")

        first_timestamp = timestamps_ns[finite_indices[0]]

        relative_times = (timestamps_ns - first_timestamp) * 1e-9

        self.frame_count = min(
            self.video_frame_count,
            len(relative_times),
        )

        if self.frame_count < 1:
            self.close()
            raise ValueError(
                "The video and timestamp sequence have no " "overlapping frames"
            )

        if self.video_frame_count != len(relative_times):
            print(
                "[warning] Video/timestamp length mismatch: "
                f"{self.video_frame_count:,} video frames, "
                f"{len(relative_times):,} timestamps. "
                f"Using the first {self.frame_count:,}."
            )

        self.frame_times = relative_times[: self.frame_count]

        self.valid_frame_indices = np.flatnonzero(np.isfinite(self.frame_times))

        if self.valid_frame_indices.size == 0:
            self.close()
            raise ValueError("No finite frame timestamps are available")

        self.valid_frame_times = self.frame_times[self.valid_frame_indices]

        # searchsorted requires a monotonic sequence.
        order = np.argsort(
            self.valid_frame_times,
            kind="stable",
        )

        self.valid_frame_times = self.valid_frame_times[order]
        self.valid_frame_indices = self.valid_frame_indices[order]

        self.current_frame_index = -1

    @property
    def start_time(self) -> float:
        """Time of the first valid frame."""
        return float(self.valid_frame_times[0])

    @property
    def end_time(self) -> float:
        """Time of the last valid frame."""
        return float(self.valid_frame_times[-1])

    def frame_index_at_time(
        self,
        target_time: float,
    ) -> int:
        """Return the video frame nearest to target_time."""
        insertion = int(
            np.searchsorted(
                self.valid_frame_times,
                target_time,
                side="left",
            )
        )

        if insertion <= 0:
            valid_position = 0

        elif insertion >= len(self.valid_frame_times):
            valid_position = len(self.valid_frame_times) - 1

        else:
            before = insertion - 1
            after = insertion

            before_distance = abs(target_time - self.valid_frame_times[before])
            after_distance = abs(self.valid_frame_times[after] - target_time)

            valid_position = before if before_distance <= after_distance else after

        return int(self.valid_frame_indices[valid_position])

    def time_at_frame(
        self,
        frame_index: int,
    ) -> float:
        """Return the relative timestamp for a video frame."""
        frame_index = int(
            np.clip(
                frame_index,
                0,
                self.frame_count - 1,
            )
        )

        frame_time = self.frame_times[frame_index]

        if np.isfinite(frame_time):
            return float(frame_time)

        nearest_valid_position = int(
            np.argmin(np.abs(self.valid_frame_indices - frame_index))
        )

        return float(self.valid_frame_times[nearest_valid_position])

    def read_frame(
        self,
        frame_index: int,
    ) -> np.ndarray:
        """Read a video frame and return it as RGB."""
        frame_index = int(
            np.clip(
                frame_index,
                0,
                self.frame_count - 1,
            )
        )

        expected_next = self.current_frame_index + 1

        if frame_index != expected_next:
            self.capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

        success, frame = self.capture.read()

        if not success or frame is None:
            # Retry after explicitly seeking.
            self.capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )
            success, frame = self.capture.read()

        if not success or frame is None:
            raise RuntimeError(
                f"Could not read frame {frame_index} " f"from {self.video_path}"
            )

        self.current_frame_index = frame_index

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

    def close(self) -> None:
        """Release the OpenCV video handle."""
        capture = getattr(self, "capture", None)

        if capture is not None:
            capture.release()
            self.capture = None

    def __del__(self) -> None:
        self.close()


# ============================================================================
# Interactive synchronized viewer
# ============================================================================


class LongTraceViewer:
    """Interactive synchronized video and eye-trace viewer."""

    def __init__(
        self,
        video_path: Path,
        frame_timestamps_ns: np.ndarray,
        native_time: np.ndarray,
        left_raw: np.ndarray,
        right_raw: np.ndarray,
        left_likelihood: np.ndarray,
        right_likelihood: np.ndarray,
        time_500: np.ndarray,
        left_500: np.ndarray,
        right_500: np.ndarray,
        left_smooth: np.ndarray,
        right_smooth: np.ndarray,
        events: pd.DataFrame,
        label_column: str,
        likelihood_threshold: float,
        fish_label: str,
        initial_window_s: float = 20.0,
        show_invalid_raw: bool = True,
        category_snippet_pre_ms: float = 50.0,
        category_snippet_post_ms: float = 250.0,
        category_snippet_linewidth: float = 3.0,
        show_event_lines: bool = True,
        show_event_labels: bool = True,
        playback_speed: float = 1.0,
        timer_interval_ms: int = 30,
    ) -> None:
        self.video = TimestampedVideo(
            video_path=video_path,
            timestamps_ns=frame_timestamps_ns,
        )

        self.native_time = np.asarray(
            native_time,
            dtype=float,
        )
        self.left_raw = np.asarray(
            left_raw,
            dtype=float,
        )
        self.right_raw = np.asarray(
            right_raw,
            dtype=float,
        )
        self.left_likelihood = np.asarray(
            left_likelihood,
            dtype=float,
        )
        self.right_likelihood = np.asarray(
            right_likelihood,
            dtype=float,
        )

        self.time_500 = np.asarray(
            time_500,
            dtype=float,
        )
        self.left_500 = np.asarray(
            left_500,
            dtype=float,
        )
        self.right_500 = np.asarray(
            right_500,
            dtype=float,
        )
        self.left_smooth = np.asarray(
            left_smooth,
            dtype=float,
        )
        self.right_smooth = np.asarray(
            right_smooth,
            dtype=float,
        )

        self.events = events.copy()
        self.label_column = label_column
        self.likelihood_threshold = float(likelihood_threshold)
        self.fish_label = fish_label

        self.show_invalid_raw = bool(show_invalid_raw)
        self.show_event_lines = bool(show_event_lines)
        self.show_event_labels = bool(show_event_labels)

        self.category_snippet_pre_s = category_snippet_pre_ms / 1000.0
        self.category_snippet_post_s = category_snippet_post_ms / 1000.0
        self.category_snippet_linewidth = float(category_snippet_linewidth)

        self.playback_speed = float(playback_speed)
        self.timer_interval_ms = int(timer_interval_ms)

        finite_trace_times = self.time_500[np.isfinite(self.time_500)]

        if finite_trace_times.size < 2:
            raise ValueError("The 500 Hz time base is empty")

        # Display only the period shared by the video and eye traces.
        self.recording_start = max(
            float(np.min(finite_trace_times)),
            self.video.start_time,
        )
        self.recording_end = min(
            float(np.max(finite_trace_times)),
            self.video.end_time,
        )

        if self.recording_end <= self.recording_start:
            raise ValueError(
                "The labeled video and eye traces do not " "overlap in time"
            )

        self.recording_duration = self.recording_end - self.recording_start

        minimum_window = min(
            0.5,
            self.recording_duration,
        )

        self.window_s = float(
            np.clip(
                initial_window_s,
                minimum_window,
                self.recording_duration,
            )
        )
        self.window_start = self.recording_start

        self.current_time = self.recording_start
        self.current_frame_index = self.video.frame_index_at_time(self.current_time)

        self.playing = False
        self.playback_anchor_wall: float | None = None
        self.playback_anchor_time: float | None = None

        self.event_artists: list[Any] = []
        self.cursor_artists: list[Any] = []

        self.figure = plt.figure(
            figsize=(16, 10),
            constrained_layout=True,
        )

        grid = GridSpec(
            3,
            1,
            figure=self.figure,
            height_ratios=[2.2, 1.0, 1.0],
        )

        self.video_axis = self.figure.add_subplot(grid[0, 0])
        self.left_axis = self.figure.add_subplot(grid[1, 0])
        self.right_axis = self.figure.add_subplot(
            grid[2, 0],
            sharex=self.left_axis,
        )

        self.trace_axes = [
            self.left_axis,
            self.right_axis,
        ]

        self._initialize_video()
        self._initialize_trace_lines()
        self._initialize_cursor()

        self.figure.canvas.mpl_connect(
            "key_press_event",
            self._on_key,
        )
        self.figure.canvas.mpl_connect(
            "scroll_event",
            self._on_scroll,
        )
        self.figure.canvas.mpl_connect(
            "button_press_event",
            self._on_click,
        )
        self.figure.canvas.mpl_connect(
            "close_event",
            self._on_close,
        )

        self.timer = self.figure.canvas.new_timer(interval=self.timer_interval_ms)
        self.timer.add_callback(self._on_timer)
        self.timer.start()

        self.update_window()
        self.seek_time(
            self.current_time,
            scroll_window=False,
        )

    def _initialize_video(self) -> None:
        """Create the video image artist."""
        first_frame = self.video.read_frame(self.current_frame_index)

        self.video_image = self.video_axis.imshow(
            first_frame,
            interpolation="nearest",
        )

        self.video_axis.set_axis_off()
        self.video_title = self.video_axis.set_title(
            "",
            fontsize=11,
        )

    def _initialize_trace_lines(self) -> None:
        """Create persistent trace-line artists."""
        (self.left_invalid_line,) = self.left_axis.plot(
            [],
            [],
            color=INVALID_RAW_COLOR,
            linewidth=0.7,
            alpha=0.55,
            label="Low-likelihood raw",
            zorder=1,
        )

        (self.left_raw_line,) = self.left_axis.plot(
            [],
            [],
            color=LEFT_RAW_COLOR,
            linewidth=0.7,
            alpha=0.30,
            label="Camera-rate raw",
            zorder=2,
        )

        (self.left_resampled_line,) = self.left_axis.plot(
            [],
            [],
            color=LEFT_RESAMPLED_COLOR,
            linewidth=0.9,
            alpha=0.55,
            label="500 Hz masked/resampled",
            zorder=3,
        )

        (self.left_smooth_line,) = self.left_axis.plot(
            [],
            [],
            color=LEFT_SMOOTH_COLOR,
            linewidth=1.5,
            alpha=0.80,
            label="500 Hz smoothed",
            zorder=4,
        )

        (self.right_invalid_line,) = self.right_axis.plot(
            [],
            [],
            color=INVALID_RAW_COLOR,
            linewidth=0.7,
            alpha=0.55,
            label="Low-likelihood raw",
            zorder=1,
        )

        (self.right_raw_line,) = self.right_axis.plot(
            [],
            [],
            color=RIGHT_RAW_COLOR,
            linewidth=0.7,
            alpha=0.30,
            label="Camera-rate raw",
            zorder=2,
        )

        (self.right_resampled_line,) = self.right_axis.plot(
            [],
            [],
            color=RIGHT_RESAMPLED_COLOR,
            linewidth=0.9,
            alpha=0.55,
            label="500 Hz masked/resampled",
            zorder=3,
        )

        (self.right_smooth_line,) = self.right_axis.plot(
            [],
            [],
            color=RIGHT_SMOOTH_COLOR,
            linewidth=1.5,
            alpha=0.80,
            label="500 Hz smoothed",
            zorder=4,
        )

        self.left_axis.set_ylabel("Left eye position (degrees)")
        self.right_axis.set_ylabel("Right eye position (degrees)")
        self.right_axis.set_xlabel("Time (s)")

        for axis in self.trace_axes:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(direction="out")
            axis.grid(
                axis="x",
                color="0.90",
                linewidth=0.5,
                zorder=0,
            )

        self.left_axis.legend(
            loc="upper right",
            frameon=False,
            fontsize=8,
        )
        self.right_axis.legend(
            loc="upper right",
            frameon=False,
            fontsize=8,
        )

    def _initialize_cursor(self) -> None:
        """Create a cursor marking the current video time."""
        for axis in self.trace_axes:
            cursor = axis.axvline(
                self.current_time,
                color="black",
                linewidth=2.0,
                alpha=0.9,
                zorder=20,
            )
            self.cursor_artists.append(cursor)

    @staticmethod
    def _time_mask(
        time_values: np.ndarray,
        start: float,
        stop: float,
    ) -> np.ndarray:
        """Return a finite mask for a time interval."""
        return np.isfinite(time_values) & (time_values >= start) & (time_values <= stop)

    def _clear_event_artists(self) -> None:
        """Remove category overlays from the previous window."""
        for artist in self.event_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass

        self.event_artists.clear()

    def _set_window_data(
        self,
        start: float,
        stop: float,
    ) -> None:
        """Update trace lines for the currently visible interval."""
        native_mask = self._time_mask(
            self.native_time,
            start,
            stop,
        )

        mask_500 = self._time_mask(
            self.time_500,
            start,
            stop,
        )

        native_time = self.native_time[native_mask]

        left_raw = self.left_raw[native_mask]
        right_raw = self.right_raw[native_mask]

        left_likelihood = self.left_likelihood[native_mask]
        right_likelihood = self.right_likelihood[native_mask]

        left_valid = (
            np.isfinite(left_raw)
            & np.isfinite(left_likelihood)
            & (left_likelihood >= self.likelihood_threshold)
        )

        right_valid = (
            np.isfinite(right_raw)
            & np.isfinite(right_likelihood)
            & (right_likelihood >= self.likelihood_threshold)
        )

        self.left_raw_line.set_data(
            native_time,
            np.where(
                left_valid,
                left_raw,
                np.nan,
            ),
        )

        self.right_raw_line.set_data(
            native_time,
            np.where(
                right_valid,
                right_raw,
                np.nan,
            ),
        )

        if self.show_invalid_raw:
            self.left_invalid_line.set_data(
                native_time,
                np.where(
                    ~left_valid & np.isfinite(left_raw),
                    left_raw,
                    np.nan,
                ),
            )

            self.right_invalid_line.set_data(
                native_time,
                np.where(
                    ~right_valid & np.isfinite(right_raw),
                    right_raw,
                    np.nan,
                ),
            )
        else:
            self.left_invalid_line.set_data([], [])
            self.right_invalid_line.set_data([], [])

        visible_time_500 = self.time_500[mask_500]

        self.left_resampled_line.set_data(
            visible_time_500,
            self.left_500[mask_500],
        )

        self.right_resampled_line.set_data(
            visible_time_500,
            self.right_500[mask_500],
        )

        self.left_smooth_line.set_data(
            visible_time_500,
            self.left_smooth[mask_500],
        )

        self.right_smooth_line.set_data(
            visible_time_500,
            self.right_smooth[mask_500],
        )

    def _plot_events(
        self,
        start: float,
        stop: float,
    ) -> None:
        """
        Draw category-colored smoothed snippets, onset lines, and labels.
        """
        relevant_events = self.events.loc[
            (self.events["onset_time_s"] + self.category_snippet_post_s >= start)
            & (self.events["onset_time_s"] - self.category_snippet_pre_s <= stop)
        ]

        text_transform = blended_transform_factory(
            self.left_axis.transData,
            self.left_axis.transAxes,
        )

        visible_label_number = 0

        for _, event in relevant_events.iterrows():
            event_time = float(event["onset_time_s"])

            class_name = normalize_label(event[self.label_column])
            color = class_color(class_name)

            snippet_start = event_time - self.category_snippet_pre_s
            snippet_stop = event_time + self.category_snippet_post_s

            snippet_mask = (
                np.isfinite(self.time_500)
                & (self.time_500 >= snippet_start)
                & (self.time_500 <= snippet_stop)
                & (self.time_500 >= start)
                & (self.time_500 <= stop)
            )

            if np.any(snippet_mask):
                (left_overlay,) = self.left_axis.plot(
                    self.time_500[snippet_mask],
                    self.left_smooth[snippet_mask],
                    color=color,
                    linewidth=self.category_snippet_linewidth,
                    alpha=1.0,
                    solid_capstyle="round",
                    label="_nolegend_",
                    zorder=8,
                )

                (right_overlay,) = self.right_axis.plot(
                    self.time_500[snippet_mask],
                    self.right_smooth[snippet_mask],
                    color=color,
                    linewidth=self.category_snippet_linewidth,
                    alpha=1.0,
                    solid_capstyle="round",
                    label="_nolegend_",
                    zorder=8,
                )

                self.event_artists.extend(
                    [
                        left_overlay,
                        right_overlay,
                    ]
                )

            if not start <= event_time <= stop:
                continue

            if self.show_event_lines:
                for axis in self.trace_axes:
                    onset_line = axis.axvline(
                        event_time,
                        color=color,
                        linestyle="--",
                        linewidth=1.1,
                        alpha=0.8,
                        zorder=7,
                    )

                    self.event_artists.append(onset_line)

            if self.show_event_labels:
                text_height = 0.98 - 0.08 * (visible_label_number % 3)
                visible_label_number += 1

                text_artist = self.left_axis.text(
                    event_time,
                    text_height,
                    class_name,
                    transform=text_transform,
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    rotation=90,
                    horizontalalignment="right",
                    verticalalignment="top",
                    clip_on=True,
                    zorder=9,
                )

                self.event_artists.append(text_artist)

    def _autoscale_y(
        self,
        start: float,
        stop: float,
    ) -> None:
        """Set robust y-limits for the visible trace interval."""
        mask_500 = self._time_mask(
            self.time_500,
            start,
            stop,
        )

        for axis, resampled, smoothed in (
            (
                self.left_axis,
                self.left_500,
                self.left_smooth,
            ),
            (
                self.right_axis,
                self.right_500,
                self.right_smooth,
            ),
        ):
            values = np.concatenate(
                (
                    resampled[mask_500],
                    smoothed[mask_500],
                )
            )

            values = values[np.isfinite(values)]

            if values.size == 0:
                axis.set_ylim(-1.0, 1.0)
                continue

            lower, upper = np.percentile(
                values,
                [0.5, 99.5],
            )

            if not (np.isfinite(lower) and np.isfinite(upper)):
                axis.set_ylim(-1.0, 1.0)
                continue

            if lower == upper:
                lower -= 1.0
                upper += 1.0

            padding = 0.10 * (upper - lower)

            axis.set_ylim(
                lower - padding,
                upper + padding,
            )

    def _set_cursor(
        self,
        time_seconds: float,
    ) -> None:
        """Move the current-video-time cursor."""
        for cursor in self.cursor_artists:
            cursor.set_xdata([time_seconds, time_seconds])

    def _ensure_time_visible(
        self,
        target_time: float,
    ) -> bool:
        """
        Scroll the trace window when video time approaches either edge.

        Returns True if the window was changed.
        """
        left_margin = self.window_start + 0.15 * self.window_s
        right_margin = self.window_start + 0.85 * self.window_s

        if target_time < left_margin or target_time > right_margin:
            self.window_start = target_time - 0.5 * self.window_s
            return True

        return False

    def update_window(self) -> None:
        """Redraw the visible trace interval."""
        maximum_start = max(
            self.recording_start,
            self.recording_end - self.window_s,
        )

        self.window_start = float(
            np.clip(
                self.window_start,
                self.recording_start,
                maximum_start,
            )
        )

        window_stop = min(
            self.window_start + self.window_s,
            self.recording_end,
        )

        self._set_window_data(
            self.window_start,
            window_stop,
        )

        self._clear_event_artists()

        self._plot_events(
            self.window_start,
            window_stop,
        )

        for axis in self.trace_axes:
            axis.set_xlim(
                self.window_start,
                window_stop,
            )

        self._autoscale_y(
            self.window_start,
            window_stop,
        )

        self._set_cursor(self.current_time)

        visible_event_count = int(
            np.sum(
                (self.events["onset_time_s"] >= self.window_start)
                & (self.events["onset_time_s"] <= window_stop)
            )
        )

        self.figure.suptitle(
            f"{self.fish_label} — "
            f"{self.window_start:.2f} to "
            f"{window_stop:.2f} s — "
            f"{visible_event_count} classified events\n"
            "Space play/pause | J/L frame | "
            "Shift+J/L 10 frames | Left/Right pan | "
            "+/- zoom | 1-4 speed | click trace to seek"
        )

        self.figure.canvas.draw_idle()

    def _update_video_title(self) -> None:
        """Update the video panel title."""
        state = "PLAYING" if self.playing else "PAUSED"

        self.video_title.set_text(
            f"{self.fish_label} — "
            f"frame {self.current_frame_index:,}/"
            f"{self.video.frame_count - 1:,} — "
            f"{self.current_time:.3f} s — "
            f"{state} ({self.playback_speed:g}x)"
        )

    def seek_frame(
        self,
        frame_index: int,
        scroll_window: bool = True,
    ) -> None:
        """Display a frame and synchronize the trace cursor."""
        frame_index = int(
            np.clip(
                frame_index,
                0,
                self.video.frame_count - 1,
            )
        )

        frame = self.video.read_frame(frame_index)
        frame_time = self.video.time_at_frame(frame_index)

        self.current_frame_index = frame_index
        self.current_time = frame_time

        self.video_image.set_data(frame)
        self._update_video_title()

        window_changed = False

        if scroll_window:
            window_changed = self._ensure_time_visible(frame_time)

        if window_changed:
            self.update_window()
        else:
            self._set_cursor(frame_time)
            self.figure.canvas.draw_idle()

    def seek_time(
        self,
        target_time: float,
        scroll_window: bool = True,
    ) -> None:
        """Seek to the video frame nearest a trace time."""
        target_time = float(
            np.clip(
                target_time,
                self.recording_start,
                self.recording_end,
            )
        )

        frame_index = self.video.frame_index_at_time(target_time)

        self.seek_frame(
            frame_index,
            scroll_window=scroll_window,
        )

    def _reset_playback_anchor(self) -> None:
        """Reset wall-clock anchoring for playback."""
        self.playback_anchor_wall = time.perf_counter()
        self.playback_anchor_time = self.current_time

    def _set_playing(
        self,
        playing: bool,
    ) -> None:
        """Start or pause playback."""
        self.playing = bool(playing)

        if self.playing:
            self._reset_playback_anchor()
        else:
            self.playback_anchor_wall = None
            self.playback_anchor_time = None

        self._update_video_title()
        self.figure.canvas.draw_idle()

    def _toggle_playback(self) -> None:
        """Toggle playback."""
        self._set_playing(not self.playing)

    def _set_playback_speed(
        self,
        speed: float,
    ) -> None:
        """Set playback speed."""
        self.playback_speed = float(speed)

        if self.playing:
            self._reset_playback_anchor()

        self._update_video_title()
        self.figure.canvas.draw_idle()

    def _on_timer(self) -> None:
        """Advance playback using elapsed wall-clock time."""
        if not self.playing:
            return

        if self.playback_anchor_wall is None or self.playback_anchor_time is None:
            self._reset_playback_anchor()
            return

        elapsed = time.perf_counter() - self.playback_anchor_wall

        target_time = self.playback_anchor_time + elapsed * self.playback_speed

        if target_time >= self.recording_end:
            self.seek_time(self.recording_end)
            self._set_playing(False)
            return

        target_frame = self.video.frame_index_at_time(target_time)

        if target_frame != self.current_frame_index:
            self.seek_frame(target_frame)

    def _pan_window(
        self,
        fraction: float,
    ) -> None:
        """Pan the trace window without changing video time."""
        self.window_start += fraction * self.window_s
        self.update_window()

    def _zoom(
        self,
        factor: float,
    ) -> None:
        """Zoom around the current video time."""
        minimum_window = min(
            0.5,
            self.recording_duration,
        )

        self.window_s = float(
            np.clip(
                self.window_s * factor,
                minimum_window,
                self.recording_duration,
            )
        )

        self.window_start = self.current_time - 0.5 * self.window_s

        self.update_window()

    def _on_key(
        self,
        event: Any,
    ) -> None:
        """Handle keyboard controls."""
        if event.key in (" ", "space"):
            self._toggle_playback()

        elif event.key in ("j", ","):
            self._set_playing(False)
            self.seek_frame(self.current_frame_index - 1)

        elif event.key in ("l", "."):
            self._set_playing(False)
            self.seek_frame(self.current_frame_index + 1)

        elif event.key == "shift+j":
            self._set_playing(False)
            self.seek_frame(self.current_frame_index - 10)

        elif event.key == "shift+l":
            self._set_playing(False)
            self.seek_frame(self.current_frame_index + 10)

        elif event.key in ("left", "a"):
            self._pan_window(-0.5)

        elif event.key in ("right", "d"):
            self._pan_window(0.5)

        elif event.key in ("+", "="):
            self._zoom(0.5)

        elif event.key in ("-", "_"):
            self._zoom(2.0)

        elif event.key == "home":
            self._set_playing(False)
            self.seek_time(self.recording_start)

        elif event.key == "end":
            self._set_playing(False)
            self.seek_time(self.recording_end)

        elif event.key == "1":
            self._set_playback_speed(0.25)

        elif event.key == "2":
            self._set_playback_speed(0.5)

        elif event.key == "3":
            self._set_playback_speed(1.0)

        elif event.key == "4":
            self._set_playback_speed(2.0)

        elif event.key in ("q", "escape"):
            plt.close(self.figure)

    def _on_scroll(
        self,
        event: Any,
    ) -> None:
        """Use the mouse wheel to pan the trace."""
        if event.button == "up":
            self._pan_window(-0.20)

        elif event.button == "down":
            self._pan_window(0.20)

    def _print_nearest_event(
        self,
        selected_time: float,
    ) -> None:
        """Print the classified event nearest selected_time."""
        if self.events.empty:
            return

        event_times = self.events["onset_time_s"].to_numpy(dtype=float)

        nearest_index = int(np.argmin(np.abs(event_times - selected_time)))

        selected = self.events.iloc[nearest_index]

        print()
        print("Nearest classified event")
        print("------------------------")
        print(
            "event_id:",
            selected.get(
                "event_id",
                nearest_index,
            ),
        )
        print(
            "fish:",
            selected.get(
                "fish",
                self.fish_label,
            ),
        )
        print(
            "time:",
            f"{selected['onset_time_s']:.6f} s",
        )
        print(
            "class:",
            normalize_label(selected[self.label_column]),
        )

        if "bconv_candidate" in selected.index:
            print(
                "BConv candidate:",
                selected["bconv_candidate"],
            )

        if "bconv_side" in selected.index:
            print(
                "BConv side:",
                selected["bconv_side"],
            )

        for metric_name in getattr(
            sp,
            "METRIC_NAMES",
            [],
        ):
            if metric_name in selected.index:
                print(f"{metric_name}: " f"{selected[metric_name]}")

    def _on_click(
        self,
        event: Any,
    ) -> None:
        """Seek the video when either eye trace is clicked."""
        if event.inaxes not in self.trace_axes or event.xdata is None:
            return

        selected_time = float(event.xdata)

        self._set_playing(False)
        self.seek_time(selected_time)

        if event.dblclick:
            self._print_nearest_event(selected_time)

    def _on_close(
        self,
        _event: Any,
    ) -> None:
        """Release resources when the figure closes."""
        self.playing = False

        if hasattr(self, "timer"):
            self.timer.stop()

        self.video.close()

    def show(self) -> None:
        """Open the interactive viewer."""
        plt.show()


# ============================================================================
# Command-line interface
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Display a synchronized Lightning Pose labeled video, "
            "raw eye traces, smoothed 500 Hz traces, and "
            "category-colored classified saccade snippets."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder.",
    )

    parser.add_argument(
        "classified_events",
        type=Path,
        help=(
            "Complete classified-event CSV containing fish, "
            "onset_time_s, and a classification column."
        ),
    )

    parser.add_argument(
        "--fish",
        required=True,
        help=(
            "Experiment/fish label matching both the CSV fish "
            "column and metadata filename stem."
        ),
    )

    parser.add_argument(
        "--labeled-video",
        type=Path,
        required=True,
        help=(
            "Path to the Lightning Pose labeled video. Its "
            "frames must correspond to video_timestamps."
        ),
    )

    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Classification column in the classified-event CSV. "
            "If omitted, it is inferred."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=("tethered", "freeswim"),
        default="freeswim",
        help="Recording mode. Default: %(default)s",
    )

    parser.add_argument(
        "--window-s",
        type=float,
        default=20.0,
        help=("Initial visible trace window in seconds. " "Default: %(default)s"),
    )

    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.9,
        help=(
            "Samples below this Lightning Pose likelihood are "
            "treated as missing. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--metric-max-gap-ms",
        type=float,
        default=20.0,
        help=(
            "Maximum low-confidence gap interpolated in the "
            "500 Hz traces. Default: %(default)s ms"
        ),
    )

    parser.add_argument(
        "--lowess-delta-threshold",
        type=float,
        default=0.5,
        help=("Step-response threshold used by custom LOWESS. " "Default: %(default)s"),
    )

    parser.add_argument(
        "--lowess-anneal-samples",
        type=int,
        default=50,
        help=(
            "Custom LOWESS annealing distance in 500 Hz samples. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--category-snippet-pre-ms",
        type=float,
        default=50.0,
        help=(
            "Duration before each event colored using its "
            "category. Default: %(default)s ms"
        ),
    )

    parser.add_argument(
        "--category-snippet-post-ms",
        type=float,
        default=250.0,
        help=(
            "Duration after each event colored using its "
            "category. Default: %(default)s ms"
        ),
    )

    parser.add_argument(
        "--category-snippet-linewidth",
        type=float,
        default=3.0,
        help=(
            "Line width of category-colored smoothed snippets. " "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help=("Initial video playback speed. " "Default: %(default)s"),
    )

    parser.add_argument(
        "--timer-interval-ms",
        type=int,
        default=30,
        help=("GUI playback timer interval in milliseconds. " "Default: %(default)s"),
    )

    parser.add_argument(
        "--hide-invalid-raw",
        action="store_true",
        help=(
            "Hide native samples below the likelihood threshold. "
            "They are shown in gray by default."
        ),
    )

    parser.add_argument(
        "--hide-event-lines",
        action="store_true",
        help="Hide category-colored event-onset lines.",
    )

    parser.add_argument(
        "--hide-event-labels",
        action="store_true",
        help="Hide event category labels above the traces.",
    )

    # BehaviorScreen directory settings.
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

    return parser


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate command-line arguments."""
    if not 0.0 <= args.likelihood_threshold <= 1.0:
        raise ValueError("--likelihood-threshold must be between 0 and 1")

    if args.window_s <= 0:
        raise ValueError("--window-s must be positive")

    if args.metric_max_gap_ms < 0:
        raise ValueError("--metric-max-gap-ms must be non-negative")

    if args.lowess_delta_threshold < 0:
        raise ValueError("--lowess-delta-threshold must be non-negative")

    if args.lowess_anneal_samples < 1:
        raise ValueError("--lowess-anneal-samples must be at least 1")

    if args.category_snippet_pre_ms < 0:
        raise ValueError("--category-snippet-pre-ms must be non-negative")

    if args.category_snippet_post_ms < 0:
        raise ValueError("--category-snippet-post-ms must be non-negative")

    if args.category_snippet_linewidth <= 0:
        raise ValueError("--category-snippet-linewidth must be positive")

    if args.playback_speed <= 0:
        raise ValueError("--playback-speed must be positive")

    if args.timer_interval_ms < 1:
        raise ValueError("--timer-interval-ms must be at least 1")

    if not args.classified_events.expanduser().is_file():
        raise FileNotFoundError(
            "Classified event CSV not found: " f"{args.classified_events}"
        )

    if not args.labeled_video.expanduser().is_file():
        raise FileNotFoundError(f"Labeled video not found: " f"{args.labeled_video}")


def main() -> None:
    """Load one experiment and start the synchronized viewer."""
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

    files = find_behavior_file(
        directories,
        args.fish,
    )

    print(f"Loading experiment: {args.fish}")
    behavior_data = load_data(files)

    (
        time_seconds,
        left_raw,
        right_raw,
        left_likelihood,
        right_likelihood,
    ) = extract_eye_data(behavior_data)

    events, label_column = load_classified_events(
        classified_events_path=args.classified_events,
        fish_label=args.fish,
        label_column=args.label_column,
    )

    print(f"Loaded {len(events):,} classified events " f"for {args.fish}")

    print(f"Using label column: {label_column}")
    print("Classes:")

    for class_name, count in events["_class_name"].value_counts().sort_index().items():
        print(f"  {class_name}: {count:,}")

    (
        time_500,
        left_500,
        right_500,
        left_smooth,
        right_smooth,
    ) = prepare_long_traces(
        time_seconds=time_seconds,
        left_raw=left_raw,
        right_raw=right_raw,
        left_likelihood=left_likelihood,
        right_likelihood=right_likelihood,
        mode=args.mode,
        likelihood_threshold=args.likelihood_threshold,
        metric_max_gap_ms=args.metric_max_gap_ms,
        lowess_delta_threshold=(args.lowess_delta_threshold),
        lowess_anneal_samples=(args.lowess_anneal_samples),
    )

    # The labeled video is synchronized to these frame timestamps.
    frame_timestamps_ns = behavior_data.video_timestamps.timestamp.to_numpy(
        dtype=np.float64
    )

    viewer = LongTraceViewer(
        video_path=args.labeled_video,
        frame_timestamps_ns=frame_timestamps_ns,
        native_time=time_seconds,
        left_raw=left_raw,
        right_raw=right_raw,
        left_likelihood=left_likelihood,
        right_likelihood=right_likelihood,
        time_500=time_500,
        left_500=left_500,
        right_500=right_500,
        left_smooth=left_smooth,
        right_smooth=right_smooth,
        events=events,
        label_column=label_column,
        likelihood_threshold=args.likelihood_threshold,
        fish_label=args.fish,
        initial_window_s=args.window_s,
        show_invalid_raw=not args.hide_invalid_raw,
        category_snippet_pre_ms=(args.category_snippet_pre_ms),
        category_snippet_post_ms=(args.category_snippet_post_ms),
        category_snippet_linewidth=(args.category_snippet_linewidth),
        show_event_lines=not args.hide_event_lines,
        show_event_labels=not args.hide_event_labels,
        playback_speed=args.playback_speed,
        timer_interval_ms=args.timer_interval_ms,
    )

    viewer.show()


if __name__ == "__main__":
    main()
