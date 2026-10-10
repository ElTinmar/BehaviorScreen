#!/usr/bin/env python3
"""
Export fish-centered, heading-aligned average videos for bout and saccade
categories.

Frame rates
-----------
--output-fps
    Temporal sampling rate used to construct the aggregate frames. For
    recordings captured at 120 FPS, use --output-fps 120 to retain the
    native temporal resolution.

--playback-fps
    Frame rate written into the output MP4 metadata. This affects only
    playback speed, not source-frame selection or the event-relative time
    axis.

For example:

    --output-fps 120 --playback-fps 5

retains 120 temporal samples per second of source time and plays the result
24 times slower than real time.

Default grouping
----------------
Bouts:
    category:  category
    direction: sign
    onset:     frame_start

Saccades:
    category:  saccade_category_name
    direction: event_direction_name
    onset:     tracking_frame

For every source frame, the image is translated and rotated so that:

1. Swim_Bladder is at the center of the output image.
2. The Swim_Bladder -> Head vector points upward.

Aggregation
-----------
mean
    Streaming, memory-efficient, and normally recommended.

median / mode
    Exact uint8 median or mode. Transformed clips are temporarily written
    to disk and reduced in chunks. These methods can require considerable
    temporary disk space and are substantially slower than mean.

Example
-------
python -m BehaviorScreen.export_average_event_videos \
    /media/martin/DATA_18TB/Screen/WT/vehicle \
    --bouts-csv bouts.csv \
    --saccades-csv saccades_augmented.csv \
    --output-dir average_event_videos \
    --statistic mean \
    --pre-ms 300 \
    --post-ms 600 \
    --output-fps 120 \
    --playback-fps 5 \
    --crop-width 240 \
    --crop-height 320
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.load import (
    BehaviorFiles,
    Directories,
    find_files,
    load_lightning_pose,
)

# ============================================================================
# General utilities
# ============================================================================


def parse_columns(value: str) -> list[str]:
    """Parse a comma-separated list of column names."""
    columns = [column.strip() for column in value.split(",") if column.strip()]

    if not columns:
        raise argparse.ArgumentTypeError("At least one column must be supplied.")

    return columns


def safe_filename(value: object) -> str:
    """Convert a category label to a filesystem-safe string."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text)
    text = text.strip("._")

    return text or "unnamed"


def group_label(
    group_columns: Sequence[str],
    group_key: object,
) -> tuple[str, dict[str, object]]:
    """Create a filename label and metadata from a groupby key."""
    if len(group_columns) == 1:
        values = (group_key,)
    else:
        values = tuple(group_key)  # type: ignore[arg-type]

    metadata = dict(zip(group_columns, values))

    label = "__".join(
        f"{safe_filename(column)}-{safe_filename(value)}"
        for column, value in metadata.items()
    )

    return label, metadata


def find_recording_column(events: pd.DataFrame) -> str:
    """Find the column identifying the source recording."""
    for column in ("file", "fish", "recording"):
        if column in events.columns:
            return column

    raise ValueError(
        "The event table needs one of these recording columns: "
        "'file', 'fish', or 'recording'."
    )


def normalize_recording_name(value: object) -> str:
    """Normalize a recording identifier."""
    if pd.isna(value):
        return ""

    return Path(str(value)).stem


def read_pixels_per_mm(metadata_path: Path) -> float:
    """Read spatial calibration from a metadata file."""
    with metadata_path.open("r") as input_file:
        metadata = json.load(input_file)

    return float(metadata["calibration"]["pix_per_mm"])


# ============================================================================
# Video access
# ============================================================================


class VideoReader:
    """
    OpenCV video reader with sequential-frame optimization.

    Consecutive frames are read normally. Small forward gaps are traversed
    with grab(), avoiding an expensive random seek for every requested
    frame. Backward and large forward jumps use CAP_PROP_POS_FRAMES.
    """

    def __init__(
        self,
        path: Path,
        maximum_grab_gap: int = 120,
    ):
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(path))

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))

        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"Invalid frame rate for {path}: {self.fps}")

        self.maximum_grab_gap = int(maximum_grab_gap)
        self.next_frame_index: int | None = None

        self.cached_frame_index: int | None = None
        self.cached_frame: np.ndarray | None = None

    def read(self, frame_index: int) -> np.ndarray | None:
        """Read one BGR frame."""
        frame_index = int(frame_index)

        if frame_index < 0 or frame_index >= self.frame_count:
            return None

        # Reuse a frame when multiple output times map to the same source
        # frame.
        if self.cached_frame_index == frame_index and self.cached_frame is not None:
            return self.cached_frame

        if self.next_frame_index is None:
            use_forward_decode = False
        else:
            gap = frame_index - self.next_frame_index
            use_forward_decode = gap >= 0 and gap <= self.maximum_grab_gap

        if use_forward_decode:
            frames_to_skip = frame_index - self.next_frame_index

            for _ in range(frames_to_skip):
                if not self.capture.grab():
                    self.next_frame_index = None
                    self.cached_frame_index = None
                    self.cached_frame = None
                    return None

        elif self.next_frame_index != frame_index:
            self.capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

        success, frame = self.capture.read()

        if not success or frame is None:
            self.next_frame_index = None
            self.cached_frame_index = None
            self.cached_frame = None
            return None

        self.next_frame_index = frame_index + 1

        if frame.ndim == 2:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )

        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR,
            )

        self.cached_frame_index = frame_index
        self.cached_frame = frame

        return frame

    def close(self) -> None:
        """Close the video."""
        self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ============================================================================
# Pose extraction
# ============================================================================


def extract_pose_point(
    tracking: pd.DataFrame,
    point_name: str,
    likelihood_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract xy coordinates and a validity mask for one pose point."""
    available_points = tracking.columns.get_level_values(0)

    if point_name not in available_points:
        available = sorted(set(available_points))

        raise KeyError(
            f"Pose point {point_name!r} was not found. "
            f"Available points: {available}"
        )

    point = tracking[point_name]

    if "x" not in point or "y" not in point:
        raise KeyError(f"Pose point {point_name!r} does not contain x and y.")

    xy = point[["x", "y"]].to_numpy(dtype=np.float32)

    valid = np.all(
        np.isfinite(xy),
        axis=1,
    )

    likelihood_column = next(
        (
            column
            for column in (
                "likelihood",
                "confidence",
                "score",
                "probability",
            )
            if column in point.columns
        ),
        None,
    )

    if likelihood_column is not None:
        likelihood = point[likelihood_column].to_numpy(dtype=float)

        valid &= np.isfinite(likelihood) & (likelihood >= likelihood_threshold)

    return xy, valid


# ============================================================================
# Frame transformation
# ============================================================================


def fish_aligned_frame(
    frame: np.ndarray,
    center_xy: np.ndarray,
    head_xy: np.ndarray,
    output_width: int,
    output_height: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Center and rotate a video frame around the fish.

    The center point is moved to the center of the output image. The
    center-to-head vector is rotated to point toward the top of the image.
    """
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])

    delta_x = float(head_xy[0] - center_x)
    delta_y = float(head_xy[1] - center_y)

    length = np.hypot(delta_x, delta_y)

    if not np.isfinite(length) or length < 1e-6:
        return None

    # Image y coordinates increase downward. This rotation moves the
    # center-to-head direction toward negative y.
    heading_degrees = np.degrees(np.arctan2(delta_y, delta_x))
    rotation_degrees = heading_degrees + 90.0

    matrix = cv2.getRotationMatrix2D(
        center=(center_x, center_y),
        angle=rotation_degrees,
        scale=scale,
    )

    output_center_x = (output_width - 1) / 2.0
    output_center_y = (output_height - 1) / 2.0

    matrix[0, 2] += output_center_x - center_x
    matrix[1, 2] += output_center_y - center_y

    transformed = cv2.warpAffine(
        frame,
        matrix,
        dsize=(output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    source_mask = np.ones(
        frame.shape[:2],
        dtype=np.uint8,
    )

    valid_mask = cv2.warpAffine(
        source_mask,
        matrix,
        dsize=(output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    return transformed, valid_mask


def extract_aligned_clip(
    event_frame: int,
    reader: VideoReader,
    center_xy: np.ndarray,
    head_xy: np.ndarray,
    center_valid: np.ndarray,
    head_valid: np.ndarray,
    time_offsets_s: np.ndarray,
    output_width: int,
    output_height: int,
    scale: float,
    stabilization: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract one onset-aligned, fish-centered video clip.

    stabilization="frame"
        Recenter and rerotate every frame using that frame's pose.

    stabilization="onset"
        Use the onset-frame center and heading for the complete clip.
    """
    number_of_output_frames = len(time_offsets_s)

    clip = np.zeros(
        (
            number_of_output_frames,
            output_height,
            output_width,
            3,
        ),
        dtype=np.uint8,
    )

    valid_mask = np.zeros(
        (
            number_of_output_frames,
            output_height,
            output_width,
        ),
        dtype=bool,
    )

    if stabilization == "onset":
        if (
            event_frame < 0
            or event_frame >= len(center_xy)
            or event_frame >= len(head_xy)
            or not center_valid[event_frame]
            or not head_valid[event_frame]
        ):
            return None

        fixed_center = center_xy[event_frame]
        fixed_head = head_xy[event_frame]

    else:
        fixed_center = None
        fixed_head = None

    # time_offsets_s is defined by --output-fps. reader.fps is the actual
    # source-video frame rate.
    source_frames = np.rint(event_frame + time_offsets_s * reader.fps).astype(np.int64)

    for output_index, source_frame_value in enumerate(source_frames):
        source_frame = int(source_frame_value)

        if (
            source_frame < 0
            or source_frame >= reader.frame_count
            or source_frame >= len(center_xy)
            or source_frame >= len(head_xy)
        ):
            continue

        if stabilization == "frame":
            if not center_valid[source_frame] or not head_valid[source_frame]:
                continue

            current_center = center_xy[source_frame]
            current_head = head_xy[source_frame]

        else:
            if fixed_center is None or fixed_head is None:
                continue

            current_center = fixed_center
            current_head = fixed_head

        frame = reader.read(source_frame)

        if frame is None:
            continue

        result = fish_aligned_frame(
            frame=frame,
            center_xy=current_center,
            head_xy=current_head,
            output_width=output_width,
            output_height=output_height,
            scale=scale,
        )

        if result is None:
            continue

        transformed, transformed_mask = result

        clip[output_index] = transformed
        valid_mask[output_index] = transformed_mask

    if not valid_mask.any():
        return None

    return clip, valid_mask


# ============================================================================
# Event iteration
# ============================================================================


def transformed_clips(
    events: pd.DataFrame,
    files_by_name: dict[str, BehaviorFiles],
    recording_column: str,
    frame_column: str,
    time_offsets_s: np.ndarray,
    output_width: int,
    output_height: int,
    likelihood_threshold: float,
    center_point: str,
    head_point: str,
    stabilization: str,
    output_pixels_per_mm: float | None,
    progress_bar: Any | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield transformed clips for all events in one category.

    The progress bar is advanced for every candidate event, including
    events that cannot produce a usable clip.
    """

    def advance_progress(number: int = 1) -> None:
        if progress_bar is not None:
            progress_bar.update(number)

    for recording_value, recording_events in events.groupby(
        recording_column,
        sort=False,
        dropna=False,
    ):
        # Processing events in onset order generally reduces backward and
        # random video seeks.
        recording_events = recording_events.sort_values(
            frame_column,
            kind="stable",
        )

        recording_name = normalize_recording_name(recording_value)
        number_of_recording_events = len(recording_events)

        behavior_files = files_by_name.get(recording_name)

        if behavior_files is None:
            tqdm.write("[skip] No files found for recording " f"{recording_name!r}")
            advance_progress(number_of_recording_events)
            continue

        if behavior_files.video is None:
            tqdm.write(f"[skip] {recording_name}: no video")
            advance_progress(number_of_recording_events)
            continue

        if behavior_files.full_tracking is None:
            tqdm.write(
                f"[skip] {recording_name}: " "no full-body Lightning Pose tracking"
            )
            advance_progress(number_of_recording_events)
            continue

        processed_events = 0

        try:
            tracking = load_lightning_pose(behavior_files.full_tracking)

            center_xy, center_valid = extract_pose_point(
                tracking=tracking,
                point_name=center_point,
                likelihood_threshold=likelihood_threshold,
            )

            head_xy, head_valid = extract_pose_point(
                tracking=tracking,
                point_name=head_point,
                likelihood_threshold=likelihood_threshold,
            )

            if output_pixels_per_mm is None:
                scale = 1.0
            else:
                source_pixels_per_mm = read_pixels_per_mm(behavior_files.metadata)
                scale = output_pixels_per_mm / source_pixels_per_mm

            with VideoReader(behavior_files.video) as reader:
                for event in recording_events.itertuples(index=False):
                    result = None

                    try:
                        frame_value = getattr(
                            event,
                            frame_column,
                        )

                        if not pd.isna(frame_value):
                            event_frame = int(round(float(frame_value)))

                            result = extract_aligned_clip(
                                event_frame=event_frame,
                                reader=reader,
                                center_xy=center_xy,
                                head_xy=head_xy,
                                center_valid=center_valid,
                                head_valid=head_valid,
                                time_offsets_s=time_offsets_s,
                                output_width=output_width,
                                output_height=output_height,
                                scale=scale,
                                stabilization=stabilization,
                            )

                    finally:
                        processed_events += 1
                        advance_progress()

                    if result is not None:
                        yield result

        except Exception as error:
            tqdm.write(f"[skip] {recording_name}: {error}")

            remaining_events = number_of_recording_events - processed_events

            if remaining_events > 0:
                advance_progress(remaining_events)


# ============================================================================
# Aggregation
# ============================================================================


def aggregate_mean(
    clips: Iterator[tuple[np.ndarray, np.ndarray]],
    output_shape: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Calculate a streaming pixel-wise mean."""
    # float32 reduces memory use and memory bandwidth compared with
    # float64 and is sufficient for ordinary event counts.
    sums = np.zeros(
        output_shape,
        dtype=np.float32,
    )

    counts = np.zeros(
        output_shape[:-1],
        dtype=np.uint32,
    )

    number_of_instances = 0

    for clip, valid in clips:
        np.add(
            sums,
            clip,
            out=sums,
            where=valid[..., None],
            casting="unsafe",
        )

        counts += valid
        number_of_instances += 1

    np.divide(
        sums,
        counts[..., None],
        out=sums,
        where=counts[..., None] > 0,
    )

    average = np.clip(
        np.rint(sums),
        0,
        255,
    ).astype(np.uint8)

    return average, counts, number_of_instances


def histogram_reduce_memmaps(
    clips: np.memmap,
    masks: np.memmap,
    number_of_instances: int,
    statistic: str,
    chunk_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate exact uint8 median or mode from temporary memmaps.

    Processing is chunked over output color values to avoid constructing a
    histogram for the entire video simultaneously.
    """
    output_shape = clips.shape[1:]
    flattened_size = int(np.prod(output_shape))

    result = np.zeros(
        flattened_size,
        dtype=np.uint8,
    )

    counts = np.zeros(
        int(np.prod(output_shape[:-1])),
        dtype=np.uint32,
    )

    clip_flat = clips[:number_of_instances].reshape(
        number_of_instances,
        flattened_size,
    )

    mask_flat = masks[:number_of_instances].reshape(
        number_of_instances,
        -1,
    )

    for start in tqdm(
        range(0, flattened_size, chunk_pixels),
        desc=f"Reducing {statistic}",
        unit="chunk",
        leave=False,
    ):
        stop = min(
            flattened_size,
            start + chunk_pixels,
        )
        size = stop - start

        color_indices = np.arange(
            start,
            stop,
        )
        mask_indices = color_indices // 3

        histogram = np.zeros(
            (256, size),
            dtype=np.uint32,
        )

        positions = np.arange(size)

        for instance_index in range(number_of_instances):
            valid = mask_flat[
                instance_index,
                mask_indices,
            ]

            if not valid.any():
                continue

            values = clip_flat[
                instance_index,
                start:stop,
            ]

            np.add.at(
                histogram,
                (
                    values[valid],
                    positions[valid],
                ),
                1,
            )

        observation_count = histogram.sum(axis=0)
        has_data = observation_count > 0

        if statistic == "mode":
            reduced = np.argmax(
                histogram,
                axis=0,
            ).astype(np.uint8)

        elif statistic == "median":
            cumulative = np.cumsum(
                histogram,
                axis=0,
                dtype=np.uint32,
            )

            # Lower median for even observation counts.
            target = (observation_count - 1) // 2

            reduced = np.argmax(
                cumulative > target[None, :],
                axis=0,
            ).astype(np.uint8)

        else:
            raise ValueError(f"Unsupported histogram statistic: {statistic}")

        result[start:stop][has_data] = reduced[has_data]

        # All three color channels use the same geometric validity mask.
        if start % 3 == 0 and stop % 3 == 0:
            counts[start // 3 : stop // 3] = observation_count[::3]

    return (
        result.reshape(output_shape),
        counts.reshape(output_shape[:-1]),
    )


def aggregate_median_or_mode(
    clips: Iterator[tuple[np.ndarray, np.ndarray]],
    output_shape: tuple[int, int, int, int],
    maximum_instances: int,
    statistic: str,
    temporary_directory: Path | None,
    chunk_pixels: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Calculate an exact median or mode using disk-backed arrays."""
    mask_shape = output_shape[:-1]

    with tempfile.TemporaryDirectory(dir=temporary_directory) as temp_name:
        temp_path = Path(temp_name)

        clip_memmap = np.memmap(
            temp_path / "clips.uint8",
            dtype=np.uint8,
            mode="w+",
            shape=(
                maximum_instances,
                *output_shape,
            ),
        )

        mask_memmap = np.memmap(
            temp_path / "masks.uint8",
            dtype=np.uint8,
            mode="w+",
            shape=(
                maximum_instances,
                *mask_shape,
            ),
        )

        number_of_instances = 0

        for clip, valid in clips:
            clip_memmap[number_of_instances] = clip
            mask_memmap[number_of_instances] = valid
            number_of_instances += 1

        clip_memmap.flush()
        mask_memmap.flush()

        if number_of_instances == 0:
            del clip_memmap
            del mask_memmap

            return (
                np.zeros(
                    output_shape,
                    dtype=np.uint8,
                ),
                np.zeros(
                    mask_shape,
                    dtype=np.uint32,
                ),
                0,
            )

        result, counts = histogram_reduce_memmaps(
            clips=clip_memmap,
            masks=mask_memmap,
            number_of_instances=number_of_instances,
            statistic=statistic,
            chunk_pixels=chunk_pixels,
        )

        del clip_memmap
        del mask_memmap

    return result, counts, number_of_instances


# ============================================================================
# Output
# ============================================================================


def write_video(
    path: Path,
    frames: np.ndarray,
    playback_fps: float,
    codec: str,
) -> None:
    """
    Write a BGR video using OpenCV.

    playback_fps affects only MP4 playback speed. It does not affect the
    temporal sampling used to generate frames.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    height = int(frames.shape[1])
    width = int(frames.shape[2])

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        float(playback_fps),
        (width, height),
        isColor=True,
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video: {path}. " "Try a different --codec."
        )

    try:
        for frame in frames:
            writer.write(np.ascontiguousarray(frame))
    finally:
        writer.release()


def validate_event_table(
    events: pd.DataFrame,
    csv_path: Path,
    group_columns: Sequence[str],
    frame_column: str,
) -> str:
    """Validate columns and return the recording column."""
    required = set(group_columns)
    required.add(frame_column)

    missing = required.difference(events.columns)

    if missing:
        raise ValueError(f"{csv_path} is missing columns: " f"{sorted(missing)}")

    return find_recording_column(events)


def process_event_table(
    *,
    table_name: str,
    csv_path: Path,
    output_directory: Path,
    files_by_name: dict[str, BehaviorFiles],
    group_columns: Sequence[str],
    frame_column: str,
    time_offsets_s: np.ndarray,
    sampling_fps: float,
    playback_fps: float,
    output_width: int,
    output_height: int,
    likelihood_threshold: float,
    center_point: str,
    head_point: str,
    stabilization: str,
    output_pixels_per_mm: float | None,
    statistic: str,
    temporary_directory: Path | None,
    histogram_chunk_pixels: int,
    codec: str,
    save_numpy: bool,
) -> list[dict[str, Any]]:
    """Export one aggregate video for every event category."""
    if not csv_path.exists():
        print("[skip] Event table does not exist: " f"{csv_path}")
        return []

    events = pd.read_csv(csv_path)

    if events.empty:
        print("[skip] Event table is empty: " f"{csv_path}")
        return []

    recording_column = validate_event_table(
        events=events,
        csv_path=csv_path,
        group_columns=group_columns,
        frame_column=frame_column,
    )

    events = events.dropna(
        subset=[
            *group_columns,
            frame_column,
            recording_column,
        ]
    ).copy()

    if events.empty:
        print("[skip] No valid categorized events in: " f"{csv_path}")
        return []

    output_shape = (
        len(time_offsets_s),
        output_height,
        output_width,
        3,
    )

    table_output = output_directory / table_name
    table_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[dict[str, Any]] = []

    grouped = events.groupby(
        list(group_columns),
        sort=True,
        dropna=False,
    )

    for group_key, category_events in tqdm(
        grouped,
        total=grouped.ngroups,
        desc=f"{table_name} groups",
        unit="group",
    ):
        label, category_metadata = group_label(
            group_columns=group_columns,
            group_key=group_key,
        )

        tqdm.write(
            f"[{table_name}] {label}: " f"{len(category_events):,} candidate events"
        )

        with tqdm(
            total=len(category_events),
            desc=f"{table_name}: {label}",
            unit="event",
            leave=False,
        ) as event_progress:
            clip_iterator = transformed_clips(
                events=category_events,
                files_by_name=files_by_name,
                recording_column=recording_column,
                frame_column=frame_column,
                time_offsets_s=time_offsets_s,
                output_width=output_width,
                output_height=output_height,
                likelihood_threshold=likelihood_threshold,
                center_point=center_point,
                head_point=head_point,
                stabilization=stabilization,
                output_pixels_per_mm=output_pixels_per_mm,
                progress_bar=event_progress,
            )

            if statistic == "mean":
                aggregate, counts, number_used = aggregate_mean(
                    clips=clip_iterator,
                    output_shape=output_shape,
                )

            else:
                (
                    aggregate,
                    counts,
                    number_used,
                ) = aggregate_median_or_mode(
                    clips=clip_iterator,
                    output_shape=output_shape,
                    maximum_instances=len(category_events),
                    statistic=statistic,
                    temporary_directory=temporary_directory,
                    chunk_pixels=histogram_chunk_pixels,
                )

            event_progress.set_postfix(
                used=number_used,
                skipped=(len(category_events) - number_used),
            )

        if number_used == 0:
            tqdm.write(f"[skip] {label}: no usable clips")
            continue

        video_path = table_output / f"{label}__{statistic}.mp4"

        write_video(
            path=video_path,
            frames=aggregate,
            playback_fps=playback_fps,
            codec=codec,
        )

        if save_numpy:
            numpy_path = table_output / f"{label}__{statistic}.npz"

            np.savez_compressed(
                numpy_path,
                frames=aggregate,
                valid_pixel_counts=counts,
                time_axis_ms=(time_offsets_s * 1000.0).astype(np.float32),
                sampling_fps=np.float32(sampling_fps),
                playback_fps=np.float32(playback_fps),
                slowdown_factor=np.float32(sampling_fps / playback_fps),
            )

        source_duration_s = len(time_offsets_s) / sampling_fps
        playback_duration_s = len(time_offsets_s) / playback_fps

        summary = {
            "table": table_name,
            **category_metadata,
            "statistic": statistic,
            "sampling_fps": float(sampling_fps),
            "playback_fps": float(playback_fps),
            "slowdown_factor": float(sampling_fps / playback_fps),
            "output_frames": int(len(time_offsets_s)),
            "source_duration_s": float(source_duration_s),
            "playback_duration_s": float(playback_duration_s),
            "candidate_events": int(len(category_events)),
            "used_events": int(number_used),
            "recordings": int(category_events[recording_column].nunique()),
            "video": str(video_path),
            "minimum_pixel_count": int(counts.min()),
            "maximum_pixel_count": int(counts.max()),
        }

        summaries.append(summary)

        tqdm.write(
            f"[saved] {video_path} "
            f"({number_used:,}/"
            f"{len(category_events):,} usable instances; "
            f"{sampling_fps:g} Hz sampling, "
            f"{playback_fps:g} FPS playback)"
        )

    return summaries


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create fish-centered, heading-aligned aggregate videos "
            "for each bout and saccade category."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment directory.",
    )

    parser.add_argument(
        "--bouts-csv",
        default="bouts.csv",
        help=(
            "Bout table relative to root, or 'none' to skip bouts. "
            "Default: bouts.csv"
        ),
    )

    parser.add_argument(
        "--saccades-csv",
        default="saccades_augmented.csv",
        help=(
            "Saccade table relative to root, or 'none' to skip "
            "saccades. Default: saccades_augmented.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="average_event_videos",
        help=("Output directory relative to root. " "Default: average_event_videos"),
    )

    parser.add_argument(
        "--bout-group-cols",
        type=parse_columns,
        default=["category", "sign"],
        help=("Comma-separated bout grouping columns. " "Default: category,sign"),
    )

    parser.add_argument(
        "--saccade-group-cols",
        type=parse_columns,
        default=[
            "saccade_category_name",
            "event_direction_name",
        ],
        help=(
            "Comma-separated saccade grouping columns. "
            "Default: "
            "saccade_category_name,event_direction_name"
        ),
    )

    parser.add_argument(
        "--bout-frame-column",
        default="frame_start",
        help=("Bout onset-frame column. " "Default: frame_start"),
    )

    parser.add_argument(
        "--saccade-frame-column",
        default="tracking_frame",
        help=("Saccade onset-frame column. " "Default: tracking_frame"),
    )

    parser.add_argument(
        "--pre-ms",
        type=float,
        default=300.0,
        help=("Source time before event onset. " "Default: 300 ms"),
    )

    parser.add_argument(
        "--post-ms",
        type=float,
        default=600.0,
        help=("Source time after event onset. " "Default: 600 ms"),
    )

    parser.add_argument(
        "--output-fps",
        "--sampling-fps",
        dest="output_fps",
        type=float,
        default=120.0,
        help=(
            "Temporal sampling rate used to construct aggregate "
            "frames. This determines source-frame selection and the "
            "event-relative time grid, but not MP4 playback speed. "
            "Default: 120"
        ),
    )

    parser.add_argument(
        "--playback-fps",
        type=float,
        default=5.0,
        help=(
            "Frame rate written into the output MP4. Use a value "
            "lower than --output-fps for slow motion. If omitted, "
            "defaults to --output-fps."
        ),
    )

    parser.add_argument(
        "--crop-width",
        type=int,
        default=240,
        help=("Output crop width in pixels. " "Default: 240"),
    )

    parser.add_argument(
        "--crop-height",
        type=int,
        default=320,
        help=("Output crop height in pixels. " "Default: 320"),
    )

    parser.add_argument(
        "--output-pixels-per-mm",
        type=float,
        default=40.0,
        help=(
            "Common output spatial scale. Each recording is "
            "rescaled from its metadata pix_per_mm calibration. "
            "Default: 40"
        ),
    )

    parser.add_argument(
        "--no-spatial-rescaling",
        action="store_true",
        help=(
            "Disable metadata-based spatial rescaling and use a "
            "scale factor of 1 for every recording."
        ),
    )

    parser.add_argument(
        "--center-point",
        default="Swim_Bladder",
        help=(
            "Lightning Pose point placed at the output center. " "Default: Swim_Bladder"
        ),
    )

    parser.add_argument(
        "--head-point",
        default="Head",
        help=("Lightning Pose point defining heading. " "Default: Head"),
    )

    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.9,
        help=("Minimum pose likelihood for center/head points. " "Default: 0.9"),
    )

    parser.add_argument(
        "--stabilization",
        choices=("frame", "onset"),
        default="frame",
        help=(
            "'frame' recenters and rerotates every frame; "
            "'onset' uses the onset pose for the complete clip. "
            "Default: frame"
        ),
    )

    parser.add_argument(
        "--statistic",
        choices=("mean", "median", "mode"),
        default="mean",
        help=(
            "Pixel-wise aggregation statistic. Mean is faster and "
            "uses less temporary disk space. Default: mean"
        ),
    )

    parser.add_argument(
        "--temporary-directory",
        type=Path,
        default=None,
        help=(
            "Temporary directory for median/mode clip storage. "
            "Use a disk with sufficient free space."
        ),
    )

    parser.add_argument(
        "--histogram-chunk-pixels",
        type=int,
        default=8192,
        help=(
            "Number of color values reduced at once for median/mode. "
            "Lower this if memory is limited. Default: 8192"
        ),
    )

    parser.add_argument(
        "--codec",
        default="mp4v",
        help=("Four-character OpenCV video codec. " "Default: mp4v"),
    )

    parser.add_argument(
        "--save-numpy",
        action="store_true",
        help=(
            "Also save aggregate frames, time axis, frame rates, "
            "and valid-pixel counts as compressed NPZ files."
        ),
    )

    # BehaviorScreen directory layout.
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


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.pre_ms < 0:
        raise ValueError("--pre-ms must be non-negative")

    if args.post_ms <= 0:
        raise ValueError("--post-ms must be positive")

    if args.output_fps <= 0:
        raise ValueError("--output-fps must be positive")

    if args.playback_fps is not None and args.playback_fps <= 0:
        raise ValueError("--playback-fps must be positive")

    if args.crop_width <= 0 or args.crop_height <= 0:
        raise ValueError("Crop dimensions must be positive")

    if not 0 <= args.likelihood_threshold <= 1:
        raise ValueError("--likelihood-threshold must be between 0 and 1")

    if args.output_pixels_per_mm is not None and args.output_pixels_per_mm <= 0:
        raise ValueError("--output-pixels-per-mm must be positive")

    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    if args.histogram_chunk_pixels < 1:
        raise ValueError("--histogram-chunk-pixels must be positive")


def resolve_optional_csv(
    root: Path,
    value: str,
) -> Path | None:
    """Resolve a CSV argument, allowing 'none' to disable it."""
    if value.lower() in {
        "none",
        "skip",
        "false",
        "",
    }:
        return None

    path = Path(value)

    if not path.is_absolute():
        path = root / path

    return path


def main() -> None:
    """Run aggregate-video export."""
    args = build_parser().parse_args()
    validate_arguments(args)

    sampling_fps = float(args.output_fps)

    playback_fps = (
        sampling_fps if args.playback_fps is None else float(args.playback_fps)
    )

    output_pixels_per_mm = (
        None if args.no_spatial_rescaling else args.output_pixels_per_mm
    )

    root = args.root.resolve()

    output_directory = Path(args.output_dir)

    if not output_directory.is_absolute():
        output_directory = root / output_directory

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.temporary_directory is not None:
        args.temporary_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    directories = Directories(
        root,
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

    behavior_files = find_files(directories)

    files_by_name = {files.metadata.stem: files for files in behavior_files}

    print(f"Found {len(files_by_name):,} recordings")

    # These frames describe source/event time and are therefore based on
    # sampling_fps, not playback_fps.
    pre_frames = int(round(args.pre_ms / 1000.0 * sampling_fps))

    post_frames = int(round(args.post_ms / 1000.0 * sampling_fps))

    time_offsets_s = (
        np.arange(
            -pre_frames,
            post_frames,
            dtype=np.float64,
        )
        / sampling_fps
    )

    output_frame_count = len(time_offsets_s)
    source_duration_s = output_frame_count / sampling_fps
    playback_duration_s = output_frame_count / playback_fps
    slowdown_factor = sampling_fps / playback_fps

    print("Aggregate-video configuration")
    print(f"  statistic: {args.statistic}")
    print(f"  output frames: {output_frame_count}")
    print(f"  sampling FPS: {sampling_fps:g}")
    print(f"  playback FPS: {playback_fps:g}")
    print(f"  slowdown factor: {slowdown_factor:g}x")
    print(f"  represented source duration: " f"{source_duration_s:g} s")
    print(f"  MP4 playback duration: " f"{playback_duration_s:g} s")
    print(f"  crop: " f"{args.crop_width} x {args.crop_height}")
    print(f"  stabilization: {args.stabilization}")
    print(f"  center/head: " f"{args.center_point} -> {args.head_point}")

    if output_pixels_per_mm is None:
        print("  spatial rescaling: disabled")
    else:
        print("  output spatial scale: " f"{output_pixels_per_mm:g} pixels/mm")

    summaries: list[dict[str, Any]] = []

    bouts_csv = resolve_optional_csv(
        root,
        args.bouts_csv,
    )

    if bouts_csv is not None:
        summaries.extend(
            process_event_table(
                table_name="bouts",
                csv_path=bouts_csv,
                output_directory=output_directory,
                files_by_name=files_by_name,
                group_columns=args.bout_group_cols,
                frame_column=args.bout_frame_column,
                time_offsets_s=time_offsets_s,
                sampling_fps=sampling_fps,
                playback_fps=playback_fps,
                output_width=args.crop_width,
                output_height=args.crop_height,
                likelihood_threshold=(args.likelihood_threshold),
                center_point=args.center_point,
                head_point=args.head_point,
                stabilization=args.stabilization,
                output_pixels_per_mm=(output_pixels_per_mm),
                statistic=args.statistic,
                temporary_directory=(args.temporary_directory),
                histogram_chunk_pixels=(args.histogram_chunk_pixels),
                codec=args.codec,
                save_numpy=args.save_numpy,
            )
        )

    saccades_csv = resolve_optional_csv(
        root,
        args.saccades_csv,
    )

    if saccades_csv is not None:
        summaries.extend(
            process_event_table(
                table_name="saccades",
                csv_path=saccades_csv,
                output_directory=output_directory,
                files_by_name=files_by_name,
                group_columns=(args.saccade_group_cols),
                frame_column=(args.saccade_frame_column),
                time_offsets_s=time_offsets_s,
                sampling_fps=sampling_fps,
                playback_fps=playback_fps,
                output_width=args.crop_width,
                output_height=args.crop_height,
                likelihood_threshold=(args.likelihood_threshold),
                center_point=args.center_point,
                head_point=args.head_point,
                stabilization=args.stabilization,
                output_pixels_per_mm=(output_pixels_per_mm),
                statistic=args.statistic,
                temporary_directory=(args.temporary_directory),
                histogram_chunk_pixels=(args.histogram_chunk_pixels),
                codec=args.codec,
                save_numpy=args.save_numpy,
            )
        )

    summary_path = output_directory / "summary.csv"

    pd.DataFrame.from_records(summaries).to_csv(
        summary_path,
        index=False,
    )

    print()
    print(f"Exported {len(summaries):,} aggregate videos")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
