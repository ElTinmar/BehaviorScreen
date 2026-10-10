#!/usr/bin/env python3
"""
Export event-aligned average body and eye keypoint coordinates.

For each event and output time point:

1. Select the corresponding source tracking frame.
2. Translate full-body coordinates so Swim_Bladder is at (0, 0).
3. Rotate coordinates so Swim_Bladder -> Head points upward.
4. Convert body coordinates from pixels to millimetres.
5. Convert cropped-eye coordinates into the same fish-centered coordinate
   system.
6. Aggregate coordinates within each bout or saccade category.

Only frame-by-frame alignment is supported. Translation and rotation are
removed independently at every frame.

Eye-coordinate conversion
-------------------------
The eye tracking is assumed to come from the cropped eye videos generated
by ``crop_around_eyes()``. Those videos are:

- centered on Head;
- rotated using Swim_Bladder -> Head;
- written with a known crop size and pixel scale.

Consequently, cropped-eye coordinates can be inserted into the body
coordinate system using:

    eye_fish_mm =
        head_fish_mm
        + (eye_crop_xy - eye_crop_center_xy) / eye_crop_pixels_per_mm

The default eye-crop settings match the current BehaviorScreen code:

    crop_size_mm = 1.6
    pixels_per_mm = 40.0

Outputs
-------
For each event category:

    <label>__mean.npz
    <label>__mean.csv

or:

    <label>__median.npz
    <label>__median.csv

NPZ arrays
----------
coordinates_mm
    Shape ``(time, keypoint, 2)``.

coordinate_std_mm
    Shape ``(time, keypoint, 2)``.

valid_counts
    Shape ``(time, keypoint)``.

time_axis_ms
    Shape ``(time,)``.

keypoints
    Shape ``(keypoint,)``.

keypoint_sources
    ``"body"`` or ``"eyes"`` for each keypoint.

Coordinate convention
---------------------
x_mm > 0
    Fish-relative right.

x_mm < 0
    Fish-relative left.

y_mm < 0
    Fish-relative forward/up.

y_mm > 0
    Fish-relative backward/down.

Example
-------
python -m BehaviorScreen.export_average_event_keypoints \
    /media/martin/DATA_18TB/Screen/WT/vehicle \
    --bouts-csv bouts.csv \
    --saccades-csv saccades_augmented.csv \
    --output-dir average_event_keypoints \
    --statistic mean \
    --pre-ms 300 \
    --post-ms 600 \
    --output-fps 120 \
    --likelihood-threshold 0.9 \
    --body-keypoints all \
    --eye-keypoints all \
    --eye-crop-size-mm 1.6 \
    --eye-crop-pixels-per-mm 40
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

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
    """Parse a comma-separated list of names."""
    columns = [column.strip() for column in value.split(",") if column.strip()]

    if not columns:
        raise argparse.ArgumentTypeError("At least one name must be supplied.")

    return columns


def safe_filename(value: object) -> str:
    """Convert a value into a filesystem-safe label."""
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
        "The event table must contain one of these recording columns: "
        "'file', 'fish', or 'recording'."
    )


def normalize_recording_name(value: object) -> str:
    """Normalize a recording identifier."""
    if pd.isna(value):
        return ""

    return Path(str(value)).stem


def scalar_for_output(value: object) -> object:
    """Convert a value to a CSV/JSON-compatible scalar."""
    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def load_recording_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Load one recording metadata file."""
    with metadata_path.open("r") as input_file:
        return json.load(input_file)


def read_source_fps(
    metadata: dict[str, Any],
) -> float:
    """Read camera FPS from recording metadata."""
    fps = float(metadata["camera"]["framerate_value"])

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid source frame rate: {fps}")

    return fps


def read_pixels_per_mm(
    metadata: dict[str, Any],
) -> float:
    """Read full-video spatial calibration."""
    pixels_per_mm = float(metadata["calibration"]["pix_per_mm"])

    if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError(
            f"Invalid full-video pixels-per-mm calibration: " f"{pixels_per_mm}"
        )

    return pixels_per_mm


# ============================================================================
# Pose extraction
# ============================================================================


def available_keypoints(
    tracking: pd.DataFrame,
) -> list[str]:
    """Return pose points containing x and y coordinates."""
    if tracking.empty:
        return []

    if not isinstance(
        tracking.columns,
        pd.MultiIndex,
    ):
        raise ValueError("Lightning Pose tracking must have MultiIndex columns.")

    point_names = tracking.columns.get_level_values(0)
    result: list[str] = []

    for point_name in dict.fromkeys(point_names):
        point = tracking[point_name]

        if "x" in point.columns and "y" in point.columns:
            result.append(str(point_name))

    return result


def extract_pose_point(
    tracking: pd.DataFrame,
    point_name: str,
    likelihood_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one pose point and its validity mask.

    Returns
    -------
    xy
        Shape ``(frames, 2)``.

    valid
        Shape ``(frames,)``.
    """
    point_names = tracking.columns.get_level_values(0)

    if point_name not in point_names:
        raise KeyError(f"Pose point {point_name!r} was not found.")

    point = tracking[point_name]

    if "x" not in point.columns or "y" not in point.columns:
        raise KeyError(f"Pose point {point_name!r} does not contain x and y.")

    xy = point[["x", "y"]].to_numpy(dtype=np.float64)

    valid = np.isfinite(xy).all(axis=1)

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


def extract_keypoint_arrays(
    tracking: pd.DataFrame,
    keypoint_names: Sequence[str],
    likelihood_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract multiple keypoints into dense arrays.

    Missing keypoints remain NaN and invalid.

    Returns
    -------
    coordinates
        Shape ``(frames, keypoints, 2)``.

    valid
        Shape ``(frames, keypoints)``.
    """
    number_of_frames = len(tracking)
    number_of_keypoints = len(keypoint_names)

    coordinates = np.full(
        (
            number_of_frames,
            number_of_keypoints,
            2,
        ),
        np.nan,
        dtype=np.float64,
    )

    valid = np.zeros(
        (
            number_of_frames,
            number_of_keypoints,
        ),
        dtype=bool,
    )

    for point_index, point_name in enumerate(keypoint_names):
        try:
            point_xy, point_valid = extract_pose_point(
                tracking=tracking,
                point_name=point_name,
                likelihood_threshold=(likelihood_threshold),
            )
        except KeyError:
            continue

        frame_count = min(
            number_of_frames,
            len(point_xy),
            len(point_valid),
        )

        coordinates[
            :frame_count,
            point_index,
        ] = point_xy[:frame_count]

        valid[
            :frame_count,
            point_index,
        ] = point_valid[:frame_count]

    return coordinates, valid


def discover_keypoints(
    behavior_files: Sequence[BehaviorFiles],
    requested: str,
    source: str,
) -> list[str]:
    """
    Resolve body or eye keypoint names.

    If ``requested`` is ``all``, names are discovered from the first
    usable tracking file of the requested source.
    """
    normalized = requested.strip().lower()

    if normalized in {
        "none",
        "skip",
        "false",
        "",
    }:
        return []

    if normalized != "all":
        keypoints = parse_columns(requested)

        if len(set(keypoints)) != len(keypoints):
            raise ValueError(f"--{source}-keypoints contains duplicate names.")

        return keypoints

    for files in behavior_files:
        if source == "body":
            tracking_path = files.full_tracking
        elif source == "eye":
            tracking_path = files.eyes_tracking
        else:
            raise ValueError(f"Unknown keypoint source: {source}")

        if tracking_path is None:
            continue

        try:
            tracking = load_lightning_pose(tracking_path)
        except Exception:
            continue

        keypoints = available_keypoints(tracking)

        if keypoints:
            return keypoints

    if source == "eye":
        print(
            "[warn] No usable eye-tracking file was found. "
            "No eye keypoints will be exported."
        )
        return []

    raise RuntimeError("No usable full-body Lightning Pose file was found.")


# ============================================================================
# Coordinate transformations
# ============================================================================


def transform_body_points(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    head_xy: np.ndarray,
    millimeters_per_pixel: float,
) -> np.ndarray | None:
    """
    Transform full-video coordinates into fish-centered coordinates.

    The center point becomes ``(0, 0)`` and center-to-head points toward
    negative y.
    """
    points_xy = np.asarray(
        points_xy,
        dtype=float,
    )
    center_xy = np.asarray(
        center_xy,
        dtype=float,
    )
    head_xy = np.asarray(
        head_xy,
        dtype=float,
    )

    if (
        center_xy.shape != (2,)
        or head_xy.shape != (2,)
        or not np.isfinite(center_xy).all()
        or not np.isfinite(head_xy).all()
    ):
        return None

    heading = head_xy - center_xy
    heading_length = float(np.linalg.norm(heading))

    if not np.isfinite(heading_length) or heading_length < 1e-6:
        return None

    heading_angle = np.arctan2(
        heading[1],
        heading[0],
    )

    # Matches crop_around_eyes() and the average-video transformation:
    #
    #     angle = heading_degrees + 90
    rotation_angle = heading_angle + np.pi / 2.0

    cosine = np.cos(rotation_angle)
    sine = np.sin(rotation_angle)

    relative = points_xy - center_xy

    transformed = np.empty_like(
        relative,
        dtype=np.float64,
    )

    transformed[:, 0] = cosine * relative[:, 0] + sine * relative[:, 1]

    transformed[:, 1] = -sine * relative[:, 0] + cosine * relative[:, 1]

    transformed *= millimeters_per_pixel

    return transformed


def calculate_eye_crop_size_px(
    eye_crop_size_mm: float,
    eye_crop_pixels_per_mm: float,
) -> int:
    """
    Reproduce the crop-size calculation in crop_around_eyes().

    Original code:

        crop_size = 2 * int(crop_size_mm * px_per_mm) // 2
    """
    crop_size = 2 * int(eye_crop_size_mm * eye_crop_pixels_per_mm) // 2

    if crop_size <= 0:
        raise ValueError("The calculated eye-crop size is not positive.")

    return crop_size


def transform_eye_points(
    eye_points_px: np.ndarray,
    head_fish_mm: np.ndarray,
    eye_crop_center_px: np.ndarray,
    eye_crop_pixels_per_mm: float,
) -> np.ndarray:
    """
    Convert cropped-eye coordinates into fish-centered millimetres.

    The cropped-eye video is already centered on Head and rotated into
    the frame-relative fish coordinate system.
    """
    eye_points_px = np.asarray(
        eye_points_px,
        dtype=float,
    )
    head_fish_mm = np.asarray(
        head_fish_mm,
        dtype=float,
    )
    eye_crop_center_px = np.asarray(
        eye_crop_center_px,
        dtype=float,
    )

    eye_relative_mm = (eye_points_px - eye_crop_center_px) / eye_crop_pixels_per_mm

    return eye_relative_mm + head_fish_mm[None, :]


def extract_aligned_keypoint_clip(
    event_frame: int,
    source_fps: float,
    body_coordinates_px: np.ndarray,
    body_valid: np.ndarray,
    body_keypoint_names: Sequence[str],
    eye_coordinates_px: np.ndarray | None,
    eye_valid: np.ndarray | None,
    eye_keypoint_names: Sequence[str],
    center_point: str,
    head_point: str,
    time_offsets_s: np.ndarray,
    body_pixels_per_mm: float,
    eye_crop_center_px: np.ndarray,
    eye_crop_pixels_per_mm: float,
) -> np.ndarray | None:
    """
    Extract one frame-by-frame aligned body-and-eye keypoint clip.

    Returns
    -------
    clip
        Shape ``(time, body + eye keypoints, 2)``. Invalid coordinates
        are represented by NaN.
    """
    number_of_body_frames = len(body_coordinates_px)

    number_of_output_frames = len(time_offsets_s)

    number_of_body_points = len(body_keypoint_names)

    number_of_eye_points = len(eye_keypoint_names)

    number_of_output_points = number_of_body_points + number_of_eye_points

    clip = np.full(
        (
            number_of_output_frames,
            number_of_output_points,
            2,
        ),
        np.nan,
        dtype=np.float32,
    )

    try:
        center_index = body_keypoint_names.index(center_point)
        head_index = body_keypoint_names.index(head_point)
    except ValueError as error:
        raise ValueError(
            f"{center_point!r} and {head_point!r} must be "
            "included in the body keypoints."
        ) from error

    source_frames = np.rint(event_frame + time_offsets_s * source_fps).astype(np.int64)

    millimeters_per_pixel = 1.0 / body_pixels_per_mm

    for output_index, source_frame_value in enumerate(source_frames):
        source_frame = int(source_frame_value)

        if source_frame < 0 or source_frame >= number_of_body_frames:
            continue

        # Frame-by-frame alignment always requires valid center and head.
        if (
            not body_valid[
                source_frame,
                center_index,
            ]
            or not body_valid[
                source_frame,
                head_index,
            ]
        ):
            continue

        transformed_body = transform_body_points(
            points_xy=body_coordinates_px[source_frame],
            center_xy=body_coordinates_px[
                source_frame,
                center_index,
            ],
            head_xy=body_coordinates_px[
                source_frame,
                head_index,
            ],
            millimeters_per_pixel=(millimeters_per_pixel),
        )

        if transformed_body is None:
            continue

        transformed_body[~body_valid[source_frame]] = np.nan

        clip[
            output_index,
            :number_of_body_points,
        ] = transformed_body.astype(
            np.float32,
            copy=False,
        )

        if (
            number_of_eye_points == 0
            or eye_coordinates_px is None
            or eye_valid is None
            or source_frame >= len(eye_coordinates_px)
        ):
            continue

        head_fish_mm = transformed_body[head_index]

        if not np.isfinite(head_fish_mm).all():
            continue

        transformed_eyes = transform_eye_points(
            eye_points_px=eye_coordinates_px[source_frame],
            head_fish_mm=head_fish_mm,
            eye_crop_center_px=(eye_crop_center_px),
            eye_crop_pixels_per_mm=(eye_crop_pixels_per_mm),
        )

        transformed_eyes[~eye_valid[source_frame]] = np.nan

        clip[
            output_index,
            number_of_body_points:,
        ] = transformed_eyes.astype(
            np.float32,
            copy=False,
        )

    if not np.isfinite(clip).any():
        return None

    return clip


# ============================================================================
# Event iteration
# ============================================================================


def transformed_keypoint_clips(
    events: pd.DataFrame,
    files_by_name: dict[str, BehaviorFiles],
    recording_column: str,
    frame_column: str,
    body_keypoint_names: Sequence[str],
    eye_keypoint_names: Sequence[str],
    center_point: str,
    head_point: str,
    likelihood_threshold: float,
    time_offsets_s: np.ndarray,
    eye_crop_size_mm: float,
    eye_crop_pixels_per_mm: float,
    progress_bar: Any | None = None,
) -> Iterator[np.ndarray]:
    """Yield transformed body-and-eye clips for all usable events."""

    def advance_progress(number: int = 1) -> None:
        if progress_bar is not None:
            progress_bar.update(number)

    eye_crop_size_px = calculate_eye_crop_size_px(
        eye_crop_size_mm=eye_crop_size_mm,
        eye_crop_pixels_per_mm=(eye_crop_pixels_per_mm),
    )

    eye_crop_center_px = np.array(
        [
            eye_crop_size_px // 2,
            eye_crop_size_px // 2,
        ],
        dtype=np.float64,
    )

    for (
        recording_value,
        recording_events,
    ) in events.groupby(
        recording_column,
        sort=False,
        dropna=False,
    ):
        recording_events = recording_events.sort_values(
            frame_column,
            kind="stable",
        )

        recording_name = normalize_recording_name(recording_value)

        number_of_recording_events = len(recording_events)

        files = files_by_name.get(recording_name)

        if files is None:
            tqdm.write(f"[skip] No files found for " f"{recording_name!r}")
            advance_progress(number_of_recording_events)
            continue

        if files.full_tracking is None:
            tqdm.write(f"[skip] {recording_name}: " "no full-body tracking")
            advance_progress(number_of_recording_events)
            continue

        processed_events = 0

        try:
            body_tracking = load_lightning_pose(files.full_tracking)

            if body_tracking.empty:
                raise ValueError("Full-body Lightning Pose tracking is empty.")

            available_body_points = set(available_keypoints(body_tracking))

            missing_anchors = {
                center_point,
                head_point,
            }.difference(available_body_points)

            if missing_anchors:
                raise KeyError(
                    "Missing body alignment keypoints: " f"{sorted(missing_anchors)}"
                )

            (
                body_coordinates_px,
                body_valid,
            ) = extract_keypoint_arrays(
                tracking=body_tracking,
                keypoint_names=(body_keypoint_names),
                likelihood_threshold=(likelihood_threshold),
            )

            eye_coordinates_px: np.ndarray | None = None
            eye_valid: np.ndarray | None = None

            if eye_keypoint_names:
                if files.eyes_tracking is None:
                    tqdm.write(
                        f"[warn] {recording_name}: " "no eye tracking; body points only"
                    )
                else:
                    try:
                        eye_tracking = load_lightning_pose(files.eyes_tracking)

                        if not eye_tracking.empty:
                            (
                                eye_coordinates_px,
                                eye_valid,
                            ) = extract_keypoint_arrays(
                                tracking=eye_tracking,
                                keypoint_names=(eye_keypoint_names),
                                likelihood_threshold=(likelihood_threshold),
                            )
                    except Exception as error:
                        tqdm.write(
                            f"[warn] {recording_name}: "
                            f"could not load eye tracking: {error}"
                        )

            metadata = load_recording_metadata(files.metadata)

            source_fps = read_source_fps(metadata)

            body_pixels_per_mm = read_pixels_per_mm(metadata)

            for event in recording_events.itertuples(index=False):
                result = None

                try:
                    frame_value = getattr(
                        event,
                        frame_column,
                    )

                    if not pd.isna(frame_value):
                        event_frame = int(round(float(frame_value)))

                        result = extract_aligned_keypoint_clip(
                            event_frame=event_frame,
                            source_fps=source_fps,
                            body_coordinates_px=(body_coordinates_px),
                            body_valid=body_valid,
                            body_keypoint_names=(body_keypoint_names),
                            eye_coordinates_px=(eye_coordinates_px),
                            eye_valid=eye_valid,
                            eye_keypoint_names=(eye_keypoint_names),
                            center_point=center_point,
                            head_point=head_point,
                            time_offsets_s=(time_offsets_s),
                            body_pixels_per_mm=(body_pixels_per_mm),
                            eye_crop_center_px=(eye_crop_center_px),
                            eye_crop_pixels_per_mm=(eye_crop_pixels_per_mm),
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
    clips: Iterator[np.ndarray],
    output_shape: tuple[int, int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """Calculate streaming coordinate-wise mean and standard deviation."""
    sums = np.zeros(
        output_shape,
        dtype=np.float64,
    )

    squared_sums = np.zeros(
        output_shape,
        dtype=np.float64,
    )

    counts = np.zeros(
        output_shape[:-1],
        dtype=np.uint32,
    )

    number_of_instances = 0

    for clip in clips:
        valid = np.isfinite(clip).all(axis=-1)

        safe_clip = np.where(
            valid[..., None],
            clip,
            0.0,
        ).astype(
            np.float64,
            copy=False,
        )

        sums += safe_clip
        squared_sums += safe_clip * safe_clip
        counts += valid

        number_of_instances += 1

    mean = np.full(
        output_shape,
        np.nan,
        dtype=np.float64,
    )

    second_moment = np.full(
        output_shape,
        np.nan,
        dtype=np.float64,
    )

    np.divide(
        sums,
        counts[..., None],
        out=mean,
        where=counts[..., None] > 0,
    )

    np.divide(
        squared_sums,
        counts[..., None],
        out=second_moment,
        where=counts[..., None] > 0,
    )

    variance = np.maximum(
        second_moment - mean**2,
        0.0,
    )

    standard_deviation = np.sqrt(variance)

    return (
        mean.astype(np.float32),
        standard_deviation.astype(np.float32),
        counts,
        number_of_instances,
    )


def aggregate_median(
    clips: Iterator[np.ndarray],
    output_shape: tuple[int, int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """Calculate exact coordinate-wise median and standard deviation."""
    collected = list(clips)
    number_of_instances = len(collected)

    if not collected:
        return (
            np.full(
                output_shape,
                np.nan,
                dtype=np.float32,
            ),
            np.full(
                output_shape,
                np.nan,
                dtype=np.float32,
            ),
            np.zeros(
                output_shape[:-1],
                dtype=np.uint32,
            ),
            0,
        )

    stacked = np.stack(
        collected,
        axis=0,
    )

    valid = np.isfinite(stacked).all(axis=-1)

    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        median = np.nanmedian(
            stacked,
            axis=0,
        )

        standard_deviation = np.nanstd(
            stacked,
            axis=0,
            ddof=0,
        )

    counts = valid.sum(axis=0).astype(np.uint32)

    return (
        median.astype(np.float32),
        standard_deviation.astype(np.float32),
        counts,
        number_of_instances,
    )


# ============================================================================
# Output
# ============================================================================


def save_coordinate_result(
    output_base: Path,
    statistic: str,
    coordinates: np.ndarray,
    standard_deviation: np.ndarray,
    counts: np.ndarray,
    time_offsets_s: np.ndarray,
    keypoint_names: Sequence[str],
    keypoint_sources: Sequence[str],
    sampling_fps: float,
    number_used: int,
    candidate_events: int,
    center_point: str,
    head_point: str,
    eye_crop_size_mm: float,
    eye_crop_pixels_per_mm: float,
    group_metadata: dict[str, object],
) -> tuple[Path, Path]:
    """Save one aggregated coordinate result as NPZ and CSV."""
    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    npz_path = output_base.with_suffix(".npz")
    csv_path = output_base.with_suffix(".csv")

    clean_group_metadata = {
        str(key): scalar_for_output(value) for key, value in group_metadata.items()
    }

    group_metadata_json = json.dumps(
        clean_group_metadata,
        sort_keys=True,
    )

    np.savez_compressed(
        npz_path,
        coordinates_mm=coordinates,
        coordinate_std_mm=(standard_deviation),
        valid_counts=counts,
        time_axis_ms=(time_offsets_s * 1000.0).astype(np.float32),
        keypoints=np.asarray(
            keypoint_names,
            dtype=str,
        ),
        keypoint_sources=np.asarray(
            keypoint_sources,
            dtype=str,
        ),
        coordinate_names=np.asarray(
            ["x", "y"],
            dtype=str,
        ),
        statistic=np.asarray(
            statistic,
            dtype=str,
        ),
        sampling_fps=np.float32(sampling_fps),
        used_events=np.int64(number_used),
        candidate_events=np.int64(candidate_events),
        alignment=np.asarray(
            "frame",
            dtype=str,
        ),
        center_point=np.asarray(
            center_point,
            dtype=str,
        ),
        head_point=np.asarray(
            head_point,
            dtype=str,
        ),
        eye_crop_size_mm=np.float32(eye_crop_size_mm),
        eye_crop_pixels_per_mm=np.float32(eye_crop_pixels_per_mm),
        group_metadata_json=np.asarray(
            group_metadata_json,
            dtype=str,
        ),
    )

    records: list[dict[str, object]] = []

    for time_index, time_s in enumerate(time_offsets_s):
        for point_index, point_name in enumerate(keypoint_names):
            x_mm = float(
                coordinates[
                    time_index,
                    point_index,
                    0,
                ]
            )

            y_mm = float(
                coordinates[
                    time_index,
                    point_index,
                    1,
                ]
            )

            records.append(
                {
                    **clean_group_metadata,
                    "statistic": statistic,
                    "sampling_fps": sampling_fps,
                    "used_events": number_used,
                    "candidate_events": (candidate_events),
                    "alignment": "frame",
                    "center_point": center_point,
                    "head_point": head_point,
                    "time_ms": float(time_s * 1000.0),
                    "keypoint": point_name,
                    "keypoint_source": (keypoint_sources[point_index]),
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "lateral_mm": x_mm,
                    "forward_mm": -y_mm,
                    "x_std_mm": float(
                        standard_deviation[
                            time_index,
                            point_index,
                            0,
                        ]
                    ),
                    "y_std_mm": float(
                        standard_deviation[
                            time_index,
                            point_index,
                            1,
                        ]
                    ),
                    "valid_count": int(
                        counts[
                            time_index,
                            point_index,
                        ]
                    ),
                }
            )

    pd.DataFrame.from_records(records).to_csv(
        csv_path,
        index=False,
        float_format="%.10g",
    )

    return npz_path, csv_path


def validate_event_table(
    events: pd.DataFrame,
    csv_path: Path,
    group_columns: Sequence[str],
    frame_column: str,
) -> str:
    """Validate an event table and return its recording column."""
    required = {
        *group_columns,
        frame_column,
    }

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
    body_keypoint_names: Sequence[str],
    eye_keypoint_names: Sequence[str],
    center_point: str,
    head_point: str,
    likelihood_threshold: float,
    time_offsets_s: np.ndarray,
    sampling_fps: float,
    statistic: str,
    eye_crop_size_mm: float,
    eye_crop_pixels_per_mm: float,
) -> list[dict[str, Any]]:
    """Export aggregate keypoints for every event category."""
    if not csv_path.exists():
        print(f"[skip] Event table does not exist: " f"{csv_path}")
        return []

    events = pd.read_csv(csv_path)

    if events.empty:
        print(f"[skip] Event table is empty: " f"{csv_path}")
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
        print(f"[skip] No categorized events in " f"{csv_path}")
        return []

    all_keypoint_names = [
        *body_keypoint_names,
        *eye_keypoint_names,
    ]

    keypoint_sources = [
        *(["body"] * len(body_keypoint_names)),
        *(["eyes"] * len(eye_keypoint_names)),
    ]

    output_shape = (
        len(time_offsets_s),
        len(all_keypoint_names),
        2,
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
            group_columns,
            group_key,
        )

        tqdm.write(
            f"[{table_name}] {label}: " f"{len(category_events):,} candidate events"
        )

        with tqdm(
            total=len(category_events),
            desc=f"{table_name}: {label}",
            unit="event",
            leave=False,
        ) as progress:
            clips = transformed_keypoint_clips(
                events=category_events,
                files_by_name=files_by_name,
                recording_column=(recording_column),
                frame_column=frame_column,
                body_keypoint_names=(body_keypoint_names),
                eye_keypoint_names=(eye_keypoint_names),
                center_point=center_point,
                head_point=head_point,
                likelihood_threshold=(likelihood_threshold),
                time_offsets_s=time_offsets_s,
                eye_crop_size_mm=(eye_crop_size_mm),
                eye_crop_pixels_per_mm=(eye_crop_pixels_per_mm),
                progress_bar=progress,
            )

            if statistic == "mean":
                (
                    aggregate,
                    standard_deviation,
                    counts,
                    number_used,
                ) = aggregate_mean(
                    clips=clips,
                    output_shape=output_shape,
                )

            elif statistic == "median":
                (
                    aggregate,
                    standard_deviation,
                    counts,
                    number_used,
                ) = aggregate_median(
                    clips=clips,
                    output_shape=output_shape,
                )

            else:
                raise ValueError(f"Unsupported statistic: {statistic}")

            progress.set_postfix(
                used=number_used,
                skipped=(len(category_events) - number_used),
            )

        if number_used == 0:
            tqdm.write(f"[skip] {label}: no usable keypoint clips")
            continue

        output_base = table_output / f"{label}__{statistic}"

        complete_group_metadata = {
            "table": table_name,
            **category_metadata,
        }

        (
            npz_path,
            output_csv_path,
        ) = save_coordinate_result(
            output_base=output_base,
            statistic=statistic,
            coordinates=aggregate,
            standard_deviation=(standard_deviation),
            counts=counts,
            time_offsets_s=time_offsets_s,
            keypoint_names=(all_keypoint_names),
            keypoint_sources=(keypoint_sources),
            sampling_fps=sampling_fps,
            number_used=number_used,
            candidate_events=len(category_events),
            center_point=center_point,
            head_point=head_point,
            eye_crop_size_mm=(eye_crop_size_mm),
            eye_crop_pixels_per_mm=(eye_crop_pixels_per_mm),
            group_metadata=(complete_group_metadata),
        )

        nonzero_counts = counts[counts > 0]

        summaries.append(
            {
                **{
                    key: scalar_for_output(value)
                    for key, value in complete_group_metadata.items()
                },
                "statistic": statistic,
                "sampling_fps": sampling_fps,
                "alignment": "frame",
                "center_point": center_point,
                "head_point": head_point,
                "candidate_events": int(len(category_events)),
                "used_events": int(number_used),
                "recordings": int(category_events[recording_column].nunique()),
                "body_keypoints": int(len(body_keypoint_names)),
                "eye_keypoints": int(len(eye_keypoint_names)),
                "output_frames": int(len(time_offsets_s)),
                "minimum_nonzero_count": (
                    int(nonzero_counts.min()) if len(nonzero_counts) else 0
                ),
                "maximum_count": int(counts.max()),
                "npz": str(npz_path),
                "csv": str(output_csv_path),
            }
        )

        tqdm.write(
            f"[saved] {npz_path} "
            f"({number_used:,}/"
            f"{len(category_events):,} usable)"
        )

    return summaries


# ============================================================================
# Command line
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create event-aligned, frame-stabilized average "
            "body-and-eye keypoint trajectories."
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
        help=("Bout table relative to root, or 'none' to skip. " "Default: bouts.csv"),
    )

    parser.add_argument(
        "--saccades-csv",
        default="saccades_augmented.csv",
        help=(
            "Saccade table relative to root, or 'none' to skip. "
            "Default: saccades_augmented.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="average_event_keypoints",
        help=("Output directory relative to root. " "Default: average_event_keypoints"),
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
        help=("Bout onset-frame column. Default: frame_start"),
    )

    parser.add_argument(
        "--saccade-frame-column",
        default="tracking_frame",
        help=("Saccade onset-frame column. Default: tracking_frame"),
    )

    parser.add_argument(
        "--pre-ms",
        type=float,
        default=300.0,
        help="Time before event onset. Default: 300 ms",
    )

    parser.add_argument(
        "--post-ms",
        type=float,
        default=600.0,
        help="Time after event onset. Default: 600 ms",
    )

    parser.add_argument(
        "--output-fps",
        "--sampling-fps",
        dest="output_fps",
        type=float,
        default=120.0,
        help=("Event-relative sampling rate. Default: 120"),
    )

    parser.add_argument(
        "--body-keypoints",
        default="all",
        help=(
            "'all', 'none', or a comma-separated list of full-body "
            "keypoints. Default: all"
        ),
    )

    parser.add_argument(
        "--eye-keypoints",
        default="all",
        help=(
            "'all', 'none', or a comma-separated list of cropped-eye "
            "keypoints. Default: all"
        ),
    )

    parser.add_argument(
        "--center-point",
        default="Swim_Bladder",
        help=("Body keypoint used as the origin. " "Default: Swim_Bladder"),
    )

    parser.add_argument(
        "--head-point",
        default="Head",
        help=("Body keypoint defining heading and eye-crop center. " "Default: Head"),
    )

    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.9,
        help=("Minimum Lightning Pose likelihood. Default: 0.9"),
    )

    parser.add_argument(
        "--eye-crop-size-mm",
        type=float,
        default=1.6,
        help=(
            "Crop-size argument used when creating eye videos. "
            "Must match crop_around_eyes(). Default: 1.6"
        ),
    )

    parser.add_argument(
        "--eye-crop-pixels-per-mm",
        type=float,
        default=40.0,
        help=(
            "Pixel scale used when creating eye videos. "
            "Must match crop_around_eyes(). Default: 40"
        ),
    )

    parser.add_argument(
        "--statistic",
        choices=("mean", "median"),
        default="mean",
        help=("Coordinate aggregation statistic. Default: mean"),
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


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate command-line arguments."""
    if args.pre_ms < 0:
        raise ValueError("--pre-ms must be non-negative")

    if args.post_ms <= 0:
        raise ValueError("--post-ms must be positive")

    if args.output_fps <= 0:
        raise ValueError("--output-fps must be positive")

    if not (0.0 <= args.likelihood_threshold <= 1.0):
        raise ValueError("--likelihood-threshold must be between 0 and 1")

    if args.eye_crop_size_mm <= 0:
        raise ValueError("--eye-crop-size-mm must be positive")

    if args.eye_crop_pixels_per_mm <= 0:
        raise ValueError("--eye-crop-pixels-per-mm must be positive")


def resolve_optional_csv(
    root: Path,
    value: str,
) -> Path | None:
    """Resolve a CSV path while allowing 'none' to disable it."""
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
    """Run average-keypoint export."""
    args = build_parser().parse_args()
    validate_arguments(args)

    root = args.root.resolve()

    output_directory = Path(args.output_dir)

    if not output_directory.is_absolute():
        output_directory = root / output_directory

    output_directory.mkdir(
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
        video_timestamp=(args.video_timestamp),
        results=args.results,
        plots=args.plots,
    )

    behavior_files = find_files(directories)

    files_by_name = {files.metadata.stem: files for files in behavior_files}

    print(f"Found {len(files_by_name):,} recordings")

    body_keypoint_names = discover_keypoints(
        behavior_files=behavior_files,
        requested=args.body_keypoints,
        source="body",
    )

    for anchor in (
        args.center_point,
        args.head_point,
    ):
        if anchor not in body_keypoint_names:
            body_keypoint_names.append(anchor)

    eye_keypoint_names = discover_keypoints(
        behavior_files=behavior_files,
        requested=args.eye_keypoints,
        source="eye",
    )

    duplicate_names = set(body_keypoint_names).intersection(eye_keypoint_names)

    if duplicate_names:
        raise ValueError(
            "Body and eye keypoint names overlap: " f"{sorted(duplicate_names)}"
        )

    sampling_fps = float(args.output_fps)

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

    eye_crop_size_px = calculate_eye_crop_size_px(
        eye_crop_size_mm=(args.eye_crop_size_mm),
        eye_crop_pixels_per_mm=(args.eye_crop_pixels_per_mm),
    )

    print("Average-keypoint configuration")
    print(f"  statistic: {args.statistic}")
    print(f"  alignment: frame-by-frame")
    print(f"  sampling FPS: {sampling_fps:g}")
    print(f"  output frames: {len(time_offsets_s)}")
    print(f"  source window: -{args.pre_ms:g} " f"to +{args.post_ms:g} ms")
    print(f"  body alignment: {args.center_point} " f"-> {args.head_point}")
    print(f"  likelihood threshold: " f"{args.likelihood_threshold:g}")
    print(
        f"  body keypoints ({len(body_keypoint_names)}): "
        f"{', '.join(body_keypoint_names)}"
    )
    print(
        f"  eye keypoints ({len(eye_keypoint_names)}): "
        f"{', '.join(eye_keypoint_names) if eye_keypoint_names else 'none'}"
    )
    print(f"  eye crop: {eye_crop_size_px} x " f"{eye_crop_size_px} pixels")
    print(f"  eye crop scale: " f"{args.eye_crop_pixels_per_mm:g} pixels/mm")
    print("  output units: millimetres")

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
                output_directory=(output_directory),
                files_by_name=files_by_name,
                group_columns=(args.bout_group_cols),
                frame_column=(args.bout_frame_column),
                body_keypoint_names=(body_keypoint_names),
                eye_keypoint_names=(eye_keypoint_names),
                center_point=args.center_point,
                head_point=args.head_point,
                likelihood_threshold=(args.likelihood_threshold),
                time_offsets_s=time_offsets_s,
                sampling_fps=sampling_fps,
                statistic=args.statistic,
                eye_crop_size_mm=(args.eye_crop_size_mm),
                eye_crop_pixels_per_mm=(args.eye_crop_pixels_per_mm),
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
                output_directory=(output_directory),
                files_by_name=files_by_name,
                group_columns=(args.saccade_group_cols),
                frame_column=(args.saccade_frame_column),
                body_keypoint_names=(body_keypoint_names),
                eye_keypoint_names=(eye_keypoint_names),
                center_point=args.center_point,
                head_point=args.head_point,
                likelihood_threshold=(args.likelihood_threshold),
                time_offsets_s=time_offsets_s,
                sampling_fps=sampling_fps,
                statistic=args.statistic,
                eye_crop_size_mm=(args.eye_crop_size_mm),
                eye_crop_pixels_per_mm=(args.eye_crop_pixels_per_mm),
            )
        )

    summary_path = output_directory / "summary.csv"

    pd.DataFrame.from_records(summaries).to_csv(
        summary_path,
        index=False,
    )

    print()
    print(f"Exported {len(summaries):,} " "aggregate keypoint groups")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
