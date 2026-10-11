#!/usr/bin/env python3
"""
Render average body-and-eye keypoint NPZ files as skeleton animations.

This script reads files created by:

    BehaviorScreen.export_average_event_keypoints

and creates one MP4 animation for each aggregate NPZ.

Default skeleton
----------------
The inferred body topology is:

    Swim_Bladder -> Head
    Swim_Bladder -> Tail_0
    Tail_0 -> Tail_1
    Tail_1 -> Tail_2
    ...

The inferred eye topology is:

    eye_left_front -> eye_left_back
    eye_right_front -> eye_right_back

The eye axes are rendered as thick colored lines.

Example
-------
python -m BehaviorScreen.render_average_skeletons \
    /media/martin/DATA_18TB/Screen/WT/vehicle/average_event_keypoints \
    --playback-fps 10 \
    --width 640 \
    --height 640 \
    --pixels-per-mm 80 \
    --minimum-valid-fraction 0.25
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Skeleton topology
# ============================================================================


def parse_skeleton_edges(
    value: str,
) -> list[tuple[str, str]]:
    """
    Parse comma-separated skeleton edges.

    Format:

        PointA:PointB,PointB:PointC
    """
    edges: list[tuple[str, str]] = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        parts = [
            part.strip()
            for part in item.split(
                ":",
                maxsplit=1,
            )
        ]

        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise argparse.ArgumentTypeError(
                "Skeleton edges must use the format " "'PointA:PointB,PointB:PointC'."
            )

        edges.append(
            (
                parts[0],
                parts[1],
            )
        )

    if not edges:
        raise argparse.ArgumentTypeError("At least one skeleton edge must be supplied.")

    return edges


def default_skeleton_edges(
    keypoint_names: Sequence[str],
    center_point: str,
    head_point: str,
) -> list[tuple[str, str]]:
    """Create the default body and eye skeleton topology."""
    available = set(keypoint_names)

    edges: list[tuple[str, str]] = []

    if center_point in available and head_point in available:
        edges.append(
            (
                center_point,
                head_point,
            )
        )

    tail_points: list[tuple[int, str]] = []

    for point_name in keypoint_names:
        match = re.fullmatch(
            r"Tail_(\d+)",
            point_name,
        )

        if match is not None:
            tail_points.append(
                (
                    int(match.group(1)),
                    point_name,
                )
            )

    tail_points.sort(key=lambda item: item[0])

    if tail_points and center_point in available:
        edges.append(
            (
                center_point,
                tail_points[0][1],
            )
        )

    for (
        (_, previous_name),
        (_, current_name),
    ) in zip(
        tail_points[:-1],
        tail_points[1:],
    ):
        edges.append(
            (
                previous_name,
                current_name,
            )
        )

    eye_edges = [
        (
            "eye_left_front",
            "eye_left_back",
        ),
        (
            "eye_right_front",
            "eye_right_back",
        ),
    ]

    for start_name, stop_name in eye_edges:
        if start_name in available and stop_name in available:
            edges.append(
                (
                    start_name,
                    stop_name,
                )
            )

    return edges


def validate_skeleton_edges(
    edges: Sequence[tuple[str, str]],
    keypoint_names: Sequence[str],
) -> None:
    """Check that all edge endpoints exist."""
    available = set(keypoint_names)

    missing = sorted(
        {
            point_name
            for edge in edges
            for point_name in edge
            if point_name not in available
        }
    )

    if missing:
        raise ValueError(
            "Skeleton topology references missing keypoints: "
            f"{missing}. Available keypoints: "
            f"{list(keypoint_names)}"
        )


def is_eye_edge(
    start_name: str,
    stop_name: str,
) -> bool:
    """Return whether both edge endpoints are eye landmarks."""
    return start_name.startswith("eye_") and stop_name.startswith("eye_")


def eye_edge_color(
    start_name: str,
    stop_name: str,
) -> tuple[int, int, int]:
    """Return a BGR color for an eye edge."""
    names = (
        start_name.lower(),
        stop_name.lower(),
    )

    if any("left" in name for name in names):
        return (220, 80, 50)

    if any("right" in name for name in names):
        return (50, 80, 220)

    return (160, 60, 160)


# ============================================================================
# NPZ loading
# ============================================================================


def scalar_from_npz(
    archive: Any,
    key: str,
    default: Any,
) -> Any:
    """Read one scalar from an NPZ archive."""
    if key not in archive.files:
        return default

    value = np.asarray(archive[key])

    if value.size != 1:
        return default

    scalar = value.reshape(-1)[0]

    if isinstance(
        scalar,
        np.generic,
    ):
        return scalar.item()

    return scalar


def load_aggregate(
    path: Path,
) -> dict[str, Any]:
    """Load and validate one aggregate-keypoint NPZ."""
    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        required = {
            "coordinates_mm",
            "valid_counts",
            "time_axis_ms",
            "keypoints",
        }

        missing = required.difference(archive.files)

        if missing:
            raise ValueError(
                f"{path} is not a compatible aggregate-keypoint "
                f"file. Missing arrays: {sorted(missing)}"
            )

        coordinates_mm = np.asarray(
            archive["coordinates_mm"],
            dtype=np.float32,
        )

        valid_counts = np.asarray(
            archive["valid_counts"],
            dtype=np.uint32,
        )

        time_axis_ms = np.asarray(
            archive["time_axis_ms"],
            dtype=np.float32,
        )

        keypoints = [str(value) for value in np.asarray(archive["keypoints"]).tolist()]

        if "keypoint_sources" in archive.files:
            keypoint_sources = [
                str(value) for value in np.asarray(archive["keypoint_sources"]).tolist()
            ]
        else:
            keypoint_sources = [
                ("eyes" if point_name.startswith("eye_") else "body")
                for point_name in keypoints
            ]

        if "coordinate_std_mm" in archive.files:
            coordinate_std_mm = np.asarray(
                archive["coordinate_std_mm"],
                dtype=np.float32,
            )
        else:
            coordinate_std_mm = np.full_like(
                coordinates_mm,
                np.nan,
            )

        metadata_json = scalar_from_npz(
            archive,
            "group_metadata_json",
            "{}",
        )

        try:
            group_metadata = json.loads(str(metadata_json))
        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            group_metadata = {}

        result = {
            "coordinates_mm": (coordinates_mm),
            "coordinate_std_mm": (coordinate_std_mm),
            "valid_counts": valid_counts,
            "time_axis_ms": time_axis_ms,
            "keypoints": keypoints,
            "keypoint_sources": (keypoint_sources),
            "sampling_fps": float(
                scalar_from_npz(
                    archive,
                    "sampling_fps",
                    np.nan,
                )
            ),
            "used_events": int(
                scalar_from_npz(
                    archive,
                    "used_events",
                    int(valid_counts.max()),
                )
            ),
            "candidate_events": int(
                scalar_from_npz(
                    archive,
                    "candidate_events",
                    0,
                )
            ),
            "statistic": str(
                scalar_from_npz(
                    archive,
                    "statistic",
                    "unknown",
                )
            ),
            "alignment": str(
                scalar_from_npz(
                    archive,
                    "alignment",
                    "frame",
                )
            ),
            "center_point": str(
                scalar_from_npz(
                    archive,
                    "center_point",
                    "Swim_Bladder",
                )
            ),
            "head_point": str(
                scalar_from_npz(
                    archive,
                    "head_point",
                    "Head",
                )
            ),
            "group_metadata": (group_metadata),
        }

    expected_coordinate_shape = (
        len(result["time_axis_ms"]),
        len(result["keypoints"]),
        2,
    )

    if result["coordinates_mm"].shape != expected_coordinate_shape:
        raise ValueError(
            f"{path}: coordinates_mm has shape "
            f"{result['coordinates_mm'].shape}; "
            f"expected {expected_coordinate_shape}."
        )

    if result["valid_counts"].shape != expected_coordinate_shape[:-1]:
        raise ValueError(
            f"{path}: valid_counts has shape "
            f"{result['valid_counts'].shape}; "
            f"expected {expected_coordinate_shape[:-1]}."
        )

    if len(result["keypoint_sources"]) != len(result["keypoints"]):
        raise ValueError(f"{path}: keypoint_sources and keypoints differ in length.")

    return result


def is_aggregate_npz(
    path: Path,
) -> bool:
    """Return whether a file is a compatible aggregate NPZ."""
    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            return {
                "coordinates_mm",
                "valid_counts",
                "time_axis_ms",
                "keypoints",
            }.issubset(archive.files)
    except Exception:
        return False


def find_input_files(
    input_path: Path,
) -> list[Path]:
    """Find aggregate NPZ files."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            raise ValueError("An input file must have the .npz suffix.")

        if not is_aggregate_npz(input_path):
            raise ValueError(
                f"{input_path} is not a compatible " "aggregate-keypoint NPZ."
            )

        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    return [
        path for path in sorted(input_path.rglob("*.npz")) if is_aggregate_npz(path)
    ]


# ============================================================================
# Rendering utilities
# ============================================================================


def coordinate_to_canvas(
    coordinate_mm: np.ndarray,
    canvas_center: tuple[float, float],
    pixels_per_mm: float,
) -> tuple[int, int]:
    """Convert a fish-centered coordinate to canvas pixels."""
    x_pixel = canvas_center[0] + float(coordinate_mm[0]) * pixels_per_mm

    y_pixel = canvas_center[1] + float(coordinate_mm[1]) * pixels_per_mm

    return (
        int(round(x_pixel)),
        int(round(y_pixel)),
    )


def point_is_near_canvas(
    point: tuple[int, int],
    width: int,
    height: int,
    margin: int = 0,
) -> bool:
    """Check whether a point is inside or near the canvas."""
    x, y = point

    return -margin <= x < width + margin and -margin <= y < height + margin


def make_title(
    path: Path,
    aggregate: dict[str, Any],
) -> str:
    """Create a short title from aggregate metadata."""
    metadata = aggregate["group_metadata"]

    if metadata:
        parts = [f"{key}={value}" for key, value in metadata.items() if key != "table"]

        if parts:
            return " | ".join(parts)

    return path.stem


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw readable text with a background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x, y = origin

    cv2.rectangle(
        frame,
        (
            x - 4,
            y - text_height - 4,
        ),
        (
            x + text_width + 4,
            y + baseline + 4,
        ),
        background,
        thickness=-1,
    )

    cv2.putText(
        frame,
        text,
        origin,
        font,
        font_scale,
        foreground,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def draw_scale_bar(
    frame: np.ndarray,
    pixels_per_mm: float,
    scale_bar_mm: float,
    margin: int = 24,
) -> None:
    """Draw a spatial scale bar."""
    if scale_bar_mm <= 0:
        return

    height, width = frame.shape[:2]

    length_pixels = int(round(scale_bar_mm * pixels_per_mm))

    x_stop = width - margin
    x_start = x_stop - length_pixels
    y = height - margin

    if x_start < margin:
        return

    cv2.line(
        frame,
        (x_start, y),
        (x_stop, y),
        (20, 20, 20),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"{scale_bar_mm:g} mm",
        (
            x_start,
            y - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


# ============================================================================
# Animation writer
# ============================================================================


def write_skeleton_animation(
    *,
    path: Path,
    source_path: Path,
    aggregate: dict[str, Any],
    edges: Sequence[tuple[str, str]],
    playback_fps: float,
    width: int,
    height: int,
    pixels_per_mm: float,
    point_radius: int,
    line_width: int,
    eye_line_width: int,
    minimum_valid_count: int,
    minimum_valid_fraction: float,
    codec: str,
    scale_bar_mm: float,
    show_point_names: bool,
    show_contributor_count: bool,
) -> dict[str, Any]:
    """Render one aggregate body-and-eye skeleton animation."""
    coordinates_mm = aggregate["coordinates_mm"]

    valid_counts = aggregate["valid_counts"]

    time_axis_ms = aggregate["time_axis_ms"]

    keypoint_names = aggregate["keypoints"]

    keypoint_sources = aggregate["keypoint_sources"]

    center_point = aggregate["center_point"]

    head_point = aggregate["head_point"]

    used_events = int(aggregate["used_events"])

    validate_skeleton_edges(
        edges=edges,
        keypoint_names=keypoint_names,
    )

    effective_minimum_count = max(
        minimum_valid_count,
        int(np.ceil(minimum_valid_fraction * used_events)),
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        float(playback_fps),
        (
            int(width),
            int(height),
        ),
        isColor=True,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}. " "Try a different --codec.")

    keypoint_indices = {
        point_name: point_index for point_index, point_name in enumerate(keypoint_names)
    }

    canvas_center = (
        (width - 1) / 2.0,
        (height - 1) / 2.0,
    )

    # OpenCV colors are BGR.
    background_color = (
        248,
        248,
        248,
    )

    body_edge_color = (
        35,
        35,
        35,
    )

    ordinary_point_color = (
        70,
        70,
        70,
    )

    head_color = (
        40,
        40,
        220,
    )

    center_color = (
        40,
        180,
        40,
    )

    tail_color = (
        220,
        110,
        30,
    )

    left_eye_color = (
        220,
        80,
        50,
    )

    right_eye_color = (
        50,
        80,
        220,
    )

    generic_eye_color = (
        160,
        60,
        160,
    )

    title = make_title(
        source_path,
        aggregate,
    )

    rendered_frame_count = 0
    frames_without_points = 0

    try:
        for frame_index, time_ms_value in enumerate(time_axis_ms):
            frame = np.full(
                (
                    height,
                    width,
                    3,
                ),
                background_color,
                dtype=np.uint8,
            )

            coordinates = coordinates_mm[frame_index]

            counts = valid_counts[frame_index]

            point_valid = np.isfinite(coordinates).all(axis=1) & (
                counts >= effective_minimum_count
            )

            if not point_valid.any():
                frames_without_points += 1

            canvas_points: list[tuple[int, int] | None] = []

            for (
                point_index,
                coordinate,
            ) in enumerate(coordinates):
                if not point_valid[point_index]:
                    canvas_points.append(None)
                    continue

                canvas_points.append(
                    coordinate_to_canvas(
                        coordinate_mm=(coordinate),
                        canvas_center=(canvas_center),
                        pixels_per_mm=(pixels_per_mm),
                    )
                )

            origin = (
                int(round(canvas_center[0])),
                int(round(canvas_center[1])),
            )

            cv2.drawMarker(
                frame,
                origin,
                (
                    190,
                    190,
                    190,
                ),
                markerType=(cv2.MARKER_CROSS),
                markerSize=12,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

            # Draw edges before points.
            for (
                start_name,
                stop_name,
            ) in edges:
                start_index = keypoint_indices[start_name]

                stop_index = keypoint_indices[stop_name]

                start_point = canvas_points[start_index]

                stop_point = canvas_points[stop_index]

                if start_point is None or stop_point is None:
                    continue

                if not (
                    point_is_near_canvas(
                        start_point,
                        width,
                        height,
                        margin=100,
                    )
                    or point_is_near_canvas(
                        stop_point,
                        width,
                        height,
                        margin=100,
                    )
                ):
                    continue

                if is_eye_edge(
                    start_name,
                    stop_name,
                ):
                    edge_color = eye_edge_color(
                        start_name,
                        stop_name,
                    )
                    thickness = eye_line_width
                else:
                    edge_color = body_edge_color
                    thickness = line_width

                cv2.line(
                    frame,
                    start_point,
                    stop_point,
                    edge_color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

            # Draw keypoints.
            for (
                point_index,
                point_name,
            ) in enumerate(keypoint_names):
                point = canvas_points[point_index]

                if point is None:
                    continue

                if not point_is_near_canvas(
                    point,
                    width,
                    height,
                ):
                    continue

                source = keypoint_sources[point_index]

                if point_name == head_point:
                    color = head_color
                    radius = point_radius + 2

                elif point_name == center_point:
                    color = center_color
                    radius = point_radius + 1

                elif source == "eyes":
                    point_name_lower = point_name.lower()

                    if "left" in point_name_lower:
                        color = left_eye_color
                    elif "right" in point_name_lower:
                        color = right_eye_color
                    else:
                        color = generic_eye_color

                    radius = point_radius + 1

                elif point_name.startswith("Tail_"):
                    color = tail_color
                    radius = point_radius

                else:
                    color = ordinary_point_color
                    radius = point_radius

                cv2.circle(
                    frame,
                    point,
                    radius,
                    color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

                cv2.circle(
                    frame,
                    point,
                    radius,
                    (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

                if show_point_names:
                    cv2.putText(
                        frame,
                        point_name,
                        (
                            point[0] + radius + 3,
                            point[1] - radius - 3,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (30, 30, 30),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

            time_ms = float(time_ms_value)

            if abs(time_ms) < 0.5:
                time_text = "t = 0 ms"
                time_color = (
                    0,
                    0,
                    220,
                )
            else:
                time_text = f"t = {time_ms:+.0f} ms"
                time_color = (
                    25,
                    25,
                    25,
                )

            draw_text_with_background(
                frame=frame,
                text=time_text,
                origin=(18, 31),
                font_scale=0.65,
                foreground=time_color,
                background=(background_color),
                thickness=2,
            )

            if title:
                draw_text_with_background(
                    frame=frame,
                    text=title,
                    origin=(18, 58),
                    font_scale=0.45,
                    foreground=(
                        25,
                        25,
                        25,
                    ),
                    background=(background_color),
                    thickness=1,
                )

            if show_contributor_count:
                visible_counts = counts[point_valid]

                if len(visible_counts):
                    contributor_text = (
                        "contributors: "
                        f"{int(visible_counts.min())}"
                        "-"
                        f"{int(visible_counts.max())}"
                        f" / {used_events}"
                    )
                else:
                    contributor_text = f"contributors: 0 / " f"{used_events}"

                cv2.putText(
                    frame,
                    contributor_text,
                    (
                        18,
                        height - 18,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (
                        80,
                        80,
                        80,
                    ),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            draw_scale_bar(
                frame=frame,
                pixels_per_mm=(pixels_per_mm),
                scale_bar_mm=scale_bar_mm,
            )

            writer.write(np.ascontiguousarray(frame))

            rendered_frame_count += 1

    finally:
        writer.release()

    sampling_fps = float(aggregate["sampling_fps"])

    return {
        "source_npz": str(source_path),
        "output_video": str(path),
        "statistic": aggregate["statistic"],
        "alignment": aggregate["alignment"],
        "used_events": used_events,
        "sampling_fps": sampling_fps,
        "playback_fps": playback_fps,
        "slowdown_factor": (
            sampling_fps / playback_fps if np.isfinite(sampling_fps) else np.nan
        ),
        "frames": rendered_frame_count,
        "frames_without_visible_points": (frames_without_points),
        "minimum_valid_count": (effective_minimum_count),
        "pixels_per_mm": pixels_per_mm,
        "width": width,
        "height": height,
    }


# ============================================================================
# Output paths
# ============================================================================


def output_path_for(
    input_file: Path,
    input_root: Path,
    output_directory: Path | None,
) -> Path:
    """Determine the MP4 output path."""
    filename = f"{input_file.stem}" "__skeleton.mp4"

    if output_directory is None:
        return input_file.parent / filename

    if input_root.is_dir():
        relative_parent = input_file.parent.relative_to(input_root)
    else:
        relative_parent = Path()

    return output_directory / relative_parent / filename


# ============================================================================
# Command line
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Render average body-and-eye keypoint NPZ files "
            "as MP4 skeleton animations."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "An aggregate NPZ file or a directory " "containing aggregate NPZ files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. By default, each MP4 "
            "is written beside its source NPZ."
        ),
    )

    parser.add_argument(
        "--skeleton-edges",
        type=parse_skeleton_edges,
        default=None,
        help=(
            "Custom edges using "
            "'PointA:PointB,PointB:PointC'. "
            "By default, body and eye topology is inferred."
        ),
    )

    parser.add_argument(
        "--playback-fps",
        type=float,
        default=10.0,
        help=("MP4 playback frame rate. Default: 10"),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Output width. Default: 640",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Output height. Default: 640",
    )

    parser.add_argument(
        "--pixels-per-mm",
        type=float,
        default=80.0,
        help=("Animation rendering scale. " "Default: 80 pixels/mm"),
    )

    parser.add_argument(
        "--point-radius",
        type=int,
        default=5,
        help="Keypoint radius. Default: 5",
    )

    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="Body skeleton line width. Default: 3",
    )

    parser.add_argument(
        "--eye-line-width",
        type=int,
        default=5,
        help="Eye-axis line width. Default: 5",
    )

    parser.add_argument(
        "--minimum-valid-count",
        type=int,
        default=1,
        help=(
            "Minimum contributing event count required to " "draw a point. Default: 1"
        ),
    )

    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=0.25,
        help=(
            "Minimum fraction of usable events required to "
            "draw a point. Default: 0.25"
        ),
    )

    parser.add_argument(
        "--scale-bar-mm",
        type=float,
        default=1.0,
        help=("Scale-bar length in millimetres. " "Use 0 to disable. Default: 1"),
    )

    parser.add_argument(
        "--show-point-names",
        action="store_true",
        help="Draw keypoint names.",
    )

    parser.add_argument(
        "--hide-contributor-count",
        action="store_true",
        help="Hide contributing-event counts.",
    )

    parser.add_argument(
        "--codec",
        default="mp4v",
        help=("Four-character OpenCV video codec. " "Default: mp4v"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing animation files.",
    )

    return parser


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate command-line arguments."""
    if args.playback_fps <= 0:
        raise ValueError("--playback-fps must be positive")

    if args.width <= 0 or args.height <= 0:
        raise ValueError("Animation dimensions must be positive")

    if args.pixels_per_mm <= 0:
        raise ValueError("--pixels-per-mm must be positive")

    if args.point_radius < 1:
        raise ValueError("--point-radius must be at least 1")

    if args.line_width < 1:
        raise ValueError("--line-width must be at least 1")

    if args.eye_line_width < 1:
        raise ValueError("--eye-line-width must be at least 1")

    if args.minimum_valid_count < 1:
        raise ValueError("--minimum-valid-count must be at least 1")

    if not (0.0 <= args.minimum_valid_fraction <= 1.0):
        raise ValueError("--minimum-valid-fraction must be between 0 and 1")

    if args.scale_bar_mm < 0:
        raise ValueError("--scale-bar-mm must be non-negative")

    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")


def main() -> None:
    """Render all requested skeleton animations."""
    args = build_parser().parse_args()
    validate_arguments(args)

    input_path = args.input.resolve()

    input_files = find_input_files(input_path)

    if not input_files:
        print("No compatible aggregate-keypoint NPZ files were found.")
        return

    output_directory = (
        args.output_dir.resolve() if args.output_dir is not None else None
    )

    if output_directory is not None:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(f"Found {len(input_files):,} aggregate keypoint files")

    print("Skeleton rendering configuration")
    print(f"  playback FPS: {args.playback_fps:g}")
    print(f"  canvas: {args.width} x {args.height}")
    print(f"  scale: {args.pixels_per_mm:g} pixels/mm")
    print(f"  minimum valid count: " f"{args.minimum_valid_count}")
    print(f"  minimum valid fraction: " f"{args.minimum_valid_fraction:g}")

    summaries: list[dict[str, Any]] = []

    for input_file in tqdm(
        input_files,
        desc="Skeleton animations",
        unit="file",
    ):
        output_path = output_path_for(
            input_file=input_file,
            input_root=input_path,
            output_directory=(output_directory),
        )

        if output_path.exists() and not args.overwrite:
            tqdm.write(f"[skip] Already exists: {output_path}")
            continue

        try:
            aggregate = load_aggregate(input_file)

            if args.skeleton_edges is None:
                edges = default_skeleton_edges(
                    keypoint_names=(aggregate["keypoints"]),
                    center_point=(aggregate["center_point"]),
                    head_point=(aggregate["head_point"]),
                )
            else:
                edges = args.skeleton_edges

            if not edges:
                raise ValueError(
                    "No skeleton edges could be constructed. "
                    "Supply --skeleton-edges explicitly."
                )

            summary = write_skeleton_animation(
                path=output_path,
                source_path=input_file,
                aggregate=aggregate,
                edges=edges,
                playback_fps=(args.playback_fps),
                width=args.width,
                height=args.height,
                pixels_per_mm=(args.pixels_per_mm),
                point_radius=(args.point_radius),
                line_width=(args.line_width),
                eye_line_width=(args.eye_line_width),
                minimum_valid_count=(args.minimum_valid_count),
                minimum_valid_fraction=(args.minimum_valid_fraction),
                codec=args.codec,
                scale_bar_mm=(args.scale_bar_mm),
                show_point_names=(args.show_point_names),
                show_contributor_count=(not args.hide_contributor_count),
            )

            summaries.append(summary)

            tqdm.write(f"[saved] {output_path}")

        except Exception as error:
            tqdm.write(f"[skip] {input_file}: {error}")

    if output_directory is None:
        summary_directory = input_path if input_path.is_dir() else input_path.parent
    else:
        summary_directory = output_directory

    summary_path = summary_directory / "skeleton_animation_summary.csv"

    pd.DataFrame.from_records(summaries).to_csv(
        summary_path,
        index=False,
    )

    print()
    print(f"Rendered {len(summaries):,} skeleton animations")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
