#!/usr/bin/env python3
"""
Render aggregate keypoint coordinates as a single time-colored image.

The script reads NPZ files created by:

    BehaviorScreen.export_average_event_keypoints

Skeletons from multiple event-relative time points are overlaid. Their
color indicates time relative to event onset.

Display convention
------------------
Horizontal axis:
    Fish-relative lateral position.

Vertical axis:
    Fish-relative forward position.

The exported coordinate convention is y_mm < 0 for forward, so this
renderer displays:

    display_y = -y_mm

Example
-------
python -m BehaviorScreen.render_average_skeleton_timecolor \
    ROOT/average_event_keypoints \
    --output-dir ROOT/average_skeleton_timecolor \
    --stride 4 \
    --minimum-valid-fraction 0.25 \
    --cmap turbo \
    --format png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
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

        parts = [part.strip() for part in item.split(":", maxsplit=1)]

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
    """Construct the default body and eye skeleton."""
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
    """Check that all edge endpoints are present."""
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
    """Return whether an edge connects two eye landmarks."""
    return start_name.startswith("eye_") and stop_name.startswith("eye_")


# ============================================================================
# NPZ loading
# ============================================================================


def scalar_from_npz(
    archive: Any,
    key: str,
    default: Any,
) -> Any:
    """Read a scalar value from an NPZ archive."""
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
            raise ValueError(f"{path} is missing arrays: " f"{sorted(missing)}")

        coordinates_mm = np.asarray(
            archive["coordinates_mm"],
            dtype=np.float64,
        )

        valid_counts = np.asarray(
            archive["valid_counts"],
            dtype=np.uint32,
        )

        time_axis_ms = np.asarray(
            archive["time_axis_ms"],
            dtype=np.float64,
        )

        keypoints = [str(value) for value in np.asarray(archive["keypoints"]).tolist()]

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
            "coordinates_mm": coordinates_mm,
            "valid_counts": valid_counts,
            "time_axis_ms": time_axis_ms,
            "keypoints": keypoints,
            "used_events": int(
                scalar_from_npz(
                    archive,
                    "used_events",
                    int(valid_counts.max()),
                )
            ),
            "statistic": str(
                scalar_from_npz(
                    archive,
                    "statistic",
                    "unknown",
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
            "group_metadata": group_metadata,
        }

    expected_shape = (
        len(time_axis_ms),
        len(keypoints),
        2,
    )

    if coordinates_mm.shape != expected_shape:
        raise ValueError(
            f"{path}: coordinates_mm has shape "
            f"{coordinates_mm.shape}; expected "
            f"{expected_shape}."
        )

    if valid_counts.shape != expected_shape[:-1]:
        raise ValueError(
            f"{path}: valid_counts has shape "
            f"{valid_counts.shape}; expected "
            f"{expected_shape[:-1]}."
        )

    return result


def is_aggregate_npz(
    path: Path,
) -> bool:
    """Return whether a path appears to be an aggregate NPZ."""
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
    """Find compatible aggregate NPZ files."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            raise ValueError("The input file must have the .npz suffix.")

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
# Plotting
# ============================================================================


def make_title(
    source_path: Path,
    aggregate: dict[str, Any],
) -> str:
    """Create a figure title from saved group metadata."""
    metadata = aggregate["group_metadata"]

    parts = [f"{key}={value}" for key, value in metadata.items() if key != "table"]

    if parts:
        group_title = " | ".join(parts)
    else:
        group_title = source_path.stem

    return (
        f"{group_title}\n"
        f"{aggregate['statistic']} skeleton, "
        f"n={aggregate['used_events']}"
    )


def selected_frame_indices(
    time_axis_ms: np.ndarray,
    stride: int,
) -> np.ndarray:
    """
    Select frames at a fixed stride while always including event onset.
    """
    indices = np.arange(
        0,
        len(time_axis_ms),
        stride,
        dtype=int,
    )

    if len(time_axis_ms):
        onset_index = int(np.argmin(np.abs(time_axis_ms)))

        indices = np.unique(
            np.concatenate(
                (
                    indices,
                    [onset_index],
                    [len(time_axis_ms) - 1],
                )
            )
        )

    return indices


def add_line_collection(
    axis: plt.Axes,
    segments: list[np.ndarray],
    segment_times: list[float],
    norm: Normalize,
    cmap: str,
    linewidth: float,
    alpha: float,
    zorder: float,
) -> None:
    """Add time-colored line segments to an axis."""
    if not segments:
        return

    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    collection.set_array(
        np.asarray(
            segment_times,
            dtype=float,
        )
    )

    axis.add_collection(collection)


def plot_time_colored_skeleton(
    *,
    source_path: Path,
    output_path: Path,
    aggregate: dict[str, Any],
    edges: Sequence[tuple[str, str]],
    stride: int,
    minimum_valid_count: int,
    minimum_valid_fraction: float,
    cmap: str,
    body_line_width: float,
    eye_line_width: float,
    point_size: float,
    alpha: float,
    show_points: bool,
    show_trajectories: bool,
    padding_mm: float,
    extent_mm: float | None,
    dpi: int,
    figure_size: float,
    transparent: bool,
) -> dict[str, Any]:
    """Create one time-colored skeleton overlay."""
    coordinates_mm = aggregate["coordinates_mm"]

    valid_counts = aggregate["valid_counts"]

    time_axis_ms = aggregate["time_axis_ms"]

    keypoint_names = aggregate["keypoints"]

    used_events = int(aggregate["used_events"])

    validate_skeleton_edges(
        edges=edges,
        keypoint_names=keypoint_names,
    )

    effective_minimum_count = max(
        minimum_valid_count,
        int(np.ceil(minimum_valid_fraction * used_events)),
    )

    frame_indices = selected_frame_indices(
        time_axis_ms=time_axis_ms,
        stride=stride,
    )

    if len(frame_indices) == 0:
        raise ValueError(f"{source_path} contains no time points.")

    selected_times = time_axis_ms[frame_indices]

    time_min = float(np.nanmin(time_axis_ms))
    time_max = float(np.nanmax(time_axis_ms))

    if time_max <= time_min:
        time_max = time_min + 1.0

    norm = Normalize(
        vmin=time_min,
        vmax=time_max,
    )

    keypoint_indices = {
        point_name: point_index for point_index, point_name in enumerate(keypoint_names)
    }

    body_segments: list[np.ndarray] = []
    body_segment_times: list[float] = []

    eye_segments: list[np.ndarray] = []
    eye_segment_times: list[float] = []

    point_coordinates: list[np.ndarray] = []
    point_times: list[float] = []

    all_display_coordinates: list[np.ndarray] = []

    # Overlay one skeleton for each selected event-relative time.
    for frame_index, time_ms in zip(
        frame_indices,
        selected_times,
    ):
        coordinates = coordinates_mm[frame_index]

        counts = valid_counts[frame_index]

        valid = np.isfinite(coordinates).all(axis=1) & (
            counts >= effective_minimum_count
        )

        # Convert from image-style y coordinates to forward-positive
        # plotting coordinates.
        displayed = coordinates.copy()
        displayed[:, 1] *= -1.0

        if valid.any():
            all_display_coordinates.append(displayed[valid])

        for start_name, stop_name in edges:
            start_index = keypoint_indices[start_name]
            stop_index = keypoint_indices[stop_name]

            if not valid[start_index] or not valid[stop_index]:
                continue

            segment = np.stack(
                (
                    displayed[start_index],
                    displayed[stop_index],
                ),
                axis=0,
            )

            if is_eye_edge(
                start_name,
                stop_name,
            ):
                eye_segments.append(segment)
                eye_segment_times.append(float(time_ms))
            else:
                body_segments.append(segment)
                body_segment_times.append(float(time_ms))

        if show_points:
            for point_index in np.flatnonzero(valid):
                point_coordinates.append(displayed[point_index])
                point_times.append(float(time_ms))

    if not all_display_coordinates:
        raise ValueError(
            "No coordinates passed the validity threshold. "
            "Lower --minimum-valid-fraction or "
            "--minimum-valid-count."
        )

    figure, axis = plt.subplots(
        figsize=(
            figure_size,
            figure_size,
        ),
        layout="constrained",
    )

    add_line_collection(
        axis=axis,
        segments=body_segments,
        segment_times=body_segment_times,
        norm=norm,
        cmap=cmap,
        linewidth=body_line_width,
        alpha=alpha,
        zorder=2,
    )

    add_line_collection(
        axis=axis,
        segments=eye_segments,
        segment_times=eye_segment_times,
        norm=norm,
        cmap=cmap,
        linewidth=eye_line_width,
        alpha=alpha,
        zorder=3,
    )

    if show_trajectories:
        for point_index in range(len(keypoint_names)):
            trajectory = coordinates_mm[
                frame_indices,
                point_index,
            ].copy()

            trajectory_counts = valid_counts[
                frame_indices,
                point_index,
            ]

            trajectory_valid = np.isfinite(trajectory).all(axis=1) & (
                trajectory_counts >= effective_minimum_count
            )

            trajectory[:, 1] *= -1.0

            # Split trajectories at invalid frames rather than connecting
            # across missing data.
            valid_indices = np.flatnonzero(trajectory_valid)

            if len(valid_indices) < 2:
                continue

            consecutive_groups = np.split(
                valid_indices,
                np.flatnonzero(np.diff(valid_indices) > 1) + 1,
            )

            for group in consecutive_groups:
                if len(group) < 2:
                    continue

                axis.plot(
                    trajectory[group, 0],
                    trajectory[group, 1],
                    color="0.55",
                    linewidth=0.5,
                    alpha=0.35,
                    zorder=1,
                )

    if show_points and point_coordinates:
        point_array = np.asarray(point_coordinates)

        axis.scatter(
            point_array[:, 0],
            point_array[:, 1],
            c=np.asarray(point_times),
            cmap=cmap,
            norm=norm,
            s=point_size,
            alpha=min(
                1.0,
                alpha + 0.15,
            ),
            linewidths=0,
            zorder=4,
        )

    # Mark the fish-centered origin.
    axis.scatter(
        [0.0],
        [0.0],
        marker="+",
        s=80,
        linewidths=1.2,
        color="black",
        zorder=6,
    )

    colorbar = figure.colorbar(
        ScalarMappable(
            norm=norm,
            cmap=cmap,
        ),
        ax=axis,
        pad=0.02,
        fraction=0.045,
    )

    colorbar.set_label("Time relative to event onset (ms)")

    if time_min <= 0 <= time_max:
        colorbar.ax.axhline(
            0.0,
            color="black",
            linewidth=1.0,
        )

    combined_coordinates = np.concatenate(
        all_display_coordinates,
        axis=0,
    )

    if extent_mm is not None:
        axis.set_xlim(
            -extent_mm,
            extent_mm,
        )
        axis.set_ylim(
            -extent_mm,
            extent_mm,
        )
    else:
        minimum = np.nanmin(
            combined_coordinates,
            axis=0,
        )
        maximum = np.nanmax(
            combined_coordinates,
            axis=0,
        )

        x_min = float(min(minimum[0], -0.25) - padding_mm)
        x_max = float(max(maximum[0], 0.25) + padding_mm)
        y_min = float(min(minimum[1], -0.25) - padding_mm)
        y_max = float(max(maximum[1], 0.25) + padding_mm)

        # Use equal x/y ranges so shape is not visually distorted.
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)

        span = max(
            x_max - x_min,
            y_max - y_min,
        )

        axis.set_xlim(
            x_center - span / 2.0,
            x_center + span / 2.0,
        )
        axis.set_ylim(
            y_center - span / 2.0,
            y_center + span / 2.0,
        )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.set_xlabel("Lateral position (mm)")
    axis.set_ylabel("Forward position (mm)")

    axis.axvline(
        0.0,
        color="0.85",
        linewidth=0.7,
        zorder=0,
    )
    axis.axhline(
        0.0,
        color="0.85",
        linewidth=0.7,
        zorder=0,
    )

    axis.grid(
        color="0.92",
        linewidth=0.5,
        zorder=0,
    )

    axis.set_title(
        make_title(
            source_path,
            aggregate,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )

    plt.close(figure)

    return {
        "source_npz": str(source_path),
        "output_image": str(output_path),
        "used_events": used_events,
        "statistic": aggregate["statistic"],
        "selected_frames": len(frame_indices),
        "stride": stride,
        "minimum_valid_count": (effective_minimum_count),
        "time_start_ms": time_min,
        "time_stop_ms": time_max,
        "cmap": cmap,
    }


# ============================================================================
# Output paths
# ============================================================================


def output_path_for(
    input_file: Path,
    input_root: Path,
    output_directory: Path | None,
    image_format: str,
) -> Path:
    """Determine the output image path."""
    filename = f"{input_file.stem}" f"__time_color.{image_format}"

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
            "Render aggregate body-and-eye skeletons as "
            "single images with time represented by color."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "An aggregate-keypoint NPZ file or a directory "
            "containing aggregate NPZ files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. By default, images are "
            "written beside their source NPZ files."
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
        "--stride",
        type=int,
        default=4,
        help=(
            "Plot every Nth aggregate frame. Event onset and "
            "the last frame are always included. Default: 4"
        ),
    )

    parser.add_argument(
        "--minimum-valid-count",
        type=int,
        default=1,
        help=(
            "Minimum number of contributing events needed to "
            "draw a keypoint. Default: 1"
        ),
    )

    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=0.25,
        help=(
            "Minimum fraction of usable events needed to draw "
            "a keypoint. Default: 0.25"
        ),
    )

    parser.add_argument(
        "--cmap",
        default="turbo",
        help=(
            "Matplotlib colormap. Examples: turbo, viridis, "
            "coolwarm, plasma. Default: turbo"
        ),
    )

    parser.add_argument(
        "--body-line-width",
        type=float,
        default=1.5,
        help=("Body skeleton line width. Default: 1.5"),
    )

    parser.add_argument(
        "--eye-line-width",
        type=float,
        default=3.0,
        help=("Eye-axis line width. Default: 3"),
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help=("Keypoint marker area. Default: 8"),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help=("Skeleton opacity. Default: 0.55"),
    )

    parser.add_argument(
        "--hide-points",
        action="store_true",
        help="Draw only skeleton edges.",
    )

    parser.add_argument(
        "--show-trajectories",
        action="store_true",
        help=("Also draw faint trajectories for every keypoint."),
    )

    parser.add_argument(
        "--padding-mm",
        type=float,
        default=0.5,
        help=("Padding around automatically selected limits. " "Default: 0.5 mm"),
    )

    parser.add_argument(
        "--extent-mm",
        type=float,
        default=None,
        help=(
            "Use fixed symmetric limits from -extent to +extent "
            "on both axes. Useful for comparing categories."
        ),
    )

    parser.add_argument(
        "--figure-size",
        type=float,
        default=7.0,
        help=("Square figure size in inches. Default: 7"),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output resolution. Default: 200",
    )

    parser.add_argument(
        "--format",
        choices=(
            "png",
            "pdf",
            "svg",
        ),
        default="png",
        help="Output format. Default: png",
    )

    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Use a transparent figure background.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output images.",
    )

    return parser


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate command-line arguments."""
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")

    if args.minimum_valid_count < 1:
        raise ValueError("--minimum-valid-count must be at least 1")

    if not (0.0 <= args.minimum_valid_fraction <= 1.0):
        raise ValueError("--minimum-valid-fraction must be between 0 and 1")

    if args.body_line_width <= 0:
        raise ValueError("--body-line-width must be positive")

    if args.eye_line_width <= 0:
        raise ValueError("--eye-line-width must be positive")

    if args.point_size < 0:
        raise ValueError("--point-size must be non-negative")

    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("--alpha must be in (0, 1]")

    if args.padding_mm < 0:
        raise ValueError("--padding-mm must be non-negative")

    if args.extent_mm is not None and args.extent_mm <= 0:
        raise ValueError("--extent-mm must be positive")

    if args.figure_size <= 0:
        raise ValueError("--figure-size must be positive")

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    # Validate the requested colormap early.
    if args.cmap not in plt.colormaps():
        raise ValueError(f"Unknown colormap {args.cmap!r}.")


def main() -> None:
    """Render all requested time-colored images."""
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
    print("Time-color rendering configuration")
    print(f"  stride: {args.stride}")
    print(f"  colormap: {args.cmap}")
    print(f"  alpha: {args.alpha:g}")
    print(f"  minimum valid fraction: " f"{args.minimum_valid_fraction:g}")

    summaries: list[dict[str, Any]] = []

    for input_file in tqdm(
        input_files,
        desc="Time-colored skeletons",
        unit="file",
    ):
        output_path = output_path_for(
            input_file=input_file,
            input_root=input_path,
            output_directory=(output_directory),
            image_format=args.format,
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
                    "No skeleton edges could be inferred. "
                    "Supply --skeleton-edges explicitly."
                )

            summary = plot_time_colored_skeleton(
                source_path=input_file,
                output_path=output_path,
                aggregate=aggregate,
                edges=edges,
                stride=args.stride,
                minimum_valid_count=(args.minimum_valid_count),
                minimum_valid_fraction=(args.minimum_valid_fraction),
                cmap=args.cmap,
                body_line_width=(args.body_line_width),
                eye_line_width=(args.eye_line_width),
                point_size=args.point_size,
                alpha=args.alpha,
                show_points=(not args.hide_points),
                show_trajectories=(args.show_trajectories),
                padding_mm=args.padding_mm,
                extent_mm=args.extent_mm,
                dpi=args.dpi,
                figure_size=args.figure_size,
                transparent=args.transparent,
            )

            summaries.append(summary)

            tqdm.write(f"[saved] {output_path}")

        except Exception as error:
            tqdm.write(f"[skip] {input_file}: {error}")

    if output_directory is None:
        summary_directory = input_path if input_path.is_dir() else input_path.parent
    else:
        summary_directory = output_directory

    summary_path = summary_directory / "skeleton_timecolor_summary.csv"

    pd.DataFrame.from_records(summaries).to_csv(
        summary_path,
        index=False,
    )

    print()
    print(f"Rendered {len(summaries):,} " "time-colored skeleton images")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
