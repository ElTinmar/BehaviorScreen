"""
Plot raw or smoothed eye-position traces grouped by saccade cluster.

Examples
--------
Plot raw traces:

    python plot_clusters.py saccades_classified.csv \
        --npz saccades.npz \
        --trace-type raw

Plot baseline-corrected smoothed traces:

    python plot_clusters.py saccades_classified.csv \
        --npz saccades.npz \
        --trace-type smooth \
        --baseline-correct \
        --xlim -100 300 \
        --ylim -50 50 \
        --output classified_clusters.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_CLUSTER_NAMES = {
    -1: "Unassigned",
    0: "Unclassified",
    1: "Conjugate left",
    2: "Conjugate right",
    3: "Miniature convergent",
    4: "Convergent",
    5: "Non-saccadic",
    6: "Divergent",
    7: "Biphasic convergent right",
    8: "Biphasic convergent left",
}


def baseline_correct_traces(
    traces: np.ndarray,
    time_axis_ms: np.ndarray,
    baseline_window_ms: tuple[float, float],
) -> np.ndarray:
    """Subtract each event's median position during a pre-onset interval."""
    start_ms, stop_ms = baseline_window_ms

    if start_ms >= stop_ms:
        raise ValueError(
            "The baseline start must be smaller than the baseline stop."
        )

    baseline_mask = (
        (time_axis_ms >= start_ms)
        & (time_axis_ms < stop_ms)
    )

    if not baseline_mask.any():
        raise ValueError(
            "No samples lie in the baseline interval "
            f"[{start_ms}, {stop_ms}) ms."
        )

    baseline = np.nanmedian(
        traces[:, baseline_mask],
        axis=1,
        keepdims=True,
    )

    return traces - baseline


def load_traces(
    npz_path: Path,
    trace_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load left-eye, right-eye, and time arrays from a trace NPZ file."""
    left_key = f"L_{trace_type}"
    right_key = f"R_{trace_type}"

    with np.load(npz_path) as data:
        required_keys = {
            left_key,
            right_key,
            "time_axis_ms",
        }
        missing_keys = required_keys.difference(data.files)

        if missing_keys:
            raise ValueError(
                f"{npz_path} is missing arrays: "
                f"{sorted(missing_keys)}. "
                f"Available arrays: {data.files}"
            )

        # Copy the arrays so they remain valid after the NPZ file closes.
        left = np.asarray(
            data[left_key],
            dtype=float,
        ).copy()
        right = np.asarray(
            data[right_key],
            dtype=float,
        ).copy()
        time_axis_ms = np.asarray(
            data["time_axis_ms"],
            dtype=float,
        ).squeeze().copy()

    return left, right, time_axis_ms


def validate_inputs(
    events: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
    time_axis_ms: np.ndarray,
    csv_path: Path,
    npz_path: Path,
) -> None:
    """Validate the event table and trace arrays."""
    if "cluster" not in events.columns:
        raise ValueError(
            f"{csv_path} does not contain a 'cluster' column."
        )

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            "Trace arrays must be two-dimensional. "
            f"Got left={left.shape}, right={right.shape}."
        )

    if left.shape != right.shape:
        raise ValueError(
            "Left and right trace shapes differ: "
            f"{left.shape} versus {right.shape}."
        )

    if time_axis_ms.ndim != 1:
        raise ValueError(
            "The time axis must be one-dimensional. "
            f"Got shape {time_axis_ms.shape}."
        )

    if left.shape[1] != len(time_axis_ms):
        raise ValueError(
            f"Trace length ({left.shape[1]}) does not match "
            f"time-axis length ({len(time_axis_ms)})."
        )

    if len(events) != len(left):
        raise ValueError(
            f"The CSV contains {len(events)} events, but {npz_path} "
            f"contains {len(left)} snippets. The files are not "
            "row-aligned."
        )


def plot_cluster_traces(
    csv_path: Path,
    npz_path: Path,
    trace_type: str = "raw",
    max_per_cluster: int = 100,
    output_path: Path | None = None,
    seed: int = 0,
    baseline_correct: bool = False,
    baseline_window_ms: tuple[float, float] = (-200.0, 0.0),
    xlim: tuple[float, float] | None = (-100.0, 300.0),
    ylim: tuple[float, float] | None = (-50.0, 50.0),
    interactive: bool = False,
) -> None:
    """
    Plot individual and median eye-position traces for each cluster.

    Parameters
    ----------
    csv_path
        Classified event CSV containing a ``cluster`` column.
    npz_path
        Companion NPZ containing ``L_raw``, ``R_raw``, ``L_smooth``,
        ``R_smooth``, and ``time_axis_ms`` arrays.
    trace_type
        Either ``"raw"`` or ``"smooth"``.
    max_per_cluster
        Maximum number of individual events displayed per cluster.
        Median traces are calculated from all events in the cluster.
    output_path
        Optional path at which to save the figure.
    seed
        Random seed used to select individual traces for display.
    baseline_correct
        If true, subtract each event's median pre-onset eye position.
    baseline_window_ms
        Start and stop of the baseline interval in milliseconds.
    xlim
        Displayed time limits in milliseconds. Use ``None`` for automatic
        limits.
    ylim
        Displayed eye-angle limits in degrees. Use ``None`` for automatic
        limits.
    interactive
        If true, display the figure after saving it. By default, the
        figure is saved and closed without opening an interactive window.
    """
    if max_per_cluster < 0:
        raise ValueError("max_per_cluster must be non-negative.")

    events = pd.read_csv(csv_path)

    left, right, time_axis_ms = load_traces(
        npz_path=npz_path,
        trace_type=trace_type,
    )

    validate_inputs(
        events=events,
        left=left,
        right=right,
        time_axis_ms=time_axis_ms,
        csv_path=csv_path,
        npz_path=npz_path,
    )

    if baseline_correct:
        left = baseline_correct_traces(
            traces=left,
            time_axis_ms=time_axis_ms,
            baseline_window_ms=baseline_window_ms,
        )
        right = baseline_correct_traces(
            traces=right,
            time_axis_ms=time_axis_ms,
            baseline_window_ms=baseline_window_ms,
        )

    cluster_labels = (
        pd.to_numeric(
            events["cluster"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )
    cluster_ids = sorted(np.unique(cluster_labels))

    if len(cluster_ids) == 0:
        raise ValueError("No cluster labels were found.")

    n_columns = min(4, len(cluster_ids))
    n_rows = int(np.ceil(len(cluster_ids) / n_columns))

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.4 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    random_generator = np.random.default_rng(seed)

    for axis, cluster_id in zip(axes, cluster_ids):
        cluster_indices = np.flatnonzero(
            cluster_labels == cluster_id
        )

        if len(cluster_indices) > max_per_cluster:
            plotted_indices = random_generator.choice(
                cluster_indices,
                size=max_per_cluster,
                replace=False,
            )
        else:
            plotted_indices = cluster_indices

        for event_index in plotted_indices:
            axis.plot(
                time_axis_ms,
                left[event_index],
                color="royalblue",
                alpha=0.08,
                linewidth=0.5,
            )
            axis.plot(
                time_axis_ms,
                right[event_index],
                color="lightcoral",
                alpha=0.08,
                linewidth=0.5,
            )

        median_left = np.nanmedian(
            left[cluster_indices],
            axis=0,
        )
        median_right = np.nanmedian(
            right[cluster_indices],
            axis=0,
        )

        axis.plot(
            time_axis_ms,
            median_left,
            color="navy",
            linewidth=2.2,
            label="Left eye median",
        )
        axis.plot(
            time_axis_ms,
            median_right,
            color="darkred",
            linewidth=2.2,
            label="Right eye median",
        )

        axis.axvline(
            0,
            color="black",
            linestyle="--",
            alpha=0.65,
            linewidth=1,
        )
        axis.axhline(
            0,
            color="black",
            linestyle=":",
            alpha=0.25,
            linewidth=0.8,
        )

        cluster_name = PAPER_CLUSTER_NAMES.get(
            int(cluster_id),
            f"Unknown cluster {cluster_id}",
        )

        axis.set_title(
            f"{cluster_id}: {cluster_name}\n"
            f"n={len(cluster_indices):,}",
            fontsize=10,
        )

        if xlim is not None:
            axis.set_xlim(*xlim)

        if ylim is not None:
            axis.set_ylim(*ylim)

    for axis in axes[len(cluster_ids) :]:
        axis.set_visible(False)

    axes[0].legend(
        fontsize=8,
        frameon=False,
        loc="best",
    )

    if baseline_correct:
        y_label = "Baseline-corrected eye angle (deg)"
        baseline_description = (
            f", baseline {baseline_window_ms[0]:g} to "
            f"{baseline_window_ms[1]:g} ms"
        )
    else:
        y_label = "Eye angle (deg)"
        baseline_description = ""

    figure.supxlabel("Time relative to onset (ms)")
    figure.supylabel(y_label)
    figure.suptitle(
        f"{trace_type.capitalize()} eye traces by published class"
        f"{baseline_description}",
        fontsize=14,
    )
    figure.tight_layout()

    if output_path is None:
        output_path = csv_path.with_name(
            f"{csv_path.stem}_clusters.png"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    print(f"Saved figure to {output_path}")

    if interactive:
        plt.show()
    else:
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw or smoothed eye-position traces grouped by "
            "published saccade class."
        )
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Classified saccade CSV.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help=(
            "Original companion NPZ containing event traces. "
            "Defaults to the CSV path with a .npz extension."
        ),
    )
    parser.add_argument(
        "--trace-type",
        choices=("raw", "smooth"),
        default="raw",
        help="Type of eye-position trace to plot. Default: raw.",
    )
    parser.add_argument(
        "--max-per-cluster",
        type=int,
        default=100,
        help=(
            "Maximum number of individual traces displayed per cluster. "
            "Default: 100."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling traces. Default: 0.",
    )
    parser.add_argument(
        "--baseline-correct",
        action="store_true",
        help="Subtract each event's pre-onset median eye position.",
    )
    parser.add_argument(
        "--baseline-window-ms",
        type=float,
        nargs=2,
        metavar=("START", "STOP"),
        default=(-200.0, 0.0),
        help=(
            "Baseline interval in milliseconds. "
            "Default: -200 0."
        ),
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-100.0, 300.0),
        help=(
            "Displayed time range in milliseconds. "
            "Default: -100 300."
        ),
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-50.0, 50.0),
        help=(
            "Displayed eye-angle range in degrees. "
            "Default: -50 50."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Defaults to "
            "<csv_stem>_clusters.png beside the input CSV."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Display the figure after saving it.",
    )

    return parser


def main() -> None:
    """Run the command-line application."""
    args = build_parser().parse_args()

    if args.npz is None:
        npz_path = args.csv.with_suffix(".npz")
    else:
        npz_path = args.npz

    plot_cluster_traces(
        csv_path=args.csv,
        npz_path=npz_path,
        trace_type=args.trace_type,
        max_per_cluster=args.max_per_cluster,
        output_path=args.output,
        seed=args.seed,
        baseline_correct=args.baseline_correct,
        baseline_window_ms=tuple(args.baseline_window_ms),
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()