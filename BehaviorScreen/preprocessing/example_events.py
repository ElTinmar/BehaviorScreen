"""
One-off: example events per category from one experiment folder, drawn like
Jouary et al. Fig. 1b (bouts) and Dowell et al. Fig. 1F (saccades).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from megabouts.utils import bouts_category_color, bouts_category_name_short

from BehaviorScreen.core import SACCADE_CLASS_COLORS, SACCADE_CLASS_NAMES
from BehaviorScreen.preprocessing.load import Directories, find_files, parse_fish
from BehaviorScreen.preprocessing.bouts import MegaboutResults, load_megabout
from BehaviorScreen.preprocessing.recording import Recording
from BehaviorScreen.stim_specs import apply_event_filters, exclude_qc_fish, load_yaml_config

TAIL_SEGMENT = 6  # 7th of 10 tail angles
TAIL_WINDOW_S = (-0.04, 0.18)  # around the first half-beat
DOWELL_ORDER = [1, 2, 4, 8, 7, 3, 6]  # ConjL, ConjR, Conv, BConvL, BConvR, ConvMini, Div
SACCADE_BASELINE_MS = (-200, 0)


def select(events: pd.DataFrame, args: argparse.Namespace, kind: str) -> pd.DataFrame:
    """Same fish and events as the heatmaps: QC exclusions and screen.yaml filters."""
    events = exclude_qc_fish(events, args.root / args.qc_csv)
    events = apply_event_filters(events, load_yaml_config(args.yaml), kind)
    if args.line:
        events = events[events.file.map(lambda f: parse_fish(f).line == args.line)]
    return events


def sample_per_category(events: pd.DataFrame, column: str, n: int, seed: int) -> pd.DataFrame:
    return events.sample(frac=1, random_state=seed).groupby(column).head(n)


def bout_traces(row: pd.Series, mb: MegaboutResults, n_pre: int, n_post: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Tail angle (deg) on a fixed window around the first half-beat (NaN-padded at recording edges), and the head
    trajectory (2 x T, mm) with the fish at the origin facing +x; both flipped so the first half-beat is positive.
    """
    angle = np.rad2deg(mb.tail.angle_smooth[:, TAIL_SEGMENT])
    on, off, hb1 = int(row.frame_start), int(row.frame_stop), int(mb.bouts.HB1[row.bout_index])
    tail = np.full(n_pre + n_post, np.nan)
    lo, hi = max(hb1 - n_pre, 0), min(hb1 + n_post, len(angle))
    tail[lo - hb1 + n_pre : hi - hb1 + n_pre] = angle[lo:hi]
    flip = -1 if angle[hb1] < 0 else 1

    traj = mb.traj
    dx, dy = traj.x_smooth[on:off] - traj.x_smooth[on], traj.y_smooth[on:off] - traj.y_smooth[on]
    c, s = np.cos(-traj.yaw_smooth[on]), np.sin(-traj.yaw_smooth[on])
    return tail * flip, np.vstack((c * dx - s * dy, (s * dx + c * dy) * flip))


def arrow_path(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, head_mm: float = 1.0) -> None:
    """
    Path ending in a filled arrowhead drawn in data units. The line stops at the base of the head and the head
    points along that last stretch of line, so the two are aligned and the line never shows past the tip.
    """
    if len(x) < 2:
        return
    remaining = np.cumsum(np.hypot(np.diff(x), np.diff(y))[::-1])[::-1]  # path length from each point to the end
    cut = int(np.flatnonzero(remaining >= head_mm)[-1]) if remaining[0] >= head_mm else 0
    tip = np.array([x[-1], y[-1]])
    direction = tip - (x[cut], y[cut])
    direction /= np.linalg.norm(direction)
    base = tip - head_mm * direction
    normal = np.array([-direction[1], direction[0]]) * head_mm * 0.4
    ax.plot([*x[: cut + 1], base[0]], [*y[: cut + 1], base[1]], color=color, lw=1.2, solid_capstyle="butt")
    ax.add_patch(Polygon([tip, base + normal, base - normal], closed=True, color=color, lw=0))


def scale_bar(ax: plt.Axes, x: float, y: float, dx: float, x_label: str, dy: float = 0, y_label: str = "") -> None:
    ax.plot([x, x + dx], [y, y], color="k", lw=1)
    ax.annotate(x_label, (x + dx / 2, y), xytext=(0, -2), textcoords="offset points", ha="center", va="top", fontsize=7)
    if dy:
        ax.plot([x, x], [y, y + dy], color="k", lw=1)
        ax.annotate(y_label, (x, y + dy / 2), xytext=(-2, 0), textcoords="offset points", ha="right", va="center",
                    rotation=90, fontsize=7)


def plot_bouts(args: argparse.Namespace) -> plt.Figure:
    bouts = select(pd.read_csv(args.root / "bouts.csv", low_memory=False), args, "bout")
    sample = sample_per_category(bouts, "category", args.n, args.seed)

    dirs = Directories(args.root, metadata="results", stimuli="results", tracking="results",
                       temperature="results", video="results", video_timestamp="results")
    files_by_name = {files.metadata.stem: files for files in find_files(dirs)}

    tails: dict[int, list[np.ndarray]] = {}
    trajectories: dict[int, list[np.ndarray]] = {}
    probas: dict[int, list[float]] = {}
    for file, rows in sample.groupby("file"):
        rec = Recording(files_by_name[file])
        mb = load_megabout(rec)
        if mb is None:
            continue
        n_pre, n_post = round(-TAIL_WINDOW_S[0] * rec.fps), round(TAIL_WINDOW_S[1] * rec.fps)
        for row in rows.itertuples():
            tail, trajectory = bout_traces(row, mb, n_pre, n_post)
            tails.setdefault(row.category, []).append(tail)
            trajectories.setdefault(row.category, []).append(trajectory)
            probas.setdefault(row.category, []).append(row.proba)
    time_ms = np.arange(-n_pre, n_post) / rec.fps * 1000

    categories = sorted(tails)  # category indices follow the Megabouts order
    n_columns = -(-(len(categories) + 1) // 2)  # two rows; one extra cell for the scale bar
    fig = plt.figure(figsize=(1.1 * n_columns + 2.2, 2.4))
    grid = fig.add_gridspec(2, n_columns + 1, width_ratios=[1] * n_columns + [2], wspace=0.05, hspace=0.2)
    tail_axes = [fig.add_subplot(grid[i // n_columns, i % n_columns]) for i in range(len(categories) + 1)]
    traj_ax = fig.add_subplot(grid[:, -1], aspect="equal")
    for ax in [*tail_axes, traj_ax]:
        ax.axis("off")

    traces = {c: np.stack(tails[c]) for c in categories}
    typical = {c: int(np.argmax(probas[c])) for c in categories}  # most confidently classified bout of the pool
    # shared y scale: rare faint traces beyond the 0.25-99.75th percentiles are clipped, bold traces never are
    shown = np.concatenate([traces[c][: args.k].ravel() for c in categories])
    bold = np.concatenate([traces[c][typical[c]] for c in categories])
    low, high = np.nanpercentile(shown, [0.25, 99.75])
    ylim = (min(low, np.nanmin(bold)), max(high, np.nanmax(bold)))
    for ax, category in zip(tail_axes, categories):
        color = bouts_category_color[category]
        ax.plot(time_ms, traces[category][: args.k].T, color=color, alpha=args.alpha, lw=0.5)  # the sample is already shuffled
        ax.plot(time_ms, traces[category][typical[category]], color=color, lw=1.5)
        ax.set(xlim=(time_ms[0], time_ms[-1]), ylim=ylim)
        ax.set_title(bouts_category_name_short[category], fontsize=8, color=color, pad=0)
        arrow_path(traj_ax, *trajectories[category][typical[category]], color)

    corner = tail_axes[-1]
    corner.set(xlim=(time_ms[0], time_ms[-1]), ylim=ylim)
    scale_bar(corner, 0, ylim[0] / 2, 100, "100 ms", 90, "90°")
    corner.set_title("aligned on\nfirst half-beat", fontsize=6, pad=0)
    traj_ax.autoscale_view()  # arrowheads are patches: include them in the limits before placing the scale bar
    scale_bar(traj_ax, traj_ax.get_xlim()[0], traj_ax.get_ylim()[0], 1, "1 mm")
    return fig


def plot_saccades(args: argparse.Namespace) -> plt.Figure:
    # saccades.npz is row-aligned with saccades.csv, hence with saccades_augmented.csv
    saccades = pd.read_csv(args.root / "saccades_augmented.csv", low_memory=False)
    with np.load(args.root / "saccades.npz") as npz:
        left, right, time_ms = npz["L_smooth"], npz["R_smooth"], npz["time_axis_ms"]
    if len(saccades) != len(left):
        raise ValueError("saccades_augmented.csv and saccades.npz are not row-aligned")

    saccades = select(saccades, args, "saccade")
    sample = sample_per_category(saccades[saccades.cluster.isin(DOWELL_ORDER)], "cluster", args.n, args.seed)
    baseline = (time_ms >= SACCADE_BASELINE_MS[0]) & (time_ms < SACCADE_BASELINE_MS[1])

    fig, axes = plt.subplots(1, len(DOWELL_ORDER), figsize=(3 * len(DOWELL_ORDER), 3.5), sharex=True, sharey=True)
    for ax, cluster in zip(axes, DOWELL_ORDER):
        rows = sample.index[sample.cluster == cluster]
        for traces, color in ((left, "royalblue"), (right, "red")):
            snippets = traces[rows] - np.nanmedian(traces[rows][:, baseline], axis=1, keepdims=True)
            ax.plot(time_ms, snippets.T, color=color, alpha=0.1, lw=0.5)
            ax.plot(time_ms, np.nanmean(snippets, axis=0), color=color, lw=2)
        ax.set_title(f"{SACCADE_CLASS_NAMES[cluster]}\n(n={len(rows)})", color=SACCADE_CLASS_COLORS[cluster])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(xlim=(-100, 300), ylim=(-50, 50), xlabel="time from onset (ms)", ylabel="eye position (°)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--n", type=int, default=100, help="events sampled per category")
    parser.add_argument("--k", type=int, default=100, help="random bouts drawn per category, plus the most confidently classified one in bold")
    parser.add_argument("--alpha", type=float, default=0.075, help="opacity of the faint bout traces")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--line", default=None, help="keep only this line (from the file name), e.g. WT")
    parser.add_argument("--qc-csv", default="qc.csv")
    parser.add_argument("--yaml", type=Path, default=Path("BehaviorScreen/screen.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots"), help="relative to root")
    args = parser.parse_args()

    output_dir = args.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_bouts(args).savefig(output_dir / "example_bouts.png", dpi=200, bbox_inches="tight")
    plot_saccades(args).savefig(output_dir / "example_saccades.png", dpi=200)
