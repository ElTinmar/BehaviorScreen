"""
Fish per line and condition in the screen's data folder after quality control (qc_all.csv
and screen.yaml's common_filters, as the screen's loaders apply them), and the recording
time they add up to. Recordings of the phototaxis protocol alone are re-runs, not more fish:
they are counted apart.

    python -m BehaviorScreen.screen.ablation.sample_sizes [--data-root DIR] [-o line_sample_sizes.png]
"""
import argparse
from pathlib import Path

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import blended_transform_factory

from BehaviorScreen.screen.ablation.line_names import line_name
from BehaviorScreen.core import Stim
from BehaviorScreen.stim_specs import apply_table_filters, exclude_qc_fish, load_yaml_config

RECORDING_S = 4490  # one fish's protocol
WILD_TYPES = {"wt", "ab", "wik"}  # WT (TL) and the backgrounds
COLORS = {"vehicle": "#2a78d6", "ronidazole": "#eb6834", "danieau": "#1baf7a"}  # in their order within a line
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7", "surface": "#fcfcfb"}
RERUN_STIMULI = {int(Stim.PHOTOTAXIS), int(Stim.DARK), int(Stim.BRIGHT)}  # all a phototaxis re-run shows
COMMON_FILTERS = load_yaml_config(Path(__file__).parents[2] / "screen.yaml")["common_filters"]


def fish_per_arm(root: Path) -> pd.DataFrame:
    """Per (line, condition), its numbers of fish (full protocol) and of phototaxis re-runs that pass quality control."""
    trials = exclude_qc_fish(pd.read_csv(root / "valid_trials_all.csv", dtype={"line": str, "condition": str}), root / "qc_all.csv")
    trials = apply_table_filters(trials, {"recording": COMMON_FILTERS}, ("recording",))
    presented = trials[trials["presented"].astype(bool)]  # valid_trials lists the whole protocol: what each recording showed
    recordings = presented.groupby(["line", "condition", "file"])["stim"].agg(lambda stim: set(stim) <= RERUN_STIMULI).rename("rerun").reset_index()
    counts = recordings.groupby(["line", "condition"])["rerun"].agg(fish=lambda rerun: int((~rerun).sum()), reruns="sum")
    return counts.reset_index()


def line_groups(counts: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Lines with fish, by group: WT and the wild-type backgrounds (WILD_TYPES, WT first), NTR lines
    (vehicle or ronidazole fish) and mutants; each sorted by name.
    """
    lines = set(counts.loc[counts["fish"] > 0, "line"])
    wild = {line for line in lines if line.lower() in WILD_TYPES}
    ntr = set(counts.loc[(counts["fish"] > 0) & counts["condition"].isin(["vehicle", "ronidazole"]), "line"]) - wild
    by_name = lambda group: sorted(group, key=lambda line: (line != "WT", line.lower()))
    return {"WT": by_name(wild), "NTR lines": by_name(ntr), "Mutants": by_name(lines - wild - ntr)}


def lines_tested(counts: pd.DataFrame) -> str:
    groups = line_groups(counts)
    wild, ntr, mutants = len(groups["WT"]), len(groups["NTR lines"]), len(groups["Mutants"])
    return f"{wild + ntr + mutants} lines tested ({wild} WT, {ntr} NTR, {mutants} mutant{'s' if mutants != 1 else ''})"


def _column(ax: plt.Axes, x: float, value: float, width: float, color: str, radius_pt: float = 3.0) -> None:
    """A column from 0 to value, its top rounded (radius_pt, in points), square at the baseline."""
    scale_x, scale_y = np.abs(ax.transData.transform((1, 1)) - ax.transData.transform((0, 0)))
    radius = radius_pt * ax.figure.dpi / 72
    rx, ry = min(radius / scale_x, width / 2), min(radius / scale_y, value / 2)
    arc = np.linspace(0, np.pi / 2, 8)
    left, right = x - width / 2, x + width / 2
    corner = lambda x0, sign: np.column_stack([x0 + sign * rx * (1 - np.cos(arc)), value - ry + ry * np.sin(arc)])
    outline = np.vstack([[[left, 0]], corner(left, 1), corner(right, -1)[::-1], [[right, 0]]])
    ax.add_patch(Polygon(outline, closed=True, color=color, lw=0))


def plot_sample_sizes(counts: pd.DataFrame, width: float = 0.36, gap: float = 0.05, max_labelled: int = 20) -> plt.Figure:
    """
    A 16:9 slide: one slot per line (its short name), grouped (WT, NTR lines, mutants), its conditions'
    fish as columns, with their counts on top when there are at most max_labelled lines.
    """
    groups = {name: lines for name, lines in line_groups(counts).items() if lines}
    headers = {"WT": "WT", "NTR lines": "NTR lines", "Mutants": "Mutants"}
    fish = counts.set_index(["line", "condition"])["fish"]
    half = lambda n: (n * width + (n - 1) * gap) / 2  # half the width of a slot's n columns
    slots, spans, x = [], [], 0.0  # (x, line, conditions); (left, right, header) per group
    for name, lines in groups.items():
        start = len(slots)
        for line in lines:
            slots.append((x, line, [c for c in COLORS if fish.get((line, c), 0) > 0]))
            x += 1.0
        (first, _, first_columns), (last, _, last_columns) = slots[start], slots[-1]
        spans.append((first - half(len(first_columns)), last + half(len(last_columns)), headers[name]))
        x += 0.6
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]}):
        fig, ax = plt.subplots(figsize=(13.33, 7.5), facecolor=INK["surface"])
        ax.set_facecolor(INK["surface"])
        y_max = fish.max() * 1.1
        ax.set(xlim=(-0.6, x - 0.6), ylim=(0, y_max))
        ax.set_xticks([x_slot for x_slot, _, _ in slots], [line_name(line) for _, line, _ in slots], rotation=45, ha="right",
                      rotation_mode="anchor", color=INK["primary"], fontsize=9)
        ax.tick_params(axis="x", length=0, pad=4)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.8, bottom=0.14)
        fig.canvas.draw()  # the bottom margin fits the longest name
        names = max(label.get_window_extent().height for label in ax.get_xticklabels()) + 4 * fig.dpi / 72
        fig.subplots_adjust(bottom=0.03 + names / fig.bbox.height)
        for x_slot, line, conditions in slots:
            for k, condition in enumerate(conditions):
                x_col, value = x_slot + (k - (len(conditions) - 1) / 2) * (width + gap), fish[(line, condition)]
                _column(ax, x_col, value, width, COLORS[condition])
                if len(slots) <= max_labelled:
                    ax.text(x_col, value + 0.012 * y_max, f"{value}", color=INK["secondary"], fontsize=7, ha="center", va="bottom")
        above = blended_transform_factory(ax.transData, ax.transAxes)
        for left, right, header in spans:
            ax.plot([left, right], [1.02, 1.02], transform=above, color=INK["baseline"], lw=0.8, clip_on=False)
            ax.text(left, 1.035, header.upper(), transform=above, color=INK["muted"], fontsize=8, fontweight="bold", va="bottom")
        ax.grid(axis="y", color=INK["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylabel("Fish after QC", color=INK["secondary"], fontsize=9)
        ax.tick_params(axis="y", colors=INK["muted"], labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK["baseline"])
        present = [c for c in COLORS if c in set(counts.loc[counts["fish"] > 0, "condition"])]
        days, rest = divmod(int(counts["fish"].sum()) * RECORDING_S // 60, 24 * 60)
        fig.text(0.06, 0.95, f"{lines_tested(counts)}, {int(counts['fish'].sum())} fish after QC",
                 color=INK["primary"], fontsize=18, fontweight="bold", va="top")
        fig.text(0.06, 0.895, f"{days} days, {rest // 60} hours and {rest % 60} minutes of recording", color=INK["secondary"], fontsize=11, va="top")
        fig.legend([Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in present],
                   [f"{c} ({int(fish.xs(c, level='condition').sum())})" for c in present], loc="upper right", bbox_to_anchor=(0.985, 0.95),
                   ncol=len(present), frameon=False, fontsize=10, labelcolor=INK["secondary"], handlelength=1.0, handleheight=0.8, columnspacing=1.4)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("/media/martin/DATA_18TB/Screen"), help="Folder of the merged *_all.csv tables.")
    parser.add_argument("-o", "--output", type=Path, default=Path("line_sample_sizes.png"))
    args = parser.parse_args()
    counts = fish_per_arm(args.data_root)
    print(counts.set_index(["line", "condition"]).to_string())
    print(lines_tested(counts))
    plot_sample_sizes(counts).savefig(args.output, dpi=300)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
