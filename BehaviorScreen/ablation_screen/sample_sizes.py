"""
Fish per line and condition in the screen's data folder after quality control (qc_all.csv
and screen.yaml's common_filters, as the screen's loaders apply them), and the recording
time they add up to. Recordings of the phototaxis protocol alone are re-runs, not more fish:
they are counted apart.

    python -m BehaviorScreen.ablation_screen.sample_sizes [--data-root DIR] [-o line_sample_sizes.png]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BehaviorScreen.core import Stim
from BehaviorScreen.stim_specs import apply_table_filters, exclude_qc_fish, load_yaml_config

RECORDING_S = 4490  # one fish's protocol
COLORS = {"vehicle": "#1f77b4", "ronidazole": "#d62728", "danieau": "#7f7f7f"}  # in their order within a line
RERUN_STIMULI = {int(Stim.PHOTOTAXIS), int(Stim.DARK), int(Stim.BRIGHT)}  # all a phototaxis re-run shows
COMMON_FILTERS = load_yaml_config(Path(__file__).parents[1] / "screen.yaml")["common_filters"]


def fish_per_arm(root: Path) -> pd.DataFrame:
    """Per (line, condition), its numbers of fish (full protocol) and of phototaxis re-runs that pass quality control."""
    trials = exclude_qc_fish(pd.read_csv(root / "valid_trials_all.csv", dtype={"line": str, "condition": str}), root / "qc_all.csv")
    trials = apply_table_filters(trials, {"recording": COMMON_FILTERS}, ("recording",))
    presented = trials[trials["presented"].astype(bool)]  # valid_trials lists the whole protocol: what each recording showed
    recordings = presented.groupby(["line", "condition", "file"])["stim"].agg(lambda stim: set(stim) <= RERUN_STIMULI).rename("rerun").reset_index()
    counts = recordings.groupby(["line", "condition"])["rerun"].agg(fish=lambda rerun: int((~rerun).sum()), reruns="sum")
    return counts.reset_index()


def plot_sample_sizes(counts: pd.DataFrame, width: float = 0.38) -> plt.Figure:
    """Each line's conditions side by side under its name."""
    order = {condition: i for i, condition in enumerate(COLORS)}
    counts = counts.sort_values(["line", "condition"], key=lambda c: c.map(order) if c.name == "condition" else c)
    fig, ax = plt.subplots(figsize=(14, 6))
    lines = counts["line"].unique()
    for x, line in enumerate(lines):
        arm = counts[counts["line"] == line]
        ax.bar(x + (np.arange(len(arm)) - (len(arm) - 1) / 2) * width, arm["fish"], width, color=[COLORS.get(c, "black") for c in arm["condition"]])
    ax.set_xticks(range(len(lines)), lines, rotation=45, ha="right")
    ax.set_ylabel("N fish (after QC)")
    present = [c for c in COLORS if c in set(counts["condition"])]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in present], present, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    days, rest = divmod(int(counts["fish"].sum()) * RECORDING_S // 60, 24 * 60)
    totals = ", ".join(f"{c} {int(counts.loc[counts['condition'] == c, 'fish'].sum())}" for c in present)
    reruns = int(counts["reruns"].sum())
    ax.set_title(f"N = {totals}; {days} days, {rest // 60} hours and {rest % 60} minutes of recording"
                 + (f" (and {reruns} phototaxis re-run{'s' if reruns > 1 else ''}, not counted)" if reruns else ""), fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("/media/martin/DATA_18TB/Screen"), help="Folder of the merged *_all.csv tables.")
    parser.add_argument("-o", "--output", type=Path, default=Path("line_sample_sizes.png"))
    args = parser.parse_args()
    counts = fish_per_arm(args.data_root)
    print(counts.set_index(["line", "condition"]).to_string())
    plot_sample_sizes(counts).savefig(args.output, dpi=300)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
