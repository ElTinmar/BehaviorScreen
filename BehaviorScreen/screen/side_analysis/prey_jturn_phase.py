"""
Ipsilateral J-turns over one cycle of the prey arc: the dot's absolute angle from the fish's heading (top) and the
J-turn rate at each phase (bottom), over every complete cycle and over the cycles after the first SKIP (the transient
at the onset of the prey left out), both prey capture epochs pooled. The bouts are those the heatmaps and models count:
quality control, usable trials and the bout filters of filters.yaml. The rate is per fish: J-turns in a phase bin over
the fish x cycles observed and the bin's duration.
    python -m BehaviorScreen.screen.side_analysis.prey_jturn_phase [--data-root DIR] [--condition vehicle] [--line ...]
        [--skip 2] [--bins 36] [-o prey_jturn_phase.png]
Default: the vehicle arms of all lines pooled, from the merged tables of this machine's data_root. The phase is
stim_phase: 0 at the start of the arc, pi at its end (the arc's phase offset, prey_arc_phase_deg, is not recorded and
taken as 0); cycle k of a trial spans [k, k + 1) / f of its time, f = speed / (2 x arc range).
"""
import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from megabouts.utils import bouts_category_name_short

from BehaviorScreen.core import Laterality
from BehaviorScreen.filters import FILTERS, apply_event_filters, exclude_qc_fish
from BehaviorScreen.preprocessing.stim_specs import exclude_unusable_trials, load_valid_trials
from BehaviorScreen.screen.paths import PREY_JTURN_PHASE, add_data_root, data_root

EPOCHS = ["prey capture left", "prey capture right"]
ARC = ["prey_arc_start_deg", "prey_arc_stop_deg", "prey_speed_deg_s"]


def select(table: pd.DataFrame, lines: Optional[List[str]], conditions: Optional[List[str]]) -> pd.DataFrame:
    for column, values in (("line", lines), ("condition", conditions)):
        if values:
            table = table[table[column].astype(str).isin(values)]
    return table


def phase_rate(jturns: pd.DataFrame, trials: pd.DataFrame, frequency: float, skip: int, edges: np.ndarray) -> np.ndarray:
    """J-turns per fish and second in each phase bin, over the complete cycles from cycle `skip` on."""
    cycles = np.floor(trials["trial_duration_s"].to_numpy() * frequency)
    exposure = np.clip(cycles - skip, 0, None).sum()
    kept = jturns[(jturns["cycle"] >= skip) & (jturns["cycle"] < jturns["complete_cycles"])]
    counts, _ = np.histogram(np.mod(kept["stim_phase"], 2 * np.pi), edges)
    return counts / (exposure * np.diff(edges) / (2 * np.pi * frequency))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_root(parser)
    parser.add_argument("--condition", nargs="*", default=["vehicle"], help="Conditions pooled (default: vehicle).")
    parser.add_argument("--line", nargs="*", help="Lines pooled (default: all).")
    parser.add_argument("--skip", type=int, default=2, help="Cycles left out at the start of each trial, for the second curve.")
    parser.add_argument("--bins", type=int, default=36, help="Phase bins over the cycle.")
    parser.add_argument("-o", "--output", type=Path, default=PREY_JTURN_PHASE)
    args = parser.parse_args()
    root = data_root(args.data_root)

    columns = ["file", "line", "condition", "epoch_name", "trial_num", "trial_time", "category", "laterality", "stim_phase", *ARC, *FILTERS["bout"]]
    analysed, prey = set(), []
    for chunk in pd.read_csv(root / "bouts_all.csv", usecols=columns, chunksize=1_000_000, low_memory=False):
        analysed.update(chunk["file"].astype(str))
        prey.append(chunk[chunk["epoch_name"].isin(EPOCHS)])
    bouts = select(pd.concat(prey), args.line, args.condition).astype({"file": str, "trial_num": int})
    if bouts.empty:
        raise SystemExit(f"no prey capture bouts in {root / 'bouts_all.csv'} for this selection")
    arcs = bouts[ARC].abs().drop_duplicates()
    if len(arcs) != 1:
        raise SystemExit(f"several prey arcs or speeds (absolute values), plot them apart:\n{arcs.to_string(index=False)}")
    start, stop, speed = arcs.iloc[0]
    frequency = speed / (2 * abs(stop - start))

    qc = root / "qc_all.csv"
    trials = exclude_qc_fish(select(load_valid_trials(root / "valid_trials_all.csv"), args.line, args.condition), qc)
    trials = trials[trials["epoch_name"].isin(EPOCHS) & trials["usable"] & trials["file"].isin(analysed)]
    bouts = exclude_unusable_trials(apply_event_filters(exclude_qc_fish(bouts, qc), "bout"), trials)
    jturns = bouts[(bouts["category"] == bouts_category_name_short.index("JT")) & (bouts["laterality"] == int(Laterality.IPSILATERAL))]
    durations = trials.set_index(["file", "epoch_name", "trial_num"])["trial_duration_s"]
    jturns = jturns.assign(
        cycle=np.floor(jturns["trial_time"] * frequency),
        complete_cycles=np.floor(durations.reindex(pd.MultiIndex.from_frame(jturns[["file", "epoch_name", "trial_num"]])).to_numpy() * frequency),
    ).dropna(subset=["stim_phase"])

    edges = np.linspace(0, 2 * np.pi, args.bins + 1)
    rates = {"all complete cycles": phase_rate(jturns, trials, frequency, 0, edges),
             f"from cycle {args.skip + 1} on": phase_rate(jturns, trials, frequency, args.skip, edges)}

    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 5), gridspec_kw={"height_ratios": [1, 1.4]})
    grid = np.linspace(0, 2 * np.pi, 400)
    top.plot(grid, start + (stop - start) * (1 - np.cos(grid)) / 2, color="black")
    top.set_ylabel("|prey angle| (deg)\nfrom heading")
    outward, inward = ("naso-temporal", "temporo-nasal") if stop > start else ("temporo-nasal", "naso-temporal")
    top.text(np.pi / 2, max(start, stop), outward, ha="center", va="top", fontsize=8)
    top.text(3 * np.pi / 2, max(start, stop), inward, ha="center", va="top", fontsize=8)
    (label_all, all_cycles), (label_late, late) = rates.items()
    bottom.bar(edges[:-1], all_cycles, width=np.diff(edges), align="edge", color="tab:blue", alpha=0.45, edgecolor="white", linewidth=0.5, label=label_all)
    bottom.stairs(late, edges, color="black", linewidth=1.2, label=label_late)
    bottom.set_ylabel("ipsilateral J-turns\nper fish (s⁻¹)")
    bottom.legend(frameon=False, fontsize=8)
    for axis in (top, bottom):
        axis.axvspan(0, np.pi, color="0.93", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    bottom.set_xlim(0, 2 * np.pi)
    bottom.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    bottom.set_xlabel(f"phase of the prey cycle ({1 / frequency:.2f} s)")
    selection = ", ".join(f"{c}: {' '.join(v)}" for c, v in (("line", args.line), ("condition", args.condition)) if v)
    figure.suptitle(f"{len(jturns)} ipsilateral J-turns, {trials['file'].nunique()} fish ({selection or 'all arms'})", fontsize=9)
    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    print(f"{len(jturns)} ipsilateral J-turns, {trials['file'].nunique()} fish, {len(trials)} trials -> {args.output}")


if __name__ == "__main__":
    main()
