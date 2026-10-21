"""
Stimulus-aligned PSTHs of every bout and saccade category: one figure per
event kind, one column per YAML stimulus (widths proportional to its
fine_time_bins window, so all columns share one time scale), one row per
category. Each panel has one rate estimate per trial, coloured by trial number, and the
average over trials in black; or, with --stack, each trial on its own baseline,
first trial on top. Directional
stimuli: ipsilateral events above zero, contralateral mirrored below.

Rate of a trial, pooled over fish (kernel estimate, edge corrected):

    rate(t) = (G * events)(t) / (G * exposure)(t)

events are the category's onsets on a DT grid, exposure the fish observed at t
(usable trials lasting past t) x DT, G a Gaussian of SD --sigma. Fish without
events count through their exposure.

The top row shows the stimulus where the event table records it, at each
event and coloured by trial: the prey position along its arc (0 at the start
of the arc, 1 at its end, from stim_phase) and the looming radius. Stimuli with
a phase get an extra column with the rate against phase, all trials together
(the phase advances at a constant speed, so exposure is uniform over phase).

    python -m BehaviorScreen.psth ROOT YAML [--sigma 0.1] [--stack 0]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from BehaviorScreen.event_rates import EventKind, PreparedEvents, load_events, prepare_events
from BehaviorScreen.eyes.plot_saccade_heatmap import saccade_kind
from BehaviorScreen.plot import BOUTS
from BehaviorScreen.stim_specs import StimEntry, load_valid_trials, load_yaml_config, read_stim_entries, select_events

DT, PHASE_BINS, PHASE_SIGMA = 0.01, 72, 2 * np.pi / 36
SIDE_SIGN = {"ipsi": 1.0, "contra": -1.0, "none": 1.0}
STIMULUS_VALUES = {  # event column: label, transform
    "stim_phase": ("prey position", lambda phase: (1 - np.cos(phase)) / 2),
    "looming_radius": ("radius (mm)", lambda radius: radius),
}
CMAP = plt.get_cmap("viridis")


@dataclass
class Panel:
    name: str
    time: np.ndarray
    per_trial: dict[tuple, np.ndarray]  # (category, side) -> trials x time
    average: dict[tuple, np.ndarray]  # (category, side) -> time
    events: pd.DataFrame
    exposure_s: float  # fish x seconds observed


def smooth(x: np.ndarray, sigma: float, step: float, mode: str = "constant") -> np.ndarray:
    return gaussian_filter1d(x, sigma / step, axis=-1, mode=mode)


def entry_panel(kind: EventKind, prepared: PreparedEvents, index: int, entry: StimEntry, sigma: float) -> Panel:
    epoch_names, sides = prepared.epoch_names[index], prepared.lateralities[index]
    start, stop = entry.bins[0][0], entry.bins[-1][1]
    time = np.arange(start, stop - DT / 2, DT) + DT / 2
    trials = prepared.trials
    trials = trials[
        trials["epoch_name"].isin(epoch_names)
        & trials["usable" if prepared.usable_only else "presented"]
        & entry.spec.get_mask(trials)
    ]
    exposure = np.zeros((int(trials["trial_num"].max()) + 1, len(time)))
    np.add.at(exposure, trials["trial_num"].to_numpy(), (trials["trial_duration_s"].to_numpy()[:, None] > time) * DT)

    events = select_events(prepared.events, entry.spec)
    events = events[
        events["epoch_name"].isin(epoch_names) & (events["trial_time"] >= start) & (events["trial_time"] < stop)
        & (events["trial_idx"] < len(exposure))
    ]
    groups = {key: group for key, group in events.groupby([kind.category, "laterality_group"], observed=True)}
    trial_exposure, total_exposure = smooth(exposure, sigma, DT), smooth(exposure.sum(axis=0), sigma, DT)
    per_trial, average = {}, {}
    for category in kind.categories:
        for side in sides:
            group = groups.get((category, side), events.iloc[0:0])
            counts = np.zeros_like(exposure)
            bins = np.minimum(((group["trial_time"].to_numpy() - start) / DT).astype(int), len(time) - 1)
            np.add.at(counts, (group["trial_idx"].to_numpy(), bins), 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_trial[category, side] = np.where(exposure > 0, smooth(counts, sigma, DT) / trial_exposure, np.nan)
                average[category, side] = smooth(counts.sum(axis=0), sigma, DT) / total_exposure
    return Panel(entry.name, time, per_trial, average, events, float(exposure.sum()))


def phase_rates(kind: EventKind, panel: Panel) -> tuple[np.ndarray, dict[tuple, np.ndarray]]:
    """Rate against stimulus phase, all trials together."""
    step = 2 * np.pi / PHASE_BINS
    exposure = panel.exposure_s / PHASE_BINS
    rates = {}
    for key in panel.average:
        group = panel.events[(panel.events[kind.category] == key[0]) & (panel.events["laterality_group"] == key[1])]
        counts = np.bincount((np.mod(group["stim_phase"].dropna(), 2 * np.pi) // step).astype(int), minlength=PHASE_BINS)[:PHASE_BINS]
        rates[key] = smooth(counts.astype(float), PHASE_SIGMA, step, mode="wrap") / exposure
    return (np.arange(PHASE_BINS) + 0.5) * step, rates


def draw_stimulus(axis: plt.Axes, panel: Panel, n_trials: int) -> None:
    for column, (label, transform) in STIMULUS_VALUES.items():
        if column in panel.events and panel.events[column].notna().any():
            axis.scatter(panel.events["trial_time"], transform(panel.events[column]), s=1, linewidths=0,
                         c=panel.events["trial_idx"], cmap=CMAP, vmin=0, vmax=max(n_trials - 1, 1))
            axis.text(0.01, 0.95, label, transform=axis.transAxes, fontsize=7, va="top")
            return


def draw_rates(axis: plt.Axes, x: np.ndarray, per_trial: dict, average: dict, category, offset: float, n_trials: int) -> None:
    for (cat, side), rates in per_trial.items():
        if cat != category:
            continue
        for k, rate in enumerate(rates):  # stacked: trial k on its own baseline, -k x offset
            axis.plot(x, SIDE_SIGN[side] * rate - offset * k, color=CMAP(k / max(n_trials - 1, 1)), lw=0.7, alpha=0.8)
    for (cat, side), rate in average.items():
        if cat == category and offset == 0:
            axis.plot(x, SIDE_SIGN[side] * rate, color="black", lw=1.4)
    trials = max([len(rates) for (cat, _), rates in per_trial.items() if cat == category] + [1])
    for k in range(trials if offset else 1):
        axis.axhline(-offset * k, color="0.6", lw=0.4)


def plot_psth(kind: EventKind, prepared: PreparedEvents, config: dict, path: Path, sigma: float, stack: float) -> None:
    panels = [entry_panel(kind, prepared, i, e, sigma) for i, e in enumerate(read_stim_entries(config, "fine_time_bins"))]
    phased = [p for p in panels if "stim_phase" in p.events and p.events["stim_phase"].notna().any()]
    widths = [p.time[-1] - p.time[0] + DT for p in panels] + [10.0] * len(phased)
    n_trials = max(len(rates) for p in panels for rates in p.per_trial.values())
    peak = {
        category: max([np.nanpercentile(v, 99) for p in panels for (c, _), v in p.per_trial.items() if c == category and np.isfinite(v).any()] + [1e-3])
        for category in kind.categories
    }
    rows, row_height = len(kind.categories), max(1.0, 0.2 * n_trials) if stack else 1.0
    figure, axes = plt.subplots(
        rows + 1, len(widths), squeeze=False, figsize=(0.12 * sum(widths) + 3, row_height * rows + 2),
        gridspec_kw={"width_ratios": widths, "height_ratios": [0.8] + [row_height] * rows, "hspace": 0.15, "wspace": 0.05},
    )
    for j, panel in enumerate(panels):
        axes[0, j].set_title(panel.name, fontsize=9)
        draw_stimulus(axes[0, j], panel, n_trials)
        for i, category in enumerate(kind.categories, start=1):
            draw_rates(axes[i, j], panel.time, panel.per_trial, panel.average, category, stack * peak[category], n_trials)
        for axis in axes[:, j]:
            axis.set_xlim(panel.time[0] - DT / 2, panel.time[-1] + DT / 2)
    for j, panel in enumerate(phased, start=len(panels)):
        axes[0, j].set_title(f"{panel.name} · by phase", fontsize=9)
        phases, rates = phase_rates(kind, panel)
        axes[0, j].plot(phases, STIMULUS_VALUES["stim_phase"][1](phases), color="black", lw=1)
        for i, category in enumerate(kind.categories, start=1):
            draw_rates(axes[i, j], phases, {}, rates, category, 0.0, n_trials)
        for axis in axes[:, j]:
            axis.set_xlim(0, 2 * np.pi)
        axes[-1, j].set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        axes[-1, j].set_xlabel("phase")

    for i, (category, label) in enumerate(zip(kind.categories, kind.labels), start=1):
        mirrored = any(side == "contra" for p in panels for (c, side) in p.per_trial if c == category)
        for axis in axes[i]:
            axis.set_ylim(-1.05 * peak[category] * (mirrored + stack * (n_trials - 1)), 1.05 * peak[category])
        axes[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        if stack:
            ticks = np.arange(0, n_trials, 5)
            axes[i, 0].set_yticks(-stack * peak[category] * ticks, [f"trial {k}" for k in ticks])
    axes[0, 0].set_ylabel("stimulus", rotation=0, ha="right", va="center", fontsize=9)
    for axis in axes[:, 1:].flat:
        axis.tick_params(labelleft=False)
    for axis in axes[:-1].flat:
        axis.tick_params(labelbottom=False)
    for axis in axes.flat:
        axis.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("time in trial (s)")
    figure.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, n_trials - 1), CMAP), ax=axes, fraction=0.01, pad=0.01, label="trial")
    figure.suptitle(
        f"{kind.name} rate (s⁻¹, fish pooled, Gaussian kernel SD {sigma:g} s), ipsi above 0, contra below; "
        + (f"trials stacked by {stack:g}x the row's 99th percentile" if stack else "black: all trials"),
        fontsize=11,
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def main(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.yaml)
    trials = load_valid_trials(args.root / args.valid_trials_csv)
    output = args.root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for kind, csv in ((BOUTS, args.bouts_csv), (saccade_kind(), args.saccades_csv)):
        if not (args.root / csv).exists():
            print(f"No {args.root / csv}, skipping {kind.name}s")
            continue
        prepared = prepare_events(
            kind, load_events(args.root / csv, kind), trials, args.root / args.qc_csv, config, not args.include_unusable_trials
        )
        suffix = f"_stacked{args.stack:g}" if args.stack else ""
        plot_psth(kind, prepared, config, output / f"{kind.name}_psth{suffix}.png", args.sigma, args.stack)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stimulus-aligned PSTHs per bout and saccade category, trial by trial.")
    parser.add_argument("root", type=Path, help="Root experiment directory")
    parser.add_argument("yaml", type=Path, help="YAML analysis configuration")
    parser.add_argument("--sigma", type=float, default=0.1, help="SD of the Gaussian kernel (s)")
    parser.add_argument("--stack", type=float, default=0.0, help="Offset between trials, in units of the row's 99th percentile (0: overlaid)")
    parser.add_argument("--bouts-csv", default="bouts.csv")
    parser.add_argument("--saccades-csv", default="saccades_augmented.csv")
    parser.add_argument("--valid-trials-csv", default="valid_trials.csv")
    parser.add_argument("--qc-csv", default="qc.csv")
    parser.add_argument("--output-dir", default="psth")
    parser.add_argument("--include-unusable-trials", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
