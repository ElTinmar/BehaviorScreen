"""
The classic heatmap of an event kind (bouts, saccades): one row per category and side (sign_group: LEFT, RIGHT, or
NONE for events without one), one column per epoch and time bin, trial-averaged frequencies across fish. The columns
come in one block per stimulus of heatmaps.yaml, named above the plot and parted by white lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pandas as pd

from BehaviorScreen.preprocessing.plots.event_rates import EventKind, PreparedEvents, aggregate, frequency_table
from BehaviorScreen.preprocessing.stim_specs import stimulus_name_order

CLASSIC_SPLIT = ("epoch_name", "sign_group")


def build_classic_matrix(
    kind: EventKind, averaged: pd.DataFrame, rows: list[tuple[Any, str]], stimulus_order: list[str]
) -> tuple[pd.DataFrame, list[str], list[str], list[tuple[str, int, int]]]:
    """
    Trial-averaged frequencies, rows: (category, side) in that order, columns: each stimulus's epochs and time bins;
    with the labels of both and each stimulus's block of columns (name, first, last).
    """
    column_tuples: list[tuple[Any, ...]] = []
    column_labels: list[str] = []
    blocks: list[tuple[str, int, int]] = []
    for stimulus in stimulus_order:
        stimulus_rows = averaged.loc[averaged["stim_name"] == stimulus]
        if stimulus_rows.empty:
            continue
        bins = stimulus_rows[["time_bin_start", "time_bin_stop"]].drop_duplicates().sort_values("time_bin_start")
        for start, stop in bins.itertuples(index=False):
            for epoch_name in sorted(stimulus_rows["epoch_name"].dropna().unique()):
                if ((stimulus_rows["epoch_name"] == epoch_name) & (stimulus_rows["time_bin_start"] == start)).any():
                    column_tuples.append((epoch_name, start))
                    column_labels.append(f"{epoch_name} | {start:g}-{stop:g}s")
        first = blocks[-1][2] + 1 if blocks else 0
        if len(column_labels) > first:
            blocks.append((stimulus, first, len(column_labels) - 1))

    pivot = averaged.pivot_table(
        index=[kind.category, "sign_group"], columns=["epoch_name", "time_bin_start"],
        values=kind.frequency, aggfunc="mean", observed=True,
    ).reindex(
        index=pd.MultiIndex.from_tuples(rows, names=[kind.category, "sign_group"]),
        columns=pd.MultiIndex.from_tuples(column_tuples, names=["epoch_name", "time_bin_start"]),
    )
    names = dict(zip(kind.categories, kind.labels))
    row_labels = [names[category] if sign == "NONE" else f"{names[category]}_{sign}" for category, sign in rows]
    return pivot, column_labels, row_labels, blocks


def plot_classic_heatmap(
    kind: EventKind, rows: list[tuple[Any, str]], prepared: PreparedEvents, config: dict,
    output_png: Path, vmax: float, interactive: bool,
) -> None:
    """<output_png stem>_classic.png, with its per-fish and averaged tables beside it."""
    out = output_png.parent
    per_fish = frequency_table(kind, prepared, config, split=CLASSIC_SPLIT)
    per_fish.to_csv(out / f"{kind.frequency}_classic.csv", index=False)
    if per_fish.empty:
        return
    averaged = aggregate(kind, per_fish, True, False, split=CLASSIC_SPLIT)
    averaged.to_csv(out / f"{kind.name}_frequency_avg_classic.csv", index=False)
    pivot, column_labels, row_labels, blocks = build_classic_matrix(kind, averaged, rows, stimulus_name_order(config))

    data = pivot.to_numpy(dtype=float)
    figure, axis = plt.subplots(
        figsize=(max(20, 0.35 * len(column_labels)), 3.4 + 0.3 * len(row_labels)), layout="constrained"
    )
    image = axis.imshow(data, aspect="auto", cmap="inferno", vmin=0.0, vmax=vmax)
    figure.colorbar(image, ax=axis, label=f"{kind.name} frequency (s⁻¹)")
    axis.set_xticks(range(data.shape[1]), column_labels, rotation=90, ha="center", fontsize=8)
    axis.set_yticks(range(data.shape[0]), row_labels, fontsize=8)
    axis.set_xlabel("epoch")
    axis.set_ylabel(f"{kind.name} category")
    above = blended_transform_factory(axis.transData, axis.transAxes)
    for name, first, last in blocks:
        axis.text((first + last) / 2, 1.01, name, transform=above, ha="center", va="bottom", fontsize=10, fontweight="bold")
        if first:
            axis.axvline(first - 0.5, color="white", linewidth=1.5)

    path = out / f"{output_png.stem}_classic{output_png.suffix}"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    if not interactive:
        plt.close(figure)
    print(f"Saved {path}")
