#!/usr/bin/env python3
"""Stimulus-aligned bout-frequency heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from megabouts.utils import bouts_category_name_short

from BehaviorScreen.core import EventDirection
from BehaviorScreen.preprocessing.plots.classic_heatmap import plot_classic_heatmap
from BehaviorScreen.preprocessing.plots.event_rates import EventKind, load_events, prepare_events
from BehaviorScreen.preprocessing.plots.plot_utils import plot_rate_heatmaps
from BehaviorScreen.filters import load_yaml_config
from BehaviorScreen.preprocessing.stim_specs import load_valid_trials

MAX_COLORBAR = 0.6

ALL_BOUT_CATEGORIES = list(bouts_category_name_short)
EXCLUDED_BOUT_CATEGORIES = {"LCS", "SCS"}
BOUT_CATEGORIES = [c for c in ALL_BOUT_CATEGORIES if c not in EXCLUDED_BOUT_CATEGORIES]
BOUT_CATEGORY_MAP = dict(enumerate(ALL_BOUT_CATEGORIES))
SIGN_LABELS = ["LEFT", "RIGHT"]
BOUT_SIGN_LABELS = {int(EventDirection.LEFT): "LEFT", int(EventDirection.RIGHT): "RIGHT"}


def assign_bout_category(bouts: pd.DataFrame) -> pd.DataFrame:
    category = pd.to_numeric(bouts["category"], errors="coerce")
    return bouts.assign(
        bout_category=category.map(BOUT_CATEGORY_MAP),
        sign_group=pd.to_numeric(bouts["sign"], errors="coerce").map(BOUT_SIGN_LABELS),
    )


BOUTS = EventKind(
    name="bout",
    category="bout_category",
    categories=BOUT_CATEGORIES,
    labels=BOUT_CATEGORIES,
    assign_category=assign_bout_category,
    required_columns=frozenset({"category", "sign"}),
    levels={"sign_group": SIGN_LABELS},
)
BOUT_CLASSIC_ROWS = [(category, sign) for category in BOUT_CATEGORIES for sign in SIGN_LABELS]


def compute_fish_count_table(trials: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Count fish with a usable trial for each epoch and trial."""
    number_of_fish = trials["file"].nunique()

    if trials.empty:
        return (
            pd.DataFrame(
                columns=[
                    "epoch_name",
                    "trial_num",
                    "n_fish",
                ]
            ),
            number_of_fish,
        )

    all_pairs = trials[
        [
            "epoch_name",
            "trial_num",
        ]
    ].drop_duplicates()

    counts = (
        trials.loc[trials["usable"]]
        .groupby(
            [
                "epoch_name",
                "trial_num",
            ],
            sort=False,
            observed=True,
        )
        .size()
        .rename("n_fish")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_frame(all_pairs)

    counts = (
        counts.set_index(
            [
                "epoch_name",
                "trial_num",
            ]
        )
        .reindex(
            full_index,
            fill_value=0,
        )
        .reset_index()
    )

    return counts, number_of_fish


def build_fish_count_matrix(
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a trial × epoch usable-fish matrix."""
    epoch_order = list(dict.fromkeys(counts["epoch_name"]))

    number_of_trials = int(counts["trial_num"].max()) + 1 if not counts.empty else 0

    pivot = counts.pivot_table(
        index="trial_num",
        columns="epoch_name",
        values="n_fish",
        aggfunc="mean",
        observed=True,
    )

    pivot = pivot.reindex(
        index=range(number_of_trials),
        columns=epoch_order,
    )

    return pivot, epoch_order


def plot_fish_count_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    pivot: pd.DataFrame,
    epoch_order: list[str],
    number_of_fish: int,
) -> None:
    """Plot usable fish count by epoch and trial."""
    data = pivot.to_numpy(dtype=float)

    image = axis.imshow(
        data,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=max(number_of_fish, 1),
    )

    figure.colorbar(
        image,
        ax=axis,
        label="number of usable fish",
        fraction=0.02,
        pad=0.01,
    )

    axis.set_xticks(range(data.shape[1]))
    axis.set_xticklabels(
        epoch_order,
        rotation=90,
        fontsize=7,
    )
    axis.set_yticks(range(data.shape[0]))
    axis.set_yticklabels(
        range(data.shape[0]),
        fontsize=7,
    )
    axis.set_xlabel("epoch name")
    axis.set_ylabel("trial number")
    axis.set_title(
        "Fish remaining per epoch × trial " f"({number_of_fish} fish total)",
        fontsize=11,
        fontweight="bold",
    )


def plot_fish_count(trials: pd.DataFrame, output_png: Path, interactive: bool) -> None:
    counts, number_of_fish = compute_fish_count_table(trials)
    counts.to_csv(output_png.parent / "fish_count.csv", index=False)
    if counts.empty:
        return

    matrix, epoch_order = build_fish_count_matrix(counts)
    figure, axis = plt.subplots(
        figsize=(max(16, 0.25 * len(epoch_order)), max(6, 0.28 * matrix.shape[0])),
        layout="constrained",
    )
    plot_fish_count_heatmap(figure, axis, matrix, epoch_order, number_of_fish)
    figure.savefig(
        output_png.parent / f"{output_png.stem}_fish_count{output_png.suffix}",
        dpi=180,
        bbox_inches="tight",
    )
    if not interactive:
        plt.close(figure)


def plot_heatmaps(
    quality_control: Path,
    input_csv: Path,
    valid_trials_csv: Path,
    config_yaml: Path,
    output_png: Path,
    exclude_unusable: bool = True,
    interactive: bool = False,
    maximum_frequency: float = MAX_COLORBAR,
) -> None:
    """Bout frequencies, their heatmaps, the classic heatmap and usable-fish counts."""
    config = load_yaml_config(config_yaml)
    prepared = prepare_events(
        BOUTS,
        load_events(input_csv, BOUTS),
        load_valid_trials(valid_trials_csv),
        quality_control,
        config,
        exclude_unusable,
    )
    per_fish = plot_rate_heatmaps(BOUTS, prepared, config, output_png, maximum_frequency, interactive)
    if not per_fish.empty:
        plot_fish_count(prepared.trials, output_png, interactive)
        plot_classic_heatmap(BOUTS, BOUT_CLASSIC_ROWS, prepared, config, output_png, maximum_frequency, interactive)
    if interactive:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description="Calculate and plot bout frequencies.")

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment directory.",
    )
    parser.add_argument(
        "yaml",
        type=Path,
        help="YAML analysis configuration.",
    )
    parser.add_argument(
        "--qc-csv",
        default="qc.csv",
    )
    parser.add_argument(
        "--bouts-csv",
        default="bouts.csv",
    )
    parser.add_argument(
        "--valid-trials-csv",
        default="valid_trials.csv",
    )
    parser.add_argument(
        "--bouts-png",
        default="bouts.png",
    )
    parser.add_argument(
        "--include-unusable-trials",
        action="store_true",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=MAX_COLORBAR,
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
    )

    return parser


def main() -> None:
    """Run bout heatmap generation."""
    args = build_parser().parse_args()

    plot_heatmaps(
        quality_control=args.root / args.qc_csv,
        input_csv=args.root / args.bouts_csv,
        valid_trials_csv=(args.root / args.valid_trials_csv),
        config_yaml=args.yaml,
        output_png=args.root / args.bouts_png,
        exclude_unusable=(not args.include_unusable_trials),
        interactive=args.interactive,
        maximum_frequency=args.vmax,
    )


if __name__ == "__main__":
    main()
