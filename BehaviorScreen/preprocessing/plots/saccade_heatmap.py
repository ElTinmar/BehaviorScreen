#!/usr/bin/env python3
"""Stimulus-aligned saccade-class frequency heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from BehaviorScreen.core import SACCADE_CATEGORY_NAMES, SACCADE_NONDIRECTIONAL_CLASSES, EventDirection
from BehaviorScreen.preprocessing.plots.classic_heatmap import plot_classic_heatmap
from BehaviorScreen.preprocessing.plots.event_rates import EventKind, load_events, prepare_events
from BehaviorScreen.preprocessing.plots.plot_utils import plot_rate_heatmaps
from BehaviorScreen.filters import load_yaml_config
from BehaviorScreen.preprocessing.stim_specs import load_valid_trials

DEFAULT_CLASS_ORDER = [
    1,  # Conjugate
    3,  # Miniature convergent
    4,  # Convergent
    6,  # Divergent
    7,  # Biphasic convergent
]


SIDES = {int(EventDirection.LEFT): "LEFT", int(EventDirection.RIGHT): "RIGHT"}


def assign_saccade_category(saccades: pd.DataFrame) -> pd.DataFrame:
    """The category, and the side of directional saccades (sign_group LEFT or RIGHT, NONE otherwise)."""
    direction = pd.to_numeric(saccades.get("event_direction", pd.Series(index=saccades.index, dtype=float)), errors="coerce")
    return saccades.assign(
        saccade_category=pd.to_numeric(saccades["saccade_category"], errors="coerce"),
        sign_group=direction.map(SIDES).fillna("NONE"),
    )


def saccade_kind(include_unassigned: bool = False) -> EventKind:
    categories = ([-1] if include_unassigned else []) + DEFAULT_CLASS_ORDER
    return EventKind(
        name="saccade",
        category="saccade_category",
        categories=categories,
        labels=[SACCADE_CATEGORY_NAMES.get(c, str(c)) for c in categories],
        assign_category=assign_saccade_category,
        required_columns=frozenset({"saccade_category", "laterality"}),
        category_names=SACCADE_CATEGORY_NAMES,
        levels={"sign_group": ["LEFT", "RIGHT", "NONE"]},
    )


def classic_rows(kind: EventKind) -> list:
    """Directional categories (conjugate, biphasic convergent) split by side, the others in one row."""
    nondirectional = SACCADE_NONDIRECTIONAL_CLASSES | {-1}
    return [(c, s) for c in kind.categories for s in (["NONE"] if c in nondirectional else ["LEFT", "RIGHT"])]


def make_saccade_heatmaps(
    input_csv: Path,
    valid_trials_csv: Path,
    quality_control: Path,
    config_yaml: Path,
    output_png: Path,
    exclude_unusable: bool,
    include_unassigned: bool,
    maximum_frequency: float,
    interactive: bool,
) -> None:
    """Calculate saccade frequencies and create heatmaps."""
    kind = saccade_kind(include_unassigned)
    config = load_yaml_config(config_yaml)
    prepared = prepare_events(
        kind,
        load_events(input_csv, kind),
        load_valid_trials(valid_trials_csv),
        quality_control,
        config,
        exclude_unusable,
    )
    plot_rate_heatmaps(kind, prepared, config, output_png, maximum_frequency, interactive)
    if "event_direction" in prepared.events.columns:
        plot_classic_heatmap(kind, classic_rows(kind), prepared, config, output_png, maximum_frequency, interactive)
    else:
        print(f"no event_direction in {input_csv}: no classic heatmap (rerun the saccades step)")
    if interactive:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate and plot stimulus-aligned " "saccade-class frequencies."
        )
    )

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
        "--saccades-csv",
        default="saccades_augmented.csv",
    )
    parser.add_argument(
        "--valid-trials-csv",
        default="valid_trials.csv",
    )
    parser.add_argument(
        "--qc-csv",
        default="qc.csv",
    )
    parser.add_argument(
        "--output",
        default="saccades.png",
    )
    parser.add_argument(
        "--include-unusable-trials",
        action="store_true",
    )
    parser.add_argument(
        "--include-unassigned",
        action="store_true",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.3,
        help=("Maximum heatmap frequency " "in saccades per second."),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
    )

    return parser


def main() -> None:
    """Run saccade heatmap generation."""
    args = build_parser().parse_args()

    make_saccade_heatmaps(
        input_csv=args.root / args.saccades_csv,
        valid_trials_csv=(args.root / args.valid_trials_csv),
        quality_control=args.root / args.qc_csv,
        config_yaml=args.yaml,
        output_png=args.root / args.output,
        exclude_unusable=(not args.include_unusable_trials),
        include_unassigned=args.include_unassigned,
        maximum_frequency=args.vmax,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()