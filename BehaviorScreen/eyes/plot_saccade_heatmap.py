#!/usr/bin/env python3
"""Calculate and plot stimulus-aligned saccade-class frequencies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.stim_specs import (
    StimSpec,
    apply_event_filters,
    exclude_qc_fish,
    exclude_unusable_trials,
    get_matching_epoch_names,
    get_single_value,
    load_valid_trials,
    load_yaml_config,
    read_stim_specs,
    stimulus_name_order,
)

SACCADE_CLASS_NAMES = {
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

DEFAULT_CLASS_ORDER = list(range(9))

HEATMAP_VARIANTS = [
    (False, False, "", "trial × time bin"),
    (True, False, "_trial_avg", "averaged over trials"),
    (False, True, "_timebin_avg", "averaged over time bins"),
    (
        True,
        True,
        "_full_avg",
        "averaged over trials and time bins",
    ),
]


def get_exposure_by_trial(
    valid_trials: pd.DataFrame,
    fish: str,
    epoch_names: list[str],
    duration: float,
    exclude_unusable: bool,
) -> tuple[pd.Series, int]:
    """
    Calculate observed seconds for each pooled trial index.

    When a stimulus specification pools multiple raw epoch names, each
    presented epoch contributes its own duration.
    """
    trials = valid_trials.loc[
        (valid_trials["file"].astype(str) == str(fish))
        & valid_trials["epoch_name"].isin(epoch_names)
        & valid_trials["presented"]
    ].copy()

    if trials.empty:
        return pd.Series(dtype=float), 0

    number_of_trials = int(trials["trial_num"].max()) + 1

    if exclude_unusable:
        trials = trials.loc[trials["usable"]]

    exposure = trials.groupby("trial_num").size().astype(float) * duration

    return exposure, number_of_trials


def compute_spec_counts(
    fish_events: pd.DataFrame,
    specification: StimSpec,
    number_of_trials: int,
    exposure_by_trial: pd.Series,
    class_order: list[int],
) -> pd.DataFrame:
    """Count saccades on a complete trial × class grid."""
    if specification.time_range is None:
        raise ValueError("Saccade heatmaps require time bins.")

    start, stop = specification.time_range
    duration = stop - start

    if duration <= 0:
        raise ValueError(
            f"Invalid time interval for {specification}: " f"{specification.time_range}"
        )

    mask = (
        specification.get_mask(fish_events)
        & (fish_events["stim"] == specification.stim)
        & (fish_events["trial_time"] >= start)
        & (fish_events["trial_time"] < stop)
    )

    selected = fish_events.loc[mask].copy()
    selected = selected.dropna(subset=["trial_num", "cluster"])

    selected["trial_idx"] = pd.to_numeric(
        selected["trial_num"],
        errors="coerce",
    )
    selected["cluster"] = pd.to_numeric(
        selected["cluster"],
        errors="coerce",
    )
    selected = selected.dropna(subset=["trial_idx", "cluster"])

    selected["trial_idx"] = selected["trial_idx"].astype(int)
    selected["cluster"] = selected["cluster"].astype(int)

    selected = selected.loc[
        (selected["trial_idx"] >= 0)
        & (selected["trial_idx"] < number_of_trials)
        & selected["cluster"].isin(class_order)
    ]

    counts = selected.groupby(["trial_idx", "cluster"]).size().rename("saccade_counts")

    full_index = pd.MultiIndex.from_product(
        [
            range(number_of_trials),
            class_order,
        ],
        names=["trial_idx", "cluster"],
    )

    result = counts.reindex(full_index, fill_value=0).reset_index()

    result["exposure_s"] = result["trial_idx"].map(exposure_by_trial).fillna(0.0)

    result["saccade_frequency"] = np.where(
        result["exposure_s"] > 0,
        result["saccade_counts"] / result["exposure_s"],
        np.nan,
    )

    result["time_bin_duration"] = duration

    return result


def compute_saccade_frequency_table(
    input_csv: Path,
    valid_trials_csv: Path,
    quality_control: Path,
    config_yaml: Path,
    exclude_unusable: bool = True,
    include_unassigned: bool = False,
) -> pd.DataFrame:
    """Construct the complete per-fish saccade-frequency table."""
    config = load_yaml_config(config_yaml)
    specifications = list(read_stim_specs(config))

    events = pd.read_csv(input_csv)
    valid_trials = load_valid_trials(valid_trials_csv)

    required_columns = {
        "file",
        "stim",
        "epoch_name",
        "trial_num",
        "trial_time",
        "cluster",
    }
    missing_columns = required_columns.difference(events.columns)

    if missing_columns:
        raise ValueError(
            f"{input_csv} is missing columns: " f"{sorted(missing_columns)}"
        )

    print(f"Total number of saccades: {len(events):,}")

    events = exclude_qc_fish(
        dataframe=events,
        quality_control_path=quality_control,
        file_column="file",
    )
    valid_trials = exclude_qc_fish(
        dataframe=valid_trials,
        quality_control_path=quality_control,
        file_column="file",
    )

    # Resolve stimulus/epoch membership before event-level filtering.
    epoch_names = {
        id(specification): get_matching_epoch_names(
            events,
            specification,
        )
        for specification in specifications
    }

    events = apply_event_filters(
        dataframe=events,
        config=config,
        event_type="saccade",
    )

    if exclude_unusable:
        events = exclude_unusable_trials(
            events=events,
            valid_trials=valid_trials,
        )

    class_order = DEFAULT_CLASS_ORDER.copy()

    if include_unassigned:
        class_order.insert(0, -1)

    tables: list[pd.DataFrame] = []

    # valid_trials defines the denominator and fish universe. Fish with no
    # surviving saccades therefore contribute zero rates where they have
    # valid exposure.
    fish_names = valid_trials["file"].astype(str).unique()

    for fish in tqdm(
        fish_names,
        desc="Saccade frequencies",
    ):
        fish_events = events.loc[events["file"].astype(str) == fish]

        for specification in specifications:
            if specification.time_range is None:
                raise ValueError(f"No time range is defined for {specification}.")

            start, stop = specification.time_range
            duration = stop - start

            exposure, number_of_trials = get_exposure_by_trial(
                valid_trials=valid_trials,
                fish=fish,
                epoch_names=epoch_names[id(specification)],
                duration=duration,
                exclude_unusable=exclude_unusable,
            )

            if number_of_trials == 0:
                continue

            counts = compute_spec_counts(
                fish_events=fish_events,
                specification=specification,
                number_of_trials=number_of_trials,
                exposure_by_trial=exposure,
                class_order=class_order,
            )

            counts["file"] = fish
            counts["stim_name"] = specification.name
            counts["time_bin_start"] = start
            counts["time_bin_stop"] = stop
            counts["cluster_name"] = counts["cluster"].map(SACCADE_CLASS_NAMES)

            for column in (
                "dpf",
                "day",
                "cos_daytime",
                "sin_daytime",
            ):
                counts[column] = get_single_value(
                    fish_events,
                    column,
                    fish,
                )

            tables.append(counts)

    if not tables:
        return pd.DataFrame()

    return pd.concat(
        tables,
        ignore_index=True,
    )


def aggregate_saccade_frequency(
    per_fish: pd.DataFrame,
    average_trial: bool,
    average_time_bin: bool,
) -> pd.DataFrame:
    """Collapse dimensions within fish and then average across fish."""
    working = per_fish.copy()

    if average_trial or average_time_bin:
        within_fish_columns = [
            "file",
            "stim_name",
            "cluster",
            "cluster_name",
        ]

        if not average_trial:
            within_fish_columns.append("trial_idx")

        if not average_time_bin:
            within_fish_columns.extend(
                [
                    "time_bin_start",
                    "time_bin_stop",
                ]
            )

        working = working.groupby(
            within_fish_columns,
            as_index=False,
            dropna=False,
        )[["saccade_counts", "exposure_s"]].sum()

        working["saccade_frequency"] = np.where(
            working["exposure_s"] > 0,
            working["saccade_counts"] / working["exposure_s"],
            np.nan,
        )

    across_fish_columns = [
        "stim_name",
        "cluster",
        "cluster_name",
    ]

    if not average_trial:
        across_fish_columns.append("trial_idx")

    if not average_time_bin:
        across_fish_columns.extend(
            [
                "time_bin_start",
                "time_bin_stop",
            ]
        )

    return working.groupby(
        across_fish_columns,
        as_index=False,
        dropna=False,
    )["saccade_frequency"].mean()


def build_heatmap_matrix(
    averaged: pd.DataFrame,
    class_order: list[int],
    stimulus_order: list[str],
) -> tuple[
    pd.DataFrame,
    list[tuple[int, int, str]],
    list[str],
    int,
]:
    """Build the class × stimulus/time-bin matrix."""
    if averaged.empty:
        raise ValueError(
            "Cannot build a heatmap matrix from an empty table."
        )

    has_trial = "trial_idx" in averaged.columns
    has_time_bin = "time_bin_start" in averaged.columns

    number_of_trials = (
        int(averaged["trial_idx"].max()) + 1
        if has_trial
        else 1
    )

    column_groups: list[tuple[int, int, str]] = []
    time_labels: list[str] = []

    if has_time_bin:
        column_values: list[tuple[str, float]] = []

        for stimulus_name in stimulus_order:
            stimulus_rows = averaged.loc[
                averaged["stim_name"] == stimulus_name
            ]

            if stimulus_rows.empty:
                continue

            group_start = len(column_values)

            bins = (
                stimulus_rows[
                    ["time_bin_start", "time_bin_stop"]
                ]
                .drop_duplicates()
                .sort_values("time_bin_start")
            )

            for start, stop in bins.itertuples(index=False):
                column_values.append(
                    (stimulus_name, float(start))
                )
                time_labels.append(
                    f"{start:g}-{stop:g}s"
                )

            column_groups.append(
                (
                    group_start,
                    len(column_values),
                    stimulus_name,
                )
            )

        column_index: pd.Index = pd.MultiIndex.from_tuples(
            column_values,
            names=["stim_name", "time_bin_start"],
        )
        pivot_columns: str | list[str] = [
            "stim_name",
            "time_bin_start",
        ]

    else:
        # With time bins averaged out, pivot_table creates an ordinary
        # Index of stimulus-name strings, not a one-level MultiIndex.
        column_values_single: list[str] = []

        for stimulus_name in stimulus_order:
            stimulus_rows = averaged.loc[
                averaged["stim_name"] == stimulus_name
            ]

            if stimulus_rows.empty:
                continue

            group_start = len(column_values_single)
            column_values_single.append(stimulus_name)
            time_labels.append("avg")

            column_groups.append(
                (
                    group_start,
                    len(column_values_single),
                    stimulus_name,
                )
            )

        column_index = pd.Index(
            column_values_single,
            name="stim_name",
        )
        pivot_columns = "stim_name"

    if has_trial:
        row_index: pd.Index = pd.MultiIndex.from_product(
            [
                class_order,
                range(number_of_trials),
            ],
            names=["cluster", "trial_idx"],
        )
        pivot_index: str | list[str] = [
            "cluster",
            "trial_idx",
        ]
    else:
        row_index = pd.Index(
            class_order,
            name="cluster",
        )
        pivot_index = "cluster"

    pivot = averaged.pivot_table(
        index=pivot_index,
        columns=pivot_columns,
        values="saccade_frequency",
        aggfunc="mean",
        observed=False,
    )

    pivot = pivot.reindex(
        index=row_index,
        columns=column_index,
    )

    if pivot.shape[1] == 0:
        raise ValueError(
            "The heatmap matrix has no columns. Check the configured "
            "stimulus names and the filtered frequency table."
        )

    if not np.isfinite(
        pivot.to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "The heatmap matrix contains only NaN values. This usually "
            "indicates an index mismatch or zero valid exposure."
        )

    return (
        pivot,
        column_groups,
        time_labels,
        number_of_trials,
    )


def plot_heatmap_matrix(
    figure: plt.Figure,
    axis: plt.Axes,
    pivot: pd.DataFrame,
    class_order: list[int],
    column_groups: list[tuple[int, int, str]],
    time_labels: list[str],
    number_of_trials: int,
    title: str,
    color_limit: tuple[float, float],
) -> None:
    """Draw one saccade-frequency heatmap."""
    data = pivot.to_numpy(dtype=float)
    number_of_rows, number_of_columns = data.shape

    image = axis.imshow(
        data,
        aspect="auto",
        cmap="inferno",
        vmin=color_limit[0],
        vmax=color_limit[1],
    )

    figure.colorbar(
        image,
        ax=axis,
        label="saccade frequency (s⁻¹)",
        fraction=0.015,
        pad=0.01,
    )

    if any(label != "avg" for label in time_labels):
        axis.set_xticks(range(number_of_columns))
        axis.set_xticklabels(
            time_labels,
            rotation=90,
            fontsize=7,
        )
    else:
        axis.set_xticks([])

    if isinstance(pivot.index, pd.MultiIndex):
        axis.set_yticks(range(number_of_rows))
        axis.set_yticklabels(
            [trial_index for _, trial_index in pivot.index],
            fontsize=7,
        )
    else:
        axis.set_yticks([])

    for start, stop, label in column_groups:
        if start > 0:
            axis.axvline(
                start - 0.5,
                color="white",
                linewidth=1.5,
            )

        axis.annotate(
            label,
            xy=((start + stop - 1) / 2, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            annotation_clip=False,
        )

    for class_index, cluster in enumerate(class_order):
        row_start = class_index * number_of_trials
        row_stop = row_start + number_of_trials

        if row_start > 0:
            axis.axhline(
                row_start - 0.5,
                color="white",
                linewidth=1.5,
            )

        axis.annotate(
            SACCADE_CLASS_NAMES.get(
                cluster,
                str(cluster),
            ),
            xy=(
                0,
                (row_start + row_stop - 1) / 2,
            ),
            xycoords=("axes fraction", "data"),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            annotation_clip=False,
        )

    axis.set_title(title, pad=42)
    axis.set_xlabel("")
    axis.set_ylabel("")


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
    """Calculate saccade frequencies and create heatmap variants."""
    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_yaml_config(config_yaml)

    per_fish = compute_saccade_frequency_table(
        input_csv=input_csv,
        valid_trials_csv=valid_trials_csv,
        quality_control=quality_control,
        config_yaml=config_yaml,
        exclude_unusable=exclude_unusable,
        include_unassigned=include_unassigned,
    )

    if per_fish.empty:
        print("No saccade-frequency data were generated.")
        return

    per_fish.to_csv(
        output_png.parent / "saccade_frequency.csv",
        index=False,
    )

    class_order = DEFAULT_CLASS_ORDER.copy()

    if include_unassigned:
        class_order.insert(0, -1)

    stimulus_order = stimulus_name_order(config)

    for (
        average_trial,
        average_time_bin,
        suffix,
        title,
    ) in HEATMAP_VARIANTS:
        averaged = aggregate_saccade_frequency(
            per_fish=per_fish,
            average_trial=average_trial,
            average_time_bin=average_time_bin,
        )

        averaged.to_csv(
            output_png.parent / f"saccade_frequency_avg{suffix}.csv",
            index=False,
        )

        (
            pivot,
            column_groups,
            time_labels,
            number_of_trials,
        ) = build_heatmap_matrix(
            averaged=averaged,
            class_order=class_order,
            stimulus_order=stimulus_order,
        )

        figure_width = max(
            14,
            0.25 * pivot.shape[1],
        )
        figure_height = max(
            6,
            0.25 * pivot.shape[0],
        )

        figure, axis = plt.subplots(
            figsize=(figure_width, figure_height),
            layout="constrained",
        )

        plot_heatmap_matrix(
            figure=figure,
            axis=axis,
            pivot=pivot,
            class_order=class_order,
            column_groups=column_groups,
            time_labels=time_labels,
            number_of_trials=number_of_trials,
            title=title,
            color_limit=(0.0, maximum_frequency),
        )

        variant_path = (
            output_png.parent / f"{output_png.stem}{suffix}" f"{output_png.suffix}"
        )

        figure.savefig(
            variant_path,
            dpi=180,
            bbox_inches="tight",
        )

        if not interactive:
            plt.close(figure)

        print(f"Saved {variant_path}")

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
        default="augmented_saccades.csv",
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
        default=0.4,
        help="Maximum heatmap frequency in saccades per second.",
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
