#!/usr/bin/env python3
"""Calculate and plot stimulus-aligned bout-frequency heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from megabouts.utils import bouts_category_name_short
from tqdm import tqdm

from BehaviorScreen.core import EventDirection
from BehaviorScreen.stim_specs import (
    StimSpec,
    apply_event_filters,
    exclude_qc_fish,
    exclude_unusable_trials,
    get_epoch_trial_count,
    get_matching_epoch_names,
    get_single_value,
    load_valid_trials,
    load_yaml_config,
    read_stim_specs,
    stimulus_name_order,
)
from BehaviorScreen.plot_utils import (
    order_lateralities,
    order_signs,
    map_laterality,
    SIGN_LABELS,
    HEATMAP_VARIANTS
)

MAX_COLORBAR = 0.6

ALL_BOUT_CATEGORIES = list(bouts_category_name_short)
EXCLUDED_BOUT_CATEGORIES = {"LCS", "SCS"}
BOUT_CATEGORIES = [
    category
    for category in ALL_BOUT_CATEGORIES
    if category not in EXCLUDED_BOUT_CATEGORIES
]
BOUT_SIGN_LABELS = {
    int(EventDirection.LEFT): "LEFT",
    int(EventDirection.RIGHT): "RIGHT",
}


def load_bouts(path: Path) -> pd.DataFrame:
    """Load the augmented bout table."""
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def map_sign(series: pd.Series) -> pd.Series:
    """Map raw bout-sign codes to LEFT and RIGHT."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(BOUT_SIGN_LABELS)


def get_laterality_labels(
    bouts: pd.DataFrame,
    specification: StimSpec,
) -> list[str]:
    """Find laterality groups represented by a stimulus specification."""
    if "laterality" not in bouts.columns:
        return ["none"]

    mask = specification.get_mask(bouts)

    if "stim" in bouts.columns:
        mask &= bouts["stim"] == specification.stim

    values = map_laterality(bouts.loc[mask, "laterality"]).dropna().unique().tolist()

    return order_lateralities(values) if values else ["none"]


def category_code_to_name(code: Any) -> str | None:
    """Convert a Megabouts category index to its short name."""
    try:
        index = int(code)
    except (TypeError, ValueError):
        return None

    if index < 0 or index >= len(ALL_BOUT_CATEGORIES):
        return None

    return ALL_BOUT_CATEGORIES[index]


def compute_epoch_bout_counts(
    fish_bouts: pd.DataFrame,
    specification: StimSpec,
    number_of_trials: int,
    laterality_labels: list[str],
) -> pd.DataFrame:
    """
    Count bouts on a complete trial × category × laterality grid.

    Missing combinations are represented by zero.
    """
    if specification.time_range is None:
        raise ValueError("Bout heatmaps require time bins.")

    start, stop = specification.time_range
    duration = stop - start

    if duration <= 0:
        raise ValueError(
            f"Invalid time interval for {specification}: " f"{specification.time_range}"
        )

    mask = (
        specification.get_mask(fish_bouts)
        & (fish_bouts["stim"] == specification.stim)
        & (fish_bouts["trial_time"] >= start)
        & (fish_bouts["trial_time"] < stop)
    )

    selected = fish_bouts.loc[mask].copy()
    selected = selected.dropna(subset=["category", "trial_num"])

    selected["trial_idx"] = pd.to_numeric(
        selected["trial_num"],
        errors="coerce",
    )
    selected["category"] = pd.to_numeric(
        selected["category"],
        errors="coerce",
    )
    selected = selected.dropna(subset=["trial_idx", "category"])

    selected["trial_idx"] = selected["trial_idx"].astype(int)
    selected["category"] = selected["category"].astype(int)

    selected = selected.loc[
        (selected["trial_idx"] >= 0) & (selected["trial_idx"] < number_of_trials)
    ]

    selected["bout_category"] = selected["category"].map(category_code_to_name)

    selected = selected.loc[
        selected["bout_category"].notna()
        & ~selected["bout_category"].isin(EXCLUDED_BOUT_CATEGORIES)
    ]

    if "laterality" in selected.columns:
        selected["laterality_group"] = map_laterality(selected["laterality"])
    else:
        selected["laterality_group"] = "none"

    counts = (
        selected.groupby(
            [
                "trial_idx",
                "bout_category",
                "laterality_group",
            ]
        )
        .size()
        .rename("bout_counts")
    )

    full_index = pd.MultiIndex.from_product(
        [
            range(number_of_trials),
            BOUT_CATEGORIES,
            laterality_labels,
        ],
        names=[
            "trial_idx",
            "bout_category",
            "laterality_group",
        ],
    )

    result = counts.reindex(full_index, fill_value=0).reset_index()

    result["time_bin_duration"] = duration
    result["bout_frequency"] = result["bout_counts"] / duration

    return result


def compute_epoch_bout_counts_classic(
    fish_bouts: pd.DataFrame,
    specification: StimSpec,
    number_of_trials: int,
    epoch_names: list[str],
) -> pd.DataFrame:
    """
    Count bouts by raw epoch name and LEFT/RIGHT bout sign.

    This retains the layout of the original "classic" heatmap.
    """
    if specification.time_range is None:
        raise ValueError("Bout heatmaps require time bins.")

    start, stop = specification.time_range
    duration = stop - start

    if duration <= 0:
        raise ValueError(
            f"Invalid time interval for {specification}: " f"{specification.time_range}"
        )

    mask = (
        specification.get_mask(fish_bouts)
        & (fish_bouts["stim"] == specification.stim)
        & (fish_bouts["trial_time"] >= start)
        & (fish_bouts["trial_time"] < stop)
    )

    selected = fish_bouts.loc[mask].copy()
    selected = selected.dropna(
        subset=[
            "category",
            "trial_num",
            "sign",
            "epoch_name",
        ]
    )

    selected["trial_idx"] = pd.to_numeric(
        selected["trial_num"],
        errors="coerce",
    )
    selected["category"] = pd.to_numeric(
        selected["category"],
        errors="coerce",
    )
    selected = selected.dropna(subset=["trial_idx", "category"])

    selected["trial_idx"] = selected["trial_idx"].astype(int)
    selected["category"] = selected["category"].astype(int)

    selected = selected.loc[
        (selected["trial_idx"] >= 0) & (selected["trial_idx"] < number_of_trials)
    ]

    selected["bout_category"] = selected["category"].map(category_code_to_name)

    selected = selected.loc[
        selected["bout_category"].notna()
        & ~selected["bout_category"].isin(EXCLUDED_BOUT_CATEGORIES)
    ]

    selected["sign_group"] = map_sign(selected["sign"])
    selected = selected.loc[selected["sign_group"].notna()]

    counts = (
        selected.groupby(
            [
                "trial_idx",
                "bout_category",
                "epoch_name",
                "sign_group",
            ]
        )
        .size()
        .rename("bout_counts")
    )

    full_index = pd.MultiIndex.from_product(
        [
            range(number_of_trials),
            BOUT_CATEGORIES,
            epoch_names,
            SIGN_LABELS,
        ],
        names=[
            "trial_idx",
            "bout_category",
            "epoch_name",
            "sign_group",
        ],
    )

    result = counts.reindex(full_index, fill_value=0).reset_index()

    result["time_bin_duration"] = duration
    result["bout_frequency"] = result["bout_counts"] / duration

    return result


def compute_bout_frequency_table(
    quality_control: Path,
    input_csv: Path,
    valid_trials: pd.DataFrame,
    config_yaml: Path,
    exclude_unusable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct direction-pooled and classic bout-frequency tables."""
    config = load_yaml_config(config_yaml)
    specifications = list(read_stim_specs(config))

    bouts = load_bouts(input_csv)

    required_columns = {
        "file",
        "stim",
        "epoch_name",
        "trial_num",
        "trial_time",
        "category",
        "sign",
    }
    missing_columns = required_columns.difference(bouts.columns)

    if missing_columns:
        raise ValueError(
            f"{input_csv} is missing columns: " f"{sorted(missing_columns)}"
        )

    print(f"Total number of bouts: {len(bouts):,}")

    bouts = exclude_qc_fish(
        dataframe=bouts,
        quality_control_path=quality_control,
        file_column="file",
    )
    valid_trials = exclude_qc_fish(
        dataframe=valid_trials,
        quality_control_path=quality_control,
        file_column="file",
    )

    # Resolve stimulus membership before event-level filtering. This avoids
    # losing an epoch name merely because all its events fail a QC filter.
    epoch_names = {
        id(specification): get_matching_epoch_names(
            bouts,
            specification,
        )
        for specification in specifications
    }

    bouts = apply_event_filters(
        dataframe=bouts,
        config=config,
        event_type="bout",
    )

    laterality_labels = {
        id(specification): get_laterality_labels(
            bouts,
            specification,
        )
        for specification in specifications
    }

    if exclude_unusable:
        bouts = exclude_unusable_trials(
            events=bouts,
            valid_trials=valid_trials,
        )

    tables: list[pd.DataFrame] = []
    classic_tables: list[pd.DataFrame] = []

    # Preserve the existing bout behavior: only fish with surviving bouts
    # contribute to the bout-frequency average.
    for fish, fish_bouts in tqdm(
        bouts.groupby("file", sort=False),
        desc="Bout frequencies",
    ):
        fish = str(fish)

        for specification in specifications:
            if specification.time_range is None:
                raise RuntimeError("Bout heatmaps require time ranges.")

            matched_epoch_names = epoch_names[id(specification)]

            number_of_trials = get_epoch_trial_count(
                valid_trials=valid_trials,
                fish=fish,
                epoch_names=matched_epoch_names,
            )

            if number_of_trials == 0:
                print(f"{fish} - {specification} was not presented; " "skipping.")
                continue

            start, stop = specification.time_range

            common_fields = {
                "file": fish,
                "dpf": get_single_value(
                    fish_bouts,
                    "dpf",
                    fish,
                ),
                "day": get_single_value(
                    fish_bouts,
                    "day",
                    fish,
                ),
                "cos_daytime": get_single_value(
                    fish_bouts,
                    "cos_daytime",
                    fish,
                ),
                "sin_daytime": get_single_value(
                    fish_bouts,
                    "sin_daytime",
                    fish,
                ),
                "stim_name": specification.name,
                "time_bin_start": start,
                "time_bin_stop": stop,
                "time_bin_duration": stop - start,
            }

            counts = compute_epoch_bout_counts(
                fish_bouts=fish_bouts,
                specification=specification,
                number_of_trials=number_of_trials,
                laterality_labels=laterality_labels[id(specification)],
            )
            counts = counts.assign(**common_fields)
            tables.append(counts)

            classic_counts = compute_epoch_bout_counts_classic(
                fish_bouts=fish_bouts,
                specification=specification,
                number_of_trials=number_of_trials,
                epoch_names=matched_epoch_names,
            )
            classic_counts = classic_counts.assign(**common_fields)
            classic_tables.append(classic_counts)

    base_columns = [
        "trial_idx",
        "bout_category",
        "bout_counts",
        "bout_frequency",
        "file",
        "dpf",
        "day",
        "cos_daytime",
        "sin_daytime",
        "stim_name",
        "time_bin_start",
        "time_bin_stop",
        "time_bin_duration",
    ]

    per_fish = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=base_columns + ["laterality_group"])
    )

    per_fish_classic = (
        pd.concat(classic_tables, ignore_index=True)
        if classic_tables
        else pd.DataFrame(columns=base_columns + ["epoch_name", "sign_group"])
    )

    return per_fish, per_fish_classic


def aggregate_bout_frequency(
    per_fish: pd.DataFrame,
    average_trial: bool,
    average_time_bin: bool,
    split_columns: tuple[str, ...] = ("laterality_group",),
) -> pd.DataFrame:
    """
    Collapse selected dimensions within fish and then average across fish.
    """
    split_columns_list = list(split_columns)
    working = per_fish.copy()

    if average_trial or average_time_bin:
        within_fish_columns = [
            "file",
            "stim_name",
            "bout_category",
            *split_columns_list,
        ]

        if not average_time_bin:
            within_fish_columns.extend(
                [
                    "time_bin_start",
                    "time_bin_stop",
                ]
            )

        if not average_trial:
            within_fish_columns.append("trial_idx")

        working = working.groupby(
            within_fish_columns,
            as_index=False,
            dropna=False,
        )[["bout_counts", "time_bin_duration"]].sum()

        working["bout_frequency"] = np.where(
            working["time_bin_duration"] > 0,
            working["bout_counts"] / working["time_bin_duration"],
            np.nan,
        )

    across_fish_columns = [
        "stim_name",
        "bout_category",
        *split_columns_list,
    ]

    if not average_time_bin:
        across_fish_columns.extend(
            [
                "time_bin_start",
                "time_bin_stop",
            ]
        )

    if not average_trial:
        across_fish_columns.append("trial_idx")

    return working.groupby(
        across_fish_columns,
        as_index=False,
        dropna=False,
    )["bout_frequency"].mean()


def build_bout_heatmap_matrix(
    averaged: pd.DataFrame,
    category_order: list[str],
    stimulus_order: list[str],
) -> tuple[
    pd.DataFrame,
    list[tuple[int, int, str]],
    list[tuple[int, int, str]],
    list[str],
    int,
]:
    """Build the matrix used by the grouped bout heatmap."""
    has_trial = "trial_idx" in averaged.columns
    has_time_bin = "time_bin_start" in averaged.columns

    number_of_trials = (
        int(averaged["trial_idx"].max()) + 1 if has_trial and not averaged.empty else 1
    )

    columns = []
    column_groups = []
    column_subgroups = []
    time_bin_labels = []

    for stimulus in stimulus_order:
        stimulus_rows = averaged.loc[averaged["stim_name"] == stimulus]

        if stimulus_rows.empty:
            continue

        stimulus_start = len(columns)

        lateralities = order_lateralities(
            stimulus_rows["laterality_group"].dropna().unique()
        )

        for laterality in lateralities:
            subgroup_rows = stimulus_rows.loc[
                stimulus_rows["laterality_group"] == laterality
            ]
            subgroup_start = len(columns)

            if has_time_bin:
                bins = (
                    subgroup_rows[["time_bin_start", "time_bin_stop"]]
                    .drop_duplicates()
                    .sort_values("time_bin_start")
                )

                for start, stop in bins.itertuples(index=False):
                    columns.append((stimulus, laterality, start))
                    time_bin_labels.append(f"{start:g}-{stop:g}s")
            else:
                columns.append((stimulus, laterality))
                time_bin_labels.append("avg")

            column_subgroups.append(
                (
                    subgroup_start,
                    len(columns),
                    laterality,
                )
            )

        column_groups.append(
            (
                stimulus_start,
                len(columns),
                stimulus,
            )
        )

    if has_trial:
        row_index = pd.MultiIndex.from_product(
            [
                category_order,
                range(number_of_trials),
            ],
            names=["bout_category", "trial_idx"],
        )
        index_columns = ["bout_category", "trial_idx"]
    else:
        row_index = pd.Index(
            category_order,
            name="bout_category",
        )
        index_columns = ["bout_category"]

    if has_time_bin:
        column_names = [
            "stim_name",
            "laterality_group",
            "time_bin_start",
        ]
    else:
        column_names = [
            "stim_name",
            "laterality_group",
        ]

    column_index = pd.MultiIndex.from_tuples(
        columns,
        names=column_names,
    )

    pivot = averaged.pivot_table(
        index=index_columns,
        columns=column_names,
        values="bout_frequency",
    )

    pivot = pivot.reindex(
        index=row_index,
        columns=column_index,
    )

    return (
        pivot,
        column_groups,
        column_subgroups,
        time_bin_labels,
        number_of_trials,
    )


def plot_bout_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    pivot: pd.DataFrame,
    category_order: list[str],
    column_groups: list[tuple[int, int, str]],
    column_subgroups: list[tuple[int, int, str]],
    time_bin_labels: list[str],
    number_of_trials: int,
    title: str,
    color_limit: tuple[float, float],
) -> None:
    """Draw a grouped bout-frequency heatmap."""
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
        label="bout frequency (s⁻¹)",
        fraction=0.015,
        pad=0.01,
    )

    if any(label != "avg" for label in time_bin_labels):
        axis.set_xticks(range(number_of_columns))
        axis.set_xticklabels(
            time_bin_labels,
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
                linewidth=1.6,
            )

        axis.annotate(
            label,
            xy=((start + stop - 1) / 2, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 38),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            annotation_clip=False,
        )

    for start, stop, label in column_subgroups:
        if start > 0:
            axis.axvline(
                start - 0.5,
                color="white",
                linewidth=0.6,
                alpha=0.7,
            )

        axis.annotate(
            label,
            xy=((start + stop - 1) / 2, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            annotation_clip=False,
        )

    for category_index, category in enumerate(category_order):
        row_start = category_index * number_of_trials
        row_stop = row_start + number_of_trials

        if row_start > 0:
            axis.axhline(
                row_start - 0.5,
                color="white",
                linewidth=1.6,
            )

        axis.annotate(
            category,
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

    axis.set_title(title, fontsize=12, pad=62)
    axis.set_xlabel("")
    axis.set_ylabel("")


def build_classic_bout_heatmap_matrix(
    averaged: pd.DataFrame,
    category_order: list[str],
    stimulus_order: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the original direction-separated bout heatmap matrix."""
    if "trial_idx" in averaged.columns:
        raise ValueError("The classic heatmap requires trial-averaged data.")

    if "time_bin_start" not in averaged.columns:
        raise ValueError("The classic heatmap requires retained time bins.")

    required_columns = {
        "epoch_name",
        "sign_group",
    }
    missing_columns = required_columns.difference(averaged.columns)

    if missing_columns:
        raise ValueError(
            "Classic heatmap data are missing columns: " f"{sorted(missing_columns)}"
        )

    signs = order_signs(averaged["sign_group"].dropna().unique())

    row_index = pd.MultiIndex.from_product(
        [category_order, signs],
        names=["bout_category", "sign_group"],
    )

    column_tuples = []
    column_labels = []

    for stimulus in stimulus_order:
        stimulus_rows = averaged.loc[averaged["stim_name"] == stimulus]

        if stimulus_rows.empty:
            continue

        bins = (
            stimulus_rows[["time_bin_start", "time_bin_stop"]]
            .drop_duplicates()
            .sort_values("time_bin_start")
        )
        epoch_names = sorted(stimulus_rows["epoch_name"].dropna().unique())

        for start, stop in bins.itertuples(index=False):
            for epoch_name in epoch_names:
                has_data = (
                    (stimulus_rows["epoch_name"] == epoch_name)
                    & (stimulus_rows["time_bin_start"] == start)
                ).any()

                if not has_data:
                    continue

                column_tuples.append((epoch_name, start))
                column_labels.append(f"{epoch_name} | {start:g}-{stop:g}s")

    column_index = pd.MultiIndex.from_tuples(
        column_tuples,
        names=["epoch_name", "time_bin_start"],
    )

    pivot = averaged.pivot_table(
        index=["bout_category", "sign_group"],
        columns=["epoch_name", "time_bin_start"],
        values="bout_frequency",
    )

    pivot = pivot.reindex(
        index=row_index,
        columns=column_index,
    )

    row_labels = [f"{category}_{sign}" for category, sign in pivot.index]

    return pivot, column_labels, row_labels


def plot_bout_heatmap_classic(
    figure: plt.Figure,
    axis: plt.Axes,
    pivot: pd.DataFrame,
    column_labels: list[str],
    row_labels: list[str],
    color_limit: tuple[float, float],
) -> None:
    """Draw the original flat bout-frequency heatmap."""
    data = pivot.to_numpy(dtype=float)

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
        label="bout frequency (s⁻¹)",
    )

    axis.set_xticks(range(data.shape[1]))
    axis.set_xticklabels(
        column_labels,
        rotation=90,
        ha="center",
        fontsize=8,
    )
    axis.set_yticks(range(data.shape[0]))
    axis.set_yticklabels(
        row_labels,
        fontsize=8,
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("bout category")


def compute_fish_count_table(
    quality_control: Path,
    valid_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Count fish with a usable trial for each epoch and trial number."""
    trials = exclude_qc_fish(
        dataframe=valid_trials,
        quality_control_path=quality_control,
        file_column="file",
    )

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

    all_pairs = trials[["epoch_name", "trial_num"]].drop_duplicates()

    counts = (
        trials.loc[trials["usable"]]
        .groupby(["epoch_name", "trial_num"])
        .size()
        .rename("n_fish")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_frame(all_pairs)

    counts = (
        counts.set_index(["epoch_name", "trial_num"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return counts, number_of_fish


def build_fish_count_matrix(
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build trial × epoch usable-fish matrix."""
    epoch_order = list(dict.fromkeys(counts["epoch_name"]))

    number_of_trials = int(counts["trial_num"].max()) + 1 if not counts.empty else 0

    pivot = counts.pivot_table(
        index="trial_num",
        columns="epoch_name",
        values="n_fish",
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
    """Calculate bout frequencies and create all output plots."""
    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_yaml_config(config_yaml)
    valid_trials = load_valid_trials(valid_trials_csv)

    per_fish, per_fish_classic = compute_bout_frequency_table(
        quality_control=quality_control,
        input_csv=input_csv,
        valid_trials=valid_trials,
        config_yaml=config_yaml,
        exclude_unusable=exclude_unusable,
    )

    per_fish.to_csv(
        output_png.parent / "bout_frequency.csv",
        index=False,
    )
    per_fish_classic.to_csv(
        output_png.parent / "bout_frequency_classic.csv",
        index=False,
    )

    fish_counts, number_of_fish = compute_fish_count_table(
        quality_control=quality_control,
        valid_trials=valid_trials,
    )
    fish_counts.to_csv(
        output_png.parent / "fish_count.csv",
        index=False,
    )

    if not fish_counts.empty:
        fish_count_matrix, epoch_order = build_fish_count_matrix(fish_counts)

        figure_width = max(
            16,
            0.25 * len(epoch_order),
        )
        figure_height = max(
            6,
            0.28 * fish_count_matrix.shape[0],
        )

        figure, axis = plt.subplots(
            figsize=(figure_width, figure_height),
            layout="constrained",
        )

        plot_fish_count_heatmap(
            figure=figure,
            axis=axis,
            pivot=fish_count_matrix,
            epoch_order=epoch_order,
            number_of_fish=number_of_fish,
        )

        fish_count_path = (
            output_png.parent / f"{output_png.stem}_fish_count" f"{output_png.suffix}"
        )
        figure.savefig(
            fish_count_path,
            dpi=180,
            bbox_inches="tight",
        )

        if not interactive:
            plt.close(figure)

    if per_fish.empty:
        print("No bouts remain after filtering.")

        if interactive:
            plt.show()

        return

    category_order = BOUT_CATEGORIES
    stim_order = stimulus_name_order(config)

    for (
        average_trial,
        average_time_bin,
        suffix,
        title,
    ) in HEATMAP_VARIANTS:
        averaged = aggregate_bout_frequency(
            per_fish=per_fish,
            average_trial=average_trial,
            average_time_bin=average_time_bin,
            split_columns=("laterality_group",),
        )

        averaged.to_csv(
            output_png.parent / f"bout_frequency_avg{suffix}.csv",
            index=False,
        )

        (
            pivot,
            column_groups,
            column_subgroups,
            time_labels,
            number_of_trials,
        ) = build_bout_heatmap_matrix(
            averaged=averaged,
            category_order=category_order,
            stimulus_order=stim_order,
        )

        figure_width = max(
            16,
            0.22 * pivot.shape[1],
        )
        figure_height = max(
            6,
            0.28 * pivot.shape[0],
        )

        figure, axis = plt.subplots(
            figsize=(figure_width, figure_height),
            layout="constrained",
        )

        plot_bout_heatmap(
            figure=figure,
            axis=axis,
            pivot=pivot,
            category_order=category_order,
            column_groups=column_groups,
            column_subgroups=column_subgroups,
            time_bin_labels=time_labels,
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

    if not per_fish_classic.empty:
        classic_average = aggregate_bout_frequency(
            per_fish=per_fish_classic,
            average_trial=True,
            average_time_bin=False,
            split_columns=("epoch_name", "sign_group"),
        )

        classic_average.to_csv(
            output_png.parent / "bout_frequency_avg_classic.csv",
            index=False,
        )

        (
            classic_pivot,
            classic_column_labels,
            classic_row_labels,
        ) = build_classic_bout_heatmap_matrix(
            averaged=classic_average,
            category_order=category_order,
            stimulus_order=stim_order,
        )

        figure_width = max(
            20,
            0.35 * len(classic_column_labels),
        )
        figure_height = max(
            10,
            0.32 * len(classic_row_labels),
        )

        figure, axis = plt.subplots(
            figsize=(figure_width, figure_height),
            layout="constrained",
        )

        plot_bout_heatmap_classic(
            figure=figure,
            axis=axis,
            pivot=classic_pivot,
            column_labels=classic_column_labels,
            row_labels=classic_row_labels,
            color_limit=(0.0, maximum_frequency),
        )

        classic_path = (
            output_png.parent / f"{output_png.stem}_classic" f"{output_png.suffix}"
        )

        figure.savefig(
            classic_path,
            dpi=180,
            bbox_inches="tight",
        )

        if not interactive:
            plt.close(figure)

        print(f"Saved {classic_path}")

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
