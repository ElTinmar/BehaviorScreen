#!/usr/bin/env python3
"""Calculate and plot stimulus-aligned bout-frequency heatmaps."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from megabouts.utils import bouts_category_name_short
from tqdm import tqdm

from BehaviorScreen.core import EventDirection
from BehaviorScreen.dataframe_utils import compact_dataframe
from BehaviorScreen.plot_utils import (
    HEATMAP_VARIANTS,
    SIGN_LABELS,
    build_grouped_heatmap_matrix,
    filename_safe,
    grouped_heatmap_figure_size,
    map_laterality,
    order_lateralities,
    order_signs,
    plot_grouped_heatmap,
)
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

MAX_COLORBAR = 0.6

ALL_BOUT_CATEGORIES = list(bouts_category_name_short)
EXCLUDED_BOUT_CATEGORIES = {"LCS", "SCS"}
BOUT_CATEGORIES = [
    category
    for category in ALL_BOUT_CATEGORIES
    if category not in EXCLUDED_BOUT_CATEGORIES
]
BOUT_CATEGORY_MAP = dict(enumerate(ALL_BOUT_CATEGORIES))

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


def compact_bout_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Use compact dtypes for a bout-frequency table."""
    return compact_dataframe(
        dataframe=dataframe,
        categorical_columns=(
            "file",
            "bout_category",
            "laterality_group",
            "epoch_name",
            "sign_group",
            "stim_name",
        ),
        integer_columns=(
            "trial_idx",
            "bout_counts",
            "dpf",
            "day",
        ),
        float_columns=(
            "bout_frequency",
            "time_bin_start",
            "time_bin_stop",
            "time_bin_duration",
            "cos_daytime",
            "sin_daytime",
        ),
    )


def preprocess_bouts(bouts: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare reusable numeric and mapped bout columns once.

    Invalid and excluded bout categories are removed here instead of being
    tested for every fish, specification, and time bin.
    """
    bouts = bouts.copy()
    bouts["file"] = bouts["file"].astype(str)

    bouts["trial_idx"] = pd.to_numeric(
        bouts["trial_num"],
        errors="coerce",
    )
    bouts["category"] = pd.to_numeric(
        bouts["category"],
        errors="coerce",
    )

    bouts = bouts.dropna(
        subset=["trial_idx", "category"],
    ).copy()

    bouts["trial_idx"] = bouts["trial_idx"].astype(np.int32)
    bouts["category"] = bouts["category"].astype(np.int16)
    bouts["bout_category"] = bouts["category"].map(BOUT_CATEGORY_MAP)

    bouts = bouts.loc[
        bouts["bout_category"].notna()
        & ~bouts["bout_category"].isin(EXCLUDED_BOUT_CATEGORIES)
    ].copy()

    if "laterality" in bouts.columns:
        bouts["laterality_group"] = map_laterality(bouts["laterality"])
    else:
        bouts["laterality_group"] = "none"

    bouts["sign_group"] = map_sign(bouts["sign"])

    # Specification regex matching has already been completed by this point.
    # These conversions reduce memory and make repeated grouping cheaper.
    for column in (
        "file",
        "stim",
        "epoch_name",
        "bout_category",
        "laterality_group",
        "sign_group",
    ):
        if column in bouts.columns:
            bouts[column] = bouts[column].astype("category")

    return bouts


def specification_selection_key(
    specification: StimSpec,
    matched_epoch_names: list[str],
) -> tuple[Any, tuple[str, ...]]:
    """Create a reusable cache key for one stimulus/epoch selection."""
    return (
        specification.stim,
        tuple(matched_epoch_names),
    )


def get_matched_laterality_labels(
    events: pd.DataFrame,
    specification: StimSpec,
    matched_epoch_names: list[str],
) -> list[str]:
    """
    Determine laterality labels from previously resolved epoch names.

    This avoids rerunning ``StimSpec.get_mask()`` for every time bin.
    """
    if "laterality" not in events.columns:
        return ["none"]

    mask = (events["stim"] == specification.stim) & events["epoch_name"].isin(
        matched_epoch_names
    )

    labels = map_laterality(events.loc[mask, "laterality"]).dropna().unique().tolist()

    return order_lateralities(labels) if labels else ["none"]


def compute_epoch_bout_counts(
    fish_spec_bouts: pd.DataFrame,
    specification: StimSpec,
    number_of_trials: int,
    laterality_labels: list[str],
    average_trial: bool = False,
) -> pd.DataFrame:
    """
    Count bouts on a complete category/laterality grid.

    ``fish_spec_bouts`` must already be restricted to the relevant fish,
    stimulus and matching epoch names.

    When ``average_trial`` is true, counts and duration are collapsed over
    trials during construction.
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
        (fish_spec_bouts["trial_time"] >= start)
        & (fish_spec_bouts["trial_time"] < stop)
        & (fish_spec_bouts["trial_idx"] >= 0)
        & (fish_spec_bouts["trial_idx"] < number_of_trials)
    )

    selected = fish_spec_bouts.loc[
        mask,
        [
            "trial_idx",
            "bout_category",
            "laterality_group",
        ],
    ]

    if average_trial:
        counts = (
            selected.groupby(
                [
                    "bout_category",
                    "laterality_group",
                ],
                sort=False,
                observed=True,
            )
            .size()
            .rename("bout_counts")
        )

        full_index = pd.MultiIndex.from_product(
            [
                BOUT_CATEGORIES,
                laterality_labels,
            ],
            names=[
                "bout_category",
                "laterality_group",
            ],
        )

        result = counts.reindex(
            full_index,
            fill_value=0,
        ).reset_index()

        result["time_bin_duration"] = duration * number_of_trials
    else:
        counts = (
            selected.groupby(
                [
                    "trial_idx",
                    "bout_category",
                    "laterality_group",
                ],
                sort=False,
                observed=True,
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

        result = counts.reindex(
            full_index,
            fill_value=0,
        ).reset_index()

        result["time_bin_duration"] = duration

    result["bout_frequency"] = np.where(
        result["time_bin_duration"] > 0,
        result["bout_counts"] / result["time_bin_duration"],
        np.nan,
    )

    return result


def compute_epoch_bout_counts_classic(
    fish_spec_bouts: pd.DataFrame,
    specification: StimSpec,
    number_of_trials: int,
    epoch_names: list[str],
) -> pd.DataFrame:
    """
    Count bouts by epoch name and LEFT/RIGHT bout sign.

    ``fish_spec_bouts`` must already be restricted to the relevant fish,
    stimulus and matching epoch names.
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
        (fish_spec_bouts["trial_time"] >= start)
        & (fish_spec_bouts["trial_time"] < stop)
        & (fish_spec_bouts["trial_idx"] >= 0)
        & (fish_spec_bouts["trial_idx"] < number_of_trials)
        & fish_spec_bouts["sign_group"].notna()
        & fish_spec_bouts["epoch_name"].notna()
    )

    selected = fish_spec_bouts.loc[
        mask,
        [
            "trial_idx",
            "bout_category",
            "epoch_name",
            "sign_group",
        ],
    ]

    counts = (
        selected.groupby(
            [
                "trial_idx",
                "bout_category",
                "epoch_name",
                "sign_group",
            ],
            sort=False,
            observed=True,
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

    result = counts.reindex(
        full_index,
        fill_value=0,
    ).reset_index()

    result["time_bin_duration"] = duration
    result["bout_frequency"] = result["bout_counts"] / duration

    return result


def compute_bout_frequency_table(
    quality_control: Path,
    input_csv: Path,
    valid_trials: pd.DataFrame,
    config_yaml: Path,
    exclude_unusable: bool = True,
    time_bins_key: str = "time_bins",
    average_trial_during_compute: bool = False,
    include_classic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct pooled and optional classic bout-frequency tables."""
    config = load_yaml_config(config_yaml)
    specifications = list(
        read_stim_specs(
            config,
            time_bins_key=time_bins_key,
        )
    )

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

    bouts = bouts.copy()
    valid_trials = valid_trials.copy()

    bouts["file"] = bouts["file"].astype(str)
    valid_trials["file"] = valid_trials["file"].astype(str)

    # Resolve regex/specification membership once before event filtering.
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

    # Time bins commonly share the same stimulus/epoch selection.
    laterality_cache: dict[
        tuple[Any, tuple[str, ...]],
        list[str],
    ] = {}
    laterality_labels: dict[int, list[str]] = {}

    for specification in specifications:
        matched_epoch_names = epoch_names[id(specification)]
        selection_key = specification_selection_key(
            specification,
            matched_epoch_names,
        )

        if selection_key not in laterality_cache:
            laterality_cache[selection_key] = get_matched_laterality_labels(
                events=bouts,
                specification=specification,
                matched_epoch_names=matched_epoch_names,
            )

        laterality_labels[id(specification)] = laterality_cache[selection_key]

    if exclude_unusable:
        bouts = exclude_unusable_trials(
            events=bouts,
            valid_trials=valid_trials,
        )

    bouts = preprocess_bouts(bouts)

    valid_trials_by_fish = {
        str(fish): group
        for fish, group in valid_trials.groupby(
            "file",
            sort=False,
            observed=True,
        )
    }

    tables: list[pd.DataFrame] = []
    classic_tables: list[pd.DataFrame] = []

    for fish, fish_bouts in tqdm(
        bouts.groupby(
            "file",
            sort=False,
            observed=True,
        ),
        desc="Bout frequencies",
    ):
        fish = str(fish)
        fish_trials = valid_trials_by_fish.get(fish)

        if fish_trials is None or fish_trials.empty:
            continue

        metadata = {
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
        }

        # These caches are local to one fish and are discarded after the
        # fish has been processed.
        event_selection_cache: dict[
            tuple[Any, tuple[str, ...]],
            pd.DataFrame,
        ] = {}
        trial_count_cache: dict[
            tuple[str, ...],
            int,
        ] = {}

        for specification in specifications:
            if specification.time_range is None:
                raise RuntimeError("Bout heatmaps require time ranges.")

            matched_epoch_names = epoch_names[id(specification)]
            epoch_key = tuple(matched_epoch_names)
            selection_key = specification_selection_key(
                specification,
                matched_epoch_names,
            )

            if selection_key not in event_selection_cache:
                selection_mask = (
                    fish_bouts["stim"] == specification.stim
                ) & fish_bouts["epoch_name"].isin(matched_epoch_names)

                event_selection_cache[selection_key] = fish_bouts.loc[selection_mask]

            fish_spec_bouts = event_selection_cache[selection_key]

            if epoch_key not in trial_count_cache:
                trial_count_cache[epoch_key] = get_epoch_trial_count(
                    valid_trials=fish_trials,
                    fish=fish,
                    epoch_names=matched_epoch_names,
                )

            number_of_trials = trial_count_cache[epoch_key]

            if number_of_trials == 0:
                continue

            start, stop = specification.time_range

            common_fields = {
                "file": fish,
                **metadata,
                "stim_name": specification.name,
                "time_bin_start": start,
                "time_bin_stop": stop,
            }

            counts = compute_epoch_bout_counts(
                fish_spec_bouts=fish_spec_bouts,
                specification=specification,
                number_of_trials=number_of_trials,
                laterality_labels=laterality_labels[id(specification)],
                average_trial=average_trial_during_compute,
            )
            counts = counts.assign(**common_fields)
            tables.append(counts)

            if include_classic:
                classic_counts = compute_epoch_bout_counts_classic(
                    fish_spec_bouts=fish_spec_bouts,
                    specification=specification,
                    number_of_trials=number_of_trials,
                    epoch_names=matched_epoch_names,
                )
                classic_counts = classic_counts.assign(**common_fields)
                classic_tables.append(classic_counts)

    base_columns = [
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
        "laterality_group",
    ]

    if not average_trial_during_compute:
        base_columns.insert(0, "trial_idx")

    if tables:
        per_fish = pd.concat(
            tables,
            ignore_index=True,
            copy=False,
        )
        per_fish = compact_bout_table(per_fish)
    else:
        per_fish = pd.DataFrame(columns=base_columns)

    classic_columns = [
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
        "epoch_name",
        "sign_group",
    ]

    if classic_tables:
        per_fish_classic = pd.concat(
            classic_tables,
            ignore_index=True,
            copy=False,
        )
        per_fish_classic = compact_bout_table(per_fish_classic)
    else:
        per_fish_classic = pd.DataFrame(columns=classic_columns)

    return per_fish, per_fish_classic


def aggregate_bout_frequency(
    per_fish: pd.DataFrame,
    average_trial: bool,
    average_time_bin: bool,
    split_columns: tuple[str, ...] = ("laterality_group",),
) -> pd.DataFrame:
    """Collapse dimensions within fish and average across fish."""
    working = per_fish
    split_columns_list = list(split_columns)
    has_trial = "trial_idx" in working.columns

    if (average_trial and has_trial) or average_time_bin:
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

        if has_trial and not average_trial:
            within_fish_columns.append("trial_idx")

        working = working.groupby(
            within_fish_columns,
            as_index=False,
            dropna=False,
            sort=False,
            observed=True,
        )[
            [
                "bout_counts",
                "time_bin_duration",
            ]
        ].sum()

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

    if has_trial and not average_trial:
        across_fish_columns.append("trial_idx")

    result = working.groupby(
        across_fish_columns,
        as_index=False,
        dropna=False,
        sort=False,
        observed=True,
    )["bout_frequency"].mean()

    return compact_bout_table(result)


def build_classic_bout_heatmap_matrix(
    averaged: pd.DataFrame,
    category_order: list[str],
    stimulus_order: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the original direction-separated bout heatmap."""
    if "trial_idx" in averaged.columns:
        raise ValueError("The classic heatmap requires trial-averaged data.")

    if "time_bin_start" not in averaged.columns:
        raise ValueError("The classic heatmap requires retained time bins.")

    signs = order_signs(averaged["sign_group"].dropna().unique())

    row_index = pd.MultiIndex.from_product(
        [
            category_order,
            signs,
        ],
        names=[
            "bout_category",
            "sign_group",
        ],
    )

    column_tuples: list[tuple[Any, ...]] = []
    column_labels: list[str] = []

    for stimulus in stimulus_order:
        stimulus_rows = averaged.loc[averaged["stim_name"] == stimulus]

        if stimulus_rows.empty:
            continue

        bins = (
            stimulus_rows[
                [
                    "time_bin_start",
                    "time_bin_stop",
                ]
            ]
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

                column_tuples.append(
                    (
                        epoch_name,
                        start,
                    )
                )
                column_labels.append(f"{epoch_name} | {start:g}-{stop:g}s")

    column_index = pd.MultiIndex.from_tuples(
        column_tuples,
        names=[
            "epoch_name",
            "time_bin_start",
        ],
    )

    pivot = averaged.pivot_table(
        index=[
            "bout_category",
            "sign_group",
        ],
        columns=[
            "epoch_name",
            "time_bin_start",
        ],
        values="bout_frequency",
        aggfunc="mean",
        observed=True,
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
    """Count fish with a usable trial for each epoch and trial."""
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
    """Calculate bout frequencies and create all plots."""
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

        figure, axis = plt.subplots(
            figsize=(
                max(16, 0.25 * len(epoch_order)),
                max(6, 0.28 * fish_count_matrix.shape[0]),
            ),
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
            output_png.parent / f"{output_png.stem}_fish_count{output_png.suffix}"
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
    stimulus_order = stimulus_name_order(config)

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
        )

        averaged.to_csv(
            output_png.parent / f"bout_frequency_avg{suffix}.csv",
            index=False,
        )

        (
            pivot,
            groups,
            subgroups,
            labels,
            number_of_trials,
        ) = build_grouped_heatmap_matrix(
            averaged=averaged,
            row_column="bout_category",
            row_order=category_order,
            value_column="bout_frequency",
            stimulus_order=stimulus_order,
        )

        figure, axis = plt.subplots(
            figsize=grouped_heatmap_figure_size(pivot.shape),
            layout="constrained",
        )

        plot_grouped_heatmap(
            figure=figure,
            axis=axis,
            pivot=pivot,
            row_group_labels=category_order,
            column_groups=groups,
            column_subgroups=subgroups,
            time_labels=labels,
            number_of_trials=number_of_trials,
            title=title,
            color_limit=(0.0, maximum_frequency),
            colorbar_label="bout frequency (s⁻¹)",
        )

        variant_path = (
            output_png.parent / f"{output_png.stem}{suffix}{output_png.suffix}"
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
            split_columns=(
                "epoch_name",
                "sign_group",
            ),
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
            stimulus_order=stimulus_order,
        )

        figure, axis = plt.subplots(
            figsize=(
                max(
                    20,
                    0.35 * len(classic_column_labels),
                ),
                max(
                    10,
                    0.32 * len(classic_row_labels),
                ),
            ),
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
            output_png.parent / f"{output_png.stem}_classic{output_png.suffix}"
        )

        figure.savefig(
            classic_path,
            dpi=180,
            bbox_inches="tight",
        )

        if not interactive:
            plt.close(figure)

        print(f"Saved {classic_path}")

    del per_fish
    del per_fish_classic

    if "averaged" in locals():
        del averaged
    if "pivot" in locals():
        del pivot
    if "classic_average" in locals():
        del classic_average
    if "classic_pivot" in locals():
        del classic_pivot

    gc.collect()

    # Fine-bin rows are collapsed over trials during construction.
    fine, _ = compute_bout_frequency_table(
        quality_control=quality_control,
        input_csv=input_csv,
        valid_trials=valid_trials,
        config_yaml=config_yaml,
        exclude_unusable=exclude_unusable,
        time_bins_key="fine_time_bins",
        average_trial_during_compute=True,
        include_classic=False,
    )

    if fine.empty:
        print("No fine-bin bout-frequency data were generated.")

        if interactive:
            plt.show()

        return

    fine_average = aggregate_bout_frequency(
        per_fish=fine,
        average_trial=False,
        average_time_bin=False,
    )

    fine_average.to_csv(
        output_png.parent / "bout_frequency_fine.csv",
        index=False,
    )

    del fine
    gc.collect()

    for stimulus in stimulus_order:
        data = fine_average.loc[fine_average["stim_name"] == stimulus]

        if data.empty:
            continue

        (
            pivot,
            groups,
            subgroups,
            labels,
            number_of_trials,
        ) = build_grouped_heatmap_matrix(
            averaged=data,
            row_column="bout_category",
            row_order=category_order,
            value_column="bout_frequency",
            stimulus_order=[stimulus],
        )

        figure, axis = plt.subplots(
            figsize=grouped_heatmap_figure_size(
                pivot.shape,
                minimum_width=12,
                minimum_height=6,
                row_height=0,
            ),
            layout="constrained",
        )

        plot_grouped_heatmap(
            figure=figure,
            axis=axis,
            pivot=pivot,
            row_group_labels=category_order,
            column_groups=groups,
            column_subgroups=subgroups,
            time_labels=labels,
            number_of_trials=number_of_trials,
            title=(f"{stimulus}: fine time bins, " "averaged over trials"),
            color_limit=(0.0, maximum_frequency),
            colorbar_label="bout frequency (s⁻¹)",
        )

        path = (
            output_png.parent / f"{output_png.stem}_fine_"
            f"{filename_safe(stimulus)}"
            f"{output_png.suffix}"
        )

        figure.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )

        if not interactive:
            plt.close(figure)

        print(f"Saved {path}")

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
