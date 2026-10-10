#!/usr/bin/env python3
"""Calculate and plot stimulus-aligned saccade-class frequencies."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.core import SACCADE_CATEGORY_NAMES
from BehaviorScreen.dataframe_utils import compact_dataframe
from BehaviorScreen.plot_utils import (
    HEATMAP_VARIANTS,
    build_grouped_heatmap_matrix,
    filename_safe,
    grouped_heatmap_figure_size,
    map_laterality,
    order_lateralities,
    plot_grouped_heatmap,
)
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

DEFAULT_CLASS_ORDER = [
    1,  # Conjugate
    3,  # Miniature convergent
    4,  # Convergent
    6,  # Divergent
    7,  # Biphasic convergent
]


def compact_saccade_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Use compact dtypes for a saccade-frequency table."""
    return compact_dataframe(
        dataframe=dataframe,
        categorical_columns=(
            "file",
            "stim_name",
            "saccade_category_name",
            "laterality_group",
        ),
        integer_columns=(
            "trial_idx",
            "saccade_category",
            "saccade_counts",
            "dpf",
            "day",
        ),
        float_columns=(
            "saccade_frequency",
            "exposure_s",
            "time_bin_start",
            "time_bin_stop",
            "time_bin_duration",
            "cos_daytime",
            "sin_daytime",
        ),
    )


def preprocess_saccades(
    events: pd.DataFrame,
    class_order: list[int],
) -> pd.DataFrame:
    """
    Prepare reusable numeric and mapped columns once.

    Classes outside ``class_order`` are removed here rather than tested for
    every fish, specification, and time bin.
    """
    events = events.copy()
    events["file"] = events["file"].astype(str)

    events["trial_idx"] = pd.to_numeric(
        events["trial_num"],
        errors="coerce",
    )
    events["saccade_category"] = pd.to_numeric(
        events["saccade_category"],
        errors="coerce",
    )

    events = events.dropna(
        subset=[
            "trial_idx",
            "saccade_category",
        ]
    ).copy()

    events["trial_idx"] = events["trial_idx"].astype(np.int32)
    events["saccade_category"] = events["saccade_category"].astype(np.int16)

    events = events.loc[events["saccade_category"].isin(class_order)].copy()

    events["laterality_group"] = map_laterality(events["laterality"])

    # Specification regex matching has already been completed.
    for column in (
        "file",
        "stim",
        "epoch_name",
        "laterality_group",
    ):
        if column in events.columns:
            events[column] = events[column].astype("category")

    return events


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
    """Determine laterality labels from resolved epoch names."""
    if "laterality" not in events.columns:
        return ["none"]

    mask = (events["stim"] == specification.stim) & events["epoch_name"].isin(
        matched_epoch_names
    )

    labels = map_laterality(events.loc[mask, "laterality"]).dropna().unique().tolist()

    return order_lateralities(labels) if labels else ["none"]


def get_trial_presentation_counts(
    fish_trials: pd.DataFrame,
    epoch_names: list[str],
    exclude_unusable: bool,
) -> tuple[pd.Series, int]:
    """
    Return valid raw-presentation counts for each pooled trial.

    Presentation counts do not depend on time-bin duration and can therefore
    be cached across every time bin using the same epoch-name group.
    """
    trials = fish_trials.loc[
        fish_trials["epoch_name"].isin(epoch_names) & fish_trials["presented"]
    ]

    if trials.empty:
        return (
            pd.Series(dtype=np.float32),
            0,
        )

    trial_numbers = pd.to_numeric(
        trials["trial_num"],
        errors="coerce",
    )
    valid_trial_numbers = trial_numbers.dropna()

    if valid_trial_numbers.empty:
        return (
            pd.Series(dtype=np.float32),
            0,
        )

    number_of_trials = int(valid_trial_numbers.max()) + 1

    if exclude_unusable:
        trials = trials.loc[trials["usable"]]

    if trials.empty:
        return (
            pd.Series(dtype=np.float32),
            number_of_trials,
        )

    presentation_counts = (
        trials.groupby(
            "trial_num",
            sort=False,
            observed=True,
        )
        .size()
        .astype(np.float32)
    )

    return presentation_counts, number_of_trials


def compute_spec_counts(
    fish_spec_events: pd.DataFrame,
    specification: StimSpec,
    number_of_trials: int,
    exposure_by_trial: pd.Series,
    class_order: list[int],
    laterality_labels: list[str],
    average_trial: bool = False,
) -> pd.DataFrame:
    """
    Count saccades on a complete class/laterality grid.

    ``fish_spec_events`` must already be restricted to the relevant fish,
    stimulus, matching epoch names, and requested saccade classes.

    When ``average_trial`` is true, trial counts and exposure are collapsed
    during construction.
    """
    if specification.time_range is None:
        raise ValueError("Saccade heatmaps require time bins.")

    start, stop = specification.time_range
    duration = stop - start

    if duration <= 0:
        raise ValueError(
            f"Invalid time interval for {specification}: " f"{specification.time_range}"
        )

    mask = (
        (fish_spec_events["trial_time"] >= start)
        & (fish_spec_events["trial_time"] < stop)
        & (fish_spec_events["trial_idx"] >= 0)
        & (fish_spec_events["trial_idx"] < number_of_trials)
    )

    selected = fish_spec_events.loc[
        mask,
        [
            "trial_idx",
            "saccade_category",
            "laterality_group",
        ],
    ]

    if average_trial:
        counts = (
            selected.groupby(
                [
                    "saccade_category",
                    "laterality_group",
                ],
                sort=False,
                observed=True,
            )
            .size()
            .rename("saccade_counts")
        )

        full_index = pd.MultiIndex.from_product(
            [
                class_order,
                laterality_labels,
            ],
            names=[
                "saccade_category",
                "laterality_group",
            ],
        )

        result = counts.reindex(
            full_index,
            fill_value=0,
        ).reset_index()

        result["exposure_s"] = float(
            exposure_by_trial.reindex(
                range(number_of_trials),
                fill_value=0.0,
            ).sum()
        )
    else:
        counts = (
            selected.groupby(
                [
                    "trial_idx",
                    "saccade_category",
                    "laterality_group",
                ],
                sort=False,
                observed=True,
            )
            .size()
            .rename("saccade_counts")
        )

        full_index = pd.MultiIndex.from_product(
            [
                range(number_of_trials),
                class_order,
                laterality_labels,
            ],
            names=[
                "trial_idx",
                "saccade_category",
                "laterality_group",
            ],
        )

        result = counts.reindex(
            full_index,
            fill_value=0,
        ).reset_index()

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
    time_bins_key: str = "time_bins",
    average_trial_during_compute: bool = False,
) -> pd.DataFrame:
    """Construct the per-fish saccade-frequency table."""
    config = load_yaml_config(config_yaml)
    specifications = list(
        read_stim_specs(
            config,
            time_bins_key=time_bins_key,
        )
    )

    events = pd.read_csv(input_csv)
    valid_trials = load_valid_trials(valid_trials_csv)

    required_columns = {
        "file",
        "stim",
        "epoch_name",
        "trial_num",
        "trial_time",
        "saccade_category",
        "saccade_category_name",
        "laterality",
    }
    missing_columns = required_columns.difference(events.columns)

    if missing_columns:
        raise ValueError(
            f"{input_csv} is missing columns: "
            f"{sorted(missing_columns)}. "
            "Re-run augment_saccades.py to add laterality."
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

    events = events.copy()
    valid_trials = valid_trials.copy()

    events["file"] = events["file"].astype(str)
    valid_trials["file"] = valid_trials["file"].astype(str)

    # Resolve regex/specification membership once before event filtering.
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
                events=events,
                specification=specification,
                matched_epoch_names=matched_epoch_names,
            )

        laterality_labels[id(specification)] = laterality_cache[selection_key]

    if exclude_unusable:
        events = exclude_unusable_trials(
            events=events,
            valid_trials=valid_trials,
        )

    class_order = DEFAULT_CLASS_ORDER.copy()

    if include_unassigned:
        class_order.insert(0, -1)

    events = preprocess_saccades(
        events=events,
        class_order=class_order,
    )

    events_by_fish = {
        str(fish): group
        for fish, group in events.groupby(
            "file",
            sort=False,
            observed=True,
        )
    }

    valid_trials_by_fish = {
        str(fish): group
        for fish, group in valid_trials.groupby(
            "file",
            sort=False,
            observed=True,
        )
    }

    empty_events = events.iloc[0:0]
    tables: list[pd.DataFrame] = []

    for fish, fish_trials in tqdm(
        valid_trials_by_fish.items(),
        total=len(valid_trials_by_fish),
        desc="Saccade frequencies",
    ):
        fish_events = events_by_fish.get(
            fish,
            empty_events,
        )

        metadata = {
            column: get_single_value(
                fish_events,
                column,
                fish,
            )
            for column in (
                "dpf",
                "day",
                "cos_daytime",
                "sin_daytime",
            )
        }

        # These caches are local to one fish.
        event_selection_cache: dict[
            tuple[Any, tuple[str, ...]],
            pd.DataFrame,
        ] = {}
        presentation_cache: dict[
            tuple[str, ...],
            tuple[pd.Series, int],
        ] = {}

        for specification in specifications:
            if specification.time_range is None:
                raise ValueError("No time range is defined for " f"{specification}.")

            start, stop = specification.time_range
            duration = stop - start

            matched_epoch_names = epoch_names[id(specification)]
            epoch_key = tuple(matched_epoch_names)
            selection_key = specification_selection_key(
                specification,
                matched_epoch_names,
            )

            if selection_key not in event_selection_cache:
                selection_mask = (
                    fish_events["stim"] == specification.stim
                ) & fish_events["epoch_name"].isin(matched_epoch_names)

                event_selection_cache[selection_key] = fish_events.loc[selection_mask]

            fish_spec_events = event_selection_cache[selection_key]

            if epoch_key not in presentation_cache:
                presentation_cache[epoch_key] = get_trial_presentation_counts(
                    fish_trials=fish_trials,
                    epoch_names=matched_epoch_names,
                    exclude_unusable=exclude_unusable,
                )

            (
                presentation_counts,
                number_of_trials,
            ) = presentation_cache[epoch_key]

            if number_of_trials == 0:
                continue

            # Only exposure duration varies between time bins.
            exposure = presentation_counts * duration

            counts = compute_spec_counts(
                fish_spec_events=fish_spec_events,
                specification=specification,
                number_of_trials=number_of_trials,
                exposure_by_trial=exposure,
                class_order=class_order,
                laterality_labels=laterality_labels[id(specification)],
                average_trial=average_trial_during_compute,
            )

            counts["file"] = fish
            counts["stim_name"] = specification.name
            counts["time_bin_start"] = start
            counts["time_bin_stop"] = stop
            counts["saccade_category_name"] = counts["saccade_category"].map(
                SACCADE_CATEGORY_NAMES
            )

            for column, value in metadata.items():
                counts[column] = value

            tables.append(counts)

    if not tables:
        return pd.DataFrame()

    result = pd.concat(
        tables,
        ignore_index=True,
        copy=False,
    )

    return compact_saccade_table(result)


def aggregate_saccade_frequency(
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
            "saccade_category",
            "saccade_category_name",
            *split_columns_list,
        ]

        if has_trial and not average_trial:
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
            sort=False,
            observed=True,
        )[
            [
                "saccade_counts",
                "exposure_s",
            ]
        ].sum()

        working["saccade_frequency"] = np.where(
            working["exposure_s"] > 0,
            working["saccade_counts"] / working["exposure_s"],
            np.nan,
        )

    across_fish_columns = [
        "stim_name",
        "saccade_category",
        "saccade_category_name",
        *split_columns_list,
    ]

    if has_trial and not average_trial:
        across_fish_columns.append("trial_idx")

    if not average_time_bin:
        across_fish_columns.extend(
            [
                "time_bin_start",
                "time_bin_stop",
            ]
        )

    result = working.groupby(
        across_fish_columns,
        as_index=False,
        dropna=False,
        sort=False,
        observed=True,
    )["saccade_frequency"].mean()

    return compact_saccade_table(result)


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

        if interactive:
            plt.show()

        return

    per_fish.to_csv(
        output_png.parent / "saccade_frequency.csv",
        index=False,
    )

    class_order = DEFAULT_CLASS_ORDER.copy()

    if include_unassigned:
        class_order.insert(0, -1)

    row_group_labels = [
        SACCADE_CATEGORY_NAMES.get(
            category,
            str(category),
        )
        for category in class_order
    ]

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
            groups,
            subgroups,
            labels,
            number_of_trials,
        ) = build_grouped_heatmap_matrix(
            averaged=averaged,
            row_column="saccade_category",
            row_order=class_order,
            value_column="saccade_frequency",
            stimulus_order=stimulus_order,
        )

        if not np.isfinite(pivot.to_numpy(dtype=float)).any():
            raise ValueError("The saccade heatmap matrix " "contains only NaN values.")

        figure, axis = plt.subplots(
            figsize=grouped_heatmap_figure_size(pivot.shape),
            layout="constrained",
        )

        plot_grouped_heatmap(
            figure=figure,
            axis=axis,
            pivot=pivot,
            row_group_labels=row_group_labels,
            column_groups=groups,
            column_subgroups=subgroups,
            time_labels=labels,
            number_of_trials=number_of_trials,
            title=title,
            color_limit=(0.0, maximum_frequency),
            colorbar_label="saccade frequency (s⁻¹)",
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

    del per_fish

    if "averaged" in locals():
        del averaged
    if "pivot" in locals():
        del pivot

    gc.collect()

    # Fine-bin rows are collapsed over trials during construction.
    fine = compute_saccade_frequency_table(
        input_csv=input_csv,
        valid_trials_csv=valid_trials_csv,
        quality_control=quality_control,
        config_yaml=config_yaml,
        exclude_unusable=exclude_unusable,
        include_unassigned=include_unassigned,
        time_bins_key="fine_time_bins",
        average_trial_during_compute=True,
    )

    if fine.empty:
        print("No fine-bin saccade-frequency data were generated.")

        if interactive:
            plt.show()

        return

    fine_average = aggregate_saccade_frequency(
        per_fish=fine,
        average_trial=False,
        average_time_bin=False,
    )

    fine_average.to_csv(
        output_png.parent / "saccade_frequency_fine.csv",
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
            row_column="saccade_category",
            row_order=class_order,
            value_column="saccade_frequency",
            stimulus_order=[stimulus],
        )

        if not np.isfinite(pivot.to_numpy(dtype=float)).any():
            print(f"Skipping {stimulus}: fine-bin matrix " "contains only NaN values.")
            continue

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
            row_group_labels=row_group_labels,
            column_groups=groups,
            column_subgroups=subgroups,
            time_labels=labels,
            number_of_trials=number_of_trials,
            title=(f"{stimulus}: fine time bins, " "averaged over trials"),
            color_limit=(0.0, maximum_frequency),
            colorbar_label="saccade frequency (s⁻¹)",
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
