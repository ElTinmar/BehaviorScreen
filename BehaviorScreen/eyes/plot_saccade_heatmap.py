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
    apply_table_filters,
    get_matching_epoch_names,
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


def get_single_value(
    dataframe: pd.DataFrame,
    column: str,
):
    """Return a unique non-null recording-level value."""
    if column not in dataframe.columns:
        return np.nan

    values = dataframe[column].dropna().unique()

    if len(values) == 1:
        return values[0]

    return np.nan


def remove_qc_fish(
    events: pd.DataFrame,
    valid_trials: pd.DataFrame,
    quality_control: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fish-level QC exclusion table."""
    if not quality_control.exists():
        return events, valid_trials

    qc = pd.read_csv(quality_control)

    if "file" not in qc.columns:
        raise ValueError(
            f"{quality_control} does not contain a file column."
        )

    excluded = set(qc["file"].astype(str))

    return (
        events[~events["file"].astype(str).isin(excluded)],
        valid_trials[
            ~valid_trials["file"].astype(str).isin(excluded)
        ],
    )


def filter_unusable_events(
    events: pd.DataFrame,
    valid_trials: pd.DataFrame,
) -> pd.DataFrame:
    """Remove events assigned to trials that are not usable."""
    usable = valid_trials.loc[
        valid_trials["usable"],
        ["file", "epoch_name", "trial_num"],
    ].drop_duplicates()

    result = events.merge(
        usable.assign(_usable=True),
        on=["file", "epoch_name", "trial_num"],
        how="left",
    )

    return result[
        result["_usable"].fillna(False)
    ].drop(columns="_usable")


def get_exposure_by_trial(
    valid_trials: pd.DataFrame,
    fish: str,
    epoch_names: list[str],
    duration: float,
    exclude_unusable_trials: bool,
) -> tuple[pd.Series, int]:
    """
    Calculate observed seconds for each pooled trial index.

    When multiple raw epoch names are pooled into one stimulus condition,
    their durations are added instead of treating them as one exposure.
    """
    trials = valid_trials[
        (valid_trials["file"].astype(str) == fish)
        & valid_trials["epoch_name"].isin(epoch_names)
        & valid_trials["presented"]
    ].copy()

    if trials.empty:
        return pd.Series(dtype=float), 0

    number_of_trials = int(trials["trial_num"].max()) + 1

    if exclude_unusable_trials:
        trials = trials[trials["usable"]]

    exposure = (
        trials.groupby("trial_num")
        .size()
        .astype(float)
        * duration
    )

    return exposure, number_of_trials


def compute_spec_counts(
    fish_events: pd.DataFrame,
    spec: StimSpec,
    number_of_trials: int,
    exposure_by_trial: pd.Series,
    class_order: list[int],
) -> pd.DataFrame:
    """Count saccade classes on a complete trial × class grid."""
    if spec.time_range is None:
        raise ValueError("Saccade heatmaps require time bins.")

    start, stop = spec.time_range
    duration = stop - start

    mask = (
        spec.get_mask(fish_events)
        & (fish_events["stim"] == spec.stim)
        & (fish_events["trial_time"] >= start)
        & (fish_events["trial_time"] < stop)
    )

    selected = fish_events.loc[mask].copy()
    selected = selected.dropna(
        subset=["trial_num", "cluster"]
    )

    selected["trial_idx"] = selected[
        "trial_num"
    ].astype(int)
    selected["cluster"] = selected[
        "cluster"
    ].astype(int)

    selected = selected[
        selected["trial_idx"] < number_of_trials
    ]
    selected = selected[
        selected["cluster"].isin(class_order)
    ]

    counts = (
        selected.groupby(["trial_idx", "cluster"])
        .size()
        .rename("saccade_counts")
    )

    full_index = pd.MultiIndex.from_product(
        [
            range(number_of_trials),
            class_order,
        ],
        names=["trial_idx", "cluster"],
    )

    result = (
        counts.reindex(full_index, fill_value=0)
        .reset_index()
    )

    result["exposure_s"] = (
        result["trial_idx"]
        .map(exposure_by_trial)
        .fillna(0.0)
    )

    result["saccade_frequency"] = np.where(
        result["exposure_s"] > 0,
        result["saccade_counts"]
        / result["exposure_s"],
        np.nan,
    )

    result["time_bin_duration"] = duration

    return result


def compute_saccade_frequency_table(
    input_csv: Path,
    valid_trials_csv: Path,
    quality_control: Path,
    config_yaml: Path,
    exclude_unusable_trials: bool = True,
    include_unassigned: bool = False,
) -> pd.DataFrame:
    """Construct a complete per-fish saccade-frequency table."""
    config = load_yaml_config(config_yaml)
    specifications = list(read_stim_specs(config))

    events = pd.read_csv(input_csv)
    valid_trials = load_valid_trials(valid_trials_csv)

    required = {
        "file",
        "stim",
        "epoch_name",
        "trial_num",
        "trial_time",
        "cluster",
    }
    missing = required.difference(events.columns)

    if missing:
        raise ValueError(
            f"{input_csv} is missing columns: {sorted(missing)}"
        )

    events, valid_trials = remove_qc_fish(
        events,
        valid_trials,
        quality_control,
    )

    events = apply_table_filters(events, config)

    if exclude_unusable_trials:
        events = filter_unusable_events(
            events,
            valid_trials,
        )

    class_order = DEFAULT_CLASS_ORDER.copy()

    if include_unassigned:
        class_order.insert(0, -1)

    epoch_names = {
        id(spec): get_matching_epoch_names(events, spec)
        for spec in specifications
    }

    tables = []

    # Use valid_trials as the fish universe. This retains fish with zero
    # saccades in a particular condition as genuine zero-frequency samples.
    fish_names = valid_trials["file"].astype(str).unique()

    for fish in tqdm(fish_names, desc="Saccade frequencies"):
        fish_events = events[
            events["file"].astype(str) == fish
        ]

        for spec in specifications:
            if spec.time_range is None:
                raise ValueError(
                    f"No time range is defined for {spec}."
                )

            start, stop = spec.time_range
            duration = stop - start
            matched_epoch_names = epoch_names[id(spec)]

            exposure, number_of_trials = (
                get_exposure_by_trial(
                    valid_trials=valid_trials,
                    fish=fish,
                    epoch_names=matched_epoch_names,
                    duration=duration,
                    exclude_unusable_trials=(
                        exclude_unusable_trials
                    ),
                )
            )

            if number_of_trials == 0:
                continue

            counts = compute_spec_counts(
                fish_events=fish_events,
                spec=spec,
                number_of_trials=number_of_trials,
                exposure_by_trial=exposure,
                class_order=class_order,
            )

            counts["file"] = fish
            counts["stim_name"] = spec.name
            counts["time_bin_start"] = start
            counts["time_bin_stop"] = stop
            counts["cluster_name"] = counts[
                "cluster"
            ].map(SACCADE_CLASS_NAMES)

            for column in (
                "dpf",
                "day",
                "cos_daytime",
                "sin_daytime",
            ):
                counts[column] = get_single_value(
                    fish_events,
                    column,
                )

            tables.append(counts)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)


def aggregate_saccade_frequency(
    per_fish: pd.DataFrame,
    average_trial: bool,
    average_time_bin: bool,
) -> pd.DataFrame:
    """Collapse trial/time dimensions within fish, then average fish."""
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

        working = (
            working.groupby(
                within_fish_columns,
                as_index=False,
                dropna=False,
            )[["saccade_counts", "exposure_s"]]
            .sum()
        )

        working["saccade_frequency"] = np.where(
            working["exposure_s"] > 0,
            working["saccade_counts"]
            / working["exposure_s"],
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

    return (
        working.groupby(
            across_fish_columns,
            as_index=False,
            dropna=False,
        )["saccade_frequency"]
        .mean()
    )


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
    """Build a class × stimulus/time matrix for plotting."""
    has_trial = "trial_idx" in averaged.columns
    has_time_bin = "time_bin_start" in averaged.columns

    number_of_trials = (
        int(averaged["trial_idx"].max()) + 1
        if has_trial and not averaged.empty
        else 1
    )

    columns = []
    column_groups = []
    time_labels = []

    for stimulus_name in stimulus_order:
        stimulus_rows = averaged[
            averaged["stim_name"] == stimulus_name
        ]

        if stimulus_rows.empty:
            continue

        group_start = len(columns)

        if has_time_bin:
            bins = (
                stimulus_rows[
                    ["time_bin_start", "time_bin_stop"]
                ]
                .drop_duplicates()
                .sort_values("time_bin_start")
            )

            for start, stop in bins.itertuples(index=False):
                columns.append((stimulus_name, start))
                time_labels.append(f"{start:g}-{stop:g}s")
        else:
            columns.append((stimulus_name,))
            time_labels.append("avg")

        column_groups.append(
            (
                group_start,
                len(columns),
                stimulus_name,
            )
        )

    if has_trial:
        row_index = pd.MultiIndex.from_product(
            [
                class_order,
                range(number_of_trials),
            ],
            names=["cluster", "trial_idx"],
        )
        index_columns = ["cluster", "trial_idx"]
    else:
        row_index = pd.Index(
            class_order,
            name="cluster",
        )
        index_columns = ["cluster"]

    if has_time_bin:
        column_names = [
            "stim_name",
            "time_bin_start",
        ]
    else:
        column_names = ["stim_name"]

    column_index = pd.MultiIndex.from_tuples(
        columns,
        names=column_names,
    )

    pivot = averaged.pivot_table(
        index=index_columns,
        columns=column_names,
        values="saccade_frequency",
    )

    pivot = pivot.reindex(
        index=row_index,
        columns=column_index,
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
        label="saccades per second",
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
            [
                trial_index
                for _, trial_index in pivot.index
            ],
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
            xy=(
                (start + stop - 1) / 2,
                1,
            ),
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
    exclude_unusable_trials: bool,
    include_unassigned: bool,
    maximum_frequency: float,
    interactive: bool,
) -> None:
    """Calculate per-fish rates and create all heatmap variants."""
    config = load_yaml_config(config_yaml)

    per_fish = compute_saccade_frequency_table(
        input_csv=input_csv,
        valid_trials_csv=valid_trials_csv,
        quality_control=quality_control,
        config_yaml=config_yaml,
        exclude_unusable_trials=exclude_unusable_trials,
        include_unassigned=include_unassigned,
    )

    if per_fish.empty:
        print("No saccade-frequency data were generated.")
        return

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            output_png.parent
            / f"saccade_frequency_avg{suffix}.csv",
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

        number_of_rows, number_of_columns = pivot.shape

        figure_width = max(
            14,
            0.25 * number_of_columns,
        )
        figure_height = max(
            6,
            0.25 * number_of_rows,
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
            output_png.parent
            / f"{output_png.stem}{suffix}"
            f"{output_png.suffix}"
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
            "Calculate and plot stimulus-aligned "
            "saccade-class frequencies."
        )
    )

    parser.add_argument("root", type=Path)
    parser.add_argument("yaml", type=Path)

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
        default=0.2,
        help="Maximum heatmap frequency in saccades/s.",
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
        valid_trials_csv=(
            args.root / args.valid_trials_csv
        ),
        quality_control=args.root / args.qc_csv,
        config_yaml=args.yaml,
        output_png=args.root / args.output,
        exclude_unusable_trials=(
            not args.include_unusable_trials
        ),
        include_unassigned=args.include_unassigned,
        maximum_frequency=args.vmax,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()