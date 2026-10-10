"""Shared utilities for behavior heatmaps and plotting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from BehaviorScreen.core import Laterality

LATERALITY_CODE_LABELS = {
    int(Laterality.IPSILATERAL): "ipsi",
    int(Laterality.CONTRALATERAL): "contra",
    int(Laterality.NONDIRECTIONAL): "none",
}

LATERALITY_ORDER = {
    "ipsi": 0,
    "contra": 1,
    "none": 2,
}

SIGN_ORDER = {
    "LEFT": 0,
    "RIGHT": 1,
}

SIGN_LABELS = [
    "LEFT",
    "RIGHT",
]

# average_trial, average_time_bin, filename suffix, title
HEATMAP_VARIANTS = [
    (
        False,
        False,
        "",
        "trial_x_time bin",
    ),
    (
        True,
        False,
        "_trial_avg",
        "averaged over trials",
    ),
    (
        False,
        True,
        "_timebin_avg",
        "averaged over time bins",
    ),
    (
        True,
        True,
        "_full_avg",
        "averaged over trials and time bins",
    ),
]


def order_lateralities(values: Any) -> list[str]:
    """Sort laterality labels in a stable display order."""
    return sorted(
        values,
        key=lambda value: LATERALITY_ORDER.get(value, 99),
    )


def order_signs(values: Any) -> list[str]:
    """Sort left/right labels in a stable display order."""
    return sorted(
        values,
        key=lambda value: SIGN_ORDER.get(value, 99),
    )


def map_laterality(series: pd.Series) -> pd.Series:
    """
    Map raw laterality codes to display labels.

    Explicit non-directional and missing values are placed in the ``none``
    group.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )
    mapped = numeric.map(LATERALITY_CODE_LABELS)

    return mapped.where(
        mapped.notna(),
        "none",
    )


def filename_safe(text: str) -> str:
    """Convert text to a filename-safe string."""
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in text
    )


def build_grouped_heatmap_matrix(
    averaged: pd.DataFrame,
    row_column: str,
    row_order: Sequence[Any],
    value_column: str,
    stimulus_order: Sequence[str],
    stimulus_column: str = "stim_name",
    laterality_column: str = "laterality_group",
    trial_column: str = "trial_idx",
    time_start_column: str = "time_bin_start",
    time_stop_column: str = "time_bin_stop",
) -> tuple[
    pd.DataFrame,
    list[tuple[int, int, str]],
    list[tuple[int, int, str]],
    list[str],
    int,
]:
    """
    Build a grouped stimulus/laterality/time-bin heatmap matrix.

    Columns are nested as:

        stimulus -> laterality -> time bin

    Rows are nested as:

        event category -> trial

    when trial-level values are retained.
    """
    if averaged.empty:
        raise ValueError("Cannot build a heatmap matrix from an empty table.")

    required_columns = {
        row_column,
        value_column,
        stimulus_column,
        laterality_column,
    }
    missing_columns = required_columns.difference(averaged.columns)

    if missing_columns:
        raise ValueError(
            "Heatmap data are missing columns: " f"{sorted(missing_columns)}"
        )

    has_trial = trial_column in averaged.columns
    has_time_bin = time_start_column in averaged.columns

    if has_time_bin and time_stop_column not in averaged.columns:
        raise ValueError(
            f"Heatmap data contain {time_start_column!r} "
            f"but not {time_stop_column!r}."
        )

    number_of_trials = int(averaged[trial_column].max()) + 1 if has_trial else 1

    columns: list[tuple[Any, ...]] = []
    column_groups: list[tuple[int, int, str]] = []
    column_subgroups: list[tuple[int, int, str]] = []
    time_labels: list[str] = []

    for stimulus in stimulus_order:
        stimulus_rows = averaged.loc[averaged[stimulus_column] == stimulus]

        if stimulus_rows.empty:
            continue

        stimulus_start = len(columns)

        lateralities = order_lateralities(
            stimulus_rows[laterality_column].dropna().unique().tolist()
        )

        for laterality in lateralities:
            laterality_rows = stimulus_rows.loc[
                stimulus_rows[laterality_column] == laterality
            ]

            laterality_start = len(columns)

            if has_time_bin:
                bins = (
                    laterality_rows[
                        [
                            time_start_column,
                            time_stop_column,
                        ]
                    ]
                    .drop_duplicates()
                    .sort_values(time_start_column)
                )

                for start, stop in bins.itertuples(index=False):
                    columns.append(
                        (
                            stimulus,
                            laterality,
                            start,
                        )
                    )
                    time_labels.append(f"{start:g}-{stop:g}s")
            else:
                columns.append(
                    (
                        stimulus,
                        laterality,
                    )
                )
                time_labels.append("avg")

            column_subgroups.append(
                (
                    laterality_start,
                    len(columns),
                    str(laterality),
                )
            )

        column_groups.append(
            (
                stimulus_start,
                len(columns),
                str(stimulus),
            )
        )

    if not columns:
        raise ValueError(
            "The heatmap matrix has no columns. Check the configured "
            "stimulus names and the filtered frequency table."
        )

    if has_trial:
        row_index: pd.Index = pd.MultiIndex.from_product(
            [
                row_order,
                range(number_of_trials),
            ],
            names=[
                row_column,
                trial_column,
            ],
        )
        pivot_index: str | list[str] = [
            row_column,
            trial_column,
        ]
    else:
        row_index = pd.Index(
            row_order,
            name=row_column,
        )
        pivot_index = row_column

    if has_time_bin:
        column_names = [
            stimulus_column,
            laterality_column,
            time_start_column,
        ]
    else:
        column_names = [
            stimulus_column,
            laterality_column,
        ]

    column_index = pd.MultiIndex.from_tuples(
        columns,
        names=column_names,
    )

    pivot = averaged.pivot_table(
        index=pivot_index,
        columns=column_names,
        values=value_column,
        aggfunc="mean",
        observed=True,
    )

    pivot = pivot.reindex(
        index=row_index,
        columns=column_index,
    )

    return (
        pivot,
        column_groups,
        column_subgroups,
        time_labels,
        number_of_trials,
    )


def plot_grouped_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    pivot: pd.DataFrame,
    row_group_labels: Sequence[str],
    column_groups: list[tuple[int, int, str]],
    column_subgroups: list[tuple[int, int, str]],
    time_labels: list[str],
    number_of_trials: int,
    title: str,
    color_limit: tuple[float, float],
    colorbar_label: str,
) -> None:
    """Draw a grouped stimulus/laterality/time-bin heatmap."""
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
        label=colorbar_label,
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
                linewidth=1.6,
            )

        axis.annotate(
            label,
            xy=(
                (start + stop - 1) / 2,
                1,
            ),
            xycoords=(
                "data",
                "axes fraction",
            ),
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
            xy=(
                (start + stop - 1) / 2,
                1,
            ),
            xycoords=(
                "data",
                "axes fraction",
            ),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            annotation_clip=False,
        )

    for group_index, group_label in enumerate(row_group_labels):
        row_start = group_index * number_of_trials
        row_stop = row_start + number_of_trials

        if row_start > 0:
            axis.axhline(
                row_start - 0.5,
                color="white",
                linewidth=1.6,
            )

        axis.annotate(
            str(group_label),
            xy=(
                0,
                (row_start + row_stop - 1) / 2,
            ),
            xycoords=(
                "axes fraction",
                "data",
            ),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            annotation_clip=False,
        )

    axis.set_title(
        title,
        fontsize=12,
        pad=62,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")


def grouped_heatmap_figure_size(
    shape: tuple[int, int],
    minimum_width: float = 16,
    minimum_height: float = 6,
    column_width: float = 0.22,
    row_height: float = 0.28,
) -> tuple[float, float]:
    """Calculate a heatmap figure size from matrix dimensions."""
    number_of_rows, number_of_columns = shape

    return (
        max(
            minimum_width,
            column_width * number_of_columns,
        ),
        max(
            minimum_height,
            row_height * number_of_rows,
        ),
    )
