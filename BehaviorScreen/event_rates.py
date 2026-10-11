"""Per-fish event rates (bouts, saccades) in stimulus-aligned time bins."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from BehaviorScreen.core import Laterality
from BehaviorScreen.dataframe_utils import compact_dataframe
from BehaviorScreen.load import fish_metadata
from BehaviorScreen.stim_specs import (
    StimEntry,
    StimSpec,
    apply_event_filters,
    exclude_qc_fish,
    exclude_unusable_trials,
    get_epoch_trial_count,
    get_matching_epoch_names,
    read_stim_entries,
    restrict_trials_to_analysed_fish,
    select_events,
)

LATERALITY_LABELS = {
    int(Laterality.IPSILATERAL): "ipsi",
    int(Laterality.CONTRALATERAL): "contra",
    int(Laterality.NONDIRECTIONAL): "none",
}
LATERALITY_ORDER = ["ipsi", "contra", "none"]


def map_laterality(series: pd.Series) -> pd.Series:
    """Laterality codes to labels; missing values go to "none"."""
    labels = pd.to_numeric(series, errors="coerce").map(LATERALITY_LABELS)
    return labels.where(labels.notna(), "none")


def order_lateralities(values: Iterable[str]) -> list[str]:
    present = set(values)
    return [label for label in LATERALITY_ORDER if label in present]


@dataclass(frozen=True)
class EventKind:
    """
    What is being counted. `assign_category` adds the `category` column (and
    any kind-specific split columns); values outside `categories` are dropped.
    """

    name: str  # also the YAML filter section: "<name>_filters"
    category: str
    categories: list
    labels: list[str]
    assign_category: Callable[[pd.DataFrame], pd.DataFrame]
    required_columns: frozenset[str] = frozenset()
    levels: dict[str, list[str]] = field(default_factory=dict)
    category_names: dict | None = None

    @property
    def counts(self) -> str:
        return f"{self.name}_counts"

    @property
    def frequency(self) -> str:
        return f"{self.name}_frequency"


@dataclass
class PreparedEvents:
    """Filtered events and the trials of the analysed fish, plus per-entry labels."""

    events: pd.DataFrame
    trials: pd.DataFrame
    epoch_names: list[list[str]]
    lateralities: list[list[str]]
    usable_only: bool


def load_events(path: Path, kind: EventKind) -> pd.DataFrame:
    events = pd.read_csv(path)
    required = {"file", "stim", "epoch_name", "trial_num", "trial_time"} | kind.required_columns
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    print(f"Total number of {kind.name}s: {len(events):,}")
    return events


def laterality_labels(events: pd.DataFrame, spec: StimSpec) -> list[str]:
    if "laterality" not in events.columns:
        return ["none"]
    labels = map_laterality(select_events(events, spec)["laterality"]).unique()
    return order_lateralities(labels) or ["none"]


def prepare_events(
    kind: EventKind,
    events: pd.DataFrame,
    valid_trials: pd.DataFrame,
    quality_control: Path,
    config: dict,
    exclude_unusable: bool = True,
) -> PreparedEvents:
    specs = [entry.spec for entry in read_stim_entries(config)]

    events = exclude_qc_fish(events, quality_control).astype({"file": str})
    trials = exclude_qc_fish(valid_trials, quality_control).astype({"file": str})
    trials = restrict_trials_to_analysed_fish(trials, events, config)

    # Before event filters, so that an epoch with no surviving event still counts.
    epoch_names = [get_matching_epoch_names(events, spec) for spec in specs]
    events = apply_event_filters(events, config, kind.name)
    lateralities = [laterality_labels(events, spec) for spec in specs]

    if exclude_unusable:
        events = exclude_unusable_trials(events, trials)

    events = kind.assign_category(events)
    events["trial_idx"] = pd.to_numeric(events["trial_num"], errors="coerce")
    events = events.loc[
        events["trial_idx"].notna() & events[kind.category].isin(kind.categories)
    ].astype({"trial_idx": np.int32})
    events["laterality_group"] = (
        map_laterality(events["laterality"]) if "laterality" in events else "none"
    )
    categorical = ["file", "stim", "epoch_name", kind.category, "laterality_group", *kind.levels]
    events = events.astype({column: "category" for column in categorical})

    return PreparedEvents(events, trials, epoch_names, lateralities, exclude_unusable)


def _count_events(
    kind: EventKind,
    events: pd.DataFrame,
    entry: StimEntry,
    n_trials: int,
    split: list[str],
    levels: list[list[str]],
) -> pd.DataFrame:
    """Event counts on the complete (trial, bin, category, *split) grid."""
    starts, stops = np.array(entry.bins).T
    t = events["trial_time"].to_numpy(float)
    bin_index = np.searchsorted(starts, t, side="right") - 1
    trial = events["trial_idx"].to_numpy()
    keep = (
        (bin_index >= 0)
        & (t < stops[np.clip(bin_index, 0, None)])
        & (trial >= 0)
        & (trial < n_trials)
    )

    keys = ["trial_idx", "bin", kind.category, *split]
    counts = (
        events.loc[keep, ["trial_idx", kind.category, *split]]
        .assign(bin=bin_index[keep])
        .groupby(keys, observed=True)
        .size()
    )
    grid = pd.MultiIndex.from_product(
        [range(n_trials), range(len(starts)), kind.categories, *levels], names=keys
    )
    return counts.reindex(grid, fill_value=0).rename(kind.counts).reset_index()


def _exposure(
    trials: pd.DataFrame,
    entry: StimEntry,
    epoch_names: list[str],
    usable_only: bool,
    by_epoch: bool,
) -> pd.Series:
    """
    Seconds of stimulus inside each bin, per trial slot (summed over the
    pooled epochs) or per (trial slot, epoch). Trials are selected with the
    same YAML rules as events.
    """
    starts, stops = np.array(entry.bins).T
    rows = trials.loc[
        trials["epoch_name"].isin(epoch_names)
        & trials["usable" if usable_only else "presented"]
        & entry.spec.get_mask(trials)
    ]
    duration = rows["trial_duration_s"].to_numpy(float)[:, None]
    overlap = np.clip(np.minimum(duration, stops) - starts, 0, None)
    index = pd.MultiIndex.from_arrays(
        [rows["epoch_name"].astype(str), rows["trial_num"]], names=["epoch_name", "trial_idx"]
    )
    exposure = pd.DataFrame(overlap, index=index).rename_axis(columns="bin").stack()
    return exposure.groupby(level=["trial_idx", "bin", *(["epoch_name"] if by_epoch else [])]).sum()


def compact_rate_table(kind: EventKind, table: pd.DataFrame) -> pd.DataFrame:
    return compact_dataframe(
        table,
        categorical_columns=(
            "file",
            "day",
            "stim_name",
            "epoch_name",
            "laterality_group",
            "sign_group",
            "bout_category",
            f"{kind.category}_name",
        ),
        integer_columns=("trial_idx", kind.counts, "dpf", "saccade_category"),
        float_columns=(
            kind.frequency,
            "time_bin_start",
            "time_bin_stop",
            "time_bin_duration",
            "cos_daytime",
            "sin_daytime",
        ),
    )


def _with_frequency(kind: EventKind, table: pd.DataFrame) -> pd.DataFrame:
    duration = table["time_bin_duration"]
    table[kind.frequency] = table[kind.counts] / duration.where(duration > 0)
    if kind.category_names is not None:
        table[f"{kind.category}_name"] = table[kind.category].map(kind.category_names)
    return table


def frequency_table(
    kind: EventKind,
    prepared: PreparedEvents,
    config: dict,
    time_bins_key: str = "time_bins",
    split: Sequence[str] = ("laterality_group",),
    collapse_trials: bool = False,
) -> pd.DataFrame:
    """
    Per-fish counts, exposure and frequency for every YAML stimulus entry and
    time bin. Fish with trials but no events contribute zeros.
    `collapse_trials` sums over trials while building (for fine bins).
    """
    split = list(split)
    by_epoch = "epoch_name" in split
    entries = read_stim_entries(config, time_bins_key)
    events_by_fish = {
        str(fish): group
        for fish, group in prepared.events.groupby("file", observed=True, sort=False)
    }
    no_events = prepared.events.iloc[0:0]
    tables: list[pd.DataFrame] = []

    for fish, fish_trials in prepared.trials.groupby("file", sort=False):
        fish_events = events_by_fish.get(str(fish), no_events)
        metadata = fish_metadata(str(fish))

        for i, entry in enumerate(entries):
            epoch_names = prepared.epoch_names[i]
            n_trials = get_epoch_trial_count(fish_trials, str(fish), epoch_names)
            if n_trials == 0:
                continue

            levels = [
                prepared.lateralities[i]
                if column == "laterality_group"
                else epoch_names
                if column == "epoch_name"
                else kind.levels[column]
                for column in split
            ]
            table = _count_events(
                kind, select_events(fish_events, entry.spec), entry, n_trials, split, levels
            )
            exposure = _exposure(fish_trials, entry, epoch_names, prepared.usable_only, by_epoch)
            keys = ["trial_idx", "bin", *(["epoch_name"] if by_epoch else [])]
            table["time_bin_duration"] = (
                exposure.reindex(pd.MultiIndex.from_frame(table[keys])).fillna(0.0).to_numpy()
            )

            if collapse_trials:
                table = table.groupby(["bin", kind.category, *split], sort=False, as_index=False)[
                    [kind.counts, "time_bin_duration"]
                ].sum()

            starts, stops = np.array(entry.bins).T
            table = table.assign(
                **metadata,
                stim_name=entry.name,
                time_bin_start=starts[table["bin"]],
                time_bin_stop=stops[table["bin"]],
            ).drop(columns="bin")
            tables.append(_with_frequency(kind, table))

    if not tables:
        return pd.DataFrame()
    return compact_rate_table(kind, pd.concat(tables, ignore_index=True))


def aggregate(
    kind: EventKind,
    per_fish: pd.DataFrame,
    average_trial: bool,
    average_time_bin: bool,
    split: Sequence[str] = ("laterality_group",),
) -> pd.DataFrame:
    """
    Pool counts and exposure within fish over the averaged dimensions, then
    average frequencies across fish.
    """
    has_trial = "trial_idx" in per_fish.columns
    keys = ["stim_name", kind.category, *split]
    if not average_time_bin:
        keys += ["time_bin_start", "time_bin_stop"]
    if has_trial and not average_trial:
        keys.append("trial_idx")

    working = per_fish
    if (average_trial and has_trial) or average_time_bin:
        working = per_fish.groupby(
            ["file", *keys], as_index=False, dropna=False, sort=False, observed=True
        )[[kind.counts, "time_bin_duration"]].sum()
        working = _with_frequency(kind, working)

    result = working.groupby(keys, as_index=False, dropna=False, sort=False, observed=True)[
        kind.frequency
    ].mean()
    if kind.category_names is not None:
        result[f"{kind.category}_name"] = result[kind.category].map(kind.category_names)
    return compact_rate_table(kind, result)
