"""Stimulus specifications of heatmaps.yaml, valid trials and epochs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from BehaviorScreen.core import Stim
from BehaviorScreen.filters import RuleSet, parse_rules


@dataclass(frozen=True)
class StimSpec:
    """One configured stimulus condition and time interval."""

    stim: Stim
    name: str
    time_range: tuple[float, float] | None
    parameters: tuple[RuleSet, ...]

    def get_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        """Evaluate parameter alternatives, combined with logical OR."""
        if not self.parameters:
            return pd.Series(True, index=dataframe.index)

        mask = pd.Series(False, index=dataframe.index)

        for rule_set in self.parameters:
            mask |= rule_set.get_mask(dataframe)

        return mask

    def __repr__(self) -> str:
        parameters = " | ".join(str(parameter) for parameter in self.parameters)
        return f"{self.name}[{parameters}]"


TimeBin = tuple[float, float]


def parse_time_bins(bins: list) -> list[TimeBin]:
    """
    Parse [[start, stop], ...] or [start, stop, step] into sorted,
    non-overlapping bins. Edges come from linspace, so they do not drift.
    """
    if not bins:
        return []

    if len(bins) == 3 and not isinstance(bins[0], list):
        start, stop, step = bins
        if stop <= start or step <= 0:
            raise ValueError("Time bins require stop > start and step > 0.")
        n = int(np.ceil((stop - start) / step - 1e-9))
        edges = np.minimum(np.linspace(start, start + n * step, n + 1), stop)
        edges = np.round(edges, 9).tolist()
        return list(zip(edges[:-1], edges[1:]))

    result = [(float(lo), float(hi)) for lo, hi in bins]
    if any(lo >= hi for lo, hi in result) or any(
        a[1] > b[0] for a, b in zip(result, result[1:])
    ):
        raise ValueError(f"Time bins must be increasing and non-overlapping: {bins}")
    return result


@dataclass(frozen=True)
class StimEntry:
    """One `stimuli` entry of the YAML: its selection rules and its time bins."""

    spec: StimSpec  # time_range is None
    bins: tuple[TimeBin, ...]

    @property
    def name(self) -> str:
        return self.spec.name


def read_stim_entries(config: dict, time_bins_key: str = "time_bins") -> list[StimEntry]:
    """Configured stimulus entries, in YAML order."""
    if "stimuli" not in config:
        raise ValueError("The YAML configuration has no 'stimuli' section.")

    entries = []
    for entry in config["stimuli"]:
        try:
            stimulus = Stim[entry["stim"]]
        except KeyError as error:
            raise ValueError(f"Unknown stimulus: {entry['stim']}") from error

        name = entry["name"]
        bins = parse_time_bins(entry.get(time_bins_key, config.get(time_bins_key, [])))
        if not bins:
            raise ValueError(f"No time bins are defined for stimulus {name!r}.")

        parameters = tuple(parse_rules(p) for p in entry.get("parameters", [{}]))
        entries.append(StimEntry(StimSpec(stimulus, name, None, parameters), tuple(bins)))
    return entries


def stimulus_name_order(config: dict) -> list[str]:
    """Return stimulus display names in YAML order."""
    names = []

    for entry in config["stimuli"]:
        name = entry["name"]

        if name not in names:
            names.append(name)

    return names


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Parse booleans safely from a CSV column."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()

    allowed = {
        "true",
        "false",
        "1",
        "0",
    }
    unexpected = set(normalized.dropna().unique()).difference(allowed)

    if unexpected:
        raise ValueError(f"Could not parse boolean values: {sorted(unexpected)}")

    return normalized.isin({"true", "1"})


def load_valid_trials(path: Path) -> pd.DataFrame:
    """Load trial presentation and tracking-quality information."""
    trials = pd.read_csv(path)

    required_columns = {
        "file",
        "epoch_name",
        "trial_num",
        "presented",
        "tracking_ok",
        "trial_duration_s",
    }
    missing_columns = required_columns.difference(trials.columns)

    if missing_columns:
        raise ValueError(f"{path} is missing columns: " f"{sorted(missing_columns)}")

    trials["file"] = trials["file"].astype(str)
    trials["epoch_name"] = trials["epoch_name"].astype(str)
    trials["trial_num"] = pd.to_numeric(
        trials["trial_num"],
        errors="raise",
    ).astype(int)

    trials["trial_duration_s"] = pd.to_numeric(trials["trial_duration_s"])
    trials["presented"] = parse_boolean_series(trials["presented"])
    trials["tracking_ok"] = parse_boolean_series(trials["tracking_ok"])
    trials["usable"] = trials["presented"] & trials["tracking_ok"]

    return trials


def get_explicit_epoch_names(specification: StimSpec) -> list[str]:
    """Extract raw epoch names explicitly named by a stimulus spec."""
    names = []

    for rule_set in specification.parameters:
        for rule in rule_set.rules:
            if rule.column != "epoch_name":
                continue

            if rule.operator == "==":
                names.append(str(rule.value))

            elif rule.operator == "in":
                names.extend(str(value) for value in rule.value)

    return list(dict.fromkeys(names))


def get_matching_epoch_names(
    events: pd.DataFrame,
    specification: StimSpec,
) -> list[str]:
    """Determine raw epoch names represented by a stimulus spec."""
    explicit_names = get_explicit_epoch_names(specification)

    if explicit_names:
        return explicit_names

    if "epoch_name" not in events.columns:
        raise ValueError(
            f"Stimulus {specification.name!r} has no explicit "
            "epoch_name rule, and the event table has no "
            "epoch_name column."
        )

    mask = specification.get_mask(events)

    if "stim" in events.columns:
        mask &= events["stim"] == specification.stim

    names = events.loc[mask, "epoch_name"].dropna().astype(str).unique().tolist()

    if not names:
        raise ValueError(
            f"Could not determine epoch names for "
            f"{specification!r}. Add an epoch_name rule "
            "to the YAML configuration."
        )

    return sorted(names)


def get_epoch_trial_count(
    valid_trials: pd.DataFrame,
    fish: str,
    epoch_names: Iterable[str],
) -> int:
    """Number of trial slots (highest presented trial_num + 1) for a stimulus spec."""
    rows = valid_trials.loc[
        (valid_trials["file"].astype(str) == str(fish))
        & valid_trials["epoch_name"].isin(list(epoch_names))
        & valid_trials["presented"]
    ]
    return 0 if rows.empty else int(rows["trial_num"].max()) + 1


def select_events(events: pd.DataFrame, specification: StimSpec) -> pd.DataFrame:
    """Events of this stimulus that satisfy all of its YAML parameter rules."""
    mask = (events["stim"] == specification.stim) & specification.get_mask(events)
    return events.loc[mask]


def restrict_trials_to_analysed_fish(valid_trials: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Keep trials of fish that produced events, so unprocessed fish don't count as zero-event fish."""
    return valid_trials.loc[valid_trials["file"].isin(events["file"].unique())]


def exclude_unusable_trials(
    events: pd.DataFrame,
    valid_trials: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only events assigned to usable trials."""
    join_columns = {
        "file",
        "epoch_name",
        "trial_num",
    }

    missing_event_columns = join_columns.difference(events.columns)
    missing_trial_columns = join_columns.difference(valid_trials.columns)

    if missing_event_columns:
        raise ValueError(
            "The event table is missing trial identity columns: "
            f"{sorted(missing_event_columns)}"
        )

    if missing_trial_columns:
        raise ValueError(
            "The valid-trials table is missing columns: "
            f"{sorted(missing_trial_columns)}"
        )

    usable_trials = (
        valid_trials.loc[
            valid_trials["usable"],
            list(join_columns),
        ]
        .drop_duplicates()
        .assign(_usable_trial=True)
    )

    before = len(events)

    result = events.merge(
        usable_trials,
        on=list(join_columns),
        how="left",
        sort=False,
        validate="many_to_one",
    )

    result = (
        result.loc[result["_usable_trial"].fillna(False)]
        .drop(columns="_usable_trial")
        .copy()
    )

    removed = before - len(result)
    fraction = removed / before if before else 0.0

    print(f"[trial QC] Removed {removed:,}/{before:,} events " f"({fraction:.2%}).")

    return result
