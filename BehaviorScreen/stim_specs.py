"""Shared stimulus specifications, filtering, and trial-quality utilities."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable

import pandas as pd
import yaml

from BehaviorScreen.core import Stim

FILTER_SECTIONS = {
    "bout": ("common_filters", "bout_filters"),
    "saccade": ("common_filters", "saccade_filters"),
}


def series_in(series: pd.Series, values: Any) -> pd.Series:
    """Return whether each series value occurs in `values`."""
    return series.isin(values)


def series_not_in(series: pd.Series, values: Any) -> pd.Series:
    """Return whether each series value does not occur in `values`."""
    return ~series.isin(values)


OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
    "in": series_in,
    "not_in": series_not_in,
}


@dataclass(frozen=True)
class Rule:
    """One DataFrame filtering rule."""

    column: str
    operator: str
    value: Any

    def get_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        """Evaluate this rule against a DataFrame."""
        if self.column not in dataframe.columns:
            raise ValueError(
                f"Column {self.column!r} required by the YAML "
                "configuration is missing."
            )

        if self.operator not in OPERATORS:
            raise ValueError(
                f"Unknown operator {self.operator!r}. "
                f"Available operators: {sorted(OPERATORS)}"
            )

        result = OPERATORS[self.operator](
            dataframe[self.column],
            self.value,
        )

        if isinstance(result, pd.Series):
            return result.reindex(dataframe.index)

        return pd.Series(result, index=dataframe.index)


@dataclass(frozen=True)
class RuleSet:
    """A collection of rules combined with logical AND."""

    rules: tuple[Rule, ...]

    def get_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        """Evaluate all rules in this rule set."""
        mask = pd.Series(True, index=dataframe.index)

        for rule in self.rules:
            mask &= rule.get_mask(dataframe).fillna(False).astype(bool)

        return mask

    def __repr__(self) -> str:
        if not self.rules:
            return "all"

        return "_".join(
            f"{rule.column}{rule.operator}{rule.value}" for rule in self.rules
        )


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


def parse_rules(config: dict | None) -> RuleSet:
    """Convert a YAML rule dictionary into a RuleSet."""
    rules = []

    for column, rule_config in (config or {}).items():
        if not isinstance(rule_config, dict):
            raise ValueError(f"Rules for column {column!r} must be a mapping.")

        for operator_name, value in rule_config.items():
            if operator_name not in OPERATORS:
                raise ValueError(
                    f"Unknown operator {operator_name!r} for " f"column {column!r}."
                )

            rules.append(
                Rule(
                    column=column,
                    operator=operator_name,
                    value=value,
                )
            )

    return RuleSet(tuple(rules))


def load_yaml_config(path: Path) -> dict:
    """Load a YAML analysis configuration."""
    with path.open("r") as input_file:
        config = yaml.safe_load(input_file)

    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")

    return config


Number = int | float
TimeBins = list[Number] | list[list[Number]]


def parse_time_bins(bins: TimeBins) -> list[tuple[Number, Number]]:
    """Parse [[start, stop], ...] or [start, stop, step]."""
    if not bins:
        return []

    if len(bins) == 3 and not isinstance(bins[0], list):
        start, stop, step = bins

        if stop <= start or step <= 0:
            raise ValueError("Time bins require stop > start and step > 0.")

        result = []
        while start < stop:
            end = min(start + step, stop)
            result.append((start, end))
            start = end
        return result

    return [tuple(bin_) for bin_ in bins]


def read_stim_specs(
    config: dict,
    ignore_time_bins: bool = False,
) -> Generator[StimSpec, None, None]:
    """Generate configured stimulus specifications."""
    global_time_bins = config.get("time_bins", [])

    if "stimuli" not in config:
        raise ValueError("The YAML configuration has no 'stimuli' section.")

    for entry in config["stimuli"]:
        try:
            stimulus = Stim[entry["stim"]]
        except KeyError as error:
            raise ValueError(f"Unknown stimulus: {entry['stim']}") from error

        name = entry["name"]

        raw_time_bins = entry.get(
            "time_bins",
            global_time_bins,
        )
        time_bins = parse_time_bins(
            raw_time_bins,
        )
        if not time_bins:
            raise ValueError(f"No time bins are defined for stimulus {name!r}.")

        parameters = tuple(
            parse_rules(parameter_config)
            for parameter_config in entry.get(
                "parameters",
                [{}],
            )
        )

        ranges = [None] if ignore_time_bins else time_bins

        for time_range in ranges:
            yield StimSpec(
                stim=stimulus,
                name=name,
                time_range=(None if time_range is None else tuple(time_range)),
                parameters=parameters,
            )


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
    """
    Return the number of presented trial slots for a stimulus spec.

    For pooled raw epoch names, this is the largest number of trials
    among the component epoch names.
    """
    epoch_names = list(epoch_names)

    rows = valid_trials.loc[
        (valid_trials["file"].astype(str) == str(fish))
        & valid_trials["epoch_name"].isin(epoch_names)
        & valid_trials["presented"]
    ]

    if rows.empty:
        return 0

    return int(rows.groupby("epoch_name").size().max())


def exclude_qc_fish(
    dataframe: pd.DataFrame,
    quality_control_path: Path,
    file_column: str = "file",
) -> pd.DataFrame:
    """Remove recordings listed in a fish-level QC CSV."""
    result = dataframe.copy()

    if not quality_control_path.exists():
        print(f"[qc] File does not exist; no fish excluded: " f"{quality_control_path}")
        return result

    if file_column not in result.columns:
        raise ValueError(f"The input table has no {file_column!r} column.")

    quality_control = pd.read_csv(quality_control_path)

    if "file" not in quality_control.columns:
        raise ValueError(f"{quality_control_path} has no 'file' column.")

    excluded_files = set(quality_control["file"].dropna().astype(str))

    before = len(result)

    result = result.loc[~result[file_column].astype(str).isin(excluded_files)].copy()

    removed = before - len(result)
    fraction = removed / before if before else 0.0

    print(
        f"[qc] Removed {removed:,}/{before:,} rows "
        f"({fraction:.2%}) from {len(excluded_files):,} "
        "excluded recordings."
    )

    return result


def apply_table_filters(
    dataframe: pd.DataFrame,
    config: dict,
    sections: tuple[str, ...],
) -> pd.DataFrame:
    """
    Apply selected YAML filter sections in order.

    Missing filter columns are errors. This prevents a misspelled or
    inappropriate filter from being silently ignored.
    """
    filtered = dataframe.copy()
    initial_count = len(filtered)

    for section in sections:
        section_config = config.get(section, {})

        if section_config is None:
            continue

        if not isinstance(section_config, dict):
            raise ValueError(f"YAML section {section!r} must be a mapping.")

        rule_set = parse_rules(section_config)

        for rule in rule_set.rules:
            if rule.column not in filtered.columns:
                raise ValueError(
                    f"Column {rule.column!r} required by YAML "
                    f"section {section!r} is missing."
                )

            before = len(filtered)

            mask = rule.get_mask(filtered).fillna(False).astype(bool)

            filtered = filtered.loc[mask].copy()

            removed = before - len(filtered)
            fraction_total = removed / initial_count if initial_count else 0.0

            print(
                f"[{section}] "
                f"{rule.column:28s} "
                f"{rule.operator:>6s} "
                f"{str(rule.value):20s} -> removed "
                f"{removed:7,d} "
                f"({fraction_total:6.2%})"
            )

    if initial_count:
        retained_fraction = len(filtered) / initial_count
        print(
            f"Filters retained {len(filtered):,}/"
            f"{initial_count:,} rows "
            f"({retained_fraction:.2%})."
        )

    return filtered


def apply_event_filters(
    dataframe: pd.DataFrame,
    config: dict,
    event_type: str,
) -> pd.DataFrame:
    """
    Apply common and event-specific filters.

    Parameters
    ----------
    event_type
        Either ``"bout"`` or ``"saccade"``.
    """
    if event_type not in FILTER_SECTIONS:
        raise ValueError(
            f"Unknown event type {event_type!r}. "
            f"Expected one of {sorted(FILTER_SECTIONS)}."
        )

    if "filters" in config:
        raise ValueError(
            "The YAML configuration still contains the legacy "
            "'filters' section. Split it into 'common_filters', "
            "'bout_filters', and 'saccade_filters'."
        )

    return apply_table_filters(
        dataframe=dataframe,
        config=config,
        sections=FILTER_SECTIONS[event_type],
    )


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


def get_single_value(
    dataframe: pd.DataFrame,
    column: str,
    group_name: str,
    default: Any = pd.NA,
) -> Any:
    """Return one unique non-null value from a recording group."""
    if column not in dataframe.columns:
        return default

    values = dataframe[column].dropna().unique()

    if len(values) == 0:
        return default

    if len(values) > 1:
        raise ValueError(
            f"Expected one value for {column!r} in "
            f"{group_name!r}, found {values!r}."
        )

    return values[0]
