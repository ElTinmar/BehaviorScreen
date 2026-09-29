from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import yaml

from BehaviorScreen.core import Stim


def series_in(series: pd.Series, values: Any) -> pd.Series:
    """Return whether each value occurs in `values`."""
    return series.isin(values)


def series_not_in(series: pd.Series, values: Any) -> pd.Series:
    """Return whether each value does not occur in `values`."""
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
        """Evaluate the rule against a DataFrame."""
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

        return OPERATORS[self.operator](
            dataframe[self.column],
            self.value,
        )


@dataclass(frozen=True)
class RuleSet:
    """A collection of rules combined using logical AND."""

    rules: tuple[Rule, ...]

    def get_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        """Evaluate all rules in the set."""
        mask = pd.Series(True, index=dataframe.index)

        for rule in self.rules:
            mask &= rule.get_mask(dataframe)

        return mask

    def __repr__(self) -> str:
        if not self.rules:
            return "all"

        return "_".join(
            f"{rule.column}{rule.operator}{rule.value}"
            for rule in self.rules
        )


@dataclass(frozen=True)
class StimSpec:
    """One configured stimulus condition and time interval."""

    stim: Stim
    name: str
    time_range: tuple[float, float] | None
    parameters: tuple[RuleSet, ...]

    def get_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        """Evaluate parameter alternatives, combined using logical OR."""
        if not self.parameters:
            return pd.Series(True, index=dataframe.index)

        mask = pd.Series(False, index=dataframe.index)

        for rule_set in self.parameters:
            mask |= rule_set.get_mask(dataframe)

        return mask

    def __repr__(self) -> str:
        parameters = " | ".join(
            str(parameter) for parameter in self.parameters
        )
        return f"{self.name}[{parameters}]"


def parse_rules(config: dict | None) -> RuleSet:
    """Convert a YAML rule dictionary into a RuleSet."""
    rules = []

    for column, rule_config in (config or {}).items():
        for operator_name, value in rule_config.items():
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


def read_stim_specs(
    config: dict,
    ignore_time_bins: bool = False,
) -> Generator[StimSpec, None, None]:
    """Generate configured stimulus specifications."""
    global_time_bins = config.get("time_bins", [])

    for entry in config["stimuli"]:
        try:
            stimulus = Stim[entry["stim"]]
        except KeyError as error:
            raise ValueError(
                f"Unknown stimulus: {entry['stim']}"
            ) from error

        name = entry["name"]
        time_bins = entry.get("time_bins", global_time_bins)

        if not time_bins:
            raise ValueError(
                f"No time bins are defined for stimulus {name!r}."
            )

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
                time_range=(
                    None
                    if time_range is None
                    else tuple(time_range)
                ),
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

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {"true", "false", "1", "0"}
    unexpected = set(normalized.unique()).difference(allowed)

    if unexpected:
        raise ValueError(
            f"Could not parse boolean values: {sorted(unexpected)}"
        )

    return normalized.isin({"true", "1"})


def load_valid_trials(path: Path) -> pd.DataFrame:
    """Load trial presentation and tracking-quality information."""
    trials = pd.read_csv(path)

    required = {
        "file",
        "epoch_name",
        "trial_num",
        "presented",
        "tracking_ok",
    }
    missing = required.difference(trials.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    trials["presented"] = parse_boolean_series(
        trials["presented"]
    )
    trials["tracking_ok"] = parse_boolean_series(
        trials["tracking_ok"]
    )
    trials["usable"] = (
        trials["presented"]
        & trials["tracking_ok"]
    )

    return trials


def get_explicit_epoch_names(spec: StimSpec) -> list[str]:
    """
    Extract raw epoch names explicitly named in a stimulus specification.

    Prefer configurations that contain rules such as:

        epoch_name:
          in: ["grating left", "grating right"]
    """
    names = []

    for rule_set in spec.parameters:
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
    spec: StimSpec,
) -> list[str]:
    """Determine raw epoch names represented by a stimulus specification."""
    explicit_names = get_explicit_epoch_names(spec)

    if explicit_names:
        return explicit_names

    if "epoch_name" not in events.columns:
        raise ValueError(
            f"Stimulus {spec.name!r} has no explicit epoch_name rule, "
            "and the event table has no epoch_name column."
        )

    mask = spec.get_mask(events)

    if "stim" in events.columns:
        mask &= events["stim"] == spec.stim

    names = (
        events.loc[mask, "epoch_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not names:
        raise ValueError(
            f"Could not determine epoch names for {spec!r}. "
            "Add an epoch_name rule to the YAML configuration."
        )

    return sorted(names)


def apply_table_filters(
    dataframe: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Apply global YAML filters to an event table."""
    filtered = dataframe.copy()
    filters = parse_rules(config.get("filters", {}))

    initial_count = len(filtered)

    for rule in filters.rules:
        before = len(filtered)
        filtered = filtered[rule.get_mask(filtered)]
        removed = before - len(filtered)
        fraction = removed / initial_count if initial_count else 0.0

        print(
            f"{rule.column:25s} {rule.operator:>6s} "
            f"{rule.value!s:20s} -> removed "
            f"{removed:7,d} ({fraction:6.2%})"
        )

    return filtered
