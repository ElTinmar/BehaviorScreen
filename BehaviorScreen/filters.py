"""screen.yaml rules and the table filters built from them."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


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


def exclude_qc_fish(
    dataframe: pd.DataFrame,
    quality_control_path: Path,
    file_column: str = "file",
) -> pd.DataFrame:
    """Remove recordings listed in a fish-level QC CSV."""
    result = dataframe.copy()

    if not quality_control_path.exists():
        raise FileNotFoundError(f"[qc] {quality_control_path} does not exist: no fish would be excluded")

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
