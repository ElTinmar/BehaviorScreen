"""screen.yaml rules and the filters built from them: what they keep, and what they refuse."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.filters import apply_event_filters, apply_table_filters, exclude_qc_fish, parse_rules

TABLE = pd.DataFrame({
    "file": ["a", "a", "b", "c", "d"],
    "dpf": [7, 7, 7, 6, 7],
    "distance": [1.0, 12.0, np.nan, 3.0, 5.0],
    "category": [0, 1, 2, 0, 3],
})


@pytest.mark.parametrize("rules, kept", [
    ({"distance": {"<=": 5.0}}, [0, 3, 4]),  # a missing value fails every rule
    ({"distance": {">": 1.0, "<": 12.0}}, [3, 4]),
    ({"dpf": {"==": 7}, "category": {"!=": 1}}, [0, 2, 4]),
    ({"category": {"in": [0, 3]}}, [0, 3, 4]),
    ({"category": {"not_in": [0, 3]}}, [1, 2]),
    ({"distance": {">=": 12.0}}, [1]),
    (None, [0, 1, 2, 3, 4]),
])
def test_rules_keep_the_rows_that_pass_all_of_them(rules: dict, kept: list) -> None:
    assert list(TABLE.index[parse_rules(rules).get_mask(TABLE)]) == kept


def test_unknown_operator_or_column_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown operator"):
        parse_rules({"dpf": {"=>": 7}})
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_rules({"dpf": 7})
    with pytest.raises(ValueError, match="missing"):
        parse_rules({"age": {"==": 7}}).get_mask(TABLE)
    with pytest.raises(ValueError, match="'age'"):
        apply_table_filters(TABLE, {"recording": {"age": {"==": 7}}}, ("recording",))


def test_table_filters_apply_the_named_sections_only() -> None:
    config = {"recording": {"dpf": {"==": 7}}, "event": {"distance": {"<=": 5.0}}, "other": {"category": {"==": 9}}}
    assert list(apply_table_filters(TABLE, config, ("recording",)).index) == [0, 1, 2, 4]
    assert list(apply_table_filters(TABLE, config, ("recording", "event")).index) == [0, 4]
    assert list(apply_table_filters(TABLE, {"recording": None}, ("recording", "event")).index) == [0, 1, 2, 3, 4]


def test_event_filters_are_the_common_ones_and_those_of_the_event_type() -> None:
    config = {"common_filters": {"dpf": {"==": 7}}, "bout_filters": {"distance": {"<=": 5.0}}, "saccade_filters": {"category": {"in": [1, 2]}}}
    assert list(apply_event_filters(TABLE, config, "bout").index) == [0, 4]
    assert list(apply_event_filters(TABLE, config, "saccade").index) == [1, 2]
    with pytest.raises(ValueError, match="bout"):
        apply_event_filters(TABLE, config, "blink")


def test_qc_fish_are_removed_and_a_missing_table_is_an_error(tmp_path: Path) -> None:
    qc = tmp_path / "qc.csv"
    pd.DataFrame({"file": ["a", "d", "z"], "reason": ["not_moving", "tracking", "tracking"]}).to_csv(qc, index=False)
    assert list(exclude_qc_fish(TABLE, qc)["file"]) == ["b", "c"]
    with pytest.raises(FileNotFoundError):
        exclude_qc_fish(TABLE, tmp_path / "absent.csv")
    pd.DataFrame({"fish": ["a"]}).to_csv(qc, index=False)
    with pytest.raises(ValueError, match="'file'"):
        exclude_qc_fish(TABLE, qc)
