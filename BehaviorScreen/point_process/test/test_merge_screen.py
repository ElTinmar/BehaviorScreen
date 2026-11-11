"""The merged tables of a small screen folder."""
from pathlib import Path

import pandas as pd
import pytest

from BehaviorScreen.merge_screen import TABLES, merge_screen


def test_tables_of_every_arm_are_stacked_with_line_and_condition(tmp_path: Path) -> None:
    for line, condition, fish in (("WT", "danieau", ["a", "b"]), ("cort", "vehicle", ["c"]), ("cort", "ronidazole", ["d", "e"])):
        folder = tmp_path / line / condition
        folder.mkdir(parents=True)
        for name in TABLES:
            pd.DataFrame({"file": fish, "value": 1}).to_csv(folder / f"{name}.csv", index=False)
    (tmp_path / "cort" / "ronidazole" / "qc.csv").unlink()
    pd.DataFrame({"file": ["old"]}).to_csv(tmp_path / "qc.csv", index=False)  # a table of the screen folder itself is not an arm's

    with pytest.warns(UserWarning, match="no qc.csv in .*cort/ronidazole"):
        fish = merge_screen(tmp_path)

    assert fish.loc["cort"].to_dict() == {"danieau": 0, "ronidazole": 2, "vehicle": 1}
    bouts, qc = pd.read_csv(tmp_path / "bouts_all.csv"), pd.read_csv(tmp_path / "qc_all.csv")
    assert list(bouts.columns) == ["line", "condition", "file", "value"]
    assert sorted(zip(bouts["line"], bouts["condition"], bouts["file"])) == [
        ("WT", "danieau", "a"), ("WT", "danieau", "b"), ("cort", "ronidazole", "d"), ("cort", "ronidazole", "e"), ("cort", "vehicle", "c")]
    assert set(qc["file"]) == {"a", "b", "c"}


def test_tables_with_different_columns_are_refused(tmp_path: Path) -> None:
    for line, columns in (("WT", ["file", "value"]), ("cort", ["file", "other"])):
        (tmp_path / line / "vehicle").mkdir(parents=True)
        for name in TABLES:
            pd.DataFrame(columns=columns).to_csv(tmp_path / line / "vehicle" / f"{name}.csv", index=False)
    with pytest.raises(ValueError, match="headers do not match"):
        merge_screen(tmp_path)
