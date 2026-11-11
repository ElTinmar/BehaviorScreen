"""
The screen's merged tables, read by the models and the screen's tests: every <line>/<condition>/<table>.csv
of the screen folder stacked into <table>_all.csv there, with line and condition columns.

    python -m BehaviorScreen.screen.merge_screen SCREEN_FOLDER

prints the fish of each arm, and warns about lines without a row in screen_lines.csv and arms lacking a table.
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

from BehaviorScreen.screen.line_tests.line_names import warn_unnamed
from BehaviorScreen.screen.merge_csv import merge

TABLES = ("bouts", "saccades_augmented", "qc", "valid_trials")


def merge_screen(root: Path) -> pd.DataFrame:
    """Writes the four merged tables; returns the fish per arm (line x condition) of the trial table."""
    files = {name: sorted(root.glob(f"*/*/{name}.csv")) for name in TABLES}
    arms = {name: {f.parent for f in found} for name, found in files.items()}
    for name in TABLES:
        if missing := sorted(str(arm.relative_to(root)) for arm in set.union(*arms.values()) - arms[name]):
            warnings.warn(f"no {name}.csv in {missing}: these arms are not in {name}_all.csv")
    for name in TABLES:
        table = merge(root, files[name])
        table.to_csv(root / f"{name}_all.csv", index=False)
        print(f"{name}_all.csv: {len(table):,} rows from {len(files[name])} arms")
    warn_unnamed(table["line"])
    return table.groupby(["line", "condition"])["file"].nunique().unstack(fill_value=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge the screen's per-arm tables into its _all tables.")
    parser.add_argument("root", type=Path, help="Screen folder: the one holding the <line>/<condition> folders.")
    print(merge_screen(parser.parse_args().root).to_string())
