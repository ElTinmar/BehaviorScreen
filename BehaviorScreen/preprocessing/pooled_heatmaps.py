"""
Heatmaps of one condition pooled across lines (default: the vehicle arms), for the slideshow's page of them:
    python -m BehaviorScreen.preprocessing.pooled_heatmaps [SCREEN_FOLDER] [--condition vehicle]
Takes the condition's rows of the screen's merged tables (<table>_all.csv, from python -m BehaviorScreen merge) and
runs the heatmaps step on them, into SCREEN_FOLDER/pooled/<condition>/plots/ (SCREEN_FOLDER: by default, this
machine's data_root in machines.yaml).
"""
import argparse
import tempfile
from pathlib import Path

import pandas as pd

from BehaviorScreen.jobs import this_machine
from BehaviorScreen.preprocessing.__main__ import build_parser, heatmaps

TABLES = ("bouts", "saccades_augmented", "valid_trials", "qc")  # merge_screen's


def pooled_heatmaps(screen: Path, condition: str) -> Path:
    """The pooled heatmaps of `condition`; their folder."""
    plots = screen / "pooled" / condition / "plots"
    with tempfile.TemporaryDirectory() as selected:
        for table in TABLES:
            merged = screen / f"{table}_all.csv"
            if not merged.exists():
                continue
            if stale := [str(t.parent) for t in screen.glob(f"*/{condition}/{table}.csv") if t.stat().st_mtime > merged.stat().st_mtime]:
                print(f"warning: {merged.name} is older than the {table}.csv of {', '.join(sorted(stale))}: rerun merge")
            rows = pd.concat(chunk[chunk["condition"] == condition] for chunk in pd.read_csv(merged, chunksize=1_000_000))
            rows.to_csv(Path(selected) / f"{table}.csv", index=False)
            print(f"{table}: {len(rows):,} rows of {condition}, {rows['line'].nunique()} lines")
        heatmaps(build_parser().parse_args([selected, "--plots", str(plots)]))
    return plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("screen", type=Path, nargs="?", help="Screen folder: the one holding the <line>/<condition> folders.")
    parser.add_argument("--condition", default="vehicle")
    args = parser.parse_args()
    print(f"-> {pooled_heatmaps(args.screen or this_machine().data_root, args.condition)}")


if __name__ == "__main__":
    main()
