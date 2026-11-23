"""
One-off, outside the repository: the line-test cells saved as .pkl (before patch 0192) rewritten as .json, the
format the line tests now read, each keeping its modification time (so the steps saved after it stay valid).

    python convert_cells_to_json.py LINE_TESTS_FOLDER            # what it would do
    python convert_cells_to_json.py LINE_TESTS_FOLDER --apply    # convert, then delete each .pkl

Run it from the repository root, in the BehaviorScreen env of the machine that wrote the cells (Raven), after
applying 0192 and while no line-test job runs. LINE_TESTS_FOLDER: e.g. line_tests (all variants).
"""
import argparse
import os
import pickle
from pathlib import Path

from BehaviorScreen.screen.line_tests.steps import write_cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    cells = sorted(args.folder.rglob("cells/*/*.pkl"))
    clashes = [p for p in cells if p.with_suffix(".json").exists()]
    print(f"{len(cells)} .pkl cells in {args.folder}; {len(clashes)} already have a .json, which is kept:")
    for path in clashes:
        print(f"  {path}")
    for path in cells:
        if path in clashes or not args.apply:
            continue
        with open(path, "rb") as f:
            record = pickle.load(f)
        target = path.with_suffix(".json")
        write_cell(target, record)
        os.utime(target, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns))
        path.unlink()
    print(f"Converted {len(cells) - len(clashes)} cells." if args.apply else "Nothing changed: --apply to convert.")


if __name__ == "__main__":
    main()
