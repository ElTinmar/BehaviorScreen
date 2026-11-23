"""
One-off, outside the repository's history: the line-test cells saved as .pkl rewritten as .json (patch 0193), each
keeping its modification time so the steps saved after it stay valid. A cell saved before arm_estimates existed
(patch 0182) gets them on the way: its two arms refitted, as Tier 1 fits them, and their Hessian SEs.

Put it in the package for the run (BehaviorScreen/convert_cells.py), and delete it afterwards. From the repository root:

    python -m BehaviorScreen.convert_cells                           # what there is to do, per variant and behavior
    python -m BehaviorScreen.convert_cells --apply                   # convert here; refits where estimates are missing
    python -m BehaviorScreen.convert_cells --submit                  # the same, one Slurm job per variant and behavior
    python -m BehaviorScreen.convert_cells --apply --model best --behavior okr

Refits take as long as the old estimates step (hours for the largest behaviors), so --submit them. A run stopped
part way resumes: each cell is written as .json before its .pkl is deleted. Run it while no line-test job runs.
"""
import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List

from BehaviorScreen.jobs import Jobs
from BehaviorScreen.screen.line_tests.screen_setup import load_screen
from BehaviorScreen.screen.line_tests.steps import write_cell
from BehaviorScreen.screen.line_tests.submit import RESOURCES
from BehaviorScreen.screen.line_tests.tier1_omnibus import arm_estimates
from BehaviorScreen.screen.paths import CONTROL_FITS, VARIANTS, data_root, line_tests_dir


def load(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def lacks_estimates(record: Dict) -> bool:
    return record["status"] == "ok" and not isinstance(record.get("arm_estimates"), dict)


def convert(variant: str, behavior: str) -> None:
    paths = sorted((line_tests_dir(variant) / "cells" / behavior).glob("*.pkl"))
    arms = None
    for path in paths:
        record = load(path)
        if lacks_estimates(record):
            if arms is None:  # the screen, once, and only for a behavior that needs refits
                arms = load_screen(data_root(None), CONTROL_FITS, variant).arms
                arms.refit_store = line_tests_dir(variant) / "control_refits"
            models, datasets = arms.models(record["line"], behavior), arms.datasets(record["line"], behavior)
            record["arm_estimates"] = {side: arm_estimates(m, m.estimate_hessian(ds)) for side, m, ds in zip(("reference", "test"), models, datasets)}
            print(f"[{variant}/{behavior}] {record['line']}: estimates added", flush=True)
        target = path.with_suffix(".json")
        write_cell(target, record)
        os.utime(target, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns))
        path.unlink()
    print(f"[{variant}/{behavior}] {len(paths)} cells converted", flush=True)


def todo(variants: List[str], behaviors: List[str]) -> Dict[tuple, tuple]:
    """(variant, behavior): (.pkl cells, of which lacking estimates)."""
    found = {}
    for variant in variants:
        for folder in sorted((line_tests_dir(variant) / "cells").glob("*")):
            paths = list(folder.glob("*.pkl"))
            if paths and (not behaviors or folder.name in behaviors):
                found[variant, folder.name] = (len(paths), sum(lacks_estimates(load(p)) for p in paths))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=VARIANTS, help="One variant (default: all).")
    parser.add_argument("--behavior", nargs="*", default=[], help="Some behaviors (default: all).")
    parser.add_argument("--apply", action="store_true", help="Convert here.")
    parser.add_argument("--submit", action="store_true", help="Convert in one Slurm job per variant and behavior.")
    args = parser.parse_args()
    found = todo([args.model] if args.model else list(VARIANTS), args.behavior)
    for (variant, behavior), (cells, lacking) in found.items():
        print(f"{variant:>12} {behavior:<32} {cells:3d} .pkl cells, {lacking:3d} lacking estimates")
    if args.submit:
        jobs = Jobs()
        for variant, behavior in found:
            jobs.submit(["python", "-m", "BehaviorScreen.convert_cells", "--apply", "--model", variant, "--behavior", behavior],
                        f"convert_{variant.replace('-', '_')}_{behavior}", RESOURCES)
        jobs.finish()
    elif args.apply:
        for variant, behavior in found:
            convert(variant, behavior)
    else:
        print("Nothing changed: --apply to convert here, --submit for Slurm jobs.")


if __name__ == "__main__":
    main()
