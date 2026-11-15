"""
Preprocess recordings, one job per <line>/<condition> folder, on Slurm or here (BehaviorScreen/machines.yaml):
    python -m BehaviorScreen preprocess PATH [--local] [pipeline options, e.g. --steps qc heatmaps]
PATH: a <line>/<condition> folder (one with results/ in it), a line folder or a screen root.
On Slurm: a GPU when the steps include pose, eye_pose or bouts (all steps by default); 12 GB of memory for qc and heatmaps alone.
"""
import argparse
from pathlib import Path
from typing import List

from BehaviorScreen.jobs import GPU, Jobs, add_local_argument
from BehaviorScreen.preprocessing.__main__ import build_parser


def resources(steps: List[str]) -> List[str]:
    if {"pose", "eye_pose", "bouts"} & set(steps):
        return [*GPU, "--cpus-per-task=12", "--mem=48000"]
    return ["--cpus-per-task=12", "--mem=12000" if set(steps) <= {"qc", "heatmaps"} else "--mem=48000"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    add_local_argument(parser)
    args, options = parser.parse_known_args()
    path = args.path.resolve()
    folders = [d for d in (path, *sorted(path.glob("*")), *sorted(path.glob("*/*"))) if (d / "results").is_dir()]
    if not folders:
        raise SystemExit(f"no folder with results/ in it at or below {path}")
    steps = build_parser().parse_args([str(path), *options]).steps
    jobs = Jobs(args.local)
    for folder in folders:
        jobs.submit(["python", "-m", "BehaviorScreen.preprocessing", str(folder), *options],
                    f"preprocessing_{folder.parent.name}/{folder.name}", resources(steps))
    jobs.finish()


if __name__ == "__main__":
    main()
