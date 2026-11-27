"""
Preprocess recordings, one job per <line>/<condition> folder, on Slurm or here (BehaviorScreen/machines.yaml):
    python -m BehaviorScreen preprocess PATH [--local] [pipeline options, e.g. --steps qc heatmaps]
    python -m BehaviorScreen preprocess PATH --status   (per folder and step, recordings: done; ready, every input there; waiting, some input missing)
PATH: a <line>/<condition> folder (one with data/ or results/ in it), a line folder or a screen root.
On Slurm: a GPU when the steps include pose, eye_pose or bouts (all steps by default); 12 GB of memory for qc and heatmaps alone.
"""
import argparse
from pathlib import Path
from typing import List

from BehaviorScreen.jobs import GPU, Jobs, add_local_argument
from BehaviorScreen.preprocessing.__main__ import build_parser
from BehaviorScreen.preprocessing.status import folders_status


def resources(steps: List[str]) -> List[str]:
    if {"pose", "eye_pose", "bouts"} & set(steps):
        return [*GPU, "--cpus-per-task=12", "--mem=48000"]
    return ["--cpus-per-task=12", "--mem=12000" if set(steps) <= {"qc", "heatmaps"} else "--mem=48000"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    add_local_argument(parser)
    parser.add_argument("--status", action="store_true", help="What each folder has done and has left, from its files; submits nothing.")
    args, options = parser.parse_known_args()
    path = args.path.resolve()
    if not path.is_dir():
        raise SystemExit(f"no folder {path}")
    folders = [d for d in (path, *sorted(path.glob("*")), *sorted(path.glob("*/*"))) if (d / "data").is_dir() or (d / "results").is_dir()]
    if not folders:
        raise SystemExit(f"no folder with data/ or results/ in it at or below {path}")
    pipeline = build_parser().parse_args([str(path), *options])
    if args.status:
        print(folders_status(folders, vars(pipeline), path).to_string())
        return
    steps = pipeline.steps
    jobs = Jobs(args.local)
    for folder in folders:
        jobs.submit(["python", "-m", "BehaviorScreen.preprocessing", str(folder), *options],
                    f"preprocessing_{folder.parent.name}/{folder.name}", resources(steps))
    jobs.finish()


if __name__ == "__main__":
    main()
