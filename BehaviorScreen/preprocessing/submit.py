"""
The preprocessing pipeline on the cluster, one Slurm job per <line>/<condition> folder (the folders with results/ in them):
    python -m BehaviorScreen.preprocessing.submit PATH [pipeline options, e.g. --steps qc heatmaps]
PATH: one such folder, a line folder or a screen root.
A GPU when the steps include pose, eye_pose or bouts (all steps by default); 12 GB of memory for qc and heatmaps alone.
"""
import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List

from BehaviorScreen.preprocessing.__main__ import build_parser

REPOSITORY = Path(__file__).resolve().parents[2]
SLURM = ["-o", "/u/mapri/logs/job.out.%j", "-e", "/u/mapri/logs/job.err.%j", "-D", str(REPOSITORY),
         "--ntasks=1", "--cpus-per-task=12", "--time=24:00:00"]
JOB = """#!/bin/bash -l
module purge
module load anaconda/3/2023.03 cuda/13.0
srun conda run --no-capture-output -n BehaviorScreen python -m BehaviorScreen.preprocessing {}
"""


def resources(steps: List[str]) -> List[str]:
    if {"pose", "eye_pose", "bouts"} & set(steps):
        return ["--constraint=gpu", "--gres=gpu:a100:1", "--mem=48000"]
    return ["--mem=12000"] if set(steps) <= {"qc", "heatmaps"} else ["--mem=48000"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    path, options = parser.parse_known_args()
    path = path.path.resolve()
    folders = [d for d in (path, *sorted(path.glob("*")), *sorted(path.glob("*/*"))) if (d / "results").is_dir()]
    if not folders:
        raise SystemExit(f"no folder with results/ in it at or below {path}")
    steps = build_parser().parse_args([str(path), *options]).steps
    for folder in folders:
        print(f"{folder}: ", end="", flush=True)
        subprocess.run(["sbatch", *SLURM, *resources(steps), f"--job-name=preprocessing_{folder.parent.name}/{folder.name}"],
                       input=JOB.format(shlex.join([str(folder), *options])), text=True, check=True)


if __name__ == "__main__":
    main()
