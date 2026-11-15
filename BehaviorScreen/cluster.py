"""Slurm jobs on the cluster: what every job shares, and one job per call (run from the login node, BehaviorScreen env active)."""
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

REPOSITORY = Path(__file__).resolve().parents[1]
LOGS = Path("/u/mapri/logs")
DATA_ROOT = Path("/ptmp/mapri")
GPU = ["--constraint=gpu", "--gres=gpu:a100:1"]
JOB = """#!/bin/bash -l
module purge
module load anaconda/3/2023.03 cuda/13.0
{threads}srun conda run --no-capture-output -n BehaviorScreen {command}
"""
ONE_THREAD = "export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"


def submit(command: List[str], name: str, resources: List[str], after: Optional[List[str]] = None, one_thread: bool = False) -> str:
    """
    command run in the BehaviorScreen env from the repository root, as one job; its job id.
    after: jobs that must all succeed first. one_thread: one BLAS thread per process (for jobs with their own worker processes).
    """
    dependency = [f"--dependency=afterok:{':'.join(after)}"] if after else []
    sbatch = ["sbatch", "--parsable", f"--job-name={name}", "-o", f"{LOGS}/job.out.%j", "-e", f"{LOGS}/job.err.%j",
              "-D", str(REPOSITORY), "--ntasks=1", "--time=24:00:00", *resources, *dependency]
    script = JOB.format(threads=ONE_THREAD if one_thread else "", command=shlex.join(command))
    job = subprocess.run(sbatch, input=script, text=True, stdout=subprocess.PIPE, check=True).stdout.strip().split(";")[0]
    print(f"{name}: job {job}", flush=True)
    return job
