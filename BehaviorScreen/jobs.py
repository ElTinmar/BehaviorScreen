"""
The pipeline's jobs, submitted to Slurm or run one after another on this computer, as machines.yaml says for this host.
Run from the machine that holds the repository, BehaviorScreen env active.
"""
import argparse
import os
import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
MACHINES = Path(__file__).with_name("machines.yaml")
GPU = ["--constraint=gpu", "--gres=gpu:a100:1"]
ONE_THREAD = {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}
SLURM_JOB = """#!/bin/bash -l
module purge
module load anaconda/3/2023.03 cuda/13.0
{exports}srun conda run --no-capture-output -n BehaviorScreen {command}
"""


@dataclass(frozen=True)
class Machine:
    data_root: Path
    slurm: bool = False
    logs: Optional[Path] = None
    workers: Optional[int] = None


def this_machine() -> Machine:
    host = socket.gethostname()
    for prefix, entry in yaml.safe_load(MACHINES.read_text()).items():
        if host.startswith(prefix):
            return Machine(**{**entry, "data_root": Path(entry["data_root"])})
    raise SystemExit(f"No entry for host {host} in {MACHINES}: add one.")


def add_local_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local", action="store_true", help="Run here, one job after another, even where machines.yaml says Slurm.")


class Jobs:
    """Jobs on this machine: submit them in order (a job's dependencies first), then finish."""

    def __init__(self, local: bool = False):
        self.machine = this_machine()
        self.slurm = self.machine.slurm and not local
        self.failed: List[str] = []

    @property
    def data_root(self) -> Path:
        return self.machine.data_root

    def workers(self, cores: int) -> int:
        """Worker processes of a job given cores on Slurm."""
        return cores if self.slurm else self.machine.workers or os.cpu_count()

    def submit(self, command: List[str], name: str, resources: List[str], after: Optional[List[str]] = None, one_thread: bool = False) -> str:
        """
        command run with the BehaviorScreen env from the repository root; the job's id (its name when run here).
        resources: sbatch options, ignored here. after: jobs that must all succeed first.
        one_thread: one BLAS and XLA thread per process (for jobs with their own worker processes).
        """
        if self.slurm:
            dependency = [f"--dependency=afterok:{':'.join(after)}"] if after else []
            sbatch = ["sbatch", "--parsable", f"--job-name={name}", "-o", f"{self.machine.logs}/job.out.%j", "-e", f"{self.machine.logs}/job.err.%j",
                      "-D", str(REPOSITORY), "--ntasks=1", "--time=24:00:00", *resources, *dependency]
            exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in ONE_THREAD.items()) if one_thread else ""
            script = SLURM_JOB.format(exports=exports, command=shlex.join(command))
            job = subprocess.run(sbatch, input=script, text=True, stdout=subprocess.PIPE, check=True).stdout.strip().split(";")[0]
            print(f"{name}: job {job}", flush=True)
            return job
        if set(after or []) & set(self.failed):
            print(f"{name}: skipped, a job it needs failed", flush=True)
            self.failed.append(name)
        else:
            print(f"{name}: running", flush=True)
            if subprocess.run(command, cwd=REPOSITORY, env=os.environ | (ONE_THREAD if one_thread else {})).returncode != 0:
                print(f"{name}: failed", flush=True)
                self.failed.append(name)
        return name

    def finish(self) -> None:
        """Here: the jobs that failed or were skipped, if any (exit status 1)."""
        if not self.slurm:
            raise SystemExit(f"Failed: {', '.join(self.failed)}" if self.failed else 0)
