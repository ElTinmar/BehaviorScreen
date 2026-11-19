"""Jobs submitted to Slurm: the sbatch options of each, without Slurm."""
import subprocess
from typing import List

import pytest

from BehaviorScreen import jobs


def test_resubmitted_copies_each_wait_for_the_one_before(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[List[str]] = []
    monkeypatch.setattr(jobs, "this_machine", lambda: jobs.Machine(data_root=jobs.Path("/data"), slurm=True, logs=jobs.Path("/logs")))
    monkeypatch.setattr(subprocess, "run", lambda sbatch, **_: calls.append(sbatch) or subprocess.CompletedProcess(sbatch, 0, f"{len(calls)}\n"))
    assert jobs.Jobs().submit(["true"], "t", ["--mem=1"], after=["7"], resubmit=2) == "3"
    dependencies = [[o for o in call if o.startswith("--dependency")] for call in calls]
    assert dependencies == [["--dependency=afterok:7"], ["--dependency=afterany:1"], ["--dependency=afterany:2"]]
