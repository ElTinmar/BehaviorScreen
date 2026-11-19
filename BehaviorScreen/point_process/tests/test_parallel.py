"""parallel_map's worker count."""
import subprocess
import sys

CODE = """
import os, joblib
from BehaviorScreen.point_process.parallel import limit_workers, parallel_map
limit_workers(1)
print(joblib.cpu_count(), len(set(parallel_map(lambda _: os.getpid(), range(4), progress=False))))
"""


def test_limit_workers_caps_the_workers_of_this_process_and_its_children() -> None:
    assert subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, check=True).stdout.split() == ["1", "1"]
