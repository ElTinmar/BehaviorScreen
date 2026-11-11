import fcntl
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import IO, Any, Callable, Iterable, List

import joblib
from tqdm import tqdm

_core_lock: List[IO] = []  # held for the worker's life: its core stays its own


def pin_to_free_core() -> None:
    """
    Pins this process to the first of its allowed cores that no other process
    holds (a lock file per core, released when the process ends). XLA runs its
    thread pool on every allowed core and has no flag to limit it, so unpinned
    workers fight over all the cores.
    """
    if _core_lock or not hasattr(os, "sched_setaffinity"):
        return
    locks = Path(tempfile.gettempdir()) / f"behaviorscreen-cores-{os.getuid()}"
    locks.mkdir(exist_ok=True)
    for core in sorted(os.sched_getaffinity(0)):
        handle = open(locks / str(core), "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        _core_lock.append(handle)
        os.sched_setaffinity(0, {core})
        return


def _pinned(function: Callable[[Any], Any], item: Any) -> Any:
    pin_to_free_core()
    return function(item)


def parallel_map(
    function: Callable[[Any], Any],
    items: Iterable[Any],
    n_jobs: int = -1,
    desc: str = "",
    progress: bool = True,
    **parallel_kwargs: Any,
) -> List[Any]:
    """joblib.Parallel over items with a progress bar; results in input order."""
    items = list(items)
    # One BLAS/OpenMP thread per worker: the workers already fill the cores.
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        results = joblib.Parallel(n_jobs=n_jobs, return_as="generator", **parallel_kwargs)(
            joblib.delayed(_pinned)(function, item) for item in items
        )
        return list(tqdm(results, total=len(items), desc=desc, disable=not progress))


def report_failures(label: str, errors: List[str], n_total: int) -> None:
    """Print how many runs failed, by error message."""
    if not errors:
        return
    print(f"[{label}] {len(errors)}/{n_total} failed:")
    for message, count in Counter(errors).most_common(3):
        print(f"    {count} x {message}")
