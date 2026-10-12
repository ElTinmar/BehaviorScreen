from collections import Counter
from typing import Any, Callable, Iterable, List

import joblib
from tqdm import tqdm


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
    results = joblib.Parallel(n_jobs=n_jobs, return_as="generator", **parallel_kwargs)(
        joblib.delayed(function)(item) for item in items
    )
    return list(tqdm(results, total=len(items), desc=desc, disable=not progress))


def report_failures(label: str, errors: List[str], n_total: int) -> None:
    """Print how many runs failed, by error message."""
    if not errors:
        return
    print(f"[{label}] {len(errors)}/{n_total} failed:")
    for message, count in Counter(errors).most_common(3):
        print(f"    {count} x {message}")
