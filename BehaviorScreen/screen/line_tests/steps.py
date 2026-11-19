"""
The steps of a line-test run after a behavior's cells, each saved as soon as it is done
(<results folder>/<behavior>/<step>.csv): a resubmitted behavior reuses a step saved after its cells (and after the
steps it uses), and recomputes it otherwise, as it reuses the cells themselves. The step being computed, and since
when, is in <behavior>/running.txt.
"""
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from BehaviorScreen.screen.paths import LINE_TEST_TABLES

STEPS = {  # in run order: step, short name
    "tier1_parameters": "estimates",
    "tier2_results": "tier2",
    "arm_surface_grids": "grids",
    "bootstrap_parameter_deltas": "bootstrap",
}
USES: Dict[str, List[str]] = {"bootstrap_parameter_deltas": ["tier1_parameters"]}  # steps whose results another step reads


def cell_files(output_dir: Path, behavior: str) -> List[Path]:
    return list((output_dir / "cells" / behavior).glob("*.pkl"))


def step_path(output_dir: Path, behavior: str, step: str) -> Path:
    return output_dir / behavior / f"{step}.csv"


def saved(output_dir: Path, behavior: str, step: str) -> bool:
    """Whether the step's table was saved after the behavior's cells and the steps it uses."""
    path = step_path(output_dir, behavior, step)
    inputs = cell_files(output_dir, behavior) + [step_path(output_dir, behavior, s) for s in USES.get(step, [])]
    return path.exists() and all(p.exists() and path.stat().st_mtime >= p.stat().st_mtime for p in inputs)


def finished(output_dir: Path, behavior: str, merged_only: bool = True) -> bool:
    """Whether the steps are saved: those whose tables are merged (the grids are figures, not needed for the merge), or all."""
    return all(saved(output_dir, behavior, step) for step in STEPS if step in LINE_TEST_TABLES or not merged_only)


def running(output_dir: Path, behavior: str) -> Optional[Tuple[str, datetime]]:
    """The step last started on the behavior and when, if one was."""
    path = output_dir / behavior / "running.txt"
    if not path.exists():
        return None
    step, started = path.read_text().split()
    return step, datetime.fromisoformat(started)


def read_table(path: Path) -> pd.DataFrame:
    """A saved table; empty if it had no columns (a behavior without hits has no bootstrap rows)."""
    try:
        return pd.read_csv(path, dtype={"line": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_step(output_dir: Path, behavior: str, step: str, compute: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """The step's table: reused if saved (see saved), else computed and saved."""
    path = step_path(output_dir, behavior, step)
    if saved(output_dir, behavior, step):
        print(f"[{behavior}] {step}: saved earlier, reused")
        return read_table(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "running.txt").write_text(f"{step} {datetime.now().isoformat(timespec='seconds')}")
    table = compute()
    table.to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)  # whole or absent, even when the job is stopped while writing
    return table
