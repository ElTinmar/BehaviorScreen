"""
Where the line tests stand, from their results folders: per model variant and behavior, the cells saved, those that
failed, the steps after the cells saved so far (steps.STEPS) and the one last started (for how long), whether the
merge has it and what is left.
"incomplete" is a behavior with work saved but not finished: its job is running or queued, or it stopped and needs
resubmitting.
    python -m BehaviorScreen test-lines --status [--model best|best-poisson|base]
Without --model, also when the model-free screen (count_readouts) last ran.
A cell that failed (status unexpected_error) is saved like the others and not tried again: delete its file, listed with
--failed, before resubmitting the behavior to retry it.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from BehaviorScreen.screen.control_models.submit import EXPERIMENTS
from BehaviorScreen.screen.line_tests.steps import STEPS, cells_in, finished, read_cell, running, saved
from BehaviorScreen.screen.paths import COUNT_RESULTS, LINE_TEST_TABLES, VARIANTS, line_tests_dir


def progress(steps: List[str], current: Optional[Tuple[str, datetime]]) -> str:
    """'estimates+tier2, grids for 1h12m': the steps saved, then the one last started and for how long."""
    if current is None or STEPS[current[0]] in steps:
        return "+".join(steps) or "-"
    minutes = int((datetime.now() - current[1]).total_seconds() // 60)
    return f"{'+'.join(steps) + ', ' if steps else ''}{STEPS[current[0]]} for {minutes // 60}h{minutes % 60:02d}m"


def contrasts() -> int:
    """Contrasts each behavior tests: those of the best model's merge (0 before one), the same for every behavior and variant."""
    path = line_tests_dir("best") / "tier1_results.csv"
    return pd.read_csv(path, usecols=["line"], dtype=str)["line"].nunique() if path.exists() else 0


def cells(store: Path) -> List[dict]:
    return [{**read_cell(path), "file": path} for path in cells_in(store)]


def variant_status(variant: str, behaviors: List[str]) -> pd.DataFrame:
    root = line_tests_dir(variant)
    merged_path = root / "tier1_results.csv"
    merged = set(pd.read_csv(merged_path, usecols=["behavior"])["behavior"]) if merged_path.exists() else set()
    saved_cells = {b: cells(root / "cells" / b) for b in behaviors}
    expected = contrasts() or max((len(c) for c in saved_cells.values()), default=0)
    rows = []
    for behavior in behaviors:
        tables = [root / behavior / f"{t}.csv" for t in LINE_TEST_TABLES]
        done = finished(root, behavior)
        steps = [short for step, short in STEPS.items() if saved(root, behavior, step)]
        failed = [c for c in saved_cells[behavior] if str(c.get("status", "")).startswith("unexpected_error")]
        in_merge = done and behavior in merged and merged_path.stat().st_mtime >= max(t.stat().st_mtime for t in tables)
        next_step = "nothing" if in_merge else "merge" if done else "incomplete" if saved_cells[behavior] else "submit"
        rows.append({"behavior": behavior, "cells": f"{len(saved_cells[behavior])}/{expected}", "failed": len(failed),
                     "after cells": "done" if done else progress(steps, running(root, behavior)), "merged": "yes" if in_merge else "-", "next": next_step})
    return pd.DataFrame(rows)


def main(variant: Optional[str] = None, behaviors: Optional[List[str]] = None, failed: bool = False) -> None:
    for name in [variant] if variant else [v for v in VARIANTS if line_tests_dir(v).exists()]:
        table = variant_status(name, behaviors or EXPERIMENTS)
        print(f"\n{name} ({line_tests_dir(name)}/)\n{table.to_string(index=False)}")
        if failed:
            for behavior in table["behavior"]:
                for c in cells(line_tests_dir(name) / "cells" / behavior):
                    if str(c.get("status", "")).startswith("unexpected_error"):
                        print(f"  {c['file']}: {c['status'][:150]}")
    if not variant:
        results = COUNT_RESULTS / "tier1_results.csv"
        when = f"last run {datetime.fromtimestamp(results.stat().st_mtime):%Y-%m-%d %H:%M}" if results.exists() else "not run (test-lines --model-free)"
        print(f"\nmodel-free ({COUNT_RESULTS}/): {when}")
