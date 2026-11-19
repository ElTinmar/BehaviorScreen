"""A step after the cells is reused while it is newer than the cells and the steps it reads."""
import os
import pickle
from pathlib import Path

import pandas as pd

from BehaviorScreen.screen.line_tests import steps


def test_steps_are_reused_until_their_inputs_change(tmp_path: Path) -> None:
    (tmp_path / "cells" / "okr").mkdir(parents=True)
    cell = tmp_path / "cells" / "okr" / "atf5b.pkl"
    cell.write_bytes(pickle.dumps({"status": "ok"}))
    os.utime(cell, (1, 1))
    computed = []
    run = lambda step: steps.run_step(tmp_path, "okr", step, lambda: computed.append(step) or pd.DataFrame({"line": ["1010"]}))
    for step in steps.STEPS:
        run(step)
    assert steps.finished(tmp_path, "okr") and run("tier1_parameters")["line"].tolist() == ["1010"]  # reused, a line name kept as text
    assert computed == list(steps.STEPS)
    for step in steps.STEPS:
        os.utime(steps.step_path(tmp_path, "okr", step), (2, 2))
    os.utime(steps.step_path(tmp_path, "okr", "tier1_parameters"), (3, 3))  # estimates recomputed after the bootstrap
    assert not steps.saved(tmp_path, "okr", "bootstrap_parameter_deltas") and steps.saved(tmp_path, "okr", "tier2_results")
    os.utime(cell, (4, 4))  # a cell recomputed after every step
    assert not any(steps.saved(tmp_path, "okr", step) for step in steps.STEPS)
