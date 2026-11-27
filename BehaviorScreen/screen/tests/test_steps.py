"""A step after the cells is reused while it is newer than the cells and the steps it reads."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from BehaviorScreen.screen.line_tests import steps
from BehaviorScreen.screen.line_tests.tier1_omnibus import cells_current


def test_steps_are_reused_until_their_inputs_change(tmp_path: Path) -> None:
    (tmp_path / "cells" / "okr").mkdir(parents=True)
    cell = tmp_path / "cells" / "okr" / "atf5b.json"
    steps.write_cell(cell, {"status": "ok", "p_value": np.float64(0.5), "n_perm_exceeding": np.int64(3)})
    assert steps.read_cell(cell) == {"status": "ok", "p_value": 0.5, "n_perm_exceeding": 3}
    os.utime(cell, (1, 1))
    computed = []
    run = lambda step: steps.run_step(tmp_path, "okr", step, lambda: computed.append(step) or pd.DataFrame({"line": ["1010"]}))
    for step in steps.STEPS:
        assert not steps.finished(tmp_path, "okr", merged_only=False)
        run(step)
    assert steps.finished(tmp_path, "okr", merged_only=False) and not list((tmp_path / "okr").glob("*.tmp"))
    assert steps.finished(tmp_path, "okr") and run("tier1_parameters")["line"].tolist() == ["1010"]  # reused, a line name kept as text
    assert computed == list(steps.STEPS)
    assert steps.running(tmp_path, "okr")[0] == list(steps.STEPS)[-1]  # the last step started
    for step in steps.STEPS:
        os.utime(steps.step_path(tmp_path, "okr", step), (2, 2))
    os.utime(steps.step_path(tmp_path, "okr", "tier1_parameters"), (3, 3))  # estimates recomputed after the bootstrap
    assert not steps.saved(tmp_path, "okr", "bootstrap_parameter_deltas") and steps.saved(tmp_path, "okr", "tier2_results")
    os.utime(cell, (4, 4))  # a cell recomputed after every step
    assert not any(steps.saved(tmp_path, "okr", step) for step in steps.STEPS)


def test_cells_run_with_other_pins_are_not_current(tmp_path: Path) -> None:
    """A behavior whose pinned values changed is rerun, even with every step saved after its cells."""
    architecture = SimpleNamespace(template=SimpleNamespace(name="m"), fixed={"mu": 5.0})
    assert not cells_current(tmp_path, architecture)
    (tmp_path / "key.json").write_text(json.dumps({"model": "m", "fixed": {"mu": 5.0, "B": 0.02}, "n_perm": 999}))
    assert not cells_current(tmp_path, architecture)
    (tmp_path / "key.json").write_text(json.dumps({"model": "m", "fixed": {"mu": 5.0}, "n_perm": 999}))
    assert cells_current(tmp_path, architecture)
