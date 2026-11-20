"""Cells whose observed deviance few permutations reached, not significant yet, are permuted further and saved."""
from pathlib import Path

import pandas as pd
import pytest

from BehaviorScreen.screen.line_tests import tier1_omnibus
from BehaviorScreen.screen.line_tests.fdr import add_fdr_per_behavior
from BehaviorScreen.screen.line_tests.tier1_config import Fresh
from BehaviorScreen.screen.line_tests.tier1_omnibus import extend_permutations, run_tier1_screen
from BehaviorScreen.screen.control_models.model_config import flat, poisson

from .test_tier1_pinning import screen


def test_cells_are_extended_and_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arms, _ = screen(0)
    cells = run_tier1_screen(arms, ["spont"], {"spont": Fresh(poisson(flat()))}, n_perm=20, n_sim=0, show_progress=False, store=tmp_path)
    monkeypatch.setattr(tier1_omnibus, "EXTEND_BELOW", 100)  # every cell counts as near the floor
    extended = extend_permutations(add_fdr_per_behavior(cells).assign(significant=False), arms, 40, store=tmp_path)
    assert ((extended["n_perm_success"] + extended["n_perm_failed"]) == 40).all()
    assert (extended["p_value"] == (extended["n_perm_exceeding"] + 1) / (extended["n_perm_success"] + 1)).all()
    saved = pd.DataFrame([pd.read_pickle(p) for p in (tmp_path / "spont").glob("*.pkl")]).set_index("line")
    assert saved["n_perm_success"].to_dict() == extended.set_index("line")["n_perm_success"].to_dict()
    again = run_tier1_screen(arms, ["spont"], {"spont": Fresh(poisson(flat()))}, n_perm=20, n_sim=0, show_progress=False, store=tmp_path)
    assert again.set_index("line")["p_value"].to_dict() == extended.set_index("line")["p_value"].to_dict()  # reused as extended
