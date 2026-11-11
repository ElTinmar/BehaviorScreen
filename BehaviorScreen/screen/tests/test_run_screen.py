"""run_screen end to end on a simulated screen: a behavior without a hit and one with a hit both give their tables and maps."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from BehaviorScreen.screen.ablation import run_screen
from BehaviorScreen.screen.ablation.tier1_config import Fresh
from BehaviorScreen.screen.analysis.model_config import flat, poisson

from .test_tier1_pinning import screen


@pytest.mark.parametrize("kept", [1.0, 0.3], ids=["no hit", "hit"])
def test_behavior_runs_and_combines(kept: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """kept: share of the events of L1's drug arm that are kept (thinning lowers its rate)."""
    arms, loader = screen(0)
    data, rng = loader.dataset, np.random.default_rng(0)
    drug = np.array([loader.arms[f] == ("L1", "ronidazole") for f in data.event_fish_idx])
    keep = ~drug | (rng.random(len(drug)) < kept)
    loader.dataset = data._with_events(keep, data.event_fish_idx[keep])
    monkeypatch.setattr(run_screen, "negative_control", lambda *args: None)
    tier2 = run_screen.run_tier2_screen
    monkeypatch.setattr(run_screen, "run_tier2_screen", lambda *args, **kwargs: tier2(*args, **{**kwargs, "n_boot": 20}))
    monkeypatch.setattr(run_screen, "bootstrap_parameter_deltas", lambda *args, n_boot: pd.DataFrame())

    run_screen.run_behavior("spont", arms, {"spont": Fresh(poisson(flat()))}, tmp_path, n_perm=99, n_sim=30)
    run_screen.combine(["spont"], arms, ["spont"], tmp_path)

    tier1 = pd.read_csv(tmp_path / "tier1_results.csv")
    assert set(tier1["status"]) == {"ok"} and len(tier1) == 3
    assert bool(tier1.set_index("line").loc["L1", "significant"]) == (kept < 1)
    assert tier1.set_index("line")["p_value_plate"].notna().to_dict() == {"WT": True, "L1": True, "mut": False}
    assert (tmp_path / "tier1_map.png").exists() and (tmp_path / "spont" / "gain_structure.json").exists()
