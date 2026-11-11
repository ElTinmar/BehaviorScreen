"""Tier 1's pinned values for a contrast: the control fit without the contrast's own arms, so they cannot depend on its labels."""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.screen.ablation.dataset_utils import Arms, mutant_contrast, ntr_contrast
from BehaviorScreen.screen.ablation.tier1_config import PinnedArchitecture
from BehaviorScreen.screen.analysis.model_config import gamma, poisson
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.poisson_process import RateKernelFactory

ARM_SIZES = {("WT", "danieau"): 30, ("WT", "vehicle"): 12, ("WT", "ronidazole"): 12, ("L1", "vehicle"): 12, ("L1", "ronidazole"): 12, ("mut", "danieau"): 12}


class Loader:
    """BehavioralDataLoader's interface for Arms, over one dataset whose fish belong to arms."""

    def __init__(self, dataset: PointProcessDataset, arms: List[Tuple[str, str]]) -> None:
        self.dataset, self.arms = dataset, arms
        self.trials = pd.DataFrame({"file": dataset.fish_ids, "line": [a[0] for a in arms], "condition": [a[1] for a in arms]})

    def subset(self, line: Optional[str] = None, conditions: Optional[Sequence[str]] = None, exclude_lines: Sequence[str] = ()) -> "Loader":
        keep = [i for i, (l, c) in enumerate(self.arms)
                if (line is None or l == line) and (conditions is None or c in conditions) and l not in exclude_lines]
        return Loader(self.dataset.fish(keep), [self.arms[i] for i in keep])

    def prepare_dataset(self, **config) -> PointProcessDataset:
        return self.dataset


def screen(seed: int, resimulate: Sequence[Tuple[str, str]] = ()) -> Tuple[Arms, Loader]:
    """Fish simulated per arm; the arms in `resimulate` from another seed. The control fit: every vehicle and danieau fish."""
    model = gamma(poisson(RateKernelFactory.spont()))
    model.set_params(np.array([0.4, -0.02, 4.0]))
    arms = [arm for arm, n in ARM_SIZES.items() for _ in range(n)]
    empty = np.array([], float)
    template = PointProcessDataset(event_times=empty, event_trials_idx=empty.astype(int), event_fish_idx=empty.astype(int),
                                   fish_trial_mask=np.ones((len(arms), 5), bool), fish_ids=np.array([f"f{i}" for i in range(len(arms))]),
                                   duration_s=10.0, marks={"laterality": empty, "category": empty})
    first, second = (model.simulate_dataset(template, np.random.default_rng(s)) for s in (seed, seed + 1))
    take = np.array([arms[f] in resimulate for f in second.event_fish_idx])
    keep = np.array([arms[f] not in resimulate for f in first.event_fish_idx])
    join = lambda a, b: np.concatenate([a[keep], b[take]])
    data = PointProcessDataset(event_times=join(first.event_times, second.event_times), event_trials_idx=join(first.event_trials_idx, second.event_trials_idx),
                               event_fish_idx=join(first.event_fish_idx, second.event_fish_idx), fish_trial_mask=template.fish_trial_mask,
                               fish_ids=template.fish_ids, duration_s=10.0, marks={})
    loader = Loader(data, arms)
    model.fit(loader.subset(conditions=["vehicle", "danieau"]).dataset)
    contrasts = [ntr_contrast("WT"), ntr_contrast("L1"), mutant_contrast("mut", ("WT", "danieau"))]
    return Arms({"bouts": loader}, {"spont": "bouts"}, {"spont": {}}, {"spont": PinnedArchitecture(model)}, contrasts), loader


class TestLeaveContrastOutPinning:
    @pytest.mark.parametrize("contrast, left_out", [
        ("WT", {("WT", "vehicle")}), ("L1", {("L1", "vehicle")}), ("mut", {("WT", "danieau"), ("mut", "danieau")}),
    ])
    def test_pins_to_the_control_fit_without_the_contrasts_control_arms(self, contrast: str, left_out: set) -> None:
        arms, loader = screen(0)
        control = loader.subset(conditions=["vehicle", "danieau"])
        rest = control.dataset.fish([i for i, arm in enumerate(control.arms) if arm not in left_out])
        direct = arms.process_factories["spont"].fitted
        direct.fit(rest)
        pinned = arms.process_factory(contrast, "spont").fitted
        assert pinned.nll(pinned.params_, rest) == pytest.approx(direct.nll(direct.params_, rest), abs=0.05)

    @pytest.mark.parametrize("contrast", ["WT", "L1", "mut"])
    def test_pinned_values_ignore_the_contrasts_own_fish(self, contrast: str) -> None:
        """New data in the contrast's arms (any labelling of them) leaves its pinned values unchanged, to the optimizer's tolerance."""
        before = screen(0)[0]
        after = screen(0, resimulate=sorted(before.contrast_arms(contrast)))[0]
        fixed_before, fixed_after = before.process_factory(contrast, "spont").fixed, after.process_factory(contrast, "spont").fixed
        assert fixed_after == pytest.approx(fixed_before, rel=1e-3)


def test_control_fit_leaves_out_mutants_and_treated_fish() -> None:
    arms = [("WT", "danieau")] * 3 + [("lak", "danieau")] * 2 + [("L1", "vehicle")] * 2 + [("L1", "ronidazole")] * 2 + [("WT", "ronidazole")]
    empty = np.array([], float)
    data = PointProcessDataset(event_times=empty, event_trials_idx=empty.astype(int), event_fish_idx=empty.astype(int),
                               fish_trial_mask=np.ones((len(arms), 2), bool), fish_ids=np.array([f"f{i}" for i in range(len(arms))]), marks={})
    control, control_arms = Arms({"bouts": Loader(data, arms)}, {"spont": "bouts"}, {"spont": {}}, {}, []).control("spont")
    assert control_arms == [("WT", "danieau")] * 3 + [("L1", "vehicle")] * 2
    assert list(control.fish_ids) == ["f0", "f1", "f2", "f5", "f6"]
