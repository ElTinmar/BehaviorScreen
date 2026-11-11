"""NTR contrasts compare siblings within a day: arms keep the days recorded in both, and fish are permuted within their day."""
from typing import Dict, Tuple

import numpy as np

from BehaviorScreen.screen.ablation.dataset_utils import Arms, mutant_contrast, ntr_contrast, recording_days
from BehaviorScreen.screen.ablation.tier1_config import PinnedArchitecture
from BehaviorScreen.screen.ablation.tier1_omnibus import permute_within, tier1_permutation_test
from BehaviorScreen.screen.analysis.model_config import gamma, poisson
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.poisson_process import RateKernelFactory

from .test_tier1_pinning import Loader

# fish per (line, condition, day)
FISH: Dict[Tuple[str, str, str], int] = {
    ("L1", "vehicle", "11_Feb_2026"): 5, ("L1", "vehicle", "12_Feb_2026"): 4, ("L1", "ronidazole", "12_Feb_2026"): 6,
    ("L1", "ronidazole", "13_Feb_2026"): 3, ("WT", "danieau", "11_Feb_2026"): 4, ("mut", "danieau", "20_Feb_2026"): 4,
}


def simulated_screen() -> Tuple[Arms, PinnedArchitecture]:
    model = gamma(poisson(RateKernelFactory.spont()))
    model.set_params(np.array([0.4, -0.02, 4.0]))
    arms = [(line, condition) for (line, condition, _), n in FISH.items() for _ in range(n)]
    names = [f"00_07dpf_{line}_Wed_{day}_10h00min00sec_fish_{i}" for (line, _, day), n in FISH.items() for i in range(n)]
    empty = np.array([], float)
    template = PointProcessDataset(event_times=empty, event_trials_idx=empty.astype(int), event_fish_idx=empty.astype(int),
                                   fish_trial_mask=np.ones((len(arms), 5), bool), fish_ids=np.array(names), duration_s=10.0, marks={})
    loader = Loader(model.simulate_dataset(template, np.random.default_rng(0)), arms)
    architecture = PinnedArchitecture(model)
    contrasts = [ntr_contrast("L1"), mutant_contrast("mut", ("WT", "danieau"))]
    return Arms({"bouts": loader}, {"spont": "bouts"}, {"spont": {}}, {"spont": architecture}, contrasts), architecture


def test_ntr_arms_keep_the_days_recorded_in_both() -> None:
    arms, _ = simulated_screen()
    reference, test = arms.datasets("L1", "spont")
    assert (reference.num_fish, test.num_fish) == (4, 6)
    assert set(recording_days(reference.fish_ids)) == set(recording_days(test.fish_ids)) == {"12_Feb_2026"}


def test_mutant_arms_keep_every_day() -> None:
    arms, _ = simulated_screen()
    reference, test = arms.datasets("mut", "spont")
    assert (reference.num_fish, test.num_fish) == (4, 4)


def test_fish_are_permuted_within_their_stratum() -> None:
    strata = np.array([0, 1, 0, 2, 1, 0, 2])
    rng = np.random.default_rng(0)
    permutations = np.array([permute_within(strata, rng) for _ in range(200)])
    assert all(sorted(p) == list(range(7)) for p in permutations)
    assert (strata[permutations] == strata).all()
    assert len({tuple(p) for p in permutations}) == 3 * 2 * 1 * 2 * 2  # every permutation within the strata (3! x 2! x 2!)


def test_permutation_test_counts_the_splits_within_days() -> None:
    arms, architecture = simulated_screen()
    reference, test = (arms._prepare("L1", condition, "spont") for condition in ("vehicle", "ronidazole"))
    reference, test = reference.fish([0, 1, 5, 6, 7]), test.fish([0, 1, 2, 6, 7])  # 2 + 3 fish of two days against 3 + 2 of two days
    days = np.array(["a", "a", "b", "b", "b", "a", "a", "a", "b", "b"])
    result = tier1_permutation_test(architecture, reference, test, n_perm=30, n_jobs=1, strata=days)
    assert result["n_distinct_permutations"] == 10 * 10  # C(5, 2) x C(5, 3)
    assert 0 < result["p_value"] <= 1
