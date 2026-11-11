"""Count readouts: trial coverage, days recorded in both arms, permutation within day."""
import numpy as np
import pytest

from BehaviorScreen.ablation_screen.count_readouts import COUNT_READOUTS, arm_counts, count_test, fish_counts, log_value, trial_weights
from BehaviorScreen.ablation_screen.dataset_utils import mutant_contrast, ntr_contrast
from BehaviorScreen.point_process.point_process.dataset import PointProcessDataset

from .test_tier1_pinning import Loader
from .test_within_day import FISH


def dataset(counts: np.ndarray, mask: np.ndarray, names: np.ndarray = None) -> PointProcessDataset:
    """counts[fish, trial] events in each stream of mask, at mid-trial, on the stimulus side."""
    fish, trial = np.nonzero(counts * mask)
    repeats = counts[fish, trial]
    n = int(repeats.sum())
    return PointProcessDataset(event_times=np.full(n, 5.0), event_trials_idx=np.repeat(trial, repeats), event_fish_idx=np.repeat(fish, repeats),
                               fish_trial_mask=mask, fish_ids=np.arange(len(mask)).astype(str) if names is None else names, duration_s=10.0,
                               marks={"laterality": np.ones(n, int), "category": np.zeros(n, int)})


def test_rate_of_a_fish_missing_late_trials_matches_the_fish_with_all_trials() -> None:
    profile = np.array([8, 4, 2, 2])  # events per trial: habituation
    full = np.ones((3, 4), bool)
    weights = trial_weights(dataset(np.tile(profile, (3, 1)), full))
    assert weights == pytest.approx(profile / profile.mean())
    mask = np.array([[True, True, True, True], [True, True, False, False]])
    arm = dataset(np.tile(profile, (2, 1)), mask)
    rate = COUNT_READOUTS["spont_dark"]
    plain, weighted = fish_counts(arm, rate), fish_counts(arm, rate, weights)
    assert plain[1, 0] / plain[1, 1] == pytest.approx(1.5 * plain[0, 0] / plain[0, 1])  # early trials only: rate looks 50% higher
    assert weighted[1, 0] / weighted[1, 1] == pytest.approx(weighted[0, 0] / weighted[0, 1])
    assert weighted[0] == pytest.approx(plain[0])


def test_permutation_within_strata() -> None:
    rng = np.random.default_rng(0)
    ref, test = np.column_stack([[5, 6, 50], [10.0] * 3]), np.column_stack([[7, 60, 55], [10.0] * 3])
    result = count_test(ref, test, "R", 500, rng, strata=np.array(["a", "a", "b", "a", "b", "b"]))
    assert result["p_value_floor"] == pytest.approx(1 / (3 * 3 + 1))  # C(3, 2) x C(3, 1) splits
    separate = count_test(ref, test, "R", 50, rng, strata=np.array(["a"] * 3 + ["b"] * 3))
    assert separate["p_value"] == 1.0  # arms on different days: no other labelling


def test_ntr_arms_keep_shared_days_and_mutants_every_day() -> None:
    arms = [(line, condition) for (line, condition, _), n in FISH.items() for _ in range(n)]
    names = np.array([f"00_07dpf_{line}_Wed_{day}_10h00min00sec_fish_{i}" for (line, _, day), n in FISH.items() for i in range(n)])
    counts = np.random.default_rng(0).poisson(3, (len(arms), 4))
    loader = Loader(dataset(counts, np.ones_like(counts, bool), names), arms)
    values, days = arm_counts({"bouts": loader, "saccades": loader}, [ntr_contrast("L1"), mutant_contrast("mut", ("WT", "danieau"))])
    assert (len(values["L1", "vehicle", "spont_dark"]), len(values["L1", "ronidazole", "spont_dark"])) == (4, 6)
    assert set(days["L1", "vehicle", "spont_dark"]) == set(days["L1", "ronidazole", "spont_dark"]) == {"12_Feb_2026"}
    assert (len(values["WT", "danieau", "spont_dark"]), len(values["mut", "danieau", "spont_dark"])) == (4, 4)
    assert np.isfinite(log_value(values["L1", "vehicle", "okr_direction"], "P"))
