"""Count readouts: trial coverage, days recorded in both arms, permutation within day."""
import numpy as np
import pytest

from BehaviorScreen.screen.line_tests.count_readouts import (
    COUNT_READOUTS, EXPERIMENTS, add_readout_significance, arm_counts, count_test, fish_counts, log_value, run_count_screen, trial_weights,
)
from BehaviorScreen.screen.line_tests.fdr import add_fdr_per_behavior
from BehaviorScreen.screen.line_tests.dataset_utils import mutant_contrast, ntr_contrast
from BehaviorScreen.point_process.dataset import PointProcessDataset

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
    result = count_test({"R": ref}, {"R": test}, 500, rng, strata=np.array(["a", "a", "b", "a", "b", "b"]))
    assert result["p_value_floor"] == pytest.approx(1 / (3 * 3 + 1))  # C(3, 2) x C(3, 1) splits
    separate = count_test({"R": ref}, {"R": test}, 50, rng, strata=np.array(["a"] * 3 + ["b"] * 3))
    assert separate["p_value"] == 1.0  # arms on different days: no other labelling


def test_ntr_arms_keep_shared_days_and_mutants_every_day() -> None:
    arms = [(line, condition) for (line, condition, _), n in FISH.items() for _ in range(n)]
    names = np.array([f"00_07dpf_{line}_Wed_{day}_10h00min00sec_fish_{i}" for (line, _, day), n in FISH.items() for i in range(n)])
    counts = np.random.default_rng(0).poisson(3, (len(arms), 4))
    loader = Loader(dataset(counts, np.ones_like(counts, bool), names), arms)
    counts = arm_counts({"bouts": loader, "saccades": loader}, [ntr_contrast("L1"), mutant_contrast("mut", ("WT", "danieau"))])
    assert (len(counts["L1", "vehicle", "spont_dark"].days), len(counts["L1", "ronidazole", "spont_dark"].days)) == (4, 6)
    assert set(counts["L1", "vehicle", "spont_dark"].days) == set(counts["L1", "ronidazole", "spont_dark"].days) == {"12_Feb_2026"}
    assert (len(counts["WT", "danieau", "spont_dark"].days), len(counts["mut", "danieau", "spont_dark"].days)) == (4, 4)
    okr = counts["L1", "vehicle", "okr"]
    assert set(okr.readouts) == {"R", "P"} and all(len(c) == len(okr.days) for c in okr.readouts.values())  # one fish set per experiment
    assert np.isfinite(log_value(okr.readouts["P"], "P"))


def poisson_counts(rate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([rng.poisson(rate * 10, n), np.full(n, 10.0)])


def test_one_readout_gives_its_own_two_sided_test() -> None:
    rng = np.random.default_rng(0)
    ref, test = poisson_counts(1.0, 12, rng), poisson_counts(1.6, 12, rng)
    result = count_test({"R": ref}, {"R": test}, 999, np.random.default_rng(1))
    assert result["p_value"] == result["readouts"]["R"]["p_value"] == result["readouts"]["R"]["p_adjusted"] < 0.01


def test_max_t_finds_the_readout_that_differs_and_adjusts_the_others() -> None:
    rng = np.random.default_rng(0)
    events = lambda p, n: np.column_stack([rng.binomial(40, p, n), np.full(n, 40)])  # direction: ipsi events of 40
    ref = {"R": poisson_counts(1.0, 15, rng), "P": events(0.5, 15)}
    test = {"R": poisson_counts(1.0, 15, rng), "P": events(0.7, 15)}
    result = count_test(ref, test, 1999, np.random.default_rng(1))
    readouts = result["readouts"]
    assert result["p_value"] == min(r["p_adjusted"] for r in readouts.values()) < 0.01
    assert readouts["P"]["p_adjusted"] < 0.05 < readouts["R"]["p_adjusted"]
    assert all(r["p_adjusted"] >= r["p_value"] for r in readouts.values())


def test_screen_has_a_row_per_contrast_and_experiment_and_one_per_readout() -> None:
    sizes = {("L1", "vehicle"): 10, ("L1", "ronidazole"): 10, ("WT", "danieau"): 8, ("mut", "danieau"): 8}
    arms = [arm for arm, n in sizes.items() for _ in range(n)]
    names = np.array([f"00_07dpf_{line}_Wed_12_Feb_2026_10h00min00sec_fish_{i}" for i, (line, _) in enumerate(arms)])
    rng = np.random.default_rng(0)
    counts = rng.poisson(np.where([a == ("L1", "ronidazole") for a in arms], 1, 4)[:, None], (len(arms), 4))
    loader = Loader(dataset(counts, np.ones_like(counts, bool), names), arms)
    contrasts = [ntr_contrast("L1"), mutant_contrast("mut", ("WT", "danieau"))]
    results, params, readout_tests = run_count_screen(arm_counts({"bouts": loader, "saccades": loader}, contrasts), contrasts, n_perm=199, n_boot=50)
    assert len(results) == len(contrasts) * len(EXPERIMENTS)
    okr = results.set_index(["line", "behavior"]).loc[("L1", "okr")]
    assert okr["status"] == "ok" and okr["p_value"] < 0.05  # the drug arm has a quarter of the events
    assert set(readout_tests.query("line == 'L1' and behavior == 'okr'")["parameter"]) == set(params.query("line == 'L1' and behavior == 'okr'")["parameter"]) == {"R", "P"}
    tier1 = add_fdr_per_behavior(results)
    marked = add_readout_significance(readout_tests, tier1)
    assert marked.query("line == 'L1' and behavior == 'okr' and parameter == 'R'")["significant"].item()
