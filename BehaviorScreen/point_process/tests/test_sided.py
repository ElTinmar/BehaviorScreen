"""Sided processes: which side D favors, against counts made by hand, and simulators that agree with the likelihood."""
from typing import Tuple

import numpy as np
import pytest

from .conftest import DummyFitResult, make_scaffold_dataset
from .test_layer3_simulation_compensator_consistency import _km_exp1_sup_distance
from .test_missing_streams import MODELS

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.poisson_process import RateKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalKernelFactory
from BehaviorScreen.point_process.sided_process import CONTRA, IPSI, SidedPoissonProcess, SidedRenewalProcess, SidedSurvivalProcess
from BehaviorScreen.point_process.survival_process import SurvivalKernelFactory

flat = RateKernelFactory.homogeneous_poisson
NUM_FISH, NUM_TRIALS, DURATION_S = 50, 4, 10.0


@pytest.fixture(scope="module")
def events() -> PointProcessDataset:
    """Events at a constant rate, 70% of them ipsilateral."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.5 * DURATION_S, (NUM_FISH, NUM_TRIALS))
    fish, trial = (np.repeat(index.ravel(), counts.ravel()) for index in np.indices(counts.shape))
    return PointProcessDataset(
        event_times=rng.uniform(0, DURATION_S, counts.sum()), event_trials_idx=trial, event_fish_idx=fish,
        fish_trial_mask=np.ones(counts.shape, bool), fish_ids=np.arange(NUM_FISH).astype(str), duration_s=DURATION_S,
        marks={"laterality": np.where(rng.random(counts.sum()) < 0.7, IPSI, CONTRA)},
    )


def side_counts(sides: np.ndarray) -> Tuple[int, int]:
    return int(np.sum(sides == IPSI)), int(np.sum(sides == CONTRA))


def test_sided_poisson_rates_are_the_counts_of_each_side(events: PointProcessDataset) -> None:
    ipsi, contra = side_counts(events.marks["laterality"])
    model = SidedPoissonProcess(flat(), flat())
    model.fit(events)
    assert model.param_dict_["B"] == pytest.approx(ipsi / (NUM_FISH * NUM_TRIALS * DURATION_S), rel=1e-3)
    assert model.param_dict_["D"] == pytest.approx(ipsi / contra, rel=1e-3)


def test_sided_renewal_d_is_the_ratio_of_the_sides(events: PointProcessDataset) -> None:
    ipsi, contra = side_counts(events.marks["laterality"])
    model = SidedRenewalProcess(flat(), flat(), RenewalKernelFactory.weibull_recovery())
    model.fit(events)
    assert model.param_dict_["D"] == pytest.approx(ipsi / contra, rel=1e-3)


def test_sided_survival_d_is_the_odds_of_a_first_event_away(events: PointProcessDataset) -> None:
    """D favors contra (preferred); the hazard: first events over the time at risk."""
    stream = events.event_fish_idx * NUM_TRIALS + events.event_trials_idx
    order = np.lexsort((events.event_times, stream))
    first = order[np.unique(stream[order], return_index=True)[1]]
    ipsi, contra = side_counts(events.marks["laterality"][first])
    at_risk = events.event_times[first].sum() + (NUM_FISH * NUM_TRIALS - len(first)) * DURATION_S
    model = SidedSurvivalProcess(SurvivalKernelFactory.constant_hazard())
    model.fit(events)
    assert model.preferred == CONTRA
    assert model.param_dict_["D"] == pytest.approx(contra / ipsi, rel=1e-3)
    assert model.param_dict_["B"] == pytest.approx(len(first) / at_risk, rel=1e-3)


@pytest.mark.parametrize("kind, num_fish, ipsi_share", [
    ("sided poisson", 150, 2 / 3), ("sided renewal", 150, 2 / 3), ("sided survival", 1500, 1 / 3),
    ("sided + trial", 150, None), ("sided renewal + side", 150, None), ("sided survival + side", 1500, None),
])
def test_simulator_agrees_with_the_likelihood(kind: str, num_fish: int, ipsi_share: float) -> None:
    """
    Simulated at D = 2: rescaled intervals are Exp(1) under the same parameters, and without a
    fish side preference the share of ipsi events is D / (1 + D) (survival: 1 / (1 + D)).
    """
    model = MODELS[kind]()
    model.set_params(np.array(model.initial_guesses, float))
    assert model.param_dict_["D"] == 2.0
    data = model.simulate_dataset(make_scaffold_dataset(num_fish, 6, DURATION_S), np.random.default_rng(7))
    model.fit_result = DummyFitResult()
    rescaled = model.time_rescaling(data)
    residuals, censored = np.asarray(rescaled["residuals"], float), np.asarray(rescaled["censored"], bool)
    assert np.sum(~censored) > 2500
    assert _km_exp1_sup_distance(model, residuals, censored) <= 0.03
    if ipsi_share is not None:
        n = len(data.event_times)
        assert np.mean(data.marks["laterality"] == IPSI) == pytest.approx(ipsi_share, abs=4 * np.sqrt(ipsi_share * (1 - ipsi_share) / n))
