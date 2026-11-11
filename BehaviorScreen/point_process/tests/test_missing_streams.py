"""Likelihoods of datasets in which some (fish, trial) streams were not observed, as in the recordings."""
from dataclasses import replace
from typing import Callable, Dict

import numpy as np
import pytest
from scipy.integrate import quad

from .conftest import reference_dataset
from .test_time_of_day import FACTORIES as GAIN_MODELS

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.frailty import SideFrailty
from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalKernelFactory, RenewalProcess
from BehaviorScreen.point_process.sided_process import SidedPoissonProcess, SidedRenewalProcess, SidedSurvivalProcess
from BehaviorScreen.point_process.survival_process import SurvivalKernelFactory, SurvivalProcess

flat, shaped = RateKernelFactory.homogeneous_poisson, RateKernelFactory.spont
recovery = RenewalKernelFactory.weibull_recovery
burst = lambda: RenewalKernelFactory.burst_excitation(1.0, rate_init=2.0, relative=True)
bump = lambda: SurvivalKernelFactory.gaussian_bump_baseline(t_init=5, t_bounds=(4, 6))

# Without a fish effect: the likelihood is a sum over streams.
STREAM_MODELS: Dict[str, Callable[[], PointProcess]] = {
    "poisson": lambda: PoissonProcess(shaped()),
    "sided poisson": lambda: SidedPoissonProcess(shaped(), flat()),
    "renewal": lambda: RenewalProcess(shaped(), recovery(), burst()),
    "sided renewal": lambda: SidedRenewalProcess(shaped(), flat(), recovery(), burst()),
    "survival": lambda: SurvivalProcess(bump()),
    "sided survival": lambda: SidedSurvivalProcess(bump()),
}
MODELS: Dict[str, Callable[[], PointProcess]] = {
    **STREAM_MODELS,
    **GAIN_MODELS,
    "sided renewal + side": lambda: SideFrailty(SidedRenewalProcess(shaped(), flat(), recovery(), burst()), trial_gain=True),
    "sided survival + side": lambda: SideFrailty(SidedSurvivalProcess(bump())),
}


def observed(dataset: PointProcessDataset, mask: np.ndarray) -> PointProcessDataset:
    """The dataset with only the streams of mask observed: the events (and other bouts) of the others are gone."""
    def keep(d: PointProcessDataset) -> PointProcessDataset:
        take = mask[d.event_fish_idx, d.event_trials_idx]
        return d._with_events(take, d.event_fish_idx[take], fish_trial_mask=mask)

    return replace(keep(replace(dataset, history=None)), history=keep(dataset.history))


@pytest.fixture(scope="module")
def complete() -> PointProcessDataset:
    return reference_dataset(num_fish=12, num_trials=6)


@pytest.fixture(scope="module")
def mask(complete: PointProcessDataset) -> np.ndarray:
    mask = np.random.default_rng(1).random(complete.fish_trial_mask.shape) > 0.25
    mask[3] = False  # a fish without any observed trial
    return mask


def nll(model: PointProcess, dataset: PointProcessDataset) -> float:
    return model._nll(list(model.initial_guesses), dataset)


def test_homogeneous_rate_counts_the_observed_streams_only(complete: PointProcessDataset, mask: np.ndarray) -> None:
    model, data, rate = PoissonProcess(flat()), observed(complete, mask), 0.6
    expected = rate * complete.duration_s * mask.sum() - len(data.event_times) * np.log(rate)
    assert model._nll([rate], data) == pytest.approx(expected, rel=1e-12)
    assert model.jax_value(np.array([rate]), data) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("kind", STREAM_MODELS)
def test_unobserved_stream_differs_from_an_empty_one_by_its_expected_events(kind: str, complete: PointProcessDataset) -> None:
    """Fish 2 in trial 4, observed without any bout against not observed: the integral of the rate over the trial."""
    model, fish, trial = STREAM_MODELS[kind](), 2, 4
    others = np.ones_like(complete.fish_trial_mask)
    others[fish, trial] = False
    unobserved = observed(complete, others)
    empty = replace(unobserved, fish_trial_mask=complete.fish_trial_mask, history=replace(unobserved.history, fish_trial_mask=complete.fish_trial_mask))
    model.set_params(np.array(model.initial_guesses, float))
    base = getattr(model, "sides", model)  # a sided renewal process: its stimulus-driven rates
    rate = lambda t: float(np.sum(base.predict(np.array([t]), np.array([trial]))))
    expected = quad(rate, 0, complete.duration_s, limit=400, points=[4, 5, 6])[0]
    assert nll(model, empty) - nll(model, unobserved) == pytest.approx(expected, rel=2e-4)


@pytest.mark.parametrize("kind", MODELS)
def test_numpy_and_jax_agree(kind: str, complete: PointProcessDataset, mask: np.ndarray) -> None:
    model, data = MODELS[kind](), observed(complete, mask)
    if not model.supports_jax():
        pytest.skip("no JAX likelihood")
    assert model.jax_value(np.array(model.initial_guesses, float), data) == pytest.approx(nll(model, data), rel=1e-8)


@pytest.mark.parametrize("kind", MODELS)
def test_fish_without_observed_trial_counts_for_nothing(kind: str, complete: PointProcessDataset, mask: np.ndarray) -> None:
    model, data = MODELS[kind](), observed(complete, mask)
    seen = np.flatnonzero(mask.any(axis=1))
    assert len(seen) == complete.num_fish - 1
    assert nll(model, data) == pytest.approx(nll(model, data.fish(seen)), abs=1e-5)  # its gain integrates to 1 by quadrature


@pytest.mark.parametrize("kind", MODELS)
def test_likelihood_is_a_product_over_fish(kind: str, complete: PointProcessDataset, mask: np.ndarray) -> None:
    model, data = MODELS[kind](), observed(complete, mask)
    seen = np.flatnonzero(mask.any(axis=1))
    assert sum(nll(model, data.fish([f])) for f in seen) == pytest.approx(nll(model, data), abs=1e-5)
