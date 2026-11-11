"""The time-of-day factor on the fish gain: g_f mu_f, mu_f = exp(beta_hour (t_f - HOUR_REFERENCE))."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from .conftest import DummyFitResult, make_scaffold_dataset, reference_dataset
from .test_layer3_simulation_compensator_consistency import _km_exp1_sup_distance

from BehaviorScreen.point_process.dataset import BehavioralDataLoader, PointProcessDataset
from BehaviorScreen.point_process.frailty import (
    HOUR_REFERENCE, GammaMixedEffectsProcess, SideFrailty, ZeroInflatedGammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalKernelFactory, RenewalProcess
from BehaviorScreen.point_process.sided_process import SidedPoissonProcess
from BehaviorScreen.point_process.survival_process import SurvivalKernelFactory, SurvivalProcess

flat = RateKernelFactory.homogeneous_poisson
renewal = lambda: RenewalProcess(flat(), RenewalKernelFactory.weibull_recovery(), RenewalKernelFactory.burst_excitation(1.0, rate_init=2.0, relative=True))
FACTORIES = {
    "whole": lambda **k: GammaMixedEffectsProcess(PoissonProcess(flat()), **k),
    "whole + trial": lambda **k: GammaMixedEffectsProcess(PoissonProcess(flat()), trial_gain=True, **k),
    "drive": lambda **k: GammaMixedEffectsProcess(renewal(), **k),
    "drive + trial": lambda **k: GammaMixedEffectsProcess(renewal(), trial_gain=True, **k),
    "zero-inflated drive": lambda **k: ZeroInflatedGammaMixedEffectsProcess(renewal(), fit_c=True, c_init=0.1, **k),
    "sided + trial": lambda **k: SideFrailty(SidedPoissonProcess(flat(), flat()), trial_gain=True, **k),
    "survival": lambda **k: GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=5, t_bounds=(4, 6))), **k),
}


def with_hours(dataset: PointProcessDataset, rng: np.random.Generator) -> PointProcessDataset:
    hours = rng.uniform(9.0, 18.5, dataset.num_fish)
    history = None if dataset.history is None else replace(dataset.history, fish_hours=hours)
    return replace(dataset, fish_hours=hours, history=history)


def with_beta(model, beta: float) -> np.ndarray:
    params = list(model.initial_guesses)
    params[model._hour_idx] = beta
    return np.asarray(params, dtype=float)


@pytest.fixture(scope="module")
def data() -> PointProcessDataset:
    return with_hours(reference_dataset(num_fish=30), np.random.default_rng(3))


@pytest.mark.parametrize("kind", FACTORIES)
def test_beta_zero_is_the_model_without_it(kind, data):
    plain, hour = FACTORIES[kind](), FACTORIES[kind](time_of_day=True)
    assert hour.param_names[hour._hour_idx] == "beta_hour" and "+Hour" in hour.name
    assert hour.param_names[: hour._hour_idx] + hour.param_names[hour._hour_idx + 1:] == plain.param_names
    p0, p1 = np.asarray(plain.initial_guesses, float), with_beta(hour, 0.0)
    assert hour._nll(list(p1), data) == pytest.approx(plain._nll(list(p0), data), rel=1e-10)
    if hour.supports_jax():
        assert hour.jax_value(p1, data) == pytest.approx(plain.jax_value(p0, data), rel=1e-9)


@pytest.mark.parametrize("kind", [k for k in FACTORIES if FACTORIES[k]().supports_jax()])
def test_numpy_and_jax_agree(kind, data):
    hour = FACTORIES[kind](time_of_day=True)
    params = with_beta(hour, -0.08)
    assert hour.jax_value(params, data) == pytest.approx(hour._nll(list(params), data), rel=1e-8)
    value, grad = hour.nll_and_grad(params, data)
    step = np.zeros_like(params)
    step[hour._hour_idx] = 1e-5
    numeric = (hour._nll(list(params + step), data) - hour._nll(list(params - step), data)) / 2e-5
    assert grad[hour._hour_idx] == pytest.approx(numeric, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("kind", ["whole", "whole + trial", "survival"])
def test_factor_scales_the_rate(kind, data):
    """A gain on the whole rate: hours shifted by c equal the rate amplitude times exp(beta c)."""
    hour, beta, c = FACTORIES[kind](time_of_day=True), -0.07, 2.5
    params = with_beta(hour, beta)
    amplitudes = [0] if kind != "survival" else [hour.param_names.index("H"), hour.param_names.index("B")]  # h(t) = H bump + B
    scaled = params.copy()
    scaled[amplitudes] *= np.exp(beta * c)
    shifted = replace(data, fish_hours=data.fish_hours + c)
    assert hour._nll(list(params), shifted) == pytest.approx(hour._nll(list(scaled), data), rel=1e-9)


def test_needs_every_fish_hour(data):
    hour = FACTORIES["whole"](time_of_day=True)
    with pytest.raises(ValueError, match="recording time"):
        hour._nll(list(with_beta(hour, 0.0)), replace(data, fish_hours=None))
    missing = data.fish_hours.copy()
    missing[0] = np.nan
    with pytest.raises(ValueError, match="recording time"):
        hour._nll(list(with_beta(hour, 0.0)), replace(data, fish_hours=missing))


def test_datasets_keep_fish_hours(data):
    picked = data.fish([3, 1, 3])
    assert np.array_equal(picked.fish_hours, data.fish_hours[[3, 1, 3]])
    assert np.array_equal(picked.history.fish_hours, data.fish_hours[[3, 1, 3]])
    assert np.array_equal(PointProcessDataset.concat(data, picked).fish_hours, np.concatenate([data.fish_hours, picked.fish_hours]))
    assert np.array_equal(data.select(laterality=1).fish_hours, data.fish_hours)
    model = FACTORIES["drive"](time_of_day=True)
    model.set_params(with_beta(model, -0.05))
    simulated = model.simulate_dataset(data, np.random.default_rng(0))
    assert np.array_equal(simulated.fish_hours, data.fish_hours) and np.array_equal(simulated.history.fish_hours, data.fish_hours)


def test_loader_reads_recording_times():
    loader = object.__new__(BehavioralDataLoader)
    hours = {"a": 9.25, "b": 17.5}
    angle = lambda h: 2 * np.pi * h / 24
    loader.events = pd.DataFrame({"file": ["a", "a", "b"], "cos_daytime": [np.cos(angle(hours[f])) for f in "aab"],
                                  "sin_daytime": [np.sin(angle(hours[f])) for f in "aab"]})
    assert loader._fish_hours(np.array(["b", "a", "c"])) == pytest.approx([17.5, 9.25, np.nan], nan_ok=True)
    loader.events = loader.events.drop(columns=["cos_daytime", "sin_daytime"])
    assert loader._fish_hours(np.array(["a"])) is None


def simulated(model, params: np.ndarray, num_fish: int, num_trials: int, duration_s: float, seed: int, template=None) -> PointProcessDataset:
    rng = np.random.default_rng(seed)
    model.set_params(params)
    template = template or make_scaffold_dataset(num_fish, num_trials, duration_s)
    return model.simulate_dataset(with_hours(template, rng), rng)


def test_beta_recovered_whole_rate():
    true = FACTORIES["whole"](time_of_day=True)
    data = simulated(true, np.array([0.6, 3.0, -0.06]), 500, 8, 10.0, seed=11)
    model = FACTORIES["whole"](time_of_day=True)
    model.fit(data)
    se = np.sqrt(np.diag(np.linalg.inv(model.jax_hessian(model.params_, data))))[model._hour_idx]
    assert abs(model.params_[model._hour_idx] + 0.06) < 3 * se and se < 0.015


@pytest.mark.slow
def test_beta_recovered_renewal_with_trial_gain():
    true = FACTORIES["drive + trial"](time_of_day=True)
    params = with_beta(true, -0.06)
    data = simulated(true, params, 300, 8, 24.0, seed=12, template=reference_dataset(num_fish=300))
    model = FACTORIES["drive + trial"](time_of_day=True)
    model.fit(data)
    se = np.sqrt(np.diag(np.linalg.inv(model.jax_hessian(model.params_, data))))[model._hour_idx]
    assert abs(model.params_[model._hour_idx] + 0.06) < 3 * se


@pytest.mark.parametrize("kind", ["whole", "drive + trial"])
def test_time_rescaling_calibrated(kind):
    """Simulated with mu_f, rescaled with mu_f: Exp(1) residuals."""
    model = FACTORIES[kind](time_of_day=True)
    params = with_beta(model, -0.1)
    template = make_scaffold_dataset(250, 6, 8.0) if kind == "whole" else reference_dataset(num_fish=150)
    data = simulated(model, params, 250, 6, 8.0, seed=21, template=template)
    model.fit_result = DummyFitResult()
    tr = model.time_rescaling(data)
    residuals, censored = np.asarray(tr["residuals"], float), np.asarray(tr["censored"], bool)
    assert _km_exp1_sup_distance(model, residuals, censored) <= 0.07


def test_estimated_gains_exclude_the_factor():
    """Fish far from the reference hour: their posterior g_f stays near 1 when mu_f explains their rate."""
    model = FACTORIES["whole"](time_of_day=True)
    data = simulated(model, np.array([0.8, 50.0, -0.15]), 400, 8, 10.0, seed=5)  # r = 50: gains near 1
    gains = model.estimate_fish_gains(data)
    late = data.fish_hours[gains.fish_idx] > 17
    assert gains.estimated_gain[late].mean() == pytest.approx(1.0, abs=0.05)
    assert gains.time_of_day_factor[late].mean() < 0.65
    assert HOUR_REFERENCE == 13.5


def test_ladder_starts_it_from_the_model_without_it():
    """In a ladder, the time-of-day rung starts at its counterpart's fit with beta_hour = 0, wherever it sits: never a worse fit."""
    from BehaviorScreen.point_process.inference import ModelComparator

    true = FACTORIES["whole + trial"](time_of_day=True)
    data = simulated(true, with_beta(true, -0.08), 200, 6, 10.0, seed=31)
    plain, other, hour = FACTORIES["whole + trial"](), FACTORIES["whole"](), FACTORIES["whole + trial"](time_of_day=True)
    assert hour.starts_from == plain.name
    ModelComparator.compare([plain, other, hour], data, n_starts=3, n_jobs=1)
    expected = dict(plain.param_dict_, beta_hour=0.0)
    assert hour.initial_guesses == pytest.approx([expected[n] for n in hour.param_names])
    assert hour.log_likelihood >= plain.log_likelihood - 1e-6
    assert hour.bounds[hour._hour_idx] == (-0.2, 0.2)
