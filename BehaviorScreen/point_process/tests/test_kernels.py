"""
Every rate and renewal kernel of the library, at its initial guesses and at a point with every
parameter moved (trial effects not zero): the integral the likelihood uses against quadrature of
the rate, and a simulation that agrees with its compensator.
"""
from typing import Callable, Dict

import numpy as np
import pytest
from scipy.integrate import quad

from .conftest import DummyFitResult, make_scaffold_dataset
from .test_layer3_simulation_compensator_consistency import _km_exp1_sup_distance

from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.poisson_process import PoissonProcess, PreyCapture, RateKernel, RateKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalKernel, RenewalKernelFactory, RenewalProcess

DURATION_S, STIM_FREQ = 24.0, 90 / 140
RATE_KERNELS: Dict[str, Callable[[], RateKernel]] = {
    **{name: getattr(RateKernelFactory, name) for name in vars(RateKernelFactory) if not name.startswith("_")},
    **{f"prey {variant}": (lambda variant=variant: PreyCapture.variant(variant, STIM_FREQ)) for variant in PreyCapture._VARIANTS},
}
flat = RateKernelFactory.homogeneous_poisson
RENEWAL_KERNELS: Dict[str, Callable[[], RenewalKernel]] = {
    "weibull_recovery": RenewalKernelFactory.weibull_recovery,
    "exponential_excitation": RenewalKernelFactory.exponential_excitation,
    "refractory_delayed_excitation": RenewalKernelFactory.refractory_delayed_excitation,
    "delayed_excitation": RenewalKernelFactory.delayed_excitation,
    "dead_time_delayed_excitation": lambda: RenewalKernelFactory.dead_time_delayed_excitation(0.075),
}
RENEWAL_PROCESSES: Dict[str, Callable[[], RenewalProcess]] = {
    **{name: (lambda kernel=kernel: RenewalProcess(flat(), kernel())) for name, kernel in RENEWAL_KERNELS.items()},
    "recovery and burst": lambda: RenewalProcess(flat(), RenewalKernelFactory.weibull_recovery(), RenewalKernelFactory.burst_excitation()),
    "recovery and relative burst": lambda: RenewalProcess(
        RateKernelFactory.spont(), RenewalKernelFactory.weibull_recovery(), RenewalKernelFactory.burst_excitation(1.0, rate_init=2.0, relative=True)
    ),
}
POINTS = ["initial guesses", "moved"]


def parameters(initial_guesses: list, bounds: list, point: str) -> np.ndarray:
    """The initial guesses, or each moved 10% of the way to the middle of its bounds (by 0.1 without bounds)."""
    x = np.array(initial_guesses, float)
    if point == "moved":
        middle = np.array([x[i] + 1.0 if None in b else (b[0] + b[1]) / 2 for i, b in enumerate(bounds)])
        x = x + 0.1 * (middle - x)
    return x


def residual_distance(model: PointProcess, num_fish: int) -> float:
    data = model.simulate_dataset(make_scaffold_dataset(num_fish, 8, DURATION_S), np.random.default_rng(3))
    model.fit_result = DummyFitResult()
    rescaled = model.time_rescaling(data)
    residuals, censored = np.asarray(rescaled["residuals"], float), np.asarray(rescaled["censored"], bool)
    assert np.sum(~censored) > 2500
    return _km_exp1_sup_distance(model, residuals, censored)


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("name", RATE_KERNELS)
def test_rate_integral_matches_quadrature(name: str, point: str) -> None:
    """On the 0.02 s grid of the likelihood, over a trial and up to a time within it, in a first and a later trial."""
    kernel = RATE_KERNELS[name]()
    params = list(parameters(kernel.initial_guesses, kernel.bounds, point))
    breaks = np.linspace(0, DURATION_S, 97)[1:-1]
    for trial in (0, 7):
        rate = lambda t: float(kernel.evaluate(np.array([t]), np.array([trial]), params)[0])
        whole = quad(rate, 0, DURATION_S, limit=2000, points=breaks, epsabs=1e-12)[0]
        part = quad(rate, 0, 3.21, limit=2000, points=breaks[breaks < 3.21], epsabs=1e-12)[0]
        assert kernel.integrate(DURATION_S, np.array([trial]), params, integration_dt=0.02)[0] == pytest.approx(whole, rel=1e-3)
        assert kernel.cumulative_integrate(np.array([3.21, DURATION_S]), trial, params, 0.02) == pytest.approx([part, whole], rel=1e-3)


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("name", RATE_KERNELS)
def test_poisson_simulation_agrees_with_the_compensator(name: str, point: str) -> None:
    kernel = RATE_KERNELS[name]()
    model = PoissonProcess(kernel)
    model.set_params(parameters(kernel.initial_guesses, kernel.bounds, point))
    assert residual_distance(model, num_fish=120) <= 0.03


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("name", RENEWAL_PROCESSES)
def test_renewal_segment_integral_matches_quadrature(name: str, point: str) -> None:
    """
    The likelihood's integral between two bouts, on its 0.02 s grid: within 1% for an interval as short
    as 0.2 s, where the recovery rises fastest (5% with a dead time: a jump inside a grid step).
    """
    model, trial = RENEWAL_PROCESSES[name](), 3.0
    params = list(parameters(model.initial_guesses, model.bounds, point))
    base, renewal, excitation = model._split(params)

    def intensity(t: float, t_bout: float, t_event: float) -> float:
        t = np.array([t])
        drive = model.kernel.evaluate(t, np.array([trial]), base)[0]
        burst = 0.0 if model.excitation is None else model.excitation.evaluate(t - t_event, excitation)[0] * (drive if model._relative else 1.0)
        return float(model.renewal_kernel.evaluate(t - t_bout, renewal)[0] * (drive + burst))

    # (start, end, last bout, last event): the event is the last bout, or an earlier one
    for t0, t1, t_bout, t_event in [(1.0, 1.83, 1.0, 1.0), (1.0, 1.21, 1.0, 0.4), (2.0, 9.0, 2.0, 2.0)]:
        expected = quad(intensity, t0, t1, args=(t_bout, t_event), limit=800, points=[t0 + 0.075], epsabs=1e-13)[0]
        assert model._segment_integral(t0, t1, trial, params, t_bout, t_event) == pytest.approx(expected, rel=5e-2 if "dead_time" in name else 1e-2)


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("name", RENEWAL_PROCESSES)
def test_renewal_simulation_agrees_with_the_compensator(name: str, point: str) -> None:
    model = RENEWAL_PROCESSES[name]()
    model.set_params(parameters(model.initial_guesses, model.bounds, point))
    assert residual_distance(model, num_fish=120) <= 0.03
