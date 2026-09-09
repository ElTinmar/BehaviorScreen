# tests/test_layer2_independent_simulator.py
"""
LAYER 2: recovery from a FROM-SCRATCH, independent reference simulator.

These reference simulators do NOT call simulate_stream, RateKernel, or
HistoryKernel AT ALL -- they are written directly with plain numpy, using
textbook formulas typed out independently. This is the direct answer to
"both simulate_stream and _nll could share a bug and this would never be
caught": since generation here shares zero code with the codebase's own
simulate_stream, a bug isolated to simulate_stream cannot silently cancel
out against a matching bug in _nll (there's no shared code for the bug to
live in).

One reference simulator is written per PROCESS FAMILY (not per RateKernel/
HistoryKernel variant) -- the point is validating the family's likelihood
machinery in a way that doesn't depend on which specific kernel shape you
plug in.
"""
import numpy as np
import pytest

from .conftest import dataset_from_streams, assert_recovered, params_dict

from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory
from BehaviorScreen.point_process.hawkes_process import HawkesProcess, HistoryKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalProcess, RenewalKernelFactory
from BehaviorScreen.point_process.survival_process import SurvivalProcess, SurvivalKernelFactory


# =============================================================================
# Reference simulators -- independent of the codebase's own machinery
# =============================================================================

def reference_simulate_homogeneous_poisson(B: float, duration_s: float, rng) -> np.ndarray:
    """Textbook exact simulation via exponential inter-arrival times."""
    events = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / B)
        if t >= duration_s:
            break
        events.append(t)
    return np.array(events)


def reference_simulate_exponential_hawkes(B, alpha, beta, duration_s, rng) -> np.ndarray:
    """
    From-scratch Ogata thinning for an exponential-kernel Hawkes process.
    Deliberately uses np.exp(-beta*dt) written inline -- not
    HistoryKernel.evaluate/decay_envelope/HawkesProcess.simulate_stream.
    """
    events: list = []
    t = 0.0
    while t < duration_s:
        if events:
            hist_now = alpha * np.sum(np.exp(-beta * (t - np.array(events))))
        else:
            hist_now = 0.0
        # +alpha covers the instantaneous jump if an event lands exactly at
        # the candidate time; *1.1 is a simple safety margin (this is a test
        # fixture, not a production-quality thinning algorithm).
        lambda_upper = (B + hist_now + alpha) * 1.1
        w = rng.exponential(1.0 / max(lambda_upper, 1e-12))
        t_candidate = t + w
        if t_candidate >= duration_s:
            break
        if events:
            lam = B + alpha * np.sum(np.exp(-beta * (t_candidate - np.array(events))))
        else:
            lam = B
        if rng.uniform() <= lam / lambda_upper:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)


def reference_simulate_exponential_renewal_excitation(B, A_exc, tau_exc, duration_s, rng) -> np.ndarray:
    """
    From-scratch thinning for RenewalProcess(homogeneous, exponential_excitation):
    rate(t) = B * (1 + A_exc*exp(-(t-t_last)/tau_exc)), t_last = most recent
    accepted event (no modulation before the first event). Written directly,
    independent of RenewalKernel/RenewalProcess.simulate_stream.
    """
    events: list = []
    t = 0.0
    t_last = None
    lambda_upper = B * (1.0 + A_exc) * 1.1
    while t < duration_s:
        w = rng.exponential(1.0 / max(lambda_upper, 1e-12))
        t_candidate = t + w
        if t_candidate >= duration_s:
            break
        if t_last is None:
            lam = B
        else:
            lam = B * (1.0 + A_exc * np.exp(-(t_candidate - t_last) / tau_exc))
        if rng.uniform() <= lam / lambda_upper:
            events.append(t_candidate)
            t_last = t_candidate
        t = t_candidate
    return np.array(events)


def reference_simulate_survival_constant_hazard(B: float, duration_s: float, rng):
    """First-passage time under a constant hazard is just Exp(B), censored
    at duration_s -- exact inverse-CDF sampling, no thinning needed."""
    t_event = rng.exponential(1.0 / B)
    if t_event >= duration_s:
        return np.array([])  # censored
    return np.array([t_event])


# =============================================================================
# Tests
# =============================================================================

class TestPoissonIndependentSimulatorRecovery:

    def test_homogeneous_rate(self, rng_factory):
        rng = rng_factory(100)
        true_B, T, n_fish, n_trials = 0.5, 15.0, 60, 4

        streams = [
            [reference_simulate_homogeneous_poisson(true_B, T, rng) for _ in range(n_trials)]
            for _ in range(n_fish)
        ]
        dataset = dataset_from_streams(streams, T, n_trials)

        model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        model.fit(dataset)
        assert_recovered("B", model.param_dict_["B"], true_B, rtol=0.08)


@pytest.mark.slow
class TestHawkesIndependentSimulatorRecovery:

    def test_homogeneous_baseline_exponential_history(self, rng_factory):
        rng = rng_factory(101)
        true_B, true_alpha, true_beta = 0.4, 0.3, 2.0
        T, n_fish, n_trials = 20.0, 60, 5

        streams = [
            [reference_simulate_exponential_hawkes(true_B, true_alpha, true_beta, T, rng) for _ in range(n_trials)]
            for _ in range(n_fish)
        ]
        dataset = dataset_from_streams(streams, T, n_trials)
        assert len(dataset.event_times) > 2000

        model = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        model.fit_multistart(dataset, n_starts=8, n_jobs=1, seed=102)

        pd = params_dict(model)
        assert_recovered("B", pd["B"], true_B, rtol=0.15)
        assert_recovered("alpha_hawkes", pd["alpha_hawkes"], true_alpha, rtol=0.35)
        assert_recovered("beta_hawkes", pd["beta_hawkes"], true_beta, rtol=0.35)


@pytest.mark.slow
class TestRenewalIndependentSimulatorRecovery:

    def test_homogeneous_baseline_exponential_excitation(self, rng_factory):
        rng = rng_factory(103)
        true_B, true_A_exc, true_tau_exc = 0.5, 1.0, 0.3
        T, n_fish, n_trials = 20.0, 60, 5

        streams = [
            [reference_simulate_exponential_renewal_excitation(true_B, true_A_exc, true_tau_exc, T, rng)
             for _ in range(n_trials)]
            for _ in range(n_fish)
        ]
        dataset = dataset_from_streams(streams, T, n_trials)
        assert len(dataset.event_times) > 2000

        model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(), RenewalKernelFactory.exponential_excitation()
        )
        model.fit_multistart(dataset, n_starts=8, n_jobs=1, seed=104)

        pd = params_dict(model)
        assert_recovered("B", pd["B"], true_B, rtol=0.15)
        assert_recovered("A_excitation", pd["A_excitation"], true_A_exc, rtol=0.35)
        assert_recovered("tau_excitation", pd["tau_excitation"], true_tau_exc, rtol=0.35)


class TestSurvivalIndependentSimulatorRecovery:

    def test_constant_hazard(self, rng_factory):
        rng = rng_factory(105)
        true_B, T, n_fish, n_trials = 0.4, 5.0, 200, 6

        streams = [
            [reference_simulate_survival_constant_hazard(true_B, T, rng) for _ in range(n_trials)]
            for _ in range(n_fish)
        ]
        dataset = dataset_from_streams(streams, T, n_trials)
        n_exact = sum(1 for _, _, t_ev in dataset.iter_streams() if len(t_ev) > 0)
        assert n_exact > 200, "sanity check: enough uncensored observations simulated"

        model = SurvivalProcess(SurvivalKernelFactory.constant_hazard())
        model.fit(dataset)
        assert_recovered("B", model.param_dict_["B"], true_B, rtol=0.10)