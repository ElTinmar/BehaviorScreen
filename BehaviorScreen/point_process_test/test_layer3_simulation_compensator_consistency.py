# tests/test_layer3_simulation_compensator_consistency.py
"""
LAYER 3: self-consistency between simulate_stream (uses RateKernel/
HistoryKernel .evaluate, pointwise) and cumulative_integrated_intensity
(uses .integrate, closed-form/quadrature) -- checked at TRUE (not fitted)
parameters via Ogata's time-rescaling theorem, with NO optimizer involved
anywhere in the test.

If the intensity that simulate_stream actually samples from and the
intensity that cumulative_integrated_intensity actually reports disagree
(e.g. because a HistoryKernel's `integral_func` doesn't truly match its
`func`, or a RateKernel's `_intensity_upper_bound` silently uses different
parameters than `predict`), this shows up here directly as non-Uniform(0,1)
rescaled residuals -- without needing an independent simulator (Layer 2) or
a working optimizer (Layer 4) at all. This is the cheapest layer to extend
to a new model family: no reference formulas to derive by hand, just
"simulate at true params, time_rescale, check uniformity."
"""
import numpy as np
import pytest
from scipy.stats import kstest

from .conftest import make_scaffold_dataset, simulate_dataset_from_model, DummyFitResult

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory
from BehaviorScreen.point_process.hawkes_process import HawkesProcess, HistoryKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalProcess, RenewalKernelFactory
from BehaviorScreen.point_process.survival_process import SurvivalProcess, SurvivalKernelFactory


def _assert_residuals_uniform(model, dataset: PointProcessDataset, min_p_value: float = 0.01):
    """
    Shared assertion: model.params_ must ALREADY be set to the TRUE
    generating parameters (no fit() call). fit_result is stubbed since
    time_rescaling doesn't need it, but some shared helper paths check for
    its existence.
    """
    model.fit_result = DummyFitResult()
    tr = model.time_rescaling(dataset)
    exact_residuals = tr["residuals"][~tr["censored"]]
    assert len(exact_residuals) > 100, "need enough residuals for the KS test to have power"

    u = 1.0 - np.exp(-exact_residuals)
    stat, p_value = kstest(u, "uniform")
    assert p_value > min_p_value, (
        f"time-rescaled residuals at TRUE parameters are not Uniform(0,1) "
        f"(KS stat={stat:.4f}, p={p_value:.4f}) -- simulate_stream and "
        f"cumulative_integrated_intensity/compensator machinery disagree "
        f"about the intensity they each implement for {model.name}."
    )


class TestPoissonSimulationCompensatorConsistency:

    def test_homogeneous(self, rng_factory):
        rng = rng_factory(200)
        model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        model.set_params(np.array([0.5]))

        scaffold = make_scaffold_dataset(num_fish=100, num_trials=6, duration_s=15.0)
        dataset = simulate_dataset_from_model(model, scaffold, rng)

        _assert_residuals_uniform(model, dataset)

    def test_shaped_kernel_omr_forward(self, rng_factory):
        """Checks a kernel with NO closed-form integral_func (falls back to
        trapezoid-on-a-grid in RateKernel.integrate) -- if that fallback
        grid construction were subtly biased, this would show up here as
        non-uniform residuals even though Layer 1 already checked
        integrate-vs-quad convergence for this same kernel."""
        rng = rng_factory(201)
        from BehaviorScreen.point_process.kernel_shapes import logit_bounded
        model = PoissonProcess(RateKernelFactory.omr_forward())
        z_dip = float(logit_bounded(0.6, 0.995))
        model.set_params(np.array([0.5, z_dip, 0.3]))

        scaffold = make_scaffold_dataset(num_fish=150, num_trials=10, duration_s=3.0)
        dataset = simulate_dataset_from_model(model, scaffold, rng)

        _assert_residuals_uniform(model, dataset)


@pytest.mark.slow
class TestHawkesSimulationCompensatorConsistency:

    def test_homogeneous_baseline_exponential_history(self, rng_factory):
        """
        This is the specific check discussed in conversation: simulate_stream
        uses history_kernel.evaluate/decay_envelope (pointwise, thinning
        proposal/acceptance), while cumulative_integrated_intensity uses
        history_kernel.integrate (closed-form exponential integral). A sign
        error or algebra mistake in HistoryKernelFactory.exponential's
        `_integral` that happened to be internally self-consistent with
        Layer 1's OWN quad check would still be caught here if it disagreed
        with what `_func` is ACTUALLY used for during simulation.
        """
        rng = rng_factory(202)
        model = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        model.set_params(np.array([0.4, 0.3, 2.0]))

        scaffold = make_scaffold_dataset(num_fish=80, num_trials=6, duration_s=20.0)
        dataset = simulate_dataset_from_model(model, scaffold, rng)
        assert len(dataset.event_times) > 2000

        _assert_residuals_uniform(model, dataset)


@pytest.mark.slow
class TestRenewalSimulationCompensatorConsistency:

    def test_homogeneous_baseline_exponential_excitation(self, rng_factory):
        rng = rng_factory(203)
        model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(), RenewalKernelFactory.exponential_excitation()
        )
        model.set_params(np.array([0.5, 1.0, 0.3]))

        scaffold = make_scaffold_dataset(num_fish=80, num_trials=6, duration_s=20.0)
        dataset = simulate_dataset_from_model(model, scaffold, rng)
        assert len(dataset.event_times) > 2000

        _assert_residuals_uniform(model, dataset)


class TestSurvivalSimulationCompensatorConsistency:

    def test_gaussian_bump_baseline(self, rng_factory):
        """
        SurvivalProcess produces AT MOST one residual per stream (its own
        stream_compensator_profile override), so this checks
        Cox-Snell-style calibration on a mostly-censored, mostly-single-
        event dataset -- a structurally different regime from the recurrent
        processes above (see stream_compensator_profile's docstring).
        """
        rng = rng_factory(204)
        model = SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=0.3, t_bounds=(0.05, 0.6)))
        model.set_params(np.array([3.0, 0.3, 0.08, 0.05]))

        scaffold = make_scaffold_dataset(num_fish=300, num_trials=10, duration_s=1.0)
        dataset = simulate_dataset_from_model(model, scaffold, rng)

        n_exact = sum(1 for _, _, t_ev in dataset.iter_streams() if len(t_ev) > 0)
        assert n_exact > 300

        _assert_residuals_uniform(model, dataset)