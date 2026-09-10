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
from BehaviorScreen.point_process.mixed_effects_process import (
    GammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.zero_inflated_mixed_effects_process import (
    ZeroInflatedGammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.baseline_only_frailty_hawkes import (
    BaselineOnlyFrailtyHawkesProcess,
)
from BehaviorScreen.point_process.zero_inflated_baseline_only_frailty_hawkes import (
    ZeroInflatedBaselineOnlyFrailtyHawkesProcess,
)
from BehaviorScreen.point_process.kernel_shapes import logit_bounded

def _km_exp1_sup_distance(
    model,
    residuals: np.ndarray,
    censored: np.ndarray,
    min_at_risk: int = 20,
) -> float:
    """
    Supremum distance between the censoring-aware KM residual CDF and the
    Exp(1) target CDF, evaluated at supported exact-event knots.
    """
    residual_grid, survival, n_at_risk = model._survival_estimate(
        residuals,
        censored,
        return_risk=True,
    )

    # Exclude artificial origin; retain stable exact-event knots.
    valid = (
        (np.arange(len(residual_grid)) > 0)
        & (n_at_risk >= min_at_risk)
        & (survival > 0.0)
    )

    assert np.any(valid), (
        "No supported exact-event knots for KM calibration. "
        "Increase the simulated sample size or reduce min_at_risk."
    )

    empirical_cdf = 1.0 - survival[valid]
    target_cdf = 1.0 - np.exp(-residual_grid[valid])

    return float(np.max(np.abs(empirical_cdf - target_cdf)))


def _assert_residuals_exp1_calibrated(
    model,
    dataset: PointProcessDataset,
    max_sup_distance: float = 0.06,
    min_at_risk: int = 20,
    min_exact: int = 100,
):
    """
    Censoring-aware known-parameter simulator/compensator consistency check.

    Unlike an ordinary KS test on exact residuals, this includes terminally
    censored and zero-event streams through the KM estimator.
    """
    model.fit_result = DummyFitResult()

    tr = model.time_rescaling(dataset)

    residuals = np.asarray(tr["residuals"], dtype=float)
    censored = np.asarray(tr["censored"], dtype=bool)

    assert len(residuals) > 0
    assert np.sum(~censored) >= min_exact
    assert np.sum(censored) > 0, (
        "Expected terminal/administrative censoring to be represented in "
        "the residual output."
    )

    distance = _km_exp1_sup_distance(
        model,
        residuals,
        censored,
        min_at_risk=min_at_risk,
    )

    assert distance <= max_sup_distance, (
        f"Known-parameter residual KM curve is not close to Exp(1): "
        f"D={distance:.4f} > {max_sup_distance:.4f}. "
        f"The simulator and compensator may disagree for {model.name}."
    )

class TestPoissonSimulationCompensatorConsistency:

    def test_homogeneous(self, rng_factory):
        rng = rng_factory(200)
        model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        model.set_params(np.array([0.5]))

        scaffold = make_scaffold_dataset(num_fish=100, num_trials=6, duration_s=15.0)
        dataset = simulate_dataset_from_model(model, scaffold, rng)

        _assert_residuals_exp1_calibrated(model, dataset)

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

        _assert_residuals_exp1_calibrated(model, dataset)


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

        _assert_residuals_exp1_calibrated(model, dataset)


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

        _assert_residuals_exp1_calibrated(model, dataset)


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

        _assert_residuals_exp1_calibrated(model, dataset)


@pytest.mark.slow
class TestGammaFrailtySimulationCompensatorConsistency:

    def test_homogeneous_poisson_base(self, rng_factory):
        rng = rng_factory(205)

        true_B = 0.5
        true_r = 3.0

        model = GammaMixedEffectsProcess(
            PoissonProcess(
                RateKernelFactory.homogeneous_poisson()
            ),
            r_init=5.0,
        )
        model.set_params(np.array([true_B, true_r]))

        scaffold = make_scaffold_dataset(
            num_fish=250,
            num_trials=6,
            duration_s=8.0,
        )

        gains = model._draw_fish_gains(
            scaffold.num_fish,
            1,
            rng,
        )[:, 0]

        dataset = simulate_dataset_from_model(
            model,
            scaffold,
            rng,
            fish_gains=gains,
        )

        _assert_residuals_exp1_calibrated(
            model,
            dataset,
            max_sup_distance=0.07,
            min_at_risk=20,
        )

@pytest.mark.slow
class TestZeroInflatedGammaFrailtySimulationCompensatorConsistency:

    def test_homogeneous_poisson_hard_nonresponders(self, rng_factory):
        rng = rng_factory(206)

        true_B = 0.5
        true_pi = 0.25
        true_r = 4.0

        model = ZeroInflatedGammaMixedEffectsProcess(
            PoissonProcess(
                RateKernelFactory.homogeneous_poisson()
            ),
            pi_init=true_pi,
            r_init=true_r,
            fit_c=False,
        )

        z_pi = float(logit_bounded(true_pi, 1.0))
        model.set_params(
            np.array([true_B, z_pi, true_r])
        )

        scaffold = make_scaffold_dataset(
            num_fish=300,
            num_trials=6,
            duration_s=8.0,
        )

        gains = model._draw_fish_gains(
            scaffold.num_fish,
            1,
            rng,
        )[:, 0]

        dataset = simulate_dataset_from_model(
            model,
            scaffold,
            rng,
            fish_gains=gains,
        )

        assert np.sum(gains == 0.0) > 30

        _assert_residuals_exp1_calibrated(
            model,
            dataset,
            max_sup_distance=0.08,
            min_at_risk=20,
        )

@pytest.mark.slow
class TestBaselineOnlyFrailtyHawkesCompensatorConsistency:

    def test_homogeneous_baseline_exponential_history(self, rng_factory):
        rng = rng_factory(207)

        true_B = 0.4
        true_alpha = 0.3
        true_beta = 2.0
        true_r = 4.0

        model = BaselineOnlyFrailtyHawkesProcess(
            HawkesProcess(
                RateKernelFactory.homogeneous_poisson(),
                HistoryKernelFactory.exponential(),
            ),
            r_init=true_r,
            n_quad_nodes=30,
        )
        model.set_params(
            np.array([
                true_B,
                true_alpha,
                true_beta,
                true_r,
            ])
        )

        scaffold = make_scaffold_dataset(
            num_fish=150,
            num_trials=6,
            duration_s=10.0,
        )

        gains = model._draw_fish_gains(
            scaffold.num_fish,
            1,
            rng,
        )[:, 0]

        dataset = simulate_dataset_from_model(
            model,
            scaffold,
            rng,
            fish_gains=gains,
        )

        _assert_residuals_exp1_calibrated(
            model,
            dataset,
            max_sup_distance=0.08,
            min_at_risk=20,
        )

@pytest.mark.slow
class TestZeroInflatedBaselineOnlyFrailtyHawkesCompensatorConsistency:

    def test_hard_nonresponders(self, rng_factory):
        rng = rng_factory(208)

        true_B = 0.4
        true_alpha = 0.3
        true_beta = 2.0
        true_pi = 0.25
        true_r = 4.0

        model = ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
            HawkesProcess(
                RateKernelFactory.homogeneous_poisson(),
                HistoryKernelFactory.exponential(),
            ),
            pi_init=true_pi,
            r_init=true_r,
            fit_c=False,
            n_quad_nodes=30,
        )

        z_pi = float(logit_bounded(true_pi, 1.0))
        model.set_params(
            np.array([
                true_B,
                true_alpha,
                true_beta,
                z_pi,
                true_r,
            ])
        )

        scaffold = make_scaffold_dataset(
            num_fish=200,
            num_trials=6,
            duration_s=10.0,
        )

        gains = model._draw_fish_gains(
            scaffold.num_fish,
            1,
            rng,
        )[:, 0]

        dataset = simulate_dataset_from_model(
            model,
            scaffold,
            rng,
            fish_gains=gains,
        )

        _assert_residuals_exp1_calibrated(
            model,
            dataset,
            max_sup_distance=0.09,
            min_at_risk=20,
        )