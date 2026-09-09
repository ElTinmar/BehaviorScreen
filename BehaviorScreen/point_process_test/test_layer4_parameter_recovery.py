# tests/test_layer4_parameter_recovery.py
"""
LAYER 4: end-to-end parameter recovery.

Simulates data using each model's OWN simulate_stream/_draw_fish_gains,
refits from scratch, and checks fitted parameters land close to ground
truth. This is a REGRESSION/SMOKE test layer, not a likelihood-correctness
test layer -- by the time you're here, Layers 0-3 should already have
validated that the likelihood formulas and simulate/compensator code paths
are internally consistent and individually correct. This layer's job is to
catch optimizer robustness issues, parametrization/plumbing bugs (e.g. a
_split_params index off-by-one), and multistart behavior -- NOT to be your
first line of defense against a shared simulate/fit bug (see
test_layer2_independent_simulator.py and test_layer3_simulation_compensator
_consistency.py for that).

TOLERANCES are asymmetric by design:
  - rate/shape kernel params (B, tau, mu, sigma, alpha_hawkes, beta_hawkes):
    tight-ish relative tolerance -- well-identified with enough exposure.
  - r_dispersion, pi_nonresponder: much looser -- Gamma-shape and mixture-
    weight MLEs have high finite-sample variance (see conversation notes on
    frailty identifiability). A tight tolerance here would make the suite
    flaky for the wrong reason.

Fixed seeds -> deterministic regression tests, not formal hypothesis tests.
If you change a dataset size/tolerance, rerun with a few different seeds
manually before trusting a single fixed-seed pass/fail.
"""
import numpy as np
import pytest

from .conftest import make_scaffold_dataset, simulate_dataset_from_model, assert_recovered, params_dict

from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory
from BehaviorScreen.point_process.hawkes_process import HawkesProcess, HistoryKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalProcess, RenewalKernelFactory
from BehaviorScreen.point_process.survival_process import SurvivalProcess, SurvivalKernelFactory
from BehaviorScreen.point_process.mixed_effects_process import GammaMixedEffectsProcess
from BehaviorScreen.point_process.zero_inflated_mixed_effects_process import (
    ZeroInflatedGammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.baseline_only_frailty_hawkes import (
    BaselineOnlyFrailtyHawkesProcess,
)
from BehaviorScreen.point_process.zero_inflated_baseline_only_frailty_hawkes import (
    ZeroInflatedBaselineOnlyFrailtyHawkesProcess,
)
from BehaviorScreen.point_process.partially_fixed_process import PartiallyFixedProcess
from BehaviorScreen.point_process.kernel_shapes import logit_bounded, sigmoid_bounded


# =============================================================================
# PoissonProcess
# =============================================================================

class TestPoissonProcessRecovery:

    def test_homogeneous_rate(self, rng_factory):
        rng = rng_factory(0)
        true_B = 0.8

        scaffold = make_scaffold_dataset(num_fish=40, num_trials=5, duration_s=20.0)
        model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        model.set_params(np.array([true_B]))

        dataset = simulate_dataset_from_model(model, scaffold, rng)
        assert len(dataset.event_times) > 1000

        fit_model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        fit_model.fit(dataset)

        assert_recovered("B", fit_model.param_dict_["B"], true_B, rtol=0.08)

    def test_shaped_kernel_omr_forward(self, rng_factory):
        rng = rng_factory(1)
        true_B, true_f_dip, true_tau_dip = 0.6, 0.7, 0.4
        true_z_dip = float(logit_bounded(true_f_dip, 0.995))

        scaffold = make_scaffold_dataset(num_fish=80, num_trials=10, duration_s=5.0)
        model = PoissonProcess(RateKernelFactory.omr_forward())
        model.set_params(np.array([true_B, true_z_dip, true_tau_dip]))

        dataset = simulate_dataset_from_model(model, scaffold, rng)

        fit_model = PoissonProcess(RateKernelFactory.omr_forward())
        fit_model.fit(dataset)

        pd = params_dict(fit_model)
        fitted_f_dip = float(sigmoid_bounded(pd["z_dip"], 0.995))

        assert_recovered("B", pd["B"], true_B, rtol=0.10)
        assert_recovered("f_dip (derived)", fitted_f_dip, true_f_dip, rtol=0.20, atol=0.03)
        assert_recovered("tau_dip", pd["tau_dip"], true_tau_dip, rtol=0.20)


# =============================================================================
# HawkesProcess
# =============================================================================

@pytest.mark.slow
class TestHawkesProcessRecovery:

    def test_homogeneous_baseline_exponential_history(self, rng_factory):
        """
        Deliberately a FLAT baseline kernel -- no competing deterministic
        ramp, i.e. the unconfounded regime from the identifiability
        discussion. If this fails, suspect the optimizer/multistart
        machinery, not a fundamental identifiability limit.
        """
        rng = rng_factory(2)
        true_B, true_alpha, true_beta = 0.4, 0.3, 2.0  # branching ratio 0.15, subcritical

        scaffold = make_scaffold_dataset(num_fish=50, num_trials=6, duration_s=20.0)
        base = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        base.set_params(np.array([true_B, true_alpha, true_beta]))

        dataset = simulate_dataset_from_model(base, scaffold, rng)
        assert len(dataset.event_times) > 2000

        fit_model = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        fit_model.fit_multistart(dataset, n_starts=8, n_jobs=1, seed=42)

        pd = params_dict(fit_model)
        assert_recovered("B", pd["B"], true_B, rtol=0.12)
        assert_recovered("alpha_hawkes", pd["alpha_hawkes"], true_alpha, rtol=0.30)
        assert_recovered("beta_hawkes", pd["beta_hawkes"], true_beta, rtol=0.30)

        true_branching = true_alpha / true_beta
        fitted_branching = pd["alpha_hawkes"] / pd["beta_hawkes"]
        assert_recovered("branching ratio (alpha/beta)", fitted_branching, true_branching, rtol=0.20)


# =============================================================================
# RenewalProcess
# =============================================================================

@pytest.mark.slow
class TestRenewalProcessRecovery:

    def test_homogeneous_baseline_exponential_excitation(self, rng_factory):
        rng = rng_factory(3)
        true_B, true_A_exc, true_tau_exc = 0.5, 1.0, 0.3

        scaffold = make_scaffold_dataset(num_fish=50, num_trials=6, duration_s=20.0)
        base = RenewalProcess(RateKernelFactory.homogeneous_poisson(), RenewalKernelFactory.exponential_excitation())
        base.set_params(np.array([true_B, true_A_exc, true_tau_exc]))

        dataset = simulate_dataset_from_model(base, scaffold, rng)
        assert len(dataset.event_times) > 2000

        fit_model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(), RenewalKernelFactory.exponential_excitation()
        )
        fit_model.fit_multistart(dataset, n_starts=8, n_jobs=1, seed=43)

        pd = params_dict(fit_model)
        assert_recovered("B", pd["B"], true_B, rtol=0.12)
        assert_recovered("A_excitation", pd["A_excitation"], true_A_exc, rtol=0.30)
        assert_recovered("tau_excitation", pd["tau_excitation"], true_tau_exc, rtol=0.30)


# =============================================================================
# SurvivalProcess
# =============================================================================

@pytest.mark.slow
class TestSurvivalProcessRecovery:

    def test_gaussian_bump_baseline(self, rng_factory):
        """Survival models discard everything but each stream's first event,
        so this needs MANY (fish, trial) streams -- not long individual
        trials -- to get enough exact (uncensored) observations."""
        rng = rng_factory(4)
        true_H, true_mu, true_sigma, true_B = 3.0, 0.3, 0.08, 0.05

        scaffold = make_scaffold_dataset(num_fish=150, num_trials=10, duration_s=1.0)
        model = SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=0.3, t_bounds=(0.05, 0.6)))
        model.set_params(np.array([true_H, true_mu, true_sigma, true_B]))

        dataset = simulate_dataset_from_model(model, scaffold, rng)
        n_exact = sum(1 for _, _, t_ev in dataset.iter_streams() if len(t_ev) > 0)
        assert n_exact > 300, f"expected enough responders for recovery, got {n_exact}"

        fit_model = SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=0.3, t_bounds=(0.05, 0.6)))
        fit_model.fit(dataset)

        pd = params_dict(fit_model)
        assert_recovered("H", pd["H"], true_H, rtol=0.25)
        assert_recovered("mu", pd["mu"], true_mu, rtol=0.15)
        assert_recovered("sigma", pd["sigma"], true_sigma, rtol=0.30)
        assert_recovered("B", pd["B"], true_B, rtol=0.40, atol=0.02)


# =============================================================================
# GammaMixedEffectsProcess
# =============================================================================

@pytest.mark.slow
class TestGammaMixedEffectsProcessRecovery:

    def test_poisson_homogeneous_base(self, rng_factory):
        rng = rng_factory(5)
        true_B, true_r = 0.6, 3.0

        scaffold = make_scaffold_dataset(num_fish=150, num_trials=6, duration_s=15.0)
        model = GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson()), r_init=5.0)
        model.set_params(np.array([true_B, true_r]))

        gains = model._draw_fish_gains(scaffold.num_fish, 1, rng)[:, 0]
        dataset = simulate_dataset_from_model(model, scaffold, rng, fish_gains=gains)

        fit_model = GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson()), r_init=5.0)
        fit_model.fit(dataset)

        assert_recovered("B", fit_model.base_process.param_dict_["B"], true_B, rtol=0.12)
        assert_recovered("r_dispersion", fit_model.dispersion_r, true_r, rtol=0.40)


# =============================================================================
# ZeroInflatedGammaMixedEffectsProcess
# =============================================================================

@pytest.mark.slow
class TestZeroInflatedGammaMixedEffectsProcessRecovery:

    def test_poisson_homogeneous_base_hard_nonresponders(self, rng_factory):
        """pi/r tolerances are the loosest in this suite -- mixture-weight
        recovery needs a LOT of fish to be reliable (Self & Liang
        boundary-of-parameter-space concern near pi=0)."""
        rng = rng_factory(6)
        true_B, true_pi, true_r = 0.6, 0.3, 4.0  # fit_c=False -> c=0.0 fixed

        scaffold = make_scaffold_dataset(num_fish=250, num_trials=6, duration_s=15.0)
        model = ZeroInflatedGammaMixedEffectsProcess(
            PoissonProcess(RateKernelFactory.homogeneous_poisson()), pi_init=0.3, r_init=5.0, fit_c=False,
        )
        z_pi_true = float(logit_bounded(true_pi, 1.0))
        model.set_params(np.array([true_B, z_pi_true, true_r]))

        gains = model._draw_fish_gains(scaffold.num_fish, 1, rng)[:, 0]
        dataset = simulate_dataset_from_model(model, scaffold, rng, fish_gains=gains)

        n_nonresponders = np.sum(np.isclose(gains, 0.0))
        assert n_nonresponders > 30, "sanity check: enough true non-responders simulated"

        fit_model = ZeroInflatedGammaMixedEffectsProcess(
            PoissonProcess(RateKernelFactory.homogeneous_poisson()), pi_init=0.3, r_init=5.0, fit_c=False,
        )
        fit_model.fit_multistart(dataset, n_starts=6, n_jobs=1, seed=44)

        assert_recovered("B", fit_model.base_process.param_dict_["B"], true_B, rtol=0.12)
        assert_recovered("pi_nonresponder", fit_model.pi_nonresponder, true_pi, rtol=0.35, atol=0.08)
        assert_recovered("r_responder", fit_model.r_responder, true_r, rtol=0.45)
        assert fit_model.c_nonresponder == 0.0


# =============================================================================
# BaselineOnlyFrailtyHawkesProcess
# =============================================================================

@pytest.mark.slow
class TestBaselineOnlyFrailtyHawkesProcessRecovery:

    def test_homogeneous_baseline_exponential_history(self, rng_factory):
        """
        NOTE: BaselineOnlyFrailtyHawkesProcess does NOT override
        _draw_fish_gains -- gains are drawn manually here. n_quad_nodes is
        reduced from the default (30) to keep runtime reasonable.
        """
        rng = rng_factory(7)
        true_B, true_alpha, true_beta, true_r = 0.4, 0.3, 2.0, 4.0

        scaffold = make_scaffold_dataset(num_fish=60, num_trials=5, duration_s=12.0)
        base = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        model = BaselineOnlyFrailtyHawkesProcess(base, r_init=5.0, n_quad_nodes=15)
        model.set_params(np.array([true_B, true_alpha, true_beta, true_r]))

        gains = rng.gamma(shape=true_r, scale=1.0 / true_r, size=scaffold.num_fish)
        dataset = simulate_dataset_from_model(model, scaffold, rng, fish_gains=gains)

        fit_base = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        fit_model = BaselineOnlyFrailtyHawkesProcess(fit_base, r_init=5.0, n_quad_nodes=15)
        fit_model.fit_multistart(dataset, n_starts=6, n_jobs=1, seed=45)

        base_pd = params_dict(fit_model.base_process)
        assert_recovered("B", base_pd["B"], true_B, rtol=0.15)
        assert_recovered("alpha_hawkes", base_pd["alpha_hawkes"], true_alpha, rtol=0.35)
        assert_recovered("beta_hawkes", base_pd["beta_hawkes"], true_beta, rtol=0.35)
        assert_recovered("r_dispersion", fit_model.dispersion_r, true_r, rtol=0.45)


# =============================================================================
# ZeroInflatedBaselineOnlyFrailtyHawkesProcess
# =============================================================================

@pytest.mark.slow
class TestZeroInflatedBaselineOnlyFrailtyHawkesProcessRecovery:

    def test_homogeneous_baseline_exponential_history_hard_nonresponders(self, rng_factory):
        rng = rng_factory(8)
        true_B, true_alpha, true_beta = 0.4, 0.3, 2.0
        true_pi, true_r = 0.3, 4.0

        scaffold = make_scaffold_dataset(num_fish=100, num_trials=5, duration_s=12.0)
        base = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        model = ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
            base, pi_init=0.3, r_init=5.0, fit_c=False, n_quad_nodes=15,
        )
        z_pi_true = float(logit_bounded(true_pi, 1.0))
        model.set_params(np.array([true_B, true_alpha, true_beta, z_pi_true, true_r]))

        gains = model._draw_fish_gains(scaffold.num_fish, 1, rng)[:, 0]
        dataset = simulate_dataset_from_model(model, scaffold, rng, fish_gains=gains)

        fit_base = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        fit_model = ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
            fit_base, pi_init=0.3, r_init=5.0, fit_c=False, n_quad_nodes=15,
        )
        fit_model.fit_multistart(dataset, n_starts=6, n_jobs=1, seed=46)

        base_pd = params_dict(fit_model.base_process)
        assert_recovered("B", base_pd["B"], true_B, rtol=0.18)
        assert_recovered("alpha_hawkes", base_pd["alpha_hawkes"], true_alpha, rtol=0.40)
        assert_recovered("beta_hawkes", base_pd["beta_hawkes"], true_beta, rtol=0.40)
        assert_recovered("pi_nonresponder", fit_model.pi_nonresponder, true_pi, rtol=0.40, atol=0.10)
        assert_recovered("r_responder", fit_model.r_responder, true_r, rtol=0.50)

    def test_pi_zero_collapses_to_baseline_only_frailty(self, rng_factory):
        """
        Structural consistency check (not a recovery check): pi -> 0 should
        make this class's likelihood numerically match
        BaselineOnlyFrailtyHawkesProcess's own, on the SAME data/params.
        Catches a broken mixture formula even if individual parameter
        recovery tolerances happen to be loose enough to hide it.
        """
        rng = rng_factory(9)
        true_B, true_alpha, true_beta, true_r = 0.4, 0.3, 2.0, 4.0

        scaffold = make_scaffold_dataset(num_fish=30, num_trials=4, duration_s=10.0)
        base_gen = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
        gen_model = BaselineOnlyFrailtyHawkesProcess(base_gen, r_init=5.0, n_quad_nodes=15)
        gen_model.set_params(np.array([true_B, true_alpha, true_beta, true_r]))
        gains = rng.gamma(shape=true_r, scale=1.0 / true_r, size=scaffold.num_fish)
        dataset = simulate_dataset_from_model(gen_model, scaffold, rng, fish_gains=gains)

        plain = BaselineOnlyFrailtyHawkesProcess(
            HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential()),
            n_quad_nodes=15,
        )
        plain.set_params(np.array([true_B, true_alpha, true_beta, true_r]))
        plain_nll = plain._nll(list(plain.params_), dataset)

        zi = ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
            HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential()),
            fit_c=False, n_quad_nodes=15,
        )
        z_pi_zero = float(logit_bounded(1e-4, 1.0))
        zi_params = [true_B, true_alpha, true_beta, z_pi_zero, true_r]
        zi_nll = zi._nll(zi_params, dataset)

        assert abs(plain_nll - zi_nll) < 0.05 * abs(plain_nll), (
            f"pi->0 should reproduce BaselineOnlyFrailtyHawkesProcess's NLL: "
            f"plain={plain_nll:.3f}, zi(pi~0)={zi_nll:.3f}"
        )


# =============================================================================
# PartiallyFixedProcess
# =============================================================================

class TestPartiallyFixedProcessRecovery:

    def test_fixing_a_true_parameter_recovers_the_rest(self, rng_factory):
        """Pin tau_dip at its TRUE value and check B, z_dip are still
        recoverable -- exercises the expand/contract parameter-vector
        plumbing, not the underlying kernel's own identifiability."""
        rng = rng_factory(10)
        true_B, true_f_dip, true_tau_dip = 0.6, 0.7, 0.4
        true_z_dip = float(logit_bounded(true_f_dip, 0.995))

        scaffold = make_scaffold_dataset(num_fish=80, num_trials=10, duration_s=5.0)
        gen_model = PoissonProcess(RateKernelFactory.omr_forward())
        gen_model.set_params(np.array([true_B, true_z_dip, true_tau_dip]))
        dataset = simulate_dataset_from_model(gen_model, scaffold, rng)

        fixed_model = PartiallyFixedProcess(
            PoissonProcess(RateKernelFactory.omr_forward()),
            fixed_values={"tau_dip": true_tau_dip},
        )
        fixed_model.fit(dataset)

        pd = params_dict(fixed_model)
        fitted_f_dip = float(sigmoid_bounded(pd["z_dip"], 0.995))

        assert_recovered("B", pd["B"], true_B, rtol=0.10)
        assert_recovered("f_dip (derived)", fitted_f_dip, true_f_dip, rtol=0.20, atol=0.03)
        assert fixed_model.base_process.param_dict_["tau_dip"] == true_tau_dip