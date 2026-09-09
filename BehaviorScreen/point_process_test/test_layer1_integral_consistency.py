# tests/test_layer1_integral_consistency.py
"""
LAYER 1: closed-form integral vs. numerical integration of the SAME
pointwise function.

Every RateKernel/HistoryKernel that ships a hand-coded `integral_func`
alongside its `func` is implicitly claiming the two describe the same
mathematical object. This layer checks that claim directly, by numerically
integrating `func` with an INDEPENDENT method (scipy.integrate.quad) and
comparing to `integral_func`'s output.

This is exactly the check that would catch a sign error or algebra mistake
in a closed-form integral -- the kind of bug that a simulate-then-fit
recovery test CANNOT see, because simulate_stream and _nll's compensator
would consistently use the same (wrong) integral together (see conversation
notes on shared-code-path risk).
"""
import numpy as np
import pytest
from scipy import integrate as scipy_integrate

from BehaviorScreen.point_process.poisson_process import RateKernelFactory
from BehaviorScreen.point_process.hawkes_process import HistoryKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalKernelFactory


class TestHistoryKernelIntegralConsistency:

    @pytest.mark.parametrize("alpha,beta,T", [
        (0.5, 2.0, 3.0),
        (1.2, 0.3, 10.0),
        (0.05, 5.0, 1.0),
        (3.0, 0.05, 0.5),  # slow decay relative to T -- stresses the closed form differently
    ])
    def test_exponential_history_kernel(self, alpha, beta, T):
        hk = HistoryKernelFactory.exponential()
        closed_form = hk.integrate(T, [alpha, beta])
        quad_result, _ = scipy_integrate.quad(
            lambda lag: hk.evaluate(np.array([lag]), [alpha, beta])[0], 0, T
        )
        assert closed_form == pytest.approx(quad_result, rel=1e-6)

    @pytest.mark.parametrize("T", [0.001, 1.0, 50.0])
    def test_exponential_history_kernel_boundary_durations(self, T):
        """Very short and very long durations -- catches edge-case bugs in
        the closed form (e.g. an unhandled T->0 or T->inf limit)."""
        alpha, beta = 0.8, 1.5
        hk = HistoryKernelFactory.exponential()
        closed_form = hk.integrate(T, [alpha, beta])
        quad_result, _ = scipy_integrate.quad(
            lambda lag: hk.evaluate(np.array([lag]), [alpha, beta])[0], 0, T
        )
        assert closed_form == pytest.approx(quad_result, rel=1e-5, abs=1e-8)

    def test_exponential_history_kernel_event_history_matches_generic_oN2(self):
        """
        HistoryKernelFactory.exponential() ships a fast O(N) recursive
        event_history_func -- this checks it agrees with HistoryKernel's
        GENERIC O(N^2) fallback (evaluate() summed pairwise), by
        constructing a second HistoryKernel with the same `func` but no
        `event_history_func` override, forcing the generic path.
        """
        from BehaviorScreen.point_process.hawkes_process import HistoryKernel

        alpha, beta = 0.6, 1.3
        t_events = np.array([0.1, 0.35, 0.4, 1.2, 1.25, 3.0])

        fast_hk = HistoryKernelFactory.exponential()
        fast_result = fast_hk.event_history(t_events, [alpha, beta])

        generic_hk = HistoryKernel(
            name="GenericFallback",
            func=fast_hk.func,
            param_names=fast_hk.param_names,
            initial_guesses=fast_hk.initial_guesses,
            bounds=fast_hk.bounds,
            integral_func=None,        # force generic integrate path too (unused here)
            event_history_func=None,   # force the O(N^2) generic event_history path
        )
        generic_result = generic_hk.event_history(t_events, [alpha, beta])

        np.testing.assert_allclose(fast_result, generic_result, rtol=1e-9)


class TestRateKernelIntegralConsistency:

    def test_homogeneous_poisson(self):
        rk = RateKernelFactory.homogeneous_poisson()
        B, T, trial = 0.7, 8.0, 3
        numeric = rk.integrate(T, trial, [B], integration_dt=0.001)
        quad_result, _ = scipy_integrate.quad(
            lambda t: rk.evaluate(np.array([t]), np.array([trial]), [B])[0], 0, T
        )
        assert numeric == pytest.approx(quad_result, rel=1e-4)

    def test_omr_forward_no_closed_form_integral_grid_convergence(self):
        """
        RateKernelFactory.omr_forward() has NO integral_func, so
        RateKernel.integrate falls back to trapezoid-on-a-grid. This checks
        that fallback CONVERGES to the quad reference as integration_dt
        shrinks -- catches a bug where the fallback grid construction
        (t_grid endpoint handling in RateKernel.integrate) is systematically
        biased rather than just discretization-noisy.
        """
        rk = RateKernelFactory.omr_forward()
        from BehaviorScreen.point_process.kernel_shapes import logit_bounded
        B, z_dip, tau_dip = 0.5, float(logit_bounded(0.6, 0.995)), 0.3
        T, trial = 2.0, 0

        quad_result, _ = scipy_integrate.quad(
            lambda t: rk.evaluate(np.array([t]), np.array([trial]), [B, z_dip, tau_dip])[0], 0, T
        )

        errors = []
        for dt in [0.05, 0.01, 0.002]:
            numeric = rk.integrate(T, trial, [B, z_dip, tau_dip], integration_dt=dt)
            errors.append(abs(numeric - quad_result))

        assert errors[-1] < errors[0], "finer grid should be at least as accurate"
        assert errors[-1] < 1e-4, f"finest grid still off from quad reference: {errors[-1]}"

    def test_cumulative_integrate_matches_integrate_at_endpoint(self):
        """cumulative_integrate(t_events, ...)'s LAST value (if t_events'
        max equals T) should match integrate(T, ...) -- both are supposed to
        compute the same total integral, via different code paths
        (cumulative_trapezoid + interp vs. plain trapezoid)."""
        rk = RateKernelFactory.omr_forward()
        from BehaviorScreen.point_process.kernel_shapes import logit_bounded
        params = [0.5, float(logit_bounded(0.6, 0.995)), 0.3]
        T, trial = 3.0, 0

        total_direct = rk.integrate(T, trial, params, integration_dt=0.002)
        cumulative = rk.cumulative_integrate(np.array([T]), trial, params, integration_dt=0.002)

        assert cumulative[-1] == pytest.approx(total_direct, rel=1e-3)


class TestRenewalKernelIntegralConsistency:

    def test_exponential_recovery(self):
        rk = RenewalKernelFactory.exponential_recovery()
        tau_r, T = 0.15, 2.0
        numeric = rk.integrate(T, [tau_r])
        quad_result, _ = scipy_integrate.quad(lambda lag: rk.evaluate(np.array([lag]), [tau_r])[0], 0, T)
        assert numeric == pytest.approx(quad_result, rel=1e-3)

    def test_hard_absorption_integral_is_identically_zero(self):
        rk = RenewalKernelFactory.hard_absorption()
        result = rk.integrate(np.array([0.0, 1.0, 100.0]), [])
        np.testing.assert_allclose(result, 0.0)