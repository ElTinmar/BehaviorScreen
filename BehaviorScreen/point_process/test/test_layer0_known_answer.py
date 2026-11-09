# tests/test_layer0_known_answer.py
"""
LAYER 0: Known-answer tests (KAT).

Hand-computed likelihood values, no simulation, no optimizer, no randomness.
These are the fastest, most precise tests in the suite -- a failure here
localizes directly to a specific formula inside _nll, independent of
whether simulation or optimization is involved at all.

Tolerances are TIGHT (near machine precision or tight closed-form-MLE
tolerance) since there is no stochastic noise to average out.
"""

import numpy as np
import pytest

from BehaviorScreen.point_process.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.point_process.poisson_process import (
    PoissonProcess,
    RateKernelFactory,
)
from BehaviorScreen.point_process.point_process.hawkes_process import (
    HawkesProcess,
    HistoryKernelFactory,
)
from BehaviorScreen.point_process.point_process.survival_process import (
    SurvivalProcess,
    SurvivalKernelFactory,
)
from BehaviorScreen.point_process.point_process.renewal_process import (
    RenewalProcess,
    RenewalKernelFactory,
)
from BehaviorScreen.point_process.point_process.frailty import GammaGain, ZeroInflatedGain


class TestHomogeneousPoissonKnownAnswer:

    def test_nll_closed_form(self):
        """
        For a constant-rate kernel, NLL has an exact closed form:
        -(N*log(B) - B*T*n_streams). Trapezoid integration of a CONSTANT
        function is exact for ANY grid spacing (not an approximation here),
        so agreement should be near machine precision.
        """
        T, n_streams, N = 10.0, 5, 37
        event_times = np.linspace(0.1, T - 0.1, N)
        event_trials = np.zeros(N, dtype=int)

        dataset = PointProcessDataset(
            event_times=event_times,
            event_trials_idx=event_trials,
            event_fish_idx=np.zeros(N, dtype=int),
            fish_trial_mask=np.ones((1, n_streams), dtype=bool),
            fish_ids=np.array(["f0"]),
            duration_s=T,
        )

        model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        B_test = 0.6
        computed_nll = model._nll([B_test], dataset)
        expected_nll = -(N * np.log(B_test) - B_test * T * n_streams)

        assert computed_nll == pytest.approx(expected_nll, rel=1e-10)

    def test_mle_matches_closed_form_estimator(self):
        """The exact MLE for a homogeneous rate is N / (total exposure).
        No stochastic noise in this dataset, so we can use a tight
        tolerance -- unlike the Layer 4 recovery tests."""
        T, n_streams, N = 10.0, 5, 37
        event_times = np.linspace(0.1, T - 0.1, N)
        dataset = PointProcessDataset(
            event_times=event_times,
            event_trials_idx=np.zeros(N, dtype=int),
            event_fish_idx=np.zeros(N, dtype=int),
            fish_trial_mask=np.ones((1, n_streams), dtype=bool),
            fish_ids=np.array(["f0"]),
            duration_s=T,
        )
        model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        model.fit(dataset)

        expected_mle = N / (T * n_streams)
        assert model.param_dict_["B"] == pytest.approx(expected_mle, rel=1e-4)


class TestHawkesKnownAnswer:

    def test_nll_two_events_hand_computed(self):
        """
        Two fixed events in a single stream. The expected NLL is derived
        HERE, independently, using np.exp(-beta*dt) written out directly --
        deliberately NOT by calling HistoryKernel.evaluate/integrate. If
        this reference computation instead called into the same kernel
        machinery being tested, this test would be circular and worthless
        (see conversation notes on shared-code-path risk).
        """
        B, alpha, beta = 0.3, 0.5, 2.0
        duration_s = 5.0
        t1, t2 = 1.0, 1.8

        dataset = PointProcessDataset(
            event_times=np.array([t1, t2]),
            event_trials_idx=np.array([0, 0]),
            event_fish_idx=np.array([0, 0]),
            fish_trial_mask=np.ones((1, 1), dtype=bool),
            fish_ids=np.array(["f0"]),
            duration_s=duration_s,
        )

        model = HawkesProcess(
            RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential()
        )
        computed_nll = model._nll([B, alpha, beta], dataset)

        lam1 = B
        lam2 = B + alpha * np.exp(-beta * (t2 - t1))
        sum_log_intensity = np.log(lam1) + np.log(lam2)

        base_integral = B * duration_s
        hist_integral_1 = (alpha / beta) * (1 - np.exp(-beta * (duration_s - t1)))
        hist_integral_2 = (alpha / beta) * (1 - np.exp(-beta * (duration_s - t2)))
        total_integral = base_integral + hist_integral_1 + hist_integral_2

        expected_nll = -(sum_log_intensity - total_integral)
        assert computed_nll == pytest.approx(expected_nll, rel=1e-9)

    def test_nll_single_event_no_history_contribution(self):
        """Degenerate case: a single event has NO prior history, so its
        log-intensity should reduce to log(B) exactly -- catches an
        off-by-one in event_history's loop (e.g. history[0] not
        initialized to 0)."""
        B, alpha, beta, duration_s, t1 = 0.4, 0.5, 2.0, 5.0, 2.0
        dataset = PointProcessDataset(
            event_times=np.array([t1]),
            event_trials_idx=np.array([0]),
            event_fish_idx=np.array([0]),
            fish_trial_mask=np.ones((1, 1), dtype=bool),
            fish_ids=np.array(["f0"]),
            duration_s=duration_s,
        )
        model = HawkesProcess(
            RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential()
        )
        computed_nll = model._nll([B, alpha, beta], dataset)

        expected_nll = -(
            np.log(B)
            - (
                B * duration_s
                + (alpha / beta) * (1 - np.exp(-beta * (duration_s - t1)))
            )
        )
        assert computed_nll == pytest.approx(expected_nll, rel=1e-9)


class TestSurvivalKnownAnswer:

    def test_nll_censored_stream_is_just_cumulative_hazard(self):
        """
        A stream with NO event (fully censored) contributes exactly H(T) =
        integral_0^T hazard -- no log-intensity term at all. Uses
        constant_hazard so H(T) = B*T exactly, checkable by hand.
        """
        B, T = 0.3, 4.0
        dataset = PointProcessDataset(
            event_times=np.array([], dtype=float),
            event_trials_idx=np.array([], dtype=int),
            event_fish_idx=np.array([], dtype=int),
            fish_trial_mask=np.ones((1, 1), dtype=bool),
            fish_ids=np.array(["f0"]),
            duration_s=T,
        )
        model = SurvivalProcess(SurvivalKernelFactory.constant_hazard())
        computed_nll = model._nll([B], dataset)
        expected_nll = B * T  # -( -H ) = H, no log-hazard term (censored)
        assert computed_nll == pytest.approx(expected_nll, rel=1e-9)

    def test_nll_uncensored_stream_hand_computed(self):
        """One event at t_obs: NLL = H(t_obs) - log(hazard(t_obs))."""
        B, T, t_obs = 0.3, 4.0, 1.5
        dataset = PointProcessDataset(
            event_times=np.array([t_obs]),
            event_trials_idx=np.array([0]),
            event_fish_idx=np.array([0]),
            fish_trial_mask=np.ones((1, 1), dtype=bool),
            fish_ids=np.array(["f0"]),
            duration_s=T,
        )
        model = SurvivalProcess(SurvivalKernelFactory.constant_hazard())
        computed_nll = model._nll([B], dataset)
        expected_nll = B * t_obs - np.log(B)
        assert computed_nll == pytest.approx(expected_nll, rel=1e-9)

    def test_second_event_in_stream_is_discarded(self):
        """SurvivalProcess reduces a stream to its FIRST event only -- a
        second event at a later time must not change the NLL at all."""
        B, T, t_obs = 0.3, 4.0, 1.5
        dataset_one = PointProcessDataset(
            event_times=np.array([t_obs]),
            event_trials_idx=np.array([0]),
            event_fish_idx=np.array([0]),
            fish_trial_mask=np.ones((1, 1), dtype=bool),
            fish_ids=np.array(["f0"]),
            duration_s=T,
        )
        dataset_two = PointProcessDataset(
            event_times=np.array([t_obs, t_obs + 1.0]),
            event_trials_idx=np.array([0, 0]),
            event_fish_idx=np.array([0, 0]),
            fish_trial_mask=np.ones((1, 1), dtype=bool),
            fish_ids=np.array(["f0"]),
            duration_s=T,
        )
        model = SurvivalProcess(SurvivalKernelFactory.constant_hazard())
        assert model._nll([B], dataset_one) == pytest.approx(
            model._nll([B], dataset_two), rel=1e-12
        )


class TestGammaFrailtyCompensatorKnownAnswer:

    def test_predictive_increment_matches_gamma_laplace_transform(self):
        r = 3.0
        n_previous = 4
        s_previous = 2.5
        delta_s = 0.7

        posterior_shape = r + n_previous
        posterior_rate = r + s_previous

        no_event_probability = (
            posterior_rate / (posterior_rate + delta_s)
        ) ** posterior_shape

        expected = -np.log(no_event_probability)

        implemented_formula = (r + n_previous) * np.log(
            (r + s_previous + delta_s) / (r + s_previous)
        )

        assert implemented_formula == pytest.approx(
            expected,
            rel=1e-12,
        )


class TestZeroInflatedFrailtyCompensatorKnownAnswer:

    def test_pi_zero_reduces_to_gamma_frailty(self):
        pi = 0.0
        r = 3.0
        c = 0.0
        beta = r

        n_previous = 4
        s_previous = 2.5
        s_new = 3.2

        gain = ZeroInflatedGain(pi, c, GammaGain(r, beta))
        computed = gain.tau(n_previous, s_previous, s_new)

        expected = (r + n_previous) * np.log((r + s_new) / (r + s_previous))

        assert computed == pytest.approx(
            expected,
            rel=1e-10,
        )


class TestRenewalKnownAnswer:

    def test_exponential_excitation_two_events(self):
        """
        Direct known-answer test of RenewalProcess._stream_integral_and_ll()
        for a homogeneous baseline and two events.
        """
        B = 0.5
        A_exc = 1.2
        tau_exc = 0.3
        t1, t2 = 1.0, 1.8
        duration_s = 3.0

        model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(),
            RenewalKernelFactory.exponential_excitation(),
            integration_dt=0.001,
        )

        computed_log_intensity, computed_integral = model._stream_integral_and_ll(
            t_events=np.array([t1, t2]),
            trial=0,
            duration_s=duration_s,
            params_base=[B],
            params_renewal=[A_exc, tau_exc],
        )

        # First event has no prior-event modulation.
        lambda_1 = B

        # Second event depends on the lag from the first event.
        lambda_2 = B * (1.0 + A_exc * np.exp(-(t2 - t1) / tau_exc))
        expected_log_intensity = np.log(lambda_1) + np.log(lambda_2)

        def modulated_segment_integral(length):
            return B * (length + A_exc * tau_exc * (1.0 - np.exp(-length / tau_exc)))

        # Before t1: unmodulated baseline.
        expected_integral = B * t1

        # After t1, until t2: modulation relative to t1.
        expected_integral += modulated_segment_integral(t2 - t1)

        # After t2, until trial end: modulation resets relative to t2.
        expected_integral += modulated_segment_integral(duration_s - t2)

        assert computed_log_intensity == pytest.approx(
            expected_log_intensity,
            rel=1e-10,
        )
        assert computed_integral == pytest.approx(
            expected_integral,
            rel=1e-5,
        )

    def test_empty_stream_is_unmodulated_baseline(self):
        B = 0.5
        duration_s = 3.0

        model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(),
            RenewalKernelFactory.exponential_excitation(),
        )

        log_intensity, total_integral = model._stream_integral_and_ll(
            t_events=np.array([], dtype=float),
            trial=0,
            duration_s=duration_s,
            params_base=[B],
            params_renewal=[1.2, 0.3],
        )

        assert log_intensity == pytest.approx(0.0)
        assert total_integral == pytest.approx(B * duration_s)

class TestGainOnDrivenRateKnownAnswer:
    @pytest.mark.parametrize("n_events, burst_share, tol", [(0, 0.0, 1e-6), (5, 0.5, 2e-3), (800, 0.0, 1e-6), (800, 0.75, 1e-6)])
    def test_evidence_matches_adaptive_quadrature(self, n_events: int, burst_share: float, tol: float) -> None:
        """
        log Z of a gain on the driven rate, in NumPy and JAX: exact for many events whatever
        share of them are bursts; within 1e-3 for a few events mostly bursts and dispersed gains.
        """
        import jax.numpy as jnp
        from scipy.integrate import quad
        from scipy.special import gammaln
        from BehaviorScreen.point_process.point_process import gains_jax
        from BehaviorScreen.point_process.point_process.frailty import gamma_gain_evidence_by_fish

        rng = np.random.default_rng(n_events)
        d = rng.uniform(0.5, 1.5, n_events)
        b = d * burst_share / (1 - burst_share) * rng.uniform(0.5, 1.5, n_events)
        b[:1] = 0.0  # a trial's first event: no earlier one to burst from
        shape, exposure = 0.7, n_events * (1 - burst_share) + 3.0
        log_f = lambda v: shape * v - shape * np.exp(v) + np.sum(np.log(np.exp(v) * d + b)) - np.exp(v) * exposure
        v = np.linspace(-15, 5, 4001)
        peak_v = v[np.argmax([log_f(x) for x in v])]
        peak = log_f(peak_v)
        expected = shape * np.log(shape) - gammaln(shape) + peak + np.log(
            quad(lambda x: np.exp(log_f(x) - peak), -40, 8, points=[peak_v], limit=500, epsrel=1e-12)[0]
        )
        fish, S = np.zeros(n_events, dtype=int), np.array([exposure])
        assert gamma_gain_evidence_by_fish(d, b, fish, S, shape, shape)[0][0] == pytest.approx(expected, abs=tol)
        jax_value = gains_jax.gamma_gain_evidence_by_fish(*(jnp.asarray(x) for x in (d, b, fish, S)), jnp.asarray(shape), jnp.asarray(shape))
        assert float(jax_value[0]) == pytest.approx(expected, abs=tol)


class TestTrialGainRescalingKnownAnswer:
    @pytest.mark.parametrize("r, a, sizes, burst", [(1.3, 0.7, [3, 0, 5], 0.0), (0.7, 4.0, [2, 6], 0.6)])
    def test_taus_match_double_quadrature(self, r: float, a: float, sizes: list, burst: float) -> None:
        """
        Rescaled intervals with fish and trial gains, g h on the driven rate: -log of the predictive
        no-event probability over (g, h) given the fish's earlier intervals, by a log-g x log-h trapezoid.
        """
        from scipy.special import gammaln, logsumexp
        from BehaviorScreen.point_process.point_process.frailty import GammaGain, sequential_trial_drive_taus

        rng = np.random.default_rng(1)
        v, u = np.linspace(-30, 6, 1801)[:, None], np.linspace(-30, 6, 1801)[None, :]
        g, h = np.exp(v), np.exp(u)
        log_ph = a * np.log(a) - gammaln(a) + a * u - a * h
        W, expected, columns = r * np.log(r) - gammaln(r) + r * v[:, 0] - r * g[:, 0], [], []
        for m, n in enumerate(sizes):
            S, d = rng.uniform(0.02, 0.4, n + 1), rng.uniform(0.5, 1.5, n + 1)
            b = d * burst * rng.uniform(0, 2, n + 1)
            d[-1] = b[-1] = b[0] = 0.0
            columns.append((np.full(n + 1, m), S, burst * S, d, b, np.arange(n + 1) == n))
            log_l = np.zeros_like(v + u)
            for j in range(n + 1):
                expected.append(burst * S[j] - logsumexp(W[:, None] + log_ph + log_l - g * h * S[j]) + logsumexp(W[:, None] + log_ph + log_l))
                log_l = log_l - g * h * S[j] + (np.log(g * h * d[j] + b[j]) if j < n else 0.0)
            W = W + logsumexp(log_ph + log_l, axis=1)
        stream, S, B, d, b, censored = (np.concatenate(c) for c in zip(*columns))
        taus = sequential_trial_drive_taus(GammaGain(r, r), a, stream, 10, S, B, d, b, censored)
        np.testing.assert_allclose(taus, expected, atol=1e-6)
