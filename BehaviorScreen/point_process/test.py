import numpy as np
import pytest

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalProcess, RenewalKernelFactory
from BehaviorScreen.point_process.hawkes_process import HawkesProcess, HistoryKernelFactory
from BehaviorScreen.point_process.mixed_effects_process import GammaMixedEffectsProcess


RTOL = 0.05  # 5% relative tolerance -- generous, since these are stochastic
             # simulations, not exact recovery; tightened per-test where
             # the simulation size makes tighter recovery reliable.


def _empty_dataset_shell(n_fish, n_trials, duration_s, binning_dt=0.1):
    return dict(
        fish_trial_mask=np.ones((n_fish, n_trials), dtype=bool),
        duration_s=duration_s,
        binning_dt=binning_dt,
    )

def _simulate_via_thinning(rng, duration_s, lambda_upper, rate_fn):
    """
    Generic thinning simulator: proposes events from a homogeneous
    Poisson process at lambda_upper (an upper bound on the TRUE,
    history-dependent rate_fn), then accepts each proposal with
    probability rate_fn(t, accepted_events_so_far) / lambda_upper.

    rate_fn(t, history) must return the TRUE intensity at time t given
    the events accepted so far (a sorted list) -- this makes it a valid,
    from-scratch simulator for any renewal-kernel shape, independent of
    the fitting code being tested.
    """
    events = []
    n_proposals = rng.poisson(lambda_upper * duration_s)
    proposals = np.sort(rng.uniform(0, duration_s, n_proposals))
    for t in proposals:
        lam = rate_fn(t, events)
        if rng.uniform() < lam / lambda_upper:
            events.append(t)
    return np.array(events)


def _build_dataset(rng, n_fish, n_trials, duration_s, lambda_upper, rate_fn):
    event_times, event_fish, event_trials = [], [], []
    for f in range(n_fish):
        for t_idx in range(n_trials):
            times = _simulate_via_thinning(rng, duration_s, lambda_upper, rate_fn)
            event_times.append(times)
            event_fish.append(np.full(len(times), f))
            event_trials.append(np.full(len(times), t_idx))
    return PointProcessDataset(
        event_times=np.concatenate(event_times) if event_times else np.array([]),
        event_trials_idx=np.concatenate(event_trials).astype(int) if event_trials else np.array([], dtype=int),
        event_fish_idx=np.concatenate(event_fish).astype(int) if event_fish else np.array([], dtype=int),
        **_empty_dataset_shell(n_fish, n_trials, duration_s),
    )

# ============================================================================
# Poisson recovery: simulate a genuinely inhomogeneous rate, check we get
# the same shape back
# ============================================================================

class TestPoissonRecovery:

    def test_recovers_homogeneous_rate(self, rng=np.random.default_rng(0)):
        true_rate = 0.8
        n_fish, n_trials, duration_s = 40, 5, 20.0

        event_times, event_fish, event_trials = [], [], []
        for f in range(n_fish):
            for t in range(n_trials):
                n = rng.poisson(true_rate * duration_s)
                times = np.sort(rng.uniform(0, duration_s, n))
                event_times.append(times)
                event_fish.append(np.full(n, f))
                event_trials.append(np.full(n, t))

        ds = PointProcessDataset(
            event_times=np.concatenate(event_times),
            event_trials_idx=np.concatenate(event_trials).astype(int),
            event_fish_idx=np.concatenate(event_fish).astype(int),
            **_empty_dataset_shell(n_fish, n_trials, duration_s),
        )

        model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        model.fit(ds)

        assert np.isclose(model.params_[0], true_rate, rtol=RTOL)

    def test_recovers_exponential_decay_shape(self, rng=np.random.default_rng(1)):
        """
        Simulate from a simple exponentially-decaying rate lambda(t) =
        B*(1 - f_dip*exp(-t/tau)) (matches RateKernelFactory.omr_forward's
        functional form) via thinning, and confirm the fitted B/f_dip/tau
        land near the true values.
        """
        true_B, true_f_dip, true_tau = 1.0, 0.6, 2.0
        duration_s = 15.0
        n_fish, n_trials = 60, 4

        def true_rate_fn(t):
            return true_B * (1 - true_f_dip * np.exp(-t / true_tau))

        lambda_max = true_B  # rate is monotonically increasing toward B

        event_times, event_fish, event_trials = [], [], []
        for f in range(n_fish):
            for t_idx in range(n_trials):
                # thinning algorithm: propose homogeneous Poisson(lambda_max),
                # accept with probability true_rate_fn(t)/lambda_max
                n_proposals = rng.poisson(lambda_max * duration_s)
                proposals = np.sort(rng.uniform(0, duration_s, n_proposals))
                accept_prob = true_rate_fn(proposals) / lambda_max
                accepted = proposals[rng.uniform(size=n_proposals) < accept_prob]
                event_times.append(accepted)
                event_fish.append(np.full(len(accepted), f))
                event_trials.append(np.full(len(accepted), t_idx))

        ds = PointProcessDataset(
            event_times=np.concatenate(event_times),
            event_trials_idx=np.concatenate(event_trials).astype(int),
            event_fish_idx=np.concatenate(event_fish).astype(int),
            **_empty_dataset_shell(n_fish, n_trials, duration_s),
        )

        model = PoissonProcess(RateKernelFactory.omr_forward())
        model.fit(ds)
        B_hat, f_dip_hat, tau_hat = model.params_

        assert np.isclose(B_hat, true_B, rtol=RTOL)
        assert np.isclose(f_dip_hat, true_f_dip, rtol=RTOL + 0.1)  # noisier param, looser band
        assert np.isclose(tau_hat, true_tau, rtol=RTOL + 0.1)


class TestExponentialRecoveryRecovery:

    def test_recovers_true_recovery_timescale(self):
        rng = np.random.default_rng(10)
        true_B, true_tau_r = 1.2, 0.4
        duration_s, n_fish, n_trials = 20.0, 60, 5

        def rate_fn(t, history):
            if not history:
                return true_B
            lag = t - history[-1]
            rho = 1.0 - np.exp(-lag / true_tau_r)
            return true_B * rho

        # upper bound: rho <= 1, so true_B is always a valid ceiling
        ds = _build_dataset(rng, n_fish, n_trials, duration_s, lambda_upper=true_B, rate_fn=rate_fn)

        model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(),
            RenewalKernelFactory.exponential_recovery(),
        )
        model.fit(ds)
        B_hat, tau_r_hat = model.params_

        assert np.isclose(B_hat, true_B, rtol=RTOL)
        assert np.isclose(tau_r_hat, true_tau_r, rtol=RTOL + 0.1)

    def test_null_data_recovers_small_tau_r(self):
        """
        Negative control: true homogeneous Poisson (no refractoriness).
        Since exponential_recovery's rho(0)=0 unconditionally (it always
        imposes SOME suppression right after an event), the best fit to
        null data should push tau_r toward its lower bound (fast
        recovery, approximating "no real effect") rather than settling on
        a large, spurious refractory period.
        """
        rng = np.random.default_rng(11)
        true_B = 1.0
        duration_s, n_fish, n_trials = 20.0, 60, 5

        event_times, event_fish, event_trials = [], [], []
        for f in range(n_fish):
            for t_idx in range(n_trials):
                n = rng.poisson(true_B * duration_s)
                times = np.sort(rng.uniform(0, duration_s, n))
                event_times.append(times)
                event_fish.append(np.full(n, f))
                event_trials.append(np.full(n, t_idx))

        ds = PointProcessDataset(
            event_times=np.concatenate(event_times),
            event_trials_idx=np.concatenate(event_trials).astype(int),
            event_fish_idx=np.concatenate(event_fish).astype(int),
            **_empty_dataset_shell(n_fish, n_trials, duration_s),
        )

        model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(),
            RenewalKernelFactory.exponential_recovery(),
        )
        model.fit(ds)
        _, tau_r_hat = model.params_

        assert tau_r_hat < 0.05  # near its lower bound (0.001)


# ============================================================================
# exponential_excitation: rho(lag) = 1 + A_exc * exp(-lag/tau_exc)
# ============================================================================

class TestExponentialExcitationRecovery:

    def test_recovers_true_excitation_params(self):
        rng = np.random.default_rng(20)
        true_B, true_A_exc, true_tau_exc = 0.8, 1.5, 0.25
        duration_s, n_fish, n_trials = 20.0, 60, 5

        def rate_fn(t, history):
            if not history:
                return true_B
            lag = t - history[-1]
            rho = 1.0 + true_A_exc * np.exp(-lag / true_tau_exc)
            return true_B * rho

        # upper bound: rho <= 1 + true_A_exc
        lambda_upper = true_B * (1.0 + true_A_exc)
        ds = _build_dataset(rng, n_fish, n_trials, duration_s, lambda_upper, rate_fn)

        model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(),
            RenewalKernelFactory.exponential_excitation(),
        )
        model.fit(ds)
        B_hat, A_exc_hat, tau_exc_hat = model.params_

        assert np.isclose(B_hat, true_B, rtol=RTOL)
        assert np.isclose(A_exc_hat, true_A_exc, rtol=RTOL + 0.15)
        assert np.isclose(tau_exc_hat, true_tau_exc, rtol=RTOL + 0.15)

    def test_null_data_recovers_zero_excitation(self):
        """
        Negative control: true homogeneous Poisson (no real event-to-event
        facilitation). A_excitation should collapse toward its lower
        bound (0.0), not settle on a spurious positive value -- this is
        the exact confound checked manually earlier for prey_capture_ipsi
        (fish heterogeneity vs. real excitation); here it's the simplest
        possible version: no heterogeneity, no excitation, single-rate
        Poisson ground truth.
        """
        rng = np.random.default_rng(21)
        true_B = 1.0
        duration_s, n_fish, n_trials = 20.0, 60, 5

        event_times, event_fish, event_trials = [], [], []
        for f in range(n_fish):
            for t_idx in range(n_trials):
                n = rng.poisson(true_B * duration_s)
                times = np.sort(rng.uniform(0, duration_s, n))
                event_times.append(times)
                event_fish.append(np.full(n, f))
                event_trials.append(np.full(n, t_idx))

        ds = PointProcessDataset(
            event_times=np.concatenate(event_times),
            event_trials_idx=np.concatenate(event_trials).astype(int),
            event_fish_idx=np.concatenate(event_fish).astype(int),
            **_empty_dataset_shell(n_fish, n_trials, duration_s),
        )

        model = RenewalProcess(
            RateKernelFactory.homogeneous_poisson(),
            RenewalKernelFactory.exponential_excitation(),
        )
        model.fit(ds)
        _, A_exc_hat, _ = model.params_

        assert A_exc_hat < 0.1  # near its lower bound (0.0)


# ============================================================================
# Hawkes recovery: simulate WITH self-excitation, confirm alpha/beta land
# near truth
# ============================================================================

class TestHawkesRecovery:

    def test_recovers_self_excitation(self, rng=np.random.default_rng(4)):
        """
        Simulate an exponential-kernel Hawkes process via Ogata's
        thinning algorithm, confirm fitted alpha_hawkes/beta_hawkes land
        near the true simulation parameters.
        """
        true_B, true_alpha, true_beta = 0.3, 0.5, 2.0
        duration_s = 25.0
        n_fish, n_trials = 40, 4

        def simulate_hawkes_stream(rng, duration_s, B, alpha, beta):
            events = []
            t = 0.0
            while t < duration_s:
                # current intensity: baseline + sum of decayed excitation
                lam = B + sum(alpha * np.exp(-beta * (t - e)) for e in events)
                lam_upper = B + sum(alpha * np.exp(-beta * (t - e)) for e in events)  # at time t, this IS the current upper bound since intensity only decays going forward until the next jump
                w = rng.exponential(1.0 / max(lam_upper, 1e-6))
                t_candidate = t + w
                if t_candidate >= duration_s:
                    break
                lam_candidate = B + sum(alpha * np.exp(-beta * (t_candidate - e)) for e in events)
                if rng.uniform() <= lam_candidate / lam_upper:
                    events.append(t_candidate)
                t = t_candidate
            return np.array(events)

        event_times, event_fish, event_trials = [], [], []
        for f in range(n_fish):
            for t_idx in range(n_trials):
                times = simulate_hawkes_stream(rng, duration_s, true_B, true_alpha, true_beta)
                event_times.append(times)
                event_fish.append(np.full(len(times), f))
                event_trials.append(np.full(len(times), t_idx))

        ds = PointProcessDataset(
            event_times=np.concatenate(event_times),
            event_trials_idx=np.concatenate(event_trials).astype(int),
            event_fish_idx=np.concatenate(event_fish).astype(int),
            **_empty_dataset_shell(n_fish, n_trials, duration_s),
        )

        model = HawkesProcess(
            RateKernelFactory.homogeneous_poisson(),
            HistoryKernelFactory.exponential(),
        )
        model.fit(ds)
        B_hat, alpha_hat, beta_hat = model.params_

        assert np.isclose(B_hat, true_B, rtol=RTOL + 0.1)
        assert np.isclose(alpha_hat, true_alpha, rtol=RTOL + 0.15)  # alpha/beta are
        assert np.isclose(beta_hat, true_beta, rtol=RTOL + 0.15)    # jointly harder to pin down precisely


# ============================================================================
# GammaMixedEffectsProcess recovery: simulate WITH real per-fish
# heterogeneity, confirm r_dispersion recovers the true amount
# ============================================================================

class TestGammaMixedEffectsRecovery:

    def test_recovers_fish_heterogeneity(self, rng=np.random.default_rng(5)):
        """
        Simulate each fish with its OWN gain g_f ~ Gamma(r, r), then
        homogeneous-Poisson events at rate g_f * B. Confirm fitted r
        lands near the true r, and that a plain (non-mixed) PoissonProcess
        fit on the SAME data shows a much higher (falsely low-
        heterogeneity) apparent fit by comparison -- i.e. the frailty
        model should win decisively on this data by construction.
        """
        true_B, true_r = 1.0, 3.0
        duration_s = 15.0
        n_fish, n_trials = 80, 5

        true_gains = rng.gamma(shape=true_r, scale=1.0 / true_r, size=n_fish)

        event_times, event_fish, event_trials = [], [], []
        for f in range(n_fish):
            fish_rate = true_B * true_gains[f]
            for t_idx in range(n_trials):
                n = rng.poisson(fish_rate * duration_s)
                times = np.sort(rng.uniform(0, duration_s, n))
                event_times.append(times)
                event_fish.append(np.full(n, f))
                event_trials.append(np.full(n, t_idx))

        ds = PointProcessDataset(
            event_times=np.concatenate(event_times),
            event_trials_idx=np.concatenate(event_trials).astype(int),
            event_fish_idx=np.concatenate(event_fish).astype(int),
            **_empty_dataset_shell(n_fish, n_trials, duration_s),
        )

        base = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        model = GammaMixedEffectsProcess(base)
        model.fit(ds)

        B_hat = model.base_process.params_[0]
        r_hat = model.dispersion_r

        assert np.isclose(B_hat, true_B, rtol=RTOL)
        assert np.isclose(r_hat, true_r, rtol=RTOL + 0.2)  # r is notoriously
                                                              # harder to pin down precisely

        # AIC sanity: the frailty model MUST beat plain Poisson decisively
        # on data simulated with real heterogeneity -- this is a strong,
        # cheap end-to-end check that the whole comparison pipeline
        # correctly favors the true generating process.
        plain = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        plain.fit(ds)
        assert model.aic < plain.aic - 50  # decisive margin, not just "slightly better"

    def test_no_heterogeneity_recovers_large_r(self, rng=np.random.default_rng(6)):
        """
        Negative control: simulate with NO fish heterogeneity (every fish
        shares the exact same rate), confirm fitted r is large (i.e. the
        frailty term correctly finds "no heterogeneity" rather than
        hallucinating some).
        """
        true_B = 1.0
        duration_s = 15.0
        n_fish, n_trials = 80, 5

        event_times, event_fish, event_trials = [], [], []
        for f in range(n_fish):
            for t_idx in range(n_trials):
                n = rng.poisson(true_B * duration_s)
                times = np.sort(rng.uniform(0, duration_s, n))
                event_times.append(times)
                event_fish.append(np.full(n, f))
                event_trials.append(np.full(n, t_idx))

        ds = PointProcessDataset(
            event_times=np.concatenate(event_times),
            event_trials_idx=np.concatenate(event_trials).astype(int),
            event_fish_idx=np.concatenate(event_fish).astype(int),
            **_empty_dataset_shell(n_fish, n_trials, duration_s),
        )

        base = PoissonProcess(RateKernelFactory.homogeneous_poisson())
        model = GammaMixedEffectsProcess(base)
        model.fit(ds)

        assert model.dispersion_r > 15  # should be large; no true heterogeneity to find