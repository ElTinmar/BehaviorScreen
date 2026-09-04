from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
from scipy.integrate import trapezoid

from .dataset import PointProcessDataset
from .point_process import PointProcess
from .poisson_process import RateKernel

@dataclass(frozen=True)
class RenewalKernel:
    """
    Generic time-since-last-event modulation kernel: rho(lag) >= 0 for
    lag >= 0. No sign or shape assumption -- rho can be suppressive
    (refractory), facilitating, rhythmic/bump-shaped, or any combination.
    Used by RenewalProcess as a multiplicative modulator of a RateKernel's
    base intensity.
    """
    name: str
    func: Callable[[np.ndarray, List[float]], np.ndarray]
    param_names: List[str]
    initial_guesses: List[float]
    bounds: List[Tuple[Optional[float], Optional[float]]]
    integral_func: Optional[Callable[[np.ndarray, List[float]], np.ndarray]] = None
    latex_formula: str = ""

    def evaluate(self, lag: np.ndarray, params: List[float]) -> np.ndarray:
        if np.any(lag < 0):
            raise ValueError("Renewal-kernel lags must be non-negative.")
        return self.func(lag, params)

    def integrate(
        self,
        duration: Union[float, np.ndarray],
        params: List[float],
        integration_dt: float = 0.02,
    ) -> np.ndarray:

        duration = np.asarray(duration, dtype=float)
        if np.any(duration < 0):
            raise ValueError("Integration duration must be non-negative.")

        if self.integral_func is not None:
            return self.integral_func(duration, params)

        scalar_input = duration.ndim == 0
        durations = np.atleast_1d(duration)
        result = np.empty_like(durations)

        for i, T in enumerate(durations):
            if T == 0:
                result[i] = 0.0
                continue

            t_grid = np.arange(0.0,T + integration_dt,integration_dt)
            if t_grid[-1] > T:
                t_grid[-1] = T
            elif t_grid[-1] < T:
                t_grid = np.append(t_grid, T)

            values = self.evaluate(t_grid, params)
            result[i] = trapezoid(values, t_grid)

        return result[0] if scalar_input else result

class RenewalKernelFactory:

    @staticmethod
    def exponential_recovery() -> RenewalKernel:
        """rho(0)=0, approaches 1 with time constant tau_r. Smooth refractory recovery."""
        def _func(lag, params):
            (tau_r,) = params
            return 1.0 - np.exp(-lag / tau_r)
        return RenewalKernel(
            name="ExponentialRecovery", func=_func,
            param_names=["tau_refractory"], initial_guesses=[0.15], bounds=[(0.001, 30.0)],
            latex_formula=r"$\rho(\Delta t) = 1 - e^{-\Delta t/\tau_r}$",
        )

    @staticmethod
    def exponential_excitation() -> RenewalKernel:
        """
        rho(0) = 1 + A_exc (elevated right after an event), decaying back to
        baseline (rho -> 1) with time constant tau_exc. The FACILITATING
        counterpart to hard_dead_time()/exponential_recovery()/
        dead_time_plus_recovery() (all of which are suppressive, rho in [0,1]).

        A_exc is bounded >= 0 so this kernel can only facilitate, never
        suppress -- keeping it a clean, unambiguous test of ONE specific
        hypothesis (excitatory influence of the immediately preceding event)
        rather than a general-purpose shape that could come out suppressive
        and duplicate hard_dead_time()/exponential_recovery()'s role.
        """
        def _func(lag, params):
            A_exc, tau_exc = params
            return 1.0 + A_exc * np.exp(-lag / tau_exc)
            # rho >= 1 for ANY lag >= 0 given A_exc >= 0 -- positive by
            # construction, no clipping needed (unlike dead_time_plus_recovery,
            # which needs np.clip because its recovery term can go negative).

        return RenewalKernel(
            name="ExponentialExcitation", func=_func,
            param_names=["A_excitation", "tau_excitation"],
            initial_guesses=[0.5, 0.2],
            bounds=[(0.0, 20.0), (0.02, 5.0)],
            latex_formula=r"$\rho(\Delta t) = 1 + A_{\text{exc}}\, e^{-\Delta t/\tau_{\text{exc}}}$",
        )

    @staticmethod
    def hard_absorption() -> RenewalKernel:
        """
        rho(lag) = 0 for all lag > 0 -- an infinite, non-recovering refractory
        period. This is the "absorbing"/survival hypothesis expressed as a
        RenewalKernel, fit on the FULL (unreduced) event stream -- unlike
        SurvivalProcess, which achieves the same assumption via a differently-
        scoped likelihood over manually-reduced data and is therefore NOT
        directly AIC-comparable to this module's other process families.

        RenewalProcess(kernel, hard_absorption) reproduces SurvivalProcess(kernel)
        EXACTLY for any stream with 0 or 1 events (verify: _stream_integral_and_ll
        reduces to integral_0^t_obs h - log(h(t_obs)) for n=1, integral_0^T h for
        n=0 -- identical to SurvivalProcess's NLL terms). The difference is that
        it also correctly penalizes (rather than silently discarding) any stream
        with >=2 events, via the existing 1e-12 floor in
        RenewalProcess._stream_integral_and_ll's per-event log-intensity clamp --
        so fitting THIS, nested against ordinary RenewalProcess/HawkesProcess
        kernels, is the statistically valid way to test whether the absorbing
        assumption holds, rather than assuming it via SurvivalProcess's data
        reduction.
        """
        def _func(lag, params):
            return np.zeros_like(lag)
        return RenewalKernel(
            name="HardAbsorption", func=_func,
            param_names=[], initial_guesses=[], bounds=[],
            integral_func=lambda duration, params: 0.0 * np.asarray(duration, dtype=float),
            latex_formula=r"$\rho(\Delta t) = 0,\ \Delta t > 0$",
        )
class RenewalProcess(PointProcess):
    """
    Modulated renewal process: intensity depends on absolute time-in-trial
    AND trial number (via `kernel`, exactly as in PoissonProcess), multiplied
    by a recovery function of time-since-THIS-STREAM'S-OWN-most-recent-event
    (via `renewal_kernel`).

        lambda(t | history) = kernel(t, m) * renewal_kernel(t - t_last)

    Distinct from HawkesProcess: multiplicative (not additive), and depends
    only on the single most recent event (order-1 Markov in the event
    sequence), not the full event history -- see conversation notes on
    process-family taxonomy. Appropriate when diagnostics show NEGATIVE
    lag-1 autocorrelation / regularity (refractoriness), as opposed to
    HawkesProcess's positive-autocorrelation / clustering regime.
    """

    def __init__(self, kernel: RateKernel, renewal_kernel: RenewalKernel, integration_dt: float = 0.02):
        super().__init__(integration_dt)
        self.name = f"Renewal rate: {kernel.name}, history: {renewal_kernel.name}"
        kernel_formula = kernel.latex_formula.strip("$")
        renewal_formula = renewal_kernel.latex_formula.strip("$")
        self.latex_formula = rf"${kernel_formula} \times {renewal_formula}$"
        self.kernel = kernel
        self.renewal_kernel = renewal_kernel
        self.initial_guesses = kernel.initial_guesses + renewal_kernel.initial_guesses
        self.bounds = kernel.bounds + renewal_kernel.bounds
        self.param_names = kernel.param_names + renewal_kernel.param_names

    def _split_params(self, params: List[float]) -> Tuple[List[float], List[float]]:
        n = len(self.kernel.param_names)
        return params[:n], params[n:]

    def _segment_integral(
        self, t0: float, t1: float, trial: float,
        params_base: List[float], t_ref: Optional[float], params_refractory: List[float],
    ) -> float:
        """Integral of kernel(s, trial) * renewal_kernel(s - t_ref) over [t0, t1]. t_ref=None -> no suppression."""
        if t1 <= t0:
            return 0.0
        grid = np.arange(t0, t1 + self.integration_dt, self.integration_dt)
        grid[-1] = min(grid[-1], t1)

        lam = self.kernel.evaluate(grid, np.full_like(grid, trial), params_base)
        if t_ref is not None:
            rho = self.renewal_kernel.evaluate(grid - t_ref, params_refractory)
            lam = lam * rho

        return trapezoid(lam, grid)
    
    def _stream_integral_and_ll(
        self, t_events: np.ndarray, trial: float, duration_s: float,
        params_base: List[float], params_renewal: List[float],
    ) -> Tuple[float, float]:

        t_events = np.sort(t_events)
        n = len(t_events)

        grid = np.arange(0.0, duration_s + self.integration_dt, self.integration_dt)
        grid[-1] = min(grid[-1], duration_s)
        if grid[-1] < duration_s:
            grid = np.append(grid, duration_s)

        base_vals = self.kernel.evaluate(grid, np.full_like(grid, trial), params_base)

        if n == 0:
            return 0.0, float(trapezoid(base_vals, grid))

        idx = np.searchsorted(t_events, grid, side='right') - 1
        has_prior = idx >= 0
        t_last = np.where(has_prior, t_events[np.clip(idx, 0, None)], 0.0)

        rho = np.ones_like(grid)
        rho[has_prior] = self.renewal_kernel.evaluate(
            grid[has_prior] - t_last[has_prior], params_renewal
        )
        lam_grid = base_vals * rho
        total_integral = float(trapezoid(lam_grid, grid))

        lam0 = max(self.kernel.evaluate(np.array([t_events[0]]), np.array([trial]), params_base)[0], 1e-12)
        sum_log_intensity = np.log(lam0)
        for i in range(1, n):
            dt = t_events[i] - t_events[i - 1]
            lam = self.kernel.evaluate(np.array([t_events[i]]), np.array([trial]), params_base)[0]
            rho_i = self.renewal_kernel.evaluate(np.array([dt]), params_renewal)[0]
            sum_log_intensity += np.log(max(lam * rho_i, 1e-12))

        return float(sum_log_intensity), total_integral

    def _renewal_nll(self, t_events, trial, duration_s, params_base, params_refractory) -> float:
        sum_log_intensity, total_integral = self._stream_integral_and_ll(
            t_events, trial, duration_s, params_base, params_refractory
        )
        return -(sum_log_intensity - total_integral)
    
    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        params_base, params_renewal = self._split_params(params)
        total_nll = 0.0
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            total_nll += self._renewal_nll(t_ev, t_idx, dataset.duration_s, params_base, params_renewal)
        return total_nll

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray],
                history_events: Optional[np.ndarray] = None) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        params_base, params_renewal = self._split_params(self.params_)

        base_rate = self.kernel.evaluate(t, trial, params_base)
        if history_events is None or len(history_events) == 0:
            return np.maximum(base_rate, 1e-9)

        t_arr = np.asarray(t, dtype=float)
        events_sorted = np.sort(np.asarray(history_events, dtype=float))
        # nearest preceding event for each evaluation point t
        idx = np.searchsorted(events_sorted, t_arr, side='right') - 1
        has_prior = idx >= 0
        t_last = np.where(has_prior, events_sorted[np.clip(idx, 0, None)], np.nan)

        rho = np.ones_like(t_arr, dtype=float)
        valid = has_prior & (t_arr > t_last)
        rho[valid] = self.renewal_kernel.evaluate(t_arr[valid] - t_last[valid], params_renewal)

        return np.maximum(base_rate * rho, 1e-9)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        n_trials, n_bins = dataset.num_trials, len(dataset.t_centers)
        rate_sum = np.zeros((n_trials, n_bins), dtype=float)
        active_count = np.zeros(n_trials, dtype=int)

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            stream_rate = self.predict(t=dataset.t_centers, trial=t_idx, history_events=t_ev)
            rate_sum[t_idx, :] += stream_rate
            active_count[t_idx] += 1

        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(active_count[:, None] > 0, rate_sum / np.maximum(active_count, 1)[:, None], 0.0)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        params_base, params_renewal = self._split_params(self.params_)
        t_events = np.sort(np.asarray(t_events))
        if len(t_events) == 0:
            return np.array([], dtype=float)

        Lambda = np.zeros_like(t_events, dtype=float)
        Lambda[0] = self._segment_integral(0.0, t_events[0], trial, params_base, None, None) if len(t_events) else 0.0
        for i in range(1, len(t_events)):
            Lambda[i] = Lambda[i - 1] + self._segment_integral(
                t_events[i - 1], t_events[i], trial, params_base, t_events[i - 1], params_renewal
            )
        return Lambda

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        See PointProcess.mixed_effects_likelihood_terms for the general
        contract. Disaggregates RenewalProcess's own _renewal_nll (summed
        across all streams) into per-fish base_ll/N_f/S_f.

        PRECONDITION CHECK: a per-fish gain g_f satisfies the "multiplies the
        ENTIRE intensity uniformly" requirement here, since RenewalProcess's
        intensity is already fully multiplicative:
            lambda(t | history) = kernel(t, m) * renewal_kernel(t - t_last)
        so g_f * lambda(t|history) factors as g_f^{N_f} in the likelihood
        product and g_f * S_f in the integral term, exactly as required.

        base_ll : sum, over EVERY event in every stream, of
                log(kernel(t_i, m) * renewal_kernel(t_i - t_last_i)) --
                i.e. exactly _renewal_nll's sum_log_intensity term, but
                accumulated with a + sign (this is a log-likelihood
                contribution, not a negative-log-likelihood) and never
                g_f-dependent.
        N_f     : total observed event count per fish, across all its streams.
        S_f     : total refractory/renewal-corrected exposure integral per
                fish -- i.e. _renewal_nll's total_integral term (which
                already accounts for renewal_kernel suppression/recovery
                after each event, using that stream's OWN event history),
                summed across every trial that fish participated in.

        Renewal history resets at each trial boundary (same convention as
        _renewal_nll/_nll) -- this method does not change that; only the
        per-fish AGGREGATION across trials is new here.
        """
        params_base, params_refractory = self._split_params(params)

        N_f = np.zeros(dataset.num_fish, dtype=float)
        S_f = np.zeros(dataset.num_fish, dtype=float)
        base_ll = 0.0

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            N_f[f_idx] += len(t_ev)

            sum_log_intensity, total_integral = self._stream_integral_and_ll(
                t_ev, t_idx, dataset.duration_s, params_base, params_refractory
            )
            base_ll += sum_log_intensity
            S_f[f_idx] += total_integral

        return base_ll, N_f, S_f

    def simulate_stream(self, dataset, t_idx, gain, rng) -> np.ndarray:
        """
        Ogata thinning WITH renewal modulation: intensity at any proposed time
        is gain * kernel(t, m) * renewal_kernel(t - t_last), where t_last is
        the most recently ACCEPTED simulated event (or no modulation before
        the first event) -- mirrors _stream_integral_and_ll's own
        interpretation of the fitted model exactly, unlike the base class's
        history-free default.

        Simpler than HawkesProcess's override: only the SINGLE most recent
        event matters (no recursive/cumulative history state to carry), per
        RenewalProcess's own multiplicative, order-1-Markov intensity
        (see class docstring).
        """
        params_base, params_renewal = self._split_params(self.params_)

        # Upper bound must account for the renewal kernel's own possible
        # amplification (e.g. ExponentialExcitation can push intensity ABOVE
        # the bare kernel's max right after an event) -- NOT just the base
        # kernel's peak, unlike a purely-suppressive renewal kernel where the
        # base kernel's max alone would suffice.
        base_upper = gain * self._intensity_upper_bound(dataset, t_idx)
        renewal_upper = self._renewal_kernel_upper_bound(params_renewal)
        lambda_upper = base_upper * renewal_upper

        events = []
        t = 0.0
        t_last = None
        while t < dataset.duration_s:
            w = rng.exponential(1.0 / max(lambda_upper, 1e-12))
            t_candidate = t + w
            if t_candidate >= dataset.duration_s:
                break

            base_c = gain * self.kernel.evaluate(
                np.array([t_candidate]), np.array([t_idx]), params_base
            )[0]
            if t_last is None:
                rho = 1.0  # no prior event this stream -- no modulation yet
            else:
                rho = self.renewal_kernel.evaluate(
                    np.array([t_candidate - t_last]), params_renewal
                )[0]
            lam_candidate = base_c * rho

            if rng.uniform() <= lam_candidate / lambda_upper:
                events.append(t_candidate)
                t_last = t_candidate   # only the SINGLE most recent event matters
            t = t_candidate

        return np.array(events)

    def _renewal_kernel_upper_bound(self, params_renewal) -> float:
        """
        Upper bound on renewal_kernel(lag) over lag >= 0, for thinning's
        proposal step. Default: evaluate on a grid and take the max --
        correct for any of the current RenewalKernelFactory shapes
        (hard_dead_time/exponential_recovery cap at 1; exponential_excitation
        caps at 1+A_exc, approached as lag->0 -- NOT unbounded, since A_exc is
        itself bounded above (20.0) at the kernel level).
        """
        lag_grid = np.linspace(0.0, 10.0, 500)  # wide enough to capture any decay
        vals = self.renewal_kernel.evaluate(lag_grid, params_renewal)
        return float(np.max(vals)) * 1.1