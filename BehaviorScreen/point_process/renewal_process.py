from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
from scipy.integrate import trapezoid

from .dataset import PointProcessDataset
from .point_process import PointProcess
from .poisson_process import RateKernel
from .kernel_shapes import peak_normalized_pulse

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

    def integrate(self, duration, params, integration_dt: float = 0.02):
        # identical to your existing HistoryKernel.integrate implementation
        ...


class RenewalKernelFactory:

    @staticmethod
    def hard_dead_time() -> RenewalKernel:
        """rho = 0 for lag < tau_r, else 1. Strict minimum inter-event interval."""
        def _func(lag, params):
            (tau_r,) = params
            return np.where(lag < tau_r, 0.0, 1.0)
        return RenewalKernel(
            name="HardDeadTime", func=_func,
            param_names=["tau_refractory"], initial_guesses=[0.15], bounds=[(0.001, 2.0)],
            latex_formula=r"$\rho(\Delta t) = \mathbb{1}[\Delta t \geq \tau_r]$",
        )

    @staticmethod
    def exponential_recovery() -> RenewalKernel:
        """rho(0)=0, approaches 1 with time constant tau_r. Smooth refractory recovery."""
        def _func(lag, params):
            (tau_r,) = params
            return 1.0 - np.exp(-lag / tau_r)
        return RenewalKernel(
            name="ExponentialRecovery", func=_func,
            param_names=["tau_refractory"], initial_guesses=[0.15], bounds=[(0.001, 2.0)],
            latex_formula=r"$\rho(\Delta t) = 1 - e^{-\Delta t/\tau_r}$",
        )

    @staticmethod
    def dead_time_plus_recovery() -> RenewalKernel:
        """Hard dead time tau_d, then exponential recovery with time constant tau_r."""
        def _func(lag, params):
            tau_d, tau_r = params
            recovered = 1.0 - np.exp(-(lag - tau_d) / tau_r)
            return np.where(lag < tau_d, 0.0, np.clip(recovered, 0.0, 1.0))
        return RenewalKernel(
            name="DeadTimePlusRecovery", func=_func,
            param_names=["tau_dead", "tau_recovery"], initial_guesses=[0.1, 0.1],
            bounds=[(0.0, 1.0), (0.001, 2.0)],
            latex_formula=r"$\rho(\Delta t) = \mathbb{1}[\Delta t \geq \tau_d]\left(1-e^{-(\Delta t-\tau_d)/\tau_r}\right)$",
        )

    @staticmethod
    def gamma_bump(fixed_shape: Optional[float] = None) -> RenewalKernel:
        """
        Rhythmic/preferred-interval kernel: rho peaks at some positive lag
        rather than at zero, then decays. Appropriate when an ISI histogram
        shows a mode away from zero rather than a monotonic refractory recovery
        (e.g. the phototaxis_contra investigation).

            rho(lag) = 1 + A * (lag/lag_peak)^k * exp(k*(1 - lag/lag_peak))

        Reuses the same peak-normalized pulse shape as dark_flash's time-domain
        kernel (peak_normalized_pulse), applied here to inter-event lag
        instead of time-in-trial. A >= 0 keeps rho >= 1 - guaranteed positive
        without needing a floor clamp; A in (-1, 0) would allow a dip before
        the bump, if that combination shape is ever needed.
        """
        def _func(lag, params):
            A, lag_peak, k = params
            return 1.0 + A * peak_normalized_pulse(lag, lag_peak, k)
        return RenewalKernel(
            name="GammaBump", func=_func,
            param_names=["A_bump", "lag_peak", "k_shape"],
            initial_guesses=[0.5, 0.5, 2.0],
            bounds=[(0.0, 5.0), (0.05, 5.0), (0.5, 20.0)],
            latex_formula=r"$\rho(\Delta t) = 1 + A_{\text{bump}}\left(\frac{\Delta t}{\Delta t_{\text{peak}}}\right)^{k}e^{k(1-\Delta t/\Delta t_{\text{peak}})}$",
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
        self.name = f"Renewal rate: {kernel.name}, refractory: {renewal_kernel.name}"
        self.latex_formula = rf"{kernel.latex_formula} \times {renewal_kernel.latex_formula}"
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

    def _renewal_nll(
        self, t_events: np.ndarray, trial: float, duration_s: float,
        params_base: List[float], params_refractory: List[float],
    ) -> float:
        t_events = np.sort(t_events)
        n = len(t_events)

        if n == 0:
            return self.kernel.integrate(duration_s, trial, params_base, self.integration_dt)

        lam0 = max(self.kernel.evaluate(np.array([t_events[0]]), np.array([trial]), params_base)[0], 1e-12)
        sum_log_intensity = np.log(lam0)

        for i in range(1, n):
            dt = t_events[i] - t_events[i - 1]
            lam = self.kernel.evaluate(np.array([t_events[i]]), np.array([trial]), params_base)[0]
            rho = self.renewal_kernel.evaluate(np.array([dt]), params_refractory)[0]
            sum_log_intensity += np.log(max(lam * rho, 1e-12))

        total_integral = self._segment_integral(0.0, t_events[0], trial, params_base, None, None)
        for i in range(1, n):
            total_integral += self._segment_integral(
                t_events[i - 1], t_events[i], trial, params_base, t_events[i - 1], params_refractory
            )
        total_integral += self._segment_integral(
            t_events[-1], duration_s, trial, params_base, t_events[-1], params_refractory
        )

        return -(sum_log_intensity - total_integral)

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        params_base, params_refractory = self._split_params(params)
        total_nll = 0.0
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            m = dataset.unique_trials[t_idx]
            total_nll += self._renewal_nll(t_ev, m, dataset.duration_s, params_base, params_refractory)
        return total_nll

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray],
                history_events: Optional[np.ndarray] = None) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        params_base, params_refractory = self._split_params(self.params_)

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
        rho[valid] = self.renewal_kernel.evaluate(t_arr[valid] - t_last[valid], params_refractory)

        return np.maximum(base_rate * rho, 1e-9)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        n_trials, n_bins = dataset.num_trials, len(dataset.t_centers)
        rate_sum = np.zeros((n_trials, n_bins), dtype=float)
        active_count = np.zeros(n_trials, dtype=int)

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            trial_val = dataset.unique_trials[t_idx]
            stream_rate = self.predict(t=dataset.t_centers, trial=trial_val, history_events=t_ev)
            rate_sum[t_idx, :] += stream_rate
            active_count[t_idx] += 1

        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(active_count[:, None] > 0, rate_sum / np.maximum(active_count, 1)[:, None], 0.0)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        params_base, params_refractory = self._split_params(self.params_)
        t_events = np.sort(np.asarray(t_events))

        Lambda = np.zeros_like(t_events, dtype=float)
        Lambda[0] = self._segment_integral(0.0, t_events[0], trial, params_base, None, None) if len(t_events) else 0.0
        for i in range(1, len(t_events)):
            Lambda[i] = Lambda[i - 1] + self._segment_integral(
                t_events[i - 1], t_events[i], trial, params_base, t_events[i - 1], params_refractory
            )
        return Lambda