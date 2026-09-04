from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
from scipy.integrate import trapezoid

from .dataset import PointProcessDataset
from .point_process import PointProcess
from .poisson_process import RateKernel

@dataclass(frozen=True)
class HistoryKernel:
    name: str
    func: Callable[[np.ndarray, List[float]], np.ndarray]
    param_names: List[str]
    initial_guesses: List[float]
    bounds: List[Tuple[Optional[float], Optional[float]]]
    integral_func: Optional[Callable[[np.ndarray, List[float]], np.ndarray]] = None
    event_history_func: Optional[Callable[[np.ndarray, List[float]], np.ndarray]] = None
    latex_formula: str = ""

    def evaluate(self, lag: np.ndarray, params: List[float]) -> np.ndarray:
        if np.any(lag < 0):
            raise ValueError("History-kernel lags must be non-negative.")

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

    def event_history(self, t_events: np.ndarray, params: List[float]) -> np.ndarray:

        t_events = np.sort(np.asarray(t_events, dtype=float))
        if len(t_events) == 0:
            return np.array([], dtype=float)

        if self.event_history_func is not None:
            return self.event_history_func(t_events, params)

        # Generic O(N²) implementation
        history = np.zeros(len(t_events), dtype=float)
        for i in range(1, len(t_events)):
            lags = t_events[i] - t_events[:i]
            history[i] = np.sum(self.evaluate(lags, params))

        return history

class HistoryKernelFactory:

    @staticmethod
    def exponential(
        alpha_initial: float = 0.1,
        beta_initial: float = 10.0,
        alpha_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        beta_bounds: Tuple[Optional[float], Optional[float]] = (1e-3, None),
    ) -> HistoryKernel:

        def _func(lag, params):
            alpha, beta = params
            return alpha * np.exp(-beta * lag)

        def _integral(duration, params):
            alpha, beta = params
            return alpha / beta*(1.0 - np.exp(-beta * duration))

        def _event_history(t_events, params):
            """Fast O(N) recursive implementation"""
            alpha, beta = params

            n_events = len(t_events)
            if n_events == 0:
                return np.array([], dtype=float)

            history = np.zeros(n_events, dtype=float)
            R = 0.0
            for i in range(1, n_events):
                dt = t_events[i] - t_events[i - 1]
                R = np.exp(-beta * dt) * (1.0 + R)
                history[i] = alpha * R

            return history

        return HistoryKernel(
            name="Exponential",
            func=_func,
            param_names=["alpha_hawkes","beta_hawkes"],
            initial_guesses=[alpha_initial,beta_initial],
            bounds=[alpha_bounds,beta_bounds],
            integral_func=_integral,
            event_history_func=_event_history,
            latex_formula=r"$h(\Delta t) = \alpha_{\mathrm{H}} e^{-\beta_{\mathrm{H}}\Delta t}$"
        )


class HawkesProcess(PointProcess):

    def __init__(
        self,
        kernel: RateKernel,
        history_kernel: HistoryKernel,
        integration_dt=0.02,
    ):
        super().__init__(integration_dt)

        self.name = f"Hawkes rate: {kernel.name}, history: {history_kernel.name}"
        kernel_formula = kernel.latex_formula.strip("$")
        history_formula = history_kernel.latex_formula.strip("$")
        self.latex_formula = rf"${kernel_formula} + {history_formula}$"
        self.kernel = kernel
        self.history_kernel = history_kernel
        self.initial_guesses = kernel.initial_guesses + history_kernel.initial_guesses
        self.bounds = kernel.bounds + history_kernel.bounds
        self.param_names = kernel.param_names + history_kernel.param_names

    def _split_params(self,params: List[float]) -> Tuple[List[float], List[float]]:
        n = len(self.kernel.param_names)
        params_base = params[:n]
        params_history = params[n:]
        return params_base, params_history
    
    def _hawkes_nll(
        self,
        t_events: np.ndarray,
        trial: float,
        duration_s: float,
        params_base: List[float],
        params_history: List[float],
    ) -> float:

        trials_arr = np.full(t_events.shape, trial, dtype=float)
        base_rates = self.kernel.evaluate(
            t_events,
            trials_arr,
            params_base,
        )
        history_rates = self.history_kernel.event_history(
            t_events,
            params_history,
        )
        intensity = base_rates + history_rates
        sum_log_intensity = np.sum(np.log(intensity))

        base_integral = self.kernel.integrate(
            duration_s, 
            trial, 
            params_base,
            integration_dt=self.integration_dt,
        )
        remaining_time = duration_s - t_events
        history_integrals = self.history_kernel.integrate(
            remaining_time,
            params_history,
            integration_dt=self.integration_dt,
        )
        total_history_integral = np.sum(history_integrals)

        return -(sum_log_intensity - (base_integral + total_history_integral))

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        params_base, params_history = self._split_params(params)
        total_nll = 0.0

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            total_nll += self._hawkes_nll(
                t_ev, t_idx, dataset.duration_s, params_base, params_history
            )

        return total_nll

    def cumulative_integrated_intensity(
        self,
        t_events: np.ndarray,
        trial: float,
    ) -> np.ndarray:

        if self.params_ is None:
            raise ValueError("Model must be fitted first.")

        params_base, params_history = self._split_params(self.params_)
        t_events = np.sort(np.asarray(t_events))

        # Baseline cumulative intensity
        Lambda_base = self.kernel.cumulative_integrate(
            t_events=t_events,
            trial=trial,
            params=params_base,
            integration_dt=self.integration_dt,
        )

        # Hawkes history contribution
        Lambda_history = np.zeros_like(t_events)
        for i, t in enumerate(t_events):
            previous_events = t_events[:i]
            lags = t - previous_events
            Lambda_history[i] = np.sum(
                self.history_kernel.integrate(
                    lags,
                    params_history,
                    integration_dt=self.integration_dt,
                )
            )

        return Lambda_base + Lambda_history

    def predict(
        self,
        t: np.ndarray,
        trial: Union[int, np.ndarray],
        history_events: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")

        params_base, params_history = self._split_params(self.params_)

        t_arr = np.asarray(t, dtype=float)
        trial_arr = np.asarray(trial)
        t_bcast, trial_bcast = np.broadcast_arrays(t_arr, trial_arr)
        
        base_rate = self.kernel.evaluate(t_bcast, trial_bcast, params_base)

        if history_events is None or len(history_events) == 0:
            return base_rate

        # Vectorized lag evaluation: (N_eval x N_events)
        t_flat = t_bcast.ravel()
        events_sorted = np.sort(np.asarray(history_events, dtype=float))

        lags = t_flat[:, None] - events_sorted[None, :]
        mask = lags > 0.0

        history_rate_flat = np.zeros_like(t_flat, dtype=float)
        if np.any(mask):
            eval_lags = lags[mask]
            vals = self.history_kernel.evaluate(eval_lags, params_history)
            history_rate_flat += np.bincount(
                np.where(mask)[0], weights=vals, minlength=len(t_flat)
            )

        history_rate = history_rate_flat.reshape(t_bcast.shape)
        return base_rate + history_rate

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        n_trials = dataset.num_trials
        n_bins = len(dataset.t_centers)

        rate_sum = np.zeros((n_trials, n_bins), dtype=float)
        active_count = np.zeros(n_trials, dtype=int)

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            stream_rate = self.predict(t=dataset.t_centers, trial=t_idx, history_events=t_ev)
            rate_sum[t_idx, :] += stream_rate
            active_count[t_idx] += 1

        with np.errstate(invalid='ignore', divide='ignore'):
            expected_rate = np.where(
                active_count[:, None] > 0,
                rate_sum / np.maximum(active_count, 1)[:, None],
                0.0,
            )

        return expected_rate

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:

        params_base, params_history = self._split_params(params)

        N_f = np.zeros(dataset.num_fish, dtype=float)
        S_f = np.zeros(dataset.num_fish, dtype=float)
        base_ll = 0.0

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            N_f[f_idx] += len(t_ev)

            if len(t_ev) > 0:
                trials_arr = np.full(t_ev.shape, t_idx, dtype=float)
                base_rates = self.kernel.evaluate(t_ev, trials_arr, params_base)
                history_rates = self.history_kernel.event_history(t_ev, params_history)
                intensity = np.maximum(base_rates + history_rates, 1e-12)
                base_ll += float(np.sum(np.log(intensity)))

            base_integral = self.kernel.integrate(
                dataset.duration_s, t_idx, params_base, integration_dt=self.integration_dt,
            )
            remaining_time = dataset.duration_s - t_ev
            history_integrals = self.history_kernel.integrate(
                remaining_time, params_history, integration_dt=self.integration_dt,
            )
            S_f[f_idx] += base_integral + float(np.sum(history_integrals))

        return base_ll, N_f, S_f

    def _base_exposure_for_stream(self, dataset: PointProcessDataset, t_idx: int) -> float:
        """self.params_ is [kernel_params, history_params] concatenated --
        only the base kernel's own integral is relevant here (this method
        only ever needs the BASE rate exposure, consistent with how
        generate_model_predicted_counts's approximation is documented:
        it does not simulate genuine self-excitation dynamics)."""
        base_params, _ = self._split_params(self.params_)
        return self.kernel.integrate(dataset.duration_s, t_idx, base_params, self.integration_dt)