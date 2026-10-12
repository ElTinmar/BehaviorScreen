from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Union

import numpy as np

from .dataset import PointProcessDataset
from .point_process import PointProcess, thin
from .lag_kernel import LagKernel
from .poisson_process import RateKernel


@dataclass(frozen=True)
class HistoryKernel(LagKernel):
    event_history_func: Optional[Callable[[np.ndarray, List[float]], np.ndarray]] = None

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

    def decay_envelope(self, lag_grid: np.ndarray, params: List[float]) -> np.ndarray:
        """
        Running supremum of evaluate(lag, params) for lag' >= lag, evaluated
        on an ascending lag_grid. A valid upper bound on the kernel's future
        values regardless of shape (monotonic, delayed/bump-shaped, etc.) --
        used by HawkesProcess.simulate_stream's thinning envelope so no
        assumption about kernel monotonicity is required.
        """
        vals = self.evaluate(lag_grid, params)
        return np.maximum.accumulate(vals[::-1])[::-1]


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
            return alpha / beta * (1.0 - np.exp(-beta * duration))

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
            param_names=["alpha_hawkes", "beta_hawkes"],
            initial_guesses=[alpha_initial, beta_initial],
            bounds=[alpha_bounds, beta_bounds],
            integral_func=_integral,
            event_history_func=_event_history,
            latex_formula=r"$h(\Delta t) = \alpha_{\mathrm{H}} e^{-\beta_{\mathrm{H}}\Delta t}$",
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

    def _hawkes_nll(
        self,
        t_events: np.ndarray,
        trial: float,
        base_integral: float,
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

        remaining_time = duration_s - t_events
        history_integrals = self.history_kernel.integrate(
            remaining_time,
            params_history,
            integration_dt=self.integration_dt,
        )
        total_history_integral = np.sum(history_integrals)

        return -(sum_log_intensity - (base_integral + total_history_integral))

    def _trial_integrals(self, dataset: PointProcessDataset, params_base: List[float]) -> np.ndarray:
        return self.kernel.integrate(
            dataset.duration_s,
            np.arange(dataset.num_trials),
            params_base,
            integration_dt=self.integration_dt,
        )

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        params_base, params_history = self._split_params(params)
        trial_integrals = self._trial_integrals(dataset, params_base)
        total_nll = 0.0

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            total_nll += self._hawkes_nll(
                t_ev,
                t_idx,
                trial_integrals[t_idx],
                dataset.duration_s,
                params_base,
                params_history,
            )

        return total_nll

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        return np.add(*self._compensator_parts(t_events, trial))

    def _compensator_parts(self, t_events: np.ndarray, trial: float) -> Tuple[np.ndarray, np.ndarray]:
        """Baseline and history compensators at sorted t_events, each earlier one an event."""
        self._require_fitted()
        params_base, params_history = self._split_params(self.params_)
        t_events = np.sort(np.asarray(t_events, dtype=float))
        base = self.kernel.cumulative_integrate(t_events, trial, params_base, integration_dt=self.integration_dt)
        history = np.array(
            [
                np.sum(self.history_kernel.integrate(t - t_events[:i], params_history, integration_dt=self.integration_dt))
                for i, t in enumerate(t_events)
            ]
        )
        return base, history.reshape(base.shape)

    def predict(
        self,
        t: np.ndarray,
        trial: Union[int, np.ndarray],
        history_events: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        self._require_fitted()

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
            stream_rate = self.predict(
                t=dataset.t_centers, trial=t_idx, history_events=t_ev
            )
            rate_sum[t_idx, :] += stream_rate
            active_count[t_idx] += 1

        with np.errstate(invalid="ignore", divide="ignore"):
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
        trial_integrals = self._trial_integrals(dataset, params_base)

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

            remaining_time = dataset.duration_s - t_ev
            history_integrals = self.history_kernel.integrate(
                remaining_time,
                params_history,
                integration_dt=self.integration_dt,
            )
            S_f[f_idx] += trial_integrals[t_idx] + float(np.sum(history_integrals))

        return base_ll, N_f, S_f

    def _estimate_decay_horizon(
        self, hist_params: List[float], tol: float = 1e-6, max_horizon: float = 1000.0
    ) -> float:
        """
        Smallest lag beyond which history_kernel's value is < tol times its
        peak (i.e. genuinely negligible, not just "small"). Used by
        simulate_stream to build a thinning envelope that correctly
        represents the kernel decaying to ~ZERO -- NOT a grid that stops at
        an arbitrary point (e.g. dataset.duration_s) and silently floors
        every lag beyond it at whatever nonzero value the grid happened to
        end on.
        """
        probe = np.arange(0.0, max_horizon, self.integration_dt)
        vals = self.history_kernel.evaluate(probe, hist_params)
        peak = vals[0] if len(vals) else 0.0
        if peak <= 0:
            return self.integration_dt
        below = np.where(vals < tol * peak)[0]
        return float(probe[below[0]]) if len(below) else max_horizon

    def _history_envelope(self, hist_params: List[float]) -> Tuple[float, np.ndarray]:
        """Decay horizon and thinning envelope on an integration_dt lag grid (last value cached)."""
        key = tuple(np.asarray(hist_params, dtype=float))
        cached = self.__dict__.get("_envelope_cache")
        if cached is None or cached[0] != key:
            horizon = self._estimate_decay_horizon(hist_params)
            lag_grid = np.arange(0.0, horizon + self.integration_dt, self.integration_dt)
            envelope = (
                self.history_kernel.decay_envelope(lag_grid, hist_params)
                * self._THINNING_SAFETY_MARGIN
            )
            cached = self._envelope_cache = (key, (horizon, envelope))
        return cached[1]

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Thinning with the gain on the whole intensity (baseline and history)."""
        return thin(*self._thinning_functions(dataset, t_idx, gain, gain), dataset.duration_s, rng)

    def _thinning_functions(
        self, dataset: PointProcessDataset, t_idx: int, base_gain: float, history_gain: float
    ) -> Tuple[Callable[[float, List[float]], float], Callable[[float, List[float]], float]]:
        """
        Intensity and upper bound for thinning. The bound uses the history
        kernel's decay envelope (running supremum), so it holds for any
        kernel shape; events older than the decay horizon contribute 0.
        """
        base_params, hist_params = self._split_params(self.params_)
        base_upper = base_gain * self._intensity_upper_bound(dataset, t_idx)
        decay_horizon, envelope = self._history_envelope(hist_params)
        dt = self.integration_dt

        def envelope_at(lag: float) -> float:
            return 0.0 if lag >= decay_horizon else envelope[max(min(int(lag / dt), len(envelope) - 1), 0)]

        def intensity(t: float, events: List[float]) -> float:
            base = base_gain * self.kernel.evaluate(np.array([t]), np.array([t_idx]), base_params)[0]
            history = float(np.sum(self.history_kernel.evaluate(t - np.asarray(events), hist_params))) if events else 0.0
            return base + history_gain * history

        def upper_bound(t: float, events: List[float]) -> float:
            return base_upper + history_gain * sum(envelope_at(t - e) for e in events)

        return intensity, upper_bound
