from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import trapezoid

from .dataset import PointProcessDataset
from .point_process import PointProcess, thin
from .kernel_shapes import peak_normalized_pulse
from .lag_kernel import LagKernel
from .poisson_process import RateKernel


@dataclass(frozen=True)
class RenewalKernel(LagKernel):
    """
    Multiplicative modulation rho(lag) >= 0 by the time since the stream's
    last event: suppressive (refractory), facilitating or bump-shaped.
    """


class RenewalKernelFactory:

    @staticmethod
    def exponential_recovery() -> RenewalKernel:
        """rho(0)=0, approaches 1 with time constant tau_r. Smooth refractory recovery."""

        def _func(lag, params):
            (tau_r,) = params
            return 1.0 - np.exp(-lag / tau_r)

        return RenewalKernel(
            name="ExponentialRecovery",
            func=_func,
            param_names=["tau_refractory"],
            initial_guesses=[0.15],
            bounds=[(0.001, 30.0)],
            latex_formula=r"$\rho(\Delta t) = 1 - e^{-\Delta t/\tau_r}$",
        )

    @staticmethod
    def exponential_excitation() -> RenewalKernel:
        """
        rho(0) = 1 + A_exc, decaying to 1 with time constant tau_exc. A_exc >= 0,
        so it can only facilitate: a test of excitation by the previous event.
        """

        def _func(lag, params):
            A_exc, tau_exc = params
            return 1.0 + A_exc * np.exp(-lag / tau_exc)

        return RenewalKernel(
            name="ExponentialExcitation",
            func=_func,
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
            name="HardAbsorption",
            func=_func,
            param_names=[],
            initial_guesses=[],
            bounds=[],
            integral_func=lambda duration, params: 0.0
            * np.asarray(duration, dtype=float),
            latex_formula=r"$\rho(\Delta t) = 0,\ \Delta t > 0$",
        )

    @staticmethod
    def refractory_delayed_excitation(
        shape_k: float = 2.0,
        tau_refractory_init: float = 0.10,
        excitation_init: float = 1.0,
        pulse_peak_lag_init: float = 0.25,
    ) -> RenewalKernel:
        """
        Smooth refractory recovery plus delayed excitation.

            rho(lag) = 1 - exp(-lag / tau_r)
                    + A_exc * pulse(lag; t_peak, k)

            pulse(lag; t_peak, k)
                = (lag / t_peak)^k
                * exp[k * (1 - lag / t_peak)]

        Properties
        ----------
        rho(0) = 0
            Complete instantaneous suppression.

        rho(lag) can exceed 1
            Delayed facilitation is possible.

        rho(lag) -> 1
            Event rate returns to the baseline rate at long lags.

        ``shape_k`` is fixed to reduce confounding between pulse width, location,
        and amplitude. ``pulse_peak_lag`` is the peak of the pulse component,
        not necessarily the exact maximum of the combined rho function.
        """
        if shape_k <= 0:
            raise ValueError("shape_k must be positive.")

        def _func(lag, params):
            tau_refractory, A_excitation, pulse_peak_lag = params

            recovery = 1.0 - np.exp(-lag / tau_refractory)

            pulse = peak_normalized_pulse(lag, pulse_peak_lag, shape_k)

            return recovery + A_excitation * pulse

        return RenewalKernel(
            name="RefractoryDelayedExcitation",
            func=_func,
            param_names=[
                "tau_refractory",
                "A_delayed_excitation",
                "pulse_peak_lag",
            ],
            initial_guesses=[
                tau_refractory_init,
                excitation_init,
                pulse_peak_lag_init,
            ],
            bounds=[
                (0.02, 1.0),
                (0.0, 20.0),
                (0.01, 2.0),
            ],
            latex_formula=(
                r"$\rho(\Delta t)=1-e^{-\Delta t/\tau_r}"
                r"+A_{\mathrm{exc}}"
                r"\left(\frac{\Delta t}{t_p}\right)^k"
                r"\exp\left[k\left(1-\frac{\Delta t}{t_p}\right)\right]$"
            ),
        )

    @staticmethod
    def delayed_excitation(
        shape_k: float = 2.0,
        excitation_init: float = 1.0,
        pulse_peak_lag_init: float = 1.0,
    ) -> RenewalKernel:
        """
        Delayed excitation without an estimated refractory-recovery term:

            rho(lag) = 1 + A_exc * pulse(lag; t_peak, k)

            pulse(lag; t_peak, k)
                = (lag / t_peak)^k
                * exp[k * (1 - lag / t_peak)]

        Properties
        ----------
        rho(0) = 1
            No fitted instantaneous suppression or excitation.

        rho(t_peak) = 1 + A_exc
            The pulse component reaches its maximum at ``t_peak``.

        rho(lag) -> 1
            The process returns to its baseline intensity at long lags.

        ``shape_k`` is fixed to reduce confounding among pulse width, location,
        and amplitude.
        """
        if shape_k <= 0:
            raise ValueError("shape_k must be positive.")

        def _func(lag, params):
            A_excitation, pulse_peak_lag = params

            pulse = peak_normalized_pulse(lag, pulse_peak_lag, shape_k)

            return 1.0 + A_excitation * pulse

        return RenewalKernel(
            name=f"DelayedExcitation(k={shape_k:g})",
            func=_func,
            param_names=[
                "A_delayed_excitation",
                "pulse_peak_lag",
            ],
            initial_guesses=[
                excitation_init,
                pulse_peak_lag_init,
            ],
            bounds=[
                (0.0, 20.0),
                (0.02, 5.0),
            ],
            latex_formula=(
                r"$\rho(\Delta t)="
                r"1+A_{\mathrm{exc}}"
                r"\left(\frac{\Delta t}{t_p}\right)^k"
                r"\exp\left[k\left(1-\frac{\Delta t}{t_p}\right)\right]$"
            ),
        )


    @staticmethod
    def dead_time_delayed_excitation(
        dead_time_s: float,
        shape_k: float = 2.0,
        excitation_init: float = 1.0,
        pulse_peak_lag_init: float = 1.0,
        floor: float = 1e-3,
    ) -> RenewalKernel:
        """
        delayed_excitation with a fixed dead time d after every event:

            rho(lag) = floor                      for lag < d
            rho(lag) = 1 + A_exc * pulse(lag)     for lag >= d

        d is a constant, not a parameter: the likelihood jumps at every
        observed gap as d moves. floor keeps a rare gap shorter than d from
        having zero likelihood.
        """
        excitation = RenewalKernelFactory.delayed_excitation(shape_k, excitation_init, pulse_peak_lag_init)
        return RenewalKernel(
            name=f"DeadTime({dead_time_s:g}s)+{excitation.name}",
            func=lambda lag, params: np.where(lag < dead_time_s, floor, excitation.func(lag, params)),
            param_names=excitation.param_names,
            initial_guesses=excitation.initial_guesses,
            bounds=excitation.bounds,
            latex_formula=(
                rf"$\rho(\Delta t)=[\Delta t\geq {dead_time_s:g}]\,("
                + excitation.latex_formula.strip("$").split("=", 1)[1]
                + ")$"
            ),
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

    def __init__(
        self,
        kernel: RateKernel,
        renewal_kernel: RenewalKernel,
        integration_dt: float = 0.02,
    ):
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

    def _segment_integral(
        self,
        t0: float,
        t1: float,
        trial: float,
        params_base: List[float],
        t_ref: Optional[float],
        params_refractory: List[float],
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

    @staticmethod
    def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
        """Weights such that dot(weights, values) equals trapezoid(values, grid)."""
        grid = np.asarray(grid, dtype=float)

        if len(grid) < 2:
            return np.zeros_like(grid)

        weights = np.empty_like(grid)
        weights[0] = 0.5 * (grid[1] - grid[0])
        weights[-1] = 0.5 * (grid[-1] - grid[-2])

        if len(grid) > 2:
            weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])

        return weights

    def _segment_grid(self, start: float, end: float) -> np.ndarray:
        """Integration grid on [start, end], including both endpoints."""
        if end <= start:
            return np.array([], dtype=float)

        grid = np.arange(start, end, self.integration_dt, dtype=float)

        if len(grid) == 0 or grid[0] != start:
            grid = np.insert(grid, 0, start)

        if grid[-1] < end:
            grid = np.append(grid, end)
        else:
            grid[-1] = end

        return grid

    def _stream_grid(self, t_events: np.ndarray, duration_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trapezoid grid of one stream, split at its events: times, weights,
        and age since the previous event (NaN before the first event).
        """
        bounds = np.concatenate([[0.0], t_events, [duration_s]])
        refs = np.concatenate([[np.nan], t_events])
        grids = [self._segment_grid(float(a), float(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        if not any(len(g) for g in grids):
            empty = np.empty(0)
            return empty, empty, empty
        times = np.concatenate(grids)
        weights = np.concatenate([self._trapezoid_weights(g) for g in grids])
        ages = np.concatenate([g - r for g, r in zip(grids, refs)])
        return times, weights, ages

    def _dataset_grid(self, dataset: PointProcessDataset) -> Dict[str, np.ndarray]:
        """All streams' grids and events in flat arrays (data only, so cached on the dataset)."""
        parts = {key: [] for key in ("times", "weights", "ages", "trials", "fish", "event_times", "event_lags", "event_trials", "event_fish")}
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            times, weights, ages = self._stream_grid(t_ev, dataset.duration_s)
            for key, value in zip(("times", "weights", "ages"), (times, weights, ages)):
                parts[key].append(value)
            parts["trials"].append(np.full(len(times), t_idx, dtype=float))
            parts["fish"].append(np.full(len(times), f_idx))
            parts["event_times"].append(t_ev)
            parts["event_lags"].append(np.diff(t_ev, prepend=np.nan))
            parts["event_trials"].append(np.full(len(t_ev), t_idx, dtype=float))
            parts["event_fish"].append(np.full(len(t_ev), f_idx))
        grid = {key: np.concatenate([np.empty(0)] + values) for key, values in parts.items()}
        grid["fish"], grid["event_fish"] = grid["fish"].astype(int), grid["event_fish"].astype(int)
        # Ages after an event mostly sit on the k * integration_dt lattice (segment
        # ends do not). The step past the lattice holds 1: no previous event.
        steps = np.rint(grid["ages"] / self.integration_dt)
        on_lattice = np.abs(grid["ages"] - steps * self.integration_dt) < 1e-9
        n_lattice = int(np.max(steps[on_lattice], initial=0)) + 1
        grid["lattice_step"] = np.where(on_lattice, steps, n_lattice).astype(np.int32)
        grid["n_lattice"] = n_lattice
        grid["off_lattice"] = np.flatnonzero(~on_lattice & ~np.isnan(grid["ages"]))
        return grid

    def _modulation(self, ages: np.ndarray, params_renewal: List[float]) -> np.ndarray:
        """renewal_kernel(age), and 1 where there is no previous event (age NaN)."""
        rho = np.ones(len(ages))
        prior = ~np.isnan(ages)
        rho[prior] = self.renewal_kernel.evaluate(ages[prior], params_renewal)
        return rho

    def _grid_modulation(self, grid: Dict[str, np.ndarray], params_renewal: List[float]) -> np.ndarray:
        """_modulation of the grid ages, evaluating the kernel once per lattice step."""
        lattice = self.renewal_kernel.evaluate(np.arange(grid["n_lattice"]) * self.integration_dt, params_renewal)
        rho = np.append(lattice, 1.0)[grid["lattice_step"]]
        off = grid["off_lattice"]
        rho[off] = self.renewal_kernel.evaluate(grid["ages"][off], params_renewal)
        return rho

    def _log_intensities_and_exposure(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Log intensity at every event, and the compensator per fish."""
        params_base, params_renewal = self._split_params(params)
        grid = dataset.cached(("renewal_grid", self.integration_dt), lambda: self._dataset_grid(dataset))
        integrand = (
            grid["weights"]
            * self.kernel.evaluate(grid["times"], grid["trials"], params_base)
            * self._grid_modulation(grid, params_renewal)
        )
        exposure = np.bincount(grid["fish"], weights=integrand, minlength=dataset.num_fish)
        event_rates = self.kernel.evaluate(grid["event_times"], grid["event_trials"], params_base) * self._modulation(
            grid["event_lags"], params_renewal
        )
        return np.log(np.maximum(event_rates, 1e-300)), exposure

    def _stream_integral_and_ll(
        self,
        t_events: np.ndarray,
        trial: float,
        duration_s: float,
        params_base: List[float],
        params_renewal: List[float],
    ) -> Tuple[float, float]:
        """Sum of log intensities at the events and compensator of one stream."""
        t_events = np.sort(np.asarray(t_events, dtype=float))
        event_rates = self.kernel.evaluate(t_events, np.full(len(t_events), trial, dtype=float), params_base)
        event_rates = event_rates * self._modulation(np.diff(t_events, prepend=np.nan), params_renewal)
        times, weights, ages = self._stream_grid(t_events, duration_s)
        integrand = weights * self.kernel.evaluate(times, np.full(len(times), trial, dtype=float), params_base)
        integral = float(np.dot(integrand, self._modulation(ages, params_renewal)))
        return float(np.sum(np.log(np.maximum(event_rates, 1e-300)))), integral

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        log_intensities, exposure = self._log_intensities_and_exposure(dataset, params)
        return -(float(np.sum(log_intensities)) - float(np.sum(exposure)))

    def predict(
        self,
        t: np.ndarray,
        trial: Union[float, np.ndarray],
        history_events: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._require_fitted()
        params_base, params_renewal = self._split_params(self.params_)

        base_rate = self.kernel.evaluate(t, trial, params_base)
        if history_events is None or len(history_events) == 0:
            return np.maximum(base_rate, 1e-9)

        t_arr = np.asarray(t, dtype=float)
        events_sorted = np.sort(np.asarray(history_events, dtype=float))
        # nearest preceding event for each evaluation point t
        idx = np.searchsorted(events_sorted, t_arr, side="right") - 1
        has_prior = idx >= 0
        t_last = np.where(has_prior, events_sorted[np.clip(idx, 0, None)], np.nan)

        rho = np.ones_like(t_arr, dtype=float)
        valid = has_prior & (t_arr > t_last)
        rho[valid] = self.renewal_kernel.evaluate(
            t_arr[valid] - t_last[valid], params_renewal
        )

        return np.maximum(base_rate * rho, 1e-9)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        n_trials, n_bins = dataset.num_trials, len(dataset.t_centers)
        rate_sum = np.zeros((n_trials, n_bins), dtype=float)
        active_count = np.zeros(n_trials, dtype=int)

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            stream_rate = self.predict(
                t=dataset.t_centers, trial=t_idx, history_events=t_ev
            )
            rate_sum[t_idx, :] += stream_rate
            active_count[t_idx] += 1

        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                active_count[:, None] > 0,
                rate_sum / np.maximum(active_count, 1)[:, None],
                0.0,
            )

    def cumulative_integrated_intensity(
        self, t_events: np.ndarray, trial: float
    ) -> np.ndarray:
        self._require_fitted()
        params_base, params_renewal = self._split_params(self.params_)
        t_events = np.sort(np.asarray(t_events))
        if len(t_events) == 0:
            return np.array([], dtype=float)

        Lambda = np.zeros_like(t_events, dtype=float)
        Lambda[0] = (
            self._segment_integral(0.0, t_events[0], trial, params_base, None, None)
            if len(t_events)
            else 0.0
        )
        for i in range(1, len(t_events)):
            Lambda[i] = Lambda[i - 1] + self._segment_integral(
                t_events[i - 1],
                t_events[i],
                trial,
                params_base,
                t_events[i - 1],
                params_renewal,
            )
        return Lambda

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        The intensity is multiplicative, so a fish gain scales all of it:
        sum of log intensities, events and compensator per fish (history
        resets every trial).
        """
        log_intensities, exposure = self._log_intensities_and_exposure(dataset, params)
        n_events = np.bincount(dataset.event_fish_idx, minlength=dataset.num_fish).astype(float)
        return float(np.sum(log_intensities)), n_events, exposure

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Thinning with gain * kernel(t) * renewal(t - last event). The bound
        includes the renewal kernel's maximum, which can exceed 1 (excitation).
        """
        params_base, params_renewal = self._split_params(self.params_)
        upper = gain * self._intensity_upper_bound(dataset, t_idx) * self._renewal_kernel_upper_bound(
            params_renewal, dataset
        )

        def intensity(t: float, events: List[float]) -> float:
            base = gain * self.kernel.evaluate(np.array([t]), np.array([t_idx]), params_base)[0]
            if not events:
                return base
            return base * self.renewal_kernel.evaluate(np.array([t - events[-1]]), params_renewal)[0]

        return thin(intensity, lambda t, events: upper, dataset.duration_s, rng)

    def _renewal_kernel_upper_bound(
        self, params_renewal, dataset: PointProcessDataset
    ) -> float:
        """
        Upper bound on renewal_kernel(lag) for lag in [0, duration_s] -- NOT
        an arbitrary fixed window. A lag between two events within one trial
        can never exceed duration_s by construction, so duration_s is the
        exact, principled ceiling to search over (not a guess) -- using a
        fixed literal here was a real correctness risk: any fitted timescale
        parameter approaching or exceeding that literal (e.g. after_looming's
        tau_slow, bounded up to 30.0) would have silently produced an
        UNDER-estimated bound, invalidating the thinning algorithm's
        accept/reject step.
        """
        lag_grid = np.arange(
            0.0, dataset.duration_s + self.integration_dt, self.integration_dt
        )
        vals = self.renewal_kernel.evaluate(lag_grid, params_renewal)
        return float(np.max(vals)) * self._THINNING_SAFETY_MARGIN

