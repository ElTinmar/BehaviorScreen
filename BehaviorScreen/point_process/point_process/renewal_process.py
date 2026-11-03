from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple, Union

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
    def burst_excitation(shape_k: float = 2.0, rate_init: float = 0.5, peak_lag_init: float = 0.4) -> RenewalKernel:
        """
        Extra rate (Hz) after an event, added to the stimulus-driven rate:

            E(lag) = A_burst * pulse(lag; t_peak, k),  pulse(t_peak) = 1

        Absolute, not relative to the driven rate: a fish's gain scales what the
        stimulus drives, not its bursts, so no fish can run away into endless bursts.
        """
        def _func(lag, params):
            A_burst, burst_peak_lag = params
            return A_burst * peak_normalized_pulse(lag, burst_peak_lag, shape_k)

        return RenewalKernel(
            name=f"Burst(k={shape_k:g})",
            func=_func,
            param_names=["A_burst", "burst_peak_lag"],
            initial_guesses=[rate_init, peak_lag_init],
            bounds=[(0.0, 20.0), (0.02, 5.0)],
            latex_formula=(
                r"$E(\Delta t)=A_{\mathrm{burst}}\left(\frac{\Delta t}{t_p}\right)^k"
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
    Modulated renewal process: a rate kernel of time in trial and trial number
    (as in PoissonProcess), plus optionally a burst rate after each event
    (selected bout), both times a recovery function of the time since the
    stream's last bout of any category or side (dataset.history):

        lambda(t | history) = renewal_kernel(t - t_last bout) * (kernel(t, m) + excitation(t - t_last event))

    The recovery describes refractoriness after any bout; the excitation
    (an absolute rate, e.g. RenewalKernelFactory.burst_excitation), bursts of the
    selected bouts such as hunting sequences of J-turns. Each is absent before the
    first bout or event of a trial. A fish gain scales kernel only
    (gain_scales_drive_only when there is an excitation).
    """

    _CHUNK: ClassVar[int] = 8192 #65536
    _TABLE_STEPS: ClassVar[int] = 4  # kernel table points per integration step

    def __init__(
        self,
        kernel: RateKernel,
        renewal_kernel: RenewalKernel,
        excitation: Optional[RenewalKernel] = None,
        integration_dt: float = 0.02,
    ):
        super().__init__(integration_dt)
        self.kernel, self.renewal_kernel, self.excitation = kernel, renewal_kernel, excitation
        parts = [kernel, renewal_kernel] + ([excitation] if excitation is not None else [])
        self.name = f"Renewal rate: {kernel.name}, history: {renewal_kernel.name}" + (
            f", events: {excitation.name}" if excitation is not None else ""
        )
        formula = lambda k: k.latex_formula.strip("$").split("=", 1)[-1]
        drive = formula(kernel) + (f" + {formula(excitation)}" if excitation is not None else "")
        self.latex_formula = rf"$\lambda = {formula(renewal_kernel)} \times \left({drive}\right)$"
        self.initial_guesses = sum((list(p.initial_guesses) for p in parts), [])
        self.bounds = sum((list(p.bounds) for p in parts), [])
        self.param_names = sum((list(p.param_names) for p in parts), [])

    @property
    def gain_scales_drive_only(self) -> bool:
        """A fish gain scales the stimulus-driven rate, not the bursts: not a gain on the whole intensity."""
        return self.excitation is not None

    def _split(self, params: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """(rate kernel, renewal kernel, excitation) parameters."""
        a = len(self.kernel.param_names)
        b = a + len(self.renewal_kernel.param_names)
        return list(params[:a]), list(params[a:b]), list(params[b:])

    def _excitation(self, lags: np.ndarray, params_excitation: List[float]) -> np.ndarray:
        """excitation(lag), and 0 without excitation or before the first event (lag NaN)."""
        result = np.zeros(np.shape(lags))
        prior = ~np.isnan(lags)
        if self.excitation is not None:
            result[prior] = self.excitation.evaluate(np.asarray(lags)[prior], params_excitation)
        return result

    def _segment_integrals(
        self, t0: float, t1: float, trial: float, params: List[float], t_bout: Optional[float], t_event: Optional[float]
    ) -> Tuple[float, float]:
        """Integrals over [t0, t1] of the driven and the burst intensity, the last bout at t_bout and the last event at t_event."""
        if t1 <= t0:
            return 0.0, 0.0
        params_base, params_renewal, params_excitation = self._split(params)
        grid = np.arange(t0, t1 + self.integration_dt, self.integration_dt)
        grid[-1] = min(grid[-1], t1)
        rho = np.ones_like(grid) if t_bout is None else self.renewal_kernel.evaluate(grid - t_bout, params_renewal)
        drive = rho * self.kernel.evaluate(grid, np.full_like(grid, trial), params_base)
        burst = rho * (self._excitation(grid - t_event, params_excitation) if t_event is not None else 0.0 * grid)
        return trapezoid(drive, grid), trapezoid(burst, grid)

    def _segment_integral(self, t0: float, t1: float, trial: float, params: List[float], t_bout: Optional[float], t_event: Optional[float]) -> float:
        return sum(self._segment_integrals(t0, t1, trial, params, t_bout, t_event))

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

    def _stream_grid(
        self, bouts: np.ndarray, events: np.ndarray, duration_s: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Trapezoid grid of one stream, split at its bouts: times, weights, time
        since the last bout and since the last event (events are bouts; NaN before the first).
        """
        bounds = np.concatenate([[0.0], bouts, [duration_s]])
        grids = [self._segment_grid(float(a), float(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        if not any(len(g) for g in grids):
            empty = np.empty(0)
            return empty, empty, empty, empty
        last_event = np.searchsorted(events, bounds[:-1], side="right") - 1  # last event at or before each segment start
        event_refs = np.where(last_event >= 0, events[np.maximum(last_event, 0)] if len(events) else np.nan, np.nan)
        bout_refs = np.concatenate([[np.nan], bouts])
        times = np.concatenate(grids)
        weights = np.concatenate([self._trapezoid_weights(g) for g in grids])
        bout_ages = np.concatenate([g - r for g, r in zip(grids, bout_refs)])
        event_ages = np.concatenate([g - r for g, r in zip(grids, event_refs)])
        return times, weights, bout_ages, event_ages

    def _table_positions(self, values: np.ndarray, columns: int) -> Tuple[np.ndarray, np.ndarray]:
        """Table point at or before each value (on the _table_dt step) and the fraction past it."""
        position = values / self._table_dt
        start = np.minimum(np.floor(position), columns - 2)
        return start, (position - start).astype(np.float32)

    def _dataset_grid(self, dataset: PointProcessDataset) -> Dict[str, np.ndarray]:
        """All streams' grids and events in flat arrays (data only, so cached on the dataset)."""
        keys = ("times", "weights", "ages", "event_ages", "trials", "stream", "event_times", "event_lags", "event_event_lags", "event_trials", "event_fish")
        parts = {key: [] for key in keys}
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            bouts = dataset.history_times(f_idx, t_idx)
            times, weights, ages, event_ages = self._stream_grid(bouts, t_ev, dataset.duration_s)
            for key, value in zip(("times", "weights", "ages", "event_ages"), (times, weights, ages, event_ages)):
                parts[key].append(value)
            parts["trials"].append(np.full(len(times), t_idx, dtype=float))
            parts["stream"].append(np.full(len(times), f_idx * dataset.num_trials + t_idx))
            parts["event_times"].append(t_ev)
            previous = np.searchsorted(bouts, t_ev) - 1  # the bout before each event
            parts["event_lags"].append(np.where(previous >= 0, t_ev - bouts[np.maximum(previous, 0)], np.nan))
            parts["event_event_lags"].append(np.diff(t_ev, prepend=np.nan))
            parts["event_trials"].append(np.full(len(t_ev), t_idx, dtype=float))
            parts["event_fish"].append(np.full(len(t_ev), f_idx))
        grid = {key: np.concatenate([np.empty(0)] + values) for key, values in parts.items()}
        grid["stream"], grid["event_fish"] = grid["stream"].astype(np.int32), grid["event_fish"].astype(int)  # events by fish, in order
        grid["weights"] = grid["weights"].astype(np.float32)
        # Ages after a bout mostly sit on the k * integration_dt lattice (segment
        # ends do not). The step past the lattice holds 1: no previous bout.
        steps = np.rint(grid["ages"] / self.integration_dt)
        on_lattice = np.abs(grid["ages"] - steps * self.integration_dt) < 1e-9
        n_lattice = int(np.max(steps[on_lattice], initial=0)) + 1
        grid["lattice_step"] = np.where(on_lattice, steps, n_lattice).astype(np.int32)
        grid["n_lattice"] = n_lattice
        grid["off_lattice"] = np.flatnonzero(~on_lattice & ~np.isnan(grid["ages"]))
        grid["off_ages"] = grid.pop("ages")[grid["off_lattice"]]
        # The rate kernel (trial x time) and the excitation (time since the last event)
        # are interpolated on the grid from tables _TABLE_STEPS times finer than it.
        columns = self._table_columns(dataset)
        start, grid["table_fraction"] = self._table_positions(grid.pop("times"), columns)
        grid["table_index"] = (grid.pop("trials") * columns + start).astype(np.int32)
        event_ages = grid.pop("event_ages")
        if self.excitation is not None:
            none = np.isnan(event_ages)  # before the first event: the table's last points hold 0
            start, fraction = self._table_positions(np.where(none, 0.0, event_ages), columns)
            grid["excitation_index"] = np.where(none, columns, start).astype(np.int32)
            grid["excitation_fraction"] = np.where(none, 0.0, fraction).astype(np.float32)
        return grid

    @property
    def _table_dt(self) -> float:
        return self.integration_dt / self._TABLE_STEPS

    def _table_columns(self, dataset: PointProcessDataset) -> int:
        return int(np.ceil(dataset.duration_s / self._table_dt)) + 2

    def _rate_table(self, dataset: PointProcessDataset, params_base: List[float]) -> np.ndarray:
        """kernel(t, m) at every table time and trial, flattened row-major (trial x time)."""
        t = np.arange(self._table_columns(dataset)) * self._table_dt
        return self.kernel.evaluate(t[None, :], np.arange(dataset.num_trials)[:, None], params_base).ravel()

    def _modulation(self, ages: np.ndarray, params_renewal: List[float]) -> np.ndarray:
        """renewal_kernel(age), and 1 where there is no previous bout (age NaN)."""
        rho = np.ones(len(ages))
        prior = ~np.isnan(ages)
        rho[prior] = self.renewal_kernel.evaluate(ages[prior], params_renewal)
        return rho

    def _drive_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        At every event (by fish, in order) the driven and the burst rate, its fish,
        and per stream (fish x trial) the integrals of both. The grid is processed
        in chunks that stay in the CPU cache.
        """
        params_base, params_renewal, params_excitation = self._split(params)
        grid = dataset.cached(
            ("renewal_grid", self.integration_dt, self.excitation is not None), lambda: self._dataset_grid(dataset)
        )
        lattice = self.renewal_kernel.evaluate(np.arange(grid["n_lattice"]) * self.integration_dt, params_renewal)
        lattice = np.append(lattice, 1.0)
        off, off_rho = grid["off_lattice"], self.renewal_kernel.evaluate(grid["off_ages"], params_renewal)
        table = self._rate_table(dataset, params_base)
        if self.excitation is not None:
            lags = np.arange(self._table_columns(dataset)) * self._table_dt
            excitation = np.append(self.excitation.evaluate(lags, params_excitation), [0.0, 0.0])
        n_streams = dataset.num_fish * dataset.num_trials
        drive_exposure, burst_exposure = np.zeros(n_streams), np.zeros(n_streams)
        interpolate = lambda values, index, fraction: values[index] + fraction * (values[index + 1] - values[index])
        for a in range(0, len(grid["weights"]), self._CHUNK):
            b = a + self._CHUNK
            rho = lattice[grid["lattice_step"][a:b]]
            lo, hi = np.searchsorted(off, [a, b])
            rho[off[lo:hi] - a] = off_rho[lo:hi]
            weight, stream = grid["weights"][a:b] * rho, grid["stream"][a:b]
            drive = weight * interpolate(table, grid["table_index"][a:b], grid["table_fraction"][a:b])
            drive_exposure += np.bincount(stream, weights=drive, minlength=n_streams)
            if self.excitation is not None:
                burst = weight * interpolate(excitation, grid["excitation_index"][a:b], grid["excitation_fraction"][a:b])
                burst_exposure += np.bincount(stream, weights=burst, minlength=n_streams)
        rho_events = self._modulation(grid["event_lags"], params_renewal)
        drive_rates = rho_events * self.kernel.evaluate(grid["event_times"], grid["event_trials"], params_base)
        burst_rates = rho_events * self._excitation(grid["event_event_lags"], params_excitation)
        shape = (dataset.num_fish, dataset.num_trials)
        return drive_rates, burst_rates, grid["event_fish"], drive_exposure.reshape(shape), burst_exposure.reshape(shape)

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        drive_rates, burst_rates, _, drive_exposure, burst_exposure = self._drive_terms(dataset, params)
        log_rates = np.log(np.maximum(drive_rates + burst_rates, 1e-300))
        return -(float(np.sum(log_rates)) - float(drive_exposure.sum() + burst_exposure.sum()))

    def drive_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        For a gain on the driven rate only: a constant log-likelihood term, the
        driven and burst rates at every event and its fish (non-decreasing), and
        the driven and burst exposure per fish.
        """
        drive_rates, burst_rates, fish, drive_exposure, burst_exposure = self._drive_terms(dataset, params)
        return 0.0, drive_rates, burst_rates, fish, drive_exposure.sum(axis=1), burst_exposure.sum(axis=1)

    def stream_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Sum of log intensities, and events and compensator per stream (fish x trial); a gain on the whole intensity."""
        if self.excitation is not None:
            raise ValueError(f"{self.name}: a gain scales its driven rate only (drive_likelihood_terms)")
        drive_rates, _, _, drive_exposure, _ = self._drive_terms(dataset, params)
        return float(np.sum(np.log(np.maximum(drive_rates, 1e-300)))), dataset.event_counts_per_stream(), drive_exposure

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Terms per fish of a gain on the whole intensity (no excitation)."""
        log_rates, n_events, exposure = self.stream_likelihood_terms(dataset, params)
        return log_rates, n_events.sum(axis=1), exposure.sum(axis=1)

    def _stream_integral_and_ll(
        self, t_events: np.ndarray, trial: float, duration_s: float, params_base: List[float], params_renewal: List[float]
    ) -> Tuple[float, float]:
        """Sum of log intensities at the events and compensator of one stream whose bouts are its events (no excitation)."""
        t_events = np.sort(np.asarray(t_events, dtype=float))
        event_rates = self.kernel.evaluate(t_events, np.full(len(t_events), trial, dtype=float), params_base)
        event_rates = event_rates * self._modulation(np.diff(t_events, prepend=np.nan), params_renewal)
        times, weights, ages, _ = self._stream_grid(t_events, t_events, duration_s)
        integrand = weights * self.kernel.evaluate(times, np.full(len(times), trial, dtype=float), params_base)
        return float(np.sum(np.log(np.maximum(event_rates, 1e-300)))), float(np.dot(integrand, self._modulation(ages, params_renewal)))

    def predict(
        self,
        t: np.ndarray,
        trial: Union[float, np.ndarray],
        history_events: Optional[np.ndarray] = None,
        selected_events: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Intensity at t (gain 1) given the stream's bouts (history_events) and its events (selected_events)."""
        self._require_fitted()
        params_base, params_renewal, params_excitation = self._split(self.params_)
        t = np.asarray(t, dtype=float)

        def since_last(points: Optional[np.ndarray]) -> np.ndarray:
            points = np.sort(np.asarray([] if points is None else points, dtype=float))
            k = np.searchsorted(points, t, side="right") - 1
            return np.where(k >= 0, t - points[np.maximum(k, 0)] if len(points) else np.nan, np.nan)

        rho = self._modulation(since_last(history_events).ravel(), params_renewal).reshape(t.shape)
        rate = rho * (self.kernel.evaluate(t, trial, params_base) + self._excitation(since_last(selected_events), params_excitation))
        return np.maximum(rate, 1e-9)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        n_trials, n_bins = dataset.num_trials, len(dataset.t_centers)
        rate_sum = np.zeros((n_trials, n_bins), dtype=float)
        active_count = np.zeros(n_trials, dtype=int)
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            rate_sum[t_idx, :] += self.predict(dataset.t_centers, t_idx, dataset.history_times(f_idx, t_idx), t_ev)
            active_count[t_idx] += 1
        return np.where(active_count[:, None] > 0, rate_sum / np.maximum(active_count, 1)[:, None], 0.0)

    def compensator_parts(
        self, t_events: np.ndarray, trial: float, history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Driven and burst compensators at each of t_events: every point but the
        last is an event (a bout). history: the stream's bouts (default: those events).
        """
        self._require_fitted()
        t_events = np.sort(np.asarray(t_events, dtype=float))
        events = t_events[:-1]
        bouts = events if history is None else np.asarray(history, dtype=float)
        edges = np.union1d(bouts, t_events)
        starts = np.concatenate([[0.0], edges[:-1]])
        last_event = np.searchsorted(events, starts, side="right") - 1
        segments = np.array([
            self._segment_integrals(a, b, trial, self.params_, a if i else None, events[k] if k >= 0 else None)
            for i, (a, b, k) in enumerate(zip(starts, edges, last_event))
        ]).reshape(-1, 2)
        positions = np.searchsorted(edges, t_events)
        return np.cumsum(segments[:, 0])[positions], np.cumsum(segments[:, 1])[positions]

    def cumulative_integrated_intensity(
        self, t_events: np.ndarray, trial: float, history: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.add(*self.compensator_parts(t_events, trial, history))

    def stream_compensator_profile(
        self, t_ev: np.ndarray, trial: float, duration_s: float, history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool, float]:
        probes = np.append(np.sort(np.asarray(t_ev, dtype=float)), float(duration_s))
        cumulative = self.cumulative_integrated_intensity(probes, trial, history=history)
        return probes, cumulative, True, float(cumulative[-1])

    def drive_stream_profile(
        self, t_ev: np.ndarray, trial: float, duration_s: float, history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For a gain on the driven rate only: driven and burst compensators at the events and trial end, and both rates at the events."""
        t_ev = np.sort(np.asarray(t_ev, dtype=float))
        drive_cum, burst_cum = self.compensator_parts(np.append(t_ev, float(duration_s)), trial, history)
        bouts = t_ev if history is None else np.asarray(history, dtype=float)
        previous = np.searchsorted(bouts, t_ev) - 1
        _, params_renewal, params_excitation = self._split(self.params_)
        rho = self._modulation(np.where(previous >= 0, t_ev - bouts[np.maximum(previous, 0)], np.nan), params_renewal)
        drive_rates = rho * self.kernel.evaluate(t_ev, np.full(t_ev.shape, trial, dtype=float), self._split(self.params_)[0])
        return drive_cum, burst_cum, drive_rates, rho * self._excitation(np.diff(t_ev, prepend=np.nan), params_excitation)

    @property
    def uses_history(self) -> bool:
        return True

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator,
        others: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Thinning of renewal * (gain * kernel + excitation), the last bout being the
        last simulated event or the last of `others` (the stream's observed bouts
        that are not events), whichever is later; the last event, the last simulated one.
        """
        others = np.empty(0) if others is None else np.sort(others)
        params_base, params_renewal, params_excitation = self._split(self.params_)
        lags = np.arange(0.0, dataset.duration_s + self.integration_dt, self.integration_dt)
        upper = (
            max(1.0, float(np.max(self.renewal_kernel.evaluate(lags, params_renewal))))
            * (gain * self._intensity_upper_bound(dataset, t_idx) + float(np.max(self._excitation(lags, params_excitation))) * self._THINNING_SAFETY_MARGIN)
        )

        def intensity(t: float, events: List[float]) -> float:
            k = np.searchsorted(others, t)
            last_bout = max(events[-1] if events else -np.inf, others[k - 1] if k else -np.inf)
            rate = gain * self.kernel.evaluate(np.array([t]), np.array([t_idx]), params_base)[0]
            if events:
                rate += self._excitation(np.array([t - events[-1]]), params_excitation)[0]
            if last_bout > -np.inf:
                rate *= self.renewal_kernel.evaluate(np.array([t - last_bout]), params_renewal)[0]
            return rate

        return thin(intensity, lambda t, events: upper, dataset.duration_s, rng)
