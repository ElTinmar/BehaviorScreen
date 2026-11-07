"""PointProcess: parameters, likelihood interface, compensator and simulation."""
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backend import Arrays, compiled, jnp
from .dataset import PointProcessDataset
from .diagnostics import DiagnosticsMixin
from .gof import GOFMixin
from .inference import InferenceMixin

StreamTaus = Dict[Tuple[int, int], List[Tuple[float, bool]]]


def thin(
    intensity: Callable[[np.ndarray, List[float]], np.ndarray],
    upper_bound: Callable[[float, List[float]], float],
    duration_s: float,
    rng: np.random.Generator,
    first_only: bool = False,
    memoryless: bool = False,
    batch: int = 64,
) -> np.ndarray:
    """
    Ogata thinning: event times on [0, duration_s) of a process whose
    conditional intensity(times, events so far) stays below upper_bound(t, events)
    from t until the next event. Candidates are drawn and tested in batches:
    the intensity at each depends only on the events before it, so the first
    accepted one is the next event and the later ones are discarded. memoryless:
    the intensity ignores the events, so all candidates are tested at once.
    """
    if memoryless:
        upper = max(upper_bound(0.0, []), 1e-12)
        candidates = np.sort(rng.uniform(0.0, duration_s, rng.poisson(upper * duration_s)))
        events = candidates[rng.uniform(size=len(candidates)) * upper <= intensity(candidates, [])]
        return events[:1] if first_only else events
    events: List[float] = []
    t = 0.0
    while True:
        upper = max(upper_bound(t, events), 1e-12)
        candidates = t + np.cumsum(rng.exponential(1.0 / upper, batch))
        candidates = candidates[candidates < duration_s]
        accepted = np.flatnonzero(rng.uniform(size=len(candidates)) * upper <= intensity(candidates, events)) if len(candidates) else []
        if len(accepted):
            t = float(candidates[accepted[0]])
            events.append(t)
            if first_only:
                break
        elif len(candidates) < batch:
            break
        else:
            t = float(candidates[-1])
    return np.array(events)


class PointProcess(InferenceMixin, GOFMixin, DiagnosticsMixin):
    # Headroom on a grid maximum used as the thinning upper bound: the
    # continuous maximum can lie between grid points.
    _THINNING_SAFETY_MARGIN: ClassVar[float] = 1.1

    def __init__(self, integration_dt: float = 0.02) -> None:
        self.name: str = ""
        self.latex_formula = ""
        self.integration_dt = integration_dt
        self.fit_result = None
        self.params_: Optional[np.ndarray] = None
        self.param_dict_: Dict[str, float] = {}
        self.initial_guesses: List[float] = []
        self.bounds: List[Tuple[Optional[float], Optional[float]]] = []
        self.param_names: List[str] = []

    def _require_fitted(self) -> None:
        if self.params_ is None:
            raise ValueError(f"{self.name}: fit the model or set its parameters first.")

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        raise NotImplementedError

    def supports_jax(self) -> bool:
        """Whether jax_arrays and jax_nll (a JAX likelihood, so exact gradients) are defined."""
        return False

    def jax_arrays(self, dataset: PointProcessDataset) -> Arrays:
        raise NotImplementedError

    def jax_nll(self, params: jnp.ndarray, arrays: Arrays) -> jnp.ndarray:
        raise NotImplementedError

    def _jax_key(self) -> Tuple:
        """What jax_nll depends on besides its arguments: one compiled likelihood per key."""
        return type(self).__name__, self.name, self.integration_dt

    def _arrays(self, dataset: PointProcessDataset) -> Arrays:
        return dataset.cached(("jax",) + self._jax_key(), lambda: self.jax_arrays(dataset))

    def nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        """The NLL: by the JAX likelihood where there is one."""
        return self.jax_value(params, dataset) if self.supports_jax() else self._nll(list(params), dataset)

    def jax_value(self, params: List[float], dataset: PointProcessDataset) -> float:
        return float(compiled(self._jax_key(), self.jax_nll)[0](jnp.asarray(params, dtype=float), self._arrays(dataset)))

    def nll_and_grad(self, params: List[float], dataset: PointProcessDataset) -> Tuple[float, np.ndarray]:
        value, grad = compiled(self._jax_key(), self.jax_nll)[1](jnp.asarray(params, dtype=float), self._arrays(dataset))
        return float(value), np.asarray(grad)

    def jax_hessian(self, params: List[float], dataset: PointProcessDataset) -> np.ndarray:
        return np.array(compiled(self._jax_key(), self.jax_nll)[2](jnp.asarray(params, dtype=float), self._arrays(dataset)))

    def set_params(self, params: np.ndarray) -> None:
        self.params_ = np.asarray(params, dtype=float)
        self.param_dict_ = dict(zip(self.param_names, self.params_))

    def _split_params(self, params: List[float]) -> Tuple[List[float], List[float]]:
        """Rate-kernel parameters and the rest (history, renewal, ...)."""
        n = len(self.kernel.param_names)
        return list(params[:n]), list(params[n:])

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        raise NotImplementedError

    def conditional_rate(self, dataset: PointProcessDataset, side: Optional[int] = None) -> Optional[np.ndarray]:
        """Rate given each fish's observed bouts and its own gains; None where not defined (Frailty)."""
        return None

    def simulated_rate(self, dataset: PointProcessDataset, side: Optional[int] = None, n_fish: int = 3000, seed: int = 0) -> np.ndarray:
        """
        Event rate (Hz, trial x time bin) the model produces on dataset's design, the
        mean of enough simulations of it for about n_fish simulated fish (at least 3),
        of all events or of one side's (laterality mark).
        The rate of models whose intensity depends on each fish's own bouts: rates
        computed from the observed bouts at the mean gain overshoot, a quiet fish being
        rested most of the time. Cached for the last dataset and parameters.
        """
        key = (id(dataset), dataset.num_fish, len(dataset.event_times), tuple(self.params_), n_fish, seed)
        if getattr(self, "_simulated_rates", (None,))[0] != key:
            rng = np.random.default_rng(seed)
            sims = [self.simulate_dataset(dataset, rng) for _ in range(max(3, -(-n_fish // dataset.num_fish)))]
            sides = np.unique(sims[0].marks["laterality"]) if "laterality" in sims[0].marks else []
            rates = {s: np.mean([(x if s is None else x.select(laterality=int(s))).time_trial_histogram_hz for x in sims], axis=0)
                     for s in [None, *sides]}
            self._simulated_rates = (key, rates)
        rates = self._simulated_rates[1]
        return rates[side] if side in rates else np.zeros_like(rates[None])

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        raise NotImplementedError

    def predict(self, t: np.ndarray, trial: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        raise NotImplementedError

    def population_survival_curve(
        self, dataset: PointProcessDataset, trial: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Model first-event survival S(t); only processes without self-history define it."""
        raise NotImplementedError

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        (sum of log base intensities, events per fish, exposure per fish):
        the sufficient statistics of a fish gain multiplying the whole intensity.
        """
        raise NotImplementedError

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator
    ) -> np.ndarray:
        raise NotImplementedError

    def _intensity_upper_bound(self, dataset: PointProcessDataset, t_idx: int) -> float:
        """Thinning bound of the rate kernel over [0, duration_s]."""
        grid = np.arange(0.0, dataset.duration_s + self.integration_dt, self.integration_dt)
        kernel_params, _ = self._split_params(self.params_)
        values = self.kernel.evaluate(grid, np.full_like(grid, t_idx), kernel_params)
        return float(np.max(values)) * self._THINNING_SAFETY_MARGIN

    def _draw_fish_gains(self, num_fish: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        """No frailty: every gain is 1."""
        return np.ones((num_fish, n_sims))

    def _fish_draws(self, num_fish: int, rng: np.random.Generator) -> list:
        """What simulate_stream gets per fish: its gain (pairs of gain and side shift for SideFrailty)."""
        return [float(g) for g in self._draw_fish_gains(num_fish, 1, rng)[:, 0]]

    def simulate_dataset(
        self, template: PointProcessDataset, rng: Optional[np.random.Generator] = None
    ) -> PointProcessDataset:
        """
        One dataset from the current parameters with the design of `template`
        (fish, trials, mask, duration): one gain per fish shared by its
        trials, history reset in every trial. A marked process's streams come
        as (times, marks) and keep their marks. The template's other bouts (not
        events) are kept as they are, around the simulated events, as history.
        """
        self._require_fitted()
        rng = rng or np.random.default_rng()
        draws = self._fish_draws(template.num_fish, rng)
        others = lambda f_idx, t_idx: {"others": template.other_times(f_idx, t_idx)} if self.uses_history else {}

        streams = [
            (f_idx, t_idx, self.simulate_stream(template, t_idx, draws[f_idx], rng, **others(f_idx, t_idx)))
            for f_idx, t_idx, _ in template.iter_streams()
        ]
        marks = [ev[1] for _, _, ev in streams if isinstance(ev, tuple)]
        streams = [(f, t, ev[0] if isinstance(ev, tuple) else ev) for f, t, ev in streams]
        return PointProcessDataset(
            event_times=np.concatenate([[]] + [ev for _, _, ev in streams]).astype(float),
            event_trials_idx=np.concatenate([[]] + [np.full(len(ev), t) for _, t, ev in streams]).astype(int),
            event_fish_idx=np.concatenate([[]] + [np.full(len(ev), f) for f, _, ev in streams]).astype(int),
            fish_trial_mask=template.fish_trial_mask.copy(),
            fish_ids=template.fish_ids.copy(),
            category_label=template.category_label,
            laterality=template.laterality,
            duration_s=template.duration_s,
            binning_dt=template.binning_dt,
            marks={key: np.concatenate([m[key] for m in marks]) for key in (marks[0] if marks else {})},
        ).with_history_from(template)

    def stream_compensator_profile(
        self, t_ev: np.ndarray, trial: float, duration_s: float, history: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool, float]:
        """
        (probe times, compensator at the probes, last probe censored, total
        exposure) for one recurrent stream: each event ends an exact interval
        and trial end one censored interval. history: every bout of the stream,
        for processes that use it (uses_history).
        """
        self._require_fitted()
        probes = np.append(np.sort(np.asarray(t_ev, dtype=float)), float(duration_s))
        cumulative = self.cumulative_integrated_intensity(probes, trial)
        return probes, cumulative, True, float(cumulative[-1])

    def _stream_tau_values(self, dataset: PointProcessDataset) -> StreamTaus:
        """
        Time-rescaled intervals (residual, censored) per stream. Valid when
        cumulative_integrated_intensity is a predictable compensator; frailty
        models override it.
        """
        result: StreamTaus = {}
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            _, cum, last_censored, _ = self.stream_compensator_profile(
                t_ev, t_idx, dataset.duration_s, history=dataset.history_times(f_idx, t_idx)
            )
            pairs = [(float(d), False) for d in np.diff(cum, prepend=0.0)]
            if last_censored:
                pairs[-1] = (pairs[-1][0], True)
            result[(f_idx, t_idx)] = pairs
        return result

    @property
    def dispersion_r(self) -> float:
        return np.inf

    @property
    def stream_gain_variance(self) -> float:
        """Variance of a stream's (fish x trial) gain on the rate; 0 without gains."""
        return 0.0

    @property
    def overdispersion_index(self) -> float:
        r = self.dispersion_r
        return 0.0 if np.isinf(r) else 1.0 / r

    @property
    def is_survival(self) -> bool:
        return False

    @property
    def uses_history(self) -> bool:
        """Whether the intensity depends on the stream's bouts outside the events (dataset.history)."""
        return False

    @property
    def gain_scales_drive_only(self) -> bool:
        """Whether a fish gain scales only part of the intensity (drive_likelihood_terms) instead of all of it."""
        return False
