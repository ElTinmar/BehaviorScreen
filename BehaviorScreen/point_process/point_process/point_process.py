"""PointProcess: parameters, likelihood interface, compensator and simulation."""
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset import PointProcessDataset
from .diagnostics import DiagnosticsMixin
from .gof import GOFMixin
from .inference import InferenceMixin

StreamTaus = Dict[Tuple[int, int], List[Tuple[float, bool]]]


def thin(
    intensity: Callable[[float, List[float]], float],
    upper_bound: Callable[[float, List[float]], float],
    duration_s: float,
    rng: np.random.Generator,
    first_only: bool = False,
) -> np.ndarray:
    """
    Ogata thinning: event times on [0, duration_s) of a process whose
    conditional intensity(t, events so far) stays below upper_bound(t, events)
    until the next event.
    """
    events: List[float] = []
    t = 0.0
    while t < duration_s:
        upper = upper_bound(t, events)
        t_candidate = t + rng.exponential(1.0 / max(upper, 1e-12))
        if t_candidate >= duration_s:
            break
        if rng.uniform() <= intensity(t_candidate, events) / upper:
            events.append(t_candidate)
            if first_only:
                break
        t = t_candidate
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

    def set_params(self, params: np.ndarray) -> None:
        self.params_ = np.asarray(params, dtype=float)
        self.param_dict_ = dict(zip(self.param_names, self.params_))

    def _split_params(self, params: List[float]) -> Tuple[List[float], List[float]]:
        """Rate-kernel parameters and the rest (history, renewal, ...)."""
        n = len(self.kernel.param_names)
        return list(params[:n]), list(params[n:])

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        raise NotImplementedError

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

    def simulate_dataset(
        self, template: PointProcessDataset, rng: Optional[np.random.Generator] = None
    ) -> PointProcessDataset:
        """
        One dataset from the current parameters with the design of `template`
        (fish, trials, mask, duration): one gain per fish shared by its
        trials, history reset in every trial. A marked process's streams come
        as (times, marks) and keep their marks.
        """
        self._require_fitted()
        rng = rng or np.random.default_rng()
        fish_gains = self._draw_fish_gains(template.num_fish, 1, rng)[:, 0]

        streams = [
            (f_idx, t_idx, self.simulate_stream(template, t_idx, float(fish_gains[f_idx]), rng))
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
        )

    def stream_compensator_profile(
        self, t_ev: np.ndarray, trial: float, duration_s: float
    ) -> Tuple[np.ndarray, np.ndarray, bool, float]:
        """
        (probe times, compensator at the probes, last probe censored, total
        exposure) for one recurrent stream: each event ends an exact interval
        and trial end one censored interval.
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
            _, cum, last_censored, _ = self.stream_compensator_profile(t_ev, t_idx, dataset.duration_s)
            pairs = [(float(d), False) for d in np.diff(cum, prepend=0.0)]
            if last_censored:
                pairs[-1] = (pairs[-1][0], True)
            result[(f_idx, t_idx)] = pairs
        return result

    @property
    def dispersion_r(self) -> float:
        return np.inf

    @property
    def overdispersion_index(self) -> float:
        r = self.dispersion_r
        return 0.0 if np.isinf(r) else 1.0 / r

    @property
    def is_survival(self) -> bool:
        return False
