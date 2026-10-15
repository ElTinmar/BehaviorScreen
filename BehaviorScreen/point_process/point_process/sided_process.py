"""
Both sides of a directional stimulus in one model.

SidedPoissonProcess, recurrent events: each side keeps its own rate kernel;
the contra amplitude is the ipsi one over the directionality D:

    lambda_ipsi(t, m)   = ipsi kernel(t, m)
    lambda_contra(t, m) = contra kernel(t, m) with B_contra = B_ipsi / D

D > 1: more events on the stimulus side; D = 1: no side preference. Wrapped
in a Frailty, one fish gain scales both sides. Events are told apart by their
'laterality' mark; the ground process (both sides) gives the compensator,
the expected rate surface and the time-rescaling diagnostics.

SidedSurvivalProcess, first events (escapes): one hazard h for an escape to
either side, split between them as competing risks with odds D:

    h_preferred = h D / (1 + D),   h_other = h / (1 + D)

so h (its H) is the strength of escaping at all and D the odds that the
escape goes the preferred way (away from a looming stimulus: contra).
"""
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from BehaviorScreen.core import Laterality

from .dataset import PointProcessDataset
from .point_process import PointProcess, thin
from .poisson_process import RateKernel
from .survival_process import SurvivalProcess

IPSI, CONTRA = int(Laterality.IPSILATERAL), int(Laterality.CONTRALATERAL)
SIDE_NAMES = {IPSI: "ipsi", CONTRA: "contra"}


class SidedPoissonProcess(PointProcess):
    def __init__(
        self,
        ipsi: RateKernel,
        contra: RateKernel,
        d_init: float = 2.0,
        d_bounds: Tuple[float, float] = (0.02, 50.0),
        integration_dt: float = 0.02,
    ) -> None:
        super().__init__(integration_dt)
        self.ipsi, self.contra = ipsi, contra
        self._n_ipsi = len(ipsi.param_names)
        self._b_ipsi = ipsi.param_names.index("B")
        self._b_contra = contra.param_names.index("B")
        rest = [i for i in range(len(contra.param_names)) if i != self._b_contra]
        self.name = f"Sided[{ipsi.name} | {contra.name}]"
        self.latex_formula = (
            rf"$\lambda_{{ipsi}}: {ipsi.latex_formula.strip('$')};\ "
            rf"\lambda_{{contra}}: {contra.latex_formula.strip('$')},\ B_{{contra}} = B / D$"
        )
        self.param_names = ipsi.param_names + ["D"] + [f"contra_{contra.param_names[i]}" for i in rest]
        self.initial_guesses = list(ipsi.initial_guesses) + [d_init] + [contra.initial_guesses[i] for i in rest]
        self.bounds = list(ipsi.bounds) + [d_bounds] + [contra.bounds[i] for i in rest]

    def _split(self, params: List[float]) -> Tuple[List[float], List[float]]:
        """(ipsi kernel parameters, contra kernel parameters) with B_contra = B / D."""
        ipsi = list(params[: self._n_ipsi])
        contra = list(params[self._n_ipsi + 1 :])
        contra.insert(self._b_contra, ipsi[self._b_ipsi] / params[self._n_ipsi])
        return ipsi, contra

    def _sides(self, params: List[float]) -> Tuple[Tuple[RateKernel, List[float]], ...]:
        return tuple(zip((self.ipsi, self.contra), self._split(params)))

    @staticmethod
    def _is_ipsi(dataset: PointProcessDataset) -> np.ndarray:
        return dataset.cached("sided_is_ipsi", lambda: np.asarray(dataset.marks["laterality"]) == IPSI)

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        is_ipsi = self._is_ipsi(dataset)
        t, m = dataset.event_times, dataset.event_trials_idx
        trials = np.arange(dataset.num_trials)
        log_rates, per_trial = 0.0, np.zeros(dataset.num_trials)
        for (kernel, p), side in zip(self._sides(params), (is_ipsi, ~is_ipsi)):
            log_rates += float(np.sum(np.log(kernel.evaluate(t[side], m[side], p))))
            per_trial += kernel.integrate(dataset.duration_s, trials, p, self.integration_dt)
        N = np.bincount(dataset.event_fish_idx, minlength=dataset.num_fish).astype(float)
        return log_rates, N, dataset.fish_trial_mask.astype(float) @ per_trial

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        log_rates, _, S = self.mixed_effects_likelihood_terms(dataset, params)
        return -(log_rates - float(np.sum(S)))

    def predict(self, t: np.ndarray, trial: np.ndarray, side: Optional[int] = None) -> np.ndarray:
        """Rate of one side (IPSI or CONTRA), or of both summed (None)."""
        self._require_fitted()
        (k_i, p_i), (k_c, p_c) = self._sides(self.params_)
        rates = {IPSI: lambda: k_i.evaluate(t, trial, p_i), CONTRA: lambda: k_c.evaluate(t, trial, p_c)}
        return rates[side]() if side is not None else rates[IPSI]() + rates[CONTRA]()

    def compute_expected_rate(self, dataset: PointProcessDataset, side: Optional[int] = None) -> np.ndarray:
        return self.predict(dataset.t_centers[None, :], np.arange(dataset.num_trials)[:, None], side)

    def ipsi_probability(self, t: np.ndarray, trial: np.ndarray) -> np.ndarray:
        """Probability that an event at (t, trial) is ipsi; a fish gain scales both sides and cancels."""
        return self.predict(t, trial, IPSI) / self.predict(t, trial)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        self._require_fitted()
        return sum(k.cumulative_integrate(t_events, trial, p, self.integration_dt) for k, p in self._sides(self.params_))

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator
    ) -> Tuple[np.ndarray, dict]:
        """One stream of a fish with gain `gain`: event times and their 'laterality' marks."""
        self._require_fitted()
        grid = np.arange(0.0, dataset.duration_s + self.integration_dt, self.integration_dt)
        times, marks = [], []
        for (kernel, p), mark in zip(self._sides(self.params_), (IPSI, CONTRA)):
            rate = lambda t, events: gain * kernel.evaluate(np.array([t]), np.array([t_idx]), p)[0]
            upper = gain * float(np.max(kernel.evaluate(grid, np.full_like(grid, t_idx), p))) * self._THINNING_SAFETY_MARGIN
            events = thin(rate, lambda t, events: upper, dataset.duration_s, rng)
            times.append(events)
            marks.append(np.full(len(events), mark))
        times, marks = np.concatenate(times), np.concatenate(marks)
        order = np.argsort(times)
        return times[order], {"laterality": marks[order]}


class SidedSurvivalProcess(SurvivalProcess):
    def __init__(
        self,
        kernel: RateKernel,
        preferred: int = CONTRA,
        d_init: float = 2.0,
        d_bounds: Tuple[float, float] = (0.02, 50.0),
        integration_dt: float = 0.02,
    ) -> None:
        # The hazard ignores D (the last parameter): every inherited survival method works unchanged.
        drop_d = lambda f: None if f is None else (lambda *args: f(*args[:-1], args[-1][:-1]))
        super().__init__(
            replace(
                kernel,
                func=drop_d(kernel.func),
                integral_func=drop_d(kernel.integral_func),
                param_names=kernel.param_names + ["D"],
                initial_guesses=list(kernel.initial_guesses) + [d_init],
                bounds=list(kernel.bounds) + [d_bounds],
            ),
            integration_dt,
        )
        self.preferred = preferred
        self.name = f"SidedSurvival {kernel.name}"
        self.latex_formula = rf"${kernel.latex_formula.strip('$')},\ h_{{preferred}} = h\,D/(1+D)$"

    def _first_event_sides(self, dataset: PointProcessDataset) -> Tuple[int, int]:
        """First events of all streams: (number on the preferred side, number on the other)."""
        def count() -> Tuple[int, int]:
            sides = np.asarray(dataset.first_events().marks["laterality"])
            preferred = int(np.sum(sides == self.preferred))
            return preferred, len(sides) - preferred
        return dataset.cached(("sided_first_events", self.preferred), count)

    def _side_log_likelihood(self, dataset: PointProcessDataset, params: List[float]) -> float:
        n_preferred, n_other = self._first_event_sides(dataset)
        d = params[-1]
        return n_preferred * np.log(d / (1.0 + d)) - n_other * np.log1p(d)

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        return super()._nll(params, dataset) - self._side_log_likelihood(dataset, params)

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        log_hazards, N, S = super().mixed_effects_likelihood_terms(dataset, params)
        return log_hazards + self._side_log_likelihood(dataset, params), N, S

    def compute_expected_rate(self, dataset: PointProcessDataset, side: Optional[int] = None) -> np.ndarray:
        """Hazard of an escape to either side (None) or to one side: its share of it."""
        hazard = super().compute_expected_rate(dataset)
        if side is None:
            return hazard
        d = self.params_[-1]
        return hazard * (d if side == self.preferred else 1.0) / (1.0 + d)

    def ipsi_probability(self, t: np.ndarray, trial: np.ndarray) -> np.ndarray:
        """Probability that an escape is ipsi: the same at every time and trial."""
        d = self.params_[-1]
        return np.full(np.shape(t), (d if self.preferred == IPSI else 1.0) / (1.0 + d))

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator
    ) -> Tuple[np.ndarray, dict]:
        """First event (or none) and its side."""
        events = super().simulate_stream(dataset, t_idx, gain, rng)
        d = self.params_[-1]
        other = IPSI if self.preferred == CONTRA else CONTRA
        sides = np.where(rng.random(len(events)) < d / (1.0 + d), self.preferred, other)
        return events, {"laterality": sides}


def sided_base(model: PointProcess) -> Optional[PointProcess]:
    """The sided process under any wrappers (gain, pinned parameters), or None."""
    while not isinstance(model, (SidedPoissonProcess, SidedSurvivalProcess)):
        if not hasattr(model, "base_process"):
            return None
        model = model.base_process
    return model


class SideView:
    """One side of a fitted sided model, with what the rate plots read: name, formula, expected rate."""

    def __init__(self, model: PointProcess, side: int) -> None:
        self.base, self.side = sided_base(model), side
        self.name = f"{model.name} ({SIDE_NAMES[side]})"
        self.latex_formula = model.latex_formula

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        return self.base.compute_expected_rate(dataset, side=self.side)
