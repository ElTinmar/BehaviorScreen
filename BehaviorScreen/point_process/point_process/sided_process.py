"""
Both sides of a directional stimulus in one model.

SidedPoissonProcess, recurrent events: each side keeps its own rate kernel;
the contra amplitude is the ipsi one over the directionality D:

    lambda_ipsi(t, m)   = ipsi kernel(t, m)
    lambda_contra(t, m) = contra kernel(t, m) with B_contra = B_ipsi / D

D > 1: more events on the stimulus side; D = 1: no side preference. Wrapped
in a Frailty, one fish gain scales both sides. Events are told apart by their
'laterality' mark; the ground process (both sides) gives the compensator,
the expected rate surface and the time-rescaling diagnostics. Equivalently:
events at the summed rate, each ipsi with probability lambda_ipsi / (sum).

SidedSurvivalProcess, first events (escapes): one hazard h for an escape to
either side, split between them as competing risks with odds D:

    h_preferred = h D / (1 + D),   h_other = h / (1 + D)

so h (its H) is the strength of escaping at all and D the odds that the
escape goes the preferred way (away from a looming stimulus: contra).

SidedRenewalProcess, recurrent events with refractoriness: the sided Poisson
rates times a renewal kernel of the time since the last bout (any category or
side), i.e. a renewal process of both sides together with the same side split.

side_terms gives, for every event that has a side, its log odds of the
preferred side; SideFrailty shifts them per fish.
"""
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
from scipy.special import expit, logit

from BehaviorScreen.core import Laterality

from .dataset import PointProcessDataset
from .point_process import PointProcess, thin
from .poisson_process import RateKernel
from .renewal_process import RenewalKernel, RenewalProcess
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
            rf"\lambda_{{contra}}: {contra.latex_formula.strip('$')},\ B_{{contra}} = B / D"
            + (r"\text{, its other parameters fitted separately}$" if rest else "$")
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

    def stream_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Sum of log rates at the events, and events and expected events per stream (fish x trial)."""
        is_ipsi = self._is_ipsi(dataset)
        t, m = dataset.event_times, dataset.event_trials_idx
        trials = np.arange(dataset.num_trials)
        log_rates, per_trial = 0.0, np.zeros(dataset.num_trials)
        for (kernel, p), side in zip(self._sides(params), (is_ipsi, ~is_ipsi)):
            log_rates += float(np.sum(np.log(kernel.evaluate(t[side], m[side], p))))
            per_trial += kernel.integrate(dataset.duration_s, trials, p, self.integration_dt)
        return log_rates, dataset.event_counts_per_stream(), dataset.fish_trial_mask * per_trial[None, :]

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        log_rates, n, S = self.stream_likelihood_terms(dataset, params)
        return log_rates, n.sum(axis=1), S.sum(axis=1)

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

    def side_terms(self, dataset: PointProcessDataset, params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every event: (fish, log odds of ipsi, is ipsi)."""
        (k_i, p_i), (k_c, p_c) = self._sides(params)
        t, m = dataset.event_times, dataset.event_trials_idx
        log_odds = np.log(k_i.evaluate(t, m, p_i)) - np.log(k_c.evaluate(t, m, p_c))
        return dataset.event_fish_idx, log_odds, self._is_ipsi(dataset)

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator, side_shift: float = 0.0
    ) -> Tuple[np.ndarray, dict]:
        """One stream of a fish with gain `gain`: event times and their 'laterality' marks; side_shift is added to the log odds of ipsi."""
        self._require_fitted()
        grid = np.arange(0.0, dataset.duration_s + self.integration_dt, self.integration_dt)
        rate = lambda t, events: gain * float(self.predict(np.array([t]), np.array([t_idx]))[0])
        upper = gain * float(np.max(self.predict(grid, np.full_like(grid, t_idx)))) * self._THINNING_SAFETY_MARGIN
        times = thin(rate, lambda t, events: upper, dataset.duration_s, rng)
        p_ipsi = expit(logit(self.ipsi_probability(times, np.full_like(times, t_idx))) + side_shift)
        return times, {"laterality": np.where(rng.random(len(times)) < p_ipsi, IPSI, CONTRA)}


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

    def side_terms(self, dataset: PointProcessDataset, params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every first event: (fish, log odds of the preferred side, is on the preferred side)."""
        first = dataset.first_events()
        preferred = np.asarray(first.marks["laterality"]) == self.preferred
        return first.event_fish_idx, np.full(len(preferred), np.log(params[-1])), preferred

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
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator, side_shift: float = 0.0
    ) -> Tuple[np.ndarray, dict]:
        """First event (or none) and its side; side_shift is added to log D."""
        events = super().simulate_stream(dataset, t_idx, gain, rng)
        d = self.params_[-1] * np.exp(side_shift)
        other = IPSI if self.preferred == CONTRA else CONTRA
        sides = np.where(rng.random(len(events)) < d / (1.0 + d), self.preferred, other)
        return events, {"laterality": sides}


class SidedRenewalProcess(PointProcess):
    """
    lambda(t, m) = rho(t - last bout) * (kernel_ipsi + kernel_contra + excitation(t - last event)),
    each event ipsi with probability kernel_ipsi / (kernel_ipsi + kernel_contra):
    kernels and D as in SidedPoissonProcess, rho a renewal kernel of the time since
    the stream's last bout of any category or side (dataset.history), excitation
    (optional, an absolute burst rate) one of the time since the last event of
    either side, whose bursts share the stimulus-driven side split. Fitted as a
    RenewalProcess of all events with kernel ipsi + contra and the side split.
    """

    def __init__(
        self,
        ipsi: RateKernel,
        contra: RateKernel,
        renewal_kernel: RenewalKernel,
        excitation: Optional[RenewalKernel] = None,
        d_init: float = 2.0,
        d_bounds: Tuple[float, float] = (0.02, 50.0),
        integration_dt: float = 0.02,
    ) -> None:
        super().__init__(integration_dt)
        self.sides = SidedPoissonProcess(ipsi, contra, d_init, d_bounds, integration_dt)
        self._n_sides = len(self.sides.param_names)
        both = RateKernel(
            name=f"{ipsi.name} + {contra.name}",
            func=lambda t, trial, p: sum(k.evaluate(t, trial, q) for k, q in self.sides._sides(p)),
            param_names=self.sides.param_names,
            initial_guesses=self.sides.initial_guesses,
            bounds=self.sides.bounds,
        )
        self.ground = RenewalProcess(both, renewal_kernel, excitation, integration_dt=integration_dt)
        self.param_names, self.initial_guesses, self.bounds = self.ground.param_names, self.ground.initial_guesses, self.ground.bounds
        self.name = f"SidedRenewal[{ipsi.name} | {contra.name}, history: {renewal_kernel.name}" + (
            f", events: {excitation.name}]" if excitation is not None else "]"
        )
        self.latex_formula = self.latex_with_gain("")

    def latex_with_gain(self, gain: str) -> str:
        """The formula with a gain (LaTeX) on the stimulus-driven rate."""
        drive = rf"{gain}(\lambda_{{ipsi}} + \lambda_{{contra}})" if gain else r"\lambda_{ipsi} + \lambda_{contra}"
        excitation = self.ground.excitation
        if self.ground._relative:
            total = rf"\lambda = \rho(\Delta t_b)\,(\lambda_{{ipsi}} + \lambda_{{contra}})\,({gain or 1} + E(\Delta t_e))"
        else:
            total = rf"\lambda = \rho(\Delta t_b)\,({drive}" + (r" + E(\Delta t_e))" if excitation is not None else ")")
        return "$" + r";\ ".join([self.sides.latex_formula.strip("$"), total] + self.ground.history_latex()) + "$"

    def set_params(self, params: np.ndarray) -> None:
        super().set_params(params)
        self.sides.set_params(self.params_[: self._n_sides])
        self.ground.set_params(self.params_)

    def side_terms(self, dataset: PointProcessDataset, params: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.sides.side_terms(dataset, list(params[: self._n_sides]))

    def _side_log_likelihood(self, dataset: PointProcessDataset, params: List[float]) -> float:
        _, log_odds, is_ipsi = self.side_terms(dataset, params)
        return -float(np.sum(np.logaddexp(0, np.where(is_ipsi, -log_odds, log_odds))))

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        return self.ground._nll(params, dataset) - self._side_log_likelihood(dataset, params)

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        log_rates, N, S = self.ground.mixed_effects_likelihood_terms(dataset, params)
        return log_rates + self._side_log_likelihood(dataset, params), N, S

    def stream_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        log_rates, n, S = self.ground.stream_likelihood_terms(dataset, params)
        return log_rates + self._side_log_likelihood(dataset, params), n, S

    @property
    def gain_scales_drive_only(self) -> bool:
        return self.ground.gain_scales_drive_only

    def drive_likelihood_terms(self, dataset: PointProcessDataset, params: List[float]):
        constant, *terms = self.ground.drive_likelihood_terms(dataset, params)
        return (constant + self._side_log_likelihood(dataset, params), *terms)

    def drive_stream_profile(self, t_ev: np.ndarray, trial: float, duration_s: float, history: Optional[np.ndarray] = None):
        return self.ground.drive_stream_profile(t_ev, trial, duration_s, history=history)

    def ipsi_probability(self, t: np.ndarray, trial: np.ndarray) -> np.ndarray:
        return self.sides.ipsi_probability(t, trial)

    def compute_expected_rate(self, dataset: PointProcessDataset, side: Optional[int] = None) -> np.ndarray:
        """Rate to either side (None) or to one, given each stream's observed bouts."""
        total = self.ground.compute_expected_rate(dataset)
        if side is None:
            return total
        p_ipsi = self.ipsi_probability(dataset.t_centers[None, :], np.arange(dataset.num_trials)[:, None])
        return total * (p_ipsi if side == IPSI else 1 - p_ipsi)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float, history: Optional[np.ndarray] = None) -> np.ndarray:
        return self.ground.cumulative_integrated_intensity(t_events, trial, history=history)

    def stream_compensator_profile(self, t_ev: np.ndarray, trial: float, duration_s: float, history: Optional[np.ndarray] = None):
        return self.ground.stream_compensator_profile(t_ev, trial, duration_s, history=history)

    @property
    def uses_history(self) -> bool:
        return True

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator,
        others: Optional[np.ndarray] = None, side_shift: float = 0.0,
    ) -> Tuple[np.ndarray, dict]:
        """Bouts of either side given the stream's other bouts, then each one's side; side_shift is added to the log odds of ipsi."""
        times = self.ground.simulate_stream(dataset, t_idx, gain, rng, others=others)
        p_ipsi = expit(logit(self.ipsi_probability(times, np.full_like(times, t_idx))) + side_shift)
        return times, {"laterality": np.where(rng.random(len(times)) < p_ipsi, IPSI, CONTRA)}


def sided_base(model: PointProcess) -> Optional[PointProcess]:
    """
    What gives the sides of a model: its SideFrailty if it has one (sides averaged
    over fish side preferences), else the sided process under any wrappers; None if neither.
    """
    while not hasattr(model, "ipsi_probability"):
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
