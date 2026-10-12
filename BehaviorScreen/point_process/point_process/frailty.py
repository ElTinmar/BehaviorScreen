"""
Per-fish gains g_f with E[g_f] = 1, integrated out of the likelihood.

A gain distribution (GammaGain, ZeroInflatedGain) holds the math; a family
(GammaFamily, ZeroInflatedFamily) maps the trailing fit parameters to one.
Frailty multiplies the whole intensity (closed form); BaselineFrailty
multiplies only the baseline of a Hawkes process (quadrature), so the
branching ratio stays the population one.
"""
from dataclasses import dataclass
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammaln

from .dataset import PointProcessDataset
from .kernel_shapes import logit_bounded, sigmoid_bounded
from .point_process import PointProcess, StreamTaus, thin
from .wrappers import Wrapper

ArrayLike = Union[float, np.ndarray]
_EVENT_CHUNK = 512  # events per chunk of the quadrature: 512 x n_nodes values stay in cache


def gauss_laguerre(n: int, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nodes and log-weights, one row per alpha, for the weight x^alpha e^-x
    (Golub-Welsch, finite for large alpha).
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))[:, None]
    k = np.arange(n)
    if len(alpha) == 1:  # one fish: the tridiagonal solver is faster than a batch of one
        nodes, vectors = eigh_tridiagonal(2 * k + alpha[0] + 1, np.sqrt(k[1:] * (k[1:] + alpha[0])))
        return nodes[None, :], 2 * np.log(np.abs(vectors[:1])) + gammaln(alpha + 1)
    off_diagonal = np.sqrt(k[1:] * (k[1:] + alpha))
    jacobi = np.zeros((len(alpha), n, n))
    jacobi[:, k, k] = 2 * k + alpha + 1
    jacobi[:, k[1:], k[:-1]] = jacobi[:, k[:-1], k[1:]] = off_diagonal
    nodes, vectors = np.linalg.eigh(jacobi)
    return nodes, 2 * np.log(np.abs(vectors[:, 0, :])) + gammaln(alpha + 1)


def gamma_gain_evidence_by_fish(
    base_rates: np.ndarray,
    history_rates: np.ndarray,
    fish: np.ndarray,
    exposure: np.ndarray,
    shape: float,
    rate: float,
    n_nodes: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per fish: log Z and E[g | events] for g ~ Gamma(shape, rate), with
    Z = integral of p(g) exp(-g exposure) prod over its events (g b + h) dg.
    fish: the fish of every event, non-decreasing.

    Generalized Gauss-Laguerre with exponent shape - 1 + k, where k is the
    effective number of baseline-driven events: the remaining factor
    prod(g b + h) / g^k is flat near the posterior mode, so the rule stays
    accurate for fish with hundreds of events (exact when h = 0).
    """
    n_fish = len(exposure)
    d = exposure + rate
    driven = base_rates * (shape + np.bincount(fish, minlength=n_fish)[fish]) / d[fish]
    total = driven + history_rates
    k = np.bincount(fish, weights=np.divide(driven, total, out=np.zeros_like(total), where=total > 0), minlength=n_fish)
    x, log_w = gauss_laguerre(n_nodes, shape - 1 + k)
    g = x / d[:, None]
    log_terms = log_w - k[:, None] * np.log(g)
    for a in range(0, len(fish), _EVENT_CHUNK):
        chunk_fish = fish[a : a + _EVENT_CHUNK]
        logs = np.log(np.maximum(g[chunk_fish] * base_rates[a : a + _EVENT_CHUNK, None] + history_rates[a : a + _EVENT_CHUNK, None], 1e-300))
        starts = np.flatnonzero(np.concatenate([[True], chunk_fish[1:] != chunk_fish[:-1]]))
        log_terms[chunk_fish[starts]] += np.add.reduceat(logs, starts, axis=0)
    peak = log_terms.max(axis=1, keepdims=True)
    terms = np.exp(log_terms - peak)
    total_terms = terms.sum(axis=1)
    log_z = shape * np.log(rate) - gammaln(shape) - (shape + k) * np.log(d) + peak[:, 0] + np.log(total_terms)
    return log_z, np.sum(terms * g, axis=1) / total_terms


def gamma_gain_evidence(
    base_rates: np.ndarray, history_rates: np.ndarray, exposure: float, shape: float, rate: float, n_nodes: int = 30
) -> Tuple[float, float]:
    """gamma_gain_evidence_by_fish for a single fish."""
    log_z, mean = gamma_gain_evidence_by_fish(
        base_rates, history_rates, np.zeros(len(base_rates), dtype=int), np.array([exposure]), shape, rate, n_nodes
    )
    return float(log_z[0]), float(mean[0])


@dataclass(frozen=True)
class GammaGain:
    """g ~ Gamma(shape, rate). N events and exposure S are under g = 1."""

    shape: float
    rate: float

    def log_marginal(self, N: ArrayLike, S: ArrayLike) -> np.ndarray:
        """log of integral p(g) g^N exp(-g S) dg."""
        r, b = self.shape, self.rate
        return r * np.log(b) - gammaln(r) + gammaln(N + r) - (N + r) * np.log(S + b)

    def tau(self, N: float, S_prev: float, S_abs: float) -> float:
        """-log P(no event on exposure (S_prev, S_abs] | N events by S_prev)."""
        return float((self.shape + N) * np.log((self.rate + S_abs) / (self.rate + S_prev)))

    def posterior(self, N: np.ndarray, S: np.ndarray) -> Dict[str, np.ndarray]:
        return {"estimated_gain": (self.shape + N) / (self.rate + S)}

    def evidence_by_fish(
        self, base_rates: np.ndarray, history_rates: np.ndarray, fish: np.ndarray, S: np.ndarray, n_nodes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return gamma_gain_evidence_by_fish(base_rates, history_rates, fish, S, self.shape, self.rate, n_nodes)

    def evidence(self, base_rates: np.ndarray, history_rates: np.ndarray, S: float, n_nodes: int) -> Tuple[float, float]:
        return gamma_gain_evidence(base_rates, history_rates, S, self.shape, self.rate, n_nodes)

    def survival(self, Lambda: np.ndarray) -> np.ndarray:
        return np.power(self.rate / (self.rate + Lambda), self.shape)

    def draw(self, rng: np.random.Generator, size: Tuple[int, int]) -> np.ndarray:
        return rng.gamma(shape=self.shape, scale=1.0 / self.rate, size=size)

    @property
    def dispersion(self) -> float:
        return self.shape


@dataclass(frozen=True)
class ZeroInflatedGain:
    """g = c with probability pi, else g ~ responder."""

    pi: float
    c: float
    responder: GammaGain

    def _log_nonresponder(self, N: ArrayLike, S: ArrayLike) -> np.ndarray:
        N, S = np.asarray(N, dtype=float), np.asarray(S, dtype=float)
        if self.c <= 0.0:
            return np.where(N == 0, 0.0, -np.inf)
        return N * np.log(self.c) - self.c * S

    def _class_log_posteriors(self, N: ArrayLike, S: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(log P(non-responder), log P(responder), log marginal)."""
        nonresponder = np.log(max(self.pi, 1e-300)) + self._log_nonresponder(N, S)
        responder = np.log(max(1.0 - self.pi, 1e-300)) + self.responder.log_marginal(N, S)
        norm = np.logaddexp(nonresponder, responder)
        return nonresponder - norm, responder - norm, norm

    def log_marginal(self, N: ArrayLike, S: ArrayLike) -> np.ndarray:
        return self._class_log_posteriors(N, S)[2]

    def tau(self, N: float, S_prev: float, S_abs: float) -> float:
        log_p_nr, log_p_r, _ = self._class_log_posteriors(N, S_prev)
        return float(
            -np.logaddexp(
                log_p_nr - self.c * (S_abs - S_prev),
                log_p_r - self.responder.tau(N, S_prev, S_abs),
            )
        )

    def posterior(self, N: np.ndarray, S: np.ndarray) -> Dict[str, np.ndarray]:
        p_responder = np.exp(self._class_log_posteriors(N, S)[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            responder_gain = self.responder.posterior(N, S)["estimated_gain"]
        return {
            "p_responder": p_responder,
            "estimated_gain": (1.0 - p_responder) * self.c + p_responder * responder_gain,
        }

    def evidence_by_fish(
        self, base_rates: np.ndarray, history_rates: np.ndarray, fish: np.ndarray, S: np.ndarray, n_nodes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        log_nr = -self.c * S + np.bincount(
            fish, weights=np.log(np.maximum(self.c * base_rates + history_rates, 1e-300)), minlength=len(S)
        )
        log_r, mean_r = self.responder.evidence_by_fish(base_rates, history_rates, fish, S, n_nodes)
        weighted_nr = np.log(max(self.pi, 1e-300)) + log_nr
        weighted_r = np.log(max(1.0 - self.pi, 1e-300)) + log_r
        log_z = np.logaddexp(weighted_nr, weighted_r)
        p_responder = np.exp(weighted_r - log_z)
        return log_z, (1.0 - p_responder) * self.c + p_responder * mean_r

    def evidence(self, base_rates: np.ndarray, history_rates: np.ndarray, S: float, n_nodes: int) -> Tuple[float, float]:
        log_z, mean = self.evidence_by_fish(base_rates, history_rates, np.zeros(len(base_rates), dtype=int), np.array([S]), n_nodes)
        return float(log_z[0]), float(mean[0])

    def survival(self, Lambda: np.ndarray) -> np.ndarray:
        return self.pi * np.exp(-self.c * Lambda) + (1.0 - self.pi) * self.responder.survival(Lambda)

    def draw(self, rng: np.random.Generator, size: Tuple[int, int]) -> np.ndarray:
        nonresponder = rng.random(size) < self.pi
        gains = self.responder.draw(rng, size)
        gains[nonresponder] = self.c
        return gains

    @property
    def dispersion(self) -> float:
        """1 / Var(g): the Gamma(r, r) with the same variance (shape is not matched)."""
        r, mu = self.responder.shape, self.responder.shape / self.responder.rate
        second_moment = self.pi * self.c**2 + (1.0 - self.pi) * mu**2 * (1.0 + 1.0 / r)
        return float(1.0 / max(second_moment - 1.0, 1e-12))


class GammaFamily:
    prefix = ""
    latex = r"g_f\sim\Gamma(r,r)"

    def __init__(self, r_init: float = 5.0, r_floor: float = 1e-3) -> None:
        self.suffix = ""
        self.param_names = ["r_dispersion"]
        self.initial_guesses = [r_init]
        self.bounds: List[Tuple[Optional[float], Optional[float]]] = [(r_floor, None)]

    def __call__(self, tail: List[float]) -> GammaGain:
        r = max(float(tail[0]), 1e-8)
        return GammaGain(r, r)


class ZeroInflatedFamily:
    """
    pi and c through bounded logits; the responder mean mu_R is set by
    E[g] = pi c + (1 - pi) mu_R = 1. pi = 0 is a boundary of the parameter
    space: a chi2 LRT against the Gamma model is conservative.
    """

    prefix = "ZeroInflated"
    latex = r"g_f \sim \pi\,\delta_c + (1-\pi)\,\Gamma(r, r/\mu_R)"

    def __init__(
        self,
        pi_init: float = 0.3,
        r_init: float = 5.0,
        fit_c: bool = False,
        c_init: float = 0.0,
        c_upper: float = 0.3,
        r_floor: float = 1e-3,
    ) -> None:
        self.fit_c, self.c_init, self.c_upper = fit_c, c_init, c_upper
        self.suffix = " (fit c)" if fit_c else ""
        self.param_names = ["z_pi_nonresponder", "r_dispersion"]
        self.initial_guesses = [float(logit_bounded(pi_init, 1.0)), r_init]
        self.bounds: List[Tuple[Optional[float], Optional[float]]] = [(-15.0, 15.0), (r_floor, None)]
        if fit_c:
            self.param_names.append("z_c_nonresponder")
            self.initial_guesses.append(float(logit_bounded(max(c_init, 1e-4), c_upper)))
            self.bounds.append((-15.0, 15.0))

    def __call__(self, tail: List[float]) -> ZeroInflatedGain:
        pi = float(sigmoid_bounded(tail[0], 1.0))
        r = max(float(tail[1]), 1e-8)
        c = float(sigmoid_bounded(tail[2], self.c_upper)) if self.fit_c else self.c_init
        mu = (1.0 - pi * c) / max(1.0 - pi, 1e-12)
        return ZeroInflatedGain(pi, c, GammaGain(r, r / mu))


GainFamily = Union[GammaFamily, ZeroInflatedFamily]


class _GainWrapper(Wrapper):
    """Base parameters followed by the gain family's parameters."""

    def __init__(self, base_process: PointProcess, gains: GainFamily) -> None:
        super().__init__(base_process)
        self.gains = gains
        self._n_base = len(base_process.param_names)
        self.param_names = base_process.param_names + gains.param_names
        self.initial_guesses = base_process.initial_guesses + gains.initial_guesses
        self.bounds = base_process.bounds + gains.bounds

    def _base_params(self, params: List[float]) -> List[float]:
        return list(params[: self._n_base])

    def _gain(self, params: Optional[List[float]] = None) -> Union[GammaGain, ZeroInflatedGain]:
        if params is None:
            self._require_fitted()
            params = self.params_
        return self.gains(params[self._n_base :])

    @property
    def gain_distribution(self) -> Union[GammaGain, ZeroInflatedGain]:
        return self._gain()

    def mixed_effects_likelihood_terms(self, dataset: PointProcessDataset, params: List[float]):
        raise NotImplementedError(f"{self.name} already integrates a fish gain.")

    def _draw_fish_gains(self, num_fish: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        return self._gain().draw(rng, (num_fish, n_sims))

    @property
    def dispersion_r(self) -> float:
        return self._gain().dispersion


class Frailty(_GainWrapper):
    """
    lambda_f = g_f * lambda_base. Needs base_process.mixed_effects_likelihood_terms
    (a gain on the whole intensity). predict and the compensators are the
    population average (E[g] = 1); use estimate_fish_gains per fish.
    """

    def __init__(self, base_process: PointProcess, gains: GainFamily) -> None:
        super().__init__(base_process, gains)
        self.name = f"{gains.prefix}GammaMixedEffects[{base_process.name}]{gains.suffix}"
        self.latex_formula = rf"${base_process.latex_formula.strip('$')} \times g_f,\ {gains.latex}$"

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        base_ll, N, S = self.base_process.mixed_effects_likelihood_terms(dataset, self._base_params(params))
        active = dataset.fish_trial_mask.any(axis=1)
        return -(base_ll + float(np.sum(self._gain(params).log_marginal(N[active], S[active]))))

    def _stream_tau_values(self, dataset: PointProcessDataset) -> StreamTaus:
        """
        Predictable compensator: each fish's trials in order, the gain
        posterior conditioned on events and exposure strictly before each probe.
        """
        gain = self._gain()
        result: StreamTaus = {}
        for f_idx, streams in groupby(dataset.iter_streams(), key=lambda stream: stream[0]):
            S_offset = S_prev = 0.0
            n_events = 0
            for _, t_idx, t_ev in streams:
                _, cum, last_censored, exposure = self.base_process.stream_compensator_profile(
                    t_ev, t_idx, dataset.duration_s
                )
                pairs = []
                for k, value in enumerate(cum):
                    S_abs = S_offset + value
                    censored = bool(last_censored and k == len(cum) - 1)
                    pairs.append((gain.tau(n_events, S_prev, S_abs), censored))
                    n_events += not censored
                    S_prev = S_abs
                if pairs:
                    result[(f_idx, t_idx)] = pairs
                S_offset += exposure
                S_prev = S_offset
        return result

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        """Whole-session posterior per fish (for reporting, not for time rescaling)."""
        gain = self._gain()
        _, N, S = self.base_process.mixed_effects_likelihood_terms(dataset, self._base_params(self.params_))
        active = dataset.fish_trial_mask.any(axis=1)
        posterior = gain.posterior(N, S)
        return pd.DataFrame(
            {
                "fish_idx": np.arange(dataset.num_fish)[active],
                "n_events": N[active],
                "expected_events_base": S[active],
                **{column: values[active] for column, values in posterior.items()},
            }
        )

    def population_survival_curve(
        self, dataset: PointProcessDataset, trial: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Marginal first-event survival E_g[exp(-g Lambda(t))]. Only meaningful
        for bases without self-history (Poisson, survival).
        """
        gain = self._gain()
        t_grid = dataset.t_centers

        def survival(tr: int) -> np.ndarray:
            return gain.survival(self.base_process.cumulative_integrated_intensity(t_grid, tr))

        if trial is not None:
            return t_grid, survival(int(trial))
        curves = np.array([survival(tr) for tr in range(dataset.num_trials)])
        weights = dataset.n_fish_per_trial
        if weights.sum() == 0:
            return t_grid, np.mean(curves, axis=0)
        return t_grid, np.average(curves, axis=0, weights=weights)


class BaselineFrailty(_GainWrapper):
    """
    lambda_f(t) = g_f * kernel(t) + history(t) for a Hawkes base: the gain
    never scales self-excitation, so no fish can become super-critical.
    No conjugacy: g_f is integrated per fish by gamma_gain_evidence.
    """

    def __init__(self, base_process: PointProcess, gains: GainFamily, n_quad_nodes: int = 30) -> None:
        if not hasattr(base_process, "history_kernel"):
            raise TypeError(f"{type(self).__name__} needs a HawkesProcess base process.")
        super().__init__(base_process, gains)
        self.n_quad_nodes = n_quad_nodes
        self.name = f"{gains.prefix}BaselineOnlyFrailty[{base_process.name}]{gains.suffix}"
        self.latex_formula = rf"${base_process.latex_formula.strip('$')}$ (baseline $\times\, g_f$), ${gains.latex}$"

    def _per_fish_terms(
        self, dataset: PointProcessDataset, base_params: List[float]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        """
        (baseline rate, history rate, fish) at every event, baseline exposure
        per fish (scaled by g_f) and history exposure per fish (not scaled).
        """
        base = self.base_process
        kernel_params, _ = base._split_params(base_params)
        events, base_rates, history_rates, history_integrals = base._event_terms(dataset, base_params)
        S_base = dataset.fish_trial_mask.astype(float) @ base._trial_integrals(dataset, kernel_params)
        S_hist = np.bincount(events["fish"], weights=history_integrals, minlength=dataset.num_fish)
        return (base_rates, history_rates, events["fish"]), S_base, S_hist

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        events, S_base, S_hist = self._per_fish_terms(dataset, self._base_params(params))
        log_z, _ = self._gain(params).evidence_by_fish(*events, S_base, self.n_quad_nodes)
        active = dataset.fish_trial_mask.any(axis=1)
        return -float(np.sum(log_z[active] - S_hist[active]))

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        """Whole-session posterior mean gain per fish (for reporting, not for time rescaling)."""
        events, S_base, _ = self._per_fish_terms(dataset, self._base_params(self.params_))
        _, gains = self._gain().evidence_by_fish(*events, S_base, self.n_quad_nodes)
        active = np.flatnonzero(dataset.fish_trial_mask.any(axis=1))
        return pd.DataFrame(
            {
                "fish_idx": active,
                "n_events": np.bincount(events[2], minlength=dataset.num_fish)[active],
                "estimated_gain": gains[active],
            }
        )

    def _stream_tau_values(self, dataset: PointProcessDataset) -> StreamTaus:
        """
        Marginal no-event probability over each interval:
        tau = dH + log Z(S) - log Z(S + dB). History resets every trial; the
        gain posterior carries over trials.
        """
        gain = self._gain()
        base = self.base_process
        kernel_params, hist_params = base._split_params(self._base_params(self.params_))
        result: StreamTaus = {}
        for f_idx, streams in groupby(dataset.iter_streams(), key=lambda stream: stream[0]):
            base_rates, history_rates = np.empty(0), np.empty(0)
            S_base = 0.0
            for _, t_idx, t_ev in streams:
                base_cum, hist_cum = base._compensator_parts(np.append(t_ev, dataset.duration_s), t_idx)
                d_base, d_hist = np.diff(base_cum, prepend=0.0), np.diff(hist_cum, prepend=0.0)
                event_base = base.kernel.evaluate(t_ev, np.full(t_ev.shape, t_idx, dtype=float), kernel_params)
                event_hist = base.history_kernel.event_history(t_ev, hist_params)
                pairs = []
                for k in range(len(d_base)):
                    log_z = gain.evidence(base_rates, history_rates, S_base, self.n_quad_nodes)[0]
                    log_z_next = gain.evidence(base_rates, history_rates, S_base + d_base[k], self.n_quad_nodes)[0]
                    tau = d_hist[k] + log_z - log_z_next
                    if tau < -1e-9:
                        raise RuntimeError(f"Negative compensator increment: fish {f_idx}, trial {t_idx}, tau {tau}")
                    censored = k == len(t_ev)
                    pairs.append((float(max(tau, 0.0)), censored))
                    S_base += d_base[k]
                    if not censored:
                        base_rates = np.append(base_rates, event_base[k])
                        history_rates = np.append(history_rates, event_hist[k])
                result[(f_idx, t_idx)] = pairs
        return result

    def simulate_stream(self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator) -> np.ndarray:
        return thin(*self.base_process._thinning_functions(dataset, t_idx, gain, 1.0), dataset.duration_s, rng)


class _ZeroInflatedProperties:
    @property
    def pi_nonresponder(self) -> float:
        return self._gain().pi

    @property
    def r_responder(self) -> float:
        return self._gain().responder.shape

    @property
    def c_nonresponder(self) -> float:
        return self._gain().c


class GammaMixedEffectsProcess(Frailty):
    def __init__(self, base_process: PointProcess, r_init: float = 5.0) -> None:
        super().__init__(base_process, GammaFamily(r_init))


class ZeroInflatedGammaMixedEffectsProcess(_ZeroInflatedProperties, Frailty):
    def __init__(
        self,
        base_process: PointProcess,
        pi_init: float = 0.3,
        r_init: float = 5.0,
        fit_c: bool = False,
        c_init: float = 0.0,
        c_upper: float = 0.3,
    ) -> None:
        super().__init__(base_process, ZeroInflatedFamily(pi_init, r_init, fit_c, c_init, c_upper))


class BaselineOnlyFrailtyHawkesProcess(BaselineFrailty):
    # r floored at 1e-2: Gauss-Laguerre exponent r - 1 near -1 is fragile.
    def __init__(self, base_process: PointProcess, r_init: float = 5.0, n_quad_nodes: int = 30) -> None:
        super().__init__(base_process, GammaFamily(r_init, r_floor=1e-2), n_quad_nodes)


class ZeroInflatedBaselineOnlyFrailtyHawkesProcess(_ZeroInflatedProperties, BaselineFrailty):
    def __init__(
        self,
        base_process: PointProcess,
        pi_init: float = 0.3,
        r_init: float = 5.0,
        fit_c: bool = False,
        c_init: float = 0.0,
        c_upper: float = 0.3,
        n_quad_nodes: int = 30,
    ) -> None:
        family = ZeroInflatedFamily(pi_init, r_init, fit_c, c_init, c_upper, r_floor=1e-2)
        super().__init__(base_process, family, n_quad_nodes)
