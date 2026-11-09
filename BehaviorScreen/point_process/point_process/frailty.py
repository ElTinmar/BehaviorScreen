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
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csr_matrix
from scipy.special import expit, gammaln, logit, logsumexp
from scipy.stats import gamma as gamma_distribution

from . import gains_jax
from .backend import Arrays, jnp, padded
from .dataset import PointProcessDataset
from .kernel_shapes import logit_bounded, sigmoid_bounded
from .point_process import PointProcess, StreamTaus, thin
from .sided_process import IPSI, SidedSurvivalProcess
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


def _log_gain_posterior_mode(
    base_rates: np.ndarray, history_rates: np.ndarray, fish: np.ndarray, exposure: np.ndarray, shape: float, rate: float, tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per fish, the mode v of the integrand of gamma_gain_evidence_by_fish in v = log g
    and minus its second derivative there (Newton steps, capped at 1, from the mode without history rates).
    """
    n_fish, total_rate = len(exposure), exposure + rate
    per_fish = lambda w: np.bincount(fish, weights=w, minlength=n_fish)
    v = np.log((shape + np.bincount(fish, minlength=n_fish)) / total_rate)
    for _ in range(100):
        driven = np.exp(v)[fish] * base_rates
        q = np.divide(driven, driven + history_rates, out=np.ones_like(driven), where=driven + history_rates > 0)
        grad = shape + per_fish(q) - total_rate * np.exp(v)
        hess = np.minimum(per_fish(q * (1 - q)) - total_rate * np.exp(v), -shape)
        step = np.clip(grad / hess, -1.0, 1.0)
        v = v - step
        if np.max(np.abs(step), initial=0.0) < tol:
            break
    return v, shape + per_fish(q**2)


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

    Generalized Gauss-Laguerre for the Gamma with the same mode and curvature in
    log g as the integrand, so that the rest of it is flat there: accurate for fish
    with hundreds of events whatever share of them the history rates explain (exact when h = 0);
    within about 1e-3 for a few events mostly explained by h, whose integrand keeps a tail at g = 0.
    """
    n_fish, total_rate = len(exposure), exposure + rate
    mode, a = _log_gain_posterior_mode(base_rates, history_rates, fish, exposure, shape, rate)
    beta = a * np.exp(-mode)
    x, log_w = gauss_laguerre(n_nodes, a - 1)
    g = x / beta[:, None]
    log_terms = log_w + x + (1 - a)[:, None] * np.log(x) + (shape - 1) * np.log(g) - total_rate[:, None] * g
    for start in range(0, len(fish), _EVENT_CHUNK):
        chunk_fish = fish[start : start + _EVENT_CHUNK]
        logs = np.log(np.maximum(g[chunk_fish] * base_rates[start : start + _EVENT_CHUNK, None] + history_rates[start : start + _EVENT_CHUNK, None], 1e-300))
        starts = np.flatnonzero(np.concatenate([[True], chunk_fish[1:] != chunk_fish[:-1]]))
        log_terms[chunk_fish[starts]] += np.add.reduceat(logs, starts, axis=0)
    peak = log_terms.max(axis=1, keepdims=True)
    terms = np.exp(log_terms - peak)
    total_terms = terms.sum(axis=1)
    log_z = shape * np.log(rate) - gammaln(shape) - np.log(beta) + peak[:, 0] + np.log(total_terms)
    return log_z, np.sum(terms * g, axis=1) / total_terms


def sequential_drive_taus(
    gain: Union["GammaGain", "ZeroInflatedGain"], fish: np.ndarray, drive_exposure: np.ndarray, burst_exposure: np.ndarray,
    drive_rates: np.ndarray, burst_rates: np.ndarray, censored: np.ndarray,
) -> np.ndarray:
    """
    Time rescaling with a gain g on the driven rate, each fish's intervals in order
    (rates: of the event ending each, which a censored one has not): tau = burst exposure
    - log E[exp(-g drive exposure)] over g's posterior given the fish's earlier intervals.
    """
    taus = np.empty(len(fish))
    bounds = np.flatnonzero(np.concatenate([[True], fish[1:] != fish[:-1], [True]]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        S, ended = drive_exposure[a:b], ~censored[a:b]
        g, log_w = gain.log_nodes(float(ended.sum()), float(S.sum()))
        logs = np.where(ended[:, None], np.log(np.maximum(g * drive_rates[a:b, None] + burst_rates[a:b, None], 1e-300)), 0.0)
        log_terms = log_w + np.cumsum(logs, axis=0) - logs - (np.cumsum(S) - S)[:, None] * g
        taus[a:b] = burst_exposure[a:b] + logsumexp(log_terms, axis=1) - logsumexp(log_terms - S[:, None] * g, axis=1)
    return np.maximum(taus, 0.0)


def sequential_trial_drive_taus(
    gain: Union["GammaGain", "ZeroInflatedGain"], a: float, stream: np.ndarray, num_trials: int, drive_exposure: np.ndarray,
    burst_exposure: np.ndarray, drive_rates: np.ndarray, burst_rates: np.ndarray, censored: np.ndarray, block: int = 32,
) -> np.ndarray:
    """
    sequential_drive_taus with a trial gain h ~ Gamma(a, a) too (rate g h d + b), each stream's intervals in order
    (its last censored): over the posterior of g given the fish's earlier trials and of h given the trial's earlier
    intervals. Given g, h is exact: a trial's first j events give prod(d + b) sum_k P_jk (g h)^k, P_j their
    Poisson-binomial distribution (probabilities d / (d + b)), and E[h^k e^(-g h X)] = a^a (a)_k (a + g X)^-(a + k).
    Nodes in g for what trials with their own gains tell about it (n a / (n + a) events per trial of n), fine
    enough for a trial's own factor (its knee in log g is about 2 / sqrt(n) wide).
    """
    taus = np.empty(len(stream))
    share = np.divide(drive_rates, drive_rates + burst_rates, out=np.ones_like(drive_rates), where=drive_rates + burst_rates > 0)
    ends = np.flatnonzero(censored) + 1
    starts = np.concatenate([[0], ends[:-1]])
    fish = stream[starts] // num_trials
    bounds = np.flatnonzero(np.concatenate([[True], fish[1:] != fish[:-1], [True]]))

    def trial_factor(log_coef: np.ndarray, g: np.ndarray, X: np.ndarray) -> np.ndarray:
        """log sum_k P_jk (a)_k g^k (a + g X_j)^-(a + k), rows j x nodes (log_coef: log P_jk (a)_k)."""
        log_c = np.log(a + g * X[:, None])
        log_x = np.log(np.maximum(g, 1e-300)) - log_c
        k = np.arange(log_coef.shape[1])[None, :, None]
        return np.concatenate([logsumexp(log_coef[i : i + block, :, None] + k * log_x[i : i + block, None, :], axis=1)
                               for i in range(0, len(X), block)]) - a * log_c

    for f0, f1 in zip(bounds[:-1], bounds[1:]):
        trials = list(zip(starts[f0:f1], ends[f0:f1]))
        n = np.array([e - s - 1 for s, e in trials], dtype=float)
        n_eff = max(float(np.sum(n * a / (n + a))), n.max() / 8)
        exposure = float(drive_exposure[starts[f0] : ends[f1 - 1]].sum())
        g, W = gain.log_nodes(n_eff, exposure * n_eff / max(n.sum(), 1.0))
        for s, e in trials:
            L = e - s
            P = np.zeros((L + 1, L + 1))
            P[0, 0] = 1.0
            for j, q in enumerate(np.where(censored[s:e], 0.0, share[s:e])):
                P[j + 1] = P[j] * (1 - q)
                P[j + 1, 1:] += P[j, :-1] * q
            with np.errstate(divide="ignore"):
                log_coef = np.log(P) + gammaln(a + np.arange(L + 1)) - gammaln(a)
            X = np.cumsum(drive_exposure[s:e])
            before = logsumexp(W + trial_factor(log_coef[:L], g, X - drive_exposure[s:e]), axis=1)
            after = trial_factor(log_coef[:L], g, X)
            taus[s:e] = burst_exposure[s:e] + before - logsumexp(W + after, axis=1)
            W = W + trial_factor(log_coef[L:], g, X[-1:])[0]
            keep = W > W.max() - 50.0
            g, W = g[keep], W[keep]
    return np.maximum(taus, 0.0)


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

    def log_nodes(self, n_events: float, exposure: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gains g and log weights w with sum w f(g) = integral of p(g) f(g) dg for the smooth
        f of time rescaling: the trapezoid rule in log g (spectrally accurate) over the prior
        and the posterior after n_events on exposure, at most 0.7 of its width apart, and
        the prior mass below them at g = 0.
        """
        r, b = self.shape, self.rate
        ends = lambda shape, rate: np.log(gamma_distribution.ppf([1e-10, 1 - 1e-10], shape, scale=1 / rate))
        (lo_prior, hi_prior), (lo_post, hi_post) = ends(r, b), ends(r + n_events, b + exposure)
        lo = max(min(lo_prior, lo_post), np.log((r + n_events) / (b + exposure)) - 20.0)
        step = 0.7 / np.sqrt(r + n_events)
        v = np.arange(lo, max(hi_prior, hi_post) + step, step)
        log_w = r * np.log(b) - gammaln(r) + r * v - b * np.exp(v) + np.log(step)
        log_w[[0, -1]] += np.log(0.5)
        return np.append(np.exp(v), 0.0), np.append(log_w, gamma_distribution.logcdf(np.exp(lo), r, scale=1 / b))

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

    def log_nodes(self, n_events: float, exposure: float) -> Tuple[np.ndarray, np.ndarray]:
        """GammaGain.log_nodes of the responders, and the non-responders' gain c."""
        g, log_w = self.responder.log_nodes(n_events, exposure)
        return np.append(g, self.c), np.append(np.log(max(1.0 - self.pi, 1e-300)) + log_w, np.log(max(self.pi, 1e-300)))

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

    def stream_likelihood_terms(self, dataset: PointProcessDataset, params: List[float]):
        raise NotImplementedError(f"{self.name} already integrates a fish gain.")

    def _draw_fish_gains(self, num_fish: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        return self._gain().draw(rng, (num_fish, n_sims))

    _fish_draws = PointProcess._fish_draws  # its own gains, not the base process's

    @property
    def dispersion_r(self) -> float:
        return self._gain().dispersion

    @property
    def stream_gain_variance(self) -> float:
        return 1.0 / self.dispersion_r


class Frailty(_GainWrapper):
    """
    lambda_f = g_f * lambda_base. Needs base_process.mixed_effects_likelihood_terms
    (a gain on the whole intensity). predict and the compensators are the
    population average (E[g] = 1); use estimate_fish_gains per fish.

    With a base whose gain scales its driven rate only (gain_scales_drive_only,
    e.g. a renewal process with bursts), lambda_f = drive * g_f + rest, g_f
    integrated by quadrature (gamma_gain_evidence_by_fish), with a trial gain
    g_f h_fm by nested quadrature (log_nested_drive_marginal).

    trial_gain (Gamma gains only): a second gain per trial, h_fm ~ Gamma(a, a)
    (a: 'r_trial'), lambda_fm = g_f h_fm lambda_base: a fish's engagement varies
    from trial to trial around its own level. Needs
    base_process.stream_likelihood_terms; h is integrated in closed form, g by
    adaptive Gauss-Hermite quadrature on log g. Time rescaling conditions on both
    (sequential_trial_drive_taus).
    """

    def __init__(self, base_process: PointProcess, gains: GainFamily, trial_gain: bool = False, a_init: float = 5.0) -> None:
        super().__init__(base_process, gains)
        self.trial_gain = trial_gain
        trial = "+Trial" if trial_gain else ""
        self.name = f"{gains.prefix}GammaMixedEffects{trial}[{base_process.name}]{gains.suffix}"
        self.latex_formula = rf"${gain_formula(base_process, trial_gain)},\ {gains.latex}" + (r",\ h_{fm}\sim\Gamma(a,a)$" if trial_gain else "$")
        if trial_gain:
            self.param_names = self.param_names + ["r_trial"]
            self.initial_guesses = self.initial_guesses + [a_init]
            self.bounds = self.bounds + [(1e-2, 1e3)]
            nodes, weights = np.polynomial.hermite_e.hermegauss(15)
            self._gain_nodes, self._gain_log_weights = nodes, np.log(weights / weights.sum())
            self._gain_mode = np.zeros(0)  # posterior log fish gains of the last call: where Newton starts

    def _r_trial(self, params: Optional[List[float]] = None) -> float:
        params = self.params_ if params is None else params
        return float(params[self._n_base + len(self.gains.param_names)])

    def supports_jax(self) -> bool:
        base = self.base_process
        if not base.supports_jax():
            return False
        if base.gain_scales_drive_only:
            return (type(self.gains) is GammaFamily or not self.trial_gain) and hasattr(base, "jax_drive_likelihood_terms")
        return hasattr(base, "jax_stream_terms")

    def _jax_zero_inflated(self, params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ZeroInflatedFamily in JAX: pi, c, and the responders' shape and rate."""
        tail, family = params[self._n_base :], self.gains
        pi, r = sigmoid_bounded(tail[0], 1.0), jnp.maximum(tail[1], 1e-8)
        c = sigmoid_bounded(tail[2], family.c_upper) if family.fit_c else family.c_init
        mu = (1.0 - pi * c) / jnp.maximum(1.0 - pi, 1e-12)
        return pi, c, r, r / mu

    def _jax_log_marginal(self, params: jnp.ndarray, N: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
        """The gain family's log marginal per fish of N events and S expected events (gain 1)."""
        if isinstance(self.gains, GammaFamily):
            r = jnp.maximum(params[self._n_base], 1e-8)
            return gains_jax.log_gamma_marginal(N, S, r, r)
        pi, c, shape, rate = self._jax_zero_inflated(params)
        return gains_jax.log_zero_inflated_marginal(N, S, pi, c, not self.gains.fit_c and self.gains.c_init <= 0.0, shape, rate)

    def jax_arrays(self, dataset: PointProcessDataset) -> Arrays:
        arrays = {"base": self.base_process._arrays(dataset), "active": jnp.asarray(dataset.fish_trial_mask.any(axis=1))}
        static = dict(num_trials=dataset.num_trials, num_fish=dataset.num_fish)
        if self.trial_gain and not self.base_process.gain_scales_drive_only:
            arrays["mask"] = jnp.asarray(dataset.fish_trial_mask, dtype=float)
        elif self.trial_gain:  # each event's index in its stream, for the Poisson-binomial distribution
            base = self.base_process
            stream = np.asarray(base.jax_drive_likelihood_terms(jnp.asarray(base.initial_guesses, dtype=float), arrays["base"])[3])
            position = np.arange(len(stream)) - np.searchsorted(stream, stream)  # padded already: phantom events are dropped
            arrays.update(position=jnp.asarray(position), mask=jnp.asarray(dataset.fish_trial_mask, dtype=float))
            real = stream[stream < dataset.num_fish * dataset.num_trials]
            block = gains_jax.POISSON_BINOMIAL_BLOCK
            static["max_events"] = -(-padded(int(np.bincount(real, minlength=1).max()), minimum=block) // block) * block
        return Arrays(arrays, static)

    def jax_nll(self, params: jnp.ndarray, a: Arrays) -> jnp.ndarray:
        if not self.base_process.gain_scales_drive_only:
            log_rates, n, S = self.base_process.jax_stream_terms(params[: self._n_base], a["base"])
            if self.trial_gain:
                r = jnp.maximum(params[self._n_base], 1e-8)
                nested = gains_jax.log_nested_gamma_marginal(
                    n, S, a["mask"], r, params[self._n_base + 1], jnp.asarray(self._gain_nodes), jnp.asarray(self._gain_log_weights)
                )
                return -(log_rates + jnp.sum(nested))
            return -(log_rates + jnp.sum(jnp.where(a["active"], self._jax_log_marginal(params, n.sum(axis=1), S.sum(axis=1)), 0.0)))
        constant, drive_rates, burst_rates, stream, drive_exposure, burst_exposure = self.base_process.jax_drive_likelihood_terms(
            params[: self._n_base], a["base"]
        )
        r = jnp.maximum(params[self._n_base], 1e-8)
        if self.trial_gain:
            log_z = gains_jax.log_nested_drive_marginal(
                drive_rates, burst_rates, stream, a["position"], a["max_events"], drive_exposure, a["mask"],
                r, params[self._n_base + 1], jnp.asarray(self._gain_nodes), jnp.asarray(self._gain_log_weights),
            )
        elif isinstance(self.gains, GammaFamily):
            log_z = gains_jax.gamma_gain_evidence_by_fish(drive_rates, burst_rates, stream // a["num_trials"], drive_exposure.sum(axis=1), r, r)
        else:
            log_z = gains_jax.zero_inflated_gain_evidence_by_fish(
                drive_rates, burst_rates, stream // a["num_trials"], drive_exposure.sum(axis=1), *self._jax_zero_inflated(params)
            )
        return -(constant + jnp.sum(jnp.where(a["active"], log_z - burst_exposure.sum(axis=1), 0.0)))

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        if self.base_process.gain_scales_drive_only:
            constant, drive_rates, burst_rates, stream, drive_exposure, burst_exposure = self.base_process.drive_likelihood_terms(
                dataset, self._base_params(params)
            )
            active = dataset.fish_trial_mask.any(axis=1)
            if self.trial_gain:
                log_z, self._gain_mode = log_nested_drive_marginal(
                    drive_rates, burst_rates, stream, drive_exposure, dataset.fish_trial_mask, self._gain(params).shape,
                    self._r_trial(params), self._gain_nodes, self._gain_log_weights, self._fish_modes(dataset),
                )
            else:
                fish = stream // dataset.num_trials
                log_z, _ = self._gain(params).evidence_by_fish(drive_rates, burst_rates, fish, drive_exposure.sum(axis=1), n_nodes=30)
            return -(constant + float(np.sum(log_z[active] - burst_exposure.sum(axis=1)[active])))
        if self.trial_gain:
            base_ll, n, S = self.base_process.stream_likelihood_terms(dataset, self._base_params(params))
            log_marginal, self._gain_mode = log_nested_gamma_marginal(
                n, S, dataset.fish_trial_mask, self._gain(params).shape, self._r_trial(params),
                self._gain_nodes, self._gain_log_weights, self._fish_modes(dataset),
            )
            return -(base_ll + float(np.sum(log_marginal)))
        base_ll, N, S = self.base_process.mixed_effects_likelihood_terms(dataset, self._base_params(params))
        active = dataset.fish_trial_mask.any(axis=1)
        return -(base_ll + float(np.sum(self._gain(params).log_marginal(N[active], S[active]))))

    def conditional_rate(self, dataset: PointProcessDataset, side: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Rate (Hz, trial x time bin) given each fish's observed bouts and its own gains,
        their posterior means per fish and trial: the time course the model predicts
        for these fish, of all events or of one side's. None without bout history or
        when the gain scales the whole rate.
        """
        if not (self.uses_history and self.base_process.gain_scales_drive_only):
            return None
        self._nll(list(self.params_), dataset)  # posterior modes of this dataset's fish
        _, d, b, stream, S, _ = self.base_process.drive_likelihood_terms(dataset, self._base_params(self.params_))
        if self.trial_gain:
            gains = posterior_stream_gains(d, b, stream, S, dataset.fish_trial_mask, self._gain().shape, self._r_trial(),
                                           self._gain_nodes, self._gain_log_weights, self._fish_modes(dataset))
        else:
            _, fish_gains = self._gain().evidence_by_fish(d, b, stream // dataset.num_trials, S.sum(axis=1), n_nodes=30)
            gains = np.repeat(fish_gains[:, None], dataset.num_trials, axis=1)
        renewal = getattr(self.base_process, "ground", self.base_process)
        rate, n = np.zeros((dataset.num_trials, len(dataset.t_centers))), np.zeros(dataset.num_trials)
        for f, m, t_ev in dataset.iter_streams():
            stream_rate = renewal.predict(dataset.t_centers, m, dataset.history_times(f, m), t_ev, gain=gains[f, m])
            rate[m] += stream_rate if side is None else stream_rate * self._side_share(dataset.t_centers, m, f, side)
            n[m] += 1
        return rate / np.maximum(n, 1)[:, None]

    def _side_share(self, t: np.ndarray, trial: int, fish: int, side: int) -> np.ndarray:
        """Share of a fish's events to one side at t."""
        p = self.base_process.ipsi_probability(t, trial)
        return p if side == IPSI else 1 - p

    def _fish_modes(self, dataset: PointProcessDataset) -> np.ndarray:
        """Posterior log fish gains of the last call (where Newton starts), zeros for a new dataset size."""
        return self._gain_mode if len(self._gain_mode) == dataset.num_fish else np.zeros(dataset.num_fish)

    @property
    def stream_gain_variance(self) -> float:
        """Var(g h) = (1 + 1/r)(1 + 1/a) - 1 with a trial gain."""
        fish = super().stream_gain_variance
        return (1 + fish) * (1 + 1 / self._r_trial()) - 1 if self.trial_gain else fish

    def _stream_gain(self, gain: float, rng: np.random.Generator) -> float:
        """A stream's gain: the fish's, times its trial's when there is a trial gain."""
        if not self.trial_gain:
            return gain
        a = self._r_trial()
        return gain * rng.gamma(a, 1.0 / a)

    def simulate_stream(self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator, **history):
        return self.base_process.simulate_stream(dataset, t_idx, self._stream_gain(gain, rng), rng, **history)

    def _stream_tau_values(self, dataset: PointProcessDataset) -> StreamTaus:
        """
        Predictable compensator: each fish's trials in order, the gain
        posterior conditioned on events and exposure strictly before each probe.
        """
        if self.trial_gain or self.base_process.gain_scales_drive_only:
            return self._drive_stream_tau_values(dataset)
        gain = self._gain()
        result: StreamTaus = {}
        for f_idx, streams in groupby(dataset.iter_streams(), key=lambda stream: stream[0]):
            S_offset = S_prev = 0.0
            n_events = 0
            for _, t_idx, t_ev in streams:
                _, cum, last_censored, exposure = self.base_process.stream_compensator_profile(
                    t_ev, t_idx, dataset.duration_s, history=dataset.history_times(f_idx, t_idx)
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

    def _drive_stream_tau_values(self, dataset: PointProcessDataset) -> StreamTaus:
        """
        Time rescaling with a gain on the driven rate only (sequential_drive_taus), the
        gain posterior carried over the fish's trials (as BaselineFrailty); with a trial
        gain, sequential_trial_drive_taus, the whole rate driven unless the base says otherwise.
        """
        if self.base_process.gain_scales_drive_only:
            drive, burst, stream, censored, drive_rates, burst_rates = self.base_process.interval_drive_terms(dataset)
        else:
            drive, stream, censored = self._whole_rate_intervals(dataset)
            burst, drive_rates, burst_rates = np.zeros_like(drive), np.ones_like(drive), np.zeros_like(drive)
        if self.trial_gain:
            taus = sequential_trial_drive_taus(self._gain(), self._r_trial(), stream, dataset.num_trials, drive, burst, drive_rates, burst_rates, censored)
        else:
            taus = sequential_drive_taus(self._gain(), stream // dataset.num_trials, drive, burst, drive_rates, burst_rates, censored)
        ends = np.flatnonzero(censored)[:-1] + 1
        return {divmod(int(s[0]), dataset.num_trials): list(zip(t.tolist(), c.tolist()))
                for s, t, c in zip(np.split(stream, ends), np.split(taus, ends), np.split(censored, ends))}

    def _whole_rate_intervals(self, dataset: PointProcessDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The base's compensator over every interval (streams in order, each one's last censored), its stream, whether censored."""
        exposures, streams = [], []
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            _, cum, _, _ = self.base_process.stream_compensator_profile(t_ev, t_idx, dataset.duration_s, history=dataset.history_times(f_idx, t_idx))
            exposures.append(np.diff(cum, prepend=0.0))
            streams.append(np.full(len(cum), f_idx * dataset.num_trials + t_idx))
        stream = np.concatenate(streams)
        return np.concatenate(exposures), stream, np.append(stream[1:] != stream[:-1], True)

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        """Whole-session posterior per fish (for reporting, not for time rescaling)."""
        if self.base_process.gain_scales_drive_only:
            _, drive_rates, burst_rates, stream, drive_exposure, _ = self.base_process.drive_likelihood_terms(dataset, self._base_params(self.params_))
            fish = stream // dataset.num_trials
            _, gains = self._gain().evidence_by_fish(drive_rates, burst_rates, fish, drive_exposure.sum(axis=1), n_nodes=30)
            active = np.flatnonzero(dataset.fish_trial_mask.any(axis=1))
            return pd.DataFrame({"fish_idx": active, "n_events": np.bincount(fish, minlength=dataset.num_fish)[active], "estimated_gain": gains[active]})
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


def log_normal_mixture(
    log_f: Callable[[np.ndarray], np.ndarray], derivatives: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    s: float, nodes: np.ndarray, log_weights: np.ndarray, mode: np.ndarray, tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    log of the integral of f_f(u) N(u; 0, s^2) du for every fish f, and the posterior
    modes, by Gauss-Hermite quadrature centred on each fish's posterior mode and scaled
    to its curvature (adaptive: fish with many events have a sharp posterior the fixed
    rule would miss). log_f maps (fish x nodes) u to log f; derivatives gives its first
    and second derivatives at (fish,) u; log f must be concave. nodes, log_weights: the
    rule for N(0, 1). Newton starts from `mode` (the previous call's: a few steps then
    suffice) and its steps are capped at 1 (log units): far from the mode a concave
    function's curvature can be nearly flat, and a full step would overshoot.
    """
    for _ in range(100):
        grad, hess = derivatives(mode)
        step = np.clip((grad - mode / s**2) / (hess - 1 / s**2), -1.0, 1.0)
        mode = mode - step
        if np.max(np.abs(step), initial=0.0) < tol:
            break
    sd = 1 / np.sqrt(1 / s**2 - hess)
    u = mode[:, None] + sd[:, None] * nodes
    return logsumexp(log_weights + log_f(u) - u**2 / (2 * s**2) + nodes**2 / 2, axis=1) + np.log(sd / s), mode


def _log_gain_mode(
    n: np.ndarray, S: np.ndarray, mask: np.ndarray, r: float, a: float, mode: np.ndarray, tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """Mode and curvature scale, per fish, of the closed-form integrand of log_nested_gamma_marginal in v = log g (Newton)."""
    w, N = (n + a) * mask, (n * mask).sum(axis=1)
    for _ in range(100):
        e = np.exp(mode)
        share = S * e[:, None] / (a + S * e[:, None])
        hess = -r * e - (w * share * (1 - share)).sum(axis=1)
        step = np.clip((r + N - r * e - (w * share).sum(axis=1)) / hess, -1.0, 1.0)
        mode = mode - step
        if np.max(np.abs(step), initial=0.0) < tol:
            break
    return mode, 1 / np.sqrt(-hess)


def log_nested_gamma_marginal(
    n: np.ndarray, S: np.ndarray, mask: np.ndarray, r: float, a: float,
    nodes: np.ndarray, log_weights: np.ndarray, mode: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per fish, the log of the integral over g ~ Gamma(r, r) of the product over its
    trials of the integral over h ~ Gamma(a, a) of (g h)^n e^(-g h S); n, S: events
    and expected events (g = h = 1) per stream, fish x trial; mask: recorded streams.
    h in closed form; v = log g by Gauss-Hermite quadrature centred on the
    (log-concave) integrand's mode and scaled to its curvature. Also returns the
    modes, where the next call's Newton starts.
    """
    mode, sd = _log_gain_mode(n, S, mask, r, a, mode)
    v = mode[:, None] + sd[:, None] * nodes
    w, N = (n + a) * mask, (n * mask).sum(axis=1)
    log_integrand = (r + N)[:, None] * v - r * np.exp(v) - np.einsum("ft,ftk->fk", w, np.log(a + S[:, :, None] * np.exp(v)[:, None, :]))
    constant = r * np.log(r) - gammaln(r) + (mask * (a * np.log(a) - gammaln(a) + gammaln(n + a))).sum(axis=1)
    log_integral = logsumexp(log_weights + log_integrand + nodes**2 / 2, axis=1) + np.log(sd) + 0.5 * np.log(2 * np.pi)
    return log_integral + constant, mode


def poisson_binomial(p: np.ndarray, stream: np.ndarray, n_streams: int) -> np.ndarray:
    """
    Per stream, the distribution of the number of successes among its events
    (success probabilities p; stream non-decreasing): n_streams x (max events + 1).
    """
    counts = np.bincount(stream, minlength=n_streams)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pmf = np.zeros((n_streams, counts.max(initial=0) + 1))
    pmf[:, 0] = 1.0
    for j in range(counts.max(initial=0)):
        rows = np.flatnonzero(counts > j)
        q = p[first[rows] + j][:, None]
        pmf[rows, 1 : j + 2] = pmf[rows, 1 : j + 2] * (1 - q) + pmf[rows, : j + 1] * q
        pmf[rows, 0] *= 1 - q[:, 0]
    return pmf


def log_nested_drive_marginal(
    drive_rates: np.ndarray, burst_rates: np.ndarray, stream: np.ndarray, S: np.ndarray, mask: np.ndarray,
    r: float, a: float, nodes: np.ndarray, log_weights: np.ndarray, mode: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per fish, the log of the integral over g ~ Gamma(r, r) of the product over its
    trials of the integral over h ~ Gamma(a, a) of e^(-g h S) prod over the trial's
    events (g h d + b); d, b: driven and burst rate at every event, stream: its
    stream (fish * trials + trial, non-decreasing), S: driven exposure, fish x trial.
    The product is prod(d + b) sum_k P_k (g h)^k, P the Poisson-binomial distribution
    of the driven events (probabilities d / (d + b)), so h is exact:
    a^a prod(d + b) (a + g S)^-a sum_k P_k (a)_k x^k, x = g / (a + g S) (Horner; in log space where it under- or overflows).
    v = log g by Gauss-Hermite centred and scaled as log_nested_gamma_marginal's
    with every event counted by its driven share (at the gain the centre is found for,
    so the result does not depend on the starting modes). Also returns the modes.
    """
    _, _, _, _, sd, log_integrand, mode = _nested_drive_integrand(drive_rates, burst_rates, stream, S, mask, r, a, nodes, mode)
    log_integral = logsumexp(log_weights + log_integrand + nodes**2 / 2, axis=1) + np.log(sd) + 0.5 * np.log(2 * np.pi)
    return log_integral + r * np.log(r) - gammaln(r), mode


def posterior_stream_gains(
    drive_rates: np.ndarray, burst_rates: np.ndarray, stream: np.ndarray, S: np.ndarray, mask: np.ndarray,
    r: float, a: float, nodes: np.ndarray, log_weights: np.ndarray, mode: np.ndarray,
) -> np.ndarray:
    """
    Posterior mean of g h per stream (fish x trial), the arguments as for
    log_nested_drive_marginal: E[g h] = sum over the nodes of g of their posterior
    weight times g E[h | g] = x sum_k P_k (a)_k (a + k) x^k / sum_k P_k (a)_k x^k.
    """
    log_coef, k, _, log_x, _, log_integrand, _ = _nested_drive_integrand(drive_rates, burst_rates, stream, S, mask, r, a, nodes, mode)
    log_post = log_weights + log_integrand + nodes**2 / 2
    post = np.exp(log_post - logsumexp(log_post, axis=1, keepdims=True)).repeat(S.shape[1], axis=0)  # streams x nodes
    terms = log_coef[:, None, :] + k * log_x[:, :, None]  # streams x nodes x k
    log_gh = log_x + logsumexp(terms + np.log(a + k), axis=2) - logsumexp(terms, axis=2)  # g E[h | g] = x sum(... (a + k)) / sum(...)
    return (post * np.exp(log_gh)).sum(axis=1).reshape(S.shape)


def _nested_drive_integrand(
    drive_rates: np.ndarray, burst_rates: np.ndarray, stream: np.ndarray, S: np.ndarray, mask: np.ndarray,
    r: float, a: float, nodes: np.ndarray, mode: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """log_nested_drive_marginal's pieces: log P_k (a)_k per stream, k, the log gains v at the nodes
    (fish x nodes), log x (streams x nodes), the nodes' scale and log integrand per fish, the modes."""
    n_fish, n_trials = S.shape
    total = np.maximum(drive_rates + burst_rates, 1e-300)
    pmf = poisson_binomial(drive_rates / total, stream, S.size)
    k = np.arange(pmf.shape[1])
    with np.errstate(divide="ignore"):
        log_coef = np.log(pmf) + gammaln(a + k) - gammaln(a)  # P_k (a)_k
    for _ in range(100):  # the driven shares depend on the gains: iterate both to their fixed point
        g_events = np.exp(mode)[stream // n_trials] * drive_rates
        share = np.bincount(stream, weights=g_events / np.maximum(g_events + burst_rates, 1e-300), minlength=S.size).reshape(S.shape)
        previous, (mode, sd) = mode, _log_gain_mode(share, S, mask, r, a, mode)
        if np.max(np.abs(mode - previous), initial=0.0) < 1e-10:
            break
    v = mode[:, None] + sd[:, None] * nodes  # fish x nodes
    log_c = np.log(a + np.exp(v).repeat(n_trials, axis=0) * S.ravel()[:, None])  # streams x nodes
    log_x = v.repeat(n_trials, axis=0) - log_c
    peak = log_coef.max(axis=1, keepdims=True)
    coef, x, poly = np.exp(log_coef - peak), np.exp(log_x), 0.0
    for j in k[::-1]:  # Horner: sum_k coef_k x^k
        poly = poly * x + coef[:, j : j + 1]
    with np.errstate(divide="ignore"):
        log_poly = peak + np.log(poly)
    bad = np.nonzero(~np.isfinite(log_poly))  # every term under- or overflowed: the same sum in log space
    if len(bad[0]):
        log_poly[bad] = logsumexp(log_coef[bad[0]] + k * log_x[bad][:, None], axis=1)
    log_total = np.bincount(stream, weights=np.log(total), minlength=S.size)
    log_z = (a * np.log(a) + log_total)[:, None] - a * log_c + log_poly
    log_integrand = r * v - r * np.exp(v) + (mask.ravel()[:, None] * log_z).reshape(n_fish, n_trials, -1).sum(axis=1)
    return log_coef, k, v, log_x, sd, log_integrand, mode


def gain_formula(base: PointProcess, trial_gain: bool) -> str:
    """The base's formula (LaTeX, no $) with the gains where they act: its driven rate only, or all of it."""
    gain = r"g_f h_{fm}" if trial_gain else "g_f"
    if base.gain_scales_drive_only:
        return base.latex_with_gain(gain).strip("$")
    return rf"\left[{base.latex_formula.strip('$')}\right] \times {gain}"


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """log(sigmoid(x)), stable, and twice as fast as -np.logaddexp(0, -x)."""
    return -np.log1p(np.exp(-np.abs(x))) - np.maximum(-x, 0)


class SideFrailty(Frailty):
    """
    Frailty of a sided process with a second fish effect: its own side
    preference. Fish f has a gain g_f (Gamma(r, r), or another gain family) on its
    rate of events to either side and a shift u_f ~ N(0, s^2) (s: 's_side') of the log odds that
    an event goes to the preferred side:

        P(ipsi | event at t, m, fish f) = sigmoid(log(lambda_ipsi / lambda_contra) + u_f)
        escapes: P(preferred) = sigmoid(log D + u_f)

    u_f leaves a fish's total rate alone, so B and D describe the typical fish
    (u = 0); the side rates and probabilities this model reports average over u.
    g_f is integrated in closed form, u_f by adaptive Gauss-Hermite quadrature.
    """

    def __init__(
        self, base_process: PointProcess, r_init: float = 5.0, s_init: float = 0.5, n_nodes: int = 15, trial_gain: bool = False,
        gains: Optional[GainFamily] = None,
    ) -> None:
        """gains: the fish gain's family (default Gamma(r, r), r_init)."""
        gains = gains or GammaFamily(r_init)
        super().__init__(base_process, gains, trial_gain=trial_gain)
        self.param_names = self.param_names + ["s_side"]
        self.initial_guesses = self.initial_guesses + [s_init]
        self.bounds = self.bounds + [(1e-3, 5.0)]
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
        self._nodes, self._log_weights = nodes, np.log(weights / weights.sum())
        self._mode = np.zeros(0)  # posterior side shifts of the last fit's fish: where the next Newton starts
        self.name = f"{gains.prefix}GammaSideMixedEffects{'+Trial' if trial_gain else ''}[{base_process.name}]{gains.suffix}"
        side = (r"P_{preferred} = \sigma(\log D + u_f)" if isinstance(base_process, SidedSurvivalProcess)
                else r"P_{ipsi} = \sigma(\log(\lambda_{ipsi}/\lambda_{contra}) + u_f)")
        self.latex_formula = (
            rf"${gain_formula(base_process, trial_gain)},\ {gains.latex},\ "
            + side
            + r",\ u_f\sim N(0,s^2)"
            + (r",\ h_{fm}\sim\Gamma(a,a)$" if trial_gain else "$")
        )

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        fish, log_odds, preferred = self.base_process.side_terms(dataset, self._base_params(params))
        to_fish = dataset.cached(("side_events_to_fish", len(fish)), lambda: csr_matrix(
            (np.ones(len(fish)), (fish, np.arange(len(fish)))), shape=(dataset.num_fish, len(fish))
        ))
        sign = np.where(preferred, 1.0, -1.0)
        observed = lambda u: sign[:, None] * (log_odds[:, None] + u[fish])  # log odds of each event's own side
        log_f = lambda u: to_fish @ log_sigmoid(observed(u))

        def derivatives(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            other = expit(-sign * (log_odds + u[fish]))
            return to_fish @ (sign * other), -(to_fish @ (other * (1 - other)))

        if len(self._mode) != dataset.num_fish:
            self._mode = np.zeros(dataset.num_fish)
        log_f0 = log_f(np.zeros((dataset.num_fish, 1)))  # the base model already has the sides at u = 0
        sides, self._mode = log_normal_mixture(
            lambda u: log_f(u) - log_f0, derivatives, params[-1], self._nodes, self._log_weights, self._mode
        )
        return super()._nll(params, dataset) - float(np.sum(sides))

    def supports_jax(self) -> bool:
        return super().supports_jax() and hasattr(self.base_process, "jax_side_terms")

    def jax_nll(self, params: jnp.ndarray, a: Arrays) -> jnp.ndarray:
        fish, log_odds, preferred = self.base_process.jax_side_terms(params[: self._n_base], a["base"])
        sides = gains_jax.side_mixture(fish, log_odds, preferred, params[-1], a["num_fish"], jnp.asarray(self._nodes), jnp.asarray(self._log_weights))
        return super().jax_nll(params, a) - jnp.sum(sides)

    def _side_share(self, t: np.ndarray, trial: int, fish: int, side: int) -> np.ndarray:
        """Share of a fish's events to one side at t, at its posterior side shift (mode)."""
        p = expit(logit(self.base_process.ipsi_probability(t, trial)) + self._mode[fish])
        return p if side == IPSI else 1 - p

    def ipsi_at_nodes(self, t: np.ndarray, trial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Probability that an event is ipsi at each quadrature node of the side shift u_f (last axis), and the nodes' weights."""
        return expit(logit(self.base_process.ipsi_probability(t, trial))[..., None] + self.params_[-1] * self._nodes), np.exp(self._log_weights)

    def ipsi_probability(self, t: np.ndarray, trial: np.ndarray) -> np.ndarray:
        """Probability that an event is ipsi, averaged over fish side preferences."""
        q, w = self.ipsi_at_nodes(t, trial)
        return q @ w

    def side_share_cv2(self, dataset: PointProcessDataset, side: int) -> np.ndarray:
        """Squared coefficient of variation across fish of the share of events to one side (trial x time)."""
        q, w = self.ipsi_at_nodes(dataset.t_centers[None, :], np.arange(dataset.num_trials)[:, None])
        q = q if side == IPSI else 1 - q
        return (q**2 @ w) / (q @ w) ** 2 - 1

    def compute_expected_rate(self, dataset: PointProcessDataset, side: Optional[int] = None) -> np.ndarray:
        """Rate to either side (None) or to one, averaged over fish side preferences."""
        if self.uses_history:
            return self.simulated_rate(dataset, side)
        total = self.base_process.compute_expected_rate(dataset)
        if side is None:
            return total
        p_ipsi = self.ipsi_probability(dataset.t_centers[None, :], np.arange(dataset.num_trials)[:, None])
        return total * (p_ipsi if side == IPSI else 1 - p_ipsi)

    def _fish_draws(self, num_fish: int, rng: np.random.Generator) -> list:
        """(gain, side shift u) per fish."""
        gains = self._draw_fish_gains(num_fish, 1, rng)[:, 0]
        return list(zip(gains, rng.normal(0.0, self.params_[-1], num_fish)))

    def simulate_stream(self, dataset: PointProcessDataset, t_idx: int, draw: Tuple[float, float], rng: np.random.Generator, **history):
        gain, u = draw
        return self.base_process.simulate_stream(dataset, t_idx, self._stream_gain(gain, rng), rng, side_shift=u, **history)


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
    def __init__(self, base_process: PointProcess, r_init: float = 5.0, trial_gain: bool = False) -> None:
        super().__init__(base_process, GammaFamily(r_init), trial_gain=trial_gain)


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
