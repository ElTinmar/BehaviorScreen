"""
JAX versions of the gain integrals in frailty.py (same quadratures, so the same values).
Where a rule is centred adaptively (on a posterior mode, by an effective event count),
its placement is not differentiated: the integral does not depend on it.
"""
from typing import Tuple

import jax
from jax.scipy.special import gammaln

from .backend import jnp, segment_sum


def gauss_laguerre(n: int, alpha: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """frailty.gauss_laguerre: nodes and log-weights, one row per alpha, for the weight x^alpha e^-x."""
    k = jnp.arange(n)
    jacobi = jax.vmap(lambda d, o: jnp.diag(d) + jnp.diag(o, 1) + jnp.diag(o, -1))(
        2 * k + alpha[:, None] + 1, jnp.sqrt(k[1:] * (k[1:] + alpha[:, None]))
    )
    nodes, vectors = jnp.linalg.eigh(jacobi)
    return nodes, 2 * jnp.log(jnp.abs(vectors[:, 0, :])) + gammaln(alpha + 1)[:, None]


def gamma_gain_evidence_by_fish(
    base_rates: jnp.ndarray, history_rates: jnp.ndarray, fish: jnp.ndarray, exposure: jnp.ndarray,
    shape: jnp.ndarray, rate: jnp.ndarray, n_nodes: int = 30,
) -> jnp.ndarray:
    """frailty.gamma_gain_evidence_by_fish: log Z per fish."""
    n_fish = exposure.shape[0]
    d = exposure + rate
    driven = base_rates * (shape + segment_sum(jnp.ones_like(base_rates), fish, n_fish)[fish]) / d[fish]
    total = driven + history_rates
    k = jax.lax.stop_gradient(segment_sum(jnp.where(total > 0, driven / jnp.where(total > 0, total, 1.0), 0.0), fish, n_fish))
    x, log_w = gauss_laguerre(n_nodes, shape - 1 + k)
    g = x / d[:, None]
    logs = jnp.log(jnp.maximum(g[fish] * base_rates[:, None] + history_rates[:, None], 1e-300))
    log_terms = log_w - k[:, None] * jnp.log(g) + segment_sum(logs, fish, n_fish)
    peak = jax.lax.stop_gradient(log_terms.max(axis=1, keepdims=True))
    return shape * jnp.log(rate) - gammaln(shape) - (shape + k) * jnp.log(d) + peak[:, 0] + jnp.log(jnp.exp(log_terms - peak).sum(axis=1))


def zero_inflated_gain_evidence_by_fish(
    base_rates: jnp.ndarray, history_rates: jnp.ndarray, fish: jnp.ndarray, exposure: jnp.ndarray,
    pi: jnp.ndarray, c: jnp.ndarray, shape: jnp.ndarray, rate: jnp.ndarray, n_nodes: int = 30,
) -> jnp.ndarray:
    """ZeroInflatedGain.evidence_by_fish: log Z per fish, gain c with probability pi, else Gamma(shape, rate)."""
    nonresponder = -c * exposure + segment_sum(jnp.log(jnp.maximum(c * base_rates + history_rates, 1e-300)), fish, exposure.shape[0])
    responder = gamma_gain_evidence_by_fish(base_rates, history_rates, fish, exposure, shape, rate, n_nodes)
    return jnp.logaddexp(jnp.log(jnp.maximum(pi, 1e-300)) + nonresponder, jnp.log(jnp.maximum(1.0 - pi, 1e-300)) + responder)


def log_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return -jnp.log1p(jnp.exp(-jnp.abs(x))) - jnp.maximum(-x, 0)


def side_mixture(
    fish: jnp.ndarray, log_odds: jnp.ndarray, preferred: jnp.ndarray, s: jnp.ndarray, n_fish: int,
    nodes: jnp.ndarray, log_weights: jnp.ndarray, tol: float = 1e-10,
) -> jnp.ndarray:
    """
    Per fish, log of the integral over u ~ N(0, s^2) of prod over its events of
    sigmoid(sign (log odds + u)) / the same at u = 0 (frailty.log_normal_mixture
    of SideFrailty): Gauss-Hermite centred on the posterior mode (Newton, steps
    capped at 1) and scaled to its curvature.
    """
    sign = jnp.where(preferred, 1.0, -1.0)
    log_f = lambda u, lo: segment_sum(log_sigmoid(sign[:, None] * (lo[:, None] + u[fish])), fish, n_fish)

    def derivatives(u: jnp.ndarray, lo: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        other = jax.nn.sigmoid(-sign * (lo + u[fish]))
        return segment_sum(sign * other, fish, n_fish), -segment_sum(other * (1 - other), fish, n_fish)

    lo, s0 = jax.lax.stop_gradient(log_odds), jax.lax.stop_gradient(s)

    def newton(state: Tuple) -> Tuple:
        mode, _, _, i = state
        grad, hess = derivatives(mode, lo)
        step = jnp.clip((grad - mode / s0**2) / (hess - 1 / s0**2), -1.0, 1.0)
        return mode - step, jnp.max(jnp.abs(step), initial=0.0), hess, i + 1

    start = (jnp.zeros(n_fish), jnp.inf, jnp.zeros(n_fish), 0)
    mode, _, hess, _ = jax.lax.while_loop(lambda st: (st[1] >= tol) & (st[3] < 100), newton, start)
    sd = 1 / jnp.sqrt(1 / s0**2 - hess)
    u = mode[:, None] + sd[:, None] * nodes
    log_mixture = log_f(u, log_odds) - log_f(jnp.zeros((n_fish, 1)), log_odds)
    return jax.scipy.special.logsumexp(log_weights + log_mixture - u**2 / (2 * s**2) + nodes**2 / 2, axis=1) + jnp.log(sd / s)


def poisson_binomial(p: jnp.ndarray, stream: jnp.ndarray, position: jnp.ndarray, n_streams: int, max_events: int) -> jnp.ndarray:
    """frailty.poisson_binomial: per stream, the distribution of the number of successes; position: each event's index in its stream."""
    q = jnp.zeros((n_streams, max_events)).at[stream, position].set(p)

    def add_event(pmf: jnp.ndarray, q_j: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
        q_j = q_j[:, None]
        return jnp.concatenate([pmf[:, :1] * (1 - q_j), pmf[:, 1:] * (1 - q_j) + pmf[:, :-1] * q_j], axis=1), None

    pmf0 = jnp.zeros((n_streams, max_events + 1)).at[:, 0].set(1.0)
    return jax.lax.scan(jax.checkpoint(add_event), pmf0, q.T)[0]


def _log_gain_mode(n: jnp.ndarray, S: jnp.ndarray, mask: jnp.ndarray, r: jnp.ndarray, a: jnp.ndarray, mode: jnp.ndarray,
                   tol: float = 1e-10) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """frailty._log_gain_mode: mode and curvature scale per fish of the nested integrand in v = log g (Newton)."""
    w, N = (n + a) * mask, (n * mask).sum(axis=1)

    def newton(state: Tuple) -> Tuple:
        mode, _, _, i = state
        e = jnp.exp(mode)
        share = S * e[:, None] / (a + S * e[:, None])
        hess = -r * e - (w * share * (1 - share)).sum(axis=1)
        step = jnp.clip((r + N - r * e - (w * share).sum(axis=1)) / hess, -1.0, 1.0)
        return mode - step, jnp.max(jnp.abs(step), initial=0.0), hess, i + 1

    mode, _, hess, _ = jax.lax.while_loop(lambda st: (st[1] >= tol) & (st[3] < 100), newton, (mode, jnp.inf, -jnp.ones_like(mode), 0))
    return mode, 1 / jnp.sqrt(-hess)


def log_nested_drive_marginal(
    drive_rates: jnp.ndarray, burst_rates: jnp.ndarray, stream: jnp.ndarray, position: jnp.ndarray, max_events: int,
    S: jnp.ndarray, mask: jnp.ndarray, r: jnp.ndarray, a: jnp.ndarray, nodes: jnp.ndarray, log_weights: jnp.ndarray,
) -> jnp.ndarray:
    """frailty.log_nested_drive_marginal (Horner replaced by the same sum in log space): log of the nested integral per fish."""
    n_fish, n_trials = S.shape
    total = jnp.maximum(drive_rates + burst_rates, 1e-300)
    pmf = poisson_binomial(drive_rates / total, stream, position, S.size, max_events)
    k = jnp.arange(max_events + 1)
    log_coef = jnp.where(pmf > 0, jnp.log(jnp.where(pmf > 0, pmf, 1.0)), -jnp.inf) + gammaln(a + k) - gammaln(a)
    d0, b0, S0, r0, a0 = (jax.lax.stop_gradient(x) for x in (drive_rates, burst_rates, S, r, a))

    def fixed_point(state: Tuple) -> Tuple:
        mode, _, _, i = state
        g_events = jnp.exp(mode)[stream // n_trials] * d0
        share = segment_sum(g_events / jnp.maximum(g_events + b0, 1e-300), stream, S.size).reshape(S.shape)
        new_mode, sd = _log_gain_mode(share, S0, mask, r0, a0, mode)
        return new_mode, jnp.max(jnp.abs(new_mode - mode), initial=0.0), sd, i + 1

    start = (jnp.zeros(n_fish), jnp.inf, jnp.ones(n_fish), 0)
    mode, _, sd, _ = jax.lax.while_loop(lambda st: (st[1] >= 1e-10) & (st[3] < 100), fixed_point, start)
    v = mode[:, None] + sd[:, None] * nodes
    v_streams = jnp.repeat(v, n_trials, axis=0)
    log_c = jnp.log(a + jnp.exp(v_streams) * S.ravel()[:, None])
    log_x = v_streams - log_c
    log_poly = jax.scipy.special.logsumexp(log_coef[:, None, :] + k * log_x[:, :, None], axis=2)
    log_z = (a * jnp.log(a) + segment_sum(jnp.log(total), stream, S.size))[:, None] - a * log_c + log_poly
    log_integrand = r * v - r * jnp.exp(v) + (mask.ravel()[:, None] * log_z).reshape(n_fish, n_trials, -1).sum(axis=1)
    log_integral = jax.scipy.special.logsumexp(log_weights + log_integrand + nodes**2 / 2, axis=1) + jnp.log(sd) + 0.5 * jnp.log(2 * jnp.pi)
    return log_integral + r * jnp.log(r) - gammaln(r)


def log_gamma_marginal(N: jnp.ndarray, S: jnp.ndarray, shape: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
    """GammaGain.log_marginal: log of the integral of p(g) g^N exp(-g S) dg."""
    return shape * jnp.log(rate) - gammaln(shape) + gammaln(N + shape) - (N + shape) * jnp.log(S + rate)


def log_zero_inflated_marginal(N: jnp.ndarray, S: jnp.ndarray, pi: jnp.ndarray, c: jnp.ndarray, no_rate: bool,
                               shape: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
    """ZeroInflatedGain.log_marginal; no_rate: non-responders have c = 0 (no events at all)."""
    nonresponder = jnp.where(N == 0, 0.0, -jnp.inf) if no_rate else N * jnp.log(c) - c * S
    return jnp.logaddexp(jnp.log(jnp.maximum(pi, 1e-300)) + nonresponder,
                         jnp.log(jnp.maximum(1.0 - pi, 1e-300)) + log_gamma_marginal(N, S, shape, rate))


def log_nested_gamma_marginal(n: jnp.ndarray, S: jnp.ndarray, mask: jnp.ndarray, r: jnp.ndarray, a: jnp.ndarray,
                              nodes: jnp.ndarray, log_weights: jnp.ndarray) -> jnp.ndarray:
    """frailty.log_nested_gamma_marginal: per fish, a gain g ~ Gamma(r, r) and a trial gain h ~ Gamma(a, a) on the whole intensity."""
    mode, sd = _log_gain_mode(*(jax.lax.stop_gradient(x) for x in (n, S, mask, r, a)), jnp.zeros(n.shape[0]))
    v = mode[:, None] + sd[:, None] * nodes
    w, N = (n + a) * mask, (n * mask).sum(axis=1)
    log_integrand = (r + N)[:, None] * v - r * jnp.exp(v) - jnp.einsum("ft,ftk->fk", w, jnp.log(a + S[:, :, None] * jnp.exp(v)[:, None, :]))
    constant = r * jnp.log(r) - gammaln(r) + (mask * (a * jnp.log(a) - gammaln(a) + gammaln(n + a))).sum(axis=1)
    log_integral = jax.scipy.special.logsumexp(log_weights + log_integrand + nodes**2 / 2, axis=1) + jnp.log(sd) + 0.5 * jnp.log(2 * jnp.pi)
    return log_integral + constant
