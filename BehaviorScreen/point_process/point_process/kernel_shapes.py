import numpy as np

from .backend import special, xp



def peak_normalized_pulse(x: np.ndarray, x_peak: float, k: float = 1.0) -> np.ndarray:
    """
    Generalized alpha-function / Gamma-shaped pulse, peak-normalized to height 1.

        f(x) = (x / x_peak)^k * exp(k * (1 - x / x_peak)),   x >= 0

    - f(0) = 0
    - f(x_peak) = 1   <- peak location & height fixed by construction
    - k=1: classic alpha function (linear rise, exponential decay)
    - k>1: sharper / more symmetric peak
    - k<1: fast rise, long tail (asymmetric)

    Non-negative for x >= 0, k > 0. No clamping required. 0 where x <= 0, without
    evaluating x^k there (its derivatives at 0 would be 0 * inf).
    """
    ratio = xp.maximum(x, 0.0) / x_peak
    positive = ratio > 0
    safe = xp.where(positive, ratio, 1.0)
    return xp.where(positive, xp.power(safe, k) * xp.exp(k * (1.0 - safe)), 0.0)


def bounded_trial_scale(trial: np.ndarray, alpha: float) -> np.ndarray:
    """Saturating logistic, always in (0, 2), equal to 1.0 at alpha=0 or trial=0."""
    return 2.0 / (1.0 + xp.exp(-alpha * trial))


def exgaussian_shape(t, mu, sigma, tau):
    """scipy.stats.exponnorm.pdf(t, tau / sigma, loc=mu, scale=sigma), as scipy computes it (log space)."""
    inv_k, z = sigma / tau, (t - mu) / sigma
    return xp.exp(inv_k * (0.5 * inv_k - z) + special.log_ndtr(z - inv_k) - xp.log(tau / sigma) - xp.log(sigma))


def sigmoid_bounded(z: np.ndarray, upper: float) -> np.ndarray:
    """
    Squash an unconstrained real z into (0, upper) via a logistic.

    Strictly bounded away from both 0 and `upper` for any FINITE z, and only
    approaches 0/upper asymptotically as z -> -inf/+inf. This means "the true
    MLE wants near-total suppression/enhancement" shows up as a LARGE z with a
    correspondingly wide bootstrap spread, rather than as a hard box-bound
    saturation (z_param pinned at a wall, CI collapsed to a point, or right-
    censored against the bound) -- see the A_ripple/f_dip boundary-pinning
    issue this replaces.
    """
    return upper / (1.0 + xp.exp(-z))


def logit_bounded(p: np.ndarray, upper: float) -> np.ndarray:
    """Inverse of sigmoid_bounded. Used only to convert an old-style directly-
    bounded point estimate/initial guess into the equivalent unconstrained z."""
    p = xp.clip(xp.asarray(p, dtype=float), 1e-6, upper - 1e-6)
    return xp.log(p / (upper - p))
