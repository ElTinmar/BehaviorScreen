import numpy as np
from scipy.stats import exponnorm

def peak_normalized_pulse(x: np.ndarray, x_peak: float, k: float = 1.0) -> np.ndarray:
    """
    Generalized alpha-function / Gamma-shaped pulse, peak-normalized to height 1.

        f(x) = (x / x_peak)^k * exp(k * (1 - x / x_peak)),   x >= 0

    - f(0) = 0
    - f(x_peak) = 1   <- peak location & height fixed by construction
    - k=1: classic alpha function (linear rise, exponential decay)
    - k>1: sharper / more symmetric peak
    - k<1: fast rise, long tail (asymmetric)

    Non-negative for x >= 0, k > 0. No clamping required.
    """
    x_safe = np.maximum(x, 0.0)
    ratio = x_safe / x_peak
    return np.power(ratio, k) * np.exp(k * (1.0 - ratio))

def bounded_trial_scale(trial: np.ndarray, alpha: float) -> np.ndarray:
    """Saturating logistic, always in (0, 2), equal to 1.0 at alpha=0 or trial=0."""
    return 2.0 / (1.0 + np.exp(-alpha * trial))


def exgaussian_shape(t, mu, sigma, tau):
    K = tau / sigma  
    return exponnorm.pdf(t, K, loc=mu, scale=sigma)