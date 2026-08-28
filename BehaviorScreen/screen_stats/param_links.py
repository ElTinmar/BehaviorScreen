"""
Bounded reparametrization so MultiGroupProcess can add/subtract dummy-variable
effects on a linear (unconstrained) scale while guaranteeing every group's
natural kernel parameter stays inside that kernel's own declared bounds --
no clipping, no boundary optimizer failures.

    natural = link.to_natural(linear)
    linear  = link.to_linear(natural)

- both bounds finite  -> scaled logistic:  natural in (lo, hi)
- lower bound only    -> softplus:         natural in (lo, inf)
- upper bound only    -> reflected softplus: natural in (-inf, hi)
- unbounded           -> identity

This is the same family of trick already used ad hoc in this codebase
(bounded_trial_scale is a special case of the two-sided logistic below,
alpha_B/alpha_peak are already fit in an unconstrained log-linear space).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _inv_softplus(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y = np.maximum(y, eps)
    # numerically stable inverse of log(1+exp(x))
    return np.where(y > 30, y, np.log(np.expm1(y)))


@dataclass(frozen=True)
class BoundedLink:
    lo: Optional[float]
    hi: Optional[float]
    eps: float = 1e-6

    def to_natural(self, x: float) -> float:
        x = float(x)
        if self.lo is not None and self.hi is not None:
            return self.lo + (self.hi - self.lo) / (1.0 + np.exp(-x))
        elif self.lo is not None:
            return self.lo + float(_softplus(np.array(x)))
        elif self.hi is not None:
            return self.hi - float(_softplus(np.array(-x)))
        else:
            return x

    def to_linear(self, y: float) -> float:
        y = float(y)
        if self.lo is not None and self.hi is not None:
            y = min(max(y, self.lo + self.eps), self.hi - self.eps)
            p = (y - self.lo) / (self.hi - self.lo)
            return float(np.log(p / (1.0 - p)))
        elif self.lo is not None:
            z = max(y - self.lo, self.eps)
            return float(_inv_softplus(np.array(z)))
        elif self.hi is not None:
            z = max(self.hi - y, self.eps)
            return -float(_inv_softplus(np.array(z)))
        else:
            return y

    @classmethod
    def from_bounds(cls, bounds) -> "BoundedLink":
        lo, hi = bounds
        return cls(lo=lo, hi=hi)