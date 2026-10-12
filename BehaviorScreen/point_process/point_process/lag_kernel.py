"""Functions of the time since an event: history (Hawkes) and renewal kernels."""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import trapezoid


@dataclass(frozen=True)
class LagKernel:
    name: str
    func: Callable[[np.ndarray, List[float]], np.ndarray]
    param_names: List[str]
    initial_guesses: List[float]
    bounds: List[Tuple[Optional[float], Optional[float]]]
    integral_func: Optional[Callable[[np.ndarray, List[float]], np.ndarray]] = None
    latex_formula: str = ""

    def evaluate(self, lag: np.ndarray, params: List[float]) -> np.ndarray:
        if np.any(lag < 0):
            raise ValueError(f"{self.name}: lags must be non-negative.")
        return self.func(lag, params)

    def integrate(
        self,
        duration: Union[float, np.ndarray],
        params: List[float],
        integration_dt: float = 0.02,
    ) -> np.ndarray:
        """Integral from lag 0 to each duration (closed form if given, else trapezoid)."""
        duration = np.asarray(duration, dtype=float)
        if np.any(duration < 0):
            raise ValueError("Integration duration must be non-negative.")
        if self.integral_func is not None:
            return self.integral_func(duration, params)

        result = np.zeros(duration.size)
        for i, T in enumerate(np.atleast_1d(duration)):
            if T > 0:
                grid = np.arange(0.0, T + integration_dt, integration_dt)
                grid = np.append(grid[grid < T], T)
                result[i] = trapezoid(self.evaluate(grid, params), grid)
        return result[0] if duration.ndim == 0 else result
