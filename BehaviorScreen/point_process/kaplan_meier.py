"""Kaplan-Meier estimation and step-function evaluation, shared by datasets, GOF and localization."""
from typing import Tuple, Union

import numpy as np


def km_table(times: np.ndarray, censored: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Knots, post-jump survival, number at risk just before each knot and
    events at each knot, all starting with the artificial point (0, 1).
    Tied exact events count before tied censorings.
    """
    times = np.asarray(times, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if times.ndim != 1 or censored.shape != times.shape:
        raise ValueError("times and censored must be one-dimensional and of equal length.")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("Survival times must be finite and non-negative.")

    order = np.argsort(times, kind="stable")
    knots, first = np.unique(times[order], return_index=True)
    events = (
        np.add.reduceat((~censored[order]).astype(int), first) if len(times) else np.array([], dtype=int)
    )
    at_risk = len(times) - first
    has_event = events > 0
    survival = np.cumprod(1.0 - events[has_event] / at_risk[has_event])
    return (
        np.r_[0.0, knots[has_event]],
        np.r_[1.0, survival],
        np.r_[len(times), at_risk[has_event]].astype(int),
        np.r_[0, events[has_event]].astype(int),
    )


def kaplan_meier(
    times: np.ndarray, censored: np.ndarray, return_risk: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Knots and survival (and the number at risk with return_risk)."""
    knots, survival, at_risk, _ = km_table(times, censored)
    return (knots, survival, at_risk) if return_risk else (knots, survival)


def evaluate_step(
    x: np.ndarray, step_x: np.ndarray, step_y: np.ndarray, max_x: float = np.inf
) -> np.ndarray:
    """Right-continuous step function at x; NaN before the first step and beyond max_x."""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan, dtype=float)
    if len(step_x) == 0:
        return result
    idx = np.searchsorted(step_x, x, side="right") - 1
    valid = (idx >= 0) & (x <= max_x)
    result[valid] = np.asarray(step_y, dtype=float)[np.clip(idx[valid], 0, len(step_y) - 1)]
    return result
