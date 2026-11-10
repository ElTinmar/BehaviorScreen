"""Processes defined on top of another process's parameters."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backend import Arrays, jnp
from .dataset import PointProcessDataset
from .point_process import PointProcess


class Wrapper(PointProcess):
    """
    Owns base_process and keeps its parameters in sync (set_params), so
    everything not overridden is the base process's own method.
    """

    def __init__(self, base_process: PointProcess) -> None:
        super().__init__(base_process.integration_dt)
        self.base_process = base_process

    def _base_params(self, params: List[float]) -> List[float]:
        raise NotImplementedError

    def set_params(self, params: np.ndarray) -> None:
        super().set_params(params)
        self.base_process.set_params(self._base_params(list(self.params_)))

    def predict(self, *args, **kwargs) -> np.ndarray:
        return self.base_process.predict(*args, **kwargs)

    def compute_expected_rate(self, dataset: PointProcessDataset, **side) -> np.ndarray:
        """With bout history, the simulated rate (gains integrated out); else the base process's."""
        if self.uses_history:
            return self.simulated_rate(dataset, **side)
        return self.base_process.compute_expected_rate(dataset, **side)

    def conditional_rate(self, dataset: PointProcessDataset, side: Optional[int] = None) -> Optional[np.ndarray]:
        return self.base_process.conditional_rate(dataset, side)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        return self.base_process.cumulative_integrated_intensity(t_events, trial)

    def stream_compensator_profile(self, t_ev: np.ndarray, trial: float, duration_s: float, history: Optional[np.ndarray] = None):
        return self.base_process.stream_compensator_profile(t_ev, trial, duration_s, history=history)

    def mixed_effects_likelihood_terms(self, dataset: PointProcessDataset, params: List[float]):
        return self.base_process.mixed_effects_likelihood_terms(dataset, self._base_params(params))

    def stream_likelihood_terms(self, dataset: PointProcessDataset, params: List[float]):
        return self.base_process.stream_likelihood_terms(dataset, self._base_params(params))

    def simulate_stream(self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator, **history) -> np.ndarray:
        return self.base_process.simulate_stream(dataset, t_idx, gain, rng, **history)

    def _intensity_upper_bound(self, dataset: PointProcessDataset, t_idx: int) -> float:
        return self.base_process._intensity_upper_bound(dataset, t_idx)

    def _draw_fish_gains(self, num_fish: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        return self.base_process._draw_fish_gains(num_fish, n_sims, rng)

    def _fish_draws(self, num_fish: int, rng: np.random.Generator, dataset: Optional[PointProcessDataset] = None) -> list:
        return self.base_process._fish_draws(num_fish, rng, dataset)

    def _stream_tau_values(self, dataset: PointProcessDataset):
        return self.base_process._stream_tau_values(dataset)

    def estimate_fish_gains(self, dataset: PointProcessDataset):
        return self.base_process.estimate_fish_gains(dataset)

    def population_survival_curve(self, dataset: PointProcessDataset, trial=None):
        return self.base_process.population_survival_curve(dataset, trial=trial)

    @property
    def dispersion_r(self) -> float:
        return self.base_process.dispersion_r

    @property
    def stream_gain_variance(self) -> float:
        return self.base_process.stream_gain_variance

    @property
    def is_survival(self) -> bool:
        return self.base_process.is_survival

    @property
    def uses_history(self) -> bool:
        return self.base_process.uses_history

    @property
    def gain_scales_drive_only(self) -> bool:
        return self.base_process.gain_scales_drive_only

    def drive_likelihood_terms(self, dataset: PointProcessDataset, params: List[float]):
        return self.base_process.drive_likelihood_terms(dataset, self._base_params(params))

    def interval_drive_terms(self, dataset: PointProcessDataset) -> Tuple[np.ndarray, ...]:
        return self.base_process.interval_drive_terms(dataset)


class PartiallyFixedProcess(Wrapper):
    """
    base_process with some parameters pinned to known values; the others
    are free. Uncertainty in the pinned values is not propagated.
    """

    def __init__(self, base_process: PointProcess, fixed_values: Dict[str, float]) -> None:
        super().__init__(base_process)
        unknown = set(fixed_values) - set(base_process.param_names)
        if unknown:
            raise ValueError(f"fixed_values contains unknown parameter names: {unknown}")
        self.fixed_values = dict(fixed_values)
        self._free_idx = [i for i, p in enumerate(base_process.param_names) if p not in fixed_values]
        if not self._free_idx:
            raise ValueError("PartiallyFixedProcess: at least one parameter must remain free.")

        self.name = f"PartiallyFixed[{base_process.name}] (fixed: {list(fixed_values)})"
        self.latex_formula = base_process.latex_formula
        self.param_names = [base_process.param_names[i] for i in self._free_idx]
        self.initial_guesses = [base_process.initial_guesses[i] for i in self._free_idx]
        self.bounds = [base_process.bounds[i] for i in self._free_idx]

    def _base_params(self, params: List[float]) -> List[float]:
        full = [self.fixed_values.get(name, 0.0) for name in self.base_process.param_names]
        for i, value in zip(self._free_idx, params):
            full[i] = value
        return full

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        return self.base_process._nll(self._base_params(params), dataset)

    def supports_jax(self) -> bool:
        return self.base_process.supports_jax()

    def _jax_key(self) -> tuple:
        return (type(self).__name__, tuple(self._free_idx)) + self.base_process._jax_key()

    def _arrays(self, dataset: PointProcessDataset) -> Arrays:
        """The base process's arrays and the pinned values (an argument: one compiled likelihood for any values)."""
        return Arrays({"base": self.base_process._arrays(dataset), "full": jnp.asarray(self._base_params([0.0] * len(self._free_idx)), dtype=float)}, {})

    def jax_nll(self, params: jnp.ndarray, a: Arrays) -> jnp.ndarray:
        return self.base_process.jax_nll(a["full"].at[jnp.asarray(self._free_idx)].set(params), a["base"])
