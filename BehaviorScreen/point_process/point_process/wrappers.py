"""Processes defined on top of another process's parameters."""
from typing import Dict, List

import numpy as np

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

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        return self.base_process.compute_expected_rate(dataset)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        return self.base_process.cumulative_integrated_intensity(t_events, trial)

    def stream_compensator_profile(self, t_ev: np.ndarray, trial: float, duration_s: float):
        return self.base_process.stream_compensator_profile(t_ev, trial, duration_s)

    def mixed_effects_likelihood_terms(self, dataset: PointProcessDataset, params: List[float]):
        return self.base_process.mixed_effects_likelihood_terms(dataset, self._base_params(params))

    def simulate_stream(self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator) -> np.ndarray:
        return self.base_process.simulate_stream(dataset, t_idx, gain, rng)

    def _intensity_upper_bound(self, dataset: PointProcessDataset, t_idx: int) -> float:
        return self.base_process._intensity_upper_bound(dataset, t_idx)

    def _draw_fish_gains(self, num_fish: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        return self.base_process._draw_fish_gains(num_fish, n_sims, rng)

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
    def is_survival(self) -> bool:
        return self.base_process.is_survival


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
