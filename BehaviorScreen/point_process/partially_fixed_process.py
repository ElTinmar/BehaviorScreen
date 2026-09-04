from typing import Dict, List
import numpy as np

from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.dataset import PointProcessDataset


class PartiallyFixedProcess(PointProcess):
    """
    Wraps base_process, pinning a subset of its parameters to fixed values
    and exposing only the REMAINING parameters as free.

    CAVEAT: fixed_values are treated as known constants. Uncertainty in the
    pooled shape estimate is NOT propagated into downstream Hessian/
    bootstrap/permutation inference on the free parameters
    """

    def __init__(self, base_process: PointProcess, fixed_values: Dict[str, float]):
        super().__init__(base_process.integration_dt)
        self.base_process = base_process
        self.fixed_values = dict(fixed_values)

        unknown = set(fixed_values) - set(base_process.param_names)
        if unknown:
            raise ValueError(f"fixed_values contains unknown parameter names: {unknown}")

        free_mask = [p not in fixed_values for p in base_process.param_names]
        self._free_idx = [i for i, f in enumerate(free_mask) if f]
        self._fixed_idx = [i for i, f in enumerate(free_mask) if not f]
        if not self._free_idx:
            raise ValueError("PartiallyFixedProcess: at least one parameter must remain free.")

        self.name = f"PartiallyFixed[{base_process.name}] (fixed: {list(fixed_values)})"
        self.latex_formula = base_process.latex_formula
        self.param_names = [base_process.param_names[i] for i in self._free_idx]
        self.initial_guesses = [base_process.initial_guesses[i] for i in self._free_idx]
        self.bounds = [base_process.bounds[i] for i in self._free_idx]

    def _expand(self, free_params: List[float]) -> List[float]:
        full = [0.0] * len(self.base_process.param_names)
        for i, v in zip(self._free_idx, free_params):
            full[i] = v
        for i in self._fixed_idx:
            full[i] = self.fixed_values[self.base_process.param_names[i]]
        return full

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        return self.base_process._nll(self._expand(params), dataset)

    def set_params(self, params: np.ndarray) -> None:
        self.params_ = np.asarray(params, dtype=float)
        self.param_dict_ = dict(zip(self.param_names, self.params_))
        self.base_process.set_params(self._expand(list(self.params_)))

    def fit(self, dataset: PointProcessDataset, method: str = "L-BFGS-B", **kwargs):
        super().fit(dataset, method=method, **kwargs)
        self.base_process.set_params(self._expand(list(self.params_)))
        return self

    # Everything else delegates to base_process, which has already been
    # synced (via set_params/fit above) to the full expanded parameter
    # vector -- identical pattern to GammaMixedEffectsProcess.
    def predict(self, *args, **kwargs):
        return self.base_process.predict(*args, **kwargs)

    def compute_expected_rate(self, dataset):
        return self.base_process.compute_expected_rate(dataset)

    def cumulative_integrated_intensity(self, t_events, trial):
        return self.base_process.cumulative_integrated_intensity(t_events, trial)

    def mixed_effects_likelihood_terms(self, dataset, params):
        return self.base_process.mixed_effects_likelihood_terms(dataset, self._expand(params))

    @property
    def dispersion_r(self) -> float:
        return self.base_process.dispersion_r

    def population_survival_curve(self, dataset, trial=None):
        return self.base_process.population_survival_curve(dataset, trial=trial)

    @property
    def is_survival(self) -> bool:
        return self.base_process.is_survival

    def estimate_fish_gains(self, dataset):
        return self.base_process.estimate_fish_gains(dataset)

    def simulate_stream(self, dataset, t_idx, gain, rng) -> np.ndarray:
        return self.base_process.simulate_stream(dataset, t_idx, gain, rng)