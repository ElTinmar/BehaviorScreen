"""
Core engine for Tier 1 (omnibus: does anything change) and Tier 2 (interaction:
is the change specific to this genotype, over and above any generic drug
effect). Both are the SAME model class with a different design matrix.

MultiGroupProcess wraps an existing, already-instantiated PointProcess
(e.g. PoissonProcess(RateKernelFactory.phototaxis_contra())) and fits it
jointly across several PointProcessDataset "groups", each carrying its own
0/1 (or continuous) covariate row. For kernel parameter p and covariate c:

    linear_p(group) = baseline_p + sum_c beta[c, p] * X[group, c]
    natural_p(group) = link_p.to_natural(linear_p(group))

_nll sums the wrapped process's OWN _nll(natural_params, dataset) over every
group -- no likelihood code is duplicated, and this works unmodified for
PoissonProcess, SurvivalProcess, HawkesProcess, RenewalProcess, and
GammaMixedEffectsProcess, since all of them expose the same
`_nll(self, params, dataset)` contract already.

Comparing two MultiGroupProcess instances that share the same `groups` but
differ by one covariate column is a valid nested-model comparison for
ModelComparator.likelihood_ratio_test (Δk = len(covariate) * n_kernel_params).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np
import joblib

from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.dataset import PointProcessDataset
from .param_links import BoundedLink


@dataclass
class GroupSpec:
    name: str
    dataset: PointProcessDataset
    covariates: Dict[str, float] = field(default_factory=dict)


class MultiGroupProcess(PointProcess):

    def __init__(
        self,
        base_process: PointProcess,
        groups: List[GroupSpec],
        covariate_names: List[str],
        integration_dt: float = 0.02,
    ):
        super().__init__(integration_dt)
        self.base_process = base_process
        self.groups = groups
        self.covariate_names = list(covariate_names)
        self.k = len(base_process.param_names)
        self.links = [BoundedLink.from_bounds(b) for b in base_process.bounds]

        self.name = f"MultiGroup[{base_process.name}] ~ 1 + {' + '.join(covariate_names) or '(intercept only)'}"
        self.latex_formula = base_process.latex_formula

        self.param_names = (
            [f"{p}[baseline]" for p in base_process.param_names]
            + [f"{p}[{c}]" for c in covariate_names for p in base_process.param_names]
        )
        baseline_linear = [
            self.links[i].to_linear(v) for i, v in enumerate(base_process.initial_guesses)
        ]
        self.initial_guesses = baseline_linear + [0.0] * (len(covariate_names) * self.k)
        self.bounds = [(None, None)] * len(self.param_names)

    # -- design -------------------------------------------------------------

    def _design_vector(self, group: GroupSpec) -> np.ndarray:
        return np.array([group.covariates.get(c, 0.0) for c in self.covariate_names])

    def _group_natural_params(self, params: np.ndarray, group: GroupSpec) -> List[float]:
        params = np.asarray(params, dtype=float)
        baseline_linear = params[: self.k]
        betas_linear = params[self.k :].reshape(len(self.covariate_names), self.k) \
            if self.covariate_names else np.zeros((0, self.k))
        x = self._design_vector(group)
        linear = baseline_linear + (x @ betas_linear if len(self.covariate_names) else 0.0)
        return [self.links[i].to_natural(linear[i]) for i in range(self.k)]

    def beta_index_range(self, covariate_name: str) -> slice:
        idx = self.covariate_names.index(covariate_name)
        start = self.k * (1 + idx)
        return slice(start, start + self.k)

    # -- likelihood -----------------------------------------------------------

    def _nll(self, params: List[float], dataset: Optional[PointProcessDataset] = None) -> float:
        total = 0.0
        for group in self.groups:
            natural = self._group_natural_params(params, group)
            total += self.base_process._nll(natural, group.dataset)
        return total

    def fit(self, method: str = "L-BFGS-B", options: Optional[dict] = None, **kwargs):
        # dataset is irrelevant here -- each group carries its own; pass None
        options = dict(options or {})
        options.setdefault("maxfun", 50000)
        options.setdefault("maxiter", 20000)
        return super().fit(dataset=None, method=method, options=options, **kwargs)

    def group_params(self, group_name: str) -> Dict[str, float]:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        group = next(g for g in self.groups if g.name == group_name)
        natural = self._group_natural_params(self.params_, group)
        return dict(zip(self.base_process.param_names, natural))

    # -- fish-level (cluster) bootstrap across ALL groups jointly -------------

    def _resample_copy(self, rng: np.random.Generator) -> "MultiGroupProcess":
        boot_groups = [
            GroupSpec(g.name, g.dataset.resample(rng), g.covariates) for g in self.groups
        ]
        return MultiGroupProcess(self.base_process, boot_groups, self.covariate_names, self.integration_dt)

    def bootstrap_params(self, n_boot: int = 200, seed: int = 0, n_jobs: int = -1) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        seeds = np.random.SeedSequence(seed).spawn(n_boot)

        def _one(s):
            rng = np.random.default_rng(s)
            model_copy = self._resample_copy(rng)
            try:
                model_copy.fit()
                return model_copy.params_
            except RuntimeError:
                return None

        results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_one)(s) for s in seeds)
        valid = [r for r in results if r is not None]
        if not valid:
            raise RuntimeError("All bootstrap fits failed.")
        print(f"[MultiGroupProcess.bootstrap_params] {len(valid)}/{n_boot} succeeded")
        return np.array(valid)