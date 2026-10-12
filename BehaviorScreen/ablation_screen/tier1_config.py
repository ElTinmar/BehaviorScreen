"""
Tier 1 behaviors from the control fits: each experiment's chosen model
(model_config tier1_model, else the best by AIC) with its fitted
parameters pinned, except the amplitude parameters, which are free.
"""
import copy
from pathlib import Path
from typing import Dict, Sequence

from BehaviorScreen.point_process.analysis.model_config import AMPLITUDE_PARAMETERS, get_model_config
from BehaviorScreen.point_process.analysis.tasks import load_best_model, load_fitted_model
from BehaviorScreen.point_process.point_process.point_process import PointProcess
from BehaviorScreen.point_process.point_process.wrappers import PartiallyFixedProcess


class PinnedArchitecture:
    """Factory of fresh, unfitted copies of a model with every non-amplitude parameter pinned."""

    def __init__(self, fitted: PointProcess) -> None:
        self.template = copy.deepcopy(fitted)
        self.template.initial_guesses = [float(v) for v in fitted.params_]  # free amplitudes start at the control fit
        self.template.params_, self.template.param_dict_, self.template.fit_result = None, {}, None
        self.fixed = {n: float(v) for n, v in zip(fitted.param_names, fitted.params_) if n not in AMPLITUDE_PARAMETERS}

    def __call__(self) -> PartiallyFixedProcess:
        return PartiallyFixedProcess(copy.deepcopy(self.template), self.fixed)


class Fresh:
    """Factory of fresh, unfitted copies of a model."""

    def __init__(self, model: PointProcess) -> None:
        self.template = model

    def __call__(self) -> PointProcess:
        return copy.deepcopy(self.template)


def tier1_behaviors(experiments: Sequence[str], fits_root: Path) -> Dict[str, dict]:
    """
    {experiment: {"dataset", "architecture", "null", "model_name", "free"}},
    from the `fit` outputs under fits_root (fits_root / experiment / models).
    """
    config = get_model_config()
    behaviors = {}
    for name in experiments:
        experiment = config[name]
        model_dir = fits_root / name / "models"
        index = experiment["tier1_model"]
        fitted = load_best_model(experiment, model_dir) if index is None else load_fitted_model(experiment, model_dir, index)
        architecture = PinnedArchitecture(fitted)
        behaviors[name] = {
            "dataset": experiment["dataset"],
            "architecture": architecture,
            "null": Fresh(experiment["null_model"]),
            "model_name": fitted.name,
            "free": [n for n in fitted.param_names if n not in architecture.fixed],
        }
    return behaviors
