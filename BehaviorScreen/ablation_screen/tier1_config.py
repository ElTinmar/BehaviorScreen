"""
Tier 1 behaviors: the model each one uses, with every parameter pinned to
the control fit of the same model (matched by name) except the amplitudes.
"""
import copy
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from BehaviorScreen.point_process.analysis.model_config import (
    FLASH_ESCAPE, FORWARD_SWIMS, LATERAL_TURNS, LOOMING_ESCAPE, OKR_SWIMS, get_model_config,
)
from BehaviorScreen.point_process.analysis.tasks import load_fitted_params
from BehaviorScreen.point_process.point_process.point_process import PointProcess
from BehaviorScreen.point_process.point_process.wrappers import PartiallyFixedProcess

# Map column labels (stimulus · events): spontaneous activity first, then protocol order with responses to one stimulus side by side
cats = "+".join
MAP_COLUMNS = {
    "spont_dark": "dark · all bouts",
    "spont_bright": "bright · all bouts",
    "prey_capture": "prey · JT",
    "prey_capture_convergent_saccade": "prey · Convergent",
    "prey_capture_divergent_saccade": "prey · Divergent",
    "phototaxis": "dark | bright · RT",
    "dark_flash": f"bright -> dark · {cats(FLASH_ESCAPE)}",
    "omr_lateral": f"grating lateral · {cats(LATERAL_TURNS)}",
    "omr_forward": f"grating forward · {cats(FORWARD_SWIMS)}",
    "okr": f"pinwheel · {cats(OKR_SWIMS)}",
    "okr_conjugate_saccades": "pinwheel · Conjugate",
    "looming": f"looming · {cats(LOOMING_ESCAPE)}",
    "looming_divergent_saccade": "looming · Divergent",
    "after_looming": "looming break · all bouts",
}

# Fitted per arm: sustained rate, escape hazard bump and the directionality of sided models.
# Transient amplitudes (A, A_peak) are pinned with the shapes: too few evoked events per fish to be reliable.
AMPLITUDE_PARAMETERS = ("B", "H", "D")


class PinnedArchitecture:
    """Factory of fresh, unfitted copies of a fitted model with every non-amplitude parameter, and `pinned`, fixed."""

    def __init__(self, fitted: PointProcess, pinned: Sequence[str] = ()) -> None:
        self.template = copy.deepcopy(fitted)
        self.template.initial_guesses = [float(v) for v in fitted.params_]  # free amplitudes start at the control fit
        self.template.params_, self.template.param_dict_, self.template.fit_result = None, {}, None
        self.fixed = {n: float(v) for n, v in zip(fitted.param_names, fitted.params_) if n not in AMPLITUDE_PARAMETERS or n in pinned}

    def __call__(self) -> PartiallyFixedProcess:
        return PartiallyFixedProcess(copy.deepcopy(self.template), self.fixed)


class Fresh:
    """Factory of fresh, unfitted copies of a model."""

    def __init__(self, model: PointProcess) -> None:
        self.template = model

    def __call__(self) -> PointProcess:
        return copy.deepcopy(self.template)


def tier1_behaviors(
    models: Dict[str, Callable[[], PointProcess]], fits_root: Path, pinned: Optional[Dict[str, Sequence[str]]] = None
) -> Dict[str, dict]:
    """
    {experiment: {"events", "dataset", "architecture", "null", "model_name", "free"}}
    for each experiment and model builder in `models`; the pinned values come
    from the `fit` run of that experiment (fits_root / experiment / models).
    `pinned` lists, per experiment, amplitudes to pin as well.
    """
    config = get_model_config()
    behaviors = {}
    for name, build in models.items():
        model = build()
        model.set_params(load_fitted_params(fits_root / name / "models", model.name))
        architecture = PinnedArchitecture(model, (pinned or {}).get(name, ()))
        behaviors[name] = {
            "events": config[name]["events"],
            "dataset": config[name]["dataset"],
            "architecture": architecture,
            "null": Fresh(config[name]["null_model"]),
            "model_name": model.name,
            "free": [n for n in model.param_names if n not in architecture.fixed],
        }
    return behaviors
