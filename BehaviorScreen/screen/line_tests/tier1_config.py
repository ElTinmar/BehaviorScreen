"""
Tier 1 behaviors: each experiment's model of a variant (paths.VARIANTS), with every
parameter pinned to the control fit except the amplitudes. A contrast pins them to the control
fit refitted without its own arms (Arms.architecture), which keeps the permutation test exact.
"""
import copy
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from BehaviorScreen.screen.control_models.model_config import (
    FLASH_ESCAPE, FORWARD_SWIMS, LATERAL_TURNS, LOOMING_ESCAPE, OKR_SWIMS, get_model_config,
)
from BehaviorScreen.screen.control_models.tasks import load_best_model, load_fit_records
from BehaviorScreen.point_process.hawkes_process import HawkesProcess
from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.renewal_process import RenewalProcess
from BehaviorScreen.point_process.sided_process import SidedRenewalProcess
from BehaviorScreen.point_process.wrappers import PartiallyFixedProcess

# Amplitudes also pinned to the control fit: these baselines are near zero (about
# one event a minute), so their per-arm log ratios are noise.
PINNED_AMPLITUDES = {name: ["B"] for name in ("looming", "looming_divergent_saccade", "dark_flash")}

# Map column labels (stimulus · events): spontaneous activity first, then protocol order with responses to one stimulus side by side
cats = "+".join
MAP_COLUMNS = {
    "spont_dark": "dark · all bouts",
    "spont_bright": "bright · all bouts",
    "prey_capture_ipsi": "prey · ipsi JT",
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

# Fitted per arm: sustained rate, transient amplitude, escape hazard bump and the directionality
# of sided models; PINNED_AMPLITUDES pins some of them per experiment.
AMPLITUDE_PARAMETERS = ("B", "A", "A_peak", "H", "D")


class PinnedArchitecture:
    """Factory of fresh, unfitted copies of a fitted model with every non-amplitude parameter, and `pinned`, fixed."""

    def __init__(self, fitted: PointProcess, pinned: Sequence[str] = ()) -> None:
        self.fitted, self.pinned = copy.deepcopy(fitted), tuple(pinned)
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


def has_history(model: PointProcess) -> bool:
    """Whether the model, under any wrappers, depends on the fish's earlier events (renewal or Hawkes)."""
    while not isinstance(model, (RenewalProcess, SidedRenewalProcess, HawkesProcess)):
        if not hasattr(model, "base_process"):
            return False
        model = model.base_process
    return True


def variant_model(config: dict, model_dir: Path, variant: str) -> PointProcess:
    """
    The model of `variant` of one experiment's fit run, with its fitted parameters: the best one, the best one
    without event history (by AIC among the fitted rungs) or the ladder's first rung (constant rate or hazard).
    """
    if variant == "best":
        return load_best_model(config, model_dir)
    records = load_fit_records(model_dir)
    candidates = [config["models"][0]] if variant == "base" else [m for m in config["models"] if not has_history(m)]
    aic = {m.name: 2 * len(m.param_names) + 2 * records[m.name]["nll"] for m in candidates if m.name in records}
    if not aic:
        raise ValueError(f"{model_dir}: no fitted {variant} model among {[m.name for m in candidates]}")
    model = next(m for m in candidates if m.name == min(aic, key=aic.get))
    model.set_params(np.asarray(records[model.name]["params"], dtype=float))
    return model


def tier1_behaviors(fits_root: Path, pinned: Optional[Dict[str, Sequence[str]]] = None, variant: str = "best") -> Dict[str, dict]:
    """
    {experiment: {"events", "dataset", "architecture", "null", "model_name", "free"}}
    for each experiment of MAP_COLUMNS: its model of `variant`, pinned to the `fit` run of that
    experiment (fits_root / experiment / models). `pinned` lists, per experiment, amplitudes to pin as well,
    where the model has them free otherwise and another amplitude remains free.
    """
    config = get_model_config()
    behaviors = {}
    for name in MAP_COLUMNS:
        model = variant_model(config[name], fits_root / name / "models", variant)
        amplitudes = [p for p in model.param_names if p in AMPLITUDE_PARAMETERS]
        pins = [p for p in (pinned or {}).get(name, ()) if p in amplitudes]
        architecture = PinnedArchitecture(model, pins if len(pins) < len(amplitudes) else ())
        behaviors[name] = {
            "events": config[name]["events"],
            "dataset": config[name]["dataset"],
            "architecture": architecture,
            "null": Fresh(config[name]["null_model"]),
            "model_name": model.name,
            "free": [n for n in model.param_names if n not in architecture.fixed],
        }
    return behaviors
