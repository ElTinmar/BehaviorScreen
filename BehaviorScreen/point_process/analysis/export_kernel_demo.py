"""
Writes the model ladders of model_config into kernel_demo.html (between its
DATA markers), in Tier 1 map order: every candidate's formula, parameters
(fitted values where a fit run is given, else the initial guesses), bounds and
structure, the Tier 1 model and its per-arm parameters (tier1_config), plus
kernel values computed here, which the page checks its own kernels against.

    python -m BehaviorScreen.point_process.analysis.export_kernel_demo [-o <fit output dir>]
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from BehaviorScreen.ablation_screen.tier1_config import AMPLITUDE_PARAMETERS, MAP_COLUMNS, PINNED_AMPLITUDES, TIER1_MODELS
from BehaviorScreen.point_process.analysis.model_config import get_model_config
from BehaviorScreen.point_process.point_process.frailty import Frailty, SideFrailty, ZeroInflatedFamily
from BehaviorScreen.point_process.point_process.lag_kernel import LagKernel
from BehaviorScreen.point_process.point_process.point_process import PointProcess
from BehaviorScreen.point_process.point_process.poisson_process import PoissonProcess, RateKernel
from BehaviorScreen.point_process.point_process.renewal_process import RenewalProcess
from BehaviorScreen.point_process.point_process.sided_process import SidedPoissonProcess, SidedRenewalProcess, SidedSurvivalProcess
from BehaviorScreen.point_process.point_process.survival_process import SurvivalProcess

PAGE = Path(__file__).resolve().parents[1] / "point_process" / "kernel_demo.html"
DATA = re.compile(r"(/\*BEGIN DATA\*/).*?(/\*END DATA\*/)", re.S)


def constants(kernel: LagKernel) -> Dict[str, Any]:
    """The kernel function's fixed settings (closure numbers and name mappings), e.g. a dip's upper bound."""
    cells = zip(kernel.func.__code__.co_freevars, kernel.func.__closure__ or ())
    return {n: c.cell_contents for n, c in cells if isinstance(c.cell_contents, (int, float, dict)) and not isinstance(c.cell_contents, bool)}


def component(kernel: LagKernel, args: Dict[str, Any], values: Dict[str, float], lag: bool, t_end: float) -> dict:
    """A kernel, which model parameter feeds each of its parameters ([a, b]: a / b), and reference values."""
    value = lambda spec: values[spec[0]] / values[spec[1]] if isinstance(spec, list) else values[spec]
    params = [value(args[k]) for k in kernel.param_names]
    t = np.linspace(0.0, t_end if not lag else 5.0, 7)
    if lag:
        reference = kernel.evaluate(t, params)
        m = np.zeros_like(t)
    else:
        t, m = np.repeat(t, 2), np.tile([0.0, 3.0], len(t))
        reference = kernel.evaluate(t, m, params)
    return {"name": kernel.name, "args": args, "constants": constants(kernel),
            "check": {"t": t.tolist(), "m": m.tolist(), "values": np.asarray(reference, float).tolist()}}


def structure(model: PointProcess, values: Dict[str, float], t_end: float) -> dict:
    """What the page draws for a model, its kernels fed by the model's parameters (by name)."""
    same = lambda k: {p: p for p in k.param_names}
    rate = lambda k, args=None: component(k, args or same(k), values, False, t_end)
    if isinstance(model, Frailty):
        gains = model.gains
        gain = {"type": "zero_inflated", "fit_c": gains.fit_c, "c": gains.c_init, "c_upper": gains.c_upper} \
            if isinstance(gains, ZeroInflatedFamily) else {"type": "gamma"}
        return {"kind": "gain", "gain": gain, "trial": model.trial_gain, "side": isinstance(model, SideFrailty),
                "base": structure(model.base_process, values, t_end)}
    if isinstance(model, SidedRenewalProcess):
        sides = structure(model.sides, values, t_end)
        return {**sides, "kind": "sided_renewal", **history(model.ground, values)}
    if isinstance(model, SidedPoissonProcess):
        contra = {p: (["B", "D"] if p == "B" else f"contra_{p}") for p in model.contra.param_names}
        return {"kind": "sided", "ipsi": rate(model.ipsi), "contra": rate(model.contra, contra)}
    if isinstance(model, RenewalProcess):
        return {"kind": "renewal", "stim": rate(model.kernel), **history(model, values)}
    if isinstance(model, SidedSurvivalProcess):
        return {"kind": "sided_survival", "hazard": rate(model.kernel), "preferred": int(model.preferred)}
    if isinstance(model, SurvivalProcess):
        return {"kind": "survival", "hazard": rate(model.kernel)}
    if isinstance(model, PoissonProcess):
        return {"kind": "poisson", "stim": rate(model.kernel)}
    raise TypeError(f"export_kernel_demo: no page structure for {type(model).__name__}")


def history(renewal: RenewalProcess, values: Dict[str, float]) -> dict:
    same = lambda k: {p: p for p in k.param_names}
    out = {"recovery": component(renewal.renewal_kernel, same(renewal.renewal_kernel), values, True, 0.0)}
    if renewal.excitation is not None:
        out["burst"] = {**component(renewal.excitation, same(renewal.excitation), values, True, 0.0),
                        "relative": renewal.excitation.relative}
    return out


def slider_range(value: float, lo: Optional[float], hi: Optional[float]) -> List[float]:
    """Bounds where they are narrow; around the value where they are wide or absent."""
    if lo is not None and hi is not None and lo > 0 and hi / lo > 100:
        return [max(lo, value / 10), min(hi, value * 10)]
    width = max(3 * abs(value), 3.0)
    return [value - width if lo is None else lo, value + width if hi is None else hi]


def model_entry(model: PointProcess, fitted: Optional[List[float]], t_end: float) -> dict:
    values = dict(zip(model.param_names, fitted if fitted is not None else model.initial_guesses))
    if len(values) != len(model.param_names):
        raise ValueError(f"{model.name}: parameter names are not unique")
    return {
        "name": model.name, "latex": model.latex_formula.strip("$"), "fitted": fitted is not None,
        "params": [{"name": n, "value": values[n], "range": slider_range(values[n], lo, hi)} for n, (lo, hi) in zip(model.param_names, model.bounds)],
        "structure": structure(model, values, t_end),
    }


def export(output_dir: Optional[Path]) -> List[dict]:
    configs = get_model_config()
    experiments = []
    for name in [n for n in MAP_COLUMNS if n in configs] + [n for n in configs if n not in MAP_COLUMNS]:
        config = configs[name]
        fits_path = output_dir / name / "models" / "fitted_models.json" if output_dir else None
        fits = json.loads(fits_path.read_text()) if fits_path and fits_path.exists() else {}
        t_end = float(config["dataset"]["t_end"])
        models = [config["null_model"]] + [m for m in config["models"] if m.name != config["null_model"].name]
        tier1 = TIER1_MODELS[name]() if name in TIER1_MODELS else None
        if tier1 is not None and tier1.name not in {m.name for m in models}:
            models.append(tier1)
        free = [] if tier1 is None else [p for p in tier1.param_names if p in AMPLITUDE_PARAMETERS and p not in PINNED_AMPLITUDES.get(name, ())]
        experiments.append({
            "name": name, "label": MAP_COLUMNS.get(name, name), "t_end": t_end, "tier1": tier1 and tier1.name, "free": free,
            "models": [model_entry(m, fits.get(m.name), t_end) for m in models],
        })
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output-dir", type=Path, help="Output folder of fit runs (<exp>/models/fitted_models.json).")
    args = parser.parse_args()
    data = json.dumps(export(args.output_dir), separators=(",", ":"))
    page = PAGE.read_text(encoding="utf-8")
    if not DATA.search(page):
        raise ValueError(f"{PAGE} has no /*BEGIN DATA*/ ... /*END DATA*/ block: restore the page from the repository")
    PAGE.write_text(DATA.sub(lambda m: f"{m.group(1)}\nconst EXPERIMENTS = {data};\n{m.group(2)}", page), encoding="utf-8")
    print(f"wrote {PAGE}")


if __name__ == "__main__":
    main()
