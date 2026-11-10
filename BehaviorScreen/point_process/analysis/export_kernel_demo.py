"""
Writes the model ladders of model_config (the experiments of the Tier 1 map) into kernel_demo_data.js, which
kernel_demo.html loads (a generated file, not in the repository), in Tier 1 map order: every candidate's formula, parameters
(fitted values where a fit run is given, else the initial guesses), bounds and
structure, the Tier 1 model (the best rung of the fit run) and its per-arm parameters, plus
kernel values computed here, which the page checks its own kernels against.
With a data root, also the events the ladders were fitted to, as histograms the
page draws beside its simulations.

    python -m BehaviorScreen.point_process.analysis.export_kernel_demo [-o <fit output dir>] [-d <data root>]
"""
import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from megabouts.utils import bouts_category_name, bouts_category_name_short

from BehaviorScreen.core import Laterality, Stim
from BehaviorScreen.ablation_screen.tier1_config import AMPLITUDE_PARAMETERS, MAP_COLUMNS, PINNED_AMPLITUDES
from BehaviorScreen.point_process.analysis.model_config import get_model_config
from BehaviorScreen.point_process.analysis.run_analysis import add_data_arguments, make_loaders
from BehaviorScreen.point_process.analysis.tasks import load_fit_records
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader, PointProcessDataset
from BehaviorScreen.point_process.point_process.frailty import Frailty, SideFrailty, ZeroInflatedFamily
from BehaviorScreen.point_process.point_process.lag_kernel import LagKernel
from BehaviorScreen.point_process.point_process.point_process import PointProcess
from BehaviorScreen.point_process.point_process.poisson_process import PoissonProcess, RateKernel
from BehaviorScreen.point_process.point_process.renewal_process import RenewalProcess
from BehaviorScreen.point_process.point_process.sided_process import CONTRA, IPSI, SidedPoissonProcess, SidedRenewalProcess, SidedSurvivalProcess
from BehaviorScreen.point_process.point_process.survival_process import SurvivalProcess

DATA = Path(__file__).resolve().parents[1] / "point_process" / "kernel_demo_data.js"
DATA_BIN = 0.05  # s; the page's time bins are multiples of it
PAGE_FISH, PAGE_DT = 200, 0.01  # the page's simulated fish and time step

BOUT_NAMES = dict(zip(bouts_category_name_short, (n.replace("_", " ") for n in bouts_category_name)))
PER_TRIAL = "log / trial"
# Parameter: (unit, meaning); contra_<name> is <name> for the contra side.
PARAMETERS = {
    "A": ("Hz", "onset transient: extra rate at stimulus onset, decaying with τ"),
    "tau": ("s", "decay time of the onset transient"),
    "B": ("Hz", "sustained rate (sided models: the ipsi side; contra is B/D)"),
    "A_peak": ("events", "size of the response peak: extra events per trial"),
    "mu": ("s", "time of the response peak (centre of its Gaussian part)"),
    "sigma": ("s", "width of the response peak"),
    "tau_decay": ("s", "tail of the response peak: how long it lingers"),
    "f_dip": ("fraction", "onset dip: fraction of B suppressed at onset"),
    "z0_dip": ("logit", "onset dip depth, 0.95·σ(z): fraction of B suppressed at onset"),
    "z_dip": ("logit", "onset dip depth, 0.995·σ(z): fraction of B suppressed at onset"),
    "tau_dip": ("s", "recovery time from the onset dip"),
    "z0_ripple": ("logit", "depth of the prey-locked ripple, 0.95·σ(z): 0 flat, 0.95 full modulation"),
    "phi1": ("rad", "phase of the ripple at the prey frequency (0.643 Hz)"),
    "phi2": ("rad", "phase of the ripple's second harmonic"),
    "alpha_peak": (PER_TRIAL, "change per trial of the transient (× e^α per trial; < 0: habituation)"),
    "alpha_baseline": (PER_TRIAL, "change per trial of the sustained rate (× e^α per trial)"),
    "alpha_shared": (PER_TRIAL, "change per trial of transient and sustained rate together (× e^α per trial)"),
    "alpha_B": (PER_TRIAL, "change per trial of the sustained rate (× e^α per trial)"),
    "alpha_ripple": (PER_TRIAL, "change per trial of the ripple depth logit"),
    "alpha_dip": (PER_TRIAL, "change per trial of the dip depth logit"),
    "alpha": (PER_TRIAL, "change per trial of the escape bump (× e^α per trial; < 0: habituation)"),
    "D": ("ratio", "direction: ipsi / contra rate ratio; 1 no preference, P_ipsi = D/(1+D) at baseline"),
    "tau_refractory": ("s", "recovery after any bout: rate × (1 − e^(−(Δt/τ_r)^k))"),
    "k_refractory": ("shape", "sharpness of the recovery: 1 exponential, larger: more regular intervals between bouts"),
    "A_burst": ("× drive", "burst after a selected bout: extra rate at its peak, as a multiple of the stimulus-driven rate"),
    "burst_peak_lag": ("s", "time from a selected bout to the burst's peak"),
    "r_dispersion": ("shape", "fish gains Γ(r, r): CV 1/√r across fish; larger: fish more alike"),
    "r_trial": ("shape", "trial gains Γ(a, a): CV 1/√a across a fish's trials; larger: trials more alike"),
    "s_side": ("log-odds", "SD across fish of the side preference u_f"),
    "z_pi_nonresponder": ("logit", "fraction of non-responding fish, σ(z)"),
    "z_c_nonresponder": ("logit", "non-responders' residual gain, 0.3·σ(z)"),
}
SURVIVAL = {
    "B": ("Hz", "baseline escape hazard outside the bump"),
    "H": ("Hz", "escape bump: peak hazard"),
    "mu": ("s", "time of the escape bump's peak"),
    "sigma": ("s", "width of the escape bump"),
    "D": ("ratio", "direction: escapes to the preferred / other side; P_preferred = D/(1+D)"),
}
EXGAUSS_SURVIVAL = {
    "H": ("–", "escape bump: hazard under the whole bump, −log P(no escape from the bump)"),
    "mu": ("s", "centre of the escape bump's Gaussian part (its peak comes slightly later)"),
    "tau": ("s", "tail of the escape bump"),
}


def describe(name: str, kernel_kind: str, exgauss: bool) -> Tuple[str, str]:
    """A parameter's unit and meaning."""
    if name.startswith("contra_"):
        unit, info = describe(name[len("contra_"):], kernel_kind, exgauss)
        return unit, f"{info} (contra side)"
    survival = kernel_kind.endswith("survival")
    for table, applies in ((EXGAUSS_SURVIVAL, survival and exgauss), (SURVIVAL, survival), (PARAMETERS, True)):
        if applies and name in table:
            return table[name]
    raise KeyError(f"export_kernel_demo: no description for parameter {name}")


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


def selection(config: dict) -> Tuple[str, str]:
    """The events an experiment's models describe: in full, and the categories alone."""
    d, kind = config["dataset"], config["events"]
    categories = [d["category"]] if isinstance(d.get("category"), str) else d.get("category") or []
    short = " or ".join(categories) or kind[:-1]
    what = (", ".join(f"{BOUT_NAMES[c]} ({c})" if c in BOUT_NAMES else c for c in categories) + f" {kind}") if categories else f"all {kind}"
    laterality = d.get("laterality")
    side = ("ipsi and contra events, relative to the stimulus side" if isinstance(laterality, list)
            else "only events without a direction" if laterality == Laterality.NONDIRECTIONAL else "any direction")
    epoch = d.get("epoch_name") or Stim(d["stim"]).name.lower().replace("_", " ")
    epoch = " or ".join(epoch) if isinstance(epoch, list) else epoch
    filters = "".join(f", trials with {k} = {v}" for k, v in d.get("trial_filters", {}).items())
    return f"{what}; {side}; epoch: {epoch}{filters}; {d['t_start']:g} to {d['t_end']:g} s of each trial", short


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
    shape = structure(model, values, t_end)
    base = shape["base"] if shape["kind"] == "gain" else shape
    exgauss = "exGauss" in model.latex_formula
    params = []
    for n, (lo, hi) in zip(model.param_names, model.bounds):
        unit, info = describe(n, base["kind"], exgauss)
        params.append({"name": n, "value": values[n], "range": slider_range(values[n], lo, hi), "unit": unit, "info": info})
    return {"name": model.name, "latex": model.latex_formula.strip("$"), "fitted": fitted is not None, "params": params, "structure": shape}


def other_bouts(dataset: PointProcessDataset) -> List[List[Optional[List[int]]]]:
    """
    For PAGE_FISH fish drawn at random, per trial: the times (in PAGE_DT steps) of its bouts that are not events,
    the last one before the trial included (negative), or None where the fish lacks the trial.
    """
    fish = np.random.default_rng(0).choice(dataset.num_fish, min(PAGE_FISH, dataset.num_fish), replace=False)
    return [[np.round(dataset.other_times(f, m) / PAGE_DT).astype(int).tolist() if dataset.fish_trial_mask[f, m] else None
             for m in range(dataset.num_trials)] for f in fish]


def observed(dataset: PointProcessDataset, sided: bool, survival: bool) -> dict:
    """Events per trial and DATA_BIN time bin (ipsi and contra apart if sided; first events if survival) and fish per trial."""
    nb = round(dataset.duration_s / DATA_BIN)
    def counts(d: PointProcessDataset) -> List[int]:
        bins = np.minimum((d.event_times / DATA_BIN).astype(int), nb - 1)
        return np.bincount(d.event_trials_idx * nb + bins, minlength=dataset.num_trials * nb).tolist()
    data = {"bin": DATA_BIN, "n_fish": dataset.num_fish, "fish": dataset.n_fish_per_trial.astype(int).tolist()}
    if survival:
        return {**data, "first": counts(dataset.first_events())}
    if sided:
        return {**data, "ipsi": counts(dataset.select(laterality=IPSI)), "contra": counts(dataset.select(laterality=CONTRA))}
    return {**data, "all": counts(dataset)}


def export(output_dir: Optional[Path], loader: Optional[Callable[[str], BehavioralDataLoader]]) -> List[dict]:
    configs = get_model_config()
    experiments = []
    for name in [n for n in MAP_COLUMNS if n in configs]:
        config = configs[name]
        fits = {n: r["params"] for n, r in load_fit_records(output_dir / name / "models").items()} if output_dir else {}
        t_end = float(config["dataset"]["t_end"])
        models = [config["null_model"]] + [m for m in config["models"] if m.name != config["null_model"].name]
        best = output_dir / name / "models" / "best_model.json" if output_dir else None
        tier1 = next(m for m in models if m.name == json.loads(best.read_text())["name"]) if best and best.exists() else None
        free = [] if tier1 is None else [p for p in tier1.param_names if p in AMPLITUDE_PARAMETERS and p not in PINNED_AMPLITUDES.get(name, ())]
        events, selected = selection(config)
        entries = [model_entry(m, fits.get(m.name), t_end) for m in models]
        experiments.append({
            "name": name, "label": MAP_COLUMNS[name], "t_end": t_end, "tier1": tier1 and tier1.name, "free": free,
            "events": events, "selected": selected, "models": entries,
        })
        if loader:
            null = entries[0]["structure"]
            kind = (null["base"] if null["kind"] == "gain" else null)["kind"]
            dataset = loader(config["events"]).prepare_dataset(**config["dataset"])
            experiments[-1]["data"] = observed(dataset, kind.startswith("sided"), kind.endswith("survival"))
            if any(m.uses_history for m in models):
                experiments[-1]["data"]["others"] = other_bouts(dataset)
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output-dir", type=Path, help="Output folder of fit runs (<exp>/models/fit_records.json).")
    add_data_arguments(parser, required=False)
    args = parser.parse_args()
    data = json.dumps(export(args.output_dir, make_loaders(args) if args.data_root else None), separators=(",", ":"))
    DATA.write_text(f"window.EXPERIMENTS = {data};\n", encoding="utf-8")
    print(f"wrote {DATA} ({DATA.stat().st_size / 1e6:.1f} MB); open kernel_demo.html next to it")


if __name__ == "__main__":
    main()
