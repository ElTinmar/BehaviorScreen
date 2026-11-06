"""The three run modes: dataset diagnostics, model fitting, consolidation across experiments."""
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from BehaviorScreen.core import Laterality
from BehaviorScreen.point_process.point_process.dataset import (
    BehavioralDataLoader,
    DatasetPlotter,
    PointProcessDataset,
)
from BehaviorScreen.point_process.point_process.diagnostics import ModelPlotter
from BehaviorScreen.point_process.point_process.frailty_analysis import (
    collect_fish_gains,
    plot_fish_gain_correlation,
)
from BehaviorScreen.point_process.point_process.inference import ModelComparator
from BehaviorScreen.point_process.point_process.io import save_csv, save_fig
from BehaviorScreen.point_process.point_process.point_process import PointProcess
from BehaviorScreen.point_process.point_process.side_diagnostics import plot_side_diagnostics
from BehaviorScreen.point_process.point_process.sided_process import CONTRA, IPSI, SIDE_NAMES, SideView, sided_base

# (subdirectory, figure name, plot function, keyword arguments)
DATASET_PLOTS = [
    ("dataset_diagnostics", "isi_histogram", DatasetPlotter.plot_isi_histogram, {}),
    ("dataset_diagnostics", "event_count_distribution", DatasetPlotter.plot_event_count_distribution, {}),
    ("dataset_diagnostics", "fish_total_count_distribution", DatasetPlotter.plot_fish_total_count_distribution, {}),
    ("dataset_diagnostics", "psth", DatasetPlotter.plot_psth, {"bin_width_s": 0.25}),
    ("dataset_diagnostics", "time_trial_rate_heatmap", DatasetPlotter.plot_time_trial_rate_heatmap, {"bin_width_s": 0.25}),
    ("dataset_diagnostics", "trial_occupancy", DatasetPlotter.plot_trial_occupancy, {}),
    ("dataset_diagnostics", "fano_by_time_bin", DatasetPlotter.plot_fano_by_time_bin, {}),
    ("dataset_diagnostics", "isi_by_trial", DatasetPlotter.plot_isi_by_trial, {}),
    ("dataset_diagnostics", "raw_raster", DatasetPlotter.plot_raw_raster, {"max_fish": 15}),
    ("dataset_diagnostics", "fish_activity_heatmap", DatasetPlotter.plot_fish_activity_heatmap, {}),
    ("dataset_diagnostics", "fish_rank_activity", DatasetPlotter.plot_fish_rank_activity, {}),
    ("survival_diagnostics", "kaplan_meier", DatasetPlotter.plot_kaplan_meier, {}),
    ("survival_diagnostics", "repeat_event_gap", DatasetPlotter.plot_repeat_event_gap, {}),
    ("survival_diagnostics", "response_by_trial", DatasetPlotter.plot_response_by_trial, {}),
    ("survival_diagnostics", "fish_response_rate_distribution", DatasetPlotter.plot_fish_response_rate_distribution, {}),
]


@dataclass(frozen=True)
class Replicates:
    """Simulate-and-refit replicates of each rung's goodness of fit, fish-bootstrap replicates of the best; new: discard the saved ones."""
    gof: int = 300
    bootstrap: int = 200
    new: bool = False

    def store(self, folder: Path) -> Path:
        """Where a set of replicates is saved (emptied first when new)."""
        if self.new:
            shutil.rmtree(folder, ignore_errors=True)
        return folder


def gof_replicates_done(rung_dir: Path) -> int:
    """Successful GOF replicates behind a rung's dashboard (rows of its replicate table)."""
    table = rung_dir / "parametric_gof_replicates.csv"
    return len(pd.read_csv(table)) if table.exists() else 0


def run_diagnostics_task(exp_name: str, dataset: PointProcessDataset, output_root: Path) -> None:
    """Dataset plots of all events and, in a two-sided experiment, of each side in its own subfolder."""
    print(f"\n--- Dataset diagnostics: {exp_name} ---")
    views = {"": dataset}
    if {Laterality(IPSI).name, Laterality(CONTRA).name} <= set(dataset.laterality.split("+")):
        views |= {
            side_name: replace(dataset.select(laterality=side), laterality=Laterality(side).name)
            for side, side_name in SIDE_NAMES.items()
        }
    for side_name, data in views.items():
        for subdirectory, name, plot, kwargs in DATASET_PLOTS:
            fig, _ = plot(data, **kwargs)
            save_fig(fig, output_root / exp_name / subdirectory / side_name, name)
            plt.close(fig)


def run_fit_task(
    exp_name: str, config: dict, dataset: PointProcessDataset, output_root: Path, n_jobs: int = -1, refit: Union[str, List[int]] = "missing",
    replicates: Replicates = Replicates(),
) -> None:
    """
    Fit and compare the candidates, each rung checked and plotted once fitted
    (models/<rung_folder>); bootstrap of the best. refit: "missing" fits only the
    rungs without a saved fit, or whose saved NLL the current code and data no
    longer reproduce; "all" every rung; a list, those rung numbers (and the missing).
    A reused rung is checked and plotted again only if its folder is incomplete or
    its dashboard has fewer replicates than asked for (replicates are saved as they
    are done: an interrupted check resumes, a larger number only adds replicates).
    """
    print(f"\n=== Fitting {exp_name} ===")
    model_dir = output_root / exp_name / "models"
    models = config["models"]
    save_rung_index(models, model_dir)
    records, reused = load_fit_records(model_dir), set()

    def is_cached(index: int, model: PointProcess) -> bool:
        if refit == "all" or (isinstance(refit, list) and index in refit):
            print(f"[{rung_folder(index, model)}] refitting on request")
            return False
        verdict = reuse_saved_fit(model, records.get(model.name), dataset)
        print(f"[{rung_folder(index, model)}] {verdict}")
        if verdict.startswith("reused"):
            reused.add(model.name)
            return True
        return False

    def on_fitted(index: int, model: PointProcess) -> None:
        if model.name not in reused:
            records[model.name] = fit_record(model)
            save_fit_records(records, model_dir)
        out_dir = model_dir / rung_folder(index, model)
        complete = all((out_dir / f).exists() for f in ("diagnose.png", "parametric_gof_summary.csv"))
        if model.name not in reused or not complete or replicates.new or gof_replicates_done(out_dir) < replicates.gof:
            plot_rung(index, model, dataset, model_dir, n_jobs, replicates)
            if hasattr(model, "multistart_table_") and model.name not in reused:
                save_csv(model.multistart_table_, out_dir, "multistart")
        save_comparison_table(exp_name, models, records, model_dir)

    _, fitted_models = ModelComparator.compare(
        models=models, dataset=dataset, null_model=config["null_model"], on_fitted=on_fitted, is_cached=is_cached, n_starts=20, n_jobs=n_jobs
    )
    best = fitted_models[0]
    save_best_model(config, best, model_dir)

    fig, _ = ModelPlotter.plot_model_fits(dataset=dataset, models=fitted_models)
    save_fig(fig, model_dir, "model_fits_overlay")

    store = replicates.store(model_dir / rung_folder(models.index(best), best) / "bootstrap_replicates")
    save_csv(best.bootstrap(dataset, n_boot=replicates.bootstrap, n_jobs=n_jobs, store=store), model_dir, f"bootstrap_{best.name}")

    plt.close("all")


def reuse_saved_fit(model: PointProcess, record: Optional[dict], dataset: PointProcessDataset, rtol: float = 1e-6) -> str:
    """
    Sets the model to its saved fit if the current code and data reproduce the saved
    NLL; returns what was decided ("reused ..." or why it is refitted).
    """
    if record is None:
        return "no saved fit: fitting"
    params = np.asarray(record["params"], dtype=float)
    if len(params) != len(model.param_names):
        return f"saved fit has {len(params)} parameters, the model {len(model.param_names)}: refitting"
    nll = model._nll(list(params), dataset)
    if not np.isclose(nll, record["nll"], rtol=rtol, atol=1e-6):
        return f"saved NLL {record['nll']:.3f} is now {nll:.3f} (code, data or model changed): refitting"
    model.set_params(params)
    model.fit_result = OptimizeResult(x=params, fun=nll, success=True, message="saved fit")
    model.multistart_, model.fit_seconds_ = record["multistart"], record["fit_seconds"]
    return f"reused the saved fit (NLL {nll:.3f}, checked)"


def fit_record(model: PointProcess) -> dict:
    """What a fit leaves for later runs: parameters, NLL, multistart summary, time, when and with which code."""
    return {
        "params": model.params_.tolist(), "nll": float(model.fit_result.fun),
        "multistart": getattr(model, "multistart_", None), "fit_seconds": getattr(model, "fit_seconds_", None),
        "date": datetime.now().isoformat(timespec="seconds"), "code": code_version(),
    }


def code_version() -> Optional[str]:
    """Commit of the BehaviorScreen checkout (+dirty with uncommitted changes), None outside git."""
    here = str(Path(__file__).parent)
    try:
        commit = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def load_fit_records(model_dir: Path) -> Dict[str, dict]:
    """fit_records.json of earlier runs (model name: fit_record), empty without one."""
    path = model_dir / "fit_records.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_fit_records(records: Dict[str, dict], model_dir: Path) -> None:
    """fit_records.json, with the fits of models no longer in the ladder."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "fit_records.json").write_text(json.dumps(records, indent=2))


def save_comparison_table(exp_name: str, models: list, records: Dict[str, dict], model_dir: Path) -> None:
    """model_comparison_table.csv, one row per rung of the ladder (fitted so far), best AIC first."""
    null = models[0].log_likelihood if models[0].params_ is not None else np.nan
    rows = []
    for index, m in enumerate(models):
        fitted = m.params_ is not None and m.fit_result is not None
        ms = (getattr(m, "multistart_", None) or {}) if fitted else {}
        rows.append({
            "Condition": exp_name, "Rung": index, "Folder": rung_folder(index, m), "Model Name": m.name,
            "Params (k)": len(m.param_names), "Log-Likelihood": m.log_likelihood if fitted else np.nan,
            "AIC": m.aic if fitted else np.nan, "McFadden R2": 1.0 - m.log_likelihood / null if fitted else np.nan,
            "r_dispersion": m.dispersion_r if fitted else np.nan, "Overdispersion (1/r)": m.overdispersion_index if fitted else np.nan,
            "Converged": fitted, "Starts": ms.get("starts"), "Starts converged": ms.get("converged"), "Starts at best": ms.get("at_best"),
            "Nearest other optimum": ms.get("nearest_gap"), "Farthest other optimum": ms.get("farthest_gap"),
            "Fit (s)": getattr(m, "fit_seconds_", None) if fitted else None,
            "Fitted on": records.get(m.name, {}).get("date"), "Code": records.get(m.name, {}).get("code"),
        })
    table = pd.DataFrame(rows)
    table["ΔAIC"] = table["AIC"] - table["AIC"].min()
    weights = np.exp(-0.5 * table["ΔAIC"])
    table["AIC Weight"] = weights / np.nansum(weights)
    save_csv(table.sort_values("AIC", na_position="last").reset_index(drop=True), model_dir, "model_comparison_table")


def run_plot_task(
    exp_name: str, config: dict, dataset: PointProcessDataset, output_root: Path, n_jobs: int = -1, replicates: Replicates = Replicates(),
) -> None:
    """The checks and plots of every rung of a finished fit run (models/<rung_folder>), from its fitted parameters, without refitting it."""
    print(f"\n=== Plotting the rungs of {exp_name} ===")
    model_dir = output_root / exp_name / "models"
    for model in config["models"]:
        try:
            params = load_fitted_params(model_dir, model.name)
        except ValueError:
            continue
        model.set_params(params)
        model.fit_result = OptimizeResult(x=params, fun=model._nll(list(params), dataset), success=True)
    plot_rungs(config["models"], dataset, model_dir, n_jobs, replicates)


def plot_model(dataset: PointProcessDataset, model: PointProcess, out_dir: Path) -> None:
    """Rate surface, trial traces and, for a sided model, each side and the side calibration."""
    fig, _ = ModelPlotter.plot_histogram(dataset=dataset, model=model)
    save_fig(fig, out_dir, "histogram_surface")
    fig, _ = ModelPlotter.plot_trial_traces(dataset=dataset, model=model)
    save_fig(fig, out_dir, "trial_traces")
    if sided_base(model) is not None:
        plot_sides(dataset, model, out_dir)
    plt.close("all")


def plot_rungs(models: list, dataset: PointProcessDataset, model_dir: Path, n_jobs: int = -1, replicates: Replicates = Replicates()) -> None:
    """plot_rung for every fitted rung, in ladder order."""
    save_rung_index(models, model_dir)
    for index, model in enumerate(models):
        if model.params_ is not None:
            plot_rung(index, model, dataset, model_dir, n_jobs, replicates)


# Name pieces -> folder tokens, in folder order; each token at most once.
FOLDER_TOKENS = [
    (r"^ZeroInflated", "zi"), (r"GammaSideMixedEffects", "side_gain"), (r"GammaMixedEffects", "gain"),
    (r"BaselineOnly", "baseline_gain"), (r"\+Trial", "trial"), (r"SidedSurvival", "sided_survival"),
    (r"(?<!Sided)Survival ", "survival"), (r"Renewal", "renewal"), (r"Hawkes", "hawkes"),
]


def rung_folder(index: int, model: PointProcess) -> str:
    """
    Short folder name of a rung: its place in the ladder, what the model is made of
    (gains, history, stimulus or flat rate, burst shape) and 4 hex digits of its
    full name, so that a model always lands in its own folder, also after the ladder changes.
    """
    name = model.name
    tokens = [token for pattern, token in FOLDER_TOKENS if re.search(pattern, name)]
    kernels = re.sub(r",? ?history:.*|\]|^(Poisson|Survival|SidedSurvival|Renewal rate:) ", "", name.split("[")[-1])
    tokens.append("flat" if set(re.split(r" ?\| ?", kernels)) <= {"Homogeneous λ", "SurvivalConstantHazard"} else "stim")
    tokens += [f"k{k}" for k in re.findall(r"Burst\(k=([\d.]+)\)", name)]
    return f"{index:02d}_{'_'.join(tokens)}_{hashlib.md5(name.encode()).hexdigest()[:4]}"


def save_rung_index(models: list, model_dir: Path) -> None:
    """models/rung_index.csv: the folder and full name of every rung."""
    save_csv(pd.DataFrame({"rung": range(len(models)), "folder": [rung_folder(i, m) for i, m in enumerate(models)],
                           "model": [m.name for m in models]}), model_dir, "rung_index")


def plot_rung(
    index: int, model: PointProcess, dataset: PointProcessDataset, model_dir: Path, n_jobs: int = -1, replicates: Replicates = Replicates(),
) -> None:
    """
    In models/<rung_folder>: plot_model and the goodness-of-fit dashboard against
    simulate-and-refit replicates (saved in gof_replicates as they are done), with
    its check table and replicate statistics. A failure is reported, not raised: the fits go on.
    """
    out_dir = model_dir / rung_folder(index, model)
    try:
        plot_model(dataset, model, out_dir)
        store = replicates.store(out_dir / "gof_replicates")
        fig, diagnosis = model.diagnose(dataset, gof_n_boot=replicates.gof, gof_n_jobs=n_jobs, gof_store=store)
        save_fig(fig, out_dir, "diagnose")
        gof = diagnosis["parametric_gof"]
        save_csv(gof["summary"], out_dir, "parametric_gof_summary")
        save_csv(gof["bootstrap_statistics"], out_dir, "parametric_gof_replicates")
    except Exception as exc:
        print(f"[{model.name}] rung plots failed: {exc!r}")
    plt.close("all")


def plot_sides(dataset: PointProcessDataset, model: PointProcess, model_dir: Path) -> None:
    """Each side's rate surface against that side's fitted rate; side calibration by time, trial and fish."""
    for side, side_name in SIDE_NAMES.items():
        events, view = replace(dataset.select(laterality=side), laterality=side), SideView(model, side, dataset)
        fig, _ = ModelPlotter.plot_histogram(dataset=events, model=view)
        save_fig(fig, model_dir, f"histogram_surface_{side_name}")
    save_fig(plot_side_diagnostics(dataset, model), model_dir, "side_diagnostics")


def save_best_model(config: dict, model: PointProcess, model_dir: Path) -> None:
    with open(model_dir / "best_model.json", "w") as f:
        json.dump({"name": model.name, "params": model.params_.tolist()}, f, indent=2)


def load_best_model(config: dict, model_dir: Path) -> PointProcess:
    """The best model of a `fit` run, with its fitted parameters."""
    with open(model_dir / "best_model.json") as f:
        saved = json.load(f)
    model = next((m for m in config["models"] if m.name == saved["name"]), None)
    if model is None:
        raise ValueError(f"{model_dir}: best model {saved['name']!r} is no longer a candidate; rerun the fit")
    model.set_params(np.asarray(saved["params"], dtype=float))
    return model


def load_fitted_params(model_dir: Path, name: str) -> np.ndarray:
    """Fitted parameters of the candidate called `name` in a `fit` run."""
    record = load_fit_records(model_dir).get(name)
    if record is None:
        raise ValueError(f"{model_dir}: no fitted parameters for {name!r}; add it to the experiment's models and rerun the fit")
    return np.asarray(record["params"], dtype=float)


def summarize_dispersion_across_conditions(datasets: Dict[str, PointProcessDataset]) -> pd.DataFrame:
    records = [
        {
            "Condition": exp_name,
            "N Fish": len(dataset.fish_total_counts),
            "N Streams": len(dataset.stream_event_counts),
            "Mean Count/Stream": np.mean(dataset.stream_event_counts),
            "Stream Fano (DI)": dataset.stream_fano_factor,
            "Fish Fano (DI)": dataset.fish_fano_factor,
            "Fano Ratio (fish/stream)": dataset.dispersion_fano_ratio,
            "Frac Streams w/ >=2 events": dataset.frac_streams_with_multiple_events,
            "Low Power Flag": dataset.is_low_power_for_dispersion,
            "Mean ISI CV": dataset.mean_isi_cv,
            "ISI Lag-1 Autocorr": dataset.stream_isi_lag1_autocorr,
        }
        for exp_name, dataset in datasets.items()
    ]
    return (
        pd.DataFrame(records)
        .sort_values("Fano Ratio (fish/stream)", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def run_consolidation_task(
    model_config: dict, loader: Callable[[str], BehavioralDataLoader], output_root: Path
) -> None:
    """Cross-experiment tables and gain correlations, from the saved best models."""
    print("\n=== Consolidating experiment results ===")
    summaries, datasets, best_models = [], {}, {}
    for exp_name, config in model_config.items():
        model_dir = output_root / exp_name / "models"
        if not (model_dir / "best_model.json").exists():
            continue
        summaries.append(pd.read_csv(model_dir / "model_comparison_table.csv"))
        datasets[exp_name] = loader(config["events"]).prepare_dataset(**config["dataset"])
        best_models[exp_name] = load_best_model(config, model_dir)
    if not summaries:
        return

    master = pd.concat(summaries, ignore_index=True)
    print(master.to_string(index=False))
    save_csv(master, output_root, "master_model_comparison")
    save_csv(summarize_dispersion_across_conditions(datasets), output_root, "dispersion_summary")

    gains = collect_fish_gains({name: (best_models[name], datasets[name]) for name in best_models})
    if gains["behavior"].nunique() >= 2:
        fig, _, _ = plot_fish_gain_correlation(
            gains, title="Pooled control population: cross-behavior frailty gain correlation"
        )
        save_fig(fig, output_root, "fish_gain_correlation_pooled_population")
