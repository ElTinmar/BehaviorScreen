"""The three run modes: dataset diagnostics, model fitting, consolidation across experiments."""
import json
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def run_diagnostics_task(exp_name: str, dataset: PointProcessDataset, output_root: Path) -> None:
    print(f"\n--- Dataset diagnostics: {exp_name} ---")
    for subdirectory, name, plot, kwargs in DATASET_PLOTS:
        fig, _ = plot(dataset, **kwargs)
        save_fig(fig, output_root / exp_name / subdirectory, name)
        plt.close(fig)


def run_fit_task(exp_name: str, config: dict, dataset: PointProcessDataset, output_root: Path, n_jobs: int = -1) -> None:
    """Fit and compare the candidates; goodness-of-fit dashboard and bootstrap of the best."""
    print(f"\n=== Fitting {exp_name} ===")
    model_dir = output_root / exp_name / "models"

    summary_table, fitted_models = ModelComparator.compare(
        models=config["models"], dataset=dataset, null_model=config["null_model"], n_starts=40, n_jobs=n_jobs
    )
    summary_table.insert(0, "Condition", exp_name)
    save_csv(summary_table, model_dir, "model_comparison_table")

    best = fitted_models[0]
    save_best_model(config, best, model_dir)
    save_fitted_models(config, model_dir)

    fig, _ = ModelPlotter.plot_model_fits(dataset=dataset, models=fitted_models)
    save_fig(fig, model_dir, "model_fits_overlay")
    fig, _ = ModelPlotter.plot_histogram(dataset=dataset, model=best)
    save_fig(fig, model_dir, f"histogram_surface_{best.name}")
    fig, _ = ModelPlotter.plot_trial_traces(dataset=dataset, model=best)
    save_fig(fig, model_dir, f"trial_traces_{best.name}")

    fig, diagnosis = best.diagnose(dataset, gof_n_jobs=n_jobs)
    save_fig(fig, model_dir, f"diagnose_{best.name}")
    gof = diagnosis["parametric_gof"]
    save_csv(gof["summary"], model_dir, f"parametric_gof_summary_{best.name}")
    save_csv(gof["bootstrap_statistics"], model_dir, f"parametric_gof_replicates_{best.name}")

    save_csv(best.bootstrap(dataset, n_boot=200, n_jobs=n_jobs), model_dir, f"bootstrap_{best.name}")

    plt.close("all")


def save_best_model(config: dict, model: PointProcess, model_dir: Path) -> None:
    index = next(i for i, m in enumerate(config["models"]) if m is model)
    with open(model_dir / "best_model.json", "w") as f:
        json.dump({"model_index": index, "name": model.name, "params": model.params_.tolist()}, f, indent=2)


def save_fitted_models(config: dict, model_dir: Path) -> None:
    """Parameters of every candidate, in config order (None where the fit failed)."""
    fitted = [{"name": m.name, "params": None if m.params_ is None else m.params_.tolist()} for m in config["models"]]
    with open(model_dir / "fitted_models.json", "w") as f:
        json.dump(fitted, f, indent=2)


def _set_saved_params(model: PointProcess, saved: dict, index: int, model_dir: Path) -> PointProcess:
    if model.name != saved["name"]:
        raise ValueError(f"{model_dir}: model {index} is now {model.name!r}, was {saved['name']!r} when fitted; rerun the fit")
    if saved["params"] is None:
        raise ValueError(f"{model_dir}: model {index} ({model.name}) failed to fit")
    model.set_params(np.asarray(saved["params"], dtype=float))
    return model


def load_best_model(config: dict, model_dir: Path) -> PointProcess:
    """The best model of a `fit` run, with its fitted parameters."""
    with open(model_dir / "best_model.json") as f:
        saved = json.load(f)
    return _set_saved_params(config["models"][saved["model_index"]], saved, saved["model_index"], model_dir)


def load_fitted_model(config: dict, model_dir: Path, index: int) -> PointProcess:
    """Candidate `index` of a `fit` run, with its fitted parameters."""
    with open(model_dir / "fitted_models.json") as f:
        saved = json.load(f)[index]
    return _set_saved_params(config["models"][index], saved, index, model_dir)


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
