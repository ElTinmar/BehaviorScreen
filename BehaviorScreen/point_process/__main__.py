from pathlib import Path
from typing import Dict
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BehaviorScreen.core import Stim, Laterality
from BehaviorScreen.point_process.dataset import (
    BehavioralDataLoader, PointProcessDataset, DatasetPlotter,
)
from BehaviorScreen.point_process.point_process import ModelComparator, ModelPlotter
from BehaviorScreen.point_process.poisson_process import RateKernelFactory, PoissonProcess, PreyCapture
from BehaviorScreen.point_process.hawkes_process import HistoryKernelFactory, HawkesProcess
from BehaviorScreen.point_process.renewal_process import RenewalKernelFactory, RenewalProcess
from BehaviorScreen.point_process.mixed_effects_process import GammaMixedEffectsProcess
from BehaviorScreen.point_process.survival_process import SurvivalProcess, SurvivalKernelFactory

def slugify(name: str) -> str:
    """Turn an arbitrary model/condition name into a filesystem-safe filename fragment."""
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)   # anything not alnum/_/-/. -> underscore
    return re.sub(r"_+", "_", name).strip("_")

def save_fig(fig: plt.Figure, out_dir: Path, filename: str, dpi: int = 150) -> Path:
    """Save fig as PNG under out_dir/filename (creating out_dir if needed) and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(filename)}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

def save_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    """Save df as CSV under out_dir/filename.csv (creating out_dir if needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(filename)}.csv"
    df.to_csv(path, index=False)
    return path

def summarize_dispersion_across_conditions(datasets: Dict[str, PointProcessDataset]) -> pd.DataFrame:
    """
    Builds a cross-condition comparison table of dispersion diagnostics
    (see PointProcessDataset.dispersion_* properties) for every dataset
    in `datasets`. Sorted by dispersion_fano_ratio descending.
    """
    records = []
    for exp_name, dataset in datasets.items():
        records.append({
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
        })

    df = pd.DataFrame(records)
    return df.sort_values(
        "Fano Ratio (fish/stream)", ascending=False, na_position="last"
    ).reset_index(drop=True)


# =============================================================================
# Setup
# =============================================================================

possible_roots = [
    Path('/home/martin/Desktop/DATA'),
    Path('/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen'),
    Path('/media/martin/DATA_18TB/Screen'),
]
# possible_roots = [Path('/media/martin/DATA_18TB/Screen/WT/danieau')]

ROOT = next((p for p in possible_roots if p.exists()), possible_roots[0])
OUTPUT_ROOT = Path("./figures")

# Prey capture stim parameters
prey_stim_speed_deg_per_s = 90
prey_stim_range_deg = 2 * 70
prey_stim_freq = prey_stim_speed_deg_per_s / prey_stim_range_deg

loader = BehavioralDataLoader(ROOT / 'bouts_control.csv')
#loader = BehavioralDataLoader(ROOT / 'bouts.csv')


model_config = {

    'prey_capture_ipsi': {
        'dataset': {
            'stim': Stim.PREY_CAPTURE,
            'bout_name': 'JT',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 24.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(PreyCapture.time_only(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.baseline(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline_shared(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline_shared_ripple(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline_ripple_shared(stim_freq=prey_stim_freq)),
            GammaMixedEffectsProcess(
                PoissonProcess(RateKernelFactory.homogeneous_poisson())
            ),
            GammaMixedEffectsProcess(
                PoissonProcess(PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq))
            ),
            RenewalProcess(
                RateKernelFactory.homogeneous_poisson(), 
                RenewalKernelFactory.exponential_excitation()
            ),
            GammaMixedEffectsProcess(
                RenewalProcess(
                    RateKernelFactory.homogeneous_poisson(), 
                    RenewalKernelFactory.exponential_excitation()
                )
            ),
            RenewalProcess(
                PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq), 
                RenewalKernelFactory.exponential_excitation()
            ),
            GammaMixedEffectsProcess(
                RenewalProcess(
                    PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq), 
                    RenewalKernelFactory.exponential_excitation()
                )
            ),
            HawkesProcess(
                PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq),
                HistoryKernelFactory.exponential()
            ),
            GammaMixedEffectsProcess(
                HawkesProcess(
                    PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq),
                    HistoryKernelFactory.exponential()
                )
            )
        ]
    },

    'prey_capture_contra': {
        'dataset': {
            'stim': Stim.PREY_CAPTURE,
            'bout_name': 'JT',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 24.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            HawkesProcess(
                RateKernelFactory.homogeneous_poisson(),
                HistoryKernelFactory.exponential()
            )
        ]
    },

    'phototaxis_ipsi': {
        'dataset': {
            'stim': Stim.PHOTOTAXIS,
            'bout_name': 'RT',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 24.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.phototaxis_ipsi()),
            PoissonProcess(RateKernelFactory.phototaxis_dip_exgaussian_peak()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.phototaxis_ipsi())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.phototaxis_dip_exgaussian_peak())),
        ]
    },

    'phototaxis_contra': {
        'dataset': {
            'stim': Stim.PHOTOTAXIS,
            'bout_name': 'RT',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 24.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.phototaxis_contra()),
            RenewalProcess(
                RateKernelFactory.phototaxis_contra(),
                RenewalKernelFactory.exponential_excitation()
            ),
            HawkesProcess(
                RateKernelFactory.phototaxis_contra(),
                HistoryKernelFactory.exponential()
            ),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.phototaxis_contra()))
        ]
    },

    'omr_lateral_ipsi': {
        'dataset': {
            'epoch_name': ["grating right", "grating left"],
            'bout_name': 'RT',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 9.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            HawkesProcess(
                RateKernelFactory.homogeneous_poisson(),
                HistoryKernelFactory.exponential()
            ),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(
                HawkesProcess(
                    RateKernelFactory.homogeneous_poisson(),
                    HistoryKernelFactory.exponential()
                )
            ),
        ]
    },

    'omr_lateral_contra': {
        'dataset': {
            'epoch_name': ["grating right", "grating left"],
            'bout_name': 'RT',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 9.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.omr_lateral_contra()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.omr_lateral_contra())),
        ]
    },

    'omr_forward': {
        'dataset': {
            'epoch_name': "grating forward",
            'bout_name': 'BS',
            'laterality': Laterality.NONDIRECTIONAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 9.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.omr_forward()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.omr_forward())),
            HawkesProcess(
                RateKernelFactory.omr_forward(),
                HistoryKernelFactory.exponential()
            )
        ]
    },

    'okr_ipsi': {
        'dataset': {
            'stim': Stim.OKR,
            'bout_name': 'S1',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 9.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson()))
        ]
    },

    'okr_contra': {
        'dataset': {
            'stim': Stim.OKR,
            'bout_name': 'S1',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 9.0,
        },
        'null_model': PoissonProcess(RateKernelFactory.homogeneous_poisson()),
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            RenewalProcess(
                RateKernelFactory.homogeneous_poisson(), 
                RenewalKernelFactory.exponential_recovery()
            ),
            RenewalProcess(
                RateKernelFactory.homogeneous_poisson(), 
                RenewalKernelFactory.exponential_excitation()
            ),
            HawkesProcess(
                RateKernelFactory.homogeneous_poisson(),
                HistoryKernelFactory.exponential()
            ),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(
                RenewalProcess(
                    RateKernelFactory.homogeneous_poisson(),
                    RenewalKernelFactory.exponential_excitation()
                )
            ),
            GammaMixedEffectsProcess(
                HawkesProcess(
                    RateKernelFactory.homogeneous_poisson(),
                    HistoryKernelFactory.exponential()
                )
            ),
        ]
    },

    'looming_ipsi': {
        'dataset': {
            'stim': Stim.LOOMING,
            'bout_name': 'SLC',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 9.0,
        },
        'null_model': SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
        'models': [
            SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
            SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=5, t_bounds=(4,6))),
            SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4,6))),
            GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=5, t_bounds=(4,6)))),
            GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4,6)))),
        ]
    },

    'looming_contra': {
        'dataset': {
            'stim': Stim.LOOMING,
            'bout_name': 'SLC',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.05,
            't_start': 0.0,
            't_end': 9.0,
        },
        'null_model': SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
        'models': [
            SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
            SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=5, t_bounds=(4,6))),
            SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4,6))),
            GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=5, t_bounds=(4,6)))),
            GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4,6)))),
        ]
    },

    'dark_flash': {
        'dataset': {
            'epoch_name': "flash dark",
            'bout_name': 'O',
            'laterality': Laterality.NONDIRECTIONAL,
            'binning_dt': 0.025,
            't_start': 0.0,
            't_end': 5.0,
        },
        'null_model': SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
        'models': [
            SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
            SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=0.2, t_bounds=(0.01,1))),
            SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=0.2, t_bounds=(0.01,1))),
            SurvivalProcess(SurvivalKernelFactory.exgaussian_bump_baseline_habituating()),
            GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline(t_init=0.2, t_bounds=(0.01,1)))),
            GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=0.2, t_bounds=(0.01,1)))),
            GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.exgaussian_bump_baseline_habituating())),
        ]
    },
}


# =============================================================================
# Phase 1: build every dataset once
# =============================================================================

print("Loading datasets for all conditions...")
datasets: Dict[str, PointProcessDataset] = {
    exp_name: loader.prepare_dataset(**config['dataset'])
    for exp_name, config in model_config.items()
}


# =============================================================================
# Phase 2: dataset-only diagnostics (no model fitting yet)
# =============================================================================

print("\n================ DISPERSION / OVERDISPERSION SUMMARY ================")
print("Review this BEFORE trusting fitted model results below -- conditions")
print("with high Fano Ratio and Low Power Flag == False likely need a")
print("fish-level heterogeneity term; see PointProcessDataset docstrings.")
dispersion_summary = summarize_dispersion_across_conditions(datasets)
print(dispersion_summary.to_string(index=False))
save_csv(dispersion_summary, OUTPUT_ROOT, "dispersion_summary")

for exp_name, dataset in datasets.items():
    diag_dir = OUTPUT_ROOT / exp_name / "dataset_diagnostics"
    print(f"\n--- Dataset diagnostics: {exp_name} ---")

    fig, _ = DatasetPlotter.plot_isi_histogram(dataset)
    save_fig(fig, diag_dir, "isi_histogram")

    fig, _ = DatasetPlotter.plot_event_count_distribution(dataset)
    save_fig(fig, diag_dir, "event_count_distribution")

    fig, _ = DatasetPlotter.plot_fish_total_count_distribution(dataset)
    save_fig(fig, diag_dir, "fish_total_count_distribution")
    plt.close("all")

    fig, _ = DatasetPlotter.plot_psth(dataset, bin_width_s=0.25)
    save_fig(fig, diag_dir, "psth")

    fig, _ = DatasetPlotter.plot_time_trial_rate_heatmap(dataset, bin_width_s=0.25)
    save_fig(fig, diag_dir, "time_trial_rate_heatmap")

    fig, _ = DatasetPlotter.plot_trial_occupancy(dataset)
    save_fig(fig, diag_dir, "trial_occupancy")

    fig, _ = DatasetPlotter.plot_fano_by_time_bin(dataset)
    save_fig(fig, diag_dir, "fano_by_time_bin")
    plt.close("all")

    fig, _ = DatasetPlotter.plot_isi_by_trial(dataset)
    save_fig(fig, diag_dir, "isi_by_trial")

    fig, _ = DatasetPlotter.plot_raw_raster(dataset, max_fish=15)
    save_fig(fig, diag_dir, "raw_raster")

    fig, _ = DatasetPlotter.plot_fish_activity_heatmap(dataset)  # now sorted by default
    save_fig(fig, diag_dir, "fish_activity_heatmap")

    fig, _ = DatasetPlotter.plot_fish_rank_activity(dataset)
    save_fig(fig, diag_dir, "fish_rank_activity")

    survival_diag_dir = OUTPUT_ROOT / exp_name / "survival_diagnostics"
    print(f"--- Survival-specific diagnostics: {exp_name} ---")

    fig, _ = DatasetPlotter.plot_kaplan_meier(dataset)
    save_fig(fig, survival_diag_dir, "kaplan_meier")

    fig, _ = DatasetPlotter.plot_repeat_event_gap(dataset)
    save_fig(fig, survival_diag_dir, "repeat_event_gap")

    fig, _ = DatasetPlotter.plot_response_by_trial(dataset)
    save_fig(fig, survival_diag_dir, "response_by_trial")

    fig, _ = DatasetPlotter.plot_fish_response_rate_distribution(dataset)
    save_fig(fig, survival_diag_dir, "fish_response_rate_distribution")

    plt.close("all")


# =============================================================================
# Phase 3: model fitting + comparison
# =============================================================================

all_summaries = []

for exp_name, config in model_config.items():

    print(f"\n==================================================")
    print(f" PROCESSING EXPERIMENT: {exp_name.upper()}")
    print(f"==================================================")

    dataset = datasets[exp_name]
    model_dir = OUTPUT_ROOT / exp_name / "models"

    summary_table, fitted_models = ModelComparator.compare(
        models=config['models'],
        dataset=dataset,
        null_model=config['null_model']
    )
    save_csv(summary_table, model_dir, "model_comparison_table")
    best_model = fitted_models[0]

    summary_table.insert(0, "Condition", exp_name)
    all_summaries.append(summary_table)

    print("\n--- MODEL COMPARISON TABLE ---")
    print(summary_table.to_string(index=False))

    fig, _ = ModelPlotter.plot_model_fits(dataset=dataset, models=fitted_models)
    save_fig(fig, model_dir, "model_fits_overlay")

    fig, _ = ModelPlotter.plot_histogram(dataset=dataset, model=best_model)
    save_fig(fig, model_dir, "histogram_surface")

    fig, _ = ModelPlotter.plot_trial_traces(dataset=dataset, model=best_model)
    save_fig(fig, model_dir, f"trial_traces_{best_model.name}")

    fig_diag, diag_results = best_model.diagnose(dataset)
    save_fig(fig_diag, model_dir, f"diagnose_{best_model.name}")

    # NOTE: this might take a while
    boot_df = best_model.bootstrap(dataset, n_boot=100)
    save_csv(boot_df, model_dir, f"bootstrap_{best_model.name}")

    plt.close('all')

master_summary_df = pd.concat(all_summaries, ignore_index=True)
print("\n================ MASTER MODEL COMPARISON TABLE ================")
print(master_summary_df.to_string(index=False))
save_csv(master_summary_df, OUTPUT_ROOT, "master_model_comparison")