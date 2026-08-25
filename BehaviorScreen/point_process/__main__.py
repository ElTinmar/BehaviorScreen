from pathlib import Path
from typing import Dict
import re
from pathlib import Path

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
    path = out_dir / f"{filename}.csv"
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
ROOT = next((p for p in possible_roots if p.exists()), possible_roots[0])
OUTPUT_ROOT = Path("./figures")

# Prey capture stim parameters
prey_stim_speed_deg_per_s = 90
prey_stim_range_deg = 2 * 70
prey_stim_freq = prey_stim_speed_deg_per_s / prey_stim_range_deg

loader = BehavioralDataLoader(ROOT / 'bouts_control.csv')

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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            GammaMixedEffectsProcess(
                PoissonProcess(RateKernelFactory.homogeneous_poisson())
            ),
            PoissonProcess(PreyCapture.time_only(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.baseline(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline_shared(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline_shared_ripple(stim_freq=prey_stim_freq)),
            PoissonProcess(PreyCapture.peak_baseline_ripple_shared(stim_freq=prey_stim_freq)),
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
            # HawkesProcess(
            #     RateKernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="A,B,gamma"),
            #     HistoryKernelFactory.exponential()
            # )
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            # HawkesProcess(
            #     RateKernelFactory.homogeneous_poisson(),
            #     HistoryKernelFactory.exponential()
            # )
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.phototaxis_ipsi()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.phototaxis_ipsi())),
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.phototaxis_contra()),
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson()))
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.omr_forward()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.omr_forward())),
            # HawkesProcess(
            #     RateKernelFactory.omr_forward(),
            #     HistoryKernelFactory.exponential()
            # )
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson()))
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.looming_gaussian()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.looming_gaussian())),
            # HawkesProcess(
            #     RateKernelFactory.looming_gaussian(),
            #     HistoryKernelFactory.exponential()
            # )
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.looming_gaussian()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.looming_gaussian())),
            # HawkesProcess(
            #     RateKernelFactory.looming_gaussian(),
            #     HistoryKernelFactory.exponential()
            # )
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
        'models': [
            PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            PoissonProcess(RateKernelFactory.dark_flash()),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.dark_flash()))
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

# <-- natural pause point: inspect the table + plots above and adjust
#     model_config['models'] per condition before Phase 3 runs, if needed -->


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
    )
    save_csv(summary_table, model_dir, "model_comparison_table")
    best_model = fitted_models[0]

    summary_table.insert(0, "Condition", exp_name)
    all_summaries.append(summary_table)

    print("\n--- MODEL COMPARISON TABLE ---")
    print(summary_table.to_string(index=False))

    fig, _ = ModelPlotter.plot_model_fits(dataset=dataset, models=fitted_models)
    save_fig(fig, model_dir, "model_fits_overlay")

    fig, _ = ModelPlotter.plot_histogram(dataset=dataset, model=fitted_models[0])
    save_fig(fig, model_dir, "histogram_surface")

    fig, _ = ModelPlotter.plot_trial_traces(dataset=dataset, model=best_model)
    save_fig(fig, model_dir, f"trial_traces_{best_model.name}")

    fig_diag, diag_results = best_model.diagnose(dataset)
    save_fig(fig_diag, model_dir, f"diagnose_{best_model.name}")
    # best_model.bootstrap(dataset, n_boot=500)

    plt.close('all')

master_summary_df = pd.concat(all_summaries, ignore_index=True)
print("\n================ MASTER MODEL COMPARISON TABLE ================")
print(master_summary_df.to_string(index=False))
save_csv(master_summary_df, OUTPUT_ROOT, "master_model_comparison")