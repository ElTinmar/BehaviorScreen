import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BehaviorScreen.core import Laterality, Stim
from BehaviorScreen.point_process.point_process.baseline_only_frailty_hawkes import (
    BaselineOnlyFrailtyHawkesProcess,
)
from BehaviorScreen.point_process.point_process.dataset import (
    BehavioralDataLoader,
    DatasetPlotter,
    PointProcessDataset,
)
from BehaviorScreen.point_process.point_process.hawkes_process import (
    HawkesProcess,
    HistoryKernelFactory,
)
from BehaviorScreen.point_process.point_process.io import save_csv, save_fig
from BehaviorScreen.point_process.point_process.mixed_effects_process import (
    GammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.point_process.point_process import (
    ModelComparator,
    ModelPlotter,
)
from BehaviorScreen.point_process.point_process.poisson_process import (
    PoissonProcess,
    PreyCapture,
    RateKernelFactory,
)
from BehaviorScreen.point_process.point_process.renewal_process import (
    RenewalKernelFactory,
    RenewalProcess,
)
from BehaviorScreen.point_process.point_process.residual_localization import (
    bootstrap_localization,
    plot_residual_localization,
)
from BehaviorScreen.point_process.point_process.survival_process import (
    SurvivalKernelFactory,
    SurvivalProcess,
)
from BehaviorScreen.point_process.point_process.zero_inflated_baseline_only_frailty_hawkes import (
    ZeroInflatedBaselineOnlyFrailtyHawkesProcess,
)
from BehaviorScreen.point_process.point_process.zero_inflated_mixed_effects_process import (
    ZeroInflatedGammaMixedEffectsProcess,
)


def get_model_config() -> dict:
    """Returns the experiment and model configuration dictionary."""
    prey_stim_speed_deg_per_s = 90
    prey_stim_range_deg = 2 * 70
    prey_stim_freq = prey_stim_speed_deg_per_s / prey_stim_range_deg

    return {
        "prey_capture_ipsi": {
            "dataset": {
                "stim": Stim.PREY_CAPTURE,
                "bout_name": "JT",
                "laterality": Laterality.IPSILATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 24.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(PreyCapture.time_only(stim_freq=prey_stim_freq)),
                PoissonProcess(PreyCapture.peak(stim_freq=prey_stim_freq)),
                PoissonProcess(PreyCapture.baseline(stim_freq=prey_stim_freq)),
                PoissonProcess(PreyCapture.peak_baseline(stim_freq=prey_stim_freq)),
                PoissonProcess(
                    PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq)
                ),
                PoissonProcess(
                    PreyCapture.peak_baseline_shared(stim_freq=prey_stim_freq)
                ),
                PoissonProcess(
                    PreyCapture.peak_baseline_shared_ripple(stim_freq=prey_stim_freq)
                ),
                PoissonProcess(
                    PreyCapture.peak_baseline_ripple_shared(stim_freq=prey_stim_freq)
                ),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                GammaMixedEffectsProcess(
                    PoissonProcess(
                        PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq)
                    )
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(
                        PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq)
                    )
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(
                        PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq)
                    ),
                    fit_c=True,
                ),
                ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
                    HawkesProcess(
                        PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq),
                        HistoryKernelFactory.exponential(),
                    ),
                ),
                HawkesProcess(
                    PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq),
                    HistoryKernelFactory.exponential(),
                ),
                BaselineOnlyFrailtyHawkesProcess(
                    HawkesProcess(
                        PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq),
                        HistoryKernelFactory.exponential(),
                    ),
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "prey_capture_contra": {
            "dataset": {
                "stim": Stim.PREY_CAPTURE,
                "bout_name": "JT",
                "laterality": Laterality.CONTRALATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 24.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                HawkesProcess(
                    RateKernelFactory.homogeneous_poisson(),
                    HistoryKernelFactory.exponential(),
                ),
                BaselineOnlyFrailtyHawkesProcess(
                    HawkesProcess(
                        RateKernelFactory.homogeneous_poisson(),
                        HistoryKernelFactory.exponential(),
                    ),
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
                    HawkesProcess(
                        RateKernelFactory.homogeneous_poisson(),
                        HistoryKernelFactory.exponential(),
                    ),
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.homogeneous_poisson(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "phototaxis_ipsi": {
            "dataset": {
                "stim": Stim.PHOTOTAXIS,
                "bout_name": "RT",
                "laterality": Laterality.IPSILATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 24.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.phototaxis_ipsi()),
                PoissonProcess(RateKernelFactory.phototaxis_dip_exgaussian_peak()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.phototaxis_ipsi())
                ),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.phototaxis_dip_exgaussian_peak())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.phototaxis_exgaussian_peak_no_dip(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "phototaxis_contra": {
            "dataset": {
                "stim": Stim.PHOTOTAXIS,
                "bout_name": "RT",
                "laterality": Laterality.CONTRALATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 24.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.phototaxis_contra()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.phototaxis_contra())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.phototaxis_contra())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.phototaxis_contra(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "omr_lateral_ipsi": {
            "dataset": {
                "epoch_name": ["grating right", "grating left"],
                "bout_name": "RT",
                "laterality": Laterality.IPSILATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 9.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.homogeneous_poisson(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "omr_lateral_contra": {
            "dataset": {
                "epoch_name": ["grating right", "grating left"],
                "bout_name": "RT",
                "laterality": Laterality.CONTRALATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 9.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.omr_lateral_contra()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.omr_lateral_contra())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.omr_lateral_contra())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.omr_lateral_contra(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "omr_forward": {
            "dataset": {
                "epoch_name": "grating forward",
                "bout_name": "BS",
                "laterality": Laterality.NONDIRECTIONAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 9.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.omr_forward()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.omr_forward())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.omr_forward())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.omr_forward(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "okr_ipsi": {
            "dataset": {
                "stim": Stim.OKR,
                "bout_name": "S1",
                "laterality": Laterality.IPSILATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 9.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.homogeneous_poisson(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "okr_contra": {
            "dataset": {
                "stim": Stim.OKR,
                "bout_name": "S1",
                "laterality": Laterality.CONTRALATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 9.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.homogeneous_poisson(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "looming_ipsi": {
            "dataset": {
                "stim": Stim.LOOMING,
                "bout_name": "SLC",
                "laterality": Laterality.IPSILATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 9.0,
            },
            "null_model": SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
            "models": [
                SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
                SurvivalProcess(
                    SurvivalKernelFactory.gaussian_bump_baseline(
                        t_init=5, t_bounds=(4, 6)
                    )
                ),
                SurvivalProcess(
                    SurvivalKernelFactory.gaussian_bump_baseline_habituating(
                        t_init=5, t_bounds=(4, 6)
                    )
                ),
                GammaMixedEffectsProcess(
                    SurvivalProcess(
                        SurvivalKernelFactory.gaussian_bump_baseline(
                            t_init=5, t_bounds=(4, 6)
                        )
                    )
                ),
                GammaMixedEffectsProcess(
                    SurvivalProcess(
                        SurvivalKernelFactory.gaussian_bump_baseline_habituating(
                            t_init=5, t_bounds=(4, 6)
                        )
                    )
                ),
            ],
        },
        "looming_contra": {
            "dataset": {
                "stim": Stim.LOOMING,
                "bout_name": "SLC",
                "laterality": Laterality.CONTRALATERAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 9.0,
            },
            "null_model": SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
            "models": [
                SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
                SurvivalProcess(
                    SurvivalKernelFactory.gaussian_bump_baseline(
                        t_init=5, t_bounds=(4, 6)
                    )
                ),
                SurvivalProcess(
                    SurvivalKernelFactory.gaussian_bump_baseline_habituating(
                        t_init=5, t_bounds=(4, 6)
                    )
                ),
                GammaMixedEffectsProcess(
                    SurvivalProcess(
                        SurvivalKernelFactory.gaussian_bump_baseline(
                            t_init=5, t_bounds=(4, 6)
                        )
                    )
                ),
                GammaMixedEffectsProcess(
                    SurvivalProcess(
                        SurvivalKernelFactory.gaussian_bump_baseline_habituating(
                            t_init=5, t_bounds=(4, 6)
                        )
                    )
                ),
            ],
        },
        "dark_flash": {
            "dataset": {
                "epoch_name": "flash dark",
                "bout_name": "O",
                "laterality": Laterality.NONDIRECTIONAL,
                "binning_dt": 0.025,
                "t_start": 0.0,
                "t_end": 5.0,
            },
            "null_model": SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
            "models": [
                SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
                SurvivalProcess(
                    SurvivalKernelFactory.gaussian_bump_baseline(
                        t_init=0.2, t_bounds=(0.01, 1)
                    )
                ),
                SurvivalProcess(
                    SurvivalKernelFactory.gaussian_bump_baseline_habituating(
                        t_init=0.2, t_bounds=(0.01, 1)
                    )
                ),
                SurvivalProcess(
                    SurvivalKernelFactory.exgaussian_bump_baseline_habituating()
                ),
                GammaMixedEffectsProcess(
                    SurvivalProcess(
                        SurvivalKernelFactory.gaussian_bump_baseline(
                            t_init=0.2, t_bounds=(0.01, 1)
                        )
                    )
                ),
                GammaMixedEffectsProcess(
                    SurvivalProcess(
                        SurvivalKernelFactory.gaussian_bump_baseline_habituating(
                            t_init=0.2, t_bounds=(0.01, 1)
                        )
                    )
                ),
                GammaMixedEffectsProcess(
                    SurvivalProcess(
                        SurvivalKernelFactory.exgaussian_bump_baseline_habituating()
                    )
                ),
            ],
        },
        "spont_dark": {
            "dataset": {
                "epoch_name": "spontaneous dark",
                "bout_name": "RT",
                "laterality": Laterality.NONDIRECTIONAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 24.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.spont()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.spont())),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.spont())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.spont(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "spont_bright": {
            "dataset": {
                "epoch_name": "spontaneous bright",
                "bout_name": "RT",
                "laterality": Laterality.NONDIRECTIONAL,
                "binning_dt": 0.05,
                "t_start": 0.0,
                "t_end": 24.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.spont()),
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.homogeneous_poisson())
                ),
                GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.spont())),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.spont())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.spont(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
        "after_looming": {
            "dataset": {
                "epoch_name": ["looming break after left", "looming break after right"],
                "bout_name": "RT",
                "laterality": [Laterality.IPSILATERAL, Laterality.CONTRALATERAL],
                "binning_dt": 0.05,
                "t_start": 0,
                "t_end": 49.0,
            },
            "null_model": PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            "models": [
                GammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.after_looming())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    PoissonProcess(RateKernelFactory.after_looming())
                ),
                ZeroInflatedGammaMixedEffectsProcess(
                    RenewalProcess(
                        RateKernelFactory.after_looming(),
                        RenewalKernelFactory.delayed_excitation(),
                    )
                ),
            ],
        },
    }


def summarize_dispersion_across_conditions(
    datasets: Dict[str, PointProcessDataset],
) -> pd.DataFrame:
    """Builds a cross-condition comparison table of dispersion diagnostics."""
    records = []
    for exp_name, dataset in datasets.items():
        records.append(
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
        )

    df = pd.DataFrame(records)
    return df.sort_values(
        "Fano Ratio (fish/stream)", ascending=False, na_position="last"
    ).reset_index(drop=True)


def run_diagnostics_task(
    exp_name: str,
    dataset: PointProcessDataset,
    output_root: Path,
):
    """Generates dataset diagnostic plots."""
    print(f"\n--- Running dataset diagnostics: {exp_name} ---")
    diag_dir = output_root / exp_name / "dataset_diagnostics"

    fig, _ = DatasetPlotter.plot_isi_histogram(dataset)
    save_fig(fig, diag_dir, "isi_histogram")

    fig, _ = DatasetPlotter.plot_event_count_distribution(dataset)
    save_fig(fig, diag_dir, "event_count_distribution")

    fig, _ = DatasetPlotter.plot_fish_total_count_distribution(dataset)
    save_fig(fig, diag_dir, "fish_total_count_distribution")

    fig, _ = DatasetPlotter.plot_psth(dataset, bin_width_s=0.25)
    save_fig(fig, diag_dir, "psth")

    fig, _ = DatasetPlotter.plot_time_trial_rate_heatmap(dataset, bin_width_s=0.25)
    save_fig(fig, diag_dir, "time_trial_rate_heatmap")

    fig, _ = DatasetPlotter.plot_trial_occupancy(dataset)
    save_fig(fig, diag_dir, "trial_occupancy")

    fig, _ = DatasetPlotter.plot_fano_by_time_bin(dataset)
    save_fig(fig, diag_dir, "fano_by_time_bin")

    fig, _ = DatasetPlotter.plot_isi_by_trial(dataset)
    save_fig(fig, diag_dir, "isi_by_trial")

    fig, _ = DatasetPlotter.plot_raw_raster(dataset, max_fish=15)
    save_fig(fig, diag_dir, "raw_raster")

    fig, _ = DatasetPlotter.plot_fish_activity_heatmap(dataset)
    save_fig(fig, diag_dir, "fish_activity_heatmap")

    fig, _ = DatasetPlotter.plot_fish_rank_activity(dataset)
    save_fig(fig, diag_dir, "fish_rank_activity")

    survival_diag_dir = output_root / exp_name / "survival_diagnostics"
    print(f"--- Survival diagnostics: {exp_name} ---")

    fig, _ = DatasetPlotter.plot_kaplan_meier(dataset)
    save_fig(fig, survival_diag_dir, "kaplan_meier")

    fig, _ = DatasetPlotter.plot_repeat_event_gap(dataset)
    save_fig(fig, survival_diag_dir, "repeat_event_gap")

    fig, _ = DatasetPlotter.plot_response_by_trial(dataset)
    save_fig(fig, survival_diag_dir, "response_by_trial")

    fig, _ = DatasetPlotter.plot_fish_response_rate_distribution(dataset)
    save_fig(fig, survival_diag_dir, "fish_response_rate_distribution")

    plt.close("all")


def run_fit_task(
    exp_name: str,
    config: dict,
    dataset: PointProcessDataset,
    output_root: Path,
):
    """Performs model fitting, AIC comparison, bootstrapping, and residual localization."""
    print(f"\n==================================================")
    print(f" PROCESSING FIT EXPERIMENT: {exp_name.upper()}")
    print(f"==================================================")

    model_dir = output_root / exp_name / "models"

    summary_table, fitted_models = ModelComparator.compare(
        models=config["models"],
        dataset=dataset,
        null_model=config["null_model"],
        n_starts=40,
    )
    summary_table.insert(0, "Condition", exp_name)
    save_csv(summary_table, model_dir, "model_comparison_table")

    best_model = fitted_models[0]

    fig, _ = ModelPlotter.plot_model_fits(dataset=dataset, models=fitted_models)
    save_fig(fig, model_dir, "model_fits_overlay")

    fig, _ = ModelPlotter.plot_histogram(dataset=dataset, model=best_model)
    save_fig(fig, model_dir, f"histogram_surface_{best_model.name}")

    fig, _ = ModelPlotter.plot_trial_traces(dataset=dataset, model=best_model)
    save_fig(fig, model_dir, f"trial_traces_{best_model.name}")

    fig_diag, diag_results = best_model.diagnose(dataset)
    save_fig(fig_diag, model_dir, f"diagnose_{best_model.name}")

    gof_result = diag_results["parametric_gof"]
    save_csv(
        gof_result["summary"],
        model_dir,
        f"parametric_gof_summary_{best_model.name}",
    )
    save_csv(
        gof_result["bootstrap_statistics"],
        model_dir,
        f"parametric_gof_replicates_{best_model.name}",
    )

    boot_df = best_model.bootstrap(dataset, n_boot=200)
    save_csv(boot_df, model_dir, f"bootstrap_{best_model.name}")

    localization = bootstrap_localization(
        best_model,
        dataset,
        n_boot=100,
        seed=123,
        refit=True,
        refit_n_starts=1,
        min_km_at_risk=1,
        r_grid=np.linspace(0, 5, 151),
        time_edges=np.linspace(0, dataset.duration_s, 7),
        gap_edges=np.linspace(
            0,
            min(dataset.duration_s, 5.0),
            41,
        ),
    )
    fig, _ = plot_residual_localization(localization)
    save_fig(fig, model_dir, f"residual_localization_{best_model.name}")

    plt.close("all")


def run_consolidation_task(
    model_config: dict,
    loader: BehavioralDataLoader,
    output_root: Path,
):
    """Consolidates cross-experiment results after cluster jobs finish."""
    print("\n================ CONSOLIDATING EXPERIMENT RESULTS ================")
    all_summaries = []
    datasets = {}
    best_models = {}

    for exp_name, config in model_config.items():
        summary_path = output_root / exp_name / "models" / "model_comparison_table.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            all_summaries.append(df)

            dataset = loader.prepare_dataset(**config["dataset"])
            datasets[exp_name] = dataset

            best_model_name = df.iloc[0]["Model"]
            best_model_obj = next(
                m for m in config["models"] if m.name == best_model_name
            )
            best_models[exp_name] = best_model_obj

    if all_summaries:
        master_summary_df = pd.concat(all_summaries, ignore_index=True)
        print("\n================ MASTER MODEL COMPARISON TABLE ================")
        print(master_summary_df.to_string(index=False))
        save_csv(master_summary_df, output_root, "master_model_comparison")

        dispersion_summary = summarize_dispersion_across_conditions(datasets)
        save_csv(dispersion_summary, output_root, "dispersion_summary")

        models_and_datasets = {
            exp_name: (best_models[exp_name], datasets[exp_name])
            for exp_name in best_models
            if hasattr(best_models[exp_name], "estimate_fish_gains")
        }
        if models_and_datasets:
            gain_df = collect_fish_gains(models_and_datasets)
            fig, ax, corr = plot_fish_gain_correlation(
                gain_df,
                title="Pooled control population: cross-behavior frailty gain correlation",
            )
            save_fig(fig, output_root, "fish_gain_correlation_pooled_population")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Point Process Analysis Suite"
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="Specific experiment key to process (e.g., prey_capture_ipsi).",
    )
    parser.add_argument(
        "--mode",
        choices=["diagnostics", "fit", "consolidate"],
        default="fit",
        help="Task execution mode.",
    )
    parser.add_argument(
        "-d",
        "--data-root",
        type=Path,
        action="append",
        dest="data_root",
        help="Path to data directory/directories (can specify multiple times).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./figures"),
        help="Directory where output figures and CSVs will be stored (default: ./figures).",
    )

    args = parser.parse_args()
    csv_filename = "bouts_control.csv"

    model_config = get_model_config()
    output_root = args.output_dir

    if args.mode == "consolidate":
        loader = BehavioralDataLoader(args.data_root / csv_filename)
        run_consolidation_task(model_config, loader, output_root)
        return

    if not args.exp:
        raise ValueError(
            "Please provide an experiment target using `--exp <exp_name>` when running in 'fit' or 'diagnostics' mode."
        )

    if args.exp not in model_config:
        raise ValueError(
            f"Unknown experiment '{args.exp}'. Valid keys: {list(model_config.keys())}"
        )

    loader = BehavioralDataLoader(args.data_root / csv_filename)
    config = model_config[args.exp]
    dataset = loader.prepare_dataset(**config["dataset"])

    if args.mode == "diagnostics":
        run_diagnostics_task(args.exp, dataset, output_root)
    elif args.mode == "fit":
        run_fit_task(args.exp, config, dataset, output_root)


if __name__ == "__main__":
    main()