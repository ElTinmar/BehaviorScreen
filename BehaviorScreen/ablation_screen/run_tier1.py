"""
Tier 1 end-to-end run. BEHAVIOR_CONFIG is the single source of truth: one
entry per behavior with its dataset filter, the (already partially-fixed)
architecture factory, and the null model. Shape parameters are pinned
directly in each 'architecture' lambda via PartiallyFixedProcess, using
values already fitted on the Phase-3 pooled-vehicle population -- no
runtime computation needed, so they're just constants written in here.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import re

from BehaviorScreen.core import Stim, Laterality
from BehaviorScreen.point_process.dataset import BehavioralDataLoader
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory, PreyCapture
from BehaviorScreen.point_process.hawkes_process import HawkesProcess, HistoryKernelFactory
from BehaviorScreen.point_process.mixed_effects_process import GammaMixedEffectsProcess
from BehaviorScreen.point_process.survival_process import SurvivalProcess, SurvivalKernelFactory
from BehaviorScreen.point_process.io import save_fig


from BehaviorScreen.point_process.partially_fixed_process import PartiallyFixedProcess
from BehaviorScreen.ablation_screen.tier1_omnibus import (
    run_tier1_screen, 
    tier1_permutation_test, 
    fit_tier1, 
    extract_fitted_parameters, 
    compute_parameter_deltas, 
    plot_parameter_change_heatmaps,
    generate_arm_surface_grids,
    build_bad_fit_triage,
    plot_volcano,
    plot_fish_gain_correlation_vehicle_vs_drug
)
from BehaviorScreen.ablation_screen.fdr import add_fdr_per_behavior
from BehaviorScreen.ablation_screen.pvalue_diagnostics import plot_pvalue_histogram
from BehaviorScreen.ablation_screen.dataset_utils import subset_loader, safe_prepare_dataset
from BehaviorScreen.ablation_screen.dataset_ops import select_fish


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


OUTPUT_DIR = Path("./tier1_results")
OUTPUT_DIR.mkdir(exist_ok=True)

ROOT = Path('/media/martin/DATA_18TB/Screen')
loader = BehavioralDataLoader(ROOT / "bouts_all.csv")

prey_stim_speed_deg_per_s = 90
prey_stim_range_deg = 2 * 70
prey_stim_freq = prey_stim_speed_deg_per_s / prey_stim_range_deg

# ===========================================================================
# ONE dict per behavior. 'architecture' already returns a fully-formed,
# fitting-ready process -- PartiallyFixedProcess wraps the raw architecture
# and pins its shape parameters to values already known from the 
# pooled-vehicle fit. Whatever's NOT listed in the fixed dict stays free and
# gets estimated per line/condition arm.
# ===========================================================================
BEHAVIOR_CONFIG = {

    'prey_capture_ipsi': {
        'dataset': {
            'stim': Stim.PREY_CAPTURE, 'bout_name': 'JT',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 24.0,
        },
        # AIC winner (ΔAIC=0). Requires the hawkes_process.py per-trial
        # integral fix to be practical for a full per-line screen -- see
        # conversation notes; re-benchmark before running the full sweep.
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(
                HawkesProcess(
                    PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq),
                    HistoryKernelFactory.exponential(),
                )
            ),
            {
                'tau': 0.812041, 'z0_ripple': 0.889359,
                'phi1': 0.267287, 'phi2': -2.720154,
                'alpha_peak': -0.192392, 'alpha_baseline': -0.100840,
                'alpha_ripple': -0.230097,
                'alpha_hawkes': 0.266288, 'beta_hawkes': 0.628789,  # pinned as shape -- see caveat
                'r_dispersion': 6.236294,
            },
        ),
        # free: A, B
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'prey_capture_contra': {
        'dataset': {
            'stim': Stim.PREY_CAPTURE, 'bout_name': 'JT',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 24.0,
        },
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            {'r_dispersion': 2.733626},
        ),
        # free: B
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'phototaxis_ipsi': {
        'dataset': {
            'stim': Stim.PHOTOTAXIS, 'bout_name': 'RT',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 24.0,
        },
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.phototaxis_dip_exgaussian_peak())),
            {
                'f_dip': 0.330746, 'tau_dip': 0.429057, 'mu': 0.282779,
                'sigma': 0.069307, 'tau_decay': 0.175063,
                'alpha_B': 0.033644, 'alpha_peak': 0.046725,
                'r_dispersion': 2.273735,
            },
        ),
        # free: B, A_peak
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'phototaxis_contra': {
        'dataset': {
            'stim': Stim.PHOTOTAXIS, 'bout_name': 'RT',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 24.0,
        },
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.phototaxis_contra())),
            {
                'tau_dip': 0.870464,
                'alpha_B': 0.037437,
                'alpha_dip': -0.151756,
                'r_dispersion': 1.851101,
            },
        ),
        # free: B, z0_dip
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'omr_lateral_ipsi': {
        'dataset': {
            'epoch_name': ["grating right", "grating left"], 'bout_name': 'RT',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 9.0,
        },
        # AIC winner (ΔAIC=0). Homogeneous baseline -> cheap regardless of
        # the Hawkes integral fix (trivial kernel.integrate), so this one
        # was never the bottleneck.
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(
                HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
            ),
            {
                'alpha_hawkes': 0.037930, 'beta_hawkes': 0.330805,  # pinned as shape -- see caveat
                'r_dispersion': 2.485696,
            },
        ),
        # free: B
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'omr_lateral_contra': {
        'dataset': {
            'epoch_name': ["grating right", "grating left"], 'bout_name': 'RT',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 9.0,
        },
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.omr_lateral_contra())),
            {'tau_dip': 1.667727, 'r_dispersion': 0.761771},
        ),
        # free: B, f_dip
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'omr_forward': {
        'dataset': {
            'epoch_name': "grating forward", 'bout_name': 'BS',
            'laterality': Laterality.NONDIRECTIONAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 9.0,
        },
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.omr_forward())),
            {'tau_dip': 1.653539, 'r_dispersion': 0.443042},
        ),
        # free: B, z_dip
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'okr_ipsi': {
        'dataset': {
            'stim': Stim.OKR, 'bout_name': 'S1',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 9.0,
        },
        # CAVEAT: no Hawkes/Renewal candidate was ever fit for this condition
        # in Phase 3 -- best of what was TRIED, not a confirmed winner over
        # history-dependence.
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.homogeneous_poisson())),
            {'r_dispersion': 0.853827},
        ),
        # free: B
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'okr_contra': {
        'dataset': {
            'stim': Stim.OKR, 'bout_name': 'S1',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 9.0,
        },
        # AIC winner (ΔAIC=0; Renewal-ExponentialExcitation close second at
        # ΔAIC=13.08, AIC weight 0.14% -- Hawkes still clearly preferred).
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(
                HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
            ),
            {
                'alpha_hawkes': 0.038189, 'beta_hawkes': 0.452438,  # pinned as shape -- see caveat
                'r_dispersion': 1.698480,
            },
        ),
        # free: B
        'null': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    },

    'looming_ipsi': {
        'dataset': {
            'stim': Stim.LOOMING, 'bout_name': 'SLC',
            'laterality': Laterality.IPSILATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 9.0,
        },
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(
                SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4, 6)))
            ),
            {
                'mu': 4.952208, 'sigma': 0.102790, 'alpha': -0.120006,
                'r_dispersion': 2.845518, 'B': 0.0284841
            },
        ),
        # free: H
        'null': lambda: SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
    },

    'looming_contra': {
        'dataset': {
            'stim': Stim.LOOMING, 'bout_name': 'SLC',
            'laterality': Laterality.CONTRALATERAL,
            'binning_dt': 0.1, 't_start': 0.0, 't_end': 9.0,
        },
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(
                SurvivalProcess(SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4, 6)))
            ),
            {
                'mu': 4.955200, 'sigma': 0.101784, 'alpha': -0.139564,
                'r_dispersion': 2.910379, 'B': 0.0289052
            },
        ),
        # free: H
        'null': lambda: SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
    },

    'dark_flash': {
        'dataset': {
            'epoch_name': "flash dark", 'bout_name': 'O',
            'laterality': Laterality.NONDIRECTIONAL,
            'binning_dt': 0.025, 't_start': 0.0, 't_end': 5.0,
        },
        'architecture': lambda: PartiallyFixedProcess(
            GammaMixedEffectsProcess(SurvivalProcess(SurvivalKernelFactory.exgaussian_bump_baseline_habituating())),
            {
                'mu': 0.073930, 'sigma': 0.016163, 'tau': 0.078449,
                'alpha': -0.187736, 'r_dispersion': 2.549803, 'B': 0.0179795
            },
        ),
        # free: H
        'null': lambda: SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
    },
}

BEHAVIORS = list(BEHAVIOR_CONFIG.keys())
DATASET_CONFIGS = {b: cfg['dataset'] for b, cfg in BEHAVIOR_CONFIG.items()}
BEHAVIOR_PROCESS_FACTORY = {b: cfg['architecture'] for b, cfg in BEHAVIOR_CONFIG.items()}
NULL_PROCESS_FACTORY = {b: cfg['null'] for b, cfg in BEHAVIOR_CONFIG.items()}

print("Per-behavior free parameters (everything else pinned in BEHAVIOR_CONFIG):")
for behavior in BEHAVIORS:
    print(f"  {behavior}: free={BEHAVIOR_PROCESS_FACTORY[behavior]().param_names}")

all_lines = sorted(loader.raw_df["line"].astype(str).unique())
STRAIGHT_MUTANTS = {"lak", "gr", "mecp2", "cort", "wik", "AB"}
NTR_LINES = [l for l in all_lines if l not in STRAIGHT_MUTANTS and l != "WT"]
LINE_LABELS = {"WT": ("danieau", "ronidazole")}


# ===========================================================================
# Step 1: single-cell smoke test BEFORE the full batch
# ===========================================================================

def find_first_valid_cell(loader, lines, behaviors, dataset_configs, line_labels, default_labels):
    for line in lines:
        veh_label, drug_label = line_labels.get(line, default_labels)
        for behavior in behaviors:
            ds_veh = safe_prepare_dataset(subset_loader(loader, line, [veh_label]), dataset_configs[behavior])
            ds_drug = safe_prepare_dataset(subset_loader(loader, line, [drug_label]), dataset_configs[behavior])
            if ds_veh is not None and ds_drug is not None and ds_veh.num_fish >= 3 and ds_drug.num_fish >= 3:
                return line, behavior, ds_veh, ds_drug
    return None, None, None, None


print("\nSmoke test: scanning for first valid (line, behavior) cell...")
smoke_line, smoke_behavior, ds_veh, ds_drug = find_first_valid_cell(
    loader, NTR_LINES + ["WT"], BEHAVIORS, DATASET_CONFIGS, LINE_LABELS, ("vehicle", "ronidazole")
)
if smoke_line is None:
    raise RuntimeError(
        "No (line, behavior) cell has both non-empty vehicle and drug data with >=3 fish each. "
        "Check DATASET_CONFIGS filters against loader.raw_df contents before proceeding."
    )
print(f"Using {smoke_line}/{smoke_behavior} for smoke test "
      f"(n_veh={ds_veh.num_fish}, n_drug={ds_drug.num_fish})")
deviance, m_veh, m_drug, m_pooled = fit_tier1(BEHAVIOR_PROCESS_FACTORY[smoke_behavior], ds_veh, ds_drug)
print(f"Smoke test OK: deviance={deviance:.3f}, LL_veh={m_veh.log_likelihood:.2f}, "
      f"LL_drug={m_drug.log_likelihood:.2f}, free_params={m_veh.param_names}")


# ===========================================================================
# Step 2: full batch
# ===========================================================================
print("\nRunning Tier 1 across all lines x behaviors...")
tier1_df = run_tier1_screen(
    loader, lines=NTR_LINES + ["WT"], behaviors=BEHAVIORS,
    dataset_configs=DATASET_CONFIGS, base_process_factories=BEHAVIOR_PROCESS_FACTORY,
    null_process_factories=NULL_PROCESS_FACTORY, line_labels=LINE_LABELS, n_perm=1000,
)
tier1_df = add_fdr_per_behavior(tier1_df, alpha=0.05)
tier1_df.to_csv(OUTPUT_DIR / "tier1_results.csv", index=False)
print(tier1_df["status"].value_counts())

bad_fits = build_bad_fit_triage(tier1_df)

param_df = extract_fitted_parameters(
    tier1_df, loader, DATASET_CONFIGS, BEHAVIOR_PROCESS_FACTORY,
    line_labels=LINE_LABELS,
)
param_df = compute_parameter_deltas(param_df)

line_order = (
    tier1_df[tier1_df["significant"]]
    .groupby("line").size()
    .reindex(sorted(param_df["line"].unique()), fill_value=0)
    .sort_values(ascending=False).index.tolist()
)

fig, axes = plot_parameter_change_heatmaps(param_df, line_order=line_order)
save_fig(fig, OUTPUT_DIR, "tier1_parameter_change_heatmaps")

generate_arm_surface_grids(
    tier1_df, loader, DATASET_CONFIGS, BEHAVIOR_PROCESS_FACTORY,
    output_dir=OUTPUT_DIR / "arm_surface_grids", line_labels=LINE_LABELS
)

fig, axes = plot_volcano(tier1_df)
save_fig(fig, OUTPUT_DIR, "tier1_volcano_by_behavior")

for line in NTR_LINES + ["WT"]:
    fig = plot_fish_gain_correlation_vehicle_vs_drug(
        line, loader, DATASET_CONFIGS, BEHAVIOR_PROCESS_FACTORY, line_labels=LINE_LABELS,
    )
    if fig is None:
        continue
    save_fig(fig, OUTPUT_DIR / "fish_gain_correlations", line)
    plt.close(fig)

# ===========================================================================
# Step 3: calibration checkpoint -- p-value histograms per behavior
# ===========================================================================
print("\nGenerating p-value histograms per behavior...")
ok_df = tier1_df[tier1_df["status"] == "ok"]
for behavior in BEHAVIORS:
    sub = ok_df[ok_df["behavior"] == behavior]["p_value"]
    if len(sub) < 5:
        continue
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_pvalue_histogram(sub, ax=ax, title=f"Tier 1 p-values: {behavior}")
    fig.savefig(OUTPUT_DIR / f"pvalue_hist_{behavior}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Step 4: negative control -- WT split-half permutation
# ===========================================================================
print("\nRunning WT split-half negative control...")
for behavior in BEHAVIORS:
    ds_wt = safe_prepare_dataset(subset_loader(loader, "WT", ["danieau"]), DATASET_CONFIGS[behavior])
    if ds_wt is None or ds_wt.num_fish < 10:
        continue
    null_ps = []
    for seed in tqdm(range(500)):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(ds_wt.num_fish)
        half = ds_wt.num_fish // 2
        ds_a = select_fish(ds_wt, perm[:half])
        ds_b = select_fish(ds_wt, perm[half:])
        try:
            result = tier1_permutation_test(BEHAVIOR_PROCESS_FACTORY[behavior], ds_a, ds_b, n_perm=200)
            if not result["permutation_unreliable"]:
                null_ps.append(result["p_value"])
        except Exception:
            continue
    if null_ps:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_pvalue_histogram(np.array(null_ps), ax=ax, title=f"Negative control (WT split-half): {behavior}")
        fig.savefig(OUTPUT_DIR / f"negctrl_pvalue_hist_{behavior}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

print("\nDone. Inspect tier1_results.csv, pvalue_hist_*.png, and negctrl_pvalue_hist_*.png "
      "before proceeding to Tier 2.")