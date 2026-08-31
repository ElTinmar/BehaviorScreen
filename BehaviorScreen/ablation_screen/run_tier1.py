"""
Tier 1 end-to-end run. Fill in DATASET_CONFIGS / BEHAVIOR_PROCESS_FACTORY /
NULL_PROCESS_FACTORY from your Phase-3 model_config winners before running.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from BehaviorScreen.core import Stim, Laterality

from BehaviorScreen.point_process.dataset import BehavioralDataLoader
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory, PreyCapture
from BehaviorScreen.point_process.mixed_effects_process import GammaMixedEffectsProcess
from BehaviorScreen.point_process.survival_process import SurvivalProcess, SurvivalKernelFactory

from BehaviorScreen.ablation_screen.tier1_omnibus import run_tier1_screen
from BehaviorScreen.ablation_screen.fdr import add_fdr
from BehaviorScreen.ablation_screen.pvalue_diagnostics import plot_pvalue_histogram
from BehaviorScreen.ablation_screen.dataset_utils import subset_loader, safe_prepare_dataset
from BehaviorScreen.ablation_screen.tier1_omnibus import tier1_permutation_test, fit_tier1
from BehaviorScreen.ablation_screen.dataset_ops import select_fish
import numpy as np

OUTPUT_DIR = Path("./tier1_results")
OUTPUT_DIR.mkdir(exist_ok=True)

loader = BehavioralDataLoader(Path("/home/martin/bouts_all.csv"))

prey_stim_speed_deg_per_s = 90
prey_stim_range_deg = 2 * 70
prey_stim_freq = prey_stim_speed_deg_per_s / prey_stim_range_deg

DATASET_CONFIGS = {

    'prey_capture_ipsi': {
        'stim': Stim.PREY_CAPTURE,
        'bout_name': 'JT',
        'laterality': Laterality.IPSILATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 24.0,
    },

    'prey_capture_contra': {
        'stim': Stim.PREY_CAPTURE,
        'bout_name': 'JT',
        'laterality': Laterality.CONTRALATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 24.0,
    },

    'phototaxis_ipsi': {
        'stim': Stim.PHOTOTAXIS,
        'bout_name': 'RT',
        'laterality': Laterality.IPSILATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 24.0,
    },

    'phototaxis_contra': {
        'stim': Stim.PHOTOTAXIS,
        'bout_name': 'RT',
        'laterality': Laterality.CONTRALATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 24.0,
    },

    'omr_lateral_ipsi': {
        'epoch_name': ["grating right", "grating left"],
        'bout_name': 'RT',
        'laterality': Laterality.IPSILATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 9.0,
    },

    'omr_lateral_contra': {
        'epoch_name': ["grating right", "grating left"],
        'bout_name': 'RT',
        'laterality': Laterality.CONTRALATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 9.0,
    },

    'omr_forward': {
        'epoch_name': "grating forward",
        'bout_name': 'BS',
        'laterality': Laterality.NONDIRECTIONAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 9.0,
    },

    'okr_ipsi': {
        'stim': Stim.OKR,
        'bout_name': 'S1',
        'laterality': Laterality.IPSILATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 9.0,
    },

    'okr_contra': {
        'stim': Stim.OKR,
        'bout_name': 'S1',
        'laterality': Laterality.CONTRALATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 9.0,
    },

    'looming_ipsi': {
        'stim': Stim.LOOMING,
        'bout_name': 'SLC',
        'laterality': Laterality.IPSILATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 9.0,
    },

    'looming_contra': {
        'stim': Stim.LOOMING,
        'bout_name': 'SLC',
        'laterality': Laterality.CONTRALATERAL,
        'binning_dt': 0.05,
        't_start': 0.0,
        't_end': 9.0,
    },

    'dark_flash': {
        'epoch_name': "flash dark",
        'bout_name': 'O',
        'laterality': Laterality.NONDIRECTIONAL,
        'binning_dt': 0.025,
        't_start': 0.0,
        't_end': 5.0,
    },
}

BEHAVIOR_PROCESS_FACTORY = {

    'prey_capture_ipsi': lambda: GammaMixedEffectsProcess(
        PoissonProcess(PreyCapture.peak_baseline_ripple(stim_freq=prey_stim_freq))
    ),

    'prey_capture_contra': lambda: GammaMixedEffectsProcess(
        PoissonProcess(RateKernelFactory.homogeneous_poisson())
    ),

    'phototaxis_ipsi': lambda: GammaMixedEffectsProcess(
        PoissonProcess(RateKernelFactory.phototaxis_dip_exgaussian_peak())
    ),

    'phototaxis_contra': lambda: GammaMixedEffectsProcess(
        PoissonProcess(RateKernelFactory.phototaxis_contra())
    ),

    'omr_lateral_ipsi': lambda: GammaMixedEffectsProcess(
        PoissonProcess(RateKernelFactory.homogeneous_poisson())
    ),

    'omr_lateral_contra': lambda: GammaMixedEffectsProcess(
        PoissonProcess(RateKernelFactory.omr_lateral_contra())
    ),

    'omr_forward': lambda: GammaMixedEffectsProcess(
        PoissonProcess(RateKernelFactory.omr_forward())
    ),

    'okr_ipsi': lambda: GammaMixedEffectsProcess(
        PoissonProcess(RateKernelFactory.homogeneous_poisson())
    ),

    'okr_contra': lambda: GammaMixedEffectsProcess(
        PoissonProcess(RateKernelFactory.homogeneous_poisson())
    ),

    'looming_ipsi': lambda: GammaMixedEffectsProcess(
        SurvivalProcess(
            SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4,6))
        )
    ),

    'looming_contra': lambda: GammaMixedEffectsProcess(
        SurvivalProcess(
            SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4,6))
        )
    ),

    'dark_flash': lambda: GammaMixedEffectsProcess(
        SurvivalProcess(
            SurvivalKernelFactory.exgaussian_bump_baseline_habituating()
        )
    ),
}

NULL_PROCESS_FACTORY = {
    'prey_capture_ipsi': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'prey_capture_contra': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'phototaxis_ipsi': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'phototaxis_contra': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'omr_lateral_ipsi': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'omr_lateral_contra': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'omr_forward': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'okr_ipsi': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'okr_contra': lambda: PoissonProcess(RateKernelFactory.homogeneous_poisson()),
    'looming_ipsi':  lambda: SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
    'looming_contra':  lambda: SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
    'dark_flash':  lambda: SurvivalProcess(SurvivalKernelFactory.constant_hazard()),
}

BEHAVIORS = list(DATASET_CONFIGS.keys())
all_lines = sorted(loader.raw_df["line"].astype(str).unique())
STRAIGHT_MUTANTS = {"lakritz", "gr", "mecp2", "cort", "wik", "AB"}
NTR_LINES = [l for l in all_lines if l not in STRAIGHT_MUTANTS and l != "WT"]
LINE_LABELS = {"WT": ("danieau", "ronidazole")}

# ===========================================================================
# Step 1: single-cell smoke test BEFORE the full batch
# ===========================================================================
print("Smoke test: one line, one behavior...")
smoke_line, smoke_behavior = NTR_LINES[0], BEHAVIORS[0]
ds_veh = safe_prepare_dataset(subset_loader(loader, smoke_line, ["vehicle"]), DATASET_CONFIGS[smoke_behavior])
ds_drug = safe_prepare_dataset(subset_loader(loader, smoke_line, ["ronidazole"]), DATASET_CONFIGS[smoke_behavior])
assert ds_veh is not None and ds_drug is not None, "Smoke test failed at data loading."
deviance, m_veh, m_drug, m_pooled = fit_tier1(BEHAVIOR_PROCESS_FACTORY[smoke_behavior], ds_veh, ds_drug)
print(f"Smoke test OK: deviance={deviance:.3f}, LL_veh={m_veh.log_likelihood:.2f}, LL_drug={m_drug.log_likelihood:.2f}")

# ===========================================================================
# Step 2: full batch
# ===========================================================================
print("\nRunning Tier 1 across all lines x behaviors...")
tier1_df = run_tier1_screen(
    loader, lines=NTR_LINES + ["WT"], behaviors=BEHAVIORS,
    dataset_configs=DATASET_CONFIGS, base_process_factories=BEHAVIOR_PROCESS_FACTORY,
    null_process_factories=NULL_PROCESS_FACTORY, line_labels=LINE_LABELS, n_perm=500,
)
tier1_df = add_fdr(tier1_df, alpha=0.05)
tier1_df.to_csv(OUTPUT_DIR / "tier1_results.csv", index=False)
print(tier1_df["status"].value_counts())

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
null_pvals_by_behavior = {}
for behavior in BEHAVIORS:
    ds_wt = safe_prepare_dataset(subset_loader(loader, "WT", ["danieau"]), DATASET_CONFIGS[behavior])
    if ds_wt is None or ds_wt.num_fish < 10:
        continue
    null_ps = []
    for seed in range(100):
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
        null_pvals_by_behavior[behavior] = np.array(null_ps)
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_pvalue_histogram(np.array(null_ps), ax=ax, title=f"Negative control (WT split-half): {behavior}")
        fig.savefig(OUTPUT_DIR / f"negctrl_pvalue_hist_{behavior}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

print("\nDone. Inspect tier1_results.csv, pvalue_hist_*.png, and negctrl_pvalue_hist_*.png "
      "before proceeding to Tier 2.")