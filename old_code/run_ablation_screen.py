from pathlib import Path
import pandas as pd
from BehaviorScreen.core import Stim, Laterality

from BehaviorScreen.point_process.dataset import BehavioralDataLoader
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory, PreyCapture
from BehaviorScreen.point_process.survival_process import SurvivalProcess, SurvivalKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalKernel, RenewalKernelFactory
from BehaviorScreen.point_process.hawkes_process import HawkesProcess, HistoryKernelFactory
from BehaviorScreen.point_process.mixed_effects_process import GammaMixedEffectsProcess

from BehaviorScreen.ablation_screen.tier0_screen import run_tier0_screen
from BehaviorScreen.ablation_screen.tier1_omnibus import run_tier1_screen
from BehaviorScreen.old_code.tier2_interaction import run_tier2_screen, build_wt_dataset_cache
from BehaviorScreen.old_code.tier3_localization import localize_effect
from BehaviorScreen.ablation_screen.fdr import add_fdr

OUTPUT_DIR = Path("./ablation_screen_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Frozen architecture per behavior -- fill in from your Phase-3 winners.
#    Must be zero-arg factories (fresh instance each call).
# ---------------------------------------------------------------------------

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

BEHAVIORS = list(DATASET_CONFIGS.keys())

# ---------------------------------------------------------------------------
# 2. Load data, define line list, condition-label conventions.
# ---------------------------------------------------------------------------
loader = BehavioralDataLoader(Path("/home/martin/bouts_all.csv"))  # full dataset incl. treated lines

all_lines = sorted(loader.raw_df["line"].astype(str).unique())
STRAIGHT_MUTANTS = {"lakritz", "gr", "mecp2", "cort"}  # adjust to your actual line names
NTR_LINES = [l for l in all_lines if l not in STRAIGHT_MUTANTS and l != "WT"]

LINE_LABELS = {"WT": ("danieau", "ronidazole")}  # everyone else defaults to (vehicle, ronidazole)

# ===========================================================================
# TIER 0 -- cheap fish-level screen (also run WT as its own "line" for
# the generic-drug background rate)
# ===========================================================================
print("Running Tier 0 (model-free fish-level screen)...")
tier0_df = run_tier0_screen(
    loader, lines=NTR_LINES + ["WT"], behaviors=BEHAVIORS,
    dataset_configs=DATASET_CONFIGS, line_labels=LINE_LABELS,
)
tier0_df = add_fdr(tier0_df, alpha=0.20)  # liberal, this is just a filter
tier0_df.to_csv(OUTPUT_DIR / "tier0_screen.csv", index=False)

tier0_candidates = (
    tier0_df[tier0_df["significant"]][["line", "behavior"]].drop_duplicates()
)
print(f"Tier 0: {len(tier0_candidates)} (line, behavior) candidates advance to Tier 1")

# ===========================================================================
# TIER 1 -- omnibus LR test (does ablation change anything)
# ===========================================================================
print("Running Tier 1 (omnibus LR test)...")
tier1_lines = sorted(tier0_candidates["line"].unique())
tier1_df = run_tier1_screen(
    loader, lines=tier1_lines, behaviors=BEHAVIORS,
    dataset_configs=DATASET_CONFIGS, base_process_factories=BEHAVIOR_PROCESS_FACTORY,
    line_labels=LINE_LABELS,
)
tier1_df = add_fdr(tier1_df, alpha=0.10)
tier1_df.to_csv(OUTPUT_DIR / "tier1_omnibus.csv", index=False)

tier1_hits = tier1_df[(tier1_df["status"] == "ok") & tier1_df["significant"]]
print(f"Tier 1: {len(tier1_hits)} hits advance to Tier 2")

# ===========================================================================
# TIER 2 -- interaction LR test (is it specific to this genotype)
# ===========================================================================
print("Running Tier 2 (interaction LR test)...")
wt_cache = build_wt_dataset_cache(loader, BEHAVIORS, DATASET_CONFIGS)
tier2_df = run_tier2_screen(
    candidates=tier1_hits, loader=loader, dataset_configs=DATASET_CONFIGS,
    base_process_factories=BEHAVIOR_PROCESS_FACTORY, wt_datasets_cache=wt_cache,
    line_labels=LINE_LABELS,
)
fitted_models = tier2_df.pop("_model_full")  # keep fitted MultiGroupProcess objects out of the CSV
tier2_df = add_fdr(tier2_df, alpha=0.05)
tier2_df.to_csv(OUTPUT_DIR / "tier2_interaction.csv", index=False)

tier2_hit_mask = (tier2_df["status"] == "ok") & tier2_df["significant"]
print(f"Tier 2: {tier2_hit_mask.sum()} confirmed ablation-specific hits")

# ===========================================================================
# TIER 3 -- which parameter changed
# ===========================================================================
print("Running Tier 3 (parameter localization)...")
tier3_tables = []
for i in tier2_df[tier2_hit_mask].index:
    row = tier2_df.loc[i]
    model_full = fitted_models.loc[i]
    result = localize_effect(model_full, n_boot=200)
    for table_name, table in result.items():
        table = table.copy()
        table.insert(0, "line", row["line"])
        table.insert(1, "behavior", row["behavior"])
        table.insert(2, "table", table_name)
        tier3_tables.append(table)

tier3_df = pd.concat(tier3_tables, ignore_index=True) if tier3_tables else pd.DataFrame()
tier3_df.to_csv(OUTPUT_DIR / "tier3_localization.csv", index=False)

print("Done. See ./ablation_screen_results/")