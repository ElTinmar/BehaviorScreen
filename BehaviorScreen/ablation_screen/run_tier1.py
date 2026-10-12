"""
Tier 1 end-to-end run. Behaviors come from the control fits (run_analysis
--mode fit): each experiment's chosen model with every parameter pinned to
its control value except the amplitudes (model_config.AMPLITUDE_PARAMETERS),
which are fitted per arm. Contrasts: drug against vehicle within each NTR
line, and each straight mutant against its wild-type background.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader
from BehaviorScreen.point_process.analysis.model_config import EVENT_TABLES
from BehaviorScreen.point_process.point_process.io import save_fig
from BehaviorScreen.ablation_screen.tier1_omnibus import (
    run_tier1_screen,
    tier1_permutation_test,
    fit_tier1,
    extract_fitted_parameters,
    compute_parameter_deltas,
    plot_tier1_map,
    generate_arm_surface_grids,
    build_bad_fit_triage,
    plot_volcano,
    plot_fish_gain_correlation_reference_vs_test,
    bootstrap_parameter_deltas,
    plot_parameter_forest
)
from BehaviorScreen.ablation_screen.fdr import add_fdr_per_behavior
from BehaviorScreen.ablation_screen.pvalue_diagnostics import plot_pvalue_histogram
from BehaviorScreen.ablation_screen.dataset_utils import Arms, mutant_contrast, ntr_contrast
from BehaviorScreen.ablation_screen.tier1_config import tier1_behaviors


OUTPUT_DIR = Path("./tier1_results")
OUTPUT_DIR.mkdir(exist_ok=True)

ROOT = Path('/media/martin/DATA_18TB/Screen')
FITS_ROOT = Path("./figures")  # run_analysis --mode fit output on the control screen
# Tables merged across screen folders with merge_csv (adds line/condition columns)
loader = BehavioralDataLoader(
    ROOT / "bouts_all.csv", [ROOT / "valid_trials_all.csv"], [ROOT / "qc_all.csv"], **EVENT_TABLES["bouts"]
)

TIER1_EXPERIMENTS = [
    "prey_capture_ipsi", "prey_capture_contra",
    "phototaxis_ipsi", "phototaxis_contra",
    "omr_lateral_ipsi", "omr_lateral_contra", "omr_forward",
    "okr_ipsi", "okr_contra",
    "looming_ipsi", "looming_contra",
    "dark_flash",
]
BEHAVIOR_CONFIG = tier1_behaviors(TIER1_EXPERIMENTS, FITS_ROOT)
BEHAVIORS = list(BEHAVIOR_CONFIG)
DATASET_CONFIGS = {b: cfg['dataset'] for b, cfg in BEHAVIOR_CONFIG.items()}
BEHAVIOR_PROCESS_FACTORY = {b: cfg['architecture'] for b, cfg in BEHAVIOR_CONFIG.items()}
NULL_PROCESS_FACTORY = {b: cfg['null'] for b, cfg in BEHAVIOR_CONFIG.items()}

print("Per-behavior model and free parameters (everything else pinned to the control fit):")
for behavior, cfg in BEHAVIOR_CONFIG.items():
    print(f"  {behavior}: {cfg['model_name']}, free={cfg['free']}")

# Straight mutants against their wild-type background. wik and AB measure how
# far background (and recording day) alone moves each behavior from WT (TL).
WT = ("WT", "danieau")
MUTANT_REFERENCES = {
    "wik": WT,
    "AB": WT,
    "lak": WT,              # atoh7 th241, TL (Tuebingen)
    "gr": WT,               # nr3c1 s357, TL (Muto et al. 2005)
    "blu": WT,              # vglut2a tz257, TL/Tuebingen (Smear et al. 2007)
    "cort": WT,             # background not published: assumed TL
    "mecp2": ("AB", "danieau"),  # outcrossed to AB (Pietri et al. 2013)
}
all_lines = sorted(loader.trials["line"].astype(str).unique())
NTR_LINES = [l for l in all_lines if l not in MUTANT_REFERENCES and l != "WT"]
CONTRASTS = (
    [ntr_contrast("WT", vehicle="danieau")]
    + [ntr_contrast(line) for line in NTR_LINES]
    + [mutant_contrast(line, reference) for line, reference in MUTANT_REFERENCES.items() if line in all_lines]
)


# ===========================================================================
# Step 1: single-cell smoke test BEFORE the full batch
# ===========================================================================

arms = Arms(loader, DATASET_CONFIGS, BEHAVIOR_PROCESS_FACTORY, CONTRASTS)


def find_first_valid_cell(arms: Arms, behaviors):
    for name in arms.contrasts:
        for behavior in behaviors:
            ds_ref, ds_test = arms.datasets(name, behavior)
            if ds_ref is not None and ds_test is not None and ds_ref.num_fish >= 3 and ds_test.num_fish >= 3:
                return name, behavior, ds_ref, ds_test
    return None, None, None, None


print("\nSmoke test: scanning for first valid (line, behavior) cell...")
smoke_line, smoke_behavior, ds_ref, ds_test = find_first_valid_cell(arms, BEHAVIORS)
if smoke_line is None:
    raise RuntimeError(
        "No (contrast, behavior) cell has both arms with >=3 fish. "
        "Check DATASET_CONFIGS filters against loader.raw_df contents before proceeding."
    )
print(f"Using {smoke_line}/{smoke_behavior} for smoke test "
      f"(n_ref={ds_ref.num_fish}, n_test={ds_test.num_fish})")
deviance, m_ref, m_test, m_pooled = fit_tier1(BEHAVIOR_PROCESS_FACTORY[smoke_behavior], ds_ref, ds_test)
print(f"Smoke test OK: deviance={deviance:.3f}, LL_ref={m_ref.log_likelihood:.2f}, "
      f"LL_test={m_test.log_likelihood:.2f}, free_params={m_ref.param_names}")


# ===========================================================================
# Step 2: full batch
# ===========================================================================
print("\nRunning Tier 1 across all lines x behaviors...")
tier1_df = run_tier1_screen(
    arms, behaviors=BEHAVIORS, null_process_factories=NULL_PROCESS_FACTORY, n_perm=1000,
)
tier1_df = add_fdr_per_behavior(tier1_df, alpha=0.05)
tier1_df.to_csv(OUTPUT_DIR / "tier1_results.csv", index=False)
print(tier1_df["status"].value_counts())

bad_fits = build_bad_fit_triage(tier1_df)

param_df = extract_fitted_parameters(tier1_df, arms)
param_df = compute_parameter_deltas(param_df)

hits = tier1_df[tier1_df["significant"]].groupby("line").size()
ntr_rows = sorted(NTR_LINES, key=lambda line: -hits.get(line, 0))
rows = ["WT"] + ntr_rows + [line for line in MUTANT_REFERENCES if line in arms.contrasts]
fig, _ = plot_tier1_map(param_df, tier1_df, BEHAVIORS, rows)
save_fig(fig, OUTPUT_DIR, "tier1_map")
param_df.to_csv(OUTPUT_DIR / "tier1_parameters.csv", index=False)

generate_arm_surface_grids(tier1_df, arms, output_dir=OUTPUT_DIR / "arm_surface_grids")

fig, axes = plot_volcano(tier1_df[tier1_df["family"] == "ntr"])
save_fig(fig, OUTPUT_DIR, "tier1_volcano_by_behavior")

for line in arms.contrasts:
    fig = plot_fish_gain_correlation_reference_vs_test(line, arms)
    if fig is None:
        continue
    save_fig(fig, OUTPUT_DIR / "fish_gain_correlations", line)
    plt.close(fig)

boot_delta_df = bootstrap_parameter_deltas(
    tier1_df, param_df, arms, n_boot=1000,
)
boot_delta_df.to_csv(OUTPUT_DIR / "bootstrap_parameter_deltas.csv", index=False)

for behavior in boot_delta_df["behavior"].unique():
    fig = plot_parameter_forest(boot_delta_df, behavior)
    if fig is None:
        continue
    save_fig(fig, OUTPUT_DIR / "parameter_forest_plots", behavior)
    plt.close(fig)

# ===========================================================================
# Step 3: calibration checkpoint -- p-value histograms per behavior
# ===========================================================================
print("\nGenerating p-value histograms per behavior...")
ok_df = tier1_df[(tier1_df["status"] == "ok") & (tier1_df["family"] == "ntr")]
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
    ds_wt = arms.dataset("WT", "danieau", behavior)
    if ds_wt is None or ds_wt.num_fish < 10:
        continue
    null_ps = []
    for seed in tqdm(range(500)):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(ds_wt.num_fish)
        half = ds_wt.num_fish // 2
        ds_a = ds_wt.fish(perm[:half])
        ds_b = ds_wt.fish(perm[half:])
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