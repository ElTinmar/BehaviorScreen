"""
Tier 1 end-to-end run. Behaviors come from the control fits (run_analysis
--mode fit): each experiment's chosen model with every parameter pinned to
its control value except the amplitudes (tier1_config.AMPLITUDE_PARAMETERS),
which are fitted per arm. Each model must be one of its experiment's control
candidates (matched by name). Contrasts: drug against vehicle within each NTR
line, and each straight mutant against its wild-type background. The
configuration lives in screen_setup.
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

from BehaviorScreen.point_process.point_process.io import save_fig
from BehaviorScreen.ablation_screen.tier1_omnibus import (
    run_tier1_screen,
    tier1_permutation_test,
    fit_tier1,
    extract_fitted_parameters,
    compute_parameter_deltas,
    plot_tier1_map,
    map_rows,
    add_testability,
    add_effect_size,
    generate_arm_surface_grids,
    build_bad_fit_triage,
    plot_volcano,
    plot_fish_gain_correlation_reference_vs_test,
    bootstrap_parameter_deltas,
    plot_parameter_forest
)
from BehaviorScreen.ablation_screen.fdr import add_fdr_per_behavior
from BehaviorScreen.ablation_screen.pvalue_diagnostics import plot_pvalue_histogram
from BehaviorScreen.ablation_screen.dataset_utils import Arms
# Model-free counterpart of this run (map, absolute values per arm): python -m BehaviorScreen.ablation_screen.count_readouts
from BehaviorScreen.ablation_screen.tier1_config import MAP_COLUMNS
from BehaviorScreen.ablation_screen.tier2_interaction import plot_tier2_map, run_tier2_screen
from BehaviorScreen.ablation_screen.screen_setup import (
    BEHAVIOR_PROCESS_FACTORY, BEHAVIORS, MAP_ORDER, NULL_PROCESS_FACTORY, OUTPUT_DIR, arms,
)


# ===========================================================================
# Step 1: single-cell smoke test BEFORE the full batch
# ===========================================================================

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
        "Check DATASET_CONFIGS filters against the event tables before proceeding."
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
tier1_df = add_testability(add_fdr_per_behavior(tier1_df, alpha=0.05))
print(tier1_df["status"].value_counts())

# Count readouts (direction, escape, evoked gain): model-free, tested alongside Tier 1 (own FDR family per readout).
# Off for now: the map shows only the model amplitudes. To restore, swap the commented lines back in here and below.
# count_df, count_params = run_count_screen(arm_counts(loaders, CONTRASTS), CONTRASTS)
# count_df = add_testability(add_fdr_per_behavior(count_df, alpha=0.05))
# screen_df = pd.concat([tier1_df, count_df], ignore_index=True)
screen_df = tier1_df

bad_fits = build_bad_fit_triage(tier1_df)

param_df = extract_fitted_parameters(tier1_df, arms)
param_df = compute_parameter_deltas(param_df)
# screen_params = compute_parameter_deltas(pd.concat([param_df, count_params], ignore_index=True))
screen_params = param_df
screen_df = add_effect_size(screen_df, param_df)  # min fold change of a hit: tier1_omnibus.MIN_FOLD
screen_df.to_csv(OUTPUT_DIR / "tier1_results.csv", index=False)

rows = map_rows(screen_df, arms.contrasts.values())
fig, _ = plot_tier1_map(screen_params, screen_df, MAP_ORDER, rows, column_labels=MAP_COLUMNS)  # + list(COUNT_READOUTS)
save_fig(fig, OUTPUT_DIR, "tier1_map")
screen_params.to_csv(OUTPUT_DIR / "tier1_parameters.csv", index=False)

# Tier 2: is a hit's drug effect different from the drug's effect on WT?
tier2_df = add_fdr_per_behavior(run_tier2_screen(tier1_df, arms, reference="WT", n_boot=200))
tier2_df.to_csv(OUTPUT_DIR / "tier2_results.csv", index=False)
if (tier2_df["status"] == "ok").any():
    fig, _ = plot_tier2_map(param_df, tier2_df, MAP_ORDER)
    save_fig(fig, OUTPUT_DIR, "tier2_map")

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
    for seed in tqdm(range(20)):
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