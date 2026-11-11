"""
Tier 1 end-to-end run. Behaviors come from the control fits (run_analysis
--mode fit): each experiment's best model with every parameter pinned to
its control value except the amplitudes (tier1_config.AMPLITUDE_PARAMETERS),
which are fitted per arm. A contrast's tests pin to the control fit refitted
without its own arms (Arms.architecture). Contrasts: drug against vehicle within each NTR
line, and each straight mutant against its wild-type background. The
configuration lives in screen_setup.

--behavior runs one behavior (its cells, FDR, Tier 2, bootstraps and checks) into
<output>/<behavior>/, each cell saved as it is done; --combine merges those into the
screen's tables and maps. Without either, every behavior and then the combination.
"""
import argparse
import json
import os
from pathlib import Path

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
from BehaviorScreen.ablation_screen.gain_structure import GainStructure, calibrate, fish_groups
# Model-free counterpart of this run (map, absolute values per arm): python -m BehaviorScreen.ablation_screen.count_readouts
from BehaviorScreen.ablation_screen.tier1_config import MAP_COLUMNS
from BehaviorScreen.ablation_screen.tier2_interaction import plot_tier2_map, run_tier2_screen

TABLES = ("tier1_results", "tier1_parameters", "tier2_results", "bootstrap_parameter_deltas")


def read_table(path: Path) -> pd.DataFrame:
    """A saved table; empty if it had no columns (a behavior without hits has no bootstrap rows)."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def smoke_test(arms: Arms, behavior: str) -> None:
    """One Tier 1 fit of the behavior's first cell with >= 3 fish per arm, before its batch."""
    for name in arms.contrasts:
        ds_ref, ds_test = arms.datasets(name, behavior)
        if ds_ref is not None and ds_test is not None and ds_ref.num_fish >= 3 and ds_test.num_fish >= 3:
            deviance, m_ref, m_test, _ = fit_tier1(arms.process_factory(name, behavior), ds_ref, ds_test)
            print(f"Smoke test {name}/{behavior} (n_ref={ds_ref.num_fish}, n_test={ds_test.num_fish}): deviance={deviance:.3f}, "
                  f"LL_ref={m_ref.log_likelihood:.2f}, LL_test={m_test.log_likelihood:.2f}, free_params={m_ref.param_names}")
            return
    raise RuntimeError(f"No contrast has both arms with >=3 fish for {behavior}: check its dataset filters against the event tables.")


def negative_control(arms: Arms, behavior: str, out: Path) -> None:
    """WT split in random halves: p-values of a contrast without an effect."""
    ds_wt = arms.dataset("WT", "danieau", behavior)
    if ds_wt is None or ds_wt.num_fish < 10:
        return
    null_ps = []
    for seed in tqdm(range(20), desc=f"Negative control {behavior}"):
        perm = np.random.default_rng(seed).permutation(ds_wt.num_fish)
        half = ds_wt.num_fish // 2
        try:
            result = tier1_permutation_test(arms.architecture(behavior, frozenset({("WT", "danieau")})), ds_wt.fish(perm[:half]), ds_wt.fish(perm[half:]), n_perm=200)
            if not result["permutation_unreliable"]:
                null_ps.append(result["p_value"])
        except Exception:
            continue
    if null_ps:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_pvalue_histogram(np.array(null_ps), ax=ax, title=f"Negative control (WT split-half): {behavior}")
        save_fig(fig, out, f"negctrl_pvalue_hist_{behavior}")
        plt.close(fig)


def gain_structure(behavior: str, arms: Arms, out: Path) -> GainStructure:
    """
    The dependence between the control fish of a line, a day and a plate, calibrated on the behavior's control
    fit and saved in out (gain_structure.json, gain_structure.csv); reused while the control fit is the same.
    """
    model = arms.process_factories[behavior].fitted
    params, path = [float(p) for p in model.params_], out / "gain_structure.json"
    if path.exists() and (saved := json.loads(path.read_text()))["params"] == params:
        return GainStructure(**saved["shares"])
    dataset, control_arms = arms.control(behavior)
    structure, table = calibrate(model, dataset, fish_groups(dataset.fish_ids, [line for line, _ in control_arms]))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "gain_structure.csv", index=False)
    path.write_text(json.dumps({"params": params, "shares": vars(structure)}))
    print(f"Dependence between the control fish ({behavior}):\n{table.round(3).to_string(index=False)}")
    return structure


def run_behavior(behavior: str, arms: Arms, null_process_factories: dict, output_dir: Path, n_perm: int, n_sim: int = 500) -> None:
    """Everything done per behavior, into output_dir / behavior. n_sim: datasets of each NTR cell's simulated null (0: none)."""
    out = output_dir / behavior
    smoke_test(arms, behavior)
    structures = {behavior: gain_structure(behavior, arms, out)} if n_sim else None
    tier1_df = run_tier1_screen(arms, behaviors=[behavior], null_process_factories=null_process_factories, n_perm=n_perm,
                                store=output_dir / "cells", structures=structures, n_sim=n_sim)
    tier1_df = add_testability(add_fdr_per_behavior(tier1_df, alpha=0.05))
    if "p_value_plate" in tier1_df:
        tier1_df = add_fdr_per_behavior(tier1_df, alpha=0.05, pval_col="p_value_plate", prefix="plate_")
    print(tier1_df["status"].value_counts())
    param_df = compute_parameter_deltas(extract_fitted_parameters(tier1_df, arms))

    # Tier 2: is a hit's drug effect different from the drug's effect on WT?
    tier2_df = add_fdr_per_behavior(run_tier2_screen(tier1_df, arms, reference="WT", n_boot=200))
    generate_arm_surface_grids(tier1_df, arms, output_dir=output_dir / "arm_surface_grids")
    boot_delta_df = bootstrap_parameter_deltas(tier1_df, param_df, arms, n_boot=1000)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in zip(TABLES, (tier1_df, param_df, tier2_df, boot_delta_df)):
        df.to_csv(out / f"{name}.csv", index=False)

    fig = plot_parameter_forest(boot_delta_df, behavior) if len(boot_delta_df) else None
    if fig is not None:
        save_fig(fig, output_dir / "parameter_forest_plots", behavior)
        plt.close(fig)

    # Calibration checkpoint: the p-values of the behavior's NTR contrasts
    ok = tier1_df[(tier1_df["status"] == "ok") & (tier1_df["family"] == "ntr")]["p_value"]
    if len(ok) >= 5:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_pvalue_histogram(ok, ax=ax, title=f"Tier 1 p-values: {behavior}")
        save_fig(fig, out, f"pvalue_hist_{behavior}")
        plt.close(fig)
    negative_control(arms, behavior, out)


def combine(behaviors: list, arms: Arms, map_order: list, output_dir: Path) -> None:
    """The behaviors' tables merged (output_dir/<table>.csv), and the maps and plots across behaviors."""
    missing = [b for b in behaviors if not (output_dir / b / f"{TABLES[0]}.csv").exists()]
    if missing:
        print(f"Not run yet (left out): {missing}")
    behaviors, map_order = [b for b in behaviors if b not in missing], [b for b in map_order if b not in missing]
    tables = {name: pd.concat([read_table(output_dir / b / f"{name}.csv") for b in behaviors], ignore_index=True) for name in TABLES}
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    tier1_df, param_df, tier2_df = tables["tier1_results"], tables["tier1_parameters"], tables["tier2_results"]
    build_bad_fit_triage(tier1_df).to_csv(output_dir / "bad_fit_triage.csv", index=False)

    rows = map_rows(tier1_df, arms.contrasts.values())
    # A mutant and its reference are two clutches: its fish-label test is not framed (clutch_variability frames it).
    siblings = tier1_df.assign(significant=tier1_df["significant"] & (tier1_df["family"] == "ntr"))
    fig, _ = plot_tier1_map(param_df, siblings, map_order, rows, column_labels=MAP_COLUMNS, frame_label="significant (q < 0.05; NTR lines)")
    save_fig(fig, output_dir, "tier1_map")
    if (tier2_df["status"] == "ok").any():
        fig, _ = plot_tier2_map(param_df, tier2_df, map_order)
        save_fig(fig, output_dir, "tier2_map")
    fig, _ = plot_volcano(tier1_df[tier1_df["family"] == "ntr"])
    save_fig(fig, output_dir, "tier1_volcano_by_behavior")
    for line in arms.contrasts:
        fig = plot_fish_gain_correlation_reference_vs_test(line, arms)
        if fig is not None:
            save_fig(fig, output_dir / "fish_gain_correlations", line)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--behavior", help="Run this behavior only.")
    parser.add_argument("--combine", action="store_true", help="Merge the behaviors already run into the screen's tables and maps.")
    parser.add_argument("--data-root", type=Path, help="Folder of the merged event, trial and QC tables (default: screen_setup's).")
    parser.add_argument("--fits-root", type=Path, help="run_analysis --mode fit output of the control screen (default: ./figures).")
    parser.add_argument("--output-dir", type=Path, help="Results (default: ./tier1_results).")
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutations per Tier 1 cell.")
    parser.add_argument("--n-sim", type=int, default=1000, help="Simulated datasets per NTR cell for p_value_plate (0: none).")
    args = parser.parse_args()
    for name, value in (("SCREEN_DATA_ROOT", args.data_root), ("SCREEN_FITS_ROOT", args.fits_root), ("SCREEN_OUTPUT_DIR", args.output_dir)):
        if value is not None:
            os.environ[name] = str(value)
    from BehaviorScreen.ablation_screen.screen_setup import BEHAVIORS, MAP_ORDER, NULL_PROCESS_FACTORY, OUTPUT_DIR, arms  # loads the tables

    if args.behavior is not None and args.behavior not in BEHAVIORS:
        raise SystemExit(f"--behavior must be one of {BEHAVIORS}")
    if not args.combine:
        for behavior in [args.behavior] if args.behavior else BEHAVIORS:
            run_behavior(behavior, arms, NULL_PROCESS_FACTORY, OUTPUT_DIR, args.n_perm, args.n_sim)
    if args.combine or args.behavior is None:
        combine(BEHAVIORS, arms, MAP_ORDER, OUTPUT_DIR)
    print(f"\nDone. Inspect {OUTPUT_DIR}: tier1_results.csv, the p-value histograms and the negative controls.")


if __name__ == "__main__":
    main()
