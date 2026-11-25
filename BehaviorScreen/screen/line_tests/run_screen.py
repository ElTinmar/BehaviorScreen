"""
Tier 1 end-to-end run. Behaviors come from the control fits (run_analysis
--mode fit): each experiment's best model with every parameter pinned to
its control value except the amplitudes (tier1_config.AMPLITUDE_PARAMETERS),
which are fitted per arm. A contrast's tests pin to the control fit refitted
without its own arms (Arms.architecture). Contrasts: drug against vehicle within each NTR
line, and each straight mutant against its wild-type background. The
configuration lives in screen_setup.

--behavior runs one behavior (its cells, FDR, Tier 2, parameter tests, bootstraps and checks) into
<output>/<behavior>/, each cell saved as it is done; --combine merges those into the
screen's tables and maps. Without either, every behavior and then the combination.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

from BehaviorScreen.point_process.io import save_fig
from BehaviorScreen.point_process.parallel import limit_workers
from BehaviorScreen.screen.line_tests.tier1_omnibus import (
    run_tier1_screen,
    tier1_permutation_test,
    fit_tier1,
    extract_fitted_parameters,
    compute_parameter_deltas,
    save_tier1_maps,
    map_rows,
    add_testability,
    extend_permutations,
    generate_arm_surface_grids,
    build_bad_fit_triage,
    plot_volcano,
    bootstrap_parameter_deltas,
    plot_parameter_forest
)
from BehaviorScreen.screen.line_tests.fdr import add_fdr_per_behavior
from BehaviorScreen.screen.line_tests.pvalue_diagnostics import plot_pvalue_histogram
from BehaviorScreen.screen.line_tests.dataset_utils import Arms
from BehaviorScreen.screen.line_tests.screen_setup import load_screen
from BehaviorScreen.screen.line_tests.steps import finished, read_table, run_step
from BehaviorScreen.screen.line_tests.gain_structure import GainStructure, calibrate, fish_groups
from BehaviorScreen.screen.line_tests.parameter_tests import run_parameter_tests
# Model-free counterpart of this run (map, absolute values per arm): python -m BehaviorScreen.screen.line_tests.count_readouts
from BehaviorScreen.screen.line_tests.tier1_config import MAP_COLUMNS
from BehaviorScreen.screen.line_tests.tier2_interaction import plot_tier2_map, run_tier2_screen
from BehaviorScreen.screen.paths import CONTROL_FITS, LINE_TEST_TABLES, VARIANTS, add_data_root, data_root, line_tests_dir

N_PERM = 1000  # permutations per Tier 1 cell
N_PERM_MAX = 10_000  # for a cell not significant whose deviance few of them reached (extend_permutations)
N_SIM = 1000  # simulated datasets per NTR cell (p_value_plate)
N_BOOT_TIER2 = 200  # parametric bootstrap of each hit's interaction test
N_BOOT_PARAMETERS = 200  # parametric bootstrap of each of a hit's parameter tests
N_BOOT_DELTAS = 1000  # fish bootstrap of each hit's parameter deltas
NEGATIVE_CONTROL_SPLITS, NEGATIVE_CONTROL_PERM = 20, 200  # random halves of the WT fish, permutations of each


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
    for seed in tqdm(range(NEGATIVE_CONTROL_SPLITS), desc=f"Negative control {behavior}"):
        perm = np.random.default_rng(seed).permutation(ds_wt.num_fish)
        half = ds_wt.num_fish // 2
        try:
            result = tier1_permutation_test(arms.architecture(behavior, frozenset({("WT", "danieau")})), ds_wt.fish(perm[:half]), ds_wt.fish(perm[half:]), n_perm=NEGATIVE_CONTROL_PERM)
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


def run_behavior(behavior: str, arms: Arms, null_process_factories: dict, output_dir: Path, n_perm: int, n_sim: int, n_perm_max: int) -> None:
    """
    Everything done per behavior, into output_dir / behavior, the steps after the cells saved as they are done (steps).
    n_sim: datasets of each NTR cell's simulated null (0: none). n_perm_max: permutations of a cell extended (extend_permutations).
    """
    out = output_dir / behavior
    smoke_test(arms, behavior)
    structures = {behavior: gain_structure(behavior, arms, out)} if n_sim else None
    tier1_df = run_tier1_screen(arms, behaviors=[behavior], null_process_factories=null_process_factories, n_perm=n_perm,
                                store=output_dir / "cells", structures=structures, n_sim=n_sim)
    tier1_df = extend_permutations(add_fdr_per_behavior(tier1_df), arms, n_perm_max, store=output_dir / "cells")
    tier1_df = add_testability(add_fdr_per_behavior(tier1_df), n_perm_max)
    if "p_value_plate" in tier1_df:
        tier1_df = add_fdr_per_behavior(tier1_df, pval_col="p_value_plate", prefix="plate_")
    print(tier1_df["status"].value_counts())
    out.mkdir(parents=True, exist_ok=True)
    tier1_df.drop(columns=["arm_estimates", "fish_ids"], errors="ignore").to_csv(out / f"{LINE_TEST_TABLES[0]}.csv", index=False)
    step = lambda name, compute: run_step(output_dir, behavior, name, compute)
    param_df = step("tier1_parameters", lambda: compute_parameter_deltas(extract_fitted_parameters(tier1_df)))
    # Tier 2: is a hit's drug effect different from the drug's effect on WT?
    step("tier2_results", lambda: add_fdr_per_behavior(run_tier2_screen(tier1_df, arms, n_boot=N_BOOT_TIER2, reference="WT")))
    # Post hoc: which of a hit's amplitudes differ between its arms?
    step("parameter_tests", lambda: run_parameter_tests(tier1_df, arms, N_BOOT_PARAMETERS, (structures or {}).get(behavior)))
    step("arm_surface_grids", lambda: generate_arm_surface_grids(tier1_df, arms, output_dir=output_dir / "arm_surface_grids"))
    boot_delta_df = step("bootstrap_parameter_deltas", lambda: bootstrap_parameter_deltas(tier1_df, param_df, arms, n_boot=N_BOOT_DELTAS))

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
    missing = [b for b in behaviors if not finished(output_dir, b)]
    if missing:
        print(f"Not run yet (left out): {missing}")
    behaviors, map_order = [b for b in behaviors if b not in missing], [b for b in map_order if b not in missing]
    tables = {name: pd.concat([read_table(output_dir / b / f"{name}.csv") for b in behaviors], ignore_index=True) for name in LINE_TEST_TABLES}
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    tier1_df, param_df, tier2_df = tables["tier1_results"], tables["tier1_parameters"], tables["tier2_results"]
    build_bad_fit_triage(tier1_df).to_csv(output_dir / "bad_fit_triage.csv", index=False)

    rows = map_rows(tier1_df, arms.contrasts.values())
    save_tier1_maps(param_df, tier1_df, map_order, rows, output_dir, MAP_COLUMNS, tables["parameter_tests"])
    if (tier2_df["status"] == "ok").any():
        fig, _ = plot_tier2_map(param_df, tier2_df, map_order)
        save_fig(fig, output_dir, "tier2_map")
    fig, _ = plot_volcano(tier1_df[tier1_df["family"] == "ntr"])
    save_fig(fig, output_dir, "tier1_volcano_by_behavior")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--behavior", help="Run this behavior only.")
    parser.add_argument("--combine", action="store_true", help="Merge the behaviors already run into the screen's tables and maps.")
    add_data_root(parser)
    parser.add_argument("--fits-root", type=Path, default=CONTROL_FITS, help="run_analysis --mode fit output of the control screen.")
    parser.add_argument("--model", choices=VARIANTS, default="best", help="Model of each behavior (paths.VARIANTS).")
    parser.add_argument("--output-dir", type=Path, help="Results (default: line_tests/<variant>).")
    parser.add_argument("--n-perm", type=int, default=N_PERM, help="Permutations per Tier 1 cell.")
    parser.add_argument("--n-perm-max", type=int, default=N_PERM_MAX, help="Permutations a cell near the floor is extended to.")
    parser.add_argument("--n-sim", type=int, default=N_SIM, help="Simulated datasets per NTR cell for p_value_plate (0: none).")
    parser.add_argument("--n-jobs", type=int, help="Worker processes (default: one per allowed core).")
    args = parser.parse_args()
    if args.n_jobs:
        limit_workers(args.n_jobs)
    args.output_dir = args.output_dir or line_tests_dir(args.model)
    if args.behavior and not args.combine and finished(args.output_dir, args.behavior, merged_only=False):
        print(f"{args.behavior}: every step already saved in {args.output_dir}, nothing to run.")
        return
    screen = load_screen(data_root(args.data_root), args.fits_root, args.model)
    screen.arms.refit_store = args.output_dir / "control_refits"
    if args.behavior is not None and args.behavior not in screen.behaviors:
        raise SystemExit(f"--behavior must be one of {screen.behaviors}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.combine:
        for behavior in [args.behavior] if args.behavior else screen.behaviors:
            run_behavior(behavior, screen.arms, screen.null_process_factory, args.output_dir, args.n_perm, args.n_sim, args.n_perm_max)
    if args.combine or args.behavior is None:
        combine(screen.behaviors, screen.arms, screen.map_order, args.output_dir)
    print(f"\nDone. Inspect {args.output_dir}: tier1_results.csv, the p-value histograms and the negative controls.")


if __name__ == "__main__":
    main()
