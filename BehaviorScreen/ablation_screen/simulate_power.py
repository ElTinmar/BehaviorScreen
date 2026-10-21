"""
Power of the Tier 1 model test against the count tests, by simulation.

For each behavior, a reference and a test arm are simulated from the control
fit (screen_setup's architecture, fish gains included) with the design (fish,
trials, mask) of a median-sized contrast. The test arm has one free amplitude
(B, A, A_peak, H or D) multiplied by a fold; fold 1 is the null, shared by all
amplitudes. On each simulated pair:

    model   deviance of fit_tier1, p from chi2 with one df per free amplitude,
            or from the fish-label permutation (--model-perm N, slow)
    counts  count_test of every count readout of that experiment

Power is the share of replicates with p < alpha. The fold-1 rows give each
test's false-positive rate. Both tests see fish-level variance only: arm and
clutch variance, which cost both alike, are not simulated.

Outputs (OUTPUT/): simulations.csv (one p per replicate and test),
power.csv (rate at alpha and at 0.05 with its binomial SE), power.png.

    python -m BehaviorScreen.ablation_screen.simulate_power OUTPUT
        [--behaviors looming okr] [--folds 0.5 0.67 1 1.5 2] [--reps 50]
        [--alpha 0.05/60] [--model-perm 0] [--jobs -1]
"""
import argparse
import copy
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from BehaviorScreen.point_process.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.point_process.io import save_fig
from BehaviorScreen.point_process.point_process.parallel import parallel_map
from BehaviorScreen.point_process.point_process.point_process import PointProcess

from .count_readouts import COUNT_READOUTS, count_test, fish_counts
from .tier1_config import PinnedArchitecture
from .tier1_omnibus import fit_tier1, tier1_permutation_test


@dataclass
class Job:
    behavior: str
    parameter: str  # "" at fold 1
    fold: float
    rep: int
    architecture: PinnedArchitecture
    control: PointProcess
    effect: PointProcess
    ds_ref: PointProcessDataset
    ds_test: PointProcessDataset
    model_perm: int
    count_perm: int


def control_model(architecture: PinnedArchitecture) -> PointProcess:
    model = copy.deepcopy(architecture.template)
    model.set_params(np.array(model.initial_guesses))
    return model


def with_fold(model: PointProcess, parameter: str, fold: float) -> PointProcess:
    effect = copy.deepcopy(model)
    effect.set_params(np.array([v * fold if n == parameter else v for n, v in zip(model.param_names, model.params_)]))
    return effect


def run_job(job: Job) -> List[dict]:
    rng = np.random.default_rng([job.rep, int(job.fold * 1000), zlib.crc32((job.behavior + job.parameter).encode())])
    ref, test = job.control.simulate_dataset(job.ds_ref, rng), job.effect.simulate_dataset(job.ds_test, rng)
    row = {"behavior": job.behavior, "parameter": job.parameter, "fold": job.fold, "rep": job.rep}
    try:
        deviance, *_ = fit_tier1(job.architecture, ref, test)
        df = len(job.architecture().param_names)
        p = (tier1_permutation_test(job.architecture, ref, test, deviance, job.model_perm, job.rep, n_jobs=1)["p_value"]
             if job.model_perm else stats.chi2.sf(deviance, df))
    except Exception:
        p = np.nan
    rows = [{**row, "test": "model", "p_value": p}]
    for name, r in COUNT_READOUTS.items():
        if r.experiment == job.behavior:
            p = count_test(fish_counts(ref, r), fish_counts(test, r), r.parameter, job.count_perm, rng)["p_value"]
            rows.append({**row, "test": name, "p_value": p})
    return rows


def design(arms, contrasts: Sequence, behavior: str) -> tuple[PointProcessDataset, PointProcessDataset]:
    """(reference, test) datasets of the contrast with the median reference size."""
    pairs = [arms.datasets(c.name, behavior) for c in contrasts]
    pairs = sorted((p for p in pairs if p[0] is not None and p[1] is not None), key=lambda p: p[0].num_fish)
    return pairs[len(pairs) // 2]


def make_jobs(behavior_config: Dict[str, dict], arms, contrasts: Sequence, folds: Sequence[float], args: argparse.Namespace) -> List[Job]:
    jobs = []
    for behavior in args.behaviors or list(behavior_config):
        cfg = behavior_config[behavior]
        control = control_model(cfg["architecture"])
        ds_ref, ds_test = design(arms, contrasts, behavior)
        print(f"{behavior}: {ds_ref.num_fish} vs {ds_test.num_fish} fish, free {cfg['free']}")
        effects = [("", 1.0)] + [(p, f) for p in cfg["free"] for f in folds if f != 1.0]
        for parameter, fold in effects:
            effect = with_fold(control, parameter, fold)
            jobs += [Job(behavior, parameter, fold, rep, cfg["architecture"], control, effect, ds_ref, ds_test, args.model_perm, args.count_perm)
                     for rep in range(args.reps)]
    return jobs


def power_table(sims: pd.DataFrame, alpha: float) -> pd.DataFrame:
    grouped = sims.dropna(subset=["p_value"]).groupby(["behavior", "parameter", "fold", "test"], sort=False)["p_value"]
    table = grouped.agg(reps="size", power=lambda p: np.mean(p < alpha), rate_05=lambda p: np.mean(p < 0.05)).reset_index()
    table["se"] = np.sqrt(table["power"] * (1 - table["power"]) / table["reps"])
    return table


def plot_power(table: pd.DataFrame, alpha: float) -> plt.Figure:
    null = table[table["fold"] == 1.0]
    panels = table[table["fold"] != 1.0][["behavior", "parameter"]].drop_duplicates().values.tolist()
    ncols = 4
    fig, axes = plt.subplots(-(-len(panels) // ncols), ncols, figsize=(3.2 * ncols, 2.6 * -(-len(panels) // ncols)), squeeze=False, sharey=True)
    for ax, (behavior, parameter) in zip(axes.flat, panels):
        panel = pd.concat([table[(table["behavior"] == behavior) & (table["parameter"] == parameter)], null[null["behavior"] == behavior]])
        for test, rows in panel.sort_values("fold").groupby("test", sort=False):
            ax.errorbar(rows["fold"], rows["power"], rows["se"], marker="o", ms=3, lw=1.5 if test == "model" else 1,
                        color="k" if test == "model" else None, label=test, capsize=0)
        ax.set_xscale("log")
        ax.set_xticks(sorted(panel["fold"].unique()), [f"{f:g}" for f in sorted(panel["fold"].unique())], minor=False)
        ax.minorticks_off()
        ax.axhline(0.8, color="0.7", ls=":", lw=0.8)
        ax.set_title(f"{behavior} · {parameter}", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"power at α = {alpha:.1e}")
    for ax in axes[-1]:
        ax.set_xlabel("fold")
    fig.tight_layout()
    return fig


def main(args: argparse.Namespace) -> None:
    from .screen_setup import BEHAVIOR_CONFIG, CONTRASTS, arms  # loads the event tables and control fits

    args.output.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(BEHAVIOR_CONFIG, arms, CONTRASTS, args.folds, args)
    sims = pd.DataFrame([row for rows in parallel_map(run_job, jobs, n_jobs=args.jobs, desc="Simulate") for row in rows])
    sims.to_csv(args.output / "simulations.csv", index=False)
    table = power_table(sims, args.alpha)
    table.to_csv(args.output / "power.csv", index=False)
    save_fig(plot_power(table, args.alpha), args.output, "power")
    print("False-positive rate at 0.05 (fold 1):")
    print(table[table["fold"] == 1.0][["behavior", "test", "reps", "rate_05"]].round(3).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulated power of the Tier 1 model test vs the count tests.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--behaviors", nargs="*", help="Experiments to simulate (default: all Tier 1 behaviors)")
    parser.add_argument("--folds", type=float, nargs="+", default=[0.5, 0.67, 1.0, 1.5, 2.0])
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05 / 60)
    parser.add_argument("--model-perm", type=int, default=0, help="Permutations of the model test; 0 uses chi2")
    parser.add_argument("--count-perm", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
