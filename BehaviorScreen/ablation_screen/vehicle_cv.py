"""
Mean and between-fish CV of each Tier 1 readout in the vehicle arms, the
counterpart of marques_cv.py's readouts.csv for the Marques et al. 2018 data.

Per vehicle arm (line, vehicle) and readout, from the datasets Tier 1 fits
(plus the SPLITS windows: phototaxis early, the first 1.5 s (dip and peak of the response), and late):

    strength   event rate over the usable trials (events/s) or, for escapes,
               the share of trials with an escape
    cv         SD / mean of the strength across fish, as observed
    frailty    the part of cv beyond counting noise (method of moments):
               c^2 = (Var - mean(noise_i)) / m^2, with noise_i = r_i / T_i for
               rates (Poisson) and p_i (1 - p_i) / k_i for escapes (binomial)
    direction  per fish, the share of events on the stimulus side (ipsi; away
               from looming), and the SD of its logit across fish

Fish per arm for 80% power, two-sided at --alpha:

    strength   2 z^2 sd^2 / log2(f)^2, sd = sqrt(log(1 + cv^2)) / log(2), cv observed
    direction  f-fold change of the odds of the stimulus side (Tier 1's D), whatever
               the bias: 2 z^2 sd_logit^2 / log(f)^2

The default alpha is the one the Marques table implies (z^2 ~ 18 at 80% power).
These counts leave out the arm and clutch variance of the power analysis.

Outputs (OUTPUT/):
    vehicle_cv_arms.csv  one row per readout and vehicle arm
    vehicle_cv.csv       one row per readout: medians over arms (within-arm,
                         comparable to Marques' single experiments), the CV of all
                         vehicle fish pooled (adds arm-to-arm differences) and fish
                         per arm from the median within-arm values

    python -m BehaviorScreen.ablation_screen.vehicle_cv ROOT OUTPUT [--alpha 0.00067]
"""
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from BehaviorScreen.core import Laterality
from BehaviorScreen.point_process.analysis.model_config import EVENT_TABLES, get_model_config
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader, PointProcessDataset

from .dataset_utils import safe_prepare_dataset
from .tier1_config import MAP_COLUMNS

ESCAPES = {"looming", "dark_flash", "looming_divergent_saccade"}  # survival readouts: share of trials with an escape
DIRECTION_SIDE = {"looming": Laterality.CONTRALATERAL}  # away; ipsi otherwise
FOLDS = (1.5, 2.0, 3.0)
MIN_FISH, MIN_DIRECTION_EVENTS = 5, 3
SPLITS = {"phototaxis": {"phototaxis_early": (0.0, 1.5), "phototaxis_late": (1.5, None)}}  # extra readouts: windows (s) within the trial
MARQUES_ALPHA = 0.05 / 75


def fish_strength(ds: PointProcessDataset, escape: bool) -> pd.DataFrame:
    """Per fish: strength, its counting-noise variance, events (escapes: trials with one) and trials."""
    trials = ds.fish_trial_mask.sum(axis=1)
    if escape:
        hits = np.bincount(ds.first_events().event_fish_idx, minlength=ds.num_fish)
        p = hits / trials
        return pd.DataFrame({"strength": p, "noise": p * (1 - p) / trials, "events": hits, "trials": trials})
    events = np.bincount(ds.event_fish_idx, minlength=ds.num_fish)
    exposure = trials * ds.duration_s
    return pd.DataFrame({"strength": events / exposure, "noise": events / exposure**2, "events": events, "trials": trials})


def strength_summary(fish: pd.DataFrame) -> dict:
    m, var = fish["strength"].mean(), fish["strength"].var()
    return {
        "n_fish": len(fish), "trials": fish["trials"].mean(), "events_per_fish": fish["events"].mean(),
        "strength_mean": m, "cv_strength": np.sqrt(var) / m,
        "cv_frailty": np.sqrt(max(var - fish["noise"].mean(), 0.0)) / m, "cv_counting": np.sqrt(fish["noise"].mean()) / m,
    }


def direction_summary(ds: PointProcessDataset, side: Laterality, first_only: bool) -> dict:
    ds = ds.first_events() if first_only else ds
    on_side = np.bincount(ds.event_fish_idx[ds.marks["laterality"] == int(side)], minlength=ds.num_fish)
    total = np.bincount(ds.event_fish_idx, minlength=ds.num_fish)
    keep = total >= MIN_DIRECTION_EVENTS
    if keep.sum() < MIN_FISH:
        return {}
    on_side, total = on_side[keep], total[keep]
    logit = np.log((on_side + 0.5) / (total - on_side + 0.5))
    return {"direction_fish": int(keep.sum()), "direction_mean": float(np.mean(on_side / total)), "sd_logit_direction": float(np.std(logit, ddof=1))}


def readouts() -> list[tuple[str, str, dict]]:
    """(readout, experiment, dataset arguments): each Tier 1 experiment, then its SPLITS windows."""
    config = get_model_config()
    result = []
    for experiment in MAP_COLUMNS:
        dataset = config[experiment]["dataset"]
        result.append((experiment, experiment, dataset))
        for name, (start, stop) in SPLITS.get(experiment, {}).items():
            result.append((name, experiment, {**dataset, "t_start": start, "t_end": dataset["t_end"] if stop is None else stop}))
    return result


def readout_row(ds: PointProcessDataset, experiment: str, two_sided: bool) -> dict:
    row = {"run_s": ds.duration_s, **strength_summary(fish_strength(ds, experiment in ESCAPES))}
    if two_sided:
        row |= direction_summary(ds, DIRECTION_SIDE.get(experiment, Laterality.IPSILATERAL), first_only=experiment in ESCAPES)
    return row


def add_fish_needed(table: pd.DataFrame, alpha: float, power: float = 0.8) -> pd.DataFrame:
    z2 = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) ** 2
    sd = np.sqrt(np.log1p(table["cv_strength"] ** 2)) / np.log(2)
    for f in FOLDS:
        table[f"fish_{f:g}x"] = np.ceil(2 * z2 * sd**2 / np.log2(f) ** 2)
    if "sd_logit_direction" in table:
        for f in FOLDS:
            table[f"fish_direction_odds_{f:g}x"] = np.ceil(2 * z2 * table["sd_logit_direction"] ** 2 / np.log(f) ** 2)
    return table


def vehicle_cv(loaders: Dict[str, BehavioralDataLoader], vehicle: str = "vehicle") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per arm, per readout) tables, before the fish counts."""
    config = get_model_config()
    arm_rows, summary_rows = [], []
    for readout, experiment, dataset in readouts():
        loader = loaders[config[experiment]["events"]]
        sides = {int(l) for l in np.atleast_1d(dataset["laterality"])} if dataset["laterality"] is not None else set()
        two_sided = {int(Laterality.IPSILATERAL), int(Laterality.CONTRALATERAL)} <= sides
        lines = sorted(loader.trials.loc[loader.trials["condition"] == vehicle, "line"].astype(str).unique())
        rows = []
        for line in lines:
            ds = safe_prepare_dataset(loader.subset(line, [vehicle]), dataset, f"{line}/{vehicle}/{readout}")
            if ds is not None and ds.num_fish >= MIN_FISH:
                rows.append({"readout": readout, "line": line, **readout_row(ds, experiment, two_sided)})
        if not rows:
            continue
        arms = pd.DataFrame(rows)
        arm_rows.append(arms)
        pooled = safe_prepare_dataset(loader.subset(None, [vehicle]), dataset, f"{vehicle}/{readout}")
        pooled_row = {} if pooled is None else strength_summary(fish_strength(pooled, experiment in ESCAPES))
        summary_rows.append({
            "readout": readout, "events": config[experiment]["events"], "arms": len(arms), "n_fish": int(arms["n_fish"].sum()),
            **arms.drop(columns=["readout", "line", "n_fish"]).median(numeric_only=True).to_dict(),
            "cv_strength_pooled": pooled_row.get("cv_strength", np.nan), "cv_frailty_pooled": pooled_row.get("cv_frailty", np.nan),
        })
    return pd.concat(arm_rows, ignore_index=True), pd.DataFrame(summary_rows)


def main(args: argparse.Namespace) -> None:
    loaders = {
        events: BehavioralDataLoader(args.root / csv, [args.root / args.valid_trials], [args.root / args.qc], **EVENT_TABLES[events])
        for events, csv in {"bouts": args.bouts, "saccades": args.saccades}.items()
    }
    arms, summary = vehicle_cv(loaders, args.vehicle)
    args.output.mkdir(parents=True, exist_ok=True)
    add_fish_needed(arms, args.alpha).to_csv(args.output / "vehicle_cv_arms.csv", index=False)
    summary = add_fish_needed(summary, args.alpha)
    summary.to_csv(args.output / "vehicle_cv.csv", index=False)
    columns = ["readout", "arms", "n_fish", "events_per_fish", "strength_mean", "cv_strength", "cv_frailty", "cv_strength_pooled", "fish_2x"]
    print(summary[[c for c in columns if c in summary]].round(3).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mean and between-fish CV of each readout in the vehicle arms (cf. marques_cv).")
    parser.add_argument("root", type=Path, help="Folder with the merged screen tables")
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=MARQUES_ALPHA, help="Two-sided test level of the fish counts (default: as in the Marques table)")
    parser.add_argument("--vehicle", default="vehicle", help="Condition of the vehicle arms")
    parser.add_argument("--bouts", default="bouts_all.csv")
    parser.add_argument("--saccades", default="saccades_augmented_all.csv")
    parser.add_argument("--valid-trials", default="valid_trials_all.csv")
    parser.add_argument("--qc", default="qc_all.csv")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
