"""
Plate variance of the count readouts. Fish are recorded four to a plate and a plate holds one arm, so the
fish of a plate share its time of day, rig and handling: if that makes them alike, fish are not exchangeable
between arms and Tier 1's fish-label permutation is too liberal.

Per readout, on each fish's own log value (count_readouts), within blocks of one arm on one day (plates of
different days also differ by clutch), for the control arms and the drug arms:

    icc                  share of the variance between fish that plates explain (one-way ANOVA pooled over
                         blocks; negative: plates differ less than chance), 95% interval from resampling blocks
    p_value              fish permuted between the plates of their block: exact test of no plate effect
    false_positive_rate  of a fish-label test at alpha = 0.05 with that icc (and with icc_hi):
                         2 Phi(-1.96 / sqrt(design effect)), design effect = 1 + (fish per plate - 1) icc

    python -m BehaviorScreen.ablation_screen.plate_variance [--output ./plate_variance]

writes fish_values.csv and plate_variance.csv.
"""
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from BehaviorScreen.point_process.analysis.model_config import get_model_config
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader

from .count_readouts import COUNT_READOUTS, fish_counts, log_value
from .dataset_utils import CONTROL_CONDITIONS, safe_prepare_dataset

PLATE = r"(?P<plate>.*_(?P<day>\d{2}_[A-Za-z]{3}_\d{4})_\d{2}h\d{2}min\d{2}sec)_fish_\d+$"


def fish_values(loaders: Dict[str, BehavioralDataLoader]) -> pd.DataFrame:
    """Each fish's log value of every count readout, with its arm and, from its file name, its day and plate."""
    config, datasets, tables = get_model_config(), {}, []
    arms = loaders["bouts"].trials[["line", "condition"]].astype(str).drop_duplicates()
    for line, condition in arms.itertuples(index=False, name=None):
        for name, r in COUNT_READOUTS.items():
            key = (line, condition, r.experiment)
            if key not in datasets:
                loader = loaders[config[r.experiment]["events"]].subset(line, [condition])
                datasets[key] = safe_prepare_dataset(loader, config[r.experiment]["dataset"], "/".join(key))
            if datasets[key] is None:
                continue
            counts = fish_counts(datasets[key], r)
            keep = counts[:, 1] > 0  # P: fish with events; others: fish with trials
            tables.append(pd.DataFrame({"readout": name, "line": line, "condition": condition, "file": datasets[key].fish_ids[keep],
                                        "value": log_value(counts[keep][:, None, :], r.parameter)}))
    values = pd.concat(tables, ignore_index=True)
    names = values["file"].str.extract(PLATE)
    if names["plate"].isna().any():
        raise ValueError(f"no plate in the file names {values.loc[names['plate'].isna(), 'file'].unique()[:5]}")
    return values.join(names)


def block_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Per block: sums of squares within and between its plates, fish, plates and sum of squared plate sizes."""
    within = (df["value"] - df.groupby("plate")["value"].transform("mean")) ** 2
    total = (df["value"] - df.groupby("block")["value"].transform("mean")) ** 2
    sizes = df.groupby(["block", "plate"]).size()
    return pd.DataFrame({
        "ssw": within.groupby(df["block"]).sum(), "ssb": (total - within).groupby(df["block"]).sum(),
        "fish": sizes.groupby("block").sum(), "plates": sizes.groupby("block").size(), "squares": (sizes**2).groupby("block").sum(),
    })


def icc(sums: pd.DataFrame) -> Tuple[float, float]:
    """(intraclass correlation of plates, fish per plate) from block_sums."""
    fish, between = sums["fish"].sum(), sums["plates"].sum() - len(sums)
    msb, msw = sums["ssb"].sum() / between, sums["ssw"].sum() / (fish - sums["plates"].sum())
    per_plate = (fish - (sums["squares"] / sums["fish"]).sum()) / between
    return (msb - msw) / (msb + (per_plate - 1) * msw), per_plate


def permutation_p(df: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> float:
    """P(between-plate variance this large) with fish permuted between the plates of their block."""
    df = df.sort_values("block")
    block, plate, x = pd.factorize(df["block"])[0], pd.factorize(df["plate"])[0], df["value"].to_numpy()
    size = np.bincount(plate)
    between = lambda v: np.sum(np.bincount(plate, weights=v) ** 2 / size)
    null = np.array([between(x[np.lexsort((rng.random(len(x)), block))]) for _ in range(n_perm)])
    return float((np.sum(null >= between(x) * (1 - 1e-12)) + 1) / (n_perm + 1))


def false_positive_rate(rho: float, per_plate: float, alpha: float = 0.05) -> float:
    return float(2 * stats.norm.sf(stats.norm.isf(alpha / 2) / np.sqrt(1 + (per_plate - 1) * max(rho, 0.0))))


def plate_variance(values: pd.DataFrame, n_perm: int = 10000, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per readout and kind of arm (control, drug), from the blocks with several plates."""
    rng = np.random.default_rng(seed)
    values = values.assign(block=values["line"] + "/" + values["condition"] + "/" + values["day"],
                           arms=np.where(values["condition"].isin(CONTROL_CONDITIONS), "control", "drug"))
    rows = []
    for (readout, arms), df in values.groupby(["readout", "arms"], sort=False):
        df = df[df.groupby("block")["plate"].transform("nunique") > 1]
        sums = block_sums(df)
        if len(sums) < 2 or sums["fish"].sum() == sums["plates"].sum():
            continue
        rho, per_plate = icc(sums)
        boot = [icc(sums.iloc[rng.integers(0, len(sums), len(sums))])[0] for _ in range(n_boot)]
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append({
            "readout": readout, "arms": arms, "blocks": len(sums), "plates": sums["plates"].sum(), "fish": sums["fish"].sum(),
            "fish_per_plate": per_plate, "icc": rho, "icc_lo": lo, "icc_hi": hi, "p_value": permutation_p(df, n_perm, rng),
            "false_positive_rate": false_positive_rate(rho, per_plate), "false_positive_rate_hi": false_positive_rate(hi, per_plate),
        })
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    from .screen_setup import loaders  # loads the event tables

    args.output.mkdir(parents=True, exist_ok=True)
    values = fish_values(loaders)
    values.to_csv(args.output / "fish_values.csv", index=False)
    table = plate_variance(values, args.n_perm, args.n_boot)
    table.to_csv(args.output / "plate_variance.csv", index=False)
    print(table.round(3).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plate variance of the count readouts within arm and day.")
    parser.add_argument("--output", type=Path, default=Path("./plate_variance"))
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=2000)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
