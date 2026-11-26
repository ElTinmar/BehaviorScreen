"""
Model-free Tier 1: readouts counted per fish on the datasets the models use
(model_config), compared between arms as in Tier 1 and written in its tables'
format, so the map, the absolute values per arm and the power analysis run on
them unchanged:

    R  rate: events per second over the experiment's window
    G  evoked gain: rate in the response window over the same fish's rate in the
       rest of the trial
    E  escape: share of trials with an event in the response window, as odds
    P  direction: share of events on the stimulus side (ipsi; away from a looming
       stimulus, first events only), as odds

Each arm pools its fish. As in Tier 1, a cell is one contrast in one experiment: its
readouts are tested together by fish-label permutation (count_test: max-T, the omnibus
p from the largest standardized difference of log values), one FDR family per
experiment (and contrast family); each readout's p, adjusted over the cell's readouts
on the same permutations, says which of a hit's readouts differ (parameter_tests.csv).
The arms of an NTR contrast keep the days recorded in both and their fish are permuted
within the day. Standard errors and per-arm intervals come from the fish bootstrap.

Rates count each trial by what the control fish do on it (trial_weights): responses
habituate over trials, so a fish missing late trials is not compared with the mean trial.

    python -m BehaviorScreen.screen.line_tests.count_readouts [--data-root DIR] [--output line_tests/model_free]

writes tier1_results.csv, tier1_parameters.csv, parameter_tests.csv, tier1_map.png, contrasts.csv,
arm_bootstrap.csv and tier1_absolute/; then
    python -m BehaviorScreen.screen.side_analysis.power line_tests/model_free --counts
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from BehaviorScreen.core import Laterality
from BehaviorScreen.screen.control_models.model_config import control_loader, get_model_config
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.screen.loader import BehavioralDataLoader

from BehaviorScreen.screen.paths import COUNT_RESULTS, add_data_root, data_root

from .dataset_utils import Contrast, recording_days, safe_prepare_dataset
from .fdr import ALPHA
from .model_qc import MIN_FISH
from .tier1_config import MAP_COLUMNS

IPSI, CONTRA = Laterality.IPSILATERAL, Laterality.CONTRALATERAL


@dataclass(frozen=True)
class CountReadout:
    experiment: str  # model_config experiment: its events table and dataset
    parameter: str  # "R", "G", "E" or "P"
    side: Laterality = IPSI  # P: the side counted
    first_only: bool = False  # P: first event of each trial only
    window: Tuple[float, float] = (0.0, 0.0)  # G, E: response window (s)


# Map columns in protocol order: each Tier 1 experiment's rate, then its transient, escape and direction readouts.
COUNT_READOUTS: Dict[str, CountReadout] = {
    "spont_dark": CountReadout("spont_dark", "R"),
    "spont_bright": CountReadout("spont_bright", "R"),
    "prey_capture": CountReadout("prey_capture", "R"),
    "prey_capture_gain": CountReadout("prey_capture", "G", window=(0.0, 5.0)),
    "prey_capture_direction": CountReadout("prey_capture", "P"),
    "prey_capture_convergent_saccade": CountReadout("prey_capture_convergent_saccade", "R"),
    "prey_capture_convergent_gain": CountReadout("prey_capture_convergent_saccade", "G", window=(0.0, 5.0)),
    "prey_capture_divergent_saccade": CountReadout("prey_capture_divergent_saccade", "R"),
    "phototaxis": CountReadout("phototaxis", "R"),
    "phototaxis_gain": CountReadout("phototaxis", "G", window=(0.0, 1.5)),
    "phototaxis_direction": CountReadout("phototaxis", "P"),
    "dark_flash": CountReadout("dark_flash", "E", window=(0.0, 1.0)),
    "omr_lateral": CountReadout("omr_lateral", "R"),
    "omr_lateral_direction": CountReadout("omr_lateral", "P"),
    "omr_forward": CountReadout("omr_forward", "R"),
    "okr": CountReadout("okr", "R"),
    "okr_direction": CountReadout("okr", "P"),
    "okr_conjugate_saccades": CountReadout("okr_conjugate_saccades", "R"),
    "okr_conjugate_direction": CountReadout("okr_conjugate_saccades", "P"),
    "looming": CountReadout("looming", "E", window=(4.0, 7.0)),
    "looming_away": CountReadout("looming", "P", side=CONTRA, first_only=True),
    "looming_divergent_saccade": CountReadout("looming_divergent_saccade", "E", window=(4.0, 7.0)),
    "after_looming": CountReadout("after_looming", "R"),
}
COUNT_COLUMNS = {r.experiment: MAP_COLUMNS.get(r.experiment, r.experiment) for r in COUNT_READOUTS.values()}  # map column labels
EVENT_COLUMNS = {"R": [0], "G": [0, 2], "E": [0], "P": [1]}  # columns of fish_counts holding events


def trial_weights(control: PointProcessDataset) -> np.ndarray:
    """Per trial, the control fish's mean events relative to their mean trial (1 for trials without control fish)."""
    n = control.fish_trial_mask.sum(axis=0)
    events = (control.event_counts_per_stream() * control.fish_trial_mask).sum(axis=0)
    profile = np.divide(events, n, out=np.full(len(n), np.nan), where=n > 0)
    return np.nan_to_num(profile / np.nanmean(profile), nan=1.0) if np.nansum(profile) > 0 else np.ones(len(n))


def fish_counts(ds: PointProcessDataset, r: CountReadout, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Per fish, the counts the readout pools: R (events, exposure), G (events in the
    window, its exposure, events outside it, their exposure), E (trials with an
    event in the window, trials), P (events on the side, events).
    weights (trial_weights): R's exposure counts each trial by its weight.
    """
    per_fish = lambda mask: np.bincount(ds.event_fish_idx[mask], minlength=ds.num_fish)
    trials = ds.fish_trial_mask.sum(axis=1)
    if r.parameter == "R":
        exposure = trials if weights is None else ds.fish_trial_mask @ weights
        return np.column_stack([per_fish(np.ones(len(ds.event_times), bool)), exposure * ds.duration_s])
    if r.parameter == "P":
        ds = ds.first_events() if r.first_only else ds
        return np.column_stack([per_fish(ds.marks["laterality"] == int(r.side)), per_fish(np.ones(len(ds.event_times), bool))])
    in_window = (ds.event_times >= r.window[0]) & (ds.event_times < r.window[1])
    if r.parameter == "E":
        streams = np.unique(np.column_stack([ds.event_fish_idx, ds.event_trials_idx])[in_window], axis=0)
        return np.column_stack([np.bincount(streams[:, 0], minlength=ds.num_fish), trials])
    length = r.window[1] - r.window[0]
    return np.column_stack([per_fish(in_window), trials * length, per_fish(~in_window), trials * (ds.duration_s - length)])


def log_value(counts: np.ndarray, parameter: str) -> np.ndarray:
    """Log of the readout pooled over fish (axis -2): log rate (R), log gain (G), log odds (P, E)."""
    s = counts.sum(axis=-2)
    if parameter == "R":
        return np.log((s[..., 0] + 0.5) / s[..., 1])
    if parameter == "G":
        return np.log((s[..., 0] + 0.5) / s[..., 1]) - np.log((s[..., 2] + 0.5) / s[..., 3])
    return np.log((s[..., 0] + 0.5) / (s[..., 1] - s[..., 0] + 0.5))


def bootstrap_logs(counts: np.ndarray, parameter: str, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return log_value(counts[rng.integers(0, len(counts), (n_boot, len(counts)))], parameter)


def count_test(
    ref: Dict[str, np.ndarray], test: Dict[str, np.ndarray], n_perm: int, rng: np.random.Generator, strata: Optional[np.ndarray] = None,
) -> dict:
    """
    An experiment's readouts (parameter: per-fish counts, the same fish in each) tested together, by max-T permutation
    (Westfall-Young): fish labels shuffled, every readout recomputed on the same shuffles, each difference of log values
    standardized by the SD of its own null. p_value: the omnibus test, from the largest |z|. readouts: per readout, its
    own p and its p adjusted step-down over the same shuffles (familywise error within the cell; with one readout, both
    are p_value). strata: per fish (reference then test), e.g. its recording day: fish are shuffled within it only.
    """
    names, n_ref = list(ref), len(next(iter(ref.values())))
    pooled = {k: np.vstack([ref[k], test[k]]) for k in names}
    strata = np.zeros(n_ref + len(next(iter(test.values()))), dtype=int) if strata is None else pd.factorize(strata)[0]
    perms, splits = np.empty((n_perm, len(strata)), dtype=int), 1
    for s in np.unique(strata):
        fish = np.flatnonzero(strata == s)
        perms[:, fish] = rng.permuted(np.tile(fish, (n_perm, 1)), axis=1)
        splits *= math.comb(len(fish), int(np.sum(fish < n_ref)))
    diff = lambda k, a, b: log_value(b, k) - log_value(a, k)
    observed = np.array([diff(k, ref[k], test[k]) for k in names])
    null = np.column_stack([diff(k, pooled[k][perms[:, :n_ref]], pooled[k][perms[:, n_ref:]]) for k in names])
    sd = null.std(axis=0)
    sd = np.where(sd > 0, sd, np.inf)
    z, z_null = np.abs(observed) / sd, np.abs(null) / sd
    p_of = lambda exceeding: (np.sum(exceeding) + 1) / (n_perm + 1)
    order = np.argsort(-z)
    adjusted = np.maximum.accumulate([p_of(z_null[:, order[rank:]].max(axis=1) >= z[k]) for rank, k in enumerate(order)])
    readouts = {names[k]: {"p_value": float(p_of(z_null[:, k] >= z[k])), "p_adjusted": float(a)} for k, a in zip(order, adjusted)}
    return {"p_value": float(adjusted[0]), "p_value_floor": 1.0 / (min(splits, n_perm) + 1), "readouts": readouts}


@dataclass
class ArmCounts:
    """One arm in one experiment: per fish (those with trials), its recording day, its events and each readout's counts."""

    days: np.ndarray
    events: np.ndarray
    readouts: Dict[str, np.ndarray]

    def fish(self, keep: np.ndarray) -> "ArmCounts":
        return ArmCounts(self.days[keep], self.events[keep], {k: c[keep] for k, c in self.readouts.items()})


EXPERIMENTS: Dict[str, List[CountReadout]] = {  # experiment: its readouts, in map order
    e: [r for r in COUNT_READOUTS.values() if r.experiment == e] for e in dict.fromkeys(r.experiment for r in COUNT_READOUTS.values())
}


def arm_counts(loaders: Dict[str, BehavioralDataLoader], contrasts: Sequence[Contrast]) -> Dict[Tuple[str, str, str], Optional[ArmCounts]]:
    """
    {(line, condition, experiment): its ArmCounts}, None where the arm has no dataset. The arms of an NTR contrast keep
    the days recorded in both.
    """
    config, counts = get_model_config(), {}
    arms = list(dict.fromkeys(arm for c in contrasts for arm in (c.reference, c.test)))
    for experiment, readouts in EXPERIMENTS.items():
        loader, dataset = loaders[config[experiment]["events"]], config[experiment]["dataset"]
        weights = trial_weights(control_loader(loader).prepare_dataset(**dataset))
        for arm in arms:
            ds = safe_prepare_dataset(loader.subset(arm[0], [arm[1]]), dataset, "/".join((*arm, experiment)))
            if ds is None:
                counts[(*arm, experiment)] = None
                continue
            keep = ds.fish_trial_mask.any(axis=1)
            counts[(*arm, experiment)] = ArmCounts(recording_days(ds.fish_ids), np.bincount(ds.event_fish_idx, minlength=ds.num_fish),
                                                   {r.parameter: fish_counts(ds, r, weights) for r in readouts}).fish(keep)
        for c in contrasts:
            ref, test = counts[(*c.reference, experiment)], counts[(*c.test, experiment)]
            if c.family == "ntr" and ref is not None and test is not None:
                counts[(*c.reference, experiment)] = ref.fish(np.isin(ref.days, test.days))
                counts[(*c.test, experiment)] = test.fish(np.isin(test.days, ref.days))
    return counts


def arm_bootstrap(counts: Dict[Tuple[str, str, str], Optional[ArmCounts]], n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Per arm, experiment and readout, in arm_absolute's format: pooled value and its 95% fish-bootstrap interval."""
    records = []
    for (line, condition, experiment), arm in counts.items():
        for parameter, c in ({} if arm is None or not len(arm.days) else arm.readouts).items():
            lo, hi = np.exp(np.percentile(bootstrap_logs(c, parameter, n_boot, rng), [2.5, 97.5]))
            records.append({"line": line, "condition": condition, "behavior": experiment, "parameter": parameter,
                            "value": float(np.exp(log_value(c, parameter))), "ci_lo": lo, "ci_hi": hi,
                            "n_boot": n_boot, "n_success": n_boot, "at_bound": False, "model": "counts"})
    return pd.DataFrame(records)


READOUT_COLUMNS = ["line", "behavior", "status", "parameter", "p_value", "p_adjusted", "significant"]


def run_count_screen(
    counts: Dict[Tuple[str, str, str], Optional[ArmCounts]], contrasts: Sequence[Contrast],
    n_perm: int = 10000, n_boot: int = 1000, min_events: int = 20, seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (Tier 1-style results, one row per contrast and experiment; parameter rows, one per readout; readout tests, one
    per readout of each tested cell, 'significant' left to add_readout_significance) for every contrast and experiment.
    NTR fish are permuted within their day.
    """
    rng = np.random.default_rng(seed)
    results: List[dict] = []
    params: List[dict] = []
    readout_tests: List[dict] = []
    for experiment, readouts in EXPERIMENTS.items():
        for c in contrasts:
            ref, test = counts.get((*c.reference, experiment)), counts.get((*c.test, experiment))
            row = {"family": c.family, "reference": "/".join(c.reference), "test": "/".join(c.test), "line": c.name, "behavior": experiment}
            if ref is None or test is None:
                results.append({**row, "status": "skipped_no_data", "p_value": np.nan})
                continue
            row |= {"n_fish_ref": len(ref.days), "n_fish_test": len(test.days), "total_events": int(ref.events.sum() + test.events.sum())}
            if min(len(ref.days), len(test.days)) < MIN_FISH or row["total_events"] < min_events:
                results.append({**row, "status": "skipped_insufficient_data", "p_value": np.nan})
                continue
            strata = np.concatenate([ref.days, test.days]) if c.family == "ntr" else None
            outcome = count_test(ref.readouts, test.readouts, n_perm, rng, strata)
            results.append({**row, "status": "ok", "p_value": outcome["p_value"], "p_value_floor": outcome["p_value_floor"]})
            readout_tests += [{"line": c.name, "behavior": experiment, "status": "ok", "parameter": k, **p} for k, p in outcome["readouts"].items()]
            for r in readouts:
                a, b = ref.readouts[r.parameter], test.readouts[r.parameter]
                v_ref, v_test = np.exp(log_value(a, r.parameter)), np.exp(log_value(b, r.parameter))
                se_ref, se_test = (float(np.std(bootstrap_logs(x, r.parameter, n_boot, rng))) for x in (a, b))
                params.append({"family": c.family, "line": c.name, "behavior": experiment, "parameter": r.parameter,
                               "value_reference": v_ref, "value_test": v_test, "se_reference": se_ref * v_ref, "se_test": se_test * v_test})
    return pd.DataFrame(results), pd.DataFrame(params), pd.DataFrame(readout_tests, columns=READOUT_COLUMNS[:-1])


def add_readout_significance(readout_tests: pd.DataFrame, tier1_df: pd.DataFrame) -> pd.DataFrame:
    """'significant': a readout whose adjusted p is under ALPHA, in a cell significant as a whole (post hoc of its hits)."""
    hits = set(tier1_df.loc[tier1_df["significant"], ["line", "behavior"]].itertuples(index=False, name=None))
    in_hit = np.array([cell in hits for cell in readout_tests[["line", "behavior"]].itertuples(index=False, name=None)], dtype=bool)
    return readout_tests.assign(significant=in_hit & (readout_tests["p_adjusted"] < ALPHA))[READOUT_COLUMNS]


def main(args: argparse.Namespace) -> None:
    from .arm_absolute import plot_all, save_contrasts
    from .fdr import add_fdr_per_behavior
    from .screen_setup import load_tables, screen_contrasts
    from .tier1_omnibus import add_testability, compute_parameter_deltas, save_tier1_maps

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    loaders = load_tables(data_root(args.data_root))
    contrasts = screen_contrasts(loaders)
    counts = arm_counts(loaders, contrasts)
    results, params, readout_tests = run_count_screen(counts, contrasts, args.n_perm, args.n_boot)
    param_df = compute_parameter_deltas(params)
    tier1_df = add_testability(add_fdr_per_behavior(results))
    tier1_df.to_csv(out / "tier1_results.csv", index=False)
    param_df.to_csv(out / "tier1_parameters.csv", index=False)
    posthoc = add_readout_significance(readout_tests, tier1_df)
    posthoc.to_csv(out / "parameter_tests.csv", index=False)
    save_contrasts(contrasts, out / "contrasts.csv")
    boot = arm_bootstrap(counts, args.n_boot, np.random.default_rng(1))
    boot.to_csv(out / "arm_bootstrap.csv", index=False)

    save_tier1_maps(param_df, tier1_df, list(EXPERIMENTS), contrasts, out, COUNT_COLUMNS, posthoc)
    plot_all(boot, contrasts, tier1_df, out / "tier1_absolute", COUNT_COLUMNS)
    print(tier1_df.groupby("behavior", sort=False)["significant"].sum().to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-free Tier 1: count readouts per arm, map, absolute values.")
    add_data_root(parser)
    parser.add_argument("--output", type=Path, default=COUNT_RESULTS)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
