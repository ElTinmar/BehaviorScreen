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

Each arm pools its fish; arms are compared by fish-label permutation of the log
value, standard errors and per-arm intervals by fish bootstrap. One FDR family
per readout (and contrast family).

    python -m BehaviorScreen.ablation_screen.count_readouts [--output ./count_results]

writes tier1_results.csv, tier1_parameters.csv, tier1_map.png, contrasts.csv,
arm_bootstrap.csv and tier1_absolute/; then
    python -m BehaviorScreen.ablation_screen.power ./count_results --counts
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from BehaviorScreen.core import Laterality
from BehaviorScreen.point_process.analysis.model_config import get_model_config
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader, PointProcessDataset
from BehaviorScreen.point_process.point_process.io import save_fig

from .dataset_utils import Contrast, safe_prepare_dataset
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
COUNT_COLUMNS = {name: MAP_COLUMNS[r.experiment] for name, r in COUNT_READOUTS.items()}  # map column labels
EVENT_COLUMNS = {"R": [0], "G": [0, 2], "E": [0], "P": [1]}  # columns of fish_counts holding events


def fish_counts(ds: PointProcessDataset, r: CountReadout) -> np.ndarray:
    """
    Per fish, the counts the readout pools: R (events, exposure), G (events in the
    window, its exposure, events outside it, their exposure), E (trials with an
    event in the window, trials), P (events on the side, events).
    """
    per_fish = lambda mask: np.bincount(ds.event_fish_idx[mask], minlength=ds.num_fish)
    trials = ds.fish_trial_mask.sum(axis=1)
    if r.parameter == "R":
        return np.column_stack([per_fish(np.ones(len(ds.event_times), bool)), trials * ds.duration_s])
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


def count_test(ref: np.ndarray, test: np.ndarray, parameter: str, n_perm: int, rng: np.random.Generator) -> dict:
    stat = lambda c: log_value(c, parameter)
    pooled, n_ref = np.vstack([ref, test]), len(ref)
    perms = rng.permuted(np.tile(np.arange(len(pooled)), (n_perm, 1)), axis=1)
    null = stat(pooled[perms[:, n_ref:]]) - stat(pooled[perms[:, :n_ref]])
    observed = stat(test) - stat(ref)
    return {
        "p_value": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)),
        "p_value_floor": 1.0 / (math.comb(len(pooled), n_ref) + 1),
    }


def arm_counts(
    loaders: Dict[str, BehavioralDataLoader], contrasts: Sequence[Contrast]
) -> Dict[Tuple[str, str, str], Optional[np.ndarray]]:
    """{(line, condition, readout): per-fish counts}, None where the arm has no dataset; each dataset prepared once."""
    config, datasets, counts = get_model_config(), {}, {}
    arms = list(dict.fromkeys(arm for c in contrasts for arm in (c.reference, c.test)))
    for name, r in COUNT_READOUTS.items():
        for arm in arms:
            key = (*arm, r.experiment)
            if key not in datasets:
                loader = loaders[config[r.experiment]["events"]].subset(arm[0], [arm[1]])
                datasets[key] = safe_prepare_dataset(loader, config[r.experiment]["dataset"], "/".join(key))
            ds = datasets[key]
            c = None if ds is None else fish_counts(ds, r)
            counts[(*arm, name)] = None if c is None else c[c[:, 1] > 0]  # P: fish with events; others: fish with trials
    return counts


def arm_bootstrap(counts: Dict[Tuple[str, str, str], Optional[np.ndarray]], n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Per arm and readout, in arm_absolute's format: pooled value and its 95% fish-bootstrap interval."""
    records = []
    for (line, condition, name), c in counts.items():
        if c is None or len(c) == 0:
            continue
        parameter = COUNT_READOUTS[name].parameter
        lo, hi = np.exp(np.percentile(bootstrap_logs(c, parameter, n_boot, rng), [2.5, 97.5]))
        records.append({"line": line, "condition": condition, "behavior": name, "parameter": parameter,
                        "value": float(np.exp(log_value(c, parameter))), "ci_lo": lo, "ci_hi": hi,
                        "n_boot": n_boot, "n_success": n_boot, "at_bound": False, "model": "counts"})
    return pd.DataFrame(records)


def run_count_screen(
    counts: Dict[Tuple[str, str, str], Optional[np.ndarray]], contrasts: Sequence[Contrast],
    n_perm: int = 10000, n_boot: int = 1000, min_fish: int = 6, min_events: int = 20, seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(Tier 1-style results, parameter rows) for every contrast and count readout."""
    rng = np.random.default_rng(seed)
    results: List[dict] = []
    params: List[dict] = []
    for name, r in COUNT_READOUTS.items():
        for c in contrasts:
            ref, test = counts.get((*c.reference, name)), counts.get((*c.test, name))
            row = {"family": c.family, "reference": "/".join(c.reference), "test": "/".join(c.test), "line": c.name, "behavior": name}
            if ref is None or test is None:
                results.append({**row, "status": "skipped_no_data", "p_value": np.nan})
                continue
            row |= {"n_fish_ref": len(ref), "n_fish_test": len(test),
                   "total_events": int(sum(a[:, EVENT_COLUMNS[r.parameter]].sum() for a in (ref, test)))}
            if min(len(ref), len(test)) < min_fish or row["total_events"] < min_events:
                results.append({**row, "status": "skipped_insufficient_data", "p_value": np.nan})
                continue
            results.append({**row, "status": "ok", **count_test(ref, test, r.parameter, n_perm, rng)})
            v_ref, v_test = np.exp(log_value(ref, r.parameter)), np.exp(log_value(test, r.parameter))
            se_ref, se_test = (float(np.std(bootstrap_logs(a, r.parameter, n_boot, rng))) for a in (ref, test))
            params.append({"family": c.family, "line": c.name, "behavior": name, "parameter": r.parameter,
                           "value_reference": v_ref, "value_test": v_test,
                           "se_reference": se_ref * v_ref, "se_test": se_test * v_test})
    return pd.DataFrame(results), pd.DataFrame(params)


def main(args: argparse.Namespace) -> None:
    from .arm_absolute import plot_all, save_contrasts
    from .fdr import add_fdr_per_behavior
    from .screen_setup import CONTRASTS, loaders  # loads the event tables
    from .tier1_omnibus import add_effect_size, add_testability, compute_parameter_deltas, map_rows, plot_tier1_map

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    counts = arm_counts(loaders, CONTRASTS)
    results, params = run_count_screen(counts, CONTRASTS, args.n_perm, args.n_boot)
    param_df = compute_parameter_deltas(params)
    tier1_df = add_effect_size(add_testability(add_fdr_per_behavior(results, alpha=0.05)), param_df)
    tier1_df.to_csv(out / "tier1_results.csv", index=False)
    param_df.to_csv(out / "tier1_parameters.csv", index=False)
    save_contrasts(list(CONTRASTS), out / "contrasts.csv")
    boot = arm_bootstrap(counts, args.n_boot, np.random.default_rng(1))
    boot.to_csv(out / "arm_bootstrap.csv", index=False)

    rows = map_rows(tier1_df, CONTRASTS)
    fig, _ = plot_tier1_map(param_df, tier1_df, list(COUNT_READOUTS), rows, column_labels=COUNT_COLUMNS)
    save_fig(fig, out, "tier1_map")
    plot_all(boot, list(CONTRASTS), tier1_df, out / "tier1_absolute", COUNT_COLUMNS)
    print(tier1_df.groupby("behavior", sort=False)["significant"].sum().to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-free Tier 1: count readouts per arm, map, absolute values.")
    parser.add_argument("--output", type=Path, default=Path("./count_results"))
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
