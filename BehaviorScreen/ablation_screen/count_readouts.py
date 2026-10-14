"""
Model-free readouts counted per fish and compared between arms:

    P  direction: share of a behavior's events going one way (ipsi, or away
       from a looming stimulus), as odds
    E  escape: share of trials with an escape in the response window, as odds
    G  evoked gain: event rate in the response window over the same fish's
       rate in the rest of the trial

Each arm pools its fish; arms are compared by fish-label permutation of the
log value, standard errors by fish bootstrap. Rows match the Tier 1 tables
(positive values), so they join FDR, the map, the clutch report and the
power analysis.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from BehaviorScreen.core import Laterality, Stim
from BehaviorScreen.point_process.analysis.model_config import (
    FLASH_ESCAPE, LATERAL_TURNS, LOOMING_ESCAPE, NONDIRECTIONAL, OKR_SWIMS, PHOTOTAXIS_FILTER, dataset,
)
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader, PointProcessDataset

from .dataset_utils import Contrast, safe_prepare_dataset

IPSI, CONTRA = Laterality.IPSILATERAL, Laterality.CONTRALATERAL
BOTH = [IPSI, CONTRA]


@dataclass(frozen=True)
class CountReadout:
    dataset: dict
    parameter: str  # "P", "E" or "G"
    events: str = "bouts"
    side: Optional[Laterality] = None  # P: the side counted
    first_only: bool = False  # P: first event of each trial only
    window: Tuple[float, float] = (0.0, 0.0)  # E, G: response window (s)


LOOMING = dataset(LOOMING_ESCAPE, BOTH, 9.0, stim=Stim.LOOMING)
PREY = dict(t_end=24.0, stim=Stim.PREY_CAPTURE)
PHOTOTAXIS = dataset("RT", BOTH, 24.0, stim=Stim.PHOTOTAXIS, trial_filters=PHOTOTAXIS_FILTER)

COUNT_READOUTS: Dict[str, CountReadout] = {
    "looming_away": CountReadout(LOOMING, "P", side=CONTRA, first_only=True),
    "omr_lateral_ipsi_share": CountReadout(dataset(LATERAL_TURNS, BOTH, 9.0, epoch_name=["grating right", "grating left"]), "P", side=IPSI),
    "okr_ipsi_share": CountReadout(dataset(OKR_SWIMS, BOTH, 9.0, stim=Stim.OKR), "P", side=IPSI),
    "phototaxis_ipsi_share": CountReadout(PHOTOTAXIS, "P", side=IPSI),
    "prey_capture_ipsi_share": CountReadout(dataset("JT", BOTH, **PREY), "P", side=IPSI),
    "dark_flash_escape": CountReadout(dataset(FLASH_ESCAPE, NONDIRECTIONAL, 5.0, epoch_name="flash dark"), "E", window=(0.0, 1.0)),
    "looming_escape": CountReadout(LOOMING, "E", window=(4.0, 7.0)),
    "prey_capture_gain": CountReadout(dataset("JT", BOTH, **PREY), "G", window=(0.0, 5.0)),
    "prey_capture_convergent_gain": CountReadout(dataset("Convergent", NONDIRECTIONAL, **PREY), "G", events="saccades", window=(0.0, 5.0)),
    "phototaxis_gain": CountReadout(PHOTOTAXIS, "G", window=(0.0, 2.0)),
}


def fish_counts(ds: PointProcessDataset, r: CountReadout) -> np.ndarray:
    """
    Per fish, the counts the readout pools: P (events on the side, events),
    E (trials with an event in the window, trials), G (events in the window,
    its exposure, events outside it, their exposure).
    """
    per_fish = lambda mask: np.bincount(ds.event_fish_idx[mask], minlength=ds.num_fish)
    if r.parameter == "P":
        ds = ds.first_events() if r.first_only else ds
        return np.column_stack([per_fish(ds.marks["laterality"] == int(r.side)), per_fish(np.ones(len(ds.event_times), bool))])
    in_window = (ds.event_times >= r.window[0]) & (ds.event_times < r.window[1])
    trials = ds.fish_trial_mask.sum(axis=1)
    if r.parameter == "E":
        streams = np.unique(np.column_stack([ds.event_fish_idx, ds.event_trials_idx])[in_window], axis=0)
        return np.column_stack([np.bincount(streams[:, 0], minlength=ds.num_fish), trials])
    length = r.window[1] - r.window[0]
    return np.column_stack([per_fish(in_window), trials * length, per_fish(~in_window), trials * (ds.duration_s - length)])


def log_value(counts: np.ndarray, parameter: str) -> np.ndarray:
    """Log of the readout pooled over fish (axis -2): log odds for P and E, log gain for G."""
    s = counts.sum(axis=-2)
    if parameter == "G":
        return np.log((s[..., 0] + 0.5) / s[..., 1]) - np.log((s[..., 2] + 0.5) / s[..., 3])
    return np.log((s[..., 0] + 0.5) / (s[..., 1] - s[..., 0] + 0.5))


def count_test(ref: np.ndarray, test: np.ndarray, parameter: str, n_perm: int, n_boot: int, rng: np.random.Generator) -> dict:
    stat = lambda c: log_value(c, parameter)
    pooled, n_ref = np.vstack([ref, test]), len(ref)
    perms = rng.permuted(np.tile(np.arange(len(pooled)), (n_perm, 1)), axis=1)
    null = stat(pooled[perms[:, n_ref:]]) - stat(pooled[perms[:, :n_ref]])
    observed = stat(test) - stat(ref)
    boot_se = lambda c: float(np.std(stat(c[rng.integers(0, len(c), (n_boot, len(c)))])))
    return {
        "p_value": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)),
        "p_value_floor": 1.0 / (math.comb(len(pooled), n_ref) + 1),
        "log_ref": float(stat(ref)), "log_test": float(stat(test)),
        "se_ref": boot_se(ref), "se_test": boot_se(test),
    }


EVENT_COLUMNS = {"P": [1], "E": [0], "G": [0, 2]}  # columns of fish_counts holding events


def run_count_screen(
    loaders: Dict[str, BehavioralDataLoader], contrasts: Sequence[Contrast],
    n_perm: int = 10000, n_boot: int = 1000, min_fish: int = 6, min_events: int = 20, seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(Tier 1-style results, parameter rows) for every contrast and count readout."""
    rng = np.random.default_rng(seed)
    results: List[dict] = []
    params: List[dict] = []
    for behavior, r in COUNT_READOUTS.items():
        counts = {}
        for c in contrasts:
            for arm in (c.reference, c.test):
                if arm not in counts:
                    ds = safe_prepare_dataset(loaders[r.events].subset(arm[0], [arm[1]]), r.dataset, f"{'/'.join(arm)}/{behavior}")
                    counts[arm] = None if ds is None else fish_counts(ds, r)
            ref, test = counts[c.reference], counts[c.test]
            if ref is None or test is None:
                continue
            ref, test = ref[ref[:, 1] > 0], test[test[:, 1] > 0]  # P: fish with events; E, G: fish with trials
            row = {"family": c.family, "reference": "/".join(c.reference), "test": "/".join(c.test), "line": c.name,
                   "behavior": behavior, "n_fish_ref": len(ref), "n_fish_test": len(test),
                   "total_events": int(sum(a[:, EVENT_COLUMNS[r.parameter]].sum() for a in (ref, test)))}
            if min(len(ref), len(test)) < min_fish or row["total_events"] < min_events:
                results.append({**row, "status": "skipped_insufficient_data", "p_value": np.nan})
                continue
            result = count_test(ref, test, r.parameter, n_perm, n_boot, rng)
            results.append({**row, "status": "ok", "p_value": result["p_value"], "p_value_floor": result["p_value_floor"]})
            v_ref, v_test = np.exp(result["log_ref"]), np.exp(result["log_test"])
            params.append({"family": c.family, "line": c.name, "behavior": behavior, "parameter": r.parameter,
                           "value_reference": v_ref, "value_test": v_test,
                           "se_reference": result["se_ref"] * v_ref, "se_test": result["se_test"] * v_test})
    return pd.DataFrame(results), pd.DataFrame(params)
