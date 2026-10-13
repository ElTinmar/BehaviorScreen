"""
Direction readouts: the share of a behavior's events that go one way (ipsi,
or away from a looming stimulus). Per arm, the log odds of the share pooled
over fish; arms compared by fish-label permutation, standard errors by fish
bootstrap. Rows match the Tier 1 tables (parameter "P", value = odds), so
they join FDR, the map, the clutch report and the power analysis.
"""
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from BehaviorScreen.core import Laterality, Stim
from BehaviorScreen.point_process.analysis.model_config import (
    LATERAL_TURNS, LOOMING_ESCAPE, OKR_SWIMS, PHOTOTAXIS_FILTER, dataset,
)
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader, PointProcessDataset

from .dataset_utils import Contrast, safe_prepare_dataset

IPSI, CONTRA = Laterality.IPSILATERAL, Laterality.CONTRALATERAL
BOTH = [IPSI, CONTRA]

# {behavior: (dataset arguments, side counted, first event per trial only)}; all bout tables.
DIRECTION_READOUTS: Dict[str, Tuple[dict, Laterality, bool]] = {
    "looming_away": (dataset(LOOMING_ESCAPE, BOTH, 9.0, stim=Stim.LOOMING), CONTRA, True),
    "omr_lateral_ipsi_share": (dataset(LATERAL_TURNS, BOTH, 9.0, epoch_name=["grating right", "grating left"]), IPSI, False),
    "okr_ipsi_share": (dataset(OKR_SWIMS, BOTH, 9.0, stim=Stim.OKR), IPSI, False),
    "phototaxis_ipsi_share": (dataset("RT", BOTH, 24.0, stim=Stim.PHOTOTAXIS, trial_filters=PHOTOTAXIS_FILTER), IPSI, False),
    "prey_capture_ipsi_share": (dataset("JT", BOTH, 24.0, stim=Stim.PREY_CAPTURE), IPSI, False),
}


def side_counts(ds: PointProcessDataset, side: Laterality, first_only: bool) -> np.ndarray:
    """(fish, 2): events on `side`, all events."""
    ds = ds.first_events() if first_only else ds
    on_side = ds.marks["laterality"] == int(side)
    return np.column_stack([
        np.bincount(ds.event_fish_idx[on_side], minlength=ds.num_fish),
        np.bincount(ds.event_fish_idx, minlength=ds.num_fish),
    ])


def log_odds(counts: np.ndarray) -> np.ndarray:
    """Log odds of the share pooled over fish (axis -2), half a count added to each side."""
    k, n = counts[..., 0].sum(axis=-1), counts[..., 1].sum(axis=-1)
    return np.log((k + 0.5) / (n - k + 0.5))


def direction_test(ref: np.ndarray, test: np.ndarray, n_perm: int, n_boot: int, rng: np.random.Generator) -> dict:
    pooled, n_ref = np.vstack([ref, test]), len(ref)
    perms = rng.permuted(np.tile(np.arange(len(pooled)), (n_perm, 1)), axis=1)
    null = log_odds(pooled[perms[:, n_ref:]]) - log_odds(pooled[perms[:, :n_ref]])
    observed = log_odds(test) - log_odds(ref)
    boot_se = lambda c: float(np.std(log_odds(c[rng.integers(0, len(c), (n_boot, len(c)))])))
    return {
        "p_value": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)),
        "p_value_floor": 1.0 / (math.comb(len(pooled), n_ref) + 1),
        "log_odds_ref": float(log_odds(ref)), "log_odds_test": float(log_odds(test)),
        "se_ref": boot_se(ref), "se_test": boot_se(test),
    }


def run_direction_screen(
    loader: BehavioralDataLoader, contrasts: Sequence[Contrast],
    n_perm: int = 10000, n_boot: int = 1000, min_fish: int = 6, min_events: int = 20, seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(Tier 1-style results, parameter rows) for every contrast and direction readout."""
    rng = np.random.default_rng(seed)
    results: List[dict] = []
    params: List[dict] = []
    for behavior, (config, side, first_only) in DIRECTION_READOUTS.items():
        counts = {}
        for c in contrasts:
            for arm in (c.reference, c.test):
                if arm not in counts:
                    ds = safe_prepare_dataset(loader.subset(arm[0], [arm[1]]), config, f"{'/'.join(arm)}/{behavior}")
                    counts[arm] = None if ds is None else side_counts(ds, side, first_only)
            ref, test = counts[c.reference], counts[c.test]
            if ref is None or test is None:
                continue
            ref, test = ref[ref[:, 1] > 0], test[test[:, 1] > 0]
            row = {"family": c.family, "reference": "/".join(c.reference), "test": "/".join(c.test), "line": c.name,
                   "behavior": behavior, "n_fish_ref": len(ref), "n_fish_test": len(test),
                   "total_events": int(ref[:, 1].sum() + test[:, 1].sum())}
            if min(len(ref), len(test)) < min_fish or row["total_events"] < min_events:
                results.append({**row, "status": "skipped_insufficient_data", "p_value": np.nan})
                continue
            result = direction_test(ref, test, n_perm, n_boot, rng)
            results.append({**row, "status": "ok", "p_value": result["p_value"], "p_value_floor": result["p_value_floor"]})
            odds_ref, odds_test = np.exp(result["log_odds_ref"]), np.exp(result["log_odds_test"])
            params.append({"family": c.family, "line": c.name, "behavior": behavior, "parameter": "P",
                           "value_reference": odds_ref, "value_test": odds_test,
                           "se_reference": result["se_ref"] * odds_ref, "se_test": result["se_test"] * odds_test})
    return pd.DataFrame(results), pd.DataFrame(params)
