# frailty_correlation.py
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def collect_fish_gains(
    fitted_mixed_models: Dict[str, "GammaMixedEffectsProcess"],
    datasets: Dict[str, "PointProcessDataset"],
    fish_ids: Dict[str, np.ndarray],  # behavior -> array of 'file' ids, aligned to fish_idx
    min_exposure_S: float = 1.0,
) -> pd.DataFrame:
    """Wide table: rows = fish ('file'), columns = behavior, values = estimated_gain
    (NaN if that fish wasn't observed / had too little exposure in that behavior)."""
    frames = []
    for behavior, model in fitted_mixed_models.items():
        gains = model.estimate_fish_gains(datasets[behavior])
        gains["file"] = fish_ids[behavior][gains["fish_idx"].values]
        gains = gains[gains["expected_events_base"] >= min_exposure_S]
        frames.append(gains.set_index("file")["estimated_gain"].rename(behavior))
    return pd.concat(frames, axis=1)

def frailty_correlation_matrix(wide_gains: pd.DataFrame, min_n_pairwise: int = 15):
    behaviors = wide_gains.columns
    n = len(behaviors)
    rho = np.full((n, n), np.nan)
    pval = np.full((n, n), np.nan)
    n_pairs = np.zeros((n, n), dtype=int)

    for i, bi in enumerate(behaviors):
        for j, bj in enumerate(behaviors):
            if j < i:
                continue
            paired = wide_gains[[bi, bj]].dropna()
            n_pairs[i, j] = n_pairs[j, i] = len(paired)
            if len(paired) >= min_n_pairwise:
                r, p = spearmanr(paired[bi], paired[bj])
                rho[i, j] = rho[j, i] = r
                pval[i, j] = pval[j, i] = p

    return (
        pd.DataFrame(rho, index=behaviors, columns=behaviors),
        pd.DataFrame(pval, index=behaviors, columns=behaviors),
        pd.DataFrame(n_pairs, index=behaviors, columns=behaviors),
    )