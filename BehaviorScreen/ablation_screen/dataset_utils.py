"""
Line/condition subsetting on top of BehavioralDataLoader, plus model-free
per-fish summary statistics shared by Tier 0 and Tier 3 reporting.
"""
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from BehaviorScreen.point_process.dataset import BehavioralDataLoader, PointProcessDataset


def subset_loader(
    loader: BehavioralDataLoader,
    line: Optional[str] = None,
    conditions: Optional[List[str]] = None,
) -> BehavioralDataLoader:
    """Cheap: shares the underlying DataFrame slice, no CSV re-read."""
    df = loader.raw_df
    if line is not None:
        df = df[df["line"] == line]
    if conditions is not None:
        df = df[df["condition"].isin(conditions)]

    new_loader = BehavioralDataLoader.__new__(BehavioralDataLoader)
    new_loader.raw_df = df.reset_index(drop=True)
    return new_loader


def safe_prepare_dataset(
    loader: BehavioralDataLoader, dataset_config: dict, context: str = ""
) -> Optional[PointProcessDataset]:
    """
    prepare_dataset() raises ValueError if trial_num isn't contiguous after
    filtering (e.g. a line has zero rows in some trial). That's expected to
    happen for low-N driver lines -- catch it and skip rather than crash the
    whole screen.
    """
    try:
        return loader.prepare_dataset(**dataset_config)
    except ValueError as e:
        warnings.warn(f"[{context}] skipped (data issue): {e}")
        return None
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"[{context}] skipped (unexpected failure): {e}")
        return None


def per_fish_summary(dataset: PointProcessDataset) -> pd.DataFrame:
    """
    Model-free per-fish metrics used by Tier 0:
      - rate_hz: total events / total observed exposure time
      - response_prob: fraction of observed trials with >=1 event
      - mean_first_latency_s: mean, across trials with a response, of the
        time of the first event that trial
    Works for both recurrent behaviors (rate_hz meaningful) and terminating/
    survival-style behaviors (response_prob, mean_first_latency_s meaningful).
    """
    n_fish = dataset.num_fish
    n_trials_obs = dataset.fish_trial_mask.sum(axis=1).astype(float)
    totals = np.zeros(n_fish)
    n_resp = np.zeros(n_fish)
    latency_sum = np.zeros(n_fish)
    latency_n = np.zeros(n_fish)

    for f_idx, t_idx, t_ev in dataset.iter_streams():
        totals[f_idx] += len(t_ev)
        if len(t_ev) > 0:
            n_resp[f_idx] += 1
            latency_sum[f_idx] += float(np.min(t_ev))
            latency_n[f_idx] += 1

    exposure_s = n_trials_obs * dataset.duration_s
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_hz = np.where(exposure_s > 0, totals / exposure_s, np.nan)
        response_prob = np.where(n_trials_obs > 0, n_resp / n_trials_obs, np.nan)
        mean_latency = np.where(latency_n > 0, latency_sum / np.maximum(latency_n, 1), np.nan)

    df = pd.DataFrame({
        "fish_idx": np.arange(n_fish),
        "n_trials_observed": n_trials_obs,
        "rate_hz": rate_hz,
        "response_prob": response_prob,
        "mean_first_latency_s": mean_latency,
    })
    return df[n_trials_obs > 0].reset_index(drop=True)