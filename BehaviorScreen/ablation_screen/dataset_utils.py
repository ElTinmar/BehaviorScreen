"""
Line/condition subsetting on top of BehavioralDataLoader, plus model-free
per-fish summary statistics used as a fallback when a parametric fit is
flagged (see model_qc.py).
"""
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from BehaviorScreen.point_process.dataset import BehavioralDataLoader, PointProcessDataset


def subset_loader(
    loader: BehavioralDataLoader,
    line: Optional[str] = None,
    conditions: Optional[List[str]] = None,
) -> BehavioralDataLoader:
    """Cheap: shares/filters the underlying DataFrame, no CSV re-read."""
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
    prepare_dataset() raises ValueError on data problems (empty slice,
    discontiguous trial_num after filtering). Catches broadly since a single
    bad (line, condition, behavior) combination should never crash a
    40-line x 12-behavior batch run.
    """
    try:
        return loader.prepare_dataset(**dataset_config)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"[{context}] skipped (data issue): {type(e).__name__}: {e}")
        return None


def per_fish_summary(dataset: PointProcessDataset) -> pd.DataFrame:
    """
    Model-free per-fish metrics used as a Tier-0 fallback when a parametric
    fit is flagged by model_qc.assess_group_fit:
      - rate_hz: total events / total observed exposure time
      - response_prob: fraction of observed trials with >=1 event
      - mean_first_latency_s: mean, across responding trials, of first-event time
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


def compare_fish_metric(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> dict:
    """Rank-based fallback comparison (Mann-Whitney), used only for
    architecture-collapse/insufficient-information fallback reporting."""
    from scipy.stats import mannwhitneyu

    a = df_a[metric].dropna().values
    b = df_b[metric].dropna().values
    if len(a) < 3 or len(b) < 3:
        return {"metric": metric, "n_a": len(a), "n_b": len(b), "p_value": np.nan}
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    n_a, n_b = len(a), len(b)
    rank_biserial_r = float(1.0 - 2.0 * stat / (n_a * n_b))
    return {
        "metric": metric, "n_a": n_a, "n_b": n_b,
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "rank_biserial_r": rank_biserial_r, "p_value": float(p),
    }