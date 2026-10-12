"""
Dataset building that tolerates missing data, plus model-free per-fish
summary statistics used as a fallback when a parametric fit is flagged
(see model_qc.py).
"""
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader, PointProcessDataset
from BehaviorScreen.point_process.point_process.point_process import PointProcess


def safe_prepare_dataset(
    loader: BehavioralDataLoader, dataset_config: dict, context: str = ""
) -> Optional[PointProcessDataset]:
    """
    None when the data cannot form a dataset (no trials, no usable trial):
    one empty (line, condition, behavior) cell must not stop a batch run.
    The loader raises ValueError for every such data problem.
    """
    try:
        return loader.prepare_dataset(**dataset_config)
    except ValueError as e:
        warnings.warn(f"[{context}] skipped (data issue): {e}")
        return None


class Arms:
    """
    Every (line, condition, behavior) arm of the screen, prepared and fitted
    at most once per run. dataset() is None when the data cannot form a
    dataset; model() raises the (cached) fit error.
    """

    def __init__(
        self,
        loader: BehavioralDataLoader,
        dataset_configs: Dict[str, dict],
        process_factories: Dict[str, Callable[[], PointProcess]],
        line_labels: Optional[Dict[str, Tuple[str, str]]] = None,
        default_labels: Tuple[str, str] = ("vehicle", "ronidazole"),
    ) -> None:
        self.loader = loader
        self.dataset_configs = dataset_configs
        self.process_factories = process_factories
        self.line_labels = line_labels or {}
        self.default_labels = default_labels
        self._datasets: Dict[Tuple[str, str, str], Optional[PointProcessDataset]] = {}
        self._models: Dict[Tuple[str, str, str], Union[PointProcess, Exception]] = {}

    def conditions(self, line: str) -> Tuple[str, str]:
        """(vehicle, drug) condition labels of a line."""
        return self.line_labels.get(line, self.default_labels)

    def dataset(self, line: str, condition: str, behavior: str) -> Optional[PointProcessDataset]:
        key = (line, condition, behavior)
        if key not in self._datasets:
            self._datasets[key] = safe_prepare_dataset(
                self.loader.subset(line, [condition]), self.dataset_configs[behavior], "/".join(key)
            )
        return self._datasets[key]

    def model(self, line: str, condition: str, behavior: str) -> PointProcess:
        key = (line, condition, behavior)
        if key not in self._models:
            model = self.process_factories[behavior]()
            try:
                model.fit(self.dataset(*key))
                self._models[key] = model
            except Exception as exc:
                self._models[key] = exc
        result = self._models[key]
        if isinstance(result, Exception):
            raise result
        return result


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