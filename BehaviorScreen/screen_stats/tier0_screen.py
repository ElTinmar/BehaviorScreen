"""
Tier 0: cheap, model-free fish-level screen. Runs before any optimizer call,
flags candidates for Tier 1 and catches degenerate lines/behaviors early.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .dataset_utils import subset_loader, safe_prepare_dataset, per_fish_summary


def compare_fish_metric(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> dict:
    a = df_a[metric].dropna().values
    b = df_b[metric].dropna().values
    if len(a) < 3 or len(b) < 3:
        return {"metric": metric, "n_a": len(a), "n_b": len(b), "p_value": np.nan}
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "metric": metric, "n_a": len(a), "n_b": len(b),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "U": float(stat), "p_value": float(p),
    }


def _tier0_one(line, behavior, dataset_config, loader, veh_label, drug_label):
    ctx = f"{line}/{behavior}"
    ds_veh = safe_prepare_dataset(subset_loader(loader, line, [veh_label]), dataset_config, ctx + "/veh")
    ds_drug = safe_prepare_dataset(subset_loader(loader, line, [drug_label]), dataset_config, ctx + "/drug")
    if ds_veh is None or ds_drug is None:
        return []

    sum_veh, sum_drug = per_fish_summary(ds_veh), per_fish_summary(ds_drug)
    records = []
    for metric in ["rate_hz", "response_prob", "mean_first_latency_s"]:
        rec = compare_fish_metric(sum_veh, sum_drug, metric)
        rec.update({"line": line, "behavior": behavior})
        records.append(rec)
    return records


def run_tier0_screen(
    loader,
    lines: List[str],
    behaviors: List[str],
    dataset_configs: Dict[str, dict],
    line_labels: Optional[Dict[str, tuple]] = None,
    default_labels: tuple = ("vehicle", "ronidazole"),
) -> pd.DataFrame:
    """
    line_labels: optional {line_name: (veh_label, drug_label)} override, e.g.
    {"WT": ("danieau", "ronidazole")}. Everything else defaults to
    ("vehicle", "ronidazole").
    """
    line_labels = line_labels or {}
    all_records = []
    for line in lines:
        veh_label, drug_label = line_labels.get(line, default_labels)
        for behavior in behaviors:
            all_records.extend(
                _tier0_one(line, behavior, dataset_configs[behavior], loader, veh_label, drug_label)
            )
    return pd.DataFrame(all_records)