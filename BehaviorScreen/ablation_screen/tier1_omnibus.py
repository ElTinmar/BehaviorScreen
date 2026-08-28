"""
Tier 1: per (line, behavior) omnibus LR test -- "does ablation change
anything about this readout", using the FROZEN model architecture chosen
once on the pooled control data (do not re-run model selection per line).
"""
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import joblib

from BehaviorScreen.point_process.point_process import ModelComparator
from .multi_group_process import GroupSpec, MultiGroupProcess
from .dataset_utils import subset_loader, safe_prepare_dataset


def fit_two_group_omnibus(base_process_factory: Callable, ds_a, ds_b,
                           name_a="group_a", name_b="group_b"):
    groups = [GroupSpec(name_a, ds_a, {}), GroupSpec(name_b, ds_b, {"drug": 1.0})]
    model_null = MultiGroupProcess(base_process_factory(), groups, covariate_names=[])
    model_alt = MultiGroupProcess(base_process_factory(), groups, covariate_names=["drug"])
    model_null.fit()
    model_alt.fit()
    lr = ModelComparator.likelihood_ratio_test(model_null, model_alt)
    return lr, model_null, model_alt


def _tier1_one(line, behavior, dataset_config, base_process_factory, loader,
                veh_label, drug_label, min_fish=3):
    ctx = f"{line}/{behavior}"
    ds_veh = safe_prepare_dataset(subset_loader(loader, line, [veh_label]), dataset_config, ctx + "/veh")
    ds_drug = safe_prepare_dataset(subset_loader(loader, line, [drug_label]), dataset_config, ctx + "/drug")

    if ds_veh is None or ds_drug is None or ds_veh.num_fish < min_fish or ds_drug.num_fish < min_fish:
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_data", "p_value": np.nan}

    try:
        lr, _, _ = fit_two_group_omnibus(base_process_factory, ds_veh, ds_drug, "vehicle", "ronidazole")
    except RuntimeError as e:
        return {"line": line, "behavior": behavior, "status": f"fit_failed: {e}", "p_value": np.nan}

    return {
        "line": line, "behavior": behavior, "status": "ok",
        "n_fish_veh": ds_veh.num_fish, "n_fish_drug": ds_drug.num_fish,
        "ll_null": lr["LL Null"], "ll_alt": lr["LL Alt"],
        "deviance": lr["Deviance (2*ΔLL)"], "df": lr["Δk (df)"],
        "p_value": lr["p-value"],
    }


def run_tier1_screen(
    loader,
    lines: List[str],
    behaviors: List[str],
    dataset_configs: Dict[str, dict],
    base_process_factories: Dict[str, Callable],
    line_labels: Optional[Dict[str, tuple]] = None,
    default_labels: tuple = ("vehicle", "ronidazole"),
    n_jobs: int = -1,
) -> pd.DataFrame:
    line_labels = line_labels or {}
    jobs = [(line, behavior) for line in lines for behavior in behaviors]

    records = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_tier1_one)(
            line, behavior, dataset_configs[behavior], base_process_factories[behavior],
            loader, *line_labels.get(line, default_labels)
        )
        for line, behavior in jobs
    )
    return pd.DataFrame(records)