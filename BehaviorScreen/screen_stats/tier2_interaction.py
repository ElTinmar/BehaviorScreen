"""
Tier 2: the actual ablation-specific effect, netting out genotype baseline
and generic drug/off-target effects. 2x2 factorial:

    covariates: genotype, drug, genotype:drug   (reference = WT-vehicle)

model_null:  genotype + drug            (line's drug response == WT's)
model_full:  genotype + drug + genotype:drug

LR test on the interaction term is THE test for "this line's phenotype is
specifically attributable to ablating this neuron population."
"""
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import joblib

from BehaviorScreen.point_process.point_process import ModelComparator
from .multi_group_process import GroupSpec, MultiGroupProcess
from .dataset_utils import subset_loader, safe_prepare_dataset


def fit_interaction_model(base_process_factory, ds_wt_veh, ds_wt_drug, ds_line_veh, ds_line_drug):
    groups = [
        GroupSpec("WT_vehicle", ds_wt_veh, {}),
        GroupSpec("WT_drug", ds_wt_drug, {"drug": 1.0}),
        GroupSpec("line_vehicle", ds_line_veh, {"genotype": 1.0}),
        GroupSpec("line_drug", ds_line_drug,
                  {"genotype": 1.0, "drug": 1.0, "genotype:drug": 1.0}),
    ]
    model_null = MultiGroupProcess(base_process_factory(), groups, covariate_names=["genotype", "drug"])
    model_full = MultiGroupProcess(base_process_factory(), groups,
                                    covariate_names=["genotype", "drug", "genotype:drug"])
    model_null.fit()
    model_full.fit()
    lr = ModelComparator.likelihood_ratio_test(model_null, model_full)
    return lr, model_null, model_full


def _tier2_one(line, behavior, dataset_config, base_process_factory, loader,
                wt_datasets_cache, line_veh_label, line_drug_label, min_fish=3):
    ctx = f"{line}/{behavior}"
    ds_wt_veh, ds_wt_drug = wt_datasets_cache[behavior]
    ds_line_veh = safe_prepare_dataset(subset_loader(loader, line, [line_veh_label]), dataset_config, ctx + "/veh")
    ds_line_drug = safe_prepare_dataset(subset_loader(loader, line, [line_drug_label]), dataset_config, ctx + "/drug")

    if any(d is None for d in [ds_wt_veh, ds_wt_drug, ds_line_veh, ds_line_drug]):
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_data", "p_value": np.nan}
    if min(ds_line_veh.num_fish, ds_line_drug.num_fish) < min_fish:
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_data", "p_value": np.nan}

    try:
        lr, model_null, model_full = fit_interaction_model(
            base_process_factory, ds_wt_veh, ds_wt_drug, ds_line_veh, ds_line_drug
        )
    except RuntimeError as e:
        return {"line": line, "behavior": behavior, "status": f"fit_failed: {e}", "p_value": np.nan}

    return {
        "line": line, "behavior": behavior, "status": "ok",
        "n_fish_line_veh": ds_line_veh.num_fish, "n_fish_line_drug": ds_line_drug.num_fish,
        "ll_null": lr["LL Null"], "ll_alt": lr["LL Alt"],
        "deviance": lr["Deviance (2*ΔLL)"], "df": lr["Δk (df)"],
        "p_value": lr["p-value"],
        "_model_full": model_full,  # kept in-process for Tier 3; drop before to_csv
    }


def build_wt_dataset_cache(loader, behaviors, dataset_configs, wt_line="WT",
                            wt_veh_label="danieau", wt_drug_label="ronidazole"):
    cache = {}
    for behavior in behaviors:
        ctx = f"WT/{behavior}"
        ds_veh = safe_prepare_dataset(subset_loader(loader, wt_line, [wt_veh_label]),
                                       dataset_configs[behavior], ctx + "/veh")
        ds_drug = safe_prepare_dataset(subset_loader(loader, wt_line, [wt_drug_label]),
                                        dataset_configs[behavior], ctx + "/drug")
        cache[behavior] = (ds_veh, ds_drug)
    return cache


def run_tier2_screen(
    candidates: pd.DataFrame,  # columns: line, behavior (Tier 1 hits)
    loader,
    dataset_configs: Dict[str, dict],
    base_process_factories: Dict[str, Callable],
    wt_datasets_cache: Dict[str, tuple],
    line_labels: Optional[Dict[str, tuple]] = None,
    default_labels: tuple = ("vehicle", "ronidazole"),
    n_jobs: int = -1,
) -> pd.DataFrame:
    line_labels = line_labels or {}
    rows = list(candidates[["line", "behavior"]].itertuples(index=False, name=None))

    records = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_tier2_one)(
            line, behavior, dataset_configs[behavior], base_process_factories[behavior],
            loader, wt_datasets_cache, *line_labels.get(line, default_labels)
        )
        for line, behavior in rows
    )
    return pd.DataFrame(records)