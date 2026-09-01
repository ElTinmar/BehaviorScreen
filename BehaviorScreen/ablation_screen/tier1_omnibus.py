"""
Tier 1: omnibus test -- "does ablation change anything for this line and
behavior" -- via three ordinary, low-dimensional fits (vehicle-only,
drug-only, pooled) on a PartiallyFixedProcess whose shape/nuisance
parameters are pinned to the Phase-3 pooled-across-all-vehicle-fish
reference fit, leaving only the effect parameters (amplitude, baseline
rate, ...) free per arm. A PERMUTATION test on the resulting deviance
statistic, not the asymptotic chi-square LRT, given small-sample concerns
at typical line_vehicle/line_drug arm sizes.
"""
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import contextlib

from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.partially_fixed_process import PartiallyFixedProcess

from .dataset_ops import pool_fish, select_fish
from .dataset_utils import subset_loader, safe_prepare_dataset, per_fish_summary, compare_fish_metric
from .arm_contrast import output_metric
from .model_qc import assess_group_fit

DEFAULT_MIN_EVENTS = 40
MIN_EVENTS_BY_BEHAVIOR: Dict[str, int] = {
    "looming_ipsi": 15, "looming_contra": 15, "dark_flash": 15,
}


def _total_events(dataset: PointProcessDataset) -> int:
    return int(dataset.stream_event_counts.sum()) if len(dataset.stream_event_counts) else 0


def _wrap(process_factory: Callable[[], PointProcess],
          fixed_shape_values: Optional[Dict[str, float]]) -> Callable[[], PointProcess]:
    """
    Returns a factory producing PartiallyFixedProcess(process_factory(), ...)
    if fixed_shape_values is non-empty, else process_factory itself unchanged
    -- so low-dimensional architectures (homogeneous_poisson, omr_forward,
    ...) that never had a convergence problem in the first place go through
    this pipeline with zero behavioral change.
    """
    if not fixed_shape_values:
        return process_factory
    return lambda: PartiallyFixedProcess(process_factory(), fixed_shape_values)


def fit_tier1(process_factory: Callable[[], PointProcess],
              ds_veh: PointProcessDataset, ds_drug: PointProcessDataset
              ) -> Tuple[float, PointProcess, PointProcess, PointProcess]:
    """Three independent fits; deviance = 2*(LL_independent - LL_pooled).
    Unchanged from the original -- process_factory is expected to already
    be the (possibly PartiallyFixedProcess-wrapped) low-dimensional
    architecture by the time it reaches here."""
    m_veh = process_factory(); m_veh.fit(ds_veh)
    m_drug = process_factory(); m_drug.fit(ds_drug)
    m_pooled = process_factory(); m_pooled.fit(pool_fish(ds_veh, ds_drug))

    deviance = max(0.0, 2.0 * ((m_veh.log_likelihood + m_drug.log_likelihood) - m_pooled.log_likelihood))
    return deviance, m_veh, m_drug, m_pooled


def tier1_permutation_test(
    process_factory: Callable[[], PointProcess],
    ds_veh: PointProcessDataset, ds_drug: PointProcessDataset,
    observed_deviance: Optional[float] = None,
    n_perm: int = 500, seed: int = 0, n_jobs: int = 1,
) -> Dict:
    """Unchanged in structure from the original; n_jobs default lowered to
    1 here since the OUTER run_tier1_screen loop is the one that should own
    process-level parallelism (see nested-parallelism discussion)."""
    if observed_deviance is None:
        observed_deviance, *_ = fit_tier1(process_factory, ds_veh, ds_drug)

    pooled = pool_fish(ds_veh, ds_drug)
    n_veh = ds_veh.num_fish
    n_distinct = math.comb(pooled.num_fish, n_veh)
    seeds = np.random.SeedSequence(seed).spawn(n_perm)

    def _one(s):
        rng = np.random.default_rng(s)
        perm = rng.permutation(pooled.num_fish)
        try:
            d, *_ = fit_tier1(process_factory,
                               select_fish(pooled, perm[:n_veh]), select_fish(pooled, perm[n_veh:]))
            return d
        except Exception:
            return None

    results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_one)(s) for s in seeds)
    null_dist = np.array([v for v in results if v is not None])
    n_failed = n_perm - len(null_dist)

    if len(null_dist) < max(20, 0.3 * n_perm):
        return {"observed_deviance": observed_deviance, "p_value": np.nan,
                "n_perm_success": len(null_dist), "n_perm_failed": n_failed,
                "n_distinct_permutations": n_distinct, "p_value_floor": 1.0 / (n_distinct + 1),
                "permutation_unreliable": True}

    p = (np.sum(null_dist >= observed_deviance) + 1) / (len(null_dist) + 1)
    return {"observed_deviance": observed_deviance, "p_value": float(p),
            "n_perm_success": len(null_dist), "n_perm_failed": n_failed,
            "n_distinct_permutations": n_distinct, "p_value_floor": 1.0 / (n_distinct + 1),
            "permutation_unreliable": False}


def tier1_effect_size(m_veh: PointProcess, ds_veh: PointProcessDataset,
                       m_drug: PointProcess, ds_drug: PointProcessDataset,
                       eps: float = 1e-3) -> Dict:
    """Unchanged -- m_veh/m_drug are ordinary fitted PointProcess instances
    (PartiallyFixedProcess delegates predict()/compute_expected_rate()/
    population_survival_curve() to its synced base_process), so
    output_metric works exactly as before."""
    val_veh, name = output_metric(m_veh, ds_veh)
    val_drug, _ = output_metric(m_drug, ds_drug)
    return {
        "metric_name": name, "metric_vehicle": val_veh, "metric_drug": val_drug,
        "log2_fold_change": float(np.log2((val_drug + eps) / (val_veh + eps))),
    }


def _tier1_one(
    line: str, behavior: str, dataset_config: dict,
    base_process_factory: Callable[[], PointProcess],
    null_process_factory: Callable[[], PointProcess],
    loader, veh_label: str, drug_label: str,
    fixed_shape_values: Optional[Dict[str, float]] = None,
    min_fish: int = 6, n_perm: int = 500,
) -> Dict:
    ctx = f"{line}/{behavior}"
    ds_veh = safe_prepare_dataset(subset_loader(loader, line, [veh_label]), dataset_config, ctx + "/veh")
    ds_drug = safe_prepare_dataset(subset_loader(loader, line, [drug_label]), dataset_config, ctx + "/drug")

    if ds_veh is None or ds_drug is None:
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_data", "p_value": np.nan}
    if ds_veh.num_fish < min_fish or ds_drug.num_fish < min_fish:
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_data", "p_value": np.nan}
    if ds_veh.num_trials != ds_drug.num_trials:
        return {"line": line, "behavior": behavior, "status": "skipped_trial_count_mismatch", "p_value": np.nan}

    min_events = MIN_EVENTS_BY_BEHAVIOR.get(behavior, DEFAULT_MIN_EVENTS)
    total_events = _total_events(ds_veh) + _total_events(ds_drug)
    if total_events < min_events:
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_information",
                "total_events": total_events, "min_events_required": min_events, "p_value": np.nan}

    wrapped_factory = _wrap(base_process_factory, fixed_shape_values)

    # --- Model QC gate, now on the wrapped (low-dimensional) process ---
    qc_veh = assess_group_fit(wrapped_factory, null_process_factory, ds_veh, line, behavior, "vehicle",
                               min_fish=min_fish)
    qc_drug = assess_group_fit(wrapped_factory, null_process_factory, ds_drug, line, behavior, "drug",
                                min_fish=min_fish)

    if qc_veh.verdict == "flagged" or qc_drug.verdict == "flagged":
        fallback_records = []
        try:
            sum_veh, sum_drug = per_fish_summary(ds_veh), per_fish_summary(ds_drug)
            for metric in ["rate_hz", "response_prob", "mean_first_latency_s"]:
                fallback_records.append(compare_fish_metric(sum_veh, sum_drug, metric))
        except Exception:
            pass
        return {
            "line": line, "behavior": behavior, "status": "flagged_architecture_collapse",
            "qc_reasons_vehicle": qc_veh.reasons_flagged, "qc_reasons_drug": qc_drug.reasons_flagged,
            "qc_notes_vehicle": qc_veh.informational_notes, "qc_notes_drug": qc_drug.informational_notes,
            "fallback_tier0": fallback_records, "p_value": np.nan,
        }

    # --- Real fit + permutation test, on the wrapped process ---
    try:
        observed_deviance, m_veh, m_drug, m_pooled = fit_tier1(wrapped_factory, ds_veh, ds_drug)
    except Exception as e:
        return {"line": line, "behavior": behavior, "status": f"fit_failed: {type(e).__name__}: {e}",
                "p_value": np.nan}

    perm = tier1_permutation_test(wrapped_factory, ds_veh, ds_drug,
                                   observed_deviance=observed_deviance, n_perm=n_perm)
    eff = tier1_effect_size(m_veh, ds_veh, m_drug, ds_drug)

    return {
        "line": line, "behavior": behavior, "status": "ok",
        "n_fish_veh": ds_veh.num_fish, "n_fish_drug": ds_drug.num_fish,
        "total_events": total_events,
        "free_params": list(wrapped_factory().param_names),
        "fixed_shape_values": fixed_shape_values or {},
        "fitted_params_vehicle": m_veh.param_dict_,
        "fitted_params_drug": m_drug.param_dict_,
        "qc_verdict_vehicle": qc_veh.verdict,
        "qc_verdict_drug": qc_drug.verdict,
        "qc_beats_null_vehicle": qc_veh.beats_null,
        "qc_beats_null_drug": qc_drug.beats_null,
        "qc_delta_aic_vehicle": qc_veh.delta_aic,
        "qc_delta_aic_drug": qc_drug.delta_aic,
        "qc_notes_vehicle": qc_veh.informational_notes,
        "qc_notes_drug": qc_drug.informational_notes,
        **perm, **eff,
    }


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def run_tier1_screen(
    loader, lines, behaviors, dataset_configs,
    base_process_factories, null_process_factories,
    fixed_shape_values_by_behavior: Optional[Dict[str, Dict[str, float]]] = None,
    line_labels=None, default_labels=("vehicle", "ronidazole"),
    n_jobs=1, n_perm=500, show_progress=True,
) -> pd.DataFrame:
    line_labels = line_labels or {}
    fixed_shape_values_by_behavior = fixed_shape_values_by_behavior or {}
    jobs = [(line, behavior) for line in lines for behavior in behaviors]

    task_iter = (
        joblib.delayed(_tier1_one)(
            line, behavior, dataset_configs[behavior],
            base_process_factories[behavior], null_process_factories[behavior],
            loader, *line_labels.get(line, default_labels),
            fixed_shape_values=fixed_shape_values_by_behavior.get(behavior),
            n_perm=n_perm,
        )
        for line, behavior in jobs
    )

    if show_progress:
        with tqdm_joblib(tqdm(total=len(jobs), desc="Tier 1", unit="cell")):
            records = joblib.Parallel(n_jobs=n_jobs)(task_iter)
    else:
        records = joblib.Parallel(n_jobs=n_jobs)(task_iter)

    return pd.DataFrame(records)