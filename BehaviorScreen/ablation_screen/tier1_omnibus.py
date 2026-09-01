"""
Tier 1: omnibus test -- "does ablation change anything for this line and
behavior" -- via three ordinary, low-dimensional fits (vehicle-only,
drug-only, pooled) and a PERMUTATION test on the resulting deviance
statistic (not the asymptotic chi-square LRT, given small-sample concerns
at typical line_vehicle/line_drug arm sizes).

No MultiGroupProcess needed for this saturated 2-group comparison --
vehicle-only + drug-only IS the saturated joint model, at lower
dimensionality and with no risk of the optimizer fragility seen in the
originally-attempted joint fits.
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

from .dataset_ops import pool_fish, select_fish
from .dataset_utils import subset_loader, safe_prepare_dataset, per_fish_summary, compare_fish_metric
from .arm_contrast import output_metric
from .model_qc import assess_group_fit

# Behavior-specific minimum total-event floor. Fish count is the WRONG
# denominator for point-process/survival identifiability -- total events
# (or uncensored first-responses) is what actually matters. Survival-style,
# low-base-rate behaviors need a much stricter floor.
DEFAULT_MIN_EVENTS = 40
MIN_EVENTS_BY_BEHAVIOR: Dict[str, int] = {
    "looming_ipsi": 15, "looming_contra": 15, "dark_flash": 15,
}


def _total_events(dataset: PointProcessDataset) -> int:
    return int(dataset.stream_event_counts.sum()) if len(dataset.stream_event_counts) else 0


def fit_tier1(process_factory: Callable[[], PointProcess],
              ds_veh: PointProcessDataset, ds_drug: PointProcessDataset
              ) -> Tuple[float, PointProcess, PointProcess, PointProcess]:
    """Three independent fits; deviance = 2*(LL_independent - LL_pooled),
    exactly the saturated-2-group joint-model comparison at lower
    dimensionality (mathematically identical target, see module docstring)."""
    m_veh = process_factory(); m_veh.fit(ds_veh)
    m_drug = process_factory(); m_drug.fit(ds_drug)
    m_pooled = process_factory(); m_pooled.fit(pool_fish(ds_veh, ds_drug))

    deviance = max(0.0, 2.0 * ((m_veh.log_likelihood + m_drug.log_likelihood) - m_pooled.log_likelihood))
    return deviance, m_veh, m_drug, m_pooled


def tier1_permutation_test(
    process_factory: Callable[[], PointProcess],
    ds_veh: PointProcessDataset, ds_drug: PointProcessDataset,
    observed_deviance: Optional[float] = None,
    n_perm: int = 500, seed: int = 0, n_jobs: int = -1,
) -> Dict:
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

    # --- Model QC gate on each arm before trusting any comparison ---
    qc_veh = assess_group_fit(base_process_factory, null_process_factory, ds_veh, line, behavior, "vehicle",
                               min_fish=min_fish)
    qc_drug = assess_group_fit(base_process_factory, null_process_factory, ds_drug, line, behavior, "drug",
                                min_fish=min_fish)

    # Only a HARD flag on either arm excludes the cell. "reliable_no_signal"
    # (architecture didn't beat null, but fit is otherwise trustworthy)
    # proceeds to the real fit_tier1 comparison below -- see GroupFitQC.verdict.
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

    # --- Real fit + permutation test ---
    try:
        observed_deviance, m_veh, m_drug, m_pooled = fit_tier1(base_process_factory, ds_veh, ds_drug)
    except Exception as e:
        return {"line": line, "behavior": behavior, "status": f"fit_failed: {type(e).__name__}: {e}",
                "p_value": np.nan}

    perm = tier1_permutation_test(base_process_factory, ds_veh, ds_drug,
                                   observed_deviance=observed_deviance, n_perm=n_perm)
    eff = tier1_effect_size(m_veh, ds_veh, m_drug, ds_drug)

    return {
        "line": line, "behavior": behavior, "status": "ok",
        "n_fish_veh": ds_veh.num_fish, "n_fish_drug": ds_drug.num_fish,
        "total_events": total_events,
        # Surfaced for reporting -- lets you see, per arm, whether the
        # architecture found real structure or collapsed to null-like
        # behavior, without that fact ever having excluded the cell.
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
    """Patches joblib to report progress into a tqdm bar. Standard recipe --
    joblib has no native tqdm hook, so this intercepts the batch-completion
    callback."""
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
    line_labels=None, default_labels=("vehicle", "ronidazole"),
    n_jobs=1, n_perm=500, show_progress=True,
) -> pd.DataFrame:
    line_labels = line_labels or {}
    jobs = [(line, behavior) for line in lines for behavior in behaviors]

    task_iter = (
        joblib.delayed(_tier1_one)(
            line, behavior, dataset_configs[behavior],
            base_process_factories[behavior], null_process_factories[behavior],
            loader, *line_labels.get(line, default_labels), n_perm=n_perm,
        )
        for line, behavior in jobs
    )

    if show_progress:
        with tqdm_joblib(tqdm(total=len(jobs), desc="Tier 1", unit="cell")):
            records = joblib.Parallel(n_jobs=n_jobs)(task_iter)
    else:
        records = joblib.Parallel(n_jobs=n_jobs)(task_iter)

    return pd.DataFrame(records)