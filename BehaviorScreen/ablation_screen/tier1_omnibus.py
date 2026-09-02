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
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.tqdm_joblib import tqdm_joblib
from BehaviorScreen.point_process.io import save_fig
from BehaviorScreen.point_process.frailty_analysis import (
    collect_fish_gains, 
    plot_fish_gain_correlation_dual, 
    plot_fish_gain_correlation,
)

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

    with tqdm_joblib(tqdm(total=n_perm, desc="Permutation", position=1, leave=False)):
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


def single_param_effect_size(m_veh: PointProcess, m_drug: PointProcess, free_param: str,
                              eps: float = 1e-9) -> Dict:

    v_veh = m_veh.param_dict_.get(free_param)
    v_drug = m_drug.param_dict_.get(free_param)
    if v_veh is None or v_drug is None:
        raise ValueError(f"{free_param} not found in one of the fitted models' param_dict_")

    return {
        "metric_name": f"param_{free_param}",
        "metric_vehicle": v_veh,
        "metric_drug": v_drug,
        "effect_size": float(np.log2(max(v_drug, eps) / max(v_veh, eps))),
        "effect_size_type": "log2_fold_change",
    }

def tier1_effect_size(m_veh: PointProcess, ds_veh: PointProcessDataset,
                       m_drug: PointProcess, ds_drug: PointProcessDataset,
                       eps: float = 1e-9) -> Dict:

    # Single-free-parameter architectures: use the parameter directly as
    # the effect size (see conversation notes -- more informative here
    # than the aggregate output_metric, since there's no other dimension
    # for the effect to hide in, and no saturation risk).
    if len(m_veh.param_names) == 1:
        return single_param_effect_size(m_veh, m_drug, m_veh.param_names[0], eps=eps)
    
    val_veh, name = output_metric(m_veh, ds_veh)
    val_drug, _ = output_metric(m_drug, ds_drug)

    if name == "response_probability":
        # Bounded metric (0,1) -- log2-ratio is NOT symmetric here (a
        # change from 0.5->0.9 and 0.5->0.1 are the "same size" in the
        # opposite direction but get very different |log2FC|; and a tiny
        # absolute change near 0 produces an enormous log2FC). Use the
        # logit difference instead: unbounded, symmetric under p<->1-p,
        # and the standard transform for exactly this situation (same
        # scale logistic regression coefficients live on).
        p_veh = np.clip(val_veh, eps, 1 - eps)
        p_drug = np.clip(val_drug, eps, 1 - eps)
        effect = float(np.log(p_drug / (1 - p_drug)) - np.log(p_veh / (1 - p_veh)))
        effect_name = "logit_diff"
    else:
        # Unbounded positive rate -- log2-ratio is appropriate here:
        # symmetric under doubling/halving, scale-invariant.
        effect = float(np.log2(max(val_drug, eps) / max(val_veh, eps)))
        effect_name = "log2_fold_change"
    
    return {
        "metric_name": name, "metric_vehicle": val_veh, "metric_drug": val_drug,
        "effect_size": effect, "effect_size_type": effect_name,
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

def run_tier1_screen(
    loader, lines, behaviors, dataset_configs,
    base_process_factories, null_process_factories,
    line_labels=None, default_labels=("vehicle", "ronidazole"),
    n_perm=500, show_progress=True,
) -> pd.DataFrame:

    line_labels = line_labels or {}
    jobs = [(line, behavior) for line in lines for behavior in behaviors]
    cell_iter = tqdm(jobs, desc="Tier 1 cells", position=0, unit="cell") if show_progress else jobs

    records = []
    for line, behavior in cell_iter:
        try:
            record = _tier1_one(
                line, behavior, dataset_configs[behavior],
                base_process_factories[behavior], null_process_factories[behavior],
                loader, *line_labels.get(line, default_labels), n_perm=n_perm,
            )
        except Exception as e:
            record = {"line": line, "behavior": behavior,
                      "status": f"unexpected_error: {type(e).__name__}: {e}", "p_value": np.nan}
        records.append(record)

    return pd.DataFrame(records)


def compute_parameter_deltas(param_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'delta' (drug vs vehicle change) and 'delta_z' (z-scored within
    each (behavior, parameter) column, across lines) to a long-format
    parameter table.

    log2-ratio vs raw-difference is chosen PER (behavior, parameter) column
    based on whether every observed value is strictly positive -- not by
    name pattern, since parameter names aren't consistent enough across
    architectures (B means 'baseline rate in Hz' in one kernel, tau means
    'decay time constant' in another). Rate/scale-like parameters (B, A, H,
    tau, sigma, mu) are virtually always positive across a whole screen and
    get log2-ratio (scale-invariant); parameters that can be negative
    (alpha_*, z0_*, z_dip -- already living on a log/logit scale by
    construction, see the reparametrization work) get a raw difference,
    which is the natural, already-additive unit for those.
    """
    df = param_df.copy()
    df["delta"] = np.nan

    for (behavior, pname), group in df.groupby(["behavior", "parameter"]):
        vals = pd.concat([group["value_vehicle"], group["value_drug"]]).dropna()
        use_log_ratio = len(vals) > 0 and (vals > 0).all()

        idx = group.index
        if use_log_ratio:
            df.loc[idx, "delta"] = np.log2(
                group["value_drug"].clip(lower=1e-9) / group["value_vehicle"].clip(lower=1e-9)
            )
        else:
            df.loc[idx, "delta"] = group["value_drug"] - group["value_vehicle"]

    # Z-score within each (behavior, parameter) column across lines --
    # makes heterogeneous units/scales comparable as "how unusual is this
    # line relative to the rest of the screen", which is what the heatmap
    # color should mean.
    df["delta_z"] = df.groupby(["behavior", "parameter"])["delta"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )
    return df

def extract_fitted_parameters(
    tier1_df: pd.DataFrame,
    loader, dataset_configs, base_process_factories,
    line_labels=None, default_labels=("vehicle", "ronidazole"),
) -> pd.DataFrame:

    line_labels = line_labels or {}
    records = []

    valid = tier1_df[tier1_df["status"] == "ok"]
    for _, row in valid.iterrows():
        line, behavior = row["line"], row["behavior"]
        veh_label, drug_label = line_labels.get(line, default_labels)

        ds_veh = safe_prepare_dataset(subset_loader(loader, line, [veh_label]), dataset_configs[behavior])
        ds_drug = safe_prepare_dataset(subset_loader(loader, line, [drug_label]), dataset_configs[behavior])
        if ds_veh is None or ds_drug is None:
            continue

        try:
            m_veh = base_process_factories[behavior](); m_veh.fit(ds_veh)
            m_drug = base_process_factories[behavior](); m_drug.fit(ds_drug)
        except Exception as e:
            tqdm.write(f"[{line}/{behavior}] refit failed during parameter extraction: {e}")
            continue

        for pname in m_veh.param_names:
            v_veh = m_veh.param_dict_.get(pname)
            v_drug = m_drug.param_dict_.get(pname)
            if v_veh is None or v_drug is None:
                continue
            records.append({
                "line": line, "behavior": behavior, "parameter": pname,
                "value_vehicle": v_veh, "value_drug": v_drug,
                "significant": bool(row.get("significant", False)),
            })

    return pd.DataFrame(records)

def plot_parameter_change_heatmaps(
    param_df: pd.DataFrame,
    line_order: Optional[List[str]] = None,
    cmap: str = "coolwarm",
    figsize_per_col: float = 1.2,
    row_height: float = 0.28,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Small-multiples grid: one heatmap panel per behavior, rows=lines,
    columns=that behavior's free parameters, color=z-scored drug-vs-vehicle
    change (delta_z). Deliberately NOT a single combined (line x
    behavior_parameter) matrix -- most parameter names are behavior-
    specific (e.g. z0_ripple only exists for prey_capture_ipsi), so a
    combined matrix would be mostly gray/NaN and hard to scan; one panel
    per behavior keeps each subplot dense and legible.

    Significant cells (per tier1's omnibus test) are marked with '*'.
    """
    behaviors = sorted(param_df["behavior"].unique())
    if line_order is None:
        line_order = sorted(param_df["line"].unique())

    n_behaviors = len(behaviors)
    n_cols = 4
    n_rows = int(np.ceil(n_behaviors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * max(2.5, len(line_order) * row_height)),
        squeeze=False,
    )

    vmax = np.nanmax(np.abs(param_df["delta_z"])) if len(param_df) else 1.0
    vmax = max(vmax, 1e-6)

    for i, behavior in enumerate(behaviors):
        ax = axes[i // n_cols, i % n_cols]
        sub = param_df[param_df["behavior"] == behavior]
        params = sorted(sub["parameter"].unique())

        matrix = np.full((len(line_order), len(params)), np.nan)
        sig_mask = np.zeros_like(matrix, dtype=bool)
        for r, line in enumerate(line_order):
            for c, pname in enumerate(params):
                cell = sub[(sub["line"] == line) & (sub["parameter"] == pname)]
                if len(cell):
                    matrix[r, c] = cell["delta_z"].iloc[0]
                    sig_mask[r, c] = cell["significant"].iloc[0]

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="lightgray")
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=-vmax, vmax=vmax)

        for r in range(len(line_order)):
            for c in range(len(params)):
                if sig_mask[r, c]:
                    ax.text(c, r, "*", ha="center", va="center", color="black", fontsize=10, fontweight="bold")

        ax.set_title(behavior, fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params, rotation=45, ha="right", fontsize=7)
        if i % n_cols == 0:
            ax.set_yticks(range(len(line_order)))
            ax.set_yticklabels(line_order, fontsize=6)
        else:
            ax.set_yticks([])

    for j in range(n_behaviors, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.5, label="Z-scored Δ (drug vs vehicle)\n* = significant (Tier 1)")

    return fig, axes

def _weighted_marginals(surface: np.ndarray, dataset: PointProcessDataset) -> Tuple[np.ndarray, np.ndarray]:
    """(unchanged from before)"""
    n_fish = dataset.n_fish_per_trial
    active = n_fish > 0

    time_marginal = np.full(surface.shape[1], np.nan)
    if active.any():
        time_marginal = np.average(surface[active], axis=0, weights=n_fish[active])

    trial_marginal = np.full(surface.shape[0], np.nan)
    trial_marginal[active] = np.mean(surface[active], axis=1)

    return time_marginal, trial_marginal

def _robust_vmax(surfaces: List[np.ndarray], percentile: float = 95.0) -> float:
    pooled = np.concatenate([s[np.isfinite(s) & (s > 0)].ravel() for s in surfaces])
    if len(pooled) == 0:
        return 1e-9
    return float(np.percentile(pooled, percentile))

def plot_arm_surface_grid_with_marginals(
    ds_veh: PointProcessDataset, ds_drug: PointProcessDataset,
    m_veh: PointProcess, m_drug: PointProcess,
    title: str = "",
    cmap: str = "plasma",
    figsize: Tuple[float, float] = (12, 10),
    vmax_percentile: float = 95.0,
) -> plt.Figure:
    """
    Per arm (vehicle, drug): empirical heatmap | model heatmap, with ONE
    shared time-marginal strip above (spanning both heatmap columns) and
    ONE shared trial-marginal column to the right -- each marginal overlays
    empirical (solid) vs model (dashed) on the same axis, rather than
    living in 4 separate locations. This directly shows fit calibration
    (the actual point of a marginal) without making the eye hop between
    two physically separate line plots, and removes the redundancy of the
    previous version (where the "vehicle" time marginal was effectively
    shown twice, once framed under each heatmap).

    Single shared GridSpec for the whole figure (not nested per-panel
    GridSpecs) -- guarantees pixel alignment across rows/columns. Colorbar
    is an explicit, manually-positioned axis, not matplotlib-placed.
    """
    surf_veh_emp = ds_veh.time_trial_histogram_hz
    surf_veh_model = m_veh.compute_expected_rate(ds_veh)
    surf_drug_emp = ds_drug.time_trial_histogram_hz
    surf_drug_model = m_drug.compute_expected_rate(ds_drug)

    tm_veh_emp, trm_veh_emp = _weighted_marginals(surf_veh_emp, ds_veh)
    tm_veh_model, trm_veh_model = _weighted_marginals(surf_veh_model, ds_veh)
    tm_drug_emp, trm_drug_emp = _weighted_marginals(surf_drug_emp, ds_drug)
    tm_drug_model, trm_drug_model = _weighted_marginals(surf_drug_model, ds_drug)

    vmax = _robust_vmax([surf_veh_emp, surf_drug_emp], percentile=vmax_percentile)
    time_ymax = max(np.nanmax(x) for x in [tm_veh_emp, tm_veh_model, tm_drug_emp, tm_drug_model]
                     if np.any(np.isfinite(x)))
    trial_xmax = max(np.nanmax(x) for x in [trm_veh_emp, trm_veh_model, trm_drug_emp, trm_drug_model]
                      if np.any(np.isfinite(x)))
    time_ymax, trial_xmax = max(time_ymax, 1e-9), max(trial_xmax, 1e-9)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        width_ratios=[4, 4, 1.3],
        height_ratios=[1, 4, 1, 4],
        wspace=0.12, hspace=0.35,
        left=0.07, right=0.86, top=0.90, bottom=0.07,
    )

    def _draw_arm(row0: int, ds, surf_emp, surf_model, tm_emp, tm_model, trm_emp, trm_model, row_label: str,
                  show_legend: bool):
        ax_time = fig.add_subplot(gs[row0, 0:2])
        ax_emp = fig.add_subplot(gs[row0 + 1, 0], sharex=ax_time)
        ax_model = fig.add_subplot(gs[row0 + 1, 1], sharex=ax_time, sharey=ax_emp)
        ax_trial = fig.add_subplot(gs[row0 + 1, 2], sharey=ax_emp)

        ax_time.plot(ds.t_centers, tm_emp, color="black", linestyle="-", linewidth=1.3, label="Empirical")
        ax_time.plot(ds.t_centers, tm_model, color="crimson", linestyle="--", linewidth=1.3, label="Model")
        ax_time.set_ylim(0, time_ymax * 1.05)
        ax_time.set_title(row_label, fontsize=11, fontweight="bold", loc="left")
        plt.setp(ax_time.get_xticklabels(), visible=False)
        ax_time.tick_params(labelsize=6)
        if show_legend:
            ax_time.legend(loc="upper right", fontsize=7, frameon=False)

        mesh = ax_emp.pcolormesh(ds.t_grid, ds.trial_edges, surf_emp, shading="flat", cmap=cmap, vmin=0.0, vmax=vmax)
        ax_model.pcolormesh(ds.t_grid, ds.trial_edges, surf_model, shading="flat", cmap=cmap, vmin=0.0, vmax=vmax)
        if row0 == 0:
            ax_emp.set_title("Empirical", fontsize=9)
            ax_model.set_title("Model", fontsize=9)
        ax_emp.set_xlabel("Time in trial (s)", fontsize=8)
        ax_model.set_xlabel("Time in trial (s)", fontsize=8)
        ax_emp.set_ylabel("Trial", fontsize=8)
        plt.setp(ax_model.get_yticklabels(), visible=False)
        ax_emp.tick_params(labelsize=7)
        ax_model.tick_params(labelsize=7)

        trial_centers = np.arange(ds.num_trials)
        ax_trial.plot(trm_emp, trial_centers, color="black", linestyle="-", linewidth=1.3)
        ax_trial.plot(trm_model, trial_centers, color="crimson", linestyle="--", linewidth=1.3)
        ax_trial.set_xlim(0, trial_xmax * 1.05)
        plt.setp(ax_trial.get_yticklabels(), visible=False)
        ax_trial.tick_params(labelsize=6)

        return mesh

    _draw_arm(0, ds_veh, surf_veh_emp, surf_veh_model, tm_veh_emp, tm_veh_model, trm_veh_emp, trm_veh_model,
              "Vehicle", show_legend=True)
    mesh = _draw_arm(2, ds_drug, surf_drug_emp, surf_drug_model, tm_drug_emp, tm_drug_model, trm_drug_emp, trm_drug_model,
                      "Drug", show_legend=False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)
    cax = fig.add_axes([0.89, 0.15, 0.02, 0.65])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Rate (Hz)", fontsize=9)

    return fig

def generate_arm_surface_grids(
    cells: pd.DataFrame,
    loader, dataset_configs, base_process_factories,
    output_dir: Path,
    line_labels=None, default_labels=("vehicle", "ronidazole"),
) -> pd.DataFrame:
    """Returns a per-(line, behavior) status table -- 'plotted' / 'no_data' /
    'fit_failed' -- so missing plots are traceable without re-deriving them
    from the output directory after the fact."""
    line_labels = line_labels or {}
    results = []

    for _, row in tqdm(cells.iterrows(), total=len(cells), desc="Arm surface grids"):
        line, behavior = row["line"], row["behavior"]
        veh_label, drug_label = line_labels.get(line, default_labels)

        ds_veh = safe_prepare_dataset(subset_loader(loader, line, [veh_label]), dataset_configs[behavior])
        ds_drug = safe_prepare_dataset(subset_loader(loader, line, [drug_label]), dataset_configs[behavior])
        if ds_veh is None or ds_drug is None:
            results.append({"line": line, "behavior": behavior, "plot_status": "no_data"})
            continue

        try:
            m_veh = base_process_factories[behavior](); m_veh.fit(ds_veh)
            m_drug = base_process_factories[behavior](); m_drug.fit(ds_drug)
        except Exception as e:
            tqdm.write(f"[{line}/{behavior}] fit failed, skipping plot: {e}")
            results.append({"line": line, "behavior": behavior, "plot_status": f"fit_failed: {e}"})
            continue

        reason = row.get("plot_reason", "")
        title = f"{line} / {behavior}" + (f"  [{reason}]" if reason else "")
        fig = plot_arm_surface_grid_with_marginals(ds_veh, ds_drug, m_veh, m_drug, title=title)
        path = save_fig(fig, output_dir / (reason or "all"), f"{line}_{behavior}")
        plt.close(fig)
        results.append({"line": line, "behavior": behavior, "plot_status": "plotted", "path": str(path)})

    status_df = pd.DataFrame(results)
    print(status_df["plot_status"].value_counts())
    return status_df

def build_bad_fit_triage(tier1_df: pd.DataFrame) -> pd.DataFrame:
    df = tier1_df.copy()
    df["cause"] = "ok"

    status = df["status"]
    df.loc[status == "skipped_insufficient_data", "cause"] = "no_data"
    df.loc[status == "skipped_trial_count_mismatch", "cause"] = "trial_count_mismatch"
    df.loc[status == "skipped_insufficient_information", "cause"] = "insufficient_events"
    df.loc[status.str.startswith("fit_failed", na=False), "cause"] = "fit_exception"
    df.loc[status.str.startswith("unexpected_error", na=False), "cause"] = "fit_exception"
    df.loc[status == "flagged_architecture_collapse", "cause"] = "architecture_collapse"
    df.loc[status == "ok_response_abolished", "cause"] = "response_abolished"  # informational, not "bad"

    is_ok = df["cause"] == "ok"

    at_floor = is_ok & (df["p_value"] <= df["p_value_floor"] * 1.5)
    df.loc[at_floor, "cause"] = "p_value_near_floor"

    unreliable = is_ok & df["permutation_unreliable"].fillna(False)
    df.loc[unreliable, "cause"] = "permutation_unreliable"

    no_signal_arm = is_ok & (
        (df["qc_verdict_vehicle"] == "reliable_no_signal") |
        (df["qc_verdict_drug"] == "reliable_no_signal")
    )
    df.loc[no_signal_arm, "cause"] = "one_arm_no_signal"

    weak_margin = is_ok & (
        (df["qc_delta_aic_vehicle"].fillna(np.inf) < 5) |
        (df["qc_delta_aic_drug"].fillna(np.inf) < 5)
    )
    df.loc[weak_margin, "cause"] = "weak_qc_margin"

    return df


def summarize_bad_fits(triage_df: pd.DataFrame) -> pd.DataFrame:
    """Cause counts, sorted descending -- the quick top-level view."""
    return (
        triage_df["cause"].value_counts()
        .rename_axis("cause").reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
    )


def plot_volcano(
    df: pd.DataFrame,
    triage_df: Optional[pd.DataFrame] = None,   # from build_bad_fit_triage, for cause-aware markers
    effect_col: str = "effect_size",
    label_col: str = "line",
    figsize_per_panel: Tuple[float, float] = (4.0, 3.5),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Volcano plot(s): log2_fold_change (x) vs -log10(p_value) (y), for every
    'ok'/'ok_response_abolished' cell. Points are colored/shaped by
    fit-quality cause (from build_bad_fit_triage) rather than plain
    significance alone -- this matters here specifically because several
    known issues (permutation-floor pinning, weak QC margin, one-arm
    collapse) can make a cell LOOK like a clean hit by q-value alone while
    still deserving a discount. If triage_df isn't supplied, falls back to
    a plain significant/not-significant coloring.
    """
    plot_df = df[df["status"].isin(["ok", "ok_response_abolished"])].copy()
    plot_df = plot_df.dropna(subset=[effect_col, "p_value"])
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_value"].clip(lower=1e-300))

    if triage_df is not None:
        merged = plot_df.merge(
            triage_df[["line", "behavior", "cause"]], on=["line", "behavior"], how="left"
        )
        plot_df["cause"] = merged["cause"].fillna("ok")
    else:
        plot_df["cause"] = np.where(plot_df.get("significant", False), "significant", "ok")

    cause_style = {
        "ok": dict(color="lightsteelblue", marker="o", alpha=0.5, label="ok, not significant"),
        "significant": dict(color="crimson", marker="o", alpha=0.85, label="significant"),
        "p_value_near_floor": dict(color="darkorange", marker="^", alpha=0.85, label="p near permutation floor"),
        "permutation_unreliable": dict(color="gray", marker="x", alpha=0.7, label="permutation unreliable"),
        "one_arm_no_signal": dict(color="purple", marker="s", alpha=0.7, label="one arm no signal vs null"),
        "weak_qc_margin": dict(color="goldenrod", marker="D", alpha=0.6, label="weak QC margin (ΔAIC<5)"),
        "response_abolished": dict(color="black", marker="*", alpha=0.9, label="response abolished (Fisher test)"),
    }
    default_style = dict(color="gray", marker="o", alpha=0.4, label="other")

    def _behavior_q_line(sub: pd.DataFrame) -> Optional[float]:
        sig_rows = sub[sub.get("significant", False)]
        return sig_rows["p_value"].max() if len(sig_rows) else None

    def _scatter_by_cause(ax, sub):
        for cause, group in sub.groupby("cause"):
            style = cause_style.get(cause, default_style)
            ax.scatter(group[effect_col], group["neg_log10_p"], s=22, edgecolors="none", **style)

    def _resolve_top_n(sub: pd.DataFrame, max_labels: int = 15, min_labels: int = 3) -> int:
        n_sig = int(sub.get("significant", pd.Series(dtype=bool)).sum())
        return int(np.clip(n_sig, min_labels, max_labels))

    behaviors = sorted(plot_df["behavior"].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(behaviors) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    y_max = plot_df["neg_log10_p"].max()

    for i, behavior in enumerate(behaviors):
        ax = axes[i // n_cols, i % n_cols]
        sub = plot_df[plot_df["behavior"] == behavior]
        _scatter_by_cause(ax, sub)
        q_line_p = _behavior_q_line(sub)          # <-- per-facet now
        if q_line_p is not None:
            ax.axhline(-np.log10(q_line_p), color="black", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.5)
        local_xmax = sub[effect_col].abs().max() if len(sub) else 1.0
        ax.set_xlim(-local_xmax * 1.15, local_xmax * 1.15)
        ax.set_ylim(0, y_max * 1.1)
        ax.set_title(behavior, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        panel_top_n = _resolve_top_n(sub, max_labels=15, min_labels=3)
        _label_top_hits(ax, sub, effect_col, label_col, panel_top_n)
        etype = sub["effect_size_type"].mode().iat[0] if "effect_size_type" in sub.columns and len(sub) else "effect_size"
        ax.set_xlabel("logit(p_drug) - logit(p_veh)" if etype == "logit_diff" else "log2(drug/vehicle rate)", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("-log10(p)", fontsize=8) 

    for j in range(len(behaviors), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    handles = [plt.Line2D([0], [0], linestyle="", marker=s["marker"], color=s["color"], label=s["label"])
               for s in cause_style.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Volcano: effect size vs. significance, by behavior", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig, axes


def _label_top_hits(ax, sub: pd.DataFrame, effect_col: str, label_col: str, top_n: int):
    if top_n <= 0 or "p_value" not in sub.columns:
        return
    top = sub.nsmallest(top_n, "p_value")
    for _, row in top.iterrows():
        ax.annotate(
            str(row[label_col]), (row[effect_col], row["neg_log10_p"]),
            fontsize=6, xytext=(3, 3), textcoords="offset points",
            rotation=45, rotation_mode="anchor", ha="left", va="bottom"
        )

def build_models_and_datasets_for_line(
    line: str, label: str,
    loader, dataset_configs: dict, base_process_factories: dict,
    behaviors: Optional[List[str]] = None,
) -> Dict[str, Tuple[PointProcess, PointProcessDataset]]:
    """
    Fits every behavior's pinned architecture for ONE (line, label) arm.
    This is the ablation-screen-specific orchestration (knows about
    loader/line/label); frailty_analysis knows nothing about any of it.
    """
    behaviors = behaviors or list(dataset_configs.keys())
    broad = subset_loader(loader, line, [label])
    result = {}

    for behavior in behaviors:
        ds = safe_prepare_dataset(broad, dataset_configs[behavior])
        if ds is None:
            continue
        model = base_process_factories[behavior]()
        try:
            model.fit(ds)
        except Exception as e:
            tqdm.write(f"[{line}/{label}/{behavior}] fit failed: {e}")
            continue
        result[behavior] = (model, ds)

    return result


def plot_fish_gain_correlation_vehicle_vs_drug(
    line: str,
    loader, dataset_configs: dict, base_process_factories: dict,
    labels: Tuple[str, str] = ("vehicle", "ronidazole"),
    line_labels: Optional[Dict[str, Tuple[str, str]]] = None,
    min_behaviors: int = 2,
) -> Optional[plt.Figure]:
    line_labels = line_labels or {}
    veh_label, drug_label = line_labels.get(line, labels)

    models_veh = build_models_and_datasets_for_line(line, veh_label, loader, dataset_configs, base_process_factories)
    models_drug = build_models_and_datasets_for_line(line, drug_label, loader, dataset_configs, base_process_factories)

    return plot_fish_gain_correlation_dual(
        collect_fish_gains(models_veh), collect_fish_gains(models_drug),
        label_a="Vehicle", label_b="Drug",
        suptitle=f"{line}: fish-level frailty gain correlation, vehicle vs drug",
        min_behaviors=min_behaviors,
    )


