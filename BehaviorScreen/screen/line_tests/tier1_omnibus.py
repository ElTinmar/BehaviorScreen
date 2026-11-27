"""
Tier 1: omnibus test -- "does ablation change anything for this line and
behavior" -- via three ordinary, low-dimensional fits (reference-only,
test-only, pooled) and a PERMUTATION test on the resulting deviance
statistic (not the asymptotic chi-square LRT, given small-sample concerns
at typical line_reference/line_test arm sizes).

No MultiGroupProcess needed for this saturated 2-group comparison --
reference-only + test-only IS the saturated joint model, at lower
dimensionality.
"""
import copy
import json
import math
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram as tree_layout, leaves_list, linkage

from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.inference import newton_step_matrix, refit_from_fit
from BehaviorScreen.point_process.parallel import parallel_map, report_failures
from BehaviorScreen.point_process.io import save_fig, slugify
from BehaviorScreen.point_process.side_diagnostics import plot_time_and_trial_calibration
from BehaviorScreen.point_process.sided_process import CONTRA, IPSI, SIDE_NAMES, sided_base

from .dataset_utils import Arms, Contrast, per_fish_summary, compare_fish_metric, recording_days
from .gain_structure import GainStructure, fish_groups
from .arm_contrast import output_metric
from .fdr import ALPHA
from .model_qc import MIN_FISH, assess_group_fit
from .steps import cells_in, read_cell, write_cell
from .tier1_browser import write_browser
from BehaviorScreen.screen.line_names import line_name
from BehaviorScreen.screen.paths import line_tests_dir

# Behavior-specific minimum total-event floor. Fish count is the WRONG
# denominator for point-process/survival identifiability -- total events
# (or uncensored first-responses) is what actually matters. Survival-style,
# low-base-rate behaviors need a much stricter floor.
DEFAULT_MIN_EVENTS = 40
WEAK_QC_MARGIN_AIC = 5.0  # an arm's ΔAIC over the null below this: a weak margin (bad_fit_triage)
MIN_BOOT_SUCCESS = 0.3  # share of a parameter's bootstrap draws that must refit to report its interval
EXTEND_BELOW = 5  # permutations reaching the observed deviance, below which a cell not yet significant is permuted further
MAX_FAILED_PERMUTATIONS = 0.05  # share of failed permutation refits flagged in bad_fit_triage
MIN_EVENTS_BY_BEHAVIOR: Dict[str, int] = {
    "looming": 15, "dark_flash": 15,
}


def _total_events(dataset: PointProcessDataset) -> int:
    return int(dataset.stream_event_counts.sum()) if len(dataset.stream_event_counts) else 0


def fit_tier1(process_factory: Callable[[], PointProcess],
              ds_ref: PointProcessDataset, ds_test: PointProcessDataset
              ) -> Tuple[float, PointProcess, PointProcess, PointProcess]:
    """Three independent fits; deviance = 2*(LL_independent - LL_pooled),
    exactly the saturated-2-group joint-model comparison at lower
    dimensionality (mathematically identical target, see module docstring)."""
    m_ref = process_factory(); m_ref.fit(ds_ref)
    m_test = process_factory(); m_test.fit(ds_test)
    m_pooled = process_factory(); m_pooled.fit(PointProcessDataset.concat(ds_ref, ds_test))
    return tier1_deviance(m_ref, m_test, m_pooled), m_ref, m_test, m_pooled


def tier1_deviance(m_ref: PointProcess, m_test: PointProcess, m_pooled: PointProcess) -> float:
    return max(0.0, 2.0 * ((m_ref.log_likelihood + m_test.log_likelihood) - m_pooled.log_likelihood))


def permute_within(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation of range(len(strata)) that moves every index within its stratum."""
    grouped = np.argsort(strata, kind="stable")
    permutation = np.empty(len(strata), dtype=int)
    permutation[grouped] = grouped[np.lexsort((rng.random(len(strata)), strata[grouped]))]
    return permutation


def tier1_permutation_test(
    process_factory: Callable[[], PointProcess],
    ds_ref: PointProcessDataset, ds_test: PointProcessDataset,
    observed_deviance: Optional[float] = None,
    *, n_perm: int, first: int = 0, seed: int = 0, n_jobs: int = -1,
    pooled_log_likelihood: Optional[float] = None,
    strata: Optional[np.ndarray] = None,
) -> Dict:
    """
    Fish shuffled between the arms (sizes kept), each arm refitted: the null
    distribution of the deviance. The pooled fit is the same for every labelling
    (the same fish), so it is fitted once, or taken from the caller.
    strata: per fish (reference then test), e.g. its recording day: fish are
    shuffled within their stratum only, each keeping its arm sizes.
    first: permutations [first, first + n_perm) of the seed's sequence, to extend an earlier run of `first`.
    p_value_floor: the smallest p this run can give, set by the distinct splits or the permutations.
    """
    pooled = PointProcessDataset.concat(ds_ref, ds_test)
    if pooled_log_likelihood is None:
        m_pooled = process_factory(); m_pooled.fit(pooled)
        pooled_log_likelihood = m_pooled.log_likelihood

    def deviance(ref: PointProcessDataset, test: PointProcessDataset) -> float:
        m_ref = process_factory(); m_ref.fit(ref)
        m_test = process_factory(); m_test.fit(test)
        return max(0.0, 2.0 * (m_ref.log_likelihood + m_test.log_likelihood - pooled_log_likelihood))

    if observed_deviance is None:
        observed_deviance = deviance(ds_ref, ds_test)
    n_ref = ds_ref.num_fish
    strata = np.zeros(pooled.num_fish, dtype=int) if strata is None else pd.factorize(strata)[0]
    n_distinct = math.prod(math.comb(int(np.sum(strata == s)), int(np.sum(strata[:n_ref] == s))) for s in np.unique(strata))
    seeds = np.random.SeedSequence(seed).spawn(first + n_perm)[first:]

    def _one(s: np.random.SeedSequence) -> float | str:
        perm = permute_within(strata, np.random.default_rng(s))
        try:
            return deviance(pooled.fish(perm[:n_ref]), pooled.fish(perm[n_ref:]))
        except Exception as exc:
            return repr(exc)

    results = parallel_map(_one, seeds, n_jobs=n_jobs, desc="Permutation")
    report_failures("permutation", [r for r in results if isinstance(r, str)], n_perm)
    null_dist = np.array([v for v in results if not isinstance(v, str)])
    exceeding = int(np.sum(null_dist >= observed_deviance))
    unreliable = len(null_dist) < max(20, 0.3 * n_perm)
    return {"observed_deviance": observed_deviance, "p_value": np.nan if unreliable else (exceeding + 1) / (len(null_dist) + 1),
            "n_perm_success": len(null_dist), "n_perm_failed": n_perm - len(null_dist), "n_perm_exceeding": exceeding,
            "n_distinct_permutations": n_distinct, "p_value_floor": 1.0 / (min(n_distinct, len(null_dist)) + 1),
            "permutation_unreliable": unreliable}


def permutation_strata(contrast: Contrast, ds_ref: PointProcessDataset, ds_test: PointProcessDataset) -> Optional[np.ndarray]:
    """The recording day of each fish (reference then test) of an NTR contrast, shuffled within day; None otherwise."""
    return recording_days(np.concatenate([ds_ref.fish_ids, ds_test.fish_ids])) if contrast.family == "ntr" else None


def tier1_simulated_test(
    process_factory: Callable[[], PointProcess], m_pooled: PointProcess,
    ds_ref: PointProcessDataset, ds_test: PointProcessDataset, observed_deviance: float,
    structure: GainStructure, n_sim: int, seed: int = 0, n_jobs: int = -1,
) -> Dict:
    """
    The deviance's null distribution from datasets simulated from the pooled fit (no difference between
    the arms) with the contrast's fish, days and plates, the gains of the fish of a day and of a plate
    dependent as in structure: what the fish-label permutation leaves out when a plate holds one arm.
    """
    pooled, n_ref = PointProcessDataset.concat(ds_ref, ds_test), ds_ref.num_fish
    groups = fish_groups(pooled.fish_ids, "")

    def _one(s: np.random.SeedSequence) -> float | str:
        rng = np.random.default_rng(s)
        try:
            simulated = copy.deepcopy(m_pooled).simulate_dataset(pooled, rng, structure.latent(groups, rng))
            return fit_tier1(process_factory, simulated.fish(np.arange(n_ref)), simulated.fish(np.arange(n_ref, pooled.num_fish)))[0]
        except Exception as exc:
            return repr(exc)

    results = parallel_map(_one, np.random.SeedSequence(seed).spawn(n_sim), n_jobs=n_jobs, desc="Simulated null")
    report_failures("simulated null", [r for r in results if isinstance(r, str)], n_sim)
    null = np.array([v for v in results if not isinstance(v, str)])
    p = (np.sum(null >= observed_deviance) + 1) / (len(null) + 1) if len(null) >= max(20, 0.3 * n_sim) else np.nan
    return {"p_value_plate": float(p), "n_sim_success": len(null)}


def single_param_effect_size(m_ref: PointProcess, m_test: PointProcess, free_param: str,
                              eps: float = 1e-9) -> Dict:

    v_ref = m_ref.param_dict_.get(free_param)
    v_test = m_test.param_dict_.get(free_param)
    if v_ref is None or v_test is None:
        raise ValueError(f"{free_param} not found in one of the fitted models' param_dict_")

    return {
        "metric_name": f"param_{free_param}",
        "metric_reference": v_ref,
        "metric_test": v_test,
        "effect_size": float(np.log2(max(v_test, eps) / max(v_ref, eps))),
        "effect_size_type": "log2_fold_change",
    }

def tier1_effect_size(m_ref: PointProcess, ds_ref: PointProcessDataset,
                       m_test: PointProcess, ds_test: PointProcessDataset,
                       eps: float = 1e-9) -> Dict:

    # One free parameter: it is the effect size (nothing else for the effect to hide in, no saturation).
    if len(m_ref.param_names) == 1:
        return single_param_effect_size(m_ref, m_test, m_ref.param_names[0], eps=eps)
    
    val_ref, name = output_metric(m_ref, ds_ref)
    val_test, _ = output_metric(m_test, ds_test)

    if name == "response_probability":
        # Bounded metric (0,1) -- log2-ratio is NOT symmetric here (a
        # change from 0.5->0.9 and 0.5->0.1 are the "same size" in the
        # opposite direction but get very different |log2FC|; and a tiny
        # absolute change near 0 produces an enormous log2FC). Use the
        # logit difference instead: unbounded, symmetric under p<->1-p,
        # and the standard transform for exactly this situation (same
        # scale logistic regression coefficients live on).
        p_ref = np.clip(val_ref, eps, 1 - eps)
        p_test = np.clip(val_test, eps, 1 - eps)
        effect = float(np.log(p_test / (1 - p_test)) - np.log(p_ref / (1 - p_ref)))
        effect_name = "logit_diff"
    else:
        # Unbounded positive rate -- log2-ratio is appropriate here:
        # symmetric under doubling/halving, scale-invariant.
        effect = float(np.log2(max(val_test, eps) / max(val_ref, eps)))
        effect_name = "log2_fold_change"
    
    return {
        "metric_name": name, "metric_reference": val_ref, "metric_test": val_test,
        "effect_size": effect, "effect_size_type": effect_name,
    }

def _tier1_one(
    line: str, behavior: str, arms: Arms,
    null_process_factory: Callable[[], PointProcess],
    n_perm: int, structure: Optional[GainStructure], n_sim: int,
) -> Dict:
    contrast = arms.contrasts[line]
    ds_ref, ds_test = arms.datasets(line, behavior)

    if ds_ref is None or ds_test is None:
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_data", "p_value": np.nan}
    if ds_ref.num_fish < MIN_FISH or ds_test.num_fish < MIN_FISH:
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_data", "p_value": np.nan}
    if ds_ref.num_trials != ds_test.num_trials:
        return {"line": line, "behavior": behavior, "status": "skipped_trial_count_mismatch", "p_value": np.nan}

    min_events = MIN_EVENTS_BY_BEHAVIOR.get(behavior, DEFAULT_MIN_EVENTS)
    total_events = _total_events(ds_ref) + _total_events(ds_test)
    if total_events < min_events:
        return {"line": line, "behavior": behavior, "status": "skipped_insufficient_information",
                "total_events": total_events, "min_events_required": min_events, "p_value": np.nan}

    # --- Model QC gate on each arm before trusting any comparison ---
    exclude = arms.contrast_arms(line)
    qc_ref = assess_group_fit(partial(arms.model, *contrast.reference, behavior, exclude), null_process_factory, ds_ref, line, behavior,
                               "reference")
    qc_test = assess_group_fit(partial(arms.model, *contrast.test, behavior, exclude), null_process_factory, ds_test, line, behavior,
                                "test")

    # Only a HARD flag on either arm excludes the cell. "reliable_no_signal"
    # (architecture didn't beat null, but fit is otherwise trustworthy)
    # proceeds to the real fit_tier1 comparison below -- see GroupFitQC.verdict.
    if qc_ref.verdict == "flagged" or qc_test.verdict == "flagged":
        fallback_records = []
        try:
            sum_ref, sum_test = per_fish_summary(ds_ref), per_fish_summary(ds_test)
            for metric in ["rate_hz", "response_prob", "mean_first_latency_s"]:
                fallback_records.append(compare_fish_metric(sum_ref, sum_test, metric))
        except Exception:
            pass
        return {
            "line": line, "behavior": behavior, "status": "flagged_architecture_collapse",
            "qc_reasons_reference": qc_ref.reasons_flagged, "qc_reasons_test": qc_test.reasons_flagged,
            "qc_notes_reference": qc_ref.informational_notes, "qc_notes_test": qc_test.informational_notes,
            "fallback_tier0": fallback_records, "p_value": np.nan,
        }

    # --- Real fit + permutation test ---
    try:
        m_ref, m_test = arms.models(line, behavior)
        m_pooled = arms.process_factory(line, behavior)()
        m_pooled.fit(PointProcessDataset.concat(ds_ref, ds_test))
        observed_deviance = tier1_deviance(m_ref, m_test, m_pooled)
    except Exception as e:
        return {"line": line, "behavior": behavior, "status": f"fit_failed: {type(e).__name__}: {e}",
                "p_value": np.nan}

    perm = tier1_permutation_test(arms.process_factory(line, behavior), ds_ref, ds_test,
                                   observed_deviance=observed_deviance, n_perm=n_perm,
                                   pooled_log_likelihood=m_pooled.log_likelihood, strata=permutation_strata(contrast, ds_ref, ds_test))
    eff = tier1_effect_size(m_ref, ds_ref, m_test, ds_test)
    estimates = {"reference": arm_estimates(m_ref, qc_ref.hessian), "test": arm_estimates(m_test, qc_test.hessian)}
    if structure is not None and contrast.family == "ntr":
        perm |= tier1_simulated_test(arms.process_factory(line, behavior), m_pooled, ds_ref, ds_test, observed_deviance, structure, n_sim)

    return {
        "line": line, "behavior": behavior, "status": "ok",
        "n_fish_ref": ds_ref.num_fish, "n_fish_test": ds_test.num_fish,
        "total_events": total_events,
        # Surfaced for reporting -- lets you see, per arm, whether the
        # architecture found real structure or collapsed to null-like
        # behavior, without that fact ever having excluded the cell.
        "qc_verdict_reference": qc_ref.verdict,
        "qc_verdict_test": qc_test.verdict,
        "qc_beats_null_reference": qc_ref.beats_null,
        "qc_beats_null_test": qc_test.beats_null,
        "qc_delta_aic_reference": qc_ref.delta_aic,
        "qc_delta_aic_test": qc_test.delta_aic,
        "qc_notes_reference": qc_ref.informational_notes,
        "qc_notes_test": qc_test.informational_notes,
        **perm, **eff, "arm_estimates": estimates,
    }

def fish_ids(*datasets: Optional[PointProcessDataset]) -> List[List[str]]:
    return [[] if ds is None else sorted(map(str, ds.fish_ids)) for ds in datasets]


def same_fish(record: Dict, ds_ref: Optional[PointProcessDataset], ds_test: Optional[PointProcessDataset]) -> bool:
    """Whether a saved cell was run on the arms' current fish: their ids, or their numbers for a cell saved without them."""
    fish = fish_ids(ds_ref, ds_test)
    if "fish_ids" in record:
        return record["fish_ids"] == fish
    return [record.get("n_fish_ref"), record.get("n_fish_test")] == [len(f) for f in fish]


def cell_path(store: Path, line: str, behavior: str) -> Path:
    return store / behavior / f"{slugify(line)}.json"


def save_cell(store: Path, record: Dict) -> None:
    write_cell(cell_path(store, record["line"], record["behavior"]), record)


def saved_cells(folder: Path, key: Dict) -> List[Dict]:
    """The cells in folder run with this key; any run with another (model or control fit) are deleted."""
    folder.mkdir(parents=True, exist_ok=True)
    key_path = folder / "key.json"
    if key_path.exists() and json.loads(key_path.read_text()) != key:
        for path in cells_in(folder):
            path.unlink()
    key_path.write_text(json.dumps(key))
    return [read_cell(path) for path in cells_in(folder)]


def refuse_other_settings(folder: Path, key: Dict) -> None:
    """Stops when the cells in folder were run with another n_perm or n_sim (a changed model or control fit replaces them)."""
    settings = lambda k: {"n_perm": k["n_perm"], "n_sim": k["structure"][-1] if k["structure"] else 0}
    path = folder / "key.json"
    if path.exists() and (saved := settings(json.loads(path.read_text()))) != settings(key):
        raise SystemExit(f"{folder}: its cells were run with {saved}, not {settings(key)}. Delete the folder to rerun them.")


def extend_permutations(tier1_df: pd.DataFrame, arms: Arms, n_perm_max: int, store: Optional[Path] = None) -> pd.DataFrame:
    """
    The 'ok' cells not significant (add_fdr_per_behavior) whose observed deviance fewer than EXTEND_BELOW
    permutations reached, so that the permutations rather than the fish limit their p: permuted further, up to
    n_perm_max in all, continuing the same sequence of permutations, and saved with the cell. FDR is then to redo.
    """
    df = tier1_df.copy()
    run = df["n_perm_success"] + df["n_perm_failed"]
    estimated = (df["p_value"] * (df["n_perm_success"] + 1) - 1).round()  # cells saved without the count
    exceeding = df["n_perm_exceeding"].fillna(estimated) if "n_perm_exceeding" in df else estimated
    todo = (df["status"] == "ok") & ~df["significant"] & (exceeding < EXTEND_BELOW) & (run < n_perm_max) & (df["n_distinct_permutations"] > run)
    for i in tqdm(df.index[todo], desc=f"Extending to {n_perm_max} permutations", unit="cell"):
        row, contrast = df.loc[i], arms.contrasts[df.loc[i, "line"]]
        ds_ref, ds_test = arms.datasets(row["line"], row["behavior"])
        more = tier1_permutation_test(arms.process_factory(row["line"], row["behavior"]), ds_ref, ds_test, row["observed_deviance"],
                                      n_perm=n_perm_max - int(run[i]), first=int(run[i]), strata=permutation_strata(contrast, ds_ref, ds_test))
        if more["permutation_unreliable"]:
            continue
        success, k = int(row["n_perm_success"]) + more["n_perm_success"], int(exceeding[i]) + more["n_perm_exceeding"]
        update = {"p_value": (k + 1) / (success + 1), "n_perm_success": success, "n_perm_exceeding": k,
                  "n_perm_failed": int(row["n_perm_failed"]) + more["n_perm_failed"]}
        for column, value in update.items():
            df.loc[i, column] = value
        if store is not None:
            save_cell(store, {**read_cell(cell_path(store, row["line"], row["behavior"])), **update})
    return df


def run_tier1_screen(
    arms: Arms, behaviors: List[str], null_process_factories: Dict[str, Callable[[], PointProcess]],
    n_perm: int, n_sim: int, show_progress: bool = True, store: Optional[Path] = None,
    structures: Optional[Dict[str, GainStructure]] = None,
) -> pd.DataFrame:
    """
    One row per (contrast, behavior); the contrast name is in column 'line'.
    structures: per behavior, the dependence between fish of a day and of a plate; NTR contrasts then
    also get p_value_plate, from n_sim datasets simulated with it (tier1_simulated_test).
    store: each row saved there as soon as it is done (store/<behavior>/), and
    reused while the behavior's model and control fit are the same, and the cell's fish (same_fish). Saved with
    other n_perm or n_sim, they are kept and the run refuses: delete them to rerun.
    """
    jobs = [(name, behavior) for name in arms.contrasts for behavior in behaviors]
    done = {}
    for behavior in behaviors if store is not None else []:
        architecture = arms.process_factories[behavior]
        key = {"model": architecture.template.name, "fixed": architecture.fixed, "n_perm": n_perm, "pinning": "control without the contrast's arms",
               "permutation": "NTR arms: days recorded in both, fish shuffled within day",
               "structure": None if not structures else [structures[behavior].line, structures[behavior].day, structures[behavior].plate, n_sim]}
        refuse_other_settings(store / behavior, key)
        saved = {(r["line"], behavior): r for r in saved_cells(store / behavior, key)}
        done.update({job: r for job, r in saved.items() if same_fish(r, *arms.datasets(*job))})
        if changed := len(saved) - len(done):
            print(f"[{behavior}] {changed} saved cells recomputed: their fish changed")
    todo = [job for job in jobs if job not in done]
    cell_iter = tqdm(todo, desc="Tier 1 cells", position=0, unit="cell") if show_progress else todo
    for name, behavior in cell_iter:
        try:
            record = _tier1_one(name, behavior, arms, null_process_factories[behavior], n_perm=n_perm,
                                structure=(structures or {}).get(behavior), n_sim=n_sim)
        except Exception as e:
            record = {"line": name, "behavior": behavior,
                      "status": f"unexpected_error: {type(e).__name__}: {e}", "p_value": np.nan}
        c = arms.contrasts[name]
        fish = fish_ids(*arms.datasets(name, behavior))
        done[name, behavior] = {"family": c.family, "reference": "/".join(c.reference), "test": "/".join(c.test),
                                **record, "n_fish_ref": len(fish[0]), "n_fish_test": len(fish[1]), "fish_ids": fish}
        if store is not None:
            save_cell(store, done[name, behavior])

    return pd.DataFrame([done[job] for job in jobs])


def log2_scale(param_df: pd.DataFrame) -> pd.Series:
    """Per row: whether its (behavior, parameter) column is compared as a log2 ratio, its values being all positive."""
    values = param_df[["value_reference", "value_test"]]
    df = param_df.assign(positive=(values.gt(0) | values.isna()).all(axis=1), any_value=values.notna().any(axis=1))
    columns = df.groupby(["behavior", "parameter"])
    return columns["positive"].transform("all") & columns["any_value"].transform("any")


def at_bound(param_df: pd.DataFrame) -> pd.Series:
    """Per row: an arm without a standard error, its parameter at a bound (estimate_hessian), or its Hessian singular."""
    return param_df[["se_reference", "se_test"]].isna().any(axis=1) & param_df[["value_reference", "value_test"]].notna().all(axis=1)


def compute_parameter_deltas(param_df: pd.DataFrame) -> pd.DataFrame:
    """
    'delta' (test vs reference) and its standard error 'delta_se'.
    log2-ratio for parameters positive across the whole screen (rate
    scales: B, A, H), raw difference otherwise; chosen per (behavior,
    parameter) column (log2_scale).
    """
    log2, ref, test = log2_scale(param_df), param_df["value_reference"], param_df["value_test"]
    se_ref, se_test = param_df["se_reference"], param_df["se_test"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return param_df.assign(
            delta=np.where(log2, np.log2(test / ref), test - ref),
            delta_se=np.where(log2, np.hypot(se_ref / ref, se_test / test) / np.log(2), np.hypot(se_ref, se_test)),
        )


def standard_errors(model: PointProcess, hessian: Optional[np.ndarray]) -> Dict[str, float]:
    """From the inverse of the fit's Hessian; NaN for parameters at a bound, and for all if it is singular or missing."""
    se = np.full(len(model.param_names), np.nan)
    if hessian is not None:
        interior = np.isfinite(np.diag(hessian))
        try:
            cov = np.linalg.inv(hessian[np.ix_(interior, interior)])
            se[interior] = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
        except np.linalg.LinAlgError:
            pass
    return dict(zip(model.param_names, se))


def arm_estimates(model: PointProcess, hessian: Optional[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """An arm's fitted parameters and their standard errors, saved with its cell (extract_fitted_parameters)."""
    return {"value": dict(model.param_dict_), "se": standard_errors(model, hessian)}


def extract_fitted_parameters(tier1_df: pd.DataFrame) -> pd.DataFrame:
    """Free parameters of both arms of every 'ok' cell, with the standard errors saved with it (arm_estimates)."""
    records = []
    for _, row in tier1_df[tier1_df["status"] == "ok"].iterrows():
        ref, test = row["arm_estimates"]["reference"], row["arm_estimates"]["test"]
        for pname in ref["value"]:
            records.append({
                "family": row["family"], "line": row["line"], "behavior": row["behavior"], "parameter": pname,
                "value_reference": ref["value"][pname], "value_test": test["value"][pname],
                "se_reference": ref["se"][pname], "se_test": test["se"][pname],
                "significant": bool(row.get("significant", False)),
            })

    columns = ["family", "line", "behavior", "parameter", "value_reference", "value_test", "se_reference", "se_test", "significant"]
    return pd.DataFrame(records, columns=columns)


def _dot_size(se: float, se_half_size: float) -> float:
    return 160.0 / (1.0 + (se / se_half_size) ** 2)


def reference_sds(param_df: pd.DataFrame, tier1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per reference arm (each once) and parameter: its SD between fish, standard error times the square root of its fish,
    on delta's scale (log2_scale); NaN for an arm at a bound (no standard error).
    """
    df = param_df.merge(tier1_df.drop_duplicates(["line", "behavior"])[["line", "behavior", "reference", "n_fish_ref"]],
                        on=["line", "behavior"], how="left")
    se = np.where(log2_scale(df), df["se_reference"] / df["value_reference"] / np.log(2), df["se_reference"])
    arms = df.assign(sd=se * np.sqrt(df["n_fish_ref"])).drop_duplicates(["reference", "behavior", "parameter"])
    return arms[["behavior", "parameter", "reference", "sd"]].reset_index(drop=True)


def control_sds(arms: pd.DataFrame) -> pd.DataFrame:
    """Per column (behavior, parameter) of reference_sds: sigma, their median, the arms it rests on and those at a bound."""
    keys = ["behavior", "parameter"]
    bound = arms[arms["sd"].isna()].groupby(keys)["reference"].agg(", ".join).rename("at_bound")
    table = arms.groupby(keys, sort=False)["sd"].agg(sigma="median", n_arms="count", q10=lambda v: v.quantile(0.1), q90=lambda v: v.quantile(0.9))
    return table.join(bound).fillna({"at_bound": ""}).reset_index()


def standardized_effects(param_df: pd.DataFrame, tier1_df: pd.DataFrame) -> pd.DataFrame:
    """
    'delta' over its column's SD between control fish (Glass's delta, one SD per behavior and parameter): control_sds'
    sigma. So a column shares one scale, and a cell with an arm at a bound (no standard error) has a value.
    """
    sigma = param_df[["behavior", "parameter"]].merge(control_sds(reference_sds(param_df, tier1_df)), how="left")["sigma"]
    return param_df.assign(delta=param_df["delta"].to_numpy() / sigma.to_numpy())


def plot_control_sds(arms: pd.DataFrame, behaviors: List[str], column_labels: Dict[str, str]) -> plt.Figure:
    """
    Each column's reference arms (reference_sds) relative to its sigma, on a log scale: dots near 1 mean control fish
    vary alike across lines, as Glass's delta assumes. Arms beyond twice or under half sigma are named; below each
    column, its arms at a bound (left out).
    """
    sigma = control_sds(arms).set_index(["behavior", "parameter"])
    columns = [(b, p) for b in behaviors for p in PARAMETER_LABELS if (b, p) in sigma.index]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(columns) + 1.5), 4.5))
    for x, (behavior, p) in enumerate(columns):
        column = arms[(arms["behavior"] == behavior) & (arms["parameter"] == p)].dropna(subset=["sd"])
        ratio = column["sd"].to_numpy() / sigma.loc[(behavior, p), "sigma"]
        ax.scatter(x + np.random.default_rng(x).uniform(-0.15, 0.15, len(ratio)), ratio, s=12, color="#4d4d4d", alpha=0.7, linewidths=0)
        for r, reference in zip(ratio, column["reference"]):
            if not 0.5 <= r <= 2:
                ax.annotate(line_name(reference.split("/")[0]), (x, r), xytext=(4, 0), textcoords="offset points", fontsize=5, va="center")
        if n := len([a for a in sigma.loc[(behavior, p), "at_bound"].split(", ") if a]):
            ax.text(x, 0.27, f"{n} at\nbound", ha="center", va="top", fontsize=5, color="#b2182b")
    for level, style in ((1, "-"), (0.5, ":"), (2, ":")):
        ax.axhline(level, color="#9e9e9e", linewidth=0.8, linestyle=style, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 5)
    ax.set_yticks([0.25, 0.5, 1, 2, 4], ["¼", "½", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xticks(range(len(columns)), [f"{column_labels.get(b, b)} · {PARAMETER_LABELS[p]}" for b, p in columns], rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylabel("reference arm's SD between fish / column sigma")
    ax.set_title("Control spread behind Glass's delta: one dot per reference arm", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# Column label of each amplitude parameter in the map.
PARAMETER_LABELS = {"B": "sustained", "A": "transient", "A_peak": "transient", "H": "escape", "D": "direction", "P": "direction", "E": "escape", "G": "transient gain", "R": "rate", "s_side": "side spread"}
MAP_COLORS = LinearSegmentedColormap.from_list("blue_grey_red", ["#2166ac", "#d9d9d9", "#b2182b"])
MISSING = "#9e9e9e"
POSTHOC_BACKGROUND = "#fbe7a1"  # off the colour scale, whose middle is grey
EFFECT_VMAX = 3.0  # colour scale of the standardized Tier 1 map, in control fish SDs
EFFECT_LABEL = "effect size: Glass's Δ (SDs of control fish)"
BOUND_LABEL = "an arm at a bound of its parameter (no standard error)"


def add_testability(tier1_df: pd.DataFrame, n_perm_max: Optional[int] = None) -> pd.DataFrame:
    """
    'testable': the cell could reach q < ALPHA. Its best possible q is its
    floor ('p_value_floor') times the number of tests in its FDR family
    (family x behavior). n_perm_max: the most permutations a cell gets
    (extend_permutations); the floor is then 1 / (that or its number of
    distinct fish splits, whichever is smaller, + 1).
    """
    df = tier1_df.copy()
    if n_perm_max is not None:
        df["p_value_floor"] = 1.0 / (df["n_distinct_permutations"].astype(float).clip(upper=n_perm_max) + 1)  # counts beyond int64: objects
    keys = ["family", "behavior"] if "family" in df.columns else ["behavior"]
    n_tests = df.groupby(keys)["p_value"].transform("count")
    df["testable"] = (df["p_value_floor"] * n_tests < ALPHA).fillna(False)
    return df


def map_rows(tier1_df: pd.DataFrame, contrasts: Iterable[Contrast]) -> List[str]:
    """Rows of the map: WT, the NTR lines with the most hits first, then the mutants."""
    contrasts = list(contrasts)
    hits = tier1_df[tier1_df["significant"]].groupby("line").size()
    ntr = [c.name for c in contrasts if c.family == "ntr" and c.name != "WT"]
    return ["WT"] + sorted(ntr, key=lambda line: -hits.get(line, 0)) + [c.name for c in contrasts if c.family == "mutant"]


ROW_ORDER = "tier1_rows.json"  # a variant's map rows (save_tier1_maps), which the other variants' maps follow
REFERENCE_VARIANTS = ("best", "best-poisson")  # whose row order every map follows: the first one combined


def clustered_rows(effects: pd.DataFrame, tier1_df: pd.DataFrame, contrasts: Iterable[Contrast]) -> Dict:
    """
    {"rows", "clustered", "linkage"}: WT, the NTR lines with a tested cell clustered by their Glass's delta
    (standardized_effects, clipped to the map's ±EFFECT_VMAX, untested cells 0; Euclidean distance, average linkage,
    optimal leaf order), the other NTR lines, then the mutants; the clustered lines in order and their tree.
    """
    contrasts = list(contrasts)
    ntr = [c.name for c in contrasts if c.family == "ntr" and c.name != "WT"]
    tested = set(tier1_df.loc[tier1_df["status"] == "ok", "line"]) & set(ntr)
    table = effects[effects["line"].isin(tested) & effects["parameter"].isin(PARAMETER_LABELS)]
    X = table.pivot_table(index="line", columns=["behavior", "parameter"], values="delta").clip(-EFFECT_VMAX, EFFECT_VMAX).fillna(0)
    Z = linkage(X.to_numpy(), "average", optimal_ordering=True) if len(X) >= 3 else None
    clustered = list(X.index) if Z is None else [X.index[i] for i in leaves_list(Z)]
    rows = ["WT"] + clustered + [line for line in ntr if line not in clustered] + [c.name for c in contrasts if c.family == "mutant"]
    return {"rows": rows, "clustered": clustered, "linkage": None if Z is None else Z.tolist()}


def shared_rows(order: Dict, out: Path) -> Dict:
    """
    order (clustered_rows) saved in out, a variant's folder of line_tests; the order of the first REFERENCE_VARIANTS
    combined instead, if it is another folder's, so that every map lists the lines alike (lines it lacks: appended).
    """
    (out / ROW_ORDER).write_text(json.dumps(order))
    for variant in REFERENCE_VARIANTS:
        path = line_tests_dir(variant, out.parent) / ROW_ORDER
        if path.exists():
            if path.parent.resolve() == out.resolve():
                return order
            shared = json.loads(path.read_text())
            return {**shared, "rows": [r for r in shared["rows"] if r in order["rows"]] + [r for r in order["rows"] if r not in shared["rows"]]}
    return order


def plot_tier1_map(
    param_df: pd.DataFrame,
    tier1_df: pd.DataFrame,
    behaviors: List[str],
    rows: List[str],
    vmax: float = 4.0,
    se_half_size: Optional[float] = 0.5,
    row_labels: Optional[Dict[str, str]] = None,
    colorbar_label: str = "log2(test / reference)",
    column_labels: Optional[Dict[str, str]] = None,
    frame_label: str = f"significant (q < {ALPHA:g})",
    posthoc: Optional[pd.DataFrame] = None,
    tree: Optional[Dict] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Matrix of the screen: rows are contrasts (in the given order, a gap
    between families), columns are each behavior's free parameters.
    Colour: param_df's 'delta' on a fixed scale centred on 0 (grey: none).
    Dots whose area shrinks with 'delta_se' (half size at se_half_size), or,
    with se_half_size None, squares. Frame: behavior significant for that
    contrast. Pale background (dots) or centre dot (squares): parameter
    significant in posthoc (its 'significant' rows by line, behavior and
    parameter). Hollow dot or hatched square: the cell could not reach
    significance with its fish ('testable' False). Diamond or "0": an arm at a
    bound (at_bound), drawn at a fixed size. Grey x: no test.
    With n_fish_ref / n_fish_test columns, a matching panel shows both arms'
    fish counts per cell: reference in the upper-left triangle, test in the
    lower-right one, each shaded by its count. tree (clustered_rows): the dendrogram of its clustered lines, on the left,
    where they are consecutive rows in that order.
    The returned axes' `cells` maps each (line, behavior, parameter) drawn to its centre in data coordinates (cell_boxes).
    """
    present = param_df[param_df["parameter"].isin(PARAMETER_LABELS)]
    columns = [
        (behavior, p) for behavior in behaviors for p in PARAMETER_LABELS
        if ((present["behavior"] == behavior) & (present["parameter"] == p)).any()
    ]
    contrasts = tier1_df.drop_duplicates("line").set_index("line")
    y, positions, gaps = 0.0, {}, []
    for i, name in enumerate(rows):
        if i and contrasts.loc[name, "family"] != contrasts.loc[rows[i - 1], "family"]:
            gaps.append(y)
            y += 1.0
        positions[name] = y
        y += 1.0
    x_of = {column: i for i, column in enumerate(columns)}
    cell_info = tier1_df.set_index(["line", "behavior"])
    show_n = {"n_fish_ref", "n_fish_test"} <= set(tier1_df.columns)
    n_fish = cell_info[["n_fish_ref", "n_fish_test"]] if show_n else None

    def label(name: str) -> str:
        return (row_labels or {}).get(name) or (
            line_name(name) if contrasts.loc[name, "family"] == "ntr"
            else f"{line_name(name)} (vs {line_name(contrasts.loc[name, 'reference'].split('/')[0])})"
        )

    # Layout in inches: one 0.32 in unit per column and row, margins from label lengths.
    shown = [b for b in behaviors if any(c[0] == b for c in columns)]
    widths = [len(columns), len(shown)] if show_n else [len(columns)]
    unit, gap, right, bottom = 0.32, 0.4, 1.3, 1.3
    clustered = (tree or {}).get("clustered", [])
    drawn_tree = (tree or {}).get("linkage") is not None and all(positions.get(n) == positions.get(clustered[0], -1) + i for i, n in enumerate(clustered))
    tree_w = 0.9 if drawn_tree else 0.0
    left = tree_w + 0.2 + 0.065 * max(len(label(n)) for n in rows)
    title = lambda behavior: (column_labels or {}).get(behavior, behavior)
    top = 0.3 + 0.065 * max(len(title(b)) for b in shown)
    content_w, bar_h = unit * sum(widths) + gap * (len(widths) - 1), max(1.2, min(2.5, unit * y))
    fig_w, fig_h = max(left + content_w + right, 6.5), top + max(unit * y, bar_h) + bottom
    right = fig_w - left - content_w
    fig, axes = plt.subplots(
        1, len(widths), figsize=(fig_w, fig_h), sharey=True, squeeze=False,
        gridspec_kw={"width_ratios": widths, "wspace": gap / (unit * sum(widths) / len(widths))},
    )
    fig.subplots_adjust(left=left / fig_w, right=1 - right / fig_w, top=1 - top / fig_h, bottom=1 - (top + unit * y) / fig_h)
    ax = axes[0, 0]
    cells = present.set_index(["line", "behavior", "parameter"])
    tested = tier1_df[tier1_df["status"] == "ok"].set_index(["line", "behavior"])
    significant = set() if posthoc is None else set(posthoc.loc[posthoc["significant"] == True, ["line", "behavior", "parameter"]].itertuples(index=False, name=None))
    norm = plt.Normalize(-vmax, vmax)
    bounded = at_bound(cells) if {"se_reference", "se_test"} <= set(cells) else pd.Series(dtype=bool)
    for name in rows:
        for behavior, p in columns:
            x, yy = x_of[(behavior, p)], positions[name]
            if (name, behavior) not in tested.index or (name, behavior, p) not in cells.index:
                ax.plot(x, yy, marker="x", color=MISSING, markersize=4, linestyle="")
                continue
            cell, marked, floor = cells.loc[(name, behavior, p)], (name, behavior, p) in significant, bounded.get((name, behavior, p), False)
            testable = bool(tested.loc[(name, behavior)].get("testable", True))
            color = MAP_COLORS(norm(cell["delta"])) if np.isfinite(cell["delta"]) else MISSING
            if se_half_size is None:
                ax.add_patch(plt.Rectangle((x - 0.4, yy - 0.4), 0.8, 0.8, facecolor=color, hatch=None if testable else "////", edgecolor="white", linewidth=0))
                if marked:
                    ax.plot(x, yy, marker="o", markersize=3.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.6, linestyle="")
                if floor:
                    ax.text(x + 0.38, yy + 0.38, "0", ha="right", va="bottom", fontsize=6, color="black")
                continue
            if marked:
                ax.add_patch(plt.Rectangle((x - 0.4, yy - 0.4), 0.8, 0.8, facecolor=POSTHOC_BACKGROUND, linewidth=0, zorder=0))
            size = _dot_size(se_half_size if floor else cell["delta_se"] if np.isfinite(cell["delta_se"]) else 4 * se_half_size, se_half_size)
            ax.scatter(x, yy, s=size, marker="D" if floor else "o", facecolors=color if testable else "white", edgecolors=color,
                       linewidths=0 if testable else 0.3 * np.sqrt(size))

    for (name, behavior), row in tested.iterrows():
        if name not in positions or not row.get("significant", False):
            continue
        xs_b = [x_of[c] for c in columns if c[0] == behavior]
        if xs_b:
            ax.add_patch(plt.Rectangle((min(xs_b) - 0.47, positions[name] - 0.47), max(xs_b) - min(xs_b) + 0.94, 0.94,
                                       fill=False, linewidth=1.2, edgecolor="black"))

    few_label = f"too few fish to reach q < {ALPHA:g}"
    handles = [plt.Line2D([], [], marker="s", markersize=11, markerfacecolor="none", markeredgecolor="black", linestyle="", label=frame_label)]
    if posthoc is not None:
        posthoc_label = f"parameter significant (post hoc, adjusted p < {ALPHA:g})"
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=POSTHOC_BACKGROUND, linewidth=0, label=posthoc_label) if se_half_size else
                       plt.Line2D([], [], marker="o", markersize=3.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.6, linestyle="", label=posthoc_label))
    if se_half_size is None:
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor=MISSING, hatch="////", edgecolor="white", linewidth=0, label=few_label),
                    plt.Rectangle((0, 0), 1, 1, facecolor=MISSING, linewidth=0, label="no estimate"),
                    plt.Line2D([], [], marker="$0$", color="black", markersize=5, linestyle="", label=BOUND_LABEL)]
    else:
        handles += [plt.Line2D([], [], marker="o", markersize=np.sqrt(_dot_size(se, se_half_size)), color=MISSING, linestyle="", label=f"SE {se:g}")
                    for se in (0.1, se_half_size, 2 * se_half_size)]
        handles.append(plt.Line2D([], [], marker="o", markersize=8, markerfacecolor="white", markeredgecolor=MISSING, markeredgewidth=2.4, linestyle="", label=few_label))
        handles.append(plt.Line2D([], [], marker="D", markersize=np.sqrt(_dot_size(se_half_size, se_half_size)), color=MISSING, linestyle="", label=BOUND_LABEL))
    handles.append(plt.Line2D([], [], marker="x", color=MISSING, linestyle="", label="not tested"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(left / fig_w, 0.1 / fig_h), ncol=3, frameon=False, fontsize=7)

    def decorate(axis: plt.Axes, cols: List[Tuple[str, str]]) -> None:
        starts = [i for i, c in enumerate(cols) if i == 0 or c[0] != cols[i - 1][0]]
        for i in starts[1:]:
            axis.axvline(i - 0.5, color="#bdbdbd", linewidth=0.6)
        for gap in gaps:
            axis.axhline(gap, color="#bdbdbd", linewidth=0.6)
        axis.set_xticks(range(len(cols)), [PARAMETER_LABELS.get(p, "") for _, p in cols], rotation=90, fontsize=7)
        top = axis.secondary_xaxis("top")
        top.spines["top"].set_visible(False)
        top.tick_params(length=0)
        top.set_xticks([(i + j - 1) / 2 for i, j in zip(starts, starts[1:] + [len(cols)])],
                       [title(cols[i][0]) for i in starts], rotation=90, fontsize=8)
        axis.set_xlim(-0.6, len(cols) - 0.4)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)


    decorate(ax, columns)
    ax.set_yticks([positions[n] for n in rows], [label(n) for n in rows], fontsize=8)
    ax.set_ylim(y - 0.4, -0.6)
    if drawn_tree:  # right of the tree: the row labels' left edge
        fig.canvas.draw()
        labels_x0 = min(t.get_window_extent().x0 for t in ax.get_yticklabels()) / fig.dpi / fig_w
        box = ax.get_position()
        ax_tree = fig.add_axes([labels_x0 - (tree_w - 0.1) / fig_w, box.y0, (tree_w - 0.15) / fig_w, box.height], sharey=ax)
        layout = tree_layout(np.asarray(tree["linkage"]), no_plot=True)
        for xs, heights in zip(layout["icoord"], layout["dcoord"]):
            ax_tree.plot(heights, positions[clustered[0]] + (np.asarray(xs) - 5) / 10, color="0.45", linewidth=0.8)
        ax_tree.invert_xaxis()
        ax_tree.axis("off")
    cax = fig.add_axes([(left + content_w + 0.15) / fig_w, 1 - (top + bar_h) / fig_h, 0.12 / fig_w, bar_h / fig_h])
    fig.colorbar(plt.cm.ScalarMappable(norm, MAP_COLORS), cax=cax, label=colorbar_label)

    if show_n:
        ax_n = axes[0, 1]
        top_n = max(float(n_fish.max().max()), 1.0)
        for c, behavior in enumerate(shown):
            for name in rows:
                if (name, behavior) not in n_fish.index or not np.isfinite(n_fish.loc[(name, behavior)]).all():
                    continue
                x0, x1, top, bottom = c - 0.45, c + 0.45, positions[name] - 0.45, positions[name] + 0.45
                triangles = (((x0, top), (x1, top), (x0, bottom)), ((x1, top), (x1, bottom), (x0, bottom)))
                for n, corners, sign in zip(n_fish.loc[(name, behavior)], triangles, (-1, 1)):
                    shade = 0.92 - 0.6 * n / top_n
                    ax_n.add_patch(plt.Polygon(corners, color=str(shade), linewidth=0))
                    ax_n.text(c + sign * 0.2, positions[name] + sign * 0.2, f"{int(n)}", ha="center", va="center",
                              fontsize=5.5, color="white" if shade < 0.55 else "black")
        decorate(ax_n, [(b, "") for b in shown])
        ax_n.set_xlabel("fish: ↖ reference, test ↘", fontsize=8)
    ax.cells = {(name, b, p): (x_of[(b, p)], positions[name]) for name in rows for b, p in columns
                if (name, b, p) in cells.index and (name, b) in tested.index}
    return fig, ax

def save_tier1_maps(
    param_df: pd.DataFrame, tier1_df: pd.DataFrame, behaviors: List[str], contrasts: Iterable[Contrast], out: Path,
    column_labels: Dict[str, str], posthoc: pd.DataFrame,
) -> None:
    """
    tier1_map: Glass's delta (standardized_effects); tier1_sigma (.png, .csv): the control spread behind it. Mutant and reference are two clutches: their
    fish-label test is not framed (clutch_variability frames it). Rows: clustered_rows, shared across variants
    (shared_rows), with their dendrogram. Then the page of all variants' maps, in out's parent
    (tier1_browser: out is a variant's folder of line_tests).
    """
    arms = reference_sds(param_df, tier1_df)
    control_sds(arms).to_csv(out / "tier1_sigma.csv", index=False)
    siblings = tier1_df.assign(significant=tier1_df["significant"] & (tier1_df["family"] == "ntr"))
    effects = standardized_effects(param_df, tier1_df)
    order = shared_rows(clustered_rows(effects, tier1_df, contrasts), out)
    rows = order["rows"]
    common = dict(column_labels=column_labels, frame_label=f"significant (q < {ALPHA:g}; NTR lines)", posthoc=posthoc, tree=order)
    fig, ax = plot_tier1_map(effects, siblings, behaviors, rows, vmax=EFFECT_VMAX, se_half_size=None, colorbar_label=EFFECT_LABEL, **common)
    (out / "tier1_map.json").write_text(json.dumps({"cells": cell_boxes(fig, ax)}))
    save_fig(fig, out, "tier1_map")
    plt.close(fig)
    save_fig(fig := plot_control_sds(arms, behaviors, column_labels), out, "tier1_sigma")
    plt.close(fig)
    print(f"Tier 1 maps of every variant: {write_browser(out.parent)}")


def cell_boxes(fig: plt.Figure, ax: plt.Axes) -> List[Dict]:
    """
    Each cell of a plot_tier1_map (ax.cells) as its line, behavior, parameter and box in the saved PNG (save_fig's tight
    bounding box): [left, top, right, bottom], fractions of the image's width and height from its top-left corner.
    """
    fig.canvas.draw()
    image = fig.get_tightbbox(fig.canvas.get_renderer()).padded(plt.rcParams["savefig.pad_inches"])
    to_inches = ax.transData + fig.dpi_scale_trans.inverted()
    boxes = []
    for (line, behavior, parameter), (x, y) in ax.cells.items():
        (x0, y0), (x1, y1) = to_inches.transform([(x - 0.5, y - 0.5), (x + 0.5, y + 0.5)])
        box = [(min(x0, x1) - image.x0) / image.width, (image.y1 - max(y0, y1)) / image.height,
               (max(x0, x1) - image.x0) / image.width, (image.y1 - min(y0, y1)) / image.height]
        boxes.append({"line": line, "behavior": behavior, "parameter": parameter, "box": [round(float(v), 5) for v in box]})
    return boxes


def rate_label(model: PointProcess) -> str:
    """What the grids' surfaces and marginals count: a survival model's first events, else all."""
    return "first escapes (Hz per fish)" if model.is_survival else "events (Hz per fish)"


def fish_label(arm: str, dataset: PointProcessDataset) -> str:
    """An arm's title with the fish its surfaces rest on (those with a trial)."""
    return f"{arm} ({int(dataset.fish_trial_mask.any(axis=1).sum())} fish)"


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

def display_surfaces(model: PointProcess, dataset: PointProcessDataset, **side: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time edges, empirical, model) surfaces of the events model describes (Hz per fish), in the dataset's bins."""
    emp = model.described(dataset)
    emp = emp.select(laterality=side["side"]) if side else emp
    return dataset.t_grid, emp.time_trial_histogram_hz, model.compute_expected_rate(dataset, **side)


def _model_vmax(models: List[np.ndarray], empirical: List[np.ndarray]) -> float:
    """
    Top of the grids' colour scale: 1.5 times the model's 99th percentile (its expected rates), so the model sits
    below the top with room for the data's noise, and sparse empirical single events saturate.
    """
    top = 1.5 * float(np.nanpercentile(np.concatenate([m.ravel() for m in models]), 99))
    return top if top > 0 else max(1e-9, float(max(np.nanmax(e) for e in empirical)))

PREDICTIVE_SIMS = 100  # datasets simulated per arm for the grids' predictive ranges


def fish_rates(events: PointProcessDataset) -> np.ndarray:
    """Each fish's (with a trial) events per trial."""
    trials = events.fish_trial_mask.sum(axis=1)
    counts = np.bincount(events.event_fish_idx, minlength=events.num_fish)
    return counts[trials > 0] / trials[trials > 0]


def count_bins(events: PointProcessDataset) -> Tuple[int, int]:
    """(width, k): at most 10 bins of `width` events up to the data's 99th percentile, then a bin of k * width or more."""
    top = max(1.0, float(np.quantile(events.stream_event_counts, 0.99)))
    width = math.ceil(top / 10)
    return width, math.ceil(top / width)


def count_shares(events: PointProcessDataset, width: int, k: int) -> np.ndarray:
    """Share of the (fish, trial) streams in each count bin (count_bins)."""
    return np.bincount(np.minimum(events.stream_event_counts // width, k), minlength=k + 1) / len(events.stream_event_counts)


def count_labels(width: int, k: int) -> List[str]:
    return [str(j * width) if width == 1 else f"{j * width}–{(j + 1) * width - 1}" for j in range(k)] + [f"{k * width}+"]


def ipsi_fractions(events: PointProcessDataset) -> np.ndarray:
    """Each fish's (with an event) share of ipsi events, sorted."""
    n = np.bincount(events.event_fish_idx, minlength=events.num_fish)
    ipsi = np.bincount(events.event_fish_idx, weights=np.asarray(events.marks["laterality"]) == IPSI, minlength=events.num_fish)
    return np.sort(ipsi[n > 0] / n[n > 0])


def plot_side_check(ax: plt.Axes, dataset: PointProcessDataset, model: PointProcess, pred: Dict) -> None:
    """Each fish's share of ipsi events, sorted, in the model's predictive range at the same quantiles (the pinned s_side)."""
    fractions, what = ipsi_fractions(model.described(dataset)), "escapes" if model.is_survival else "events"
    ranks = np.arange(1, len(fractions) + 1)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.fill_between(ranks, *pred["ipsi"], color="crimson", alpha=0.15, lw=0, label="model, 95% range")
    ax.plot(ranks, fractions, "o", color="black", ms=3, label="observed")
    ax.set(xlabel=f"fish with {what}, sorted", ylabel=f"share of ipsi {what}", ylim=(0, 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Side preference per fish", fontsize=9)


def cell_summary(row: pd.Series, params: pd.DataFrame, posthoc: pd.DataFrame) -> str:
    """A cell's test and, per free parameter, its reference -> test value (± SE), Glass's delta and post hoc mark (*)."""
    se = lambda v: "" if not np.isfinite(v) else f" ± {v:.2g}"
    test = f"p = {row['p_value']:.3g}, q = {row['q_value']:.3g}" + (" (significant)" if row.get("significant") else "")
    marked = set(posthoc.loc[posthoc["significant"] == True, "parameter"]) if len(posthoc) else set()
    parts = [f"{PARAMETER_LABELS.get(r['parameter'], r['parameter'])} ({r['parameter']}{' *' if r['parameter'] in marked else ''}): "
             f"{r['value_reference']:.3g}{se(r['se_reference'])} → {r['value_test']:.3g}{se(r['se_test'])}, Δ = {r['delta']:+.2f} SD"
             for _, r in params.iterrows()]
    return "    ·    ".join([test, *parts]) + ("    (* post hoc, adjusted p < 0.05)" if marked else "")


def predictive(model: PointProcess, dataset: PointProcessDataset, sides: Sequence[Optional[int]] = (None,)) -> Dict:
    """
    The model's predictive 95% ranges on dataset's design (its fish, trials and hours), from PREDICTIVE_SIMS simulated
    datasets, of what the grids show of the events it describes: per side (None: all events) the time and trial marginals;
    each fish's events per trial, sorted; the share of trials in each count bin (count_bins, from the data).
    """
    rng = np.random.default_rng(0)
    sims = [model.described(model.simulate_dataset(dataset, rng)) for _ in range(PREDICTIVE_SIMS)]
    band = lambda values: np.nanpercentile(np.array(values), [2.5, 97.5], axis=0)
    bins = count_bins(model.described(dataset))
    result = {"bins": bins, "fish": band([np.sort(fish_rates(s)) for s in sims]), "counts": band([count_shares(s, *bins) for s in sims])}
    if sided_base(model) is not None:
        observed = ipsi_fractions(model.described(dataset))
        positions = (np.arange(len(observed)) + 0.5) / max(len(observed), 1)
        result["ipsi"] = band([np.quantile(f, positions) if len(f) else np.full(len(positions), np.nan) for f in map(ipsi_fractions, sims)])
    for side in sides:
        marginals = [_weighted_marginals((s if side is None else s.select(laterality=side)).time_trial_histogram_hz, dataset) for s in sims]
        result[side] = {"time": band([m[0] for m in marginals]), "trial": band([m[1] for m in marginals])}
    return result


def plot_count_checks(ax_fish: plt.Axes, ax_counts: plt.Axes, dataset: PointProcessDataset, model: PointProcess, pred: Dict) -> None:
    """Each fish's events per trial, sorted, and the share of trials with 0, 1, ... events: the data in the model's predictive range."""
    events, what = model.described(dataset), "escapes" if model.is_survival else "events"
    fish = np.sort(fish_rates(events))
    ranks = np.arange(1, len(fish) + 1)
    ax_fish.fill_between(ranks, *pred["fish"], color="crimson", alpha=0.15, lw=0, label="model, 95% range")
    ax_fish.plot(ranks, fish, "o", color="black", ms=3, label="observed")
    ax_fish.set(xlabel="fish, sorted", ylabel=f"{what} per trial", ylim=(0, None))
    ax_fish.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_fish.set_title(f"{what.capitalize()} per fish", fontsize=9)
    ax_fish.legend(fontsize=6, frameon=False, loc="upper left")
    x = np.arange(pred["bins"][1] + 1)
    ax_counts.vlines(x, *pred["counts"], color="crimson", alpha=0.35, lw=8)
    ax_counts.plot(x, count_shares(events, *pred["bins"]), "o", color="black", ms=4)
    ax_counts.set_xticks(x, count_labels(*pred["bins"]), rotation=45 if pred["bins"][0] > 1 else 0)
    ax_counts.set(xlabel=f"{what} in a trial", ylabel="share of trials", ylim=(0, 1), xlim=(-0.6, x[-1] + 0.6))
    ax_counts.set_title(f"{what.capitalize()} per trial", fontsize=9)


def plot_arm_surface_grid_with_marginals(
    ds_ref: PointProcessDataset, ds_test: PointProcessDataset, m_ref: PointProcess, m_test: PointProcess,
    title: str = "", cmap: str = "plasma", figsize: Tuple[float, float] = (12, 13), subtitle: str = "",
) -> plt.Figure:
    """
    Per arm (reference, test): the empirical and model surfaces of the events the model describes, their time marginal
    above (one strip spanning both: empirical solid, model dashed) and trial marginal on the right, each with the model's
    predictive 95% range (predictive); below, the events per fish and per trial against it (plot_count_checks).
    subtitle: under the title (cell_summary).
    """
    arms = {fish_label("Reference", ds_ref): (ds_ref, m_ref), fish_label("Test", ds_test): (ds_test, m_test)}
    shown = {label: display_surfaces(m, ds) for label, (ds, m) in arms.items()}
    preds = {label: predictive(m, ds) for label, (ds, m) in arms.items()}
    marginals = {label: [_weighted_marginals(x, arms[label][0]) for x in shown[label][1:]] for label in arms}
    vmax = _model_vmax([x[2] for x in shown.values()], [x[1] for x in shown.values()])
    time_max = max(1e-9, *(np.nanmax(np.r_[tm, preds[label][None]["time"][1]]) for label, pair in marginals.items() for tm, _ in pair))
    trial_max = max(1e-9, *(np.nanmax(np.r_[trm, preds[label][None]["trial"][1]]) for label, pair in marginals.items() for _, trm in pair))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 3, width_ratios=[4, 4, 1.3], height_ratios=[1, 4, 1.8] * 2, wspace=0.25, hspace=0.5,
                          left=0.07, right=0.86, top=0.92, bottom=0.05)
    for block, (label, (ds, m)) in enumerate(arms.items()):
        row, (t_edges, emp, model), pred = 3 * block, shown[label], preds[label][None]
        (tm_emp, trm_emp), (tm_model, trm_model) = marginals[label]
        ax_time = fig.add_subplot(gs[row, 0:2])
        t_centers = (t_edges[:-1] + t_edges[1:]) / 2
        ax_time.fill_between(t_centers, *pred["time"], color="crimson", alpha=0.15, lw=0, label="model, 95% range")
        ax_time.plot(t_centers, tm_emp, color="black", lw=1.3, label="Empirical")
        ax_time.plot(t_centers, tm_model, color="crimson", ls="--", lw=1.3, label="Model")
        ax_time.set(ylim=(0, time_max * 1.05), xlim=(t_edges[0], t_edges[-1]))
        ax_time.set_title(label, fontsize=11, fontweight="bold", loc="left")
        ax_time.set_ylabel(rate_label(m), fontsize=7)
        if block == 0:
            ax_time.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3, fontsize=7, frameon=False)
        for col, (surface, name) in enumerate(((emp, "Empirical"), (model, "Model"))):
            ax = fig.add_subplot(gs[row + 1, col])
            mesh = ax.pcolormesh(t_edges, ds.trial_edges, surface, shading="flat", cmap=cmap, vmin=0.0, vmax=vmax)
            ax.set_xlabel("Time in trial (s)", fontsize=8)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if block == 0:
                ax.set_title(name, fontsize=9)
            if col:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel("Trial", fontsize=8)
        ax_trial = fig.add_subplot(gs[row + 1, 2])
        trials = np.arange(ds.num_trials)
        ax_trial.fill_betweenx(trials, *pred["trial"], color="crimson", alpha=0.15, lw=0)
        ax_trial.plot(trm_emp, trials, color="black", lw=1.3)
        ax_trial.plot(trm_model, trials, color="crimson", ls="--", lw=1.3)
        ax_trial.set(xlim=(0, trial_max * 1.05), ylim=(-0.5, ds.num_trials - 0.5), xlabel=f"{rate_label(m)},\nmean over the trial")
        plt.setp(ax_trial.get_yticklabels(), visible=False)
        plot_count_checks(fig.add_subplot(gs[row + 2, 0]), fig.add_subplot(gs[row + 2, 1]), ds, m, preds[label])
    for ax in fig.axes:
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.975)
    fig.text(0.5, 0.952, subtitle, ha="center", va="center", fontsize=7, wrap=True)
    fig.colorbar(mesh, cax=fig.add_axes([0.89, 0.4, 0.02, 0.4])).set_label(rate_label(m_ref), fontsize=9)
    return fig

SIDE_COLORS = {IPSI: "#0F6E56", CONTRA: "#534AB7"}


def plot_sided_arm_surface_grid(
    ds_ref: PointProcessDataset, ds_test: PointProcessDataset, m_ref: PointProcess, m_test: PointProcess,
    title: str = "", cmap: str = "plasma", figsize: Tuple[float, float] = (22, 20), subtitle: str = "",
) -> plt.Figure:
    """
    plot_arm_surface_grid_with_marginals for a sided model, each side apart: per arm (reference, test), the
    empirical and model rate surfaces of ipsi and of contra events, their time marginals above (empirical solid,
    model dashed), both sides' trial marginals on the right, each with the model's predictive 95% range; below, the
    side calibration by time and by trial, the events per fish and per trial (plot_count_checks) and the side
    preference per fish (plot_side_check). subtitle: under the title (cell_summary).
    """
    arms = {"Reference": (ds_ref, m_ref), "Test": (ds_test, m_test)}
    binned = {(arm, side): display_surfaces(m, ds, side=side) for arm, (ds, m) in arms.items() for side in SIDE_NAMES}
    edges, surfaces = {key: b[0] for key, b in binned.items()}, {key: b[1:] for key, b in binned.items()}
    marginals = {key: [_weighted_marginals(surface, arms[key[0]][0]) for surface in pair] for key, pair in surfaces.items()}
    preds = {arm: predictive(m, ds, sides=list(SIDE_NAMES)) for arm, (ds, m) in arms.items()}
    vmax = _model_vmax([model for _, model in surfaces.values()], [emp for emp, _ in surfaces.values()])
    time_max = max(1e-9, *(np.nanmax(np.r_[tm, preds[arm][side]["time"][1]]) for (arm, side), pair in marginals.items() for tm, _ in pair))
    trial_max = max(1e-9, *(np.nanmax(np.r_[trm, preds[arm][side]["trial"][1]]) for (arm, side), pair in marginals.items() for _, trm in pair))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(8, 5, width_ratios=[4, 4, 4, 4, 1.6], height_ratios=[1, 4, 1.8, 1.8] * 2,
                          wspace=0.2, hspace=0.5, left=0.05, right=0.9, top=0.94, bottom=0.04)
    for block, (arm, (ds, m)) in enumerate(arms.items()):
        row = 4 * block
        ax_trial = fig.add_subplot(gs[row + 1, 4])
        for col, side in zip((0, 2), SIDE_NAMES):
            (emp, model), ((tm_emp, trm_emp), (tm_model, trm_model)) = surfaces[arm, side], marginals[arm, side]
            ax_time = fig.add_subplot(gs[row, col:col + 2])
            t_edges = edges[arm, side]
            ax_time.fill_between((t_edges[:-1] + t_edges[1:]) / 2, *preds[arm][side]["time"], color="crimson", alpha=0.15, lw=0, label="model, 95% range")
            ax_time.plot((t_edges[:-1] + t_edges[1:]) / 2, tm_emp, color="black", lw=1.2, label="Empirical")
            ax_time.plot((t_edges[:-1] + t_edges[1:]) / 2, tm_model, color="crimson", ls="--", lw=1.2, label="Model")
            ax_time.set(ylim=(0, time_max * 1.05), xlim=(ds.t_grid[0], ds.t_grid[-1]))
            ax_time.set_title(f"{fish_label(arm, ds)}: {SIDE_NAMES[side]}", fontsize=11, fontweight="bold", loc="left", color=SIDE_COLORS[side])
            ax_time.tick_params(labelsize=6)  # its own time ticks: it spans both surfaces below
            if col == 0:
                ax_time.set_ylabel(rate_label(m), fontsize=7)
            if block == 0 and col == 0:
                ax_time.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3, fontsize=7, frameon=False)
            for offset, (surface, label) in enumerate(((emp, "Empirical"), (model, "Model"))):
                ax = fig.add_subplot(gs[row + 1, col + offset])
                mesh = ax.pcolormesh(t_edges, ds.trial_edges, surface, shading="flat", cmap=cmap, vmin=0.0, vmax=vmax)
                ax.set_title(label, fontsize=9)
                ax.set_xlabel("Time in trial (s)", fontsize=8)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                if col + offset:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    ax.set_ylabel("Trial", fontsize=8)
            trials = np.arange(ds.num_trials)
            ax_trial.fill_betweenx(trials, *preds[arm][side]["trial"], color=SIDE_COLORS[side], alpha=0.12, lw=0)
            ax_trial.plot(trm_emp, trials, color=SIDE_COLORS[side], lw=1.2, label=f"{SIDE_NAMES[side]} empirical")
            ax_trial.plot(trm_model, trials, color=SIDE_COLORS[side], ls="--", lw=1.2, label=f"{SIDE_NAMES[side]} model")
        ax_trial.set(xlim=(0, trial_max * 1.05), ylim=(-0.5, ds.num_trials - 0.5), xlabel=f"{rate_label(m)},\nmean over the trial")
        plt.setp(ax_trial.get_yticklabels(), visible=False)
        if block == 0:
            ax_trial.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
        ax_by_trial = fig.add_subplot(gs[row + 2, 2:4])
        plot_time_and_trial_calibration(fig.add_subplot(gs[row + 2, 0:2]), ax_by_trial, ds, m)
        ax_by_trial.xaxis.set_major_locator(MaxNLocator(integer=True))
        plot_count_checks(fig.add_subplot(gs[row + 3, 0:2]), fig.add_subplot(gs[row + 3, 2]), ds, m, preds[arm])
        plot_side_check(fig.add_subplot(gs[row + 3, 3:5]), ds, m, preds[arm])
    for ax in fig.axes:
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.965, subtitle, ha="center", va="center", fontsize=8)
    fig.colorbar(mesh, cax=fig.add_axes([0.92, 0.4, 0.012, 0.3])).set_label(rate_label(m_ref), fontsize=9)
    return fig


def generate_arm_surface_grids(
    cells: pd.DataFrame, arms: Arms, output_dir: Path, param_df: Optional[pd.DataFrame] = None, posthoc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Each cell's arm surface grid, its test and parameters under the title (cell_summary, from param_df: the cells'
    compute_parameter_deltas, and posthoc: their parameter_tests); a per-(line, behavior) status table, 'plotted',
    'no_data' or 'fit_failed', so missing plots are traceable.
    """
    results = []
    effects = standardized_effects(param_df, cells) if param_df is not None and len(param_df) else pd.DataFrame(columns=["line", "behavior"])
    posthoc = posthoc if posthoc is not None and len(posthoc) else pd.DataFrame(columns=["line", "behavior", "parameter", "significant"])

    for _, row in tqdm(cells.iterrows(), total=len(cells), desc="Arm surface grids"):
        line, behavior = row["line"], row["behavior"]
        ds_ref, ds_test = arms.datasets(line, behavior)
        if ds_ref is None or ds_test is None:
            results.append({"line": line, "behavior": behavior, "plot_status": "no_data"})
            continue

        try:
            m_ref, m_test = arms.models(line, behavior)
        except Exception as e:
            tqdm.write(f"[{line}/{behavior}] fit failed, skipping plot: {e}")
            results.append({"line": line, "behavior": behavior, "plot_status": f"fit_failed: {e}"})
            continue

        reason = row.get("plot_reason", "")
        title = f"{line_name(line)} / {behavior}" + (f"  [{reason}]" if reason else "")
        plot = plot_arm_surface_grid_with_marginals if sided_base(m_ref) is None else plot_sided_arm_surface_grid
        of_cell = lambda df: df[(df["line"] == line) & (df["behavior"] == behavior)]
        subtitle = cell_summary(row, of_cell(effects), of_cell(posthoc)) if row.get("status") == "ok" else ""
        fig = plot(ds_ref, ds_test, m_ref, m_test, title=title, subtitle=subtitle)
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

    run_floor = 1.0 / (np.minimum(df["n_distinct_permutations"], df["n_perm_success"]) + 1)  # of the permutations run
    at_floor = is_ok & (df["p_value"] <= run_floor * 1.5)
    df.loc[at_floor, "cause"] = "p_value_near_floor"

    unreliable = is_ok & df["permutation_unreliable"].fillna(False)
    df.loc[unreliable, "cause"] = "permutation_unreliable"

    failed = is_ok & (df["n_perm_failed"] > MAX_FAILED_PERMUTATIONS * (df["n_perm_success"] + df["n_perm_failed"]))
    df.loc[failed, "cause"] = "permutations_failed"

    no_signal_arm = is_ok & (
        (df["qc_verdict_reference"] == "reliable_no_signal") |
        (df["qc_verdict_test"] == "reliable_no_signal")
    )
    df.loc[no_signal_arm, "cause"] = "one_arm_no_signal"

    weak_margin = is_ok & (
        (df["qc_delta_aic_reference"].fillna(np.inf) < WEAK_QC_MARGIN_AIC) |
        (df["qc_delta_aic_test"].fillna(np.inf) < WEAK_QC_MARGIN_AIC)
    )
    df.loc[weak_margin, "cause"] = "weak_qc_margin"

    return df


def plot_volcano(
    df: pd.DataFrame,
    triage_df: Optional[pd.DataFrame] = None,   # from build_bad_fit_triage, for cause-aware markers
    effect_col: str = "effect_size",
    label_col: str = "line",
    figsize_per_panel: Tuple[float, float] = (4.0, 3.5),
    max_labels_per_panel: int = 15,
    min_labels_per_panel: int = 3,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Volcano plot(s): effect_size (x) vs -log10(p_value) (y), for every
    'ok'/'ok_response_abolished' cell, one panel per behavior. Points are
    colored/shaped by fit-quality cause (from build_bad_fit_triage) rather
    than plain significance alone -- this matters here specifically
    because several known issues (permutation-floor pinning, weak QC
    margin, one-arm collapse) can make a cell LOOK like a clean hit by
    q-value alone while still deserving a discount. If triage_df isn't
    supplied, falls back to a plain significant/not-significant coloring.

    The x-axis label names the effect size of the behavior's cells
    (tier1_effect_size): the logit difference of a response probability,
    or the log2 fold change of a rate or of the single free parameter.
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
        "permutations_failed": dict(color="dimgray", marker="v", alpha=0.7, label=f"> {MAX_FAILED_PERMUTATIONS:.0%} permutations failed"),
        "one_arm_no_signal": dict(color="purple", marker="s", alpha=0.7, label="one arm no signal vs null"),
        "weak_qc_margin": dict(color="goldenrod", marker="D", alpha=0.6, label=f"weak QC margin (ΔAIC<{WEAK_QC_MARGIN_AIC:g})"),
        "response_abolished": dict(color="black", marker="*", alpha=0.9, label="response abolished (Fisher test)"),
    }
    default_style = dict(color="gray", marker="o", alpha=0.4, label="other")

    def _behavior_q_line(sub: pd.DataFrame) -> Optional[float]:
        sig_rows = sub[sub.get("significant", False)]
        return sig_rows["p_value"].max() if len(sig_rows) else None

    def _scatter_by_cause(ax: plt.Axes, sub: pd.DataFrame) -> None:
        for cause, group in sub.groupby("cause"):
            style = cause_style.get(cause, default_style)
            ax.scatter(group[effect_col], group["neg_log10_p"], s=22, edgecolors="none", **style)

    def _resolve_top_n(sub: pd.DataFrame, max_labels: int, min_labels: int) -> int:
        n_sig = int(sub.get("significant", pd.Series(dtype=bool)).sum())
        return int(np.clip(n_sig, min_labels, max_labels))

    def _axis_label_for(sub: pd.DataFrame) -> str:

        if "metric_name" not in sub.columns or sub.empty:
            return "effect size"
        metric_names = sub["metric_name"].dropna().unique()
        if len(metric_names) == 0:
            return "effect size"
        if len(metric_names) > 1:
            return f"effect size (mixed metrics: {', '.join(metric_names)})"

        name = metric_names[0]
        etype = (
            sub["effect_size_type"].dropna().iloc[0]
            if "effect_size_type" in sub.columns and sub["effect_size_type"].notna().any()
            else None
        )

        if name.startswith("param_"):
            pname = name[len("param_"):]
            return f"log2({pname}_test / {pname}_reference)"
        if name == "response_probability" and etype == "logit_diff":
            return "logit(p_test) - logit(p_reference)"
        if etype == "log2_fold_change":
            return f"log2({name}_test / {name}_reference)"
        return f"effect size ({name})"

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

        q_line_p = _behavior_q_line(sub)
        if q_line_p is not None:
            ax.axhline(-np.log10(q_line_p), color="black", linestyle=":", linewidth=0.8)  # the legend's FDR line
        ax.axvline(0, color="black", linewidth=0.5)

        local_xmax = sub[effect_col].abs().max() if len(sub) else 1.0
        ax.set_xlim(-local_xmax * 1.15, local_xmax * 1.15)
        ax.set_ylim(0, y_max * 1.1)
        ax.set_title(behavior, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        panel_top_n = _resolve_top_n(sub, max_labels=max_labels_per_panel, min_labels=min_labels_per_panel)
        _label_top_hits(ax, sub, effect_col, label_col, panel_top_n)

        ax.set_xlabel(_axis_label_for(sub), fontsize=7)
        if i % n_cols == 0:
            ax.set_ylabel("-log10(p)", fontsize=8)

    for j in range(len(behaviors), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    handles = [plt.Line2D([0], [0], linestyle="", marker=s["marker"], color=s["color"], label=s["label"])
               for s in cause_style.values()]
    handles.append(plt.Line2D([0], [0], color="black", linestyle=":", linewidth=0.8, label=f"largest significant p (q < {ALPHA:g})"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Volcano: effect size vs. significance, by behavior", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig, axes

def _label_top_hits(ax: plt.Axes, sub: pd.DataFrame, effect_col: str, label_col: str, top_n: int) -> None:
    if top_n <= 0 or "p_value" not in sub.columns:
        return
    top = sub.nsmallest(top_n, "p_value")
    for _, row in top.iterrows():
        ax.annotate(
            str(row[label_col]), (row[effect_col], row["neg_log10_p"]),
            fontsize=6, xytext=(3, 3), textcoords="offset points",
            rotation=45, rotation_mode="anchor", ha="left", va="bottom"
        )

def _bootstrap_one_delta_iteration(
    seed_seq: np.random.SeedSequence, ds_ref: PointProcessDataset, ds_test: PointProcessDataset, behavior: str,
    fits: Tuple[PointProcess, PointProcess], steps: Tuple[Optional[np.ndarray], Optional[np.ndarray]], transform_lookup: dict,
) -> Optional[Dict[str, float]]:
    """
    One bootstrap iteration: resample both arms, refit both from their fits
    (Newton steps with their Hessians, else the optimizer), return the
    delta for every free parameter. Returns None on any fit failure (kept
    silent here -- failure counting happens at the aggregation level).
    Top-level function (not a closure) so it pickles cleanly for
    joblib's process-based backends.
    """
    rng = np.random.default_rng(seed_seq)
    ds_ref_b = ds_ref.resample(rng)
    ds_test_b = ds_test.resample(rng)
    try:
        m_ref_b, _ = refit_from_fit(fits[0], ds_ref_b, steps[0])
        m_test_b, _ = refit_from_fit(fits[1], ds_test_b, steps[1])
    except Exception:
        return None

    out = {}
    for pname in m_ref_b.param_names:
        v_v, v_d = m_ref_b.param_dict_.get(pname), m_test_b.param_dict_.get(pname)
        if v_v is None or v_d is None:
            continue
        transform = transform_lookup.get((behavior, pname), "difference")
        out[pname] = (
            np.log2(max(v_d, 1e-9) / max(v_v, 1e-9)) if transform == "log2_ratio" else v_d - v_v
        )
    return out

def _newton_steps(model: PointProcess, dataset: PointProcessDataset) -> Optional[np.ndarray]:
    """newton_step_matrix of an arm fit's Hessian; None (refits by the optimizer) if it cannot be estimated."""
    try:
        return newton_step_matrix(model.estimate_hessian(dataset))
    except Exception:
        return None


def bootstrap_parameter_deltas(
    tier1_df: pd.DataFrame,
    param_df: pd.DataFrame,
    arms: Arms,
    cells: Optional[pd.DataFrame] = None,
    *, n_boot: int, ci: float = 95.0, seed: int = 0, n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Bootstrap CI on the test-vs-reference DELTA of each free parameter, for a
    subset of (line, behavior) cells: the cells in turn, each one's draws over the workers.
    A parameter is reported when at least MIN_BOOT_SUCCESS of its draws refitted.
    """
    if cells is None:
        cells = tier1_df[tier1_df["significant"] == True][["line", "behavior"]]

    transform_lookup = {}
    for (behavior, pname), group in param_df.groupby(["behavior", "parameter"]):
        vals = pd.concat([group["value_reference"], group["value_test"]]).dropna()
        transform_lookup[(behavior, pname)] = "log2_ratio" if len(vals) > 0 and (vals > 0).all() else "difference"

    def _point_delta(v_ref: float, v_test: float, transform: str) -> float:
        if transform == "log2_ratio":
            return np.log2(max(v_test, 1e-9) / max(v_ref, 1e-9))
        return v_test - v_ref

    records = []
    for _, row in tqdm(cells.iterrows(), total=len(cells), desc="Bootstrapping parameter deltas (cells)"):
        line, behavior = row["line"], row["behavior"]
        ds_ref, ds_test = arms.datasets(line, behavior)
        if ds_ref is None or ds_test is None:
            continue

        try:
            m_ref0, m_test0 = arms.models(line, behavior)
        except Exception as e:
            tqdm.write(f"[{line}/{behavior}] initial fit failed: {e}")
            continue

        param_names = m_ref0.param_names
        seeds = np.random.SeedSequence(seed).spawn(n_boot)

        results = parallel_map(
            partial(
                _bootstrap_one_delta_iteration,
                ds_ref=ds_ref,
                ds_test=ds_test,
                behavior=behavior,
                fits=(m_ref0, m_test0),
                steps=(_newton_steps(m_ref0, ds_ref), _newton_steps(m_test0, ds_test)),
                transform_lookup=transform_lookup,
            ),
            seeds,
            n_jobs=n_jobs,
            desc=f"  {line}/{behavior}",
        )

        boot_deltas = {p: [] for p in param_names}
        for r in results:
            if r is None:
                continue
            for pname, val in r.items():
                if pname in boot_deltas:
                    boot_deltas[pname].append(val)

        alpha = (100.0 - ci) / 2.0
        min_success = max(1, int(MIN_BOOT_SUCCESS * n_boot))
        for pname in param_names:
            arr = np.array(boot_deltas[pname])
            if len(arr) < min_success:
                continue
            transform = transform_lookup.get((behavior, pname), "difference")
            point_delta = _point_delta(m_ref0.param_dict_.get(pname), m_test0.param_dict_.get(pname), transform)
            records.append({
                "line": line, "behavior": behavior, "parameter": pname, "transform": transform,
                "delta": point_delta, "boot_mean": float(np.mean(arr)),
                "ci_lo": float(np.percentile(arr, alpha)), "ci_hi": float(np.percentile(arr, 100 - alpha)),
                "n_boot_success": len(arr),
            })

    columns = ["line", "behavior", "parameter", "transform", "delta", "boot_mean", "ci_lo", "ci_hi", "n_boot_success"]
    return pd.DataFrame(records, columns=columns)

def plot_parameter_forest(
    boot_delta_df: pd.DataFrame,
    behavior: str,
    sort_by_abs: bool = False,
) -> Optional[plt.Figure]:
    """
    Forest plot for one behavior: one subplot per free parameter, lines on
    the y-axis (sorted by delta, or |delta| if sort_by_abs), bootstrap CI
    as a horizontal whisker. Gray points = CI crosses zero (not
    distinguishable from no change at this line's sample size); colored
    points = CI excludes zero in that direction.
    """
    sub = boot_delta_df[boot_delta_df["behavior"] == behavior]
    if sub.empty:
        return None

    parameters = sorted(sub["parameter"].unique())
    n_lines = sub["line"].nunique()
    fig_h = max(3.0, 0.35 * n_lines + 1.0)
    fig, axes = plt.subplots(1, len(parameters), figsize=(4.2 * len(parameters), fig_h), squeeze=False)
    axes = axes[0]

    for ax, pname in zip(axes, parameters):
        psub = sub[sub["parameter"] == pname].copy()
        sort_key = psub["delta"].abs() if sort_by_abs else psub["delta"]
        psub = psub.assign(_sort_key=sort_key).sort_values("_sort_key").reset_index(drop=True)

        y_pos = np.arange(len(psub))
        excludes_zero = ~((psub["ci_lo"] <= 0) & (psub["ci_hi"] >= 0))
        colors = np.where(~excludes_zero, "lightgray", np.where(psub["delta"] > 0, "crimson", "steelblue"))

        xerr_lo = np.maximum(psub["delta"] - psub["ci_lo"], 0)
        xerr_hi = np.maximum(psub["ci_hi"] - psub["delta"], 0)

        ax.errorbar(psub["delta"], y_pos, xerr=[xerr_lo, xerr_hi],
                    fmt="none", ecolor="lightgray", elinewidth=1.3, capsize=2, zorder=1)
        ax.scatter(psub["delta"], y_pos, c=colors, s=28, zorder=3, edgecolors="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(psub["line"].map(line_name), fontsize=7)
        transform = psub["transform"].iloc[0]
        xlabel = f"log2({pname}_test / {pname}_reference)" if transform == "log2_ratio" else f"Δ{pname} (test − reference)"
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(f"{PARAMETER_LABELS.get(pname, pname)} ({pname})", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(f"{behavior}: bootstrap parameter deltas (95% CI)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig