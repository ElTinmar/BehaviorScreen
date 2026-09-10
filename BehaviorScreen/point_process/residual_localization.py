"""
Stratified Cox-Snell residual-localization diagnostics.

These diagnostics help localize a global calibration failure by examining:

1. trial block;
2. interval start time within trial;
3. event order within trial;
4. raw time since the preceding event;
5. estimated fish-frailty quantile;
6. stream event-count category.

For event-order stratification, a terminally censored interval is assigned
to the order of the next event that could have occurred. For example:

- an empty recurrent trial contributes a censored Event-1 interval;
- a trial with two events contributes exact Event-1 and Event-2 intervals,
  followed by a censored Event-3 interval.

This preserves the relevant censoring information within each event-order
stratum.

The bootstrap can either refit every simulated dataset or perform a faster
fixed-parameter predictive simulation.

This figure is intended for exploratory localization of a global GOF
failure, not as six independent formal hypothesis tests.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from tqdm import tqdm

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.tqdm_joblib import tqdm_joblib

# =============================================================================
# Residual metadata
# =============================================================================


def residual_interval_frame(
    model,
    dataset: PointProcessDataset,
    n_gain_quantiles: int = 4,
) -> pd.DataFrame:
    """
    Return one row per time-rescaled waiting interval.

    For recurrent models:
      - one exact interval per event;
      - one terminally censored interval per active fish x trial.

    For survival models:
      - one exact first-event interval; or
      - one censored first-event interval ending at trial termination.

    Event-order convention
    ----------------------
    ``event_order`` is the order of the event that terminates, or could have
    terminated, the waiting interval.

    Therefore:

      - onset -> first event: Event 1, exact;
      - first -> second event: Event 2, exact;
      - second event -> trial end: Event 3, censored;
      - onset -> trial end in an empty trial: Event 1, censored.

    This convention keeps exact and censored observations in the same
    event-order risk set.

    Returned columns
    ----------------
    fish_idx
    trial_idx
    interval_start_s
    interval_end_s
    raw_interval_s
    event_order
    censored
    tau
    stream_event_count
    estimated_gain
    gain_quantile
    """
    if model.params_ is None:
        raise ValueError("Model must be fitted or assigned parameters.")

    tau_by_stream = model._stream_tau_values(dataset)
    records: List[Dict[str, Any]] = []

    for f_idx, t_idx, t_ev in dataset.iter_streams():
        t_ev = np.sort(np.asarray(t_ev, dtype=float))
        pairs = tau_by_stream.get((f_idx, t_idx), [])

        if not pairs:
            continue

        if model.is_survival:
            tau, censored = pairs[0]
            interval_end = dataset.duration_s if censored else float(t_ev[0])

            records.append(
                {
                    "fish_idx": int(f_idx),
                    "trial_idx": int(t_idx),
                    "interval_start_s": 0.0,
                    "interval_end_s": float(interval_end),
                    "raw_interval_s": float(interval_end),
                    "event_order": 1,
                    "censored": bool(censored),
                    "tau": float(tau),
                    "stream_event_count": int(len(t_ev)),
                }
            )
            continue

        previous_time = 0.0
        event_cursor = 0

        for tau, censored in pairs:
            if censored:
                interval_end = float(dataset.duration_s)

                # The terminal interval is the waiting time for the next
                # event that did not occur before trial termination.
                event_order = event_cursor + 1
            else:
                if event_cursor >= len(t_ev):
                    raise RuntimeError(
                        "More exact residuals than observed events for "
                        f"fish={f_idx}, trial={t_idx}."
                    )

                interval_end = float(t_ev[event_cursor])
                event_order = event_cursor + 1
                event_cursor += 1

            raw_interval = interval_end - previous_time
            if raw_interval < -1e-10:
                raise RuntimeError(
                    "Negative raw interval encountered for "
                    f"fish={f_idx}, trial={t_idx}: "
                    f"start={previous_time}, end={interval_end}."
                )

            records.append(
                {
                    "fish_idx": int(f_idx),
                    "trial_idx": int(t_idx),
                    "interval_start_s": float(previous_time),
                    "interval_end_s": float(interval_end),
                    "raw_interval_s": float(max(raw_interval, 0.0)),
                    "event_order": int(event_order),
                    "censored": bool(censored),
                    "tau": float(tau),
                    "stream_event_count": int(len(t_ev)),
                }
            )

            previous_time = interval_end

        if event_cursor != len(t_ev):
            raise RuntimeError(
                "Number of exact residuals does not match observed events for "
                f"fish={f_idx}, trial={t_idx}: exact residuals={event_cursor}, "
                f"events={len(t_ev)}."
            )

    frame = pd.DataFrame.from_records(records)

    if frame.empty:
        return frame

    frame["estimated_gain"] = np.nan
    frame["gain_quantile"] = "All fish"

    try:
        gains = model.estimate_fish_gains(dataset)
        required_columns = {"fish_idx", "estimated_gain"}

        if isinstance(gains, pd.DataFrame) and required_columns.issubset(gains.columns):
            gain_map = gains.set_index("fish_idx")["estimated_gain"]
            frame["estimated_gain"] = frame["fish_idx"].map(gain_map)

            fish_gain = (
                frame[["fish_idx", "estimated_gain"]]
                .drop_duplicates("fish_idx")
                .dropna()
            )

            enough_fish = len(fish_gain) >= n_gain_quantiles
            enough_unique = fish_gain["estimated_gain"].nunique() >= n_gain_quantiles

            if enough_fish and enough_unique:
                labels = [f"Q{i + 1}" for i in range(n_gain_quantiles)]

                # Ranking avoids qcut failures when several fish have
                # identical posterior gain estimates.
                ranked_gains = fish_gain["estimated_gain"].rank(method="first")
                fish_gain["gain_quantile"] = pd.qcut(
                    ranked_gains,
                    q=n_gain_quantiles,
                    labels=labels,
                ).astype(str)

                quantile_map = fish_gain.set_index("fish_idx")["gain_quantile"]
                frame["gain_quantile"] = (
                    frame["fish_idx"].map(quantile_map).fillna("Unassigned")
                )

    except (NotImplementedError, AttributeError, ValueError):
        # Models without fish-level gain estimates retain "All fish".
        pass

    return frame


# =============================================================================
# Group definitions
# =============================================================================


def _make_time_labels(time_edges: np.ndarray) -> List[str]:
    """Create stable labels for interval-start-time bins."""
    return [
        f"{left:g}–{right:g}s" for left, right in zip(time_edges[:-1], time_edges[1:])
    ]


def _event_order_group(event_order: int) -> str:
    """Map the next-event order to a plotting category."""
    if event_order <= 1:
        return "Event 1"
    if event_order <= 3:
        return "Events 2–3"
    if event_order <= 6:
        return "Events 4–6"
    return "Events 7+"


def add_localization_groups(
    frame: pd.DataFrame,
    dataset: PointProcessDataset,
    time_edges: np.ndarray,
) -> pd.DataFrame:
    """
    Add trial-block, interval-start-time, and event-order labels.

    Censoring is not used to define the event-order group. Exact and censored
    intervals awaiting the same next-event order belong to the same group.
    """
    frame = frame.copy()

    trial_indices = np.arange(dataset.num_trials)
    trial_blocks = np.array_split(trial_indices, 3)
    trial_block_map: Dict[int, str] = {}

    for label, block in zip(
        ["Early trials", "Middle trials", "Late trials"],
        trial_blocks,
    ):
        for trial_idx in block:
            trial_block_map[int(trial_idx)] = label

    frame["trial_block"] = frame["trial_idx"].map(trial_block_map)

    time_labels = _make_time_labels(time_edges)
    frame["time_bin"] = pd.cut(
        frame["interval_start_s"],
        bins=time_edges,
        labels=time_labels,
        include_lowest=True,
        right=False,
    )

    frame["event_order_group"] = frame["event_order"].map(_event_order_group)
    return frame


# =============================================================================
# Cox-Snell calculations
# =============================================================================


def _evaluate_step(
    x: np.ndarray,
    knots: np.ndarray,
    values: np.ndarray,
    max_x: float,
) -> np.ndarray:
    """Evaluate a right-continuous step function up to ``max_x``."""
    result = np.full_like(x, np.nan, dtype=float)

    if len(knots) == 0:
        return result

    indices = np.searchsorted(knots, x, side="right") - 1
    valid = (indices >= 0) & (x <= max_x)
    valid_indices = np.clip(indices[valid], 0, len(values) - 1)
    result[valid] = values[valid_indices]

    return result


def cox_snell_curve(
    model,
    frame: pd.DataFrame,
    r_grid: np.ndarray,
    min_km_at_risk: int = 1,
) -> np.ndarray:
    """
    Evaluate ``-log(KM survival)`` on a common Cox-Snell residual grid.

    Exact and censored observations in ``frame`` are passed together to the
    Kaplan-Meier estimator.
    """
    if frame.empty:
        return np.full_like(r_grid, np.nan, dtype=float)

    tau = frame["tau"].to_numpy(dtype=float)
    censored = frame["censored"].to_numpy(dtype=bool)

    knots, survival, n_at_risk = model._survival_estimate(
        tau,
        censored,
        return_risk=True,
    )

    knot_indices = np.arange(len(knots))
    supported = (knot_indices > 0) & (n_at_risk >= min_km_at_risk) & (survival > 0.0)

    if not np.any(supported):
        return np.full_like(r_grid, np.nan, dtype=float)

    last_supported_index = np.where(supported)[0][-1]
    r_limit = float(knots[last_supported_index])

    survival_grid = _evaluate_step(
        r_grid,
        knots,
        survival,
        max_x=r_limit,
    )

    result = np.full_like(r_grid, np.nan, dtype=float)
    valid = np.isfinite(survival_grid) & (survival_grid > 0.0)
    result[valid] = -np.log(survival_grid[valid])

    return result


def signed_calibration_area(
    model,
    frame: pd.DataFrame,
    r_grid: np.ndarray,
    min_km_at_risk: int = 1,
) -> float:
    """
    Compute the mean signed Cox-Snell departure from the identity line.

    Negative values:
        The empirical curve lies below the diagonal. Residuals are relatively
        large, so the model tends to accumulate intensity too quickly.

    Positive values:
        The empirical curve lies above the diagonal. Residuals are relatively
        small, so the model tends to accumulate intensity too slowly.
    """
    curve = cox_snell_curve(
        model,
        frame,
        r_grid,
        min_km_at_risk=min_km_at_risk,
    )

    valid = np.isfinite(curve)
    if np.sum(valid) < 2:
        return np.nan

    area = trapezoid(
        curve[valid] - r_grid[valid],
        x=r_grid[valid],
    )
    width = r_grid[valid][-1] - r_grid[valid][0]

    return float(area / max(width, 1e-12))


# =============================================================================
# Summary extraction
# =============================================================================


def _count_category(count: int) -> str:
    """Map a stream event count to a plotting category."""
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2–3"
    return "4+"


def localization_summary(
    model,
    dataset: PointProcessDataset,
    r_grid: np.ndarray,
    time_edges: np.ndarray,
    gap_edges: np.ndarray,
    min_km_at_risk: int = 1,
    include_frame: bool = False,
) -> Dict[str, Any]:
    """Compute all quantities needed by the six localization panels."""
    frame = residual_interval_frame(model, dataset)
    frame = add_localization_groups(frame, dataset, time_edges)

    trial_groups = ["Early trials", "Middle trials", "Late trials"]
    order_groups = ["Event 1", "Events 2–3", "Events 4–6", "Events 7+"]
    time_groups = _make_time_labels(time_edges)

    trial_curves = {
        group: cox_snell_curve(
            model,
            frame[frame["trial_block"] == group],
            r_grid,
            min_km_at_risk=min_km_at_risk,
        )
        for group in trial_groups
    }

    order_curves = {
        group: cox_snell_curve(
            model,
            frame[frame["event_order_group"] == group],
            r_grid,
            min_km_at_risk=min_km_at_risk,
        )
        for group in order_groups
    }

    gain_groups = sorted(
        str(group) for group in frame["gain_quantile"].dropna().unique()
    )
    gain_curves = {
        group: cox_snell_curve(
            model,
            frame[frame["gain_quantile"] == group],
            r_grid,
            min_km_at_risk=min_km_at_risk,
        )
        for group in gain_groups
    }

    time_areas = {
        group: signed_calibration_area(
            model,
            frame[frame["time_bin"] == group],
            r_grid,
            min_km_at_risk=min_km_at_risk,
        )
        for group in time_groups
    }

    # Event order >= 2 means the exact interval began after a preceding event.
    # Censored intervals are excluded because they do not provide an observed
    # post-event waiting time.
    post_event_gaps = frame.loc[
        (~frame["censored"]) & (frame["event_order"] >= 2),
        "raw_interval_s",
    ].to_numpy(dtype=float)

    gap_counts, _ = np.histogram(post_event_gaps, bins=gap_edges)
    gap_widths = np.diff(gap_edges)

    if gap_counts.sum() > 0:
        gap_density = gap_counts / gap_counts.sum() / gap_widths
    else:
        gap_density = np.full(len(gap_edges) - 1, np.nan)

    stream_counts = dataset.stream_event_counts.astype(int)
    count_groups = ["0", "1", "2–3", "4+"]
    stream_categories = np.asarray(
        [_count_category(int(count)) for count in stream_counts]
    )

    count_proportions = {
        group: float(np.mean(stream_categories == group)) for group in count_groups
    }

    result = {
        "trial_curves": trial_curves,
        "order_curves": order_curves,
        "gain_curves": gain_curves,
        "time_areas": time_areas,
        "gap_density": gap_density,
        "count_proportions": count_proportions,
    }

    if include_frame:
        result["frame"] = frame

    return result


# =============================================================================
# Parallel bootstrap worker
# =============================================================================


def _run_localization_bootstrap_replicate(
    seed_seq,
    model,
    dataset: PointProcessDataset,
    refit: bool,
    refit_n_starts: int,
    min_km_at_risk: int,
    r_grid: np.ndarray,
    time_edges: np.ndarray,
    gap_edges: np.ndarray,
) -> Dict[str, Any]:
    """Run one simulate-refit-localize bootstrap replicate."""
    rng = np.random.default_rng(seed_seq)

    try:
        model_b = copy.deepcopy(model)
        dataset_b = model_b.simulate_dataset(template=dataset, rng=rng)

        if refit:
            model_b.initial_guesses = list(np.asarray(model.params_, dtype=float))

            if refit_n_starts <= 1:
                model_b.fit(dataset_b)
            else:
                model_b.fit_multistart(
                    dataset_b,
                    n_starts=refit_n_starts,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    n_jobs=1,
                )

        summary = localization_summary(
            model_b,
            dataset_b,
            r_grid=r_grid,
            time_edges=time_edges,
            gap_edges=gap_edges,
            min_km_at_risk=min_km_at_risk,
            include_frame=False,
        )

        return {"success": True, "summary": summary}

    except Exception as exc:
        return {"success": False, "error": repr(exc)}


# =============================================================================
# Parallel bootstrap orchestration
# =============================================================================


def bootstrap_localization(
    model,
    dataset: PointProcessDataset,
    n_boot: int = 100,
    seed: int = 42,
    refit: bool = True,
    refit_n_starts: int = 1,
    min_km_at_risk: int = 1,
    r_grid: Optional[np.ndarray] = None,
    time_edges: Optional[np.ndarray] = None,
    gap_edges: Optional[np.ndarray] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Simulate residual-localization summaries under the fitted model.

    Bootstrap replicates are parallelized across processes. Each replicate
    performs:

        simulate -> optionally refit -> recompute residual localization

    Parameters
    ----------
    model
        Fitted point-process model.

    dataset
        Dataset defining the fish x trial design.

    n_boot
        Number of bootstrap replicates.

    seed
        Root random seed.

    refit
        If True, refit the model to every simulated dataset. If False, use
        the observed fitted parameters for every simulated dataset.

    refit_n_starts
        Number of optimizer starts per bootstrap replicate. Inner multistart
        fitting is serial to prevent nested parallelism.

    min_km_at_risk
        Minimum KM risk-set size used to retain a Cox-Snell curve segment.

    r_grid
        Common Cox-Snell residual grid.

    time_edges
        Edges defining interval-start-time groups.

    gap_edges
        Histogram edges for exact post-event waiting times.

    n_jobs
        Number of parallel outer workers. Use -1 for all available CPUs.

    verbose
        Whether to display the tqdm progress bar.
    """
    if model.params_ is None:
        raise ValueError("Model must be fitted first.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if min_km_at_risk < 1:
        raise ValueError("min_km_at_risk must be at least 1.")

    if r_grid is None:
        r_grid = np.linspace(0.0, 5.0, 151)

    if time_edges is None:
        time_edges = np.linspace(0.0, dataset.duration_s, 7)

    if gap_edges is None:
        gap_bin_width = max(dataset.binning_dt, 0.05)
        gap_max = min(dataset.duration_s, 5.0)
        gap_edges = np.arange(
            0.0,
            gap_max + gap_bin_width,
            gap_bin_width,
        )

    r_grid = np.asarray(r_grid, dtype=float)
    time_edges = np.asarray(time_edges, dtype=float)
    gap_edges = np.asarray(gap_edges, dtype=float)

    if np.any(np.diff(r_grid) <= 0):
        raise ValueError("r_grid must be strictly increasing.")
    if np.any(np.diff(time_edges) <= 0):
        raise ValueError("time_edges must be strictly increasing.")
    if np.any(np.diff(gap_edges) <= 0):
        raise ValueError("gap_edges must be strictly increasing.")
    if time_edges[0] > 0.0 or time_edges[-1] < dataset.duration_s:
        raise ValueError(
            "time_edges must cover the complete trial interval "
            "[0, dataset.duration_s]."
        )

    observed = localization_summary(
        model,
        dataset,
        r_grid=r_grid,
        time_edges=time_edges,
        gap_edges=gap_edges,
        min_km_at_risk=min_km_at_risk,
        include_frame=True,
    )

    seeds = np.random.SeedSequence(seed).spawn(n_boot)
    progress = tqdm(
        total=n_boot,
        desc="Residual localization bootstrap",
        disable=not verbose,
    )

    # The outer bootstrap is parallel. BLAS/OpenMP thread pools inside each
    # worker are restricted to one thread to avoid CPU oversubscription.
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        with tqdm_joblib(progress):
            worker_results = joblib.Parallel(
                n_jobs=n_jobs,
                batch_size=1,
                pre_dispatch="n_jobs",
            )(
                joblib.delayed(_run_localization_bootstrap_replicate)(
                    seed_seq=seed_seq,
                    model=model,
                    dataset=dataset,
                    refit=refit,
                    refit_n_starts=refit_n_starts,
                    min_km_at_risk=min_km_at_risk,
                    r_grid=r_grid,
                    time_edges=time_edges,
                    gap_edges=gap_edges,
                )
                for seed_seq in seeds
            )

    replicates = [result["summary"] for result in worker_results if result["success"]]
    errors = [result["error"] for result in worker_results if not result["success"]]

    print(
        f"Residual localization bootstrap: {len(replicates)}/{n_boot} "
        f"successful ({len(errors)} failures)."
    )

    if not replicates:
        raise RuntimeError(
            "All localization bootstrap replicates failed. "
            f"Example errors: {errors[:3]}"
        )

    return {
        "observed": observed,
        "replicates": replicates,
        "r_grid": r_grid,
        "time_edges": time_edges,
        "gap_edges": gap_edges,
        "n_requested": int(n_boot),
        "n_successful": len(replicates),
        "n_failed": len(errors),
        "errors": errors,
        "refit": bool(refit),
    }


# =============================================================================
# Bootstrap summary helpers
# =============================================================================


def _curve_band(
    replicates: Sequence[Dict[str, Any]],
    section: str,
    group: str,
    ci: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bootstrap median and pointwise percentile band for one curve."""
    curves = [
        replicate[section][group]
        for replicate in replicates
        if group in replicate[section]
    ]

    if not curves:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    curves = np.asarray(curves, dtype=float)
    alpha = (100.0 - ci) / 2.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(curves, axis=0)
        lower = np.nanpercentile(curves, alpha, axis=0)
        upper = np.nanpercentile(curves, 100.0 - alpha, axis=0)

    return median, lower, upper


def _scalar_band(
    replicates: Sequence[Dict[str, Any]],
    section: str,
    group: str,
    ci: float,
) -> Tuple[float, float, float]:
    """Return bootstrap median and percentile interval for one scalar."""
    values = np.asarray(
        [replicate[section].get(group, np.nan) for replicate in replicates],
        dtype=float,
    )
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    alpha = (100.0 - ci) / 2.0

    return (
        float(np.median(values)),
        float(np.percentile(values, alpha)),
        float(np.percentile(values, 100.0 - alpha)),
    )


def _plot_grouped_cox_snell_curves(
    ax: plt.Axes,
    observed_curves: Dict[str, np.ndarray],
    replicates: Sequence[Dict[str, Any]],
    replicate_section: str,
    groups: Sequence[str],
    r_grid: np.ndarray,
    ci: float,
) -> None:
    """
    Plot grouped observed Cox-Snell curves and bootstrap reference bands.

    Solid lines are observed curves. Dotted lines are bootstrap medians.
    Shaded regions are pointwise bootstrap intervals.
    """
    colors = plt.cm.tab10.colors

    for index, group in enumerate(groups):
        observed_curve = observed_curves.get(group)
        if observed_curve is None:
            continue

        color = colors[index % len(colors)]
        median, lower, upper = _curve_band(
            replicates,
            replicate_section,
            group,
            ci,
        )

        if len(lower):
            valid_band = np.isfinite(lower) & np.isfinite(upper)
            ax.fill_between(
                r_grid[valid_band],
                lower[valid_band],
                upper[valid_band],
                color=color,
                alpha=0.10,
            )

        if len(median):
            valid_median = np.isfinite(median)
            ax.plot(
                r_grid[valid_median],
                median[valid_median],
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
            )

        valid_observed = np.isfinite(observed_curve)
        ax.plot(
            r_grid[valid_observed],
            observed_curve[valid_observed],
            color=color,
            linewidth=1.8,
            label=group,
        )

    ax.plot(r_grid, r_grid, "k--", linewidth=1.2, label="Ideal")
    ax.set_xlabel("Cox–Snell residual $r$")
    ax.set_ylabel(r"$-\log \widehat{S}_{\mathrm{KM}}(r)$")
    ax.grid(True, linestyle=":", alpha=0.3)


# =============================================================================
# Main figure
# =============================================================================


def plot_residual_localization(
    result: Dict[str, Any],
    ci: float = 95.0,
    figsize: Tuple[float, float] = (17, 12),
) -> Tuple[plt.Figure, np.ndarray]:
    """Create the 2 x 3 residual-localization figure."""
    observed = result["observed"]
    replicates = result["replicates"]
    r_grid = result["r_grid"]
    gap_edges = result["gap_edges"]
    gap_centers = 0.5 * (gap_edges[:-1] + gap_edges[1:])

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # ------------------------------------------------------------------
    # A. Trial block
    # ------------------------------------------------------------------

    trial_groups = ["Early trials", "Middle trials", "Late trials"]

    _plot_grouped_cox_snell_curves(
        ax=axes[0, 0],
        observed_curves=observed["trial_curves"],
        replicates=replicates,
        replicate_section="trial_curves",
        groups=trial_groups,
        r_grid=r_grid,
        ci=ci,
    )
    axes[0, 0].set_title("A. Cox–Snell calibration by trial block")
    axes[0, 0].legend(fontsize=8)

    # ------------------------------------------------------------------
    # B. Time within trial
    # ------------------------------------------------------------------

    ax = axes[0, 1]
    time_groups = list(observed["time_areas"])
    x = np.arange(len(time_groups))

    observed_values = np.asarray(
        [observed["time_areas"][group] for group in time_groups]
    )

    median_values = []
    lower_values = []
    upper_values = []

    for group in time_groups:
        median, lower, upper = _scalar_band(
            replicates,
            "time_areas",
            group,
            ci,
        )
        median_values.append(median)
        lower_values.append(lower)
        upper_values.append(upper)

    median_values = np.asarray(median_values)
    lower_values = np.asarray(lower_values)
    upper_values = np.asarray(upper_values)

    ax.fill_between(
        x,
        lower_values,
        upper_values,
        color="steelblue",
        alpha=0.2,
        label=f"{ci:.0f}% bootstrap range",
    )
    ax.plot(
        x,
        median_values,
        color="steelblue",
        linestyle=":",
        marker="o",
        label="Bootstrap median",
    )
    ax.plot(
        x,
        observed_values,
        color="crimson",
        marker="o",
        linewidth=2,
        label="Observed",
    )
    ax.axhline(0.0, color="black", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(time_groups, rotation=35, ha="right", fontsize=8)
    ax.set_title("B. Signed calibration error by interval start time")
    ax.set_ylabel("Mean signed Cox–Snell departure")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)

    # ------------------------------------------------------------------
    # C. Event order
    # ------------------------------------------------------------------

    order_groups = ["Event 1", "Events 2–3", "Events 4–6", "Events 7+"]

    _plot_grouped_cox_snell_curves(
        ax=axes[0, 2],
        observed_curves=observed["order_curves"],
        replicates=replicates,
        replicate_section="order_curves",
        groups=order_groups,
        r_grid=r_grid,
        ci=ci,
    )
    axes[0, 2].set_title("C. Cox–Snell calibration by next-event order")
    axes[0, 2].legend(fontsize=7)

    # ------------------------------------------------------------------
    # D. Post-event raw gap density
    # ------------------------------------------------------------------

    ax = axes[1, 0]
    observed_gap = observed["gap_density"]
    gap_matrix = np.asarray(
        [replicate["gap_density"] for replicate in replicates],
        dtype=float,
    )

    alpha = (100.0 - ci) / 2.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gap_median = np.nanmedian(gap_matrix, axis=0)
        gap_lower = np.nanpercentile(gap_matrix, alpha, axis=0)
        gap_upper = np.nanpercentile(gap_matrix, 100.0 - alpha, axis=0)

    ax.fill_between(
        gap_centers,
        gap_lower,
        gap_upper,
        color="steelblue",
        alpha=0.2,
        label=f"{ci:.0f}% bootstrap range",
    )
    ax.plot(
        gap_centers,
        gap_median,
        color="steelblue",
        linestyle=":",
        label="Bootstrap median",
    )
    ax.plot(
        gap_centers,
        observed_gap,
        color="crimson",
        linewidth=2,
        label="Observed",
    )
    ax.set_title("D. Exact post-event waiting-time density")
    ax.set_xlabel("Time since preceding event (s)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)

    # ------------------------------------------------------------------
    # E. Fish-gain quantile
    # ------------------------------------------------------------------

    gain_groups = list(observed["gain_curves"])

    _plot_grouped_cox_snell_curves(
        ax=axes[1, 1],
        observed_curves=observed["gain_curves"],
        replicates=replicates,
        replicate_section="gain_curves",
        groups=gain_groups,
        r_grid=r_grid,
        ci=ci,
    )
    axes[1, 1].set_title("E. Cox–Snell calibration by fish-gain quantile")
    axes[1, 1].legend(fontsize=8)

    # ------------------------------------------------------------------
    # F. Stream count categories
    # ------------------------------------------------------------------

    ax = axes[1, 2]
    count_groups = ["0", "1", "2–3", "4+"]
    x = np.arange(len(count_groups))

    observed_counts = np.asarray(
        [observed["count_proportions"][group] for group in count_groups]
    )

    count_medians = []
    count_lowers = []
    count_uppers = []

    for group in count_groups:
        median, lower, upper = _scalar_band(
            replicates,
            "count_proportions",
            group,
            ci,
        )
        count_medians.append(median)
        count_lowers.append(lower)
        count_uppers.append(upper)

    count_medians = np.asarray(count_medians)
    count_lowers = np.asarray(count_lowers)
    count_uppers = np.asarray(count_uppers)

    lower_errors = np.maximum(count_medians - count_lowers, 0.0)
    upper_errors = np.maximum(count_uppers - count_medians, 0.0)
    yerr = np.vstack([lower_errors, upper_errors])

    ax.bar(
        x - 0.18,
        observed_counts,
        width=0.36,
        color="crimson",
        alpha=0.75,
        label="Observed",
    )
    ax.bar(
        x + 0.18,
        count_medians,
        width=0.36,
        color="steelblue",
        alpha=0.75,
        yerr=yerr,
        capsize=3,
        label="Model bootstrap",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(count_groups)
    ax.set_title("F. Stream event-count categories")
    ax.set_xlabel("Events per fish × trial")
    ax.set_ylabel("Fraction of streams")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    mode = (
        "simulate–refit bootstrap"
        if result["refit"]
        else "fixed-parameter predictive simulation"
    )
    fig.suptitle(
        f"Residual localization diagnostics "
        f"({mode}; {result['n_successful']}/{result['n_requested']} successful)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig, axes
