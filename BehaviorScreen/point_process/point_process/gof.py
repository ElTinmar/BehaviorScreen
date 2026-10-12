"""
Goodness of fit. Time rescaling (with Kaplan-Meier for the censored interval
ending each trial), the statistics checked against simulate-and-refit
replicates, and binned residuals.
"""
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import norm

from .dataset import PointProcessDataset
from .inference import refit_replicates
from .kaplan_meier import evaluate_step, kaplan_meier

TRIAL_BLOCKS = ("Early trials", "Middle trials", "Late trials")
ORDER_GROUPS = ("Event 1", "Events 2-3", "Events 4-6", "Events 7+")
# Upper-tail discrepancies, then predictive summaries tested two-sided.
DISCREPANCIES = ("calibration_ks", "calibration_cvm", "max_abs_event_acf")
PREDICTIVE = (
    "zero_stream_fraction",
    "multiple_event_stream_fraction",
    "mean_stream_count",
    "variance_stream_count",
    "variance_fish_total",
    "max_fish_total",
    "gap_q10",
    "gap_median",
)
CHECKS = DISCREPANCIES + PREDICTIVE


def interval_frame(dataset: PointProcessDataset, taus: Dict[Tuple[int, int], List[Tuple[float, bool]]]) -> pd.DataFrame:
    """
    One row per rescaled interval: stream, start time, order of the event
    that ends (or, censored, would have ended) it, censoring, tau and the
    raw length of exact intervals.
    """
    rows = []
    for f_idx, t_idx, t_ev in dataset.iter_streams():
        pairs = taus.get((f_idx, t_idx), [])
        starts = np.concatenate([[0.0], t_ev])[: len(pairs)]
        for order, ((tau, censored), start) in enumerate(zip(pairs, starts), start=1):
            length = np.nan if censored else float(t_ev[order - 1] - start)
            rows.append((f_idx, t_idx, float(start), order, bool(censored), float(tau), length))
    frame = pd.DataFrame(rows, columns=["fish_idx", "trial_idx", "start_s", "order", "censored", "tau", "length_s"])
    blocks = np.empty(dataset.num_trials, dtype=object)
    for label, trials in zip(TRIAL_BLOCKS, np.array_split(np.arange(dataset.num_trials), 3)):
        blocks[trials] = label
    frame["trial_block"] = blocks[frame["trial_idx"].to_numpy(int)]
    frame["order_group"] = pd.cut(frame["order"], [0, 1, 3, 6, np.inf], labels=ORDER_GROUPS).astype(str)
    return frame


def km_survival_on_grid(
    tau: np.ndarray, censored: np.ndarray, r_grid: np.ndarray, min_at_risk: int = 1
) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    KM survival of the rescaled intervals on r_grid, NaN beyond the last
    event knot with at least min_at_risk at risk and positive survival (no
    limit without censoring). Also the limit and the KM table.
    """
    knots, survival, at_risk = kaplan_meier(tau, censored, return_risk=True)
    if np.any(censored):
        supported = np.flatnonzero((np.arange(len(knots)) > 0) & (at_risk >= min_at_risk) & (survival > 0.0))
        r_limit = float(knots[supported[-1]]) if len(supported) else 0.0
    else:
        r_limit = np.inf
    return evaluate_step(r_grid, knots, survival, max_x=r_limit), r_limit, (knots, survival, at_risk)


def interval_acf(frame: pd.DataFrame, max_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lag-1..max_lags autocorrelation of normal-scored exact intervals within
    streams, and the number of pairs behind each lag.
    """
    exact = frame[~frame["censored"]]
    u = np.clip(1.0 - np.exp(-exact["tau"].to_numpy(float)), 1e-10, 1 - 1e-10)
    z = norm.ppf(u)
    sequences = np.split(z, np.flatnonzero(np.diff(exact[["fish_idx", "trial_idx"]].to_numpy(), axis=0).any(axis=1)) + 1)
    acf, pairs = np.full(max_lags, np.nan), np.zeros(max_lags, dtype=int)
    if len(z) <= max_lags or np.var(z) == 0:
        return acf, pairs
    centered = [s - z.mean() for s in sequences]
    for lag in range(1, max_lags + 1):
        products = [np.sum(s[:-lag] * s[lag:]) for s in centered if len(s) > lag]
        pairs[lag - 1] = sum(len(s) - lag for s in centered if len(s) > lag)
        if pairs[lag - 1]:
            acf[lag - 1] = sum(products) / (pairs[lag - 1] * np.var(z))
    return acf, pairs


def _tail_p_values(reference: np.ndarray, values: np.ndarray, upper: bool) -> np.ndarray:
    """Bootstrap p-values of `values` against the finite `reference` replicates."""
    reference = reference[np.isfinite(reference)]
    n = len(reference)
    above = (1.0 + np.sum(reference[None, :] >= values[:, None], axis=1)) / (n + 1.0)
    if upper:
        return above
    below = (1.0 + np.sum(reference[None, :] <= values[:, None], axis=1)) / (n + 1.0)
    return np.minimum(1.0, 2.0 * np.minimum(above, below))


def check_summary(
    observed: Dict[str, Any], replicates: Sequence[Dict[str, Any]], ci: float
) -> Tuple[pd.DataFrame, float]:
    """
    Per statistic: observed value, bootstrap median and interval, p-value.
    The omnibus p-value calibrates the smallest per-statistic p-value against
    the same minimum computed for every replicate.
    """
    alpha = (100.0 - ci) / 2.0
    records, observed_p, replicate_p = [], [], []
    for name in CHECKS:
        values = np.asarray([r[name] for r in replicates], dtype=float)
        finite = values[np.isfinite(values)]
        value = float(observed[name])
        if not len(finite) or not np.isfinite(value):
            continue
        upper = name in DISCREPANCIES
        p = float(_tail_p_values(finite, np.array([value]), upper)[0])
        observed_p.append(p)
        replicate_p.append(np.where(np.isfinite(values), _tail_p_values(finite, values, upper), np.nan))
        records.append(
            {
                "statistic": name,
                "observed": value,
                "bootstrap_median": float(np.median(finite)),
                "ci_lower": float(np.percentile(finite, alpha)),
                "ci_upper": float(np.percentile(finite, 100.0 - alpha)),
                "bootstrap_percentile": 100.0 * float(np.mean(finite <= value)),
                "p_upper": p if upper else np.nan,
                "p_two_sided": float(_tail_p_values(finite, np.array([value]), False)[0]),
                "p_value": p,
                "n_boot_finite": len(finite),
            }
        )
    summary = pd.DataFrame(records)
    if not records:
        return summary, np.nan
    replicate_min = np.nanmin(np.vstack(replicate_p), axis=0)
    omnibus = (1.0 + np.sum(replicate_min <= min(observed_p))) / (len(replicate_min) + 1.0)
    return summary, float(omnibus)


class GOFMixin:
    _survival_estimate = staticmethod(kaplan_meier)

    def gof_grids(self, dataset: PointProcessDataset, u_grid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Evaluation grids shared by the observed data and every replicate."""
        return {
            "u_grid": np.linspace(0.0, 0.995, 300) if u_grid is None else np.asarray(u_grid, dtype=float),
            "r_grid": np.linspace(0.0, 5.0, 151),
            "time_edges": np.linspace(0.0, dataset.duration_s, 5),
            "survival_grid": np.linspace(0.0, dataset.duration_s, 101),
            "gap_edges": np.geomspace(0.01, dataset.duration_s, 61),
        }

    def gof_statistics(
        self,
        dataset: PointProcessDataset,
        u_grid: np.ndarray,
        r_grid: np.ndarray,
        time_edges: np.ndarray,
        survival_grid: np.ndarray,
        gap_edges: np.ndarray,
        acf_lags: int = 20,
        min_km_at_risk: int = 1,
    ) -> Dict[str, Any]:
        """
        Everything the dashboard compares with replicates: calibration of the
        rescaled intervals on the uniform scale (KS, CvM), the same on the
        Cox-Snell scale by trial block, time in trial and event order, the
        interval autocorrelation, gaps after an event (binned on gap_edges,
        values outside go to the end bins), first-event survival, and counts
        (first events only for survival models).
        """
        if self.is_survival:
            dataset = dataset.first_events()
        frame = interval_frame(dataset, self._stream_tau_values(dataset))
        tau, censored = frame["tau"].to_numpy(float), frame["censored"].to_numpy(bool)

        survival, r_limit, (knots, km_survival, at_risk) = km_survival_on_grid(
            tau, censored, -np.log1p(-u_grid), min_km_at_risk
        )
        calibration_cdf = 1.0 - survival
        valid = np.isfinite(calibration_cdf)
        discrepancy = calibration_cdf[valid] - u_grid[valid]

        def cox_snell(selection: np.ndarray) -> np.ndarray:
            if not selection.any():
                return np.full(len(r_grid), np.nan)
            s = km_survival_on_grid(tau[selection], censored[selection], r_grid, min_km_at_risk)[0]
            return -np.log(np.where(s > 0, s, np.nan))

        start_bins = np.digitize(frame["start_s"].to_numpy(float), time_edges[1:-1])
        time_labels = [f"{a:g}-{b:g} s" for a, b in zip(time_edges[:-1], time_edges[1:])]
        acf, acf_pairs = interval_acf(frame, acf_lags)
        first_times, first_censored = dataset.first_event_times()
        first_knots, first_survival = kaplan_meier(first_times, first_censored)
        gaps = frame.loc[~frame["censored"] & (frame["order"] >= 2), "length_s"].to_numpy(float)
        gap_counts = np.histogram(np.clip(gaps, gap_edges[0], gap_edges[-1]), bins=gap_edges)[0]
        stream_counts = dataset.stream_event_counts.astype(int)
        fish_totals = dataset.fish_total_counts.astype(int)

        return {
            "calibration_ks": float(np.max(np.abs(discrepancy))) if valid.any() else np.nan,
            "calibration_cvm": float(trapezoid(discrepancy**2, x=u_grid[valid])) if valid.sum() >= 2 else np.nan,
            "max_abs_event_acf": float(np.nanmax(np.abs(acf))) if np.isfinite(acf).any() else np.nan,
            "zero_stream_fraction": float(np.mean(stream_counts == 0)),
            "multiple_event_stream_fraction": float(np.mean(stream_counts >= 2)),
            "mean_stream_count": float(np.mean(stream_counts)),
            "variance_stream_count": float(np.var(stream_counts)),
            "variance_fish_total": float(np.var(fish_totals)),
            "max_fish_total": float(np.max(fish_totals)),
            "gap_q10": float(np.percentile(gaps, 10)) if len(gaps) else np.nan,
            "gap_median": float(np.median(gaps)) if len(gaps) else np.nan,
            "n_exact": int(np.sum(~censored)),
            "n_censored": int(np.sum(censored)),
            "has_censoring": bool(censored.any()),
            "calibration_cdf": calibration_cdf,
            "r_limit": r_limit,
            "km_residual_grid": knots,
            "km_survival": km_survival,
            "km_n_at_risk": at_risk,
            "localization": {
                "trial_block": {g: cox_snell(frame["trial_block"].to_numpy() == g) for g in TRIAL_BLOCKS},
                "time_in_trial": {g: cox_snell(start_bins == i) for i, g in enumerate(time_labels)},
                "event_order": {g: cox_snell(frame["order_group"].to_numpy() == g) for g in ORDER_GROUPS},
            },
            "acf": acf,
            "acf_pairs": acf_pairs,
            "first_event_survival": evaluate_step(survival_grid, first_knots, first_survival),
            "gap_counts": gap_counts,
            "stream_counts": stream_counts,
            "fish_totals": np.sort(fish_totals),
        }

    def time_rescaling(self, dataset: PointProcessDataset, acf_lags: int = 20) -> Dict[str, Any]:
        """Rescaled intervals pooled over streams, their KM estimate and autocorrelation."""
        self._require_fitted()
        frame = interval_frame(dataset, self._stream_tau_values(dataset))
        residuals, censored = frame["tau"].to_numpy(float), frame["censored"].to_numpy(bool)
        residual_grid, survival = kaplan_meier(residuals, censored)
        acf, pairs = interval_acf(frame, acf_lags)
        with np.errstate(divide="ignore"):
            acf_conf = 1.96 / np.sqrt(pairs)
        return {
            "residuals": residuals,
            "censored": censored,
            "residual_grid": residual_grid,
            "survival_estimate": survival,
            "n_rescaled": len(residuals),
            "n_exact": int(np.sum(~censored)),
            "acf_lags": np.arange(1, acf_lags + 1),
            "acf": acf,
            "acf_conf": acf_conf,
        }

    def parametric_gof_bootstrap(
        self,
        dataset: PointProcessDataset,
        n_boot: int = 300,
        seed: int = 42,
        ci: float = 95.0,
        acf_lags: int = 20,
        min_km_at_risk: int = 1,
        min_envelope_fraction: float = 0.80,
        refit_n_starts: int = 1,
        n_jobs: int = -1,
        u_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        gof_statistics of the data against simulate-and-refit replicates:
        check table with an omnibus p-value, pointwise band and simultaneous
        envelope of the calibration curve. Pointwise bands are kept where at
        least min_envelope_fraction of replicates reach; the simultaneous
        envelope spans the range from u = 0 supported by the data and by
        that many replicates.
        """
        self._require_fitted()
        grids = self.gof_grids(dataset, u_grid)
        statistic = partial(GOFMixin.gof_statistics, **grids, acf_lags=acf_lags, min_km_at_risk=min_km_at_risk)
        observed = statistic(self, dataset)
        replicates, errors = refit_replicates(
            self, dataset, statistic, n_boot, seed, refit_n_starts=refit_n_starts, n_jobs=n_jobs,
            desc="Parametric GOF bootstrap",
        )
        if not replicates:
            raise RuntimeError(f"All parametric GOF bootstrap replicates failed. Example errors: {errors[:5]}")
        summary, omnibus_p = check_summary(observed, replicates, ci)

        u = grids["u_grid"]
        alpha = (100.0 - ci) / 2.0
        curves = np.array([r["calibration_cdf"] for r in replicates])
        contributors = np.isfinite(curves).sum(axis=0)
        reliable = contributors >= max(1, int(np.ceil(min_envelope_fraction * len(replicates))))
        lower, median, upper = (np.full(len(u), np.nan) for _ in range(3))
        if reliable.any():
            lower[reliable], median[reliable], upper[reliable] = np.nanpercentile(
                curves[:, reliable], [alpha, 50.0, 100.0 - alpha], axis=0
            )

        common = reliable & np.isfinite(observed["calibration_cdf"])
        n_common = int(np.argmin(common)) if not common.all() else len(u)
        common_range = np.arange(len(u)) < n_common
        complete = np.isfinite(curves[:, common_range]).all(axis=1) if n_common else np.zeros(len(curves), bool)
        simultaneous_lower, simultaneous_upper = np.full(len(u), np.nan), np.full(len(u), np.nan)
        critical = np.nan
        if n_common and complete.any():
            sup = np.max(np.abs(curves[complete][:, common_range] - u[common_range]), axis=1)
            critical = float(np.percentile(sup, ci))
            simultaneous_lower[common_range] = np.clip(u[common_range] - critical, 0.0, 1.0)
            simultaneous_upper[common_range] = np.clip(u[common_range] + critical, 0.0, 1.0)

        return {
            **grids,
            "observed": observed,
            "replicates": replicates,
            "summary": summary,
            "omnibus_p": omnibus_p,
            "bootstrap_statistics": pd.DataFrame([{name: r[name] for name in CHECKS} for r in replicates]),
            "n_requested": int(n_boot),
            "n_successful": len(replicates),
            "n_failed": len(errors),
            "failure_messages": errors,
            "ci": float(ci),
            "min_km_at_risk": int(min_km_at_risk),
            "min_envelope_fraction": float(min_envelope_fraction),
            "observed_calibration_cdf": observed["calibration_cdf"],
            "bootstrap_cdf_median": median,
            "bootstrap_cdf_lower": lower,
            "bootstrap_cdf_upper": upper,
            "bootstrap_cdf_contributors": contributors,
            "bootstrap_cdf_reliable_grid": reliable,
            "simultaneous_cdf_lower": simultaneous_lower,
            "simultaneous_cdf_upper": simultaneous_upper,
            "simultaneous_critical_value": critical,
            "n_simultaneous_curves": int(complete.sum()),
        }

    def compute_residuals(self, dataset: PointProcessDataset) -> Dict[str, np.ndarray]:
        """
        Counts per (trial, time bin) against the model, pooled over fish.
        Recurrent models: expected rate x fish x bin width, negative-binomial
        variance with dispersion r per fish (r x fish for the pooled count).
        Survival models: first events against the drop of the marginal
        survival curve, Poisson variance.
        """
        self._require_fitted()
        n_fish = dataset.n_fish_per_trial[:, None]
        if self.is_survival:
            dataset = dataset.first_events()
            survival = np.array([self.population_survival_curve(dataset, trial=m)[1] for m in range(dataset.num_trials)])
            mu_pred = n_fish * -np.diff(survival, axis=1, prepend=1.0)
        else:
            mu_pred = self.compute_expected_rate(dataset) * n_fish * dataset.binning_dt
        r_eff = np.inf if self.is_survival or np.isinf(self.dispersion_r) else self.dispersion_r * n_fish
        y_obs = dataset.time_trial_histogram_counts

        with np.errstate(divide="ignore", invalid="ignore"):
            var_pred = mu_pred + mu_pred**2 / r_eff
            pearson = (y_obs - mu_pred) / np.sqrt(var_pred)
            term = np.where(y_obs > 0, y_obs * np.log(y_obs / mu_pred), 0.0)
            if np.isscalar(r_eff):
                deviance_sq = 2.0 * (term - (y_obs - mu_pred))
            else:
                deviance_sq = 2.0 * (term - (y_obs + r_eff) * np.log((y_obs + r_eff) / (mu_pred + r_eff)))
            deviance = np.sign(y_obs - mu_pred) * np.sqrt(np.maximum(0.0, deviance_sq))

        no_fish = dataset.n_fish_per_trial == 0
        pearson[no_fish] = np.nan
        deviance[no_fish] = np.nan
        return {
            "y_obs": y_obs,
            "mu_pred": mu_pred,
            "var_pred": var_pred,
            "r_eff": r_eff,
            "pearson_residuals": pearson,
            "deviance_residuals": deviance,
        }
