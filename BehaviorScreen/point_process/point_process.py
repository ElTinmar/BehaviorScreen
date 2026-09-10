from typing import List, Tuple, Dict, Optional, Any, Union, ClassVar
import copy
import re

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import trapezoid

from .tqdm_joblib import tqdm_joblib
from .dataset import PointProcessDataset

def _fit_single_bootstrap(seed_seq, dataset: PointProcessDataset, model: "PointProcess"):
    rng = np.random.default_rng(seed_seq)
    ds_boot = dataset.resample(rng)
    model_copy = copy.deepcopy(model)
    model_copy.initial_guesses = list(model.params_) # warm-start at parent MLE
    try:
        model_copy.fit(ds_boot)
        return model_copy.params_
    except Exception:
        return None

def _fit_single_parametric_gof(
    seed_seq,
    dataset: PointProcessDataset,
    fitted_model: "PointProcess",
    u_grid: np.ndarray,
    acf_lags: int,
    min_km_at_risk: int,
    refit_n_starts: int,
):
    """
    One parametric-bootstrap GOF replicate:

        simulate complete dataset
            -> refit model
            -> recompute identical GOF statistics
    """
    rng = np.random.default_rng(seed_seq)

    try:
        model_b = copy.deepcopy(fitted_model)

        # Generate using the observed-data fitted parameters.
        dataset_b = model_b.simulate_dataset(
            template=dataset,
            rng=rng,
        )

        # Refit to the generated dataset.
        model_b.initial_guesses = list(
            np.asarray(fitted_model.params_, dtype=float)
        )

        if refit_n_starts <= 1:
            model_b.fit(dataset_b)
        else:
            model_b.fit_multistart(
                dataset_b,
                n_starts=refit_n_starts,
                seed=int(rng.integers(0, 2**31 - 1)),
                n_jobs=1,  # avoid nested parallelism
            )

        return model_b.gof_statistics(
            dataset_b,
            u_grid=u_grid,
            acf_lags=acf_lags,
            min_km_at_risk=min_km_at_risk,
        )

    except Exception as exc:
        return {
            "_failed": True,
            "_error": repr(exc),
        }

    
class PointProcess:
    # Heuristic: parameter names matching these patterns get log-uniform
    # sampling and log-normal jitter instead of linear -- they are
    # scale/time-constant-like quantities where the likelihood surface is
    # naturally curved in log-space (tau, sigma, beta_hawkes, r_dispersion,
    # A_ripple/tau_excitation, etc).
    _LOG_SCALE_PATTERN = re.compile(
        r"(tau|sigma|beta|r_dispersion|A_excitation|A_ripple)", re.IGNORECASE
    )

    _THINNING_SAFETY_MARGIN: ClassVar[float] = 1.1
    """
    Multiplicative safety factor applied to a grid-estimated intensity
    maximum, used as the proposal upper bound for Ogata thinning simulation.
    Needed because the TRUE continuous maximum can exceed the max observed on
    a finite grid (however fine) between grid points -- 1.1 (10% headroom) is
    a conservative default, not a fitted or derived quantity. If thinning's
    acceptance rate is ever suspiciously low (most proposals rejected) for a
    sharply-peaked kernel, consider raising this rather than assuming a bug
    elsewhere.
    """

    def __init__(self, integration_dt: float = 0.02):
        self.name: str = ""
        self.latex_formula = ""
        self.integration_dt = integration_dt
        self.fit_result: Optional[Any] = None
        self.params_: Optional[np.ndarray] = None
        self.param_dict_: Dict[str, float] = {}
        self.initial_guesses: List[float] = None
        self.bounds: List[Tuple[Optional[float], Optional[float]]] = None
        self.param_names: List[str] = None

    def _nll(self, params: List[float], dataset: PointProcessDataset): 
        raise NotImplementedError

    def fit(self, dataset: PointProcessDataset, method: str = 'L-BFGS-B', **kwargs):
        res = minimize(
            self._nll,
            x0=self.initial_guesses,
            args=(dataset,),
            method=method,
            bounds=self.bounds,
            **kwargs
        )

        valid = (
            res.success
            and np.isfinite(res.fun)
            and np.all(np.isfinite(res.x))
        )
        if not valid:
            raise RuntimeError(
                f"Optimization failed for {self.name}: "
                f"success={res.success}, fun={res.fun}, message={res.message}"
            )
    
        self.fit_result = res
        self.set_params(res.x)
        return self

    def set_params(self, params: np.ndarray) -> None:
        self.params_ = np.asarray(params, dtype=float)
        self.param_dict_ = dict(zip(self.param_names, self.params_))

    def _infer_log_scale_params(self) -> set:
        return {p for p in self.param_names if self._LOG_SCALE_PATTERN.search(p)}

    def _sampling_bounds(self, i: int, default: np.ndarray, fallback_scale: float = 10.0):
        lo, hi = self.bounds[i]
        d = default[i]
        if lo is None:
            lo = d - fallback_scale * (abs(d) + 1.0)
        if hi is None:
            hi = d + fallback_scale * (abs(d) + 1.0)
        return lo, hi

    def _generate_multistart_inits(
        self,
        rng: np.random.Generator,
        n_starts: int,
        jitter_frac: float = 0.3,
        log_scale_params: Optional[set] = None,
    ) -> List[np.ndarray]:
        default = np.asarray(self.initial_guesses, dtype=float)
        k = len(default)
        log_scale = log_scale_params if log_scale_params is not None else self._infer_log_scale_params()

        starts = [default.copy()]
        n_jitter = (n_starts - 1) // 2
        n_uniform = n_starts - 1 - n_jitter

        # (a) local jitter around the default guess
        for _ in range(n_jitter):
            x = default.copy()
            for i in range(k):
                lo, hi = self._sampling_bounds(i, default)
                name = self.param_names[i]
                if name in log_scale and default[i] > 0:
                    x[i] = default[i] * np.exp(rng.normal(0, jitter_frac))
                else:
                    span = (hi - lo) if np.isfinite(hi - lo) else max(abs(default[i]), 1.0)
                    x[i] = default[i] + rng.normal(0, jitter_frac * span)
                x[i] = np.clip(x[i], lo, hi)
            starts.append(x)

        # (b) global draws across the (sampling) bounds
        for _ in range(n_uniform):
            x = np.empty(k)
            for i in range(k):
                lo, hi = self._sampling_bounds(i, default)
                name = self.param_names[i]
                if name in log_scale and lo > 0:
                    x[i] = np.exp(rng.uniform(np.log(lo), np.log(hi)))
                else:
                    x[i] = rng.uniform(lo, hi)
            starts.append(x)

        return starts

    def fit_multistart(
        self,
        dataset: PointProcessDataset,
        n_starts: int = 20,
        method: str = 'L-BFGS-B',
        seed: int = 0,
        n_jobs: int = -1,
        log_scale_params: Optional[set] = None,
        tol_same_optimum: float = 2e-2,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Refit from n_starts different initializations, keep the best
        converged result as self.params_/self.fit_result (same
        postcondition as .fit()), and return a per-start table so
        multimodality is visible rather than hidden by AIC silently
        picking whichever local optimum was reached.
        """
        rng = np.random.default_rng(seed)
        inits = self._generate_multistart_inits(rng, n_starts, log_scale_params=log_scale_params)

        def _run_one(x0):
            try:
                res = minimize(self._nll, x0=x0, args=(dataset,), method=method,
                                bounds=self.bounds, **kwargs)
                ok = res.success and np.isfinite(res.fun) and np.all(np.isfinite(res.x))
                return res if ok else None
            except Exception:
                return None

        with tqdm_joblib(tqdm(total=n_starts, desc="Fit multistart")):
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_run_one)(x0) for x0 in inits
            )

        valid = [r for r in results if r is not None]
        if not valid:
            raise RuntimeError(f"All {n_starts} multistart optimizations failed for {self.name}.")

        best = min(valid, key=lambda r: r.fun)
        n_converged = len(valid)
        n_at_global = sum(1 for r in valid if r.fun <= best.fun + tol_same_optimum)

        status = "OK (looks unimodal)" if n_at_global == n_converged else \
                 f"WARNING: {n_converged - n_at_global} start(s) converged to a DIFFERENT, worse optimum " \
                 f"-- {self.name} may be poorly identified; treat AIC comparisons involving it with caution."
        print(f"[{self.name}] multistart: {n_converged}/{n_starts} converged, "
              f"best NLL={best.fun:.3f}. {status}")

        self.fit_result = best
        self.set_params(best.x)

        return pd.DataFrame({
            "start_idx": range(len(results)),
            "converged": [r is not None for r in results],
            "nll": [r.fun if r is not None else np.nan for r in results],
        })
    
    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        raise NotImplementedError

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray: 
        raise NotImplementedError

    def predict(self, t, trial, **kwargs):
        raise NotImplementedError

    def bootstrap(
        self,
        dataset: PointProcessDataset,
        n_boot: int = 500,
        seed: int = 42,
        ci: float = 95.0,
        n_jobs: int = -1,
    ) -> pd.DataFrame:

        if self.params_ is None:
            raise ValueError("Model must be fitted before running bootstrap.")

        seeds = np.random.SeedSequence(seed).spawn(n_boot)

        with tqdm_joblib(tqdm(total=n_boot, desc="Bootstrap")):
            boot_results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_fit_single_bootstrap)(s, dataset, self)
                for s in seeds
            )
        
        valid_boot_params = [p for p in boot_results if p is not None]
        print(f"Bootstrap fit: {len(valid_boot_params)}/{n_boot}")
        if len(valid_boot_params) == 0:
            raise RuntimeError("All bootstrap optimization attempts failed.")

        boot_params = np.array(valid_boot_params)

        alpha = (100.0 - ci) / 2.0
        return pd.DataFrame(
            [
                {
                    "parameter": name,
                    "fitted_val": self.params_[i],
                    "boot_mean": np.mean(boot_params[:, i]),
                    "boot_std": np.std(boot_params[:, i]),
                    f"ci_{alpha:.1f}%": np.percentile(boot_params[:, i], alpha),
                    f"ci_{100-alpha:.1f}%": np.percentile(
                        boot_params[:, i], 100 - alpha
                    ),
                }
                for i, name in enumerate(self.param_names)
            ]
        )
    
    @property
    def log_likelihood(self) -> float:
        return -self.fit_result.fun if self.fit_result else np.nan

    @property
    def aic(self) -> float:
        k = len(self.params_)
        return 2 * k - 2 * self.log_likelihood

    def estimate_fish_gains(self, dataset):
        raise NotImplementedError

    def mixed_effects_likelihood_terms(self, dataset, params) -> Tuple[float, np.ndarray, np.ndarray]:
        """Used for mixed-effect model at the fish level"""
        raise NotImplementedError

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng
    ) -> np.ndarray:
        raise NotImplementedError

    def simulate_dataset(
        self,
        template: PointProcessDataset,
        rng: Optional[np.random.Generator] = None,
    ) -> PointProcessDataset:
        """
        Simulate one complete dataset from the fitted model while preserving the
        observed experimental design:

        - same number of fish and trials;
        - same fish_trial_mask;
        - same trial duration and trial indices;
        - one newly drawn frailty per fish, shared across that fish's trials;
        - within-trial event history resets because simulate_stream() is called
        independently for each trial.

        The fitted model parameters are held fixed during generation.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted before simulating a dataset.")

        rng = rng or np.random.default_rng()

        # Shape = (num_fish, 1): one frailty draw per fish, reused over trials.
        fish_gains = self._draw_fish_gains(
            num_fish=template.num_fish,
            n_sims=1,
            rng=rng,
        )[:, 0]

        event_times = []
        event_trials = []
        event_fish = []

        for f_idx, t_idx, _ in template.iter_streams():
            simulated_events = np.asarray(
                self.simulate_stream(
                    dataset=template,
                    t_idx=t_idx,
                    gain=float(fish_gains[f_idx]),
                    rng=rng,
                ),
                dtype=float,
            )

            if len(simulated_events) == 0:
                continue

            event_times.append(simulated_events)
            event_trials.append(
                np.full(len(simulated_events), t_idx, dtype=int)
            )
            event_fish.append(
                np.full(len(simulated_events), f_idx, dtype=int)
            )

        return PointProcessDataset(
            event_times=(
                np.concatenate(event_times)
                if event_times else np.array([], dtype=float)
            ),
            event_trials_idx=(
                np.concatenate(event_trials)
                if event_trials else np.array([], dtype=int)
            ),
            event_fish_idx=(
                np.concatenate(event_fish)
                if event_fish else np.array([], dtype=int)
            ),
            fish_trial_mask=template.fish_trial_mask.copy(),
            fish_ids=template.fish_ids.copy(),
            bout_name=template.bout_name,
            laterality=template.laterality,
            duration_s=template.duration_s,
            binning_dt=template.binning_dt,
        )

    @staticmethod
    def _evaluate_step_function(
        x: np.ndarray,
        step_x: np.ndarray,
        step_y: np.ndarray,
        max_x: float = np.inf,
    ) -> np.ndarray:
        """
        Evaluate a right-continuous step function.

        Values larger than ``max_x`` are returned as NaN. This prevents a
        censored KM curve from being extrapolated into an unsupported tail.

        For an uncensored empirical distribution, use max_x=np.inf so the final
        step is continued to the right.
        """
        x = np.asarray(x, dtype=float)
        step_x = np.asarray(step_x, dtype=float)
        step_y = np.asarray(step_y, dtype=float)

        if step_x.ndim != 1 or step_y.ndim != 1:
            raise ValueError("step_x and step_y must be one-dimensional.")
        if len(step_x) != len(step_y):
            raise ValueError("step_x and step_y must have equal length.")
        if len(step_x) == 0:
            return np.full_like(x, np.nan, dtype=float)
        if np.any(np.diff(step_x) < 0):
            raise ValueError("step_x must be sorted.")

        result = np.full_like(x, np.nan, dtype=float)

        idx = np.searchsorted(step_x, x, side="right") - 1
        valid = (idx >= 0) & (x <= max_x)

        clipped_idx = np.clip(idx[valid], 0, len(step_y) - 1)
        result[valid] = step_y[clipped_idx]

        return result

    def gof_statistics(
        self,
        dataset: PointProcessDataset,
        u_grid: Optional[np.ndarray] = None,
        acf_lags: int = 20,
        min_km_at_risk: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute scalar GOF statistics and time-rescaling calibration curves.

        For uncensored recurrent-process residuals, the empirical residual CDF
        is evaluated over the complete Uniform-scale grid.

        For censored residuals, a Kaplan-Meier estimate is used. Calibration is
        evaluated only through the last exact-event knot satisfying:

            n_at_risk >= min_km_at_risk

        and having positive post-jump survival. This prevents an unsupported or
        numerically infinite KM tail from affecting the diagnostic.

        The same rule is applied to the observed dataset and every parametric-
        bootstrap replicate.
        """
        if self.params_ is None:
            raise ValueError(
                "Model must be fitted before computing GOF statistics."
            )
        if min_km_at_risk < 1:
            raise ValueError("min_km_at_risk must be at least 1.")

        if u_grid is None:
            # Avoid u=1 because r=-log(1-u) diverges.
            u_grid = np.linspace(0.0, 0.995, 300)
        else:
            u_grid = np.asarray(u_grid, dtype=float)

        if u_grid.ndim != 1:
            raise ValueError("u_grid must be one-dimensional.")
        if np.any(~np.isfinite(u_grid)):
            raise ValueError("u_grid must be finite.")
        if np.any((u_grid < 0.0) | (u_grid >= 1.0)):
            raise ValueError("u_grid values must lie in [0, 1).")
        if np.any(np.diff(u_grid) <= 0):
            raise ValueError("u_grid must be strictly increasing.")

        tr = self.time_rescaling(
            dataset,
            acf_lags=acf_lags,
        )

        residuals = np.asarray(tr["residuals"], dtype=float)
        censored = np.asarray(tr["censored"], dtype=bool)
        has_censoring = bool(np.any(censored))

        residual_grid, km_survival, km_n_at_risk = self._survival_estimate(
            residuals,
            censored,
            return_risk=True,
        )

        # Exact KM knots, retained for the Cox-Snell plot.
        km_u = 1.0 - np.exp(-residual_grid)
        km_cdf = 1.0 - km_survival

        if has_censoring:
            knot_index = np.arange(len(residual_grid))

            valid_failure_knots = (
                (knot_index > 0)
                & (km_n_at_risk >= min_km_at_risk)
                & (km_survival > 0.0)
            )

            if np.any(valid_failure_knots):
                last_valid_idx = np.where(valid_failure_knots)[0][-1]
                r_limit = float(residual_grid[last_valid_idx])
                u_limit = float(km_u[last_valid_idx])
            else:
                r_limit = 0.0
                u_limit = 0.0
        else:
            # An uncensored empirical CDF is valid after its largest observation:
            # it simply remains equal to one. Do not truncate it at the final
            # event.
            r_limit = np.inf
            u_limit = float(u_grid[-1])

        r_eval = -np.log1p(-u_grid)

        survival_on_grid = self._evaluate_step_function(
            x=r_eval,
            step_x=residual_grid,
            step_y=km_survival,
            max_x=r_limit,
        )
        calibration_cdf = 1.0 - survival_on_grid

        valid_grid = np.isfinite(calibration_cdf)

        if np.any(valid_grid):
            discrepancy = calibration_cdf[valid_grid] - u_grid[valid_grid]
            calibration_ks = float(np.max(np.abs(discrepancy)))

            if np.sum(valid_grid) >= 2:
                calibration_cvm = float(
                    trapezoid(
                        discrepancy**2,
                        x=u_grid[valid_grid],
                    )
                )
            else:
                calibration_cvm = np.nan
        else:
            calibration_ks = np.nan
            calibration_cvm = np.nan

        acf = np.asarray(tr["acf"], dtype=float)
        finite_acf = acf[np.isfinite(acf)]
        max_abs_event_acf = (
            float(np.max(np.abs(finite_acf)))
            if len(finite_acf) > 0 else np.nan
        )

        stream_counts = dataset.stream_event_counts.astype(float)
        fish_totals = dataset.fish_total_counts.astype(float)

        return {
            # Calibration discrepancies: larger means worse.
            "calibration_ks": calibration_ks,
            "calibration_cvm": calibration_cvm,
            "max_abs_event_acf": max_abs_event_acf,

            # Predictive summaries.
            "zero_stream_fraction": (
                float(np.mean(stream_counts == 0))
                if len(stream_counts) else np.nan
            ),
            "multiple_event_stream_fraction": (
                float(np.mean(stream_counts >= 2))
                if len(stream_counts) else np.nan
            ),
            "mean_stream_count": (
                float(np.mean(stream_counts))
                if len(stream_counts) else np.nan
            ),
            "variance_stream_count": (
                float(np.var(stream_counts))
                if len(stream_counts) else np.nan
            ),
            "mean_fish_total": (
                float(np.mean(fish_totals))
                if len(fish_totals) else np.nan
            ),
            "variance_fish_total": (
                float(np.var(fish_totals))
                if len(fish_totals) else np.nan
            ),
            "max_fish_total": (
                float(np.max(fish_totals))
                if len(fish_totals) else np.nan
            ),

            # Residual metadata.
            "n_rescaled": int(tr["n_rescaled"]),
            "n_exact": int(tr["n_exact"]),
            "n_censored": int(np.sum(censored)),
            "has_censoring": has_censoring,
            "min_km_at_risk": int(min_km_at_risk),

            # Common-grid representation for bootstrap envelopes.
            "u_grid": u_grid,
            "r_eval_grid": r_eval,
            "calibration_cdf": calibration_cdf,
            "r_limit": r_limit,
            "u_limit": u_limit,

            # Exact observed KM knots for Cox-Snell plotting.
            "km_residual_grid": residual_grid,
            "km_survival": km_survival,
            "km_n_at_risk": km_n_at_risk,
            "km_u": km_u,
            "km_cdf": km_cdf,
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
        Parametric-bootstrap goodness-of-fit assessment.

        Each replicate performs:

            simulate -> refit -> recompute GOF

        For censored residuals, each replicate's KM curve is truncated using the
        same minimum-risk-set rule. Pointwise bands are reported only where at
        least ``min_envelope_fraction`` of successful replicates contribute.

        The simultaneous envelope is constructed over the common range supported
        by the observed KM curve and sufficiently many bootstrap curves.
        """
        if self.params_ is None:
            raise ValueError(
                "Model must be fitted before running parametric GOF bootstrap."
            )
        if n_boot < 1:
            raise ValueError("n_boot must be at least 1.")
        if not 0.0 < ci < 100.0:
            raise ValueError("ci must lie strictly between 0 and 100.")
        if not 0.0 < min_envelope_fraction <= 1.0:
            raise ValueError(
                "min_envelope_fraction must lie in (0, 1]."
            )

        if u_grid is None:
            u_grid = np.linspace(0.0, 0.995, 300)
        else:
            u_grid = np.asarray(u_grid, dtype=float)

        observed = self.gof_statistics(
            dataset,
            u_grid=u_grid,
            acf_lags=acf_lags,
            min_km_at_risk=min_km_at_risk,
        )

        seeds = np.random.SeedSequence(seed).spawn(n_boot)

        with tqdm_joblib(
            tqdm(total=n_boot, desc="Parametric GOF bootstrap")
        ):
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_fit_single_parametric_gof)(
                    seed_seq=s,
                    dataset=dataset,
                    fitted_model=self,
                    u_grid=u_grid,
                    acf_lags=acf_lags,
                    min_km_at_risk=min_km_at_risk,
                    refit_n_starts=refit_n_starts,
                )
                for s in seeds
            )

        failures = [
            result
            for result in results
            if result is None or result.get("_failed", False)
        ]
        valid = [
            result
            for result in results
            if result is not None and not result.get("_failed", False)
        ]

        print(
            f"Parametric GOF bootstrap: {len(valid)}/{n_boot} successful "
            f"({len(failures)} failures)."
        )

        if not valid:
            example_errors = [
                result.get("_error", "Unknown error")
                for result in failures[:5]
                if isinstance(result, dict)
            ]
            raise RuntimeError(
                "All parametric GOF bootstrap replicates failed. "
                f"Example errors: {example_errors}"
            )

        alpha = (100.0 - ci) / 2.0

        # ------------------------------------------------------------------
        # Scalar statistic summaries
        # ------------------------------------------------------------------

        scalar_names = [
            "calibration_ks",
            "calibration_cvm",
            "max_abs_event_acf",
            "zero_stream_fraction",
            "multiple_event_stream_fraction",
            "mean_stream_count",
            "variance_stream_count",
            "mean_fish_total",
            "variance_fish_total",
            "max_fish_total",
            "n_rescaled",
            "n_exact",
            "n_censored",
        ]

        upper_tail_discrepancies = {
            "calibration_ks",
            "calibration_cvm",
            "max_abs_event_acf",
        }

        summary_records = []

        for name in scalar_names:
            observed_value = float(observed[name])

            bootstrap_values = np.asarray(
                [result[name] for result in valid],
                dtype=float,
            )
            bootstrap_values = bootstrap_values[
                np.isfinite(bootstrap_values)
            ]

            if len(bootstrap_values) == 0:
                summary_records.append({
                    "statistic": name,
                    "observed": observed_value,
                    "bootstrap_mean": np.nan,
                    "bootstrap_median": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "bootstrap_percentile": np.nan,
                    "p_upper": np.nan,
                    "p_two_sided": np.nan,
                    "n_boot_finite": 0,
                })
                continue

            bootstrap_percentile = (
                100.0 * np.mean(bootstrap_values <= observed_value)
            )

            if (
                name in upper_tail_discrepancies
                and np.isfinite(observed_value)
            ):
                p_upper = (
                    1.0
                    + np.sum(bootstrap_values >= observed_value)
                ) / (len(bootstrap_values) + 1.0)
            else:
                p_upper = np.nan

            if np.isfinite(observed_value):
                lower_tail = (
                    1.0
                    + np.sum(bootstrap_values <= observed_value)
                ) / (len(bootstrap_values) + 1.0)

                upper_tail = (
                    1.0
                    + np.sum(bootstrap_values >= observed_value)
                ) / (len(bootstrap_values) + 1.0)

                p_two_sided = min(
                    1.0,
                    2.0 * min(lower_tail, upper_tail),
                )
            else:
                p_two_sided = np.nan

            summary_records.append({
                "statistic": name,
                "observed": observed_value,
                "bootstrap_mean": float(np.mean(bootstrap_values)),
                "bootstrap_median": float(np.median(bootstrap_values)),
                "ci_lower": float(
                    np.percentile(bootstrap_values, alpha)
                ),
                "ci_upper": float(
                    np.percentile(
                        bootstrap_values,
                        100.0 - alpha,
                    )
                ),
                "bootstrap_percentile": bootstrap_percentile,
                "p_upper": p_upper,
                "p_two_sided": p_two_sided,
                "n_boot_finite": len(bootstrap_values),
            })

        summary = pd.DataFrame(summary_records)

        # ------------------------------------------------------------------
        # Bootstrap residual curves
        # ------------------------------------------------------------------

        bootstrap_cdfs = np.asarray(
            [result["calibration_cdf"] for result in valid],
            dtype=float,
        )

        bootstrap_r_limits = np.asarray(
            [result["r_limit"] for result in valid],
            dtype=float,
        )
        bootstrap_u_limits = np.asarray(
            [result["u_limit"] for result in valid],
            dtype=float,
        )

        n_contributors = np.sum(
            np.isfinite(bootstrap_cdfs),
            axis=0,
        )

        min_contributors = max(
            1,
            int(np.ceil(min_envelope_fraction * len(valid))),
        )

        reliable_grid = n_contributors >= min_contributors

        cdf_lower = np.full(len(u_grid), np.nan, dtype=float)
        cdf_upper = np.full(len(u_grid), np.nan, dtype=float)
        cdf_median = np.full(len(u_grid), np.nan, dtype=float)

        if np.any(reliable_grid):
            cdf_lower[reliable_grid] = np.nanpercentile(
                bootstrap_cdfs[:, reliable_grid],
                alpha,
                axis=0,
            )
            cdf_upper[reliable_grid] = np.nanpercentile(
                bootstrap_cdfs[:, reliable_grid],
                100.0 - alpha,
                axis=0,
            )
            cdf_median[reliable_grid] = np.nanmedian(
                bootstrap_cdfs[:, reliable_grid],
                axis=0,
            )

        # ------------------------------------------------------------------
        # Simultaneous envelope
        # ------------------------------------------------------------------

        observed_curve = np.asarray(
            observed["calibration_cdf"],
            dtype=float,
        )

        # Common display range:
        #   - supported by the observed curve;
        #   - supported by the required fraction of bootstrap curves.
        common_grid = (
            reliable_grid
            & np.isfinite(observed_curve)
        )

        simultaneous_lower = np.full(
            len(u_grid),
            np.nan,
            dtype=float,
        )
        simultaneous_upper = np.full(
            len(u_grid),
            np.nan,
            dtype=float,
        )
        simultaneous_critical_value = np.nan
        n_simultaneous_curves = 0

        common_indices = np.where(common_grid)[0]

        if len(common_indices) > 0:
            # Require a contiguous range starting at u=0. This avoids retaining
            # isolated tail points after missing values.
            first_bad = np.where(~common_grid[:common_indices[-1] + 1])[0]

            if len(first_bad) > 0:
                common_last_idx = first_bad[0] - 1
            else:
                common_last_idx = common_indices[-1]

            if common_last_idx >= 0:
                common_mask = (
                    np.arange(len(u_grid)) <= common_last_idx
                )

                complete_rows = np.all(
                    np.isfinite(bootstrap_cdfs[:, common_mask]),
                    axis=1,
                )

                n_simultaneous_curves = int(np.sum(complete_rows))

                if n_simultaneous_curves > 0:
                    deviations = np.abs(
                        bootstrap_cdfs[complete_rows][:, common_mask]
                        - u_grid[None, common_mask]
                    )
                    sup_deviations = np.max(
                        deviations,
                        axis=1,
                    )

                    simultaneous_critical_value = float(
                        np.percentile(sup_deviations, ci)
                    )

                    simultaneous_lower[common_mask] = np.clip(
                        u_grid[common_mask]
                        - simultaneous_critical_value,
                        0.0,
                        1.0,
                    )
                    simultaneous_upper[common_mask] = np.clip(
                        u_grid[common_mask]
                        + simultaneous_critical_value,
                        0.0,
                        1.0,
                    )

        # ------------------------------------------------------------------
        # Replicate-level scalar table
        # ------------------------------------------------------------------

        bootstrap_scalar_records = []

        for replicate_index, result in enumerate(valid):
            row = {"replicate": replicate_index}

            for name in scalar_names:
                row[name] = result[name]

            row["r_limit"] = result["r_limit"]
            row["u_limit"] = result["u_limit"]
            bootstrap_scalar_records.append(row)

        bootstrap_statistics = pd.DataFrame(
            bootstrap_scalar_records
        )

        failure_messages = [
            result.get("_error", "Unknown error")
            for result in failures
            if isinstance(result, dict)
        ]

        return {
            "observed": observed,
            "summary": summary,
            "bootstrap_statistics": bootstrap_statistics,

            "n_requested": int(n_boot),
            "n_successful": int(len(valid)),
            "n_failed": int(len(failures)),
            "failure_messages": failure_messages,

            "ci": float(ci),
            "min_km_at_risk": int(min_km_at_risk),
            "min_envelope_fraction": float(min_envelope_fraction),

            "u_grid": u_grid,
            "r_grid": -np.log1p(-u_grid),

            "observed_calibration_cdf": observed_curve,

            "bootstrap_cdf_median": cdf_median,
            "bootstrap_cdf_lower": cdf_lower,
            "bootstrap_cdf_upper": cdf_upper,

            "bootstrap_r_limits": bootstrap_r_limits,
            "bootstrap_u_limits": bootstrap_u_limits,
            "bootstrap_cdf_contributors": n_contributors,
            "bootstrap_cdf_reliable_grid": reliable_grid,

            "simultaneous_cdf_lower": simultaneous_lower,
            "simultaneous_cdf_upper": simultaneous_upper,
            "simultaneous_critical_value": (
                simultaneous_critical_value
            ),
            "n_simultaneous_curves": n_simultaneous_curves,
        }

    def plot_parametric_gof_calibration(
        self,
        gof_result: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        show_pointwise_band: bool = False,
    ) -> plt.Axes:
        """
        Plot the observed time-rescaling CDF against parametric-bootstrap
        envelopes obtained after simulating and refitting the model.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        u = gof_result["u_grid"]
        observed_cdf = gof_result["observed_calibration_cdf"]

        if show_pointwise_band:
            ax.fill_between(
                u,
                gof_result["bootstrap_cdf_lower"],
                gof_result["bootstrap_cdf_upper"],
                color="steelblue",
                alpha=0.18,
                label="Pointwise parametric-bootstrap band",
            )

        ax.fill_between(
            u,
            gof_result["simultaneous_cdf_lower"],
            gof_result["simultaneous_cdf_upper"],
            color="gray",
            alpha=0.25,
            label="Simultaneous parametric-bootstrap envelope",
        )

        ax.plot(
            u,
            gof_result["bootstrap_cdf_median"],
            color="steelblue",
            linestyle=":",
            linewidth=1.5,
            label="Bootstrap median fitted-residual CDF",
        )

        ax.step(
            u,
            observed_cdf,
            where="post",
            color="crimson",
            linewidth=2,
            label="Observed fitted-residual CDF",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1.5,
            label="Uniform ideal",
        )

        summary = gof_result["summary"]
        ks_row = summary.loc[
            summary["statistic"] == "calibration_ks"
        ].iloc[0]

        ax.set_title(
            "Parametric-bootstrap time-rescaling GOF\n"
            f"KS discrepancy={ks_row['observed']:.3f}, "
            f"bootstrap p={ks_row['p_upper']:.3f}"
        )
        ax.set_xlabel("Uniform-scale residual $u$")
        ax.set_ylabel("Empirical residual CDF")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.3)

        return ax

    def _intensity_upper_bound(self, dataset, t_idx) -> float:
        raise NotImplementedError

    def estimate_hessian(
        self, 
        dataset: PointProcessDataset, 
        eps: float = 1e-5,
    ) -> np.ndarray:
        if self.params_ is None or self.fit_result is None:
            raise ValueError("Model must be fitted before estimating the Hessian.")

        params = np.array(self.params_, dtype=float)
        k = len(params)
        
        def obj(p):
            return self._nll(p, dataset)

        f0 = self.fit_result.fun
        hessian = np.zeros((k, k))
        h = eps * (1.0 + np.abs(params))

        for i in range(k):
            for j in range(i, k):
                if i == j:
                    p_plus = params.copy(); p_plus[i] += h[i]
                    p_minus = params.copy(); p_minus[i] -= h[i]
                    hessian[i, i] = (obj(p_plus) - 2 * f0 + obj(p_minus)) / (h[i] ** 2)
                else:
                    p_pp = params.copy(); p_pp[i] += h[i]; p_pp[j] += h[j]
                    p_pm = params.copy(); p_pm[i] += h[i]; p_pm[j] -= h[j]
                    p_mp = params.copy(); p_mp[i] -= h[i]; p_mp[j] += h[j]
                    p_mm = params.copy(); p_mm[i] -= h[i]; p_mm[j] -= h[j]
                    
                    val = (obj(p_pp) - obj(p_pm) - obj(p_mp) + obj(p_mm)) / (4 * h[i] * h[j])
                    hessian[i, j] = val
                    hessian[j, i] = val

        return hessian

    def estimate_parameter_correlation(
        self, 
        dataset: PointProcessDataset, 
        eps: float = 1e-5,
    ) -> np.ndarray:
        hessian = self.estimate_hessian(dataset, eps=eps)
        
        try:
            cov = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(hessian)

        std_devs = np.sqrt(np.maximum(0, np.diag(cov)))
        outer_std = np.outer(std_devs, std_devs)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / outer_std
            corr = np.nan_to_num(corr, nan=0.0)
            np.clip(corr, -1.0, 1.0, out=corr)

        return corr

    def stream_compensator_profile(
        self, t_ev: np.ndarray, trial: float, duration_s: float,
    ) -> Tuple[np.ndarray, np.ndarray, bool, float]:
        """
        Polymorphic hook: for ONE (fish, trial) stream, which points count as
        exact calibration residuals under THIS process's own likelihood
        convention, whether the last one is right-censored, and how much
        exposure this trial contributes to a fish's running total (consumed
        by GammaMixedEffectsProcess). Also used by _stream_tau_values/
        time_rescaling below.

        Returns (probe_times, compensator_at_probes, last_is_censored,
        full_trial_exposure).

        DEFAULT = RECURRENT convention (correct as-is for PoissonProcess/
        HawkesProcess/RenewalProcess): every real event in t_ev is an exact
        residual, nothing is censored, and exposure always accrues to
        duration_s regardless of event count -- exactly what these classes'
        own _nll methods already assume. Override only for a terminating
        process (see SurvivalProcess).
        """
        if len(t_ev) == 0:
            full_exposure = self.cumulative_integrated_intensity(
                np.array([duration_s]), trial
            )[0]
            return np.array([]), np.array([]), False, float(full_exposure)

        t_sorted = np.sort(t_ev)
        probes = np.append(t_sorted, duration_s)
        cum = self.cumulative_integrated_intensity(probes, trial)
        return t_sorted, cum[:-1], False, float(cum[-1])

    def _stream_tau_values(
        self, dataset: PointProcessDataset
    ) -> Dict[Tuple[int, int], List[Tuple[float, bool]]]:
        """
        Compute time-rescaled residuals for every (fish, trial) stream, now as
        a list of (residual, is_censored) pairs per stream rather than a bare
        array -- built on stream_compensator_profile, so any override of that
        hook (e.g. SurvivalProcess's terminating-process convention) is picked
        up automatically here with no isinstance checks anywhere.

        Default behavior (recurrent processes: PoissonProcess, HawkesProcess,
        RenewalProcess) is numerically IDENTICAL to before this change -- every
        pair has is_censored=False, since stream_compensator_profile's default
        never censors.

        PRECONDITION UNCHANGED FROM BEFORE: this default assumes
        cumulative_integrated_intensity is already a valid PREDICTABLE
        compensator (see GammaMixedEffectsProcess's own override and its
        docstring for why a marginalized shared random effect breaks that
        assumption and requires a different override).
        """
        result: Dict[Tuple[int, int], List[Tuple[float, bool]]] = {}
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            probes, cum, last_censored, _ = self.stream_compensator_profile(
                t_ev, t_idx, dataset.duration_s
            )
            if len(probes) == 0:
                continue
            diffs = np.diff(np.insert(cum, 0, 0.0))
            pairs = [(float(d), False) for d in diffs]
            if last_censored:
                pairs[-1] = (pairs[-1][0], True)
            result[(f_idx, t_idx)] = pairs
        return result

    @staticmethod
    def _autocorrelation(z_seqs: List[np.ndarray], max_lags: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Lag-1..max_lags autocorrelation of pooled, per-stream z-score
        sequences (Gaussian-transformed EXACT time-rescaled residuals).
        Extracted from time_rescaling's former nested closure so it's
        independently testable/reusable.
        """
        all_z = np.concatenate(z_seqs) if len(z_seqs) > 0 else np.array([])
        n = len(all_z)

        if n < max_lags + 1:
            return (np.array([]), np.array([]), 0.0)

        z_mean = np.mean(all_z)
        z_var = np.var(all_z)
        lags = np.arange(1, max_lags + 1)

        if z_var == 0:
            return (lags, np.zeros(max_lags), 1.96 / np.sqrt(n))

        z_centered_seqs = [seq - z_mean for seq in z_seqs]
        acf = []
        for lag in lags:
            cov_sum = 0.0
            pair_count = 0
            for z_c in z_centered_seqs:
                if len(z_c) > lag:
                    cov_sum += np.sum(z_c[:-lag] * z_c[lag:])
                    pair_count += (len(z_c) - lag)
            r_j = (cov_sum / (pair_count * z_var)) if pair_count > 0 else 0.0
            acf.append(r_j)

        conf_limit = 1.96 / np.sqrt(n)
        return (lags, np.array(acf), conf_limit)

    def _pool_stream_residuals(
        self, dataset: PointProcessDataset
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[Tuple[float, bool]]], List[np.ndarray]]:
        """
        Walks _stream_tau_values(dataset) once and builds every pooled
        representation the calibration/ACF/per-fish checks need:
        - all_vals, all_censored: flat arrays of every (residual, censored) pair
        - fish_pairs: same pairs, grouped by fish (bootstrap bands / D_n)
        - pooled_z_seqs: per-stream sequences of Gaussian-transformed EXACT
            residuals only -- a trailing censored residual is not a real
            inter-event gap, so it's excluded here specifically for ACF.
        """
        tau_by_stream = self._stream_tau_values(dataset)

        all_vals: List[float] = []
        all_censored: List[bool] = []
        fish_pairs: Dict[int, List[Tuple[float, bool]]] = {f: [] for f in range(dataset.num_fish)}
        pooled_z_seqs: List[np.ndarray] = []

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            pairs = tau_by_stream.get((f_idx, t_idx))
            if not pairs:
                continue

            exact_vals = [v for v, c in pairs if not c]
            if exact_vals:
                u = np.clip(1.0 - np.exp(-np.array(exact_vals)), 1e-10, 1 - 1e-10)
                pooled_z_seqs.append(norm.ppf(u))

            for v, c in pairs:
                all_vals.append(v)
                all_censored.append(c)
                fish_pairs[f_idx].append((v, c))

        return np.array(all_vals), np.array(all_censored), fish_pairs, pooled_z_seqs

    def _compute_fish_dn_stats(
        self,
        fish_pairs: Dict[int, List[Tuple[float, bool]]],
        min_events_per_fish: int,
    ) -> np.ndarray:
        """
        Per-fish product-limit sup-distance to the Exp(1) reference -- the
        censoring-aware analog of a per-fish KS D_n statistic. Fish with
        fewer than min_events_per_fish pooled residuals are excluded.
        """
        fish_dn_stats = []
        for f_idx, pairs in fish_pairs.items():
            if len(pairs) < min_events_per_fish:
                continue
            vals = np.array([v for v, _ in pairs])
            cens = np.array([c for _, c in pairs])
            g_grid, g_S = self._survival_estimate(vals, cens)
            fish_dn_stats.append(float(np.max(np.abs(g_S - np.exp(-g_grid)))))
        return np.array(fish_dn_stats)

    def time_rescaling(
            self, 
            dataset: PointProcessDataset,
            acf_lags: int = 20,
            min_events_per_fish: int = 5
        ) -> Dict[str, Any]:
        """
        Ogata time-rescaling / Cox-Snell calibration check, censoring-aware.
        Orchestrates four independently-testable helpers:
        _pool_stream_residuals   -- gather + pool residuals across streams
        _survival_estimate        -- product-limit estimator (shared, static)
        _autocorrelation           -- lag-k ACF of exact residuals
        _compute_fish_dn_stats      -- per-fish calibration effect size

        For any process with no censored residuals (every recurrent process
        family in this module), every quantity below is numerically identical
        to the plain-ECDF/uncensored KS diagnostics this replaced.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted before running time-rescaling analysis.")

        all_vals_arr, all_censored_arr, fish_pairs, pooled_z_seqs = self._pool_stream_residuals(dataset)
        n_rescaled = len(all_vals_arr)

        residual_grid, surv = self._survival_estimate(all_vals_arr, all_censored_arr)
        neg_log_S = -np.log(np.maximum(surv, 1e-12))

        lags, acf, conf_limit = self._autocorrelation(pooled_z_seqs, acf_lags)
        fish_dn_stats = self._compute_fish_dn_stats(fish_pairs, min_events_per_fish)
        n_multi_residual_streams = sum(1 for seq in pooled_z_seqs if len(seq) >= 2)

        return {
            "residuals": all_vals_arr,
            "censored": all_censored_arr,
            "residual_grid": residual_grid,
            "survival_estimate": surv,
            "neg_log_survival": neg_log_S,
            "n_rescaled": n_rescaled,
            "n_exact": int(np.sum(~all_censored_arr)),
            "n_multi_residual_streams": n_multi_residual_streams,
            "fish_pairs": fish_pairs,
            "fish_dn_stats": fish_dn_stats,
            "median_fish_dn": float(np.median(fish_dn_stats)) if len(fish_dn_stats) > 0 else np.nan,
            "mean_fish_dn": float(np.mean(fish_dn_stats)) if len(fish_dn_stats) > 0 else np.nan,
            "acf_lags": lags,
            "acf": acf,
            "acf_conf": conf_limit,
        }

    @staticmethod
    def _survival_estimate(
        times: np.ndarray,
        censored: np.ndarray,
        return_risk: bool = False,
    ):
        """
        Kaplan-Meier estimate for non-negative, possibly right-censored values.

        Parameters
        ----------
        times
            Exact or right-censored residual times.

        censored
            Boolean array, with True indicating right censoring.

        return_risk
            If True, additionally return the number at risk immediately before
            each exact-event knot.

        Returns
        -------
        grid, survival
            KM failure-time knots and corresponding post-jump survival values.
            The arrays begin with the artificial point (0, 1).

        grid, survival, n_at_risk
            Returned when ``return_risk=True``. ``n_at_risk[j]`` is the number
            at risk immediately before the exact events at ``grid[j]``.
        """
        times = np.asarray(times, dtype=float)
        censored = np.asarray(censored, dtype=bool)

        if times.ndim != 1 or censored.ndim != 1:
            raise ValueError("times and censored must be one-dimensional.")
        if len(times) != len(censored):
            raise ValueError("times and censored must have equal length.")
        if np.any(~np.isfinite(times)):
            raise ValueError("Survival times must be finite.")
        if np.any(times < 0):
            raise ValueError("Survival times must be non-negative.")

        if len(times) == 0:
            grid = np.array([0.0], dtype=float)
            survival = np.array([1.0], dtype=float)
            risk = np.array([0], dtype=int)
            return (grid, survival, risk) if return_risk else (grid, survival)

        # Stable sorting gives deterministic tie handling. At a tied time,
        # exact events are treated as occurring before censoring because all
        # observations at that time remain in n_at_risk.
        order = np.argsort(times, kind="stable")
        t_sorted = times[order]
        c_sorted = censored[order]

        n = len(t_sorted)
        grid = [0.0]
        survival = [1.0]
        risk = [n]

        S = 1.0
        i = 0

        while i < n:
            t = t_sorted[i]
            j = i

            n_events = 0
            while j < n and t_sorted[j] == t:
                if not c_sorted[j]:
                    n_events += 1
                j += 1

            n_at_risk = n - i

            if n_events > 0:
                S *= 1.0 - n_events / n_at_risk
                grid.append(float(t))
                survival.append(float(S))
                risk.append(int(n_at_risk))

            # Both failures and censorings at this time leave the risk set after
            # the update.
            i = j

        grid = np.asarray(grid, dtype=float)
        survival = np.asarray(survival, dtype=float)
        risk = np.asarray(risk, dtype=int)

        if return_risk:
            return grid, survival, risk
        return grid, survival

    @staticmethod
    def bootstrap_pooled_survival_band(
        fish_pairs: Dict[int, List[Tuple[float, bool]]],
        n_boot: int = 300,
        grid: np.ndarray = np.linspace(0, 5, 201),
        ci: float = 95.0,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fish-level bootstrap CI band on the KM SURVIVAL curve itself (bounded
        in [0,1]), evaluated on `grid` (a grid of Cox-Snell residual values).
        Deliberately computed in bounded S-space rather than -log(S)-space:
        percentiles of an UNBOUNDED quantity blow up in a sparse tail (a
        handful of bootstrap resamples with slightly different rare large
        residuals produce wildly different -log(S) values), which is exactly
        what makes diagnose()'s Panel B look unstable if plotted that way.
        """
        rng = np.random.default_rng(seed)
        fish_ids = [f for f, pairs in fish_pairs.items() if len(pairs) > 0]
        if not fish_ids:
            return np.full_like(grid, np.nan), np.full_like(grid, np.nan)

        boot_curves = np.empty((n_boot, len(grid)))
        for b in range(n_boot):
            sampled_fish = rng.choice(fish_ids, size=len(fish_ids), replace=True)
            pooled_pairs = [p for f in sampled_fish for p in fish_pairs[f]]
            if not pooled_pairs:
                boot_curves[b, :] = np.nan
                continue
            vals = np.array([v for v, _ in pooled_pairs])
            cens = np.array([c for _, c in pooled_pairs])
            residual_grid, km_S = PointProcess._survival_estimate(vals, cens)
            boot_curves[b, :] = np.interp(grid, residual_grid, km_S, left=1.0, right=km_S[-1])

        alpha = (100.0 - ci) / 2.0
        lower = np.nanpercentile(boot_curves, alpha, axis=0)
        upper = np.nanpercentile(boot_curves, 100 - alpha, axis=0)
        return lower, upper

    def compute_residuals(self, dataset: PointProcessDataset) -> Dict[str, np.ndarray]:
        if self.params_ is None:
            raise ValueError("Model must be fitted before computing residuals.")

        y_obs = dataset.time_trial_histogram_counts
        mu_pred = self.compute_expected_rate(dataset) * dataset.n_fish_per_trial[:, None] * dataset.binning_dt

        r = self.dispersion_r  # inf for Poisson/Hawkes/Renewal; finite for GammaPoissonProcess
        # r is a per-fish quantity; pooling n_fish_per_trial independent fish
        # (same rate, independent gains) sums to an effective NB dispersion of
        # r_eff = r * n_fish_per_trial (sum of iid NB(r, p) is NB(n*r, p)).
        # r_eff -> inf recovers the ordinary Poisson variance/deviance exactly
        # (verified: mu + mu^2/r_eff -> mu; NB deviance -> Poisson deviance).
        if np.isinf(r):
            var_pred = mu_pred
        else:
            r_eff = r * dataset.n_fish_per_trial[:, None]
            var_pred = mu_pred + mu_pred**2 / r_eff

        pearson_res = (y_obs - mu_pred) / np.sqrt(var_pred)

        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y_obs > 0, y_obs * np.log(y_obs / mu_pred), 0.0)

            if np.isinf(r):
                deviance_sq = 2.0 * (term - (y_obs - mu_pred))
            else:
                r_eff = r * dataset.n_fish_per_trial[:, None]
                term_nb = (y_obs + r_eff) * np.log((y_obs + r_eff) / (mu_pred + r_eff))
                deviance_sq = 2.0 * (term - term_nb)

            deviance_res = np.sign(y_obs - mu_pred) * np.sqrt(np.maximum(0.0, deviance_sq))

        return {
            "y_obs": y_obs,
            "mu_pred": mu_pred,
            "var_pred": var_pred,   # new: exposes the variance function actually used
            "pearson_residuals": pearson_res,
            "deviance_residuals": deviance_res,
        }
    
    def _draw_fish_gains(self, num_fish, n_sims, rng):
        """Base default: no frailty, gain always 1."""
        return np.ones((num_fish, n_sims))

    def generate_model_predicted_counts(self, dataset, n_sims=1, rng=None):
        rng = rng or np.random.default_rng()
        fish_gains = self._draw_fish_gains(dataset.num_fish, n_sims, rng)  # shared per fish, per replicate

        predicted_counts = []
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            for sim_i in range(n_sims):
                events = self.simulate_stream(dataset, t_idx, fish_gains[f_idx, sim_i], rng)
                predicted_counts.append(len(events))
        return np.array(predicted_counts)

    def plot_predicted_vs_observed(
        self,
        dataset: PointProcessDataset,
        n_sims: int = 20,
        rng: Optional[np.random.Generator] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Overlays this model's parametric-bootstrap-simulated count
        distribution (see generate_model_predicted_counts) against the
        REAL observed per-(fish,trial) count histogram. Answers: "does data
        GENERATED BY this fitted model look like the data I actually
        collected?" -- the most direct, assumption-free goodness-of-fit
        check available for the model's marginal count structure, valid
        identically across every model class (Poisson, Hawkes, Renewal,
        GammaMixedEffects, ...) with no per-class overrides required.

        NOTE: uses a fixed default rng (see below) so that repeated calls on
        the same fitted model produce a visually stable panel rather than
        fresh sampling noise every time diagnose() is re-run.

        Does NOT, by itself, establish that any fitted heterogeneity
        parameter (e.g. GammaMixedEffectsProcess.dispersion_r) is uniquely
        identified or that its assumed distributional family (Gamma) is
        correct -- a good match here is consistent with, but not proof of,
        either. See GammaMixedEffectsProcess.plot_gain_distribution (Panel J)
        for a more targeted test of the frailty family assumption specifically.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")

        rng = rng or np.random.default_rng(0)  # fixed default seed: stable across repeated diagnose() calls

        observed = dataset.stream_event_counts
        simulated = self.generate_model_predicted_counts(dataset, n_sims=n_sims, rng=rng)

        ax = ax or plt.gca()
        max_k = int(max(observed.max(), simulated.max())) if len(observed) and len(simulated) else 1
        bin_edges = np.arange(-0.5, max_k + 1.5, 1.0)

        ax.hist(observed, bins=bin_edges, density=True, alpha=0.6,
                color='steelblue', edgecolor='none', label='Observed')
        ax.hist(simulated, bins=bin_edges, density=True, alpha=0.8,
                color='green', histtype='step', linewidth=2,
                label=f'Model-simulated (n_sims={n_sims})')

        obs_di = observed.var() / observed.mean() if observed.mean() > 0 else np.nan
        sim_di = simulated.var() / simulated.mean() if simulated.mean() > 0 else np.nan
        ax.set_title(
            f"Observed DI={obs_di:.2f}  |  Simulated DI={sim_di:.2f}",
            fontsize=10,
        )
        ax.set_xlabel("Event count per (fish, trial) stream", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)
        return ax

    def residual_2d_autocorrelation(
        self,
        dataset: PointProcessDataset,
        max_trial_lag: int = 10,
        max_time_lag: int = 30,
    ) -> Dict[str, np.ndarray]:
        """Computes 2D residual autocorrelation across trial and time lag displacements."""
        res_data = self.compute_residuals(dataset)
        residuals = res_data["deviance_residuals"].copy()

        invalid_mask = dataset.n_fish_per_trial == 0
        residuals[invalid_mask, :] = np.nan

        n_trials, n_time = residuals.shape
        residuals -= np.nanmean(residuals)
        variance = np.nanmean(residuals ** 2)

        if not np.isfinite(variance) or variance == 0:
            raise ValueError("Residual variance is zero or undefined.")

        max_trial_lag = min(max_trial_lag, max(0, n_trials - 1))
        max_time_lag = min(max_time_lag, max(0, n_time - 1))

        trial_lags = np.arange(-max_trial_lag, max_trial_lag + 1)
        time_lags_bins = np.arange(-max_time_lag, max_time_lag + 1)
        acf2d = np.full((len(trial_lags), len(time_lags_bins)), np.nan)

        def _get_overlap_slices(n, lag):
            if abs(lag) >= n:
                return slice(0, 0), slice(0, 0)
            if lag >= 0:
                return slice(lag, n), slice(0, n - lag)
            else:
                return slice(0, n + lag), slice(-lag, n)

        for i, dm in enumerate(trial_lags):
            m_x, m_y = _get_overlap_slices(n_trials, dm)
            for j, dt in enumerate(time_lags_bins):
                t_x, t_y = _get_overlap_slices(n_time, dt)

                x = residuals[m_x, t_x]
                y = residuals[m_y, t_y]

                prod = x * y
                n_valid = np.sum(~np.isnan(prod))
                if n_valid > 1:
                    acf2d[i, j] = np.nanmean(prod) / variance

        return {
            "trial_lags": trial_lags,
            "time_lags_bins": time_lags_bins,
            "time_lags_sec": time_lags_bins * dataset.binning_dt,
            "acf2d": acf2d,
            "conf_limit": 1.96 / np.sqrt(n_trials * n_time),  # see note below
        }

    def compute_diagnostics_data(
        self,
        dataset: PointProcessDataset,
        max_trial_lag: int = 10,
        max_time_lag: int = 30,
        eps: float = 1e-5,
        run_parametric_gof: bool = True,
        gof_n_boot: int = 300,
        gof_seed: int = 42,
        gof_ci: float = 95.0,
        gof_acf_lags: int = 20,
        gof_min_km_at_risk: int = 1,
        gof_min_envelope_fraction: float = 0.80,
        gof_refit_n_starts: int = 1,
        gof_n_jobs: int = -1,
        gof_u_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute all dashboard diagnostics once.

        When ``run_parametric_gof=True``, this includes a full
        simulate-refit-recompute parametric bootstrap.
        """
        if self.params_ is None:
            raise ValueError(
                "Model must be fitted before computing diagnostics."
            )

        residuals = self.compute_residuals(dataset)

        time_rescaling = self.time_rescaling(
            dataset,
            acf_lags=gof_acf_lags,
        )

        parameter_correlation = self.estimate_parameter_correlation(
            dataset,
            eps=eps,
        )

        acf2d = self.residual_2d_autocorrelation(
            dataset,
            max_trial_lag=max_trial_lag,
            max_time_lag=max_time_lag,
        )

        if run_parametric_gof:
            parametric_gof = self.parametric_gof_bootstrap(
                dataset=dataset,
                n_boot=gof_n_boot,
                seed=gof_seed,
                ci=gof_ci,
                acf_lags=gof_acf_lags,
                min_km_at_risk=gof_min_km_at_risk,
                min_envelope_fraction=gof_min_envelope_fraction,
                refit_n_starts=gof_refit_n_starts,
                n_jobs=gof_n_jobs,
                u_grid=gof_u_grid,
            )
        else:
            parametric_gof = None

        return {
            "residuals": residuals,
            "time_rescaling": time_rescaling,
            "parameter_correlation": parameter_correlation,
            "acf2d": acf2d,
            "parametric_gof": parametric_gof,
        }


    def plot_panel_residual_surface(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Panel A: 2D deviance residual surface over (trial, time)."""
        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        deviance_res = diag_data["residuals"]["deviance_residuals"]
        vmax = np.percentile(np.abs(deviance_res), 98)
        im = ax.imshow(
            deviance_res, aspect='auto', origin='lower',
            extent=[dataset.t_grid[0], dataset.t_grid[-1], 0, dataset.num_trials],
            cmap='coolwarm', vmin=-vmax, vmax=vmax,
        )
        ax.set_title("A. Deviance Residual Surface $(m, t)$", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time in Trial (s)")
        ax.set_ylabel("Trial Number")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        ax.figure.colorbar(im, cax=cax, label="Deviance Residual")
        return ax

    def plot_panel_calibration_ks(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:

        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        tr_data = diag_data["time_rescaling"]
        n_rescaled = tr_data["n_rescaled"]

        if n_rescaled == 0:
            ax.text(0.5, 0.5, "No events available for calibration check", ha='center', va='center')
            ax.set_title("B. Time-Rescaling Calibration", fontsize=11, fontweight='bold')
            return ax

        r_grid, surv = tr_data["residual_grid"], tr_data["survival_estimate"]
        u_grid = 1.0 - np.exp(-r_grid)
        empirical_cdf = 1.0 - surv
        ax.step(u_grid, empirical_cdf, where='post', color='crimson', lw=2,
                label="Empirical CDF of $U_k$")

        grid_r_boot = np.linspace(0, r_grid[-1], 201)
        lower_S, upper_S = self.bootstrap_pooled_survival_band(tr_data["fish_pairs"], grid=grid_r_boot)
        u_boot = 1.0 - np.exp(-grid_r_boot)
        ax.fill_between(u_boot, 1 - upper_S, 1 - lower_S, color="crimson", alpha=0.15,
                        label="95% band (fish-level bootstrap)", zorder=1)

        ax.plot([0, 1], [0, 1], 'k--', label="Uniform(0,1) Ideal", lw=1.5)
        ks_bound = 1.36 / np.sqrt(n_rescaled)
        ax.plot([0, 1], [ks_bound, 1 + ks_bound], 'k:', alpha=0.5, label="95% KS Limits")
        ax.plot([0, 1], [-ks_bound, 1 - ks_bound], 'k:', alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_title("B. Time-Rescaling: Rescaled-Interval CDF vs. Uniform(0,1)",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel("Transformed Interval ($U_k$)")
        ax.set_ylabel("Cumulative Probability")
        ax.legend(loc="upper left", fontsize=8)
        return ax


    def plot_panel_residual_histogram(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Panel C: deviance residual distribution vs N(0,1)."""
        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        flat_dev = diag_data["residuals"]["deviance_residuals"].flatten()
        ax.hist(flat_dev, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="none")
        x_norm = np.linspace(-4, 4, 200)
        ax.plot(x_norm, norm.pdf(x_norm), 'r--', lw=2, label=r"$\mathcal{N}(0, 1)$ Ref")
        ax.set_title("C. Deviance Residual Distribution", fontsize=11, fontweight='bold')
        ax.set_xlabel("Deviance Residual Value")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8)
        return ax

    def plot_panel_event_lag_acf(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Panel D: autocorrelation of time-rescaled EXACT residuals across event
        lag. Meaningless (and will naturally render empty/near-zero) for any
        stream capped at <=1 event -- e.g. SurvivalProcess -- since there's no
        sequence of intervals within a stream to autocorrelate; no special
        casing needed, this falls out automatically because time_rescaling
        only feeds exact-residual sequences into the ACF pool.
        """
        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        tr_data = diag_data["time_rescaling"]
        ax.vlines(tr_data["acf_lags"], 0, tr_data["acf"], color="navy", lw=2)
        ax.axhline(0, color="black", lw=1)
        conf_limit = tr_data["acf_conf"]
        ax.axhline(conf_limit, color="red", linestyle="--", alpha=0.7, label="95% CI")
        ax.axhline(-conf_limit, color="red", linestyle="--", alpha=0.7)
        ax.set_title("D. Time-rescaled event autocorrelation (Event Lag)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Lag (event)")
        ax.set_ylabel("Autocorrelation")
        ax.set_ylim([-0.5, 0.5])
        ax.legend(loc="upper right", fontsize=8)
        return ax

    def plot_parametric_gof_cox_snell(
        self,
        gof_result: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        show_pointwise_band: bool = False,
    ) -> plt.Axes:
        """
        Parametric-bootstrap Cox-Snell residual calibration plot.

        The observed curve is:

            -log(S_hat_KM(r)) versus r,

        where r is the Cox-Snell residual. Under correct calibration, the curve
        follows the identity line.

        The bootstrap curves were constructed on the equivalent Uniform-CDF
        scale and are transformed back here. Unsupported tails remain NaN and
        are therefore not drawn.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        observed = gof_result["observed"]

        if not observed["has_censoring"]:
            raise ValueError(
                "plot_parametric_gof_cox_snell is intended for "
                "censored residuals."
            )

        r_grid = np.asarray(
            gof_result["r_grid"],
            dtype=float,
        )

        def cdf_to_cox_snell(cdf: np.ndarray) -> np.ndarray:
            cdf = np.asarray(cdf, dtype=float)
            survival = 1.0 - cdf

            result = np.full_like(
                survival,
                np.nan,
                dtype=float,
            )

            # Survival equal to zero implies an infinite ordinate and should not
            # be drawn as a finite point.
            valid = (
                np.isfinite(survival)
                & (survival > 0.0)
                & (survival <= 1.0)
            )

            result[valid] = -np.log(survival[valid])
            return result

        # ------------------------------------------------------------------
        # Bootstrap envelopes
        # ------------------------------------------------------------------

        if show_pointwise_band:
            pointwise_lower = cdf_to_cox_snell(
                gof_result["bootstrap_cdf_lower"]
            )
            pointwise_upper = cdf_to_cox_snell(
                gof_result["bootstrap_cdf_upper"]
            )

            valid_pointwise = (
                np.isfinite(r_grid)
                & np.isfinite(pointwise_lower)
                & np.isfinite(pointwise_upper)
            )

            ax.fill_between(
                r_grid[valid_pointwise],
                pointwise_lower[valid_pointwise],
                pointwise_upper[valid_pointwise],
                color="steelblue",
                alpha=0.16,
                label="Pointwise parametric-bootstrap band",
                zorder=1,
            )

        simultaneous_lower = cdf_to_cox_snell(
            gof_result["simultaneous_cdf_lower"]
        )
        simultaneous_upper = cdf_to_cox_snell(
            gof_result["simultaneous_cdf_upper"]
        )

        valid_simultaneous = (
            np.isfinite(r_grid)
            & np.isfinite(simultaneous_lower)
            & np.isfinite(simultaneous_upper)
        )

        if np.any(valid_simultaneous):
            ax.fill_between(
                r_grid[valid_simultaneous],
                simultaneous_lower[valid_simultaneous],
                simultaneous_upper[valid_simultaneous],
                color="gray",
                alpha=0.25,
                label="Simultaneous parametric-bootstrap envelope",
                zorder=2,
            )

        bootstrap_median = cdf_to_cox_snell(
            gof_result["bootstrap_cdf_median"]
        )

        valid_median = (
            np.isfinite(r_grid)
            & np.isfinite(bootstrap_median)
        )

        if np.any(valid_median):
            ax.plot(
                r_grid[valid_median],
                bootstrap_median[valid_median],
                color="steelblue",
                linestyle=":",
                linewidth=1.5,
                label="Bootstrap median fitted-residual curve",
                zorder=3,
            )

        # ------------------------------------------------------------------
        # Exact observed KM curve
        # ------------------------------------------------------------------

        observed_r = np.asarray(
            observed["km_residual_grid"],
            dtype=float,
        )
        observed_survival = np.asarray(
            observed["km_survival"],
            dtype=float,
        )
        observed_risk = np.asarray(
            observed["km_n_at_risk"],
            dtype=int,
        )

        observed_r_limit = float(observed["r_limit"])
        min_km_at_risk = int(observed["min_km_at_risk"])

        valid_observed = (
            np.isfinite(observed_r)
            & np.isfinite(observed_survival)
            & (observed_survival > 0.0)
            & (observed_r <= observed_r_limit)
        )

        # The origin should always be retained.
        if len(valid_observed) > 0:
            valid_observed[0] = True

        # This is redundant with r_limit but makes the plotting rule explicit.
        if len(valid_observed) > 1:
            valid_observed[1:] &= (
                observed_risk[1:] >= min_km_at_risk
            )

        observed_neg_log_survival = -np.log(
            observed_survival[valid_observed]
        )

        ax.step(
            observed_r[valid_observed],
            observed_neg_log_survival,
            where="post",
            color="crimson",
            linewidth=2.0,
            label=r"Observed $-\log\widehat S_{\mathrm{KM}}(r)$",
            zorder=4,
        )

        # ------------------------------------------------------------------
        # Ideal line and limits
        # ------------------------------------------------------------------

        finite_bootstrap_r = r_grid[
            valid_simultaneous | valid_median
        ]

        candidate_xmax = [observed_r_limit]

        if len(finite_bootstrap_r) > 0:
            candidate_xmax.append(
                float(np.max(finite_bootstrap_r))
            )

        x_max = max(
            max(candidate_xmax),
            1e-6,
        )

        finite_y = []

        if len(observed_neg_log_survival) > 0:
            finite_y.append(
                float(np.max(observed_neg_log_survival))
            )
        if np.any(valid_median):
            finite_y.append(
                float(np.max(bootstrap_median[valid_median]))
            )
        if np.any(valid_simultaneous):
            finite_y.append(
                float(
                    np.max(
                        simultaneous_upper[valid_simultaneous]
                    )
                )
            )

        y_max = max(
            finite_y + [x_max, 1e-6]
        )

        diagonal_max = max(x_max, y_max)

        ax.plot(
            [0.0, diagonal_max],
            [0.0, diagonal_max],
            "k--",
            linewidth=1.5,
            label="Exp(1) ideal",
            zorder=3,
        )

        # ------------------------------------------------------------------
        # Title
        # ------------------------------------------------------------------

        summary = gof_result["summary"]
        ks_rows = summary.loc[
            summary["statistic"] == "calibration_ks"
        ]

        if len(ks_rows) > 0:
            ks_row = ks_rows.iloc[0]
            ks_value = ks_row["observed"]
            p_value = ks_row["p_upper"]

            if np.isfinite(ks_value) and np.isfinite(p_value):
                statistic_text = (
                    f"KS={ks_value:.3f}, bootstrap p={p_value:.3f}"
                )
            elif np.isfinite(ks_value):
                statistic_text = f"KS={ks_value:.3f}"
            else:
                statistic_text = "calibration statistic unavailable"
        else:
            statistic_text = "calibration statistic unavailable"

        ax.set_title(
            "B. Parametric-bootstrap Cox–Snell calibration\n"
            f"exact={observed['n_exact']}, "
            f"censored={observed['n_censored']}; "
            f"{statistic_text}",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlabel("Cox–Snell residual $r$")
        ax.set_ylabel(r"$-\log\widehat S_{\mathrm{KM}}(r)$")
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, y_max * 1.05)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

        return ax

    def plot_panel_calibration_cox_snell(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Survival-analysis-native Cox-Snell residual plot: -log(S_hat(r)) vs r,
        checked against the Exp(1) diagonal, with N total/exact/censored
        annotated in the title (the risk-composition context a survival-
        trained reader expects, per Klein & Moeschberger). Caller (diagnose())
        is responsible for choosing this over plot_panel_calibration_ks based
        on whether any residual is censored.
        """
        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        tr_data = diag_data["time_rescaling"]
        n_rescaled = tr_data["n_rescaled"]

        if n_rescaled == 0:
            ax.text(0.5, 0.5, "No events available for calibration check", ha='center', va='center')
            ax.set_title("B. Cox-Snell Residual Plot", fontsize=11, fontweight='bold')
            return ax

        r_grid, surv = tr_data["residual_grid"], tr_data["survival_estimate"]
        neg_log_S = -np.log(np.maximum(surv, 1e-12))
        max_r = r_grid[-1]
        ax.step(r_grid, neg_log_S, where='post', color='navy', lw=2, label=r"$-\log \hat{S}(r)$")

        grid_r_boot = np.linspace(0, max_r, 201)
        lower_S, upper_S = self.bootstrap_pooled_survival_band(tr_data["fish_pairs"], grid=grid_r_boot)
        # Band legitimately widens toward the sparse tail -- expected feature
        # of Cox-Snell plots, not an artifact to hide (unlike plotting
        # -log(S) directly from raw bootstrap percentiles, which is unstable;
        # this transforms bounded S-space percentiles instead).
        neg_log_lower = -np.log(np.maximum(upper_S, 1e-12))
        neg_log_upper = -np.log(np.maximum(lower_S, 1e-12))
        ax.fill_between(grid_r_boot, neg_log_lower, neg_log_upper, color='navy', alpha=0.15,
                        label="95% band (fish-level bootstrap)")

        ax.plot([0, max_r], [0, max_r], 'k--', lw=1.5, label="Ideal (Exp(1))")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        n_censored = int(np.sum(tr_data["censored"]))
        ax.set_title(f"B. Cox-Snell Residual Plot (N={n_rescaled}, "
                    f"exact={tr_data['n_exact']}, censored={n_censored})",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel("Cox-Snell residual $r$")
        ax.set_ylabel(r"$-\log \hat{S}(r)$")
        ax.legend(loc="upper left", fontsize=8)
        return ax

    def plot_panel_survival_curve(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Empirical first-event survival curve (product-limit estimator, built
        directly from dataset.iter_streams() -- needs no model), overlaid with
        this model's own population_survival_curve() IF it defines one.
        Checked via hasattr, not isinstance: this lets SurvivalProcess and
        GammaMixedEffectsProcess(SurvivalProcess(...)) contribute a model
        overlay without this base-class method (or diagnose(), which decides
        to call this panel at all) referencing either class by name. Any
        process that doesn't define population_survival_curve (Poisson/
        Hawkes/Renewal) still gets the empirical curve alone -- a reasonable
        fallback, though diagnose() should not normally route here for those
        (see n_multi_residual_streams).
        """
        if ax is None:
            _, ax = plt.subplots()

        times, censored = [], []
        for _, _, t_ev in dataset.iter_streams():
            if len(t_ev) == 0:
                times.append(dataset.duration_s); censored.append(True)
            else:
                times.append(float(np.min(t_ev))); censored.append(False)
        times, censored = np.array(times), np.array(censored)
        grid, surv = self._survival_estimate(times, censored)
        ax.step(grid, surv, where='post', color='black', linewidth=2, label=r'Empirical $\hat{S}(t)$')

        if hasattr(self, "population_survival_curve"):
            model_t, model_S = self.population_survival_curve(dataset)
            ax.plot(model_t, model_S, color='crimson', linestyle='--', linewidth=2, label='Model $S(t)$')

        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Time in trial (s)")
        ax.set_ylabel("P(no first event by t)")
        ax.set_title("D. First-Event Survival Curve: Empirical vs Model", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)
        return ax

    def plot_panel_residual_acf2d(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Panel E: 2D residual autocorrelation surface R(delta_m, delta_t)."""
        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        acf2d_data = diag_data["acf2d"]
        acf2d = acf2d_data["acf2d"]
        trial_lags = acf2d_data["trial_lags"]
        time_lags_sec = acf2d_data["time_lags_sec"]
        conf_lim_2d = acf2d_data["conf_limit"]

        m_zero_idx = np.where(trial_lags == 0)[0][0]
        t_zero_idx = np.where(acf2d_data["time_lags_bins"] == 0)[0][0]
        acf_offdiag = acf2d.copy()
        acf_offdiag[m_zero_idx, t_zero_idx] = np.nan
        vmax_2d = max(0.05, np.nanmax(np.abs(acf_offdiag)))

        dt = dataset.binning_dt
        extent_2d = [
            time_lags_sec[0] - dt / 2, time_lags_sec[-1] + dt / 2,
            trial_lags[0] - 0.5, trial_lags[-1] + 0.5
        ]
        im_2d = ax.imshow(acf2d, extent=extent_2d, origin="lower", cmap="coolwarm",
                        vmin=-vmax_2d, vmax=vmax_2d, aspect="auto")

        T_mesh, M_mesh = np.meshgrid(time_lags_sec, trial_lags)
        contours = ax.contour(T_mesh, M_mesh, np.abs(acf2d), levels=[conf_lim_2d],
                            colors="black", linewidths=1.0, linestyles="--")
        ax.clabel(contours, fmt={conf_lim_2d: "95% CI"}, inline=True, fontsize=7)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_title("E. Autocorrelation of deviance residuals $R(\\Delta m, \\Delta t)$",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel("Time Lag $\\Delta t$ (s)")
        ax.set_ylabel("Trial Lag $\\Delta m$ (trials)")
        divider_2d = make_axes_locatable(ax)
        cax_2d = divider_2d.append_axes("right", size="3%", pad=0.08)
        ax.figure.colorbar(im_2d, cax=cax_2d, label="Autocorrelation")
        return ax

    def plot_panel_fish_dn_distribution(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Panel F: per-fish calibration effect-size (KM/product-limit sup-distance to Exp(1))."""
        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        tr_data = diag_data["time_rescaling"]
        dn_values = tr_data["fish_dn_stats"]
        median_dn = tr_data["median_fish_dn"]

        if len(dn_values) > 0:
            ax.hist(dn_values, bins='auto', density=True, alpha=0.6,
                    color='skyblue', edgecolor='navy', label='Per-Fish $D_n$')
            ax.axvline(median_dn, color='darkblue', linestyle='--', linewidth=2,
                    label=f'Median $D_n$ ({median_dn:.3f})')
            ax.axvspan(0.0, 0.05, color='green', alpha=0.1, label='Good Fit ($D_n < 0.05$)')
            ax.set_title(f"F. Per-Fish $D_n$ Distribution ($N_{{fish}}={len(dn_values)}$)",
                        fontsize=11, fontweight='bold')
            ax.set_xlabel("KS Distance ($D_n$)")
            ax.set_ylabel("Density")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Insufficient events per fish for $D_n$", ha='center', va='center')
        return ax

    def plot_panel_parameter_correlation(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Panel G: parameter correlation matrix."""
        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        corr_matrix = diag_data["parameter_correlation"]
        im_corr = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        ax.set_title("G. Parameter Correlation Matrix", fontsize=11, fontweight='bold')
        n_params = len(self.param_names)
        ax.set_xticks(np.arange(n_params)); ax.set_yticks(np.arange(n_params))
        ax.set_xticklabels(self.param_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(self.param_names, fontsize=9)
        for i in range(n_params):
            for j in range(n_params):
                val = corr_matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color="white" if abs(val) > 0.6 else "black", fontsize=8)
        divider_corr = make_axes_locatable(ax)
        cax_corr = divider_corr.append_axes("right", size="3%", pad=0.08)
        ax.figure.colorbar(im_corr, cax=cax_corr, label="Correlation")
        return ax

    def plot_panel_summary_text(
        self, dataset: PointProcessDataset, diag_data: Optional[Dict[str, Any]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Panel H: global diagnostic summary metrics."""
        if diag_data is None:
            diag_data = self.compute_diagnostics_data(dataset,run_parametric_gof=False)
        if ax is None:
            _, ax = plt.subplots()

        tr_data = diag_data["time_rescaling"]
        acf2d_data = diag_data["acf2d"]
        m_zero_idx = np.where(acf2d_data["trial_lags"] == 0)[0][0]
        t_zero_idx = np.where(acf2d_data["time_lags_bins"] == 0)[0][0]
        acf_offdiag = acf2d_data["acf2d"].copy()
        acf_offdiag[m_zero_idx, t_zero_idx] = np.nan

        ax.axis('off')
        summary_text = (
            f"DIAGNOSTIC SUMMARY METRICS\n"
            f"----------------------------------------\n"
            f"Log-Likelihood      : {self.log_likelihood:.2f}\n"
            f"Akaike Info (AIC)   : {self.aic:.2f}\n"
            f"Pooled Residuals    : N = {tr_data['n_rescaled']} "
            f"(exact={tr_data['n_exact']})\n"
            f"Median Per-Fish D_n : {tr_data['median_fish_dn']:.4f}\n"
            f"Max 2D Autocorr     : {np.nanmax(np.abs(acf_offdiag)):.4f}\n"
            f"95% 2D CI Limit     : ±{acf2d_data['conf_limit']:.4f}\n"
        )
        ax.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.3))
        ax.set_title("H. Global Model Diagnostics", fontsize=11, fontweight='bold')
        return ax

    def diagnose(
        self,
        dataset: PointProcessDataset,
        figsize: Tuple[int, int] = (15, 22),
        eps: float = 1e-5,
        max_trial_lag: int = 10,
        max_time_lag: int = 30,
        run_parametric_gof: bool = True,
        gof_n_boot: int = 300,
        gof_seed: int = 42,
        gof_ci: float = 95.0,
        gof_acf_lags: int = 20,
        gof_min_km_at_risk: int = 1,
        gof_min_envelope_fraction: float = 0.80,
        gof_refit_n_starts: int = 1,
        gof_n_jobs: int = -1,
        gof_u_grid: Optional[np.ndarray] = None,
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Assemble the diagnostic dashboard.

        Panel B uses:
        - bootstrap-calibrated Cox-Snell KM diagnostics for censored residuals;
        - bootstrap-calibrated Uniform-CDF diagnostics for uncensored residuals;
        - the corresponding descriptive fallback when the parametric bootstrap
            is disabled.
        """
        diag_data = self.compute_diagnostics_data(
            dataset=dataset,
            max_trial_lag=max_trial_lag,
            max_time_lag=max_time_lag,
            eps=eps,
            run_parametric_gof=run_parametric_gof,
            gof_n_boot=gof_n_boot,
            gof_seed=gof_seed,
            gof_ci=gof_ci,
            gof_acf_lags=gof_acf_lags,
            gof_min_km_at_risk=gof_min_km_at_risk,
            gof_min_envelope_fraction=gof_min_envelope_fraction,
            gof_refit_n_starts=gof_refit_n_starts,
            gof_n_jobs=gof_n_jobs,
            gof_u_grid=gof_u_grid,
        )

        tr_data = diag_data["time_rescaling"]
        parametric_gof = diag_data["parametric_gof"]

        fig, axes = plt.subplots(
            5,
            2,
            figsize=figsize,
        )
        plt.subplots_adjust(
            hspace=0.38,
            wspace=0.3,
        )

        fig.suptitle(
            self.latex_formula,
            fontsize=15,
            fontweight="bold",
            y=0.99,
        )

        # A
        self.plot_panel_residual_surface(
            dataset,
            diag_data,
            ax=axes[0, 0],
        )

        # B
        if parametric_gof is not None:
            if parametric_gof["observed"]["has_censoring"]:
                self.plot_parametric_gof_cox_snell(
                    gof_result=parametric_gof,
                    ax=axes[0, 1],
                )
            else:
                self.plot_parametric_gof_calibration(
                    gof_result=parametric_gof,
                    ax=axes[0, 1],
                )
        elif np.any(tr_data["censored"]):
            self.plot_panel_calibration_cox_snell(
                dataset,
                diag_data,
                ax=axes[0, 1],
            )
        else:
            self.plot_panel_calibration_ks(
                dataset,
                diag_data,
                ax=axes[0, 1],
            )

        # C
        self.plot_panel_residual_histogram(
            dataset,
            diag_data,
            ax=axes[1, 0],
        )

        # D
        if tr_data["n_multi_residual_streams"] > 0:
            self.plot_panel_event_lag_acf(
                dataset,
                diag_data,
                ax=axes[1, 1],
            )
        else:
            self.plot_panel_survival_curve(
                dataset,
                diag_data,
                ax=axes[1, 1],
            )

        # E
        self.plot_panel_residual_acf2d(
            dataset,
            diag_data,
            ax=axes[2, 0],
        )

        # F
        self.plot_panel_fish_dn_distribution(
            dataset,
            diag_data,
            ax=axes[2, 1],
        )

        # G
        self.plot_panel_parameter_correlation(
            dataset,
            diag_data,
            ax=axes[3, 0],
        )

        # H
        self.plot_panel_summary_text(
            dataset,
            diag_data,
            ax=axes[3, 1],
        )

        self.plot_predicted_vs_observed(
            dataset,
            ax=axes[4, 0],
        )

        axes[4, 1].axis("off")

        return fig, diag_data

    @property
    def dispersion_r(self) -> float:
        return np.inf

    @property
    def overdispersion_index(self) -> float:
        r = self.dispersion_r
        return 0.0 if np.isinf(r) else 1.0 / r

    @property
    def is_survival(self) -> bool:
        return False


def _fit_one_model(
    model: PointProcess,
    dataset: PointProcessDataset,
    method: str,
    kwargs: dict,
) -> Tuple[PointProcess, Optional[str]]:
    try:
        model.fit_multistart(dataset, method=method, **kwargs)
        return model, None
    except RuntimeError as e:
        return model, str(e)
    
class ModelComparator:

    @staticmethod
    def likelihood_ratio_test(
        model_null: PointProcess, 
        model_alt: PointProcess
    ) -> Dict[str, Union[str, int, float, bool]]:
        """Assumes models are nested"""
    
        k_null = len(model_null.params_)
        k_alt = len(model_alt.params_)
        df = k_alt - k_null

        ll_null = model_null.log_likelihood
        ll_alt = model_alt.log_likelihood
        lr_stat = 2.0 * (ll_alt - ll_null)
        lr_stat_clamped = max(0.0, lr_stat)
        
        p_val = float(chi2.sf(lr_stat_clamped, df))

        return {
            "Null Model": model_null.name,
            "Alt Model": model_alt.name,
            "LL Null": ll_null,
            "LL Alt": ll_alt,
            "Deviance (2*ΔLL)": lr_stat,
            "Δk (df)": df,
            "p-value": p_val,
            "Significant (α=0.05)": p_val < 0.05
        }

    @staticmethod
    def compare(
        models: List[PointProcess],
        dataset: PointProcessDataset,
        method: str = "L-BFGS-B",
        null_model: Optional[PointProcess] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[PointProcess]]:

        if null_model is not None:
            null_fitted, null_error = _fit_one_model(null_model, dataset, method, kwargs)
            if null_error is not None:
                print(f"[ModelComparator] WARNING: null model fit failed: {null_error}")
                ll_null = np.nan
            else:
                ll_null = null_fitted.log_likelihood
        else:
            ll_null = np.nan

        results = [_fit_one_model(m, dataset, method, kwargs) for m in models]
        fitted_models: List[PointProcess] = []
        records = []

        for model, error in results:
            if error is None:
                fitted_models.append(model)
                mcfadden_r2 = (
                    1.0 - model.log_likelihood / ll_null
                    if not np.isnan(ll_null) else np.nan
                )
                records.append({
                    "Model Name": model.name,
                    "Params (k)": len(model.param_names),
                    "Log-Likelihood": model.log_likelihood,
                    "AIC": model.aic,
                    "McFadden R2": mcfadden_r2,
                    "r_dispersion": model.dispersion_r,
                    "Overdispersion (1/r)": model.overdispersion_index,
                    "Converged": True,
                })
            else:
                print(f"[ModelComparator] WARNING: fit failed for {model.name}: {error}")
                records.append({
                    "Model Name": model.name,
                    "Params (k)": len(model.param_names),
                    "Log-Likelihood": np.nan,
                    "AIC": np.nan,
                    "McFadden R2": np.nan,
                    "r_dispersion": np.nan,
                    "Overdispersion (1/r)": np.nan,
                    "Converged": False,
                })

        df = pd.DataFrame(records)

        if df["AIC"].notna().any():
            min_aic = df["AIC"].min()
            df["ΔAIC"] = df["AIC"] - min_aic
            weights = np.exp(-0.5 * df["ΔAIC"])
            df["AIC Weight"] = weights / np.nansum(weights)
        else:
            df["ΔAIC"] = np.nan
            df["AIC Weight"] = np.nan

        sort_idx = df["AIC"].argsort(kind="stable").values
        df = df.iloc[sort_idx].reset_index(drop=True)
        fitted_models.sort(key=lambda m: m.aic)

        return df, fitted_models
    
class ModelPlotter:

    @staticmethod
    def _weighted_marginals(surface: np.ndarray, dataset: PointProcessDataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        time_marginal: rate(t), n_fish-weighted average across trials --
            same convention as dataset.time_histogram_hz, so empirical and
            model surfaces are reduced identically.
        trial_marginal: rate(trial), simple mean across time bins. Trials
            with n_fish_per_trial == 0 are NaN (not 0) -- distinguishes
            "no data" from "zero rate", matching
            plot_time_trial_rate_heatmap's convention.
        """
        n_fish = dataset.n_fish_per_trial
        active = n_fish > 0

        time_marginal = np.full(surface.shape[1], np.nan)
        if active.any():
            time_marginal = np.average(surface[active], axis=0, weights=n_fish[active])

        trial_marginal = np.full(surface.shape[0], np.nan)
        trial_marginal[active] = np.mean(surface[active], axis=1)

        return time_marginal, trial_marginal
    
    @staticmethod
    def plot_histogram(
        dataset: PointProcessDataset,
        model: PointProcess,
        figsize: Tuple[float, float] = (22, 8),
        cmap: str = "plasma",
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Empirical vs. model rate surface, each with overlaid empirical
        (solid) / model (dashed) marginals: one time-marginal strip
        spanning both heatmap columns, one shared trial-marginal column to
        the right. Same layout convention as the ablation_screen arm-
        surface-grid plots -- single shared GridSpec (not per-panel
        independent ones) so heatmaps and marginals stay pixel-aligned,
        and an explicit colorbar axis (not make_axes_locatable) so the
        colorbar never overlaps a title.
        """
        surf_emp = dataset.time_trial_histogram_hz
        surf_model = model.compute_expected_rate(dataset)

        tm_emp, trm_emp = ModelPlotter._weighted_marginals(surf_emp, dataset)
        tm_model, trm_model = ModelPlotter._weighted_marginals(surf_model, dataset)

        vmax = max(np.nanmax(surf_emp), np.nanmax(surf_model))
        vmax = max(vmax, 1e-9)
        time_ymax = max(np.nanmax(x) for x in [tm_emp, tm_model] if np.any(np.isfinite(x)))
        trial_xmax = max(np.nanmax(x) for x in [trm_emp, trm_model] if np.any(np.isfinite(x)))
        time_ymax, trial_xmax = max(time_ymax, 1e-9), max(trial_xmax, 1e-9)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            nrows=2, ncols=3,
            width_ratios=[4, 4, 1.3],
            height_ratios=[1, 4],
            wspace=0.12, hspace=0.12,
            left=0.07, right=0.86, top=0.90, bottom=0.09,
        )

        ax_time = fig.add_subplot(gs[0, 0:2])
        ax_emp = fig.add_subplot(gs[1, 0], sharex=ax_time)
        ax_mod = fig.add_subplot(gs[1, 1], sharex=ax_time, sharey=ax_emp)
        ax_trial = fig.add_subplot(gs[1, 2], sharey=ax_emp)

        ax_time.plot(dataset.t_centers, tm_emp, color="black", linestyle="-", linewidth=1.3, label="Empirical")
        ax_time.plot(dataset.t_centers, tm_model, color="crimson", linestyle="--", linewidth=1.3, label="Model")
        ax_time.set_ylim(0, time_ymax * 1.05)
        plt.setp(ax_time.get_xticklabels(), visible=False)
        ax_time.tick_params(labelsize=7)
        ax_time.legend(loc="upper right", fontsize=8, frameon=False)

        mesh = ax_emp.pcolormesh(dataset.t_grid, dataset.trial_edges, surf_emp,
                                  shading="flat", cmap=cmap, vmin=0.0, vmax=vmax)
        ax_mod.pcolormesh(dataset.t_grid, dataset.trial_edges, surf_model,
                           shading="flat", cmap=cmap, vmin=0.0, vmax=vmax)
        ax_emp.set_title("Empirical Surface", fontsize=11, fontweight="bold")
        ax_mod.set_title(f"Fitted Surface: {model.name}", fontsize=11, fontweight="bold")
        ax_emp.set_xlabel("Time in Trial (s)", fontsize=10)
        ax_mod.set_xlabel("Time in Trial (s)", fontsize=10)
        ax_emp.set_ylabel("Trial Number", fontsize=10)
        plt.setp(ax_mod.get_yticklabels(), visible=False)
        ax_emp.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])
        ax_emp.tick_params(labelsize=8)
        ax_mod.tick_params(labelsize=8)

        trial_centers = np.arange(dataset.num_trials)
        ax_trial.plot(trm_emp, trial_centers, color="black", linestyle="-", linewidth=1.3)
        ax_trial.plot(trm_model, trial_centers, color="crimson", linestyle="--", linewidth=1.3)
        ax_trial.set_xlim(0, trial_xmax * 1.05)
        plt.setp(ax_trial.get_yticklabels(), visible=False)
        ax_trial.tick_params(labelsize=7)

        cax = fig.add_axes([0.89, 0.15, 0.02, 0.65])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Event Rate [Hz]", fontsize=10)

        return fig, np.array([ax_time, ax_emp, ax_mod, ax_trial])

    @staticmethod
    def plot_model_fits(
        dataset: PointProcessDataset,
        models: List[PointProcess],
        figsize: Tuple[int, int] = (12, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Empirical PSTH
        ax.plot(
            dataset.t_centers, 
            dataset.time_histogram_hz, 
            color='black', 
            linewidth=2.0, 
            label='Empirical PSTH',
            zorder=4  # Kept on top for visibility
        )

        # 2. Evaluate Models & Build Labels with LaTeX Formulas
        default_colors = plt.cm.tab10.colors

        for idx, model in enumerate(models):
            pred_surface = model.compute_expected_rate(dataset)
            mean_pred = np.average(pred_surface, axis=0, weights=dataset.n_fish_per_trial)
            color = default_colors[idx % len(default_colors)]
            label = f"{model.name}:  {model.latex_formula}"
            ax.plot(
                dataset.t_centers, 
                mean_pred, 
                linestyle='--', 
                linewidth=1.8, 
                color=color, 
                label=label,
                zorder=3
            )

        # 3. Plot Formatting
        ax.set_title(
            f"Bout: {dataset.bout_name} (Laterality: {dataset.laterality})", 
            fontsize=12, 
            fontweight='bold'
        )
        ax.set_xlabel("Time in Trial (s)", fontsize=11)
        ax.set_ylabel("Rate [Hz]", fontsize=11)
        ax.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle=':', alpha=0.6)

        # 4. Legend below the axes
        ax.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.18), 
            ncol=1,                     # 1 column so long formulas don't overlap horizontally
            frameon=True, 
            facecolor='white', 
            framealpha=0.9,
            fontsize=10
        )

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_trial_traces(
        dataset: PointProcessDataset,
        model: PointProcess,
        trial_step: int = 2,
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = "viridis",
        marker_size: float = 28.0,
        data_alpha: float = 0.35,
        model_alpha: float = 1.0
    ) -> Tuple[plt.Figure, plt.Axes]:

        fig, ax = plt.subplots(figsize=figsize)

        model_surface = model.compute_expected_rate(dataset)

        # 2. Color Mapping Setup
        norm = plt.Normalize(vmin=0, vmax=dataset.num_trials)
        base_cmap = plt.get_cmap(cmap)

        # 3. Plot Selected Trials
        selected_indices = range(0, dataset.num_trials, trial_step)

        for idx in selected_indices:
            color = base_cmap(norm(idx))

            # Empirical Data (Markers Only, Semi-Transparent, Plotted Underneath)
            ax.scatter(
                dataset.t_centers,
                dataset.time_trial_histogram_hz[idx, :],
                color=color,
                s=marker_size,
                alpha=data_alpha,
                linewidths=0,
                zorder=2
            )

            # Model Fits (Continuous Dashed Line, Plotted On Top)
            ax.plot(
                dataset.t_centers,
                model_surface[idx, :],
                color=color,
                linestyle='--',
                linewidth=1.8,
                alpha=model_alpha,
                zorder=3
            )

        # 4. Legend to clarify Data vs. Model styling
        ax.scatter([], [], color='gray', s=marker_size, alpha=0.5, label='Empirical Data (Points)')
        ax.plot([], [], color='gray', linestyle='--', linewidth=1.8, label='Model Fit (Dashed Line)')

        # Plot Formatting
        ax.set_title(
            f"Trial-by-Trial Overlay | {model.name} (Every {trial_step} Trials)", 
            fontsize=12, 
            fontweight='bold'
        )
        ax.set_xlabel("Time in Trial (s)", fontsize=11)
        ax.set_ylabel("Rate [Hz]", fontsize=11)
        ax.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

        # 5. Colorbar for Trial Progression
        sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
        sm.set_array([])
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.12)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Trial Index", fontsize=10)

        plt.tight_layout()
        return fig, ax