"""Fitting, uncertainty and model comparison, as methods of every PointProcess."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial
import copy
import re

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .parallel import parallel_map, report_failures
from .dataset import PointProcessDataset

if TYPE_CHECKING:
    from .point_process import PointProcess


def _replicate(
    seed_seq: np.random.SeedSequence,
    model: PointProcess,
    template: PointProcessDataset,
    statistic: Callable[[PointProcess, PointProcessDataset], Any],
    parametric: bool,
    refit: bool,
    refit_n_starts: int,
) -> Tuple[Any, Optional[str]]:
    """(statistic, None) on one simulated or fish-resampled dataset, or (None, error)."""
    rng = np.random.default_rng(seed_seq)
    try:
        model_b = copy.deepcopy(model)
        dataset_b = model_b.simulate_dataset(template, rng) if parametric else template.resample(rng)
        if refit:
            model_b.initial_guesses = list(np.asarray(model.params_, dtype=float))
            if refit_n_starts <= 1:
                model_b.fit(dataset_b)
            else:
                model_b.fit_multistart(
                    dataset_b, n_starts=refit_n_starts, seed=int(rng.integers(0, 2**31 - 1)), n_jobs=1
                )
        return statistic(model_b, dataset_b), None
    except Exception as exc:
        return None, repr(exc)


def refit_replicates(
    model: PointProcess,
    template: PointProcessDataset,
    statistic: Callable[[PointProcess, PointProcessDataset], Any],
    n_boot: int,
    seed: int = 42,
    parametric: bool = True,
    refit: bool = True,
    refit_n_starts: int = 1,
    n_jobs: int = -1,
    desc: str = "Bootstrap",
    progress: bool = True,
) -> Tuple[List[Any], List[str]]:
    """
    statistic(model, dataset) on n_boot datasets simulated from the fitted
    model (parametric) or resampled by fish, each refitted from the fitted
    parameters (refit). Returns the successful values and the errors.
    """
    model._require_fitted()
    seeds = np.random.SeedSequence(seed).spawn(n_boot)
    job = partial(
        _replicate,
        model=model,
        template=template,
        statistic=statistic,
        parametric=parametric,
        refit=refit,
        refit_n_starts=refit_n_starts,
    )
    # One BLAS/OpenMP thread per worker to avoid oversubscription.
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        results = parallel_map(
            job, seeds, n_jobs=n_jobs, desc=desc, progress=progress, batch_size=1, pre_dispatch="n_jobs"
        )
    values = [value for value, error in results if error is None]
    errors = [error for _, error in results if error is not None]
    print(f"{desc}: {len(values)}/{n_boot} successful.")
    report_failures(f"{model.name} {desc}", errors, n_boot)
    return values, errors


def fitted_params(model: PointProcess, dataset: PointProcessDataset) -> np.ndarray:
    return model.params_


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


class InferenceMixin:
    _LOG_SCALE_PATTERN = re.compile(
        r"(tau|sigma|beta|r_dispersion|excitation)", re.IGNORECASE
    )

    def fit(self, dataset: PointProcessDataset, method: str = "L-BFGS-B", **kwargs):
        res = minimize(
            self._nll,
            x0=self.initial_guesses,
            args=(dataset,),
            method=method,
            bounds=self.bounds,
            **kwargs,
        )

        valid = res.success and np.isfinite(res.fun) and np.all(np.isfinite(res.x))
        if not valid:
            raise RuntimeError(
                f"Optimization failed for {self.name}: "
                f"success={res.success}, fun={res.fun}, message={res.message}"
            )

        self.fit_result = res
        self.set_params(res.x)
        return self

    def _infer_log_scale_params(self) -> set:
        return {p for p in self.param_names if self._LOG_SCALE_PATTERN.search(p)}

    def _sampling_bounds(
        self, i: int, default: np.ndarray, fallback_scale: float = 10.0
    ):
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
        log_scale = (
            log_scale_params
            if log_scale_params is not None
            else self._infer_log_scale_params()
        )

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
                    span = (
                        (hi - lo) if np.isfinite(hi - lo) else max(abs(default[i]), 1.0)
                    )
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
        method: str = "L-BFGS-B",
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
        inits = self._generate_multistart_inits(
            rng, n_starts, log_scale_params=log_scale_params
        )

        def _run_one(x0: np.ndarray) -> Any:
            try:
                res = minimize(
                    self._nll,
                    x0=x0,
                    args=(dataset,),
                    method=method,
                    bounds=self.bounds,
                    **kwargs,
                )
            except Exception as exc:
                return repr(exc)
            if res.success and np.isfinite(res.fun) and np.all(np.isfinite(res.x)):
                return res
            return f"not converged: {res.message}"

        results = parallel_map(_run_one, inits, n_jobs=n_jobs, desc="Fit multistart")
        report_failures(
            f"{self.name} multistart", [r for r in results if isinstance(r, str)], n_starts
        )
        results = [None if isinstance(r, str) else r for r in results]

        valid = [r for r in results if r is not None]
        if not valid:
            raise RuntimeError(
                f"All {n_starts} multistart optimizations failed for {self.name}."
            )

        best = min(valid, key=lambda r: r.fun)
        n_converged = len(valid)
        n_at_global = sum(1 for r in valid if r.fun <= best.fun + tol_same_optimum)

        status = (
            "OK (looks unimodal)"
            if n_at_global == n_converged
            else f"WARNING: {n_converged - n_at_global} start(s) converged to a DIFFERENT, worse optimum "
            f"-- {self.name} may be poorly identified; treat AIC comparisons involving it with caution."
        )
        print(
            f"[{self.name}] multistart: {n_converged}/{n_starts} converged, "
            f"best NLL={best.fun:.3f}. {status}"
        )

        self.fit_result = best
        self.set_params(best.x)

        return pd.DataFrame(
            {
                "start_idx": range(len(results)),
                "converged": [r is not None for r in results],
                "nll": [r.fun if r is not None else np.nan for r in results],
            }
        )

    def bootstrap(
        self,
        dataset: PointProcessDataset,
        n_boot: int = 500,
        seed: int = 42,
        ci: float = 95.0,
        n_jobs: int = -1,
    ) -> pd.DataFrame:

        valid_boot_params, _ = refit_replicates(
            self, dataset, fitted_params, n_boot, seed, parametric=False, n_jobs=n_jobs, desc="Bootstrap"
        )
        if not valid_boot_params:
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

    def estimate_hessian(
        self,
        dataset: PointProcessDataset,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        Central-difference Hessian of the NLL. Rows and columns of parameters
        within one step of a bound are NaN: the NLL may be undefined beyond
        the bound, and the curvature there is not a standard error anyway.
        """
        if self.params_ is None or self.fit_result is None:
            raise ValueError("Model must be fitted before estimating the Hessian.")

        params = np.array(self.params_, dtype=float)
        k = len(params)
        h = eps * (1.0 + np.abs(params))
        lower = np.array([-np.inf if lo is None else lo for lo, _ in self.bounds])
        upper = np.array([np.inf if hi is None else hi for _, hi in self.bounds])
        interior = (params - h >= lower) & (params + h <= upper)

        def obj(step: np.ndarray) -> float:
            return self._nll(params + step, dataset)

        f0 = self.fit_result.fun
        hessian = np.full((k, k), np.nan)
        unit = np.eye(k) * h

        for i in np.flatnonzero(interior):
            hessian[i, i] = (obj(unit[i]) - 2 * f0 + obj(-unit[i])) / (h[i] ** 2)
            for j in np.flatnonzero(interior[i + 1 :]) + i + 1:
                val = (
                    obj(unit[i] + unit[j])
                    - obj(unit[i] - unit[j])
                    - obj(-unit[i] + unit[j])
                    + obj(-unit[i] - unit[j])
                ) / (4 * h[i] * h[j])
                hessian[i, j] = val
                hessian[j, i] = val

        return hessian

    def estimate_parameter_correlation(
        self,
        dataset: PointProcessDataset,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Correlation from the inverse Hessian; NaN for parameters at a bound."""
        hessian = self.estimate_hessian(dataset, eps=eps)
        interior = np.isfinite(np.diag(hessian))
        sub_hessian = hessian[np.ix_(interior, interior)]

        try:
            cov = np.linalg.inv(sub_hessian)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(sub_hessian)

        std_devs = np.sqrt(np.maximum(0, np.diag(cov)))
        with np.errstate(divide="ignore", invalid="ignore"):
            sub_corr = np.clip(cov / np.outer(std_devs, std_devs), -1.0, 1.0)

        corr = np.full_like(hessian, np.nan)
        corr[np.ix_(interior, interior)] = sub_corr
        return corr


class ModelComparator:

    @staticmethod
    def likelihood_ratio_test(
        model_null: PointProcess, model_alt: PointProcess
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
            "Significant (α=0.05)": p_val < 0.05,
        }

    @staticmethod
    def compare(
        models: List[PointProcess],
        dataset: PointProcessDataset,
        method: str = "L-BFGS-B",
        null_model: Optional[PointProcess] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[PointProcess]]:
        names = [m.name for m in models]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ValueError(f"Model names must be unique: {duplicated}")

        if null_model is not None:
            null_fitted, null_error = _fit_one_model(
                null_model, dataset, method, kwargs
            )
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
                    if not np.isnan(ll_null)
                    else np.nan
                )
                records.append(
                    {
                        "Model Name": model.name,
                        "Params (k)": len(model.param_names),
                        "Log-Likelihood": model.log_likelihood,
                        "AIC": model.aic,
                        "McFadden R2": mcfadden_r2,
                        "r_dispersion": model.dispersion_r,
                        "Overdispersion (1/r)": model.overdispersion_index,
                        "Converged": True,
                    }
                )
            else:
                print(
                    f"[ModelComparator] WARNING: fit failed for {model.name}: {error}"
                )
                records.append(
                    {
                        "Model Name": model.name,
                        "Params (k)": len(model.param_names),
                        "Log-Likelihood": np.nan,
                        "AIC": np.nan,
                        "McFadden R2": np.nan,
                        "r_dispersion": np.nan,
                        "Overdispersion (1/r)": np.nan,
                        "Converged": False,
                    }
                )

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
