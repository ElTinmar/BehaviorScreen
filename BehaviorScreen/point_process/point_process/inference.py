"""Fitting, uncertainty and model comparison, as methods of every PointProcess."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial
import copy
import json
import pickle
import time
from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import chi2

from .parallel import parallel_map, report_failures
from .dataset import PointProcessDataset

if TYPE_CHECKING:
    from .point_process import PointProcess


SCREEN_DT = 0.04  # coarse integration step (s) where models with bout history first fit; the final fit is on their own


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(0, np.diag(cov)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(cov / np.outer(std, std), -1.0, 1.0)


def set_integration_dt(model: "PointProcess", dt: float) -> None:
    """Integration step of a model and of every process under it."""
    model.integration_dt = dt
    for part in vars(model).values():
        if hasattr(part, "integration_dt") and hasattr(part, "_nll"):
            set_integration_dt(part, dt)


def newton_step_matrix(hessian: np.ndarray) -> Optional[np.ndarray]:
    """
    Inverse Hessian for Newton steps, its eigenvalues floored at 1e-8 of the
    largest (barely determined directions take bounded steps). Parameters at a
    bound (NaN rows of estimate_hessian) get zero rows and columns: Newton steps
    hold them there. None if the rest is not finite.
    """
    free = np.isfinite(np.diag(hessian))
    block = hessian[np.ix_(free, free)]
    if not free.any() or not np.all(np.isfinite(block)):
        return None
    w, v = np.linalg.eigh((block + block.T) / 2)
    if w.max() <= 0:
        return None
    steps = np.zeros_like(hessian)
    steps[np.ix_(free, free)] = (v / np.maximum(w, 1e-8 * w.max())) @ v.T
    return steps


def newton_refit(
    model: PointProcess, dataset: PointProcessDataset, step_matrix: np.ndarray, max_steps: int = 20, tol: float = 0.01, eps: float = 1e-5,
) -> Optional[str]:
    """
    Refit from model.initial_guesses (an optimum of similar data) by Newton steps
    with a fixed inverse Hessian and central-difference gradients, each step halved
    until the NLL decreases and kept inside the bounds; done once a step gains, or
    would gain (half the Newton decrement), less than tol. Parameters with a zero
    row in step_matrix stay at their bound, provided the NLL rises when they move
    inside it. Returns None when done, else why not (the model's initial_guesses
    are then the best point reached, where the optimizer should continue).
    """
    lower = np.array([-np.inf if lo is None else lo for lo, _ in model.bounds])
    upper = np.array([np.inf if hi is None else hi for _, hi in model.bounds])
    nll = lambda x: model.nll(x, dataset)
    x = np.array(model.initial_guesses, dtype=float)
    f = nll(x)
    free = np.diag(step_matrix) > 0

    def stop(reason: str) -> str:
        model.initial_guesses = list(x)
        return reason

    def done() -> Optional[str]:
        h = eps * (1.0 + np.abs(x))
        inward = np.where(x - lower < upper - x, h, -h)  # held parameters move off their nearer bound
        if any(nll(x + np.eye(len(x))[i] * inward[i]) < f for i in np.flatnonzero(~free)):
            return stop("a parameter leaves its bound")
        model.set_params(x)
        model.fit_result = OptimizeResult(x=x, fun=f, success=True, message="Newton steps with a fixed Hessian")
        return None

    for _ in range(max_steps):
        h = eps * (1.0 + np.abs(x))
        if np.any((x - h < lower)[free]) or np.any((x + h > upper)[free]):
            return stop("reached a bound")
        g = np.zeros(len(x))
        if model.supports_jax():
            g[free] = model.nll_and_grad(x, dataset)[1][free]
        else:
            g[free] = [(nll(x + dx) - nll(x - dx)) / (2 * hi) for dx, hi in zip(np.diag(h)[free], h[free])]
        step, t = -step_matrix @ g, 1.0
        if -g @ step / 2 < tol:  # already within tol of the optimum
            return done()
        while (f_new := nll(x_new := np.clip(x + t * step, lower, upper))) >= f:
            t /= 2
            if t < 1 / 16:
                return stop("no decrease along the step")
        gain, x, f = f - f_new, x_new, f_new
        if gain < tol:
            return done()
    return stop(f"not converged in {max_steps} steps")


def refit_from_fit(model: PointProcess, dataset: PointProcessDataset, step_matrix: Optional[np.ndarray]) -> Tuple[PointProcess, str]:
    """
    A copy of a fitted model refitted to similar data from its parameters: by Newton
    steps with step_matrix (newton_step_matrix of its Hessian), else by the optimizer,
    from where the steps stopped. Also returns how: "newton", or why the optimizer was needed.
    """
    refit = copy.deepcopy(model)
    refit.initial_guesses = list(np.asarray(model.params_, dtype=float))
    reason = "no Hessian" if step_matrix is None else newton_refit(refit, dataset, step_matrix)
    if reason is not None:
        refit.fit(dataset)  # from where the steps stopped, near the optimum: no coarse screening
    return refit, reason or "newton"


def _replicate(
    seed_seq: np.random.SeedSequence,
    model: PointProcess,
    template: PointProcessDataset,
    statistic: Callable[[PointProcess, PointProcessDataset], Any],
    parametric: bool,
    refit: bool,
    refit_n_starts: int,
    step_matrix: Optional[np.ndarray] = None,
) -> Tuple[Any, Optional[str], str]:
    """(statistic, None, how it was refitted) on one simulated or fish-resampled dataset, or (None, error, "")."""
    rng = np.random.default_rng(seed_seq)
    try:
        dataset_b = copy.deepcopy(model).simulate_dataset(template, rng) if parametric else template.resample(rng)
        model_b, how = copy.deepcopy(model), "not refitted"
        if refit and refit_n_starts > 1:
            model_b.initial_guesses = list(np.asarray(model.params_, dtype=float))
            model_b.fit_multistart(dataset_b, n_starts=refit_n_starts, seed=int(rng.integers(0, 2**31 - 1)), n_jobs=1)
            how = "multistart"
        elif refit:
            model_b, how = refit_from_fit(model, dataset_b, step_matrix)
        return statistic(model_b, dataset_b), None, how
    except Exception as exc:
        return None, repr(exc), ""


def _stored_replicate(job: Tuple[int, np.random.SeedSequence], store: Optional[Path], **kwargs) -> Tuple[Any, Optional[str], str]:
    """_replicate, its successful result written to store/<index>.pkl as soon as it is done."""
    index, seed_seq = job
    result = _replicate(seed_seq, **kwargs)
    if store is not None and result[1] is None:
        with open(store / f"{index:05d}.pkl", "wb") as f:
            pickle.dump(result, f)
    return result


def load_replicates(store: Path, key: dict) -> Dict[str, Any]:
    """What is saved in store for this key (file stem: content); anything saved for another key is deleted."""
    store.mkdir(parents=True, exist_ok=True)
    key_path = store / "key.json"
    if key_path.exists() and json.loads(key_path.read_text()) != key:
        for path in store.glob("*.pkl"):
            path.unlink()
    key_path.write_text(json.dumps(key))
    done = {}
    for path in store.glob("*.pkl"):
        with open(path, "rb") as f:
            done[path.stem] = pickle.load(f)
    return done


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
    store: Optional[Path] = None,
) -> Tuple[List[Any], List[str]]:
    """
    statistic(model, dataset) on n_boot datasets simulated from the fitted
    model (parametric) or resampled by fish, each refitted from the fitted
    parameters (refit): by Newton steps with the Hessian of the fit (one estimate
    for all), else, where those fail, by the optimizer. Returns the successful
    values and the errors.

    store: a folder where each successful replicate is saved as it is done, and
    reused by later calls with the same model parameters, data size, seed and kind:
    an interrupted call resumes, a call with a larger n_boot only adds replicates
    (replicate i always has the same random seed), a smaller one uses the first.
    """
    model._require_fitted()
    seeds = np.random.SeedSequence(seed).spawn(n_boot)
    key = {"desc": desc, "seed": seed, "parametric": parametric, "refit": refit, "refit_n_starts": refit_n_starts,
           "params": [float(p) for p in model.params_], "n_fish": template.num_fish, "n_events": len(template.event_times)}
    done = {} if store is None else {int(i): r for i, r in load_replicates(store, key).items() if int(i) < n_boot}
    todo = [i for i in range(n_boot) if i not in done]
    step_matrix = newton_step_matrix(model.estimate_hessian(template)) if todo and refit and refit_n_starts <= 1 else None
    job = partial(
        _stored_replicate,
        store=store,
        model=model,
        template=template,
        statistic=statistic,
        parametric=parametric,
        refit=refit,
        refit_n_starts=refit_n_starts,
        step_matrix=step_matrix,
    )
    if done:
        print(f"{desc}: {len(done)} replicates reused from {store}, {len(todo)} to compute.")
    computed = dict(zip(todo, parallel_map(job, [(i, seeds[i]) for i in todo], n_jobs=n_jobs, desc=desc, progress=progress, batch_size=1, pre_dispatch="n_jobs"))) if todo else {}
    results = [done[i] if i in done else computed[i] for i in range(n_boot)]
    values = [value for value, error, _ in results if error is None]
    errors = [error for _, error, _ in results if error is not None]
    hows = Counter(how for _, error, how in results if error is None)
    others = ", ".join(f"{n} {how}" for how, n in hows.most_common() if how != "newton")
    print(f"{desc}: {len(values)}/{n_boot} successful" + (f", {hows['newton']} refitted by Newton steps" + (f"; by the optimizer: {others}" if others else "") + "." if refit and refit_n_starts <= 1 and hows else "."))
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
        r"(tau|sigma|beta|r_dispersion|r_trial|excitation|^D$)", re.IGNORECASE
    )

    def _minimize(
        self, x0: Union[List[float], np.ndarray], dataset: PointProcessDataset, method: str, max_restarts: int = 5, precise: bool = True, **kwargs,
    ) -> Any:
        """
        minimize, restarted from its own result while that improves the NLL: L-BFGS-B
        can stop well short of the optimum in a curved valley once its Hessian estimate,
        built from far-away steps, goes stale; a fresh start from there continues.
        precise (exact gradients): then continued with tighter tolerances (1e-10 can still stop
        short in a slow valley); from a converged point its line search can end ABNORMAL
        (no decrease left in the NLL's last digits), which keeps its point if not worse.
        """
        fun, jac = (self.nll_and_grad, True) if self.supports_jax() else (self._nll, None)
        run = lambda x: minimize(fun, x0=x, args=(dataset,), method=method, jac=jac, bounds=self.bounds, **kwargs)
        res = run(x0)
        for _ in range(max_restarts):
            again = run(res.x)
            if not (again.success and again.fun < res.fun - 1e-3):
                break
            res = again
        if precise and jac and method == "L-BFGS-B" and res.success:
            options = {"ftol": 1e-11, "gtol": 1e-5, **kwargs.get("options", {})}
            tight = minimize(fun, x0=res.x, args=(dataset,), method=method, jac=jac, bounds=self.bounds, **{**kwargs, "options": options})
            if np.isfinite(tight.fun) and tight.fun <= res.fun:
                res = OptimizeResult({**tight, "success": True, "nfev": res.nfev + tight.nfev})
        return res

    def fit(self, dataset: PointProcessDataset, method: str = "L-BFGS-B", **kwargs):
        res = self._minimize(self.initial_guesses, dataset, method, **kwargs)

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
        global_factor: float = 5.0,
        global_width: float = 4.0,
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
                    span = min(hi - lo, 2 * global_width) if np.isfinite(hi - lo) else max(abs(default[i]), 1.0)
                    x[i] = default[i] + rng.normal(0, jitter_frac * span)
                x[i] = np.clip(x[i], lo, hi)
            starts.append(x)

        # (b) wide draws: within the bounds, up to a factor global_factor (log-scale) or
        # +-global_width (others) from the default. Wider, they land where a dip or a
        # logit saturates and the likelihood is flat: optimizers stop there, short of the optimum.
        for _ in range(n_uniform):
            x = np.empty(k)
            for i in range(k):
                lo, hi = self._sampling_bounds(i, default)
                if self.param_names[i] in log_scale and lo > 0 and default[i] > 0:
                    lo, hi = max(lo, default[i] / global_factor), min(hi, default[i] * global_factor)
                    x[i] = np.exp(rng.uniform(np.log(lo), np.log(hi)))
                else:
                    x[i] = rng.uniform(max(lo, default[i] - global_width), min(hi, default[i] + global_width))
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
        tol_same_optimum: float = 0.5,
        screen_dt: Optional[float] = SCREEN_DT,
        n_refine: int = 3,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Refit from n_starts different initializations, keep the best
        converged result as self.params_/self.fit_result (same
        postcondition as .fit()), and return a per-start table so
        multimodality is visible rather than hidden by AIC silently
        picking whichever local optimum was reached.

        Models with bout history (a renewal grid, the costly part) first fit every
        start on a coarser grid (screen_dt), then refit the n_refine best from where
        they stopped on their own grid: the final fit is the same, at a fraction of the
        cost. Screening stops at scipy's default tolerance, which ranks the starts; the refits
        are precise. The per-start NLLs (and the multimodality report) are then the screening's.
        """
        rng = np.random.default_rng(seed)
        inits = self._generate_multistart_inits(
            rng, n_starts, log_scale_params=log_scale_params
        )

        def _run_one(x0: np.ndarray, precise: bool = True) -> Any:
            try:
                res = self._minimize(x0, dataset, method, precise=precise, **kwargs)
            except Exception as exc:
                return repr(exc)
            if res.success and np.isfinite(res.fun) and np.all(np.isfinite(res.x)):
                return res
            return f"not converged: {res.message}"

        fine_dt = self.integration_dt
        screen = self.uses_history and screen_dt is not None and screen_dt > fine_dt
        try:
            if screen:
                set_integration_dt(self, screen_dt)
            run = partial(_run_one, precise=not screen)  # screening only ranks the starts: the refits are precise
            results = parallel_map(run, inits, n_jobs=n_jobs, desc=f"Fit multistart{f' (dt {screen_dt:g} s)' if screen else ''}")
        finally:
            set_integration_dt(self, fine_dt)
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
        refined = []
        if screen:
            ranked = sorted(valid, key=lambda r: r.fun)[:n_refine]
            refined = [r for r in parallel_map(_run_one, [r.x for r in ranked], n_jobs=n_jobs, desc=f"Refine (dt {fine_dt:g} s)")
                       if not isinstance(r, str)]
            if not refined:
                raise RuntimeError(f"{self.name}: no refit on the {fine_dt:g} s grid converged.")
        n_converged = len(valid)
        n_at_global = sum(1 for r in valid if r.fun <= best.fun + tol_same_optimum)

        worse = [r.fun - best.fun for r in valid if r.fun > best.fun + tol_same_optimum]
        print(
            f"[{self.name}] multistart: {n_converged}/{n_starts} converged, {n_at_global} at the best NLL={best.fun:.3f}"
            + (f", {len(worse)} stopped {min(worse):.1f} to {max(worse):.1f} above it" if worse else "")
            + (". WARNING: only one start reached the best; a better optimum may exist." if n_at_global == 1 else ".")
            + (f" Screened on a {screen_dt:g} s grid; the best {len(refined)} refitted on {fine_dt:g} s: NLL={min(r.fun for r in refined):.3f}." if screen else "")
        )
        if screen:
            best = min(refined, key=lambda r: r.fun)

        self.fit_result = best
        self.set_params(best.x)
        self.multistart_ = {
            "starts": n_starts, "converged": n_converged, "at_best": n_at_global,
            "nearest_gap": min(worse) if worse else np.nan, "farthest_gap": max(worse) if worse else np.nan,
            "screen_dt": screen_dt if screen else np.nan,
        }
        self.multistart_table_ = pd.DataFrame(
            {
                "start_idx": range(len(results)),
                "converged": [r is not None for r in results],
                "nll": [r.fun if r is not None else np.nan for r in results],
                **{name: [r.x[i] if r is not None else np.nan for r in results] for i, name in enumerate(self.param_names)},
            }
        )
        return self.multistart_table_

    def bootstrap(
        self,
        dataset: PointProcessDataset,
        n_boot: int = 500,
        seed: int = 42,
        ci: float = 95.0,
        n_jobs: int = -1,
        store: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Fish bootstrap of the parameters; store: see refit_replicates."""
        valid_boot_params, _ = refit_replicates(
            self, dataset, fitted_params, n_boot, seed, parametric=False, n_jobs=n_jobs, desc="Bootstrap", store=store
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
        Hessian of the NLL: exact with a JAX likelihood, else by central differences.
        Rows and columns of parameters within one step of a bound are NaN: the NLL
        may be undefined beyond the bound, and the curvature there is not a standard error anyway.
        """
        if self.params_ is None or self.fit_result is None:
            raise ValueError("Model must be fitted before estimating the Hessian.")

        params = np.array(self.params_, dtype=float)
        k = len(params)
        h = eps * (1.0 + np.abs(params))
        lower = np.array([-np.inf if lo is None else lo for lo, _ in self.bounds])
        upper = np.array([np.inf if hi is None else hi for _, hi in self.bounds])
        interior = (params - h >= lower) & (params + h <= upper)
        if self.supports_jax():
            hessian = self.jax_hessian(params, dataset)
            hessian[~interior, :] = hessian[:, ~interior] = np.nan
            return hessian

        def obj(step: np.ndarray) -> float:
            return self._nll(params + step, dataset)

        f0 = obj(np.zeros(k))
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

    def estimate_parameter_covariance(self, dataset: PointProcessDataset, eps: float = 1e-5) -> np.ndarray:
        """Inverse Hessian; NaN rows and columns for parameters at a bound."""
        hessian = self.estimate_hessian(dataset, eps=eps)
        interior = np.isfinite(np.diag(hessian))
        sub_hessian = hessian[np.ix_(interior, interior)]
        try:
            sub_cov = np.linalg.inv(sub_hessian)
        except np.linalg.LinAlgError:
            sub_cov = np.linalg.pinv(sub_hessian)
        cov = np.full_like(hessian, np.nan)
        cov[np.ix_(interior, interior)] = sub_cov
        return cov

    def estimate_parameter_correlation(self, dataset: PointProcessDataset, eps: float = 1e-5) -> np.ndarray:
        """Correlation from the inverse Hessian; NaN for parameters at a bound."""
        return correlation_from_covariance(self.estimate_parameter_covariance(dataset, eps=eps))


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
        on_fitted: Optional[Callable[[int, PointProcess], None]] = None,
        is_cached: Optional[Callable[[int, PointProcess], bool]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[PointProcess]]:
        """
        on_fitted(index, model): called after each successful fit, before the next model's.
        is_cached(index, model): True when it has set the model's parameters and fit_result
        (a saved fit still valid): the model is not refitted. fit_seconds_ is set on fitted models.
        """
        names = [m.name for m in models]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ValueError(f"Model names must be unique: {duplicated}")

        if null_model is not None and not any(m is null_model for m in models):
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

        results = []
        fitted_values: Dict[str, float] = {}  # parameters already fitted, by name: where the next models start
        for index, m in enumerate(models):
            m.initial_guesses = [
                float(np.clip(fitted_values.get(n, g), lo if lo is not None else -np.inf, hi if hi is not None else np.inf))
                for n, g, (lo, hi) in zip(m.param_names, m.initial_guesses, m.bounds)
            ]
            if is_cached is not None and is_cached(index, m):
                results.append((m, None))
            else:
                started = time.perf_counter()
                results.append(_fit_one_model(m, dataset, method, kwargs))
                m.fit_seconds_ = time.perf_counter() - started
            if results[-1][1] is None:
                fitted_values.update(m.param_dict_)
                if on_fitted is not None:
                    on_fitted(index, m)
        fitted_models: List[PointProcess] = []
        records = []
        for model, error in results:  # a null among the models: fitted with them
            if model is null_model:
                ll_null = model.log_likelihood if error is None else np.nan

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
