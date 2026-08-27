# point_process/survival_process.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .dataset import PointProcessDataset
from .point_process import PointProcess
from .poisson_process import RateKernel

class SurvivalKernelFactory:

    @staticmethod
    def constant_hazard() -> RateKernel:
        """Exponential (memoryless) hazard -- the survival analog of
        homogeneous_poisson. Use as the null_model for SurvivalProcess
        comparisons: fits ONLY overall response propensity, no latency
        structure at all."""
        def _func(t, trial, params):
            (B,) = params
            return B * np.ones_like(t + 0.0 * trial)
        return RateKernel(
            name="SurvivalConstantHazard",
            func=_func,
            param_names=["B"],
            initial_guesses=[0.3],
            bounds=[(1e-4, 20.0)],
            latex_formula=r"$h = B$",
        )

    @staticmethod
    def gaussian_bump_time_only() -> RateKernel:
        """
        h(t) = H * exp(-(t-mu)^2 / 2*sigma^2)

        No baseline (see class-level rationale: a tonic floor explains ongoing
        bouts in a recurrent model, but is largely redundant once only the
        first response per trial matters). H simultaneously sets peak hazard
        AND -- via its Gaussian integral -- the trial's total implied response
        probability; mu/sigma give latency and its spread. This is the
        survival-analysis equivalent of a mixture-cure model, with cure
        fraction and latency density coupled through H/sigma rather than fit
        as separate parameters -- deliberately, to keep this minimal.
        """
        def _func(t, trial, params):
            H, mu, sigma = params
            return H * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        return RateKernel(
            name="SurvivalGaussianBump(TimeOnly)",
            func=_func,
            param_names=["H", "mu", "sigma"],
            initial_guesses=[1.0, 0.15, 0.1],
            bounds=[(0.001, 30.0), (0.01, 2.0), (0.005, 3.0)],
            latex_formula=r"$h(t) = H \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)$",
        )

    @staticmethod
    def gaussian_bump_habituating() -> RateKernel:
        """As above, +1 param: H(m) = H * exp(alpha * m). Only add this if
        LR test against the time_only nested null (see below) justifies it."""
        def _func(t, trial, params):
            H, mu, sigma, alpha = params
            height = H * np.exp(alpha * trial)
            return height * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        return RateKernel(
            name="SurvivalGaussianBump(Habituating)",
            func=_func,
            param_names=["H", "mu", "sigma", "alpha"],
            initial_guesses=[1.0, 0.15, 0.1, 0.0],
            bounds=[(0.001, 30.0), (0.01, 2.0), (0.005, 3.0), (-2.0, 2.0)],
            latex_formula=r"$h(t,m) = H e^{\alpha m} \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)$",
        )

    @staticmethod
    def survival_looming_bump(t_critical: float = 5.0) -> RateKernel:
        def _func(t, trial, params):
            H, mu, sigma = params
            return H * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        return RateKernel(
            name="SurvivalLoomingBump",
            func=_func,
            param_names=["H", "mu", "sigma"],
            initial_guesses=[1.2, t_critical, 0.15],
            bounds=[(0.001, 30.0), (t_critical - 1.5, t_critical + 1.0), (0.005, 3.0)],
            latex_formula=r"$h(t) = H \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)$",
        )

    @staticmethod
    def gaussian_bump_baseline() -> RateKernel:
        """h(t) = H*exp(-(t-mu)^2/2sigma^2) + B. Add ONLY if LR test against
        the no-baseline bump is significant -- KM plateau not being flat
        (slow decline continuing well past the burst) is the empirical
        trigger for trying this."""
        def _func(t, trial, params):
            H, mu, sigma, B = params
            return H * np.exp(-0.5 * ((t - mu) / sigma) ** 2) + B
        return RateKernel(
            name="SurvivalGaussianBump(Baseline)",
            func=_func,
            param_names=["H", "mu", "sigma", "B"],
            initial_guesses=[1.0, 0.15, 0.1, 0.02],
            bounds=[(0.001, 30.0), (0.01, 2.0), (0.005, 3.0), (1e-4, 1.0)],
            latex_formula=r"$h(t) = H \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right) + B$",
        )

class SurvivalProcess(PointProcess):
    """
    First-passage-time (right-censored survival) model built on the SAME
    RateKernel objects used by PoissonProcess: lambda(t, m) is reinterpreted
    as a hazard h(t, m), and each (fish, trial) stream is reduced to a
    single observation -- the time of its FIRST event, or duration_s if
    none occurred (right-censored) -- discarding any subsequent events in
    that stream.

    THEORETICAL NOTE: for a stream with at most one event, the ordinary
    point-process likelihood

        L = h(t_event) * exp(-Lambda(t_event))          [event observed]
        L = exp(-Lambda(duration_s))                     [censored]

    with Lambda(t) = integral_0^t h -- is exactly the standard survival
    likelihood (h == hazard, Lambda == cumulative hazard, exp(-Lambda) ==
    survival function). This is a constrained special case of the same
    machinery used elsewhere in this module, not a different model family
    -- see _nll below, which is almost syntactically identical to
    PoissonProcess._nll, just truncated at each stream's first event (or
    duration_s) instead of always integrating to duration_s and summing
    log-intensity over every event.

    WHEN TO USE INSTEAD OF PoissonProcess/HawkesProcess/RenewalProcess:
    stimuli that provoke "nothing, then at most one reaction" (dark flash
    O-bends, looming SLCs) rather than a modulated ongoing bout train.
    Check dataset.frac_streams_with_multiple_events first -- if repeat
    within-trial bouts are common, this model discards real structure and
    a recurrent model is more appropriate.

    COMPATIBLE WITH GammaMixedEffectsProcess: mixed_effects_likelihood_terms
    below satisfies the "gain multiplies the ENTIRE intensity uniformly"
    precondition -- g_f * h(t) implies P(respond by T) = 1 - exp(-g_f *
    Lambda(T)), i.e. a Gamma-frailty survival model (Vaupel, Manton &
    Stallard 1979). GammaMixedEffectsProcess(SurvivalProcess(kernel)) is
    supported and meaningful.

    KNOWN LIMITATION of that combination: GammaMixedEffectsProcess's own
    _stream_tau_values override assumes a recurrent stream (walks every
    event in t_ev per trial, not just the first), so
    GammaMixedEffectsProcess(SurvivalProcess(...)).diagnose() will use
    that recurrent-process assumption, NOT this class's censoring-aware
    cox_snell_calibration(). Not fixed here -- if you need calibration
    diagnostics for the frailty-wrapped version, extract
    base_process.params_ after fit() and run the unwrapped SurvivalProcess's
    own diagnostics as a substitute (approximate: ignores frailty).

    CAVEAT on ModelPlotter/ModelComparator's generic (Hz-based) plots:
    compute_expected_rate() returns the raw fitted HAZARD h(t, m), not a
    first-passage density -- purely so the existing generic plotting
    utilities (which compare against dataset.time_histogram_hz, pooling
    EVERY detected bout) work unmodified. Where repeat bouts are rare this
    mismatch is cosmetic; prefer population_survival_curve() /
    kaplan_meier_curve() for anything beyond a rough visual check.

    ALSO NOT OVERRIDDEN: compute_residuals() / residual_2d_autocorrelation()
    (inherited from PointProcess unchanged). They bin ALL events per
    (trial, time) cell and compare to a Poisson/NB mean -- a model that
    doesn't reflect the shrinking-risk-set structure of first-passage
    data. Not used by this class's diagnose(); call them directly at your
    own risk.
    """

    def __init__(self, kernel: RateKernel, integration_dt: float = 0.02):
        super().__init__(integration_dt)
        self.name = f"Survival {kernel.name}"
        self.latex_formula = kernel.latex_formula
        self.kernel = kernel
        self.initial_guesses = kernel.initial_guesses
        self.bounds = kernel.bounds
        self.param_names = kernel.param_names

    # -- Core reduction: stream -> (t_obs, censored) ------------------------

    @staticmethod
    def _first_event_or_censor(t_ev: np.ndarray, duration_s: float) -> Tuple[float, bool]:
        """
        Reduce a (fish, trial) stream to a single first-passage observation.
        Any events after the first are discarded (see class CAVEAT re:
        multi-event streams -- check dataset.frac_streams_with_multiple_events
        before trusting this reduction for a given condition).
        """
        if len(t_ev) == 0:
            return float(duration_s), True
        return float(np.min(t_ev)), False

    # -- Likelihood -----------------------------------------------------------

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        total = 0.0
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            t_obs, censored = self._first_event_or_censor(t_ev, dataset.duration_s)

            H = float(self.kernel.integrate(
                duration_s=t_obs, trial=t_idx, params=params,
                integration_dt=self.integration_dt,
            ))

            if censored:
                total += H
            else:
                h = self.kernel.evaluate(
                    np.array([t_obs]), np.array([float(t_idx)]), params
                )[0]
                total += H - np.log(max(h, 1e-12))
        return total

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        N_f: total observed (uncensored) first-passage events per fish.
        S_f: total cumulative hazard exposure per fish, i.e. sum over that
        fish's trials of Lambda(t_obs) -- exactly the "S_f" a Gamma-frailty
        survival model needs (Vaupel et al.); see class docstring.
        """
        N_f = np.zeros(dataset.num_fish, dtype=float)
        S_f = np.zeros(dataset.num_fish, dtype=float)
        base_ll = 0.0

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            t_obs, censored = self._first_event_or_censor(t_ev, dataset.duration_s)

            H = float(self.kernel.integrate(
                duration_s=t_obs, trial=t_idx, params=params,
                integration_dt=self.integration_dt,
            ))
            S_f[f_idx] += H

            if not censored:
                N_f[f_idx] += 1.0
                h = self.kernel.evaluate(
                    np.array([t_obs]), np.array([float(t_idx)]), params
                )[0]
                base_ll += np.log(max(h, 1e-12))

        return base_ll, N_f, S_f

    # -- Prediction / population-level curves --------------------------------

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray]) -> np.ndarray:
        """Raw fitted hazard h(t, m) -- see class CAVEAT re: plotting semantics."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return self.kernel.evaluate(t, trial, self.params_)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        """Raw fitted hazard surface, matching PoissonProcess's convention
        exactly so ModelPlotter/ModelComparator work unmodified."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        t_2d = dataset.t_centers[None, :]
        trials_2d = np.arange(dataset.num_trials)[:, None]
        return self.predict(t_2d, trials_2d)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        return self.kernel.cumulative_integrate(
            t_events=t_events, trial=trial, params=self.params_,
            integration_dt=self.integration_dt,
        )

    def population_survival_curve(
        self, dataset: PointProcessDataset, trial: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model-implied survival curve S(t) = exp(-Lambda(t)) on dataset.t_centers.
        trial=None averages across trials, weighted by n_fish_per_trial (same
        weighting convention as ModelPlotter.plot_model_fits). This is the
        quantity that should be compared to kaplan_meier_curve()'s empirical
        estimate -- NOT compute_expected_rate()'s raw hazard.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")

        t_grid = dataset.t_centers
        if trial is not None:
            Lambda_t = self.kernel.cumulative_integrate(
                t_grid, int(trial), self.params_, self.integration_dt
            )
            return t_grid, np.exp(-Lambda_t)

        S_matrix = np.zeros((dataset.num_trials, len(t_grid)))
        for tr in range(dataset.num_trials):
            Lambda_t = self.kernel.cumulative_integrate(
                t_grid, tr, self.params_, self.integration_dt
            )
            S_matrix[tr] = np.exp(-Lambda_t)

        weights = dataset.n_fish_per_trial
        if weights.sum() == 0:
            return t_grid, np.mean(S_matrix, axis=0)
        return t_grid, np.average(S_matrix, axis=0, weights=weights)

    # -- Model-free empirical reference (Kaplan-Meier) ------------------------

    @staticmethod
    def _kaplan_meier(times: np.ndarray, censored: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generic Kaplan-Meier estimator for right-censored non-negative
        'times'. Used both for calendar-time survival (kaplan_meier_curve)
        and for Cox-Snell residuals (cox_snell_calibration) -- KM doesn't
        care what the time axis represents, only which values are exact
        vs. censored. Standard convention: ties at the same value are
        treated as censoring occurring after events at that value. Returns
        (t_grid, S_hat) with a leading (0, 1.0) point; grid only advances at
        times where >=1 exact event occurred (censoring-only times don't
        change S_hat and are omitted from the returned grid).
        """
        if len(times) == 0:
            return np.array([0.0]), np.array([1.0])

        order = np.argsort(times)
        t_sorted = times[order]
        c_sorted = censored[order]

        n = len(t_sorted)
        grid, surv = [0.0], [1.0]
        S = 1.0
        i = 0
        while i < n:
            t = t_sorted[i]
            j = i
            d = 0
            while j < n and t_sorted[j] == t:
                if not c_sorted[j]:
                    d += 1
                j += 1
            n_at_risk = n - i
            if n_at_risk > 0 and d > 0:
                S *= (1.0 - d / n_at_risk)
                grid.append(float(t))
                surv.append(S)
            i = j

        return np.array(grid), np.array(surv)

    @staticmethod
    def kaplan_meier_curve(dataset: PointProcessDataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model-free, nonparametric estimate of the population survival
        function S(t) = P(no response by time t), pooling every (fish,
        trial) stream as an independent censored first-event observation.
        Empirical reference curve for population_survival_curve().
        """
        times, censored = [], []
        for _, _, t_ev in dataset.iter_streams():
            t_obs, c = SurvivalProcess._first_event_or_censor(t_ev, dataset.duration_s)
            times.append(t_obs)
            censored.append(c)
        if not times:
            return np.array([]), np.array([])
        return SurvivalProcess._kaplan_meier(np.array(times), np.array(censored))

    # -- Calibration diagnostics (Cox-Snell / KM, replaces time_rescaling) ---

    def _stream_tau_values(self, dataset: PointProcessDataset) -> Dict[Tuple[int, int], np.ndarray]:
        raise NotImplementedError(
            "SurvivalProcess streams contain at most one (possibly censored) "
            "event, so the base class's Ogata time-rescaling machinery "
            "(built for recurrent sequences of intervals per stream) does "
            "not apply. Use cox_snell_calibration() instead -- the correct, "
            "right-censoring-aware analogue (Cox-Snell residuals + "
            "Kaplan-Meier) -- which diagnose() uses automatically."
        )

    def cox_snell_calibration(self, dataset: PointProcessDataset) -> Dict[str, Any]:
        """
        Censoring-aware goodness-of-fit check, the survival-analysis
        counterpart to time_rescaling(). Computes the Cox-Snell residual
        r_i = Lambda(t_obs_i) for every stream (exact if that stream had a
        response, right-censored at Lambda(duration_s) otherwise). Under a
        correctly specified model these residuals behave like a censored
        sample from Exp(1) (this IS Ogata's rescaling theorem, applied to a
        process capped at one event -- see class docstring). Calibration is
        checked via a Kaplan-Meier estimate of the residuals' own survival
        function: if -log(S_hat(r)) tracks the y=x line, the model is
        well-calibrated.

        Also returns a per-fish KM-based sup-distance to the Exp(1)
        reference as a rough effect-size analogue of the base class's
        per-fish D_n (NOT a formal test statistic / p-value -- same caveat
        that already applies to the base class's own D_n usage).
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted before running calibration diagnostics.")

        residuals, censored_flags, fish_ids = [], [], []
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            t_obs, censored = self._first_event_or_censor(t_ev, dataset.duration_s)
            H = float(self.kernel.integrate(
                duration_s=t_obs, trial=t_idx, params=self.params_,
                integration_dt=self.integration_dt,
            ))
            residuals.append(H)
            censored_flags.append(censored)
            fish_ids.append(f_idx)

        residuals = np.array(residuals)
        censored_flags = np.array(censored_flags)
        fish_ids = np.array(fish_ids)

        grid_r, S_hat = self._kaplan_meier(residuals, censored_flags)
        neg_log_S = -np.log(np.maximum(S_hat, 1e-12))

        fish_dn = []
        for f_idx in np.unique(fish_ids):
            mask = fish_ids == f_idx
            if mask.sum() < 3:
                continue
            g_r, g_S = self._kaplan_meier(residuals[mask], censored_flags[mask])
            ref = np.exp(-g_r)
            fish_dn.append(float(np.max(np.abs(g_S - ref))))
        fish_dn = np.array(fish_dn)

        return {
            "residuals": residuals,
            "censored": censored_flags,
            "km_grid": grid_r,
            "km_survival": S_hat,
            "neg_log_survival": neg_log_S,
            "n_streams": len(residuals),
            "n_events": int(np.sum(~censored_flags)),
            "fish_dn_stats": fish_dn,
            "median_fish_dn": float(np.median(fish_dn)) if len(fish_dn) else np.nan,
        }

    # -- Dashboard -------------------------------------------------------------

    def diagnose(
        self, dataset: PointProcessDataset, figsize: Tuple[int, int] = (13, 14), eps: float = 1e-5,
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Same (fig, dict) contract as PointProcess.diagnose(), but with
        panels appropriate to censored first-passage data instead of a
        recurrent event stream: no event-lag ACF (meaningless with <=1
        event/stream), Cox-Snell calibration in place of the pooled KS
        plot, and an explicit empirical-vs-model survival curve overlay.
        """
        cs = self.cox_snell_calibration(dataset)
        corr_matrix = self.estimate_parameter_correlation(dataset, eps=eps)
        km_t, km_S = self.kaplan_meier_curve(dataset)
        model_t, model_S = self.population_survival_curve(dataset)

        fig, axes = plt.subplots(3, 2, figsize=figsize)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        fig.suptitle(self.latex_formula, fontsize=15, fontweight='bold', y=0.99)

        # A. Empirical KM vs model survival curve
        ax_km = axes[0, 0]
        if len(km_t) > 0:
            ax_km.step(km_t, km_S, where='post', color='black', linewidth=2, label='Empirical KM $S(t)$')
        ax_km.plot(model_t, model_S, color='crimson', linestyle='--', linewidth=2, label='Model $S(t)$')
        ax_km.set_ylim(0, 1.02)
        ax_km.set_xlabel("Time in trial (s)")
        ax_km.set_ylabel("P(no response by t)")
        ax_km.set_title("A. Survival Function: Empirical vs Model", fontsize=11, fontweight='bold')
        ax_km.legend(loc='upper right', fontsize=9)
        ax_km.grid(True, linestyle=':', alpha=0.4)

        # B. Cox-Snell calibration plot
        ax_cs = axes[0, 1]
        r_grid, neg_log_S = cs["km_grid"], cs["neg_log_survival"]
        if len(r_grid) > 1:
            ax_cs.step(r_grid, neg_log_S, where='post', color='navy', linewidth=2, label='$-\\log(\\hat{S}(r))$')
            max_r = r_grid[-1]
            ax_cs.plot([0, max_r], [0, max_r], 'k--', linewidth=1.5, label='Ideal (Exp(1))')
        else:
            ax_cs.text(0.5, 0.5, "Insufficient events for calibration curve", ha='center', va='center')
        ax_cs.set_xlabel("Cox-Snell residual $r$")
        ax_cs.set_ylabel("$-\\log(\\hat{S}(r))$")
        ax_cs.set_title("B. Cox-Snell Calibration", fontsize=11, fontweight='bold')
        ax_cs.legend(loc='upper left', fontsize=9)
        ax_cs.grid(True, linestyle=':', alpha=0.4)

        # C. Residual histogram (uncensored only) vs Exp(1) reference
        ax_hist = axes[1, 0]
        obs_r = cs["residuals"][~cs["censored"]]
        if len(obs_r) > 0:
            ax_hist.hist(obs_r, bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='none',
                         label='Uncensored residuals')
            x = np.linspace(0, max(obs_r.max(), 1.0), 200)
            ax_hist.plot(x, np.exp(-x), 'r--', linewidth=2, label='Exp(1) density')
        else:
            ax_hist.text(0.5, 0.5, "No uncensored events", ha='center', va='center')
        ax_hist.set_title("C. Residual Histogram (uncensored streams only --\nsee panel B for the censoring-aware check)",
                          fontsize=10, fontweight='bold')
        ax_hist.set_xlabel("Cox-Snell residual $r$")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(loc='upper right', fontsize=9)

        # D. Per-fish D_n effect size
        ax_dn = axes[1, 1]
        dn = cs["fish_dn_stats"]
        if len(dn) > 0:
            ax_dn.hist(dn, bins='auto', density=True, alpha=0.6, color='skyblue', edgecolor='navy',
                      label='Per-fish $D_n$')
            ax_dn.axvline(cs["median_fish_dn"], color='darkblue', linestyle='--', linewidth=2,
                         label=f'Median ({cs["median_fish_dn"]:.3f})')
            ax_dn.axvspan(0.0, 0.05, color='green', alpha=0.1, label='Good fit ($D_n < 0.05$)')
        else:
            ax_dn.text(0.5, 0.5, "Insufficient events per fish for $D_n$", ha='center', va='center')
        ax_dn.set_title(f"D. Per-Fish Calibration $D_n$ (N={len(dn)})", fontsize=11, fontweight='bold')
        ax_dn.set_xlabel("KM sup-distance to Exp(1)")
        ax_dn.legend(loc='upper right', fontsize=8)

        # E. Parameter correlation matrix
        ax_corr = axes[2, 0]
        im_corr = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        n_params = len(self.param_names)
        ax_corr.set_xticks(np.arange(n_params)); ax_corr.set_yticks(np.arange(n_params))
        ax_corr.set_xticklabels(self.param_names, rotation=45, ha='right', fontsize=9)
        ax_corr.set_yticklabels(self.param_names, fontsize=9)
        for i in range(n_params):
            for j in range(n_params):
                val = corr_matrix[i, j]
                ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center',
                            color="white" if abs(val) > 0.6 else "black", fontsize=8)
        ax_corr.set_title("E. Parameter Correlation Matrix", fontsize=11, fontweight='bold')
        divider = make_axes_locatable(ax_corr)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        fig.colorbar(im_corr, cax=cax, label="Correlation")

        # F. Summary
        ax_text = axes[2, 1]
        ax_text.axis('off')
        n_streams = cs["n_streams"]
        n_events = cs["n_events"]
        frac_responded = n_events / n_streams if n_streams > 0 else np.nan
        summary_text = (
            f"DIAGNOSTIC SUMMARY METRICS\n"
            f"----------------------------------------\n"
            f"Log-Likelihood      : {self.log_likelihood:.2f}\n"
            f"Akaike Info (AIC)   : {self.aic:.2f}\n"
            f"N streams           : {n_streams}\n"
            f"N responded         : {n_events}  ({frac_responded:.1%})\n"
            f"Median Per-Fish D_n : {cs['median_fish_dn']:.4f}\n"
        )
        ax_text.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.3))
        ax_text.set_title("F. Global Model Diagnostics", fontsize=11, fontweight='bold')

        return fig, {
            "cox_snell": cs,
            "kaplan_meier_empirical": (km_t, km_S),
            "population_survival_curve": (model_t, model_S),
            "parameter_correlation": corr_matrix,
        }