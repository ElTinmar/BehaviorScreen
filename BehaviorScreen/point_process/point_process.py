from typing import List, Tuple, Dict, Optional, Any, Union
import copy

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.stats import norm, kstest, chi2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .dataset import PointProcessDataset

def _fit_single_bootstrap(seed_seq, dataset: PointProcessDataset, model: "PointProcess"):
    rng = np.random.default_rng(seed_seq)
    ds_boot = dataset.resample(rng)
    model_copy = copy.deepcopy(model)
    try:
        model_copy.fit(ds_boot)
        return model_copy.params_
    except Exception:
        return None

    
class PointProcess:

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

        self.fit_result = res
        self.params_ = res.x
        self.param_dict_ = dict(zip(self.param_names, res.x))
        return self
    
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

    def _fish_scale_factors(self, dataset: PointProcessDataset) -> np.ndarray:
        """
        Per-fish multiplicative correction applied to the population-average
        cumulative intensity before time-rescaling, for models (like
        GammaPoissonProcess) whose predict()/cumulative_integrated_intensity()
        describe only the population-average rate (E[g_f] = 1), not any
        individual fish's true rate.

        Default: no correction (all ones) -- correct for PoissonProcess and
        HawkesProcess, which have no fish-level heterogeneity term.
        """
        return np.ones(dataset.num_fish, dtype=float)
    
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
    
    def time_rescaling(
            self, 
            dataset: PointProcessDataset,
            acf_lags: int = 20,
            min_events_per_fish: int = 5
        ) -> Dict[str, Any]:

        def autocorrelation(z_seqs: List[np.ndarray], max_lags: int) -> Tuple[np.ndarray, np.ndarray, float]:
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

        if self.params_ is None:
            raise ValueError("Model must be fitted before running time-rescaling analysis.")

        fish_scales = self._fish_scale_factors(dataset)

        pooled_u = []
        pooled_z = []
        fish_u = {f_idx: [] for f_idx in range(dataset.num_fish)}

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            if len(t_ev) == 0:
                continue

            Lambda = self.cumulative_integrated_intensity(t_events=t_ev, trial=t_idx)
            Lambda = Lambda * fish_scales[f_idx]  

            tau = np.diff(np.insert(Lambda, 0, 0.0))
            tau = tau[tau > 1e-12]

            u = 1.0 - np.exp(-tau)
            u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
            z = norm.ppf(u_clipped)

            pooled_u.extend(u)
            pooled_z.append(z)
            fish_u[f_idx].extend(u)


        rescaled_u = np.sort(np.array(pooled_u))
        n_pooled = len(rescaled_u)

        lags, acf, conf_limit = autocorrelation(pooled_z, acf_lags)
        
        fish_dn_stats = []
        for f_idx, u_list in fish_u.items():
            if len(u_list) >= min_events_per_fish:
                d_stat, _ = kstest(u_list, 'uniform')
                fish_dn_stats.append(d_stat)

        fish_dn_stats = np.array(fish_dn_stats)

        return {
            "rescaled_u": rescaled_u,
            "n_rescaled": n_pooled,
            "fish_dn_stats": fish_dn_stats,
            "median_fish_dn": float(np.median(fish_dn_stats)) if len(fish_dn_stats) > 0 else np.nan,
            "mean_fish_dn": float(np.mean(fish_dn_stats)) if len(fish_dn_stats) > 0 else np.nan,
            "acf_lags": lags,
            "acf": acf,
            "acf_conf": conf_limit
        }


    def compute_residuals(self, dataset: PointProcessDataset) -> Dict[str, np.ndarray]:
        if self.params_ is None:
            raise ValueError("Model must be fitted before computing residuals.")

        y_obs = dataset.time_trial_histogram_counts
        mu_pred = self.compute_expected_rate(dataset) * dataset.n_fish_per_trial[:, None] * dataset.binning_dt

        pearson_res = (y_obs - mu_pred) / np.sqrt(mu_pred)

        # Deviance Residuals
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(
                y_obs > 0,
                y_obs * np.log(y_obs / mu_pred),
                0.0,
            )
            deviance_sq = 2.0 * (term - (y_obs - mu_pred))
            deviance_res = np.sign(y_obs - mu_pred) * np.sqrt(np.maximum(0.0, deviance_sq))

        return {
            "y_obs": y_obs,
            "mu_pred": mu_pred,
            "pearson_residuals": pearson_res,
            "deviance_residuals": deviance_res,
        }

    def residual_2d_autocorrelation(
        self,
        dataset: PointProcessDataset,
        max_trial_lag: int = 10,
        max_time_lag: int = 30,
    ) -> Dict[str, np.ndarray]:
        """Computes 2D residual autocorrelation across trial and time lag displacements."""
        res_data = self.compute_residuals(dataset)
        residuals = res_data[f"deviance_residuals"]

        # Filter out unobserved trials (n_fish == 0)
        valid_mask = dataset.n_fish_per_trial > 0
        residuals = residuals[valid_mask, :].copy()

        n_trials, n_time = residuals.shape
        residuals -= np.mean(residuals)
        variance = np.mean(residuals ** 2)

        if variance == 0:
            raise ValueError("Residual variance is zero.")

        # Clamp maximum lags to dataset bounds
        max_trial_lag = min(max_trial_lag, max(0, n_trials - 1))
        max_time_lag = min(max_time_lag, max(0, n_time - 1))

        trial_lags = np.arange(-max_trial_lag, max_trial_lag + 1)
        time_lags_bins = np.arange(-max_time_lag, max_time_lag + 1)
        
        acf2d = np.full((len(trial_lags), len(time_lags_bins)), np.nan)

        def _get_overlap_slices(n: int, lag: int) -> Tuple[slice, slice]:
            """Returns safe, matching slice pairs for array overlap under displacement `lag`."""
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

                if x.size > 1:
                    acf2d[i, j] = np.mean(x * y) / variance

        return {
            "trial_lags": trial_lags,
            "time_lags_bins": time_lags_bins,
            "time_lags_sec": time_lags_bins * dataset.binning_dt,
            "acf2d": acf2d,
            "conf_limit": 1.96 / np.sqrt(n_trials * n_time),
        }

    def diagnose(
        self, 
        dataset: PointProcessDataset, 
        figsize: Tuple[int, int] = (15, 18),
        eps: float = 1e-5,
        max_trial_lag: int = 10,
        max_time_lag: int = 30,
    ) -> Dict[str, Any]:

        # 1. Execute sub-analyses
        res_data = self.compute_residuals(dataset)
        tr_data = self.time_rescaling(dataset)
        corr_matrix = self.estimate_parameter_correlation(dataset, eps=eps)
        acf2d_data = self.residual_2d_autocorrelation(
            dataset, max_trial_lag=max_trial_lag, max_time_lag=max_time_lag
        )
        deviance_res = res_data["deviance_residuals"]

        # 2. Render Plot Dashboard (4 rows x 2 columns)
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        plt.subplots_adjust(hspace=0.38, wspace=0.3)

        fig.suptitle(self.latex_formula, fontsize=15, fontweight='bold', y=0.99)

        # Panel A: 2D Residual Surface
        ax_heat = axes[0, 0]
        vmax = np.percentile(np.abs(deviance_res), 98)
        im = ax_heat.imshow(
            deviance_res,
            aspect='auto',
            origin='lower',
            extent=[dataset.t_grid[0], dataset.t_grid[-1], 0, dataset.num_trials],
            cmap='coolwarm',
            vmin=-vmax, vmax=vmax
        )
        ax_heat.set_title("A. Deviance Residual Surface $(m, t)$", fontsize=11, fontweight='bold')
        ax_heat.set_xlabel("Time in Trial (s)")
        ax_heat.set_ylabel("Trial Number")
        divider = make_axes_locatable(ax_heat)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        fig.colorbar(im, cax=cax, label="Deviance Residual")

        # Panel B: Time-Rescaling Kolmogorov-Smirnov Plot (Pooled)
        ax_ks = axes[0, 1]
        n_rescaled = tr_data["n_rescaled"]

        if n_rescaled > 0:
            e_cdf = np.arange(1, n_rescaled + 1) / n_rescaled
            ax_ks.plot(tr_data["rescaled_u"], e_cdf, label="Empirical CDF", color="crimson", lw=2)
            ax_ks.plot([0, 1], [0, 1], 'k--', label="Uniform(0,1) Ideal", lw=1.5)

            ks_bound = 1.36 / np.sqrt(n_rescaled)
            ax_ks.plot([0, 1], [ks_bound, 1 + ks_bound], 'k:', alpha=0.5, label="95% KS Limits")
            ax_ks.plot([0, 1], [-ks_bound, 1 - ks_bound], 'k:', alpha=0.5)

            ax_ks.set_xlim([0, 1])
            ax_ks.set_ylim([0, 1])
        else:
            ax_ks.text(0.5, 0.5, "No events available for KS test", ha='center', va='center')
        ax_ks.set_title("B. Time-Rescaling Kolmogorov-Smirnov Plot", fontsize=11, fontweight='bold')
        ax_ks.set_xlabel("Transformed Interval ($U_k$)")
        ax_ks.set_ylabel("Cumulative Probability")
        ax_ks.legend(loc="upper left", fontsize=8)

        # Panel C: Deviance Residual Distribution
        ax_dist = axes[1, 0]
        flat_dev = deviance_res.flatten()
        ax_dist.hist(flat_dev, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="none")

        x_norm = np.linspace(-4, 4, 200)
        ax_dist.plot(x_norm, norm.pdf(x_norm), 'r--', lw=2, label=r"$\mathcal{N}(0, 1)$ Ref")
        ax_dist.set_title("C. Deviance Residual Distribution", fontsize=11, fontweight='bold')
        ax_dist.set_xlabel("Deviance Residual Value")
        ax_dist.set_ylabel("Density")
        ax_dist.legend(loc="upper right", fontsize=8)

        # Panel D: time rescaled event autocorrelation
        ax_acf = axes[1, 1]
        ax_acf.vlines(tr_data["acf_lags"], 0, tr_data["acf"], color="navy", lw=2)
        ax_acf.axhline(0, color="black", lw=1)

        conf_limit = tr_data["acf_conf"]
        ax_acf.axhline(conf_limit, color="red", linestyle="--", alpha=0.7, label="95% CI")
        ax_acf.axhline(-conf_limit, color="red", linestyle="--", alpha=0.7)

        ax_acf.set_title("D. Time-rescaled event autocorrelation (Event Lag)", fontsize=11, fontweight='bold')
        ax_acf.set_xlabel("Lag (event)")
        ax_acf.set_ylabel("Autocorrelation")
        ax_acf.set_ylim([-0.5, 0.5])
        ax_acf.legend(loc="upper right", fontsize=8)

        # Panel E: 2D Residual Autocorrelation Surface R(Δm, Δt)
        ax_acf2d = axes[2, 0]
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

        im_2d = ax_acf2d.imshow(
            acf2d, extent=extent_2d, origin="lower", cmap="coolwarm",
            vmin=-vmax_2d, vmax=vmax_2d, aspect="auto"
        )
        
        T_mesh, M_mesh = np.meshgrid(time_lags_sec, trial_lags)
        contours = ax_acf2d.contour(
            T_mesh, M_mesh, np.abs(acf2d), levels=[conf_lim_2d],
            colors="black", linewidths=1.0, linestyles="--"
        )
        ax_acf2d.clabel(contours, fmt={conf_lim_2d: f"95% CI"}, inline=True, fontsize=7)
        ax_acf2d.axhline(0, color="gray", lw=0.8, ls=":")
        ax_acf2d.axvline(0, color="gray", lw=0.8, ls=":")

        ax_acf2d.set_title("E. Autocorrelation of deviance residuals $R(\\Delta m, \\Delta t)$", fontsize=11, fontweight='bold')
        ax_acf2d.set_xlabel("Time Lag $\\Delta t$ (s)")
        ax_acf2d.set_ylabel("Trial Lag $\\Delta m$ (trials)")
        
        divider_2d = make_axes_locatable(ax_acf2d)
        cax_2d = divider_2d.append_axes("right", size="3%", pad=0.08)
        fig.colorbar(im_2d, cax=cax_2d, label="Autocorrelation")

        # Panel F: Per-Fish D_n Effect Size Distribution
        ax_dn = axes[2, 1]
        dn_values = tr_data["fish_dn_stats"]
        median_dn = tr_data["median_fish_dn"]

        if len(dn_values) > 0:
            ax_dn.hist(
                dn_values, bins='auto', density=True, alpha=0.6, 
                color='skyblue', edgecolor='navy', label='Per-Fish $D_n$'
            )
            ax_dn.axvline(
                median_dn, color='darkblue', linestyle='--', linewidth=2, 
                label=f'Median $D_n$ ({median_dn:.3f})'
            )
            ax_dn.axvspan(0.0, 0.05, color='green', alpha=0.1, label='Good Fit ($D_n < 0.05$)')
            ax_dn.set_title(f"F. Per-Fish $D_n$ Distribution ($N_{{fish}}={len(dn_values)}$)", fontsize=11, fontweight='bold')
            ax_dn.set_xlabel("KS Distance ($D_n$)")
            ax_dn.set_ylabel("Density")
            ax_dn.legend(loc="upper right", fontsize=8)
        else:
            ax_dn.text(0.5, 0.5, "Insufficient events per fish for $D_n$", ha='center', va='center')

        # Panel G: Parameter Correlation Matrix
        ax_corr = axes[3, 0]
        im_corr = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        ax_corr.set_title("G. Parameter Correlation Matrix", fontsize=11, fontweight='bold')

        n_params = len(self.param_names)

        ax_corr.set_xticks(np.arange(n_params))
        ax_corr.set_yticks(np.arange(n_params))
        ax_corr.set_xticklabels(self.param_names, rotation=45, ha='right', fontsize=9)
        ax_corr.set_yticklabels(self.param_names, fontsize=9)

        for i in range(n_params):
            for j in range(n_params):
                val = corr_matrix[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=8)

        divider_corr = make_axes_locatable(ax_corr)
        cax_corr = divider_corr.append_axes("right", size="3%", pad=0.08)
        fig.colorbar(im_corr, cax=cax_corr, label="Correlation")

        # Panel H: Summary Diagnostic Metrics Text Box
        ax_text = axes[3, 1]
        ax_text.axis('off')
        
        summary_text = (
            f"DIAGNOSTIC SUMMARY METRICS\n"
            f"----------------------------------------\n"
            f"Log-Likelihood      : {self.log_likelihood:.2f}\n"
            f"Akaike Info (AIC)   : {self.aic:.2f}\n"
            f"Pooled KS Rescaled  : N = {n_rescaled}\n"
            f"Median Per-Fish D_n : {tr_data['median_fish_dn']:.4f}\n"
            f"Max 2D Autocorr     : {np.nanmax(np.abs(acf_offdiag)):.4f}\n"
            f"95% 2D CI Limit     : ±{conf_lim_2d:.4f}\n"
        )
        ax_text.text(
            0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.3)
        )
        ax_text.set_title("H. Global Model Diagnostics", fontsize=11, fontweight='bold')

        plt.show()

        return {
            "residuals": res_data,
            "time_rescaling": tr_data,
            "parameter_correlation": corr_matrix,
            "acf2d": acf2d_data,
        }


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
        method: str = 'L-BFGS-B',
        **kwargs
    ) -> Tuple[pd.DataFrame, List[PointProcess]]:
        
        fitted_models = []
        records = []

        for model in models:
            model.fit(dataset, method=method, **kwargs)
            fitted_models.append(model)
            records.append({
                "Model Name": model.name,
                "Params (k)": len(model.param_names),
                "Log-Likelihood": model.log_likelihood,
                "AIC": model.aic,
                "Converged": model.fit_result.success
            })

        df = pd.DataFrame(records)        
        min_aic = df["AIC"].min()
        df["ΔAIC"] = df["AIC"] - min_aic
        weights = np.exp(-0.5 * df["ΔAIC"])
        df["AIC Weight"] = weights / np.sum(weights)

        # Sort BOTH the DataFrame and the list by AIC rank
        sort_idx = df["AIC"].argsort().values
        df = df.iloc[sort_idx].reset_index(drop=True)
        fitted_models = [fitted_models[i] for i in sort_idx]

        return df, fitted_models
    
class ModelPlotter:

    @staticmethod
    def plot_histogram(
        dataset: PointProcessDataset,
        model: PointProcess,
        figsize: Tuple[int, int] = (14, 5),
        cmap: str = "plasma",
    ) -> Tuple[plt.Figure, np.ndarray]:

        fig, (ax_emp, ax_mod) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        model_surface = model.compute_expected_rate(dataset)
        vmax = max(np.max(dataset.time_trial_histogram_hz), np.max(model_surface))

        # Panel 1: Empirical Data
        ax_emp.pcolormesh(
            dataset.t_grid, dataset.trial_edges, dataset.time_trial_histogram_hz, 
            shading='flat', cmap=cmap, vmin=0.0, vmax=vmax
        )
        ax_emp.set_title("Empirical Surface & Raster", fontsize=12, fontweight='bold')
        ax_emp.set_xlabel("Time in Trial (s)", fontsize=11)
        ax_emp.set_ylabel("Trial Number", fontsize=11)
        ax_emp.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])

        # Panel 2: Model Surface
        mesh_mod = ax_mod.pcolormesh(
            dataset.t_grid, dataset.trial_edges, model_surface, 
            shading='flat', cmap=cmap, vmin=0.0, vmax=vmax
        )
        ax_mod.set_title(f"Fitted Surface: {model.name}", fontsize=12, fontweight='bold')
        ax_mod.set_xlabel("Time in Trial (s)", fontsize=11)
        ax_mod.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])

        # Colorbar
        divider = make_axes_locatable(ax_mod)
        cax = divider.append_axes("right", size="3%", pad=0.12)
        cbar = fig.colorbar(mesh_mod, cax=cax)
        cbar.set_label("Event Rate [Hz]", fontsize=10)

        plt.tight_layout()
        return fig, np.array([ax_emp, ax_mod])

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