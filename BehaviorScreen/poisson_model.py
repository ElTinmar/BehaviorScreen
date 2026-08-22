from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
import gc

import joblib
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
from scipy.stats import chi2, norm, kstest
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short


@dataclass(frozen=True)
class PoissonDataset:
    event_times: np.ndarray             
    event_trials_idx: np.ndarray            
    event_fish_idx: np.ndarray          
    fish_trial_mask: np.ndarray

    unique_trials: np.ndarray 
    bout_name: str = ""
    laterality: str = ""
    duration_s: float = 24.0
    binning_dt: float = 0.02

    @property
    def num_fish(self) -> int:
        return self.fish_trial_mask.shape[0]

    @property
    def num_trials(self) -> int:
        return self.fish_trial_mask.shape[1]

    @property
    def t_grid(self) -> np.ndarray:
        return np.arange(0, self.duration_s + self.binning_dt, self.binning_dt)

    @property
    def t_centers(self) -> np.ndarray:
        return 0.5 * (self.t_grid[:-1] + self.t_grid[1:])

    @property
    def trial_edges(self) -> np.ndarray:
        return np.arange(self.num_trials + 1) - 0.5

    @property
    def n_fish_per_trial(self) -> np.ndarray:
        return self.fish_trial_mask.sum(axis=0).astype(float)

    @property
    def time_histogram_counts(self) -> np.ndarray:
        counts, _ = np.histogram(self.event_times, bins=self.t_grid)
        return counts

    @property
    def time_histogram_hz(self) -> np.ndarray:
        total_exposure_time = np.sum(self.n_fish_per_trial) * self.binning_dt
        if total_exposure_time == 0:
            return np.zeros_like(self.time_histogram_counts, dtype=float)
        return self.time_histogram_counts / total_exposure_time

    @property
    def time_trial_histogram_counts(self) -> np.ndarray:
        trial_edges = np.append(self.unique_trials - 0.5, self.unique_trials[-1] + 0.5)
        counts, _, _ = np.histogram2d(
            self.event_trials_idx,
            self.event_times,
            bins=[trial_edges, self.t_grid]
        )
        return counts

    @property
    def time_trial_histogram_hz(self) -> np.ndarray:
        counts = self.time_trial_histogram_counts
        n_fish = self.n_fish_per_trial[:, None]
        safe_denom = np.where(n_fish > 0, n_fish * self.binning_dt, 1.0)
        return np.where(n_fish > 0, counts / safe_denom, 0.0)
    
    def resample(self, rng: np.random.Generator) -> 'PoissonDataset':
        n_fish = self.fish_trial_mask.shape[0]
        boot_fish_idx = rng.choice(n_fish, size=n_fish, replace=True)

        boot_times = []
        boot_trials = []
        boot_fish = []

        for new_id, orig_id in enumerate(boot_fish_idx):
            mask = (self.event_fish_idx == orig_id)
            if np.any(mask):
                boot_times.append(self.event_times[mask])
                boot_trials.append(self.event_trials_idx[mask])
                boot_fish.append(np.full(np.sum(mask), new_id, dtype=int))

        return PoissonDataset(
            event_times=np.concatenate(boot_times) if boot_times else np.array([]),
            event_trials_idx=np.concatenate(boot_trials) if boot_trials else np.array([]),
            event_fish_idx=np.concatenate(boot_fish) if boot_fish else np.array([]),
            fish_trial_mask=self.fish_trial_mask[boot_fish_idx, :],
            duration_s=self.duration_s,
            binning_dt=self.binning_dt,
            bout_name=self.bout_name,
            laterality=self.laterality,
            unique_trials=self.unique_trials
        )


class BehavioralDataLoader:
    def __init__(self, csv_path: Union[Path, str]):
        self.raw_df = pd.read_csv(csv_path)

    def prepare_dataset(
        self,
        bout_name: str,
        laterality: Union[Laterality, str],
        stim: Optional[Union[Stim, str]] = None,
        epoch_name: Optional[Union[str, List[str]]] = None,
        binning_dt: float = 0.02,
        t_start: float = 0.0,
        t_end: float = 24.0,
    ) -> PoissonDataset:

        # 1. Filter sub_df by stimulus or epoch_name
        sub_df = self.raw_df
        if stim is not None:
            sub_df = sub_df[sub_df['stim'] == stim]
            if stim == Stim.PHOTOTAXIS or stim == 'phototaxis':
                sub_df = sub_df[sub_df['foreground_color'] == '[0.1, 0.1, 0.0, 1.0]']
        elif epoch_name is not None:
            if isinstance(epoch_name, list):
                sub_df = sub_df[sub_df['epoch_name'].isin(epoch_name)]
            else:
                sub_df = sub_df[sub_df['epoch_name'] == epoch_name]
        else:
            raise ValueError("Either 'stim' or 'epoch_name' must be provided to filter dataset.")

        # 2. Extract metadata & build integer index mappings
        all_fish_ids = np.sort(sub_df['file'].unique())
        unique_trials = np.sort(sub_df['trial_num'].unique()) 

        fish_map = {f_id: idx for idx, f_id in enumerate(all_fish_ids)}
        trial_map = {t_num: idx for idx, t_num in enumerate(unique_trials)}

        # 3. Build (N_fish, N_trials) boolean presence matrix
        n_fish = len(all_fish_ids)
        n_trials = len(unique_trials)
        fish_trial_mask = np.zeros((n_fish, n_trials), dtype=bool)

        active_pairs = sub_df[['file', 'trial_num']].drop_duplicates()
        f_indices = active_pairs['file'].map(fish_map).values
        t_indices = active_pairs['trial_num'].map(trial_map).values
        fish_trial_mask[f_indices, t_indices] = True

        # 4. Filter target events
        bout_idx = bouts_category_name_short.index(bout_name)
        is_target_event = (
            (sub_df['category'] == bout_idx) & 
            (sub_df['laterality'] == laterality)
        )
        event_mask = (
            is_target_event & 
            (sub_df['trial_time'] >= t_start) & 
            (sub_df['trial_time'] < t_end)
        )
        events = sub_df[event_mask]

        # 5. Extract event arrays as integer indices & floats
        event_times = (events['trial_time'] - t_start).values.astype(float)
        event_trials_idx = events['trial_num'].map(trial_map).values.astype(int)
        event_fish_idx = events['file'].map(fish_map).values.astype(int)

        return PoissonDataset(
            event_times=event_times,
            event_trials_idx=event_trials_idx,
            event_fish_idx=event_fish_idx,
            fish_trial_mask=fish_trial_mask,
            duration_s=t_end-t_start,
            binning_dt=binning_dt,
            bout_name=bout_name,
            laterality=str(laterality),
            unique_trials=unique_trials,
        )

@dataclass(frozen=True)
class RateKernel:
    name: str
    func: Callable[[np.ndarray, np.ndarray, List[float]], np.ndarray]
    param_names: List[str]
    initial_guesses: List[float]
    bounds: List[Tuple[Optional[float], Optional[float]]]
    latex_formula: str = ""
    integral_func: Optional[Callable[[float, float, np.ndarray, List[float]], np.ndarray]] = None

    def evaluate(self, t: np.ndarray, trial: np.ndarray, params: List[float]) -> np.ndarray:
        rate = self.func(t, trial, params)            
        target_shape = np.broadcast(t, trial).shape
        return np.broadcast_to(rate, target_shape)

    def integrate(
            self, 
            duration_s: float, 
            trial: np.ndarray, 
            params: List[float], 
            integration_dt: float = 0.02
        ) -> np.ndarray:
            
            if self.integral_func is not None:
                return self.integral_func(0, duration_s, trial, params)

            t_grid = np.arange(0, duration_s + integration_dt, integration_dt)
            t_2d = t_grid[None, :]                     # Shape: (1, N_time)
            trials_2d = np.atleast_1d(trial)[:, None]  # Shape: (N_trials, 1)

            rate_surface = self.evaluate(t_2d, trials_2d, params)
            rate_surface = np.maximum(rate_surface, 1e-9) # TODO find a way to get rid of that

            integrals = trapezoid(rate_surface, dx=integration_dt, axis=1)

            # Preserve scalar input shape if a scalar trial was passed
            return integrals if np.iterable(trial) else integrals[0]

    def cumulative_integrate(
        self, 
        t_events: np.ndarray, 
        trial: Union[float, int], 
        params: List[float], 
        integration_dt: float = 0.02
    ) -> np.ndarray:
        """
        Computes cumulative integrated intensity Lambda(t_k) at specific event timestamps.
        Used primarily for Time-Rescaling diagnostic transforms.
        """
        if len(t_events) == 0:
            return np.array([], dtype=float)

        # 1. Analytical route: use numpy vectorization
        if self.integral_func is not None:
            return self.integral_func(0, t_events, trial, params)

        # 2. Numerical fallback using cumulative trapezoid
        t_max = np.max(t_events)
        if t_max <= 0:
            return np.zeros_like(t_events, dtype=float)

        t_grid = np.arange(0, t_max + integration_dt, integration_dt)
        trials_2d = np.atleast_1d(trial)[:, None]
        rate_surface = self.evaluate(t_grid[None, :], trials_2d, params)
        rate_surface = np.maximum(rate_surface, 1e-9) # TODO find a way to get rid of that

        # Compute continuous cumulative surface & interpolate exact event timestamps
        cum_integral = cumulative_trapezoid(rate_surface, t_grid, initial=0.0, axis=1).squeeze()
        return np.interp(t_events, t_grid, cum_integral)

    def is_nested_in(self, parent_kernel: "RateKernel") -> bool:
        return set(parent_kernel.param_names).issubset(set(self.param_names))


def _fit_single_bootstrap(seed_seq, dataset, kernel):
    rng = np.random.default_rng(seed_seq)
    ds_boot = dataset.resample(rng)
    boot_model = PoissonProcess(kernel)
    try:
        boot_model.fit(ds_boot)
        return boot_model.params_
    except Exception:
        return None
    
class PoissonProcess:
    """
    Maximum Likelihood Estimator for Homogeneous and Inhomogeneous Poisson Processes.
    """
    def __init__(self, kernel: RateKernel, integration_dt: float = 0.02):
        self.kernel = kernel
        self.integration_dt = integration_dt
        self.fit_result: Optional[Any] = None
        self.params_: Optional[np.ndarray] = None
        self.param_dict_: Dict[str, float] = {}
        self.initial_guesses = kernel.initial_guesses
        self.bounds = kernel.bounds

    def _nll(
        self, 
        params: List[float], 
        dataset: PoissonDataset
    ) -> float:

        # Term 1: Sum of Log Intensity at Observed Events
        event_rates = self.kernel.evaluate(dataset.event_times, dataset.event_trials_idx, params)
        event_rates = np.maximum(event_rates, 1e-9)
        sum_log_rates = np.sum(np.log(event_rates))

        # Term 2: Expected Total Events (Surface Integration over Time)
        trial_integrals = self.kernel.integrate(
            duration_s=dataset.duration_s, 
            trial=dataset.unique_trials, 
            params=params, 
            integration_dt=self.integration_dt
        )
        
        # Scale expected events by trial-specific observing fish count
        total_expected_events = np.sum(trial_integrals * dataset.n_fish_per_trial)

        return -(sum_log_rates - total_expected_events)

    def fit(
        self, 
        dataset: PoissonDataset, 
        method: str = 'L-BFGS-B', 
        **kwargs
    ):
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
        self.param_dict_ = dict(zip(self.kernel.param_names, res.x))
        return self

    def estimate_hessian(
        self, 
        dataset: PoissonDataset, 
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Estimates the Observed Information Matrix (Hessian of NLL at MLE) via central finite differences."""
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
        dataset: PoissonDataset, 
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Computes normalized parameter correlation matrix (-1 to 1)."""
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

    def diagnose(
        self, 
        dataset: PoissonDataset, 
        figsize: Tuple[int, int] = (15, 18),
        eps: float = 1e-5,
        max_trial_lag: int = 10,
        max_time_lag: int = 30,
    ) -> Dict[str, Any]:

        # 1. Execute sub-analyses
        res_data = self.binned_residuals(dataset)
        tr_data = self.time_rescaling(dataset)
        corr_matrix = self.estimate_parameter_correlation(dataset, eps=eps)
        acf2d_data = self.residual_2d_autocorrelation(
            dataset, max_trial_lag=max_trial_lag, max_time_lag=max_time_lag
        )
        deviance_res = res_data["deviance_residuals"]

        # 2. Render Plot Dashboard (4 rows x 2 columns)
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        plt.subplots_adjust(hspace=0.38, wspace=0.3)

        model_formula = self.kernel.latex_formula 
        fig.suptitle(model_formula, fontsize=15, fontweight='bold', y=0.99)

        # Panel A: 2D Residual Surface
        ax_heat = axes[0, 0]
        vmax = np.percentile(np.abs(deviance_res), 98)
        im = ax_heat.imshow(
            deviance_res,
            aspect='auto',
            origin='lower',
            extent=[dataset.t_grid[0], dataset.t_grid[-1], dataset.unique_trials[0], dataset.unique_trials[-1]],
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

        param_names = self.kernel.param_names
        n_params = len(param_names)

        ax_corr.set_xticks(np.arange(n_params))
        ax_corr.set_yticks(np.arange(n_params))
        ax_corr.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
        ax_corr.set_yticklabels(param_names, fontsize=9)

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

    def bootstrap(
        self,
        dataset: PoissonDataset,
        n_boot: int = 500,
        seed: int = 42,
        ci: float = 95.0,
        n_jobs: int = -1,
    ) -> pd.DataFrame:

        if self.params_ is None:
            raise ValueError("Model must be fitted before running bootstrap.")

        seeds = np.random.SeedSequence(seed).spawn(n_boot)

        boot_results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_fit_single_bootstrap)(s, dataset, self.kernel)
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
                for i, name in enumerate(self.kernel.param_names)
            ]
        )

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray]) -> np.ndarray:
        """Predicts expected rate intensity lambda(t, trial) per single trial observation."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        expected_rate = self.kernel.evaluate(t, trial, self.params_)
        expected_rate = np.maximum(expected_rate, 1e-9)
        return expected_rate

    @property
    def log_likelihood(self) -> float:
        return -self.fit_result.fun if self.fit_result else np.nan

    @property
    def aic(self) -> float:
        k = len(self.params_)
        return 2 * k - 2 * self.log_likelihood

    def binned_residuals(self, dataset: PoissonDataset) -> Dict[str, np.ndarray]:
        """Calculates 2D Pearson and Deviance residual matrices (Trial x Time)."""
        if self.params_ is None:
            raise ValueError("Model must be fitted before computing residuals.")

        y_obs = dataset.time_trial_histogram_counts
        t_2d = dataset.t_centers[None, :]
        trials_2d = dataset.unique_trials[:, None]

        rate_surface = self.predict(t_2d, trials_2d)
        mu_pred = rate_surface * dataset.n_fish_per_trial[:, None] * dataset.binning_dt

        pearson_res = (y_obs - mu_pred) / np.sqrt(np.maximum(mu_pred, 1e-9))

        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(y_obs > 0, y_obs * np.log(y_obs / np.maximum(mu_pred, 1e-9)), 0.0)
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
        dataset: PoissonDataset,
        max_trial_lag: int = 10,
        max_time_lag: int = 30,
    ) -> Dict[str, np.ndarray]:
        """Computes 2D residual autocorrelation across trial and time lag displacements."""
        res_data = self.binned_residuals(dataset)
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

    def cumulative_integrated_intensity(
        self,
        t_events,
        trial,
    ):
        return self.kernel.cumulative_integrate(
            t_events=t_events,
            trial=trial,
            params=self.params_,
            integration_dt=self.integration_dt,
        )

    def time_rescaling(
            self, 
            dataset: PoissonDataset,
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

        pooled_u = []
        pooled_z = []
        fish_u = {f_idx: [] for f_idx in range(dataset.num_fish)}

        for idx, m in enumerate(dataset.unique_trials):
            trial_mask = (dataset.event_trials_idx == m)
            if not np.any(trial_mask):
                continue

            active_fish = np.where(dataset.fish_trial_mask[:, idx])[0]

            for f_idx in active_fish:
                fish_mask = trial_mask & (dataset.event_fish_idx == f_idx)
                t_ev = np.sort(dataset.event_times[fish_mask])

                if len(t_ev) == 0:
                    continue

                Lambda = self.cumulative_integrated_intensity(
                    t_events=t_ev,
                    trial=float(m),
                )

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

class HawkesProcess(PoissonProcess):
    """
    Inhomogeneous Hawkes Process with exponential self-excitation memory kernel.
    Extends baseline rate kernel lambda_0(t, m) with parameters [alpha, beta].
    """

    def __init__(
        self,
        kernel,
        alpha_initial=0.1,
        beta_initial=10.0,
        alpha_bounds=(0.0, None),
        beta_bounds=(1e-3, None),
        integration_dt=0.02,
    ):
        super().__init__(kernel, integration_dt)
        self.initial_guesses += [alpha_initial, beta_initial]
        self.bounds += [alpha_bounds, beta_bounds]

    def _hawkes_nll_single_stream(
        self, 
        t_events: np.ndarray, 
        trial_idx: float, 
        duration_s: float, 
        params_base: List[float], 
        alpha: float, 
        beta: float
    ) -> float:
        n_events = len(t_events)
        
        # 1. Base rate integration integral
        base_integral = self.kernel.integrate(duration_s, trial_idx, params_base)
        
        if n_events == 0:
            return base_integral  # No history terms if no events occurred

        # 2. Recursive computation of R_i for log-intensity term: log(lambda_0(t_i) + alpha * R_i)
        t_sorted = np.sort(t_events)
        dt_seq = np.diff(t_sorted)
        
        R = np.zeros(n_events)
        for i in range(1, n_events):
            R[i] = np.exp(-beta * dt_seq[i-1]) * (1.0 + R[i-1])

        base_rates = self.kernel.evaluate(t_sorted, trial_idx, params_base)
        intensity_at_events = np.maximum(base_rates + alpha * R, 1e-9)
        
        sum_log_intensity = np.sum(np.log(intensity_at_events))

        # 3. Exact analytical integral of Hawkes memory term over [0, T]
        hawkes_integral = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (duration_s - t_sorted)))

        return -(sum_log_intensity - (base_integral + hawkes_integral))

    def _nll(self, params: List[float], dataset: PoissonDataset) -> float:
        *params_base, alpha, beta = params
        total_nll = 0.0

        # Group operations per unique fish/trial stream
        for f_idx in range(dataset.num_fish):
            for t_idx, m in enumerate(dataset.unique_trials):
                if not dataset.fish_trial_mask[f_idx, t_idx]:
                    continue

                mask = (dataset.event_fish_idx == f_idx) & (dataset.event_trials_idx == t_idx)
                t_ev = dataset.event_times[mask]

                total_nll += self._hawkes_nll_single_stream(
                    t_ev, m, dataset.duration_s, params_base, alpha, beta
                )

        return total_nll

    def cumulative_integrated_intensity(
        self,
        t_events: np.ndarray,
        trial: float,
    ) -> np.ndarray:

        if self.params_ is None:
            raise ValueError("Model must be fitted first.")

        *base_params, alpha, beta = self.params_

        t_events = np.sort(np.asarray(t_events))

        # Baseline cumulative intensity
        Lambda_base = self.kernel.cumulative_integrate(
            t_events=t_events,
            trial=trial,
            params=base_params,
            integration_dt=self.integration_dt,
        )

        # Hawkes history contribution
        history = np.zeros_like(t_events)

        for i, t in enumerate(t_events):
            previous_events = t_events[:i]

            history[i] = np.sum(
                1.0 - np.exp(-beta * (t - previous_events))
            )

        Lambda_hawkes = (alpha / beta) * history

        return Lambda_base + Lambda_hawkes
        
class KernelFactory:

    @staticmethod
    def homogeneous_poisson() -> RateKernel:
        def _func(t, trial, params):
            B = params[0]
            return B * np.ones_like(t + 0.0 * trial)

        return RateKernel(
            name="Homogeneous Poisson λ()",
            func=_func,
            param_names=["B"],
            initial_guesses=[0.5],
            bounds=[(0.001, 20.0)],
            latex_formula=r"$\lambda = B$",
        )

    # TODO make positive
    @staticmethod
    def prey_capture(stim_freq: float, plasticity: Optional[str] = None) -> RateKernel:
        key = (plasticity or "").lower().replace(" ", "")

        W_latex = r"\frac{1}{2}\left[\sin(\omega t + \phi_1) + \sin(2\omega t + \phi_2)\right]"

        PRESETS = {
            "": (
                "TimeOnly", [], lambda p: (0.0, 0.0, 0.0),
                rf"$\lambda(t, m) = \left(A e^{{-t/\tau}} + B\right) \left(1 + A_{{\text{{ripple}}}} {W_latex}\right)$",
            ),
            "shared": (
                "Shared", ["alpha_shared"], lambda p: (p[0], p[0], p[0]),
                rf"$\lambda(t, m) = \left(A e^{{-t/\tau}} + B\right) \left(1 + A_{{\text{{ripple}}}} {W_latex}\right) e^{{\alpha m}}$",
            ),
            "rate_shared": (
                "RateShared", ["alpha_rate"], lambda p: (p[0], p[0], 0.0),
                rf"$\lambda(t, m) = \left(A e^{{-t/\tau}} + B\right) e^{{\alpha_{{\text{{rate}}}} m}} \left(1 + A_{{\text{{ripple}}}} {W_latex}\right)$",
            ),
            "rate_shared,gamma": (
                "RateShared_Gamma", ["alpha_rate", "alpha_gamma"], lambda p: (p[0], p[0], p[1]),
                rf"$\lambda(t, m) = \left(A e^{{-t/\tau}} + B\right) e^{{\alpha_{{\text{{rate}}}} m}} \left(1 + A_{{\text{{ripple}}}} e^{{\alpha_\gamma m}} {W_latex}\right)$",
            ),
        }

        if key in PRESETS:
            suffix, alpha_names, get_alphas, latex = PRESETS[key]
        else:
            active_terms = [t for t in ["a", "b", "gamma"] if t in key.split(",")]
            alpha_names = [f"alpha_{'gamma' if t == 'gamma' else t.upper()}" for t in active_terms]
            suffix = "_".join(n.replace("alpha_", "") for n in alpha_names) or "TimeOnly"

            def get_alphas(p, names=alpha_names):
                p_map = dict(zip(names, p))
                return p_map.get("alpha_A", 0.0), p_map.get("alpha_B", 0.0), p_map.get("alpha_gamma", 0.0)

            t_A = r"A e^{-t/\tau}" + (r" e^{\alpha_A m}" if "alpha_A" in alpha_names else "")
            t_B = r"B" + (r" e^{\alpha_B m}" if "alpha_B" in alpha_names else "")
            t_g = r"A_{\text{ripple}}" + (r" e^{\alpha_\gamma m}" if "alpha_gamma" in alpha_names else "")
            latex = rf"$\lambda(t, m) = \left({t_A} + {t_B}\right) \left(1 + {t_g} {W_latex}\right)$"

        names = ["A", "tau", "B", "A_ripple", "phi1", "phi2"] + alpha_names
        n_alphas = len(alpha_names)
        guesses = [0.56, 1.15, 0.40, 0.1, 0.0, 0.0] + [-0.05] * n_alphas
        bounds = [(0.01, 10.0), (0.1, 5.0), (0.01, 5.0), (0.0, 0.99), (-np.pi, np.pi), (-np.pi, np.pi)] + [(-2.0, 2.0)] * n_alphas

        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2 = params[:6]
            a_A, a_B, a_g = get_alphas(params[6:])

            w = 2.0 * np.pi * stim_freq
            phase = w * t

            transient = A * np.exp(-t / tau) * np.exp(a_A * trial)
            baseline = B * np.exp(a_B * trial)
            
            wave = 0.5 * (np.sin(phase + phi1) + np.sin(2.0 * phase + phi2))
            ripple_mod = 1.0 + A_ripple * wave * np.exp(a_g * trial)

            return (transient + baseline) * ripple_mod

        return RateKernel(
            name=f"PreyCapture({suffix})",
            func=_func,
            param_names=names,
            initial_guesses=guesses,
            bounds=bounds,
            latex_formula=latex,
        )

    # TODO make positive
    @staticmethod
    def phototaxis_ipsi() -> RateKernel:
        def _func(t, trial, params):
            B, f_dip, A_peak, tau, alpha_B, alpha_peak = params

            mod_B = B * np.exp(alpha_B * trial)
            mod_peak = A_peak * np.exp(alpha_peak * trial)
            A_dip = f_dip * mod_B  

            transient = (mod_peak * (t / tau) - A_dip) * np.exp(-t / tau)

            return mod_B + transient

        return RateKernel(
            name="Phototaxis Minimal Dip+Peak λ(t, m)",
            func=_func,
            param_names=["B", "f_dip", "A_peak", "tau", "alpha_B", "alpha_peak"],
            initial_guesses=[0.4, 0.5, 1.5, 0.2, 0.0, 0.0],
            bounds=[
                (0.01, 5.0),   # B
                (0.0, 0.99),   # f_dip in [0, 0.99) guarantees positivity at t=0
                (0.0, 10.0),   # A_peak
                (0.01, 2.0),   # tau
                (-0.2, 0.2),   # alpha_B
                (-0.2, 0.2)    # alpha_peak
            ],
            latex_formula=r"$\lambda(t, m) = B e^{\alpha_B m} + \left(A_{\text{peak}} e^{\alpha_{\text{peak}} m} \frac{t}{\tau} - f_{\text{dip}} B e^{\alpha_B m}\right) e^{-t/\tau}$"
        )

    # TODO make positive
    @staticmethod
    def phototaxis_contra() -> RateKernel:
        def _func(t, trial, params):
            B, A_dip, tau_dip, alpha_B, alpha_dip = params

            mod_B = B * np.exp(alpha_B * trial)
            mod_dip = A_dip * np.exp(alpha_dip * trial) * np.exp(-t / tau_dip)

            return mod_B - mod_dip

        return RateKernel(
            name="Phototaxis Contra λ(t, m)",
            func=_func,
            param_names=["B", "A_dip", "tau_dip", "alpha_B", "alpha_dip"],
            initial_guesses=[0.4, 0.3, 0.5, 0.0, 0.0],
            bounds=[
                (0.01, 10.0), (0.0, 5.0), (0.01, 5.0), (-0.1, 0.1),
                (-0.1, 0.1)
            ],
            latex_formula=r"$\lambda(t, m) = B e^{\alpha_B m} - A_{\text{dip}} e^{\alpha_{\text{dip}} m} e^{-t/\tau_{\text{dip}}}$",
        )
    
    @staticmethod
    def omr_forward() -> RateKernel:
        def _func(t, trial, params):
            B, f_dip, tau_dip = params
            return B * (1.0 - f_dip * np.exp(-t / tau_dip))

        return RateKernel(
            name="OMR forward λ(t)",
            func=_func,
            param_names=["B", "f_dip", "tau_dip"],
            initial_guesses=[0.4, 0.5, 0.5],
            bounds=[
                (0.01, 5.0), 
                (0.0, 0.99), 
                (0.01, 5.0)
            ],
            latex_formula=r"$\lambda(t) = B \left(1 - f_{\text{dip}} e^{-t/\tau_{\text{dip}}}\right)$",
        )

    @staticmethod
    def omr_lateral_contra() -> RateKernel:
        def _func(t, trial, params):
            B, f_dip, tau_dip = params
            return B * (1.0 - f_dip * np.exp(-t / tau_dip))

        return RateKernel(
            name="OMR lateral contra λ(t)",
            func=_func,
            param_names=["B", "f_dip", "tau_dip"],
            initial_guesses=[0.4, 0.5, 0.5],
            bounds=[
                (0.01, 5.0), 
                (0.0, 0.99), 
                (0.01, 5.0)
            ],
            latex_formula=r"$\lambda(t) = B \left(1 - f_{\text{dip}} e^{-t/\tau_{\text{dip}}}\right)$",
        )
            
    @staticmethod
    def looming_gaussian(t_critical: float = 5.0) -> RateKernel:
        def _func(t, trial, params):
            B, H, alpha, mu, sigma = params

            height = H * np.exp(alpha * trial)
            exponent = -0.5 * ((t - mu) / sigma)**2
            return B + (height * np.exp(exponent))

        return RateKernel(
            name="Looming Gaussian λ(t, m)",
            func=_func,
            param_names=["B", "H", "alpha", "mu", "sigma"],
            initial_guesses=[0.1, 1.2, 0.0, 5.0, 0.1],
            bounds=[
                (0.001, 10.0), (0.001, 10.0), (-2.0, 2.0), 
                (t_critical-1.5, t_critical+1), (0.001, 3.0)
            ],
            latex_formula=r"$\lambda(t, m) = B + H e^{\alpha m} \exp\left(-\frac{(t - \mu)^2}{2\sigma^2}\right)$",
        )

    # TODO make positive
    @staticmethod
    def dark_flash() -> RateKernel:
        def _func(t, trial, params):
            A, k, tau, B, alpha_1, alpha_2, alpha_B = params
            x = t / tau

            kernel = np.power(x / k, k) * np.exp(k - x)
            rational_scale = (1.0 + alpha_1 * trial) / (1.0 + alpha_2 * trial**2)
            mod_A = A * rational_scale
            mod_B = B * np.exp(alpha_B * trial)

            return mod_A * kernel + mod_B

        return RateKernel(
            name="Dark Flash λ(t, m)",
            func=_func,
            param_names=["A", "k", "tau", "B", "alpha_1", "alpha_2", "alpha_B"],
            initial_guesses=[10.0, 2.0, 0.25, 0.2, 1.5, 0.5, 0.0],
            bounds=[
                (0.0, None), (0.1, 10.0), (0.01, None), (0.0, None),
                (0.0, 10.0), (0.0, 5.0), (-0.2, 0.2)
            ],
            latex_formula=r"$\lambda(t, m) = A \frac{1 + \alpha_1 m}{1 + \alpha_2 m^2} \left(\frac{t}{k\tau}\right)^k e^{k - t/\tau} + B e^{\alpha_B m}$",
        )


class ModelComparator:

    @staticmethod
    def likelihood_ratio_test(
        model_null: PoissonProcess, 
        model_alt: PoissonProcess
    ) -> Dict[str, Union[str, int, float, bool]]:


        is_nested = model_alt.kernel.is_nested_in(model_null.kernel)
        if not is_nested:
            raise ValueError(f"Models are not nested.")

        k_null = len(model_null.params_)
        k_alt = len(model_alt.params_)
        df = k_alt - k_null

        ll_null = model_null.log_likelihood
        ll_alt = model_alt.log_likelihood
        lr_stat = 2.0 * (ll_alt - ll_null)
        lr_stat_clamped = max(0.0, lr_stat)
        
        p_val = float(chi2.sf(lr_stat_clamped, df))

        return {
            "Null Model": model_null.kernel.name,
            "Alt Model": model_alt.kernel.name,
            "LL Null": ll_null,
            "LL Alt": ll_alt,
            "Deviance (2*ΔLL)": lr_stat,
            "Δk (df)": df,
            "p-value": p_val,
            "Significant (α=0.05)": p_val < 0.05
        }


    @staticmethod
    def compare(
        kernels: List[RateKernel], 
        dataset: PoissonDataset,
        method: str = 'L-BFGS-B',
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, PoissonProcess]]:
        """Fits and benchmarks multiple RateKernels on a dataset or raw arrays."""
        models = {}
        records = []

        for kernel in kernels:
            model = PoissonProcess(kernel)
            model.fit(dataset, method=method, **kwargs)
            models[kernel.name] = model

            records.append({
                "Model Name": kernel.name,
                "Params (k)": len(kernel.param_names),
                "Log-Likelihood": model.log_likelihood,
                "AIC": model.aic,
                "Converged": model.fit_result.success
            })

        df = pd.DataFrame(records)
        
        # Calculate Delta AIC & Akaike Weights
        min_aic = df["AIC"].min()
        df["ΔAIC"] = df["AIC"] - min_aic
        weights = np.exp(-0.5 * df["ΔAIC"])
        df["AIC Weight"] = weights / np.sum(weights)

        df = df.sort_values(by="AIC").reset_index(drop=True)
        return df, models


class PoissonVisualizer:

    @staticmethod
    def plot_histogram(
        dataset: PoissonDataset,
        model: PoissonProcess,
        figsize: Tuple[int, int] = (14, 5),
        cmap: str = "plasma",
    ) -> Tuple[plt.Figure, np.ndarray]:

        fig, (ax_emp, ax_mod) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # 1. Evaluate Model Surface using dataset grid
        t_2d = dataset.t_centers[None, :]
        trials_2d = dataset.unique_trials[:, None]
        model_surface = model.predict(t_2d, trials_2d)

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
        kernel_name = getattr(getattr(model, 'kernel', None), 'name', 'Poisson Model')
        ax_mod.set_title(f"Fitted Surface: {kernel_name}", fontsize=12, fontweight='bold')
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
        dataset: PoissonDataset,
        models: Dict[str, PoissonProcess],
        figsize: Tuple[int, int] = (12, 6),
        palette: Optional[Dict[str, Any]] = None
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
        t_2d = dataset.t_centers[None, :]
        trials_2d = dataset.unique_trials[:, None]
        default_colors = plt.cm.tab10.colors

        for idx, (name, model) in enumerate(models.items()):
            pred_surface = model.predict(t_2d, trials_2d)
            mean_pred = np.average(pred_surface, axis=0, weights=dataset.n_fish_per_trial)

            color = palette.get(name) if palette else default_colors[idx % len(default_colors)]
            
            # Extract LaTeX formula from model kernel if available
            latex_formula = getattr(getattr(model, 'kernel', None), 'latex_formula', None)
            if latex_formula:
                label = f"{name}:  {latex_formula}"
            else:
                label = f"Model: {name}"

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
        dataset: PoissonDataset,
        model: PoissonProcess,
        trial_step: int = 2,
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = "viridis",
        marker_size: float = 28.0,
        data_alpha: float = 0.35,
        model_alpha: float = 1.0
    ) -> Tuple[plt.Figure, plt.Axes]:

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Evaluate Model Surface: Shape (N_trials, N_time_bins)
        t_2d = dataset.t_centers[None, :]
        trials_2d = dataset.unique_trials[:, None]
        model_surface = model.predict(t_2d, trials_2d)

        # 2. Color Mapping Setup
        norm = plt.Normalize(vmin=dataset.unique_trials[0], vmax=dataset.unique_trials[-1])
        base_cmap = plt.get_cmap(cmap)

        # 3. Plot Selected Trials
        selected_indices = range(0, len(dataset.unique_trials), trial_step)

        for idx in selected_indices:
            trial_idx = dataset.unique_trials[idx]
            color = base_cmap(norm(trial_idx))

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
        kernel_name = getattr(getattr(model, 'kernel', None), 'name', 'Poisson Model')
        ax.set_title(
            f"Trial-by-Trial Overlay | {kernel_name} (Every {trial_step} Trials)", 
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

if __name__ == '__main__':

    possible_roots = [
        Path('/home/martin/Desktop/DATA'),
        Path('/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen'),
        Path('/media/martin/DATA_18TB/Screen'),
    ]
    ROOT = next((p for p in possible_roots if p.exists()), possible_roots[0])

    # Prey capture stim parameters
    prey_stim_speed_deg_per_s = 90
    prey_stim_range_deg = 2 * 70
    prey_stim_freq = prey_stim_speed_deg_per_s / prey_stim_range_deg

    # 1. Load Data
    loader = BehavioralDataLoader(ROOT / 'bouts_control.csv')

    model_config = {

        'prey_capture_ipsi': {
            'dataset': {
                'stim':Stim.PREY_CAPTURE,
                'bout_name':'JT',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity=None),
                KernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="A"),
                KernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="A,B"),
                KernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="rate_shared,gamma"),
                KernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="shared"),
                KernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="A,B,gamma"),
            ]
        },

        'prey_capture_contra': {
            'dataset': {
                'stim':Stim.PREY_CAPTURE,
                'bout_name':'JT',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
            ]
        },

        'phototaxis_ipsi': {
            'dataset': {
                'stim':Stim.PHOTOTAXIS,
                'bout_name':'RT',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.phototaxis_ipsi(),
            ]
        },

        'phototaxis_contra': {
            'dataset': {
                'stim':Stim.PHOTOTAXIS,
                'bout_name':'RT',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.phototaxis_contra()
            ]
        },

        'omr_lateral_ipsi': {
            'dataset': {
                'epoch_name':["grating right", "grating left"],
                'bout_name':'RT',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson()
            ]
        },

        'omr_lateral_contra': {
            'dataset': {
                'epoch_name':["grating right", "grating left"],
                'bout_name':'RT',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.omr_lateral_contra()
            ]
        },

        'omr_forward': {
            'dataset': {
                'epoch_name':"grating forward",
                'bout_name':'BS',
                'laterality':Laterality.NONDIRECTIONAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.omr_forward()
            ]
        },

        'okr_ipsi': {
            'dataset': {
                'stim':Stim.OKR,
                'bout_name':'S1',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson()
            ]
        },

        'okr_contra': {
            'dataset': {
                'stim':Stim.OKR,
                'bout_name':'S1',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson()
            ]
        },

        'looming_ipsi': {
            'dataset': {
                'stim':Stim.LOOMING,
                'bout_name':'SLC',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.looming_gaussian()
            ]
        },

        'looming_contra': {
            'dataset': {
                'stim':Stim.LOOMING,
                'bout_name':'SLC',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.looming_gaussian()
            ]
        },

        'dark_flash': {
            'dataset': {
                'epoch_name':"flash dark",
                'bout_name':'O',
                'laterality':Laterality.NONDIRECTIONAL,
                'binning_dt':0.025, 
                't_start':0.0, 
                't_end':5.0, 
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.dark_flash()
            ]
        },
    }

    all_summaries = []

    for exp_name, config in model_config.items():
        
        print(f"\n==================================================")
        print(f" PROCESSING EXPERIMENT: {exp_name.upper()}")
        print(f"==================================================")

        dataset = loader.prepare_dataset(**config['dataset'])

        summary_table, fitted_models = ModelComparator.compare(
            kernels=config['kernels'],
            dataset=dataset
        )
        
        summary_table.insert(0, "Condition", exp_name)
        all_summaries.append(summary_table)

        print("\n--- MODEL COMPARISON TABLE ---")
        print(summary_table.to_string(index=False))

        fig1, ax1 = PoissonVisualizer.plot_model_fits(
            dataset=dataset,
            models=fitted_models,
        )
        plt.show(block=False)

        best_model_name = summary_table.iloc[0]["Model Name"]
        best_model = fitted_models[best_model_name]

        fig2, axes2 = PoissonVisualizer.plot_histogram(
            dataset=dataset,
            model=best_model,
        )
        plt.show(block=False)

        fig3, axes3 = PoissonVisualizer.plot_trial_traces(
            dataset=dataset,
            model=best_model,
        )
        plt.show(block=False)

        best_model.diagnose(dataset)
        #best_model.bootstrap(dataset, n_boot=500)

        del dataset
        del fitted_models
        del best_model
        del summary_table
        plt.close('all')
        gc.collect()

    master_summary_df = pd.concat(all_summaries, ignore_index=True)
    print("\n================ MASTER MODEL COMPARISON TABLE ================")
    print(master_summary_df.to_string(index=False))