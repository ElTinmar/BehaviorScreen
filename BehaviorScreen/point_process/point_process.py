from typing import List, Tuple, Dict, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, kstest, chi2

from .dataset import PointProcessDataset

class PointProcess:

    def __init__(self, integration_dt: float = 0.02):
        self.name: str = ""
        self.integration_dt = integration_dt
        self.fit_result: Optional[Any] = None
        self.params_: Optional[np.ndarray] = None
        self.param_dict_: Dict[str, float] = {}
        self.initial_guesses: List[float] = None
        self.bounds: List[Tuple[Optional[float], Optional[float]]] = None
        self.param_names: List[str] = None

    def _nll(self, params: List[float], dataset: PointProcessDataset): 
        ...

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

    def predict(self, t: np.ndarray, trial: np.ndarray) -> np.ndarray: 
        ...

    def cumulative_integrated_intensity(self, *args): 
        ...

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
    ) -> Tuple[pd.DataFrame, Dict[str, PointProcess]]:
        
        records = []

        for model in models:
            model.fit(dataset, method=method, **kwargs)
            records.append({
                "Model Name": model.name,
                "Params (k)": len(model.param_names),
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

class ModelPlotter:

    @staticmethod
    def plot_histogram(
        dataset: PointProcessDataset,
        model: PointProcess,
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
        dataset: PointProcessDataset,
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
        dataset: PointProcessDataset,
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