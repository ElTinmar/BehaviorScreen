from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional, Union, Any
import numpy as np
import pandas as pd

import joblib
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .dataset import PointProcessDataset
from .point_process import PointProcess


def _fit_single_bootstrap(seed_seq, dataset: PointProcessDataset, kernel):
    rng = np.random.default_rng(seed_seq)
    ds_boot = dataset.resample(rng)
    boot_model = PoissonProcess(kernel)
    try:
        boot_model.fit(ds_boot)
        return boot_model.params_
    except Exception:
        return None

    
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

        if len(t_events) == 0:
            return np.array([], dtype=float)

        if self.integral_func is not None:
            return self.integral_func(0, t_events, trial, params)

        t_max = np.max(t_events)
        if t_max <= 0:
            return np.zeros_like(t_events, dtype=float)

        t_grid = np.arange(0, t_max + integration_dt, integration_dt)
        trials_2d = np.atleast_1d(trial)[:, None]
        rate_surface = self.evaluate(t_grid[None, :], trials_2d, params)
        rate_surface = np.maximum(rate_surface, 1e-9) # TODO find a way to get rid of that

        cum_integral = cumulative_trapezoid(rate_surface, t_grid, initial=0.0, axis=1).squeeze()
        return np.interp(t_events, t_grid, cum_integral)

    def is_nested_in(self, parent_kernel: "RateKernel") -> bool:
        return set(parent_kernel.param_names).issubset(set(self.param_names))


class RateKernelFactory:

    @staticmethod
    def homogeneous_poisson() -> RateKernel:
        def _func(t, trial, params):
            B = params[0]
            return B * np.ones_like(t + 0.0 * trial)

        return RateKernel(
            name="Homogeneous λ",
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
    
class PoissonProcess(PointProcess):

    def __init__(self, kernel: RateKernel, integration_dt: float = 0.02):

        super.__init__(self, integration_dt)

        self.name = f"Poisson {kernel.name}"
        self.kernel = kernel
        self.initial_guesses = kernel.initial_guesses
        self.bounds = kernel.bounds
        self.param_names = kernel.param_names

    def _nll(
        self, 
        params: List[float], 
        dataset: PointProcessDataset
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
    
    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray]) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        
        expected_rate = self.kernel.evaluate(t, trial, self.params_)
        expected_rate = np.maximum(expected_rate, 1e-9)
        return expected_rate

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
                for i, name in enumerate(self.param_names)
            ]
        )

    def binned_residuals(self, dataset: PointProcessDataset) -> Dict[str, np.ndarray]:
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
        dataset: PointProcessDataset,
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


    def diagnose(
        self, 
        dataset: PointProcessDataset, 
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


