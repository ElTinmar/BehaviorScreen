from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short


@dataclass
class PoissonDataset:
    event_times: np.ndarray
    event_trials: np.ndarray
    unique_trials: np.ndarray
    t_grid: np.ndarray
    num_fish: int
    bout_name: str
    laterality: str
    binned_counts: pd.DataFrame


class BehavioralDataLoader:
    def __init__(self, csv_path: Path):
        self.raw_df = pd.read_csv(csv_path)

    def prepare_dataset(
        self,
        bout_name: str,
        laterality: Laterality,
        stim: Optional[Union[Stim, str]] = None,
        epoch_name: Optional[Union[str, List[str]]] = None,
        dt: float = 0.02,
        t_start: float = 0.0,
        t_end: float = 24.0,
        window_duration: float = 0.33
    ) -> PoissonDataset:

        # 1. Filter by stimulus or epoch_name
        sub_df = self.raw_df.copy()
        if stim is not None:
            sub_df = sub_df[sub_df['stim'] == stim].copy()
        elif epoch_name is not None:
            if isinstance(epoch_name, list):
                sub_df = sub_df[sub_df['epoch_name'].isin(epoch_name)].copy()
            else:
                sub_df = sub_df[sub_df['epoch_name'] == epoch_name].copy()
        else:
            raise ValueError("Either 'stim' or 'epoch_name' must be provided to filter dataset.")

        # 2. Setup Time Grid
        t_grid = np.arange(t_start, t_end + dt, dt)
        window_size_steps = int(window_duration / dt) | 1

        # 3. Create Event Masks for all categories
        for idx, b_name in enumerate(bouts_category_name_short):
            for lat in Laterality:
                sub_df[f"{lat}_{b_name}"] = (
                    (sub_df['category'] == idx) & 
                    (sub_df['laterality'] == lat)
                )

        # 4. Bin & Aggregate Counts
        agg_dict = {
            f"count_{lat}_{b}": (f"{lat}_{b}", 'sum')
            for b in bouts_category_name_short
            for lat in Laterality
        }
        sub_df['time_bin'] = pd.cut(sub_df['trial_time'], bins=t_grid, right=False)
        
        counts = (
            sub_df.groupby(['file', 'trial_num', 'time_bin'], observed=False)
            .agg(**agg_dict)
        )

        # 5. Compute Rolling Hz
        count_cols = [f"count_{lat}_{b}" for b in bouts_category_name_short for lat in Laterality]
        rolling_cols = [f"rolling_{lat}_{b}" for b in bouts_category_name_short for lat in Laterality]
        hz_cols = [f"{lat}_{b}_hz" for b in bouts_category_name_short for lat in Laterality]

        counts[rolling_cols] = (
            counts.groupby(level=['file', 'trial_num'])[count_cols]
            .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
        )

        counts = counts.reset_index()
        counts[hz_cols] = counts[rolling_cols] / window_duration
        counts['time_sec'] = counts['time_bin'].apply(lambda x: x.left).astype(float)

        # 6. Extract target event times for Poisson fitting
        target_col = f"{laterality}_{bout_name}"
        event_mask = (
            sub_df[target_col] & 
            (sub_df['trial_time'] >= t_start) & 
            (sub_df['trial_time'] <= t_end)
        )
        events = sub_df[event_mask]

        return PoissonDataset(
            event_times=(events['trial_time'] - t_start).values,
            event_trials=events['trial_num'].values,
            unique_trials=np.sort(counts['trial_num'].unique()),
            t_grid=t_grid,
            num_fish=len(counts['file'].unique()),
            bout_name=bout_name,
            laterality=str(laterality),
            binned_counts=counts
        )


@dataclass
class RateKernel:
    """
    Standardized wrapper for rate kernels. 
    Handles lambda(), lambda(t), and lambda(t, trial) seamlessly.
    """
    name: str
    func: Callable[[np.ndarray, np.ndarray, List[float]], np.ndarray]
    param_names: List[str]
    initial_guesses: List[float]
    bounds: List[Tuple[Union[float, None], Union[float, None]]]
    stimulus_type: str = "General"
    description: str = ""

    def evaluate(self, t: np.ndarray, trial: np.ndarray, params: List[float]) -> np.ndarray:
        rate = self.func(t, trial, params)            
        target_shape = np.broadcast(t, trial).shape
        return np.broadcast_to(rate, target_shape)


class PoissonProcess:
    """
    Maximum Likelihood Estimator for Homogeneous and Inhomogeneous Poisson Processes.
    """
    def __init__(self, kernel: RateKernel):
        self.kernel = kernel
        self.fit_result: Optional[dict] = None
        self.params_: Optional[np.ndarray] = None
        self.param_dict_: Dict[str, float] = {}
        self.n_events_: int = 0
        self.n_fish_: float = 1.0

    def _nll(self, params, t_events, trial_events, unique_trials, t_grid, n_fish):
        # Term 1: Event Log-Rates
        event_rates = self.kernel.evaluate(t_events, trial_events, params)
        event_rates = np.maximum(event_rates, 1e-9)
        sum_log_rates = np.sum(np.log(event_rates))

        # Term 2: Numerical 2D Surface Integration
        t_2d = t_grid[None, :]               # Shape: (1, N_time)
        trials_2d = unique_trials[:, None]   # Shape: (N_trials, 1)

        rate_surface = self.kernel.evaluate(t_2d, trials_2d, params)
        rate_surface = np.maximum(rate_surface, 1e-9)

        # Simpson's Rule integration over time axis
        trial_integrals = simpson(rate_surface, x=t_grid, axis=1)
        total_expected_events = np.sum(trial_integrals) * n_fish

        return -(sum_log_rates - total_expected_events)

    def fit(
        self, 
        dataset_or_t_events: Union[PoissonDataset, np.ndarray], 
        trial_events: Optional[np.ndarray] = None, 
        unique_trials: Optional[np.ndarray] = None, 
        t_grid: Optional[np.ndarray] = None, 
        n_fish: float = 1.0, 
        method: str = 'L-BFGS-B', 
        **kwargs
    ):
        """
        Fits kernel parameters to event data.
        Accepts either a PoissonDataset instance OR raw NumPy arrays.
        """
        if isinstance(dataset_or_t_events, PoissonDataset):
            ds = dataset_or_t_events
            t_events = ds.event_times
            trial_events = ds.event_trials
            unique_trials = ds.unique_trials
            t_grid = ds.t_grid
            n_fish = float(ds.num_fish)
        else:
            t_events = dataset_or_t_events
            if trial_events is None or unique_trials is None or t_grid is None:
                raise ValueError(
                    "When passing raw arrays to .fit(), trial_events, unique_trials, and t_grid must be provided."
                )

        self.n_events_ = len(t_events)
        self.n_fish_ = n_fish

        res = minimize(
            self._nll,
            x0=self.kernel.initial_guesses,
            args=(t_events, trial_events, unique_trials, t_grid, n_fish),
            method=method,
            bounds=self.kernel.bounds,
            **kwargs
        )

        self.fit_result = res
        self.params_ = res.x
        self.param_dict_ = dict(zip(self.kernel.param_names, res.x))
        return self

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray]) -> np.ndarray:
        """Predicts rate intensity lambda(t, trial)."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return np.maximum(self.kernel.evaluate(t, trial, self.params_), 1e-9)

    @property
    def log_likelihood(self) -> float:
        return -self.fit_result.fun if self.fit_result else np.nan

    @property
    def aic(self) -> float:
        k = len(self.params_)
        return 2 * k - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        k = len(self.params_)
        return k * np.log(self.n_events_) - 2 * self.log_likelihood


class KernelFactory:

    @staticmethod
    def homogeneous_poisson() -> RateKernel:
        def _func(t, trial, params):
            mu = params[0]
            return mu * np.ones_like(t + 0.0 * trial)

        return RateKernel(
            name="Homogeneous Poisson λ()",
            func=_func,
            param_names=["μ"],
            initial_guesses=[0.5],
            bounds=[(0.001, 20.0)],
            description="Constant background firing rate without temporal or trial modulation."
        )

    @staticmethod
    def prey_capture_time_only(stim_freq: float) -> RateKernel:
        def _func(t, trial, params):
            A, tau, k, B, b1, b2, b3, b4 = params
            w = 2.0 * np.pi * stim_freq
            phase = w * t

            transient = A * (t ** k) * np.exp(-t / tau)
            baseline = B
            phase_ripple = (
                b1 * np.sin(phase) + b2 * np.cos(phase) +
                b3 * np.sin(2.0 * phase) + b4 * np.cos(2.0 * phase)
            )
            return transient + baseline + phase_ripple

        return RateKernel(
            name="Prey Capture Time-Only λ(t)",
            func=_func,
            param_names=["A", "tau", "k", "B", "b1", "b2", "b3", "b4"],
            initial_guesses=[0.56, 1.15, 2.0, 0.40, 0.0, 0.0, 0.0, 0.0],
            bounds=[
                (0.01, 10.0), (0.1, 5.0), (0.05, 10.0), (0.01, 5.0),
                (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)
            ],
            stimulus_type="Prey Capture",
            description="Time-varying kinetics and phase preferences without trial plasticity."
        )

    @staticmethod
    def prey_capture_full(stim_freq: float) -> RateKernel:
        def _func(t, trial, params):
            A, tau, k, B, b1, b2, b3, b4, a_A, a_B, a_g = params
            w = 2.0 * np.pi * stim_freq
            phase = w * t

            transient = A * (t ** k) * np.exp(-t / tau) * np.exp(a_A * trial)
            baseline = B * np.exp(a_B * trial)
            phase_ripple = (
                b1 * np.sin(phase) + b2 * np.cos(phase) +
                b3 * np.sin(2.0 * phase) + b4 * np.cos(2.0 * phase)
            ) * np.exp(a_g * trial)

            return transient + baseline + phase_ripple

        return RateKernel(
            name="Prey Capture Full Plasticity λ(t, m)",
            func=_func,
            param_names=["A", "tau", "k", "B", "b1", "b2", "b3", "b4", "alpha_A", "alpha_B", "alpha_gamma"],
            initial_guesses=[0.56, 1.15, 2.0, 0.40, 0.0, 0.0, 0.0, 0.0, -0.05, -0.05, -0.05],
            bounds=[
                (0.01, 10.0), (0.1, 5.0), (0.05, 10.0), (0.01, 5.0),
                (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
                (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
            ],
            stimulus_type="Prey Capture",
            description="Full kinetics, second-harmonic spatial tuning, and trial-by-trial plasticity."
        )

    @staticmethod
    def phototaxis() -> RateKernel:
        def _func(t, trial, params):
            B, A_dip, tau_dip, alpha_B, alpha_dip = params

            mod_B = B * np.exp(alpha_B * trial)
            mod_dip = A_dip * np.exp(alpha_dip * trial) * np.exp(-t / tau_dip)

            return mod_B - mod_dip

        return RateKernel(
            name="Phototaxis λ(t, m)",
            func=_func,
            param_names=["B", "A_dip", "tau_dip", "alpha_B", "alpha_dip"],
            initial_guesses=[0.4, 0.3, 0.5, 0.0, 0.0],
            bounds=[
                (0.01, 10.0), (0.0, 5.0), (0.01, 5.0), (-0.1, 0.1),
                (-0.1, 0.1)
            ],
            stimulus_type="Phototaxis",
            description="Full kinetics and trial-by-trial plasticity."
        )

    @staticmethod
    def omr_forward() -> RateKernel:
        def _func(t, trial, params):
            B, A_dip, tau_dip = params

            dip = A_dip * np.exp(-t / tau_dip)
            return B - dip

        return RateKernel(
            name="OMR forward λ(t)",
            func=_func,
            param_names=["B", "A_dip", "tau_dip"],
            initial_guesses=[0.4, 0.5, 0.5],
            bounds=[
                (0.01, 5.0), (0.0, 5.0), (0.01, 5.0)
            ],
            stimulus_type="OMR forward",
            description=""
        )
            
    @staticmethod
    def looming_exponential(lv_ratio: float = 0.1) -> RateKernel:
        def _func(t, trial, params):
            A, tau, t_collision, B, alpha = params
            time_to_collision = np.maximum(t_collision - t, 0.001)
            expansion_rate = (lv_ratio / (time_to_collision ** 2 + lv_ratio ** 2))
            
            looming_drive = A * np.exp(expansion_rate / tau) * np.exp(alpha * trial)
            return looming_drive + B

        return RateKernel(
            name="Looming Expansion Model λ(t, m)",
            func=_func,
            param_names=["A", "tau", "t_collision", "B", "alpha"],
            initial_guesses=[0.1, 1.0, 5.0, 0.1, -0.05],
            bounds=[(0.001, 10.0), (0.1, 10.0), (1.0, 20.0), (0.001, 5.0), (-2.0, 2.0)],
            stimulus_type="Looming",
            description="Non-linear escape bout triggering driven by visual expansion rates."
        )

    @staticmethod
    def looming_gaussian(t_critical: float = 5) -> RateKernel:
        def _func(t, trial, params):
            B, H, alpha, mu, sigma = params

            height = H * np.exp(alpha * trial)
            exponent = -0.5 * ((t - mu) / sigma)**2
            return B + (height * np.exp(exponent))

        return RateKernel(
            name="Looming Gaussian λ(t, m)",
            func=_func,
            param_names=["B", "H", "alpha", "mu", "sigma"],
            initial_guesses=[0.1, 1.2, 0.0, 5.0, 0.2],
            bounds=[
                (0.0, 10.0), (0.001, 10.0), (-2.0, 2.0), 
                (t_critical-1.5, t_critical+1.5), (0.01, 1.0)
            ],
            stimulus_type="Looming",
            description=""
        )

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
            stimulus_type="Dark flash",
            description=""
        )
    

class PoissonBootstrapper:
    """
    Performs Non-Parametric Trial-Level Resampling (Cluster Bootstrap) 
    for fitted PoissonProcess models.
    """
    def __init__(
        self, 
        model: PoissonProcess, 
        n_bootstraps: int = 200, 
        random_state: Optional[int] = 42
    ):
        self.model = model
        self.n_bootstraps = n_bootstraps
        self.rng = np.random.default_rng(random_state)
        self.bootstrap_params_: Optional[np.ndarray] = None  # Shape: (N_boot, N_params)
        self.summary_df_: Optional[pd.DataFrame] = None

    def fit(self, dataset: PoissonDataset) -> "PoissonBootstrapper":
        """Resamples trials with replacement and refits the kernel."""
        unique_trials = dataset.unique_trials
        n_trials = len(unique_trials)

        # Pre-index event times per trial for high-speed resampling
        events_by_trial = {
            tr: dataset.event_times[dataset.event_trials == tr]
            for tr in unique_trials
        }

        boot_params = []

        for b in range(self.n_bootstraps):
            # 1. Resample trials with replacement
            sampled_trials = self.rng.choice(unique_trials, size=n_trials, replace=True)

            # 2. Reconstruct event arrays with virtual trial indices (0 to n_trials - 1)
            boot_event_times = []
            boot_event_trials = []

            for new_trial_idx, original_trial in enumerate(sampled_trials):
                t_ev = events_by_trial[original_trial]
                boot_event_times.extend(t_ev)
                boot_event_trials.extend([new_trial_idx] * len(t_ev))

            boot_event_times = np.array(boot_event_times)
            boot_event_trials = np.array(boot_event_trials)
            boot_unique_trials = np.arange(n_trials)

            # 3. Fit fresh model instance
            boot_model = PoissonProcess(self.model.kernel)
            try:
                boot_model.fit(
                    dataset_or_t_events=boot_event_times,
                    trial_events=boot_event_trials,
                    unique_trials=boot_unique_trials,
                    t_grid=dataset.t_grid,
                    n_fish=dataset.num_fish
                )
                if boot_model.fit_result.success:
                    boot_params.append(boot_model.params_)
            except Exception:
                continue

        self.bootstrap_params_ = np.array(boot_params)
        self._compute_summary()
        return self

    def _compute_summary(self):
        """Computes point estimates, standard errors, and percentile 95% CIs."""
        param_names = self.model.kernel.param_names
        means = np.mean(self.bootstrap_params_, axis=0)
        stds = np.std(self.bootstrap_params_, axis=0)
        ci_lower = np.percentile(self.bootstrap_params_, 2.5, axis=0)
        ci_upper = np.percentile(self.bootstrap_params_, 97.5, axis=0)

        self.summary_df_ = pd.DataFrame({
            "Parameter": param_names,
            "MLE Estimate": self.model.params_,
            "Boot Mean": means,
            "Std Error": stds,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper
        })

    def predict_interval(
        self, 
        t: np.ndarray, 
        trial: Union[float, np.ndarray], 
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates rate curves across all bootstrapped parameter sets.
        Returns (mean_rate, lower_ci, upper_ci).
        """
        rates = []
        for params in self.bootstrap_params_:
            r = self.model.kernel.evaluate(t, trial, params)
            rates.append(r)
        rates = np.array(rates)

        mean_rate = np.mean(rates, axis=0)
        lower = np.percentile(rates, 100 * (alpha / 2.0), axis=0)
        upper = np.percentile(rates, 100 * (1.0 - alpha / 2.0), axis=0)

        return mean_rate, lower, upper

    
class ModelComparator:

    @staticmethod
    def likelihood_ratio_test(
        model_null: PoissonProcess, 
        model_alt: PoissonProcess
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Performs a Likelihood Ratio Test (LRT) between two nested Poisson Process models.
        
        Null Model (H0): Restricted model with k_null parameters.
        Alt Model  (H1): Complex model with k_alt parameters (k_alt > k_null).
        """
        k_null = len(model_null.params_)
        k_alt = len(model_alt.params_)

        if k_alt <= k_null:
            raise ValueError(
                f"LRT requires the alternative model to have more parameters than the null model. "
                f"Got k_null={k_null}, k_alt={k_alt}."
            )

        ll_null = model_null.log_likelihood
        ll_alt = model_alt.log_likelihood

        # LRT Statistic: D = 2 * (LL_alt - LL_null)
        lr_stat = 2.0 * (ll_alt - ll_null)
        
        # Ensure numerical safety if optimization converged slightly off
        lr_stat_clamped = max(0.0, lr_stat)
        df = k_alt - k_null
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
    def sequential_lrt(
        fitted_models: Dict[str, PoissonProcess]
    ) -> pd.DataFrame:
        """
        Runs sequential LRT comparisons across an ordered chain of nested models.
        Assumes models are passed in ascending order of complexity.
        """
        model_items = list(fitted_models.items())
        results = []

        for i in range(len(model_items) - 1):
            _, m_null = model_items[i]
            _, m_alt = model_items[i + 1]

            res = ModelComparator.likelihood_ratio_test(m_null, m_alt)
            results.append(res)

        return pd.DataFrame(results)

    @staticmethod
    def compare(
        kernels: List[RateKernel], 
        dataset_or_t_events: Union[PoissonDataset, np.ndarray],
        trial_events: Optional[np.ndarray] = None,
        unique_trials: Optional[np.ndarray] = None,
        t_grid: Optional[np.ndarray] = None,
        n_fish: float = 1.0,
        method: str = 'L-BFGS-B',
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, PoissonProcess]]:
        """Fits and benchmarks multiple RateKernels on a dataset or raw arrays."""
        models = {}
        records = []

        for kernel in kernels:
            model = PoissonProcess(kernel)
            
            if isinstance(dataset_or_t_events, PoissonDataset):
                model.fit(dataset_or_t_events, method=method, **kwargs)
            else:
                model.fit(
                    dataset_or_t_events,
                    trial_events=trial_events,
                    unique_trials=unique_trials,
                    t_grid=t_grid,
                    n_fish=n_fish,
                    method=method,
                    **kwargs
                )
            models[kernel.name] = model

            records.append({
                "Model Name": kernel.name,
                "Params (k)": len(kernel.param_names),
                "Log-Likelihood": model.log_likelihood,
                "AIC": model.aic,
                "BIC": model.bic,
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
    def plot_model_fits(
        dataset: PoissonDataset,
        fitted_models: Dict[str, PoissonProcess],
        selected_trials: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (12, 5),
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the empirical PSTH (Peri-Stimulus Time Histogram) and overlays
        predicted intensity functions lambda(t, trial) averaged across selected trials.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 1. Compute & Plot Empirical PSTH
        hz_col = f"{dataset.laterality}_{dataset.bout_name}_hz"
        if selected_trials is not None:
            df_sub = dataset.binned_counts[dataset.binned_counts['trial_num'].isin(selected_trials)]
            trials_to_eval = np.array(selected_trials)
        else:
            df_sub = dataset.binned_counts
            trials_to_eval = dataset.unique_trials

        psth = df_sub.groupby('time_sec')[hz_col].agg(['mean', 'sem']).reset_index()

        ax.bar(
            psth['time_sec'], psth['mean'], 
            width=dataset.t_grid[1] - dataset.t_grid[0],
            color='lightgray', edgecolor='darkgray', alpha=0.6,
            label='Empirical PSTH (Data)', align='edge'
        )
        ax.fill_between(
            psth['time_sec'], 
            psth['mean'] - psth['sem'], 
            psth['mean'] + psth['sem'], 
            color='gray', alpha=0.2
        )

        # 2. Evaluate and Overlay Model Intensity Curves
        colors = plt.cm.tab10(np.linspace(0, 1, len(fitted_models)))

        for idx, (model_name, model) in enumerate(fitted_models.items()):
            t_2d = dataset.t_grid[None, :]               # Shape: (1, N_time)
            trials_2d = trials_to_eval[:, None]          # Shape: (N_trials, 1)

            rate_surface = model.predict(t_2d, trials_2d) # Shape: (N_trials, N_time)
            mean_rate = np.mean(rate_surface, axis=0)     # Shape: (N_time,)

            label = f"{model_name} (AIC: {model.aic:.1f})"
            ax.plot(dataset.t_grid, mean_rate, label=label, linewidth=2.5, color=colors[idx])

        # 3. Formatting
        ax.set_xlabel("Time in Trial (s)", fontsize=11)
        ax.set_ylabel("Event Rate (Hz)", fontsize=11)
        
        main_title = title or f"Model Fits: {dataset.bout_name} ({dataset.laterality})"
        ax.set_title(main_title, fontsize=13, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_raster_and_surface(
        dataset: PoissonDataset,
        model: PoissonProcess,
        figsize: Tuple[int, int] = (14, 5),
        cmap: str = "viridis",
        raster_color: str = "white",
        raster_alpha: float = 0.7,
        raster_size: float = 14
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Symmetric 2-panel visualizer overlaying empirical raster onto rate surfaces:
          Panel 1: Empirical Rate Surface + Overlaid Event Raster (Data)
          Panel 2: Model Predicted Rate Surface lambda(t, m) + Overlaid Event Raster (Model)
        """
        fig, (ax_emp, ax_mod) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # -------------------------------------------------------------
        # 1. Compute Empirical Firing Rate Surface Matrix
        # -------------------------------------------------------------
        hz_col = f"{dataset.laterality}_{dataset.bout_name}_hz"
        pivoted = (
            dataset.binned_counts
            .groupby(['trial_num', 'time_sec'])[hz_col]
            .mean()
            .unstack(level='time_sec')
        )
        pivoted = pivoted.reindex(index=dataset.unique_trials).fillna(0.0)
        empirical_surface = pivoted.values
        time_sec_bins = pivoted.columns.values

        # -------------------------------------------------------------
        # 2. Compute Model Rate Surface Matrix
        # -------------------------------------------------------------
        t_2d = time_sec_bins[None, :]                  # Shape: (1, N_time_bins)
        trials_2d = dataset.unique_trials[:, None]      # Shape: (N_trials, 1)
        model_surface = model.predict(t_2d, trials_2d)  # Shape: (N_trials, N_time_bins)

        # -------------------------------------------------------------
        # 3. Shared Color Scale for Direct Visual Comparison
        # -------------------------------------------------------------
        vmax = max(np.max(empirical_surface), np.max(model_surface))
        vmin = 0.0

        # -------------------------------------------------------------
        # Panel 1: Data (Empirical Surface + Raster)
        # -------------------------------------------------------------
        mesh_emp = ax_emp.pcolormesh(
            time_sec_bins, 
            dataset.unique_trials, 
            empirical_surface, 
            shading='auto', 
            cmap=cmap,
            vmin=vmin, 
            vmax=vmax
        )
        ax_emp.scatter(
            dataset.event_times, 
            dataset.event_trials, 
            color=raster_color, s=raster_size, alpha=raster_alpha, marker='|'
        )
        ax_emp.set_title("Empirical Data & Raster", fontsize=12, fontweight='bold')
        ax_emp.set_xlabel("Time in Trial (s)", fontsize=11)
        ax_emp.set_ylabel("Trial Number", fontsize=11)
        ax_emp.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])

        # -------------------------------------------------------------
        # Panel 2: Model (Predicted Surface + Raster)
        # -------------------------------------------------------------
        mesh_mod = ax_mod.pcolormesh(
            time_sec_bins, 
            dataset.unique_trials, 
            model_surface, 
            shading='auto', 
            cmap=cmap,
            vmin=vmin, 
            vmax=vmax
        )
        ax_mod.scatter(
            dataset.event_times, 
            dataset.event_trials, 
            color=raster_color, s=raster_size, alpha=raster_alpha, marker='|'
        )
        ax_mod.set_title(f"Fitted Model: {model.kernel.name}", fontsize=12, fontweight='bold')
        ax_mod.set_xlabel("Time in Trial (s)", fontsize=11)
        ax_mod.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])

        # -------------------------------------------------------------
        # 4. Colorbar Tucked Tightly to the Right Panel
        # -------------------------------------------------------------
        divider = make_axes_locatable(ax_mod)
        cax = divider.append_axes("right", size="3%", pad=0.12)
        cbar = fig.colorbar(mesh_mod, cax=cax)
        cbar.set_label("Event Rate [Hz]", fontsize=10)

        plt.tight_layout()
        return fig, np.array([ax_emp, ax_mod])


    @staticmethod
    def plot_bootstrapped_fit(
        dataset: PoissonDataset,
        bootstrapper: PoissonBootstrapper,
        figsize: Tuple[int, int] = (12, 5)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots empirical PSTH alongside bootstrapped mean rate and 95% CI band."""
        fig, ax = plt.subplots(figsize=figsize)

        # 1. Plot Empirical PSTH
        hz_col = f"{dataset.laterality}_{dataset.bout_name}_hz"
        psth = dataset.binned_counts.groupby('time_sec')[hz_col].agg(['mean', 'sem']).reset_index()

        ax.bar(
            psth['time_sec'], psth['mean'], 
            width=dataset.t_grid[1] - dataset.t_grid[0],
            color='lightgray', edgecolor='darkgray', alpha=0.6,
            label='Empirical PSTH (Data)', align='edge'
        )

        # 2. Evaluate Bootstrapped Prediction Surface over all trials
        t_2d = dataset.t_grid[None, :]
        trials_2d = dataset.unique_trials[:, None]

        mean_surface, lower_surface, upper_surface = bootstrapper.predict_interval(
            t_2d, trials_2d, alpha=0.05
        )

        # Average across trials to compare with overall PSTH
        mean_rate = np.mean(mean_surface, axis=0)
        lower_ci = np.mean(lower_surface, axis=0)
        upper_ci = np.mean(upper_surface, axis=0)

        # 3. Overlay Fit & Confidence Band
        kernel_name = bootstrapper.model.kernel.name
        ax.plot(dataset.t_grid, mean_rate, color='crimson', linewidth=2, label=f'{kernel_name} (Boot Mean)')
        ax.fill_between(
            dataset.t_grid, lower_ci, upper_ci, 
            color='crimson', alpha=0.25, label='95% Bootstrap CI'
        )

        ax.set_xlabel("Time in Trial (s)", fontsize=11)
        ax.set_ylabel("Event Rate (Hz)", fontsize=11)
        ax.set_title(f"Bootstrapped Model Fit: {dataset.bout_name} ({dataset.laterality})", fontweight='bold')
        ax.legend(frameon=True, facecolor='white')
        ax.grid(True, linestyle='--', alpha=0.4)

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
                'dt':0.02, 
                't_start':0.0, 
                't_end':24.0, 
                'window_duration':0.33
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.prey_capture_time_only(stim_freq=prey_stim_freq),
                KernelFactory.prey_capture_full(stim_freq=prey_stim_freq)
            ]
        },

        'prey_capture_contra': {
            'dataset': {
                'stim':Stim.PREY_CAPTURE,
                'bout_name':'JT',
                'laterality':Laterality.CONTRALATERAL,
                'dt':0.02, 
                't_start':0.0, 
                't_end':24.0, 
                'window_duration':0.33
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
                'dt':0.02, 
                't_start':0.0, 
                't_end':24.0, 
                'window_duration':0.33
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.phototaxis()
            ]
        },

        'phototaxis_contra': {
            'dataset': {
                'stim':Stim.PHOTOTAXIS,
                'bout_name':'RT',
                'laterality':Laterality.CONTRALATERAL,
                'dt':0.02, 
                't_start':0.0, 
                't_end':24.0, 
                'window_duration':0.33
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
            ]
        },

        'omr_lateral_ipsi': {
            'dataset': {
                'epoch_name':["grating right", "grating left"],
                'bout_name':'RT',
                'laterality':Laterality.IPSILATERAL,
                'dt':0.02, 
                't_start':0.0, 
                't_end':10.0, 
                'window_duration':0.33
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
                'dt':0.02, 
                't_start':0.0, 
                't_end':10.0, 
                'window_duration':0.33
            },
            'kernels': [
                KernelFactory.homogeneous_poisson()
            ]
        },

        'omr_forward': {
            'dataset': {
                'epoch_name':"grating forward",
                'bout_name':'BS',
                'laterality':Laterality.NONDIRECTIONAL,
                'dt':0.02, 
                't_start':0.0, 
                't_end':10.0, 
                'window_duration':0.33
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
                'dt':0.02, 
                't_start':0.0, 
                't_end':10.0, 
                'window_duration':0.33
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
                'dt':0.02, 
                't_start':0.0, 
                't_end':10.0, 
                'window_duration':0.33
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
                'dt':0.02, 
                't_start':0.0, 
                't_end':10.0, 
                'window_duration':0.33
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
                'dt':0.02, 
                't_start':0.0, 
                't_end':10.0, 
                'window_duration':0.33
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.looming_gaussian()
            ]
        },

        'dark_flash': {
            'dataset': {
                'epoch_name':"flash dark",
                'bout_name':'SLC',
                'laterality':Laterality.IPSILATERAL,
                'dt':0.02, 
                't_start':0.0, 
                't_end':10.0, 
                'window_duration':0.33
            },
            'kernels': [
                KernelFactory.homogeneous_poisson(),
                KernelFactory.dark_flash()
            ]
        },
    }

    all_summaries = []
    processed_experiments = {}

    for exp_name, config in model_config.items():
        print(f"\n==================================================")
        print(f" PROCESSING EXPERIMENT: {exp_name.upper()}")
        print(f"==================================================")

        # A. Prepare Dataset
        dataset = loader.prepare_dataset(**config['dataset'])

        # B. Run Model Comparison
        summary_table, fitted_models = ModelComparator.compare(
            kernels=config['kernels'],
            dataset_or_t_events=dataset
        )
        
        summary_table.insert(0, "Condition", exp_name)
        all_summaries.append(summary_table)
        processed_experiments[exp_name] = {
            "dataset": dataset,
            "models": fitted_models,
            "summary": summary_table
        }

        print("\n--- MODEL COMPARISON TABLE ---")
        print(summary_table.to_string(index=False))

        # C. Sequential Likelihood Ratio Tests (if multiple models are compared)
        if len(config['kernels']) > 1:
            lrt_table = ModelComparator.sequential_lrt(fitted_models)
            print("\n--- SEQUENTIAL LIKELIHOOD RATIO TESTS ---")
            print(lrt_table.to_string(index=False))

        # D. Visualizations
        fig1, ax1 = PoissonVisualizer.plot_model_fits(
            dataset=dataset,
            fitted_models=fitted_models,
            title=f"Condition: {exp_name} | {dataset.bout_name} ({dataset.laterality})"
        )
        plt.show()

        best_model_name = summary_table.iloc[0]["Model Name"]
        best_model = fitted_models[best_model_name]

        fig2, axes2 = PoissonVisualizer.plot_raster_and_surface(
            dataset=dataset,
            model=best_model,
            cmap="plasma"
        )
        plt.show()

    master_summary_df = pd.concat(all_summaries, ignore_index=True)
    print("\n================ MASTER MODEL COMPARISON TABLE ================")
    print(master_summary_df.to_string(index=False))