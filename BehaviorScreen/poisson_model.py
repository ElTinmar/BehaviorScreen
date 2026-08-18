from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import gc
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
    num_file_trials: int  
    bout_name: str
    laterality: str
    binned_counts: pd.DataFrame
    histogram_counts: np.ndarray 
    histogram_hz: np.ndarray


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
            sub_df = sub_df[sub_df['stim'] == stim]

            # handle special cases
            if stim == Stim.PHOTOTAXIS:
                sub_df = sub_df[sub_df['foreground_color'] == '[0.1, 0.1, 0.0, 1.0]']

        elif epoch_name is not None:
            if isinstance(epoch_name, list):
                sub_df = sub_df[sub_df['epoch_name'].isin(epoch_name)]
            else:
                sub_df = sub_df[sub_df['epoch_name'] == epoch_name]

        else:
            raise ValueError("Either 'stim' or 'epoch_name' must be provided to filter dataset.")


        # 2. Setup Time Grid
        t_grid = np.arange(t_start, t_end + dt, dt)
        window_size_steps = int(window_duration / dt) | 1

        # 3. Create Event Masks for all categories
        event_masks = {}
        for idx, b_name in enumerate(bouts_category_name_short):
            for lat in Laterality:
                col_name = f"{lat}_{b_name}"
                event_masks[col_name] = (
                    (sub_df['category'] == idx) & 
                    (sub_df['laterality'] == lat)
                )

        sub_df = pd.concat([sub_df, pd.DataFrame(event_masks, index=sub_df.index)], axis=1)

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

        rolling_df = counts.groupby(level=['file', 'trial_num'])[count_cols].transform(
            lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum()
        )
        rolling_df.columns = rolling_cols

        hz_df = rolling_df / window_duration
        hz_df.columns = hz_cols

        counts = pd.concat([counts, rolling_df, hz_df], axis=1).reset_index()
        counts['time_sec'] = counts['time_bin'].apply(lambda x: x.left).astype(float)

        # 6. Extract target event times for Poisson fitting
        target_col = f"{laterality}_{bout_name}"
        event_mask = (
            sub_df[target_col] & 
            (sub_df['trial_time'] >= t_start) & 
            (sub_df['trial_time'] <= t_end)
        )
        events = sub_df[event_mask]

        # compute histogram
        num_file_trials = len(counts[['file', 'trial_num']].drop_duplicates())
        hist_counts, _ = np.histogram(events['trial_time'].values, bins=t_grid)
        if num_file_trials > 0:
            hist_hz = hist_counts / (num_file_trials * dt)
        else:
            hist_hz = np.zeros_like(hist_counts, dtype=float)

        return PoissonDataset(
            event_times=(events['trial_time'] - t_start).values,
            event_trials=events['trial_num'].values,
            unique_trials=np.sort(counts['trial_num'].unique()),
            t_grid=t_grid,
            num_fish=len(counts['file'].unique()),
            num_file_trials=num_file_trials,
            bout_name=bout_name,
            laterality=str(laterality),
            binned_counts=counts,
            histogram_counts=hist_counts,
            histogram_hz=hist_hz
        )


@dataclass
class RateKernel:
    name: str
    func: Callable[[np.ndarray, np.ndarray, List[float]], np.ndarray]
    param_names: List[str]
    initial_guesses: List[float]
    bounds: List[Tuple[Union[float, None], Union[float, None]]]
    latex_formula: str = ""

    def evaluate(self, t: np.ndarray, trial: np.ndarray, params: List[float]) -> np.ndarray:
        rate = self.func(t, trial, params)            
        target_shape = np.broadcast(t, trial).shape
        return np.broadcast_to(rate, target_shape)


class PoissonProcess:
    """
    Maximum Likelihood Estimator for Homogeneous and Inhomogeneous Poisson Processes.
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.fit_result: Optional[dict] = None
        self.params_: Optional[np.ndarray] = None
        self.param_dict_: Dict[str, float] = {}
        self.n_events_: int = 0
        self.n_observations_: int = 1

    def _nll(self, params, t_events, trial_events, unique_trials, t_grid, num_file_trials):
        # Term 1: Sum of Log Intensity at Observed Events
        event_rates = self.kernel.evaluate(t_events, trial_events, params)
        event_rates = np.maximum(event_rates, 1e-9)
        sum_log_rates = np.sum(np.log(event_rates))

        # Term 2: Expected Total Events (Surface Integration over Time)
        t_2d = t_grid[None, :]               # Shape: (1, N_time)
        trials_2d = unique_trials[:, None]   # Shape: (N_trials, 1)

        rate_surface = self.kernel.evaluate(t_2d, trials_2d, params)
        rate_surface = np.maximum(rate_surface, 1e-9)

        # Integrate rate over time for each trial layout
        trial_integrals = simpson(rate_surface, x=t_grid, axis=1)
        
        # Scale expected events by average per-trial integral times total observation units
        avg_trial_integral = np.mean(trial_integrals)
        total_expected_events = avg_trial_integral * num_file_trials

        return -(sum_log_rates - total_expected_events)

    def fit(
        self, 
        dataset_or_t_events: Union[PoissonDataset, np.ndarray], 
        trial_events: Optional[np.ndarray] = None, 
        unique_trials: Optional[np.ndarray] = None, 
        t_grid: Optional[np.ndarray] = None, 
        num_file_trials: int = 1, 
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
            num_file_trials = ds.num_file_trials
        else:
            t_events = dataset_or_t_events
            if trial_events is None or unique_trials is None or t_grid is None:
                raise ValueError(
                    "When passing raw arrays to .fit(), trial_events, unique_trials, and t_grid must be provided."
                )

        self.n_events_ = len(t_events)
        self.n_observations_ = num_file_trials

        res = minimize(
            self._nll,
            x0=self.kernel.initial_guesses,
            args=(t_events, trial_events, unique_trials, t_grid, num_file_trials),
            method=method,
            bounds=self.kernel.bounds,
            **kwargs
        )

        self.fit_result = res
        self.params_ = res.x
        self.param_dict_ = dict(zip(self.kernel.param_names, res.x))
        return self

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray]) -> np.ndarray:
        """Predicts expected rate intensity lambda(t, trial) per single trial observation."""
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
            latex_formula=r"$\lambda = \mu$",
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
            latex_formula=r"$\lambda(t) = A t^k e^{-t/\tau} + B + \sum_{n=1}^2 \left(b_{2n-1}\sin(n\omega t) + b_{2n}\cos(n\omega t)\right)$",
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
            latex_formula=r"$\lambda(t, m) = A t^k e^{-t/\tau} e^{\alpha_A m} + B e^{\alpha_B m} + \left[ \sum_{n=1}^{2} \left( b_{2n-1} \sin(n \omega t) + b_{2n} \cos(n \omega t) \right) \right] e^{\alpha_\gamma m}$",
        )


    @staticmethod
    def phototaxis_ipsi() -> RateKernel:
        def _func(t, trial, params):
            B, A_dip, A_peak, tau, alpha_B, alpha_transient = params

            mod_B = B * np.exp(alpha_B * trial)
            transient = (A_peak * (t / tau) - A_dip) * np.exp(-t / tau) * np.exp(alpha_transient * trial)

            return mod_B + transient

        return RateKernel(
            name="Phototaxis Minimal Dip+Peak λ(t, m)",
            func=_func,
            param_names=["B", "A_dip", "A_peak", "tau", "alpha_B", "alpha_transient"],
            initial_guesses=[0.4, 0.2, 1.5, 0.2, 0.0, 0.0],
            bounds=[
                (0.01, 5.0),   # B (baseline)
                (0.0, 2.0),    # A_dip (depth below baseline at t=0)
                (0.0, 10.0),   # A_peak (peak height factor)
                (0.01, 2.0),   # tau (shared timescale for dip recovery & peak decay)
                (-0.2, 0.2),   # alpha_B
                (-0.2, 0.2)    # alpha_transient (shared plasticity for transient shape)
            ],
            latex_formula=r"$\lambda(t, m) = B e^{\alpha_B m} + \left(A_{\text{peak}} \frac{t}{\tau} - A_{\text{dip}}\right) e^{-t/\tau} e^{\alpha_{\text{transient}} m}$"
        )

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
            latex_formula=r"$\lambda(t) = B - A_{\text{dip}} e^{-t/\tau_{\text{dip}}}$",
        )

    @staticmethod
    def omr_lateral_contra() -> RateKernel:
        def _func(t, trial, params):
            B, A_dip, tau_dip = params

            dip = A_dip * np.exp(-t / tau_dip)
            return B - dip

        return RateKernel(
            name="OMR lateral contra λ(t)",
            func=_func,
            param_names=["B", "A_dip", "tau_dip"],
            initial_guesses=[0.4, 0.5, 0.5],
            bounds=[
                (0.01, 5.0), (0.0, 5.0), (0.01, 5.0)
            ],
            latex_formula=r"$\lambda(t) = B - A_{\text{dip}} e^{-t/\tau_{\text{dip}}}$",
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
                (0.0, 10.0), (0.001, 10.0), (-2.0, 2.0), 
                (t_critical-1.5, t_critical+1), (0.001, 3.0)
            ],
            latex_formula=r"$\lambda(t, m) = B + H e^{\alpha m} \exp\left(-\frac{(t - \mu)^2}{2\sigma^2}\right)$",
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
            latex_formula=r"$\lambda(t, m) = A \frac{1 + \alpha_1 m}{1 + \alpha_2 m^2} \left(\frac{t}{k\tau}\right)^k e^{k - t/\tau} + B e^{\alpha_B m}$",
        )

    
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

        fig, ax = plt.subplots(figsize=figsize)
        dt = dataset.t_grid[1] - dataset.t_grid[0]

        # 1. Empirical PSTH Rate Calculation
        if selected_trials is not None:
            trials_to_eval = np.array(selected_trials)
            trial_mask = np.isin(dataset.event_trials, trials_to_eval)
            events = dataset.event_times[trial_mask]

            # Calculate total observations matching the selected trials across all fish
            file_trials_count = dataset.binned_counts[
                dataset.binned_counts['trial_num'].isin(trials_to_eval)
            ][['file', 'trial_num']].drop_duplicates().shape[0]

            counts, bin_edges = np.histogram(events, bins=dataset.t_grid)
            psth_hz = counts / (file_trials_count * dt) if file_trials_count > 0 else np.zeros_like(counts, dtype=float)
        else:
            trials_to_eval = dataset.unique_trials
            bin_edges = dataset.t_grid
            # Use pre-computed, correctly normalized PSTH from dataset
            psth_hz = dataset.histogram_hz

        # Plot empirical histogram
        bin_centers = bin_edges[:-1]
        ax.bar(
            bin_centers, 
            psth_hz, 
            width=dt,
            color='#D3D3D3', 
            edgecolor='#A0A0A0', 
            alpha=0.7,
            label='Empirical PSTH (Histogram)', 
            align='edge'
        )

        # 2. Evaluate and Overlay Model Intensity Curves
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(fitted_models))))

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

        fig, (ax_emp, ax_mod) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        dt = dataset.t_grid[1] - dataset.t_grid[0]

        # 1. Compute Exact Un-smoothed Empirical Surface Matrix from Histogram
        num_trials = len(dataset.unique_trials)
        num_time_bins = len(dataset.t_grid) - 1
        raw_counts = np.zeros((num_trials, num_time_bins), dtype=float)

        # Map each trial to its matrix row index
        trial_to_idx = {trial: idx for idx, trial in enumerate(dataset.unique_trials)}

        # Bin events into (trial, time_bin) 2D matrix
        for t_evt, trial_evt in zip(dataset.event_times, dataset.event_trials):
            if trial_evt in trial_to_idx:
                bin_idx = np.searchsorted(dataset.t_grid, t_evt, side='right') - 1
                if 0 <= bin_idx < num_time_bins:
                    raw_counts[trial_to_idx[trial_evt], bin_idx] += 1.0

        # Calculate file-trials per trial index for exact normalization across fish
        file_trials_per_trial = (
            dataset.binned_counts.groupby('trial_num')[['file']]
            .nunique()
            .reindex(dataset.unique_trials)
            .fillna(1.0)
            .values.ravel()
        )

        # Convert counts to Rate (Hz) matrix: Shape (N_trials, N_time_bins)
        empirical_surface = raw_counts / (file_trials_per_trial[:, None] * dt)

        # 2. Compute Model Surface Matrix
        t_centers = 0.5 * (dataset.t_grid[:-1] + dataset.t_grid[1:])
        t_2d = t_centers[None, :]
        trials_2d = dataset.unique_trials[:, None]
        model_surface = model.predict(t_2d, trials_2d)

        # 3. Define Y-axis grid boundaries (edges) for pcolormesh
        # Shift boundaries by 0.5 so trial numbers align with row centers
        trial_step = np.diff(dataset.unique_trials).mean() if num_trials > 1 else 1.0
        y_grid = np.append(
            dataset.unique_trials - 0.5 * trial_step, 
            dataset.unique_trials[-1] + 0.5 * trial_step
        )

        # 4. Shared Color Scale
        vmax = max(np.max(empirical_surface), np.max(model_surface))
        vmin = 0.0

        # Panel 1: Empirical Data + Raster
        mesh_emp = ax_emp.pcolormesh(
            dataset.t_grid, 
            y_grid, 
            empirical_surface, 
            shading='flat', 
            cmap=cmap,
            vmin=vmin, 
            vmax=vmax
        )
        ax_emp.scatter(
            dataset.event_times, 
            dataset.event_trials, 
            color=raster_color, s=raster_size, alpha=raster_alpha, marker='|'
        )
        ax_emp.set_title("Empirical Surface & Raster", fontsize=12, fontweight='bold')
        ax_emp.set_xlabel("Time in Trial (s)", fontsize=11)
        ax_emp.set_ylabel("Trial Number", fontsize=11)
        ax_emp.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])

        # Panel 2: Model Surface + Raster
        mesh_mod = ax_mod.pcolormesh(
            dataset.t_grid, 
            y_grid, 
            model_surface, 
            shading='flat', 
            cmap=cmap,
            vmin=vmin, 
            vmax=vmax
        )
        ax_mod.scatter(
            dataset.event_times, 
            dataset.event_trials, 
            color=raster_color, s=raster_size, alpha=raster_alpha, marker='|'
        )
        ax_mod.set_title(f"Fitted Model: {getattr(model.kernel, 'name', 'Poisson')}", fontsize=12, fontweight='bold')
        ax_mod.set_xlabel("Time in Trial (s)", fontsize=11)
        ax_mod.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])

        # Colorbar
        divider = make_axes_locatable(ax_mod)
        cax = divider.append_axes("right", size="3%", pad=0.12)
        cbar = fig.colorbar(mesh_mod, cax=cax)
        cbar.set_label("Event Rate [Hz]", fontsize=10)

        plt.tight_layout()
        return fig, np.array([ax_emp, ax_mod])


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
                'dt':0.05, 
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
                'dt':0.05, 
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
                'window_duration':0.33
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
                'window_duration':0.33
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
                'window_duration':0.33
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
                'window_duration':0.175
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
                'dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
                'window_duration':0.175
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
                'dt':0.025, 
                't_start':0.0, 
                't_end':5.0, 
                'window_duration':0.1
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

        # A. Prepare Dataset
        dataset = loader.prepare_dataset(**config['dataset'])

        # B. Run Model Comparison
        summary_table, fitted_models = ModelComparator.compare(
            kernels=config['kernels'],
            dataset_or_t_events=dataset
        )
        
        summary_table.insert(0, "Condition", exp_name)
        all_summaries.append(summary_table)

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

        del dataset
        del fitted_models
        del best_model
        del summary_table
        gc.collect()

    master_summary_df = pd.concat(all_summaries, ignore_index=True)
    print("\n================ MASTER MODEL COMPARISON TABLE ================")
    print(master_summary_df.to_string(index=False))