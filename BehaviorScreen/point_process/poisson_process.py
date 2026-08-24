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

def _peak_normalized_pulse(x: np.ndarray, x_peak: float, k: float = 1.0) -> np.ndarray:
    """
    Generalized alpha-function / Gamma-shaped pulse, peak-normalized to height 1.

        f(x) = (x / x_peak)^k * exp(k * (1 - x / x_peak)),   x >= 0

    - f(0) = 0
    - f(x_peak) = 1   <- peak location & height fixed by construction
    - k=1: classic alpha function (linear rise, exponential decay)
    - k>1: sharper / more symmetric peak
    - k<1: fast rise, long tail (asymmetric)

    Non-negative for x >= 0, k > 0. No clamping required.
    """
    x_safe = np.maximum(x, 0.0)
    ratio = x_safe / x_peak
    return np.power(ratio, k) * np.exp(k * (1.0 - ratio))


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

    @staticmethod
    def phototaxis_ipsi() -> RateKernel:
        def _func(t, trial, params):
            B, f_dip, A_peak, tau, alpha_B, alpha_peak = params

            mod_B = B * np.exp(alpha_B * trial)
            mod_peak = A_peak * np.exp(alpha_peak * trial)

            # Guaranteed positive dip component
            dip_component = mod_B * (1.0 - f_dip * np.exp(-t / tau))
            # Non-negative peak component for t >= 0
            peak_component = mod_peak * (t / tau) * np.exp(-t / tau)

            return dip_component + peak_component

        return RateKernel(
            name="Phototaxis Minimal Dip+Peak λ(t, m)",
            func=_func,
            param_names=["B", "f_dip", "A_peak", "tau", "alpha_B", "alpha_peak"],
            initial_guesses=[0.4, 0.5, 1.5, 0.2, 0.0, 0.0],
            bounds=[
                (0.01, 5.0),   # B
                (0.0, 0.99),   # f_dip in [0, 0.99) guarantees positivity at all t
                (0.0, 10.0),   # A_peak
                (0.01, 2.0),   # tau
                (-0.2, 0.2),   # alpha_B
                (-0.2, 0.2)    # alpha_peak
            ],
            latex_formula=r"$\lambda(t, m) = B e^{\alpha_B m} \left(1 - f_{\text{dip}} e^{-t/\tau}\right) + A_{\text{peak}} e^{\alpha_{\text{peak}} m} \left(\frac{t}{\tau}\right) e^{-t/\tau}$"
        )

    @staticmethod
    def phototaxis_contra() -> RateKernel:
        def _func(t, trial, params):
            B, f_dip, tau_dip, alpha_B, alpha_dip = params

            mod_B = B * np.exp(alpha_B * trial)
            
            # Sigmoid modulation ensures the trial-scaling factor stays in (0, 1)
            # trial=0 corresponds to standard f_dip depth
            sig_trial = 2.0 / (1.0 + np.exp(-alpha_dip * trial))  # sig_trial(0) = 1.0
            effective_f_dip = f_dip * (sig_trial / 2.0)          # strictly < 0.99
            
            # dip_factor is mathematically strictly > 0 for all t >= 0 and trial >= 0
            dip_factor = 1.0 - effective_f_dip * np.exp(-t / tau_dip)

            return mod_B * dip_factor

        return RateKernel(
            name="Phototaxis Contra λ(t, m)",
            func=_func,
            param_names=["B", "f_dip", "tau_dip", "alpha_B", "alpha_dip"],
            initial_guesses=[0.4, 0.5, 0.5, 0.0, 0.0],
            bounds=[
                (0.01, 10.0), # B: baseline rate
                (0.0, 0.98),  # f_dip: bound below 1.0
                (0.01, 5.0),  # tau_dip: decay time constant
                (-0.1, 0.1),  # alpha_B: baseline modulation across trials
                (-0.2, 0.2),  # alpha_dip: dip depth modulation rate across trials
            ],
            latex_formula=r"$\lambda(t, m) = B e^{\alpha_B m} \left(1 - \frac{f_{\text{dip}}}{1 + e^{-\alpha_{\text{dip}} m}} e^{-t/\tau_{\text{dip}}}\right)$",
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


    @staticmethod
    def dark_flash() -> RateKernel:
        def _func(t, trial, params):
            A, t_peak, k, B, alpha_B, tau_decay = params
            time_pulse = _peak_normalized_pulse(t, t_peak, k)
            trial_scale = np.exp(-trial/tau_decay)
            baseline = B * np.exp(alpha_B * trial)
            return A * time_pulse * trial_scale + baseline

        return RateKernel(
            name="Dark Flash",
            func=_func,
            param_names=["A", "t_peak", "k", "B", "alpha_B", "tau_decay"],
            initial_guesses=[2.2, 0.108, 5.1, 0.02, -0.18, 3.0],
            bounds=[(0.0, None), (0.005, 2.0), (0.5, 20.0), (0.001, 10.0), (-1.0, 1.0), (0.1, 20.0)],
            latex_formula=(
                r"$\lambda(t,m) = A \left(\frac{t}{t_{\text{peak}}}\right)^{k}"
                r"e^{k(1-t/t_{\text{peak}})} e^{-m/\tau_{\text{decay}}} + B e^{\alpha_B m}$"
            ),
        )

class PoissonProcess(PointProcess):

    def __init__(self, kernel: RateKernel, integration_dt: float = 0.02):

        super().__init__(integration_dt)

        self.name = f"Poisson {kernel.name}"
        self.latex_formula = kernel.latex_formula
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

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")

        t_2d = dataset.t_centers[None, :]
        trials_2d = dataset.unique_trials[:, None]
        expected_rate = self.predict(t_2d, trials_2d)
        #expected_rate = np.maximum(expected_rate, 1e-9)
        
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
    


