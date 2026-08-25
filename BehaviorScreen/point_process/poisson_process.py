from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.special import gammaln

from .dataset import PointProcessDataset
from .point_process import PointProcess
from .kernel_shapes import peak_normalized_pulse, bounded_trial_scale

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
        cum_integral = cumulative_trapezoid(rate_surface, t_grid, initial=0.0, axis=1).squeeze()
        return np.interp(t_events, t_grid, cum_integral)



class PreyCapture:

    _PARAM_NAMES: List[str] = ["A", "tau", "B", "A_ripple", "phi1", "phi2"]
    _GUESSES: List[float] = [0.56, 1.15, 0.40, 0.1, 0.0, 0.0]
    _BOUNDS: List[Tuple[Optional[float], Optional[float]]] = [
        (0.01, 10.0),   # A: transient peak amplitude
        (0.1, 5.0),     # tau: transient decay time constant
        (0.01, 5.0),    # B: baseline rate
        (0.0, 0.49),    # A_ripple: capped at 0.49 so 2*A_ripple < 0.98,
                        #   guaranteeing ripple_mod > 0 for ANY alpha_ripple/trial
        (-np.pi, np.pi),  # phi1
        (-np.pi, np.pi),  # phi2
    ]
    _ALPHA_BOUNDS: Tuple[Optional[float], Optional[float]] = (-2.0, 2.0)
    _ALPHA_GUESS: float = -0.05
    _RIPPLE_WAVE_LATEX = r"\frac{1}{2}\left[\sin(\omega t + \phi_1) + \sin(2\omega t + \phi_2)\right]"

    @classmethod
    def _rate(
        cls,
        t: np.ndarray,
        trial: np.ndarray,
        stim_freq: float,
        A: float, tau: float, B: float, A_ripple: float, phi1: float, phi2: float,
        alpha_peak: float = 0.0,
        alpha_baseline: float = 0.0,
        alpha_ripple: float = 0.0,
    ) -> np.ndarray:

        w = 2.0 * np.pi * stim_freq
        phase = w * t

        transient = A * np.exp(-t / tau) * np.exp(alpha_peak * trial)
        baseline = B * np.exp(alpha_baseline * trial)

        wave = 0.5 * (np.sin(phase + phi1) + np.sin(2.0 * phase + phi2))
        ripple_amplitude = A_ripple * bounded_trial_scale(trial, alpha_ripple)
        ripple_mod = 1.0 + ripple_amplitude * wave

        return (transient + baseline) * ripple_mod

    # -- Variants ----------------------------------------------------------


    @classmethod
    def time_only(cls, stim_freq: float) -> RateKernel:
        """No trial-dependent plasticity."""
        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2 = params
            return cls._rate(t, trial, stim_freq, A, tau, B, A_ripple, phi1, phi2)

        return RateKernel(
            name="PreyCapture(TimeOnly)",
            func=_func,
            param_names=list(cls._PARAM_NAMES),
            initial_guesses=list(cls._GUESSES),
            bounds=list(cls._BOUNDS),
            latex_formula=(
                r"$\lambda(t) = \left(A e^{-t/\tau} + B\right)"
                rf"\left(1 + A_{{\text{{ripple}}}} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak(cls, stim_freq: float) -> RateKernel:
        """Only the transient (peak) amplitude is modulated across trials."""
        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2, alpha_peak = params
            return cls._rate(t, trial, stim_freq, A, tau, B, A_ripple, phi1, phi2,
                              alpha_peak=alpha_peak)

        return RateKernel(
            name="PreyCapture(Peak)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_peak"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} e^{\alpha_{\text{peak}} m} + B\right)"
                rf"\left(1 + A_{{\text{{ripple}}}} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def baseline(cls, stim_freq: float) -> RateKernel:
        """Only the tonic baseline is modulated across trials."""
        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2, alpha_baseline = params
            return cls._rate(t, trial, stim_freq, A, tau, B, A_ripple, phi1, phi2,
                              alpha_baseline=alpha_baseline)

        return RateKernel(
            name="PreyCapture(Baseline)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_baseline"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} + B e^{\alpha_{\text{base}} m}\right)"
                rf"\left(1 + A_{{\text{{ripple}}}} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline(cls, stim_freq: float) -> RateKernel:
        """Peak and baseline each independently modulated across trials."""
        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2, alpha_peak, alpha_baseline = params
            return cls._rate(t, trial, stim_freq, A, tau, B, A_ripple, phi1, phi2,
                              alpha_peak=alpha_peak, alpha_baseline=alpha_baseline)

        return RateKernel(
            name="PreyCapture(Peak_Baseline)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_peak", "alpha_baseline"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS, cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS, cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} e^{\alpha_{\text{peak}} m}"
                r" + B e^{\alpha_{\text{base}} m}\right)"
                rf"\left(1 + A_{{\text{{ripple}}}} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline_ripple(cls, stim_freq: float) -> RateKernel:
        """Peak, baseline, and ripple depth each independently modulated (fully general)."""
        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2, alpha_peak, alpha_baseline, alpha_ripple = params
            return cls._rate(t, trial, stim_freq, A, tau, B, A_ripple, phi1, phi2,
                              alpha_peak=alpha_peak, alpha_baseline=alpha_baseline,
                              alpha_ripple=alpha_ripple)

        return RateKernel(
            name="PreyCapture(Peak_Baseline_Ripple)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_peak", "alpha_baseline", "alpha_ripple"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS] * 3,
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS] * 3,
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} e^{\alpha_{\text{peak}} m}"
                r" + B e^{\alpha_{\text{base}} m}\right)"
                r"\left(1 + \frac{2A_{\text{ripple}}}{1+e^{-\alpha_{\text{ripple}} m}}"
                rf"{cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline_shared(cls, stim_freq: float) -> RateKernel:
        """Peak and baseline share a single modulation parameter; ripple unmodulated."""
        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2, alpha_shared = params
            return cls._rate(t, trial, stim_freq, A, tau, B, A_ripple, phi1, phi2,
                              alpha_peak=alpha_shared, alpha_baseline=alpha_shared)

        return RateKernel(
            name="PreyCapture(Peak_Baseline_Shared)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_shared"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} + B\right) e^{\alpha_{\text{shared}} m}"
                rf"\left(1 + A_{{\text{{ripple}}}} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline_shared_ripple(cls, stim_freq: float) -> RateKernel:
        """Peak & baseline share one modulation parameter; ripple has its own, separate one."""
        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2, alpha_shared, alpha_ripple = params
            return cls._rate(t, trial, stim_freq, A, tau, B, A_ripple, phi1, phi2,
                              alpha_peak=alpha_shared, alpha_baseline=alpha_shared,
                              alpha_ripple=alpha_ripple)

        return RateKernel(
            name="PreyCapture(Peak_Baseline_Shared_Ripple)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_shared", "alpha_ripple"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS, cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS, cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} + B\right) e^{\alpha_{\text{shared}} m}"
                r"\left(1 + \frac{2A_{\text{ripple}}}{1+e^{-\alpha_{\text{ripple}} m}}"
                rf"{cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline_ripple_shared(cls, stim_freq: float) -> RateKernel:
        """Peak, baseline, and ripple all share a single modulation parameter."""
        def _func(t, trial, params):
            A, tau, B, A_ripple, phi1, phi2, alpha_shared = params
            return cls._rate(t, trial, stim_freq, A, tau, B, A_ripple, phi1, phi2,
                              alpha_peak=alpha_shared, alpha_baseline=alpha_shared,
                              alpha_ripple=alpha_shared)

        return RateKernel(
            name="PreyCapture(Peak_Baseline_Ripple_Shared)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_shared"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} + B\right)"
                r"\left(1 + \frac{2A_{\text{ripple}}}{1+e^{-\alpha_{\text{shared}} m}}"
                rf"{cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )
    

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

            # bounded_trial_scale(trial=0, alpha) == 1.0 by construction, so
            # effective_f_dip(trial=0) == f_dip exactly (matches PreyCapture's
            # ripple_amplitude convention). bounded_trial_scale is always in
            # (0, 2), so effective_f_dip is always in (0, 2*f_dip).
            effective_f_dip = f_dip * bounded_trial_scale(trial, alpha_dip)

            # Guaranteed positive for all t >= 0: effective_f_dip < 2*f_dip <= 0.98,
            # so dip_factor > 1 - 0.98 = 0.02 > 0.
            dip_factor = 1.0 - effective_f_dip * np.exp(-t / tau_dip)

            return mod_B * dip_factor

        return RateKernel(
            name="Phototaxis Contra λ(t, m)",
            func=_func,
            param_names=["B", "f_dip", "tau_dip", "alpha_B", "alpha_dip"],
            initial_guesses=[0.4, 0.5, 0.5, 0.0, 0.0],
            bounds=[
                (0.01, 10.0),  # B: baseline rate
                (0.0, 0.49),   # f_dip: dip depth AT TRIAL 0. Capped at 0.49 so
                            #   2*f_dip < 0.98, guaranteeing dip_factor > 0
                            #   for ANY alpha_dip/trial (mirrors PreyCapture's
                            #   A_ripple bound).
                (0.01, 5.0),   # tau_dip: decay time constant
                (-0.1, 0.1),   # alpha_B: baseline modulation across trials
                (-0.2, 0.2),   # alpha_dip: dip-depth modulation rate across trials
            ],
            latex_formula=(
                r"$\lambda(t, m) = B e^{\alpha_B m}\left(1 - f_{\text{dip}}\,"
                r"\frac{2}{1+e^{-\alpha_{\text{dip}} m}}\, e^{-t/\tau_{\text{dip}}}\right)$"
            ),
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
            time_pulse = peak_normalized_pulse(t, t_peak, k)
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
        sum_log_rates = np.sum(np.log(event_rates))

        # Term 2: Expected Total Events (Surface Integration over Time)
        trial_integrals = self.kernel.integrate(
            duration_s=dataset.duration_s, 
            trial=np.arange(dataset.num_trials), 
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
        return expected_rate

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")

        t_2d = dataset.t_centers[None, :]
        trials_2d = np.arange(dataset.num_trials)[:, None]
        expected_rate = self.predict(t_2d, trials_2d)
        
        return expected_rate 
    
    def cumulative_integrated_intensity(
        self,
        t_events: np.ndarray,
        trial: float,
    ) -> np.ndarray:
        
        return self.kernel.cumulative_integrate(
            t_events=t_events,
            trial=trial,
            params=self.params_,
            integration_dt=self.integration_dt,
        )
    


class GammaPoissonProcess(PointProcess):
    """
    A PoissonProcess extended with fish-level rate heterogeneity:

        rate for fish f  =  g_f * lambda_base(t, m)
        g_f ~ Gamma(r, r)   (E[g_f] = 1, Var[g_f] = 1/r), independent per fish

    Conditional on g_f, each fish's events are an ordinary (inhomogeneous)
    Poisson process -- exactly PoissonProcess's model. g_f is never observed,
    so it is integrated out analytically; the resulting MARGINAL distribution
    of event counts is Negative Binomial (hence "Gamma-Poisson"), which is
    why this class needs only ONE extra parameter (r) regardless of how many
    fish are in the dataset, rather than one parameter per fish.

    r -> infinity recovers PoissonProcess exactly (no heterogeneity).
    Smaller r means more between-fish heterogeneity; 1/r is a natural
    "overdispersion index" on the same footing as PointProcessDataset's
    dispersion_fano_ratio diagnostic.

    LIMITATION: predict(), compute_expected_rate(), and
    cumulative_integrated_intensity() all describe the POPULATION-AVERAGE
    process (E[g_f] = 1), not any specific fish's rate. PointProcess.
    time_rescaling() calls cumulative_integrated_intensity() per (fish,
    trial) stream without fish-specific rescaling, so its output for this
    class reflects average-fish calibration, not per-fish calibration.
    Use estimate_fish_gains() + rescale manually if per-fish time-rescaling
    diagnostics are needed (see note on that method below).
    """

    def __init__(self, kernel: RateKernel, integration_dt: float = 0.02, r_init: float = 5.0):
        super().__init__(integration_dt)

        self.name = f"GammaPoisson {kernel.name}"
        self.latex_formula = kernel.latex_formula
        self.kernel = kernel
        self.initial_guesses = kernel.initial_guesses + [r_init]
        self.bounds = kernel.bounds + [(1e-3, None)]  # r > 0
        self.param_names = kernel.param_names + ["r_dispersion"]

    def _split_params(self, params: List[float]) -> Tuple[List[float], float]:
        return params[:-1], params[-1]

    def _fish_sufficient_stats(
        self, dataset: PointProcessDataset, kernel_params: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes, for each fish f:
            N_f = total observed event count across all of f's trials
            S_f = total EXPECTED count under lambda_base across f's trials
                  (i.e. sum of kernel.integrate(...) over the trials f
                  actually participated in, per fish_trial_mask)

        These are the only quantities the NB marginal likelihood needs
        per fish -- no per-fish parameters are fit.
        """
        trial_integrals = self.kernel.integrate(
            duration_s=dataset.duration_s,
            trial=np.arange(dataset.num_trials),
            params=kernel_params,
            integration_dt=self.integration_dt,
        )  # shape (num_trials,)

        N_f = np.zeros(dataset.num_fish)
        S_f = np.zeros(dataset.num_fish)
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            N_f[f_idx] += len(t_ev)
            S_f[f_idx] += trial_integrals[t_idx]

        return N_f, S_f

    def _fish_scale_factors(self, dataset: PointProcessDataset) -> np.ndarray:
        """
        Posterior-mean gain estimate per fish: (r + N_f) / (r + S_f).
        Used both here (to correct time_rescaling) and in estimate_fish_gains
        (single source of truth for the formula).
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        kernel_params, r = self._split_params(self.params_)
        N_f, S_f = self._fish_sufficient_stats(dataset, kernel_params)
        return (r + N_f) / (r + S_f)
    
    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        kernel_params, r = self._split_params(params)
        r = max(r, 1e-8)  # guard against optimizer probing r <= 0 despite bounds

        # Shape term: identical in form to PoissonProcess -- this is the
        # part of the likelihood that is unaffected by fish heterogeneity,
        # since conditional on g_f each fish is still ordinary Poisson.
        # Uses actual trial VALUES (not positional indices) for evaluate(),
        # matching HawkesProcess's convention.
        event_rates = self.kernel.evaluate(dataset.event_times, dataset.event_trials_idx, kernel_params)
        event_rates = np.maximum(event_rates, 1e-9)
        shape_term = np.sum(np.log(event_rates))

        # Fish-level Negative Binomial term, replacing PoissonProcess's
        # plain "-sum(Lambda * n_fish)" count term.
        N_f, S_f = self._fish_sufficient_stats(dataset, kernel_params)
        nb_term = np.sum(
            r * np.log(r) - gammaln(r) + gammaln(N_f + r) - (N_f + r) * np.log(S_f + r)
        )

        return -(shape_term + nb_term)

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray]) -> np.ndarray:
        """Population-average rate (E[g_f] = 1); does not reflect any single fish's gain."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        kernel_params, _ = self._split_params(self.params_)
        expected_rate = self.kernel.evaluate(t, trial, kernel_params)
        return np.maximum(expected_rate, 1e-9)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        t_2d = dataset.t_centers[None, :]
        trials_2d = np.arange(dataset.num_trials)[:, None]
        return self.predict(t_2d, trials_2d)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        """Population-average cumulative intensity; see class docstring limitation."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        kernel_params, _ = self._split_params(self.params_)
        return self.kernel.cumulative_integrate(
            t_events=t_events, trial=trial, params=kernel_params,
            integration_dt=self.integration_dt,
        )

    # -- Heterogeneity-specific extras ---------------------------------------

    @property
    def dispersion_r(self) -> float:
        """Fitted r (Gamma shape/rate parameter). Larger r = less heterogeneity."""
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        return float(self.params_[-1])

    @property
    def overdispersion_index(self) -> float:
        """1/r: Var[g_f] under the fitted Gamma population distribution. 0 = no heterogeneity."""
        return 1.0 / self.dispersion_r

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        """Posterior mean gain per fish, from Gamma-Poisson conjugacy. See _fish_scale_factors."""
        if self.params_ is None:
            raise ValueError("Model must be fitted before estimating fish gains.")

        kernel_params, r = self._split_params(self.params_)
        N_f, S_f = self._fish_sufficient_stats(dataset, kernel_params)
        g_hat = self._fish_scale_factors(dataset)
        active = dataset.fish_trial_mask.any(axis=1)

        return pd.DataFrame({
            "fish_idx": np.arange(dataset.num_fish)[active],
            "n_events": N_f[active],
            "expected_events_base": S_f[active],
            "estimated_gain": g_hat[active],
        })