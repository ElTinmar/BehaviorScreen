from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid

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
    """
    λ(t, m) = [A·e^(−t/τ)·e^(α_peak m) + B·e^(α_baseline m)]
              · [1 + A_ripple·s(m, α_ripple)·wave(t)]
    wave(t) = ½[sin(ωt+φ1) + sin(2ωt+φ2)] ∈ [-1, 1]
    s(m, α) = bounded_trial_scale(m, α) ∈ (0, 2)

    No reparametrization needed for A/B -- already conform to convention.

    Parameters
    ----------
    A : Hz. LITERAL height of the onset-burst at t=0, trial m=0 (the
        exponential decay means this IS the true peak, unlike
        phototaxis_ipsi's old A_peak). Comparable in Hz to B, H, etc.
    tau : seconds. Onset-burst decay time constant. Not comparable across
        kernels.
    B : Hz. Sustained/tonic rate during ongoing stimulus, trial m=0.
        Comparable in Hz to other kernels' baseline terms.
    A_ripple : dimensionless, in (0, 0.49). NOT an amplitude in Hz --
        this is the depth of an oscillatory modulation phase-locked to
        the stimulus's own motion. Do not compare to A/B/H; it answers a
        qualitatively different question (bout-timing coordination with
        the stimulus, not response magnitude).
    phi1, phi2 : radians. Phase offsets of the ripple's fundamental and
        2nd harmonic. Timing-coordination parameters, not amplitudes.
    alpha_peak, alpha_baseline : 1/trial (log-scale). Independent
        trial-modulation of A and B respectively.
    alpha_ripple : 1/trial. Modulates A_ripple via the same saturating
        logistic as phototaxis_contra's alpha_dip -- bounded effect,
        unlike alpha_peak/alpha_baseline's unbounded exponential.
    """

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
                r"$\lambda(t,m) = \left(A e^{-t/\tau} + B\right) e^{\alpha_{\text{shared}} m}"
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
        """
        λ(t, m) = B·e^(α_B m)·(1 − f_dip·e^(−t/τ)) + A_peak·e^(α_peak m)·e·(t/τ)·e^(−t/τ)

        Parameters
        ----------
        B : Hz. Tonic/asymptotic rate at trial m=0. Comparable in Hz across
            kernels.
        f_dip : dimensionless fraction, in [0, 0.99). Depth of onset
            suppression AS A FRACTION OF B, at trial m=0. Structurally
            guarantees rate(t=0, m=0) = B*(1-f_dip) > 0 for ANY B > 0 -- no
            runtime clamping needed, unlike an independently-fit Hz depth.
            For a Hz-comparable "dip depth" (e.g. to compare against other
            kernels' amplitude terms), compute the DERIVED quantity
            dip_depth_hz = f_dip * B post-fit (see dip_depth_hz() below) --
            do not fit this product directly.
        tau : seconds. Shared time constant for dip recovery AND peak-bump
            dynamics -- not comparable across kernels; coupled, see caveat
            in earlier discussion.
        A_peak : Hz. Literal peak height of the delayed excitatory bump above
            the dip-recovery curve, at t=tau, trial m=0 (peak-normalized).
        alpha_B : 1/trial (log-scale). Trial-modulation of B (dip term scales
            proportionally, since it's a fixed fraction of B).
        alpha_peak : 1/trial (log-scale). Trial-modulation of A_peak.
        """
        def _func(t, trial, params):
            B, f_dip, tau, A_peak, alpha_B, alpha_peak = params

            mod_B = B * np.exp(alpha_B * trial)
            mod_peak = A_peak * np.exp(alpha_peak * trial)

            dip_component = mod_B * (1.0 - f_dip * np.exp(-t / tau))
            peak_component = mod_peak * np.e * (t / tau) * np.exp(-t / tau)

            return dip_component + peak_component

        return RateKernel(
            name="Phototaxis Minimal Dip+Peak λ(t, m)",
            func=_func,
            param_names=["B", "f_dip", "tau", "A_peak", "alpha_B", "alpha_peak"],
            initial_guesses=[0.4, 0.5, 0.2, 0.55, 0.0, 0.0],
            bounds=[
                (0.01, 5.0),   # B (Hz)
                (0.0, 0.99),   # f_dip -- guarantees positivity for ANY B
                (0.01, 2.0),   # tau (s)
                (0.0, 27.0),   # A_peak (Hz)
                (-0.2, 0.2),   # alpha_B
                (-0.2, 0.2),   # alpha_peak
            ],
            latex_formula=(
                r"$\lambda(t, m) = B e^{\alpha_B m} \left(1 - f_{\text{dip}} e^{-t/\tau}\right)"
                r" + A_{\text{peak}} e^{\alpha_{\text{peak}} m}\, e\,(t/\tau) e^{-t/\tau}$"
            ),
        )

    @staticmethod
    def phototaxis_contra() -> RateKernel:
        """
        λ(t, m) = B·e^(α_B m)·(1 − f_dip·s(m, α_dip)·e^(−t/τ_dip))
        s(m, α) = bounded_trial_scale(m, α) ∈ (0, 2), s(0, α) = 1

        Parameters
        ----------
        B : Hz. Tonic rate at trial m=0.
        f_dip : dimensionless, in [0, 0.49). Dip depth as a fraction of B at
            trial m=0. Capped at 0.49 (not 0.99) so that even at s(m)=2 (its
            max), 2*f_dip < 0.98, structurally guaranteeing positivity for
            ANY trial/alpha_dip combination -- no clamp needed.
        tau_dip : seconds. Dip recovery time constant.
        alpha_B : 1/trial (log-scale). Trial-modulation of B.
        alpha_dip : 1/trial. Modulates dip depth via a saturating logistic --
            bounded effect (max 2x), unlike alpha_B's unbounded exponential.
        """
        def _func(t, trial, params):
            B, f_dip, tau_dip, alpha_B, alpha_dip = params

            mod_B = B * np.exp(alpha_B * trial)
            effective_f_dip = f_dip * bounded_trial_scale(trial, alpha_dip)
            dip_factor = 1.0 - effective_f_dip * np.exp(-t / tau_dip)

            return mod_B * dip_factor

        return RateKernel(
            name="Phototaxis Contra λ(t, m)",
            func=_func,
            param_names=["B", "f_dip", "tau_dip", "alpha_B", "alpha_dip"],
            initial_guesses=[0.4, 0.5, 0.5, 0.0, 0.0],
            bounds=[
                (0.01, 10.0),
                (0.0, 0.49),
                (0.01, 5.0),
                (-0.1, 0.1),
                (-0.2, 0.2),
            ],
            latex_formula=(
                r"$\lambda(t, m) = B e^{\alpha_B m}\left(1 - f_{\text{dip}}\,"
                r"\frac{2}{1+e^{-\alpha_{\text{dip}} m}}\, e^{-t/\tau_{\text{dip}}}\right)$"
            ),
        )
 
    @staticmethod
    def omr_forward() -> RateKernel:
        """
        λ(t) = B·(1 − f_dip·e^(−t/τ_dip))

        Parameters
        ----------
        B : Hz. Tonic rate.
        f_dip : dimensionless, in [0, 0.99). Dip depth as a fraction of B.
            Structurally positive for any B; no trial modulation currently.
        tau_dip : seconds. Dip recovery time constant.
        """
        def _func(t, trial, params):
            B, f_dip, tau_dip = params
            return B * (1.0 - f_dip * np.exp(-t / tau_dip))

        return RateKernel(
            name="OMR forward λ(t)",
            func=_func,
            param_names=["B", "f_dip", "tau_dip"],
            initial_guesses=[0.4, 0.5, 0.5],
            bounds=[(0.01, 5.0), (0.0, 0.99), (0.01, 5.0)],
            latex_formula=r"$\lambda(t) = B \left(1 - f_{\text{dip}} e^{-t/\tau_{\text{dip}}}\right)$",
        )


    @staticmethod
    def omr_lateral_contra() -> RateKernel:
        """Identical parameterization/convention to omr_forward -- see there."""
        def _func(t, trial, params):
            B, f_dip, tau_dip = params
            return B * (1.0 - f_dip * np.exp(-t / tau_dip))

        return RateKernel(
            name="OMR lateral contra λ(t)",
            func=_func,
            param_names=["B", "f_dip", "tau_dip"],
            initial_guesses=[0.4, 0.5, 0.5],
            bounds=[(0.01, 5.0), (0.0, 0.99), (0.01, 5.0)],
            latex_formula=r"$\lambda(t) = B \left(1 - f_{\text{dip}} e^{-t/\tau_{\text{dip}}}\right)$",
        )
            
    @staticmethod
    def looming_gaussian(t_critical: float = 5.0) -> RateKernel:
        """
        λ(t, m) = B + H·e^(α m)·exp(−(t−μ)²/2σ²)

        No reparametrization needed -- already conforms to convention.

        Parameters
        ----------
        B : Hz. Additive tonic floor (present at all t, unaffected by the
            burst). Comparable in Hz to other kernels' baseline terms.
        H : Hz. LITERAL peak height of the burst above B, at t=mu, trial m=0.
            Directly comparable to dark_flash's A and phototaxis_ipsi's
            A_peak.
        mu : seconds. Latency of peak response (not a rate parameter).
        sigma : seconds. Width (std) of the burst. Not comparable across
            kernels.
        alpha : 1/trial (log-scale). Trial-modulation of H only.
        """
        def _func(t, trial, params):
            B, H, alpha, mu, sigma = params
            height = H * np.exp(alpha * trial)
            exponent = -0.5 * ((t - mu) / sigma) ** 2
            return B + (height * np.exp(exponent))

        return RateKernel(
            name="Looming Gaussian λ(t, m)",
            func=_func,
            param_names=["B", "H", "alpha", "mu", "sigma"],
            initial_guesses=[0.1, 1.2, 0.0, 5.0, 0.1],
            bounds=[
                (0.001, 10.0), (0.001, 10.0), (-2.0, 2.0),
                (t_critical - 1.5, t_critical + 1), (0.001, 3.0)
            ],
            latex_formula=r"$\lambda(t, m) = B + H e^{\alpha m} \exp\left(-\frac{(t - \mu)^2}{2\sigma^2}\right)$",
        )

    @staticmethod
    def dark_flash_smooth() -> RateKernel:
        """
        λ(t, m) = A·pulse(t; t_peak, k)·e^(−m/τ_hab) + B·e^(α_B m)
        pulse(x; x_peak, k) peak-normalized to 1 at x=x_peak (see kernel_shapes)

        No reparametrization needed for A/B -- already conforms to convention.
        Renamed tau_decay -> tau_habituation (pure rename, same values/units)
        to avoid confusion with time-domain tau parameters in other kernels:
        this tau is in TRIALS, not seconds, and governs how fast the flash
        burst amplitude A shrinks across the session -- it's a trial-
        plasticity parameter in disguise, not a within-trial dynamic.

        Parameters
        ----------
        A : Hz. LITERAL peak height of the flash-evoked burst, at t=t_peak,
            trial m=0. Directly comparable to looming's H.
        t_peak : seconds. Latency to peak.
        k : dimensionless. Pulse shape/sharpness -- not comparable across
            kernels.
        B : Hz. Additive baseline at trial m=0.
        alpha_B : 1/trial (log-scale). Trial-modulation of B.
        tau_habituation : TRIALS (not seconds). Time constant for A's decay
            across the session: A(m) = A * exp(-m / tau_habituation).
        """
        def _func(t, trial, params):
            A, t_peak, k, B, alpha_B, tau_habituation = params
            time_pulse = peak_normalized_pulse(t, t_peak, k)
            trial_scale = np.exp(-trial / tau_habituation)
            baseline = B * np.exp(alpha_B * trial)
            return A * time_pulse * trial_scale + baseline

        return RateKernel(
            name="Dark Flash",
            func=_func,
            param_names=["A", "t_peak", "k", "B", "alpha_B", "tau_habituation"],
            initial_guesses=[2.2, 0.108, 5.1, 0.02, -0.18, 3.0],
            bounds=[(0.0, None), (0.005, 2.0), (0.5, 20.0), (0.001, 10.0), (-1.0, 1.0), (0.1, 20.0)],
            latex_formula=(
                r"$\lambda(t,m) = A \left(\frac{t}{t_{\text{peak}}}\right)^{k}"
                r"e^{k(1-t/t_{\text{peak}})} e^{-m/\tau_{\text{hab}}} + B e^{\alpha_B m}$"
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
    
    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:

        event_rates = self.kernel.evaluate(
            dataset.event_times, dataset.event_trials_idx, params
        )
        base_ll = float(np.sum(np.log(event_rates)))

        # N_f: observed event count per fish
        N_f = np.zeros(dataset.num_fish, dtype=float)
        np.add.at(N_f, dataset.event_fish_idx, 1.0)

        # S_f: per-fish expected count under the baseline kernel
        trial_integrals = self.kernel.integrate(
            duration_s=dataset.duration_s,
            trial=np.arange(dataset.num_trials),
            params=params,
            integration_dt=self.integration_dt,
        )  # shape (num_trials,)

        S_f = dataset.fish_trial_mask.astype(float) @ trial_integrals  # shape (num_fish,)

        return base_ll, N_f, S_f

