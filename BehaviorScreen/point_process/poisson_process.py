from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid

from .dataset import PointProcessDataset
from .point_process import PointProcess
from .kernel_shapes import exgaussian_shape, bounded_trial_scale, sigmoid_bounded, logit_bounded

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
              · [1 + depth(m)·wave(t)]
    wave(t)  = ½[sin(ωt+φ1) + sin(2ωt+φ2)] ∈ [-1, 1]
    depth(m) = sigmoid_bounded(z0_ripple + α_ripple·m, RIPPLE_UPPER)
             = RIPPLE_UPPER / (1 + e^{-(z0_ripple + α_ripple m)})  ∈ (0, RIPPLE_UPPER)

    No reparametrization needed for A/B -- already conform to convention.

    REPARAMETRIZATION NOTE: z0_ripple/alpha_ripple replace the old
    (A_ripple, alpha_ripple) pair, where A_ripple was box-bounded to
    (0, 0.49) purely to guarantee A_ripple*bounded_trial_scale(m,.) < 1 for
    the WORST-CASE trial scale factor (max 2x) -- an artificial ceiling on
    A_ripple itself, not a real belief about ripple depth. Bootstrap fits
    were pinning exactly at that wall (fitted=0.49, CI collapsed to a point).
    depth(m) is now guaranteed in (0, RIPPLE_UPPER) for ANY (z0_ripple,
    alpha_ripple, m) via a logistic -- no worst-case analysis, no artificial
    margin below the true limit. Positional argument order is unchanged, so
    every classmethod below only needed a rename + formula fix, not a logic
    change.

    Parameters
    ----------
    A : Hz. LITERAL height of the onset-burst at t=0, trial m=0 (the
        exponential decay means this IS the true peak, unlike
        phototaxis_ipsi's old A_peak). Comparable in Hz to B, H, etc.
    tau : seconds. Onset-burst decay time constant. Not comparable across
        kernels.
    B : Hz. Sustained/tonic rate during ongoing stimulus, trial m=0.
        Comparable in Hz to other kernels' baseline terms.
    z0_ripple : unconstrained (logit scale). NOT an amplitude or a
        probability -- depth(m=0) = sigmoid_bounded(z0_ripple, RIPPLE_UPPER).
        Recover the interpretable depth-at-trial-0 post-fit via
        sigmoid_bounded(z0_ripple, RIPPLE_UPPER); do not interpret z0_ripple
        itself. Do not compare to A/B/H; it answers a qualitatively
        different question (bout-timing coordination with the stimulus, not
        response magnitude).
    phi1, phi2 : radians. Phase offsets of the ripple's fundamental and
        2nd harmonic. Timing-coordination parameters, not amplitudes.
    alpha_peak, alpha_baseline : 1/trial (log-scale). Independent
        trial-modulation of A and B respectively (unbounded exponential).
    alpha_ripple : 1/trial, acting on the LOGIT scale (additive to
        z0_ripple, not multiplicative on a probability) -- bounded effect on
        depth(m) via the same saturating logistic as z0_ripple itself.
    """

    _PARAM_NAMES: List[str] = ["A", "tau", "B", "z0_ripple", "phi1", "phi2"]
    _RIPPLE_UPPER: float = 0.95
    _GUESSES: List[float] = [0.56, 1.15, 0.40, float(logit_bounded(0.1, _RIPPLE_UPPER)), 0.0, 0.0]
    _BOUNDS: List[Tuple[Optional[float], Optional[float]]] = [
        (0.01, 10.0),   # A: transient peak amplitude
        (0.1, 5.0),     # tau: transient decay time constant
        (0.01, 5.0),    # B: baseline rate
        (-15.0, 15.0),  # z0_ripple: unconstrained logit scale
        (-np.pi, np.pi),  # phi1
        (-np.pi, np.pi),  # phi2
    ]
    _ALPHA_BOUNDS: Tuple[Optional[float], Optional[float]] = (-2.0, 2.0)
    _ALPHA_RIPPLE_BOUNDS: Tuple[Optional[float], Optional[float]] = (-4.0, 4.0)
    _ALPHA_GUESS: float = -0.05
    _RIPPLE_WAVE_LATEX = r"\frac{1}{2}\left[\sin(\omega t + \phi_1) + \sin(2\omega t + \phi_2)\right]"

    @classmethod
    def _depth_latex(cls, alpha_symbol: Optional[str] = None) -> str:
        """Shared LaTeX snippet for depth(m), used by every variant below so
        the formula always matches _rate's actual sigmoid_bounded computation."""
        u = cls._RIPPLE_UPPER
        if alpha_symbol is None:
            return rf"\frac{{{u}}}{{1+e^{{-z_{{0,\text{{ripple}}}}}}}}"
        return rf"\frac{{{u}}}{{1+e^{{-(z_{{0,\text{{ripple}}}}+\alpha_{{{alpha_symbol}}} m)}}}}"

    @classmethod
    def _rate(
        cls,
        t: np.ndarray,
        trial: np.ndarray,
        stim_freq: float,
        A: float, tau: float, B: float, z0_ripple: float, phi1: float, phi2: float,
        alpha_peak: float = 0.0,
        alpha_baseline: float = 0.0,
        alpha_ripple: float = 0.0,
    ) -> np.ndarray:

        w = 2.0 * np.pi * stim_freq
        phase = w * t

        transient = A * np.exp(-t / tau) * np.exp(alpha_peak * trial)
        baseline = B * np.exp(alpha_baseline * trial)

        wave = 0.5 * (np.sin(phase + phi1) + np.sin(2.0 * phase + phi2))
        ripple_depth = sigmoid_bounded(z0_ripple + alpha_ripple * trial, cls._RIPPLE_UPPER)
        ripple_mod = 1.0 + ripple_depth * wave

        return (transient + baseline) * ripple_mod

    # -- Variants ----------------------------------------------------------

    @classmethod
    def time_only(cls, stim_freq: float) -> RateKernel:
        """No trial-dependent plasticity."""
        def _func(t, trial, params):
            A, tau, B, z0_ripple, phi1, phi2 = params
            return cls._rate(t, trial, stim_freq, A, tau, B, z0_ripple, phi1, phi2)

        return RateKernel(
            name="PreyCapture(TimeOnly)",
            func=_func,
            param_names=list(cls._PARAM_NAMES),
            initial_guesses=list(cls._GUESSES),
            bounds=list(cls._BOUNDS),
            latex_formula=(
                r"$\lambda(t) = \left(A e^{-t/\tau} + B\right)"
                rf"\left(1 + {cls._depth_latex()} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak(cls, stim_freq: float) -> RateKernel:
        """Only the transient (peak) amplitude is modulated across trials."""
        def _func(t, trial, params):
            A, tau, B, z0_ripple, phi1, phi2, alpha_peak = params
            return cls._rate(t, trial, stim_freq, A, tau, B, z0_ripple, phi1, phi2,
                              alpha_peak=alpha_peak)

        return RateKernel(
            name="PreyCapture(Peak)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_peak"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} e^{\alpha_{\text{peak}} m} + B\right)"
                rf"\left(1 + {cls._depth_latex()} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def baseline(cls, stim_freq: float) -> RateKernel:
        """Only the tonic baseline is modulated across trials."""
        def _func(t, trial, params):
            A, tau, B, z0_ripple, phi1, phi2, alpha_baseline = params
            return cls._rate(t, trial, stim_freq, A, tau, B, z0_ripple, phi1, phi2,
                              alpha_baseline=alpha_baseline)

        return RateKernel(
            name="PreyCapture(Baseline)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_baseline"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} + B e^{\alpha_{\text{base}} m}\right)"
                rf"\left(1 + {cls._depth_latex()} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline(cls, stim_freq: float) -> RateKernel:
        """Peak and baseline each independently modulated across trials."""
        def _func(t, trial, params):
            A, tau, B, z0_ripple, phi1, phi2, alpha_peak, alpha_baseline = params
            return cls._rate(t, trial, stim_freq, A, tau, B, z0_ripple, phi1, phi2,
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
                rf"\left(1 + {cls._depth_latex()} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline_ripple(cls, stim_freq: float) -> RateKernel:
        """Peak, baseline, and ripple depth each independently modulated (fully general)."""
        def _func(t, trial, params):
            A, tau, B, z0_ripple, phi1, phi2, alpha_peak, alpha_baseline, alpha_ripple = params
            return cls._rate(t, trial, stim_freq, A, tau, B, z0_ripple, phi1, phi2,
                              alpha_peak=alpha_peak, alpha_baseline=alpha_baseline,
                              alpha_ripple=alpha_ripple)

        return RateKernel(
            name="PreyCapture(Peak_Baseline_Ripple)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_peak", "alpha_baseline", "alpha_ripple"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS, cls._ALPHA_GUESS, 0.0],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS, cls._ALPHA_BOUNDS, cls._ALPHA_RIPPLE_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} e^{\alpha_{\text{peak}} m}"
                r" + B e^{\alpha_{\text{base}} m}\right)"
                rf"\left(1 + {cls._depth_latex('ripple')} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline_shared(cls, stim_freq: float) -> RateKernel:
        """Peak and baseline share a single modulation parameter; ripple unmodulated."""
        def _func(t, trial, params):
            A, tau, B, z0_ripple, phi1, phi2, alpha_shared = params
            return cls._rate(t, trial, stim_freq, A, tau, B, z0_ripple, phi1, phi2,
                              alpha_peak=alpha_shared, alpha_baseline=alpha_shared)

        return RateKernel(
            name="PreyCapture(Peak_Baseline_Shared)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_shared"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} + B\right) e^{\alpha_{\text{shared}} m}"
                rf"\left(1 + {cls._depth_latex()} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline_shared_ripple(cls, stim_freq: float) -> RateKernel:
        """Peak & baseline share one modulation parameter; ripple has its own, separate one."""
        def _func(t, trial, params):
            A, tau, B, z0_ripple, phi1, phi2, alpha_shared, alpha_ripple = params
            return cls._rate(t, trial, stim_freq, A, tau, B, z0_ripple, phi1, phi2,
                              alpha_peak=alpha_shared, alpha_baseline=alpha_shared,
                              alpha_ripple=alpha_ripple)

        return RateKernel(
            name="PreyCapture(Peak_Baseline_Shared_Ripple)",
            func=_func,
            param_names=cls._PARAM_NAMES + ["alpha_shared", "alpha_ripple"],
            initial_guesses=cls._GUESSES + [cls._ALPHA_GUESS, 0.0],
            bounds=cls._BOUNDS + [cls._ALPHA_BOUNDS, cls._ALPHA_RIPPLE_BOUNDS],
            latex_formula=(
                r"$\lambda(t,m) = \left(A e^{-t/\tau} + B\right) e^{\alpha_{\text{shared}} m}"
                rf"\left(1 + {cls._depth_latex('ripple')} {cls._RIPPLE_WAVE_LATEX}\right)$"
            ),
        )

    @classmethod
    def peak_baseline_ripple_shared(cls, stim_freq: float) -> RateKernel:
        """Peak, baseline, and ripple all share a single modulation parameter."""
        def _func(t, trial, params):
            A, tau, B, z0_ripple, phi1, phi2, alpha_shared = params
            return cls._rate(t, trial, stim_freq, A, tau, B, z0_ripple, phi1, phi2,
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
                rf"\left(1 + {cls._depth_latex('shared')} {cls._RIPPLE_WAVE_LATEX}\right)$"
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
    def phototaxis_contra(f_dip_upper: float = 0.95) -> RateKernel:
        """
        h(t,m) = B*exp(alpha_B*m) * (1 - depth(m)*exp(-t/tau_dip))
        depth(m) = sigmoid_bounded(z0_dip + alpha_dip*m, f_dip_upper)

        Reparametrized from f_dip (box-bounded to [0, 0.49) purely so that
        f_dip * bounded_trial_scale(m, alpha_dip) -- whose worst case over m
        approaches 2*f_dip -- stayed under 1). That coupling made 0.49 an
        artificial ceiling on f_dip itself, not a real belief about dip depth,
        and bootstrap fits were pinning against it exactly (CI collapsed to a
        point at 0.49). Bounding depth(m) directly via a logistic of an
        unconstrained (z0_dip, alpha_dip) removes the coupling: depth(m) is
        guaranteed in (0, f_dip_upper) for ANY z0_dip, alpha_dip, ANY trial m --
        no worst-case analysis, no artificial margin below the true limit.
        """
        def _func(t, trial, params):
            B, z0_dip, tau_dip, alpha_B, alpha_dip = params

            mod_B = B * np.exp(alpha_B * trial)
            depth = sigmoid_bounded(z0_dip + alpha_dip * trial, f_dip_upper)
            dip_factor = 1.0 - depth * np.exp(-t / tau_dip)

            return mod_B * dip_factor

        z0_dip_init = float(logit_bounded(0.5, f_dip_upper))

        return RateKernel(
            name="Phototaxis Contra λ(t, m)",
            func=_func,
            param_names=["B", "z0_dip", "tau_dip", "alpha_B", "alpha_dip"],
            initial_guesses=[0.4, z0_dip_init, 0.5, 0.0, 0.0],
            bounds=[
                (0.01, 10.0),
                (-15.0, 15.0),   # z0_dip -- unconstrained on the logit scale
                (0.01, 5.0),
                (-0.1, 0.1),
                (-0.5, 0.5),     # alpha_dip -- widened since it now acts on logit,
                                # not probability, scale (logistic derivative <=0.25)
            ],
            latex_formula=(
                r"$\lambda(t, m) = B e^{\alpha_B m}\left(1 - \frac{" + f"{f_dip_upper}"
                r"}{1+e^{-(z_{0,\text{dip}}+\alpha_{\text{dip}} m)}}\, e^{-t/\tau_{\text{dip}}}\right)$"
            ),
        )

    @staticmethod
    def spont() -> RateKernel:
        """
        h(t,m) = B*exp(alpha_B*m)
        """
        def _func(t, trial, params):
            B, alpha_B = params
            return B * np.exp(alpha_B * trial)

        return RateKernel(
            name="Spontaneous λ(t, m)",
            func=_func,
            param_names=["B", "alpha_B",],
            initial_guesses=[0.4, 0.0],
            bounds=[
                (0.01, 10.0),
                (-0.5, 0.5),     
            ],
            latex_formula=(
                r"$\lambda(t, m) = B e^{\alpha_B m}$"
            ),
        )

    @staticmethod
    def after_looming(f_dip_upper: float = 0.95) -> RateKernel:
        """
        h(t,m) = B*exp(alpha_B*m) * (1 - depth(m)*exp(-t/tau_dip))
        depth(m) = sigmoid_bounded(z0_dip + alpha_dip*m, f_dip_upper)

        Reparametrized from f_dip (box-bounded to [0, 0.49) purely so that
        f_dip * bounded_trial_scale(m, alpha_dip) -- whose worst case over m
        approaches 2*f_dip -- stayed under 1). That coupling made 0.49 an
        artificial ceiling on f_dip itself, not a real belief about dip depth,
        and bootstrap fits were pinning against it exactly (CI collapsed to a
        point at 0.49). Bounding depth(m) directly via a logistic of an
        unconstrained (z0_dip, alpha_dip) removes the coupling: depth(m) is
        guaranteed in (0, f_dip_upper) for ANY z0_dip, alpha_dip, ANY trial m --
        no worst-case analysis, no artificial margin below the true limit.
        """
        def _func(t, trial, params):
            B, z0_dip, tau_dip, alpha_B, alpha_dip = params

            mod_B = B * np.exp(alpha_B * trial)
            depth = sigmoid_bounded(z0_dip + alpha_dip * trial, f_dip_upper)
            dip_factor = 1.0 - depth * np.exp(-t / tau_dip)

            return mod_B * dip_factor

        z0_dip_init = float(logit_bounded(0.5, f_dip_upper))

        return RateKernel(
            name="Looming recovery λ(t, m)",
            func=_func,
            param_names=["B", "z0_dip", "tau_dip", "alpha_B", "alpha_dip"],
            initial_guesses=[0.4, z0_dip_init, 0.5, 0.0, 0.0],
            bounds=[
                (0.01, 10.0),
                (-10, 10),   # z0_dip -- unconstrained on the logit scale
                (0.05, 15.0),
                (-0.1, 0.1),
                (-2.0, 2.0),     # alpha_dip -- widened since it now acts on logit,
                                # not probability, scale (logistic derivative <=0.25)
            ],
            latex_formula=(
                r"$\lambda(t, m) = B e^{\alpha_B m}\left(1 - \frac{" + f"{f_dip_upper}"
                r"}{1+e^{-(z_{0,\text{dip}}+\alpha_{\text{dip}} m)}}\, e^{-t/\tau_{\text{dip}}}\right)$"
            ),
        )
        
    @staticmethod
    def omr_forward(f_dip_upper: float = 0.995) -> RateKernel:
        """
        h(t) = B*(1 - f_dip*exp(-t/tau_dip)), f_dip = sigmoid_bounded(z_dip, f_dip_upper).

        Reparametrized from a directly box-bounded f_dip in [0, 0.99): bootstrap
        fits were landing at/near that box bound (~upper CI == 0.99), right-
        censoring the reported uncertainty. z_dip has no such ceiling -- an f_dip
        that "wants" near-total suppression shows up as a large z_dip with wide
        bootstrap spread on the z-scale, not a clipped point mass at 0.99.
        """
        def _func(t, trial, params):
            B, z_dip, tau_dip = params
            f_dip = sigmoid_bounded(z_dip, f_dip_upper)
            return B * (1.0 - f_dip * np.exp(-t / tau_dip))

        z_dip_init = float(logit_bounded(0.5, f_dip_upper))

        return RateKernel(
            name="OMR forward λ(t)",
            func=_func,
            param_names=["B", "z_dip", "tau_dip"],
            initial_guesses=[0.4, z_dip_init, 0.5],
            bounds=[(0.01, 5.0), (-15.0, 15.0), (0.01, 5.0)],
            latex_formula=(
                r"$\lambda(t) = B\left(1 - \frac{" + f"{f_dip_upper}"
                r"}{1+e^{-z_{\text{dip}}}}\, e^{-t/\tau_{\text{dip}}}\right)$"
            ),
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
    def phototaxis_dip_exgaussian_peak(
            tau_dip_init: float = 1.0, tau_dip_bounds: Tuple[float, float] = (0.05, 5.0),
            mu_init: float = 0.4, mu_bounds: Tuple[float, float] = (0.001, 3.0),
            sigma_init: float = 0.15, sigma_bounds: Tuple[float, float] = (0.01, 2.0),
            tau_decay_init: float = 0.3, tau_decay_bounds: Tuple[float, float] = (0.01, 5.0),
        ) -> RateKernel:
        """
        lambda(t,m) = B*exp(alpha_B*m)*(1 - f_dip*exp(-t/tau_dip))
                    + A_peak*exp(alpha_peak*m) * exGaussian(t; mu, sigma, tau_decay)

        mu/sigma control the onset location/sharpness of the peak
        (analogous to a Gaussian rise), tau_decay is an independent
        exponential-tail rate -- as tau_decay -> 0 this collapses toward a
        symmetric Gaussian peak; as sigma -> 0 it collapses toward a plain
        exponential decay from t=mu. tau_dip is now free to differ from
        tau_decay, removing the forced dip/peak timescale coupling.

        Numerically stable formulation: computes exp(A - arg**2) * erfcx(arg)
        instead of exp(A) * erfc(arg) to avoid overflow/underflow when
        lam/2*(2*mu + lam*sigma**2 - 2*t) is large and positive (i.e. well
        past the peak, where erfc(arg) underflows before the exponential
        prefactor overflows).
        """
        def _func(t, trial, params):
            (B, f_dip, tau_dip, A_peak, alpha_B, alpha_peak,
            mu, sigma, tau_decay) = params

            baseline = (
                B * np.exp(alpha_B * trial)
                * (1.0 - f_dip * np.exp(-t / tau_dip))
            )
            height = A_peak * np.exp(alpha_peak * trial)
            peak_shape = exgaussian_shape(t, mu, sigma, tau_decay)

            return baseline + height * peak_shape

        return RateKernel(
            name="PhototaxisDipExGaussianPeak",
            func=_func,
            param_names=[
                "B", "f_dip", "tau_dip", "A_peak", "alpha_B", "alpha_peak",
                "mu", "sigma", "tau_decay",
            ],
            initial_guesses=[
                0.3, 0.5, tau_dip_init, 0.3, 0.0, 0.0,
                mu_init, sigma_init, tau_decay_init,
            ],
            bounds=[
                (1e-4, 20.0),          # B
                (0.0, 1.0),            # f_dip
                tau_dip_bounds,        # tau_dip
                (0.001, 30.0),         # A_peak
                (-2.0, 2.0),           # alpha_B
                (-2.0, 2.0),           # alpha_peak
                mu_bounds,             # mu
                sigma_bounds,          # sigma
                tau_decay_bounds,      # tau_decay
            ],
            latex_formula=(
                r"$\lambda(t,m) = B e^{\alpha_B m}(1 - f_{\mathrm{dip}} e^{-t/\tau_{\mathrm{dip}}}) "
                r"+ A_{\mathrm{peak}} e^{\alpha_{\mathrm{peak}} m}\cdot"
                r"\mathrm{exGauss}(t;\mu,\sigma,\tau_{\mathrm{decay}})$"
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

    def _base_exposure_for_stream(self, dataset: PointProcessDataset, t_idx: int) -> float:
        """self.params_ IS the kernel's own params directly -- no splitting needed."""
        return self.kernel.integrate(dataset.duration_s, t_idx, self.params_, self.integration_dt)