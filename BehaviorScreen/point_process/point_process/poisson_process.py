from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Union

import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid

from .dataset import PointProcessDataset
from .point_process import PointProcess, thin
from .kernel_shapes import (
    exgaussian_shape,
    bounded_trial_scale,
    sigmoid_bounded,
    logit_bounded,
)


@dataclass(frozen=True)
class RateKernel:
    name: str
    func: Callable[[np.ndarray, np.ndarray, List[float]], np.ndarray]
    param_names: List[str]
    initial_guesses: List[float]
    bounds: List[Tuple[Optional[float], Optional[float]]]
    latex_formula: str = ""
    integral_func: Optional[
        Callable[[float, float, np.ndarray, List[float]], np.ndarray]
    ] = None

    def evaluate(
        self, t: np.ndarray, trial: np.ndarray, params: List[float]
    ) -> np.ndarray:
        rate = self.func(t, trial, params)
        target_shape = np.broadcast(t, trial).shape
        return np.broadcast_to(rate, target_shape)

    def integrate(
        self,
        duration_s: float,
        trial: np.ndarray,
        params: List[float],
        integration_dt: float = 0.02,
    ) -> np.ndarray:

        if self.integral_func is not None:
            return self.integral_func(0, duration_s, trial, params)

        t_grid = np.arange(0, duration_s, integration_dt)
        t_grid = np.append(t_grid, duration_s)
        t_2d = t_grid[None, :]  # Shape: (1, N_time)
        trials_2d = np.atleast_1d(trial)[:, None]  # Shape: (N_trials, 1)

        rate_surface = self.evaluate(t_2d, trials_2d, params)
        integrals = trapezoid(rate_surface, x=t_grid, axis=1)

        # Preserve scalar input shape if a scalar trial was passed
        return integrals if np.iterable(trial) else integrals[0]

    def cumulative_integrate(
        self,
        t_events: np.ndarray,
        trial: Union[float, int],
        params: List[float],
        integration_dt: float = 0.02,
    ) -> np.ndarray:
        """
        Integral of the rate from 0 to each t in t_events: trapezoid on an
        integration_dt grid, with the last partial interval ending exactly
        at t (same value as integrate(t, trial, ...)).
        """
        t_events = np.asarray(t_events, dtype=float)
        if len(t_events) == 0:
            return np.array([], dtype=float)

        if self.integral_func is not None:
            return self.integral_func(0, t_events, trial, params)

        t_max = np.max(t_events)
        if t_max <= 0:
            return np.zeros_like(t_events, dtype=float)

        t_grid = np.append(np.arange(0, t_max, integration_dt), t_max)
        rate = self.evaluate(t_grid, np.full_like(t_grid, trial), params)
        cumulative = cumulative_trapezoid(rate, t_grid, initial=0.0)
        k = np.clip(np.searchsorted(t_grid, t_events, side="right") - 1, 0, len(t_grid) - 1)
        rate_at_t = self.evaluate(t_events, np.full_like(t_events, trial), params)
        return cumulative[k] + (t_events - t_grid[k]) * (rate[k] + rate_at_t) / 2

class PreyCapture:
    """
    lambda(t, m) = [A e^(-t/tau) e^(alpha_peak m) + B e^(alpha_baseline m)] (1 + depth(m) wave(t))
    wave(t)  = (sin(w t + phi1) + sin(2 w t + phi2)) / 2,  w = 2 pi stim_freq
    depth(m) = RIPPLE_UPPER / (1 + e^-(z0_ripple + alpha_ripple m))

    A: onset-burst height (Hz) at trial 0; tau: its decay (s); B: tonic rate
    (Hz) at trial 0; z0_ripple: ripple depth on the logit scale (depth at
    trial 0 = sigmoid_bounded(z0_ripple, RIPPLE_UPPER)); phi1, phi2: phases.
    The alphas modulate peak and baseline log-linearly and the ripple depth
    on the logit scale, per trial; variants differ in which alphas are free
    and which are shared.
    """

    _PARAM_NAMES: List[str] = ["A", "tau", "B", "z0_ripple", "phi1", "phi2"]
    _RIPPLE_UPPER: float = 0.95
    _GUESSES: List[float] = [0.56, 1.15, 0.40, float(logit_bounded(0.1, _RIPPLE_UPPER)), 0.0, 0.0]
    _BOUNDS: List[Tuple[Optional[float], Optional[float]]] = [
        (0.01, 10.0),
        (0.1, 5.0),
        (0.01, 5.0),
        (-15.0, 15.0),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    ]
    _ALPHA_BOUNDS: Tuple[float, float] = (-2.0, 2.0)
    _ALPHA_RIPPLE_BOUNDS: Tuple[float, float] = (-4.0, 4.0)
    _ALPHA_GUESS: float = -0.05
    # variant: {modulated term: parameter}
    _VARIANTS: Dict[str, Dict[str, str]] = {
        "TimeOnly": {},
        "Peak": {"peak": "alpha_peak"},
        "Baseline": {"baseline": "alpha_baseline"},
        "Peak_Baseline": {"peak": "alpha_peak", "baseline": "alpha_baseline"},
        "Peak_Baseline_Ripple": {"peak": "alpha_peak", "baseline": "alpha_baseline", "ripple": "alpha_ripple"},
        "Peak_Baseline_Shared": {"peak": "alpha_shared", "baseline": "alpha_shared"},
        "Peak_Baseline_Shared_Ripple": {"peak": "alpha_shared", "baseline": "alpha_shared", "ripple": "alpha_ripple"},
        "Peak_Baseline_Ripple_Shared": {"peak": "alpha_shared", "baseline": "alpha_shared", "ripple": "alpha_shared"},
    }

    @classmethod
    def _rate(
        cls,
        t: np.ndarray,
        trial: np.ndarray,
        stim_freq: float,
        A: float,
        tau: float,
        B: float,
        z0_ripple: float,
        phi1: float,
        phi2: float,
        alpha_peak: float = 0.0,
        alpha_baseline: float = 0.0,
        alpha_ripple: float = 0.0,
    ) -> np.ndarray:
        phase = 2.0 * np.pi * stim_freq * t
        transient = A * np.exp(-t / tau) * np.exp(alpha_peak * trial)
        baseline = B * np.exp(alpha_baseline * trial)
        wave = 0.5 * (np.sin(phase + phi1) + np.sin(2.0 * phase + phi2))
        depth = sigmoid_bounded(z0_ripple + alpha_ripple * trial, cls._RIPPLE_UPPER)
        return (transient + baseline) * (1.0 + depth * wave)

    @classmethod
    def variant(cls, variant: str, stim_freq: float) -> RateKernel:
        modulated = cls._VARIANTS[variant]
        alphas = list(dict.fromkeys(modulated.values()))
        names = cls._PARAM_NAMES + alphas

        def _func(t: np.ndarray, trial: np.ndarray, params: List[float]) -> np.ndarray:
            values = dict(zip(names, params))
            return cls._rate(
                t,
                trial,
                stim_freq,
                *(values[p] for p in cls._PARAM_NAMES),
                **{f"alpha_{term}": values[p] for term, p in modulated.items()},
            )

        def factor(term: str) -> str:
            p = modulated.get(term)
            return rf" e^{{\alpha_{{\text{{{p[6:]}}}}} m}}" if p else ""

        ripple = modulated.get("ripple")
        z = rf"z_{{0,\text{{ripple}}}}" + (rf"+\alpha_{{\text{{{ripple[6:]}}}}} m" if ripple else "")
        depth = rf"\frac{{{cls._RIPPLE_UPPER}}}{{1+e^{{-({z})}}}}"
        wave = r"\frac{1}{2}\left[\sin(\omega t + \phi_1) + \sin(2\omega t + \phi_2)\right]"
        return RateKernel(
            name=f"PreyCapture({variant})",
            func=_func,
            param_names=names,
            initial_guesses=cls._GUESSES + [0.0 if p == "alpha_ripple" else cls._ALPHA_GUESS for p in alphas],
            bounds=cls._BOUNDS + [cls._ALPHA_RIPPLE_BOUNDS if p == "alpha_ripple" else cls._ALPHA_BOUNDS for p in alphas],
            latex_formula=(
                rf"$\lambda(t,m) = \left(A e^{{-t/\tau}}{factor('peak')} + B{factor('baseline')}\right)"
                rf"\left(1 + {depth} {wave}\right)$"
            ),
        )

    time_only = classmethod(lambda cls, stim_freq: cls.variant("TimeOnly", stim_freq))
    peak = classmethod(lambda cls, stim_freq: cls.variant("Peak", stim_freq))
    baseline = classmethod(lambda cls, stim_freq: cls.variant("Baseline", stim_freq))
    peak_baseline = classmethod(lambda cls, stim_freq: cls.variant("Peak_Baseline", stim_freq))
    peak_baseline_ripple = classmethod(lambda cls, stim_freq: cls.variant("Peak_Baseline_Ripple", stim_freq))
    peak_baseline_shared = classmethod(lambda cls, stim_freq: cls.variant("Peak_Baseline_Shared", stim_freq))
    peak_baseline_shared_ripple = classmethod(
        lambda cls, stim_freq: cls.variant("Peak_Baseline_Shared_Ripple", stim_freq)
    )
    peak_baseline_ripple_shared = classmethod(
        lambda cls, stim_freq: cls.variant("Peak_Baseline_Ripple_Shared", stim_freq)
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
                (0.01, 5.0),  # B (Hz)
                (0.0, 0.99),  # f_dip -- guarantees positivity for ANY B
                (0.01, 2.0),  # tau (s)
                (0.0, 27.0),  # A_peak (Hz)
                (-0.2, 0.2),  # alpha_B
                (-0.2, 0.2),  # alpha_peak
            ],
            latex_formula=(
                r"$\lambda(t, m) = B e^{\alpha_B m} \left(1 - f_{\text{dip}} e^{-t/\tau}\right)"
                r" + A_{\text{peak}} e^{\alpha_{\text{peak}} m}\, e\,(t/\tau) e^{-t/\tau}$"
            ),
        )

    @staticmethod
    def _recovering_dip(
        name: str,
        f_dip_upper: float,
        z0_dip_bounds: Tuple[float, float],
        tau_dip_bounds: Tuple[float, float],
        alpha_dip_bounds: Tuple[float, float],
    ) -> RateKernel:
        """
        lambda(t, m) = B e^(alpha_B m) (1 - depth(m) e^(-t/tau_dip)),
        depth(m) = f_dip_upper / (1 + e^-(z0_dip + alpha_dip m)) in (0, f_dip_upper)
        for any parameters and trial, so the rate stays positive without
        box-bounding the depth itself.
        """

        def _func(t: np.ndarray, trial: np.ndarray, params: List[float]) -> np.ndarray:
            B, z0_dip, tau_dip, alpha_B, alpha_dip = params
            depth = sigmoid_bounded(z0_dip + alpha_dip * trial, f_dip_upper)
            return B * np.exp(alpha_B * trial) * (1.0 - depth * np.exp(-t / tau_dip))

        return RateKernel(
            name=name,
            func=_func,
            param_names=["B", "z0_dip", "tau_dip", "alpha_B", "alpha_dip"],
            initial_guesses=[0.4, float(logit_bounded(0.5, f_dip_upper)), 0.5, 0.0, 0.0],
            bounds=[(0.01, 10.0), z0_dip_bounds, tau_dip_bounds, (-0.1, 0.1), alpha_dip_bounds],
            latex_formula=(
                r"$\lambda(t, m) = B e^{\alpha_B m}\left(1 - \frac{" + f"{f_dip_upper}"
                r"}{1+e^{-(z_{0,\text{dip}}+\alpha_{\text{dip}} m)}}\, e^{-t/\tau_{\text{dip}}}\right)$"
            ),
        )

    @staticmethod
    def phototaxis_contra(f_dip_upper: float = 0.95) -> RateKernel:
        return RateKernelFactory._recovering_dip(
            "Phototaxis Contra λ(t, m)", f_dip_upper, (-15.0, 15.0), (0.01, 5.0), (-0.5, 0.5)
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
            param_names=[
                "B",
                "alpha_B",
            ],
            initial_guesses=[0.4, 0.0],
            bounds=[
                (0.01, 10.0),
                (-0.5, 0.5),
            ],
            latex_formula=(r"$\lambda(t, m) = B e^{\alpha_B m}$"),
        )

    @staticmethod
    def after_looming(f_dip_upper: float = 0.95) -> RateKernel:
        return RateKernelFactory._recovering_dip(
            "Looming recovery λ(t, m)", f_dip_upper, (-10, 10), (0.05, 15.0), (-2.0, 2.0)
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
        """h(t) = B*(1 - f_dip*exp(-t/tau_dip)), f_dip box-bounded in [0, 0.99]."""

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
        tau_dip_init: float = 1.0,
        tau_dip_bounds: Tuple[float, float] = (0.05, 5.0),
        mu_init: float = 0.4,
        mu_bounds: Tuple[float, float] = (0.001, 3.0),
        sigma_init: float = 0.15,
        sigma_bounds: Tuple[float, float] = (0.01, 2.0),
        tau_decay_init: float = 0.3,
        tau_decay_bounds: Tuple[float, float] = (0.01, 5.0),
    ) -> RateKernel:
        """
        lambda(t,m) = B*exp(alpha_B*m)*(1 - f_dip*exp(-t/tau_dip))
                    + A_peak*exp(alpha_peak*m) * exGaussian(t; mu, sigma, tau_decay)

        mu/sigma control the onset location/sharpness of the peak
        (analogous to a Gaussian rise), tau_decay is an independent
        exponential-tail rate -- as tau_decay -> 0 this collapses toward a
        symmetric Gaussian peak; as sigma -> 0 it collapses toward a plain
        exponential decay from t=mu. tau_dip is independent of tau_decay.
        """

        def _func(t, trial, params):
            B, f_dip, tau_dip, A_peak, alpha_B, alpha_peak, mu, sigma, tau_decay = (
                params
            )

            baseline = (
                B * np.exp(alpha_B * trial) * (1.0 - f_dip * np.exp(-t / tau_dip))
            )
            height = A_peak * np.exp(alpha_peak * trial)
            peak_shape = exgaussian_shape(t, mu, sigma, tau_decay)

            return baseline + height * peak_shape

        return RateKernel(
            name="PhototaxisDipExGaussianPeak",
            func=_func,
            param_names=[
                "B",
                "f_dip",
                "tau_dip",
                "A_peak",
                "alpha_B",
                "alpha_peak",
                "mu",
                "sigma",
                "tau_decay",
            ],
            initial_guesses=[
                0.3,
                0.5,
                tau_dip_init,
                0.3,
                0.0,
                0.0,
                mu_init,
                sigma_init,
                tau_decay_init,
            ],
            bounds=[
                (1e-4, 20.0),  # B
                (0.0, 1.0),  # f_dip
                tau_dip_bounds,  # tau_dip
                (0.001, 30.0),  # A_peak
                (-2.0, 2.0),  # alpha_B
                (-2.0, 2.0),  # alpha_peak
                mu_bounds,  # mu
                sigma_bounds,  # sigma
                tau_decay_bounds,  # tau_decay
            ],
            latex_formula=(
                r"$\lambda(t,m) = B e^{\alpha_B m}(1 - f_{\mathrm{dip}} e^{-t/\tau_{\mathrm{dip}}}) "
                r"+ A_{\mathrm{peak}} e^{\alpha_{\mathrm{peak}} m}\cdot"
                r"\mathrm{exGauss}(t;\mu,\sigma,\tau_{\mathrm{decay}})$"
            ),
        )

    @staticmethod
    def phototaxis_exgaussian_peak_no_dip(
        mu_init: float = 0.25,
        mu_bounds: Tuple[float, float] = (0.001, 3.0),
        sigma_init: float = 0.05,
        sigma_bounds: Tuple[float, float] = (0.01, 2.0),
        tau_decay_init: float = 0.4,
        tau_decay_bounds: Tuple[float, float] = (0.01, 5.0),
    ) -> RateKernel:
        """
        Tonic baseline plus a trial-modulated ex-Gaussian transient:

            lambda_0(t, m)
                = B * exp(alpha_B * m)
                + A_peak * exp(alpha_peak * m)
                    * exGaussian(t; mu, sigma, tau_decay)

        This is the reduced version of phototaxis_dip_exgaussian_peak after
        removing f_dip and tau_dip. Those parameters are not identifiable when
        the fitted dip amplitude is zero.

        Notes
        -----
        ``exgaussian_shape`` is a probability-density shape with unit integral.
        Consequently, ``A_peak`` is a scale multiplying that density; it is not
        literally the height of the resulting peak in Hz. The existing name is
        retained for compatibility with the other phototaxis kernels.
        """

        def _func(t, trial, params):
            (
                B,
                A_peak,
                alpha_B,
                alpha_peak,
                mu,
                sigma,
                tau_decay,
            ) = params

            baseline = B * np.exp(alpha_B * trial)
            transient_scale = A_peak * np.exp(alpha_peak * trial)
            transient_shape = exgaussian_shape(
                t,
                mu,
                sigma,
                tau_decay,
            )

            return baseline + transient_scale * transient_shape

        return RateKernel(
            name="PhototaxisExGaussianPeakNoDip",
            func=_func,
            param_names=[
                "B",
                "A_peak",
                "alpha_B",
                "alpha_peak",
                "mu",
                "sigma",
                "tau_decay",
            ],
            initial_guesses=[
                0.25,
                0.17,
                0.03,
                0.02,
                mu_init,
                sigma_init,
                tau_decay_init,
            ],
            bounds=[
                (1e-4, 20.0),  # B
                (1e-4, 30.0),  # A_peak
                (-2.0, 2.0),  # alpha_B
                (-2.0, 2.0),  # alpha_peak
                mu_bounds,  # mu
                sigma_bounds,  # sigma
                tau_decay_bounds,  # tau_decay
            ],
            latex_formula=(
                r"$\lambda_0(t,m)="
                r"B e^{\alpha_Bm}"
                r"+A_{\mathrm{peak}}e^{\alpha_{\mathrm{peak}}m}"
                r"\operatorname{exGauss}(t;\mu,\sigma,\tau_{\mathrm{decay}})$"
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

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:

        # Term 1: Sum of Log Intensity at Observed Events
        event_rates = self.kernel.evaluate(
            dataset.event_times, dataset.event_trials_idx, params
        )
        sum_log_rates = np.sum(np.log(event_rates))

        # Term 2: Expected Total Events (Surface Integration over Time)
        trial_integrals = self.kernel.integrate(
            duration_s=dataset.duration_s,
            trial=np.arange(dataset.num_trials),
            params=params,
            integration_dt=self.integration_dt,
        )

        # Scale expected events by trial-specific observing fish count
        total_expected_events = np.sum(trial_integrals * dataset.n_fish_per_trial)
        return -(sum_log_rates - total_expected_events)

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray]) -> np.ndarray:
        self._require_fitted()

        expected_rate = self.kernel.evaluate(t, trial, self.params_)
        return expected_rate

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        self._require_fitted()

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

        S_f = (
            dataset.fish_trial_mask.astype(float) @ trial_integrals
        )  # shape (num_fish,)

        return base_ll, N_f, S_f

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng: np.random.Generator
    ) -> np.ndarray:
        """One stream of a fish with frailty gain `gain`, by thinning."""
        upper = gain * self._intensity_upper_bound(dataset, t_idx)
        return thin(
            lambda t, events: gain * self.kernel.evaluate(np.array([t]), np.array([t_idx]), self.params_)[0],
            lambda t, events: upper,
            dataset.duration_s,
            rng,
        )
