# point_process/survival_process.py
from typing import List, Optional, Tuple, Union

import numpy as np


from .dataset import PointProcessDataset
from .point_process import PointProcess
from .poisson_process import RateKernel

class SurvivalKernelFactory:

    @staticmethod
    def constant_hazard() -> RateKernel:
        """Exponential (memoryless) hazard -- the survival analog of
        homogeneous_poisson. Use as the null_model for SurvivalProcess
        comparisons: fits ONLY overall response propensity, no latency
        structure at all."""
        def _func(t, trial, params):
            (B,) = params
            return B * np.ones_like(t + 0.0 * trial)
        return RateKernel(
            name="SurvivalConstantHazard",
            func=_func,
            param_names=["B"],
            initial_guesses=[0.3],
            bounds=[(1e-4, 20.0)],
            latex_formula=r"$h = B$",
        )

    @staticmethod
    def gaussian_bump_baseline(
            t_init: float = 1,
            t_bounds: Tuple[float, float] = (0.001,2)
        ) -> RateKernel:
        """h(t) = H*exp(-(t-mu)^2/2sigma^2) + B. Add ONLY if LR test against
        the no-baseline bump is significant -- KM plateau not being flat
        (slow decline continuing well past the burst) is the empirical
        trigger for trying this."""
        def _func(t, trial, params):
            H, mu, sigma, B = params
            return H * np.exp(-0.5 * ((t - mu) / sigma) ** 2) + B
        return RateKernel(
            name="SurvivalGaussianBump(Baseline)",
            func=_func,
            param_names=["H", "mu", "sigma", "B"],
            initial_guesses=[1.0, t_init, 0.1, 0.02],
            bounds=[(0.001, 30.0), t_bounds, (0.005, 3.0), (1e-4, 1.0)],
            latex_formula=r"$h(t) = H \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right) + B$",
        )

    @staticmethod
    def gaussian_bump_baseline_habituating(
            t_init: float = 1,
            t_bounds: Tuple[float, float] = (0.001,2)
        ) -> RateKernel:
        """
        h(t,m) = H * exp(alpha*m) * exp(-(t-mu)^2/2sigma^2) + B

        Combines gaussian_bump_baseline (B absorbs the slow KM tail past
        the burst) with gaussian_bump_habituating (H declines across the
        session) -- habituation acts on the bump amplitude H only, B is
        trial-constant, same asymmetry as looming_bump_habituating's
        alpha_H-only design (see plot_response_by_trial: if P(respond)
        declines across trials while nothing suggests LATENCY itself
        shifts, that decline is better attributed to a shrinking bump than
        a shrinking/growing floor).

        Nest this against gaussian_bump_baseline (LR test on alpha) before
        keeping it, and against dark_flash_pulse-style shapes if the
        symmetric bump is still visibly mismatched in Panel D even with
        baseline+habituation added (shape mismatch and missing habituation
        are separate problems -- fixing one does not fix the other).
        """
        def _func(t, trial, params):
            H, mu, sigma, B, alpha = params
            height = H * np.exp(alpha * trial)
            return height * np.exp(-0.5 * ((t - mu) / sigma) ** 2) + B
        return RateKernel(
            name="SurvivalGaussianBump(Baseline_Habituating)",
            func=_func,
            param_names=["H", "mu", "sigma", "B", "alpha"],
            initial_guesses=[1.0, t_init, 0.1, 0.02, 0.0],
            bounds=[(0.001, 30.0), t_bounds, (0.005, 3.0), (1e-4, 1.0), (-2.0, 2.0)],
            latex_formula=(
                r"$h(t,m) = H e^{\alpha m} \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right) + B$"
            ),
        )

class SurvivalProcess(PointProcess):
    """
    First-passage-time (right-censored survival) model built on the SAME
    RateKernel objects used by PoissonProcess: lambda(t, m) is reinterpreted
    as a hazard h(t, m), and each (fish, trial) stream is reduced to a
    single observation -- the time of its FIRST event, or duration_s if
    none occurred (right-censored) -- discarding any subsequent events in
    that stream.

    THEORETICAL NOTE: for a stream with at most one event, the ordinary
    point-process likelihood

        L = h(t_event) * exp(-Lambda(t_event))          [event observed]
        L = exp(-Lambda(duration_s))                      [censored]

    with Lambda(t) = integral_0^t h -- is exactly the standard survival
    likelihood (h == hazard, Lambda == cumulative hazard, exp(-Lambda) ==
    survival function): a constrained special case of the machinery used
    elsewhere in this module, not a different model family.

    WHEN TO USE INSTEAD OF PoissonProcess/HawkesProcess/RenewalProcess:
    stimuli that provoke "nothing, then at most one reaction" (dark flash
    O-bends, looming SLCs). Check dataset.frac_streams_with_multiple_events
    first -- if repeat within-trial bouts are common, this reduction
    discards real structure; cross-check against a
    RenewalProcess(kernel, RenewalKernelFactory.hard_absorption()) fit on
    the FULL dataset before trusting this class's AIC for that condition
    (that comparison is scoped identically to HawkesProcess/RenewalProcess;
    this class's own AIC is NOT directly comparable to theirs -- see
    class discussion elsewhere).

    COMPATIBLE WITH GammaMixedEffectsProcess via ordinary polymorphism --
    the only thing this class teaches the rest of the codebase is
    stream_compensator_profile below; GammaMixedEffectsProcess never
    needs to know this class exists.
    """

    def __init__(self, kernel: RateKernel, integration_dt: float = 0.02):
        super().__init__(integration_dt)
        self.name = f"Survival {kernel.name}"
        self.latex_formula = kernel.latex_formula
        self.kernel = kernel
        self.initial_guesses = kernel.initial_guesses
        self.bounds = kernel.bounds
        self.param_names = kernel.param_names

    # -- Core reduction: stream -> (t_obs, censored) ------------------------

    @staticmethod
    def _first_event_or_censor(t_ev: np.ndarray, duration_s: float) -> Tuple[float, bool]:
        """
        Reduce a (fish, trial) stream to a single first-passage observation.
        Any events after the first are discarded -- check
        dataset.frac_streams_with_multiple_events before trusting this
        reduction for a given condition.
        """
        if len(t_ev) == 0:
            return float(duration_s), True
        return float(np.min(t_ev)), False

    # -- The one thing this class teaches the base class ---------------------

    def stream_compensator_profile(
        self, t_ev: np.ndarray, trial: float, duration_s: float,
    ) -> Tuple[np.ndarray, np.ndarray, bool, float]:
        """
        Overrides PointProcess's recurrent default: this process
        terminates at the first event. Everything downstream (calibration
        diagnostics in PointProcess.time_rescaling, GammaMixedEffectsProcess's
        frailty-corrected _stream_tau_values) is correct automatically once
        this single method is right -- no isinstance checks needed anywhere
        else in the codebase.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        t_obs, censored = self._first_event_or_censor(t_ev, duration_s)
        H = float(self.kernel.integrate(
            duration_s=t_obs, trial=trial, params=self.params_,
            integration_dt=self.integration_dt,
        ))
        return np.array([t_obs]), np.array([H]), censored, H

    # -- Likelihood -----------------------------------------------------------

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        total = 0.0
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            t_obs, censored = self._first_event_or_censor(t_ev, dataset.duration_s)

            H = float(self.kernel.integrate(
                duration_s=t_obs, trial=t_idx, params=params,
                integration_dt=self.integration_dt,
            ))

            if censored:
                total += H
            else:
                h = self.kernel.evaluate(
                    np.array([t_obs]), np.array([float(t_idx)]), params
                )[0]
                total += H - np.log(max(h, 1e-12))
        return total

    def mixed_effects_likelihood_terms(
        self, dataset: PointProcessDataset, params: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        N_f: total observed (uncensored) first-passage events per fish.
        S_f: total cumulative hazard exposure per fish, i.e. sum over that
        fish's trials of Lambda(t_obs) -- exactly the sufficient statistic
        a Gamma-frailty survival model needs (Vaupel, Manton & Stallard 1979).
        """
        N_f = np.zeros(dataset.num_fish, dtype=float)
        S_f = np.zeros(dataset.num_fish, dtype=float)
        base_ll = 0.0

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            t_obs, censored = self._first_event_or_censor(t_ev, dataset.duration_s)

            H = float(self.kernel.integrate(
                duration_s=t_obs, trial=t_idx, params=params,
                integration_dt=self.integration_dt,
            ))
            S_f[f_idx] += H

            if not censored:
                N_f[f_idx] += 1.0
                h = self.kernel.evaluate(
                    np.array([t_obs]), np.array([float(t_idx)]), params
                )[0]
                base_ll += np.log(max(h, 1e-12))

        return base_ll, N_f, S_f

    # -- Prediction / population-level curves --------------------------------

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray]) -> np.ndarray:
        """Raw fitted hazard h(t, m). NOTE: NOT a first-passage density;
        prefer population_survival_curve() for anything beyond a rough
        visual check against ModelPlotter's generic overlays."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return self.kernel.evaluate(t, trial, self.params_)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        """Raw fitted hazard surface, matching PoissonProcess's convention
        exactly so ModelPlotter/ModelComparator work unmodified."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        t_2d = dataset.t_centers[None, :]
        trials_2d = np.arange(dataset.num_trials)[:, None]
        return self.predict(t_2d, trials_2d)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        return self.kernel.cumulative_integrate(
            t_events=t_events, trial=trial, params=self.params_,
            integration_dt=self.integration_dt,
        )

    def population_survival_curve(
        self, dataset: PointProcessDataset, trial: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model-implied survival curve S(t) = exp(-Lambda(t)) on
        dataset.t_centers. Safe to evaluate on an arbitrary grid here
        specifically because SurvivalProcess has no self-history --
        cumulative_integrated_intensity(t, trial) IS the marginal
        first-event hazard regardless of what points you probe it at.
        trial=None averages across trials, weighted by n_fish_per_trial.
        Compare against DatasetPlotter.plot_kaplan_meier's empirical
        estimate for a real goodness-of-fit check -- NOT
        compute_expected_rate()'s raw hazard.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")

        t_grid = dataset.t_centers
        if trial is not None:
            Lambda_t = self.cumulative_integrated_intensity(t_grid, int(trial))
            return t_grid, np.exp(-Lambda_t)

        S_matrix = np.zeros((dataset.num_trials, len(t_grid)))
        for tr in range(dataset.num_trials):
            Lambda_t = self.cumulative_integrated_intensity(t_grid, tr)
            S_matrix[tr] = np.exp(-Lambda_t)

        weights = dataset.n_fish_per_trial
        if weights.sum() == 0:
            return t_grid, np.mean(S_matrix, axis=0)
        return t_grid, np.average(S_matrix, axis=0, weights=weights)