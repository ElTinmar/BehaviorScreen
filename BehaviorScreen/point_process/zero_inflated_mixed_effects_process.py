# point_process/zero_inflated_mixed_effects_process.py
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .point_process import PointProcess
from .dataset import PointProcessDataset
from .kernel_shapes import sigmoid_bounded, logit_bounded


class ZeroInflatedGammaMixedEffectsProcess(PointProcess):
    """
    Two-population frailty: a point-mass "non-/low-responder" class (gain
    fixed at c, e.g. c=0 for TRUE non-responders) mixed with weight pi
    against a Gamma-distributed "responder" class (gain ~ Gamma(r, r/mu_R)),
    marginalized analytically (no quadrature -- unlike
    BaselineOnlyFrailtyHawkesProcess, the discrete + conjugate-Gamma
    structure stays closed-form).

    STRICT GENERALIZATION of GammaMixedEffectsProcess: pi -> 0 recovers it
    exactly, term for term (mu_R -> 1, beta -> r, the compensator formula
    below collapses to GammaMixedEffectsProcess's own).

    E[g_f] = pi*c + (1-pi)*mu_R is CONSTRAINED to 1 (mu_R is a DERIVED
    quantity, not a free parameter -- see _mu_responder), so predict()/
    compute_expected_rate()/cumulative_integrated_intensity() delegate to
    base_process exactly like GammaMixedEffectsProcess -- no extra
    derivation needed.

    PRECONDITION: requires the same "gain multiplies the ENTIRE intensity
    uniformly" contract as GammaMixedEffectsProcess (see
    base_process.mixed_effects_likelihood_terms docstring). Do NOT wrap
    BaselineOnlyFrailtyHawkesProcess (its gain scales baseline only) or any
    process that doesn't expose mixed_effects_likelihood_terms.

    IDENTIFIABILITY: testing "pi=0" is a boundary-of-parameter-space test
    (classic finite-mixture LRT irregularity, Self & Liang 1987) -- an
    ordinary chi2(df=2) LRT p-value against GammaMixedEffectsProcess will be
    conservative. Prefer a parametric bootstrap of the LR statistic under
    the pi=0 null. Also sanity-check bootstrap CIs on pi (and c, if fit_c)
    aren't pinned at a boundary.

    LIMITATION (documented, not silently hidden): compute_residuals()'s
    Pearson/deviance residuals assume a single unimodal NB shape. dispersion_r
    below is chosen to at least match the TOTAL gain variance exactly, so
    the variance-based residuals are correctly scaled -- but the mixture's
    true bimodal count distribution is not otherwise reflected in those
    panels. Use estimate_fish_gains()['p_responder'] and
    plot_predicted_vs_observed() for that.
    """

    def __init__(
        self,
        base_process: PointProcess,
        pi_init: float = 0.3,
        r_init: float = 5.0,
        fit_c: bool = False,
        c_init: float = 0.0,
        c_upper: float = 0.3,
    ):
        super().__init__(base_process.integration_dt)
        self.base_process = base_process
        self.fit_c = fit_c
        self.c_upper = c_upper
        self._fixed_c = c_init  # used only when fit_c=False

        self.name = f"ZeroInflatedGammaMixedEffects[{base_process.name}]"
        base_formula = base_process.latex_formula.strip("$")
        self.latex_formula = (
            rf"${base_formula} \times g_f,\ "
            rf"g_f \sim \pi\,\delta_c + (1-\pi)\,\Gamma(r, r/\mu_R)$"
        )

        z_pi_init = float(logit_bounded(pi_init, 1.0))
        extra_names = ["z_pi_nonresponder", "r_dispersion"]
        extra_guesses = [z_pi_init, r_init]
        extra_bounds: List[Tuple[Optional[float], Optional[float]]] = [(-15.0, 15.0), (1e-3, None)]

        if fit_c:
            z_c_init = float(logit_bounded(max(c_init, 1e-4), c_upper))
            extra_names.append("z_c_nonresponder")
            extra_guesses.append(z_c_init)
            extra_bounds.append((-15.0, 15.0))

        self.initial_guesses = base_process.initial_guesses + extra_guesses
        self.bounds = base_process.bounds + extra_bounds
        self.param_names = base_process.param_names + extra_names

    # -- Parameter bookkeeping -------------------------------------------

    def _split_params(self, params: List[float]) -> Tuple[List[float], float, float, float]:
        n_base = len(self.base_process.param_names)
        base_params = params[:n_base]
        z_pi, r = params[n_base], params[n_base + 1]
        if self.fit_c:
            z_c = params[n_base + 2]
            c = float(sigmoid_bounded(z_c, self.c_upper))
        else:
            c = self._fixed_c
        pi = float(sigmoid_bounded(z_pi, 1.0))
        return base_params, pi, max(float(r), 1e-8), c

    def _mu_responder(self, pi: float, c: float) -> float:
        """E[g_f]=1 constraint -- mu_R is DERIVED, never a free parameter."""
        return (1.0 - pi * c) / max(1.0 - pi, 1e-12)

    def fit(self, dataset: PointProcessDataset, method: str = "L-BFGS-B", **kwargs):
        super().fit(dataset, method=method, **kwargs)
        base_params, *_ = self._split_params(self.params_)
        self.base_process.params_ = np.asarray(base_params, dtype=float)
        self.base_process.param_dict_ = dict(zip(self.base_process.param_names, base_params))
        return self

    def set_params(self, params: np.ndarray) -> None:
        super().set_params(params)
        base_params, *_ = self._split_params(list(self.params_))
        self.base_process.set_params(base_params)

    # -- Shared branch-likelihood machinery --------------------------------

    @staticmethod
    def _log_branch_likelihoods(
        N: Union[float, np.ndarray], S: Union[float, np.ndarray], r: float, c: float, beta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unnormalized log-likelihood of sufficient stats (N events, S
        exposure) under each branch, EXCLUDING the sum-log-base-rate term
        (which is common to both branches -- given g, L(data|g) = g^N *
        [prod lambda_base(t_i)] * exp(-g*S) -- so it cancels in any
        NR-vs-R ratio/mixture and can be added once, separately, wherever
        the caller needs an absolute log-likelihood).

        log_L_NR : point-mass branch, g=c fixed.
        log_L_R  : Gamma(r, beta) branch, integrated out (Gamma-Poisson
                   conjugate marginal -- same formula as
                   GammaMixedEffectsProcess._nll's nb_term, generalized to
                   rate beta = r/mu_R instead of r).
        """
        N = np.asarray(N, dtype=float)
        S = np.asarray(S, dtype=float)

        if c <= 0.0:
            log_L_NR = np.where(N == 0, 0.0, -np.inf)
        else:
            log_L_NR = N * np.log(c) - c * S

        log_L_R = r * np.log(beta) - gammaln(r) + gammaln(N + r) - (N + r) * np.log(S + beta)
        return log_L_NR, log_L_R

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        base_params, pi, r, c = self._split_params(params)
        base_ll, N_f, S_f = self.base_process.mixed_effects_likelihood_terms(dataset, base_params)

        mu_R = self._mu_responder(pi, c)
        beta = r / mu_R
        log_L_NR, log_L_R = self._log_branch_likelihoods(N_f, S_f, r, c, beta)

        active = dataset.fish_trial_mask.any(axis=1)
        log_pi = np.log(max(pi, 1e-300))
        log_1mpi = np.log(max(1.0 - pi, 1e-300))

        log_mix = np.logaddexp(log_pi + log_L_NR[active], log_1mpi + log_L_R[active])
        return -(base_ll + float(np.sum(log_mix)))

    # -- Predictable compensator for time-rescaling ------------------------

    def _predictable_tau(
        self, N_count: float, S_prev: float, S_abs: float, pi: float, r: float, c: float, beta: float
    ) -> float:
        """
        Exact compensator increment tau = -log S(t | history), where
        S(t|history) is the marginal (mixture-integrated) survival
        probability of no event over (prev, t], conditional ONLY on
        (N_count, S_prev) -- sufficient stats from STRICTLY BEFORE this
        probe, preserving Ogata's predictability requirement (same
        discipline as GammaMixedEffectsProcess._stream_tau_values).

        S(t|history) = p_NR * exp(-c*dS) + p_R * ((beta+S_prev)/(beta+S_abs))^(N+r)

        p_NR, p_R: posterior class responsibilities given (N_count, S_prev)
        alone (the common base-rate factor cancels in this ratio, same as
        in _nll/_log_branch_likelihoods).

        Reduces EXACTLY to GammaMixedEffectsProcess's
        (r+N)*log((r+S_abs)/(r+S_prev)) when pi=0 (log_p_NR -> -inf,
        beta -> r) -- a genuine consistency check, not a coincidence.
        """
        log_L_NR, log_L_R = self._log_branch_likelihoods(N_count, S_prev, r, c, beta)
        log_pi = np.log(max(pi, 1e-300))
        log_1mpi = np.log(max(1.0 - pi, 1e-300))

        log_post_NR = log_pi + log_L_NR
        log_post_R = log_1mpi + log_L_R
        log_norm = np.logaddexp(log_post_NR, log_post_R)
        log_p_NR = log_post_NR - log_norm
        log_p_R = log_post_R - log_norm

        dS = S_abs - S_prev
        term_NR = log_p_NR - c * dS
        term_R = log_p_R + (N_count + r) * np.log((beta + S_prev) / (beta + S_abs))

        return float(-np.logaddexp(term_NR, term_R))

    def _stream_tau_values(
        self, dataset: PointProcessDataset
    ) -> Dict[Tuple[int, int], List[Tuple[float, bool]]]:
        """
        Correct predictable compensator for the marginalized two-population
        frailty -- see class docstring / _predictable_tau. Structurally
        identical walk to GammaMixedEffectsProcess._stream_tau_values
        (delegates trial-level reduction to base_process.
        stream_compensator_profile, so SurvivalProcess/PoissonProcess/etc.
        bases all work with no isinstance checks), only the tau formula
        itself differs.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        _, pi, r, c = self._split_params(self.params_)
        mu_R = self._mu_responder(pi, c)
        beta = r / mu_R

        result: Dict[Tuple[int, int], List[Tuple[float, bool]]] = {}

        for f_idx in range(dataset.num_fish):
            S_offset = 0.0
            S_prev = 0.0
            N_count = 0

            for t_idx in range(dataset.num_trials):
                if not dataset.fish_trial_mask[f_idx, t_idx]:
                    continue

                t_ev = dataset._stream_index.get((f_idx, t_idx), np.array([], dtype=float))
                probes, cum, last_censored, full_exposure = self.base_process.stream_compensator_profile(
                    t_ev, t_idx, dataset.duration_s
                )

                pairs: List[Tuple[float, bool]] = []
                for k, cum_val in enumerate(cum):
                    S_abs = S_offset + cum_val
                    censored_here = (k == len(cum) - 1) and last_censored
                    tau_val = self._predictable_tau(N_count, S_prev, S_abs, pi, r, c, beta)
                    pairs.append((tau_val, bool(censored_here)))
                    if not censored_here:
                        N_count += 1
                    S_prev = S_abs

                if pairs:
                    result[(f_idx, t_idx)] = pairs

                S_offset += full_exposure
                S_prev = S_offset

        return result

    # -- Population-average delegation (E[g_f]=1 by construction) ----------

    def predict(self, t, trial, **kwargs):
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return self.base_process.predict(t, trial, **kwargs)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return self.base_process.compute_expected_rate(dataset)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return self.base_process.cumulative_integrated_intensity(t_events=t_events, trial=trial)

    def population_survival_curve(
        self, dataset: PointProcessDataset, trial: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Marginal survival curve for the two-population mixture:
        S_pop(t) = pi*exp(-c*Lambda(t)) + (1-pi)*(beta/(beta+Lambda(t)))^r

        Same CAVEAT as GammaMixedEffectsProcess.population_survival_curve:
        only meaningful when base_process has no genuine self-history
        (PoissonProcess/SurvivalProcess bases in this codebase's
        model_config), not documented via isinstance guard, by convention.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        base_params, pi, r, c = self._split_params(self.params_)
        self.base_process.params_ = np.asarray(base_params, dtype=float)  # already synced by fit()
        mu_R = self._mu_responder(pi, c)
        beta = r / mu_R
        t_grid = dataset.t_centers

        def _s(tr: int) -> np.ndarray:
            Lambda_t = self.base_process.cumulative_integrated_intensity(t_grid, tr)
            s_nr = np.exp(-c * Lambda_t)
            s_r = np.power(beta / (beta + Lambda_t), r)
            return pi * s_nr + (1.0 - pi) * s_r

        if trial is not None:
            return t_grid, _s(int(trial))

        S_matrix = np.array([_s(tr) for tr in range(dataset.num_trials)])
        weights = dataset.n_fish_per_trial
        if weights.sum() == 0:
            return t_grid, np.mean(S_matrix, axis=0)
        return t_grid, np.average(S_matrix, axis=0, weights=weights)

    # -- Per-fish posterior class / gain (whole-session, for reporting) ----

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        """
        Whole-session posterior class responsibility and posterior mean
        gain per fish -- valid for descriptive reporting, NOT for
        time-rescaling (see _stream_tau_values docstring / same caveat as
        GammaMixedEffectsProcess._fish_scale_factors).
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted before estimating fish gains.")

        base_params, pi, r, c = self._split_params(self.params_)
        _, N_f, S_f = self.base_process.mixed_effects_likelihood_terms(dataset, base_params)
        mu_R = self._mu_responder(pi, c)
        beta = r / mu_R

        log_L_NR, log_L_R = self._log_branch_likelihoods(N_f, S_f, r, c, beta)
        log_pi = np.log(max(pi, 1e-300))
        log_1mpi = np.log(max(1.0 - pi, 1e-300))
        log_post_NR = log_pi + log_L_NR
        log_post_R = log_1mpi + log_L_R
        log_norm = np.logaddexp(log_post_NR, log_post_R)
        p_responder = np.exp(log_post_R - log_norm)

        with np.errstate(divide="ignore", invalid="ignore"):
            g_hat_responder = (N_f + r) / (S_f + beta)
        g_hat = (1.0 - p_responder) * c + p_responder * g_hat_responder

        active = dataset.fish_trial_mask.any(axis=1)
        return pd.DataFrame({
            "fish_idx": np.arange(dataset.num_fish)[active],
            "n_events": N_f[active],
            "expected_events_base": S_f[active],
            "p_responder": p_responder[active],
            "estimated_gain": g_hat[active],
        })

    def mixed_effects_likelihood_terms(self, dataset, params):
        raise NotImplementedError(
            "ZeroInflatedGammaMixedEffectsProcess's marginal gain distribution "
            "is a two-component mixture (point mass + Gamma), not a single "
            "Gamma -- it does NOT reduce to the (base_ll, N_f, S_f) "
            "sufficient-statistic contract required by GammaMixedEffectsProcess "
            "(or any other consumer of mixed_effects_likelihood_terms). It "
            "already IS its own frailty-integrated model; do not wrap it further."
        )

    # -- Dispersion reporting ------------------------------------------------

    @property
    def dispersion_r(self) -> float:
        """
        EFFECTIVE Gamma-equivalent dispersion: the r of a single Gamma(r,r)
        frailty producing the SAME total gain variance as this mixture
        (Var(g_f) = 1/r_eff, since E[g_f]=1 by construction). Used by
        compute_residuals()'s NB variance formula and ModelComparator's
        summary table -- using the raw fitted r_responder there would
        understate total heterogeneity (it ignores the between-class
        variance contributed by pi/c).

        Reduces EXACTLY to r_responder when pi=0 (c-branch vanishes,
        mu_R=1) -- strict generalization of GammaMixedEffectsProcess.dispersion_r.

        CAVEAT: matches total VARIANCE only, not shape -- see class docstring.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        _, pi, r, c = self._split_params(self.params_)
        mu_R = self._mu_responder(pi, c)
        e_g2 = pi * c**2 + (1.0 - pi) * mu_R**2 * (1.0 + 1.0 / r)
        var_g = max(e_g2 - 1.0, 1e-12)
        return float(1.0 / var_g)

    @property
    def r_responder(self) -> float:
        """Raw fitted Gamma shape/rate parameter for the RESPONDER sub-population only."""
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        _, _, r, _ = self._split_params(self.params_)
        return r

    @property
    def pi_nonresponder(self) -> float:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        _, pi, _, _ = self._split_params(self.params_)
        return pi

    @property
    def c_nonresponder(self) -> float:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        _, _, _, c = self._split_params(self.params_)
        return c

    @property
    def is_survival(self) -> bool:
        return self.base_process.is_survival

    # -- Simulation ----------------------------------------------------------

    def _draw_fish_gains(self, num_fish, n_sims, rng):
        """Used by generate_model_predicted_counts / plot_predicted_vs_observed."""
        _, pi, r, c = self._split_params(self.params_)
        mu_R = self._mu_responder(pi, c)
        is_nonresponder = rng.random((num_fish, n_sims)) < pi
        gains = rng.gamma(shape=r, scale=mu_R / r, size=(num_fish, n_sims))
        gains[is_nonresponder] = c
        return gains

    def simulate_stream(self, dataset, t_idx, gain, rng) -> np.ndarray:
        return self.base_process.simulate_stream(dataset, t_idx, gain, rng)

    def _intensity_upper_bound(self, dataset: PointProcessDataset, t_idx: int) -> float:
        return self.base_process._intensity_upper_bound(dataset, t_idx)