# point_process/zero_inflated_baseline_only_frailty_hawkes.py
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.special import gammaln, roots_genlaguerre, logsumexp

from .point_process import PointProcess
from .dataset import PointProcessDataset
from .hawkes_process import HawkesProcess
from .baseline_only_frailty_hawkes import BaselineOnlyFrailtyHawkesProcess
from .kernel_shapes import sigmoid_bounded, logit_bounded


class ZeroInflatedBaselineOnlyFrailtyHawkesProcess(BaselineOnlyFrailtyHawkesProcess):
    """
    BaselineOnlyFrailtyHawkesProcess + ZeroInflatedGammaMixedEffectsProcess's
    two-population gain distribution, combined:

        g_f ~ pi * delta_c + (1-pi) * Gamma(r, r/mu_R),   E[g_f] = 1

    g_f still scales ONLY the baseline term (never self-excitation) -- the
    parent class's core safety guarantee is fully preserved: no fish's
    branching ratio can exceed alpha_hawkes/beta_hawkes regardless of gain,
    responder or not.

    STRICT GENERALIZATION of BaselineOnlyFrailtyHawkesProcess: pi -> 0
    recovers it exactly (log_L_NR branch gets -inf mixing weight, beta -> r,
    mu_R -> 1) -- verify this if you ever touch _fish_log_likelihood.

    NON-CONJUGACY IS UNCHANGED FROM THE PARENT: g_f multiplying only the
    baseline term already broke the "uniform multiplicative gain" contract
    that lets GammaMixedEffectsProcess/ZeroInflatedGammaMixedEffectsProcess
    integrate out g_f in closed form. So EACH branch of this mixture is
    computed the same way the parent computes its single Gamma branch:
    the non-responder branch is closed-form (deterministic g=c, O(1), no
    integral); the responder branch is the SAME generalized Gauss-Laguerre
    quadrature as the parent, just with rate beta=r/mu_R instead of r. Total
    added cost over the parent is negligible (one O(1) term per fish).

    "NON-RESPONDER" UNDER SELF-EXCITATION (read before setting fit_c=False
    for real): because gain scales ONLY the baseline term, a non-responder
    (g_f=c, esp. c=0) can still produce OBSERVED events if the history term
    is nonzero at that instant -- e.g. a spontaneous event triggering a
    short self-exciting burst. This is intentional here: it lets "no
    stimulus-locked drive" and "capable of self-sustaining bursts" be fit
    as separate, non-confounded phenomena, rather than forcing any fish
    with >=1 event into the responder branch. See _fish_log_likelihood.

    COMPENSATOR CAVEAT (inherited, unchanged in spirit from the parent):
    _stream_tau_values uses the same PLUG-IN posterior-mean-gain
    approximation as BaselineOnlyFrailtyHawkesProcess -- not an exact
    marginal survival compensator. The same non-conjugate structure that
    forces quadrature in the likelihood also prevents an exact closed-form
    compensator here; only the posterior-mean-gain computation itself is
    now mixture-aware (_posterior_mean_gain below).

    IDENTIFIABILITY: pi=0 is a boundary-of-parameter-space test (Self &
    Liang 1987) -- treat a chi2 LRT against BaselineOnlyFrailtyHawkesProcess
    as conservative; prefer a parametric bootstrap of the LR statistic.

    KNOWN GAP CARRIED FROM THE PARENT: BaselineOnlyFrailtyHawkesProcess does
    not itself override _draw_fish_gains, so its own
    plot_predicted_vs_observed silently uses gain=1 for every simulated
    fish (PointProcess's no-frailty default) rather than actually sampling
    its fitted frailty distribution. This class DOES override
    _draw_fish_gains correctly (see below) -- but be aware the parent's own
    diagnostic panel doesn't reflect its fitted heterogeneity if compared
    side-by-side.
    """

    def __init__(
        self,
        base_process: HawkesProcess,
        pi_init: float = 0.3,
        r_init: float = 5.0,
        fit_c: bool = False,
        c_init: float = 0.0,
        c_upper: float = 0.3,
        n_quad_nodes: int = 30,
    ):
        # Deliberately calls PointProcess.__init__, NOT
        # BaselineOnlyFrailtyHawkesProcess.__init__ -- the parameter tail
        # layout differs (pi, r, [c] instead of just r), so re-deriving
        # initial_guesses/bounds/param_names here avoids ever relying on
        # the parent's (wrong-shaped) versions.
        PointProcess.__init__(self, base_process.integration_dt)
        if not hasattr(base_process, "history_kernel"):
            raise TypeError(
                "ZeroInflatedBaselineOnlyFrailtyHawkesProcess requires a "
                "HawkesProcess base_process (needs .kernel, .history_kernel, "
                "._split_params)."
            )
        self.base_process = base_process
        self.n_quad_nodes = n_quad_nodes
        self.fit_c = fit_c
        self.c_upper = c_upper
        self._fixed_c = c_init

        self.name = f"ZeroInflatedBaselineOnlyFrailty[{base_process.name}]"
        base_formula = base_process.latex_formula.strip("$")
        self.latex_formula = (
            rf"${base_formula}$ (baseline $\times\, g_f$), "
            rf"$g_f \sim \pi\,\delta_c + (1-\pi)\,\Gamma(r, r/\mu_R)$"
        )

        z_pi_init = float(logit_bounded(pi_init, 1.0))
        extra_names = ["z_pi_nonresponder", "r_dispersion"]
        extra_guesses = [z_pi_init, r_init]
        # r floored at 1e-2, matching the parent: alpha=r-1 approaching -1
        # makes generalized Gauss-Laguerre numerically fragile.
        extra_bounds: List[Tuple[Optional[float], Optional[float]]] = [
            (-15.0, 15.0),
            (1e-2, None),
        ]

        if fit_c:
            z_c_init = float(logit_bounded(max(c_init, 1e-4), c_upper))
            extra_names.append("z_c_nonresponder")
            extra_guesses.append(z_c_init)
            extra_bounds.append((-15.0, 15.0))

        self.initial_guesses = base_process.initial_guesses + extra_guesses
        self.bounds = base_process.bounds + extra_bounds
        self.param_names = base_process.param_names + extra_names

    # -- Parameter bookkeeping (OVERRIDES parent: different tail layout) ---

    def _split_params(
        self, params: List[float]
    ) -> Tuple[List[float], float, float, float]:
        """
        OVERRIDES BaselineOnlyFrailtyHawkesProcess._split_params (which
        returns (base_params, r)). Returns (base_params, pi, r, c) instead
        -- every method below that calls this must unpack 4 values, not 2.
        """
        n_base = len(self.base_process.param_names)
        base_params = list(params[:n_base])
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
        # PointProcess.fit directly -- BaselineOnlyFrailtyHawkesProcess.fit
        # would call our _split_params expecting a 2-tuple and crash.
        PointProcess.fit(self, dataset, method=method, **kwargs)
        base_params, *_ = self._split_params(self.params_)
        self.base_process.params_ = np.asarray(base_params, dtype=float)
        self.base_process.param_dict_ = dict(
            zip(self.base_process.param_names, base_params)
        )
        return self

    def set_params(self, params: np.ndarray) -> None:
        PointProcess.set_params(self, params)
        base_params, *_ = self._split_params(list(self.params_))
        self.base_process.set_params(base_params)

    # -- Likelihood (OVERRIDES parent: mixture instead of single Gamma) ----

    def _fish_log_likelihood(
        self,
        events,
        S_base_f: float,
        S_hist_f: float,
        pi: float,
        r: float,
        c: float,
        beta: float,
    ) -> float:
        """
        Mixture generalization of the parent's _fish_log_likelihood: the
        non-responder branch is closed-form (deterministic g=c); the
        responder branch is IDENTICAL quadrature to the parent, generalized
        from Gamma(r,r) to Gamma(r, beta).
        """
        if events:
            br = np.array([e[0] for e in events])
            hr = np.array([e[1] for e in events])
            intensities_nr = np.maximum(c * br + hr, 1e-300)
            log_L_NR_raw = -c * S_base_f + float(np.sum(np.log(intensities_nr)))
        else:
            br = hr = np.array([])
            log_L_NR_raw = -c * S_base_f

        alpha = max(r - 1.0, -0.999)
        nodes, weights = roots_genlaguerre(self.n_quad_nodes, alpha)
        denom = S_base_f + beta
        g_vals = nodes / denom

        log_terms_per_node = np.zeros(self.n_quad_nodes)
        if events:
            for k, g in enumerate(g_vals):
                intensities = np.maximum(g * br + hr, 1e-300)
                log_terms_per_node[k] = np.sum(np.log(intensities))

        log_weighted = np.log(np.maximum(weights, 1e-300)) + log_terms_per_node
        log_quad_sum = logsumexp(log_weighted)
        log_prefactor = r * np.log(beta) - gammaln(r) - r * np.log(denom)
        log_L_R_raw = log_prefactor + log_quad_sum

        log_pi = np.log(max(pi, 1e-300))
        log_1mpi = np.log(max(1.0 - pi, 1e-300))
        log_mix_raw = np.logaddexp(log_pi + log_L_NR_raw, log_1mpi + log_L_R_raw)

        return log_mix_raw - S_hist_f

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        base_params, pi, r, c = self._split_params(params)
        mu_R = self._mu_responder(pi, c)
        beta = r / mu_R

        # _per_fish_terms is inherited UNCHANGED from the parent -- it only
        # touches self.base_process, never self.params_ directly.
        event_pairs, S_base, S_hist = self._per_fish_terms(dataset, base_params)

        total_ll = 0.0
        active = dataset.fish_trial_mask.any(axis=1)
        for f_idx in np.where(active)[0]:
            total_ll += self._fish_log_likelihood(
                event_pairs.get(f_idx, []), S_base[f_idx], S_hist[f_idx], pi, r, c, beta
            )
        return -total_ll

    # -- Mixture-aware posterior mean gain (OVERRIDES parent's signature) --

    def _posterior_mean_gain(
        self,
        events_so_far,
        S_base_so_far: float,
        pi: float,
        r: float,
        c: float,
        beta: float,
    ) -> float:
        """
        OVERRIDES BaselineOnlyFrailtyHawkesProcess._posterior_mean_gain
        (different signature: pi/r/c/beta instead of just r). Posterior
        class responsibility combined with the responder branch's posterior
        mean gain (quadrature, same style as the parent). Used identically
        by _stream_tau_values (STRICTLY-before-current-point, predictable)
        and estimate_fish_gains (whole-session).
        """
        if events_so_far:
            br = np.array([e[0] for e in events_so_far])
            hr = np.array([e[1] for e in events_so_far])
            intensities_nr = np.maximum(c * br + hr, 1e-300)
            log_L_NR_raw = -c * S_base_so_far + float(np.sum(np.log(intensities_nr)))
        else:
            br = hr = np.array([])
            log_L_NR_raw = -c * S_base_so_far

        alpha = max(r - 1.0, -0.999)
        nodes, weights = roots_genlaguerre(self.n_quad_nodes, alpha)
        denom = S_base_so_far + beta
        g_vals = nodes / denom

        log_terms = np.zeros(self.n_quad_nodes)
        if events_so_far:
            for k, g in enumerate(g_vals):
                log_terms[k] = np.sum(np.log(np.maximum(g * br + hr, 1e-300)))

        log_w = np.log(np.maximum(weights, 1e-300)) + log_terms
        log_w_norm = log_w - logsumexp(log_w)
        post_weights_R = np.exp(log_w_norm)
        e_g_given_R = float(np.sum(post_weights_R * g_vals))

        log_prefactor = r * np.log(beta) - gammaln(r) - r * np.log(denom)
        log_L_R_raw = log_prefactor + logsumexp(log_w)

        log_pi = np.log(max(pi, 1e-300))
        log_1mpi = np.log(max(1.0 - pi, 1e-300))
        log_post_NR = log_pi + log_L_NR_raw
        log_post_R = log_1mpi + log_L_R_raw
        log_norm = np.logaddexp(log_post_NR, log_post_R)
        p_R = float(np.exp(log_post_R - log_norm))

        return (1.0 - p_R) * c + p_R * e_g_given_R

    # -- estimate_fish_gains (OVERRIDES: adds p_responder, mixture gain) ---

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        if self.params_ is None:
            raise ValueError("Model must be fitted before estimating fish gains.")
        base_params, pi, r, c = self._split_params(self.params_)
        mu_R = self._mu_responder(pi, c)
        beta = r / mu_R
        event_pairs, S_base, S_hist = self._per_fish_terms(dataset, base_params)
        active = dataset.fish_trial_mask.any(axis=1)

        gains = np.array(
            [
                self._posterior_mean_gain(
                    event_pairs.get(f, []), S_base[f], pi, r, c, beta
                )
                for f in np.where(active)[0]
            ]
        )
        return pd.DataFrame(
            {
                "fish_idx": np.where(active)[0],
                "n_events": [len(event_pairs.get(f, [])) for f in np.where(active)[0]],
                "estimated_gain": gains,
            }
        )

    def _stream_tau_values(
        self,
        dataset: PointProcessDataset,
    ) -> Dict[Tuple[int, int], List[Tuple[float, bool]]]:
        """
        Exact marginal time-rescaling residuals for the zero-inflated,
        baseline-only frailty Hawkes model.

        Every active recurrent trial contributes one exact interval per event
        and one terminal right-censored interval. Hawkes history resets between
        trials, while fish-level mixture/frailty information persists.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")

        base_params, pi, r, c = self._split_params(list(self.params_))
        mu_responder = self._mu_responder(pi, c)
        beta = r / mu_responder

        kernel_params, hist_params = self.base_process._split_params(base_params)

        kernel = self.base_process.kernel
        history_kernel = self.base_process.history_kernel
        integration_dt = self.base_process.integration_dt

        result: Dict[
            Tuple[int, int],
            List[Tuple[float, bool]],
        ] = {}

        for f_idx in range(dataset.num_fish):
            fish_events_so_far: List[Tuple[float, float]] = []
            S_base_so_far = 0.0

            for t_idx in range(dataset.num_trials):
                if not dataset.fish_trial_mask[f_idx, t_idx]:
                    continue

                t_ev = dataset._stream_index.get(
                    (f_idx, t_idx),
                    np.array([], dtype=float),
                )
                t_ev = np.sort(np.asarray(t_ev, dtype=float))

                trial_events: List[float] = []
                previous_time = 0.0
                pairs: List[Tuple[float, bool]] = []

                # ----------------------------------------------------------
                # Exact event-ending intervals
                # ----------------------------------------------------------

                for event_time_raw in t_ev:
                    event_time = float(event_time_raw)

                    d_base = float(
                        kernel.integrate(
                            event_time,
                            t_idx,
                            kernel_params,
                            integration_dt=integration_dt,
                        )
                        - kernel.integrate(
                            previous_time,
                            t_idx,
                            kernel_params,
                            integration_dt=integration_dt,
                        )
                    )

                    d_history = self._history_compensator_segment(
                        event_times_this_trial=trial_events,
                        a=previous_time,
                        b=event_time,
                        hist_params=hist_params,
                    )

                    log_evidence_before = self._log_mixture_gain_evidence(
                        events_so_far=fish_events_so_far,
                        S_base_so_far=S_base_so_far,
                        pi=pi,
                        r=r,
                        c=c,
                        beta=beta,
                    )
                    log_evidence_after_no_event = self._log_mixture_gain_evidence(
                        events_so_far=fish_events_so_far,
                        S_base_so_far=(S_base_so_far + d_base),
                        pi=pi,
                        r=r,
                        c=c,
                        beta=beta,
                    )

                    tau = d_history + log_evidence_before - log_evidence_after_no_event

                    if tau < -1e-9:
                        raise RuntimeError(
                            "Negative mixture marginal compensator increment: "
                            f"fish={f_idx}, trial={t_idx}, "
                            f"event_time={event_time}, tau={tau}."
                        )

                    pairs.append((float(max(tau, 0.0)), False))

                    S_base_so_far += d_base

                    base_rate = float(
                        kernel.evaluate(
                            np.array([event_time]),
                            np.array([t_idx]),
                            kernel_params,
                        )[0]
                    )

                    if trial_events:
                        lags = event_time - np.asarray(
                            trial_events,
                            dtype=float,
                        )
                        history_rate = float(
                            np.sum(
                                history_kernel.evaluate(
                                    lags,
                                    hist_params,
                                )
                            )
                        )
                    else:
                        history_rate = 0.0

                    fish_events_so_far.append((base_rate, history_rate))
                    trial_events.append(event_time)
                    previous_time = event_time

                # ----------------------------------------------------------
                # Administratively censored terminal interval
                # ----------------------------------------------------------

                d_base_terminal = float(
                    kernel.integrate(
                        dataset.duration_s,
                        t_idx,
                        kernel_params,
                        integration_dt=integration_dt,
                    )
                    - kernel.integrate(
                        previous_time,
                        t_idx,
                        kernel_params,
                        integration_dt=integration_dt,
                    )
                )

                d_history_terminal = self._history_compensator_segment(
                    event_times_this_trial=trial_events,
                    a=previous_time,
                    b=dataset.duration_s,
                    hist_params=hist_params,
                )

                log_evidence_before = self._log_mixture_gain_evidence(
                    events_so_far=fish_events_so_far,
                    S_base_so_far=S_base_so_far,
                    pi=pi,
                    r=r,
                    c=c,
                    beta=beta,
                )
                log_evidence_after_no_event = self._log_mixture_gain_evidence(
                    events_so_far=fish_events_so_far,
                    S_base_so_far=(S_base_so_far + d_base_terminal),
                    pi=pi,
                    r=r,
                    c=c,
                    beta=beta,
                )

                terminal_tau = (
                    d_history_terminal
                    + log_evidence_before
                    - log_evidence_after_no_event
                )

                if terminal_tau < -1e-9:
                    raise RuntimeError(
                        "Negative terminal mixture compensator increment: "
                        f"fish={f_idx}, trial={t_idx}, "
                        f"tau={terminal_tau}."
                    )

                pairs.append((float(max(terminal_tau, 0.0)), True))

                S_base_so_far += d_base_terminal
                result[(f_idx, t_idx)] = pairs

        return result

    # -- Simulation-support overrides (params layout changed) --------------

    def _base_exposure_for_stream(
        self, dataset: PointProcessDataset, t_idx: int
    ) -> float:
        base_params, _, _, _ = self._split_params(self.params_)
        kernel_params, _ = self.base_process._split_params(base_params)
        return self.base_process.kernel.integrate(
            dataset.duration_s, t_idx, kernel_params, self.base_process.integration_dt
        )

    def simulate_stream(
        self, dataset: PointProcessDataset, t_idx: int, gain: float, rng
    ) -> np.ndarray:
        """Identical Ogata-thinning logic to the parent's simulate_stream --
        only the params-splitting line changed (variable-length tail)."""
        base_params, _, _, _ = self._split_params(self.params_)
        kernel_params, hist_params = self.base_process._split_params(base_params)
        kernel = self.base_process.kernel
        hk = self.base_process.history_kernel

        def _base(t_scalar: float) -> float:
            return (
                gain
                * kernel.evaluate(
                    np.array([t_scalar]), np.array([t_idx]), kernel_params
                )[0]
            )

        def _history_intensity(t_eval: float, events: List[float]) -> float:
            if not events:
                return 0.0
            lags = t_eval - np.asarray(events)
            return float(np.sum(hk.evaluate(lags, hist_params)))

        base_upper = gain * self.base_process._intensity_upper_bound(dataset, t_idx)
        decay_horizon = self.base_process._estimate_decay_horizon(hist_params)
        lag_grid = np.arange(
            0.0, decay_horizon + self.integration_dt, self.integration_dt
        )
        envelope_grid = (
            hk.decay_envelope(lag_grid, hist_params) * self._THINNING_SAFETY_MARGIN
        )

        def _envelope(lag: float) -> float:
            if lag >= decay_horizon:
                return 0.0
            idx = min(int(lag / self.integration_dt), len(envelope_grid) - 1)
            return envelope_grid[max(idx, 0)]

        events: List[float] = []
        t = 0.0
        while t < dataset.duration_s:
            hist_upper = sum(_envelope(t - e) for e in events)
            lambda_upper = base_upper + hist_upper

            w = rng.exponential(1.0 / max(lambda_upper, 1e-12))
            t_candidate = t + w
            if t_candidate >= dataset.duration_s:
                break

            lam_candidate = _base(t_candidate) + _history_intensity(t_candidate, events)
            if rng.uniform() <= lam_candidate / lambda_upper:
                events.append(t_candidate)
            t = t_candidate

        return np.array(events)

    def _draw_fish_gains(self, num_fish, n_sims, rng):
        """Used by generate_model_predicted_counts / plot_predicted_vs_observed
        -- NOTE the parent class doesn't override this (see class docstring)."""
        _, pi, r, c = self._split_params(self.params_)
        mu_R = self._mu_responder(pi, c)
        is_nonresponder = rng.random((num_fish, n_sims)) < pi
        gains = rng.gamma(shape=r, scale=mu_R / r, size=(num_fish, n_sims))
        gains[is_nonresponder] = c
        return gains

    # -- Dispersion reporting ------------------------------------------------

    @property
    def dispersion_r(self) -> float:
        """
        EFFECTIVE Gamma-equivalent dispersion matching TOTAL gain variance
        (see ZeroInflatedGammaMixedEffectsProcess.dispersion_r for the same
        derivation/caveat) -- reduces to r_responder exactly when pi=0.
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

    def mixed_effects_likelihood_terms(self, dataset, params):
        raise NotImplementedError(
            "ZeroInflatedBaselineOnlyFrailtyHawkesProcess satisfies NEITHER "
            "precondition required by mixed_effects_likelihood_terms: (1) "
            "gain here multiplies ONLY the baseline term, not history (same "
            "issue as BaselineOnlyFrailtyHawkesProcess), and (2) its gain "
            "distribution is a two-component mixture, not a single Gamma "
            "(same issue as ZeroInflatedGammaMixedEffectsProcess). It "
            "already IS its own frailty-integrated model; use its _nll "
            "directly rather than composing it further."
        )

    def _log_mixture_gain_evidence(
        self,
        events_so_far: List[Tuple[float, float]],
        S_base_so_far: float,
        pi: float,
        r: float,
        c: float,
        beta: float,
    ) -> float:
        """
        Log gain-dependent marginal evidence for the point-mass/Gamma mixture.

            Z_mix(S) =
                pi Z_NR(S) + (1-pi) Z_R(S).

        The exact marginal interval compensator is

            tau = dH + log Z_mix(S) - log Z_mix(S + dB).

        ``Z_NR`` is evaluated at deterministic gain c. ``Z_R`` is evaluated by
        generalized Gauss-Laguerre quadrature for Gamma(r, beta).
        """
        S_base_so_far = float(S_base_so_far)
        r = max(float(r), 1e-8)
        beta = max(float(beta), 1e-12)

        if events_so_far:
            base_rates = np.asarray(
                [event[0] for event in events_so_far],
                dtype=float,
            )
            history_rates = np.asarray(
                [event[1] for event in events_so_far],
                dtype=float,
            )
        else:
            base_rates = np.array([], dtype=float)
            history_rates = np.array([], dtype=float)

        # --------------------------------------------------------------
        # Deterministic nonresponder branch
        # --------------------------------------------------------------

        log_nonresponder = -c * S_base_so_far

        if len(base_rates) > 0:
            intensities_nr = c * base_rates + history_rates

            if np.any(intensities_nr <= 0.0):
                log_nonresponder = -np.inf
            else:
                log_nonresponder += float(np.sum(np.log(intensities_nr)))

        # --------------------------------------------------------------
        # Gamma responder branch
        # --------------------------------------------------------------

        alpha = max(r - 1.0, -0.999)
        nodes, weights = roots_genlaguerre(
            self.n_quad_nodes,
            alpha,
        )

        denom = beta + S_base_so_far
        gains = nodes / denom

        log_event_terms = np.zeros(
            self.n_quad_nodes,
            dtype=float,
        )

        if len(base_rates) > 0:
            for q, gain in enumerate(gains):
                intensities = gain * base_rates + history_rates

                if np.any(intensities <= 0.0):
                    log_event_terms[q] = -np.inf
                else:
                    log_event_terms[q] = float(np.sum(np.log(intensities)))

        log_responder = (
            r * np.log(beta)
            - gammaln(r)
            - r * np.log(denom)
            + logsumexp(np.log(np.maximum(weights, 1e-300)) + log_event_terms)
        )

        return float(
            np.logaddexp(
                np.log(max(pi, 1e-300)) + log_nonresponder,
                np.log(max(1.0 - pi, 1e-300)) + log_responder,
            )
        )
