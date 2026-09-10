# point_process/baseline_only_frailty_process.py
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import gammaln, roots_genlaguerre, logsumexp

from .point_process import PointProcess
from .dataset import PointProcessDataset
from .hawkes_process import HawkesProcess


class BaselineOnlyFrailtyHawkesProcess(PointProcess):
    """
    lambda_f(t) = g_f * kernel(t, m) + history(t),  g_f ~ Gamma(r, r)

    g_f scales ONLY the baseline rate, never the self-exciting history
    term -- unlike GammaMixedEffectsProcess(HawkesProcess(...)), where
    g_f scaling the WHOLE intensity makes the effective branching ratio
    g_f*alpha/beta, so a large gain draw can make a fish's own
    self-excitation super-critical (empirically confirmed: simulating
    that model produced streams with catastrophic event counts for a
    small fraction of fish, badly distorting plot_predicted_vs_observed).
    Here, alpha_hawkes/beta_hawkes are shared population constants --
    the branching ratio is exactly alpha/beta for every fish, always,
    regardless of g_f. No fish can destabilize its own dynamics.

    E[g_f] = 1, so E[lambda_f(t)] = kernel(t,m) + history(t) EXACTLY
    matches base_process's own (population-level) intensity -- predict/
    compute_expected_rate/cumulative_integrated_intensity all delegate
    directly to base_process with NO extra derivation needed (unlike
    GammaMixedEffectsProcess, which only needed this for the base rate).

    COST: no closed-form marginal (that conjugacy needs g_f to scale the
    ENTIRE intensity uniformly -- see HawkesProcess.mixed_effects_
    likelihood_terms docstring). Integrates out g_f per fish via
    GENERALIZED Gauss-Laguerre quadrature (weight x^(r-1)*e^-x, matching
    Gamma(r,r)'s own g^(r-1) term exactly -- NOT plain alpha=0 Laguerre,
    which converges poorly once r deviates from 1). Nodes/weights depend
    on r, so they're recomputed every _nll call.
    """

    def __init__(self, base_process: HawkesProcess, r_init: float = 5.0, n_quad_nodes: int = 30):
        super().__init__(base_process.integration_dt)
        if not hasattr(base_process, "history_kernel"):
            raise TypeError(
                "BaselineOnlyFrailtyHawkesProcess requires a HawkesProcess "
                "base_process (needs .kernel, .history_kernel, ._split_params)."
            )
        self.base_process = base_process
        self.n_quad_nodes = n_quad_nodes

        self.name = f"BaselineOnlyFrailty[{base_process.name}]"
        base_formula = base_process.latex_formula.strip("$")
        self.latex_formula = rf"${base_formula}$ (baseline $\times\, g_f$), $g_f\sim\Gamma(r,r)$"

        self.initial_guesses = base_process.initial_guesses + [r_init]
        # r floored at 1e-2, not 1e-3: alpha=r-1 approaching -1 makes
        # generalized Gauss-Laguerre numerically fragile.
        self.bounds = base_process.bounds + [(1e-2, None)]
        self.param_names = base_process.param_names + ["r_dispersion"]

    def _split_params(self, params: List[float]) -> Tuple[List[float], float]:
        return params[:-1], params[-1]

    def fit(self, dataset: PointProcessDataset, method: str = 'L-BFGS-B', **kwargs):
        super().fit(dataset, method=method, **kwargs)
        base_params, _ = self._split_params(self.params_)
        self.base_process.params_ = np.asarray(base_params, dtype=float)
        self.base_process.param_dict_ = dict(zip(self.base_process.param_names, base_params))
        return self

    def set_params(self, params: np.ndarray) -> None:
        super().set_params(params)
        base_params, _ = self._split_params(list(self.params_))
        self.base_process.set_params(base_params)
        
    # -- Likelihood -----------------------------------------------------

    def _per_fish_terms(
        self, dataset: PointProcessDataset, base_params: List[float]
    ) -> Tuple[Dict[int, List[Tuple[float, float]]], np.ndarray, np.ndarray]:
        """
        event_pairs[f]: list of (base_rate_i, history_rate_i) for EVERY
                        event across ALL of fish f's trials (history
                        resets per trial, matching HawkesProcess's own
                        convention).
        S_base[f]: total baseline exposure -- the part g_f multiplies.
        S_hist[f]: total history exposure -- population-level, NEVER
                   scaled by g_f.
        """
        kernel_params, hist_params = self.base_process._split_params(base_params)
        kernel = self.base_process.kernel
        history_kernel = self.base_process.history_kernel
        idt = self.base_process.integration_dt

        event_pairs: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        S_base = np.zeros(dataset.num_fish, dtype=float)
        S_hist = np.zeros(dataset.num_fish, dtype=float)

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            S_base[f_idx] += kernel.integrate(dataset.duration_s, t_idx, kernel_params, integration_dt=idt)

            if len(t_ev) > 0:
                trials_arr = np.full(t_ev.shape, t_idx, dtype=float)
                base_rates = kernel.evaluate(t_ev, trials_arr, kernel_params)
                hist_rates = history_kernel.event_history(t_ev, hist_params)
                event_pairs[f_idx].extend(zip(base_rates.tolist(), hist_rates.tolist()))

                remaining_time = dataset.duration_s - t_ev
                hist_integrals = history_kernel.integrate(remaining_time, hist_params, integration_dt=idt)
                S_hist[f_idx] += float(np.sum(hist_integrals))

        return event_pairs, S_base, S_hist

    def _fish_log_likelihood(self, events, S_base_f: float, S_hist_f: float, r: float) -> float:
        alpha = max(r - 1.0, -0.999)
        nodes, weights = roots_genlaguerre(self.n_quad_nodes, alpha)

        denom = S_base_f + r
        g_vals = nodes / denom

        log_terms_per_node = np.zeros(self.n_quad_nodes)
        if events:
            br = np.array([e[0] for e in events])
            hr = np.array([e[1] for e in events])
            for k, g in enumerate(g_vals):
                intensities = np.maximum(g * br + hr, 1e-300)
                log_terms_per_node[k] = np.sum(np.log(intensities))

        log_weighted = np.log(np.maximum(weights, 1e-300)) + log_terms_per_node
        log_quad_sum = logsumexp(log_weighted)
        log_prefactor = r * np.log(r) - gammaln(r) - r * np.log(denom)

        return log_prefactor + log_quad_sum - S_hist_f

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        base_params, r = self._split_params(params)
        r = max(r, 1e-8)

        event_pairs, S_base, S_hist = self._per_fish_terms(dataset, base_params)

        total_ll = 0.0
        active = dataset.fish_trial_mask.any(axis=1)
        for f_idx in np.where(active)[0]:
            total_ll += self._fish_log_likelihood(event_pairs.get(f_idx, []), S_base[f_idx], S_hist[f_idx], r)

        return -total_ll

    # -- Population-average prediction: delegates directly, no scaling needed --

    def predict(self, t, trial, **kwargs):
        """E[g_f]=1, so E[lambda] == base_process's own prediction exactly."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.base_process.predict(t, trial, **kwargs)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.base_process.compute_expected_rate(dataset)

    def cumulative_integrated_intensity(self, t_events, trial):
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        return self.base_process.cumulative_integrated_intensity(t_events, trial)

    @property
    def dispersion_r(self) -> float:
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        return float(self.params_[-1])

    def _base_exposure_for_stream(self, dataset: PointProcessDataset, t_idx: int) -> float:
        """Baseline-only exposure (gain applied by the caller, e.g. simulate_stream) --
        delegates to base_process's own kernel, ignoring its history term entirely
        (history is never gain-scaled in this model)."""
        kernel_params, _ = self.base_process._split_params(self.params_[:-1])
        return self.base_process.kernel.integrate(
            dataset.duration_s, t_idx, kernel_params, self.base_process.integration_dt
        )

    def _intensity_upper_bound(self, dataset: PointProcessDataset, t_idx: int) -> float:
        """Baseline kernel's own bound -- same helper base_process already
        implements for itself; delegate directly."""
        return self.base_process._intensity_upper_bound(dataset, t_idx)

    def simulate_stream(self, dataset: PointProcessDataset, t_idx: int, gain: float, rng) -> np.ndarray:
        """
        Ogata thinning with gain applied ONLY to the baseline term -- history
        (self-excitation) uses the SHARED, population-level alpha_hawkes/
        beta_hawkes, completely gain-independent. Unlike GammaMixedEffects
        Process(HawkesProcess(...)), there is no branching-ratio instability
        here: the effective branching ratio is always exactly alpha/beta,
        regardless of gain, because gain never touches the history term.
        """
        kernel_params, hist_params = self.base_process._split_params(self.params_[:-1])
        kernel = self.base_process.kernel
        hk = self.base_process.history_kernel

        def _base(t_scalar: float) -> float:
            return gain * kernel.evaluate(np.array([t_scalar]), np.array([t_idx]), kernel_params)[0]

        def _history_intensity(t_eval: float, events: List[float]) -> float:
            if not events:
                return 0.0
            lags = t_eval - np.asarray(events)
            return float(np.sum(hk.evaluate(lags, hist_params)))

        base_upper = gain * self.base_process._intensity_upper_bound(dataset, t_idx)
        decay_horizon = self.base_process._estimate_decay_horizon(hist_params)
        lag_grid = np.arange(0.0, decay_horizon + self.integration_dt, self.integration_dt)
        envelope_grid = hk.decay_envelope(lag_grid, hist_params) * self._THINNING_SAFETY_MARGIN

        def _envelope(lag: float) -> float:
            if lag >= decay_horizon:
                return 0.0
            idx = min(int(lag / self.integration_dt), len(envelope_grid) - 1)
            return envelope_grid[max(idx, 0)]

        events: List[float] = []
        t = 0.0
        while t < dataset.duration_s:
            # NOTE: no `gain *` here -- history is never gain-scaled, unlike
            # GammaMixedEffectsProcess(HawkesProcess(...))'s equivalent line.
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

    def mixed_effects_likelihood_terms(self, dataset, params):
        raise NotImplementedError(
            "BaselineOnlyFrailtyHawkesProcess does not satisfy the uniform-"
            "multiplicative-gain precondition required by "
            "mixed_effects_likelihood_terms (g_f here scales ONLY the "
            "baseline term, not history) -- it cannot be wrapped in "
            "GammaMixedEffectsProcess or any other consumer of this contract. "
            "It already IS its own frailty-integrated model; use its _nll "
            "directly rather than composing it further."
        )

    def _posterior_mean_gain(self, events_so_far: List[Tuple[float, float]], S_base_so_far: float, r: float) -> float:
        """
        E[g_f | events strictly before current point], via generalized
        Gauss-Laguerre quadrature -- the quadrature analog of
        GammaMixedEffectsProcess._fish_scale_factors's closed-form (r+N)/(r+S),
        generalized to baseline-only scaling. Used ONLY with events/exposure
        STRICTLY BEFORE the point being rescaled (see _stream_tau_values),
        preserving Ogata's predictability requirement.
        """
        alpha = max(r - 1.0, -0.999)
        nodes, weights = roots_genlaguerre(self.n_quad_nodes, alpha)
        denom = S_base_so_far + r
        g_vals = nodes / denom

        log_terms = np.zeros(self.n_quad_nodes)
        if events_so_far:
            br = np.array([e[0] for e in events_so_far])
            hr = np.array([e[1] for e in events_so_far])
            for k, g in enumerate(g_vals):
                log_terms[k] = np.sum(np.log(np.maximum(g * br + hr, 1e-300)))

        log_w = np.log(np.maximum(weights, 1e-300)) + log_terms
        log_w_norm = log_w - logsumexp(log_w)
        post_weights = np.exp(log_w_norm)
        return float(np.sum(post_weights * g_vals))

    def _history_compensator_segment(
        self, event_times_this_trial: List[float], a: float, b: float, hist_params: List[float],
    ) -> float:
        """
        Integral of history(t) = sum_j history_kernel.evaluate(t - e_j) over
        t in [a, b], for e_j in event_times_this_trial (all e_j <= a, since
        this is always called walking forward sequentially through one
        trial's own events). History resets at trial boundaries -- callers
        must NEVER pass events from a different trial here, matching
        HawkesProcess's own convention (history_kernel.integrate/event_history
        are always called per-trial elsewhere in this codebase too).

        General identity: integral of h(t-e_j) over [a,b] = H(b-e_j) - H(a-e_j),
        where H(x) = history_kernel.integrate(x, ...) is the cumulative
        integral from 0 -- this is exactly the same identity HawkesProcess.
        _hawkes_nll already uses (remaining_time = duration_s - t_events,
        then history_kernel.integrate(remaining_time, ...)), just generalized
        from "integrate up to duration_s" to "integrate between two arbitrary
        points a and b."
        """
        if not event_times_this_trial:
            return 0.0
        events_arr = np.array(event_times_this_trial)
        lag_b = np.maximum(b - events_arr, 0.0)
        lag_a = np.maximum(a - events_arr, 0.0)
        hk = self.base_process.history_kernel
        H_b = np.asarray(hk.integrate(lag_b, hist_params, integration_dt=self.integration_dt))
        H_a = np.asarray(hk.integrate(lag_a, hist_params, integration_dt=self.integration_dt))
        return float(np.sum(H_b - H_a))

    def _stream_tau_values(
        self,
        dataset: PointProcessDataset,
    ) -> Dict[Tuple[int, int], List[Tuple[float, bool]]]:
        """
        Exact marginal time-rescaling residuals for baseline-only Gamma frailty.

        For every active recurrent trial, returns:

        - one exact residual for each observed event;
        - one right-censored residual from the final event to trial end;
        - for an empty trial, one right-censored residual from trial onset to
        trial end.

        Hawkes event history resets at every trial boundary. Fish-level frailty
        information persists across trials through ``fish_events_so_far`` and
        ``S_base_so_far``.

        The interval compensator is evaluated from the marginal no-event
        probability:

            tau = dH + log Z(S) - log Z(S + dB),

        rather than by holding a posterior-mean gain fixed over the interval.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")

        base_params, r = self._split_params(
            list(self.params_)
        )
        kernel_params, hist_params = (
            self.base_process._split_params(base_params)
        )

        kernel = self.base_process.kernel
        history_kernel = self.base_process.history_kernel
        integration_dt = self.base_process.integration_dt

        result: Dict[
            Tuple[int, int],
            List[Tuple[float, bool]],
        ] = {}

        for f_idx in range(dataset.num_fish):
            # Information about the persistent fish-level gain. Event-history
            # rates here are the rates at the times of earlier events; they do
            # not imply that Hawkes history crosses trial boundaries.
            fish_events_so_far: List[
                Tuple[float, float]
            ] = []
            S_base_so_far = 0.0

            for t_idx in range(dataset.num_trials):
                if not dataset.fish_trial_mask[f_idx, t_idx]:
                    continue

                t_ev = dataset._stream_index.get(
                    (f_idx, t_idx),
                    np.array([], dtype=float),
                )
                t_ev = np.sort(
                    np.asarray(t_ev, dtype=float)
                )

                trial_events: List[float] = []
                previous_time = 0.0
                pairs: List[Tuple[float, bool]] = []

                # ----------------------------------------------------------
                # Exact event-ending intervals
                # ----------------------------------------------------------

                for event_time_raw in t_ev:
                    event_time = float(event_time_raw)

                    base_integral_end = kernel.integrate(
                        event_time,
                        t_idx,
                        kernel_params,
                        integration_dt=integration_dt,
                    )
                    base_integral_start = kernel.integrate(
                        previous_time,
                        t_idx,
                        kernel_params,
                        integration_dt=integration_dt,
                    )
                    d_base = float(
                        base_integral_end
                        - base_integral_start
                    )

                    d_history = (
                        self._history_compensator_segment(
                            event_times_this_trial=trial_events,
                            a=previous_time,
                            b=event_time,
                            hist_params=hist_params,
                        )
                    )

                    log_evidence_before = (
                        self._log_gain_evidence(
                            events_so_far=fish_events_so_far,
                            S_base_so_far=S_base_so_far,
                            r=r,
                        )
                    )
                    log_evidence_after_no_event = (
                        self._log_gain_evidence(
                            events_so_far=fish_events_so_far,
                            S_base_so_far=(
                                S_base_so_far + d_base
                            ),
                            r=r,
                        )
                    )

                    tau = (
                        d_history
                        + log_evidence_before
                        - log_evidence_after_no_event
                    )

                    if tau < -1e-9:
                        raise RuntimeError(
                            "Negative marginal compensator increment: "
                            f"fish={f_idx}, trial={t_idx}, "
                            f"event_time={event_time}, tau={tau}."
                        )

                    pairs.append(
                        (float(max(tau, 0.0)), False)
                    )

                    # Update baseline exposure before adding the current event.
                    S_base_so_far += d_base

                    base_rate = float(
                        kernel.evaluate(
                            np.array([event_time]),
                            np.array([t_idx]),
                            kernel_params,
                        )[0]
                    )

                    if trial_events:
                        lags = (
                            event_time
                            - np.asarray(
                                trial_events,
                                dtype=float,
                            )
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

                    fish_events_so_far.append(
                        (base_rate, history_rate)
                    )
                    trial_events.append(event_time)
                    previous_time = event_time

                # ----------------------------------------------------------
                # Administratively censored terminal interval
                # ----------------------------------------------------------

                terminal_base_integral = kernel.integrate(
                    dataset.duration_s,
                    t_idx,
                    kernel_params,
                    integration_dt=integration_dt,
                )
                previous_base_integral = kernel.integrate(
                    previous_time,
                    t_idx,
                    kernel_params,
                    integration_dt=integration_dt,
                )
                d_base_terminal = float(
                    terminal_base_integral
                    - previous_base_integral
                )

                d_history_terminal = (
                    self._history_compensator_segment(
                        event_times_this_trial=trial_events,
                        a=previous_time,
                        b=dataset.duration_s,
                        hist_params=hist_params,
                    )
                )

                log_evidence_before = (
                    self._log_gain_evidence(
                        events_so_far=fish_events_so_far,
                        S_base_so_far=S_base_so_far,
                        r=r,
                    )
                )
                log_evidence_after_no_event = (
                    self._log_gain_evidence(
                        events_so_far=fish_events_so_far,
                        S_base_so_far=(
                            S_base_so_far
                            + d_base_terminal
                        ),
                        r=r,
                    )
                )

                terminal_tau = (
                    d_history_terminal
                    + log_evidence_before
                    - log_evidence_after_no_event
                )

                if terminal_tau < -1e-9:
                    raise RuntimeError(
                        "Negative terminal marginal compensator increment: "
                        f"fish={f_idx}, trial={t_idx}, "
                        f"tau={terminal_tau}."
                    )

                pairs.append(
                    (float(max(terminal_tau, 0.0)), True)
                )

                # No event occurred during the terminal interval, but its
                # baseline exposure updates the frailty posterior carried into
                # the next trial.
                S_base_so_far += d_base_terminal

                result[(f_idx, t_idx)] = pairs

        return result

    def _draw_fish_gains(self, num_fish, n_sims, rng):
        r = self.dispersion_r
        return rng.gamma(
            shape=r,
            scale=1.0 / r,
            size=(num_fish, n_sims),
        )

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        """Whole-session posterior mean gain per fish (uses ALL of that fish's
        events/exposure, unlike _stream_tau_values's (t-)-only version) --
        valid for descriptive reporting, NOT for time-rescaling (see
        _stream_tau_values docstring / GammaMixedEffectsProcess._fish_scale_
        factors for why whole-session gains are invalid for that purpose)."""
        if self.params_ is None:
            raise ValueError("Model must be fitted before estimating fish gains.")
        base_params, r = self._split_params(self.params_)
        event_pairs, S_base, S_hist = self._per_fish_terms(dataset, base_params)
        active = dataset.fish_trial_mask.any(axis=1)

        gains = np.array([
            self._posterior_mean_gain(event_pairs.get(f, []), S_base[f], r)
            for f in np.where(active)[0]
        ])
        return pd.DataFrame({
            "fish_idx": np.where(active)[0],
            "n_events": [len(event_pairs.get(f, [])) for f in np.where(active)[0]],
            "estimated_gain": gains,
        })


    def _log_gain_evidence(
        self,
        events_so_far: List[Tuple[float, float]],
        S_base_so_far: float,
        r: float,
    ) -> float:
        """
        Log marginal evidence for the gain-dependent part of one fish's history.

        For prior event pairs (b_l, h_l),

            Z(S) =
                integral p(g) exp(-g S)
                product_l (g b_l + h_l) dg,

        with g ~ Gamma(r, r).

        Terms involving only the Hawkes-history exposure do not depend on g and
        are deliberately omitted. They cancel in the evidence ratio used for
        the marginal no-event compensator.

        The exact marginal compensator increment over an interval with baseline
        exposure dB and history exposure dH is

            tau = dH + log Z(S) - log Z(S + dB).

        The computation is exact up to generalized Gauss-Laguerre quadrature.
        """
        r = max(float(r), 1e-8)
        S_base_so_far = float(S_base_so_far)

        alpha = max(r - 1.0, -0.999)
        nodes, weights = roots_genlaguerre(
            self.n_quad_nodes,
            alpha,
        )

        denom = r + S_base_so_far
        gains = nodes / denom

        log_event_terms = np.zeros(
            self.n_quad_nodes,
            dtype=float,
        )

        if events_so_far:
            base_rates = np.asarray(
                [event[0] for event in events_so_far],
                dtype=float,
            )
            history_rates = np.asarray(
                [event[1] for event in events_so_far],
                dtype=float,
            )

            for q, gain in enumerate(gains):
                intensities = (
                    gain * base_rates + history_rates
                )

                if np.any(intensities <= 0.0):
                    log_event_terms[q] = -np.inf
                else:
                    log_event_terms[q] = float(
                        np.sum(np.log(intensities))
                    )

        log_quadrature = logsumexp(
            np.log(np.maximum(weights, 1e-300))
            + log_event_terms
        )

        return float(
            r * np.log(r)
            - gammaln(r)
            - r * np.log(denom)
            + log_quadrature
        )