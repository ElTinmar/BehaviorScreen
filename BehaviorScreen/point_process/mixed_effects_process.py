from typing import Any, Dict, List, Tuple, Union

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.special import gammaln

from .point_process import PointProcess
from .dataset import PointProcessDataset


class GammaMixedEffectsProcess(PointProcess):
    """
    Wraps any base PointProcess implementing mixed_effects_likelihood_terms,
    adding a shared per-fish multiplicative random effect g_f ~ Gamma(r, r),
    E[g_f] = 1, marginalized analytically rather than fit as a free
    parameter per fish.

    Equivalent construction to a "shared frailty model" in the survival /
    recurrent-event literature (Vaupel, Manton & Stallard 1979; Duchateau &
    Janssen, "The Frailty Model", 2008), generalized here to work with any
    base intensity (Poisson, Renewal, Hawkes) rather than just a baseline
    hazard -- see PointProcess.mixed_effects_likelihood_terms docstring for
    the precondition this requires of the base process.

    OWNERSHIP: base_process is owned exclusively by this wrapper. fit()
    permanently syncs the fitted base-effect parameter slice onto
    base_process.params_/param_dict_, so predict()/compute_expected_rate()/
    cumulative_integrated_intensity() can delegate to base_process's own
    methods directly with no extra plumbing. Do not fit base_process
    independently, or share the same instance across two wrappers -- either
    would silently overwrite this synced state.

    LIMITATION: predict(), compute_expected_rate(), and
    cumulative_integrated_intensity() all describe the POPULATION-AVERAGE
    process (E[g_f] = 1), not any specific fish's rate. Use
    estimate_fish_gains() for per-fish posterior-mean estimates.
    """

    def __init__(self, base_process: PointProcess, r_init: float = 5.0):
        super().__init__(base_process.integration_dt)
        self.base_process = base_process
        self.name = f"GammaMixedEffects[{base_process.name}]"

        base_formula = base_process.latex_formula.strip("$")
        self.latex_formula = rf"${base_formula} \times g_f,\ g_f\sim\Gamma(r,r)$"
        self.initial_guesses = base_process.initial_guesses + [r_init]
        self.bounds = base_process.bounds + [(1e-3, None)]
        self.param_names = base_process.param_names + ["r_dispersion"]

    def _split_params(self, params: List[float]) -> Tuple[List[float], float]:
        return params[:-1], params[-1]

    def fit(self, dataset: PointProcessDataset, method: str = 'L-BFGS-B', **kwargs):
        """
        Same as PointProcess.fit(), but additionally writes the fitted
        base-process parameter slice onto self.base_process.params_/
        param_dict_ once fitting succeeds -- see class docstring "OWNERSHIP"
        note for why this is safe and why every other method below can
        then delegate to base_process directly.
        """
        super().fit(dataset, method=method, **kwargs)
        base_params, _ = self._split_params(self.params_)
        self.base_process.params_ = np.asarray(base_params, dtype=float)
        self.base_process.param_dict_ = dict(zip(self.base_process.param_names, base_params))
        return self

    def _nll(self, params: List[float], dataset: PointProcessDataset) -> float:
        base_params, r = self._split_params(params)
        r = max(r, 1e-8)  # guard against the optimizer probing r <= 0 during
                           # a line search despite the (1e-3, None) bound

        base_ll, N_f, S_f = self.base_process.mixed_effects_likelihood_terms(
            dataset, base_params
        )
        nb_term = np.sum(
            r * np.log(r) - gammaln(r) + gammaln(N_f + r) - (N_f + r) * np.log(S_f + r)
        )
        return -(base_ll + nb_term)

    def _fish_scale_factors(self, dataset: PointProcessDataset) -> np.ndarray:
        """
        Posterior-mean gain per fish using that fish's ENTIRE observed
        history: (r + N_f) / (r + S_f).

        Used ONLY by estimate_fish_gains() -- a legitimate whole-session
        point estimate. Deliberately NOT used by _stream_tau_values() below:
        using a fish's entire history (including events AFTER any given
        rescaled interval) to correct that same interval violates the
        predictability condition required by Ogata's time-rescaling
        theorem -- see _stream_tau_values docstring.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        base_params, r = self._split_params(self.params_)
        _, N_f, S_f = self.base_process.mixed_effects_likelihood_terms(dataset, base_params)
        return (r + N_f) / (r + S_f)

    def predict(self, t: np.ndarray, trial: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        """
        Population-average rate (E[g_f] = 1); does not reflect any single
        fish's gain. **kwargs forwarded to base_process.predict() (e.g.
        history_events, for a Hawkes/Renewal base process).
        """
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return self.base_process.predict(t, trial, **kwargs)

    def compute_expected_rate(self, dataset: PointProcessDataset) -> np.ndarray:
        """
        Delegates entirely to base_process.compute_expected_rate(), which
        already knows how to loop over dataset.iter_streams() and thread
        per-stream event history where the base process needs it
        (Hawkes/Renewal) -- the wrapper doesn't need to know which base
        process family it's wrapping.
        """
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return self.base_process.compute_expected_rate(dataset)

    def cumulative_integrated_intensity(self, t_events: np.ndarray, trial: float) -> np.ndarray:
        """Population-average cumulative intensity; see class docstring limitation."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
        return self.base_process.cumulative_integrated_intensity(t_events=t_events, trial=trial)

    def estimate_fish_gains(self, dataset: PointProcessDataset) -> pd.DataFrame:
        """Posterior mean gain per fish, from Gamma-Poisson conjugacy. See _fish_scale_factors."""
        if self.params_ is None:
            raise ValueError("Model must be fitted before estimating fish gains.")

        base_params, r = self._split_params(self.params_)
        _, N_f, S_f = self.base_process.mixed_effects_likelihood_terms(dataset, base_params)
        g_hat = self._fish_scale_factors(dataset)
        active = dataset.fish_trial_mask.any(axis=1)

        return pd.DataFrame({
            "fish_idx": np.arange(dataset.num_fish)[active],
            "n_events": N_f[active],
            "expected_events_base": S_f[active],
            "estimated_gain": g_hat[active],
        })

    def _stream_tau_values(self, dataset: PointProcessDataset) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Correct predictable compensator (see PointProcess._stream_tau_values
        base docstring for why this override is required at all).

        For each fish, walks its OWN observed trials in ascending t_idx
        order (== chronological order, assuming trial_num reflects actual
        session order -- guaranteed zero-based/contiguous by
        PointProcessDataset, but ascending-order-== chronological is not
        independently verified here), maintaining a running (N_count,
        S_offset) using only that fish's own history strictly before each
        event -- unlike _fish_scale_factors (whole-session posterior mean),
        this never uses information from AFTER the event being rescaled.

        Implementation note: reuses base_process.cumulative_integrated_
        intensity(t_events, trial) directly, probing it at [this fish's
        real event times in this trial] + [duration_s]. This works for ANY
        base process family because that method's existing contract
        already treats every entry in t_events as potential self/
        refractory history for later entries in the SAME array --
        appending duration_s as the last (largest) probe point correctly
        returns the trial's full exposure integral, using this trial's own
        real events as history, with no extra base-process-specific logic
        needed here.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        _, r = self._split_params(self.params_)  # base_process.params_ already synced by fit()

        result: Dict[Tuple[int, int], np.ndarray] = {}

        for f_idx in range(dataset.num_fish):
            S_offset = 0.0   # cumulative baseline exposure from all fully completed prior trials
            S_prev = 0.0     # absolute cumulative exposure at the last processed reference point
            N_count = 0      # cumulative event count strictly before the current event

            for t_idx in range(dataset.num_trials):
                if not dataset.fish_trial_mask[f_idx, t_idx]:
                    continue  # unobserved trial: no exposure, no events -- matches _nll's convention

                mask = (dataset.event_fish_idx == f_idx) & (dataset.event_trials_idx == t_idx)
                t_ev = np.sort(dataset.event_times[mask])

                probe_times = np.append(t_ev, dataset.duration_s)
                cum_within = self.base_process.cumulative_integrated_intensity(
                    t_events=probe_times, trial=t_idx
                )

                if len(t_ev) > 0:
                    tau_vals = np.empty(len(t_ev))
                    for j in range(len(t_ev)):
                        S_abs = S_offset + cum_within[j]
                        tau_vals[j] = (r + N_count) * np.log((r + S_abs) / (r + S_prev))
                        N_count += 1
                        S_prev = S_abs
                    result[(f_idx, t_idx)] = tau_vals

                # Full-trial exposure always accrues, even for a trial with
                # zero real events (cum_within[-1] == integral at
                # duration_s regardless of len(t_ev)).
                S_offset += cum_within[-1]
                S_prev = S_offset  # reset reference point to this trial's end / next trial's start

        return result

    def diagnose(self, dataset: PointProcessDataset, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        return self.base_process.diagnose(dataset, *args, **kwargs)

    @property
    def dispersion_r(self) -> float:
        """Fitted r (Gamma shape/rate parameter). Larger r = less heterogeneity."""
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        return float(self.params_[-1])