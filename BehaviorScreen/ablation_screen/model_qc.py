from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from BehaviorScreen.point_process.point_process import PointProcess, ModelComparator
from BehaviorScreen.point_process.dataset import PointProcessDataset


@dataclass
class GroupFitQC:
    line: str
    behavior: str
    group_name: str
    n_fish: int
    mean_events_per_stream: float
    is_low_power: bool
    converged: bool = False
    aic_architecture: float = np.nan
    aic_null: float = np.nan
    delta_aic: float = np.nan
    beats_null: bool = False
    params_near_bound: List[str] = field(default_factory=list)
    hessian_positive_definite: bool = True
    max_abs_param_correlation: float = np.nan

    # Hard failures: fit is not trustworthy -- these exclude the cell from
    # the parametric comparison and trigger the Tier-0 fallback.
    reasons_flagged: List[str] = field(default_factory=list)

    # Soft/diagnostic notes: worth surfacing in the report, but NOT grounds
    # to distrust the fit or exclude the cell. "architecture doesn't beat
    # null" belongs here -- a well-converged, well-conditioned fit whose
    # complex kernel collapses to null-like behavior is a legitimate result
    # (e.g. ablation flattened the response), not a broken fit.
    informational_notes: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """
        - "flagged": hard failure, do not trust this fit for any comparison
          (too few fish, fit exception, non-PD Hessian, near-collinear
          params). Triggers Tier-0 fallback.
        - "reliable_no_signal": fit converged cleanly and is well-identified,
          but the architecture didn't beat the null in THIS arm. Still fine
          to feed into fit_tier1's deviance test -- in fact this is exactly
          the pattern a real ablation-abolishes-the-response effect would
          produce in the drug arm.
        - "reliable": converged, well-identified, and beats the null.
        """
        if self.reasons_flagged:
            return "flagged"
        if self.converged and not self.beats_null:
            return "reliable_no_signal"
        return "reliable"


def _near_bound(value: float, bound: Tuple[Optional[float], Optional[float]],
                 rel_tol: float = 0.02) -> bool:
    lo, hi = bound
    if lo is not None and hi is not None:
        rng = hi - lo
        return (value - lo) <= rel_tol * rng or (hi - value) <= rel_tol * rng
    if lo is not None:
        margin = max(1e-3, 0.02 * abs(value))
        return (value - lo) <= margin
    if hi is not None:
        margin = max(1e-3, 0.02 * abs(value))
        return (hi - value) <= margin
    return False


def assess_group_fit(
    process_factory: Callable[[], PointProcess],
    null_process_factory: Callable[[], PointProcess],
    dataset: PointProcessDataset,
    line: str,
    behavior: str,
    group_name: str,
    delta_aic_threshold: float = 2.0,
    corr_threshold: float = 0.95,
    min_fish: int = 6,
) -> GroupFitQC:

    n_fish = dataset.num_fish
    mean_ev = float(np.mean(dataset.stream_event_counts)) if len(dataset.stream_event_counts) else np.nan
    qc = GroupFitQC(
        line=line, behavior=behavior, group_name=group_name,
        n_fish=n_fish, mean_events_per_stream=mean_ev,
        is_low_power=dataset.is_low_power_for_dispersion,
    )

    if n_fish < min_fish:
        qc.reasons_flagged.append(f"too_few_fish (n={n_fish})")
        return qc

    model_arch = process_factory()
    model_null = null_process_factory()

    try:
        model_arch.fit(dataset)
        model_null.fit(dataset)
    except Exception as e:
        qc.reasons_flagged.append(f"fit_failed: {type(e).__name__}: {e}")
        return qc

    qc.converged = True
    qc.aic_architecture = model_arch.aic
    qc.aic_null = model_null.aic
    qc.delta_aic = qc.aic_null - qc.aic_architecture
    qc.beats_null = qc.delta_aic > delta_aic_threshold

    if not qc.beats_null:
        # Informational only -- see verdict docstring. Do NOT add to
        # reasons_flagged: this must not exclude the cell or trigger fallback.
        qc.informational_notes.append(
            f"architecture_does_not_beat_null (ΔAIC={qc.delta_aic:.2f}) -- "
            f"kernel collapsed to null-like behavior in this arm; may itself "
            f"be the finding (e.g. ablation abolished the response) rather "
            f"than a fit problem."
        )

    kernel = getattr(model_arch, "kernel", None)
    if kernel is not None:
        n_kernel_params = len(kernel.param_names)
        for name, val, bound in zip(kernel.param_names, model_arch.params_[:n_kernel_params], kernel.bounds):
            if _near_bound(val, bound):
                qc.params_near_bound.append(name)
    if qc.params_near_bound:
        # Boundary-pinning invalidates Wald-style SEs on that parameter
        # (relevant for Tier 2 parameter inference) but does not by itself
        # mean the log-likelihood/deviance used by Tier 1 is wrong -- still
        # treated as a hard flag here since MultiGroupProcess's per-parameter
        # contrasts downstream would be unreliable for this arm.
        qc.reasons_flagged.append(f"params_near_bound: {qc.params_near_bound}")

    try:
        hessian = model_arch.estimate_hessian(dataset)
        eigvals = np.linalg.eigvalsh(hessian)
        qc.hessian_positive_definite = bool(np.all(eigvals > 1e-8))
        if not qc.hessian_positive_definite:
            qc.reasons_flagged.append("hessian_not_positive_definite (flat/ridge likelihood)")

        corr = model_arch.estimate_parameter_correlation(dataset)
        off_diag = corr - np.diag(np.diag(corr))
        qc.max_abs_param_correlation = float(np.max(np.abs(off_diag))) if off_diag.size else np.nan
        if qc.max_abs_param_correlation > corr_threshold:
            i, j = np.unravel_index(np.argmax(np.abs(off_diag)), off_diag.shape)
            qc.reasons_flagged.append(
                f"near_collinear_parameters: {model_arch.param_names[i]}~{model_arch.param_names[j]} "
                f"(r={off_diag[i,j]:.2f})"
            )
    except Exception as e:
        qc.reasons_flagged.append(f"hessian_estimation_failed: {type(e).__name__}: {e}")

    if qc.is_low_power:
        # Also informational-only: low power is a caveat on interpretation
        # (wide CIs, less sensitive test) but not evidence the fit itself is
        # broken -- it shouldn't trigger the Tier-0 fallback on its own.
        qc.informational_notes.append("dataset_is_low_power_for_dispersion (mean count/stream < 1)")

    return qc