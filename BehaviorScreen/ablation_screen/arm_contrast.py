from typing import Tuple

import numpy as np

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.point_process import PointProcess


def output_metric(fitted_process: PointProcess, dataset: PointProcessDataset) -> Tuple[float, str]:
    """Single-number behavioral summary for an ALREADY-FITTED process,
    evaluated on the dataset it was fit on."""
    if fitted_process.is_survival:
        _, surv = fitted_process.population_survival_curve(dataset)
        return float(1.0 - surv[-1]), "response_probability"

    expected_rate = fitted_process.compute_expected_rate(dataset)  # (n_trials, n_bins)
    per_trial_total = np.trapz(expected_rate, dataset.t_centers, axis=1)
    weights = dataset.n_fish_per_trial
    mean_total = (
        float(np.average(per_trial_total, weights=weights))
        if weights.sum() > 0 else float(np.mean(per_trial_total))
    )
    return mean_total, "expected_events_per_fish_per_trial"