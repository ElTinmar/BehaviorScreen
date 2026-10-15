"""
Side calibration of a sided model: given an event at (t, trial), the model
gives the probability it is ipsi, lambda_ipsi / (lambda_ipsi + lambda_contra)
(the fish gain cancels; for escapes, the constant share set by D). The
observed ipsi fraction is compared with it by time, by trial and by fish;
the fish panel tests whether one D fits every fish.
"""
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .dataset import PointProcessDataset
from .point_process import PointProcess
from .sided_process import IPSI, SidedSurvivalProcess, sided_base


def side_events(dataset: PointProcessDataset, model: PointProcess) -> Dict[str, np.ndarray]:
    """Events the model gives a side to (first events for escapes), their side and its model probability."""
    base = sided_base(model)
    events = dataset.first_events() if isinstance(base, SidedSurvivalProcess) else dataset
    return {
        "time": events.event_times,
        "trial": events.event_trials_idx,
        "fish": events.event_fish_idx,
        "is_ipsi": (np.asarray(events.marks["laterality"]) == IPSI).astype(float),
        "p_ipsi": base.ipsi_probability(events.event_times, events.event_trials_idx),
    }


def _binned(groups: np.ndarray, n_groups: int, ev: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
    """Per group: events, observed ipsi count, expected count and its variance under the model."""
    count = lambda w=None: np.bincount(groups, w, minlength=n_groups)
    p = ev["p_ipsi"]
    return count(), count(ev["is_ipsi"]), count(p), count(p * (1 - p))


def _plot_calibration(ax: plt.Axes, x: np.ndarray, ev: Dict[str, np.ndarray], groups: np.ndarray, xlabel: str) -> None:
    n, observed, expected, var = _binned(groups, len(x), ev)
    shown = n > 0
    n, observed, expected, sd = n[shown], observed[shown], expected[shown], np.sqrt(var[shown])
    x = x[shown]
    ax.fill_between(x, (expected - 1.96 * sd) / n, (expected + 1.96 * sd) / n, color="crimson", alpha=0.15, lw=0)
    ax.plot(x, expected / n, color="crimson", ls="--", lw=1.3, label="model (95% band)")
    ax.plot(x, observed / n, "o", color="black", ms=3.5, label="observed")
    ax.set(xlabel=xlabel, ylabel="ipsi fraction", ylim=(0, 1))
    ax.legend(fontsize=8, frameon=False)


def plot_side_diagnostics(dataset: PointProcessDataset, model: PointProcess, n_time_bins: int = 30) -> plt.Figure:
    ev = side_events(dataset, model)
    fig, (ax_t, ax_m, ax_f) = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

    edges = np.linspace(0, dataset.duration_s, n_time_bins + 1)
    _plot_calibration(ax_t, (edges[:-1] + edges[1:]) / 2, ev,
                      np.clip(np.digitize(ev["time"], edges) - 1, 0, n_time_bins - 1), "Time in trial (s)")
    ax_t.set_title("Side calibration by time")
    _plot_calibration(ax_m, np.arange(dataset.num_trials), ev, ev["trial"], "Trial")
    ax_m.set_title("Side calibration by trial")

    n, observed, expected, var = _binned(ev["fish"], dataset.num_fish, ev)
    shown = var > 0
    z = (observed[shown] - expected[shown]) / np.sqrt(var[shown])
    chi2 = float(np.sum(z**2))
    df = max(int(shown.sum()) - 1, 1)
    ax_f.semilogx(n[shown], z, "o", color="black", ms=3.5, alpha=0.6)
    for level in (-1.96, 0, 1.96):
        ax_f.axhline(level, color="crimson", ls="--" if level else "-", lw=1)
    ax_f.set(xlabel="events per fish", ylabel="(observed − expected ipsi) / SD")
    ax_f.set_title(f"Per-fish side bias: dispersion {chi2 / df:.2f}, p = {stats.chi2.sf(chi2, df):.3g}")
    fig.suptitle(f"{model.name}: is each event's side as likely as the model says?", fontsize=11)
    return fig
