"""
Variance budget of each Tier 1 column, from the vehicle arms and without a
model: how many events a fish contributes, how its fish-to-fish noise splits
into counting noise (falls with trials) and real fish differences (a floor
only more fish lower), how the response habituates over the session, and the
fish this implies, next to the power analysis.

Per fish and trial, the readout matching the column's free amplitude:
    evoked  events in the response window minus the fish's own baseline there (A, A_peak)
    rate    events in a window: the whole trial (B), or the late baseline of an evoked column
    escape  share of trials with an event in the response window (H)

Counting variance is Poisson for counts and binomial for escapes. The CV^2
across fish, pooled over arms before dividing, is fitted against trials per fish:

    CV^2 = a / trials + b

b: real fish differences. a above its Poisson value: events come in clusters
(a hunt is several J-turns). On the log2 scale the per-fish SD is about
CV / ln 2 (delta method, rough when CV is large), comparable to power.py's
sigma: the power analysis' sigma well above it means noise beyond fish
sampling (clutches, lines, fits), which more trials cannot remove.

    python -m BehaviorScreen.ablation_screen.variance_budget /path/to/Screen ./tier1_results
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BehaviorScreen.point_process.analysis.model_config import EVENT_TABLES, get_model_config
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader, PointProcessDataset
from BehaviorScreen.point_process.point_process.io import save_fig

from .dataset_utils import safe_prepare_dataset
from .power import fish_needed, per_fish_sd

Window = Tuple[float, float]
EVENTS_CSV = {"bouts": "bouts_all.csv", "saccades": "saccades_augmented_all.csv"}
MIN_FISH = 10
DRAWS = 50
# Response windows, read off the WT PSTHs: adjust here
EVOKED: Dict[str, Tuple[Window, Window, str]] = {  # (response, baseline, parameter)
    "prey_capture_ipsi": ((0.0, 2.5), (10.0, 24.0), "A"),
    "prey_capture_convergent_saccade": ((0.0, 2.5), (10.0, 24.0), "A"),
    "prey_capture_divergent_saccade": ((0.0, 2.5), (10.0, 24.0), "A"),
    "phototaxis_ipsi": ((0.2, 0.6), (5.0, 24.0), "A_peak"),
}
ESCAPES: Dict[str, Window] = {"looming": (4.0, 7.0), "looming_divergent_saccade": (4.0, 7.0), "dark_flash": (0.0, 1.0)}


@dataclass(frozen=True)
class Readout:
    behavior: str
    kind: str  # "evoked", "rate" or "escape"
    parameter: str
    window: Window
    baseline: Optional[Window] = None


def readouts(config: Dict[str, dict]) -> List[Readout]:
    out = []
    for behavior, cfg in config.items():
        if behavior in EVOKED:
            window, baseline, parameter = EVOKED[behavior]
            out += [Readout(behavior, "evoked", parameter, window, baseline), Readout(behavior, "rate", "B", baseline)]
        elif behavior in ESCAPES:
            out.append(Readout(behavior, "escape", "H", ESCAPES[behavior]))
        else:
            out.append(Readout(behavior, "rate", "B", (cfg["dataset"]["t_start"], cfg["dataset"]["t_end"])))
    return out


def per_fish(ds: PointProcessDataset, r: Readout, trials: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(value, counting variance) per fish with trials among `trials`, per trial."""
    n = ds.fish_trial_mask[:, trials].sum(axis=1).astype(float)
    keep = np.isin(ds.event_trials_idx, trials)
    t, f, tr = ds.event_times[keep], ds.event_fish_idx[keep], ds.event_trials_idx[keep]
    inside = lambda w: (t >= w[0]) & (t < w[1])
    seen = n > 0
    n = np.maximum(n, 1)
    if r.kind == "escape":
        hits = np.unique(np.column_stack([f, tr])[inside(r.window)], axis=0)
        q = np.bincount(hits[:, 0], minlength=ds.num_fish) / n
        return q[seen], (q * (1 - q) / n)[seen]
    count = lambda w: np.bincount(f[inside(w)], minlength=ds.num_fish).astype(float)
    c, s = count(r.window), 0.0
    c_base = count(r.baseline) if r.baseline else np.zeros_like(c)
    if r.baseline:
        s = (r.window[1] - r.window[0]) / (r.baseline[1] - r.baseline[0])
    return ((c - s * c_base) / n)[seen], ((c + s**2 * c_base) / n**2)[seen]


def pooled(datasets: List[PointProcessDataset], r: Readout, trials) -> np.ndarray:
    """(mean, counting variance, total variance) across fish, averaged over arms weighted by fish."""
    parts = []
    for ds in datasets:
        value, var = per_fish(ds, r, trials(ds))
        parts.append([value.mean(), var.mean(), value.var(ddof=1)])
    return np.average(parts, axis=0, weights=[ds.num_fish for ds in datasets])


def fish_x2(cv2: np.ndarray) -> np.ndarray:
    """Fish per arm for a x2 change on the log scale, one planned test (80% power, alpha 0.05)."""
    return fish_needed(np.sqrt(cv2) / np.log(2), 1.0, 0.05)


def budget(datasets: List[PointProcessDataset], r: Readout, rng: np.random.Generator) -> Optional[Tuple[dict, dict]]:
    """(summary row, curves for the plots) of one readout; None without events."""
    k_now = min(ds.num_trials for ds in datasets)
    mean, poisson, total = pooled(datasets, r, lambda ds: np.arange(ds.num_trials))
    if mean == 0:
        return None
    ks = np.unique(np.linspace(2, k_now, 6).astype(int))
    draws = np.array([[pooled(datasets, r, lambda ds: rng.choice(ds.num_trials, k, replace=False)) for _ in range(DRAWS)] for k in ks]).mean(axis=1)
    cv2 = draws[:, 2] / mean**2
    a, b = np.polyfit(1 / ks, cv2, 1)
    b = max(b, 0.0)
    by_trial = np.array([pooled(datasets, r, lambda ds: np.array([k])) for k in range(k_now)])
    first = np.array([pooled(datasets, r, lambda ds: np.arange(k)) for k in range(2, k_now + 1)])
    first_fish = fish_x2(first[:, 2] / first[:, 0] ** 2)
    best = int(np.argmin(first_fish))
    row = {
        "behavior": r.behavior, "readout": r.kind, "parameter": r.parameter, "arms": len(datasets),
        "trials now": k_now, "per fish per trial": mean,
        "counting slope": a, "Poisson slope": poisson * k_now / mean**2, "fish floor": b,
        "sigma within (log2)": np.sqrt(total) / abs(mean) / np.log(2),
        "fish x2 now": fish_x2(total / mean**2), "fish x2, 2x trials": fish_x2(a / (2 * k_now) + b),
        "fish x2, unlimited trials": fish_x2(b),
        "best first k": best + 2, "fish x2, best first k": first_fish[best],
    }
    return row, {"ks": ks, "cv2": cv2, "a": a, "b": b, "by_trial": by_trial, "n_fish": sum(ds.num_fish for ds in datasets)}


def plot_curves(curves: Dict[Readout, dict]) -> plt.Figure:
    """CV^2 against trials per fish, with the fit and its fish floor."""
    fig, axes = _grid(len(curves))
    for ax, (r, c) in zip(axes, curves.items()):
        k_fine = np.linspace(c["ks"].min(), 4 * c["ks"].max(), 100)
        ax.plot(c["ks"], c["cv2"], "o", color="C3", markersize=3)
        ax.plot(k_fine, c["a"] / k_fine + c["b"], color="C3", alpha=0.6, linewidth=1)
        if c["b"] > 0:
            ax.axhline(c["b"], color="C3", linestyle=":", alpha=0.6, linewidth=1)
        ax.axvline(c["ks"].max(), color="0.7", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{r.behavior} · {r.kind}", fontsize=7)
    fig.supxlabel("trials per fish (vertical line: protocol now; dotted: fish floor)", fontsize=8)
    fig.supylabel("per-fish CV²", fontsize=8)
    fig.tight_layout()
    return fig


def plot_habituation(curves: Dict[Readout, dict]) -> plt.Figure:
    """The readout per fish, trial by trial, pooled over vehicle arms (± SE)."""
    fig, axes = _grid(len(curves))
    for ax, (r, c) in zip(axes, curves.items()):
        mean, _, var = c["by_trial"].T
        ax.errorbar(np.arange(1, len(mean) + 1), mean, yerr=np.sqrt(var / c["n_fish"]), fmt="o-", color="C3", markersize=3, linewidth=1)
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.set_title(f"{r.behavior} · {r.kind}", fontsize=7)
    fig.supxlabel("trial", fontsize=8)
    fig.supylabel("per fish per trial (pooled vehicle)", fontsize=8)
    fig.tight_layout()
    return fig


def _grid(n: int, ncols: int = 5):
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.3 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat[:n]


def main(root: Path, results_dir: Path) -> None:
    config = get_model_config()
    loaders = {e: BehavioralDataLoader(root / csv, [root / "valid_trials_all.csv"], [root / "qc_all.csv"], **EVENT_TABLES[e])
               for e, csv in EVENTS_CSV.items()}
    rng = np.random.default_rng(0)
    rows, curves, arms = [], {}, {}
    for r in readouts(config):
        if r.behavior not in arms:
            loader = loaders[config[r.behavior]["events"]]
            datasets = (safe_prepare_dataset(loader.subset(line, ["vehicle"]), config[r.behavior]["dataset"], f"{line}/{r.behavior}")
                        for line in sorted(loader.trials["line"].astype(str).unique()))
            arms[r.behavior] = [ds for ds in datasets if ds is not None and ds.num_fish >= MIN_FISH]
        result = budget(arms[r.behavior], r, rng) if arms[r.behavior] else None
        if result:
            rows.append(result[0])
            curves[r] = result[1]
    table = pd.DataFrame(rows)
    if (results_dir / "tier1_results.csv").exists():
        sigma = per_fish_sd(pd.read_csv(results_dir / "tier1_parameters.csv"), pd.read_csv(results_dir / "tier1_results.csv"))
        table = table.merge(sigma[["behavior", "parameter", "sigma"]].rename(columns={"sigma": "sigma power (log2)"}),
                            on=["behavior", "parameter"], how="left")
    out = results_dir / "variance_budget"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "variance_budget.csv", index=False)
    save_fig(plot_curves(curves), out, "cv2_vs_trials")
    save_fig(plot_habituation(curves), out, "habituation")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0]), Path(args[1] if len(args) > 1 else "./tier1_results"))
