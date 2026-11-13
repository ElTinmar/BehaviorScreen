from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import PointProcessDataset
from .point_process import PointProcess


def collect_fish_gains(
    models_and_datasets: Dict[str, Tuple[PointProcess, PointProcessDataset]],
) -> pd.DataFrame:
    """
    models_and_datasets: {behavior: (fitted model, dataset)}; models without fish gains are skipped.
    Long format: fish_id (dataset.fish_ids, the same fish across behaviors), behavior, estimated_gain
    (posterior mean) and gain_variance (the gain distribution's variance, 1 / r).
    """
    records = []
    for behavior, (model, dataset) in models_and_datasets.items():
        try:
            gains = model.estimate_fish_gains(dataset)
        except NotImplementedError:
            print(f"[{behavior}] {model.name} has no fish gains; skipped")
            continue
        records.append(pd.DataFrame({"fish_id": dataset.fish_ids[gains["fish_idx"].values], "behavior": behavior,
                                     "estimated_gain": gains["estimated_gain"].values, "gain_variance": 1.0 / model.dispersion_r}))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["fish_id", "behavior", "estimated_gain", "gain_variance"])


def gain_correlations(gains: pd.DataFrame, n_boot: int = 500, seed: int = 0, min_fish_shared: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Correlation across fish of their posterior gains in each pair of behaviors, and the same corrected for
    shrinkage: a posterior mean keeps a share of the gain's variance (its reliability, Var(E[g | data]) / Var(g),
    lower with fewer events per fish), and a correlation of posterior means is the gains' correlation times
    sqrt(reliability_a reliability_b). 95% intervals of the corrected values from resampling fish.
    """
    wide = gains.pivot_table(index="fish_id", columns="behavior", values="estimated_gain")[gains["behavior"].unique()]
    prior = gains.groupby("behavior")["gain_variance"].first()[wide.columns]

    def correct(w: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        observed, reliability = w.corr(min_periods=min_fish_shared), (w.var() / prior).clip(upper=1.0)
        return observed, reliability, observed / np.sqrt(np.outer(reliability, reliability))

    observed, reliability, corrected = correct(wide)
    rng = np.random.default_rng(seed)
    boot = np.stack([correct(wide.iloc[rng.integers(0, len(wide), len(wide))])[2].to_numpy() for _ in range(n_boot)])
    present = wide.notna().astype(int)
    frame = lambda values: pd.DataFrame(values, index=wide.columns, columns=wide.columns)
    return {
        "observed": observed, "corrected": corrected,
        "corrected_low": frame(np.nanpercentile(boot, 2.5, axis=0)), "corrected_high": frame(np.nanpercentile(boot, 97.5, axis=0)),
        "reliability": reliability.rename("reliability").to_frame(), "n_fish": present.T.dot(present),
    }


def plot_correlation_matrix(corr: pd.DataFrame, ax: plt.Axes, title: str, low: Optional[pd.DataFrame] = None,
                            high: Optional[pd.DataFrame] = None, labels: Optional[list] = None, row_labels: bool = True) -> None:
    """Heatmap of a behavior x behavior correlation matrix, each value written in its cell (with its interval if given)."""
    ax.imshow(corr.clip(-1, 1), cmap="coolwarm", vmin=-1, vmax=1)
    ticks = labels or list(corr.columns)
    ax.set_xticks(range(len(ticks)), ticks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ticks)), ticks if row_labels else [], fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            if pd.isna(value) or i == j:
                continue
            text = f"{value:.2f}" + ("" if low is None else f"\n[{low.iloc[i, j]:.2f}, {high.iloc[i, j]:.2f}]")
            ax.text(j, i, text, ha="center", va="center", fontsize=5, color="white" if abs(value) > 0.5 else "black")
    ax.set_title(title, fontsize=10, fontweight="bold")


def plot_gain_correlations(results: Dict[str, pd.DataFrame], suptitle: str = "") -> plt.Figure:
    """gain_correlations as two heatmaps: as observed, and corrected for shrinkage (reliability beside each behavior)."""
    labels = [f"{b} ({r:.2f})" for b, r in results["reliability"]["reliability"].items()]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), layout="constrained")
    plot_correlation_matrix(results["observed"], axes[0], "Posterior gains, as observed", labels=labels)
    plot_correlation_matrix(results["corrected"], axes[1], "Corrected for shrinkage (95% interval)",
                            results["corrected_low"], results["corrected_high"], labels=labels, row_labels=False)
    fig.colorbar(axes[1].images[0], ax=axes, shrink=0.6, label="Correlation across fish (behavior: reliability)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    return fig
