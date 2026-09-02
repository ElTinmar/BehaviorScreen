from typing import Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from .dataset import PointProcessDataset
from .point_process import PointProcess


def collect_fish_gains(
    models_and_datasets: Dict[str, Tuple[PointProcess, PointProcessDataset]],
) -> pd.DataFrame:
    """
    models_and_datasets: {behavior_name: (fitted_model, dataset)}. Only
    models exposing estimate_fish_gains (GammaMixedEffectsProcess, or any
    wrapper delegating to one) contribute rows; others are skipped
    silently.

    Returns long format: one row per (fish_id, behavior, estimated_gain).
    fish_id comes from dataset.fish_ids -- the real, dataset-independent
    identifier -- NOT dataset-local fish_idx, which only has meaning
    within a single dataset.
    """
    records = []
    for behavior, (model, dataset) in models_and_datasets.items():
        if not hasattr(model, "estimate_fish_gains"):
            continue
        if dataset.fish_ids is None:
            raise ValueError(
                f"[{behavior}] dataset.fish_ids is None -- this dataset was built "
                f"without fish identity tracking. collect_fish_gains needs it to "
                f"join fish correctly across behaviors."
            )
        try:
            gains = model.estimate_fish_gains(dataset)
        except Exception:
            continue

        gains = gains.copy()
        gains["fish_id"] = dataset.fish_ids[gains["fish_idx"].values]
        gains["behavior"] = behavior
        records.append(gains[["fish_id", "behavior", "estimated_gain"]])

    if not records:
        return pd.DataFrame(columns=["fish_id", "behavior", "estimated_gain"])
    return pd.concat(records, ignore_index=True)


def fish_gain_correlation(
    gain_long_df: pd.DataFrame, min_fish_shared: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pivots to fish x behavior, returns (correlation matrix, pairwise
    shared-fish-count matrix). Kept separate from plotting so both are
    independently usable (e.g. compare_to_pooled_baseline).

    n_pairs MUST be computed from an int-cast (not bool) notna() matrix --
    DataFrame.dot() between two bool frames performs boolean matrix
    multiplication (any-of-AND), silently returning True/False instead of
    the intended pairwise count.
    """
    wide = gain_long_df.pivot_table(index="fish_id", columns="behavior", values="estimated_gain")
    corr = wide.corr(min_periods=min_fish_shared)
    notna_int = wide.notna().astype(int)
    n_pairs = notna_int.T.dot(notna_int)
    return corr, n_pairs


def plot_fish_gain_correlation(
    gain_long_df: pd.DataFrame,
    min_fish_shared: int = 5,
    cmap: str = "coolwarm",
    figsize: Tuple[float, float] = (7, 6),
    ax: Optional[plt.Axes] = None,
    title: str = "Fish-level frailty gain correlation across behaviors",
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Draws into a caller-supplied `ax` if given, else creates its own figure."""
    corr, n_pairs = fish_gain_correlation(gain_long_df, min_fish_shared=min_fish_shared)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)

    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            n = n_pairs.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}\n(n={n})", ha="center", va="center",
                        color="white" if abs(val) > 0.5 else "black", fontsize=6)

    ax.set_title(title, fontsize=10, fontweight="bold")
    return fig, ax, corr


def plot_fish_gain_correlation_dual(
    gain_long_df_a: pd.DataFrame,
    gain_long_df_b: pd.DataFrame,
    label_a: str = "Group A",
    label_b: str = "Group B",
    suptitle: str = "",
    min_fish_shared: int = 5,
    min_behaviors: int = 2,
) -> Optional[plt.Figure]:
    """
    Side-by-side fish-gain correlation heatmaps for two groups (e.g.
    vehicle vs drug, or any two conditions), sharing a colorbar. Generic
    over what "group A"/"group B" mean.
    """
    if gain_long_df_a["behavior"].nunique() < min_behaviors or gain_long_df_b["behavior"].nunique() < min_behaviors:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_fish_gain_correlation(gain_long_df_a, ax=axes[0], title=label_a, min_fish_shared=min_fish_shared)
    plot_fish_gain_correlation(gain_long_df_b, ax=axes[1], title=label_b, min_fish_shared=min_fish_shared)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.colorbar(axes[1].images[0], ax=axes, shrink=0.7, label="Correlation")
    return fig