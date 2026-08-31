"""Calibration diagnostics -- run on Tier 1's real output AND on the
negative-control (WT split-half) output before trusting any p-value."""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_pvalue_histogram(
    pvals: np.ndarray, bins: int = 20, title: str = "P-value distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[~np.isnan(pvals)]
    n = len(pvals)

    ax.hist(pvals, bins=bins, range=(0, 1), density=True,
            color="steelblue", edgecolor="white", alpha=0.85)
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Uniform(0,1) null")

    bin_width = 1.0 / bins
    expected_per_bin = n * bin_width
    se = np.sqrt(expected_per_bin * (1 - bin_width)) if expected_per_bin > 0 else 0
    band = 1.96 * se / (n * bin_width) if n > 0 else 0
    ax.axhspan(1 - band, 1 + band, color="crimson", alpha=0.1)

    ax.set_xlim(0, 1)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title(f"{title} (N={n})")
    ax.legend(loc="upper right", fontsize=9)
    return ax