"""
plot_amplitude_heatmap.py
==========================
2D histogram of left vs. right eye saccade amplitude, both pooled and
split by cluster -- ported from BF1_saccadehistograms.m's core
"histogramplot" logic. This is the most direct sanity check that
clusters correspond to sensible amplitude-space quadrants (conjugate
L/R, convergent, divergent, biphasic) independent of the UMAP topology.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_amplitude_heatmap(
    csv_path: Path,
    output_png: Path | None = None,
    bin_edges: np.ndarray | None = None,
    by_cluster: bool = True,
) -> None:
    events = pd.read_csv(csv_path)

    if bin_edges is None:
        bin_edges = np.arange(-40, 41, 1.5)

    x, y = events["Amp_L"].to_numpy(), events["Amp_R"].to_numpy()

    if not by_cluster:
        fig, ax = plt.subplots(figsize=(6, 6))
        h = ax.hist2d(x, y, bins=[bin_edges, bin_edges], cmap="hot")
        fig.colorbar(h[3], ax=ax, label="probability")
        ax.plot(bin_edges[[0, -1]], bin_edges[[0, -1]], "w--", lw=1)   # y=x (conjugate line)
        ax.plot(bin_edges[[0, -1]], bin_edges[[-1, 0]], "w--", lw=1)   # y=-x (conv/div line)
        ax.set_xlabel("ΔL eye (deg)"); ax.set_ylabel("ΔR eye (deg)")
        ax.set_aspect("equal")
    else:
        clusters = sorted(c for c in events["cluster"].unique() if c != -1)
        ncols = 4
        nrows = int(np.ceil(len(clusters) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows),
                                  sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for ax, cl in zip(axes, clusters):
            mask = events["cluster"] == cl
            ax.hist2d(x[mask], y[mask], bins=[bin_edges, bin_edges], cmap="hot")
            ax.plot(bin_edges[[0, -1]], bin_edges[[0, -1]], "w--", lw=0.7)
            ax.plot(bin_edges[[0, -1]], bin_edges[[-1, 0]], "w--", lw=0.7)
            ax.set_title(f"cluster {cl} (n={mask.sum()})", fontsize=9)
            ax.set_aspect("equal")

        for ax in axes[len(clusters):]:
            ax.set_visible(False)
        fig.supxlabel("ΔL eye (deg)")
        fig.supylabel("ΔR eye (deg)")

    fig.tight_layout()
    if output_png is not None:
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--pooled", action="store_true", help="single pooled heatmap instead of per-cluster grid")
    args = p.parse_args()
    plot_amplitude_heatmap(args.csv, args.output, by_cluster=not args.pooled)