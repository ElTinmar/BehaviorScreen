"""
plot_umap.py
============
Sanity-check plot: UMAP embedding colored by cluster label. Reads the CSV
produced by saccade_cli.py (must contain embed_x, embed_y, cluster columns).

Usage
-----
    python plot_umap.py saccades.csv
    python plot_umap.py saccades.csv --color-by Vergence --output umap_vergence.png
    python plot_umap.py saccades.csv --max-points 50000  # subsample for speed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_umap_clusters(
    csv_path: Path,
    output_png: Path | None = None,
    color_by: str | None = None,
    max_points: int | None = None,
    point_size: float = 3.0,
    alpha: float = 0.4,
    seed: int = 0,
) -> None:
    events = pd.read_csv(csv_path)

    if "embed_x" not in events.columns or "embed_y" not in events.columns:
        raise ValueError("CSV has no embed_x/embed_y columns -- run clustering first.")

    if max_points is not None and len(events) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(events), size=max_points, replace=False)
        events = events.iloc[idx]
        print(f"Subsampled to {max_points} points for plotting")

    fig, ax = plt.subplots(figsize=(9, 8))

    if color_by is not None:
        # continuous colormap over a chosen metric (e.g. Vergence, Amp_L)
        if color_by not in events.columns:
            raise ValueError(f"'{color_by}' not found in CSV columns")
        vals = events[color_by].to_numpy(dtype=float)
        vmin, vmax = np.nanpercentile(vals, [1, 99])  # robust to outliers
        sc = ax.scatter(
            events["embed_x"], events["embed_y"], c=vals,
            cmap="coolwarm", vmin=vmin, vmax=vmax,
            s=point_size, alpha=alpha, linewidths=0, rasterized=True,
        )
        fig.colorbar(sc, ax=ax, label=color_by)
        ax.set_title(f"UMAP embedding colored by {color_by}")

    else:
        # discrete colors per cluster, noise (-1) in gray
        clusters = sorted(events["cluster"].unique())
        n_real = len([c for c in clusters if c != -1])
        cmap = cm.get_cmap("tab20", max(n_real, 1))

        for i, cl in enumerate([c for c in clusters if c != -1]):
            mask = events["cluster"] == cl
            ax.scatter(
                events.loc[mask, "embed_x"], events.loc[mask, "embed_y"],
                s=point_size, alpha=alpha, linewidths=0, rasterized=True,
                color=cmap(i), label=f"cluster {cl} (n={mask.sum()})",
            )
            # label centroid with cluster id
            cx, cy = events.loc[mask, ["embed_x", "embed_y"]].median()
            ax.annotate(str(cl), (cx, cy), fontsize=12, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle", fc="white", ec="black", alpha=0.8))

        noise_mask = events["cluster"] == -1
        if noise_mask.any():
            ax.scatter(
                events.loc[noise_mask, "embed_x"], events.loc[noise_mask, "embed_y"],
                s=point_size * 0.7, alpha=alpha * 0.5, linewidths=0, rasterized=True,
                color="lightgray", label=f"unclustered (n={noise_mask.sum()})",
            )

        ax.legend(fontsize=8, frameon=False, markerscale=2, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0))
        ax.set_title("UMAP embedding by cluster")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if output_png is not None:
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_png}")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot UMAP embedding colored by cluster (or a metric).")
    p.add_argument("csv", type=Path, help="saccades.csv produced by saccade_cli.py")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--color-by", type=str, default=None,
                    help="Color by a continuous metric column instead of cluster "
                         "(e.g. Vergence, Amp_L, Vel_cw_L) -- useful for checking "
                         "whether clusters actually separate along a metric gradient")
    p.add_argument("--max-points", type=int, default=None,
                    help="Randomly subsample for faster/less cluttered plotting")
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.4)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_umap_clusters(
        args.csv, args.output, args.color_by,
        args.max_points, args.point_size, args.alpha,
    )