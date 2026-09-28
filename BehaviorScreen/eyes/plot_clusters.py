"""
plot_clusters.py
================
Sanity-check plot: raw + smoothed eye-position traces around saccade onset,
grouped by cluster label. Reads the CSV + companion .npz produced by
saccade_cli.py.

Usage
-----
    python plot_clusters.py saccades.csv --trace-type raw --max-per-cluster 40
    python plot_clusters.py saccades.csv --trace-type smooth --output clusters.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_clusters_raw_traces(csv_path: Path, npz_path: Path, trace_type: str = "raw",
                              max_per_cluster: int = 50, output_png: Path | None = None,
                              seed: int = 0) -> None:
    events = pd.read_csv(csv_path)
    data = np.load(npz_path)

    L = data[f"L_{trace_type}"]
    R = data[f"R_{trace_type}"]
    time_axis = data["time_axis_ms"]

    if len(events) != len(L):
        raise ValueError(
            f"CSV ({len(events)} rows) and npz ({len(L)} snippets) are out of sync "
            "-- did you re-run detection without regenerating both files?"
        )

    clusters = sorted(events["cluster"].unique())
    ncols = 4
    nrows = int(np.ceil(len(clusters) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(seed)

    for ax, cl in zip(axes, clusters):
        idx_all = events.index[events["cluster"] == cl].to_numpy()
        idx_plot = (rng.choice(idx_all, size=max_per_cluster, replace=False)
                    if len(idx_all) > max_per_cluster else idx_all)

        for i in idx_plot:
            ax.plot(time_axis, L[i], color="b", alpha=0.15, lw=0.7)
            ax.plot(time_axis, R[i], color="r", alpha=0.15, lw=0.7)

        # median trace computed over the FULL cluster population, not just the subsample
        ax.plot(time_axis, np.nanmedian(L[idx_all], axis=0), color="navy", lw=2, label="L (median)")
        ax.plot(time_axis, np.nanmedian(R[idx_all], axis=0), color="darkred", lw=2, label="R (median)")

        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_title(f"cluster {cl}  (n={len(idx_all)})")

    for ax in axes[len(clusters):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)

    fig.supxlabel("Time relative to onset (ms)")
    fig.supylabel("Eye angle (deg)")
    fig.suptitle(f"{trace_type} traces by cluster")
    fig.tight_layout()

    if output_png is not None:
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_png}")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot raw/smoothed traces grouped by cluster.")
    p.add_argument("csv", type=Path, help="saccades.csv produced by saccade_cli.py")
    p.add_argument("--npz", type=Path, default=None,
                    help="companion .npz (defaults to csv with .npz extension)")
    p.add_argument("--trace-type", choices=["raw", "smooth"], default="raw")
    p.add_argument("--max-per-cluster", type=int, default=50)
    p.add_argument("--output", type=Path, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    npz_path = args.npz or args.csv.with_suffix(".npz")
    plot_clusters_raw_traces(args.csv, npz_path, args.trace_type, args.max_per_cluster, args.output)