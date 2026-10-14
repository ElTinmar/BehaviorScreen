"""
Classified saccades in the reference UMAP, coloured by assigned class, over
the tethered reference events (Dowell et al. 2024, grey) they were labelled
from. Events rejected as too far from the reference are left out.

    python -m BehaviorScreen.eyes.plot_umap saccades_augmented_all.csv \
        --model BehaviorScreen/eyes/paper_data/paper_reference.joblib \
        --output saccades_umap.png [--line WT] [--condition vehicle]
"""
import argparse
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BehaviorScreen.core import SACCADE_CLASS_NAMES
from BehaviorScreen.eyes.paper_reference import LABEL_COLORS

REFERENCE_COLOR = "#d9d9d9"


def subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]


def load_events(path: Path, line: Optional[str], condition: Optional[str]) -> pd.DataFrame:
    events = pd.read_csv(path, low_memory=False)
    for column, value in (("line", line), ("condition", condition)):
        if value is not None:
            events = events[events[column].astype(str) == value]
    return events[np.isfinite(events["embed_x"]) & np.isfinite(events["embed_y"])]


def plot_umap(events: pd.DataFrame, reference: np.ndarray, max_points: int = 500_000, seed: int = 0) -> plt.Figure:
    rng = np.random.default_rng(seed)
    assigned = events[~events["assignment_rejected"].astype(bool)]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(*subsample(reference, max_points, rng).T, s=1, color=REFERENCE_COLOR, linewidths=0, rasterized=True)
    shown = assigned.iloc[rng.permutation(len(assigned))[:max_points]]  # random draw order: no class hides another
    labels = shown["cluster"].astype(int)
    ax.scatter(shown["embed_x"], shown["embed_y"], s=2, alpha=0.5, linewidths=0, rasterized=True,
               c=[LABEL_COLORS.get(label, "black") for label in labels])
    counts = assigned["cluster"].astype(int).value_counts()
    for label in sorted(counts.index):
        ax.scatter([], [], s=20, color=LABEL_COLORS.get(label, "black"),
                   label=f"{SACCADE_CLASS_NAMES.get(label, label)} ({counts[label]:,})")
    ax.legend(fontsize=7, frameon=False, loc="best")
    ax.set_title(f"Assigned saccade class ({len(assigned):,} events; grey: reference)", fontsize=10)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Classified saccades in the reference UMAP.")
    parser.add_argument("events", type=Path, help="Classified saccade table (embed_x, embed_y, cluster, ...).")
    parser.add_argument("--model", type=Path, required=True, help="paper_reference.joblib")
    parser.add_argument("--output", type=Path, default=Path("saccades_umap.png"))
    parser.add_argument("--line")
    parser.add_argument("--condition")
    parser.add_argument("--max-points", type=int, default=100_000, help="Events (and reference events) drawn: a random subsample.")
    args = parser.parse_args()

    events = load_events(args.events, args.line, args.condition)
    fig = plot_umap(events, joblib.load(args.model)["reference_embedding"], args.max_points)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
