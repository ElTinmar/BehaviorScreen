#!/usr/bin/env python3
"""
Classify detected saccades using the Dowell et al. reference model.

The input CSV must contain:

    fish
    Amp_L
    Amp_R
    Vergence
    MaxMedAmp_L
    MaxMedAmp_R
    Vel_ccw_L
    Vel_cw_L
    Vel_ccw_R
    Vel_cw_R

Example
-------
python classify_saccades.py \
    --model paper_data/paper_reference.joblib \
    --events saccades.csv \
    --output saccades_classified.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def winsorize_zscore_per_fish(
    features: np.ndarray,
    fish_ids: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Winsorize and standardize each feature within each fish."""
    features = np.asarray(features, dtype=float)
    fish_ids = np.asarray(fish_ids)

    if features.ndim != 2:
        raise ValueError("features must be two-dimensional.")

    if len(features) != len(fish_ids):
        raise ValueError(
            "features and fish_ids must have the same length."
        )

    result = np.full_like(
        features,
        np.nan,
        dtype=float,
    )

    for fish_id in pd.unique(fish_ids):
        mask = fish_ids == fish_id
        fish_features = features[mask].copy()

        lower = np.nanpercentile(
            fish_features,
            lower_percentile,
            axis=0,
        )
        upper = np.nanpercentile(
            fish_features,
            upper_percentile,
            axis=0,
        )

        fish_features = np.clip(
            fish_features,
            lower,
            upper,
        )

        mean = np.nanmean(fish_features, axis=0)
        standard_deviation = np.nanstd(
            fish_features,
            axis=0,
            ddof=0,
        )

        invalid_scale = (
            ~np.isfinite(standard_deviation)
            | (standard_deviation == 0)
        )
        standard_deviation[invalid_scale] = 1.0

        result[mask] = (
            fish_features - mean
        ) / standard_deviation

    return result


def assign_clusters(
    transformed: np.ndarray,
    reference_embedding: np.ndarray,
    reference_labels: np.ndarray,
    k_neighbors: int,
    distance_cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign labels using nearest-neighbor voting in UMAP space."""
    k_neighbors = min(
        k_neighbors,
        len(reference_embedding),
    )

    nearest_neighbors = NearestNeighbors(
        n_neighbors=k_neighbors,
        metric="euclidean",
        n_jobs=-1,
    )
    nearest_neighbors.fit(reference_embedding)

    distances, neighbor_indices = (
        nearest_neighbors.kneighbors(transformed)
    )

    median_distances = np.median(
        distances,
        axis=1,
    )

    assigned_labels = np.full(
        len(transformed),
        -1,
        dtype=int,
    )
    confidence = np.full(
        len(transformed),
        np.nan,
        dtype=float,
    )

    for row_index in range(len(transformed)):
        if (
            distance_cutoff is not None
            and median_distances[row_index] > distance_cutoff
        ):
            continue

        neighbor_labels = reference_labels[
            neighbor_indices[row_index]
        ]

        values, counts = np.unique(
            neighbor_labels,
            return_counts=True,
        )
        winning_index = int(np.argmax(counts))

        assigned_labels[row_index] = int(
            values[winning_index]
        )
        confidence[row_index] = (
            counts[winning_index] / k_neighbors
        )

    return assigned_labels, median_distances, confidence


def apply_bconv_reassignment(
    events: pd.DataFrame,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Reassign flagged convergent events to the paper's BConv classes.

    Paper label mapping:
        4 = Convergent
        7 = Biphasic convergent right
        8 = Biphasic convergent left
    """
    required_columns = {
        "bconv_flag",
        "bconv_side",
    }

    if not required_columns.issubset(events.columns):
        print(
            "[warn] BConv columns are absent; skipping "
            "BConv reassignment."
        )
        return labels

    result = labels.copy()

    flags = (
        events["bconv_flag"]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    sides = (
        events["bconv_side"]
        .fillna("")
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    convergent = result == 4

    result[
        convergent & flags & (sides == "L")
    ] = 8
    result[
        convergent & flags & (sides == "R")
    ] = 7

    return result


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify detected saccades using the paper reference."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Reference model created by paper_reference.py.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Detected-event CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output classified-event CSV.",
    )
    parser.add_argument(
        "--fish-column",
        default="fish",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
        help="Override the model's default neighbor count.",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=None,
        help="Override the model's rejection-distance threshold.",
    )
    parser.add_argument(
        "--no-distance-rejection",
        action="store_true",
        help="Assign every valid event regardless of distance.",
    )
    parser.add_argument(
        "--no-bconv-reassignment",
        action="store_true",
        help="Do not apply the post-classification BConv correction.",
    )

    return parser


def main() -> None:
    """Classify the input event table."""
    args = build_parser().parse_args()

    if not args.model.exists():
        raise FileNotFoundError(args.model)

    if not args.events.exists():
        raise FileNotFoundError(args.events)

    model = joblib.load(args.model)
    events = pd.read_csv(args.events)

    feature_names = model["feature_names"]
    required_columns = {
        args.fish_column,
        *feature_names,
    }
    missing_columns = required_columns.difference(
        events.columns
    )

    if missing_columns:
        raise ValueError(
            f"{args.events} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    raw_features = events[
        feature_names
    ].to_numpy(dtype=float)

    fish_ids = events[
        args.fish_column
    ].to_numpy()

    standardized_features = (
        winsorize_zscore_per_fish(
            raw_features,
            fish_ids,
        )
    )

    valid = np.isfinite(
        standardized_features
    ).all(axis=1)

    embedding = np.full(
        (len(events), 2),
        np.nan,
        dtype=float,
    )
    knn_labels = np.full(
        len(events),
        -1,
        dtype=int,
    )
    median_distances = np.full(
        len(events),
        np.nan,
        dtype=float,
    )
    confidence = np.full(
        len(events),
        np.nan,
        dtype=float,
    )

    if valid.any():
        print(
            f"Transforming {valid.sum():,} valid events..."
        )

        transformed = model["reducer"].transform(
            standardized_features[valid].astype(
                np.float32
            )
        )
        embedding[valid] = transformed

        k_neighbors = (
            args.k_neighbors
            if args.k_neighbors is not None
            else model["k_neighbors"]
        )

        if args.no_distance_rejection:
            distance_cutoff = None
        elif args.distance_cutoff is not None:
            distance_cutoff = args.distance_cutoff
        else:
            distance_cutoff = model["distance_cutoff"]

        (
            local_labels,
            local_distances,
            local_confidence,
        ) = assign_clusters(
            transformed=transformed,
            reference_embedding=model[
                "reference_embedding"
            ],
            reference_labels=model[
                "reference_labels"
            ],
            k_neighbors=k_neighbors,
            distance_cutoff=distance_cutoff,
        )

        valid_indices = np.flatnonzero(valid)

        knn_labels[valid_indices] = local_labels
        median_distances[valid_indices] = local_distances
        confidence[valid_indices] = local_confidence

    if args.no_bconv_reassignment:
        final_labels = knn_labels.copy()
    else:
        final_labels = apply_bconv_reassignment(
            events=events,
            labels=knn_labels,
        )

    label_names = model["label_names"]

    result = events.copy()
    result["embed_x"] = embedding[:, 0]
    result["embed_y"] = embedding[:, 1]
    result["cluster_knn"] = knn_labels
    result["cluster"] = final_labels
    result["cluster_name"] = [
        (
            label_names.get(
                int(label),
                f"Unknown cluster {label}",
            )
            if label >= 0
            else "Unassigned"
        )
        for label in final_labels
    ]
    result["assignment_median_distance"] = (
        median_distances
    )
    result["assignment_confidence"] = confidence
    result["assignment_rejected"] = (
        knn_labels == -1
    )
    result["features_valid"] = valid

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        args.output,
        index=False,
        float_format="%.10g",
    )

    print(f"Saved classified events to {args.output}")
    print("\nClass distribution:")

    distribution = (
        result[["cluster", "cluster_name"]]
        .value_counts(dropna=False)
        .sort_index()
    )
    print(distribution)

    rejected_fraction = np.mean(
        result["assignment_rejected"]
    )
    print(f"\nRejected: {rejected_fraction:.1%}")


if __name__ == "__main__":
    main()