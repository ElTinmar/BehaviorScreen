#!/usr/bin/env python3

from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def winsorize_zscore_per_fish(
    X,
    fish_ids,
    lower=0.5,
    upper=99.5,
):
    X = np.asarray(X, dtype=float)
    fish_ids = np.asarray(fish_ids)

    result = np.full_like(X, np.nan)

    for fish in pd.unique(fish_ids):
        mask = fish_ids == fish
        sub = X[mask].copy()

        lo = np.nanpercentile(sub, lower, axis=0)
        hi = np.nanpercentile(sub, upper, axis=0)
        sub = np.clip(sub, lo, hi)

        mean = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0, ddof=0)
        sd[(sd == 0) | ~np.isfinite(sd)] = 1.0

        result[mask] = (sub - mean) / sd

    return result


def read_table(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_table(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        table.to_csv(path, index=False)
    else:
        table.to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fish-column", default="fish")
    parser.add_argument("--distance-cutoff", type=float, default=None)
    args = parser.parse_args()

    model = joblib.load(args.model)
    events = read_table(args.events)

    feature_names = model["feature_names"]

    required = [args.fish_column, *feature_names]
    missing = [name for name in required if name not in events]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X_raw = events[feature_names].to_numpy(float)
    fish = events[args.fish_column].to_numpy()

    X_z = winsorize_zscore_per_fish(X_raw, fish)
    valid = np.isfinite(X_z).all(axis=1)

    embedding = np.full((len(events), 2), np.nan)
    labels = np.full(len(events), -1, dtype=int)
    median_distance = np.full(len(events), np.nan)
    confidence = np.full(len(events), np.nan)

    transformed = model["reducer"].transform(
        X_z[valid].astype(np.float32)
    )
    embedding[valid] = transformed

    reference_embedding = model["reference_embedding"]
    reference_labels = model["reference_labels"]
    k = min(model["k_neighbors"], len(reference_embedding))

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        n_jobs=-1,
    ).fit(reference_embedding)

    distances, indices = nn.kneighbors(transformed)

    cutoff = (
        args.distance_cutoff
        if args.distance_cutoff is not None
        else model["distance_cutoff"]
    )

    output_rows = np.flatnonzero(valid)

    for local_row, output_row in enumerate(output_rows):
        distance = np.median(distances[local_row])
        median_distance[output_row] = distance

        if distance > cutoff:
            continue

        neighbor_labels = reference_labels[indices[local_row]]
        values, counts = np.unique(
            neighbor_labels,
            return_counts=True,
        )

        winner = np.argmax(counts)
        labels[output_row] = int(values[winner])
        confidence[output_row] = counts[winner] / k

    result = events.copy()
    result["embed_x"] = embedding[:, 0]
    result["embed_y"] = embedding[:, 1]
    result["cluster"] = labels
    result["cluster_name"] = [
        model["label_names"].get(int(label), "Unassigned")
        if label >= 0
        else "Unassigned"
        for label in labels
    ]
    result["assignment_median_distance"] = median_distance
    result["assignment_confidence"] = confidence
    result["assignment_rejected"] = labels == -1
    result["features_valid"] = valid

    write_table(result, args.output)

    print(result["cluster_name"].value_counts(dropna=False))
    print(f"Rejected: {(labels == -1).mean():.1%}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()