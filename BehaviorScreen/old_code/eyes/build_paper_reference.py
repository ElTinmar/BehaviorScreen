#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
import umap


FEATURE_NAMES = [
    "Amp_L",
    "Amp_R",
    "Vergence",
    "MaxMedAmp_L",
    "MaxMedAmp_R",
    "Vel_ccw_L",
    "Vel_cw_L",
    "Vel_ccw_R",
    "Vel_cw_R",
]

LABEL_NAMES = {
    0: "Unclassified",
    1: "Conjugate left",
    2: "Conjugate right",
    3: "Miniature convergent",
    4: "Convergent",
    5: "Non-saccadic",
    6: "Divergent",
    7: "Biphasic convergent right",
    8: "Biphasic convergent left",
}


def read_hdf5_vector(path: Path, dataset_path: str) -> np.ndarray:
    """Read one numeric vector from a MATLAB v7.3 MAT file."""
    with h5py.File(path, "r") as handle:
        if dataset_path not in handle:
            print("Available top-level HDF5 objects:")
            handle.visit(print)
            raise KeyError(
                f"{dataset_path!r} was not found in {path}"
            )

        return np.asarray(handle[dataset_path]).squeeze()


def load_released_reference(data_dir: Path):
    metrics_path = data_dir / "AgMetrics_05_06_2022_updated.mat"
    nmap_path = data_dir / "nmapIdx220606.mat"
    initial_labels_path = data_dir / "umapIdx220606.mat"

    # Read only this small vector from the very large v7.3 file.
    in_bout = read_hdf5_vector(
        metrics_path,
        "AgMetrics/InBoutDex",
    ).astype(bool)

    nmap = loadmat(nmap_path, simplify_cells=True)

    X_all = np.asarray(
        nmap["umapAll"]["raw_data"],
        dtype=np.float32,
    )
    paper_embedding_all = np.asarray(
        nmap["umapAll"]["embedding"],
        dtype=np.float32,
    )
    labels_all = np.asarray(
        nmap["IdxAll"],
    ).squeeze().astype(np.int16)

    X_heldout = np.asarray(
        nmap["umapUnclass"]["raw_data"],
        dtype=np.float32,
    )
    paper_embedding_heldout = np.asarray(
        nmap["umapUnclass"]["embedding"],
        dtype=np.float32,
    )
    labels_heldout = np.asarray(
        nmap["Idx_unclass"],
    ).squeeze().astype(np.int16)

    initial = loadmat(initial_labels_path, simplify_cells=True)
    labels_training_original = np.asarray(
        initial["Idx"]
    ).squeeze().astype(np.int16)

    n_total = len(in_bout)
    training_mask = ~in_bout

    assert n_total == 335_442
    assert training_mask.sum() == 213_462
    assert in_bout.sum() == 121_980

    assert X_all.shape == (n_total, 9)
    assert paper_embedding_all.shape == (n_total, 2)
    assert labels_all.shape == (n_total,)

    assert X_heldout.shape == (in_bout.sum(), 9)
    assert paper_embedding_heldout.shape == (in_bout.sum(), 2)
    assert labels_heldout.shape == (in_bout.sum(),)
    assert labels_training_original.shape == (training_mask.sum(),)

    # Confirm that nmapIdx220606 preserves the original AgMetrics row order.
    checks = {
        "heldout_features": np.allclose(
            X_all[in_bout],
            X_heldout,
            equal_nan=True,
        ),
        "heldout_embedding": np.allclose(
            paper_embedding_all[in_bout],
            paper_embedding_heldout,
            equal_nan=True,
        ),
        "heldout_labels": np.array_equal(
            labels_all[in_bout],
            labels_heldout,
        ),
        "training_labels": np.array_equal(
            labels_all[training_mask],
            labels_training_original,
        ),
    }

    print("Alignment checks:")
    for name, passed in checks.items():
        print(f"  {name:22s}: {passed}")

    if not all(checks.values()):
        raise RuntimeError(
            "The released arrays are not aligned as expected. "
            "Do not build the model until this is resolved."
        )

    return {
        "X_all": X_all,
        "paper_embedding_all": paper_embedding_all,
        "labels_all": labels_all,
        "training_mask": training_mask,
        "in_bout_mask": in_bout,
    }


def fit_reference(
    data_dir: Path,
    output: Path,
    figures: Path | None,
    random_state: int,
):
    released = load_released_reference(data_dir)

    training_mask = released["training_mask"]

    X_train = released["X_all"][training_mask]
    labels_train = released["labels_all"][training_mask]
    paper_embedding_train = released["paper_embedding_all"][training_mask]

    finite = np.isfinite(X_train).all(axis=1)

    print(f"Training events: {len(X_train):,}")
    print(f"Finite events:   {finite.sum():,}")

    print("\nPublished training-label distribution:")
    unique, counts = np.unique(labels_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(
            f"  {label}: {count:7,d}  "
            f"{LABEL_NAMES.get(int(label), 'unknown')}"
        )

    # Fit a Python UMAP because the serialized MATLAB UMAP object cannot
    # reliably be used by umap-learn.
    reducer = umap.UMAP(
        n_neighbors=199,
        min_dist=0.11,
        n_components=2,
        metric="euclidean",
        random_state=random_state,
        transform_seed=random_state,
        low_memory=True,
        verbose=True,
    )

    print("\nFitting Python UMAP...")
    python_embedding = reducer.fit_transform(X_train[finite])

    # Label 0 is unclassified and label 5 is explicitly non-saccadic.
    # Neither should vote for one of the biological saccade classes.
    biological = finite & ~np.isin(labels_train, [0, 5])

    # The reducer was fit only on finite rows, so convert the mask to the
    # corresponding finite-array indexing.
    labels_finite = labels_train[finite]
    biological_finite = ~np.isin(labels_finite, [0, 5])

    reference_embedding = python_embedding[biological_finite]
    reference_labels = labels_finite[biological_finite]

    # Calibrate a rejection threshold in the Python embedding rather than
    # copying the paper's MATLAB-UMAP threshold of 0.3.
    k = min(101, len(reference_embedding))

    neighbors = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        n_jobs=-1,
    ).fit(reference_embedding)

    distances, _ = neighbors.kneighbors(reference_embedding)

    # Remove each point's distance to itself.
    distances = distances[:, 1:]
    median_distance = np.median(distances, axis=1)
    distance_cutoff = float(np.quantile(median_distance, 0.995))

    print(f"Calibrated median-distance cutoff: {distance_cutoff:.4f}")

    model = {
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "reducer": reducer,
        "reference_embedding": reference_embedding.astype(np.float32),
        "reference_labels": reference_labels.astype(np.int16),
        "python_training_embedding": python_embedding.astype(np.float32),
        "training_labels": labels_finite.astype(np.int16),
        "paper_training_embedding": paper_embedding_train[finite],
        "X_training_z": X_train[finite],
        "k_neighbors": 100,
        "distance_cutoff": distance_cutoff,
        "preprocessing": {
            "winsorization": [0.5, 99.5],
            "zscore": "within fish",
            "input_order": FEATURE_NAMES,
        },
        "excluded_voting_labels": [0, 5],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output, compress=3)
    print(f"Saved model to {output}")

    metadata = {
        "data_directory": str(data_dir),
        "n_training_events": int(finite.sum()),
        "n_voting_reference_events": int(biological_finite.sum()),
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "distance_cutoff": distance_cutoff,
        "random_state": random_state,
    }

    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2)
    )

    if figures is not None:
        make_figures(model, figures)


def scatter_labels(embedding, labels, output, title):
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab10")

    rng = np.random.default_rng(0)
    if len(embedding) > 100_000:
        selected = rng.choice(
            len(embedding),
            size=100_000,
            replace=False,
        )
    else:
        selected = np.arange(len(embedding))

    for label in sorted(np.unique(labels)):
        mask = labels[selected] == label
        points = embedding[selected][mask]

        color = (
            "lightgray"
            if label == 0
            else cmap((int(label) - 1) % 10)
        )

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=2,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            color=color,
            label=(
                f"{label}: "
                f"{LABEL_NAMES.get(int(label), 'unknown')}"
            ),
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
        markerscale=3,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def scatter_features(embedding, X, output, title):
    rng = np.random.default_rng(0)
    n = min(60_000, len(embedding))
    selected = rng.choice(len(embedding), n, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))

    for column, ax in enumerate(axes.ravel()):
        values = np.clip(X[selected, column], -2, 2)

        scatter = ax.scatter(
            embedding[selected, 0],
            embedding[selected, 1],
            c=values,
            cmap="coolwarm",
            vmin=-2,
            vmax=2,
            s=1,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(FEATURE_NAMES[column])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(
        scatter,
        ax=axes.ravel().tolist(),
        label="Within-fish standardized value",
        fraction=0.025,
        pad=0.02,
    )
    fig.suptitle(title)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(model, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

    scatter_labels(
        model["paper_training_embedding"],
        model["training_labels"],
        directory / "paper_umap_labels.png",
        "Released MATLAB UMAP and published labels",
    )

    scatter_features(
        model["paper_training_embedding"],
        model["X_training_z"],
        directory / "paper_umap_features.png",
        "Released MATLAB UMAP colored by published standardized metrics",
    )

    scatter_labels(
        model["python_training_embedding"],
        model["training_labels"],
        directory / "python_umap_labels.png",
        "Python UMAP with published labels",
    )

    scatter_features(
        model["python_training_embedding"],
        model["X_training_z"],
        directory / "python_umap_features.png",
        "Python UMAP colored by published standardized metrics",
    )

    print(f"Saved figures to {directory}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/home/martin/Downloads/"
            "Saccade_Dowell_et_al_2024/Data"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_tethered_reference.joblib"),
    )
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    fit_reference(
        args.data_dir,
        args.output,
        args.figures,
        args.random_state,
    )


if __name__ == "__main__":
    main()