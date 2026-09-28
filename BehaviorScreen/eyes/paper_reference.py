#!/usr/bin/env python3
"""
Download, extract, build, and plot the Dowell et al. saccade reference.
https://doi.org/10.1016/j.cub.2024.08.008

The downloaded archive is:

    https://data.mendeley.com/public-api/zip/vd5zdfwc37/download/1

Only these files are extracted:

    AgMetrics_05_06_2022_updated.mat
    nmapIdx220606.mat

Commands
--------
Download and selectively extract the required MAT files:

    python paper_reference.py download \
        --output-dir paper_data

Export the relevant paper data to CSV:

    python paper_reference.py extract \
        --data-dir paper_data \
        --output paper_data/paper_reference.csv

Fit a Python UMAP reference and save a joblib model:

    python paper_reference.py build \
        --reference-csv paper_data/paper_reference.csv \
        --output paper_data/paper_reference.joblib

Plot the released MATLAB UMAP and the fitted Python UMAP:

    python paper_reference.py plot \
        --reference-csv paper_data/paper_reference.csv \
        --model paper_data/paper_reference.joblib \
        --output paper_data/reference_umap.png
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
import umap


DOWNLOAD_URL = (
    "https://data.mendeley.com/public-api/zip/vd5zdfwc37/download/1"
)

REQUIRED_MAT_FILES = {
    "AgMetrics_05_06_2022_updated.mat",
    "nmapIdx220606.mat",
}

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

STANDARDIZED_FEATURE_NAMES = [
    f"{name}_z" for name in FEATURE_NAMES
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

LABEL_COLORS = {
    0: "#bdbdbd",
    1: "#66c2a5",
    2: "#e78ac3",
    3: "#d95f02",
    4: "#377eb8",
    5: "#969696",
    6: "#8c510a",
    7: "#1f78b4",
    8: "#00bfc4",
}


def download_with_progress(url: str, destination: Path) -> None:
    """Download a URL while displaying approximate progress."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    def report(
        block_count: int,
        block_size: int,
        total_size: int,
    ) -> None:
        downloaded = block_count * block_size

        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            downloaded_mb = downloaded / 1024**2
            total_mb = total_size / 1024**2
            print(
                f"\rDownloading: {percent:6.2f}% "
                f"({downloaded_mb:,.1f}/{total_mb:,.1f} MB)",
                end="",
                flush=True,
            )
        else:
            downloaded_mb = downloaded / 1024**2
            print(
                f"\rDownloaded: {downloaded_mb:,.1f} MB",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(
        url,
        destination,
        reporthook=report,
    )
    print()


def selectively_extract_zip(
    zip_path: Path,
    output_dir: Path,
    required_names: set[str],
) -> None:
    """Extract selected files from a ZIP archive by basename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_names: set[str] = set()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            basename = Path(member.filename).name

            if basename not in required_names:
                continue

            destination = output_dir / basename

            print(f"Extracting {basename}...")
            with archive.open(member) as source:
                with destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

            extracted_names.add(basename)

    missing = required_names.difference(extracted_names)

    if missing:
        raise FileNotFoundError(
            "The downloaded archive did not contain: "
            f"{sorted(missing)}"
        )


def download_paper_data(
    output_dir: Path,
    keep_zip: bool = False,
) -> None:
    """Download the paper archive and extract only required MAT files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "dowell_2024_data.zip"

    existing = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }

    if REQUIRED_MAT_FILES.issubset(existing):
        print("Required MAT files already exist; skipping download.")
        return

    if not zip_path.exists():
        print(f"Downloading data from:\n{DOWNLOAD_URL}")
        download_with_progress(DOWNLOAD_URL, zip_path)
    else:
        print(f"Using existing archive: {zip_path}")

    selectively_extract_zip(
        zip_path=zip_path,
        output_dir=output_dir,
        required_names=REQUIRED_MAT_FILES,
    )

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"Removed archive: {zip_path}")

    print(f"Paper data are available in: {output_dir}")


def read_in_bout_mask(metrics_path: Path) -> np.ndarray:
    """Read only InBoutDex from the large MATLAB v7.3 metrics file."""
    dataset_path = "AgMetrics/InBoutDex"

    with h5py.File(metrics_path, "r") as handle:
        if dataset_path not in handle:
            print("Available HDF5 objects:")
            handle.visit(print)
            raise KeyError(
                f"{dataset_path!r} was not found in {metrics_path}"
            )

        in_bout = np.asarray(
            handle[dataset_path]
        ).squeeze()

    return in_bout.astype(bool)


def load_reference_arrays(data_dir: Path) -> dict[str, np.ndarray]:
    """Load and validate the released reference arrays."""
    metrics_path = (
        data_dir / "AgMetrics_05_06_2022_updated.mat"
    )
    nmap_path = data_dir / "nmapIdx220606.mat"

    for path in (metrics_path, nmap_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run the download command first."
            )

    in_bout = read_in_bout_mask(metrics_path)
    released = loadmat(nmap_path, simplify_cells=True)

    features_all = np.asarray(
        released["umapAll"]["raw_data"],
        dtype=np.float64,
    )
    embedding_all = np.asarray(
        released["umapAll"]["embedding"],
        dtype=np.float64,
    )
    labels_all = np.asarray(
        released["IdxAll"]
    ).squeeze().astype(np.int16)

    heldout_features = np.asarray(
        released["umapUnclass"]["raw_data"],
        dtype=np.float64,
    )
    heldout_embedding = np.asarray(
        released["umapUnclass"]["embedding"],
        dtype=np.float64,
    )
    heldout_labels = np.asarray(
        released["Idx_unclass"]
    ).squeeze().astype(np.int16)

    number_of_events = len(in_bout)

    expected_shapes = {
        "features_all": (number_of_events, 9),
        "embedding_all": (number_of_events, 2),
        "labels_all": (number_of_events,),
    }
    actual_shapes = {
        "features_all": features_all.shape,
        "embedding_all": embedding_all.shape,
        "labels_all": labels_all.shape,
    }

    for name, expected_shape in expected_shapes.items():
        if actual_shapes[name] != expected_shape:
            raise ValueError(
                f"{name} has shape {actual_shapes[name]}; "
                f"expected {expected_shape}."
            )

    number_of_heldout_events = int(in_bout.sum())

    if heldout_features.shape != (number_of_heldout_events, 9):
        raise ValueError(
            "Unexpected held-out feature shape: "
            f"{heldout_features.shape}"
        )

    if heldout_embedding.shape != (number_of_heldout_events, 2):
        raise ValueError(
            "Unexpected held-out embedding shape: "
            f"{heldout_embedding.shape}"
        )

    if heldout_labels.shape != (number_of_heldout_events,):
        raise ValueError(
            "Unexpected held-out label shape: "
            f"{heldout_labels.shape}"
        )

    checks = {
        "heldout_features": np.allclose(
            features_all[in_bout],
            heldout_features,
            equal_nan=True,
        ),
        "heldout_embedding": np.allclose(
            embedding_all[in_bout],
            heldout_embedding,
            equal_nan=True,
        ),
        "heldout_labels": np.array_equal(
            labels_all[in_bout],
            heldout_labels,
        ),
    }

    print("Alignment checks:")
    for name, passed in checks.items():
        print(f"  {name:20s}: {passed}")

    if not all(checks.values()):
        raise RuntimeError(
            "The released feature, embedding, and label arrays are "
            "not aligned as expected."
        )

    print(f"All tethered events:     {number_of_events:,}")
    print(f"Reference events:        {(~in_bout).sum():,}")
    print(f"Held-out in-bout events: {in_bout.sum():,}")

    return {
        "features_all": features_all,
        "embedding_all": embedding_all,
        "labels_all": labels_all,
        "in_bout": in_bout,
    }


def extract_reference_csv(
    data_dir: Path,
    output_path: Path,
) -> None:
    """Export the relevant released arrays to a Python-friendly CSV."""
    arrays = load_reference_arrays(data_dir)

    features = arrays["features_all"]
    embedding = arrays["embedding_all"]
    labels = arrays["labels_all"]
    in_bout = arrays["in_bout"]

    table = pd.DataFrame(
        features,
        columns=STANDARDIZED_FEATURE_NAMES,
    )

    table.insert(
        0,
        "paper_event_index",
        np.arange(len(table)),
    )
    table["training_reference"] = ~in_bout
    table["in_bout_heldout"] = in_bout
    table["cluster"] = labels
    table["cluster_name"] = [
        LABEL_NAMES.get(int(label), f"Unknown {label}")
        for label in labels
    ]
    table["paper_umap_x"] = embedding[:, 0]
    table["paper_umap_y"] = embedding[:, 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        output_path,
        index=False,
        float_format="%.10g",
    )

    print(f"Saved {len(table):,} events to {output_path}")
    print("\nReference label distribution:")

    reference = table[table["training_reference"]]
    distribution = (
        reference[["cluster", "cluster_name"]]
        .value_counts()
        .sort_index()
    )
    print(distribution)


def calculate_distance_cutoff(
    embedding: np.ndarray,
    k_neighbors: int,
    quantile: float,
) -> float:
    """Calculate a reference-neighborhood rejection threshold."""
    number_of_neighbors = min(
        k_neighbors + 1,
        len(embedding),
    )

    nearest_neighbors = NearestNeighbors(
        n_neighbors=number_of_neighbors,
        metric="euclidean",
        n_jobs=-1,
    )
    nearest_neighbors.fit(embedding)

    distances, _ = nearest_neighbors.kneighbors(embedding)

    if distances.shape[1] > 1:
        distances = distances[:, 1:]

    median_distances = np.median(distances, axis=1)
    cutoff = float(np.quantile(median_distances, quantile))

    return cutoff


def build_reference_model(
    reference_csv: Path,
    output_path: Path,
    random_state: int = 0,
    n_neighbors: int = 199,
    min_dist: float = 0.11,
    k_neighbors: int = 100,
    distance_quantile: float = 0.995,
) -> None:
    """
    Fit a Python UMAP on the paper's tethered non-swim reference events.

    All published labels are retained as voting classes, including:
        0 = Unclassified
        5 = Non-saccadic
    """
    reference_table = pd.read_csv(reference_csv)

    required_columns = {
        "training_reference",
        "cluster",
        *STANDARDIZED_FEATURE_NAMES,
    }
    missing_columns = required_columns.difference(
        reference_table.columns
    )

    if missing_columns:
        raise ValueError(
            f"{reference_csv} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    training_mask = (
        reference_table["training_reference"]
        .astype(bool)
        .to_numpy()
    )

    training_table = reference_table.loc[
        training_mask
    ].reset_index(drop=True)

    features = training_table[
        STANDARDIZED_FEATURE_NAMES
    ].to_numpy(dtype=np.float32)

    labels = training_table["cluster"].to_numpy(
        dtype=np.int16
    )

    finite = np.isfinite(features).all(axis=1)

    if not finite.all():
        print(
            f"Removing {(~finite).sum():,} reference events with "
            "non-finite features."
        )
        features = features[finite]
        labels = labels[finite]

    print(f"Fitting UMAP on {len(features):,} reference events...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=random_state,
        transform_seed=random_state,
        low_memory=True,
        verbose=True,
    )

    embedding = reducer.fit_transform(features)

    distance_cutoff = calculate_distance_cutoff(
        embedding=embedding,
        k_neighbors=k_neighbors,
        quantile=distance_quantile,
    )

    model = {
        "model_version": 1,
        "feature_names": FEATURE_NAMES,
        "standardized_feature_names": STANDARDIZED_FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "reducer": reducer,
        "reference_embedding": embedding.astype(np.float32),
        "reference_labels": labels.astype(np.int16),
        "k_neighbors": k_neighbors,
        "distance_cutoff": distance_cutoff,
        "distance_quantile": distance_quantile,
        "umap_parameters": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": "euclidean",
            "random_state": random_state,
        },
        "preprocessing": {
            "winsorization_percentiles": [0.5, 99.5],
            "zscore": "within fish",
        },
        # Explicitly retain all paper labels, including 0 and 5.
        "included_labels": sorted(
            int(label) for label in np.unique(labels)
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path, compress=3)

    print(f"Saved model to {output_path}")
    print(f"Distance cutoff: {distance_cutoff:.4f}")
    print(f"Included labels: {model['included_labels']}")


def plot_labelled_embedding(
    axis: plt.Axes,
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    max_points: int,
    seed: int,
) -> None:
    """Plot an embedding colored by published class."""
    finite = (
        np.isfinite(embedding).all(axis=1)
        & np.isfinite(labels)
    )
    embedding = embedding[finite]
    labels = labels[finite]

    random_generator = np.random.default_rng(seed)

    if len(embedding) > max_points:
        selected = random_generator.choice(
            len(embedding),
            size=max_points,
            replace=False,
        )
        embedding = embedding[selected]
        labels = labels[selected]

    for label in sorted(np.unique(labels)):
        mask = labels == label
        label_value = int(label)

        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=2,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            color=LABEL_COLORS.get(label_value, "black"),
            label=(
                f"{label_value}: "
                f"{LABEL_NAMES.get(label_value, 'Unknown')}"
            ),
        )

    axis.set_title(title)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_aspect("equal", adjustable="datalim")


def plot_reference_umaps(
    reference_csv: Path,
    model_path: Path,
    output_path: Path,
    max_points: int = 100_000,
    seed: int = 0,
) -> None:
    """Plot the released MATLAB UMAP beside the fitted Python UMAP."""
    table = pd.read_csv(reference_csv)
    model = joblib.load(model_path)

    training = table[
        table["training_reference"].astype(bool)
    ].reset_index(drop=True)

    paper_embedding = training[
        ["paper_umap_x", "paper_umap_y"]
    ].to_numpy(dtype=float)

    paper_labels = training["cluster"].to_numpy(dtype=int)

    python_embedding = model["reference_embedding"]
    python_labels = model["reference_labels"]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(17, 8),
    )

    plot_labelled_embedding(
        axis=axes[0],
        embedding=paper_embedding,
        labels=paper_labels,
        title="Released MATLAB UMAP",
        max_points=max_points,
        seed=seed,
    )

    plot_labelled_embedding(
        axis=axes[1],
        embedding=python_embedding,
        labels=python_labels,
        title="Python UMAP used for classification",
        max_points=max_points,
        seed=seed,
    )

    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=3,
    )

    figure.suptitle(
        "Dowell et al. tethered non-swim reference",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 0.86, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved UMAP figure to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and build the Dowell et al. "
            "saccade-classification reference."
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download and selectively extract the paper data.",
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_data"),
    )
    download_parser.add_argument(
        "--keep-zip",
        action="store_true",
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract relevant paper arrays to CSV.",
    )
    extract_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("paper_data"),
    )
    extract_parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_data/paper_reference.csv"),
    )

    build_model_parser = subparsers.add_parser(
        "build",
        help="Fit the Python UMAP reference model.",
    )
    build_model_parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
    )
    build_model_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    build_model_parser.add_argument(
        "--random-state",
        type=int,
        default=0,
    )
    build_model_parser.add_argument(
        "--n-neighbors",
        type=int,
        default=199,
    )
    build_model_parser.add_argument(
        "--min-dist",
        type=float,
        default=0.11,
    )
    build_model_parser.add_argument(
        "--k-neighbors",
        type=int,
        default=100,
    )
    build_model_parser.add_argument(
        "--distance-quantile",
        type=float,
        default=0.995,
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot released and Python reference UMAPs.",
    )
    plot_parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
    )
    plot_parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    plot_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    plot_parser.add_argument(
        "--max-points",
        type=int,
        default=100_000,
    )
    plot_parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser


def main() -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args()

    if args.command == "download":
        download_paper_data(
            output_dir=args.output_dir,
            keep_zip=args.keep_zip,
        )
    elif args.command == "extract":
        extract_reference_csv(
            data_dir=args.data_dir,
            output_path=args.output,
        )
    elif args.command == "build":
        build_reference_model(
            reference_csv=args.reference_csv,
            output_path=args.output,
            random_state=args.random_state,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            k_neighbors=args.k_neighbors,
            distance_quantile=args.distance_quantile,
        )
    elif args.command == "plot":
        plot_reference_umaps(
            reference_csv=args.reference_csv,
            model_path=args.model,
            output_path=args.output,
            max_points=args.max_points,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()