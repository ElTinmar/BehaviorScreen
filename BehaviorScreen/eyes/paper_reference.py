#!/usr/bin/env python3
"""
Download, extract, build, and plot the Dowell et al. saccade reference.

Paper
-----
Dowell et al. (2024)
https://doi.org/10.1016/j.cub.2024.08.008

Dataset
-------
https://data.mendeley.com/datasets/vd5zdfwc37/1

Commands
--------
Download the complete ZIP and extract the required MAT files:

    python -m BehaviorScreen.eyes.paper_reference download \
        --output-dir paper_data

Keep the downloaded ZIP:

    python -m BehaviorScreen.eyes.paper_reference download \
        --output-dir paper_data \
        --keep-zip

Force a new download:

    python -m BehaviorScreen.eyes.paper_reference download \
        --output-dir paper_data \
        --overwrite

Export the relevant reference arrays to CSV:

    python -m BehaviorScreen.eyes.paper_reference extract \
        --data-dir paper_data \
        --output paper_data/paper_reference.csv

Fit a Python UMAP reference model:

    python -m BehaviorScreen.eyes.paper_reference build \
        --reference-csv paper_data/paper_reference.csv \
        --output paper_data/paper_reference.joblib

Plot the released MATLAB UMAP and the fitted Python UMAP:

    python -m BehaviorScreen.eyes.paper_reference plot \
        --reference-csv paper_data/paper_reference.csv \
        --model paper_data/paper_reference.joblib \
        --output paper_data/reference_umap.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import umap
from requests.adapters import HTTPAdapter
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from urllib3.util.retry import Retry


DATASET_ID = "vd5zdfwc37"
DATASET_VERSION = 1

DATASET_PAGE_URL = (
    f"https://data.mendeley.com/datasets/"
    f"{DATASET_ID}/{DATASET_VERSION}"
)

DOWNLOAD_URL = (
    f"https://data.mendeley.com/public-api/zip/"
    f"{DATASET_ID}/download/{DATASET_VERSION}"
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
    f"{feature_name}_z"
    for feature_name in FEATURE_NAMES
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

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

DOWNLOAD_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "application/zip, application/octet-stream;q=0.9, "
        "*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
    "Referer": DATASET_PAGE_URL,
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------
# Download and extraction
# ---------------------------------------------------------------------


def create_download_session() -> requests.Session:
    """Create an HTTP session with browser-like headers and retries."""
    retry_policy = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=2,
        pool_maxsize=2,
    )

    session = requests.Session()
    session.headers.update(DOWNLOAD_HEADERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def print_download_progress(
    downloaded_bytes: int,
    total_bytes: int | None,
) -> None:
    """Print progress for a streaming download."""
    downloaded_mb = downloaded_bytes / 1024**2

    if total_bytes is None or total_bytes <= 0:
        message = f"\rDownloaded: {downloaded_mb:,.1f} MB"
    else:
        total_mb = total_bytes / 1024**2
        percentage = min(
            100.0,
            downloaded_bytes / total_bytes * 100.0,
        )
        message = (
            f"\rDownloading: {percentage:6.2f}% "
            f"({downloaded_mb:,.1f}/{total_mb:,.1f} MB)"
        )

    print(message, end="", flush=True)


def validate_zip_file(path: Path) -> None:
    """Raise an informative error if a downloaded file is not a ZIP."""
    if zipfile.is_zipfile(path):
        return

    preview = b""

    if path.exists():
        with path.open("rb") as input_file:
            preview = input_file.read(500)

    raise RuntimeError(
        "The downloaded file is not a valid ZIP archive.\n"
        f"Path: {path}\n"
        f"First bytes: {preview!r}"
    )


def download_with_requests(
    url: str,
    destination: Path,
) -> None:
    """Download the full archive with requests."""
    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )
    temporary_path.unlink(missing_ok=True)

    session = create_download_session()

    try:
        # Establish any cookies that the download service expects.
        try:
            landing_response = session.get(
                DATASET_PAGE_URL,
                allow_redirects=True,
                timeout=(30, 60),
            )
            print(
                "Dataset page response: "
                f"HTTP {landing_response.status_code}"
            )
        except requests.RequestException as error:
            print(
                "[warn] Could not open the dataset page before "
                f"downloading: {error}"
            )

        print(f"Downloading full archive from:\n{url}")

        with session.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=(30, 1800),
        ) as response:
            print(f"Download response: HTTP {response.status_code}")
            print(f"Final URL: {response.url}")

            if response.status_code == 403:
                preview = response.content[:500]
                raise PermissionError(
                    "Mendeley rejected the archive request with HTTP 403.\n"
                    f"Final URL: {response.url}\n"
                    f"Response preview: {preview!r}"
                )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                "",
            )
            content_length = response.headers.get(
                "Content-Length"
            )

            total_bytes = (
                int(content_length)
                if content_length is not None
                else None
            )

            print(f"Content type: {content_type}")

            downloaded_bytes = 0

            with temporary_path.open("wb") as output_file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    output_file.write(chunk)
                    downloaded_bytes += len(chunk)

                    print_download_progress(
                        downloaded_bytes=downloaded_bytes,
                        total_bytes=total_bytes,
                    )

        print()

        validate_zip_file(temporary_path)
        temporary_path.replace(destination)

        print(f"Saved archive: {destination}")

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    finally:
        session.close()


def download_with_curl(
    url: str,
    destination: Path,
) -> None:
    """
    Download the full archive using curl.

    This is used as a fallback if requests is rejected by the server.
    """
    curl_path = shutil.which("curl")

    if curl_path is None:
        raise RuntimeError(
            "The requests-based download failed and curl is not "
            "installed, so the fallback downloader cannot be used."
        )

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )
    temporary_path.unlink(missing_ok=True)

    command = [
        curl_path,
        "--fail",
        "--location",
        "--show-error",
        "--progress-bar",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "7200",
        "--user-agent",
        USER_AGENT,
        "--referer",
        DATASET_PAGE_URL,
        "--header",
        "Accept: application/zip, application/octet-stream, */*",
        "--output",
        str(temporary_path),
        url,
    ]

    print("Trying curl fallback:")
    print(" ".join(command))

    try:
        subprocess.run(command, check=True)
        validate_zip_file(temporary_path)
        temporary_path.replace(destination)
        print(f"Saved archive: {destination}")

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def download_with_progress(
    url: str,
    destination: Path,
    overwrite: bool = False,
) -> None:
    """
    Download and validate the complete ZIP archive.

    A temporary ``.part`` file is used so that an interrupted download is
    never mistaken for a completed archive.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        if zipfile.is_zipfile(destination):
            print(f"Using existing archive: {destination}")
            return

        print(
            f"Removing invalid existing archive: {destination}"
        )
        destination.unlink()

    if overwrite:
        destination.unlink(missing_ok=True)

    try:
        download_with_requests(
            url=url,
            destination=destination,
        )
    except Exception as requests_error:
        print(
            "[warn] requests-based download failed:\n"
            f"{requests_error}"
        )
        print("[warn] Trying curl as a fallback.")

        try:
            download_with_curl(
                url=url,
                destination=destination,
            )
        except Exception as curl_error:
            raise RuntimeError(
                "Both download methods failed.\n\n"
                f"requests error:\n{requests_error}\n\n"
                f"curl error:\n{curl_error}\n\n"
                "You can download the archive manually from:\n"
                f"{DATASET_PAGE_URL}"
            ) from curl_error


def selectively_extract_zip(
    zip_path: Path,
    output_dir: Path,
    required_names: set[str],
    overwrite: bool = False,
) -> None:
    """Extract selected files from the downloaded ZIP archive."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_zip_file(zip_path)

    extracted_names: set[str] = set()

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            basename = Path(member.filename).name

            if basename not in required_names:
                continue

            destination = output_dir / basename

            if destination.exists() and not overwrite:
                print(f"[skip] Already extracted: {destination}")
                extracted_names.add(basename)
                continue

            print(
                f"Extracting {member.filename} "
                f"as {destination.name}..."
            )

            temporary_path = destination.with_suffix(
                destination.suffix + ".part"
            )
            temporary_path.unlink(missing_ok=True)

            try:
                with archive.open(member, mode="r") as source:
                    with temporary_path.open("wb") as target:
                        shutil.copyfileobj(
                            source,
                            target,
                            length=1024 * 1024,
                        )

                temporary_path.replace(destination)
                extracted_names.add(basename)

            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

        archive_basenames = {
            Path(member.filename).name
            for member in archive.infolist()
            if not member.is_dir()
        }

    missing_names = required_names.difference(
        extracted_names
    )

    if missing_names:
        raise FileNotFoundError(
            "The downloaded archive did not contain all required files.\n"
            f"Missing: {sorted(missing_names)}\n"
            f"Available archive files: {sorted(archive_basenames)}"
        )


def download_paper_data(
    output_dir: Path,
    keep_zip: bool = False,
    overwrite: bool = False,
) -> None:
    """Download the complete archive and extract the required MAT files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / "dowell_2024_data.zip"

    required_paths = [
        output_dir / filename
        for filename in sorted(REQUIRED_MAT_FILES)
    ]

    if (
        not overwrite
        and all(path.exists() for path in required_paths)
    ):
        print("All required MAT files already exist:")

        for path in required_paths:
            print(f"  {path}")

        print("Skipping archive download.")
        return

    download_with_progress(
        url=DOWNLOAD_URL,
        destination=archive_path,
        overwrite=overwrite,
    )

    selectively_extract_zip(
        zip_path=archive_path,
        output_dir=output_dir,
        required_names=REQUIRED_MAT_FILES,
        overwrite=overwrite,
    )

    if not keep_zip:
        archive_path.unlink(missing_ok=True)
        print(f"Removed archive: {archive_path}")

    print("Required paper files are available:")

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Expected extracted file is missing: {path}"
            )

        print(f"  {path}")


# ---------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------


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

    if in_bout.ndim != 1:
        raise ValueError(
            f"InBoutDex must be one-dimensional after squeezing; "
            f"got shape {in_bout.shape}."
        )

    return in_bout.astype(bool)


def load_reference_arrays(
    data_dir: Path,
) -> dict[str, np.ndarray]:
    """Load and validate the released reference arrays."""
    metrics_path = (
        data_dir / "AgMetrics_05_06_2022_updated.mat"
    )
    nmap_path = data_dir / "nmapIdx220606.mat"

    for path in (metrics_path, nmap_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run the download "
                "command first."
            )

    in_bout = read_in_bout_mask(metrics_path)
    released = loadmat(
        nmap_path,
        simplify_cells=True,
    )

    features_all = np.asarray(
        released["umapAll"]["raw_data"],
        dtype=np.float64,
    )
    embedding_all = np.asarray(
        released["umapAll"]["embedding"],
        dtype=np.float64,
    )
    labels_all = (
        np.asarray(released["IdxAll"])
        .squeeze()
        .astype(np.int16)
    )

    heldout_features = np.asarray(
        released["umapUnclass"]["raw_data"],
        dtype=np.float64,
    )
    heldout_embedding = np.asarray(
        released["umapUnclass"]["embedding"],
        dtype=np.float64,
    )
    heldout_labels = (
        np.asarray(released["Idx_unclass"])
        .squeeze()
        .astype(np.int16)
    )

    number_of_events = len(in_bout)
    number_of_heldout_events = int(in_bout.sum())

    expected_shapes = {
        "features_all": (number_of_events, 9),
        "embedding_all": (number_of_events, 2),
        "labels_all": (number_of_events,),
        "heldout_features": (
            number_of_heldout_events,
            9,
        ),
        "heldout_embedding": (
            number_of_heldout_events,
            2,
        ),
        "heldout_labels": (
            number_of_heldout_events,
        ),
    }

    actual_shapes = {
        "features_all": features_all.shape,
        "embedding_all": embedding_all.shape,
        "labels_all": labels_all.shape,
        "heldout_features": heldout_features.shape,
        "heldout_embedding": heldout_embedding.shape,
        "heldout_labels": heldout_labels.shape,
    }

    for name, expected_shape in expected_shapes.items():
        actual_shape = actual_shapes[name]

        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} has shape {actual_shape}; "
                f"expected {expected_shape}."
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
            "The released feature, embedding, and label arrays "
            "are not aligned as expected."
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
    """Export the released reference arrays to CSV."""
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
        LABEL_NAMES.get(
            int(label),
            f"Unknown cluster {label}",
        )
        for label in labels
    ]
    table["paper_umap_x"] = embedding[:, 0]
    table["paper_umap_y"] = embedding[:, 1]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    table.to_csv(
        output_path,
        index=False,
        float_format="%.10g",
    )

    print(f"Saved {len(table):,} events to {output_path}")

    reference = table.loc[
        table["training_reference"]
    ]

    print("\nPublished training-reference distribution:")
    print(
        reference[["cluster", "cluster_name"]]
        .value_counts()
        .sort_index()
    )


# ---------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------


def parse_boolean_series(series: pd.Series) -> np.ndarray:
    """Convert a CSV boolean column into a NumPy boolean array."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    valid_values = {
        "true",
        "false",
        "1",
        "0",
    }

    unknown = set(normalized.unique()).difference(
        valid_values
    )

    if unknown:
        raise ValueError(
            "Could not parse boolean values: "
            f"{sorted(unknown)}"
        )

    return normalized.isin({"true", "1"}).to_numpy()


def calculate_distance_cutoff(
    embedding: np.ndarray,
    k_neighbors: int,
    quantile: float,
) -> float:
    """Calculate a nearest-neighbor distance rejection threshold."""
    if not 0.0 < quantile <= 1.0:
        raise ValueError(
            "distance quantile must lie in (0, 1]."
        )

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

    distances, _ = nearest_neighbors.kneighbors(
        embedding
    )

    # The first neighbor of each reference point is itself.
    if distances.shape[1] > 1:
        distances = distances[:, 1:]

    median_distances = np.median(
        distances,
        axis=1,
    )

    return float(
        np.quantile(
            median_distances,
            quantile,
        )
    )


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
    Fit a Python UMAP on the paper's non-swim tethered reference events.

    All published labels are retained as voting classes, including:

    - 0: Unclassified
    - 5: Non-saccadic
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

    training_mask = parse_boolean_series(
        reference_table["training_reference"]
    )

    training_table = (
        reference_table.loc[training_mask]
        .reset_index(drop=True)
    )

    features = training_table[
        STANDARDIZED_FEATURE_NAMES
    ].to_numpy(dtype=np.float32)

    labels = training_table[
        "cluster"
    ].to_numpy(dtype=np.int16)

    finite = np.isfinite(features).all(axis=1)

    if not finite.all():
        print(
            f"Removing {(~finite).sum():,} reference events "
            "with non-finite features."
        )
        features = features[finite]
        labels = labels[finite]
        training_table = training_table.loc[
            finite
        ].reset_index(drop=True)

    print(
        f"Fitting Python UMAP on "
        f"{len(features):,} reference events..."
    )

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

    paper_embedding = training_table[
        ["paper_umap_x", "paper_umap_y"]
    ].to_numpy(dtype=np.float32)

    included_labels = sorted(
        int(label)
        for label in np.unique(labels)
    )

    model = {
        "model_version": 1,
        "feature_names": FEATURE_NAMES,
        "standardized_feature_names": (
            STANDARDIZED_FEATURE_NAMES
        ),
        "label_names": LABEL_NAMES,
        "reducer": reducer,
        "reference_embedding": embedding.astype(
            np.float32
        ),
        "reference_labels": labels.astype(np.int16),
        "reference_features_z": features.astype(
            np.float32
        ),
        "paper_reference_embedding": paper_embedding,
        "k_neighbors": int(k_neighbors),
        "distance_cutoff": float(distance_cutoff),
        "distance_quantile": float(distance_quantile),
        "umap_parameters": {
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "metric": "euclidean",
            "random_state": int(random_state),
        },
        "preprocessing": {
            "winsorization_percentiles": [
                0.5,
                99.5,
            ],
            "zscore": "within fish",
        },
        # Includes labels 0 and 5.
        "included_labels": included_labels,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        model,
        output_path,
        compress=3,
    )

    print(f"Saved model to {output_path}")
    print(f"Distance cutoff: {distance_cutoff:.4f}")
    print(f"Included labels: {included_labels}")

    print("\nPublished label distribution:")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    for label, count in zip(
        unique_labels,
        counts,
    ):
        print(
            f"  {int(label)}: {count:8,d}  "
            f"{LABEL_NAMES.get(int(label), 'Unknown')}"
        )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


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
        label_value = int(label)
        mask = labels == label

        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=2,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            color=LABEL_COLORS.get(
                label_value,
                "black",
            ),
            label=(
                f"{label_value}: "
                f"{LABEL_NAMES.get(label_value, 'Unknown')}"
            ),
        )

    axis.set_title(title)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_aspect(
        "equal",
        adjustable="datalim",
    )


def plot_reference_umaps(
    reference_csv: Path,
    model_path: Path,
    output_path: Path,
    max_points: int = 100_000,
    seed: int = 0,
) -> None:
    """Plot the released MATLAB and fitted Python reference UMAPs."""
    reference_table = pd.read_csv(reference_csv)
    model = joblib.load(model_path)

    training_mask = parse_boolean_series(
        reference_table["training_reference"]
    )

    training_table = (
        reference_table.loc[training_mask]
        .reset_index(drop=True)
    )

    paper_embedding = training_table[
        ["paper_umap_x", "paper_umap_y"]
    ].to_numpy(dtype=float)

    paper_labels = training_table[
        "cluster"
    ].to_numpy(dtype=int)

    finite_features = np.isfinite(
        training_table[
            STANDARDIZED_FEATURE_NAMES
        ].to_numpy(dtype=float)
    ).all(axis=1)

    paper_embedding = paper_embedding[
        finite_features
    ]
    paper_labels = paper_labels[
        finite_features
    ]

    python_embedding = np.asarray(
        model["reference_embedding"]
    )
    python_labels = np.asarray(
        model["reference_labels"]
    )

    if len(paper_embedding) != len(python_embedding):
        raise ValueError(
            "The paper and Python reference embeddings have "
            "different numbers of rows."
        )

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

    handles, legend_labels = (
        axes[1].get_legend_handles_labels()
    )

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
    figure.tight_layout(
        rect=(0, 0, 0.86, 1)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved UMAP figure to {output_path}")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------


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
        help=(
            "Download the complete archive and extract the "
            "required MAT files."
        ),
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_data"),
        help="Destination directory. Default: paper_data.",
    )
    download_parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded complete ZIP archive.",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Redownload the archive and replace extracted files."
        ),
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Export the relevant paper arrays to CSV.",
    )
    extract_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("paper_data"),
        help="Directory containing the extracted MAT files.",
    )
    extract_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "paper_data/paper_reference.csv"
        ),
        help="Output reference CSV.",
    )

    build_model_parser = subparsers.add_parser(
        "build",
        help="Fit the Python UMAP reference model.",
    )
    build_model_parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
        help="Reference CSV created by the extract command.",
    )
    build_model_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output joblib model.",
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
        help=(
            "Plot the released MATLAB UMAP and fitted "
            "Python UMAP."
        ),
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
    """Run the selected command."""
    args = build_parser().parse_args()

    if args.command == "download":
        download_paper_data(
            output_dir=args.output_dir,
            keep_zip=args.keep_zip,
            overwrite=args.overwrite,
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

    else:
        raise RuntimeError(
            f"Unknown command: {args.command}"
        )


if __name__ == "__main__":
    main()