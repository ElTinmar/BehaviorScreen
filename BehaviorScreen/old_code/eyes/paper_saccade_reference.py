#!/usr/bin/env python3
"""
Build and use a saccade-classification reference from the data released with:

    doi: 10.1016/j.cub.2024.08.008

Commands
--------
Inspect MAT-file contents:

    python paper_saccade_reference.py inspect Data/*.mat

Build a reference model:

    python paper_saccade_reference.py build \
        --metrics Data/AgMetrics_05_06_2022_updated.mat \
        --labels Data/umapIdx220606.mat \
        --labels-key Idx \
        --output paper_reference.joblib \
        --table paper_reference.parquet \
        --figdir paper_figures

Classify free-swimming events:

    python paper_saccade_reference.py classify \
        --model paper_reference.joblib \
        --events my_freeswim_events.csv \
        --output my_freeswim_classified.parquet

Plot an already-built reference:

    python paper_saccade_reference.py plot \
        --model paper_reference.joblib \
        --figdir paper_figures

Important
---------
The exact variable names inside the released MAT files must first be identified
with the `inspect` command. The --labels-key, --embedding-key, and --z-key
arguments use dotted paths, for example:

    --labels-key Idx
    --embedding-key umap.embedding
    --z-key umap.raw_data

This implementation uses the paper's published event labels, if available.
It fits a new Python umap-learn model because MATLAB UMAP model objects are
generally not compatible with Python's umap-learn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
from sklearn.neighbors import NearestNeighbors
import umap


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Order indicated by BF1_UMAPplots.m / the paper's Figure 1.
PAPER_FEATURE_NAMES = [
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

# Mapping inferred directly from BF1_saccadehistograms.m:
#
# ids      = [1 2 4 8 7 3 6 5 0]
# sacnames = {'LConj','RConj','Conv','BConvL','BConvR',
#             'ConvMini','Div','NonSac','Unclust'}
PAPER_LABEL_NAMES = {
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


# ---------------------------------------------------------------------
# MAT-file utilities
# ---------------------------------------------------------------------

def load_mat_file(path: Path) -> dict[str, Any]:
    """
    Load either an ordinary MAT file or a MATLAB v7.3/HDF5 MAT file.
    """
    path = Path(path)

    try:
        data = loadmat(path, simplify_cells=True)
    except (NotImplementedError, ValueError, TypeError):
        try:
            import mat73
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is MATLAB v7.3. Install support with:\n"
                "    pip install mat73 h5py"
            ) from exc

        data = mat73.loadmat(str(path))

    return {
        key: value
        for key, value in data.items()
        if not str(key).startswith("__")
    }

def iter_objects(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Recursively walk dictionaries loaded with simplify_cells=True."""
    yield prefix, obj

    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_objects(value, child)


def resolve_path(data: dict[str, Any], path: str) -> Any:
    """Resolve a dotted dictionary path, case-insensitively."""
    obj: Any = data

    for component in path.split("."):
        if not isinstance(obj, dict):
            raise KeyError(
                f"Cannot resolve {path!r}: {component!r} is below "
                f"a non-struct object"
            )

        matches = [
            key for key in obj
            if str(key).lower() == component.lower()
        ]

        if len(matches) != 1:
            raise KeyError(
                f"Could not uniquely resolve {component!r} in {list(obj)}"
            )

        obj = obj[matches[0]]

    return obj


def find_struct_with_field(data: dict[str, Any], field: str) -> dict[str, Any]:
    """Find the first nested dictionary containing a field."""
    field_lower = field.lower()

    for _, obj in iter_objects(data):
        if not isinstance(obj, dict):
            continue

        if any(str(key).lower() == field_lower for key in obj):
            return obj

    raise KeyError(f"No MATLAB struct containing field {field!r} was found")


def get_field(struct: dict[str, Any], field: str) -> Any:
    """Get a struct field case-insensitively."""
    matches = [
        key for key in struct
        if str(key).lower() == field.lower()
    ]

    if len(matches) != 1:
        raise KeyError(
            f"Could not uniquely find {field!r}. Fields are: {list(struct)}"
        )

    return struct[matches[0]]


def inspect_mat_files(paths: list[Path]) -> None:
    for path in paths:
        print(f"\n{'=' * 78}")
        print(path)
        print("=" * 78)

        try:
            for name, shape, kind in whosmat(path):
                print(f"top-level: {name:30s} shape={shape!s:18s} type={kind}")
        except Exception as exc:
            print(f"whosmat failed: {exc}")

        try:
            data = load_mat_file(path)
        except Exception as exc:
            print(f"load failed: {exc}")
            continue

        print("\nNested values:")
        for key, obj in iter_objects(data):
            if not key:
                continue

            if isinstance(obj, np.ndarray):
                print(
                    f"  {key:50s} "
                    f"shape={str(obj.shape):18s} dtype={obj.dtype}"
                )
            elif np.isscalar(obj):
                print(f"  {key:50s} scalar={obj!r}")
            elif isinstance(obj, dict):
                print(f"  {key:50s} struct fields={list(obj)}")
            else:
                print(f"  {key:50s} type={type(obj).__name__}")


# ---------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------

def ensure_event_first(array: np.ndarray, n_events: int) -> np.ndarray:
    """
    Move the event axis to the front when it can be identified uniquely.
    """
    array = np.asarray(array)

    if array.shape[0] == n_events:
        return array

    matching_axes = [
        axis for axis, size in enumerate(array.shape)
        if size == n_events
    ]

    if len(matching_axes) != 1:
        raise ValueError(
            f"Could not identify event axis in shape {array.shape}; "
            f"expected one axis of length {n_events}"
        )

    return np.moveaxis(array, matching_axes[0], 0)


def extract_reference_metrics(
    metrics_path: Path,
) -> dict[str, Any]:
    """
    Extract the nine clustering metrics from the released AgMetrics struct.

    The formulas follow BF1_saccadehistograms.m:

        amplitude = AllSacs(:,11,:) - AllSacs(:,4,:)
        max-median = AllSacs(:,7,:) - AllSacs(:,11,:)
        vergence = post_left - post_right
        velocities = [SacVel(:,:,1), SacVel(:,:,2)]

    MATLAB columns are converted to zero-based Python indexes.
    """
    data = load_mat_file(metrics_path)
    ag = find_struct_with_field(data, "AllSacs")

    all_sacs = np.asarray(get_field(ag, "AllSacs"), dtype=float)

    if all_sacs.ndim != 3:
        raise ValueError(
            f"Expected AllSacs to be 3-D, got shape {all_sacs.shape}"
        )

    # Expected orientation: events x metrics x eyes.
    possible_event_counts = all_sacs.shape
    if all_sacs.shape[1] < 11 or all_sacs.shape[2] != 2:
        raise ValueError(
            "Unexpected AllSacs shape. Expected events x >=11 metrics x 2 eyes, "
            f"got {all_sacs.shape}"
        )

    n_all = all_sacs.shape[0]

    in_bout = np.asarray(get_field(ag, "InBoutDex")).squeeze()
    if in_bout.size != n_all:
        raise ValueError(
            f"InBoutDex has {in_bout.size} entries but AllSacs has "
            f"{n_all} events"
        )

    # Initial UMAP was fitted to tethered events outside swimming bouts.
    keep_mask = ~in_bout.astype(bool)
    n_keep = int(keep_mask.sum())

    pre = all_sacs[:, 3, :]          # MATLAB column 4
    max_post = all_sacs[:, 6, :]     # MATLAB column 7
    median_post = all_sacs[:, 10, :] # MATLAB column 11

    amplitude = median_post - pre
    max_med = max_post - median_post
    vergence = median_post[:, 0] - median_post[:, 1]

    sac_vel = np.asarray(get_field(ag, "SacVel"), dtype=float)
    sac_vel = ensure_event_first(sac_vel, n_all)

    if sac_vel.ndim == 3:
        # MATLAB:
        # [AgMetrics.SacVel(:,:,1), AgMetrics.SacVel(:,:,2)]
        velocity = np.concatenate(
            [sac_vel[:, :, 0], sac_vel[:, :, 1]],
            axis=1,
        )
    elif sac_vel.ndim == 2:
        velocity = sac_vel
    else:
        raise ValueError(f"Unexpected SacVel shape: {sac_vel.shape}")

    if velocity.shape[1] < 4:
        raise ValueError(
            f"Expected at least four velocity columns, got {velocity.shape}"
        )

    # BF1_UMAPplots labels these as:
    # CCW L, CW L, CCW R, CW R.
    X_all = np.column_stack([
        amplitude[:, 0],
        amplitude[:, 1],
        vergence,
        max_med[:, 0],
        max_med[:, 1],
        velocity[:, 0],
        velocity[:, 1],
        velocity[:, 2],
        velocity[:, 3],
    ])

    fish_all = all_sacs[:, 0, 0]

    result: dict[str, Any] = {
        "agmetrics": ag,
        "n_all": n_all,
        "n_keep": n_keep,
        "keep_mask": keep_mask,
        "X_raw": X_all[keep_mask],
        "fish": fish_all[keep_mask],
    }

    # Optional event traces for Figure 1-like plots.
    for field, output_name in [
        ("SaccadeL", "L_traces"),
        ("SaccadeR", "R_traces"),
    ]:
        try:
            traces = np.asarray(get_field(ag, field), dtype=float)

            # Typically time x events in MATLAB.
            if traces.ndim == 2 and traces.shape[1] == n_all:
                traces = traces[:, keep_mask].T
            elif traces.ndim == 2 and traces.shape[0] == n_all:
                traces = traces[keep_mask]
            else:
                print(
                    f"[warn] Ignoring {field}: unexpected shape {traces.shape}"
                )
                continue

            result[output_name] = traces
        except KeyError:
            pass

    try:
        result["premov"] = int(np.asarray(get_field(ag, "premov")).squeeze())
    except (KeyError, ValueError):
        result["premov"] = None

    try:
        result["ti"] = float(np.asarray(get_field(ag, "ti")).squeeze())
    except (KeyError, ValueError):
        result["ti"] = None

    return result


# ---------------------------------------------------------------------
# Labels, embeddings and standardized data
# ---------------------------------------------------------------------

def select_reference_rows(
    array: np.ndarray,
    keep_mask: np.ndarray,
    description: str,
) -> np.ndarray:
    """
    Accept either an array containing all events or only non-bout events.
    """
    array = np.asarray(array)
    n_all = len(keep_mask)
    n_keep = int(keep_mask.sum())

    if array.ndim == 0:
        raise ValueError(f"{description} is scalar")

    if array.shape[0] == n_keep:
        return array

    if array.shape[0] == n_all:
        return array[keep_mask]

    # Labels loaded from MATLAB are often shaped 1 x N.
    if array.ndim == 2 and array.shape[1] in (n_all, n_keep):
        array = array.T
        if array.shape[0] == n_all:
            return array[keep_mask]
        return array

    raise ValueError(
        f"{description} has shape {array.shape}; expected first dimension "
        f"{n_all} (all events) or {n_keep} (non-bout events)"
    )


def load_reference_array(
    mat_path: Path,
    key: str,
    keep_mask: np.ndarray,
    description: str,
) -> np.ndarray:
    data = load_mat_file(mat_path)
    array = np.asarray(resolve_path(data, key))
    return select_reference_rows(array, keep_mask, description)


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def winsorize_zscore_per_fish(
    features: np.ndarray,
    fish_ids: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """
    Winsorize and z-score each feature separately within each animal.
    """
    features = np.asarray(features, dtype=float)
    fish_ids = np.asarray(fish_ids)

    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if len(features) != len(fish_ids):
        raise ValueError("features and fish_ids have different lengths")

    result = np.full_like(features, np.nan, dtype=float)

    for fish in pd.unique(fish_ids):
        mask = fish_ids == fish
        sub = features[mask].copy()

        lo = np.nanpercentile(sub, lower_percentile, axis=0)
        hi = np.nanpercentile(sub, upper_percentile, axis=0)
        sub = np.clip(sub, lo, hi)

        mean = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0, ddof=0)

        bad_sd = (~np.isfinite(sd)) | (sd == 0)
        sd[bad_sd] = 1.0

        result[mask] = (sub - mean) / sd

    return result


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def subsample_indices(n: int, maximum: int, seed: int = 0) -> np.ndarray:
    if n <= maximum:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=maximum, replace=False))


def plot_embedding_by_label(
    embedding: np.ndarray,
    labels: np.ndarray,
    output: Path,
    max_points: int = 100_000,
) -> None:
    idx = subsample_indices(len(embedding), max_points)
    emb = embedding[idx]
    lab = labels[idx]

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab10")

    for label in sorted(np.unique(lab)):
        mask = lab == label
        name = PAPER_LABEL_NAMES.get(int(label), f"cluster {label}")

        color = "lightgray" if label == 0 else cmap((int(label) - 1) % 10)

        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=2,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            color=color,
            label=f"{label}: {name}",
        )

    ax.set_title("Paper tethered events: Python UMAP with published labels")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=8,
        markerscale=3,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_by_features(
    embedding: np.ndarray,
    X_z: np.ndarray,
    output: Path,
    max_points: int = 60_000,
) -> None:
    idx = subsample_indices(len(embedding), max_points)

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    axes = axes.ravel()

    for column, (ax, name) in enumerate(zip(axes, PAPER_FEATURE_NAMES)):
        values = X_z[idx, column]
        values = np.clip(values, -2, 2)

        scatter = ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=values,
            cmap="coolwarm",
            vmin=-2,
            vmax=2,
            s=1,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(
        scatter,
        ax=axes.tolist(),
        label="Within-fish standardized value",
        fraction=0.025,
        pad=0.02,
    )
    fig.suptitle("Paper tethered UMAP colored by oculomotor metrics")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_amplitude_heatmaps(
    X_raw: np.ndarray,
    labels: np.ndarray,
    output: Path,
) -> None:
    amp_l = X_raw[:, PAPER_FEATURE_NAMES.index("Amp_L")]
    amp_r = X_raw[:, PAPER_FEATURE_NAMES.index("Amp_R")]

    cluster_ids = sorted(np.unique(labels))
    ncols = 3
    nrows = int(np.ceil((len(cluster_ids) + 1) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    edges = np.arange(-40, 41, 1.5)

    groups = [("All events", np.ones(len(labels), dtype=bool))]
    groups.extend(
        (
            f"{label}: {PAPER_LABEL_NAMES.get(int(label), 'unknown')}",
            labels == label,
        )
        for label in cluster_ids
    )

    for ax, (title, mask) in zip(axes, groups):
        ax.hist2d(
            amp_l[mask],
            amp_r[mask],
            bins=[edges, edges],
            cmap="hot",
            norm=LogNorm(),
        )
        ax.plot([-40, 40], [-40, 40], "w--", lw=0.8)
        ax.plot([-40, 40], [40, -40], "w--", lw=0.8)
        ax.axhline(0, color="white", lw=0.4)
        ax.axvline(0, color="white", lw=0.4)
        ax.set_title(f"{title}\n(n={mask.sum():,})", fontsize=9)
        ax.set_aspect("equal")

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    fig.supxlabel("Left-eye amplitude (degrees)")
    fig.supylabel("Right-eye amplitude (degrees)")
    fig.suptitle("Saccade amplitude distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_traces(
    L: np.ndarray,
    R: np.ndarray,
    labels: np.ndarray,
    output: Path,
    premov: int | None,
    ti: float | None,
    max_per_cluster: int = 500,
    seed: int = 0,
) -> None:
    if len(L) != len(labels) or len(R) != len(labels):
        raise ValueError("Trace and label arrays are not row-aligned")

    n_time = L.shape[1]

    if premov is None:
        premov = min(100, n_time // 3)

    if ti is None:
        ti = 0.002

    # Match the plotting code's approximate time convention.
    time_s = (np.arange(n_time) + 1 - premov) * ti

    baseline_end = min(max(premov, 1), n_time)
    L0 = L - np.nanmedian(L[:, :baseline_end], axis=1, keepdims=True)
    R0 = R - np.nanmedian(R[:, :baseline_end], axis=1, keepdims=True)

    cluster_ids = sorted(np.unique(labels))
    ncols = 3
    nrows = int(np.ceil(len(cluster_ids) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 3.4 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(seed)

    for ax, label in zip(axes, cluster_ids):
        rows = np.flatnonzero(labels == label)

        if len(rows) > max_per_cluster:
            selected = rng.choice(rows, max_per_cluster, replace=False)
        else:
            selected = rows

        for row in selected:
            ax.plot(time_s, L0[row], color="blue", alpha=0.015, lw=0.5)
            ax.plot(time_s, R0[row], color="red", alpha=0.015, lw=0.5)

        ax.plot(
            time_s,
            np.nanmedian(L0[rows], axis=0),
            color="navy",
            lw=2,
            label="Left",
        )
        ax.plot(
            time_s,
            np.nanmedian(R0[rows], axis=0),
            color="darkred",
            lw=2,
            label="Right",
        )

        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set_xlim(-0.05, 0.25)
        ax.set_ylim(-50, 50)
        ax.set_title(
            f"{label}: {PAPER_LABEL_NAMES.get(int(label), 'unknown')}\n"
            f"n={len(rows):,}",
            fontsize=9,
        )

    for ax in axes[len(cluster_ids):]:
        ax.set_visible(False)

    axes[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Time relative to onset (s)")
    fig.supylabel("Baseline-corrected eye position (degrees)")
    fig.suptitle("Published classes: event-aligned traces")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_reference_figures(model: dict[str, Any], figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)

    # Use the released MATLAB embedding for visual reproduction if it was
    # provided. Otherwise use the fitted Python embedding.
    display_embedding = model.get(
        "paper_embedding",
        model["reference_embedding"],
    )

    plot_embedding_by_label(
        display_embedding,
        model["reference_labels_all"],
        figdir / "reference_umap_labels.png",
    )

    plot_embedding_by_features(
        display_embedding,
        model["X_reference_z_all"],
        figdir / "reference_umap_features.png",
    )

    plot_amplitude_heatmaps(
        model["X_reference_raw_all"],
        model["reference_labels_all"],
        figdir / "reference_amplitude_heatmaps.png",
    )

    if "L_traces" in model and "R_traces" in model:
        plot_cluster_traces(
            model["L_traces"],
            model["R_traces"],
            model["reference_labels_all"],
            figdir / "reference_cluster_traces.png",
            model.get("premov"),
            model.get("ti"),
        )

    print(f"Saved reference figures to {figdir}")


# ---------------------------------------------------------------------
# Build reference
# ---------------------------------------------------------------------

def parse_integer_list(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",")]


def save_table(table: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        table.to_csv(path, index=False)
    else:
        table.to_parquet(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def build_reference(args: argparse.Namespace) -> None:
    extracted = extract_reference_metrics(args.metrics)

    X_raw = extracted["X_raw"]
    fish = extracted["fish"]
    keep_mask = extracted["keep_mask"]

    labels = load_reference_array(
        args.labels,
        args.labels_key,
        keep_mask,
        "labels",
    ).squeeze()

    if labels.ndim != 1:
        raise ValueError(f"Labels must be one-dimensional, got {labels.shape}")

    if len(labels) != len(X_raw):
        raise ValueError(
            f"Labels have {len(labels)} rows but metrics have {len(X_raw)}"
        )

    if not np.all(np.isfinite(labels)):
        raise ValueError("Published labels contain NaN/Inf")

    labels = labels.astype(int)

    # Use exact released standardized features if they are available.
    if args.z_key is not None:
        X_z = load_reference_array(
            args.labels,
            args.z_key,
            keep_mask,
            "standardized feature matrix",
        )
        X_z = np.asarray(X_z, dtype=float)

        if X_z.shape != X_raw.shape:
            raise ValueError(
                f"Released standardized data have shape {X_z.shape}, "
                f"expected {X_raw.shape}"
            )

        print(f"Using released standardized data: {args.z_key}")
    else:
        X_z = winsorize_zscore_per_fish(X_raw, fish)
        print("Reconstructed per-fish winsorization and z-scoring")

    finite = np.isfinite(X_z).all(axis=1)

    if not finite.all():
        print(
            f"[warn] Excluding {(~finite).sum():,} events with invalid features "
            "from the Python UMAP/reference-neighbor model"
        )

    print(
        f"Fitting Python UMAP to {finite.sum():,} valid paper tethered events..."
    )

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric="euclidean",
        random_state=args.random_state,
        transform_seed=args.random_state,
        low_memory=True,
        verbose=True,
    )

    python_embedding_valid = reducer.fit_transform(
        X_z[finite].astype(np.float32)
    )

    python_embedding_all = np.full((len(X_z), 2), np.nan, dtype=np.float32)
    python_embedding_all[finite] = python_embedding_valid

    excluded_labels = set(args.exclude_labels)

    classifier_mask = finite & ~np.isin(labels, list(excluded_labels))

    reference_embedding = python_embedding_all[classifier_mask]
    reference_labels = labels[classifier_mask]

    if len(reference_embedding) == 0:
        raise ValueError("No reference events remain after label exclusion")

    # Calibrate a Python-UMAP-specific distance rejection threshold.
    k_calibration = min(args.k_neighbors + 1, len(reference_embedding))

    nn = NearestNeighbors(
        n_neighbors=k_calibration,
        metric="euclidean",
        n_jobs=-1,
    ).fit(reference_embedding)

    distances, _ = nn.kneighbors(reference_embedding)

    # First neighbor is the reference point itself.
    if distances.shape[1] > 1:
        distances = distances[:, 1:]

    median_neighbor_distance = np.median(distances, axis=1)
    distance_cutoff = float(
        np.quantile(median_neighbor_distance, args.distance_quantile)
    )

    print(
        f"Reference-distance cutoff: {distance_cutoff:.4f} "
        f"(quantile={args.distance_quantile})"
    )

    model: dict[str, Any] = {
        "model_version": 1,
        "feature_names": PAPER_FEATURE_NAMES,
        "label_names": PAPER_LABEL_NAMES,
        "reducer": reducer,
        "reference_embedding": reference_embedding.astype(np.float32),
        "reference_labels": reference_labels.astype(np.int16),
        "reference_embedding_all": python_embedding_all,
        "reference_labels_all": labels.astype(np.int16),
        "X_reference_raw_all": X_raw.astype(np.float32),
        "X_reference_z_all": X_z.astype(np.float32),
        "fish_reference_all": fish,
        "valid_reference_mask": finite,
        "classifier_reference_mask": classifier_mask,
        "excluded_reference_labels": sorted(excluded_labels),
        "k_neighbors": args.k_neighbors,
        "distance_cutoff": distance_cutoff,
        "distance_quantile": args.distance_quantile,
        "preprocessing": {
            "winsorization_percentiles": [0.5, 99.5],
            "zscore": "within animal",
        },
        "umap_parameters": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "euclidean",
            "random_state": args.random_state,
        },
    }

    if args.embedding_key is not None:
        paper_embedding = load_reference_array(
            args.labels,
            args.embedding_key,
            keep_mask,
            "paper embedding",
        )
        paper_embedding = np.asarray(paper_embedding, dtype=float)

        if paper_embedding.shape != (len(X_raw), 2):
            raise ValueError(
                f"Paper embedding has shape {paper_embedding.shape}; "
                f"expected {(len(X_raw), 2)}"
            )

        model["paper_embedding"] = paper_embedding.astype(np.float32)
        print(f"Loaded released MATLAB embedding: {args.embedding_key}")

    for name in ("L_traces", "R_traces", "premov", "ti"):
        if name in extracted:
            value = extracted[name]
            if isinstance(value, np.ndarray):
                value = value.astype(np.float32)
            model[name] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output, compress=3)
    print(f"Saved reference model to {args.output}")

    table = pd.DataFrame(
        X_raw,
        columns=PAPER_FEATURE_NAMES,
    )
    table.insert(0, "fish", fish)
    table["published_cluster"] = labels
    table["published_name"] = [
        PAPER_LABEL_NAMES.get(int(label), f"cluster_{label}")
        for label in labels
    ]
    table["python_umap_x"] = python_embedding_all[:, 0]
    table["python_umap_y"] = python_embedding_all[:, 1]

    if "paper_embedding" in model:
        table["paper_umap_x"] = model["paper_embedding"][:, 0]
        table["paper_umap_y"] = model["paper_embedding"][:, 1]

    if args.table is not None:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        save_table(table, args.table)
        print(f"Saved extracted reference table to {args.table}")

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "metrics_file": str(args.metrics),
        "labels_file": str(args.labels),
        "labels_key": args.labels_key,
        "z_key": args.z_key,
        "embedding_key": args.embedding_key,
        "n_all_events": extracted["n_all"],
        "n_non_bout_events": extracted["n_keep"],
        "n_valid_reference_events": int(finite.sum()),
        "n_classifier_reference_events": int(classifier_mask.sum()),
        "feature_names": PAPER_FEATURE_NAMES,
        "label_names": PAPER_LABEL_NAMES,
        "excluded_reference_labels": sorted(excluded_labels),
        "distance_cutoff": distance_cutoff,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {metadata_path}")

    if args.figdir is not None:
        make_reference_figures(model, args.figdir)


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

def classify_events(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    model = joblib.load(args.model)
    events = read_table(args.events)

    feature_names = model["feature_names"]

    missing = [
        column for column in [args.fish_column, *feature_names]
        if column not in events.columns
    ]

    if missing:
        raise ValueError(f"Input event table is missing columns: {missing}")

    X_raw = events[feature_names].to_numpy(dtype=float)
    fish_ids = events[args.fish_column].to_numpy()

    X_z = winsorize_zscore_per_fish(X_raw, fish_ids)
    valid = np.isfinite(X_z).all(axis=1)

    embedding = np.full((len(events), 2), np.nan, dtype=float)
    assigned = np.full(len(events), -1, dtype=int)
    median_distance = np.full(len(events), np.nan, dtype=float)
    confidence = np.full(len(events), np.nan, dtype=float)

    if valid.any():
        print(f"Transforming {valid.sum():,} valid events...")
        transformed = model["reducer"].transform(
            X_z[valid].astype(np.float32)
        )
        embedding[valid] = transformed

        reference_embedding = model["reference_embedding"]
        reference_labels = model["reference_labels"]

        k = min(
            args.k_neighbors or model["k_neighbors"],
            len(reference_embedding),
        )

        neighbors = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean",
            n_jobs=-1,
        ).fit(reference_embedding)

        distances, indices = neighbors.kneighbors(transformed)
        local_median_distance = np.median(distances, axis=1)

        cutoff = (
            args.distance_cutoff
            if args.distance_cutoff is not None
            else model["distance_cutoff"]
        )

        valid_rows = np.flatnonzero(valid)

        for local_index, output_index in enumerate(valid_rows):
            median_distance[output_index] = local_median_distance[local_index]

            if local_median_distance[local_index] > cutoff:
                continue

            neighbor_labels = reference_labels[indices[local_index]]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            winner_index = int(np.argmax(counts))

            assigned[output_index] = int(values[winner_index])
            confidence[output_index] = counts[winner_index] / len(neighbor_labels)

    output = events.copy()
    output["embed_x"] = embedding[:, 0]
    output["embed_y"] = embedding[:, 1]
    output["cluster"] = assigned
    output["cluster_name"] = [
        PAPER_LABEL_NAMES.get(int(label), "Unassigned")
        if label >= 0 else "Unassigned"
        for label in assigned
    ]
    output["assignment_median_distance"] = median_distance
    output["assignment_confidence"] = confidence
    output["assignment_rejected"] = assigned == -1
    output["features_valid"] = valid

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_table(output, args.output)

    print(f"Saved classified events to {args.output}")
    print("\nClass distribution:")
    print(
        output[["cluster", "cluster_name"]]
        .value_counts(dropna=False)
        .sort_index()
    )
    print(f"\nRejected: {(assigned == -1).mean():.1%}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and use the paper's tethered saccade reference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect released MATLAB files",
    )
    inspect_parser.add_argument("mat_files", nargs="+", type=Path)

    build_parser_ = subparsers.add_parser(
        "build",
        help="Build a Python reference from the paper's tethered data",
    )
    build_parser_.add_argument(
        "--metrics",
        required=True,
        type=Path,
        help="AgMetrics MAT file containing AllSacs and SacVel",
    )
    build_parser_.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="MAT file containing final published labels",
    )
    build_parser_.add_argument(
        "--labels-key",
        required=True,
        help="Dotted path to final labels, e.g. Idx or results.Idx",
    )
    build_parser_.add_argument(
        "--embedding-key",
        default=None,
        help="Optional dotted path to released UMAP coordinates",
    )
    build_parser_.add_argument(
        "--z-key",
        default=None,
        help="Optional dotted path to released standardized 9-D data",
    )
    build_parser_.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output joblib reference model",
    )
    build_parser_.add_argument(
        "--table",
        type=Path,
        default=None,
        help="Optional extracted reference CSV/Parquet table",
    )
    build_parser_.add_argument(
        "--figdir",
        type=Path,
        default=None,
        help="Optional output directory for reference figures",
    )
    build_parser_.add_argument("--n-neighbors", type=int, default=199)
    build_parser_.add_argument("--min-dist", type=float, default=0.11)
    build_parser_.add_argument("--random-state", type=int, default=0)
    build_parser_.add_argument("--k-neighbors", type=int, default=100)
    build_parser_.add_argument(
        "--distance-quantile",
        type=float,
        default=0.995,
        help=(
            "Reference nearest-neighbor distance quantile used to reject "
            "out-of-distribution events"
        ),
    )
    build_parser_.add_argument(
        "--exclude-labels",
        type=parse_integer_list,
        default=[0],
        help=(
            "Comma-separated published labels excluded from the assignment "
            "reference. Default: 0 (unclassified)"
        ),
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate plots from a built reference model",
    )
    plot_parser.add_argument("--model", required=True, type=Path)
    plot_parser.add_argument("--figdir", required=True, type=Path)

    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify a new free-swimming event table",
    )
    classify_parser.add_argument("--model", required=True, type=Path)
    classify_parser.add_argument("--events", required=True, type=Path)
    classify_parser.add_argument("--output", required=True, type=Path)
    classify_parser.add_argument("--fish-column", default="fish")
    classify_parser.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
        help="Override the model's default number of neighbors",
    )
    classify_parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=None,
        help="Override the model's calibrated rejection distance",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "inspect":
        inspect_mat_files(args.mat_files)
    elif args.command == "build":
        build_reference(args)
    elif args.command == "plot":
        model = joblib.load(args.model)
        make_reference_figures(model, args.figdir)
    elif args.command == "classify":
        classify_events(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()