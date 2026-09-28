"""
Minimal CLI: load each fish with the existing BehaviorScreen.load machinery,
compute raw eye angles from the eyes_tracking keypoints, and run them
through saccade_pipeline (coarse detection -> pairing -> onset refinement ->
metrics -> UMAP/DBSCAN clustering).

No eye-preprocessing is done here (no likelihood filtering, no clipping,
no interpolation/smoothing) -- saccade_pipeline resamples to 100/500 Hz
and smooths internally.

Outputs
-------
- <output>.csv : one row per detected binocular saccade event, with the
  9 oculomotor metrics, BConv flag, and cluster label.
- <output>.npz : companion file with raw + smoothed eye-position trace
  snippets around each event's onset (same row order as the CSV), for
  the cluster sanity-check plots in plot_clusters.py.

Usage
-----
    # fit a new clustering model
    python saccade_cli.py /data/WT_oct_2025 --output saccades.csv \
        --save-clustering-model model.joblib

    # reuse an existing model (transform only)
    python saccade_cli.py /data/WT_nov_2025 --output saccades_nov.csv \
        --load-clustering-model model.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.load import Directories, find_files, load_data, BehaviorData
from BehaviorScreen.process import compute_angle_between_vectors

import BehaviorScreen.eyes.saccade_pipeline as sp


# ----------------------------------------------------------------------
# Raw eye angles (no filtering / interpolation / smoothing)
# ----------------------------------------------------------------------

def extract_raw_eye_angles(behavior_data: BehaviorData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes raw L/R eye angle (deg) from eyes_tracking keypoints and
    a seconds-timebase from video_timestamps. No QC/interpolation/smoothing
    -- saccade_pipeline resamples & smooths downstream.
    """
    left_vector = (
        behavior_data.eyes_tracking.eye_left_back[["x", "y"]].to_numpy()
        - behavior_data.eyes_tracking.eye_left_front[["x", "y"]].to_numpy()
    )
    right_vector = (
        behavior_data.eyes_tracking.eye_right_back[["x", "y"]].to_numpy()
        - behavior_data.eyes_tracking.eye_right_front[["x", "y"]].to_numpy()
    )

    L = np.rad2deg(compute_angle_between_vectors(left_vector, np.array([0, 1])))
    R = np.rad2deg(compute_angle_between_vectors(right_vector, np.array([0, 1])))

    ts_ns = behavior_data.video_timestamps.timestamp.to_numpy()
    n = min(len(ts_ns), len(L), len(R))
    t = (ts_ns[:n].astype(np.float64) - ts_ns[0]) * 1e-9

    return t, L[:n], R[:n]


# ----------------------------------------------------------------------
# Trace snippet extraction (for cluster sanity-check plots)
# ----------------------------------------------------------------------

def extract_event_snippet(trace: np.ndarray, onset_idx: int, fs: float,
                           pre_ms: float = 300, post_ms: float = 600) -> np.ndarray:
    """
    Extracts a fixed-length window around `onset_idx`, padding with NaN
    if the window runs off the start/end of the trace.
    """
    pre_n = int(round(pre_ms / 1000 * fs))
    post_n = int(round(post_ms / 1000 * fs))
    n = len(trace)

    snippet = np.full(pre_n + post_n, np.nan)
    lo, hi = onset_idx - pre_n, onset_idx + post_n
    src_lo, src_hi = max(0, lo), min(n, hi)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    snippet[dst_lo:dst_hi] = trace[src_lo:src_hi]
    return snippet


# ----------------------------------------------------------------------
# Per-fish pipeline (stages 1-5 + BConv flag)
# ----------------------------------------------------------------------

def process_fish(
    fish_label: str,
    t: np.ndarray,
    L_raw: np.ndarray,
    R_raw: np.ndarray,
    mode: str = "tethered",
    snippet_pre_ms: float = 300,
    snippet_post_ms: float = 600,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs stages 1-5 (+ BConv flag) of the pipeline for a single fish.

    Returns
    -------
    events_df : one row per detected event, 9 metrics + BConv flag
    L_raw_snips, R_raw_snips : (n_events, snippet_len) arrays, 500 Hz
        resampled but PRE-smoothing eye position around each onset
    L_smooth_snips, R_smooth_snips : same, but POST custom-LOWESS smoothing
        (i.e. what actually fed into the 9 metrics)
    """
    empty = pd.DataFrame(), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))

    # Stage 1: coarse detection @ 100 Hz
    coarse = sp.coarse_detect_events(t, L_raw, R_raw, fs=100.0)

    # Stage 2: binocular pairing + 300 ms refractory exclusion
    bino_times, _, _ = sp.pair_binocular_events(coarse["L_events"]["t"], coarse["R_events"]["t"])
    bino_times = sp.discard_overlapping_events(bino_times, refractory_s=0.3)

    if len(bino_times) == 0:
        return empty

    # Stage 3: resample to 500 Hz + custom LOWESS smoothing
    t500, L500 = sp.interp_to_rate(t, L_raw, fs=500.0)
    _, R500 = sp.interp_to_rate(t, R_raw, fs=500.0)
    L_smooth = sp.smooth_trace_for_metrics(L500, mode=mode)
    R_smooth = sp.smooth_trace_for_metrics(R500, mode=mode)

    records = []
    L_raw_snips, R_raw_snips = [], []
    L_smooth_snips, R_smooth_snips = [], []

    for ev_time in bino_times:
        coarse_idx = int(np.clip(round(ev_time * 500.0), 0, len(L_smooth) - 1))

        # Stage 4: refined onset
        onset_idx = sp.refine_onset_time(L_smooth, 500.0, coarse_idx)

        # Stage 5: 9 oculomotor metrics
        Lm = sp.event_position_velocity_metrics(L_smooth, 500.0, onset_idx)
        Rm = sp.event_position_velocity_metrics(R_smooth, 500.0, onset_idx)
        feats = sp.compute_9_metrics(Lm, Rm)

        # BConv flag (precursor to stage 8, computed here while traces are in memory)
        is_bconv, details = sp.detect_biphasic_convergent(
            L_smooth, R_smooth, 500.0, onset_idx, is_tethered=(mode == "tethered")
        )
        bconv_side = None
        if is_bconv:
            vL = details["L"]["max_vel"]
            vR = details["R"]["max_vel"]
            vL = -np.inf if vL is None or np.isnan(vL) else vL
            vR = -np.inf if vR is None or np.isnan(vR) else vR
            bconv_side = "L" if vL >= vR else "R"

        rec = dict(
            fish=fish_label,
            onset_time_s=onset_idx / 500.0,
            bconv_flag=is_bconv,
            bconv_side=bconv_side,
        )
        rec.update(zip(sp.METRIC_NAMES, feats))
        records.append(rec)

        L_raw_snips.append(extract_event_snippet(L500, onset_idx, 500.0, snippet_pre_ms, snippet_post_ms))
        R_raw_snips.append(extract_event_snippet(R500, onset_idx, 500.0, snippet_pre_ms, snippet_post_ms))
        L_smooth_snips.append(extract_event_snippet(L_smooth, onset_idx, 500.0, snippet_pre_ms, snippet_post_ms))
        R_smooth_snips.append(extract_event_snippet(R_smooth, onset_idx, 500.0, snippet_pre_ms, snippet_post_ms))

    events_df = pd.DataFrame.from_records(records)
    return (events_df, np.array(L_raw_snips), np.array(R_raw_snips),
            np.array(L_smooth_snips), np.array(R_smooth_snips))


def collect_all_events(
    directories: Directories,
    mode: str,
    snippet_pre_ms: float = 300,
    snippet_post_ms: float = 600,
) -> Tuple[pd.DataFrame, Optional[Dict[str, np.ndarray]]]:
    """
    Iterates over all fish found under `directories`, runs process_fish
    on each, and stacks results. Returns (events_df, snippets_dict).
    snippets_dict is None if no events were found anywhere.
    """
    behavior_files = find_files(directories)
    print(f"Found {len(behavior_files)} experiments")

    all_events, all_L_raw, all_R_raw, all_L_smooth, all_R_smooth = [], [], [], [], []

    for files in tqdm(behavior_files, desc="Fish"):
        fish_label = files.metadata.stem
        behavior_data = load_data(files)

        if behavior_data.eyes_tracking.empty or behavior_data.video_timestamps.empty:
            print(f"[skip] {fish_label}: no eye tracking / timestamps")
            continue

        t, L_raw, R_raw = extract_raw_eye_angles(behavior_data)

        events, L_r, R_r, L_s, R_s = process_fish(
            fish_label, t, L_raw, R_raw, mode=mode,
            snippet_pre_ms=snippet_pre_ms, snippet_post_ms=snippet_post_ms,
        )

        if not events.empty:
            all_events.append(events)
            all_L_raw.append(L_r)
            all_R_raw.append(R_r)
            all_L_smooth.append(L_s)
            all_R_smooth.append(R_s)
            print(f"[ok]   {fish_label}: {len(events)} events")

    if not all_events:
        return pd.DataFrame(), None

    events = pd.concat(all_events, ignore_index=True)

    pre_n = int(round(snippet_pre_ms / 1000 * 500.0))
    post_n = int(round(snippet_post_ms / 1000 * 500.0))
    time_axis_ms = np.arange(-pre_n, post_n) / 500.0 * 1000.0

    snippets = dict(
        L_raw=np.concatenate(all_L_raw, axis=0),
        R_raw=np.concatenate(all_R_raw, axis=0),
        L_smooth=np.concatenate(all_L_smooth, axis=0),
        R_smooth=np.concatenate(all_R_smooth, axis=0),
        time_axis_ms=time_axis_ms,
    )
    return events, snippets


# ----------------------------------------------------------------------
# Clustering (fit or transform-only)
# ----------------------------------------------------------------------

def run_clustering(
    events: pd.DataFrame,
    load_model_path: Optional[Path],
    save_model_path: Optional[Path],
) -> pd.DataFrame:

    features = events[sp.METRIC_NAMES].to_numpy(dtype=float)
    feats_z = sp.winsorize_zscore_per_fish(features, events["fish"].to_numpy())

    events = events.copy()

    if load_model_path is not None and load_model_path.exists():
        print(f"Loading pretrained clustering model: {load_model_path}")
        model = joblib.load(load_model_path)
        labels, embedding = sp.assign_heldout_events(
            model["reducer"], model["embedding"], model["labels"], feats_z
        )
    else:
        print(f"Fitting UMAP on {len(feats_z)} events...")
        reducer, embedding = sp.fit_umap_embedding(feats_z)

        print("Running DBSCAN...")
        labels = sp.run_dbscan(embedding, eps=0.34, min_samples=570)
        labels = sp.reassign_border_points(embedding, labels)

        n_clusters = len(set(labels) - {-1})
        unclustered_frac = np.mean(labels == -1)
        print(f"Found {n_clusters} clusters ({unclustered_frac:.1%} unclustered)")

        if save_model_path is not None:
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(dict(reducer=reducer, embedding=embedding, labels=labels), save_model_path)
            print(f"Saved clustering model to {save_model_path}")

    events["cluster"] = labels
    events["embed_x"] = embedding[:, 0]
    events["embed_y"] = embedding[:, 1]
    return events


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and classify saccades from eye-tracking data "
                     "(coarse detection -> pairing -> LOWESS/UMAP/DBSCAN)."
    )
    parser.add_argument("root", type=Path, help="Root experiment folder")
    parser.add_argument("--output", type=Path, default=Path("saccades.csv"),
                         help="Output CSV path; a companion .npz with trace "
                              "snippets is saved alongside it")
    parser.add_argument("--mode", choices=["tethered", "freeswim"], default="freeswim",
                         help="Selects LOWESS smoothing spans + BConv velocity threshold")

    # Directory layout overrides (mirrors your existing tool's conventions)
    parser.add_argument("--metadata", default="results")
    parser.add_argument("--stimuli", default="results")
    parser.add_argument("--tracking", default="results")
    parser.add_argument("--lightning-pose", default="lightning_pose",
                         help="Subfolder with lightning-pose CSVs (used for both "
                              "full_tracking and eyes_tracking)")
    parser.add_argument("--temperature", default="results")
    parser.add_argument("--video", default="results")
    parser.add_argument("--video-timestamp", default="results")
    parser.add_argument("--results", default="results")
    parser.add_argument("--plots", default="plots")

    # Trace snippet window (for sanity-check plots)
    parser.add_argument("--snippet-pre-ms", type=float, default=300,
                         help="Window before onset to save for plotting (ms)")
    parser.add_argument("--snippet-post-ms", type=float, default=600,
                         help="Window after onset to save for plotting (ms)")

    # Clustering model persistence
    parser.add_argument("--load-clustering-model", type=Path, default=None,
                         help="Path to a previously-fit joblib model. If given, "
                              "events are only *transformed*, not used to refit.")
    parser.add_argument("--save-clustering-model", type=Path, default=None,
                         help="Where to save a freshly-fit clustering model")

    return parser


def main(args: argparse.Namespace) -> None:
    directories = Directories(
        args.root,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        full_tracking=args.lightning_pose,
        eyes_tracking=args.lightning_pose,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
    )

    events, snippets = collect_all_events(
        directories, mode=args.mode,
        snippet_pre_ms=args.snippet_pre_ms, snippet_post_ms=args.snippet_post_ms,
    )
    n_fish = events["fish"].nunique() if not events.empty else 0
    print(f"Detected {len(events)} binocular saccade events across {n_fish} fish")

    if events.empty:
        print("No events detected -- nothing to cluster/save.")
        return

    events = run_clustering(events, args.load_clustering_model, args.save_clustering_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False)
    print(f"Saved {len(events)} events to {args.output}")

    npz_path = args.output.with_suffix(".npz")
    np.savez(npz_path, **snippets)
    print(f"Saved raw/smoothed trace snippets to {npz_path}")

    print("\nCluster distribution:")
    print(events["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main(build_parser().parse_args())