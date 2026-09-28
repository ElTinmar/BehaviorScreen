"""
saccade_cli.py
==============
Minimal CLI: load each fish with the existing BehaviorScreen.load machinery,
compute raw eye angles from the eyes_tracking keypoints, and run them
through saccade_pipeline (coarse detection -> pairing -> onset refinement ->
metrics -> UMAP/DBSCAN clustering).

No eye-preprocessing is done here (no likelihood filtering, no clipping,
no interpolation/smoothing) -- saccade_pipeline resamples to 100/500 Hz
and smooths internally.

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
from typing import Optional, Tuple

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
# Per-fish pipeline (stages 1-5 + BConv flag)
# ----------------------------------------------------------------------

def process_fish(fish_label: str, t: np.ndarray, L_raw: np.ndarray, R_raw: np.ndarray,
                  mode: str = "tethered") -> pd.DataFrame:

    coarse = sp.coarse_detect_events(t, L_raw, R_raw, fs=100.0)
    bino_times, _, _ = sp.pair_binocular_events(coarse["L_events"]["t"], coarse["R_events"]["t"])
    bino_times = sp.discard_overlapping_events(bino_times, refractory_s=0.3)
    if len(bino_times) == 0:
        return pd.DataFrame()

    t500, L500 = sp.interp_to_rate(t, L_raw, fs=500.0)
    _, R500 = sp.interp_to_rate(t, R_raw, fs=500.0)
    L_smooth = sp.smooth_trace_for_metrics(L500, mode=mode)
    R_smooth = sp.smooth_trace_for_metrics(R500, mode=mode)

    records = []
    for ev_time in bino_times:
        coarse_idx = int(np.clip(round(ev_time * 500.0), 0, len(L_smooth) - 1))
        onset_idx = sp.refine_onset_time(L_smooth, 500.0, coarse_idx)

        Lm = sp.event_position_velocity_metrics(L_smooth, 500.0, onset_idx)
        Rm = sp.event_position_velocity_metrics(R_smooth, 500.0, onset_idx)
        feats = sp.compute_9_metrics(Lm, Rm)

        is_bconv, details = sp.detect_biphasic_convergent(
            L_smooth, R_smooth, 500.0, onset_idx, is_tethered=(mode == "tethered")
        )
        bconv_side = None
        if is_bconv:
            vL = details["L"]["max_vel"] or -np.inf
            vR = details["R"]["max_vel"] or -np.inf
            bconv_side = "L" if (vL if not np.isnan(vL) else -np.inf) >= (vR if not np.isnan(vR) else -np.inf) else "R"

        rec = dict(fish=fish_label, onset_time_s=onset_idx / 500.0,
                   bconv_flag=is_bconv, bconv_side=bconv_side)
        rec.update(zip(sp.METRIC_NAMES, feats))
        records.append(rec)

    return pd.DataFrame.from_records(records)


def collect_all_events(directories: Directories, mode: str) -> pd.DataFrame:
    behavior_files = find_files(directories)
    print(f"Found {len(behavior_files)} experiments")

    all_events = []
    for files in tqdm(behavior_files, desc="Fish"):
        fish_label = files.metadata.stem
        behavior_data = load_data(files)

        if behavior_data.eyes_tracking.empty or behavior_data.video_timestamps.empty:
            print(f"[skip] {fish_label}: no eye tracking / timestamps")
            continue

        t, L_raw, R_raw = extract_raw_eye_angles(behavior_data)
        events = process_fish(fish_label, t, L_raw, R_raw, mode=mode)
        if not events.empty:
            all_events.append(events)
            print(f"[ok]   {fish_label}: {len(events)} events")

    return pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()


# ----------------------------------------------------------------------
# Clustering (fit or transform-only)
# ----------------------------------------------------------------------

def run_clustering(events: pd.DataFrame, load_model_path: Optional[Path],
                    save_model_path: Optional[Path]) -> pd.DataFrame:

    features = events[sp.METRIC_NAMES].to_numpy(dtype=float)
    feats_z = sp.winsorize_zscore_per_fish(features, events["fish"].to_numpy())

    events = events.copy()

    if load_model_path is not None and load_model_path.exists():
        model = joblib.load(load_model_path)
        labels, embedding = sp.assign_heldout_events(
            model["reducer"], model["embedding"], model["labels"], feats_z
        )
    else:
        reducer, embedding = sp.fit_umap_embedding(feats_z)
        labels = sp.run_dbscan(embedding, eps=0.34, min_samples=570)
        labels = sp.reassign_border_points(embedding, labels)
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
    parser = argparse.ArgumentParser(description="Detect and classify saccades from eye-tracking data.")
    parser.add_argument("root", type=Path, help="Root experiment folder")
    parser.add_argument("--output", type=Path, default=Path("saccades.csv"))
    parser.add_argument("--mode", choices=["tethered", "freeswim"], default="tethered")

    parser.add_argument("--metadata", default="data")
    parser.add_argument("--stimuli", default="data")
    parser.add_argument("--tracking", default="data")
    parser.add_argument("--lightning-pose", default="lightning_pose")
    parser.add_argument("--temperature", default="data")
    parser.add_argument("--video", default="video")
    parser.add_argument("--video-timestamp", default="video")
    parser.add_argument("--results", default="results")
    parser.add_argument("--plots", default="plots")

    parser.add_argument("--load-clustering-model", type=Path, default=None)
    parser.add_argument("--save-clustering-model", type=Path, default=None)

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

    events = collect_all_events(directories, mode=args.mode)
    print(f"Detected {len(events)} events across {events['fish'].nunique() if not events.empty else 0} fish")

    if events.empty:
        return

    events = run_clustering(events, args.load_clustering_model, args.save_clustering_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False)
    print(f"Saved {len(events)} events to {args.output}")
    print(events["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main(build_parser().parse_args())