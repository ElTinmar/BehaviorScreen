# remove fish with bad online tracking
# remove fish that did not move
# remove empty wells
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.load import Directories, find_files
from BehaviorScreen.process import get_epoch_trial_presentation
from BehaviorScreen.recording import Recording


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Run megabout pipeline on tracking data from Lightning Pose"
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder (e.g. WT_oct_2025)",
    )

    parser.add_argument(
        "--qc-csv",
        default="qc.csv",
        help="Output CSV file (fish-level QC exclusions)",
    )

    parser.add_argument(
        "--valid-trials-csv",
        default="valid_trials.csv",
        help="Output CSV file (per-trial presentation + tracking quality record)",
    )

    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
        help="Minimum fraction of frames within a trial where online tracking "
        "agrees with offline tracking, for that trial to count as "
        "'tracking_ok' (default: 0.8)",
    )

    parser.add_argument(
        "--trial-centroid-threshold-mm",
        type=float,
        default=1.0,
        help="Per-frame online/offline centroid mismatch (mm) above which a "
        "frame is considered mismatched, for trial coverage purposes "
        "(default: 1.0, same value used by the session-level check)",
    )

    parser.add_argument(
        "--trial-heading-threshold-deg",
        type=float,
        default=15.0,
        help="Per-frame online/offline heading mismatch (deg) above which a "
        "frame is considered mismatched, for trial coverage purposes "
        "(default: 15.0, same value used by the session-level check)",
    )

    # Directory layout overrides
    parser.add_argument(
        "--metadata",
        default="results",
        help="Subfolder containing metadata files (default: data)",
    )

    parser.add_argument(
        "--stimuli",
        default="results",
        help="Subfolder containing stimulus log files (default: data)",
    )

    parser.add_argument(
        "--tracking",
        default="results",
        help="Subfolder containing tracking CSV files (default: data)",
    )

    parser.add_argument(
        "--lightning-pose",
        default="lightning_pose",
        help="Subfolder containing lightning pose tracking CSV files (default: lightning_pose)",
    )

    parser.add_argument(
        "--temperature",
        default="results",
        help="Subfolder containing temperature logs (default: data)",
    )

    parser.add_argument(
        "--video",
        default="results",
        help="Subfolder containing raw video files (default: video)",
    )

    parser.add_argument(
        "--video-timestamp",
        default="results",
        help="Subfolder containing video timestamp files (default: video)",
    )

    parser.add_argument(
        "--results",
        default="results",
        help="Subfolder where per-animal exports will be written (default: results)",
    )

    parser.add_argument(
        "--plots",
        default="plots",
        help="Subfolder containing plots (default: plots)",
    )

    return parser


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unsigned angle (deg) between rows of u and v, both (n, 2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))


def get_per_frame_tracking_error(rec: Recording) -> pd.DataFrame:
    """
    Online-vs-offline tracking discrepancy for every pose frame; NaN where
    the online tracker has no row. Offline (Lightning Pose) is ground truth:
    megabouts runs on it, online tracking only drives acquisition.
    """
    n = len(rec.pose)
    online = rec.online.iloc[:n]
    swim_bladder = rec.pose.Swim_Bladder[["x", "y"]].to_numpy(float)
    head = rec.pose.Head[["x", "y"]].to_numpy(float)
    online_centroid = online[["centroid_x", "centroid_y"]].to_numpy(float)
    online_heading = online[["pc1_x", "pc1_y"]].to_numpy(float)

    return pd.DataFrame(
        {
            "centroid_distance_mm": np.linalg.norm(swim_bladder - online_centroid, axis=1)
            / rec.pix_per_mm,
            "heading_angle_deg": angle_between(online_heading, head - swim_bladder),
        }
    )


def get_average_speed(rec: Recording, epoch_name: str = "spontaneous dark") -> float:
    # Summing every frame-to-frame step also sums keypoint jitter: the noise
    # floor is ~1.1 * jitter_px * fps / pix_per_mm, which grows with fps and
    # can exceed the not-moving threshold. If immobile fish stop being flagged,
    # alternatives:
    #   - subsample positions (e.g. every 1 s) before diff: noise becomes
    #     jitter_px / (interval * pix_per_mm), but loops within an interval
    #     are missed (net displacement, not path length)
    #   - low-pass positions (~50-100 ms rolling mean / savgol) before diff
    #   - use megabouts bout rate during spontaneous dark (QC would need to
    #     run after megabouts)

    epoch = rec.trials[rec.trials.epoch_name == epoch_name]
    if epoch.empty:
        return np.nan

    start, stop = int(epoch.start_frame.min()), int(epoch.stop_frame.max())
    duration_s = (rec.frame_time_ns[stop] - rec.frame_time_ns[start]) * 1e-9
    centroid = rec.pose.Swim_Bladder[["x", "y"]].to_numpy(float)[start:stop]
    distance_px = np.sum(np.linalg.norm(np.diff(centroid, axis=0), axis=1))
    return distance_px / (duration_s * rec.pix_per_mm)


def is_online_tracking_bad(
    per_frame_error: pd.DataFrame,
    centroid_threshold_mm_per_frame: float = 1,
    heading_threshold_deg_per_frame: float = 15,
) -> tuple[bool, bool]:
    """Whole-session mean online/offline mismatch above threshold."""
    return (
        bool(np.nanmean(per_frame_error["centroid_distance_mm"]) >= centroid_threshold_mm_per_frame),
        bool(np.nanmean(per_frame_error["heading_angle_deg"]) >= heading_threshold_deg_per_frame),
    )


def is_fish_not_moving(rec: Recording, speed_threshold_mm_per_sec: float = 0.2) -> bool:
    # False if the dark epoch is not found
    return bool(get_average_speed(rec) < speed_threshold_mm_per_sec)


def get_trial_tracking_coverage(
    trials: pd.DataFrame,
    per_frame_error: pd.DataFrame,
    centroid_threshold_mm_per_frame: float = 1.0,
    heading_threshold_deg_per_frame: float = 15.0,
) -> np.ndarray:
    """
    Per trial, the fraction of frames in [start_frame, stop_frame) where online
    and offline tracking agree. Frames without online tracking count as
    disagreeing. NaN for empty trials.
    """
    agrees = (
        (per_frame_error["centroid_distance_mm"] < centroid_threshold_mm_per_frame)
        & (per_frame_error["heading_angle_deg"] < heading_threshold_deg_per_frame)
    ).to_numpy()
    cumulative = np.concatenate(([0], np.cumsum(agrees)))
    start = np.clip(trials["start_frame"].to_numpy(), 0, len(agrees))
    stop = np.clip(trials["stop_frame"].to_numpy(), 0, len(agrees))
    n_frames = (trials["stop_frame"] - trials["start_frame"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_frames > 0, (cumulative[stop] - cumulative[start]) / n_frames, np.nan)


def quality_control(
    root: Path,
    output_csv: str,
    valid_trials_csv: str,
    metadata: str,
    stimuli: str,
    tracking: str,
    lightning_pose: str,
    temperature: str,
    video: str,
    video_timestamp: str,
    results: str,
    plots: str,
    coverage_threshold: float = 0.8,
    trial_centroid_threshold_mm: float = 1.0,
    trial_heading_threshold_deg: float = 15.0,
) -> None:
    """
    Writes the fish-level exclusions (qc.csv) and the per-trial record
    (valid_trials.csv): whether each expected trial was presented and
    whether online tracking agreed with offline tracking during it.
    """
    directories = Directories(
        root,
        metadata=metadata,
        stimuli=stimuli,
        tracking=tracking,
        full_tracking=lightning_pose,
        temperature=temperature,
        video=video,
        video_timestamp=video_timestamp,
        results=results,
        plots=plots,
    )

    bad_fish = []
    valid_trial_tables = []

    for behavior_file in tqdm(find_files(directories)):
        rec = Recording(behavior_file)
        per_frame_error = get_per_frame_tracking_error(rec)

        not_moving = is_fish_not_moving(rec)
        centroid_issue, heading_issue = is_online_tracking_bad(per_frame_error)
        well_not_found = rec.well is None
        if not_moving or centroid_issue or heading_issue or well_not_found:
            bad_fish.append(
                (rec.name, not_moving, centroid_issue, heading_issue, well_not_found)
            )

        trials = rec.trials.assign(
            tracking_coverage=get_trial_tracking_coverage(
                rec.trials,
                per_frame_error,
                trial_centroid_threshold_mm,
                trial_heading_threshold_deg,
            )
        )
        trials["tracking_ok"] = trials["tracking_coverage"] >= coverage_threshold

        valid = get_epoch_trial_presentation(trials)
        # expected but never logged trials have no tracking_ok
        valid["tracking_ok"] = valid["tracking_ok"].astype("boolean").fillna(False).astype(bool)
        valid["file"] = rec.name
        valid_trial_tables.append(valid)

    header = ["file", "not_moving", "centroid_issue", "heading_issue", "well_not_found"]
    pd.DataFrame(bad_fish, columns=header).to_csv(root / output_csv, index=False)
    pd.concat(valid_trial_tables, ignore_index=True).to_csv(
        root / valid_trials_csv, index=False
    )


def main(args: argparse.Namespace) -> None:
    quality_control(
        root=args.root,
        output_csv=args.qc_csv,
        valid_trials_csv=args.valid_trials_csv,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        lightning_pose=args.lightning_pose,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
        coverage_threshold=args.coverage_threshold,
        trial_centroid_threshold_mm=args.trial_centroid_threshold_mm,
        trial_heading_threshold_deg=args.trial_heading_threshold_deg,
    )


if __name__ == "__main__":

    main(build_parser().parse_args())
