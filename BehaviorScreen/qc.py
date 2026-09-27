# remove fish with bad online tracking
# remove fish that did not move
# remove empty wells
from pathlib import Path
import argparse
from typing import Optional
from tqdm import tqdm
from BehaviorScreen.load import (
    Directories,
    BehaviorData,
    find_files,
    load_data
)
from BehaviorScreen.process import timestamp_to_frame, get_trials, get_epoch_trial_presentation
from BehaviorScreen.core import Stim
import pandas as pd
import numpy as np


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
        default='qc.csv',
        help="Output CSV file (fish-level QC exclusions)",
    )

    parser.add_argument(
        "--valid-trials-csv",
        default='valid_trials.csv',
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


def get_dark_epoch(behavior_data: BehaviorData) -> tuple[int, int]:

    res = (-1, -1)

    for i in range(len(behavior_data.stimuli) - 10):

        is_sequence_start = all(
            behavior_data.stimuli[i + j].get('stim_select') == Stim.DARK
            for j in range(10)
        )

        if is_sequence_start:
            start_ts = behavior_data.stimuli[i].get('timestamp')
            stop_ts = behavior_data.stimuli[i + 10].get('timestamp')
            start_frame = timestamp_to_frame(behavior_data, start_ts)
            stop_frame = timestamp_to_frame(behavior_data, stop_ts)
            res = (start_frame, stop_frame)

    return res


def angle_between(u, v):
    # u and v are shape (n, 2)
    norm_u = np.linalg.norm(u, axis=1, keepdims=True)
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_u = u / norm_u
        unit_v = v / norm_v
    dot_product = np.sum(unit_u * unit_v, axis=1)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.rad2deg(angle)


def get_per_frame_tracking_error(behavior_data: BehaviorData) -> pd.DataFrame:
    """
    Per-frame online-vs-offline tracking discrepancy, indexed by frame index.
    Offline (full_tracking / lightning pose) is treated as ground truth
    throughout this module -- megabouts' bout detection runs directly on
    full_tracking, so this measures whether ONLINE tracking (used for
    real-time ROI/ID assignment during acquisition) agrees with it.

    This is the single source of per-frame numbers shared by:
      - get_tracking_error / is_online_tracking_bad: whole-session average,
        one flag per fish.
      - get_trial_tracking_coverage: fraction of "agreeing" frames within a
        single trial window.

    Computed ONCE per fish and passed into both, rather than recomputed.
    """
    online = behavior_data.tracking.set_index('index')
    offline = behavior_data.full_tracking.copy()
    offline.columns = [f"{level0}_{level1}" if level1 else level0
                        for level0, level1 in offline.columns]
    # full_tracking's row position IS the frame index (megabouts reads it
    # positionally) -- make that explicit rather than relying on whatever
    # index full_tracking happens to already have.
    offline.index = np.arange(len(offline))

    common = online.join(offline, how='inner')
    if common.empty:
        return pd.DataFrame(columns=['centroid_distance_mm', 'heading_angle_deg'])

    pix_per_mm = behavior_data.metadata['calibration']['pix_per_mm']

    online_centroid = np.column_stack((common.centroid_x, common.centroid_y))
    offline_centroid = np.column_stack((common.Swim_Bladder_x, common.Swim_Bladder_y))
    centroid_distance_mm = (
        np.linalg.norm(offline_centroid - online_centroid, axis=1) / pix_per_mm
    )

    online_heading = np.column_stack((common.pc1_x, common.pc1_y))
    head = np.column_stack((common.Head_x, common.Head_y))
    sb = np.column_stack((common.Swim_Bladder_x, common.Swim_Bladder_y))
    offline_heading = head - sb
    heading_angle_deg = angle_between(online_heading, offline_heading)

    return pd.DataFrame(
        {
            'centroid_distance_mm': centroid_distance_mm,
            'heading_angle_deg': heading_angle_deg,
        },
        index=common.index,
    )


def get_tracking_error(
        behavior_data: BehaviorData,
        per_frame_error: Optional[pd.DataFrame] = None,
    ) -> tuple[float, float]:
    """Whole-session mean per-frame online/offline mismatch. Delegates to
    get_per_frame_tracking_error so the trial-level check can reuse the
    same per-frame numbers instead of recomputing them."""

    if per_frame_error is None:
        per_frame_error = get_per_frame_tracking_error(behavior_data)

    n_frames = len(per_frame_error)
    if n_frames == 0:
        return np.nan, np.nan

    centroid_error = np.nansum(per_frame_error['centroid_distance_mm']) / n_frames
    heading_error = np.nansum(per_frame_error['heading_angle_deg']) / n_frames
    return centroid_error, heading_error


def get_average_speed(behavior_data: BehaviorData) -> float:

    dark_start, dark_stop = get_dark_epoch(behavior_data)
    if dark_start == -1:
        return np.nan

    pix_per_mm = behavior_data.metadata['calibration']['pix_per_mm']
    fps = behavior_data.metadata['camera']['framerate_value']
    duration = (dark_stop - dark_start) / fps

    offline_centroid = np.column_stack((
        behavior_data.full_tracking.Swim_Bladder['x'],
        behavior_data.full_tracking.Swim_Bladder['y']
    ))
    total_distance_traveled = np.sum(np.linalg.norm(np.diff(offline_centroid[dark_start:dark_stop], axis=0), axis=1))
    average_speed = total_distance_traveled / (duration * pix_per_mm)
    return average_speed


def is_online_tracking_bad(
        behavior_data: BehaviorData,
        centroid_threshold_mm_per_frame: float = 1,
        heading_threshold_deg_per_frame: float = 15,
        per_frame_error: Optional[pd.DataFrame] = None,
    ) -> tuple[bool, bool]:

    centroid_error_mm_per_frame, heading_error_deg_per_frame = get_tracking_error(
        behavior_data, per_frame_error=per_frame_error
    )
    return (
        centroid_error_mm_per_frame >= centroid_threshold_mm_per_frame,
        heading_error_deg_per_frame >= heading_threshold_deg_per_frame
    )


def is_offline_tracking_bad(
        behavior_data: BehaviorData,
    ) -> bool:
    # TODO maybe check the likelihood for the tail?
    pass


def is_fish_not_moving(
        behavior_data: BehaviorData,
        speed_threshold_mm_per_sec: float = 0.2
    ) -> bool:

    # returns False if dark epoch not found
    average_speed_mm_per_sec = get_average_speed(behavior_data)
    return (average_speed_mm_per_sec < speed_threshold_mm_per_sec)


# ---------------------------------------------------------------------------
# Per-trial tracking quality.
#
# Answers "did online tracking agree with offline (ground-truth) tracking
# during THIS trial" -- independent of whether the trial was even expected
# under the protocol (that's get_epoch_trial_presentation's job). The two
# are merged into a single output table (valid_trials.csv, see
# quality_control below) since both are keyed at the exact same
# (epoch_name, trial_num) granularity, and every downstream consumer needs
# BOTH to agree (presented AND tracking_ok) before trusting a trial --
# kept as separate COLUMNS so it stays visible which of the two
# disqualified any given row.
# ---------------------------------------------------------------------------

def get_trial_tracking_coverage(
        per_frame_error: pd.DataFrame,
        start_frame: int,
        stop_frame: int,
        centroid_threshold_mm_per_frame: float = 1.0,
        heading_threshold_deg_per_frame: float = 15.0,
    ) -> float:
    """
    Fraction of frames within [start_frame, stop_frame) where online and
    offline tracking AGREE (both per-frame centroid distance and heading
    angle mismatch under threshold).

    Denominator is the EXPECTED frame count (stop_frame - start_frame), not
    the number of rows actually present in per_frame_error -- so a frame
    dropped from the online<->offline join counts against coverage rather
    than being silently excluded from the average.
    """
    if stop_frame <= start_frame:
        return np.nan

    expected_n = stop_frame - start_frame

    if per_frame_error.empty:
        return 0.0

    window = per_frame_error[
        (per_frame_error.index >= start_frame) & (per_frame_error.index < stop_frame)
    ]

    agrees = (
        (window['centroid_distance_mm'] < centroid_threshold_mm_per_frame) &
        (window['heading_angle_deg'] < heading_threshold_deg_per_frame)
    )
    return int(agrees.sum()) / expected_n


def get_epoch_trial_tracking_quality(
        behavior_data: BehaviorData,
        per_frame_error: pd.DataFrame,
        centroid_threshold_mm_per_frame: float = 1.0,
        heading_threshold_deg_per_frame: float = 15.0,
        coverage_threshold: float = 0.8,
    ) -> pd.DataFrame:
    """
    Per-trial tracking-quality record, one row per (epoch_name, trial_num)
    that ACTUALLY occurred in the log (unlike get_epoch_trial_presentation,
    this does not iterate PROTOCOL_SPEC -- a trial that never happened has
    no tracking window to evaluate, so no row here).

    trial_num assigned POSITIONALLY within each epoch_name group, over ALL
    occurrences -- identical convention to get_epoch_trial_presentation and
    megabouts.get_bout_metrics' bouts.csv trial_num, so this merges cleanly
    with the presentation table on (epoch_name, trial_num).
    """
    stim_trials = get_trials(behavior_data)

    rows = []
    for epoch_name, epoch_trials in stim_trials.groupby('epoch_name'):
        for trial_num, (_, row) in enumerate(epoch_trials.iterrows()):

            start_frame = timestamp_to_frame(behavior_data, row.start_timestamp)
            stop_frame = timestamp_to_frame(behavior_data, row.stop_timestamp)
            coverage = get_trial_tracking_coverage(
                per_frame_error,
                start_frame,
                stop_frame,
                centroid_threshold_mm_per_frame=centroid_threshold_mm_per_frame,
                heading_threshold_deg_per_frame=heading_threshold_deg_per_frame,
            )

            rows.append({
                'epoch_name': epoch_name,
                'trial_num': trial_num,
                'tracking_coverage': coverage,
                'tracking_ok': bool(coverage >= coverage_threshold) if np.isfinite(coverage) else False,
            })

    return pd.DataFrame(rows)


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
        plots=plots
    )
    behavior_files = find_files(directories)

    bad_fish = []
    valid_trial_tables = []

    for behavior_file in tqdm(behavior_files):
        behavior_data = load_data(behavior_file)

        # shared per-frame online/offline comparison, computed once and
        # reused by both the session-level and trial-level checks below
        per_frame_error = get_per_frame_tracking_error(behavior_data)

        # fish-level QC (unchanged): coarse, whole-session veto
        not_moving = is_fish_not_moving(behavior_data)
        centroid_issue, heading_issue = is_online_tracking_bad(
            behavior_data, per_frame_error=per_frame_error
        )
        if not_moving | centroid_issue | heading_issue:
            bad_fish.append((behavior_file.metadata.stem, not_moving, centroid_issue, heading_issue))

        # per-trial epoch presentation (structural: was it expected & run)
        presentation = get_epoch_trial_presentation(behavior_data)

        # per-trial tracking quality (data-quality: was it trustworthy),
        # merged into the SAME table on (epoch_name, trial_num).
        tracking_quality = get_epoch_trial_tracking_quality(
            behavior_data,
            per_frame_error,
            centroid_threshold_mm_per_frame=trial_centroid_threshold_mm,
            heading_threshold_deg_per_frame=trial_heading_threshold_deg,
            coverage_threshold=coverage_threshold,
        )

        merged = presentation.merge(
            tracking_quality, on=['epoch_name', 'trial_num'], how='left'
        )
        # tracking_quality has no row for (epoch_name, trial_num) combinations
        # that were expected (per PROTOCOL_SPEC) but never actually logged --
        # those are already presented=False from get_epoch_trial_presentation,
        # so tracking_ok is moot for them, but must still be resolved to an
        # explicit False here: a left join leaves it NaN, and NaN in a bool
        # column silently upcasts to object dtype, which a later
        # .astype(bool) would misread as True (numpy/pandas has no
        # bool-with-missing dtype).
        merged['tracking_ok'] = merged['tracking_ok'].fillna(False).astype(bool)
        merged['file'] = behavior_file.metadata.stem
        valid_trial_tables.append(merged)

    header = ['file', 'not_moving', 'centroid_issue', 'heading_issue']
    pd.DataFrame(bad_fish, columns=header).to_csv(root / output_csv, index=False)

    valid_trials = (
        pd.concat(valid_trial_tables, ignore_index=True) if valid_trial_tables
        else pd.DataFrame(columns=[
            'epoch_name', 'stim', 'expected_repeats', 'trial_num',
            'presented', 'matches_expected_parameters', 'trial_duration_s',
            'tracking_coverage', 'tracking_ok', 'file',
        ])
    )
    valid_trials.to_csv(root / valid_trials_csv, index=False)


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


if __name__ == '__main__':

    main(build_parser().parse_args())