import numpy as np
import pandas as pd
from typing import NamedTuple, Dict, List
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm

from megabouts.tracking_data import TrackingConfig, FullTrackingData
from megabouts.pipeline import FullTrackingPipeline
from megabouts.pipeline.freely_swimming_pipeline import EthogramFullTracking
from megabouts.classification.classification import TailBouts
from megabouts.segmentation.segmentation import SegmentationResult
from megabouts.preprocessing.tail_preprocessing import TailPreprocessingResult
from megabouts.preprocessing.traj_preprocessing import TrajPreprocessingResult

from BehaviorScreen.core import GROUPING_PARAMETER, Stim
from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    BehaviorFiles,
    find_files, 
    load_data
)
from BehaviorScreen.process import (
    get_trials, 
    get_well_coords_mm, 
    compute_eye_angle_from_keypoints
)
from BehaviorScreen.stimulus import prey_capture_arc_stimulus_cosine

class MegaboutResults(NamedTuple):
    timestamp: np.ndarray
    ethogram: EthogramFullTracking
    bouts: TailBouts
    segments: SegmentationResult
    tail: TailPreprocessingResult
    traj: TrajPreprocessingResult

def EyeData_from_lp(lp_csv: str, threshold: float = 0.95) -> pd.DataFrame:

    df = pd.read_csv(lp_csv, header=[0,1,2])

    def eye_angles(side):
        
        keep = ((df.heatmap_tracker[f"Eye_{side}_Front"].likelihood > threshold) &
                (df.heatmap_tracker[f"Eye_{side}_Back"].likelihood > threshold))
        
        return compute_eye_angle_from_keypoints(
            front_x = df.heatmap_tracker[f"Eye_{side}_Front"].x,
            front_y = df.heatmap_tracker[f"Eye_{side}_Front"].y,
            back_x = df.heatmap_tracker[f"Eye_{side}_Back"].x,
            back_y = df.heatmap_tracker[f"Eye_{side}_Back"].y,
            head_x = df.heatmap_tracker.Head.x,
            head_y = df.heatmap_tracker.Head.y,
            swimbladder_x = df.heatmap_tracker.Swim_Bladder.x,
            swimbladder_y = df.heatmap_tracker.Swim_Bladder.y,
            mask = keep
        )

    return pd.DataFrame({
        "left_eye_angle": eye_angles("Left"),
        "right_eye_angle": eye_angles("Right")
    })

def FullTrackingData_from_lp(df: pd.DataFrame, mm_per_pix: float) -> FullTrackingData:
        
    head_x = df.heatmap_tracker.Head.x.values * mm_per_pix
    head_y = df.heatmap_tracker.Head.y.values * mm_per_pix
    tail_parts = [f"Tail_{i}" for i in range(9)] # exclude last tail point
    tail_x = df.loc[:, ("heatmap_tracker", tail_parts, "x")].values * mm_per_pix
    tail_y = df.loc[:, ("heatmap_tracker", tail_parts, "y")].values * mm_per_pix
    tracking_data = FullTrackingData.from_keypoints(
        head_x=head_x, head_y=head_y, tail_x=tail_x, tail_y=tail_y
    )
    return tracking_data

def megabout_fulltracking_pipeline(
        behavior_data: BehaviorData,
        min_bout_duration_ms: int = 70,
        segmentation_threshold: float = 7.5,
        tail_speed_boxcar_filter_ms: int = 30,
        savgol_window_ms: int = 20
    ) -> MegaboutResults:

    mm_per_pix = 1/behavior_data.metadata['calibration']['pix_per_mm']
    fps = behavior_data.metadata['camera']['framerate_value']
    timestamps = behavior_data.tracking.timestamp.values
    # timestamps = behavior_data.video_timestamps.timestamp.values

    # configure pipeline
    tracking_cfg = TrackingConfig(fps=fps, tracking="full_tracking")
    pipeline = FullTrackingPipeline(tracking_cfg, exclude_CS=True)
    pipeline.segmentation_cfg.min_bout_duration_ms = min_bout_duration_ms
    pipeline.segmentation_cfg.threshold = segmentation_threshold
    pipeline.tail_preprocessing_cfg.tail_speed_boxcar_filter_ms = tail_speed_boxcar_filter_ms
    pipeline.tail_preprocessing_cfg.savgol_window_ms = savgol_window_ms
    
    # run pipeline
    tracking_data = FullTrackingData_from_lp(behavior_data.full_tracking, mm_per_pix)
    ethogram, bouts, segments, tail, traj = pipeline.run(tracking_data)

    # TODO, full tracking is one frame longer ???
    megabout_results = MegaboutResults(
            timestamps, 
            ethogram, 
            bouts, 
            segments,
            tail, 
            traj
        ) 

    return megabout_results

def get_bout_metrics(
        directories: Directories,
        behavior_data: BehaviorData, 
        behavior_files: BehaviorFiles,
        megabout: MegaboutResults,
        rollover_time_s: int = 3600
    ) -> List[Dict]:

    well_coords_mm = get_well_coords_mm(directories, behavior_files, behavior_data)
    fps = behavior_data.metadata['camera']['framerate_value']
    stim_trials = get_trials(behavior_data)
    bout_start = megabout.timestamp[megabout.bouts.onset]
    bout_stop = megabout.timestamp[megabout.bouts.offset]

    # these are needed for QC
    online_tracking_centroid = behavior_data.tracking[['centroid_x','centroid_y']].values
    posthoc_tracking_centroid = behavior_data.full_tracking.heatmap_tracker.Swim_Bladder[['x', 'y']].values
    online_tracking_heading =  behavior_data.tracking[['pc1_x','pc1_y']].values
    posthoc_tracking_heading = behavior_data.full_tracking.heatmap_tracker.Head[['x', 'y']].values - posthoc_tracking_centroid
    posthoc_tracking_heading = posthoc_tracking_heading / np.linalg.norm(posthoc_tracking_heading, axis=1, keepdims=True)
    
    cx,cy,_ = well_coords_mm[0,:]
    rows = []

    for stim_select, stim_data in stim_trials.groupby('stim_select'):
        
        stim = Stim(stim_select)
        if not stim in GROUPING_PARAMETER:
            continue

        for condition, condition_data in stim_data.groupby(GROUPING_PARAMETER[stim]):

            for trial_idx, (trial, row) in enumerate(condition_data.iterrows()):

                mask = (bout_start > row.start_timestamp) & (bout_stop < row.stop_timestamp) 

                off_previous = 0
                for on, off, category, sign, proba, bout_index in zip(
                    megabout.bouts.onset[mask], 
                    megabout.bouts.offset[mask],
                    megabout.bouts.category[mask],
                    megabout.bouts.sign[mask],
                    megabout.bouts.proba[mask],
                    np.flatnonzero(mask)
                ):

                    # heading change
                    heading_change = megabout.traj.yaw_smooth[off] - megabout.traj.yaw_smooth[on]

                    # distance 
                    delta_x =  np.diff(megabout.traj.x_smooth[on:off])
                    delta_y = np.diff(megabout.traj.y_smooth[on:off])
                    distance = np.sum(np.sqrt(delta_x**2 + delta_y**2))
                    radial_distance = np.sqrt((megabout.traj.x_smooth[on]-cx)**2 + (megabout.traj.y_smooth[on]-cy)**2)

                    # bout duration
                    bout_duration = (off-on)/fps

                    # interbout duration
                    interbout_duration = (on-off_previous)/fps

                    # peak axial speed
                    axial_speed = megabout.traj.axial_speed[on:off]
                    peak_axial_speed = axial_speed[np.argmax(np.abs(axial_speed))]

                    # peak yaw speed
                    yaw_speed = megabout.traj.yaw_speed[on:off]
                    peak_yaw_speed = yaw_speed[np.argmax(np.abs(yaw_speed))]

                    # time
                    trial_time = 1e-9*(megabout.timestamp[on] - row.start_timestamp)
                    start_time = 1e-9*(megabout.timestamp[on] - megabout.timestamp[0])
                    stop_time = 1e-9*(megabout.timestamp[off] - megabout.timestamp[0])

                    # Stimulus phase
                    stim_phase = np.nan
                    if stim == Stim.PREY_CAPTURE:
                        stim_phase = prey_capture_arc_stimulus_cosine(
                            row.start_time_sec,
                            trial_time,
                            rollover_time_s,
                            row.prey_arc_start_deg,
                            row.prey_arc_stop_deg,
                            row.prey_speed_deg_s
                        )

                    # Quality control: posthoc vs online tracking during interbout + bout
                    centroid_distance = np.linalg.norm(online_tracking_centroid[off_previous:off,:] - posthoc_tracking_centroid[off_previous:off,:], axis=1)
                    centroid_mismatch_avg = centroid_distance.sum() / (off-off_previous)
                    centroid_mismatch_max = centroid_distance.max()

                    dot = np.sum(online_tracking_heading[off_previous:off,:] * posthoc_tracking_heading[off_previous:off,:], axis=1)
                    dot = np.clip(dot, -1.0, 1.0)
                    angular_distance = np.rad2deg(np.arccos(dot))
                    heading_mismatch_avg = angular_distance.sum() / (off-off_previous)
                    heading_mismatch_max = angular_distance.max() 
                    heading_flip = np.any(angular_distance > 160)

                    # prepare next iteration
                    off_previous = off

                    rows.append({
                        'file': behavior_files.metadata.stem,
                        'bout_index': bout_index,
                        'frame_start': on,
                        'frame_stop': off,
                        'time_start': start_time,
                        'time_stop': stop_time,
                        'stim': stim_select,
                        'stim_variable_name': GROUPING_PARAMETER[stim],
                        'stim_variable_value': str(condition),
                        'stim_start_time': 1e-9*(row.start_timestamp - stim_trials.start_timestamp[0]),
                        'trial_num': trial_idx,
                        'trial_time': trial_time,
                        'heading_change': heading_change,
                        'distance': distance,
                        'distance_center': radial_distance,
                        'bout_duration': bout_duration,
                        'stim_phase': stim_phase,
                        'interbout_duration': interbout_duration,
                        'peak_axial_speed': peak_axial_speed,
                        'peak_yaw_speed': peak_yaw_speed,
                        'centroid_mismatch_avg': centroid_mismatch_avg,
                        'heading_mismatch_avg': heading_mismatch_avg,
                        'centroid_mismatch_max': centroid_mismatch_max,
                        'heading_mismatch_max': heading_mismatch_max,
                        'heading_flip': heading_flip,
                        'category': category,
                        'sign': sign,
                        'proba': proba
                    })

    return rows

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
        "--bouts-csv",
        default='bouts.csv',
        help="Output CSV file containing bout x visual stim",
    )

    parser.add_argument(
        "--megabout",
        default='megabout.pkl',
        help="Output pickle file containing megabouts results",
    )

    # Directory layout overrides
    parser.add_argument(
        "--metadata",
        default="data",
        help="Subfolder containing metadata files (default: data)",
    )

    parser.add_argument(
        "--stimuli",
        default="data",
        help="Subfolder containing stimulus log files (default: data)",
    )

    parser.add_argument(
        "--tracking",
        default="data",
        help="Subfolder containing tracking CSV files (default: data)",
    )

    parser.add_argument(
        "--lightning-pose",
        default="lightning_pose",
        help="Subfolder containing lightning pose tracking CSV files (default: lightning_pose)",
    )

    parser.add_argument(
        "--temperature",
        default="data",
        help="Subfolder containing temperature logs (default: data)",
    )

    parser.add_argument(
        "--video",
        default="video",
        help="Subfolder containing raw video files (default: video)",
    )

    parser.add_argument(
        "--video-timestamp",
        default="video",
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

    # Run megabout on CPU if no compatible GPU
    parser.add_argument("--cpu", action="store_true")

    return parser

def run_megabouts(
        root: Path,
        output_csv: str,
        output_megabout: str,
        metadata: str,
        stimuli: str,
        tracking: str,
        lightning_pose: str,
        temperature: str,
        video: str,
        video_timestamp: str,
        results: str,
        plots: str,
        cpu: bool,
    ) -> None:

    if cpu:
        # Force running on CPU if GPU is not compatible
        import torch
        torch.cuda.is_available = lambda: False

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

    bout_stim = []
    megabout_results = {}
    for behavior_file in tqdm(behavior_files):
        behavior_data = load_data(behavior_file)
        megabout = megabout_fulltracking_pipeline(behavior_data)
        megabout_results[behavior_file.metadata.stem] = megabout
        bout_metrics = get_bout_metrics(directories, behavior_data, behavior_file, megabout)
        bout_stim.extend(bout_metrics)

    bouts = pd.DataFrame(bout_stim)
    bouts.to_csv(
        root / output_csv, 
        header=True, 
        index=False
    )

    with open(root / output_megabout, "wb") as f:
        pickle.dump(megabout_results, f)

def main(args: argparse.Namespace) -> None:
    run_megabouts(
        root=args.root,
        output_csv=args.bouts_csv,
        output_megabout=args.megabout,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        lightning_pose=args.lightning_pose,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
        cpu=args.cpu
    )
    
if __name__ == '__main__':

    main(build_parser().parse_args())