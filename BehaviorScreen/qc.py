# remove fish with bad online tracking
# remove fish that did not move 
# remove empty wells
from pathlib import Path
import argparse
import argparse
from pathlib import Path
from tqdm import tqdm
from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    find_files, 
    load_data
)
from BehaviorScreen.process import timestamp_to_frame
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
        help="Output CSV file containing bout x visual stim",
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

def get_dark_epoch(behavior_data: BehaviorData) -> tuple[int,int]:
    
    res = (-1, -1)
    
    for i in range(len(behavior_data.stimuli) - 10):

        is_sequence_start = all(
            behavior_data.stimuli[i+j].get('stim_select') == Stim.DARK 
            for j in range(10)
        )
        
        if is_sequence_start:
            start_ts = behavior_data.stimuli[i].get('timestamp')
            stop_ts = behavior_data.stimuli[i+10].get('timestamp')
            start_frame = timestamp_to_frame(behavior_data,start_ts)
            stop_frame = timestamp_to_frame(behavior_data,stop_ts)
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

def get_tracking_error(behavior_data: BehaviorData) -> tuple[float, float]:

    online = behavior_data.tracking.set_index('index')
    offline = behavior_data.full_tracking
    offline.columns = [f"{level0}_{level1}" if level1 else level0 
                   for level0, level1 in offline.columns]
    
    common = online.join(offline, how='inner')
    n_frames = len(common)

    # centroid error
    online_centroid = np.column_stack((
        common.centroid_x,
        common.centroid_y
    ))
    offline_centroid = np.column_stack((
        common.Swim_Bladder_x,
        common.Swim_Bladder_y
    ))
    distance_centroid = np.linalg.norm(offline_centroid-online_centroid, axis=1)
    pix_per_mm = behavior_data.metadata['calibration']['pix_per_mm']
    centroid_error = np.nansum(distance_centroid)/ (pix_per_mm*n_frames)

    # heading axis error (maybe add measure of occasional flip?)
    online_heading = np.column_stack((common.pc1_x, common.pc1_y))
    head = np.column_stack((common.Head_x, common.Head_y))
    sb =  np.column_stack((common.Swim_Bladder_x, common.Swim_Bladder_y))
    offline_heading = head - sb
    distance_angle = angle_between(online_heading, offline_heading)
    heading_error = np.nansum(distance_angle) / n_frames

    return centroid_error, heading_error

def get_average_speed(behavior_data: BehaviorData) -> float:

    dark_start, dark_stop = get_dark_epoch(behavior_data)
    if dark_start == -1:
        return np.nan 
    
    pix_per_mm = behavior_data.metadata['calibration']['pix_per_mm']
    fps = behavior_data.metadata['camera']['framerate_value']
    duration = (dark_stop - dark_start)/fps

    offline_centroid = np.column_stack((
        behavior_data.full_tracking.Swim_Bladder['x'],
        behavior_data.full_tracking.Swim_Bladder['y']
    ))
    total_distance_traveled = np.sum(np.linalg.norm(np.diff(offline_centroid[dark_start:dark_stop], axis=0), axis=1))
    average_speed = total_distance_traveled / (duration*pix_per_mm)
    return average_speed
    
def is_online_tracking_bad(
        behavior_data: BehaviorData, 
        centroid_threshold_mm_per_frame: float = 1,
        heading_threshold_deg_per_frame: float = 15 
    ) -> tuple[bool, bool]:
    
    centroid_error_mm_per_frame, heading_error_deg_per_frame = get_tracking_error(behavior_data)
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
    
def quality_control(
        root: Path,
        output_csv: str,
        metadata: str,
        stimuli: str,
        tracking: str,
        lightning_pose: str,
        temperature: str,
        video: str,
        video_timestamp: str,
        results: str,
        plots: str,
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
    for behavior_file in tqdm(behavior_files):
        behavior_data = load_data(behavior_file)
        not_moving = is_fish_not_moving(behavior_data)
        centroid_issue, heading_issue = is_online_tracking_bad(behavior_data)
        if not_moving | centroid_issue | heading_issue:
            bad_fish.append((behavior_file.metadata.stem, not_moving, centroid_issue, heading_issue))  
    
    header = ['file', 'not_moving', 'centroid_issue', 'heading_issue']
    pd.DataFrame(bad_fish, columns=header).to_csv(root / output_csv, index=False)

def main(args: argparse.Namespace) -> None:
    quality_control(
        root=args.root,
        output_csv=args.qc_csv,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        lightning_pose=args.lightning_pose,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
    )

if __name__ == '__main__':

    main(build_parser().parse_args())