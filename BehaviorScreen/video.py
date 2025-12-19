
from typing import Dict

import json
import numpy as np
from tqdm import tqdm

from video_tools import OpenCV_VideoWriter, OpenCV_VideoReader, CPU_VideoProcessor
from BehaviorScreen.load import BehaviorData, BehaviorFiles, Directories
from BehaviorScreen.core import Stim

def ensure_results_dir(directories: Directories) -> None:
    directories.results.mkdir(parents=True, exist_ok=True)

def export_single_animal_metadata(
    directories: Directories,
    behavior_file: BehaviorFiles,
    behavior_data: BehaviorData,
) -> None:

    ensure_results_dir(directories)
    
    for i, _ in enumerate(behavior_data.metadata['identity']['ROIs']):
        metadata_file = behavior_file.metadata.stem + f"_fish_{i}.metadata"
        out_path = directories.results / metadata_file

        with open(out_path, 'w') as fp:
            json.dump(behavior_data.metadata, fp)

def export_single_animal_tracking(
    directories: Directories,
    behavior_file: BehaviorFiles,
    behavior_data: BehaviorData,
) -> None:

    ensure_results_dir(directories)

    df = behavior_data.tracking
    for i, _ in enumerate(behavior_data.metadata['identity']['ROIs']):
        tracking_file = behavior_file.tracking.stem + f"_fish_{i}.csv"
        out_path = directories.results / tracking_file
        df[df.identity == i].set_index('index').to_csv(out_path)
        

def export_single_animal_timestamps(
    directories: Directories,
    behavior_file: BehaviorFiles,
    behavior_data: BehaviorData,
) -> None:

    ensure_results_dir(directories)

    for i, _ in enumerate(behavior_data.metadata['identity']['ROIs']):
        timestamp_file = behavior_file.video_timestamps.stem + f"_fish_{i}.csv"
        out_path = directories.results / timestamp_file 
        behavior_data.video_timestamps.to_csv(out_path)

def export_single_animal_stimuli(
    directories: Directories,
    behavior_file: BehaviorFiles,
    behavior_data: BehaviorData,
) -> None:

    ensure_results_dir(directories)

    for i, _ in enumerate(behavior_data.metadata['identity']['ROIs']):
        stim_file = behavior_file.stimuli.stem + f"_fish_{i}.json"
        out_path = directories.results / stim_file 
        with open(out_path, 'w') as fp:
            json.dump(behavior_data.stimuli, fp)

def export_single_animal_videos(
    directories: Directories,
    behavior_file: BehaviorFiles,
    behavior_data: BehaviorData,
    quality: int = 18,
) -> None:

    ensure_results_dir(directories)

    video_cropper = CPU_VideoProcessor(
        str(behavior_file.video),
        quality=quality,
    )

    for i, (x, y, w, h) in enumerate(
        behavior_data.metadata['identity']['ROIs']
    ):
        video_cropper.crop(
            x, y, w, h,
            suffix=f"fish_{i}",
            dest_folder=str(directories.results),
        )

def export_single_animal(
    directories: Directories,
    behavior_file: BehaviorFiles,
    behavior_data: BehaviorData,
    quality: int = 18,
) -> None:

    export_single_animal_tracking(
        directories, behavior_file, behavior_data
    )
    export_single_animal_timestamps(
        directories, behavior_file, behavior_data
    )
    export_single_animal_stimuli(
        directories, behavior_file, behavior_data
    )
    export_single_animal_metadata(
        directories, behavior_file, behavior_data
    )
    export_single_animal_videos(
        directories, behavior_file, behavior_data, quality
    )

def timestamp_to_frame_index(behavior_data: BehaviorData, timestamp: int) -> int:
    distance = behavior_data.video_timestamps['timestamp'] - timestamp
    idx_closest = distance.abs().argmin()
    frame_index = behavior_data.video_timestamps['index'][idx_closest]
    return frame_index

def superimpose_video_trials(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData,
        trial_duration_sec: float,
        grouping_parameter: Dict[Stim, str]
    ) -> None:

    directories.results.mkdir(parents=True, exist_ok=True)
    
    height = behavior_data.metadata['camera']['height_value']
    width = behavior_data.metadata['camera']['width_value']
    fps = int(behavior_data.metadata['camera']['framerate_value'])
    num_frames = int(trial_duration_sec * behavior_data.metadata['camera']['framerate_value'])

    stim_trials = get_trials(
        behavior_data,
        grouping_parameter.keys()
    )
    for stim, stim_data in tqdm(stim_trials.groupby('stim_select')):
        for parameter_value, data in stim_data.groupby(grouping_parameter[Stim(stim)]):
            
            output_path = directories.results / f"{behavior_file.video.stem}_{Stim(stim)}_{grouping_parameter[Stim(stim)]}_{parameter_value}.mp4"
            writer = OpenCV_VideoWriter(
                height = height, 
                width = width,
                fps = fps,
                filename = str(output_path),
                fourcc = 'mp4v'
            )

            readers = []
            for start_timestamp in data['start_timestamp']:
                frame_index_start = timestamp_to_frame_index(behavior_data, start_timestamp)
                reader = OpenCV_VideoReader()
                reader.open_file(str(behavior_file.video))
                reader.seek_to(frame_index_start)
                readers.append(reader)

            num_trials = len(data['start_timestamp'])
            mip = np.zeros((height, width, num_trials), dtype=np.uint8)
            for _ in tqdm(range(num_frames)):
                for trial_idx in range(num_trials):
                    _, frame = readers[trial_idx].next_frame()
                    mip[:,:,trial_idx] =  frame[:,:,0]
                writer.write_frame(np.min(mip, axis=2))

            writer.close()

