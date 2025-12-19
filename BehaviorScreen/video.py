
from typing import Dict

import json
import numpy as np
from tqdm import tqdm

from video_tools import OpenCV_VideoWriter, OpenCV_VideoReader, CPU_VideoProcessor
from BehaviorScreen.load import BehaviorData, BehaviorFiles, Directories
from BehaviorScreen.core import Stim

def export_single_animal_videos(
        directories: Directories, 
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData,
        quality: int = 18
    ) -> None:

    directories.results.mkdir(parents=True, exist_ok=True)
    video_cropper = CPU_VideoProcessor(str(behavior_file.video), quality = quality)

    for i, (x,y,w,h) in  enumerate(behavior_data.metadata['identity']['ROIs']):

        # tracking
        tracking_file = behavior_file.tracking.stem + f"_fish_{i}.csv"
        df = behavior_data.tracking
        df[df.identity == i].set_index('index').to_csv(directories.results / tracking_file)

        # timestamps
        video_timestamp = behavior_file.video_timestamps.stem + f"_fish_{i}.csv"
        behavior_data.video_timestamps.to_csv(directories.results / video_timestamp)

        # stimuli
        stim_file = behavior_file.stimuli.stem + f"_fish_{i}.json"
        with open(directories.results / stim_file, 'w') as fp:
            json.dump(behavior_data.stimuli, fp)

        # metadata
        metadata_file = behavior_file.metadata.stem + f"_fish_{i}.metadata"
        with open(directories.results / metadata_file, 'w') as fp:
            json.dump(behavior_data.metadata, fp)

        # cropped video
        video_cropper.crop(
            x,y,w,h,
            suffix=f"fish_{i}",
            dest_folder=str(directories.results)
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

