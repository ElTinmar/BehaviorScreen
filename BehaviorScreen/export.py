import json
import argparse
from pathlib import Path
import numpy as np

from video_tools import CPU_VideoProcessor
from BehaviorScreen.load import (
    BehaviorData, 
    BehaviorFiles, 
    Directories,
    find_files,
    load_data
)
from BehaviorScreen.process import get_background_image_safe

def ensure_results_dir(directories: Directories) -> None:
    directories.results.mkdir(parents=True, exist_ok=True)

def export_metadata(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData,
    ) -> None:

    ensure_results_dir(directories)
    
    for i, (x,y,w,h) in enumerate(behavior_data.metadata['identity']['ROIs']):
        metadata_file = behavior_file.metadata.stem + f"_fish_{i}.metadata"
        out_path = directories.results / metadata_file
        metadata = behavior_data.metadata.copy()

        try:
            background_img = np.asarray(metadata['background']['image'])
        except KeyError:
            # TODO fix the issue 
            background_img = get_background_image_safe(behavior_data)

        metadata['background']['image_ROI'] = background_img[y:y+h, x:x+w].tolist()
        metadata['export'] = {}
        metadata['export']['fish_ID'] = i

        with open(out_path, 'w') as fp:
            json.dump(metadata, fp)

def export_tracking(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData,
    ) -> None:

    ensure_results_dir(directories)

    df = behavior_data.tracking
    for i, _ in enumerate(behavior_data.metadata['identity']['ROIs']):
        tracking_file = behavior_file.tracking.stem + f"_fish_{i}.csv"
        out_path = directories.results / tracking_file
        current_df = df[df.identity == i].set_index('index').copy()

        offset_x, offset_y, _, _ = behavior_data.metadata['identity']['ROIs'][i]
        coords_to_transform = [
            'centroid',
            'left_eye',
            'right_eye'
        ]
        n_tail_points = behavior_data.metadata['settings']['tracking']['n_tail_pts_interp']
        coords_to_transform.extend([f'tail_point_{n:03}' for n in range(n_tail_points)])
        
        for coord in coords_to_transform:
            current_df[coord + '_x'] -= offset_x
            current_df[coord + '_y'] -= offset_y

        current_df.to_csv(out_path)
        

def export_timestamps(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData,
    ) -> None:

    ensure_results_dir(directories)

    for i, _ in enumerate(behavior_data.metadata['identity']['ROIs']):
        timestamp_file = behavior_file.video_timestamps.stem + f"_fish_{i}.csv"
        out_path = directories.results / timestamp_file 
        behavior_data.video_timestamps.to_csv(out_path, index=False)

def export_stimuli(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData,
    ) -> None:
    # NOTE: not using JSON properly

    ensure_results_dir(directories)

    for i, _ in enumerate(behavior_data.metadata['identity']['ROIs']):
        stim_file = behavior_file.stimuli.stem + f"_fish_{i}.json"
        out_path = directories.results / stim_file 
        with open(out_path, 'w') as fp:
            for line in behavior_data.stimuli:
                fp.write(json.dumps(line) + '\n')

def export_videos(
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

def export(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData,
        *,
        tracking_flag: bool = True,
        timestamps_flag: bool = True,
        stimuli_flag: bool = True,
        metadata_flag: bool = True,
        videos_flag: bool = True,
        quality: int = 18,
    ) -> None:

    if tracking_flag:
        export_tracking(
            directories, behavior_file, behavior_data
        )

    if timestamps_flag:
        export_timestamps(
            directories, behavior_file, behavior_data
        )

    if stimuli_flag:
        export_stimuli(
            directories, behavior_file, behavior_data
        )

    if metadata_flag:
        export_metadata(
            directories, behavior_file, behavior_data
        )

    if videos_flag:
        export_videos(
            directories, behavior_file, behavior_data, quality
        )

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Export per-animal behavior data (tracking, metadata, stimuli, videos)"
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder (e.g. WT_oct_2025)",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=18,
        help="Video encoding quality (lower = better quality, default: 18)",
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

    # Feature toggles
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--no-stimuli", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-videos", action="store_true")

    return parser

def main(args: argparse.Namespace) -> None:
    export_single_animals(
        root=args.root,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
        quality=args.quality,
        tracking_flag=not args.no_tracking,
        timestamps_flag=not args.no_timestamps,
        stimuli_flag=not args.no_stimuli,
        metadata_flag=not args.no_metadata,
        videos_flag=not args.no_videos,
    )

def export_single_animals(
        root: Path,
        *,
        metadata: str = 'data',
        stimuli: str = 'data',
        tracking: str = 'data',
        temperature: str = 'data',
        video: str = 'video',
        video_timestamp: str = 'video',
        results: str = 'results',
        plots: str = 'plots',
        quality: int = 18,
        tracking_flag: bool = True,
        timestamps_flag: bool = True,
        stimuli_flag: bool = True,
        metadata_flag: bool = True,
        videos_flag: bool = True,
    ) -> None:

    directories = Directories(
        root=root,
        metadata=metadata,
        stimuli=stimuli,
        tracking=tracking,
        temperature=temperature,
        video=video,
        video_timestamp=video_timestamp,
        results=results,
        plots=plots,
    )

    behavior_files = find_files(directories)

    for file in behavior_files:
        print(f'processing {file.metadata.stem}', flush=True)
        behavior_data = load_data(file)
        export(
            directories,
            file,
            behavior_data,
            tracking_flag = tracking_flag,
            timestamps_flag = timestamps_flag,
            stimuli_flag = stimuli_flag,
            metadata_flag = metadata_flag,
            videos_flag = videos_flag,
            quality = quality,
        )
    
if __name__ == '__main__':

    main(build_parser().parse_args())

