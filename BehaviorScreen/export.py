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

def ensure_results_dir(directories: Directories) -> None:
    directories.results.mkdir(parents=True, exist_ok=True)

def export_single_animal_metadata(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData,
    ) -> None:

    ensure_results_dir(directories)
    
    for i, (x,y,w,h) in enumerate(behavior_data.metadata['identity']['ROIs']):
        metadata_file = behavior_file.metadata.stem + f"_fish_{i}.metadata"
        out_path = directories.results / metadata_file
        metadata = behavior_data.metadata.copy()
        background_img = np.asarray(metadata['background']['image'])
        metadata['background']['image_ROI'] = background_img[x:x+w, y:y+h].tolist()
        metadata['export'] = {}
        metadata['export']['fish_ID'] = i

        with open(out_path, 'w') as fp:
            json.dump(metadata, fp)

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
    # NOTE: not using JSON properly

    ensure_results_dir(directories)

    for i, _ in enumerate(behavior_data.metadata['identity']['ROIs']):
        stim_file = behavior_file.stimuli.stem + f"_fish_{i}.json"
        out_path = directories.results / stim_file 
        with open(out_path, 'w') as fp:
            for line in behavior_data.stimuli:
                fp.write(json.dumps(line) + '\n')

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
        *,
        export_tracking: bool = True,
        export_timestamps: bool = True,
        export_stimuli: bool = True,
        export_metadata: bool = True,
        export_videos: bool = True,
        quality: int = 18,
    ) -> None:

    if export_tracking:
        export_single_animal_tracking(
            directories, behavior_file, behavior_data
        )

    if export_timestamps:
        export_single_animal_timestamps(
            directories, behavior_file, behavior_data
        )

    if export_stimuli:
        export_single_animal_stimuli(
            directories, behavior_file, behavior_data
        )

    if export_metadata:
        export_single_animal_metadata(
            directories, behavior_file, behavior_data
        )

    if export_videos:
        export_single_animal_videos(
            directories, behavior_file, behavior_data, quality
        )

def export_cli(
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
        export_tracking: bool = True,
        export_timestamps: bool = True,
        export_stimuli: bool = True,
        export_metadata: bool = True,
        export_videos: bool = True,
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
        print(f'processing {file.metadata.stem}')
        behavior_data = load_data(file)
        export_single_animal(
            directories,
            file,
            behavior_data,
            export_tracking = export_tracking,
            export_timestamps = export_timestamps,
            export_stimuli = export_stimuli,
            export_metadata = export_metadata,
            export_videos = export_videos,
            quality = quality,
        )
    
if __name__ == '__main__':

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

    args = parser.parse_args()

    export_cli(
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
        export_tracking=not args.no_tracking,
        export_timestamps=not args.no_timestamps,
        export_stimuli=not args.no_stimuli,
        export_metadata=not args.no_metadata,
        export_videos=not args.no_videos,
    )