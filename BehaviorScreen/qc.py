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
    find_files, 
    load_data
)

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

def is_online_tracking_bad():
    ...

def is_fish_not_moving():
    ...
    
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
        cpu: bool,
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

    for behavior_file in tqdm(behavior_files):
        behavior_data = load_data(behavior_file)
        qc = is_fish_not_moving(behavior_data) | is_online_tracking_bad(behavior_data)

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