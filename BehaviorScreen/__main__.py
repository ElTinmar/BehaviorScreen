import argparse
from pathlib import Path

from BehaviorScreen.export import export_single_animals
from BehaviorScreen.megabouts import run_megabouts
from BehaviorScreen.lightning_pose import estimate_pose
#from BehaviorScreen.plot import plot_bout_heatmap

# TODO eye tracking OKR
# TODO eye tracking + tail tracking and classification J-turn PREY_CAPTURE
# TODO separate analysis and plotting. Use multiprocessing for analysis here
# TODO linear mixed effects analysis to get within and between individual variability
# TODO indentify the main source of variability within/between individuals
# TODO overlay reconstructed stimulus on top of video 
# TODO overlay video with ethogram
# TODO megabout segmentation sanity checks
# TODO permutation tests with DARK? 
# TODO plot trajectories loomings 

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
        "model_dir",
        type=Path,
        help="Root experiment folder (e.g. WT_oct_2025)",
    )

    parser.add_argument(
        "--bouts-csv",
        default='bouts.csv',
        help="Output CSV file",
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
        "--lightning-pose",
        default="lightning_pose",
        help="Subfolder containing lightning pose tracking CSV files (default: data)",
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
    parser.add_argument("--cpu", action="store_true")

    return parser

def main(args: argparse.Namespace) -> None:

    print("1. export ROIs")
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

    print("2. pose estimation")
    estimate_pose(
        model_directory=args.model_dir,
        video_directory=args.root / args.results,
        output_directory=args.root / args.lightning_pose
    )
    
    print("3. extract bout metrics")
    run_megabouts(
        root=args.root,
        output_csv=args.bouts_csv,
        metadata=args.results,
        stimuli=args.results,
        tracking=args.results,
        lightning_pose=args.lightning_pose,
        temperature=args.results,
        video=args.results,
        video_timestamp=args.results,
        results=args.results,
        plots=args.plots,
        cpu=args.cpu
    )

    #print("4. plot")
    #plot_bout_heatmap()

if __name__ == '__main__':

    main(build_parser().parse_args())