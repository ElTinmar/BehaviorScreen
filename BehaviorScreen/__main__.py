import argparse
from pathlib import Path

from BehaviorScreen.export import export_single_animals
from BehaviorScreen.megabouts import run_megabouts
from BehaviorScreen.pose_estimation import estimate_pose, export_cropped_eyes_video
from BehaviorScreen.plot import run_plot

# TODO separate analysis and plotting
# TODO linear mixed effects analysis to get within and between individual variability
# TODO indentify the main source of variability within/between individuals
# TODO permutation tests with DARK? 
# TODO plot trajectories loomings 

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Export per-animal behavior data (tracking, metadata, stimuli, videos)"
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Path to root experiment folder (e.g. WT_oct_2025)",
    )

    parser.add_argument(
        "--full_model_dir",
        type=str,
        default='Martin_Jan2026_Full/model',
        help="Path to LightningPose trained full model directory",
    )

    parser.add_argument(
        "--eyes_model_dir",
        type=str,
        default='Martin_Mar2026_Eyes/model',
        help="Path to LightningPose trained eyes model directory",
    )

    parser.add_argument(
        "--yaml",
        default = 'BehaviorScreen/screen.yaml',
        help="Plot config file",
    )

    parser.add_argument(
        "--bouts-csv",
        default='bouts.csv',
        help="Output CSV file",
    )

    parser.add_argument(
        "--megabout",
        default='megabout.pkl',
        help="Output pickle file containing megabouts results",
    )

    parser.add_argument(
        "--bouts-png",
        default='bouts.png',
        help="output PNG file",
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
        "--eyes-video-dir",
        default="eyes",
        help="Subfolder where cropped eyes video will be written (default: eyes)",
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

    print("1. export ROIs", flush=True)
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

    print("2. pose estimation", flush=True)
    estimate_pose(
        model_directory=args.full_model_dir,
        video_directory=args.root / args.results,
        output_directory=args.root / args.lightning_pose
    )
    export_cropped_eyes_video(
        full_video_directory=args.root / args.results,
        eye_video_directory=args.root / args.eyes_video_dir,
        full_tracking_directory=args.root / args.lightning_pose
    )
    estimate_pose(
        model_directory=args.eyes_model_dir,
        video_directory=args.root / args.eyes_video_dir,
        output_directory=args.root / args.lightning_pose
    )
    
    print("3. extract bout metrics", flush=True)
    run_megabouts(
        root=args.root,
        output_csv=args.bouts_csv,
        output_megabout=args.megabout,
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

    print("4. extract eye metrics", flush=True)
    # TODO

    print("5. plot", flush=True)
    run_plot(
        bouts_csv=args.bouts_csv,
        bouts_png = args.bouts_png,
        config_yaml = args.yaml,
        root = args.root,
        metadata=args.results,
        stimuli=args.results,
        tracking=args.results,
        lightning_pose=args.lightning_pose,
        temperature=args.results,
        video=args.results,
        video_timestamp=args.results,
        results=args.results,
        plots=args.plots
    )

    print("6. overlay")
    # TODO

if __name__ == '__main__':

    main(build_parser().parse_args())