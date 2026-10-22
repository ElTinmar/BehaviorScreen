import argparse
from pathlib import Path
from typing import Callable, Dict

from BehaviorScreen.status import SACCADES_CSV


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
        default='Martin_Apr2026_Full/model',
        help="Path to LightningPose trained full model directory",
    )

    parser.add_argument(
        "--eyes_model_dir",
        type=str,
        default='Martin_Mar2026_Eyes/model',
        help="Path to LightningPose trained eyes model directory",
    )

    parser.add_argument(
        "--saccade-model",
        type=Path,
        default=Path('BehaviorScreen/eyes/paper_data/paper_reference.joblib'),
        help=(
            "Path to the Dowell saccade classification model. "
            "Required when starting from detect or classify."
        ),
    )

    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("BehaviorScreen/screen.yaml"),
        help="Plot config file",
    )

    parser.add_argument(
        "--qc-csv",
        default='qc.csv',
        help="quality control: fish not moving and tracking issues",
    )

    parser.add_argument(
        "--valid-trials-csv",
        default='valid_trials.csv',
        help="per-trial epoch presentation + tracking quality record (used by plot and point-process fitting)",
    )

    parser.add_argument(
        "--bouts-csv",
        default='bouts.csv',
        help="Bouts CSV file",
    )

    parser.add_argument(
        "--filtered-bouts-csv",
        default='filtered_bouts.csv',
        help="Filtered bouts CSV file",
    )

    parser.add_argument(
        "--bouts-png",
        default='bouts.png',
        help="Bout heatmap PNG file",
    )

    parser.add_argument(
        "--saccades-png",
        default='saccades.png',
        help="Saccade heatmap PNG file",
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
        help="Subfolder where the bout and saccade plots are written (default: plots)",
    )

    # Feature toggles
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--no-stimuli", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS), help="Steps to run, in pipeline order (default: all)")

    return parser

def export(args: argparse.Namespace) -> None:
    from BehaviorScreen.export import export_single_animals
    export_single_animals(
        root=args.root,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        quality=args.quality,
        tracking_flag=not args.no_tracking,
        timestamps_flag=not args.no_timestamps,
        stimuli_flag=not args.no_stimuli,
        metadata_flag=not args.no_metadata,
        videos_flag=not args.no_videos,
    )


def pose(args: argparse.Namespace) -> None:
    from BehaviorScreen.pose_estimation import estimate_pose
    estimate_pose(
        model_directory=args.full_model_dir,
        video_directory=args.root / args.results,
        output_directory=args.root / args.lightning_pose
    )


def eye_crop(args: argparse.Namespace) -> None:
    from BehaviorScreen.pose_estimation import export_cropped_eyes_video
    export_cropped_eyes_video(
        full_video_directory=args.root / args.results,
        eye_video_directory=args.root / args.eyes_video_dir,
        full_tracking_directory=args.root / args.lightning_pose
    )


def eye_pose(args: argparse.Namespace) -> None:
    from BehaviorScreen.pose_estimation import estimate_pose
    estimate_pose(
        model_directory=args.eyes_model_dir,
        video_directory=args.root / args.eyes_video_dir,
        output_directory=args.root / args.lightning_pose
    )


def qc(args: argparse.Namespace) -> None:
    from BehaviorScreen.qc import quality_control
    quality_control(
        root=args.root,
        output_csv=args.qc_csv,
        valid_trials_csv=args.valid_trials_csv,
        metadata=args.results,
        stimuli=args.results,
        tracking=args.results,
        lightning_pose=args.lightning_pose,
        temperature=args.results,
        video=args.results,
        video_timestamp=args.results,
        results=args.results,
    )


def bouts(args: argparse.Namespace) -> None:
    from BehaviorScreen.megabouts import run_megabouts
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
        cpu=args.cpu,
    )


def heatmaps(args: argparse.Namespace) -> None:
    """Bout and saccade heatmaps, each when its table exists."""
    from BehaviorScreen.eyes.plot_saccade_heatmap import make_saccade_heatmaps
    from BehaviorScreen.plot import plot_heatmaps
    plots = args.root / args.plots
    plots.mkdir(parents=True, exist_ok=True)
    common = dict(quality_control=args.root / args.qc_csv, valid_trials_csv=args.root / args.valid_trials_csv, config_yaml=args.yaml)
    bouts, saccades = args.root / args.bouts_csv, args.root / SACCADES_CSV
    if bouts.exists():
        plot_heatmaps(input_csv=bouts, output_png=plots / args.bouts_png, **common)
    else:
        print(f"no {bouts}: bout heatmaps skipped")
    if saccades.exists():
        make_saccade_heatmaps(input_csv=saccades, output_png=plots / args.saccades_png, exclude_unusable=True,
                              include_unassigned=False, maximum_frequency=0.3, interactive=False, **common)
    else:
        print(f"no {saccades}: saccade heatmaps skipped")


def saccades(args: argparse.Namespace) -> None:
    from BehaviorScreen.eyes.run_analysis import run_analysis
    run_analysis(
        root=args.root,
        model_path=args.saccade_model,
        results=args.results,
        lightning_pose=args.lightning_pose,
        plots=args.plots,
        heatmaps=False,
    )


# Pipeline order; imports are inside the steps so the parser (and the GUI) load fast.
STEPS: Dict[str, Callable[[argparse.Namespace], None]] = {
    "export": export, "pose": pose, "eye_crop": eye_crop, "eye_pose": eye_pose,
    "qc": qc, "bouts": bouts, "saccades": saccades, "heatmaps": heatmaps,
}


def main(args: argparse.Namespace) -> None:
    for i, name in enumerate([s for s in STEPS if s in args.steps], 1):
        print(f"{i}. {name}", flush=True)
        STEPS[name](args)


if __name__ == '__main__':

    main(build_parser().parse_args())
