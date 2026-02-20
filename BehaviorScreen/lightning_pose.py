from pathlib import Path
from typing import List
import subprocess
import argparse

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Estimate pose with LightningPose"
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
        "--lightning-pose",
        default="lightning_pose",
        help="Subfolder containing lightning pose tracking CSV files (default: lightning_pose)",
    )

    parser.add_argument(
        "--results",
        default="results",
        help="Subfolder where per-animal exports will be written (default: results)",
    )


    return parser

def estimate_pose(
        model_directory: Path,
        video_directory: Path,
        output_directory: Path,
        video_extensions: List[str] = [".mp4", ".avi"],
        lightning_pose_conda_env: str = "LightningPose"
    ) -> None: 

    video_directory = Path(video_directory)
    model_directory = Path(model_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    videos = [v for v in video_directory.iterdir() if v.suffix.lower() in video_extensions]
    if not videos:
        print(f"No video files found in {video_directory}", flush=True)
        return  
    
    for video in videos:
        print(f"Processing {video}...", flush=True)
        cmd = [
            "conda", "run", "-n", lightning_pose_conda_env,
            "litpose",
            "predict",
            str(model_directory), 
            str(video),
            '--prediction_dir', output_directory 
        ]
        subprocess.run(cmd, check=True)

def main(args: argparse.Namespace):

    estimate_pose(
        model_directory=args.model_dir,
        video_directory=args.root / args.results,
        output_directory=args.root / args.lightning_pose
    )
    
if __name__ == '__main__':
    
    main(build_parser().parse_args())