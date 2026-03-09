from pathlib import Path
from typing import List
import subprocess
import argparse
import pandas as pd
import numpy as np
import cv2
from video_tools import FFMPEG_VideoWriter_CPU
from tqdm import tqdm

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
        "--eyes_video_dir",
        default="eyes",
        help="Directory to store cropped eyes video",
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

# TODO check what happens if no tracking data for a frame
def crop_around_eyes(
        input_video: Path,
        output_video: Path,
        lightningpose_csv: Path,
        crop_size_mm: float = 1.0,
        px_per_mm: float = 40
    ):
    
    crop_size = 2 * int(crop_size_mm * px_per_mm) // 2

    lp_data = pd.read_csv(lightningpose_csv, header=[0,1,2])

    swimbladder = lp_data.Swim_Bladder[['x', 'y']].to_numpy()
    head =  lp_data.Head[['x', 'y']].to_numpy()
    heading = head - swimbladder
    theta_rad = np.arctan2(heading[:,1], heading[:,0]) 

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = FFMPEG_VideoWriter_CPU(
        height = crop_size,
        width = crop_size,
        fps = fps,
        q = 18,
        filename = output_video,
        preset = 'fast'
    )
    
    try:

        for x, y, th in tqdm(zip(head[:,0], head[:,1], theta_rad)):
            
            ret, frame = cap.read()
            if not ret:
                break

            M = cv2.getRotationMatrix2D(
                center = (x, y),
                angle = np.rad2deg(th) + 90,
                scale = 1.0,
            )
            M[0, 2] += crop_size // 2 - x
            M[1, 2] += crop_size // 2 - y

            warped = cv2.warpAffine(
                frame,
                M,
                (crop_size, crop_size),
                flags = cv2.INTER_CUBIC,
                borderMode = cv2.BORDER_CONSTANT,
                borderValue = (0, 0, 0),
            )

            writer.write_frame(warped)
    
    finally:
        writer.close()
    
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

        print(f"Pose estimation: {video}...", flush=True)

        # pose estimation for the full fish
        cmd = [
            "conda", "run", "-n", lightning_pose_conda_env,
            "litpose",
            "predict",
            str(model_directory), 
            str(video),
            '--prediction_dir', output_directory,
            '--overrides', f"data.data_dir={model_directory.parent}" # FileNotFoundError: Could not find csv file at /path/CollectedData.csv!
        ]
        subprocess.run(cmd, check=True)

def export_cropped_eyes_video(
        full_video_directory: Path,
        eye_video_directory: Path,
        full_tracking_directory: Path,
        video_extensions: List[str] = [".mp4", ".avi"],
        overwrite: bool = False
    ) -> None: 

    full_video_directory = Path(full_video_directory)
    full_tracking_directory = Path(full_tracking_directory)
    eye_video_directory.mkdir(parents=True, exist_ok=True)

    videos = [v for v in full_video_directory.iterdir() if v.suffix.lower() in video_extensions]
    if not videos:
        print(f"No video files found in {full_video_directory}", flush=True)
        return  

    for video in videos:

        print(f"Export eye video: {video}...", flush=True)
        
        eyes_video = eye_video_directory / ("eyes_" + video.name)

        if eyes_video.exists() and not overwrite:
            print(f"{eyes_video} already exists, skipping ...")
            continue

        lightningpose_csv = full_tracking_directory / (video.stem + '.csv')

        if not lightningpose_csv.exists():
            raise FileNotFoundError(f'{lightningpose_csv} does not exist')

        crop_around_eyes(
            video, 
            eyes_video, 
            lightningpose_csv,
            crop_size_mm = 1.6,
            px_per_mm = 40.0 # TODO get this from metadata
        )

def main(args: argparse.Namespace):

    estimate_pose(
        model_directory = args.full_model_dir,
        video_directory = args.root / args.results,
        output_directory = args.root / args.lightning_pose
    )

    export_cropped_eyes_video(
        full_video_directory = args.root / args.results,
        eye_video_directory = args.root / args.eyes_video_dir,
        full_tracking_directory = args.root / args.lightning_pose
    )

    estimate_pose(
        model_directory = args.eyes_model_dir,
        video_directory = args.root / args.eyes_video_dir,
        output_directory = args.root / args.lightning_pose
    )
    
if __name__ == '__main__':
    
    main(build_parser().parse_args())