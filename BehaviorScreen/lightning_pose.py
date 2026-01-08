from pathlib import Path
from typing import List
from urllib.request import urlretrieve
import subprocess
from zipfile import ZipFile

from BehaviorScreen.config import LIGHTNING_POSE_MODEL_URL

def download_model(
        url: str = LIGHTNING_POSE_MODEL_URL, 
        destination: Path = Path('')
    ):

    destination.mkdir(parents=True, exist_ok=True)
    zip_file = destination / "model.zip"
    print(f"Downloading {url}...")
    urlretrieve(url, zip_file)

    print("Extracting...")
    with ZipFile(zip_file, "r") as zip:
        zip.extractall(destination)

    print("Cleaning up...")
    zip_file.unlink()

def estimate_pose(
        model_directory: Path,
        video_directory: Path,
        video_extensions: List[str] = [".mp4", ".avi"],
        lightning_pose_conda_env: str = "LightningPose"
    ) -> None: 

    videos = [v for v in video_directory.iterdir() if v.suffix.lower() in video_extensions]
    if not videos:
        print(f"No video files found in {video_directory}")
        return  
    
    for video in videos:
        print(f"Processing {video}...")
        cmd = [
            "conda", "run", "-n", lightning_pose_conda_env,
            "litpose",
            "predict",
            str(model_directory), 
            str(video),
        ]
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    
    download_model()
