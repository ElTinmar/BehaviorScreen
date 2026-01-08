from pathlib import Path
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
        video: Path,
        lightning_pose_conda_env: str = "LightningPose"
    ) -> None: 

    cmd = [
        "conda", "run", "-n", lightning_pose_conda_env,
        "litpose",
        "predict",
        str(model_directory), str(video),
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    
    download_model()
