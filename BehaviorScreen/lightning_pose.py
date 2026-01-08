from pathlib import Path
from urllib.request import urlretrieve
import subprocess

from BehaviorScreen.config import LIGHTNING_POSE_MODEL_URL

def download_model(
        url: str = LIGHTNING_POSE_MODEL_URL, 
        destination: Path = Path('')
    ):

    destination.mkdir(parents=True, exist_ok=True)
    file = destination / "lightning_pose.ckpt"
    print("Downloading...")
    urlretrieve(url, file)

def estimate_pose(
        ckpt_file: Path,
        video: Path,
        lightning_pose_conda_env: str = "LightningPose"
    ) -> None: 

    cmd = [
        "conda", "run", "-n", lightning_pose_conda_env,
        "litpose",
        "predict",
        str(ckpt_file), str(video),
    ]
    subprocess.run(cmd, check=True)