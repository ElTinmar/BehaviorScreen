from pathlib import Path
from urllib.request import urlretrieve
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
        ckpt_file: Path
    ):
    ...