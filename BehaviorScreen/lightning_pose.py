from pathlib import Path
from typing import List
from urllib.request import urlretrieve
import subprocess
import shutil
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


def move_directory_contents(src: Path, dst: Path, overwrite: bool = True) -> None:
    """
    Move the contents of src into dst

    Parameters
    ----------
    src : Path
        Source directory whose contents will be moved.
    dst : Path
        Destination directory.
    overwrite : bool
        Whether to overwrite existing files/directories in dst.
    """
    src = Path(src)
    dst = Path(dst)

    if not src.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name

        if target.exists():
            if not overwrite:
                raise FileExistsError(f"Target already exists: {target}")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

        shutil.move(str(item), str(target))

    # Remove src if it's empty after moving
    try:
        src.rmdir()
    except OSError:
        pass

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

    move_directory_contents(
        model_directory / "video_preds",
        output_directory
    )

if __name__ == '__main__':
    
    download_model()
