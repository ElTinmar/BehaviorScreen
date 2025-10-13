# Plug-and-Play automated behavioral tracking of zebrafish larvae with DeepLabCut and SLEAP: pre-trained networks and datasets of annotated poses
# https://doi.org/10.1101/2025.06.04.657938 

from BehaviorScreen.core import DLC_MODELS_FOLDER, DLC_MODELS_URL
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

def download_and_extract_DLC_models(url: str, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    zip_file = dest_folder / "dlc_models.zip"

    print("Downloading...")
    urlretrieve(url, zip_file)

    print("Extracting...")
    with ZipFile(zip_file, "r") as zip:
        zip.extractall(dest_folder)

    zip_file.unlink()

if __name__ == '__main__':

    download_and_extract_DLC_models(DLC_MODELS_URL, DLC_MODELS_FOLDER)