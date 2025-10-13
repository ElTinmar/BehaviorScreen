# Plug-and-Play automated behavioral tracking of zebrafish larvae with DeepLabCut and SLEAP: pre-trained networks and datasets of annotated poses
# https://doi.org/10.1101/2025.06.04.657938 
# https://doi.org/10.26188/29275838

from BehaviorScreen.core import MODELS_FOLDER, MODELS_URL
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

def download_and_extract_models(url: str, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    zip_file = dest_folder / "models.zip"

    print("Downloading...")
    urlretrieve(url, zip_file)

    print("Extracting...")
    with ZipFile(zip_file, "r") as zip:
        zip.extractall(dest_folder)

    zip_file.unlink()

if __name__ == '__main__':

    download_and_extract_models(MODELS_URL, MODELS_FOLDER)