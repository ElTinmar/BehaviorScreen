from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

URL = "https://owncloud.gwdg.de/"

def download_model(url: str = URL, destination: Path = ''):

    destination.mkdir(parents=True, exist_ok=True)
    zip_file = destination / "models.zip"

    print("Downloading...")
    urlretrieve(url, zip_file)

    print("Extracting...")
    with ZipFile(zip_file, "r") as zip:
        zip.extractall(destination)

    zip_file.unlink()

def estimate_pose():
    ...