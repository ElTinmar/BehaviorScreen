from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

TAIL_MODEL_URL_JAN_26 = "https://owncloud.gwdg.de/index.php/s/cysLNkUMxr3emTn/download"
TAIL_MODEL_URL_APRIL_26 = "https://owncloud.gwdg.de/index.php/s/2opYRUKAachVHQJ/download"
EYES_MODEL_URL = "https://owncloud.gwdg.de/index.php/s/l5xbLSfATCCydEx/download"


def download_model(
        urls: list[str] = [TAIL_MODEL_URL_APRIL_26, EYES_MODEL_URL], 
        destination: Path = Path('')
    ):

    destination.mkdir(parents=True, exist_ok=True)
    
    for url in urls:

        zip_file = destination / "model.zip"
        
        print(f"Downloading {url}...", flush=True)
        urlretrieve(url, zip_file)

        print("Extracting...", flush=True)
        with ZipFile(zip_file, "r") as zip:
            zip.extractall(destination)

        print("Cleaning up...", flush=True)
        zip_file.unlink()

if __name__ == '__main__':
    
    download_model()
