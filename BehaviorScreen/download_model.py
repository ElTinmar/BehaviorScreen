"""
Download the models into a folder (default: here), where the pipeline's
default model paths point:

    python -m BehaviorScreen.download_model [full] [eyes] [saccades] [--destination FOLDER]

Zip archives (the Lightning Pose models) are unpacked; any other file is saved
under its target name.
"""
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile, is_zipfile

TAIL_MODEL_URL_JAN_26 = "https://owncloud.gwdg.de/index.php/s/cysLNkUMxr3emTn/download"
TAIL_MODEL_URL_APRIL_26 = "https://owncloud.gwdg.de/index.php/s/2opYRUKAachVHQJ/download"
EYES_MODEL_URL = "https://owncloud.gwdg.de/index.php/s/l5xbLSfATCCydEx/download"
SACCADE_MODEL_URL = "https://owncloud.gwdg.de/index.php/s/PLACEHOLDER/download"  # TODO: share paper_reference.joblib

# name: (url, target relative to the destination: a folder to unpack into, or a file)
MODELS = {
    "full": (TAIL_MODEL_URL_APRIL_26, Path(".")),
    "eyes": (EYES_MODEL_URL, Path(".")),
    "saccades": (SACCADE_MODEL_URL, Path("BehaviorScreen/eyes/paper_data/paper_reference.joblib")),
}


def download(url: str, target: Path) -> None:
    """Unpack a zip archive into the folder `target`, or save any other file as `target`."""
    if "PLACEHOLDER" in url:
        raise ValueError(f"No download link yet for {target.name}: set it in BehaviorScreen/download_model.py")
    folder = target.parent if target.suffix else target
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / "download.part"
    print(f"Downloading {url}...", flush=True)
    urlretrieve(url, part)
    if is_zipfile(part):
        print("Extracting...", flush=True)
        with ZipFile(part) as archive:
            archive.extractall(folder)
        part.unlink()
    else:
        part.replace(target)
    print(f"Saved in {target}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the pipeline's models.")
    parser.add_argument("models", nargs="*", choices=list(MODELS), help="Models to download (default: all)")
    parser.add_argument("--destination", type=Path, default=Path("."), help="Folder the models go in (default: here)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    for name in args.models or MODELS:
        url, target = MODELS[name]
        download(url, args.destination / target)
