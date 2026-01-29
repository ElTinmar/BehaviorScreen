"""
Script to convert SLEAP project to LP project

Usage:
$ python slp2lp.py --slp_file /path/to/<project>.pkg.slp --lp_dir /path/to/lp/dir

Arguments:
--slp_file    Path to the SLEAP project file (.pkg.slp)
--lp_dir      Path to the output LP project directory

"""

import argparse
import os
import pandas as pd
import sleap_io as sio
from sleap import PredictedInstance
from pathlib import Path
import cv2


def keep_user_labeled_only(input_slp, output_slp):

    labels = sio.load_file(input_slp)
    labels.labeled_frames = [
        f for f in labels.labeled_frames 
        if any([True for i in f.instances if not isinstance(i, PredictedInstance)])
    ]
    labels.suggestions = []
    labels.save(output_slp, embed=True)


def slp2lp(slp_pkg_file: Path, base_output_dir: Path) -> pd.DataFrame:

    labels = sio.load_file(slp_pkg_file)

    rows = []
    index = []

    for frame in labels.labeled_frames:

        # skip if no user instance
        if not frame.user_instances:
            continue
        
        user_instance = frame.user_instances[0]

        video_file = Path(frame.video.backend.source_filename).stem
        image_file = f'labeled-data/{video_file}/img{frame.frame_idx:08}.png'
        index.append(image_file)    

        # extract image from video
        image_path = base_output_dir / image_file
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_path, frame.image) 
    
        # extract keypoints
        keypoints_flat = user_instance.numpy().flatten().tolist()
        rows.append(keypoints_flat)

    keypoint_names = [node.name for node in labels.skeleton.nodes]
    columns = pd.MultiIndex.from_product(
        [["lightning_tracker"], keypoint_names, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )

    df = pd.DataFrame(rows, columns=columns, index=index)
    df.to_csv(base_output_dir / "CollectedData.csv")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--slp_file", type=str)
    parser.add_argument("--lp_dir", type=str)
    args = parser.parse_args()
    slp_file = args.slp_file
    lp_dir = args.lp_dir

    print(f"Converting SLEAP project located at {slp_file} to LP project located at {lp_dir}", flush=True)

    # Check provided SLEAP path exists
    if not os.path.exists(slp_file):
        raise FileNotFoundError(f"did not find the file {slp_file}")

    # Check paths are not the same
    if slp_file == lp_dir:
        raise NameError("slp_file and lp_dir cannot be the same")
    
    tmp_slp_file = Path('tmp.slp')
    keep_user_labeled_only(slp_file, tmp_slp_file)
    slp2lp(tmp_slp_file, lp_dir)
    tmp_slp_file.unlink()

    