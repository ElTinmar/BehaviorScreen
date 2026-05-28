from pathlib import Path
import pickle
import argparse
import numpy as np
import pandas as pd

from BehaviorScreen.megabouts import MegaboutResults

def load_megabout_results(megabout_file: Path) -> MegaboutResults:
    with open(megabout_file, 'rb') as f:
        data = pickle.load(f)
    return data

def export_kinematics(data: MegaboutResults, output_csv: Path) -> None:
    n_tail_points = data.tail.angle_smooth.shape[1] 
    kinematics = np.column_stack((
        data.traj.x_smooth,
        data.traj.y_smooth,
        data.traj.yaw_smooth,
        data.tail.angle_smooth
    ))
    df = pd.DataFrame(kinematics, columns=["x", "y", "yaw"] + [f"tail_{i}" for i in range(n_tail_points)])
    df.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Export kinematic data from a Megabout results pickle file to a CSV."
    )
    
    parser.add_argument(
        "megabout_file",
        type=Path,
        help="Path to the input Megabout .pkl / .pickle file."
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        dest="output_csv",
        help="Path to the output CSV file. (Default: input_filename_kinematics.csv)"
    )

    args = parser.parse_args()

    if not args.megabout_file.is_file():
        parser.error(f"The input file '{args.megabout_file}' does not exist.")

    if args.output_csv is None:
        args.output_csv = args.megabout_file.with_name(f"{args.megabout_file.stem}_kinematics.csv")

    print(f"Loading {args.megabout_file}...")
    data = load_megabout_results(args.megabout_file)
    export_kinematics(data, args.output_csv)

if __name__ == "__main__":
    main()