#!/usr/bin/env python3
"""Add recording, stimulus, trial, and position context to saccades."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from tqdm import tqdm

from BehaviorScreen.event_context import event_context, posthoc_positions_at, trial_indices
from BehaviorScreen.load import Directories, find_files
from BehaviorScreen.recording import Recording
from BehaviorScreen.protocol import saccade_laterality
from BehaviorScreen.core import (
    SACCADE_CATEGORY_BY_CLUSTER,
    SACCADE_CATEGORY_NAMES,
    SACCADE_CLASS_DIRECTIONS,
    SACCADE_NONDIRECTIONAL_CLASSES,
)


def augment_fish_events(rec: Recording, events: pd.DataFrame) -> list[dict[str, Any]]:
    """Contextual records for one recording's saccades."""
    # onset_time_s is relative to the first video frame (see detect_saccades)
    onset_ns = np.rint(events["onset_time_s"].to_numpy(float) * 1e9).astype(np.int64)
    event_timestamps = rec.frame_time_ns[0] + onset_ns
    trials = trial_indices(rec, event_timestamps)
    position = posthoc_positions_at(rec, event_timestamps)

    return [
        event_context(
            rec,
            int(t),
            int(trial),
            float(position["x_mm"][i]),
            float(position["y_mm"][i]),
            float(position["heading"][i]),
        )
        | {
            "tracking_frame": int(position["tracking_frame"][i]),
            "tracking_time_error_ms": float(position["tracking_time_error_ms"][i]),
        }
        for i, (t, trial) in enumerate(zip(event_timestamps, trials))
    ]


def augment_saccades(
    input_csv: Path,
    output_csv: Path,
    directories: Directories,
) -> None:
    """Add common behavioral context to all classified saccades."""
    saccades = pd.read_csv(input_csv)

    required_columns = {
        "fish",
        "onset_time_s",
        "cluster",
    }
    missing_columns = required_columns.difference(saccades.columns)

    if missing_columns:
        raise ValueError(
            f"{input_csv} is missing columns: " f"{sorted(missing_columns)}"
        )

    files_by_name = {files.metadata.stem: files for files in find_files(directories)}

    contexts_by_index: dict[int, dict[str, Any]] = {}

    for fish, fish_events in tqdm(
        saccades.groupby("fish", sort=False),
        desc="Saccade context",
    ):
        fish = str(fish)

        if fish not in files_by_name:
            raise FileNotFoundError(f"No BehaviorFiles were found for {fish}.")

        contexts = augment_fish_events(Recording(files_by_name[fish]), fish_events)

        if len(contexts) != len(fish_events):
            raise RuntimeError(
                f"Context count mismatch for {fish}: "
                f"{len(contexts)} versus {len(fish_events)}."
            )

        for dataframe_index, context in zip(
            fish_events.index,
            contexts,
        ):
            contexts_by_index[int(dataframe_index)] = context

    context_table = pd.DataFrame.from_dict(
        contexts_by_index,
        orient="index",
    ).reindex(saccades.index)

    if len(context_table) != len(saccades):
        raise RuntimeError("The context table does not match the saccade table.")

    overlapping_columns = set(context_table.columns).intersection(saccades.columns)

    if overlapping_columns:
        context_table = context_table.rename(
            columns={column: f"context_{column}" for column in overlapping_columns}
        )

    augmented = pd.concat(
        [
            saccades.reset_index(drop=True),
            context_table.reset_index(drop=True),
        ],
        axis=1,
    )

    if {
        "fish",
        "file",
    }.issubset(augmented.columns):
        mismatch = augmented["fish"].astype(str) != augmented["file"].astype(str)

        if mismatch.any():
            examples = augmented.loc[
                mismatch,
                ["fish", "file"],
            ].head()

            raise ValueError(
                "Saccade fish labels do not match recording files:\n" f"{examples}"
            )

    if "event_id" in augmented.columns:
        if augmented["event_id"].duplicated().any():
            raise ValueError("Duplicate event_id values were found.")

    cluster_numeric = pd.to_numeric(
        augmented["cluster"],
        errors="coerce",
    )

    direction_by_cluster = {
        int(cluster): int(direction)
        for cluster, direction in SACCADE_CLASS_DIRECTIONS.items()
    }

    augmented["saccade_category"] = cluster_numeric.map(
        SACCADE_CATEGORY_BY_CLUSTER
    ).astype("Int64")

    augmented["saccade_category_name"] = (
        augmented["saccade_category"].map(SACCADE_CATEGORY_NAMES).astype("string")
    )

    # Numeric EventDirection code:
    #   -1 = LEFT
    #   +1 = RIGHT
    #   NA = no defined left/right direction
    augmented["event_direction"] = cluster_numeric.map(direction_by_cluster).astype(
        "Int64"
    )

    # Readable direction. Genuine directionless saccades are explicitly
    # marked NONDIRECTIONAL. Invalid/excluded classifications remain NA.
    augmented["event_direction_name"] = (
        augmented["event_direction"]
        .map(
            {
                -1: "LEFT",
                1: "RIGHT",
            }
        )
        .astype("string")
    )

    nondirectional_mask = cluster_numeric.isin(SACCADE_NONDIRECTIONAL_CLASSES)

    augmented.loc[
        nondirectional_mask,
        "event_direction_name",
    ] = "NONDIRECTIONAL"

    # Assign ipsi/contra/none only to events in a recognised stimulus trial.
    # Events outside trials retain NaN laterality.
    augmented["laterality"] = np.nan

    laterality_mask = (
        augmented["in_stimulus_trial"].fillna(False).astype(bool)
        & augmented["epoch_name"].notna()
    )

    augmented.loc[laterality_mask, "laterality"] = [
        saccade_laterality(
            epoch_name=epoch_name,
            cluster=cluster,
        )
        for epoch_name, cluster in augmented.loc[
            laterality_mask,
            ["epoch_name", "cluster"],
        ].itertuples(index=False, name=None)
    ]

    augmented["laterality_name"] = (
        pd.to_numeric(
            augmented["laterality"],
            errors="coerce",
        )
        .map(
            {
                -1: "CONTRALATERAL",
                0: "NONDIRECTIONAL",
                1: "IPSILATERAL",
            }
        )
        .astype("string")
    )
    
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    augmented.to_csv(
        output_csv,
        index=False,
        float_format="%.10g",
    )

    print(f"Saved {len(augmented):,} events to {output_csv}")

    if "in_stimulus_trial" in augmented.columns:
        print(
            "Events assigned to a stimulus trial: "
            f"{augmented['in_stimulus_trial'].mean():.1%}"
        )

    if "distance_center" in augmented.columns:
        print(
            "Events with valid position: "
            f"{augmented['distance_center'].notna().mean():.1%}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Add recording, stimulus, trial, and position context "
            "to classified saccades."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment directory.",
    )
    parser.add_argument(
        "--input",
        default="saccades_classified.csv",
        help=(
            "Input classified-saccade CSV relative to root. "
            "Default: saccades_classified.csv."
        ),
    )
    parser.add_argument(
        "--output",
        default="saccades_augmented.csv",
        help=(
            "Output augmented-saccade CSV relative to root. "
            "Default: saccades_augmented.csv."
        ),
    )
    parser.add_argument("--metadata", default="results")
    parser.add_argument("--stimuli", default="results")
    parser.add_argument("--tracking", default="results")
    parser.add_argument(
        "--lightning-pose",
        default="lightning_pose",
    )
    parser.add_argument("--temperature", default="results")
    parser.add_argument("--video", default="results")
    parser.add_argument(
        "--video-timestamp",
        default="results",
    )
    parser.add_argument("--results", default="results")
    parser.add_argument("--plots", default="plots")

    return parser


def main() -> None:
    """Run saccade augmentation."""
    args = build_parser().parse_args()

    root = args.root.resolve()
    input_csv = root / args.input
    output_csv = root / args.output

    if not input_csv.exists():
        raise FileNotFoundError(f"Classified saccade CSV does not exist: {input_csv}")

    directories = Directories(
        root,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        full_tracking=args.lightning_pose,
        eyes_tracking=args.lightning_pose,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
    )

    augment_saccades(
        input_csv=input_csv,
        output_csv=output_csv,
        directories=directories,
    )


if __name__ == "__main__":
    main()
