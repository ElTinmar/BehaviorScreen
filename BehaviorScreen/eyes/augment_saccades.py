#!/usr/bin/env python3
"""Add recording, stimulus, trial, and position context to saccades."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from BehaviorScreen.event_context import RecordingContext
from BehaviorScreen.load import (
    BehaviorData,
    BehaviorFiles,
    Directories,
    find_files,
    load_data,
)


def augment_fish_events(
    events: pd.DataFrame,
    directories: Directories,
    behavior_data: BehaviorData,
    behavior_files: BehaviorFiles,
    rollover_time_s: int,
) -> list[dict[str, Any]]:
    """Create contextual records for one recording's saccades."""
    recording = RecordingContext(
        directories=directories,
        behavior_files=behavior_files,
        behavior_data=behavior_data,
        rollover_time_s=rollover_time_s,
    )

    event_timestamps = (
        recording.relative_seconds_to_timestamps(
            events["onset_time_s"].to_numpy(dtype=float)
        )
    )

    trial_indices = recording.trial_indices(
        event_timestamps
    )

    tracking_context = recording.posthoc_positions_at(
        event_timestamps
    )

    contexts: list[dict[str, Any]] = []

    for event_index, (
        event_timestamp,
        trial_index,
    ) in enumerate(
        zip(event_timestamps, trial_indices)
    ):
        context = recording.event_context(
            event_timestamp=int(event_timestamp),
            trial_index=int(trial_index),
            x_mm=float(
                tracking_context["x_mm"][event_index]
            ),
            y_mm=float(
                tracking_context["y_mm"][event_index]
            ),
            heading=float(
                tracking_context["heading"][event_index]
            ),
        )

        context.update(
            {
                "tracking_frame": int(
                    tracking_context[
                        "tracking_frame"
                    ][event_index]
                ),
                "tracking_time_error_ms": float(
                    tracking_context[
                        "tracking_time_error_ms"
                    ][event_index]
                ),
            }
        )

        contexts.append(context)

    return contexts


def augment_saccades(
    input_csv: Path,
    output_csv: Path,
    directories: Directories,
    rollover_time_s: int,
) -> None:
    """Add common behavioral context to all classified saccades."""
    saccades = pd.read_csv(input_csv)

    required_columns = {
        "fish",
        "onset_time_s",
        "cluster",
    }
    missing_columns = required_columns.difference(
        saccades.columns
    )

    if missing_columns:
        raise ValueError(
            f"{input_csv} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    files_by_name = {
        files.metadata.stem: files
        for files in find_files(directories)
    }

    contexts_by_index: dict[int, dict[str, Any]] = {}

    for fish, fish_events in tqdm(
        saccades.groupby("fish", sort=False),
        desc="Saccade context",
    ):
        fish = str(fish)

        if fish not in files_by_name:
            raise FileNotFoundError(
                f"No BehaviorFiles were found for {fish}."
            )

        behavior_files = files_by_name[fish]
        behavior_data = load_data(behavior_files)

        contexts = augment_fish_events(
            events=fish_events,
            directories=directories,
            behavior_data=behavior_data,
            behavior_files=behavior_files,
            rollover_time_s=rollover_time_s,
        )

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
        raise RuntimeError(
            "The context table does not match the saccade table."
        )

    overlapping_columns = set(
        context_table.columns
    ).intersection(saccades.columns)

    if overlapping_columns:
        context_table = context_table.rename(
            columns={
                column: f"context_{column}"
                for column in overlapping_columns
            }
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
        mismatch = (
            augmented["fish"].astype(str)
            != augmented["file"].astype(str)
        )

        if mismatch.any():
            examples = augmented.loc[
                mismatch,
                ["fish", "file"],
            ].head()

            raise ValueError(
                "Saccade fish labels do not match recording files:\n"
                f"{examples}"
            )

    if "event_id" in augmented.columns:
        if augmented["event_id"].duplicated().any():
            raise ValueError(
                "Duplicate event_id values were found."
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
        default="classified_saccades.csv",
        help=(
            "Input classified-saccade CSV relative to root. "
            "Default: classified_saccades.csv."
        ),
    )
    parser.add_argument(
        "--output",
        default="augmented_saccades.csv",
        help=(
            "Output augmented-saccade CSV relative to root. "
            "Default: augmented_saccades.csv."
        ),
    )
    parser.add_argument(
        "--rollover-time-s",
        type=int,
        default=3600,
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
        raise FileNotFoundError(
            f"Classified saccade CSV does not exist: {input_csv}"
        )

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
        rollover_time_s=args.rollover_time_s,
    )


if __name__ == "__main__":
    main()