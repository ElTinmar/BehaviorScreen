#!/usr/bin/env python3
"""Add stimulus and trial context to classified saccades."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.core import Stim
from BehaviorScreen.load import (
    BehaviorData,
    BehaviorFiles,
    Directories,
    encode_time_of_day,
    find_files,
    load_data,
    parse_fish,
)
from BehaviorScreen.process import get_trials
from BehaviorScreen.stimulus import (
    looming_constant_velocity_approach,
    prey_capture_arc_stimulus_cosine,
)


def scalar_for_csv(value: Any) -> Any:
    """Convert a trial-table value into a CSV-compatible scalar."""
    if value is None:
        return np.nan

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (str, int, float, bool)):
        return value

    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass

    if hasattr(value, "value"):
        return value.value

    return str(value)


def prepare_trial_table(
    behavior_data: BehaviorData,
) -> pd.DataFrame:
    """
    Add the same epoch-local trial numbering used by the bout pipeline.
    """
    trials = get_trials(behavior_data)

    if trials.empty:
        return trials.copy()

    records = []

    for epoch_name, epoch_trials in trials.groupby(
        "epoch_name",
        sort=False,
    ):
        for trial_num, (epoch_index, row) in enumerate(
            epoch_trials.iterrows()
        ):
            record = row.to_dict()
            record["epoch_name"] = epoch_name
            record["epoch_idx"] = epoch_index
            record["trial_num"] = trial_num
            records.append(record)

    prepared = pd.DataFrame.from_records(records)

    return prepared.sort_values(
        "start_timestamp",
        kind="stable",
    ).reset_index(drop=True)


def find_trial_indices(
    event_timestamps: np.ndarray,
    trial_starts: np.ndarray,
    trial_stops: np.ndarray,
) -> np.ndarray:
    """
    Return the trial containing each event, or -1 when outside all trials.
    """
    indices = np.searchsorted(
        trial_starts,
        event_timestamps,
        side="right",
    ) - 1

    valid = (
        (indices >= 0)
        & (indices < len(trial_starts))
    )

    safe_indices = np.clip(
        indices,
        0,
        max(len(trial_starts) - 1, 0),
    )

    if len(trial_starts):
        valid &= (
            event_timestamps
            < trial_stops[safe_indices]
        )

    indices[~valid] = -1

    return indices


def calculate_stimulus_values(
    trial: pd.Series,
    trial_time: float,
    rollover_time_s: int,
) -> dict[str, float]:
    """Calculate time-dependent stimulus values at saccade onset."""
    result = {
        "stim_phase": np.nan,
        "looming_radius": np.nan,
    }

    stimulus = trial.get("stim_select", None)

    if stimulus == Stim.PREY_CAPTURE:
        result["stim_phase"] = (
            prey_capture_arc_stimulus_cosine(
                trial.start_time_sec,
                trial_time,
                rollover_time_s,
                trial.prey_arc_start_deg,
                trial.prey_arc_stop_deg,
                trial.prey_speed_deg_s,
            )
        )

    if stimulus == Stim.LOOMING:
        result["looming_radius"] = (
            looming_constant_velocity_approach(
                trial.start_time_sec,
                trial_time,
                rollover_time_s,
                trial.looming_angle_start_deg,
                trial.looming_angle_stop_deg,
                trial.looming_size_to_speed_ratio_ms,
                trial.looming_distance_to_screen_mm,
            )
        )

    return result


def augment_fish_events(
    events: pd.DataFrame,
    behavior_data: BehaviorData,
    behavior_files: BehaviorFiles,
    rollover_time_s: int,
) -> list[dict[str, Any]]:
    """Create stimulus-context records for one recording."""
    fish = behavior_files.metadata.stem
    fish_info = parse_fish(fish)
    time_cosine, time_sine = encode_time_of_day(fish_info)

    trials = prepare_trial_table(behavior_data)

    if trials.empty:
        raise ValueError(f"No stimulus trials were found for {fish}.")

    video_timestamps = (
        behavior_data.video_timestamps.timestamp.to_numpy()
    )

    if len(video_timestamps) == 0:
        raise ValueError(f"No video timestamps were found for {fish}.")

    recording_start = int(video_timestamps[0])

    event_timestamps = (
        recording_start
        + np.rint(
            events["onset_time_s"].to_numpy(dtype=float)
            * 1e9
        ).astype(np.int64)
    )

    trial_starts = trials[
        "start_timestamp"
    ].to_numpy(dtype=np.int64)
    trial_stops = trials[
        "stop_timestamp"
    ].to_numpy(dtype=np.int64)

    trial_indices = find_trial_indices(
        event_timestamps,
        trial_starts,
        trial_stops,
    )

    first_trial_start = int(trial_starts.min())
    output = []

    for event_timestamp, trial_index in zip(
        event_timestamps,
        trial_indices,
    ):
        context: dict[str, Any] = {
            "file": fish,
            "dpf": fish_info.age,
            "day": (
                f"{fish_info.day}."
                f"{fish_info.month}."
                f"{fish_info.year}"
            ),
            "cos_daytime": float(time_cosine),
            "sin_daytime": float(time_sine),
            "event_timestamp": int(event_timestamp),
            "in_stimulus_trial": trial_index >= 0,
            "stim": np.nan,
            "epoch_name": np.nan,
            "epoch_idx": np.nan,
            "trial_num": np.nan,
            "trial_time": np.nan,
            "stim_start_time": np.nan,
            "stim_phase": np.nan,
            "looming_radius": np.nan,
        }

        if trial_index < 0:
            output.append(context)
            continue

        trial = trials.iloc[int(trial_index)]
        trial_time = (
            event_timestamp - int(trial.start_timestamp)
        ) * 1e-9

        context.update(
            {
                "stim": scalar_for_csv(
                    trial.get("stim_select", np.nan)
                ),
                "epoch_name": scalar_for_csv(
                    trial.epoch_name
                ),
                "epoch_idx": scalar_for_csv(
                    trial.epoch_idx
                ),
                "trial_num": int(trial.trial_num),
                "trial_time": float(trial_time),
                "stim_start_time": (
                    int(trial.start_timestamp)
                    - first_trial_start
                )
                * 1e-9,
            }
        )

        # Copy stimulus parameters using the same names as bouts.csv.
        excluded = {
            "start_timestamp",
            "stop_timestamp",
            "epoch_name",
            "epoch_idx",
            "trial_num",
            "stim_select",
        }

        for column, value in trial.items():
            if column in excluded:
                continue

            if column not in context:
                context[column] = scalar_for_csv(value)

        try:
            context.update(
                calculate_stimulus_values(
                    trial=trial,
                    trial_time=trial_time,
                    rollover_time_s=rollover_time_s,
                )
            )
        except (AttributeError, TypeError, ValueError):
            pass

        output.append(context)

    return output


def augment_saccades(
    input_csv: Path,
    output_csv: Path,
    directories: Directories,
    rollover_time_s: int,
) -> None:
    """Add stimulus context to all classified saccades."""
    saccades = pd.read_csv(input_csv)

    required = {
        "fish",
        "onset_time_s",
        "cluster",
    }
    missing = required.difference(saccades.columns)

    if missing:
        raise ValueError(
            f"{input_csv} is missing columns: {sorted(missing)}"
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

        files = files_by_name[fish]
        behavior_data = load_data(files)

        contexts = augment_fish_events(
            events=fish_events,
            behavior_data=behavior_data,
            behavior_files=files,
            rollover_time_s=rollover_time_s,
        )

        for index, context in zip(
            fish_events.index,
            contexts,
        ):
            contexts_by_index[int(index)] = context

    context_table = pd.DataFrame.from_dict(
        contexts_by_index,
        orient="index",
    ).reindex(saccades.index)

    overlapping = set(context_table).intersection(
        saccades.columns
    )

    # `fish` and `file` deliberately coexist. Any other overlap is
    # renamed to avoid silently replacing classification columns.
    overlapping.discard("file")

    if overlapping:
        context_table = context_table.rename(
            columns={
                column: f"context_{column}"
                for column in overlapping
            }
        )

    augmented = pd.concat(
        [
            saccades.reset_index(drop=True),
            context_table.reset_index(drop=True),
        ],
        axis=1,
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
    print(
        "Events assigned to a trial: "
        f"{augmented['in_stimulus_trial'].mean():.1%}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Add stimulus and trial context to classified saccades."
        )
    )

    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--input",
        default="classified_saccades.csv",
        help="Classified saccade CSV.",
    )
    parser.add_argument(
        "--output",
        default="augmented_saccades.csv",
    )
    parser.add_argument(
        "--rollover-time-s",
        type=int,
        default=3600,
    )

    parser.add_argument("--metadata", default="results")
    parser.add_argument("--stimuli", default="results")
    parser.add_argument("--tracking", default="results")
    parser.add_argument("--lightning-pose",default="lightning_pose")
    parser.add_argument("--temperature", default="results")
    parser.add_argument("--video", default="results")
    parser.add_argument("--video-timestamp", default="results")
    parser.add_argument("--results", default="results")
    parser.add_argument("--plots", default="plots")

    return parser


def main() -> None:
    """Run saccade augmentation."""
    args = build_parser().parse_args()

    input_csv = args.root / args.input
    output_csv = args.root / args.output

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Classified saccade CSV does not exist: {input_csv}"
        )

    directories = Directories(
        args.root,
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