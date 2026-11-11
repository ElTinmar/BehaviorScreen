"""The loader, from the merged tables of a small screen to the datasets the models are fitted to."""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.core import Stim
from BehaviorScreen.screen.loader import BehavioralDataLoader

# presentation order of the two epochs of the stimulus: left 0, right 0, left 1, right 1
ORDER = {("left", 0): 10, ("right", 0): 11, ("left", 1): 12, ("right", 1): 13}
ARMS = {"A": "vehicle", "B": "vehicle", "C": "ronidazole", "D": "vehicle", "E": "vehicle"}
UNUSABLE = {("B", "left", 1): "tracking_ok", ("B", "right", 1): "presented"}
# file, epoch, trial_num, trial_time, category, laterality, distance_center
EVENTS = [
    ("A", "left", 0, 1.0, 0, 1, 2.0), ("A", "left", 0, 30.0, 0, 1, 2.0), ("A", "right", 0, 2.0, 1, -1, 2.0), ("A", "left", 1, 3.0, 0, 1, 20.0),
    ("A", "right", 1, 4.0, 0, np.nan, 2.0), ("B", "left", 0, 0.5, 0, -1, 2.0), ("B", "left", 1, 5.0, 0, 1, 2.0),
    ("D", "left", 0, 1.0, 0, 1, 2.0), ("E", "left", 0, 1.0, 0, 1, 2.0),
]


def write_screen(folder: Path) -> Dict[str, List[Path]]:
    """Five fish of line L1: D is listed in the QC table, E is 6 dpf; B has two unusable trials."""
    trials = pd.DataFrame([
        {"file": fish, "line": "L1", "condition": arm, "dpf": 6 if fish == "E" else 7, "stim": int(Stim.OMR), "epoch_name": epoch,
         "trial_num": trial_num, "epoch_idx": index, "trial_duration_s": 24.0,
         "presented": UNUSABLE.get((fish, epoch, trial_num)) != "presented", "tracking_ok": UNUSABLE.get((fish, epoch, trial_num)) != "tracking_ok"}
        for fish, arm in ARMS.items() for (epoch, trial_num), index in ORDER.items()
    ])
    events = pd.DataFrame(EVENTS, columns=["file", "epoch_name", "trial_num", "trial_time", "category", "laterality", "distance_center"])
    events = events.assign(line="L1", condition=events["file"].map(ARMS), dpf=np.where(events["file"] == "E", 6, 7))
    paths = {"events": folder / "bouts_all.csv", "trials": folder / "valid_trials_all.csv", "qc": folder / "qc_all.csv"}
    events.to_csv(paths["events"], index=False)
    trials.to_csv(paths["trials"], index=False)
    pd.DataFrame({"file": ["D"], "reason": ["not_moving"]}).to_csv(paths["qc"], index=False)
    return paths


@pytest.fixture
def loader(tmp_path: Path) -> BehavioralDataLoader:
    paths = write_screen(tmp_path)
    with pytest.warns(UserWarning, match="1 events without category or laterality"):
        return BehavioralDataLoader(paths["events"], [paths["trials"]], [paths["qc"]],
                                    event_filters={"distance_center": {"<=": 10}}, recording_filters={"dpf": {"==": 7}})


def test_streams_are_the_usable_trials_in_presentation_order(loader: BehavioralDataLoader) -> None:
    data = loader.prepare_dataset(stim=Stim.OMR, t_end=24.0)
    assert list(data.fish_ids) == ["A", "B", "C"]  # D: quality control; E: recording filter
    assert data.fish_trial_mask.tolist() == [[True] * 4, [True, True, False, False], [True] * 4]
    assert data.duration_s == 24.0


def test_events_are_those_of_usable_trials_within_the_window(loader: BehavioralDataLoader) -> None:
    """Gone: the bout after the window, the one too far from the center, the one without a side, the one of an untracked trial."""
    data = loader.prepare_dataset(stim=Stim.OMR, t_end=24.0)
    found = sorted(zip(data.fish_ids[data.event_fish_idx], data.event_trials_idx, data.event_times, data.marks["category"], data.marks["laterality"]))
    assert found == [("A", 0, 1.0, 0, 1), ("A", 1, 2.0, 1, -1), ("B", 0, 0.5, 0, -1)]


def test_selected_events_keep_every_bout_as_history(loader: BehavioralDataLoader) -> None:
    data = loader.prepare_dataset(category=[0], laterality=[1], stim=Stim.OMR, t_end=24.0)
    assert (data.event_times.tolist(), data.event_fish_idx.tolist()) == ([1.0], [0])
    assert sorted(data.history.event_times) == [0.5, 1.0, 2.0]
    assert np.array_equal(data.history.fish_trial_mask, data.fish_trial_mask)


def test_window_start_shifts_event_times(loader: BehavioralDataLoader) -> None:
    data = loader.prepare_dataset(stim=Stim.OMR, t_start=1.0, t_end=24.0)
    assert sorted(data.event_times) == [0.0, 1.0] and data.duration_s == 23.0


def test_one_epoch_has_its_own_trials(loader: BehavioralDataLoader) -> None:
    data = loader.prepare_dataset(epoch_name="left", t_end=24.0)
    assert data.fish_trial_mask.tolist() == [[True, True], [True, False], [True, True]]
    assert sorted(zip(data.event_fish_idx, data.event_trials_idx)) == [(0, 0), (1, 0)]


def test_subset_keeps_the_trial_indices_of_the_whole_screen(loader: BehavioralDataLoader) -> None:
    drug = loader.subset(conditions=["ronidazole"]).prepare_dataset(stim=Stim.OMR, t_end=24.0)
    assert list(drug.fish_ids) == ["C"] and drug.fish_trial_mask.shape == (1, 4) and len(drug.event_times) == 0
    vehicle = loader.subset(line="L1", conditions=["vehicle"]).prepare_dataset(stim=Stim.OMR, t_end=24.0)
    assert list(vehicle.fish_ids) == ["A", "B"]
    assert loader.subset(exclude_lines=["L1"]).trials.empty


def test_trial_listed_twice_is_refused(tmp_path: Path) -> None:
    paths = write_screen(tmp_path)
    trials = pd.read_csv(paths["trials"])
    pd.concat([trials, trials.iloc[:1]]).to_csv(paths["trials"], index=False)
    with pytest.raises(ValueError, match="twice"):
        BehavioralDataLoader(paths["events"], [paths["trials"]])


def test_loader_reads_recording_times():
    loader = object.__new__(BehavioralDataLoader)
    hours = {"a": 9.25, "b": 17.5}
    angle = lambda h: 2 * np.pi * h / 24
    loader.events = pd.DataFrame({"file": ["a", "a", "b"], "cos_daytime": [np.cos(angle(hours[f])) for f in "aab"],
                                  "sin_daytime": [np.sin(angle(hours[f])) for f in "aab"]})
    assert loader._fish_hours(np.array(["b", "a", "c"])) == pytest.approx([17.5, 9.25, np.nan], nan_ok=True)
    loader.events = loader.events.drop(columns=["cos_daytime", "sin_daytime"])
    assert loader._fish_hours(np.array(["a"])) is None
