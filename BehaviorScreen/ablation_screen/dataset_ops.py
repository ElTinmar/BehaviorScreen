from dataclasses import replace

import numpy as np

from BehaviorScreen.point_process.point_process.dataset import PointProcessDataset


def select_fish(dataset: PointProcessDataset, fish_indices: np.ndarray) -> PointProcessDataset:
    """
    Only the given fish (0-based, into `dataset`), renumbered 0..k-1 in the
    order given; the other fish are dropped entirely.
    """
    fish_indices = np.asarray(fish_indices, dtype=int)
    new_index = np.full(dataset.num_fish, -1)
    new_index[fish_indices] = np.arange(len(fish_indices))
    keep = new_index[dataset.event_fish_idx] >= 0
    return replace(
        dataset,
        event_times=dataset.event_times[keep],
        event_trials_idx=dataset.event_trials_idx[keep],
        event_fish_idx=new_index[dataset.event_fish_idx[keep]],
        fish_trial_mask=dataset.fish_trial_mask[fish_indices],
        fish_ids=dataset.fish_ids[fish_indices],
        event_marks=None if dataset.event_marks is None else dataset.event_marks[keep],
    )


def pool_fish(ds_a: PointProcessDataset, ds_b: PointProcessDataset) -> PointProcessDataset:
    """ds_a's fish followed by ds_b's; both must come from the same condition."""
    if ds_a.num_trials != ds_b.num_trials or ds_a.duration_s != ds_b.duration_s:
        raise ValueError("pool_fish: datasets differ in trials or duration")
    marks = (ds_a.event_marks, ds_b.event_marks)
    return replace(
        ds_a,
        event_times=np.concatenate([ds_a.event_times, ds_b.event_times]),
        event_trials_idx=np.concatenate([ds_a.event_trials_idx, ds_b.event_trials_idx]),
        event_fish_idx=np.concatenate([ds_a.event_fish_idx, ds_b.event_fish_idx + ds_a.num_fish]),
        fish_trial_mask=np.vstack([ds_a.fish_trial_mask, ds_b.fish_trial_mask]),
        fish_ids=np.concatenate([ds_a.fish_ids, ds_b.fish_ids]),
        event_marks=None if any(m is None for m in marks) else np.concatenate(marks),
    )
