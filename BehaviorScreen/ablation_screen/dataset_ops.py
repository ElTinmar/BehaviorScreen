import numpy as np
from BehaviorScreen.point_process.dataset import PointProcessDataset

def select_fish(dataset: PointProcessDataset, fish_indices: np.ndarray) -> PointProcessDataset:
    """
    Build a new dataset containing ONLY the given (0-based, into `dataset`)
    fish indices, remapped to a fresh contiguous 0..k-1 range in the order
    given. Fish not in `fish_indices` are dropped entirely (not just masked).
    """
    fish_indices = np.asarray(fish_indices, dtype=int)
    remap = {int(old): new for new, old in enumerate(fish_indices)}

    mask = np.isin(dataset.event_fish_idx, fish_indices)
    if np.any(mask):
        new_fish_idx = np.array([remap[int(f)] for f in dataset.event_fish_idx[mask]], dtype=int)
        new_times = dataset.event_times[mask]
        new_trials = dataset.event_trials_idx[mask].astype(int)
    else:
        new_fish_idx = np.array([], dtype=int)
        new_times = np.array([], dtype=float)
        new_trials = np.array([], dtype=int)

    return PointProcessDataset(
        event_times=new_times,
        event_trials_idx=new_trials,
        event_fish_idx=new_fish_idx,
        fish_trial_mask=dataset.fish_trial_mask[fish_indices, :],
        fish_ids=dataset.fish_ids[fish_indices],
        duration_s=dataset.duration_s,
        binning_dt=dataset.binning_dt,
        bout_name=dataset.bout_name,
        laterality=dataset.laterality,
    )


def pool_fish(ds_a: PointProcessDataset, ds_b: PointProcessDataset) -> PointProcessDataset:
    """
    Concatenate two datasets' fish populations into one, with ds_b's fish
    re-indexed after ds_a's. Requires matching trial structure (same
    num_trials/duration_s/binning_dt) -- both datasets must come from the
    same behavior/stimulus protocol, just different line/condition subsets.
    """
    if ds_a.num_trials != ds_b.num_trials:
        raise ValueError(
            f"pool_fish: trial count mismatch ({ds_a.num_trials} vs {ds_b.num_trials}) -- "
            f"datasets must share the same trial structure (same behavior/protocol)."
        )
    if abs(ds_a.duration_s - ds_b.duration_s) > 1e-9:
        raise ValueError("pool_fish: duration_s mismatch between datasets.")

    n_a = ds_a.num_fish
    has_events = len(ds_a.event_fish_idx) or len(ds_b.event_fish_idx)
    pooled_fish_idx = (
        np.concatenate([ds_a.event_fish_idx.astype(int), (ds_b.event_fish_idx + n_a).astype(int)])
        if has_events else np.array([], dtype=int)
    )
    pooled_fish_ids = np.concatenate([ds_a.fish_ids, ds_b.fish_ids])
    
    return PointProcessDataset(
        event_times=np.concatenate([ds_a.event_times, ds_b.event_times]),
        event_trials_idx=np.concatenate([ds_a.event_trials_idx, ds_b.event_trials_idx]).astype(int),
        event_fish_idx=pooled_fish_idx,
        fish_trial_mask=np.vstack([ds_a.fish_trial_mask, ds_b.fish_trial_mask]),
        fish_ids=pooled_fish_ids,
        duration_s=ds_a.duration_s,
        binning_dt=ds_a.binning_dt,
        bout_name=ds_a.bout_name,
        laterality=ds_a.laterality,
    )