from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short

@dataclass(frozen=True)
class PointProcessDataset:
    event_times: np.ndarray             
    event_trials_idx: np.ndarray            
    event_fish_idx: np.ndarray          
    fish_trial_mask: np.ndarray

    unique_trials: np.ndarray 
    bout_name: str = ""
    laterality: str = ""
    duration_s: float = 24.0
    binning_dt: float = 0.02

    @property
    def num_fish(self) -> int:
        return self.fish_trial_mask.shape[0]

    @property
    def num_trials(self) -> int:
        return self.fish_trial_mask.shape[1]

    @property
    def t_grid(self) -> np.ndarray:
        return np.arange(0, self.duration_s + self.binning_dt, self.binning_dt)

    @property
    def t_centers(self) -> np.ndarray:
        return 0.5 * (self.t_grid[:-1] + self.t_grid[1:])

    @property
    def trial_edges(self) -> np.ndarray:
        return np.arange(self.num_trials + 1) - 0.5

    @property
    def n_fish_per_trial(self) -> np.ndarray:
        return self.fish_trial_mask.sum(axis=0).astype(float)

    @property
    def time_histogram_counts(self) -> np.ndarray:
        counts, _ = np.histogram(self.event_times, bins=self.t_grid)
        return counts

    @property
    def time_histogram_hz(self) -> np.ndarray:
        total_exposure_time = np.sum(self.n_fish_per_trial) * self.binning_dt
        if total_exposure_time == 0:
            return np.zeros_like(self.time_histogram_counts, dtype=float)
        return self.time_histogram_counts / total_exposure_time

    @property
    def time_trial_histogram_counts(self) -> np.ndarray:
        trial_edges = np.append(self.unique_trials - 0.5, self.unique_trials[-1] + 0.5)
        counts, _, _ = np.histogram2d(
            self.event_trials_idx,
            self.event_times,
            bins=[trial_edges, self.t_grid]
        )
        return counts

    @property
    def time_trial_histogram_hz(self) -> np.ndarray:
        counts = self.time_trial_histogram_counts
        n_fish = self.n_fish_per_trial[:, None]
        safe_denom = np.where(n_fish > 0, n_fish * self.binning_dt, 1.0)
        return np.where(n_fish > 0, counts / safe_denom, 0.0)
    
    def resample(self, rng: np.random.Generator) -> 'PointProcessDataset':
        n_fish = self.fish_trial_mask.shape[0]
        boot_fish_idx = rng.choice(n_fish, size=n_fish, replace=True)

        boot_times = []
        boot_trials = []
        boot_fish = []

        for new_id, orig_id in enumerate(boot_fish_idx):
            mask = (self.event_fish_idx == orig_id)
            if np.any(mask):
                boot_times.append(self.event_times[mask])
                boot_trials.append(self.event_trials_idx[mask])
                boot_fish.append(np.full(np.sum(mask), new_id, dtype=int))

        return PointProcessDataset(
            event_times=np.concatenate(boot_times) if boot_times else np.array([]),
            event_trials_idx=np.concatenate(boot_trials) if boot_trials else np.array([]),
            event_fish_idx=np.concatenate(boot_fish) if boot_fish else np.array([]),
            fish_trial_mask=self.fish_trial_mask[boot_fish_idx, :],
            duration_s=self.duration_s,
            binning_dt=self.binning_dt,
            bout_name=self.bout_name,
            laterality=self.laterality,
            unique_trials=self.unique_trials
        )


class BehavioralDataLoader:
    def __init__(self, csv_path: Union[Path, str]):
        self.raw_df = pd.read_csv(csv_path)

    def prepare_dataset(
        self,
        bout_name: str,
        laterality: Union[Laterality, str],
        stim: Optional[Union[Stim, str]] = None,
        epoch_name: Optional[Union[str, List[str]]] = None,
        binning_dt: float = 0.02,
        t_start: float = 0.0,
        t_end: float = 24.0,
    ) -> PointProcessDataset:

        # 1. Filter sub_df by stimulus or epoch_name
        sub_df = self.raw_df
        if stim is not None:
            sub_df = sub_df[sub_df['stim'] == stim]
            if stim == Stim.PHOTOTAXIS or stim == 'phototaxis':
                sub_df = sub_df[sub_df['foreground_color'] == '[0.1, 0.1, 0.0, 1.0]']
        elif epoch_name is not None:
            if isinstance(epoch_name, list):
                sub_df = sub_df[sub_df['epoch_name'].isin(epoch_name)]
            else:
                sub_df = sub_df[sub_df['epoch_name'] == epoch_name]
        else:
            raise ValueError("Either 'stim' or 'epoch_name' must be provided to filter dataset.")

        # 2. Extract metadata & build integer index mappings
        all_fish_ids = np.sort(sub_df['file'].unique())
        unique_trials = np.sort(sub_df['trial_num'].unique()) 

        fish_map = {f_id: idx for idx, f_id in enumerate(all_fish_ids)}
        trial_map = {t_num: idx for idx, t_num in enumerate(unique_trials)}

        # 3. Build (N_fish, N_trials) boolean presence matrix
        n_fish = len(all_fish_ids)
        n_trials = len(unique_trials)
        fish_trial_mask = np.zeros((n_fish, n_trials), dtype=bool)

        active_pairs = sub_df[['file', 'trial_num']].drop_duplicates()
        f_indices = active_pairs['file'].map(fish_map).values
        t_indices = active_pairs['trial_num'].map(trial_map).values
        fish_trial_mask[f_indices, t_indices] = True

        # 4. Filter target events
        bout_idx = bouts_category_name_short.index(bout_name)
        is_target_event = (
            (sub_df['category'] == bout_idx) & 
            (sub_df['laterality'] == laterality)
        )
        event_mask = (
            is_target_event & 
            (sub_df['trial_time'] >= t_start) & 
            (sub_df['trial_time'] < t_end)
        )
        events = sub_df[event_mask]

        # 5. Extract event arrays as integer indices & floats
        event_times = (events['trial_time'] - t_start).values.astype(float)
        event_trials_idx = events['trial_num'].map(trial_map).values.astype(int)
        event_fish_idx = events['file'].map(fish_map).values.astype(int)

        return PointProcessDataset(
            event_times=event_times,
            event_trials_idx=event_trials_idx,
            event_fish_idx=event_fish_idx,
            fish_trial_mask=fish_trial_mask,
            duration_s=t_end-t_start,
            binning_dt=binning_dt,
            bout_name=bout_name,
            laterality=str(laterality),
            unique_trials=unique_trials,
        )
