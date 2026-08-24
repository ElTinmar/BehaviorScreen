from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short
from scipy.special import gammaln   
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    def iter_streams(self, active_only: bool = True):
        """
        Yields (fish_idx, trial_idx, sorted_event_times) for every (fish, trial)
        pair in the dataset.
        If active_only=True (default), skips (fish, trial) pairs not marked
        present in fish_trial_mask.
        """
        for f_idx in range(self.num_fish):
            for t_idx in range(self.num_trials):
                if active_only and not self.fish_trial_mask[f_idx, t_idx]:
                    continue
                mask = (self.event_fish_idx == f_idx) & (self.event_trials_idx == t_idx)
                t_ev = np.sort(self.event_times[mask])
                yield f_idx, t_idx, t_ev

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

class DatasetPlotter:
    """
    Model-free diagnostic visualizations of a PointProcessDataset.

    These plots use only raw event data (no fitted model required) and are
    intended to be run BEFORE any model fitting, to catch structural
    features -- refractoriness, rhythmicity, overdispersion, dead fish,
    missing trials -- that would otherwise only surface indirectly through
    a fitted model's residual diagnostics.
    """

    @staticmethod
    def _pooled_isis(dataset: PointProcessDataset) -> np.ndarray:
        """Pooled inter-event intervals across every observed (fish, trial) stream."""
        all_isis = [
            np.diff(t_ev)
            for _, _, t_ev in dataset.iter_streams()
            if len(t_ev) > 1
        ]
        return np.concatenate(all_isis) if all_isis else np.array([], dtype=float)

    # -- Panel 1: Pooled ISI histogram --------------------------------------

    @staticmethod
    def plot_isi_histogram(
        dataset: PointProcessDataset,
        bins: int = 100,
        max_isi: Optional[float] = None,
        log_y: bool = False,
        figsize: Tuple[int, int] = (8, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Pooled inter-event-interval distribution across all fish/trials.

        What to look for:
        - A visible near-zero gap/dead-zone  -> refractory period.
        - A mode away from zero (a "preferred" ISI) -> rhythmic bout trains
          (a bump-shaped, not monotonic, history kernel would be needed).
        - Comb-like multiple modes (peaks at tau, 2*tau, ...) -> periodicity.
        - Excess density near zero relative to an exponential tail -> bursting.
        """
        isis = DatasetPlotter._pooled_isis(dataset)
        fig, ax = plt.subplots(figsize=figsize)

        if len(isis) == 0:
            ax.text(0.5, 0.5, "No inter-event intervals available\n(need >= 2 events per stream)",
                    ha='center', va='center')
            ax.set_title("Pooled ISI Distribution", fontsize=12, fontweight='bold')
            return fig, ax

        plot_isis = isis[isis <= max_isi] if max_isi is not None else isis

        ax.hist(plot_isis, bins=bins, color='steelblue', edgecolor='none', alpha=0.85)

        median_isi = np.median(isis)
        ax.axvline(median_isi, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'Median ISI = {median_isi:.3f}s')

        if log_y:
            ax.set_yscale('log')

        ax.set_xlabel("Inter-event interval (s)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Pooled ISI Distribution (N = {len(isis)} intervals)",
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)

        plt.tight_layout()
        return fig, ax

    # -- Panel 2: ISI distribution across trials ----------------------------

    @staticmethod
    def plot_isi_by_trial(
        dataset: PointProcessDataset,
        bins: int = 60,
        max_isi: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (9, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        ISI density as a function of trial number: a (trial x ISI-bin) heatmap.

        Reveals whether temporal fine-structure (refractoriness, rhythmicity)
        itself changes across trials -- e.g., bout rhythm tightening or
        habituating with repeated stimulus exposure -- which a trial-only
        rate-modulation term (alpha_B, alpha_dip, etc.) cannot capture.
        """
        per_trial_isis: Dict[int, List[np.ndarray]] = {t_idx: [] for t_idx in range(dataset.num_trials)}
        for _, t_idx, t_ev in dataset.iter_streams():
            if len(t_ev) > 1:
                per_trial_isis[t_idx].append(np.diff(t_ev))

        pooled = DatasetPlotter._pooled_isis(dataset)
        fig, ax = plt.subplots(figsize=figsize)

        if len(pooled) == 0:
            ax.text(0.5, 0.5, "No inter-event intervals available", ha='center', va='center')
            return fig, ax

        upper = max_isi if max_isi is not None else np.percentile(pooled, 99)
        bin_edges = np.linspace(0, upper, bins + 1)

        density_matrix = np.full((dataset.num_trials, bins), np.nan)
        for t_idx in range(dataset.num_trials):
            if not per_trial_isis[t_idx]:
                continue
            isis_t = np.concatenate(per_trial_isis[t_idx])
            counts, _ = np.histogram(isis_t, bins=bin_edges, density=True)
            density_matrix[t_idx, :] = counts

        mesh = ax.pcolormesh(
            bin_edges, np.arange(dataset.num_trials + 1) - 0.5, density_matrix,
            shading='flat', cmap=cmap,
        )
        ax.set_xlabel("Inter-event interval (s)", fontsize=11)
        ax.set_ylabel("Trial Number", fontsize=11)
        ax.set_title("ISI Density by Trial", fontsize=12, fontweight='bold')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(mesh, cax=cax, label="Density")

        plt.tight_layout()
        return fig, ax

    # -- Panel 3: Raw event raster -------------------------------------------

    @staticmethod
    def plot_raw_raster(
        dataset: PointProcessDataset,
        fish_subset: Optional[np.ndarray] = None,
        max_fish: int = 15,
        cmap: str = "viridis",
        marker_size: float = 4.0,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Classic event raster: one row per (fish, trial) stream, grouped by fish,
        colored by trial index. Useful for spotting individual-fish idiosyncrasies
        (e.g. one hyperactive fish dominating pooled statistics), gross
        artifacts, or qualitative bursting/rhythmicity patterns by eye.

        `fish_subset` lets you pick specific fish indices; otherwise the first
        `max_fish` fish (by index) are shown to keep the plot readable.
        """
        if fish_subset is None:
            fish_subset = np.arange(min(max_fish, dataset.num_fish))
        fish_subset_set = set(int(f) for f in fish_subset)

        norm = plt.Normalize(vmin=0, vmax=max(dataset.num_trials - 1, 1))
        base_cmap = plt.get_cmap(cmap)

        fig, ax = plt.subplots(figsize=figsize)

        row = 0
        y_ticks, y_labels = [], []
        row_gap = 1  # blank row between fish blocks
        current_fish = None
        fish_start_row = 0

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            if f_idx not in fish_subset_set:
                continue

            if f_idx != current_fish:
                if current_fish is not None:
                    y_ticks.append(0.5 * (fish_start_row + row - 1))
                    y_labels.append(f"Fish {current_fish}")
                    row += row_gap
                current_fish = f_idx
                fish_start_row = row

            if len(t_ev) > 0:
                color = base_cmap(norm(t_idx))
                ax.scatter(t_ev, np.full_like(t_ev, row, dtype=float),
                          marker='|', s=marker_size * 20, color=color, linewidths=1.0)
            row += 1

        if current_fish is not None:
            y_ticks.append(0.5 * (fish_start_row + row - 1))
            y_labels.append(f"Fish {current_fish}")

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel("Time in Trial (s)", fontsize=11)
        ax.set_title(f"Raw Event Raster ({len(fish_subset_set)} fish, colored by trial)",
                    fontsize=12, fontweight='bold')
        ax.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])
        ax.invert_yaxis()

        sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.15)
        fig.colorbar(sm, cax=cax, label="Trial Index")

        plt.tight_layout()
        return fig, ax

    # -- Panel 4: Event count distribution / overdispersion check -----------

    @staticmethod
    def plot_event_count_distribution(
        dataset: PointProcessDataset,
        figsize: Tuple[int, int] = (8, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Histogram of event counts per (fish, trial) stream, overlaid with a
        Poisson pmf of matching mean.

        Reports the index of dispersion (variance / mean):
        - ~1.0  : consistent with a Poisson process (no strong history effects)
        - >> 1.0: overdispersed -> clustering / bursting / self-excitation
                  (motivates a Hawkes-type additive history kernel)
        - << 1.0: underdispersed -> regularity / refractoriness
                  (motivates a renewal-type multiplicative suppression kernel)

        This is a fast, fit-free first signal for which process family to
        reach for, before running full time-rescaling diagnostics.
        """
        counts = np.array([len(t_ev) for _, _, t_ev in dataset.iter_streams()])

        fig, ax = plt.subplots(figsize=figsize)

        if len(counts) == 0:
            ax.text(0.5, 0.5, "No observed (fish, trial) streams", ha='center', va='center')
            return fig, ax

        mean_count = np.mean(counts)
        var_count = np.var(counts)
        dispersion_index = var_count / mean_count if mean_count > 0 else np.nan

        max_count = int(counts.max())
        bin_edges = np.arange(-0.5, max_count + 1.5, 1.0)
        ax.hist(counts, bins=bin_edges, density=True, alpha=0.65,
               color='steelblue', edgecolor='none', label='Observed')

        k_vals = np.arange(0, max_count + 1)
        if mean_count > 0:
            poisson_pmf = np.exp(
                k_vals * np.log(mean_count) - mean_count - gammaln(k_vals + 1)
            )
        else:
            poisson_pmf = np.zeros_like(k_vals, dtype=float)
        ax.plot(k_vals, poisson_pmf, 'r--', linewidth=2,
               label=f'Poisson(mean={mean_count:.2f})')

        ax.set_xlabel("Event count per (fish, trial) stream", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"Event Count Distribution  |  Mean={mean_count:.2f}, "
            f"Var={var_count:.2f}, Dispersion Index={dispersion_index:.2f}",
            fontsize=11, fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)

        plt.tight_layout()
        return fig, ax

    # -- Panel 5: Fish x trial activity heatmap ------------------------------

    @staticmethod
    def plot_fish_activity_heatmap(
        dataset: PointProcessDataset,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        (fish x trial) matrix of event counts. Unobserved (fish, trial) pairs
        (per fish_trial_mask) are shown in gray, not zero, so missing data is
        visually distinct from "fish present but did not respond."

        Useful for spotting dead/inactive fish, missing trial blocks, or
        fish-level outliers before they silently dominate a pooled fit.
        """
        counts_matrix = np.full((dataset.num_fish, dataset.num_trials), np.nan)
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            counts_matrix[f_idx, t_idx] = len(t_ev)

        fig, ax = plt.subplots(figsize=figsize)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='lightgray')

        masked = np.ma.masked_invalid(counts_matrix)
        im = ax.imshow(masked, aspect='auto', cmap=cmap_obj, origin='lower')

        ax.set_xlabel("Trial Index", fontsize=11)
        ax.set_ylabel("Fish Index", fontsize=11)
        ax.set_title("Event Count per Fish x Trial (gray = not observed)",
                    fontsize=12, fontweight='bold')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(im, cax=cax, label="Event Count")

        plt.tight_layout()
        return fig, ax