from dataclasses import dataclass
from pathlib import Path
from itertools import groupby
from functools import cached_property
from typing import List, Optional, Union, Tuple, ClassVar, Dict
import numpy as np
import pandas as pd
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short
from scipy.special import gammaln   
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

@dataclass(frozen=True)
class PointProcessDataset:

    _LOW_POWER_MEAN_COUNT_THRESHOLD: ClassVar[float] = 1.0

    event_times: np.ndarray             
    event_trials_idx: np.ndarray            
    event_fish_idx: np.ndarray          
    fish_trial_mask: np.ndarray

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

    @cached_property
    def _stream_index(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        One-time grouping of event times by (fish_idx, trial_idx), cached on
        this dataset instance. iter_streams() is called inside _nll(), which
        scipy.optimize.minimize() may evaluate thousands of times per fit() --
        without this cache, every call re-scans the full event arrays for
        every (fish, trial) pair. Computed once, reused for the life of this
        (immutable) dataset.
        """
        order = np.lexsort((self.event_times, self.event_trials_idx, self.event_fish_idx))
        keys = list(zip(self.event_fish_idx[order], self.event_trials_idx[order]))
        times = self.event_times[order]

        result = {}
        start = 0
        for key, group in groupby(keys):
            n = len(list(group))
            result[key] = times[start : start + n]  # already sorted by time, within group
            start += n
        return result

    def iter_streams(self, active_only: bool = True):
        """
        Yields (fish_idx, trial_idx, sorted_event_times) for every (fish, trial)
        pair in the dataset. If active_only=True (default), skips (fish, trial)
        pairs not marked present in fish_trial_mask.
        """
        empty = np.array([], dtype=float)
        for f_idx in range(self.num_fish):
            for t_idx in range(self.num_trials):
                if active_only and not self.fish_trial_mask[f_idx, t_idx]:
                    continue
                yield f_idx, t_idx, self._stream_index.get((f_idx, t_idx), empty)

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
        counts, _, _ = np.histogram2d(
            self.event_trials_idx,
            self.event_times,
            bins=[self.trial_edges, self.t_grid]
        )
        return counts

    @property
    def time_trial_histogram_hz(self) -> np.ndarray:
        counts = self.time_trial_histogram_counts
        n_fish = self.n_fish_per_trial[:, None]
        safe_denom = np.where(n_fish > 0, n_fish * self.binning_dt, 1.0)
        return np.where(n_fish > 0, counts / safe_denom, 0.0)


    @property
    def stream_event_counts(self) -> np.ndarray:
        """Event count per (fish, trial) stream."""
        return np.array([len(t_ev) for _, _, t_ev in self.iter_streams()])

    @property
    def fish_total_counts(self) -> np.ndarray:
        """Total event count per fish, summed across all trials (active fish only)."""
        totals = np.zeros(self.num_fish)
        for f_idx, t_idx, t_ev in self.iter_streams():
            totals[f_idx] += len(t_ev)
        return totals[self.fish_trial_mask.any(axis=1)]

    @property
    def stream_fano_factor(self) -> float:
        """Variance/mean of stream_event_counts (a.k.a. dispersion index)."""
        counts = self.stream_event_counts
        mean = np.mean(counts) if len(counts) else np.nan
        return (np.var(counts) / mean) if mean > 0 else np.nan

    @property
    def fish_fano_factor(self) -> float:
        """Variance/mean of fish_total_counts."""
        totals = self.fish_total_counts
        mean = np.mean(totals) if len(totals) else np.nan
        return (np.var(totals) / mean) if mean > 0 else np.nan

    @property
    def dispersion_fano_ratio(self) -> float:
        """
        fish_fano_factor / stream_fano_factor.

        >> 1 (with is_low_power_for_dispersion == False): between-fish rate
          heterogeneity dominates -> a fish-level gain/random-effect term
          is warranted.
        ~1 or < 1: heterogeneity is not the main driver; if stream_fano_factor
          is itself far from 1, investigate within-stream temporal structure
          (refractory / clustering) via PointProcess.time_rescaling instead.
        """
        stream_ff = self.stream_fano_factor
        fish_ff = self.fish_fano_factor
        return fish_ff / stream_ff if (not np.isnan(stream_ff) and stream_ff > 0) else np.nan

    @property
    def frac_streams_with_multiple_events(self) -> float:
        """Fraction of (fish, trial) streams with >= 2 events (i.e. contribute to ISI stats)."""
        counts = self.stream_event_counts
        return float(np.mean(counts >= 2)) if len(counts) else np.nan

    @property
    def is_low_power_for_dispersion(self) -> bool:
        """
        True when mean event count per stream is low enough that Fano-factor
        comparisons carry little statistical power (most streams are
        empty/singleton). Prefer PointProcess.time_rescaling diagnostics for
        such conditions instead.
        """
        counts = self.stream_event_counts
        mean = np.mean(counts) if len(counts) else np.nan
        return bool(mean < self._LOW_POWER_MEAN_COUNT_THRESHOLD) if not np.isnan(mean) else True

    @property
    def stream_isi_cv(self) -> np.ndarray:
        """
        Coefficient of variation (std/mean) of inter-event intervals, computed
        separately for each (fish, trial) stream with >= 2 events.

        Scale-invariant: a stream's overall rate does not affect its CV, so
        this isolates within-stream ISI SHAPE (axis 2) from between-stream
        RATE heterogeneity (axis 1) -- unlike the pooled ISI histogram, which
        confounds the two.

        CV ~ 1   : consistent with a Poisson/exponential-ISI process.
        CV < 1   : more regular than Poisson -> refractory / rhythmic dynamics.
        CV > 1   : burstier than Poisson -> clustering / self-excitation.
        """
        cvs = []
        for _, _, t_ev in self.iter_streams():
            if len(t_ev) < 3:  # need >= 2 ISIs for a meaningful std
                continue
            isis = np.diff(t_ev)
            if np.mean(isis) > 0:
                cvs.append(np.std(isis) / np.mean(isis))
        return np.array(cvs)

    @property
    def mean_isi_cv(self) -> float:
        """Population mean of stream_isi_cv, with NaN if no eligible streams."""
        cvs = self.stream_isi_cv
        return float(np.mean(cvs)) if len(cvs) > 0 else np.nan

    @property
    def stream_isi_lag1_autocorr(self) -> float:
        """
        Lag-1 autocorrelation of raw ISIs, computed WITHIN each (fish, trial)
        stream separately then pooled.

        Sign convention matches PointProcess.time_rescaling's ACF panel:
        positive -> bursty/rhythmic trains; negative -> refractory alternation.
        """
        cov_sum, pair_count, all_isis = 0.0, 0, []

        for _, _, t_ev in self.iter_streams():
            if len(t_ev) < 3:
                continue
            isis = np.diff(t_ev)
            all_isis.append(isis)

        if not all_isis:
            return np.nan

        pooled_mean = np.mean(np.concatenate(all_isis))
        pooled_var = np.var(np.concatenate(all_isis))
        if pooled_var == 0:
            return np.nan

        for isis in all_isis:
            centered = isis - pooled_mean
            if len(centered) > 1:
                cov_sum += np.sum(centered[:-1] * centered[1:])
                pair_count += len(centered) - 1

        return (cov_sum / (pair_count * pooled_var)) if pair_count > 0 else np.nan

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
            event_times=np.concatenate(boot_times) if boot_times else np.array([], dtype=float),
            event_trials_idx=np.concatenate(boot_trials) if boot_trials else np.array([], dtype=int),
            event_fish_idx=np.concatenate(boot_fish) if boot_fish else np.array([], dtype=int),
            fish_trial_mask=self.fish_trial_mask[boot_fish_idx, :],
            duration_s=self.duration_s,
            binning_dt=self.binning_dt,
            bout_name=self.bout_name,
            laterality=self.laterality,
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

        n_trials = len(unique_trials)
        expected = np.arange(n_trials)
        if not np.array_equal(unique_trials, expected):
            raise ValueError(
                f"'trial_num' is expected to be a zero-based, contiguous index "
                f"(0..{n_trials - 1}) after filtering by stim/epoch_name, but got "
                f"unique values {unique_trials.tolist()}. This usually means a "
                f"trial had zero surviving rows after filtering, breaking "
                f"contiguity. PointProcessDataset and every RateKernel/HistoryKernel "
                f"assume trial_num IS the positional trial index"
            )

        fish_map = {f_id: idx for idx, f_id in enumerate(all_fish_ids)}

        # 3. Build (N_fish, N_trials) boolean presence matrix
        n_fish = len(all_fish_ids)
        fish_trial_mask = np.zeros((n_fish, n_trials), dtype=bool)

        active_pairs = sub_df[['file', 'trial_num']].drop_duplicates()
        f_indices = active_pairs['file'].map(fish_map).values
        t_indices = active_pairs['trial_num'].values
        fish_trial_mask[f_indices, t_indices] = True

        # KNOWN LIMITATION (not fixed -- believed marginal, revisit if dispersion/
        # rate estimates look off for low-count conditions, e.g. SLC/looming):
        #
        # `active_pairs` is derived from `sub_df`, which at this point is filtered
        # only by stim/epoch_name -- it still contains one row per detected bout
        # EVENT of ANY category/laterality. So a (fish, trial) pair is only marked
        # present here if that fish produced >=1 bout of ANY kind that trial.
        #
        # A fish that was genuinely tracked/present but emitted ZERO bouts of every
        # category that trial (froze, sub-threshold movement, or just a true zero
        # under a low base rate -- more likely in short-duration conditions like
        # looming) is indistinguishable here from a fish that was never tracked
        # (protocol aborted, lost tracking). Both produce zero rows in sub_df, so
        # both get fish_trial_mask=False.
        #
        # Effect: this silently excludes true zero-count exposures from the risk
        # set, which biases every fitted rate (PoissonProcess/HawkesProcess/...)
        # UPWARD, and biases stream_fano_factor/fish_fano_factor DOWNWARD (dropped
        # zeros shrink variance relative to mean) -- i.e. it can make the dataset
        # look less overdispersed than it really is. Effect size scales with how
        # often a genuinely-present fish has an all-category zero-bout trial, so
        # it's expected to be worse for low base-rate / short-duration conditions
        # (e.g. looming, where trials are shorter) than for long, high-rate ones.
        #
        # Correct fix (not applied): build presence from an independent
        # tracking/participation record, not from bout rows. 

        occupancy = fish_trial_mask.mean()
        min_fish_per_trial = fish_trial_mask.sum(axis=0).min()
        min_trials_per_fish = fish_trial_mask.sum(axis=1).min()
        print(
            f"[{bout_name}/{laterality}] fish_trial_mask occupancy: {occupancy:.1%} "
            f"({n_fish} fish x {n_trials} trials); "
            f"min fish/trial = {min_fish_per_trial}, min trials/fish = {min_trials_per_fish}"
        )

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
        event_trials_idx = events['trial_num'].values.astype(int)
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

    # POINT PROCESS PLOTS ------

    @staticmethod
    def _pooled_isis(dataset: PointProcessDataset) -> np.ndarray:
        """Pooled inter-event intervals across every observed (fish, trial) stream."""
        all_isis = [
            np.diff(t_ev)
            for _, _, t_ev in dataset.iter_streams()
            if len(t_ev) > 1
        ]
        return np.concatenate(all_isis) if all_isis else np.array([], dtype=float)

    @staticmethod
    def _plot_count_distribution(
        counts: np.ndarray,
        ax: plt.Axes,
        xlabel: str,
        title_prefix: str,
    ) -> None:
        """
        Shared rendering logic for 'observed count histogram vs. reference
        pmfs' plots, used at both the stream level and fish level.

        Overlays two MODEL-FREE reference distributions (both computed directly
        from the observed counts via simple moment-matching -- no optimizer,
        consistent with this class running BEFORE any model fitting):

        - Poisson(mean): null hypothesis of no extra-Poisson variability.
        - NegBinom(mean, var): method-of-moments fit -- the count-level analog
        of GammaPoissonProcess's marginal distribution. If this tracks the
        histogram much better than Poisson, that's an early, fit-free signal
        that GammaPoissonProcess-style heterogeneity is worth trying (see
        PointProcessDataset.dispersion_fano_ratio for the matching numeric
        diagnostic).

        CAVEAT (fish-level use only): for plot_fish_total_count_distribution,
        fish contribute totals summed over DIFFERENT numbers of trials
        (fish_trial_mask occupancy is not uniform -- some fish have far fewer
        observed trials than others). This overlay's moment-matched r does NOT
        account for that unequal exposure, unlike GammaPoissonProcess's actual
        fitted r (which uses per-fish exposure S_f explicitly, see
        GammaPoissonProcess._fish_sufficient_stats). Some of the apparent
        overdispersion in the fish-total-count histogram may therefore reflect
        unequal trial counts across fish rather than true rate heterogeneity --
        treat this overlay as a rough visual guide, not a substitute for the
        model's own fitted r_dispersion.
        """
        if len(counts) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

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
            ax.plot(k_vals, poisson_pmf, 'r--', linewidth=2,
                label=f'Poisson(mean={mean_count:.2f})')

        # NB only well-defined when var > mean (overdispersed relative to
        # Poisson). If var <= mean, moment-matching would require r <= 0
        # (undefined) -- in that regime Poisson is already the relevant
        # reference, so NB is simply omitted rather than clamped/faked.
        if mean_count > 0 and var_count > mean_count:
            r_mom = mean_count**2 / (var_count - mean_count)
            p_mom = r_mom / (r_mom + mean_count)
            log_nb_pmf = (
                gammaln(k_vals + r_mom) - gammaln(r_mom) - gammaln(k_vals + 1)
                + r_mom * np.log(p_mom) + k_vals * np.log(1 - p_mom)
            )
            ax.plot(k_vals, np.exp(log_nb_pmf), 'g-.', linewidth=2,
                label=f'NegBinom(mean={mean_count:.2f}, r={r_mom:.2f})')

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"{title_prefix}  |  Mean={mean_count:.2f}, "
            f"Var={var_count:.2f}, Dispersion Index={dispersion_index:.2f}",
            fontsize=11, fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)


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


    @staticmethod
    def plot_event_count_distribution(
        dataset: PointProcessDataset,
        figsize: Tuple[int, int] = (8, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Histogram of event counts per (fish, trial) STREAM, overlaid with a
        Poisson pmf of matching mean.

        Dispersion Index (variance / mean):
        - ~1.0  : consistent with Poisson (no strong extra structure)
        - >> 1.0: overdispersed -- could reflect within-stream clustering/
                  bursting (self-excitation) OR between-fish rate
                  heterogeneity OR both. See `plot_fish_total_count_distribution`
                  to help distinguish between these.
        - << 1.0: underdispersed -- regularity / refractoriness.
        """
        counts = np.array([len(t_ev) for _, _, t_ev in dataset.iter_streams()])

        fig, ax = plt.subplots(figsize=figsize)
        DatasetPlotter._plot_count_distribution(
            counts, ax,
            xlabel="Event count per (fish, trial) stream",
            title_prefix="Event Count Distribution (per stream)",
        )
        plt.tight_layout()
        return fig, ax


    @staticmethod
    def plot_fish_total_count_distribution(
        dataset: PointProcessDataset,
        figsize: Tuple[int, int] = (8, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Histogram of TOTAL event counts per fish (summed across all trials),
        overlaid with a Poisson pmf of matching mean.

        Companion to `plot_event_count_distribution`, used to distinguish
        two distinct sources of overdispersion at the stream level:

        - If dispersion stays high (or increases) here after aggregating
          across trials, it points to genuine between-fish rate
          heterogeneity (some fish are just more/less active than others)
          -- this calls for a fish-level gain/random-effect term in the
          rate model, NOT a history-dependent (Hawkes/Renewal) kernel.

        - If dispersion drops substantially toward ~1.0 here relative to
          the per-stream plot, the stream-level overdispersion is more
          likely driven by within-stream temporal clustering, which
          averages out across many trials -- pointing toward a
          history-dependent kernel instead.
        """
        fish_totals = np.zeros(dataset.num_fish)
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            fish_totals[f_idx] += len(t_ev)

        # Only include fish observed in at least one trial
        active_fish_mask = dataset.fish_trial_mask.any(axis=1)
        fish_totals = fish_totals[active_fish_mask]

        fig, ax = plt.subplots(figsize=figsize)
        DatasetPlotter._plot_count_distribution(
            fish_totals, ax,
            xlabel="Total event count per fish (summed over trials)",
            title_prefix=f"Total Event Count Distribution (N = {len(fish_totals)} fish)",
        )
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_fish_activity_heatmap(
        dataset: PointProcessDataset,
        cmap: str = "viridis",
        sort_by: Optional[str] = "total_count",  # "total_count", "rate", or None
        min_row_height_in: float = 0.02,
        figsize_width: float = 10,
        figsize_height_cap: float = 40.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        (fish x trial) matrix of event counts. Unobserved (fish, trial) pairs
        (per fish_trial_mask) are shown in gray, not zero, so missing data is
        visually distinct from "fish present but did not respond."

        Useful for spotting dead/inactive fish, missing trial blocks, or
        fish-level outliers before they silently dominate a pooled fit.

        NOTE on readability with many fish / few trials (common shape for this
        data: hundreds of fish x a handful of trials): with sort_by=None, rows
        are in arbitrary fish-index order and structure only shows up as
        apparent "static" once there are more fish than vertical pixels --
        the eye can't tell "this fish is inactive" from "this row got
        antialiased into its neighbor". Default sort_by="total_count" instead
        orders fish by total activity (descending, active fish only, inactive
        fish trailing at the bottom labeled separately), which turns real
        between-fish heterogeneity (see PointProcessDataset.dispersion_fano_ratio,
        plot_fish_total_count_distribution) into a visible smooth gradient
        instead of noise. Figure height also scales with num_fish (up to
        figsize_height_cap) so each row gets a minimum real pixel allotment
        instead of being silently downsampled by matplotlib.

        For datasets with many hundreds of fish, also consider
        plot_fish_rank_activity, which shows the same per-fish totals without
        an axis that has to be spatially legible per-row.
        """
        counts_matrix = np.full((dataset.num_fish, dataset.num_trials), np.nan)
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            counts_matrix[f_idx, t_idx] = len(t_ev)

        active_fish_mask = dataset.fish_trial_mask.any(axis=1)
        fish_order = np.arange(dataset.num_fish)

        if sort_by is not None:
            totals = np.nansum(counts_matrix, axis=1)
            n_active_trials = dataset.fish_trial_mask.sum(axis=1)
            if sort_by == "rate":
                with np.errstate(invalid='ignore', divide='ignore'):
                    score = np.where(n_active_trials > 0, totals / n_active_trials, -np.inf)
            else:  # "total_count"
                score = np.where(active_fish_mask, totals, -np.inf)
            # active fish sorted descending by score, inactive fish trail at bottom
            active_idx = np.where(active_fish_mask)[0]
            inactive_idx = np.where(~active_fish_mask)[0]
            active_sorted = active_idx[np.argsort(-score[active_idx])]
            fish_order = np.concatenate([active_sorted, inactive_idx])

        counts_matrix = counts_matrix[fish_order]

        fig_height = min(figsize_height_cap, max(4.0, dataset.num_fish * min_row_height_in))
        fig, ax = plt.subplots(figsize=(figsize_width, fig_height))
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='lightgray')

        masked = np.ma.masked_invalid(counts_matrix)
        im = ax.imshow(masked, aspect='auto', cmap=cmap_obj, origin='lower',
                        interpolation='nearest')

        ax.set_xlabel("Trial Index", fontsize=11)
        ylabel = "Fish (sorted by activity, desc.)" if sort_by is not None else "Fish Index"
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title("Event Count per Fish x Trial (gray = not observed)",
                    fontsize=12, fontweight='bold')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(im, cax=cax, label="Event Count")

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_fish_rank_activity(
        dataset: PointProcessDataset,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Per-fish total event count, fish sorted by rank (descending), with each
        fish's per-trial counts overlaid as individual points colored by trial.

        Designed for the common shape of this data (hundreds of fish, a
        handful of trials) where plot_fish_activity_heatmap's 2D matrix has far
        more rows than can be made individually legible. This plot instead uses
        a rank axis: the black step line shows total-count heterogeneity across
        fish (same information as plot_fish_total_count_distribution's long
        right tail, but preserving identity/rank so you can see e.g. whether
        the top ~5% of fish are individually much more active or whether it's
        a smooth continuum), and the colored points show, per fish, whether its
        per-trial counts are consistent (tight vertical scatter) or itself
        highly variable trial-to-trial (spread-out scatter) -- a fish-level
        analog of dataset.stream_isi_cv's "is variability within-unit or
        across-unit" question, but for rate rather than ISI shape.
        """
        active_fish_mask = dataset.fish_trial_mask.any(axis=1)
        active_fish_idx = np.where(active_fish_mask)[0]

        per_trial_counts = np.full((dataset.num_fish, dataset.num_trials), np.nan)
        for f_idx, t_idx, t_ev in dataset.iter_streams():
            per_trial_counts[f_idx, t_idx] = len(t_ev)

        totals = np.nansum(per_trial_counts[active_fish_idx], axis=1)
        order = np.argsort(-totals)
        ranked_fish_idx = active_fish_idx[order]
        ranks = np.arange(1, len(ranked_fish_idx) + 1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.step(ranks, totals[order], where='mid', color='black', linewidth=1.5,
                label='Total count per fish', zorder=3)

        norm = plt.Normalize(vmin=0, vmax=max(dataset.num_trials - 1, 1))
        base_cmap = plt.get_cmap(cmap)
        for rank, f_idx in zip(ranks, ranked_fish_idx):
            trial_counts = per_trial_counts[f_idx]
            observed = ~np.isnan(trial_counts)
            t_idxs = np.where(observed)[0]
            ax.scatter(np.full(t_idxs.shape, rank), trial_counts[t_idxs],
                    c=[base_cmap(norm(t)) for t in t_idxs], s=8, alpha=0.6, zorder=2)

        ax.set_xlabel("Fish rank (by total activity, descending)", fontsize=11)
        ax.set_ylabel("Event count", fontsize=11)
        ax.set_title(f"Fish Activity by Rank (N = {len(ranked_fish_idx)} active fish)",
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)

        sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.1)
        fig.colorbar(sm, cax=cax, label="Trial Index")

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_psth(
        dataset: PointProcessDataset,
        bin_width_s: float = 0.5,
        band: str = "sem",  # "sem" or "std"
        figsize: Tuple[int, int] = (10, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Population PSTH: mean event rate across fish as a function of time,
        coarse-binned for visualization (raw binning_dt is typically fine
        enough that most bins contain 0/1 events per trial), with a band
        showing trial-to-trial variability (SEM or STD across trials).

        Run this before fitting anything. Look for:
        - A transient bump/dip locked to stimulus onset -> needs a RateKernel,
        not a constant baseline rate.
        - A flat trial-averaged curve -> a constant/slowly-varying baseline
        may suffice.
        - A wide band relative to the mean -> substantial trial-to-trial
        variability in the *time course* itself, which a single RateKernel
        shared across all trials won't capture (cross-check against
        plot_time_trial_rate_heatmap to see if that variability is
        structured, e.g. a trend across trials, vs. just noise).
        """
        factor = max(1, int(round(bin_width_s / dataset.binning_dt)))
        counts = dataset.time_trial_histogram_counts  # (n_trials, n_fine_bins)
        n_coarse = counts.shape[1] // factor
        if n_coarse == 0:
            raise ValueError("bin_width_s too large relative to duration_s/binning_dt")
        trimmed = counts[:, : n_coarse * factor]
        coarse_counts = trimmed.reshape(dataset.num_trials, n_coarse, factor).sum(axis=2)
        coarse_centers = dataset.t_centers[: n_coarse * factor].reshape(n_coarse, factor).mean(axis=1)
        coarse_dt = factor * dataset.binning_dt

        n_fish_per_trial = dataset.n_fish_per_trial
        active = n_fish_per_trial > 0
        rate_per_trial = np.full_like(coarse_counts, np.nan, dtype=float)
        rate_per_trial[active] = coarse_counts[active] / (n_fish_per_trial[active, None] * coarse_dt)

        mean_rate = np.nanmean(rate_per_trial, axis=0)
        n_active_trials = int(np.sum(active))
        spread = np.nanstd(rate_per_trial, axis=0)
        if band == "sem":
            spread = spread / np.sqrt(max(n_active_trials, 1))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(coarse_centers, mean_rate, color='steelblue', linewidth=2, label='Mean rate across trials')
        ax.fill_between(coarse_centers, mean_rate - spread, mean_rate + spread,
                        color='steelblue', alpha=0.3,
                        label=f'±1 {band.upper()} (across {n_active_trials} trials)')
        ax.set_xlabel("Time in trial (s)", fontsize=11)
        ax.set_ylabel("Rate (events/s per fish)", fontsize=11)
        ax.set_title(f"Population PSTH (bin={bin_width_s}s)", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_time_trial_rate_heatmap(
        dataset: PointProcessDataset,
        bin_width_s: float = 0.5,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        (trial x time) heatmap of event rate, coarse-binned in time for a
        readable signal.

        Complements plot_psth: reveals whether the time course itself (not
        just its level) changes across trials -- e.g. a stimulus-locked
        response habituating/sharpening over the session, or a tonic
        component only appearing in later trials. Either motivates a
        trial-index interaction on the RateKernel rather than one kernel
        shared across all trials. Trials with no observed fish are masked
        gray, distinguishing "no data" from "zero rate".
        """
        factor = max(1, int(round(bin_width_s / dataset.binning_dt)))
        counts = dataset.time_trial_histogram_counts
        n_coarse = counts.shape[1] // factor
        if n_coarse == 0:
            raise ValueError("bin_width_s too large relative to duration_s/binning_dt")
        trimmed = counts[:, : n_coarse * factor]
        coarse_counts = trimmed.reshape(dataset.num_trials, n_coarse, factor).sum(axis=2)
        coarse_edges = dataset.t_grid[: n_coarse * factor + 1 : factor]
        coarse_dt = factor * dataset.binning_dt

        n_fish_per_trial = dataset.n_fish_per_trial
        active = n_fish_per_trial > 0
        rate_matrix = np.full_like(coarse_counts, np.nan, dtype=float)
        rate_matrix[active] = coarse_counts[active] / (n_fish_per_trial[active, None] * coarse_dt)

        fig, ax = plt.subplots(figsize=figsize)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='lightgray')
        masked = np.ma.masked_invalid(rate_matrix)

        mesh = ax.pcolormesh(coarse_edges, dataset.trial_edges, masked, shading='flat', cmap=cmap_obj)
        ax.set_xlabel("Time in trial (s)", fontsize=11)
        ax.set_ylabel("Trial Index", fontsize=11)
        ax.set_title(f"Rate (Hz) by Time x Trial (bin={bin_width_s}s, gray = no fish observed)",
                    fontsize=12, fontweight='bold')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(mesh, cax=cax, label="Rate (Hz)")
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_trial_occupancy(
        dataset: PointProcessDataset,
        figsize: Tuple[int, int] = (9, 4),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Number of fish observed (per fish_trial_mask) in each trial.

        Flags attrition across the session (lost tracking, or the
        "zero-bout-of-any-category" masking gap noted in
        BehavioralDataLoader.prepare_dataset). Trials with few observed fish
        give unreliable rate estimates in plot_psth / plot_time_trial_rate_heatmap
        and should be discounted accordingly.
        """
        n_fish_per_trial = dataset.n_fish_per_trial
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(np.arange(dataset.num_trials), n_fish_per_trial, color='slategray')
        ax.axhline(dataset.num_fish, color='crimson', linestyle='--', linewidth=1,
                label=f'Total fish = {dataset.num_fish}')
        ax.set_xlabel("Trial Index", fontsize=11)
        ax.set_ylabel("# fish observed", fontsize=11)
        ax.set_title("Fish Occupancy per Trial", fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_fano_by_time_bin(
        dataset: PointProcessDataset,
        bin_width_s: float = 1.0,
        figsize: Tuple[int, int] = (9, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Fano factor (variance/mean of per-stream counts), computed
        independently within each time bin, across all active (fish, trial)
        streams.

        Complements dataset.stream_fano_factor (one number over the whole
        trial) by localizing WHERE overdispersion happens. A spike right at
        stimulus onset (fish reacting in near-lockstep) looks very different
        from uniformly-elevated dispersion throughout, and calls for a
        different fix (a stimulus-locked synchrony term vs. a fish-level
        gain/random-effect, i.e. GammaPoissonProcess).
        """
        edges = np.arange(0, dataset.duration_s + bin_width_s, bin_width_s)
        centers = 0.5 * (edges[:-1] + edges[1:])

        rows = [np.histogram(t_ev, bins=edges)[0] for _, _, t_ev in dataset.iter_streams()]
        stream_counts = np.array(rows) if rows else np.zeros((0, len(centers)))

        mean_per_bin = stream_counts.mean(axis=0) if len(stream_counts) else np.array([])
        var_per_bin = stream_counts.var(axis=0) if len(stream_counts) else np.array([])
        fano_per_bin = np.divide(var_per_bin, mean_per_bin,
                                out=np.full(len(centers), np.nan), where=mean_per_bin > 0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(centers, fano_per_bin, marker='o', color='darkorange')
        ax.axhline(1.0, color='k', linestyle='--', linewidth=1, label='Poisson (=1)')
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Fano factor (var/mean of stream counts)", fontsize=11)
        ax.set_title(f"Time-resolved Dispersion (bin={bin_width_s}s)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        return fig, ax

    # SURVIVAL PLOTS ------

    @staticmethod
    def plot_kaplan_meier(
        dataset: PointProcessDataset,
        ci: float = 95.0,
        figsize: Tuple[int, int] = (8, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Model-free empirical survival curve S(t) = P(no response by t), pooling
        every (fish, trial) stream as an independent right-censored first-event
        observation (subsequent events in a stream are ignored -- exactly the
        reduction SurvivalProcess uses; see frac_streams_with_multiple_events
        below for whether that's a safe reduction for this condition).

        Run BEFORE fitting anything. Look for:
        - A clear plateau well below 1.0 -> most fish never respond; the
        plateau height is your model-free target for H(duration_s).
        - A plateau near 0.0 -> nearly everyone responds; a constant-hazard
        (exponential) kernel may already be adequate, no bump needed.
        - Greenwood CI band very wide at late t -> few fish still at risk
        there; don't over-interpret tail shape.
        """
        times, censored = [], []
        for _, _, t_ev in dataset.iter_streams():
            if len(t_ev) == 0:
                times.append(dataset.duration_s); censored.append(True)
            else:
                times.append(float(np.min(t_ev))); censored.append(False)
        times, censored = np.array(times), np.array(censored)

        order = np.argsort(times)
        t_sorted, c_sorted = times[order], censored[order]
        n = len(t_sorted)

        grid, surv, var_sum = [0.0], [1.0], [0.0]
        S, V = 1.0, 0.0
        i = 0
        while i < n:
            t = t_sorted[i]
            j = i
            d = 0
            while j < n and t_sorted[j] == t:
                if not c_sorted[j]:
                    d += 1
                j += 1
            n_at_risk = n - i
            if n_at_risk > 0 and d > 0:
                S *= (1.0 - d / n_at_risk)
                if n_at_risk > d:
                    V += d / (n_at_risk * (n_at_risk - d))
                grid.append(float(t)); surv.append(S); var_sum.append(V)
            i = j

        grid, surv, var_sum = np.array(grid), np.array(surv), np.array(var_sum)
        se = surv * np.sqrt(var_sum)  # Greenwood's formula
        z = norm.ppf(0.5 + ci / 200.0)
        lower = np.clip(surv - z * se, 0.0, 1.0)
        upper = np.clip(surv + z * se, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.step(grid, surv, where='post', color='black', linewidth=2, label='Kaplan-Meier $\\hat{S}(t)$')
        ax.fill_between(grid, lower, upper, step='post', color='steelblue', alpha=0.25,
                        label=f'{ci:.0f}% CI (Greenwood)')
        ax.axhline(surv[-1], color='crimson', linestyle='--', linewidth=1,
                label=f'Plateau = {surv[-1]:.2f} (frac. never responding)')
        ax.set_xlabel("Time in trial (s)"); ax.set_ylabel("P(no response by t)")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Empirical Survival Curve (N={n} streams, {int(np.sum(~censored))} responders)",
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9); ax.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_repeat_event_gap(
        dataset: PointProcessDataset,
        bins: int = 40,
        figsize: Tuple[int, int] = (8, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        For streams with >=2 events, the gap between the 1st and 2nd event.
        Model-free sanity check on the absorption assumption underlying
        SurvivalProcess / RenewalKernelFactory.hard_absorption(): a gap
        distribution piled up near zero (tight, short-latency repeats) argues
        the 'first event only' reduction is throwing away a real, structured
        second response (bad for absorption); a gap distribution that looks
        flat/uniform over the remaining trial window is more consistent with
        unrelated background activity that's fine to discard.
        """
        gaps = [np.diff(np.sort(t_ev))[0] for _, _, t_ev in dataset.iter_streams() if len(t_ev) >= 2]
        n_multi = len(gaps)
        n_total = len(dataset.stream_event_counts)

        fig, ax = plt.subplots(figsize=figsize)
        if n_multi == 0:
            ax.text(0.5, 0.5, f"No streams with >=2 events\n(0 / {n_total})",
                    ha='center', va='center')
        else:
            ax.hist(gaps, bins=bins, color='indianred', alpha=0.75, edgecolor='none')
            ax.set_xlabel("Gap: 1st -> 2nd event (s)")
            ax.set_ylabel("Count")
        ax.set_title(f"1st-to-2nd Event Gap ({n_multi}/{n_total} = "
                    f"{n_multi/n_total:.1%} of streams affected)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_response_by_trial(
        dataset: PointProcessDataset,
        figsize: Tuple[int, int] = (10, 5),
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Per trial: (top) fraction of streams that responded, with a binomial
        CI band; (bottom) latency of responders as a scatter, censored streams
        omitted from the scatter but counted in the top panel.

        A declining top-panel trend -> habituation of RESPONSE PROBABILITY
        (motivates an alpha/habituation term on H). A rising/falling bottom-
        panel trend -> habituation of LATENCY, a distinct effect the
        Gaussian-bump kernels here don't currently capture (mu is fit
        trial-constant) -- if this trend is visible, that's a concrete,
        data-driven argument for adding trial-dependence to mu, not just H.
        """
        n_resp = np.zeros(dataset.num_trials)
        n_obs = np.zeros(dataset.num_trials)
        latencies = {t: [] for t in range(dataset.num_trials)}

        for f_idx, t_idx, t_ev in dataset.iter_streams():
            n_obs[t_idx] += 1
            if len(t_ev) > 0:
                n_resp[t_idx] += 1
                latencies[t_idx].append(float(np.min(t_ev)))

        with np.errstate(invalid='ignore', divide='ignore'):
            p_hat = np.where(n_obs > 0, n_resp / n_obs, np.nan)
            se = np.where(n_obs > 0, np.sqrt(p_hat * (1 - p_hat) / n_obs), np.nan)

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        trials = np.arange(dataset.num_trials)
        ax_top.errorbar(trials, p_hat, yerr=1.96 * se, fmt='o-', color='darkorange', capsize=3)
        ax_top.set_ylabel("P(respond)"); ax_top.set_ylim(0, 1.02)
        ax_top.set_title("Response Probability & Latency by Trial", fontsize=12, fontweight='bold')
        ax_top.grid(True, linestyle=':', alpha=0.4)

        for t_idx, lats in latencies.items():
            if lats:
                ax_bot.scatter(np.full(len(lats), t_idx), lats, color='steelblue', alpha=0.5, s=12)
        ax_bot.set_xlabel("Trial Index"); ax_bot.set_ylabel("Latency (s)")
        ax_bot.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        return fig, (ax_top, ax_bot)

    @staticmethod
    def plot_fish_response_rate_distribution(
        dataset: PointProcessDataset,
        figsize: Tuple[int, int] = (8, 5),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Per fish: fraction of its observed trials in which it responded at all
        (ignores WHEN within the trial -- pure yes/no), vs. a Binomial(n_f, p_pop)
        null built from the pooled population response rate. Overdispersion
        relative to that binomial null (variance clearly exceeding p(1-p)/n)
        is the survival-analysis analog of dataset.dispersion_fano_ratio, and
        is the model-free argument for GammaMixedEffectsProcess(SurvivalProcess(...))
        over a plain SurvivalProcess.
        """
        resp_frac, n_trials_obs = [], []
        for f_idx in range(dataset.num_fish):
            trial_idxs = np.where(dataset.fish_trial_mask[f_idx])[0]
            if len(trial_idxs) == 0:
                continue
            n_resp = sum(
                1 for t_idx in trial_idxs
                if len(dataset._stream_index.get((f_idx, t_idx), np.array([]))) > 0
            )
            resp_frac.append(n_resp / len(trial_idxs))
            n_trials_obs.append(len(trial_idxs))

        resp_frac = np.array(resp_frac)
        p_pop = np.mean(resp_frac) if len(resp_frac) else np.nan

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(resp_frac, bins=np.linspace(0, 1, 21), density=True, alpha=0.65,
                color='mediumseagreen', edgecolor='none', label='Observed per-fish response rate')
        if np.isfinite(p_pop):
            mean_n = np.mean(n_trials_obs)
            se_null = np.sqrt(p_pop * (1 - p_pop) / mean_n)
            x = np.linspace(0, 1, 200)
            ax.plot(x, norm.pdf(x, p_pop, se_null), 'r--', linewidth=2,
                    label=f'Binomial null (p={p_pop:.2f}, mean n={mean_n:.1f})')
        ax.set_xlabel("Fraction of trials responded"); ax.set_ylabel("Density")
        ax.set_title(f"Per-Fish Response Rate Heterogeneity (N={len(resp_frac)} fish)",
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        return fig, ax