from typing import List, Tuple, Generator, Any, NamedTuple
import argparse
from pathlib import Path
import re
from dataclasses import dataclass
import operator
import textwrap
import warnings

import yaml
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm
from megabouts.utils import bouts_category_name_short

from BehaviorScreen.core import Stim
from BehaviorScreen.load import (
    base_regexp,
    FileNameInfo,
    Directories,
    BehaviorData,
    BehaviorFiles,
    load_data,
    find_files
)
from BehaviorScreen.process import (
    get_trials,
    compute_angle_between_vectors,
    get_target_time,
    interpolate_ts
)


def pd_series_in(s: pd.Series, v: Any) -> pd.Series:
    return s.isin(v)


def pd_series_not_in(s: pd.Series, v: Any) -> pd.Series:
    return ~s.isin(v)


_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
    "in": pd_series_in,
    "not_in": pd_series_not_in,
}


@dataclass
class Rule:
    column: str
    operator: str
    value: Any

    def get_mask(self, df: pd.DataFrame) -> pd.Series:
        op_func = _OPS[self.operator]
        return op_func(df[self.column], self.value)


@dataclass
class RuleSet:
    rules: tuple[Rule, ...]

    def get_mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for rule in self.rules:
            mask &= rule.get_mask(df)
        return mask

    def __repr__(self):
        if not self.rules:
            return "all"
        return "_".join([f"{r.column}{r.operator}{r.value}" for r in self.rules])


@dataclass
class StimSpec:
    stim: Stim
    trials: range
    name: str
    time_range: Tuple[float, float] | None
    parameters: List[RuleSet]

    def get_mask(self, df: pd.DataFrame) -> pd.Series:
        if not self.parameters:
            return pd.Series(True, index=df.index)
        mask = pd.Series(False, index=df.index)
        for ruleset in self.parameters:
            mask |= ruleset.get_mask(df)
        return mask

    def __repr__(self) -> str:
        params = " | ".join(str(p) for p in self.parameters)
        return f"{self.name}[{params}]"


class EyesTimeseries(NamedTuple):
    timestamps: np.ndarray
    angle_left_deg: np.ndarray
    angle_right_deg: np.ndarray
    angle_left_smooth_deg: np.ndarray
    angle_right_smooth_deg: np.ndarray
    version_angle_deg: np.ndarray
    vergence_angle_deg: np.ndarray


def load_bouts(bout_csv: Path) -> pd.DataFrame:
    return pd.read_csv(bout_csv)


def parse_rules(cfg: dict) -> RuleSet:
    rules = []
    for column, rule_dict in (cfg or {}).items():
        for op_name, value in rule_dict.items():
            rules.append(Rule(column, op_name, value))
    return RuleSet(tuple(rules))


def filter_bouts(quality_control: Path, bouts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # TODO: should the filtered bouts be counted as NaNs instead of
    # just being removed?

    filtered = bouts.copy()
    n0 = len(filtered)
    print(f'TOTAL NUM BOUTS: {n0}')

    if quality_control.exists():
        qc = pd.read_csv(quality_control)
        before = len(filtered)
        filtered = filtered[~filtered['file'].isin(qc['file'])]
        after = len(filtered)
        removed = before - after
        frac_total = removed / n0 if n0 else 0
        print(f"Quality control → removed {removed:6d} ({frac_total:6.2%})")

    filters = parse_rules(cfg["filters"])
    for rule in filters.rules:
        before = len(filtered)
        mask = rule.get_mask(filtered)
        filtered = filtered[mask]
        after = len(filtered)
        removed = before - after
        frac_total = removed / n0 if n0 else 0
        print(f"{rule.column:25s} {rule.operator:>2} {rule.value} → removed {removed:6d} ({frac_total:6.2%})")

    return filtered


def load_yaml_config(path: Path) -> dict:
    """Load YAML config from file"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def read_stim_specs(
        cfg: dict,
        ignore_time_bins: bool = False
    ) -> Generator[StimSpec, None, None]:

    global_time_bins = cfg.get("time_bins", [])

    for entry in cfg["stimuli"]:

        try:
            stim = Stim[entry["stim"]]
        except KeyError:
            raise ValueError(f"Unknown stimulus: {entry['stim']}")

        name = entry["name"]

        bins = entry.get("time_bins", global_time_bins)
        if not bins:
            raise ValueError(f"No time_bins defined for stimulus '{name}'")

        trials = range(
            entry["trial_range"]["start"],
            entry["trial_range"]["stop"],
            entry["trial_range"]["step"]
        )

        # each item is one condition (e.g. one direction); they are pooled
        # together with OR and disambiguated later via `laterality`
        parameters = [parse_rules(p) for p in entry.get("parameters", [{}])]
        time_ranges = [None] if ignore_time_bins else bins

        for time_range in time_ranges:
            yield StimSpec(
                stim=stim,
                name=name,
                trials=trials,
                time_range=time_range,
                parameters=parameters,
            )


def stim_name_order(cfg: dict) -> List[str]:
    """Order of stimulus names as they appear in the yaml config."""
    seen: List[str] = []
    for entry in cfg["stimuli"]:
        name = entry["name"]
        if name not in seen:
            seen.append(name)
    return seen


def parse_fish(fish: str) -> FileNameInfo:

    fish_regexp = re.compile(base_regexp)
    m = fish_regexp.match(fish)
    if m is None:
        raise RuntimeError(f"failed to parse: {fish}")
    g = m.groupdict()

    return FileNameInfo(
        fish_id = int(g["fish_id"]),
        age = int(g["age"]),
        line = g["line"],
        weekday = g["weekday"],
        day = int(g["day"]),
        month = g["month"],
        year = int(g["year"]),
        hour = int(g["hour"]),
        minute = int(g["minute"]),
        second = int(g["second"]),
        extra = g["extra"]
    )


def cosinor(info: FileNameInfo) -> Tuple[float, float]:
    seconds = info.hour*3600 + info.minute*60 + info.second
    theta = 2 * np.pi * (seconds / (24 * 3600))
    return (np.cos(theta), np.sin(theta))


def get_behavior_data(behavior_files: List[BehaviorFiles], fish: str) -> BehaviorData | None:
    for f in behavior_files:
        if fish in str(f.metadata):
            return load_data(f)


def get_valid_trial_count(behavior_data: BehaviorData, spec: StimSpec) -> int:
    """How many of spec.trials were actually presented to this fish."""

    stim_trials = get_trials(behavior_data)
    if stim_trials.empty:
        return 0

    mask = spec.get_mask(stim_trials) & (stim_trials.stim_select == spec.stim)
    spec_data = stim_trials[mask]
    if spec_data.empty:
        return 0

    return min(len(spec.trials), len(spec_data))


def stim_presented(behavior_data: BehaviorData, spec: StimSpec) -> bool:

    n_valid = get_valid_trial_count(behavior_data, spec)
    if n_valid == 0:
        return False

    if spec.time_range is None:
        return True

    stim_trials = get_trials(behavior_data)
    mask = spec.get_mask(stim_trials) & (stim_trials.stim_select == spec.stim)
    spec_data = stim_trials[mask].iloc[:n_valid]

    trial_duration = 1e-9 * (spec_data.stop_timestamp - spec_data.start_timestamp)
    valid_time_range = spec.time_range[0] < trial_duration

    return bool(valid_time_range.any())


def get_eye_traces(
        behavior_data: BehaviorData,
        likelihood_threshold: float = 0.9,
        divergence_threshold_deg: float = -10,
        convergence_threshold_deg: float = 60,
        window_length: int = 41
    ) -> EyesTimeseries:

    assert window_length % 2 == 1

    # extract data
    left_front_keypoint = behavior_data.eyes_tracking.eye_left_front[['x', 'y']].to_numpy()
    left_front_likelihood = behavior_data.eyes_tracking.eye_left_front.likelihood.to_numpy()

    left_back_keypoint = behavior_data.eyes_tracking.eye_left_back[['x', 'y']].to_numpy()
    left_back_likelihood = behavior_data.eyes_tracking.eye_left_back.likelihood.to_numpy()

    right_front_keypoint = behavior_data.eyes_tracking.eye_right_front[['x', 'y']].to_numpy()
    right_front_likelihood = behavior_data.eyes_tracking.eye_right_front.likelihood.to_numpy()

    right_back_keypoint = behavior_data.eyes_tracking.eye_right_back[['x', 'y']].to_numpy()
    right_back_likelihood = behavior_data.eyes_tracking.eye_right_back.likelihood.to_numpy()

    left_vector = left_back_keypoint - left_front_keypoint
    right_vector = right_back_keypoint - right_front_keypoint

    L_rad = compute_angle_between_vectors(left_vector, np.array([0, 1]))
    R_rad = compute_angle_between_vectors(right_vector, np.array([0, 1]))
    L = np.rad2deg(L_rad)
    R = np.rad2deg(R_rad)

    # remove outliers
    L[(left_front_likelihood < likelihood_threshold) | (left_back_likelihood < likelihood_threshold)] = np.nan
    R[(right_front_likelihood < likelihood_threshold) | (right_back_likelihood < likelihood_threshold)] = np.nan
    L[(-L < divergence_threshold_deg) | (-L > convergence_threshold_deg)] = np.nan
    R[(R < divergence_threshold_deg) | (R > convergence_threshold_deg)] = np.nan

    # interpolate and smooth
    L = pd.Series(L).interpolate(limit_direction="both").to_numpy()
    R = pd.Series(R).interpolate(limit_direction="both").to_numpy()
    L_s = savgol_filter(L, window_length, polyorder=2)
    R_s = savgol_filter(R, window_length, polyorder=2)
    version_angle = (L_s + R_s) / 2
    vergence_angle = R_s - L_s

    timestamps = behavior_data.video_timestamps.timestamp.to_numpy()
    n = len(timestamps)
    if n != len(version_angle):
        warnings.warn(f"frame mismatch: timestamps: {n} | eye tracking: {len(version_angle)}")
        # NOTE this happens for a file in WT/ronidazole

    res = EyesTimeseries(
        timestamps=timestamps,
        angle_left_deg=L[:n],
        angle_right_deg=R[:n],
        angle_left_smooth_deg=L_s[:n],
        angle_right_smooth_deg=R_s[:n],
        version_angle_deg=version_angle[:n],
        vergence_angle_deg=vergence_angle[:n]
    )
    return res


def plot_eyes(
        quality_control: Path,
        config_yaml: Path,
        output_png: Path,
        behavior_files: List[BehaviorFiles],
        target_fps: float = 120,
        max_trial_duration_s: float = 30
    ):
    # TODO split processing and plotting

    output_npz = output_png.with_suffix('.npz')

    removed_by_qc = []
    if quality_control.exists():
        qc = pd.read_csv(quality_control)
        removed_by_qc.extend(qc.file)

    cfg = load_yaml_config(config_yaml)
    stim_specs = list(read_stim_specs(cfg, ignore_time_bins=True))
    target_time = get_target_time(max_trial_duration_s, target_fps)

    N_fish = len(behavior_files)
    N_trials = max([len(spec.trials) for spec in stim_specs])
    N_epochs = len(stim_specs)
    N_samples = len(target_time)

    vergence_angle = np.full((N_fish, N_trials, N_epochs, N_samples), np.nan)
    version_angle = np.full((N_fish, N_trials, N_epochs, N_samples), np.nan)

    for fish_idx, behavior_file in tqdm(enumerate(behavior_files)):

        if behavior_file.metadata.stem in removed_by_qc:
            continue

        behavior_data: BehaviorData = load_data(behavior_file)
        stim_trials = get_trials(behavior_data)
        eyes = get_eye_traces(behavior_data, likelihood_threshold=0.9)

        for spec_idx, spec in enumerate(stim_specs):
            spec_mask = spec.get_mask(stim_trials) & (stim_trials.stim_select == spec.stim)
            spec_data = stim_trials[spec_mask]
            if spec_data.empty:
                continue

            valid_trials = [i for i, trial in enumerate(spec.trials) if i < len(spec_data)]
            trial_data = spec_data.iloc[valid_trials]

            for trial_idx, (trial, row) in enumerate(trial_data.iterrows()):
                mask = (eyes.timestamps > row.start_timestamp) & (eyes.timestamps < row.stop_timestamp)
                trial_duration = 1e-9 * (row.stop_timestamp - row.start_timestamp)
                trial_time = 1e-9 * (eyes.timestamps[mask] - row.start_timestamp)
                n = np.searchsorted(target_time, trial_duration)
                version_angle[fish_idx, trial_idx, spec_idx, :n] = interpolate_ts(target_time[:n], trial_time, eyes.version_angle_deg[mask])
                vergence_angle[fish_idx, trial_idx, spec_idx, :n] = interpolate_ts(target_time[:n], trial_time, eyes.vergence_angle_deg[mask])

    with open(output_npz, 'wb') as fp:
        np.savez(fp,
                 version=version_angle,
                 vergence=vergence_angle)

    vergence_per_fish = np.nanmean(vergence_angle, axis=1)
    version_per_fish = np.nanmean(version_angle, axis=1)
    data = {
        'vergence_mean': np.nanmean(vergence_per_fish, axis=0).flatten(),
        'vergence_std': np.nanstd(vergence_per_fish, axis=0).flatten(),
        'version_mean': np.nanmean(version_per_fish, axis=0).flatten(),
        'version_std': np.nanstd(version_per_fish, axis=0).flatten()
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(24, 6),
                              sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 0.5]},
                              layout='constrained')

    x_axis = np.arange(len(data['vergence_mean']))

    axes[0].plot(x_axis, data['vergence_mean'], color='black', lw=2)
    axes[0].fill_between(x_axis,
                          data['vergence_mean'] - data['vergence_std'],
                          data['vergence_mean'] + data['vergence_std'],
                          color='black', alpha=0.2, edgecolor='none')

    axes[1].plot(x_axis, data['version_mean'], color='black', lw=2)
    axes[1].fill_between(x_axis,
                          data['version_mean'] - data['version_std'],
                          data['version_mean'] + data['version_std'],
                          color='black', alpha=0.2, edgecolor='none')

    axes[0].set_ylabel('<vergence [deg]>')
    axes[0].set_ylim((15, 65))
    axes[0].legend(loc='upper right', frameon=False)

    axes[1].set_ylabel('<version [deg]>')
    axes[1].axhline(0, linestyle='--', color='gray', alpha=0.5)
    axes[1].set_ylim((-15, 15))

    axes[2].set_axis_off()
    N_samples = len(data['vergence_mean']) // len(stim_specs)
    for idx, stim in enumerate(stim_specs):
        text_label = textwrap.fill(str(stim), width=20)
        x_pos = idx * N_samples + N_samples // 2
        axes[2].text(x_pos, 1.0, text_label, ha='right', va='top', rotation=45, fontsize=9)

    # Scale Bar
    scale_duration_sec = 10
    scale_width_samples = scale_duration_sec * target_fps
    scalebar = AnchoredSizeBar(axes[1].transData, scale_width_samples,
                                f'{scale_duration_sec} s', 'lower right',
                                pad=0.5, color='black', frameon=False, size_vertical=0.2)
    axes[1].add_artist(scalebar)

    plt.savefig(output_png, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Bout heatmap
#
# Per-fish/per-epoch bout counting is fully vectorized (groupby + reindex on
# the full trial x category x laterality grid). Averaging is done over fish
# ONLY -- trial and time bin are preserved and end up as the two axes of a
# 2D block per bout category. All the blocks (one per bout category, split
# horizontally by stimulus x laterality) are then assembled into a SINGLE
# heatmap image.
# ---------------------------------------------------------------------------

LATERALITY_ORDER = {"ipsi": 0, "contra": 1, "all": 2}


def _order_lateralities(values) -> List[str]:
    return sorted(values, key=lambda v: LATERALITY_ORDER.get(v, 99))


def get_laterality_labels(bouts: pd.DataFrame, spec: StimSpec) -> List[str]:
    """
    Laterality labels (ipsi/contra) relevant to this stim, based on what's
    actually present in the `laterality` column for matching bouts. Falls
    back to a single "all" bucket for non-lateralized stimuli.
    """
    mask = (bouts.stim == spec.stim) & spec.get_mask(bouts)
    values = bouts.loc[mask, "laterality"].dropna().unique().tolist()
    return _order_lateralities(values) if values else ["all"]


def compute_epoch_bout_counts(
        fish_bouts: pd.DataFrame,
        spec: StimSpec,
        valid_n_trials: int,
        categories: List[str],
        laterality_labels: List[str],
    ) -> pd.DataFrame:
    """
    Bout counts/frequency for one fish x one stim epoch, on the full
    (trial_idx, bout_category, laterality_group) grid -- missing
    combinations are filled with 0, not dropped.
    """

    lo, hi = spec.time_range
    duration = hi - lo

    mask = (
        (fish_bouts.stim == spec.stim) &
        spec.get_mask(fish_bouts) &
        (fish_bouts.trial_time >= lo) &
        (fish_bouts.trial_time < hi)
    )
    epoch_bouts = fish_bouts[mask].copy()

    trial_map = {t: i for i, t in enumerate(spec.trials)}
    epoch_bouts["trial_idx"] = epoch_bouts.trial_num.map(trial_map)
    epoch_bouts = epoch_bouts.dropna(subset=["trial_idx"])
    epoch_bouts["trial_idx"] = epoch_bouts["trial_idx"].astype(int)
    epoch_bouts = epoch_bouts[epoch_bouts.trial_idx < valid_n_trials]
    epoch_bouts["category"] = epoch_bouts["category"].astype(int)

    # ipsi/contra when meaningful, otherwise pool everything as "all"
    if "laterality" in epoch_bouts.columns:
        epoch_bouts["laterality_group"] = epoch_bouts["laterality"].where(
            epoch_bouts["laterality"].notna(), "all"
        )
    else:
        epoch_bouts["laterality_group"] = "all"

    epoch_bouts["bout_category"] = epoch_bouts["category"].map(lambda i: categories[i])

    counts = (
        epoch_bouts
        .groupby(["trial_idx", "bout_category", "laterality_group"])
        .size()
        .rename("bout_counts")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [range(valid_n_trials), categories, laterality_labels],
        names=["trial_idx", "bout_category", "laterality_group"],
    )
    counts = (
        counts
        .set_index(["trial_idx", "bout_category", "laterality_group"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    counts["bout_frequency"] = counts["bout_counts"] / duration

    return counts


def compute_bout_frequency_table(
        quality_control: Path,
        input_csv: Path,
        config_yaml: Path,
        behavior_files: List[BehaviorFiles],
    ) -> pd.DataFrame:

    cfg = load_yaml_config(config_yaml)
    stim_specs = list(read_stim_specs(cfg))
    bouts = load_bouts(input_csv)
    filtered_bouts = filter_bouts(quality_control, bouts, cfg)

    categories = list(bouts_category_name_short)
    laterality_labels = {
        id(spec): get_laterality_labels(filtered_bouts, spec) for spec in stim_specs
    }

    tables = []
    fish_groups = filtered_bouts.groupby("file")

    for fish, fish_bouts in tqdm(fish_groups):

        fish_info = parse_fish(fish)
        time_cos, time_sin = cosinor(fish_info)

        behavior_data = get_behavior_data(behavior_files, fish)
        if behavior_data is None:
            raise RuntimeError(f"{fish} not found, aborting")

        for spec in stim_specs:

            if spec.time_range is None:
                raise RuntimeError('time range should not be None')

            if not stim_presented(behavior_data, spec):
                print(f"{fish} - {spec} not presented, skipping")
                continue

            valid_n_trials = get_valid_trial_count(behavior_data, spec)
            if valid_n_trials == 0:
                continue

            counts = compute_epoch_bout_counts(
                fish_bouts, spec, valid_n_trials, categories, laterality_labels[id(spec)]
            )

            counts["file"] = fish
            counts["dpf"] = fish_info.age
            counts["day"] = f"{fish_info.day}.{fish_info.month}.{fish_info.year}"
            counts["time_of_day_cos"] = time_cos
            counts["time_of_day_sin"] = time_sin
            counts["stim_name"] = spec.name
            counts["time_bin_start"] = spec.time_range[0]
            counts["time_bin_stop"] = spec.time_range[1]
            counts["time_bin_duration"] = spec.time_range[1] - spec.time_range[0]

            tables.append(counts)

    if not tables:
        return pd.DataFrame(columns=[
            "trial_idx", "bout_category", "laterality_group", "bout_counts",
            "bout_frequency", "file", "dpf", "day", "time_of_day_cos",
            "time_of_day_sin", "stim_name", "time_bin_start", "time_bin_stop",
            "time_bin_duration",
        ])

    return pd.concat(tables, ignore_index=True)


def build_bout_heatmap_matrix(
        avg: pd.DataFrame,
        category_order: List[str],
        stim_order: List[str],
    ) -> Tuple[pd.DataFrame, List[Tuple[int, int, str]], List[Tuple[int, int, str]], List[str], int]:
    """
    Assemble the tidy per-(stim, time_bin, trial, category, laterality)
    table into a single 2D matrix ready to be passed to imshow.

    Rows:    (bout_category, trial_idx) stacked -- one block of `n_trials`
             rows per bout category.
    Columns: (stim_name, laterality_group, time_bin_start) stacked -- one
             block of time bins per (stim, laterality) pair.

    Returns the pivoted DataFrame plus group boundaries (for separator
    lines / labels) at both the stim level and the (stim, laterality) level.
    """

    n_trials = int(avg.trial_idx.max()) + 1

    columns: List[Tuple[str, str, float]] = []
    col_groups: List[Tuple[int, int, str]] = []      # stim-level blocks
    col_subgroups: List[Tuple[int, int, str]] = []   # (stim, laterality) blocks
    time_bin_labels: List[str] = []

    for stim in stim_order:
        stim_rows = avg[avg.stim_name == stim]
        if stim_rows.empty:
            continue

        stim_start = len(columns)
        lateralities = _order_lateralities(stim_rows.laterality_group.unique())

        for later in lateralities:
            sub_rows = stim_rows[stim_rows.laterality_group == later]
            bins = (
                sub_rows[["time_bin_start", "time_bin_stop"]]
                .drop_duplicates()
                .sort_values("time_bin_start")
            )

            sub_start = len(columns)
            for t_start, t_stop in bins.itertuples(index=False):
                columns.append((stim, later, t_start))
                time_bin_labels.append(f"{t_start:g}-{t_stop:g}s")
            col_subgroups.append((sub_start, len(columns), later))

        col_groups.append((stim_start, len(columns), stim))

    row_index = pd.MultiIndex.from_tuples(
        [(cat, t) for cat in category_order for t in range(n_trials)],
        names=["bout_category", "trial_idx"],
    )
    col_index = pd.MultiIndex.from_tuples(
        columns, names=["stim_name", "laterality_group", "time_bin_start"]
    )

    pivot = avg.pivot_table(
        index=["bout_category", "trial_idx"],
        columns=["stim_name", "laterality_group", "time_bin_start"],
        values="bout_frequency",
    )
    pivot = pivot.reindex(index=row_index, columns=col_index)

    return pivot, col_groups, col_subgroups, time_bin_labels, n_trials


def plot_bout_heatmap(
        fig: plt.Figure,
        ax: plt.Axes,
        pivot: pd.DataFrame,
        category_order: List[str],
        col_groups: List[Tuple[int, int, str]],
        col_subgroups: List[Tuple[int, int, str]],
        time_bin_labels: List[str],
        n_trials: int,
        cmap: str = 'inferno',
        clim: Tuple[float, float] = (0, 0.45),
    ) -> None:

    data = pivot.to_numpy(dtype=float)
    n_rows, n_cols = data.shape

    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=clim[0], vmax=clim[1])
    fig.colorbar(im, ax=ax, label='bout frequency', fraction=0.015, pad=0.005)

    # x ticks: time bin label per column
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(time_bin_labels, rotation=90, fontsize=6)

    # y ticks: trial number per row
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([t for _, t in pivot.index], fontsize=6)

    # stim-level separators + labels (above the plot)
    for start, end, label in col_groups:
        if start > 0:
            ax.axvline(start - 0.5, color='white', lw=1.6)
        ax.text(
            (start + end - 1) / 2, -0.05 * n_rows, label,
            ha='center', va='bottom', fontsize=9, clip_on=False,
        )

    # (stim, laterality) separators + labels (just above tick labels)
    for start, end, label in col_subgroups:
        if start > 0:
            ax.axvline(start - 0.5, color='white', lw=0.6, alpha=0.7)
        ax.text(
            (start + end - 1) / 2, -0.012 * n_rows, label,
            ha='center', va='bottom', fontsize=6, clip_on=False,
        )

    # bout category separators + labels (left of the plot)
    for idx, cat in enumerate(category_order):
        row_start = idx * n_trials
        row_end = row_start + n_trials
        if row_start > 0:
            ax.axhline(row_start - 0.5, color='white', lw=1.6)
        ax.text(
            -0.006 * n_cols, (row_start + row_end - 1) / 2, cat,
            ha='right', va='center', fontsize=8, clip_on=False,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_heatmap(
        quality_control: Path,
        input_csv: Path,
        config_yaml: Path,
        output_png: Path,
        behavior_files: List[BehaviorFiles]
    ) -> None:

    output_csv = output_png.parent / 'bout_frequency.csv'
    output_avg_csv = output_png.parent / 'bout_frequency_avg.csv'

    cfg = load_yaml_config(config_yaml)
    per_fish = compute_bout_frequency_table(quality_control, input_csv, config_yaml, behavior_files)
    per_fish.to_csv(output_csv, index=False)

    if per_fish.empty:
        print("No bouts found, skipping heatmap plot")
        return

    # average over fish only -- trial and time bin are preserved
    group_cols = [
        "stim_name", "time_bin_start", "time_bin_stop", "time_bin_duration",
        "trial_idx", "bout_category", "laterality_group",
    ]
    avg = per_fish.groupby(group_cols, as_index=False)["bout_frequency"].mean()
    avg.to_csv(output_avg_csv, index=False)

    category_order = list(bouts_category_name_short)
    stim_order = stim_name_order(cfg)

    pivot, col_groups, col_subgroups, time_bin_labels, n_trials = build_bout_heatmap_matrix(
        avg, category_order, stim_order
    )

    n_rows, n_cols = pivot.shape
    fig_w = max(16, 0.22 * n_cols)
    fig_h = max(12, 0.28 * n_rows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout='constrained')
    plot_bout_heatmap(fig, ax, pivot, category_order, col_groups, col_subgroups, time_bin_labels, n_trials)

    fig.savefig(output_png, bbox_inches='tight')
    plt.show()


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Collect bout.csv and plot results"
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder (e.g. WT_oct_2025)",
    )

    parser.add_argument(
        "yaml",
        type=Path,
        help="plot config file",
    )

    parser.add_argument(
        "--qc-csv",
        default='qc.csv',
        help="quality control: fish not moving and tracking issues",
    )

    parser.add_argument(
        "--bouts-csv",
        default='bouts.csv',
        help="input CSV file",
    )

    parser.add_argument(
        "--bouts-png",
        default='bouts.png',
        help="output bout PNG file",
    )

    parser.add_argument(
        "--eyes-png",
        default='eyes.png',
        help="output eye PNG file",
    )

    # Directory layout overrides
    parser.add_argument(
        "--metadata",
        default="results",
        help="Subfolder containing metadata files (default: results)",
    )

    parser.add_argument(
        "--stimuli",
        default="results",
        help="Subfolder containing stimulus log files (default: results)",
    )

    parser.add_argument(
        "--tracking",
        default="results",
        help="Subfolder containing tracking CSV files (default: results)",
    )

    parser.add_argument(
        "--lightning-pose",
        default="lightning_pose",
        help="Subfolder containing lightning pose tracking CSV files (default: lightning_pose)",
    )

    parser.add_argument(
        "--temperature",
        default="results",
        help="Subfolder containing temperature logs (default: results)",
    )

    parser.add_argument(
        "--video",
        default="results",
        help="Subfolder containing raw video files (default: results)",
    )

    parser.add_argument(
        "--video-timestamp",
        default="results",
        help="Subfolder containing video timestamp files (default: results)",
    )

    parser.add_argument(
        "--results",
        default="results",
        help="Subfolder where per-animal exports will be written (default: results)",
    )

    parser.add_argument(
        "--plots",
        default="plots",
        help="Subfolder containing plots (default: plots)",
    )

    return parser


def run_plot(
        qc_csv: str,
        bouts_csv: str,
        bouts_png: str,
        eyes_png: str,
        config_yaml: Path,
        root: Path,
        metadata: str,
        stimuli: str,
        tracking: str,
        lightning_pose: str,
        temperature: str,
        video: str,
        video_timestamp: str,
        results: str,
        plots: str,
    ) -> None:

    quality_control = root / qc_csv
    input_csv = root / bouts_csv
    output_bouts_png = root / bouts_png
    output_eyes_png = root / eyes_png

    directories = Directories(
        root,
        metadata=metadata,
        stimuli=stimuli,
        tracking=tracking,
        full_tracking=lightning_pose,
        temperature=temperature,
        video=video,
        video_timestamp=video_timestamp,
        results=results,
        plots=plots
    )
    behavior_files = find_files(directories)

    plot_heatmap(
        quality_control,
        input_csv,
        config_yaml,
        output_bouts_png,
        behavior_files
    )

    plot_eyes(
        quality_control,
        config_yaml,
        output_eyes_png,
        behavior_files,
        max_trial_duration_s=30,
        target_fps=120
    )


def main(args: argparse.Namespace) -> None:

    run_plot(
        qc_csv=args.qc_csv,
        bouts_csv=args.bouts_csv,
        bouts_png=args.bouts_png,
        eyes_png=args.eyes_png,
        config_yaml=args.yaml,
        root=args.root,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        lightning_pose=args.lightning_pose,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        plots=args.plots,
    )


if __name__ == "__main__":

    main(build_parser().parse_args())