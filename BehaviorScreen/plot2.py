from typing import List, Tuple, Generator, Any
import argparse
from pathlib import Path
import re
from dataclasses import dataclass
import operator

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from BehaviorScreen.process import get_trials


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
    """
    A stimulus epoch to analyze.

    Trials/bouts are matched by `stim` + `parameters`. `parameters` is a
    list of RuleSets combined with OR: a row counts for this spec if it
    matches ANY of the rulesets. Each ruleset should normally include an
    `epoch_name` rule to disambiguate sub-conditions that share the same
    `Stim` enum value (e.g. "OMR lateral" vs "OMR forward", both Stim.OMR;
    or "dark" vs "bright -> dark", both Stim.DARK) -- when a ruleset pools
    several raw epoch_name values (e.g. left+right), that's how ipsi/contra
    trials end up grouped under one display name; the actual side is
    resolved later via the per-bout `laterality` column.

    `stim` is technically redundant with a sufficiently specific
    `epoch_name` rule, but is kept as a cheap consistency guard (and for
    readability in the YAML) rather than an active filter dependency.

    There is no separate trial range/count: `trial_num` in the bouts table
    is already a 0-based, contiguous index local to each raw epoch_name
    value (assigned upstream by the megabouts step), so it's used directly.
    """
    stim: Stim
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

        parameters = [parse_rules(p) for p in entry.get("parameters", [{}])]
        time_ranges = [None] if ignore_time_bins else bins

        for time_range in time_ranges:
            yield StimSpec(
                stim=stim,
                name=name,
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


def get_matched_trial_rows(behavior_data: BehaviorData, spec: StimSpec) -> pd.DataFrame:
    """All trials (from the stimulus log) that match this spec, in their
    natural row order."""
    stim_trials = get_trials(behavior_data)
    if stim_trials.empty:
        return stim_trials
    mask = spec.get_mask(stim_trials) & (stim_trials.stim_select == spec.stim)
    return stim_trials[mask]


def stim_presented(behavior_data: BehaviorData, spec: StimSpec) -> bool:

    matched = get_matched_trial_rows(behavior_data, spec)
    if matched.empty:
        return False

    if spec.time_range is None:
        return True

    trial_duration = 1e-9 * (matched.stop_timestamp - matched.start_timestamp)
    return bool((spec.time_range[0] < trial_duration).any())


def get_epoch_trial_counts(behavior_data: BehaviorData, spec: StimSpec) -> int:
    """
    Number of trial slots ("trial_idx" values) to use for this spec.

    trial_num is 0-based and contiguous WITHIN each raw epoch_name value.
    When a spec pools several raw epoch_name values (e.g. "grating right" +
    "grating left"), trial_num=k in each pooled group means "k-th
    presentation of that direction" -- so the grid width is the size of the
    largest constituent group, not their sum.
    """
    matched = get_matched_trial_rows(behavior_data, spec)
    if matched.empty:
        return 0
    return int(matched.groupby("epoch_name").size().max())


# ---------------------------------------------------------------------------
# Bout heatmap
#
# Per-fish/per-epoch bout counting is fully vectorized (groupby + reindex on
# the full trial x category x laterality grid). Four aggregation levels are
# then produced from the same tidy table:
#   - full detail       : trial x time bin, per bout category
#   - trial-averaged     : time bin only, per bout category
#   - time-bin-averaged  : trial only, per bout category
#   - fully averaged     : one value per bout category
# In all cases, averaging across FISH is a plain mean of per-fish rates
# (each fish is one sample). Averaging across TRIAL/TIME BIN is instead
# done by summing bout_counts and duration within each fish first, then
# recomputing frequency = counts / duration -- this correctly weights
# unequal time-bin durations, and is equivalent to a plain mean of
# frequencies when durations are equal (e.g. across trials).
# ---------------------------------------------------------------------------

LATERALITY_ORDER = {"ipsi": 0, "contra": 1, "none": 2}


def _order_lateralities(values) -> List[str]:
    return sorted(values, key=lambda v: LATERALITY_ORDER.get(v, 99))


def get_laterality_labels(bouts: pd.DataFrame, spec: StimSpec) -> List[str]:
    """
    Laterality labels (ipsi/contra) relevant to this stim, based on what's
    actually present in the `laterality` column for matching bouts. Falls
    back to a single "none" bucket for non-lateralized stimuli.
    """
    mask = (bouts.stim == spec.stim) & spec.get_mask(bouts)
    values = bouts.loc[mask, "laterality"].dropna().unique().tolist()
    return _order_lateralities(values) if values else ["none"]


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

    `trial_num` is already a 0-based, contiguous index local to each raw
    epoch_name (assigned upstream in the megabouts step), so it's used
    directly as `trial_idx` -- no remapping needed.
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

    # some bouts may have an undefined/NaN category or trial_num -- drop
    # those rather than crashing on int casting/indexing
    epoch_bouts = epoch_bouts.dropna(subset=["category", "trial_num"])
    epoch_bouts["trial_idx"] = epoch_bouts["trial_num"].astype(int)
    epoch_bouts = epoch_bouts[epoch_bouts.trial_idx < valid_n_trials]
    epoch_bouts["category"] = epoch_bouts["category"].astype(int)

    # ipsi/contra when meaningful, otherwise pool everything as "none"
    if "laterality" in epoch_bouts.columns:
        epoch_bouts["laterality_group"] = epoch_bouts["laterality"].where(
            epoch_bouts["laterality"].notna(), "none"
        )
    else:
        epoch_bouts["laterality_group"] = "none"

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

            valid_n_trials = get_epoch_trial_counts(behavior_data, spec)
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


def aggregate_bout_frequency(
        per_fish: pd.DataFrame,
        average_trial: bool,
        average_time_bin: bool,
    ) -> pd.DataFrame:
    """
    Aggregate the tidy per-fish bout frequency table, optionally collapsing
    the trial and/or time-bin dimensions. bout_category and laterality_group
    are never collapsed.

    Two-step aggregation, to stay statistically correct:
      1. WITHIN each fish, sum bout_counts and duration across whichever
         dimension(s) are being collapsed, then recompute
         bout_frequency = counts / duration. This correctly weights
         unequal time-bin durations (a naive mean of per-bin frequencies
         would overweight short bins), and reduces to a plain mean when
         durations are equal (e.g. collapsing across trials, which share
         the same duration for a given time bin).
      2. ACROSS fish, take a plain mean of the (possibly collapsed)
         per-fish bout_frequency -- each fish counts as one sample.
    """

    working = per_fish

    if average_trial or average_time_bin:
        keep_cols = ["file", "stim_name", "bout_category", "laterality_group"]
        if not average_time_bin:
            keep_cols += ["time_bin_start", "time_bin_stop", "time_bin_duration"]
        if not average_trial:
            keep_cols += ["trial_idx"]

        working = (
            per_fish
            .groupby(keep_cols, as_index=False)[["bout_counts", "time_bin_duration"]]
            .sum()
        )
        working["bout_frequency"] = working["bout_counts"] / working["time_bin_duration"]

    group_cols = ["stim_name", "bout_category", "laterality_group"]
    if not average_time_bin:
        group_cols += ["time_bin_start", "time_bin_stop", "time_bin_duration"]
    if not average_trial:
        group_cols += ["trial_idx"]

    avg = working.groupby(group_cols, as_index=False)["bout_frequency"].mean()
    return avg


def build_bout_heatmap_matrix(
        avg: pd.DataFrame,
        category_order: List[str],
        stim_order: List[str],
    ) -> Tuple[pd.DataFrame, List[Tuple[int, int, str]], List[Tuple[int, int, str]], List[str], int]:
    """
    Assemble an aggregated bout-frequency table into a single 2D matrix
    ready to be passed to imshow.

    Whether `trial_idx` / `time_bin_start` are still present as columns in
    `avg` (i.e. whether that dimension was averaged out) determines whether
    rows are split by trial and whether columns are split by time bin.

    Rows:    bout_category, optionally x trial_idx.
    Columns: stim_name x laterality_group, optionally x time_bin_start.
    """

    has_trial = "trial_idx" in avg.columns
    has_time_bin = "time_bin_start" in avg.columns

    n_trials = int(avg.trial_idx.max()) + 1 if has_trial else 1

    columns: list = []
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
            sub_start = len(columns)

            if has_time_bin:
                bins = (
                    sub_rows[["time_bin_start", "time_bin_stop"]]
                    .drop_duplicates()
                    .sort_values("time_bin_start")
                )
                for t_start, t_stop in bins.itertuples(index=False):
                    columns.append((stim, later, t_start))
                    time_bin_labels.append(f"{t_start:g}-{t_stop:g}s")
            else:
                columns.append((stim, later))
                time_bin_labels.append("avg")

            col_subgroups.append((sub_start, len(columns), later))

        col_groups.append((stim_start, len(columns), stim))

    if has_trial:
        row_index = pd.MultiIndex.from_tuples(
            [(cat, t) for cat in category_order for t in range(n_trials)],
            names=["bout_category", "trial_idx"],
        )
        index_cols = ["bout_category", "trial_idx"]
    else:
        row_index = pd.Index(category_order, name="bout_category")
        index_cols = ["bout_category"]

    col_names = (
        ["stim_name", "laterality_group", "time_bin_start"] if has_time_bin
        else ["stim_name", "laterality_group"]
    )
    col_index = pd.MultiIndex.from_tuples(columns, names=col_names)

    pivot = avg.pivot_table(index=index_cols, columns=col_names, values="bout_frequency")
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

    # x ticks: time bin label (or "avg") per column
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(time_bin_labels, rotation=90, fontsize=6)

    # y ticks: trial number per row, if trials weren't averaged out
    has_trial_rows = isinstance(pivot.index, pd.MultiIndex)
    if has_trial_rows:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([t for _, t in pivot.index], fontsize=6)
    else:
        ax.set_yticks([])

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


# (average_trial, average_time_bin, filename_suffix, title)
HEATMAP_VARIANTS: List[Tuple[bool, bool, str, str]] = [
    (False, False, "", "trial x time bin"),
    (True, False, "_trial_avg", "averaged over trials"),
    (False, True, "_timebin_avg", "averaged over time bins"),
    (True, True, "_full_avg", "averaged over trials and time bins"),
]


def plot_heatmap(
        quality_control: Path,
        input_csv: Path,
        config_yaml: Path,
        output_png: Path,
        behavior_files: List[BehaviorFiles]
    ) -> None:

    output_csv = output_png.parent / 'bout_frequency.csv'

    cfg = load_yaml_config(config_yaml)
    per_fish = compute_bout_frequency_table(quality_control, input_csv, config_yaml, behavior_files)
    per_fish.to_csv(output_csv, index=False)

    if per_fish.empty:
        print("No bouts found, skipping heatmap plots")
        return

    category_order = list(bouts_category_name_short)
    stim_order = stim_name_order(cfg)

    for average_trial, average_time_bin, suffix, title in HEATMAP_VARIANTS:

        avg = aggregate_bout_frequency(per_fish, average_trial, average_time_bin)
        avg.to_csv(output_png.parent / f'bout_frequency_avg{suffix}.csv', index=False)

        pivot, col_groups, col_subgroups, time_bin_labels, n_trials = build_bout_heatmap_matrix(
            avg, category_order, stim_order
        )

        n_rows, n_cols = pivot.shape
        fig_w = max(16, 0.22 * n_cols)
        fig_h = max(6, 0.28 * n_rows)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout='constrained')
        plot_bout_heatmap(fig, ax, pivot, category_order, col_groups, col_subgroups, time_bin_labels, n_trials)
        ax.set_title(title, fontsize=10)

        variant_png = output_png.parent / f"{output_png.stem}{suffix}{output_png.suffix}"
        fig.savefig(variant_png, bbox_inches='tight')

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
        help="output bout PNG file (variants are saved alongside with suffixes)",
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


def main(args: argparse.Namespace) -> None:

    run_plot(
        qc_csv=args.qc_csv,
        bouts_csv=args.bouts_csv,
        bouts_png=args.bouts_png,
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