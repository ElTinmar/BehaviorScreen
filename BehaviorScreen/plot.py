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

from BehaviorScreen.core import Stim, Laterality, BoutSign
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


MAX_COLORBAR = 0.6


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
# Two parallel per-fish tables are built from the SAME per-fish/per-epoch
# loop (the expensive part -- behavior_data lookups -- is only done once):
#
#   - per_fish          : split by laterality_group (ipsi/contra/none),
#                          direction-pooled per spec (e.g. "grating right" +
#                          "grating left" both count as "OMR lateral").
#                          Used for the 4 main heatmap variants.
#
#   - per_fish_classic  : split by RAW epoch_name (stimulus direction, e.g.
#                          "grating right" vs "grating left" as SEPARATE
#                          columns) x sign_group (LEFT/RIGHT, the fish's own
#                          kinematic turn direction). This mirrors the
#                          original pre-refactor plot, which had one column
#                          per raw stimulus parameter combination (e.g.
#                          omr_angle_deg==-90 / ==90) and rows split by raw
#                          bout sign -- i.e. every (stim direction) x (bout
#                          sign) combination is visible, NOT pooled the way
#                          the 4 main variants pool it via `laterality`.
#
# Aggregation levels produced from `per_fish`:
#   - full detail        : trial x time bin, per bout category
#   - trial-averaged      : time bin only, per bout category
#   - time-bin-averaged   : trial only, per bout category
#   - fully averaged      : one value per bout category
# `per_fish_classic` only ever feeds the "classic" heatmap (trial-averaged,
# time bins preserved).
#
# In all cases, averaging across FISH is a plain mean of per-fish rates
# (each fish is one sample). Averaging across TRIAL/TIME BIN is instead
# done by summing bout_counts and duration within each fish first, then
# recomputing frequency = counts / duration -- this correctly weights
# unequal time-bin durations, and is equivalent to a plain mean of
# frequencies when durations are equal (e.g. across trials).
# ---------------------------------------------------------------------------

# Bout categories dropped from every table/plot. Encoded category indices
# in bouts.csv still refer to the FULL bouts_category_name_short list, so
# that list must stay intact for the int -> name mapping -- only the
# display/grid order (BOUT_CATEGORIES) is filtered.
ALL_BOUT_CATEGORIES = list(bouts_category_name_short)
EXCLUDED_BOUT_CATEGORIES = {"LCS", "SCS"}
BOUT_CATEGORIES = [c for c in ALL_BOUT_CATEGORIES if c not in EXCLUDED_BOUT_CATEGORIES]

# Raw `laterality` column in bouts.csv holds Laterality enum values
# (IPSILATERAL=1, NONDIRECTIONAL=0, CONTRALATERAL=-1). Bouts under
# non-lateralized stimuli (no laterality assigned at all) show up as NaN.
LATERALITY_CODE_LABELS = {
    Laterality.IPSILATERAL: "ipsi",
    Laterality.CONTRALATERAL: "contra",
    Laterality.NONDIRECTIONAL: "none",
}
LATERALITY_ORDER = {"ipsi": 0, "contra": 1, "none": 2}

# Raw `sign` column in bouts.csv holds BoutSign enum values (LEFT=-1,
# RIGHT=1) -- every bout has a sign, regardless of stimulus, unlike
# laterality which is only meaningful for lateralized stimuli.
BOUT_SIGN_LABELS = {
    BoutSign.LEFT: "LEFT",
    BoutSign.RIGHT: "RIGHT",
}
SIGN_ORDER = {"LEFT": 0, "RIGHT": 1}
SIGN_LABELS = ["LEFT", "RIGHT"]


def _order_lateralities(values) -> List[str]:
    return sorted(values, key=lambda v: LATERALITY_ORDER.get(v, 99))


def _order_signs(values) -> List[str]:
    return sorted(values, key=lambda v: SIGN_ORDER.get(v, 99))


def _map_laterality(series: pd.Series) -> pd.Series:
    """
    Map raw Laterality codes to display labels.

    NONDIRECTIONAL (0) is a genuine category (straight bouts under a
    lateralized stim), not a fallback -- it maps to "none" just like the
    fallback for bouts with no laterality assigned at all (NaN, under a
    non-lateralized stim), so both end up sharing the "none" column.
    """
    mapped = series.map(LATERALITY_CODE_LABELS)
    return mapped.where(mapped.notna(), "none")


def _map_sign(series: pd.Series) -> pd.Series:
    """Map raw BoutSign codes (-1/1) to LEFT/RIGHT display labels."""
    return series.map(BOUT_SIGN_LABELS)


def get_laterality_labels(bouts: pd.DataFrame, spec: StimSpec) -> List[str]:
    """
    Laterality labels (ipsi/contra/none) relevant to this stim, based on
    what's actually present in the `laterality` column for matching bouts.
    Falls back to a single "none" bucket for non-lateralized stimuli.
    """
    mask = (bouts.stim == spec.stim) & spec.get_mask(bouts)
    if "laterality" not in bouts.columns:
        return ["none"]
    values = _map_laterality(bouts.loc[mask, "laterality"]).unique().tolist()
    return _order_lateralities(values) if values else ["none"]


def get_epoch_name_labels(bouts: pd.DataFrame, spec: StimSpec) -> List[str]:
    """
    Distinct RAW epoch_name values matching this spec (e.g. "grating right"
    and "grating left" for the pooled "OMR lateral" spec). Used only for
    the classic heatmap, which -- unlike the 4 main variants -- shows each
    stimulus direction as its own column rather than pooling them.
    """
    mask = (bouts.stim == spec.stim) & spec.get_mask(bouts)
    if "epoch_name" not in bouts.columns:
        return [spec.name]
    values = bouts.loc[mask, "epoch_name"].dropna().unique().tolist()
    return sorted(values) if values else [spec.name]


def compute_epoch_bout_counts(
        fish_bouts: pd.DataFrame,
        spec: StimSpec,
        valid_n_trials: int,
        laterality_labels: List[str],
    ) -> pd.DataFrame:
    """
    Bout counts/frequency for one fish x one stim epoch, on the full
    (trial_idx, bout_category, laterality_group) grid -- missing
    combinations are filled with 0, not dropped. Categories in
    EXCLUDED_BOUT_CATEGORIES are dropped entirely (not shown, not counted).
    Stimulus direction is POOLED here (e.g. "grating right" + "grating
    left" both count towards "OMR lateral") -- ipsi/contra is resolved
    from the per-bout `laterality` column, not from direction.

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

    # map integer category code -> name using the FULL category list (the
    # code is fixed by megabouts and doesn't change when we drop display
    # categories), then drop excluded categories entirely
    epoch_bouts["bout_category"] = epoch_bouts["category"].map(lambda i: ALL_BOUT_CATEGORIES[i])
    epoch_bouts = epoch_bouts[~epoch_bouts["bout_category"].isin(EXCLUDED_BOUT_CATEGORIES)]

    # map raw numeric laterality codes to ipsi/contra/none
    if "laterality" in epoch_bouts.columns:
        epoch_bouts["laterality_group"] = _map_laterality(epoch_bouts["laterality"])
    else:
        epoch_bouts["laterality_group"] = "none"

    counts = (
        epoch_bouts
        .groupby(["trial_idx", "bout_category", "laterality_group"])
        .size()
        .rename("bout_counts")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [range(valid_n_trials), BOUT_CATEGORIES, laterality_labels],
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


def compute_epoch_bout_counts_classic(
        fish_bouts: pd.DataFrame,
        spec: StimSpec,
        valid_n_trials: int,
        epoch_name_labels: List[str],
    ) -> pd.DataFrame:
    """
    Bout counts/frequency for one fish x one stim epoch, split by RAW
    epoch_name (stimulus direction, e.g. "grating right" vs "grating left")
    x bout sign (LEFT/RIGHT, the fish's OWN kinematic turn direction) --
    used only for the "classic" heatmap.

    Unlike compute_epoch_bout_counts, directions are NOT pooled here: each
    raw epoch_name gets its own column, exactly like the original
    pre-refactor plot (which had one column per raw stimulus parameter
    combination, e.g. omr_angle_deg==-90 vs ==90). This lets every
    (stimulus direction) x (bout sign) combination be read off directly,
    which pooling by `laterality` would otherwise hide.
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

    epoch_bouts = epoch_bouts.dropna(subset=["category", "trial_num", "sign", "epoch_name"])
    epoch_bouts["trial_idx"] = epoch_bouts["trial_num"].astype(int)
    epoch_bouts = epoch_bouts[epoch_bouts.trial_idx < valid_n_trials]
    epoch_bouts["category"] = epoch_bouts["category"].astype(int)

    epoch_bouts["bout_category"] = epoch_bouts["category"].map(lambda i: ALL_BOUT_CATEGORIES[i])
    epoch_bouts = epoch_bouts[~epoch_bouts["bout_category"].isin(EXCLUDED_BOUT_CATEGORIES)]

    epoch_bouts["sign_group"] = _map_sign(epoch_bouts["sign"])

    counts = (
        epoch_bouts
        .groupby(["trial_idx", "bout_category", "epoch_name", "sign_group"])
        .size()
        .rename("bout_counts")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [range(valid_n_trials), BOUT_CATEGORIES, epoch_name_labels, SIGN_LABELS],
        names=["trial_idx", "bout_category", "epoch_name", "sign_group"],
    )
    counts = (
        counts
        .set_index(["trial_idx", "bout_category", "epoch_name", "sign_group"])
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two tidy per-fish tables, built from the same fish/epoch loop
    (the expensive part -- per-fish behavior_data lookups -- is only done
    once):

      - per_fish          : split by laterality_group (direction-pooled),
                             used for the 4 main heatmap variants.
      - per_fish_classic  : split by raw epoch_name (direction NOT pooled)
                             x sign_group, used only for the classic plot.
    """

    cfg = load_yaml_config(config_yaml)
    stim_specs = list(read_stim_specs(cfg))
    bouts = load_bouts(input_csv)
    filtered_bouts = filter_bouts(quality_control, bouts, cfg)

    laterality_labels = {
        id(spec): get_laterality_labels(filtered_bouts, spec) for spec in stim_specs
    }
    epoch_name_labels = {
        id(spec): get_epoch_name_labels(filtered_bouts, spec) for spec in stim_specs
    }

    tables = []
    tables_classic = []
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

            common_fields = {
                "file": fish,
                "dpf": fish_info.age,
                "day": f"{fish_info.day}.{fish_info.month}.{fish_info.year}",
                "time_of_day_cos": time_cos,
                "time_of_day_sin": time_sin,
                "stim_name": spec.name,
                "time_bin_start": spec.time_range[0],
                "time_bin_stop": spec.time_range[1],
                "time_bin_duration": spec.time_range[1] - spec.time_range[0],
            }

            counts = compute_epoch_bout_counts(
                fish_bouts, spec, valid_n_trials, laterality_labels[id(spec)]
            )
            for key, value in common_fields.items():
                counts[key] = value
            tables.append(counts)

            counts_classic = compute_epoch_bout_counts_classic(
                fish_bouts, spec, valid_n_trials, epoch_name_labels[id(spec)]
            )
            for key, value in common_fields.items():
                counts_classic[key] = value
            tables_classic.append(counts_classic)

    base_cols = [
        "trial_idx", "bout_category", "bout_counts", "bout_frequency",
        "file", "dpf", "day", "time_of_day_cos", "time_of_day_sin",
        "stim_name", "time_bin_start", "time_bin_stop", "time_bin_duration",
    ]

    per_fish = (
        pd.concat(tables, ignore_index=True) if tables
        else pd.DataFrame(columns=base_cols + ["laterality_group"])
    )
    per_fish_classic = (
        pd.concat(tables_classic, ignore_index=True) if tables_classic
        else pd.DataFrame(columns=base_cols + ["epoch_name", "sign_group"])
    )

    return per_fish, per_fish_classic


def aggregate_bout_frequency(
        per_fish: pd.DataFrame,
        average_trial: bool,
        average_time_bin: bool,
        split_columns: Tuple[str, ...] = ("laterality_group",),
    ) -> pd.DataFrame:
    """
    Aggregate a tidy per-fish bout frequency table, optionally collapsing
    the trial and/or time-bin dimensions. bout_category and every column in
    `split_columns` are never collapsed.

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

    split_columns = list(split_columns)
    working = per_fish

    if average_trial or average_time_bin:
        keep_cols = ["file", "stim_name", "bout_category"] + split_columns
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

    group_cols = ["stim_name", "bout_category"] + split_columns
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
    Assemble an aggregated bout-frequency table (split by laterality_group)
    into a single 2D matrix ready to be passed to imshow.

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
        title: str | None = None,
        cmap: str = 'inferno',
        clim: Tuple[float, float] = (0, MAX_COLORBAR),
    ) -> None:

    data = pivot.to_numpy(dtype=float)
    n_rows, n_cols = data.shape

    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=clim[0], vmax=clim[1])
    fig.colorbar(im, ax=ax, label='bout frequency', fraction=0.015, pad=0.01)

    # x ticks: time bin label per column -- skip entirely when there's no
    # real time-bin information (all columns would just say "avg", which
    # adds no information and only clutters the plot)
    show_time_bin_ticks = any(lbl != "avg" for lbl in time_bin_labels)
    if show_time_bin_ticks:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(time_bin_labels, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])

    # y ticks: trial number per row, if trials weren't averaged out
    has_trial_rows = isinstance(pivot.index, pd.MultiIndex)
    if has_trial_rows:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([t for _, t in pivot.index], fontsize=7)
    else:
        ax.set_yticks([])

    # Labels below use FIXED POINT offsets (via annotate + offset points),
    # not fractions of the data range -- this keeps them legible and
    # non-overlapping regardless of how many rows/columns the heatmap has
    # (unlike e.g. `-0.05 * n_rows`, which shrinks to nothing for small,
    # heavily-averaged heatmaps and causes labels to collide with each
    # other / the title).

    # stim-level separators + labels (well above the axes)
    for start, end, label in col_groups:
        if start > 0:
            ax.axvline(start - 0.5, color='white', lw=1.6)
        ax.annotate(
            label,
            xy=((start + end - 1) / 2, 1), xycoords=("data", "axes fraction"),
            xytext=(0, 38), textcoords="offset points",
            ha='center', va='bottom', fontsize=10, annotation_clip=False,
        )

    # (stim, laterality) separators + labels (just above the axes)
    for start, end, label in col_subgroups:
        if start > 0:
            ax.axvline(start - 0.5, color='white', lw=0.6, alpha=0.7)
        ax.annotate(
            label,
            xy=((start + end - 1) / 2, 1), xycoords=("data", "axes fraction"),
            xytext=(0, 16), textcoords="offset points",
            ha='center', va='bottom', fontsize=8, annotation_clip=False,
        )

    # bout category separators + labels (left of the axes)
    for idx, cat in enumerate(category_order):
        row_start = idx * n_trials
        row_end = row_start + n_trials
        if row_start > 0:
            ax.axhline(row_start - 0.5, color='white', lw=1.6)
        ax.annotate(
            cat,
            xy=(0, (row_start + row_end - 1) / 2), xycoords=("axes fraction", "data"),
            xytext=(-10, 0), textcoords="offset points",
            ha='right', va='center', fontsize=9, annotation_clip=False,
        )

    # title pad (points) is large enough to clear the stim-level group
    # labels above (offset 38 pts + their own text height)
    if title:
        ax.set_title(title, fontsize=12, pad=62)

    ax.set_xlabel("")
    ax.set_ylabel("")


# (average_trial, average_time_bin, filename_suffix, title)
HEATMAP_VARIANTS: List[Tuple[bool, bool, str, str]] = [
    (False, False, "", "trial x time bin"),
    (True, False, "_trial_avg", "averaged over trials"),
    (False, True, "_timebin_avg", "averaged over time bins"),
    (True, True, "_full_avg", "averaged over trials and time bins"),
]


def build_classic_bout_heatmap_matrix(
        avg: pd.DataFrame,
        category_order: List[str],
        stim_order: List[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build the "classic" heatmap matrix, matching the layout of the
    original (pre-refactor) heatmap:
      - rows:    flat list of (bout_category, sign_group) -- every bout
                 category always gets both LEFT and RIGHT rows, since every
                 bout has a sign regardless of stimulus.
      - columns: for each stim, TIME BIN is the outer loop and raw
                 epoch_name (stimulus direction, e.g. "grating right" vs
                 "grating left") is the inner loop -- i.e. directions are
                 INTERLEAVED per time bin (bin1_left, bin1_right,
                 bin2_left, bin2_right, ...), matching the original
                 pre-refactor code's `product(time_ranges, parameters)`
                 column ordering (time bin outer, parameter/direction
                 inner).
      - no group separators.

    `avg` must be the classic-specific table (epoch_name + sign_group
    columns), trial-averaged already (no `trial_idx`), with time bins still
    present.
    """

    if "trial_idx" in avg.columns:
        raise ValueError("classic heatmap expects trial-averaged data (no trial_idx column)")
    if "time_bin_start" not in avg.columns:
        raise ValueError("classic heatmap expects time bins to be preserved")
    if "epoch_name" not in avg.columns or "sign_group" not in avg.columns:
        raise ValueError("classic heatmap expects epoch_name and sign_group columns")

    signs = _order_signs(avg.sign_group.unique())

    row_index = pd.MultiIndex.from_tuples(
        [(cat, sign) for cat in category_order for sign in signs],
        names=["bout_category", "sign_group"],
    )

    col_tuples = []
    col_labels = []
    for stim in stim_order:
        stim_rows = avg[avg.stim_name == stim]
        if stim_rows.empty:
            continue

        # time bin OUTER, epoch_name (direction) INNER -- interleaves
        # e.g. "grating left"/"grating right" per bin, matching the
        # original plot's product(time_ranges, parameters) column order.
        bins = (
            stim_rows[["time_bin_start", "time_bin_stop"]]
            .drop_duplicates()
            .sort_values("time_bin_start")
        )
        epoch_names = sorted(stim_rows.epoch_name.unique())

        for t_start, t_stop in bins.itertuples(index=False):
            for epoch_name in epoch_names:
                has_data = (
                    (stim_rows.epoch_name == epoch_name) &
                    (stim_rows.time_bin_start == t_start)
                ).any()
                if not has_data:
                    continue
                col_tuples.append((epoch_name, t_start))
                col_labels.append(f"{epoch_name} | {t_start:g}-{t_stop:g}s")

    col_index = pd.MultiIndex.from_tuples(col_tuples, names=["epoch_name", "time_bin_start"])

    pivot = avg.pivot_table(
        index=["bout_category", "sign_group"],
        columns=["epoch_name", "time_bin_start"],
        values="bout_frequency",
    )
    pivot = pivot.reindex(index=row_index, columns=col_index)

    row_labels = [f"{cat}_{sign}" for cat, sign in pivot.index]

    return pivot, col_labels, row_labels


def plot_bout_heatmap_classic(
        fig: plt.Figure,
        ax: plt.Axes,
        pivot: pd.DataFrame,
        col_labels: List[str],
        row_labels: List[str],
        cmap: str = 'inferno',
        clim: Tuple[float, float] = (0, MAX_COLORBAR),
    ) -> None:
    """Simple, flat heatmap -- same style as the original pre-refactor plot."""

    data = pivot.to_numpy(dtype=float)

    im = ax.imshow(data, aspect='auto', cmap=cmap)
    im.set_clim(*clim)
    fig.colorbar(im, ax=ax, label='bout frequency')
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(col_labels, rotation=90, ha='center', fontsize=8)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("bout category")


def plot_heatmap(
        quality_control: Path,
        input_csv: Path,
        config_yaml: Path,
        output_png: Path,
        behavior_files: List[BehaviorFiles],
        interactive: bool = True
    ) -> None:

    output_csv = output_png.parent / 'bout_frequency.csv'
    output_csv_classic = output_png.parent / 'bout_frequency_classic.csv'

    cfg = load_yaml_config(config_yaml)
    per_fish, per_fish_classic = compute_bout_frequency_table(
        quality_control, input_csv, config_yaml, behavior_files
    )
    per_fish.to_csv(output_csv, index=False)
    per_fish_classic.to_csv(output_csv_classic, index=False)

    if per_fish.empty:
        print("No bouts found, skipping heatmap plots")
        return

    category_order = BOUT_CATEGORIES
    stim_order = stim_name_order(cfg)

    for average_trial, average_time_bin, suffix, title in HEATMAP_VARIANTS:

        avg = aggregate_bout_frequency(
            per_fish, average_trial, average_time_bin, split_columns=("laterality_group",)
        )
        avg.to_csv(output_png.parent / f'bout_frequency_avg{suffix}.csv', index=False)

        pivot, col_groups, col_subgroups, time_bin_labels, n_trials = build_bout_heatmap_matrix(
            avg, category_order, stim_order
        )

        n_rows, n_cols = pivot.shape
        fig_w = max(16, 0.22 * n_cols)
        fig_h = max(6, 0.28 * n_rows)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout='constrained')
        plot_bout_heatmap(
            fig, ax, pivot, category_order, col_groups, col_subgroups, time_bin_labels, n_trials,
            title=title,
        )

        variant_png = output_png.parent / f"{output_png.stem}{suffix}{output_png.suffix}"
        fig.savefig(variant_png, bbox_inches='tight')

    # classic heatmap: same layout as the original pre-refactor plot
    # (trial-averaged, flat category x LEFT/RIGHT rows, flat epoch_name x
    # time-bin columns -- stimulus direction NOT pooled -- no separator
    # lines).
    classic_avg = aggregate_bout_frequency(
        per_fish_classic, average_trial=True, average_time_bin=False,
        split_columns=("epoch_name", "sign_group"),
    )
    classic_pivot, classic_col_labels, classic_row_labels = build_classic_bout_heatmap_matrix(
        classic_avg, category_order, stim_order
    )
    classic_avg.to_csv(output_png.parent / 'bout_frequency_avg_classic.csv', index=False)

    fig_w = max(20, 0.35 * len(classic_col_labels))
    fig_h = max(10, 0.32 * len(classic_row_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout='constrained')
    plot_bout_heatmap_classic(fig, ax, classic_pivot, classic_col_labels, classic_row_labels)

    classic_png = output_png.parent / f"{output_png.stem}_classic{output_png.suffix}"
    fig.savefig(classic_png, bbox_inches='tight')

    if interactive:
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

    parser.add_argument(
        "--interactive",
        action='store_true'
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
        interactive: bool
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
        behavior_files,
        interactive
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
        interactive=args.interactive
    )


if __name__ == "__main__":

    main(build_parser().parse_args())