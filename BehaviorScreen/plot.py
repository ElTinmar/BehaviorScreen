from typing import List, Tuple, Generator, Any
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import yaml
from dataclasses import dataclass
import operator
from itertools import product

import matplotlib.pyplot as plt
from tqdm import tqdm

from megabouts.utils import bouts_category_name_short   
from BehaviorScreen.core import Stim, BoutSign
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
        return "_".join([f"{r.column}{r.operator}{r.value}" for r in self.rules])

@dataclass
class StimSpec:
    stim: Stim
    trials: range
    name: str
    time_range: Tuple[int, int] | None
    parameters: RuleSet

def load_bouts(bout_csv: Path) -> pd.DataFrame:
    return pd.read_csv(bout_csv)

def parse_rules(cfg: dict) -> RuleSet:
    rules = []

    for column, rule_dict in cfg.items():
        for op_name, value in rule_dict.items():
            rules.append(Rule(column, op_name, value))

    return RuleSet(tuple(rules))

def filter_bouts(bouts: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    filtered = bouts.copy()
    n0 = len(filtered)
    print(f'TOTAL NUM BOUTS: {n0}')

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

        parameters = [parse_rules(p) for p in entry.get("parameters", [None])]
        time_ranges = [None] if ignore_time_bins else bins

        for time_range, params in product(time_ranges, parameters):
            yield StimSpec(
                stim=stim,
                name=name,
                trials=trials,
                time_range=time_range,
                parameters=params,
            )

def plot_bout_heatmap(
        fig: plt.Figure, 
        ax: plt.Axes, 
        data: np.ndarray, 
        x_labels: list[str],
        y_labels: list[str],
        cmap: str = 'inferno', 
        clim: Tuple[float, float] = (0, 0.35)
    ) -> None:

    im = ax.imshow(data, aspect='auto', cmap=cmap)
    im.set_clim(*clim)
    fig.colorbar(im, ax=ax, label='bout frequency')
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(x_labels, rotation=90, ha='center')
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("epoch")
    ax.set_ylabel("bout category")

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

def stim_presented(behavior_data: BehaviorData, spec: StimSpec) -> bool:

    stim_trials = get_trials(behavior_data)
    
    if stim_trials.empty:
        return False
    
    spec_mask = (
        spec.parameters.get_mask(stim_trials) &
        (stim_trials.stim_select == spec.stim)
    )
    spec_data = stim_trials[spec_mask]
    if spec_data.empty: 
        return False
    
    # NOTE: this can miss the intent if the protocol has the given trial number but  
    # the protocol is different
    valid_trials = [i for i in spec.trials if i < len(spec_data)]
    if not valid_trials:
        return False
    
    if spec.time_range is not None:
        trial_data = spec_data.iloc[valid_trials]
        trial_duration = 1e-9 * (trial_data.stop_timestamp - trial_data.start_timestamp)
        valid_time_range = spec.time_range[0] < trial_duration 
        if not valid_time_range.any():
            return False

    return True

def plot_heatmap(
        input_csv: Path, 
        config_yaml: Path, 
        output_png: Path,
        behavior_files: List[BehaviorFiles]
    ) -> None:

    # TODO move the computation to megabouts and keep only the plotting here
    output_npz = output_png.with_suffix('.npz')
    output_csv = output_png.parent / 'bout_frequency.csv'

    cfg = load_yaml_config(config_yaml)
    stim_specs = list(read_stim_specs(cfg)) 
    bouts = load_bouts(input_csv)
    filtered_bouts = filter_bouts(bouts, cfg)
    sides = [BoutSign.LEFT, BoutSign.RIGHT]

    fish_names = filtered_bouts.file.unique()
    N_fish = len(fish_names)
    N_trials = max(filtered_bouts.trial_num)
    N_epochs = len(stim_specs)
    N_bouts = len(bouts_category_name_short)
    N_sides = len(sides)
    bout_frequency = np.full((N_fish, N_trials, N_epochs, N_bouts, N_sides), np.nan)
    rows = []

    fish_groups = filtered_bouts.groupby("file")
    
    for fish_idx, fish in tqdm(enumerate(fish_names)):

        if fish not in fish_groups.groups:
            continue
        fish_bouts = fish_groups.get_group(fish)

        fish_info = parse_fish(fish)
        time_cos, time_sin = cosinor(fish_info)
        
        behavior_data = get_behavior_data(behavior_files, fish)
        if behavior_data is None:
            raise RuntimeError(f"{fish} not found, aborting")

        stim_groups = fish_bouts.groupby("stim")

        for epoch_num, spec in enumerate(stim_specs):

            if spec.time_range is None:
                raise RuntimeError('time range should not be None')
            
            if not stim_presented(behavior_data, spec):
                print(f"{spec} not presented, skipping")
                continue
            
            if spec.stim not in stim_groups.groups:
                continue
            
            epoch_bouts = stim_groups.get_group(spec.stim)
            rule_mask = spec.parameters.get_mask(epoch_bouts)
            epoch_bouts = epoch_bouts[rule_mask]
            
            lo, hi = spec.time_range
            duration = hi - lo
            time_mask = (epoch_bouts.trial_time >= lo) & (epoch_bouts.trial_time < hi)
            epoch_bouts = epoch_bouts[time_mask]

            for trial_idx, trial_num in enumerate(spec.trials):
                trial_bouts = epoch_bouts[epoch_bouts.trial_num == trial_num]

                for category, cat_name in enumerate(bouts_category_name_short):
                    cat_bouts = trial_bouts[trial_bouts.category == category]

                    for side_idx, side in enumerate(sides):
                        
                        counts = (cat_bouts.sign == side).sum()
                        freq = counts / duration

                        bout_frequency[fish_idx, trial_idx, epoch_num, category, side_idx] = freq
                        # TODO add setup (oceanus vs chronus)?
                        # TODO measure and add fish length ?
                        rows.append({
                            "fish": fish,
                            "dpf": fish_info.age,
                            "day": f"{fish_info.day}.{fish_info.month}.{fish_info.year}",
                            "time_of_day_cos": time_cos,
                            "time_of_day_sin": time_sin,
                            "epoch_name": spec.name,
                            "stim_param": str(spec.parameters),
                            "trial_num": trial_idx,
                            "trial_time": spec.time_range[0] + duration/2,
                            "time_bin_duration": duration,
                            "bout_category": cat_name,
                            "bout_side": side,
                            "bout_frequency": freq,
                            'bout_counts': counts
                        })
    
    bout_frequency_tall = pd.DataFrame(rows)
    bout_frequency_tall.to_csv(output_csv, index=False)

    # save bout frequency table
    bin_names = [f"{s.name}_{s.parameters}_{s.time_range[0]}s-{s.time_range[1]}s" for s in stim_specs]     

    with open(output_npz, 'wb') as fp:
        np.savez(
            fp, 
            labels_0 = fish_names, 
            labels_1 = [f'trial_{n:03d}' for n in range(N_trials)],
            labels_2 = bin_names, 
            labels_3 = bouts_category_name_short,
            labels_4 = [str(s) for s in sides],
            bout_frequency = bout_frequency
        )

    bout_frequency_interleaved = bout_frequency.reshape(*bout_frequency.shape[:-2], -1)
    trial_avg = np.nanmean(bout_frequency_interleaved, axis=1)
    fish_avg =  np.nanmean(trial_avg, axis=0)
    row_names = [f"{cat}_{str(side)}" for cat in bouts_category_name_short for side in sides]

    # plot and save
    fig = plt.figure(figsize=(26, 14))
    ax = fig.gca()
    plot_bout_heatmap(fig, ax, fish_avg.T, bin_names, row_names)
    fig.tight_layout()
    plt.savefig(output_png)
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
        "--bouts-csv",
        default='bouts.csv',
        help="input CSV file",
    )

    parser.add_argument(
        "--bouts-png",
        default='bouts.png',
        help="output PNG file",
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

    input_csv = root / bouts_csv
    output_png = root / bouts_png

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

    plot_heatmap(input_csv, config_yaml, output_png, behavior_files)

def main(args: argparse.Namespace) -> None:

    run_plot(
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