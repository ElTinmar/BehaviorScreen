from typing import List, Tuple, Optional, NamedTuple, Generator, Any
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import yaml
from dataclasses import dataclass
import operator

import matplotlib.pyplot as plt

from megabouts.utils import bouts_category_name_short   
from BehaviorScreen.core import Stim, BoutSign

def pd_series_in(s: pd.Series, v: Any) -> pd.Series:
    return s.isin(v)

def pd_series_not_in(s: pd.Series, v: Any) -> pd.Series:
    return ~s.isin(v)

_OPS = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "in": pd_series_in,
    "not_in": pd_series_not_in,
}

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

    return parser


def normalize_stim_values(x: str) -> str:

    _int_float = re.compile(r'^(-?\d+)\.0+$')

    m = _int_float.fullmatch(x)
    if m:
        return m.group(1)

    return x

def load_bouts(bout_csv: Path) -> pd.DataFrame:

    bouts = pd.read_csv(
        bout_csv,
        converters={
            "stim_variable_value": normalize_stim_values,
        }
    )

    return bouts

def filter_bouts(bouts: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    filtered = bouts.copy()
    n0 = len(filtered)

    for col, rule in cfg["filters"].items():
        op_name = rule["op"]
        value = rule["value"]

        if op_name not in _OPS:
            raise ValueError(f"Unknown operator '{op_name}' for column '{col}'")

        if col not in filtered.columns:
            raise KeyError(f"Column '{col}' not found in bouts DataFrame")

        op_func = _OPS[op_name]

        before = len(filtered)
        mask = op_func(filtered[col], value)
        filtered = filtered[mask]
        after = len(filtered)

        removed = before - after
        frac_total = removed / n0 if n0 else 0

        print(
            f"{col:25s} {op_name:>2} {value} "
            f"→ removed {removed:6d} ({frac_total:6.2%} of total)"
        )

    return filtered

@dataclass
class StimSpec:
    stim: Stim
    trials: range
    name: str
    time_range: Optional[Tuple[int, int]] = None
    param: Optional[str] = None
    
def create_mask(
        bouts: pd.DataFrame, 
        stim: Stim,
        trial_num: int,
        time_range: Optional[Tuple[int, int]] = None,
        param: Optional[str] = None,
    ) -> pd.Series:
    
    mask = (bouts.stim == stim) & (bouts.trial_num == trial_num)

    if param is not None:
        mask &= (bouts.stim_variable_value == param) 

    if time_range is not None:
        lo, hi = time_range
        mask &= (bouts.trial_time >= lo) & (bouts.trial_time < hi)

    return mask

def load_yaml_config(path: Path) -> dict:
    """Load YAML config from file"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def read_stim_specs(cfg: dict) -> Generator[StimSpec, None, None]:

    global_time_bins = cfg.get("time_bins", [])

    for entry in cfg["stimuli"]:
        bins = entry.get("time_bins", global_time_bins)
        params = entry.get("params", [None])

        trials = range(
            entry["trial_range"]["start"], 
            entry["trial_range"]["stop"], 
            entry["trial_range"]["step"]
        )

        for t_start, t_stop in bins:
            for p in params:
                yield StimSpec(
                    stim=Stim[entry["stim"]],
                    param=p,
                    name=entry["name"],
                    time_range=(t_start, t_stop),
                    trials=trials,
                )

def count_bouts(df: pd.DataFrame, sides: Tuple[BoutSign, BoutSign]) -> np.ndarray:
    
    counts = []
    for idx, _ in enumerate(bouts_category_name_short):
        side_0 = df[(df.category == idx) & (df.sign == sides[0])].shape[0]
        side_1 = df[(df.category == idx) & (df.sign == sides[1])].shape[0]
        counts.extend([side_0, side_1])

    return np.asarray(counts)
    
def plot_bout_heatmap(
        fig: plt.Figure, 
        ax: plt.Axes, 
        data: np.ndarray, 
        x_labels: list[str],
        y_labels: list[str],
        max_frequency: float = 0.35
    ) -> None:

    im = ax.imshow(data, aspect='auto', cmap='inferno')
    im.set_clim(0, max_frequency)
    fig.colorbar(im, ax=ax, label='bout frequency')
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(x_labels, rotation=90, ha='center')
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("epoch")
    ax.set_ylabel("bout category")

def plot_heatmap(
        input_csv: Path, 
        config_yaml: Path, 
        output_png: Path
    ) -> None:

    bouts = load_bouts(input_csv)
    cfg = load_yaml_config(config_yaml)
    output_npz = output_png.with_suffix('.npz')
    sides = [BoutSign.LEFT, BoutSign.RIGHT]

    filtered_bouts = filter_bouts(bouts, cfg)
    stim_specs = list(read_stim_specs(cfg)) # maybe return list[list[StimSpec]] grouped by stim type
    fish_names = filtered_bouts.file.unique()

    N_fish = len(fish_names)
    N_trials = max(filtered_bouts.trial_num)
    N_epochs = len(stim_specs)
    N_bouts = len(sides)*len(bouts_category_name_short)
    bout_frequency = np.full((N_fish, N_trials, N_epochs, N_bouts), np.nan)
    
    # Do I need (N_fish, N_stim_type, N_stim_parameter, N_trials, N_trial_time, N_bouts) ?
    # Add time of day?
    # Make a LM/GLM: bout frequency (cat x side) = b0 + b1*stim_type + b2*stim_parameter + b3*trial_num + b4*trial_time or something like that
    # -> split everything into equally (or increasingly long) spaced time bins ?
    
    for fish_idx, fish in enumerate(fish_names):
        fish_df = filtered_bouts[filtered_bouts.file == fish]
        for epoch_num, spec in enumerate(stim_specs):
            for trial_num in spec.trials:
                mask = create_mask(
                    fish_df, 
                    spec.stim,
                    trial_num,
                    spec.time_range,
                    spec.param
                )
                counts = count_bouts(fish_df[mask], sides)
                bout_frequency[fish_idx, trial_num, epoch_num, :] = counts / (spec.time_range[1] - spec.time_range[0])

    # save bout frequency table
    column_names = [f"{s.name}_{s.param}_{s.time_range[0]}s-{s.time_range[1]}s" for s in stim_specs]     
    row_names = [f"{cat}_{str(side)}" for cat in bouts_category_name_short for side in sides]

    with open(output_npz, 'wb') as fp:
        np.savez(
            fp, 
            fish_names=fish_names, 
            column_names=column_names, 
            row_names=row_names, 
            bout_frequency=bout_frequency
        )

    # TODO export (stim by stim?) variance structure table (ss_trial, ss_indiv, ss_time, ss_epoch, ss_total)
    # ideally largest source of variance is the epoch aka visual stimulus
    # stim with habituation should also display variance in time / trials
    trial_avg = np.nanmean(bout_frequency, axis=1)
    fish_avg =  np.nanmean(trial_avg, axis=0)
    fish_std = np.nanstd(trial_avg, axis=0)

    # plot and save
    fig = plt.figure(figsize=(22, 10))
    ax = fig.gca()
    plot_bout_heatmap(fig, ax, fish_avg.T, column_names, row_names)
    fig.tight_layout()
    plt.savefig(output_png)
    plt.show()

def main(args: argparse.Namespace) -> None:

    input_csv = args.root / args.bouts_csv
    output_png = args.root / args.bouts_png
    config_yaml = args.yaml
    plot_heatmap(input_csv, config_yaml, output_png)

if __name__ == "__main__":

    main(build_parser().parse_args())