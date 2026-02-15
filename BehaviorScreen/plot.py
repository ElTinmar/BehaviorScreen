from typing import Tuple, Optional, Generator, Any
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import yaml
from dataclasses import dataclass
import operator

import matplotlib.pyplot as plt
from tqdm import tqdm

from megabouts.utils import bouts_category_name_short   
from BehaviorScreen.core import Stim, BoutSign
from BehaviorScreen.load import base_regexp, FileNameInfo

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
    print(f'TOTAL NUM BOUTS: {n0}')

    for col, rule in cfg["filters"].items():
        for op_name, value in rule.items():

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
    time_range: Tuple[int, int]
    param: Optional[str] = None
    
def create_mask(
        bouts: pd.DataFrame, 
        fish: str,
        category: int,
        side: BoutSign,
        stim: Stim,
        trial_num: int,
        time_range: Tuple[int, int],
        param: Optional[str] = None,
    ) -> pd.Series:
    
    lo, hi = time_range

    mask = (
        (bouts.stim == stim) & 
        (bouts.trial_num == trial_num) &
        (bouts.file == fish) &
        (bouts.trial_time >= lo) & 
        (bouts.trial_time < hi) &
        (bouts.category == category) &
        (bouts.sign == side)
    )

    # TODO: more complex stim variable masks
    if param is not None:
        mask &= (bouts.stim_variable_value == param) 

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

def plot_heatmap(
        input_csv: Path, 
        config_yaml: Path, 
        output_png: Path
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
    
    # 5 nested for loops ... but I dont really care how long it takes
    for fish_idx, fish in tqdm(enumerate(fish_names)):
        fish_info = parse_fish(fish)
        time_cos, time_sin = cosinor(fish_info)
        for epoch_num, spec in enumerate(stim_specs):
            for trial_idx, trial_num in enumerate(spec.trials):
                for category, cat_name in enumerate(bouts_category_name_short):
                    for side_idx, side in enumerate(sides):
                        
                        mask = create_mask(
                            bouts = filtered_bouts, 
                            fish = fish,
                            category = category,
                            side = side,
                            stim = spec.stim,
                            trial_num = trial_num,
                            time_range = spec.time_range,
                            param = spec.param
                        )
                        counts = mask.sum() or np.nan # NOTE check that this is ok
                        duration = spec.time_range[1] - spec.time_range[0]
                        freq = counts / duration

                        bout_frequency[fish_idx, trial_idx, epoch_num, category, side_idx] = freq
                        # TODO add setup (oceanus vs chronus)?
                        # TODO measure and add fish length ?
                        # NOTE I want each cell in the heatmap to be a slope + intercept (ideally reducing time dimension)? 
                        # -> estimate number of parameters
                        # maybe slope in time + intercept for each cell in the heatmap?
                        # mabe dont bin bouts in time for the linear model and use counts / poisson?
                        rows.append({
                            "fish": fish,
                            "dpf": fish_info.age,
                            "day": f"{fish_info.day}.{fish_info.month}.{fish_info.year}",
                            "time_of_day_cos": time_cos,
                            "time_of_day_sin": time_sin,
                            "epoch_name": spec.name,
                            "stim_param": spec.param,
                            "trial_num": trial_idx,
                            "trial_time": spec.time_range[0] + duration/2,
                            "bout_category": cat_name,
                            "bout_side": side,
                            "bout_frequency": freq,
                            'bout_counts': counts
                        })
    
    bout_frequency_tall = pd.DataFrame(rows)
    bout_frequency_tall.to_csv(output_csv, index=False)

    # save bout frequency table
    bin_names = [f"{s.name}_{s.param}_{s.time_range[0]}s-{s.time_range[1]}s" for s in stim_specs]     

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
    fig = plt.figure(figsize=(22, 10))
    ax = fig.gca()
    plot_bout_heatmap(fig, ax, fish_avg.T, bin_names, row_names)
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