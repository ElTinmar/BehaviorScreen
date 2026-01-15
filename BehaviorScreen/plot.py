from typing import List, Tuple, Optional, NamedTuple, Generator
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import yaml
from dataclasses import dataclass

import matplotlib.pyplot as plt

from megabouts.utils import bouts_category_name_short   
from BehaviorScreen.core import Stim

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
        help="Root experiment folder (e.g. WT_oct_2025)",
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

def filter_bouts(bouts: pd.DataFrame) -> pd.DataFrame:

    # TODO adapt this
    return bouts

    filtered_bouts = bouts.copy()
    filtered_bouts[filtered_bouts['distance_center'] > 9] = np.nan 
    filtered_bouts[filtered_bouts['proba'] < 0.75] = np.nan
    filtered_bouts[filtered_bouts['distance'] > 20] = np.nan
    filtered_bouts[filtered_bouts['peak_axial_speed'] > 300] = np.nan

    return filtered_bouts

class MaskResult(NamedTuple):
    name: str
    mask: pd.Series

@dataclass
class StimSpec:
    stim: Stim
    time_range: Optional[Tuple[int, int]] = None
    trial_range: Optional[Tuple[int, int]] = None
    param: Optional[str] = None


def create_mask(bouts: pd.DataFrame, stim_spec: StimSpec) -> Optional[MaskResult]:
    
    name_parts = [str(stim_spec.stim)]
    mask = (bouts.stim == stim_spec.stim)

    if stim_spec.param is not None:
        mask &= (bouts.stim_variable_value == stim_spec.param) 
        name_parts.append(stim_spec.param)

    if stim_spec.time_range is not None:
        lo, hi = stim_spec.time_range
        mask &= (bouts.trial_time >= lo) & (bouts.trial_time < hi)
        name_parts.append(f"{lo}s-{hi}s")    
    
    if stim_spec.trial_range is not None:
        lo, hi = stim_spec.trial_range
        mask &= (bouts.trial_num >= lo) & (bouts.trial_num < hi)
        name_parts.append(f"trial_#{lo}-trial_#{hi}")    

    name = "_".join(name_parts)

    return MaskResult(name, mask)

def load_yaml_config(path: Path) -> dict:
    """Load YAML config from file"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

STIM_MAP = {
    'DARK': Stim.DARK,
    'BRIGHT': Stim.BRIGHT,
    'PREY_CAPTURE': Stim.PREY_CAPTURE,
    'PHOTOTAXIS': Stim.PHOTOTAXIS,
    'OMR': Stim.OMR,
    'OKR': Stim.OKR,
    'LOOMING': Stim.LOOMING,
}

def expand_stimuli(cfg: dict) -> Generator[StimSpec, None, None]:

    global_time_bins = cfg.get("time_bins", [])

    for entry in cfg["stimuli"]:
        bins = entry.get("time_bins", global_time_bins)
        params = entry.get("params", [None])
        trials = entry.get("trial_range", None)

        for t_start, t_stop in bins:
            for p in params:
                yield StimSpec(
                    stim=STIM_MAP[entry["stim"]],
                    param=p,
                    time_range=(t_start, t_stop),
                    trial_range=trials,
                )

def construct_all_masks(bouts: pd.DataFrame, cfg_path: Path) -> List[MaskResult]:

    cfg = load_yaml_config(cfg_path)
    masks: List[MaskResult] = []

    for spec in expand_stimuli(cfg):
        result = create_mask(bouts, spec)
        if result is not None:
            masks.append(result)

    return masks

def add_heatmap_column(
        bouts: pd.DataFrame, 
        heatmap_df: pd.DataFrame,
        name: str, 
        mask: pd.Series
    ) -> pd.DataFrame:
    
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    df_sub = bouts[mask]

    counts = []
    for idx, cat_name in enumerate(bouts_category_name_short):
        left = df_sub[(df_sub.category == idx) & (df_sub.sign == -1)].shape[0]
        right = df_sub[(df_sub.category == idx) & (df_sub.sign == 1)].shape[0]
        counts.extend([left, right])

    counts = pd.Series(counts, index=row_labels)
    if counts.sum() > 0:
        counts /= counts.sum()

    heatmap_df[name] = counts
    
    return heatmap_df

def plot_bout_heatmap(fig, ax, heatmap_df, max_prob: float = 0.35) -> None:

    im = ax.imshow(heatmap_df, aspect='auto', cmap='inferno')
    im.set_clim(0, max_prob)
    fig.colorbar(im, ax=ax, label='prob.')
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=90, ha='center')
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Category")

def plot_heatmap(
        input_csv: Path, 
        config_yaml: Path, 
        output_png: Path
    ) -> None:

    bouts = load_bouts(input_csv)
    filtered_bouts = filter_bouts(bouts)

    heatmap_df = pd.DataFrame()
    for mask in construct_all_masks(filtered_bouts, config_yaml):
        add_heatmap_column(filtered_bouts, heatmap_df, mask.name, mask.mask)

    # plot and save
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    plot_bout_heatmap(fig, ax, heatmap_df)
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