from typing import List, Tuple, Optional, Any, Protocol, NamedTuple
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt

from megabouts.utils import bouts_category_name_short   
from BehaviorScreen.core import Stim

num_bouts_categories = len(bouts_category_name_short)

time_bins = [
    (0, 2.5),
    (2.5, 5),
    (5, 7.5),
    (7.5, 10),
    (10, 15),
    (15, 20),
    (20, 30),
]

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

class MaskFn(Protocol):
    def __call__(
        self, 
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Optional[MaskResult]:
        ...

# TODO there might be a way to use only one function and avoid code duplication
@dataclass
class StimSpec:
    stim: Stim
    max_time: Optional[float] = None
    max_trial: Optional[int] = None
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
        
        if stim_spec.max_time is not None:
            if lo > stim_spec.max_time:
                return None

        mask &= (bouts.trial_time >= lo) & (bouts.trial_time < hi)
        name_parts.append(f"{lo}s-{hi}s")    
    
    if stim_spec.trial_range is not None:

        lo, hi = stim_spec.trial_range

        if stim_spec.max_trial is not None:
            if lo > stim_spec.max_trial:
                return None

        mask &= (bouts.trial_num >= lo) & (bouts.trial_num < hi)
        name_parts.append(f"trial_#{lo}-trial_#{hi}")    

    name = "_".join(name_parts)

    return MaskResult(name, mask)

MASKS = {

}

def prey_capture_mask(
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Tuple[bool, str, pd.Series]:

    name = f"PREY_CAPTURE_{param}_{t_start}-{t_stop}s"
    mask = (
        (bouts.stim == Stim.PREY_CAPTURE) &
        (bouts.stim_variable_value == param) &
        (bouts.trial_time >= t_start) &
        (bouts.trial_time <= t_stop)
    )
    return MaskResult(name, mask)

def omr_mask(
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Tuple[bool, str, pd.Series]:

    if t_start >= 10:
        return None
    
    name = f"OMR_{param}_{t_start}-{t_stop}s"
    mask = (
        (bouts.stim == Stim.OMR) &
        (bouts.stim_variable_value == param) &
        (bouts.trial_time >= t_start) &
        (bouts.trial_time <= t_stop)
    )
    return MaskResult(name, mask)

def okr_mask(
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Tuple[bool, str, pd.Series]:

    if t_start >= 10:
        return None
    
    name = f"OKR_{param}_{t_start}-{t_stop}s"
    mask = (
        (bouts.stim == Stim.OKR) &
        (bouts.stim_variable_value == param) &
        (bouts.trial_time >= t_start) &
        (bouts.trial_time <= t_stop)
    )
    return MaskResult(name, mask)

def phototaxis_mask(
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Tuple[bool, str, pd.Series]:

    if t_start >= 5:
        return None
    
    name = f"PHOTOTAXIS_{param}_{t_start}-{t_stop}s"
    mask = (
        (bouts.stim == Stim.PHOTOTAXIS) &
        (bouts.stim_variable_value == param) &
        (bouts.trial_time >= t_start) &
        (bouts.trial_time <= t_stop)
    )
    return MaskResult(name, mask)

def loomings_mask(
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Tuple[bool, str, pd.Series]:

    if t_start >= 10:
        return None
    
    name = f"LOOMING_{param}_{t_start}-{t_stop}s"
    mask = (
        (bouts.stim == Stim.LOOMING) &
        (bouts.stim_variable_value == param) &
        (bouts.trial_time >= t_start) &
        (bouts.trial_time <= t_stop)
    )
    return MaskResult(name, mask)

# TODO: these are a bit brittle since they rely on the organization of the protocol
# (hardcoded trial_num for given stimulus)
def dark_mask(        
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Tuple[bool, str, pd.Series]:

    name = f"DARK_{t_start}-{t_stop}s"
    mask = (
        (bouts.stim == Stim.DARK) &
        (bouts.trial_num >= 10) &
        (bouts.trial_num < 20) &
        (bouts.trial_time >= t_start) &
        (bouts.trial_time <= t_stop)
    )
    return MaskResult(name, mask)

def bright_mask(        
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Tuple[bool, str, pd.Series]:

    name = f"BRIGHT_{t_start}-{t_stop}s"
    mask = (
        (bouts.stim == Stim.BRIGHT) &
        (bouts.stim_variable_value == '[0.2, 0.2, 0.0, 1.0]') &
        (bouts.trial_num >= 5) &
        (bouts.trial_time >= t_start) &
        (bouts.trial_time <= t_stop)
    )
    return MaskResult(name, mask)

def bright2dark_mask(        
        bouts: pd.DataFrame, 
        param: Optional[Any], 
        t_start: float, 
        t_stop: float
    ) -> Tuple[bool, str, pd.Series]:

    if t_start >= 5:
        return None
        
    name = f"BRIGHT->DARK_{t_start}-{t_stop}s"
    mask = (
        (bouts.stim == Stim.DARK) &
        (bouts.trial_num >= 20) &
        (bouts.trial_num < 25) &
        (bouts.trial_time >= t_start) &
        (bouts.trial_time <= t_stop)
    )
    return MaskResult(name, mask)

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

def add_stim(
        bouts: pd.DataFrame,
        heatmap_df: pd.DataFrame,
        mask_function: MaskFn,
        stim_parameters: List
    ) -> pd.DataFrame:

    for t_start, t_stop in time_bins:
        for param in stim_parameters:
            result = mask_function(bouts, param, t_start, t_stop)
            
            if result is None:
                continue
                
            heatmap_df = add_heatmap_column(
                bouts, 
                heatmap_df,
                result.name,
                result.mask
            )
    
    return heatmap_df

def plot_bout_heatmap(fig, ax, heatmap_df) -> None:

    im = ax.imshow(heatmap_df, aspect='auto', cmap='inferno')
    fig.colorbar(im, ax=ax, label='prob.')
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=90, ha='center')
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Category")

def plot_heatmap(input_csv: Path, output_png: Path) -> None:

    # load bouts
    bouts = load_bouts(input_csv)
    filtered_bouts = filter_bouts(bouts)

    # construct heatmap
    # TODO read from JSON file how you want to construct the heatmap
    heatmap_df = pd.DataFrame()
    heatmap_df = add_stim(filtered_bouts, heatmap_df, dark_mask, [None])
    heatmap_df = add_stim(filtered_bouts, heatmap_df, bright_mask, [None])
    heatmap_df = add_stim(filtered_bouts, heatmap_df, prey_capture_mask, ['-20', '20'])
    heatmap_df = add_stim(filtered_bouts, heatmap_df, phototaxis_mask, ['-1', '1'])
    heatmap_df = add_stim(filtered_bouts, heatmap_df, omr_mask, ['-90','90'])
    heatmap_df = add_stim(filtered_bouts, heatmap_df, omr_mask, ['0'])
    heatmap_df = add_stim(filtered_bouts, heatmap_df, okr_mask, ['-36','36'])
    heatmap_df = add_stim(filtered_bouts, heatmap_df, loomings_mask, ['-3','3'])
    heatmap_df = add_stim(filtered_bouts, heatmap_df, bright2dark_mask, [None])

    # plot and save
    fig = plt.figure(figsize=(20, 8))
    ax = fig.gca()
    plot_bout_heatmap(fig, ax, heatmap_df)
    fig.tight_layout()
    plt.savefig(output_png)
    plt.show()

def main(args: argparse.Namespace) -> None:

    input_csv = args.root / args.bouts_csv
    output_png = args.root / args.bouts_png
    plot_heatmap(input_csv, output_png)

if __name__ == "__main__":

    main(build_parser().parse_args())