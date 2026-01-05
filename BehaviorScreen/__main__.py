from typing import List, Dict, Optional, Iterable, Sequence
import os
from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt
plt.plot()
plt.show()

import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np

from BehaviorScreen.core import (
    GROUPING_PARAMETER, 
    COLORS,
    Stim
)
from BehaviorScreen.config import (
    ROOT_FOLDER, 
    NUM_PROCESSES
)
from BehaviorScreen.load import (
    Directories, 
    BehaviorFiles,
    find_files, 
    load_data
)
from BehaviorScreen.video import superimpose_video_trials
from BehaviorScreen.process import extract_time_series
from BehaviorScreen.export import export_single_animal
from BehaviorScreen.plot import (
    plot_tracking_metrics, 
    plot_trajectories
)
from BehaviorScreen.megabouts import (
    megabout_headtracking_pipeline,
    megabout_fulltracking_pipeline, 
    get_bout_metrics,
    get_bout_metrics2,
)
from megabouts.utils import bouts_category_name_short
from scipy.stats import ttest_rel
import statsmodels.stats.multitest as smm
from statsmodels.stats.anova import AnovaRM
import itertools

# SLEAP
# TODO eye tracking OKR
# TODO eye tracking + tail tracking and classification J-turn PREY_CAPTURE

# TODO separate analysis and plotting. Use multiprocessing for analysis here

# TODO linear mixed effects analysis to get within and between individual variability

# TODO indentify the main source of variability within/between individuals

# TODO overlay reconstructed stimulus on top of video 

# TODO overlay video with ethogram

# TODO megabout segmentation sanity checks

# TODO permutation tests with DARK? 

# TODO: plot trajectories loomings 

def _run_superimpose(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    superimpose_video_trials(directories, behavior_file, behavior_data, 30, GROUPING_PARAMETER)

def _run_single_animal(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    export_single_animal(directories, behavior_file, behavior_data, quality=18)

def _run_megabouts(behavior_file: BehaviorFiles, directories: Directories) -> List[Dict]:
    behavior_data = load_data(behavior_file)
    megabout = megabout_headtracking_pipeline(behavior_data)
    return get_bout_metrics(directories, behavior_data, behavior_file, megabout)

def _run_megabouts_full(behavior_file: BehaviorFiles, directories: Directories) -> List[Dict]:
    behavior_data = load_data(behavior_file)
    megabout = megabout_fulltracking_pipeline(behavior_data)
    return get_bout_metrics2(directories, behavior_data, behavior_file, megabout)

def _run_timeseries(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    return extract_time_series(directories, behavior_data, behavior_file)

if __name__ == '__main__':

    directories = Directories(
        root = ROOT_FOLDER / 'WT_oct_2025',
        metadata = 'data',
        stimuli = 'data',
        tracking = 'data',
        temperature = 'data',
        video = 'video',
        video_timestamp = 'video',
        results = 'results',
        plots = 'plots'
    )
    behavior_files = find_files(directories)
    
    #download_and_extract_models(MODELS_URL, MODELS_FOLDER)

    bouts_data = []
    for behavior_file in tqdm(behavior_files):
        bouts_data.extend(_run_megabouts(behavior_file, directories))
    bouts = pd.DataFrame(bouts_data)
    bouts.to_csv('bouts.csv', mode="a")
  
    bouts = pd.read_csv(
        "bouts.csv",
        converters={
            "stim_variable_value": lambda x: str(x),
        }
    )

    # filtering outliers
    bouts[bouts['distance_center']>9] = np.nan # remove bouts on the edge
    bouts.loc[bouts['distance']> 20, 'distance'] = np.nan
    bouts.loc[bouts['peak_axial_speed']> 300, 'peak_axial_speed'] = np.nan

    bouts[
        (bouts['file']=='03_07dpf_WT_Fri_10_Oct_2025_14h35min10sec') &
        (bouts['identity']==0) 
    ] = np.nan


    write_header = True
    filename = "timeseries.csv"
    if os.path.exists(filename):
        os.remove(filename)

    for behavior_file in tqdm(behavior_files):
        df = pd.DataFrame(_run_timeseries(behavior_file, directories))
        df.to_csv(
            filename,
            mode="a",
            header=write_header,
            index=False
        )  
        write_header = False

    timeseries = pd.read_csv(
        "timeseries.csv",
        converters={
            "stim_variable_value": lambda x: str(x)
        }
    )
    
    # filtering outliers
    timeseries.loc[timeseries['speed']> 400, 'speed'] = np.nan

    # tracking failure during bright 
    timeseries[
        (timeseries['file']=='03_07dpf_WT_Fri_10_Oct_2025_14h35min10sec') &
        (timeseries['identity']==0) 
    ] = np.nan

    def plot_mean_and_sem(ax, x, col='k', label=''):
        m = x.mean()
        s = x.sem()
        ax.plot(m.index, m.values, color = col, label=label)
        ax.fill_between(
            m.index,
            m.values - s.values,
            m.values + s.values,
            color = col,
            alpha = 0.3,
            edgecolor='none'
        )


    def rm_anova(groups, group_names):

        n_subjects = len(groups[0])
        
        # Build long-form DataFrame
        data = []
        for subj_idx in range(n_subjects):
            for cond_idx, g in enumerate(groups):
                data.append({
                    "subject": subj_idx,
                    "condition": group_names[cond_idx],
                    "value": g[subj_idx]
                })
        df = pd.DataFrame(data)

        # Run Repeated-Measures ANOVA
        anova = AnovaRM(df, depvar="value", subject="subject", within=["condition"])
        res = anova.fit()
        
        # Extract F-statistic and p-value
        F_stat = res.anova_table["F Value"].values[0]
        p_value = res.anova_table["Pr > F"].values[0]
        
        return F_stat, p_value

    def asterisk(p) -> str:
        if p < 1e-4: return "****"
        if p < 1e-3: return "***"
        if p < 1e-2: return "**"
        if p < 0.05: return "*"
        return "ns"

    def significance_bridges(ax, pairs, ystarts, fontsize=12):
        """
        Draw precomputed non-overlapping bridges.
        pairs: list of ((i,j), p_corrected)
        ystarts: list of y-coordinates (same length)
        """
        for (i, j), p, y in zip(pairs, pairs, ystarts):
            sign = asterisk(p[1])
            x0, x1 = p[0]
            M = y

            ax.plot([x0, x0, x1, x1],
                    [M, M+0.02*M, M+0.02*M, M],
                    lw=1.5, c="#555")

            ax.text((x0+x1)/2, M+0.03*M, sign,
                    ha='center', va='bottom', fontsize=fontsize)

    def anova_ttest_plot(
        ax,
        groups: Sequence[Iterable],
        group_names: Sequence[str],
        ylabel: str,
        colors: Iterable,
        fontsize: int = 12,
        ylim: Optional[tuple] = None
    ):
        """
        groups: e.g. [v0, v1, ctl] where each v is a vector of repeated measures
        """

        groups = [np.asarray(g) for g in groups]
        k = len(groups)
        if k >= 3:
            _, anova_p = rm_anova(groups, group_names)
            #_, anova_p = friedmanchisquare(*groups)
            ax.set_title(f'RM-ANOVA test: {anova_p:.3f} ({asterisk(anova_p)})')
        else:
            anova_p = 0

        # post-hoc tests
        pairs = list(itertools.combinations(range(k), 2))
        p_raw = []

        for i, j in pairs:
            _, p = ttest_rel(groups[i], groups[j], nan_policy='omit')
            #_, p = wilcoxon(groups[i], groups[j], nan_policy='omit')
            p_raw.append(p)

        # Holm correction
        if len(pairs) > 1:
            _, p_corrected, _, _ = smm.multipletests(p_raw, method="holm")
        else:
            p_corrected = p_raw

        # map corrected p-values back to pairs
        pair_results = [((pairs[i][0], pairs[i][1]), p_corrected[i]) for i in range(len(pairs))]

        df = pd.DataFrame({name: g for name, g in zip(group_names, groups)})
        df_m = df.melt(var_name="group", value_name="value")

        sns.stripplot(
            ax=ax,
            data=df_m,
            x="group",
            y="value",
            hue="group",
            dodge=False,
            alpha=.5,
            palette=colors,
            legend=False
        )

        sns.pointplot(
            ax=ax,
            data=df_m,
            x="group",
            y="value",
            hue="group",
            dodge=False,
            errorbar=None,
            markers="_",
            markersize=30, 
            markeredgewidth=3,
            linestyle="",
            palette=colors,            
            legend=False
        )
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")

        # Compute heights above each group
        ymax = np.nanmax(df.values)  
        height_step = 0.14 * ymax

        bridge_heights = []
        for idx, ((i, j), p) in enumerate(pair_results):
            base = ymax + (idx + 1) * height_step
            bridge_heights.append(base)

        for ((i, j), p_corr), y in zip(pair_results, bridge_heights):
            start = i+0.1
            stop = j-0.1
            sig = asterisk(p_corr)
            ax.plot([start, start, stop, stop], [y, y+height_step*0.3, y+height_step*0.3, y], color="#555", lw=1.5)
            ax.text((start+stop)/2, y+height_step*0.35, sig,
                    ha='center', va='bottom', fontsize=fontsize)

        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(np.min(groups)-2*height_step, max(bridge_heights) + 2*height_step)

    def group(df, variable, last=29.99):
        theta_avg_trials = (
            df.groupby(['file', 'identity', 'time'])[variable]
            .mean()  # average over trials first
            .groupby(['time'])
        )
        last = max(theta_avg_trials.groups.keys())
        theta_last = theta_avg_trials.get_group((last,)).values
        return theta_avg_trials, theta_last

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Prey capture')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.PREY_CAPTURE) & (timeseries['stim_variable_value']=='20.0')], variable='theta')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.PREY_CAPTURE) & (timeseries['stim_variable_value']=='-20.0')], variable='theta')
    ctl, ctl_last = group(timeseries[(timeseries['stim']==Stim.DARK) & (0 < timeseries['stim_start_time'])], variable='theta') # use all 20 dark trials
    plot_mean_and_sem(ax[0], ctl, COLORS[2], label='dark')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='o |')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='| o')
    ax[0].set_ylabel('<cumulative angle (rad)>')
    ax[0].set_xlabel('time [s]')
    ax[0].legend()
    ax[0].set_ylim(-2, 2)
    ax[0].text(
        x=-0.15,
        y=-2,       
        s="Right",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].text(
        x=-0.1,
        y=2,       
        s="Left",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].hlines(0, 0, 30, linestyles='dotted', color='k')
    anova_ttest_plot(
        ax[1],
        groups = [x_last, ctl_last, y_last],
        group_names=['| o', 'dark', 'o |'],
        ylabel='<cumulative angle (rad)>',
        colors=[COLORS[0], COLORS[2], COLORS[1]],
    )
    plt.savefig('preycapture_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Phototaxis')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.PHOTOTAXIS) & (timeseries['stim_variable_value']=='1.0')], variable='theta')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.PHOTOTAXIS) & (timeseries['stim_variable_value']=='-1.0')], variable='theta')
    ctl, ctl_last = group(timeseries[(timeseries['stim']==Stim.DARK)], variable='theta')
    plot_mean_and_sem(ax[0], ctl, COLORS[2], label='dark')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='Bright | Dark')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='Dark | Bright')
    ax[0].set_ylabel('<cumulative angle (rad)>')
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylim(-3, 3)
    ax[0].text(
        x=-0.1,
        y=-3,       
        s="Right",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].text(
        x=-0.1,
        y=3,       
        s="Left",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].hlines(0, 0, 30, linestyles='dotted', color='k')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, ctl_last, y_last],
        group_names=['Bright | Dark', 'dark', 'Dark | Bright'],
        ylabel='<cumulative angle (rad)>',
        colors=[COLORS[0], COLORS[2], COLORS[1]],
    )
    plt.savefig('phototaxis_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Phototaxis')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.PHOTOTAXIS) & (timeseries['stim_variable_value']=='1.0')], 'theta', 2.0)
    y, y_last = group(timeseries[(timeseries['stim']==Stim.PHOTOTAXIS) & (timeseries['stim_variable_value']=='-1.0')],'theta', 2.0)
    ctl, ctl_last = group(timeseries[(timeseries['stim']==Stim.DARK)], 'theta', 2.0)
    plot_mean_and_sem(ax[0], ctl, COLORS[2], label='dark')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='Bright | Dark')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='Dark | Bright')
    ax[0].set_ylabel('<cumulative angle (rad)>')
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylim(-0.4, 0.4)
    ax[0].set_xlim(0, 2)
    ax[0].text(
        x=-0.1,
        y=-3,       
        s="Right",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].text(
        x=-0.1,
        y=3,       
        s="Left",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].hlines(0, 0, 30, linestyles='dotted', color='k')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, ctl_last, y_last],
        group_names=['Bright | Dark', 'dark', 'Dark | Bright'],
        ylabel='<cumulative angle (rad)>',
        colors=[COLORS[0], COLORS[2], COLORS[1]],
    )
    plt.savefig('phototaxis_timeseries_first3sec.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Photokinesis')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.DARK) & (timeseries['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < timeseries['stim_start_time'])], 'speed')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (timeseries['stim_start_time'] < 3000)], 'speed')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='Dark')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='Bright')
    ax[0].set_ylabel('<speed (mm.s-1)>')
    ax[0].set_xlabel('time [s]')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, y_last],
        group_names=['dark', 'bright'],
        ylabel='<speed (mm.s-1)>',
        colors=[COLORS[0], COLORS[1]],
    )
    plt.savefig('photokinesis_speed_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Photokinesis')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.DARK) & (timeseries['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < timeseries['stim_start_time'])], 'theta')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (timeseries['stim_start_time'] < 3000)], 'theta')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='Dark')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='Bright')
    ax[0].set_ylabel('<heading change (rad)>')
    ax[0].set_xlabel('time [s]')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, y_last],
        group_names=['dark', 'bright'],
        ylabel='<heading change (rad)>',
        colors=[COLORS[0], COLORS[1]],
    )
    plt.savefig('photokinesis_heading_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Photokinesis')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.DARK) & (timeseries['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < timeseries['stim_start_time'])], 'distance')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (timeseries['stim_start_time'] < 3000)], 'distance')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='Dark')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='Bright')
    ax[0].set_ylabel('<distance (mm)>')
    ax[0].set_xlabel('time [s]')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, y_last],
        group_names=['dark', 'bright'],
        ylabel='<distance (mm)>',
        colors=[COLORS[0], COLORS[1]],
    )
    plt.savefig('photokinesis_distance_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Photokinesis')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.DARK) & (timeseries['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < timeseries['stim_start_time'])], 'distance_center')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (timeseries['stim_start_time'] < 3000)], 'distance_center')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='Dark')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='Bright')
    ax[0].set_ylabel('<radial distance (mm)>')
    ax[0].set_xlabel('time [s]')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, y_last],
        group_names=['dark', 'bright'],
        ylabel='<radial distance (mm)>',
        colors=[COLORS[0], COLORS[1]],
    )
    plt.savefig('photokinesis_radial_distance_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('OMR')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.OMR) & (timeseries['stim_variable_value']=='90.0')], variable='theta')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.OMR) & (timeseries['stim_variable_value']=='-90.0')], variable='theta')
    ctl, ctl_last = group(timeseries[(timeseries['stim']==Stim.DARK)], variable='theta')
    plot_mean_and_sem(ax[0], ctl, COLORS[2], label='dark')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='-->')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='<--')
    ax[0].set_ylabel('<cumulative angle (rad)>')
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylim(-15, 15)
    ax[0].text(
        x=-0.1,
        y=-15,       
        s="Right",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].text(
        x=-0.1,
        y=15,       
        s="Left",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].hlines(0, 0, 30, linestyles='dotted', color='k')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, ctl_last, y_last],
        group_names=['-->', 'dark', '<--'],
        ylabel='<cumulative angle (rad)>',
        colors=[COLORS[0], COLORS[2], COLORS[1]],
    )
    plt.savefig('OMR_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('OKR')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.OKR) & (timeseries['stim_variable_value']=='36.0')], variable='theta')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.OKR) & (timeseries['stim_variable_value']=='-36.0')], variable='theta')
    ctl, ctl_last = group(timeseries[(timeseries['stim']==Stim.DARK)], variable='theta')
    plot_mean_and_sem(ax[0], ctl, COLORS[2], label='dark')
    plot_mean_and_sem(ax[0], x, COLORS[0], label='CW')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='CCW')
    ax[0].set_ylabel('<cumulative angle (rad)>')
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylim(-8, 8)
    ax[0].text(
        x=-0.1,
        y=-8,       
        s="Right",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].text(
        x=-0.1,
        y=8,       
        s="Left",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].hlines(0, 0, 30, linestyles='dotted', color='k')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, ctl_last, y_last],
        group_names=['CW', 'dark', 'CCW'],
        ylabel='<cumulative angle (rad)>',
        colors=[COLORS[0], COLORS[2], COLORS[1]],
    )
    plt.savefig('OKR_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Loomings')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (timeseries['stim_start_time'] < 3000)], 'speed', 5.0)
    y, y_last = group(timeseries[(timeseries['stim']==Stim.LOOMING)], 'speed', 5.0)
    plot_mean_and_sem(ax[0], x, col=COLORS[0], label='Bright')
    plot_mean_and_sem(ax[0], y, col=COLORS[1], label='Looming')
    ax[0].set_xlim(0, 10)
    ax[0].set_ylabel('<speed (mm.s-1)>')
    ax[0].set_xlabel('time [s]')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, y_last],
        group_names=['brigth', 'looming'],
        ylabel='<speed (mm.s-1)>',
        colors=[COLORS[0], COLORS[1]],
    )
    plt.savefig('Looming_speed_timeseries.png')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Looming')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.LOOMING) & (timeseries['stim_variable_value']=='3.0')], 'theta', 10.0)
    y, y_last = group(timeseries[(timeseries['stim']==Stim.LOOMING) & (timeseries['stim_variable_value']=='-3.0')], 'theta', 10.0)
    plot_mean_and_sem(ax[0], x, COLORS[0], label='o | ')
    plot_mean_and_sem(ax[0], y, COLORS[1], label=' | o')
    ax[0].set_xlim(0, 10)
    ax[0].set_ylabel('<cumulative angle (rad)>')
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylim(-0.5, 0.5)
    ax[0].text(
        x=-0.1,
        y=-0.5,       
        s="Right",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].text(
        x=-0.1,
        y=0.5,       
        s="Left",
        ha='right',   
        va='center',     
        transform=ax[0].get_yaxis_transform(),
        rotation=90
    )
    ax[0].hlines(0, 0, 30, linestyles='dotted', color='k')
    ax[0].legend()
    anova_ttest_plot(
        ax[1],
        groups = [x_last, y_last],
        group_names=['o | ', ' | o'],
        ylabel='<cumulative angle (rad)>',
        colors=[COLORS[0], COLORS[1]],
    )
    plt.savefig('looming_angle_timeseries.png')
    plt.show()

    # Bouts

    fig = plt.figure(figsize=(6,6))
    plt.title('prey capture first 0-5 sec')
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['stim_variable_value']=='20.0') & (bouts['trial_time']<=5)]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[0], label='| o')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['stim_variable_value']=='-20.0') & (bouts['trial_time']<=5)]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[1], label='o |')
    plt.xlim(-180, 180)
    plt.legend()
    plt.text(
        x=-180,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=180,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (deg)')
    plt.savefig('preycapture_bouts.png')
    plt.show(block=False)

    fig = plt.figure(figsize=(6,6))
    plt.title('prey capture first 25-30 sec')
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['stim_variable_value']=='20.0') & (bouts['trial_time']<=5)]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[0], label='| o')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['stim_variable_value']=='-20.0') & (bouts['trial_time']<=5)]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[1], label='o |')
    plt.xlim(-180, 180)
    plt.legend()
    plt.text(
        x=-180,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=180,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (deg)')
    plt.savefig('preycapture_bouts.png')
    plt.show(block=False)

    fig = plt.figure(figsize=(6,6))
    plt.title('phototaxis')
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True,label='dark')
    bouts[(bouts['stim']==Stim.PHOTOTAXIS) & (bouts['stim_variable_value']=='1.0')]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[0], label='Bright | Dark')
    bouts[(bouts['stim']==Stim.PHOTOTAXIS) & (bouts['stim_variable_value']=='-1.0')]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[1], label='Dark | Bright')
    plt.xlim(-180, 180)
    plt.legend()
    plt.text(
        x=-180,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=180,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (deg)')
    plt.savefig('phototaxis_bouts.png')
    plt.show()


    fig = plt.figure(figsize=(6,6))
    plt.title('photokinesis')
    bouts[(bouts['stim']==Stim.DARK) & (bouts['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < bouts['stim_start_time'])]['distance'].plot.kde(color=COLORS[0], label='Dark')
    bouts[(bouts['stim']==Stim.BRIGHT) & (bouts['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (bouts['stim_start_time'] < 3000)]['distance'].plot.kde(color=COLORS[1], label='Bright')
    plt.xlim(-1, 20)
    plt.legend()
    plt.xlabel('distance (mm)')
    plt.savefig('photokinesis_bout_distance.png')
    plt.show()

    fig = plt.figure(figsize=(6,6))
    plt.title('photokinesis')
    bouts[(bouts['stim']==Stim.DARK) & (bouts['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < bouts['stim_start_time'])]['peak_axial_speed'].plot.kde(color=COLORS[0], label='Dark')
    bouts[(bouts['stim']==Stim.BRIGHT) & (bouts['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (bouts['stim_start_time'] < 3000)]['peak_axial_speed'].plot.kde(color=COLORS[1], label='Bright')
    plt.xlim(-1, 200)
    plt.legend()
    plt.xlabel('peak axial speed (mm.s-1)')
    plt.savefig('photokinesis_bout_speed.png')
    plt.show()

    fig = plt.figure(figsize=(6,6))
    plt.title('photokinesis')
    bouts[(bouts['stim']==Stim.DARK) & (bouts['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < bouts['stim_start_time'])]['bout_duration'].plot.kde(color=COLORS[0], label='Dark')
    bouts[(bouts['stim']==Stim.BRIGHT) & (bouts['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (bouts['stim_start_time'] < 3000)]['bout_duration'].plot.kde(color=COLORS[1], label='Bright')
    plt.xlim(0, 1)
    plt.legend()
    plt.xlabel('bout_duration (s)')
    plt.savefig('photokinesis_bout_duration.png')
    plt.show()

    fig = plt.figure(figsize=(6,6))
    plt.title('photokinesis')
    bouts[(bouts['stim']==Stim.DARK) & (bouts['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < bouts['stim_start_time'])]['interbout_duration'].plot.kde(color=COLORS[0], label='Dark')
    bouts[(bouts['stim']==Stim.BRIGHT) & (bouts['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (bouts['stim_start_time'] < 3000)]['interbout_duration'].plot.kde(color=COLORS[1], label='Bright')
    plt.xlim(-1, 4)
    plt.legend()
    plt.xlabel('interbout_duration (s)')
    plt.savefig('photokinesis_interbout_duration.png')
    plt.show()

    fig = plt.figure(figsize=(6,6))
    plt.title('photokinesis')
    bouts[(bouts['stim']==Stim.DARK) & (bouts['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < bouts['stim_start_time'])]['heading_change'].plot.kde(color=COLORS[0], label='Dark')
    bouts[(bouts['stim']==Stim.BRIGHT) & (bouts['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (bouts['stim_start_time'] < 3000)]['heading_change'].plot.kde(color=COLORS[1], label='Bright')
    plt.xlim(-3, 3)
    plt.legend()
    plt.xlabel('heading_change (rad)')
    plt.savefig('photokinesis_heading_change.png')
    plt.show()

    fig = plt.figure(figsize=(6,6))
    plt.title('thigmotaxis dark/bright')
    bouts[(bouts['stim']==Stim.DARK) & (bouts['stim_variable_value']=='[0.0, 0.0, 0.0, 1.0]') & (1500 < bouts['stim_start_time'])]['distance_center'].plot.kde(color=COLORS[0], label='Dark')
    bouts[(bouts['stim']==Stim.BRIGHT) & (bouts['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]') & (bouts['stim_start_time'] < 3000)]['distance_center'].plot.kde(color=COLORS[1], label='Bright')
    plt.xlim(-1, 11)
    plt.legend()
    plt.xlabel('radial_distance (mm)')
    plt.savefig('photokinesis_radial_distance.png')
    plt.show()

    fig = plt.figure(figsize=(6,6))
    n = 4
    plt.title(f'phototaxis, first {n} bouts')
    first_bouts = bouts[bouts['stim']==Stim.PHOTOTAXIS].groupby(['file', 'identity', 'stim_variable_value', 'trial_num'], group_keys=False).head(n)
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    first_bouts[first_bouts['stim_variable_value']=='1.0']['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[0], label='Bright | Dark')
    first_bouts[first_bouts['stim_variable_value']=='-1.0']['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[1], label='Dark | Bright')
    plt.xlim(-180, 180)
    plt.legend()
    plt.text(
        x=-180,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=180,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (deg)')
    plt.savefig('phototaxis_bouts_first_4.png')
    plt.show(block=False)


    fig = plt.figure(figsize=(6,6))
    plt.title('OMR directional')
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    bouts[(bouts['stim']==Stim.OMR) & (bouts['stim_variable_value']=='90.0')]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[0], label='-->')
    bouts[(bouts['stim']==Stim.OMR) & (bouts['stim_variable_value']=='-90.0')]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[1], label='<--')
    plt.xlim(-180, 180)
    plt.legend()
    plt.text(
        x=-180,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=180,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (deg)')
    plt.savefig('OMR_bouts.png')
    plt.show(block=False)


    fig = plt.figure(figsize=(6,6))
    plt.title('OKR')
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    bouts[(bouts['stim']==Stim.OKR) & (bouts['stim_variable_value']=='36.0')]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[0], label='CW')
    bouts[(bouts['stim']==Stim.OKR) & (bouts['stim_variable_value']=='-36.0')]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[1], label='CCW')
    plt.xlim(-180, 180)
    plt.legend()
    plt.text(
        x=-180,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=180,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (rad)')
    plt.savefig('OKR_bouts.png')
    plt.show(block=False)

    fig = plt.figure(figsize=(6,6))
    plt.title('Looming')
    bouts[(bouts['stim']==Stim.BRIGHT)]['peak_yaw_speed'].plot.hist(color='k', bins=180, alpha=0.1, density=True, label='bright')
    bouts[(bouts['stim']==Stim.LOOMING) & (bouts['stim_variable_value']=='3.0') & (bouts['trial_time']>=4) & (bouts['trial_time']<=6)]['peak_yaw_speed'].plot.kde(color=COLORS[0], label='o | ', bw_method=0.1)
    bouts[(bouts['stim']==Stim.LOOMING) & (bouts['stim_variable_value']=='-3.0') & (bouts['trial_time']>=4) & (bouts['trial_time']<=6)]['peak_yaw_speed'].plot.kde(color=COLORS[1], label=' | o', bw_method=0.1)
    plt.xlabel('yaw speed (deg/sec)')
    plt.legend()
    plt.text(
        x=-200,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=200,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlim(-200, 200)
    plt.savefig('Looming_bouts_yaw_speed.png')
    plt.show()

    fig = plt.figure(figsize=(6,6))
    plt.title('Looming')
    bouts[(bouts['stim']==Stim.BRIGHT)]['heading_change'].plot.hist(color='k', bins=180, alpha=0.1, density=True, label='bright')
    bouts[(bouts['stim']==Stim.LOOMING) & (bouts['stim_variable_value']=='3.0') & (bouts['trial_time']>=4) & (bouts['trial_time']<=6)]['heading_change'].plot.kde(color=COLORS[0], label='o | ', bw_method=0.15)
    bouts[(bouts['stim']==Stim.LOOMING) & (bouts['stim_variable_value']=='-3.0') & (bouts['trial_time']>=4) & (bouts['trial_time']<=6)]['heading_change'].plot.kde(color=COLORS[1], label=' | o', bw_method=0.15)
    plt.xlabel('heading_change (rad)')
    plt.legend()
    plt.text(
        x=-3,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=3,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlim(-3, 3)
    plt.savefig('Looming_bouts_heading_change.png')
    plt.show()

    # bout categories 

    stimuli = {
        Stim.DARK: ['[0.0, 0.0, 0.0, 1.0]'],
        Stim.BRIGHT: ['[0.2, 0.2, 0.0, 1.0]'],
        Stim.PREY_CAPTURE: ['-20', '20'],
        Stim.PHOTOTAXIS: ['-1','1'],
        Stim.OMR: ['-90', '90', '0'],
        Stim.OKR: ['-36', '36'],
        Stim.LOOMING: ['-2', '2', '-3', '3']
    }


    num_cat = len(bouts_category_name_short)
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    full_index = list(range(num_cat * len(sides)))  

    heatmap_df = pd.DataFrame()

    for stim, param_list in stimuli.items():
        for p in param_list:
            df_sub = bouts[(bouts['stim'] == stim) & 
                        (bouts['stim_variable_value'] == p) & 
                        (bouts['proba'] > 0.5) &
                        (bouts['distance_center'] < 15)
                    ]
            
            if stim == Stim.PHOTOTAXIS:
                df_sub = df_sub.groupby(['file', 'identity', 'stim_variable_value', 'trial_num'], group_keys=False).head(4)

            if stim == Stim.LOOMING:
                df_sub = df_sub[(bouts['trial_time']>=4) & (bouts['trial_time']<=6)]

            if stim == Stim.PREY_CAPTURE:
                df_sub = df_sub[(bouts['trial_time']<=10)]
            
            counts = []
            for cat in range(num_cat):
                left_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == -1)].shape[0]
                right_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == 1)].shape[0]
                counts.extend([left_count, right_count])
            
            counts = pd.Series(counts, index=row_labels)
            counts = counts / counts.sum()  
            heatmap_df[stim.name + ' ' + str(p)] = counts

    plt.figure(figsize=(6, 8))
    plt.imshow(heatmap_df, aspect='auto', cmap='inferno')
    plt.colorbar(label='prob.')

    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90, ha='right')
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    plt.xlabel("Stimulus")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig('categories.png')
    plt.show()

## PREY CAPTURE

    num_cat = len(bouts_category_name_short)
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    full_index = list(range(num_cat * len(sides)))  

    heatmap_df = pd.DataFrame()

    stim = Stim.PREY_CAPTURE
    for start, stop in [(0,5),(5,10),(10,15),(15,20),(20,25)]:
        mask = (bouts['trial_time']>=start) & (bouts['trial_time']<=stop)
        for p in ['-20.0', '20.0']:
            df_sub = bouts[(bouts['stim'] == stim) & 
                        (bouts['stim_variable_value'] == p) & 
                        (bouts['proba'] > 0.5) &
                        (bouts['distance_center'] < 15) &
                        mask
                    ]

            counts = []
            for cat in range(num_cat):
                left_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == -1)].shape[0]
                right_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == 1)].shape[0]
                counts.extend([left_count, right_count])
            
            counts = pd.Series(counts, index=row_labels)
            counts = counts / counts.sum()  
            heatmap_df[stim.name + f'_{start}-{stop}s_' + str(p)] = counts
        
    plt.figure(figsize=(6, 8))
    plt.imshow(heatmap_df, aspect='auto', cmap='inferno')
    plt.colorbar(label='prob.')

    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90, ha='right')
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    plt.xlabel("Stimulus")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig('categories_pc.png')
    plt.show()

## PHOTOTAXIS

    num_cat = len(bouts_category_name_short)
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    full_index = list(range(num_cat * len(sides)))  

    heatmap_df = pd.DataFrame()

    stim = Stim.PHOTOTAXIS
    for start, stop in [(0,2.5),(2.5,5),(5,7.5),(7.5,10)]:
        mask = (bouts['trial_time']>=start) & (bouts['trial_time']<=stop)
        for p in ['-1.0', '1.0']:
            df_sub = bouts[(bouts['stim'] == stim) & 
                        (bouts['stim_variable_value'] == p) & 
                        (bouts['proba'] > 0.5) &
                        (bouts['distance_center'] < 15) &
                        mask
                    ]

            counts = []
            for cat in range(num_cat):
                left_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == -1)].shape[0]
                right_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == 1)].shape[0]
                counts.extend([left_count, right_count])
            
            counts = pd.Series(counts, index=row_labels)
            counts = counts / counts.sum()  
            heatmap_df[stim.name + f'_{start}-{stop}s_' + str(p)] = counts
        
    plt.figure(figsize=(6, 8))
    plt.imshow(heatmap_df, aspect='auto', cmap='inferno')
    plt.colorbar(label='prob.')

    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90, ha='right')
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    plt.xlabel("Stimulus")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig('categories_ptx.png')
    plt.show()

## LOOMINGS

    num_cat = len(bouts_category_name_short)
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    full_index = list(range(num_cat * len(sides)))  

    heatmap_df = pd.DataFrame()

    stim = Stim.LOOMING
    for start, stop in [(0,2.5),(2.5,5),(5,7.5),(7.5,10)]:
        mask = (bouts['trial_time']>=start) & (bouts['trial_time']<=stop)
        for p in ['-3.0', '3.0']:
            df_sub = bouts[(bouts['stim'] == stim) & 
                        (bouts['stim_variable_value'] == p) & 
                        (bouts['proba'] > 0.5) &
                        (bouts['distance_center'] < 15) &
                        mask
                    ]

            counts = []
            for cat in range(num_cat):
                left_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == -1)].shape[0]
                right_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == 1)].shape[0]
                counts.extend([left_count, right_count])
            
            counts = pd.Series(counts, index=row_labels)
            counts = counts / counts.sum()  
            heatmap_df[stim.name + f'_{start}-{stop}s_' + str(p)] = counts
        
    plt.figure(figsize=(6, 8))
    plt.imshow(heatmap_df, aspect='auto', cmap='inferno')
    plt.colorbar(label='prob.')

    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90, ha='right')
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    plt.xlabel("Stimulus")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig('categories_looming.png')
    plt.show()

## OMR

    num_cat = len(bouts_category_name_short)
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    full_index = list(range(num_cat * len(sides)))  

    heatmap_df = pd.DataFrame()

    stim = Stim.OMR
    for start, stop in [(0,5),(5,10),(10,15),(15,20),(20,25)]:
        mask = (bouts['trial_time']>=start) & (bouts['trial_time']<=stop)
        for p in ['-90.0', '90.0']:
            df_sub = bouts[(bouts['stim'] == stim) & 
                        (bouts['stim_variable_value'] == p) & 
                        (bouts['proba'] > 0.5) &
                        (bouts['distance_center'] < 15) &
                        mask
                    ]

            counts = []
            for cat in range(num_cat):
                left_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == -1)].shape[0]
                right_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == 1)].shape[0]
                counts.extend([left_count, right_count])
            
            counts = pd.Series(counts, index=row_labels)
            counts = counts / counts.sum()  
            heatmap_df[stim.name + f'_{start}-{stop}s_' + str(p)] = counts
        
    plt.figure(figsize=(6, 8))
    plt.imshow(heatmap_df, aspect='auto', cmap='inferno')
    plt.colorbar(label='prob.')

    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90, ha='right')
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    plt.xlabel("Stimulus")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig('categories_omr.png')
    plt.show()

## OKR

    num_cat = len(bouts_category_name_short)
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    full_index = list(range(num_cat * len(sides)))  

    heatmap_df = pd.DataFrame()

    stim = Stim.OKR
    for start, stop in [(0,5),(5,10),(10,15),(15,20),(20,25)]:
        mask = (bouts['trial_time']>=start) & (bouts['trial_time']<=stop)
        for p in ['-36.0', '36.0']:
            df_sub = bouts[(bouts['stim'] == stim) & 
                        (bouts['stim_variable_value'] == p) & 
                        (bouts['proba'] > 0.5) &
                        (bouts['distance_center'] < 15) &
                        mask
                    ]

            counts = []
            for cat in range(num_cat):
                left_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == -1)].shape[0]
                right_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == 1)].shape[0]
                counts.extend([left_count, right_count])
            
            counts = pd.Series(counts, index=row_labels)
            counts = counts / counts.sum()  
            heatmap_df[stim.name + f'_{start}-{stop}s_' + str(p)] = counts
        
    plt.figure(figsize=(6, 8))
    plt.imshow(heatmap_df, aspect='auto', cmap='inferno')
    plt.colorbar(label='prob.')

    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90, ha='right')
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    plt.xlabel("Stimulus")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig('categories_okr.png')
    plt.show()

## ALL

    # TODO add those: 
    # DARK: bouts[(bouts.stim == Stim.DARK) & (bouts.trial_num >= 10) & (bouts.trial_num < 20)]
    # O-BEND: bouts[(bouts.stim == Stim.DARK) & (bouts.trial_num >= 20) & (bouts.trial_num <25) & (bouts.trial_time<1)] 
    # BRIGHT: bouts[(bouts.stim == Stim.BRIGHT) & (bouts.stim_variable_value=='[0.2, 0.2, 0.0, 1.0]') & (bouts.trial_num >= 5)]

    num_cat = len(bouts_category_name_short)
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    full_index = list(range(num_cat * len(sides)))  

    heatmap_df = pd.DataFrame()

    for stim, param_list in stimuli.items():
        for start, stop in [(0,2.5),(2.5,5),(5,7.5),(7.5,10),(10,15),(15,20),(20,30)]:
            
            if (stim in [Stim.OKR, Stim.OMR, Stim.LOOMING]) & (start>=10):
                continue

            if (stim == Stim.PHOTOTAXIS) & (start>=5):
                continue

            mask = (bouts['trial_time']>=start) & (bouts['trial_time']<=stop)
            for p in param_list:
                df_sub = bouts[(bouts['stim'] == stim) & 
                            (bouts['stim_variable_value'] == p) & 
                            (bouts['proba'] > 0.5) &
                            (bouts['distance_center'] < 15) &
                            mask
                        ]
                
                counts = []
                for cat in range(num_cat):
                    left_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == -1)].shape[0]
                    right_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == 1)].shape[0]
                    counts.extend([left_count, right_count])
                
                counts = pd.Series(counts, index=row_labels)
                counts = counts / counts.sum()  
                heatmap_df[stim.name + f'_{start}-{stop}s_' + str(p)] = counts

    plt.figure(figsize=(20, 8))
    plt.imshow(heatmap_df, aspect='auto', cmap='inferno')
    plt.colorbar(label='prob.')

    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90, ha='center')
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    plt.xlabel("Stimulus")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig('categories_vs_time.png')
    plt.show()


    ## Prey capture phase

    HAT = 7
    JTURN = 6

    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(121, projection='polar')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['category']==HAT)].stim_phase.hist(ax = ax0, bins=36)
    ax0.set_title(bouts_category_name_short[HAT])
    ax0.vlines(x=np.pi/2, ymin=0, ymax=100, color='r')
    ax0.vlines(x=3*np.pi/2, ymin=0, ymax=100, color='r')

    ax1 = fig.add_subplot(122, projection='polar')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['category']==JTURN)].stim_phase.hist(ax = ax1, bins=36)
    ax1.set_title(bouts_category_name_short[JTURN])
    ax1.vlines(x=np.pi/2, ymin=0, ymax=150, color='r')
    ax1.vlines(x=3*np.pi/2, ymin=0, ymax=150, color='r')
    plt.savefig('prey_capture_phase.png')
    plt.show()

    num_cat = len(bouts_category_name_short)
    n = int(np.ceil(np.sqrt(num_cat)))

    fig = plt.figure(figsize=(12, 12))
    for idx, name in enumerate(bouts_category_name_short):
        ax = fig.add_subplot(n,n,idx+1,projection='polar')
        bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['category']==idx)].stim_phase.hist(ax=ax, bins=36)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig('prey_capture_phase_all.png')
    plt.show()


## 
    run_superimpose = partial(_run_superimpose, directories = directories)
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_superimpose, behavior_files)

    run_single_animal = partial(_run_single_animal, directories = directories)
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_single_animal, behavior_files)


    ##### After exporting single individuals

    directories = Directories(
        ROOT_FOLDER / 'oceanus',
        metadata='results',
        stimuli='results',
        tracking='results',
        full_tracking= 'video_preds_WT_dec_2025_oceanus',
        video='results',
        video_timestamp='results',
        results = 'results',
        plots = 'plots'
    )
    behavior_files = find_files(directories)

    bouts_data = []
    for behavior_file in tqdm(behavior_files):
        bouts_data.extend(_run_megabouts_full(behavior_file, directories))
    bouts = pd.DataFrame(bouts_data)
    bouts.to_csv('bouts_full_tracking.csv', mode="a", header=True, index=False)


### 

O_bends = bouts[(bouts.stim == Stim.DARK) & (bouts.trial_num >= 20) & (bouts.trial_num <25)]

n_O_bends = []
for i in range(25):
    n_O_bends.append(sum(bouts[(bouts.stim == Stim.DARK) & (bouts.trial_num == i) & (bouts.proba > 0.5) & (bouts.trial_time<1)].category == 10))


###

stimuli = {
    Stim.PREY_CAPTURE: ['-20', '20'],
    Stim.PHOTOTAXIS: ['-1','1'],
    Stim.OMR: ['-90', '90', '0'],
    Stim.OKR: ['-36', '36'],
    Stim.LOOMING: ['-3', '3']
}

# time bins (s)
time_bins = [
    (0, 2.5),
    (2.5, 5),
    (5, 7.5),
    (7.5, 10),
    (10, 15),
    (15, 20),
    (20, 30),
]

sides = ['L', 'R']
num_cat = len(bouts_category_name_short)
row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]

heatmap_df = pd.DataFrame()

epochs = {}

# DARK
for start, stop in time_bins:
    name = f"DARK_{start}-{stop}s"
    epochs[name] = (
        (bouts.stim == Stim.DARK) &
        (bouts.trial_num >= 10) &
        (bouts.trial_num < 20) &
        (bouts.trial_time >= start) &
        (bouts.trial_time <= stop)
    )
    
# BRIGHT
for start, stop in time_bins:
    name = f"BRIGHT_{start}-{stop}s"
    epochs[name] = (
        (bouts.stim == Stim.BRIGHT) &
        (bouts.stim_variable_value == '[0.2, 0.2, 0.0, 1.0]') &
        (bouts.trial_num >= 5) &
        (bouts.trial_time >= start) &
        (bouts.trial_time <= stop)
    )

for stim, param_list in stimuli.items():
    for start, stop in time_bins:
        for p in param_list:

            if stim in {Stim.OKR, Stim.OMR, Stim.LOOMING} and start >= 10:
                continue
            if stim is Stim.PHOTOTAXIS and start >= 5:
                continue

            name = f"{stim.name}_{p}_{start}-{stop}s"

            epochs[name] = (
                (bouts.stim == stim) &
                (bouts.stim_variable_value == p) &
                (bouts.trial_time >= start) &
                (bouts.trial_time <= stop)
            )

# O-BEND
for start, stop in time_bins:
    name = f"BRIGHT->DARK_{start}-{stop}s"
    epochs[name] = (
        (bouts.stim == Stim.DARK) &
        (bouts.trial_num >= 20) &
        (bouts.trial_num < 25) &
        (bouts.trial_time >= start) &
        (bouts.trial_time <= stop)
    )

for name, mask in epochs.items():

    df_sub = bouts[
        mask &
        (bouts.proba > 0.5) &
        (bouts.distance_center < 15)
    ]

    counts = []
    for cat in range(num_cat):
        left = df_sub[(df_sub.category == cat) & (df_sub.sign == -1)].shape[0]
        right = df_sub[(df_sub.category == cat) & (df_sub.sign == 1)].shape[0]
        counts.extend([left, right])

    counts = pd.Series(counts, index=row_labels)
    if counts.sum() > 0:
        counts /= counts.sum()

    heatmap_df[name] = counts

plt.figure(figsize=(20, 8))
plt.imshow(heatmap_df, aspect='auto', cmap='inferno')
plt.colorbar(label='prob.')

plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90, ha='center')
plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

plt.xlabel("Epoch")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("categories_vs_time.png")
plt.show()