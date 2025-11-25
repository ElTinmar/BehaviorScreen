from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
plt.plot()
plt.show()

from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Iterable, Sequence
import os

DARK_YELLOW = '#dbc300'

from BehaviorScreen.core import (
    GROUPING_PARAMETER, 
    BASE_DIR, 
    NUM_PROCESSES, 
    COLORS,
    Stim
)
from BehaviorScreen.load import (
    Directories, 
    BehaviorFiles,
    find_files, 
    load_data
)
from BehaviorScreen.process import (
    superimpose_video_trials,
    export_single_animal_videos,
    extract_time_series
)
from BehaviorScreen.plot import (
    plot_tracking_metrics, 
    plot_trajectories
)

from BehaviorScreen.megabouts import megabout_headtracking_pipeline, get_bout_metrics
from megabouts.utils import bouts_category_name_short
from scipy.stats import wilcoxon, friedmanchisquare, ttest_rel
import statsmodels.stats.multitest as smm
from statsmodels.stats.anova import AnovaRM
import itertools

# DLC
# TODO eye tracking OKR
# TODO eye tracking + tail tracking and classification J-turn PREY_CAPTURE

# megabouts
# TODO bout segmentation and distribution of heading change per bout
# TODO bout classification for every behavior + ethogram? 

# TODO auto detect edges/center coordinates on video/picture
# TODO filter bouts on edges (bout that starts and ends on the edge)?

# TODO separate analysis and plotting. Use multiprocessing for analysis here

# TODO linear mixed effects analysis to get within and between individual variability

# TODO filter dark/bright events to remove transition/rest periods

# TODO overlay reconstructed stimulus on top of video 

# TODO overlay video with ethogram

# TODO plot ethogram as in Marques et al CurrBiol 2018?


def _run_superimpose(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    superimpose_video_trials(directories, behavior_file, behavior_data, 30, GROUPING_PARAMETER)

def _run_single_animal(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    export_single_animal_videos(directories, behavior_file, behavior_data, quality=18)

def _run_megabouts(behavior_file: BehaviorFiles, directories: Directories) -> List[Dict]:
    behavior_data = load_data(behavior_file)
    meg = megabout_headtracking_pipeline(behavior_data)
    return get_bout_metrics(directories, behavior_data, behavior_file, meg)

def _run_timeseries(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    return extract_time_series(directories, behavior_data, behavior_file)

if __name__ == '__main__':

    directories = Directories(BASE_DIR)
    behavior_files = find_files(directories)
    
    #download_and_extract_models(MODELS_URL, MODELS_FOLDER)

    bouts_data = []
    for behavior_file in tqdm(behavior_files):
        bouts_data.extend(_run_megabouts(behavior_file, directories))
    bouts = pd.DataFrame(bouts_data)
    bouts.to_csv('bouts.csv')

    bouts = pd.read_csv(
        "bouts.csv",
        converters={
            "stim_variable_value": lambda x: str(x)
        }
    )

    # filtering outliers
    # bouts[bouts['distance_center']>9] = np.nan # remove bouts on the edge
    bouts.loc[bouts['distance']> 20, 'distance'] = np.nan
    bouts.loc[bouts['peak_axial_speed']> 300, 'peak_axial_speed'] = np.nan


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

    def friedman_wilcoxon_plot(
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

        # ---------- 1. Convert and check ----------
        groups = [np.asarray(g) for g in groups]
        k = len(groups)
        if k < 3:
            raise ValueError("Need at least 3 groups for Friedman test.")

        # ---------- 2. Friedman test ----------
        #friedman_stat, friedman_p = friedmanchisquare(*groups)
        friedman_stat, friedman_p = rm_anova(groups, group_names)

        # ---------- 3. Post-hoc Wilcoxon tests ----------
        pairs = list(itertools.combinations(range(k), 2))
        p_raw = []

        for i, j in pairs:
            _, p = ttest_rel(groups[i], groups[j], nan_policy='omit')
            #_, p = wilcoxon(groups[i], groups[j], nan_policy='omit')
            p_raw.append(p)

        # Holm correction
        reject, p_corrected, _, _ = smm.multipletests(p_raw, method="holm")

        # map corrected p-values back to pairs
        pair_results = [((pairs[i][0], pairs[i][1]), p_corrected[i]) for i in range(len(pairs))]

        # ---------- 4. Melt for plotting ----------
        df = pd.DataFrame({name: g for name, g in zip(group_names, groups)})
        df_m = df.melt(var_name="group", value_name="value")

        # ---------- 5. Plot points + means ----------
        ax.set_title(f'RM-ANOVA test: {friedman_p:.3f} ({asterisk(friedman_p)})')

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

        # ---------- 6. Draw non-overlapping significance bridges ----------
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
        #last = max(theta_avg_trials.groups.keys())
        theta_last = theta_avg_trials.get_group((last,)).values
        return theta_avg_trials, theta_last

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Prey capture')
    x, x_last = group(timeseries[(timeseries['stim']==Stim.PREY_CAPTURE) & (timeseries['stim_variable_value']=='20.0')], variable='theta')
    y, y_last = group(timeseries[(timeseries['stim']==Stim.PREY_CAPTURE) & (timeseries['stim_variable_value']=='-20.0')], variable='theta')
    ctl, ctl_last = group(timeseries[(timeseries['stim']==Stim.DARK)], variable='theta')
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
    friedman_wilcoxon_plot(
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
    friedman_wilcoxon_plot(
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
    friedman_wilcoxon_plot(
        ax[1],
        groups = [x_last, ctl_last, y_last],
        group_names=['Bright | Dark', 'dark', 'Dark | Bright'],
        ylabel='<cumulative angle (rad)>',
        colors=[COLORS[0], COLORS[2], COLORS[1]],
    )
    plt.savefig('phototaxis_timeseries_first3sec.png')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.title('Photokinesis')
    ax = plt.gca()
    plot_mean_and_sem(ax, timeseries[(timeseries['stim']==Stim.DARK)].groupby('time')['distance'], COLORS[0], label='Dark')
    plot_mean_and_sem(ax, timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]')].groupby('time')['distance'], COLORS[1], label='Bright')
    plt.ylabel('<cumulative distance (mm)>')
    plt.xlabel('time [s]')
    plt.legend()
    plt.savefig('photokinesis_timeseries.png')
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
    friedman_wilcoxon_plot(
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
    friedman_wilcoxon_plot(
        ax[1],
        groups = [x_last, ctl_last, y_last],
        group_names=['CW', 'dark', 'CCW'],
        ylabel='<cumulative angle (rad)>',
        colors=[COLORS[0], COLORS[2], COLORS[1]],
    )
    plt.savefig('OKR_timeseries.png')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.title('Loomings')
    ax = plt.gca()
    plot_mean_and_sem(ax, timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]')].groupby('time')['speed'], col=COLORS[0], label='Bright')
    plot_mean_and_sem(ax, timeseries[(timeseries['stim']==Stim.LOOMING)].groupby('time')['speed'], col=COLORS[1], label='Looming')
    plt.ylabel('<speed [mm/s]>')
    plt.xlabel('time [s]')
    plt.xlim(0,10)
    plt.savefig('Looming_timeseries.png')
    plt.show()

    # Bouts

    fig = plt.figure(figsize=(6,6))
    plt.title('prey capture')
    num_bouts = bouts[bouts['stim']==Stim.PREY_CAPTURE].shape[0]//2
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].apply(np.rad2deg).sample(num_bouts).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['stim_variable_value']=='20.0')]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[0], label='| o')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['stim_variable_value']=='-20.0')]['heading_change'].apply(np.rad2deg).plot.kde(color=COLORS[1], label='o |')
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
    num_bouts = bouts[bouts['stim']==Stim.PHOTOTAXIS].shape[0]//2
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True,label='dark')
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
    n = 4
    plt.title(f'phototaxis, first {n} bouts')
    first_bouts = bouts[bouts['stim']==Stim.PHOTOTAXIS].groupby(['file', 'identity', 'stim_variable_value', 'trial_num'], group_keys=False).head(n)
    num_bouts = first_bouts.shape[0]//2
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).apply(np.rad2deg).plot.hist(color='k', bins=80, alpha=0.1, density=True, label='dark')
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
    num_bouts = bouts[bouts['stim']==Stim.OMR].shape[0]//2
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
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
    num_bouts = bouts[bouts['stim']==Stim.OKR].shape[0]//2
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).apply(np.rad2deg).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
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
    num_bouts = bouts[(bouts['stim']==Stim.LOOMING)].shape[0]//2
    bouts[(bouts['stim']==Stim.BRIGHT)]['peak_yaw_speed'].sample(num_bouts).plot.hist(color='k', bins=80, alpha=0.5, density=True, label='bright')
    bouts[(bouts['stim']==Stim.LOOMING) & (bouts['stim_variable_value']=='2.0')]['peak_yaw_speed'].plot.kde(color=COLORS[0], label='o | ')
    bouts[(bouts['stim']==Stim.LOOMING) & (bouts['stim_variable_value']=='-2.0')]['peak_yaw_speed'].plot.kde(color=COLORS[1], label=' | o')
    plt.xlabel('yaw speed (rad/sec)')
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
    plt.savefig('Looming_bouts.png')
    plt.show()

    # bout categories 

    stimuli = {
        Stim.DARK: ['[0.0, 0.0, 0.0, 1.0]'],
        Stim.BRIGHT: ['[0.2, 0.2, 0.0, 1.0]'],
        Stim.PREY_CAPTURE: ['-20.0', '20.0'],
        Stim.PHOTOTAXIS: ['-1.0','1.0'],
        Stim.OMR: ['-90.0', '90.0'],
        Stim.OKR: ['-36.0', '36.0'],
        Stim.LOOMING: ['-2.0', '2.0']
    }


    num_cat = len(bouts_category_name_short)
    sides = ['L', 'R']
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]
    full_index = list(range(num_cat * len(sides)))  

    heatmap_df = pd.DataFrame()

    for stim, param_list in stimuli.items():
        for p in param_list:
            # Filter bouts by stim, stim_variable_value, and probability
            df_sub = bouts[(bouts['stim'] == stim) & 
                        (bouts['stim_variable_value'] == p) & 
                        (bouts['proba'] > 0.5)]
            
            if stim == Stim.PHOTOTAXIS:
                df_sub = df_sub.groupby(['file', 'identity', 'stim_variable_value', 'trial_num'], group_keys=False).head(4)
            
            counts = []
            for cat in range(num_cat):
                # left
                left_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == -1)].shape[0]
                # right
                right_count = df_sub[(df_sub['category'] == cat) & (df_sub['sign'] == 1)].shape[0]
                counts.extend([left_count, right_count])
            
            counts = pd.Series(counts, index=row_labels)
            counts = counts / counts.sum()  # normalize
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


    run_superimpose = partial(_run_superimpose, directories = directories)
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_superimpose, behavior_files)

    run_single_animal = partial(_run_single_animal, directories = directories)
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_single_animal, behavior_files)

    run_metrics = partial(_run_metrics, directories = directories)
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_metrics, behavior_files)
