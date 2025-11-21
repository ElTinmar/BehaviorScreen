from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
plt.plot()
plt.show()

from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Iterable, Tuple

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
    extract_metrics, 
    get_well_coords_mm,
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
from scipy.stats import ranksums, ttest_rel

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

# TODO plot individual data (last point of the timeseries) and statistical tests

# TODO plot ethogram as in Marques et al CurrBiol 2018?

# TODO average over trials in one fish, then over fish

def _run_superimpose(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    superimpose_video_trials(directories, behavior_file, behavior_data, 30, GROUPING_PARAMETER)

def _run_single_animal(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    export_single_animal_videos(directories, behavior_file, behavior_data, quality=18)

def _run_metrics(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    well_coords_mm = get_well_coords_mm(directories, behavior_file, behavior_data)
    metrics = extract_metrics(behavior_data, well_coords_mm)

    for identity, data in metrics.items():
        plot_tracking_metrics(data)
        plot_trajectories(data)

def _run_megabouts(behavior_file: BehaviorFiles) -> List[Dict]:
    behavior_data = load_data(behavior_file)
    meg = megabout_headtracking_pipeline(behavior_data)
    return get_bout_metrics(behavior_data, behavior_file, meg)

def _run_timeseries(behavior_file: BehaviorFiles):
    behavior_data = load_data(behavior_file)
    return extract_time_series(behavior_data, behavior_file)

if __name__ == '__main__':

    directories = Directories(BASE_DIR)
    behavior_files = find_files(directories)
    
    #download_and_extract_models(MODELS_URL, MODELS_FOLDER)

    bouts_data = []
    for behavior_file in tqdm(behavior_files):
        bouts_data.extend(_run_megabouts(behavior_file))
    bouts = pd.DataFrame(bouts_data)
    bouts.to_csv('bouts.csv')

    bouts = pd.read_csv(
        "bouts.csv",
        converters={
            "stim_variable_value": lambda x: str(x)
        }
    )

    # filtering outliers
    bouts.loc[bouts['distance']> 20, 'distance'] = np.nan
    bouts.loc[bouts['peak_axial_speed']> 300, 'peak_axial_speed'] = np.nan

    timeseries_data = []
    for behavior_file in tqdm(behavior_files):
        timeseries_data.extend(_run_timeseries(behavior_file))
    timeseries = pd.DataFrame(timeseries_data)
    timeseries.to_csv('timeseries.csv')

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

    def plot_last_value(ax, x, y, names):
        a, b = x.get_group(29.99).values, y.get_group(29.99).values
        

    def asterisk(p_value: float) -> str:

        if p_value < 0.0001:
            significance = '****'
        elif p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns' 

        return significance

    def significance_bridge(ax,x,y,p_value,fontsize,prct_offset=0.05):

        bottom, top = ax.get_ylim()
        height = top-bottom
        offset = prct_offset * height

        Mx = np.nanmax(x) + 1.5 * offset
        My = np.nanmax(y) + 1.5 * offset
        Mxy = np.nanmax((Mx,My)) + offset
        ax.plot([0, 0, 1, 1], [Mx, Mxy, Mxy, My], color='#555555', lw=1.5)

        significance = asterisk(p_value)
        ax.text(
            0.5, Mxy + offset, 
            f'{significance}', 
            horizontalalignment='center', 
            fontsize=fontsize
        )
        
        ax.set_ylim(bottom, Mxy + 3*offset)

    def ranksum_plot(
            ax,
            x, 
            y, 
            cat_names: Iterable, 
            ylabel: str, 
            title: str,
            col: Iterable, 
            fontsize: int = 12, 
            ylim: Optional[Tuple] = None,
            *args, 
            **kwargs):
            
        stat, p_value = ranksums(x, y, nan_policy='omit', *args, **kwargs)

        df = pd.DataFrame({cat_names[0]: x, cat_names[1]: y})
        df_melted = df.melt(var_name='cat', value_name='val')

        ax.set_title(title)

        sns.stripplot(ax = ax,
            data=df_melted, x='cat', y='val', hue='cat',
            alpha=.5, legend=False, palette=sns.color_palette(col),
            s=7.5
        )
        sns.pointplot(
            ax = ax,
            data=df_melted, x='cat', y="val", hue='cat',
            linestyle="none", errorbar=None,
            marker="_", markersize=30, markeredgewidth=3,
            palette=sns.color_palette(col)
        )
        ax.set_xlim(-0.5, 1.5)

        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_box_aspect(1)

        significance_bridge(ax,x,y,p_value,fontsize)

        if ylim is not None:
            ax.set_ylim(*ylim)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.title('Prey capture')
    df = timeseries[
        (timeseries['stim'] == Stim.PREY_CAPTURE) &
        (timeseries['stim_variable_value'] == '20.0')
    ]
    theta_avg_trials = (
        df.groupby(['file', 'identity', 'time'])['theta']
        .mean()  # average over trials first
        .groupby(['time'])
    )
    theta_mean = theta_avg_trials.mean().values
    theta_sem = theta_avg_trials.sem().values
    theta_last = theta_avg_trials.get_group(29.99).values
    
    x = timeseries[(timeseries['stim']==Stim.PREY_CAPTURE) & (timeseries['stim_variable_value']=='20.0')].groupby('time')['theta']
    y = timeseries[(timeseries['stim']==Stim.PREY_CAPTURE) & (timeseries['stim_variable_value']=='-20.0')].groupby('time')['theta']
    plot_mean_and_sem(ax[0], x, COLORS[0], label='| o')
    plot_mean_and_sem(ax[0], y, COLORS[1], label='o |')
    ax[0].set_ylabel('<cumulative angle (rad)>')
    ax[0].set_xlabel('time [s]')
    ax[0].legend()
    ax[0].set_ylim(-2, 2)
    ax[0].text(
        x=-0.1,
        y=-2,       
        s="Right",
        ha='right',   
        va='center',     
        transform=plt.gca().get_yaxis_transform(),
        rotation=90
    )
    ax[0].text(
        x=-0.1,
        y=2,       
        s="Left",
        ha='right',   
        va='center',     
        transform=plt.gca().get_yaxis_transform(),
        rotation=90
    )
    ax[0].hlines(0, 0, 30, linestyles='dotted', color='k')
    ranksum_plot(
        ax[1],
        x.get_group(29.99).values, 
        y.get_group(29.99).values,
        ['20.0', '-20.0'],
        ylabel='',
        title='',
        col=COLORS
    )
    plt.savefig('preycapture_timeseries.png')
    plt.show()


    plt.figure(figsize=(6,6))
    plt.title('Phototaxis')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.PHOTOTAXIS) & (timeseries['stim_variable_value']=='1.0')].groupby('time')['theta'], COLORS[0], label='Bright | Dark')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.PHOTOTAXIS) & (timeseries['stim_variable_value']=='-1.0')].groupby('time')['theta'], COLORS[1], label='Dark | Bright')
    plt.ylabel('<cumulative angle (rad)>')
    plt.xlabel('time [s]')
    plt.ylim(-3, 3)
    plt.text(
        x=-0.1,
        y=-3,       
        s="Right",
        ha='right',   
        va='center',     
        transform=plt.gca().get_yaxis_transform(),
        rotation=90
    )
    plt.text(
        x=-0.1,
        y=3,       
        s="Left",
        ha='right',   
        va='center',     
        transform=plt.gca().get_yaxis_transform(),
        rotation=90
    )
    plt.hlines(0, 0, 30, linestyles='dotted', color='k')
    plt.legend()
    plt.savefig('phototaxis_timeseries.png')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.title('Photokinesis')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.DARK)].groupby('time')['distance'], COLORS[0], label='Dark')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]')].groupby('time')['distance'], COLORS[1], label='Bright')
    plt.ylabel('<cumulative distance (mm)>')
    plt.xlabel('time [s]')
    plt.legend()
    plt.savefig('photokinesis_timeseries.png')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.title('OMR')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.OMR) & (timeseries['stim_variable_value']=='90.0')].groupby('time')['theta'], COLORS[0], label='-->')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.OMR) & (timeseries['stim_variable_value']=='-90.0')].groupby('time')['theta'], COLORS[1], label='<--')
    plt.ylabel('<cumulative angle (rad)>')
    plt.xlabel('time [s]')
    plt.ylim(-15, 15)
    plt.text(
        x=-0.1,
        y=-15,       
        s="Right",
        ha='right',   
        va='center',     
        transform=plt.gca().get_yaxis_transform(),
        rotation=90
    )
    plt.text(
        x=-0.1,
        y=15,       
        s="Left",
        ha='right',   
        va='center',     
        transform=plt.gca().get_yaxis_transform(),
        rotation=90
    )
    plt.hlines(0, 0, 30, linestyles='dotted', color='k')
    plt.legend()
    plt.savefig('OMR_timeseries.png')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.title('OKR')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.OKR) & (timeseries['stim_variable_value']=='36.0')].groupby('time')['theta'], COLORS[0], label='CW')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.OKR) & (timeseries['stim_variable_value']=='-36.0')].groupby('time')['theta'], COLORS[1], label='CCW')
    plt.ylabel('<cumulative angle (rad)>')
    plt.xlabel('time [s]')
    plt.ylim(-8, 8)
    plt.text(
        x=-0.1,
        y=-8,       
        s="Right",
        ha='right',   
        va='center',     
        transform=plt.gca().get_yaxis_transform(),
        rotation=90
    )
    plt.text(
        x=-0.1,
        y=8,       
        s="Left",
        ha='right',   
        va='center',     
        transform=plt.gca().get_yaxis_transform(),
        rotation=90
    )
    plt.hlines(0, 0, 30, linestyles='dotted', color='k')
    plt.legend()
    plt.savefig('OKR_timeseries.png')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.title('Loomings')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.BRIGHT) & (timeseries['stim_variable_value']=='[0.2, 0.2, 0.0, 1.0]')].groupby('time')['speed'], col=COLORS[0], label='Bright')
    plot_mean_and_sem(timeseries[(timeseries['stim']==Stim.LOOMING)].groupby('time')['speed'], col=COLORS[1], label='Looming')
    plt.ylabel('<speed [mm/s]>')
    plt.xlabel('time [s]')
    plt.xlim(0,10)
    plt.savefig('Looming_timeseries.png')
    plt.show()


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

    fig = plt.figure(figsize=(12,6))
    num_cat = len(bouts_category_name_short)
    counts = bouts[(bouts['stim'] == Stim.PREY_CAPTURE) & (bouts['proba']>0.8)]['category'].value_counts().sort_index()
    plt.bar(counts.index, counts.values, width=0.8)
    plt.xticks(range(num_cat), bouts_category_name_short)
    plt.xlim(-0.5, num_cat-0.5)
    plt.show()

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
    plt.show(block=False)

    # TODO select only subset
    fig = plt.figure(figsize=(6,6))
    bouts[(bouts['stim']==Stim.DARK)]['distance'].plot.hist(bins=180, alpha=0.5, density=True)
    bouts[(bouts['stim']==Stim.BRIGHT)]['distance'].plot.hist(bins=180, alpha=0.5, density=True)
    plt.show()

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

    fig = plt.figure(figsize=(12,6))
    num_cat = len(bouts_category_name_short)
    counts = bouts[(bouts['stim'] == Stim.OMR) & (bouts['proba']>0.8)]['category'].value_counts().sort_index()
    plt.bar(counts.index, counts.values, width=0.8)
    plt.xticks(range(num_cat), bouts_category_name_short)
    plt.xlim(-0.5, num_cat-0.5)
    plt.show()

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

    fig = plt.figure(figsize=(12,6))
    num_cat = len(bouts_category_name_short)
    counts = bouts[(bouts['stim'] == Stim.OKR) & (bouts['proba']>0.8)]['category'].value_counts().sort_index()
    plt.bar(counts.index, counts.values, width=0.8)
    plt.xticks(range(num_cat), bouts_category_name_short)
    plt.xlim(-0.5, num_cat-0.5)
    plt.show()

    # TODO pick fastest bouts
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

    fast_bouts = bouts[(bouts['stim']==Stim.LOOMING) & (abs(bouts['peak_yaw_speed'])>=60)]
    fig = plt.figure(figsize=(12,6))
    num_cat = len(bouts_category_name_short)
    counts = fast_bouts[fast_bouts['proba']>0.8]['category'].value_counts().sort_index()
    plt.bar(counts.index, counts.values, width=0.8)
    plt.xticks(range(num_cat), bouts_category_name_short)
    plt.xlim(-0.5, num_cat-0.5)
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
