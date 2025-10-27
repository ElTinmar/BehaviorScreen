from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
plt.plot()
plt.show()

import pandas as pd
import numpy as np
from typing import List, Dict

DARK_YELLOW = '#dbc300'

from BehaviorScreen.core import (
    GROUPING_PARAMETER, 
    BASE_DIR, 
    NUM_PROCESSES, 
    MODELS_FOLDER, 
    MODELS_URL,
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
)
from BehaviorScreen.plot import (
    plot_tracking_metrics, 
    plot_trajectories
)
from BehaviorScreen.get_models import download_and_extract_models
from BehaviorScreen.megabouts import megabout_headtracking_pipeline, get_bout_metrics


from megabouts.utils import (
    bouts_category_name,
    bouts_category_name_short,
    bouts_category_color,
    cmp_bouts,
)

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
    metrics = get_bout_metrics(behavior_data, behavior_file, meg)
    return metrics
    
    
if __name__ == '__main__':

    directories = Directories(BASE_DIR)
    behavior_files = find_files(directories)
    
    #download_and_extract_models(MODELS_URL, MODELS_FOLDER)
    rows = []
    for behavior_file in behavior_files:
        print(behavior_file)
        rows.extend(_run_megabouts(behavior_file))
    bouts = pd.DataFrame(rows)
    bouts.to_csv('bouts.csv')

    # filtering outliers
    bouts.loc[bouts['distance']> 20, 'distance'] = np.nan
    bouts.loc[bouts['peak_axial_speed']> 300, 'peak_axial_speed'] = np.nan

    fig = plt.figure(figsize=(6,6))
    plt.title('prey capture')
    num_bouts = bouts[bouts['stim']==Stim.PREY_CAPTURE].shape[0]//2
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['stim_variable_value']==20)]['heading_change'].plot.kde(color=COLORS[0], label='prey Right')
    bouts[(bouts['stim']==Stim.PREY_CAPTURE) & (bouts['stim_variable_value']==-20)]['heading_change'].plot.kde(color=COLORS[1], label='prey Left')
    plt.xlim(-np.pi, np.pi)
    plt.legend()
    plt.text(
        x=-np.pi,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=np.pi,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (deg)')
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
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).plot.hist(color='k', bins=180, alpha=0.1, density=True,label='dark')
    bouts[(bouts['stim']==Stim.PHOTOTAXIS) & (bouts['stim_variable_value']==1)]['heading_change'].plot.kde(color=COLORS[0], label='Bright | Dark')
    bouts[(bouts['stim']==Stim.PHOTOTAXIS) & (bouts['stim_variable_value']==-1)]['heading_change'].plot.kde(color=COLORS[1], label='Dark | Bright')
    plt.xlim(-np.pi, np.pi)
    plt.legend()
    plt.text(
        x=-np.pi,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=np.pi,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (rad)')
    plt.show()

    fig = plt.figure(figsize=(6,6))
    n = 4
    plt.title(f'phototaxis, first {n} bouts')
    first_bouts = bouts[bouts['stim']==Stim.PHOTOTAXIS].groupby(['file', 'identity', 'stim_variable_value', 'trial_num'], group_keys=False).head(n)
    num_bouts = first_bouts.shape[0]//2
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).plot.hist(color='k', bins=80, alpha=0.1, density=True, label='dark')
    first_bouts[first_bouts['stim_variable_value']==1]['heading_change'].plot.kde(color=COLORS[0], label='Bright | Dark')
    first_bouts[first_bouts['stim_variable_value']==-1]['heading_change'].plot.kde(color=COLORS[1], label='Dark | Bright')
    plt.xlim(-np.pi, np.pi)
    plt.legend()
    plt.text(
        x=-np.pi,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=np.pi,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (rad)')
    plt.show(block=False)

    # TODO select only subset
    fig = plt.figure(figsize=(6,6))
    bouts[(bouts['stim']==Stim.DARK)]['distance'].plot.hist(bins=180, alpha=0.5, density=True)
    bouts[(bouts['stim']==Stim.BRIGHT)]['distance'].plot.hist(bins=180, alpha=0.5, density=True)
    plt.show()

    fig = plt.figure(figsize=(6,6))
    plt.title('OMR directional')
    num_bouts = bouts[bouts['stim']==Stim.OMR].shape[0]//2
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    bouts[(bouts['stim']==Stim.OMR) & (bouts['stim_variable_value']==90)]['heading_change'].plot.kde(color=COLORS[0], label='-->')
    bouts[(bouts['stim']==Stim.OMR) & (bouts['stim_variable_value']==-90)]['heading_change'].plot.kde(color=COLORS[1], label='<--')
    plt.xlim(-np.pi, np.pi)
    plt.legend()
    plt.text(
        x=-np.pi,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=np.pi,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (rad)')
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
    bouts[(bouts['stim']==Stim.DARK)]['heading_change'].sample(num_bouts).plot.hist(color='k', bins=180, alpha=0.1, density=True, label='dark')
    bouts[(bouts['stim']==Stim.OKR) & (bouts['stim_variable_value']==36)]['heading_change'].plot.kde(color=COLORS[0], label='CW')
    bouts[(bouts['stim']==Stim.OKR) & (bouts['stim_variable_value']==-36)]['heading_change'].plot.kde(color=COLORS[1], label='CCW')
    plt.xlim(-np.pi, np.pi)
    plt.legend()
    plt.text(
        x=-np.pi,
        y=-0.075,       
        s="Right",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.text(
        x=np.pi,
        y=-0.075,       
        s="Left",
        ha='center',   
        va='top',     
        transform=plt.gca().get_xaxis_transform() 
    )
    plt.xlabel('bout heading change (rad)')
    plt.show(block=False)

    fig = plt.figure(figsize=(12,6))
    num_cat = len(bouts_category_name_short)
    counts = bouts[(bouts['stim'] == Stim.OKR) & (bouts['proba']>0.8)]['category'].value_counts().sort_index()
    plt.bar(counts.index, counts.values, width=0.8)
    plt.xticks(range(num_cat), bouts_category_name_short)
    plt.xlim(-0.5, num_cat-0.5)
    plt.show()

    # sample as many bouts in bright
    fig = plt.figure(figsize=(6,6))
    num_bouts = bouts[bouts['stim']==Stim.LOOMING].shape[0]
    bouts[(bouts['stim']==Stim.BRIGHT)]['peak_axial_speed'].abs().sample(num_bouts).plot.hist(color=DARK_YELLOW, bins=180, alpha=0.5, density=True, label='bright')
    bouts[(bouts['stim']==Stim.LOOMING)]['peak_axial_speed'].abs().plot.hist(bins=180, alpha=0.5, density=True)
    plt.show()


    fig = plt.figure(figsize=(6,6))
    plt.title('Looming')
    bouts[(bouts['stim']==Stim.LOOMING)]['distance'].plot.hist(color='k', bins=180, alpha=0.2, density=True)
    bouts[(bouts['stim']==Stim.BRIGHT)]['distance'].sample(num_bouts).plot.hist(color=DARK_YELLOW, bins=180, alpha=0.5, density=True)
    plt.show(block=False)

    fig = plt.figure(figsize=(6,6))
    bouts[(bouts['stim']==Stim.BRIGHT)]['peak_yaw_speed'].sample(num_bouts).plot.hist(color=DARK_YELLOW, bins=180, alpha=0.5, density=True)
    bouts[(bouts['stim']==Stim.LOOMING) & (bouts['stim_variable_value']==2)]['peak_yaw_speed'].plot.kde(color=COLORS[0])
    bouts[(bouts['stim']==Stim.LOOMING) & (bouts['stim_variable_value']==-2)]['peak_yaw_speed'].plot.kde(color=COLORS[1])
    plt.show()

    fig = plt.figure(figsize=(12,6))
    num_cat = len(bouts_category_name_short)
    counts = bouts[(bouts['stim'] == Stim.LOOMING) & (bouts['proba']>0.8)]['category'].value_counts().sort_index()
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
