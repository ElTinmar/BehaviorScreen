from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

from BehaviorScreen.core import (
    GROUPING_PARAMETER, 
    BASE_DIR, 
    NUM_PROCESSES, 
    MODELS_FOLDER, 
    MODELS_URL
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
    df = pd.DataFrame(rows)

    from .core import Stim
    
    plt.hist(df[(df['stim']==Stim.PREY_CAPTURE) & (df['stim_variable_value']==20)]['heading_change'], bins=180, alpha=0.5, density=True)
    plt.hist(df[(df['stim']==Stim.PREY_CAPTURE) & (df['stim_variable_value']==-20)]['heading_change'], bins=180, alpha=0.5, density=True)
    plt.show()

    plt.hist(df[(df['stim']==Stim.PHOTOTAXIS) & (df['stim_variable_value']==1)]['heading_change'], bins=180, alpha=0.5, density=True)
    plt.hist(df[(df['stim']==Stim.PHOTOTAXIS) & (df['stim_variable_value']==-1)]['heading_change'], bins=180, alpha=0.5, density=True)
    plt.show()

    # TODO select only subset
    plt.hist(df[(df['stim']==Stim.DARK)]['distance'], bins=180, alpha=0.5, density=True)
    plt.hist(df[(df['stim']==Stim.BRIGHT)]['distance'], bins=180, alpha=0.5, density=True)
    plt.show()

    plt.hist(df[(df['stim']==Stim.OMR) & (df['stim_variable_value']==90)]['heading_change'], bins=180, alpha=0.5, density=True)
    plt.hist(df[(df['stim']==Stim.OMR) & (df['stim_variable_value']==-90)]['heading_change'], bins=180, alpha=0.5, density=True)
    plt.show()

    plt.hist(df[(df['stim']==Stim.OKR) & (df['stim_variable_value']==36)]['heading_change'], bins=180, alpha=0.5, density=True)
    plt.hist(df[(df['stim']==Stim.OKR) & (df['stim_variable_value']==-36)]['heading_change'], bins=180, alpha=0.5, density=True)
    plt.show()

    # sample as many bouts in bright
    num_bouts = df[df['stim']==Stim.LOOMING].shape[0]
    plt.hist(df[(df['stim']==Stim.BRIGHT)]['peak_axial_speed'].abs().sample(num_bouts), bins=180, alpha=0.5, density=True)
    plt.hist(df[(df['stim']==Stim.LOOMING)]['peak_axial_speed'].abs(), bins=180, alpha=0.5, density=True)
    plt.show()

    plt.hist(df[(df['stim']==Stim.BRIGHT)]['distance'].sample(num_bouts), bins=180, alpha=0.5, density=True)
    plt.hist(df[(df['stim']==Stim.LOOMING)]['distance'], bins=180, alpha=0.5, density=True)
    plt.show()

    plt.hist(df[(df['stim']==Stim.LOOMING) & (df['stim_variable_value']==2)]['peak_yaw_speed'], bins=180, alpha=0.5, density=True)
    plt.hist(df[(df['stim']==Stim.LOOMING) & (df['stim_variable_value']==-2)]['peak_yaw_speed'], bins=180, alpha=0.5, density=True)
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
