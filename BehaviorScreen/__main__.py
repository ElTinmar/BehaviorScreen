from multiprocessing import Pool
from functools import partial

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
    track_with_SLEAP
)
from BehaviorScreen.plot import (
    plot_tracking_metrics, 
    plot_trajectories
)
from BehaviorScreen.get_models import download_and_extract_models


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
    export_single_animal_videos(directories, behavior_file, behavior_data)

def _run_metrics(behavior_file: BehaviorFiles, directories: Directories):
    behavior_data = load_data(behavior_file)
    well_coords_mm = get_well_coords_mm(directories, behavior_file, behavior_data)
    metrics = extract_metrics(behavior_data, well_coords_mm)

    for identity, data in metrics.items():
        plot_tracking_metrics(data)
        plot_trajectories(data)
    
if __name__ == '__main__':

    directories = Directories(BASE_DIR)
    behavior_files = find_files(directories)

    download_and_extract_models(MODELS_URL, MODELS_FOLDER)

    run_superimpose = partial(_run_superimpose, directories = directories)
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_superimpose, behavior_files)

    run_single_animal = partial(_run_single_animal, directories = directories)
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_single_animal, behavior_files)

    run_metrics = partial(_run_metrics, directories = directories)
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_metrics, behavior_files)
