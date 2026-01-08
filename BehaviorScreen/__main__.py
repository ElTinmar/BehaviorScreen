from BehaviorScreen.export import export_single_animals
from BehaviorScreen.megabouts import run_megabouts
from BehaviorScreen.lightning_pose import estimate_pose
from BehaviorScreen.plot import plot_bout_heatmap

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

if __name__ == '__main__':

    # 1. export ROIs separately
    export_single_animals()

    # 2. popse estimation with lightning pose
    estimate_pose()
    
    # 3. extract bout metrics in relation with stimuli
    run_megabouts()

    # 4. plot
    plot_bout_heatmap()