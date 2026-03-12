from pathlib import Path
from typing import List
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import NamedTuple
from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    BehaviorFiles,
    find_files, 
    load_data
)
from BehaviorScreen.megabouts import MegaboutResults
from BehaviorScreen.process import get_trials, compute_angle_between_vectors
from BehaviorScreen.core import Stim, GROUPING_PARAMETER
from BehaviorScreen.plot import load_yaml_config, read_stim_specs

class EyesTimeseries(NamedTuple):
    angle_left_deg: np.ndarray
    angle_right_deg: np.ndarray
    angle_left_smooth_deg: np.ndarray
    angle_right_smooth_deg: np.ndarray
    version_angle_deg: np.ndarray
    vergence_angle_deg: np.ndarray

def get_eye_traces(
        data: pd.DataFrame, 
        likelihood_threshold: float = 0.9,
        divergence_threshold_deg: float = -10,
        convergence_threshold_deg: float = 60,
        window_length: int = 41
    ) -> EyesTimeseries:

    assert window_length % 2 == 1

    # extract data
    left_front_keypoint = data.eye_left_front[['x', 'y']].to_numpy()
    left_front_likelihood = data.eye_left_front.likelihood.to_numpy()

    left_back_keypoint = data.eye_left_back[['x', 'y']].to_numpy()
    left_back_likelihood = data.eye_left_back.likelihood.to_numpy()

    right_front_keypoint = data.eye_right_front[['x', 'y']].to_numpy()
    right_front_likelihood = data.eye_right_front.likelihood.to_numpy()

    right_back_keypoint = data.eye_right_back[['x', 'y']].to_numpy()
    right_back_likelihood = data.eye_right_back.likelihood.to_numpy()

    left_vector = left_back_keypoint - left_front_keypoint
    right_vector = right_back_keypoint - right_front_keypoint

    L_rad = compute_angle_between_vectors(left_vector, np.array([0,1]))
    R_rad = compute_angle_between_vectors(right_vector, np.array([0,1]))
    L = np.rad2deg(L_rad)
    R = np.rad2deg(R_rad)

    # remove outliers
    L[(left_front_likelihood < likelihood_threshold) | (left_back_likelihood < likelihood_threshold)] = np.nan
    R[(right_front_likelihood < likelihood_threshold) | (right_back_likelihood < likelihood_threshold)] = np.nan
    L[(-L<divergence_threshold_deg) | (-L>convergence_threshold_deg)] = np.nan 
    R[(R<divergence_threshold_deg) | (R>convergence_threshold_deg)] = np.nan

    # interpolate and smooth
    L = pd.Series(L).interpolate(limit_direction="both").to_numpy()
    R = pd.Series(R).interpolate(limit_direction="both").to_numpy()
    L_s = savgol_filter(L, window_length, polyorder=2)
    R_s = savgol_filter(R, window_length, polyorder=2)
    version_angle = (L_s + R_s)/2
    vergence_angle = R_s - L_s

    res = EyesTimeseries(
        angle_left_deg=L,
        angle_right_deg=R,
        angle_left_smooth_deg=L_s,
        angle_right_smooth_deg=R_s,
        version_angle_deg=version_angle,
        vergence_angle_deg=vergence_angle
    )
    return res

ROOT = Path('/media/martin/DATA_18TB/Screen/WT/danieau')
ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/Screen/WT/danieau')
#ROOT = Path('/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/DATA/WT/danieau')

directories = Directories(
    root = ROOT,
    metadata='results',
    stimuli='results',
    tracking='results',
    full_tracking='lightning_pose',
    eyes_tracking='lightning_pose',
    temperature='results',
    video='results',
    video_timestamp='results',
    results='results',
    plots=''
)
files: List[BehaviorFiles] = find_files(directories)
behavior_file = files[0]
behavior_data: BehaviorData = load_data(behavior_file)

with open(ROOT / 'megabout.pkl', 'rb') as fp:
    mb = pickle.load(fp) 

megabout = mb[behavior_file.metadata.stem]

# ----------------------------
fs = 120
pooled_vergence = np.full((len(files), 500_000), np.nan)
pooled_version = np.full((len(files), 500_000), np.nan)
for idx, behavior_file in enumerate(files): 
    if '11_Dec' in str(behavior_file.metadata): # FIXME this is just a hack
        behavior_data: BehaviorData = load_data(behavior_file)
        eyes = get_eye_traces(behavior_data.eyes_tracking, likelihood_threshold=0.9)
        n = len(eyes.version_angle_deg)
        pooled_vergence[idx, 0:n] = eyes.vergence_angle_deg
        pooled_version[idx, 0:n] = eyes.version_angle_deg


plt.figure()
plt.title('WT')
plt.plot(np.arange(500_000)/fs, np.nanmean(pooled_vergence, axis=0)) 
plt.xlabel('time')
plt.ylabel('<vergence angle [deg]>')
plt.show()

plt.figure()
plt.title('WT')
plt.plot(np.arange(500_000)/fs, np.nanmean(pooled_version, axis=0))    
plt.xlabel('time')
plt.ylabel('<version angle [deg]>')
plt.show()

# --------------------------------------------------------------------------------
config_yaml = Path('BehaviorScreen/screen.yaml')
cfg = load_yaml_config(config_yaml)
stim_specs = list(read_stim_specs(cfg, ignore_time_bins=True))

N_fish = len(files)
N_trials = max([len(spec.trials) for spec in stim_specs])
N_epochs = len(stim_specs)
N_samples = 30 * 120

vergence_angle = np.full((N_fish, N_trials, N_epochs, N_samples), np.nan)
version_angle = np.full((N_fish, N_trials, N_epochs, N_samples), np.nan)

for idx, behavior_file in enumerate(files):

    print(idx)
    behavior_data: BehaviorData = load_data(behavior_file)
    #timestamps = behavior_data.tracking.timestamp.to_numpy()
    timestamps = behavior_data.video_timestamps.timestamp.to_numpy()
    stim_trials = get_trials(behavior_data)
    eyes = get_eye_traces(behavior_data.eyes_tracking, likelihood_threshold=0.9)

    for spec_idx, spec in enumerate(stim_specs):
        spec_mask = (
            spec.parameters.get_mask(stim_trials) &
            (stim_trials.stim_select == spec.stim)
        )
        spec_data = stim_trials[spec_mask]
        if spec_data.empty: 
            continue
        
        valid_trials = [i for i in spec.trials if i < len(spec_data)]
        trial_data = spec_data.iloc[valid_trials]

        for trial_idx, (trial, row) in enumerate(trial_data.iterrows()):
            mask = (timestamps > row.start_timestamp) & (timestamps < row.stop_timestamp) 
            n = sum(mask)
            version_angle[idx, trial_idx, spec_idx, 0:n] = eyes.version_angle_deg[mask]
            vergence_angle[idx, trial_idx, spec_idx, 0:n] = eyes.vergence_angle_deg[mask]

with open('wt_eyes.npz', 'wb') as fp:
    np.savez(fp, version_angle, vergence_angle)

vergence_trial_avg = np.nanmean(vergence_angle, axis=1)
vergence_fish_avg = np.nanmean(vergence_angle, axis=0)
vergence_fish_trial_avg = np.nanmean(vergence_trial_avg, axis=0)

version_trial_avg = np.nanmean(version_angle, axis=1)
version_fish_avg = np.nanmean(version_angle, axis=0)
version_fish_trial_avg = np.nanmean(version_trial_avg, axis=0)

plt.plot(vergence_fish_trial_avg.reshape(-1,))
plt.plot(version_fish_trial_avg.reshape(-1,))
plt.hlines(0,0,np.prod(vergence_fish_trial_avg.shape))
plt.hlines(46,0,np.prod(vergence_fish_trial_avg.shape))
plt.show()

## plots
# average over trials
# average over fish
# average over trials then over fish