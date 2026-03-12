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

N_fish = len(files)
N_trials = 100
N_epochs = 30
N_samples = 30 * 120

vergence_angle = np.full((N_fish, N_trials, N_epochs, N_samples), np.nan)
version_angle = np.full((N_fish, N_trials, N_epochs, N_samples), np.nan)

for idx, behavior_file in enumerate(files):

    behavior_data: BehaviorData = load_data(behavior_file)
    timestamps = behavior_data.tracking.timestamp.to_numpy()
    stim_trials = get_trials(behavior_data)
    eyes = get_eye_traces(behavior_data.eyes_tracking, likelihood_threshold=0.9)

    idx_epoch = 0 # you can do better probably?

    for stim_select, stim_data in stim_trials.groupby('stim_select'):

        stim = Stim(stim_select)
        if not stim in GROUPING_PARAMETER:
            continue

        for condition, condition_data in stim_data.groupby(GROUPING_PARAMETER[stim]):

            for trial_idx, (trial, row) in enumerate(condition_data.iterrows()):
                mask = (timestamps > row.start_timestamp) & (timestamps < row.stop_timestamp) 
                n = sum(mask)
                version_angle[idx, trial_idx, idx_epoch, 0:n] = eyes.version_angle_deg[mask]
                vergence_angle[idx, trial_idx, idx_epoch, 0:n] = eyes.vergence_angle_deg[mask]

            idx_epoch += 1

## plots
# average over trials
# average over fish
# average over trials then over fish