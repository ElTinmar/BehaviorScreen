from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    BehaviorFiles,
    find_files, 
    load_data
)
from pathlib import Path
from typing import List
import pickle
from BehaviorScreen.megabouts import MegaboutResults
from BehaviorScreen.process import get_trials
from BehaviorScreen.core import Stim, GROUPING_PARAMETER

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

# --------------------------------------------------------------------------------
# TODO add tracking eyes to BehaviorFiles and BehaviorData + regexp in load

timestamps = behavior_data.tracking.timestamp.to_numpy()
stim_trials = get_trials(behavior_data)

L, R = get_eye_traces(behavior_data.eyes_tracking, likelihood_threshold=0.9)
L_s = savgol_filter(L, window_length=41, polyorder=2)
R_s = savgol_filter(R, window_length=41, polyorder=2)
version_angle = (L_s + R_s)/2
vergence_angle = R_s - L_s

for stim_select, stim_data in stim_trials.groupby('stim_select'):

    stim = Stim(stim_select)
    if not stim in GROUPING_PARAMETER:
        continue

    for condition, condition_data in stim_data.groupby(GROUPING_PARAMETER[stim]):
        for trial_idx, (trial, row) in enumerate(condition_data.iterrows()):
            mask = (timestamps > row.start_timestamp) & (timestamps < row.stop_timestamp) 
            version_angle[mask]
            vergence_angle[mask]

# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BehaviorScreen.process import compute_angle_between_vectors
from scipy.signal import savgol_filter
from pathlib import Path

#filename = '/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/eye_models/current/01_07dpf_WT_Thu_11_Dec_2025_13h15min29sec_fish_3_eyes.csv'

def get_eye_traces(
        data: pd.DataFrame, 
        likelihood_threshold: float = 0.9,
        divergence_threshold_deg: float = -10,
        convergence_threshold_deg: float = 60
    ) -> tuple[np.ndarray, np.ndarray]:

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

    L[(left_front_likelihood < likelihood_threshold) | (left_back_likelihood < likelihood_threshold)] = np.nan
    R[(right_front_likelihood < likelihood_threshold) | (right_back_likelihood < likelihood_threshold)] = np.nan
    L[(-L<divergence_threshold_deg) & (-L>convergence_threshold_deg)] = np.nan
    R[(R<divergence_threshold_deg) & (R>convergence_threshold_deg)] = np.nan

    L = pd.Series(L).interpolate(limit_direction="both").to_numpy()
    R = pd.Series(R).interpolate(limit_direction="both").to_numpy()

    return L, R

fs = 120
BASE_DIR = '/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/eye_models/current'
filenames = [f for f in Path(BASE_DIR).glob('eyes_*.csv')]

pooled_vergence = np.full((len(filenames), 500_000), np.nan)
pooled_version = np.full((len(filenames), 500_000), np.nan)

for idx, filename in enumerate(filenames):

    df = pd.read_csv(filename, header=[0,1,2])
    n = len(df)
    t = np.arange(n)/fs
    L, R = get_eye_traces(df.heatmap_tracker, likelihood_threshold=0.9)

    L_s = savgol_filter(L, window_length=41, polyorder=2)
    R_s = savgol_filter(R, window_length=41, polyorder=2)

    version_angle = (L_s + R_s)/2
    vergence_angle = R_s - L_s
    pooled_vergence[idx, 0:n] = vergence_angle
    pooled_version[idx, 0:n] = version_angle

# TODO average first over trials then over fish

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


