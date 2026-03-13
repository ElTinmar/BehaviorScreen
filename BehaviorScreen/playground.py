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

from megabouts.utils import bouts_category_name_short
from BehaviorScreen.plot import plot_bout_heatmap

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

## Bootstraping bout freq

row_names = [f"{cat}_{str(side)}" for cat in bouts_category_name_short for side in sides]

def bootstrap_wt(wt, n_mut, n_boot=2000, rng=None):
    rng = np.random.default_rng(rng)

    n_wt = wt.shape[0]

    idx = rng.integers(0, n_wt, size=(n_boot, n_mut))
    boot = np.nanmean(wt[idx], axis=1)

    return boot

def bootstrap_difference(a, b, n_boot=2000, rng=None):
    rng = np.random.default_rng(rng)

    na = a.shape[0]
    nb = b.shape[0]

    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))

    boot_a = np.nanmean(a[idx_a],axis=1)
    boot_b = np.nanmean(b[idx_b],axis=1)

    return boot_b - boot_a

def bootstrap_effect_size(a, b, n_boot=2000, rng=None):
    rng = np.random.default_rng(rng)
    
    # Resample indices
    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    
    # Get bootstrap samples
    boot_samples_a = a[idx_a]
    boot_samples_b = b[idx_b]
    
    # Calculate means for each bootstrap iteration
    means_a = np.nanmean(boot_samples_a, axis=1)
    means_b = np.nanmean(boot_samples_b, axis=1)
    
    # Calculate pooled std for EACH bootstrap iteration
    var_a = np.nanvar(boot_samples_a, axis=1)
    var_b = np.nanvar(boot_samples_b, axis=1)
    
    na, nb = len(a), len(b)
    pooled_stds = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    
    # Cohen's d distribution
    return (means_b - means_a) / pooled_stds


ROOT = Path('/home/martin/Desktop/DATA')

comparisons = {
    ROOT / 'WT/ronidazole/bouts.npz': [
        p for f in ROOT.iterdir() 
        if f.is_dir() and 'WT' not in f.parts
        for p in [f / 'ronidazole/bouts.npz'] 
        if p.exists()
    ],
    ROOT / 'WT/danieau/bouts.npz': [
        p for f in ROOT.iterdir() 
        if f.is_dir() and 'WT' not in f.parts
        for p in [f / 'vehicle/bouts.npz', f / 'danieau/bouts.npz']
        if p.exists()
    ],
}


def load_bouts(file):

    with np.load(file, allow_pickle=True) as data:
        fish_names = data["labels_0"]
        trial_labels = data["labels_1"]
        bin_names = data["labels_2"]
        bout_categories = data["labels_3"]
        sides = data["labels_4"]
        bout_frequency = data["bout_frequency"]

        bout_frequency_interleaved = bout_frequency.reshape(*bout_frequency.shape[:-2], -1)
        trial_avg = np.nanmean(bout_frequency_interleaved, axis=1)

    return trial_avg, bin_names

for ref, comp_list in comparisons.items():

    ref_trial_avg, bin_names = load_bouts(ref)

    for p in comp_list:

        exp_trial_avg, _ = load_bouts(p)

        cohen_d_boot = bootstrap_effect_size(ref_trial_avg, exp_trial_avg)
        ci_low, cohen_d_median,  ci_high = np.percentile(cohen_d_boot, [2.5, 50, 97.5], axis=0)
        data = cohen_d_median.T
        sigmask = (ci_low.T > 0) | (ci_high.T < 0)
        data[~sigmask] = 0

        fig = plt.figure(figsize=(26, 14))
        ax = fig.gca()
        im = ax.imshow(data, aspect='auto', cmap='bwr')
        im.set_clim(-3,3)
        fig.colorbar(im, ax=ax, label="effect size (cohen's d)")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(bin_names, rotation=90, ha='center')
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(row_names)
        ax.set_xlabel("epoch")
        ax.set_ylabel("bout category")
        ax.set_title(f"{ref}-{p}")
        fig.tight_layout()
        plt.savefig(f"{ref}_{p.parents[1].stem}.png")


boot_diff = bootstrap_difference(ref_trial_avg, exp_trial_avg)
diff = np.nanmean(ref_trial_avg, axis=0) - np.nanmean(exp_trial_avg, axis=0)
ci_low, boot_med,  ci_high = np.percentile(boot_diff, [2.5, 50, 97.5], axis=0)

fig = plt.figure(figsize=(26, 14))
ax = fig.gca()
plot_bout_heatmap(fig, ax, boot_med.T, bin_names, row_names, 'bwr', (-0.3, 0.3))
asterisk_y, asterisk_x = np.where(ci_low.T >= 0)
ax.scatter(asterisk_x, asterisk_y, s=20, color='black', marker='o', zorder=2)
asterisk_y, asterisk_x = np.where(ci_high.T <= 0)
ax.scatter(asterisk_x, asterisk_y, s=20, color='black', marker='x', zorder=2)
fig.tight_layout()
plt.show()

#######

boot_wt = bootstrap_wt(wt_trial_avg, exp_trial_avg.shape[0], n_boot=10_000)
ci_low, ci_high = np.percentile(boot_wt, [2.5, 97.5], axis=0)
p_low  = np.mean(boot_wt <= np.nanmean(exp_trial_avg, axis=0), axis=0)
p_high = np.mean(boot_wt >= np.nanmean(exp_trial_avg, axis=0), axis=0)
p = 2 * np.minimum(p_low, p_high)
p = np.minimum(p, 1.0)

fig = plt.figure(figsize=(26, 14))
ax = fig.gca()
plot_bout_heatmap(fig, ax, p.T <=0.0001, bin_names, row_names, (-1, 1))
fig.tight_layout()
plt.show()


###
from statsmodels.stats.multitest import multipletests

# p_flat = p.ravel()
# reject, p_fdr, _, _ = multipletests(p_flat, alpha=0.05, method='fdr_bh')
# p_corrected = p_fdr.reshape(p.shape)
# sig_mask = reject.reshape(p.shape)
sig_mask = p <= 0.0000001

fig, ax = plt.subplots(figsize=(26, 14))
cmap = plt.get_cmap("bwr")
im = ax.imshow(diff.T, cmap=cmap, aspect='auto')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mutant - WT bout frequency")
asterisk_y, asterisk_x = np.where(sig_mask.T)
ax.scatter(asterisk_x, asterisk_y, s=20, color='black', marker='o', zorder=2)
ax.set_xlabel("Bout type")
ax.set_ylabel("Epoch")
ax.set_xticks(range(diff.shape[0]))
ax.set_xticklabels(bin_names, rotation=90, ha='center')
ax.set_yticks(range(diff.shape[1]))
ax.set_yticklabels(row_names)
plt.tight_layout()
plt.show()