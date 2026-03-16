from pathlib import Path
from typing import List
import pickle
import pandas as pd
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
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
nsamp = 600_000
t = np.arange(nsamp) / fs
pooled_vergence = np.full((len(files), nsamp), np.nan)
pooled_version = np.full((len(files), nsamp), np.nan)
for idx, behavior_file in enumerate(files): 
        behavior_data: BehaviorData = load_data(behavior_file)
        print(idx, len(behavior_data.stimuli))
        if len(behavior_data.stimuli) < 180: 
            continue
        eyes = get_eye_traces(behavior_data.eyes_tracking, likelihood_threshold=0.9)
        n = len(eyes.version_angle_deg)
        pooled_vergence[idx, 0:n] = eyes.vergence_angle_deg
        pooled_version[idx, 0:n] = eyes.version_angle_deg

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 8), sharex=True)
axes[0].plot(t, np.nanmean(pooled_vergence, axis=0))
axes[0].set_title('WT - Vergence')
axes[0].set_ylabel(r'$\langle \text{vergence angle [deg]} \rangle$')
axes[1].plot(t, np.nanmean(pooled_version, axis=0))
axes[1].set_title('WT - Version')
axes[1].set_xlabel(r'$\text{time [s]}$')
axes[1].set_ylabel(r'$\langle \text{version angle [deg]} \rangle$')
plt.tight_layout()
plt.savefig('vergence_version_wt.png')
plt.show()

from BehaviorScreen.load import load_lightning_pose
files = [f for f in Path('/home/martin/Desktop/DATA/mecp2/danieau/lightning_pose').glob('*.csv') if 'temporal' not in str(f)]
fs = 120
pooled_vergence = np.full((len(files), 600_000), np.nan)
pooled_version = np.full((len(files), 600_000), np.nan)
for idx, behavior_file in enumerate(files): 
        print(behavior_file)
        tracking = load_lightning_pose(behavior_file)
        eyes = get_eye_traces(tracking, likelihood_threshold=0.9)
        n = len(eyes.version_angle_deg)
        pooled_vergence[idx, 0:n] = eyes.vergence_angle_deg
        pooled_version[idx, 0:n] = eyes.version_angle_deg

plt.figure(figsize=(26, 4))
plt.title('mecp2')
plt.plot(np.arange(600_000)/fs, np.nanmean(pooled_vergence, axis=0)) 
plt.xlabel('time')
plt.ylabel('<vergence angle [deg]>')
plt.savefig('mecp2_vergence.png')
plt.show()

plt.figure(figsize=(26,4))
plt.title('mecp2')
plt.plot(np.arange(600_000)/fs, np.nanmean(pooled_version, axis=0))    
plt.xlabel('time')
plt.ylabel('<version angle [deg]>')
plt.savefig('mecp2_version.png')
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

for fish_idx, behavior_file in enumerate(files):

    print(fish_idx)
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
            version_angle[fish_idx, trial_idx, spec_idx, 0:n] = eyes.version_angle_deg[mask]
            vergence_angle[fish_idx, trial_idx, spec_idx, 0:n] = eyes.vergence_angle_deg[mask]

with open('wt_eyes.npz', 'wb') as fp:
    np.savez(fp, version_angle, vergence_angle)

vergence_trial_avg = np.nanmean(vergence_angle, axis=1)
vergence_fish_avg = np.nanmean(vergence_angle, axis=0)
vergence_fish_trial_avg = np.nanmean(vergence_trial_avg, axis=0)

version_trial_avg = np.nanmean(version_angle, axis=1)
version_fish_avg = np.nanmean(version_angle, axis=0)
version_fish_trial_avg = np.nanmean(version_trial_avg, axis=0)


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(24, 6), 
                         sharex=True, 
                         gridspec_kw={'height_ratios': [1, 1, 0.5]},
                         layout='constrained')

total_samples = np.prod(vergence_fish_trial_avg.shape)

axes[0].plot(vergence_fish_trial_avg.reshape(-1,))
axes[0].set_title('WT - Vergence')
axes[0].set_ylabel('<vergence [deg]>')

axes[1].plot(version_fish_trial_avg.reshape(-1,))
axes[1].set_ylabel('<version [deg]>')
axes[1].axhline(0, linestyle='--', color='gray', alpha=0.7)

axes[2].set_axis_off() 
for idx, stim in enumerate(stim_specs):
    label = f"{stim.name} : {stim.parameters}" 
    label_wrapped = textwrap.fill(label, width=25)
    x_pos = idx * N_samples + N_samples // 2
    axes[2].text(x_pos, 1.0, label_wrapped, 
                 ha='right', va='top', 
                 rotation=45, 
                 fontsize=10,
                 clip_on=False)

scale_duration_sec = 10 
scale_width_samples = scale_duration_sec * fs 
scalebar = AnchoredSizeBar(axes[1].transData,
                           scale_width_samples, 
                           f'{scale_duration_sec} s', 
                           'lower right', 
                           pad=0.5,
                           color='black',
                           frameon=False,
                           size_vertical=0.2)

axes[1].add_artist(scalebar)

plt.savefig('organized_vergence_plot.png', bbox_inches='tight')
plt.show()

## plots
# average over trials
# average over fish
# average over trials then over fish

## Bootstraping bout freq

sides = ['L', 'R']
row_names = [f"{cat}_{str(side)}" for cat in bouts_category_name_short for side in sides]

def bootstrap_cohen_d(a, b, n_boot=2000, rng=None):
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

def plot_heatmap(
        data, 
        title,
        row_names,
        col_names
    ):
    
    fig = plt.figure(figsize=(26, 14))
    ax = fig.gca()
    im = ax.imshow(data, aspect='auto', cmap='bwr')
    im.set_clim(-3,3)
    fig.colorbar(im, ax=ax, label="effect size (cohen's d)")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(col_names, rotation=90, ha='center')
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(row_names)
    ax.set_xlabel("epoch")
    ax.set_ylabel("bout category")
    ax.set_title(title)
    fig.tight_layout()

for ref, comp_list in comparisons.items():

    ref_trial_avg, bin_names = load_bouts(ref)
    ref_fish_trial_avg = np.nanmean(ref_trial_avg, axis=0).T
    
    for p in comp_list:
    
        exp_trial_avg, _ = load_bouts(p)
        exp_fish_trial_avg = np.nanmean(exp_trial_avg, axis=0).T

        cohen_d_boot = bootstrap_cohen_d(ref_trial_avg, exp_trial_avg)
        ci_low, cohen_d_median,  ci_high = np.percentile(cohen_d_boot, [2.5, 50, 97.5], axis=0)
        data = cohen_d_median.T
        ci_mask = (ci_low.T > 0) | (ci_high.T < 0) # also cut off bout with very low freq)
        val_mask = np.stack((ref_fish_trial_avg, exp_fish_trial_avg)).max(axis=0) < 0.1 # should I bootstrap this too ?
        data[~ci_mask] = 0
        data[val_mask] = 0

        title = f"{p.relative_to(ROOT).parent} - {ref.relative_to(ROOT).parent}".replace('/',':')
        plot_heatmap(data, title, row_names, bin_names)
        plt.savefig(f"{title}.png")
        plt.close()

