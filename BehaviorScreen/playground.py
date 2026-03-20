from pathlib import Path
from typing import List, Tuple
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from statsmodels.stats.multitest import multipletests

from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    BehaviorFiles,
    find_files, 
    load_data
)
from BehaviorScreen.plot import get_eye_traces, read_stim_specs
from megabouts.utils import bouts_category_name_short

ROOT = Path('/media/martin/DATA_18TB/Screen/WT/danieau')
ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/Screen/WT/danieau')
ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/Screen/mecp2/danieau')
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


ROOT = Path('/media/martin/DATA_18TB/Screen/cort/vehicle')
behavior_file = [f for f in files if '00_07dpf_cort-veh_Wed_11_Feb_2026_18h59min38sec_fish_1' in str(f.metadata)][0]
behavior_data: BehaviorData = load_data(behavior_file)


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
        eyes = get_eye_traces(behavior_data, likelihood_threshold=0.9)
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

def process_eye_data(file_path):
    
    with np.load(file_path, allow_pickle=True) as data:
        version_angle = data['version']
        vergence_angle = data['vergence']

    vergence_per_fish = np.nanmean(vergence_angle, axis=1)
    version_per_fish = np.nanmean(version_angle, axis=1)

    return {
        'vergence_mean': np.nanmean(vergence_per_fish, axis=0).flatten(),
        'vergence_std': np.nanstd(vergence_per_fish, axis=0).flatten(),
        'version_mean': np.nanmean(version_per_fish, axis=0).flatten(),
        'version_std': np.nanstd(version_per_fish, axis=0).flatten()
    }

def plot_comparative_eyes(datasets, labels, stim_specs, fs, save_path='comparison.png'):

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(24, 8), 
                             sharex=True, gridspec_kw={'height_ratios': [1, 1, 0.5]},
                             layout='constrained')
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (data, label) in enumerate(zip(datasets, labels)):
        color = colors[i % len(colors)]
        x_axis = np.arange(len(data['vergence_mean']))
        
        axes[0].plot(x_axis, data['vergence_mean'], label=label, color=color, lw=2)
        axes[0].fill_between(x_axis, 
                             data['vergence_mean'] - data['vergence_std'], 
                             data['vergence_mean'] + data['vergence_std'], 
                             color=color, alpha=0.2, edgecolor='none')
        
        axes[1].plot(x_axis, data['version_mean'], label=label, color=color, lw=2)
        axes[1].fill_between(x_axis, 
                             data['version_mean'] - data['version_std'], 
                             data['version_mean'] + data['version_std'], 
                             color=color, alpha=0.2, edgecolor='none')

    axes[0].set_ylabel('<vergence [deg]>')
    axes[0].set_ylim((15, 65))
    axes[0].legend(loc='upper right', frameon=False)

    axes[1].set_ylabel('<version [deg]>')
    axes[1].axhline(0, linestyle='--', color='gray', alpha=0.5)
    axes[1].set_ylim((-15, 15))

    axes[2].set_axis_off()
    N_samples = len(datasets[0]['vergence_mean']) // len(stim_specs)
    for idx, stim in enumerate(stim_specs):
        text_label = textwrap.fill(f"{stim.name}: {stim.parameters}", width=20)
        x_pos = idx * N_samples + N_samples // 2
        axes[2].text(x_pos, 1.0, text_label, ha='right', va='top', rotation=45, fontsize=9)

    # Scale Bar
    scale_duration_sec = 10 
    scale_width_samples = scale_duration_sec * fs 
    scalebar = AnchoredSizeBar(axes[1].transData, scale_width_samples, 
                               f'{scale_duration_sec} s', 'lower right', 
                               pad=0.5, color='black', frameon=False, size_vertical=0.2)
    axes[1].add_artist(scalebar)

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

stim_specs = read_stim_specs('BehaviorScreen/mecp2.yaml', ignore_time_bins=True)
wt_data = process_eye_data('wt_eyes.npz')
mutant_data = process_eye_data('mecp2_eyes.npz')

plot_comparative_eyes(
    datasets=[wt_data, mutant_data],
    labels=['WT', 'mecp2-null'],
    stim_specs=stim_specs, 
    fs=fs                   
)

## Bootstraping bout freq

sides = ['L', 'R']
row_names = [f"{cat}_{str(side)}" for cat in bouts_category_name_short for side in sides]

def bootstrap_cohen_d(a, b, n_boot=2000, rng=None):
    rng = np.random.default_rng(rng)
    
    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    boot_samples_a = a[idx_a]
    boot_samples_b = b[idx_b]
    
    means_a = np.nanmean(boot_samples_a, axis=1)
    means_b = np.nanmean(boot_samples_b, axis=1)
    var_a = np.nanvar(boot_samples_a, axis=1)
    var_b = np.nanvar(boot_samples_b, axis=1)
    
    na, nb = len(a), len(b)
    pooled_stds = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    
    return (means_b - means_a) / pooled_stds

def compute_t_and_d(group_a, group_b):
    m_a, m_b = np.nanmean(group_a, axis=0), np.nanmean(group_b, axis=0)
    v_a, v_b = np.nanvar(group_a, axis=0), np.nanvar(group_b, axis=0)
    na, nb = len(group_a), len(group_b)
    
    pooled_std = np.sqrt(((na - 1) * v_a + (nb - 1) * v_b) / (na + nb - 2))
    pooled_std[pooled_std == 0] = np.nan
    
    t_stat = (m_b - m_a) / (pooled_std * np.sqrt(1/na + 1/nb))
    cohen_d = (m_b - m_a) / pooled_std
    
    return t_stat, cohen_d

def permutation_analysis(a, b, n_perm=5000, alpha=0.05, rng=None):

    rng = np.random.default_rng(rng)

    obs_t, obs_d = compute_t_and_d(a, b)
    
    combined = np.concatenate([a, b], axis=0)
    n_a = len(a)
    null_t_stats = []
    
    rng = np.random.default_rng()
    for _ in range(n_perm):
        shuffled = rng.permutation(combined, axis=0)
        perm_t, _ = compute_t_and_d(shuffled[:n_a], shuffled[n_a:])
        null_t_stats.append(perm_t)
        
    null_t_stats = np.array(null_t_stats)
    
    # Two-tailed p-values
    p_values = np.mean(np.abs(null_t_stats) >= np.abs(obs_t), axis=0)
    p_shape = p_values.shape
    reject, p_corrected, _, _ = multipletests(p_values.ravel(), alpha=alpha, method='fdr_bh')
    
    return obs_d, p_corrected.reshape(p_shape)

ROOT = Path('/home/martin/Desktop/DATA')
ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/Screen')

comparisons = {
    ROOT / 'WT/ronidazole/bouts.npz': [
        p for f in ROOT.iterdir() 
        if f.is_dir() 
        for p in [f / 'ronidazole/bouts.npz'] 
        if p.exists()
    ] + [ ROOT / 'WT/danieau/bouts.npz'],
    ROOT / 'WT/danieau/bouts.npz': [
        p for f in ROOT.iterdir() 
        if f.is_dir()
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
        ref,
        exp,
        effect_size, 
        mask,
        title,
        row_names,
        col_names,
        clim: Tuple[float, float] = (0, 0.35)
    ):
    # Create 3 vertically stacked subplots
    # Increased height (30) to accommodate three large heatmaps
    fig, axes = plt.subplots(3, 1, figsize=(24, 26), sharex=True)
    
    # 1. Plot Reference
    im0 = axes[0].imshow(ref, aspect='auto', cmap='inferno')
    im0.set_clim(*clim)
    axes[0].set_title(f"{title} - Reference")
    fig.colorbar(im0, ax=axes[0], label="Bout Frequency")
    asterisk_y, asterisk_x = np.where(mask)
    axes[0].scatter(asterisk_x, asterisk_y, s=8, color='lime', marker='o', zorder=2)
    
    # 2. Plot Experimental (Comp)
    im1 = axes[1].imshow(exp, aspect='auto', cmap='inferno')
    im1.set_clim(*clim)
    axes[1].set_title(f"{title} - Experimental")
    fig.colorbar(im1, ax=axes[1], label="Bout Frequency")
    
    # 3. Plot Effect Size
    im2 = axes[2].imshow(effect_size, aspect='auto', cmap='bwr')
    im2.set_clim(-3, 3)
    axes[2].set_title(f"{title} - Effect Size (Cohen's d)")
    fig.colorbar(im2, ax=axes[2], label="Effect size (Cohen's d)")
    
    # Formatting across all axes
    for i, ax in enumerate(axes):
        # Y-axis labels for every plot
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names)
        ax.set_ylabel("bout category")
        
        # X-axis labels (only rotate and show for the bottom plot to save space)
        ax.set_xticks(range(len(col_names)))
        if i == 2:
            ax.set_xticklabels(col_names, rotation=90, ha='center')
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    return fig, axes

# I want a method of quantifying significance/effect size where 
# - lak danieau vs WT danieau show lots of differences
# - WT danieau vs WT ronidazole does not
alpha = 0.01
value_threshold = 0.05

for ref, comp_list in comparisons.items():

    ref_trial_avg, bin_names = load_bouts(ref)
    ref_fish_trial_avg = np.nanmean(ref_trial_avg, axis=0).T
    
    for p in comp_list:
    
        exp_trial_avg, _ = load_bouts(p)
        exp_fish_trial_avg = np.nanmean(exp_trial_avg, axis=0).T

        #cohen_d_boot = bootstrap_cohen_d(ref_trial_avg, exp_trial_avg)
        #ci_low, cohen_d_median,  ci_high = np.percentile(cohen_d_boot, [2.5, 50, 97.5], axis=0)
        #ci_mask = (ci_low.T > 0) | (ci_high.T < 0) 
        #data = cohen_d_median.T

        d_map, p_map = permutation_analysis(ref_trial_avg, exp_trial_avg)
        data = d_map.T
        sig_mask = p_map.T < alpha

        low_bout_freq = np.stack((ref_fish_trial_avg, exp_fish_trial_avg)).max(axis=0) < value_threshold 
        mask_out = low_bout_freq | (~sig_mask)
        data[mask_out] = 0

        high_bout_freq = np.stack((ref_fish_trial_avg, exp_fish_trial_avg)).max(axis=0) >= value_threshold 
        scatter_mask = high_bout_freq & sig_mask

        title = f"{p.relative_to(ROOT).parent} - {ref.relative_to(ROOT).parent}".replace('/',':')
        plot_heatmap(ref_fish_trial_avg, exp_fish_trial_avg, data, scatter_mask, title, row_names, bin_names)
        plt.savefig(f"{title}_alpha_{alpha}.png")
        plt.close()


## playing with head embedding ===================================================================================


import numpy as np

# Learning wall interaction: add "distance to wall" and "angle of incidence" to the features 
from BehaviorScreen.process import get_well_coords_mm, get_background_image
well_coords_mm = get_well_coords_mm(directories, behavior_file, behavior_data)

def extract_features(df):
    # main axis pointing towards the tail 
    heading_x = (df[('Swim_Bladder', 'x')] - df[('Head', 'x')]).values
    heading_y = (df[('Swim_Bladder', 'y')] - df[('Head', 'y')]).values
    
    tail_angles = []
    for i in range(9):
        segment_x = (df[(f'Tail_{i+1}', 'x')] - df[(f'Tail_{i}', 'x')]).values
        segment_y = (df[(f'Tail_{i+1}', 'y')] - df[(f'Tail_{i}', 'y')]).values
        
        dot = heading_x * segment_x + heading_y * segment_y
        det = heading_x * segment_y - heading_y * segment_x
        angle = np.arctan2(det, dot)
        tail_angles.append(angle)

    tail_angles = np.stack(tail_angles, axis=1)
    tail_velocity = np.diff(tail_angles, axis=0, prepend=tail_angles[:1])
    features = np.hstack([tail_angles, tail_velocity])
    
    return features

def extract_targets(df):
    h_x, h_y = df[('Head', 'x')].values, df[('Head', 'y')].values
    s_x, s_y = df[('Swim_Bladder', 'x')].values, df[('Swim_Bladder', 'y')].values
    
    theta = np.arctan2(h_y - s_y, h_x - s_x)
    diff_x = np.diff(h_x)
    diff_y = np.diff(h_y)
    diff_theta = np.diff(np.unwrap(theta))
    
    t_start = theta[:-1]
    dx_local = diff_x * np.cos(t_start) + diff_y * np.sin(t_start)
    dy_local = -diff_x * np.sin(t_start) + diff_y * np.cos(t_start)
    
    return np.column_stack([dx_local, dy_local, diff_theta])

features = extract_features(behavior_data.full_tracking)
targets = extract_targets(behavior_data.full_tracking) 


