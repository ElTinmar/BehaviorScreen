from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from statsmodels.stats.multitest import multipletests

from BehaviorScreen.core import Stim, AGAROSE_WELL_DIMENSIONS
from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    BehaviorFiles,
    find_files, 
    load_data
)
from BehaviorScreen.plot import get_eye_traces, read_stim_specs, load_yaml_config
from BehaviorScreen.process import get_circle_hough, get_background_image
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

ROOT = Path('/media/martin/DATA_18TB/Screen')

cfg = load_yaml_config('BehaviorScreen/screen.yaml')
stim_specs = read_stim_specs(cfg, ignore_time_bins=True)
wt_data = process_eye_data(ROOT / 'WT/danieau/eyes.npz')
mutant_data = process_eye_data(ROOT / 'gr/danieau/eyes.npz')

plot_comparative_eyes(
    datasets=[wt_data, mutant_data],
    labels=['WT', 'gr'],
    stim_specs=list(stim_specs), 
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

def compute_t_student_and_d(group_a, group_b):
    m_a, m_b = np.nanmean(group_a, axis=0), np.nanmean(group_b, axis=0)
    v_a, v_b = np.nanvar(group_a, axis=0), np.nanvar(group_b, axis=0)
    na, nb = len(group_a), len(group_b)
    
    pooled_std = np.sqrt(((na - 1) * v_a + (nb - 1) * v_b) / (na + nb - 2))
    pooled_std[pooled_std == 0] = np.nan
    
    t_stat = (m_b - m_a) / (pooled_std * np.sqrt(1/na + 1/nb))
    cohen_d = (m_b - m_a) / pooled_std
    
    return t_stat, cohen_d

def compute_t_and_d(group_a, group_b):
    m_a, m_b = np.nanmean(group_a, axis=0), np.nanmean(group_b, axis=0)
    v_a, v_b = np.nanvar(group_a, axis=0, ddof=1), np.nanvar(group_b, axis=0, ddof=1)
    na, nb = len(group_a), len(group_b)

    pooled_var = ((na - 1) * v_a + (nb - 1) * v_b) / (na + nb - 2)
    welch_var = (v_a / na) + (v_b / nb)
    zero_var = welch_var <= 1e-15 
    
    # T-stat
    se_welch = np.sqrt(welch_var)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = (m_b - m_a) / se_welch
    t_stat[zero_var] = 0  
    
    # Cohen's d
    pooled_std = np.sqrt(pooled_var)
    with np.errstate(divide='ignore', invalid='ignore'):
        cohen_d = (m_b - m_a) / pooled_std
    cohen_d[zero_var] = 0
    
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

    # Two-tailed p-values (Phipson & Smyth correction)
    b = np.sum(np.abs(null_t_stats) >= np.abs(obs_t), axis=0)
    p_values = (b + 1) / (n_perm + 1)

    p_shape = p_values.shape
    reject, p_corrected, _, _ = multipletests(p_values.ravel(), alpha=alpha, method='fdr_bh')
    
    return obs_d, p_corrected.reshape(p_shape)

ROOT = Path('/home/martin/Desktop/DATA')
ROOT = Path('/media/martin/DATA_18TB/Screen')
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
        y_labels,
        x_labels,
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
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("bout category")
        
        # X-axis labels (only rotate and show for the bottom plot to save space)
        ax.set_xticks(range(len(x_labels)))
        if i == 2:
            ax.set_xticklabels(x_labels, rotation=90, ha='center')
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    return fig, axes

# I want a method of quantifying significance/effect size where 
# - lak danieau vs WT danieau show lots of differences
# - WT danieau vs WT ronidazole does not
alpha = 0.05
#value_threshold = 0.05

capture_strikes = ['LCS_L','LCS_R','SCS_L','SCS_R']
keep = [i for i,r in enumerate(row_names) if r not in capture_strikes]
bouts_cat = [r for r in row_names if r not in capture_strikes]

for ref, comp_list in comparisons.items():

    ref_trial_avg, bin_names = load_bouts(ref)
    ref_trial_avg = ref_trial_avg[...,keep]
    ref_fish_trial_avg = np.nanmean(ref_trial_avg, axis=0).T
    
    for p in comp_list:
    
        exp_trial_avg, _ = load_bouts(p)
        exp_trial_avg = exp_trial_avg[...,keep]
        exp_fish_trial_avg = np.nanmean(exp_trial_avg, axis=0).T

        #cohen_d_boot = bootstrap_cohen_d(ref_trial_avg, exp_trial_avg)
        #ci_low, cohen_d_median,  ci_high = np.percentile(cohen_d_boot, [2.5, 50, 97.5], axis=0)
        #ci_mask = (ci_low.T > 0) | (ci_high.T < 0) 
        #data = cohen_d_median.T

        d_map, p_map = permutation_analysis(ref_trial_avg, exp_trial_avg)
        data = d_map.T
        sig_mask = p_map.T < alpha
        #data[~sig_mask] = 0

        #low_bout_freq = np.stack((ref_fish_trial_avg, exp_fish_trial_avg)).max(axis=0) < value_threshold 
        #mask_out = low_bout_freq | (~sig_mask)

        #high_bout_freq = np.stack((ref_fish_trial_avg, exp_fish_trial_avg)).max(axis=0) >= value_threshold 
        #scatter_mask = high_bout_freq & sig_mask

        title = f"{p.relative_to(ROOT).parent} - {ref.relative_to(ROOT).parent}".replace('/',':')
        plot_heatmap(ref_fish_trial_avg, exp_fish_trial_avg, data, sig_mask, title, bouts_cat, bin_names)
        plt.savefig(f"{title}_alpha_{alpha}.png")
        plt.close()


## playing with head embedding ===================================================================================


# Learning wall interaction: add "distance to wall" and "angle of incidence" to the features 
import numpy as np

def extract_features(behavior_data):

    df = behavior_data.full_tracking
    pix_per_mm = behavior_data.metadata['calibration']['pix_per_mm']
    background_image = get_background_image(behavior_data)
    circle = get_circle_hough(
        background_image, pix_per_mm, 2, AGAROSE_WELL_DIMENSIONS, 2.5, 0.3
    )

    head = df.Head[['x', 'y']].to_numpy()
    sb = df.Swim_Bladder[['x', 'y']].to_numpy()
    heading_v = head - sb
    radial_v = sb - circle.center

    dot_w = np.einsum('ij,ij->i', heading_v, radial_v)
    det_w = heading_v[:, 0] * radial_v[:, 1] - heading_v[:, 1] * radial_v[:, 0]
    angle_to_wall = np.arctan2(det_w, dot_w)
    dist_from_center = np.linalg.norm(radial_v, axis=1)
    distance_to_wall = (circle.radius - dist_from_center) / circle.radius

    tail_coords = np.stack([df[f'Tail_{i}'][['x', 'y']].to_numpy() for i in range(10)], axis=1)
    segments = np.diff(tail_coords, axis=1) # Shape: (N, 9, 2)
    h_expanded = -1*heading_v[:, np.newaxis, :] # Shape: (N, 1, 2)
    t_dot = h_expanded[..., 0] * segments[..., 0] + h_expanded[..., 1] * segments[..., 1]
    t_det = h_expanded[..., 0] * segments[..., 1] - h_expanded[..., 1] * segments[..., 0]
    tail_angles = np.arctan2(t_det, t_dot) # Shape: (N, 9)
    tail_velocity = np.diff(tail_angles, axis=0, prepend=tail_angles[:1, :])

    features = np.column_stack([
        tail_angles, 
        tail_velocity, 
        angle_to_wall, 
        distance_to_wall
    ])
    
    return features

def extract_targets(behavior_data: BehaviorData):

    df = behavior_data.full_tracking
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

def target_to_trajectory(targets, start_pos=(0, 0), start_theta=0):

    n_steps = len(targets)
    recon_x = np.zeros(n_steps + 1)
    recon_y = np.zeros(n_steps + 1)
    recon_theta = np.zeros(n_steps + 1)
    recon_x[0], recon_y[0] = start_pos
    recon_theta[0] = start_theta
    dx_local = targets[:, 0]
    dy_local = targets[:, 1]
    d_theta = targets[:, 2]
    
    for i in range(n_steps):
        recon_theta[i+1] = recon_theta[i] + d_theta[i]
        cos_t = np.cos(recon_theta[i])
        sin_t = np.sin(recon_theta[i])
        dx_global = dx_local[i] * cos_t - dy_local[i] * sin_t
        dy_global = dx_local[i] * sin_t + dy_local[i] * cos_t
        recon_x[i+1] = recon_x[i] + dx_global
        recon_y[i+1] = recon_y[i] + dy_global
        
    return recon_x, recon_y, recon_theta

def add_history(X, n_history):
    """
    X: (N, features) array
    n_history: number of previous frames to include (e.g., 5)
    Returns: (N - n_history, features * (n_history + 1))
    """
    # Create a list of shifted arrays
    # We want [X_t, X_{t-1}, X_{t-2}, ...]
    shifted_blocks = []
    for i in range(n_history + 1):
        # Shift the data and trim the ends to keep lengths equal
        # i=0 is current frame, i=1 is 1 frame ago...
        start_idx = n_history - i
        end_idx = X.shape[0] - i
        shifted_blocks.append(X[start_idx:end_idx, :])
    
    # Concatenate horizontally
    return np.hstack(shifted_blocks)

X=[]
y=[]
n_history = 10
for line in Path('/media/martin/DATA_18TB/Screen').iterdir():
    for condition in line.iterdir():
        if condition.is_dir():
            directories = Directories(
                root = condition,
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
            for behavior_file in find_files(directories):
                print(behavior_file.metadata)
                behavior_data = load_data(behavior_file)
                features = extract_features(behavior_data)
                targets = extract_targets(behavior_data) 
                X_history = add_history(features, n_history)
                X_history = X_history[:-1, :] 
                y_aligned = targets[n_history:]
            
                X.append(X_history) 
                y.append(y_aligned)
                            
# train
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

plt.figure(figsize=(10, 4))
plot_acf(features[:, 4] , lags=60)
plt.title("Autocorrelation of Tail Segment 4")
plt.xlabel("Lags (Frames)")
plt.show()

X = features[:-1, :] 
y = targets            

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Mean Squared Error: {np.mean((y_test - predictions)**2)}")

traj_test = target_to_trajectory(y_test)
traj_pred = target_to_trajectory(predictions)
plt.plot(traj_test[0], traj_test[1], 'k', alpha=0.5)
plt.plot(traj_pred[0], traj_pred[1], 'r', alpha=0.5)
plt.axis('equal')
plt.show()

n_history = 10
X_history = add_history(features, n_history)
X_history = X_history[:-1, :] 
y_aligned = targets[n_history:]

X_train, X_test, y_train, y_test = train_test_split(X_history, y_aligned, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
scores = r2_score(y_test, predictions, multioutput='raw_values')
print(f"Mean Squared Error: {np.mean((y_test - predictions)**2)}")
print(f"R2 : {r2_score(y_test, predictions):.3f}")
print(f"R2 Forward (dx): {scores[0]:.3f}")
print(f"R2 Lateral (dy): {scores[1]:.3f}")
print(f"R2 Turning (dTh): {scores[2]:.3f}")


### this is really fast
model_dx = HistGradientBoostingRegressor(max_iter=200)
model_dy = HistGradientBoostingRegressor(max_iter=200)
model_dtheta = HistGradientBoostingRegressor(max_iter=200)
model_dx.fit(X_train, y_train[:, 0])
model_dy.fit(X_train, y_train[:, 1])
model_dtheta.fit(X_train, y_train[:, 2])
dx_pred = model_dx.predict(X_test)
dy_pred = model_dy.predict(X_test)
dtheta_pred = model_dtheta.predict(X_test)
predictions = np.column_stack([dx_pred, dy_pred, dtheta_pred])
scores = r2_score(y_test, predictions, multioutput='raw_values')
print(f"Mean Squared Error: {np.mean((y_test - predictions)**2)}")
print(f"R2 : {r2_score(y_test, predictions):.3f}")
print(f"R2 Forward (dx): {scores[0]:.3f}")
print(f"R2 Lateral (dy): {scores[1]:.3f}")
print(f"R2 Turning (dTh): {scores[2]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
targets_names = ['dx_local', 'dy_local', 'diff_theta']
for i in range(3):
    axes[i].scatter(y_test[:, i], predictions[:, i], alpha=0.1, s=1)
    axes[i].plot([y_test[:, i].min(), y_test[:, i].max()], 
                 [y_test[:, i].min(), y_test[:, i].max()], 'r--')
    axes[i].set_title(f'Actual vs Predicted: {targets_names[i]}')
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')
plt.tight_layout()
plt.show()

residuals = y_test - predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Forward (dx)', 'Lateral (dy)', 'Turning (dTheta)']
for i in range(3):
    axes[i].scatter(predictions[:, i], residuals[:, i], alpha=0.1, s=1)
    axes[i].axhline(0, color='red', linestyle='--')
    axes[i].set_title(f'Residuals for {titles[i]}')
    axes[i].set_xlabel('Predicted Value')
    axes[i].set_ylabel('Error (Actual - Pred)')
plt.tight_layout()
plt.show()

plt.scatter(X_test[:, -1], residuals[:, 0], alpha=0.1) # Distance to wall vs dx error
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Distance to Wall (normalized)')
plt.ylabel('Prediction Error (dx)')
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
titles = ['Forward Velocity (dx)', 'Lateral Velocity (dy)', 'Angular Velocity (dTheta)']
colors = ['black', 'red']
for i in range(3):
    axes[i].plot(y_test[:, i], color=colors[0], label='Actual', alpha=0.7)
    axes[i].plot(predictions[:, i], color=colors[1], label='Predicted', alpha=0.8)
    axes[i].set_ylabel('Units/Frame')
    axes[i].set_title(titles[i])
    if i == 0:
        axes[i].legend(loc='upper right')
axes[-1].set_xlabel('Frames (Time)')
plt.tight_layout()
plt.show()


## TODO overlay on video
import cv2
model = {
    'dx':model_dx,
    'dy':model_dy,
    'dtheta':model_dtheta
}

def overlay_video(behavior_data, model, output_path):

    fps = behavior_data.video.get_fps()
    height = behavior_data.video.get_height()
    width = behavior_data.video.get_width()
    num_frames = behavior_data.video.get_number_of_frame()

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    features = extract_features(behavior_data)
    X_history = add_history(features, n_history)
    X_history = X_history[:-1, :] 
    dx_pred = model['dx'].predict(X_history)
    dy_pred = model['dy'].predict(X_history)
    dtheta_pred = model['dtheta'].predict(X_history)
    predictions = np.column_stack([dx_pred, dy_pred, dtheta_pred])

    h_x, h_y = behavior_data.full_tracking[('Head', 'x')].values, behavior_data.full_tracking[('Head', 'y')].values
    s_x, s_y = behavior_data.full_tracking[('Swim_Bladder', 'x')].values, behavior_data.full_tracking[('Swim_Bladder', 'y')].values
    theta = np.arctan2(h_y - s_y, h_x - s_x)
    rx, ry, rtheta = target_to_trajectory(predictions, (s_x[0], s_y[0]), theta[0])

    for frame_idx in range(num_frames):
        print(frame_idx)
        if frame_idx % (120*5) == 0:
            rx = rx - rx[frame_idx] + s_x[frame_idx]
            ry = ry - ry[frame_idx] + s_y[frame_idx] 
            rtheta = rtheta - rtheta[frame_idx] + theta[frame_idx] 
            
        ret, frame  =  behavior_data.video.next_frame()  

        # prediction
        x = int(rx[frame_idx])
        y = int(ry[frame_idx])
        arrow_len = 20
        end = (
            int(x + arrow_len * np.cos(rtheta[frame_idx])),
            int(y + arrow_len * np.sin(rtheta[frame_idx]))
        )
        cv2.arrowedLine(frame, (x, y), end, (255, 0, 0), 2)

        # tracking
        x = int(s_x[frame_idx])
        y = int(s_y[frame_idx])
        arrow_len = 20
        end = (
            int(x + arrow_len * np.cos(theta[frame_idx])),
            int(y + arrow_len * np.sin(theta[frame_idx]))
        )
        cv2.arrowedLine(frame, (x, y), end, (0, 0, 255), 2)
        out.write(frame)

        if frame_idx == 120*40:
            break

    out.release()

#########

# find /media/martin/DATA_18TB/Screen -name bouts.csv  | xargs python -m BehaviorScreen.merge_csv -o all_bouts.csv
df = pd.read_csv('/home/martin/Code/BehaviorScreen/all_bouts.csv')

df.bout_duration[df.bout_duration < 1].hist(bins=120)
plt.show()

df.interbout_duration[df.interbout_duration < 4].hist(bins=120)
plt.show()

df.peak_yaw_speed.hist(bins=200)
plt.show()

df.peak_axial_speed[(df.peak_axial_speed > -50) & (df.peak_axial_speed < 150)].hist(bins=200)
plt.show()

# weird outliers + O-Bend sitch
df.centroid_mismatch_max[df.centroid_mismatch_max < 50].hist(bins=100)
plt.show()

df.centroid_mismatch_avg[df.centroid_mismatch_avg < 50].hist(bins=100)
plt.show()

df.heading_mismatch_avg[df.heading_mismatch_avg < 20].hist(bins=100)
plt.show()

df.heading_mismatch_max.hist(bins=100)
plt.show()

df.proba.hist(bins=100)
plt.show()

df[
    (df.stim == Stim.PREY_CAPTURE) & 
    (df.category == bouts_category_name_short.index('JT')) & 
    (df.file.str.contains('WT', case=False)) 
].stim_phase.hist(bins=50, density=True, alpha=0.75)
df[
    (df.stim == Stim.PREY_CAPTURE) & 
    (df.category == bouts_category_name_short.index('RT')) & 
    (df.file.str.contains('WT', case=False)) 
].stim_phase.hist(bins=50, density=True, alpha=0.75)
plt.show()

df[
    (df.stim == Stim.PREY_CAPTURE) & 
    (df.category == bouts_category_name_short.index('JT')) & 
    (df.file.str.contains('WT', case=False)) 
].stim_phase.hist(bins=50, density=True, alpha=0.75)
df[
    (df.stim == Stim.PREY_CAPTURE) & 
    (df.category == bouts_category_name_short.index('JT')) & 
    (df.file.str.contains('mecp2', case=False))
].stim_phase.hist(bins=50, density=True, alpha=0.75)
plt.show()

df[
    (df.stim == Stim.PREY_CAPTURE) & 
    (df.category == bouts_category_name_short.index('JT')) & 
    (df.file.str.contains('WT', case=False)) 
].stim_phase.hist(bins=50, density=True, alpha=0.75)
df[
    (df.stim == Stim.PREY_CAPTURE) & 
    (df.category == bouts_category_name_short.index('JT')) & 
    (df.file.str.contains('mafaa-switch', case=False))
].stim_phase.hist(bins=50, density=True, alpha=0.75)
plt.show()