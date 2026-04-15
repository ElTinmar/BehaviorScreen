from pathlib import Path
from enum import IntEnum
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, sem, gaussian_kde
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import kruskal

from BehaviorScreen.load import Directories, find_files, load_data, BehaviorData
from BehaviorScreen.process import get_trials, get_well_coords_mm, timestamp_to_frame
from BehaviorScreen.core import Stim, BoutSign
from BehaviorScreen.plot import load_yaml_config, read_stim_specs
from megabouts.utils import bouts_category_name_short

COLOR_MECP2 = '#D95319'
COLOR_WT = '#0072BD'  
COLOR_WT_TLN = '#2E8B57'

plt.rcParams.update({
    'font.size': 12,          # Global default
    'axes.titlesize': 18,     # Title
    'axes.labelsize': 16,     # X and Y labels
    'xtick.labelsize': 14,    # X tick labels
    'ytick.labelsize': 14,    # Y tick labels
    'legend.fontsize': 12,    # Legend
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'], 
    'axes.linewidth': 1.5
})

ROOT = Path('/home/martin/Desktop/DATA')
ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/Screen')
ROOT = Path('/media/martin/DATA_18TB/Screen')

# N=48, N=40
#groups = ['mecp2/danieau/bouts.csv', 'AB/danieau/bouts.csv']
#groups = ['mecp2/danieau/bouts.csv', 'WT/danieau/bouts.csv']
#groups_name = ['mecp2-mutant', 'wild type']
#groups_color = {'mecp2-mutant': COLOR_MECP2, 'wild type': COLOR_WT}

groups = ['mecp2/danieau/bouts.csv', 'AB/danieau/bouts.csv', 'WT/danieau/bouts.csv']
groups_name = ['mecp2-mutant', 'wild type (AB)', 'wild type (TLN)']
groups_color = {'mecp2-mutant': COLOR_MECP2, 'wild type (AB)': COLOR_WT, 'wild type (TLN)': COLOR_WT_TLN}

##########

def epoch_masks(df):
    masks = [
        (df.stim == Stim.DARK) & (df.trial_num >= 10) & (df.trial_num < 20),
        (df.stim == Stim.BRIGHT) & (df.trial_num >= 5) & (df.trial_num < 15) & (df.time_start > 1049),
    ]
    # use last trials
    #masks = [
    #    (df.stim == Stim.DARK) & (df.trial_num >= 15) & (df.trial_num < 20),
    #    (df.stim == Stim.BRIGHT) & (df.trial_num >= 10) & (df.trial_num < 15) & (df.time_start > 1049),
    #]
    mask_names = ['spont_dark', 'spont_bright']
    return masks, mask_names

all_data = [] 
for g_idx, g in enumerate(groups):
    df = pd.read_csv(ROOT/g)
    df['group'] = g  
    all_data.append(df)
combined_df = pd.concat(all_data)
combined_df['speed'] = combined_df['distance']/combined_df['bout_duration']       

filtered_df = combined_df[
    (combined_df.bout_duration < 0.75) &  
    (combined_df.bout_duration > 0.05) &  
    (combined_df.interbout_duration < 4) &
    (combined_df.interbout_duration > 0.05) &
    (combined_df.distance < 10) & 
    (combined_df.speed < 30) &
    (combined_df.distance_center < 10) 
]
e_masks, e_mask_names = epoch_masks(filtered_df)

def plot_mean_sem_kde(
        df, 
        value_col, 
        x_range, 
        xlabel, 
        ylabel,
        groups, 
        groups_name, 
        groups_color, 
        ax=None, 
        bw=0.2
    ) -> None:
    
    if ax is None: ax = plt.gca()
    
    for g, g_name, g_color in zip(groups, groups_name, groups_color):
        group_df = df[df['group'] == g]
        group_densities = []
        subjects = group_df['file'].unique()
        
        for subj in subjects:
            subj_data = group_df[group_df['file'] == subj][value_col].dropna().values
            
            if len(subj_data) < 5 or np.std(subj_data) == 0:
                continue

            kde = gaussian_kde(subj_data, bw_method=bw)
            group_densities.append(kde(x_range))
            
        density_array = np.array(group_densities)
        mean_kde = np.mean(density_array, axis=0)
        sem_kde = sem(density_array, axis=0)
        
        line, = ax.plot(x_range, mean_kde, label=g_name, color=g_color, lw=2)
        ax.fill_between(x_range, 
                        mean_kde - sem_kde, 
                        mean_kde + sem_kde, 
                        color=line.get_color(), 
                        alpha=0.2, 
                        edgecolor='none')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

fig, axes = plt.subplots(nrows=len(e_masks), ncols=5, figsize=(5*4, 4*len(e_masks)))

for i, (mask, m_name) in enumerate(zip(e_masks, e_mask_names)):
    epoch_df = filtered_df[mask]

    model_bout_duration = smf.mixedlm("bout_duration ~ group", epoch_df, groups=epoch_df["file"])
    result_bout_duration = model_bout_duration.fit()
    print(result_bout_duration.summary())
    plot_mean_sem_kde(
        epoch_df, 
        value_col='bout_duration', 
        x_range=np.linspace(0, 0.75, 200),
        xlabel = 'bout duration (s)' if i==len(e_masks)-1 else '',
        ylabel = f'{m_name} density',
        groups=groups,
        groups_name=groups_name,
        groups_color=[groups_color[k] for k in groups_name],
        ax=axes[i,0]
    )

    model_interbout_duration = smf.mixedlm("interbout_duration ~ group", epoch_df, groups=epoch_df["file"])
    result_interbout_duration = model_interbout_duration.fit()
    print(result_interbout_duration.summary())
    plot_mean_sem_kde(
        epoch_df, 
        value_col='interbout_duration', 
        x_range=np.linspace(0, 4, 200),
        xlabel = 'interbout duration (s)' if i==len(e_masks)-1 else '',
        ylabel = '',
        groups=groups,
        groups_name=groups_name,
        groups_color=[groups_color[k] for k in groups_name],
        ax=axes[i,1]
    )

    model_bout_distance = smf.mixedlm("distance ~ group", epoch_df, groups=epoch_df["file"])
    result_bout_distance = model_bout_distance.fit()
    print(result_bout_distance.summary())
    plot_mean_sem_kde(
        epoch_df, 
        value_col='distance', 
        x_range=np.linspace(0, 10, 200),
        xlabel = 'distance (mm)' if i==len(e_masks)-1 else '',
        ylabel = '',
        groups=groups,
        groups_name=groups_name,
        groups_color=[groups_color[k] for k in groups_name],
        ax=axes[i,2]
    )

    model_bout_speed = smf.mixedlm("speed ~ group", epoch_df, groups=epoch_df["file"])
    result_bout_speed = model_bout_speed.fit()
    print(result_bout_speed.summary())
    plot_mean_sem_kde(
        epoch_df, 
        value_col='speed', 
        x_range=np.linspace(0, 30, 200),
        xlabel = 'speed (mm/s)' if i==len(e_masks)-1 else '',
        ylabel = '',
        groups=groups,
        groups_name=groups_name,
        groups_color=[groups_color[k] for k in groups_name],
        ax=axes[i,3]
    )

    model_distance_center = smf.mixedlm("distance_center ~ group", epoch_df, groups=epoch_df["file"])
    result_distance_center = model_distance_center.fit()
    print(result_distance_center.summary())
    plot_mean_sem_kde(
        epoch_df, 
        value_col='distance_center', 
        x_range=np.linspace(0, 10, 200),
        xlabel = 'distance to center (mm)' if i==len(e_masks)-1 else '',
        ylabel = '',
        groups=groups,
        groups_name=groups_name,
        groups_color=[groups_color[k] for k in groups_name],
        ax=axes[i,4]
    )

plt.tight_layout()
plt.tight_layout()
plt.savefig(f"distributions_spont.svg", format='svg', bbox_inches='tight')
plt.savefig(f"distributions_spont.png", format='png', dpi=100, bbox_inches='tight')
plt.show()

### TODO plot bout frequency during looming / total distance travelled (looming + recovery)

def get_bright_stim_frame_mask(behavior_data: BehaviorData):
    "find 10 consecutive bright stim"

    target_stim = 0.0
    target_color = [0.2, 0.2, 0.2, 1.0]
    
    mask = np.zeros((behavior_data.tracking.shape[0],), dtype=bool)
    
    for i in range(len(behavior_data.stimuli) - 10):

        is_sequence_start = all(
            behavior_data.stimuli[i+j].get('stim_select') == target_stim and 
            behavior_data.stimuli[i+j].get('foreground_color') == target_color
            for j in range(10)
        )
        
        if is_sequence_start:
            for j in range(10):
                start_ts = behavior_data.stimuli[i+j].get('timestamp')
                stop_ts = behavior_data.stimuli[i+j+1].get('timestamp')
                start_frame = timestamp_to_frame(behavior_data,start_ts)
                stop_frame = timestamp_to_frame(behavior_data,stop_ts)
                mask[start_frame:stop_frame] = True
            break 
            
    return mask

fig, axes = plt.subplots(1, len(groups), figsize=(6*len(groups), 6), sharey=True)
edges = np.linspace(-11, 11, 221) # 0.1 mm resolution

for idx, (g, gname, gcolor) in enumerate(zip(groups, groups_name, groups_color.values())):

    all_trajectories = []

    bout_csv = ROOT/g
    directories = Directories(
        bout_csv.parent,
        metadata='results',
        stimuli='results',
        tracking='results',
        full_tracking='lightning_pose',
        video='results',
        video_timestamp='results',
        results='results',
        plots='plots'
    )
    behavior_files = find_files(directories)
    n_ind = 0
    for behavior_file in behavior_files:
        print(behavior_file.metadata)
        behavior_data = load_data(behavior_file)
        cx,cy,_ = get_well_coords_mm(directories, behavior_file, behavior_data)

        bright_stim_mask = get_bright_stim_frame_mask(behavior_data)

        traj = behavior_data.tracking[['centroid_x', 'centroid_y']].to_numpy()
        traj_centered = traj/behavior_data.metadata['calibration']['pix_per_mm'] - np.array([cx, cy])
        traj_spont = traj_centered[bright_stim_mask]
        if traj_spont.size > 0:
            all_trajectories.append(traj_centered[bright_stim_mask])
            n_ind += 1
        else:
            print('bright not found, skipping')

    all_trajectories = np.vstack(all_trajectories)

    fps = behavior_data.metadata["camera"]["framerate_value"]
    normalization_weight = 1.0 / (fps * n_ind)
    weights = np.ones(len(all_trajectories)) * normalization_weight

    custom_cmap = LinearSegmentedColormap.from_list("black_to_color", ["black", gcolor])
    h = axes[idx].hist2d(
        all_trajectories[:, 0], 
        all_trajectories[:, 1], 
        bins=[edges,edges], 
        cmap=custom_cmap,
        weights=weights
    )
    h[3].set_clim([0, 0.1])
    axes[idx].set_aspect('equal')
    axes[idx].set_xlabel('X (mm)')
    if idx == 0:
        axes[idx].set_ylabel('Y (mm)')

    axes[idx].set_title(gname)
    
    cbar = fig.colorbar(h[3], ax=axes[idx], fraction=0.046, pad=0.04)
    if idx == (len(groups)-1):
        cbar.set_label('Mean Time per Fish (s)')

plt.tight_layout()
plt.savefig(f"thigmotaxis_2d_hist.svg", format='svg', bbox_inches='tight')
plt.savefig(f"thigmotaxis_2d_hist.png", format='png', dpi=100, bbox_inches='tight')
plt.show()

##### TODO distribution of eye vergence angles
config_yml = Path('BehaviorScreen/screen.yaml')
cfg = load_yaml_config(config_yml)
stim_specs = list(read_stim_specs(cfg, ignore_time_bins=True))
epochs = [2,3] 
trials = [0,1,2]
bw = 0.1
x_range = np.linspace(-10, 90, 101)

plt.figure(figsize=(6,6))
for idx, (g, g_name, g_color) in enumerate(zip(groups, groups_name, groups_color.values())):

    bout_csv = ROOT/g
    eyes_data = bout_csv.with_name('eyes.npz')
    data = np.load(eyes_data)
    vergence = data['vergence']

    group_densities = []
    for ind in range(vergence.shape[0]):
        kde_data = vergence[ind,trials][:, epochs].reshape(-1)
        kde_data = kde_data[~np.isnan(kde_data)]
        if len(kde_data) < 5:
            print(f'fish {ind} not enough data, skipping')
            continue
        kde = gaussian_kde(kde_data[~np.isnan(kde_data)], bw_method=bw)
        group_densities.append(kde(x_range))

    density_array = np.array(group_densities)
    mean_kde = np.mean(density_array, axis=0)
    sem_kde = sem(density_array, axis=0)
    
    line, = plt.plot(x_range, mean_kde, label=g_name, color=g_color, lw=2)
    plt.fill_between(x_range, 
                    mean_kde - sem_kde, 
                    mean_kde + sem_kde, 
                    color=line.get_color(), 
                    alpha=0.2, 
                    edgecolor='none')

plt.legend(frameon=False)
plt.xlabel('eye vergence (deg)')
plt.ylabel('density')
plt.savefig(f"eye_vergence.svg", format='svg', bbox_inches='tight')
plt.savefig(f"eye_vergence.png", format='png', dpi=100, bbox_inches='tight')
plt.show()

###
JTURN = bouts_category_name_short.index('JT')
prob_threshold = 0.5
trial_duration_s = 25
N_fish = 48

N_trials = 5
time_bins = [
    [0, 2.5],
    [2.5, 5],
    [5, 7.5],
    [7.5, 10],
    [10, 15],
    [15, 25]
]

class PreySide(IntEnum):
    LEFT = -20
    RIGHT = 20

ipsilateral = [(BoutSign.LEFT, PreySide.LEFT), (BoutSign.RIGHT, PreySide.RIGHT)]
contralateral = [(BoutSign.LEFT, PreySide.RIGHT), (BoutSign.RIGHT, PreySide.LEFT)]
laterality = [ipsilateral, contralateral]

JT_freq = np.full((len(groups), N_fish, len(laterality), N_trials, len(time_bins)), np.nan, dtype=np.float32)
JT_count = np.full((len(groups), N_fish, len(laterality), N_trials, len(time_bins)), np.nan, dtype=np.float32)
JT_proba = np.full((len(groups), N_fish, len(laterality), N_trials, len(time_bins)), np.nan, dtype=np.float32)

for g_idx, g in enumerate(groups):

    bout_file = ROOT/g 
    df = pd.read_csv(bout_file)
    file = df[df.stim == Stim.PREY_CAPTURE].file.unique()
    
    for fish_idx, fish in enumerate(file):
        for lat_idx, lat in enumerate(laterality): 
            for trial in range(N_trials):
                for bin_idx, (t_start, t_stop) in enumerate(time_bins):
                    for bout_sign, prey_side in lat: 
                        mask_JT = (
                            (df.file == fish) &
                            (df.stim == Stim.PREY_CAPTURE) &
                            (df.category == JTURN) & 
                            (df.proba > prob_threshold) &
                            (df.trial_time >= t_start) &
                            (df.trial_time < t_stop) &
                            (df.trial_num == trial) &
                            (df.sign == bout_sign) & 
                            (df.prey_arc_start_deg == prey_side)
                        )
                        mask_all_bouts = (
                            (df.file == fish) &
                            (df.stim == Stim.PREY_CAPTURE) &
                            (df.proba > prob_threshold) &
                            (df.trial_time >= t_start) &
                            (df.trial_time < t_stop) &
                            (df.trial_num == trial) &
                            (df.sign == bout_sign) & 
                            (df.prey_arc_start_deg == prey_side)
                        )
                        count_all_bouts =  mask_all_bouts.sum()
                        count_JT = mask_JT.sum()
                        JT_freq[g_idx, fish_idx, lat_idx, trial, bin_idx] = count_JT / (t_stop - t_start)
                        JT_count[g_idx, fish_idx, lat_idx, trial, bin_idx] = count_JT
                        JT_proba[g_idx, fish_idx, lat_idx, trial, bin_idx] = count_JT / count_all_bouts if count_all_bouts > 0 else 0


lat_names = ['Ipsilateral', 'Contralateral']
bin_labels = [f"{b[0]}-{b[1]}s" for b in time_bins]

def plot_heatmap(
        data, 
        label,
        vmax = 0.6
    ):

    fig, axes = plt.subplots(
        len(groups_name), 
        len(lat_names), 
        figsize=(len(lat_names)*5, len(groups_name)*5), 
        sharex=True, 
        sharey=True
    )

    for g_idx in range(len(groups_name)):
        for lat_idx in range(len(lat_names)):
            ax = axes[g_idx, lat_idx]
            data_avg = np.nanmean(data[g_idx, :,lat_idx, :, :], axis=0)
            
            sns.heatmap(data_avg, 
                        annot=True,       
                        fmt=".2f",        
                        cmap="magma",     
                        vmin=0,           
                        vmax=vmax,        
                        xticklabels=bin_labels,
                        ax=ax,
                        cbar=(lat_idx == len(lat_names) - 1),
                        cbar_kws={'label': label})
            
            ax.set_title(f"{groups_name[g_idx]} | {lat_names[lat_idx]}")
            
            if g_idx == len(groups_name) - 1:
                ax.set_xlabel("Time Bins")
            if lat_idx == 0:
                ax.set_ylabel("Trial Number")

    plt.savefig(f"{label}_heatmap.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_heatmap.png", format='png', dpi=100, bbox_inches='tight')
    plt.show()

def pval_to_star(p):
    if p <= 0.0001: return "****"
    if p <= 0.001:  return "***"
    if p <= 0.01:   return "**"
    if p <= 0.05:   return "*"
    return "n.s."

def add_pval_star(ax, x1, x2, y, p_val):
    text = pval_to_star(p_val)
    ax.plot([x1, x1, x2, x2], [y, y*1.02, y*1.02, y], lw=1.5, color='black', zorder=4)
    ax.text((x1 + x2) / 2, y, text, ha='center', va='bottom', fontsize=20)

def plot_barplot(
        data, 
        label,
        trials = [0,1,2],
        time_bins = [0]
    ):
    
    groups_color = {'Mecp2': COLOR_MECP2, 'AB': COLOR_WT, 'TLN': COLOR_WT_TLN}

    data_dict = {
        # We slice the first 3 dims [Group, :, Lat], 
        # then use idx for the last 2 [Trials, TimeBins]
        'Mecp2_Ipsi':   np.nanmean(data[0, :, 0][:, trials][..., time_bins], axis=(1, 2)),
        'Mecp2_Contra': np.nanmean(data[0, :, 1][:, trials][..., time_bins], axis=(1, 2)),
        'AB_Ipsi':      np.nanmean(data[1, :, 0][:, trials][..., time_bins], axis=(1, 2)),
        'AB_Contra':    np.nanmean(data[1, :, 1][:, trials][..., time_bins], axis=(1, 2)),
        'TLN_Ipsi':     np.nanmean(data[2, :, 0][:, trials][..., time_bins], axis=(1, 2)),
        'TLN_Contra':   np.nanmean(data[2, :, 1][:, trials][..., time_bins], axis=(1, 2))
    }

    def filter_non_responders(arr):
        # This keeps only fish that have at least one non-zero/non-nan value
        #return arr[~np.isnan(arr) & (arr != 0)]
        return arr[~np.isnan(arr)]

    data_dict = {k: filter_non_responders(v) for k, v in data_dict.items()}

    # 1. Omnibus Test: Are the three groups different at all?
    # We do this for Ipsilateral and Contralateral separately
    stat_k_ipsi, p_k_ipsi = kruskal(
        data_dict['Mecp2_Ipsi'],
        data_dict['AB_Ipsi'],
        data_dict['TLN_Ipsi']
    )
    
    stat_k_contra, p_k_contra = kruskal(
        data_dict['Mecp2_Contra'],
        data_dict['AB_Contra'],
        data_dict['TLN_Contra']
    )

    print(f"Kruskal-Wallis (Ipsi): p={p_k_ipsi:.4f}")
    print(f"Kruskal-Wallis (Contra): p={p_k_contra:.4f}")

    # 2. Stats: Update to compare Mecp2 vs AB and Mecp2 vs TLN
    def get_p(a, b):
        a_clean = a[~np.isnan(a)]
        b_clean = b[~np.isnan(b)]
        if len(a_clean) == 0 or len(b_clean) == 0: return 1.0
        return mannwhitneyu(a_clean, b_clean, alternative='less').pvalue

    p_ipsi_m_ab = get_p(data_dict['Mecp2_Ipsi'], data_dict['AB_Ipsi'])
    p_ipsi_m_tln = get_p(data_dict['Mecp2_Ipsi'], data_dict['TLN_Ipsi'])

    # Bonferroni correction for the 4 new comparisons
    pvals = [p_ipsi_m_ab, p_ipsi_m_tln]
    _, corrected_p, _, _ = multipletests(pvals, alpha=0.05, method='holm')
    names = ['Ipsi: M vs AB', 'Ipsi: M vs TLN']
    for name, raw, corr in zip(names, pvals, corrected_p):
        print(f"{name} -> Raw: {raw:.4f}, Corrected: {corr:.4f}")

    # 3. Update DataFrame construction
    groups = []
    lateralities = []
    values = []

    for key in data_dict.keys():
        val_array = data_dict[key]
        group_name = key.split('_')[0] # 'Mecp2', 'AB', or 'TLN'
        lat_name = 'ipsilateral' if 'Ipsi' in key else 'contralateral'
        
        values.extend(val_array)
        groups.extend([group_name] * len(val_array))
        lateralities.extend([lat_name] * len(val_array))

    df_plot = pd.DataFrame({
        'value': values,
        'group': groups,
        'laterality': lateralities
    }).dropna(subset=['value'])

    plt.figure(figsize=(8, 6)) 

    ax = sns.barplot(
        data=df_plot,
        x='laterality',
        y='value',
        hue='group',
        hue_order=['Mecp2', 'AB', 'TLN'], # Explicit order
        palette=groups_color, # Ensure this dict has 'Mecp2', 'AB', and 'TLN' keys
        errorbar='se',    
        capsize=0.05,      
        edgecolor='.2', 
        linewidth=1.5,
        gap=0.1
    )

    sns.stripplot(
        data=df_plot,
        x='laterality',
        y='value',
        hue='group',
        hue_order=['Mecp2', 'AB', 'TLN'],
        palette=groups_color, 
        jitter=0.15,
        dodge=True,
        alpha=0.4,
        edgecolor='white', 
        linewidth=0.5,
        size=4
    )

    x_m, x_ab, x_tln = -0.26, 0, 0.26
    y_max = df_plot['value'].max() * 1.1
    h_inc = y_max * 0.1 # Height increment for stacking stars

    add_pval_star(ax, x_m, x_ab,  y_max, corrected_p[0]) # Mecp2 vs AB (Ipsi)
    add_pval_star(ax, x_m, x_tln, y_max + h_inc, corrected_p[1]) # Mecp2 vs TLN (Ipsi)

    plt.ylabel(f"J-turn {label}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{label}_barplot.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_barplot.png", format='png', dpi=100, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.kdeplot(data_dict['Mecp2_Ipsi'], color=groups_color['Mecp2'], label='Mecp2') 
    sns.kdeplot(data_dict['AB_Ipsi'], color=groups_color['AB'], label='AB') 
    sns.kdeplot(data_dict['TLN_Ipsi'], color=groups_color['TLN'], label='TLN') 
    plt.title("Ipsilateral")
    plt.legend()
    plt.savefig(f"{label}_kde.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_kde.png", format='png', dpi=100, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.ecdfplot(data_dict['Mecp2_Ipsi'], color=groups_color['Mecp2'], label='Mecp2') 
    sns.ecdfplot(data_dict['AB_Ipsi'], color=groups_color['AB'], label='AB') 
    sns.ecdfplot(data_dict['TLN_Ipsi'], color=groups_color['TLN'], label='TLN') 
    plt.title("Ipsilateral")
    plt.legend()
    plt.savefig(f"{label}_ecdf.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_ecdf.png", format='png', dpi=100, bbox_inches='tight')
    plt.show()

for data_type, data in [('Frequency (Hz)', JT_freq), ('Probability', JT_proba)]:

    plot_heatmap(
        data,
        data_type,
        vmax = 0.6
    )

    plot_barplot(
        data,
        data_type,
        trials=[0,1],
        time_bins=[0,1,2]
    )


