from pathlib import Path
from enum import IntEnum
import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, sem, gaussian_kde, permutation_test
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import statsmodels.formula.api as smf

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
plt.savefig(f"distributions_spont.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

### TODO plot bout frequency during looming / total distance travelled (looming + recovery)

def get_bright_stim_frame_mask(behavior_data: BehaviorData):
    "find 10 consecutive bright stim"

    target_stim = 1.0
    target_color = [0.2, 0.2, 0.0, 1.0]
    
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
    h[3].set_clim([0, 0.04])
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
plt.savefig(f"thigmotaxis_2d_hist.png", format='png', dpi=300, bbox_inches='tight')
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
plt.savefig(f"eye_vergence.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

###
groups = ['mecp2/danieau/bouts.csv', 'AB/danieau/bouts.csv', 'WT/danieau/bouts.csv']
groups_name = ['mecp2-mutant', 'wild type (AB)', 'wild type (TLN)']
groups_color = {'mecp2-mutant': COLOR_MECP2, 'wild type (AB)': COLOR_WT, 'wild type (TLN)': COLOR_WT_TLN}

ROOT = Path('/home/martin/Desktop/DATA')
ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/Screen')
ROOT = Path('/media/martin/DATA_18TB/Screen')

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

for g_idx, g in enumerate(groups):

    bout_file = ROOT/g 

    qc_file = bout_file.with_name("qc.csv")
    blacklisted = []
    if qc_file.exists():
        quality_control = pd.read_csv(qc_file)
        blacklisted.extend(quality_control.file.to_list())

    df = pd.read_csv(bout_file)
    file = df[df.stim == Stim.PREY_CAPTURE].file.unique()
    
    for fish_idx, fish in enumerate(file):

        if fish in blacklisted:
            print(f'{fish} did not pass qc')
            continue

        for lat_idx, lat in enumerate(laterality): 
            for trial in range(N_trials):
                for bin_idx, (t_start, t_stop) in enumerate(time_bins):
                    count_all_bouts = 0
                    count_JT = 0
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
                        count_JT += mask_JT.sum()
                    JT_freq[g_idx, fish_idx, lat_idx, trial, bin_idx] = count_JT / (len(lat) * (t_stop - t_start))


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
            ax.set_aspect('equal')
            
            if g_idx == len(groups_name) - 1:
                ax.set_xlabel("Time Bins")
            if lat_idx == 0:
                ax.set_ylabel("Trial Number")

    plt.savefig(f"{label}_heatmap.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_heatmap.png", format='png', dpi=300, bbox_inches='tight')
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
    def get_p_permutation(a, b):

        a_clean = a[~np.isnan(a)]
        b_clean = b[~np.isnan(b)]
        
        if len(a_clean) == 0 or len(b_clean) == 0: 
            return 1.0

        def statistic(x, y):
            return np.mean(x) - np.mean(y)

        # Perform the permutation test
        # 'less' tests the null hypothesis that mean(a) >= mean(b)
        res = permutation_test(
            (a_clean, b_clean), 
            statistic, 
            permutation_type='independent', 
            vectorized=False, 
            n_resamples=10000, 
            alternative='less'
        )
        
        return res.pvalue

    def get_p(a, b):
        a_clean = a[~np.isnan(a)]
        b_clean = b[~np.isnan(b)]
        if len(a_clean) == 0 or len(b_clean) == 0: return 1.0
        return mannwhitneyu(a_clean, b_clean, alternative='less').pvalue

    p_ipsi_m_ab = get_p_permutation(data_dict['Mecp2_Ipsi'], data_dict['AB_Ipsi'])
    p_ipsi_m_tln = get_p_permutation(data_dict['Mecp2_Ipsi'], data_dict['TLN_Ipsi'])

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
    plt.savefig(f"{label}_barplot.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.kdeplot(data_dict['Mecp2_Ipsi'], color=groups_color['Mecp2'], label='Mecp2') 
    sns.kdeplot(data_dict['AB_Ipsi'], color=groups_color['AB'], label='AB') 
    sns.kdeplot(data_dict['TLN_Ipsi'], color=groups_color['TLN'], label='TLN') 
    plt.title("Ipsilateral")
    plt.legend()
    plt.savefig(f"{label}_kde.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_kde.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.ecdfplot(data_dict['Mecp2_Ipsi'], color=groups_color['Mecp2'], label='Mecp2') 
    sns.ecdfplot(data_dict['AB_Ipsi'], color=groups_color['AB'], label='AB') 
    sns.ecdfplot(data_dict['TLN_Ipsi'], color=groups_color['TLN'], label='TLN') 
    plt.title("Ipsilateral")
    plt.legend()
    plt.savefig(f"{label}_ecdf.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_ecdf.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

for data_type, data in [('Frequency (Hz)', JT_freq)]:

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


######################### distributions

wt_ipsi_data = JT_freq[2, :, 0, :, :]
num_trials = wt_ipsi_data.shape[1]      
num_time_bins = wt_ipsi_data.shape[2] 
fig, axes = plt.subplots(num_trials, num_time_bins, 
                         figsize=(8, 8), 
                         sharex=True, 
                         sharey=True)

max_freq = int(np.nanmax(wt_ipsi_data))
bins = np.linspace(0, max_freq, 10)

for trial_idx in range(num_trials):
    for bin_idx in range(num_time_bins):
        ax = axes[trial_idx, bin_idx]
        
        counts = wt_ipsi_data[:, trial_idx, bin_idx]
        ax.hist(counts, bins=bins, color='lightgray', edgecolor='gray', alpha=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        if trial_idx == 0:
            ax.set_title(f"Bin {bin_idx}", fontsize=10)
        if bin_idx == 0:
            ax.set_ylabel(f"Trial {trial_idx}", fontsize=10)

plt.tight_layout()
plt.savefig(f"JT_histograms.svg", format='svg', bbox_inches='tight')
plt.show()

##########
groups = ['mecp2/bouts.csv', 'nacre/bouts.csv']
groups_name = ['mecp2-mutant', 'wild type']
groups_color = {'mecp2-mutant': COLOR_MECP2, 'wild type': COLOR_WT}

ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/PreyCapture_mecp2/uv_intensity')
ROOT = Path('/media/martin/DATA_18TB/PreyCapture_mecp2/uv_intensity')

JTURN = bouts_category_name_short.index('JT')
ROUTINE_TURN = bouts_category_name_short.index('RT')

prob_threshold = 0.5
trial_duration_s = 7 # 7 total duration of the trial
N_fish = 40

class PreySide(IntEnum):
    LEFT = -20
    RIGHT = 20

ipsilateral = [(BoutSign.LEFT, PreySide.LEFT), (BoutSign.RIGHT, PreySide.RIGHT)]
contralateral = [(BoutSign.LEFT, PreySide.RIGHT), (BoutSign.RIGHT, PreySide.LEFT)]
laterality = [ipsilateral, contralateral]
uv_intensities = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0]

JT_freq = np.full((len(groups), N_fish, len(laterality), len(uv_intensities)), np.nan, dtype=np.float32)
RT_freq = np.full((len(groups), N_fish, len(laterality), len(uv_intensities)), np.nan, dtype=np.float32)

for g_idx, g in enumerate(groups):

    bout_file = ROOT/g 

    qc_file = bout_file.with_name("qc.csv")
    blacklisted = []
    if qc_file.exists():
        quality_control = pd.read_csv(qc_file)
        blacklisted.extend(quality_control.file.to_list())

    df = pd.read_csv(bout_file)
    file = df[df.stim == Stim.PREY_CAPTURE].file.unique()
    
    for fish_idx, fish in enumerate(file):

        if fish in blacklisted:
            print(f'{fish} did not pass qc')
            continue

        for uv_idx, intensity in enumerate(uv_intensities):
            for lat_idx, lat in enumerate(laterality): 
                count_JT = 0
                count_RT = 0
                for bout_sign, prey_side in lat: 
                    mask_JT = (
                        (df.file == fish) &
                        (df.stim == Stim.PREY_CAPTURE) &
                        (df.category == JTURN) & 
                        (df.proba > prob_threshold) &
                        (df.sign == bout_sign) & 
                        (df.trial_time <= trial_duration_s) & 
                        (df.prey_arc_start_deg == prey_side) &
                        (df.foreground_color == f'[0.0, 0.0, {intensity}, 1.0]')
                    )
                    count_JT += mask_JT.sum()

                    mask_RT = (
                        (df.file == fish) &
                        (df.stim == Stim.PREY_CAPTURE) &
                        (df.category == ROUTINE_TURN) & 
                        (df.proba > prob_threshold) &
                        (df.sign == bout_sign) & 
                        (df.trial_time <= trial_duration_s) &
                        (df.prey_arc_start_deg == prey_side) &
                        (df.foreground_color == f'[0.0, 0.0, {intensity}, 1.0]')
                    )
                    count_RT += mask_RT.sum()

                JT_freq[g_idx, fish_idx, lat_idx, uv_idx] = count_JT / (len(lat)*trial_duration_s)
                RT_freq[g_idx, fish_idx, lat_idx, uv_idx] = count_RT / (len(lat)*trial_duration_s)

mecp2_ipsi_jt = JT_freq[0,:,0,:]
mecp2_contra_jt = JT_freq[0,:,1,:]
nacre_ipsi_jt = JT_freq[1,:,0,:]
nacre_contra_jt = JT_freq[1,:,1,:]

def plot_with_shading(ax, x, data, color, label, linestyle='-'):
    mu = np.nanmean(data, axis=0)
    err = sem(data, axis=0, nan_policy='omit') 
    ax.plot(x, mu, color=color, label=label, linestyle=linestyle, lw=2)
    ax.fill_between(x, mu - err, mu + err, color=color, alpha=0.2, lw=0)

fig, ax = plt.subplots(figsize=(8, 6))
plot_with_shading(ax, uv_intensities, mecp2_ipsi_jt, COLOR_MECP2, 'mecp2-mutant (Ipsi)', '-')
plot_with_shading(ax, uv_intensities, mecp2_contra_jt, COLOR_MECP2, 'mecp2-mutant (Contra)', '--')
plot_with_shading(ax, uv_intensities, nacre_ipsi_jt, COLOR_WT, 'wild type (Ipsi)', '-')
plot_with_shading(ax, uv_intensities, nacre_contra_jt, COLOR_WT, 'wild type (Contra)', '--')
ax.set_xscale('log') 
ax.set_xlabel('UV Intensity')
ax.set_ylabel('JT Frequency (Hz)')
ax.legend(frameon=False, loc='upper left')
sns.despine() 
plt.tight_layout()
plt.savefig(f"UV_intensity_JT_IPSI_CONTRA.png", format='png', dpi=300, bbox_inches='tight')
plt.savefig(f"UV_intensity_JT_IPSI_CONTRA.svg", format='svg', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
plot_with_shading(ax, uv_intensities, RT_freq[0,:,0,:], COLOR_MECP2, 'mecp2-mutant (Ipsi)', '-')
plot_with_shading(ax, uv_intensities, RT_freq[0,:,1,:], COLOR_MECP2, 'mecp2-mutant (Contra)', '--')
plot_with_shading(ax, uv_intensities, RT_freq[1,:,0,:], COLOR_WT, 'wild type (Ipsi)', '-')
plot_with_shading(ax, uv_intensities, RT_freq[1,:,1,:], COLOR_WT, 'wild type (Contra)', '--')
ax.set_xscale('log') 
ax.set_xlabel('UV Intensity')
ax.set_ylabel('RT Frequency (Hz)')
ax.legend(frameon=False, loc='upper left')
sns.despine() 
plt.tight_layout()
plt.savefig(f"UV_intensity_RT.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

valence_index_stim_side_mecp2 = (JT_freq[0,:,0,:] - RT_freq[0,:,1,:]) / (JT_freq[0,:,0,:] + RT_freq[0,:,1,:])
valence_index_ctrl_mecp2 = (JT_freq[0,:,1,:] - RT_freq[0,:,0,:]) / (JT_freq[0,:,1,:] + RT_freq[0,:,0,:])
valence_index_stim_side_wt = (JT_freq[1,:,0,:] - RT_freq[1,:,1,:]) / (JT_freq[1,:,0,:] + RT_freq[1,:,1,:])
valence_index_ctrl_wt = (JT_freq[1,:,1,:] - RT_freq[1,:,0,:]) / (JT_freq[1,:,1,:] + RT_freq[1,:,0,:])

fig, ax = plt.subplots(figsize=(8, 6))
plot_with_shading(ax, uv_intensities, valence_index_stim_side_mecp2, COLOR_MECP2, 'mecp2-mutant stim side', '-')
plot_with_shading(ax, uv_intensities, valence_index_ctrl_mecp2, COLOR_MECP2, 'mecp2-mutant ctrl', '--')
plot_with_shading(ax, uv_intensities, valence_index_stim_side_wt, COLOR_WT, 'WT stim side', '-')
plot_with_shading(ax, uv_intensities, valence_index_ctrl_wt, COLOR_WT, 'WT ctrl', '--')
ax.set_xscale('log') 
ax.set_xlabel('UV Intensity')
ax.set_ylabel('VI')
ax.legend(frameon=False, loc='upper left')
sns.despine() 
plt.tight_layout()
plt.savefig(f"UV_intensity_VI.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mecp2_ipsi_jt   = JT_freq[0, :, 0, :]  
mecp2_contra_jt = JT_freq[0, :, 1, :]
nacre_ipsi_jt   = JT_freq[1, :, 0, :]
nacre_contra_jt = JT_freq[1, :, 1, :]

def remove_empty_fish(matrix):
    return matrix[~np.isnan(matrix).all(axis=1)]

# Filter your matrices to only display active fish
mecp2_ipsi_clean   = remove_empty_fish(mecp2_ipsi_jt)
mecp2_contra_clean = remove_empty_fish(mecp2_contra_jt)
nacre_ipsi_clean   = remove_empty_fish(nacre_ipsi_jt)
nacre_contra_clean = remove_empty_fish(nacre_contra_jt)

# Calculate a single, unified color ceiling across the active data
global_vmax = np.nanpercentile([
    mecp2_ipsi_jt, 
    mecp2_contra_jt, 
    nacre_ipsi_jt, 
    nacre_contra_jt
],99.5)

# Initialize the grid
fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=False)
(ax_m_ipsi, ax_m_contra), (ax_n_ipsi, ax_n_contra) = axes

heatmap_configs = [
    (mecp2_ipsi_clean,   ax_m_ipsi, COLOR_MECP2),
    (mecp2_contra_clean, ax_m_contra, COLOR_MECP2),
    (nacre_ipsi_clean,   ax_n_ipsi, COLOR_WT),
    (nacre_contra_clean, ax_n_contra, COLOR_WT)
]

x_labels = [str(intensity) for intensity in uv_intensities]

for matrix_data, ax_target, col in heatmap_configs:
    num_rows, num_cols = matrix_data.shape
    cell_aspect = num_cols / num_rows
    
    sns.heatmap(
        data=matrix_data,
        ax=ax_target,
        cmap="gist_yarg",
        vmin=0.0,
        vmax=global_vmax,
        xticklabels=x_labels,
        yticklabels=True,  
        cbar=False,
        robust=True
    )
    ax_target.set_aspect(cell_aspect, adjustable='box')    
    for spine in ax_target.spines.values():
        spine.set_visible(True)
        spine.set_color(col)
        spine.set_linewidth(1.2)

for ax in axes.flat:
    ax.label_outer()  

axes[0, 0].set_title("ipsilateral", fontsize=12, fontweight='bold')
axes[0, 1].set_title("contralateral", fontsize=12, fontweight='bold')
axes[0, 0].text(-0.15, 0.5, "mecp2-mutant", color=COLOR_MECP2, fontsize=12, 
                fontweight='bold', ha='center', va='center', rotation=90, transform=axes[0,0].transAxes)
axes[0, 0].text(-0.075, 0.5, f"#Larvae (n = {mecp2_ipsi_clean.shape[0]})", color='black', 
                fontweight='bold', ha='center', va='center', rotation=90, transform=axes[0,0].transAxes)
axes[1, 0].text(-0.15, 0.5, "wild type", color=COLOR_WT, fontsize=12, 
                fontweight='bold', ha='center', va='center', rotation=90, transform=axes[1,0].transAxes)
axes[1, 0].text(-0.075, 0.5, f"#Larvae (n = {nacre_ipsi_clean.shape[0]})", color='black',  
                fontweight='bold', ha='center', va='center', rotation=90, transform=axes[1,0].transAxes)

axes[0, 0].set_yticklabels([])
axes[1, 0].set_yticklabels([])
axes[0, 1].set_yticks([])
axes[1, 1].set_yticks([])

# Configure the bottom row x-axis labels layout cleanly
for ax in axes[-1, :]:
    ax.set_xlabel("UV Intensity", fontsize=11, labelpad=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=10)

# Adjust padding structure tightly
plt.tight_layout()
fig.subplots_adjust(right=0.84, wspace=0.02, hspace=0.1)
cbar_ax = fig.add_axes([0.87, 0.28, 0.02, 0.45])  # [left, bottom, width, height]  
sm = plt.cm.ScalarMappable(cmap="gist_yarg", norm=plt.Normalize(vmin=0.0, vmax=global_vmax))
sm.set_array([])

# Add a matching border to the colorbar frame for visual consistency
cb = fig.colorbar(sm, cax=cbar_ax, label='J-Turn Frequency (Hz)')
cb.outline.set_visible(True)
cb.outline.set_edgecolor('black')
cb.outline.set_linewidth(1.0)

plt.savefig("UV_Intensity_Heatmaps.png", format='png', dpi=300, bbox_inches='tight')
plt.savefig("UV_Intensity_Heatmaps.svg", format='svg', bbox_inches='tight')
plt.show()

######### stats

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def plot_significance_spans(ax, x_values, is_significant, color='gray', alpha=0.15, zorder=0):

    x_array = np.asarray(x_values)
    sig_array = np.asarray(is_significant, dtype=bool)
    if len(x_array) != len(sig_array):
        raise ValueError("x_values and is_significant must have the same length.")
        
    log_x = np.log10(x_array)
    widths = np.diff(log_x)

    log_bounds = np.zeros(len(x_array) + 1)
    log_bounds[1:-1] = log_x[:-1] + widths / 2.0
    log_bounds[0] = log_x[0] - widths[0] / 2.0
    log_bounds[-1] = log_x[-1] + widths[-1] / 2.0
    bounds = 10**log_bounds

    in_block = False
    block_start = None
    for i, sig in enumerate(sig_array):
        if sig and not in_block:
            block_start = bounds[i]
            in_block = True
        elif not sig and in_block:
            ax.axvspan(block_start, bounds[i], color=color, alpha=alpha, zorder=zorder)
            in_block = False
            
    if in_block:
        ax.axvspan(block_start, bounds[-1], color=color, alpha=alpha, zorder=zorder)

p_values = []
u_statistics = []
mecp2_means = []
nacre_means = []
num_intensities = mecp2_ipsi_clean.shape[1]
uv_labels = [str(intensity) for intensity in uv_intensities]
for i in range(num_intensities):
    mecp2_dist = mecp2_ipsi_clean[:, i]
    nacre_dist = nacre_ipsi_clean[:, i]
    u_stat, p_val = mannwhitneyu(mecp2_dist, nacre_dist, alternative='two-sided', nan_policy='omit')
    u_statistics.append(u_stat)
    p_values.append(p_val)
    mecp2_means.append(np.nanmean(mecp2_dist))
    nacre_means.append(np.nanmean(nacre_dist))
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

stats_df = pd.DataFrame({
    'UV_Intensity': uv_labels,
    'Mecp2_Mean_Hz': mecp2_means,
    'Nacre_Mean_Hz': nacre_means,
    'U_Statistic': u_statistics,
    'Raw_p': p_values,
    'FDR_Corrected_p': p_corrected,
    'Raw_p < 0.05': np.array(p_values) < 0.05
})

print("=========================================================")
print("  GENOTYPE COMPARISON: MECP2 (N=40) vs. NACRE (N=40)     ")
print("=========================================================")
print(stats_df.to_string(index=False, formatters={
    'Mecp2_Mean_Hz': '{:,.3f}'.format,
    'Nacre_Mean_Hz': '{:,.3f}'.format,
    'Raw_p': '{:,.4e}'.format,
    'FDR_Corrected_p': '{:,.4e}'.format
}))

plot_contra = True

fig, ax = plt.subplots(figsize=(12, 5))
plot_with_shading(ax, uv_intensities, mecp2_ipsi_clean, COLOR_MECP2, 'mecp2-mutant (Ipsi)', '-')
plot_with_shading(ax, uv_intensities, nacre_ipsi_clean, COLOR_WT, 'wild type (Ipsi)', '-')
if plot_contra:
    plot_with_shading(ax, uv_intensities, mecp2_contra_jt, COLOR_MECP2, 'mecp2-mutant (Contra)', '--')
    plot_with_shading(ax, uv_intensities, nacre_contra_jt, COLOR_WT, 'wild type (Contra)', '--')

is_significant = np.array(p_values) < 0.05
plot_significance_spans(ax, uv_intensities, is_significant, color='gray', alpha=0.15, zorder=0)
ax.set_xscale('log') 
ax.set_ylim(0, ax.get_ylim()[1])
ax.set_xlabel("UV Intensity",)
ax.set_ylabel("J-Turn Frequency (Hz)")
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.legend(frameon=False, loc='upper left')
plt.tight_layout()

filename = "UV_Intensity_ANOVA_IPSI"
if plot_contra:
    filename += "_CONTRA"
plt.savefig(filename + ".png", format='png', dpi=300, bbox_inches='tight')
plt.savefig(filename + ".svg", format='svg', bbox_inches='tight')
plt.show()

import statsmodels.api as sm
from statsmodels.formula.api import ols

def anova(quantity, xrange):

    q1 = quantity[0][:, xrange]
    q2 = quantity[1][:, xrange]
    
    stim1 = np.tile(np.arange(1, q1.shape[1] + 1), (q1.shape[0], 1))
    stim2 = np.tile(np.arange(1, q2.shape[1] + 1), (q2.shape[0], 1))
    stim = np.vstack([stim1, stim2]).flatten()
    
    cond1 = np.ones(q1.shape)
    cond2 = 2 * np.ones(q2.shape)
    cond = np.vstack([cond1, cond2]).flatten()
    
    quant = np.vstack([q1, q2]).flatten()
    
    df = pd.DataFrame({
        'quant': quant,
        'cond': pd.Categorical(cond),  # Treat as categorical/factors
        'stim': pd.Categorical(stim)
    })
    
    # The formula 'quant ~ cond + stim' specifies a two-way ANOVA without interaction
    model = ols('quant ~ cond + stim', data=df).fit()
    tbl = sm.stats.anova_lm(model, typ=1)  # typ=1 matches MATLAB's default sequential SSE
    
    # Extract p-values for 'cond' and 'stim'
    p = tbl['PR(>F)'].dropna().tolist()
    
    return p, tbl

quantity = [mecp2_ipsi_clean, nacre_ipsi_clean]
xrange = np.arange(18)
anova(quantity, xrange)

############# Sigmoid

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import legend_handler as handler
from scipy.optimize import curve_fit

# 1. Clean your data arrays
mecp2_ipsi_clean   = remove_empty_fish(mecp2_ipsi_jt)
mecp2_contra_clean = remove_empty_fish(mecp2_contra_jt)
nacre_ipsi_clean   = remove_empty_fish(nacre_ipsi_jt)
nacre_contra_clean = remove_empty_fish(nacre_contra_jt)

intensity_stop = 15

log_x = np.log10(uv_intensities[0:intensity_stop])
x_smooth = np.linspace(log_x.min(), log_x.max(), 200) 

# --- Models ---
def sigmoidal_model(x, bottom, top, log_ec50, hill_slope):
    return bottom + (top - bottom) / (1 + 10**((log_ec50 - x) * hill_slope))

def linear_model(x, slope, intercept):
    return slope * x + intercept

def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# --- Optimization Settings ---
n_mecp2 = mecp2_ipsi_clean.shape[0]
n_nacre = nacre_ipsi_clean.shape[0]

initial_guesses_sig = [np.min(mecp2_ipsi_clean), np.max(mecp2_ipsi_clean), -1.0, 1.0]
initial_guesses_lin = [0.0, np.mean(mecp2_contra_clean)]

bounds_sig = (
    [0, 0, log_x.min()-1, 0.01],   
    [10, 10, log_x.max()+1, 5]
)
# Allowing slopes to be slightly positive or negative for the control
bounds_lin = ([-2, 0], [2, 10]) 

# --- Tracking Arrays for Bootstrap ---
preds_mecp2_ipsi = []
preds_nacre_ipsi = []
preds_mecp2_contra = []
preds_nacre_contra = []

ec50_diffs = []
ec50_m_list = []
ec50_n_list = []
slope_m_contra_list = []
slope_n_contra_list = []
r2_m_ipsi_list = []
r2_n_ipsi_list = []
r2_m_contra_list = []
r2_n_contra_list = []
rmse_m_ipsi_list = []
rmse_n_ipsi_list = []
rmse_m_contra_list = []
rmse_n_contra_list = []

n_iterations = 10_000

print(f"Running {n_iterations:,} bootstrap iterations...")
for i in range(n_iterations):
    boot_m = np.random.choice(n_mecp2, size=n_mecp2, replace=True)
    boot_n = np.random.choice(n_nacre, size=n_nacre, replace=True)
    
    # Calculate means for this bootstrap sample
    mean_m_ipsi   = np.mean(mecp2_ipsi_clean[boot_m, 0:intensity_stop], axis=0)
    mean_n_ipsi   = np.mean(nacre_ipsi_clean[boot_n, 0:intensity_stop], axis=0)
    mean_m_contra = np.mean(mecp2_contra_clean[boot_m, 0:intensity_stop], axis=0)
    mean_n_contra = np.mean(nacre_contra_clean[boot_n, 0:intensity_stop], axis=0)
    
    try:
        # Fit Ipsilateral groups with Sigmoid
        popt_m_i, _ = curve_fit(sigmoidal_model, log_x, mean_m_ipsi, p0=initial_guesses_sig, bounds=bounds_sig, maxfev=5000)
        popt_n_i, _ = curve_fit(sigmoidal_model, log_x, mean_n_ipsi, p0=initial_guesses_sig, bounds=bounds_sig, maxfev=5000)
        popt_m_c, _ = curve_fit(linear_model, log_x, mean_m_contra, p0=initial_guesses_lin, bounds=bounds_lin, maxfev=5000)
        popt_n_c, _ = curve_fit(linear_model, log_x, mean_n_contra, p0=initial_guesses_lin, bounds=bounds_lin, maxfev=5000)

        # Metrics extraction
        ec50_m = 10**popt_m_i[2]
        ec50_n = 10**popt_n_i[2]
        ec50_diffs.append(ec50_m - ec50_n)
        ec50_m_list.append(ec50_m) 
        ec50_n_list.append(ec50_n)
        slope_m_contra_list.append(popt_m_c[0])
        slope_n_contra_list.append(popt_n_c[0])

        # goodness of fit
        pred_m_i = sigmoidal_model(log_x, *popt_m_i)
        pred_n_i = sigmoidal_model(log_x, *popt_n_i)
        pred_m_c = linear_model(log_x, *popt_m_c)
        pred_n_c = linear_model(log_x, *popt_n_c)
        r2_m_ipsi_list.append(calculate_r_squared(mean_m_ipsi, pred_m_i))
        r2_n_ipsi_list.append(calculate_r_squared(mean_n_ipsi, pred_n_i))
        r2_m_contra_list.append(calculate_r_squared(mean_m_contra, pred_m_c))
        r2_n_contra_list.append(calculate_r_squared(mean_n_contra, pred_n_c))
        rmse_m_ipsi_list.append(calculate_rmse(mean_m_ipsi, pred_m_i))
        rmse_n_ipsi_list.append(calculate_rmse(mean_n_ipsi, pred_n_i))
        rmse_m_contra_list.append(calculate_rmse(mean_m_contra, pred_m_c))
        rmse_n_contra_list.append(calculate_rmse(mean_n_contra, pred_n_c))

        # Save continuous curve predictions over the smooth X grid
        preds_mecp2_ipsi.append(sigmoidal_model(x_smooth, *popt_m_i))
        preds_nacre_ipsi.append(sigmoidal_model(x_smooth, *popt_n_i))
        preds_mecp2_contra.append(linear_model(x_smooth, *popt_m_c))
        preds_nacre_contra.append(linear_model(x_smooth, *popt_n_c))
        
    except RuntimeError:
        continue

# --- Process Statistics ---
ec50_diffs = np.array(ec50_diffs)
ec50_m_list = np.array(ec50_m_list)
ec50_n_list = np.array(ec50_n_list)
slope_m_contra_list = np.array(slope_m_contra_list)
slope_n_contra_list = np.array(slope_n_contra_list)
r2_m_ipsi_list = np.array(r2_m_ipsi_list)
r2_n_ipsi_list = np.array(r2_n_ipsi_list)
r2_m_contra_list = np.array(r2_m_contra_list)
r2_n_contra_list = np.array(r2_n_contra_list)
rmse_m_ipsi_list = np.array(rmse_m_ipsi_list)
rmse_n_ipsi_list = np.array(rmse_n_ipsi_list)
rmse_m_contra_list = np.array(rmse_m_contra_list)
rmse_n_contra_list = np.array(rmse_n_contra_list)

ci_lower = np.percentile(ec50_diffs, 2.5)
ci_upper = np.percentile(ec50_diffs, 97.5)
p_value = 2 * min(np.mean(ec50_diffs > 0), np.mean(ec50_diffs < 0))

# Extract Confidence Intervals for Ipsilateral Curves
preds_mecp2_ipsi = np.array(preds_mecp2_ipsi)
preds_nacre_ipsi = np.array(preds_nacre_ipsi)
med_m, low_m, high_m = np.percentile(preds_mecp2_ipsi, [50, 2.5, 97.5], axis=0)
med_n, low_n, high_n = np.percentile(preds_nacre_ipsi, [50, 2.5, 97.5], axis=0)

# Extract Confidence Intervals for Contralateral Curves
preds_mecp2_contra = np.array(preds_mecp2_contra)
preds_nacre_contra = np.array(preds_nacre_contra)
med_m_c, low_m_c, high_m_c = np.percentile(preds_mecp2_contra, [50, 2.5, 97.5], axis=0)
med_n_c, low_n_c, high_n_c = np.percentile(preds_nacre_contra, [50, 2.5, 97.5], axis=0)

log_ec50_m_med = np.log10(np.median(ec50_m_list))
log_ec50_m_low = np.log10(np.percentile(ec50_m_list, 2.5))
log_ec50_m_high = np.log10(np.percentile(ec50_m_list, 97.5))
log_ec50_n_med = np.log10(np.median(ec50_n_list))
log_ec50_n_low = np.log10(np.percentile(ec50_n_list, 2.5))
log_ec50_n_high = np.log10(np.percentile(ec50_n_list, 97.5))

y_mid_m = np.min(med_m) + (np.max(med_m) - np.min(med_m)) / 2
y_mid_n = np.min(med_n) + (np.max(med_n) - np.min(med_n)) / 2

# --- Print Summary Status ---
print("\n=== SUMMARY ===")
print(f"mecp2 Group EC50: {np.median(ec50_m_list):.4f} [95% CI: {np.percentile(ec50_m_list, 2.5):.4f}, {np.percentile(ec50_m_list, 97.5):.4f}]")
print(f"nacre Group EC50: {np.median(ec50_n_list):.4f} [95% CI: {np.percentile(ec50_n_list, 2.5):.4f}, {np.percentile(ec50_n_list, 97.5):.4f}]")
print(f"Group Difference: {np.median(ec50_diffs):.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Empirical P-value: {p_value:.4f}")

print("\n=== CONTROL SLOPE VALIDATION ===")
print(f"mecp2 Contra Slope: {np.median(slope_m_contra_list):.4f} [95% CI: {np.percentile(slope_m_contra_list, 2.5):.4f}, {np.percentile(slope_m_contra_list, 97.5):.4f}]")
print(f"nacre Contra Slope: {np.median(slope_n_contra_list):.4f} [95% CI: {np.percentile(slope_n_contra_list, 2.5):.4f}, {np.percentile(slope_n_contra_list, 97.5):.4f}]")
print("\n=== BOOTSTRAPPED GOODNESS OF FIT ===")

print(f"mecp2 Ipsi Sigmoid $R^2$: {np.median(r2_m_ipsi_list):.4f} [95% CI: {np.percentile(r2_m_ipsi_list, 2.5):.4f}, {np.percentile(r2_m_ipsi_list, 97.5):.4f}]")
print(f"nacre Ipsi Sigmoid $R^2$: {np.median(r2_n_ipsi_list):.4f} [95% CI: {np.percentile(r2_n_ipsi_list, 2.5):.4f}, {np.percentile(r2_n_ipsi_list, 97.5):.4f}]")
print(f"mecp2 Contra Linear $R^2$: {np.median(r2_m_contra_list):.4f} [95% CI: {np.percentile(r2_m_contra_list, 2.5):.4f}, {np.percentile(r2_m_contra_list, 97.5):.4f}]")
print(f"nacre Contra Linear $R^2$: {np.median(r2_n_contra_list):.4f} [95% CI: {np.percentile(r2_n_contra_list, 2.5):.4f}, {np.percentile(r2_n_contra_list, 97.5):.4f}]")
print(f"mecp2 Ipsi Sigmoid RMSE: {np.median(rmse_m_ipsi_list):.4f} [95% CI: {np.percentile(rmse_m_ipsi_list, 2.5):.4f}, {np.percentile(rmse_m_ipsi_list, 97.5):.4f}]")
print(f"nacre Ipsi Sigmoid RMSE: {np.median(rmse_n_ipsi_list):.4f} [95% CI: {np.percentile(rmse_n_ipsi_list, 2.5):.4f}, {np.percentile(rmse_n_ipsi_list, 97.5):.4f}]")
print(f"mecp2 Contra Linear RMSE: {np.median(rmse_m_contra_list):.4f} [95% CI: {np.percentile(rmse_m_contra_list, 2.5):.4f}, {np.percentile(rmse_m_contra_list, 97.5):.4f}]")
print(f"nacre Contra Linear RMSE: {np.median(rmse_n_contra_list):.4f} [95% CI: {np.percentile(rmse_n_contra_list, 2.5):.4f}, {np.percentile(rmse_n_contra_list, 97.5):.4f}]")

left_out_alpha = 0.4

# --- PLOTTING ---
plt.figure(figsize=(9, 6.5))

# 1. Plot Ipsilateral Sigmoid Fits (Capture lines AND shading patches)
line_m_ipsi, = plt.plot(x_smooth, med_m, color=COLOR_MECP2, lw=2.5, zorder=1)
fill_m_ipsi = plt.fill_between(x_smooth, low_m, high_m, color=COLOR_MECP2, alpha=0.15, edgecolor=None, zorder=0)

line_n_ipsi, = plt.plot(x_smooth, med_n, color=COLOR_WT, lw=2.5, zorder=1)
fill_n_ipsi = plt.fill_between(x_smooth, low_n, high_n, color=COLOR_WT, alpha=0.15, edgecolor=None, zorder=0)

# 2. Plot Contralateral Linear Fits (Dashed Lines)
line_m_contra, = plt.plot(x_smooth, med_m_c, color=COLOR_MECP2, lw=1.5, linestyle='--', zorder=1)
fill_m_contra = plt.fill_between(x_smooth, low_m_c, high_m_c, color=COLOR_MECP2, alpha=0.15, edgecolor=None, zorder=0)

line_n_contra, = plt.plot(x_smooth, med_n_c, color=COLOR_WT, lw=1.5, linestyle='--', zorder=1)
fill_n_contra = plt.fill_between(x_smooth, low_n_c, high_n_c, color=COLOR_WT, alpha=0.15, edgecolor=None, zorder=0)

# 3. Handle EC50 Horizontal Error Bars
plt.errorbar(
    x=log_ec50_m_med, y=y_mid_m, 
    xerr=[[log_ec50_m_med - log_ec50_m_low], [log_ec50_m_high - log_ec50_m_med]],
    fmt='none', color=COLOR_MECP2, capsize=5, elinewidth=1.5, capthick=1.5, zorder=5
)
plt.errorbar(
    x=log_ec50_n_med, y=y_mid_n, 
    xerr=[[log_ec50_n_med - log_ec50_n_low], [log_ec50_n_high - log_ec50_n_med]],
    fmt='none', color=COLOR_WT, capsize=5, elinewidth=1.5, capthick=1.5, zorder=5
)

# 4. Plot Raw Data Points
raw_mean_m_ipsi = np.mean(mecp2_ipsi_clean, axis=0)
raw_mean_n_ipsi = np.mean(nacre_ipsi_clean, axis=0)
raw_mean_m_contra = np.mean(mecp2_contra_clean, axis=0)
raw_mean_n_contra = np.mean(nacre_contra_clean, axis=0)

# Excluded intensities (faded points)
plt.scatter(np.log10(uv_intensities)[intensity_stop:], raw_mean_m_ipsi[intensity_stop:], color=COLOR_MECP2, edgecolor='k', zorder=5, alpha=left_out_alpha)
plt.scatter(np.log10(uv_intensities)[intensity_stop:], raw_mean_n_ipsi[intensity_stop:], color=COLOR_WT, edgecolor='k', zorder=5, alpha=left_out_alpha)
plt.scatter(np.log10(uv_intensities)[intensity_stop:], raw_mean_m_contra[intensity_stop:], marker='^', color=COLOR_MECP2, edgecolor='k', zorder=5, alpha=left_out_alpha)
plt.scatter(np.log10(uv_intensities)[intensity_stop:], raw_mean_n_contra[intensity_stop:],  marker='^', color=COLOR_WT, edgecolor='k', zorder=5, alpha=left_out_alpha)

# Included Ipsilateral Points (Circles)
scat_m_ipsi = plt.scatter(log_x, raw_mean_m_ipsi[:intensity_stop], color=COLOR_MECP2, marker='o', edgecolor='k', s=45, zorder=5)
scat_n_ipsi = plt.scatter(log_x, raw_mean_n_ipsi[:intensity_stop], color=COLOR_WT, marker='o', edgecolor='k', s=45, zorder=5)

# Included Contralateral Points (Triangles)
scat_m_contra = plt.scatter(log_x, raw_mean_m_contra[:intensity_stop], color=COLOR_MECP2, marker='^', edgecolor='k', s=45, zorder=5)
scat_n_contra = plt.scatter(log_x, raw_mean_n_contra[:intensity_stop], color=COLOR_WT, marker='^', edgecolor='k', s=45, zorder=5)

# 5. Aesthetics & Legibility
regular_ticks = [0.01, 0.03, 0.1, 0.3, 1.0]
plt.xticks(np.log10(regular_ticks), [str(t) for t in regular_ticks])
plt.xlim(np.log10(0.008), np.log10(1.2))

plt.xlabel('UV Intensity')
plt.ylabel('J-Turn frequency (Hz)')

# --- CUSTOM CLEAN LEGEND MAPPING ---
# Superpose the shading block, line style, and marker shape on top of each other!
legend_handles = [
    (line_m_ipsi, scat_m_ipsi),
    (line_n_ipsi, scat_n_ipsi),
    (line_m_contra, scat_m_contra),
    (line_n_contra, scat_n_contra)
]
legend_labels = [
    'mecp2-mutant (Ipsi)',
    'wild type (Ipsi)',
    'mecp2-mutant (Contra)',
    'wild type (Contra)'
]

plt.legend(
    handles=legend_handles, 
    labels=legend_labels, 
    loc='upper left', 
    frameon=False, 
    handler_map={tuple: handler.HandlerTuple(ndivide=None)}
)

plt.grid(True, which='both', linestyle='--', alpha=0.3) 
plt.savefig("UV_Intensity_Sigmoid.png", format='png', dpi=300, bbox_inches='tight')
plt.savefig("UV_Intensity_Sigmoid.svg", format='svg', bbox_inches='tight')
plt.show()

#############

groups = ['mecp2/bouts.csv', 'nacre/bouts.csv']
groups_name = ['mecp2-mutant', 'wild type']
groups_color = {'mecp2-mutant': COLOR_MECP2, 'wild type': COLOR_WT}

ROOT = Path('/media/martin/DATA_18TB/PreyCapture_mecp2/size')
JTURN = bouts_category_name_short.index('JT')
ROUTINE_TURN = bouts_category_name_short.index('RT')

prob_threshold = 0.5
trial_duration_s = 7
N_fish = 40

class PreySide(IntEnum):
    LEFT = -20
    RIGHT = 20

ipsilateral = [(BoutSign.LEFT, PreySide.LEFT), (BoutSign.RIGHT, PreySide.RIGHT)]
contralateral = [(BoutSign.LEFT, PreySide.RIGHT), (BoutSign.RIGHT, PreySide.LEFT)]
laterality = [ipsilateral, contralateral]
prey_sz = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]

JT_freq = np.full((len(groups), N_fish, len(laterality), len(prey_sz)), np.nan, dtype=np.float32)
RT_freq = np.full((len(groups), N_fish, len(laterality), len(prey_sz)), np.nan, dtype=np.float32)

for g_idx, g in enumerate(groups):

    bout_file = ROOT/g 

    qc_file = bout_file.with_name("qc.csv")
    blacklisted = []
    if qc_file.exists():
        quality_control = pd.read_csv(qc_file)
        blacklisted.extend(quality_control.file.to_list())

    df = pd.read_csv(bout_file)
    file = df[df.stim == Stim.PREY_CAPTURE].file.unique()
    
    for fish_idx, fish in enumerate(file):

        if fish in blacklisted:
            print(f'{fish} did not pass qc')
            continue

        for sz_idx, sz in enumerate(prey_sz):
            for lat_idx, lat in enumerate(laterality): 
                count_JT = 0
                count_RT = 0
                for bout_sign, prey_side in lat: 
                    mask_JT = (
                        (df.file == fish) &
                        (df.stim == Stim.PREY_CAPTURE) &
                        (df.category == JTURN) & 
                        (df.proba > prob_threshold) &
                        (df.sign == bout_sign) & 
                        (df.trial_time <= trial_duration_s) & 
                        (df.prey_arc_start_deg == prey_side) &
                        (df.prey_radius_mm == sz)
                    )
                    count_JT += mask_JT.sum()

                    mask_RT = (
                        (df.file == fish) &
                        (df.stim == Stim.PREY_CAPTURE) &
                        (df.category == ROUTINE_TURN) & 
                        (df.proba > prob_threshold) &
                        (df.sign == bout_sign) & 
                        (df.trial_time <= trial_duration_s) &
                        (df.prey_arc_start_deg == prey_side) &
                        (df.prey_radius_mm == sz)
                    )
                    count_RT += mask_RT.sum()

                JT_freq[g_idx, fish_idx, lat_idx, sz_idx] = count_JT / (len(lat)*trial_duration_s)
                RT_freq[g_idx, fish_idx, lat_idx, sz_idx] = count_RT / (len(lat)*trial_duration_s)


def plot_with_shading(ax, x, data, color, label, linestyle='-'):
    mu = np.nanmean(data, axis=0)
    err = sem(data, axis=0, nan_policy='omit') 
    ax.plot(x, mu, color=color, label=label, linestyle=linestyle, lw=2)
    ax.fill_between(x, mu - err, mu + err, color=color, alpha=0.2, lw=0)


fig, ax = plt.subplots(figsize=(8, 6))
plot_with_shading(ax, prey_sz, JT_freq[0,:,0,:], COLOR_MECP2, 'mecp2-mutant (Ipsi)', '-')
plot_with_shading(ax, prey_sz, JT_freq[0,:,1,:], COLOR_MECP2, 'mecp2-mutant (Contra)', '--')
plot_with_shading(ax, prey_sz, JT_freq[1,:,0,:], COLOR_WT, 'wild type (Ipsi)', '-')
plot_with_shading(ax, prey_sz, JT_freq[1,:,1,:], COLOR_WT, 'wild type (Contra)', '--')
ax.set_xscale('log') 
ax.set_xlabel('prey radius (mm)')
ax.set_ylabel('JT Frequency (Hz)')
ax.legend(frameon=False, loc='upper left')
sns.despine() 
plt.tight_layout()
plt.savefig(f"prey_size_JT.png", format='png', dpi=100, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
plot_with_shading(ax, prey_sz, RT_freq[0,:,0,:], COLOR_MECP2, 'mecp2-mutant (Ipsi)', '-')
plot_with_shading(ax, prey_sz, RT_freq[0,:,1,:], COLOR_MECP2, 'mecp2-mutant (Contra)', '--')
plot_with_shading(ax, prey_sz, RT_freq[1,:,0,:], COLOR_WT, 'wild type (Ipsi)', '-')
plot_with_shading(ax, prey_sz, RT_freq[1,:,1,:], COLOR_WT, 'wild type (Contra)', '--')
ax.set_xscale('log') 
ax.set_xlabel('prey radius (mm)')
ax.set_ylabel('RT Frequency (Hz)')
ax.legend(frameon=False, loc='upper left')
sns.despine() 
plt.tight_layout()
plt.savefig(f"prey_size_RT.png", format='png', dpi=100, bbox_inches='tight')
plt.show()

### approach to escape ratio?
valence_index_stim_side_mecp2 = (JT_freq[0,:,0,:] - RT_freq[0,:,1,:]) / (JT_freq[0,:,0,:] + RT_freq[0,:,1,:])
valence_index_ctrl_mecp2 = (JT_freq[0,:,1,:] - RT_freq[0,:,0,:]) / (JT_freq[0,:,1,:] + RT_freq[0,:,0,:])
valence_index_stim_side_wt = (JT_freq[1,:,0,:] - RT_freq[1,:,1,:]) / (JT_freq[1,:,0,:] + RT_freq[1,:,1,:])
valence_index_ctrl_wt = (JT_freq[1,:,1,:] - RT_freq[1,:,0,:]) / (JT_freq[1,:,1,:] + RT_freq[1,:,0,:])

fig, ax = plt.subplots(figsize=(8, 6))
plot_with_shading(ax, prey_sz, valence_index_stim_side_mecp2, COLOR_MECP2, 'mecp2-mutant stim side', '-')
#plot_with_shading(ax, prey_sz, valence_index_ctrl_mecp2, COLOR_MECP2, 'mecp2-mutant ctrl', '--')
plot_with_shading(ax, prey_sz, valence_index_stim_side_wt, COLOR_WT, 'WT stim side', '-')
#plot_with_shading(ax, prey_sz, valence_index_ctrl_wt, COLOR_WT, 'WT ctrl', '--')
#ax.set_xscale('log') 
ax.set_xlabel('Prey radius')
ax.set_ylabel('VI')
ax.legend(frameon=False, loc='upper left')
sns.despine() 
plt.tight_layout()
plt.savefig(f"prey_size_VI.png", format='png', dpi=300, bbox_inches='tight')
plt.show()


###########################

groups = ['mecp2/danieau/bouts.csv', 'AB/danieau/bouts.csv', 'WT/danieau/bouts.csv']
groups_name = ['mecp2-mutant', 'wild type (AB)', 'wild type (TLN)']
groups_color = {'mecp2-mutant': COLOR_MECP2, 'wild type (AB)': COLOR_WT, 'wild type (TLN)': COLOR_WT_TLN}

ROOT = Path('/home/martin/Desktop/DATA')
ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/Screen')
ROOT = Path('/media/martin/DATA_18TB/Screen')

RT = bouts_category_name_short.index('RT')
prob_threshold = 0.5
trial_duration_s = 25
N_fish = 48

N_trials = 5
time_bins = [
    [0, 2.5],
    [2.5, 5],
    [5, 7.5],
    [7.5, 10]
]

class GratingSide(IntEnum):
    LEFT = -90
    RIGHT = 90

ipsilateral = [(BoutSign.LEFT, GratingSide.LEFT), (BoutSign.RIGHT, GratingSide.RIGHT)]
contralateral = [(BoutSign.LEFT, GratingSide.RIGHT), (BoutSign.RIGHT, GratingSide.LEFT)]
laterality = [ipsilateral, contralateral]

RT_freq = np.full((len(groups), N_fish, len(laterality), N_trials, len(time_bins)), np.nan, dtype=np.float32)

for g_idx, g in enumerate(groups):

    bout_file = ROOT/g 

    qc_file = bout_file.with_name("qc.csv")
    blacklisted = []
    if qc_file.exists():
        quality_control = pd.read_csv(qc_file)
        blacklisted.extend(quality_control.file.to_list())

    df = pd.read_csv(bout_file)
    file = df[df.stim == Stim.OMR].file.unique()
    
    for fish_idx, fish in enumerate(file):

        if fish in blacklisted:
            print(f'{fish} did not pass qc')
            continue

        for lat_idx, lat in enumerate(laterality): 
            for trial in range(N_trials):
                for bin_idx, (t_start, t_stop) in enumerate(time_bins):
                    count_all_bouts = 0
                    count_RT = 0
                    for bout_sign, grating_side in lat: 
                        mask_RT = (
                            (df.file == fish) &
                            (df.stim == Stim.OMR) &
                            (df.category == RT) & 
                            (df.proba > prob_threshold) &
                            (df.trial_time >= t_start) &
                            (df.trial_time < t_stop) &
                            (df.trial_num == trial) &
                            (df.sign == bout_sign) & 
                            (df.omr_angle_deg == grating_side)
                        )
                        count_RT += mask_RT.sum()
                    RT_freq[g_idx, fish_idx, lat_idx, trial, bin_idx] = count_RT / (len(lat) * (t_stop - t_start))


lat_names = ['Ipsilateral', 'Contralateral']
bin_labels = [f"{b[0]}-{b[1]}s" for b in time_bins]

for data_type, data in [('Frequency (Hz)', RT_freq)]:

    plot_heatmap(
        data,
        data_type,
        vmax = 0.6
    )

    # plot_barplot(
    #     data,
    #     data_type,
    #     trials=[0,1],
    #     time_bins=[0,1,2]
    # )



####
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import umap 

X_list = []
y_list = []

for g, gname in zip(groups, groups_name):
    filename = (ROOT/g).with_suffix('.npz')
    with np.load(filename, allow_pickle=True) as data:
        bout_frequency = data['bout_frequency']
        epoch_bin_names = data['labels_2']

    epoch_index = [idx for (idx, name) in enumerate(epoch_bin_names) if 'prey capture' in name]
    subset = bout_frequency[:, :, epoch_index, :, :]
    
    valid_fish_mask = ~np.all(np.isnan(subset), axis=(1, 2, 3, 4))
    clean_subset = subset[valid_fish_mask, :, :, :, :]

    actual_trials_mask = ~np.all(np.isnan(clean_subset), axis=(0, 2, 3, 4))
    clean_subset = clean_subset[:, actual_trials_mask, :, :, :]
    print(np.isnan(clean_subset).sum())
    
    num_valid_fish = clean_subset.shape[0]
    flattened_features = clean_subset.reshape(num_valid_fish, -1)
    
    X_list.append(flattened_features)
    y_list.extend([gname] * num_valid_fish)

X_final = np.vstack(X_list)
y_final = np.array(y_list)

X_scaled = StandardScaler().fit_transform(X_final)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

reducer = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y_final)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

embeddings = [X_pca, X_umap, X_lda]
titles = [
    f"PCA (Linear)\nVar Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%",
    "UMAP (Non-Linear)",
    "LDA (Supervised Linear)"
]
x_labels = ["PC 1", "UMAP 1", "LD 1"]
y_labels = ["PC 2", "UMAP 2", "LD 2"]

for i, ax in enumerate(axes):
    # Clean background and borders
    ax.set_facecolor('#fafafa')
    for spine in ax.spines.values():
        spine.set_visible(False)
    #ax.tick_params(both=False, labelbottom=False, labelleft=False) # Hide raw coordinate ticks
    ax.grid(True, linestyle=':', color='#e0e0e0', alpha=0.5)
    
    # Scatter plot
    sns.scatterplot(
        x=embeddings[i][:, 0], 
        y=embeddings[i][:, 1], 
        hue=y_final, 
        palette=groups_color,
        alpha=0.85, 
        edgecolor='white', 
        s=70, 
        ax=ax,
        legend=(i == 2) # Only show legend on the last plot to keep things clean
    )
    
    ax.set_title(titles[i], fontsize=13, weight='bold', pad=12, color='#232F34')
    ax.set_xlabel(x_labels[i], fontsize=11, color='#555555')
    ax.set_ylabel(y_labels[i], fontsize=11, color='#555555')

# Style the final legend nicely
axes[2].legend(title="Genotype", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

plt.suptitle("Dimensional Embedding of Zebrafish Larvae Behavioral Data", 
             fontsize=16, weight='bold', color='#232F34', y=1.05)
plt.tight_layout()
plt.show()



_, n_trials, n_epochs, n_cats, n_sides = clean_subset.shape

feature_names = []
for t in range(n_trials):
    for e in range(n_epochs):
        for c in range(n_cats):
            for s in range(n_sides):
                # Map the indices back to their names
                cat_name = bout_categories[c]
                side_name = "Ipsi" if s == 0 else "Contra"
                
                name = f"Trial_{t}_Epoch_{e}_{cat_name}_{side_name}"
                feature_names.append(name)

pca_loadings = pd.DataFrame(
    pca.components_.T,  # Transpose to make features rows
    columns=['PC1', 'PC2'],
    index=feature_names
)

lda_loadings = pd.DataFrame(
    lda.scalings_,  
    columns=['LD1', 'LD2'],
    index=feature_names
)
