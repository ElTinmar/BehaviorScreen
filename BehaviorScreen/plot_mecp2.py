from pathlib import Path
from enum import IntEnum
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon , mannwhitneyu, sem, gaussian_kde
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from BehaviorScreen.load import Directories, find_files, load_data
from BehaviorScreen.process import get_trials
from BehaviorScreen.core import Stim, BoutSign
from megabouts.utils import bouts_category_name_short

COLOR_MECP2 = '#D95319'
COLOR_WT = '#0072BD'  

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
groups = ['mecp2/danieau/bouts.csv', 'WT/danieau/bouts.csv']
groups_name = ['mecp2-mutant', 'wild type']
groups_color = {'mecp2-mutant': COLOR_MECP2, 'wild type': COLOR_WT}

##########

def epoch_masks(df):
    masks = [
        (df.stim == Stim.DARK) & (df.trial_num >= 10) & (df.trial_num < 20),
        (df.stim == Stim.BRIGHT) & (df.trial_num >= 5) & (df.trial_num < 15) & (df.time_start > 1049),
    ]
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
plt.show()

### TODO plot bout frequency during looming / distance travelled (looming + recovery)

for g in groups:

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
    for behavior_file in behavior_files:
        behavior_data = load_data(behavior_file)
        stim_trials = get_trials(behavior_data)
        looming_trials = stim_trials[stim_trials.stim_select == Stim.LOOMING]

        traj = behavior_data.tracking[['centroid_x', 'centroid_y']].to_numpy()
        traj = behavior_data.full_tracking.Swim_Bladder[['x','y']].to_numpy()

#####

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


group_names = ['Mecp2', 'WT']
lat_names = ['Ipsilateral', 'Contralateral']
bin_labels = [f"{b[0]}-{b[1]}s" for b in time_bins]
trial_labels = [f"Trial {i}" for i in range(N_trials)]

def plot_heatmap(
        data, 
        label,
        vmax = 0.6
    ):

    fig, axes = plt.subplots(len(group_names), len(lat_names), 
                            figsize=(15, 10), sharex=True, sharey=True)

    for g_idx in range(len(group_names)):
        for lat_idx in range(len(lat_names)):
            ax = axes[g_idx, lat_idx]
            data_avg = np.nanmean(data[g_idx, :,lat_idx, :, :], axis=0)
            
            sns.heatmap(data_avg, 
                        annot=True,       
                        fmt=".3f",        
                        cmap="magma",     
                        vmin=0,           
                        vmax=vmax,        
                        xticklabels=bin_labels,
                        yticklabels=trial_labels,
                        ax=ax,
                        cbar_kws={'label': label})
            
            ax.set_title(f"Group: {group_names[g_idx]} | {lat_names[lat_idx]}", fontweight='bold')
            
            if g_idx == len(group_names) - 1:
                ax.set_xlabel("Time Bins")
            if lat_idx == 0:
                ax.set_ylabel("Trial Number")

    plt.tight_layout()
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

    data_dict = {
        'Mecp2_Ipsi':   np.nanmean(data[0, :, 0, trials, time_bins], axis=0),
        'Mecp2_Contra': np.nanmean(data[0, :, 1, trials, time_bins], axis=0),
        'WT_Ipsi':      np.nanmean(data[1, :, 0, trials, time_bins], axis=0),
        'WT_Contra':    np.nanmean(data[1, :, 1, trials, time_bins], axis=0)
    }

    p_ipsi_between = mannwhitneyu(
        data_dict['Mecp2_Ipsi'][~np.isnan(data_dict['Mecp2_Ipsi'])],
        data_dict['WT_Ipsi'][~np.isnan(data_dict['WT_Ipsi'])]
    ).pvalue

    p_contra_between = mannwhitneyu(
        data_dict['Mecp2_Contra'][~np.isnan(data_dict['Mecp2_Contra'])],
        data_dict['WT_Contra'][~np.isnan(data_dict['WT_Contra'])]
    ).pvalue

    # Mecp2 (paired)
    mask_m = ~np.isnan(data_dict['Mecp2_Ipsi']) & ~np.isnan(data_dict['Mecp2_Contra'])
    p_mecp2 = wilcoxon(
        data_dict['Mecp2_Ipsi'][mask_m],
        data_dict['Mecp2_Contra'][mask_m]
    ).pvalue

    # WT (paired)
    mask_w = ~np.isnan(data_dict['WT_Ipsi']) & ~np.isnan(data_dict['WT_Contra'])
    p_wt = wilcoxon(
        data_dict['WT_Ipsi'][mask_w],
        data_dict['WT_Contra'][mask_w]
    ).pvalue

    pvals = (p_ipsi_between, p_contra_between, p_mecp2, p_wt)
    rejected, (p_ipsi_between_bf, p_contra_between_bf, p_mecp2_bf, p_wt_bf), _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

    df_plot = pd.DataFrame({
        'value': np.concatenate(list(data_dict.values())),
        'group': (
            ['mecp2-mutant'] * len(data_dict['Mecp2_Ipsi']) +
            ['mecp2-mutant'] * len(data_dict['Mecp2_Contra']) +
            ['wild type']    * len(data_dict['WT_Ipsi']) +
            ['wild type']    * len(data_dict['WT_Contra'])
        ),
        'laterality': (
            ['ipsilateral'] * len(data_dict['Mecp2_Ipsi']) +
            ['contralateral'] * len(data_dict['Mecp2_Contra']) +
            ['ipsilateral'] * len(data_dict['WT_Ipsi']) +
            ['contralateral'] * len(data_dict['WT_Contra'])
        )
    })
    df_plot = df_plot.dropna(subset=['value'])    

    plt.figure(figsize=(6, 6))

    ax = sns.barplot(
        data=df_plot,
        x='laterality',
        y='value',
        hue='group',
        palette=groups_color,
        errorbar='se',    
        capsize=0.05,      
        edgecolor='.2', 
        linewidth=1.5,
        gap=0.1,
        err_kws={'zorder': 0}
    )

    sns.stripplot(
        data=df_plot,
        x='laterality',
        y='value',
        hue='group',
        palette=groups_color, 
        jitter=0.15,
        dodge=True,
        alpha=0.5,
        edgecolor='white', 
        linewidth=1,
        size=6     
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False)

    plt.ylabel(f"J-turn {label}")
    plt.xlabel('')

    add_pval_star(ax, -0.2, 0.2,  0.55, p_ipsi_between_bf)
    add_pval_star(ax, -0.2, 0.8, 0.6, p_mecp2_bf)
    add_pval_star(ax, 0.2, 1.2, 0.65, p_wt_bf)
    add_pval_star(ax, 0.8, 1.2, 0.1, p_contra_between_bf)

    #plt.ylim(0, 0.725)
    plt.tight_layout()
    plt.savefig(f"{label}_comp.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_comp.png", format='png', dpi=100, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.kdeplot(data_dict['Mecp2_Ipsi'], bw_adjust=0.5, color=COLOR_MECP2, label='mecp2-mutant') 
    sns.kdeplot(data_dict['WT_Ipsi'],  bw_adjust=0.5, color=COLOR_WT, label='wild type') 
    plt.ylabel("Ispilateral J-turn PDF")
    plt.xlabel(label)
    plt.legend(frameon=False)
    plt.savefig(f"{label}_kde.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_kde.png", format='png', dpi=100, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.ecdfplot(data_dict['Mecp2_Ipsi'], color=COLOR_MECP2, label='mecp2-mutant')
    sns.ecdfplot(data_dict['WT_Ipsi'], color=COLOR_WT, label='wild type')
    plt.ylabel(f"Ispilateral J-turn CDF")
    plt.xlabel(label)
    plt.ylim(-0.1, 1.1)
    plt.legend(frameon=False)
    plt.savefig(f"{label}_cdf.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{label}_cdf.png", format='png', dpi=100, bbox_inches='tight')
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
        trials=[0,1,2],
        time_bins=[0]
    )


