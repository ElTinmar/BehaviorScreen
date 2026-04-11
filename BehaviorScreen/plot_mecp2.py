from pathlib import Path
from enum import IntEnum
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon , mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

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

groups = ['mecp2/danieau/bouts.csv','AB/danieau/bouts.csv']
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
                        JT_freq[g_idx, fish_idx, lat_idx, trial, bin_idx] = mask_JT.sum() / (t_stop - t_start)
                        JT_count[g_idx, fish_idx, lat_idx, trial, bin_idx] = mask_JT.sum()
                        JT_proba[g_idx, fish_idx, lat_idx, trial, bin_idx] = mask_JT.sum() / mask_all_bouts.sum()


group_names = ['Mecp2', 'WT']
lat_names = ['Ipsilateral', 'Contralateral']
bin_labels = [f"{b[0]}-{b[1]}s" for b in time_bins]
trial_labels = [f"Trial {i}" for i in range(N_trials)]

# Frequency
vmax = 0.6

fig, axes = plt.subplots(len(group_names), len(lat_names), 
                         figsize=(15, 10), sharex=True, sharey=True)

for g_idx in range(len(group_names)):
    for lat_idx in range(len(lat_names)):
        ax = axes[g_idx, lat_idx]
        data = np.nanmean(JT_proba[g_idx, :,lat_idx, :, :], axis=0)
        
        sns.heatmap(data, 
                    annot=True,       
                    fmt=".3f",        
                    cmap="magma",     
                    vmin=0,           
                    vmax=vmax,        
                    xticklabels=bin_labels,
                    yticklabels=trial_labels,
                    ax=ax,
                    cbar_kws={'label': '<J-Turn frequency>_fish (Hz)'})
        
        ax.set_title(f"Group: {group_names[g_idx]} | {lat_names[lat_idx]}", fontweight='bold')
        
        if g_idx == len(group_names) - 1:
            ax.set_xlabel("Time Bins")
        if lat_idx == 0:
            ax.set_ylabel("Trial Number")

plt.tight_layout()
plt.show()

# PROBA
vmax = 0.6

fig, axes = plt.subplots(len(group_names), len(lat_names), 
                         figsize=(15, 10), sharex=True, sharey=True)

for g_idx in range(len(group_names)):
    for lat_idx in range(len(lat_names)):
        ax = axes[g_idx, lat_idx]
        data = np.nanmean(JT_proba[g_idx, :,lat_idx, :, :], axis=0)
        
        sns.heatmap(data, 
                    annot=True,       
                    fmt=".3f",        
                    cmap="magma",     
                    vmin=0,           
                    vmax=vmax,        
                    xticklabels=bin_labels,
                    yticklabels=trial_labels,
                    ax=ax,
                    cbar_kws={'label': '<J-Turn probability>_fish'})
        
        ax.set_title(f"Group: {group_names[g_idx]} | {lat_names[lat_idx]}", fontweight='bold')
        
        if g_idx == len(group_names) - 1:
            ax.set_xlabel("Time Bins")
        if lat_idx == 0:
            ax.set_ylabel("Trial Number")

plt.tight_layout()
plt.show()

### German's barplot

data_dict = {
    'Mecp2_Ipsi':   np.nanmean(JT_freq[0, :, 0, 0:3, 0], axis=1),
    'Mecp2_Contra': np.nanmean(JT_freq[0, :, 1, 0:3, 0], axis=1),
    'WT_Ipsi':      np.nanmean(JT_freq[1, :, 0, 0:3, 0], axis=1),
    'WT_Contra':    np.nanmean(JT_freq[1, :, 1, 0:3, 0], axis=1)
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
    x='group',
    y='value',
    hue='laterality',
    palette=['#4C72B0', '#55A868'],
    errorbar='se',    
    capsize=0,      
    alpha=0.9,      
    edgecolor='.2', 
    linewidth=1.5,
    gap=0.1
)

sns.stripplot(
    data=df_plot,
    x='group',
    y='value',
    hue='laterality',
    palette=['#4C72B0', '#55A868'], 
    jitter=0.15,
    dodge=True,
    alpha=0.5,
    edgecolor='white', 
    linewidth=1,
    size=6     
)

# Fix legend duplication
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title='Prey side')

plt.ylabel('J-turn frequency (Hz)')
plt.xlabel('')

def add_pval(ax, x1, x2, y, text):
    ax.plot([x1, x1, x2, x2], [y, y*1.05, y*1.05, y], lw=1.5, color='black')
    ax.text((x1 + x2) / 2, y*1.08, text, ha='center', va='bottom')

y_max = 0.8* df_plot['value'].max()

# Positions depend on dodge → approximate:
# Mecp2: x=0, WT: x=1
# Ipsi ~ -0.2, Contra ~ +0.2 offset
add_pval(ax, -0.2, 0.2, y_max*0.85, f"p={p_mecp2_bf:.3e}")
add_pval(ax, 0.8, 1.2, y_max*1.25, f"p={p_wt_bf:.3e}")
add_pval(ax, -0.2, 0.8, y_max*1.4, f"p={p_ipsi_between_bf:.3e}")

plt.ylim(0, y_max*1.6)
plt.tight_layout()
#plt.savefig('jturn_analysis.svg', format='svg', bbox_inches='tight')
#plt.savefig('jturn_analysis.png', format='png', dpi=100, bbox_inches='tight')
plt.show()



###
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

genotype_palette = {'mecp2-mutant': COLOR_MECP2, 'wild type': COLOR_WT}

plt.figure(figsize=(6, 6))

ax = sns.barplot(
    data=df_plot,
    x='laterality',
    y='value',
    hue='group',
    palette=genotype_palette,
    errorbar='se',    
    capsize=0.05,      
    edgecolor='.2', 
    linewidth=1.5,
    gap=0.1,
    err_kws={'zorder': 0}
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], frameon=False)

plt.ylabel('J-turn frequency (Hz)')
plt.xlabel('')

add_pval_star(ax, -0.2, 0.2,  0.55, p_ipsi_between_bf)
add_pval_star(ax, -0.2, 0.8, 0.6, p_mecp2_bf)
add_pval_star(ax, 0.2, 1.2, 0.65, p_wt_bf)
add_pval_star(ax, 0.8, 1.2, 0.1, p_contra_between_bf)

plt.ylim(0, 0.725)
plt.tight_layout()
plt.savefig('jturn_analysis.svg', format='svg', bbox_inches='tight')
plt.savefig('jturn_analysis.png', format='png', dpi=100, bbox_inches='tight')
plt.show()