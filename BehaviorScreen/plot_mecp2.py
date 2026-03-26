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

plt.rcParams.update({
    'font.size': 12,          # Global default
    'axes.titlesize': 18,     # Title
    'axes.labelsize': 16,     # X and Y labels
    'xtick.labelsize': 14,    # X tick labels
    'ytick.labelsize': 14,    # Y tick labels
    'legend.fontsize': 12,    # Legend
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'] # Common for biology journals
})

ROOT = Path('/home/martin/Desktop/DATA')
groups = ['mecp2/danieau/bouts.csv','WT/danieau/bouts.csv']
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

heatmap = np.full((len(groups), N_fish, len(laterality), N_trials, len(time_bins)), np.nan, dtype=np.float32)

for g_idx, g in enumerate(groups):

    bout_file = ROOT/g 
    df = pd.read_csv(bout_file)
    file = df[df.stim == Stim.PREY_CAPTURE].file.unique()
    
    for fish_idx, fish in enumerate(file):
        for lat_idx, lat in enumerate(laterality): 
            for trial in range(N_trials):
                for bin_idx, (t_start, t_stop) in enumerate(time_bins):
                    for bout_sign, prey_side in lat: 
                        mask = (
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
                        heatmap[g_idx, fish_idx, lat_idx, trial, bin_idx] = mask.sum() / (t_stop - t_start)


group_names = ['Mecp2', 'WT']
lat_names = ['Ipsilateral', 'Contralateral']
bin_labels = [f"{b[0]}-{b[1]}s" for b in time_bins]
trial_labels = [f"Trial {i}" for i in range(N_trials)]

# 2. Find a global maximum for consistent color scaling
vmax = 0.6

fig, axes = plt.subplots(len(group_names), len(lat_names), 
                         figsize=(15, 10), sharex=True, sharey=True)

for g_idx in range(len(group_names)):
    for lat_idx in range(len(lat_names)):
        ax = axes[g_idx, lat_idx]
        data = np.nanmean(heatmap[g_idx, :,lat_idx, :, :], axis=0)
        
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

### German's barplot

data_dict = {
    'Mecp2_Ipsi':   np.nanmean(heatmap[0, :, 0, 0:3, 0], axis=1),
    'Mecp2_Contra': np.nanmean(heatmap[0, :, 1, 0:3, 0], axis=1),
    'WT_Ipsi':      np.nanmean(heatmap[1, :, 0, 0:3, 0], axis=1),
    'WT_Contra':    np.nanmean(heatmap[1, :, 1, 0:3, 0], axis=1)
}

p_ipsi_between = mannwhitneyu(
    data_dict['Mecp2_Ipsi'][~np.isnan(data_dict['Mecp2_Ipsi'])],
    data_dict['WT_Ipsi'][~np.isnan(data_dict['WT_Ipsi'])]
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

pvals = (p_ipsi_between, p_mecp2, p_wt)
rejected, (p_ipsi_between_bf, p_mecp2_bf, p_wt_bf), _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

df_plot = pd.DataFrame({
    'value': np.concatenate(list(data_dict.values())),
    'group': (
        ['Mecp2'] * len(data_dict['Mecp2_Ipsi']) +
        ['Mecp2'] * len(data_dict['Mecp2_Contra']) +
        ['WT']    * len(data_dict['WT_Ipsi']) +
        ['WT']    * len(data_dict['WT_Contra'])
    ),
    'laterality': (
        ['Ipsi'] * len(data_dict['Mecp2_Ipsi']) +
        ['Contra'] * len(data_dict['Mecp2_Contra']) +
        ['Ipsi'] * len(data_dict['WT_Ipsi']) +
        ['Contra'] * len(data_dict['WT_Contra'])
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
    errorbar='se',    # Standard Error of the mean
    capsize=0,      # Width of the error bar caps
    alpha=0.9,         # Make bars slightly transparent to see dots better
    edgecolor='.2', # Dark grey border
    linewidth=1.5,
    gap=0.1
)

sns.stripplot(
    data=df_plot,
    x='group',
    y='value',
    hue='laterality',
    palette=['#4C72B0', '#55A868'], # Match the bars
    jitter=0.15,
    dodge=True,
    alpha=0.5,
    edgecolor='white', # The 'halo'
    linewidth=1,
    size=6     # Slightly larger dots
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
plt.savefig('jturn_analysis.svg', format='svg', bbox_inches='tight')
plt.savefig('jturn_analysis.png', format='png', dpi=100, bbox_inches='tight')
plt.show()