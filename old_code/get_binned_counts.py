import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short
jt_index = bouts_category_name_short.index("JT")
slc_index = bouts_category_name_short.index("SLC")
llc_index = bouts_category_name_short.index("LLC")


df = pd.read_csv('filtered_bouts.csv')


#######

pc = df[df['stim'] == Stim.PREY_CAPTURE].copy()
pc['ipsi_jturn'] = (pc['category'] == jt_index) & (pc['laterality'] == Laterality.IPSILATERAL)

time_bins = [0, 0.5, 1, 2, 3, 4, 5, 10, 15, 25]
pc['time_bin'] = pd.cut(pc['trial_time'], bins=time_bins, right=False)

ipsi_counts = (
    pc.groupby(['file', 'trial_num', 'time_bin'], observed=False)['ipsi_jturn']
    .sum()
    .reset_index(name='bout_count')
)
ipsi_counts['bin_width'] = ipsi_counts['time_bin'].apply(lambda x: x.right - x.left)
ipsi_counts['time_sec'] = ipsi_counts['time_bin'].apply(lambda x: x.left)
ipsi_counts['frequency_hz'] = ipsi_counts['bout_count'] / ipsi_counts['bin_width']

fig, ax = plt.subplots()
sns.lineplot(
    data=ipsi_counts, 
    x='time_sec', 
    y='frequency_hz',
    hue='trial_num', 
    errorbar='se',  
    ax=ax
)
plt.show()

heatmap_data = (
    ipsi_counts.groupby(['trial_num', 'time_bin'], observed=False)['frequency_hz']
    .mean()
    .reset_index()
)
heatmap_matrix = heatmap_data.pivot(
    index='trial_num', 
    columns='time_bin', 
    values='frequency_hz'
)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    data=heatmap_matrix,
    annot=True,           
    fmt=".2f",            
    cbar_kws={'label': 'Mean IPSI J-turn Frequency (Hz)'},
    ax=ax
)

ax.set_title('Ipsilateral J-turn Frequency Matrix')
ax.set_xlabel('Trial Time Window (s)')
ax.set_ylabel('Trial Number')

plt.tight_layout()
plt.show()


####### 
loomings = df[df['stim'] == Stim.LOOMING].copy()
loomings['contra_escape'] = (
    ((loomings['category'] == slc_index) | (loomings['category'] == llc_index))
    & (loomings['laterality'] == Laterality.IPSILATERAL)
)

time_bins = [0, 4, 6, 10]
loomings['time_bin'] = pd.cut(loomings['trial_time'], bins=time_bins, right=False)

contra_counts = (
    loomings.groupby(['file', 'trial_num', 'time_bin'], observed=False)['contra_escape']
    .sum()
    .reset_index(name='bout_count')
)
contra_counts['bin_width'] = contra_counts['time_bin'].apply(lambda x: x.right - x.left)
contra_counts['time_sec'] = contra_counts['time_bin'].apply(lambda x: x.left)
contra_counts['frequency_hz'] = contra_counts['bout_count'] / contra_counts['bin_width']

heatmap_data = (
    contra_counts.groupby(['trial_num', 'time_bin'], observed=False)['frequency_hz']
    .mean()
    .reset_index()
)
heatmap_matrix = heatmap_data.pivot(
    index='trial_num', 
    columns='time_bin', 
    values='frequency_hz'
)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    data=heatmap_matrix,
    annot=True,           
    fmt=".2f",            
    cbar_kws={'label': 'Mean contra escape Frequency (Hz)'},
    ax=ax
)

ax.set_title('Contralateral LLC/SLC Frequency Matrix')
ax.set_xlabel('Trial Time Window (s)')
ax.set_ylabel('Trial Number')

plt.tight_layout()
plt.show()