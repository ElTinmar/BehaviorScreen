from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short

rt_index = bouts_category_name_short.index("RT")
hat_index = bouts_category_name_short.index("HAT")

ROOT = Path('/media/martin/DATA_18TB/Screen')
df = pd.read_csv(ROOT / 'bouts_control.csv')


print(f"#fish: {len(df.file.unique())}")

fine_dt = 0.02
time_bins = np.arange(0, 10 + fine_dt, fine_dt)
window_size_steps = int(1.0 / fine_dt)

sub_df = df[df['stim'] == Stim.OMR].copy()

sub_df['ipsi_rt'] = ((sub_df['category'] == rt_index) | (sub_df['category'] == hat_index)) & (sub_df['laterality'] == Laterality.IPSILATERAL)
sub_df['contra_rt'] = ((sub_df['category'] == rt_index) | (sub_df['category'] == hat_index)) & (sub_df['laterality'] == Laterality.CONTRALATERAL)
sub_df['time_bin'] = pd.cut(sub_df['trial_time'], bins=time_bins, right=False)


# We leave the data inside the MultiIndex structure (no .reset_index() yet) 
# to protect the execution order of the upcoming rolling window.
counts = (
    sub_df.groupby(['file', 'trial_num', 'time_bin'], observed=False)
    .agg(
        rt_ipsi_count=('ipsi_rt', 'sum'),
        rt_contra_count=('contra_rt', 'sum'),
    )
)

# Apply centered rolling sum across the timeline per fish/trial
counts['rolling_rt_ipsi'] = (
    counts.groupby(['file', 'trial_num'])['rt_ipsi_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

counts['rolling_rt_contra'] = (
    counts.groupby(['file', 'trial_num'])['rt_contra_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

# Flatten the MultiIndex now that the time-sensitive rolling math is finished
counts = counts.reset_index()

# Convert the interval objects into floating-point seconds for the X-axis
counts['time_sec'] = counts['time_bin'].apply(lambda x: x.left).astype(float)

# Calculate true instantaneous frequency (Bouts per second)
counts['rt_ipsi_hz'] = counts['rolling_rt_ipsi'] / 1.0
counts['rt_contra_hz'] = counts['rolling_rt_contra'] / 1.0

fig, (ax_ipsi, ax_contra) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, sharey=True)

sns.lineplot(
    data=counts, x='time_sec', y='rt_ipsi_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_ipsi, zorder=3
)
ax_ipsi.set_ylabel('Ipsi RT+HAT turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_ipsi.grid(True, linestyle=':', alpha=0.5)

sns.lineplot(
    data=counts, x='time_sec', y='rt_contra_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_contra, zorder=3,
    legend=False # Suppress duplicate legend
)
ax_contra.set_xlabel('Trial Time (s)', fontsize=12)
ax_contra.set_ylabel('Contra RT+HAT Frequency (Hz)', fontsize=12, fontweight='bold')
ax_contra.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_omr.svg', 
    bbox_inches='tight',
    transparent=True 
)
plt.show()


### RATIO

counts['asymmetry_index'] = (counts['rt_ipsi_hz'] - counts['rt_contra_hz']) / (counts['rt_ipsi_hz'] + counts['rt_contra_hz'])

fig = plt.figure(figsize=(12, 10))
ax = fig.gca()

sns.lineplot(
    data=counts, x='time_sec', y='asymmetry_index',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax, zorder=3,
)
ax.set_xlabel('Trial Time (s)', fontsize=12)
ax.set_ylabel('Asymmetry Index\n(← contra | ipsi →)', fontsize=12, fontweight='bold')
ax.set_ylim(-1.05, 1.05)
ax.grid(True, linestyle=':', alpha=0.5)
ax.axhline(0.0, color='grey', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_omr_index.svg', 
    bbox_inches='tight',
    transparent=True 
)
plt.show()