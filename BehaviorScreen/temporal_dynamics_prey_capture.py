from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short
jt_index = bouts_category_name_short.index("JT")
rt_index = bouts_category_name_short.index("RT")
hat_index = bouts_category_name_short.index("HAT")

ROOT = Path('/media/martin/DATA_18TB/Screen')
df = pd.read_csv(ROOT / 'bouts_control.csv')

print(f"#fish: {len(df.file.unique())}")

fine_dt = 0.02
time_bins = np.arange(0, 25 + fine_dt, fine_dt)
window_size_steps = int(1.0 / fine_dt)


pc_df = df[df['stim'] == Stim.PREY_CAPTURE].copy()

pc_df['ipsi_jturn'] = (pc_df['category'] == jt_index) & (pc_df['laterality'] == Laterality.IPSILATERAL)
pc_df['contra_jturn'] = (pc_df['category'] == jt_index) & (pc_df['laterality'] == Laterality.CONTRALATERAL)
pc_df['ipsi_rt'] = ((pc_df['category'] == rt_index) | (pc_df['category'] == hat_index)) & (pc_df['laterality'] == Laterality.IPSILATERAL)
pc_df['contra_rt'] = ((pc_df['category'] == rt_index) | (pc_df['category'] == hat_index)) & (pc_df['laterality'] == Laterality.CONTRALATERAL)

# Transform circular phases to Cartesian vectors to calculate valid means later
pc_df['phase_sin'] = np.sin(pc_df['stim_phase'])
pc_df['phase_cos'] = np.cos(pc_df['stim_phase'])

# Bin the raw continuous timestamps
pc_df['time_bin'] = pd.cut(pc_df['trial_time'], bins=time_bins, right=False)


# We leave the data inside the MultiIndex structure (no .reset_index() yet) 
# to protect the execution order of the upcoming rolling window.
counts = (
    pc_df.groupby(['file', 'trial_num', 'time_bin'], observed=False)
    .agg(
        jt_ipsi_count=('ipsi_jturn', 'sum'),
        jt_contra_count=('contra_jturn', 'sum'),
        rt_ipsi_count=('ipsi_rt', 'sum'),
        rt_contra_count=('contra_rt', 'sum'),
        mean_sin=('phase_sin', 'mean'),
        mean_cos=('phase_cos', 'mean')
    )
)

# Apply centered rolling sum across the timeline per fish/trial
counts['rolling_jt_ipsi'] = (
    counts.groupby(['file', 'trial_num'])['jt_ipsi_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

counts['rolling_jt_contra'] = (
    counts.groupby(['file', 'trial_num'])['jt_contra_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

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
counts['jt_ipsi_hz'] = counts['rolling_jt_ipsi'] / 1.0
counts['jt_contra_hz'] = counts['rolling_jt_contra'] / 1.0
counts['rt_ipsi_hz'] = counts['rolling_rt_ipsi'] / 1.0
counts['rt_contra_hz'] = counts['rolling_rt_contra'] / 1.0

# Reconstruct the circular mean phase angle using the arctangent of vector components
counts['avg_phase'] = np.arctan2(counts['mean_sin'], counts['mean_cos']) % (2 * np.pi)


def overlay_phase_axis(parent_ax):
    """Helper function to cleanly superimpose the phase trace behind behavioral data."""
    twin_ax = parent_ax.twinx()
    sns.lineplot(
        data=counts, x='time_sec', y='avg_phase',
        color='crimson', linewidth=1.2, linestyle='--', alpha=0.4,
        errorbar=None, ax=twin_ax
    )
    twin_ax.set_ylabel('Average Stim Phase (rad)', color='crimson', fontsize=11)
    twin_ax.tick_params(axis='y', labelcolor='crimson')
    twin_ax.set_ylim(0, 2 * np.pi)
    twin_ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    twin_ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    return twin_ax


fig, (ax_ipsi, ax_contra) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, sharey=True)

# --- PANEL 1: IPSILATERAL ---
ax_ipsi_twin = overlay_phase_axis(ax_ipsi)
sns.lineplot(
    data=counts, x='time_sec', y='jt_ipsi_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_ipsi, zorder=3
)
ax_ipsi.set_ylabel('Ipsi J-turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_ipsi.grid(True, linestyle=':', alpha=0.5)

# --- PANEL 2: CONTRALATERAL ---
ax_contra_twin = overlay_phase_axis(ax_contra)
sns.lineplot(
    data=counts, x='time_sec', y='jt_contra_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_contra, zorder=3,
)
ax_contra.set_xlabel('Trial Time (s)', fontsize=12)
ax_contra.set_ylabel('Contra J-turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_contra.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_JT.svg', 
    bbox_inches='tight',
    transparent=True 
)
plt.show()



fig, (ax_ipsi, ax_contra) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, sharey=True)

ax_ipsi_twin = overlay_phase_axis(ax_ipsi)
sns.lineplot(
    data=counts, x='time_sec', y='rt_ipsi_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_ipsi, zorder=3
)
ax_ipsi.set_ylabel('Ipsi RT+HAT turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_ipsi.grid(True, linestyle=':', alpha=0.5)

ax_contra_twin = overlay_phase_axis(ax_contra)
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
    'temporal_dynamics_prey_capture_RT_HAT.svg', 
    bbox_inches='tight',
    transparent=True 
)
plt.show()