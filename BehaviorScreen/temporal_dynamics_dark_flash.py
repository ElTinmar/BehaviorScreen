from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short

o_index = bouts_category_name_short.index("O")

ROOT = Path('/media/martin/DATA_18TB/Screen')
df = pd.read_csv(ROOT / 'bouts_control.csv')


print(f"#fish: {len(df.file.unique())}")

fine_dt = 0.02
time_bins = np.arange(0, 5 + fine_dt, fine_dt)
window_size_steps = int(1.0 / fine_dt)

sub_df = df[df['epoch_name'] == "flash dark"].copy()

sub_df['o-bend'] = (sub_df['category'] == o_index)
sub_df['time_bin'] = pd.cut(sub_df['trial_time'], bins=time_bins, right=False)


# We leave the data inside the MultiIndex structure (no .reset_index() yet) 
# to protect the execution order of the upcoming rolling window.
counts = (
    sub_df.groupby(['file', 'trial_num', 'time_bin'], observed=False)
    .agg(
        o_bend_count=('o-bend', 'sum'),
    )
)

# Apply centered rolling sum across the timeline per fish/trial
counts['rolling_o-bend'] = (
    counts.groupby(['file', 'trial_num'])['o_bend_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

# Flatten the MultiIndex now that the time-sensitive rolling math is finished
counts = counts.reset_index()

# Convert the interval objects into floating-point seconds for the X-axis
counts['time_sec'] = counts['time_bin'].apply(lambda x: x.left).astype(float)

# Calculate true instantaneous frequency (Bouts per second)
counts['o-bend_hz'] = counts['rolling_o-bend'] / 1.0

fig = plt.figure(figsize=(12, 10))
ax = fig.gca()

sns.lineplot(
    data=counts, x='time_sec', y='o-bend_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax, zorder=3
)
ax.set_ylabel('O-bend Frequency (Hz)', fontsize=12, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_dark_flash.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_dark_flash.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

