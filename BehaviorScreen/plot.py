import pandas as pd
import numpy as np

bouts = pd.read_csv(
    "bouts.csv",
    converters={
        "stim_variable_value": lambda x: str(x),
    }
)

# filtering outliers
bouts[bouts['distance_center']>9] = np.nan # remove bouts on the edge
bouts.loc[bouts['distance']> 20, 'distance'] = np.nan
bouts.loc[bouts['peak_axial_speed']> 300, 'peak_axial_speed'] = np.nan

time_bins = [
    (0, 2.5),
    (2.5, 5),
    (5, 7.5),
    (7.5, 10),
    (10, 15),
    (15, 20),
    (20, 30),
]

# TODO maybe split that in separate functions for prey capture, OKR ...  and write a function to merge them all together?

def get_bouts_heatmap(
    bouts: pd.DataFrame,
    bouts_category_name_short: List[str] = bouts_category_name_short,
    time_bins: List[Tuple[float, float]] = time_bins,
) -> pd.DataFrame:

    stimuli = {
        Stim.PREY_CAPTURE: ['-20', '20'],
        Stim.PHOTOTAXIS: ['-1','1'],
        Stim.OMR: ['-90', '90', '0'],
        Stim.OKR: ['-36', '36'],
        Stim.LOOMING: ['-3', '3']
    }
    
    sides = ['L', 'R']
    num_cat = len(bouts_category_name_short)
    row_labels = [f"{cat}_{side}" for cat in bouts_category_name_short for side in sides]

    heatmap_df = pd.DataFrame()

    epochs = {}

    # DARK
    for start, stop in time_bins:
        name = f"DARK_{start}-{stop}s"
        epochs[name] = (
            (bouts.stim == Stim.DARK) &
            (bouts.trial_num >= 10) &
            (bouts.trial_num < 20) &
            (bouts.trial_time >= start) &
            (bouts.trial_time <= stop)
        )
        
    # BRIGHT
    for start, stop in time_bins:
        name = f"BRIGHT_{start}-{stop}s"
        epochs[name] = (
            (bouts.stim == Stim.BRIGHT) &
            (bouts.stim_variable_value == '[0.2, 0.2, 0.0, 1.0]') &
            (bouts.trial_num >= 5) &
            (bouts.trial_time >= start) &
            (bouts.trial_time <= stop)
        )

    for stim, param_list in stimuli.items():
        for start, stop in time_bins:
            for p in param_list:

                if stim in {Stim.OKR, Stim.OMR, Stim.LOOMING} and start >= 10:
                    continue
                if stim is Stim.PHOTOTAXIS and start >= 5:
                    continue

                name = f"{stim.name}_{p}_{start}-{stop}s"

                epochs[name] = (
                    (bouts.stim == stim) &
                    (bouts.stim_variable_value == p) &
                    (bouts.trial_time >= start) &
                    (bouts.trial_time <= stop)
                )

    # O-BEND
    for start, stop in time_bins:
        if start >= 5:
            break

        name = f"BRIGHT->DARK_{start}-{stop}s"
        epochs[name] = (
            (bouts.stim == Stim.DARK) &
            (bouts.trial_num >= 20) &
            (bouts.trial_num < 25) &
            (bouts.trial_time >= start) &
            (bouts.trial_time <= stop)
        )

    for name, mask in epochs.items():

        df_sub = bouts[
            mask &
            (bouts.proba > 0.5) &
            (bouts.distance_center < 15)
        ]

        counts = []
        for cat in range(num_cat):
            left = df_sub[(df_sub.category == cat) & (df_sub.sign == -1)].shape[0]
            right = df_sub[(df_sub.category == cat) & (df_sub.sign == 1)].shape[0]
            counts.extend([left, right])

        counts = pd.Series(counts, index=row_labels)
        if counts.sum() > 0:
            counts /= counts.sum()

        heatmap_df[name] = counts
    
    return heatmap_df


def plot_bout_heatmap(fig, ax, heatmap_df) -> None:

    im = ax.imshow(heatmap_df, aspect='auto', cmap='inferno')
    fig.colorbar(im, ax=ax, label='prob.')

    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=90, ha='center')

    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Category")


heatmap_df = get_bouts_heatmap(bouts)
fig = plt.figure(figsize=(20, 8))
ax = fig.gca()
plot_bout_heatmap(fig, ax, heatmap_df)
fig.tight_layout()
plt.savefig("categories_vs_time.png")
plt.show()
