from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from megabouts.utils import bouts_category_name_short

# TODO t-stats is maybe not the right stat for count data
# also reconsider cohen's d ?


capture_strikes = ['LCS_L','LCS_R','SCS_L','SCS_R']
sides = ['L', 'R']
row_names = [f"{cat}_{str(side)}" for cat in bouts_category_name_short for side in sides]

def compute_t_and_d(group_a, group_b):

    m_a, m_b = np.nanmean(group_a, axis=0), np.nanmean(group_b, axis=0)
    v_a, v_b = np.nanvar(group_a, axis=0, ddof=1), np.nanvar(group_b, axis=0, ddof=1)
    na, nb = len(group_a), len(group_b)

    pooled_var = ((na - 1) * v_a + (nb - 1) * v_b) / (na + nb - 2)
    welch_var = (v_a / na) + (v_b / nb)
    zero_var = welch_var <= 1e-15 
    
    se_welch = np.sqrt(welch_var)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = (m_b - m_a) / se_welch
    t_stat[zero_var] = 0  
    
    pooled_std = np.sqrt(pooled_var)
    with np.errstate(divide='ignore', invalid='ignore'):
        cohen_d = (m_b - m_a) / pooled_std
    cohen_d[zero_var] = 0
    
    return t_stat, cohen_d, zero_var

# TODO use scipy.stats.permutation_test ?
def permutation_analysis(a, b, n_perm=5000, alpha=0.05, rng=None):

    rng = np.random.default_rng(rng)
    obs_t, obs_d, zero_mask = compute_t_and_d(a, b)
    
    combined = np.concatenate([a, b], axis=0)
    n_a = len(a)
    
    count_geq = np.zeros(obs_t.shape)
    
    for _ in range(n_perm):
        shuffled = rng.permutation(combined, axis=0)
        perm_t, _, _ = compute_t_and_d(shuffled[:n_a], shuffled[n_a:])
        count_geq += (np.abs(perm_t) >= np.abs(obs_t))
        
    # Calculate raw P-values (Phipson & Smyth correction)
    p_values = (count_geq + 1) / (n_perm + 1)
    p_shape = p_values.shape
    p_flat = p_values.ravel()
    mask_flat = zero_mask.ravel()
    
    testable_indices = np.where(~mask_flat)[0]
    p_corrected_flat = np.full(p_flat.shape, 1.0)
    
    if len(testable_indices) > 0:
        _, p_corr_subset, _, _ = multipletests(
            p_flat[testable_indices], 
            alpha=alpha, 
            method='fdr_bh'
        )
        p_corrected_flat[testable_indices] = p_corr_subset
    
    return obs_d, p_corrected_flat.reshape(p_shape)


def load_bouts(file):

    with np.load(file, allow_pickle=True) as data:
        fish_names = data["labels_0"]
        trial_labels = data["labels_1"]
        bin_names = data["labels_2"]
        bout_categories = data["labels_3"]
        sides = data["labels_4"]
        bout_frequency = data["bout_frequency"]

        bout_frequency_interleaved = bout_frequency.reshape(*bout_frequency.shape[:-2], -1)
        trial_avg = np.nanmean(bout_frequency_interleaved, axis=1)

    return trial_avg, bin_names


def plot_heatmap(
        ref,
        exp,
        effect_size, 
        mask,
        title,
        y_labels,
        x_labels,
        clim: Tuple[float, float] = (0, 0.35)
    ):
    # Create 3 vertically stacked subplots
    # Increased height (30) to accommodate three large heatmaps
    fig, axes = plt.subplots(3, 1, figsize=(24, 26), sharex=True)
    
    # 1. Plot Reference
    im0 = axes[0].imshow(ref, aspect='auto', cmap='inferno')
    im0.set_clim(*clim)
    axes[0].set_title(f"{title} - Reference")
    fig.colorbar(im0, ax=axes[0], label="Bout Frequency")
    asterisk_y, asterisk_x = np.where(mask)
    axes[0].scatter(asterisk_x, asterisk_y, s=8, color='lime', marker='o', zorder=2)
    
    # 2. Plot Experimental (Comp)
    im1 = axes[1].imshow(exp, aspect='auto', cmap='inferno')
    im1.set_clim(*clim)
    axes[1].set_title(f"{title} - Experimental")
    fig.colorbar(im1, ax=axes[1], label="Bout Frequency")
    
    # 3. Plot Effect Size
    im2 = axes[2].imshow(effect_size, aspect='auto', cmap='bwr')
    im2.set_clim(-3, 3)
    axes[2].set_title(f"{title} - Effect Size (Cohen's d)")
    fig.colorbar(im2, ax=axes[2], label="Effect size (Cohen's d)")
    
    # Formatting across all axes
    for i, ax in enumerate(axes):
        # Y-axis labels for every plot
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("bout category")
        
        # X-axis labels (only rotate and show for the bottom plot to save space)
        ax.set_xticks(range(len(x_labels)))
        if i == 2:
            ax.set_xticklabels(x_labels, rotation=90, ha='center')
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":

        
    ROOT = Path('/home/martin/Desktop/DATA')
    ROOT = Path('/media/martin/DATA/Behavioral_screen/DATA/Screen')
    ROOT = Path('/media/martin/DATA_18TB/Screen')

    alpha = 0.05
    effect_size_thresh = 0.5

    comparisons = {
        ROOT / 'WT/ronidazole/bouts.npz': [
            p for f in ROOT.iterdir() 
            if f.is_dir() 
            for p in [f / 'ronidazole/bouts.npz'] 
            if p.exists()
        ] + [ ROOT / 'WT/danieau/bouts.npz'],
        ROOT / 'WT/danieau/bouts.npz': [
            p for f in ROOT.iterdir() 
            if f.is_dir()
            for p in [f / 'vehicle/bouts.npz', f / 'danieau/bouts.npz']
            if p.exists()
        ],
    }

    keep = [i for i,r in enumerate(row_names) if r not in capture_strikes]
    bouts_cat = [r for r in row_names if r not in capture_strikes]

    for ref, comp_list in comparisons.items():

        ref_trial_avg, bin_names = load_bouts(ref)
        ref_trial_avg = ref_trial_avg[...,keep]
        ref_fish_trial_avg = np.nanmean(ref_trial_avg, axis=0).T
        
        for p in comp_list:
        
            exp_trial_avg, _ = load_bouts(p)
            exp_trial_avg = exp_trial_avg[...,keep]
            exp_fish_trial_avg = np.nanmean(exp_trial_avg, axis=0).T

            d_map, p_map = permutation_analysis(ref_trial_avg, exp_trial_avg)
            effect_sz = d_map.T
            mask = (p_map.T < alpha) & (abs(effect_sz) > effect_size_thresh)

            title = f"{p.relative_to(ROOT).parent}-{ref.relative_to(ROOT).parent}".replace('/','_')
            plot_heatmap(ref_fish_trial_avg, exp_fish_trial_avg, effect_sz, mask, title, bouts_cat, bin_names)
            plt.savefig(p.parent / f"{title}_alpha_{alpha}.png")
            plt.close()