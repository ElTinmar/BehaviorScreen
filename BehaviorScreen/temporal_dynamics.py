from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Tuple
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short

possible_roots = [
    Path('/home/martin/Desktop/DATA'),
    Path('/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen'),
    Path('/media/martin/DATA_18TB/Screen'),
]
ROOT = next((p for p in possible_roots if p.exists()), possible_roots[0])

dt = 0.02
t_start = 0.0 
t_end = 24.0  
window_duration = 0.33

# prey capture stim parameters
prey_stim_speed_deg_per_s = 90
prey_stim_range_deg = 2*70
prey_stim_freq =  prey_stim_speed_deg_per_s / prey_stim_range_deg

def extract_dataframe(
        root: Path, 
        csv_file: str, 
        stim: Stim,
        dt: float = 0.02,
        t_start: float = 0.0,
        t_end: float = 25.0,
        window_duration: float = 0.33
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    csv = root / csv_file
    df = pd.read_csv(csv)
    sub_df = df[df['stim'] == stim].copy()

    time_bins = np.arange(t_start, t_end + dt, dt)
    window_size_steps = int(window_duration / dt)
    window_size_steps |= 1

    for bout_index, bout_name in enumerate(bouts_category_name_short):
        for laterality in Laterality:
            sub_df[f"{laterality}_{bout_name}"] = (sub_df['category'] == bout_index) & (sub_df['laterality'] == laterality)

    agg_dict = {
        f"count_{laterality}_{bout_name}": (f"{laterality}_{bout_name}", 'sum')
        for bout_name in bouts_category_name_short
        for laterality in Laterality
    }

    sub_df['time_bin'] = pd.cut(sub_df['trial_time'], bins=time_bins, right=False)

    # PC
    sub_df['phase_sin'] = np.sin(sub_df['stim_phase'])
    sub_df['phase_cos'] = np.cos(sub_df['stim_phase'])
    sub_df['phase_sin2'] = np.sin(2.0 * sub_df['stim_phase'])
    sub_df['phase_cos2'] = np.cos(2.0 * sub_df['stim_phase'])

    agg_dict.update({
        'mean_sin': ('phase_sin', 'mean'),
        'mean_cos': ('phase_cos', 'mean'),
        'mean_sin2': ('phase_sin2', 'mean'),
        'mean_cos2': ('phase_cos2', 'mean'),
    })

    # LOOMINGS
    # TODO add later
    
    # groupby 
    counts = (
        sub_df.groupby(['file', 'trial_num', 'time_bin'], observed=False)
        .agg(**agg_dict)
    )

    count_cols = [f"count_{lat}_{b}" for b in bouts_category_name_short for lat in Laterality]
    rolling_cols = [f"rolling_{lat}_{b}" for b in bouts_category_name_short for lat in Laterality]
    hz_cols = [f"{lat}_{b}_hz" for b in bouts_category_name_short for lat in Laterality]

    counts[rolling_cols] = (
        counts.groupby(level=['file', 'trial_num'])[count_cols]
        .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
    )

    counts = counts.reset_index()

    counts[hz_cols] = counts[rolling_cols] / window_duration

    counts['time_sec'] = counts['time_bin'].apply(lambda x: x.left).astype(float)
    counts['avg_phase'] = np.arctan2(counts['mean_sin'], counts['mean_cos']) % (2 * np.pi)

    return sub_df, counts

#counts['asymmetry_index_JT'] = (counts['jt_ipsi_hz'] - counts['jt_contra_hz']) / (counts['jt_ipsi_hz'] + counts['jt_contra_hz'])

data, counts = extract_dataframe(
    ROOT,
    'bouts_control.csv',
    Stim.PREY_CAPTURE,
    dt = 0.02,
    t_start = 0.0, 
    t_end = 24.0,
    window_duration = 0.33
)

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

ax_ipsi_twin = overlay_phase_axis(ax_ipsi)
sns.lineplot(
    data=counts, x='time_sec', y='IPSILATERAL_JT_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_ipsi, zorder=3
)
ax_ipsi.set_ylabel('Ipsi J-turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_ipsi.grid(True, linestyle=':', alpha=0.5)

ax_contra_twin = overlay_phase_axis(ax_contra)
sns.lineplot(
    data=counts, x='time_sec', y='CONTRALATERAL_JT_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_contra, zorder=3,
)
ax_contra.set_xlabel('Trial Time (s)', fontsize=12)
ax_contra.set_ylabel('Contra J-turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_contra.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_JT.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_JT.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

#################################################

mask = data['IPSILATERAL_JT'] & (data['trial_time'] >= t_start) & (data['trial_time'] <= t_end)
events = data[mask]

event_times = (events['trial_time'] - t_start).values
event_trials = events['trial_num'].values
event_phase_sin = events['phase_sin'].values
event_phase_cos = events['phase_cos'].values
event_phase_sin2 = events['phase_sin2'].values
event_phase_cos2 = events['phase_cos2'].values

# grid for integration
t_grid = np.arange(t_start, t_end+dt, dt)

unique_trials = np.sort(counts['trial_num'].unique())
num_fish = len(counts['file'].unique())

def lambda_poisson(t, trial, params, stim_freq=0.642857):

    A, tau, k, B, b1, b2, b3, b4, alpha_A, alpha_B, alpha_gamma = params

    w = 2.0 * np.pi * stim_freq
    phase = w * t
    
    transient = A * (t ** k) * np.exp(-t / tau) * np.exp(alpha_A * trial)
    baseline = B * np.exp(alpha_B * trial)
    phase_ripple = (
        b1 * np.sin(phase) + b2 * np.cos(phase) + 
        b3 * np.sin(2.0 * phase) + b4 * np.cos(2.0 * phase)
    ) * np.exp(alpha_gamma * trial)

    return transient + baseline + phase_ripple

def poisson_nll(params, lambda_func, t_events, trial_events, unique_trials, t_grid, n_fish=1.0):
    """
    Negative Log-Likelihood for Inhomogeneous Poisson Processes
    
    Parameters
    ----------
    params : array-like
        Vector of parameters to optimize.
    lambda_func : callable
        Signature: lambda_func(t, trial, params)
        Must support NumPy array broadcasting.
    t_events : np.ndarray
        1D array of event timestamps.
    trial_events : np.ndarray
        1D array of trial IDs for each event.
    unique_trials : np.ndarray
        1D array of all unique trial indices in the experiment.
    t_grid : np.ndarray
        1D array specifying the uniform integration time grid [t_start, t_end].
    n_fish : float
        Multiplicative scale factor for number of subjects.
    """
    # -------------------------------------------------------------
    # Term 1: Sum of Log-Rates at Observed Event Times
    # -------------------------------------------------------------
    event_rates = lambda_func(t_events, trial_events, params)
    
    # Consistent non-negativity floor
    event_rates = np.maximum(event_rates, 1e-9)
    sum_log_rates = np.sum(np.log(event_rates))
    
    # -------------------------------------------------------------
    # Term 2: Numerical Integration Across (Trial x Time) Space
    # -------------------------------------------------------------
    # Broadcast t_grid (1, N_time) and unique_trials (N_trials, 1) to form 2D mesh
    t_2d = t_grid[None, :]          # Shape: (1, N_time)
    trials_2d = unique_trials[:, None] # Shape: (N_trials, 1)
    
    # Evaluate 2D rate surface: shape (N_trials, N_time)
    rate_surface = lambda_func(t_2d, trials_2d, params)
    rate_surface = np.maximum(rate_surface, 1e-9)
    
    # Integrate across the time axis (axis=1) using Simpson's Rule
    trial_integrals = simpson(rate_surface, x=t_grid, axis=1)
    
    # Sum total expected events across all trials and scale by n_fish
    total_expected_events = np.sum(trial_integrals) * n_fish
    
    return -(sum_log_rates - total_expected_events)


initial_guesses = [0.56, 1.15, 2.0, 0.40, 0.0, 0.0, 0.0, 0.0, -0.05, -0.05, -0.05]
bounds = (
    (0.01, 10.0),    # A
    (0.1, 5.0),      # tau
    (0.4, 10.0),     # k 
    (0.01, 5.0),     # B
    (-5.0, 5.0),     # b1 (Fundamental Sin)
    (-5.0, 5.0),     # b2 (Fundamental Cos)
    (-5.0, 5.0),     # b3 (Second-Harmonic Sin)
    (-5.0, 5.0),     # b4 (Second-Harmonic Cos)
    (-2.0, 2.0),     # alpha_A
    (-2.0, 2.0),     # alpha_B
    (-2.0, 2.0)      # alpha_gamma
)


poisson_fit = minimize(
    poisson_nll,
    x0=initial_guesses,
    args=(
        lambda_poisson, 
        event_times, 
        event_trials, 
        unique_trials, 
        t_grid, 
        num_fish
    ),
    method='L-BFGS-B',
    bounds=bounds
)


# Extract parameters
A_p, tau_p, k_p, B_p, b1_p, b2_p, b3_p, b4_p, a_A, a_B, a_g = poisson_fit.x

# ==========================================
# 4. DECODING THE TWO PHASES
# ==========================================
# To find the two peak phase angles, evaluate the phase-only equation over 0 to 2*pi
phi_eval = np.linspace(0, 2 * np.pi, 360)
ripple_eval = (
    b1_p * np.sin(phi_eval) + b2_p * np.cos(phi_eval) + 
    b3_p * np.sin(2.0 * phi_eval) + b4_p * np.cos(2.0 * phi_eval)
)

# Find localized peak indices
peak_indices, _ = find_peaks(ripple_eval)
peak_phases = phi_eval[peak_indices]

print(f"Detected {len(peak_phases)} phase peaks in the cycle:")
for i, phi in enumerate(peak_phases):
    deg = np.degrees(phi)
    # Convert phase angle back to physical coordinate (e.g., degrees or mm)
    physical_val = 20.0 + 70.0 * ((1.0 - np.cos(phi)) / 2.0)
    direction = "N->T" if np.sin(phi) > 0 else "T->N"
    print(f"Peak {i+1}: Phase = {deg:.1f}° | Physical Value = {physical_val:.2f}, direction: {direction}")

peak_str_1 = "N/A"
peak_str_2 = "N/A"

if len(peak_phases) >= 1:
    p1_deg = np.degrees(peak_phases[0])
    p1_phys = 20.0 + 70.0 * ((1.0 - np.cos(peak_phases[0])) / 2.0)
    peak_str_1 = f"{p1_deg:.1f}° ({p1_phys:.1f} °) " + ("N->T" if np.sin(peak_phases[0]) > 0 else "T->N")
if len(peak_phases) >= 2:
    p2_deg = np.degrees(peak_phases[1])
    p2_phys = 20.0 + 70.0 * ((1.0 - np.cos(peak_phases[1])) / 2.0)
    peak_str_2 = f"{p2_deg:.1f}° ({p2_phys:.1f} °) " + ("N->T" if np.sin(peak_phases[1]) > 0 else "T->N")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [2.8, 1.4]})

select_trials = [unique_trials[0], unique_trials[len(unique_trials)//2], unique_trials[-1]]
colors = ['#1f77b4', '#d62728', '#2ca02c']

for tm, col in zip(select_trials, colors):
    trial_data = counts[counts['trial_num'] == tm]
    mean_data = trial_data.groupby('time_sec').agg(
        r_smooth=('IPSILATERAL_JT_hz', 'mean')
    )
    w_mask = (mean_data.index >= t_start) & (mean_data.index <= t_end)
    t_smooth = mean_data.index[w_mask] - t_start
    r_model = lambda_poisson(t_smooth, tm, poisson_fit.x)
    
    ax1.plot(t_smooth, mean_data.loc[w_mask, 'r_smooth'].values, '--', color=col, alpha=0.35)
    ax1.plot(t_smooth, r_model, '-', color=col, lw=2.5, label=f'Trial {tm} Model')

ax1.set_ylabel('Bout Rate $\lambda(t, m)$  (Hz)')
ax1.set_xlabel('Time (s)')

# --- TITLE UPDATED TO REFLECT DETAILED FOURIER SECOND-HARMONIC EXPANSION ---
ax1.set_title(r"$\lambda(t, m) = \max\left( A e^{\alpha_A \cdot m} t^k e^{-t/\tau} + B e^{\alpha_B \cdot m} + [b_1 \sin(\phi) + b_2 \cos(\phi) + b_3 \sin(2\phi) + b_4 \cos(2\phi)] e^{\alpha_\gamma \cdot m},\, 10^{-9}\right)$", fontsize=10.5)

ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_ylim(bottom=0)

overlay_phase_axis(ax1)

ax2.axis('off')

# --- ANNOTATION UPDATED TO MAP BI-MODAL WEIGHTS AND LOGISTICAL PEAKS ---
latex_annotation = r"""$\text{Peak Time} = \tau \cdot k$ [Transient Delay]  |  Second Harmonic Fourier Expansion Profile (Bi-Modal Preference Tuning)

Fitted Parameters & Baseline Variables:
$m$ = trial number (ordinal tracking index of consecutive stimulus exposures)
$A$ = """ + f"{A_p:.3f}" + r""" Hz [Transient peak height]  |  $B$ = """ + f"{B_p:.3f}" + r""" Hz [Stable background floor]
$\tau$ = """ + f"{tau_p:.3f}" + r""" sec  |  $k$ = """ + f"{k_p:.3f}" + r"""  |  Kinetic Peak Time = """ + f"{tau_p * k_p:.3f}" + r""" sec

Fourier Coefficients:
1st Harmonic (Fundamental):  $b_1$ = """ + f"{b1_p:.3f}" + r""", $b_2$ = """ + f"{b2_p:.3f}" + r"""
2nd Harmonic (Octave):       $b_3$ = """ + f"{b3_p:.3f}" + r""", $b_4$ = """ + f"{b4_p:.3f}" + r"""

Decoded Spatial Response Maxima:
Detected Peak 1 Angle/Position = """ + peak_str_1 + r"""
Detected Peak 2 Angle/Position = """ + peak_str_2 + r"""

Trial-by-Trial Modulators:
$\alpha_A$ = """ + f"{a_A:.4f}" + r""" [Sensory Adaptation: Habituation/sensitization rate of explosive stimulus peak]
$\alpha_B$ = """ + f"{a_B:.4f}" + r""" [Baseline Motivational Shift: Continuous background driving baseline drift]
$\alpha_{\gamma}$ = """ + f"{a_g:.4f}" + r""" [Entrainment Tuning Change: Systemic modulation of spatial phase tuning power over time]"""

ax2.text(0.01, 0.95, latex_annotation, transform=ax2.transAxes, fontsize=9.5, verticalalignment='top', horizontalalignment='left', multialignment='left')
plt.tight_layout()
plt.savefig('temporal_dynamics_prey_capture_bimodal_NHpsi_2.svg', bbox_inches='tight')
plt.savefig('temporal_dynamics_prey_capture_bimodal_NHpsi_2.png', dpi=300, bbox_inches='tight')
plt.show()

