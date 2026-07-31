from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Tuple
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import ks_1samp, uniform, kstest, chi2
import matplotlib.pyplot as plt
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short

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


ROOT = Path('/media/martin/DATA_18TB/Screen')
ROOT = Path('/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen')
ROOT = Path('/home/martin/Desktop/DATA')
df = pd.read_csv(ROOT / 'bouts_control.csv')

print(f"#fish: {len(df.file.unique())}")

data, counts = extract_dataframe(
    ROOT,
    'bouts_control.csv',
    Stim.PREY_CAPTURE
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

# --- PANEL 1: IPSILATERAL ---
ax_ipsi_twin = overlay_phase_axis(ax_ipsi)
sns.lineplot(
    data=counts, x='time_sec', y='IPSILATERAL_JT_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_ipsi, zorder=3
)
ax_ipsi.set_ylabel('Ipsi J-turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_ipsi.grid(True, linestyle=':', alpha=0.5)

# --- PANEL 2: CONTRALATERAL ---
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
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_RT_HAT.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

### RATIO


counts['asymmetry_index_JT'] = (counts['jt_ipsi_hz'] - counts['jt_contra_hz']) / (counts['jt_ipsi_hz'] + counts['jt_contra_hz'])
fig = plt.figure(figsize=(12, 10))
ax = fig.gca()

sns.lineplot(
    data=counts, x='time_sec', y='asymmetry_index_JT',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax, zorder=3,
)
ax.set_xlabel('Trial Time (s)', fontsize=12)
ax.set_ylabel('Asymmetry Index\n(← contra | ipsi →)', fontsize=12, fontweight='bold')
ax.set_ylim(-1.05, 1.05)
ax.grid(True, linestyle=':', alpha=0.5)
ax.axhline(0.0, color='grey', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_JT_index.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_JT_index.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

counts['asymmetry_index_RT'] = (counts['rt_ipsi_hz'] - counts['rt_contra_hz']) / (counts['rt_ipsi_hz'] + counts['rt_contra_hz'])
fig = plt.figure(figsize=(12, 10))
ax = fig.gca()

sns.lineplot(
    data=counts, x='time_sec', y='asymmetry_index_RT',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax, zorder=3,
)
ax.set_xlabel('Trial Time (s)', fontsize=12)
ax.set_ylabel('Asymmetry Index\n(← contra | ipsi →)', fontsize=12, fontweight='bold')
ax.set_ylim(-1.05, 1.05)
ax.grid(True, linestyle=':', alpha=0.5)
ax.axhline(0.0, color='grey', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_RT_HAT_index.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_RT_HAT_index.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

#################################################
#
#
#                   MODELS
#
#
################################################


T_start = 0.0 
T_end = 24.0  
T_duration = T_end - T_start

decay_events = pc_df[pc_df['ipsi_jturn'] & (pc_df['trial_time'] >= T_start) & (pc_df['trial_time'] <= T_end)]
spike_times = (decay_events['trial_time'] - T_start).values
spike_trials = decay_events['trial_num'].values
spike_sin = decay_events['phase_sin'].values
spike_cos = decay_events['phase_cos'].values
spike_sin2 = decay_events['phase_sin2'].values
spike_cos2 = decay_events['phase_cos2'].values

window_mask_bins = (counts['time_sec'] >= T_start) & (counts['time_sec'] <= T_end)
timeline_template = counts[window_mask_bins].groupby('time_sec').agg(
    bin_sin=('mean_sin', 'mean'),
    bin_cos=('mean_cos', 'mean'),
    bin_sin2=('mean_sin2', 'mean'),
    bin_cos2=('mean_cos2', 'mean')
).reset_index()

t_grid = timeline_template['time_sec'].values - T_start
grid_sin = timeline_template['bin_sin'].values
grid_cos = timeline_template['bin_cos'].values
grid_sin2 = timeline_template['bin_sin2'].values
grid_cos2 = timeline_template['bin_cos2'].values
dt = fine_dt 

num_trials = len(pc_df.groupby(['file', 'trial_num']))


######################### homogeneous poisson process  ########################

### IPSI


# Calculate analytical MLE constant rate (Total Events / Total Tracking Space)
n_events = len(spike_times)
total_time_per_trial = T_duration
total_operational_space = num_trials * total_time_per_trial

C_mle = n_events / total_operational_space

# Compute the exact baseline Negative Log-Likelihood
nll_hpp = -(n_events * np.log(np.maximum(C_mle, 1e-9)) - C_mle * total_operational_space)

print("--- Homogeneous Poisson Process (Null Model) ---")
print(f"Constant Baseline Rate (C): {C_mle:.3f} bouts/sec")
print(f"HPP Null Model NLL: {nll_hpp:.3f}")


# ==========================================
# 3. PLOT HOMOGENEOUS POISSON BASELINE DASHBOARD
# ==========================================
# Average rolling counts across all fish and trials to get the mean population curve
mean_rolling = counts.groupby('time_sec')['jt_ipsi_hz'].mean()

# Isolate the time points matching your decay window
t_smoothed_window = mean_rolling.index[(mean_rolling.index >= T_start) & (mean_rolling.index <= T_end)]
smoothed_rates_window = mean_rolling.loc[t_smoothed_window]

# Generate points for the flat, constant HPP baseline rate
t_model_hpp = np.linspace(0, T_duration, 200)
fitted_rate_hpp = np.full_like(t_model_hpp, C_mle)

# Create a 2-row layout matching your standard dashboard framework
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(10, 8), 
    gridspec_kw={'height_ratios': [3, 1.2]},
    sharex=False
)

# --- TOP ROW: DATA AXIS ---
ax1.plot(t_smoothed_window - T_start, smoothed_rates_window, color='darkgray', lw=2, label='Your Rolling Average Data')
ax1.plot(t_model_hpp, fitted_rate_hpp, color='royalblue', lw=3, ls='--', label='HPP Null Model Fit')
ax1.set_ylabel('Ipsilateral J-turn rate $\lambda(t)$  (Hz)')
ax1.set_xlabel('Time from Peak (seconds)')
ax1.set_title(r"$\lambda(t) = C$ (Constant Rate Assumption)", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_ylim((0, 1.2))  # Force y-axis to start at 0 to honestly show the flat baseline comparison

# --- BOTTOM ROW: ANNOTATION DASHBOARD AXIS ---
ax2.axis('off')

# Clean HPP summary notation string using safe raw-string concatenation
latex_annotation_hpp = r"""Null Model Formula: $\lambda(t) = C$

Fitted Parameters & Interpretations:
$C$ = """ + f"{C_mle:.3f}" + r""" Hz [Constant Baseline Rate: Represents the global average firing intensity across all time points]

Statistical Meaning:
This model assumes behavioral probability is completely independent of time, stimulus onset, or phase cycles. 
It serves as the formal baseline (Null Hypothesis) for your Likelihood Ratio Tests. Overturning this model 
proves that your experimental features possess statistically significant temporal structures."""

# Render the text directly inside the blank bottom panel
ax2.text(
    0.01, 0.95,                  
    latex_annotation_hpp, 
    transform=ax2.transAxes,
    fontsize=10, 
    verticalalignment='top',
    horizontalalignment='left',
    multialignment='left'
)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_model_HPP_ipsi.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_model_HPP_ipsi.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()


### CONTRA


decay_events = pc_df[pc_df['contra_jturn'] & (pc_df['trial_time'] >= T_start) & (pc_df['trial_time'] <= T_end)]
spike_times = (decay_events['trial_time'] - T_start).values
num_trials = len(pc_df.groupby(['file', 'trial_num']))

# Calculate analytical MLE constant rate (Total Events / Total Tracking Space)
n_events = len(spike_times)
total_time_per_trial = T_duration
total_operational_space = num_trials * total_time_per_trial

C_mle = n_events / total_operational_space

# Compute the exact baseline Negative Log-Likelihood
nll_hpp = -(n_events * np.log(np.maximum(C_mle, 1e-9)) - C_mle * total_operational_space)

print("--- Homogeneous Poisson Process (Null Model) ---")
print(f"Constant Baseline Rate (C): {C_mle:.3f} bouts/sec")
print(f"HPP Null Model NLL: {nll_hpp:.3f}")


# ==========================================
# 3. PLOT HOMOGENEOUS POISSON BASELINE DASHBOARD
# ==========================================
# Average rolling counts across all fish and trials to get the mean population curve
mean_rolling = counts.groupby('time_sec')['jt_contra_hz'].mean()

# Isolate the time points matching your decay window
t_smoothed_window = mean_rolling.index[(mean_rolling.index >= T_start) & (mean_rolling.index <= T_end)]
smoothed_rates_window = mean_rolling.loc[t_smoothed_window]

# Generate points for the flat, constant HPP baseline rate
t_model_hpp = np.linspace(0, T_duration, 200)
fitted_rate_hpp = np.full_like(t_model_hpp, C_mle)

# Create a 2-row layout matching your standard dashboard framework
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(10, 8), 
    gridspec_kw={'height_ratios': [3, 1.2]},
    sharex=False
)

# --- TOP ROW: DATA AXIS ---
ax1.plot(t_smoothed_window - T_start, smoothed_rates_window, color='darkgray', lw=2, label='Your Rolling Average Data')
ax1.plot(t_model_hpp, fitted_rate_hpp, color='royalblue', lw=3, ls='--', label='HPP Null Model Fit')
ax1.set_ylabel('Contralateral J-turn rate $\lambda(t)$  (Hz)')
ax1.set_xlabel('Time from Peak (seconds)')
ax1.set_title(r"$\lambda(t) = C$ (Constant Rate Assumption)", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_ylim((0, 1.2))  # Force y-axis to start at 0 to honestly show the flat baseline comparison

# --- BOTTOM ROW: ANNOTATION DASHBOARD AXIS ---
ax2.axis('off')

# Clean HPP summary notation string using safe raw-string concatenation
latex_annotation_hpp = r"""Null Model Formula: $\lambda(t) = C$

Fitted Parameters & Interpretations:
$C$ = """ + f"{C_mle:.3f}" + r""" Hz [Constant Baseline Rate: Represents the global average firing intensity across all time points]

Statistical Meaning:
This model assumes behavioral probability is completely independent of time, stimulus onset, or phase cycles. 
It serves as the formal baseline (Null Hypothesis) for your Likelihood Ratio Tests. Overturning this model 
proves that your experimental features possess statistically significant temporal structures."""

# Render the text directly inside the blank bottom panel
ax2.text(
    0.01, 0.95,                  
    latex_annotation_hpp, 
    transform=ax2.transAxes,
    fontsize=10, 
    verticalalignment='top',
    horizontalalignment='left',
    multialignment='left'
)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_model_HPP_contra.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_model_HPP_contra.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

######################## Non homogeneous poisson ##################################


##### Exponential
decay_events = pc_df[pc_df['ipsi_jturn'] & (pc_df['trial_time'] >= T_start) & (pc_df['trial_time'] <= T_end)]
spike_times = (decay_events['trial_time'] - T_start).values
num_trials = len(pc_df.groupby(['file', 'trial_num']))


def lambda_t(t, A, tau, B):
    # Bound below by a tiny number to avoid log(0)
    return np.maximum(A * np.exp(-t / tau) + B, 1e-9)

def negative_log_likelihood(params, t_events, T, N_trials):
    A, tau, B = params
    
    # Term 1: Sum of log-rates at the exact time of every event
    sum_log_rates = np.sum(np.log(lambda_t(t_events, A, tau, B)))
    
    # Term 2: Integral of expected events per trial * total trials
    # Integral of A*e^(-t/tau) + B from 0 to T
    integral = (-A * tau * np.exp(-T / tau) + B * T) - (-A * tau)

    #integral,_ = quad(lambda_t,0,T,args=(A, tau, B))
    
    return -(sum_log_rates - N_trials*integral)

initial_guesses = [2.0, 1.5, 0.1] 
bounds = ((0, None), (0.01, None), (0, None)) # Parameters must be positive

result = minimize(
    negative_log_likelihood, 
    x0=initial_guesses, 
    args=(spike_times, T_duration, num_trials),
    bounds=bounds
)

A_fit, tau_fit, B_fit = result.x

print("\n--- Fit Results ---")
print(f"Amplitude (A): {A_fit:.3f} bouts/sec")
print(f"Time Constant (tau): {tau_fit:.3f} seconds (or {tau_fit*1000:.1f} ms)")
print(f"Baseline (B): {B_fit:.3f} bouts/sec")

# ==========================================
# 3. PLOT AVERAGED PROPENSITY WITH DASHBOARD BELOW
# ==========================================
# Average rolling counts across all fish and trials to get the mean population curve
mean_rolling = counts.groupby('time_sec')['jt_ipsi_hz'].mean()


# Isolate the time points matching your decay window
t_smoothed_window = mean_rolling.index[(mean_rolling.index >= T_start) & (mean_rolling.index <= T_end)]
smoothed_rates_window = mean_rolling.loc[t_smoothed_window]

# Generate points for your parametric model fit
t_model = np.linspace(0, T_duration, 200)
fitted_rate = A_fit * np.exp(-t_model / tau_fit) + B_fit

# Create a 2-row layout. 
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(10, 8), 
    gridspec_kw={'height_ratios': [3, 1.2]},
    sharex=False
)

# --- TOP ROW: DATA AXIS ---
ax1.plot(t_smoothed_window - T_start, smoothed_rates_window, color='darkgray', lw=2, label='Your Rolling Average Data')
ax1.plot(t_model, fitted_rate, color='crimson', lw=3, label='Poisson MLE Fit')
ax1.set_ylabel('Bout Rate $\lambda(t)$  (Hz)')
ax1.set_xlabel('Time from Peak (seconds)')
ax1.set_title(r"$\lambda(t) = \max \left( A \cdot e^{-t/\tau} + B ,\, 10^{-9} \right)$", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_ylim(bottom=0)  # Force y-axis to start at 0

# --- BOTTOM ROW: ANNOTATION DASHBOARD AXIS ---
ax2.axis('off')

# Concise exponential summary annotation using safe raw-string concatenation
latex_annotation = r"""
Fitted Parameters & Interpretations:
$A$ = """ + f"{A_fit:.3f}" + r""" Hz [Scales the initial height of the behavioral response peak at $t=0$]
$B$ = """ + f"{B_fit:.3f}" + r""" Hz [The stable background baseline rate floor after the response decays]
$\tau$ = """ + f"{tau_fit:.3f}" + r""" sec (""" + f"{tau_fit*1000:.1f}" + r""" ms) [The characteristic time constant governing the rate of exponential decay]"""

# Render the text directly inside the blank bottom panel
ax2.text(
    0.01, 0.95,                  
    latex_annotation, 
    transform=ax2.transAxes,
    fontsize=10, 
    verticalalignment='top',
    horizontalalignment='left',
    multialignment='left'
)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_model_NHPP_ipsi_0.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_model_NHPP_ipsi_0.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

################## Include trial num in the model

# 1. Gather all events and match them to their respective trial number
behavior_mask = pc_df['ipsi_jturn'] == True
decay_events = pc_df[behavior_mask & (pc_df['trial_time'] >= T_start) & (pc_df['trial_time'] <= T_end)].copy()

# Store paired arrays of event timestamps and their trial indices
spike_times = (decay_events['trial_time'] - T_start).values
spike_trials = decay_events['trial_num'].values

# Get a listing of all valid trials present in your overall control dataframe
all_trials = pc_df.groupby(['file', 'trial_num']).size().reset_index()[['file', 'trial_num']]
unique_trial_counts = all_trials['trial_num'].values # Vector of every trial run in the experiment

# 2. Modify the model rate function to accept trial index vectors
def lambda_t_joint(t, trial_k, A, tau, B, alpha):
    base_rate = A * np.exp(-t / tau) + B
    trial_modulation = np.exp(alpha * trial_k)
    return np.maximum(base_rate * trial_modulation, 1e-9)

# 3. Define the updated combined negative log likelihood
def joint_negative_log_likelihood(params, t_events, trial_events, unique_trials, T):
    A, tau, B, alpha = params
    
    # Term 1: Rate evaluated for every event at its specific time and trial index
    sum_log_rates = np.sum(np.log(lambda_t_joint(t_events, trial_events, A, tau, B, alpha)))
    
    # Term 2: Summed integrals across every single trial run in the dataset
    # We must integrate each trial individually because the trial number modulates the total area
    base_integral = (-A * tau * np.exp(-T / tau) + B * T) - (-A * tau)
    total_expected_events = np.sum(base_integral * np.exp(alpha * unique_trials))
    
    return -(sum_log_rates - total_expected_events)

# 4. Fit the unified model [A, tau, B, alpha]
initial_guesses_joint = [2.0, 1.5, 0.1, 0.0] 
joint_bounds = ((0, None), (0.01, None), (0, None), (None, None))

joint_result = minimize(
    joint_negative_log_likelihood, 
    x0=initial_guesses_joint, 
    args=(spike_times, spike_trials, unique_trial_counts, T_duration),
    bounds=joint_bounds
)

A_j, tau_j, B_j, alpha_j = joint_result.x

print("\n--- Unified Model Results ---")
print(f"Base Amplitude (A): {A_j:.3f} bouts/sec")
print(f"Time Constant (tau): {tau_j:.3f} seconds")
print(f"Base Baseline (B): {B_j:.3f} bouts/sec")
print(f"Trial Modulation Index (alpha): {alpha_j:.4f}")

# maybe report A+B, tau and alpha?

unique_trials = np.sort(counts['trial_num'].unique())
num_colors = len(unique_trials)
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 0.85, num_colors)] 


plt.figure(figsize=(10, 6))
t_model = np.linspace(0, T_duration, 200)

for t_num, color in zip(unique_trials, colors):
    trial_data = counts[counts['trial_num'] == t_num]
    mean_rolling = trial_data.groupby('time_sec')['jt_ipsi_hz'].mean()
    
    window_mask = (mean_rolling.index >= T_start) & (mean_rolling.index <= T_end)
    t_smooth = mean_rolling.index[window_mask] - T_start
    r_smooth = mean_rolling.values[window_mask]
    
    r_model = (A_j * np.exp(-t_model / tau_j) + B_j) * np.exp(alpha_j * t_num)
    label = f'Trial {t_num}' if t_num in [unique_trials[0], unique_trials[num_colors//2], unique_trials[-1]] else ""
    
    plt.plot(t_smooth, r_smooth, '--', color=color, alpha=0.3, label=f'Data {label}' if label else "")
    plt.plot(t_model, r_model, '-', color=color, lw=2, label=f'Model {label}' if label else "")

plt.xlabel('Time from Peak Response (seconds)')
plt.ylabel('Bout Rate (Hz)')
plt.title(f'Unified Model Fit Across Trials (alpha = {alpha_j:.4f})')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()


### Add stimulus phase
# ==========================================
# 0. EXTRACT RAW EVENTS (THE SPIKE VARIABLES)
# ==========================================

T_start = 0.0 
T_end = 24.0  
T_duration = T_end - T_start

behavior_mask = pc_df['ipsi_jturn'] == True

# Filter for events occurring exactly within our decay window
decay_events = pc_df[behavior_mask & (pc_df['trial_time'] >= T_start) & (pc_df['trial_time'] <= T_end)].copy()

spike_times = (decay_events['trial_time'] - T_start).values
spike_trials = decay_events['trial_num'].values
spike_sin = decay_events['phase_sin'].values
spike_cos = decay_events['phase_cos'].values

# --- FIXED: Get unique trial numbers AND total number of fish/files ---
# We need the unique array of trial IDs (e.g., 0 to 20)
unique_trial_indices = np.sort(pc_df['trial_num'].unique())
# We need to know how many fish are pooled so we scale the expected events correctly
num_fish = len(pc_df['file'].unique()) 

# ==========================================
# 1. PREPARE THE INTEGRAL TEMPLATE FROM BIN DATA
# ==========================================
counts['time_sec'] = counts['time_bin'].apply(lambda x: x.left).astype(float)
window_mask_bins = (counts['time_sec'] >= T_start) & (counts['time_sec'] <= T_end)

# Get one unique row per time bin (averaged across trials/fish)
timeline_template = counts[window_mask_bins].groupby('time_sec').agg(
    bin_sin=('mean_sin', 'mean'),
    bin_cos=('mean_cos', 'mean')
).reset_index()

t_grid = timeline_template['time_sec'].values - T_start
grid_sin = timeline_template['bin_sin'].values
grid_cos = timeline_template['bin_cos'].values
dt = fine_dt 

# ==========================================
# 2. CORRECTED LOG-LIKELIHOOD FUNCTION
# ==========================================
def lambda_t_phase(t, trial_k, s_sin, s_cos, A, tau, B, alpha, b1, b2):
    base_rate = A * np.exp(-t / tau) + B
    trial_modulation = np.exp(alpha * trial_k)
    phase_modulation = np.exp(np.clip(b1 * s_sin + b2 * s_cos, -10, 10))
    return np.maximum(base_rate * trial_modulation * phase_modulation, 1e-9)

def phase_negative_log_likelihood_corrected(params, t_events, trial_events, s_sin, s_cos, unique_trials, n_fish):
    A, tau, B, alpha, b1, b2 = params
    
    # Term 1: Log-rates at exact event times
    rates = lambda_t_phase(t_events, trial_events, s_sin, s_cos, A, tau, B, alpha, b1, b2)
    sum_log_rates = np.sum(np.log(rates))
    
    # Term 2: Numerical Riemann Integral scaled by trial sequence and fish count
    grid_base_rate = A * np.exp(-t_grid / tau) + B
    grid_phase_mod = np.exp(np.clip(b1 * grid_sin + b2 * grid_cos, -10, 10))
    single_trial_integral = np.sum(grid_base_rate * grid_phase_mod) * dt
    
    # Total expected events = (Area of 1 trial at k=0) * (Sum of trial decays) * (Number of fish pooled)
    total_expected_events = single_trial_integral * np.sum(np.exp(alpha * unique_trials)) * n_fish
    
    nll = -(sum_log_rates - total_expected_events)
    return nll if np.isfinite(nll) else 1e10

# ==========================================
# 3. RUN THE BOUNDED OPTIMIZATION
# ==========================================
# Start with phase parameters slightly off-zero so the optimizer sees a clear initial gradient direction
initial_guesses_phase = [0.56, 1.15, 0.40, -0.15, 0.0, 0.0]
bounds_phase = (
    (0.01, 10.0),    # A
    (0.1, 5.0),      # tau
    (0.01, 5.0),     # B
    (-2.0, 2.0),     # alpha
    (-3.0, 3.0),     # b1
    (-3.0, 3.0)      # b2
)

phase_result = minimize(
    phase_negative_log_likelihood_corrected, 
    x0=initial_guesses_phase, 
    args=(spike_times, spike_trials, spike_sin, spike_cos, unique_trial_indices, num_fish),
    method='L-BFGS-B',
    bounds=bounds_phase
)

A_p, tau_p, B_p, alpha_p, b1_p, b2_p = phase_result.x

gamma = np.sqrt(b1_p**2 + b2_p**2)
phi_pref = np.arctan2(b1_p, b2_p)

print("\n--- Corrected Model Results ---")
print(f"Base Amplitude (A): {A_p:.3f} bouts/sec")
print(f"Time Constant (tau): {tau_p:.3f} seconds")
print(f"Base Baseline (B): {B_p:.3f} bouts/sec")
print(f"Trial Modulation (alpha): {alpha_p:.4f}")
print(f"Phase Tuning Strength (gamma): {gamma:.3f}")
print(f"Preferred Phase Angle (degrees): {np.degrees(phi_pref):.1f}°")

import numpy as np
import matplotlib.pyplot as plt

unique_trials = np.sort(counts['trial_num'].unique())
num_colors = len(unique_trials)
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 0.85, num_colors)] 

plt.figure(figsize=(10, 6))

for t_num, color in zip(unique_trials, colors):
    # 1. Extract empirical rolling data
    trial_data = counts[counts['trial_num'] == t_num]
    mean_data = trial_data.groupby('time_sec').agg(
        r_smooth=('jt_ipsi_hz', 'mean'),
        mean_sin=('mean_sin', 'mean'),
        mean_cos=('mean_cos', 'mean')
    )
    
    window_mask = (mean_data.index >= T_start) & (mean_data.index <= T_end)
    t_smooth = mean_data.index[window_mask] - T_start
    r_smooth = mean_data.loc[window_mask, 'r_smooth'].values
    
    # 2. Extract the actual average phase profile matching this timeline
    # This prevents the line from zig-zagging while maintaining the phase scaling trend
    s_sin = mean_data.loc[window_mask, 'mean_sin'].values
    s_cos = mean_data.loc[window_mask, 'mean_cos'].values
    
    # 3. Calculate the Phase-Modulated Model Rate
    # lambda = (A * e^(-t/tau) + B) * e^(alpha * k) * e^(b1*sin + b2*cos)
    base_rate = A_p * np.exp(-t_smooth / tau_p) + B_p
    trial_mod = np.exp(alpha_p * t_num)
    phase_mod = np.exp(b1_p * s_sin + b2_p * s_cos)
    
    r_model = base_rate * trial_mod * phase_mod
    
    # 4. Handle Labels
    label = f'Trial {t_num}' if t_num in [unique_trials[0], unique_trials[num_colors//2], unique_trials[-1]] else ""
    
    # 5. Plot
    plt.plot(t_smooth, r_smooth, '--', color=color, alpha=0.3, label=f'Data {label}' if label else "")
    plt.plot(t_smooth, r_model, '-', color=color, lw=2, label=f'Phase Model {label}' if label else "")

plt.xlabel('Time from Peak Response (seconds)')
plt.ylabel('Bout Rate (Hz)')
plt.title(f'Phase-Modulated Unified Model Fit Across Trials (alpha = {alpha_p:.4f})')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()



# ==========================================
# FIXED: TRUE INDIVIDUAL PHASE DIAGNOSTICS
# ==========================================
df_ev = pd.DataFrame({
    'file': decay_events['file'].values,       
    'k': decay_events['trial_num'].values,    
    't': spike_times, 
    'sin': spike_sin,   # Exact raw phase at event start
    'cos': spike_cos    # Exact raw phase at event end
})

rescaled_intervals = []

for (fish_id, k_idx), group in df_ev.groupby(['file', 'k']):
    group = group.sort_values('t')
    times = group['t'].values
    
    if len(times) < 2:
        continue
        
    trial_raw = pc_df[(pc_df['file'] == fish_id) & (pc_df['trial_num'] == k_idx)].copy()
    trial_raw['t_rel'] = trial_raw['trial_time'] - T_start
    
    raw_t = trial_raw['t_rel'].values
    raw_sin = trial_raw['phase_sin'].values
    raw_cos = trial_raw['phase_cos'].values
    
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i+1]
        sub_t = np.linspace(t_start, t_end, num=20)
        sub_sin = np.interp(sub_t, raw_t, raw_sin)
        sub_cos = np.interp(sub_t, raw_t, raw_cos)
        rates_slice = lambda_t_phase(sub_t, k_idx, sub_sin, sub_cos, A_p, tau_p, B_p, alpha_p, b1_p, b2_p)
        z_i = np.trapz(rates_slice, sub_t)
        rescaled_intervals.append(z_i)

# Transform and sort for the KS Plot
u = 1 - np.exp(-np.array(rescaled_intervals))
u_sorted = np.sort(u)
N_intervals = len(u_sorted)
eCDF = np.arange(1, N_intervals + 1) / N_intervals

# Calculate a formal p-value using a 1-sample KS test against a uniform distribution
ks_stat, p_val = ks_1samp(u_sorted, uniform.cdf)

print("\n--- Diagnostic Results ---")
print(f"Number of analyzed inter-event intervals: {N_intervals}")
print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.4f}")
print(f"KS Test p-value: {p_val:.4f} (p > 0.05 means model fully captures data structure)")

# ==========================================
# 2. GENERATE THE DIAGNOSTIC KS-PLOT
# ==========================================
plt.figure(figsize=(6, 6))

# Perfect model line
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Inhomogeneous Poisson Process')

# 95% Confidence Bounds
c_bound = 1.36 / np.sqrt(N_intervals)
plt.plot([0, 1], [c_bound, 1+c_bound], 'r:', alpha=0.5, label='95% Confidence Bounds')
plt.plot([0, 1], [-c_bound, 1-c_bound], 'r:', alpha=0.5)

# Our model profile
plt.plot(u_sorted, eCDF, color='crimson', lw=2.5, label=f'Phase Model (p={p_val:.3f})')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Theoretical Cumulative Quantiles')
plt.ylabel('Empirical Cumulative Quantiles')
plt.title('KS Diagnostic Plot via Time-Rescaling')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.15)
plt.show()



######## Alpha function instead of pure exponential


def lambda_t_phase_alpha(t, trial_k, s_sin, s_cos, A, tau, B, alpha, b1, b2):
    # Alpha Activation: t * exp(-t/tau) allows the rate to start at B, 
    # rise smoothly to a peak at t = tau, and then decay.
    base_rate = A * (t / tau) * np.exp(-t / tau) + B
    
    trial_modulation = np.exp(alpha * trial_k)
    phase_modulation = np.exp(np.clip(b1 * s_sin + b2 * s_cos, -5, 5))
    return np.maximum(base_rate * trial_modulation * phase_modulation, 1e-9)

def phase_negative_log_likelihood_alpha(params, t_events, trial_events, s_sin, s_cos, unique_trials, t_grid, grid_sin, grid_cos, dt):
    A, tau, B, alpha, b1, b2 = params
    
    # Term 1: Log-rates at exact event times using Alpha dynamics
    rates = lambda_t_phase_alpha(t_events, trial_events, s_sin, s_cos, A, tau, B, alpha, b1, b2)
    sum_log_rates = np.sum(np.log(rates))
    
    # Term 2: Numerical Riemann Integral across the single template grid
    # Crucial: Must mirror the exact same alpha activation formula used above!
    grid_base_rate = A * (t_grid / tau) * np.exp(-t_grid / tau) + B
    grid_phase_mod = np.exp(np.clip(b1 * grid_sin + b2 * grid_cos, -5, 5))
    
    # Area under the curve for a single baseline trial (k=0)
    single_trial_integral = np.sum(grid_base_rate * grid_phase_mod) * dt
    
    # Total expected events scaled across all unique trials and pooled fish
    total_expected_events = single_trial_integral * np.sum(np.exp(alpha * unique_trials))
    
    nll = -(sum_log_rates - total_expected_events)
    return nll if np.isfinite(nll) else 1e10


initial_guesses_phase = [0.56, 1.15, 0.40, -0.15, 0.0, 0.0]
bounds_phase = (
    (0.01, 10.0),    # A
    (0.1, 5.0),      # tau
    (0.01, 5.0),     # B
    (-2.0, 2.0),     # alpha
    (-3.0, 3.0),     # b1
    (-3.0, 3.0)      # b2
)

phase_result = minimize(
    phase_negative_log_likelihood_alpha, 
    x0=initial_guesses_phase, 
    # Appended global template grid variables to the optimization args
    args=(spike_times, spike_trials, spike_sin, spike_cos, unique_trial_counts, t_grid, grid_sin, grid_cos, dt),
    method='L-BFGS-B',
    bounds=bounds_phase
)

A_p, tau_p, B_p, alpha_p, b1_p, b2_p = phase_result.x

gamma = np.sqrt(b1_p**2 + b2_p**2)
phi_pref = np.arctan2(b1_p, b2_p)

print("\n--- Alpha Phase Model Results ---")
print(f"Base Amplitude (A): {A_p:.3f} bouts/sec")
print(f"Time Constant (tau): {tau_p:.3f} seconds")
print(f"Base Baseline (B): {B_p:.3f} bouts/sec")
print(f"Trial Modulation (alpha): {alpha_p:.4f}")
print(f"Phase Tuning Strength (gamma): {gamma:.3f}")
print(f"Preferred Phase Angle (degrees): {np.degrees(phi_pref):.1f}°")

# ==========================================
# 4. PLOT EMPIRICAL VS MODEL PROPENSITY
# ==========================================
unique_trials = np.sort(counts['trial_num'].unique())
num_colors = len(unique_trials)
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 0.85, num_colors)] 

plt.figure(figsize=(10, 6))

for t_num, color in zip(unique_trials, colors):
    # 1. Extract empirical rolling data
    trial_data = counts[counts['trial_num'] == t_num]
    mean_data = trial_data.groupby('time_sec').agg(
        r_smooth=('jt_ipsi_hz', 'mean'),
        mean_sin=('mean_sin', 'mean'),
        mean_cos=('mean_cos', 'mean')
    )
    
    window_mask = (mean_data.index >= T_start) & (mean_data.index <= T_end)
    t_smooth = mean_data.index[window_mask] - T_start
    r_smooth = mean_data.loc[window_mask, 'r_smooth'].values
    
    s_sin = mean_data.loc[window_mask, 'mean_sin'].values
    s_cos = mean_data.loc[window_mask, 'mean_cos'].values
    
    # 2. Calculate Alpha-Modulated Model Rate
    # UPDATED: Implemented the non-monotonic rise time base rate here
    base_rate = A_p * (t_smooth / tau_p) * np.exp(-t_smooth / tau_p) + B_p
    trial_mod = np.exp(alpha_p * t_num)
    phase_mod = np.exp(np.clip(b1_p * s_sin + b2_p * s_cos, -5, 5))
    
    r_model = base_rate * trial_mod * phase_mod
    
    # 3. Handle Labels
    label = f'Trial {t_num}' if t_num in [unique_trials[0], unique_trials[num_colors//2], unique_trials[-1]] else ""
    
    # 4. Plot
    plt.plot(t_smooth, r_smooth, '--', color=color, alpha=0.3, label=f'Data {label}' if label else "")
    plt.plot(t_smooth, r_model, '-', color=color, lw=2, label=f'Alpha Model {label}' if label else "")

plt.xlabel('Time from Peak Response (seconds)')
plt.ylabel('Bout Rate (Hz)')
plt.title(f'Alpha-Activated Unified Model Fit Across Trials (alpha = {alpha_p:.4f})')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# ==========================================
# FIXED: TRUE INDIVIDUAL PHASE DIAGNOSTICS
# ==========================================
df_ev = pd.DataFrame({
    'file': decay_events['file'].values,       
    'k': decay_events['trial_num'].values,    
    't': spike_times, 
    'sin': spike_sin,   
    'cos': spike_cos    
})

rescaled_intervals = []

for (fish_id, k_idx), group in df_ev.groupby(['file', 'k']):
    group = group.sort_values('t')
    times = group['t'].values
    
    if len(times) < 2:
        continue
        
    trial_raw = pc_df[(pc_df['file'] == fish_id) & (pc_df['trial_num'] == k_idx)].copy()
    trial_raw['t_rel'] = trial_raw['trial_time'] - T_start
    
    raw_t = trial_raw['t_rel'].values
    raw_sin = trial_raw['phase_sin'].values
    raw_cos = trial_raw['phase_cos'].values
    
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i+1]
        sub_t = np.linspace(t_start, t_end, num=20)
        sub_sin = np.interp(sub_t, raw_t, raw_sin)
        sub_cos = np.interp(sub_t, raw_t, raw_cos)
        
        # UPDATED: Slicing integration via Alpha-activation parameters
        rates_slice = lambda_t_phase_alpha(sub_t, k_idx, sub_sin, sub_cos, A_p, tau_p, B_p, alpha_p, b1_p, b2_p)
        z_i = np.trapz(rates_slice, sub_t)
        rescaled_intervals.append(z_i)

# Transform and sort for the KS Plot
u = 1 - np.exp(-np.array(rescaled_intervals))
u_sorted = np.sort(u)
N_intervals = len(u_sorted)
eCDF = np.arange(1, N_intervals + 1) / N_intervals

# FIXED: Standardized natively via scipy's 1-sample uniform distribution test
ks_stat, p_val = kstest(u, 'uniform')

print("\n--- Diagnostic Results ---")
print(f"Number of analyzed inter-event intervals: {N_intervals}")
print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.4f}")
print(f"KS Test p-value: {p_val:.4f} (p > 0.05 means model fully captures data structure)")

# ==========================================
# 5. GENERATE THE DIAGNOSTIC KS-PLOT
# ==========================================
plt.figure(figsize=(6, 6))

# Perfect model line
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Inhomogeneous Poisson Process')

# 95% Confidence Bounds
c_bound = 1.36 / np.sqrt(N_intervals)
plt.plot([0, 1], [c_bound, 1+c_bound], 'r:', alpha=0.5, label='95% Confidence Bounds')
plt.plot([0, 1], [-c_bound, 1-c_bound], 'r:', alpha=0.5)

# Our model profile
plt.plot(u_sorted, eCDF, color='crimson', lw=2.5, label=f'Alpha Model (p={p_val:.3f})')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Theoretical Cumulative Quantiles')
plt.ylabel('Empirical Cumulative Quantiles')
plt.title('KS Diagnostic Plot via Alpha Time-Rescaling')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.15)
plt.show()


############## gamma activation

# ==========================================
# 1. INTENSITY & LOG-LIKELIHOOD FUNCTIONS (GAMMA)
# ==========================================
def lambda_t_phase_gamma(t, trial_k, s_sin, s_cos, A, tau, k, B, alpha, b1, b2):
    # Gamma Activation: (t^k) * exp(-t/tau) decouple rise time from decay.
    # np.maximum(t, 0) shields the optimizer from taking fractional roots of negative numbers.
    t_safe = np.maximum(t, 0)
    base_rate = A * (t_safe ** k) * np.exp(-t_safe / tau) + B
    
    trial_modulation = np.exp(alpha * trial_k)
    phase_modulation = np.exp(np.clip(b1 * s_sin + b2 * s_cos, -5, 5))
    return np.maximum(base_rate * trial_modulation * phase_modulation, 1e-9)

def phase_negative_log_likelihood_gamma(params, t_events, trial_events, s_sin, s_cos, unique_trials, t_grid, grid_sin, grid_cos, dt):
    A, tau, k, B, alpha, b1, b2 = params
    
    # Term 1: Log-rates at exact event times using Gamma dynamics
    rates = lambda_t_phase_gamma(t_events, trial_events, s_sin, s_cos, A, tau, k, B, alpha, b1, b2)
    sum_log_rates = np.sum(np.log(rates))
    
    # Term 2: Numerical Riemann Integral across the single template grid
    t_grid_safe = np.maximum(t_grid, 0)
    grid_base_rate = A * (t_grid_safe ** k) * np.exp(-t_grid_safe / tau) + B
    grid_phase_mod = np.exp(np.clip(b1 * grid_sin + b2 * grid_cos, -5, 5))
    
    # Area under the curve for a single baseline trial (k=0)
    single_trial_integral = np.sum(grid_base_rate * grid_phase_mod) * dt
    
    # Total expected events scaled across all unique trials
    total_expected_events = single_trial_integral * np.sum(np.exp(alpha * unique_trials))
    
    nll = -(sum_log_rates - total_expected_events)
    return nll if np.isfinite(nll) else 1e10

# ==========================================
# 2. RUN THE BOUNDED OPTIMIZATION
# ==========================================
# Parameters structured as: [A, tau, k, B, alpha, b1, b2]
# Starting k at 2.0 allows a smooth, suppressed onset at t=0 to stop overshooting
initial_guesses_phase = [0.56, 1.15, 2.0, 0.40, -0.15, 0.0, 0.0]
bounds_phase = (
    (0.01, 10.0),    # A
    (0.1, 5.0),      # tau
    (0.5, 10.0),     # k (Keep >= 0.5 to keep gradients stable near t=0)
    (0.01, 5.0),     # B
    (-2.0, 2.0),     # alpha
    (-3.0, 3.0),     # b1
    (-3.0, 3.0)      # b2
)

phase_result = minimize(
    phase_negative_log_likelihood_gamma, 
    x0=initial_guesses_phase, 
    args=(spike_times, spike_trials, spike_sin, spike_cos, unique_trial_counts, t_grid, grid_sin, grid_cos, dt),
    method='L-BFGS-B',
    bounds=bounds_phase
)

A_p, tau_p, k_p, B_p, alpha_p, b1_p, b2_p = phase_result.x

gamma = np.sqrt(b1_p**2 + b2_p**2)
phi_pref = np.arctan2(b1_p, b2_p)

theta_pref = np.deg2rad(70) * ((1 - np.cos(phi_pref))/2) # check that

print("\n--- Gamma Phase Model Results ---")
print(f"Base Amplitude (A): {A_p:.3f}")
print(f"Time Constant (tau): {tau_p:.3f} seconds")
print(f"Shape Parameter (k): {k_p:.3f}")
print(f"Peak time (k*tau): {k_p* tau_p:.3f} seconds")
print(f"Base Baseline (B): {B_p:.3f} bouts/sec")
print(f"Trial Modulation (alpha): {alpha_p:.4f}")
print(f"Phase Tuning Strength (gamma): {gamma:.3f}")
print(f"Preferred Phase Angle (degrees): {np.degrees(phi_pref):.1f}°")

# ==========================================
# 3. PLOT EMPIRICAL VS MODEL PROPENSITY
# ==========================================
unique_trials = np.sort(counts['trial_num'].unique())
num_colors = len(unique_trials)
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 0.85, num_colors)] 

plt.figure(figsize=(10, 6))

for t_num, color in zip(unique_trials, colors):
    trial_data = counts[counts['trial_num'] == t_num]
    mean_data = trial_data.groupby('time_sec').agg(
        r_smooth=('jt_ipsi_hz', 'mean'),
        mean_sin=('mean_sin', 'mean'),
        mean_cos=('mean_cos', 'mean')
    )
    
    window_mask = (mean_data.index >= T_start) & (mean_data.index <= T_end)
    t_smooth = mean_data.index[window_mask] - T_start
    r_smooth = mean_data.loc[window_mask, 'r_smooth'].values
    
    s_sin = mean_data.loc[window_mask, 'mean_sin'].values
    s_cos = mean_data.loc[window_mask, 'mean_cos'].values
    
    # Calculate Gamma-Modulated Predicted Model Rate
    t_smooth_safe = np.maximum(t_smooth, 0)
    base_rate = A_p * (t_smooth_safe ** k_p) * np.exp(-t_smooth_safe / tau_p) + B_p
    trial_mod = np.exp(alpha_p * t_num)
    phase_mod = np.exp(np.clip(b1_p * s_sin + b2_p * s_cos, -5, 5))
    
    r_model = base_rate * trial_mod * phase_mod
    
    label = f'Trial {t_num}' if t_num in [unique_trials[0], unique_trials[num_colors//2], unique_trials[-1]] else ""
    
    plt.plot(t_smooth, r_smooth, '--', color=color, alpha=0.3, label=f'Data {label}' if label else "")
    plt.plot(t_smooth, r_model, '-', color=color, lw=2, label=f'Gamma Model {label}' if label else "")

plt.xlabel('Time from Peak Response (seconds)')
plt.ylabel('Bout Rate (Hz)')
plt.title(f'Gamma-Activated Unified Model Fit Across Trials (alpha = {alpha_p:.4f})')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# ==========================================
# 4. TRUE INDIVIDUAL PHASE DIAGNOSTICS
# ==========================================
df_ev = pd.DataFrame({
    'file': decay_events['file'].values,       
    'k': decay_events['trial_num'].values,    
    't': spike_times, 
    'sin': spike_sin,   
    'cos': spike_cos    
})

rescaled_intervals = []

for (fish_id, k_idx), group in df_ev.groupby(['file', 'k']):
    group = group.sort_values('t')
    times = group['t'].values
    
    if len(times) < 2:
        continue
        
    trial_raw = pc_df[(pc_df['file'] == fish_id) & (pc_df['trial_num'] == k_idx)].copy()
    trial_raw['t_rel'] = trial_raw['trial_time'] - T_start
    
    raw_t = trial_raw['t_rel'].values
    raw_sin = trial_raw['phase_sin'].values
    raw_cos = trial_raw['phase_cos'].values
    
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i+1]
        sub_t = np.linspace(t_start, t_end, num=20)
        sub_sin = np.interp(sub_t, raw_t, raw_sin)
        sub_cos = np.interp(sub_t, raw_t, raw_cos)
        
        # Integration slices utilizing Gamma-activation logic
        rates_slice = lambda_t_phase_gamma(sub_t, k_idx, sub_sin, sub_cos, A_p, tau_p, k_p, B_p, alpha_p, b1_p, b2_p)
        z_i = np.trapz(rates_slice, sub_t)
        rescaled_intervals.append(z_i)

u = 1 - np.exp(-np.array(rescaled_intervals))
u_sorted = np.sort(u)
N_intervals = len(u_sorted)
eCDF = np.arange(1, N_intervals + 1) / N_intervals

ks_stat, p_val = kstest(u, 'uniform')

print("\n--- Diagnostic Results ---")
print(f"Number of analyzed inter-event intervals: {N_intervals}")
print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.4f}")
print(f"KS Test p-value: {p_val:.4f} (p > 0.05 means model fully captures data structure)")

# ==========================================
# 5. GENERATE THE DIAGNOSTIC KS-PLOT
# ==========================================
plt.figure(figsize=(6, 6))

plt.plot([0, 1], [0, 1], 'k--', label='Perfect Inhomogeneous Poisson Process')

c_bound = 1.36 / np.sqrt(N_intervals)
plt.plot([0, 1], [c_bound, 1+c_bound], 'r:', alpha=0.5, label='95% Confidence Bounds')
plt.plot([0, 1], [-c_bound, 1-c_bound], 'r:', alpha=0.5)

plt.plot(u_sorted, eCDF, color='crimson', lw=2.5, label=f'Gamma Model (p={p_val:.3f})')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Theoretical Cumulative Quantiles')
plt.ylabel('Empirical Cumulative Quantiles')
plt.title('KS Diagnostic Plot via Gamma Time-Rescaling')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.15)
plt.show()


############### GAMMA / PHASE MULTIPLICATIVE / NO TRIALS

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import kstest

# ==========================================
# 1. INTENSITY & LOG-LIKELIHOOD FUNCTIONS (TRIAL AVERAGED)
# ==========================================
def lambda_t_phase_gamma_averaged(t, s_sin, s_cos, A, tau, k, B, b1, b2):
    # Alpha/Gamma parameter vector dropped the alpha parameter. 
    # Dynamics are now identical across all trial numbers.
    t_safe = np.maximum(t, 0)
    base_rate = A * (t_safe ** k) * np.exp(-t_safe / tau) + B
    phase_modulation = np.exp(np.clip(b1 * s_sin + b2 * s_cos, -5, 5))
    return np.maximum(base_rate * phase_modulation, 1e-9)

def phase_negative_log_likelihood_gamma_averaged(params, t_events, s_sin, s_cos, total_num_trials, t_grid, grid_sin, grid_cos, dt):
    # Parameter footprint reduced to 6 dimensions
    A, tau, k, B, b1, b2 = params
    
    # Term 1: Log-rates at exact event times (unmodulated by trial progression)
    rates = lambda_t_phase_gamma_averaged(t_events, s_sin, s_cos, A, tau, k, B, b1, b2)
    sum_log_rates = np.sum(np.log(rates))
    
    # Term 2: Numerical Riemann Integral across the single template grid
    t_grid_safe = np.maximum(t_grid, 0)
    grid_base_rate = A * (t_grid_safe ** k) * np.exp(-t_grid_safe / tau) + B
    grid_phase_mod = np.exp(np.clip(b1 * grid_sin + b2 * grid_cos, -5, 5))
    
    single_trial_integral = np.sum(grid_base_rate * grid_phase_mod) * dt
    
    # Total expected events = Area of 1 uniform trial * Total trial blocks tracked
    total_expected_events = single_trial_integral * total_num_trials
    
    nll = -(sum_log_rates - total_expected_events)
    return nll if np.isfinite(nll) else 1e10

# ==========================================
# 2. RUN THE BOUNDED OPTIMIZATION
# ==========================================
# Dropped alpha from vectors. Format: [A, tau, k, B, b1, b2]
initial_guesses_phase = [0.56, 1.15, 2.0, 0.40, 0.0, 0.0]
bounds_phase = (
    (0.01, 10.0),    # A
    (0.1, 5.0),      # tau
    (0.5, 10.0),     # k 
    (0.01, 5.0),     # B
    (-3.0, 3.0),     # b1
    (-3.0, 3.0)      # b2
)

# Extract total count of unique trial execution instances across dataset
num_trials = len(pc_df.groupby(['file', 'trial_num']))

phase_result = minimize(
    phase_negative_log_likelihood_gamma_averaged, 
    x0=initial_guesses_phase, 
    # Adjusted arguments to feed the overall trial count scalar directly
    args=(spike_times, spike_sin, spike_cos, num_trials, t_grid, grid_sin, grid_cos, dt),
    method='L-BFGS-B',
    bounds=bounds_phase
)

A_p, tau_p, k_p, B_p, b1_p, b2_p = phase_result.x

gamma = np.sqrt(b1_p**2 + b2_p**2)
phi_pref = np.arctan2(b1_p, b2_p)

print("\n--- Trial-Averaged Gamma Phase Model Results ---")
print(f"Base Amplitude (A): {A_p:.3f}")
print(f"Time Constant (tau): {tau_p:.3f} seconds")
print(f"Shape Parameter (k): {k_p:.3f}")
print(f"Peak time (k*tau): {k_p * tau_p:.3f} seconds")
print(f"Base Baseline (B): {B_p:.3f} bouts/sec")
print(f"Phase Tuning Strength (gamma): {gamma:.3f}")
print(f"Preferred Phase Angle (degrees): {np.degrees(phi_pref):.1f}°")

# ==========================================
# 3. PLOT EMPIRICAL VS UNIFORM MODEL PROPENSITY
# ==========================================
# Since trials are averaged, we aggregate empirical data globally over time bins
mean_data = counts.groupby('time_sec').agg(
    r_smooth=('jt_ipsi_hz', 'mean'),
    mean_sin=('mean_sin', 'mean'),
    mean_cos=('mean_cos', 'mean')
)

window_mask = (mean_data.index >= T_start) & (mean_data.index <= T_end)
t_smooth = mean_data.index[window_mask] - T_start
r_smooth = mean_data.loc[window_mask, 'r_smooth'].values
s_sin = mean_data.loc[window_mask, 'mean_sin'].values
s_cos = mean_data.loc[window_mask, 'mean_cos'].values

# Calculate single trial averaged trajectory
t_smooth_safe = np.maximum(t_smooth, 0)
base_rate = A_p * (t_smooth_safe ** k_p) * np.exp(-t_smooth_safe / tau_p) + B_p
phase_mod = np.exp(np.clip(b1_p * s_sin + b2_p * s_cos, -5, 5))
r_model = base_rate * phase_mod

plt.figure(figsize=(10, 6))
plt.plot(t_smooth, r_smooth, 'k--', alpha=0.5, label='Averaged Empirical Data')
plt.plot(t_smooth, r_model, 'b-', lw=2.5, label='Trial-Averaged Gamma Model')

plt.xlabel('Time from Peak Response (seconds)')
plt.ylabel('Bout Rate (Hz)')
plt.title('Trial-Averaged Gamma-Activated Unified Model Fit')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# ==========================================
# 4. TRUE INDIVIDUAL PHASE DIAGNOSTICS
# ==========================================
df_ev = pd.DataFrame({
    'file': decay_events['file'].values,       
    'k': decay_events['trial_num'].values,    
    't': spike_times, 
    'sin': spike_sin,   
    'cos': spike_cos    
})

rescaled_intervals = []

for (fish_id, k_idx), group in df_ev.groupby(['file', 'k']):
    group = group.sort_values('t')
    times = group['t'].values
    
    if len(times) < 2:
        continue
        
    trial_raw = pc_df[(pc_df['file'] == fish_id) & (pc_df['trial_num'] == k_idx)].copy()
    trial_raw['t_rel'] = trial_raw['trial_time'] - T_start
    
    raw_t = trial_raw['t_rel'].values
    raw_sin = trial_raw['phase_sin'].values
    raw_cos = trial_raw['phase_cos'].values
    
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i+1]
        sub_t = np.linspace(t_start, t_end, num=20)
        sub_sin = np.interp(sub_t, raw_t, raw_sin)
        sub_cos = np.interp(sub_t, raw_t, raw_cos)
        
        # Integration slices utilizing trial-averaged logic
        rates_slice = lambda_t_phase_gamma_averaged(sub_t, sub_sin, sub_cos, A_p, tau_p, k_p, B_p, b1_p, b2_p)
        z_i = np.trapz(rates_slice, sub_t)
        rescaled_intervals.append(z_i)

u = 1 - np.exp(-np.array(rescaled_intervals))
u_sorted = np.sort(u)
N_intervals = len(u_sorted)
eCDF = np.arange(1, N_intervals + 1) / N_intervals

ks_stat, p_val = kstest(u, 'uniform')

print("\n--- Diagnostic Results ---")
print(f"Number of analyzed inter-event intervals: {N_intervals}")
print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.4f}")
print(f"KS Test p-value: {p_val:.4f}")

# ==========================================
# 5. GENERATE THE DIAGNOSTIC KS-PLOT
# ==========================================
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Inhomogeneous Poisson Process')

c_bound = 1.36 / np.sqrt(N_intervals)
plt.plot([0, 1], [c_bound, 1+c_bound], 'r:', alpha=0.5, label='95% Confidence Bounds')
plt.plot([0, 1], [-c_bound, 1-c_bound], 'r:', alpha=0.5)

plt.plot(u_sorted, eCDF, color='crimson', lw=2.5, label=f'Averaged Gamma Model (p={p_val:.3f})')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Theoretical Cumulative Quantiles')
plt.ylabel('Empirical Cumulative Quantiles')
plt.title('KS Diagnostic Plot via Averaged Time-Rescaling')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.15)
plt.show()


################################ PHASE ADDITIVE / NO TRIALS / GAMMA


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import kstest

# ==========================================
# 1. LINEAR INTENSITY & LOG-LIKELIHOOD FUNCTIONS
# ==========================================
def lambda_t_phase_gamma_linear(t, s_sin, s_cos, A, tau, k, B, b1, b2):
    t_safe = np.maximum(t, 0)
    
    transient_peak = A * (t_safe ** k) * np.exp(-t_safe / tau)
    static_baseline = B
    
    # Strict Linear Phase: Phase components are strictly ADDED, not multiplied.
    phase_ripple = b1 * s_sin + b2 * s_cos
    
    # Enforce global rate positivity cleanly across the entire sum
    return np.maximum(transient_peak + static_baseline + phase_ripple, 1e-9)

def phase_negative_log_likelihood_gamma_linear(params, t_events, s_sin, s_cos, total_pool_size, t_grid, grid_sin, grid_cos, dt):
    A, tau, k, B, b1, b2 = params
    
    # Term 1: Log-rates at exact event times
    rates = lambda_t_phase_gamma_linear(t_events, s_sin, s_cos, A, tau, k, B, b1, b2)
    sum_log_rates = np.sum(np.log(rates))
    
    # Term 2: Numerical Riemann Integral across the single template grid
    t_grid_safe = np.maximum(t_grid, 0)
    grid_transient = A * (t_grid_safe ** k) * np.exp(-t_grid_safe / tau)
    grid_phase = b1 * grid_sin + b2 * grid_cos
    
    # Integrated expected rate of a single uniform trial run
    single_trial_integral = np.sum(np.maximum(grid_transient + B + grid_phase, 1e-9)) * dt
    total_expected_events = single_trial_integral * total_pool_size
    
    nll = -(sum_log_rates - total_expected_events)
    return nll if np.isfinite(nll) else 1e10

# ==========================================
# 2. RUN THE LINEAR OPTIMIZATION
# ==========================================
# Since b1 and b2 are no longer inside an exponent, their units change from log-rate shifts 
# to direct additions in Hz. We shift initial guesses to 0.0 to let it discover this scale.
initial_guesses_phase = [0.56, 1.15, 2.0, 0.40, 0.0, 0.0]
bounds_phase = (
    (0.01, 10.0),    # A
    (0.1, 5.0),      # tau
    (0.4, 10.0),     # k 
    (0.01, 5.0),     # B
    (-5.0, 5.0),     # b1 (expanded slightly since linear weights map to absolute Hz)
    (-5.0, 5.0)      # b2
)

num_trials = len(pc_df.groupby(['file', 'trial_num']))
num_trials = len(pc_df.groupby(['file']))*len(pc_df.groupby(['trial_num']))

phase_result = minimize(
    phase_negative_log_likelihood_gamma_linear, 
    x0=initial_guesses_phase, 
    args=(spike_times, spike_sin, spike_cos, num_trials, t_grid, grid_sin, grid_cos, dt),
    method='L-BFGS-B',
    bounds=bounds_phase
)

A_p, tau_p, k_p, B_p, b1_p, b2_p = phase_result.x

# Note: In linear space, gamma directly represents the peak-to-trough amplitude in Hz!
gamma = np.sqrt(b1_p**2 + b2_p**2)
phi_pref = np.arctan2(b1_p, b2_p)

# Physical spatial mapping for right eye stimulus (20 to 90 degrees)
arc_start_deg = 20.0
angle_range_deg = 90.0 - 20.0
normalized_pos = (1.0 - np.cos(phi_pref)) / 2.0
theta_pref_deg = arc_start_deg + (angle_range_deg * normalized_pos)

print("\n--- Linear Gamma Phase Model Results ---")
print(f"Base Amplitude (A): {A_p:.3f}")
print(f"Time Constant (tau): {tau_p:.3f} seconds")
print(f"Shape Parameter (k): {k_p:.3f}")
print(f"Peak time (k*tau): {k_p * tau_p:.3f} seconds")
print(f"Base Baseline (B): {B_p:.3f} bouts/sec")
print(f"Phase Ripple Amplitude (gamma): {gamma:.3f} Hz modulation")
print(f"Preferred Phase Angle (degrees): {np.degrees(phi_pref):.1f}°")
print(f"Preferred Visual Angle (theta): {theta_pref_deg:.1f}° in right field")

# ==========================================
# 3. PLOT AVERAGED PROPENSITY WITH DASHBOARD BELOW
# ==========================================
mean_data = counts.groupby('time_sec').agg(
    r_smooth=('jt_ipsi_hz', 'mean'),
    mean_sin=('mean_sin', 'mean'),
    mean_cos=('mean_cos', 'mean')
)

window_mask = (mean_data.index >= T_start) & (mean_data.index <= T_end)
t_smooth = mean_data.index[window_mask] - T_start
r_smooth = mean_data.loc[window_mask, 'r_smooth'].values
s_sin = mean_data.loc[window_mask, 'mean_sin'].values
s_cos = mean_data.loc[window_mask, 'mean_cos'].values

# Calculate linear model trajectory
r_model = lambda_t_phase_gamma_linear(t_smooth, s_sin, s_cos, A_p, tau_p, k_p, B_p, b1_p, b2_p)

# Create a 2-row layout. 
# height_ratios=[3, 1] means the top plot gets 75% of the height, bottom axis gets 25%.
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(10, 8), 
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=False
)

# --- TOP ROW: DATA AXIS ---
ax1.plot(t_smooth, r_smooth, 'k--', alpha=0.5, label='Averaged Empirical Data')
ax1.plot(t_smooth, r_model, 'darkorange', lw=2.5, label='Strict Linear Phase Gamma Model')
ax1.set_ylabel('Bout Rate $\lambda(t)$  (Hz)')
ax1.set_xlabel('time (s)')
ax1.set_title(r"$\lambda(t) = \max \left( A \cdot t^k e^{-t/\tau} + B + b_1 \sin(\phi) + b_2 \cos(\phi),\, 10^{-9} \right)$")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_ylim(bottom=0)

# --- BOTTOM ROW: ANNOTATION DASHBOARD AXIS ---
# Hide the structural axis elements (ticks, spine box, labels) to create a clean white slate
ax2.axis('off')

# --- REFACTORED DASHBOARD WITH SYSTEMATIC INTERPRETATIONS ---
latex_annotation = r"""
$\text{Peak Time} = \tau \cdot k$ [Time from stimulus onset to max transient response]
$\gamma = \sqrt{b_1^2 + b_2^2}$ [Single-sided oscillation amplitude]
$\phi_{\text{pref}} = \text{atan2}(b_1, b_2)$ [Preferred phase coordinate]

Fitted Parameters & Interpretations:
$A$ = """ + f"{A_p:.3f}" + r""" Hz [Scales the height of the initial stimulus-evoked peak]
$B$ = """ + f"{B_p:.3f}" + r""" Hz [Stable baseline rate floor after the initial transient activation settles down]

$\tau$ = """ + f"{tau_p:.3f}" + r""" sec [Exponential decay rate]
$k$ = """ + f"{k_p:.3f}" + r""" [Initial acceleration ramp ]
Peak Time = """ + f"{tau_p * k_p:.3f}" + r""" sec [Exact time delay where the transient response reaches its absolute maximum]

$2\gamma$ = """ + f"{2*gamma:.3f}" + r""" Hz [The full peak-to-trough variation window forced by the back-and-forth movement of the target]
$\theta_{\text{pref}}$ = """ + f"{theta_pref_deg:.1f}" + r"""$^\circ$ [The physical position in the visual field where the target triggers the highest response probability]"""

# Render the text directly inside the blank bottom panel
ax2.text(
    0.01, 0.95,                  # Position slightly inset from the upper-left of ax2
    latex_annotation, 
    transform=ax2.transAxes,
    fontsize=10, 
    verticalalignment='top',
    horizontalalignment='left',
    multialignment='left'
)

# Use tight_layout to automatically space the panels so they don't overlap
plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_model_NHPP_ipsi_1.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_model_NHPP_ipsi_1.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()


# ==========================================
# 4. TRUE INDIVIDUAL PHASE DIAGNOSTICS
# ==========================================
df_ev = pd.DataFrame({
    'file': decay_events['file'].values,       
    'k': decay_events['trial_num'].values,    
    't': spike_times, 
    'sin': spike_sin,   
    'cos': spike_cos    
})

rescaled_intervals = []

for (fish_id, k_idx), group in df_ev.groupby(['file', 'k']):
    group = group.sort_values('t')
    times = group['t'].values
    
    if len(times) < 2:
        continue
        
    trial_raw = pc_df[(pc_df['file'] == fish_id) & (pc_df['trial_num'] == k_idx)].copy()
    trial_raw['t_rel'] = trial_raw['trial_time'] - T_start
    
    raw_t = trial_raw['t_rel'].values
    raw_sin = trial_raw['phase_sin'].values
    raw_cos = trial_raw['phase_cos'].values
    
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i+1]
        sub_t = np.linspace(t_start, t_end, num=20)
        sub_sin = np.interp(sub_t, raw_t, raw_sin)
        sub_cos = np.interp(sub_t, raw_t, raw_cos)
        
        # Integration slices utilizing linear addition logic
        rates_slice = lambda_t_phase_gamma_linear(sub_t, sub_sin, sub_cos, A_p, tau_p, k_p, B_p, b1_p, b2_p)
        z_i = np.trapz(rates_slice, sub_t)
        rescaled_intervals.append(z_i)

u = 1 - np.exp(-np.array(rescaled_intervals))
u_sorted = np.sort(u)
N_intervals = len(u_sorted)
eCDF = np.arange(1, N_intervals + 1) / N_intervals

ks_stat, p_val = kstest(u, 'uniform')

print("\n--- Diagnostic Results ---")
print(f"Number of analyzed inter-event intervals: {N_intervals}")
print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.4f}")
print(f"KS Test p-value: {p_val:.4f}")

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Inhomogeneous Poisson Process')

c_bound = 1.36 / np.sqrt(N_intervals)
plt.plot([0, 1], [c_bound, 1+c_bound], 'r:', alpha=0.5, label='95% Confidence Bounds')
plt.plot([0, 1], [-c_bound, 1-c_bound], 'r:', alpha=0.5)

plt.plot(u_sorted, eCDF, color='crimson', lw=2.5, label=f'Averaged Gamma Model (p={p_val:.3f})')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Theoretical Cumulative Quantiles')
plt.ylabel('Empirical Cumulative Quantiles')
plt.title('KS Diagnostic Plot via Averaged Time-Rescaling')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.15)
plt.show()



########################## WITH TRIALS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import kstest

# ==========================================
# 1. TRI-MODULATED INTENSITY & LOG-LIKELIHOOD
# ==========================================
def lambda_t_phase_gamma_tri_modulated(t, trial_k, s_sin, s_cos, A, tau, k, B, b1, b2, alpha_A, alpha_B, alpha_gamma):
    t_safe = np.maximum(t, 0)
    
    # Each structural parameter module is scaled by its unique trial-decay coefficient
    transient_peak = (A * (t_safe ** k) * np.exp(-t_safe / tau)) * np.exp(alpha_A * trial_k)
    static_baseline = B * np.exp(alpha_B * trial_k)
    phase_ripple = (b1 * s_sin + b2 * s_cos) * np.exp(alpha_gamma * trial_k)
    
    return np.maximum(transient_peak + static_baseline + phase_ripple, 1e-9)

def phase_negative_log_likelihood_tri_modulated(params, t_events, trial_events, s_sin, s_cos, unique_trials, n_fish, t_grid, grid_sin, grid_cos, dt):
    A, tau, k, B, b1, b2, alpha_A, alpha_B, alpha_gamma = params
    
    # Term 1: Event Log-Rates
    rates = lambda_t_phase_gamma_tri_modulated(t_events, trial_events, s_sin, s_cos, A, tau, k, B, b1, b2, alpha_A, alpha_B, alpha_gamma)
    sum_log_rates = np.sum(np.log(rates))
    
    # Term 2: Numerical Riemann Integral across all sequential trials
    t_grid_safe = np.maximum(t_grid, 0)
    
    # Evaluate a baseline template for each core process component over the grid
    grid_transient = A * (t_grid_safe ** k) * np.exp(-t_grid_safe / tau)
    grid_phase = b1 * grid_sin + b2 * grid_cos
    
    total_expected_events = 0.0
    for tk in unique_trials:
        # Reconstruct the exact rate profile for trial index tk
        trial_grid_rate = np.maximum(
            grid_transient * np.exp(alpha_A * tk) + 
            B * np.exp(alpha_B * tk) + 
            grid_phase * np.exp(alpha_gamma * tk), 
            1e-9
        )
        total_expected_events += np.sum(trial_grid_rate) * dt
        
    # Scale across entire fish pool population
    total_expected_events *= n_fish
    
    nll = -(sum_log_rates - total_expected_events)
    return nll if np.isfinite(nll) else 1e10

# ==========================================
# 2. RUN THE TRI-MODULATED OPTIMIZATION
# ==========================================
# Parameter footprint: [A, tau, k, B, b1, b2, alpha_A, alpha_B, alpha_gamma]
initial_guesses_phase = [0.56, 1.15, 2.0, 0.40, 0.0, 0.0, -0.05, -0.05, -0.05]
bounds_phase = (
    (0.01, 10.0),    # A
    (0.1, 5.0),      # tau
    (0.4, 10.0),     # k 
    (0.01, 5.0),     # B
    (-5.0, 5.0),     # b1 
    (-5.0, 5.0),     # b2
    (-2.0, 2.0),     # alpha_A
    (-2.0, 2.0),     # alpha_B
    (-2.0, 2.0)      # alpha_gamma
)

unique_trials = np.sort(counts['trial_num'].unique())
num_fish = len(decay_events['file'].unique())

phase_result = minimize(
    phase_negative_log_likelihood_tri_modulated, 
    x0=initial_guesses_phase, 
    args=(spike_times, spike_trials, spike_sin, spike_cos, unique_trials, num_fish, t_grid, grid_sin, grid_cos, dt),
    method='L-BFGS-B',
    bounds=bounds_phase
)

A_p, tau_p, k_p, B_p, b1_p, b2_p, a_A, a_B, a_g = phase_result.x

gamma = np.sqrt(b1_p**2 + b2_p**2)
phi_pref = np.arctan2(b1_p, b2_p)
theta_pref_deg = 20.0 + 70.0 * ((1.0 - np.cos(phi_pref)) / 2.0)

# ==========================================
# 3. DIAGNOSTIC VISUALIZATION DASHBOARD (CLEANED NOTATION)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [2.8, 1.4]})

# Plot the progression of the first, middle, and final trials to visualize the decay patterns
select_trials = [unique_trials[0], unique_trials[len(unique_trials)//2], unique_trials[-1]]
colors = ['#1f77b4', '#d62728', '#2ca02c']

for tm, col in zip(select_trials, colors):
    trial_data = counts[counts['trial_num'] == tm]
    mean_data = trial_data.groupby('time_sec').agg(
        r_smooth=('jt_ipsi_hz', 'mean'), mean_sin=('mean_sin', 'mean'), mean_cos=('mean_cos', 'mean')
    )
    w_mask = (mean_data.index >= T_start) & (mean_data.index <= T_end)
    t_smooth = mean_data.index[w_mask] - T_start
    
    # Passing the trial index 'tm' cleanly
    r_model = lambda_t_phase_gamma_tri_modulated(
        t_smooth, tm, mean_data.loc[w_mask, 'mean_sin'].values, mean_data.loc[w_mask, 'mean_cos'].values,
        A_p, tau_p, k_p, B_p, b1_p, b2_p, a_A, a_B, a_g
    )
    
    ax1.plot(t_smooth, mean_data.loc[w_mask, 'r_smooth'].values, '--', color=col, alpha=0.35)
    ax1.plot(t_smooth, r_model, '-', color=col, lw=2.5, label=f'Trial {tm} Model')

ax1.set_ylabel('Bout Rate $\lambda(t, m)$  (Hz)')
ax1.set_xlabel('Time (s)')

# --- UPDATED TITLE: 'm' USED FOR TRIAL NUMBER INDEX ---
ax1.set_title(r"$\lambda(t, m) = \max\left( A e^{\alpha_A \cdot m} t^k e^{-t/\tau} + B e^{\alpha_B \cdot m} + (b_1 \sin(\phi) + b_2 \cos(\phi)) e^{\alpha_\gamma \cdot m},\, 10^{-9}\right)$", fontsize=11)

ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_ylim(bottom=0)

ax2.axis('off')

# --- UPDATED DASHBOARD TEXT: 'm' USED FOR TRIAL NUMBER INDEX ---
latex_annotation = r"""$\text{Peak Time} = \tau \cdot k$ [Transient Max Delay]  |  $\gamma = \sqrt{b_1^2 + b_2^2}$ [Phase Swing]  |  $\phi_{\text{pref}} = \text{atan2}(b_1, b_2)$ [Phase Angle]

Fitted Parameters & Baseline Variables:
$m$ = trial number (ordinal tracking index of consecutive stimulus exposures)
$A$ = """ + f"{A_p:.3f}" + r""" Hz [Scales the height of the initial stimulus-evoked peak]
$B$ = """ + f"{B_p:.3f}" + r""" Hz [Stable baseline rate floor after the initial transient activation settles down]
$2\gamma$ = """ + f"{2*gamma:.3f}" + r""" Hz [The full peak-to-trough variation window forced by target movement]
$\tau$ = """ + f"{tau_p:.3f}" + r""" sec  |  $k$ = """ + f"{k_p:.3f}" + r"""  |  Peak Time = """ + f"{tau_p * k_p:.3f}" + r""" sec
$\theta_{\text{pref}}$ = """ + f"{theta_pref_deg:.1f}" + r"""$^\circ$ [The physical position in the visual field where the target triggers the highest response probability]

Trial-by-Trial Modulators:
$\alpha_A$ = """ + f"{a_A:.4f}" + r""" [Sensory Adaptation: Rates of habituation/sensitization of the initial explosive peak over trials]
$\alpha_B$ = """ + f"{a_B:.4f}" + r""" [Baseline Motivational Shift: Long-term continuous background driving rate decay or ramp]
$\alpha_{\gamma}$ = """ + f"{a_g:.4f}" + r""" [Entrainment Tuning Change: Change in directional spatial phase response sensitivity over time]"""

ax2.text(0.01, 0.95, latex_annotation, transform=ax2.transAxes, fontsize=9.5, verticalalignment='top', horizontalalignment='left', multialignment='left')
plt.tight_layout()
plt.savefig(
    'temporal_dynamics_prey_capture_model_NHPP_ipsi_2.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_prey_capture_model_NHPP_ipsi_2.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

########################## WITH TRIALS / TWO PHASE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# 1. BI-MODAL INTENSITY & LOG-LIKELIHOOD
# ==========================================
def lambda_t_phase_gamma_tri_modulated(
    t, trial_k, s_sin, s_cos, s_sin2, s_cos2, 
    A, tau, k, B, b1, b2, b3, b4, alpha_A, alpha_B, alpha_gamma
):
    t_safe = np.maximum(t, 0)
    
    # Transient peak and static baseline remain the same
    transient_peak = (A * (t_safe ** k) * np.exp(-t_safe / tau)) * np.exp(alpha_A * trial_k)
    static_baseline = B * np.exp(alpha_B * trial_k)
    
    # Expanded Phase Ripple: Fundamental + Second Harmonic components
    phase_ripple = (
        b1 * s_sin + b2 * s_cos + 
        b3 * s_sin2 + b4 * s_cos2
    ) * np.exp(alpha_gamma * trial_k)
    
    return np.maximum(transient_peak + static_baseline + phase_ripple, 1e-9)

def phase_negative_log_likelihood_tri_modulated(
    params, t_events, trial_events, s_sin, s_cos, s_sin2, s_cos2,
    unique_trials, n_fish, t_grid, grid_sin, grid_cos, grid_sin2, grid_cos2, dt
):
    A, tau, k, B, b1, b2, b3, b4, alpha_A, alpha_B, alpha_gamma = params
    
    # Term 1: Event Log-Rates (with second-harmonic inputs)
    rates = lambda_t_phase_gamma_tri_modulated(
        t_events, trial_events, s_sin, s_cos, s_sin2, s_cos2,
        A, tau, k, B, b1, b2, b3, b4, alpha_A, alpha_B, alpha_gamma
    )
    sum_log_rates = np.sum(np.log(rates))
    
    # Term 2: Numerical Riemann Integral across all sequential trials
    t_grid_safe = np.maximum(t_grid, 0)
    
    grid_transient = A * (t_grid_safe ** k) * np.exp(-t_grid_safe / tau)
    grid_phase = b1 * grid_sin + b2 * grid_cos + b3 * grid_sin2 + b4 * grid_cos2
    
    total_expected_events = 0.0
    for tk in unique_trials:
        trial_grid_rate = np.maximum(
            grid_transient * np.exp(alpha_A * tk) + 
            B * np.exp(alpha_B * tk) + 
            grid_phase * np.exp(alpha_gamma * tk), 
            1e-9
        )
        total_expected_events += np.sum(trial_grid_rate) * dt
        
    total_expected_events *= n_fish
    
    nll = -(sum_log_rates - total_expected_events)
    return nll if np.isfinite(nll) else 1e10

# ==========================================
# 3. RUN THE OPTIMIZATION
# ==========================================
# Parameter footprint: [A, tau, k, B, b1, b2, b3, b4, alpha_A, alpha_B, alpha_gamma]
initial_guesses_phase = [0.56, 1.15, 2.0, 0.40, 0.0, 0.0, 0.0, 0.0, -0.05, -0.05, -0.05]
bounds_phase = (
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

unique_trials = np.sort(counts['trial_num'].unique())
num_fish = len(decay_events['file'].unique())

phase_result = minimize(
    phase_negative_log_likelihood_tri_modulated, 
    x0=initial_guesses_phase, 
    args=(
        spike_times, spike_trials, spike_sin, spike_cos, spike_sin2, spike_cos2,
        unique_trials, num_fish, t_grid, grid_sin, grid_cos, grid_sin2, grid_cos2, dt
    ),
    method='L-BFGS-B',
    bounds=bounds_phase
)

# Extract parameters
A_p, tau_p, k_p, B_p, b1_p, b2_p, b3_p, b4_p, a_A, a_B, a_g = phase_result.x

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
from scipy.signal import find_peaks
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
    peak_str_1 = f"{p1_deg:.1f}° ({p1_phys:.1f} °) " + "N->T" if np.sin(peak_phases[0]) > 0 else "T->N"
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
        r_smooth=('jt_ipsi_hz', 'mean'), 
        mean_sin=('mean_sin', 'mean'), 
        mean_cos=('mean_cos', 'mean'),
        mean_sin2=('mean_sin2', 'mean'), # Extracted second harmonics
        mean_cos2=('mean_cos2', 'mean')  # Extracted second harmonics
    )
    w_mask = (mean_data.index >= T_start) & (mean_data.index <= T_end)
    t_smooth = mean_data.index[w_mask] - T_start
    
    # Passing fundamental AND second-harmonic grid vectors cleanly into the bi-modal function
    r_model = lambda_t_phase_gamma_tri_modulated(
        t_smooth, tm, 
        mean_data.loc[w_mask, 'mean_sin'].values, mean_data.loc[w_mask, 'mean_cos'].values,
        mean_data.loc[w_mask, 'mean_sin2'].values, mean_data.loc[w_mask, 'mean_cos2'].values,
        A_p, tau_p, k_p, B_p, b1_p, b2_p, b3_p, b4_p, a_A, a_B, a_g
    )
    
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
################################

def run_likelihood_ratio_test(nll_null, nll_complex, df_null, df_complex):
    """
    Computes Wilks' Chi-Squared test statistic to compare nested architectures.
    """
    # Test statistic D = 2 * (LL_complex - LL_null) = 2 * (NLL_null - NLL_complex)
    dev_statistic = 2.0 * (nll_null - nll_complex)
    
    # Degrees of freedom delta
    df_delta = df_complex - df_null
    
    # Calculate the upper-tail probability (p-value) of Chi-squared distribution
    p_value = chi2.sf(dev_statistic, df_delta)
    
    print("\n==========================================")
    print("       LIKELIHOOD RATIO TEST RESULTS      ")
    print("==========================================")
    print(f"Null Model NLL:     {nll_null:.3f}  (df={df_null})")
    print(f"Complex Model NLL:  {nll_complex:.3f}  (df={df_complex})")
    print(f"Degrees of Freedom: {df_delta}")
    print(f"Chi-Square Stat (D): {dev_statistic:.3f}")
    
    if p_value < 0.001:
        print(f"LRT p-value:        p = {p_value:.5f} < 0.001 (Highly Significant) 🌟")
    else:
        print(f"LRT p-value:        p = {p_value:.5f}")
        
    if dev_statistic > 0 and p_value < 0.05:
        print("\nConclusion: Reject the Null Hypothesis. The additional parameters")
        print("provide a statistically superior description of the data.")
    else:
        print("\nConclusion: Fail to reject the Null Hypothesis. The complex parameters")
        print("do not justify the added mathematical complexity.")
    print("==========================================")
    
    return dev_statistic, p_value

