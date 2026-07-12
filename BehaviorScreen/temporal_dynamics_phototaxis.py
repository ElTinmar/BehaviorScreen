from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short
from scipy.optimize import minimize


rt_index = bouts_category_name_short.index("RT")
hat_index = bouts_category_name_short.index("HAT")

ROOT = Path('/media/martin/DATA_18TB/Screen')
df = pd.read_csv(ROOT / 'bouts_control.csv')


print(f"#fish: {len(df.file.unique())}")

fine_dt = 0.02
time_bins = np.arange(0, 25 + fine_dt, fine_dt)
window_duration = 0.5
window_size_steps = int(window_duration / fine_dt)
window_size_steps |= 1

sub_df = df[df['stim'] == Stim.PHOTOTAXIS].copy()

print(sub_df['foreground_color'].unique())
sub_df = sub_df[sub_df['foreground_color'] == '[0.1, 0.1, 0.0, 1.0]']

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
counts['rt_ipsi_hz'] = counts['rolling_rt_ipsi'] / window_duration
counts['rt_contra_hz'] = counts['rolling_rt_contra'] / window_duration
counts['asymmetry_index'] = (counts['rt_ipsi_hz'] - counts['rt_contra_hz']) / (counts['rt_ipsi_hz'] + counts['rt_contra_hz'])

trial_averaged = (
    counts.groupby(['file', 'time_bin'], observed=False)
    .agg(
        rt_ipsi_hz=('rt_ipsi_hz', 'mean'),
        rt_contra_hz=('rt_contra_hz', 'mean'),
        asymmetry_index=('asymmetry_index', 'mean')
    )
    .reset_index()
)
trial_averaged['time_sec'] = trial_averaged['time_bin'].apply(lambda x: x.left).astype(float)

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
    'temporal_dynamics_phototaxis.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_phototaxis.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()


### RATIO

fig = plt.figure(figsize=(12, 10))
ax = fig.gca()

sns.lineplot(
    data=counts, x='time_sec', y='asymmetry_index',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax, zorder=3,
)
ax.set_xlabel('Trial Time (s)', fontsize=12)
ax.set_ylabel('Asymmetry Index\n(← dark | bright →)', fontsize=12, fontweight='bold')
ax.set_ylim(-1.05, 1.05)
ax.grid(True, linestyle=':', alpha=0.5)
ax.axhline(0.0, color='grey', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_phototaxis_index.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_phototaxis_index.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

##### AVERAGE TRIALS

fig, (ax_ipsi, ax_contra) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, sharey=True)

sns.lineplot(
    data=trial_averaged, x='time_sec', y='rt_ipsi_hz', 
    errorbar='se', palette='viridis', ax=ax_ipsi, zorder=3
)
ax_ipsi.set_ylabel('Ipsi RT+HAT turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_ipsi.grid(True, linestyle=':', alpha=0.5)

sns.lineplot(
    data=trial_averaged, x='time_sec', y='rt_contra_hz',
    errorbar='se', palette='viridis', ax=ax_contra, zorder=3,
    legend=False # Suppress duplicate legend
)
ax_contra.set_xlabel('Trial Time (s)', fontsize=12)
ax_contra.set_ylabel('Contra RT+HAT Frequency (Hz)', fontsize=12, fontweight='bold')
ax_contra.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_phototaxis_avg.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_phototaxis_avg.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

### RATIO

fig = plt.figure(figsize=(12, 10))
ax = fig.gca()

sns.lineplot(
    data=trial_averaged, x='time_sec', y='asymmetry_index',
    errorbar='se', palette='viridis', ax=ax, zorder=3,
)
ax.set_xlabel('Trial Time (s)', fontsize=12)
ax.set_ylabel('Asymmetry Index\n(← dark | bright →)', fontsize=12, fontweight='bold')
ax.set_ylim(-1.05, 1.05)
ax.grid(True, linestyle=':', alpha=0.5)
ax.axhline(0.0, color='grey', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_phototaxis_index.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_phototaxis_index.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()



#################
#
#
#   MODEL
#
#
#################

T_start = 0.0 
T_end = 24.0  
T_duration = T_end - T_start

events = sub_df[sub_df['ipsi_rt'] & (sub_df['trial_time'] >= T_start) & (sub_df['trial_time'] <= T_end)]
spike_times = (events['trial_time'] - T_start).values
spike_trials = events['trial_num'].values
num_trials = len(sub_df.groupby(['file', 'trial_num']))

# ==========================================
# DIFFERENCE OF EXPONENTIALS FUNCTIONS
# ==========================================
def lambda_t_diff_exp(t, B, A_dip, tau_dip, A_peak, tau_peak):
    t_safe = np.maximum(t, 0)
    
    # Core dual-exponential engine
    dip = A_dip * np.exp(-t_safe / tau_dip)
    peak = A_peak * np.exp(-t_safe / tau_peak)
    
    rate = B - dip + peak
    return np.maximum(rate, 1e-9)

def negative_log_likelihood_diff_exp(params, t_events, T, N_trials):
    B, A_dip, tau_dip, A_peak, tau_peak = params
    
    # Term 1: Log-rates at exact event arrival times
    rates = lambda_t_diff_exp(t_events, B, A_dip, tau_dip, A_peak, tau_peak)
    log_term = np.sum(np.log(rates))
    
    # Term 2: Analytical Integral of Exponentials over [0, T]
    # Integral of e^(-t/tau) from 0 to T is: tau * (1 - e^(-T/tau))
    integral_dip = A_dip * tau_dip * (1.0 - np.exp(-T / tau_dip))
    integral_peak = A_peak * tau_peak * (1.0 - np.exp(-T / tau_peak))
    
    total_integral = (B * T) - integral_dip + integral_peak
    
    return -(log_term - N_trials * total_integral)

# ==========================================
# FIT THE MODEL
# ==========================================
# Initial guesses: [B, A_dip, tau_dip, A_peak, tau_peak]
initial_guesses = [0.4, 0.5, 0.5, 0.5, 3.0]

# Enforce nesting rules and boundary conditions via parameter constraints
bounds = (
    (0.01, 5.0),    # B (Baseline floor)
    (0.0, 5.0),     # A_dip 
    (0.01, 5.0),    # tau_dip (Fast suppression time constant)
    (0.0, 5.0),     # A_peak
    (0.1, 15.0)     # tau_peak (Slower execution decay time constant)
)

result_diff_exp = minimize(
    negative_log_likelihood_diff_exp,
    x0=initial_guesses,
    args=(spike_times, T_duration, num_trials),
    method='L-BFGS-B',
    bounds=bounds
)

B_f, Ad_f, taud_f, Ap_f, taup_f = result_diff_exp.x

# Analytical calculations for starting rate value at t=0
initial_rate = B_f - Ad_f + Ap_f

# ==========================================
# GENERATE PLOT AND DASHBOARD
# ==========================================
mean_rolling = counts.groupby('time_sec')['rt_ipsi_hz'].mean()
t_smoothed_window = mean_rolling.index[(mean_rolling.index >= T_start) & (mean_rolling.index <= T_end)]
smoothed_rates_window = mean_rolling.loc[t_smoothed_window]
t_model = np.linspace(0, T_duration, 500)
fitted_rate = lambda_t_diff_exp(t_model, B_f, Ad_f, taud_f, Ap_f, taup_f)

fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(10, 8), 
    gridspec_kw={"height_ratios": [3, 1.3]}
)

# ---- Top Panel: Data Plot ----
ax1.plot(t_smoothed_window - T_start, smoothed_rates_window, color="darkgray", lw=2, label="Rolling average")
ax1.plot(t_model, fitted_rate, color="crimson", lw=3, label="Difference of Exponentials MLE")
ax1.axhline(B_f, color="blue", ls=":", alpha=0.7, label="Fitted Baseline (B)")
ax1.set_xlabel("Time from stimulus (s)")
ax1.set_ylabel(r"Bout rate $\lambda(t)$ (Hz)")
ax1.set_ylim(bottom=0)
ax1.set_title(r"$\lambda(t) = \max \left( B - A_{\text{dip}} e^{-t/\tau_{\text{dip}}} + A_{\text{peak}} e^{-t/\tau_{\text{peak}}} ,\, 10^{-9} \right)$", fontsize=11)
ax1.grid(alpha=0.25)
ax1.legend()

# ---- Bottom Panel: Dash Annotations ----
ax2.axis("off")
annotation = rf"""
Fitted Difference of Exponentials Parameters:

Baseline (B)        = {B_f:.3f} Hz  [The continuous background driving rate floor]
Initial Rate (t=0)  = {initial_rate:.3f} Hz  [Calculated via: B - A_dip + A_peak]

Dip Module (Initial Suppression):
Dip Amplitude (A)   = {Ad_f:.3f} Hz  [Weight of the transient suppression engine]
Dip Time Const (τ)  = {taud_f:.3f} s   [Decay rate of the suppression phase]

Peak Module (Behavioral Overshoot):
Peak Amplitude (A)  = {Ap_f:.3f} Hz  [Weight of the secondary activation engine]
Peak Time Const (τ) = {taup_f:.3f} s   [Decay rate of the behavioral activation surge]
"""

ax2.text(
    0.01, 0.98, 
    annotation, 
    transform=ax2.transAxes, 
    va="top", 
    ha="left", 
    fontsize=10, 
    family="monospace"
)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_phototaxis_model_NHPP_ipsi_0.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_phototaxis_model_NHPP_ipsi_0.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()


#### CONTRA


T_start = 0.0 
T_end = 24.0  
T_duration = T_end - T_start

events = sub_df[sub_df['contra_rt'] & (sub_df['trial_time'] >= T_start) & (sub_df['trial_time'] <= T_end)]
spike_times = (events['trial_time'] - T_start).values
spike_trials = events['trial_num'].values
num_trials = len(sub_df.groupby(['file', 'trial_num']))

# ==========================================
# EXPONENTIALS FUNCTIONS
# ==========================================
def lambda_t_diff_exp(t, B, A_dip, tau_dip):
    t_safe = np.maximum(t, 0)
    
    dip = A_dip * np.exp(-t_safe / tau_dip)
    
    rate = B - dip
    return np.maximum(rate, 1e-9)

def negative_log_likelihood_diff_exp(params, t_events, T, N_trials):
    B, A_dip, tau_dip,  = params
    
    # Term 1: Log-rates at exact event arrival times
    rates = lambda_t_diff_exp(t_events, B, A_dip, tau_dip)
    log_term = np.sum(np.log(rates))
    
    # Term 2: Analytical Integral of Exponentials over [0, T]
    # Integral of e^(-t/tau) from 0 to T is: tau * (1 - e^(-T/tau))
    integral_dip = A_dip * tau_dip * (1.0 - np.exp(-T / tau_dip))
    
    total_integral = (B * T) - integral_dip
    
    return -(log_term - N_trials * total_integral)

# ==========================================
# FIT THE MODEL
# ==========================================
# Initial guesses: [B, A_dip, tau_dip]
initial_guesses = [0.4, 0.5, 0.5]

# Enforce nesting rules and boundary conditions via parameter constraints
bounds = (
    (0.01, 5.0),    # B (Baseline floor)
    (0.0, 5.0),     # A_dip 
    (0.01, 5.0),    # tau_dip (Fast suppression time constant)
)

result_diff_exp = minimize(
    negative_log_likelihood_diff_exp,
    x0=initial_guesses,
    args=(spike_times, T_duration, num_trials),
    method='L-BFGS-B',
    bounds=bounds
)

B_f, Ad_f, taud_f = result_diff_exp.x

# Analytical calculations for starting rate value at t=0
initial_rate = B_f - Ad_f 

# ==========================================
# GENERATE PLOT AND DASHBOARD
# ==========================================
mean_rolling = counts.groupby('time_sec')['rt_contra_hz'].mean()
t_smoothed_window = mean_rolling.index[(mean_rolling.index >= T_start) & (mean_rolling.index <= T_end)]
smoothed_rates_window = mean_rolling.loc[t_smoothed_window]
t_model = np.linspace(0, T_duration, 500)
fitted_rate = lambda_t_diff_exp(t_model, B_f, Ad_f, taud_f)

fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(10, 8), 
    gridspec_kw={"height_ratios": [3, 1.3]}
)

# ---- Top Panel: Data Plot ----
ax1.plot(t_smoothed_window - T_start, smoothed_rates_window, color="darkgray", lw=2, label="Rolling average")
ax1.plot(t_model, fitted_rate, color="crimson", lw=3, label="Difference of Exponentials MLE")
ax1.axhline(B_f, color="blue", ls=":", alpha=0.7, label="Fitted Baseline (B)")
ax1.set_xlabel("Time from stimulus (s)")
ax1.set_ylabel(r"Bout rate $\lambda(t)$ (Hz)")
ax1.set_ylim(bottom=0)
ax1.set_title(r"$\lambda(t) = \max \left( B - A_{\text{dip}} e^{-t/\tau_{\text{dip}}} ,\, 10^{-9} \right)$", fontsize=11)
ax1.grid(alpha=0.25)
ax1.legend()

# ---- Bottom Panel: Dash Annotations ----
ax2.axis("off")
annotation = rf"""
Fitted Difference of Exponentials Parameters:

Baseline (B)        = {B_f:.3f} Hz  [The continuous background driving rate floor]
Initial Rate (t=0)  = {initial_rate:.3f} Hz  [Calculated via: B - A_dip]

Dip Module (Initial Suppression):
Dip Amplitude (A)   = {Ad_f:.3f} Hz  [Weight of the transient suppression engine]
Dip Time Const (τ)  = {taud_f:.3f} s   [Decay rate of the suppression phase]

"""

ax2.text(
    0.01, 0.98, 
    annotation, 
    transform=ax2.transAxes, 
    va="top", 
    ha="left", 
    fontsize=10, 
    family="monospace"
)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_phototaxis_model_NHPP_contra_0.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_phototaxis_model_NHPP_contra_0.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()


###### adding trials

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# PREPARE TRIAL-AWARE DATA
# ==========================================
T_start = 0.0
T_end = 24.0
T_duration = T_end - T_start

events = sub_df[
    sub_df['ipsi_rt']
    & (sub_df['trial_time'] >= T_start)
    & (sub_df['trial_time'] <= T_end)
]

# Extract exact timestamps AND corresponding trial indices
spike_times = (events['trial_time'] - T_start).values
spike_trials = events['trial_num'].values

unique_trials = np.sort(sub_df['trial_num'].unique())
num_trials = len(sub_df.groupby(['file', 'trial_num']))

# ==========================================
# TRIAL-MODULATED DIFFERENCE OF EXPONENTIALS
# ==========================================
def lambda_t_diff_exp_trials(t, m, B, A_dip, tau_dip, A_peak, tau_peak, alpha_B, alpha_dip, alpha_peak):
    t_safe = np.maximum(t, 0)
    
    # Each module scales exponentially based on its alpha parameter and trial number m
    mod_B = B * np.exp(alpha_B * m)
    mod_dip = A_dip * np.exp(alpha_dip * m) * np.exp(-t_safe / tau_dip)
    mod_peak = A_peak * np.exp(alpha_peak * m) * np.exp(-t_safe / tau_peak)
    
    rate = mod_B - mod_dip + mod_peak
    return np.maximum(rate, 1e-9)

def negative_log_likelihood_diff_exp_trials(params, t_events, m_events, T, unique_trial_set, N_total_trials):
    B, A_dip, tau_dip, A_peak, tau_peak, alpha_B, alpha_dip, alpha_peak = params
    
    # Term 1: Log-rates at exact event arrival times considering their specific trial numbers
    rates = lambda_t_diff_exp_trials(t_events, m_events, B, A_dip, tau_dip, A_peak, tau_peak, alpha_B, alpha_dip, alpha_peak)
    log_term = np.sum(np.log(rates))
    
    # Term 2: Analytical Integral calculated per unique trial and summed up across the total tracking space
    # Scale factor adjusts for multiple fish per trial block
    fish_per_trial = N_total_trials / len(unique_trial_set)
    
    total_integral = 0.0
    for m in unique_trial_set:
        integral_dip = A_dip * np.exp(alpha_dip * m) * tau_dip * (1.0 - np.exp(-T / tau_dip))
        integral_peak = A_peak * np.exp(alpha_peak * m) * tau_peak * (1.0 - np.exp(-T / tau_peak))
        integral_B = B * np.exp(alpha_B * m) * T
        
        total_integral += (integral_B - integral_dip + integral_peak) * fish_per_trial
        
    return -(log_term - total_integral)

# ==========================================
# FIT THE MODULATED MODEL
# ==========================================
# Guesses: [B, A_dip, tau_dip, A_peak, tau_peak, alpha_B, alpha_dip, alpha_peak]
initial_guesses = [0.4, 0.5, 0.5, 0.5, 3.0, 0.0, 0.0, 0.0]

bounds = (
    (0.01, 5.0),     # B
    (0.0, 5.0),      # A_dip
    (0.01, 5.0),     # tau_dip
    (0.0, 5.0),      # A_peak
    (0.1, 15.0),     # tau_peak
    (-0.1, 0.1),     # alpha_B (Baseline drift window)
    (-0.1, 0.1),     # alpha_dip (Suppression adaptation window)
    (-0.1, 0.1)      # alpha_peak (Overshoot habituation window)
)

result_trials = minimize(
    negative_log_likelihood_diff_exp_trials,
    x0=initial_guesses,
    args=(spike_times, spike_trials, T_duration, unique_trials, num_trials),
    method='L-BFGS-B',
    bounds=bounds
)

B_f, Ad_f, taud_f, Ap_f, taup_f, aB_f, adip_f, apeak_f = result_trials.x

# ==========================================
# GENERATE DISPLAY PLOT (SHOWING PROGRESSION)
# ==========================================
mean_rolling = counts.groupby(['time_sec', 'trial_num'])['rt_ipsi_hz'].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2.8, 1.4]})

# Visualize specific early, middle, and late trials to see adaptation changes clearly
select_trials = [unique_trials[0], unique_trials[len(unique_trials)//2], unique_trials[-1]]
colors = ['#1f77b4', '#d62728', '#2ca02c']
t_model = np.linspace(0, T_duration, 500)

for tm, col in zip(select_trials, colors):
    # Plot real rolling average per chosen trial
    trial_data = mean_rolling[mean_rolling['trial_num'] == tm]
    ax1.plot(trial_data['time_sec'] - T_start, trial_data['rt_ipsi_hz'], '--', color=col, alpha=0.3)
    
    # Plot matching model projection
    fitted_rate = lambda_t_diff_exp_trials(t_model, tm, B_f, Ad_f, taud_f, Ap_f, taup_f, aB_f, adip_f, apeak_f)
    ax1.plot(t_model, fitted_rate, '-', color=col, lw=2.5, label=f'Trial {tm} Model')

ax1.set_xlabel("Time from stimulus (s)")
ax1.set_ylabel(r"Bout rate $\lambda(t, m)$ (Hz)")
ax1.set_ylim(bottom=0)
ax1.set_title(r"$\lambda(t, m) = \max \left( B e^{\alpha_B \cdot m} - A_{\text{dip}} e^{\alpha_{\text{dip}} \cdot m} e^{-t/\tau_{\text{dip}}} + A_{\text{peak}} e^{\alpha_{\text{peak}} \cdot m} e^{-t/\tau_{\text{peak}}} ,\, 10^{-9} \right)$", fontsize=11)
ax1.grid(alpha=0.2)
ax1.legend(loc='upper right')

# ---- Bottom Panel: Dash Annotations ----
ax2.axis("off")

# Clean documentation utilizing direct variable formatting drops
latex_annotation = r"""Fitted Model Parameters & Trial Modulators:
$m$ = trial number (ordinal tracking index)
$B$ = """ + f"{B_f:.3f}" + r""" Hz  |  $A_{\text{dip}}$ = """ + f"{Ad_f:.3f}" + r""" Hz  |  $A_{\text{peak}}$ = """ + f"{Ap_f:.3f}" + r""" Hz
$\tau_{\text{dip}}$ = """ + f"{taud_f:.3f}" + r""" s [Suppression Phase Const]  |  $\tau_{\text{peak}}$ = """ + f"{taup_f:.3f}" + r""" s [Activation Decay Const]

Trial-by-Trial Modulators (Alpha Scales):
$\alpha_B$    = """ + f"{aB_f:.4f}" + r""" [Baseline Shift: Continuous background driving floor adaptation]
$\alpha_{\text{dip}}$  = """ + f"{adip_f:.4f}" + r""" [Latency/Suppression Change: Adaptation of initial suppression magnitude]
$\alpha_{\text{peak}}$ = """ + f"{apeak_f:.4f}" + r""" [Sensory Habituation: Changes in peak overshoot response power over time]"""

ax2.text(0.01, 0.95, latex_annotation, transform=ax2.transAxes, fontsize=9.5, va="top", ha="left", multialignment="left")

plt.tight_layout()
plt.savefig('temporal_dynamics_phototaxis_model_NHPP_ipsi_1.svg', bbox_inches='tight')
plt.savefig('temporal_dynamics_phototaxis_model_NHPP_ipsi_1.png', dpi=300, bbox_inches='tight')
plt.show()


#####

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# PREPARE TRIAL-AWARE CONTRALATERAL DATA
# ==========================================
T_start = 0.0
T_end = 24.0
T_duration = T_end - T_start

# Note the switch to contra_rt as requested by your pipeline modifications
events = sub_df[
    sub_df['contra_rt']
    & (sub_df['trial_time'] >= T_start)
    & (sub_df['trial_time'] <= T_end)
]

# Extract timestamps AND corresponding trial indices
spike_times = (events['trial_time'] - T_start).values
spike_trials = events['trial_num'].values

unique_trials = np.sort(sub_df['trial_num'].unique())
num_trials = len(sub_df.groupby(['file', 'trial_num']))

# ==========================================
# TRIAL-MODULATED SINGLE EXPONENTIAL DIP
# ==========================================
def lambda_t_dip_trials(t, m, B, A_dip, tau_dip, alpha_B, alpha_dip):
    t_safe = np.maximum(t, 0)
    
    # Baseline and dip components scale exponentially with trial index m
    mod_B = B * np.exp(alpha_B * m)
    mod_dip = A_dip * np.exp(alpha_dip * m) * np.exp(-t_safe / tau_dip)
    
    rate = mod_B - mod_dip
    return np.maximum(rate, 1e-9)

def negative_log_likelihood_dip_trials(params, t_events, m_events, T, unique_trial_set, N_total_trials):
    B, A_dip, tau_dip, alpha_B, alpha_dip = params
    
    # Term 1: Log-rates at exact event arrival times based on specific trial conditions
    rates = lambda_t_dip_trials(t_events, m_events, B, A_dip, tau_dip, alpha_B, alpha_dip)
    log_term = np.sum(np.log(rates))
    
    # Term 2: Analytical Integral over [0, T] calculated per unique trial
    fish_per_trial = N_total_trials / len(unique_trial_set)
    
    total_integral = 0.0
    for m in unique_trial_set:
        integral_B = B * np.exp(alpha_B * m) * T
        integral_dip = A_dip * np.exp(alpha_dip * m) * tau_dip * (1.0 - np.exp(-T / tau_dip))
        
        total_integral += (integral_B - integral_dip) * fish_per_trial
        
    return -(log_term - total_integral)

# ==========================================
# FIT THE MODULATED DIP MODEL
# ==========================================
# Guesses: [B, A_dip, tau_dip, alpha_B, alpha_dip]
initial_guesses = [0.4, 0.3, 0.5, 0.0, 0.0]

bounds = (
    (0.01, 5.0),     # B
    (0.0, 5.0),      # A_dip
    (0.01, 5.0),     # tau_dip
    (-0.1, 0.1),     # alpha_B (Baseline drift window)
    (-0.1, 0.1)      # alpha_dip (Suppression depth adaptation window)
)

result_trials_contra = minimize(
    negative_log_likelihood_dip_trials,
    x0=initial_guesses,
    args=(spike_times, spike_trials, T_duration, unique_trials, num_trials),
    method='L-BFGS-B',
    bounds=bounds
)

B_f, Ad_f, taud_f, aB_f, adip_f = result_trials_contra.x

# ==========================================
# GENERATE PROGRESSION VISUALIZATION
# ==========================================
# Grouping by rt_contra_hz to match your target laterality
mean_rolling = counts.groupby(['time_sec', 'trial_num'])['rt_contra_hz'].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2.8, 1.4]})

# Visualize specific early, middle, and late trials to capture adaptation behavior
select_trials = [unique_trials[0], unique_trials[len(unique_trials)//2], unique_trials[-1]]
colors = ['#1f77b4', '#d62728', '#2ca02c']
t_model = np.linspace(0, T_duration, 500)

for tm, col in zip(select_trials, colors):
    # Plot real contra rolling average for chosen trial index
    trial_data = mean_rolling[mean_rolling['trial_num'] == tm]
    ax1.plot(trial_data['time_sec'] - T_start, trial_data['rt_contra_hz'], '--', color=col, alpha=0.3)
    
    # Plot model prediction profile matching the trial state
    fitted_rate = lambda_t_dip_trials(t_model, tm, B_f, Ad_f, taud_f, aB_f, adip_f)
    ax1.plot(t_model, fitted_rate, '-', color=col, lw=2.5, label=f'Trial {tm} Model')

ax1.set_xlabel("Time from stimulus (s)")
ax1.set_ylabel(r"Contra Bout rate $\lambda(t, m)$ (Hz)")
ax1.set_ylim(bottom=0)
ax1.set_title(r"$\lambda(t, m) = \max \left( B e^{\alpha_B \cdot m} - A_{\text{dip}} e^{\alpha_{\text{dip}} \cdot m} e^{-t/\tau_{\text{dip}}} ,\, 10^{-9} \right)$", fontsize=11)
ax1.grid(alpha=0.2)
ax1.legend(loc='upper right')

# ---- Bottom Panel: Dash Annotations ----
ax2.axis("off")

# Complete descriptive breakdown block
latex_annotation = r"""Fitted Model Parameters & Trial Modulators (Contralateral Side):
$m$ = trial number (ordinal tracking index)
$B$ = """ + f"{B_f:.3f}" + r""" Hz [Base Baseline Floor]  |  $A_{\text{dip}}$ = """ + f"{Ad_f:.3f}" + r""" Hz [Base Suppression Depth]
$\tau_{\text{dip}}$ = """ + f"{taud_f:.3f}" + r""" s [Suppression Recovery Decay Constant]

Trial-by-Trial Modulators (Alpha Scales):
$\alpha_B$    = """ + f"{aB_f:.4f}" + r""" [Baseline Shift: Continuous background driving rate change over trials]
$\alpha_{\text{dip}}$  = """ + f"{adip_f:.4f}" + r""" [Suppression Adaptation: Habituation/sensitization of stimulus-driven inhibition over trials]"""

ax2.text(0.01, 0.95, latex_annotation, transform=ax2.transAxes, fontsize=9.5, va="top", ha="left", multialignment="left")

plt.tight_layout()
plt.savefig('temporal_dynamics_phototaxis_model_NHPP_contra_1.svg', bbox_inches='tight')
plt.savefig('temporal_dynamics_phototaxis_model_NHPP_contra_1.png', dpi=300, bbox_inches='tight')
plt.show()