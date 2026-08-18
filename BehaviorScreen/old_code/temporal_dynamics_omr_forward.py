from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from megabouts.utils import bouts_category_name_short
from scipy.optimize import minimize


s1_index = bouts_category_name_short.index("S1")
s2_index = bouts_category_name_short.index("S2")
bs_index = bouts_category_name_short.index("BS")

ROOT = Path('/media/martin/DATA_18TB/Screen')
df = pd.read_csv(ROOT / 'bouts_control.csv')

print(f"#fish: {len(df.file.unique())}")

fine_dt = 0.02
time_bins = np.arange(0, 10 + fine_dt, fine_dt)
window_duration = 0.33
window_size_steps = int(window_duration / fine_dt)
window_size_steps |= 1

sub_df = df[df['epoch_name'] == "grating forward"].copy()

sub_df['s1'] = sub_df['category'] == s1_index
sub_df['s2'] = sub_df['category'] == s2_index
sub_df['bs'] = sub_df['category'] == bs_index
sub_df['time_bin'] = pd.cut(sub_df['trial_time'], bins=time_bins, right=False)


# We leave the data inside the MultiIndex structure (no .reset_index() yet) 
# to protect the execution order of the upcoming rolling window.
counts = (
    sub_df.groupby(['file', 'trial_num', 'time_bin'], observed=False)
    .agg(
        s1_count=('s1', 'sum'),
        s2_count=('s2', 'sum'),
        bs_count=('bs', 'sum'),
    )
)

# Apply centered rolling sum across the timeline per fish/trial
counts['rolling_s1'] = (
    counts.groupby(['file', 'trial_num'])['s1_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

counts['rolling_s2'] = (
    counts.groupby(['file', 'trial_num'])['s2_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

counts['rolling_bs'] = (
    counts.groupby(['file', 'trial_num'])['bs_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

# Flatten the MultiIndex now that the time-sensitive rolling math is finished
counts = counts.reset_index()

# Convert the interval objects into floating-point seconds for the X-axis
counts['time_sec'] = counts['time_bin'].apply(lambda x: x.left).astype(float)

# Calculate true instantaneous frequency (Bouts per second)
counts['s1_hz'] = counts['rolling_s1'] / window_duration
counts['s2_hz'] = counts['rolling_s2'] / window_duration
counts['bs_hz'] = counts['rolling_bs'] / window_duration


fig, (ax_s1, ax_s2, ax_bs) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True, sharey=True)

sns.lineplot(
    data=counts, x='time_sec', y='s1_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_s1, zorder=3
)
ax_s1.set_ylabel('S1 Frequency (Hz)', fontsize=12, fontweight='bold')
ax_s1.grid(True, linestyle=':', alpha=0.5)

sns.lineplot(
    data=counts, x='time_sec', y='s2_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_s2, zorder=3,
    legend=False # Suppress duplicate legend
)
ax_s2.set_ylabel('S2 Frequency (Hz)', fontsize=12, fontweight='bold')
ax_s2.grid(True, linestyle=':', alpha=0.5)

sns.lineplot(
    data=counts, x='time_sec', y='bs_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_bs, zorder=3,
    legend=False # Suppress duplicate legend
)
ax_bs.set_xlabel('Trial Time (s)', fontsize=12)
ax_bs.set_ylabel('BS Frequency (Hz)', fontsize=12, fontweight='bold')
ax_bs.grid(True, linestyle=':', alpha=0.5)


plt.tight_layout()
plt.savefig(
    'temporal_dynamics_omr_forward.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_omr_forward.png',
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
T_end = 10.0  
T_duration = T_end - T_start

events = sub_df[sub_df['bs'] & (sub_df['trial_time'] >= T_start) & (sub_df['trial_time'] <= T_end)]
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
mean_rolling = counts.groupby('time_sec')['bs_hz'].mean()
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
    'temporal_dynamics_omr_forward_NHPP_contra_0.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_omr_forward_NHPP_contra_0.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()
