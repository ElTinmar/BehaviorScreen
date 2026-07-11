from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
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

#### MODEL

T_start = 0.0 
T_end = 25.0  
T_duration = T_end - T_start

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
    total_expected_events = N_trials * integral
    
    return -(sum_log_rates - total_expected_events)

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



# Average your rolling counts across all fish and trials to get the mean population curve
mean_rolling = counts.groupby('time_sec')['rolling_jt_ipsi'].mean()

# Isolate the time points matching your decay window
t_smoothed_window = mean_rolling.index[(mean_rolling.index >= T_start) & (mean_rolling.index <= T_end)]
smoothed_rates_window = mean_rolling.loc[t_smoothed_window]

# Generate points for your new parametric model fit
t_model = np.linspace(0, T_duration, 200)
fitted_rate = A_fit * np.exp(-t_model / tau_fit) + B_fit

# Plot
plt.figure(figsize=(9, 5))
plt.plot(t_smoothed_window - T_start, smoothed_rates_window, color='darkgray', lw=2, label='Your Rolling Average Data')
plt.plot(t_model, fitted_rate, color='crimson', lw=3, label='Poisson MLE Fit')
plt.xlabel('Time from Peak (seconds)')
plt.ylabel('Bout Rate (Hz)')
plt.title('Ipsi J-Turn Fitting Results')
plt.legend()
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
    mean_rolling = trial_data.groupby('time_sec')['rolling_jt_ipsi'].mean()
    
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
initial_guesses_phase = [0.56, 1.15, 0.40, -0.15, 0.1, 0.1]
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
        r_smooth=('rolling_jt_ipsi', 'mean'),
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



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_1samp, uniform

# ==========================================
# 1. COMPUTE TIME-RESCALED INTERVALS
# ==========================================
# Reconstruct a dataframe of events sorted chronologically within each trial
df_ev = pd.DataFrame({
    't': spike_times, 
    'k': spike_trials,
    'sin': spike_sin,
    'cos': spike_cos
}).sort_values(['k', 't'])

rescaled_intervals = []

# Loop through each trial to calculate the integral of lambda between events
for k_idx, group in df_ev.groupby('k'):
    times = group['t'].values
    if len(times) < 2:
        continue
    
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i+1]
        
        # Isolate the segment of our pre-calculated grid that falls between these two events
        grid_mask = (t_grid >= t_start) & (t_grid <= t_end)
        
        # If the interval is wider than our bin size (0.02s), integrate numerically
        if np.sum(grid_mask) > 0:
            sub_t = t_grid[grid_mask]
            sub_sin = grid_sin[grid_mask]
            sub_cos = grid_cos[grid_mask]
            
            # Evaluate lambda across this small slice
            rates_slice = lambda_t_phase(sub_t, k_idx, sub_sin, sub_cos, A_p, tau_p, B_p, alpha_p, b1_p, b2_p)
            z_i = np.sum(rates_slice) * dt
        else:
            # If the events are closer than 20ms, approximate with a single step midpoint
            mid_t = (t_start + t_end) / 2
            mid_sin = (group['sin'].iloc[i] + group['sin'].iloc[i+1]) / 2
            mid_cos = (group['cos'].iloc[i] + group['cos'].iloc[i+1]) / 2
            z_i = lambda_t_phase(mid_t, k_idx, mid_sin, mid_cos, A_p, tau_p, B_p, alpha_p, b1_p, b2_p) * (t_end - t_start)
            
        rescaled_intervals.append(z_i)

# Transform rescaled intervals to a uniform distribution: u_i = 1 - exp(-z_i)
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