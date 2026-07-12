from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BehaviorScreen.core import Stim, Laterality
from megabouts.utils import bouts_category_name_short

llc_index = bouts_category_name_short.index("LLC")
slc_index = bouts_category_name_short.index("SLC")

ROOT = Path('/media/martin/DATA_18TB/Screen')
df = pd.read_csv(ROOT / 'bouts_control.csv')

print(f"#fish: {len(df.file.unique())}")

fine_dt = 0.02
time_bins = np.arange(0, 10 + fine_dt, fine_dt)
window_duration = 0.175
window_size_steps = int(window_duration / fine_dt)
window_size_steps |= 1

sub_df = df[df['stim'] == Stim.LOOMING].copy()
sub_df = sub_df[sub_df['trial_time'] < 10] # bug in 00_07dpf_WT_Wed_17_Dec_2025_11h28min24sec

sub_df['ipsi_escape'] = ((sub_df['category'] == llc_index) | (sub_df['category'] == slc_index)) & (sub_df['laterality'] == Laterality.IPSILATERAL)
sub_df['contra_escape'] = ((sub_df['category'] == llc_index) | (sub_df['category'] == slc_index)) & (sub_df['laterality'] == Laterality.CONTRALATERAL)
sub_df['time_bin'] = pd.cut(sub_df['trial_time'], bins=time_bins, right=False)


# We leave the data inside the MultiIndex structure (no .reset_index() yet) 
# to protect the execution order of the upcoming rolling window.
counts = (
    sub_df.groupby(['file', 'trial_num', 'time_bin'], observed=False)
    .agg(
        escape_ipsi_count=('ipsi_escape', 'sum'),
        escape_contra_count=('contra_escape', 'sum'),
    )
)

# Apply centered rolling sum across the timeline per fish/trial
counts['rolling_escape_ipsi'] = (
    counts.groupby(['file', 'trial_num'])['escape_ipsi_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

counts['rolling_escape_contra'] = (
    counts.groupby(['file', 'trial_num'])['escape_contra_count']
    .transform(lambda x: x.rolling(window=window_size_steps, min_periods=1, center=True).sum())
)

# Flatten the MultiIndex now that the time-sensitive rolling math is finished
counts = counts.reset_index()

# Conveescape the interval objects into floating-point seconds for the X-axis
counts['time_sec'] = counts['time_bin'].apply(lambda x: x.left).astype(float)

# Calculate true instantaneous frequency (Bouts per second)
counts['escape_ipsi_hz'] = counts['rolling_escape_ipsi'] / window_duration
counts['escape_contra_hz'] = counts['rolling_escape_contra'] / window_duration

fig, (ax_ipsi, ax_contra) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, sharey=True)

sns.lineplot(
    data=counts, x='time_sec', y='escape_ipsi_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_ipsi, zorder=3
)
ax_ipsi.set_ylabel('Ipsi LLC+SLC turn Frequency (Hz)', fontsize=12, fontweight='bold')
ax_ipsi.grid(True, linestyle=':', alpha=0.5)

sns.lineplot(
    data=counts, x='time_sec', y='escape_contra_hz',
    hue='trial_num', errorbar='se', palette='viridis', ax=ax_contra, zorder=3,
    legend=False # Suppress duplicate legend
)
ax_contra.set_xlabel('Trial Time (s)', fontsize=12)
ax_contra.set_ylabel('Contra LLC+SLC Frequency (Hz)', fontsize=12, fontweight='bold')
ax_contra.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_looming.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_looming.png',
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

T_start = 4.5
T_end = 5.25  
T_duration = T_end - T_start


######################### homogeneous poisson process  ########################

### IPSI

events = sub_df[sub_df['ipsi_escape'] & (sub_df['trial_time'] >= T_start) & (sub_df['trial_time'] <= T_end)]
spike_times = (events['trial_time'] - T_start).values
num_trials = len(sub_df.groupby(['file', 'trial_num']))

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
mean_rolling = counts.groupby('time_sec')['escape_ipsi_hz'].mean()

# Isolate the time points matching your decay window
#t_smoothed_window = mean_rolling.index[(mean_rolling.index >= T_start) & (mean_rolling.index <= T_end)]
#smoothed_rates_window = mean_rolling.loc[t_smoothed_window]

# Generate points for the flat, constant HPP baseline rate
t_model_hpp = np.linspace(T_start, T_end, 200)
fitted_rate_hpp = np.full_like(t_model_hpp, C_mle)

# Create a 2-row layout matching your standard dashboard framework
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(10, 8), 
    gridspec_kw={'height_ratios': [3, 1.2]},
    sharex=False
)

# --- TOP ROW: DATA AXIS ---
ax1.plot(mean_rolling.index, mean_rolling, color='darkgray', lw=2, label='Your Rolling Average Data')
ax1.plot(t_model_hpp , fitted_rate_hpp, color='royalblue', lw=3, ls='--', label='HPP Null Model Fit')
ax1.set_ylabel('Ipsilateral escape rate $\lambda(t)$  (Hz)')
ax1.set_xlabel('Time from Peak (seconds)')
ax1.set_title(r"$\lambda(t) = C$ (Constant Rate Assumption)", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)

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
    'temporal_dynamics_looming_model_HPP_ipsi.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_looming_model_HPP_ipsi.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

##############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==============================================================================
# 1. THE SEPARATE GAUSSIAN RATE FUNCTION
# ==============================================================================
def gaussian_rate_with_baseline(t, baseline, amplitude_A, mu_location, sigma_scale):
    """
    Computes lambda(t) using a symmetric Gaussian profile plus a background baseline.
    
    Units:
      t: seconds
      baseline: Hz
      amplitude_A: Hz 
      mu_location: seconds (peak timing)
      sigma_scale: seconds (burst width)
    """
    if sigma_scale <= 0 or baseline < 0 or amplitude_A <= 0:
        return np.full_like(t, baseline) if isinstance(t, np.ndarray) else baseline
        
    # Standard Gaussian PDF formula
    gaussian_pdf = np.exp(-0.5 * ((t - mu_location) / sigma_scale)**2)
    
    return baseline + (amplitude_A * gaussian_pdf)

# ==============================================================================
# 2. DATA PROCESSING & INITIALIZATION
# ==============================================================================
# Extract raw event times from your global experimental framework
global_events = sub_df[sub_df['ipsi_escape']]
spike_times = global_events['trial_time'].values
num_trials = len(sub_df.groupby(['file', 'trial_num']))
T_total = 10.0

# Broad empirical initial guesses derived directly from the data layout
spontaneous_spikes_early = np.sum(spike_times < 2.0)
baseline_init = spontaneous_spikes_early / (num_trials * 2.0)
amplitude_init = len(spike_times) / num_trials

# Anchor initial guesses directly to what we visually observe in the plots
mu_init = 4.9 
sigma_init = 0.20  

# ==============================================================================
# 3. GLOBAL MLE OPTIMIZATION (Smooth Gradient Surface)
# ==============================================================================
def gaussian_global_nll(params, spike_times, num_trials, T_total):
    baseline, amplitude_A, mu_location, sigma_scale = params
    
    if baseline < 0 or amplitude_A <= 0 or sigma_scale <= 0:
        return 1e10
        
    # Term 1: Evaluating the log of rates at the precise moments of spikes
    rates_at_spikes = gaussian_rate_with_baseline(spike_times, baseline, amplitude_A, mu_location, sigma_scale)
    sum_log_lambda = np.sum(np.log(np.maximum(rates_at_spikes, 1e-9)))
    
    # Term 2: Continuous integral over the 10-second workspace
    t_fine = np.linspace(0, T_total, 1000)
    dt_fine = t_fine[1] - t_fine[0]
    rates_fine = gaussian_rate_with_baseline(t_fine, baseline, amplitude_A, mu_location, sigma_scale)
    integral_intensity = np.sum(rates_fine) * dt_fine
    
    return -(sum_log_lambda - (num_trials * integral_intensity))

initial_guess = [baseline_init, amplitude_init, mu_init, sigma_init]
bounds = [(0.0, 1.0), (1e-3, None), (3.5, 6.0), (0.01, 1.0)]

print("Running smooth global Gaussian optimization...")
result = minimize(
    gaussian_global_nll, 
    initial_guess, 
    args=(spike_times, num_trials, T_total), 
    method='L-BFGS-B',
    bounds=bounds
)

c_mle, amp_mle, mu_mle, sigma_mle = result.x
print(f"\nOptimization Successful!")
print(f"Baseline (C): {c_mle:.3f} Hz | Peak Time (mu): {mu_mle:.3f} s | Firing Width (sigma): {sigma_mle:.3f} s")

# ==============================================================================
# 4. GENERATE CONTINUOUS PLOT LINE & GRAPH DASHBOARD
# ==============================================================================
t_model = np.linspace(0, T_total, 1000)
fitted_rate = gaussian_rate_with_baseline(t_model, c_mle, amp_mle, mu_mle, sigma_mle)

# Calculate empirical average population rate from your counts dataframe (200ms window)
mean_rolling_hz = counts.groupby('time_sec')['escape_ipsi_hz'].mean()

fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(11, 9), 
    gridspec_kw={'height_ratios': [3, 1.3]}
)

# --- TOP ROW: DATA & SMOOTH GAUSSIAN FIT ---
ax1.plot(
    mean_rolling_hz.index, mean_rolling_hz, 
    color='darkgray', lw=2, label='Empirical Data (200ms Window)'
)
ax1.plot(
    t_model, fitted_rate, 
    color='crimson', lw=3, label='Symmetric Gaussian Profile Fit'
)
ax1.set_ylabel('Ipsilateral Escape Rate $\lambda(t)$ (Hz)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Trial Time (seconds)', fontsize=12)

# Perfectly safe Matplotlib string formatting for the title
ax1.set_title(
    r"$\lambda(t) = C + A \cdot \exp\left( -\frac{(t - \mu)^2}{2\sigma^2} \right)$", 
    fontsize=12, pad=15
)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xlim(0, T_total)
ax1.set_ylim(0, max(mean_rolling_hz.max(), fitted_rate.max()) * 1.15)

# --- BOTTOM ROW: METRICS SUMMARY PANEL ---
ax2.axis('off')
latex_annotation = r"""Model Architecture: Baseline-Corrected Inhomogeneous Gaussian Poisson Process

Fitted Parameters & Biological Meanings:
$\bullet \ \mathbf{C\ (Baseline)}$ = """ + f"{c_mle:.3f}" + r""" Hz  [Spontaneous Firing: Background baseline rate independent of stimulus]
$\bullet \ \mathbf{A\ (Amplitude)}$ = """ + f"{amp_mle:.3f}" + r""" Hz [Excess Volume: Total number of non-baseline escape events per trial]
$\bullet \ \mathbf{\mu\ (Location)}$ = """ + f"{mu_mle:.3f}" + r""" s  [Peak Processing Time: The exact moment the circuit reaches maximum network firing]
$\bullet \ \mathbf{\sigma\ (Scale)}$ = """ + f"{sigma_mle:.3f}" + r""" s  [Temporal Tuning Width: Standard deviation mapping the window of structural activity]"""

ax2.text(
    0.01, 0.95, 
    latex_annotation, 
    transform=ax2.transAxes, 
    fontsize=10.5, 
    linespacing=1.3, 
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_looming_model_NHPP_ipsi_0.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_looming_model_NHPP_ipsi_0.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()



########################### With trials



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==============================================================================
# 1. PEAK-HEIGHT EXPONENTIALLY SCALED RATE FUNCTION
# ==============================================================================
def gaussian_rate_peak_exponential(t, baseline, H0_initial, alpha_rate, trial_idx, mu_location, sigma_scale):
    """
    Computes lambda(t) where H0_initial is the pure peak height (Hz) above baseline 
    on Trial 0, scaled exponentially across subsequent trial numbers.
    """
    if sigma_scale <= 0 or baseline < 0 or H0_initial <= 0:
        return np.full_like(t, baseline) if isinstance(t, np.ndarray) else baseline
        
    # Peak height for this specific trial iteration (in Hz)
    height_j = H0_initial * np.exp(alpha_rate * trial_idx)
    
    # Pure Gaussian exponent shape without area normalization factors
    exponent = -0.5 * ((t - mu_location) / sigma_scale)**2
    return baseline + (height_j * np.exp(exponent))

# ==============================================================================
# 2. DATA ORCHESTRATION & FISH REPLICATE COUNTING
# ==============================================================================
# Filter out target event timestamps
global_events = sub_df[sub_df['ipsi_escape']]
unique_trials = sorted(global_events['trial_num'].unique())

spikes_by_trial = {}
fish_count_by_trial = {}

for tn in unique_trials:
    # Segregate precise event times for this trial sequence step
    spikes_by_trial[tn] = global_events[global_events['trial_num'] == tn]['trial_time'].values
    
    # Calculate exactly how many unique fish/files contributed data to this specific trial index
    fish_count_by_trial[tn] = len(sub_df[sub_df['trial_num'] == tn].groupby('file'))

T_total = 10.0

# Calculate initial global baseline guess from the pre-stimulus period (0 to 2 seconds)
num_total_trial_records = len(sub_df.groupby(['file', 'trial_num']))
spontaneous_spikes_early = np.sum(global_events['trial_time'].values < 2.0)
baseline_init = spontaneous_spikes_early / (num_total_trial_records * 2.0)

# ==============================================================================
# 3. GLOBAL MLE OPTIMIZATION (Corrected Cohort Weights)
# ==============================================================================
def global_peak_habituation_nll(params):
    # Unpack fixed parameter space
    baseline, H0_initial, alpha_rate, mu_location, sigma_scale = params
    
    if baseline < 0 or H0_initial <= 0 or sigma_scale <= 0:
        return 1e10
        
    total_nll = 0.0
    
    for j in unique_trials:
        trial_spikes = spikes_by_trial[j]
        n_fish_j = fish_count_by_trial[j] # Number of distinct animal datasets pooled in trial j
        
        # Term 1: Evaluating the log of rates at exact spike points
        rates_at_spikes = gaussian_rate_peak_exponential(
            trial_spikes, baseline, H0_initial, alpha_rate, j, mu_location, sigma_scale
        )
        sum_log_lambda = np.sum(np.log(np.maximum(rates_at_spikes, 1e-9)))
        
        # Term 2: Continuous numerical integral across the 10s trial timeline
        t_fine = np.linspace(0, T_total, 500)
        dt_fine = t_fine[1] - t_fine[0]
        rates_fine = gaussian_rate_peak_exponential(
            t_fine, baseline, H0_initial, alpha_rate, j, mu_location, sigma_scale
        )
        integral_intensity = np.sum(rates_fine) * dt_fine
        
        # Mathematically scale the integral expectation by the active cohort sample size
        total_nll += -(sum_log_lambda - (n_fish_j * integral_intensity))
        
    return total_nll

# Initial guess parameters: Baseline (Hz), Peak Height H0 (Hz), Decay Alpha, Mu (s), Sigma (s)
initial_guess = [baseline_init, 1.2, 0.0, 4.9, 0.20]
bounds = [(0.0, 1.0), (1e-3, 10.0), (-2.0, 2.0), (3.5, 6.0), (0.01, 1.0)]

print("Running corrected peak-height exponential optimization across pooled cohorts...")
result = minimize(global_peak_habituation_nll, initial_guess, method='L-BFGS-B', bounds=bounds)

c_mle, h0_mle, alpha_mle, mu_mle, sigma_mle = result.x

print("\nOptimization Complete! Structural parameters mapping cleanly to visual axes:")
print(f"Shared Baseline (C): {c_mle:.3f} Hz")
print(f"Trial 0 Peak Height Above Baseline (H0): {h0_mle:.3f} Hz")
print(f"Habituation Decay Constant (alpha): {alpha_mle:.4f}")
print(f"Shared Peak Time (mu): {mu_mle:.3f} s")
print(f"Shared Width (sigma): {sigma_mle:.3f} s")

print("\nCalculated Peak Values per Trial Exposure:")
heights_by_trial = []
for j in unique_trials:
    h_j = h0_mle * np.exp(alpha_mle * j)
    heights_by_trial.append(h_j)
    print(f" -> Trial {j} (n={fish_count_by_trial[j]} fish): Model Peak Above Baseline = {h_j:.2f} Hz (Total Peak Height = {c_mle + h_j:.2f} Hz)")

# ==============================================================================
# 4. GRAPHICAL VISUALIZATION DASHBOARD
# ==============================================================================
t_model = np.linspace(0, T_total, 1000)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_trials)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.5), gridspec_kw={'height_ratios': [3, 1.3]})

# --- TOP ROW: DATA & SMOOTH DECAY FITS ---
for idx, j in enumerate(unique_trials):
    fitted_rate_j = gaussian_rate_peak_exponential(
        t_model, c_mle, h0_mle, alpha_mle, j, mu_mle, sigma_mle
    )
    
    # Overlay faint historical empirical rolling window tracking lines if present
    if 'trial_num' in counts.columns:
        emp_trace = counts[counts['trial_num'] == j].groupby('time_sec')['escape_ipsi_hz'].mean()
        ax1.plot(emp_trace.index, emp_trace, color=colors[idx], alpha=0.18, lw=1.5, linestyle='--')
        
    ax1.plot(
        t_model, fitted_rate_j, 
        color=colors[idx], lw=2.5, 
        label=f'Trial {j} (Peak = {c_mle + heights_by_trial[idx]:.2f} Hz)'
    )

ax1.set_ylabel('Ipsilateral Escape Rate $\lambda(t)$ (Hz)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Trial Time (seconds)', fontsize=12)
ax1.set_title(r"Corrected Peak-Height Decay Form: $\lambda_j(t) = C + (H_0 e^{\alpha \cdot j}) \cdot \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)$", fontsize=12, pad=15)
ax1.legend(loc='upper right', title="Session Sequence")
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.set_xlim(0, T_total)
ax1.set_ylim(0, 2.2)

# --- BOTTOM ROW: METRICS DISPLAY ---
ax2.axis('off')
latex_annotation = f"""Model Architecture: Peak-Height Scaled Multi-Trial Inhomogeneous Gaussian Process (Corrected Fish Replicate Weights)

Fitted Global Metrics:
$\\bullet \\ \\mathbf{{C\\ (Baseline) = {c_mle:.3f}}}$ Hz  [Spontaneous background rate]
$\\bullet \\ \\mathbf{{H_0\\ (Initial\\ Peak\\ Height) = {h0_mle:.3f}}}$ Hz  [Maximum network intensity above baseline on Trial 0]
$\\bullet \\ \\mathbf{{\\alpha\\ (Habituation\\ Rate) = {alpha_mle:.4f}}}$      [Systematic scale shift per exposure: {alpha_mle*100:.2f}%]
$\\bullet \\ \\mathbf{{\\mu\\ (Peak\\ Horizon) = {mu_mle:.3f}}}$ s  [Circuit turning point alignment]
$\\bullet \\ \\mathbf{{\\sigma\\ (Burst\\ Width) = {sigma_mle:.3f}}}$ s  [Response timeline standard deviation footprint]
"""

ax2.text(0.01, 0.95, latex_annotation, transform=ax2.transAxes, fontsize=10.5, linespacing=1.3, verticalalignment='top')
plt.tight_layout()
plt.savefig(
    'temporal_dynamics_looming_model_NHPP_ipsi_1.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_looming_model_NHPP_ipsi_1.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()


#### CONTRA 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==============================================================================
# 1. PEAK-HEIGHT EXPONENTIALLY SCALED RATE FUNCTION
# ==============================================================================
def gaussian_rate_peak_exponential(t, baseline, H0_initial, alpha_rate, trial_idx, mu_location, sigma_scale):
    """
    Computes lambda(t) where H0_initial is the pure peak height (Hz) above baseline 
    on Trial 0, scaled exponentially across subsequent trial numbers.
    """
    if sigma_scale <= 0 or baseline < 0 or H0_initial <= 0:
        return np.full_like(t, baseline) if isinstance(t, np.ndarray) else baseline
        
    # Peak height for this specific trial iteration (in Hz)
    height_j = H0_initial * np.exp(alpha_rate * trial_idx)
    
    # Pure Gaussian exponent shape without area normalization factors
    exponent = -0.5 * ((t - mu_location) / sigma_scale)**2
    return baseline + (height_j * np.exp(exponent))

# ==============================================================================
# 2. DATA ORCHESTRATION & FISH REPLICATE COUNTING
# ==============================================================================
# Filter out target event timestamps
global_events = sub_df[sub_df['contra_escape']]
unique_trials = sorted(global_events['trial_num'].unique())

spikes_by_trial = {}
fish_count_by_trial = {}

for tn in unique_trials:
    # Segregate precise event times for this trial sequence step
    spikes_by_trial[tn] = global_events[global_events['trial_num'] == tn]['trial_time'].values
    
    # Calculate exactly how many unique fish/files contributed data to this specific trial index
    fish_count_by_trial[tn] = len(sub_df[sub_df['trial_num'] == tn].groupby('file'))

T_total = 10.0

# Calculate initial global baseline guess from the pre-stimulus period (0 to 2 seconds)
num_total_trial_records = len(sub_df.groupby(['file', 'trial_num']))
spontaneous_spikes_early = np.sum(global_events['trial_time'].values < 2.0)
baseline_init = spontaneous_spikes_early / (num_total_trial_records * 2.0)

# ==============================================================================
# 3. GLOBAL MLE OPTIMIZATION (Corrected Cohort Weights)
# ==============================================================================
def global_peak_habituation_nll(params):
    # Unpack fixed parameter space
    baseline, H0_initial, alpha_rate, mu_location, sigma_scale = params
    
    if baseline < 0 or H0_initial <= 0 or sigma_scale <= 0:
        return 1e10
        
    total_nll = 0.0
    
    for j in unique_trials:
        trial_spikes = spikes_by_trial[j]
        n_fish_j = fish_count_by_trial[j] # Number of distinct animal datasets pooled in trial j
        
        # Term 1: Evaluating the log of rates at exact spike points
        rates_at_spikes = gaussian_rate_peak_exponential(
            trial_spikes, baseline, H0_initial, alpha_rate, j, mu_location, sigma_scale
        )
        sum_log_lambda = np.sum(np.log(np.maximum(rates_at_spikes, 1e-9)))
        
        # Term 2: Continuous numerical integral across the 10s trial timeline
        t_fine = np.linspace(0, T_total, 500)
        dt_fine = t_fine[1] - t_fine[0]
        rates_fine = gaussian_rate_peak_exponential(
            t_fine, baseline, H0_initial, alpha_rate, j, mu_location, sigma_scale
        )
        integral_intensity = np.sum(rates_fine) * dt_fine
        
        # Mathematically scale the integral expectation by the active cohort sample size
        total_nll += -(sum_log_lambda - (n_fish_j * integral_intensity))
        
    return total_nll

# Initial guess parameters: Baseline (Hz), Peak Height H0 (Hz), Decay Alpha, Mu (s), Sigma (s)
initial_guess = [baseline_init, 1.2, 0.0, 4.9, 0.20]
bounds = [(0.0, 1.0), (1e-3, 10.0), (-2.0, 2.0), (3.5, 6.0), (0.01, 1.0)]

print("Running corrected peak-height exponential optimization across pooled cohorts...")
result = minimize(global_peak_habituation_nll, initial_guess, method='L-BFGS-B', bounds=bounds)

c_mle, h0_mle, alpha_mle, mu_mle, sigma_mle = result.x

print("\nOptimization Complete! Structural parameters mapping cleanly to visual axes:")
print(f"Shared Baseline (C): {c_mle:.3f} Hz")
print(f"Trial 0 Peak Height Above Baseline (H0): {h0_mle:.3f} Hz")
print(f"Habituation Decay Constant (alpha): {alpha_mle:.4f}")
print(f"Shared Peak Time (mu): {mu_mle:.3f} s")
print(f"Shared Width (sigma): {sigma_mle:.3f} s")

print("\nCalculated Peak Values per Trial Exposure:")
heights_by_trial = []
for j in unique_trials:
    h_j = h0_mle * np.exp(alpha_mle * j)
    heights_by_trial.append(h_j)
    print(f" -> Trial {j} (n={fish_count_by_trial[j]} fish): Model Peak Above Baseline = {h_j:.2f} Hz (Total Peak Height = {c_mle + h_j:.2f} Hz)")

# ==============================================================================
# 4. GRAPHICAL VISUALIZATION DASHBOARD
# ==============================================================================
t_model = np.linspace(0, T_total, 1000)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_trials)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.5), gridspec_kw={'height_ratios': [3, 1.3]})

# --- TOP ROW: DATA & SMOOTH DECAY FITS ---
for idx, j in enumerate(unique_trials):
    fitted_rate_j = gaussian_rate_peak_exponential(
        t_model, c_mle, h0_mle, alpha_mle, j, mu_mle, sigma_mle
    )
    
    # Overlay faint historical empirical rolling window tracking lines if present
    if 'trial_num' in counts.columns:
        emp_trace = counts[counts['trial_num'] == j].groupby('time_sec')['escape_contra_hz'].mean()
        ax1.plot(emp_trace.index, emp_trace, color=colors[idx], alpha=0.18, lw=1.5, linestyle='--')
        
    ax1.plot(
        t_model, fitted_rate_j, 
        color=colors[idx], lw=2.5, 
        label=f'Trial {j} (Peak = {c_mle + heights_by_trial[idx]:.2f} Hz)'
    )

ax1.set_ylabel('Contralateral Escape Rate $\lambda(t)$ (Hz)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Trial Time (seconds)', fontsize=12)
ax1.set_title(r"Corrected Peak-Height Decay Form: $\lambda_j(t) = C + (H_0 e^{\alpha \cdot j}) \cdot \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)$", fontsize=12, pad=15)
ax1.legend(loc='upper right', title="Session Sequence")
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.set_xlim(0, T_total)
ax1.set_ylim(0, 2.2)

# --- BOTTOM ROW: METRICS DISPLAY ---
ax2.axis('off')
latex_annotation = f"""Model Architecture: Peak-Height Scaled Multi-Trial Inhomogeneous Gaussian Process (Corrected Fish Replicate Weights)

Fitted Global Metrics:
$\\bullet \\ \\mathbf{{C\\ (Baseline) = {c_mle:.3f}}}$ Hz  [Spontaneous background rate]
$\\bullet \\ \\mathbf{{H_0\\ (Initial\\ Peak\\ Height) = {h0_mle:.3f}}}$ Hz  [Maximum network intensity above baseline on Trial 0]
$\\bullet \\ \\mathbf{{\\alpha\\ (Habituation\\ Rate) = {alpha_mle:.4f}}}$      [Systematic scale shift per exposure: {alpha_mle*100:.2f}%]
$\\bullet \\ \\mathbf{{\\mu\\ (Peak\\ Horizon) = {mu_mle:.3f}}}$ s  [Circuit turning point alignment]
$\\bullet \\ \\mathbf{{\\sigma\\ (Burst\\ Width) = {sigma_mle:.3f}}}$ s  [Response timeline standard deviation footprint]
"""

ax2.text(0.01, 0.95, latex_annotation, transform=ax2.transAxes, fontsize=10.5, linespacing=1.3, verticalalignment='top')
plt.tight_layout()
plt.savefig(
    'temporal_dynamics_looming_model_NHPP_contra_1.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_looming_model_NHPP_contra_1.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()