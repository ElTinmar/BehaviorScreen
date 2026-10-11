from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from megabouts.utils import bouts_category_name_short
from scipy.optimize import minimize
from scipy.special import gammainc, gamma

o_index = bouts_category_name_short.index("O")

ROOT = Path('/media/martin/DATA_18TB/Screen')
df = pd.read_csv(ROOT / 'bouts_control.csv')


print(f"#fish: {len(df.file.unique())}")

fine_dt = 0.02
time_bins = np.arange(0, 5 + fine_dt, fine_dt)
window_duration = 0.1
window_size_steps = int(window_duration / fine_dt)
window_size_steps |= 1


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
counts['o-bend_hz'] = counts['rolling_o-bend'] / window_duration

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



#################
#
#   MODEL
#
#################

from scipy.optimize import minimize
from scipy.special import gammainc, gamma

T_start = 0.0
T_end = 5.0
T_duration = T_end - T_start

events = sub_df[
    sub_df["o-bend"]
    & (sub_df["trial_time"] >= T_start)
    & (sub_df["trial_time"] <= T_end)
]

spike_times = (events["trial_time"] - T_start).values

num_trials = sub_df.groupby(["file", "trial_num"]).ngroups


# ---------------------------------------------------------
# λ(t)
# ---------------------------------------------------------

def lambda_t(t, A, k, tau, B):
    """
    Normalized generalized-gamma kernel.

    Peak occurs at t = k*tau.
    Peak value = A + B.
    """

    x = t / tau

    kernel = np.power(x / k, k) * np.exp(k - x)

    return np.maximum(A * kernel + B, 1e-12)


# ---------------------------------------------------------
# ∫ λ(t) dt
# ---------------------------------------------------------

def integrated_rate(T, A, k, tau, B):

    signal = (
        A
        * tau
        * np.exp(k)
        / (k ** k)
        * gamma(k + 1)
        * gammainc(k + 1, T / tau)
    )

    return signal + B * T


# ---------------------------------------------------------
# Negative log likelihood
# ---------------------------------------------------------

def negative_log_likelihood(params, spike_times, T, n_trials):

    A, k, tau, B = params

    lam = lambda_t(spike_times, A, k, tau, B)

    log_term = np.sum(np.log(lam))

    integral = integrated_rate(T, A, k, tau, B)

    return -(log_term - n_trials * integral)


# ---------------------------------------------------------
# Fit
# ---------------------------------------------------------

initial_guess = [
    10.0,   # A
    2.0,    # k
    0.25,   # tau
    0.2     # baseline
]

bounds = [
    (0, None),      # A
    (0.1, 10),      # k
    (0.01, None),   # tau
    (0, None)       # baseline
]

result = minimize(
    negative_log_likelihood,
    initial_guess,
    args=(spike_times, T_duration, num_trials),
    bounds=bounds,
)

if not result.success:
    raise RuntimeError(result.message)

A_fit, k_fit, tau_fit, B_fit = result.x

peak_time = k_fit * tau_fit
peak_rate = A_fit + B_fit

print("\n===== FIT RESULTS =====")
print(f"A        = {A_fit:.3f} Hz")
print(f"k        = {k_fit:.3f}")
print(f"tau      = {tau_fit:.3f} s")
print(f"B        = {B_fit:.3f} Hz")
print(f"Peak t   = {peak_time:.3f} s")
print(f"Peak Hz  = {peak_rate:.3f}")


mean_rate = counts.groupby("time_sec")["o-bend_hz"].mean()

t_model = np.linspace(T_start, T_end, 600)

fit = lambda_t(
    t_model,
    A_fit,
    k_fit,
    tau_fit,
    B_fit,
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    gridspec_kw={"height_ratios": [3, 1]},
)

# --------------------------------------------------
# Data + model
# --------------------------------------------------

ax1.plot(
    mean_rate.index,
    mean_rate.values,
    color="0.5",
    lw=2,
    label="Rolling average",
)

ax1.plot(
    t_model,
    fit,
    color="crimson",
    lw=3,
    label="Poisson MLE",
)

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("O-bend rate (Hz)")
ax1.set_ylim(bottom=0)

ax1.set_title(
    r"$\lambda(t)=A\left(\frac{t}{k\tau}\right)^k e^{\,k-t/\tau}+B$"
)

ax1.grid(alpha=0.3)
ax1.legend()

# --------------------------------------------------
# Parameter panel
# --------------------------------------------------

ax2.axis("off")

text = (
    "Maximum-likelihood estimates\n\n"
    f"A        = {A_fit:.3f} Hz\n"
    f"k        = {k_fit:.3f}\n"
    f"tau      = {tau_fit:.3f} s\n"
    f"B        = {B_fit:.3f} Hz\n\n"
    f"Peak time = {peak_time:.3f} s\n"
    f"Peak rate = {peak_rate:.3f} Hz"
)

ax2.text(
    0.01,
    0.98,
    text,
    va="top",
    ha="left",
    fontsize=11,
    family="monospace",
)

plt.tight_layout()
plt.savefig(
    'temporal_dynamics_dark_flash_model_NHPP.svg', 
    bbox_inches='tight'
)
plt.savefig(
    'temporal_dynamics_dark_flash_model_NHPP.png',
    dpi=300, 
    bbox_inches='tight'
)
plt.show()


### trials

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammainc, gamma

# ==========================================
# PREPARE TRIAL-AWARE DATA
# ==========================================
T_start = 0.0
T_end = 5.0
T_duration = T_end - T_start

events = sub_df[
    sub_df["o-bend"]
    & (sub_df["trial_time"] >= T_start)
    & (sub_df["trial_time"] <= T_end)
]

# Extract exact timestamps and corresponding trial numbers
spike_times = (events["trial_time"] - T_start).values
spike_trials = events["trial_num"].values

unique_trials = np.sort(sub_df['trial_num'].unique()) # Expected: [0, 1, 2, 3, 4]
num_trials = sub_df.groupby(["file", "trial_num"]).ngroups

# ==========================================
# RATIONAL FUNCTION TRIAL-MODULATED KERNEL
# ==========================================
def lambda_t_gamma_rational_trials(t, m, A, k, tau, B, alpha_1, alpha_2, alpha_B):
    t_safe = np.maximum(t, 0)
    x = t_safe / tau
    
    # 1. Kinetic Time Kernel (Generalized Gamma Profile)
    kernel = np.power(x / k, k) * np.exp(k - x)
    
    # 2. Rational Function Amplitude Modulator
    # (1 + a1*m) handles the initial sensitization climb
    # (1 + a2*m^2) dominates later trials to handle habituation
    rational_scale = (1.0 + alpha_1 * m) / (1.0 + alpha_2 * m**2)
    
    mod_A = A * rational_scale
    mod_B = B * np.exp(alpha_B * m)
    
    return np.maximum(mod_A * kernel + mod_B, 1e-12)

def negative_log_likelihood_rational(params, t_events, m_events, T, unique_trial_set, N_total_trials):
    A, k, tau, B, alpha_1, alpha_2, alpha_B = params
    
    # Term 1: Sum of log-rates at precise event occurrences
    rates = lambda_t_gamma_rational_trials(t_events, m_events, A, k, tau, B, alpha_1, alpha_2, alpha_B)
    log_term = np.sum(np.log(rates))
    
    # Term 2: Exact Analytical Spatiotemporal Integral
    fish_per_trial = N_total_trials / len(unique_trial_set)
    base_gamma_integral = tau * np.exp(k) / (k ** k) * gamma(k + 1)
    
    total_integral = 0.0
    for m in unique_trial_set:
        rational_scale = (1.0 + alpha_1 * m) / (1.0 + alpha_2 * m**2)
        
        mod_A = A * rational_scale
        signal_part = mod_A * base_gamma_integral * gammainc(k + 1, T / tau)
        baseline_part = B * np.exp(alpha_B * m) * T
        
        total_integral += (signal_part + baseline_part) * fish_per_trial
        
    return -(log_term - total_integral)

# ==========================================
# EXECUTE GRADIENT OPTIMIZATION
# ==========================================
# Guesses: [A, k, tau, B, alpha_1, alpha_2, alpha_B]
initial_guess = [10.0, 2.0, 0.25, 0.2, 1.5, 0.5, 0.0]

bounds = [
    (0.0, None),     # A
    (0.1, 10.0),     # k
    (0.01, None),    # tau
    (0.0, None),     # B
    (0.0, 10.0),     # alpha_1 (Numerator growth coefficient)
    (0.0, 5.0),      # alpha_2 (Denominator decay coefficient)
    (-0.2, 0.2)      # alpha_B
]

result_rational = minimize(
    negative_log_likelihood_rational,
    initial_guess,
    args=(spike_times, spike_trials, T_duration, unique_trials, num_trials),
    method='L-BFGS-B',
    bounds=bounds,
)

if not result_rational.success:
    raise RuntimeError(result_rational.message)

A_fit, k_fit, tau_fit, B_fit, a1_fit, a2_fit, aB_fit = result_rational.x

# ==========================================
# GENERATE PRODUCTION DASHBOARD
# ==========================================
mean_rate = counts.groupby(["time_sec", "trial_num"])["o-bend_hz"].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2.8, 1.4]})

# High-contrast palette to distinguish individual trial traces easily
colors = ['#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363'] 
t_model = np.linspace(0, T_duration, 600)

for m_idx, col in zip(unique_trials, colors):
    # Plot empirical data curves
    trial_data = mean_rate[mean_rate['trial_num'] == m_idx]
    ax1.plot(trial_data['time_sec'] - T_start, trial_data['o-bend_hz'], '--', color=col, alpha=0.25)
    
    # Plot matched rational function model profiles
    fit = lambda_t_gamma_rational_trials(t_model, m_idx, A_fit, k_fit, tau_fit, B_fit, a1_fit, a2_fit, aB_fit)
    ax1.plot(t_model, fit, '-', color=col, lw=2.5, label=f'Trial {m_idx} Model')

ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel("O-bend rate (Hz)", fontsize=11)
ax1.set_ylim(bottom=0)
# Change the title line in the plotting section to this:
ax1.set_title(
    r"$\lambda(t, m) = \max \left( A \cdot \frac{1 + \alpha_1 m}{1 + \alpha_2 m^2} \left(\frac{t}{k\tau}\right)^k e^{\,k-t/\tau} + B e^{\alpha_B \cdot m} ,\, 10^{-12} \right)$", 
    fontsize=11
)
ax1.grid(alpha=0.15)
ax1.legend(loc='upper right')

# ---- Bottom Panel: Analytical Summary ----
ax2.axis("off")
peak_time = k_fit * tau_fit

# Calculate the precise effective amplitude scale factor across your trials using the rational function formula
m_axis = np.array([0, 1, 2, 3, 4])
rational_scales = (1.0 + a1_fit * m_axis) / (1.0 + a2_fit * m_axis**2)

text_dump = f"""Maximum-Likelihood Estimates (Rational Function Polynomial Profiling):

Fitted Base Parameters:
A (Scaling Amplitude) = {A_fit:.3f} Hz  |  B (Base Background Floor) = {B_fit:.3f} Hz
k (Shape Param)       = {k_fit:.3f}                 |  tau (Kinetic Scale)        = {tau_fit:.3f} s
Kinetic Peak Arrival Delay (k * tau) = {peak_time:.3f} s

Fitted Rational Modulators:
α1 (Numerator Growth Term)    = {a1_fit:.4f}
α2 (Denominator Decay Term)   = {a2_fit:.4f}

Derived True Amplitude Peak Signal Capacity Across Trials:
Trial 0 Net Scaling Factor: {rational_scales[0]:.2f}  --> Calculated Peak Signal Component: {A_fit * rational_scales[0]:.2f} Hz
Trial 1 Net Scaling Factor: {rational_scales[1]:.2f}  --> Calculated Peak Signal Component: {A_fit * rational_scales[1]:.2f} Hz (Target Max 🌟)
Trial 2 Net Scaling Factor: {rational_scales[2]:.2f}  --> Calculated Peak Signal Component: {A_fit * rational_scales[2]:.2f} Hz
Trial 3 Net Scaling Factor: {rational_scales[3]:.2f}  --> Calculated Peak Signal Component: {A_fit * rational_scales[3]:.2f} Hz
Trial 4 Net Scaling Factor: {rational_scales[4]:.2f}  --> Calculated Peak Signal Component: {A_fit * rational_scales[4]:.2f} Hz
"""

ax2.text(0.01, 0.95, text_dump, transform=ax2.transAxes, fontsize=10, va="top", ha="left", family="monospace")
plt.tight_layout()

# Save final clean copies
plt.savefig('temporal_dynamics_dark_flash_model_NHPP_1.svg', bbox_inches='tight')
plt.savefig('temporal_dynamics_dark_flash_model_NHPP_1.png', dpi=300, bbox_inches='tight')
plt.show()