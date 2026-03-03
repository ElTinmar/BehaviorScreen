from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    BehaviorFiles,
    find_files, 
    load_data
)
from pathlib import Path
from typing import List
import pickle
from BehaviorScreen.megabouts import MegaboutResults

ROOT = Path('/media/martin/DATA1/Behavioral_screen/DATA/WT/danieau')
#ROOT = Path('/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/DATA/WT/danieau')

directories = Directories(
    root = ROOT,
    metadata='results',
    stimuli='results',
    tracking='results',
    full_tracking='lightning_pose',
    temperature='results',
    video='results',
    video_timestamp='results',
    results='results',
    plots=''
)
files: List[BehaviorFiles] = find_files(directories)
behavior_file = files[0]
behavior_data: BehaviorData = load_data(behavior_file)

with open(ROOT / 'megabout.pkl', 'rb') as fp:
    mb = pickle.load(fp) 

megabout = mb[behavior_file.metadata.stem]


###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BehaviorScreen.process import compute_angle_between_vectors
from scipy.signal import savgol_filter

filename = '/home/martin/Downloads/11-44-00/video_preds/01_07dpf_WT_Thu_11_Dec_2025_13h15min29sec_fish_3_eyes.csv'
fs = 120

df = pd.read_csv(filename, header=[0,1,2])

left_front = np.array([0, 64]) - df.heatmap_tracker.eye_left_front[['x', 'y']].to_numpy()
left_back = np.array([0, 64]) - df.heatmap_tracker.eye_left_back[['x', 'y']].to_numpy()
right_front = np.array([0, 64]) - df.heatmap_tracker.eye_right_front[['x', 'y']].to_numpy()
right_back = np.array([0, 64]) - df.heatmap_tracker.eye_right_back[['x', 'y']].to_numpy()

left_vector = left_front - left_back
right_vector = right_front - right_back

likelihood = df.heatmap_tracker.xs('likelihood', axis=1, level=1).prod(axis=1)
remove = likelihood < 0.95

t = np.arange(len(df))/fs
L = compute_angle_between_vectors(left_vector, np.array([0,1]))
R = compute_angle_between_vectors(right_vector, np.array([0,1]))

L_s = savgol_filter(L, window_length=11, polyorder=2)
dL_s = savgol_filter(L, window_length=11, polyorder=2, deriv=1)
ddL_s = savgol_filter(L, window_length=11, polyorder=2, deriv=2)

R_s = savgol_filter(R, window_length=11, polyorder=2)
dR_s = savgol_filter(R, window_length=11, polyorder=2, deriv=1)
ddR_s = savgol_filter(R, window_length=11, polyorder=2, deriv=2)

dt = 1 / fs

dL = np.gradient(L_s, dt)
dR = np.gradient(R_s, dt)

H = savgol_filter((L+R)/(2*dt), window_length=11, polyorder=2)
dH = savgol_filter((L+R)/(2*dt), window_length=11, polyorder=2, deriv=1)

V = R_s - L_s

dV = np.gradient(V, dt)

dH[remove] = np.nan
mask = abs(dH) > 3

#angle_left[remove] = np.nan
#angle_right[remove] = np.nan

plt.plot(t, np.rad2deg(L_s))
plt.plot(t, np.rad2deg(R_s))
plt.show()

def sliding_corr(x, y, window):
    n = len(x)
    rho = np.zeros(n)
    half = window // 2
    
    for i in range(half, n-half):
        xs = x[i-half:i+half]
        ys = y[i-half:i+half]
        rho[i] = np.corrcoef(xs, ys)[0,1]
        
    return rho

rho_slow = sliding_corr(dL, dR, window=int(5*fs))  
rho = sliding_corr(dL, dR, window=int(2*fs))  

congruent = (rho_slow + rho) > 1.2
uncongruent = (rho < -0.5) 
vergent = V > 1.2

def mask_to_intervals(mask):
    """Convert boolean mask to list of (start, end) index intervals."""
    intervals = []
    in_block = False
    start = 0
    
    for i, val in enumerate(mask):
        if val and not in_block:
            in_block = True
            start = i
        elif not val and in_block:
            in_block = False
            intervals.append((start, i))
    
    if in_block:
        intervals.append((start, len(mask)))
    
    return intervals


cong_intervals = mask_to_intervals(congruent)
div_intervals = mask_to_intervals(uncongruent)
ver_intervals = mask_to_intervals(vergent)


# --- Plot Left Eye ---
plt.figure()
plt.plot(t, L_s)
plt.plot(t, R_s)

for start, end in ver_intervals:
    plt.axvspan(t[start], t[end-1], alpha=0.2, color='blue')

for start, end in cong_intervals:
    plt.axvspan(t[start], t[end-1], alpha=0.2, color='green')

for start, end in div_intervals:
    plt.axvspan(t[start], t[end-1], alpha=0.2, color='red')

plt.xlabel("Time (s)")
plt.ylabel("Left Eye Angle")
plt.title("Left Eye with Congruent / Divergent Patches")
plt.show()

### 

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal import welch

def rolling_psd(signal, fs, fmin, fmax, window_sec=10.0):
    n = len(signal)
    window_samples = int(window_sec * fs)
    half_win = window_samples // 2
    
    power_full = np.zeros(n)
    
    for center in range(half_win, n - half_win):
        start = center - half_win
        end = center + half_win
        seg = signal[start:end]
        
        f, Pxx = welch(seg, fs=fs, nperseg=window_samples)
        band_power = Pxx[(f >= fmin) & (f <= fmax)].sum()
        
        power_full[center] = band_power
    
    # Optionally, pad start/end with first/last computed power
    power_full[:half_win] = power_full[half_win]
    power_full[-half_win:] = power_full[-half_win-1]
    
    return power_full


def bandpower_continuous(signal, fs, fmin, fmax):
    # Bandpass filter
    b, a = butter(3, [fmin/(fs/2), fmax/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal)

    # Analytic signal
    analytic = hilbert(filtered)
    
    # Instantaneous power
    amplitude = np.abs(analytic)
    power = amplitude**2
    phase = np.angle(analytic)
    return power, phase

from scipy.signal import spectrogram

# Spectrogram parameters
nperseg = int(1.0 * fs)  # 1-second window
noverlap = int(0.9 * nperseg)  # 90% overlap for smooth time resolution
fmin, fmax = 0, 30  # frequency range to display (Hz)

# Compute spectrogram
f, t_spec, Sxx = spectrogram(H, fs=fs, nperseg=nperseg, noverlap=noverlap)

# Limit frequency range for plotting
freq_mask = (f >= fmin) & (f <= fmax)

plt.figure(figsize=(10,4))
plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading='gouraud')
plt.colorbar(label='Power (dB)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of Conjugate Eye Signal')
plt.show()