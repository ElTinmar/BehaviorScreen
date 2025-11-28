import numpy as np

def get_shader_time(start_time_sec, trial_time_sec, rollover_time_sec) -> float:
    return (trial_time_sec + start_time_sec) % rollover_time_sec

def get_shader_trial_time(start_time_sec, trial_time_sec, rollover_time_sec) -> float:
    return get_shader_time(start_time_sec, trial_time_sec, rollover_time_sec) - start_time_sec

def omr(shader_time, omr_speed_mm_per_sec, omr_spatial_period_mm) -> float:
    temporal_freq = omr_speed_mm_per_sec  / omr_spatial_period_mm
    phase = temporal_freq * shader_time
    return phase % 2*np.pi

def okr(shader_time, okr_speed_deg_per_sec):
    angular_temporal_freq = np.deg2rad(okr_speed_deg_per_sec)
    phase = angular_temporal_freq * shader_time
    return phase % 2*np.pi

def looming_linear_radius():
    pass