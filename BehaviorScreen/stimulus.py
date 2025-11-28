from typing import Tuple
import numpy as np


def get_shader_time(start_time_sec, trial_time_sec, rollover_time_sec) -> float:
    return (trial_time_sec + start_time_sec) % rollover_time_sec


def get_shader_trial_time(start_time_sec, trial_time_sec, rollover_time_sec) -> float:
    return get_shader_time(start_time_sec, trial_time_sec, rollover_time_sec) - start_time_sec


def omr(
        shader_time: float, 
        omr_speed_mm_per_sec: float, 
        omr_spatial_period_mm: float
    ) -> float:

    temporal_freq = omr_speed_mm_per_sec  / omr_spatial_period_mm
    phase = temporal_freq * shader_time
    return phase % 2*np.pi


def okr(
        shader_time: float, 
        okr_speed_deg_per_sec: float
    ) -> float:

    angular_temporal_freq = np.deg2rad(okr_speed_deg_per_sec)
    phase = angular_temporal_freq * shader_time
    return phase % 2*np.pi


def looming_linear_radius(
        shader_trial_time: float, 
        looming_period_sec: float, 
        looming_expansion_time_sec: float,
        looming_expansion_speed_mm_per_sec: float
    ) -> float:

    relative_time = shader_trial_time % looming_period_sec
    looming_on = relative_time <= looming_expansion_time_sec
    looming_radius = looming_expansion_speed_mm_per_sec * relative_time * looming_on
    return looming_radius


def looming_linear_angle(
        shader_trial_time: float, 
        looming_period_sec: float, 
        looming_expansion_time_sec: float,
        looming_expansion_speed_deg_per_sec: float,
        looming_distance_to_screen_mm: float
    ) -> float:

    relative_time = shader_trial_time % looming_period_sec
    looming_on = relative_time <= looming_expansion_time_sec
    visual_angle = np.deg2rad(looming_expansion_speed_deg_per_sec) * relative_time * looming_on
    looming_radius = looming_distance_to_screen_mm * np.tan(visual_angle/2)
    return looming_radius


def looming_constant_velocity_approach(
        shader_trial_time: float,
        looming_angle_start_deg: float,
        looming_angle_stop_deg: float,
        looming_size_to_speed_ratio_ms: float,
        looming_distance_to_screen_mm: float
    ) -> float:

    angle_start_rad = np.deg2rad(looming_angle_start_deg)
    angle_stop_rad = np.deg2rad(looming_angle_stop_deg)
    t_0 = looming_size_to_speed_ratio_ms / np.tan(angle_start_rad/2)
    t_f = looming_size_to_speed_ratio_ms / np.tan(angle_stop_rad/2)
    period_ms = t_0 - t_f
    relative_time_ms = (1000*shader_trial_time) % period_ms
    looming_radius = looming_distance_to_screen_mm * looming_size_to_speed_ratio_ms / (t_0 - relative_time_ms)
    return looming_radius


def prey_capture_arc_stimulus_sine(
        shader_trial_time: float,
        prey_arc_start_deg: float,
        prey_arc_stop_deg: float,
        prey_speed_deg_s: float
    ) -> Tuple[float, float]:
    
    arc_start_rad = np.deg2rad(prey_arc_start_deg)
    arc_stop_rad = np.deg2rad(prey_arc_stop_deg)
    angle_range_rad = arc_stop_rad - arc_start_rad
    angle_rad = arc_start_rad
    freq = np.deg2rad(prey_speed_deg_s) / (2*np.abs(angle_range_rad))
    phase = freq * shader_trial_time
    angle_rad += angle_range_rad * (np.sin(2*np.pi*phase)/2 + 0.5)
    return angle_rad, phase % 2*np.pi # TODO fix phase pls


def prey_capture_arc_stimulus_modulo(
        shader_trial_time: float,
        prey_arc_start_deg: float,
        prey_arc_stop_deg: float,
        prey_speed_deg_s: float
    ) -> float:
    
    arc_start_rad = np.deg2rad(prey_arc_start_deg)
    arc_stop_rad = np.deg2rad(prey_arc_stop_deg)
    angle_range_rad = arc_stop_rad - arc_start_rad
    angle_rad = arc_start_rad
    period = np.abs(angle_range_rad) / np.deg2rad(prey_speed_deg_s)
    angle_rad += angle_range_rad * (shader_trial_time % period)/period
    return angle_rad