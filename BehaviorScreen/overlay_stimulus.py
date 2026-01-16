import numpy as np

PI = np.pi

LINEAR = 0
POWER_LAW = 1

LINEAR_RADIUS = 0
LINEAR_ANGLE = 1
CONSTANT_VELOCITY = 2

RING = 0
RANDOM_CLOUD = 1
ARC = 2

BOUNDING_BOX_CENTER = 0
FISH_CENTERED = 1
FISH_EGOCENTRIC = 2

COSINE = 0
MODULO = 1
TRIANGLE = 2
SQUARE = 3

DARK = 0
BRIGHT = 1
PHOTOTAXIS = 2
OMR = 3
OKR = 4
LOOMING = 5
PREY_CAPTURE = 6
CONCENTRIC_GRATING = 7
DOT = 8
IMAGE = 9
RAMP = 10
TURING = 11

def mod(a, b):
    return a - b * np.floor(a / b)

def mix(a, b, t):
    return a * (1 - t)[..., None] + b * t[..., None]

def hash1(x):
    return np.mod(np.sin(x * 127.1) * 43758.5453, 1.0)


def alpha_blend(background_rgb, overlay_rgba):
    alpha = overlay_rgba[..., 3:4]
    return background_rgb * (1 - alpha) + overlay_rgba[..., :3] * alpha

def make_coords_mm(width_px, height_px, mm_per_pixel, center_mm=(0.0, 0.0)):
    x = (np.arange(width_px) - width_px / 2) * mm_per_pixel + center_mm[0]
    y = (height_px / 2 - np.arange(height_px)) * mm_per_pixel + center_mm[1]
    X, Y = np.meshgrid(x, y)
    return X, Y

def dark_stimulus_vec(X, Y, p):
    H, W = X.shape
    return np.broadcast_to(p.u_background_color, (H, W, 4))

def bright_stimulus_vec(X, Y, p):
    H, W = X.shape
    return np.broadcast_to(p.u_foreground_color, (H, W, 4))

def phototaxis_stimulus_vec(X, Y, p):
    mask = (p.u_phototaxis_polarity * X) > 0
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def omr_stimulus_vec(X, Y, p):
    angle_rad = np.deg2rad(p.u_omr_angle_deg)
    orientation_vector = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    position = X * orientation_vector[0] + Y * orientation_vector[1]

    spatial_freq = 1.0 / p.u_omr_spatial_period_mm
    temporal_freq = p.u_omr_speed_mm_per_sec / p.u_omr_spatial_period_mm
    angle = spatial_freq * position
    phase = temporal_freq * p.u_time_s

    mask = np.sin(2 * PI * (angle - phase)) > 0
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def dot_stimulus_vec(X, Y, p):
    dist = np.sqrt((X - p.u_dot_center_mm[0])**2 + (Y - p.u_dot_center_mm[1])**2)
    mask = dist <= p.u_dot_radius_mm
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def concentric_grating_stimulus_vec(X, Y, p):
    spatial_freq = 1.0 / p.u_concentric_spatial_period_mm
    temporal_freq = p.u_concentric_speed_mm_per_sec / p.u_concentric_spatial_period_mm
    distance = np.sqrt(X**2 + Y**2)
    angle = spatial_freq * distance
    phase = temporal_freq * p.u_time_s

    mask = np.sin(2 * PI * (angle + phase)) > 0
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def looming_stimulus_vec(X, Y, p):
    relative_time = mod(p.u_time_s - p.u_start_time_s, p.u_looming_period_sec)
    looming_on = float(relative_time <= p.u_looming_expansion_time_sec)

    if p.u_looming_type == LINEAR_RADIUS:
        looming_radius = p.u_looming_expansion_speed_mm_per_sec * relative_time * looming_on

    elif p.u_looming_type == LINEAR_ANGLE:
        visual_angle = np.deg2rad(p.u_looming_expansion_speed_deg_per_sec) * relative_time * looming_on
        looming_radius = p.u_looming_distance_to_screen_mm * np.tan(visual_angle / 2)

    elif p.u_looming_type == CONSTANT_VELOCITY:
        angle_start_rad = np.deg2rad(p.u_looming_angle_start_deg)
        angle_stop_rad = np.deg2rad(p.u_looming_angle_stop_deg)
        t_0 = p.u_looming_size_to_speed_ratio_ms / np.tan(angle_start_rad / 2)
        t_f = p.u_looming_size_to_speed_ratio_ms / np.tan(angle_stop_rad / 2)
        period_ms = t_0 - t_f
        relative_time_ms = mod(1000 * (p.u_time_s - p.u_start_time_s), period_ms)
        looming_radius = (
            p.u_looming_distance_to_screen_mm
            * p.u_looming_size_to_speed_ratio_ms
            / (t_0 - relative_time_ms)
        )
    else:
        looming_radius = 0.0

    dist = np.sqrt((X - p.u_looming_center_mm[0])**2 + (Y - p.u_looming_center_mm[1])**2)
    mask = dist <= looming_radius
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)


def ramp_stimulus_vec(X, Y, p):
    relative_time = np.mod(p.u_time_s - p.u_start_time_s, p.u_ramp_duration_sec)
    frac = np.clip(relative_time / p.u_ramp_duration_sec, 0.0, 1.0)

    if p.u_ramp_type == p.LINEAR:
        ramp_value = frac
    elif p.u_ramp_type == p.POWER_LAW:
        ramp_value = frac ** (1 / p.u_ramp_powerlaw_exponent)
    else:
        ramp_value = 0.0

    return mix(p.u_background_color, p.u_foreground_color, ramp_value)

def turing_stimulus_vec(X, Y, p):
    angle_rad = np.deg2rad(p.u_turing_angle_deg)
    velocity = p.u_turing_speed_mm_per_sec * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    Xp = X - velocity[0] * p.u_time_s
    Yp = Y - velocity[1] * p.u_time_s
    k0 = 4 * PI / p.u_turing_spatial_period_mm

    wave_sum = np.zeros_like(X)
    for i in range(int(p.u_turing_n_waves)):
        angle = (i + hash1(i)) / p.u_turing_n_waves * 2 * PI
        phase = hash1(i * 12.34) * 2 * PI
        dirx, diry = np.cos(angle), np.sin(angle)
        wave = np.sin(k0 * (Xp * dirx + Yp * diry) + phase)
        wave_sum += wave

    mask = wave_sum > 0
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def okr_stimulus_vec(X, Y, p):
    angular_spatial_freq = np.deg2rad(p.u_okr_spatial_frequency_deg)
    angular_temporal_freq = np.deg2rad(p.u_okr_speed_deg_per_sec)
    angle = np.arctan2(Y, X)
    phase = angular_temporal_freq * p.u_time_s
    mask = mod(angle - phase, angular_spatial_freq) > angular_spatial_freq / 2
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def prey_capture_stimulus_vec(X, Y, bbox_mm, p):
    H, W = X.shape
    result = np.zeros((H, W), dtype=bool)

    if p.u_prey_capture_type == RING:
        phase = np.deg2rad(p.u_prey_speed_deg_s) * p.u_time_s
        for i in range(int(p.u_n_preys)):
            angle = i * 2 * PI / p.u_n_preys
            prey_offset = p.u_prey_trajectory_radius_mm * np.array([
                np.cos(angle + phase),
                np.sin(angle + phase)
            ])
            dist = np.sqrt((X - prey_offset[0])**2 + (Y - prey_offset[1])**2)
            result |= dist <= p.u_prey_radius_mm

    elif p.u_prey_capture_type == ARC:
        relative_time_s = p.u_time_s - p.u_start_time_s
        arc_start_rad = np.deg2rad(p.u_prey_arc_start_deg)
        arc_stop_rad = np.deg2rad(p.u_prey_arc_stop_deg)
        arc_phase_rad = np.deg2rad(p.u_prey_arc_phase_deg)
        angle_range_rad = arc_stop_rad - arc_start_rad

        angle_rad = arc_start_rad
        if p.u_prey_periodic_function == COSINE:
            freq = np.deg2rad(p.u_prey_speed_deg_s) / (2 * abs(angle_range_rad))
            angle_rad += angle_range_rad * ((1 - np.cos(2 * PI * freq * relative_time_s + arc_phase_rad)) / 2)
        elif p.u_prey_periodic_function == MODULO:
            period = abs(angle_range_rad) / np.deg2rad(p.u_prey_speed_deg_s)
            angle_rad += angle_range_rad * mod(relative_time_s, period) / period

        prey_pos = np.array([
            -p.u_prey_trajectory_radius_mm * np.sin(angle_rad),
             p.u_prey_trajectory_radius_mm * np.cos(angle_rad)
        ])
        dist = np.sqrt((X - prey_pos[0])**2 + (Y - prey_pos[1])**2)
        result = dist <= p.u_prey_radius_mm

    elif p.u_prey_capture_type == RANDOM_CLOUD:
        for i in range(int(p.u_n_preys)):
            prey_pos_mm = p.u_prey_position[i] / p.u_pix_per_mm_proj
            prey_dir = np.array([
                np.cos(p.u_prey_trajectory_angle[i]),
                np.sin(p.u_prey_trajectory_angle[i])
            ])
            pos = mod(prey_pos_mm + p.u_time_s * p.u_prey_speed_mm_s * prey_dir, bbox_mm[2:4]) - bbox_mm[2:4] / 2
            dist = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
            result |= dist <= p.u_prey_radius_mm

    return np.where(result[..., None], p.u_foreground_color, p.u_background_color)
