from pathlib import Path
import argparse 
from dataclasses import dataclass, field

import numpy as np
from  tqdm import tqdm
import cv2

from BehaviorScreen.load import (
    Directories, 
    find_files, 
    load_data
)
from BehaviorScreen.core import Stim

from video_tools import FFMPEG_VideoWriter_CPU

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

MAX_PREY = 64

# TODO save this in ZebVR metadata
rollover_time_sec = 3600

@dataclass
class Param:
    u_time_s: float = 0
    u_start_time_sec: float = 0
    u_pix_per_mm_proj: float = 0 # TODO check that 

    # Colors
    u_foreground_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    u_background_color: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])

    # General
    u_coordinate_system: int = 0
    u_stim_select: Stim = Stim.DARK
    u_phototaxis_polarity: float = 1.0

    # OMR
    u_omr_spatial_period_mm: float = 10.0
    u_omr_angle_deg: float = 90.0
    u_omr_speed_mm_per_sec: float = 10.0

    # Turing
    u_turing_spatial_period_mm: float = 10.0
    u_turing_angle_deg: float = 0.0
    u_turing_speed_mm_per_sec: float = 0.0
    u_turing_n_waves: float = 1.0

    # Concentric
    u_concentric_spatial_period_mm: float = 10.0
    u_concentric_speed_mm_per_sec: float = 0.0

    # OKR
    u_okr_spatial_frequency_deg: float = 1.0
    u_okr_speed_deg_per_sec: float = 0.0

    # Looming
    u_looming_type: int = 0
    u_looming_center_mm: list = field(default_factory=lambda: [0.0, 0.0])
    u_looming_period_sec: float = 1.0
    u_looming_expansion_time_sec: float = 0.5
    u_looming_expansion_speed_mm_per_sec: float = 0.0
    u_looming_expansion_speed_deg_per_sec: float = 0.0
    u_looming_angle_start_deg: float = 0.0
    u_looming_angle_stop_deg: float = 90.0
    u_looming_size_to_speed_ratio_ms: float = 1.0
    u_looming_distance_to_screen_mm: float = 100.0

    # Dot
    u_dot_center_mm: list = field(default_factory=lambda: [0.0, 0.0])
    u_dot_radius_mm: float = 1.0

    # Prey capture
    u_prey_capture_type: int = 0
    u_prey_periodic_function: int = 0
    u_n_preys: float = 0.0
    u_prey_radius_mm: float = 1.0
    u_prey_trajectory_radius_mm: float = 5.0
    u_prey_speed_mm_s: float = 0.0
    u_prey_speed_deg_s: float = 0.0
    u_prey_arc_start_deg: float = 0.0
    u_prey_arc_stop_deg: float = 360.0
    u_prey_arc_phase_deg: float = 0.0
    u_prey_position: np.ndarray = field(default_factory=lambda: np.zeros((MAX_PREY, 2)))
    u_prey_trajectory_angle: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PREY))

    # Image stimulus
    u_image_texture: int = 0
    u_image_size: list = field(default_factory=lambda: [0.0, 0.0])
    u_image_res_px_per_mm: float = 10.0
    u_image_offset_mm: list = field(default_factory=lambda: [0.0, 0.0])

    # Ramp
    u_ramp_duration_sec: float = 1.0
    u_ramp_powerlaw_exponent: float = 1.0
    u_ramp_type: int = 0

def mod(a, b):
    return a - b * np.floor(a / b)

def mix(a, b, t):
    return a * (1 - t)[..., None] + b * t[..., None]

def hash1(x):
    return np.mod(np.sin(x * 127.1) * 43758.5453, 1.0)

def alpha_blend(background_rgb, overlay_rgba, alpha_max = 0.5):

    bg = background_rgb.astype(np.float32) / 255.0
    fg = overlay_rgba.astype(np.float32)
    
    alpha = alpha_max * fg[..., 3:4]  
    blended = bg * (1 - alpha) + fg[..., :3] * alpha
    return (blended * 255).clip(0, 255).astype(np.uint8)

# TODO handle different coordinate system
def fish_centered():
    pass

def bbox_centered():
    pass

def image_coord_grid(height_px, width_px):
    xs = np.arange(width_px)
    ys = np.arange(height_px)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X, Y], axis=-1).astype(np.float32)
    return coords

def egocentric_coords_mm(coords, centroid, pc1, pc2, mm_per_pixel):

    centroid = np.asarray(centroid, dtype=float)
    coords_centered = coords - centroid

    pc1 = np.asarray(pc1, dtype=float)
    pc2 = np.asarray(pc2, dtype=float)
    R = np.stack([pc2, pc1], axis=1)

    coords_rot = coords_centered @ R
    coords_mm = coords_rot * mm_per_pixel

    return coords_mm

def dark_overlay(X, Y, p):
    H, W = X.shape
    return np.broadcast_to(p.u_background_color, (H, W, 4))

def bright_overlay(X, Y, p):
    H, W = X.shape
    return np.broadcast_to(p.u_foreground_color, (H, W, 4))

def phototaxis_overlay(X, Y, p):
    mask = (p.u_phototaxis_polarity * X) > 0
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def omr_overlay(X, Y, p):
    angle_rad = np.deg2rad(p.u_omr_angle_deg)
    orientation_vector = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    position = X * orientation_vector[0] + Y * orientation_vector[1]

    spatial_freq = 1.0 / p.u_omr_spatial_period_mm
    temporal_freq = p.u_omr_speed_mm_per_sec / p.u_omr_spatial_period_mm
    angle = spatial_freq * position
    phase = temporal_freq * p.u_time_s

    mask = np.sin(2 * PI * (angle - phase)) > 0
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def dot_overlay(X, Y, p):
    dist = np.sqrt((X - p.u_dot_center_mm[0])**2 + (Y - p.u_dot_center_mm[1])**2)
    mask = dist <= p.u_dot_radius_mm
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def concentric_grating_overlay(X, Y, p):
    spatial_freq = 1.0 / p.u_concentric_spatial_period_mm
    temporal_freq = p.u_concentric_speed_mm_per_sec / p.u_concentric_spatial_period_mm
    distance = np.sqrt(X**2 + Y**2)
    angle = spatial_freq * distance
    phase = temporal_freq * p.u_time_s

    mask = np.sin(2 * PI * (angle + phase)) > 0
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

def looming_overlay(X, Y, p):
    relative_time = mod(p.u_time_s - p.u_start_time_sec, p.u_looming_period_sec)
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
        relative_time_ms = mod(1000 * (p.u_time_s - p.u_start_time_sec), period_ms)
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

def ramp_overlay(X, Y, p):
    relative_time = np.mod(p.u_time_s - p.u_start_time_sec, p.u_ramp_duration_sec)
    frac = np.clip(relative_time / p.u_ramp_duration_sec, 0.0, 1.0)

    if p.u_ramp_type == LINEAR:
        ramp_value = frac
    elif p.u_ramp_type == POWER_LAW:
        ramp_value = frac ** (1 / p.u_ramp_powerlaw_exponent)
    else:
        ramp_value = 0.0

    return mix(p.u_background_color, p.u_foreground_color, ramp_value)

def turing_overlay(X, Y, p):
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

def okr_overlay(X, Y, p):
    angular_spatial_freq = np.deg2rad(p.u_okr_spatial_frequency_deg)
    angular_temporal_freq = np.deg2rad(p.u_okr_speed_deg_per_sec)
    angle = np.arctan2(Y, X)
    phase = angular_temporal_freq * p.u_time_s
    mask = mod(angle - phase, angular_spatial_freq) > angular_spatial_freq / 2
    return np.where(mask[..., None], p.u_foreground_color, p.u_background_color)

# TODO fix this 
def image_overlay(X, Y, p):
    H, W = X.shape
    overlay = np.broadcast_to(p.u_background_color, (H, W, 4)).copy()

    if p.u_image_texture is None:
        return overlay

    image_size_mm = np.array(p.u_image_size) / p.u_image_res_px_per_mm
    coords_x = (X - p.u_image_offset_mm[0]) / image_size_mm[0] + 0.5
    coords_y = (Y - p.u_image_offset_mm[1]) / image_size_mm[1] + 0.5

    mask = (coords_x >= 0) & (coords_x <= 1) & (coords_y >= 0) & (coords_y <= 1)
    if not np.any(mask):
        return overlay

    H_tex, W_tex, _ = p.u_image_texture.shape
    ix = np.clip((coords_x * W_tex).astype(int), 0, W_tex - 1)
    iy = np.clip((coords_y * H_tex).astype(int), 0, H_tex - 1)

    overlay[mask] = p.u_image_texture[iy[mask], ix[mask]]

    return overlay

# TODO fix that (bbox argument should not be here)
def prey_capture_overlay(X, Y, p):
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
        relative_time_s = p.u_time_s - p.u_start_time_sec
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

    # elif p.u_prey_capture_type == RANDOM_CLOUD:
    #     for i in range(int(p.u_n_preys)):
    #         prey_pos_mm = p.u_prey_position[i] / p.u_pix_per_mm_proj
    #         prey_dir = np.array([
    #             np.cos(p.u_prey_trajectory_angle[i]),
    #             np.sin(p.u_prey_trajectory_angle[i])
    #         ])
    #         pos = mod(prey_pos_mm + p.u_time_s * p.u_prey_speed_mm_s * prey_dir, bbox_mm[2:4]) - bbox_mm[2:4] / 2
    #         dist = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
    #         result |= dist <= p.u_prey_radius_mm

    return np.where(result[..., None], p.u_foreground_color, p.u_background_color)

def get_active_stimulus(stimuli, timestamp):
    stimuli_sorted = sorted(stimuli, key=lambda s: s['timestamp'])
    active_stim = None
    for stim in stimuli_sorted:
        if stim['timestamp'] <= timestamp:
            active_stim = stim
        else:
            break
    return active_stim

overlay_funcs = {
    Stim.DARK: dark_overlay,
    Stim.BRIGHT: bright_overlay,
    Stim.PHOTOTAXIS: phototaxis_overlay,
    Stim.OMR: omr_overlay,
    Stim.OKR: okr_overlay,
    Stim.LOOMING: looming_overlay,    
    Stim.PREY_CAPTURE: prey_capture_overlay,
    Stim.CONCENTRIC_GRATING: concentric_grating_overlay,
    Stim.DOT: dot_overlay,
    Stim.IMAGE: image_overlay,
    Stim.RAMP: ramp_overlay,
    Stim.TURING: turing_overlay,
}

def stim_to_param(stim: dict, time_sec: float) -> Param:
    """Convert stimulus dict to Param dataclass using Stim enum."""
    p = Param(u_time_s=time_sec)

    if stim is None:
        return p

    # Convert stim_select to Stim enum
    try:
        stim_enum = Stim(int(stim.get('stim_select', 0)))
    except ValueError:
        stim_enum = Stim.DARK  

    p.u_stim_select = stim_enum
    p.u_start_time_sec = stim.get('start_time_sec', p.u_start_time_sec)
    p.u_foreground_color = stim.get('foreground_color', p.u_foreground_color)
    p.u_background_color = stim.get('background_color', p.u_background_color)
    p.u_coordinate_system = stim.get('coordinate_system', p.u_coordinate_system)

    if stim_enum == Stim.DOT:
        p.u_dot_center_mm = stim.get('dot_center_mm', p.u_dot_center_mm)
        p.u_dot_radius_mm = stim.get('dot_radius_mm', p.u_dot_radius_mm)

    elif stim_enum == Stim.OMR:
        p.u_omr_spatial_period_mm = stim.get('omr_spatial_period_mm', p.u_omr_spatial_period_mm)
        p.u_omr_angle_deg = stim.get('omr_angle_deg', p.u_omr_angle_deg)
        p.u_omr_speed_mm_per_sec = stim.get('omr_speed_mm_per_sec', p.u_omr_speed_mm_per_sec)

    elif stim_enum == Stim.TURING:
        p.u_turing_spatial_period_mm = stim.get('turing_spatial_period_mm', p.u_turing_spatial_period_mm)
        p.u_turing_angle_deg = stim.get('turing_angle_deg', p.u_turing_angle_deg)
        p.u_turing_speed_mm_per_sec = stim.get('turing_speed_mm_per_sec', p.u_turing_speed_mm_per_sec)
        p.u_turing_n_waves = stim.get('turing_n_waves', p.u_turing_n_waves)

    elif stim_enum == Stim.CONCENTRIC_GRATING:
        p.u_concentric_spatial_period_mm = stim.get('concentric_spatial_period_mm', p.u_concentric_spatial_period_mm)
        p.u_concentric_speed_mm_per_sec = stim.get('concentric_speed_mm_per_sec', p.u_concentric_speed_mm_per_sec)

    elif stim_enum == Stim.LOOMING:
        p.u_looming_type = stim.get('looming_type', p.u_looming_type)
        p.u_looming_center_mm = stim.get('looming_center_mm', p.u_looming_center_mm)
        p.u_looming_period_sec = stim.get('looming_period_sec', p.u_looming_period_sec)
        p.u_looming_expansion_time_sec = stim.get('looming_expansion_time_sec', p.u_looming_expansion_time_sec)
        p.u_looming_expansion_speed_mm_per_sec = stim.get('looming_expansion_speed_mm_per_sec', p.u_looming_expansion_speed_mm_per_sec)
        p.u_looming_expansion_speed_deg_per_sec = stim.get('looming_expansion_speed_deg_per_sec', p.u_looming_expansion_speed_deg_per_sec)
        p.u_looming_angle_start_deg = stim.get('looming_angle_start_deg', p.u_looming_angle_start_deg)
        p.u_looming_angle_stop_deg = stim.get('looming_angle_stop_deg', p.u_looming_angle_stop_deg)
        p.u_looming_size_to_speed_ratio_ms = stim.get('looming_size_to_speed_ratio_ms', p.u_looming_size_to_speed_ratio_ms)
        p.u_looming_distance_to_screen_mm = stim.get('looming_distance_to_screen_mm', p.u_looming_distance_to_screen_mm)

    elif stim_enum == Stim.OKR:
        p.u_okr_spatial_frequency_deg = stim.get('okr_spatial_frequency_deg', p.u_okr_spatial_frequency_deg)
        p.u_okr_speed_deg_per_sec = stim.get('okr_speed_deg_per_sec', p.u_okr_speed_deg_per_sec)

    elif stim_enum == Stim.PREY_CAPTURE:
        p.u_prey_capture_type = stim.get('prey_capture_type', p.u_prey_capture_type)
        p.u_prey_periodic_function = stim.get('prey_periodic_function', p.u_prey_periodic_function)
        p.u_n_preys = stim.get('n_preys', p.u_n_preys)
        p.u_prey_radius_mm = stim.get('prey_radius_mm', p.u_prey_radius_mm)
        p.u_prey_trajectory_radius_mm = stim.get('prey_trajectory_radius_mm', p.u_prey_trajectory_radius_mm)
        p.u_prey_speed_mm_s = stim.get('prey_speed_mm_s', p.u_prey_speed_mm_s)
        p.u_prey_speed_deg_s = stim.get('prey_speed_deg_s', p.u_prey_speed_deg_s)
        p.u_prey_arc_start_deg = stim.get('prey_arc_start_deg', p.u_prey_arc_start_deg)
        p.u_prey_arc_stop_deg = stim.get('prey_arc_stop_deg', p.u_prey_arc_stop_deg)
        p.u_prey_arc_phase_deg = stim.get('prey_arc_phase_deg', p.u_prey_arc_phase_deg)
        p.u_prey_position = stim.get('prey_position', p.u_prey_position)
        p.u_prey_trajectory_angle = stim.get('prey_trajectory_angle', p.u_prey_trajectory_angle)
        p.u_pix_per_mm_proj = stim.get('pix_per_mm_proj', 1.0)  # Needed for RANDOM_CLOUD

    elif stim_enum == Stim.IMAGE:
        p.u_image_texture = stim.get('image_texture', p.u_image_texture)
        p.u_image_size = stim.get('image_size', p.u_image_size)
        p.u_image_res_px_per_mm = stim.get('image_res_px_per_mm', p.u_image_res_px_per_mm)
        p.u_image_offset_mm = stim.get('image_offset_mm', p.u_image_offset_mm)

    elif stim_enum == Stim.RAMP:
        p.u_ramp_duration_sec = stim.get('ramp_duration_sec', p.u_ramp_duration_sec)
        p.u_ramp_powerlaw_exponent = stim.get('ramp_powerlaw_exponent', p.u_ramp_powerlaw_exponent)
        p.u_ramp_type = stim.get('ramp_type', p.u_ramp_type)

    elif stim_enum == Stim.PHOTOTAXIS:
        p.u_phototaxis_polarity = stim.get('phototaxis_polarity', p.u_phototaxis_polarity)

    return p

def overlay_stimulus(X,Y,p):

    fn = overlay_funcs.get(p.u_stim_select, None)
    if fn is not None:
        return fn(X,Y,p)

def add_label(
        image: np.ndarray,
        label: str,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale = 1,
        color = (255, 255, 255),
        thickness = 2,
        position = (10,30),
    ) -> None:

    cv2.putText(image, label, position, font, font_scale, color, thickness, cv2.LINE_AA)

    
def overlay(       
        root: Path,
        overlay_dir: str,
        metadata: str,
        stimuli: str,
        tracking: str,
        video: str,
        video_timestamp: str,
    ) -> None:


    directories = Directories(
        root,
        metadata=metadata,
        stimuli=stimuli,
        tracking=tracking,
        video=video,
        video_timestamp=video_timestamp,
    )
    behavior_files = find_files(directories)

    output_dir = root / overlay_dir 
    output_dir.mkdir(parents=True, exist_ok=True)

    for behavior_file in tqdm(behavior_files):

        output_video = output_dir / behavior_file.video.name
        behavior_data = load_data(behavior_file)

        mm_per_pixel = 1/behavior_data.metadata['calibration']['pix_per_mm']
        timestamp_start = behavior_data.video_timestamps.loc[0, 'timestamp']

        height_px = behavior_data.video.get_height()
        width_px = behavior_data.video.get_width()
        fps = behavior_data.video.get_fps()
        
        grid = image_coord_grid(height_px, width_px)

        writer = FFMPEG_VideoWriter_CPU(
            filename = output_video,
            height = height_px, 
            width = width_px, 
            fps = fps, 
            q = 20,
        )

        num_frames = min(
            behavior_data.video.get_number_of_frame(), 
            behavior_data.tracking.shape[0]
        )

        for frame_idx in tqdm(range(num_frames), leave=False):

            ret, image = behavior_data.video.next_frame()
            
            if not ret:
                raise RuntimeError(f'failed to read image #{frame_idx}')

            # TODO write functions for the different coordinate systems
            coords_mm = egocentric_coords_mm(
                grid,
                centroid = behavior_data.tracking.loc[frame_idx, ['centroid_x', 'centroid_y']].values,
                pc1 = behavior_data.tracking.loc[frame_idx, ['pc1_x', 'pc1_y']].values,
                pc2 = behavior_data.tracking.loc[frame_idx, ['pc2_x', 'pc2_y']].values, 
                mm_per_pixel=mm_per_pixel
            )

            timestamp = behavior_data.video_timestamps.loc[frame_idx, 'timestamp']
            time_sec = (1e-9*timestamp) % rollover_time_sec  
            exp_time_sec = 1e-9*(timestamp-timestamp_start)
            current_stim = get_active_stimulus(behavior_data.stimuli, timestamp)
            
            if current_stim is None:
                stim = image
                label = f'{exp_time_sec:.2f}'
            else:
                parameters = stim_to_param(current_stim, time_sec)
                oly = overlay_stimulus(
                    coords_mm[:,:,0],
                    coords_mm[:,:,1],
                    parameters
                )
                stim = alpha_blend(image, oly)
                label = f'{exp_time_sec:.2f}-{parameters.u_stim_select.name}'

            add_label(stim, label)
            writer.write_frame(stim)    

        writer.close()

def main(args: argparse.Namespace) -> None:
    overlay(
        root=args.root,
        overlay_dir = args.overlay_dir,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        video=args.video,
        video_timestamp=args.video_timestamp,
    )

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Run megabout pipeline on tracking data from Lightning Pose"
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder (e.g. WT_oct_2025)",
    )

    parser.add_argument(
        "--overlay-dir",
        default='overlay',
        help="Directory to store overlay videos",
    )

    # Directory layout overrides
    parser.add_argument(
        "--metadata",
        default="",
        help="Subfolder containing metadata files (default: data)",
    )

    parser.add_argument(
        "--stimuli",
        default="",
        help="Subfolder containing stimulus log files (default: data)",
    )

    parser.add_argument(
        "--tracking",
        default="",
        help="Subfolder containing tracking CSV files (default: data)",
    )

    parser.add_argument(
        "--video",
        default="",
        help="Subfolder containing raw video files (default: video)",
    )

    parser.add_argument(
        "--video-timestamp",
        default="",
        help="Subfolder containing video timestamp files (default: video)",
    )

    return parser

if __name__ == '__main__':

    main(build_parser().parse_args())

