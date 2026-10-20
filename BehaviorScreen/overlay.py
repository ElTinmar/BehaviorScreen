"""
Overlay video of one recording: the whole field with the stimulus the fish
saw, its trajectory, online tracking and the well (left); a fish-centred view
with the post-hoc pose, bout and saccade labels (right); tail and eye angles,
bouts, saccades and tracking agreement over a moving time window (bottom).

    python -m BehaviorScreen.overlay ROOT --fish NAME [--saccades saccades_augmented_all.csv]
        [--start 0] [--duration 30] [--speed 0.5] [--fps 30]

Only the requested window is read and rendered, at --fps output frames per
second; --speed < 1 is slow motion.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
from megabouts.utils import bouts_category_name, bouts_category_name_short, bouts_category_color
from tqdm import tqdm
from video_tools import FFMPEG_VideoWriter_CPU

from BehaviorScreen.core import SACCADE_CLASS_COLORS, SACCADE_CLASS_NAMES, EventDirection, Stim
from BehaviorScreen.eyes.detect_saccades import extract_raw_eye_angles
from BehaviorScreen.load import Directories, find_files
from BehaviorScreen.megabouts import load_megabout
from BehaviorScreen.pose_estimation import eye_crop_matrix, eye_crop_size
from BehaviorScreen.process import compute_angle_between_vectors
from BehaviorScreen.qc import get_per_frame_tracking_error
from BehaviorScreen.recording import Recording, nearest_indices

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
    u_pix_per_mm_proj: float = 0  # TODO check that

    # Colors
    u_foreground_color: np.ndarray = field(
        default_factory=lambda: np.array([255, 255, 255], dtype=np.uint8)
    )
    u_background_color: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0], dtype=np.uint8)
    )

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
    u_prey_trajectory_angle: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_PREY)
    )

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


def alpha_blend(bg, fg, alpha=128):
    bg16 = bg.astype(np.uint16)
    fg16 = fg.astype(np.uint16)
    alpha16 = np.uint16(alpha)
    out16 = bg16 + ((fg16 - bg16) * alpha16 >> 8)
    return out16.astype(np.uint8)


def sum_blend(bg, fg, gain=1.0):
    out = bg.astype(np.uint16) + (fg.astype(np.uint16) * gain)
    return np.clip(out, 0, 255).astype(np.uint8)


def egocentric_coords_mm(coords, centroid, pc1, pc2, mm_per_pixel):
    transform = np.stack([pc2, pc1], axis=1) * mm_per_pixel
    return (coords - centroid) @ transform


# TODO handle different coordinate system
def fish_centered():
    pass


def bbox_centered():
    pass


def image_coord_grid(height_px, width_px, downsample: int = 1):
    xs = np.arange(0, width_px, downsample)
    ys = np.arange(0, height_px, downsample)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X, Y], axis=-1).astype(np.float32)
    return coords


def dark_overlay(X, Y, p):
    H, W = X.shape
    return np.broadcast_to(p.u_background_color, (H, W, 3))


def bright_overlay(X, Y, p):
    H, W = X.shape
    return np.broadcast_to(p.u_foreground_color, (H, W, 3))


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
    dist = np.sqrt((X - p.u_dot_center_mm[0]) ** 2 + (Y - p.u_dot_center_mm[1]) ** 2)
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
        looming_radius = (
            p.u_looming_expansion_speed_mm_per_sec * relative_time * looming_on
        )

    elif p.u_looming_type == LINEAR_ANGLE:
        visual_angle = (
            np.deg2rad(p.u_looming_expansion_speed_deg_per_sec)
            * relative_time
            * looming_on
        )
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

    dist = np.sqrt(
        (X - p.u_looming_center_mm[0]) ** 2 + (Y - p.u_looming_center_mm[1]) ** 2
    )
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

    ramp_value = np.full(X.shape[:2], ramp_value, dtype=np.float32)
    return mix(p.u_background_color, p.u_foreground_color, ramp_value)


def turing_overlay(X, Y, p):
    angle_rad = np.deg2rad(p.u_turing_angle_deg)
    velocity = p.u_turing_speed_mm_per_sec * np.array(
        [-np.sin(angle_rad), np.cos(angle_rad)]
    )
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
    overlay = np.broadcast_to(p.u_background_color, (H, W, 3)).copy()

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
            prey_offset = p.u_prey_trajectory_radius_mm * np.array(
                [np.cos(angle + phase), np.sin(angle + phase)]
            )
            dist = np.sqrt((X - prey_offset[0]) ** 2 + (Y - prey_offset[1]) ** 2)
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
            angle_rad += angle_range_rad * (
                (1 - np.cos(2 * PI * freq * relative_time_s + arc_phase_rad)) / 2
            )
        elif p.u_prey_periodic_function == MODULO:
            period = abs(angle_range_rad) / np.deg2rad(p.u_prey_speed_deg_s)
            angle_rad += angle_range_rad * mod(relative_time_s, period) / period

        prey_pos = np.array(
            [
                -p.u_prey_trajectory_radius_mm * np.sin(angle_rad),
                p.u_prey_trajectory_radius_mm * np.cos(angle_rad),
            ]
        )
        dist = np.sqrt((X - prey_pos[0]) ** 2 + (Y - prey_pos[1]) ** 2)
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
    stimuli_sorted = sorted(stimuli, key=lambda s: s["timestamp"])
    active_stim = None
    for stim in stimuli_sorted:
        if stim["timestamp"] <= timestamp:
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
        stim_enum = Stim(int(stim.get("stim_select", 0)))
    except ValueError:
        stim_enum = Stim.DARK

    p.u_stim_select = stim_enum
    p.u_start_time_sec = stim.get("start_time_sec", p.u_start_time_sec)
    p.u_foreground_color = (
        255 * np.asarray(stim.get("foreground_color", p.u_foreground_color))[:3]
    ).astype(np.uint8)
    p.u_background_color = (
        255 * np.asarray(stim.get("background_color", p.u_background_color))[:3]
    ).astype(np.uint8)
    p.u_coordinate_system = stim.get("coordinate_system", p.u_coordinate_system)

    if stim_enum == Stim.DOT:
        p.u_dot_center_mm = stim.get("dot_center_mm", p.u_dot_center_mm)
        p.u_dot_radius_mm = stim.get("dot_radius_mm", p.u_dot_radius_mm)

    elif stim_enum == Stim.OMR:
        p.u_omr_spatial_period_mm = stim.get(
            "omr_spatial_period_mm", p.u_omr_spatial_period_mm
        )
        p.u_omr_angle_deg = stim.get("omr_angle_deg", p.u_omr_angle_deg)
        p.u_omr_speed_mm_per_sec = stim.get(
            "omr_speed_mm_per_sec", p.u_omr_speed_mm_per_sec
        )

    elif stim_enum == Stim.TURING:
        p.u_turing_spatial_period_mm = stim.get(
            "turing_spatial_period_mm", p.u_turing_spatial_period_mm
        )
        p.u_turing_angle_deg = stim.get("turing_angle_deg", p.u_turing_angle_deg)
        p.u_turing_speed_mm_per_sec = stim.get(
            "turing_speed_mm_per_sec", p.u_turing_speed_mm_per_sec
        )
        p.u_turing_n_waves = stim.get("turing_n_waves", p.u_turing_n_waves)

    elif stim_enum == Stim.CONCENTRIC_GRATING:
        p.u_concentric_spatial_period_mm = stim.get(
            "concentric_spatial_period_mm", p.u_concentric_spatial_period_mm
        )
        p.u_concentric_speed_mm_per_sec = stim.get(
            "concentric_speed_mm_per_sec", p.u_concentric_speed_mm_per_sec
        )

    elif stim_enum == Stim.LOOMING:
        p.u_looming_type = stim.get("looming_type", p.u_looming_type)
        p.u_looming_center_mm = stim.get("looming_center_mm", p.u_looming_center_mm)
        p.u_looming_period_sec = stim.get("looming_period_sec", p.u_looming_period_sec)
        p.u_looming_expansion_time_sec = stim.get(
            "looming_expansion_time_sec", p.u_looming_expansion_time_sec
        )
        p.u_looming_expansion_speed_mm_per_sec = stim.get(
            "looming_expansion_speed_mm_per_sec", p.u_looming_expansion_speed_mm_per_sec
        )
        p.u_looming_expansion_speed_deg_per_sec = stim.get(
            "looming_expansion_speed_deg_per_sec",
            p.u_looming_expansion_speed_deg_per_sec,
        )
        p.u_looming_angle_start_deg = stim.get(
            "looming_angle_start_deg", p.u_looming_angle_start_deg
        )
        p.u_looming_angle_stop_deg = stim.get(
            "looming_angle_stop_deg", p.u_looming_angle_stop_deg
        )
        p.u_looming_size_to_speed_ratio_ms = stim.get(
            "looming_size_to_speed_ratio_ms", p.u_looming_size_to_speed_ratio_ms
        )
        p.u_looming_distance_to_screen_mm = stim.get(
            "looming_distance_to_screen_mm", p.u_looming_distance_to_screen_mm
        )

    elif stim_enum == Stim.OKR:
        p.u_okr_spatial_frequency_deg = stim.get(
            "okr_spatial_frequency_deg", p.u_okr_spatial_frequency_deg
        )
        p.u_okr_speed_deg_per_sec = stim.get(
            "okr_speed_deg_per_sec", p.u_okr_speed_deg_per_sec
        )

    elif stim_enum == Stim.PREY_CAPTURE:
        p.u_prey_capture_type = stim.get("prey_capture_type", p.u_prey_capture_type)
        p.u_prey_periodic_function = stim.get(
            "prey_periodic_function", p.u_prey_periodic_function
        )
        p.u_n_preys = stim.get("n_preys", p.u_n_preys)
        p.u_prey_radius_mm = stim.get("prey_radius_mm", p.u_prey_radius_mm)
        p.u_prey_trajectory_radius_mm = stim.get(
            "prey_trajectory_radius_mm", p.u_prey_trajectory_radius_mm
        )
        p.u_prey_speed_mm_s = stim.get("prey_speed_mm_s", p.u_prey_speed_mm_s)
        p.u_prey_speed_deg_s = stim.get("prey_speed_deg_s", p.u_prey_speed_deg_s)
        p.u_prey_arc_start_deg = stim.get("prey_arc_start_deg", p.u_prey_arc_start_deg)
        p.u_prey_arc_stop_deg = stim.get("prey_arc_stop_deg", p.u_prey_arc_stop_deg)
        p.u_prey_arc_phase_deg = stim.get("prey_arc_phase_deg", p.u_prey_arc_phase_deg)
        p.u_prey_position = stim.get("prey_position", p.u_prey_position)
        p.u_prey_trajectory_angle = stim.get(
            "prey_trajectory_angle", p.u_prey_trajectory_angle
        )
        p.u_pix_per_mm_proj = stim.get(
            "pix_per_mm_proj", 1.0
        )  # Needed for RANDOM_CLOUD

    elif stim_enum == Stim.IMAGE:
        p.u_image_texture = stim.get("image_texture", p.u_image_texture)
        p.u_image_size = stim.get("image_size", p.u_image_size)
        p.u_image_res_px_per_mm = stim.get(
            "image_res_px_per_mm", p.u_image_res_px_per_mm
        )
        p.u_image_offset_mm = stim.get("image_offset_mm", p.u_image_offset_mm)

    elif stim_enum == Stim.RAMP:
        p.u_ramp_duration_sec = stim.get("ramp_duration_sec", p.u_ramp_duration_sec)
        p.u_ramp_powerlaw_exponent = stim.get(
            "ramp_powerlaw_exponent", p.u_ramp_powerlaw_exponent
        )
        p.u_ramp_type = stim.get("ramp_type", p.u_ramp_type)

    elif stim_enum == Stim.PHOTOTAXIS:
        p.u_phototaxis_polarity = stim.get(
            "phototaxis_polarity", p.u_phototaxis_polarity
        )

    return p


def overlay_stimulus(X, Y, p):

    fn = overlay_funcs.get(p.u_stim_select, None)
    if fn is not None:
        return fn(X, Y, p)




# ---------------------------------------------------------------------------
# Overlay video
# ---------------------------------------------------------------------------

PANEL_PX, STRIP_PX, MARGIN_PX = 720, 300, 80
ZOOM_MM = 9.0
WINDOW_S = 4.0
SACCADE_LABEL_S = 0.3
QC_CENTROID_MM, QC_HEADING_DEG = 1.0, 15.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY = (235, 235, 235), (150, 150, 150)
ONLINE_BGR, POSE_BGR, TRAIL_BGR = (0, 165, 255), (110, 220, 120), (230, 190, 90)
OK_BGR, BAD_BGR = (90, 160, 90), (60, 60, 220)
EYE_BGR = {"left": (240, 150, 60), "right": (70, 120, 240)}
SKELETON_ALPHA, NODE_RADIUS = 0.7, 5
QC_LABEL = "QC: online vs pose tracking"
TAIL_PARTS = ["Swim_Bladder"] + [f"Tail_{i}" for i in range(9)]


def bgr(rgb: Tuple[float, ...]) -> Tuple[int, int, int]:
    return tuple(int(255 * c) for c in rgb[2::-1])


BOUT_BGR = [bgr(matplotlib.colors.to_rgb(c)) for c in bouts_category_color]
SACCADE_BGR = {k: bgr(matplotlib.colors.to_rgb(v)) for k, v in SACCADE_CLASS_COLORS.items()}


@dataclass
class Tracks:
    """Everything drawn, on the recording's frame clock (pixels, degrees, seconds from the first frame)."""

    time_s: np.ndarray
    head: np.ndarray
    tail: np.ndarray  # (n, 10, 2): swim bladder, then Tail_0..8
    tail_angle: np.ndarray
    online: np.ndarray  # (n, 6): centroid, pc1, pc2
    tracking_ok: np.ndarray
    bouts: pd.DataFrame  # onset, offset (frames), category, sign, proba
    saccades: pd.DataFrame  # onset_s, cluster
    eye_angle: Dict[str, np.ndarray] = field(default_factory=dict)
    eyes: Dict[str, np.ndarray] = field(default_factory=dict)  # side -> (n, 2, 2) front, back in eye-crop pixels


def fit_length(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full((n, *values.shape[1:]), np.nan)
    out[: min(n, len(values))] = values[:n]
    return out


def load_tracks(rec: Recording, saccades: Optional[pd.DataFrame]) -> Tracks:
    n = min(len(rec.pose), rec.n_frames)
    pose = rec.pose.iloc[:n]
    xy = lambda frame, part: frame[part][["x", "y"]].to_numpy(float)
    head = xy(pose, "Head")
    tail = np.stack([xy(pose, part) for part in TAIL_PARTS], axis=1)
    error = get_per_frame_tracking_error(rec).iloc[:n]

    megabout = load_megabout(rec)
    bouts = pd.DataFrame(columns=["onset", "offset", "category", "sign", "proba"])
    if megabout is not None:
        b = megabout.bouts
        k = min(len(b.onset), len(b.offset), len(b.category), len(b.sign), len(b.proba))
        bouts = pd.DataFrame({"onset": b.onset[:k], "offset": b.offset[:k], "category": b.category[:k], "sign": b.sign[:k]}).astype(int)
        bouts["proba"] = np.asarray(b.proba[:k], float)
        bouts = bouts[(bouts.onset >= 0) & (bouts.offset > bouts.onset)]

    events = saccades[saccades["fish"].astype(str) == rec.name] if saccades is not None else pd.DataFrame()
    tracks = Tracks(
        time_s=(rec.frame_time_ns[:n] - rec.frame_time_ns[0]) * 1e-9,
        head=head,
        tail=tail,
        tail_angle=np.rad2deg(compute_angle_between_vectors(tail[:, -1] - tail[:, 1], tail[:, 0] - head)),
        online=rec.online.iloc[:n][["centroid_x", "centroid_y", "pc1_x", "pc1_y", "pc2_x", "pc2_y"]].to_numpy(float),
        tracking_ok=((error.centroid_distance_mm < QC_CENTROID_MM) & (error.heading_angle_deg < QC_HEADING_DEG)).to_numpy(),
        bouts=bouts,
        saccades=pd.DataFrame({"onset_s": events.get("onset_time_s", []), "cluster": events.get("cluster", [])}),
    )
    if not rec.eyes.empty:
        _, left, right, _, _ = extract_raw_eye_angles(rec)
        tracks.eye_angle = {"left": fit_length(left, n), "right": fit_length(right, n)}
        tracks.eyes = {
            side: fit_length(np.stack([xy(rec.eyes, f"eye_{side}_{end}") for end in ("front", "back")], axis=1), n)
            for side in ("left", "right")
        }
    return tracks


class FrameReader:
    """Video frames by index: reads forward, seeks only when going back or far ahead."""

    def __init__(self, path: Path) -> None:
        self.capture, self.position = cv2.VideoCapture(str(path)), 0
        self.size = (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def read(self, index: int) -> np.ndarray:
        if index < self.position or index > self.position + 100:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.position = index
        while self.position < index:
            self.capture.grab()
            self.position += 1
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError(f"cannot read video frame {index}")
        self.position += 1
        return image


def view_matrix(center: np.ndarray, scale: float, angle: float) -> np.ndarray:
    """Image pixels to a square panel: rotate by `angle` about `center`, scale, centre in the panel."""
    rotation = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.hstack([rotation, (PANEL_PX / 2 - rotation @ center)[:, None]])


def to_panel(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def polyline(image: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Draw the finite runs of `points` (n, 2)."""
    finite = np.isfinite(points).all(axis=1)
    for run in np.split(np.arange(len(points)), np.flatnonzero(np.diff(finite.astype(int))) + 1):
        if len(run) > 1 and finite[run[0]]:
            cv2.polylines(image, [np.round(points[run]).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def text(image: np.ndarray, label: str, origin: Tuple[int, int], color: Tuple[int, int, int] = WHITE, scale: float = 0.6) -> None:
    """Label on a darkened box, readable over any background."""
    (width, height), baseline = cv2.getTextSize(label, FONT, scale, 1)
    x, y = origin
    box = image[max(y - height - 4, 0) : y + baseline + 2, max(x - 4, 0) : x + width + 4]
    box[:] = box // 3
    cv2.putText(image, label, origin, FONT, scale, color, 1, cv2.LINE_AA)


@dataclass
class Scene:
    rec: Recording
    tracks: Tracks
    size: Tuple[int, int]  # video height, width (px)
    downsample: int = 2  # stimulus grid step, panel px

    def __post_init__(self) -> None:
        rec, t = self.rec, self.tracks
        height, width = self.size
        well = rec.well
        if well is None:
            self.field_view = view_matrix(np.array([width, height]) / 2, PANEL_PX / max(width, height), 0.0)
        else:
            center, radius = np.array([well.x_mm, well.y_mm]) * rec.pix_per_mm, well.radius_mm * rec.pix_per_mm
            self.field_view = view_matrix(center, PANEL_PX / (2.1 * radius), 0.0)
        self.panel_grid = image_coord_grid(PANEL_PX, PANEL_PX, self.downsample)
        self.stimulus_start = np.array([s["timestamp"] for s in rec.stimuli], dtype=np.int64)
        self.trial_start = rec.trials["start_frame"].to_numpy()
        self.tail_range = max(float(np.nanpercentile(np.abs(t.tail_angle), 99.5)), 10.0)
        angles = np.concatenate(list(t.eye_angle.values())) if t.eye_angle else np.array([-30.0, 30.0])
        self.eye_range = tuple(np.nanpercentile(angles, [0.5, 99.5]))

    def frame(self, k: int, image: np.ndarray) -> np.ndarray:
        return np.vstack([np.hstack([self.field_panel(k, image), self.zoom_panel(k, image)]), self.strip(k)])

    def panel(self, k: int, image: np.ndarray, view: np.ndarray) -> np.ndarray:
        """The video seen through `view`, with the stimulus as the shader drew it from the online tracking it used."""
        panel = cv2.warpAffine(image, view, (PANEL_PX, PANEL_PX), flags=cv2.INTER_LINEAR)
        timestamp, online = self.rec.frame_time_ns[k], self.tracks.online[k]
        active = np.searchsorted(self.stimulus_start, timestamp, side="right") - 1
        if active < 0 or not np.isfinite(online).all():
            return panel
        stimulus = self.rec.stimuli[active]
        image_px = to_panel(cv2.invertAffineTransform(view), self.panel_grid)
        coords = egocentric_coords_mm(image_px, online[:2], online[2:4], online[4:6], 1 / self.rec.pix_per_mm)
        drawn = overlay_stimulus(coords[..., 0], coords[..., 1], stim_to_param(stimulus, (1e-9 * timestamp) % rollover_time_sec))
        if drawn is None:
            return panel
        drawn = cv2.resize(np.ascontiguousarray(drawn[..., ::-1], dtype=np.uint8), (PANEL_PX, PANEL_PX), interpolation=cv2.INTER_LINEAR)
        return sum_blend(panel, drawn, 5.0 if stimulus["stim_select"] == Stim.PREY_CAPTURE else 1.0)

    def field_panel(self, k: int, image: np.ndarray) -> np.ndarray:
        rec, t, view = self.rec, self.tracks, self.field_view
        panel = self.panel(k, image, view)
        if rec.well is not None:
            center = to_panel(view, np.array([rec.well.x_mm, rec.well.y_mm]) * rec.pix_per_mm)
            radius = rec.well.radius_mm * rec.pix_per_mm * view[0, 0]
            cv2.circle(panel, tuple(np.round(center).astype(int)), int(round(radius)), GREY, 1, cv2.LINE_AA)
            cv2.drawMarker(panel, tuple(np.round(center).astype(int)), GREY, cv2.MARKER_CROSS, 10, 1)
        trial = np.searchsorted(self.trial_start, k, side="right") - 1
        start = self.trial_start[trial] if trial >= 0 else 0
        polyline(panel, to_panel(view, t.tail[start : k + 1 : 2, 0]), TRAIL_BGR, 1)
        online = t.online[k]
        if np.isfinite(online).all():
            center = to_panel(view, online[:2])
            tip = to_panel(view, online[:2] + online[2:4] * rec.pix_per_mm)
            cv2.drawMarker(panel, tuple(np.round(center).astype(int)), ONLINE_BGR, cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.line(panel, tuple(np.round(center).astype(int)), tuple(np.round(tip).astype(int)), ONLINE_BGR, 2, cv2.LINE_AA)
        if np.isfinite(t.tail[k, 0]).all():
            cv2.circle(panel, tuple(np.round(to_panel(view, t.tail[k, 0])).astype(int)), 4, POSE_BGR, -1, cv2.LINE_AA)

        row = rec.trials.iloc[trial] if trial >= 0 else None
        label = f"{t.time_s[k]:7.2f} s" + (f"   {row.epoch_name}  #{row.trial_num + 1}" if row is not None else "")
        text(panel, label, (12, 28))
        text(panel, f"{QC_LABEL}: {'agree' if t.tracking_ok[k] else 'mismatch'}", (12, PANEL_PX - 14),
             OK_BGR if t.tracking_ok[k] else BAD_BGR, 0.5)
        text(panel, "x online", (PANEL_PX - 190, PANEL_PX - 14), ONLINE_BGR, 0.5)
        text(panel, "o pose", (PANEL_PX - 95, PANEL_PX - 14), POSE_BGR, 0.5)
        return panel

    def zoom_panel(self, k: int, image: np.ndarray) -> np.ndarray:
        rec, t = self.rec, self.tracks
        around = slice(max(k - 3, 0), k + 4)
        heading = np.nanmean(t.head[around] - t.tail[around, 0], axis=0)
        center = t.tail[k, 0] if np.isfinite(t.tail[k, 0]).all() else t.online[k, :2]
        if not (np.isfinite(heading).all() and np.isfinite(center).all()):
            return np.zeros((PANEL_PX, PANEL_PX, 3), np.uint8)
        view = view_matrix(center, PANEL_PX / (ZOOM_MM * rec.pix_per_mm), -np.pi / 2 - np.arctan2(heading[1], heading[0]))
        panel = self.panel(k, image, view)

        panel = self.draw_fish(panel, k, view)

        time_s = t.time_s[k]
        bout = t.bouts[(t.bouts.onset <= k) & (k < t.bouts.offset)]
        if len(bout):
            b = next(bout.itertuples())
            direction = EventDirection(b.sign).name.lower() if b.sign in (-1, 1) else ""
            text(panel, f"{bouts_category_name[b.category]} {direction}", (12, 30), BOUT_BGR[b.category], 0.8)
            cv2.rectangle(panel, (12, 42), (172, 50), (60, 60, 60), -1)
            cv2.rectangle(panel, (12, 42), (12 + int(160 * b.proba), 50), BOUT_BGR[b.category], -1)
            text(panel, f"p = {b.proba:.2f}", (182, 51), BOUT_BGR[b.category], 0.5)
        recent = t.saccades[(t.saccades.onset_s <= time_s) & (time_s < t.saccades.onset_s + SACCADE_LABEL_S)]
        if len(recent):
            cluster = int(recent.cluster.iloc[-1])
            text(panel, f"saccade: {SACCADE_CLASS_NAMES.get(cluster, cluster)}", (12, 84), SACCADE_BGR.get(cluster, WHITE), 0.7)

        bar = PANEL_PX / ZOOM_MM
        cv2.line(panel, (16, PANEL_PX - 20), (16 + int(bar), PANEL_PX - 20), WHITE, 3)
        text(panel, "1 mm", (16, PANEL_PX - 30), WHITE, 0.5)
        return panel

    def draw_fish(self, panel: np.ndarray, k: int, view: np.ndarray) -> np.ndarray:
        """Translucent SLEAP/DLC style skeleton: body in one colour, each eye in its time-series colour."""
        t = self.tracks
        parts = [(to_panel(view, np.vstack([t.head[k], t.tail[k]])), POSE_BGR)]
        if t.eyes and np.isfinite(t.head[k]).all() and np.isfinite(t.tail[k, 0]).all():
            to_frame = cv2.invertAffineTransform(eye_crop_matrix(t.head[k], t.tail[k, 0], eye_crop_size()))
            parts += [(to_panel(view, to_panel(to_frame, points[k])), EYE_BGR[side]) for side, points in t.eyes.items()]
        layer = panel.copy()
        for points, color in parts:
            for p0, p1 in zip(points[:-1], points[1:]):
                gap = (p1 - p0) * NODE_RADIUS / max(np.hypot(*(p1 - p0)), 2 * NODE_RADIUS)  # edges stop at the node rims
                if np.isfinite(gap).all():
                    cv2.line(layer, tuple(np.round(p0 + gap).astype(int)), tuple(np.round(p1 - gap).astype(int)), color, 2, cv2.LINE_AA)
            for point in points[np.isfinite(points).all(axis=1)]:
                cv2.circle(layer, tuple(np.round(point).astype(int)), NODE_RADIUS, color, 2, cv2.LINE_AA)
        return cv2.addWeighted(layer, SKELETON_ALPHA, panel, 1 - SKELETON_ALPHA, 0)

    def strip(self, k: int) -> np.ndarray:
        t = self.tracks
        strip = np.zeros((STRIP_PX, 2 * PANEL_PX, 3), np.uint8)
        width = 2 * PANEL_PX - MARGIN_PX - 10
        start, stop = t.time_s[k] - WINDOW_S / 2, t.time_s[k] + WINDOW_S / 2
        window = slice(*np.searchsorted(t.time_s, [start, stop]))
        x_of = lambda s: MARGIN_PX + (np.asarray(s) - start) / WINDOW_S * width
        rows = {"tail": (26, 124), "eyes": (160, 258), "QC": (282, 292)}  # labels go in the lane above each row

        def y_of(values: np.ndarray, row: str, low: float, high: float) -> np.ndarray:
            top, bottom = rows[row]
            return bottom - (np.clip(values, low, high) - low) / (high - low) * (bottom - top)

        top, bottom = rows["tail"]
        for b in t.bouts[(t.bouts.offset > window.start) & (t.bouts.onset < window.stop)].itertuples():
            x0, x1 = x_of(t.time_s[b.onset]), x_of(t.time_s[min(b.offset, len(t.time_s) - 1)])
            shade = tuple(int(0.45 * c) for c in BOUT_BGR[b.category])
            x0, x1 = np.clip([x0, x1], MARGIN_PX, MARGIN_PX + width).astype(int)
            cv2.rectangle(strip, (x0, top), (x1, bottom), shade, -1)
            text(strip, bouts_category_name_short[b.category], (x0 + 3, top - 6), BOUT_BGR[b.category], 0.42)
            cv2.rectangle(strip, (x0, bottom + 3), (x1, bottom + 7), (60, 60, 60), -1)
            cv2.rectangle(strip, (x0, bottom + 3), (x0 + int(b.proba * (x1 - x0)), bottom + 7), BOUT_BGR[b.category], -1)
        x = x_of(t.time_s[window])
        polyline(strip, np.column_stack([x, y_of(t.tail_angle[window], "tail", -self.tail_range, self.tail_range)]), WHITE, 1)

        for side, angle in t.eye_angle.items():
            polyline(strip, np.column_stack([x, y_of(angle[window], "eyes", *self.eye_range)]), EYE_BGR[side], 1)
        top, bottom = rows["eyes"]
        for s in t.saccades[(t.saccades.onset_s >= start) & (t.saccades.onset_s < stop)].itertuples():
            x_s, color = int(x_of(s.onset_s)), SACCADE_BGR.get(int(s.cluster), WHITE)
            cv2.line(strip, (x_s, top), (x_s, bottom), color, 2)
            text(strip, SACCADE_CLASS_NAMES.get(int(s.cluster), str(s.cluster)), (x_s + 4, top - 6), color, 0.42)

        top, bottom = rows["QC"]
        for x0, x1, ok in zip(x[:-1], x[1:], t.tracking_ok[window][:-1]):
            cv2.rectangle(strip, (int(x0), top), (int(np.ceil(x1)), bottom), OK_BGR if ok else BAD_BGR, -1)

        for row, (top, bottom) in rows.items():
            text(strip, row, (8, (top + bottom) // 2 + 5), GREY, 0.5)
        for i, (side, color) in enumerate(EYE_BGR.items()):
            text(strip, side, (8, rows["eyes"][0] + 14 + 70 * i), color, 0.45)
        center = int(x_of(t.time_s[k]))
        cv2.line(strip, (center, 4), (center, STRIP_PX - 4), WHITE, 1)
        text(strip, f"window {WINDOW_S:g} s", (MARGIN_PX + width - 110, rows["QC"][0] - 8), GREY, 0.45)
        text(strip, f"{QC_LABEL}:", (MARGIN_PX, rows["QC"][0] - 8), GREY, 0.42)
        text(strip, "agree", (MARGIN_PX + 200, rows["QC"][0] - 8), OK_BGR, 0.42)
        text(strip, "mismatch", (MARGIN_PX + 255, rows["QC"][0] - 8), BAD_BGR, 0.42)
        return strip


def render(scene: Scene, reader: FrameReader, output: Path, start_s: float, duration_s: float, speed: float, fps: float) -> None:
    rec, time_s = scene.rec, scene.tracks.time_s
    frames = np.unique(nearest_indices(time_s, start_s + np.arange(0, duration_s, speed / fps)))
    writer = FFMPEG_VideoWriter_CPU(filename=output, height=PANEL_PX + STRIP_PX, width=2 * PANEL_PX, fps=fps, q=18)
    try:
        for k in tqdm(frames, desc=rec.name):
            writer.write_frame(scene.frame(k, reader.read(int(rec.video_frames[k]))))
    finally:
        writer.close()


def main(args: argparse.Namespace) -> None:
    directories = Directories(
        args.root, metadata=args.metadata, stimuli=args.stimuli, tracking=args.tracking,
        full_tracking=args.lightning_pose, eyes_tracking=args.lightning_pose,
        video=args.video, video_timestamp=args.video_timestamp,
    )
    recordings = [Recording(f) for f in find_files(directories) if args.fish in f.metadata.stem]
    if not recordings:
        raise FileNotFoundError(f"no recording matches {args.fish!r} in {directories.metadata}")
    saccades = pd.read_csv(args.root / args.saccades) if args.saccades else None
    output_dir = args.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for rec in recordings:
        reader = FrameReader(rec.files.video)
        scene = Scene(rec, load_tracks(rec, saccades), reader.size, args.downsample)
        output = output_dir / f"{rec.name}_{args.start:g}s.mp4"
        render(scene, reader, output, args.start, args.duration, args.speed, args.fps)
        print(f"Saved {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay video of a recording and everything the pipeline extracted from it.")
    parser.add_argument("root", type=Path, help="Root experiment folder")
    parser.add_argument("--fish", required=True, help="Part of the recording name (metadata file stem)")
    parser.add_argument("--saccades", default="", help="Classified saccade table relative to root (fish, onset_time_s, cluster)")
    parser.add_argument("--start", type=float, default=0.0, help="Window start, s from the first frame")
    parser.add_argument("--duration", type=float, default=30.0, help="Window length, s of recording")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed (0.5: half speed)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output frames per second")
    parser.add_argument("--downsample", type=int, default=2, help="Stimulus reconstruction grid step (panel px)")
    parser.add_argument("--output-dir", default="overlay")
    parser.add_argument("--metadata", default="results")
    parser.add_argument("--stimuli", default="results")
    parser.add_argument("--tracking", default="results")
    parser.add_argument("--lightning-pose", default="lightning_pose")
    parser.add_argument("--video", default="results")
    parser.add_argument("--video-timestamp", default="results")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
