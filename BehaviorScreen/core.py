from enum import IntEnum
from typing import TypedDict
from pathlib import Path

class Stim(IntEnum):

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

    def __str__(self):
        return self.name
    
class WellDimensions(TypedDict):
    well_radius_mm: float
    distance_between_well_centers_mm: float
    
AGAROSE_WELL_DIMENSIONS: WellDimensions = {
    'well_radius_mm': 19.5/2,
    'distance_between_well_centers_mm': 22
}

GROUPING_PARAMETER = {
    Stim.DARK: 'background_color',
    Stim.BRIGHT: 'foreground_color',
    Stim.PHOTOTAXIS: 'phototaxis_polarity',
    Stim.OMR: 'omr_angle_deg',
    Stim.OKR: 'okr_speed_deg_per_sec',
    Stim.LOOMING: 'looming_center_mm_x',
    Stim.PREY_CAPTURE: 'prey_arc_start_deg'
}

STIM_PARAMETERS = [
    'looming_angle_start_deg',
    'looming_angle_stop_deg',
    'looming_center_mm',
    'looming_distance_to_screen_mm',
    'looming_expansion_speed_deg_per_sec',
    'looming_expansion_speed_mm_per_sec',
    'looming_expansion_time_sec',
    'looming_period_sec',
    'looming_size_to_speed_ratio_ms',
    'looming_type',
    'n_preys',
    'okr_spatial_frequency_deg',
    'okr_speed_deg_per_sec',
    'omr_angle_deg',
    'omr_spatial_period_mm',
    'omr_speed_mm_per_sec',
    'phototaxis_polarity',
    'prey_arc_start_deg',
    'prey_arc_stop_deg',
    'prey_capture_type',
    'prey_radius_mm',
    'prey_speed_deg_s',
    'prey_speed_mm_s',
    'prey_trajectory_radius_mm',
    'ramp_duration_sec',
    'ramp_powerlaw_exponent',
    'ramp_type',
    'start_time_sec'
]

COLORS = ("#F1500A", "#0A6EF1", "#333333")

#ROOT_FOLDER = Path('/media/martin/DATA1/Behavioral_screen')
ROOT_FOLDER = Path('/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen')
#ROOT_FOLDER = Path('/home/martin/Downloads/')
BASE_DIR = ROOT_FOLDER / 'output'

MODELS_URL = "https://figshare.unimelb.edu.au/ndownloader/articles/29275838/versions/2"
MODELS_FOLDER = ROOT_FOLDER / 'SLEAP_DLC'


NUM_PROCESSES = 6

