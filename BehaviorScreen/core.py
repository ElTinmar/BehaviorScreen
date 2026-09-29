from enum import IntEnum
from typing import TypedDict

TIME_TOLERANCE_S: float = 40

class MyIntEnum(IntEnum):
    def __str__(self):
        return self.name
    
# Valid for my data, needs checking
class EventDirection(MyIntEnum):
    LEFT = -1 
    RIGHT = 1 

class Laterality(MyIntEnum):
    IPSILATERAL = 1
    NONDIRECTIONAL = 0
    CONTRALATERAL = -1

## Stim and stim parameters (from ZebVR)

class Stim(MyIntEnum):
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

## Physical dimensions of the experimental arenas

class WellDimensions(TypedDict):
    well_radius_mm: float
    distance_between_well_centers_mm: float
    
AGAROSE_WELL_DIMENSIONS: WellDimensions = {
    'well_radius_mm': 19.5/2,
    'distance_between_well_centers_mm': 22
}

SACCADE_CLASS_NAMES = {
    -1: "Unassigned",
    0: "Unclassified",
    1: "Conjugate left",
    2: "Conjugate right",
    3: "Miniature convergent",
    4: "Convergent",
    5: "Non-saccadic",
    6: "Divergent",
    7: "Biphasic convergent right",
    8: "Biphasic convergent left",
}

SACCADE_CLASS_DIRECTIONS = {
    1: EventDirection.LEFT,
    2: EventDirection.RIGHT,
    7: EventDirection.RIGHT,
    8: EventDirection.LEFT
}