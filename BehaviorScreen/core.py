from enum import IntEnum
from typing import TypedDict

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

COLORS = ("#F1500A", "#0A6EF1")
