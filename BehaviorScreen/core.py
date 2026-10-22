from enum import IntEnum
from pathlib import Path
from typing import Callable, TypedDict

TIME_TOLERANCE_S: float = 40
ROLLOVER_TIME_S: int = 3600  # ZebVR shader clock period


def write_complete(target: Path, write: Callable[[Path], None]) -> None:
    """
    write(path) into .partial/ next to target, then move the file to target: an
    interrupted write never leaves a file that looks finished.
    """
    partial = target.parent / ".partial" / target.name
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    write(partial)
    if not partial.exists():
        raise RuntimeError(f"{target.name} was not written")
    partial.replace(target)


class MyIntEnum(IntEnum):
    def __str__(self) -> str:
        return self.name


# Valid for my data, needs checking
class EventDirection(MyIntEnum):
    LEFT = -1
    RIGHT = 1

# Event laterality with respect to stimulus
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
    "looming_angle_start_deg",
    "looming_angle_stop_deg",
    "looming_center_mm",
    "looming_distance_to_screen_mm",
    "looming_expansion_speed_deg_per_sec",
    "looming_expansion_speed_mm_per_sec",
    "looming_expansion_time_sec",
    "looming_period_sec",
    "looming_size_to_speed_ratio_ms",
    "looming_type",
    "n_preys",
    "okr_spatial_frequency_deg",
    "okr_speed_deg_per_sec",
    "omr_angle_deg",
    "omr_spatial_period_mm",
    "omr_speed_mm_per_sec",
    "phototaxis_polarity",
    "prey_arc_start_deg",
    "prey_arc_stop_deg",
    "prey_capture_type",
    "prey_radius_mm",
    "prey_speed_deg_s",
    "prey_speed_mm_s",
    "prey_trajectory_radius_mm",
    "ramp_duration_sec",
    "ramp_powerlaw_exponent",
    "ramp_type",
    "start_time_sec",
]

## Physical dimensions of the experimental arenas


class WellDimensions(TypedDict):
    well_radius_mm: float
    distance_between_well_centers_mm: float


AGAROSE_WELL_DIMENSIONS: WellDimensions = {
    "well_radius_mm": 19.5 / 2,
    "distance_between_well_centers_mm": 22,
}

SACCADE_CLASS_NAMES = {
    -1: "Unassigned",
    0: "Unclassified",
    1: "Conjugate left",  # NOTE left is probably right in my setup
    2: "Conjugate right",
    3: "Miniature convergent",
    4: "Convergent",
    5: "Non-saccadic",
    6: "Divergent",
    7: "Biphasic convergent right",
    8: "Biphasic convergent left",
}

# Class colours of Dowell et al. 2024
SACCADE_CLASS_COLORS = {
    0: "#bdbdbd",
    1: "#66c2a5",
    2: "#e78ac3",
    3: "#d95f02",
    4: "#377eb8",
    5: "#969696",
    6: "#8c510a",
    7: "#1f78b4",
    8: "#00bfc4",
}

# NOTE flipped for my dataset
SACCADE_CLASS_DIRECTIONS = {
    1: EventDirection.RIGHT,
    2: EventDirection.LEFT,
    7: EventDirection.LEFT,
    8: EventDirection.RIGHT,
}


SACCADE_NONDIRECTIONAL_CLASSES = {
    3,  # Miniature convergent
    4,  # Convergent
    6,  # Divergent
}

SACCADE_CATEGORY_BY_CLUSTER = {
    -1: -1,  # Unassigned
    0: 0,  # Unclassified
    1: 1,  # Conjugate left  -> Conjugate
    2: 1,  # Conjugate right -> Conjugate
    3: 3,  # Miniature convergent
    4: 4,  # Convergent
    5: 5,  # Non-saccadic
    6: 6,  # Divergent
    7: 7,  # Biphasic convergent right -> Biphasic convergent
    8: 7,  # Biphasic convergent left  -> Biphasic convergent
}

SACCADE_CATEGORY_NAMES = {
    -1: "Unassigned",
    0: "Unclassified",
    1: "Conjugate",
    3: "Miniature convergent",
    4: "Convergent",
    5: "Non-saccadic",
    6: "Divergent",
    7: "Biphasic convergent",
}
