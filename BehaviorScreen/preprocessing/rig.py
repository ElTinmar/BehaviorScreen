"""The recording setup (rig.yaml), and the left and right of events in its image."""
from pathlib import Path
from typing import Dict, Union

import numpy as np

from BehaviorScreen.core import SACCADE_CLASS_NAMES, SACCADE_REFERENCE_DIRECTIONS, EventDirection
from BehaviorScreen.filters import load_yaml_config

RIG = load_yaml_config(Path(__file__).with_name("rig.yaml"))
if RIG["orientation"]["bout_sign_left"] not in (-1, 1):
    raise ValueError("rig.yaml: bout_sign_left must be -1 or 1")


def saccade_directions(mirrored: bool) -> Dict[int, EventDirection]:
    """Direction of each directional saccade class in the rig's image."""
    return {cluster: EventDirection(-direction if mirrored else direction) for cluster, direction in SACCADE_REFERENCE_DIRECTIONS.items()}


def saccade_names(mirrored: bool) -> Dict[int, str]:
    """Name of each saccade class with the direction it has in the rig's image."""
    swap = {"left": "right", "right": "left"} if mirrored else {}
    return {cluster: " ".join(swap.get(word, word) for word in name.split()) for cluster, name in SACCADE_CLASS_NAMES.items()}


def bout_direction(sign: Union[int, np.ndarray], sign_left: int = RIG["orientation"]["bout_sign_left"]) -> Union[int, np.ndarray]:
    """EventDirection code (LEFT = -1) of a megabouts bout sign."""
    return -sign_left * sign


SACCADE_CLASS_DIRECTIONS = saccade_directions(RIG["orientation"]["saccade_classes_mirrored"])
SACCADE_NAMES = saccade_names(RIG["orientation"]["saccade_classes_mirrored"])
