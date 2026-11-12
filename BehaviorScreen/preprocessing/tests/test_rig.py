"""Left and right in the rig's image (rig.yaml), and the laterality of bouts and saccades that follows from it."""
import numpy as np
import pytest

from BehaviorScreen.core import SACCADE_CLASS_NAMES, SACCADE_NONDIRECTIONAL_CLASSES, SACCADE_REFERENCE_DIRECTIONS, EventDirection, Laterality
from BehaviorScreen.preprocessing.protocol import bout_laterality, saccade_laterality
from BehaviorScreen.preprocessing.rig import RIG, SACCADE_CLASS_DIRECTIONS, SACCADE_NAMES, bout_direction, saccade_directions, saccade_names

LEFT, RIGHT = EventDirection.LEFT, EventDirection.RIGHT
IPSI, CONTRA, NONE = Laterality.IPSILATERAL, Laterality.CONTRALATERAL, Laterality.NONDIRECTIONAL


def test_reference_directions_are_those_of_the_class_names() -> None:
    assert SACCADE_REFERENCE_DIRECTIONS == {cluster: LEFT if "left" in name else RIGHT for cluster, name in SACCADE_CLASS_NAMES.items() if "left" in name or "right" in name}
    assert saccade_directions(mirrored=False) == SACCADE_REFERENCE_DIRECTIONS


def test_mirrored_rig_swaps_left_and_right() -> None:
    assert saccade_directions(mirrored=True) == {1: RIGHT, 2: LEFT, 7: LEFT, 8: RIGHT}


@pytest.mark.parametrize("mirrored", [False, True])
def test_names_say_the_direction_in_the_rig(mirrored: bool) -> None:
    names, directions = saccade_names(mirrored), saccade_directions(mirrored)
    assert {cluster: LEFT if "left" in name else RIGHT for cluster, name in names.items() if "left" in name or "right" in name} == directions
    assert {c: n for c, n in names.items() if c not in directions} == {c: n for c, n in SACCADE_CLASS_NAMES.items() if c not in directions}


def test_bout_direction_follows_the_sign_of_a_leftward_bout() -> None:
    assert (bout_direction(-1, sign_left=-1), bout_direction(1, sign_left=-1)) == (LEFT, RIGHT)
    assert (bout_direction(1, sign_left=1), bout_direction(-1, sign_left=1)) == (LEFT, RIGHT)
    assert bout_direction(np.array([1, -1, 1]), sign_left=1).tolist() == [-1, 1, -1]


def test_this_rig() -> None:
    """The setup of the screen: a change of rig.yaml shows here."""
    assert RIG["orientation"] == {"saccade_classes_mirrored": True, "bout_sign_left": -1}
    assert SACCADE_CLASS_DIRECTIONS == saccade_directions(mirrored=True)
    assert (SACCADE_NAMES[1], SACCADE_NAMES[2], SACCADE_NAMES[7]) == ("Conjugate right", "Conjugate left", "Biphasic convergent left")
    assert RIG["well"] == {"well_radius_mm": 9.75, "distance_between_well_centers_mm": 22}
    assert RIG["eye_crop"] == {"size_mm": 1.6, "px_per_mm": 40.0} and RIG["shader_clock_period_s"] == 3600


@pytest.mark.parametrize("epoch, toward", [
    ("grating right", RIGHT), ("grating left", LEFT), ("pinwheel clockwise", RIGHT), ("pinwheel counter-clockwise", LEFT),
    ("looming left", LEFT), ("phototaxis bright right", RIGHT), ("prey capture left", LEFT),
])
def test_events_toward_the_side_of_the_stimulus_are_ipsilateral(epoch: str, toward: EventDirection) -> None:
    assert bout_laterality(epoch, int(toward)) == IPSI and bout_laterality(epoch, -int(toward)) == CONTRA
    for cluster, direction in SACCADE_CLASS_DIRECTIONS.items():
        assert saccade_laterality(epoch, cluster) == (IPSI if direction == toward else CONTRA)
    assert all(saccade_laterality(epoch, cluster) == NONE for cluster in SACCADE_NONDIRECTIONAL_CLASSES)


def test_events_without_a_side_have_no_laterality() -> None:
    assert np.isnan(bout_laterality("grating right", 0)) and np.isnan(bout_laterality("not an epoch", 1))
    assert all(np.isnan(saccade_laterality("grating right", cluster)) for cluster in (-1, 0, 5, "x"))
    assert np.isnan(saccade_laterality(None, 1))
