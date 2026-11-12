from enum import IntEnum


class MyIntEnum(IntEnum):
    def __str__(self) -> str:
        return self.name


# In the rig's image (preprocessing/rig.yaml)
class EventDirection(MyIntEnum):
    LEFT = -1
    RIGHT = 1

# Event laterality with respect to stimulus
class Laterality(MyIntEnum):
    IPSILATERAL = 1
    NONDIRECTIONAL = 0
    CONTRALATERAL = -1


## Stim (from ZebVR)


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

# As named by Dowell et al. 2024; a rig's own left and right: preprocessing/rig.yaml
SACCADE_REFERENCE_DIRECTIONS = {
    1: EventDirection.LEFT,
    2: EventDirection.RIGHT,
    7: EventDirection.RIGHT,
    8: EventDirection.LEFT,
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
