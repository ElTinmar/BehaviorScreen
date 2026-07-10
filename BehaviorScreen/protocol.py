from BehaviorScreen.core import BoutSign, Laterality
from typing import Dict, Tuple, List

EpochName = str

## PROTOCOLS ---------------------------------------------------

# Full protocol
protocol: List[EpochName] = ["adaptation", "ramp 0"]
protocol += 5 * [
    "prey capture right", 
    "prey capture break after right", 
    "prey capture left", 
    "prey capture break after left"
]
protocol += ["ramp 1"]
protocol += 10 * [
    "phototaxis bright right", 
    "phototaxis break after bright right", 
    "phototaxis bright left", 
    "phototaxis break after bright left"
]
protocol += ["ramp 2"]
protocol += 10 * ["spontaneous dark"]
protocol += ["ramp 3"]
protocol += 5 * ["flash dark", "flash ramp", "flash bright"]
protocol += ["ramp 4"]
protocol += 5 * [
    "grating right", 
    "grating break after right", 
    "grating left", 
    "grating break after left", 
    "grating forward", 
    "grating break after forward"
]
protocol += ["ramp 5"]
protocol += 10 * ["spontaneous bright"]
protocol += ["ramp 6"]
protocol += 5 * [
    "pinwheel clockwise", 
    "pinwheel break after clockwise", 
    "pinwheel counter-clockwise", 
    "pinwheel break after counter-clockwise"
]
protocol += ["ramp 7"]
protocol += 7 * [
    "looming left", 
    "looming break after left", 
    "looming right", 
    "looming break after right"
]

# phototaxis only
protocol_ptx = ["adaptation"]
protocol_ptx += 10 * [
    "phototaxis bright right", 
    "phototaxis break after bright right", 
    "phototaxis bright left", 
    "phototaxis break after bright left"
]

### LATERALITY -------------------------------------------------------

STIM_LATERALITY: Dict[Tuple[EpochName, BoutSign], Laterality] = {
    ("prey capture right", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("prey capture right", BoutSign.RIGHT): Laterality.IPSILATERAL,
    ("prey capture break after right", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("prey capture break after right", BoutSign.RIGHT): Laterality.IPSILATERAL,
    ("prey capture left", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("prey capture left", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("prey capture break after left", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("prey capture break after left", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("phototaxis bright right", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("phototaxis bright right", BoutSign.RIGHT): Laterality.IPSILATERAL,
    ("phototaxis break after bright right", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("phototaxis break after bright right", BoutSign.RIGHT): Laterality.IPSILATERAL,
    ("phototaxis bright left", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("phototaxis bright left", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("phototaxis break after bright left", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("phototaxis break after bright left", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("grating right", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("grating right", BoutSign.RIGHT): Laterality.IPSILATERAL,
    ("grating break after right", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("grating break after right", BoutSign.RIGHT): Laterality.IPSILATERAL,
    ("grating left", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("grating left", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("grating break after left", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("grating break after left", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("pinwheel clockwise", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("pinwheel clockwise", BoutSign.RIGHT): Laterality.IPSILATERAL,
    ("pinwheel break after clockwise", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("pinwheel break after clockwise", BoutSign.RIGHT): Laterality.IPSILATERAL,
    ("pinwheel counter-clockwise", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("pinwheel counter-clockwise", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("pinwheel break after counter-clockwise", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("pinwheel break after counter-clockwise", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("looming left", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("looming left", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("looming break after left", BoutSign.LEFT): Laterality.IPSILATERAL,
    ("looming break after left", BoutSign.RIGHT): Laterality.CONTRALATERAL,
    ("looming right", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("looming right", BoutSign.RIGHT):  Laterality.IPSILATERAL,
    ("looming break after right", BoutSign.LEFT): Laterality.CONTRALATERAL,
    ("looming break after right", BoutSign.RIGHT):  Laterality.IPSILATERAL
}

non_directional_stim = [
    "adaptation", 
    "ramp 0",
    "ramp 1",
    "ramp 2",
    "spontaneous dark",
    "ramp 3",
    "flash dark", 
    "flash ramp", 
    "flash bright",
    "ramp 4",
    "grating forward", 
    "grating break after forward",
    "ramp 5",
    "spontaneous bright",
    "ramp 6",
    "ramp 7"
]

for stim in non_directional_stim:
    for sign in BoutSign:
        STIM_LATERALITY[(stim, sign)] = Laterality.NONDIRECTIONAL
