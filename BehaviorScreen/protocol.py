from BehaviorScreen.core import BoutSign, Laterality, Stim
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from collections import Counter

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

EPOCH_LATERALITY: Dict[Tuple[EpochName, BoutSign], Laterality] = {
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
        EPOCH_LATERALITY[(stim, sign)] = Laterality.NONDIRECTIONAL


###

@dataclass(frozen=True)
class EpochSpec:
    name: EpochName
    stim: Stim
    expected_repeats: int
    parameters: Optional[Dict[str, List[Any]]] = None


EPOCH_STIM: Dict[EpochName, Stim] = {
    "adaptation": Stim.DARK,                    
    "ramp 0": Stim.RAMP, "ramp 1": Stim.RAMP, "ramp 2": Stim.RAMP,
    "ramp 3": Stim.RAMP, "ramp 4": Stim.RAMP, "ramp 5": Stim.RAMP,
    "ramp 6": Stim.RAMP, "ramp 7": Stim.RAMP,
    "prey capture right": Stim.PREY_CAPTURE,
    "prey capture break after right": Stim.PREY_CAPTURE,
    "prey capture left": Stim.PREY_CAPTURE,
    "prey capture break after left": Stim.PREY_CAPTURE,
    "phototaxis bright right": Stim.PHOTOTAXIS,
    "phototaxis break after bright right": Stim.PHOTOTAXIS,
    "phototaxis bright left": Stim.PHOTOTAXIS,
    "phototaxis break after bright left": Stim.PHOTOTAXIS,
    "spontaneous dark": Stim.DARK,
    "flash dark": Stim.DARK, "flash ramp": Stim.RAMP, "flash bright": Stim.BRIGHT,
    "grating right": Stim.OMR, "grating break after right": Stim.OMR,
    "grating left": Stim.OMR, "grating break after left": Stim.OMR,
    "grating forward": Stim.OMR, "grating break after forward": Stim.OMR,
    "spontaneous bright": Stim.BRIGHT,
    "pinwheel clockwise": Stim.OKR, "pinwheel break after clockwise": Stim.OKR,
    "pinwheel counter-clockwise": Stim.OKR,
    "pinwheel break after counter-clockwise": Stim.OKR,
    "looming left": Stim.LOOMING, "looming break after left": Stim.LOOMING,
    "looming right": Stim.LOOMING, "looming break after right": Stim.LOOMING,
}


def _build_epoch_specs(epoch_names: List[EpochName]) -> List[EpochSpec]:
    """expected_repeats derived directly from the flattened protocol list --
    protocol/protocol_ptx stay the single source of truth, no counts to keep
    in sync by hand."""
    counts = Counter(epoch_names)
    seen = set()
    specs = []
    for name in epoch_names:
        if name in seen:
            continue
        seen.add(name)
        specs.append(EpochSpec(name=name, stim=EPOCH_STIM[name], expected_repeats=counts[name]))
    return specs


PROTOCOL_SPEC: List[EpochSpec] = _build_epoch_specs(protocol)
PROTOCOL_PTX_SPEC: List[EpochSpec] = _build_epoch_specs(protocol_ptx)