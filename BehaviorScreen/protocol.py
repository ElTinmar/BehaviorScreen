from BehaviorScreen.core import BoutSign, Laterality, Stim
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

EpochName = str
ProtocolEntry = Tuple[EpochName, Stim]

## PROTOCOLS ---------------------------------------------------

# Full protocol
protocol: List[ProtocolEntry] = [
    ("adaptation", Stim.BRIGHT), 
    ("ramp 0", Stim.RAMP)
]
protocol += 5 * [
    ("prey capture right", Stim.PREY_CAPTURE),
    ("prey capture break after right", Stim.DARK), 
    ("prey capture left", Stim.PREY_CAPTURE), 
    ("prey capture break after left", Stim.DARK)
]
protocol += [("ramp 1", Stim.RAMP)]
protocol += 10 * [
    ("phototaxis bright right", Stim.PHOTOTAXIS), 
    ("phototaxis break after bright right", Stim.BRIGHT), 
    ("phototaxis bright left", Stim.PHOTOTAXIS), 
    ("phototaxis break after bright left", Stim.BRIGHT)
]
protocol += [("ramp 2", Stim.RAMP)]
protocol += 10 * [("spontaneous dark", Stim.DARK)]
protocol += [("ramp 3", Stim.RAMP)]
protocol += 5 * [
    ("flash dark", Stim.DARK), 
    ("flash ramp", Stim.RAMP), 
    ("flash bright", Stim.BRIGHT)
]
protocol += [("ramp 4", Stim.RAMP)]
protocol += 5 * [
    ("grating right", Stim.OMR), 
    ("grating break after right", Stim.BRIGHT), 
    ("grating left", Stim.OMR), 
    ("grating break after left", Stim.BRIGHT), 
    ("grating forward", Stim.OMR), 
    ("grating break after forward", Stim.BRIGHT)
]
protocol += [("ramp 5", Stim.RAMP)]
protocol += 10 * [("spontaneous bright", Stim.BRIGHT)]
protocol += [("ramp 6", Stim.RAMP)]
protocol += 5 * [
    ("pinwheel clockwise", Stim.OKR), 
    ("pinwheel break after clockwise", Stim.BRIGHT), 
    ("pinwheel counter-clockwise", Stim.OKR), 
    ("pinwheel break after counter-clockwise", Stim.BRIGHT)
]
protocol += [("ramp 7", Stim.RAMP)]
protocol += 7 * [
    ("looming left", Stim.LOOMING), 
    ("looming break after left", Stim.BRIGHT), 
    ("looming right", Stim.LOOMING), 
    ("looming break after right", Stim.BRIGHT)
]

# phototaxis only
protocol_ptx: List[ProtocolEntry] = [("adaptation", Stim.BRIGHT)]
protocol_ptx += 10 * [
    ("phototaxis bright right", Stim.PHOTOTAXIS), 
    ("phototaxis break after bright right", Stim.BRIGHT), 
    ("phototaxis bright left", Stim.PHOTOTAXIS), 
    ("phototaxis break after bright left", Stim.BRIGHT)
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

non_directional_epoch = [
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

for epoch in non_directional_epoch:
    for sign in BoutSign:
        EPOCH_LATERALITY[(epoch, sign)] = Laterality.NONDIRECTIONAL


###

@dataclass(frozen=True)
class EpochSpec:
    name: EpochName
    stim: Stim
    expected_repeats: int
    parameters: Optional[Dict[str, List[Any]]] = None


def _build_epoch_specs(protocol_entries: List[ProtocolEntry]) -> List[EpochSpec]:
    counts = {}
    for epoch_name, _ in protocol_entries:
        counts[epoch_name] = counts.get(epoch_name, 0) + 1

    seen = set()
    specs = []
    for epoch_name, stim in protocol_entries:
        if epoch_name in seen:
            continue
        seen.add(epoch_name)
        specs.append(EpochSpec(name=epoch_name, stim=stim, expected_repeats=counts[epoch_name]))
    return specs


PROTOCOL_SPEC: List[EpochSpec] = _build_epoch_specs(protocol)
PROTOCOL_PTX_SPEC: List[EpochSpec] = _build_epoch_specs(protocol_ptx)