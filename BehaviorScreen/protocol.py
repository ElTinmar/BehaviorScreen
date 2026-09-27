from BehaviorScreen.core import BoutSign, Laterality, Stim
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

EpochName = str


@dataclass(frozen=True)
class Epoch:
    """
    One declared epoch: its Stim, and -- for directional stimuli -- which
    BoutSign counts as IPSILATERAL while it's showing (e.g. BoutSign.RIGHT
    for "...right"/"...clockwise" epochs). Leave ipsi_sign=None for
    non-directional epochs (both bout signs then map to NONDIRECTIONAL).
    """
    name: EpochName
    stim: Stim
    ipsi_sign: Optional[BoutSign] = None


_REGISTRY: Dict[EpochName, Epoch] = {}


def epoch(name: EpochName, stim: Stim, ipsi_sign: Optional[BoutSign] = None) -> Epoch:
    """
    Declare (or re-fetch) one epoch, for use inside `protocol`/`protocol_ptx`.

    This is now the ONLY place a given epoch_name's stim/laterality side is
    declared. Previously the same name had to be retyped identically in
    `protocol`, in a hand-written laterality dict, and (for non-directional
    epochs) in a separate name list -- with nothing checking the three
    stayed in sync. Re-declaring the SAME name (protocol_ptx reuses several
    names already declared while building `protocol`) is fine as long as
    stim/ipsi_sign agree; if they don't, that's a genuine inconsistency and
    raises immediately here, instead of silently producing a wrong/missing
    laterality entry discovered only later, deep in megabouts.py.
    """
    existing = _REGISTRY.get(name)
    if existing is not None:
        if existing.stim != stim or existing.ipsi_sign != ipsi_sign:
            raise ValueError(
                f"epoch '{name}' redeclared with different stim/ipsi_sign: "
                f"{existing} vs Epoch({name!r}, {stim}, {ipsi_sign})"
            )
        return existing

    new_epoch = Epoch(name=name, stim=stim, ipsi_sign=ipsi_sign)
    _REGISTRY[name] = new_epoch
    return new_epoch


## PROTOCOLS ---------------------------------------------------

# Full protocol
protocol: List[Epoch] = [
    epoch("adaptation", Stim.BRIGHT),
    epoch("ramp 0", Stim.RAMP),
]
protocol += 5 * [
    epoch("prey capture right", Stim.PREY_CAPTURE, BoutSign.RIGHT),
    epoch("prey capture break after right", Stim.DARK, BoutSign.RIGHT),
    epoch("prey capture left", Stim.PREY_CAPTURE, BoutSign.LEFT),
    epoch("prey capture break after left", Stim.DARK, BoutSign.LEFT),
]
protocol += [epoch("ramp 1", Stim.RAMP)]
protocol += 10 * [
    epoch("phototaxis bright right", Stim.PHOTOTAXIS, BoutSign.RIGHT),
    epoch("phototaxis break after bright right", Stim.BRIGHT, BoutSign.RIGHT),
    epoch("phototaxis bright left", Stim.PHOTOTAXIS, BoutSign.LEFT),
    epoch("phototaxis break after bright left", Stim.BRIGHT, BoutSign.LEFT),
]
protocol += [epoch("ramp 2", Stim.RAMP)]
protocol += 10 * [epoch("spontaneous dark", Stim.DARK)]
protocol += [epoch("ramp 3", Stim.RAMP)]
protocol += 5 * [
    epoch("flash dark", Stim.DARK),
    epoch("flash ramp", Stim.RAMP),
    epoch("flash bright", Stim.BRIGHT),
]
protocol += [epoch("ramp 4", Stim.RAMP)]
protocol += 5 * [
    epoch("grating right", Stim.OMR, BoutSign.RIGHT),
    epoch("grating break after right", Stim.BRIGHT, BoutSign.RIGHT),
    epoch("grating left", Stim.OMR, BoutSign.LEFT),
    epoch("grating break after left", Stim.BRIGHT, BoutSign.LEFT),
    epoch("grating forward", Stim.OMR),
    epoch("grating break after forward", Stim.BRIGHT),
]
protocol += [epoch("ramp 5", Stim.RAMP)]
protocol += 10 * [epoch("spontaneous bright", Stim.BRIGHT)]
protocol += [epoch("ramp 6", Stim.RAMP)]
protocol += 5 * [
    epoch("pinwheel clockwise", Stim.OKR, BoutSign.RIGHT),
    epoch("pinwheel break after clockwise", Stim.BRIGHT, BoutSign.RIGHT),
    epoch("pinwheel counter-clockwise", Stim.OKR, BoutSign.LEFT),
    epoch("pinwheel break after counter-clockwise", Stim.BRIGHT, BoutSign.LEFT),
]
protocol += [epoch("ramp 7", Stim.RAMP)]
protocol += 7 * [
    epoch("looming left", Stim.LOOMING, BoutSign.LEFT),
    epoch("looming break after left", Stim.BRIGHT, BoutSign.LEFT),
    epoch("looming right", Stim.LOOMING, BoutSign.RIGHT),
    epoch("looming break after right", Stim.BRIGHT, BoutSign.RIGHT),
]

# phototaxis only
protocol_ptx: List[Epoch] = [epoch("adaptation", Stim.BRIGHT)]
protocol_ptx += 10 * [
    epoch("phototaxis bright right", Stim.PHOTOTAXIS, BoutSign.RIGHT),
    epoch("phototaxis break after bright right", Stim.BRIGHT, BoutSign.RIGHT),
    epoch("phototaxis bright left", Stim.PHOTOTAXIS, BoutSign.LEFT),
    epoch("phototaxis break after bright left", Stim.BRIGHT, BoutSign.LEFT),
]


### LATERALITY -------------------------------------------------------

def _build_laterality_table(
        registry: Dict[EpochName, Epoch]
    ) -> Dict[Tuple[EpochName, BoutSign], Laterality]:
    """
    Derived directly from each epoch's declared ipsi_sign -- replaces the
    previously hand-written laterality dict AND the separate
    non_directional-name list, both of which required retyping every
    epoch name a second/third time with nothing checking they matched
    `protocol`.
    """
    table: Dict[Tuple[EpochName, BoutSign], Laterality] = {}
    for name, ep in registry.items():
        for sign in BoutSign:
            if ep.ipsi_sign is None:
                table[(name, sign)] = Laterality.NONDIRECTIONAL
            elif sign == ep.ipsi_sign:
                table[(name, sign)] = Laterality.IPSILATERAL
            else:
                table[(name, sign)] = Laterality.CONTRALATERAL
    return table


EPOCH_LATERALITY: Dict[Tuple[EpochName, BoutSign], Laterality] = _build_laterality_table(_REGISTRY)


### PRESENCE SPEC -----------------------------------------------------

@dataclass(frozen=True)
class EpochSpec:
    name: EpochName
    stim: Stim
    expected_repeats: int
    parameters: Optional[Dict[str, List[Any]]] = None


def _build_epoch_specs(protocol_entries: List[Epoch]) -> List[EpochSpec]:
    counts = {}
    for ep in protocol_entries:
        counts[ep.name] = counts.get(ep.name, 0) + 1

    seen = set()
    specs = []
    for ep in protocol_entries:
        if ep.name in seen:
            continue
        seen.add(ep.name)
        specs.append(EpochSpec(name=ep.name, stim=ep.stim, expected_repeats=counts[ep.name]))
    return specs


PROTOCOL_SPEC: List[EpochSpec] = _build_epoch_specs(protocol)
PROTOCOL_PTX_SPEC: List[EpochSpec] = _build_epoch_specs(protocol_ptx)