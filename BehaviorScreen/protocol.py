from BehaviorScreen.core import EventDirection, Laterality, Stim, SACCADE_CLASS_DIRECTIONS
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import numpy as np

EpochName = str


@dataclass(frozen=True)
class Epoch:
    """
    One declared epoch: its Stim, and -- for directional stimuli -- which
    EventDirection counts as IPSILATERAL while it's showing (e.g. EventDirection.RIGHT
    for "...right"/"...clockwise" epochs). Leave ipsi_sign=None for
    non-directional epochs (both bout signs then map to NONDIRECTIONAL).

    `parameters`, when given, constrains which stimulus-log rows count as
    a genuine presentation of this epoch, beyond just matching epoch_name
    (e.g. phototaxis's foreground_color). Only appropriate when the
    constraint is a fixed property of THIS epoch's identity -- i.e. true
    for every repetition of this name across the whole protocol, not
    something that varies presentation-to-presentation.
    """
    name: EpochName
    stim: Stim
    ipsi_sign: Optional[EventDirection] = None
    parameters: Optional[Dict[str, List[Any]]] = None


_REGISTRY: Dict[EpochName, Epoch] = {}


def epoch(
        name: EpochName,
        stim: Stim,
        ipsi_sign: Optional[EventDirection] = None,
        parameters: Optional[Dict[str, List[Any]]] = None,
    ) -> Epoch:
    """
    Declare (or re-fetch) one epoch, for use inside `protocol`/`protocol_ptx`.

    This is the ONLY place a given epoch_name's stim/laterality
    side/parameter constraints are declared -- everything derived below
    (EPOCH_LATERALITY, PROTOCOL_SPEC) is built from this registry, instead
    of being hand-written a second/third time with nothing checking it
    stayed in sync with `protocol`.

    Re-declaring the SAME name (protocol_ptx reuses several names already
    declared while building `protocol`) is fine as long as
    stim/ipsi_sign/parameters all agree; if they don't, that's a genuine
    inconsistency and raises immediately here, instead of silently
    producing a wrong/missing laterality or presence entry discovered only
    later, deep in megabouts.py or plot.py.
    """
    existing = _REGISTRY.get(name)
    if existing is not None:
        if (existing.stim != stim
                or existing.ipsi_sign != ipsi_sign
                or existing.parameters != parameters):
            raise ValueError(
                f"epoch '{name}' redeclared with different stim/ipsi_sign/parameters: "
                f"{existing} vs Epoch({name!r}, {stim}, {ipsi_sign}, {parameters})"
            )
        return existing

    new_epoch = Epoch(name=name, stim=stim, ipsi_sign=ipsi_sign, parameters=parameters)
    _REGISTRY[name] = new_epoch
    return new_epoch


## PROTOCOLS ---------------------------------------------------


_PHOTOTAXIS_PARAMS = {"foreground_color": ["[0.1, 0.1, 0.0, 1.0]"]}

# Full protocol
protocol: List[Epoch] = [
    epoch("adaptation", Stim.BRIGHT),
    epoch("ramp 0", Stim.RAMP),
]
protocol += 5 * [
    epoch("prey capture right", Stim.PREY_CAPTURE, EventDirection.RIGHT),
    epoch("prey capture break after right", Stim.DARK, EventDirection.RIGHT),
    epoch("prey capture left", Stim.PREY_CAPTURE, EventDirection.LEFT),
    epoch("prey capture break after left", Stim.DARK, EventDirection.LEFT),
]
protocol += [epoch("ramp 1", Stim.RAMP)]
protocol += 10 * [
    epoch("phototaxis bright right", Stim.PHOTOTAXIS, EventDirection.RIGHT, _PHOTOTAXIS_PARAMS),
    epoch("phototaxis break after bright right", Stim.BRIGHT, EventDirection.RIGHT),
    epoch("phototaxis bright left", Stim.PHOTOTAXIS, EventDirection.LEFT, _PHOTOTAXIS_PARAMS),
    epoch("phototaxis break after bright left", Stim.BRIGHT, EventDirection.LEFT),
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
    epoch("grating right", Stim.OMR, EventDirection.RIGHT),
    epoch("grating break after right", Stim.BRIGHT, EventDirection.RIGHT),
    epoch("grating left", Stim.OMR, EventDirection.LEFT),
    epoch("grating break after left", Stim.BRIGHT, EventDirection.LEFT),
    epoch("grating forward", Stim.OMR),
    epoch("grating break after forward", Stim.BRIGHT),
]
protocol += [epoch("ramp 5", Stim.RAMP)]
protocol += 10 * [epoch("spontaneous bright", Stim.BRIGHT)]
protocol += [epoch("ramp 6", Stim.RAMP)]
protocol += 5 * [
    epoch("pinwheel clockwise", Stim.OKR, EventDirection.RIGHT),
    epoch("pinwheel break after clockwise", Stim.BRIGHT, EventDirection.RIGHT),
    epoch("pinwheel counter-clockwise", Stim.OKR, EventDirection.LEFT),
    epoch("pinwheel break after counter-clockwise", Stim.BRIGHT, EventDirection.LEFT),
]
protocol += [epoch("ramp 7", Stim.RAMP)]
protocol += 7 * [
    epoch("looming left", Stim.LOOMING, EventDirection.LEFT),
    epoch("looming break after left", Stim.BRIGHT, EventDirection.LEFT),
    epoch("looming right", Stim.LOOMING, EventDirection.RIGHT),
    epoch("looming break after right", Stim.BRIGHT, EventDirection.RIGHT),
]

# phototaxis only
protocol_ptx: List[Epoch] = [epoch("adaptation", Stim.BRIGHT)]
protocol_ptx += 10 * [
    epoch("phototaxis bright right", Stim.PHOTOTAXIS, EventDirection.RIGHT, _PHOTOTAXIS_PARAMS),
    epoch("phototaxis break after bright right", Stim.BRIGHT, EventDirection.RIGHT),
    epoch("phototaxis bright left", Stim.PHOTOTAXIS, EventDirection.LEFT, _PHOTOTAXIS_PARAMS),
    epoch("phototaxis break after bright left", Stim.BRIGHT, EventDirection.LEFT),
]


### LATERALITY -------------------------------------------------------

def _build_laterality_table(
        registry: Dict[EpochName, Epoch]
    ) -> Dict[Tuple[EpochName, EventDirection], Laterality]:
    """
    Derived directly from each epoch's declared ipsi_sign -- replaces a
    hand-written (epoch_name, EventDirection) -> Laterality dict and a separate
    non-directional-name list, both of which previously required retyping
    every epoch name a second/third time with nothing checking it matched
    `protocol`.
    """
    table: Dict[Tuple[EpochName, EventDirection], Laterality] = {}
    for name, ep in registry.items():
        for sign in EventDirection:
            if ep.ipsi_sign is None:
                table[(name, sign)] = Laterality.NONDIRECTIONAL
            elif sign == ep.ipsi_sign:
                table[(name, sign)] = Laterality.IPSILATERAL
            else:
                table[(name, sign)] = Laterality.CONTRALATERAL
    return table


EPOCH_LATERALITY: Dict[Tuple[EpochName, EventDirection], Laterality] = _build_laterality_table(_REGISTRY)


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
        specs.append(EpochSpec(
            name=ep.name,
            stim=ep.stim,
            expected_repeats=counts[ep.name],
            parameters=ep.parameters,
        ))
    return specs


PROTOCOL_SPEC: List[EpochSpec] = _build_epoch_specs(protocol)


def bout_laterality(
    epoch_name: object,
    sign: object,
) -> float:
    """Return bout laterality from epoch identity and bout direction."""
    try:
        direction = EventDirection(int(sign))
    except (TypeError, ValueError):
        return np.nan

    return EPOCH_LATERALITY.get(
        (str(epoch_name), direction),
        np.nan,
    )

def saccade_laterality(
    epoch_name: object,
    cluster: object,
) -> float:
    """
    Return laterality for a classified saccade.

    Directionless saccade classes are NONDIRECTIONAL. Events without a
    valid epoch/class, or a directional event in an undeclared epoch,
    receive NaN.
    """
    if epoch_name is None:
        return np.nan

    try:
        cluster = int(cluster)
    except (TypeError, ValueError):
        return np.nan

    direction = SACCADE_CLASS_DIRECTIONS.get(cluster)

    # Classes such as convergent/divergent do not have a left/right
    # direction and therefore stay in their own "none" group.
    if direction is None:
        return int(Laterality.NONDIRECTIONAL)

    return EPOCH_LATERALITY.get(
        (str(epoch_name), direction),
        np.nan,
    )