from BehaviorScreen.core import Laterality
from BehaviorScreen.stim_specs import StimSpec
from typing import Any
import pandas as pd

LATERALITY_CODE_LABELS = {
    int(Laterality.IPSILATERAL): "ipsi",
    int(Laterality.CONTRALATERAL): "contra",
    int(Laterality.NONDIRECTIONAL): "none",
}
LATERALITY_ORDER = {
    "ipsi": 0,
    "contra": 1,
    "none": 2,
}

SIGN_ORDER = {
    "LEFT": 0,
    "RIGHT": 1,
}
SIGN_LABELS = ["LEFT", "RIGHT"]

# average_trial, average_time_bin, filename suffix, title
HEATMAP_VARIANTS = [
    (False, False, "", "trial_x_time bin"),
    (True, False, "_trial_avg", "averaged over trials"),
    (False, True, "_timebin_avg", "averaged over time bins"),
    (
        True,
        True,
        "_full_avg",
        "averaged over trials and time bins",
    ),
]


def order_lateralities(values: Any) -> list[str]:
    """Sort laterality labels in a stable display order."""
    return sorted(
        values,
        key=lambda value: LATERALITY_ORDER.get(value, 99),
    )


def order_signs(values: Any) -> list[str]:
    """Sort left/right labels in a stable display order."""
    return sorted(
        values,
        key=lambda value: SIGN_ORDER.get(value, 99),
    )


def map_laterality(series: pd.Series) -> pd.Series:
    """
    Map raw laterality codes to display labels.

    Both explicit non-directional values and missing laterality values are
    placed in the ``none`` group.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(LATERALITY_CODE_LABELS)

    return mapped.where(mapped.notna(), "none")


def get_laterality_labels(
    events: pd.DataFrame,
    specification: StimSpec,
) -> list[str]:
    if "laterality" not in events.columns:
        return ["none"]

    mask = specification.get_mask(events)

    if "stim" in events.columns:
        mask &= events["stim"] == specification.stim

    labels = map_laterality(events.loc[mask, "laterality"]).dropna().unique().tolist()

    return order_lateralities(labels) if labels else ["none"]
