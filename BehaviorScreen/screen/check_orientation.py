"""
Do the left and right of bouts, saccades and stimuli agree (rig.yaml)? Three behaviors of the control
fish, each tying two of them, from the merged tables:

    turns during a lateral grating         with the grating (ipsilateral)        bouts and stimuli
    conjugate saccades during the pinwheel  against the rotation (contralateral)  saccades and stimuli
    a turn and a conjugate saccade within PAIR_WINDOW_S, without stimulus: to the same side   bouts and saccades

Per fish, the share of events as expected; over fish, their median and a sign test against one half.
Two tests against expectation name the orientation they share; one alone is not an orientation.

    python -m BehaviorScreen.screen.check_orientation SCREEN_FOLDER
"""
import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from megabouts.utils import bouts_category_name_short
from scipy.stats import binomtest

from BehaviorScreen.core import SACCADE_CATEGORY_NAMES, Laterality, Stim
from BehaviorScreen.screen.control_models.model_config import CONTROL_CONDITIONS, MUTANT_REFERENCES

TURNS = [bouts_category_name_short.index(name) for name in ("RT", "HAT")]
CONJUGATE = next(code for code, name in SACCADE_CATEGORY_NAMES.items() if name == "Conjugate")
PAIR_WINDOW_S, MIN_EVENTS, ALPHA = 0.1, 10, 0.01
TESTS = {"turns with the grating": ("bout", "stimulus"), "saccades against the pinwheel": ("saccade", "stimulus"), "turn and saccade to the same side": ("bout", "saccade")}
SWITCHES = {"bout": "bout_sign_left", "saccade": "saccade_classes_mirrored", "stimulus": "stimulus_mirrored"}
BOUT_COLUMNS = ["file", "line", "condition", "stim", "category", "sign", "laterality", "event_timestamp"]
SACCADE_COLUMNS = ["file", "line", "condition", "stim", "saccade_category", "event_direction", "laterality", "event_timestamp"]


def control(events: pd.DataFrame) -> pd.DataFrame:
    return events[events["condition"].isin(CONTROL_CONDITIONS) & ~events["line"].astype(str).isin(MUTANT_REFERENCES)]


def sided(events: pd.DataFrame, stim: Stim) -> pd.DataFrame:
    return events[(events["stim"] == int(stim)) & events["laterality"].isin([int(Laterality.IPSILATERAL), int(Laterality.CONTRALATERAL)])]


def pairs(turns: pd.DataFrame, saccades: pd.DataFrame) -> pd.DataFrame:
    """Each conjugate saccade without stimulus with the nearest turn of its fish within PAIR_WINDOW_S."""
    quiet = lambda events: events[events["stim"].isin([int(Stim.DARK), int(Stim.BRIGHT)])].dropna(subset=["event_timestamp"]).sort_values("event_timestamp")
    return pd.merge_asof(
        quiet(saccades[saccades["event_direction"].isin([-1, 1])]), quiet(turns[turns["sign"].isin([-1, 1])])[["file", "event_timestamp", "sign"]],
        on="event_timestamp", by="file", direction="nearest", tolerance=int(PAIR_WINDOW_S * 1e9),
    ).dropna(subset=["sign"])


def verdict(as_expected: pd.Series, fish: pd.Series) -> Dict[str, object]:
    """as_expected: per event. Fish with MIN_EVENTS events or more count."""
    per_fish = as_expected.groupby(fish.to_numpy()).agg(["mean", "size"])
    shares = per_fish.loc[per_fish["size"] >= MIN_EVENTS, "mean"]
    above, below = int((shares > 0.5).sum()), int((shares < 0.5).sum())
    p = binomtest(above, above + below).pvalue if above + below else 1.0
    status = "inconclusive" if p >= ALPHA else "as expected" if above > below else "inverted"
    return {"events": len(as_expected), "fish": len(shares), "median_share": shares.median() if len(shares) else np.nan, "p_value": p, "status": status}


def check(bouts: pd.DataFrame, saccades: pd.DataFrame) -> pd.DataFrame:
    turns = control(bouts)[lambda d: d["category"].isin(TURNS)]
    conjugate = control(saccades)[lambda d: d["saccade_category"] == CONJUGATE]
    grating, pinwheel, paired = sided(turns, Stim.OMR), sided(conjugate, Stim.OKR), pairs(turns, conjugate)
    events = {
        "turns with the grating": (grating["laterality"] == int(Laterality.IPSILATERAL), grating["file"]),
        "saccades against the pinwheel": (pinwheel["laterality"] == int(Laterality.CONTRALATERAL), pinwheel["file"]),
        "turn and saccade to the same side": (paired["sign"] == paired["event_direction"], paired["file"]),
    }
    return pd.DataFrame([{"test": name, "ties": " and ".join(TESTS[name]), **verdict(*events[name])} for name in TESTS])


def diagnosis(table: pd.DataFrame) -> str:
    status = table.set_index("test")["status"]
    inverted = [name for name in TESTS if status[name] == "inverted"]
    unclear = [name for name in TESTS if status[name] == "inconclusive"]
    if len(inverted) == 2:
        (shared,) = set(TESTS[inverted[0]]) & set(TESTS[inverted[1]])
        return f"INVERTED: {shared}s disagree with the two others: change {SWITCHES[shared]} in rig.yaml, then run preprocessing again from the bouts and saccades steps"
    if inverted:
        return f"INCONSISTENT: '{inverted[0]}' alone is against expectation; an orientation would show in two tests. Look at the epochs and classes it uses" + (
            f" ({', '.join(unclear)}: inconclusive)" if unclear else "")
    if unclear:
        return f"no disagreement found, but inconclusive: {', '.join(unclear)}"
    return "bouts, saccades and stimuli agree"


def main(root: Path) -> None:
    table = check(pd.read_csv(root / "bouts_all.csv", usecols=BOUT_COLUMNS), pd.read_csv(root / "saccades_augmented_all.csv", usecols=SACCADE_COLUMNS))
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    message = diagnosis(table)
    if message.startswith("IN"):
        sys.exit(message)
    print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agreement of the left and right of bouts, saccades and stimuli in the control fish.")
    parser.add_argument("root", type=Path, help="Screen folder holding bouts_all.csv and saccades_augmented_all.csv")
    main(parser.parse_args().root)
