"""
Dependence between the gains of fish recorded together, for nulls simulated from the model.

The models give every fish its own gain, independent of the others. Fish of one plate (and of one
day, and of one line) are more alike than that (plate_variance), and a plate holds one arm: two arms
then differ more than independent fish would, without any effect of the treatment.

GainStructure holds the shares of a fish's latent gain common to its line, its day and its plate. The
latent values are standard normal; PointProcess.simulate_dataset maps them to gains by quantile, so
the gains keep the distribution of the fitted model whatever its family. calibrate sets the shares so
that control data simulated from the control fit show the dependence of the control fish: the
intraclass correlations of the fish's log rates between lines, between the days of a line and between
the plates of a day (dependence). Simulated correlations short of the observed ones mean the model's
gains cannot carry that much dependence: its nulls are then still too narrow.
"""
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from BehaviorScreen.point_process.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.point_process.point_process import PointProcess

from .dataset_utils import RECORDING
from .plate_variance import block_sums, icc

NESTING = {"line": "all", "day": "line", "plate": "day"}  # level: the level its groups are compared within
LEVELS = tuple(NESTING)
MAX_SHARED = 0.9  # largest fraction a level takes of the latent variance left to it


@dataclass(frozen=True)
class GainStructure:
    """Shares of a fish's latent gain common to its line, its day and its plate."""

    line: float = 0.0
    day: float = 0.0
    plate: float = 0.0

    def latent(self, groups: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """A standard normal value per fish of groups (fish_groups), with these shares in common."""
        z = np.sqrt(1.0 - self.line - self.day - self.plate) * rng.standard_normal(len(groups))
        for level in LEVELS:
            codes, labels = pd.factorize(groups[level])
            z += np.sqrt(getattr(self, level)) * rng.standard_normal(len(labels))[codes]
        return z


def fish_groups(fish_ids: np.ndarray, lines: Union[str, Sequence[str]]) -> pd.DataFrame:
    """Per fish: its line, its day (of that line) and its plate, from its file name; a fish without one is its own plate."""
    ids = pd.Series(np.asarray(fish_ids, dtype=str))
    names, line = ids.str.extract(RECORDING), pd.Series(np.broadcast_to(lines, len(ids)).astype(str))
    return pd.DataFrame({"line": line, "day": line + "/" + names["day"].fillna(""), "plate": names["plate"].fillna(ids)})


def log_rates(dataset: PointProcessDataset, first_only: bool = False) -> np.ndarray:
    """Per fish, log events per trial (first events only: escapes); NaN without trials."""
    events = dataset.first_events() if first_only else dataset
    trials = dataset.fish_trial_mask.sum(axis=1)
    counts = np.bincount(events.event_fish_idx, minlength=dataset.num_fish)
    return np.where(trials > 0, np.log((counts + 0.5) / np.maximum(trials, 1)), np.nan)


def dependence(values: np.ndarray, groups: pd.DataFrame) -> Dict[str, float]:
    """Per level, the intraclass correlation of the fish's values between its groups, within the level above (NaN: not estimable)."""
    df = groups.assign(value=values, all="").dropna(subset=["value"])
    result = {}
    for level, within in NESTING.items():
        d = pd.DataFrame({"block": df[within], "plate": df[level], "value": df["value"]})
        sums = block_sums(d[d.groupby("block")["plate"].transform("nunique") > 1])
        estimable = len(sums) and sums["plates"].sum() > len(sums) and sums["fish"].sum() > sums["plates"].sum()
        result[level] = icc(sums)[0] if estimable else np.nan
    return result


def _shares(fractions: Dict[str, float]) -> GainStructure:
    """The structure whose line, day and plate take these fractions of what is left below the level above."""
    line = fractions["line"]
    day = fractions["day"] * (1.0 - line)
    return GainStructure(line, day, fractions["plate"] * (1.0 - line - day))


def calibrate(
    model: PointProcess, dataset: PointProcessDataset, groups: pd.DataFrame, n_rep: int = 10, seed: int = 0, guess: float = 0.3,
) -> Tuple[GainStructure, pd.DataFrame]:
    """
    The structure with which datasets simulated from model (fitted to dataset) show dataset's dependence,
    and per level the observed and simulated correlations and the share.

    Levels are set from the plate up, each as the fraction of the latent variance left below the level
    above it, so that a level leaves the correlations of the ones below it alone. A level's simulated
    correlation is taken as linear in its fraction, from what the levels below give alone (fraction 0)
    to its value at `guess`; one more pass rescales every fraction by its remaining mismatch. Simulated
    correlations are means over n_rep datasets.
    """
    rng = np.random.default_rng(seed)
    values = lambda ds: dependence(log_rates(ds, model.is_survival), groups)
    observed = values(dataset)

    def simulated(fractions: Dict[str, float]) -> Dict[str, float]:
        structure = _shares(fractions)
        sims = [values(model.simulate_dataset(dataset, rng, structure.latent(groups, rng))) for _ in range(n_rep)]
        return {level: float(np.mean([sim[level] for sim in sims])) for level in LEVELS}

    def solved(level: str, at: float, value: float, floor: float) -> float:
        """The fraction giving the observed correlation, on the line through (0, floor) and (at, value)."""
        if not value - floor > 1e-3 or not observed[level] > floor:  # also NaN: a level that is not estimable
            return 0.0
        return float(np.clip(at * (observed[level] - floor) / (value - floor), 0.0, MAX_SHARED))

    fractions, floors = dict.fromkeys(LEVELS, 0.0), dict.fromkeys(LEVELS, 0.0)
    for level in reversed(LEVELS):
        if level != "plate":
            floors[level] = simulated(fractions)[level]
        fractions[level] = solved(level, guess, simulated(fractions | {level: guess})[level], floors[level])
    again = simulated(fractions)
    fractions = {level: solved(level, fractions[level], again[level], floors[level]) for level in LEVELS}
    structure, final = _shares(fractions), simulated(fractions)
    table = pd.DataFrame({"level": LEVELS, "observed": [observed[l] for l in LEVELS], "simulated": [final[l] for l in LEVELS],
                          "share": [structure.line, structure.day, structure.plate]})
    return structure, table
