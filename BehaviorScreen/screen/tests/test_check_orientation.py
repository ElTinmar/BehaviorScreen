"""The orientation check on simulated control fish whose recorded left and right are flipped or not."""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.core import Laterality, Stim
from BehaviorScreen.screen.check_orientation import CONJUGATE, TURNS, check, diagnosis, main

IPSI, CONTRA = int(Laterality.IPSILATERAL), int(Laterality.CONTRALATERAL)


def screen(bout: int = 1, saccade: int = 1, stimulus: int = 1, num_fish: int = 12, line: str = "WT", condition: str = "danieau",
           seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fish that turn with the grating, reset their eyes against the pinwheel and pair a saccade with a turn to the same side,
    each 80% of the time. bout, saccade, stimulus: -1 if the recorded side of that kind is the opposite of the true one.
    """
    rng, bouts, saccades = np.random.default_rng(seed), [], []
    often = lambda side: np.where(rng.random(len(side)) < 0.8, side, -side)
    laterality = lambda recorded, side: np.where(recorded == stimulus * side, IPSI, CONTRA)
    for f in range(num_fish):
        file, side, time = f"{line}_{condition}_{f}", rng.choice([-1, 1], 40), np.arange(40) * 10**9
        turn, reset, spontaneous = bout * often(side), saccade * often(-side), rng.choice([-1, 1], 40)
        common = {"file": file, "line": line, "condition": condition}
        bouts += [pd.DataFrame({**common, "stim": int(Stim.OMR), "category": TURNS[0], "sign": turn, "laterality": laterality(turn, side), "event_timestamp": time}),
                  pd.DataFrame({**common, "stim": int(Stim.DARK), "category": TURNS[1], "sign": bout * spontaneous, "laterality": 0, "event_timestamp": time + 50 * 10**9})]
        saccades += [pd.DataFrame({**common, "stim": int(Stim.OKR), "saccade_category": CONJUGATE, "event_direction": reset, "laterality": laterality(reset, side), "event_timestamp": time}),
                     pd.DataFrame({**common, "stim": int(Stim.DARK), "saccade_category": CONJUGATE, "event_direction": saccade * often(spontaneous), "laterality": 0,
                                   "event_timestamp": time + 50 * 10**9 + 30 * 10**6})]
    return pd.concat(bouts, ignore_index=True), pd.concat(saccades, ignore_index=True)


def test_orientations_that_agree() -> None:
    table = check(*screen())
    assert list(table["status"]) == ["as expected"] * 3 and (table["fish"] == 12).all()
    assert table["median_share"].to_numpy() == pytest.approx(0.8, abs=0.08)
    assert diagnosis(table) == "bouts, saccades and stimuli agree"


def test_all_three_flipped_still_agree() -> None:
    assert diagnosis(check(*screen(bout=-1, saccade=-1, stimulus=-1))) == "bouts, saccades and stimuli agree"


@pytest.mark.parametrize("flipped, switch", [
    ({"bout": -1}, "bout_sign_left"), ({"saccade": -1}, "saccade_classes_mirrored"), ({"stimulus": -1}, "stimulus_mirrored"),
    ({"bout": -1, "saccade": -1}, "stimulus_mirrored"), ({"bout": -1, "stimulus": -1}, "saccade_classes_mirrored"),
])
def test_one_orientation_against_the_others_is_named(flipped: dict, switch: str) -> None:
    table = check(*screen(**flipped))
    assert sorted(table["status"]) == ["as expected", "inverted", "inverted"]
    assert diagnosis(table).startswith("INVERTED") and f"change {switch} in rig.yaml" in diagnosis(table)


def test_one_test_alone_against_expectation_is_not_an_orientation() -> None:
    bouts, saccades = screen()
    grating = bouts["stim"] == int(Stim.OMR)
    bouts.loc[grating, "laterality"] = -bouts.loc[grating, "laterality"]  # the grating epochs declared on the wrong side
    table = check(bouts, saccades)
    assert list(table["status"]) == ["inverted", "as expected", "as expected"]
    assert diagnosis(table).startswith("INCONSISTENT: 'turns with the grating'")


def test_only_control_fish_count() -> None:
    """Treated fish and mutants with every side inverted do not change the result."""
    flipped = dict(bout=-1, num_fish=30)
    tables = [screen(), screen(**flipped, line="L1", condition="ronidazole"), screen(**flipped, line="lak", condition="danieau")]
    table = check(*(pd.concat(kind, ignore_index=True) for kind in zip(*tables)))
    assert (table["fish"] == 12).all() and diagnosis(table) == "bouts, saccades and stimuli agree"


def test_few_fish_or_events_are_inconclusive() -> None:
    table = check(*screen(num_fish=3))
    assert set(table["status"]) == {"inconclusive"} and diagnosis(table).startswith("no disagreement found, but inconclusive")
    bouts, saccades = screen()
    assert (check(bouts.groupby("file").head(5), saccades.groupby("file").head(5))["fish"] == 0).all()


def test_command_reads_the_merged_tables_and_fails_on_a_disagreement(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    def write(**flipped: int) -> None:
        bouts, saccades = screen(**flipped)
        bouts.assign(distance=1.0).to_csv(tmp_path / "bouts_all.csv", index=False)
        saccades.to_csv(tmp_path / "saccades_augmented_all.csv", index=False)

    write()
    main(tmp_path)
    assert "bouts, saccades and stimuli agree" in capsys.readouterr().out
    write(saccade=-1)
    with pytest.raises(SystemExit, match="change saccade_classes_mirrored"):
        main(tmp_path)
