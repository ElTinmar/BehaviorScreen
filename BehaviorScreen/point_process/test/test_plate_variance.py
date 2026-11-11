"""Plate variance: the intraclass correlation of simulated plates is recovered, and its test is calibrated without one."""
import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.ablation_screen.plate_variance import PLATE, block_sums, icc, permutation_p, plate_variance


def simulate(rho: float, seed: int, blocks: int = 40, plates: int = 3) -> pd.DataFrame:
    """Blocks (one line and day each, with its own mean) of plates of 3 or 4 fish; plates explain rho of the variance."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(blocks):
        for p in range(plates):
            plate_effect = rng.normal(0, np.sqrt(rho))
            for f in range(rng.choice([3, 4])):
                rows.append({"readout": "rate", "line": f"L{b}", "condition": "vehicle", "day": "01_Jan_2026", "plate": f"L{b}_{p}",
                             "file": f"L{b}_{p}_fish_{f}", "value": b + plate_effect + rng.normal(0, np.sqrt(1 - rho))})
    return pd.DataFrame(rows)


def test_file_name_gives_plate_and_day() -> None:
    names = pd.Series(["02_07dpf_GR_Wed_18_Mar_2026_15h30min46sec_fish_1"]).str.extract(PLATE)
    assert names.loc[0].to_dict() == {"plate": "02_07dpf_GR_Wed_18_Mar_2026_15h30min46sec", "day": "18_Mar_2026"}


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_icc_is_recovered(rho: float) -> None:
    estimates = [icc(block_sums(simulate(rho, seed).assign(block=lambda d: d["line"])))[0] for seed in range(30)]
    assert np.mean(estimates) == pytest.approx(rho, abs=0.03)


def test_permutation_is_calibrated_without_plate_effect_and_detects_one() -> None:
    p = lambda rho, seed: permutation_p(simulate(rho, seed).assign(block=lambda d: d["line"]), 200, np.random.default_rng(seed))
    assert np.mean([p(0.0, seed) < 0.05 for seed in range(100)]) < 0.11
    assert p(0.3, 0) < 0.01


def test_table_reports_inflation() -> None:
    row = plate_variance(simulate(0.3, 1), n_perm=200, n_boot=200).iloc[0]
    assert (row["arms"], row["blocks"], row["plates"]) == ("control", 40, 120)
    assert row["icc_lo"] < row["icc"] < row["icc_hi"] and row["false_positive_rate"] > 0.08
