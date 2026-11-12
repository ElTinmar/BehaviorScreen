"""Exclusions per arm, from small merged tables."""
from pathlib import Path

import pandas as pd
import pytest

from BehaviorScreen.screen.line_tests.exclusions import counts, with_shares


def test_counts_and_shares_per_arm(tmp_path: Path) -> None:
    arm = {"a": ("L1", "vehicle"), "b": ("L1", "vehicle"), "c": ("L1", "ronidazole"), "d": ("L1", "ronidazole")}
    rows = lambda files, **columns: pd.DataFrame({"file": files, "line": [arm[f][0] for f in files], "condition": [arm[f][1] for f in files], **columns})
    rows(list("aabbccdd"), presented=[True] * 7 + [False], tracking_ok=[True, True, True, False, True, True, False, False]).to_csv(tmp_path / "valid_trials_all.csv", index=False)
    rows(["d"], not_moving=[True], centroid_issue=[False], heading_issue=[False], well_not_found=[False], frame_order_issue=[False]).to_csv(tmp_path / "qc_all.csv", index=False)
    rows(list("aaabccd"), epoch_name=["prey", "prey", None, "prey", "prey", "prey", "prey"], distance_center=[1, 12, 30, 3, 11, 2, 20],
         centroid_mismatch_max=[0, 0, 0, 5000, 0, 0, 0], distance=[5, 5, 5, 5, 5, 500, 5], peak_axial_speed=10, heading_mismatch_max=20,
         within_trial=[False, True, True, True, True, True, True]).to_csv(tmp_path / "bouts_all.csv", index=False)

    table = with_shares(counts(tmp_path))
    vehicle, drug = table.loc[("L1", "vehicle")], table.loc[("L1", "ronidazole")]
    assert (vehicle["fish_recorded"], vehicle["fish_excluded"], drug["fish_excluded"], drug["fish_not_moving"]) == (2, 0, 1, 1)
    assert (vehicle["trials_presented"], vehicle["trials_untracked"], drug["trials_presented"], drug["trials_untracked"]) == (4, 1, 2, 0)
    assert (vehicle["bouts"], vehicle["bouts_dropped_distance_center"], vehicle["bouts_dropped_centroid_mismatch_max"]) == (3, 1, 1)
    assert (drug["bouts"], drug["bouts_dropped_distance_center"]) == (2, 1)
    assert (vehicle["bouts_dropped_within_trial"], drug["bouts_dropped_distance"], vehicle["bouts_dropped_peak_axial_speed"]) == (1, 1, 0)
    assert drug["share_fish_excluded"] == pytest.approx(0.5) and vehicle["share_trials_untracked"] == pytest.approx(0.25)
