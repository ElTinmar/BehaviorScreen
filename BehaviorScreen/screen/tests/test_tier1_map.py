"""The Tier 1 map's effect size, a Cohen's d from each difference's standard error, and its post hoc marks."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.screen.line_tests.tier1_omnibus import plot_tier1_map, standardized_effects

matplotlib.use("Agg")


@pytest.mark.parametrize("n_ref, n_test", [(8, 8), (10, 40), (60, 60)])
def test_d_is_the_difference_over_the_sd_between_fish_whatever_their_number(n_ref: int, n_test: int) -> None:
    sd, delta = 0.8, 0.4
    params = pd.DataFrame({"line": ["L1"], "behavior": ["b"], "parameter": ["B"], "delta": [delta],
                           "delta_se": [sd * np.sqrt(1 / n_ref + 1 / n_test)]})
    tier1 = pd.DataFrame({"line": ["L1"], "behavior": ["b"], "n_fish_ref": [n_ref], "n_fish_test": [n_test]})
    assert standardized_effects(params, tier1)["delta"].item() == pytest.approx(delta / sd)


def map_inputs() -> tuple:
    params = pd.DataFrame({"line": ["L1", "L1"], "behavior": ["b", "b"], "parameter": ["H", "D"], "delta": [0.9, np.nan], "delta_se": [0.2, np.nan]})
    tier1 = pd.DataFrame({"line": ["L1"], "behavior": ["b"], "family": ["ntr"], "status": ["ok"], "significant": [True], "testable": [True]})
    posthoc = pd.DataFrame({"line": ["L1", "L1"], "behavior": ["b", "b"], "parameter": ["H", "D"], "significant": [True, False]})
    return params, tier1, posthoc


def test_the_square_map_marks_a_post_hoc_parameter_with_a_centre_dot() -> None:
    params, tier1, posthoc = map_inputs()
    _, ax = plot_tier1_map(params, tier1, ["b"], ["L1"], se_half_size=None, posthoc=posthoc)
    assert len(ax.patches) == 3  # two cells and the frame
    assert [line.get_xdata()[0] for line in ax.lines if line.get_marker() == "o"] == [0]


def test_the_dot_map_marks_a_post_hoc_parameter_with_a_background() -> None:
    params, tier1, posthoc = map_inputs()
    _, ax = plot_tier1_map(params, tier1, ["b"], ["L1"], posthoc=posthoc)
    backgrounds = [patch.get_x() + 0.4 for patch in ax.patches if patch.get_fill()]
    assert backgrounds == [0] and len(ax.collections) == 2  # one dot per cell
