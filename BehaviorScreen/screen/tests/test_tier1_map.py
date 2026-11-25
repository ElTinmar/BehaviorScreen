"""The Tier 1 map's effect size, a Cohen's d from each difference's standard error, and its post hoc marks."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.screen.line_tests.tier1_omnibus import (
    at_bound, compute_parameter_deltas, control_sds, plot_tier1_map, reference_sds, standardized_effects,
)

matplotlib.use("Agg")


def test_delta_is_over_the_column_sd_of_control_fish_even_with_an_arm_at_a_bound() -> None:
    """Reference arms of 9 and 25 fish whose standard errors imply an SD of 0.5 log2 units; L3's test arm at its bound."""
    sd, ln2 = 0.5, np.log(2)
    params = pd.DataFrame({
        "line": ["L1", "L2", "L3"], "behavior": "b", "parameter": "A", "value_reference": 1.0, "value_test": [2.0, 0.5, 0.01],
        "se_reference": [sd * ln2 / 3, sd * ln2 / 5, sd * ln2 / 3], "se_test": [0.1, 0.1, np.nan],
    })
    tier1 = pd.DataFrame({"line": ["L1", "L2", "L3"], "behavior": "b", "reference": ["L1/v", "L2/v", "L3/v"], "n_fish_ref": [9, 25, 9]})
    effects = standardized_effects(compute_parameter_deltas(params), tier1)["delta"]
    np.testing.assert_allclose(effects, np.log2([2.0, 0.5, 0.01]) / sd)
    assert at_bound(params).tolist() == [False, False, True]


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


def test_an_arm_at_a_bound_is_marked_on_both_maps() -> None:
    params, tier1, posthoc = map_inputs()
    params = params.assign(value_reference=1.0, value_test=[0.01, 1.0], se_reference=0.1, se_test=[np.nan, 0.1])
    _, squares = plot_tier1_map(params, tier1, ["b"], ["L1"], se_half_size=None)
    _, dots = plot_tier1_map(params, tier1, ["b"], ["L1"])
    assert [t.get_text() for t in squares.texts] == ["0"]
    assert [len(c.get_paths()[0].vertices) for c in dots.collections][0] == 5  # H at the bound: a diamond, D a circle


def test_sigma_is_the_median_arm_sd_and_names_the_arms_at_a_bound() -> None:
    sds = np.array([0.4, 0.5, 0.9])
    params = pd.DataFrame({"line": ["L1", "L2", "L3", "L4"], "behavior": "b", "parameter": "D", "value_reference": 0.0, "value_test": 0.1,
                           "se_reference": [*sds / 3, np.nan], "se_test": 0.1})
    tier1 = pd.DataFrame({"line": ["L1", "L2", "L3", "L4"], "behavior": "b", "reference": ["L1/v", "L2/v", "L3/v", "L4/v"], "n_fish_ref": 9})
    table = control_sds(reference_sds(params, tier1)).iloc[0]
    assert (table["sigma"], table["n_arms"], table["at_bound"]) == (pytest.approx(0.5), 3, "L4/v")
