"""Arm surface grids: a sided model's arms are drawn side by side, with their side calibration."""
import matplotlib
import numpy as np

matplotlib.use("Agg")

from BehaviorScreen.point_process.tests.conftest import make_scaffold_dataset
from BehaviorScreen.screen.control_models.model_config import flat, sided_models
from BehaviorScreen.screen.line_tests.tier1_config import PinnedArchitecture
from BehaviorScreen.screen.line_tests.tier1_omnibus import plot_sided_arm_surface_grid


def test_sided_grid_has_both_sides_and_their_calibration_per_arm() -> None:
    rng = np.random.default_rng(0)
    control = sided_models(flat, flat)[2]  # a fish gain with a side preference, no bout history
    control.set_params(np.array(control.initial_guesses, float))
    datasets = [control.simulate_dataset(make_scaffold_dataset(12, 4, 6.0), rng) for _ in range(2)]
    architecture = PinnedArchitecture(control.fit(datasets[0]))
    models = [architecture().fit(ds) for ds in datasets]
    fig = plot_sided_arm_surface_grid(*datasets, *models, title="L1 / okr")
    titles = [ax.get_title(loc="left") or ax.get_title() for ax in fig.axes]
    assert [t for t in titles if ": " in t] == ["Reference: ipsi", "Reference: contra", "Test: ipsi", "Test: contra"]
    assert titles.count("Side calibration by time") == titles.count("Side calibration by trial") == 2
