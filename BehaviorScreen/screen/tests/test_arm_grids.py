"""Arm surface grids: a sided model's arms are drawn side by side, with their fish and side calibration."""
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
    datasets = [control.simulate_dataset(make_scaffold_dataset(n, 4, 6.0), rng) for n in (12, 9)]
    architecture = PinnedArchitecture(control.fit(datasets[0]))
    models = [architecture().fit(ds) for ds in datasets]
    fig = plot_sided_arm_surface_grid(*datasets, *models, title="L1 / okr")
    titles = [ax.get_title(loc="left") or ax.get_title() for ax in fig.axes]
    assert [t for t in titles if ": " in t] == ["Reference (12 fish): ipsi", "Reference (12 fish): contra", "Test (9 fish): ipsi", "Test (9 fish): contra"]
    assert titles.count("Side calibration by time") == titles.count("Side calibration by trial") == 2


def test_the_predictive_range_holds_data_simulated_from_the_model() -> None:
    from BehaviorScreen.screen.line_tests.tier1_omnibus import count_shares, fish_rates, predictive
    model = sided_models(flat, flat)[2]
    model.set_params(np.array(model.initial_guesses, float))
    data = model.simulate_dataset(make_scaffold_dataset(30, 6, 6.0), np.random.default_rng(1))
    model.fit_result = object()
    pred = predictive(model, data, sides=[None])
    inside = lambda value, band: np.mean((band[0] <= value) & (value <= band[1]))
    assert inside(np.sort(fish_rates(data)), pred["fish"]) > 0.8 and inside(count_shares(data, pred["k"]), pred["counts"]) > 0.8
    assert inside(np.mean(data.time_trial_histogram_hz, axis=1), pred[None]["trial"]) > 0.6
