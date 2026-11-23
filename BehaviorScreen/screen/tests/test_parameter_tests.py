"""Parameter tests of a hit: of its free amplitudes, the one that differs between the arms is found."""
import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.tests.conftest import make_scaffold_dataset
from BehaviorScreen.screen.control_models.model_config import flat, sided
from BehaviorScreen.screen.line_tests.fdr import holm
from BehaviorScreen.screen.line_tests import parameter_tests as module
from BehaviorScreen.screen.line_tests.parameter_tests import fit_arms, parameter_tests, run_parameter_tests
from BehaviorScreen.screen.line_tests.tier1_config import Fresh


def arm(rate: float, rng: np.random.Generator) -> PointProcessDataset:
    """12 fish with events to either side at `rate` on the ipsi side, twice as many as on the other (D = 2)."""
    model = sided(flat(), flat())
    model.set_params(np.array([rate, 2.0]))
    return model.simulate_dataset(make_scaffold_dataset(12, 5, 9.0), rng)


def test_holm_scales_the_smallest_p_most_and_keeps_their_order() -> None:
    np.testing.assert_allclose(holm(np.array([0.01, 0.04, np.nan, 0.03])), [0.03, 0.06, np.nan, 0.06])


def test_joint_fit_without_a_common_parameter_is_the_two_arms_fits() -> None:
    rng = np.random.default_rng(0)
    datasets, factory = [arm(0.6, rng), arm(0.3, rng)], Fresh(sided(flat(), flat()))
    separate = sum(factory().fit(ds).log_likelihood for ds in datasets)
    assert fit_arms(factory, datasets).log_likelihood == pytest.approx(separate, abs=1e-5)
    assert fit_arms(factory, datasets, "B").log_likelihood < separate - 5


def test_the_parameter_that_differs_is_the_one_found() -> None:
    """Arms with the same directionality D and half the rate B in one."""
    rng = np.random.default_rng(0)
    table = parameter_tests(Fresh(sided(flat(), flat())), [arm(0.6, rng), arm(0.3, rng)], n_boot=60, n_jobs=1).set_index("parameter")
    assert table["significant"].to_dict() == {"B": True, "D": False}


def test_a_cell_whose_fits_fail_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    class Arms:
        process_factory = staticmethod(lambda line, behavior: Fresh(sided(flat(), flat())))
        datasets = staticmethod(lambda line, behavior: (None, None))
    monkeypatch.setattr(module, "parameter_tests", lambda *args, **kwargs: 1 / 0)
    hits = pd.DataFrame({"family": ["ntr"], "significant": [True], "line": ["L1"], "behavior": ["okr"]})
    table = run_parameter_tests(hits, Arms(), n_boot=10)
    assert table["status"].tolist() == ["fit_failed: ZeroDivisionError: division by zero"] and list(table.columns) == module.COLUMNS
