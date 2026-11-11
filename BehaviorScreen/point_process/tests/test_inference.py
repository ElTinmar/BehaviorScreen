"""Model selection: the multistart fit keeps its best start, and AIC counts every parameter twice."""
from types import SimpleNamespace

import numpy as np
import pytest

from .conftest import make_scaffold_dataset

from BehaviorScreen.point_process.frailty import GammaMixedEffectsProcess
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory


def test_multistart_keeps_the_start_with_the_lowest_nll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each start 'converges' where it began, to an NLL that grows with its distance from B = 0.7."""
    model, reached = PoissonProcess(RateKernelFactory.spont()), []

    def minimize(x0: np.ndarray, *args, **kwargs) -> SimpleNamespace:
        reached.append(SimpleNamespace(success=True, message="", x=np.array(x0, float), fun=100.0 + (x0[0] - 0.7) ** 2))
        return reached[-1]

    monkeypatch.setattr(model, "_minimize", minimize)
    model.fit_multistart(make_scaffold_dataset(2, 2, 1.0), n_starts=12, n_jobs=1)
    best = min(reached, key=lambda r: r.fun)
    assert len({r.fun for r in reached}) > 6
    assert model.fit_result.fun == best.fun and model.params_ == pytest.approx(best.x)


def test_aic_is_twice_the_parameters_plus_twice_the_nll() -> None:
    model = GammaMixedEffectsProcess(PoissonProcess(RateKernelFactory.spont()))
    model.set_params(np.array(model.initial_guesses, float))
    data = model.simulate_dataset(make_scaffold_dataset(20, 4, 10.0), np.random.default_rng(0))
    model.fit(data)
    assert len(model.params_) == 3
    assert model.aic == pytest.approx(2 * 3 + 2 * model.nll(model.params_, data), rel=1e-12)
