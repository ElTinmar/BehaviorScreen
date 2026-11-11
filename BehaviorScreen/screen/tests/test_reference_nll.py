"""
Every ladder model's NLL on a fixed synthetic dataset, at its initial guesses and
at a perturbed point, against reference values (reference_nll.json); models with a
JAX likelihood must reproduce them too, with the right gradient.
`python -m <this module>` rewrites the references.
"""
import json
from functools import cache
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from BehaviorScreen.screen.analysis.model_config import get_model_config
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.tests.conftest import reference_dataset

REFERENCE = Path(__file__).with_name("reference_nll.json")
RTOL = 1e-9


@pytest.fixture(scope="session")
def reference_data() -> PointProcessDataset:
    return reference_dataset()


def points(model) -> Dict[str, np.ndarray]:
    """The initial guesses, and a point moved 5% (or 0.05) toward the middle of the bounds."""
    x = np.array(model.initial_guesses, dtype=float)
    lower = np.array([-np.inf if lo is None else lo for lo, _ in model.bounds])
    upper = np.array([np.inf if hi is None else hi for _, hi in model.bounds])
    moved = np.clip(x + np.maximum(0.05 * np.abs(x), 0.05), lower, upper)
    return {"initial": x, "moved": moved}


@cache
def all_models() -> Dict[str, object]:
    return {m.name: m for config in get_model_config().values() for m in config["models"]}


def compute() -> Dict[str, Dict[str, float]]:
    dataset = reference_dataset()
    return {name: {key: float(model._nll(list(x), dataset)) for key, x in points(model).items()} for name, model in all_models().items()}


class TestReferenceNLL:
    @pytest.mark.parametrize("name", sorted(json.loads(REFERENCE.read_text())) if REFERENCE.exists() else [])
    def test_reference_nll(self, name: str, reference_data: PointProcessDataset) -> None:
        model = all_models()[name]
        expected = json.loads(REFERENCE.read_text())[name]
        for key, x in points(model).items():
            assert model._nll(list(x), reference_data) == pytest.approx(expected[key], rel=RTOL), key

    @pytest.mark.parametrize("name", sorted(n for n, m in all_models().items() if m.supports_jax()))
    def test_jax_likelihood(self, name: str, reference_data: PointProcessDataset) -> None:
        """The JAX likelihood has the reference values, and its gradient the NumPy likelihood's central differences."""
        model = all_models()[name]
        expected = json.loads(REFERENCE.read_text())[name]
        for key, x in points(model).items():
            assert model.jax_value(x, reference_data) == pytest.approx(expected[key], rel=RTOL), key
        x = points(model)["moved"]
        h = 1e-6 * (1.0 + np.abs(x))
        numeric = [(model._nll(list(x + e), reference_data) - model._nll(list(x - e), reference_data)) / (2 * hi) for e, hi in zip(np.diag(h), h)]
        assert model.nll_and_grad(x, reference_data)[1] == pytest.approx(np.array(numeric), rel=1e-4, abs=1e-3)

    def test_every_ladder_model_has_a_reference(self) -> None:
        assert set(all_models()) <= set(json.loads(REFERENCE.read_text()))


if __name__ == "__main__":
    REFERENCE.write_text(json.dumps(compute(), indent=1, sort_keys=True))
