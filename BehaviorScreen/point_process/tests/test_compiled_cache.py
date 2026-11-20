"""Compiled executables are released before the process runs out of memory maps."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from BehaviorScreen.point_process import backend
from BehaviorScreen.point_process.frailty import GammaMixedEffectsProcess
from BehaviorScreen.point_process.poisson_process import PoissonProcess, PreyCapture
from BehaviorScreen.point_process.sided_process import CONTRA, IPSI, SidedSurvivalProcess
from BehaviorScreen.point_process.survival_process import SurvivalKernelFactory


def test_released_when_maps_run_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = backend.compiled("test_compiled_cache", lambda params, arrays: jnp.sum(params * arrays))[0]
    for n in range(1, 6):
        value(jnp.ones(n), jnp.ones(n))
    assert value._cache_size() == 5
    limit = tmp_path / "max_map_count"
    limit.write_text("1000000000")
    monkeypatch.setattr(backend, "_MAP_LIMIT", limit)
    backend.release_if_crowded()
    assert value._cache_size() == 5  # plenty of maps left: kept
    limit.write_text("2")
    backend.release_if_crowded()
    assert value._cache_size() == 0
    assert float(value(jnp.ones(2), jnp.ones(2))) == 2.0  # compiled again on the next call


def test_models_differing_outside_their_name_do_not_share_a_likelihood() -> None:
    prey = lambda stim_freq: GammaMixedEffectsProcess(PoissonProcess(PreyCapture.variant("Peak", stim_freq)))
    assert prey(0.5).name == prey(1.0).name and prey(0.5)._jax_key() == prey(0.5)._jax_key() != prey(1.0)._jax_key()
    escape = lambda preferred: SidedSurvivalProcess(SurvivalKernelFactory.constant_hazard(), preferred)
    assert escape(IPSI)._jax_key() != escape(CONTRA)._jax_key()


def test_checked_after_each_compilation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limit = tmp_path / "max_map_count"
    limit.write_text("2")
    monkeypatch.setattr(backend, "_MAP_LIMIT", limit)
    backend.release_if_crowded()
    nll = lambda params, arrays: jnp.sum(params * arrays)
    value = backend.compiled("test_compiled_count", nll)[0]
    value(np.ones(3), np.ones(3))
    assert value._cache_size() == 1
    backend.compiled("test_compiled_count", nll)
    assert value._cache_size() == 0  # counted at the call after the compilation
    checks = []
    monkeypatch.setattr(backend, "release_if_crowded", lambda: checks.append(1))
    backend.compiled("test_compiled_count", nll)
    assert not checks  # nothing compiled since: not counted
