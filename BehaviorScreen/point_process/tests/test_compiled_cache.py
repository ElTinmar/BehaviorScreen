"""Compiled executables are released before the process runs out of memory maps."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from BehaviorScreen.point_process import backend


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


def test_checked_after_100_compilations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limit = tmp_path / "max_map_count"
    limit.write_text("2")
    monkeypatch.setattr(backend, "_MAP_LIMIT", limit)
    backend.release_if_crowded()
    value = backend.compiled("test_compiled_count", lambda params, arrays: jnp.sum(params * arrays))[0]
    for n in range(1, 100):
        value(np.ones(n), np.ones(n))
    backend.compiled("test_compiled_count", value)
    assert value._cache_size() == 99  # 99 compilations: not counted yet
    value(np.ones(100), np.ones(100))
    backend.compiled("test_compiled_count", value)
    assert value._cache_size() == 0
