"""
NumPy or JAX. Kernels are written once against `xp`, whose functions run in JAX when
any argument is a JAX array (inside a traced likelihood) and in NumPy otherwise
(simulation, plots). Likelihoods are JAX functions of parameters and `Arrays`.
"""
from functools import lru_cache
from typing import Any, Callable, Dict, Hashable, Tuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.scipy.special as jsp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.special as sp  # noqa: E402


_NUMPY = (np.ndarray, np.generic, float, int)


def is_jax(*args: Any) -> bool:
    """Whether any argument, or element of a list or tuple argument, is a JAX array (NumPy ones checked first: they are fast to tell)."""
    for a in args:
        if isinstance(a, _NUMPY):
            continue
        if isinstance(a, jax.Array) or (isinstance(a, (list, tuple)) and is_jax(*a)):
            return True
    return False


class _Namespace:
    """np.<name> or jnp.<name> (scipy.special or jax.scipy.special for `special`), by the arguments."""

    def __init__(self, name: str, numpy_module: Any, jax_module: Any) -> None:
        self._name, self._numpy, self._jax = name, numpy_module, jax_module

    def __reduce__(self) -> Tuple:
        return getattr, (__import__(__name__, fromlist=["_"]), self._name)  # the module's own instance

    @lru_cache(maxsize=None)
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        numpy_f, jax_f = getattr(self._numpy, name), getattr(self._jax, name, None)
        if not callable(numpy_f) or jax_f is None:
            return numpy_f
        return lambda *args, **kwargs: (jax_f if is_jax(*args, *kwargs.values()) else numpy_f)(*args, **kwargs)


xp = _Namespace("xp", np, jnp)
special = _Namespace("special", sp, jsp)


@jax.tree_util.register_pytree_node_class
class Arrays:
    """What a likelihood needs from a dataset: arrays (traced) and static values (part of the compiled function)."""

    def __init__(self, arrays: Dict[str, Any], static: Dict[str, Hashable]) -> None:
        self.arrays, self.static = arrays, static

    def __getitem__(self, key: str) -> Any:
        return self.arrays[key] if key in self.arrays else self.static[key]

    def tree_flatten(self) -> Tuple[tuple, tuple]:
        keys = tuple(sorted(self.arrays))
        return tuple(self.arrays[k] for k in keys), (keys, tuple(sorted(self.static.items())))

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: tuple) -> "Arrays":
        keys, static = aux
        return cls(dict(zip(keys, children)), dict(static))


def segment_sum(values: Any, segments: Any, n: int) -> Any:
    """Sums per segment 0..n-1; values of segments >= n (padding) are dropped."""
    return jax.ops.segment_sum(values, segments, num_segments=n)


def padded(n: int, minimum: int = 16, steps_per_doubling: int = 4) -> int:
    """Array length for n entries, in steps of 2^(1/4) by default: datasets of similar size share a compiled likelihood."""
    k = steps_per_doubling
    return int(np.ceil(minimum * 2 ** (np.ceil(k * np.log2(max(n, minimum) / minimum) - 1e-9) / k)))


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    """Weights such that dot(weights, values) is the trapezoid integral on grid."""
    weights = np.zeros(len(grid))
    weights[1:] += np.diff(grid) / 2
    weights[:-1] += np.diff(grid) / 2
    return weights


def pad(values: np.ndarray, size: int, fill: Any) -> np.ndarray:
    return np.concatenate([values, np.full(size - len(values), fill, dtype=np.asarray(values).dtype)])


_COMPILED: Dict[Hashable, Tuple[Callable, Callable, Callable]] = {}


def _hessian(nll: Callable) -> Callable:
    """Hessian of nll(params, arrays) one column at a time: the memory of one gradient, not of k at once (jax.hessian)."""
    grad = jax.grad(nll)
    return lambda params, arrays: jax.lax.map(lambda v: jax.jvp(lambda p: grad(p, arrays), (params,), (v,))[1], jnp.eye(params.size))


def compiled(key: Hashable, nll: Callable) -> Tuple[Callable, Callable, Callable]:
    """Jitted value, value and gradient, and Hessian of nll(params, arrays), one set per key (a model's structure)."""
    if key not in _COMPILED:
        _COMPILED[key] = (jax.jit(nll), jax.jit(jax.value_and_grad(nll)), jax.jit(_hessian(nll)))
    return _COMPILED[key]
