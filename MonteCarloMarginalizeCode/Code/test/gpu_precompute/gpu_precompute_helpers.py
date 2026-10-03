"""Plain helpers for the gpu_precompute tests.

Import these from here, not from conftest: pytest keeps one module named
``conftest`` in sys.modules, so ``from conftest import ...`` breaks when
another directory's conftest.py is loaded first.
"""
import numpy as np


def to_host(x):
    try:
        import cupy
        if isinstance(x, cupy.ndarray):
            return cupy.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)
