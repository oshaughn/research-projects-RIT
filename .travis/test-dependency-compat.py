#!/usr/bin/env python3
"""Smoke-test upstream scientific-library compatibility for RIFT CI.

This intentionally stays small: the full RIFT import sweep catches broad
breakage, while this file makes the dependency surface in issue #17 explicit
and gives CI logs a compact version matrix for scipy/h5py/numba/matplotlib.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import numba
import numpy as np
import scipy
from scipy import integrate, optimize, stats

from RIFT.precision import RIFT_FLOAT_HIGH_PRECISION, RIFT_FLOAT_NAME, RiftFloat


@numba.njit(cache=False)
def _numba_quadratic(x):
    return x * x + 2.0 * x + 1.0


def _check_scipy() -> None:
    integral, err = integrate.quad(lambda x: x * x, 0.0, 1.0)
    if not np.isclose(integral, 1.0 / 3.0, atol=1e-12):
        raise AssertionError(f"scipy.integrate returned {integral} +/- {err}")

    root = optimize.brentq(lambda x: x * x - 2.0, 0.0, 2.0)
    if not np.isclose(root, np.sqrt(2.0), atol=1e-12):
        raise AssertionError(f"scipy.optimize returned root {root}")

    cdf = stats.norm.cdf(0.0)
    if not np.isclose(cdf, 0.5, atol=1e-15):
        raise AssertionError(f"scipy.stats returned normal cdf {cdf}")


def _check_h5py() -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "compat.h5"
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("values", data=values)
            h5_file.attrs["library"] = "h5py"
        with h5py.File(h5_path, "r") as h5_file:
            roundtrip = h5_file["values"][()]
            marker = h5_file.attrs["library"]
    if marker != "h5py" or not np.array_equal(values, roundtrip):
        raise AssertionError("h5py dataset/attribute round trip failed")


def _check_numba() -> None:
    sample = np.asarray([0.0, 1.0, 2.0])
    expected = np.asarray([1.0, 4.0, 9.0])
    actual = _numba_quadratic(sample)
    if not np.allclose(actual, expected):
        raise AssertionError(f"numba compiled function returned {actual}")


def _check_matplotlib() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = Path(tmpdir) / "compat.png"
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
        if png_path.stat().st_size <= 0:
            raise AssertionError("matplotlib produced an empty PNG")


def _check_rift_precision() -> None:
    dtype = np.dtype(RiftFloat)
    if dtype.itemsize < np.dtype(np.float64).itemsize:
        raise AssertionError(f"RiftFloat unexpectedly narrower than float64: {dtype}")
    if RIFT_FLOAT_HIGH_PRECISION != (dtype.itemsize > np.dtype(np.float64).itemsize):
        raise AssertionError("RIFT_FLOAT_HIGH_PRECISION does not match RiftFloat width")


def main() -> None:
    matrix = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "numba": numba.__version__,
        "matplotlib": matplotlib.__version__,
        "rift_float": RIFT_FLOAT_NAME,
        "rift_float_high_precision": RIFT_FLOAT_HIGH_PRECISION,
    }
    print(json.dumps(matrix, indent=2, sort_keys=True))

    _check_rift_precision()
    _check_scipy()
    _check_h5py()
    _check_numba()
    _check_matplotlib()


if __name__ == "__main__":
    main()
