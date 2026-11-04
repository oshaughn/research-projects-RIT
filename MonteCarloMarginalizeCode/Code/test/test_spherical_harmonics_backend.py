#!/usr/bin/env python
"""Backend dispatch of `SphericalHarmonicsVectorized`.

The helper used to bind `xpy=xpy_default` in its signature, and `xpy_default` is
cupy on any host where cupy merely *imports*.  Callers working in host numpy that
did not name a backend therefore dispatched a GPU kernel onto host arrays and
raised `TypeError: Unsupported type <class 'numpy.ndarray'>` -- on GPU hosts only,
so the cupy-free CI runners never saw it.

These tests fake the GPU backend rather than requiring one, so they hold on any
runner.  The real-cupy leg is skipped where cupy is absent.
"""

import inspect

import numpy as np
import pytest

from RIFT.likelihood import SphericalHarmonics_gpu as sh


MODES = np.array([[2, -2], [2, 0], [2, 2]])


class _FakeGPUArray(object):
    """Stands in for cupy.ndarray: host arrays are never instances of it."""


def _reject_host(*args, **kwargs):
    raise TypeError(
        "Unsupported type <class 'numpy.ndarray'>: a GPU kernel was handed a host array"
    )


class _FakeGPU(object):
    """Minimal stand-in for the cupy module.

    Every array entry point rejects host input the way cupy's ufuncs do, so a
    test that accidentally dispatches here fails with the production symptom
    rather than an incidental AttributeError.
    """

    ndarray = _FakeGPUArray
    empty = staticmethod(_reject_host)
    cos = staticmethod(_reject_host)
    sin = staticmethod(_reject_host)
    square = staticmethod(_reject_host)


def test_default_xpy_is_not_bound_at_import():
    """The default must be resolved per call, not frozen to a module at def time.

    A live module here is the original defect: monkeypatching `xpy_default`
    afterwards cannot dislodge it, and every unadorned caller inherits cupy.
    """
    default = inspect.signature(sh.SphericalHarmonicsVectorized).parameters["xpy"].default
    assert default is None


def test_infer_xpy_prefers_numpy_for_host_arrays_even_with_a_gpu_present(monkeypatch):
    monkeypatch.setattr(sh, "cupy_here", True)
    monkeypatch.setattr(sh, "cupy", _FakeGPU)
    assert sh._infer_xpy(np.linspace(0.1, 3.0, 5)) is np


def test_infer_xpy_selects_the_gpu_backend_for_device_arrays(monkeypatch):
    monkeypatch.setattr(sh, "cupy_here", True)
    monkeypatch.setattr(sh, "cupy", _FakeGPU)
    assert sh._infer_xpy(_FakeGPUArray()) is _FakeGPU


def test_infer_xpy_is_numpy_when_no_gpu_backend_is_installed(monkeypatch):
    monkeypatch.setattr(sh, "cupy_here", False)
    assert sh._infer_xpy(np.linspace(0.1, 3.0, 5)) is np


def test_public_entrypoint_resolves_a_host_backend_on_a_simulated_gpu_host(monkeypatch):
    """Drive SphericalHarmonicsVectorized itself, with `xpy_default` faked to the GPU.

    The `_infer_xpy` tests above exercise the helper in isolation; this one pins
    that SphericalHarmonicsVectorized actually CONSULTS it.  Faking `xpy_default`
    is the whole point -- that module global is where the original bug came from,
    so a body that reads it instead of inferring (`xpy = xpy_default` when xpy is
    None, a plausible "simplification") is caught only here.  Without this test
    that mutation leaves the entire suite green on a cupy-free runner.
    """
    monkeypatch.setattr(sh, "cupy_here", True)
    monkeypatch.setattr(sh, "cupy", _FakeGPU)
    monkeypatch.setattr(sh, "xpy_default", _FakeGPU)

    theta = np.linspace(0.1, np.pi - 0.1, 7)
    phi = np.linspace(0.0, 2.0 * np.pi, 7)

    inferred = sh.SphericalHarmonicsVectorized(MODES, theta, phi, l_max=2)
    assert isinstance(inferred, np.ndarray)

    # An explicit backend must still win over the faked default.
    explicit = sh.SphericalHarmonicsVectorized(MODES, theta, phi, xpy=np, l_max=2)
    np.testing.assert_array_equal(inferred, explicit)


def test_host_arrays_give_the_same_answer_with_and_without_an_explicit_backend():
    """Unadorned call on host arrays must work, and agree with `xpy=np`.

    This is the assertion that fails on a cupy-capable host if the inference is
    removed; it is a tautology on a cupy-free one, which is why it is paired with
    the fake-GPU tests above.
    """
    theta = np.linspace(0.1, np.pi - 0.1, 7)
    phi = np.linspace(0.0, 2.0 * np.pi, 7)

    inferred = sh.SphericalHarmonicsVectorized(MODES, theta, phi, l_max=2)
    explicit = sh.SphericalHarmonicsVectorized(MODES, theta, phi, xpy=np, l_max=2)

    assert isinstance(inferred, np.ndarray)
    np.testing.assert_array_equal(inferred, explicit)


@pytest.mark.skipif(not sh.cupy_here, reason="requires a working cupy install")
def test_device_arrays_stay_on_the_device_without_an_explicit_backend():
    import cupy

    theta = cupy.linspace(0.1, np.pi - 0.1, 7)
    phi = cupy.linspace(0.0, 2.0 * np.pi, 7)

    out = sh.SphericalHarmonicsVectorized(MODES, theta, phi, l_max=2)
    assert isinstance(out, cupy.ndarray)

    host = sh.SphericalHarmonicsVectorized(
        MODES, cupy.asnumpy(theta), cupy.asnumpy(phi), l_max=2
    )
    np.testing.assert_allclose(cupy.asnumpy(out), host, rtol=0, atol=1e-14)
