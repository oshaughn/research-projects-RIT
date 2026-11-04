"""
test_simpson_weights.py: the time-integration rule of the xpy likelihoods.

simpson_function(numpy) is scipy's simpson itself; for other backends it applies scipy's
weights, so CPU and GPU integrate the same way for odd and even numbers of time bins.  The
non-numpy path is exercised here with a stand-in backend module that wraps numpy.
"""

import types

import numpy as np
import pytest

import RIFT.likelihood.factored_likelihood as fl

_stub = types.ModuleType("stub_backend")
_stub.asarray = np.asarray


@pytest.mark.parametrize("npts", [3, 4, 5, 6, 11, 12, 613, 614])
def test_weights_reproduce_scipy(npts):
    rng = np.random.default_rng(npts)
    y = rng.standard_normal((7, npts))
    dx = 1.0 / 4096
    ref = fl.my_simps(y, dx=dx, axis=-1)
    got = fl.simpson_function(_stub)(y, dx=dx, axis=-1)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13 * np.max(np.abs(ref)))
    assert fl.simpson_weights(npts, dx, xpy=_stub) is fl.simpson_weights(npts, dx, xpy=_stub)


def test_numpy_path_is_scipy():
    assert fl.simpson_function(np) is fl.my_simps


def test_only_last_axis():
    with pytest.raises(NotImplementedError):
        fl.simpson_function(_stub)(np.ones((4, 5)), dx=1.0, axis=0)
