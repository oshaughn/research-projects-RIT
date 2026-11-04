"""rotate_hlm_static accepts every input type it documents, and they agree.

The FrequencySeries branch tested isinstance against lal.Complex16FrequencySeries, which
does not exist (the type is lal.COMPLEX16FrequencySeries), so FrequencySeries and ndarray
inputs raised AttributeError and only TimeSeries worked.  The ndarray branch also
allocated a real output array.  Found 2026-10-18 by the cupy XPHM port (PR #453).
"""

import lal
import numpy as np
import pytest

import RIFT.lalsimutils as lsu

N = 16
ANGLES = (0.3, 0.7, 1.1)
POL = 0.2


def _modes(lmax=3, skip=()):
    rng = np.random.default_rng(0)
    return {(l, m): rng.normal(size=N) + 1j * rng.normal(size=N)
            for l in range(2, lmax + 1) for m in range(-l, l + 1) if (l, m) not in skip}


def _wrap(arrs, kind):
    out = {}
    for k, v in arrs.items():
        if kind == "ndarray":
            out[k] = v.copy()
            continue
        if kind == "time":
            s = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0), 0, 1. / 16,
                                              lsu.lsu_DimensionlessUnit, N)
        else:
            s = lal.CreateCOMPLEX16FrequencySeries("h", lal.LIGOTimeGPS(0), 0, 0.25,
                                                   lsu.lsu_HertzUnit, N)
        s.data.data[:] = v
        out[k] = s
    return out


def _unwrap(h):
    return {k: (v if isinstance(v, np.ndarray) else v.data.data) for k, v in h.items()}


def _reference(arrs, alpha, beta, gamma, pol):
    lmax = max(l for l, _ in arrs)
    out = {}
    for l in range(2, lmax + 1):
        for m in range(-l, l + 1):
            acc = np.zeros(N, dtype=complex)
            for (lp, mp), v in arrs.items():
                if lp == l:
                    acc += lal.WignerDMatrix(l, m, mp, alpha, beta, gamma) * v
            out[(l, m)] = acc * np.exp(-2j * pol)
    return out


@pytest.mark.parametrize("kind", ["time", "freq", "ndarray"])
def test_matches_direct_wigner_sum(kind):
    arrs = _modes()
    out = lsu.rotate_hlm_static(_wrap(arrs, kind), *ANGLES, extra_polarization=POL)
    ref = _reference(arrs, *ANGLES, POL)
    expected_type = {"time": lal.COMPLEX16TimeSeries, "freq": lal.COMPLEX16FrequencySeries,
                     "ndarray": np.ndarray}[kind]
    assert set(out) == set(ref)
    for k, v in _unwrap(out).items():
        assert isinstance(out[k], expected_type)
        np.testing.assert_allclose(v, ref[k], rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["time", "freq", "ndarray"])
def test_inverse_rotation_recovers_input(kind):
    arrs = _modes()
    alpha, beta, gamma = ANGLES
    once = lsu.rotate_hlm_static(_wrap(arrs, kind), alpha, beta, gamma)
    back = _unwrap(lsu.rotate_hlm_static(once, -gamma, -beta, -alpha))
    for k, v in arrs.items():
        np.testing.assert_allclose(back[k], v, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["time", "freq", "ndarray"])
def test_partial_input_gives_full_output(kind):
    # the docstring promises input need not contain every (l, m)
    arrs = _modes(skip=((2, -1), (3, 0)))
    out = _unwrap(lsu.rotate_hlm_static(_wrap(arrs, kind), *ANGLES, extra_polarization=POL))
    ref = _reference(arrs, *ANGLES, POL)
    assert set(out) == set(ref)
    for k in ref:
        np.testing.assert_allclose(out[k], ref[k], rtol=0, atol=1e-12)
