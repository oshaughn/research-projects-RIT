#!/usr/bin/env python
"""std_and_conj_hlmoff must key modes by (l,m) when hlmoft returns a SphHarmTimeSeries."""

import lal
import lalsimulation as lalsim
import numpy as np
import pytest

import RIFT.lalsimutils as lsu


DELTA_T = 1. / 256
NPTS = 256   # nextPow2 of the 227-sample TaylorT4 signal below


def _sphharm_modes(npts=None):
    # Build lists with a real generator: chaining SphHarmTimeSeriesAddMode from
    # python lets swig free earlier list heads, which drops modes.
    hlms = lalsim.SimInspiralChooseTDModes(
        0., DELTA_T, 10 * lal.MSUN_SI, 10 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0,
        40., 40., 1e8 * lal.PC_SI, None, 2, lalsim.TaylorT4)
    if npts is not None:
        hlms = lalsim.ResizeSphHarmTimeSeries(hlms, 0, npts)
    return hlms


def _fourier(data):
    ts = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., DELTA_T,
                                       lal.DimensionlessUnit, len(data))
    ts.data.data[:] = data
    return lsu.DataFourier(ts).data.data


@pytest.mark.parametrize("delta_f", [1. / (NPTS * DELTA_T), None])
def test_sphharm_branch_keys_modes(monkeypatch, delta_f):
    npts_in = NPTS if delta_f else None   # None branch pads to NPTS itself
    monkeypatch.setattr(lsu, "hlmoft",
                        lambda P, Lmax, **kw: _sphharm_modes(npts_in))

    P = lsu.ChooseWaveformParams()
    P.deltaT = DELTA_T
    P.deltaF = delta_f
    hlmsF, hlms_conj_F = lsu.std_and_conj_hlmoff(P, Lmax=2)

    ref = _sphharm_modes(NPTS)
    modes = [(2, m) for m in range(-2, 3)]
    assert set(hlmsF) == set(modes)
    assert set(hlms_conj_F) == set(modes)
    for l, m in modes:
        h = lalsim.SphHarmTimeSeriesGetMode(ref, l, m).data.data.copy()
        np.testing.assert_allclose(hlmsF[(l, m)].data.data, _fourier(h))
        np.testing.assert_allclose(hlms_conj_F[(l, m)].data.data,
                                   _fourier(np.conj(h)))
