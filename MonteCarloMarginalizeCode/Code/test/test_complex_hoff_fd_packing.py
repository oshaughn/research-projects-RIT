"""complex_hoff's FD-approximant branch must match DataFourier of the same complex h(t).

The old packing wrote the upper half of the two-sided array two bins off and never wrote
the last bin, which kept whatever the uninitialized lal buffer held (#492).  The reference
inverse-FFTs the same one-sided h+, hx spectra, forms h+ - i hx, and transforms with
DataFourier, so it shares RIFT's two-sided packing.
"""
import types

import lal
import lalsimulation as lalsim
import numpy as np

from RIFT import lalsimutils as lsu


def _params():
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 35 * lal.MSUN_SI, 30 * lal.MSUN_SI
    P.s1z, P.s2z = 0.3, -0.2
    P.fmin = P.fref = 20.0
    P.deltaT, P.deltaF = 1.0 / 2048, 1.0 / 4
    P.incl = 0.7
    P.approx = lalsim.IMRPhenomD
    P.radec = False
    return P


def _reference(P):
    n = int(round(1 / (P.deltaT * P.deltaF)))
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        P.m1, P.m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z,
        P.dist, P.incl, P.phiref, P.psi, P.eccentricity, P.meanPerAno,
        P.deltaF, P.fmin, n * P.deltaF / 2, P.fref,
        P.to_lal_dict_extended(extra_args_dict={}), P.approx)
    plan = lal.CreateReverseREAL8FFTPlan(n, 0)
    pol = []
    for fs in (hp, hc):
        fs = lal.ResizeCOMPLEX16FrequencySeries(fs, 0, n // 2 + 1)
        ts = lal.CreateREAL8TimeSeries("t", fs.epoch, 0, P.deltaT, lal.DimensionlessUnit, n)
        lal.REAL8FreqTimeFFT(ts, fs, plan)
        pol.append(np.array(ts.data.data))
    ht = lal.CreateCOMPLEX16TimeSeries("h", hp.epoch, 0, P.deltaT, lal.DimensionlessUnit, n)
    ht.data.data = pol[0] - 1j * pol[1]  # complex_hoff default sgn = -1
    return np.array(lsu.DataFourier(ht).data.data)


def test_fd_branch_matches_datafourier():
    P = _params()
    ref = _reference(P)
    got = np.array(lsu.complex_hoff(P).data.data)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * np.max(np.abs(ref)))


def test_fd_branch_writes_every_bin(monkeypatch):
    def poisoned_create(*args):
        fs = lal.CreateCOMPLEX16FrequencySeries(*args)
        fs.data.data = np.full(fs.data.length, np.nan, dtype=complex)
        return fs

    proxy = types.SimpleNamespace(**vars(lal))
    proxy.CreateCOMPLEX16FrequencySeries = poisoned_create
    monkeypatch.setattr(lsu, "lal", proxy)
    assert np.all(np.isfinite(lsu.complex_hoff(_params()).data.data))
