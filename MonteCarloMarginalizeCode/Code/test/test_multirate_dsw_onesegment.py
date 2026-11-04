"""Acceptance test 1 for the early-time multirate precompute (DESIGN_early_time_multirate.md):
a one-segment, full-rate data-side-weighted Q reproduces ComputeModeIPTimeSeries, under
production flags (fd_alignment_postevent_time=2 on a power-of-2 segment of at least 8 s)."""
import numpy as np
import pytest
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood import factored_likelihood_multirate as flm

SRATE, SEGLEN, FMIN, FMAX = 4096, 16, 30., 1700.


def _setup(inv_spec_trunc_Q=False, T_spec=0.):
    deltaT, deltaF = 1. / SRATE, 1. / SEGLEN
    N = SRATE * SEGLEN
    rng = np.random.default_rng(3)
    ht = lal.CreateCOMPLEX16TimeSeries("d", lal.LIGOTimeGPS(1e9), 0., deltaT, lsu.lsu_DimensionlessUnit, N)
    ht.data.data = rng.standard_normal(N) * 1e-21 + 0j
    data = lsu.DataFourier(ht)
    n1 = N // 2 + 1
    psd = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0), 0., deltaF, lsu.lsu_HertzUnit, n1)
    f = np.arange(n1) * deltaF
    psd.data.data = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(max(x, 10.)) for x in f])
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 1.4 * lal.MSUN_SI, 1.3 * lal.MSUN_SI
    P.fmin = FMIN; P.fref = FMIN; P.deltaT = deltaT; P.deltaF = deltaF
    P.approx = lalsim.IMRPhenomD; P.dist = 100e6 * lal.PC_SI; P.taper = lsu.lsu_TAPER_START
    hlms, _ = lsu.std_and_conj_hlmoff(P, Lmax=2, fd_alignment_postevent_time=2)
    return data, psd, hlms, deltaT


@pytest.mark.parametrize("trunc", [(False, 0.), (True, 4.)])
def test_one_segment_dsw_matches_ComputeModeIPTimeSeries(trunc):
    data, psd, hlms, deltaT = _setup(*trunc)
    N_shift, N_window = -205, 410
    ref = fl.ComputeModeIPTimeSeries(hlms, data, psd, FMIN, FMAX, 0.5 / deltaT, N_shift, N_window,
                                     False, trunc[0], trunc[1])
    dbar = flm.data_side_weighted(data, psd, FMIN, FMAX, 0.5 / deltaT, False, trunc[0], trunc[1])
    new = flm.ComputeModeIPTimeSeriesDSW(hlms, dbar, data.epoch, deltaT, N_shift, N_window)
    assert set(ref) == set(new)
    for k in ref:
        a, b = ref[k].data.data, new[k].data.data
        assert abs(float(ref[k].epoch) - float(new[k].epoch)) < 1e-9
        assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a))


@pytest.mark.parametrize("trunc", [(False, 0.), (True, 4.)])
def test_one_segment_UV_match_ComputeModeCrossTermIP(trunc):
    """Acceptance test 1b: U and V in the time-domain form equal ComputeModeCrossTermIP."""
    data, psd, hlms, deltaT = _setup(*trunc)
    P_conj = {}
    for k, v in hlms.items():                    # conjugate modes as std_and_conj_hlmoff builds them
        t = lsu.DataInverseFourier(v)
        t.data.data = np.conj(t.data.data)
        P_conj[k] = lsu.DataFourier(t)
    for A, prefix in ((hlms, "U"), (P_conj, "V")):
        ref = fl.ComputeModeCrossTermIP(A, hlms, psd, FMIN, FMAX, 0.5 / deltaT, 1. / SEGLEN, False,
                                        trunc[0], trunc[1], verbose=False, prefix=prefix, batched=False)
        new = flm.ComputeModeCrossTermIPDSW(A, hlms, psd, FMIN, FMAX, 0.5 / deltaT, 1. / SEGLEN, False,
                                            trunc[0], trunc[1], prefix=prefix)
        scale = max(abs(v) for v in ref.values())
        for key in ref:
            assert abs(ref[key] - new[key]) <= 1e-12 * scale, (prefix, key)
