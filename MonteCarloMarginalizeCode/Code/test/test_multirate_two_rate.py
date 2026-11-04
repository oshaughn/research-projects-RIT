"""Two-rate DSWc Q against the full-rate ComputeModeIPTimeSeries on a short case
(DESIGN_early_time_multirate.md). Templates: RIFT's own calls with per-|m| f_min, XHM internal
multibanding off on every piece (its grid depends on f_min), and a post-event time long enough
that the FD-branch end taper stays off the merger. The archival, full-size version of this
check is acceptance test 2 in RIFT_roboto_paper analyses/early_time_compression/rift_interface/."""
import numpy as np
import pytest
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood import factored_likelihood_multirate as flm

FS, FS_E, SEGLEN, FLOW, FMAX, POST = 4096, 128, 256, 20., 1700., 24
LATE_BUF, LATE_START, TAU_TR = 256, 150., 80.
MBAND_OFF = dict(PhenomXHMThresholdMband=0, PhenomXPHMThresholdMband=0)
M1, M2 = 1.4, 1.35


def _modes(srate, seglen, f22_start, restrict=True):
    out = {}
    for am in (1, 2):
        extra = dict(MBAND_OFF)
        if restrict:
            extra["ModeArray"] = flm.mode_array_for_m(2, am)
        P = lsu.ChooseWaveformParams()
        P.m1, P.m2 = M1 * lal.MSUN_SI, M2 * lal.MSUN_SI
        P.fmin = (am / 2.) * f22_start; P.fref = FLOW
        P.deltaT, P.deltaF = 1. / srate, 1. / seglen
        P.approx = lalsim.IMRPhenomXHM; P.dist = 100e6 * lal.PC_SI; P.taper = lsu.lsu_TAPER_START
        hF, _ = lsu.std_and_conj_hlmoff(P, Lmax=2, fd_alignment_postevent_time=POST, extra_waveform_args=extra)
        out.update({k: v for k, v in hF.items() if abs(k[1]) == am})
    return out


def _td(hf):
    return np.array(lsu.DataInverseFourier(hf).data.data)


@pytest.fixture(scope="module")
def setup():
    dt, df, N = 1. / FS, 1. / SEGLEN, FS * SEGLEN
    psd = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0), 0., df, lsu.lsu_HertzUnit, N // 2 + 1)
    lalsim.SimNoisePSDaLIGOAdVO4T1800545(psd, 10.)
    v = np.array(psd.data.data); v[v <= 0] = np.inf; psd.data.data = v
    ref = _modes(FS, SEGLEN, FLOW)
    early = _modes(FS_E, SEGLEN, FLOW)
    mc = (M1 * M2)**0.6 / (M1 + M2)**0.2
    late = _modes(FS, LATE_BUF, float(flm.f22_newtonian(LATE_START, mc)))
    Y = complex(lal.SpinWeightedSphericalHarmonic(0.6, 0.3, -2, 2, 2))
    inj = np.real(0.5 * np.exp(0.3j) * Y * _td(ref[(2, 2)]))
    rng = np.random.default_rng(5)
    d = lal.CreateCOMPLEX16TimeSeries("d", lal.LIGOTimeGPS(0), 0., dt, lsu.lsu_DimensionlessUnit, N)
    d.data.data = inj + 1e-3 * np.max(np.abs(inj)) * rng.standard_normal(N) + 0j
    data = lsu.DataFourier(d)
    sch = flm.TwoRateSchedule(FS, FS_E, TAU_TR, 4., 160., mc_min=0.999 * mc, m_max=2, lag_halfwidth=0.05)
    return dict(dt=dt, N=N, psd=psd, ref=ref, early=early, late=late, data=data, sch=sch)


def _two_rate(s, keys, wh_scale=1.0, early_method="interp"):
    dt, N, sch = s["dt"], s["N"], s["sch"]
    nlag = int(round(0.05 * FS)); lags = np.arange(-nlag, nlag + 1)
    j_peak = N - POST * FS
    dbar = flm.data_side_weighted(s["data"], s["psd"], FLOW, FMAX, FS / 2.)
    prep = flm.prepare_data_two_rate(dbar, dt, j_peak, sch)
    Qr = fl.ComputeModeIPTimeSeries(s["ref"], s["data"], s["psd"], FLOW, FMAX, FS / 2., -nlag, 2 * nlag + 1)
    out = {}
    for k in keys:
        hE = _td(flm.taper_top_quarter(lal.ResizeCOMPLEX16FrequencySeries(s["early"][k], 0, s["early"][k].data.length), FS_E))
        q = flm.Q_two_rate(prep, wh_scale * hE, _td(s["late"][k]), j_peak, N - LATE_BUF * FS, sch, dt, lags,
                         early_method=early_method)
        out[k] = (q, np.array(Qr[k].data.data))
    return out


@pytest.mark.parametrize("early_method", ["interp", "subphase"])
def test_two_rate_matches_full_rate(setup, early_method):
    for k, (q, r) in _two_rate(setup, [(2, 2), (2, 1)], early_method=early_method).items():
        assert np.max(np.abs(q - r)) <= 1e-6 * np.max(np.abs(r)), k


def test_interp_matches_subphase(setup):
    """One coarse-lag FFT plus lag interpolation reproduces the M sub-phase FFTs."""
    a = _two_rate(setup, [(2, 2), (2, 1)], early_method="interp")
    b = _two_rate(setup, [(2, 2), (2, 1)], early_method="subphase")
    for k in a:
        assert np.max(np.abs(a[k][0] - b[k][0])) <= 1e-8 * np.max(np.abs(b[k][1])), k


def test_mode_array_lmax4_all_m():
    """Every per-|m| ModeArray at l <= 4 is accepted by XHM and returns exactly that |m|'s modes."""
    for am in (1, 2, 3, 4):
        P = lsu.ChooseWaveformParams()
        P.m1, P.m2 = M1 * lal.MSUN_SI, M2 * lal.MSUN_SI
        P.fmin = am * 15.; P.fref = FLOW; P.deltaT, P.deltaF = 1. / FS_E, 1. / 64
        P.approx = lalsim.IMRPhenomXHM; P.dist = 100e6 * lal.PC_SI; P.taper = lsu.lsu_TAPER_START
        hF, _ = lsu.std_and_conj_hlmoff(P, Lmax=4, fd_alignment_postevent_time=POST,
                                        extra_waveform_args=dict(MBAND_OFF, ModeArray=flm.mode_array_for_m(4, am)))
        want = {(l, s * m) for (l, m) in flm.XHM_MODES if m == am for s in (1, -1)}
        assert set(hF) == want, (am, sorted(hF))


def test_mode_array_call_matches_full_call():
    """A per-|m| call restricted by ModeArray (no (2,2) in the |m| = 1 call) returns the same
    modes as the unrestricted call at the same f_min."""
    a = _modes(FS_E, SEGLEN, FLOW, restrict=True)
    b = _modes(FS_E, SEGLEN, FLOW, restrict=False)
    assert set(a) == set(b)
    for k in a:
        assert np.max(np.abs(a[k].data.data - b[k].data.data)) <= 1e-12 * np.max(np.abs(b[k].data.data)), k


def test_two_rate_detects_wrong_early_amplitude(setup):
    """The comparison has teeth: a 0.1% error on the early template is caught."""
    q, r = _two_rate(setup, [(2, 2)], wh_scale=1.001)[(2, 2)]
    assert np.max(np.abs(q - r)) > 1e-6 * np.max(np.abs(r))
