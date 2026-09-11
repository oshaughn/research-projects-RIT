"""Unit tests for the simultaneous slow-rotation + finite-arm response."""
from __future__ import division, print_function

import numpy as np
import lal
import lalsimulation as lalsim

from RIFT.likelihood import factored_likelihood_rotating_freqresponse as combined
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood import factored_likelihood_freqresponse as flfr
from RIFT.likelihood import slowrot_freqresponse as sfr
from RIFT import lalsimutils as lsu

if not getattr(fl, "numba_on", True):
    fl.lalylm = np.vectorize(lal.SpinWeightedSphericalHarmonic, otypes=[complex])


def test_compound_index_set_carries_exact_nonzero_harmonics():
    a0 = combined.compound_index_set(Qmax=4, p_max=0)
    a1 = combined.compound_index_set(Qmax=4, p_max=1)
    assert len(a0) == 50
    assert len(a1) == 112
    assert len(a0) == len(set(a0))
    assert len(a1) == len(set(a1))
    for b, p, n in a1:
        assert abs(n) <= combined.response_harmonic_width(b) + p


def test_basis_harmonics_reconstruct_frequency_response_coefficients():
    rng = np.random.RandomState(20260909)
    nsamp = 5
    ra = rng.uniform(0.0, 2.0 * np.pi, nsamp)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, nsamp))
    psi = rng.uniform(0.0, np.pi, nsamp)
    tref = 1126259462.4
    qmax = 4
    C = combined.combined_response_coefficients_vector(
        'H1', ra, dec, psi, tref, p_max=0, Qmax=qmax, L_arm=40000.0)
    gmst0 = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(tref)))

    worst = 0.0
    for dt in (0.0, 1234.0, 7200.0):
        post = lambda n: np.exp(1j * n * 2.0 * np.pi * combined.flwr.F_SIDEREAL * dt)
        for i in range(nsamp):
            geom = sfr.finite_size_geometry(
                'H1', ra[i], dec[i], psi[i],
                gmst=gmst0 + 2.0 * np.pi * combined.flwr.F_SIDEREAL * dt,
                L_arm=40000.0)
            want = {0: geom['F0']}
            want.update({1 + q: beta for q, beta in enumerate(sfr.finite_size_beta(geom, qmax))})
            for b in range(qmax + 2):
                got = sum(C.get((b, 0, n), np.zeros(nsamp))[i] * post(n)
                          for n in range(-qmax - 2, qmax + 3))
                worst = max(worst, abs(got - want[b]))
    assert worst < 1e-11, "compound sidereal reconstruction error %g" % worst


def test_delay_order_extends_each_frequency_basis_width_without_leakage():
    ra = np.array([0.7, 2.4])
    dec = np.array([-0.3, 0.8])
    psi = np.array([0.2, 1.1])
    qmax = 3
    pmax = 2
    C = combined.combined_response_coefficients_vector(
        'L1', ra, dec, psi, 1126259462.4, pmax, Qmax=qmax, L_arm=40000.0)
    allowed = set(combined.compound_index_set(qmax, pmax))
    assert set(C).issubset(allowed)
    for b in range(qmax + 2):
        for p in range(pmax + 1):
            edge = combined.response_harmonic_width(b) + p
            assert all(abs(n) <= edge for bb, pp, n in C if bb == b and pp == p)


def test_compound_likelihood_reduces_to_freqresponse_at_zero_rotation_rate():
    """Exercise waveform generation, compound packing, and the shared NoLoop contraction."""
    fsample = 1024.0
    event_time = 1000000000.0
    t_window = 0.08
    fmax = 400.0
    psig = lsu.ChooseWaveformParams(
        fmin=30.0, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0,
        psi=0.4, m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI, detector='H1',
        dist=200e6 * lal.PC_SI, deltaT=1.0 / fsample, tref=event_time,
        deltaF=0.5)
    data = {'H1': lsu.non_herm_hoff(psig)}
    psd = {'H1': lalsim.SimNoisePSDaLIGOZeroDetHighPower}

    _, ct_f, ctv_f, rho_f, meta_f = flfr.PrecomputeLikelihoodTermsFreqResponse(
        event_time, t_window, psig, data, psd, 2, fmax, Qmax=0, L_arm=40000.0,
        analyticPSD_Q=True, verbose=False, quiet=True, skip_interpolation=True)
    lk_f, rho_arr_f, u_f, v_f, ep_f = flfr.pack_freqresponse_arrays(
        meta_f, rho_f, ct_f, ctv_f)

    _, ct_c, ctv_c, rho_c, meta = combined.PrecomputeLikelihoodTermsRotatingFreqResponse(
        event_time, t_window, psig, data, psd, 2, fmax, Qmax=0, L_arm=40000.0,
        p_max=1, f_sidereal=0.0, analyticPSD_Q=True, verbose=False, quiet=True,
        skip_interpolation=True)
    lk_c, rho_arr_c, u_c, v_c, ep_c = combined.pack_rotating_freqresponse_arrays(
        meta, rho_c, ct_c, ctv_c)

    pvec = psig.manual_copy()
    for name, val in [('phi', 1.0), ('theta', 0.2), ('incl', 0.7),
                      ('phiref', 0.9), ('psi', 0.5)]:
        setattr(pvec, name, np.full(2, val))
    pvec.dist = np.full(2, 300e6 * lal.PC_SI)
    pvec.tref = event_time
    pvec.deltaT = 1.0 / fsample
    tvals = np.linspace(-0.04, 0.04, 64)

    want = flfr.DiscreteFactoredLogLikelihoodFreqResponseNoLoop(
        tvals, pvec, meta_f, lk_f, rho_arr_f, u_f, v_f, ep_f, Lmax=2,
        array_output=True, time_interp='cubic')
    got = combined.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, pvec, meta, lk_c, rho_arr_c, u_c, v_c, ep_c, Lmax=2,
        array_output=True, time_interp='cubic')
    worst = float(np.max(np.abs(got - want)))
    assert got.shape == want.shape
    assert worst < 1e-8, "compound zero-rotation limit mismatch %g" % worst

    ip = lsu.ComplexIP(30.0, fmax, fsample / 2.0, data['H1'].deltaF,
                       psd['H1'], True, False, 0.0)
    half_dd = 0.5 * ip.ip(data['H1'], data['H1']).real
    assert np.max(got) <= half_dd + 1e-7, "zero-rate compound likelihood violates bound"

    _, ct_w, ctv_w, rho_w, meta_w = combined.PrecomputeLikelihoodTermsRotatingFreqResponse(
        event_time, t_window, psig, data, psd, 2, fmax, Qmax=0, L_arm=40000.0,
        p_max=1, f_sidereal=combined.flwr.F_SIDEREAL, analyticPSD_Q=True,
        verbose=False, quiet=True, skip_interpolation=True)
    lk_w, rho_arr_w, u_w, v_w, ep_w = combined.pack_rotating_freqresponse_arrays(
        meta_w, rho_w, ct_w, ctv_w)
    got_w = combined.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, pvec, meta_w, lk_w, rho_arr_w, u_w, v_w, ep_w, Lmax=2,
        array_output=True, time_interp='cubic')
    assert np.all(np.isfinite(got_w))
    assert np.max(got_w) <= half_dd + 1e-7, "rotating finite-arm likelihood violates bound"
