"""
Validate the frequency-domain primitives of factored_likelihood_with_rotation against
LAL FFT round trips (numpy + lal only; no glue / full RIFT stack needed).

Checks:
  1. LAL forward/reverse COMPLEX16 FFT round-trips to identity (normalization sanity).
  2. Which signed frequency LAL assigns to a tone, vs evaluate_fvals_from_length -> fixes
     the sign FT_SIGN in the time-derivative weight.
  3. fd_apply_time_derivative reproduces d^p/dt^p exactly for a multi-tone signal.
  4. _lal_freq_modulate reproduces exp(i coef Omega t) multiplication exactly.
  5. the O(N^2) reference apply_sidereal_modulation_array agrees with the LAL round trip.

Run: python test_slowrot_fd_ops.py   (also usable under pytest)
"""
from __future__ import print_function, division

import os
import importlib.util

import numpy as np
import lal

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "flwr", os.path.join(_here, "factored_likelihood_with_rotation.py"))
flwr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flwr)

N = 256
DELTA_T = 1.0 / 512.0
DELTA_F = 1.0 / (N * DELTA_T)
_T = np.arange(N) * DELTA_T


def _make_timeseries(cvec):
    ts = lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., DELTA_T,
                                       lal.DimensionlessUnit, N)
    ts.data.data[:] = cvec.astype(complex)
    return ts


def _forward(ts):
    """DataFourier-style forward transform."""
    fwd = lal.CreateForwardCOMPLEX16FFTPlan(N, 0)
    hf = lal.CreateCOMPLEX16FrequencySeries("hf", ts.epoch, 0., DELTA_F,
                                            lal.HertzUnit, N)
    lal.COMPLEX16TimeFreqFFT(hf, ts, fwd)
    return hf


def _reverse(hf):
    rev = lal.CreateReverseCOMPLEX16FFTPlan(N, 0)
    ts = lal.CreateCOMPLEX16TimeSeries("h", hf.epoch, 0., DELTA_T,
                                       lal.DimensionlessUnit, N)
    lal.COMPLEX16FreqTimeFFT(ts, hf, rev)
    return ts


def _multitone():
    """h(t) = sum_j c_j exp(2 pi i f_j t), f_j on distinct grid bins."""
    bins = [3, 7, -5, 12, -11]
    coeffs = [1.0, 0.5 - 0.3j, -0.8j, 0.4, 0.2 + 0.1j]
    h = np.zeros(N, dtype=complex)
    for b, c in zip(bins, coeffs):
        h += c * np.exp(2.0j * np.pi * (b * DELTA_F) * _T)
    return h, bins, coeffs


def test_roundtrip_identity():
    h, _, _ = _multitone()
    back = _reverse(_forward(_make_timeseries(h))).data.data
    assert np.max(np.abs(back - h)) < 1e-10, "LAL round trip not identity"
    print("roundtrip: max err = %.2e" % np.max(np.abs(back - h)))


def test_tone_frequency_assignment_and_FT_SIGN():
    """Determine FT_SIGN so that (FT_SIGN 2 pi i f)^1 gives the true derivative."""
    f0_bin = 7
    h = np.exp(2.0j * np.pi * (f0_bin * DELTA_F) * _T)
    hf = _forward(_make_timeseries(h))
    H = hf.data.data
    fvals = flwr.evaluate_fvals_from_length(N, DELTA_F)
    kpeak = int(np.argmax(np.abs(H)))
    f_assigned = fvals[kpeak]
    # true single-tone first derivative is (2 pi i f0) h; the reconstructed weight is
    # (FT_SIGN 2 pi i f_assigned).  Equate -> FT_SIGN = f0 / f_assigned.
    f0 = f0_bin * DELTA_F
    needed_sign = np.sign(f0 / f_assigned)
    print("tone at bin %d: f0=%.4f, evaluate_fvals assigns %.4f -> FT_SIGN should be %+d"
          % (f0_bin, f0, f_assigned, int(needed_sign)))
    assert abs(abs(f_assigned) - abs(f0)) < 1e-9, "tone landed at wrong |f|"
    assert flwr.FT_SIGN == needed_sign, (
        "module FT_SIGN=%g but LAL convention requires %g" % (flwr.FT_SIGN, needed_sign))


def test_time_derivative_exact():
    h, bins, coeffs = _multitone()
    for p in (1, 2, 3):
        # analytic derivative
        dh = np.zeros(N, dtype=complex)
        for b, c in zip(bins, coeffs):
            w = (2.0j * np.pi * b * DELTA_F)
            dh += c * (w ** p) * np.exp(2.0j * np.pi * (b * DELTA_F) * _T)
        hf = _forward(_make_timeseries(h))
        hf_d = flwr.fd_apply_time_derivative(hf, p)
        num = _reverse(hf_d).data.data
        err = np.max(np.abs(num - dh)) / np.max(np.abs(dh))
        print("derivative p=%d: rel err = %.2e" % (p, err))
        assert err < 1e-9, "derivative order %d inexact: %g" % (p, err)


def test_sidereal_modulation_exact():
    h, _, _ = _multitone()
    f_sid = 0.05 * DELTA_F   # exaggerated so coef*f_sid is an appreciable sub-bin shift
    for coef in (1, 2, -1, -2):
        expected = np.exp(2.0j * np.pi * coef * f_sid * _T) * h
        hf = _forward(_make_timeseries(h))
        hf_mod = flwr._lal_freq_modulate(hf, coef, f_sidereal=f_sid)
        num = _reverse(hf_mod).data.data
        err = np.max(np.abs(num - expected)) / np.max(np.abs(expected))
        print("modulation coef=%+d: rel err = %.2e" % (coef, err))
        assert err < 1e-9, "modulation coef %d inexact: %g" % (coef, err)


def test_reference_matrix_matches_lal_modulation():
    """The O(N^2) reference and the LAL round trip must agree."""
    h, _, _ = _multitone()
    f_sid = 0.05 * DELTA_F
    hf = _forward(_make_timeseries(h))
    for coef in (1, -2):
        via_lal = flwr._lal_freq_modulate(hf, coef, f_sidereal=f_sid).data.data
        via_mat = flwr.apply_sidereal_modulation_array(hf.data.data, coef, DELTA_F, f_sid)
        err = np.max(np.abs(via_lal - via_mat)) / np.max(np.abs(via_lal))
        print("reference-vs-LAL coef=%+d: rel err = %.2e" % (coef, err))
        assert err < 1e-9, "reference matrix disagrees with LAL: %g" % err


if __name__ == "__main__":
    test_roundtrip_identity()
    test_tone_frequency_assignment_and_FT_SIGN()
    test_time_derivative_exact()
    test_sidereal_modulation_exact()
    test_reference_matrix_matches_lal_modulation()
    print("ALL FD-PRIMITIVE CHECKS PASSED")
