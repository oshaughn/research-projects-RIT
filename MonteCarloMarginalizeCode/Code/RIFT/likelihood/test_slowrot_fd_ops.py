"""
Validate the frequency-domain primitives of factored_likelihood_with_rotation against
LAL FFT round trips (numpy + lal only; no glue / full RIFT stack needed).

Checks:
  1. LAL forward/reverse COMPLEX16 FFT round-trips to identity (normalization sanity).
  2. Which signed frequency LAL assigns to a tone, vs evaluate_fvals_from_length -> fixes
     the sign FT_SIGN in the time-derivative weight.
  3. fd_apply_time_derivative reproduces d^p/dt^p exactly for a multi-tone signal.
  4. fd_apply_time_derivative COMMUTES with conjugation and maps real -> real when the
     signal has Nyquist-bin content, AND gives the right VALUE there at both parities of p
     (issue #159).
  5. _lal_freq_modulate reproduces exp(i coef Omega t) multiplication exactly.
  6. the O(N^2) reference apply_sidereal_modulation_array agrees with the LAL round trip.

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


def test_derivative_commutes_with_conjugation_at_nyquist():
    """d/dt conj(h) == conj(d/dt h), with the Nyquist bin POPULATED.  See issue #159.

    This packing carries +fNyq (index 0) but not -fNyq, so a derivative weight -- odd in f --
    has no consistent value there.  Left at +(2 pi i fNyq)^p, the two routes below disagree in
    that one bin by a sign for odd p.  Nothing in the U cross terms notices, because both
    factors come from the same template family; V = <chi_a^*|chi_a'> pairs the two routes
    against each other, and the sidereal modulation (a sub-bin shift done as a time-domain
    phase) then spreads that single bin across the whole band.  In the p_max=1 slow-rotation
    bank that was worth 1.5e-07 of the model norm -- enough to break Cauchy-Schwarz.

    Zeroing the Nyquist weight AT ODD p is what makes these two routes agree AND keeps
    d^p/dt^p of a real series real; both are asserted here, at odd and even p alike (even p
    already commutes, and must keep doing so).  Without the Nyquist tone this test passes
    either way, so keep the tone.  Consistency does NOT pin the weight's value -- any real
    w[+fNyq] passes this test -- so read it together with
    test_nyquist_derivative_value_both_parities, which does.
    """
    h, bins, coeffs = _multitone()
    h = h + 0.6 * np.exp(2.0j * np.pi * (N // 2 * DELTA_F) * _T)      # the +fNyq bin
    hf_nyq = _forward(_make_timeseries(h)).data.data[0]
    assert abs(hf_nyq) > 1e-3 * np.max(np.abs(h)), (
        "this test is vacuous unless the Nyquist bin actually carries power (got %g)"
        % abs(hf_nyq))
    for p in range(1, 7):
        a = np.conj(_reverse(flwr.fd_apply_time_derivative(
            _forward(_make_timeseries(h)), p)).data.data)              # differentiate, then conj
        b = _reverse(flwr.fd_apply_time_derivative(
            _forward(_make_timeseries(np.conj(h))), p)).data.data      # conj, then differentiate
        err = np.max(np.abs(a - b)) / np.max(np.abs(b))
        print("conj/derivative commutation p=%d: rel err = %.2e" % (p, err))
        # 1e-9 is the same gate test_time_derivative_exact uses, and it is a ROUNDOFF
        # bound, not slack: the two routes are the same arithmetic through different FFTs,
        # and (2 pi f)^p amplifies the round trip, so the residual grows with p while the
        # odd-p normalisation shrinks (the zeroed Nyquist term drops out of the
        # denominator).  Measured with the fix in: 2.8e-15 / 6.1e-16 / 3.3e-13 / 4.2e-16 /
        # 3.3e-11 / 4.6e-16 at p = 1..6.  Without it the residual is 1.7e+00 to 3.0e+01 --
        # eight orders clear of this gate, so tightening it buys nothing and p >= 5 would
        # fail on precision alone.
        assert err < 1e-9, (
            "d/dt does not commute with conjugation at order %d (rel %g): the Nyquist bin of "
            "time_derivative_weight is inconsistent, and crossTermsV_rot pairs the two orders "
            "-- see issue #159" % (p, err))

        # ... and the derivative of a REAL series must be real.
        r = np.real(h)
        dr = _reverse(flwr.fd_apply_time_derivative(
            _forward(_make_timeseries(r.astype(complex))), p)).data.data
        imag = np.max(np.abs(np.imag(dr))) / np.max(np.abs(dr))
        print("real-in real-out p=%d: |Im|/|.| = %.2e" % (p, imag))
        assert imag < 1e-9, (
            "d^%d/dt^%d of a real series came back complex (|Im|/|.| = %g)" % (p, p, imag))


def test_nyquist_derivative_value_both_parities():
    """Pin the VALUE of the Nyquist weight, at both parities.  See issue #159.

    The commutation test below is necessary but NOT sufficient: ANY REAL value of
    w[+fNyq] commutes with conjugation and keeps a real series real, so consistency alone
    does not pin the weight.  This one does, from the sampled signal:

      * the real Nyquist component is (-1)^j = cos(2 pi fNyq t) sampled.  Its ODD
        derivatives are -2 pi fNyq sin(2 pi fNyq t) etc, which vanish at every sample, so
        the correct weight at odd p is exactly ZERO -- and that is also the only value that
        can serve both +fNyq and -fNyq, which share this one bin.
      * its EVEN derivatives are (-(2 pi fNyq)^2)^(p/2) (-1)^j, exactly representable, so
        the untouched weight is correct and zeroing it would be a regression.  An earlier
        revision of the #159 fix zeroed every p >= 1: that removes the even-p Nyquist term
        ENTIRELY, so it fails below at rel err 1.00 (90% at p = 2 and 99% at p = 4 when
        measured against a full multitone rather than the isolated tone).
    """
    fnyq = 1.0 / (2.0 * DELTA_T)
    nyq = np.exp(2.0j * np.pi * (N // 2 * DELTA_F) * _T)          # == (-1)^j, real
    assert np.max(np.abs(np.imag(nyq))) < 1e-12
    base, _, _ = _multitone()
    h = np.real(base) + 0.6 * np.real(nyq)                        # real, WITH Nyquist power
    hf = _forward(_make_timeseries(h.astype(complex)))
    assert abs(hf.data.data[0]) > 1e-3 * np.max(np.abs(h)), (
        "vacuous unless the Nyquist bin carries power (got %g)" % abs(hf.data.data[0]))

    for p in range(1, 7):        # p >= 5 too: --rotation-p-max is an unbounded int
        # the Nyquist tone's own contribution, isolated: differentiate it alone.
        hf_n = _forward(_make_timeseries((0.6 * np.real(nyq)).astype(complex)))
        got_n = _reverse(flwr.fd_apply_time_derivative(hf_n, p)).data.data
        scale = np.max(np.abs(_reverse(flwr.fd_apply_time_derivative(hf, 0)).data.data))
        if p % 2:
            err = np.max(np.abs(got_n)) / (scale * (2.0 * np.pi * fnyq) ** p)
            print("nyquist value p=%d (odd, want 0): |d^p x_nyq| / scale = %.2e" % (p, err))
            assert err < 1e-12, (
                "odd derivative of the sampled Nyquist component must vanish (got %g of "
                "the naive weight); w[+fNyq] is not zero -- see issue #159" % err)
        else:
            want = 0.6 * (-(2.0 * np.pi * fnyq) ** 2) ** (p // 2) * np.real(nyq)
            err = np.max(np.abs(got_n - want)) / np.max(np.abs(want))
            print("nyquist value p=%d (even, want exact): rel err = %.2e" % (p, err))
            assert err < 1e-10, (
                "even derivative of the Nyquist component IS representable and must be "
                "exact (rel %g) -- do not zero the Nyquist weight for even p, see #159"
                % err)


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
    test_derivative_commutes_with_conjugation_at_nyquist()
    test_nyquist_derivative_value_both_parities()
    test_time_derivative_exact()
    test_sidereal_modulation_exact()
    test_reference_matrix_matches_lal_modulation()
    print("ALL FD-PRIMITIVE CHECKS PASSED")
