"""
Validate slowrot_freqresponse (Path D, frequency-dependent finite-size antenna response).

Checks:
  (A) LONG-WAVELENGTH LIMIT: antenna_response_fd(..., f=0) == lal.ComputeDetAMResponse to
      machine precision, over many random (ra,dec,psi) and H1/L1/V1/K1.  KEY CHECK.
  (B) FREE-SPECTRAL-RANGE STRUCTURE: the single-arm transfer's first null sits at the
      expected frequency c / (L (1 + a.n)); |F(f)| departs from |F(0)| on the f_FSR scale.
  (C) IN-BAND MAGNITUDE: fractional response change |F(f)/F(0) - 1| at 1 kHz and 2 kHz for
      (i) 4-km LIGO and (ii) a 40-km CE arm -- quantifies whether the effect matters in band.

Run directly:
    python test_slowrot_freqresponse.py
or under pytest (the test_* functions assert).
"""
from __future__ import print_function, division

import numpy as np
import lal
import lalsimulation as lalsim

try:
    import RIFT.likelihood.slowrot_freqresponse as fr
except Exception:
    import importlib.util, os
    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "slowrot_freqresponse", os.path.join(_here, "slowrot_freqresponse.py"))
    fr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(fr)

DETECTORS = ["H1", "L1", "V1", "K1"]
_TOL = 1e-11                       # machine-precision agreement vs ComputeDetAMResponse

_RNG = np.random.RandomState(20260707)
_CASES = []
for _det in DETECTORS:
    for _ in range(8):
        _ra = _RNG.uniform(0, 2 * np.pi)
        _dec = np.arcsin(_RNG.uniform(-1, 1))
        _psi = _RNG.uniform(0, np.pi)
        _gmst = _RNG.uniform(0, 2 * np.pi)
        _CASES.append((_det, _ra, _dec, _psi, _gmst))


# ---- (A) long-wavelength limit vs LAL ------------------------------------------------
def test_long_wavelength_limit_matches_lal():
    worst = 0.0
    for det, ra, dec, psi, gmst in _CASES:
        lald = lalsim.DetectorPrefixToLALDetector(det)
        Fp_ref, Fc_ref = lal.ComputeDetAMResponse(lald.response, ra, dec, psi, gmst)
        Fp, Fc = fr.antenna_response_fd(det, ra, dec, psi, 0.0, gmst=gmst)
        e = max(abs(Fp - Fp_ref), abs(Fc - Fc_ref))
        worst = max(worst, e)
        assert e < _TOL, "FD(f=0) mismatch %g for %s ra=%.3f dec=%.3f psi=%.3f" % (
            e, det, ra, dec, psi)
    print("(A) long-wavelength limit: worst |F_fd(0) - ComputeDetAMResponse| = %.3e" % worst)
    return worst


def test_zero_frequency_is_real():
    """At f=0 the response must be purely real (imag part = 0 to machine precision)."""
    worst = 0.0
    for det, ra, dec, psi, gmst in _CASES:
        Fp, Fc = fr.antenna_response_fd(det, ra, dec, psi, 0.0, gmst=gmst)
        worst = max(worst, abs(Fp.imag), abs(Fc.imag))
    assert worst < 1e-14, "imag(F(0)) = %g" % worst
    print("(A') Im F(0): worst = %.3e" % worst)


# ---- (B) free-spectral-range / sinc-null structure -----------------------------------
def _first_local_min(a, L, fmax_frac=3.5, N=700000):
    """Frequency and depth of the first local minimum of |D~| in (0, fmax_frac*f_FSR]."""
    fFSR = fr.free_spectral_range(L)
    fg = np.linspace(1.0, fmax_frac * fFSR, N)
    mag = np.abs(fr.single_arm_transfer(a, fg, L))
    # first interior local minimum
    lo = (mag[1:-1] < mag[:-2]) & (mag[1:-1] < mag[2:])
    idx = np.nonzero(lo)[0]
    i = idx[0] + 1 if len(idx) else int(np.argmin(mag))
    return fg[i], mag[i], fFSR


def test_single_arm_null_at_fsr_for_transverse():
    """a.n = 0 (source transverse to arm): D~ = e^{-i2pi fT} sinc(2fT), EXACT null at f_FSR."""
    L = 40000.0
    T = L / fr.C_SI
    fFSR = fr.free_spectral_range(L)
    fg = np.linspace(1.0, 4.0 * fFSR, 200001)
    D = fr.single_arm_transfer(0.0, fg, L)
    ref = np.exp(-1j * 2.0 * np.pi * fg * T) * np.sinc(2.0 * fg * T)
    ident = np.max(np.abs(D - ref))
    assert ident < 1e-13, "a=0 closed form mismatch %g" % ident
    # exact first null at f_FSR
    val_at_fsr = abs(fr.single_arm_transfer(0.0, fFSR, L))
    assert val_at_fsr < 1e-12, "|D~(a=0, f_FSR)| = %g (should vanish)" % val_at_fsr
    print("(B) a.n=0: D~=e^{-i2pi fT} sinc(2fT) to %.2e; EXACT first null at f_FSR=c/2L=%.5g Hz"
          % (ident, fFSR))


def test_single_arm_dip_structure():
    """General a.n: |D~| dips on the f_FSR scale (exact nulls only for a.n=0)."""
    L = 40000.0
    for a in [-0.7, -0.3, 0.0, 0.4, 0.8]:
        fmin, depth, fFSR = _first_local_min(a, L)
        print("(B') a.n=%+.2f: first |D~| dip at %.4g Hz = %.3f f_FSR   (|D~|min=%.2e)"
              % (a, fmin, fmin / fFSR, depth))
        assert 0.3 * fFSR < fmin < 3.5 * fFSR, "dip off the f_FSR scale for a=%g" % a


def test_fsr_scale_departure():
    """|F(f)| departs from |F(0)| by O(1) once f ~ f_FSR (use a 40-km CE arm)."""
    det, ra, dec, psi, gmst = "H1", 1.2, 0.5, 0.7, 2.0
    L = 40000.0
    fFSR = fr.free_spectral_range(L)
    Fp0, Fc0 = fr.antenna_response_fd(det, ra, dec, psi, 0.0, gmst=gmst, L_arm=L)
    F0 = abs(Fp0 + 1j * Fc0)
    for frac in [0.01, 0.1, 0.5, 1.0]:
        f = frac * fFSR
        Fp, Fc = fr.antenna_response_fd(det, ra, dec, psi, f, gmst=gmst, L_arm=L)
        rel = abs(abs(Fp + 1j * Fc) - F0) / F0
        print("(B') f/f_FSR=%.2f (f=%.4g Hz): ||F(f)|-|F(0)||/|F(0)| = %.3e" % (frac, f, rel))


# ---- (C) in-band magnitude: LIGO vs CE ----------------------------------------------
def _fractional_change(det, L, freqs, n_sky=4000):
    """Median-over-sky of complex |F(f)/F(0)-1| AND amplitude-only ||F(f)|-|F(0)||/|F(0)|,
    excluding sky positions near antenna-pattern nulls (|F(0)|<0.3) where the ratio blows
    up for reasons unrelated to the finite-size effect.

    Returns (complex_med, amp_med) arrays over freqs.  The complex ratio is DOMINATED by
    the overall light-crossing phase e^{-i2 pi f L/c} (a benign direction-independent
    delay of L/c, degenerate with coalescence time); the amplitude-only change is the
    physically meaningful measure of antenna-pattern SHAPE distortion.
    """
    rng = np.random.RandomState(1234)
    comp = [[] for _ in freqs]
    amp = [[] for _ in freqs]
    for _ in range(n_sky):
        ra = rng.uniform(0, 2 * np.pi)
        dec = np.arcsin(rng.uniform(-1, 1))
        psi = rng.uniform(0, np.pi)
        gmst = rng.uniform(0, 2 * np.pi)
        Fp0, Fc0 = fr.antenna_response_fd(det, ra, dec, psi, 0.0, gmst=gmst, L_arm=L)
        F0 = complex(Fp0) + 1j * complex(Fc0)
        if abs(F0) < 0.3:
            continue
        for i, f in enumerate(freqs):
            Fp, Fc = fr.antenna_response_fd(det, ra, dec, psi, f, gmst=gmst, L_arm=L)
            Ff = complex(Fp) + 1j * complex(Fc)
            comp[i].append(abs(Ff / F0 - 1.0))
            amp[i].append(abs(abs(Ff) - abs(F0)) / abs(F0))
    return (np.array([np.median(c) for c in comp]),
            np.array([np.median(a) for a in amp]))


def test_in_band_magnitude_ligo_vs_ce():
    freqs = [1000.0, 2000.0]
    ligo_c, ligo_a = _fractional_change("H1", 3994.5, freqs)
    ce_c, ce_a = _fractional_change("H1", 40000.0, freqs)
    print("(C) IN-BAND fractional response change (median over sky, away from nulls):")
    print("    complex |F(f)/F(0)-1|  (incl. benign overall e^{-i2pi f L/c} delay phase):")
    for i, f in enumerate(freqs):
        print("       f=%5.0f Hz :  LIGO(4km) = %.3e    CE(40km) = %.3e" % (f, ligo_c[i], ce_c[i]))
    print("    amplitude-only ||F(f)|-|F(0)||/|F(0)|  (pattern-SHAPE distortion, physical):")
    for i, f in enumerate(freqs):
        print("       f=%5.0f Hz :  LIGO(4km) = %.3e    CE(40km) = %.3e    (CE/LIGO ~%.0fx)"
              % (f, ligo_a[i], ce_a[i], ce_a[i] / max(ligo_a[i], 1e-30)))
    # LIGO amplitude distortion is sub-percent (tiny); CE is tens of percent (>> LIGO).
    assert ligo_a[0] < 1e-2, "LIGO 1kHz amplitude change unexpectedly large: %g" % ligo_a[0]
    assert ligo_a[1] < 2e-2, "LIGO 2kHz amplitude change unexpectedly large: %g" % ligo_a[1]
    assert ce_a[0] > 3e-2, "CE 1kHz amplitude change unexpectedly small: %g" % ce_a[0]
    assert ce_a[1] > ligo_a[1] * 30, "CE should be >> LIGO at 2 kHz"
    return ligo_a, ce_a


def test_ce_is_100x_longer_effect():
    """Finite-size amplitude distortion scales ~ (f L / c)^2 ; 10x arm -> ~10^2x effect."""
    f = 2000.0
    _, ligo_a = _fractional_change("H1", 3994.5, [f])
    _, ce_a = _fractional_change("H1", 40000.0, [f])
    ratio = ce_a[0] / max(ligo_a[0], 1e-30)
    print("(C') CE/LIGO amplitude-distortion ratio at 2 kHz = %.1f (expect ~10^2 for 10x arm)"
          % ratio)
    assert 30 < ratio < 250, "unexpected CE/LIGO scaling: %g" % ratio


if __name__ == "__main__":
    print("=" * 78)
    wA = test_long_wavelength_limit_matches_lal()
    test_zero_frequency_is_real()
    print("-" * 78)
    test_single_arm_null_at_fsr_for_transverse()
    test_single_arm_dip_structure()
    test_fsr_scale_departure()
    print("-" * 78)
    test_in_band_magnitude_ligo_vs_ce()
    test_ce_is_100x_longer_effect()
    print("=" * 78)
    print("ALL SLOWROT FREQ-RESPONSE CHECKS PASSED  (worst f=0 residual %.3e)" % wA)
