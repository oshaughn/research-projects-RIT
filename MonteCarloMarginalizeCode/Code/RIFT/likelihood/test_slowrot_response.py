"""
Validate slowrot_response against LAL over a full sidereal day.

The harmonic decomposition is an exact algebraic identity in the hour angle g, so the
reconstruction must match lal.ComputeDetAMResponse and lal.TimeDelayFromEarthCenter to
machine precision (up to LAL's own internal rounding).  Run directly:

    python test_slowrot_response.py

or under pytest (the test_* functions assert).
"""
from __future__ import print_function, division

import numpy as np

import lal
import lalsimulation as lalsim

# Load the leaf module directly by path so this test needs only numpy+lal, not the full
# RIFT package stack (whose __init__ pulls in glue/lalsimutils).  When the package is
# properly installed, `import RIFT.likelihood.slowrot_response` is equivalent.
try:
    import RIFT.likelihood.slowrot_response as sr
except Exception:
    import importlib.util
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "slowrot_response", os.path.join(_here, "slowrot_response.py"))
    sr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sr)

DETECTORS = ["H1", "L1", "V1", "K1"]
SIDEREAL_DAY = 86164.0905   # s
T0_GPS = 1000000000.0       # arbitrary reference epoch

_TOL = 1e-11                # radians of F, seconds of tau -- essentially machine precision


def _lal_detector(det):
    return lalsim.DetectorPrefixToLALDetector(det)


def _sample_gmst(n=512):
    tgps = T0_GPS + np.linspace(0.0, SIDEREAL_DAY, n, endpoint=False)
    gmst = np.array([lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(t))) for t in tgps])
    return tgps, gmst


def _max_antenna_error(det, ra, dec, psi):
    lald = _lal_detector(det)
    A = sr.antenna_harmonics(lald.response, dec, psi)
    _, gmst = _sample_gmst()
    # model
    Fp_mod, Fc_mod = sr.antenna_response(A, gmst, ra)
    # reference (LAL), sampled at the same GMST values
    Fp_ref = np.empty_like(gmst)
    Fc_ref = np.empty_like(gmst)
    for i, g in enumerate(gmst):
        Fp_ref[i], Fc_ref[i] = lal.ComputeDetAMResponse(lald.response, ra, dec, psi, float(g))
    return max(np.max(np.abs(Fp_mod - Fp_ref)), np.max(np.abs(Fc_mod - Fc_ref)))


def _max_delay_error(det, ra, dec):
    lald = _lal_detector(det)
    B = sr.delay_harmonics(lald.location, dec)
    tgps, gmst = _sample_gmst()
    g = sr.greenwich_hour_angle(gmst, ra)
    tau_mod = sr.delay_from_harmonics(B, g)
    tau_ref = np.array([
        lal.TimeDelayFromEarthCenter(lald.location, ra, dec, lal.LIGOTimeGPS(float(t)))
        for t in tgps
    ])
    return np.max(np.abs(tau_mod - tau_ref))


# ---- deterministic parameter sweep -------------------------------------------------
_RNG = np.random.RandomState(20260703)
_CASES = []
for _det in DETECTORS:
    for _ in range(5):
        _ra = _RNG.uniform(0, 2 * np.pi)
        _dec = np.arcsin(_RNG.uniform(-1, 1))
        _psi = _RNG.uniform(0, np.pi)
        _CASES.append((_det, _ra, _dec, _psi))


def test_antenna_harmonics_match_lal():
    worst = 0.0
    for det, ra, dec, psi in _CASES:
        e = _max_antenna_error(det, ra, dec, psi)
        worst = max(worst, e)
        assert e < _TOL, "antenna mismatch %g for %s ra=%.3f dec=%.3f psi=%.3f" % (
            e, det, ra, dec, psi)
    print("antenna: worst |dF| = %.3e" % worst)


def test_delay_harmonics_match_lal():
    worst = 0.0
    for det, ra, dec, psi in _CASES:
        e = _max_delay_error(det, ra, dec)
        worst = max(worst, e)
        assert e < _TOL, "delay mismatch %g s for %s ra=%.3f dec=%.3f" % (e, det, ra, dec)
    print("delay:   worst |dtau| = %.3e s" % worst)


def test_only_five_and_three_harmonics():
    """Confirm there is genuinely no content beyond |n|=2 (antenna) / |n|=1 (delay)."""
    lald = _lal_detector("H1")
    ra, dec, psi = 1.0, 0.4, 0.7
    _, gmst = _sample_gmst(1024)
    g = sr.greenwich_hour_angle(gmst, ra)
    # antenna: DFT of F(g) sampled uniformly in g should have power only at |n|<=2
    gg = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    F = sr.antenna_from_harmonics(sr.antenna_harmonics(lald.response, dec, psi), gg)
    coeffs = np.fft.fft(F) / len(gg)
    power = np.abs(coeffs)
    keep = np.zeros_like(power, dtype=bool)
    for n in (-2, -1, 0, 1, 2):
        keep[n % len(gg)] = True
    assert np.max(power[~keep]) < 1e-12, "antenna has power beyond |n|=2: %g" % np.max(power[~keep])
    tau = sr.delay_from_harmonics(sr.delay_harmonics(lald.location, dec), gg)
    tcoef = np.abs(np.fft.fft(tau) / len(gg))
    keep = np.zeros_like(tcoef, dtype=bool)
    for n in (-1, 0, 1):
        keep[n % len(gg)] = True
    assert np.max(tcoef[~keep]) < 1e-15, "delay has power beyond |n|=1: %g" % np.max(tcoef[~keep])
    print("harmonic content confirmed: antenna |n|<=2, delay |n|<=1")


if __name__ == "__main__":
    test_antenna_harmonics_match_lal()
    test_delay_harmonics_match_lal()
    test_only_five_and_three_harmonics()
    print("ALL SLOWROT RESPONSE CHECKS PASSED")
