"""Regression test for the hlmoft TD dispatch gap (IMRPhenomT / IMRPhenomTHM).

TD-only IMR approximants (SimInspiralImplementedFDApproximants == 0) miss the
hlmoft_FromFD_dict branch of RIFT.lalsimutils.hlmoft.  Unless they are named in
the explicit SimInspiralChooseTDModes branch they fall through to a fallback
that conditions the mode array with a DIFFERENT epoch and merger placement
(e.g. IMRPhenomT: epoch -9.38 s / peak at 58.6% of the array, vs -7.79 s /
48.7% for a properly dispatched approximant, at fmin=50, seglen=16 s,
2.2+1.8 Msun).  Time-sensitive likelihoods (the slow-rotation U/V cross terms)
then produce Cauchy-Schwarz-violating lnL.

This test asserts that every TD-only IMR approximant in the list below, when
available in the installed lalsuite, yields an epoch and fractional peak
position consistent with the reference approximant IMRPhenomTPHM.  Unavailable
approximants are skipped with a clear message.

Run directly (python test_td_dispatch_epoch.py) or under pytest.
"""

import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils

REF_NAME = "IMRPhenomTPHM"
# TD-only IMR approximants that hlmoft must route through SimInspiralChooseTDModes
# (or an equivalently conditioned path).  Extend as new TD-only models appear.
TEST_NAMES = ["IMRPhenomT", "IMRPhenomTHM", "SEOBNRv4"]

# The bug displaces the epoch by ~1.6 s and the peak by ~10% of the array;
# legitimate inter-approximant scatter is ~4 ms and ~0.03%.
EPOCH_TOL_S = 0.1
PEAK_FRAC_TOL = 0.01


def _available(name):
    if not hasattr(lalsim, name):
        return False, "lalsimulation has no approximant '{}'".format(name)
    a = getattr(lalsim, name)
    if lalsim.SimInspiralImplementedTDApproximants(a) != 1:
        return False, "'{}' is not TD-implemented in this lalsuite".format(name)
    return True, ""


def _measure(name):
    """Return (epoch_seconds, peak_index_fraction) of the (2,2) mode from hlmoft."""
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = 2.2 * lal.MSUN_SI
    P.m2 = 1.8 * lal.MSUN_SI
    P.s1x = P.s1y = P.s1z = P.s2x = P.s2y = P.s2z = 0.0
    P.fmin = 50.0
    P.deltaT = 1.0 / 16384
    P.deltaF = 1.0 / 16
    P.dist = 100e6 * lal.PC_SI
    P.approx = getattr(lalsim, name)
    hlms = lalsimutils.hlmoft(P, Lmax=2)
    try:
        h22 = hlms[(2, 2)]
    except TypeError:
        # SphHarmTimeSeries linked list (some hlmoft branches)
        h22 = lalsim.SphHarmTimeSeriesGetMode(hlms, 2, 2)
    amp = np.abs(h22.data.data)
    ipk = int(np.argmax(amp))
    return float(h22.epoch), ipk / float(h22.data.length)


def test_td_dispatch_epoch():
    ok, why = _available(REF_NAME)
    if not ok:
        _skip("reference approximant unavailable: " + why)
        return
    ref_epoch, ref_frac = _measure(REF_NAME)
    print("{:15s} epoch {:+.5f} s   peak {:.3f}%  (reference)".format(
        REF_NAME, ref_epoch, 100 * ref_frac))
    failures = []
    for name in TEST_NAMES:
        ok, why = _available(name)
        if not ok:
            print("{:15s} SKIP: {}".format(name, why))
            continue
        try:
            epoch, frac = _measure(name)
        except Exception as e:
            failures.append("{}: hlmoft raised {}: {}".format(name, type(e).__name__, e))
            continue
        d_epoch = abs(epoch - ref_epoch)
        d_frac = abs(frac - ref_frac)
        status = "OK" if (d_epoch < EPOCH_TOL_S and d_frac < PEAK_FRAC_TOL) else "FAIL"
        print("{:15s} epoch {:+.5f} s   peak {:.3f}%   d_epoch {:.4f} s  d_frac {:.5f}  {}".format(
            name, epoch, 100 * frac, d_epoch, d_frac, status))
        if status == "FAIL":
            failures.append(
                "{}: epoch {:+.5f} s (ref {:+.5f}, tol {} s), peak frac {:.5f} (ref {:.5f}, tol {})".format(
                    name, epoch, ref_epoch, EPOCH_TOL_S, frac, ref_frac, PEAK_FRAC_TOL))
    assert not failures, (
        "TD dispatch epoch/merger placement inconsistent with {}:\n  ".format(REF_NAME)
        + "\n  ".join(failures))


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("SKIP: " + msg)


if __name__ == "__main__":
    test_td_dispatch_epoch()
    print("test_td_dispatch_epoch: PASS")
