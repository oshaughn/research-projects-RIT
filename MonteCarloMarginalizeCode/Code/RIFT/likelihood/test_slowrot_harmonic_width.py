"""
test_slowrot_harmonic_width : the precompute must carry EVERY harmonic the response
coefficients populate (issue #142).

`rotation_coefficients` builds C_{(p,ntilde)} by convolving the antenna harmonics
(|n| <= 2) with the delay-drift harmonics (|m| <= 1) once per derivative order, so the
harmonic index widens by exactly one per order and the bank needs |ntilde| <= 2 + p_max.
`PrecomputeLikelihoodTermsWithRotation` builds one elementary-template band per requested
harmonic, and a coefficient with no band is dropped WITHOUT COMPLAINT by both maintained
evaluators (the NoLoop's Cg/Cg_d return zero for a missing `a`; the JAX packer in
jax_ile.banded packs only `a_list`).  A too-narrow `harmonics` therefore used to yield a
quietly truncated model.

Checks:
  W0  the antenna / delay half-widths the module hard-codes are the ones slowrot_response
      actually produces (so N_ANTENNA_HARMONICS / N_DELAY_HARMONICS cannot drift silently)
  W1  the measured index set of rotation_coefficients (and _vector) is exactly
      -(2+p_max) .. +(2+p_max), for p_max = 0..3 -- i.e. required_harmonic_width is
      MEASURED, not asserted
  W2  a too-narrow request is widened, and says so (RuntimeWarning naming the width)
  W3  the resulting bank drops NO response coefficient -- the property that matters, and
      the one that is evaluator-independent
  W4  the control: with widen_harmonics=False (the pre-fix behaviour) the same request DOES
      drop coefficients, is flagged meta['harmonics_truncated'], and moves lnL.  Without
      this, W2/W3 would be guards nobody has seen fail.
  W5  the JAX packer: every key jax_ile.response_slowrot produces has a band, and
      jax_ile.banded.build_rotation_data itself packs the full widened bank and refuses to
      accept a truncated one in silence
  W6  the MAINTAINED evaluator: pack_rotation_arrays + the vectorized NoLoop.  The fix
      changes what that path receives (|a_list| 10 -> 14 for the default request at
      p_max=1), so the widened bank must run through it and must move lnL relative to the
      truncated one; and packing a truncated bank must warn rather than evaluate quietly.

Run: PYTHONPATH=.../Code python RIFT/likelihood/test_slowrot_harmonic_width.py
"""
from __future__ import print_function, division

import warnings

import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
import RIFT.likelihood.slowrot_response as srr

fmin = 30.; fmax = 1700.; event_time = 1e9; t_window = 0.1; Lmax = 2
deltaT = 1 / 4096.; deltaF = 1 / 4.
DET = 'H1'
P_MAX = 1                       # the first p_max at which the (-2..2) default is too narrow
NARROW = (-2, -1, 0, 1, 2)      # the module default: the p_max=0 answer
# Truncation must move lnL by at least this much.  Measured on this configuration:
# 7.66e+03 nats (scalar) / 7.48e+03 nats (NoLoop), so this is ~3.5 orders of margin -- far
# above float noise, and it asserts the truncation is MATERIAL, not merely nonzero.
DLNL_MIN = 1.0

Psig = lsu.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI, detector=DET,
    dist=200e6 * lal.PC_SI, deltaT=deltaT, tref=event_time, deltaF=deltaF)
data_dict = {DET: lsu.non_herm_hoff(Psig)}
psd_dict = {DET: lalsim.SimNoisePSDaLIGOZeroDetHighPower}

extr = lsu.ChooseWaveformParams(radec=True, phi=1.0, theta=0.2, psi=0.4, incl=0.3,
                                phiref=0.0, tref=event_time, dist=200e6 * lal.PC_SI)

_BANKS = {}


def _bank(widen):
    """Precompute with the NARROW default request; widen=False is the pre-fix behaviour."""
    if widen not in _BANKS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rr = flwr.PrecomputeLikelihoodTermsWithRotation(
                event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
                harmonics=NARROW, p_max=P_MAX, f_sidereal=flwr.F_SIDEREAL,
                analyticPSD_Q=True, verbose=False, quiet=True,
                skip_interpolation=False, widen_harmonics=widen)
        msgs = [str(c.message) for c in caught
                if issubclass(c.category, RuntimeWarning) and 'harmonics' in str(c.message)]
        _BANKS[widen] = (rr, msgs)
    return _BANKS[widen]


def _coef_keys(p_max):
    """Every (p, ntilde) the response coefficients actually populate at these extrinsics."""
    return set(flwr.rotation_coefficients(DET, extr.phi, extr.theta, extr.psi,
                                          event_time, p_max))


def _lnL(rr):
    return float(flwr.FactoredLogLikelihoodWithRotation(extr, rr[0], rr[1], rr[2], rr[4], Lmax))


def _lnL_noloop(rr):
    """Same bank through the MAINTAINED vectorized path.  Returns (lnL, warning messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lk, ra, cu, cv, ep = flwr.pack_rotation_arrays(rr[4], rr[3], rr[1], rr[2])
    msgs = [str(c.message) for c in caught
            if issubclass(c.category, RuntimeWarning) and 'pack_rotation_arrays' in str(c.message)]
    Pv = Psig.manual_copy()
    for k, v in [('phi', extr.phi), ('theta', extr.theta), ('incl', extr.incl),
                 ('phiref', extr.phiref), ('psi', extr.psi), ('dist', extr.dist)]:
        setattr(Pv, k, np.ones(1) * v)
    Pv.tref = event_time; Pv.deltaT = deltaT
    tvals = np.arange(200) * deltaT - 0.01
    out = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        tvals, Pv, rr[4], lk, ra, cu, cv, ep, Lmax=Lmax, array_output=False, xpy=np)
    return float(out[0]), msgs


# ---------------------------------------------------------------------------
def test_W0_antenna_and_delay_half_widths():
    lald = lalsim.DetectorPrefixToLALDetector(DET)
    A = srr.antenna_harmonics(lald.response, 0.2, 0.5)
    B = srr.delay_harmonics(lald.location, 0.2)
    wA = max(abs(int(n)) for n in A)
    wB = max(abs(int(m)) for m in B)
    print("W0 antenna half-width=%d (module says %d)   delay half-width=%d (module says %d)"
          % (wA, flwr.N_ANTENNA_HARMONICS, wB, flwr.N_DELAY_HARMONICS))
    assert wA == flwr.N_ANTENNA_HARMONICS, \
        "antenna half-width drifted: %d vs N_ANTENNA_HARMONICS=%d" % (wA, flwr.N_ANTENNA_HARMONICS)
    assert wB == flwr.N_DELAY_HARMONICS, \
        "delay half-width drifted: %d vs N_DELAY_HARMONICS=%d" % (wB, flwr.N_DELAY_HARMONICS)


def test_W1_required_width_is_measured():
    for p_max in (0, 1, 2, 3):
        C = flwr.rotation_coefficients(DET, 1.0, 0.2, 0.5, event_time, p_max)
        ns = sorted(set(n for (_, n) in C))
        Cv = flwr.rotation_coefficients_vector(DET, np.array([1.0]), np.array([0.2]),
                                               np.array([0.5]), event_time, p_max)
        nsv = sorted(set(n for (_, n) in Cv))
        w = flwr.required_harmonic_width(p_max)
        print("W1 p_max=%d -> harmonic indices %s ; required_harmonic_width=%d" % (p_max, ns, w))
        assert ns == nsv, "scalar/vector coefficient index sets disagree: %s vs %s" % (ns, nsv)
        assert ns == list(range(-w, w + 1)), \
            "required_harmonic_width(%d)=%d does not match the measured index set %s" % (p_max, w, ns)


def test_W2_narrow_request_is_widened_and_says_so():
    rr, msgs = _bank(True)
    meta = rr[4]
    w = flwr.required_harmonic_width(P_MAX)
    print("W2 requested=%s -> carried=%s (required half-width %d); warning: %s"
          % (meta['harmonics_requested'], meta['harmonics'], w,
             msgs[0] if msgs else "NONE"))
    assert meta['harmonics_requested'] == NARROW
    assert set(range(-w, w + 1)).issubset(set(meta['harmonics'])), \
        "bank still too narrow: %s" % (meta['harmonics'],)
    assert meta['harmonics_required'] == w
    assert meta['harmonics_truncated'] is False
    assert meta['harmonics'] == tuple(sorted(set(NARROW) | set(range(-w, w + 1)))), \
        "widened set is not the union of the request with the required range: %s" % (meta['harmonics'],)
    assert msgs, "widening happened silently -- no RuntimeWarning was raised"
    # not `str(w) in msgs[0]`: "3" also appears in "p_max=1" arithmetic and in "(-3, -2, ...".
    assert ("2 + p_max = %d" % w) in msgs[0], \
        "the warning does not name the required width as such: %s" % msgs[0]


def test_W3_widened_bank_drops_no_coefficient():
    rr, _ = _bank(True)
    a_list = set(rr[4]['a_list'])
    missing = sorted(_coef_keys(P_MAX) - a_list)
    print("W3 widened bank: |a_list|=%d, response coefficients with no band: %s"
          % (len(a_list), missing))
    assert not missing, \
        "response coefficients %s have no elementary-template band and will be dropped" % (missing,)


def test_W4_control_narrow_bank_really_does_truncate():
    """The guard above is only worth something if it can fail.  widen_harmonics=False is
    the pre-fix behaviour, in-tree: it must drop coefficients, flag itself, and move lnL."""
    rr_n, msgs_n = _bank(False)
    rr_w, _ = _bank(True)
    a_list = set(rr_n[4]['a_list'])
    missing = sorted(_coef_keys(P_MAX) - a_list)
    lnL_n, lnL_w = _lnL(rr_n), _lnL(rr_w)
    print("W4 narrow bank: |a_list|=%d, dropped %s, truncated=%s"
          % (len(a_list), missing, rr_n[4]['harmonics_truncated']))
    print("W4 lnL narrow=%.9f  widened=%.9f  dlnL=%+.6e nats" % (lnL_n, lnL_w, lnL_w - lnL_n))
    assert missing, "widen_harmonics=False did not truncate -- W3 cannot fail, so it proves nothing"
    assert rr_n[4]['harmonics_truncated'] is True, "truncation was not recorded in meta"
    assert not msgs_n, "widen_harmonics=False should not warn about widening it did not do"
    assert abs(lnL_w - lnL_n) > DLNL_MIN, \
        "truncation moved lnL by only %.3e nats -- W4 is not exercising the bug" % abs(lnL_w - lnL_n)


def test_W5_jax_packer_loses_nothing():
    try:
        import jax  # noqa: F401
    except ImportError:
        print("W5 SKIPPED (no jax)")
        return
    import RIFT.likelihood.jax_ile.response_slowrot as jrs
    rr, _ = _bank(True)
    a_list = [(int(p), int(n)) for (p, n) in rr[4]['a_list']]
    lald = lalsim.DetectorPrefixToLALDetector(DET)
    gmst = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(event_time))))
    cdict = jrs.rotation_coefficients_dict(
        np.asarray(lald.response), np.asarray(lald.location),
        np.array([extr.phi]), np.array([extr.theta]), np.array([extr.psi]), gmst, P_MAX)
    missing = sorted(set((int(p), int(n)) for (p, n) in cdict) - set(a_list))
    print("W5 jax coefficient keys with no band in a_list: %s" % (missing,))
    assert not missing, "the JAX packer would silently drop %s" % (missing,)

    # ...and go through the real packer, which is where the drop would happen.
    from RIFT.likelihood.jax_ile.banded import build_rotation_data
    tvals = np.arange(200) * deltaT - 0.01
    for widen, want_warn in ((True, False), (False, True)):
        b = _bank(widen)[0]
        lk, ra, cu, cv, ep = flwr.pack_rotation_arrays(b[4], b[3], b[1], b[2])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = build_rotation_data(b[4], lk, ra, cu, cv, ep, deltaT, tvals)
        got = [str(c.message) for c in caught
               if issubclass(c.category, RuntimeWarning) and 'build_rotation_data' in str(c.message)]
        print("W5 jax packer, widen=%s: A=%d bands, warned=%s"
              % (widen, len(data.band['a_list']), bool(got)))
        assert len(data.band['a_list']) == len(b[4]['a_list'])
        assert bool(got) is want_warn, \
            "build_rotation_data warning: got %r, wanted %r (widen=%s)" % (bool(got), want_warn, widen)


def test_W6_maintained_noloop_path():
    """The fix changes what the NoLoop is handed; run it, and make the truncated bank
    announce itself at the packer instead of evaluating a short model in silence."""
    rr_w, _ = _bank(True)
    rr_n, _ = _bank(False)
    lnL_w, msgs_w = _lnL_noloop(rr_w)
    lnL_n, msgs_n = _lnL_noloop(rr_n)
    print("W6 NoLoop |a_list| widened=%d narrow=%d" % (len(rr_w[4]['a_list']), len(rr_n[4]['a_list'])))
    print("W6 NoLoop lnL widened=%.9f  narrow=%.9f  dlnL=%+.6e nats" % (lnL_w, lnL_n, lnL_w - lnL_n))
    print("W6 packer warning on the truncated bank: %s" % (msgs_n[0] if msgs_n else "NONE"))
    assert np.isfinite(lnL_w), "the widened bank does not evaluate through the NoLoop: %r" % lnL_w
    assert not msgs_w, "the widened bank must not warn at the packer: %s" % msgs_w
    assert abs(lnL_w - lnL_n) > DLNL_MIN, \
        "truncation is invisible to the MAINTAINED path (dlnL=%.3e)" % abs(lnL_w - lnL_n)
    assert msgs_n, "pack_rotation_arrays accepted a truncated bank silently"
    assert 'TRUNCATED' in msgs_n[0], "the packer warning does not say the model is truncated: %s" % msgs_n[0]


if __name__ == "__main__":
    test_W0_antenna_and_delay_half_widths()
    test_W1_required_width_is_measured()
    test_W2_narrow_request_is_widened_and_says_so()
    test_W3_widened_bank_drops_no_coefficient()
    test_W4_control_narrow_bank_really_does_truncate()
    test_W5_jax_packer_loses_nothing()
    test_W6_maintained_noloop_path()
    print("ALL SLOWROT HARMONIC-WIDTH CHECKS PASSED")
