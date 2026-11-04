"""
test_td_calmarg_guard.py:  Regression tests for the TD-inner-product / calibration-self-term
incompatibility guard.

THE TRAP.  ``PrecomputeLikelihoodTerms`` threads ``IP_factory`` into every inner product it
builds (crossTerms, crossTermsV, ComputeModeIPTimeSeries) EXCEPT the per-realization
calibration self-term block, which hardcodes ``lsu.ComplexIP``.  It has to: the self-term
route forms <C_c h|C_c h> = <h|h>_{|C_c|^2/S} by ASSIGNING ``IP.weights2side`` between
``IP.ip`` calls (CalibrationSelfTermCrossTermsFromBasis), and only a frequency-domain-DIAGONAL
kernel reads that attribute.  ``ToeplitzComplexIP`` computes 2 dT conj(h1_td)^T K h2_td and
never touches ``weights2side`` on its default branch, so a naive thread-through would set a
dead attribute and silently return the UNWEIGHTED base kernel for every realization.

So running ILE with BOTH ``--internal-ile-inner-product td-toeplitz`` AND calibration
marginalization with the self term would evaluate U/V with the TD precision kernel and the
per-realization calibration self terms with the FD-diagonal kernel: two inner products in one
likelihood, no warning.  The fix is to REFUSE the combination early.

Groups:

  (a) PREDICATE -- CalibrationSelfTermInnerProductSupported over the four relevant factory
      kinds (None; a real TDInnerProductFactory; an opted-in factory; an unrelated object),
      and the assertion wrapper's raise/return behavior + message content.

  (b) PRECOMPUTE SEAM -- drive the real PrecomputeLikelihoodTerms and check the guard fires
      for exactly the trapping combination and stays out of the way otherwise.  The guard
      sits BEFORE waveform generation, so these run in milliseconds on a stub waveform
      generator; a sentinel exception from that stub is the "guard did not fire" signal.

  (c) DRIVER -- integrate_likelihood_extrinsic_batchmode carries the same check at startup
      (so a misconfigured DAG node dies before per-point setup), expressed through the SAME
      shared predicate so the two cannot drift apart.

Environment: ~/.conda/envs/junior_rift/bin/python with
PYTHONPATH=<worktree>/MonteCarloMarginalizeCode/Code.  Runnable as
    pytest test_td_calmarg_guard.py
    python  test_td_calmarg_guard.py
"""

import os

import numpy as np

import lal
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
from RIFT.likelihood.td_inner_product import TDInnerProductFactory


# --------------------------------------------------------------------------- fixtures

FS = 4096.0
FNYQ = FS / 2.0
N = 1024                     # two-sided length (tiny: nothing here is ever transformed)
DELTAF = FS / N
DELTAT = 1.0 / FS
LEN1 = N // 2 + 1


class _SentinelReached(Exception):
    """Raised by the stub waveform generator: proof that the guard did NOT fire."""


class _OptedInFactory(object):
    """A factory that claims to implement the |C_c|^2-weighted self term."""
    supports_calibration_self_term = True


def make_data_series():
    """A minimal two-sided COMPLEX16FrequencySeries (only its deltaF/epoch are read)."""
    out = lal.CreateCOMPLEX16FrequencySeries(
        "data", lal.LIGOTimeGPS(0.0), 0.0, DELTAF, lsu.lsu_HertzUnit, N)
    out.data.data = np.zeros(N, dtype=complex)
    return out


def make_psd_series():
    out = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0.0), 0.0, DELTAF, lsu.lsu_HertzUnit, LEN1)
    out.data.data = np.full(LEN1, 1e-46)
    return out


def make_cal_realizations(n_cal=4, seed=42):
    """Per-detector calibration realizations shaped as ILE supplies them: (len2side, n_cal)."""
    rng = np.random.default_rng(seed)
    amp = 1.0 + 0.05 * rng.standard_normal((N, n_cal))
    return amp.astype(complex)


def make_params():
    P = lsu.ChooseWaveformParams()
    P.deltaT = DELTAT
    P.deltaF = DELTAF
    P.fmin = 40.0
    return P


def call_precompute(IP_factory, calibration_self_term=True, with_cal=True,
                    return_calibration_crossterms=True, monkeypatch_generator=True):
    """Drive PrecomputeLikelihoodTerms far enough to exercise the guard.

    The stub waveform generator raises _SentinelReached, which can only happen if execution
    got PAST the guard -- so the caller distinguishes "refused" (ValueError) from "allowed"
    (_SentinelReached) without ever generating a waveform.
    """
    dets = ["H1", "L1"]
    data_dict = {d: make_data_series() for d in dets}
    psd_dict = {d: make_psd_series() for d in dets}
    cal_dict = {d: make_cal_realizations() for d in dets} if with_cal else None

    saved = fl.internal_hlm_generator
    if monkeypatch_generator:
        def _stub(*args, **kwargs):
            raise _SentinelReached("reached waveform generation")
        fl.internal_hlm_generator = _stub
    try:
        return fl.PrecomputeLikelihoodTerms(
            lal.LIGOTimeGPS(0.0), 0.075, make_params(), data_dict, psd_dict, 2, 1000.0,
            quiet=True, verbose=False,
            calibration_realizations=cal_dict,
            return_calibration_crossterms=return_calibration_crossterms,
            calibration_self_term=calibration_self_term,
            IP_factory=IP_factory)
    finally:
        fl.internal_hlm_generator = saved


# --------------------------------------------------------------- (a) predicate + assertion

def test_predicate_matrix():
    """The predicate admits the FD-diagonal default and explicit opt-ins, nothing else."""
    assert fl.CalibrationSelfTermInnerProductSupported(None), \
        "IP_factory=None IS the frequency-domain-diagonal lsu path and must stay supported"

    td = TDInnerProductFactory()
    assert not fl.CalibrationSelfTermInnerProductSupported(td), \
        "TDInnerProductFactory does not implement the |C_c|^2-weighted self term"

    assert fl.CalibrationSelfTermInnerProductSupported(_OptedInFactory()), \
        "a factory advertising supports_calibration_self_term=True must be honored"

    assert not fl.CalibrationSelfTermInnerProductSupported(object()), \
        "an object with no opinion must NOT be assumed capable"


def test_td_factory_does_not_claim_selfterm_support():
    """Pin the opt-out on the real TD factory: flipping it on is a deliberate act.

    If someone later implements the weighted self term for the TD kernel they will set this
    attribute, and this test is the place that says 'and prove it with a numerical test'.
    """
    td = TDInnerProductFactory()
    assert getattr(td, "supports_calibration_self_term", False) is False


def test_assertion_raises_with_actionable_message():
    td = TDInnerProductFactory()
    try:
        fl.AssertCalibrationSelfTermInnerProductSupported(td)
    except ValueError as e:
        msg = str(e)
    else:
        raise AssertionError("AssertCalibrationSelfTermInnerProductSupported did not raise")
    # The message must name BOTH escape hatches -- a bare 'unsupported' would leave the user
    # with a dead run and no next step.
    assert "fd-diagonal" in msg, "message must offer the fd-diagonal inner product"
    assert "--calibration-global-norm" in msg, "message must offer the global-norm route"
    assert "supports_calibration_self_term" in msg, \
        "message must name the opt-in hook for a future TD self-term implementation"


def test_assertion_silent_when_supported():
    """No exception, no return value contract to trip over."""
    assert fl.AssertCalibrationSelfTermInnerProductSupported(None) is None
    assert fl.AssertCalibrationSelfTermInnerProductSupported(_OptedInFactory()) is None


# ----------------------------------------------------------------- (b) the Precompute seam

def test_precompute_refuses_td_plus_calmarg_selfterm():
    """THE REGRESSION: td-toeplitz + calmarg self term must raise, not silently mix kernels."""
    try:
        call_precompute(TDInnerProductFactory(), calibration_self_term=True, with_cal=True)
    except _SentinelReached:
        raise AssertionError(
            "TD inner product + calibration self term was ALLOWED: the U/V cross terms would "
            "use the TD precision kernel while the per-realization calibration self terms use "
            "the FD-diagonal kernel")
    except ValueError as e:
        assert "calibration" in str(e).lower()
    else:
        raise AssertionError("PrecomputeLikelihoodTerms returned without raising")


def test_precompute_guard_precedes_waveform_generation():
    """The guard must fire before any waveform work, so a bad config costs nothing.

    Run with the REAL generator (not the stub): if the guard were downstream of it, this
    would burn a waveform call (and fail with something other than our ValueError).
    """
    try:
        call_precompute(TDInnerProductFactory(), calibration_self_term=True, with_cal=True,
                        monkeypatch_generator=False)
    except ValueError as e:
        assert "inner-product factory" in str(e)
    else:
        raise AssertionError("guard did not fire ahead of waveform generation")


def test_precompute_allows_td_with_global_norm():
    """--calibration-global-norm (calibration_self_term=False) is the supported TD route.

    It never builds the |C_c|^2-weighted cross terms; every realization reuses the
    cal-independent <h|h> from crossTerms, which IS computed with the factory.  One kernel.
    """
    try:
        call_precompute(TDInnerProductFactory(), calibration_self_term=False, with_cal=True)
    except _SentinelReached:
        return
    raise AssertionError("global-norm calmarg on the TD path must NOT be refused")


def test_precompute_allows_td_without_calibration():
    """The population-prior configuration (TD, no calmarg) must be completely untouched."""
    try:
        call_precompute(TDInnerProductFactory(), with_cal=False)
    except _SentinelReached:
        return
    raise AssertionError("plain TD Precompute (no calibration) must not be refused")


def test_precompute_allows_fd_diagonal_calmarg():
    """The production calmarg path (IP_factory=None) must be completely untouched."""
    try:
        call_precompute(None, calibration_self_term=True, with_cal=True)
    except _SentinelReached:
        return
    raise AssertionError("fd-diagonal calmarg with the self term must not be refused")


def test_precompute_allows_td_when_crossterms_not_requested():
    """A six-value-API caller never receives the self terms, so nothing is mixed.

    This mirrors the _build_cal_ct condition exactly: the guard must be gated on the same
    predicate the build is, not on 'calibration realizations were supplied'.
    """
    try:
        call_precompute(TDInnerProductFactory(), calibration_self_term=True, with_cal=True,
                        return_calibration_crossterms=False)
    except _SentinelReached:
        return
    raise AssertionError(
        "the guard must not fire when the self-term cross terms are never built")


def test_precompute_allows_opted_in_factory():
    """A factory that advertises support gets through -- the guard is a capability check."""
    try:
        call_precompute(_OptedInFactory(), calibration_self_term=True, with_cal=True)
    except _SentinelReached:
        return
    raise AssertionError("an opted-in factory must be allowed through the guard")


# --------------------------------------------------------------------------- (c) driver

def _driver_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, os.pardir, "bin", "integrate_likelihood_extrinsic_batchmode")


def test_driver_carries_the_startup_guard():
    """ILE must refuse the combination at STARTUP, via the shared predicate.

    A source check on purpose: the driver is a script with a very expensive import-and-setup
    path, and what matters is that (i) an early guard exists at all and (ii) it delegates to
    factored_likelihood rather than re-deriving the rule (which is how the two would drift).
    """
    path = _driver_path()
    if not os.path.isfile(path):
        import pytest
        pytest.skip("ILE driver not found next to the test tree: %s" % path)
    with open(path) as f:
        src = f.read()
    assert "CalibrationSelfTermInnerProductSupported" in src, \
        ("integrate_likelihood_extrinsic_batchmode must gate td-toeplitz + calmarg on the "
         "shared factored_likelihood predicate, so the driver and library rules cannot drift")
    assert "--calibration-global-norm" in src


# --------------------------------------------------------------------------- script entry

if __name__ == "__main__":
    import time
    tests = [
        ("(a) predicate matrix", test_predicate_matrix),
        ("(a) TD factory opts out", test_td_factory_does_not_claim_selfterm_support),
        ("(a) assertion message is actionable", test_assertion_raises_with_actionable_message),
        ("(a) assertion silent when supported", test_assertion_silent_when_supported),
        ("(b) TD + calmarg self term REFUSED", test_precompute_refuses_td_plus_calmarg_selfterm),
        ("(b) guard precedes waveform gen", test_precompute_guard_precedes_waveform_generation),
        ("(b) TD + global norm allowed", test_precompute_allows_td_with_global_norm),
        ("(b) TD without calmarg allowed", test_precompute_allows_td_without_calibration),
        ("(b) fd-diagonal calmarg allowed", test_precompute_allows_fd_diagonal_calmarg),
        ("(b) crossterms not requested -> allowed", test_precompute_allows_td_when_crossterms_not_requested),
        ("(b) opted-in factory allowed", test_precompute_allows_opted_in_factory),
        ("(c) driver startup guard", test_driver_carries_the_startup_guard),
    ]
    t0 = time.time()
    failures = 0
    for name, fn in tests:
        print("\n=== %s ===" % name)
        try:
            fn()
            print("PASS: %s" % name)
        except AssertionError as e:
            failures += 1
            print("FAIL: %s : %s" % (name, e))
    print("\nTotal runtime: %.2f s ; %d/%d passed"
          % (time.time() - t0, len(tests) - failures, len(tests)))
    raise SystemExit(1 if failures else 0)
