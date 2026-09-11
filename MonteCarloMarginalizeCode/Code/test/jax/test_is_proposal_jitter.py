#!/usr/bin/env python
"""Issue #227: a Gaussian importance proposal must be SCORED under the matrix it
was DRAWN from.

WHAT WENT WRONG.  Seven sites drew ``theta ~ N(mu, cov + 1e-12 I)`` by Cholesky
and then evaluated ``logq`` under bare ``cov``.  The regularizer is ABSOLUTE, so
it is negligible only while ``cov`` is O(1).  MEASURED on the real S250114ax
H1/L1 point this was found on (rho ~ 49): ``--mode map`` reports a Fisher
diagonal of [5.5e5 9.8e4 5.3e5 5.3e5 1.6e4 1.6], i.e. angular scales ~1e-3 rad
(the issue quotes ~1e-5; 1e-3 is what the Fisher there actually gives, and the
argument does not need the smaller number).  ``run_laplace_is``'s adaptation then
contracts far below that: ``cov`` reached 3e-21, the Mahalanobis term became
``(1e-6/sqrt(3e-21))**2 ~ 3e8`` per dimension, and ``--mode laplace-is`` -- the
driver's DEFAULT -- returned ``lnZ = 5.85e9`` with ``neff = 1.0`` and exited 0.

WHY THESE TESTS LOOK LIKE THIS.  A unit test of the jitter helper cannot see the
defect: the defect is not in either matrix, it is in the two of them being
different at the call site.  So the behavioural tests below drive the REAL
``run_laplace_is`` on a synthetic likelihood (pure numpy -- no frames, no PSDs,
no jax evaluation), and a separate AST test gates the CLASS across both files,
because a fix at one of seven sites is not a fix.

FLOATING POINT.  Nothing in this file is precision-sensitive: the synthetic
likelihood, the proposal algebra and the reference integral are all numpy
float64 regardless of jax's x64 flag, and the 0.1-nat tolerance on the reference
comparison is far above float32 resolution.  x64 is still requested below so that running
this file FIRST in a session cannot change what any later file sees.
"""

import ast
import importlib.machinery
import importlib.util
import io
import contextlib
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DRIVER = os.path.join(_CODE, "bin", "integrate_likelihood_extrinsic_jax")
_SAMPLERS = os.path.join(_CODE, "RIFT", "likelihood", "jax_ile", "samplers.py")


def _driver():
    loader = importlib.machinery.SourceFileLoader("_isj_drv", _DRIVER)
    spec = importlib.util.spec_from_loader("_isj_drv", loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = "_isj_drv"          # keep the __main__ guard from firing
    loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Synthetic likelihood: an isotropic Gaussian in the five angles.
#
# ``sig`` is the whole experiment.  sig ~ 1e-5 is the production regime this
# issue was found in (rho ~ 49) and the regime no prior pilot can resolve;
# sig ~ 0.15 is a posterior a prior pilot CAN find, and is where the estimator
# is supposed to work and must keep working.
# --------------------------------------------------------------------------
_MU = np.array([1.3, 0.2, 1.0, 1.1, 3.0])


class _GaussianAngles(object):
    def __init__(self, sig, peak):
        self.sig = float(sig)
        self.peak = float(peak)

    def log_likelihood(self, *cols):
        th = np.stack([np.asarray(c) for c in cols], axis=-1)
        d = (th[..., :5] - _MU[None, :]) / self.sig
        return self.peak - 0.5 * np.sum(d * d, axis=-1)


def _run(sig, peak, n_max=120000, seed=3):
    """Drive the real ``run_laplace_is``; return its outputs and its log."""
    mod = _driver()
    optp = mod.build_parser()
    opts, _ = optp.parse_args(["--inj-mode", "--n-max", str(n_max),
                               "--seed", str(seed)])
    rng = np.random.default_rng(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = mod.run_laplace_is(_GaussianAngles(sig, peak), opts, rng, 5, False)
    return mod, opts, out, buf.getvalue()


def _reference_logZ(mod, opts, sig, peak, n=2000000, seed=99):
    """Independent estimate of ln Z = ln int L(theta) p(theta) dtheta.

    Deliberately shares NO machinery with the estimator under test: the
    proposal is written out here, drawn directly, and centred on the known peak
    at twice its width, which makes this a near-perfect importance proposal
    (ESS ~ 0.25 n).  It uses the driver's ``log_prior`` only because both
    estimators must integrate against the SAME prior to be comparable.
    """
    rng = np.random.default_rng(seed)
    s = 2.0 * sig
    th = _MU[None, :] + s * rng.standard_normal((n, 5))
    lnL = _GaussianAngles(sig, peak).log_likelihood(*[th[:, i] for i in range(5)])
    logp = mod.log_prior(th, opts, False)
    logq = (-0.5 * np.sum(((th - _MU[None, :]) / s) ** 2, axis=1)
            - 0.5 * 5 * np.log(2 * np.pi * s * s))
    lw = lnL + logp - logq
    lw = lw[np.isfinite(lw)]
    m = lw.max()
    w = np.exp(lw - m)
    return float(m + np.log(w.mean()))


###
### 1. The helper's contract
###

def test_regularize_cov_regularizer_is_relative_to_the_covariance():
    """The whole defect is an absolute epsilon meeting a 1e-21 covariance."""
    from RIFT.likelihood.jax_ile.samplers import regularize_cov
    base = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    for scale in (1.0, 1e-10, 1e-20, 1e-30):
        cov = scale * base
        out = regularize_cov(cov)
        added = np.diag(out - cov)
        assert np.allclose(added, added[0])                    # isotropic
        # the nudge is a fixed FRACTION of the covariance scale, never a floor
        assert added[0] == pytest.approx(1e-12 * np.trace(cov) / 5, rel=1e-12)
        assert 0 < added[0] < 1e-11 * float(np.max(np.diag(cov)))


def test_regularize_cov_is_scale_equivariant():
    """f(a C) == a f(C): the property an absolute jitter does not have, and the
    reason the proposal can no longer be swamped by its own regularizer."""
    from RIFT.likelihood.jax_ile.samplers import regularize_cov
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    C = A @ A.T
    for a in (1e-8, 1.0, 1e8):
        assert np.allclose(regularize_cov(a * C), a * regularize_cov(C),
                           rtol=1e-12, atol=0.0)


def test_regularize_cov_still_conditions_a_degenerate_covariance():
    """trace == 0 has no scale to be relative to; it must still come back
    Cholesky-able rather than raising inside a sampler."""
    from RIFT.likelihood.jax_ile.samplers import regularize_cov
    out = regularize_cov(np.zeros((4, 4)))
    np.linalg.cholesky(out)                                    # must not raise
    assert np.all(np.diag(out) > 0)


###
### 2. The CLASS gate.  Seven sites shared the pattern; a fix at one is not a fix.
###

def _cholesky_calls_with_an_inline_identity(path):
    """Every ``np.linalg.cholesky(X)`` in ``path`` whose argument builds an
    identity inline -- i.e. regularizes a matrix at the DRAW while some other
    expression is what gets scored.  Returns (lineno, source) pairs."""
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    lines = src.splitlines()
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "cholesky"):
            continue
        arg = node.args[0] if node.args else None
        if arg is None:
            continue
        for sub in ast.walk(arg):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in ("eye", "identity")):
                bad.append((node.lineno, lines[node.lineno - 1].strip()))
                break
    return bad


@pytest.mark.parametrize("path", [_DRIVER, _SAMPLERS])
def test_no_cholesky_regularizes_a_matrix_inline(path):
    """Structural gate on the #227 pattern.

    HONEST SCOPE: this is a shape check, not a correctness proof -- it cannot
    see a caller that passes two different *named* matrices.  It exists because
    the behavioural test below reaches only ONE of the seven sites (the default
    mode); the other six are inside flowMC / NUTS / SMC paths that need numpyro,
    flowMC and a real likelihood to run.  Reverting ANY of the seven to
    ``cholesky(cov + 1e-12*np.eye(d))`` fails this test.

    The fix is to call ``regularize_cov(cov)`` once and hand the SAME object to
    the Cholesky and to _gaussian_logq/_mixture_logq.
    """
    bad = _cholesky_calls_with_an_inline_identity(path)
    assert not bad, (
        "%s regularizes a covariance inside np.linalg.cholesky(...):\n%s\n"
        "Call regularize_cov(cov) once and score under the SAME matrix (#227)."
        % (os.path.basename(path),
           "\n".join("  line %d: %s" % b for b in bad)))


def test_the_class_gate_can_actually_fail():
    """A structural test that never fails on anything is not coverage.  Prove
    the detector fires on the exact pre-fix source line."""
    import tempfile
    src = ("import numpy as np\n"
           "def f(cov, dim):\n"
           "    Lc = np.linalg.cholesky(cov + 1e-12 * np.eye(dim))\n"
           "    return Lc\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        found = _cholesky_calls_with_an_inline_identity(tmp)
        assert len(found) == 1 and found[0][0] == 3
    finally:
        os.unlink(tmp)


def test_regularize_cov_is_the_single_definition_both_files_use():
    """The driver must not grow its own copy: the bug was one rule written
    twice (drawn one way, scored another), and a second helper is the same
    mistake one level up."""
    with open(_DRIVER) as f:
        drv = f.read()
    assert "from RIFT.likelihood.jax_ile.samplers import regularize_cov" in drv
    assert "def regularize_cov" not in drv


###
### 3. Behaviour: the default mode on a posterior no prior pilot can resolve.
###    This is the configuration that returned 5.85e9 on real data.
###

_NARROW = dict(sig=2e-5, peak=1539.0)


def test_laplace_is_never_reports_evidence_above_the_peak_likelihood():
    """The #227 regression assertion, stated as the issue asks for it.

    For a NORMALIZED prior, Z = E_prior[L] <= max L, so ln Z <= max lnL always.
    The shipped code returned ln Z = 5.85e9 against a peak lnL of 1808 on real
    S250114ax data, and 4.9e9 - 5.6e9 here on every seed tried.  Reproduces in
    seconds with no frames, because the defect is in the proposal algebra rather
    than in the physics.
    """
    _mod, _opts, out, _log = _run(n_max=200000, seed=11, **_NARROW)
    logZ, _sig, _neff, _n, _theta, lnL, _logw = out
    max_lnL = float(np.nanmax(lnL[np.isfinite(lnL)]))
    assert np.isnan(logZ) or logZ <= max_lnL + 5.0, (
        "ln Z = %r exceeds max lnL = %r; the weights were computed against a "
        "distribution that was never sampled (#227)" % (logZ, max_lnL))


@pytest.mark.parametrize("seed", [3, 5, 11, 12])
def test_a_proposal_that_walked_off_the_peak_is_never_published(seed):
    """Fixing the jitter is NOT sufficient, and this is the test that says so.

    With the draw and the density matched, the same configuration returns a
    SELF-CONSISTENT number computed from a proposal that never found the peak:
    a plausible wrong answer in place of an implausible one.

    THE ASSERTION IS A PROPERTY, NOT AN OUTCOME, and that is a correction from
    external review.  This test used to assert ``isnan`` on every seed, which
    quietly made a SUCCESS into a test failure: had the adaptation ever recovered
    the peak here, the suite would have reported a regression.  A test that can
    only pass while the code fails cannot witness the guard being too aggressive,
    which is precisely the risk under review.  So what is pinned is the property
    that matters -- *no inaccurate number is ever published* -- with a recovered,
    accurate answer explicitly allowed through.
    """
    mod, opts, out, log = _run(n_max=120000, seed=seed, **_NARROW)
    logZ = out[0]
    if np.isnan(logZ):
        assert "Markov floor" in log
        return
    ref = _reference_logZ(mod, opts, **_NARROW)
    assert abs(logZ - ref) < 0.5, (
        "published lnZ = %r from a collapsed proposal (reference %r)" % (logZ, ref))


def test_the_pilot_floor_is_a_markov_bound_not_the_raw_estimate():
    """External review, P1: the pilot is UNBIASED for Z, not a bound on it.

    For a mode of prior mass m a single lucky draw gives ~L_max/n_pilot against a
    truth of ~L_max*m, overshooting by 1/(n_pilot*m).  MEASURED, not argued: at a
    synthetic width of 0.05 rad (n_pilot*m = 1.6e-3) the pilot ran up to +5.46
    nats ABOVE the truth, P = 1.1e-3 over 900 seeds -- so a raw-estimate floor
    set at 5 nats rejects correct answers at about that rate.

    Markov needs only non-negativity and unbiasedness: P(Zhat >= Z/rate) <= rate,
    so ln Zhat + ln rate is a lower confidence bound at level 1 - rate.  The
    threshold is a chosen false-positive rate, and it is distribution-free -- it
    does not assume the pilot resolved anything, which matters because the
    pilot's ESS is ~1 in every regime where this guard does any work.
    """
    mod = _driver()
    for rate in (3.4e-4, 1e-2, 0.5):
        for lz in (-3.0, 0.0, 1234.5):
            assert mod.prior_pilot_floor(lz, rate) == pytest.approx(
                lz + np.log(rate), rel=0, abs=1e-12)
            assert mod.prior_pilot_floor(lz, rate) < lz     # always a DISCOUNT
    # a smaller admitted false-positive rate must push the floor DOWN, never up
    assert mod.prior_pilot_floor(0.0, 1e-6) < mod.prior_pilot_floor(0.0, 1e-2)
    # the shipped rate is the one the operating curve was read at
    assert mod.PILOT_FLOOR_FP_RATE == pytest.approx(np.exp(-8.0), rel=0.02)
    # a pilot that estimated nothing must not manufacture a floor
    assert mod.prior_pilot_floor(np.nan) == -np.inf
    assert mod.prior_pilot_floor(-np.inf) == -np.inf


def test_an_inflated_pilot_does_not_reject_an_accurate_high_ess_answer():
    """The regression case external review asked for, and it is a REAL run.

    Reviewer's scenario: a sparse pilot hit both inflates the pilot's estimate
    AND seeds a good proposal, so the correct adapted answer is compared against
    a reference that its own lucky draw pushed up.  Found by sweeping 5400 runs
    for the shape -- pilot ABOVE truth, adapted accurate, high ESS.  At
    sig = 0.15, seed = 885 the pilot lands +1.35 nats above the reference while
    the adapted estimate is right to 0.001 nats at neff ~ 1.4e4.

    The exhaustive sweep is the honest part: over those 5400 runs the largest
    pilot-minus-adapted gap on an accurate run was +1.654 nats, so this case does
    NOT reach the shipped threshold and no false positive was ever observed.
    What this pins is the margin -- lower the threshold under ~1.7 nats, or go
    back to comparing against the raw pilot at a 5-nat cut without the Markov
    discount, and a correct high-ESS answer starts being thrown away.
    """
    mod, opts, out, log = _run(sig=0.15, peak=100.0, n_max=120000, seed=885)
    logZ, _s, neff = out[0], out[1], out[2]
    ref = _reference_logZ(mod, opts, sig=0.15, peak=100.0)
    assert neff > 1000.0
    assert abs(logZ - ref) < 0.1, "the case no longer has the reviewer's shape"
    assert not np.isnan(logZ), "an accurate, high-ESS answer was rejected"
    assert "Markov floor" not in log


###
### 4. The regime the estimator DOES work in must be untouched.
###

_HEALTHY = [(0.30, 20.0), (0.20, 20.0), (0.15, 20.0)]


@pytest.mark.parametrize("sig,peak", _HEALTHY)
@pytest.mark.parametrize("seed", [3, 5, 12])
def test_laplace_is_matches_an_independent_reference_where_it_works(sig, peak, seed):
    """A posterior a prior pilot CAN resolve.  Two jobs:

    1.  THE FIX CHANGED NOTHING HERE.  The proposal covariance is ~1e-2, ten
        orders above the old absolute jitter, so old and new differ in the
        eleventh significant figure: on (sig=0.15, seed=3) the parent commit
        gives 8.7451703051592702 and this one 8.7451703051683118.  All nine
        cases below print identically to 5 dp on both sides.
    2.  THE GUARD IS NOT TRIGGER-HAPPY.  An earlier version of this guard keyed
        on the pilot's ESS, and ESS turned out to be a poor predictor: it fired
        on (sig=0.15, seed=4) -- a run whose final neff was 17843 and whose
        answer was right to 0.003 nats -- and made the answer 1.5 nats WORSE.
        Several seeds and widths are swept because a single seed hid that.
    """
    mod, opts, out, log = _run(sig=sig, peak=peak, n_max=120000, seed=seed)
    logZ, _sig, neff, _n, _theta, _lnL, _logw = out
    ref = _reference_logZ(mod, opts, sig=sig, peak=peak)
    assert neff > 1000.0
    assert abs(logZ - ref) < 0.1, "ln Z = %.5f vs reference %.5f" % (logZ, ref)
    assert "laplace-is]" not in log, "the guard fired on a healthy run: %s" % log


def test_the_evidence_sanity_rule_is_wired_into_both_driver_estimators():
    """WIRING, honestly labelled: a call-site check, not a behavioural one.

    ``_finalize_evidence`` (ln Z <= max lnL, neff >= 1.5) is the library
    samplers' rule; these two driver estimators applied NO rule at all, which is
    why #227's 5.8e9 was reported as a success.  With the jitter fixed there is
    no longer a synthetic that reaches it -- the pilot comparison above catches
    the collapse first -- so what is testable is that the belt is still attached
    to the braces.
    """
    with open(_DRIVER) as f:
        tree = ast.parse(f.read())
    for name in ("run_laplace_is", "run_nuts"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        calls = {n.func.id for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_finalize_evidence" in calls, "%s does not finalize its evidence" % name


###
### 5. A non-finite evidence must FAIL the event, not be published as one.
###

def test_require_finite_evidence_passes_a_number_and_refuses_a_nan():
    mod = _driver()
    mod.require_finite_evidence(1462.36, 4.5, "laplace-is")      # must not raise
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(RuntimeError) as ei:
            mod.require_finite_evidence(bad, 1.0, "laplace-is")
        assert "laplace-is" in str(ei.value)


def test_analyze_one_refuses_before_it_writes_either_artifact():
    """WIRING, not presence.  On real S250114ax data the pre-fix driver wrote
    ``lnL = 5.848741051595e+09`` into ``out_0_.dat`` and exited 0; the estimator
    now returns nan there, and a nan row published to a CIP fit is no better.
    So the refusal has to come BEFORE write_samples/write_dat -- checked by
    statement order inside ``analyze_one``, because a call that runs after the
    files exist is the same defect with a tidier log.
    """
    with open(_DRIVER) as f:
        tree = ast.parse(f.read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "analyze_one")
    seen = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            seen.setdefault(node.func.id, node.lineno)
    for name in ("require_finite_evidence", "write_samples", "write_dat"):
        assert name in seen, "analyze_one never calls %s" % name
    assert seen["require_finite_evidence"] < seen["write_samples"]
    assert seen["require_finite_evidence"] < seen["write_dat"]


###
### 6. The pilot and the adapted estimator must integrate the SAME quantity.
###

def test_pilot_and_explicit_weights_share_one_normalization_under_limit_distance():
    """THE MATH, not the wiring -- the wiring is the AST test below, and this
    test is deliberately unable to see a caller that drops the term.  (It was
    written as a behavioural test first, and a mutation that deleted the term
    from run_laplace_is left it green: it reimplements the two weight
    expressions rather than reaching them, which is the same "tests the helper,
    not the call site" mistake the #227 defect itself is an instance of.)

    What it does establish is why the term has to be there at all: the 5-nat
    collapse guard compares two DIFFERENT estimators, and it means nothing
    unless both are on the same normalization.

    They are built differently on purpose.  The adapted loop forms
    ``ln w = lnL + ln p - ln q`` explicitly, and ``log_prior`` is normalized over
    the physical ``[d_min,d_max]``; the prior pilot draws from ``sample_prior``,
    which samples the prior RESTRICTED to the ``--limit-distance`` box, so its
    weights need ``-log_distance_box_correction`` to land on that same scale.
    Drop that term -- which the docstring of ``log_distance_box_correction`` used
    to tell a reader to do, because it named ``run_laplace_is`` as an estimator
    that must not subtract -- and the two differ by a CONSTANT.  A constant
    offset does not make the guard noisy, it makes it wrong in one direction:
    the box here is a factor of 17.86 in prior mass, i.e. 2.88 nats, and a wider
    limit silently walks past the 5-nat threshold and disables the guard
    entirely while every existing test still passes.

    Taking L == 1 makes both estimators exactly computable: Z = int_box p_full
    = the box's prior mass fraction, so the assertion is against a number
    derived on paper rather than against the other estimator.
    """
    mod = _driver()
    optp = mod.build_parser()
    opts, _ = optp.parse_args(["--inj-mode", "--d-min", "10.0", "--d-max", "1000.0",
                               "--limit-distance", "200.0,400.0"])
    lo, hi = mod.resolve_distance_limit(opts)
    assert (lo, hi) == (200.0, 400.0), "the box did not narrow; test is vacuous"

    mass = (hi ** 3 - lo ** 3) / (opts.d_max ** 3 - opts.d_min ** 3)
    assert 0.0 < mass < 0.1                      # a box small enough to matter

    rng = np.random.default_rng(7)
    n = 400000
    theta_p, _ = mod.sample_prior(n, opts, rng, True)

    # (a) the PILOT form: proposal == restricted prior, lnL == 0
    lw_pilot = np.zeros(n) - mod.log_distance_box_correction(opts, True)
    Z_pilot = np.exp(lw_pilot).mean()

    # (b) the ADAPTED form: ln w = lnL + ln p - ln q, written out against a
    #     uniform-in-d proposal over the same box (q is known exactly here).
    #     The angles keep their prior draw, so q's angular factor IS the driver's
    #     own angles-only log_prior -- taken from the code rather than re-derived
    #     here, since a hand-written copy of a density is the mistake this whole
    #     file is about.  Only the distance proposal differs (uniform on the box).
    th = np.array(theta_p, dtype=float, copy=True)
    th[:, 5] = rng.uniform(lo, hi, size=n)
    logq = mod.log_prior(th, opts, False) - np.log(hi - lo)
    lw_adapt = mod.log_prior(th, opts, True) - logq
    Z_adapt = np.exp(lw_adapt[np.isfinite(lw_adapt)]).mean()

    assert Z_pilot == pytest.approx(mass, rel=1e-12), (
        "pilot weights do not carry the full-range normalization: %r vs %r"
        % (Z_pilot, mass))
    assert Z_adapt == pytest.approx(mass, rel=0.02), (
        "explicit lnL+lnp-lnq weights are on a different scale: %r vs %r"
        % (Z_adapt, mass))
    # and therefore on the same scale as each other, which is the guard's premise
    assert abs(np.log(Z_pilot) - np.log(Z_adapt)) < 0.05


def test_the_pilot_normalization_test_would_catch_the_dropped_correction():
    """The mutation the test above exists to stop, stated so it is visible: with
    the box correction dropped the pilot is low by ln(1/mass) -- 2.7 nats here,
    and larger for a wider box, which is how it would slip past a 5-nat guard."""
    mod = _driver()
    optp = mod.build_parser()
    opts, _ = optp.parse_args(["--inj-mode", "--d-min", "10.0", "--d-max", "1000.0",
                               "--limit-distance", "200.0,400.0"])
    corr = mod.log_distance_box_correction(opts, True)
    assert corr > 0.0
    assert corr == pytest.approx(2.8824, abs=0.01)


def test_the_pilot_reference_is_wired_to_the_box_correction():
    """WIRING, and the test that actually holds run_laplace_is to the maths above.

    The numerical test cannot: it writes the two weight expressions out itself,
    so deleting ``- log_distance_box_correction(...)`` from run_laplace_is left
    it green.  This one reads the call site.  ``evidence_from_logweights`` is
    called twice in the file -- once in run_prior_mc, once for this pilot -- and
    BOTH are "proposal == prior" estimators, so both arguments must carry the
    correction.  The adapted loop must not, and that is asserted too, because
    the failure mode is symmetric: adding the term where ln w is already
    explicit breaks the comparison just as thoroughly as dropping it here.
    """
    with open(_DRIVER) as f:
        src = f.read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_laplace_is")

    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "evidence_from_logweights"]
    # TWO, and they are the two halves of the comparison: the pilot (raw prior
    # draws, needs the correction) and the adapted estimate (ln w already
    # explicit, must not have it).  Pinning the count is what makes "the pilot
    # is the one with the correction" a statement rather than a search.
    assert len(calls) == 2, (
        "expected exactly two evidence estimates in run_laplace_is (prior pilot "
        "and adapted), found %d" % len(calls))
    pilot = [c for c in calls if "log_distance_box_correction" in ast.dump(c.args[0])]
    assert len(pilot) == 1, (
        "the prior pilot's weights do not subtract log_distance_box_correction, so "
        "under --limit-distance it sits ln(1/box mass) BELOW the adapted estimate it "
        "is compared against -- 2.88 nats for the box in the test above, and a wider "
        "limit walks straight past the 5-nat guard and disables it (#227).")
    arg = pilot[0].args[0]
    assert isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Sub), (
        "the correction must be SUBTRACTED from the pilot weights, not added")
    other = [c for c in calls if c is not pilot[0]]
    assert isinstance(other[0].args[0], ast.Name) and other[0].args[0].id == "logw", (
        "the adapted estimate should be taken on the explicit ln w array")

    # ... and the adapted loop, which forms ln w explicitly, must NOT correct again.
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "logw"):
            assert "log_distance_box_correction" not in ast.dump(node.value), (
                "the adapted loop already forms lnL + ln p - ln q against a "
                "full-range-normalized log_prior; correcting again double-counts")
