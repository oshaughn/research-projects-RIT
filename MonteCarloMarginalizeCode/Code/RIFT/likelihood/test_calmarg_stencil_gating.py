# RIFT-CI-GATE: q-window-stencil
# ^ registers this file with .travis/test-q-window-stencil.sh, run by ci.yml's
#   q-window-stencil-check job.  Membership lives here, in the test file, so that
#   adding a test needs no edit to any shared list.  Do not reword the line above.
# ---------------------------------------------------------------------------------
# WHY THIS FILE IS IN q-window-stencil-check.  Moved verbatim from the comment block
# above that job's hand-maintained file list in .github/workflows/ci.yml; it lives
# here now so that registering a test needs no edit to a shared file.
#
# test_calmarg_stencil_gating runs its CPU arms without a GPU (its GPU arm is additive),
# so it belongs here: it is what stops cubic/sinc being routed to the fused calibration
# kernel, which is implemented for 'nearest' only.
# ---------------------------------------------------------------------------------
"""
test_calmarg_stencil_gating : the fused calibration-marginalization kernel is implemented
ONLY for time_interp='nearest', and everything else must fall back to the 'loop' path.

Three things are checked.

(a) LIBRARY-LEVEL REFUSAL (executed).
    factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(...,
    cal_method='fused', time_interp='cubic'|'sinc') must raise NotImplementedError, and
    cal_method='fused' with time_interp='nearest' must NOT raise -- and must actually run
    the fused reduction to a finite lnL, not merely survive the guard.  ('sinc' is the new
    stencil; the guard predates it, so the point of the test is that the guard is written
    against 'nearest' rather than against a hard-coded list of the stencils that existed
    when it was written.)

(b) DRIVER-LEVEL GATING (executed, on the driver's own source expressions).
    bin/integrate_likelihood_extrinsic_batchmode chooses, at three NoLoop call sites,
        cal_method = ('fused' if <cond> and opts._noloop_time_interp == 'nearest' else 'loop')
    and, at two of them,
        cal_distmarg = (cal_distmarg_dict if opts._noloop_time_interp == 'nearest' else None).
    Rather than restate those expressions (which would test nothing), this test PARSES the
    driver with `ast`, extracts the actual keyword-argument expressions from the actual
    call sites, and evaluates them over the full truth table of
    (gate condition) x (stencil in TIME_INTERP_CHOICES).  So the assertion is against the
    code as written, and a later edit that drops the `== 'nearest'` clause fails here.
    The driver is not importable as a module (it is a script that parses argv and builds a
    full ILE state), so its surrounding control flow is NOT executed -- only the gating
    expressions themselves are.

(c) THE STENCIL REALLY TAKES EFFECT THROUGH THE CALMARG PATH (executed).
    n_cal>1 with time_interp='sinc' must give finite lnL, must NOT be bit-identical to the
    'cubic' calmarg result, and the two interpolating stencils must agree with each other
    far better than either agrees with 'nearest' -- which is what their error orders
    predict (nearest is O(h) in the sub-sample offset; cubic is O(h^4) and sinc is
    window-limited, so both sit close to the exact band-limited value and therefore close
    to each other).  A conservative factor of 5 is required; the observed factor is ~40.

Runs on CPU (numpy), so it needs no GPU; the GPU legs of (c) are added when cupy is
available.  The heavy precompute is shared with test_noloop_gpu_stencils.

    OMP_NUM_THREADS=1 PYTHONPATH=<worktree>/MonteCarloMarginalizeCode/Code \
      ~/RIFT_develUWM/bin/python RIFT/likelihood/test_calmarg_stencil_gating.py
"""
from __future__ import print_function, division

import ast
import os

import numpy as np

import RIFT.likelihood.factored_likelihood as fl
from RIFT.likelihood.test_noloop_gpu_stencils import (
    HAVE_GPU, N_CAL, Lmax, T_HALFWIDTH, deltaT, data_dict,
    _setup, _P_vec, _P_vec_to_gpu, _banks_to_gpu,
)

if HAVE_GPU:
    import cupy


def _tvals():
    return np.arange(int(2 * T_HALFWIDTH / deltaT)) * deltaT - T_HALFWIDTH


# ---------------------------------------------------------------------------
# (a) library-level refusal / acceptance
# ---------------------------------------------------------------------------
def test_a_fused_is_nearest_only():
    banks = _setup()['cal']
    lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict = banks
    Pv = _P_vec()
    tvals = _tvals()

    def _call(interp):
        return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, Pv, lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict,
            Lmax=Lmax, xpy=np, n_cal=N_CAL, cal_method='fused', time_interp=interp)

    for interp in ('cubic', 'sinc'):
        raised = None
        try:
            _call(interp)
        except NotImplementedError as e:
            raised = e
        except Exception as e:                                # noqa: BLE001 - want the type
            raise AssertionError(
                "cal_method='fused', time_interp=%r raised %s (%s), expected NotImplementedError"
                % (interp, type(e).__name__, e))
        assert raised is not None, \
            "cal_method='fused', time_interp=%r did NOT raise NotImplementedError" % interp
        print("(a) fused + %-7s -> NotImplementedError: %s" % (interp, raised))

    lnL_fused = np.asarray(_call('nearest'))
    assert np.all(np.isfinite(lnL_fused)), \
        "cal_method='fused', time_interp='nearest' produced non-finite lnL"
    print("(a) fused + nearest  -> ran, lnL finite, max|lnL| = %.6g" % np.max(np.abs(lnL_fused)))

    # The fused kernel and the loop reduction compute the same quantity; they must agree.
    lnL_loop = np.asarray(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict,
        Lmax=Lmax, xpy=np, n_cal=N_CAL, cal_method='loop', time_interp='nearest'))
    d = float(np.max(np.abs(lnL_fused - lnL_loop)))
    tol = 1e-8 + 1e-11 * float(np.max(np.abs(lnL_loop)))
    print("(a) fused vs loop, nearest, n_cal=%d : max|diff| = %.3e (tol %.3e)" % (N_CAL, d, tol))
    assert d < tol, "fused and loop calmarg disagree at nearest: %g >= %g" % (d, tol)


# ---------------------------------------------------------------------------
# (b) driver gating expressions, extracted and evaluated
# ---------------------------------------------------------------------------
_DRIVER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'bin', 'integrate_likelihood_extrinsic_batchmode')

_NOLOOP_NAME = 'DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop'


def _default_cal_method():
    """The library's own default for cal_method, read off the signature (py2/py3 safe)."""
    try:
        import inspect
        return inspect.signature(
            fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop).parameters['cal_method'].default
    except (ImportError, AttributeError):                     # pragma: no cover - py2 fallback
        import inspect
        spec = inspect.getargspec(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop)
        return spec.defaults[spec.args.index('cal_method') - (len(spec.args) - len(spec.defaults))]


_DEFAULT_CAL_METHOD = _default_cal_method()


class _Opts(object):
    def __init__(self, interp):
        self._noloop_time_interp = interp


def _noloop_call_sites():
    """(lineno, {kw: ast expression}) for every NoLoop call in the driver."""
    with open(_DRIVER, 'r') as f:
        tree = ast.parse(f.read(), filename=_DRIVER)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)
        if name != _NOLOOP_NAME:
            continue
        kws = dict((kw.arg, kw.value) for kw in node.keywords if kw.arg)
        sites.append((node.lineno, kws))
    return sorted(sites)


def _eval_expr(expr, ns):
    mod = ast.Expression(body=expr)
    ast.fix_missing_locations(mod)
    return eval(compile(mod, _DRIVER, 'eval'), dict(ns))


def test_b_driver_gating_expressions():
    sites = _noloop_call_sites()
    assert sites, "found no %s call sites in %s" % (_NOLOOP_NAME, _DRIVER)
    print("(b) driver: %s call sites at lines %s"
          % (_NOLOOP_NAME, [ln for ln, _ in sites]))

    sentinel = {'table': 'CAL_DISTMARG_TABLE'}
    conditional = []
    for lineno, kws in sites:
        cm = kws.get('cal_method')
        if cm is None:
            # Site relies on the library default; that default must be the safe one.
            assert _DEFAULT_CAL_METHOD == 'loop', \
                "line %d omits cal_method and the library default is %r, not 'loop'" \
                % (lineno, _DEFAULT_CAL_METHOD)
            print("(b)   line %-5d cal_method omitted -> library default %r -- always safe"
                  % (lineno, _DEFAULT_CAL_METHOD))
            continue
        if isinstance(cm, ast.Str) or (isinstance(cm, ast.Constant) and isinstance(cm.value, str)):
            lit = cm.s if isinstance(cm, ast.Str) else cm.value
            assert lit == 'loop', \
                "line %d passes a LITERAL cal_method=%r; only 'loop' may be hard-wired, " \
                "'fused' must be gated on the stencil" % (lineno, lit)
            print("(b)   line %-5d cal_method literal %r -- always safe" % (lineno, lit))
            continue
        conditional.append((lineno, kws, cm))

    assert conditional, \
        "no conditional cal_method expression found -- the fused/loop gate has disappeared"
    print("(b) conditional cal_method gates at lines %s" % [ln for ln, _, _ in conditional])

    for lineno, kws, cm_expr in conditional:
        cd_expr = kws.get('cal_distmarg')
        for gate in (True, False):
            for interp in fl.TIME_INTERP_CHOICES:
                ns = {'use_fused_calmarg': gate,
                      'cal_distmarg_dict': (sentinel if gate else None),
                      'opts': _Opts(interp)}
                got = _eval_expr(cm_expr, ns)
                want = 'fused' if (gate and interp == 'nearest') else 'loop'
                assert got == want, \
                    "driver line %d: cal_method evaluated to %r for gate=%s, stencil=%r; " \
                    "expected %r" % (lineno, got, gate, interp, want)
                if cd_expr is not None:
                    got_cd = _eval_expr(cd_expr, ns)
                    want_cd = (sentinel if gate else None) if interp == 'nearest' else None
                    assert got_cd == want_cd, \
                        "driver line %d: cal_distmarg evaluated to %r for gate=%s, " \
                        "stencil=%r; expected %r" % (lineno, got_cd, gate, interp, want_cd)
        print("(b)   line %-5d cal_method -> fused iff (gate and nearest); "
              "cal_distmarg %s"
              % (lineno, "gated on nearest" if cd_expr is not None else "(not passed)"))

    # There must be no route by which a non-nearest stencil reaches the fused kernel.
    for lineno, kws, cm_expr in conditional:
        for interp in ('cubic', 'sinc'):
            for gate in (True, False):
                ns = {'use_fused_calmarg': gate,
                      'cal_distmarg_dict': (sentinel if gate else None),
                      'opts': _Opts(interp)}
                assert _eval_expr(cm_expr, ns) == 'loop', \
                    "driver line %d routes stencil %r to the fused kernel" % (lineno, interp)
    print("(b) no driver call site routes 'cubic' or 'sinc' to cal_method='fused'")


# ---------------------------------------------------------------------------
# (c) the stencil takes effect through the calmarg loop path
# ---------------------------------------------------------------------------
def _calmarg_lnL(banks, interp, xpy, Pv, tvals):
    lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict = banks
    if xpy is np:
        out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, Pv, lookupNKDict, rholmArrayDict, ctUArrayDict, ctVArrayDict, epochDict,
            Lmax=Lmax, xpy=np, n_cal=N_CAL, cal_method='loop', time_interp=interp)
        return np.asarray(out)
    rG, uG, vG = _banks_to_gpu(rholmArrayDict, ctUArrayDict, ctVArrayDict)
    out = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        cupy.asarray(tvals), _P_vec_to_gpu(Pv), lookupNKDict, rG, uG, vG, epochDict,
        Lmax=Lmax, xpy=cupy, n_cal=N_CAL, cal_method='loop', time_interp=interp)
    return cupy.asnumpy(out)


def test_c_sinc_takes_effect_through_calmarg():
    banks = _setup()['cal']
    Pv = _P_vec()
    tvals = _tvals()

    backends = [('CPU', np)]
    if HAVE_GPU:
        backends.append(('GPU', cupy))

    for tag, xpy in backends:
        lnL = dict((i, _calmarg_lnL(banks, i, xpy, Pv, tvals)) for i in fl.TIME_INTERP_CHOICES)
        for i in fl.TIME_INTERP_CHOICES:
            assert np.all(np.isfinite(lnL[i])), \
                "(c) %s n_cal=%d loop, %s: non-finite lnL" % (tag, N_CAL, i)
        sep_sc = float(np.max(np.abs(lnL['sinc'] - lnL['cubic'])))
        sep_nc = float(np.max(np.abs(lnL['nearest'] - lnL['cubic'])))
        sep_ns = float(np.max(np.abs(lnL['nearest'] - lnL['sinc'])))
        print("(c) %s n_cal=%d loop: max|lnL| = %.6g ; "
              "max|sinc-cubic| = %.3e ; max|nearest-cubic| = %.3e ; max|nearest-sinc| = %.3e"
              % (tag, N_CAL, np.max(np.abs(lnL['sinc'])), sep_sc, sep_nc, sep_ns))
        assert sep_sc > 0.0, \
            "(c) %s: sinc and cubic are bit-identical through the calmarg path -- the " \
            "stencil is not taking effect" % tag
        assert sep_nc > 0.0 and sep_ns > 0.0, \
            "(c) %s: nearest is bit-identical to an interpolating stencil" % tag
        assert sep_sc < sep_nc / 5.0, \
            "(c) %s: sinc-vs-cubic (%.3e) is not much smaller than nearest-vs-cubic (%.3e); " \
            "the two sub-sample stencils should bracket the exact value far more tightly " \
            "than nearest does" % (tag, sep_sc, sep_nc)


if __name__ == "__main__":
    test_a_fused_is_nearest_only()
    test_b_driver_gating_expressions()
    test_c_sinc_takes_effect_through_calmarg()
    print("CALMARG STENCIL GATING CHECK DONE (GPU legs %s)"
          % ("included" if HAVE_GPU else "skipped: no GPU"))
