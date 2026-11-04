"""Each time-quadrature name is ONE rule on every backend.

Until 2026-10 the vectorized likelihoods chose their fixed-grid rule by backend:
scipy's Simpson on CPU, the vendored ``optimized_gpu_tools.simps`` (old scipy,
``even='avg'``) on GPU.  For even npts (srate 4096 and 8192) the vendored rule is the
trapezoidal rule in the interior, so identical inputs gave lnL differing by up to
ln(3/2) nats between devices (oshaughnessy-junior/research-projects-RIT#204).
``factored_likelihood.time_quadrature_rule`` now picks the rule by name.  This file
pins the rules, the wiring into the shipped likelihood, and the CPU/GPU agreement.
"""
import ast
import inspect
import os
import subprocess
import sys
import warnings

os.environ.setdefault("RIFT_LOWLATENCY", "1")

import numpy as np
import pytest
from scipy import integrate

import lal
import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood import time_marginalization_quadrature as tmq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import gpu_slot_probe                                        # noqa: E402  (sibling helper)

simpson = getattr(integrate, 'simpson', None) or integrate.simps
trapezoid = getattr(integrate, 'trapezoid', None) or integrate.trapz

DELTAT = 1.0 / 4096.0
NPTS = 614                       # marginalization_time_grid(0.075, 1/4096): EVEN
N_BUFFER = 4096
CODE_DIR = os.path.abspath(os.path.join(_HERE, '..'))


# ---- the rules themselves -----------------------------------------------------------

def test_simpson_on_numpy_is_scipy_itself():
    # CPU results are unchanged bit for bit: the selector returns scipy's function.
    assert fl.time_quadrature_rule('simpson', np) is fl.my_simps
    for name in ('bandlimited', 'peak-local'):     # their unrefined rows
        assert fl.time_quadrature_rule(name, np) is fl.my_simps


@pytest.mark.parametrize('n', [2, 3, 4, 5, 8, 9, 613, 614])
@pytest.mark.parametrize('axis', [-1, 0])
def test_host_weight_rules_match_scipy(n, axis):
    rng = np.random.RandomState(n)
    y = rng.uniform(0.0, 3.0, size=(n, 7) if axis == 0 else (7, n))
    got_s = fl._HostWeightRule('simpson', np)(y, dx=DELTAT, axis=axis)
    got_t = fl._HostWeightRule('trapezoid', np)(y, dx=DELTAT, axis=axis)
    np.testing.assert_allclose(got_s, simpson(y, dx=DELTAT, axis=axis), rtol=1e-13, atol=0)
    np.testing.assert_allclose(got_t, trapezoid(y, dx=DELTAT, axis=axis), rtol=1e-13, atol=0)


def test_trapezoid_is_selected_by_name():
    rule = fl.time_quadrature_rule('trapezoid', np)
    y = np.exp(-0.5 * ((np.arange(NPTS) - 300.3) / 0.8) ** 2)[None, :]
    np.testing.assert_allclose(rule(y, dx=DELTAT, axis=-1), trapezoid(y, dx=DELTAT, axis=-1),
                               rtol=1e-13)
    # ... and differs from Simpson at even npts on a sharp peak, so the selection is visible
    assert abs(rule(y, dx=DELTAT)[0] / simpson(y, dx=DELTAT)[0] - 1) > 1e-3


def test_selector_refuses_an_unknown_name():
    for bad in ('Trapezoid', 'log-hermite', 'trapz'):
        with pytest.raises(ValueError):
            fl.time_quadrature_rule(bad, np)


def test_nothing_in_RIFT_reaches_the_vendored_rule():
    # The defect's form was a bare reference at a call site
    # (`optimized_gpu_tools.simps if on_gpu else my_simps`); an aliased import
    # (`from .optimized_gpu_tools import simps as s`) is the same defect.  The CPU
    # suite cannot see either one, so this guard is what CI has.
    rift_dir = os.path.dirname(os.path.dirname(os.path.abspath(fl.__file__)))
    offenders = []
    for root, _dirs, files in os.walk(rift_dir):
        for fname in sorted(files):
            if (not fname.endswith('.py') or fname == 'optimized_gpu_tools.py'
                    or fname.startswith('test_')):
                continue
            path = os.path.join(root, fname)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')        # invalid escapes in parsed sources
                tree = ast.parse(open(path).read())
            for node in ast.walk(tree):
                hit = (isinstance(node, ast.Attribute) and node.attr == 'simps'
                       and 'optimized_gpu_tools' in ast.dump(node.value))
                hit = hit or (isinstance(node, ast.ImportFrom)
                              and (node.module or '').endswith('optimized_gpu_tools')
                              and any(a.name == 'simps' for a in node.names))
                if hit:
                    offenders.append('%s:%d' % (os.path.relpath(path, rift_dir), node.lineno))
    assert not offenders, offenders


# ---- the wiring into the shipped NoLoop likelihood ------------------------------------

def _kappa_buffer():
    ts = np.arange(N_BUFFER) * DELTAT
    ms = np.arange(1, 400)
    T = N_BUFFER * DELTAT
    c = np.exp(-2j * np.pi * ms * (N_BUFFER // 2) * DELTAT / T) / (1.0 + (ms / 120.0) ** 2)
    return np.exp(2j * np.pi * np.outer(ts, ms) / T) @ c


def _inputs(dists_Mpc):
    """One detector, one mode, zero cross terms: lnL_t is the scaled Re kappa(t)."""
    dists_Mpc = np.asarray(dists_Mpc, dtype=float)
    n = dists_Mpc.size
    P = lsu.ChooseWaveformParams()
    P.deltaT = DELTAT
    P.tref = 1000000000.0
    for name in ('phi', 'theta', 'phiref', 'incl', 'psi'):
        setattr(P, name, np.zeros(n))
    P.dist = dists_Mpc * 1e6 * lal.PC_SI
    det = 'H1'
    return (P, {det: np.asarray(_kappa_buffer(), dtype=complex)[None, :]},
            {det: np.array([[2, 2]])}, {det: np.zeros((1, 1), dtype=complex)},
            {det: P.tref - 0.5})


# Peaks of roughly 1700, 21 and 7 nats: sharp to resolved.
DISTS = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0, fl.distMpcRef * 3.0])


def _noloop(args, xpy=np, **kw):
    P, rholms, lookupNK, ct, epochs = args
    tvals = fl.marginalization_time_grid(0.075, DELTAT)
    assert len(tvals) == NPTS
    return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, lookupNK, rholms, ct, ct, epochs, Lmax=2, xpy=xpy, **kw)


def _reference(lnL_t, rule):
    m = lnL_t.max(axis=-1, keepdims=True)
    return m[:, 0] + np.log(rule(np.exp(lnL_t - m), dx=DELTAT, axis=-1))


@pytest.mark.parametrize('name, rule', [('simpson', simpson), ('trapezoid', trapezoid)])
def test_noloop_integrates_with_the_named_rule(name, rule):
    args = _inputs(DISTS)
    lnL_t = _noloop(args, return_lnLt=True)
    np.testing.assert_allclose(_noloop(args, time_quadrature=name), _reference(lnL_t, rule),
                               rtol=0, atol=1e-10)


def test_noloop_trapezoid_and_simpson_differ_on_the_sharp_row():
    args = _inputs(DISTS)
    d = _noloop(args, time_quadrature='trapezoid') - _noloop(args, time_quadrature='simpson')
    assert abs(d[0]) > 0.1          # under-resolved peak: the rules disagree visibly
    assert abs(d[-1]) < 1e-3        # resolved peak: they agree


def test_rotation_and_freqresponse_take_their_rule_from_the_selector():
    from RIFT.likelihood import factored_likelihood_freqresponse as flf
    from RIFT.likelihood import factored_likelihood_with_rotation as flr
    for mod in (flf, flr):
        calls = [n for n in ast.walk(ast.parse(inspect.getsource(mod)))
                 if isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'time_quadrature_rule']
        assert calls, mod.__name__


# ---- CPU and GPU agree --------------------------------------------------------------

@pytest.mark.parametrize('name', ['simpson', 'trapezoid'])
def test_cpu_and_gpu_agree_at_even_npts(name):
    """The defect itself: identical inputs, same answer on both devices.

    Before the fix the sharp row differed by 0.405 nats under 'simpson' (#204).
    """
    cupy = gpu_slot_probe.cupy_or_skip(require_rift_backend=True)
    import copy
    args = _inputs(DISTS)
    P, rholms, lookupNK, ct, epochs = args
    Pg = copy.deepcopy(P)
    for attr in ('phi', 'theta', 'phiref', 'incl', 'psi', 'dist'):
        setattr(Pg, attr, cupy.asarray(np.asarray(getattr(P, attr))))
    gargs = (Pg, {k: cupy.asarray(v) for k, v in rholms.items()}, lookupNK,
             {k: cupy.asarray(v) for k, v in ct.items()}, epochs)
    cpu = _noloop(args, time_quadrature=name)
    gpu = cupy.asnumpy(cupy.asarray(_noloop(gargs, xpy=cupy, time_quadrature=name)))
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-9)


# ---- refusal where a driver cannot honour the name -------------------------------------

def _jax_time_quad_choices():
    # Read from the source: importing jax_ile.core needs jax, which the CI runner
    # for this gate does not install, and a skip there fails the gate.
    path = os.path.join(CODE_DIR, 'RIFT', 'likelihood', 'jax_ile', 'core.py')
    for node in ast.parse(open(path).read()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', None) == '_TIME_QUAD_CHOICES'):
            return set(ast.literal_eval(node.value))
    raise AssertionError('_TIME_QUAD_CHOICES not found as a literal in %s' % path)


def test_noloop_only_list_matches_the_jax_driver():
    assert set(tmq.NOLOOP_ONLY_TIME_QUADRATURES) == (
        set(tmq.TIME_QUADRATURE_CHOICES) - _jax_time_quad_choices())


@pytest.mark.parametrize('args, refused', [
    ('--time-marginalization --time-marginalization-quadrature simpson', False),
    ('--time-marginalization-quadrature trapezoid', True),
    ('--time-marginalization-quadrature peak-local', True),
    ('--time-marginalization-quadrature bandlimited', False),
    # a hand-passed value after the helper's wins, as it does in optparse
    ('--time-marginalization-quadrature simpson --time-marginalization-quadrature trapezoid', True),
    ('--time-marginalization-quadrature trapezoid --time-marginalization-quadrature simpson', False),
    ('--time-marginalization', False),
])
def test_jax_refusal_reads_the_assembled_arguments(args, refused):
    if refused:
        with pytest.raises(ValueError, match='does not implement'):
            tmq.refuse_jax_unsupported_time_quadrature(args, 'args_ile.txt')
    else:
        tmq.refuse_jax_unsupported_time_quadrature(args, 'args_ile.txt')


def test_pseudo_pipe_refuses_trapezoid_with_jax_ile():
    env = dict(os.environ, PYTHONPATH=CODE_DIR + os.pathsep + os.environ.get('PYTHONPATH', ''))
    proc = subprocess.run(
        [sys.executable, os.path.join(CODE_DIR, 'bin', 'util_RIFT_pseudo_pipe.py'),
         '--use-jax-ile', '--internal-ile-time-marginalization-quadrature', 'trapezoid'],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert proc.returncode != 0
    assert 'integrate_likelihood_extrinsic_jax does not implement it' in proc.stdout, proc.stdout[-2000:]
