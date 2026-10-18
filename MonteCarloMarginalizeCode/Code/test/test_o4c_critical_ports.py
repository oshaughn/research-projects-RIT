"""Adversarial checks for the O4c #385/#439/#433/#311/#255/#256 port bundle.

The NoLoop oracle is frozen from target commit 28b1f71d9, before these ports.
It is exercised with nonzero data terms, fractional arrival times, phase
marginalization, custom callbacks and continuous-time export.
"""
import ast
import copy
import os
from pathlib import Path

import numpy as np
import pytest

from RIFT.likelihood import factored_likelihood as fl
from test_noloop_accumulator_shapes import _inputs

CODE = Path(__file__).resolve().parents[1]
namespace = dict(vars(fl))
reference_path = Path(__file__).with_name('noloop_o4c_reference.py')
exec(compile(reference_path.read_text(), str(reference_path), 'exec'), namespace)
legacy_noloop = namespace['_DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked']


@pytest.mark.parametrize('driver', ['util_ConstructIntrinsicPosterior_GenericCoordinates.py',
                                    'util_ConstructEOSPosterior.py'])
@pytest.mark.parametrize('queries', [
    [[float('nan')], [float('inf')], [-float('inf')], [1e38], [0.25]],
    [[float('nan')], [float('inf')], [1e38]],
    [],
])
def test_rf_invalid_rows_are_floored_even_if_none_are_valid(driver, queries):
    source = CODE / 'bin' / driver
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'fit_rf')
    scope = dict(np=np, lnL_default_large_negative=-500)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    x = np.linspace(0, 1, 20)[:, None]
    predict = scope['fit_rf'](x, 20-x[:, 0]**2)
    q = np.array(queries).reshape(-1, 1)
    result = predict(q)
    valid = np.isfinite(q[:, 0]) & (np.abs(q[:, 0]) <= 1e37)
    assert result.shape == (len(q),)
    np.testing.assert_array_equal(result[~valid], -500)
    if valid.any():
        assert np.all((result[valid] > 18) & (result[valid] <= 20))


def _nonlinear(kappa, rho):
    return np.logaddexp(0., kappa-0.5*rho)


def _mutating(kappa, rho):
    # An existing public callback could legally write into the formerly dense norm.
    # Time-dependent writes also detect writable CuPy stride-zero races.
    rho += np.linspace(0., 0.1, rho.shape[-1])
    return kappa-0.5*rho


@pytest.mark.parametrize('interp', ['nearest', 'cubic'])
@pytest.mark.parametrize('phase', [False, True])
@pytest.mark.parametrize('quad', ['simpson', 'bandlimited'])
@pytest.mark.parametrize('callback', [fl._factored_lnL_helper, _nonlinear, _mutating])
def test_noloop_matches_frozen_o4c(interp, phase, quad, callback):
    args = _inputs(n_ex=7, npts=16)
    # Vary distances and avoid the zero-extended-Q trap in baseline comparisons.
    args[1].dist *= np.geomspace(0.8, 2., 7)
    saved = copy.deepcopy(args)
    kw = dict(time_interp=interp, phase_marginalization=phase,
              time_quadrature=quad, loglikelihood=callback, xpy=np)
    if quad == 'bandlimited' and callback is _mutating:
        # A time-varying norm violates the existing bandlimited precondition.
        # Preserve the rejection instead of weakening it to accommodate a test.
        with pytest.raises(NotImplementedError, match='time-independent'):
            legacy_noloop(*args, **kw)
        with pytest.raises(NotImplementedError, match='time-independent'):
            fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, **kw)
        return
    expected = legacy_noloop(*args, **kw)
    got = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, **kw)
    np.testing.assert_allclose(got, expected, rtol=0., atol=3e-10)
    for a, b in zip(args[2:], saved[2:]):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    for name in ('phi', 'theta', 'psi', 'phiref', 'incl', 'dist'):
        np.testing.assert_array_equal(getattr(args[1], name), getattr(saved[1], name))


@pytest.mark.parametrize('interp', ['nearest', 'cubic'])
@pytest.mark.parametrize('phase', [False, True])
def test_continuous_export_matches_frozen_o4c(interp, phase):
    args = _inputs(n_ex=5, npts=16)
    kw = dict(time_interp=interp, phase_marginalization=phase,
              time_quadrature='bandlimited', return_time_draw=True,
              time_draw_uniforms=np.random.default_rng(9).random((5, 2)), xpy=np,
              time_draw_minimum_srate=16384)
    expected = legacy_noloop(*args, **kw)
    got = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, **kw)
    np.testing.assert_allclose(got, expected, rtol=0., atol=3e-10)


def test_heterogeneous_modes_do_not_share_harmonics():
    args = _inputs(n_ex=7)
    args[2]['L1'] = np.array([[2, 1], [2, -1]])
    kw = dict(return_lnLt=True, xpy=np)
    np.testing.assert_array_equal(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, **kw),
                                  legacy_noloop(*args, **kw))


def test_mutated_host_mode_list_is_not_cached_by_identity():
    args = _inputs(n_ex=7)
    # Distinct arrays that start equal, then change between calls.
    for det in args[2]:
        args[2][det] = args[2][det].copy()
    fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, xpy=np)
    args[2]['L1'][:] = [[2, 1], [2, -1]]
    np.testing.assert_array_equal(
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, xpy=np, return_lnLt=True),
        legacy_noloop(*args, xpy=np, return_lnLt=True))


def test_device_caches_are_separate(monkeypatch):
    from types import SimpleNamespace
    device = [0]
    backend = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: device[0])),
                              asarray=np.asarray, zeros=np.zeros, arange=np.arange,
                              concatenate=np.concatenate)
    first = fl._detector_geometry('H1', backend)
    w0 = fl._simps_weights(fl.my_simps, 16, 0.001, backend)
    device[0] = 1
    second = fl._detector_geometry('H1', backend)
    w1 = fl._simps_weights(fl.my_simps, 16, 0.001, backend)
    assert first is not second and w0 is not w1
    device[0] = 0
    assert fl._detector_geometry('H1', backend) is first
    assert fl._simps_weights(fl.my_simps, 16, 0.001, backend) is w0


def test_batched_ip_rejects_mismatched_frequency_grid_and_bad_chunks():
    from RIFT.likelihood.test_precompute_crossterm_batching import (
        _psd_array, _series, FMIN, FMAX, FNYQ, DELTAF)
    from RIFT import lalsimutils as lsu
    ip = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, _psd_array(), False, False, 0.)
    a, b = _series(2, 73)
    for chunk in (0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            ip.ip_matrix([a], [b], chunk=chunk)
    assert ip.ip_matrix([], [b]).shape == (0, 1)
    b.deltaF *= 2
    with pytest.raises(AssertionError):
        ip.ip_matrix([a], [b])


def test_real_fd_modes_batched_banks_preserve_downstream_likelihood():
    """Generate physical FD modes, then compare both U/V banks and their consumer."""
    import lal
    import lalsimulation as lalsim
    from RIFT import lalsimutils as lsu
    from test_fd_end_taper_cap import _params
    P = _params(8)
    P.dist = fl.distMpcRef * 1e6 * lal.PC_SI
    modes, conjugates = lsu.std_and_conj_hlmoff(P, Lmax=2)
    keys = [(2, 2), (2, -2)]
    modes = {k: modes[k] for k in keys}
    conjugates = {k: conjugates[k] for k in keys}
    args = list(_inputs(n_ex=11, npts=32))
    answers = []
    for batched in (False, True):
        banks = []
        for prefix, second in [('U', modes), ('V', conjugates)]:
            bank = fl.ComputeModeCrossTermIP(
                modes, second, lalsim.SimNoisePSDaLIGOZeroDetHighPower,
                20., 256., 512., P.deltaF, analyticPSD_Q=True,
                verbose=False, prefix=prefix, same_waveform_Q=True, batched=batched)
            banks.append(np.array([[bank[(a, b)] for b in keys] for a in keys]))
        args[4] = {det: banks[0].copy() for det in args[2]}
        args[5] = {det: banks[1].copy() for det in args[2]}
        answers.append(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            *args, xpy=np, time_interp='cubic'))
    np.testing.assert_allclose(*answers, rtol=0., atol=1e-10)


@pytest.mark.parametrize('prefix', ['U', 'V'])
def test_same_waveform_batching_accepts_reordered_mode_dictionary(prefix):
    from RIFT.likelihood.test_precompute_crossterm_batching import (
        _hlms, _psd_array, MODES, FMIN, FMAX, FNYQ, DELTAF)
    a = _hlms(MODES, 16)
    b = {k: a[k] for k in reversed(MODES)}
    kw = dict(verbose=False, prefix=prefix, same_waveform_Q=True)
    ref = fl.ComputeModeCrossTermIP(a, b, _psd_array(), FMIN, FMAX, FNYQ, DELTAF,
                                   batched=False, **kw)
    got = fl.ComputeModeCrossTermIP(a, b, _psd_array(), FMIN, FMAX, FNYQ, DELTAF,
                                   batched=True, **kw)
    np.testing.assert_allclose([got[k] for k in ref], list(ref.values()), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 16, 17, 614, 2457])
def test_simpson_blocked_weights_preserve_each_rule(n):
    rng = np.random.default_rng(3)
    y = rng.normal(size=(4, n))
    def rectangle(y, dx, axis):
        return np.sum(y, axis=axis)*dx
    rules = [fl.my_simps, rectangle]
    for rule in rules:
        w = fl._simps_weights(rule, n, 0.001, np, block=3)
        np.testing.assert_allclose(y @ w, rule(y, dx=0.001, axis=-1),
                                   rtol=1e-12, atol=1e-15)


def test_gpu_port_parity():
    try:
        import cupy as cp
        assert cp.cuda.runtime.getDeviceCount() > 0
    except Exception as exc:
        if os.environ.get('RIFT_CI_REQUIRE_GPU') == '1':
            pytest.fail('Required CUDA device unavailable: {}'.format(exc))
        pytest.skip('CUDA device unavailable')
    for interp in ('nearest', 'cubic'):
        for phase in (False, True):
            args = list(_inputs(n_ex=19, npts=32))
            # Oracle on the same GPU preserves the vendored Simpson rule.
            args[1] = copy.copy(args[1])
            for name in ('phi', 'theta', 'psi', 'phiref', 'incl', 'dist'):
                setattr(args[1], name, cp.asarray(getattr(args[1], name)))
            for i in (2, 3, 4, 5):
                args[i] = {k: cp.asarray(v) for k, v in args[i].items()}
            kw = dict(xpy=cp, time_interp=interp, phase_marginalization=phase)
            expected_ts = legacy_noloop(*args, return_lnLt=True, **kw)
            got_ts = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, return_lnLt=True, **kw)
            np.testing.assert_array_equal(cp.asnumpy(got_ts), cp.asnumpy(expected_ts))
            for quad in ('simpson', 'bandlimited'):
                expected = legacy_noloop(*args, time_quadrature=quad, **kw)
                got = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(*args, time_quadrature=quad, **kw)
                np.testing.assert_allclose(cp.asnumpy(got), cp.asnumpy(expected), rtol=0., atol=3e-10)
            def mutating(kappa, rho):
                rho += cp.linspace(0., 0.1, rho.shape[-1])
                return kappa-0.5*rho
            expected = legacy_noloop(*args, loglikelihood=mutating, **kw)
            got = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
                *args, loglikelihood=mutating, **kw)
            np.testing.assert_allclose(cp.asnumpy(got), cp.asnumpy(expected), rtol=0., atol=3e-10)
    # Pin both even/odd vendored Simpson rules on the actual device backend.
    for n in (1, 2, 3, 4, 5, 16, 17, 614, 2457):
        y = cp.asarray(np.random.default_rng(n).normal(size=(5, n)))
        rule = fl.optimized_gpu_tools.simps
        w = fl._simps_weights(rule, n, 0.001, cp, block=3)
        np.testing.assert_allclose(cp.asnumpy(y @ w), cp.asnumpy(rule(y, dx=0.001, axis=-1)),
                                   rtol=1e-12, atol=1e-15)
