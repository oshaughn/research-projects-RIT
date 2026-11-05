"""Row packing, numerical equivalence and worker failure regressions.

Set RIFT_TEST_GPU=1 to require CuPy on an assigned GPU (no CPU fallback).
"""
import ast
import os
from pathlib import Path

import numpy as np
import pytest
from RIFT.likelihood import time_marginalization_quadrature as tq


class PackedNumpy:
    """Exercise the GPU packing policy with independently observable FFT calls."""
    __name__ = 'cupy'

    def __init__(self):
        self.shapes = []
        self.fft = self

    def __getattr__(self, name):
        return getattr(np, name)

    def fft(self, x, n=None, axis=-1):
        self.shapes.append((x.shape, x.strides, n, False))
        return np.fft.fft(x, n=n, axis=axis)

    def ifft(self, x, n=None, axis=-1):
        self.shapes.append((x.shape, x.strides, n, True))
        return np.fft.ifft(x, n=n, axis=axis)


@pytest.mark.parametrize('inverse,n', [(False, None), (True, None), (False, 137), (True, 29)])
def test_fixed_batch_shapes_preserve_all_rows_and_strides(inverse, n):
    backend = PackedNumpy()
    # fft is both a namespace and a function in this small test adapter.
    from types import SimpleNamespace
    backend.fft = SimpleNamespace(fft=backend.__class__.fft.__get__(backend),
                                  ifft=backend.ifft)
    rng = np.random.default_rng(915)
    for count in [1, 63, 64, 65, 127, 143, 257]:
        raw = rng.normal(size=(count, 102)) + 1j*rng.normal(size=(count, 102))
        rows = raw[:, ::-2]
        original = rows.copy()
        result = tq._fft_rows(rows, inverse=inverse, n=n, xpy=backend)
        reference = (np.fft.ifft if inverse else np.fft.fft)(rows, n=n, axis=-1)
        np.testing.assert_allclose(result, reference, atol=2e-13, rtol=2e-13)
        np.testing.assert_array_equal(rows, original)
    assert {shape for shape, _, _, _ in backend.shapes} == {(64, 51)}
    assert len({strides for _, strides, _, _ in backend.shapes}) == 1


def test_long_fft_reduces_capacity_and_keeps_partial_tile():
    shapes = []
    def fft(x, n, axis):
        shapes.append(x.shape)
        return np.fft.fft(x, n=n, axis=axis)
    rows = np.arange(5*17).reshape(5, 17).astype(np.complex128)
    got = tq._gpu_fft_rows(rows, fft, n=131073, xpy=np)
    np.testing.assert_allclose(got, np.fft.fft(rows, n=131073), rtol=0, atol=1e-11)
    assert shapes == [(2, 17)]*3


@pytest.mark.parametrize('failure', [MemoryError('1234'), RuntimeError('CUFFT_INTERNAL_ERROR'),
                                    RuntimeError('unexpected transform failure')])
def test_transform_errors_propagate_without_full_padding(monkeypatch, failure):
    def fail(*a, **kw):
        raise failure
    def forbidden(*a, **kw):
        pytest.fail('resource/unexpected failure retried a larger FFT')
    monkeypatch.setattr(tq, '_reflected_bandlimited_upsample_retained', fail)
    monkeypatch.setattr(tq, 'reflected_bandlimited_upsample', forbidden)
    with pytest.raises(type(failure)) as caught:
        tq._reflected_upsample_for_integration(np.ones((2, 17)), 8, {}, tq._new_transform_report())
    assert caught.value is failure


def test_worker_resource_failure_raises_before_empty_or_soft_success():
    cli = Path(os.environ.get('RIFT_TEST_CLI', Path(__file__).parents[1] / 'bin' /
                              'integrate_likelihood_extrinsic_batchmode'))
    tree = ast.parse(cli.read_text())
    handler = next(h for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler)
                   and h.name == 'exception_failure')
    # Execute the actual CLI handler in isolation, including both dangerous
    # legacy flags. The resource guard must precede any option access/product IO.
    probe = ast.Module(body=handler.body, type_ignores=[])
    from types import SimpleNamespace
    scope = dict(QTimePregridError=type('QTimePregridError', (Exception,), {}),
                 factored_likelihood=SimpleNamespace(time_quad=tq))
    class ByteCountError(RuntimeError):
        pass
    ByteCountError.__module__ = 'cupy.cuda.memory'
    for failure in [MemoryError('1234'), ByteCountError('4096'),
                    RuntimeError('CUFFT_INTERNAL_ERROR'), RuntimeError('out of memory')]:
        scope['exception_failure'] = failure
        # The guard's bare raise runs within the active exception context.
        try:
            raise failure
        except Exception:
            with pytest.raises(type(failure)) as caught:
                exec(compile(probe, str(cli), 'exec'), scope)
            assert caught.value is failure
    assert not tq.is_gpu_resource_error(ValueError('bad waveform parameter'))


@pytest.fixture
def gpu():
    if os.environ.get('RIFT_TEST_GPU') != '1':
        pytest.skip('GPU tests require explicit assigned-GPU lane')
    import cupy as cp
    cp.cuda.Device(0).use()
    return cp


@pytest.mark.parametrize('n,factor', [(17, 3), (614, 8), (819, 8), (1228, 4), (2457, 2)])
def test_gpu_reflection_matches_full_padding(gpu, n, factor):
    cp = gpu
    rng = np.random.default_rng(846)
    # Different rows and a tile boundary catch omission/repetition/reordering.
    host = rng.normal(size=(67, n)) + 1j*rng.normal(size=(67, n))
    rows = cp.asarray(host)
    reference = tq.reflected_bandlimited_upsample(host, factor)
    report = tq._new_transform_report()
    got = tq._reflected_upsample_for_integration(rows, factor, {}, report, xpy=cp)
    np.testing.assert_allclose(cp.asnumpy(got), reference, rtol=0, atol=2e-11)
    np.testing.assert_allclose(cp.asnumpy(got[:, ::factor]), host, rtol=0, atol=2e-11)


@pytest.mark.parametrize('positive_peak', [True, False])
def test_gpu_integral_and_conditional_draw_match_reference(gpu, monkeypatch, positive_peak):
    cp = gpu
    n, count = 819, 67
    t = np.arange(n)
    peak = 408 + np.linspace(-.45, .45, count)[:, None]
    host = (20 + np.arange(count)[:, None]*.7) * np.exp(-((t-peak)/3.1)**2)
    host = host.astype(np.complex128) * np.exp(.007j*(t-(408 if positive_peak else 0)))
    k = cp.asarray(host)
    rho = cp.zeros_like(k.real)
    uniforms = cp.asarray(np.random.default_rng(92).uniform(size=(count, 2)))
    fn = lambda k, r: k - .5*r
    args = dict(xpy=cp, return_time_draw=True, draw_uniforms=uniforms,
                simps=lambda y, dx, axis: cp.sum(y, axis=axis)*dx)
    got = tq.time_marginalize_bandlimited(k, rho, 1/4096, fn, **args)
    def full(x, factor, plan_cache, report, xpy=np):
        return tq.reflected_bandlimited_upsample(x, factor, xpy=xpy)
    monkeypatch.setattr(tq, '_reflected_upsample_for_integration', full)
    reference = tq.time_marginalize_bandlimited(k, rho, 1/4096, fn, **args)
    # Nearly flat posteriors condition the inverse CDF poorly: allow 10 ns for
    # the draw (versus a 244 us physical knot), retaining the strict nat bound
    # on both the marginal integral and the density at the drawn time.
    tolerances = (2e-10, 2e-10 if positive_peak else 1e-8, 2e-10)
    for actual, expected, tolerance in zip(got, reference, tolerances):
        np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected), rtol=0, atol=tolerance)
