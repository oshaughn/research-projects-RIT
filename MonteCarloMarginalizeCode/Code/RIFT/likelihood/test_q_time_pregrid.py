#!/usr/bin/env python3
# RIFT-CI-GATE: q-window-stencil
"""Focused tests for the opt-in reflected Q pregrid."""

import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from RIFT.likelihood.factored_likelihood import (
    _cubic_Q_window_numpy,
    _q_inner_product_explicit_times,
    _q_sample_positions,
    build_reflected_q_pregrid,
    prepare_reflected_q_pregrid,
)
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood import time_marginalization_quadrature as tmq

try:
    import cupy
    HAVE_GPU = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    cupy = None
    HAVE_GPU = False


def test_reflected_pregrid_roundtrip_odd_even_and_size():
    rng = np.random.RandomState(811)
    for n_time in (31, 32):
        coarse = rng.normal(size=(3, n_time)) + 1j*rng.normal(size=(3, n_time))
        fine, report = build_reflected_q_pregrid(coarse, factor=8)
        assert fine.shape == (3, (n_time - 1)*8 + 1)
        np.testing.assert_allclose(fine[..., ::8], coarse, rtol=5e-13, atol=5e-13)
        assert report['factor'] == 8
        assert report['output_bytes'] == fine.nbytes
        assert report['retained_bytes'] == fine.nbytes
        assert report['peak_allocation_bytes'] > fine.nbytes
        assert fine.flags.owndata
        assert fine.base is None


def test_backend_oom_rolls_back_whole_dictionary_and_cleans_up():
    original = {'H1': np.ones((2, 9)), 'L1': np.ones((2, 9))*2}
    calls = []
    cleaned = []

    def transfer(value):
        calls.append(value.shape[-1])
        if calls == [65, 65]:
            raise MemoryError('forced device OOM')
        return np.array(value, copy=True)

    got, reports, error = prepare_reflected_q_pregrid(
        original, factor=8, transfer=transfer, cleanup=lambda: cleaned.append(True))
    assert isinstance(error, MemoryError)
    assert reports == []
    assert cleaned == [True]
    assert calls == [65, 65, 9, 9]
    for det in original:
        np.testing.assert_array_equal(got[det], original[det])


def test_reflection_is_load_bearing_at_both_nonperiodic_edges():
    # A smooth finite-window ramp has deliberately unlike endpoints.  Direct
    # periodic interpolation joins them and rings; even reflection preserves
    # the local continuation at both edges.  This test fails if reflection is
    # mutated to direct periodic upsampling.
    n = 64
    factor = 8
    x = np.linspace(-1.0, 1.0, n)
    coarse = (x + 0.15*x**2)[None, :]
    direct = tmq.bandlimited_upsample(coarse, factor)[0]
    reflected, _ = build_reflected_q_pregrid(coarse, factor=factor)
    dense_x = np.linspace(-1.0, 1.0, (n - 1)*factor + 1)
    truth = dense_x + 0.15*dense_x**2
    edge = np.r_[1:factor, len(truth)-factor:len(truth)-1]
    reflected_error = np.max(np.abs(reflected[0, edge] - truth[edge]))
    direct_error = np.max(np.abs(direct[edge] - truth[edge]))
    assert reflected_error < 0.2*direct_error, (reflected_error, direct_error)


def test_separate_grid_refuses_unimplemented_stencils():
    p = SimpleNamespace(deltaT=1.0, q_deltaT=0.125, phi=np.array([0.0]),
                        theta=np.array([0.0]), phiref=np.array([0.0]),
                        incl=np.array([0.0]), psi=np.array([0.0]),
                        dist=np.array([fl.distMpcRef*1e6*fl.lal.PC_SI]), tref=0.0)
    args = (np.arange(2.0), p, {}, {}, {}, {}, {})
    for stencil in ('nearest', 'sinc'):
        try:
            fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
                *args, time_interp=stencil, return_lnLt=True)
        except NotImplementedError:
            pass
        else:
            raise AssertionError('%s silently accepted a separate Q spacing' % stencil)


def test_separate_q_spacing_preserves_coarse_integration_nodes():
    t_det = np.array([10.25, 11.5])
    tvals = np.arange(7)*0.25 - 0.5
    starts, fractions, per_time, stride = _q_sample_positions(
        t_det, tvals, 0.25, 0.25/8, 'cubic', False)
    assert not per_time
    assert stride == 8
    target = (t_det + tvals[0])/(0.25/8)
    np.testing.assert_array_equal(starts, np.floor(target).astype(np.int32))
    np.testing.assert_allclose(fractions, target - np.floor(target))
    # The geocentric nodes are still separated by the original 0.25 seconds;
    # only their coordinates on Q advance by eight samples.
    grid = np.arange(200, dtype=float)
    q = (grid**3 - 2*grid + 1).astype(complex)[:, None]
    got = _cubic_Q_window_numpy(q, np.array([20]), np.array([0.25]), 7,
                                time_stride=stride)[0, :, 0]
    x = 20.25 + np.arange(7)*8
    np.testing.assert_allclose(got, x**3 - 2*x + 1, rtol=2e-13)


def test_factor_one_keeps_historical_scalar_window_gather():
    starts, fractions, per_time, stride = _q_sample_positions(
        np.array([4.25, 8.75]), np.arange(5)*0.5 - 1.0,
        0.5, 0.5, 'cubic', False)
    assert not per_time
    assert stride == 1
    assert starts.shape == (2,)
    expected_samples = (np.array([4.25, 8.75]) - 1.0)/0.5
    np.testing.assert_allclose(fractions, expected_samples - np.floor(expected_samples))


def test_cubic_explicit_gather_matches_cubic_truth_and_zero_extends_edges():
    # A cubic polynomial is reproduced exactly by the four-tap stencil.
    grid = np.arange(20, dtype=float)
    q = (grid**3 - 2*grid**2 + 0.5*grid + 3).astype(complex)[:, None]
    starts = np.array([[4, 8, 12]], dtype=np.int32)
    fractions = np.array([[0.125, 0.5, 0.875]])
    amplitude = np.array([[2.0 - 0.25j]])
    got = _q_inner_product_explicit_times(
        q, amplitude, starts, fractions, 'cubic', xpy=np)
    x = starts + fractions
    truth = amplitude[0, 0]*(x**3 - 2*x**2 + 0.5*x + 3)
    np.testing.assert_allclose(got, truth, rtol=2e-13, atol=2e-12)

    # Far outside the captured Q interval every tap is unavailable: fail closed
    # to zero rather than wrapping reflected-pregrid samples across an edge.
    outside = _q_inner_product_explicit_times(
        q, amplitude, np.array([[-10, 30]], dtype=np.int32),
        np.array([[0.5, 0.5]]), 'cubic', xpy=np)
    np.testing.assert_array_equal(outside, 0.0)


def _phase_noloop(q_rows, q_delta_t, stride, fractional):
    n_time = q_rows.shape[-1]
    start = 8
    integration_dt = q_delta_t*stride
    t_det = (start + fractional)*q_delta_t
    p = SimpleNamespace(
        deltaT=integration_dt, q_deltaT=q_delta_t,
        phi=np.array([0.1]), theta=np.array([0.2]),
        phiref=np.array([0.3]), incl=np.array([0.4]), psi=np.array([0.5]),
        dist=np.array([fl.distMpcRef*1e6*fl.lal.PC_SI]), tref=0.0)
    y = np.array([[1.2 + 0.4j, -0.7 + 0.2j]])
    response = np.array([0.8 - 0.3j])
    tvals = np.arange(3)*integration_dt
    lookup = {'H1': np.array([[2, 2], [2, -2]])}
    rho = {'H1': q_rows}
    zeros = {'H1': np.zeros((2, 2), dtype=complex)}
    epochs = {'H1': 0.0}
    with patch.object(fl, '_detector_geometry', return_value=(None, None)), \
         patch.object(fl, 'SourcePolarizationBasis', return_value=(None, None)), \
         patch.object(fl, 'SourcePropagationDirection', return_value=None), \
         patch.object(fl, 'ComputeDetAMResponsePrecomputed', return_value=response), \
         patch.object(fl, 'TimeDelayFromEarthCenterPrecomputed',
                      return_value=np.array([t_det])), \
         patch.object(fl, 'SphericalHarmonicsVectorized', return_value=y.copy()):
        got = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, p, lookup, rho, zeros, zeros, epochs, Lmax=2, xpy=np,
            return_lnLt=True, phase_marginalization=True, time_interp='cubic')
    q_block = np.column_stack((q_rows[0], np.conj(q_rows[1])))
    sampled = _cubic_Q_window_numpy(
        q_block, np.array([start]), np.array([fractional]), 3,
        time_stride=stride)[0]
    y_phase = y.copy(); y_phase[:, 1] = np.conj(y_phase[:, 1])
    factors = np.array([[response[0], np.conj(response[0])]])*y_phase
    expected = np.abs(np.einsum('ti,i->t', sampled, np.conj(factors[0])))
    np.testing.assert_allclose(got[0], expected, rtol=2e-13, atol=2e-13)


def test_cpu_phase_marginalization_scalar_and_pregrid_match_reference():
    grid = np.arange(40.0)
    coarse = np.vstack((np.exp(0.08j*grid), (1 + 0.01*grid)*np.exp(-0.05j*grid)))
    _phase_noloop(coarse, 1.0, 1, 0.25)
    fine, _ = build_reflected_q_pregrid(coarse, factor=8)
    _phase_noloop(fine, 1.0/8, 8, 0.25)


def test_gpu_stride8_cubic_matches_cpu_at_fractional_and_edge_starts():
    if not HAVE_GPU:
        return
    rng = np.random.RandomState(91)
    q = rng.normal(size=(70, 3)) + 1j*rng.normal(size=(70, 3))
    amplitude = rng.normal(size=(4, 3)) + 1j*rng.normal(size=(4, 3))
    starts = np.array([-2, 3, 58, 68], dtype=np.int32)
    fractions = np.array([0.2, 0.75, 0.4, 0.9])
    cpu_q = _cubic_Q_window_numpy(q, starts, fractions, 5, time_stride=8)
    expected = np.einsum('eti,ei->et', cpu_q, amplitude)
    got = fl.Q_inner_product.Q_inner_product_cubic_cupy(
        cupy.asarray(q), cupy.asarray(amplitude), cupy.asarray(starts),
        cupy.asarray(fractions), 5, time_stride=8)
    np.testing.assert_allclose(cupy.asnumpy(got), expected, rtol=2e-12, atol=2e-12)
