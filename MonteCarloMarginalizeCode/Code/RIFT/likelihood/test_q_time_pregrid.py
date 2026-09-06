#!/usr/bin/env python3
# RIFT-CI-GATE: q-window-stencil
"""Focused tests for the opt-in reflected Q pregrid."""

import numpy as np

from RIFT.likelihood.factored_likelihood import (
    _cubic_Q_window_numpy,
    _q_inner_product_explicit_times,
    _q_sample_positions,
    build_reflected_q_pregrid,
)


def test_reflected_pregrid_roundtrip_odd_even_and_size():
    rng = np.random.RandomState(811)
    for n_time in (31, 32):
        coarse = rng.normal(size=(3, n_time)) + 1j*rng.normal(size=(3, n_time))
        fine, report = build_reflected_q_pregrid(coarse, factor=8)
        assert fine.shape == (3, (n_time - 1)*8 + 1)
        np.testing.assert_allclose(fine[..., ::8], coarse, rtol=5e-13, atol=5e-13)
        assert report['factor'] == 8
        assert report['output_bytes'] == fine.nbytes


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
