#!/usr/bin/env python3
"""Regression tests for sub-sample time-posterior export."""

import importlib.util
import os

import numpy as np
import pytest

DRIVER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bin",
    "integrate_likelihood_extrinsic_batchmode")
MODULE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "RIFT", "likelihood",
    "time_posterior.py")
SPEC = importlib.util.spec_from_file_location("time_posterior", MODULE)
TIME_POSTERIOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIME_POSTERIOR)
draw_continuous_time_posterior = TIME_POSTERIOR.draw_continuous_time_posterior
resolve_time_posterior_export_mode = TIME_POSTERIOR.resolve_time_posterior_export_mode


def test_auto_contract_tracks_subsample_interpolation():
    assert resolve_time_posterior_export_mode("auto", "nearest") == "grid"
    assert resolve_time_posterior_export_mode("auto", "cubic") == "continuous"
    assert resolve_time_posterior_export_mode("auto", "sinc") == "continuous"
    assert resolve_time_posterior_export_mode("grid", "cubic") == "grid"
    assert resolve_time_posterior_export_mode("continuous", "nearest") == "continuous"


def test_continuous_draws_are_not_on_the_input_lattice():
    tvals = np.linspace(-0.01, 0.01, 41)
    lnlt = -0.5 * (tvals / 0.002) ** 2
    rng = np.random.RandomState(20260829)
    draws = np.array([draw_continuous_time_posterior(tvals, lnlt, rng)[0]
                      for _ in range(200)])
    phase = (draws - tvals[0]) / (tvals[1] - tvals[0])
    assert np.all(draws >= tvals[0]) and np.all(draws <= tvals[-1])
    assert np.count_nonzero(np.isclose(phase, np.round(phase), atol=1e-10)) == 0


def test_gaussian_posterior_moments_and_interpolated_logl():
    sigma = 0.0017
    center = 0.0008
    tvals = np.linspace(-0.012, 0.012, 65)
    lnlt = -0.5 * ((tvals - center) / sigma) ** 2
    rng = np.random.RandomState(17)
    draws, logls = zip(*(draw_continuous_time_posterior(tvals, lnlt, rng)
                         for _ in range(12000)))
    draws = np.asarray(draws)
    logls = np.asarray(logls)
    assert np.mean(draws) == pytest.approx(center, abs=5e-5)
    assert np.std(draws) == pytest.approx(sigma, rel=0.035)
    np.testing.assert_allclose(logls, -0.5 * ((draws - center) / sigma) ** 2,
                               rtol=0, atol=2e-12)


def test_batched_rows_draw_from_their_own_posteriors():
    tvals = np.linspace(-0.02, 0.02, 81)
    centers = np.array([-0.006, 0.0, 0.007])
    lnlt = -0.5 * ((tvals[None, :] - centers[:, None]) / 0.001) ** 2
    draws, logls = draw_continuous_time_posterior(
        tvals, lnlt, np.random.RandomState(9))
    assert draws.shape == logls.shape == centers.shape
    assert np.all(np.abs(draws - centers) < 0.004)


def test_driver_wires_continuous_draw_before_legacy_grid_choice():
    with open(DRIVER) as handle:
        source = handle.read()
    continuous = source.index("draw_continuous_time_posterior(tvals, lnLt)")
    grid = source.index("indx_choose = np.random.choice", continuous)
    assert continuous < grid
    assert 'opts._time_posterior_export == "continuous"' in source
    assert 'opts._time_posterior_export == "grid"' in source
