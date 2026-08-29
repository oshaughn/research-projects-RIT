#!/usr/bin/env python3
"""Regression tests for sub-sample time-posterior export."""

import importlib.util
import os

import numpy as np
import pytest

DRIVER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bin",
    "integrate_likelihood_extrinsic_batchmode")
LISA_DRIVER = DRIVER + "_lisa"
MODULE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "RIFT", "likelihood",
    "time_posterior.py")
SPEC = importlib.util.spec_from_file_location("time_posterior", MODULE)
TIME_POSTERIOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIME_POSTERIOR)
TMARG_MODULE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "RIFT", "likelihood",
    "time_marginalization_quadrature.py")
TMARG_SPEC = importlib.util.spec_from_file_location(
    "time_marginalization_quadrature", TMARG_MODULE)
TMARG = importlib.util.module_from_spec(TMARG_SPEC)
TMARG_SPEC.loader.exec_module(TMARG)
draw_continuous_time_posterior = TIME_POSTERIOR.draw_continuous_time_posterior
resolve_time_posterior_export_mode = TIME_POSTERIOR.resolve_time_posterior_export_mode
legacy_time_interpolation_enabled = TIME_POSTERIOR.legacy_time_interpolation_enabled
_interval_log_envelopes = TIME_POSTERIOR._interval_log_envelopes


def test_auto_contract_tracks_subsample_interpolation():
    assert resolve_time_posterior_export_mode("auto", "nearest") == "grid"
    assert resolve_time_posterior_export_mode("auto", "cubic") == "continuous"
    assert resolve_time_posterior_export_mode("auto", "sinc") == "continuous"
    assert resolve_time_posterior_export_mode("grid", "cubic") == "grid"
    assert resolve_time_posterior_export_mode("continuous", "nearest") == "continuous"


def test_lisa_legacy_interpolation_parser_does_not_treat_false_as_truthy():
    for value in (False, "False", "false", "0", "off", "none", "nearest"):
        assert legacy_time_interpolation_enabled(value) is False
    for value in (True, "True", "1", "yes", "on", "cubic", "sinc"):
        assert legacy_time_interpolation_enabled(value) is True
    with pytest.raises(ValueError, match="unrecognised LISA value"):
        legacy_time_interpolation_enabled("sinK")


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


def test_stationary_point_envelope_bounds_overshooting_cubics():
    from scipy.interpolate import CubicSpline

    rng = np.random.RandomState(37)
    knots = np.linspace(-1.0, 1.0, 9)
    for _ in range(20):
        values = rng.normal(size=knots.size)
        spline = CubicSpline(knots, values)
        maxima = _interval_log_envelopes(spline, knots, values)
        for interval in range(knots.size - 1):
            probes = np.linspace(knots[interval], knots[interval + 1], 1001)
            assert np.max(spline(probes)) <= maxima[interval] + 5e-14


def test_batched_rows_draw_from_their_own_posteriors():
    tvals = np.linspace(-0.02, 0.02, 81)
    centers = np.array([-0.006, 0.0, 0.007])
    lnlt = -0.5 * ((tvals[None, :] - centers[:, None]) / 0.001) ** 2
    draws, logls = draw_continuous_time_posterior(
        tvals, lnlt, np.random.RandomState(9))
    assert draws.shape == logls.shape == centers.shape
    assert np.all(np.abs(draws - centers) < 0.004)


def test_bandlimited_refinement_uses_components_and_preserves_coarse_samples():
    n = 65
    phase = np.linspace(-np.pi, np.pi, n)
    # A narrow, band-limited peak whose measured curvature requires refinement.
    kappa = (80.0 * np.cos(phase)[None, :]).astype(complex)
    rho_sq = np.zeros(kappa.shape)
    dense, factor = TMARG.refine_time_posterior_bandlimited(
        kappa, rho_sq, 1.0,
        lambda data_term, self_term: data_term - 0.5 * self_term)
    assert factor > 1
    np.testing.assert_allclose(dense[:, ::factor], kappa.real, rtol=0, atol=2e-12)
    assert dense.shape[-1] == (n - 1) * factor + 1

    # Calibration realizations must be reconstructed before their weighted
    # log-sum-exp reduction, not spline-interpolated after marginalization.
    kappa_cal = np.stack((kappa, kappa - 2.0), axis=1)
    rho_cal = np.zeros(kappa_cal.shape)
    weights = np.log(np.array([1.5, 0.5]))
    dense_cal, factor_cal = TMARG.refine_time_posterior_bandlimited(
        kappa_cal, rho_cal, 1.0,
        lambda data_term, self_term: data_term - 0.5 * self_term,
        cal_log_weights=weights)
    coarse_weighted = np.log(
        (1.5 * np.exp(kappa.real) + 0.5 * np.exp(kappa.real - 2.0)) / 2.0)
    np.testing.assert_allclose(
        dense_cal[:, ::factor_cal], coarse_weighted, rtol=0, atol=2e-12)


def test_negative_infinity_knots_are_zero_mass_not_an_arbitrary_log_floor():
    from scipy.interpolate import PchipInterpolator

    tvals = np.linspace(-1.0, 1.0, 5)
    lnlt = np.array([-np.inf, -1.0, 0.0, -1.0, -np.inf])
    rng = np.random.RandomState(81)
    draws, logls = zip(*(draw_continuous_time_posterior(tvals, lnlt, rng)
                         for _ in range(200)))
    draws = np.asarray(draws)
    logls = np.asarray(logls)
    density = np.exp(np.where(np.isfinite(lnlt), lnlt, -np.inf))
    expected = PchipInterpolator(tvals, density)(draws)
    np.testing.assert_allclose(np.exp(logls), expected, rtol=2e-14, atol=0)
    assert np.all(np.isfinite(logls))
    assert np.all((draws > tvals[0]) & (draws < tvals[-1]))


@pytest.mark.parametrize("bad", [
    np.array([0.0, np.nan, -1.0]),
    np.array([0.0, np.inf, -1.0]),
])
def test_nan_and_positive_infinity_fail_loudly(bad):
    with pytest.raises(ValueError, match=r"NaN or \+inf"):
        draw_continuous_time_posterior(np.arange(3.0), bad)


def test_all_negative_infinity_has_no_posterior_mass():
    with pytest.raises(ValueError, match="no finite positive mass"):
        draw_continuous_time_posterior(
            np.arange(3.0), np.full(3, -np.inf))


def test_pathological_rejection_cannot_hang_the_driver():
    class AlwaysReject(object):
        def choice(self, size, p):
            return 0

        def uniform(self, *bounds):
            return 0.5 * (bounds[0] + bounds[1]) if bounds else 1.0

    with pytest.raises(RuntimeError, match="exhausted 3 proposals"):
        draw_continuous_time_posterior(
            np.arange(3.0), np.array([0.0, -100.0, 0.0]),
            rng=AlwaysReject(), max_attempts=3)


def test_driver_wires_continuous_draw_before_legacy_grid_choice():
    with open(DRIVER) as handle:
        source = handle.read()
    components = source.index("return_time_components=True")
    refinement = source.index("refine_time_posterior_bandlimited(", components)
    dense_labels = source.index("xpy_default.arange(n_dense)", refinement)
    continuous = source.index("draw_continuous_time_posterior(tvals, lnLt)", dense_labels)
    grid = source.index("indx_choose = np.random.choice", continuous)
    assert components < refinement < dense_labels < continuous < grid
    assert continuous < grid
    assert "return_time_components=True" in source
    assert "refine_time_posterior_bandlimited(" in source
    assert 'opts._time_posterior_export == "continuous"' in source
    assert 'opts._time_posterior_export == "grid"' in source


def test_lisa_twin_refuses_continuous_mode_without_faithful_components():
    with open(LISA_DRIVER) as handle:
        source = handle.read()
    assert '"--time-posterior-export"' in source
    assert "legacy_time_interpolation_enabled(opts.interpolate_time)" in source
    assert ("opts.resample_time_marginalization and\n"
            "        opts._time_posterior_export == \"continuous\"") in source
    assert "does not expose the band-limited time components" in source
    assert "draw_continuous_time_posterior(tvals, lnLt)" not in source
