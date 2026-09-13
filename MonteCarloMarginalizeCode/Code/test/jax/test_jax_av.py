"""Contract tests for the value-only JAX -> AV/portfolio adapter."""

import importlib.machinery
import importlib.util
import os
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import samplers

_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DRIVER = os.path.join(_CODE, "bin", "integrate_likelihood_extrinsic_jax")


def _trapezoid(y, x):
    """Integrate on NumPy versions before and after ``trapz`` was removed."""
    trapezoid = getattr(np, "trapezoid", None)
    return trapezoid(y, x) if trapezoid is not None else np.trapz(y, x)


def _driver_module():
    loader = importlib.machinery.SourceFileLoader("_jax_av_driver", _DRIVER)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_waveform_precompute_kwargs_match_production_ile_controls():
    driver = _driver_module()
    parser = driver.build_parser()

    opts, _ = parser.parse_args([
        "--approximant", "IMRPhenomXPHM",
        "--internal-waveform-fd-L-frame",
        "--internal-waveform-fd-no-condition",
    ])
    got = driver._waveform_precompute_kwargs(opts)
    assert got == {
        "use_gwsignal": False,
        "use_gwsignal_approx": None,
        "ignore_threshold": None,
        "no_memory": False,
        "extra_waveform_kwargs": {
            "fd_alignment_postevent_time": 2,
            "e_freq": 1,
            "fd_L_frame": True,
            "no_condition": True,
        },
    }

    defaults, _ = parser.parse_args(["--approximant", "IMRPhenomD"])
    assert driver._waveform_precompute_kwargs(defaults)[
        "extra_waveform_kwargs"] == {
            "fd_alignment_postevent_time": 2, "e_freq": 1}


def test_internal_sample_rate_controls_precompute_cadence():
    driver = _driver_module()
    parser = driver.build_parser()
    defaults, _ = parser.parse_args(["--srate", "1024"])
    internal, _ = parser.parse_args(
        ["--srate", "1024", "--srate-internal", "4096"])
    assert driver._analysis_delta_t(defaults) == 1.0 / 1024.0
    assert driver._analysis_delta_t(internal) == 1.0 / 4096.0


class _ToySkyLikelihood:
    ANGULAR_PARAM_ORDER = ("ra", "dec", "incl")

    def __init__(self):
        centre = jnp.array([2.0, 0.2, 1.0])
        scale = jnp.array([0.5, 0.35, 0.4])

        def scalar(theta):
            return -0.5 * jnp.sum(((theta - centre) / scale) ** 2)

        self._scalar = scalar
        self._value_and_grad = jax.jit(jax.value_and_grad(scalar))
        self._hessian = jax.jit(jax.hessian(scalar))
        self.batch_shapes = []

    def log_likelihood(self, *cols):
        theta = jnp.stack(cols, axis=-1)
        self.batch_shapes.append(tuple(theta.shape))
        return jax.vmap(self._scalar)(theta)

    def value_and_grad(self, theta):
        value, grad = self._value_and_grad(jnp.asarray(theta))
        return float(value), np.asarray(grad)

    def fisher(self, theta):
        return -np.asarray(self._hessian(jnp.asarray(theta)))


def test_fixed_shape_callback_pads_only_the_hidden_tail():
    like = _ToySkyLikelihood()
    callback = samplers._fixed_shape_value_callback(like, 3, 8)
    theta = np.arange(57, dtype=float).reshape(19, 3) / 20.0

    got = callback(*theta.T)
    want = np.asarray(jax.vmap(like._scalar)(jnp.asarray(theta)))

    np.testing.assert_allclose(got, want)
    assert like.batch_shapes[:3] == [(8, 3), (8, 3), (8, 3)]
    assert got.shape == (19,)  # padded rows never escape the adapter


def test_physical_coordinate_priors_are_normalized():
    for name in ("ra", "dec", "psi", "incl", "phiref", "distMpc"):
        lo, hi, density = samplers._av_prior_spec(name, 10.0, 100.0)
        x = np.linspace(lo, hi, 20001)
        np.testing.assert_allclose(_trapezoid(density(x), x), 1.0,
                                   rtol=2e-6, atol=2e-6)


def test_pseudo_cosmo_distance_prior_density_and_draw_are_consistent():
    from RIFT.likelihood import priors_utils

    lo, hi = 1.0, 10000.0
    _, _, density = samplers._av_prior_spec(
        "distMpc", lo, hi, distance_prior="pseudo_cosmo")
    grid = np.linspace(lo, hi, 50001)
    pdf = density(grid)
    np.testing.assert_allclose(_trapezoid(pdf, grid), 1.0, rtol=2e-7)
    norm = priors_utils.dist_prior_pseudo_cosmo_eval_norm(lo, hi)
    np.testing.assert_allclose(
        pdf, priors_utils.dist_prior_pseudo_cosmo(grid, nm=norm), rtol=1e-14)

    draw = samplers._av_prior_draw(
        ("distMpc",), 30000, np.random.default_rng(240426), lo, hi,
        distance_prior="pseudo_cosmo")[:, 0]
    cdf = np.concatenate(([0.0], np.cumsum(
        0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))))
    cdf /= cdf[-1]
    expected_median = np.interp(0.5, cdf, grid)
    assert abs(np.median(draw) - expected_median) < 0.015 * expected_median
    assert np.all((draw >= lo) & (draw <= hi))


def test_sampling_window_does_not_renormalize_physical_prior():
    bounds = {"ra": (1.17, 1.23), "dec": (0.27, 0.33)}
    ra_lo, ra_hi, ra_pdf = samplers._av_prior_spec(
        "ra", 10.0, 100.0, sample_bounds=bounds)
    dec_lo, dec_hi, dec_pdf = samplers._av_prior_spec(
        "dec", 10.0, 100.0, sample_bounds=bounds)

    assert (ra_lo, ra_hi) == bounds["ra"]
    assert (dec_lo, dec_hi) == bounds["dec"]
    np.testing.assert_allclose(
        _trapezoid(ra_pdf(np.linspace(ra_lo, ra_hi, 10001)),
                   np.linspace(ra_lo, ra_hi, 10001)),
        (ra_hi - ra_lo) / (2 * np.pi), rtol=1e-10)
    np.testing.assert_allclose(
        _trapezoid(dec_pdf(np.linspace(dec_lo, dec_hi, 10001)),
                   np.linspace(dec_lo, dec_hi, 10001)),
        0.5 * (np.sin(dec_hi) - np.sin(dec_lo)), rtol=1e-9)


def test_prior_draw_respects_sky_sampling_window():
    bounds = {"ra": (1.17, 1.23), "dec": (0.27, 0.33)}
    draw = samplers._av_prior_draw(
        ("ra", "dec", "incl"), 10000, np.random.default_rng(8),
        1.0, 100.0, bounds)
    assert np.all((draw[:, 0] >= 1.17) & (draw[:, 0] <= 1.23))
    assert np.all((draw[:, 1] >= 0.27) & (draw[:, 1] <= 0.33))
    assert np.std(draw[:, 2]) > 0.6


@pytest.mark.parametrize("bounds,match", [
    ({"ra": (6.1, 0.1)}, "invalid sampling bounds"),
    ({"dec": (-2.0, 0.1)}, "must lie within"),
    ({"bogus": (0.0, 1.0)}, "absent from likelihood"),
])
def test_sampling_window_validation(bounds, match):
    with pytest.raises(ValueError, match=match):
        samplers._av_sample_bounds(
            _ToySkyLikelihood.ANGULAR_PARAM_ORDER, 1.0, 100.0,
            sample_bounds=bounds)


def test_fisher_sky_seed_targets_sky_but_randomizes_other_coordinates():
    like = _ToySkyLikelihood()
    callback = samplers._fixed_shape_value_callback(like, 3, 64)
    cloud, modes, mode_lnL = samplers._fisher_sky_seed(
        like, like.ANGULAR_PARAM_ORDER, callback, np.random.default_rng(7),
        1.0, 100.0, n_seed=600, n_pilot=200, n_modes=2,
        sky_inflate=1.5, prior_frac=0.1)

    assert cloud.shape == (600, 3)
    assert modes.shape[1] == 3 and np.all(np.isfinite(mode_lnL))
    assert np.std(cloud[:, 2]) > 0.5  # inclination came from its broad prior
    assert np.min(np.abs(np.angle(np.exp(1j * (cloud[:, 0] - 2.0))))) < 0.1
    assert np.all((-np.pi / 2 <= cloud[:, 1]) & (cloud[:, 1] <= np.pi / 2))


def test_fisher_sky_seed_uses_marginal_not_conditional_covariance():
    class CorrelatedSky(_ToySkyLikelihood):
        def __init__(self):
            # F[:2,:2]^{-1} is diag(0.01), but marginalizing the correlated
            # nuisance coordinate makes var(ra)=1.0.
            fisher = jnp.array([[100.0, 0.0, 9.94987437],
                                [0.0, 100.0, 0.0],
                                [9.94987437, 0.0, 1.0]])
            centre = jnp.array([2.0, 0.2, 1.0])

            def scalar(theta):
                delta = theta - centre
                return -0.5 * delta @ fisher @ delta

            self._scalar = scalar
            self._value_and_grad = jax.jit(jax.value_and_grad(scalar))
            self._hessian = jax.jit(jax.hessian(scalar))
            self.batch_shapes = []

    like = CorrelatedSky()
    callback = samplers._fixed_shape_value_callback(like, 3, 64)
    cloud, modes, _ = samplers._fisher_sky_seed(
        like, like.ANGULAR_PARAM_ORDER, callback, np.random.default_rng(9),
        1.0, 100.0, n_seed=4000, n_pilot=10, n_modes=1,
        sky_inflate=1.0, prior_frac=0.0,
        initial_points=np.array([[2.0, 0.2, 1.0]]))

    dra = np.angle(np.exp(1j * (cloud[:, 0] - modes[0, 0])))
    assert np.std(dra) > 0.5


def test_fixed_distance_likelihood_is_five_dimensional(monkeypatch):
    from RIFT.likelihood.jax_ile import wrapper

    # Avoid constructing physical data: instantiate the class shell with the
    # same public/JAX scalar contract used by the fixed-distance view.
    raw = object.__new__(wrapper.JAXExtrinsicLikelihood)
    raw.data = object()
    raw.interp = "linear"
    raw.phase_marginalization = False
    raw.time_quadrature = "simpson"
    raw._scalar = lambda theta: jnp.sum(theta ** 2)
    raw.log_likelihood = lambda *cols: jnp.sum(jnp.stack(cols) ** 2, axis=0)

    fixed = wrapper.JAXFixedDistanceLikelihood(raw, 17.0)
    theta = np.arange(15, dtype=float).reshape(3, 5) / 10.0
    got = fixed.log_likelihood(*theta.T)
    want = np.sum(theta ** 2, axis=1) + 17.0 ** 2

    assert fixed.ANGULAR_PARAM_ORDER == ("ra", "dec", "psi", "incl", "phiref")
    np.testing.assert_allclose(got, want)
    assert fixed.fisher(theta[0]).shape == (5, 5)


def test_fixed_distance_likelihood_can_center_periodic_phase():
    from RIFT.likelihood.jax_ile import wrapper

    raw = object.__new__(wrapper.JAXExtrinsicLikelihood)
    raw.data = object()
    raw.interp = "linear"
    raw.phase_marginalization = False
    raw.time_quadrature = "simpson"
    raw._scalar = lambda theta: theta[4]
    raw.log_likelihood = lambda *cols: cols[4]

    fixed = wrapper.JAXFixedDistanceLikelihood(raw, 17.0, phase_shift=np.pi)
    physical = np.array([2.0, 0.2, 0.5, 1.0, 0.0])
    sampler_theta = fixed.to_sampler_coordinates(physical)

    assert fixed.ANGULAR_PARAM_ORDER[-1] == "phiref_shifted"
    np.testing.assert_allclose(sampler_theta[-1], np.pi)
    np.testing.assert_allclose(fixed.to_physical_coordinates(sampler_theta), physical)
    np.testing.assert_allclose(fixed.value(sampler_theta), 0.0, atol=1e-12)


def test_rotated_phase_wrapper_roundtrips_and_preserves_likelihood():
    from RIFT.likelihood.jax_ile import wrapper

    raw = object.__new__(wrapper.JAXExtrinsicLikelihood)
    raw.data = object(); raw.interp = "linear"
    raw.phase_marginalization = False; raw.time_quadrature = "simpson"
    raw._scalar = lambda theta: theta[2] + 2.0 * theta[4]
    raw.log_likelihood = lambda ra, dec, psi, incl, phase, dist: psi + 2 * phase
    fixed = wrapper.JAXFixedDistanceLikelihood(raw, 17.0, phase_shift=np.pi)
    rotated = wrapper.JAXRotatedPhaseLikelihood(fixed)
    physical = np.array([1.2, 0.3, 0.5, 1.05, 0.2])
    theta = rotated.to_sampler_coordinates(physical)

    assert rotated.ANGULAR_PARAM_ORDER == (
        "ra", "dec", "phase_p", "incl", "phase_m")
    np.testing.assert_allclose(rotated.to_physical_coordinates(theta), physical)
    np.testing.assert_allclose(rotated.value(theta), 0.9)
    for name in ("phase_p", "phase_m"):
        lo, hi, density = samplers._av_prior_spec(name, 1.0, 100.0)
        assert (lo, hi) == (0.0, 4.0 * np.pi)
        np.testing.assert_allclose(density(np.array([1.0])), 1.0 / (4.0 * np.pi))


def test_av_and_seeded_portfolio_return_driver_contract():
    common = dict(d_min=1.0, d_max=100.0, nmax=20000, neff=25,
                  n_chunk=2000, eval_chunk=512, seed=11, verbose=False)
    av = samplers.adaptive_volume_sample(_ToySkyLikelihood(),
                                         sampler_method="AV", **common)
    portfolio = samplers.adaptive_volume_sample(
        _ToySkyLikelihood(), sampler_method="portfolio",
        seed_method="fisher-sky", seed_pilot=100, seed_modes=2,
        seed_points=500, **common)

    for result in (av, portfolio):
        assert result["theta"].ndim == 2 and result["theta"].shape[1] == 3
        assert len(result["theta"]) == len(result["lnL"])
        assert np.isfinite(result["logZ"])
        assert result["neff"] >= 25
        assert result["n_eval"] <= 20000
        assert result["eval_chunk"] == 512
    assert av["log_weight"] is not None  # weighted retained AV population
    assert portfolio["log_weight"] is None  # portfolio performed its fair draw


def test_pure_av_runs_inside_a_narrow_sky_sampling_window():
    bounds = {"ra": (1.7, 2.3), "dec": (-0.1, 0.5)}
    result = samplers.adaptive_volume_sample(
        _ToySkyLikelihood(), 1.0, 100.0, sampler_method="AV",
        sample_bounds=bounds, nmax=20000, neff=25, n_chunk=2000,
        eval_chunk=512, seed=14)
    assert result["sample_bounds"]["ra"] == bounds["ra"]
    assert np.all((result["theta"][:, 0] >= 1.7) &
                  (result["theta"][:, 0] <= 2.3))
    assert np.all((result["theta"][:, 1] >= -0.1) &
                  (result["theta"][:, 1] <= 0.5))


def test_pure_av_never_evaluates_or_retains_points_outside_sampling_window():
    """A live bin at the upper edge must not extend beyond the declared box."""
    class OutwardRisingLikelihood:
        ANGULAR_PARAM_ORDER = ("ra",)

        def __init__(self):
            self.evaluated = []

        def log_likelihood(self, ra):
            values = np.asarray(ra, dtype=float)
            self.evaluated.append(values.copy())
            # Force the retained live volume against the upper boundary, where
            # fractional bin counts used to let the final bin overshoot.
            return 2000.0 * values

    like = OutwardRisingLikelihood()
    bounds = {"ra": (1.1, 1.3)}
    result = samplers.adaptive_volume_sample(
        like, 1.0, 100.0, sampler_method="AV", sample_bounds=bounds,
        nmax=4000, neff=1000000, n_chunk=400, eval_chunk=128, seed=1409)

    evaluated = np.concatenate(like.evaluated)
    assert np.all((evaluated >= 1.1) & (evaluated <= 1.3))
    assert np.all((result["theta"][:, 0] >= 1.1) &
                  (result["theta"][:, 0] <= 1.3))


def test_caller_supplied_oracle_cloud_bootstraps_portfolio():
    rng = np.random.default_rng(31)
    centre = np.array([2.0, 0.2, 1.0])
    cloud = centre + rng.normal(size=(500, 3)) * np.array([0.2, 0.15, 0.2])
    result = samplers.adaptive_volume_sample(
        _ToySkyLikelihood(), 1.0, 100.0, sampler_method="portfolio",
        initial_samples=cloud, nmax=20000, neff=25, n_chunk=2000,
        eval_chunk=512, seed=31)

    np.testing.assert_array_equal(result["seed_cloud"], cloud)
    assert result["neff"] >= 25


def test_portfolio_gmm_uses_two_components_for_periodic_boundary_modes():
    rng = np.random.default_rng(32)
    cloud = np.column_stack([
        rng.normal(2.0, 0.1, 500), rng.normal(0.2, 0.08, 500),
        rng.normal(1.0, 0.1, 500)])
    result = samplers.adaptive_volume_sample(
        _ToySkyLikelihood(), 1.0, 100.0, sampler_method="portfolio",
        portfolio_members=("GMM",), initial_samples=cloud,
        nmax=10000, neff=20, n_chunk=1000, eval_chunk=256, seed=32)

    gmm = result["sampler"].portfolio_realizations[0]
    assert gmm.integrator.n_comp == 2


def test_oracle_cloud_and_internal_seed_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        samplers.adaptive_volume_sample(
            _ToySkyLikelihood(), 1.0, 100.0, sampler_method="portfolio",
            initial_samples=np.ones((10, 3)), seed_method="fisher-sky",
            nmax=100, neff=2, n_chunk=20)


def test_driver_exposes_sampler_as_an_orthogonal_backend(monkeypatch):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    driver = _driver_module()
    parser = driver.build_parser()
    opts, _ = parser.parse_args([
        "--mode", "flowmc-phipsimarg", "--distance-marginalization",
        "--sampler-method", "portfolio", "--sampler-portfolio", "AV,GMM",
        "--jax-av-seed", "fisher-sky", "--n-eff", "321"])
    driver.check_critical_and_report(opts, parser)

    assert opts.mode == "flowmc-phipsimarg"  # still chooses likelihood geometry
    assert opts.sampler_method == "portfolio"
    assert opts.sampler_portfolio == ["AV,GMM"]
    assert opts.jax_av_seed == "fisher-sky"
    assert opts.n_eff == 321


def test_driver_accepts_pseudo_cosmo_only_for_av_backend(monkeypatch):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    driver = _driver_module()
    parser = driver.build_parser()
    opts, _ = parser.parse_args([
        "--sampler-method", "AV", "--d-prior", "pseudo_cosmo"])
    driver.check_critical_and_report(opts, parser)

    unsupported, _ = parser.parse_args([
        "--sampler-method", "AV", "--d-prior", "cosmo_sourceframe"])
    with pytest.raises(SystemExit):
        driver.check_critical_and_report(unsupported, parser)


def test_driver_validates_and_activates_av_sky_limits(monkeypatch):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    driver = _driver_module()
    parser = driver.build_parser()
    opts, _ = parser.parse_args([
        "--sampler-method", "AV", "--limit-right-ascension", "1.17,1.23",
        "--limit-declination", "0.27,0.33"])
    driver.check_critical_and_report(opts, parser)
    assert driver.resolve_av_angular_limits(opts) == {
        "ra": (1.17, 1.23), "dec": (0.27, 0.33)}

    opts_bad, _ = parser.parse_args([
        "--sampler-method", "AV", "--limit-right-ascension", "6.1,0.1"])
    with pytest.raises(SystemExit):
        driver.check_critical_and_report(opts_bad, parser)


def test_driver_refuses_seed_knobs_without_jax_av_backend(monkeypatch):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    driver = _driver_module()
    parser = driver.build_parser()
    opts, _ = parser.parse_args(["--jax-av-seed", "fisher-sky"])
    with pytest.raises(SystemExit):
        driver.check_critical_and_report(opts, parser)


def test_driver_distance_limits_follow_likelihood_dimension(monkeypatch):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    driver = _driver_module()
    assert driver.av_distance_sampling_kwargs(_ToySkyLikelihood(), 10.0, 90.0) == {}

    class WithDistance:
        ANGULAR_PARAM_ORDER = ("ra", "dec", "distMpc")

    assert driver.av_distance_sampling_kwargs(WithDistance(), 10.0, 90.0) == {
        "sample_d_min": 10.0, "sample_d_max": 90.0}


def test_fisher_sky_seed_respects_restricted_sky_window():
    like = _ToySkyLikelihood()
    callback = samplers._fixed_shape_value_callback(like, 3, 64)
    bounds = {"ra": (1.8, 2.2), "dec": (0.0, 0.4)}
    cloud, _, _ = samplers._fisher_sky_seed(
        like, like.ANGULAR_PARAM_ORDER, callback, np.random.default_rng(71),
        1.0, 100.0, n_seed=400, n_pilot=100, n_modes=1,
        sky_inflate=2.0, prior_frac=0.1, sample_bounds=bounds)
    assert np.all((cloud[:, 0] >= 1.8) & (cloud[:, 0] <= 2.2))
    assert np.all((cloud[:, 1] >= 0.0) & (cloud[:, 1] <= 0.4))
