#!/usr/bin/env python
"""--d-prior cosmo / cosmo_sourceframe on the JAX ILE.

The tabulated prior (RIFT.likelihood.jax_ile.distance_prior) is checked against
two references that do not use its table: the closed-form density from astropy
with a finite-difference dd_L/dz, and the batchmode ILE's own construction
(priors_utils.norm_and_inverse_via_grid_interp).  The marginalized likelihood is
checked against the same kernel fed quadrature weights built directly in
redshift, so neither the table nor z(d_L) enters the reference.
"""

import importlib.machinery
import importlib.util
import os
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                       # noqa: E402

pytest.importorskip("astropy")
import astropy.units as u                                     # noqa: E402
from astropy.cosmology import z_at_value                      # noqa: E402

from RIFT.likelihood import priors_utils                      # noqa: E402
from RIFT.likelihood.jax_ile import core as _core             # noqa: E402
from RIFT.likelihood.jax_ile import distance_prior as dp      # noqa: E402
from RIFT.likelihood.jax_ile import samplers as _samplers     # noqa: E402
from RIFT.likelihood.jax_ile.core import (                    # noqa: E402
    make_distance_grid, make_distance_grid_loguniform)
from RIFT.likelihood.jax_ile.wrapper import (                 # noqa: E402
    JAXDistanceMarginalizedLikelihood, JAXDistPhiMargLikelihood)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_angle_marg_exact import make_synth                  # noqa: E402

KINDS = dp.COSMO_DISTANCE_PRIORS
COSMO = priors_utils.get_astropy_cosmology("Planck15")
_DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       "bin", "integrate_likelihood_extrinsic_jax")


def _trapezoid(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x)


def _s(kind):
    return 1.0 if kind == "cosmo_sourceframe" else 0.0


def _dVdz(z, kind):
    return COSMO.differential_comoving_volume(z).value / (1.0 + z) ** _s(kind)


def _z_of_d(d):
    return np.array([z_at_value(COSMO.luminosity_distance, x * u.Mpc).value
                     for x in np.atleast_1d(d)])


def _exact_density(d, kind, lo, hi):
    """p(d) from astropy alone: dVc/dz (1+z)^-s / (dd_L/dz), normalized in z."""
    z = _z_of_d(d)
    h = 1e-6 * z
    ddl = (COSMO.luminosity_distance(z + h).value
           - COSMO.luminosity_distance(z - h).value) / (2 * h)
    zz = np.linspace(_z_of_d(lo)[0], _z_of_d(hi)[0], 200001)
    norm = _trapezoid(_dVdz(zz, kind), zz)
    return _dVdz(z, kind) / ddl / norm


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("lo,hi", [(1.0, 10000.0), (100.0, 50000.0)])
def test_density_matches_astropy_closed_form(kind, lo, hi):
    pr = dp.cosmo_distance_prior(kind)
    d = np.geomspace(lo * 1.01, hi * 0.99, 25)
    mine = np.exp(pr.log_density(d, lo, hi))
    np.testing.assert_allclose(mine, _exact_density(d, kind, lo, hi), rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_density_matches_batchmode_ile_construction(kind):
    lo, hi = 10.0, 20000.0
    zmin, zmax = _z_of_d(lo)[0], _z_of_d(hi)[0]
    pdf, _, _ = priors_utils.norm_and_inverse_via_grid_interp(
        lambda z: _dVdz(z, kind), [zmin, zmax], vectorized=True,
        y_of_x=lambda z: COSMO.luminosity_distance(z).value)
    d = np.geomspace(lo * 1.01, hi * 0.99, 500)
    mine = np.exp(dp.cosmo_distance_prior(kind).log_density(d, lo, hi))
    # the ILE differentiates a 1000-node PCHIP CDF; 3e-5 is its own accuracy
    np.testing.assert_allclose(mine, pdf(d), rtol=1e-4)


def test_sourceframe_is_cosmo_over_one_plus_z():
    """Guards against the (1+z) factor being dropped from either kind."""
    d = np.geomspace(5.0, 40000.0, 50)
    a = dp.cosmo_distance_prior("cosmo").log_density_unnormalized(d)
    b = dp.cosmo_distance_prior("cosmo_sourceframe").log_density_unnormalized(d)
    np.testing.assert_allclose(a - b, np.log1p(_z_of_d(d)), atol=2e-6)


def test_low_redshift_limit_is_volumetric():
    pr = dp.cosmo_distance_prior("cosmo_sourceframe")
    d = np.array([1e-6, 1e-4, 1e-3])   # deviation from d^2 is O(z)
    np.testing.assert_allclose(pr.log_density_unnormalized(d), 2 * np.log(d),
                               atol=1e-5)


def test_refuses_distances_beyond_the_table():
    pr = dp.cosmo_distance_prior("cosmo")
    with pytest.raises(ValueError, match="tabulated"):
        pr.log_mass(1.0, 10 * pr.d_table_max)


@pytest.mark.parametrize("kind", KINDS)
def test_jax_traced_density_equals_numpy_and_is_differentiable(kind):
    pr = dp.cosmo_distance_prior(kind)
    d = np.geomspace(2.0, 30000.0, 101)
    f = jax.jit(lambda x: pr.log_density_unnormalized(x, xp=jnp))
    np.testing.assert_allclose(np.asarray(f(jnp.asarray(d))),
                               pr.log_density_unnormalized(d), rtol=0, atol=1e-12)
    g = jax.vmap(jax.grad(lambda x: pr.log_density_unnormalized(x, xp=jnp)))(
        jnp.asarray(d))
    h = 1e-3 * d
    fd = (pr.log_density_unnormalized(d + h)
          - pr.log_density_unnormalized(d - h)) / (2 * h)
    # d * dlnp/dd is O(1); the interpolant's slope is piecewise constant
    np.testing.assert_allclose(np.asarray(g) * d, fd * d, rtol=0, atol=2e-3)


@pytest.mark.parametrize("kind", KINDS)
def test_draws_follow_the_density(kind):
    pr = dp.cosmo_distance_prior(kind)
    lo, hi = 50.0, 20000.0
    draws = pr.sample(200000, np.random.default_rng(1), lo, hi)
    assert draws.min() >= lo and draws.max() <= hi
    grid = np.linspace(lo, hi, 20001)
    p = np.exp(pr.log_density(grid, lo, hi))
    for q in (0.1, 0.5, 0.9):
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1])
                                               * np.diff(grid))))
        assert abs(np.mean(draws <= np.interp(q, cdf, grid)) - q) < 4e-3


# --------------------------------------------------------------------------
# Quadrature grids
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_uniform_grid_weights_are_the_cosmological_prior(kind):
    lo, hi, n = 1.0, 20000.0, 4000
    x, lw = make_distance_grid(lo, hi, n, kind, distMpcRef=100.0)
    d = 100.0 / np.asarray(x)
    w = np.exp(np.asarray(lw))
    assert abs(w.sum() - 1.0) < 1e-12
    # the prior mean distance, against the redshift-space reference
    zz = np.linspace(_z_of_d(lo)[0], _z_of_d(hi)[0], 200001)
    pz = _dVdz(zz, kind)
    ref = np.sum(pz * COSMO.luminosity_distance(zz).value) / np.sum(pz)
    assert abs(np.sum(w * d) / ref - 1.0) < 1e-3
    # and it is not the volumetric answer
    xe, lwe = make_distance_grid(lo, hi, n, "euclidean", distMpcRef=100.0)
    de = 100.0 / np.asarray(xe)
    assert np.sum(np.exp(np.asarray(lwe)) * de) / ref - 1.0 > 0.05


@pytest.mark.parametrize("kind", KINDS)
def test_narrowed_grid_carries_the_box_prior_mass(kind):
    lo, hi = 1.0, 20000.0
    box = (2000.0, 6000.0)
    _, lw = make_distance_grid(box[0], box[1], 4000, kind, distMpcRef=100.0,
                               d_prior_range=(lo, hi))
    pr = dp.cosmo_distance_prior(kind)
    mass = np.exp(pr.log_mass(*box) - pr.log_mass(lo, hi))
    assert abs(np.exp(np.asarray(lw)).sum() / mass - 1.0) < 2e-3


@pytest.mark.parametrize("kind", KINDS)
def test_loguniform_grid_accepts_cosmological_prior(kind):
    x, lw = make_distance_grid_loguniform(10.0, 20000.0, rho_max=30.0,
                                          d_prior=kind, distMpcRef=100.0)
    d = 100.0 / np.asarray(x)
    w = np.exp(np.asarray(lw))
    pr = dp.cosmo_distance_prior(kind)
    ref = np.exp(pr.log_density(d, 10.0, 20000.0))
    # interior nodes: trapezoid weight / spacing is the normalized density
    dd = 0.5 * (d[2:] - d[:-2])
    np.testing.assert_allclose(w[1:-1] / dd, ref[1:-1], rtol=1e-3)


# --------------------------------------------------------------------------
# Marginalized likelihood against an independently built redshift quadrature
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _far_synth():
    """A synthetic whose best-fit distance is ~2 Gpc, where cosmology matters."""
    data = make_synth(scale=100.0, kappa_boost=20.0)   # d* ~ 397 scale/boost Mpc
    rng = np.random.default_rng(7)
    n = 24
    a5 = (rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n)),
          rng.uniform(0, np.pi, n), np.arccos(rng.uniform(-1, 1, n)),
          rng.uniform(0, 2 * np.pi, n))
    K, R = _core._accumulate_unit(data, *a5, _core.JAX_INTERP_DEFAULT, False)
    K, R = np.asarray(K.real), np.maximum(np.asarray(R), 1e-30)
    snr2 = np.where(K > 0, K * K / R, -np.inf)
    i, j = np.unravel_index(int(np.argmax(snr2)), snr2.shape)
    d_star = float(data.distMpcRef) * R[i, j] / K[i, j]
    return data, a5, d_star


def test_far_synth_premise(_far_synth):
    _, _, d_star = _far_synth
    assert 500.0 < d_star < 8000.0, d_star


@pytest.mark.parametrize("kind", KINDS)
def test_distmarg_likelihood_matches_redshift_quadrature(kind, _far_synth):
    data, a5, _ = _far_synth
    lo, hi = 1.0, 20000.0
    like = JAXDistanceMarginalizedLikelihood(data, lo, hi, n_grid=4000,
                                             d_prior=kind)
    lnL = np.asarray(like.log_likelihood(*a5))
    # reference: same kernel and nodes, weights from astropy alone (z(d) by
    # inverting a fine z grid, dd_L/dz by finite difference; no table)
    d = float(data.distMpcRef) / np.asarray(like.x_grid)
    zf = np.geomspace(1e-7, _z_of_d(hi * 1.01)[0], 400001)
    z = np.interp(d, COSMO.luminosity_distance(zf).value, zf)
    h = 1e-6 * z
    ddl = (COSMO.luminosity_distance(z + h).value
           - COSMO.luminosity_distance(z - h).value) / (2 * h)
    w = _dVdz(z, kind) / ddl
    ref = np.asarray(_core.fused_log_likelihood_distmarg(
        data, *[jnp.asarray(c) for c in a5], like.x_grid,
        jnp.asarray(np.log(w / w.sum())),
        time_quadrature=like.time_quadrature))
    np.testing.assert_allclose(lnL, ref, rtol=0, atol=1e-5)
    eu = np.asarray(JAXDistanceMarginalizedLikelihood(
        data, lo, hi, n_grid=4000).log_likelihood(*a5))
    assert np.max(np.abs(lnL - eu)) > 0.05     # the prior actually moved lnL


def test_gh_distance_quadrature_refuses_cosmological_prior():
    data = make_synth()
    old = _core.get_distmarg_gh_nodes()
    try:
        _core.set_distmarg_gh_nodes(16)
        with pytest.raises(ValueError, match="volumetric prior only"):
            JAXDistPhiMargLikelihood(data, 1.0, 1000.0, nphi=8, n_grid=64,
                                     d_prior="cosmo_sourceframe")
        JAXDistPhiMargLikelihood(data, 1.0, 1000.0, nphi=8, n_grid=64)
    finally:
        _core.set_distmarg_gh_nodes(old)


# --------------------------------------------------------------------------
# AV / portfolio
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_av_density_and_draw_are_the_cosmological_prior(kind):
    lo, hi = 10.0, 20000.0
    a, b, pdf = _samplers._av_prior_spec("distMpc", lo, hi, distance_prior=kind)
    assert (a, b) == (lo, hi)
    grid = np.linspace(lo, hi, 40001)
    p = pdf(grid)
    assert abs(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(grid)) - 1.0) < 1e-6
    np.testing.assert_allclose(
        p[::4000], np.exp(dp.cosmo_distance_prior(kind).log_density(
            grid[::4000], lo, hi)), rtol=1e-12)
    draws = _samplers._av_distance_prior_draw(
        100000, np.random.default_rng(3), 2000.0, 6000.0, lo, hi, kind)
    assert draws.min() >= 2000.0 and draws.max() <= 6000.0


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _driver_module():
    loader = importlib.machinery.SourceFileLoader("_jax_cosmo_driver", _DRIVER)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _driver_module()


@pytest.mark.parametrize("value,expect", [
    (None, "euclidean"), ("Euclidean", "euclidean"), ("volumetric", "euclidean"),
    ("cosmo", "cosmo"), ("cosmo_sourceframe", "cosmo_sourceframe"),
    ("COSMO_SOURCEFRAME", "cosmo_sourceframe"), ("pseudo_cosmo", "euclidean")])
def test_driver_grid_distance_prior(driver, value, expect):
    parser = driver.build_parser()
    args = [] if value is None else ["--d-prior", value]
    opts, _ = parser.parse_args(args)
    assert driver.grid_distance_prior(opts) == expect


@pytest.mark.parametrize("method", ["AV", "portfolio"])
@pytest.mark.parametrize("kind", KINDS)
def test_driver_accepts_cosmological_prior_for_av(driver, monkeypatch, method,
                                                   kind):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    parser = driver.build_parser()
    opts, _ = parser.parse_args(["--sampler-method", method, "--d-prior", kind])
    driver.check_critical_and_report(opts, parser)


def test_driver_no_longer_reports_cosmo_as_ignored(driver, monkeypatch, capsys):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    parser = driver.build_parser()
    opts, _ = parser.parse_args(["--d-prior", "cosmo_sourceframe"])
    driver.check_critical_and_report(opts, parser)
    assert "--d-prior 'cosmo_sourceframe' is accepted but IGNORED" not in \
        capsys.readouterr().out
    opts, _ = parser.parse_args(["--d-prior", "pseudo_cosmo"])
    driver.check_critical_and_report(opts, parser)
    assert "--d-prior 'pseudo_cosmo' is accepted but IGNORED" in \
        capsys.readouterr().out


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("limit", [None, "2000,6000"])
def test_driver_six_d_prior_is_normalized_and_matches_its_draws(driver, kind,
                                                                 limit):
    parser = driver.build_parser()
    args = ["--d-prior", kind, "--d-min", "10", "--d-max", "20000"]
    if limit:
        args += ["--limit-distance", limit]
    opts, _ = parser.parse_args(args)
    lo, hi = driver.resolve_distance_limit(opts)
    theta, logp = driver.sample_prior(100000, opts, np.random.default_rng(5),
                                      True)
    d = theta[:, 5]
    assert d.min() >= lo and d.max() <= hi
    # log_prior's distance factor integrates to the box mass over [d_min,d_max]
    grid = np.linspace(lo, hi, 40001)
    th = np.tile(theta[:1], (grid.size, 1))
    th[:, 5] = grid
    lp = driver.log_prior(th, opts, True) - driver.log_prior(
        th[:, :5], opts, False)
    p = np.exp(lp)
    mass = np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))
    assert abs(np.log(mass) + driver.log_distance_box_correction(opts, True)) < 1e-6
    # the draws follow log_prior: median of draws vs its CDF
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))))
    med = np.interp(0.5 * cdf[-1], cdf, grid)
    assert abs(np.mean(d <= med) - 0.5) < 5e-3
    np.testing.assert_allclose(logp, driver.log_prior(theta, opts, True))
