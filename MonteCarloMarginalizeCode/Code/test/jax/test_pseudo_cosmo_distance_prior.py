#!/usr/bin/env python
"""--d-prior pseudo_cosmo on the JAX ILE distance grids and 6-D prior.

Before this was wired, AV/portfolio + --distance-marginalization + pseudo_cosmo
built every distance grid with d^2 and printed nothing.  Every reference below is
priors_utils.dist_prior_pseudo_cosmo / dist_prior_pseudo_cosmo_eval_norm, the
batchmode ILE's own functions; each test also checks that the result differs
from the volumetric one, so a fallback to d^2 cannot pass.
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

from RIFT.likelihood import priors_utils                      # noqa: E402
from RIFT.likelihood.jax_ile import core as _core             # noqa: E402
from RIFT.likelihood.jax_ile import distance_prior as dp      # noqa: E402
from RIFT.likelihood.jax_ile import samplers as _samplers     # noqa: E402
from RIFT.likelihood.jax_ile.core import (                    # noqa: E402
    make_distance_grid, make_distance_grid_adaptive,
    make_distance_grid_loguniform)
from RIFT.likelihood.jax_ile.wrapper import (                 # noqa: E402
    JAXDistanceMarginalizedLikelihood)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_angle_marg_exact import make_synth                  # noqa: E402

_DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       "bin", "integrate_likelihood_extrinsic_jax")
PC = "pseudo_cosmo"


def _ref_pdf(d, lo, hi):
    """Batchmode's normalized pseudo_cosmo density on [lo, hi]."""
    nm = priors_utils.dist_prior_pseudo_cosmo_eval_norm(lo, hi)
    return priors_utils.dist_prior_pseudo_cosmo(np.asarray(d, dtype=float),
                                                nm=nm, xpy=np)


# --------------------------------------------------------------------------
# Prior object
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lo,hi", [(1.0, 10000.0), (100.0, 50000.0)])
def test_density_is_priors_utils(lo, hi):
    pr = dp.grid_distance_prior_object(PC)
    d = np.geomspace(lo, hi, 501)
    np.testing.assert_allclose(np.exp(pr.log_density(d, lo, hi)),
                               _ref_pdf(d, lo, hi), rtol=1e-13)


def test_jax_traced_density_equals_numpy_and_is_differentiable():
    pr = dp.grid_distance_prior_object(PC)
    d = np.geomspace(5.0, 30000.0, 301)
    f = jax.jit(lambda x: pr.log_density_unnormalized(x, xp=jnp))
    np.testing.assert_allclose(np.asarray(f(jnp.asarray(d))),
                               pr.log_density_unnormalized(d), rtol=1e-13)
    g = np.asarray(jax.vmap(jax.grad(lambda x: pr.log_density_unnormalized(
        x, xp=jnp)))(jnp.asarray(d)))
    c = priors_utils.will_cosmo_const[::-1]
    expect = 2.0 / d - np.polyval(np.polyder(c), d / 1e3) / (
        1e3 * np.polyval(c, d / 1e3))
    np.testing.assert_allclose(g, expect, rtol=1e-12)


def test_av_draws_follow_the_density():
    lo, hi, box = 10.0, 20000.0, (2000.0, 6000.0)
    draws = _samplers._av_distance_prior_draw(
        100000, np.random.default_rng(3), box[0], box[1], lo, hi, PC)
    assert draws.min() >= box[0] and draws.max() <= box[1]
    g = np.linspace(*box, 20001)
    pg = _ref_pdf(g, lo, hi)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pg[1:] + pg[:-1]) * np.diff(g))))
    q = np.interp(0.5 * cdf[-1], cdf, g)
    assert abs(np.mean(draws <= q) - 0.5) < 5e-3
    vol = np.cbrt(np.random.default_rng(3).uniform(box[0] ** 3, box[1] ** 3,
                                                   100000))
    assert abs(np.mean(vol <= q) - 0.5) > 2e-2      # d^2 would fail the check


# --------------------------------------------------------------------------
# Distance grids
# --------------------------------------------------------------------------
def test_uniform_grid_weights_are_pseudo_cosmo():
    lo, hi, n = 10.0, 20000.0, 512
    x, log_w = make_distance_grid(lo, hi, n, d_prior=PC)
    d = np.linspace(lo, hi, n)
    np.testing.assert_allclose(np.asarray(x), _core.DIST_MPC_REF / d, rtol=1e-15)
    p = _ref_pdf(d, lo, hi)
    np.testing.assert_allclose(np.exp(np.asarray(log_w)), p / p.sum(), rtol=1e-12)
    _, log_w_eu = make_distance_grid(lo, hi, n)
    assert np.max(np.abs(np.asarray(log_w) - np.asarray(log_w_eu))) > 0.5


def test_narrowed_grid_carries_the_box_prior_mass():
    lo, hi, box, n = 10.0, 20000.0, (2000.0, 6000.0), 4000
    _, log_w = make_distance_grid(box[0], box[1], n, d_prior=PC,
                                  d_prior_range=(lo, hi))
    mass = np.exp(np.asarray(log_w)).sum()
    g = np.linspace(*box, 200001)
    pg = _ref_pdf(g, lo, hi)
    expect = np.sum(0.5 * (pg[1:] + pg[:-1]) * np.diff(g))
    np.testing.assert_allclose(mass, expect, rtol=2e-3)
    _, log_w_eu = make_distance_grid(box[0], box[1], n, d_prior_range=(lo, hi))
    assert abs(np.exp(np.asarray(log_w_eu)).sum() - expect) > 0.02 * expect


def test_loguniform_and_adaptive_grids_use_pseudo_cosmo():
    lo, hi = 10.0, 20000.0
    for build in (lambda dpr: make_distance_grid_loguniform(lo, hi, 20.0,
                                                            d_prior=dpr),
                  lambda dpr: make_distance_grid_adaptive(lo, hi, 3000.0, 300.0,
                                                          d_prior=dpr)):
        x, log_w = build(PC)
        d = _core.DIST_MPC_REF / np.asarray(x)
        _, log_w_eu = build("euclidean")
        # same nodes, so the log-weight difference is ln(p_pc/d^2) + const
        diff = np.asarray(log_w) - np.asarray(log_w_eu)
        expect = np.log(_ref_pdf(d, lo, hi)) - 2.0 * np.log(d)
        np.testing.assert_allclose(diff - diff[0], expect - expect[0],
                                   rtol=0, atol=1e-10)
        assert np.ptp(expect) > 0.5


def test_distmarg_likelihood_matches_priors_utils_quadrature():
    data = make_synth(scale=100.0, kappa_boost=20.0)   # best fit ~2 Gpc
    rng = np.random.default_rng(7)
    n = 24
    a5 = (rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n)),
          rng.uniform(0, np.pi, n), np.arccos(rng.uniform(-1, 1, n)),
          rng.uniform(0, 2 * np.pi, n))
    lo, hi = 1.0, 20000.0
    like = JAXDistanceMarginalizedLikelihood(data, lo, hi, n_grid=4000,
                                             d_prior=PC)
    lnL = np.asarray(like.log_likelihood(*a5))
    d = float(data.distMpcRef) / np.asarray(like.x_grid)
    w = _ref_pdf(d, lo, hi)
    ref = np.asarray(_core.fused_log_likelihood_distmarg(
        data, *[jnp.asarray(c) for c in a5], like.x_grid,
        jnp.asarray(np.log(w / w.sum())),
        time_quadrature=like.time_quadrature))
    np.testing.assert_allclose(lnL, ref, rtol=0, atol=1e-10)
    eu = np.asarray(JAXDistanceMarginalizedLikelihood(
        data, lo, hi, n_grid=4000).log_likelihood(*a5))
    assert np.max(np.abs(lnL - eu)) > 0.05     # the prior actually moved lnL


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def driver():
    loader = importlib.machinery.SourceFileLoader("_jax_pc_driver", _DRIVER)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["AV", "portfolio"])
def test_driver_av_distmarg_forwards_pseudo_cosmo(
        driver, monkeypatch, capsys, method):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    parser = driver.build_parser()
    opts, _ = parser.parse_args(["--sampler-method", method,
                                 "--distance-marginalization",
                                 "--d-prior", PC])
    driver.check_critical_and_report(opts, parser)
    assert "--d-prior" not in capsys.readouterr().out
    assert driver.grid_distance_prior(opts) == PC


@pytest.mark.parametrize("args", [
    ["--distance-gh-nodes", "16"],
    ["--angle-marg-scheme", "multipeak"],
    ["--angle-marg-scheme", "multipeak-jax"]])
def test_driver_refuses_volumetric_only_schemes(driver, monkeypatch, capsys,
                                                args):
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    parser = driver.build_parser()
    opts, _ = parser.parse_args(["--d-prior", PC] + args)
    old = _core.get_distmarg_gh_nodes()
    try:
        with pytest.raises(SystemExit):
            driver.check_critical_and_report(opts, parser)
        assert "--d-prior pseudo_cosmo" in capsys.readouterr().err
    finally:
        _core.set_distmarg_gh_nodes(old)


@pytest.mark.parametrize("limit", [None, "2000,6000"])
def test_driver_six_d_prior_is_priors_utils(driver, limit):
    parser = driver.build_parser()
    args = ["--d-prior", PC, "--d-min", "10", "--d-max", "20000"]
    if limit:
        args += ["--limit-distance", limit]
    opts, _ = parser.parse_args(args)
    lo, hi = driver.resolve_distance_limit(opts)
    theta, logp = driver.sample_prior(100000, opts, np.random.default_rng(5),
                                      True)
    d = theta[:, 5]
    assert d.min() >= lo and d.max() <= hi
    ang = driver.log_prior(theta, driver.build_parser().parse_args(
        ["--d-min", "10", "--d-max", "20000"])[0], False)
    np.testing.assert_allclose(logp - ang, np.log(_ref_pdf(d, 10.0, 20000.0)),
                               rtol=0, atol=1e-12)
    corr = driver.log_distance_box_correction(opts, True)
    expect = np.log(priors_utils.dist_prior_pseudo_cosmo_eval_norm(lo, hi)
                    / priors_utils.dist_prior_pseudo_cosmo_eval_norm(10.0, 20000.0))
    np.testing.assert_allclose(corr, expect, rtol=0, atol=1e-12)
    if limit:
        assert abs(corr - np.log((20000.0 ** 3 - 10.0 ** 3)
                                 / (hi ** 3 - lo ** 3))) > 0.05
    g = np.linspace(lo, hi, 20001)
    pg = _ref_pdf(g, 10.0, 20000.0)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pg[1:] + pg[:-1]) * np.diff(g))))
    q = np.interp(0.5 * cdf[-1], cdf, g)
    assert abs(np.mean(d <= q) - 0.5) < 5e-3
