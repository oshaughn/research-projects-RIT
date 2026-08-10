"""Tests for RIFT.interpolators.nal_io -- the generic NAL reader/evaluator.

Everything is checked against an answer known analytically or by brute force; nothing is
self-referential.
"""
import json
import os
import sys

import numpy as np
import pytest

# Import by PATH, not as RIFT.interpolators.nal_io: importing the package pulls in lalsimutils
# and hence glue/lal, which this module does not need.  nal_io is deliberately pure-numpy (h5py
# only for the optional gwalk view), and loading it standalone here both keeps the test runnable
# in a bare environment and asserts that independence.
import importlib.util as _ilu                                           # noqa: E402
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                  "RIFT", "interpolators", "nal_io.py")
_spec = _ilu.spec_from_file_location("nal_io", os.path.abspath(_p))
nal_io = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(nal_io)


def _make(d=4, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    G = A @ A.T + d * np.eye(d)
    mu = rng.standard_normal(d)
    return mu, G


def test_lnL_matches_the_quadratic_form():
    mu, G = _make()
    n = nal_io.NAL(mu, G, ["mc", "delta_mc", "xi", "chiMinus"], lnL_peak=7.5)
    X = mu + np.random.default_rng(1).standard_normal((50, 4)) * 0.3
    want = 7.5 - 0.5 * np.einsum("ij,jk,ik->i", X - mu, G, X - mu)
    assert np.allclose(n.lnL(X), want, atol=1e-12)
    assert np.isclose(n.lnL(mu)[0], 7.5)


def test_marginal_is_the_schur_complement_not_the_conditional():
    """The distinction that is easy to get backwards: marginal = sub-block of Gamma^-1."""
    mu, G = _make(d=5, seed=3)
    n = nal_io.NAL(mu, G, list("abcde"))
    m = n.marginal(["a", "b"])
    Sig = np.linalg.inv(G)
    assert np.allclose(m.cov(), Sig[np.ix_([0, 1], [0, 1])], atol=1e-12)
    conditional = np.linalg.inv(G[np.ix_([0, 1], [0, 1])])
    # they must genuinely differ, else the test proves nothing
    assert not np.allclose(m.cov(), conditional, atol=1e-6)
    # and the marginal must be the WIDER of the two
    assert np.all(np.diag(m.cov()) > np.diag(conditional))


def test_bounds_zero_the_likelihood_outside():
    mu, G = _make(d=2, seed=5)
    b = np.stack([mu - 0.5, mu + 0.5], 1)
    n = nal_io.NAL(mu, G, ["mc", "eta"], bounds=b)
    assert np.isfinite(n.lnL(mu)[0])
    assert n.lnL(mu + 10.0)[0] == -np.inf


def test_renormalization_is_not_a_product_of_1d_marginals():
    """A correlated truncated Gaussian: the factorised mass is biased, the MC one is not."""
    mu = np.zeros(3)
    rho = 0.9
    C = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
    n = nal_io.NAL(mu, np.linalg.inv(C), list("abc"),
                   bounds=np.stack([-np.ones(3), np.ones(3)], 1))
    logm = n._log_mass(n=400000, seed=2)
    from scipy.stats import norm
    factorised = np.log(np.prod([norm.cdf(1) - norm.cdf(-1)] * 3))
    assert np.exp(logm) > np.exp(factorised) * 1.2      # correlation concentrates mass in the box
    # and the MC mass must agree with an independent brute-force estimate
    G = np.random.default_rng(9).multivariate_normal(mu, C, 400000)
    brute = np.log(np.all(np.abs(G) <= 1, axis=1).mean())
    assert abs(logm - brute) < 0.02


def test_roundtrip_artifact(tmp_path):
    mu, G = _make()
    names = ["mc", "delta_mc", "xi", "chiMinus"]
    base = str(tmp_path / "ev1")
    np.savez(base + ".npz", theta_star=mu, gamma=G,
             bounds=np.stack([mu - 5, mu + 5], 1))
    json.dump({"coord_names": names, "lnL_peak": 3.25, "chart": "NAL:aligned",
               "frame": "detector"}, open(base + ".meta.json", "w"))
    n = nal_io.load_nal(base + ".npz")
    assert n.coord_names == names and np.isclose(n.lnL_peak, 3.25)
    assert np.allclose(n.lnL(mu)[0], 3.25)


def test_meta_without_coord_names_is_rejected(tmp_path):
    """A NAL with no declared chart is uninterpretable and must not load silently."""
    mu, G = _make(d=2)
    base = str(tmp_path / "bad")
    np.savez(base + ".npz", theta_star=mu, gamma=G)
    json.dump({"lnL_peak": 0.0}, open(base + ".meta.json", "w"))
    with pytest.raises(KeyError):
        nal_io.load_nal(base + ".npz")


def test_plugin_hook_contract(tmp_path, monkeypatch):
    """Exercise exactly what CIP/EOSPosterior do: prepare(config, coords) then lnL(*x)."""
    mu, G = _make(d=2, seed=11)
    names = ["mc", "delta_mc"]
    for i in range(2):                                   # two events -> contributions ADD
        base = str(tmp_path / ("ev%d" % i))
        np.savez(base + ".npz", theta_star=mu, gamma=G)
        json.dump({"coord_names": names, "lnL_peak": 1.0}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", str(tmp_path / "*.npz"))
    nal_io._STATE.update(set=None, coords=None, renormalize=False)
    nal_io.prepare_nal_lnL(config=None, coords=names)
    out = nal_io.nal_lnL(np.array([mu[0]]), np.array([mu[1]]))
    assert np.isclose(out[0], 2.0)                       # 1.0 per event, summed


def test_plugin_derives_delta_mc_from_eta(tmp_path, monkeypatch):
    """Sampler in (mc, eta); artifact chart in (mc, delta_mc).  Must convert, not fail."""
    mu = np.array([30.0, 0.3])                           # delta_mc = 0.3 -> eta = 0.2275
    G = np.diag([1.0, 4.0])
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=G)
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": 0.0},
              open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    nal_io._STATE.update(set=None, coords=None, renormalize=False)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "eta"])
    eta = 0.25 * (1 - 0.3 ** 2)
    out = nal_io.nal_lnL(np.array([30.0]), np.array([eta]))
    assert np.isclose(out[0], 0.0, atol=1e-10)           # lands exactly on the peak


def test_unbuildable_coordinate_raises_named_error(tmp_path, monkeypatch):
    mu, G = _make(d=2, seed=2)
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=G)
    json.dump({"coord_names": ["mc", "s1x_bar"], "lnL_peak": 0.0},
              open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    nal_io._STATE.update(set=None, coords=None, renormalize=False)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "eta"])
    with pytest.raises(KeyError, match="s1x_bar"):
        nal_io.nal_lnL(np.array([30.0]), np.array([0.2]))
