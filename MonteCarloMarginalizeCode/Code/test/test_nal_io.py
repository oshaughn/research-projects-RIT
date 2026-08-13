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


def _declare_run(monkeypatch, frame="detector", chart="NAL:aligned"):
    """Declare the frame and chart the RUN samples in, as an ini-less deployment would."""
    monkeypatch.setenv("RIFT_NAL_SAMPLER_FRAME", frame)
    monkeypatch.setenv("RIFT_NAL_SAMPLER_CHART", chart)


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
    logm = n.log_mass(seed=2)
    from scipy.stats import norm
    factorised = np.log(np.prod([norm.cdf(1) - norm.cdf(-1)] * 3))
    assert np.exp(logm) > np.exp(factorised) * 1.2      # correlation concentrates mass in the box
    # and the MC mass must agree with an independent brute-force estimate
    G = np.random.default_rng(9).multivariate_normal(mu, C, 400000)
    brute = np.log(np.all(np.abs(G) <= 1, axis=1).mean())
    assert abs(logm - brute) < 0.02


def test_truncation_constant_is_computed_once_and_reused():
    """`renormalize=True` must not re-run the Monte Carlo on every likelihood call."""
    mu = np.zeros(2)                                     # mass in the box ~ 0.466, comfortably < 1
    n = nal_io.NAL(mu, np.eye(2), ["mc", "eta"], bounds=np.stack([mu - 1.0, mu + 1.0], 1))

    calls = {"n": 0}
    real = nal_io._rng

    def counting_rng(seed):
        calls["n"] += 1
        return real(seed)

    nal_io._rng = counting_rng
    try:
        first = n.log_mass()
        for _ in range(5):
            n.lnL(mu, renormalize=True)
        assert calls["n"] == 1, "truncation mass recomputed inside lnL: %d draws sets" % calls["n"]
    finally:
        nal_io._rng = real
    assert n.log_mass() == first < 0.0


def test_unresolvable_truncation_mass_raises_rather_than_guessing():
    """A mass too small to estimate must fail, not come back floored at 1/n.

    1-D standard normal on [6, 7]: true mass ~1e-9, so no affordable number of draws lands in the
    box.  The old floor returned ~5e-6 -- an 8.5 nat error presented as a measurement.
    """
    n = nal_io.NAL([0.0], [[1.0]], ["mc"], bounds=[[6.0, 7.0]])
    with pytest.raises(ValueError, match="unresolved"):
        n.log_mass(max_draws=200000, batch=100000)
    with pytest.raises(ValueError, match="unresolved"):
        n.lnL(np.array([[6.5]]), renormalize=True)
    # ... unless the artifact declares the value it knows
    n.meta["log_truncation_mass"] = -20.6
    assert np.isclose(n.log_mass(), -20.6)
    assert np.isclose(n.lnL(np.array([[6.5]]), renormalize=True)[0], -0.5 * 6.5 ** 2 + 20.6)


def test_log_mass_is_accurate_for_a_small_but_reachable_mass():
    """Against the analytic answer, where the old fixed-n floor would have been consulted."""
    from scipy.stats import norm
    n = nal_io.NAL([0.0], [[1.0]], ["mc"], bounds=[[3.0, 4.0]])
    want = np.log(norm.cdf(4.0) - norm.cdf(3.0))        # ~ -6.8
    got = n.log_mass(rel_tol=0.02, max_draws=8000000)
    assert abs(got - want) < 0.1


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
        json.dump({"coord_names": names, "lnL_peak": 1.0, "chart": "NAL:aligned",
                   "frame": "detector"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", str(tmp_path / "*.npz"))
    _declare_run(monkeypatch)
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    nal_io.prepare_nal_lnL(config=None, coords=names)
    out = nal_io.nal_lnL(np.array([mu[0]]), np.array([mu[1]]))
    # 1.0 per event, summed -- then centred by the summed peak, so the peak sits at 0
    assert np.isclose(nal_io.nal_lnL_offset(), 2.0)
    assert np.isclose(out[0], 0.0)


def test_plugin_derives_delta_mc_from_eta(tmp_path, monkeypatch):
    """Sampler in (mc, eta); artifact chart in (mc, delta_mc).  Must convert, not fail."""
    mu = np.array([30.0, 0.3])                           # delta_mc = 0.3 -> eta = 0.2275
    G = np.diag([1.0, 4.0])
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=G)
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": 0.0, "frame": "detector",
               "chart": "NAL:aligned"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    _declare_run(monkeypatch)
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "eta"])
    eta = 0.25 * (1 - 0.3 ** 2)
    out = nal_io.nal_lnL(np.array([30.0]), np.array([eta]))
    assert np.isclose(out[0], 0.0, atol=1e-10)           # lands exactly on the peak


def test_unbuildable_coordinate_raises_named_error(tmp_path, monkeypatch):
    mu, G = _make(d=2, seed=2)
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=G)
    json.dump({"coord_names": ["mc", "s1x_bar"], "lnL_peak": 0.0, "frame": "detector",
               "chart": "NAL:aligned"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    _declare_run(monkeypatch)
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "eta"])
    with pytest.raises(KeyError, match="s1x_bar"):
        nal_io.nal_lnL(np.array([30.0]), np.array([0.2]))


def test_wrong_basis_from_the_driver_would_evaluate_at_the_wrong_point(tmp_path, monkeypatch):
    """`coords` must be the driver's SAMPLING basis, which is not its FIT basis.

    Concretely, `--parameter mc --parameter eta --parameter-implied delta_mc
    --parameter-nofit s1z` gives fit coord_names = [mc, eta, delta_mc] but sampling
    low_level_coord_names = [mc, eta, s1z], and the sampler calls the plugin with one array per
    SAMPLING coordinate.  Declaring the fit basis zips 'delta_mc' onto the s1z array: same length,
    no error, silently the wrong point.  This test pins both halves -- the right basis lands on
    the peak, the wrong one does not -- so the failure mode stays visible rather than numerical.
    """
    mu = np.array([30.0, 0.3])                           # chart is (mc, delta_mc)
    G = np.diag([1.0, 4.0])
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=G)
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": 0.0, "frame": "detector",
               "chart": "NAL:aligned"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    _declare_run(monkeypatch)

    mc, eta, s1z = 30.0, 0.25 * (1 - 0.3 ** 2), -0.4     # delta_mc = 0.3 exactly
    x = [np.array([mc]), np.array([eta]), np.array([s1z])]

    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "eta", "s1z"])          # sampling basis
    assert np.isclose(nal_io.nal_lnL(*x)[0], 0.0, atol=1e-10)

    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "eta", "delta_mc"])     # fit basis: WRONG
    wrong = nal_io.nal_lnL(*x)[0]
    # s1z has been read as delta_mc: lnL = -1/2 * 4 * (s1z - 0.3)^2
    assert np.isclose(wrong, -0.5 * 4.0 * (s1z - 0.3) ** 2)
    assert wrong < -0.5                                  # and it is nowhere near the peak


def test_environment_only_configuration_will_not_guess_the_basis(tmp_path, monkeypatch):
    """RIFT_NAL_ARTIFACTS alone must fail closed: the incoming arrays have no names.

    The dangerous case has the RIGHT number of arrays, so no dimension check can catch it: a
    sampler in (mc, eta) against an artifact in (mc, delta_mc) would evaluate eta as delta_mc.
    """
    mu = np.array([30.0, 0.3])
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=np.diag([1.0, 4.0]))
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": 0.0, "frame": "detector",
               "chart": "NAL:aligned"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    monkeypatch.delenv("RIFT_NAL_SAMPLER_COORDS", raising=False)
    _declare_run(monkeypatch)

    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    eta = 0.25 * (1 - 0.3 ** 2)
    with pytest.raises(ValueError, match="sampling basis is unknown"):
        nal_io.nal_lnL(np.array([30.0]), np.array([eta]))

    # declaring the basis explicitly is the supported way out, and it then converts eta -> delta_mc
    monkeypatch.setenv("RIFT_NAL_SAMPLER_COORDS", "mc, eta")
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    out = nal_io.nal_lnL(np.array([30.0]), np.array([eta]))
    assert np.isclose(out[0], 0.0, atol=1e-10)           # lands on the peak, not off it


def test_driver_coords_override_the_environment_declaration(tmp_path, monkeypatch):
    """The driver knows the real sampling basis; a stale environment value must not win."""
    mu = np.array([30.0, 0.3])
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=np.diag([1.0, 4.0]))
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": 0.0, "frame": "detector",
               "chart": "NAL:aligned"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    monkeypatch.setenv("RIFT_NAL_SAMPLER_COORDS", "mc,delta_mc")
    _declare_run(monkeypatch)
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "eta"])
    assert nal_io._STATE["coords"] == ["mc", "eta"]


def test_wrong_number_of_arrays_is_rejected(tmp_path, monkeypatch):
    mu = np.array([30.0, 0.3])
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=np.diag([1.0, 4.0]))
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": 0.0, "frame": "detector",
               "chart": "NAL:aligned"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    _declare_run(monkeypatch)
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "eta"])
    with pytest.raises(ValueError, match="sampling basis"):
        nal_io.nal_lnL(np.array([30.0]))


# ------------------------------------------------------------------- summing artifacts / charts

def _nal(meta, seed=0):
    mu, G = _make(d=2, seed=seed)
    return nal_io.NAL(mu, G, ["mc", "delta_mc"], meta=meta)


def test_set_refuses_artifacts_in_different_frames():
    """Same coordinate NAMES, different meanings: detector-frame mc is not source-frame mc."""
    with pytest.raises(ValueError, match="frames"):
        nal_io.NALSet([_nal({"frame": "detector"}),
                       _nal({"frame": "source", "cosmology": {"name": "Planck15"},
                             "d_prior": {"name": "cosmo_sourceframe"}}, seed=1)])


def test_set_refuses_undeclared_frame():
    """Fail closed: an artifact that will not state its frame cannot be shown compatible."""
    with pytest.raises(ValueError, match="frame"):
        nal_io.NALSet([_nal({"frame": "detector"}), _nal({}, seed=1)])


def test_set_refuses_mismatched_cosmology_or_distance_prior():
    a = {"frame": "source", "chart": "NAL:aligned", "cosmology": {"name": "Planck15"},
         "d_prior": {"name": "cosmo_sourceframe", "d_max": 10000.0}}
    b = dict(a, cosmology={"name": "Planck18"})
    with pytest.raises(ValueError, match="cosmology"):
        nal_io.NALSet([_nal(a), _nal(b, seed=1)])
    c = dict(a, d_prior={"name": "uniform_comoving", "d_max": 10000.0})
    with pytest.raises(ValueError, match="d_prior"):
        nal_io.NALSet([_nal(a), _nal(c, seed=1)])


def test_set_requires_cosmology_for_source_frame_artifacts():
    m = {"frame": "source", "chart": "NAL:aligned"}
    with pytest.raises(ValueError, match="cosmology"):
        nal_io.NALSet([_nal(m), _nal(m, seed=1)])


def test_set_refuses_mismatched_charts():
    with pytest.raises(ValueError, match="chart"):
        nal_io.NALSet([_nal({"frame": "detector", "chart": "NAL:aligned"}),
                       _nal({"frame": "detector", "chart": "NAL:precessing"}, seed=1)])


def test_set_refuses_artifacts_that_all_omit_the_chart():
    """Silence from every artifact is not agreement.

    Same coord_names, same frame, no chart anywhere: nothing establishes that the two were built
    in the same coordinate CONVENTIONS (spin basis, mass pairing, angle reference), and the sum
    would be well-defined arithmetic over two different meanings of theta. `write_nal(chart=None)`
    produces exactly these, so the all-undeclared case is the one that actually occurs -- and it
    is the case that a "declared values must match" check would wave through.
    """
    with pytest.raises(ValueError, match="chart"):
        nal_io.NALSet([_nal({"frame": "detector"}), _nal({"frame": "detector"}, seed=1)])
    # partially declared is no better: one artifact still cannot be shown to agree
    with pytest.raises(ValueError, match="chart"):
        nal_io.NALSet([_nal({"frame": "detector", "chart": "NAL:aligned"}),
                       _nal({"frame": "detector"}, seed=1)])
    # nor is a declaration that says nothing
    with pytest.raises(ValueError, match="chart"):
        nal_io.NALSet([_nal({"frame": "detector", "chart": ""}),
                       _nal({"frame": "detector", "chart": ""}, seed=1)])
    # and the escape hatch for a caller who has established equivalence by other means still works
    nal_io.NALSet([_nal({"frame": "detector"}), _nal({"frame": "detector"}, seed=1)],
                  require_compatible=False)


def test_written_artifacts_without_a_chart_cannot_be_summed(tmp_path):
    """End to end: write_nal(chart=None) is loadable and usable alone, but not addable."""
    mu, G = _make(d=2)
    paths = []
    for i in range(2):
        base = str(tmp_path / ("ev%d" % i))
        nal_io.write_nal(base, nal_io.NAL(mu, G, ["mc", "delta_mc"]), frame="detector")
        paths.append(base)
    loaded = [nal_io.load_nal(p + ".npz") for p in paths]
    nal_io.NALSet([loaded[0]])                            # alone: fine, nothing is being added
    with pytest.raises(ValueError, match="chart"):
        nal_io.NALSet(loaded)
    for p in paths:                                       # ... and declaring one fixes it
        nal_io.write_nal(p, nal_io.NAL(mu, G, ["mc", "delta_mc"]), chart="NAL:aligned",
                         frame="detector")
    nal_io.NALSet([nal_io.load_nal(p + ".npz") for p in paths])


def test_set_accepts_matching_metadata_and_a_lone_artifact():
    m = {"frame": "detector", "chart": "NAL:aligned"}
    s = nal_io.NALSet([_nal(m), _nal(m, seed=1)])
    assert s.coord_names == ["mc", "delta_mc"]
    # a single artifact is never checked: nothing is being added to it
    assert nal_io.NALSet([_nal({})]).coord_names == ["mc", "delta_mc"]
    # dict ordering is not a difference
    nal_io.NALSet([_nal({"frame": "source", "chart": "NAL:aligned",
                         "cosmology": {"name": "Planck15", "h": 0.679},
                         "d_prior": {"name": "p", "d_max": 1.0}}),
                   _nal({"frame": "source", "chart": "NAL:aligned",
                         "cosmology": {"h": 0.679, "name": "Planck15"},
                         "d_prior": {"d_max": 1.0, "name": "p"}}, seed=1)])


# --------------------------------------------------------- the artifacts vs the RUN's own chart

def test_sampler_check_rejects_an_undeclared_or_mismatched_run_frame():
    """The set check compares artifacts with each other; it says nothing about the sampler."""
    ok = {"frame": "detector", "chart": "NAL:aligned"}
    with pytest.raises(ValueError, match="sampling frame is undeclared"):
        nal_io.check_sampler_compatible([_nal(ok)], "", "NAL:aligned")
    with pytest.raises(ValueError, match="frame"):
        nal_io.check_sampler_compatible([_nal(ok)], "source", "NAL:aligned")
    with pytest.raises(ValueError, match="sampler_frame"):
        nal_io.check_sampler_compatible([_nal(ok)], "geocenter", "NAL:aligned")
    nal_io.check_sampler_compatible([_nal(ok)], "detector", "NAL:aligned")


def test_sampler_check_rejects_an_undeclared_or_mismatched_run_chart():
    ok = {"frame": "detector", "chart": "NAL:aligned"}
    with pytest.raises(ValueError, match="sampling chart is undeclared"):
        nal_io.check_sampler_compatible([_nal(ok)], "detector", "")
    with pytest.raises(ValueError, match="chart"):
        nal_io.check_sampler_compatible([_nal(ok)], "detector", "NAL:precessing")
    with pytest.raises(ValueError, match="chart"):        # artifact declares nothing
        nal_io.check_sampler_compatible([_nal({"frame": "detector"})], "detector", "NAL:aligned")


def test_a_lone_source_frame_artifact_is_still_checked_against_the_run(tmp_path, monkeypatch):
    """The hole the set check cannot cover: one artifact is never compared with anything.

    Source- and detector-frame charts wear the same coordinate names, so a source-frame NAL fed
    the driver's default detector-frame samples raises nothing on names or array count.
    """
    mu = np.array([30.0, 0.3])
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=np.diag([1.0, 4.0]))
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": 0.0, "frame": "source",
               "chart": "NAL:aligned", "cosmology": {"name": "Planck15"},
               "d_prior": {"name": "cosmo_sourceframe"}}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    _declare_run(monkeypatch, frame="detector")           # the driver's default basis
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    with pytest.raises(ValueError, match="frame"):
        nal_io.prepare_nal_lnL(config=None, coords=["mc", "delta_mc"])
    # declaring the run honestly is what makes it usable
    _declare_run(monkeypatch, frame="source")
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "delta_mc"])
    assert np.isclose(nal_io.nal_lnL(np.array([30.0]), np.array([0.3]))[0], 0.0, atol=1e-10)


def test_contribution_is_centred_so_the_drivers_exponentiation_cannot_overflow(tmp_path,
                                                                               monkeypatch):
    """Both drivers' DEFAULT path is likelihood_function(*x) * np.exp(supplemental(*x)).

    float64 exp overflows above ~709, so a loud but perfectly valid artifact (lnL_peak ~ SNR^2/2)
    would contribute inf for every sample.  The contribution is centred by the summed peak; the
    constant is recoverable from nal_lnL_offset().
    """
    mu = np.array([30.0, 0.3])
    loud = 3386.0                                        # a real SNR~82 event's lnL_peak
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=np.diag([1.0, 4.0]))
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": loud, "frame": "detector",
               "chart": "NAL:aligned"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    _declare_run(monkeypatch)
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    nal_io.prepare_nal_lnL(config=None, coords=["mc", "delta_mc"])

    X = [np.array([30.0, 30.2, 29.5]), np.array([0.3, 0.35, 0.2])]
    out = nal_io.nal_lnL(*X)
    assert np.isclose(nal_io.nal_lnL_offset(), loud)
    assert np.all(out <= 0.0)                            # never positive, so exp never overflows
    assert np.all(np.isfinite(np.exp(out)))
    # SHAPE is untouched: only a constant has moved
    assert np.allclose(out - out[0], [0.0, -0.5 * (0.2 ** 2 + 4 * 0.05 ** 2),
                                      -0.5 * (0.5 ** 2 + 4 * 0.1 ** 2)])


# ------------------------------------------------------------------------ bounded marginalization

def _correlated(rho=0.9, d=2):
    C = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    return np.zeros(d), np.linalg.inv(C)


def test_marginal_rejects_a_bounded_correlated_nuisance():
    """Integrating out a truncated, correlated coordinate is not a covariance sub-block.

    The exact marginal picks up the mass of the dropped coordinate's CONDITIONAL distribution
    inside its bounds, whose mean moves with the retained coordinate -- a theta-dependent factor,
    so the untruncated answer has the wrong SHAPE, not just the wrong normalisation.
    """
    mu, G = _correlated()
    n = nal_io.NAL(mu, G, ["mc", "delta_mc"], bounds=np.stack([mu - 0.5, mu + 0.5], 1))
    with pytest.raises(ValueError, match="delta_mc"):
        n.marginal(["mc"])
    # and the escape hatch still gives the untruncated sub-block
    m = n.marginal(["mc"], ignore_truncation=True)
    assert np.allclose(m.cov(), np.linalg.inv(G)[:1, :1])


def test_marginal_allowed_when_the_dropped_bound_does_not_bite():
    """Bounds far outside the fit are a formality: the truncation factor is 1 to ~1e-6."""
    mu, G = _correlated()
    wide = np.stack([mu - 50.0, mu + 50.0], 1)
    n = nal_io.NAL(mu, G, ["mc", "delta_mc"], bounds=wide)
    m = n.marginal(["mc"])
    assert np.allclose(m.cov(), np.linalg.inv(G)[:1, :1])
    assert np.allclose(m.bounds, wide[:1])


def test_marginal_allowed_when_the_dropped_coordinate_is_uncorrelated():
    """No correlation -> the truncation factor is a constant, which only shifts lnL_peak."""
    mu = np.zeros(2)
    n = nal_io.NAL(mu, np.diag([1.0, 4.0]), ["mc", "delta_mc"],
                   bounds=np.stack([mu - 0.1, mu + 0.1], 1))
    m = n.marginal(["mc"])
    assert np.allclose(m.gamma, [[1.0]])


def test_unbounded_marginal_is_unaffected():
    """The plain (bounds-free) marginal must keep working exactly as before."""
    mu, G = _make(d=5, seed=3)
    n = nal_io.NAL(mu, G, list("abcde"))
    assert np.allclose(n.marginal(["a", "b"]).cov(),
                       np.linalg.inv(G)[np.ix_([0, 1], [0, 1])], atol=1e-12)


# --------------------------------------------------------------------------- writer / provenance

def test_write_read_roundtrip_preserves_everything(tmp_path):
    mu, G = _make(d=3, seed=7)
    names = ["mc", "delta_mc", "xi"]
    n = nal_io.NAL(mu, G, names, lnL_peak=12.5,
                   bounds=np.stack([mu - 3, mu + 3], 1))
    src = tmp_path / "all.net"
    src.write_text("dummy grid\n")
    base = str(tmp_path / "ev")
    nal_io.write_nal(base, n, chart="NAL:aligned", frame="detector",
                     parents=[str(src)], run_id="unit-test",
                     validation={"chi2_red": 1.02})
    back = nal_io.load_nal(base + ".npz")
    assert back.coord_names == names
    assert np.allclose(back.mu, mu) and np.allclose(back.gamma, G)
    assert np.isclose(back.lnL_peak, 12.5)
    X = mu + 0.1
    assert np.allclose(back.lnL(X), n.lnL(X), atol=1e-12)
    meta = json.load(open(base + ".meta.json"))
    assert meta["schema"] == nal_io.SCHEMA_VERSION and meta["chart"] == "NAL:aligned"
    assert meta["parents"] and len(meta["parents"][0]["sha256"]) == 64
    assert meta["validation"]["chi2_red"] == 1.02


def test_frame_invariant_rejects_source_frame_with_distance():
    """An artifact may carry u_d OR source-frame masses, never both."""
    with pytest.raises(ValueError, match="u_d"):
        nal_io.check_frame_invariant(["mc", "delta_mc", "u_d"], "source",
                                     cosmology={"name": "Planck15"},
                                     d_prior={"name": "cosmo_sourceframe"})


def test_frame_invariant_requires_cosmology_and_prior_for_source_frame():
    with pytest.raises(ValueError, match="cosmology"):
        nal_io.check_frame_invariant(["mc", "delta_mc"], "source")
    with pytest.raises(ValueError, match="distance prior"):
        nal_io.check_frame_invariant(["mc", "delta_mc"], "source",
                                     cosmology={"name": "Planck15"})
    # fully declared: fine
    nal_io.check_frame_invariant(["mc", "delta_mc"], "source",
                                 cosmology={"name": "Planck15"},
                                 d_prior={"name": "cosmo_sourceframe",
                                          "d_min": 1.0, "d_max": 10000.0})


def test_frame_invariant_rejects_bad_frame_name():
    with pytest.raises(ValueError, match="frame"):
        nal_io.check_frame_invariant(["mc"], "det")


def test_write_refuses_undeclared_source_frame(tmp_path):
    """The writer must not emit an artifact that cannot state its own frame honestly."""
    mu, G = _make(d=2, seed=8)
    n = nal_io.NAL(mu, G, ["mc", "delta_mc"])
    with pytest.raises(ValueError):
        nal_io.write_nal(str(tmp_path / "bad"), n, frame="source")
    assert not os.path.exists(str(tmp_path / "bad.npz"))


def test_frame_invariant_rejects_source_frame_carrying_dist(tmp_path):
    """`dist` is a distance coordinate exactly as much as `u_d` is -- _derive interconverts them.

    Checking only 'u_d' let write_nal emit the artifact this invariant exists to forbid: masses
    declared source-frame, a distance prior recorded as already integrated out, and the distance
    still sitting in the chart.
    """
    with pytest.raises(ValueError, match="dist"):
        nal_io.check_frame_invariant(["mc", "delta_mc", "dist"], "source",
                                     cosmology={"name": "Planck15"},
                                     d_prior={"name": "cosmo_sourceframe"})
    # detector-frame is where a distance coordinate belongs, under either spelling
    nal_io.check_frame_invariant(["mc", "delta_mc", "dist"], "detector")
    # ... and the writer must refuse it too, without leaving a file behind
    mu, G = _make(d=3, seed=12)
    n = nal_io.NAL(mu, G, ["mc", "delta_mc", "dist"])
    with pytest.raises(ValueError, match="dist"):
        nal_io.write_nal(str(tmp_path / "bad"), n, chart="NAL:aligned", frame="source",
                         cosmology={"name": "Planck15"},
                         d_prior={"name": "cosmo_sourceframe"})
    assert not os.path.exists(str(tmp_path / "bad.npz"))
    assert not os.path.exists(str(tmp_path / "bad.meta.json"))


def test_extra_may_not_overwrite_validated_metadata(tmp_path):
    """`extra` is applied after check_frame_invariant, so it must not reach the checked keys.

    Otherwise frame='detector' is what gets validated and frame='source' is what gets recorded:
    a source-frame artifact with no cosmology, no distance prior, and a distance coordinate.
    """
    mu, G = _make(d=3, seed=13)
    n = nal_io.NAL(mu, G, ["mc", "delta_mc", "u_d"])
    base = str(tmp_path / "ev")
    with pytest.raises(ValueError, match="frame"):
        nal_io.write_nal(base, n, chart="NAL:aligned", frame="detector",
                         extra={"frame": "source"})
    assert not os.path.exists(base + ".npz")             # rejected before anything is written
    with pytest.raises(ValueError, match="cosmology"):
        nal_io.write_nal(base, n, chart="NAL:aligned", frame="detector",
                         extra={"cosmology": {"name": "Planck15"}})
    # extra that only ADDS is still honoured
    nal_io.write_nal(base, n, chart="NAL:aligned", frame="detector",
                     extra={"pipeline_note": "synthetic"})
    meta = json.load(open(base + ".meta.json"))
    assert meta["frame"] == "detector" and meta["pipeline_note"] == "synthetic"


def _hand_written(tmp_path, meta, name="ev", coord_names=("mc", "delta_mc", "u_d")):
    """An artifact assembled by hand, exactly as a foreign exporter produces one.

    Deliberately does NOT go through write_nal: the whole point of the consumer-side check is that
    most artifacts a run loads were never near this module's writer.
    """
    mu, G = _make(d=len(coord_names), seed=17)
    base = str(tmp_path / name)
    np.savez(base + ".npz", theta_star=mu, gamma=G)
    full = {"coord_names": list(coord_names), "lnL_peak": 0.0, "chart": "NAL:aligned"}
    full.update(meta)
    json.dump(full, open(base + ".meta.json", "w"))
    return base


def test_loaded_artifact_frame_invariant_is_enforced_on_the_consumer_side(tmp_path):
    """The invariant write_nal enforces must also hold for artifacts it did not write.

    A source-frame artifact still carrying the distance coordinate, or one declaring source-frame
    masses with no cosmology and no distance prior, has integrated the mass-redshift degeneracy
    against a prior nobody recorded.  Loaded, it evaluates perfectly happily: right dimension,
    right names, wrong masses, no error anywhere downstream.
    """
    base = _hand_written(tmp_path, {"frame": "source", "cosmology": {"name": "Planck15"},
                                    "d_prior": {"name": "cosmo_sourceframe"}}, name="carries_ud")
    with pytest.raises(ValueError, match="u_d"):
        nal_io.load_nal(base + ".npz")

    base = _hand_written(tmp_path, {"frame": "source"}, name="no_cosmo",
                         coord_names=("mc", "delta_mc"))
    with pytest.raises(ValueError, match="cosmology"):
        nal_io.load_nal(base + ".npz")

    base = _hand_written(tmp_path, {"frame": "sourceframe"}, name="bad_frame",
                         coord_names=("mc", "delta_mc"))
    with pytest.raises(ValueError, match="frame"):
        nal_io.load_nal(base + ".npz")

    # ... and a consistent one loads, with the file named on the object for later error messages
    base = _hand_written(tmp_path, {"frame": "detector"}, name="ok")
    n = nal_io.load_nal(base + ".npz")
    assert n.source == base and n.meta["frame"] == "detector"


def test_undeclared_frame_loads_but_never_reaches_an_evaluation(tmp_path, monkeypatch):
    """The shipped O3/O4 catalogue declares no frame at all; it must stay loadable, not usable.

    Loading is how such an artifact gets inspected and rewritten with its frame recorded, so the
    load-time check does not reject it.  Every path that would EVALUATE it does: the plugin entry
    point, and check_artifact_frame_invariant itself when asked to fail closed.
    """
    base = _hand_written(tmp_path, {}, coord_names=("mc", "delta_mc"))   # no 'frame' key at all
    n = nal_io.load_nal(base + ".npz")                    # loads: nothing has been evaluated yet
    assert n.meta.get("frame") is None

    with pytest.raises(ValueError, match="no 'frame'"):
        nal_io.check_artifact_frame_invariant(n, require_frame=True)

    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    _declare_run(monkeypatch, frame="detector")
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    with pytest.raises(ValueError, match="frame"):
        nal_io.prepare_nal_lnL(config=None, coords=["mc", "delta_mc"])


def test_offset_restores_the_absolute_likelihood_and_evidence(tmp_path, monkeypatch):
    """nal_lnL(x) + nal_lnL_offset() is the artifacts' TRUE lnL -- what the drivers must report.

    The centring keeps the drivers' exponentiation in range, and cancels in the posterior; it does
    not cancel in an absolute lnL or an evidence.  Adding the offset back must recover the
    uncentred value exactly, in both renormalize modes -- the offset tracks whichever constant was
    actually removed, so a driver that adds it back needs to know nothing about the plugin's mode.
    """
    mu = np.array([30.0, 0.3])
    loud = 3386.0
    base = str(tmp_path / "ev")
    np.savez(base + ".npz", theta_star=mu, gamma=np.diag([1.0, 4.0]),
             bounds=np.array([[25.0, 35.0], [0.0, 1.0]]))
    json.dump({"coord_names": ["mc", "delta_mc"], "lnL_peak": loud, "frame": "detector",
               "chart": "NAL:aligned"}, open(base + ".meta.json", "w"))
    monkeypatch.setenv("RIFT_NAL_ARTIFACTS", base + ".npz")
    _declare_run(monkeypatch)
    X = [np.array([30.0, 30.2, 29.5]), np.array([0.3, 0.35, 0.2])]
    theta = np.stack(X, 1)

    for renormalize in (False, True):
        nal_io._STATE.update(set=None, coords=None, renormalize=renormalize, offset=0.0)
        nal_io.prepare_nal_lnL(config=None, coords=["mc", "delta_mc"])
        centred = nal_io.nal_lnL(*X)
        want = nal_io._STATE["set"].lnL(theta, renormalize=renormalize)
        assert np.all(centred <= 0.0)                     # exp() stays in range for the driver
        assert np.allclose(centred + nal_io.nal_lnL_offset(), want, atol=1e-9)
        # the constant is large enough to matter: reporting the centred value would put the
        # evidence out by thousands of nat, not by a rounding error
        assert nal_io.nal_lnL_offset() > 3000.0

    # before preparation the offset is a harmless zero, so a driver may query it unconditionally
    nal_io._STATE.update(set=None, coords=None, renormalize=False, offset=0.0)
    assert nal_io.nal_lnL_offset() == 0.0


def test_gwalk_offset_conversion_and_scale_max(tmp_path):
    """offset = lnL_peak + D/2 ln2pi - 1/2 ln|Gamma|, and scale_max must clear gwalk's 500 cap."""
    h5py = pytest.importorskip("h5py")
    mu, G = _make(d=3, seed=4)
    loud = 3386.0                                    # a real SNR~82 event's lnL_peak
    n = nal_io.NAL(mu, G, ["mc", "delta_mc", "xi"], lnL_peak=loud)
    path = str(tmp_path / "view.h5")
    off = nal_io.write_gwalk_view(path, n, "S250114ax/NAL:aligned:test:nal")
    sign, logdet = np.linalg.slogdet(G)
    assert np.isclose(off, loud + 0.5 * 3 * np.log(2 * np.pi) - 0.5 * logdet)
    with h5py.File(path, "r") as f:
        g = f["S250114ax/NAL:aligned:test:nal"]
        assert set(["mu", "std", "cor", "cov", "limits", "offset", "scale"]) <= set(g.keys())
        assert g.attrs["scale_max"] > abs(off) > 500.0     # would trip gwalk's default cap
