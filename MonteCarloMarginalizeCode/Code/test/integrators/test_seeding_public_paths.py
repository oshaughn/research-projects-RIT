#!/usr/bin/env python
"""
--seed reproducibility, asserted through the PRODUCTION CALL PATHS.

Companion to test_seeding_reproducibility.py, which pins the seeding HELPERS
(seed_everything, derived_rng, next_derived_rng).  Those helper tests are not
enough, and the gap is not academic: with the helpers in place but the call
sites reverted to np.random.RandomState(seed) / np.random.default_rng(), the
whole of test/integrators/ stays green.  A merge, a revert or a refactor could
therefore put the defect back with CI reporting nothing.

So this file never calls a helper.  It drives the public entry points a RIFT
driver actually calls -- MCSampler.bootstrap_from_samples / _from_gaussian /
_from_gaussian_mixture, build_warm_seed, ResamplingOracle.draw_simplified, and
the calmarg cal-realization draws -- and asserts on what they produce.

Each entry point is checked for four properties, because different mutations
break different ones:

  1. same seed  -> identical output          (the reproducibility fix)
  2. other seed -> different output          (seeded, not frozen)
  3. successive calls in ONE run differ      (not "seeded and self-correlated":
                                              every intrinsic point must not
                                              share one cloud)
  4. an explicit seed= argument still wins   (the API promise is not taken over)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from RIFT.integrators import mcsamplerAdaptiveVolume as mcsamplerAV
from RIFT.integrators import seeding


NAMES = ["a", "b", "c", "d"]
NDIM = len(NAMES)
LO = np.zeros(NDIM)
HI = np.ones(NDIM)


@pytest.fixture(autouse=True)
def _restore_module_state():
    prior_seed = seeding._seed_used
    prior_counters = dict(seeding._stream_counters)
    yield
    seeding._seed_used = prior_seed
    seeding._stream_counters.clear()
    seeding._stream_counters.update(prior_counters)


def _sampler(n_chunk=2000):
    """Bound to the active backend exactly as the ILE driver does."""
    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = mcsamplerAV.xpy_default
    s.identity_convert = mcsamplerAV.identity_convert
    for name in NAMES:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)),
                        adaptive_sampling=True)
    return s


def _spy_cloud(sampler):
    """Capture the point cloud the warm start actually hands to the grid builder.

    Asserting on the returned _warm grid would be a weaker test: the grid is a
    lossy function of the cloud, so two different cover clouds can bin to the
    same live volume and a mutation would slip through.  The cloud is the thing
    the RNG produces, so that is what we compare.
    """
    seen = []
    original = sampler._build_grid_from_points

    def _capture(X, *a, **kw):
        seen.append(np.array(X, dtype=float, copy=True))
        return original(X, *a, **kw)

    sampler._build_grid_from_points = _capture
    return seen


def _core_cloud(n=400, spread=0.02, seed=3):
    """A concentrated, FULL-RANK seed cloud: cover_frac is what we are testing,
    so the core must not be the thing that triggers the puff path."""
    return np.clip(0.5 + spread * np.random.RandomState(seed).randn(n, NDIM), 0.0, 1.0)


# ---------------------------------------------------------------------------
# bootstrap_from_samples -- the live defect: cover_frac defaults to 0.5 in all
# three driver warm-start options, so this cloud is drawn on every warm start.
# ---------------------------------------------------------------------------

def _from_samples(core, seed=None):
    s = _sampler()
    seen = _spy_cloud(s)
    s.bootstrap_from_samples(core, cover_frac=0.5, seed=seed)
    assert seen, "bootstrap_from_samples did not reach the grid builder"
    return seen[-1]


def test_bootstrap_from_samples_cover_cloud_reproduces_under_the_same_seed():
    core = _core_cloud()
    seeding.seed_everything(101, verbose=False)
    a1, a2 = _from_samples(core), _from_samples(core)
    seeding.seed_everything(101, verbose=False)
    b1, b2 = _from_samples(core), _from_samples(core)
    seeding.seed_everything(202, verbose=False)
    c1 = _from_samples(core)

    assert a1.shape == b1.shape
    assert (a1 == b1).all(), "same --seed gave a different cover cloud"
    assert (a2 == b2).all(), "the SECOND warm start of the run did not reproduce"
    assert not (a1 == c1).all(), "a different --seed gave the identical cover cloud"
    assert not (a1 == a2).all(), (
        "two warm starts in one run share a cover cloud; every intrinsic point "
        "would be seeded with the same uniform points")


def test_bootstrap_from_samples_honours_an_explicit_seed():
    core = _core_cloud()
    seeding.seed_everything(101, verbose=False)
    a = _from_samples(core, seed=7)
    seeding.seed_everything(202, verbose=False)
    b = _from_samples(core, seed=7)
    assert (a == b).all(), "an explicit seed= must not be overridden by --seed"


# ---------------------------------------------------------------------------
# bootstrap_from_gaussian / _from_gaussian_mixture -- the Fisher/flow oracle
# seeds.  Same shape, and they also exercise multivariate_normal / multinomial,
# which the derived path serves from a Generator rather than a RandomState.
# ---------------------------------------------------------------------------

def _from_gaussian(seed=None):
    s = _sampler()
    seen = _spy_cloud(s)
    s.bootstrap_from_gaussian(0.5 * np.ones(NDIM), 0.01 * np.eye(NDIM),
                              n=500, seed=seed)
    assert seen
    return seen[-1]


def _from_mixture(seed=None):
    s = _sampler()
    seen = _spy_cloud(s)
    s.bootstrap_from_gaussian_mixture(
        [0.3 * np.ones(NDIM), 0.7 * np.ones(NDIM)],
        [0.01 * np.eye(NDIM), 0.01 * np.eye(NDIM)],
        n=500, seed=seed)
    assert seen
    return seen[-1]


@pytest.mark.parametrize("draw", [_from_gaussian, _from_mixture])
def test_gaussian_warm_starts_reproduce_under_the_same_seed(draw):
    seeding.seed_everything(101, verbose=False)
    a1, a2 = draw(), draw()
    seeding.seed_everything(101, verbose=False)
    b1, b2 = draw(), draw()
    seeding.seed_everything(202, verbose=False)
    c1 = draw()

    assert (a1 == b1).all() and (a2 == b2).all(), "same --seed gave a different seed cloud"
    assert not (a1 == c1).all(), "a different --seed gave the identical seed cloud"
    assert not (a1 == a2).all(), "successive warm starts share one cloud"


@pytest.mark.parametrize("draw", [_from_gaussian, _from_mixture])
def test_gaussian_warm_starts_honour_an_explicit_seed(draw):
    seeding.seed_everything(101, verbose=False)
    a = draw(seed=7)
    seeding.seed_everything(202, verbose=False)
    b = draw(seed=7)
    assert (a == b).all()


# ---------------------------------------------------------------------------
# build_warm_seed -- the L0 rescue / sequential warm-start puff.  This one was
# RandomState(0): never irreproducible, but --seed-INERT and self-correlated,
# so a replicate-seed study had its rescue arm frozen identically across arms.
# ---------------------------------------------------------------------------

def _rank_deficient_pass(n=40):
    """Points confined to a 1-D line: rank-deficient, so the puff path runs."""
    t = np.linspace(0.4, 0.6, n)
    X = np.tile(0.5, (n, NDIM))
    X[:, 0] = t
    lnL = 100.0 - 1e-3 * (t - 0.5) ** 2
    return X, lnL


def _puff(seed=None):
    X, lnL = _rank_deficient_pass()
    out, info = mcsamplerAV.build_warm_seed(X, lnL, LO, HI, list(range(NDIM)),
                                            deltalnL=15.0, n_puff=300, seed=seed)
    assert info.get('puffed'), "the puff path did not run; this test proves nothing"
    return np.asarray(out, dtype=float)


def test_build_warm_seed_puff_depends_on_the_run_seed():
    seeding.seed_everything(101, verbose=False)
    a1, a2 = _puff(), _puff()
    seeding.seed_everything(101, verbose=False)
    b1, b2 = _puff(), _puff()
    seeding.seed_everything(202, verbose=False)
    c1 = _puff()

    assert (a1 == b1).all() and (a2 == b2).all(), "same --seed gave a different puff"
    assert not (a1 == c1).all(), (
        "the puff is the same under --seed 101 and --seed 202; a replicate-seed "
        "study would have its rescue arm frozen across arms")
    assert not (a1 == a2).all(), (
        "every intrinsic point of the run is puffed with the same deviates")


def test_build_warm_seed_honours_an_explicit_seed():
    seeding.seed_everything(101, verbose=False)
    a = _puff(seed=7)
    seeding.seed_everything(202, verbose=False)
    b = _puff(seed=7)
    assert (a == b).all()


# ---------------------------------------------------------------------------
# ResamplingOracle -- the skymap oracle (--skymap-file, default sampler).  The
# RNG here was always seeded; what was not was WHICH block of the seeded stream
# reached which parameter, because the parameter list came out of a set of
# STRINGS and str hashing is salted per process.  So this one can only be
# tested across processes: PYTHONHASHSEED has to actually differ.
# ---------------------------------------------------------------------------

_ORACLE_PROBE = r"""
import io, contextlib
import numpy as np
from RIFT.integrators.seeding import seed_everything
from RIFT.integrators.unreliable_oracle.resampling import ResamplingOracle

names = ["right_ascension", "declination", "distance", "psi", "phi_orb", "incl", "t_ref"]
o = ResamplingOracle()
for p in names:
    o.add_parameter(p, pdf=None, left_limit=0.0, right_limit=1000.0)
ref = np.random.RandomState(0).uniform(size=(500, 2))
with contextlib.redirect_stdout(io.StringIO()):
    o.setup(reference_samples=ref, reference_params=["right_ascension", "declination"])
seed_everything(101, verbose=False)
_, _, rv = o.draw_simplified(64)
print(",".join(o.other_params))
print(" ".join("%.12g" % v for v in rv[:, o.params_ordered.index("distance")]))
"""


def _oracle_draw(hashseed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    out = subprocess.check_output([sys.executable, "-c", _ORACLE_PROBE],
                                  env=env, stderr=subprocess.DEVNULL)
    order, draws = out.decode().strip().splitlines()[-2:]
    return order, draws


def test_skymap_oracle_draws_do_not_depend_on_string_hash_salt():
    """Reachable from --skymap-file with the DEFAULT sampler, and it trains the
    sampling prior (and seeds the AV live volume), so it reaches lnZ."""
    orders, draws = zip(*[_oracle_draw(h) for h in (1, 2, 3, 4, 5)])
    assert len(set(orders)) == 1, (
        "the uniform-fill parameter order still varies with PYTHONHASHSEED: %r" % (set(orders),))
    assert len(set(draws)) == 1, (
        "same --seed, different PYTHONHASHSEED, different draws -- %d distinct "
        "results across 5 processes" % len(set(draws)))


def test_skymap_oracle_fill_order_follows_params_ordered():
    """Pins the property rather than the symptom: a future refactor that
    reintroduces any unordered container fails here without needing 5 subprocesses."""
    import io
    import contextlib
    from RIFT.integrators.unreliable_oracle.resampling import ResamplingOracle
    names = ["right_ascension", "declination", "distance", "psi", "phi_orb"]
    o = ResamplingOracle()
    for p in names:
        o.add_parameter(p, pdf=None, left_limit=0.0, right_limit=1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        o.setup(reference_samples=np.zeros((10, 2)),
                reference_params=["right_ascension", "declination"])
    expect = [p for p in o.params_ordered if p not in set(o.valid_params)]
    assert o.other_params == expect, "%r != %r" % (o.other_params, expect)


# ---------------------------------------------------------------------------
# calmarg cal realizations.  The ILE driver always passes an explicit rng, so
# the rng=None fallback is a guard rather than a live defect -- but a guard with
# no test is how the hole comes back when a caller is added.
# ---------------------------------------------------------------------------

def _envelope_file(path):
    """Minimal calibration envelope: freq median_mag median_phase 16_* 84_*."""
    f = np.linspace(5.0, 2000.0, 40)
    dat = np.column_stack([f, np.ones_like(f), np.zeros_like(f),
                           0.95 * np.ones_like(f), -0.05 * np.ones_like(f),
                           1.05 * np.ones_like(f), 0.05 * np.ones_like(f)])
    np.savetxt(path, dat)


def _prior_nodes():
    import RIFT.calmarg.generate_realizations as genr
    with tempfile.TemporaryDirectory() as d:
        _envelope_file(os.path.join(d, "H1.txt"))
        ret = genr.draw_prior_realizations_with_nodes(
            d, ["H1"], 4.0, 1.0 / 4096, 20.0, 1000.0, 4, 8, rng=None)
    return np.asarray(ret["nodes"], dtype=float)


def test_calmarg_prior_node_draws_reproduce_when_the_caller_omits_rng():
    seeding.seed_everything(101, verbose=False)
    a1, a2 = _prior_nodes(), _prior_nodes()
    seeding.seed_everything(101, verbose=False)
    b1, b2 = _prior_nodes(), _prior_nodes()
    seeding.seed_everything(202, verbose=False)
    c1 = _prior_nodes()

    assert (a1 == b1).all() and (a2 == b2).all(), "same --seed gave different cal nodes"
    assert not (a1 == c1).any(), "a different --seed gave the identical cal nodes"
    assert not (a1 == a2).any(), (
        "a second cal draw in one run repeats the first; growing the cal set "
        "would append copies of draws already in it")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
