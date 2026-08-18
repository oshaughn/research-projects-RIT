#!/usr/bin/env python
"""
Regression tests for --seed reproducibility, especially on GPU.

Background (the bug these tests lock down): the ILE drivers implemented --seed
as a bare ``numpy.random.seed(opts.seed)``.  The samplers, however, draw their
variates through the *array backend* they were configured with -- ``self.xpy``
on an instance, ``xpy_default`` at module scope -- and that backend is cupy
whenever the job runs on a GPU.  cupy keeps its own global generator per
device, which numpy.random.seed does not touch, so a GPU run was irreproducible
even when the user explicitly asked for a seed.

Two byte-identical invocations of the ILE demo with --seed 101 returned
lnL = 75.857 (n_eff 1.02) and lnL = 71.687 (n_eff 2.04) -- a 4.17 nat spread.
Beyond being unbisectable, that silently invalidates any paired /
replicate-seed comparison design run on GPU, because the "same seed" arms are
not in fact paired.

Seeding the RNGs turned out to be necessary but not sufficient.  The adapted
sampling histogram (RIFT.likelihood.vectorized_general_tools.histogram) is
built with a weighted cupy.bincount, which accumulates through float atomicAdd;
the summation order is set by GPU thread scheduling, so the adapted CDF -- and
therefore every draw taken through it -- differed at the ULP level between
otherwise identical runs.  seed_everything therefore also switches that one
reduction to a scheduler-independent summation order.

The GPU-specific tests skip cleanly on a CPU-only machine; the rest of the file
exercises the parts that can be checked without a device.
"""

import numpy as np
import pytest

from RIFT.integrators import seeding
from RIFT.likelihood import vectorized_general_tools as vgt


try:
    import cupy
    cupy.array(0)                       # fails if cuda/cupy is not actually usable
    HAS_GPU = True
except Exception:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="no usable cupy/GPU")


@pytest.fixture(autouse=True)
def _restore_module_state():
    """seed_everything mutates process-global state; put it back afterwards."""
    prior_det = vgt.DETERMINISTIC_REDUCTIONS
    prior_seed = seeding._seed_used
    prior_counters = dict(seeding._stream_counters)
    yield
    vgt.DETERMINISTIC_REDUCTIONS = prior_det
    seeding._seed_used = prior_seed
    seeding._stream_counters.clear()
    seeding._stream_counters.update(prior_counters)


def test_seed_everything_reports_numpy_and_python():
    status = seeding.seed_everything(101, verbose=False)
    assert status['numpy'] == 'seeded'
    assert status['python'] == 'seeded'
    assert seeding.get_seed() == 101


def test_seed_everything_enables_deterministic_reductions():
    """The whole point: asking for a seed must also close the atomics hole."""
    vgt.DETERMINISTIC_REDUCTIONS = False
    status = seeding.seed_everything(101, verbose=False)
    assert status['gpu_reductions'] == 'deterministic'
    assert vgt.DETERMINISTIC_REDUCTIONS is True


def test_seed_everything_absent_backend_is_not_an_error():
    """A CPU-only install has no cupy; that must be reported, not raised."""
    status = seeding.seed_everything(7, verbose=False)
    for backend in ('cupy', 'torch'):
        assert (status[backend] == 'seeded'
                or status[backend] == 'absent'
                or status[backend].startswith('failed:')), status[backend]


def test_derived_rng_is_reproducible_under_the_same_seed():
    """default_rng() takes OS entropy, so paths that build their own Generator
    (the calibration error probe, the adaptive cal draw growth) escaped --seed
    entirely and could change n_cal / the cal realizations run to run."""
    seeding.seed_everything(101, verbose=False)
    a = seeding.derived_rng('calmarg.error_probe').standard_normal(64)
    seeding.seed_everything(101, verbose=False)
    b = seeding.derived_rng('calmarg.error_probe').standard_normal(64)
    seeding.seed_everything(202, verbose=False)
    c = seeding.derived_rng('calmarg.error_probe').standard_normal(64)

    assert (a == b).all(), "same seed did not reproduce the derived stream"
    assert not (a == c).all(), "different seeds gave an identical stream"


def test_derived_rng_streams_do_not_collide():
    """Reproducible must not mean 'everyone draws the same numbers': distinct
    call sites, distinct counters, and the seed's own default_rng(seed) stream
    all have to stay independent, or growing the cal draw set would just append
    copies of draws already in it."""
    seeding.seed_everything(101, verbose=False)
    probe0 = seeding.derived_rng('calmarg.error_probe', 0).standard_normal(64)
    probe1 = seeding.derived_rng('calmarg.error_probe', 1).standard_normal(64)
    extra0 = seeding.derived_rng('calmarg.extra_draws', 0).standard_normal(64)
    plain = np.random.default_rng(101).standard_normal(64)

    for lhs, rhs, what in ((probe0, probe1, "counters"),
                           (probe0, extra0, "stream names"),
                           (probe0, plain, "derived vs default_rng(seed)")):
        assert not (lhs == rhs).any(), "%s share draws" % what


def test_derived_rng_is_unseeded_when_the_run_was_not_seeded():
    """No --seed must still mean fresh entropy, not a fixed fallback stream."""
    seeding._seed_used = None
    a = seeding.derived_rng('calmarg.error_probe').standard_normal(64)
    b = seeding.derived_rng('calmarg.error_probe').standard_normal(64)
    assert not (a == b).any()


def test_derived_rng_stream_label_is_stable_across_processes():
    """The label must not come from hash(): str hashing is salted per process,
    so a 'stable' identifier built that way would silently drift between the
    two runs the user is trying to compare."""
    seeding.seed_everything(101, verbose=False)
    got = seeding.derived_rng('calmarg.error_probe', 3).standard_normal(8)
    import zlib
    expect = np.random.default_rng(
        [101, zlib.crc32(b'calmarg.error_probe'), 3]).standard_normal(8)
    assert (got == expect).all()


def test_next_derived_rng_advances_so_repeated_calls_do_not_share_draws():
    """A call site inside a loop (one warm start per intrinsic point, one bootstrap
    per integral) must not hand back the same numbers every time.  Reproducible and
    self-correlated is WORSE than unseeded: it would give every intrinsic point the
    identical uniform coverage cloud."""
    seeding.seed_everything(101, verbose=False)
    a = seeding.next_derived_rng('unit.test').standard_normal(64)
    b = seeding.next_derived_rng('unit.test').standard_normal(64)
    assert not (a == b).any(), "successive calls to one stream share draws"
    # and they are the counter-0/counter-1 streams, i.e. still derived, not entropy
    seeding.seed_everything(101, verbose=False)
    assert (a == seeding.derived_rng('unit.test', 0).standard_normal(64)).all()
    assert (b == seeding.derived_rng('unit.test', 1).standard_normal(64)).all()


def test_next_derived_rng_repeats_the_whole_sequence_under_the_same_seed():
    """What --seed actually promises: two identical INVOCATIONS agree.  Re-seeding
    restarts the counters, so run 2 replays run 1's sequence."""
    seeding.seed_everything(101, verbose=False)
    run1 = [seeding.next_derived_rng('unit.test').standard_normal(16) for _ in range(3)]
    seeding.seed_everything(101, verbose=False)
    run2 = [seeding.next_derived_rng('unit.test').standard_normal(16) for _ in range(3)]
    seeding.seed_everything(202, verbose=False)
    run3 = [seeding.next_derived_rng('unit.test').standard_normal(16) for _ in range(3)]

    for x, y in zip(run1, run2):
        assert (x == y).all(), "same seed did not replay the sequence"
    for x, z in zip(run1, run3):
        assert not (x == z).any(), "different seeds gave an identical sequence"


def test_next_derived_rng_is_unseeded_when_the_run_was_not_seeded():
    """No --seed must still mean fresh entropy, not a fixed fallback sequence."""
    seeding._seed_used = None
    seeding._stream_counters.clear()
    a = seeding.next_derived_rng('unit.test').standard_normal(64)
    seeding._stream_counters.clear()
    b = seeding.next_derived_rng('unit.test').standard_normal(64)
    assert not (a == b).any()


def test_av_warm_start_cover_cloud_is_reproducible_under_seed():
    """The one live likelihood-feeding hole this pass closes.

    The bootstrap_from_* family drew its uniform coverage cloud from
    RandomState(None) -- fresh OS entropy, unreachable by seed_everything -- and
    the driver's warm-start options default cover_frac to 0.5, so the cloud IS
    drawn.  It shapes the AV live volume, hence the draws, hence lnZ: two runs
    with the same --seed built different live volumes.
    """
    from RIFT.integrators import mcsamplerAdaptiveVolume as av

    def draw():
        rng = av._warm_seed_rng(None, 'av.bootstrap_from_samples.cover')
        return rng.uniform(np.zeros(4), np.ones(4), size=(32, 4))

    seeding.seed_everything(101, verbose=False)
    a1, a2 = draw(), draw()
    seeding.seed_everything(101, verbose=False)
    b1, b2 = draw(), draw()
    seeding.seed_everything(202, verbose=False)
    c1, _ = draw(), draw()

    assert (a1 == b1).all() and (a2 == b2).all(), "same seed gave a different cover cloud"
    assert not (a1 == c1).any(), "different seeds gave the same cover cloud"
    assert not (a1 == a2).any(), "successive warm starts share one cover cloud"


def test_av_warm_start_explicit_seed_still_wins():
    """An explicit integer seed is an API promise of its own; deriving from --seed
    must not take it over."""
    from RIFT.integrators import mcsamplerAdaptiveVolume as av
    seeding.seed_everything(101, verbose=False)
    got = av._warm_seed_rng(7, 'av.bootstrap_from_samples.cover').uniform(0, 1, 16)
    expect = np.random.RandomState(7).uniform(0, 1, 16)
    assert (got == expect).all()


def test_bootstrap_lnZ_quantiles_is_reproducible_and_leaves_numpy_alone():
    """The lnZ_ci90 diagnostic is reporting-only, so it gets a stream of its own:
    reproducible under --seed, and NOT drawn from numpy's global RNG -- the samplers
    draw from that, so spending draws here would move lnL, which a diagnostic is
    never allowed to do."""
    from RIFT.integrators.statutils import bootstrap_lnZ_quantiles

    lw = np.log(np.random.RandomState(0).exponential(1.0, 500))

    def run():
        np.random.seed(3)
        before = np.random.random(4)          # position in the global stream
        q = bootstrap_lnZ_quantiles(lw)
        after = np.random.random(4)           # must be unaffected by the bootstrap
        return q, before, after

    seeding.seed_everything(101, verbose=False)
    qa, ba, aa = run()
    seeding.seed_everything(101, verbose=False)
    qb, bb, ab = run()
    seeding.seed_everything(202, verbose=False)
    qc, _, _ = run()

    assert qa is not None
    assert (qa == qb).all(), "same seed gave a different bootstrap interval"
    assert not (qa == qc).any(), "different seeds gave an identical bootstrap interval"
    assert (ba == bb).all() and (aa == ab).all()
    # the global stream must be exactly where it would be with no bootstrap at all
    np.random.seed(3)
    np.random.random(4)
    assert (aa == np.random.random(4)).all(), "the diagnostic consumed numpy's global RNG"


def test_calmarg_rng_fallback_is_derived_not_entropy():
    """The ILE driver always passes an explicit rng to the cal draw helpers, so this
    is a guard, not a live defect: a NEW caller that forgets must not silently
    reintroduce an unseeded likelihood."""
    from RIFT.calmarg.generate_realizations import _default_cal_rng

    seeding.seed_everything(101, verbose=False)
    a = _default_cal_rng('unit.cal').standard_normal(32)
    seeding.seed_everything(101, verbose=False)
    b = _default_cal_rng('unit.cal').standard_normal(32)
    seeding.seed_everything(202, verbose=False)
    c = _default_cal_rng('unit.cal').standard_normal(32)
    assert (a == b).all()
    assert not (a == c).any()


def test_deterministic_histogram_agrees_with_atomic_branch():
    """The reproducible branch must be the same histogram, not a different one."""
    rng = np.random.RandomState(0)
    samples = rng.rand(50000)
    weights = rng.exponential(1.0, 50000)

    vgt.DETERMINISTIC_REDUCTIONS = False
    h_atomic = vgt.histogram(samples, 100, xpy=np, weights=weights)
    vgt.DETERMINISTIC_REDUCTIONS = True
    h_det = vgt.histogram(samples, 100, xpy=np, weights=weights)

    assert h_det.shape == h_atomic.shape == (100,)
    np.testing.assert_allclose(h_det, h_atomic, rtol=1e-10)


def test_deterministic_histogram_accuracy_on_peaked_weights():
    """Pin the known accuracy cost of prefix-sum differencing.

    A bin total is the difference of two partial sums both of order the grand
    total, so a bin's relative error is amplified by (total / bin).  With
    exp(lnL)-peaked weights that measured 5e-11 vs an exact rational reference
    (per-bin atomics manage 3e-14).  That is fine for a proposal density, but it
    should not be allowed to get quietly worse.
    """
    from fractions import Fraction

    n_bins = 100
    rng = np.random.RandomState(5)
    idx = rng.randint(0, n_bins, 100000).astype(np.int32)
    wts = np.exp(rng.normal(0, 8, 100000))          # spans ~decades, like exp(lnL)

    acc = [Fraction(0)] * n_bins
    for i, w in zip(idx, wts):
        acc[int(i)] += Fraction(float(w))
    ref = np.array([float(a) for a in acc])

    vgt.DETERMINISTIC_REDUCTIONS = True
    got = vgt._bincount_weighted(idx, wts, n_bins, np)

    rel = np.abs(got - ref) / np.abs(ref)
    assert rel.max() < 1e-9, "deterministic bincount accuracy regressed: %g" % rel.max()


def test_deterministic_histogram_handles_the_unweighted_branch():
    """Unweighted calls pass a read-only broadcast_to view; it must be reorderable."""
    rng = np.random.RandomState(0)
    samples = rng.rand(5000)
    vgt.DETERMINISTIC_REDUCTIONS = True
    h = vgt.histogram(samples, 50, xpy=np)
    assert h.shape == (50,)
    np.testing.assert_allclose(h.sum(), 50.0, rtol=1e-10)


@requires_gpu
def test_cupy_weighted_bincount_is_nondeterministic():
    """Documents WHY the deterministic branch exists.

    If cupy ever makes weighted bincount deterministic this test starts
    failing, which is the signal to revisit -- not a reason to delete the
    deterministic branch, since RIFT must work with older cupy too.
    """
    rng = np.random.RandomState(0)
    idx = cupy.asarray(rng.randint(0, 100, 200000).astype(np.int32))
    wts = cupy.asarray(rng.exponential(1.0, 200000))
    ref = cupy.asnumpy(cupy.bincount(idx, minlength=100, weights=wts))
    differs = any(
        not (cupy.asnumpy(cupy.bincount(idx, minlength=100, weights=wts)) == ref).all()
        for _ in range(32)
    )
    assert differs, "cupy weighted bincount now looks deterministic on this build"


@requires_gpu
def test_deterministic_gpu_histogram_is_bit_reproducible():
    """The fix, at the level of the reduction it repairs."""
    rng = np.random.RandomState(0)
    samples = cupy.asarray(rng.rand(200000))
    weights = cupy.asarray(rng.exponential(1.0, 200000))

    vgt.DETERMINISTIC_REDUCTIONS = True
    ref = cupy.asnumpy(vgt.histogram(samples, 100, xpy=cupy, weights=weights))
    for _ in range(8):
        again = cupy.asnumpy(vgt.histogram(samples, 100, xpy=cupy, weights=weights))
        assert (again == ref).all(), "deterministic GPU histogram is not bit-stable"


@requires_gpu
def test_deterministic_gpu_histogram_handles_the_unweighted_branch():
    """cupy.broadcast_to is also read-only and zero-stride; the deterministic
    path reorders the weights, so this branch must still work on device."""
    rng = np.random.RandomState(0)
    samples = cupy.asarray(rng.rand(20000))
    vgt.DETERMINISTIC_REDUCTIONS = True
    h = cupy.asnumpy(vgt.histogram(samples, 50, xpy=cupy))
    assert h.shape == (50,)
    np.testing.assert_allclose(h.sum(), 50.0, rtol=1e-10)


@requires_gpu
def test_gpu_sampler_draws_are_reproducible_under_seed_everything():
    """End-to-end at the draw level: same seed -> same cupy stream, and a
    different seed must still give a different stream (seeded, not frozen)."""
    seeding.seed_everything(101, verbose=False)
    a = cupy.asnumpy(cupy.random.uniform(0.0, 1.0, 10000))
    seeding.seed_everything(101, verbose=False)
    b = cupy.asnumpy(cupy.random.uniform(0.0, 1.0, 10000))
    seeding.seed_everything(202, verbose=False)
    c = cupy.asnumpy(cupy.random.uniform(0.0, 1.0, 10000))

    assert (a == b).all(), "same seed did not reproduce the cupy stream"
    assert not (a == c).all(), "different seeds gave an identical stream"


@requires_gpu
def test_numpy_seed_alone_does_not_reproduce_the_gpu_stream():
    """The original defect, stated as a test: numpy.random.seed is not enough."""
    np.random.seed(101)
    a = cupy.asnumpy(cupy.random.uniform(0.0, 1.0, 10000))
    np.random.seed(101)
    b = cupy.asnumpy(cupy.random.uniform(0.0, 1.0, 10000))
    assert not (a == b).all(), (
        "numpy.random.seed now appears to seed cupy too; if so the driver's "
        "old behaviour was sufficient and this file needs revisiting")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
