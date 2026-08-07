#!/usr/bin/env python
"""Tests for MC-error replica pooling and the L0 coverage warning.

Both concern the same trap: an estimate that is PRECISE but computed over truncated support looks
better by every efficiency metric than a noisy estimate over full support.

Run:  python test_replica_pooling.py
"""
import os
import re
import types

import numpy


def _load_driver_helpers():
    """Import the helpers out of the driver script without executing it."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "bin", "integrate_likelihood_extrinsic_batchmode")
    src = open(os.path.normpath(path)).read()
    mod = types.ModuleType("drv")
    mod.numpy = numpy
    for fn in ("_rvs_len", "_pool_replica_rvs", "_lnZ_of_rvs", "_kish_neff_of_rvs"):
        m = re.search(r"^def %s\(.*?(?=\n\ndef |\n\nclass )" % fn, src, re.S | re.M)
        assert m, "helper %s not found in the driver" % fn
        exec(compile(m.group(0), "<drv>", "exec"), mod.__dict__)
    return mod


DRV = _load_driver_helpers()


class _S(object):
    identity_convert = staticmethod(lambda x: x)


def _replica(rng, n, lnZ, spread):
    """A record whose importance weights average to exp(lnZ)."""
    lw = rng.normal(0, spread, size=n)
    lw = lw - numpy.log(numpy.mean(numpy.exp(lw))) + lnZ
    return dict(log_integrand=lw, log_joint_prior=numpy.zeros(n),
                log_joint_s_prior=numpy.zeros(n), x=rng.normal(size=n))


def test_pooling_reproduces_the_combined_evidence():
    """The exported posterior must be a draw from the SAME mixture the reported lnZ describes."""
    rng = numpy.random.RandomState(0)
    reps = [_replica(rng, 4000, 0.0, 1.2), _replica(rng, 3000, 0.05, 1.2),
            _replica(rng, 5000, -0.03, 1.0)]
    Zk = [numpy.mean(numpy.exp(r['log_integrand'])) for r in reps]
    lnZ_comb = numpy.log(numpy.mean(Zk))                      # the reported combination
    pooled = DRV._pool_replica_rvs(reps, _S())
    assert abs(DRV._lnZ_of_rvs(pooled) - lnZ_comb) < 1e-9, (
        "pooled samples imply lnZ {} but the reported combination is {}".format(
            DRV._lnZ_of_rvs(pooled), lnZ_comb))
    # unequal replica sizes must still be handled: pooling is 1/(K n_k), not a plain concatenation
    assert len(pooled['x']) == 12000


def test_max_neff_selection_would_export_the_collapsed_replica():
    """Why selection by n_eff is the wrong rule: n_eff measures CONCENTRATION, not coverage, so a
    mode-collapsed replica scores highest and would be the one exported alongside a combined
    evidence it does not represent."""
    rng = numpy.random.RandomState(1)
    broad_a = _replica(rng, 4000, 0.0, 1.2)
    broad_b = _replica(rng, 4000, 0.05, 1.2)
    narrow = _replica(rng, 4000, -0.9, 0.05)          # collapsed: low Z, tiny weight spread
    neffs = [DRV._kish_neff_of_rvs(r) for r in (broad_a, broad_b, narrow)]
    assert int(numpy.argmax(neffs)) == 2, neffs
    pooled = DRV._pool_replica_rvs([broad_a, broad_b, narrow], _S())
    # the pooled n_eff must be honest: smaller than the naive sum over replicas
    assert DRV._kish_neff_of_rvs(pooled) < sum(neffs)


def test_pooling_does_not_repair_a_truncated_support_estimate():
    """The reason the L0 rescue WARNS rather than pooling.

    Averaging Z is unbiased only when every term is unbiased.  A warm pass over truncated support
    is biased low, and pooling it with a full-support pass yields (Z + Z/2)/2 = 0.75 Z -- better
    than warm-only, still wrong.  Pinned here so nobody 'improves' the rescue by pooling it."""
    rng = numpy.random.RandomState(2)
    full = _replica(rng, 4000, 0.0, 1.0)                       # unbiased
    truncated = _replica(rng, 4000, numpy.log(0.5), 0.2)       # missed half the mass
    pooled = DRV._pool_replica_rvs([full, truncated], _S())
    bias = DRV._lnZ_of_rvs(pooled) - 0.0
    assert abs(bias - numpy.log(0.75)) < 0.05, (
        "expected pooling to inherit log(0.75) of bias, got {:+.3f}".format(bias))


def test_lnZ_of_rvs_handles_a_single_run_and_a_pooled_record():
    rng = numpy.random.RandomState(3)
    r = _replica(rng, 2000, 0.25, 0.8)
    assert abs(DRV._lnZ_of_rvs(r, already_pooled=False) - 0.25) < 1e-9
    pooled = DRV._pool_replica_rvs([r, r], _S())
    assert abs(DRV._lnZ_of_rvs(pooled) - 0.25) < 1e-9


def test_missing_columns_degrade_to_the_first_replica():
    """A degraded export is recoverable; a silently mis-weighted one is not."""
    rng = numpy.random.RandomState(4)
    a = dict(x=rng.normal(size=10)); b = dict(x=rng.normal(size=10))
    out = DRV._pool_replica_rvs([a, b], _S())
    assert out is a
    assert DRV._kish_neff_of_rvs(a) is None




def test_pooling_uses_reported_lnZ_when_records_are_not_raw():
    """Production _rvs may already be thresholded or fairdraw-resampled.

    Then sum_i w_ki over the RETAINED rows is no longer Z_k * n_k, so a 1/n_k rescale mis-weights
    the replica -- a fairdraw record is already posterior-resampled and would be weighted twice.
    Given the reported per-replica lnZ, each block is renormalized to contribute exactly Z_k/K,
    which is correct whether the rows are raw, pruned or resampled.
    """
    rng = numpy.random.RandomState(7)
    raw_a = _replica(rng, 4000, 0.0, 1.0)
    raw_b = _replica(rng, 4000, 0.3, 1.0)
    lnZ = [0.0, 0.3]
    # simulate pruning: keep only the top-weight half of replica b
    lw_b = raw_b['log_integrand']
    keep = numpy.argsort(lw_b)[len(lw_b) // 2:]
    pruned_b = {k: numpy.asarray(v)[keep] for k, v in raw_b.items()}

    target = numpy.log(numpy.mean(numpy.exp(numpy.array(lnZ))))
    pooled = DRV._pool_replica_rvs([raw_a, pruned_b], _S(), rep_lnZ=lnZ)
    got = DRV._lnZ_of_rvs(pooled)
    assert abs(got - target) < 1e-9, (
        "pooled lnZ {} != reported combination {} for a pruned replica".format(got, target))

    # without the reported lnZ the naive 1/n_k rescale gets the pruned replica wrong -- which is
    # exactly the failure mode this guards against
    naive = DRV._lnZ_of_rvs(DRV._pool_replica_rvs([raw_a, pruned_b], _S()))
    assert abs(naive - target) > 0.05, (
        "expected the naive rescale to mis-weight a pruned replica; it did not, so this test "
        "no longer demonstrates the hazard")




def _fairdraw(rng, rec):
    """Resample a record in proportion to its own weights, as the samplers do for fairdraw output.

    The returned rows keep their ORIGINAL weight columns -- which is exactly the trap.
    """
    lw = (numpy.asarray(rec['log_integrand']) + numpy.asarray(rec['log_joint_prior'])
          - numpy.asarray(rec['log_joint_s_prior']))
    w = numpy.exp(lw - numpy.max(lw))
    w = w / w.sum()
    idx = rng.choice(len(w), size=len(w), replace=True, p=w)
    return {k: numpy.asarray(v)[idx] for k, v in rec.items()}


def test_fairdraw_blocks_are_not_weighted_twice():
    """Fairdraw samples were already drawn in proportion to their weights.

    Reusing those weights applies them a second time, so the block follows w^2 rather than w.
    Renormalizing to Z_k/K fixes the block's SCALE but not its SHAPE, which is why
    already_resampled needs its own handling: a fairdraw block is an equal-weight draw from its own
    posterior and must contribute constant weights within the block.
    """
    rng = numpy.random.RandomState(11)
    raw = _replica(rng, 6000, 0.0, 1.3)
    fd = _fairdraw(rng, raw)
    lnZ = [0.0, 0.0]

    pooled = DRV._pool_replica_rvs([fd, fd], _S(), rep_lnZ=lnZ, already_resampled=True)
    lw = (numpy.asarray(pooled['log_integrand']) + numpy.asarray(pooled['log_joint_prior'])
          - numpy.asarray(pooled['log_joint_s_prior']))
    # within-block weights must be CONSTANT -- that is what "already carries its weights" means
    assert numpy.ptp(lw) < 1e-9, "fairdraw block did not get equal within-block weights"
    # and the pooled evidence must still be the reported combination
    assert abs(DRV._lnZ_of_rvs(pooled) - 0.0) < 1e-9

    # the un-handled path leaves a w^2 spread: strictly wider than the w spread it came from
    naive = DRV._pool_replica_rvs([fd, fd], _S(), rep_lnZ=lnZ)
    lw_naive = (numpy.asarray(naive['log_integrand']) + numpy.asarray(naive['log_joint_prior'])
                - numpy.asarray(naive['log_joint_s_prior']))
    assert numpy.ptp(lw_naive) > 1.0, (
        "expected the naive path to retain a spread of weights on an already-resampled block; "
        "this test no longer demonstrates the hazard")
    # concretely: n_eff of the naive pooling is much worse, because w^2 concentrates
    assert DRV._kish_neff_of_rvs(naive) < 0.5 * DRV._kish_neff_of_rvs(pooled)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("replica pooling / coverage-warning invariants hold")
