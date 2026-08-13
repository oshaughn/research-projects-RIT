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
    # ln_weights_from_rvs first: the others now delegate to it (one canonical definition of the
    # importance weight, see the driver docstring).
    for fn in ("_rvs_lnL_convention", "ln_weights_from_rvs", "_rvs_len", "_pool_replica_rvs",
               "_lnZ_of_rvs", "_kish_neff_of_rvs"):
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




def test_cached_log_weights_follow_the_pooled_components():
    """The .dgrid and calibration-posterior exporters PREFER a cached `log_weights` column and only
    fall back to the components.  mcsamplerPortfolio writes that column, so concatenating the
    per-replica caches unchanged would hand those scientific outputs the ORIGINAL weights while the
    evidence used the corrected ones -- replica rebalancing ignored, fairdraw blocks double-weighted
    again, in exactly the products this pooling exists to make consistent."""
    rng = numpy.random.RandomState(21)
    a = _replica(rng, 3000, 0.0, 1.1)
    b = _replica(rng, 3000, 0.4, 1.1)
    for r in (a, b):        # a stale cache, as the portfolio would leave behind
        r['log_weights'] = (numpy.asarray(r['log_integrand']) + numpy.asarray(r['log_joint_prior'])
                            - numpy.asarray(r['log_joint_s_prior']))
    stale_a = numpy.array(a['log_weights'])

    pooled = DRV._pool_replica_rvs([a, b], _S(), rep_lnZ=[0.0, 0.4])
    comp = (numpy.asarray(pooled['log_integrand']) + numpy.asarray(pooled['log_joint_prior'])
            - numpy.asarray(pooled['log_joint_s_prior']))
    assert numpy.allclose(pooled['log_weights'], comp), \
        "cached log_weights disagree with the pooled components the estimate used"
    # and it must actually have CHANGED -- otherwise the test would pass on the buggy code
    assert not numpy.allclose(pooled['log_weights'][:len(stale_a)], stale_a), \
        "pooled log_weights are the stale per-replica values; the cache was not rebuilt"


def test_cached_weights_follow_a_flat_fairdraw_block():
    """The case that matters most: a fairdraw block's cache must become constant too, or the
    exporters reapply the weights the resampling already used."""
    rng = numpy.random.RandomState(22)
    raw = _replica(rng, 4000, 0.0, 1.3)
    fd = _fairdraw(rng, raw)
    fd['log_weights'] = (numpy.asarray(fd['log_integrand']) + numpy.asarray(fd['log_joint_prior'])
                         - numpy.asarray(fd['log_joint_s_prior']))
    pooled = DRV._pool_replica_rvs([fd, fd], _S(), rep_lnZ=[0.0, 0.0], already_resampled=True)
    assert numpy.ptp(pooled['log_weights']) < 1e-9, \
        "fairdraw block's cached log_weights are not constant: exporters would double-weight it"


#
# RAW-FIELD ('integrand'/'joint_prior'/'joint_s_prior') RECORDS UNDER THE LOG CONVENTION.
#
# mcsamplerEnsemble writes no log_* columns at all, and under return_lnI its 'integrand' holds lnL.
# Pooling REWRITES joint_s_prior to force a block's weights, and the equation to solve is
# convention-dependent -- so this path has to be tested in its own right.
#


def _replica_raw(rng, n, lnZ, spread):
    """A GMM-style record: raw columns only, 'integrand' holding lnL (many rows negative)."""
    lnL = rng.normal(0, spread, size=n)
    lnL = lnL - numpy.log(numpy.mean(numpy.exp(lnL))) + lnZ
    return dict(integrand=lnL, joint_prior=numpy.ones(n), joint_s_prior=numpy.ones(n),
                x=rng.normal(size=n))


def _lw_raw(rec, use_lnL):
    return DRV.ln_weights_from_rvs(rec, use_lnL=use_lnL)


def test_raw_flat_block_keeps_a_positive_proposal_density_under_the_log_convention():
    """The reconstruction is  js = ig*jp/exp(target)  for a LINEAR integrand.

    Applied to an lnL record it yields js < 0 for every row with lnL < 0 -- a negative proposal
    density -- and the block weights are not constant, which is the whole purpose of a flat block.
    The log-convention equation is  js = exp(lnL + log(jp) - target).
    """
    rng = numpy.random.RandomState(31)
    fd = _replica_raw(rng, 4000, 0.0, 1.2)
    assert numpy.any(numpy.asarray(fd['integrand']) < 0), "test record has no lnL < 0 rows"

    pooled = DRV._pool_replica_rvs([fd, fd], _S(), rep_lnZ=[0.0, 0.0],
                                   already_resampled=True, use_lnL=True)
    js = numpy.asarray(pooled['joint_s_prior'], dtype=float)
    assert numpy.all(js > 0), "pooling produced a NON-POSITIVE sampling prior on {} rows".format(
        int(numpy.sum(js <= 0)))
    lw = _lw_raw(pooled, use_lnL=True)
    assert numpy.all(numpy.isfinite(lw))
    assert numpy.ptp(lw) < 1e-9, "fairdraw block did not get equal within-block weights"
    assert abs(DRV._lnZ_of_rvs(pooled, use_lnL=True) - 0.0) < 1e-9

    # the un-fixed reading is the hazard this pins: negative densities and a non-flat block
    wrong = DRV._pool_replica_rvs([fd, fd], _S(), rep_lnZ=[0.0, 0.0],
                                  already_resampled=True, use_lnL=False)
    js_wrong = numpy.asarray(wrong['joint_s_prior'], dtype=float)
    assert numpy.any(js_wrong <= 0), (
        "expected the linear reconstruction to produce a non-positive density on an lnL record; "
        "this test no longer demonstrates the hazard")
    assert numpy.ptp(_lw_raw(wrong, use_lnL=True)[numpy.isfinite(_lw_raw(wrong, use_lnL=True))]) > 1.0


def test_raw_pooling_reproduces_the_combined_evidence_under_the_log_convention():
    rng = numpy.random.RandomState(32)
    reps = [_replica_raw(rng, 4000, 0.0, 1.1), _replica_raw(rng, 3000, 0.2, 1.1),
            _replica_raw(rng, 5000, -0.1, 0.9)]
    Zk = [numpy.mean(numpy.exp(_lw_raw(r, use_lnL=True))) for r in reps]
    target = numpy.log(numpy.mean(Zk))
    pooled = DRV._pool_replica_rvs(reps, _S(), use_lnL=True)
    got = DRV._lnZ_of_rvs(pooled, use_lnL=True)
    assert abs(got - target) < 1e-9, "raw pooled lnZ {} != combination {}".format(got, target)
    assert len(pooled['x']) == 12000


def test_raw_pooling_applies_the_reported_lnZ_renormalization():
    """The raw branch used to rescale joint_s_prior by a hardcoded K*n_k, which is only the
    FALLBACK value of `scale`.  So whenever a reported per-replica lnZ was available it skipped the
    renormalization the log branch applied, and a pruned/thresholded replica was mis-weighted."""
    rng = numpy.random.RandomState(33)
    raw_a = _replica_raw(rng, 4000, 0.0, 1.0)
    raw_b = _replica_raw(rng, 4000, 0.3, 1.0)
    lnZ = [0.0, 0.3]
    lw_b = _lw_raw(raw_b, use_lnL=True)
    keep = numpy.argsort(lw_b)[len(lw_b) // 2:]
    pruned_b = {k: numpy.asarray(v)[keep] for k, v in raw_b.items()}

    target = numpy.log(numpy.mean(numpy.exp(numpy.array(lnZ))))
    pooled = DRV._pool_replica_rvs([raw_a, pruned_b], _S(), rep_lnZ=lnZ, use_lnL=True)
    got = DRV._lnZ_of_rvs(pooled, use_lnL=True)
    assert abs(got - target) < 1e-9, (
        "raw pooled lnZ {} != reported combination {} for a pruned replica".format(got, target))

    naive = DRV._lnZ_of_rvs(DRV._pool_replica_rvs([raw_a, pruned_b], _S(), use_lnL=True),
                            use_lnL=True)
    assert abs(naive - target) > 0.05, (
        "expected the un-renormalized rescale to mis-weight a pruned replica; it did not, so this "
        "test no longer demonstrates the hazard")


def test_raw_cached_weights_are_rebuilt_in_the_right_convention():
    """The .dgrid and calibration exporters PREFER a cached 'log_weights'.  Rebuilding it with the
    linear formula on an lnL record hands them log(lnL)-flattened weights -- the original bug, now
    re-entered through the back door of the pooled record."""
    rng = numpy.random.RandomState(34)
    a = _replica_raw(rng, 3000, 0.0, 1.1)
    b = _replica_raw(rng, 3000, 0.4, 1.1)
    for r in (a, b):
        r['log_weights'] = _lw_raw(r, use_lnL=True)
    pooled = DRV._pool_replica_rvs([a, b], _S(), rep_lnZ=[0.0, 0.4], use_lnL=True)
    assert numpy.allclose(pooled['log_weights'], _lw_raw(pooled, use_lnL=True)), \
        "cached log_weights disagree with the pooled components the estimate used"
    assert numpy.all(numpy.isfinite(pooled['log_weights'])), \
        "rebuilt cache dropped the lnL <= 0 rows"


def test_raw_linear_records_are_unaffected():
    """The P2 guard, at the pooling level: a record that really does store linear L must pool
    exactly as before."""
    rng = numpy.random.RandomState(35)
    reps = []
    for lnZ in (0.0, 0.25):
        lnL = rng.normal(0, 1.0, size=3000)
        lnL = lnL - numpy.log(numpy.mean(numpy.exp(lnL))) + lnZ
        reps.append(dict(integrand=numpy.exp(lnL), joint_prior=numpy.ones(3000),
                         joint_s_prior=numpy.ones(3000), x=rng.normal(size=3000)))
    target = numpy.log(numpy.mean([numpy.mean(numpy.exp(_lw_raw(r, use_lnL=False))) for r in reps]))
    pooled = DRV._pool_replica_rvs(reps, _S())              # default: linear, as before
    got = DRV._lnZ_of_rvs(pooled)
    assert abs(got - target) < 1e-9, "linear pooling changed: {} vs {}".format(got, target)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("replica pooling / coverage-warning invariants hold")
