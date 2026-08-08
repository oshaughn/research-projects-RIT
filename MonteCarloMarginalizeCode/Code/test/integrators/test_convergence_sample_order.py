#!/usr/bin/env python
"""`convergence_test_NormalSubIntegrals` needs DRAW order, not weight order.

The save-P thresholding block reindexes every `_rvs` column by cumulative-weight order
(`mcsampler.py:739-752`), so surviving rows come out sorted by weight ascending.  The test then
splits them into `ncopies` CONTIGUOUS segments and assumes those are independent -- but sorted rows
put the smallest weights in the first segment and the largest in the last, so the sub-integrals
differ by construction.

The long-standing workaround (recompute the weights from the components rather than reading the
cached column) does not help: the components were permuted by the same reindexing.  `sample_n`,
written just before that block as an iteration number, is what survives it.

Run:  python test_convergence_sample_order.py
"""
import numpy


def _thresholded_record(n=2000, seed=0):
    """An _rvs record after the save-P threshold block has reindexed it by weight."""
    rng = numpy.random.RandomState(seed)
    rvs = dict(integrand=numpy.exp(rng.normal(5.0, 2.0, size=n)),
               joint_prior=numpy.ones(n), joint_s_prior=numpy.ones(n))
    rvs["weights"] = rvs["integrand"] * rvs["joint_prior"] / rvs["joint_s_prior"]
    rvs["sample_n"] = numpy.arange(n)
    wt = rvs["integrand"] * rvs["joint_prior"] / rvs["joint_s_prior"]
    idx_sorted = numpy.lexsort((numpy.arange(len(wt)), wt))
    pairs = numpy.array([[k, wt[k]] for k in idx_sorted])
    cum = numpy.cumsum(pairs[:, 1]); cum = cum / cum[-1]
    keep = [int(pairs[k, 0]) for k, v in enumerate(cum > 1e-7) if v]
    return {k: v[keep] for k, v in rvs.items()}


def _ascending_fraction(a):
    a = numpy.asarray(a, dtype=float)
    return float(numpy.mean(numpy.diff(a) >= 0))


def test_recomputing_does_not_undo_the_sort():
    """Pins why the old workaround was ineffective, so it is not reinstated."""
    rvs = _thresholded_record()
    cached = _ascending_fraction(rvs["weights"])
    recomputed = _ascending_fraction(rvs["integrand"] * rvs["joint_prior"] / rvs["joint_s_prior"])
    assert cached > 0.99, cached
    assert recomputed > 0.99, (
        "recomputed weights came out unsorted; this record no longer reproduces the hazard")


def test_sample_n_restores_draw_order():
    rvs = _thresholded_record()
    w = rvs["integrand"] * rvs["joint_prior"] / rvs["joint_s_prior"]
    restored = numpy.asarray(w)[numpy.argsort(numpy.asarray(rvs["sample_n"]))]
    frac = _ascending_fraction(restored)
    assert 0.35 < frac < 0.65, (
        "draw order should look random (~0.5 ascending), got {:.3f}".format(frac))


def test_segments_are_comparable_only_after_reordering():
    """The property the test actually depends on: contiguous segments must have comparable mass."""
    rvs = _thresholded_record()
    w = numpy.asarray(rvs["integrand"] * rvs["joint_prior"] / rvs["joint_s_prior"])

    def segment_ratio(arr, ncopies=10):
        part = len(arr) // ncopies
        sums = [arr[i * part:(i + 1) * part].sum() for i in range(ncopies)]
        return max(sums) / max(min(sums), 1e-300)

    sorted_ratio = segment_ratio(w)
    restored_ratio = segment_ratio(w[numpy.argsort(numpy.asarray(rvs["sample_n"]))])
    assert sorted_ratio > 100 * restored_ratio, (
        "weight-ordered segments should be wildly unequal vs draw-ordered; got {:.3g} vs {:.3g}"
        .format(sorted_ratio, restored_ratio))




def _threshold_block(rvs):
    """Faithful replica of mcsampler.py:727-752, including the sample_n (re)creation."""
    if "integrand" in rvs:
        rvs["sample_n"] = numpy.arange(len(rvs["integrand"]))
        wt = rvs["integrand"] * rvs["joint_prior"] / rvs["joint_s_prior"]
        idx = numpy.lexsort((numpy.arange(len(wt)), wt))
        pairs = numpy.array([[k, wt[k]] for k in idx])
        cum = numpy.cumsum(pairs[:, 1]); cum = cum / cum[-1]
        keep = [int(pairs[k, 0]) for k, v in enumerate(cum > 1e-7) if v]
        for k in list(rvs.keys()):
            rvs[k] = rvs[k][keep]
    return rvs


def _fresh(n, seed):
    r = numpy.random.RandomState(seed)
    return dict(integrand=numpy.exp(r.normal(5.0, 2.0, size=n)),
                joint_prior=numpy.ones(n), joint_s_prior=numpy.ones(n))


def test_sample_n_is_stale_and_short_on_a_reused_sampler():
    """Why reordering by sample_n is NOT a valid fix, part 1.

    A reused sampler appends new rows to the weight columns.  Until the threshold block re-runs,
    `sample_n` still has the PREVIOUS length, so indexing the weights by it silently drops every
    new sample -- numpy fancy-indexing just returns the shorter array."""
    rvs = _threshold_block(_fresh(1500, 1))
    new = _fresh(1500, 2)
    for k in ("integrand", "joint_prior", "joint_s_prior"):
        rvs[k] = numpy.hstack([rvs[k], new[k]])
    assert len(rvs["sample_n"]) < len(rvs["integrand"]), "expected a stale, short sample_n"
    w = rvs["integrand"] * rvs["joint_prior"] / rvs["joint_s_prior"]
    reordered = numpy.asarray(w)[numpy.argsort(numpy.asarray(rvs["sample_n"]))]
    assert len(reordered) < len(w), (
        "indexing by a short sample_n should silently drop samples; got {} of {}".format(
            len(reordered), len(w)))


def test_sample_n_does_not_encode_draw_order_after_reuse():
    """Why reordering by sample_n is NOT a valid fix, part 2.

    Even once the block re-runs and the lengths agree, `sample_n = arange(len(...))` was assigned to
    an ALREADY-permuted array, so it encodes the order at the start of that call rather than the
    draw order.  The segment imbalance the test cares about is only partly repaired."""
    rvs = _threshold_block(_fresh(1500, 1))
    new = _fresh(1500, 2)
    for k in ("integrand", "joint_prior", "joint_s_prior"):
        rvs[k] = numpy.hstack([rvs[k], new[k]])
    rvs = _threshold_block(rvs)
    assert len(rvs["sample_n"]) == len(rvs["integrand"])          # lengths agree again
    w = numpy.asarray(rvs["integrand"] * rvs["joint_prior"] / rvs["joint_s_prior"])
    restored = w[numpy.argsort(numpy.asarray(rvs["sample_n"]))]
    frac = _ascending_fraction(restored)
    assert frac > 0.65, (
        "after reuse, sample_n should NOT restore draw order (~0.5); got {:.3f} -- if this now "
        "passes at ~0.5 the samplers grew stable append-time ids and the caveat can be revisited"
        .format(frac))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("convergence-test sample ordering holds")
