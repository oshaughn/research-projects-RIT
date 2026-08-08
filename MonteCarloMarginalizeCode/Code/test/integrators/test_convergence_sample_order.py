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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("convergence-test sample ordering holds")
