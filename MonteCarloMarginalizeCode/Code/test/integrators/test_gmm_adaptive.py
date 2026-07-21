"""Unit tests for the flexible / data-driven GMM component allocation added to
RIFT.integrators.gaussian_mixture_model:

  * fit_gmm_adaptive  -- choose k by BIC, then prune dead components
  * gmm.prune_components
  * gmm._match_components -- O(k^3) Hungarian == old O(k!) permutation optimum

Backend note: gaussian_mixture_model uses cupy when a GPU is visible, numpy
otherwise.  These tests are backend-agnostic; run with CUDA_VISIBLE_DEVICES set
to a GPU, or with a numpy-only build.  No ILE data required (seconds to run).

Run:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
    python test/integrators/test_gmm_adaptive.py
"""
from __future__ import print_function
import sys, itertools
import numpy as np
from RIFT.integrators import gaussian_mixture_model as GMM

cvt = GMM.identity_convert
rng = np.random.RandomState(20250721)


def _bounds(d, lo=-8., hi=8.):
    b = np.empty((d, 2)); b[:, 0] = lo; b[:, 1] = hi
    return GMM.xpy_default.array(b)


def test_bic_picks_one_for_single_gaussian():
    """A single Gaussian blob should be modeled with k=1 (BIC penalizes extra
    components that do not improve the weighted likelihood)."""
    d = 3
    X = rng.normal(0.0, 0.7, size=(4000, d))
    model = GMM.fit_gmm_adaptive(GMM.xpy_default.array(X), _bounds(d), k_max=8,
                                 defensive_frac=0.0)
    print("  single-gaussian -> k =", model.k)
    assert model.k == 1, "expected k=1 for a single blob, got %d" % model.k


def test_bic_grows_for_separated_modes():
    """Three well-separated blobs should earn more than one component."""
    d = 2
    centers = np.array([[-5., -5.], [0., 5.], [5., -5.]])
    X = np.vstack([rng.normal(c, 0.4, size=(1500, d)) for c in centers])
    model = GMM.fit_gmm_adaptive(GMM.xpy_default.array(X), _bounds(d), k_max=8,
                                 defensive_frac=0.0)
    print("  three-modes -> k =", model.k)
    assert model.k >= 3, "expected k>=3 for three separated modes, got %d" % model.k


def test_bic_respects_weights():
    """With importance weights that select one of two blobs, BIC should prefer
    fewer components (only the up-weighted blob carries effective mass)."""
    d = 2
    A = rng.normal([-4, 0], 0.4, size=(2000, d))
    B = rng.normal([4, 0], 0.4, size=(2000, d))
    X = np.vstack([A, B])
    # up-weight only blob A
    lw = np.concatenate([np.zeros(len(A)), -50.0 * np.ones(len(B))])
    model = GMM.fit_gmm_adaptive(GMM.xpy_default.array(X), _bounds(d),
                                 log_sample_weights=GMM.xpy_default.array(lw),
                                 k_max=8, defensive_frac=0.0)
    print("  weighted-one-of-two -> k =", model.k)
    assert model.k <= 2, "expected small k when weights select one blob, got %d" % model.k
    # the fitted mass should sit near blob A (-4,0), not the midpoint
    means = np.array([cvt(m) for m in model.means])
    mean_un = model._unnormalize(GMM.xpy_default.array(means))
    mx = float(cvt(mean_un)[:, 0].mean())
    print("  weighted mean x =", mx)
    assert mx < -1.0, "weighted fit should sit on the up-weighted blob"


def test_prune_removes_dead_components():
    d = 2
    model = GMM.gmm(4, _bounds(d))
    # fit to a single blob so 3 of 4 components collapse to ~zero weight
    X = rng.normal(0.0, 0.5, size=(3000, d))
    model.fit(GMM.xpy_default.array(X))
    k_before = model.k
    model.prune_components(weight_floor=1e-2)
    print("  prune: k %d -> %d" % (k_before, model.k))
    assert model.k <= k_before
    w = np.asarray(cvt(model.weights), dtype=float)
    assert abs(w.sum() - 1.0) < 1e-6, "weights must renormalize to 1"
    assert len(model.means) == model.k and len(model.covariances) == model.k


def test_matching_matches_permutation_optimum():
    """Hungarian _match_components must reproduce the exact permutation optimum."""
    def objective(order, om, oc, nm, nc):
        val = 0.0
        for i, j in enumerate(order):
            diff = nm[j] - om[i]
            val += np.sqrt(diff @ np.linalg.inv(oc[i]) @ diff)
            val += np.sqrt(diff @ np.linalg.inv(nc[j]) @ diff)
        return val
    for k in [2, 3, 4, 5]:
        d = 3
        model = GMM.gmm(k, _bounds(d))
        new = GMM.gmm(k, _bounds(d))
        model.d = new.d = d
        model.means = [GMM.xpy_default.array(rng.randn(d)) for _ in range(k)]
        new.means = [GMM.xpy_default.array(rng.randn(d)) for _ in range(k)]
        mk = lambda: (lambda A: GMM.xpy_default.array(A @ A.T + np.eye(d)))(rng.randn(d, d))
        model.covariances = [mk() for _ in range(k)]
        new.covariances = [mk() for _ in range(k)]
        om = [cvt(m) for m in model.means]; nm = [cvt(m) for m in new.means]
        oc = [cvt(c) for c in model.covariances]; nc = [cvt(c) for c in new.covariances]
        # brute-force optimum
        best = min(itertools.permutations(range(k)),
                   key=lambda o: objective(o, om, oc, nm, nc))
        got = model._match_components(new)
        assert abs(objective(got, om, oc, nm, nc) - objective(best, om, oc, nm, nc)) < 1e-9, \
            "k=%d: Hungarian objective != permutation optimum" % k
    print("  matching == permutation optimum for k in 2..5")


if __name__ == "__main__":
    tests = [test_bic_picks_one_for_single_gaussian,
             test_bic_grows_for_separated_modes,
             test_bic_respects_weights,
             test_prune_removes_dead_components,
             test_matching_matches_permutation_optimum]
    nfail = 0
    for t in tests:
        try:
            print("[RUN]", t.__name__)
            t()
            print("[PASS]", t.__name__)
        except AssertionError as e:
            nfail += 1
            print("[FAIL]", t.__name__, "->", e)
    print("\n%d/%d passed" % (len(tests) - nfail, len(tests)))
    sys.exit(1 if nfail else 0)
