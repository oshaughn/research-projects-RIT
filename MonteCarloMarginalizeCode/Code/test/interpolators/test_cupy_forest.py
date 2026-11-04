"""GPU forest predictor (RIFT.interpolators.cupy_forest) against sklearn. Needs cupy and a device.

The adapter must reproduce sklearn's decisions exactly: sklearn casts X to float32 and tests
x <= threshold in float64. The threshold test below queries points that sit exactly on the float32
value nearest a threshold, where storing nearest-rounded float32 thresholds would send them the wrong
way; it fails for that broken variant and passes for the round-down rule the adapter uses.
"""
import numpy as np
import pytest

try:
    import cupy as cp
    cp.zeros(1)
except Exception as err:     # no cupy, or no usable device
    pytest.skip("needs cupy and a CUDA device: %s" % type(err).__name__, allow_module_level=True)

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import RIFT.interpolators.cupy_forest as cf


def _data(n=4000, d=5, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    y = np.sin(3 * X[:, 0]) * X[:, 1] + 0.3 * X[:, 2] ** 2 + rng.normal(0, 0.05, n)
    return X, y, rng.uniform(0.2, 3.0, n)


@pytest.mark.parametrize("cls", [ExtraTreesRegressor, RandomForestRegressor])
@pytest.mark.parametrize("n_out", [1, 3])
def test_predictions_match_sklearn(cls, n_out):
    X, y, w = _data()
    Y = y if n_out == 1 else np.c_[y, -2 * y, X[:, 3]]
    m = cls(n_estimators=20, n_jobs=1, random_state=1).fit(X, Y, sample_weight=w)
    f = cf.CupyForest(m, batch_size=1000)                # several batches
    Xq = np.r_[X[:500], np.random.default_rng(4).uniform(-1.3, 1.3, (3000, X.shape[1]))]
    got = cp.asnumpy(f.predict(Xq))
    ref = m.predict(Xq)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) < 1e-12


def test_threshold_rounding_is_round_down():
    X, y, w = _data(n=3000, d=3)
    m = ExtraTreesRegressor(n_estimators=5, n_jobs=1, random_state=2).fit(X, y)
    t = m.estimators_[0].tree_
    internal = np.flatnonzero(t.children_left >= 0)
    thr, feat = t.threshold[internal], t.feature[internal]
    t_near = thr.astype(np.float32)
    hit = t_near.astype(np.float64) > thr                 # nearest float32 lies above the threshold
    assert hit.sum() > 10
    # queries exactly at that float32: sklearn sends them right (x > threshold)
    Xq = np.tile(np.median(X, axis=0), (int(hit.sum()), 1))
    Xq[np.arange(len(Xq)), feat[hit]] = t_near[hit].astype(np.float64)
    ref = m.predict(Xq)
    good = cp.asnumpy(cf.CupyForest(m).predict(Xq))
    assert np.max(np.abs(good - ref)) < 1e-12
    broken = cf._threshold_f32_down
    try:
        cf._threshold_f32_down = lambda a: a.astype(np.float32)   # nearest rounding: the defect
        bad = cp.asnumpy(cf.CupyForest(m).predict(Xq))
    finally:
        cf._threshold_f32_down = broken
    assert np.max(np.abs(bad - ref)) > 1e-6


def test_corrupted_leaf_is_detected():
    X, y, w = _data()
    m = ExtraTreesRegressor(n_estimators=10, n_jobs=1, random_state=0).fit(X, y)
    f = cf.CupyForest(m)
    leaf = int(cp.asnumpy(cp.flatnonzero(f._left < 0)[0]))
    f._val[leaf] += 1.0
    assert np.max(np.abs(cp.asnumpy(f.predict(X)) - m.predict(X))) > 1e-3
