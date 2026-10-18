"""GPU prediction for an already fitted sklearn tree ensemble (ExtraTrees / RandomForest regressor).

A prediction adapter, never a fitter. The trees are copied to the device once; each CUDA thread walks
one sample through every tree. Decisions are the same as sklearn's: sklearn casts X to float32 and
tests ``x <= threshold`` with a float64 threshold, which for float32 x is the same test as
``x <= t32`` with t32 the largest float32 not above the threshold. Leaf values stay float64 and are
summed in tree order, then divided by the number of trees, as sklearn does; only the order of that
sum differs (sklearn's is thread-scheduled), so predictions agree to float64 roundoff.
"""
import numpy as np

_KERNEL = r"""
extern "C" __global__
void forest_predict(const float* __restrict__ X, const long long n, const int d,
                    const int* __restrict__ left, const int* __restrict__ right,
                    const int* __restrict__ feature, const float* __restrict__ thr,
                    const double* __restrict__ value, const int n_out,
                    const int* __restrict__ roots, const int n_trees, double* __restrict__ out)
{
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const float* x = X + i * d;
    double acc[MAX_OUT];
    for (int k = 0; k < n_out; k++) acc[k] = 0.0;
    for (int t = 0; t < n_trees; t++) {
        int node = roots[t];
        int l = left[node];
        while (l >= 0) {
            node = (x[feature[node]] <= thr[node]) ? l : right[node];
            l = left[node];
        }
        const double* v = value + (long long)node * n_out;
        for (int k = 0; k < n_out; k++) acc[k] += v[k];
    }
    for (int k = 0; k < n_out; k++) out[i * n_out + k] = acc[k] / n_trees;
}
"""
MAX_OUT = 8


def _threshold_f32_down(thr):
    t32 = thr.astype(np.float32)
    above = t32.astype(np.float64) > thr
    t32[above] = np.nextafter(t32[above], np.float32(-np.inf))
    return t32


class CupyForest:
    """Device copy of a fitted sklearn forest regressor; ``predict`` matches ``est.predict``.

    ``predict`` accepts numpy or cupy input and returns a cupy array, shape (n,) for one output and
    (n, n_outputs) otherwise. Inputs must be finite and below float32 range, as for sklearn
    (callers already mask such rows)."""

    def __init__(self, est, batch_size=1 << 20):
        import cupy as cp
        trees = [e.tree_ for e in est.estimators_]
        self.n_features_in_ = int(est.n_features_in_)
        self.n_outputs = int(trees[0].value.shape[1])
        if self.n_outputs > MAX_OUT:
            raise ValueError("CupyForest supports at most %d outputs" % MAX_OUT)
        if any(t.value.shape[2] != 1 for t in trees):
            raise ValueError("CupyForest supports regressors only")
        sizes = np.array([t.node_count for t in trees], dtype=np.int64)
        if sizes.sum() >= 2 ** 31:
            raise ValueError("forest too large for int32 node indices (%d nodes)" % sizes.sum())
        offs = np.r_[0, np.cumsum(sizes)[:-1]]
        n = int(sizes.sum())
        # filled one tree at a time: host temporaries stay at one tree's size
        self._left = cp.empty(n, dtype=cp.int32)
        self._right = cp.empty(n, dtype=cp.int32)
        self._feat = cp.empty(n, dtype=cp.int32)
        self._thr = cp.empty(n, dtype=cp.float32)
        self._val = cp.empty((n, self.n_outputs), dtype=cp.float64)
        for t, o in zip(trees, offs):
            sl = slice(int(o), int(o) + t.node_count)
            cl, cr = t.children_left, t.children_right
            self._left[sl] = cp.asarray(np.where(cl >= 0, cl + o, -1).astype(np.int32))
            self._right[sl] = cp.asarray(np.where(cr >= 0, cr + o, -1).astype(np.int32))
            self._feat[sl] = cp.asarray(np.maximum(t.feature, 0).astype(np.int32))
            self._thr[sl] = cp.asarray(_threshold_f32_down(t.threshold))
            self._val[sl] = cp.asarray(np.ascontiguousarray(t.value[:, :, 0], dtype=np.float64))
        self._roots = cp.asarray(offs.astype(np.int32))
        self.n_trees = len(trees)
        self.n_nodes = int(sizes.sum())
        self.batch_size = int(batch_size)
        self._kern = cp.RawKernel(_KERNEL.replace("MAX_OUT", str(MAX_OUT)), "forest_predict")

    @classmethod
    def from_device_trees(cls, trees, n_features, batch_size=1 << 20):
        """From per-tree dicts of arrays (left, right, feat, thr float64, val; numpy or cupy), as grown by
        RIFT.interpolators.cupy_extratrees; leaves have left = right = -1. Single output. The device
        arrays are allocated once and filled one tree at a time."""
        import cupy as cp
        self = cls.__new__(cls)
        self.n_features_in_, self.n_outputs = int(n_features), 1
        sizes = np.array([len(t["left"]) for t in trees], dtype=np.int64)
        if sizes.sum() >= 2 ** 31:
            raise ValueError("forest too large for int32 node indices (%d nodes)" % sizes.sum())
        offs = np.r_[0, np.cumsum(sizes)[:-1]]
        n = int(sizes.sum())
        self._left = cp.empty(n, dtype=cp.int32); self._right = cp.empty(n, dtype=cp.int32)
        self._feat = cp.empty(n, dtype=cp.int32); self._thr = cp.empty(n, dtype=cp.float32)
        self._val = cp.empty((n, 1), dtype=cp.float64)
        for t, o in zip(trees, offs):
            sl = slice(int(o), int(o) + len(t["left"]))
            lf, rt = cp.asarray(t["left"]), cp.asarray(t["right"])
            self._left[sl] = cp.where(lf >= 0, lf + int(o), -1)
            self._right[sl] = cp.where(rt >= 0, rt + int(o), -1)
            self._feat[sl] = cp.maximum(cp.asarray(t["feat"]), 0)
            thr = cp.asarray(t["thr"], dtype=cp.float64)
            t32 = thr.astype(cp.float32)
            self._thr[sl] = cp.where(t32.astype(cp.float64) > thr, cp.nextafter(t32, cp.float32(-cp.inf)), t32)
            self._val[sl, 0] = cp.asarray(t["val"])
        self._roots = cp.asarray(offs.astype(np.int32))
        self.n_trees, self.n_nodes, self.batch_size = len(trees), n, int(batch_size)
        self._kern = cp.RawKernel(_KERNEL.replace("MAX_OUT", str(MAX_OUT)), "forest_predict")
        return self

    @property
    def device_bytes(self):
        return sum(int(a.nbytes) for a in (self._left, self._right, self._feat, self._thr, self._val, self._roots))

    def predict(self, X):
        import cupy as cp
        X = cp.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("expected X of shape (n, %d)" % self.n_features_in_)
        n = X.shape[0]
        out = cp.empty((n, self.n_outputs), dtype=cp.float64)
        threads = 128
        for s0 in range(0, n, self.batch_size):
            xb = cp.ascontiguousarray(X[s0:s0 + self.batch_size].astype(cp.float32))
            nb = xb.shape[0]
            ob = out[s0:s0 + nb]
            self._kern(((nb + threads - 1) // threads,), (threads,),
                       (xb, np.int64(nb), np.int32(self.n_features_in_), self._left, self._right, self._feat,
                        self._thr, self._val, np.int32(self.n_outputs), self._roots, np.int32(self.n_trees), ob))
        return out[:, 0] if self.n_outputs == 1 else out
