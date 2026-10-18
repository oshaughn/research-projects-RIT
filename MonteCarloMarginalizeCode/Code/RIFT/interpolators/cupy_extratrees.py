"""ExtraTrees regression forest grown on the GPU, with sklearn's ExtraTreesRegressor algorithm.

Reproduces ExtraTreesRegressor(criterion='squared_error', max_features=1.0, min_samples_split=2,
min_samples_leaf=1, bootstrap=False) with sample weights. At every node and for every feature that is
not constant on the node (max <= min + 1e-7, sklearn's FEATURE_THRESHOLD, in float32 X), the threshold
is drawn uniformly on [min, max) (a draw equal to max is moved to min, as in sklearn); the split with
the largest weighted-MSE proxy improvement S_L^2/W_L + S_R^2/W_R is kept, ties (equal partitions)
broken uniformly as sklearn's random feature order does (tolerance 1e-13 of the proxy). A node is a leaf when it
holds fewer than two samples, its weighted impurity is <= double epsilon, or every feature is constant
on it; its value is the weighted mean of y. Samples go left when float32(x) <= threshold.

The trees are grown level by level, a group of trees at a time, one CUDA thread per (tree, sample)
pair. Node and split sums are fixed-point integer sums (see _quanta), so a fit with a given
random_state is bit-for-bit reproducible; the uniforms come from a counter-based hash of (seed, tree,
node, feature), so they do not depend on how trees are grouped. A forest equals an sklearn forest in
distribution, not draw for draw; ``_reference_tree`` below is the same algorithm in numpy, driven by supplied uniforms, against
which a GPU tree grown from the same uniforms is checked exactly. Prediction is by
RIFT.interpolators.cupy_forest.CupyForest.
"""
import numpy as np

_FEATURE_THRESHOLD = 1e-7
_EPS = np.finfo(np.float64).eps

_KERNELS = r"""
__device__ __forceinline__ int ord_f(float f) { int i = __float_as_int(f); return i >= 0 ? i : i ^ 0x7FFFFFFF; }

extern "C" __global__
void node_stats(const int* __restrict__ act, const long long n_act, const int N, const int F,
                const float* __restrict__ X, const long long* __restrict__ qw, const long long* __restrict__ qs,
                const long long* __restrict__ qq, const int* __restrict__ slot,
                unsigned long long* W, unsigned long long* S1, unsigned long long* S2, int* cnt, int* cntw,
                int* fmin, int* fmax)
{
    long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n_act) return;
    int p = act[k]; int i = p % N; int s = slot[p];
    // fixed-point integer sums: exact, so the result does not depend on the order of the atomics
    atomicAdd(W + s, (unsigned long long)qw[i]); atomicAdd(S1 + s, (unsigned long long)qs[i]);
    atomicAdd(S2 + s, (unsigned long long)qq[i]); atomicAdd(cnt + s, 1);
    if (qw[i] > 0) atomicAdd(cntw + s, 1);
    const float* x = X + (long long)i * F;
    for (int f = 0; f < F; f++) {
        int v = ord_f(x[f]);
        atomicMin(fmin + (long long)s * F + f, v); atomicMax(fmax + (long long)s * F + f, v);
    }
}

extern "C" __global__
void split_stats(const int* __restrict__ act, const long long n_act, const int N, const int F,
                 const float* __restrict__ X, const long long* __restrict__ qw, const long long* __restrict__ qs,
                 const int* __restrict__ slot, const unsigned char* __restrict__ splitting,
                 const double* __restrict__ thr, const unsigned char* __restrict__ cand,
                 unsigned long long* LW, unsigned long long* LS, int* LCW)
{
    long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n_act) return;
    int p = act[k]; int s = slot[p];
    if (!splitting[s]) return;
    int i = p % N;
    unsigned long long wi = (unsigned long long)qw[i], wy = (unsigned long long)qs[i];
    const float* x = X + (long long)i * F;
    for (int f = 0; f < F; f++) {
        long long sf = (long long)s * F + f;
        if (cand[sf] && (double)x[f] <= thr[sf]) {
            atomicAdd(LW + sf, wi); atomicAdd(LS + sf, wy);
            if (qw[i] > 0) atomicAdd(LCW + sf, 1);
        }
    }
}

extern "C" __global__
void route(const int* __restrict__ act, const long long n_act, const int N, const int F,
           const float* __restrict__ X, int* slot, const unsigned char* __restrict__ splitting,
           const int* __restrict__ bfeat, const double* __restrict__ bthr, const int* __restrict__ child0,
           unsigned char* alive)
{
    long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n_act) return;
    int p = act[k]; int s = slot[p];
    if (!splitting[s]) { alive[k] = 0; return; }
    int i = p % N;
    float xv = X[(long long)i * F + bfeat[s]];
    slot[p] = child0[s] + (((double)xv <= bthr[s]) ? 0 : 1);
    alive[k] = 1;
}
"""


# splitmix64 applied in turn to seed, tree, node and feature, each key through its own mixing stage and
# odd multiplier (seeding z with the seed and adding the tree would alias (s, t+1) with (s+1, t));
# 53-bit uniform on [0, 1)
_HASH_UNIFORM = r'''
unsigned long long z = 0ULL;
const unsigned long long keys[4] = {(unsigned long long)seed, (unsigned long long)tree,
                                    (unsigned long long)node, (unsigned long long)f};
const unsigned long long mult[4] = {0xD1B54A32D192ED03ULL, 0xAEF17502108EF2D9ULL,
                                    0xF39CC0605CEDC835ULL, 0xDB4F0B9175AE2165ULL};
for (int k = 0; k < 4; k++) {
    z += keys[k] * mult[k] + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
}
u = (double)(z >> 11) * (1.0 / 9007199254740992.0);
'''


def _pick(xp, proxy, u_tie):
    """Best feature per row. Features whose proxy is within 1e-13 (relative) of the best are tied --
    typically several features give the same partition of a small node -- and one is chosen uniformly
    with u_tie, as sklearn's random feature order does. A fixed rule (argmax: lowest index) would bias
    which feature carries the cut. The tolerance is relative to the whole proxy, which includes the node
    offset S^2/W, so it must sit above the proxy's roundoff (a few eps for the small nodes where ties
    occur); improvements closer than 1e-13 S^2/W also count as tied."""
    pmax = xp.max(proxy, axis=1, keepdims=True)
    tied = (proxy >= pmax - 1e-13 * xp.abs(pmax)) & xp.isfinite(proxy)
    n_tied = xp.sum(tied, axis=1)
    j = xp.minimum(xp.floor(u_tie * n_tied), xp.maximum(n_tied - 1, 0)).astype(xp.int64)
    rank = xp.cumsum(tied, axis=1) - 1
    return xp.argmax(tied & (rank == j[:, None]), axis=1).astype(xp.int32)


def _quanta(y, w):
    """Fixed-point integer quanta of w, w (y - c) and w (y - c)^2, c the weighted mean of y.

    Node and split sums are then sums of int64 values, exact in any order, so a seeded fit is
    reproducible although the GPU adds them with atomics. Each scale is a power of two chosen so no
    node sum (at most the sum over all samples of one tree) can overflow 2^62. Centering y shifts
    every candidate split's proxy at a node by the same constant, so the chosen split is unchanged;
    leaf values add c back."""
    y = np.asarray(y, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    c = float(np.sum(w * y) / np.sum(w)) if np.sum(w) > 0 else 0.0
    yc = y - c
    out = {"c": c, "q": []}
    for key, v in (("sw", w), ("ss", w * yc), ("sq", w * yc * yc)):
        tot = float(np.sum(np.abs(v)))
        scale = 2.0 ** int(np.floor(np.log2(2.0 ** 62 / max(tot, 1e-300)))) if tot > 0 else 1.0
        out[key] = scale
        out["q"].append(np.rint(v * scale).astype(np.int64))
    return out


def _from_q(xp, a, scale):
    """Integer sums (stored as uint64, two's complement) back to float64."""
    return a.view(xp.int64).astype(xp.float64) / scale


def _decode_ord(cp, a):
    a = a.astype(cp.int32)
    return cp.where(a >= 0, a, a ^ 0x7FFFFFFF).view(cp.float32)


class _Tree:
    """sklearn-like tree_ arrays (host numpy) for CupyForest."""

    def __init__(self, left, right, feat, thr, val):
        self.children_left, self.children_right = left, right
        self.feature, self.threshold = feat, thr
        self.value = val[:, None, None]
        self.node_count = len(left)


class _Est:
    def __init__(self, tree):
        self.tree_ = tree


class CupyExtraTreesRegressor:
    """GPU ExtraTreesRegressor (single output). ``fit(X, y, sample_weight)``; ``forest()`` returns a
    CupyForest for prediction. Finished trees are held on the host; ``estimators_`` exposes them as
    sklearn-like ``tree_`` arrays."""

    def __init__(self, n_estimators=100, trees_per_group=None, random_state=None, verbose=False):
        self.n_estimators = int(n_estimators)
        self.trees_per_group = trees_per_group
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, uniforms=None):
        """uniforms: optional callable (level, tree_ids, node_ids, n_features) -> (K, F) array of U[0,1)
        draws, used instead of the RNG (for the exact reference check)."""
        import cupy as cp
        Xh = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y)) or (sample_weight is not None and not np.all(np.isfinite(sample_weight))):
            raise ValueError("cupy ExtraTrees: y and sample_weight must be finite (as sklearn requires)")
        if sample_weight is not None:
            # sklearn grows on positively weighted samples only: zero weights set no range and no count
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            keep = sample_weight > 0
            Xh, y, sample_weight = np.ascontiguousarray(Xh[keep]), y[keep], sample_weight[keep]
        N, F = Xh.shape
        if self.n_estimators * 1.6 * N >= 2 ** 31:     # trees have ~1.57 N nodes; CupyForest uses int32 ids
            raise ValueError("cupy ExtraTrees: %d trees of %d rows exceed int32 node indices" % (self.n_estimators, N))
        Xd = cp.asarray(Xh)
        w_h = np.ones(N) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        Q = _quanta(y, w_h)
        qw, qs, qq = (cp.asarray(a) for a in Q["q"])
        sw, ss, sq, yc0 = Q["sw"], Q["ss"], Q["sq"], Q["c"]
        mod = cp.RawModule(code=_KERNELS)
        k_stats, k_split, k_route = (mod.get_function(n) for n in ("node_stats", "split_stats", "route"))
        seed64 = np.uint64(self.random_state if self.random_state is not None
                           else int(np.random.SeedSequence().generate_state(2, dtype=np.uint64)[0]))
        hash_u = cp.ElementwiseKernel("uint64 seed, int64 tree, int64 node, int64 f", "float64 u", _HASH_UNIFORM,
                                      "et_hash_uniform")
        pool = cp.get_default_memory_pool()

        def group_size(remaining):
            if self.trees_per_group:
                return min(int(self.trees_per_group), remaining)
            # ~(64 F + 40) bytes per (tree, sample) at the widest level (measured 13.3 GB for 20 trees,
            # 1.2M rows, F = 9); use at most half the device memory free now. Finished trees live on
            # the host, so later groups see the same budget.
            pool.free_all_blocks()
            free = cp.cuda.Device().mem_info[0]
            t = int(0.5 * free // ((64 * F + 40) * N))
            if t < 1:
                raise MemoryError("cupy ExtraTrees: %.1f GB free is too little for one tree of %d rows" % (free / 1e9, N))
            return min(t, remaining, int(2 ** 31 // (2 * N)) - 1, 20)
        self.n_features_in_ = F
        self._trees = []          # per tree: dict of host arrays
        TH = 256
        g0 = 0
        while g0 < self.n_estimators:
            T = group_size(self.n_estimators - g0)
            while True:      # an out-of-memory group (e.g. a co-tenant grew) is retried at half size
                try:
                    cap = 2 * N                                   # nodes per tree are at most 2N - 1
                    left = cp.full(T * cap, -1, dtype=cp.int32); right = cp.full(T * cap, -1, dtype=cp.int32)
                    feat = cp.full(T * cap, -2, dtype=cp.int32); thr_out = cp.full(T * cap, -2.0, dtype=cp.float64)
                    val = cp.zeros(T * cap, dtype=cp.float64)
                    n_nodes = np.ones(T, dtype=np.int64)          # root of each tree exists
                    # frontier: K nodes, each (tree, node id); pairs carry their frontier slot
                    ftree = cp.arange(T, dtype=cp.int32); fid = cp.zeros(T, dtype=cp.int32)
                    slot = cp.repeat(cp.arange(T, dtype=cp.int32), N)
                    act = cp.arange(T * N, dtype=cp.int32)
                    level = 0
                    while len(act):
                        K = len(ftree)
                        n_act = np.int64(len(act))
                        blocks = (int((n_act + TH - 1) // TH),)
                        Wq = cp.zeros(K, dtype=cp.uint64); S1q = cp.zeros(K, dtype=cp.uint64); S2q = cp.zeros(K, dtype=cp.uint64)
                        cnt = cp.zeros(K, dtype=cp.int32); cntw = cp.zeros(K, dtype=cp.int32)
                        fmin = cp.full((K, F), np.iinfo(np.int32).max, dtype=cp.int32)
                        fmax = cp.full((K, F), np.iinfo(np.int32).min, dtype=cp.int32)
                        k_stats(blocks, (TH,), (act, n_act, np.int32(N), np.int32(F), Xd, qw, qs, qq, slot, Wq, S1q, S2q, cnt, cntw, fmin, fmax))
                        W, S1, S2 = _from_q(cp, Wq, sw), _from_q(cp, S1q, ss), _from_q(cp, S2q, sq)
                        lo = _decode_ord(cp, fmin).astype(cp.float64); hi = _decode_ord(cp, fmax).astype(cp.float64)
                        imp = S2 / W - (S1 / W) ** 2
                        cand = ~(hi.astype(cp.float32) <= (lo.astype(cp.float32) + np.float32(_FEATURE_THRESHOLD)))
                        splitting = (cnt >= 2) & (imp > _EPS) & cp.any(cand, axis=1)
                        if uniforms is None:
                            # counter-based draws: a node's uniforms depend on (seed, tree, node, feature)
                            # only, not on how trees are grouped (which follows free device memory)
                            u = hash_u(seed64, (ftree.astype(cp.int64) + g0)[:, None], fid.astype(cp.int64)[:, None],
                                       cp.arange(F + 1, dtype=cp.int64)[None, :])
                        else:
                            u = cp.asarray(uniforms(level, cp.asnumpy(ftree) + g0, cp.asnumpy(fid), F + 1), dtype=cp.float64)
                        thr = (hi - lo) * u[:, :F] + lo
                        thr = cp.where(thr == hi, lo, thr)
                        LWq = cp.zeros((K, F), dtype=cp.uint64); LSq = cp.zeros((K, F), dtype=cp.uint64)
                        LCW = cp.zeros((K, F), dtype=cp.int32)
                        k_split(blocks, (TH,), (act, n_act, np.int32(N), np.int32(F), Xd, qw, qs, slot,
                                                splitting.astype(cp.uint8), thr, cand.astype(cp.uint8), LWq, LSq, LCW))
                        RWq = Wq[:, None] - LWq; RSq = S1q[:, None] - LSq      # exact integer complements
                        LW, LS, RW, RS = _from_q(cp, LWq, sw), _from_q(cp, LSq, ss), _from_q(cp, RWq, sw), _from_q(cp, RSq, ss)
                        # each side needs positive weight (sklearn's proxy is NaN otherwise, so the split loses);
                        # tested on counts of positive-weight samples, since W - LW leaves roundoff
                        valid = cand & (LCW > 0) & (cntw[:, None] - LCW > 0)
                        proxy = cp.where(valid, LS * LS / cp.where(valid, LW, 1.0) + RS * RS / cp.where(valid, RW, 1.0), -cp.inf)
                        bf = _pick(cp, proxy, u[:, F])
                        splitting &= cp.isfinite(cp.max(proxy, axis=1))
                        sp8 = splitting.astype(cp.uint8)
                        bthr = thr[cp.arange(K), bf]
                        # node records
                        gid = ftree.astype(cp.int64) * cap + fid
                        val[gid] = S1 / W + yc0
                        # children: consecutive ids per tree, in frontier (= tree) order
                        spi = splitting.astype(cp.int64)
                        ns = cp.asnumpy(cp.bincount(ftree, weights=spi, minlength=T)).astype(np.int64)
                        csum = cp.cumsum(spi) - spi                                   # global rank among splits
                        tree_first = cp.asarray(np.r_[0, np.cumsum(ns)[:-1]])          # rank of the tree's first split
                        rank_in_tree = csum - tree_first[ftree]
                        child_id = cp.asarray(n_nodes)[ftree] + 2 * rank_in_tree
                        sgid = gid[splitting]
                        left[sgid] = child_id[splitting].astype(cp.int32)
                        right[sgid] = (child_id[splitting] + 1).astype(cp.int32)
                        feat[sgid] = bf[splitting]
                        thr_out[sgid] = bthr[splitting]
                        n_nodes += 2 * ns
                        # next frontier: two children per splitting node, slot = 2 * global rank (+1)
                        child0 = (2 * csum).astype(cp.int32)
                        ftree_s = ftree[splitting]
                        ftree = cp.repeat(ftree_s, 2)
                        fid = cp.stack([child_id[splitting], child_id[splitting] + 1], axis=1).reshape(-1).astype(cp.int32)
                        alive = cp.empty(len(act), dtype=cp.uint8)
                        k_route(blocks, (TH,), (act, n_act, np.int32(N), np.int32(F), Xd, slot, sp8, bf, bthr, child0, alive))
                        act = act[alive.astype(bool)]
                        level += 1
                        # release this level's per-node arrays and the pool's cache, which would otherwise
                        # grow past the (64 F + 40)-byte estimate (up to 1.7x on uniform data)
                        del W, S1, S2, Wq, S1q, S2q, LWq, LSq, RWq, RSq, cnt, cntw, fmin, fmax, lo, hi, imp, cand, u, thr, LW, LS, LCW, RW, RS, valid, proxy
                        del bf, bthr, gid, spi, csum, child_id, sgid, child0, alive, splitting, sp8
                        pool.free_all_blocks()
                    for t in range(T):      # finished trees go to the host
                        n = int(n_nodes[t]); o = t * cap
                        self._trees.append(dict(left=left[o:o + n].get(), right=right[o:o + n].get(), feat=feat[o:o + n].get(),
                                                thr=thr_out[o:o + n].get(), val=val[o:o + n].get()))
                    if self.verbose:
                        print(" cupy ExtraTrees: trees %d-%d grown, %d levels, nodes %s" % (g0, g0 + T - 1, level, n_nodes.tolist()))
                    del left, right, feat, thr_out, val, slot, act
                    break
                except cp.cuda.memory.OutOfMemoryError:
                    pool.free_all_blocks()
                    if T == 1:
                        raise
                    T = max(1, T // 2)
                    if self.verbose:
                        print(" cupy ExtraTrees: out of device memory; retrying with %d trees per group" % T)
            g0 += T
        pool.free_all_blocks()
        return self

    @property
    def estimators_(self):
        out = []
        for t in self._trees:
            out.append(_Est(_Tree(t["left"], t["right"], t["feat"], t["thr"], t["val"])))
        return out

    def forest(self, release=False):
        """CupyForest for prediction. The trees are kept on the host; release=True drops that copy
        (estimators_ then unavailable)."""
        from RIFT.interpolators.cupy_forest import CupyForest
        f = CupyForest.from_device_trees(self._trees, self.n_features_in_)
        if release:
            self._trees = None
        return f


def _reference_tree(X, y, w, uniforms, tree_id=0):
    """Same algorithm in numpy, breadth first, driven by uniforms(level, tree_ids, node_ids, F) like
    CupyExtraTreesRegressor.fit. Returns (left, right, feature, threshold, value) with the same node
    numbering (children of a level's splitting nodes numbered consecutively in frontier order)."""
    X = np.asarray(X, dtype=np.float32)
    keep = np.asarray(w) > 0                # as fit(): positively weighted samples only
    X, y, w = X[keep], np.asarray(y)[keep], np.asarray(w)[keep]
    N, F = X.shape
    Q = _quanta(y, w)                       # as fit(): exact integer sums of fixed-point quanta
    qw, qs, qq = Q["q"]; sw, ss, sq, c0 = Q["sw"], Q["ss"], Q["sq"], Q["c"]
    left, right, feat, thr_o, val = [-1], [-1], [-2], [-2.0], [0.0]
    frontier = [(0, np.arange(N))]
    level = 0
    while frontier:
        u = np.asarray(uniforms(level, np.full(len(frontier), tree_id), np.array([n for n, _ in frontier]), F + 1))
        nxt = []
        for k, (nid, idx) in enumerate(frontier):
            xx = X[idx]
            Wq, S1q, S2q = qw[idx].sum(), qs[idx].sum(), qq[idx].sum()
            W, S1, S2 = np.float64(Wq) / sw, np.float64(S1q) / ss, np.float64(S2q) / sq
            pos = qw[idx] > 0
            val[nid] = S1 / W + c0
            lo, hi = xx.min(axis=0).astype(np.float64), xx.max(axis=0).astype(np.float64)
            cand = ~(hi.astype(np.float32) <= lo.astype(np.float32) + np.float32(_FEATURE_THRESHOLD))
            if len(idx) < 2 or not (S2 / W - (S1 / W) ** 2 > _EPS) or not cand.any():
                continue
            t = (hi - lo) * u[k, :F] + lo
            t = np.where(t == hi, lo, t)
            goes_left = xx.astype(np.float64) <= t[None, :]
            LWq = (qw[idx][:, None] * goes_left).sum(0); LSq = (qs[idx][:, None] * goes_left).sum(0)
            LW, LS = LWq.astype(np.float64) / sw, LSq.astype(np.float64) / ss
            RW, RS = (Wq - LWq).astype(np.float64) / sw, (S1q - LSq).astype(np.float64) / ss
            lcw = (goes_left & pos[:, None]).sum(0)
            valid = cand & (lcw > 0) & (pos.sum() - lcw > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                proxy = np.where(valid, LS ** 2 / LW + RS ** 2 / RW, -np.inf)
            if not np.isfinite(proxy.max()):
                continue
            f = int(_pick(np, proxy[None, :], u[k, F:F + 1])[0])
            c = len(left)
            left[nid], right[nid], feat[nid], thr_o[nid] = c, c + 1, f, t[f]
            left += [-1, -1]; right += [-1, -1]; feat += [-2, -2]; thr_o += [-2.0, -2.0]; val += [0.0, 0.0]
            m = goes_left[:, f]
            nxt += [(c, idx[m]), (c + 1, idx[~m])]
        frontier = nxt
        level += 1
    return (np.array(left), np.array(right), np.array(feat), np.array(thr_o), np.array(val))
