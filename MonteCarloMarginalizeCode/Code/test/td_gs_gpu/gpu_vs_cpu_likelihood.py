#!/usr/bin/env python3
"""numpy-vs-cupy agreement of the td-gs code paths of
DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop (RIFT/likelihood/factored_likelihood.py).

Every case evaluates the same synthetic inputs once with xpy=numpy and once with xpy=cupy
and reports the largest relative difference (max |gpu - cpu| / max |cpu| over the entries
finite on both sides; the non-finite pattern must agree).  Target: 1e-12.

Cases
  1. NoLoop, td_subsample K = 1, 4, 8: the stride-K gathers of the data term and the
     time-indexed self term, nearest and cubic time_interp, with explicit phase and analytic-phase refusal, time-marginalized lnL and lnL(t); even (12) and odd (13) numbers
     of time bins (the Simpson rule on both backends); a window running past the buffer
     edge (zero extension); the kernels Q_inner_product_cupy / Q_inner_product_cubic_cupy
     against the numpy windows directly, with negative and over-running start indices;
     _time_indexed_self_term with 23 modes (552 pseudo-mode columns > the 512-column
     chunk of one kernel launch).
  2. The SVD-compressed self term: compress_time_indexed_cross_terms on the CPU, then
     TimeIndexedCrossTermsSVD.to_xpy(cupy), K = 1 and 4; the compressed vs the
     uncompressed likelihood on each backend; the compression done on the GPU (cuSOLVER)
     vs on the CPU.
  3. Calibration marginalization, cal_method='loop', with 4-D per-realization self terms
     ctUArrayDict_cal (n_cal, K*N_window, n, n) and per-realization rholm blocks, K = 1
     and 4: lnL, importance-weighted lnL, return_lnLt, return_cal_components, nearest and
     cubic, with explicit phase and analytic-phase refusal; the burn-in layout (n_cal=1 with the
     4-D arrays); cal_method='fused' refused on both backends. Global-norm burn-in
     and active subsets are checked against independently sliced stored Q blocks.
  4. simpson_function(cupy) against scipy for even and odd npts.

The synthetic inputs come from test/test_td_time_indexed_U.py::_make_case (random Q, U, V
with the shapes and conventions of the td-gs Precompute: rholms (n_lm, npts_full),
U, V (npts_full, n_lm, n_lm), epoch per detector); on the 1/K grid npts_full = K x N_window.
pytest is stubbed when it is not installed (the container may lack it).

Usage:  python3 gpu_vs_cpu_likelihood.py [--out results.json] [--numpy-only]
  --numpy-only runs the numpy side twice (script check on a machine without cupy).
Exit status 0 when every comparison is below the threshold, 1 otherwise.
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(CODE, "test"))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

try:
    import pytest  # noqa: F401
except ImportError:
    # test_td_time_indexed_U decorates with pytest.mark.parametrize at import time
    class _Mark(object):
        def parametrize(self, *a, **k):
            return lambda f: f

        def __getattr__(self, name):
            return lambda *a, **k: (lambda f: f)

    _stub = types.ModuleType("pytest")
    _stub.mark = _Mark()
    _stub.fixture = lambda *a, **k: (lambda f: f) if not (a and callable(a[0])) else a[0]
    _stub.raises = None
    _stub.skip = lambda *a, **k: None
    _stub.main = lambda *a, **k: 0
    sys.modules["pytest"] = _stub

import lal  # noqa: E402
import test_td_time_indexed_U as ttu  # noqa: E402
import RIFT.likelihood.factored_likelihood as fl  # noqa: E402

THRESH = 1e-12
RESULTS = []


# --------------------------------------------------------------------------- helpers

def _to_np(x):
    if hasattr(x, "get") and not isinstance(x, dict):
        return x.get()
    return np.asarray(x)


def rel_diff(a, b):
    """max |a - b| / max |b| over the entries finite on both sides (b: the numpy result)."""
    a = _to_np(a).astype(np.complex128 if np.iscomplexobj(_to_np(a)) else np.float64)
    b = _to_np(b).astype(np.complex128 if np.iscomplexobj(_to_np(b)) else np.float64)
    if a.shape != b.shape:
        return float("inf"), "shape %s vs %s" % (a.shape, b.shape)
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return float("inf"), "non-finite pattern differs (%d vs %d finite)" % (fa.sum(), fb.sum())
    if fb.sum() == 0:
        return 0.0, "no finite entries"
    scale = np.max(np.abs(b[fb]))
    if scale == 0:
        return float(np.max(np.abs(a[fa] - b[fb]))), "absolute (reference is zero)"
    return float(np.max(np.abs(a[fa] - b[fb])) / scale), "%d finite of %d" % (int(fb.sum()), b.size)


def record(name, a, b, thresh=THRESH, note=""):
    d, info = rel_diff(a, b)
    ok = d <= thresh
    RESULTS.append(dict(name=name, rel_diff=d, thresh=thresh, ok=bool(ok), info=info, note=note))
    print("  %-78s %9.2e  %s  (%s%s)" % (name, d, "ok  " if ok else "FAIL", info, ("; " + note) if note else ""))
    return d


def dict_to(xpy, d):
    out = {}
    for k, v in d.items():
        if isinstance(v, fl.TimeIndexedCrossTermsSVD):
            out[k] = v.to_xpy(xpy)
        elif v is None:
            out[k] = None
        else:
            out[k] = xpy.asarray(np.ascontiguousarray(_to_np(v)))
    return out


def pvec_to(xpy, P):
    Q = ttu._PVec()
    for name in ("phi", "theta", "psi", "incl", "phiref", "dist"):
        setattr(Q, name, xpy.asarray(_to_np(getattr(P, name))))
    Q.tref = P.tref
    Q.deltaT = P.deltaT
    return Q


def call_noloop(xpy, case, U, V, npts=None, **kw):
    tvals = case["tvals"] if npts is None else np.linspace(-npts // 2 * case["deltaT"], npts // 2 * case["deltaT"], npts)
    U_x = dict_to(xpy, U)
    # a compressed U stands for U and V together: ctVArrayDict[det] must be that same object (or None)
    V_x = {d: (U_x[d] if V[d] is U[d] else dict_to(xpy, {d: V[d]})[d]) for d in V}
    for name in ("ctUArrayDict_cal", "ctVArrayDict_cal"):
        if kw.get(name) is not None:
            kw[name] = dict_to(xpy, kw[name])
    if kw.get("cal_log_weights") is not None:
        kw["cal_log_weights"] = xpy.asarray(kw["cal_log_weights"])
    return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        xpy.asarray(tvals), pvec_to(xpy, case["P"]), case["lookupNKDict"], dict_to(xpy, case["rholmsArrayDict"]),
        U_x, V_x, case["epochDict"], Lmax=2, xpy=xpy, **kw)


def both(case, U, V, name, npts=None, **kw):
    """numpy and cupy NoLoop on the same inputs; records lnL and lnL(t)."""
    if kw.get("phase_marginalization") and any(
            isinstance(v, fl.TimeIndexedCrossTermsSVD) or np.ndim(v) == 3 for v in U.values()):
        variants = [(name, {})]
        if not kw.get("return_cal_components"):
            variants.append((name + " [lnL(t)]", {"return_lnLt": True}))
        for label, extra in variants:
            for backend in (np, XP):
                try:
                    call_noloop(backend, case, U, V, npts=npts, **dict(kw, **extra))
                except NotImplementedError as exc:
                    assert "explicit phase sampling" in str(exc), str(exc)
                else:
                    raise AssertionError("time-indexed analytic phase marginalization was accepted")
            record(label + " [phase refused]", 0., 0., note="explicit phase sampling required")
            RESULTS[-1]["kind"] = "refusal"
        return None, None
    ref = call_noloop(np, case, U, V, npts=npts, **kw)
    got = call_noloop(XP, case, U, V, npts=npts, **kw)
    record(name, got, ref)
    if not kw.get("return_cal_components"):
        ref_t = call_noloop(np, case, U, V, npts=npts, return_lnLt=True, **kw)
        got_t = call_noloop(XP, case, U, V, npts=npts, return_lnLt=True, **kw)
        record(name + " [lnL(t)]", got_t, ref_t)
    return ref, got


def make_case(seed, K, modes, npts_full_per_K=256, **kw):
    """_make_case on the 1/K grid: npts_full = K x N_window, and P.deltaT the coarse spacing."""
    return ttu._make_case(seed=seed, modes=modes, npts_full=K * npts_full_per_K, **kw)


MODES3 = ((2, 2), (2, -2), (2, 1))
MODES2 = ((2, 2), (2, -2))


# --------------------------------------------------------------------------- 1. NoLoop with stride K

def section_noloop():
    print("\n[1] NoLoop, td_subsample K: stride-K gathers, nearest/cubic, phase marginalization, 12 and 13 bins")
    for K in (1, 4, 8):
        for interp in ("nearest", "cubic"):
            for pm in (False, True):
                modes = MODES2 if pm else MODES3
                case = make_case(100 + K, K, modes)
                for npts in (12, 13):
                    both(case, case["U_t"], case["V_t"], "K=%d %-7s pm=%d npts=%d" % (K, interp, pm, npts),
                         npts=npts, time_interp=interp, phase_marginalization=pm, td_subsample=K)
        # scalar (2-D) U on the stride-K grid: the historical self term with the strided data term
        case = make_case(200 + K, K, MODES3)
        both(case, case["U_s"], case["V_s"], "K=%d scalar U, nearest" % K, time_interp="nearest", td_subsample=K)
        # window past the buffer edge (zero extension on both backends)
        case = make_case(300 + K, K, MODES2, npts_full_per_K=64, epoch_offset=-0.0005)
        for interp in ("nearest", "cubic"):
            both(case, case["U_t"], case["V_t"], "K=%d %s window past the buffer edge" % (K, interp),
                 time_interp=interp, td_subsample=K)


def section_kernels():
    print("\n[1b] Q_inner_product kernels vs the numpy windows (stride 1, 4, 8; negative and over-running starts)")
    if XP is not np:
        import RIFT.likelihood.Q_inner_product as QIP
    rng = np.random.default_rng(5)
    n_time, n_lm, ne = 600, 7, 9
    Q = rng.standard_normal((n_time, n_lm)) + 1j * rng.standard_normal((n_time, n_lm))
    A = rng.standard_normal((ne, n_lm)) + 1j * rng.standard_normal((ne, n_lm))
    for stride in (1, 4, 8):
        for npts in (12, 13):
            starts = np.array([-3, -1, 0, 1, 7, 100, n_time - 5 * stride, n_time - 2, n_time + 4], dtype=np.int32)
            frac = rng.uniform(0, 1, ne)
            frac[2] = 0.0
            ref_n = np.einsum("ek,etk->et", A, fl._nearest_Q_window_numpy(Q, starts, npts, time_stride=stride))
            ref_c = np.einsum("ek,etk->et", A, fl._cubic_Q_window_numpy(Q, starts, frac, npts, time_stride=stride))
            if XP is np:
                got_n, got_c = ref_n, ref_c
            else:
                got_n = QIP.Q_inner_product_cupy(XP.asarray(Q), XP.asarray(A), XP.asarray(starts), npts, time_stride=stride)
                got_c = QIP.Q_inner_product_cubic_cupy(XP.asarray(Q), XP.asarray(A), XP.asarray(starts), XP.asarray(frac), npts, time_stride=stride)
            record("Q_inner_product_cupy stride=%d npts=%d" % (stride, npts), got_n, ref_n)
            record("Q_inner_product_cubic_cupy stride=%d npts=%d" % (stride, npts), got_c, ref_c)
    print("\n[1c] _time_indexed_self_term with 23 modes (552 pseudo-mode columns: two kernel launches)")
    n_lm, ne, npts = 23, 6, 12
    for stride in (1, 4, 8):
        n_full = 300 * stride
        U = rng.standard_normal((n_full, n_lm, n_lm)) + 1j * rng.standard_normal((n_full, n_lm, n_lm))
        V = rng.standard_normal((n_full, n_lm, n_lm)) + 1j * rng.standard_normal((n_full, n_lm, n_lm))
        F = rng.standard_normal(ne) + 1j * rng.standard_normal(ne)
        Y = rng.standard_normal((ne, n_lm)) + 1j * rng.standard_normal((ne, n_lm))
        ifirst = np.array([0, 3, 50, 120, 200, 290], dtype=np.int32)
        frac = rng.uniform(0, 1, ne)
        for interp in ("nearest", "cubic"):
            ref = fl._time_indexed_self_term(U, V, F, Y, ifirst, frac, npts, interp, xpy=np, stride=stride)
            got = fl._time_indexed_self_term(XP.asarray(U), XP.asarray(V), XP.asarray(F), XP.asarray(Y), XP.asarray(ifirst),
                                             XP.asarray(frac), npts, interp, xpy=XP, stride=stride)
            record("_time_indexed_self_term n_lm=23 stride=%d %s" % (stride, interp), got, ref)


# --------------------------------------------------------------------------- 2. SVD-compressed self term

def section_svd():
    print("\n[2] SVD-compressed self term (compress on CPU, to_xpy(cupy)), K = 1 and 4")
    modes = ((2, 2), (2, -2), (2, 1), (2, -1), (2, 0))
    for K in (1, 4):
        case = make_case(400 + K, K, modes)
        comp = {d: fl.compress_time_indexed_cross_terms(case["U_t"][d], case["V_t"][d], 1e-13) for d in case["dets"]}
        rs = sorted(set(c.r for c in comp.values()))
        print("  kept components r = %s of %d columns" % (rs, comp["H1"].n_cols))
        for interp in ("nearest", "cubic"):
            for npts in (12, 13):
                ref, got = both(case, comp, comp, "K=%d compressed %-7s npts=%d" % (K, interp, npts),
                                npts=npts, time_interp=interp, td_subsample=K)
                # compressed vs uncompressed on each backend (all components kept: rounding level)
                ref_u = call_noloop(np, case, case["U_t"], case["V_t"], npts=npts, time_interp=interp, td_subsample=K)
                got_u = call_noloop(XP, case, case["U_t"], case["V_t"], npts=npts, time_interp=interp, td_subsample=K)
                record("K=%d %s npts=%d compressed vs uncompressed [numpy]" % (K, interp, npts), ref, ref_u, thresh=1e-10)
                record("K=%d %s npts=%d compressed vs uncompressed [cupy]" % (K, interp, npts), got, got_u, thresh=1e-10)
        # ctVArrayDict[det] = None is allowed with a compressed U
        V_none = {d: None for d in case["dets"]}
        both(case, comp, V_none, "K=%d compressed, V=None, nearest" % K, time_interp="nearest", td_subsample=K)
        # compression on the GPU (cuSOLVER) vs on the CPU: a different SVD, the same reconstruction
        if XP is not np:
            comp_g = {d: fl.compress_time_indexed_cross_terms(XP.asarray(case["U_t"][d]), XP.asarray(case["V_t"][d]), 1e-13, xpy=XP)
                      for d in case["dets"]}
            print("  GPU compression: r = %s" % sorted(set(c.r for c in comp_g.values())))
            for interp in ("nearest", "cubic"):
                ref = call_noloop(np, case, comp, comp, time_interp=interp, td_subsample=K)
                got = call_noloop(XP, case, comp_g, comp_g, time_interp=interp, td_subsample=K)
                record("K=%d %s compressed on GPU (cuSOLVER) vs on CPU (LAPACK)" % (K, interp), got, ref, thresh=1e-10,
                       note="different SVDs; agreement bounded by the dropped components")


# --------------------------------------------------------------------------- 3. calibration loop

def cal_case(seed, K, modes, n_cal=3, N_window=256):
    """rholms (n_lm, n_cal*K*N_window) as n_cal contiguous realization blocks; base U(t) one block
    long (K*N_window, n, n); per-realization U_c(t) (n_cal, K*N_window, n, n)."""
    case = ttu._make_case(seed=seed, modes=modes, npts_full=n_cal * K * N_window)
    B = K * N_window
    U_base = {d: case["U_t"][d][:B].copy() for d in case["dets"]}
    V_base = {d: case["V_t"][d][:B].copy() for d in case["dets"]}
    U_cal = {d: case["U_t"][d].reshape(n_cal, B, len(modes), len(modes)).copy() for d in case["dets"]}
    V_cal = {d: case["V_t"][d].reshape(n_cal, B, len(modes), len(modes)).copy() for d in case["dets"]}
    return case, U_base, V_base, U_cal, V_cal


def section_cal():
    print("\n[3] calibration marginalization, cal_method='loop', 4-D per-realization self terms, K = 1 and 4")
    n_cal = 3
    log_w = np.log(np.array([0.6, 1.5, 0.9]))
    for K in (1, 4):
        for pm in (False, True):
            modes = MODES2 if pm else MODES3
            case, U_b, V_b, U_c, V_c = cal_case(500 + K + 10 * pm, K, modes, n_cal=n_cal)
            for interp in ("nearest", "cubic"):
                kw = dict(time_interp=interp, phase_marginalization=pm, td_subsample=K, n_cal=n_cal, cal_method="loop")
                for npts in (12, 13):
                    name = "cal K=%d %-7s pm=%d npts=%d" % (K, interp, pm, npts)
                    both(case, U_b, V_b, name, npts=npts, ctUArrayDict_cal=U_c, ctVArrayDict_cal=V_c, **kw)
                both(case, U_b, V_b, "cal K=%d %s pm=%d weighted" % (K, interp, pm),
                     ctUArrayDict_cal=U_c, ctVArrayDict_cal=V_c, cal_log_weights=log_w, **kw)
                both(case, U_b, V_b, "cal K=%d %s pm=%d return_cal_components" % (K, interp, pm),
                     ctUArrayDict_cal=U_c, ctVArrayDict_cal=V_c, return_cal_components=True, **kw)
            # the burn-in layout: n_cal=1 on the full rholm buffers with the 4-D cal arrays (block 0)
            both(case, U_b, V_b, "cal K=%d pm=%d burn-in layout (n_cal=1, 4-D cal arrays)" % (K, pm),
                 time_interp="nearest", phase_marginalization=pm, td_subsample=K, n_cal=1,
                 ctUArrayDict_cal=U_c, ctVArrayDict_cal=V_c)
            # full-length base U(t) instead of one block
            both(case, case["U_t"], case["V_t"], "cal K=%d pm=%d full-length base U(t)" % (K, pm),
                 time_interp="nearest", phase_marginalization=pm, td_subsample=K, n_cal=n_cal,
                 ctUArrayDict_cal=U_c, ctVArrayDict_cal=V_c)
            # cal_method='fused' is refused on both backends
            for xpy in (np, XP):
                try:
                    call_noloop(xpy, case, U_b, V_b, td_subsample=K, n_cal=n_cal, cal_method="fused",
                                ctUArrayDict_cal=U_c, ctVArrayDict_cal=V_c)
                    refused = False
                except NotImplementedError:
                    refused = True
                RESULTS.append(dict(name="cal K=%d pm=%d cal_method='fused' refused [%s]" % (K, pm, xpy.__name__),
                                    rel_diff=0.0 if refused else float("inf"), thresh=THRESH, ok=refused, info="NotImplementedError", note=""))
                print("  %-78s %s" % ("cal K=%d pm=%d cal_method='fused' refused [%s]" % (K, pm, xpy.__name__), "ok" if refused else "FAIL"))


def section_global_norm_blocks():
    """Global-norm burn-in/subsets vs independently evaluated stored Q blocks."""
    from scipy.special import logsumexp
    print("\n[3b] global norm, stored Q blocks, burn-in and active subsets")
    case = ttu._make_case(seed=701, dets=("H1",), modes=MODES2,
                          npts_full=512, npts=80, epoch_offset=-0.001)
    U = {d: case["U_t"][d][:128] for d in case["dets"]}
    V = {d: case["V_t"][d][:128] for d in case["dets"]}
    for interp in ("nearest", "cubic"):
        for active in (1, 2, 4):
            for return_lnLt in (False, True):
                kw = dict(time_interp=interp, return_lnLt=return_lnLt)
                separate = []
                for c in range(active):
                    one = dict(case, rholmsArrayDict={
                        d: case["rholmsArrayDict"][d][:, c*128:(c+1)*128] for d in case["dets"]})
                    separate.append(call_noloop(np, one, U, V, **kw))
                ref = logsumexp(np.asarray(separate), axis=0) - np.log(active)
                for backend in (np, XP):
                    got = call_noloop(backend, case, U, V, n_cal=active, **kw)
                    record("global norm %s active=%d lnLt=%s [%s]" %
                           (interp, active, return_lnLt, backend.__name__), got, ref)


# --------------------------------------------------------------------------- 4. Simpson rule

def section_simpson():
    print("\n[4] simpson_function(cupy) vs scipy, even and odd npts")
    rng = np.random.default_rng(9)
    for npts in (2, 3, 12, 13, 33, 40):
        y = rng.uniform(0.5, 2.0, (5, npts))
        ref = fl.my_simps(y, dx=1.0 / 4096, axis=-1)
        got = fl.simpson_function(XP)(XP.asarray(y), dx=1.0 / 4096, axis=-1)
        record("simpson npts=%d" % npts, got, ref)


# --------------------------------------------------------------------------- main

def main():
    global XP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="JSON file for the per-case results")
    ap.add_argument("--numpy-only", action="store_true")
    a = ap.parse_args()
    info = dict(host=os.uname().nodename, numpy=np.__version__)
    if a.numpy_only:
        XP = np
        info["backend"] = "numpy (script check; both sides numpy)"
    else:
        import cupy
        XP = cupy
        dev = cupy.cuda.Device()
        props = cupy.cuda.runtime.getDeviceProperties(dev.id)
        info["backend"] = "cupy %s" % cupy.__version__
        info["gpu"] = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        info["compute_capability"] = "%d.%d" % (props["major"], props["minor"])
        info["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
        x = cupy.arange(8.0)
        assert float((x * x).sum()) == 140.0, "cupy cannot run a kernel here"
    print("backend: %s  host: %s  %s" % (info["backend"], info["host"], info.get("gpu", "")))
    t0 = time.time()
    section_noloop()
    section_kernels()
    section_svd()
    section_cal()
    section_global_norm_blocks()
    section_simpson()
    n_fail = sum(1 for r in RESULTS if not r["ok"])
    worst = max((r["rel_diff"] for r in RESULTS if np.isfinite(r["rel_diff"])), default=0.0)
    print("\n%d comparisons, %d above threshold; largest finite relative difference %.2e; %.1f s"
          % (len(RESULTS), n_fail, worst, time.time() - t0))
    if n_fail:
        print("FAILED:")
        for r in RESULTS:
            if not r["ok"]:
                print("   %s: %.2e (%s)" % (r["name"], r["rel_diff"], r["info"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(info=info, n_fail=n_fail, worst=worst, results=RESULTS), fh, indent=1)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
