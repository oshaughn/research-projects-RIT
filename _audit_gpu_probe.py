"""Adversarial probe of Q_inner_sinc vs the CPU reference."""
from __future__ import print_function, division
import os, sys, itertools
import numpy as np
import cupy

import RIFT.likelihood.factored_likelihood as FL
from RIFT.likelihood import Q_inner_product as QIP


def cpu_ref(Q, A, starts, fracs, npts, a):
    Qlms = FL._sinc_Q_window_numpy(Q, starts, fracs, npts, a=a)
    return np.einsum("ej,etj->et", A, Qlms)


def run(n_time, npts, n_lm, starts, fracs, a, tx, ty, si_dtype=np.int32,
        frac_dtype=np.float64, seed=3):
    rng = np.random.RandomState(seed)
    Q = rng.randn(n_time, n_lm) + 1j*rng.randn(n_time, n_lm)
    A = rng.randn(len(starts), n_lm) + 1j*rng.randn(len(starts), n_lm)
    os.environ["RIFT_Q_SINC_THREADS_X"] = str(tx)
    os.environ["RIFT_Q_SINC_THREADS_Y"] = str(ty)
    ref = cpu_ref(Q, A, starts.astype(np.int64), fracs, npts, a)
    got = cupy.asnumpy(QIP.Q_inner_product_sinc_cupy(
        cupy.asarray(Q), cupy.asarray(A),
        cupy.asarray(starts.astype(si_dtype)),
        cupy.asarray(fracs.astype(frac_dtype)), npts, halfwidth=a))
    scale = np.max(np.abs(Q))*np.max(np.abs(A))*n_lm
    return float(np.max(np.abs(ref-got)))/scale


def main():
    fails = []
    n_time = 1024
    base_starts = np.array(list(range(-20, 20)) + list(range(400, 440)) +
                           list(range(n_time-60, n_time+20)), dtype=np.int32)
    rng = np.random.RandomState(9)
    base_fracs = rng.rand(len(base_starts))

    print("=== A. block-shape sweep (a=8, n_lm=5, npts=200 > default 128) ===")
    for tx, ty in [(4,128),(1,1),(1,1024),(2,8),(8,4),(8,2),(16,16),(32,32),(4,3),(3,5),(64,16),(4,7)]:
        if tx*ty > 1024: 
            print("  skip %dx%d (>1024 threads)"%(tx,ty)); continue
        try:
            d = run(n_time, 200, 5, base_starts, base_fracs, 8, tx, ty)
            ok = d < 1e-13
            print("  THREADS_X=%-3d THREADS_Y=%-5d  maxrel=%.3e  %s" % (tx, ty, d, "OK" if ok else "*** FAIL ***"))
            if not ok: fails.append(("blockshape",tx,ty,d))
        except Exception as e:
            print("  THREADS_X=%-3d THREADS_Y=%-5d  EXCEPTION %s: %s" % (tx,ty,type(e).__name__,e))
            fails.append(("blockshape-exc",tx,ty,str(e)))

    print("=== B. halfwidth sweep (default block) ===")
    for a in (1,2,4,8,16,32,64):
        try:
            d = run(n_time, 60, 3, base_starts, base_fracs, a, 4, 128)
            ok = d < 1e-13
            print("  a=%-3d maxrel=%.3e %s" % (a, d, "OK" if ok else "*** FAIL ***"))
            if not ok: fails.append(("halfwidth",a,d))
        except Exception as e:
            print("  a=%-3d EXCEPTION %s: %s" % (a,type(e).__name__,e)); fails.append(("hw-exc",a,str(e)))

    print("=== C. shapes ===")
    for n_ex in (1,3,63,64,65):
        s = base_starts[:n_ex] if n_ex<=len(base_starts) else base_starts
        f = base_fracs[:len(s)]
        for npts in (1,17,128,129,301):
            d = run(n_time, npts, 2, s, f, 8, 4, 128)
            ok = d < 1e-13
            print("  n_ex=%-3d npts=%-4d maxrel=%.3e %s" % (len(s), npts, d, "OK" if ok else "*** FAIL ***"))
            if not ok: fails.append(("shape",len(s),npts,d))

    print("=== D. n_lm sweep ===")
    for n_lm in (1,2,3,5,9,16):
        d = run(n_time, 64, n_lm, base_starts, base_fracs, 8, 4, 128)
        ok = d < 1e-13
        print("  n_lm=%-3d maxrel=%.3e %s" % (n_lm, d, "OK" if ok else "*** FAIL ***"))
        if not ok: fails.append(("nlm",n_lm,d))

    print("=== E. dtype abuse (start_indices int64, fracs float32) ===")
    for si in (np.int32, np.int64, np.intc):
        try:
            d = run(n_time, 40, 3, base_starts, base_fracs, 8, 4, 128, si_dtype=si)
            print("  start_indices dtype=%-8s maxrel=%.3e %s" % (np.dtype(si).name, d, "OK" if d<1e-13 else "*** MISMATCH ***"))
        except Exception as e:
            print("  start_indices dtype=%-8s EXCEPTION %s: %s"%(np.dtype(si).name,type(e).__name__,e))
    for fd in (np.float64, np.float32):
        try:
            d = run(n_time, 40, 3, base_starts, base_fracs, 8, 4, 128, frac_dtype=fd)
            print("  fracs dtype=%-8s maxrel=%.3e %s" % (np.dtype(fd).name, d, "OK" if d<1e-13 else "*** MISMATCH ***"))
        except Exception as e:
            print("  fracs dtype=%-8s EXCEPTION %s: %s"%(np.dtype(fd).name,type(e).__name__,e))

    print("=== F. u exactly 0 and 1-eps ===")
    s = np.array([100,200,300,400], dtype=np.int32)
    for f0 in (0.0, 1e-17, 0.5, 1.0-1e-16, 1.0):
        f = np.full(len(s), f0)
        d = run(n_time, 32, 3, s, f, 8, 4, 128)
        print("  u=%.17g maxrel=%.3e %s" % (f0, d, "OK" if d<1e-13 else "*** FAIL ***"))

    print("=== G. huge start indices (overflow probe) ===")
    for big in (2**30, 2**31-1-500, -(2**31)+10):
        s = np.array([big], dtype=np.int32)
        f = np.array([0.3])
        try:
            d = run(n_time, 8, 2, s, f, 8, 4, 128)
            print("  start=%-14d maxrel=%.3e %s" % (big, d, "OK" if d<1e-13 else "*** FAIL ***"))
            if not d<1e-13: fails.append(("bigstart",big,d))
        except Exception as e:
            print("  start=%-14d EXCEPTION %s: %s"%(big,type(e).__name__,e))

    print()
    print("FAILURES:", fails if fails else "none")

main()
