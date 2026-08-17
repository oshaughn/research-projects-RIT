#!/usr/bin/env python3
"""test_q_window_interp_gpu -- GPU/CPU parity for the Q(t) sub-sample stencils.

test_q_window_interp.py pins down how ACCURATE each stencil is against a known band-limited
signal.  This file pins down something different and equally necessary: that the CUDA kernels
compute the SAME stencil the numpy reference does.  Accuracy evidence gathered on the CPU only
transfers to production -- which runs --gpu -- if the two agree.

Three levels, cheapest first, so a failure localises itself:

  1. weights.  _sinc_lanczos_weight_matrix is evaluated with the numpy and the cupy backend.
     Both are the same source expression, so this only measures the difference between the two
     sin() implementations.  If THIS is the thing that is large, nothing downstream is a kernel
     bug.
  2. kernel.  Q_inner_product_{,cubic_,sinc_}cupy against the numpy window builder contracted
     with the same A, on random data -- including windows deliberately placed so the stencil
     hangs off both ends of the Q buffer, which is the one place the per-tap zero-extension
     guard can differ between the two implementations.
  3. likelihood.  Covered by test_slowrot_gpu.py / test_slowrot_freqresponse_gpu.py, which loop
     over all three stencils.

SKIPPED if cupy / a GPU is unavailable.  Run on a GPU node:
    python RIFT/likelihood/test_q_window_interp_gpu.py

MEASURED (2026-08): identical to the last digit on an RTX 2080 Ti (sm_75) and on an RTX PRO 4000
Blackwell (sm_120), so the kernel is not architecture-sensitive.

If cupy raises "nvrtc: error: invalid value for --gpu-architecture (-arch)" you are on a card
newer than your cupy knows.  cupy 10.6 computes min(arch, nvrtc_max_cc) on STRINGS, so
min("120","86") == "120" and it hands nvrtc an sm_120 it cannot target.  BOTH of these are needed
to work around it (either alone still fails) -- pin the PTX target and let the driver JIT forward:

    export CUPY_COMPILE_WITH_PTX=1
    # plus a sitecustomize.py early on PYTHONPATH:
    #   import cupy.cuda.compiler as _c; _c._get_arch = lambda: "86"

That is a test-time workaround for an old cupy, NOT something to carry into production; the real
fix is a container whose CUDA can target the card directly.
"""
from __future__ import print_function, division

import numpy as np

import RIFT.likelihood.factored_likelihood as FL

from RIFT.likelihood._gpu_test_support import skip_without_gpu

try:
    import cupy
    _ = cupy.array(1.0) + 1.0    # force a real device op
    from RIFT.likelihood import Q_inner_product as QIP
    HAVE_GPU = True
except Exception as e:                                            # pragma: no cover
    HAVE_GPU = False
    _WHY = str(e)

# Agreement demanded of the kernels.  The CPU builder sums taps then contracts over lm; the
# kernels fuse the two, so the summation order differs and bitwise equality is not available.
# What IS available is agreement at the level double-precision reassociation allows.
TOL_REL = 1e-13


def _cpu_reference(Q, A, starts, fracs, npts, time_interp):
    """(n_ex, npts) product, built the CPU way: window first, then contract over lm."""
    Qlms = FL._q_window_numpy_interp(Q, starts, fracs, npts, time_interp)
    return np.einsum("ej,etj->et", A, Qlms)


def _gpu(Q, A, starts, fracs, npts, time_interp):
    return cupy.asnumpy(FL._q_inner_product_gpu(
        cupy.asarray(Q), cupy.asarray(A), cupy.asarray(starts.astype(np.int32)),
        cupy.asarray(fracs), npts, time_interp))


def test_weight_backends_agree():
    """Level 1: the shared weight formula, numpy backend vs cupy backend."""
    if not HAVE_GPU:
        if skip_without_gpu(HAVE_GPU, _WHY): return
    u = np.concatenate([np.linspace(0.0, 1.0, 257), [0.0, 0.5, 1.0 - 1e-12]])
    _, w_np = FL._sinc_lanczos_weight_matrix(u)
    _, w_cp = FL._sinc_lanczos_weight_matrix(cupy.asarray(u), xpy=cupy)
    d = float(np.max(np.abs(w_np - cupy.asnumpy(w_cp))))
    print("(GPU) sinc weights, numpy vs cupy backend      : max|diff| = %.3e" % d)
    assert d < 1e-14, "the two backends' sinc() disagree by more than round-off: %g" % d
    # Partition of unity must survive on the device too, or a constant is not reproduced.
    s = float(np.max(np.abs(cupy.asnumpy(w_cp).sum(axis=1) - 1.0)))
    print("(GPU) sinc weights, device partition of unity  : max|sum-1| = %.3e" % s)
    assert s < 1e-12, "device weights do not sum to one: %g" % s


def _kernel_case(label, n_time, npts, n_lm, starts, seed=3):
    rng = np.random.RandomState(seed)
    Q = (rng.randn(n_time, n_lm) + 1j * rng.randn(n_time, n_lm))
    A = (rng.randn(len(starts), n_lm) + 1j * rng.randn(len(starts), n_lm))
    fracs = rng.rand(len(starts))
    scale = np.max(np.abs(Q)) * np.max(np.abs(A)) * n_lm
    for interp in FL.TIME_INTERP_CHOICES:
        s = np.round(starts + fracs).astype(np.int32) if interp == 'nearest' else starts.astype(np.int32)
        f = np.zeros(len(starts)) if interp == 'nearest' else fracs
        cpu = _cpu_reference(Q, A, s, f, npts, interp)
        gpu = _gpu(Q, A, s, f, npts, interp)
        d = float(np.max(np.abs(cpu - gpu))) / scale
        print("(GPU) %-18s interp=%-8s : max|diff|/scale = %.3e" % (label, interp, d))
        assert d < TOL_REL, "%s kernel disagrees with CPU (%s): %g" % (label, interp, d)


def test_kernels_match_cpu_interior():
    """Level 2a: windows well inside the buffer, where no tap is ever dropped."""
    if not HAVE_GPU:
        if skip_without_gpu(HAVE_GPU, _WHY): return
    n_time, npts, n_lm = 2048, 32, 5
    starts = np.random.RandomState(11).randint(64, n_time - 64 - npts, size=64)
    _kernel_case("interior", n_time, npts, n_lm, starts)


def test_kernels_match_cpu_at_edges():
    """Level 2b: windows hanging off BOTH ends.

    This is the case that separates a correct kernel from a plausible one.  The sinc stencil is
    2a=16 taps wide, so it reaches much further past the buffer than the cubic's 4, and the
    weights are normalised over the FULL stencil before any tap is dropped -- dropped taps are
    NOT renormalised away.  A kernel that renormalised the surviving taps, or that let a
    negative index wrap, would still look perfect in the interior test above.
    """
    if not HAVE_GPU:
        if skip_without_gpu(HAVE_GPU, _WHY): return
    n_time, npts, n_lm = 512, 24, 3
    a = FL.SINC_HALFWIDTH_DEFAULT
    # deliberately straddle 0 and n_time by more than the widest stencil
    starts = np.array(
        list(range(-a - 2, a + 3)) +
        list(range(n_time - npts - a - 2, n_time - npts + a + 3)),
        dtype=np.int32)
    _kernel_case("edge/zero-extend", n_time, npts, n_lm, starts, seed=5)


if __name__ == "__main__":
    test_weight_backends_agree()
    test_kernels_match_cpu_interior()
    test_kernels_match_cpu_at_edges()
    print("Q WINDOW GPU PARITY DONE" if HAVE_GPU else "Q WINDOW GPU PARITY SKIPPED (no GPU)")
