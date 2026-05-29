"""
Python wrapper for the fused calibration-marginalized factored log-likelihood
kernel (Option C).  Mirrors Q_inner_product.py.

See cuda_Q_fused_calmarg.cu for the kernel and array-layout documentation.
"""
from __future__ import division

import os

import numpy as np
import cupy

_cuda_code = None
_kernel = None


def _get_kernel():
    global _cuda_code, _kernel
    if _kernel is None:
        path = os.path.join(os.path.dirname(__file__), 'cuda_Q_fused_calmarg.cu')
        if not os.path.isfile(path):
            path = os.path.join(os.path.split(os.path.dirname(__file__))[0],
                                'cuda_Q_fused_calmarg.cu')
        with open(path, 'r') as f:
            _cuda_code = f.read()
        _kernel = cupy.RawKernel(_cuda_code, "Q_fused_calmarg")
    return _kernel


def Q_fused_calmarg_cupy(Q, A, ifirst, invDist, rho_sq, w_t, n_cal, N_window,
                         threads_per_block=256):
    """Compute the calibration-marginalized factored log likelihood per extrinsic
    sample in a single kernel launch.

    Parameters
    ----------
    Q : (n_det, npts_full, n_lms) complex128
        Per-detector rholm timeseries (transposed), holding n_cal contiguous
        length-N_window calibration-realization blocks.
    A : (n_det, n_ext, n_lms) complex128
        Per-detector conj(F * Ylm).
    ifirst : (n_det, n_ext) int32
        Per-detector within-block window start index.
    invDist : (n_ext,) float64
        distMpcRef / distMpc.
    rho_sq : (n_ext, npts) float64
        Calibration-independent template (U,V) term, pre-summed over detectors.
    w_t : (npts,) float64
        Composite-Simpson quadrature weights (including dx=deltaT).
    n_cal : int
    N_window : int
        Per-realization block length inside Q (npts_full = N_window * n_cal).

    Returns
    -------
    out : (n_ext,) float64 cupy array
        log( (1/n_cal) sum_c sum_t w_t exp(lnL_t(j,c,t)) ).
    """
    Q = cupy.ascontiguousarray(Q, dtype=cupy.complex128)
    A = cupy.ascontiguousarray(A, dtype=cupy.complex128)
    ifirst = cupy.ascontiguousarray(ifirst, dtype=cupy.int32)
    invDist = cupy.ascontiguousarray(invDist, dtype=cupy.float64)
    rho_sq = cupy.ascontiguousarray(rho_sq, dtype=cupy.float64)
    w_t = cupy.ascontiguousarray(w_t, dtype=cupy.float64)

    n_det, npts_full, n_lms = Q.shape
    _, n_ext, _ = A.shape
    npts = w_t.shape[0]

    assert A.shape == (n_det, n_ext, n_lms)
    assert ifirst.shape == (n_det, n_ext)
    assert invDist.shape == (n_ext,)
    assert rho_sq.shape == (n_ext, npts)
    assert npts_full == N_window * n_cal, \
        "npts_full=%d != N_window*n_cal=%d*%d" % (npts_full, N_window, n_cal)

    out = cupy.empty(n_ext, dtype=cupy.float64)

    fn = _get_kernel()
    grid = ((n_ext + threads_per_block - 1) // threads_per_block,)
    block = (threads_per_block,)
    fn(grid, block, (
        Q, A, ifirst, invDist, rho_sq, w_t,
        np.int32(n_det), np.int32(n_cal), np.int32(N_window), np.int32(npts),
        np.int32(n_lms), np.int32(n_ext), np.int32(npts_full),
        out,
    ))
    return out
