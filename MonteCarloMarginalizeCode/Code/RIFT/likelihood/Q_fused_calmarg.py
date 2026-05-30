"""
Fused calibration-marginalized factored log-likelihood (Option C).

Two backends with identical results (validated against each other and the loop path):
  - GPU: the CUDA kernels in cuda_Q_fused_calmarg*.cu (memory-efficient, one launch).
  - CPU: pure-numpy implementations below (no CUDA needed -- runs on a laptop, and
         gives an independent cross-check of the kernel math).

cupy is imported lazily, so this module imports fine on a machine without CUDA.
See cuda_Q_fused_calmarg.cu for the array-layout documentation.
"""
from __future__ import division

import os

import numpy as np

_kernel = None
_kernel_distmarg = None


def _load_kernel(filename, entry):
    import cupy
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.isfile(path):
        path = os.path.join(os.path.split(os.path.dirname(__file__))[0], filename)
    with open(path, 'r') as f:
        code = f.read()
    return cupy.RawKernel(code, entry)


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = _load_kernel('cuda_Q_fused_calmarg.cu', "Q_fused_calmarg")
    return _kernel


def _get_kernel_distmarg():
    global _kernel_distmarg
    if _kernel_distmarg is None:
        _kernel_distmarg = _load_kernel('cuda_Q_fused_calmarg_distmarg.cu',
                                        "Q_fused_calmarg_distmarg")
    return _kernel_distmarg


def _prep_log_w(cal_log_weights, n_cal):
    """Return (log_w cupy array length n_cal, log_w_norm float = logsumexp(log_w)).
    cal_log_weights=None -> uniform (log_w=0, log_w_norm=log(n_cal)), i.e. the plain
    (1/n_cal) average."""
    import cupy
    if cal_log_weights is None:
        log_w = cupy.zeros(n_cal, dtype=cupy.float64)
        log_w_norm = float(np.log(n_cal))
    else:
        # cupy.asarray (not ascontiguousarray) so a host numpy array is accepted/transferred
        log_w = cupy.ascontiguousarray(cupy.asarray(cal_log_weights, dtype=cupy.float64))
        assert log_w.shape == (n_cal,), \
            "cal_log_weights shape %s != (%d,)" % (log_w.shape, n_cal)
        mx = float(cupy.max(log_w))
        log_w_norm = mx + float(cupy.log(cupy.sum(cupy.exp(log_w - mx))))
    return log_w, log_w_norm


def Q_fused_calmarg_cupy(Q, A, ifirst, invDist, rho_sq, w_t, n_cal, N_window,
                         cal_log_weights=None, threads_per_block=256):
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
    import cupy
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

    log_w, log_w_norm = _prep_log_w(cal_log_weights, n_cal)
    out = cupy.empty(n_ext, dtype=cupy.float64)

    fn = _get_kernel()
    grid = ((n_ext + threads_per_block - 1) // threads_per_block,)
    block = (threads_per_block,)
    fn(grid, block, (
        Q, A, ifirst, invDist, rho_sq, w_t, log_w, np.float64(log_w_norm),
        np.int32(n_det), np.int32(n_cal), np.int32(N_window), np.int32(npts),
        np.int32(n_lms), np.int32(n_ext), np.int32(npts_full),
        out,
    ))
    return out


def Q_fused_calmarg_distmarg_cupy(Q, A, ifirst, invDist, rho_sq, w_t,
                                  n_cal, N_window, distmarg,
                                  cal_log_weights=None, threads_per_block=256):
    """Fused calibration + distance marginalization (Option C stage 2).

    Same as Q_fused_calmarg_cupy, but applies the distance-marginalization
    loglikelihood on-board instead of the default helper.

    distmarg : dict with the distance-marginalization table and transform params:
        lnI_array (ns, nt) float64, s0, ds, smin, smax, t0, dt, tmax,
        xmin, xmax, sqrt_bmax, bref.
        (s0/ds = s_array[0]/(s_array[1]-s_array[0]); smin/smax = s_array[0]/[-1];
         analogously for t; xmin=distMpcRef/dmax, xmax=distMpcRef/dmin.)
    """
    import cupy
    Q = cupy.ascontiguousarray(Q, dtype=cupy.complex128)
    A = cupy.ascontiguousarray(A, dtype=cupy.complex128)
    ifirst = cupy.ascontiguousarray(ifirst, dtype=cupy.int32)
    invDist = cupy.ascontiguousarray(invDist, dtype=cupy.float64)
    rho_sq = cupy.ascontiguousarray(rho_sq, dtype=cupy.float64)
    w_t = cupy.ascontiguousarray(w_t, dtype=cupy.float64)
    lnI = cupy.ascontiguousarray(distmarg["lnI_array"], dtype=cupy.float64)

    n_det, npts_full, n_lms = Q.shape
    _, n_ext, _ = A.shape
    npts = w_t.shape[0]
    ns, nt = lnI.shape

    assert A.shape == (n_det, n_ext, n_lms)
    assert ifirst.shape == (n_det, n_ext)
    assert invDist.shape == (n_ext,)
    assert rho_sq.shape == (n_ext, npts)
    assert npts_full == N_window * n_cal, \
        "npts_full=%d != N_window*n_cal=%d*%d" % (npts_full, N_window, n_cal)

    log_w, log_w_norm = _prep_log_w(cal_log_weights, n_cal)
    out = cupy.empty(n_ext, dtype=cupy.float64)

    fn = _get_kernel_distmarg()
    grid = ((n_ext + threads_per_block - 1) // threads_per_block,)
    block = (threads_per_block,)
    fn(grid, block, (
        Q, A, ifirst, invDist, rho_sq, w_t, log_w, np.float64(log_w_norm), lnI,
        np.float64(distmarg["s0"]), np.float64(distmarg["ds"]),
        np.float64(distmarg["smin"]), np.float64(distmarg["smax"]), np.int32(ns),
        np.float64(distmarg["t0"]), np.float64(distmarg["dt"]),
        np.float64(distmarg["tmax"]), np.int32(nt),
        np.float64(distmarg["xmin"]), np.float64(distmarg["xmax"]),
        np.float64(distmarg["sqrt_bmax"]), np.float64(distmarg["bref"]),
        np.int32(n_det), np.int32(n_cal), np.int32(N_window), np.int32(npts),
        np.int32(n_lms), np.int32(n_ext), np.int32(npts_full),
        out,
    ))
    return out


# ---------------------------------------------------------------------------
# CPU / numpy backend (no CUDA).  Mirrors the kernels exactly; independent
# implementation, so agreement with the GPU path cross-validates both.
# ---------------------------------------------------------------------------
def _distmarg_lnL_numpy(kappa_sq, rho_sq, d):
    """Numpy version of the on-board distmarg transform (mirrors
    cuda_Q_fused_calmarg_distmarg.cu and the ILE EvenBivariateLinearInterpolator).
    Out-of-table points return -inf (contribute nothing)."""
    lnI_arr = np.asarray(d["lnI_array"], dtype=np.float64)
    ns, nt = lnI_arr.shape
    xmin, xmax = d["xmin"], d["xmax"]
    sqrt_bmax, bref = d["sqrt_bmax"], d["bref"]
    smin, smax, tmax = d["smin"], d["smax"], d["tmax"]
    s0, ds, t0, dt = d["s0"], d["ds"], d["t0"], d["dt"]

    x0 = kappa_sq / rho_sq
    s = np.arcsinh(sqrt_bmax * (x0 - xmin)) - np.arcsinh(sqrt_bmax * (xmax - x0))
    t = np.arcsinh(rho_sq / bref)

    out = np.full(x0.shape, -np.inf, dtype=np.float64)
    i_mid = (s - s0) / ds
    j_mid = (t - t0) / dt
    i_lo = np.floor(i_mid).astype(int); i_hi = np.ceil(i_mid).astype(int)
    j_lo = np.floor(j_mid).astype(int); j_hi = np.ceil(j_mid).astype(int)
    ok = ((s > smin) & (s < smax) & (t < tmax) &
          (i_lo >= 0) & (i_hi < ns) & (j_lo >= 0) & (j_hi < nt))
    if np.any(ok):
        il, ih = i_lo[ok], i_hi[ok]
        jl, jh = j_lo[ok], j_hi[ok]
        p = i_mid[ok] - il; q = j_mid[ok] - jl
        p_, q_ = 1.0 - p, 1.0 - q
        lnI = (p_ * q_ * lnI_arr[il, jl] + p * q_ * lnI_arr[ih, jl]
               + p_ * q * lnI_arr[il, jh] + p * q * lnI_arr[ih, jh])
        x0c = np.clip(x0[ok], xmin, xmax)
        out[ok] = rho_sq[ok] * x0c * (x0[ok] - 0.5 * x0c) + lnI
    return out


def Q_fused_calmarg_numpy(Q, A, ifirst, invDist, rho_sq, w_t, n_cal, N_window,
                          distmarg=None, cal_log_weights=None):
    """Pure-numpy equivalent of Q_fused_calmarg_cupy / _distmarg_cupy.

    Same arguments and result; distmarg=None uses the default helper, otherwise the
    distmarg table dict.  Materializes (n_cal, n_ext, npts) -- fine for CPU / testing.
    """
    Q = np.asarray(Q, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)
    ifirst = np.asarray(ifirst, dtype=np.int64)
    invDist = np.asarray(invDist, dtype=np.float64)
    rho_sq = np.asarray(rho_sq, dtype=np.float64)
    w_t = np.asarray(w_t, dtype=np.float64)

    n_det, npts_full, n_lms = Q.shape
    _, n_ext, _ = A.shape
    npts = w_t.shape[0]
    assert npts_full == N_window * n_cal

    if cal_log_weights is None:
        log_w = np.zeros(n_cal)
        log_w_norm = float(np.log(n_cal))
    else:
        log_w = np.asarray(cal_log_weights, dtype=np.float64)
        _mx = float(np.max(log_w))
        log_w_norm = _mx + float(np.log(np.sum(np.exp(log_w - _mx))))

    tgrid = np.arange(npts)
    lnLt_all = np.empty((n_cal, n_ext, npts), dtype=np.float64)
    for c in range(n_cal):
        kappa = np.zeros((n_ext, npts), dtype=np.complex128)
        for dd in range(n_det):
            within = ifirst[dd][:, None] + tgrid[None, :]        # (n_ext, npts)
            valid = (within >= 0) & (within < N_window)
            idx = np.clip(within + c * N_window, 0, npts_full - 1)
            gathered = Q[dd][idx]                                # (n_ext, npts, n_lms)
            gathered[~valid] = 0.0                               # out-of-block -> 0
            kappa += np.einsum("jl,jtl->jt", A[dd], gathered)
        kappa_sq = (kappa * invDist[:, None]).real
        if distmarg is None:
            lnLt = kappa_sq - 0.5 * rho_sq
        else:
            lnLt = _distmarg_lnL_numpy(kappa_sq, rho_sq, distmarg)
        lnLt_all[c] = lnLt + log_w[c]

    # lnL[j] = log( sum_c sum_t w_t exp(lnLt_all[c,j,t]) ) - log_w_norm
    mx = np.max(lnLt_all, axis=(0, 2))                           # (n_ext,)
    finite = np.isfinite(mx)
    lnL = np.full(n_ext, -np.inf)
    if np.any(finite):
        contrib = w_t[None, None, :] * np.exp(lnLt_all[:, finite, :] - mx[finite][None, :, None])
        S = np.sum(contrib, axis=(0, 2))                        # (n_ext_finite,)
        lnL[finite] = mx[finite] + np.log(S) - log_w_norm
    return lnL
