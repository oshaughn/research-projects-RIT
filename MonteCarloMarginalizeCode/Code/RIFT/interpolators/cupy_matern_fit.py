"""GPU (CuPy, float64) hyperparameter fit for the bounded Matérn recipe of matern_gp.

Same model as matern_gp.fit_matern_gp: standardized features, normalized targets,
Constant * anisotropic Matérn-5/2 + White, per-row alpha = sigma^2/std(y)^2, one
bounded L-BFGS-B start in log-hyperparameters. Only the marginal likelihood and its
gradient run on the GPU, so training sizes several times the CPU bound are affordable.
Returns a CachedMaternMean, the same predictor the cupy prediction backend uses.
"""
import time

import numpy as np
from scipy.optimize import minimize

from RIFT.interpolators.cached_matern_gp import CachedMaternMean
from RIFT.interpolators.matern_gp import array_hash, select_training_rows

# log bounds identical to matern_gp: ConstantKernel(.01,100), Matern ls (.03,30), White (1e-5,.1)
_C_BOUNDS, _L_BOUNDS, _W_BOUNDS = (.01, 100.), (.03, 30.), (1e-5, .1)


def _solve_lower(xp, chol, rhs):
    if xp.__name__ == "cupy":
        from cupyx.scipy.linalg import solve_triangular
    else:
        from scipy.linalg import solve_triangular
    return solve_triangular(chol, rhs, lower=True)


def _nlml_and_grad(theta, xs, yn, alpha_diag, xp):
    """Negative log marginal likelihood and its gradient in theta = log([C, l_1..l_d, W]).

    Peak memory is about six n x n float64 arrays; temporaries are released as soon as possible.
    """
    n, d = xs.shape
    const, lengths, white = np.exp(theta[0]), np.exp(theta[1:1 + d]), np.exp(theta[-1])
    z = xs / xp.asarray(lengths)
    sq = xp.zeros((n, n), dtype=xp.float64)
    for k in range(d):
        diff = z[:, k, None] - z[None, :, k]
        diff *= diff
        sq += diff
    del diff
    r5 = xp.sqrt(5.0 * sq)
    e = xp.exp(-r5)
    k_signal = const * (1 + r5 + (5.0 / 3.0) * sq) * e
    common = const * (5.0 / 3.0) * (1 + r5) * e      # dK/dlog l_k = common * (dz_k)^2
    del sq, r5, e
    kmat = k_signal + xp.diag(alpha_diag + white)
    try:
        chol = xp.linalg.cholesky(kmat)
    except Exception:
        return np.inf, np.zeros_like(theta)
    del kmat
    if not bool(xp.all(xp.isfinite(chol))):
        return np.inf, np.zeros_like(theta)
    logdet_half = float(xp.sum(xp.log(xp.diag(chol))))
    linv = _solve_lower(xp, chol, xp.eye(n, dtype=xp.float64))
    del chol
    w = linv.T @ linv                                 # K^-1
    del linv
    a = w @ yn
    nlml = 0.5 * float(yn @ a) + logdet_half + 0.5 * n * np.log(2 * np.pi)
    w -= xp.outer(a, a)                               # dNLML/dtheta = 0.5 tr(W dK/dtheta)
    grad = np.empty_like(theta)
    grad[0] = 0.5 * float(xp.sum(w * k_signal))
    grad[-1] = 0.5 * white * float(xp.trace(w))
    del k_signal
    w *= common
    del common
    for k in range(d):
        diff = z[:, k, None] - z[None, :, k]
        diff *= diff
        diff *= w
        grad[1 + k] = 0.5 * float(xp.sum(diff))
    return nlml, grad


def fit_matern_gp_cupy(x, y, y_errors, *, max_train_points=4800, optimizer_maxiter=25,
                       seed=25062842, rho1=None, feature_names=None, provenance=None,
                       batch_size=4096, xp=None):
    """Return (CachedMaternMean, record); `xp` defaults to cupy (numpy for CPU tests)."""
    if xp is None:
        import cupy as xp
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    errors = np.asarray(y_errors, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or errors.shape != y.shape:
        raise ValueError("Expected X(n,d), y(n), y_errors(n)")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(errors))) or np.any(errors < 0):
        raise ValueError("Training features/targets/errors must be finite; errors nonnegative")
    indices, selection = select_training_rows(y, max_train_points, seed, rho1)
    xt, yt = x[indices], y[indices]
    fmean, fscale = xt.mean(0), xt.std(0)
    fscale[fscale == 0] = 1.0                    # StandardScaler convention
    tmean, tstd = float(yt.mean()), float(yt.std())
    if not (tstd > 0 and np.isfinite(tstd)):
        raise ValueError("Selected targets need nonzero finite variance")
    xs = xp.asarray((xt - fmean) / fscale)
    yn = xp.asarray((yt - tmean) / tstd)
    alpha_diag = xp.asarray(np.maximum(errors[indices] ** 2 / tstd ** 2, 1e-10))
    d = x.shape[1]
    theta0 = np.log(np.r_[1.0, np.ones(d), .001])
    bounds = [tuple(np.log(_C_BOUNDS))] + [tuple(np.log(_L_BOUNDS))] * d + [tuple(np.log(_W_BOUNDS))]
    start = time.monotonic()
    res = minimize(_nlml_and_grad, theta0, args=(xs, yn, alpha_diag, xp), method="L-BFGS-B", jac=True,
                   bounds=bounds, options={"maxiter": int(optimizer_maxiter)})
    theta = res.x
    lo, hi = np.array(bounds).T
    at_bounds = [i for i in range(len(theta)) if min(theta[i] - lo[i], hi[i] - theta[i]) < 1e-3]
    if not res.success or at_bounds:
        print(" WARNING cupy_matern_fit: optimizer success=%s, hyperparameters at bounds (index into [C, l_1..l_d, W]): %s"
              % (bool(res.success), at_bounds))
    const, lengths, white = np.exp(theta[0]), np.exp(theta[1:1 + d]), np.exp(theta[-1])
    # Final weights alpha_ = K^-1 y_n at the optimum (white noise in K, as sklearn does)
    z = xs / xp.asarray(lengths)
    sq = xp.zeros((len(xs), len(xs)), dtype=xp.float64)
    for k in range(d):
        diff = z[:, k, None] - z[None, :, k]
        sq += diff * diff
    r5 = np.sqrt(5.0) * xp.sqrt(sq)
    kmat = const * (1 + r5 + (5.0 / 3.0) * sq) * xp.exp(-r5) + xp.diag(alpha_diag + white)
    chol = xp.linalg.cholesky(kmat)
    weights = xp.linalg.solve(chol.T, xp.linalg.solve(chol, yn))
    weights = weights.get() if hasattr(weights, "get") else np.asarray(weights)
    xs_host = xs.get() if hasattr(xs, "get") else np.asarray(xs)
    elapsed = time.monotonic() - start
    model = CachedMaternMean(x_train=xs_host, alpha=weights, feature_mean=fmean, feature_scale=fscale,
                             length_scale=lengths, constant=const, target_mean=tmean, target_scale=tstd,
                             backend="cupy" if xp.__name__ == "cupy" else "numpy", batch_size=batch_size)
    record = dict(recipe="standardized float64 Constant*anisotropic Matern5/2 + White (CuPy fit)",
                  native_rows=len(y), training_rows=len(indices), dimensions=d,
                  feature_names=None if feature_names is None else list(feature_names),
                  selection=selection, seed=int(seed), selected_indices=indices.tolist(),
                  selected_indices_sha256=array_hash(indices), X_sha256=array_hash(x), Y_sha256=array_hash(y),
                  target_std=tstd, constant=float(const), length_scale=lengths.tolist(), white=float(white),
                  theta=theta.tolist(), optimizer=dict(success=bool(res.success), iterations=int(res.nit),
                  evaluations=int(res.nfev), message=str(res.message), nlml=float(res.fun),
                  gradient_max_abs=float(np.max(np.abs(res.jac))), at_bounds=at_bounds),
                  error_sha256=array_hash(errors), alpha_sha256=array_hash(alpha_diag.get() if hasattr(alpha_diag, "get") else alpha_diag),
                  optimizer_maxiter=int(optimizer_maxiter), elapsed_seconds=elapsed,
                  provenance=dict(provenance or {}))
    return model, record
