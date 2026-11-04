"""Opt-in bounded standardized Matérn-5/2 interpolation for fresh native CIP fits.

This recipe is independent of historical RIFT ``gp`` (RBF). Exact float64
training remains on CPU; the cached adapter may accelerate deterministic means.
Selection balances transverse/likelihood strata when rho1 is supplied, otherwise
likelihood strata. This is a compute bound, not an accuracy guarantee.
"""
import hashlib
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def native_rho1(lowlevel, names):
    """Known native geometry only; None when this stage has no transverse axes."""
    values = np.asarray(lowlevel, dtype=np.float64)
    names = list(names)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Native low-level coordinates must align with names")
    if "chi1" in names and "cos_theta1" in names:
        chi = values[:, names.index("chi1")]
        cosine = values[:, names.index("cos_theta1")]
        rho = np.where(chi == 0, 0., chi * np.sqrt(np.maximum(0., 1. - cosine**2)))
    elif "s1x" in names and "s1y" in names:
        rho = np.hypot(values[:, names.index("s1x")], values[:, names.index("s1y")])
    elif "chi1_perp" in names:
        rho = values[:, names.index("chi1_perp")]
    else:
        return None
    if not np.all(np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError("Nonfinite/negative native transverse geometry")
    return rho


def select_training_rows(y, max_train_points=4800, seed=25062842, rho1=None):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)):
        raise ValueError("y must be a nonempty finite vector")
    if isinstance(max_train_points, bool) or int(max_train_points) != max_train_points or max_train_points < 2:
        raise ValueError("max_train_points must be an integer >=2")
    rng = np.random.default_rng(seed)
    likelihood_bin = np.digitize(y.max() - y, [5., 15.])
    if rho1 is None:
        groups = [rng.permutation(np.flatnonzero(likelihood_bin == j)).tolist() for j in range(3)]
        mode = "balanced delta-lnL strata [5,15]; transverse unavailable"
    else:
        rho1 = np.asarray(rho1, dtype=np.float64)
        if rho1.shape != y.shape or not np.all(np.isfinite(rho1)) or np.any(rho1 < 0):
            raise ValueError("rho1 must be a finite nonnegative vector aligned with y")
        transverse_bin = np.digitize(rho1, [.1, .3, .5, .7])
        groups = [rng.permutation(np.flatnonzero((transverse_bin == i) & (likelihood_bin == j))).tolist()
                  for i in range(5) for j in range(3)]
        mode = "balanced rho1 [0.1,0.3,0.5,0.7] x delta-lnL [5,15] strata"
    order = []
    while any(groups) and len(order) < max_train_points:
        for group in groups:
            if group and len(order) < max_train_points:
                order.append(group.pop())
    return np.asarray(order, dtype=np.int64), mode


def fit_matern_gp(x, y, y_errors, *, max_train_points=4800, optimizer_maxiter=25,
                  seed=25062842, rho1=None, feature_names=None, provenance=None):
    """Return fitted sklearn pipeline and a durable numerical/selection record."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    errors = np.asarray(y_errors, dtype=np.float64)
    if x.ndim != 2 or not x.shape[1] or y.shape != (len(x),) or errors.shape != y.shape:
        raise ValueError("Expected X(n,d), y(n), y_errors(n)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError("Training features/errors must be finite; errors nonnegative")
    if isinstance(optimizer_maxiter, bool) or int(optimizer_maxiter) != optimizer_maxiter or optimizer_maxiter < 1:
        raise ValueError("optimizer_maxiter must be a positive integer")
    if feature_names is not None and len(feature_names) != x.shape[1]:
        raise ValueError("feature_names must match the actual stage dimensionality")
    indices, selection = select_training_rows(y, max_train_points, seed, rho1)
    target_std = float(np.std(y[indices]))
    if target_std <= 0 or not np.isfinite(target_std):
        raise ValueError("Selected targets need nonzero finite variance")
    alpha = np.maximum(errors[indices] ** 2 / target_std ** 2, 1e-10)
    optimization = []

    def optimizer(objective, theta, bounds):
        start = time.monotonic()
        result = minimize(objective, theta, method="L-BFGS-B", jac=True,
                          bounds=bounds, options={"maxiter": int(optimizer_maxiter)})
        optimization.append(dict(success=bool(result.success), iterations=int(result.nit),
                                 evaluations=int(result.nfev), message=str(result.message),
                                 negative_log_marginal_likelihood=float(result.fun),
                                 gradient_max_abs=float(np.max(np.abs(result.jac))),
                                 elapsed_seconds=time.monotonic() - start))
        return result.x, result.fun

    kernel = (ConstantKernel(1., (.01, 100.)) *
              Matern(np.ones(x.shape[1]), (.03, 30.), nu=2.5) +
              WhiteKernel(.001, (1e-5, .1)))
    model = Pipeline([("scale", StandardScaler()),
                      ("gp", GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                       normalize_y=True, optimizer=optimizer, n_restarts_optimizer=0,
                       random_state=seed))])
    start = time.monotonic()
    model.fit(x[indices], y[indices])
    # Do not pickle an unimportable closure; this is already fitted.
    model.named_steps["gp"].optimizer = None
    record = dict(recipe="standardized float64 Constant*anisotropic Matern5/2 + White",
                  native_rows=len(y), training_rows=len(indices), dimensions=x.shape[1],
                  feature_names=None if feature_names is None else list(feature_names),
                  selection=selection, seed=int(seed), selected_indices=indices.tolist(),
                  selected_indices_sha256=array_hash(indices), X_sha256=array_hash(x),
                  Y_sha256=array_hash(y), error_sha256=array_hash(errors),
                  alpha_sha256=array_hash(alpha), alpha_floor=1e-10,
                  target_std=target_std, normalize_y=True,
                  kernel=str(model.named_steps["gp"].kernel_),
                  theta=model.named_steps["gp"].kernel_.theta.tolist(),
                  kernel_log_bounds=model.named_steps["gp"].kernel_.bounds.tolist(),
                  optimizer_maxiter=int(optimizer_maxiter), optimizer_restarts=0,
                  optimization=optimization, elapsed_seconds=time.monotonic() - start,
                  provenance=dict(provenance or {}),
                  warning="No independent heldout validation is performed by this fresh-fit mode; inspect interpolation accuracy separately.")
    model.rift_matern_provenance = record
    return model, record
