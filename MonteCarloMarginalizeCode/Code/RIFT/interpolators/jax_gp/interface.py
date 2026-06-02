"""
Common interface and shared plumbing for JAX likelihood interpolators.

Subclasses fit on *whitened* coordinates and define a single pure-JAX method,
``_lnL_whitened(xw)``, mapping a whitened coordinate vector ``[d]`` to a scalar
lnL expressed in whitened-target units.  The base class owns everything else:

  * per-dimension input whitening and target centering / scaling,
  * the batched ``predict(X)`` NumPy contract consumed by CIP,
  * the differentiable ``lnL_and_grad(theta)`` / ``lnL_physical(theta)`` contract
    used for the AD-compatible export.

Because whitening is an affine map, JAX's autodiff threads the chain rule
through it automatically -- the gradient returned by ``lnL_and_grad`` is the
gradient with respect to the *physical* coordinates, in physical lnL units.
"""
from __future__ import annotations

import abc

import numpy as np
import jax
import jax.numpy as jnp


def _as2d(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    return X


class BaseInterpolator(abc.ABC):
    """Abstract base for whitened-coordinate JAX interpolators."""

    #: short name, overridden by subclasses (matches CIP ``--fit-method`` suffix)
    name = "base"
    #: bumped if the on-disk export layout changes incompatibly
    SCHEMA_VERSION = 1

    def __init__(self, jitter=1e-6):
        self.jitter = float(jitter)
        self._fitted = False
        # populated by fit():
        self.x_mean = None   # jnp [d]
        self.x_std = None    # jnp [d]
        self.y_mean = None   # float
        self.y_std = None    # float
        self._lnL_whitened_v = None  # jitted, vmapped predictor

    # ------------------------------------------------------------------ #
    # fit                                                                 #
    # ------------------------------------------------------------------ #
    def fit(self, X, y, y_errors=None):
        """Standardize inputs/targets, delegate to ``_fit_whitened``.

        Parameters
        ----------
        X : array [n, d]   coordinates (1-D is promoted to a single column)
        y : array [n]      lnL values
        y_errors : array [n], optional   per-point lnL uncertainties
        """
        X = _as2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.x_mean = jnp.asarray(np.mean(X, axis=0))
        x_std = np.std(X, axis=0)
        x_std[x_std == 0] = 1.0                      # guard constant columns
        self.x_std = jnp.asarray(x_std)
        self.y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        self.y_std = y_std if y_std > 0 else 1.0

        Xw = (jnp.asarray(X) - self.x_mean) / self.x_std
        yw = (jnp.asarray(y) - self.y_mean) / self.y_std
        yerr_w = None
        if y_errors is not None:
            yerr_w = jnp.asarray(np.asarray(y_errors, dtype=np.float64).ravel()) / self.y_std

        self._fit_whitened(Xw, yw, yerr_w)
        self._fitted = True
        # batched, jitted whitened predictor for the hot predict() path
        self._lnL_whitened_v = jax.jit(jax.vmap(self._lnL_whitened))
        return self

    @abc.abstractmethod
    def _fit_whitened(self, Xw, yw, yerr_w):
        """Fit on whitened data; store params needed by ``_lnL_whitened``."""

    @abc.abstractmethod
    def _lnL_whitened(self, xw):
        """Pure-JAX: whitened coord vector ``[d]`` -> scalar lnL (whitened units)."""

    # ------------------------------------------------------------------ #
    # serialization hooks (implemented per method)                        #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _export_params(self):
        """Return a dict of NumPy arrays sufficient to rebuild ``_lnL_whitened``."""

    @abc.abstractmethod
    def _import_params(self, params):
        """Restore method-specific state from the dict produced by ``_export_params``."""

    def export_state(self):
        """Return ``(meta, arrays)`` fully describing the fitted model.

        ``arrays`` holds the whitening vectors and method parameters (NumPy);
        ``meta`` holds scalars and identifying info.  Together they round-trip
        through :mod:`RIFT.interpolators.jax_gp.export`.
        """
        if not self._fitted:
            raise RuntimeError("export_state() called before fit()")
        arrays = {
            "x_mean": np.asarray(self.x_mean),
            "x_std": np.asarray(self.x_std),
        }
        arrays.update({"_param_" + k: np.asarray(v)
                       for k, v in self._export_params().items()})
        meta = {
            "schema": self.SCHEMA_VERSION,
            "method": self.name,
            "d": int(np.asarray(self.x_mean).shape[0]),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
            "jitter": float(self.jitter),
        }
        return meta, arrays

    @classmethod
    def from_state(cls, meta, arrays):
        """Reconstruct a (predict-only) model from ``export_state`` output."""
        obj = cls.__new__(cls)               # bypass __init__: ctor args are unknown
        BaseInterpolator.__init__(obj, jitter=meta.get("jitter", 1e-6))
        obj.x_mean = jnp.asarray(arrays["x_mean"])
        obj.x_std = jnp.asarray(arrays["x_std"])
        obj.y_mean = float(meta["y_mean"])
        obj.y_std = float(meta["y_std"])
        params = {k[len("_param_"):]: arrays[k]
                  for k in arrays if k.startswith("_param_")}
        obj._import_params(params)
        obj._fitted = True
        obj._lnL_whitened_v = jax.jit(jax.vmap(obj._lnL_whitened))
        return obj

    # ------------------------------------------------------------------ #
    # prediction (CIP contract)                                          #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        """Batched mean prediction. NumPy in, NumPy out -- the CIP contract."""
        if not self._fitted:
            raise RuntimeError("predict() called before fit()")
        X = _as2d(X)
        Xw = (jnp.asarray(X) - self.x_mean) / self.x_std
        yw = self._lnL_whitened_v(Xw)
        return np.asarray(self.y_mean + self.y_std * yw)

    def predict_callable(self):
        """Return ``lambda X: self.predict(X)`` for the CIP fit dispatch."""
        return lambda X: self.predict(X)

    # ------------------------------------------------------------------ #
    # differentiable export contract                                     #
    # ------------------------------------------------------------------ #
    def lnL_physical(self, theta):
        """Pure-JAX scalar lnL at a single *physical* coordinate vector ``[d]``.

        This is the function downstream users differentiate.  Keep it pure JAX.
        """
        xw = (theta - self.x_mean) / self.x_std
        return self.y_mean + self.y_std * self._lnL_whitened(xw)

    def lnL_and_grad(self, theta):
        """Value and gradient of lnL at one physical point (NumPy out)."""
        theta = jnp.asarray(theta, dtype=jnp.float64)
        val, grad = jax.value_and_grad(self.lnL_physical)(theta)
        return float(val), np.asarray(grad)

    def grad_fn(self):
        """Return a jitted pure-JAX ``theta -> (lnL, grad)`` for batched/AD use."""
        return jax.jit(jax.value_and_grad(self.lnL_physical))
