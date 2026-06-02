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
