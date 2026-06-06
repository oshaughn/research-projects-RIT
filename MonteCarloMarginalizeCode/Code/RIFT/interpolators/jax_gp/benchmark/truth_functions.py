"""
Synthetic ground-truth lnL functions with *analytic* values and gradients.

These exist so the harness can measure interpolation accuracy -- and crucially
gradient accuracy -- against an exact answer, in the d=8-12 regime where CIP
actually operates.  Each function exposes:

    lnL(X)          : [n, d] -> [n]          exact log-likelihood
    grad(X)         : [n, d] -> [n, d]       exact gradient of lnL
    sample_domain(n, rng) : -> [n, d]        draws covering the interesting region

The shapes are chosen to stress the failure modes we care about:

  * CorrelatedGaussian -- the easy, strongly-correlated quadratic peak (the bread
    and butter of lnL surfaces; tests basic fidelity and length-scale handling).
  * BananaRidge        -- a curved Rosenbrock ridge in the first two dims, a stand-in
    for the strong mc-eta style degeneracies that break stationary kernels.
  * MultimodalMixture  -- several separated Gaussian modes (tests global behaviour).
  * SharpPeak          -- a narrow peak on a broad shoulder (tests dynamic range and
    the non-stationarity that RFF struggles with).
"""
from __future__ import annotations

import numpy as np


def _as2d(X):
    X = np.asarray(X, dtype=np.float64)
    return X[None, :] if X.ndim == 1 else X


class TruthFunction:
    name = "base"

    def __init__(self, d):
        self.d = int(d)

    def lnL(self, X):
        raise NotImplementedError

    def grad(self, X):
        raise NotImplementedError

    def sample_domain(self, n, rng):
        raise NotImplementedError


class CorrelatedGaussian(TruthFunction):
    name = "correlated_gaussian"

    def __init__(self, d, seed=1):
        super().__init__(d)
        rng = np.random.default_rng(seed)
        # random correlated precision matrix P = L L^T + small ridge
        L = rng.normal(size=(d, d)) / np.sqrt(d)
        self.P = L @ L.T + 0.5 * np.eye(d)
        self.mu = rng.normal(size=d)

    def lnL(self, X):
        X = _as2d(X)
        dx = X - self.mu
        return -0.5 * np.einsum("ni,ij,nj->n", dx, self.P, dx)

    def grad(self, X):
        X = _as2d(X)
        return -(X - self.mu) @ self.P.T

    def sample_domain(self, n, rng):
        cov = np.linalg.inv(self.P)
        return rng.multivariate_normal(self.mu, 2.0 * cov, size=n)


class BananaRidge(TruthFunction):
    """Curved ridge in dims (0,1); independent Gaussian in the remaining dims."""

    name = "banana_ridge"

    def __init__(self, d, b=0.5, seed=2):
        super().__init__(d)
        self.b = float(b)
        self.rng_seed = seed

    def lnL(self, X):
        X = _as2d(X)
        x0, x1 = X[:, 0], X[:, 1]
        ridge = -0.5 * (x0 ** 2 / 4.0) - 0.5 * (x1 - self.b * (x0 ** 2 - 4.0)) ** 2
        rest = -0.5 * np.sum(X[:, 2:] ** 2, axis=1)
        return ridge + rest

    def grad(self, X):
        X = _as2d(X)
        x0, x1 = X[:, 0], X[:, 1]
        g = np.zeros_like(X)
        r = x1 - self.b * (x0 ** 2 - 4.0)
        g[:, 0] = -x0 / 4.0 + r * self.b * 2.0 * x0
        g[:, 1] = -r
        g[:, 2:] = -X[:, 2:]
        return g

    def sample_domain(self, n, rng):
        X = np.zeros((n, self.d))
        X[:, 0] = rng.normal(0, 2.0, size=n)
        X[:, 1] = self.b * (X[:, 0] ** 2 - 4.0) + rng.normal(0, 1.0, size=n)
        if self.d > 2:
            X[:, 2:] = rng.normal(0, 1.0, size=(n, self.d - 2))
        return X


class MultimodalMixture(TruthFunction):
    """log-sum-exp of K separated Gaussian modes (smooth, multimodal)."""

    name = "multimodal_mixture"

    def __init__(self, d, k=3, sep=4.0, seed=3):
        super().__init__(d)
        rng = np.random.default_rng(seed)
        self.centers = rng.normal(0, sep, size=(k, d))
        self.k = int(k)

    def lnL(self, X):
        X = _as2d(X)
        # quad[n,k] = -0.5 ||x - c_k||^2 ; lnL = logsumexp_k quad
        diff = X[:, None, :] - self.centers[None, :, :]
        quad = -0.5 * np.sum(diff ** 2, axis=2)
        m = np.max(quad, axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.sum(np.exp(quad - m), axis=1)))

    def grad(self, X):
        X = _as2d(X)
        diff = X[:, None, :] - self.centers[None, :, :]
        quad = -0.5 * np.sum(diff ** 2, axis=2)
        w = np.exp(quad - np.max(quad, axis=1, keepdims=True))
        w = w / np.sum(w, axis=1, keepdims=True)            # softmax responsibilities
        # grad = sum_k w_k * (-(x - c_k))
        return -np.einsum("nk,nki->ni", w, diff)

    def sample_domain(self, n, rng):
        which = rng.integers(0, self.k, size=n)
        return self.centers[which] + rng.normal(0, 1.2, size=(n, self.d))


class SharpPeak(TruthFunction):
    """Narrow Gaussian peak superposed on a broad shoulder (non-stationary)."""

    name = "sharp_peak"

    def __init__(self, d, narrow=0.3, broad=3.0, amp=6.0, seed=4):
        super().__init__(d)
        self.s2n = narrow ** 2
        self.s2b = broad ** 2
        self.amp = float(amp)

    def lnL(self, X):
        X = _as2d(X)
        r2 = np.sum(X ** 2, axis=1)
        return self.amp * np.exp(-0.5 * r2 / self.s2n) - 0.5 * r2 / self.s2b

    def grad(self, X):
        X = _as2d(X)
        r2 = np.sum(X ** 2, axis=1, keepdims=True)
        peak = self.amp * np.exp(-0.5 * r2 / self.s2n)
        return -peak * X / self.s2n - X / self.s2b

    def sample_domain(self, n, rng):
        # mix draws near the sharp core and over the broad shoulder
        n_core = n // 2
        core = rng.normal(0, 0.5, size=(n_core, self.d))
        broad = rng.normal(0, 2.5, size=(n - n_core, self.d))
        return np.vstack([core, broad])


def all_truths(d, seed=0):
    """Return one instance of each truth at dimension ``d``."""
    return [
        CorrelatedGaussian(d, seed=seed + 1),
        BananaRidge(d, seed=seed + 2),
        MultimodalMixture(d, seed=seed + 3),
        SharpPeak(d, seed=seed + 4),
    ]
