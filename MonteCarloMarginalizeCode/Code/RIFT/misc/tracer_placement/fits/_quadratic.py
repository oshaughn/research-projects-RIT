"""Cheap quadratic fit (Tikhonov-regularized). Not for production per
2026-05-13 design call; kept available for unit tests + smoke runs."""
import numpy as np
from ._base import FitBase
from ..samplers.surrogate import QuadFit as _QuadFit


class QuadraticFit(FitBase):
    def __init__(self, X, Y, sigma=None, ridge=1e-3):
        weight = None if sigma is None else 1.0 / (np.asarray(sigma) ** 2 + 1e-12)
        self._inner = _QuadFit(X, Y, ridge=ridge, eval_weight=weight)

    def predict(self, Z):
        return self._inner.f(Z)

    def grad(self, Z, eps=None):
        return self._inner.grad(Z)        # analytic
