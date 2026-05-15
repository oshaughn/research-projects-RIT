"""Degree-N polynomial fit (default degree 3). Kept available; quadratic is the
degree-2 special case via _quadratic.py."""
import numpy as np
from ._base import FitBase


class PolynomialFit(FitBase):
    def __init__(self, X, Y, sigma=None, degree=3, ridge=1e-3):
        from itertools import combinations_with_replacement
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        self.d = X.shape[1]; self.degree = degree
        # mean-normalize
        self.mu = X.mean(axis=0); self.sd = X.std(axis=0) + 1e-12
        Xn = (X - self.mu) / self.sd
        terms = []
        self._idx = []  # list of tuples of column indices in each term
        for k in range(degree + 1):
            for combo in combinations_with_replacement(range(self.d), k):
                self._idx.append(combo)
                terms.append(np.prod(Xn[:, combo], axis=1) if combo else np.ones(len(Xn)))
        D = np.column_stack(terms)
        W = ridge * np.eye(D.shape[1])
        if sigma is None:
            self._theta = np.linalg.lstsq(D.T @ D + W, D.T @ Y, rcond=None)[0]
        else:
            w = 1.0 / (np.asarray(sigma) ** 2 + 1e-12)
            DW = D * w[:, None]
            self._theta = np.linalg.lstsq(DW.T @ D + W, DW.T @ Y, rcond=None)[0]

    def _design(self, Zn):
        from itertools import combinations_with_replacement
        cols = []
        for combo in self._idx:
            cols.append(np.prod(Zn[:, combo], axis=1) if combo else np.ones(len(Zn)))
        return np.column_stack(cols)

    def predict(self, Z):
        Z = np.atleast_2d(Z)
        Zn = (Z - self.mu) / self.sd
        return self._design(Zn) @ self._theta
