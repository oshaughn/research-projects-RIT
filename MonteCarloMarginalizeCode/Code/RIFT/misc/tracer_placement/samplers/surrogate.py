"""Local quadratic surrogate of ln-likelihood, RIFT BayesianLeastSquares-flavored.

f(x) = a + b^T x + 0.5 x^T C x   with Tikhonov regularization on coefficients.
Gradient: b + C x. Closed form.
"""
import numpy as np

def _design(X):
    n, d = X.shape
    cols = [np.ones(n), X]                              # 1, linear
    quad = []
    for i in range(d):
        for j in range(i, d):
            quad.append(X[:, i] * X[:, j])
        # square terms get factor 1, cross-terms factor 1 (we'll handle 0.5 in eval)
    cols.append(np.column_stack(quad))
    return np.concatenate([np.ones((n,1)), X, np.column_stack(quad)], axis=1)

def _coef_unpack(theta, d):
    a = theta[0]
    b = theta[1:1+d]
    # quad terms in order (0,0),(0,1),...,(0,d-1),(1,1),(1,2),...
    C = np.zeros((d, d))
    k = 1 + d
    for i in range(d):
        for j in range(i, d):
            v = theta[k]; k += 1
            if i == j:
                C[i, i] = 2.0 * v   # since 0.5 x^T C x => C_ii contributes 0.5 C_ii x_i^2
            else:
                C[i, j] = v
                C[j, i] = v
    return a, b, C

class QuadFit:
    def __init__(self, X, y, ridge=1e-3, eval_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.d = X.shape[1]
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-12
        Xn = (X - self.X_mean) / self.X_std
        D = _design(Xn)
        W = np.eye(D.shape[1]) * ridge
        if eval_weight is None:
            A = D.T @ D + W
            rhs = D.T @ y
        else:
            wv = np.asarray(eval_weight, dtype=float)
            DW = D * wv[:, None]
            A = DW.T @ D + W
            rhs = DW.T @ y
        theta, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        self.theta = theta
        self.a, self.b_n, self.C_n = _coef_unpack(theta, self.d)

    def _to_normed(self, X):
        return (np.atleast_2d(X) - self.X_mean) / self.X_std

    def f(self, X):
        Xn = self._to_normed(X)
        v = self.a + Xn @ self.b_n + 0.5 * np.einsum('ni,ij,nj->n', Xn, self.C_n, Xn)
        return v

    def grad(self, X):
        Xn = self._to_normed(X)
        gn = self.b_n + Xn @ self.C_n                  # gradient in normed coords
        return gn / self.X_std                          # back to native coords
