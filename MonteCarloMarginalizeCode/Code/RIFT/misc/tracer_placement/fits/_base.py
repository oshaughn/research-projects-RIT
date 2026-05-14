"""Common Fit base class. Provides finite-difference gradient as a default."""
import numpy as np


class FitBase:
    """Subclasses must implement .predict(Z). Optionally override .grad(Z)."""

    def predict(self, Z):
        raise NotImplementedError

    def f(self, Z):
        return self.predict(Z)

    def grad(self, Z, eps=1e-3):
        Z = np.atleast_2d(Z)
        d = Z.shape[1]
        out = np.zeros_like(Z)
        for k in range(d):
            zp = Z.copy(); zp[:, k] += eps
            zm = Z.copy(); zm[:, k] -= eps
            out[:, k] = (self.predict(zp) - self.predict(zm)) / (2 * eps)
        return out
