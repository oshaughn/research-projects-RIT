"""Random-Forest fit. Production default per project owner (2026-05-13).
Gradient via finite differences from FitBase."""
import numpy as np
from ._base import FitBase

try:
    from sklearn.ensemble import RandomForestRegressor
    _HAVE_RF = True
except ImportError:
    _HAVE_RF = False


class RandomForestFit(FitBase):
    def __init__(self, X, Y, sigma=None, n_estimators=100, n_jobs=-1):
        if not _HAVE_RF:
            raise ImportError("sklearn not available; install scikit-learn or "
                              "choose --tracer-fit-method quadratic")
        self._rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
        weight = None if sigma is None else 1.0 / (np.asarray(sigma) ** 2 + 1e-12)
        self._rf.fit(X, Y, sample_weight=weight)

    def predict(self, Z):
        return self._rf.predict(np.atleast_2d(Z))
