"""RBF interpolator fit. Smooth gradients; good for MALA."""
import numpy as np
from ._base import FitBase

try:
    from scipy.interpolate import RBFInterpolator
    _HAVE_RBF = True
except ImportError:
    _HAVE_RBF = False


class RBFFit(FitBase):
    def __init__(self, X, Y, sigma=None, smoothing=1e-3):
        if not _HAVE_RBF:
            raise ImportError("scipy not available; cannot use --tracer-fit-method rbf")
        # scipy's RBFInterpolator doesn't take sample weights; use smoothing as a
        # proxy when sigma is supplied.
        s = smoothing
        if sigma is not None:
            s = float(np.median(np.asarray(sigma) ** 2)) + smoothing
        self._rbf = RBFInterpolator(X, Y, smoothing=s)

    def predict(self, Z):
        return self._rbf(np.atleast_2d(Z))
