"""Common Fit base class. Provides finite-difference gradient as a default,
and an optional `predict_with_std` mode for acquisition-function placement
(see samplers.ucb)."""
import numpy as np


class FitBase:
    """Subclasses must implement .predict(Z). Optionally override .grad(Z)
    and .predict_with_std(Z) for fits that expose a calibrated posterior std.

    The default predict_with_std returns (mean, zeros); subclasses that have a
    real uncertainty estimate (RF tree-disagreement, GP posterior variance,
    BayesianLeastSquares posterior covariance, ...) should override it so the
    UCB sampler can use mu + kappa*sigma. Sampler-side code is responsible for
    falling back gracefully when sigma is identically zero (e.g. by widening
    kappa or by warning the user).

    Subclasses may signal "no real uncertainty" by setting
    `self.has_uncertainty = False`. Default is False; RF, GP, quadratic
    Bayesian-least-squares fits should set True.
    """

    has_uncertainty = False
    # Hint for the UCB local-polish step: pure piecewise-constant fits
    # (e.g. random forests) have zero or near-zero gradient nearly everywhere,
    # so gradient ascent doesn't work and we need a coordinate-hop polish.
    smooth_gradient = True

    def predict(self, Z):
        raise NotImplementedError

    def predict_with_std(self, Z):
        """Return (mean, std) on Z. Default: (predict(Z), zeros).
        Override in subclasses that expose a real uncertainty."""
        mean = self.predict(Z)
        return mean, np.zeros_like(mean)

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
