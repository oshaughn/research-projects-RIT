"""Random-Forest fit. Production default per project owner (2026-05-13).

For UCB placement, exposes a tree-disagreement std as the uncertainty
estimate via predict_with_std. This is *not* a calibrated posterior std --
it is the empirical spread of the per-tree predictions, which is large in
unexplored regions (because trees disagree on extrapolation) and small in
well-sampled regions (because trees fit similar values). That qualitative
behavior is what UCB needs; for calibration use a GP fit when one becomes
available.
"""
import numpy as np
from ._base import FitBase

try:
    from sklearn.ensemble import RandomForestRegressor
    _HAVE_RF = True
except ImportError:
    _HAVE_RF = False


class RandomForestFit(FitBase):
    has_uncertainty = True
    # Tree predictions are piecewise-constant -- finite-difference gradients
    # are zero almost everywhere. Tell the UCB sampler to use a coordinate
    # hop polish instead of gradient ascent.
    smooth_gradient = False

    def __init__(self, X, Y, sigma=None, n_estimators=100, n_jobs=-1):
        if not _HAVE_RF:
            raise ImportError("sklearn not available; install scikit-learn or "
                              "choose --tracer-fit-method quadratic")
        self._rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
        weight = None if sigma is None else 1.0 / (np.asarray(sigma) ** 2 + 1e-12)
        self._rf.fit(X, Y, sample_weight=weight)

    def predict(self, Z):
        return self._rf.predict(np.atleast_2d(Z))

    def predict_with_std(self, Z):
        """Return (mean, tree_disagreement_std). Std is computed across the
        forest's per-tree predictions; larger where trees disagree."""
        Z = np.atleast_2d(Z)
        per_tree = np.stack([t.predict(Z) for t in self._rf.estimators_], axis=0)
        return per_tree.mean(axis=0), per_tree.std(axis=0)
