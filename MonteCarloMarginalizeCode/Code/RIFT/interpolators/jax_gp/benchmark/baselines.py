"""
Non-AD baselines for benchmarking the JAX interpolators.

The key comparator is **random forest** (ExtraTreesRegressor), which is RIFT's
default CIP fit method (`fit_rf`). RFs are the bar to beat: robust, very fast,
cover the full lnL dynamic range, and need no hyperparameter tuning. Their one
disqualifying weakness for the AD-export goal is that they are **not
differentiable** (piecewise-constant) -- so they expose no `lnL_and_grad`.

Including RF in the harness answers the honest question: how close do the
(differentiable, exportable) GP methods get to the robust non-AD default?
"""
from __future__ import annotations

import numpy as np


class RFBaseline:
    """Random-forest lnL regressor matching RIFT's CIP `fit_rf` (ExtraTrees, 100).

    Exposes the predict half of the interpolator contract (`fit`, `predict`,
    `predict_callable`) but deliberately NOT `lnL_and_grad`/`grad_fn`: a forest
    has no usable gradient, which is exactly the property that motivates the GP
    work.  `y_errors` are accepted (as sample weights ~ 1/sigma^2) for parity.
    """

    name = "rf"

    def __init__(self, n_estimators=100, random_state=0, **kw):
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def fit(self, X, y, y_errors=None):
        from sklearn.ensemble import ExtraTreesRegressor
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, float).ravel()
        sample_weight = None
        if y_errors is not None:
            ye = np.asarray(y_errors, float).ravel()
            sample_weight = 1.0 / np.clip(ye, 1e-3, None) ** 2
        self._rf = ExtraTreesRegressor(n_estimators=self.n_estimators,
                                       random_state=self.random_state)
        self._rf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        return self._rf.predict(X)

    def predict_callable(self):
        return lambda X: self.predict(X)
