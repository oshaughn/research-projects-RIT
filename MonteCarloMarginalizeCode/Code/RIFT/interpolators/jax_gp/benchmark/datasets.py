"""
Loaders for real RIFT ILE output (``.net`` / ``.composite`` files), so the
interpolators can be exercised on production-shaped lnL data rather than only
synthetic truths.

The standard RIFT ILE row layout (whitespace-separated) is::

    indx  m1 m2  s1x s1y s1z  s2x s2y s2z  lambda1 lambda2  lnL  sigma_lnL  ntot ...

``load_ile_net`` pulls out the chosen fit parameters and the lnL column, and (by
default) de-duplicates repeated intrinsic points, keeping the max lnL per point.
"""
from __future__ import annotations

import numpy as np


# Standard RIFT ILE column indices (0-based).
ILE_COLS = {
    "indx": 0, "m1": 1, "m2": 2,
    "s1x": 3, "s1y": 4, "s1z": 5,
    "s2x": 6, "s2y": 7, "s2z": 8,
    "lambda1": 9, "lambda2": 10,
    "lnL": 11, "sigma_lnL": 12, "ntot": 13,
}

DEFAULT_FIT_PARAMS = ("m1", "m2", "s1z", "s2z", "lambda1", "lambda2")

# Good low-mass / BNS fit coordinates (decorrelated). mu1,mu2 are Morisaki's
# orthogonalized PN-phase combinations; LambdaTilde/DeltaLambdaTilde the standard
# tidal combinations. Far easier for a GP than raw (m1,m2,lambda1,lambda2).
BNS_FIT_COORDS = ("mu1", "mu2", "delta_mc", "LambdaTilde", "DeltaLambdaTilde")


def mc_delta_from_m1m2(m1, m2):
    """(m1, m2) -> (chirp mass, delta_mc=(m1-m2)/(m1+m2))."""
    m1 = np.asarray(m1, float); m2 = np.asarray(m2, float)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return mc, (m1 - m2) / (m1 + m2)


def to_fit_coordinates(X_low, low_level_names, fit_coord_names):
    """Map low-level intrinsic columns to RIFT fit coordinates.

    Thin wrapper over ``lalsimutils.convert_waveform_coordinates`` -- the same
    transform CIP applies before fitting. Requires RIFT (lalsimutils) importable.
    ``X_low`` columns must be in ``low_level_names`` order.
    """
    import RIFT.lalsimutils as lsu
    return lsu.convert_waveform_coordinates(
        np.asarray(X_low, dtype=np.float64),
        coord_names=list(fit_coord_names),
        low_level_coord_names=list(low_level_names))


def load_ile_net(path, fit_params=DEFAULT_FIT_PARAMS, cols=None,
                 lnL_col="lnL", dedupe=True, max_rows=None):
    """Load an ILE ``.net`` file into (X, y, coord_names).

    Parameters
    ----------
    path : str
    fit_params : sequence of column names (keys of ``cols``) -> columns of X
    cols : dict name->index (defaults to the standard RIFT layout)
    lnL_col : which column is the target lnL
    dedupe : collapse repeated intrinsic points, keeping the max lnL per point
    max_rows : optional cap for quick experiments

    Returns
    -------
    X : ndarray [n, d]
    y : ndarray [n]            lnL values
    coord_names : list[str]    the fit parameter names (axes of X)
    """
    cols = dict(ILE_COLS) if cols is None else cols
    data = np.loadtxt(path, max_rows=max_rows)
    X = np.column_stack([data[:, cols[p]] for p in fit_params]).astype(np.float64)
    y = data[:, cols[lnL_col]].astype(np.float64)

    # Drop non-finite rows (failed evaluations).
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]

    if dedupe:
        keys, inv = np.unique(X, axis=0, return_inverse=True)
        ymax = np.full(len(keys), -np.inf)
        np.maximum.at(ymax, inv, y)
        X, y = keys, ymax

    return X, y, list(fit_params)
