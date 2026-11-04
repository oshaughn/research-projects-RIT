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
                 lnL_col="lnL", err_col="sigma_lnL", sigma_cut=None,
                 dedupe=True, return_errors=False, max_rows=None):
    """Load an ILE ``.net`` file into (X, y[, yerr], coord_names).

    Parameters
    ----------
    path : str
    fit_params : sequence of column names (keys of ``cols``) -> columns of X
    cols : dict name->index (defaults to the standard RIFT layout)
    lnL_col : which column is the target lnL
    err_col : which column is the per-point lnL Monte-Carlo error (sigma_lnL).
        In RIFT ILE output this is the "sigma/L" column (= sigma_L/L = std of lnL).
    sigma_cut : if set, drop points with reported error above this (CIP default 0.6)
    dedupe : collapse repeated intrinsic points. With errors this is an
        inverse-variance combine (the statistically correct merge of repeated MC
        evaluations); without, it keeps the max lnL.
    return_errors : also return the per-point lnL error array
    max_rows : optional cap for quick experiments

    Returns
    -------
    X : ndarray [n, d]
    y : ndarray [n]                    lnL values
    yerr : ndarray [n]                 (only if return_errors) per-point sigma_lnL
    coord_names : list[str]            the fit parameter names (axes of X)
    """
    cols = dict(ILE_COLS) if cols is None else cols
    data = np.loadtxt(path, max_rows=max_rows)
    X = np.column_stack([data[:, cols[p]] for p in fit_params]).astype(np.float64)
    y = data[:, cols[lnL_col]].astype(np.float64)
    yerr = data[:, cols[err_col]].astype(np.float64) if err_col in cols else None

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if yerr is not None:
        ok &= np.isfinite(yerr)
        if sigma_cut is not None:
            ok &= yerr <= sigma_cut
    X, y = X[ok], y[ok]
    yerr = yerr[ok] if yerr is not None else None

    if dedupe:
        keys, inv = np.unique(X, axis=0, return_inverse=True)
        if yerr is not None:
            # inverse-variance combine of repeated evaluations of the same point
            w = 1.0 / np.clip(yerr, 1e-3, None) ** 2
            wsum = np.zeros(len(keys)); wy = np.zeros(len(keys))
            np.add.at(wsum, inv, w)
            np.add.at(wy, inv, w * y)
            y = wy / wsum
            yerr = 1.0 / np.sqrt(wsum)
        else:
            ymax = np.full(len(keys), -np.inf)
            np.maximum.at(ymax, inv, y)
            y = ymax
        X = keys

    if return_errors:
        return X, y, yerr, list(fit_params)
    return X, y, list(fit_params)
