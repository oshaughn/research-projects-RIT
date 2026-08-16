"""build(method, X, Y, sigma=None, lnl_floor_delta=None) -> Fit."""
import sys

import numpy as np


def apply_lnl_floor(Y, delta):
    """Clamp lnL from below at max(lnL) - delta, returning the clamped copy.

    RIFT elsewhere CUTS instead: `indx_ok = Y > np.max(Y) - opts.lnL_offset`.
    Cutting is right when the discarded points are uninformative, but with
    catastrophic-fit outliers (a failed waveform / failed radiative-transfer
    model can land lnL at -1e9) it also discards the GEOMETRY of the known-bad
    region: the surrogate is then fit only to the good ridge and has no idea
    the cliff exists. Clamping keeps those points as anchors that still pin the
    surrogate's length scale and signal variance -- which is what makes a GP's
    sf^2 meaningful -- while removing the numerical damage of a -1e9 value.

    `delta=None` (the default everywhere) returns Y untouched, so the legacy
    behaviour is bit-for-bit unchanged.
    """
    if delta is None:
        return Y
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(f"lnl_floor_delta must be a positive finite number, "
                         f"got {delta!r}")
    Yv = np.asarray(Y, dtype=float)
    finite = np.isfinite(Yv)
    if not finite.any():
        raise ValueError("lnl_floor_delta given but no finite lnL values")
    floor = float(np.max(Yv[finite])) - delta
    n_below = int(np.sum(~(Yv >= floor)))       # counts NaN / -inf as below
    if n_below:
        sys.stderr.write(
            f"fits.build: lnL floor at max-{delta:g} = {floor:.4g} clamped "
            f"{n_below}/{len(Yv)} training point(s) (kept as anchors rather "
            f"than cut).\n")
    return np.where(Yv >= floor, Yv, floor)


def build(method, X, Y, sigma=None, lnl_floor_delta=None, **kw):
    method = method.lower().replace("-", "_")
    Y = apply_lnl_floor(Y, lnl_floor_delta)
    if method == "rf":
        from ._rf import RandomForestFit
        return RandomForestFit(X, Y, sigma=sigma, **kw)
    if method == "rbf":
        from ._rbf import RBFFit
        return RBFFit(X, Y, sigma=sigma, **kw)
    if method == "quadratic":
        from ._quadratic import QuadraticFit
        return QuadraticFit(X, Y, sigma=sigma, **kw)
    if method == "polynomial":
        from ._polynomial import PolynomialFit
        return PolynomialFit(X, Y, sigma=sigma, **kw)
    if method == "gp_linmean":
        from ._gp_linmean import LinearMeanGPFit
        return LinearMeanGPFit(X, Y, sigma=sigma, **kw)
    raise ValueError(f"unknown fit method {method!r}")
