"""Fits for the tracer engine.

Public entry point: build(method, X, Y, sigma=None, lnl_floor_delta=None) -> Fit.

`lnl_floor_delta` (default None = off, legacy behaviour bit-for-bit) clamps the
training lnL from below at max(lnL) - delta instead of cutting those points;
see _dispatch.apply_lnl_floor.

Fit objects expose:
    .predict(Z)           -> ndarray of len(Z)
    .predict_with_std(Z)  -> (mean, std); real std only where
                             .has_uncertainty is True (rf, gp_linmean)
    .grad(Z)              -> ndarray (len(Z), d)   (analytic where available,
                             FD otherwise)
"""
from ._dispatch import apply_lnl_floor, build

__all__ = ["build", "apply_lnl_floor"]
