"""Fits for the tracer engine.

Public entry point: build(method, X, Y, sigma=None) -> Fit.

Fit objects expose:
    .predict(Z)  -> ndarray of len(Z)
    .grad(Z)     -> ndarray (len(Z), d)   (analytic where available, FD otherwise)
"""
from ._dispatch import build

__all__ = ["build"]
