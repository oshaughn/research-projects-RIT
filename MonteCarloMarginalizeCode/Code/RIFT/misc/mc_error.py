"""Helpers for normalizing Monte Carlo integration error conventions."""

import math

import numpy as np


def relative_mc_error(result, variance, *, log_space=False):
    """Return the relative standard error for linear- or log-space results.

    Linear-space integrators return ``(Z, Var[Z])``.  Log-space integrators
    return ``(ln Z, ln Var[Z])``; taking ``sqrt`` of the latter is invalid and
    was the source of ``nan`` CIP evidence annotations for AV sampling.
    """
    # Cartesian samplers accumulate in extended precision. Keep that range
    # until after the square root and division, even in finiteness checks.
    if not np.isfinite(result) or np.isnan(variance):
        return math.nan
    if log_space:
        if variance == -math.inf:
            return 0.0
        if variance == math.inf:
            return math.inf
        try:
            return math.exp(0.5 * variance - result)
        except OverflowError:
            return math.inf
    if variance == math.inf:
        return math.inf
    if variance < 0:
        raise ValueError("linear-space integration variance must be non-negative")
    if result == 0:
        return 0.0 if variance == 0 else math.inf
    return np.sqrt(variance) / abs(result)
