"""Eccentric proposal defaults; native priors remain the caller's choice."""
import math


def iteration_points(points, dimensions, added_dimensions=2):
    """Scale a d-dimensional grid linearly when eccentric axes are added."""
    if dimensions <= 0 or added_dimensions < 0 or points <= 0:
        raise ValueError("Require positive points/dimensions and nonnegative added dimensions")
    return int(math.ceil(points * (dimensions + added_dimensions) / dimensions))


def reflect_into(values, lower, upper):
    """Reflect any number of excursions into a finite closed interval."""
    import numpy as np
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("Reflection requires finite lower < upper")
    width = upper - lower
    folded = np.mod(values - lower, 2 * width)
    return lower + width - np.abs(folded - width)
