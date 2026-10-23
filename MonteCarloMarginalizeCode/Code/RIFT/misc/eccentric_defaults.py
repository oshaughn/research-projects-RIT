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


def eccentricity_interval(configured='[0,0.9]', lower=None, upper=None, prior='uniform', logarithmic=False):
    """Resolve native bounds, preserving each endpoint without an override."""
    import ast
    bounds = ast.literal_eval(configured) if isinstance(configured, str) else configured
    if len(bounds) != 2:
        raise ValueError('Eccentricity requires two bounds')
    lo, hi = bounds
    lo = float(lo if lower is None else lower)
    hi = float(hi if upper is None else upper)
    if (prior == 'log_uniform' or logarithmic) and lo == 0:
        lo = .001  # same zero-floor correction as CIP
    if (prior == 'log_uniform' or logarithmic) and lo < 0:
        raise ValueError('Logarithmic eccentricity requires positive bounds')
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError('Eccentricity requires finite increasing bounds')
    return lo, hi


def override_puff_eccentricity(arguments, lower=None, upper=None, prior='uniform', logarithmic=False):
    """Merge explicit overrides with the helper's final eccentricity interval."""
    import shlex
    tokens = shlex.split(arguments)
    names = [tokens[i+1] for i, token in enumerate(tokens) if token == '--downselect-parameter']
    ranges = [tokens[i+1] for i, token in enumerate(tokens) if token == '--downselect-parameter-range']
    if len(names) != len(ranges):
        raise ValueError('Require one range per downselect parameter')
    indices = [i+1 for i, token in enumerate(tokens) if token == '--downselect-parameter-range']
    matching = [i for i, name in enumerate(names) if name == 'eccentricity']
    if not matching and lower is None and upper is None:
        return arguments
    configured = ranges[matching[-1]] if matching else '[0,0.9]'
    lo, hi = eccentricity_interval(configured, lower, upper, prior, logarithmic)
    resolved = '[{},{}]'.format(lo, hi)
    if matching:
        for index in matching:
            tokens[indices[index]] = resolved
        return shlex.join(tokens)
    return arguments + (' --downselect-parameter eccentricity'
                        ' --downselect-parameter-range ' + resolved + ' ')
