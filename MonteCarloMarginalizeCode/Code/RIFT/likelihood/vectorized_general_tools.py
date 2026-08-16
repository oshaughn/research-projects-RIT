# Dan Wysocki

import numpy as np
import numpy

# When True, use a summation order that does not depend on GPU thread
# scheduling, so that a seeded run is bit-reproducible.  Off by default; set by
# RIFT.integrators.seeding.seed_everything when the user asks for a seed, so
# that reproducibility costs nothing on unseeded production runs.  See
# _bincount_weighted below.
DETERMINISTIC_REDUCTIONS = False


def _bincount_weighted(indices, weights, n_bins, xpy):
    """Weighted bincount, optionally with a run-to-run reproducible sum order.

    cupy.bincount with weights accumulates through float atomicAdd, whose
    ordering is set by GPU thread scheduling and therefore varies between
    otherwise identical runs.  Measured on an RTX 2080 Ti, repeated calls on
    byte-identical inputs disagree at ~2e-15 relative.  That is negligible as
    an error, but it is not negligible as a *reproducibility* defect: this
    histogram becomes the adapted sampling CDF, so the perturbation is injected
    into every subsequent draw and a seeded GPU run cannot be reproduced bit
    for bit.

    The deterministic branch sorts by bin and takes differences of a prefix
    sum, so the summation order is fixed by the data rather than by the
    scheduler.  It costs ~1.2-1.5x the atomic version on calls that happen once
    per parameter per adaptation, i.e. far off the likelihood hot path.
    """
    if not DETERMINISTIC_REDUCTIONS:
        return xpy.bincount(indices, minlength=n_bins, weights=weights)

    order = xpy.argsort(indices)
    idx_sorted = indices[order]
    wts_sorted = weights[order]
    # Prefix sum with a leading zero, so bin b is csum[end_b] - csum[start_b].
    csum = xpy.concatenate(
        (xpy.zeros(1, dtype=wts_sorted.dtype), xpy.cumsum(wts_sorted))
    )
    edges = xpy.searchsorted(
        idx_sorted, xpy.arange(n_bins + 1, dtype=idx_sorted.dtype), side='left'
    )
    return csum[edges[1:]] - csum[edges[:-1]]


def histogram(samples, n_bins, xpy=numpy,weights=None):
    """
    samples : data between [0,1]
    n_bins:    number of bins of output
    weights:  weights in histogram
    """
    n_samples = samples.size

    # sometimes due to input conditioning issues (floats!) the samples may be very slightly out of range - negative or greater than 1! Prevent this
    blank_array = xpy.zeros((n_samples,))
    samples_conditioned = xpy.maximum(samples, blank_array)
#samples * xpy.heavyside(samples,1)   # zero out any samples which are <0
    blank_array += 1 - 1e-3/n_bins
    samples_conditioned = xpy.minimum(samples_conditioned, blank_array) # don't let any samples be larger than 1

    # Compute the histogram counts.
    indices = xpy.trunc(samples_conditioned * n_bins).astype(np.int32)
    if isinstance(weights,type(None)):
        wts  =xpy.broadcast_to(
            xpy.asarray([float(n_bins)/n_samples]),
            (n_samples,)
            )
    else:
        wts=weights
    # broadcast_to gives a read-only, zero-stride view; the deterministic path
    # reorders it, so hand it a real array.
    wts = xpy.ascontiguousarray(wts)
    histogram_counts = _bincount_weighted(indices, wts, n_bins, xpy)
    return histogram_counts[:n_bins]  # force target length, we should never have points in top bin if it occurs : scaled to [0,1)



def interp(x, xp, fp, left=None, right=None, period=None, xpy=numpy):
    """
    One-dimensional linear interpolation.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.

    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument
        `period` is not specified. Otherwise, `xp` is internally sorted after
        normalizing the periodic boundaries with ``xp = xp % period``.

    fp : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.

    left : optional float or complex corresponding to fp
        Value to return for `x < xp[0]`, default is `fp[0]`.

    right : optional float or complex corresponding to fp
        Value to return for `x > xp[-1]`, default is `fp[-1]`.

    period : None or float, optional
        A period for the x-coordinates. This parameter allows the proper
        interpolation of angular x-coordinates. Parameters `left` and `right`
        are ignored if `period` is specified.

        .. versionadded:: 1.10.0

    Returns
    -------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as `x`.

    Raises
    ------
    ValueError
        If `xp` and `fp` have different length
        If `xp` or `fp` are not 1-D sequences
        If `period == 0`

    Notes
    -----
    Does not check that the x-coordinate sequence `xp` is increasing.
    If `xp` is not increasing, the results are nonsense.
    A simple check for increasing is::

        np.all(np.diff(xp) > 0)
    """
    # TODO: Implement periodic interpolation if needed.
    if period is not None:
        raise NotImplementedError("Periodic interpolation not yet implemented.")

    # Check shapes.
    input_shape = xpy.shape(xp)
    output_shape = xpy.shape(x)

    # Validate shapes.
    if len(input_shape) != 1:
        raise ValueError("`xp` is not a 1-D sequence")
    if input_shape != xpy.shape(fp):
        raise ValueError("`xp` and `fp` have different lengths.")

    # Pull out number of samples, now that we know input is 1-D.
    n_samples, = input_shape

    # Fill in default values for left and right if not given
    if left is None:
        left = fp[0]
    if right is None:
        right = fp[-1]

    # Initialize output array.
    y = xpy.empty_like(x, dtype=fp.dtype)

    # Compute (f(x1)-f(x0)) / (x1-x0)
    slopes = xpy.diff(fp) / xpy.diff(xp)

    # Get indices of different regions.
    i_below = x <= xp[0]
    i_above = x >= xp[-1]
    i_inside = ~(i_below | i_above)

    # Process points beyond edges.
    y[i_below] = left
    y[i_above] = right

    # Process interior.  First get indices corresponding to bins.
    ## TODO: Vectorize.  Tricky without `np.searchsorted` or `np.digitize`.
    x_inside = x[i_inside]
    x_bin_indices = xpy.empty_like(x_inside, dtype=int)
    for i in numpy.ndindex(*x_inside.shape):
        x_bin_indices[i] = xpy.argmax(
            (xp[:-1] < x_inside[i]) & (x_inside[i] <= xp[1:])
        )

    y[i_inside] = (
        slopes[x_bin_indices]*(x_inside-xp[x_bin_indices]) + fp[x_bin_indices]
    )

    return y
