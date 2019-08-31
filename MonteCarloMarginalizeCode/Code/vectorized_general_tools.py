# Dan Wysocki

import numpy as np
import numpy

def histogram(samples, n_bins, xpy=numpy):
    n_samples = samples.size

    # Compute the histogram counts.
    indices = xpy.trunc(samples * n_bins).astype(np.int32)
    histogram_counts = xpy.bincount(
        indices, minlength=n_bins,
        weights=xpy.broadcast_to(
            xpy.asarray([float(n_bins)/n_samples]),
            (n_samples,),
        ),
    )
    return histogram_counts



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
