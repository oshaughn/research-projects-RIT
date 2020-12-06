# Leo Fang
# https://github.com/leofang/cupy/blob/interp/cupy/_math/misc.py
# Porting here temporarily for testing in our end-to-end MC integration code for robustness

import cupy

@cupy._util.memoize(for_each_device=True)
def _get_interp_kernel():
    # TODO(leofang): check NaN; for complex numbers it seems a bit involved
    # TODO(leofang): better memory access pattern?
    # TODO(leofang): investigate if we could use texture to accelerate
    in_params = 'raw X x, raw U idx, '
    in_params += 'raw X fx, raw Y fy, U len, raw X left, raw X right'
    out_params = 'Y y'
    code = r'''
        U x_idx = idx[i] - 1;
        if (x_idx < 0) { y = left[0]; }
        else if (x_idx >= len - 1) { y = right[0]; }
        else {
            y = (fy[x_idx+1] - fy[x_idx]) / (fx[x_idx+1] - fx[x_idx]) \
                * (x[i] - fx[x_idx]) + fy[x_idx];
        }
    '''
    return cupy.ElementwiseKernel(in_params, out_params, code, 'cupy_interp')


def interp(x, xp, fp, left=None, right=None, period=None):
    """ One-dimensional linear interpolation.
    Args:
        x (cupy.ndarray): a 1-dimensional input on which the interpolation is
            performed.
        xp (cupy.ndarray): a 1-dimensional input on which the function values
            (``fp``) are known.
        fp (cupy.ndarray): a 1-dimensional input containing the function values
            corresponding to the ``xp`` points.
        left (float or complex): value to return if ``x < xp[0]``. Default is
            ``fp[0]``.
        right (float or complex): value to return if ``x > xp[-1]``. Default is
            ``fp[-1]``.
        period (optional): refer to the Numpy documentation for detail.
    Returns:
        cupy.ndarray: The one-dimensional piecewise linear interpolant to a
            function with given discrete data points (``xp``, ``fp``),
            evaluated at ``x``.
    .. note::
        This function may synchronize if ``left`` or ``right`` is not on the
        device already.
    .. seealso:: :func:`numpy.interp`
    """

    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError('xp and fp must be 1D arrays')
    if xp.size != fp.size:
        raise ValueError('fp and xp are not of the same length')
    if xp.size == 0:
        raise ValueError('array of sample points is empty')
    if not x.flags.c_contiguous:
        raise NotImplementedError('Non-C-contiguous x is currently not '
                                  'supported')
    if period is not None:
        # The handling of "period" below is borrowed from NumPy

        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None

        x = x.astype(cupy.float64)
        xp = xp.asdtype(cupy.float64)

        # normalizing periodic boundaries
        x %= period
        xp %= period
        asort_xp = cupy.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = cupy.concatenate((xp[-1:]-period, xp, xp[0:1]+period))
        fp = cupy.concatenate((fp[-1:], fp, fp[0:1]))
        assert xp.flags.c_contiguous
        assert fp.flags.c_contiguous

    output = cupy.empty(x.shape, dtype=fp.dtype)
    idx = cupy.searchsorted(xp, x, side='right')
    left = fp[0] if left is None else cupy.array(left, xp.dtype)
    right = fp[-1] if right is None else cupy.array(right, xp.dtype)
    kern = _get_interp_kernel()
    kern(x, idx, xp, fp, xp.size, left, right, output)
    return output
