import math
import numpy as np
try:
  import cupy
  import optimized_gpu_tools
  import Q_inner_product
  import SphericalHarmonics_gpu
  xpy_default=cupy
except:
  print(' no cupy')
  import numpy as cupy
  optimized_gpu_tools=None
  Q_inner_product=None
  xpy_default=np


_coeff2m2 = math.sqrt(5.0 / (64.0 * math.pi))
_coeff2m1 = math.sqrt(5.0 / (16.0 * math.pi))
_coeff20 = math.sqrt(15.0 / (32.0 * math.pi))
_coeff2p1 = math.sqrt(5.0 / (16.0 * math.pi))
_coeff2p2 = math.sqrt(5.0 / (64.0 * math.pi))

def SphericalHarmonicsVectorized_orig(
        lm,
        theta, phi,
        xpy=cupy, dtype=np.complex128,
    ):
    """
    Compute spherical harmonics Y_{lm}(theta, phi) for a given set of lm's,
    thetas, and phis.

    Parameters
    ----------
    lm : array_like, shape = (n_indices, 2)

    theta : array_like, shape = (n_params,)

    phi : array_like, shape = (n_params,)

    Returns
    -------
    Ylm : array_like, shape = (n_params, n_indices)
    """
    l, m = lm.T

    n_indices = l.size
    n_params = theta.size

    Ylm = xpy.empty((n_params, n_indices), dtype=dtype)

    # Ensure coefficients are on the board if using cupy.
    c2m2 = xpy.asarray(_coeff2m2)
    c2m1 = xpy.asarray(_coeff2m1)
    c20 = xpy.asarray(_coeff20)
    c2p1 = xpy.asarray(_coeff2p1)
    c2p2 = xpy.asarray(_coeff2p2)
    imag = xpy.asarray(1.0j)

    # Precompute 1 +/- cos(theta).
    cos_theta = xpy.cos(theta)
    one_minus_cos_theta = (1.0 - cos_theta)
    one_plus_cos_theta = xpy.add(1.0, cos_theta, out=cos_theta)
    del cos_theta

    # Precompute sin(theta).
    sin_theta = xpy.sin(theta)

    for i, m_i in enumerate(m):
        if m_i == -2:
            Ylm[...,i] = (
                c2m2 *
                xpy.square(one_minus_cos_theta) *
                xpy.exp(imag * m_i * phi)
            )
        elif m_i == -1:
            Ylm[...,i] = (
                c2m1 *
                xpy.multiply(sin_theta, one_minus_cos_theta) *
                xpy.exp(imag * m_i * phi)
            )
        elif m_i == 0:
            Ylm[...,i] = (
                c20 *
                xpy.square(sin_theta)
            )
        elif m_i == +1:
            Ylm[...,i] = (
                c2p1 *
                xpy.multiply(sin_theta, one_plus_cos_theta) *
                xpy.exp(imag * m_i * phi)
            )
        elif m_i == +2:
            Ylm[...,i] = (
                c2p2 *
                xpy.square(one_plus_cos_theta) *
                xpy.exp(imag * m_i * phi)
            )
        else:
            Ylm[...,i] = xpy.nan

    return Ylm
