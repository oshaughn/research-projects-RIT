"""
Author: Daniel Wysocki (daniel.m.wysocki@gmail.com)
"""
from __future__ import print_function

import six
import math
import numpy as np

try:
    import cupy
    xpy_default = cupy
    cupy_here = True
    junk_to_check_installed = cupy.array(5)  # this will fail if GPU not installed correctly
except: # ImportError:
    print(" no cupy")
    cupy_here = False
    import numpy as cupy
    xpy_default = np


# Many coefficients include division by powers of two. Multiplication is
# cheaper, and reciprocals of powers of two can be written exactly, so we just
# multiply by inverses.  Reciprocal powers of two greater than eight are given
# names for clarity, as only 1/2 = 0.5 and 1/4 = 0.25 can be expected to be
# understood immediately.
_one_by_eight = 0.125
_one_by_sixteen = 0.0625
_one_by_thirtytwo = 0.03125
_one_by_sixtyfour = 0.015625
_one_by_onetwentyeight = 0.0078125
_one_by_twofiftysix = 0.00390625
_one_by_fivetwelve = 0.001953125
_one_by_fortyninetysix = 0.000244140625

# Coefficients that appear in the various spherical harmonics.
_coeffs_np = {
    2 : {
        -2: math.sqrt(5.0 / (64.0 * math.pi)),
        -1: math.sqrt(5.0 / (16.0 * math.pi)),
        0: math.sqrt(15.0 / (32.0 * math.pi)),
        +1: math.sqrt(5.0 / (16.0 * math.pi)),
        +2: math.sqrt(5.0 / (64.0 * math.pi)),
    },
    3 : {
        -3: math.sqrt(21.0 / (2.0 * math.pi)),
        -2: math.sqrt(7.0 / (4.0 * math.pi)),
        -1: math.sqrt(35.0 / (2.0 * math.pi)) * _one_by_thirtytwo,
        0: math.sqrt(105.0 / (2.0 * math.pi)) * 0.25,
        +1: -math.sqrt(35.0 / (2.0 * math.pi)) * _one_by_thirtytwo,
        +2: math.sqrt(7.0 / math.pi) * 0.5,
        +3: -math.sqrt(21.0 / (2.0 * math.pi)),
    },
    4: {
        -4: 3.0 * math.sqrt(7.0 / math.pi),
        -3: 3.0 * math.sqrt(7.0 / (2.0 * math.pi)),
        -2: 3.0 / math.sqrt(math.pi) * 0.25,
        -1: 3.0 / math.sqrt(2.0 * math.pi) * _one_by_thirtytwo,
        0: 3.0 * math.sqrt(5.0 / (2.0 * math.pi)) * _one_by_sixteen,
        +1: 3.0 / math.sqrt(2.0*math.pi) * _one_by_thirtytwo,
        +2: 3.0 / math.sqrt(math.pi) * 0.25,
        +3: -3.0 * math.sqrt(7.0 / (2.0 * math.pi)),
        +4: 3.0 * math.sqrt(7.0 / math.pi),
    },
    5: {
        -5: math.sqrt(330.0 / math.pi),
        -4: math.sqrt(33.0 / math.pi),
        -3: math.sqrt(33.0 / (2.0 * math.pi)) * 0.25,
        -2: math.sqrt(11.0 / math.pi) * _one_by_eight,
        -1: math.sqrt(77.0 / math.pi) * _one_by_twofiftysix,
        0: math.sqrt(1155.0 / (2.0 * math.pi)) * _one_by_thirtytwo,
        +1: math.sqrt(77.0 / math.pi) * _one_by_twofiftysix,
        +2: math.sqrt(11.0 / math.pi) * _one_by_eight,
        +3: -math.sqrt(33.0 / (2.0 * math.pi)) * 0.25,
        +4: math.sqrt(33.0 / math.pi),
        +5: -math.sqrt(330.0 / math.pi),
    },
    6: {
        -6: 3.0 * math.sqrt(715.0 / math.pi) * 0.5,
        -5: math.sqrt(2145.0 / math.pi) * 0.5,
        -4: math.sqrt(195.0 / (2.0 * math.pi)) * _one_by_eight,
        -3: 3.0 * math.sqrt(13.0 / math.pi) * _one_by_thirtytwo,
        -2: math.sqrt(13.0 / math.pi) * _one_by_twofiftysix,
        -1: math.sqrt(65.0 / (2.0 * math.pi)) * _one_by_sixtyfour,
        0: math.sqrt(1365.0 / math.pi) * _one_by_fivetwelve,
        +1: math.sqrt(65.0 / (2.0 * math.pi)) * _one_by_sixtyfour,
        +2: math.sqrt(13.0 / math.pi) * _one_by_twofiftysix,
        +3: -3.0 * math.sqrt(13.0 / math.pi) * _one_by_thirtytwo,
        +4: math.sqrt(195.0 / (2.0 * math.pi)) * _one_by_eight,
        +5: -math.sqrt(2145.0 / math.pi) * 0.5,
        +6: 3.0 * math.sqrt(715.0 / math.pi) * 0.5,
    },
    7: {
        -7: math.sqrt(15015.0 / (2.0 * math.pi)),
        -6: math.sqrt(2145.0 / math.pi) * 0.5,
        -5: math.sqrt(165.0 / (2.0 * math.pi)) * _one_by_eight,
        -4: math.sqrt(165.0 / (2.0 * math.pi)) * _one_by_sixteen,
        -3: math.sqrt(15.0 / (2.0 * math.pi)) * _one_by_onetwentyeight,
        -2: math.sqrt(15.0 / math.pi) * _one_by_fivetwelve,
        -1: 3.0 * math.sqrt(5.0 / (2.0 * math.pi)) * _one_by_fivetwelve,
        0: 3.0 * math.sqrt(35.0 / math.pi) * _one_by_fivetwelve,
        +1: 3.0 * math.sqrt(5.0 / (2.0 * math.pi)) * _one_by_fivetwelve,
        +2: math.sqrt(15.0 / math.pi) * _one_by_fivetwelve,
        +3: -math.sqrt(15.0 / (2.0 * math.pi)) * _one_by_onetwentyeight,
        +4: math.sqrt(165.0 / (2.0 * math.pi)) * _one_by_sixteen,
        +5: -math.sqrt(165.0 / (2.0 * math.pi)) * _one_by_eight,
        +6: math.sqrt(2145.0 / math.pi) * 0.5,
        +7: -math.sqrt(15015.0 / (2.0 * math.pi)),
    },
    8: {
        -8: math.sqrt(34034.0 / math.pi),
        -7: math.sqrt(17017.0 / (2.0 * math.pi)),
        -6: math.sqrt(255255.0 / math.pi),
        -5: math.sqrt(12155.0 / (2.0 * math.pi)) * _one_by_eight,
        -4: math.sqrt(935.0 / (2.0 * math.pi)) * _one_by_thirtytwo,
        -3: math.sqrt(561.0 / (2.0 * math.pi)) * _one_by_onetwentyeight,
        -2: math.sqrt(17.0 / math.pi) * _one_by_fivetwelve,
        -1: math.sqrt(595.0 / (2.0 * math.pi)) * _one_by_fivetwelve,
        0: 3.0 * math.sqrt(595.0 / math.pi) * _one_by_fortyninetysix,
        +1: math.sqrt(595.0 / (2.0 * math.pi)) * _one_by_fivetwelve,
        +2: math.sqrt(17.0 / math.pi) * _one_by_fivetwelve,
        +3: -math.sqrt(561.0 / (2.0 * math.pi)) * _one_by_onetwentyeight,
        +4: math.sqrt(935.0 / (2.0 * math.pi)) * _one_by_thirtytwo,
        +5: -math.sqrt(12155.0 / (2.0 * math.pi)) * _one_by_eight,
        +6: math.sqrt(255255.0 / math.pi),
        +7: -math.sqrt(17017.0 / (2.0 * math.pi)),
        +8: math.sqrt(34034.0 / math.pi),
    },
}

# If cupy is loaded, put the coefficients into GPU memory.
# `_coeffs` dictionary will have entries for both numpy and cupy, depending on
# which is used.
# Also make a dictionary with the imaginary number in both numpy and cupy
# formats.
if cupy_here:
    _coeffs_cp = {
        l : {
            m: cupy.asarray(coeff)
            for m, coeff in six.iteritems(m_coeffs)
        }
        for l, m_coeffs in six.iteritems(_coeffs_np)
    }

    _coeffs = {
        np: _coeffs_np,
        cupy: _coeffs_cp,
    }
    _imag = {
        np: 1.0j,
        cupy: cupy.asarray(1.0j),
    }

# If cupy is not loaded, only entry will be for numpy.
else:
    _coeffs = {
        np: _coeffs_np,
    }
    _imag = {
        np: 1.0j,
    }



def SphericalHarmonicsVectorized(
        lm,
        theta, phi,
        xpy=xpy_default, dtype=np.complex128,
        l_max=8,
    ):
    """
    Compute spherical harmonics Y_{lm}(theta, phi) for a given set of lm's,
    thetas, and phis.

    Parameters
    ----------
    lm : array_like, shape = (n_indices, 2)
      Array of pairs of `l, m` values to evaluate harmonics for.
    theta : array_like, shape = (n_params,)
      Array of polar angles to evaluate harmonics at.
    phi : array_like, shape = (n_params,)
      Array of azimuth angles to evaluate harmonics at.
    xpy : numpy or cupy (default is cupy if loaded, else numpy)
      Numpy implementation to use.
    dtype : numpy.dtype (default is numpy.complex128)
      Datatype to use for output.  Must be complex.
    l_max : int (default is 8)
      Highest value of `l` that is allowed to appear in `lm`.  Will cause an
      exception if set lower than what's actually used.  If unsure, simply let
      it use the default, which uses the highest order the current
      implementation supports.

    Returns
    -------
    Ylm : array_like, shape = (n_params, n_indices)
      Array of spherical harmonics.  First axis varies `theta, phi`, and second
      axis varies `l, m`.
    """
    l, m = lm.T

    n_indices = l.size
    n_params = theta.size

    Ylm = xpy.empty((n_params, n_indices), dtype=dtype)

    coeffs = _coeffs[xpy]
    imag = _imag[xpy]

    ## Precompute commonly used functions of theta.
    # Values needed for all `l_max` values.
    cos_theta = xpy.cos(theta)
    sin_theta = xpy.sin(theta)

    one_minus_cos_theta = 1.0 - cos_theta
    one_plus_cos_theta = 1.0 + cos_theta

    # Values needed for `l_max` up to 3.
    if l_max >= 3:
        half_theta = 0.5 * theta
        two_theta = 2.0 * theta
        three_theta = 3.0 * theta

        cos_half_theta = xpy.cos(half_theta)
        sin_half_theta = xpy.sin(half_theta)

        sin_two_theta = xpy.sin(two_theta)
        sin_three_theta = xpy.sin(three_theta)

    # Values needed for `l_max` up to 4.
    if l_max >= 4:
        four_theta = 4.0 * theta

        cos_two_theta = xpy.cos(two_theta)
        sin_four_theta = xpy.sin(four_theta)

    # Values needed for `l_max` up to 5.
    if l_max >= 5:
        five_theta = 5.0 * theta

        cos_three_theta = xpy.cos(three_theta)
        sin_five_theta = xpy.sin(five_theta)

    # Values needed for `l_max` up to 6.
    if l_max >= 6:
        cos_four_theta = xpy.cos(four_theta)

    # Values needed for `l_max` up to 7.
    if l_max >= 7:
        cos_five_theta = xpy.cos(five_theta)

    # Values needed for `l_max` up to 8.
    if l_max >= 8:
        six_theta = 6.0 * theta

        cos_six_theta = xpy.cos(six_theta)


    for i, (l_i, m_i) in enumerate(lm):
        l_i, m_i = int(l_i), int(m_i)
        coeff = coeffs.get(l_i, {}).get(m_i, None)
        if l_i == 2:
            if m_i == -2:
                Ylm[...,i] = (
                    coeff *
                    xpy.square(one_minus_cos_theta)
                )
            elif m_i == -1:
                Ylm[...,i] = (
                    coeff *
                    sin_theta * one_minus_cos_theta
                )
            elif m_i == 0:
                Ylm[...,i] = (
                    coeff *
                    xpy.square(sin_theta)
                )
            elif m_i == +1:
                Ylm[...,i] = (
                    coeff *
                    sin_theta * one_plus_cos_theta
                )
            elif m_i == +2:
                Ylm[...,i] = (
                    coeff *
                    xpy.square(one_plus_cos_theta)
                )
        elif l_i == 3:
            if m_i == -3:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta * xpy.power(sin_half_theta, 5.0)
                )
            elif m_i == -2:
                Ylm[...,i] = (
                    coeff *
                    (2.0 + 3.0*cos_theta) * xpy.power(sin_half_theta, 4.0)
                )
            elif m_i == -1:
                Ylm[...,i] = (
                    coeff *
                    (sin_theta + 4.0*sin_two_theta - 3.0*sin_three_theta)
                )
            elif m_i == 0:
                Ylm[...,i] = (
                    coeff *
                    cos_theta * xpy.power(sin_theta, 2.0)
                )
            elif m_i == +1:
                Ylm[...,i] = (
                    coeff *
                    (sin_theta - 4.0*sin_two_theta - 3.0*sin_three_theta)
                )
            elif m_i == +2:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    (-2.0 + 3.0*cos_theta)
                )
            elif m_i == +3:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 5.0) *
                    sin_half_theta
                )
        elif l_i == 4:
            if m_i == -4:
                Ylm[...,i] = (
                    coeff *
                    xpy.square(cos_half_theta) *
                    xpy.power(sin_half_theta, 6.0)
                )
            elif m_i == -3:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta *
                    (1.0 + 2.0*cos_theta) *
                    xpy.power(sin_half_theta, 5.0)
                )
            elif m_i == -2:
                Ylm[...,i] = (
                    coeff *
                    (9.0 + 14.0*cos_theta + 7.0*cos_two_theta) *
                    xpy.power(sin_half_theta, 4.0)
                )
            elif m_i == -1:
                Ylm[...,i] = (
                    coeff *
                    (
                        3.0*sin_theta +
                        2.0*sin_two_theta +
                        7.0*sin_three_theta -
                        7.0*sin_four_theta
                    )
                )
            elif m_i == 0:
                Ylm[...,i] = (
                    coeff *
                    (5.0 + 7.0*cos_two_theta) *
                    xpy.square(sin_theta)
                )
            elif m_i == +1:
                Ylm[...,i] = (
                    coeff *
                    (
                        3.0*sin_theta -
                        2.0*sin_two_theta +
                        7.0*sin_three_theta +
                        7.0*sin_four_theta
                    )
                )
            elif m_i == +2:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    (9.0 - 14.0*cos_theta + 7.0*cos_two_theta)
                )
            elif m_i == +3:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 5.0) *
                    (-1.0 + 2.0*cos_theta) *
                    sin_half_theta
                )
            elif m_i == +4:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 6.0) *
                    xpy.square(sin_half_theta)
                )
        elif l_i == 5:
            if m_i == -5:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 3.0) *
                    xpy.power(sin_half_theta, 7.0)
                )
            elif m_i == -4:
                Ylm[...,i] = (
                    coeff *
                    xpy.square(cos_half_theta) *
                    (2.0 + 5.0*cos_theta) *
                    xpy.power(sin_half_theta, 6.0)
                )
            elif m_i == -3:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta *
                    (17.0 + 24.0*cos_theta + 15.0*cos_two_theta) *
                    xpy.power(sin_half_theta, 5.0)
                )
            elif m_i == -2:
                Ylm[...,i] = (
                    coeff *
                    (
                        32.0 +
                        57.0*cos_theta +
                        36.0*cos_two_theta +
                        15.0*cos_three_theta
                    ) *
                    xpy.power(sin_half_theta, 4.0)
                )
            elif m_i == -1:
                Ylm[...,i] = (
                    coeff *
                    (
                        2.0*sin_theta +
                        8.0*sin_two_theta +
                        3.0*sin_three_theta +
                        12.0*sin_four_theta -
                        15.0*sin_five_theta
                    )
                )
            elif m_i == 0:
                Ylm[...,i] = (
                    coeff *
                    (5.0*cos_theta + 3.0*cos_three_theta) *
                    xpy.square(sin_theta)
                )
            elif m_i == +1:
                Ylm[...,i] = (
                    coeff *
                    (
                        -2.0*sin_theta +
                        8.0*sin_two_theta -
                        3.0*sin_three_theta +
                        12.0*sin_four_theta +
                        15.0*sin_five_theta
                    )
                )
            elif m_i == +2:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    (
                        -32.0 +
                        57.0*cos_theta -
                        36.0*cos_two_theta +
                        15.0*cos_three_theta
                    )
                )
            elif m_i == +3:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 5.0) *
                    (17.0 - 24.0*cos_theta + 15.0*cos_two_theta) *
                    sin_half_theta
                )
            elif m_i == +4:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 6.0) *
                    (-2.0 + 5.0*cos_theta) *
                    xpy.square(sin_half_theta)
                )
            elif m_i == +5:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 7.0) *
                    xpy.power(sin_half_theta, 3.0)
                )
        elif l_i == 6:
            if m_i == -6:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    xpy.power(sin_half_theta, 8.0)
                )
            elif m_i == -5:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 3.0) *
                    (1.0 + 3.0*cos_theta) *
                    xpy.power(sin_half_theta, 7.0)
                )
            elif m_i == -4:
                Ylm[...,i] = (
                    coeff *
                    xpy.square(cos_half_theta) *
                    (35.0 + 44.0*cos_theta + 33.0*cos_two_theta) *
                    xpy.power(sin_half_theta, 6.0)
                )
            elif m_i == -3:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta *
                    (
                        98.0 +
                        185.0*cos_theta +
                        110.0*cos_two_theta +
                        55.0*cos_three_theta
                    ) *
                    xpy.power(sin_half_theta, 5.0)
                )
            elif m_i == -2:
                Ylm[...,i] = (
                    coeff *
                    (
                        1709.0 +
                        3096.0*cos_theta +
                        2340.0*cos_two_theta +
                        1320.0*cos_three_theta +
                        495.0*cos_four_theta
                    ) *
                    xpy.power(sin_half_theta, 4.0)
                )
            elif m_i == -1:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta *
                    (
                        161.0 +
                        252.0*cos_theta +
                        252.0*cos_two_theta +
                        132.0*cos_three_theta +
                        99.0*cos_four_theta
                    ) *
                    xpy.power(sin_half_theta, 3.0)
                )
            elif m_i == 0:
                Ylm[...,i] = (
                    coeff *
                    (35.0 + 60.0*cos_two_theta + 33.0*cos_four_theta) *
                    xpy.square(sin_theta)
                )
            elif m_i == +1:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 3.0) *
                    (
                        161.0 -
                        252.0*cos_theta +
                        252.0*cos_two_theta -
                        132.0*cos_three_theta +
                        99.0*cos_four_theta
                    ) *
                    sin_half_theta
                )
            elif m_i == +2:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    (
                        1709.0 -
                        3096.0*cos_theta +
                        2340.0*cos_two_theta -
                        1320.0*cos_three_theta +
                        495.0*cos_four_theta
                    )
                )
            elif m_i == +3:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 5.0) *
                    (
                        -98.0 +
                        185.0*cos_theta -
                        110.0*cos_two_theta +
                        55.0*cos_three_theta
                    ) *
                    sin_half_theta
                )
            elif m_i == +4:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 6.0) *
                    (35.0 - 44.0*cos_theta + 33.0*cos_two_theta) *
                    xpy.square(sin_half_theta)
                )
            elif m_i == +5:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 7.0) *
                    (-1.0 + 3.0*cos_theta) *
                    xpy.power(sin_half_theta, 3.0)
                )
            elif m_i == +6:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 8.0) *
                    xpy.power(sin_half_theta, 4.0)
                )
        elif l_i == 7:
            if m_i == -7:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 5.0) *
                    xpy.power(sin_half_theta, 9.0)
                )
            elif m_i == -6:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    (2.0 + 7.0*cos_theta) *
                    xpy.power(sin_half_theta, 8.0)
                )
            elif m_i == -5:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 3.0) *
                    (93.0 + 104.0*cos_theta + 91.0*cos_two_theta) *
                    xpy.power(sin_half_theta, 7.0)
                )
            elif m_i == -4:
                Ylm[...,i] = (
                    coeff *
                    xpy.square(cos_half_theta) *
                    (
                        140.0 +
                        285.0*cos_theta +
                        156.0*cos_two_theta +
                        91.0*cos_three_theta
                    ) *
                    xpy.power(sin_half_theta, 6.0)
                )
            elif m_i == -3:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta *
                    (
                        3115.0 +
                        5456.0*cos_theta +
                        4268.0*cos_two_theta +
                        2288.0*cos_three_theta +
                        1001.0*cos_four_theta
                    ) *
                    xpy.power(sin_half_theta, 5.0)
                )
            elif m_i == -2:
                Ylm[...,i] = (
                    coeff *
                    (
                        5220.0 +
                        9810.0*cos_theta +
                        7920.0*cos_two_theta +
                        5445.0*cos_three_theta +
                        2860.0*cos_four_theta +
                        1001.0*cos_five_theta
                    ) *
                    xpy.power(sin_half_theta, 4.0)
                )
            elif m_i == -1:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta *
                    (
                        1890.0 +
                        4130.0*cos_theta +
                        3080.0*cos_two_theta +
                        2805.0*cos_three_theta +
                        1430.0*cos_four_theta +
                        1001.0*cos_five_theta
                    ) *
                    xpy.power(sin_half_theta, 3.0)
                )
            elif m_i == 0:
                Ylm[...,i] = (
                    coeff *
                    cos_theta *
                    (109.0 + 132.0*cos_two_theta + 143.0*cos_four_theta) *
                    xpy.square(sin_theta)
                )
            elif m_i == +1:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 3.0) *
                    (
                        -1890.0 +
                        4130.0*cos_theta -
                        3080.0*cos_two_theta +
                        2805.0*cos_three_theta -
                        1430.0*cos_four_theta +
                        1001.0*cos_five_theta
                    ) *
                    sin_half_theta
                )
            elif m_i == +2:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    (
                        -5220.0 +
                        9810.0*cos_theta -
                        7920.0*cos_two_theta +
                        5445.0*cos_three_theta -
                        2860.0*cos_four_theta +
                        1001.0*cos_five_theta
                    )
                )
            elif m_i == +3:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 5.0) *
                    (
                        3115.0 -
                        5456.0*cos_theta +
                        4268.0*cos_two_theta -
                        2288.0*cos_three_theta +
                        1001.0*cos_four_theta
                    ) *
                    sin_half_theta
                )
            elif m_i == +4:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 6.0) *
                    (
                        -140.0 +
                        285.0*cos_theta -
                        156.0*cos_two_theta +
                        91.0*cos_three_theta
                    ) *
                    xpy.square(sin_half_theta)
                )
            elif m_i == +5:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 7.0) *
                    (93.0 - 104.0*cos_theta + 91.0*cos_two_theta) *
                    xpy.power(sin_half_theta, 3.0)
                )
            elif m_i == +6:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 8.0) *
                    (-2.0 + 7.0*cos_theta) *
                    xpy.power(sin_half_theta, 4.0)
                )
            elif m_i == +7:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 9.0) *
                    xpy.power(sin_half_theta, 5.0)
                )
        elif l_i == 8:
            if m_i == -8:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 6.0) *
                    xpy.power(sin_half_theta, 10.0)
                )
            elif m_i == -7:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 5.0) *
                    (1.0 + 4.0*cos_theta) *
                    xpy.power(sin_half_theta, 9.0)
                )
            elif m_i == -6:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    (1.0 + 2.0*cos_theta) *
                    xpy.sin(0.25*xpy.pi - half_theta) *
                    xpy.sin(0.25*xpy.pi + half_theta) *
                    xpy.power(sin_half_theta, 8.0)
                )
            elif m_i == -5:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 3.0) *
                    (
                        19.0 +
                        42.0*cos_theta +
                        21.0*cos_two_theta +
                        14.0*cos_three_theta
                    ) *
                    xpy.power(sin_half_theta, 7.0)
                )
            elif m_i == -4:
                Ylm[...,i] = (
                    coeff *
                    xpy.square(cos_half_theta) *
                    (
                        265.0 +
                        442.0*cos_theta +
                        364.0*cos_two_theta +
                        182.0*cos_three_theta +
                        91.0*cos_four_theta
                    ) *
                    xpy.power(sin_half_theta, 6.0)
                )
            elif m_i == -3:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta *
                    (
                        869.0 +
                        1660.0*cos_theta +
                        1300.0*cos_two_theta +
                        910.0*cos_three_theta +
                        455.0*cos_four_theta +
                        182.0*cos_five_theta
                    ) *
                    xpy.power(sin_half_theta, 5.0)
                )
            elif m_i == -2:
                Ylm[...,i] = (
                    coeff *
                    (
                        7626.0 +
                        14454.0*cos_theta +
                        12375.0*cos_two_theta +
                        9295.0*cos_three_theta +
                        6006.0*cos_four_theta +
                        3003.0*cos_five_theta +
                        1001.0*cos_six_theta
                    ) *
                    xpy.power(sin_half_theta, 4.0)
                )
            elif m_i == -1:
                Ylm[...,i] = (
                    coeff *
                    cos_half_theta *
                    (
                        798.0 +
                        1386.0*cos_theta +
                        1386.0*cos_two_theta +
                        1001.0*cos_three_theta +
                        858.0*cos_four_theta +
                        429.0*cos_five_theta +
                        286.0*cos_six_theta
                    ) *
                    xpy.power(sin_half_theta, 3.0)
                )
            elif m_i == 0:
                Ylm[...,i] = (
                    coeff *
                    (
                        210.0 +
                        385.0*cos_two_theta +
                        286.0*cos_four_theta +
                        143.0*cos_six_theta
                    ) *
                    xpy.square(sin_theta)
                )
            elif m_i == +1:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 3.0) *
                    (
                        798.0 -
                        1386.0*cos_theta +
                        1386.0*cos_two_theta -
                        1001.0*cos_three_theta +
                        858.0*cos_four_theta -
                        429.0*cos_five_theta +
                        286.0*cos_six_theta
                    ) *
                    sin_half_theta
                )
            elif m_i == +2:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 4.0) *
                    (
                        7626.0 -
                        14454.0*cos_theta +
                        12375.0*cos_two_theta -
                        9295.0*cos_three_theta +
                        6006.0*cos_four_theta -
                        3003.0*cos_five_theta +
                        1001.0*cos_six_theta
                    )
                )
            elif m_i == +3:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 5.0) *
                    (
                        -869.0 +
                        1660.0*cos_theta -
                        1300.0*cos_two_theta +
                        910.0*cos_three_theta -
                        455.0*cos_four_theta +
                        182.0*cos_five_theta
                    ) *
                    sin_half_theta
                )
            elif m_i == +4:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 6.0) *
                    (
                        265.0 -
                        442.0*cos_theta +
                        364.0*cos_two_theta -
                        182.0*cos_three_theta +
                        91.0*cos_four_theta
                    ) *
                    xpy.square(sin_half_theta)
                )
            elif m_i == +5:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 7.0) *
                    (
                        -19.0 +
                        42.0*cos_theta -
                        21.0*cos_two_theta +
                        14.0*cos_three_theta
                    ) *
                    xpy.power(sin_half_theta, 3.0)
                )
            elif m_i == +6:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 8.0) *
                    (-1.0 + 2.0*cos_theta) *
                    xpy.sin(0.25*xpy.pi - half_theta) *
                    xpy.sin(0.25*xpy.pi + half_theta) *
                    xpy.power(sin_half_theta, 4.0)
                )
            elif m_i == +7:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 9.0) *
                    (-1.0 + 4.0*cos_theta) *
                    xpy.power(sin_half_theta, 5.0)
                )
            elif m_i == +8:
                Ylm[...,i] = (
                    coeff *
                    xpy.power(cos_half_theta, 10.0) *
                    xpy.power(sin_half_theta, 6.0)
                )


        # Chosen l, m values either not implemented yet or unphysical.
        else:
            Ylm[...,i] = xpy.nan
    # Multiply everything by exp(i * m * phi)
    Ylm *= xpy.exp(imag * xpy.einsum("i,j", phi, m))

    return Ylm
