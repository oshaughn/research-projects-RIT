#
# Author: D. Wysocki
import six
import math
import numpy as np
try:
  import cupy
  xpy_default=cupy
  cupy_here=True
except:
  print ' no cupy'
  cupy_here=False
  import numpy as cupy
  xpy_default=np


_one_by_eight = 0.125
_one_by_sixteen = 0.0625
_one_by_thirtytwo = 0.03125
_one_by_sixtyfour = 0.015625
_one_by_onetwentyeight = 0.0078125
_one_by_twofiftysix = 0.00390625
_one_by_fivetwelve = 0.001953125

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
}

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


def SphericalHarmonicsVectorized(
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

    coeffs = _coeffs[xpy]
    imag = _imag[xpy]

    # Precompute x*theta for several values of x
    half_theta = 0.5 * theta
    two_theta = 2.0 * theta
    three_theta = 3.0 * theta
    four_theta = 4.0 * theta
    five_theta = 5.0 * theta

    # Precompute cos(theta) and sin(theta).
    cos_theta = xpy.cos(theta)
    sin_theta = xpy.sin(theta)

    # Precompute 1 +/- cos(theta).
    one_minus_cos_theta = 1.0 - cos_theta
    one_plus_cos_theta = 1.0 + cos_theta

    # Precompute cos(x*theta) and sin(x*theta) for some values of x
    cos_half_theta = xpy.cos(half_theta)
    sin_half_theta = xpy.sin(half_theta)

    cos_two_theta = xpy.cos(two_theta)
    sin_two_theta = xpy.sin(two_theta)

    cos_three_theta = xpy.cos(three_theta)
    sin_three_theta = xpy.sin(three_theta)

    cos_four_theta = xpy.cos(four_theta)
    sin_four_theta = xpy.sin(four_theta)

    cos_five_theta = xpy.cos(five_theta)
    sin_five_theta = xpy.sin(five_theta)

    # Precompute exp(I * m * phi)
    exp_i_m_phi = xpy.exp(imag * xpy.einsum("i,j", phi, m))

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
                    )*
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

        # Chosen l, m values either not implemented yet or unphysical.
        else:
            Ylm[...,i] = xpy.nan

    Ylm *= exp_i_m_phi

    return Ylm
