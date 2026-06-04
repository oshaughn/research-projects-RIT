"""
Spin (-2) weighted spherical harmonics in JAX.

This is a faithful re-expression of
``RIFT.likelihood.SphericalHarmonics_gpu.SphericalHarmonicsVectorized`` for
``l = 2, 3, 4`` (the orders used in production O4 analyses; higher l raises a
clear error and can be transcribed in the same pattern if needed).

The *numeric coefficients* are imported directly from the production module so
they can never drift from the reference; only the (concrete, trace-independent)
theta-shape polynomials are transcribed.  Because ``lm`` is known at trace time,
we loop over the modes in plain Python and ``jnp.stack`` the resulting columns
-- the JAX analogue of the original's ``Ylm[..., i] = ...`` item assignment.

The ``e^{i m phi}`` azimuthal factor is applied at the end, identically to the
original (the caller passes ``phi = -phiref``).
"""

import jax.numpy as jnp

# Pull the exact coefficient table from the production implementation so the
# two can never disagree numerically.
from RIFT.likelihood.SphericalHarmonics_gpu import _coeffs_np as _COEFFS

_SUPPORTED_LMAX = 4


def _theta_shape(l, m, t):
    """Return the (real) theta-dependent polynomial for mode (l, m).

    ``t`` is a dict of precomputed trig quantities (see below).  These match
    the expressions in SphericalHarmonics_gpu term-for-term.
    """
    cos = t["cos_theta"]
    sin = t["sin_theta"]
    omc = t["one_minus_cos_theta"]
    opc = t["one_plus_cos_theta"]
    if l == 2:
        if m == -2:
            return jnp.square(omc)
        if m == -1:
            return sin * omc
        if m == 0:
            return jnp.square(sin)
        if m == 1:
            return sin * opc
        if m == 2:
            return jnp.square(opc)
    ch = t.get("cos_half_theta")
    sh = t.get("sin_half_theta")
    s2 = t.get("sin_two_theta")
    s3 = t.get("sin_three_theta")
    if l == 3:
        if m == -3:
            return ch * jnp.power(sh, 5.0)
        if m == -2:
            return (2.0 + 3.0 * cos) * jnp.power(sh, 4.0)
        if m == -1:
            return sin + 4.0 * s2 - 3.0 * s3
        if m == 0:
            return cos * jnp.power(sin, 2.0)
        if m == 1:
            return sin - 4.0 * s2 - 3.0 * s3
        if m == 2:
            return jnp.power(ch, 4.0) * (-2.0 + 3.0 * cos)
        if m == 3:
            return jnp.power(ch, 5.0) * sh
    c2 = t.get("cos_two_theta")
    s4 = t.get("sin_four_theta")
    if l == 4:
        if m == -4:
            return jnp.square(ch) * jnp.power(sh, 6.0)
        if m == -3:
            return ch * (1.0 + 2.0 * cos) * jnp.power(sh, 5.0)
        if m == -2:
            return (9.0 + 14.0 * cos + 7.0 * c2) * jnp.power(sh, 4.0)
        if m == -1:
            return 3.0 * sin + 2.0 * s2 + 7.0 * s3 - 7.0 * s4
        if m == 0:
            return (5.0 + 7.0 * c2) * jnp.square(sin)
        if m == 1:
            return 3.0 * sin - 2.0 * s2 + 7.0 * s3 + 7.0 * s4
        if m == 2:
            return jnp.power(ch, 4.0) * (9.0 - 14.0 * cos + 7.0 * c2)
        if m == 3:
            return jnp.power(ch, 5.0) * (-1.0 + 2.0 * cos) * sh
        if m == 4:
            return jnp.power(ch, 6.0) * jnp.square(sh)
    raise NotImplementedError(
        "jax_ile spherical harmonics implemented for l<=%d; got (l,m)=(%d,%d)"
        % (_SUPPORTED_LMAX, l, m)
    )


def spherical_harmonics_vectorized(lm, theta, phi, l_max=_SUPPORTED_LMAX):
    """Compute -2 Y_lm(theta, phi).

    Parameters
    ----------
    lm : sequence of (l, m) int pairs, length K (concrete / static).
    theta : array_like, shape (S,)   (inclination).
    phi : array_like, shape (S,)     (the caller passes -phiref).
    l_max : highest l present (used to skip unneeded trig precompute).

    Returns
    -------
    array_like (complex128), shape (S, K)
        Axis 0 varies theta/phi, axis 1 varies (l, m) -- matching the
        production ``SphericalHarmonicsVectorized``.
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)
    if l_max > _SUPPORTED_LMAX:
        raise NotImplementedError(
            "jax_ile spherical harmonics implemented for l<=%d (requested l_max=%d)"
            % (_SUPPORTED_LMAX, l_max)
        )

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    trig = {
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "one_minus_cos_theta": 1.0 - cos_theta,
        "one_plus_cos_theta": 1.0 + cos_theta,
    }
    if l_max >= 3:
        half_theta = 0.5 * theta
        trig["cos_half_theta"] = jnp.cos(half_theta)
        trig["sin_half_theta"] = jnp.sin(half_theta)
        trig["sin_two_theta"] = jnp.sin(2.0 * theta)
        trig["sin_three_theta"] = jnp.sin(3.0 * theta)
    if l_max >= 4:
        trig["cos_two_theta"] = jnp.cos(2.0 * theta)
        trig["sin_four_theta"] = jnp.sin(4.0 * theta)

    cols = []
    m_vals = []
    for (l_i, m_i) in lm:
        l_i, m_i = int(l_i), int(m_i)
        coeff = _COEFFS[l_i][m_i]
        cols.append(coeff * _theta_shape(l_i, m_i, trig))
        m_vals.append(m_i)

    Ylm = jnp.stack(cols, axis=-1).astype(jnp.complex128)  # (S, K)
    m_arr = jnp.asarray(m_vals, dtype=jnp.float64)
    # exp(i m phi), broadcast over (S, K)
    phase = jnp.exp(1.0j * (phi[:, None] * m_arr[None, :]))
    return Ylm * phase
