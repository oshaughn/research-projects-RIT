"""
Spin (-2) weighted spherical harmonics in JAX, l = 2 .. 8.

Faithful re-expression of
``RIFT.likelihood.SphericalHarmonics_gpu.SphericalHarmonicsVectorized`` for the
full ``l = 2 .. 8`` range used across RIFT analyses (incl. high-l NR
applications).  The *numeric coefficients* are imported directly from the
production module so they can never drift; only the (concrete,
trace-independent) theta-shape polynomials are transcribed here.

Because ``lm`` is known at trace time, we loop over the modes in plain Python
and ``jnp.stack`` the columns -- the JAX analogue of the original's
``Ylm[..., i] = ...`` item assignment.  The ``e^{i m phi}`` azimuthal factor is
applied at the end, identically to the original (the caller passes
``phi = -phiref``).

Run ``python spherical.py`` to validate every (l, m) for l<=8 against the
independent gold-standard ``lal.SpinWeightedSphericalHarmonic`` and against the
production numpy ``SphericalHarmonicsVectorized``.
"""

import jax.numpy as jnp

# Pull the exact coefficient table from the production implementation.
from RIFT.likelihood.SphericalHarmonics_gpu import _coeffs_np as _COEFFS

_SUPPORTED_LMAX = 8


def _theta_shape(l, m, t):
    """Real theta-dependent polynomial for mode (l, m).

    ``t`` is a dict of precomputed trig quantities (see
    :func:`spherical_harmonics_vectorized`).  Term-for-term identical to
    ``SphericalHarmonics_gpu``.
    """
    cos = t["cos_theta"]; sin = t["sin_theta"]
    omc = t["one_minus_cos_theta"]; opc = t["one_plus_cos_theta"]
    if l == 2:
        if m == -2: return jnp.square(omc)
        if m == -1: return sin * omc
        if m == 0:  return jnp.square(sin)
        if m == 1:  return sin * opc
        if m == 2:  return jnp.square(opc)
    ch = t.get("cos_half_theta"); sh = t.get("sin_half_theta")
    s2 = t.get("sin_two_theta"); s3 = t.get("sin_three_theta")
    if l == 3:
        if m == -3: return ch * jnp.power(sh, 5.0)
        if m == -2: return (2.0 + 3.0 * cos) * jnp.power(sh, 4.0)
        if m == -1: return sin + 4.0 * s2 - 3.0 * s3
        if m == 0:  return cos * jnp.power(sin, 2.0)
        if m == 1:  return sin - 4.0 * s2 - 3.0 * s3
        if m == 2:  return jnp.power(ch, 4.0) * (-2.0 + 3.0 * cos)
        if m == 3:  return jnp.power(ch, 5.0) * sh
    c2 = t.get("cos_two_theta"); s4 = t.get("sin_four_theta")
    if l == 4:
        if m == -4: return jnp.square(ch) * jnp.power(sh, 6.0)
        if m == -3: return ch * (1.0 + 2.0 * cos) * jnp.power(sh, 5.0)
        if m == -2: return (9.0 + 14.0 * cos + 7.0 * c2) * jnp.power(sh, 4.0)
        if m == -1: return 3.0 * sin + 2.0 * s2 + 7.0 * s3 - 7.0 * s4
        if m == 0:  return (5.0 + 7.0 * c2) * jnp.square(sin)
        if m == 1:  return 3.0 * sin - 2.0 * s2 + 7.0 * s3 + 7.0 * s4
        if m == 2:  return jnp.power(ch, 4.0) * (9.0 - 14.0 * cos + 7.0 * c2)
        if m == 3:  return jnp.power(ch, 5.0) * (-1.0 + 2.0 * cos) * sh
        if m == 4:  return jnp.power(ch, 6.0) * jnp.square(sh)
    c3 = t.get("cos_three_theta"); s5 = t.get("sin_five_theta")
    if l == 5:
        if m == -5: return jnp.power(ch, 3.0) * jnp.power(sh, 7.0)
        if m == -4: return jnp.square(ch) * (2.0 + 5.0 * cos) * jnp.power(sh, 6.0)
        if m == -3: return ch * (17.0 + 24.0 * cos + 15.0 * c2) * jnp.power(sh, 5.0)
        if m == -2: return (32.0 + 57.0 * cos + 36.0 * c2 + 15.0 * c3) * jnp.power(sh, 4.0)
        if m == -1: return 2.0 * sin + 8.0 * s2 + 3.0 * s3 + 12.0 * s4 - 15.0 * s5
        if m == 0:  return (5.0 * cos + 3.0 * c3) * jnp.square(sin)
        if m == 1:  return -2.0 * sin + 8.0 * s2 - 3.0 * s3 + 12.0 * s4 + 15.0 * s5
        if m == 2:  return jnp.power(ch, 4.0) * (-32.0 + 57.0 * cos - 36.0 * c2 + 15.0 * c3)
        if m == 3:  return jnp.power(ch, 5.0) * (17.0 - 24.0 * cos + 15.0 * c2) * sh
        if m == 4:  return jnp.power(ch, 6.0) * (-2.0 + 5.0 * cos) * jnp.square(sh)
        if m == 5:  return jnp.power(ch, 7.0) * jnp.power(sh, 3.0)
    c4 = t.get("cos_four_theta")
    if l == 6:
        if m == -6: return jnp.power(ch, 4.0) * jnp.power(sh, 8.0)
        if m == -5: return jnp.power(ch, 3.0) * (1.0 + 3.0 * cos) * jnp.power(sh, 7.0)
        if m == -4: return jnp.square(ch) * (35.0 + 44.0 * cos + 33.0 * c2) * jnp.power(sh, 6.0)
        if m == -3: return ch * (98.0 + 185.0 * cos + 110.0 * c2 + 55.0 * c3) * jnp.power(sh, 5.0)
        if m == -2: return (1709.0 + 3096.0 * cos + 2340.0 * c2 + 1320.0 * c3 + 495.0 * c4) * jnp.power(sh, 4.0)
        if m == -1: return ch * (161.0 + 252.0 * cos + 252.0 * c2 + 132.0 * c3 + 99.0 * c4) * jnp.power(sh, 3.0)
        if m == 0:  return (35.0 + 60.0 * c2 + 33.0 * c4) * jnp.square(sin)
        if m == 1:  return jnp.power(ch, 3.0) * (161.0 - 252.0 * cos + 252.0 * c2 - 132.0 * c3 + 99.0 * c4) * sh
        if m == 2:  return jnp.power(ch, 4.0) * (1709.0 - 3096.0 * cos + 2340.0 * c2 - 1320.0 * c3 + 495.0 * c4)
        if m == 3:  return jnp.power(ch, 5.0) * (-98.0 + 185.0 * cos - 110.0 * c2 + 55.0 * c3) * sh
        if m == 4:  return jnp.power(ch, 6.0) * (35.0 - 44.0 * cos + 33.0 * c2) * jnp.square(sh)
        if m == 5:  return jnp.power(ch, 7.0) * (-1.0 + 3.0 * cos) * jnp.power(sh, 3.0)
        if m == 6:  return jnp.power(ch, 8.0) * jnp.power(sh, 4.0)
    c5 = t.get("cos_five_theta")
    if l == 7:
        if m == -7: return jnp.power(ch, 5.0) * jnp.power(sh, 9.0)
        if m == -6: return jnp.power(ch, 4.0) * (2.0 + 7.0 * cos) * jnp.power(sh, 8.0)
        if m == -5: return jnp.power(ch, 3.0) * (93.0 + 104.0 * cos + 91.0 * c2) * jnp.power(sh, 7.0)
        if m == -4: return jnp.square(ch) * (140.0 + 285.0 * cos + 156.0 * c2 + 91.0 * c3) * jnp.power(sh, 6.0)
        if m == -3: return ch * (3115.0 + 5456.0 * cos + 4268.0 * c2 + 2288.0 * c3 + 1001.0 * c4) * jnp.power(sh, 5.0)
        if m == -2: return (5220.0 + 9810.0 * cos + 7920.0 * c2 + 5445.0 * c3 + 2860.0 * c4 + 1001.0 * c5) * jnp.power(sh, 4.0)
        if m == -1: return ch * (1890.0 + 4130.0 * cos + 3080.0 * c2 + 2805.0 * c3 + 1430.0 * c4 + 1001.0 * c5) * jnp.power(sh, 3.0)
        if m == 0:  return cos * (109.0 + 132.0 * c2 + 143.0 * c4) * jnp.square(sin)
        if m == 1:  return jnp.power(ch, 3.0) * (-1890.0 + 4130.0 * cos - 3080.0 * c2 + 2805.0 * c3 - 1430.0 * c4 + 1001.0 * c5) * sh
        if m == 2:  return jnp.power(ch, 4.0) * (-5220.0 + 9810.0 * cos - 7920.0 * c2 + 5445.0 * c3 - 2860.0 * c4 + 1001.0 * c5)
        if m == 3:  return jnp.power(ch, 5.0) * (3115.0 - 5456.0 * cos + 4268.0 * c2 - 2288.0 * c3 + 1001.0 * c4) * sh
        if m == 4:  return jnp.power(ch, 6.0) * (-140.0 + 285.0 * cos - 156.0 * c2 + 91.0 * c3) * jnp.square(sh)
        if m == 5:  return jnp.power(ch, 7.0) * (93.0 - 104.0 * cos + 91.0 * c2) * jnp.power(sh, 3.0)
        if m == 6:  return jnp.power(ch, 8.0) * (-2.0 + 7.0 * cos) * jnp.power(sh, 4.0)
        if m == 7:  return jnp.power(ch, 9.0) * jnp.power(sh, 5.0)
    c6 = t.get("cos_six_theta"); halfp = t.get("half_theta")
    if l == 8:
        # m=+-6 use the original's sin(pi/4 -/+ half_theta) factorization
        sp = jnp.sin(0.25 * jnp.pi - halfp); sm = jnp.sin(0.25 * jnp.pi + halfp)
        if m == -8: return jnp.power(ch, 6.0) * jnp.power(sh, 10.0)
        if m == -7: return jnp.power(ch, 5.0) * (1.0 + 4.0 * cos) * jnp.power(sh, 9.0)
        if m == -6: return jnp.power(ch, 4.0) * (1.0 + 2.0 * cos) * sp * sm * jnp.power(sh, 8.0)
        if m == -5: return jnp.power(ch, 3.0) * (19.0 + 42.0 * cos + 21.0 * c2 + 14.0 * c3) * jnp.power(sh, 7.0)
        if m == -4: return jnp.square(ch) * (265.0 + 442.0 * cos + 364.0 * c2 + 182.0 * c3 + 91.0 * c4) * jnp.power(sh, 6.0)
        if m == -3: return ch * (869.0 + 1660.0 * cos + 1300.0 * c2 + 910.0 * c3 + 455.0 * c4 + 182.0 * c5) * jnp.power(sh, 5.0)
        if m == -2: return (7626.0 + 14454.0 * cos + 12375.0 * c2 + 9295.0 * c3 + 6006.0 * c4 + 3003.0 * c5 + 1001.0 * c6) * jnp.power(sh, 4.0)
        if m == -1: return ch * (798.0 + 1386.0 * cos + 1386.0 * c2 + 1001.0 * c3 + 858.0 * c4 + 429.0 * c5 + 286.0 * c6) * jnp.power(sh, 3.0)
        if m == 0:  return (210.0 + 385.0 * c2 + 286.0 * c4 + 143.0 * c6) * jnp.square(sin)
        if m == 1:  return jnp.power(ch, 3.0) * (798.0 - 1386.0 * cos + 1386.0 * c2 - 1001.0 * c3 + 858.0 * c4 - 429.0 * c5 + 286.0 * c6) * sh
        if m == 2:  return jnp.power(ch, 4.0) * (7626.0 - 14454.0 * cos + 12375.0 * c2 - 9295.0 * c3 + 6006.0 * c4 - 3003.0 * c5 + 1001.0 * c6)
        if m == 3:  return jnp.power(ch, 5.0) * (-869.0 + 1660.0 * cos - 1300.0 * c2 + 910.0 * c3 - 455.0 * c4 + 182.0 * c5) * sh
        if m == 4:  return jnp.power(ch, 6.0) * (265.0 - 442.0 * cos + 364.0 * c2 - 182.0 * c3 + 91.0 * c4) * jnp.square(sh)
        if m == 5:  return jnp.power(ch, 7.0) * (-19.0 + 42.0 * cos - 21.0 * c2 + 14.0 * c3) * jnp.power(sh, 3.0)
        if m == 6:  return jnp.power(ch, 8.0) * (-1.0 + 2.0 * cos) * sp * sm * jnp.power(sh, 4.0)
        if m == 7:  return jnp.power(ch, 9.0) * (-1.0 + 4.0 * cos) * jnp.power(sh, 5.0)
        if m == 8:  return jnp.power(ch, 10.0) * jnp.power(sh, 6.0)
    raise NotImplementedError(
        "jax_ile spherical harmonics implemented for l<=%d; got (l,m)=(%d,%d)"
        % (_SUPPORTED_LMAX, l, m))


def _build_trig(theta, l_max):
    cos_theta = jnp.cos(theta); sin_theta = jnp.sin(theta)
    t = {
        "cos_theta": cos_theta, "sin_theta": sin_theta,
        "one_minus_cos_theta": 1.0 - cos_theta,
        "one_plus_cos_theta": 1.0 + cos_theta,
    }
    if l_max >= 3:
        half = 0.5 * theta
        t["half_theta"] = half
        t["cos_half_theta"] = jnp.cos(half)
        t["sin_half_theta"] = jnp.sin(half)
        t["sin_two_theta"] = jnp.sin(2.0 * theta)
        t["sin_three_theta"] = jnp.sin(3.0 * theta)
    if l_max >= 4:
        t["cos_two_theta"] = jnp.cos(2.0 * theta)
        t["sin_four_theta"] = jnp.sin(4.0 * theta)
    if l_max >= 5:
        t["cos_three_theta"] = jnp.cos(3.0 * theta)
        t["sin_five_theta"] = jnp.sin(5.0 * theta)
    if l_max >= 6:
        t["cos_four_theta"] = jnp.cos(4.0 * theta)
    if l_max >= 7:
        t["cos_five_theta"] = jnp.cos(5.0 * theta)
    if l_max >= 8:
        t["cos_six_theta"] = jnp.cos(6.0 * theta)
    return t


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
        Axis 0 varies theta/phi, axis 1 varies (l, m).
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)
    if l_max > _SUPPORTED_LMAX:
        raise NotImplementedError(
            "jax_ile spherical harmonics implemented for l<=%d (requested l_max=%d)"
            % (_SUPPORTED_LMAX, l_max))

    trig = _build_trig(theta, l_max)
    cols, m_vals = [], []
    for (l_i, m_i) in lm:
        l_i, m_i = int(l_i), int(m_i)
        coeff = _COEFFS[l_i][m_i]
        cols.append(coeff * _theta_shape(l_i, m_i, trig))
        m_vals.append(m_i)

    Ylm = jnp.stack(cols, axis=-1).astype(jnp.complex128)  # (S, K)
    m_arr = jnp.asarray(m_vals, dtype=jnp.float64)
    phase = jnp.exp(1.0j * (phi[:, None] * m_arr[None, :]))
    return Ylm * phase


# ---------------------------------------------------------------------------
# Embedded validation: run `python spherical.py`
# ---------------------------------------------------------------------------
def _validate(verbose=True):
    """Validate every (l,m), l<=8, against lal.SpinWeightedSphericalHarmonic
    (independent gold standard) and the production numpy implementation."""
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import lal
    from RIFT.likelihood.SphericalHarmonics_gpu import SphericalHarmonicsVectorized

    rng = np.random.default_rng(0)
    S = 11
    theta = rng.uniform(0.05, np.pi - 0.05, S)
    phi = rng.uniform(0.0, 2 * np.pi, S)

    worst_lal = 0.0
    worst_np = 0.0
    for l in range(2, 9):
        modes = [(l, m) for m in range(-l, l + 1)]
        Y_jax = np.asarray(spherical_harmonics_vectorized(modes, theta, phi, l_max=l))
        Y_np = SphericalHarmonicsVectorized(np.array(modes), theta, phi, xpy=np, l_max=l)
        e_lal_l = 0.0
        for i, (ll, mm) in enumerate(modes):
            Y_lal = np.array([lal.SpinWeightedSphericalHarmonic(float(th), float(ph), -2, ll, mm)
                              for th, ph in zip(theta, phi)])
            e_lal = np.max(np.abs(Y_jax[:, i] - Y_lal))
            e_np = np.max(np.abs(Y_jax[:, i] - Y_np[:, i]))
            e_lal_l = max(e_lal_l, e_lal)
            worst_lal = max(worst_lal, e_lal)
            worst_np = max(worst_np, e_np)
            if e_lal > 1e-10 and verbose:
                print("  MISMATCH (%d,%d): vs lal=%.2e" % (ll, mm, e_lal))
        if verbose:
            print("  l=%d : max|jax-lal|=%.2e  (%d modes)" % (l, e_lal_l, len(modes)))
    assert worst_lal < 1e-10, "JAX SH disagrees with lal (%.2e)" % worst_lal
    assert worst_np < 1e-12, "JAX SH disagrees with numpy reference (%.2e)" % worst_np
    print("OK: jax_ile spherical harmonics l=2..8 match lal (max %.2e) "
          "and the numpy reference (max %.2e)" % (worst_lal, worst_np))


if __name__ == "__main__":
    _validate()
