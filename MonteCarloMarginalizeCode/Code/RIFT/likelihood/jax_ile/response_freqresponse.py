"""
JAX port of the finite-size (Path D) response coefficients.

The extrinsic layer of the frequency-dependent-response likelihood
(``factored_likelihood_freqresponse``) needs, per detector, the complex
coefficients ``b_p`` of the response basis ``p = 0 .. Qmax+1`` -- these multiply
the (sky-independent) precompute banks ``Q^p_lm``, ``U^{p,p'}``, ``V^{p,p'}``.
The frequency basis ``c_q(f)`` is already folded into the precompute weights
``W_p(f)``, so the extrinsic layer only needs the *scalar* sky/pol coefficients

    b_0     = F0        (exact long-wavelength lal.ComputeDetAMResponse)
    b_{1+q} = beta_q    = (1/2)[ zx^2 a_x^q - zy^2 a_y^q ]  , q = 0 .. Qmax

which are closed-form analytic functions of ``(RA, DEC, psi)`` (with ``GMST(tref)``
a host constant).  Ported to ``jax.numpy`` (differentiable in the sky/pol angles),
mirroring ``slowrot_freqresponse.finite_size_geometry`` / ``finite_size_beta`` and
``factored_likelihood_freqresponse.response_coefficients``.  Validated against the
numpy reference to ~1e-12 by ``test/jax/test_jax_freqresponse.py``.

Unlike the sidereal-rotation case, every basis weight ``W_p`` is Hermitian, so the
V cross term needs NO harmonic reflection: the reflection index is the identity.

Detector geometry (response tensor ``D``, arm unit vectors ``x_arm``, ``y_arm``,
arm length ``L``) is supplied by the caller as host constants (from
``slowrot_freqresponse.detector_geometry``); only ``RA, DEC, psi`` are JAX leaves.
"""

import numpy as np
import jax.numpy as jnp


def _triad_jax(dec, psi, g):
    """Polarization triad X, Y and source direction nhat at hour angle g=GMST-RA.

    JAX port of ``slowrot_freqresponse._triad``; vectors carry the 3-component on
    the last axis.  dec, psi, g are (S,) JAX arrays.
    """
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    cp, sp = jnp.cos(psi), jnp.sin(psi)
    cg, sg = jnp.cos(g), jnp.sin(g)
    ones = jnp.ones_like(sg)
    X = jnp.stack([-cp * sg - sp * cg * sd,
                   -cp * cg + sp * sg * sd,
                   sp * cd * ones], axis=-1)
    Y = jnp.stack([sp * sg - cp * cg * sd,
                   sp * cg + cp * sg * sd,
                   cp * cd * ones], axis=-1)
    nhat = jnp.stack([cd * cg, -cd * sg, sd * ones], axis=-1)
    return X, Y, nhat


def _lwl_response_jax(D, X, Y):
    """Long-wavelength F_+, F_x (== ComputeDetAMResponse).  D host, X/Y JAX (S,3)."""
    XDX = jnp.einsum('...i,ij,...j->...', X, D, X)
    YDY = jnp.einsum('...i,ij,...j->...', Y, D, Y)
    XDY = jnp.einsum('...i,ij,...j->...', X, D, Y)
    YDX = jnp.einsum('...i,ij,...j->...', Y, D, X)
    return XDX - YDY, XDY + YDX


def response_coefficients_dict(response, x_arm, y_arm, RA, DEC, psi, gmst_tref,
                               Qmax):
    """JAX analogue of ``response_coefficients``: ``{p: (S,) complex}``.

    Parameters
    ----------
    response : (3,3) host array   detector response tensor.
    x_arm, y_arm : (3,) host arrays  Earth-fixed arm unit vectors.
    RA, DEC, psi : (S,) JAX arrays.
    gmst_tref : float             GMST(tref) [rad], host constant.
    Qmax : int                    highest arm-projection power retained.

    b_0 = F0 (exact lal baseline), b_{1+q} = beta_q.  The arm length L does NOT
    enter here (it lives in the precompute's W_p weights); only the arm *unit
    vectors* enter, through the projections a_x, a_y and zx, zy.
    """
    RA = jnp.asarray(RA, dtype=jnp.float64)
    DEC = jnp.asarray(DEC, dtype=jnp.float64)
    psi = jnp.asarray(psi, dtype=jnp.float64)
    D = jnp.asarray(response, dtype=jnp.float64)
    xa = jnp.asarray(np.asarray(x_arm, dtype=float))
    ya = jnp.asarray(np.asarray(y_arm, dtype=float))

    g = gmst_tref - RA
    X, Y, nhat = _triad_jax(DEC, psi, g)
    Fp_lwl, Fc_lwl = _lwl_response_jax(D, X, Y)
    F0 = Fp_lwl + 1j * Fc_lwl                        # (S,), exact lal baseline

    Xx = jnp.einsum('...i,i->...', X, xa)
    Yx = jnp.einsum('...i,i->...', Y, xa)
    Xy = jnp.einsum('...i,i->...', X, ya)
    Yy = jnp.einsum('...i,i->...', Y, ya)
    zx = Xx + 1j * Yx
    zy = Xy + 1j * Yy
    ax = jnp.einsum('...i,i->...', nhat, xa)
    ay = jnp.einsum('...i,i->...', nhat, ya)
    zx2, zy2 = zx ** 2, zy ** 2

    b = {0: F0}
    for q in range(Qmax + 1):
        b[1 + q] = 0.5 * (zx2 * ax ** q - zy2 * ay ** q)
    return b


def pack_coefficients(coeff_dict, p_list, S):
    """Align a ``{p: (S,)}`` coefficient dict to the fixed ``p_list`` order.

    Returns a ``(A, S)`` complex JAX array, row ``i`` = ``coeff_dict[p_list[i]]``.
    """
    rows = []
    for p in p_list:
        rows.append(jnp.broadcast_to(coeff_dict[int(p)], (S,)).astype(jnp.complex128))
    return jnp.stack(rows, axis=0)


def reflection_index(p_list):
    """Identity map (A,) -- the finite-size V term needs no reflection (W_p Hermitian)."""
    return np.arange(len(p_list), dtype=np.int64)


def response_coefficients_packed(response, x_arm, y_arm, RA, DEC, psi, gmst_tref,
                                 Qmax, p_list):
    """Convenience: ``response_coefficients_dict`` + ``pack_coefficients`` -> (A,S)."""
    S = int(jnp.asarray(RA).shape[0])
    cdict = response_coefficients_dict(response, x_arm, y_arm, RA, DEC, psi,
                                       gmst_tref, Qmax)
    return pack_coefficients(cdict, p_list, S)
