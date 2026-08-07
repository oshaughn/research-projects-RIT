"""
JAX ports of the slow-rotation (Path A / Path B) response coefficients.

The extrinsic layer of the rotation-aware likelihood
(``factored_likelihood_with_rotation``) needs, per detector, the complex
coefficients ``C_a`` of each elementary modulated template ``a = (p, n)`` --
these multiply the (sky-independent) precompute banks ``Q^a_lm``, ``U^{a,a'}``,
``V^{a,a'}``.  ``C_a`` is a closed-form analytic function of ``(RA, DEC, psi)``
(with ``GMST(tref)`` a host constant), so it ports cleanly to ``jax.numpy`` and
is differentiable in the sky/polarization angles.

This mirrors, term for term, the numpy reference
``factored_likelihood_with_rotation.rotation_coefficients_vector`` (which in turn
builds on ``slowrot_response.antenna_harmonics_vector`` /
``delay_harmonics_vector``).  Validated against it to ~1e-12 by
``test/jax/test_jax_slowrot.py``.

The detector-fixed inputs (``response`` tensor, ``location`` vector) are host
constants supplied by the caller (from ``lalsimulation.DetectorPrefixToLALDetector``);
only ``DEC, psi, RA`` are JAX (differentiable) leaves and ``gmst_tref`` a host float.
"""

import math

import numpy as np
import jax.numpy as jnp

# Sidereal angular rate [rad/s]; identical to
# factored_likelihood_with_rotation.OMEGA_EARTH (used only for meta consistency).
OMEGA_EARTH = 7.292115e-5
C_SI = 299792458.0  # m/s, matches slowrot_response.C_SI


def _antenna_harmonics_jax(D, dec, psi):
    """JAX port of ``slowrot_response.antenna_harmonics_vector``.

    Returns ``{n: (S,) complex}`` for ``n in (-2,-1,0,1,2)`` -- the complex
    antenna-pattern harmonics ``A_n`` with ``F(t)=sum_n A_n exp(i n g)``,
    ``g = GMST(t) - RA``.  Depends only on the (host) response tensor ``D`` and
    the (JAX) declination/polarization.
    """
    D = jnp.asarray(D, dtype=jnp.float64)
    dec = jnp.asarray(dec, dtype=jnp.float64)
    psi = jnp.asarray(psi, dtype=jnp.float64)
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    cp, sp = jnp.cos(psi), jnp.sin(psi)
    z = jnp.zeros_like(dec)

    def vec(a, b, c):
        return jnp.stack(jnp.broadcast_arrays(a, b, c), axis=-1)  # (S,3)

    Xc = vec(-sp * sd, -cp, z)
    Xs = vec(-cp, sp * sd, z)
    X0 = vec(z, z, sp * cd)
    Yc = vec(-cp * sd, sp, z)
    Ys = vec(sp, cp * sd, z)
    Y0 = vec(z, z, cp * cd)

    def B(u, v):
        return jnp.einsum('...i,ij,...j->...', u, D, v)

    Pp0 = 0.5 * (B(Xc, Xc) + B(Xs, Xs)) + B(X0, X0) - (0.5 * (B(Yc, Yc) + B(Ys, Ys)) + B(Y0, Y0))
    Pp1 = 2.0 * (B(Xc, X0) - B(Yc, Y0))
    Qp1 = 2.0 * (B(Xs, X0) - B(Ys, Y0))
    Pp2 = 0.5 * ((B(Xc, Xc) - B(Xs, Xs)) - (B(Yc, Yc) - B(Ys, Ys)))
    Qp2 = B(Xc, Xs) - B(Yc, Ys)

    Pc0 = B(Xc, Yc) + B(Xs, Ys) + 2.0 * B(X0, Y0)
    Pc1 = 2.0 * (B(Xc, Y0) + B(X0, Yc))
    Qc1 = 2.0 * (B(Xs, Y0) + B(X0, Ys))
    Pc2 = B(Xc, Yc) - B(Xs, Ys)
    Qc2 = B(Xc, Ys) + B(Xs, Yc)

    P0 = Pp0 + 1j * Pc0
    P1 = Pp1 + 1j * Pc1
    Q1 = Qp1 + 1j * Qc1
    P2 = Pp2 + 1j * Pc2
    Q2 = Qp2 + 1j * Qc2
    return {
        0: P0,
        1: 0.5 * (P1 - 1j * Q1),
        -1: 0.5 * (P1 + 1j * Q1),
        2: 0.5 * (P2 - 1j * Q2),
        -2: 0.5 * (P2 + 1j * Q2),
    }


def _delay_harmonics_jax(location, dec):
    """JAX port of ``slowrot_response.delay_harmonics_vector``.

    Returns ``{m: (S,) complex}`` for ``m in (-1,0,1)`` -- the geometric-delay
    harmonics ``B_m`` [s], ``tau(t)=sum_m B_m exp(i m g)``.
    """
    r = np.asarray(location, dtype=float)
    dec = jnp.asarray(dec, dtype=jnp.float64)
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    T0 = -(r[2] * sd) / C_SI
    T1c = -(cd * r[0]) / C_SI
    T1s = (cd * r[1]) / C_SI
    return {0: T0 + 0j, 1: 0.5 * (T1c - 1j * T1s), -1: 0.5 * (T1c + 1j * T1s)}


def _convolve_harmonics(a, b):
    """Convolve two harmonic sequences (dicts {m: coef}) -> dict {m: coef}.

    Same as ``factored_likelihood_with_rotation._convolve_harmonics`` but the
    coefficients are JAX arrays.
    """
    out = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            out[m1 + m2] = out.get(m1 + m2, 0.0) + c1 * c2
    return out


def rotation_coefficients_dict(response, location, RA, DEC, psi, gmst_tref,
                               p_max):
    """JAX analogue of ``rotation_coefficients_vector``: ``{(p,n): (S,) complex}``.

    Parameters
    ----------
    response : (3,3) host array   detector response tensor.
    location : (3,) host array    detector location [m].
    RA, DEC, psi : (S,) JAX arrays (differentiable leaves).
    gmst_tref : float             GMST(tref) [rad], host constant.
    p_max : int                   0 = Path A (amplitude only); >=1 = Path B.

    Same algebra as the numpy reference; ``g_ev = gmst_tref - RA``.
    """
    RA = jnp.asarray(RA, dtype=jnp.float64)
    g_ev = gmst_tref - RA
    A = _antenna_harmonics_jax(response, DEC, psi)
    Atil = {n: A[n] * jnp.exp(1j * n * g_ev) for n in A}
    if p_max == 0:
        return {(0, n): Atil[n] for n in Atil}
    Bd = _delay_harmonics_jax(location, DEC)
    Btil = {m: Bd[m] * jnp.exp(1j * m * g_ev) for m in Bd}
    tau0 = jnp.real(sum(Btil.values()))
    D = {m: Btil[m] for m in Btil}
    D[0] = D[0] - tau0
    negD = {m: -D[m] for m in D}
    C = {}
    E = {0: jnp.ones_like(g_ev, dtype=jnp.complex128)}
    for p in range(p_max + 1):
        if p > 0:
            E = _convolve_harmonics(E, negD)
        inv = 1.0 / math.factorial(p)
        for n, an in Atil.items():
            for m, em in E.items():
                key = (p, n + m)
                C[key] = C.get(key, 0.0) + inv * an * em
    return C


def pack_coefficients(coeff_dict, a_list, S):
    """Align a ``{(p,n): (S,)}`` coefficient dict to the fixed ``a_list`` order.

    Returns a ``(A, S)`` complex JAX array with row ``i`` = ``coeff_dict[a_list[i]]``
    (zeros where the dict lacks that key), exactly matching the numpy NoLoop
    ``Cg`` behaviour (keys of the dict outside ``a_list`` are dropped; ``a_list``
    entries absent from the dict contribute zero).
    """
    rows = []
    for a in a_list:
        a = (int(a[0]), int(a[1]))
        if a in coeff_dict:
            rows.append(jnp.broadcast_to(coeff_dict[a], (S,)).astype(jnp.complex128))
        else:
            rows.append(jnp.zeros((S,), dtype=jnp.complex128))
    return jnp.stack(rows, axis=0)  # (A, S)


def reflection_index(a_list):
    """Static index map ``i -> j`` with ``a_list[j] = (p, -n)`` for ``a_list[i]=(p,n)``.

    The rotation V cross term contracts ``C_{(p,-n)}`` (harmonic reflection).  The
    harmonic set is symmetric (the reference asserts this), so ``(p,-n)`` is always
    present in ``a_list`` and the map is total.  Returns an ``(A,)`` int numpy array.
    """
    a_list = [(int(p), int(n)) for (p, n) in a_list]
    pos = {a: i for i, a in enumerate(a_list)}
    refl = []
    for (p, n) in a_list:
        key = (p, -n)
        if key not in pos:
            raise ValueError(
                "reflection partner (p,-n)=%r absent from a_list -- harmonic set "
                "must be symmetric for the V term" % (key,))
        refl.append(pos[key])
    return np.asarray(refl, dtype=np.int64)


def rotation_coefficients_packed(response, location, RA, DEC, psi, gmst_tref,
                                 p_max, a_list):
    """Convenience: ``rotation_coefficients_dict`` + ``pack_coefficients``.

    Returns a ``(A, S)`` complex array aligned to ``a_list``.
    """
    S = int(jnp.asarray(RA).shape[0])
    cdict = rotation_coefficients_dict(response, location, RA, DEC, psi,
                                       gmst_tref, p_max)
    return pack_coefficients(cdict, a_list, S)
