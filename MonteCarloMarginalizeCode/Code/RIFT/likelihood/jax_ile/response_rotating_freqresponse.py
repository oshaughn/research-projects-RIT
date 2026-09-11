"""JAX coefficients for simultaneous slow rotation and finite-arm response.

This is the JAX analogue of
``factored_likelihood_rotating_freqresponse.combined_response_coefficients_vector``.
The packed elementary-template index is ``a=(b,p,n)``: finite-frequency basis,
delay-derivative order, and sidereal harmonic.
"""

import math

import numpy as np
import jax.numpy as jnp

from . import response_freqresponse as _rf
from . import response_slowrot as _rs


def response_harmonic_width(basis_index):
    return 2 if int(basis_index) == 0 else int(basis_index) + 1


def _basis_harmonics(response, x_arm, y_arm, dec, psi, Qmax):
    """Exact small-DFT coefficients of each finite-response sky factor."""
    dec = jnp.asarray(dec, dtype=jnp.float64)
    psi = jnp.asarray(psi, dtype=jnp.float64)
    D = jnp.asarray(response, dtype=jnp.float64)
    xa = jnp.asarray(np.asarray(x_arm, dtype=float))
    ya = jnp.asarray(np.asarray(y_arm, dtype=float))

    width = int(Qmax) + 2
    ngrid = 2 * width + 1
    g = 2.0 * jnp.pi * jnp.arange(ngrid, dtype=jnp.float64) / float(ngrid)
    X, Y, nhat = _rf._triad_jax(dec[:, None], psi[:, None], g[None, :])
    Fp, Fc = _rf._lwl_response_jax(D, X, Y)
    zx = jnp.einsum('...i,i->...', X, xa) + 1j * jnp.einsum('...i,i->...', Y, xa)
    zy = jnp.einsum('...i,i->...', X, ya) + 1j * jnp.einsum('...i,i->...', Y, ya)
    ax = jnp.einsum('...i,i->...', nhat, xa)
    ay = jnp.einsum('...i,i->...', nhat, ya)

    values = {0: Fp + 1j * Fc}
    for q in range(int(Qmax) + 1):
        values[1 + q] = 0.5 * (zx ** 2 * ax ** q - zy ** 2 * ay ** q)

    out = {}
    for b, vals in values.items():
        bw = response_harmonic_width(b)
        out[b] = {
            n: jnp.mean(vals * jnp.exp(-1j * n * g)[None, :], axis=1)
            for n in range(-bw, bw + 1)
        }
    return out


def coefficients_dict(response, location, x_arm, y_arm, RA, DEC, psi,
                      gmst_tref, Qmax, p_max):
    """Return ``{(b,p,n): (S,)}`` compound response coefficients."""
    RA = jnp.asarray(RA, dtype=jnp.float64)
    DEC = jnp.asarray(DEC, dtype=jnp.float64)
    psi = jnp.asarray(psi, dtype=jnp.float64)
    g_ev = float(gmst_tref) - RA

    basis = _basis_harmonics(response, x_arm, y_arm, DEC, psi, Qmax)
    basis_tref = {
        b: {n: value * jnp.exp(1j * n * g_ev) for n, value in harmonics.items()}
        for b, harmonics in basis.items()
    }

    delay = _rs._delay_harmonics_jax(location, DEC)
    delay_tref = {m: value * jnp.exp(1j * m * g_ev)
                  for m, value in delay.items()}
    tau0 = jnp.real(sum(delay_tref.values()))
    drift = dict(delay_tref)
    drift[0] = drift[0] - tau0
    neg_drift = {m: -value for m, value in drift.items()}

    out = {}
    expansion = {0: jnp.ones_like(g_ev, dtype=jnp.complex128)}
    for p in range(int(p_max) + 1):
        if p:
            expansion = _rs._convolve_harmonics(expansion, neg_drift)
        inv_fact = 1.0 / math.factorial(p)
        for b, harmonics in basis_tref.items():
            for n, amplitude in harmonics.items():
                for m, delay_amplitude in expansion.items():
                    key = (b, p, n + m)
                    out[key] = (out.get(key, 0.0)
                                + inv_fact * amplitude * delay_amplitude)
    return out


def coefficients_packed(response, location, x_arm, y_arm, RA, DEC, psi,
                        gmst_tref, Qmax, p_max, a_list):
    """Return compound coefficients as an ``(A,S)`` array aligned to ``a_list``."""
    S = int(jnp.asarray(RA).shape[0])
    coeff = coefficients_dict(response, location, x_arm, y_arm, RA, DEC, psi,
                              gmst_tref, Qmax, p_max)
    rows = []
    for a in a_list:
        key = tuple(int(v) for v in a)
        rows.append(jnp.broadcast_to(coeff.get(
            key, jnp.zeros((S,), dtype=jnp.complex128)), (S,)))
    return jnp.stack(rows, axis=0).astype(jnp.complex128)


def reflection_index(a_list):
    """Map ``(b,p,n)`` to ``(b,p,-n)`` for the V contraction."""
    keys = [tuple(int(v) for v in a) for a in a_list]
    pos = {a: i for i, a in enumerate(keys)}
    reflected = []
    for b, p, n in keys:
        key = (b, p, -n)
        if key not in pos:
            raise ValueError("reflection partner %r absent from compound a_list" % (key,))
        reflected.append(pos[key])
    return np.asarray(reflected, dtype=np.int64)
