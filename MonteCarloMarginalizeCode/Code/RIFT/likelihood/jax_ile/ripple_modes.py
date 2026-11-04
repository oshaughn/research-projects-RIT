"""IMRPhenomXPHM modes on the accelerator, from ripple (optional dependency).

ripple (``pip install rippleGW``, tested with 0.4.1) implements IMRPhenomXPHM in
JAX: XHM co-precessing modes twisted up with MSA angles, as lalsimulation does by
default.  It returns only (h+, hx).  Inside its twist-up, each ``twist_*`` routine
contracts the J-frame modes with -2Y_lm(theta_JN, 0), and h+, hx are linear in
those harmonics.  ``RippleXPHMModes`` re-runs the twist-up with the harmonic
functions replaced by one-hot indicators, one call per (l, m), so each call
returns one mode's coefficient:

    h_lm(f > 0) = h+ - i hx,      h_lm(-f) = conj(h+ + i hx)

on RIFT's two-sided (non-Hermitian, LAL-ordered) frequency grid.

The modes are in the J frame for one fixed viewing choice (inclination 0.3,
phiRef 0.2).  Any orientation-marginalized likelihood (uniform cos incl, phiref,
psi) is invariant under that frame choice, so the result is usable for such an
objective.  It is NOT a drop-in replacement for RIFT's L-frame lalsim modes in
ILE.  Time convention: t = 0 at the LAL FD reference (no RIFT conditioning);
mode epoch 0.

Approximation: XPHM (here and in lalsimulation) builds its co-precessing modes
with reflection symmetry, h_l,-m = (-1)^l conj(h_lm), so there are no
antisymmetric modes.  Runs that analyse with antisymmetric modes on (e.g.
SEOBNRv5PHM with enable_antisymmetric_modes) see a model difference here.  The
mode extraction and ``conj_modes`` below are exact given ripple's h+, hx,
and nothing here assumes the symmetry: every m is extracted separately and V is
built from the conjugate modes directly, so models with antisymmetric content
pass through unchanged.

This module needs only jax and ripple.  It is a targeting/development path; the
dependency-free cupy port is separate work.
"""
import numpy as np

import jax
import jax.numpy as jnp

_VIEW_INCL, _VIEW_PHI = 0.3, 0.2


def _import_ripple():
    try:
        import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM as X
        from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXHM import (
            XLALSimIMRPhenomXHMGethlmModes, build_pWF22)
        from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import (
            IMRPhenomX_Initialize_MSA_System, lal_M_sec)
        from ripplegw.constants import MPC, MRSUN, MTSUN
    except ImportError as exc:
        raise ImportError("ripple_modes needs ripple (pip install rippleGW==0.4.1)") from exc
    return X, XLALSimIMRPhenomXHMGethlmModes, build_pWF22, \
        IMRPhenomX_Initialize_MSA_System, lal_M_sec, MPC, MRSUN, MTSUN


class RippleXPHMModes:
    """Inertial (J-frame) XPHM modes h_lm(f), l <= l_max, on a two-sided grid.

    freqs : (N,) two-sided frequencies in the order of the data arrays
        (``data.f0 + data.deltaF * arange(N)``).  Modes are nonzero only for
        fmin <= |f| <= fmax.
    """

    def __init__(self, freqs, fmin, fmax, fref, l_max=4, dist_mpc=1000.0):
        (self._X, self._gethlm, self._pwf22, self._msa, self._msec,
         MPC, self._MRSUN, self._MTSUN) = _import_ripple()
        self._MPC = MPC
        freqs = np.asarray(freqs, dtype=float)
        self.N = len(freqs)
        fmax = fmax if fmax else np.max(np.abs(freqs))
        pos = np.flatnonzero((freqs >= fmin) & (freqs <= fmax))
        fband = freqs[pos]
        # negative-frequency bins whose |f| is in band, mapped onto the band index
        # -f must have a +f partner on the grid: drop the -fNyq bin (no +fNyq bin)
        neg = np.flatnonzero((-freqs >= fmin) & (-freqs <= fmax) & (-freqs < -freqs[0]))
        lookup = {round(f / (freqs[1] - freqs[0])): i for i, f in enumerate(fband)}
        negmap = np.array([lookup[round(-freqs[k] / (freqs[1] - freqs[0]))] for k in neg])
        self._pos, self._neg, self._negmap = map(jnp.asarray, (pos, neg, negmap))
        self.fband = jnp.asarray(fband)
        self.fref = float(fref)
        self.dist_mpc = float(dist_mpc)
        self.lms = [(l, m) for l in range(2, int(l_max) + 1) for m in range(-l, l + 1)]
        self._fn = jax.jit(self._modes)

    def _modes(self, m1, m2, chi1, chi2):
        hpos, hneg = self.modes_at(self.fband, m1, m2, chi1, chi2)
        full = jnp.zeros((len(self.lms), self.N), dtype=jnp.complex128)
        full = full.at[:, self._pos].set(hpos)
        full = full.at[:, self._neg].set(hneg[:, self._negmap])
        return full

    def modes_at(self, fpos, m1, m2, chi1, chi2):
        """(h_lm(+f), h_lm(-f)) at positive frequencies fpos, each (K, len(fpos))."""
        X = self._X
        Mf = fpos * self._msec(m1, m2)
        Mtot = m1 + m2
        amp0 = Mtot * self._MRSUN * Mtot * self._MTSUN / (self.dist_mpc * self._MPC)
        pairs = [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]
        msa = self._msa(mass_1=m1, mass_2=m2, chi1x=chi1[0], chi1y=chi1[1], chi1z=chi1[2],
                        chi2x=chi2[0], chi2y=chi2[1], chi2z=chi2[2],
                        reference_frequency=self.fref)
        pwf = self._pwf22(m1, m2, chi1[2], chi2[2], self.fref, msa_SAv2=msa[15],
                          msa_S1L_pav=msa[32], msa_S2L_pav=msa[33])
        hd = self._gethlm(Mf, pwf, phi0=0.0, ell_mm_pairs=pairs)
        seeds = jnp.stack([(-1 if l % 2 else 1) * hd[(l, m)] for (l, m) in pairs]) * amp0
        # One-hot harmonics keyed on a traced mode index J, vmapped over J: the
        # precession angles do not depend on J, so they are computed once.
        code = {lm: k for k, lm in enumerate(self.lms)}
        holder = {}

        def make(l):
            def f(theta, m):
                return jnp.where(holder["J"] == code.get((l, m), -1), 1.0, 0.0) + 0.0 * theta
            return f

        def one(J):
            holder["J"] = J
            return X.twistup(Mf, m1, m2, chi1[0], chi1[1], chi1[2], chi2[0], chi2[1],
                             chi2[2], _VIEW_PHI, _VIEW_INCL, self.fref, seeds)

        saved = (X.compute_sminus2_l2, X.compute_sminus2_l3, X.compute_sminus2_l4)
        try:
            X.compute_sminus2_l2, X.compute_sminus2_l3, X.compute_sminus2_l4 = make(2), make(3), make(4)
            hp, hc = jax.vmap(one)(jnp.arange(len(self.lms)))
        finally:
            X.compute_sminus2_l2, X.compute_sminus2_l3, X.compute_sminus2_l4 = saved
        return hp - 1j * hc, jnp.conj(hp + 1j * hc)

    def __call__(self, m1, m2, chi1, chi2):
        """(K, N) complex modes for masses in Msun, chi1/chi2 = (x, y, z) at fref."""
        return self._fn(jnp.asarray(m1, dtype=jnp.float64), jnp.asarray(m2, dtype=jnp.float64),
                        jnp.asarray(chi1, dtype=jnp.float64), jnp.asarray(chi2, dtype=jnp.float64))


def conj_modes(h):
    """FT of conj(h(t)) on the LAL two-sided grid: conj(h[(N - k) % N])."""
    N = h.shape[-1]
    idx = (N - jnp.arange(N)) % N
    return jnp.conj(h[..., idx])


def reduce_qvu(h, data_fd, weights, delta_f, n_shift, n_window):
    """Q (n_window, K), U, V (K, K) as gpu_precompute.compound_precompute_arrays (A=1)."""
    N = h.shape[-1]
    alt = 1 - 2 * (jnp.arange(N) & 1)
    integrand = 2.0 * jnp.conj(h) * data_fd[None, :] * weights[None, :]
    full = jnp.fft.ifft(integrand, axis=-1) * alt * (N * delta_f)
    q = jnp.roll(full, -n_shift, axis=-1)[:, :n_window]
    hw = h * weights[None, :]
    u = 2.0 * delta_f * (jnp.conj(h) @ hw.T)
    v = 2.0 * delta_f * (jnp.conj(conj_modes(h)) @ hw.T)
    return q.T, u, v
