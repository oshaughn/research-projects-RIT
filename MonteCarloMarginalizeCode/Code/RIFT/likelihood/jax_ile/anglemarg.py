"""
Exact (phi_ref, psi) angle marginalization for the JAX factored likelihood.

WHY THIS MODULE EXISTS
----------------------
:func:`core.fused_log_likelihood_distphipsimarg` marginalizes (phi_ref, psi)
by averaging exp(lnL) over the SAME small grid it evaluates the likelihood on
(nphi x npsi, production 8x8).  That is an 8-node quadrature of a function
whose peak width is ~1/SNR, so the error grows without bound with SNR
(measured against a converged reference: ~1e2 nats at SNR 40, ~7e3 nats at
SNR 320 for the psi marginal at npsi=8).  The 8-point phi grid additionally
puts the n=4 phi harmonic (present for any m_max=2 signal through rho^2)
exactly AT Nyquist, where it aliases onto n=-4.

THE STRUCTURE THE FIX EXPLOITS (analytic, not tunable)
------------------------------------------------------
At fixed time and distance the factored lnL is a bivariate trigonometric
polynomial of KNOWN low order in (phi_ref, psi):

  * the antenna pattern enters as F(psi) = F(0) e^{-2 i psi} -- linearly in
    kappa (u-harmonics +-1, with u = 2 psi) and quadratically in rho^2
    (u-harmonics {0, +-2});
  * the harmonics Y_lm carry e^{i m phi_ref} -- kappa has phi-harmonics up to
    m_max, rho^2 up to 2*m_max.

So the two unit-distance fields the likelihood is built from,

    A(phi, psi; t) = Re kappa_unit      (phi order <= m_max,   u order <= 1)
    B(phi, psi; t) = rho^2_unit         (phi order <= 2*m_max, u order <= 2)

are EXACTLY determined by their values on a small Nyquist-sized sample grid.
The expensive :func:`core._accumulate_unit` evaluations are needed only to
pin those Fourier coefficients; every subsequent evaluation of
lnL_t(phi, psi | x) = x*A - 0.5*x^2*B (x = distMpcRef/d) is pure arithmetic.
The number of expensive evaluations is fixed by MODE CONTENT, never by SNR,
and is asserted -- there is no accuracy-vs-cost knob to set too small.

TWO MARGINALIZATION SCHEMES (plus a selector, see the wrapper/driver)
---------------------------------------------------------------------
exact   : reconstruct lnL_t on a dense (phi, u) product grid from the
          coefficient tables and average exp(.) over it.  The dense grid is
          free (no likelihood calls); its size is derived from a DATA-DERIVED
          amplitude bound (:func:`estimate_angle_amplitude`, computed from
          the coefficient tables themselves -- never from a caller's SNR
          estimate) via :func:`_dense_grid_sizes`, floored at the crossover
          amplitude.  Best at low/moderate amplitude.
laplace : marginalize psi ANALYTICALLY by Laplace's method at every
          (phi, distance-node, time) point -- at fixed (phi, x, t) the
          u-exponent is exactly a + b cos(u-beta) + d cos(2u-delta), whose
          stationary points Newton finds from u0 = beta, beta+pi in a few
          elementary iterations, and whose curvature is closed-form.  This
          removes the psi axis entirely (cost ~SNR instead of ~SNR^2) and its
          O(1/amplitude) error SHRINKS as SNR grows.  Best at high amplitude.

Both schemes marginalize distance with the same quadrature machinery as the
grid path (:func:`core._logsumexp_grid_blocked`, or the adaptive
:func:`core._distmarg_gh_logL` when JAX_ILE_DISTMARG_GH is set -- exact
scheme only), and use the same normalization convention (mean over uniform
angle grids, i.e. the uniform priors dphi/2pi, dpsi/pi), so they are
drop-in replacements for the grid function and agree with it wherever the
grid is converged (pinned in test/jax/test_angle_marg_exact.py).

Everything here is jit/vmap/grad-compatible: no numpy in the hot path, no
scipy special functions, lax.scan (checkpointed) bounds memory by CHUNK, not
by grid size.
"""

import numpy as np
import jax
import jax.numpy as jnp

from . import core as _core
from .core import (JAX_INTERP_DEFAULT, _accumulate_unit, _time_marginalize,
                   _logsumexp_grid_blocked, _distmarg_gh_logL,
                   make_distance_gh)

__all__ = [
    "angle_sample_grid_sizes",
    "angle_coefficient_tables",
    "estimate_angle_amplitude",
    "fused_log_likelihood_distphipsimarg_exact",
    "fused_log_likelihood_distphipsimarg_laplace",
    "choose_angle_marg_scheme",
    "ANGLE_MARG_CROSSOVER_AMPLITUDE",
]


# ---------------------------------------------------------------------------
# Selector crossover and dense-grid sizing constants.
#
# Calibrated 2026-08-27 on the SEOBNRv4 35+30 Msun HLV injection harness
# (the configuration behind the measured production-grid errors quoted in the
# module docstring), on the full distance+phi+psi-marginalized lnL, against a
# brute-force dense product-grid reference (which agrees with the exact
# scheme to ~2e-12 wherever it is affordable).  Measured errors in nats:
#
#     SNR   A=rho^2/2   exact(self-conv)  laplace-exact   grid 32x8   grid 8x8
#      10        50          7e-15          -1.1e-03       -5.9e-02   -9.5e-02
#      20       200          0              -1.8e-04       -8.8e-01   -1.7e+00
#      40       800          2.4e-09        -1.6e-05       -5.6e+00   -1.1e+01
#      80      3200          4.6e-13        -2.8e-06       -2.7e+01   -5.1e+01
#
# The Laplace error falls FASTER than 1/A here; the isolated-kernel error law
# is ~0.1/b nats (pinned in test_angle_marg_exact.py).  The crossover sits
# where BOTH schemes are deep in their accurate regimes (laplace ~1e-4,
# exact ~machine), so the switch is insensitive to the O(1) slack in the
# measured amplitude bound that drives it, and tests evaluate both schemes in
# the overlap region and assert agreement -- the crossover is a validated
# constant, not a tuning knob.
# ---------------------------------------------------------------------------
ANGLE_MARG_CROSSOVER_AMPLITUDE = 450.0     # A = rho^2/2; rho = 30.  NOTE the
# auto selector compares the MARGINED data-derived bound (~2x the true
# amplitude) to this, so laplace engages from true A ~ 225 (SNR ~ 21).  That
# early engagement is safe by measurement: laplace is at -1.8e-4 nats by
# A = 200 on the injection ladder and improves upward, while exact remains
# valid (crossover-floored sizing) below.
# Dense-size rule N = ceil(K * sqrt(A)) points, from the trapezoid aliasing
# error of exp(trig poly): relative error ~ exp(-c N^2 / A).  The constants
# carry a >= 2x margin in N over the empirically adequate values (error
# floor reached in the calibration ladder).
_DENSE_K_U = 8.0        # u = 2 psi axis
_DENSE_K_PHI = 16.0     # phi axis: harmonics up to 2*m_max, dominant n=2
_DENSE_FLOOR_U = 64
_DENSE_FLOOR_PHI = 128


def angle_sample_grid_sizes(m_max):
    """Nyquist-derived (nphi_s, npsi_s) sample-grid sizes for mode content m_max.

    lnL_t carries phi-harmonics up to 2*m_max and u-harmonics up to 2
    (u = 2 psi), so unaliased sampling needs nphi_s > 2*(2*m_max) and
    npsi_s > 2*2.  These are DERIVED and asserted, not options: the historical
    defect was precisely a settable sample size (nphi=8 aliases the n=4
    harmonic; npsi=8 under-resolves nothing at the SAMPLING stage but the
    grid was also used as the quadrature).
    """
    m_max = int(m_max)
    if m_max < 1:
        raise ValueError("m_max must be >= 1, got %r" % m_max)
    nphi_s = 4 * m_max + 8     # >= 2*(2 m_max)+2, margin >= 6 harmonics
    npsi_s = 8                 # u content is <= 2 for ANY mode set (spin-2)
    assert nphi_s >= 2 * (2 * m_max) + 2
    assert npsi_s >= 2 * 2 + 2
    return nphi_s, npsi_s


def _data_m_max(data):
    lms = np.asarray(data.lms)
    return int(np.max(np.abs(lms[:, 1])))


def angle_coefficient_tables(data, ra, dec, incl, interp=JAX_INTERP_DEFAULT,
                             sample_chunk=None):
    """Exact 2-D Fourier coefficient tables of A = Re kappa_unit, B = rho^2_unit.

    Samples :func:`core._accumulate_unit` on the Nyquist-sized
    (nphi_s x npsi_s) grid from :func:`angle_sample_grid_sizes` and
    accumulates the discrete Fourier coefficients

        C[kp, ks] = (1/Ns) sum_j f(phi_j, psi_j) e^{-i kp phi_j} e^{-i ks u_j}

    for the harmonics the physics allows: A keeps kp in 0..m_max,
    ks in -1..1; B keeps kp in 0..2*m_max, ks in -2..2 (u = 2 psi;
    negative-kp coefficients follow from Hermitian symmetry of the real
    fields and are not stored).  Reconstruction weights are w_kp = 1 for
    kp = 0 and 2 for kp > 0 (the kp = 0 row stores both ks signs, whose
    conjugate pairing is already real).

    Memory: the tables are (m_max+1, 3, S, npts) and (2*m_max+1, 5, S, npts)
    complex -- independent of every grid size.  The sample scan runs in
    chunks of ``sample_chunk`` grid points (default npsi_s, i.e. one phi row
    per step), checkpointed so reverse-mode AD does not store per-step
    intermediates.

    Returns ``(C_A, C_B, meta)`` with ``meta = dict(m_max, nphi_s, npsi_s)``.
    """
    m_max = _data_m_max(data)
    nphi_s, npsi_s = angle_sample_grid_sizes(m_max)
    if sample_chunk is None:
        sample_chunk = npsi_s
    KPA, KSA = m_max + 1, 1          # ks in -KSA..KSA
    KPB, KSB = 2 * m_max + 1, 2

    phi_s = np.linspace(0.0, 2.0 * np.pi, nphi_s, endpoint=False)
    psi_s = np.linspace(0.0, np.pi, npsi_s, endpoint=False)
    PH, PS = np.meshgrid(phi_s, psi_s, indexing="ij")
    pairs = np.stack([PH.ravel(), PS.ravel()], axis=-1)          # (Ns, 2)
    Ns = pairs.shape[0]
    if Ns % sample_chunk:
        raise ValueError("sample_chunk must divide nphi_s*npsi_s")

    def _phase_table(kp_max, ks_max):
        kp = np.arange(kp_max + 1)
        ks = np.arange(-ks_max, ks_max + 1)
        return np.exp(-1j * (pairs[:, 0, None, None] * kp[None, :, None]
                             + 2.0 * pairs[:, 1, None, None] * ks[None, None, :])
                      ) / Ns                                     # (Ns, KP, KS)

    phase_A = _phase_table(m_max, KSA)
    phase_B = _phase_table(2 * m_max, KSB)

    ra = jnp.asarray(ra, dtype=jnp.float64)
    dec = jnp.asarray(dec, dtype=jnp.float64)
    incl = jnp.asarray(incl, dtype=jnp.float64)
    S = ra.shape[0]
    npts = data.npts
    c = int(sample_chunk)
    nsteps = Ns // c

    xs = (jnp.asarray(pairs.reshape(nsteps, c, 2)),
          jnp.asarray(phase_A.reshape(nsteps, c, KPA, 2 * KSA + 1)),
          jnp.asarray(phase_B.reshape(nsteps, c, KPB, 2 * KSB + 1)))

    def _step(carry, x):
        CA, CB = carry
        prs, pA, pB = x                                # (c,2),(c,KPA,3),(c,KPB,5)
        # batch the c grid points against the S parameter rows: (c*S,)
        ra_b = jnp.broadcast_to(ra[None, :], (c, S)).reshape(-1)
        dec_b = jnp.broadcast_to(dec[None, :], (c, S)).reshape(-1)
        incl_b = jnp.broadcast_to(incl[None, :], (c, S)).reshape(-1)
        phi_b = jnp.broadcast_to(prs[:, 0][:, None], (c, S)).reshape(-1)
        psi_b = jnp.broadcast_to(prs[:, 1][:, None], (c, S)).reshape(-1)
        ku, rs = _accumulate_unit(data, ra_b, dec_b, psi_b, incl_b, phi_b,
                                  interp, False)
        A = ku.real.reshape(c, S, npts)
        B = rs.reshape(c, S, npts)
        CA = CA + jnp.einsum("ckq,cst->kqst", pA, A)
        CB = CB + jnp.einsum("ckq,cst->kqst", pB, B)
        return (CA, CB), None

    CA0 = jnp.zeros((KPA, 2 * KSA + 1, S, npts), dtype=jnp.complex128)
    CB0 = jnp.zeros((KPB, 2 * KSB + 1, S, npts), dtype=jnp.complex128)
    (C_A, C_B), _ = jax.lax.scan(jax.checkpoint(_step), (CA0, CB0), xs)
    meta = dict(m_max=m_max, nphi_s=nphi_s, npsi_s=npsi_s)
    return C_A, C_B, meta


def _kp_weights(KP):
    w = np.ones(KP)
    w[1:] = 2.0
    return jnp.asarray(w)


def _reconstruct_field(C, phi, u):
    """Evaluate the real trig polynomial with coefficient table C at (phi, u).

    C: (KP, 2*KS+1, S, npts) complex from :func:`angle_coefficient_tables`;
    phi, u: (c,) points.  Returns (c, S, npts) float64.
    """
    KP = C.shape[0]
    KS = (C.shape[1] - 1) // 2
    kp = jnp.arange(KP, dtype=jnp.float64)
    ks = jnp.arange(-KS, KS + 1, dtype=jnp.float64)
    E = jnp.exp(1j * (phi[:, None, None] * kp[None, :, None]
                      + u[:, None, None] * ks[None, None, :]))    # (c,KP,KS)
    E = E * _kp_weights(KP)[None, :, None]
    return jnp.einsum("ckq,kqst->cst", E, C).real


def _dense_grid_sizes(amp):
    """(nphi_d, nu_d) dense reconstruction sizes adequate for amplitude ``amp``.

    Derived from the trapezoid aliasing error of exp(trig poly of amplitude
    A): N = K*sqrt(A) with the calibrated constants above (>= 2x margin), and
    hard floors.  This is NOT a settable knob; callers pass the DATA-DERIVED
    amplitude bound from :func:`estimate_angle_amplitude` (which is floored
    at the crossover).
    """
    amp = max(float(amp), 25.0)
    n_u = max(_DENSE_FLOOR_U, int(np.ceil(_DENSE_K_U * np.sqrt(amp))))
    n_phi = max(_DENSE_FLOOR_PHI, int(np.ceil(_DENSE_K_PHI * np.sqrt(amp))))
    # round up to multiples of 16 so chunking stays regular
    n_u = ((n_u + 15) // 16) * 16
    n_phi = ((n_phi + 15) // 16) * 16
    return n_phi, n_u


ANGLE_AMP_SKY_POINTS = 64     # sky/inclination draws for the amplitude bound
ANGLE_AMP_MARGIN = 2.0        # covers the finite sky sample: the amplitude is
                              # a smooth O(1)-varying function of sky position,
                              # and the sizing error enters only through
                              # sqrt(amp), so a 2x margin in amplitude is a
                              # 1.4x margin in grid size


def estimate_angle_amplitude(data, x_grid, interp=JAX_INTERP_DEFAULT,
                             n_sky=ANGLE_AMP_SKY_POINTS, seed=0,
                             margin=ANGLE_AMP_MARGIN,
                             _n_phi_e=None, _n_u_e=24):
    """DATA-DERIVED bound on the (phi, psi)-exponent amplitude A.

    This is the number that sizes the dense reconstruction grids, and it is
    computed from the very function being integrated -- NOT from a caller's
    SNR estimate.  (External review, correctly: sizing from ``guess_snr``
    meant a missing or underestimated SNR silently under-resolved the dense
    quadrature, quietly reintroducing exactly the failure mode this module
    exists to remove -- the n_psi=8 defect again, one level up.)

    Method: evaluate the coefficient tables EAGERLY (concrete numpy inputs,
    build time -- grid sizes must be static under jit, so this cannot run
    inside the traced likelihood) at ``n_sky`` random sky/inclination
    points.  The PRIMARY estimate is the EMPIRICAL maximum of the exponent
    over a dense angular reconstruction: A and B are trig polynomials of
    known order (<= (2*m_max, 2)), so a 96 x 24 grid reconstructs them
    exactly up to interpolation and the grid max understates the continuum
    max by < 1% (peak offset <= half a cell, curvature <= (k_max)^2 A) --
    absorbed in ``margin``.  Per angle point the distance max is closed
    form: B >= 0 makes x*A - x^2/2*B concave in x, so the max over the
    ACTUAL x support is at clip(A/B, x_min, x_max).

    A second, analytic bound max_x (x*M_A - x^2/2*B0)+ (M_A = sum w|C_A|
    pointwise-bounds |A|; B0 = angular mean of B) is kept as a runtime
    CROSS-CHECK: it pairs the max of A with the MEAN of B, which review
    item 5 correctly noted is heuristic (B can dip below its mean where A
    peaks).  Empirically it over-bounds by 1.5-1.9x; if it ever reads BELOW
    the empirical max, the disagreement is printed and the larger value is
    used -- the failure is never silent in the too-small direction.

    Returns ``margin`` times the empirical max, UNfloored: the auto selector
    compares it to the crossover (a floor here would push every quiet target
    into the laplace branch); the WRAPPER floors the SIZING amplitude at the
    crossover separately, so grids are never sized below the calibration
    point.
    """
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 2.0 * np.pi, n_sky)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, n_sky))
    incl = np.arccos(rng.uniform(-1.0, 1.0, n_sky))
    C_A, C_B, meta = angle_coefficient_tables(data, ra, dec, incl,
                                              interp=interp)
    C_A = np.asarray(C_A)
    C_B = np.asarray(C_B)
    x = np.asarray(x_grid)
    x_min, x_max = float(x.min()), float(x.max())

    # dense angular reconstruction matrices (numpy mirror of
    # _reconstruct_field; content <= (2*m_max, 2) so 96 x 24 is ~6x Nyquist)
    def _recon_matrix(KP, KS, phis, us):
        kp = np.arange(KP)
        ks = np.arange(-KS, KS + 1)
        E = np.exp(1j * (phis[:, None, None] * kp[None, :, None]
                         + us[:, None, None] * ks[None, None, :]))
        w = np.ones(KP)
        w[1:] = 2.0
        return (E * w[None, :, None]).reshape(len(phis), -1)   # (n_ang, KP*KS)

    # The reconstruction grid is DERIVED from the mode content and ASSERTED,
    # exactly like the sample grid (fail-closed): sampled at >= 8x per
    # highest harmonic, a band-limited trig polynomial's grid max under-reads
    # its continuum max by < 1% (absorbed in `margin`), while a coarser grid
    # can miss an adversarially-phased n = 2*m_max harmonic entirely -- so a
    # too-small grid is refused, not trusted.  _n_phi_e/_n_u_e are test
    # hooks; production callers take the derived defaults.
    m_max_e = meta["m_max"]
    if _n_phi_e is None:
        _n_phi_e = max(96, 16 * (2 * m_max_e))
    n_phi_e, n_u_e = int(_n_phi_e), int(_n_u_e)
    assert n_phi_e >= 8 * (2 * m_max_e), \
        "estimator phi grid %d under-samples 2*m_max=%d content" \
        % (n_phi_e, 2 * m_max_e)
    assert n_u_e >= 8 * 2, \
        "estimator u grid %d under-samples order-2 content" % (n_u_e,)
    PH, UU = np.meshgrid(np.linspace(0, 2 * np.pi, n_phi_e, endpoint=False),
                         np.linspace(0, 2 * np.pi, n_u_e, endpoint=False),
                         indexing="ij")
    phis, us = PH.ravel(), UU.ravel()
    E_A = _recon_matrix(C_A.shape[0], (C_A.shape[1] - 1) // 2, phis, us)
    E_B = _recon_matrix(C_B.shape[0], (C_B.shape[1] - 1) // 2, phis, us)

    amp_emp = 0.0
    for j in range(n_sky):          # per-sky loop bounds the transient
        A_g = (E_A @ C_A[:, :, j].reshape(-1, C_A.shape[-1])).real
        B_g = np.maximum((E_B @ C_B[:, :, j].reshape(-1, C_B.shape[-1])).real,
                         0.0)                                  # (n_ang, npts)
        x_hat = np.clip(A_g / np.maximum(B_g, 1e-300), x_min, x_max)
        val = x_hat * A_g - 0.5 * np.square(x_hat) * B_g
        amp_emp = max(amp_emp, float(val.max()))
    amp_emp = max(amp_emp, 0.0)

    # analytic cross-check (heuristic direction documented above)
    w = np.ones(C_A.shape[0])
    w[1:] = 2.0
    M_A = np.einsum("k,kqst->st", w, np.abs(C_A))
    ks0 = (C_B.shape[1] - 1) // 2
    B0 = np.maximum(C_B[0, ks0].real, 0.0)
    expo = (x[:, None, None] * M_A[None]
            - 0.5 * np.square(x)[:, None, None] * B0[None])
    amp_analytic = float(np.clip(expo, 0.0, None).max())
    if amp_analytic < amp_emp * (1.0 - 1e-9):
        # The analytic expression should over-bound (measured 1.5-1.9x); if
        # it reads below the near-exact empirical max, say so LOUDLY -- the
        # empirical value stands either way, so the too-small failure mode
        # cannot occur silently.
        print("estimate_angle_amplitude: analytic bound %.6g fell BELOW the "
              "empirical max %.6g (the review-flagged heuristic direction); "
              "the empirical value governs." % (amp_analytic, amp_emp))
    return margin * amp_emp


def _require_amp_sizing(amp_sizing):
    if amp_sizing is None:
        raise ValueError(
            "amp_sizing is required: pass a sound UPPER bound on the "
            "(phi,psi)-exponent amplitude A ~ rho^2/2, e.g. "
            "estimate_angle_amplitude(data, x_grid).  There is deliberately "
            "no default: a too-small value silently under-resolves the dense "
            "quadrature, which is the defect this module exists to fix.")
    return float(amp_sizing)


def _lse_update(m, s, e, axis=0):
    """Running log-sum-exp: fold block ``e`` (reduced over ``axis``) into (m, s).

    Robust to all--inf blocks and an all--inf carry (both yield exp(-inf)=0
    rather than the exp(-inf - -inf) = nan of the naive update): padded chunk
    tails and rejected Laplace bins produce -inf entries by design.
    """
    m_blk = jnp.max(e, axis=axis)
    m_safe = jnp.where(jnp.isfinite(m_blk), m_blk, 0.0)
    s_blk = jnp.sum(jnp.exp(e - jnp.expand_dims(m_safe, axis)), axis=axis)
    s_blk = jnp.where(jnp.isfinite(m_blk), s_blk, 0.0)
    m_new = jnp.maximum(m, m_blk)
    m_new_safe = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
    s_new = (s * jnp.exp(jnp.where(jnp.isfinite(m), m - m_new_safe, -jnp.inf))
             + s_blk * jnp.exp(jnp.where(jnp.isfinite(m_blk),
                                         m_blk - m_new_safe, -jnp.inf)))
    return m_new, s_new


def _pad_chunks(values, chunk):
    """Split (N,) point arrays into (nsteps, chunk) with -inf log-pad weights."""
    N = values[0].shape[0]
    nsteps = (N + chunk - 1) // chunk
    pad = nsteps * chunk - N
    lw = np.zeros(N)
    out = []
    for v in values:
        out.append(np.pad(v, (0, pad), mode="edge").reshape(nsteps, chunk))
    lw = np.pad(lw, (0, pad), constant_values=-np.inf).reshape(nsteps, chunk)
    return [jnp.asarray(o) for o in out] + [jnp.asarray(lw)]


def fused_log_likelihood_distphipsimarg_exact(
        data, ra, dec, incl, x_grid, log_w_grid,
        interp=JAX_INTERP_DEFAULT, amp_sizing=None,
        dense_chunk=8, grid_block=32):
    """Distance-, phi_ref- AND psi-marginalized lnL: exact-coefficient scheme.

    Drop-in replacement for :func:`core.fused_log_likelihood_distphipsimarg`
    (same signature contract minus the two grid arguments, same normalization
    convention: uniform priors dphi/2pi, dpsi/pi).  The expensive likelihood
    is sampled ONLY on the Nyquist grid fixed by mode content; the (phi, psi)
    quadrature runs on a dense reconstruction whose size follows
    :func:`_dense_grid_sizes` for ``amp_sizing`` -- a REQUIRED upper bound on
    the exponent amplitude A ~ rho^2/2, obtained from
    :func:`estimate_angle_amplitude` (the wrapper does this automatically).
    There is no default: a silently-undersized grid is the defect this
    module exists to fix.  Honors JAX_ILE_DISTMARG_GH exactly as the grid
    path does.

    Memory is bounded by ``dense_chunk`` (points per scan step), never by the
    dense grid size: the largest transient is the inner distance-quadrature
    slab (dense_chunk * S, npts, grid_block), ~0.8 GB f64 at the defaults for
    a batched S=64, npts=614 call -- these two are COST/MEMORY knobs only,
    with no effect on the result.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    C_A, C_B, meta = angle_coefficient_tables(data, ra, dec, incl, interp)
    S = ra.shape[0]
    npts = data.npts

    amp_sizing = _require_amp_sizing(amp_sizing)
    nphi_d, nu_d = _dense_grid_sizes(amp_sizing)
    phi_d = np.linspace(0.0, 2.0 * np.pi, nphi_d, endpoint=False)
    u_d = np.linspace(0.0, 2.0 * np.pi, nu_d, endpoint=False)   # u = 2 psi
    PH, UU = np.meshgrid(phi_d, u_d, indexing="ij")
    c = int(dense_chunk)
    phi_x, u_x, lw_x = _pad_chunks([PH.ravel(), UU.ravel()], c)
    n_dense = nphi_d * nu_d

    a_g = x_grid
    b_g = -0.5 * jnp.square(x_grid)
    _use_gh = _core._DISTMARG_GH_N > 0
    if _use_gh:
        gh_xi, gh_logw = make_distance_gh(_core._DISTMARG_GH_N)
        x_min = jnp.min(x_grid)
        x_max = jnp.max(x_grid)

    def _step(carry, x):
        m, s = carry
        phw, uw, lww = x
        A = _reconstruct_field(C_A, phw, uw)                  # (c,S,npts)
        B = _reconstruct_field(C_B, phw, uw)
        K2 = A.reshape(c * S, npts)
        R2 = B.reshape(c * S, npts)
        if _use_gh:
            lnL = _distmarg_gh_logL(K2, R2, gh_xi, gh_logw, x_min, x_max)
        else:
            lnL = _logsumexp_grid_blocked(K2, R2, a_g, b_g, log_w_grid,
                                          grid_block)
        lnL = lnL.reshape(c, S, npts) + lww[:, None, None]
        m_new, s_new = _lse_update(m, s, lnL, axis=0)
        return (m_new, s_new), None

    m0 = jnp.full((S, npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, npts), dtype=jnp.float64)
    (m, s), _ = jax.lax.scan(jax.checkpoint(_step), (m0, s0),
                             (phi_x, u_x, lw_x))
    lnL_t = m + jnp.log(s) - jnp.log(float(n_dense))
    return _time_marginalize(lnL_t, data.w_t)


# ---------------------------------------------------------------------------
# Analytic psi Laplace
# ---------------------------------------------------------------------------

# Series/Laplace handover (external review, item 2: a hard switch at
# b + 2d = 0.5 left a WIDE bad window just above the cut -- the truncated
# 2-term series stopped exactly where Laplace is still O(1)-wrong, giving
# 0.27-0.53 nats of value error and a SIGN-INVERTED |c2| gradient across
# b + 2d in [0.5, ~5]).  The series now carries Bessel cross terms to k = 5
# (each I_n as a truncated power series -- elementary ops, no scipy) and is
# accurate through b + 2d ~ 6, Laplace is accurate above ~5, and the two are
# blended C^1-smoothly over [LO, HI] so no bin ever crosses a hard branch
# boundary as (ra, dec, incl) move.
# Band placement: the blended value is C^1, but its gradient carries
# dw/dtheta * (series - laplace), i.e. the blend-weight slope times the local
# BRANCH DISAGREEMENT (= Laplace's O(1/A) error, worst ~1.75/(b+d)).  Placing
# the band at [10, 16] keeps that term <= ~0.2 in the worst draw and a few
# 1e-2 typically, with the series machine-exact through t = 10.
_LAPLACE_BLEND_LO = 10.0      # pure series below this in t = b + 2d
_LAPLACE_BLEND_HI = 16.0      # pure Laplace above this
_LAPLACE_SERIES_TERMS = 26    # power-series length per Bessel; at the series
                              # clamp b <= 16 the last term is ~1e-9 relative
_LAPLACE_SERIES_KMAX = 8      # cross-term order; verified to 1e-10 against
                              # quadrature across t <= BLEND_LO by the sweep
                              # tests (worst split b = 8, d = 4)
_LAPLACE_BRACKET_CELLS = 24   # sign-scan cells for the stationary points of
                              # f(u): f' is a degree-2 trig polynomial, so it
                              # has AT MOST 4 transversal zeros on the circle
                              # (its resultant quartic 2d e^{-i delta} z^4 +
                              # b e^{-i beta} z^3 - b e^{i beta} z -
                              # 2d e^{i delta} = 0, z = e^{iu}); 24 cells
                              # (width 0.26) place adjacent zeros in distinct
                              # cells except for a MERGING max-min pair
                              # (f' = f'' = 0), which cannot contain the
                              # global maximum and carries negligible weight.
_LAPLACE_MAX_ROOTS = 4


def _scaled_iv(n, x, terms=None):
    """I_n(z) / z^n as a fixed-length power series in x = z^2 (Horner).

    I_n(z)/z^n = (1/2^n) sum_m (z^2/4)^m / (m! (m+n)!) -- entire in x, all
    coefficients positive, so the truncation error is bounded by the first
    dropped term: ~1e-14 relative at the series clamp z <= 6 with the
    default length.  Elementary ops only (the kernel may not touch scipy).
    """
    import math
    if terms is None:
        terms = _LAPLACE_SERIES_TERMS
    q = x / 4.0
    coefs = [1.0 / (math.factorial(m) * math.factorial(m + n))
             for m in range(terms)]
    acc = jnp.zeros_like(q) + coefs[-1]
    for cm in reversed(coefs[:-1]):
        acc = acc * q + cm
    return acc / (2.0 ** n)


def _laplace_psi_lnI(a, c1, c2):
    """log[(1/pi) int_0^pi exp(a + Re(c1 e^{iu}) + Re(c2 e^{2iu})) dpsi], u = 2 psi.

    Laplace's method with ALL maxima enumerated.  An earlier revision seeded
    Newton only at the extrema of the FIRST harmonic (u0 = beta, beta+pi with
    beta = -arg c1); that assumes b >> d and fails outright when the first
    harmonic cancels: for c1 = 0, c2 = -d, d > 0.5 both seeds land on MINIMA
    and the routine returned -inf for a finite integral (found in external
    review).  This version brackets every transversal zero of f' by a sign
    scan over _LAPLACE_BRACKET_CELLS cells (interval-based, so coincident
    roots cannot be double-counted), bisects under stop_gradient, applies one
    differentiable Newton polish step (Newton is a contraction, so a single
    step from the converged point carries the correct implicit derivative
    without a deep 1/H^2 gradient chain), keeps roots with curvature H below
    a small POSITIVE tolerance (so a near-degenerate maximum contributes with
    the floored curvature instead of being dropped), and sums the Laplace
    factors.  Everything is angle-free -- f, f', f'' are evaluated directly
    from c1, c2, so arg(0) never appears and b = 0 is a regular point.

    At small-to-moderate amplitude the EXACT Bessel expansion
    (1/pi) int = e^a [I0(b) I0(d) + 2 sum_k I_2k(b) I_k(d) cos(k(2beta-delta))]
    is used instead, truncated at k = _LAPLACE_SERIES_KMAX with each I_n a
    fixed-length power series (Laplace degenerates as the curvature
    vanishes; the series converges fastest exactly there).  The phases are
    division-free: with w = c2 conj(c1)^2, cos(k(2beta-delta)) Bessel
    prefactors combine to polynomial coefficients times Re(w^k).  The two
    branches are blended C^1-smoothly over b + 2d in [_LAPLACE_BLEND_LO,
    _LAPLACE_BLEND_HI] -- a hard switch put sign-inverted gradients in the
    window just above the old cut (external review, item 2).

    Elementary functions only (no scipy Bessels, no eigensolvers);
    differentiable; any input shape (elementwise over broadcast a, c1, c2).
    """
    mag1 = jnp.square(c1.real) + jnp.square(c1.imag)
    mag2 = jnp.square(c2.real) + jnp.square(c2.imag)
    b = jnp.sqrt(mag1 + 1e-300)
    d = jnp.sqrt(mag2 + 1e-300)

    t_amp = b + 2.0 * d
    lap_dummy = t_amp < _LAPLACE_BLEND_LO      # blend weight is exactly 1 here
    # jnp.where's VJP sends a ZERO cotangent through the unselected branch,
    # and 0 * inf = nan: the Laplace branch must have BOUNDED derivatives
    # (including second, for .fisher()) even on the pure-series bins.  Feed
    # it safe dummy amplitudes there (their contribution is weighted 0 by
    # the blend) and floor the curvature RELATIVE to the amplitude scale.
    c1l = jnp.where(lap_dummy, 5.0 + 0.0j, c1)
    c2l = jnp.where(lap_dummy, 0.25 + 0.0j, c2)
    bl = jnp.sqrt(jnp.square(c1l.real) + jnp.square(c1l.imag) + 1e-300)
    dl = jnp.sqrt(jnp.square(c2l.real) + jnp.square(c2l.imag) + 1e-300)
    h_floor = 1e-6 * (bl + 4.0 * dl)

    def fval(u):
        eiu = jnp.exp(1j * u)
        return (c1l * eiu).real + (c2l * eiu * eiu).real

    def fp(u):
        eiu = jnp.exp(1j * u)
        return -(c1l * eiu).imag - 2.0 * (c2l * eiu * eiu).imag

    def fpp(u):
        eiu = jnp.exp(1j * u)
        return -(c1l * eiu).real - 4.0 * (c2l * eiu * eiu).real

    def _guard(H):
        # sign-preserving denominator floor
        return jnp.where(jnp.abs(H) >= h_floor, H,
                         jnp.where(H >= 0, h_floor, -h_floor))

    # ---- bracket every transversal zero of f' (at most 4; see the constant)
    # Signs are taken one grid node at a time so the transient stays one
    # X-sized array; roots are assigned to at most _LAPLACE_MAX_ROOTS slot
    # registers in encounter order.  Interval-based bracketing cannot yield
    # duplicate roots: the sign sequence flips exactly once per transversal
    # crossing, including a crossing that sits exactly on a grid node.
    N = _LAPLACE_BRACKET_CELLS
    ug = np.linspace(0.0, 2.0 * np.pi, N + 1)
    cell = ug[1] - ug[0]
    zero_f = jnp.zeros_like(b)
    false_x = jnp.zeros_like(b, dtype=bool)
    s_prev = fp(jnp.asarray(ug[0])) >= 0
    count = zero_f
    los = [zero_f for _ in range(_LAPLACE_MAX_ROOTS)]
    s_los = [false_x for _ in range(_LAPLACE_MAX_ROOTS)]
    filled = [false_x for _ in range(_LAPLACE_MAX_ROOTS)]
    for k in range(N):
        s_next = fp(jnp.asarray(ug[k + 1])) >= 0
        flip = s_prev != s_next
        for j in range(_LAPLACE_MAX_ROOTS):
            take = flip & (count == j)
            los[j] = jnp.where(take, ug[k], los[j])
            s_los[j] = jnp.where(take, s_prev, s_los[j])
            filled[j] = filled[j] | take
        count = count + flip.astype(count.dtype)
        s_prev = s_next

    # ---- per-slot bisection (value-only) + one differentiable polish step
    terms = []
    for j in range(_LAPLACE_MAX_ROOTS):
        lo = los[j]
        hi = lo + cell
        slo = s_los[j]
        for _ in range(20):           # cell/2^20 ~ 2.5e-7, then Newton
            mid = 0.5 * (lo + hi)
            go_right = (fp(mid) >= 0) == slo
            lo = jnp.where(go_right, mid, lo)
            hi = jnp.where(go_right, hi, mid)
        u0 = jax.lax.stop_gradient(0.5 * (lo + hi))
        u = u0 - fp(u0) / _guard(fpp(u0))
        H = fpp(u)
        # tolerant acceptance: a maximum with H in [-h_floor, +h_floor) is a
        # (near-)degenerate flat top; drop it and a finite integral could
        # come back -inf, so keep it with the floored curvature instead
        # (its Laplace weight is then merely inaccurate, never absent).
        ok = filled[j] & (H < h_floor)
        Hm = jnp.minimum(H, -h_floor)
        t = jnp.where(ok,
                      a + fval(u)
                      + 0.5 * jnp.log(2.0 * jnp.pi / (-Hm))
                      - jnp.log(2.0 * jnp.pi),      # (1/2 du/dpsi) * (1/pi)
                      -jnp.inf)
        terms.append(t)

    # guarded log-add-exp over the root slots: an all--inf slot set has a NaN
    # backward pass under the naive form, and the NaN leaks through jnp.where.
    mt = terms[0]
    for t in terms[1:]:
        mt = jnp.maximum(mt, t)
    mts = jnp.where(jnp.isfinite(mt), mt, 0.0)
    ssum = zero_f
    for t in terms:
        ssum = ssum + jnp.exp(t - mts)
    ln_laplace = jnp.where(ssum > 0,
                           mts + jnp.log(jnp.maximum(ssum, 1e-300)),
                           -jnp.inf)

    # ---- Bessel-series branch (exact expansion, truncated): inputs are
    # CLAMPED to the largest amplitudes the blend can weight (b <= 16,
    # d <= 8; magnitude-only scaling preserves the phases) so the fixed
    # power series never overflows on the pure-Laplace bins it is weighted
    # 0 on -- an unclamped b ~ 1e4 would overflow to inf and the inf leaks
    # through the blend's chain rule as 0 * inf = nan.
    b_s = jnp.minimum(b, 16.0)
    d_s = jnp.minimum(d, 8.0)
    c1_s = c1 * (b_s / b)
    c2_s = c2 * (d_s / d)
    x_b = b_s * b_s
    x_d = d_s * d_s
    w1 = c2_s * jnp.conj(c1_s) ** 2            # |w1| = b_s^2 d_s, arg = 2b-d
    series = _scaled_iv(0, x_b) * _scaled_iv(0, x_d)
    wk = w1
    for k in range(1, _LAPLACE_SERIES_KMAX + 1):
        series = series + (2.0 * _scaled_iv(2 * k, x_b)
                           * _scaled_iv(k, x_d) * wk.real)
        wk = wk * w1
    ln_series = a + jnp.log(jnp.maximum(series, 1e-300))

    # ---- C^1 blend: pure series below LO, pure Laplace above HI
    r = jnp.clip((_LAPLACE_BLEND_HI - t_amp)
                 / (_LAPLACE_BLEND_HI - _LAPLACE_BLEND_LO), 0.0, 1.0)
    wgt = r * r * (3.0 - 2.0 * r)               # smoothstep
    # ln_laplace cannot be -inf for t_amp >= LO (a periodic f' has >= 2 sign
    # flips and the tolerant acceptance keeps the global maximum), but guard
    # the 0-weight product against a hypothetical -inf anyway.
    ln_lap = jnp.where(jnp.isfinite(ln_laplace), ln_laplace, ln_series)
    return wgt * ln_series + (1.0 - wgt) * ln_lap


def fused_log_likelihood_distphipsimarg_laplace(
        data, ra, dec, incl, x_grid, log_w_grid,
        interp=JAX_INTERP_DEFAULT, amp_sizing=None,
        phi_chunk=16, dist_block=4):
    """Distance-, phi_ref- AND psi-marginalized lnL: analytic psi-Laplace scheme.

    Same contract and normalization as
    :func:`fused_log_likelihood_distphipsimarg_exact`, but the psi axis is
    removed analytically (see :func:`_laplace_psi_lnI`): at every
    (dense-phi, distance-node, time) point the u-exponent coefficients follow
    directly from the SAME coefficient tables,

        a  = x A0(phi) - x^2/2 B0(phi)
        c1 = x A1(phi) - x^2/2 B1(phi)          (order e^{iu})
        c2 =           - x^2/2 B2(phi)          (order e^{2iu})

    so no additional likelihood evaluations are needed.  Cost scales ~sqrt(A)
    (the dense phi axis) instead of ~A; the Laplace error is O(1/A) and
    SHRINKS with SNR.  The adaptive distance quadrature
    (JAX_ILE_DISTMARG_GH) is NOT supported on this path -- it would need a
    psi-marginal node-placement rule this PR does not validate -- and raises
    rather than being silently ignored.

    Memory is bounded by ``phi_chunk`` x ``dist_block``, never by grid sizes.
    """
    if _core._DISTMARG_GH_N > 0:
        raise ValueError(
            "JAX_ILE_DISTMARG_GH is set, but the 'laplace' angle-marg scheme "
            "does not support the adaptive distance quadrature (its node "
            "placement is defined per fixed-psi exponent).  Use "
            "--angle-marg-scheme exact, or unset JAX_ILE_DISTMARG_GH.")
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    C_A, C_B, meta = angle_coefficient_tables(data, ra, dec, incl, interp)
    m_max = meta["m_max"]
    S = ra.shape[0]
    npts = data.npts

    amp_sizing = _require_amp_sizing(amp_sizing)
    nphi_d, _ = _dense_grid_sizes(amp_sizing)
    phi_d = np.linspace(0.0, 2.0 * np.pi, nphi_d, endpoint=False)
    c = int(phi_chunk)
    phi_x, lw_x = _pad_chunks([phi_d], c)

    wA = _kp_weights(m_max + 1)
    wB = _kp_weights(2 * m_max + 1)
    kpA = jnp.arange(m_max + 1, dtype=jnp.float64)
    kpB = jnp.arange(2 * m_max + 1, dtype=jnp.float64)
    G = x_grid.shape[0]
    blk = int(dist_block)

    def _step(carry, x):
        m, s = carry
        phw, lww = x                                          # (c,)
        EA = jnp.exp(1j * phw[:, None] * kpA[None, :]) * wA[None, :]  # (c,KPA)
        EB = jnp.exp(1j * phw[:, None] * kpB[None, :]) * wB[None, :]

        def MA(ks_idx):
            return jnp.einsum("ck,kst->cst", EA, C_A[:, ks_idx])

        def MB(ks_idx):
            return jnp.einsum("ck,kst->cst", EB, C_B[:, ks_idx])

        # psi-Fourier coefficient FIELDS at the dense phi points (c,S,npts):
        # A(u) = A0 + Re(A1 e^{iu});  B(u) = B0 + Re(B1 e^{iu}) + Re(B2 e^{2iu})
        A0 = MA(1).real                       # ks index 1 == ks 0
        A1 = MA(2) + jnp.conj(MA(0))          # ks +1 plus conj(ks -1)
        B0 = MB(2).real
        B1 = MB(3) + jnp.conj(MB(1))
        B2 = MB(4) + jnp.conj(MB(0))

        # distance quadrature: blocked, vectorized over the block (AD-fast),
        # running log-sum-exp across blocks
        mx = jnp.full((c, S, npts), -jnp.inf, dtype=jnp.float64)
        sx = jnp.zeros((c, S, npts), dtype=jnp.float64)
        for start in range(0, G, blk):
            sl = slice(start, min(start + blk, G))
            xg = x_grid[sl][:, None, None, None]              # (g,1,1,1)
            lwg = log_w_grid[sl][:, None, None, None]
            av = xg * A0[None] - 0.5 * jnp.square(xg) * B0[None]
            c1 = xg * A1[None] - 0.5 * jnp.square(xg) * B1[None]
            c2 = -0.5 * jnp.square(xg) * B2[None]
            e = _laplace_psi_lnI(av, c1, c2) + lwg            # (g,c,S,npts)
            mx, sx = _lse_update(mx, sx, e, axis=0)
        lnI = (mx + jnp.where(sx > 0, jnp.log(jnp.maximum(sx, 1e-300)), -jnp.inf)
               + lww[:, None, None])                          # (c,S,npts)
        m_new, s_new = _lse_update(m, s, lnI, axis=0)
        return (m_new, s_new), None

    m0 = jnp.full((S, npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, npts), dtype=jnp.float64)
    (m, s), _ = jax.lax.scan(jax.checkpoint(_step), (m0, s0), (phi_x, lw_x))
    lnL_t = m + jnp.log(s) - jnp.log(float(nphi_d))
    return _time_marginalize(lnL_t, data.w_t)


def choose_angle_marg_scheme(amplitude, gh_enabled=None):
    """Select 'exact' or 'laplace' from a measured amplitude bound.

    ``amplitude`` is the DATA-DERIVED bound from
    :func:`estimate_angle_amplitude` (A ~ rho^2/2 scale) -- deliberately not
    an SNR guess: selection and dense-grid sizing both key on the measured
    coefficient tables, so a missing or wrong external SNR estimate can
    affect neither (external-review defect 2).

    The crossover ANGLE_MARG_CROSSOVER_AMPLITUDE sits where BOTH schemes are
    deep in their accurate regimes (see the constant's derivation note):
    laplace error ~1e-4 nats and falling, exact at machine precision with the
    crossover-sized dense grid.  The switch therefore tolerates the O(1)
    slack in the amplitude bound, and tests evaluate both schemes in the
    overlap region and assert agreement -- a validated constant, not a
    tuning knob.

    Returns ``(scheme, info)``; ``info`` is a provenance dict the caller MUST
    surface in the run log (this pipeline has a documented history of
    silently-inert flags).
    """
    if gh_enabled is None:
        gh_enabled = _core._DISTMARG_GH_N > 0
    if amplitude is None:
        return "exact", dict(reason="no amplitude bound available; exact "
                                    "scheme is the conservative branch",
                             amplitude=None,
                             crossover=ANGLE_MARG_CROSSOVER_AMPLITUDE)
    amp = float(amplitude)
    if gh_enabled:
        return "exact", dict(reason="JAX_ILE_DISTMARG_GH set: laplace does "
                                    "not support the adaptive distance "
                                    "quadrature", amplitude=amp,
                             crossover=ANGLE_MARG_CROSSOVER_AMPLITUDE)
    scheme = "laplace" if amp >= ANGLE_MARG_CROSSOVER_AMPLITUDE else "exact"
    return scheme, dict(reason="measured amplitude bound %s crossover"
                               % ("above" if scheme == "laplace" else "below"),
                        amplitude=amp,
                        crossover=ANGLE_MARG_CROSSOVER_AMPLITUDE)
