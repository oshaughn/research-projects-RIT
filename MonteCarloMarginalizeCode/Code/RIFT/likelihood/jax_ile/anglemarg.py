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
          free (no likelihood calls); its size is derived from the amplitude
          the branch must cover (see :func:`_dense_grid_sizes`) and, in the
          auto selector, is floored at the crossover amplitude so a wrong
          (low) SNR estimate can only ever OVERSIZE it.  Best at
          low/moderate amplitude.
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
    "fused_log_likelihood_distphipsimarg_exact",
    "fused_log_likelihood_distphipsimarg_laplace",
    "choose_angle_marg_scheme",
    "ANGLE_MARG_CROSSOVER_AMPLITUDE",
]


# ---------------------------------------------------------------------------
# Selector crossover and dense-grid sizing constants.
#
# Calibrated 2026-08-27 on the SEOBNRv4 35+30 Msun HLV injection harness
# (the same configuration behind the measured production-grid errors quoted
# in the module docstring), against a brute-force dense (phi,psi) reference
# grid: see the PR that adds this module for the ladder.  The Laplace error
# falls like 1/A (A = rho^2/2); it is already < 1e-3 nats by A ~ 200 and the
# exact scheme's dense-grid truncation error at the sizes below is < 1e-6
# nats up to the crossover.  The crossover is placed where BOTH schemes are
# accurate (< 1e-3 nats), so the auto selector's switch is validated at
# runtime by construction -- tests evaluate both schemes in the overlap
# region and assert agreement, and either branch alone is accurate there.
# ---------------------------------------------------------------------------
ANGLE_MARG_CROSSOVER_AMPLITUDE = 450.0     # A = rho^2/2; rho = 30
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
    hard floors.  This is NOT a settable knob; callers pass the amplitude the
    branch must cover (auto selection floors it at the crossover, so a wrong
    SNR estimate can only oversize the grid).
    """
    amp = max(float(amp), 25.0)
    n_u = max(_DENSE_FLOOR_U, int(np.ceil(_DENSE_K_U * np.sqrt(amp))))
    n_phi = max(_DENSE_FLOOR_PHI, int(np.ceil(_DENSE_K_PHI * np.sqrt(amp))))
    # round up to multiples of 16 so chunking stays regular
    n_u = ((n_u + 15) // 16) * 16
    n_phi = ((n_phi + 15) // 16) * 16
    return n_phi, n_u


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
        dense_chunk=16, grid_block=64):
    """Distance-, phi_ref- AND psi-marginalized lnL: exact-coefficient scheme.

    Drop-in replacement for :func:`core.fused_log_likelihood_distphipsimarg`
    (same signature contract minus the two grid arguments, same normalization
    convention: uniform priors dphi/2pi, dpsi/pi).  The expensive likelihood
    is sampled ONLY on the Nyquist grid fixed by mode content; the (phi, psi)
    quadrature runs on a dense reconstruction whose size follows
    :func:`_dense_grid_sizes` for ``amp_sizing`` (peak-amplitude bound
    A ~ rho^2/2 this call must cover; the wrapper floors it at the auto
    crossover).  Honors JAX_ILE_DISTMARG_GH exactly as the grid path does.

    Memory is bounded by ``dense_chunk`` (points per scan step), never by the
    dense grid size.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    C_A, C_B, meta = angle_coefficient_tables(data, ra, dec, incl, interp)
    S = ra.shape[0]
    npts = data.npts

    if amp_sizing is None:
        amp_sizing = ANGLE_MARG_CROSSOVER_AMPLITUDE
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

_LAPLACE_SERIES_CUT = 0.5     # b + 2 d below this: small-amplitude Bessel series


def _laplace_psi_lnI(a, c1, c2):
    """log[(1/pi) int_0^pi exp(a + Re(c1 e^{iu}) + Re(c2 e^{2iu})) dpsi], u = 2 psi.

    Writes the exponent as a + b cos(u - beta) + d cos(2u - delta) with
    b = |c1|, beta = -arg(c1), d = |c2|, delta = -arg(c2).  Maxima are found
    by Newton from u0 = beta and beta + pi (b >> d in practice, so both
    branches converge in a few elementary iterations); the Laplace factor is
    closed-form.  Below b + 2d < _LAPLACE_SERIES_CUT the truncated Bessel
    series log[I0(b) I0(d) + 2 I2(b) I1(d) cos(2 beta - delta)] (small-argument
    polynomial I_k) is used instead -- Laplace degenerates as the curvature
    vanishes, the series is accurate exactly there, and such bins carry
    e^{-O(A)} relative weight in the high-amplitude regime this scheme serves.

    Elementary functions only (no scipy Bessels); differentiable; any input
    shape (applied elementwise over broadcasted a, c1, c2).
    """
    # |.| via sqrt(re^2 + im^2 + tiny): jnp.abs of an exactly-zero complex has
    # a NaN gradient, and c2 vanishes identically for special geometries.
    mag1 = jnp.square(c1.real) + jnp.square(c1.imag)
    mag2 = jnp.square(c2.real) + jnp.square(c2.imag)
    b = jnp.sqrt(mag1 + 1e-300)
    d = jnp.sqrt(mag2 + 1e-300)
    # angle() of an exactly-zero complex has a NaN gradient; mask those bins
    # ON THE UNFLOORED MAGNITUDE (b, d are floored by construction, so a mask
    # on them would never trigger).  Their cos term is ~0-weighted anyway.
    c1m = jnp.where(mag1 > 1e-280, c1, 1.0 + 0.0j)
    c2m = jnp.where(mag2 > 1e-280, c2, 1.0 + 0.0j)
    beta = -jnp.angle(c1m)
    delta = -jnp.angle(c2m)

    use_series = b + 2.0 * d < _LAPLACE_SERIES_CUT
    # jnp.where's VJP sends a ZERO cotangent through the unselected branch,
    # and 0 * inf = nan: the Laplace branch must therefore have BOUNDED
    # gradients even on the bins the series branch serves.  Feed it safe
    # dummy amplitudes there (the result is discarded by the where below),
    # and floor the curvature RELATIVE to the amplitude scale everywhere.
    bl = jnp.where(use_series, 1.0, b)
    dl = jnp.where(use_series, 0.1, d)
    h_floor = 1e-6 * (bl + 4.0 * dl)

    def fval(u):
        return bl * jnp.cos(u - beta) + dl * jnp.cos(2.0 * u - delta)

    def fp(u):
        return -bl * jnp.sin(u - beta) - 2.0 * dl * jnp.sin(2.0 * u - delta)

    def fpp(u):
        return -bl * jnp.cos(u - beta) - 4.0 * dl * jnp.cos(2.0 * u - delta)

    def _guard(H):
        # sign-preserving denominator floor
        return jnp.where(jnp.abs(H) >= h_floor, H,
                         jnp.where(H >= 0, h_floor, -h_floor))

    terms = []
    for u0 in (beta, beta + jnp.pi):
        # value-only Newton (fixed count: quadratic convergence, not a knob)
        # under stop_gradient, then ONE differentiable polish step -- Newton is
        # a contraction, so a single step from the converged point carries the
        # correct implicit derivative without an 8-deep 1/H^2 gradient chain.
        u = u0
        for _ in range(8):
            u = u - fp(u) / _guard(fpp(u))
        u = jax.lax.stop_gradient(u)
        u = u - fp(u) / _guard(fpp(u))
        H = fpp(u)
        ok = H < 0
        Hm = jnp.minimum(H, -h_floor)              # bounded away from 0
        t = jnp.where(ok,
                      a + fval(u)
                      + 0.5 * jnp.log(2.0 * jnp.pi / (-Hm))
                      - jnp.log(2.0 * jnp.pi),      # (1/2 du/dpsi) * (1/pi)
                      -jnp.inf)
        terms.append(t)
    # guarded log-add-exp: jnp.logaddexp(-inf, -inf) has a NaN backward pass
    # (exp(t - ans) with t = ans = -inf), and bins where BOTH stationary
    # points are rejected do occur; the NaN then leaks through jnp.where's
    # chain rule into every gradient.
    t0, t1 = terms
    mt = jnp.maximum(t0, t1)
    mts = jnp.where(jnp.isfinite(mt), mt, 0.0)
    ssum = jnp.exp(t0 - mts) + jnp.exp(t1 - mts)
    ln_laplace = jnp.where(ssum > 0,
                           mts + jnp.log(jnp.maximum(ssum, 1e-300)),
                           -jnp.inf)

    # small-amplitude branch: I0(z) ~ 1 + z^2/4 + z^4/64, I1 ~ z/2 + z^3/16,
    # I2 ~ z^2/8 (arguments < 0.5 here, truncation < 1e-5)
    i0b = 1.0 + b * b / 4.0 + b ** 4 / 64.0
    i0d = 1.0 + d * d / 4.0 + d ** 4 / 64.0
    i2b = b * b / 8.0
    i1d = d / 2.0 + d ** 3 / 16.0
    series = i0b * i0d + 2.0 * i2b * i1d * jnp.cos(2.0 * beta - delta)
    ln_series = a + jnp.log(jnp.maximum(series, 1e-300))

    return jnp.where(use_series, ln_series, ln_laplace)


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

    if amp_sizing is None:
        amp_sizing = ANGLE_MARG_CROSSOVER_AMPLITUDE
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


def choose_angle_marg_scheme(guess_snr, gh_enabled=None):
    """Select 'exact' or 'laplace' from the run's SNR estimate.

    The crossover is the amplitude A = rho^2/2 = ANGLE_MARG_CROSSOVER_AMPLITUDE
    where both schemes are accurate (see the constant's derivation note): the
    exact scheme's dense grid is sized to cover exactly up to the crossover
    (so its cost is bounded and its accuracy guaranteed on its branch), and
    the Laplace O(1/A) error is already negligible there and shrinks upward.

    Returns ``(scheme, info)`` where ``info`` is a provenance dict the caller
    MUST surface in the run log (this pipeline has a documented history of
    silently-inert flags).
    """
    if gh_enabled is None:
        gh_enabled = _core._DISTMARG_GH_N > 0
    if guess_snr is None:
        return "exact", dict(reason="no SNR estimate; exact scheme is valid "
                                    "at all amplitudes (grid sized for the "
                                    "crossover)", guess_snr=None,
                             amplitude=None,
                             crossover=ANGLE_MARG_CROSSOVER_AMPLITUDE)
    amp = 0.5 * float(guess_snr) ** 2
    if gh_enabled:
        return "exact", dict(reason="JAX_ILE_DISTMARG_GH set: laplace does "
                                    "not support the adaptive distance "
                                    "quadrature", guess_snr=float(guess_snr),
                             amplitude=amp,
                             crossover=ANGLE_MARG_CROSSOVER_AMPLITUDE)
    scheme = "laplace" if amp >= ANGLE_MARG_CROSSOVER_AMPLITUDE else "exact"
    return scheme, dict(reason="amplitude %s crossover"
                               % ("above" if scheme == "laplace" else "below"),
                        guess_snr=float(guess_snr), amplitude=amp,
                        crossover=ANGLE_MARG_CROSSOVER_AMPLITUDE)
