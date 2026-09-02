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
:func:`core._distmarg_gh_logL` when JAX_ILE_DISTMARG_GH is set; the laplace
scheme cannot call that function, whose nodes are placed per FIXED psi, and
uses the psi-MARGINAL placement documented above
:func:`_gh_psi_node_offsets` instead -- for m_max <= 2 only, raising above
it), and use the same normalization convention (mean over uniform
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
from .core import (JAX_INTERP_DEFAULT, TIME_QUAD_DEFAULT, _accumulate_unit,
                   _time_marginalize_terminal,
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


def _dense_grid_sizes(amp, m_max=2):
    """(nphi_d, nu_d) dense reconstruction sizes adequate for amplitude ``amp``
    and mode content ``m_max``.

    Derived from the trapezoid aliasing error of exp(trig poly of amplitude
    A): N = K*sqrt(A) with the calibrated constants above (>= 2x margin,
    calibrated at m_max = 2) and hard floors.  The phi axis ADDITIONALLY
    scales linearly with m_max: the exponent's phi content extends to
    harmonic ~(2*m_max)*sqrt(A)-ish, so amplitude alone under-resolves
    higher modes (external review 3, P1: a pure order-8 term at A = 450 was
    phase-dependently wrong by ~0.037 nats, order 16 by up to 1.2 nats,
    under the amplitude-only rule).  The u axis never scales with m_max --
    psi enters at spin-2 for every mode.  This is NOT a settable knob;
    callers pass the DATA-DERIVED amplitude bound from
    :func:`estimate_angle_amplitude` (floored at the crossover) and the
    data's m_max.
    """
    amp = max(float(amp), 25.0)
    m_scale = max(1.0, float(m_max) / 2.0)     # calibration point is m_max=2
    n_u = max(_DENSE_FLOOR_U, int(np.ceil(_DENSE_K_U * np.sqrt(amp))))
    n_phi = max(int(np.ceil(_DENSE_FLOOR_PHI * m_scale)),
                int(np.ceil(_DENSE_K_PHI * m_scale * np.sqrt(amp))))
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

    THIS IS AN ESTIMATOR, NOT A PROVEN BOUND (review 3, P2): the sky
    maximum is located by sampling, and no finite sample proves a maximum
    was not missed.  Three mechanisms stand behind it instead of a
    bound-shaped claim:

    1. the sky sample is random draws PLUS deterministic extremes (the
       inclination poles and a coarse uniform sky grid, which cover the
       known response maximizers);
    2. a split-half convergence check: if the second half of the sample
       moves the running maximum by more than 20%, the sample is doubled
       (up to twice) and the growth is printed -- an under-sampled sky
       variation is detected empirically rather than assumed away;
    3. a RUNTIME fail-safe in the fused functions: every jitted likelihood
       call recomputes the amplitude its own coefficient tables reach and
       prints a loud warning if it exceeds the amp_sizing the dense grids
       were built for -- so an underestimate is DETECTED at the point of
       use, with the recourse (rebuild with the reported amplitude) named
       in the message.  The failure mode is never silent.

    A second, analytic expression max_x (x*M_A - x^2/2*B0)+ (M_A = sum
    w|C_A| pointwise-bounds |A|; B0 = angular mean of B) is kept as a
    build-time cross-check: it pairs the max of A with the MEAN of B, which
    review 2 already noted is heuristic (B can dip below its mean where A
    peaks); empirically it over-reads by 1.5-1.9x, and if it ever reads
    BELOW the empirical max the disagreement is printed.

    Returns ``margin`` times the empirical max, UNfloored: the auto selector
    compares it to the crossover (a floor here would push every quiet target
    into the laplace branch); the WRAPPER floors the SIZING amplitude at the
    crossover separately, so grids are never sized below the calibration
    point.
    """
    x = np.asarray(x_grid)
    x_min, x_max = float(x.min()), float(x.max())

    def _per_sky_amps(ra, dec, incl):
        """Per-sky-point empirical amplitude maxima from exact reconstruction."""
        C_A, C_B, meta = angle_coefficient_tables(data, ra, dec, incl,
                                                  interp=interp)
        C_A = np.asarray(C_A)
        C_B = np.asarray(C_B)
        # dense angular reconstruction matrices (numpy mirror of
        # _reconstruct_field).  The grid is DERIVED from the mode content
        # and ASSERTED, exactly like the sample grid (fail-closed): sampled
        # at >= 8x per highest harmonic, a band-limited trig polynomial's
        # grid max under-reads its continuum max by < 1% (absorbed in
        # `margin`), while a coarser grid can miss an adversarially-phased
        # n = 2*m_max harmonic entirely -- so a too-small grid is refused,
        # not trusted.  _n_phi_e/_n_u_e are test hooks; production callers
        # take the derived defaults.
        m_max_e = meta["m_max"]
        n_phi_e = int(_n_phi_e) if _n_phi_e is not None \
            else max(96, 16 * (2 * m_max_e))
        n_u_e_ = int(_n_u_e)
        assert n_phi_e >= 8 * (2 * m_max_e), \
            "estimator phi grid %d under-samples 2*m_max=%d content" \
            % (n_phi_e, 2 * m_max_e)
        assert n_u_e_ >= 8 * 2, \
            "estimator u grid %d under-samples order-2 content" % (n_u_e_,)
        PH, UU = np.meshgrid(
            np.linspace(0, 2 * np.pi, n_phi_e, endpoint=False),
            np.linspace(0, 2 * np.pi, n_u_e_, endpoint=False),
            indexing="ij")
        phis, us = PH.ravel(), UU.ravel()

        def _recon_matrix(KP, KS):
            kp = np.arange(KP)
            ks = np.arange(-KS, KS + 1)
            E = np.exp(1j * (phis[:, None, None] * kp[None, :, None]
                             + us[:, None, None] * ks[None, None, :]))
            w = np.ones(KP)
            w[1:] = 2.0
            return (E * w[None, :, None]).reshape(len(phis), -1)

        E_A = _recon_matrix(C_A.shape[0], (C_A.shape[1] - 1) // 2)
        E_B = _recon_matrix(C_B.shape[0], (C_B.shape[1] - 1) // 2)
        amps = []
        for j in range(C_A.shape[2]):      # per-sky loop bounds the transient
            A_g = (E_A @ C_A[:, :, j].reshape(-1, C_A.shape[-1])).real
            B_g = np.maximum(
                (E_B @ C_B[:, :, j].reshape(-1, C_B.shape[-1])).real, 0.0)
            # B >= 0 makes x*A - x^2/2*B concave in x: the max over the
            # actual support is at the clipped stationary point
            x_hat = np.clip(A_g / np.maximum(B_g, 1e-300), x_min, x_max)
            val = x_hat * A_g - 0.5 * np.square(x_hat) * B_g
            amps.append(max(float(val.max()), 0.0))
        return np.array(amps), C_A, C_B

    def _draw(n, rng):
        ra = rng.uniform(0.0, 2.0 * np.pi, n)
        dec = np.arcsin(rng.uniform(-1.0, 1.0, n))
        incl = np.arccos(rng.uniform(-1.0, 1.0, n))
        return ra, dec, incl

    rng = np.random.default_rng(seed)
    ra, dec, incl = _draw(n_sky, rng)
    # deterministic extremes: face-on/face-off inclinations over a coarse
    # uniform sky grid (the known maximizers of the response amplitude)
    g_ra, g_dec = np.meshgrid(np.linspace(0, 2 * np.pi, 6, endpoint=False),
                              np.array([-1.0, -0.35, 0.35, 1.0]),
                              indexing="ij")
    for i0_ in (0.0, np.pi):
        ra = np.concatenate([ra, g_ra.ravel()])
        dec = np.concatenate([dec, g_dec.ravel()])
        incl = np.concatenate([incl, np.full(g_ra.size, i0_)])

    amps, C_A, C_B = _per_sky_amps(ra, dec, incl)
    # split-half convergence check (mechanism 2 of the docstring): compare
    # the max WITHOUT the second half of the random draws against the max
    # with them; growth > 20% means the sky variation is under-sampled, so
    # draw again (at most twice) and say so.
    half = np.concatenate([amps[: n_sky // 2], amps[n_sky:]])   # + extremes
    amp_emp = float(amps.max())
    amp_ref = float(half.max())
    grows = amp_emp > 1.2 * amp_ref + 1e-12
    n_extra = 0
    while grows and n_extra < 2 * n_sky:
        print("estimate_angle_amplitude: sky maximum still growing "
              "(%.4g -> %.4g); doubling the sample." % (amp_ref, amp_emp))
        ra2, dec2, incl2 = _draw(n_sky, rng)
        amps2, _, _ = _per_sky_amps(ra2, dec2, incl2)
        amp_ref = amp_emp
        amp_emp = max(amp_emp, float(amps2.max()))
        grows = amp_emp > 1.2 * amp_ref + 1e-12
        n_extra += n_sky

    # analytic cross-check (mechanism documented above; heuristic direction)
    w = np.ones(C_A.shape[0])
    w[1:] = 2.0
    M_A = np.einsum("k,kqst->st", w, np.abs(C_A))
    ks0 = (C_B.shape[1] - 1) // 2
    B0 = np.maximum(C_B[0, ks0].real, 0.0)
    expo = (x[:, None, None] * M_A[None]
            - 0.5 * np.square(x)[:, None, None] * B0[None])
    amp_analytic = float(np.clip(expo, 0.0, None).max())
    if amp_analytic < amp_emp * (1.0 - 1e-9):
        print("estimate_angle_amplitude: analytic cross-check %.6g fell "
              "BELOW the empirical max %.6g (the review-flagged heuristic "
              "direction); the empirical value governs." % (amp_analytic,
                                                            amp_emp))
    return margin * amp_emp


_AMP_FAILSAFE = {"tripped": False, "n_calls": 0, "worst_amp": 0.0,
                 "amp_sizing": None, "scheme": None}


def reset_amp_failsafe():
    """Clear the undersizing record (call once per event, before sampling).

    Barriers first: an in-flight callback from the PREVIOUS event must not land
    after the reset and mislabel this one.
    """
    try:
        jax.effects_barrier()
    except Exception:
        pass
    _AMP_FAILSAFE.update(tripped=False, n_calls=0, worst_amp=0.0,
                         amp_sizing=None, scheme=None)


def amp_failsafe_state(barrier=True):
    """Host-side record of whether the dense grids were ever undersized.

    ``barrier=True`` calls :func:`jax.effects_barrier` first, so queued debug
    callbacks have landed before the record is read.  Without it a caller can
    read CLEAN while a tripped callback is still in flight, or reset for the
    next event before the previous event's callback arrives.

    Returns a dict; ``tripped`` is the load-bearing field.  Consumers should
    LABEL their output rather than discard it -- see the note in
    :func:`_runtime_amp_failsafe` about why this is not fatal and not a NaN.
    """
    if barrier:
        try:
            jax.effects_barrier()
        except Exception:
            pass
    return dict(_AMP_FAILSAFE)


def _record_amp_failsafe(tripped, amp_call, amp_sizing, scheme_name):
    """Host callback.  Runs outside the traced graph; never alters a value."""
    _AMP_FAILSAFE["n_calls"] += 1
    if bool(tripped):
        _AMP_FAILSAFE["tripped"] = True
        _AMP_FAILSAFE["worst_amp"] = max(_AMP_FAILSAFE["worst_amp"], float(amp_call))
        _AMP_FAILSAFE["amp_sizing"] = float(amp_sizing)
        _AMP_FAILSAFE["scheme"] = scheme_name


def _runtime_amp_failsafe(C_A, C_B, x_grid, amp_sizing, scheme_name):
    """Mechanism 3 of :func:`estimate_angle_amplitude`'s contract: DETECT, at
    the point of use, a call whose coefficient tables reach amplitudes the
    dense grids were not sized for (i.e. the build-time estimator missed a
    hotter sky region -- it is an estimator, not a proven bound).

    Uses the cheap analytic expression max over (S, t) of the concave-in-x
    maximum of x*M_A - x^2/2*B0 (closed form, no distance-grid axis).  That
    expression OVER-reads the true amplitude by a measured 1.5-1.9x, so the
    trigger threshold is 2*amp_sizing: it fires when the true local
    amplitude exceeds ~1.3-2x the sizing bound -- comfortably BEFORE the
    dense grids actually degrade (their calibrated constants carry a 2x
    margin in N, i.e. 4x in amplitude).  The warning prints from inside jit
    via jax.debug.print (no value is altered; the recourse is named in the
    message).  Everything under stop_gradient: the check must not appear in
    the AD graph.
    """
    w = _kp_weights(C_A.shape[0])
    M_A = jnp.einsum("k,kqst->st", jnp.asarray(w), jnp.abs(C_A))
    ks0 = (C_B.shape[1] - 1) // 2
    B0 = jnp.maximum(C_B[0, ks0].real, 0.0)
    x_min = jnp.min(x_grid)
    x_max = jnp.max(x_grid)
    x_hat = jnp.clip(M_A / jnp.maximum(B0, 1e-300), x_min, x_max)
    amp_call = jnp.max(jnp.clip(
        x_hat * M_A - 0.5 * jnp.square(x_hat) * B0, 0.0, None))
    amp_call = jax.lax.stop_gradient(amp_call)
    # FAIL CLOSED.  A warning printed from inside jit does not stop anything:
    # a production run would finish and publish biased likelihoods, samples and
    # evidence while the "fail-safe" scrolled past in a log.  So in addition to
    # the message we return a POISON term the caller ADDS to its result, making
    # the output non-finite.  A NaN lnL cannot be silently consumed -- samplers
    # reject or abort on it -- whereas an under-resolved finite number is
    # indistinguishable from a good one.  Kept under stop_gradient so the check
    # never enters the AD graph.
    jax.lax.cond(
        amp_call > 2.0 * amp_sizing,
        lambda a_: jax.debug.print(
            "WARNING anglemarg/" + scheme_name + ": this call's coefficient "
            "tables reach an amplitude scale ~{a:.4g} (analytic over-reading "
            "expression), above 2x the amp_sizing=" + "%.4g" % amp_sizing
            + " the dense (phi,psi) grids were built for.  "
            "estimate_angle_amplitude underestimated the sky maximum; the "
            "marginal may be under-resolved at such points.  Rebuild the "
            "likelihood with amp_sizing >= the reported amplitude.", a=a_),
        lambda a_: None,
        amp_call)
    # DELIBERATELY NOT FATAL, AND DELIBERATELY NOT A NaN.
    #
    # An earlier version returned NaN to "fail closed".  That was worse than the
    # warning it replaced: every consumer of this likelihood FILTERS non-finite
    # values -- flowMC/MALA reject such proposals as invalid, the SMC path drops
    # non-finite lnL, and write_samples discards non-finite rows.  So a NaN over
    # a hot sky region the estimator missed does not stop anything; it silently
    # EXCISES exactly that region and publishes a clean-looking posterior over
    # what remains.  Invisible mutilation beats visible failure only from the
    # code's point of view, never from the operator's.
    #
    # Aborting is also wrong here: this is a configuration estimate, and hard
    # failure would destroy a multi-hour run over a recoverable condition.
    #
    # So: the value is untouched, the run completes, and the condition is
    # recorded on the HOST so the driver can LABEL the result as suspect in its
    # provenance.  A labelled result an operator can judge beats both a vanished
    # region and a dead run.
    # The callback sits INSIDE lax.cond so the ORDINARY path has no host
    # callback at all.  An unconditional callback fires once per likelihood
    # evaluation -- once per MALA/flowMC proposal, per chain -- transferring to
    # the host and destroying accelerator throughput even when undersizing never
    # happens.  Only the rare tripped branch pays.
    #
    # Reliability caveat, stated because it bounds what this record can be used
    # for: jax.debug.callback effects may be dropped, duplicated or reordered
    # under transformation, and may land AFTER the result is ready.  So this is
    # a best-effort DIAGNOSTIC LABEL, not a correctness gate -- consumers must
    # call jax.effects_barrier() before reading or resetting the state, and must
    # not treat a clean read as proof of adequacy.
    jax.lax.cond(
        amp_call > 2.0 * amp_sizing,
        lambda a_: jax.debug.callback(
            _record_amp_failsafe, True, a_,
            jnp.asarray(amp_sizing, dtype=jnp.float64), scheme_name),
        lambda a_: None,
        amp_call)


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
        dense_chunk=8, grid_block=32,
        time_quadrature=TIME_QUAD_DEFAULT, return_lnLt=False):
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
    _runtime_amp_failsafe(C_A, C_B, x_grid, amp_sizing, "exact")
    nphi_d, nu_d = _dense_grid_sizes(amp_sizing, m_max=meta["m_max"])
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
    if return_lnLt:
        return lnL_t
    return _time_marginalize_terminal(lnL_t, data, time_quadrature)


# ---------------------------------------------------------------------------
# Analytic psi Laplace
# ---------------------------------------------------------------------------

# Quadrature/Laplace handover (external reviews, items 2 and P2-local).
#
# HISTORY, because two revisions got this boundary wrong in two different
# ways: a hard series/Laplace switch at b + 2d = 0.5 left a wide window of
# 0.27-0.53 nats value error with a SIGN-INVERTED |c2| gradient just above
# the cut (review 2, item 2); a truncated-Bessel-series fix moved the
# handover to b + 2d ~ 16, but a third review showed the subdominance
# argument used to tolerate the remaining O(0.25)-nat worst-phase Laplace
# error near the handover is GLOBAL while the kernel is evaluated at every
# proposed sky position -- at a low-response proposal such bins are locally
# dominant (counterexample: b = 11.887, d = 3.516, beta = -1.890,
# delta = -0.650 -> +0.251 nats).
#
# The resolution: below the handover the u-integral is done by FIXED-N
# trapezoid quadrature of exp(f) -- for a periodic band-limited exponent the
# trapezoid rule converges super-exponentially, so with N = 320 the branch
# is machine-accurate for every t = b + 2d <= BLEND_HI = 300 (aliasing
# ~ I_{N/2}(t)/I_0(t) ~ e^-40 at the band edge; the counterexample bin is
# now exact).  N is fixed by the HANDOVER amplitude, not by the data, so
# the laplace scheme keeps its ~sqrt(A) cost scaling.  Above the band the
# enumerated-maxima Laplace applies, where its worst-phase error
# (~3.9/(b+d), adversarial constant from review 3) is <= ~0.026 and typical
# error is ~1e-3, falling as 1/A.  The two are blended C^1-smoothly over
# [BLEND_LO, BLEND_HI]: the blend gradient carries dw * (quad - laplace),
# i.e. the branch disagreement, which the band placement bounds.
_LAPLACE_BLEND_LO = 220.0     # pure quadrature below this in t = b + 2d
_LAPLACE_BLEND_HI = 300.0     # pure Laplace above this
_LAPLACE_QUAD_N = 320         # u-quadrature points; content of exp(f) extends
                              # to ~2*5.25*sqrt(t/2) harmonics (d-dominated
                              # worst case) = 128 at t = 300, Nyquist 160
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


def _psi_lnI_amplitudes(c1, c2):
    """(b, d, t_amp) for the kernel and the block dispatcher: harmonic
    magnitudes and the blend variable t = b + 2 d.  Factored out so the
    dispatcher can bound t without tracing either branch."""
    mag1 = jnp.square(c1.real) + jnp.square(c1.imag)
    mag2 = jnp.square(c2.real) + jnp.square(c2.imag)
    b = jnp.sqrt(mag1 + 1e-300)
    d = jnp.sqrt(mag2 + 1e-300)
    return b, d, b + 2.0 * d


def _psi_lnI_lap_branch(a, c1, c2, b, t_amp):
    """The enumerated-maxima Laplace branch of :func:`_laplace_psi_lnI`
    (verbatim code motion; see that docstring for the contract).  Returns
    ln_laplace; -inf is impossible for a finite integral by the tolerant
    acceptance, and hypothetical nonfinite values are guarded by callers."""
    lap_dummy = t_amp < _LAPLACE_BLEND_LO      # blend weight is exactly 1 here
    # jnp.where's VJP sends a ZERO cotangent through the unselected branch,
    # and 0 * inf = nan: the Laplace branch must have BOUNDED derivatives
    # (including second, for .fisher()) even on the pure-quadrature bins.
    # Feed it safe dummy amplitudes there (their contribution is weighted 0
    # by the blend) and floor the curvature RELATIVE to the amplitude scale.
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

    def fpppp(u):
        eiu = jnp.exp(1j * u)
        return (c1l * eiu).real + 16.0 * (c2l * eiu * eiu).real

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
    #
    # STRUCTURE, not mathematics: the cell walk is a lax.scan and the slot
    # registers are one stacked (_LAPLACE_MAX_ROOTS, X) array rather than a
    # Python loop over cells and slots.  The elementwise updates are the
    # same; the Python version unrolled ~24 x 4 select chains into the traced
    # graph, and this kernel is instantiated once per distance block, which
    # multiplied that unroll into an XLA graph that took over an hour (and
    # >20 GiB) to compile at production sizes.  Same fix pattern below for
    # the bisection and the fixed-N quadrature.
    N = _LAPLACE_BRACKET_CELLS
    ug = np.linspace(0.0, 2.0 * np.pi, N + 1)
    cell = ug[1] - ug[0]
    # a, c1 and c2 may have different but broadcast-compatible shapes.  The
    # scan carry must start at their combined shape: lax.scan forbids the
    # Python-loop behaviour of expanding a scalar carry on the first step.
    zero_f = jnp.zeros_like(t_amp)
    false_x = jnp.zeros_like(t_amp, dtype=bool)
    nR = _LAPLACE_MAX_ROOTS
    slot_ids = jnp.arange(nR, dtype=zero_f.dtype).reshape(
        (nR,) + (1,) * zero_f.ndim)
    s_prev0 = fp(jnp.asarray(ug[0])) >= 0
    los0 = jnp.stack([zero_f] * nR)
    s_los0 = jnp.stack([false_x] * nR)
    filled0 = jnp.stack([false_x] * nR)

    def _bracket_step(carry, edges):
        s_prev, count, los, s_los, filled = carry
        u_left, u_right = edges
        s_next = fp(u_right) >= 0
        flip = s_prev != s_next
        take = flip[None] & (count[None] == slot_ids)
        los = jnp.where(take, u_left, los)
        s_los = jnp.where(take, s_prev[None], s_los)
        filled = filled | take
        count = count + flip.astype(count.dtype)
        return (s_next, count, los, s_los, filled), None

    (_, _, los, s_los, filled), _ = jax.lax.scan(
        _bracket_step, (s_prev0, zero_f, los0, s_los0, filled0),
        (jnp.asarray(ug[:-1]), jnp.asarray(ug[1:])))

    # ---- per-slot bisection (value-only) + one differentiable polish step
    # (batched over the slot axis; the 20 halvings are a fori_loop)
    slo = s_los
    def _bisect_step(_, lohi):        # cell/2^20 ~ 2.5e-7, then Newton
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        go_right = (fp(mid) >= 0) == slo
        return (jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid))
    lo, hi = jax.lax.fori_loop(0, 20, _bisect_step, (los, los + cell))
    u0 = jax.lax.stop_gradient(0.5 * (lo + hi))
    u = u0 - fp(u0) / _guard(fpp(u0))
    H = fpp(u)
    # tolerant acceptance: a maximum with H in [-h_floor, +h_floor) is a
    # (near-)degenerate flat top; drop it and a finite integral could
    # come back -inf, so keep it with the floored curvature instead
    # (its Laplace weight is then merely inaccurate, never absent).
    ok = filled & (H < h_floor)
    Hm = jnp.minimum(H, -h_floor)
    # Peak width: the Gaussian factor sqrt(2 pi/|H|) OVERESTIMATES a
    # near-degenerate (quartic-flat) maximum by nats -- at the exactly
    # aligned b = 4d configuration the floored-curvature form was ~5
    # nats high (review 3's local-error standard).  The quartic width
    # int exp(f4 u^4/24) du = Gamma(1/4)/2 * (24/|f4|)^(1/4) is closed
    # form (f'''' is elementary for a trig polynomial).  The widths are
    # combined as W = W_g (1 + rho)^-1/2 with rho = (W_g/W_q)^2 -- exact
    # at both ends and at most 0.083 nats off on the scale-free
    # Gaussian x quartic family (vs 0.26 for min() and ~5 for the
    # floored Gaussian alone) -- but GATED on rho: for a REGULAR
    # maximum rho ~ 1/sqrt(b) and the ungated correction would inject
    # an O(1/sqrt(A)) systematic where plain Laplace errs only O(1/A)
    # (measured: sweep worst at t = 1000 rose 3.7e-3 -> 4.6e-2
    # ungated).  The gate turns the correction on smoothly over
    # rho in [0.2, 0.8], i.e. only where the peak is genuinely
    # quartic-contaminated.
    F4 = fpppp(u)
    f4_floor = 1e-6 * (bl + 16.0 * dl)
    F4m = jnp.minimum(F4, -f4_floor)
    lnW_gauss = 0.5 * jnp.log(2.0 * jnp.pi / (-Hm))
    lnW_quart = 0.5949217316 + 0.25 * jnp.log(24.0 / (-F4m))
    rho = jnp.exp(jnp.clip(2.0 * (lnW_gauss - lnW_quart), -50.0, 50.0))
    g8 = jnp.clip((rho - 0.2) / 0.6, 0.0, 1.0)
    g8 = g8 * g8 * (3.0 - 2.0 * g8)
    lnW = lnW_gauss - 0.5 * jnp.log1p(rho * g8)
    terms = jnp.where(ok,
                      a + fval(u) + lnW
                      - jnp.log(2.0 * jnp.pi),      # (1/2 du/dpsi) * (1/pi)
                      -jnp.inf)                     # (nR,) + X

    # guarded log-add-exp over the root slots: an all--inf slot set has a NaN
    # backward pass under the naive form, and the NaN leaks through jnp.where.
    # Explicit left fold (not jnp.max/sum) preserves the pre-restructure
    # accumulation order bit for bit.
    mt = terms[0]
    for j in range(1, nR):
        mt = jnp.maximum(mt, terms[j])
    mts = jnp.where(jnp.isfinite(mt), mt, 0.0)
    ssum = zero_f
    for j in range(nR):
        ssum = ssum + jnp.exp(terms[j] - mts)
    ln_laplace = jnp.where(ssum > 0,
                           mts + jnp.log(jnp.maximum(ssum, 1e-300)),
                           -jnp.inf)
    return ln_laplace


def _psi_lnI_quad_branch(a, c1, c2, b, n_quad):
    """The fixed-N trapezoid u-quadrature branch of :func:`_laplace_psi_lnI`
    (verbatim code motion, N parameterized; n_quad must be a multiple of the
    16-point scan chunk).  Bit-identical to the pre-split code at
    n_quad = _LAPLACE_QUAD_N.
    """
    # ---- fixed-N u-quadrature branch: mean of exp(f) over a uniform u grid
    # equals (1/pi) int dpsi.  Uses the TRUE c1, c2 (no dummies needed: no
    # divisions, and the running-max log-sum-exp keeps exp() in range even
    # for the huge-t bins whose blend weight is 0).  Chunked (as a lax.scan
    # over precomputed host phase tables, one traced body instead of
    # n_quad unrolled evaluations) so the transient stays a few
    # X-sized arrays.
    uq = np.linspace(0.0, 2.0 * np.pi, n_quad, endpoint=False)
    QCH = 16
    e1 = np.exp(1j * uq)                       # host phases, as before
    e2 = e1 * e1                               # == eiu * eiu elementwise
    p1 = jnp.asarray(e1.reshape(-1, QCH))      # (nq, QCH)
    p2 = jnp.asarray(e2.reshape(-1, QCH))
    bshape = jnp.broadcast_shapes(jnp.shape(c1), jnp.shape(c2))
    pshape = (QCH,) + (1,) * len(bshape)

    def _quad_step(carry, phases):
        mq, sq = carry
        p1k, p2k = phases                      # (QCH,) complex
        blk = ((c1[None] * p1k.reshape(pshape)).real
               + (c2[None] * p2k.reshape(pshape)).real)
        return _lse_update(mq, sq, blk, axis=0), None

    mq0 = jnp.full(bshape, -jnp.inf, dtype=b.dtype)
    sq0 = jnp.zeros(bshape, dtype=b.dtype)
    (mq, sq), _ = jax.lax.scan(_quad_step, (mq0, sq0), (p1, p2))
    ln_quad = (a + mq + jnp.log(jnp.maximum(sq, 1e-300))
               - jnp.log(float(n_quad)))
    return ln_quad


def _laplace_psi_lnI(a, c1, c2):
    """log[(1/pi) int_0^pi exp(a + Re(c1 e^{iu}) + Re(c2 e^{2iu})) dpsi], u = 2 psi.

    Two regimes, C^1-blended on t = b + 2d (b = |c1|, d = |c2|); see the
    constants block above for the placement rationale and review history.

    t < BLEND_HI: fixed-N trapezoid quadrature of exp(f) over u -- machine-
    accurate for a periodic band-limited exponent up to the handover, which
    is what makes the kernel's LOCAL error small at every reachable bin (a
    global-amplitude subdominance argument is not available: the kernel runs
    at every proposed sky position, review 3).

    t > BLEND_LO: Laplace's method with ALL maxima enumerated.  An early
    revision seeded Newton only at the extrema of the FIRST harmonic, which
    fails outright when that harmonic cancels (c1 = 0, c2 = -d: both seeds
    are minima; -inf was returned for a finite integral -- review 1).  Every
    transversal zero of f' is bracketed by a sign scan (interval-based, so
    coincident roots cannot be double-counted), bisected under
    stop_gradient, polished by one differentiable Newton step (a contraction
    step from the converged point carries the implicit derivative without a
    deep 1/H^2 gradient chain); near-degenerate maxima are kept with floored
    curvature rather than dropped, so -inf is impossible for a finite
    integral.  Angle-free throughout: f, f', f'' are evaluated directly from
    c1, c2, so arg(0) never appears and b = 0 is a regular point.

    Elementary functions only (no scipy, no eigensolvers); differentiable;
    any input shape (elementwise over broadcast a, c1, c2).
    """
    # Materialize the documented elementwise broadcast before introducing
    # the leading root-slot axis.  Otherwise a data axis contributed only by
    # ``a`` can collide with the four-root axis (silently when its length is
    # four, or as a shape error for any other length).
    a, c1, c2 = jnp.broadcast_arrays(a, c1, c2)
    b, d, t_amp = _psi_lnI_amplitudes(c1, c2)
    ln_laplace = _psi_lnI_lap_branch(a, c1, c2, b, t_amp)
    ln_quad = _psi_lnI_quad_branch(a, c1, c2, b, _LAPLACE_QUAD_N)
    # ---- C^1 blend: pure quadrature below LO, pure Laplace above HI
    r = jnp.clip((_LAPLACE_BLEND_HI - t_amp)
                 / (_LAPLACE_BLEND_HI - _LAPLACE_BLEND_LO), 0.0, 1.0)
    wgt = r * r * (3.0 - 2.0 * r)               # smoothstep
    # ln_laplace cannot be -inf for t_amp >= LO (a periodic f' has >= 2 sign
    # flips and the tolerant acceptance keeps the global maximum), but guard
    # the 0-weight product against a hypothetical -inf anyway.
    ln_lap = jnp.where(jnp.isfinite(ln_laplace), ln_laplace, ln_quad)
    return wgt * ln_quad + (1.0 - wgt) * ln_lap


# ---------------------------------------------------------------------------
# Block-dispatched execution of the kernel (2026-08-28 execution-cost fix).
#
# The C^1 blend evaluates BOTH branches at every lattice point, and the
# quadrature is sized for the handover amplitude (N = 320 at t = 300)
# regardless of the local t.  Measured on the production-shaped lattice
# (amp_sizing ~ 1109, SNR-40 scale), 99.5% of (distance x dense-phi x sample
# x time) points sit at t < BLEND_LO -- 89% at t < 20 -- while every point
# that carries posterior weight sits at t > 900: each branch does needed
# work on a small, DISJOINT part of the lattice, yet both were paid
# everywhere (quad 55% / root-finding 39% of execution, additively).
#
# Per-point branching cannot save work under SIMD (select evaluates both
# sides), but the fused driver already evaluates the kernel in blocks
# (dist_block x phi_chunk x S x npts), and a block-level scalar bound on
# t = b + 2d makes the choice discrete via lax.switch (one branch executes):
#   - every point has t >= BLEND_HI  -> Laplace branch only;
#   - every point has t <  BLEND_LO  -> quadrature only (weight is exactly
#     1 and the blend gradient exactly 0 there, so values AND derivatives
#     equal the shipped kernel's), with N from the ladder below;
#   - otherwise -> the full blended kernel, unchanged.
#
# The N ladder applies the SHIPPED sizing rule locally: the aliasing error
# of the N-point trapezoid rule on exp(f) is ~ I_{N/2}(t)/I_0(t), and the
# shipped pair (N = 320, t = 300) fixes the accepted exponent
#   E(N, t) = sqrt(nu^2 + t^2) - nu asinh(nu/t) - t = -41.73   (nu = N/2)
# (the constants block above quotes e^-40 for the same pair).  Each rung's
# threshold t_ok is rounded DOWN from the exact E = -41.73 contour, so every
# rung is at least as accurate as the shipped band edge: relative aliasing
# <= e^-41.7 ~ 8e-19, i.e. below f64 roundoff of the result.  Rungs are
# multiples of the 16-point scan chunk.  Exact contour values:
# N=32: 0.918, 48: 3.583, 64: 8.101, 96: 22.38, 128: 43.27, 160: 70.54,
# 224: 143.8, 320: 300 (test_angle_marg_block_dispatch.py recomputes these).
_QUAD_LADDER_N = (32, 48, 64, 96, 128, 160, 224, 320)
_QUAD_LADDER_TOK = (0.9, 3.5, 8.0, 22.0, 43.0, 70.0, 143.0,
                    _LAPLACE_BLEND_HI)


def _laplace_psi_lnI_block(a, c1, c2):
    """Same value and derivatives as :func:`_laplace_psi_lnI` (see the
    dispatch comment above for the exact equivalence statement), evaluated
    with one lax.switch branch chosen by scalar bounds of t over the WHOLE
    input block.  Intended for the fused driver's per-(distance-block,
    phi-chunk) kernel calls; for pointwise use, call _laplace_psi_lnI.

    Two deliberate differences from the shipped kernel, both confined to
    cases the review history establishes as unreachable or sub-roundoff:
    (1) in the pure-Laplace branch a hypothetical nonfinite ln_laplace
    falls back to ``a`` instead of ln_quad (ln_quad is not computed there;
    the fallback cannot fire for t >= BLEND_LO, see the blend comment);
    (2) pure-quadrature blocks use the ladder N instead of N = 320, with
    relative aliasing <= e^-41.7 at every rung edge (vs e^-41.7 at the
    shipped band edge itself).
    """
    # Keep the data axes distinct from the dispatcher's root-slot axis just
    # as _laplace_psi_lnI does.  The block dispatcher is the production call
    # path, so it must preserve the same three-input broadcast contract.
    a, c1, c2 = jnp.broadcast_arrays(a, c1, c2)
    b, d, t_amp = _psi_lnI_amplitudes(c1, c2)
    tmin = jnp.min(t_amp)
    tmax = jnp.max(t_amp)

    def _pure_lap(_):
        ln_l = _psi_lnI_lap_branch(a, c1, c2, b, t_amp)
        return jnp.where(jnp.isfinite(ln_l), ln_l, a)

    def _quad_rung(nq):
        def _q(_):
            return _psi_lnI_quad_branch(a, c1, c2, b, nq)
        return _q

    def _full(_):
        return _laplace_psi_lnI(a, c1, c2)

    branches = ([_pure_lap] + [_quad_rung(n) for n in _QUAD_LADDER_N]
                + [_full])
    n_rungs = len(_QUAD_LADDER_N)
    rung = jnp.zeros((), dtype=jnp.int32)
    for tok in _QUAD_LADDER_TOK[:-1]:
        rung = rung + (tmax > tok).astype(jnp.int32)
    idx = jnp.where(tmin >= _LAPLACE_BLEND_HI, jnp.int32(0),
                    jnp.where(tmax < _LAPLACE_BLEND_LO,
                              jnp.int32(1) + rung, jnp.int32(1 + n_rungs)))
    return jax.lax.switch(idx, branches, None)


# ---------------------------------------------------------------------------
# psi-marginal node placement for the adaptive distance quadrature
# (JAX_ILE_DISTMARG_GH on the laplace path)
#
# core._distmarg_gh_logL places its frozen nodes at clip(K/R) +- 7/sqrt(R) for a
# FIXED psi.  On this path psi has already been integrated out analytically, so
# the nodes have to bracket the psi-MARGINAL distance integrand
#     I(x) = (1/pi) int dpsi exp(x A(u) - x^2/2 B(u)),  u = 2 psi,
# a mixture over u of Gaussians of centre x*(u) = A(u)/B(u) and width
# 1/sqrt(B(u)).  Two facts make a closed-form bracket sufficient:
#
#   * every component is NARROWER than sigma = 1/sqrt(R_lo) with
#     R_lo = B0 - |B1| - |B2| <= min_u B, so 7 sigma past the extreme
#     component centre covers the mixture to the same 1e-11 the fixed-psi
#     rule covers its single Gaussian;
#   * the component centres that carry weight span a BOUNDED number of sigma.
#
# Both were measured on the ladder-2 injection (35+30 Msun, H1/L1/V1, SEOBNRv4,
# l_max = 2 -> lms {(2,-2),(2,2)}), at the sky points the campaign's own
# posterior occupies, over 235,776 (dense-phi, sample, time) bins per rung:
#
#   rho    W = sqrt(min_u B/R_lo)   C = |x*(u_cf)-x*(u_exact)|/sigma   S span/sigma
#   40.77  median 1.0000 max 1.0000  median 0.0014 p99 0.636 max 0.689  p99 3.899 max 4.085
#   163.1  median 1.0000 max 1.0000  median 0.0090 p99 0.097 max 0.112  p99 0.863 max 0.887
#
# with R_lo <= 0 at 0.0000% of ALL bins at both rungs.  W == 1 is an IDENTITY,
# not a lucky bound: the spin-2 response F(psi) ~ e^{-2 i psi} puts the kappa
# term at exactly one u-harmonic and the rho^2 term at exactly harmonics 0 and
# 2, so A0 and B1 vanish for EVERY mode set -- measured |A0|/|A1| ~ 7e-17 and
# |B1|/|B0| ~ 6e-16 on real IMRPhenomXHM data at m_max = 2, 3 AND 4, and on
# synthetic data with random U/V.  Hence B(u) = B0 + Re(B2 e^{2iu}) and
# R_lo = B0 - |B2| IS min_u B.  The m_max gate below is therefore CONSERVATIVE
# rather than load-bearing for the width; it stands because the half-span
# constant was measured on (2,+-2) fixtures and the higher-mode verdict is
# owned by another session.
#
# CENTRING.  A0 == 0 makes A(u) a pure first harmonic, so the u that maximises
# A is closed form; but the u that maximises the DISTANCE-maximum exponent
# A(u)^2/(2 B(u)) is what the bracket must sit on, and the two part company as
# |B2|/B0 grows.  With A0 = B1 = 0 the stationary condition 2 A' B = A B'
# reduces, in z = e^{iu}, to
#     z^2 (B0 A1 - conj(A1) B2) = conj(B0 A1 - conj(A1) B2)
# so with w = B0*A1 - conj(A1)*B2 the maximiser is EXACTLY
#     e^{i u*} = +- conj(w)/|w|            (sign chosen so A(u*) > 0)
# -- closed form, angle-free, and equal to conj(A1)/|A1| when B2 = 0.  (B has
# only even u-harmonics, so E(u) = E(u+pi) and the two roots of z^2 are the
# same maximum; the other two stationary points are the A = 0 minima, divided
# out.)  Checked against a 400,001-point brute-force argmax on 20,000 random
# (A1, B0, B2) with |B2|/B0 up to 0.99999: the brute force never beats it by
# more than 6.5e-16 relative.  Using argmax A instead costs nothing on the
# ladder (the two centres differ by 0.64 sigma at rho 40.77, 0.10 at 163.08)
# but is catastrophic elsewhere in the family -- see below.
#
# HALF-SPAN.  Because A0 = B1 = 0 hold for ANY mode set, the whole problem
# reduces after scaling to three numbers -- rho = |A1|/sqrt(B0), r = |B2|/B0,
# and the relative phase -- so the bracket can be scanned EXHAUSTIVELY instead
# of sampled on a fixture.  Over 57,082 well-resolved points of that family
# (r up to 0.999, rho 1..1500, 61 phases, v grid 262144, weight threshold 100
# nats), the one-sided reach that the half-span must cover is
#     centre = argmax A            : p50 8.1   p99 2701   MAX 7172   sigma
#     centre = argmax A^2/(2B)     : p50 3.5   p99 13.1   MAX 14.14  sigma
# and the 14.14 = sqrt(2 * 100 nats) bound is attained in the weak-signal
# corner where the 100-nat window is the whole circle.  Hence 7 + 14.14 -> 22.
# On the ladder-2 injection itself the requirement is far smaller (7 + 3.46 =
# 10.5 sigma at rho 40.77, 7.8 at 163.08, 11.5 at 652), so the shipped span is
# ~2x what the operating point needs.  ("Weight-carrying" = within 100 nats of
# the bin's own maximum over u of the clipped exponent; psi below that
# contribute < e^-100, and an under-reaching bracket can only UNDER-estimate
# them, never inflate them, the trapezoid being exponentially accurate at this
# spacing.)
_GH_PSI_HALF_SIGMA = 22.0     # node half-span, in units of sigma = 1/sqrt(R_lo)
_GH_PSI_MIN_NODES = 49        # floor: 44 sigma / 48 gaps = 0.92 sigma spacing,
                              # trapezoid aliasing on a Gaussian ~ 2e^-2pi^2/h^2
                              # = 2e-10 -- below the f64 noise of the result
_GH_PSI_M_MAX = 2             # mode content the path is SHIPPED for.  The
                              # A0 == B1 == 0 identity itself is structural and
                              # measured through m_max = 4, but the higher-mode
                              # verdict is owned elsewhere; keyed on mode
                              # content the way angle_sample_grid_sizes is.


def _gh_psi_node_offsets(n_nodes):
    """Node offsets ``(z, z_prev, z_next, n)`` for the psi-marginal bracket.

    ``z`` spans +-``_GH_PSI_HALF_SIGMA`` instead of the fixed-psi rule's +-7,
    and the count is raised in proportion so the NODE DENSITY the caller asked
    for through JAX_ILE_DISTMARG_GH is preserved rather than diluted by the
    wider bracket (floored at ``_GH_PSI_MIN_NODES``).

    ``z_prev``/``z_next`` are ``z`` with the INDEX clamped at the ends.  The
    composite-trapezoid weight of node k is then 0.5*(x[k+1] - x[k-1]) with the
    same end convention as :func:`core._distmarg_gh_logL`'s
    ``diff``-and-concatenate form -- identical weights, but computable one
    block at a time, so the distance axis stays scanned and memory stays
    bounded by ``dist_block``.
    """
    n = max(int(_GH_PSI_MIN_NODES),
            1 + int(np.ceil((int(n_nodes) - 1) * _GH_PSI_HALF_SIGMA / 7.0)))
    z = np.linspace(-_GH_PSI_HALF_SIGMA, _GH_PSI_HALF_SIGMA, n)
    idx = np.arange(n)
    return (z, z[np.maximum(idx - 1, 0)], z[np.minimum(idx + 1, n - 1)], n)


def fused_log_likelihood_distphipsimarg_laplace(
        data, ra, dec, incl, x_grid, log_w_grid,
        interp=JAX_INTERP_DEFAULT, amp_sizing=None,
        phi_chunk=16, dist_block=4,
        time_quadrature=TIME_QUAD_DEFAULT, return_lnLt=False):
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
    SHRINKS with SNR.

    The adaptive distance quadrature (JAX_ILE_DISTMARG_GH) is honoured for
    ``m_max <= _GH_PSI_M_MAX`` via the psi-marginal node placement documented
    above ``_gh_psi_node_offsets``; ``x_grid``/``log_w_grid`` then only supply
    the support [x_min, x_max] and the prior normalization, exactly as on the
    exact path.  Richer mode content still RAISES rather than being silently
    accepted: the placement rests on an A0 == B1 == 0 identity that is
    established for (2,+-2) only.

    Memory is bounded by ``phi_chunk`` x ``dist_block``, never by grid sizes.
    """
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    C_A, C_B, meta = angle_coefficient_tables(data, ra, dec, incl, interp)
    m_max = meta["m_max"]
    S = ra.shape[0]
    npts = data.npts

    _use_gh = _core._DISTMARG_GH_N > 0
    if _use_gh and int(m_max) > _GH_PSI_M_MAX:
        raise ValueError(
            "JAX_ILE_DISTMARG_GH is set and the 'laplace' angle-marg scheme's "
            "psi-marginal distance-node placement is validated for mode "
            "content m_max <= %d only (it rests on the A0 == B1 == 0 identity "
            "that holds for (2,+-2)); this data has m_max = %d.  Use "
            "--angle-marg-scheme exact, or unset JAX_ILE_DISTMARG_GH."
            % (_GH_PSI_M_MAX, int(m_max)))

    amp_sizing = _require_amp_sizing(amp_sizing)
    # x_grid is still the right argument under GH: the adaptive nodes are
    # CLIPPED into [min x_grid, max x_grid], so the amplitude bound the
    # failsafe computes over x_grid bounds the nodes actually used.
    _runtime_amp_failsafe(C_A, C_B, x_grid, amp_sizing, "laplace")
    nphi_d, _ = _dense_grid_sizes(amp_sizing, m_max=m_max)
    phi_d = np.linspace(0.0, 2.0 * np.pi, nphi_d, endpoint=False)
    c = int(phi_chunk)
    phi_x, lw_x = _pad_chunks([phi_d], c)

    wA = _kp_weights(m_max + 1)
    wB = _kp_weights(2 * m_max + 1)
    kpA = jnp.arange(m_max + 1, dtype=jnp.float64)
    kpB = jnp.arange(2 * m_max + 1, dtype=jnp.float64)
    G = x_grid.shape[0]
    blk = int(dist_block)
    # Distance nodes packed into (n_dblk, blk) for the lax.scan below; the
    # tail block (if G % blk) is edge-padded with -inf log-weights, exactly
    # the _pad_chunks convention, so padded nodes contribute exactly 0 to the
    # running log-sum-exp.  A Python loop here instantiated the FULL
    # _laplace_psi_lnI kernel once per distance block inside the (already
    # checkpointed) phi scan body -- at the production n_grid=256, blk=4 that
    # is 64 copies of an already-large kernel, and XLA compile time/memory on
    # the resulting graph was the >1 h, >20 GiB wall the 2026-08-28 bake-off
    # hit.  The scan traces the kernel ONCE.  Numerics are unchanged.
    n_dblk = (G + blk - 1) // blk
    pad_d = n_dblk * blk - G
    if pad_d:
        x_pad = jnp.concatenate(
            [x_grid, jnp.broadcast_to(x_grid[-1], (pad_d,))])
        lw_pad = jnp.concatenate(
            [log_w_grid, jnp.full((pad_d,), -jnp.inf, dtype=jnp.float64)])
    else:
        x_pad, lw_pad = x_grid, log_w_grid
    xg_blk = x_pad.reshape(n_dblk, blk)
    lwg_blk = lw_pad.reshape(n_dblk, blk)

    if _use_gh:
        # Adaptive nodes replace the static grid entirely; x_grid survives only
        # as the physical support and the (dref-independent) prior norm C0, the
        # same two roles it plays inside core._distmarg_gh_logL.
        x_min = jnp.min(x_grid)
        x_max = jnp.max(x_grid)
        gh_C0 = jnp.log(3.0) - jnp.log(jnp.min(x_grid) ** (-3.0)
                                       - jnp.max(x_grid) ** (-3.0))
        z_np, zp_np, zn_np, n_gh = _gh_psi_node_offsets(_core._DISTMARG_GH_N)
        n_zblk = (n_gh + blk - 1) // blk
        pad_z = n_zblk * blk - n_gh
        pad_lw = np.zeros(n_gh)
        if pad_z:
            z_np = np.pad(z_np, (0, pad_z), mode="edge")
            zp_np = np.pad(zp_np, (0, pad_z), mode="edge")
            zn_np = np.pad(zn_np, (0, pad_z), mode="edge")
            pad_lw = np.pad(pad_lw, (0, pad_z), constant_values=-np.inf)
        zg_blk = jnp.asarray(z_np.reshape(n_zblk, blk), jnp.float64)
        zpg_blk = jnp.asarray(zp_np.reshape(n_zblk, blk), jnp.float64)
        zng_blk = jnp.asarray(zn_np.reshape(n_zblk, blk), jnp.float64)
        zpad_blk = jnp.asarray(pad_lw.reshape(n_zblk, blk), jnp.float64)
        # Never let the bracket exceed the physical support: as R_lo -> 0 (a
        # bin with no response at all, where the exponent is flat in x) sigma
        # would otherwise blow up and every node would clip onto one of the two
        # rails.  Capped, such a bin degrades to a uniform-in-x tiling of the
        # support instead of a 2-point one.  Inactive by ~3 orders of magnitude
        # wherever the data carry signal (test_angle_marg_gh_laplace.py pins it).
        gh_sigma_cap = (x_max - x_min) / (2.0 * _GH_PSI_HALF_SIGMA)

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
        # running log-sum-exp across blocks (a lax.scan; see the packing note
        # above -- one traced kernel instead of G/blk unrolled copies)
        def _dist_step(carry, xw):
            mx, sx = carry
            xgb, lwgb = xw                                    # (blk,)
            xg = xgb[:, None, None, None]                     # (g,1,1,1)
            lwg = lwgb[:, None, None, None]
            av = xg * A0[None] - 0.5 * jnp.square(xg) * B0[None]
            c1 = xg * A1[None] - 0.5 * jnp.square(xg) * B1[None]
            c2 = -0.5 * jnp.square(xg) * B2[None]
            e = _laplace_psi_lnI_block(av, c1, c2) + lwg        # (g,c,S,npts)
            return _lse_update(mx, sx, e, axis=0), None

        if _use_gh:
            # ---- psi-marginal adaptive node placement, all FROZEN ----------
            # Centre on the psi that maximises the (unclipped) distance-maximum
            # exponent A(u)^2/(2 B(u)) -- available in CLOSED FORM here, see
            # the derivation above _gh_psi_node_offsets:
            #     e^{i u*} = +- conj(w)/|w|,  w = B0*A1 - conj(A1)*B2
            # with the sign picking the branch where A(u*) > 0 (x must be
            # positive).  Angle-free, so arg(0) never appears and w = 0 is a
            # regular point; reduces to conj(A1)/|A1| -- the maximiser of A
            # itself -- when B2 = 0.
            w_st = B0 * A1 - jnp.conj(A1) * B2
            ph1 = jnp.conj(w_st) / jnp.maximum(jnp.abs(w_st), 1e-300)
            sgn = jnp.where((A1 * ph1).real >= 0, 1.0, -1.0)
            ph1 = ph1 * sgn                                    # e^{i u*}
            A_st = A0 + (A1 * ph1).real                        # A(u*)
            B_st = B0 + (B1 * ph1).real + (B2 * ph1 * ph1).real
            R_lo = B0 - jnp.abs(B1) - jnp.abs(B2)              # <= min_u B
            gh_center = jax.lax.stop_gradient(
                jnp.clip(A_st / jnp.maximum(B_st, 1e-30), x_min, x_max))
            gh_sigma = jax.lax.stop_gradient(
                jnp.minimum(1.0 / jnp.sqrt(jnp.maximum(R_lo, 1e-30)),
                            gh_sigma_cap))

            def _gh_dist_step(carry, zw):
                mx, sx = carry
                zb, zpb, znb, zpadb = zw                       # (blk,)

                def _node(zz):
                    return jnp.clip(
                        gh_center[None] + gh_sigma[None] * zz[:, None, None, None],
                        x_min, x_max)

                xg = _node(zb)                                 # (g,c,S,npts)
                # composite-trapezoid weight, index-clamped at both ends:
                # identical to core._distmarg_gh_logL's diff/concatenate form.
                w = 0.5 * (_node(znb) - _node(zpb))
                pos = w > 0                                    # live (unclipped)
                lwg = jnp.where(pos, jnp.log(jnp.where(pos, w, 1.0))
                                - 4.0 * jnp.log(xg), -jnp.inf)
                lwg = lwg + zpadb[:, None, None, None]         # -inf on pad slots
                av = xg * A0[None] - 0.5 * jnp.square(xg) * B0[None]
                c1 = xg * A1[None] - 0.5 * jnp.square(xg) * B1[None]
                c2 = -0.5 * jnp.square(xg) * B2[None]
                e = _laplace_psi_lnI_block(av, c1, c2) + lwg
                return _lse_update(mx, sx, e, axis=0), None

            mx0 = jnp.full((c, S, npts), -jnp.inf, dtype=jnp.float64)
            sx0 = jnp.zeros((c, S, npts), dtype=jnp.float64)
            (mx, sx), _ = jax.lax.scan(
                _gh_dist_step, (mx0, sx0),
                (zg_blk, zpg_blk, zng_blk, zpad_blk))
            lnI = (mx + jnp.where(sx > 0, jnp.log(jnp.maximum(sx, 1e-300)),
                                  -jnp.inf)
                   + gh_C0 + lww[:, None, None])               # (c,S,npts)
            m_new, s_new = _lse_update(m, s, lnI, axis=0)
            return (m_new, s_new), None

        mx0 = jnp.full((c, S, npts), -jnp.inf, dtype=jnp.float64)
        sx0 = jnp.zeros((c, S, npts), dtype=jnp.float64)
        (mx, sx), _ = jax.lax.scan(_dist_step, (mx0, sx0), (xg_blk, lwg_blk))
        lnI = (mx + jnp.where(sx > 0, jnp.log(jnp.maximum(sx, 1e-300)), -jnp.inf)
               + lww[:, None, None])                          # (c,S,npts)
        m_new, s_new = _lse_update(m, s, lnI, axis=0)
        return (m_new, s_new), None

    m0 = jnp.full((S, npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, npts), dtype=jnp.float64)
    (m, s), _ = jax.lax.scan(jax.checkpoint(_step), (m0, s0), (phi_x, lw_x))
    lnL_t = m + jnp.log(s) - jnp.log(float(nphi_d))
    if return_lnLt:
        return lnL_t
    return _time_marginalize_terminal(lnL_t, data, time_quadrature)


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
