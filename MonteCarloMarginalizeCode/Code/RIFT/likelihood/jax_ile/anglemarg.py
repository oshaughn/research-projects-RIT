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

import sys

import numpy as np
import jax
import jax.numpy as jnp

from . import core as _core
from .core import (JAX_INTERP_DEFAULT, TIME_QUAD_DEFAULT, _accumulate_unit,
                   _time_marginalize_terminal,
                   _logsumexp_grid_blocked, _distmarg_gh_logL,
                   make_distance_gh)

__all__ = [
    "ANGLE_MARG_DEFAULT",
    "ANGLE_MARG_LEGACY",
    "ANGLE_MARG_CHOICES",
    "angle_sample_grid_sizes",
    "angle_coefficient_tables",
    "estimate_angle_amplitude",
    "coefficient_table_distphipsimarg_exact",
    "coefficient_table_distphipsimarg_laplace",
    "fused_log_likelihood_distphipsimarg_exact",
    "fused_log_likelihood_distphipsimarg_laplace",
    "choose_angle_marg_scheme",
    "fused_log_likelihood_distphipsimarg_peaklocal",
    "fused_log_likelihood_distphipsimarg_multipeak",
    "gh_laplace_supported",
    "gh_laplace_supported_for_data",
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
# THE (phi_ref, psi) SCHEME DEFAULT.  One definition; every entry point imports
# it, so the driver flag and the wrapper argument cannot drift.  This is the
# same discipline JAX_INTERP_DEFAULT already uses, and for the same reason: the
# last default move on this path (interp linear -> sinc) had the value re-typed
# in many places.
#
# CHANGED 2026-09-02: 'grid' -> 'exact'.  THIS CHANGES RESULTS for any caller
# that does not pass the scheme explicitly.  Pass --angle-marg-scheme grid (or
# angle_marg=ANGLE_MARG_LEGACY) to reproduce a pre-2026-09-02 run.
#
# Why 'exact' and not 'auto': 'auto' selects 'laplace' above
# ANGLE_MARG_CROSSOVER_AMPLITUDE (rho ~26-30; see that constant's note for why
# 26 and not the 21 this line used to say), which is an ACCURACY crossover.
# But 'laplace' on the DEFAULT UNIFORM distance grid was measured 43.2 nats
# from 'exact'+GH16 at rho 163 (mean; 16.3 median) -- an error on the
# DISTANCE axis, not the angular one, which is ~1e-6 nats there.  (This
# comment used to say laplace 'cannot use the per-sample adaptive distance
# quadrature'.  It can, and does, for m_max <= _GH_PSI_M_MAX via
# _gh_psi_node_offsets, gated by gh_laplace_supported; the 43.2 nats is what
# the UNIFORM grid costs, not what the scheme costs.)  The log-uniform distance
# grid is opt-in, so a caller who selects laplace without moving the distance
# axis with it pays that 43.2 nats.
# A default that is correct and slow beats one that is fast and tens of nats
# wrong.  'auto' becomes the right default once laplace has a sound distance
# quadrature, and ANGLE_MARG_CROSSOVER_AMPLITUDE should then be re-derived from
# COST as well as accuracy -- the measured cost crossover is rho ~200-326, an
# order of magnitude above the accuracy one.
#
# Why not 'grid': its quadrature error grows without bound with SNR (it averages
# exp(lnL), whose peak width is ~1/SNR, on n_phi x n_psi nodes).  Measured on
# the paper-1 ladder-2 injection at rho 652: the best of the 4 distinct
# n_phi=8 nodes is 37,419 nats below the true phi_ref profile peak, and the
# recovered sky position is displaced 0.53 deg -- the grid scheme ranks that
# artifact ABOVE the injection and the correct peak BELOW it, by 900.6 nats.
# Evidence: RIFT_roboto_paper analyses/sky_offset_diagnosis/
# RESULTS_phigrid_2026-09-02.md (commit 3f1f66f).
ANGLE_MARG_DEFAULT = "exact"
ANGLE_MARG_LEGACY = "grid"      # the spelling that reproduces pre-2026-09-02 runs
ANGLE_MARG_CHOICES = ("grid", "exact", "laplace", "peak-local", "phi-local",
                      "multipeak", "multipeak-jax", "auto")

#: 'peak-local' is deliberately NOT reachable from 'auto' yet.  It agrees with 'exact'
#: to 1e-13 nats on the tables measured so far and is device-independent (the same answer
#: on CPU and on an NVIDIA Blackwell GPU), but nothing has yet compared the two head to
#: head on a production campaign, and a scheme that changes the likelihood must not
#: become reachable by default on the strength of unit tests.  Explicit-only is what lets
#: a pilot run both and decide; promoting it into `choose_angle_marg_scheme` is a
#: separate change with its own evidence.

# ---------------------------------------------------------------------------
ANGLE_MARG_CROSSOVER_AMPLITUDE = 450.0     # A = rho^2/2; rho = 30.  NOTE the
# auto selector compares the MARGINED data-derived bound to this, so laplace
# engages below rho = 30.  TWO DIFFERENT NUMBERS LIVE HERE and an earlier version
# of this comment ran them together:
#   * the INTENDED margin is the `margin=2.0` argument of
#     estimate_angle_amplitude -- a deliberate parameter, not an estimate;
#   * the ratio the SWITCH actually keys on was measured AT THE LADDER INJECTION
#     (rho = 40.77): bound 1109.17 against the nominal rho^2/2 = 831.1, i.e.
#     1.335, not 2.
# That gives 450 / 1.335 -> engagement at nominal A ~ 337, rho ~ 26.0, which is
# what the manuscript quotes; this comment previously said rho ~ 21 by assuming
# the factor equalled the margin.  Keep the two distinct or the code and the
# paper quote different crossovers for the same switch.
#
# THREE LIMITS ON 1.335, so it is not read as more than it is:
#   (a) the denominator is the NOMINAL rho^2/2, not a measured maximum of the
#       (phi,psi) exponent.  Reading "the raw estimator sits at 0.667x TRUE A"
#       goes through the identification true A == rho^2/2, which is this file's
#       own convention but is not a measurement.
#   (b) the ratio is constant to 6e-5 across rungs rho = 40.77 ... 652.31.  That
#       is ARITHMETIC, NOT EVIDENCE: the ladder is one injection replayed at
#       scaled amplitudes, so the exponent rescales uniformly and the ratio is
#       forced.  Four decades of agreement validate nothing about the margin.
#   (c) it is ONE injection's sky-sample realization.  The shortfall's size is
#       set by how sharp the sky peak is relative to the sample -- the sample is
#       random draws plus a coarse uniform grid and does not contain the
#       injection's sky position, while the exponent is sharp enough that 1 of
#       9824 (sky, time) points sits within 23 nats of the peak.  A shortfall is
#       expected by design there, and nothing here bounds it for another event.
# So: at this injection the margin is load-bearing rather than decorative -- by
# about 1.5x -- and that is the whole claim.
#
# TWO OTHER RATIOS CIRCULATE FOR THIS LADDER AND NEITHER IS THE MARGIN.  Measured
# on it: raw-estimator/(rho^2/2) = 0.1888 and margined-bound/guess = 7.069.  Both
# are the SNR-GUESS DEFICIT SQUARED -- guess_amp == guess_snr^2/2 exactly, and
# rho/guess_snr = 2.3014 constant, so 0.1888 = 1/2.3014^2 and 7.069 =
# 2.3014^2 * 1.33465.  guess_snr is the ABANDONED sizing route (external review
# removed it precisely because an underestimated SNR silently under-resolved the
# dense quadrature).  If 7.069 lands here as "the margin" it inflates a ~1.5x
# effect to 7x and credits the live estimator with the dead route's deficit.
# They are easy to accept because they AGREE with the conclusion above -- for an
# unrelated reason -- so they read as corroboration and are not.
# The genuinely reportable fact in them: on this ladder guess_snr sits 2.30x
# below the true rho, so the abandoned route would have sized the dense grids
# from an amplitude too small BY A FACTOR THAT DEPENDS ON WHAT YOU DIVIDE BY --
#     7.069x against the LIVE data-derived bound (the thing that sizes grids
#            today, so this is the operative figure), and
#     5.296x against the nominal rho^2/2,
# the two differing by exactly the 1.335 above.  A reader handed "7.069x" with no
# denominator cannot tell which, and will be off by 1.335 either way: that is the
# same unnamed-denominator defect this block exists to guard against, and I
# shipped it here one commit before fixing it.  The docstring's stated failure
# mode, measured on a real configuration.  One injection, one guess_snr.  What actually protects the general case is
# _runtime_amp_failsafe, which recomputes the amplitude from the tables at the
# point of use and warns if it exceeds amp_sizing -- independent of whether the
# margin was well chosen.  (Ratios measured by the paper-1 ladder session.)
# Early engagement is safe by measurement either way: laplace is at -1.8e-04 nats
# by A = 200 on the injection ladder (the table above, spelled to match it so a
# grep finds both) and improves upward, while exact remains
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
                             sample_chunk=None, guard=0):
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

    ``guard`` requests primitive-only reconstruction support from the same
    accumulation operation as the terminal band-limited path.  The returned
    time axis then has ``data.npts + 2*guard`` samples; callers must discard the
    support after reconstruction and compare two guard widths before accepting.

    Memory: the tables are (m_max+1, 3, S, ntime) and (2*m_max+1, 5, S, ntime)
    complex -- independent of every grid size.  The sample scan runs in
    chunks of ``sample_chunk`` grid points (default npsi_s, i.e. one phi row
    per step), checkpointed so reverse-mode AD does not store per-step
    intermediates.

    Returns ``(C_A, C_B, meta)`` with grid sizes and the effective ``guard``
    and ``ntime`` support recorded in ``meta``.
    """
    m_max = _data_m_max(data)
    guard = int(guard)
    if guard < 0:
        raise ValueError("guard must be non-negative")
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
    npts = data.npts + 2 * guard
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
                                  interp, False, guard=guard)
        A = ku.real.reshape(c, S, npts)
        B = rs.reshape(c, S, npts)
        CA = CA + jnp.einsum("ckq,cst->kqst", pA, A)
        CB = CB + jnp.einsum("ckq,cst->kqst", pB, B)
        return (CA, CB), None

    CA0 = jnp.zeros((KPA, 2 * KSA + 1, S, npts), dtype=jnp.complex128)
    CB0 = jnp.zeros((KPB, 2 * KSB + 1, S, npts), dtype=jnp.complex128)
    (C_A, C_B), _ = jax.lax.scan(jax.checkpoint(_step), (CA0, CB0), xs)
    meta = dict(m_max=m_max, nphi_s=nphi_s, npsi_s=npsi_s,
                guard=guard, ntime=npts)
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
ENDPOINT_GUARD_BAND = 15.0    # nats below the amplitude maximum within which a
                              # sky point's dominant peak still counts for the
                              # near-boundary diagnostic below.  Applied per sky
                              # point against that point's own maximum AND again
                              # against the global one, so what survives is
                              # within [BAND, 2*BAND] nats of the global maximum
                              # -- a bound worth stating, since the two-stage
                              # form is what avoids a second pass over the
                              # per-entry arrays the sky loop exists to keep
                              # transient.


def _endpoint_bell(k):
    """``k exp(-k^2/2)``, the shape of a Gaussian's DERIVATIVE at a point ``k``
    widths from its peak, clamped at 0 for exterior (non-positive) clearances.

    This is the only ``k`` dependence in the truncated-endpoint error term of
    :func:`core.loguniform_endpoint_error`; it is maximal (0.6065) at exactly
    ONE width of clearance, and vanishes both far out in the tail AND at the
    peak itself -- an endpoint sitting ON the peak is a stationary point, where
    the Euler-Maclaurin endpoint term has nothing to correct.  So "closer to the
    edge" is NOT monotonically worse, and a guard shaped as a bare minimum
    clearance would refuse the k -> 0 case that the alias law already covers.
    """
    k = np.maximum(np.asarray(k, dtype=float), 0.0)
    return k * np.exp(-0.5 * np.square(k))


def _peak_clearance(A, B, x_min, x_max):
    """``(rho, clearance from the d_min edge, clearance from the d_max edge)``.

    The distance integrand ``exp(x A - x^2 B/2)`` is a Gaussian in ``x`` peaked
    at ``x* = A/B``, whose width in ``ln d`` is ``1/rho`` with
    ``rho = A/sqrt(B)`` -- scale free, which is the whole basis of the
    log-uniform grid (see :func:`core.make_distance_grid_loguniform`).  The
    clearances are that peak's distance from each prior edge IN THOSE UNITS,
    which is what decides whether the untruncated alias law applies.

    ``x = distMpcRef/d`` inverts the edges: the ``d_min`` edge is ``x_max``.
    Non-positive clearances mean the maximizer is EXTERIOR -- the boundary-layer
    regime section 1a refuses outright -- not a resolved peak, and callers must
    treat them as such rather than as "very close to the edge".
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Bs = np.maximum(B, 1e-300)
    rho = np.where((A > 0.0) & (B > 0.0), A / np.sqrt(Bs), 0.0)
    x_star = np.where(rho > 0.0, A / Bs, 0.0)
    ok = x_star > 0.0
    xs = np.where(ok, x_star, 1.0)
    k_lo = np.where(ok, rho * np.log(float(x_max) / xs), 0.0)
    k_hi = np.where(ok, rho * np.log(xs / float(x_min)), 0.0)
    return rho, k_lo, k_hi


def estimate_angle_amplitude(data, x_grid, interp=JAX_INTERP_DEFAULT,
                             n_sky=ANGLE_AMP_SKY_POINTS, seed=0,
                             margin=ANGLE_AMP_MARGIN,
                             _n_phi_e=None, _n_u_e=24,
                             return_diagnostics=False):
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
        amps_unclipped = []
        pk_A = []
        pk_B = []
        pk_ep = []
        pk_ext = []
        for j in range(C_A.shape[2]):      # per-sky loop bounds the transient
            A_g = (E_A @ C_A[:, :, j].reshape(-1, C_A.shape[-1])).real
            B_g = np.maximum(
                (E_B @ C_B[:, :, j].reshape(-1, C_B.shape[-1])).real, 0.0)
            # B >= 0 makes x*A - x^2/2*B concave in x: the max over the
            # actual support is at the clipped stationary point
            x_hat = np.clip(A_g / np.maximum(B_g, 1e-300), x_min, x_max)
            val = x_hat * A_g - 0.5 * np.square(x_hat) * B_g
            amps.append(max(float(val.max()), 0.0))
            # UNCLIPPED companion: the stationary value A^2/(2B) itself, which
            # is what the exponent would reach if the maximizing distance were
            # inside the prior support.  Only used to DETECT that it is not --
            # the returned amplitude is unchanged.  A <= 0 puts the stationary
            # point at negative x, where the max over x >= 0 is 0.
            # The A_g > 0 arm is inert FOR THIS CONSUMER and load-bearing for
            # a different one, which is the only useful thing to say about it.
            # A^2 is even in A, so the arm changes nothing whenever the
            # selection is an ARGMAX: the lattice maximum of A^2/(2B) sits at
            # A > 0 anyway, and removing the arm leaves clip_excess
            # bit-identical on every mode set tried ((2,+-2), +(2,+-1),
            # +(3,+-3), interior and exterior supports).  Nothing here can fail
            # if it is deleted.  It becomes load-bearing the moment a consumer
            # selects a BAND rather than a maximum -- over a threshold the
            # A < 0 branch is exactly degenerate with its mirror and survives
            # the cut, which is a documented trap (a weight cut ranked on the
            # unconstrained A^2/(2B) came out a factor ~300 wrong, and the tell
            # was that the answer did not move between a 10-nat and a 100-nat
            # threshold).  So: keep it, and if you add a banded selection here,
            # that is the point at which it needs a test.
            val_u = np.where(A_g > 0.0,
                             np.square(A_g) / (2.0 * np.maximum(B_g, 1e-300)),
                             0.0)
            amps_unclipped.append(max(float(val_u.max()), 0.0))
            # (A, B) of THIS sky point's DOMINANT configuration -- the entry
            # that attains its clipped maximum, i.e. the one whose distance
            # integral this sky direction's marginal is made of.  Kept (two
            # scalars per sky point, not the arrays) so the near-boundary
            # diagnostic below can be formed after the global maximum is known,
            # without a second pass over the per-entry arrays.
            i_hat = int(np.argmax(val))
            pk_A.append(float(A_g.ravel()[i_hat]))
            pk_B.append(float(B_g.ravel()[i_hat]))
            # ...and the endpoint term over EVERY entry of this sky point, not
            # just its dominant one.  A sky direction can have a far-interior
            # maximum and a near-equal secondary (phi, psi, time) entry whose
            # distance peak sits a width from an edge; that entry contributes
            # materially to the marginal and carries the full endpoint error,
            # while the dominant one scores ~0.  Reducing per sky to a scalar
            # keeps the memory bounded (the arrays never leave this loop) and
            # is CONSERVATIVE: it is a max over more entries than the band
            # would keep, and the rho^2 factor self-limits the quiet ones.
            # Measured under-read of the argmax-only form on a quiet
            # synthetic: up to 7562x (external review, P1).
            rho_e, klo_e, khi_e = _peak_clearance(A_g.ravel(), B_g.ravel(),
                                                  x_min, x_max)
            # The interior mask differs from no mask only when ONE edge is
            # exterior and the OTHER is within a few widths -- i.e. on a
            # support only a few 1/rho wide.  No prior this code is run with is
            # anywhere near that: [1,1000], [1,10000] and [1,3000] Mpc are
            # 69-947 widths across at rho 10-103, and even a narrow [100,2000]
            # box is 30-308, so the far edge's bell underflows to exactly 0 and
            # the two forms are numerically identical.  Dropping the mask
            # therefore survives the gate, and that is an equivalent mutant
            # rather than a coverage gap -- do not add a test for it.
            ep = np.where((klo_e > 0.0) & (khi_e > 0.0),
                          np.square(rho_e)
                          * (_endpoint_bell(klo_e) + _endpoint_bell(khi_e)),
                          0.0)
            pk_ep.append(float(ep.max()) if ep.size else 0.0)
            # ...and the loudest EXTERIOR entry of this sky point.  The
            # endpoint term above cannot represent one: its bell is clamped at
            # k <= 0, and the Euler-Maclaurin expansion is the wrong instrument
            # for a peak outside the support anyway (that is a boundary layer,
            # section 1a).  The exterior guards do not see it either --
            # clip_excess is a ratio of GLOBAL maxima and peak_clearance
            # describes the global argmax -- so an interior dominant entry with
            # a near-equal exterior secondary reads completely clean.  Carried
            # as its own scalar so the wrapper can band it.
            _ext = ~((klo_e > 0.0) & (khi_e > 0.0))
            pk_ext.append(float(val.ravel()[_ext].max()) if _ext.any()
                          else -np.inf)
        return (np.array(amps), np.array(amps_unclipped),
                np.array(pk_A), np.array(pk_B), np.array(pk_ep),
                np.array(pk_ext), C_A, C_B)

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

    (amps, amps_u, pk_A, pk_B, pk_ep, pk_ext,
     C_A, C_B) = _per_sky_amps(ra, dec, incl)
    # Concatenated across sky BATCHES, in the same idiom the two maxima use:
    # the near-boundary diagnostic is formed after the loop, so it must see the
    # re-drawn batches too or it reads a first batch that a later one displaced.
    amps_cat, pk_A_cat, pk_B_cat = amps, pk_A, pk_B
    pk_ep_cat, pk_ext_cat = pk_ep, pk_ext
    # split-half convergence check (mechanism 2 of the docstring): compare
    # the max WITHOUT the second half of the random draws against the max
    # with them; growth > 20% means the sky variation is under-sampled, so
    # draw again (at most twice) and say so.
    half = np.concatenate([amps[: n_sky // 2], amps[n_sky:]])   # + extremes
    amp_emp = float(amps.max())
    # Accumulated in the SAME idiom as amp_emp, and adjacently: dropping the
    # unclipped update in the re-draw loop below takes clip_excess below 1 and
    # disarms the F1 exteriority refusal, with no observable difference on any
    # fixture.  DESIGN_jax_distance_quadrature.md section 5(c);
    # test_sky_doubling_updates_the_unclipped_maximum_too pins it at the source.
    amp_u_emp = float(amps_u.max()) if len(amps_u) else 0.0
    amp_ref = float(half.max())
    grows = amp_emp > 1.2 * amp_ref + 1e-12
    n_extra = 0
    while grows and n_extra < 2 * n_sky:
        print("estimate_angle_amplitude: sky maximum still growing "
              "(%.4g -> %.4g); doubling the sample." % (amp_ref, amp_emp))
        ra2, dec2, incl2 = _draw(n_sky, rng)
        (amps2, amps_u2, pk_A2, pk_B2, pk_ep2, pk_ext2,
         _, _) = _per_sky_amps(ra2, dec2, incl2)
        amp_ref = amp_emp
        amp_emp = max(amp_emp, float(amps2.max()))
        amp_u_emp = max(amp_u_emp, float(amps_u2.max()))
        grows = amp_emp > 1.2 * amp_ref + 1e-12
        n_extra += n_sky
        amps_cat = np.concatenate([amps_cat, amps2])
        pk_A_cat = np.concatenate([pk_A_cat, pk_A2])
        pk_B_cat = np.concatenate([pk_B_cat, pk_B2])
        pk_ep_cat = np.concatenate([pk_ep_cat, pk_ep2])
        pk_ext_cat = np.concatenate([pk_ext_cat, pk_ext2])

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
    if return_diagnostics:
        amp_unclipped = amp_u_emp
        # NEAR-BOUNDARY diagnostic, for the log-uniform distance grid's OTHER
        # precondition: clip_excess sees a maximizer that has left the support,
        # but the spacing law is a statement about an effectively UNTRUNCATED
        # Gaussian, and a peak that is interior yet only ~1 width inside an edge
        # breaks it while clip_excess reads exactly 1.  Reduced here to one
        # scalar the wrapper turns into an error with its own spacing:
        # max over the loud sky points of rho^2 * (bell(k_lo) + bell(k_hi)),
        # which is the Euler-Maclaurin endpoint term stripped of the grid factor
        # (core.loguniform_endpoint_error puts it back).  BOTH edges, summed:
        # they are separate corrections and a narrow support has both.
        #
        # WHAT IS AND IS NOT COVERED.  EVERY reconstructed (phi, psi, time)
        # entry of every sky point within ENDPOINT_GUARD_BAND of the maximum --
        # not just each point's dominant configuration.  An earlier revision
        # kept only the per-sky argmax, so a sky direction with a far-interior
        # maximum and a near-equal secondary entry one width from an edge
        # scored ~0 while the secondary carried the full endpoint error;
        # measured under-read up to 7562x (external review, P1).  The per-entry
        # max is formed inside _per_sky_amps and reduced to one scalar per sky
        # there, so the per-entry arrays never leave that loop.  The band is
        # still on the sky point, which makes this conservative: it covers
        # entries a per-entry band would drop, and the rho^2 factor self-limits
        # them.  The band is a threshold on the CLIPPED value, never on
        # A^2/(2B): the A < 0 mirror is exactly degenerate under an
        # unconstrained ranking and survives such a cut (trap in
        # _per_sky_amps).
        # The band is a threshold on the CLIPPED value, never on A^2/(2B): the
        # A < 0 mirror is exactly degenerate under an unconstrained ranking and
        # survives such a cut, which is the trap recorded in _per_sky_amps.
        keep = amps_cat >= amp_emp - ENDPOINT_GUARD_BAND
        rho_pk, k_lo, k_hi = _peak_clearance(pk_A_cat, pk_B_cat, x_min, x_max)
        endpoint_scale = (float(np.where(keep, pk_ep_cat, 0.0).max())
                          if pk_ep_cat.size else 0.0)
        _ext_best = (float(np.max(pk_ext_cat)) if pk_ext_cat.size
                     and np.isfinite(pk_ext_cat).any() else -np.inf)
        i_dom = int(np.argmax(amps_cat)) if amps_cat.size else 0
        return margin * amp_emp, dict(
            amp_clipped=float(amp_emp),
            amp_unclipped=amp_unclipped,
            # the Euler-Maclaurin endpoint term, grid-independent half
            endpoint_scale=endpoint_scale,
            # nats by which the loudest EXTERIOR entry sits BELOW the global
            # maximum.  inf when there is none.  Small means a boundary-layer
            # configuration is contributing materially while every other
            # diagnostic here reads clean -- see the wrapper's refusal.
            exterior_gap=(float(amp_emp - _ext_best)
                          if np.isfinite(_ext_best) else float("inf")),
            # the globally dominant peak itself, reported so a refusal can name
            # a number the caller can act on (and so a test can BUILD a support
            # with a chosen clearance rather than hunt for one)
            peak_x=(float(pk_A_cat[i_dom] / max(pk_B_cat[i_dom], 1e-300))
                    if amps_cat.size else 0.0),
            peak_rho=float(rho_pk[i_dom]) if rho_pk.size else 0.0,
            peak_clearance=float(min(k_lo[i_dom], k_hi[i_dom]))
                           if rho_pk.size else 0.0,
            # > 1 means the exponent's maximizing distance x* = A/B lies
            # OUTSIDE [x_min, x_max] for the dominant angles, i.e. the
            # distance posterior rails against a prior edge.  See
            # DESIGN_jax_distance_quadrature.md section 1a.
            clip_excess=(amp_unclipped / amp_emp) if amp_emp > 0.0
                        else (float("inf") if amp_unclipped > 0.0 else 1.0),
            x_min=x_min, x_max=x_max)
    return margin * amp_emp


AMP_FAILSAFE_TRIP_FACTOR = 2.0
# The multiple of amp_sizing at which _runtime_amp_failsafe speaks up, and
# therefore the largest amplitude a run may reach while remaining UNLABELLED.
# It is a named constant because a second consumer now sizes from it: the
# log-uniform distance grid resolves peaks up to
# rho = sqrt(2 * TRIP * amp_sizing), so that every call the guard admits in
# silence is one the distance spacing already covers.  Sizing that grid from
# amp_sizing itself (as an earlier draft did) leaves a factor sqrt(TRIP) in SNR
# -- and hence, through the Gaussian alias law, a tol -> sqrt(2*tol) hole --
# open BELOW the trigger, where nothing is printed and nothing is recorded.
# Raise this and the distance grid follows automatically; that coupling is the
# point of the constant.


_AMP_FAILSAFE = {"tripped": False, "n_calls": 0, "worst_amp": 0.0,
                 "amp_sizing": None, "scheme": None}


def reset_amp_failsafe():
    """Clear the undersizing record (call once per event, before sampling).

    No barrier is needed: the record is written synchronously at the Python
    boundary by :func:`record_amp_failsafe`, never by a queued device effect.
    """
    _AMP_FAILSAFE.update(tripped=False, n_calls=0, worst_amp=0.0,
                         amp_sizing=None, scheme=None)


def amp_failsafe_state(barrier=True):
    """Host-side record of the deterministic output-cloud amplitude checks.

    ``barrier`` is accepted for API compatibility and ignored: there are no
    queued device effects left to drain.  The jitted kernels RETURN their
    amplitude metric as ordinary data and the wrapper accumulates it
    synchronously once each batch is ready, so a read here is already
    ordered after every batch the caller has taken delivery of.

    Keeping host effects out of these graphs is load-bearing beyond tidiness:
    ``jax/_src/compiler.py::_cache_write`` refuses to write a persistent cache
    entry for any module carrying host callbacks ("because it uses host
    callbacks"), so a ``jax.debug.print``/``jax.debug.callback`` anywhere in
    the angle-marginalization graph makes the most expensive compile in RIFT
    permanently uncacheable.

    Returns a dict; ``tripped`` is the load-bearing field.  Consumers should
    LABEL their output rather than discard it -- see the note in
    :func:`_runtime_amp_failsafe` about why this is not fatal and not a NaN.
    """
    return dict(_AMP_FAILSAFE)


def record_amp_failsafe(amp_call, amp_sizing, scheme_name):
    """Accumulate one already-evaluated batch's amplitude maximum.

    Runs at the Python boundary after the device result is ready, never inside
    a JIT.  Maxima accumulate across chunks and calls for the whole event; the
    likelihood values are neither altered nor filtered.
    """
    amp_call = float(np.max(np.asarray(amp_call)))
    amp_sizing = float(amp_sizing)
    tripped = amp_call > AMP_FAILSAFE_TRIP_FACTOR * amp_sizing
    _AMP_FAILSAFE["n_calls"] += 1
    _AMP_FAILSAFE["worst_amp"] = max(_AMP_FAILSAFE["worst_amp"], amp_call)
    _AMP_FAILSAFE["amp_sizing"] = amp_sizing
    _AMP_FAILSAFE["scheme"] = scheme_name
    if tripped:
        _AMP_FAILSAFE["tripped"] = True
        sys.stderr.write(
            "WARNING anglemarg/%s: this batch's coefficient tables reach an "
            "amplitude scale ~%.4g (analytic over-reading expression), above "
            "%gx the amp_sizing=%.4g the dense (phi,psi) grids were built "
            "for.  estimate_angle_amplitude underestimated the sky maximum; "
            "the marginal may be under-resolved at such points.  Rebuild the "
            "likelihood with amp_sizing >= the reported amplitude.\n"
            % (scheme_name, amp_call, AMP_FAILSAFE_TRIP_FACTOR, amp_sizing))


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
    margin in N, i.e. 4x in amplitude).  It RETURNS the metric as ordinary JAX
    data; no value is altered and there are deliberately no host effects here,
    because such effects make this graph ineligible for JAX's persistent
    compilation cache.  Everything under stop_gradient: the check must not
    appear in the AD graph.
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
    # The hazard this check answers: a warning printed from inside jit does not stop
    # anything, so a production run could finish and publish biased likelihoods, samples
    # and evidence while the "fail-safe" scrolled past in a log.  The recourse chosen is
    # a HOST-RECORDED LABEL, not a poisoned value -- see the block below, which gives the
    # reasoning and the two rejected alternatives.  This function alters no value; it
    # RETURNS the metric as ordinary JAX data and the caller records it at the Python
    # boundary.  Everything is under stop_gradient so the check never enters the AD graph.
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
    # So: the value is untouched, the run completes, and the metric is RETURNED
    # for the host to record, so the driver can LABEL the result as suspect in
    # its provenance.  A labelled result an operator can judge beats both a
    # vanished region and a dead run.
    #
    # WHY THIS IS RETURNED RATHER THAN REPORTED FROM INSIDE THE GRAPH, which is
    # what it used to be.  Two reasons, and the second is why it changed.
    # (1) Reliability: debug-callback effects may be dropped, duplicated or
    # reordered under transformation, so a clean read never proved adequacy and
    # every consumer had to say so.  (2) Cacheability, which is load-bearing:
    # jax/_src/compiler.py::_cache_write declines to write a persistent cache
    # entry for any module that carries host callbacks.  With one in this
    # function the whole angle-marginalization graph -- the most expensive
    # compile in RIFT, minutes for peak-local -- could never be persistently
    # cached, on any scheme, in any run.  Returning the metric fixes both: the
    # record is now synchronous and exact for every batch the caller takes
    # delivery of.
    #
    # The COVERAGE that buys is narrower than "every traced call", and is stated
    # rather than implied: the wrapper records on the BATCHED path (pilot,
    # reweight and final output-cloud evaluations, i.e. every point that reaches
    # a published artifact) and deliberately not on the scalar AD/flow-training
    # path, whose proposals do not enter those artifacts.
    return amp_call


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


def coefficient_table_distphipsimarg_exact(
        C_A, C_B, x_grid, log_w_grid, *, amp_sizing=None, m_max=None,
        dense_chunk=8, grid_block=32, return_amp=False):
    """Stream the exact angle/distance reserve from coefficient tables.

    This is the common fixed-point seam between the dense/exact reserve and
    all-axis peak-local work.  ``C_A`` and ``C_B`` may be the batched
    ``(KP,KS,S,Ntime)`` tables returned by :func:`angle_coefficient_tables`, or
    an unbatched ``C_A`` of shape ``(KP,KS,Ntime)`` together with a collapsed,
    time-independent ``C_B`` of shape ``(KP,KS)``.  The latter is exactly the
    compact representation used by ``all_axis_peaklocal``.

    The result has shape ``(S,Ntime)`` and is normalized over the two periodic
    angles.  With the fixed-grid distance path, normalization is entirely
    determined by ``log_w_grid``.  When ``JAX_ILE_DISTMARG_GH`` is enabled, the
    existing adaptive-distance contract instead reads only the support from
    ``x_grid`` and applies its built-in normalized volumetric ``x**-4`` measure.
    Time is deliberately not integrated here.  Keeping those measures explicit
    prevents an empirical local result using continuous ``x**-4 dx`` from being
    silently compared with a differently normalized production distance prior.

    Dense angle coordinates are generated procedurally from each scan index and
    distance blocks are streamed, so peak workspace is controlled by
    ``dense_chunk`` and ``grid_block`` rather than the complete dense angle
    lattice.  No waveform or packed U,V/Q contraction is repeated.
    """
    C_A = jnp.asarray(C_A, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    if C_A.ndim == 3:
        C_A = C_A[:, :, None, :]
    if C_A.ndim != 4:
        raise ValueError("C_A must have shape (KP,KS[,S],Ntime)")
    if C_B.ndim == 2:
        C_B = jnp.broadcast_to(
            C_B[:, :, None, None],
            C_B.shape + (C_A.shape[2], C_A.shape[3]))
    if C_B.ndim != 4 or C_B.shape[2:] != C_A.shape[2:]:
        raise ValueError(
            "C_B must be collapsed (KP,KS) or match C_A sample/time axes")
    if C_A.shape[1] % 2 != 1 or C_B.shape[1] % 2 != 1:
        raise ValueError("angular harmonic axes must have odd length")
    if x_grid.ndim != 1 or log_w_grid.shape != x_grid.shape or x_grid.size < 2:
        raise ValueError("x_grid/log_w_grid must be matching one-dimensional grids")
    if not (int(dense_chunk) > 0 and int(grid_block) > 0):
        raise ValueError("dense_chunk and grid_block must be positive")
    inferred_m_max = int(C_A.shape[0] - 1)
    if m_max is None:
        m_max = inferred_m_max
    m_max = int(m_max)
    if m_max != inferred_m_max:
        raise ValueError("m_max does not match the C_A harmonic order")
    if C_B.shape[0] < 2 * m_max + 1:
        raise ValueError("C_B does not contain the required norm harmonics")

    S = C_A.shape[2]
    npts = C_A.shape[3]
    amp_sizing = _require_amp_sizing(amp_sizing)
    amp_call = _runtime_amp_failsafe(C_A, C_B, x_grid, amp_sizing,
                                     "exact-tables")
    nphi_d, nu_d = _dense_grid_sizes(amp_sizing, m_max=m_max)
    c = int(dense_chunk)
    n_dense = nphi_d * nu_d
    nsteps = (n_dense + c - 1) // c
    lane = jnp.arange(c, dtype=jnp.int32)

    a_g = x_grid
    b_g = -0.5 * jnp.square(x_grid)
    _use_gh = _core._DISTMARG_GH_N > 0
    if _use_gh:
        gh_xi, gh_logw = make_distance_gh(_core._DISTMARG_GH_N)
        x_min = jnp.min(x_grid)
        x_max = jnp.max(x_grid)

    def _step(carry, step):
        m, s = carry
        flat = step * c + lane
        live = flat < n_dense
        safe = jnp.minimum(flat, n_dense - 1)
        iphi = safe // nu_d
        iu = safe - iphi * nu_d
        phw = (2.0 * jnp.pi / float(nphi_d)) * iphi
        uw = (2.0 * jnp.pi / float(nu_d)) * iu
        lww = jnp.where(live, 0.0, -jnp.inf)
        A = _reconstruct_field(C_A, phw, uw)
        B = _reconstruct_field(C_B, phw, uw)
        K2 = A.reshape(c * S, npts)
        R2 = B.reshape(c * S, npts)
        if _use_gh:
            lnL = _distmarg_gh_logL(K2, R2, gh_xi, gh_logw, x_min, x_max)
        else:
            lnL = _logsumexp_grid_blocked(
                K2, R2, a_g, b_g, log_w_grid, grid_block)
        lnL = lnL.reshape(c, S, npts) + lww[:, None, None]
        m_new, s_new = _lse_update(m, s, lnL, axis=0)
        return (m_new, s_new), None

    m0 = jnp.full((S, npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, npts), dtype=jnp.float64)
    (m, s), _ = jax.lax.scan(
        jax.checkpoint(_step), (m0, s0),
        jnp.arange(nsteps, dtype=jnp.int32))
    lnL_t = m + jnp.log(s) - jnp.log(float(n_dense))
    return (lnL_t, amp_call) if return_amp else lnL_t


def fused_log_likelihood_distphipsimarg_exact(
        data, ra, dec, incl, x_grid, log_w_grid,
        interp=JAX_INTERP_DEFAULT, amp_sizing=None,
        dense_chunk=8, grid_block=32,
        time_quadrature=TIME_QUAD_DEFAULT, return_lnLt=False,
        return_amp=False):
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
    lnL_t, amp_call = coefficient_table_distphipsimarg_exact(
        C_A, C_B, x_grid, log_w_grid, amp_sizing=amp_sizing,
        m_max=meta["m_max"], dense_chunk=dense_chunk,
        grid_block=grid_block, return_amp=True)
    out = lnL_t if return_lnLt else _time_marginalize_terminal(
        lnL_t, data, time_quadrature)
    return (out, amp_call) if return_amp else out


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

#: Maximum number of independent ``(sample, time)`` points presented to one
#: distance/psi kernel invocation.  This is an execution-only tile: neither a
#: quadrature count nor an accuracy knob.  At the shipped ``QCH=16``,
#: ``dist_block=4`` and ``phi_chunk=16``, the largest pure-quadrature slab is
#:
#:     16 * 4 * 16 * LAPLACE_POINT_BLOCK * sizeof(float64) = 32 MiB.
#:
#: This is a bound on the EXPLICIT sample/time axes of a direct kernel call,
#: not on arbitrary enclosing transformations: an outer ``vmap`` (flowMC maps
#: its scalar AD target over chains) adds another batch axis that this function
#: cannot see when it chooses ``pblk``.
#:
#: Before this point axis was rolled, that last factor was ``S * npts``.  The
#: historical pre-cap JAX acceptance call at ``S=4000, npts=1193`` therefore
#: asked XLA for one 36.41-GiB buffer.  That number is the repository-recorded
#: allocation request for this ONE logical f64 value, not a measurement of
#: current production ILE peak memory.  The sampler-side device cap reduces S
#: for current host-batched callers, but direct ``log_likelihood`` calls do not,
#: and a device-memory fraction does not bound the total live graph or its AD
#: residuals.  The coefficient tables and the output still scale as O(S*npts);
#: this constant removes only that multiplicative quadrature slab for explicit
#: batches.
LAPLACE_POINT_BLOCK = 4096


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


def coefficient_table_distphipsimarg_laplace(
        C_A, C_B, x_grid, log_w_grid, *, amp_sizing=None, m_max=None,
        phi_chunk=16, dist_block=4, point_block=LAPLACE_POINT_BLOCK,
        return_amp=False):
    """Stream the psi-Laplace angle/distance reserve from coefficient tables.

    Same seam and normalization contract as
    :func:`coefficient_table_distphipsimarg_exact`: it consumes the tables
    :func:`angle_coefficient_tables` returns, produces ``(S, Ntime)``
    normalized over the two periodic angles, and does NOT integrate time.
    Extracted from :func:`fused_log_likelihood_distphipsimarg_laplace`, which
    is now a thin wrapper over it, so the two paths cannot drift.

    It exists because the four-axis policy's reserve consumes already-built
    tables at refined time nodes, and the only table-level reserve was the
    exact one.  That made the reserve method un-selectable: the composite could
    only ever fall back to exact angles, whatever the amplitude.

    Costs ~sqrt(A) rather than ~A -- the lattice is dense in phi only, the psi
    axis being removed analytically -- and its error SHRINKS with amplitude, so
    it is the reserve to use above the selector crossover.  The caller owns that
    choice; this function does not select.

    The adaptive distance quadrature is honoured exactly as in the fused
    kernel, and carries the same restriction: the psi-marginal node placement
    rests on the A0 == 0 / B1 == 0 identity, established for mode content up to
    ``_GH_PSI_M_MAX``.  That identity is a property of the DATA and cannot be
    measured here, where the tables are tracers; the caller must have gated it
    with :func:`gh_laplace_supported` on concrete tables.
    """
    C_A = jnp.asarray(C_A, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    if C_A.ndim == 3:
        C_A = C_A[:, :, None, :]
    if C_A.ndim != 4:
        raise ValueError("C_A must have shape (KP,KS[,S],Ntime)")
    if C_B.ndim == 2:
        C_B = jnp.broadcast_to(
            C_B[:, :, None, None],
            C_B.shape + (C_A.shape[2], C_A.shape[3]))
    if C_B.ndim != 4 or C_B.shape[2:] != C_A.shape[2:]:
        raise ValueError(
            "C_B must be collapsed (KP,KS) or match C_A sample/time axes")
    inferred_m_max = int(C_A.shape[0] - 1)
    if m_max is None:
        m_max = inferred_m_max
    m_max = int(m_max)
    if m_max != inferred_m_max:
        raise ValueError("m_max does not match the C_A harmonic order")
    S = int(C_A.shape[2])
    npts = int(C_A.shape[3])
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)

    _use_gh = _core._DISTMARG_GH_N > 0
    # This runs under jit/grad, where C_A and C_B are TRACERS, so the identity
    # cannot be measured here -- it is a property of the DATA and is enforced
    # once, on concrete tables, by JAXDistPhiPsiMargLikelihood (which gates
    # EVERY scheme, not just 'auto').  A caller invoking this kernel directly
    # under GH is responsible for calling gh_laplace_supported itself; the
    # m_max test below is the only check available at trace time.
    if _use_gh and int(m_max) > _GH_PSI_M_MAX:
        raise ValueError(
            "distance-GH-nodes is set (--distance-gh-nodes / "
            "JAX_ILE_DISTMARG_GH) and the 'laplace' angle-marg scheme's "
            "psi-marginal distance-node placement is validated for mode "
            "content m_max <= %d only (it rests on the A0 == 0 / B1 == 0 "
            "identity); this data has m_max = %d.  Use --angle-marg-scheme "
            "exact, or pass --distance-gh-nodes 0 (or unset "
            "JAX_ILE_DISTMARG_GH)."
            % (_GH_PSI_M_MAX, int(m_max)))

    amp_sizing = _require_amp_sizing(amp_sizing)
    # x_grid is still the right argument under GH: the adaptive nodes are
    # CLIPPED into [min x_grid, max x_grid], so the amplitude bound the
    # failsafe computes over x_grid bounds the nodes actually used.
    amp_call = _runtime_amp_failsafe(C_A, C_B, x_grid, amp_sizing, "laplace")
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
    pblk = min(int(point_block), S * npts)
    if pblk < 1:
        raise ValueError("point_block must be at least 1")
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
        # psi-Fourier coefficient FIELDS at the dense phi points (c,S,npts).
        # Shared with gh_laplace_supported so the identity that predicate
        # measures IS the one this placement depends on.
        A0, A1, B0, B1, B2 = psi_harmonics_at_phi(C_A, C_B, phw, m_max)

        # Roll the combined independent (sample,time) point axis BEFORE adding
        # distance and quadrature axes.  The old body formed
        # (quad_chunk, dist_block, phi_chunk, S, npts) at once; the sampler cap
        # only hid that from some callers.  Edge padding is safe because each
        # padded result is discarded before the phi reduction.  Repeating the
        # edge (rather than zero-padding the coefficients) also keeps every
        # branch finite, which matters to reverse-mode AD even for dead outputs.
        npoint = S * npts
        n_pblk = (npoint + pblk - 1) // pblk
        pad_p = n_pblk * pblk - npoint

        def _pack_points(v):
            v = v.reshape(c, npoint)
            if pad_p:
                v = jnp.concatenate(
                    [v, jnp.broadcast_to(v[:, -1:], (c, pad_p))], axis=1)
            return jnp.swapaxes(v.reshape(c, n_pblk, pblk), 0, 1)

        fields = tuple(_pack_points(v) for v in (A0, A1, B0, B1, B2))

        def _point_step(field_block):
            A0p, A1p, B0p, B1p, B2p = field_block             # (c,pblk)

            # distance quadrature: blocked, vectorized over the block (AD-fast),
            # running log-sum-exp across blocks (a lax.scan; see the packing note
            # above -- one traced kernel instead of G/blk unrolled copies)
            def _dist_step(carry, xw):
                mx, sx = carry
                xgb, lwgb = xw                                # (blk,)
                xg = xgb[:, None, None]                       # (g,1,1)
                lwg = lwgb[:, None, None]
                av = xg * A0p[None] - 0.5 * jnp.square(xg) * B0p[None]
                c1 = xg * A1p[None] - 0.5 * jnp.square(xg) * B1p[None]
                c2 = -0.5 * jnp.square(xg) * B2p[None]
                e = _laplace_psi_lnI_block(av, c1, c2) + lwg  # (g,c,pblk)
                return _lse_update(mx, sx, e, axis=0), None

            if _use_gh:
                # ---- psi-marginal adaptive node placement, all FROZEN ------
                # Centre on the psi that maximises the (unclipped)
                # distance-maximum exponent A(u)^2/(2 B(u)); see the derivation
                # above _gh_psi_node_offsets.
                w_st = B0p * A1p - jnp.conj(A1p) * B2p
                ph1 = jnp.conj(w_st) / jnp.maximum(jnp.abs(w_st), 1e-300)
                sgn = jnp.where((A1p * ph1).real >= 0, 1.0, -1.0)
                ph1 = ph1 * sgn                                # e^{i u*}
                A_st = A0p + (A1p * ph1).real                  # A(u*)
                B_st = B0p + (B1p * ph1).real + (B2p * ph1 * ph1).real
                R_lo = B0p - jnp.abs(B1p) - jnp.abs(B2p)       # <= min_u B
                gh_center = jax.lax.stop_gradient(
                    jnp.clip(A_st / jnp.maximum(B_st, 1e-30), x_min, x_max))
                gh_sigma = jax.lax.stop_gradient(
                    jnp.minimum(1.0 / jnp.sqrt(jnp.maximum(R_lo, 1e-30)),
                                gh_sigma_cap))

                def _gh_dist_step(carry, zw):
                    mx, sx = carry
                    zb, zpb, znb, zpadb = zw                   # (blk,)

                    def _node(zz):
                        return jnp.clip(
                            gh_center[None]
                            + gh_sigma[None] * zz[:, None, None],
                            x_min, x_max)

                    xg = _node(zb)                             # (g,c,pblk)
                    # Composite-trapezoid weight, index-clamped at both ends:
                    # identical to core._distmarg_gh_logL's convention.
                    w = 0.5 * (_node(znb) - _node(zpb))
                    pos = w > 0                               # live (unclipped)
                    lwg = jnp.where(pos, jnp.log(jnp.where(pos, w, 1.0))
                                    - 4.0 * jnp.log(xg), -jnp.inf)
                    lwg = lwg + zpadb[:, None, None]           # -inf on pad slots
                    av = xg * A0p[None] - 0.5 * jnp.square(xg) * B0p[None]
                    c1 = xg * A1p[None] - 0.5 * jnp.square(xg) * B1p[None]
                    c2 = -0.5 * jnp.square(xg) * B2p[None]
                    e = _laplace_psi_lnI_block(av, c1, c2) + lwg
                    return _lse_update(mx, sx, e, axis=0), None

                mx0 = jnp.full((c, pblk), -jnp.inf, dtype=jnp.float64)
                sx0 = jnp.zeros((c, pblk), dtype=jnp.float64)
                (mx, sx), _ = jax.lax.scan(
                    _gh_dist_step, (mx0, sx0),
                    (zg_blk, zpg_blk, zng_blk, zpad_blk))
                return (mx + jnp.where(
                    sx > 0, jnp.log(jnp.maximum(sx, 1e-300)), -jnp.inf)
                        + gh_C0)

            mx0 = jnp.full((c, pblk), -jnp.inf, dtype=jnp.float64)
            sx0 = jnp.zeros((c, pblk), dtype=jnp.float64)
            (mx, sx), _ = jax.lax.scan(
                _dist_step, (mx0, sx0), (xg_blk, lwg_blk))
            return mx + jnp.where(sx > 0,
                                  jnp.log(jnp.maximum(sx, 1e-300)), -jnp.inf)

        # Avoid wrapping the overwhelmingly common scalar/small-test case in a
        # one-trip map: it buys no memory and adds another control-flow region
        # for XLA/AD to compile.  Production batches cross the bound and take
        # the rolled path below.
        if n_pblk == 1:
            lnI_blk = _point_step(tuple(v[0] for v in fields))[None]
        else:
            lnI_blk = jax.lax.map(jax.checkpoint(_point_step), fields)
        lnI = (jnp.swapaxes(lnI_blk, 0, 1).reshape(c, n_pblk * pblk)
               [:, :npoint].reshape(c, S, npts)
               + lww[:, None, None])                          # (c,S,npts)
        m_new, s_new = _lse_update(m, s, lnI, axis=0)
        return (m_new, s_new), None

    m0 = jnp.full((S, npts), -jnp.inf, dtype=jnp.float64)
    s0 = jnp.zeros((S, npts), dtype=jnp.float64)
    (m, s), _ = jax.lax.scan(jax.checkpoint(_step), (m0, s0), (phi_x, lw_x))
    lnL_t = m + jnp.log(s) - jnp.log(float(nphi_d))
    return (lnL_t, amp_call) if return_amp else lnL_t


def fused_log_likelihood_distphipsimarg_laplace(
        data, ra, dec, incl, x_grid, log_w_grid,
        interp=JAX_INTERP_DEFAULT, amp_sizing=None,
        phi_chunk=16, dist_block=4, point_block=LAPLACE_POINT_BLOCK,
        time_quadrature=TIME_QUAD_DEFAULT, return_lnLt=False,
        return_amp=False):
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

    Two DIFFERENT axes, and conflating them has already misled a reader.  The
    paragraph above is about the PER-SAMPLE adaptive quadrature.  The STATIC
    distance grid is separate and is not restricted here at all:
    ``--distance-grid-scheme loguniform`` is supported and gated on this path,
    and needs no node-placement rule because it locates no peak -- one relative
    spacing resolves every per-sample peak wherever it sits.  See
    DESIGN_jax_distance_quadrature.md.  The two cannot be combined: with
    JAX_ILE_DISTMARG_GH set the per-sample quadrature consumes only the SUPPORT
    of ``x_grid``, so the log-uniform option would be bit-identically inert and
    is refused rather than silently ignored.

    Memory of the multiplicative quadrature slab is bounded by ``phi_chunk`` x
    ``dist_block`` x ``point_block``, never by the full sample x time product or
    by grid sizes.  ``point_block`` rolls independent ``(sample, time)`` bins and
    changes no quadrature rule or reduction order within a bin.
    """
    # RESPONSE-MODEL PRECONDITION, before anything is built.  This function is
    # public (__all__) and is called directly by the wrapper and by several test
    # modules, so a wrapper-only gate leaves a live bypass: a direct call with a
    # banded response and m_max <= 2 would execute the unsupported placement
    # while the wrapper correctly refused it.  `feature` is a plain Python
    # attribute -- static and trace-safe -- so unlike the numerical A0/B1
    # measurement (which needs concrete tables and therefore stays in the
    # wrapper) it costs nothing, and checking it here also avoids paying for a
    # coefficient-table build that is about to be rejected.
    if _core._DISTMARG_GH_N > 0:
        _feature = getattr(data, "feature", None)
        if _feature not in _GH_PSI_STATIC_FEATURES:
            raise ValueError(
                "distance-GH-nodes is set (--distance-gh-nodes / "
                "JAX_ILE_DISTMARG_GH), but the 'laplace' angle-marg "
                "scheme's psi-marginal distance-node placement requires the "
                "static detector response: it is DERIVED from A0 == 0 and "
                "B1 == 0, which follow from F+(psi) + i Fx(psi) = "
                "(F+(0) + i Fx(0)) e^{-2i psi}.  This data has feature=%r, "
                "which does not have that factorization.  Use "
                "--angle-marg-scheme exact, or pass --distance-gh-nodes 0 "
                "(or unset JAX_ILE_DISTMARG_GH)."
                % (_feature,))
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    C_A, C_B, meta = angle_coefficient_tables(data, ra, dec, incl, interp)
    lnL_t, amp_call = coefficient_table_distphipsimarg_laplace(
        C_A, C_B, x_grid, log_w_grid, amp_sizing=amp_sizing,
        m_max=meta["m_max"], phi_chunk=phi_chunk, dist_block=dist_block,
        point_block=point_block, return_amp=True)
    out = lnL_t if return_lnLt else _time_marginalize_terminal(
        lnL_t, data, time_quadrature)
    return (out, amp_call) if return_amp else out


# Relative size at which A0 / B1 count as nonzero.  The identity the psi-marginal
# node placement rests on (A0 == 0, B1 == 0, so R_lo = B0 - |B2| IS min_u B) is a
# property of the SPIN-2 detector response, not of the source, and is measured at
# ~1e-16 relative on every non-precessing mode set tried through m_max = 4.  It is
# NOT measured under precession.
#
# WHY 1e-8, and what is NOT claimed for it.  The identity is a STRUCTURAL
# precondition, not a numerical one: the closed-form psi maximiser below is
# DERIVED from A0 == 0 and B1 == 0 (that is what reduces stationarity to
# z^2 w = conj(w)).  So the tolerance's job is to separate "numerically zero"
# from "structurally nonzero", not to bound an error.  Observed values are
# ~1e-16 relative on every mode set tried, and the mutation sweep in
# test_angle_marg_gh_selection.py shows planted harmonics at 1e-3 are caught,
# so 1e-8 sits ~8 orders above the noise and ~5 below the smallest breach the
# tests exercise.
#
# An earlier revision of this comment also claimed 1e-8 was "~8 orders below a
# value that would move the bracket".  That was never measured and is removed
# rather than left standing: an attempt to measure it produced a centre error
# FLAT at ~2 sigma across six decades of planted A0/B1, including where the
# identity holds -- a hand-rolled reimplementation of the maximiser failing its
# own flatness check, not a property of the code.  If the upper end is ever
# wanted, measure it through the shipped kernel, not a re-derivation.
#
# Values are not bit-portable: they come through BLAS-heavy reconstruction in
# angle_coefficient_tables, so anything pinned off them needs a RELATIVE
# tolerance.  This comparison is already relative and one-sided.
GH_PSI_IDENTITY_TOL = 1e-8

# The response models for which A0 == 0 / B1 == 0 hold at EVERY extrinsic point.
# This is the angle-independent half of the gate and it is the actual guarantee:
# the static path builds F = F+ + i Fx through compute_detamresponse (LAL's
# ComputeDetAMResponse), where the polarization enters as an exact rotation,
#     F+(psi) + i Fx(psi) = (F+(0) + i Fx(0)) e^{-2 i psi},
# a SINGLE u-harmonic (u = 2 psi).  kappa is linear in F and rho^2 quadratic, so
# A carries only u-harmonics +-1 and B only {0, +-2}, for every (ra, dec, incl).
# The banded features do not use that response -- "freqresponse" and "rotation"
# build their coefficients from the arm vectors and a time-varying orientation --
# so the factorization, and with it the identity, is not guaranteed there.
#
# READ THE ALLOWLIST POSITIVELY, because the negative phrasing inverts: the ONLY
# admitted value is the static response, which is the ABSENCE of a feature tag
# (None).  Every named feature -- "rotation", "freqresponse", and anything added
# later -- is refused.  Fail-closed by construction: a new response model must be
# added to this tuple deliberately rather than inherit a placement whose premise
# nobody checked.
#
# SCOPE, so this is not over-read: the identity and this gate are about the PSI
# axis.  Under precession the PHI content of the coefficient tables IS
# materially redistributed (measured 2026-09-02 on SEOBNRv5PHM: the A phi-slot-0
# weight moves from 1.1e-16 aligned to 2.9e-2 precessing, while staying
# band-limited to ~6e-15), and that is a property of the SOURCE, not the
# detector.  It is not an identity failure and this gate is right to admit it --
# the psi harmonics are unchanged -- but "the identity holds under precession"
# must not be read as a statement about the phi axis.
_GH_PSI_STATIC_FEATURES = (None,)

# Bin denominators are floored at this fraction of their own global maximum, so
# that response-free bins (numerator and denominator both ~0) are not scored as
# 0/0 violations.  Small enough that a bin carrying any real response is judged
# on its own scale.
_GH_PSI_BIN_FLOOR = 1e-6


def psi_harmonics_at_phi(C_A, C_B, phi, m_max):
    """The psi-Fourier FIELDS (A0, A1, B0, B1, B2) at the given phi points.

    A(u) = A0 + Re(A1 e^{iu});  B(u) = B0 + Re(B1 e^{iu}) + Re(B2 e^{2iu}), u = 2 psi.

    THE ONLY DEFINITION.  Both the laplace kernel and :func:`gh_laplace_supported`
    call this, so the identity the predicate measures is by construction the one
    the kernel's node placement depends on.  They were separate once, and the
    predicate silently measured a DIFFERENT quantity: it read the real part of the
    coefficient SLICE ``C_A[:, 1]`` rather than of the phi-reconstruction
    ``MA(1)``, so a purely imaginary coefficient gave a nonzero A0(phi) that the
    check reported as zero; and it read only ``C_B[:, 3]`` while the field also
    carries ``conj(C_B[:, 1])``.  Either could pass a dataset whose identity does
    not hold.  Do not re-derive these five lines anywhere.
    """
    phi = jnp.asarray(phi, dtype=jnp.float64)
    wA = _kp_weights(m_max + 1)
    wB = _kp_weights(2 * m_max + 1)
    kpA = jnp.arange(m_max + 1, dtype=jnp.float64)
    kpB = jnp.arange(2 * m_max + 1, dtype=jnp.float64)
    EA = jnp.exp(1j * phi[:, None] * kpA[None, :]) * wA[None, :]
    EB = jnp.exp(1j * phi[:, None] * kpB[None, :]) * wB[None, :]
    MA = lambda k: jnp.einsum("ck,kst->cst", EA, C_A[:, k])
    MB = lambda k: jnp.einsum("ck,kst->cst", EB, C_B[:, k])
    return (MA(1).real,                    # A0   (ks index 1 == ks 0)
            MA(2) + jnp.conj(MA(0)),       # A1   (ks +1 plus conj(ks -1))
            MB(2).real,                    # B0
            MB(3) + jnp.conj(MB(1)),       # B1
            MB(4) + jnp.conj(MB(0)))       # B2


# Generic probe direction for the build-time identity check.  The A0==0/B1==0
# identity is a property of the spin-2 detector response, so it does not depend
# on where we probe; a single generic (ra, dec, incl) away from any pole or
# face-on/edge-on special case is enough, and keeps the check O(1).
#
# These lived in wrapper.py, reached from its one call site.  They are here
# because there are now TWO consumers -- the angle scheme and the policy's
# reserve roster -- and a second copy of a probe direction is a second
# definition of what "the identity holds on this data" means.
_GH_PROBE_RA = (1.0,)
_GH_PROBE_DEC = (0.3,)
_GH_PROBE_INCL = (1.0,)


def _gh_laplace_precondition(m_max, feature):
    """The two conditions that need no tables.  ``None`` means both hold.

    Split out because there are two entry points and the cheap half must run
    FIRST at both of them.  ``gh_laplace_supported_for_data`` builds tables, and
    building them for a banded dataset takes the banded accumulation route --
    a different code path, which fails on its own terms before the response
    model is ever looked at.  So the caller that starts from a dataset has to
    be able to refuse before it builds anything.
    """
    if int(m_max) > _GH_PSI_M_MAX:
        # The tables are SIZED by m_max, so a mismatched m_max is a shape error
        # rather than a measurement.
        return dict(gh_laplace_ok=False, m_max=int(m_max),
                    identity_A0_over_A1=None, identity_B1_over_B0=None,
                    feature=feature,
                    gh_laplace_reason="mode content m_max=%d above the "
                                      "validated %d"
                                      % (int(m_max), _GH_PSI_M_MAX))
    # ANGLE-INDEPENDENT CONDITION, and the one that actually generalises.  A
    # numerical check can only ever speak for the angles it was evaluated at,
    # and the placement runs at arbitrary sampled angles; the response model is
    # a property of the packed data and holds for all of them.
    if feature not in _GH_PSI_STATIC_FEATURES:
        return dict(gh_laplace_ok=False, m_max=int(m_max),
                    identity_A0_over_A1=None, identity_B1_over_B0=None,
                    feature=feature,
                    gh_laplace_reason="response model %r does not give "
                                      "the exact e^{-2i psi} polarization "
                                      "factorization the A0 == 0 / B1 == 0 "
                                      "identity rests on" % (feature,))
    return None


def gh_laplace_supported_for_data(data, interp=JAX_INTERP_DEFAULT):
    """:func:`gh_laplace_supported` on tables this builds at a probe direction.

    THE ONLY WAY to ask the question of a dataset rather than of a table.  The
    predicate itself cannot be called under jit/grad -- the tables are tracers
    there -- so every caller needs concrete tables, and every caller that builds
    its own would be choosing its own probe direction.

    Returns ``(ok, info)`` exactly as :func:`gh_laplace_supported` does.  Costs
    one O(1) table build.  Says nothing about whether the distance quadrature in
    use NEEDS the identity: that is the caller's condition (it is needed by the
    per-sample adaptive node placement, not by a static grid).
    """
    m_max = _data_m_max(data)
    feature = getattr(data, "feature", None)
    # Cheap half first: see _gh_laplace_precondition.  Building the probe
    # tables for a banded dataset would take the banded route and raise on its
    # own missing fields, so a refusal here must not depend on tables.
    refused = _gh_laplace_precondition(m_max, feature)
    if refused is not None:
        return False, refused
    C_A, C_B = angle_coefficient_tables(
        data,
        jnp.asarray(_GH_PROBE_RA, dtype=jnp.float64),
        jnp.asarray(_GH_PROBE_DEC, dtype=jnp.float64),
        jnp.asarray(_GH_PROBE_INCL, dtype=jnp.float64),
        interp)[:2]
    return gh_laplace_supported(C_A, C_B, m_max, feature=feature)


def gh_laplace_supported(C_A, C_B, m_max, feature=None):
    """May 'laplace' use the per-sample adaptive distance quadrature on THIS data?

    Returns ``(ok, info)``.  Two conditions, both necessary:

    1. ``m_max <= _GH_PSI_M_MAX`` -- what the path is shipped and validated for.
    2. The A0 == 0 / B1 == 0 identity actually HOLDS on these coefficient
       tables, MEASURED rather than assumed.

    (2) exists because (1) does not imply it.  m_max is the largest |m| in the
    mode list, so a PRECESSING l=2 system has m_max = 2 and passes (1) while
    breaking the aligned-spin symmetry h_{l,-m} = (-1)^l conj(h_lm) that the
    identity has only ever been tested under.  Every measurement of the identity
    to date -- IMRPhenomXHM through m_max = 4, and a zero-spin SEOBNRv5PHM run --
    is non-precessing.  The analytic argument (F+ + i Fx ~ e^{-2i psi} is one
    psi-harmonic, so A is linear in it and B quadratic, making this a property of
    the DETECTOR) says it should extend; what has been measured is the CODE's
    tables, and two non-precessing tests cannot separate those.  So the code
    CHECKS instead of trusting the argument: cost is O(size of the coefficient
    tables), once, at build time.
    """
    import numpy as _np
    # Measure the RECONSTRUCTED fields the kernel uses, on a phi grid dense
    # enough to resolve their phi content (harmonics to 2*m_max), NOT the
    # coefficient slices -- see psi_harmonics_at_phi's docstring for the two
    # ways reading slices gave the wrong answer.
    refused = _gh_laplace_precondition(m_max, feature)
    if refused is not None:
        return False, refused
    n_phi_probe = max(8 * int(m_max) + 8, 16)
    phi_probe = _np.linspace(0.0, 2.0 * _np.pi, n_phi_probe, endpoint=False)
    A0f, A1f, B0f, B1f, B2f = psi_harmonics_at_phi(C_A, C_B, phi_probe, m_max)
    A0f = _np.abs(_np.asarray(A0f)); A1f = _np.abs(_np.asarray(A1f))
    B0f = _np.abs(_np.asarray(B0f)); B1f = _np.abs(_np.asarray(B1f))
    # POINTWISE, not a ratio of global maxima.  The placement uses the centre
    # and width computed at EACH (phi, sample, time) bin independently, so a
    # single locally invalid bin is enough to invalidate it there -- and
    # max|A0| / max|A1| hides exactly that, because a large A1 somewhere else
    # shrinks the ratio (bins (1e-3, 1) and (0, 1e6) give a passing 1e-9 while
    # the first violates by 1e-3).  Denominators are floored at a small fraction
    # of their own global maximum so that bins with no response at all -- where
    # numerator and denominator are both ~0 and nothing is at stake -- do not
    # register as 0/0 violations.
    a_floor = _GH_PSI_BIN_FLOOR * A1f.max() if A1f.size else 0.0
    b_floor = _GH_PSI_BIN_FLOOR * B0f.max() if B0f.size else 0.0
    r_A0 = float((A0f / _np.maximum(A1f, a_floor)).max()) if a_floor > 0 else _np.inf
    r_B1 = float((B1f / _np.maximum(B0f, b_floor)).max()) if b_floor > 0 else _np.inf
    ok_ident = (r_A0 <= GH_PSI_IDENTITY_TOL) and (r_B1 <= GH_PSI_IDENTITY_TOL)
    if not ok_ident:
        reason = ("the A0==0/B1==0 identity does NOT hold pointwise on this data "
                  "(worst-bin |A0|/|A1|=%.3g, |B1|/B0=%.3g, tol %.0e)"
                  % (r_A0, r_B1, GH_PSI_IDENTITY_TOL))
    else:
        reason = ("m_max=%d, response %r, and the A0==0/B1==0 identity holds at "
                  "every probed bin" % (int(m_max), feature))
    return ok_ident, dict(gh_laplace_ok=bool(ok_ident), feature=feature,
                                         gh_laplace_reason=reason,
                                         identity_A0_over_A1=r_A0,
                                         identity_B1_over_B0=r_B1,
                                         m_max=int(m_max))


def fused_log_likelihood_distphipsimarg_peaklocal(
        data, ra, dec, incl, x_grid, log_w_grid,
        interp=JAX_INTERP_DEFAULT, amp_sizing=None,
        time_quadrature=TIME_QUAD_DEFAULT, return_lnLt=False,
        phi_chunk=None, return_amp=False):
    """Distance-, phi_ref- AND psi-marginalized lnL: PEAK-LOCAL scheme.

    Same contract and normalization as
    :func:`fused_log_likelihood_distphipsimarg_exact`.  What changes is the psi axis:
    rather than a dense grid sized ``~sqrt(A)``, the u-stationary points are obtained
    EXACTLY -- they are the unit-circle roots of a quartic, the u-degree being pinned at
    2 for any mode set -- the sorted points partition the circle, and each cell is
    integrated on a window set by its own curvature.  Windowed cells need only 48 nodes,
    but rejected Newton centres span whole cells, so production sizes the shared static
    count from ``amp_sizing``.  The node axis is streamed in fixed-size blocks; cost grows
    as sqrt(amplitude), while its live memory does not.

    THE PHI AXIS IS STILL DENSE HERE and is sized by
    :func:`~RIFT.likelihood.jax_ile.joint_anglemarg_peaklocal.required_n_phi` from the
    same ``amp_sizing`` the other schemes use.  Localizing phi as well exists as a numpy
    reference; it is not in this jitted path.  See DESIGN_peak_local_framework.md.

    Measured against ``..._exact``: -3.6e-05, -7.1e-15 and -9.1e-13 nats at kappa boost
    1, 10 and 100, and the same figure on a CUDA device as on CPU.

    The adaptive distance quadrature (``JAX_ILE_DISTMARG_GH``) is REFUSED rather than
    silently ignored, for the reason the laplace branch refuses it: this kernel sums the
    caller's distance grid directly and implements no psi-marginal node placement.
    """
    if _core._DISTMARG_GH_N > 0:
        raise ValueError(
            "distance-GH-nodes is set (--distance-gh-nodes / "
            "JAX_ILE_DISTMARG_GH), but the 'peak-local' angle-marg scheme "
            "does not implement the adaptive distance quadrature (it sums "
            "the caller's distance grid directly).  Use --angle-marg-scheme "
            "exact, or pass --distance-gh-nodes 0 (or unset "
            "JAX_ILE_DISTMARG_GH).")
    _require_amp_sizing(amp_sizing)
    from . import joint_anglemarg_peaklocal as _jp

    C_A, C_B, _meta = angle_coefficient_tables(data, ra, dec, incl, interp=interp)

    # THE RUNTIME AMPLITUDE FAILSAFE APPLIES HERE TOO, and omitting it was a review
    # finding rather than a judgement call.  The u axis is localized and needs no
    # sizing, but THE PHI AXIS IS STILL DENSE and is sized from `amp_sizing`, which
    # `estimate_angle_amplitude` is explicit about being an estimator and NOT a proven
    # bound -- so a hotter sampled sky location can under-resolve phi exactly as it can
    # for the exact and laplace schemes.  Skipping the check would publish that
    # silently, and would also leave the artifact without the standing best-effort
    # label, which is worse than the undersizing itself.
    amp_call = _runtime_amp_failsafe(C_A, C_B, x_grid, amp_sizing,
                                     "peak-local")

    n_phi = _jp.required_n_phi(amp_sizing, m_max=_data_m_max(data))
    # Size the u axis through the SINGLE SOURCE OF TRUTH rather than letting the kernel
    # fall back to its own constant: the batch-memory guard in samplers.py models this
    # same number from the same amp_sizing, and the two live in different files.  Passing
    # it explicitly is what makes them provably the same value rather than two defaults
    # that happen to agree.  The derived count is streamed inside the kernel, so raising
    # accuracy does not materialize that entire axis across the outer batches.
    kw = {"n_nodes": _jp.u_nodes_in_use(amp_sizing)}
    if phi_chunk is not None:
        kw["phi_chunk"] = int(phi_chunk)

    # tables are (KP, 2KS+1, S, npts); move the batch axes to the front so one nested
    # vmap covers both and the kernel sees a plain 2-D table per (sample, time).
    A = jnp.moveaxis(jnp.asarray(C_A), (2, 3), (0, 1))
    B = jnp.moveaxis(jnp.asarray(C_B), (2, 3), (0, 1))

    def _one(a, b):
        return _jp.joint_lnL_phi_dense(a, b, x_grid, log_w_grid, n_phi=n_phi, **kw)

    lnL_t = jax.vmap(jax.vmap(_one))(A, B)          # (S, npts)
    out = lnL_t if return_lnLt else _time_marginalize_terminal(
        lnL_t, data, time_quadrature)
    return (out, amp_call) if return_amp else out


def fused_log_likelihood_distphipsimarg_phi_local(
        data, ra, dec, incl, x_grid, log_w_grid,
        interp=JAX_INTERP_DEFAULT, amp_sizing=None,
        time_quadrature=TIME_QUAD_DEFAULT, return_lnLt=False,
        x_chunk=None, pt_chunk=None, n_slots=None, return_ok=False,
        return_amp=False):
    """Distance-, phi_ref- AND psi-marginalized lnL with BOTH ANGLE AXES LOCALIZED.

    Same contract and normalization as the other ``fused_log_likelihood_distphipsimarg_*``
    entries.  What changes against ``peak-local`` is the phi axis: rather than a dense grid
    sized ``~sqrt(A)``, phi is localized around the maxima of the u-profile and the omitted
    mass is BOUNDED, so cost stops growing with amplitude.

    IT DECLINES, AND THE CALLER GETS THE DENSE ANSWER WHEN IT DOES.  ``phi_local_lnI`` is
    fail-closed: a row whose omitted-mass bound, convergence probes or u sizing do not pass
    returns ``ok = False``, and across a distance grid ``ok`` is the CONJUNCTION over nodes.
    This entry evaluates the dense peak-local scheme as well and selects elementwise, so a
    decline costs time and never accuracy.  Pass ``return_ok=True`` to get the mask and
    account for how often the localized path actually carried the row -- a scheme that
    silently fell back on every sample would otherwise look like it worked.

    THAT MAKES THIS SLOWER THAN ``peak-local`` UNTIL THE FALLBACK CAN BE SKIPPED, which
    needs an acceptance rate measured on production tables rather than assumed.  It is
    therefore reachable only by name and is not in ``auto``.

    ``JAX_ILE_DISTMARG_GH`` is REFUSED for the same reason the dense peak-local branch
    refuses it: no psi-marginal node placement exists yet.  The seam it will attach to,
    :func:`~RIFT.likelihood.jax_ile.joint_anglemarg_peaklocal.phi_local_lnI_at_distance`,
    is public and takes one distance node, so that work does not have to modify this
    function.
    """
    if _core._DISTMARG_GH_N > 0:
        raise ValueError(
            "distance-GH-nodes is set (--distance-gh-nodes / "
            "JAX_ILE_DISTMARG_GH), but the 'phi-local' angle-marg scheme "
            "does not implement the adaptive distance quadrature (it sums "
            "the caller's distance grid directly).  Use --angle-marg-scheme "
            "exact, or pass --distance-gh-nodes 0 (or unset "
            "JAX_ILE_DISTMARG_GH).")
    _require_amp_sizing(amp_sizing)
    from . import joint_anglemarg_peaklocal as _jp

    C_A, C_B, _meta = angle_coefficient_tables(data, ra, dec, incl, interp=interp)
    amp_call = _runtime_amp_failsafe(C_A, C_B, x_grid, amp_sizing,
                                     "phi-local")

    n_phi = _jp.required_n_phi(amp_sizing, m_max=_data_m_max(data))
    u_nodes = _jp.u_nodes_in_use(amp_sizing)
    kw = {"u_nodes": u_nodes,
          "n_bound": int(_jp.required_bound_grid(amp_sizing)),
          "pt_chunk": int(pt_chunk if pt_chunk is not None else _jp.PT_CHUNK_DEFAULT)}
    if n_slots is not None:
        kw["n_slots"] = int(n_slots)

    A = jnp.moveaxis(jnp.asarray(C_A), (2, 3), (0, 1))
    B = jnp.moveaxis(jnp.asarray(C_B), (2, 3), (0, 1))
    xc = int(x_chunk if x_chunk is not None else _jp.X_CHUNK_DEFAULT)

    def _one(a, b):
        loc, ok, _ = _jp.joint_lnL_phi_local(a, b, x_grid, log_w_grid,
                                             x_chunk=xc, **kw)
        dense = _jp.joint_lnL_phi_dense(a, b, x_grid, log_w_grid, n_phi=n_phi,
                                        n_nodes=u_nodes)
        return jnp.where(ok, loc, dense), ok

    lnL_t, ok_t = jax.vmap(jax.vmap(_one))(A, B)          # (S, npts) each
    # return_amp appends the amplitude metric as the LAST element whatever the
    # other flags select, so the three optional returns compose unambiguously.
    out = lnL_t if return_lnLt else _time_marginalize_terminal(
        lnL_t, data, time_quadrature)
    if return_ok:
        return (out, ok_t, amp_call) if return_amp else (out, ok_t)
    return (out, amp_call) if return_amp else out


def _multipeak_sigma_t(rows_A, guard):
    """Predicted time-peak width in native samples, from the tables alone.

    sigma_t = 1 / (2 pi rho sigma_f).  sigma_f is the RMS frequency of the time
    primitive's own spectrum, in cycles per sample, so no rate conversion enters;
    rho comes from the exponent's peak.  Both are properties of the data, which is
    the point: the reserve's node placement is PREDICTED before any evaluation
    rather than discovered by evaluating everywhere.
    """
    from .time_first_peaklocal import _time_primitive_spectrum
    flat = jnp.asarray(rows_A[0]).reshape((-1, rows_A.shape[-1]))
    coeff, freq, _ = _time_primitive_spectrum(flat, int(guard))
    p = np.abs(np.asarray(coeff)) ** 2
    f = np.asarray(freq)
    w = p.sum(axis=0)
    sigma_f = float(np.sqrt((w * f * f).sum() / max(w.sum(), 1.0e-300)))
    env = np.abs(np.asarray(rows_A)).sum(axis=(1, 2))
    rho = float(np.sqrt(2.0 * np.max(env)))
    return 1.0 / max(2.0 * np.pi * rho * sigma_f, 1.0e-300)


def _multipeak_reserve_rule(CA_full, guard, data, sigma_t, n_sigma, pts_per_sigma):
    """Peak-local time nodes and trapezoid weights, in production time units.

    Node count does not grow with rho: the window is +-n_sigma sigma_t and the
    spacing is sigma_t / pts_per_sigma, so the two scale together.
    """
    from .all_axis_peaklocal import (_time_primitive_spectrum,
                                     _evaluate_time_spectrum)
    n_nat = CA_full.shape[-1] - 2 * int(guard)
    enum = np.arange(0.0, n_nat - 1 + 1.0e-9, 0.25)
    flat = jnp.asarray(CA_full).reshape((-1, CA_full.shape[-1]))
    coeff, freq, off = _time_primitive_spectrum(flat, int(guard))
    env = np.abs(np.asarray(_evaluate_time_spectrum(
        coeff, freq, jnp.asarray(enum), off))).sum(axis=0)
    keep = env >= env.max() * np.exp(-0.5 * 40.0 / max(env.max(), 1.0e-300))
    peaks = [j for j in range(1, len(env) - 1)
             if env[j] >= env[j - 1] and env[j] >= env[j + 1] and keep[j]]
    if not peaks:
        peaks = [int(np.argmax(env))]
    half = float(n_sigma) * float(sigma_t)
    step = float(sigma_t) / float(pts_per_sigma)
    wins = sorted((max(0.0, enum[j] - half), min(float(n_nat - 1), enum[j] + half))
                  for j in peaks)
    merged = [list(wins[0])]
    for w0, w1 in wins[1:]:
        if w0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], w1)
        else:
            merged.append([w0, w1])
    nd, wt = [], []
    for w0, w1 in merged:
        k = max(2, int(np.ceil((w1 - w0) / step)) + 1)
        g = np.linspace(w0, w1, k)
        h = (w1 - w0) / (k - 1)
        ww = np.full(k, h)
        ww[0] = ww[-1] = 0.5 * h
        nd.append(g)
        wt.append(ww)
    nodes = np.concatenate(nd)
    # BUCKET THE SHAPE so a batch compiles once, not once per row.  Padding is
    # zero-weight, so the value is identical.
    weights = np.concatenate(wt)
    tgt = next((b for b in (256, 512, 1024, 2048) if b >= len(nodes)), len(nodes))
    pad = tgt - len(nodes)
    if pad > 0:
        # compute `pad` BEFORE reassigning nodes: deriving it from len(nodes)
        # afterwards gives zero and leaves weights shorter than nodes.
        nodes = np.concatenate([nodes, np.full(pad, nodes[-1])])
        weights = np.concatenate([weights, np.zeros(pad)])
    assert len(nodes) == len(weights) == tgt
    w_t = np.asarray(data.w_t, dtype=float)
    scale = float(np.sum(w_t)) / ((int(data.npts) - 1) * float(data.deltaT))
    return jnp.asarray(nodes), jnp.asarray(weights * float(data.deltaT) * scale)


def fused_log_likelihood_distphipsimarg_multipeak(
        data, ra, dec, incl, x_grid, log_w_grid,
        interp=JAX_INTERP_DEFAULT, amp_sizing=None, guard=16,
        tier0=(2, 3, 24), tier1=(3, 5, 48), log_integral_tol=1.0e-3,
        cell_sigma=5.0, quadrature_order=7, refine_iterations=18,
        reserve_sigma=12.0, reserve_pts_per_sigma=8.0, return_record=False):
    """Distance-, phi_ref-, psi- AND time-marginalized lnL: MULTIPEAK scheme.

    The four-axis controller of
    :func:`~RIFT.likelihood.jax_ile.multipeak_planner.multipeak_local_marginalize`,
    reachable from ``--angle-marg-scheme multipeak``.  Unlike every other entry
    in this family it OWNS THE TIME INTEGRAL, so there is no ``lnL(t)`` and no
    ``time_quadrature``: the caller gets one value per sample.  Callers that
    need ``lnL(t)`` must use another scheme.

    The default operating point is the one measured on the ladder-2 injection at
    rho 40.77, 163.08 and 652.31 (64 rows per rung, inclination banded +-0.20 rad
    about the injection): 64/64 accepted at every rung, 4.74-5.17 s per row and
    ~218 MiB peak device memory, error against a peak-local reference of 2.2e-05
    nats median at rho 40.77 and 5.6e-04 at 163.08.  See
    analyses/va_sequence_20260902/RESULTS_20260909_multipeak_ladder.md in the
    RIFT_roboto_paper record store.  ``guard`` defaults to the value that
    measurement used; the driver's production default is larger and the caller
    passes it explicitly.

    The reserve is a PEAK-LOCAL time rule, not a refined whole window: nodes are
    placed only in +-``reserve_sigma`` sigma_t windows about the peaks of the
    coefficient envelope, at spacing sigma_t/``reserve_pts_per_sigma``.  Window
    and spacing both scale as sigma_t, so the node count does not grow with rho.
    A refined whole-window reserve was measured at 4905 nodes for a peak 0.06
    native samples wide at rho 163, and its own half-refined warrant failed at
    rho 40.77; this rule reproduced it to 0.0 on every row with 193 nodes.
    """
    from . import multipeak_planner as _mp
    from . import all_axis_peaklocal as _aap
    from . import direct_marginalization_policy as _pol
    from .core import _time_marginalize

    _require_amp_sizing(amp_sizing)
    guard = int(guard)
    if guard < 2:
        raise ValueError("multipeak needs guard >= 2 for the time primitive")
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    C_A, C_B, meta = angle_coefficient_tables(data, ra, dec, incl, interp,
                                              guard=guard)
    _runtime_amp_failsafe(C_A, C_B, x_grid, amp_sizing, "multipeak")
    log_measure, _ = _pol.policy_log_normalization(
        data, np.asarray(x_grid), np.asarray(log_w_grid))
    x_min = float(np.min(np.asarray(x_grid)))
    x_max = float(np.max(np.asarray(x_grid)))
    m_max = int(meta["m_max"])

    rows_A = np.moveaxis(np.asarray(C_A), 2, 0)          # (S,KP,KS,ntime+2g)
    rows_B = np.moveaxis(np.asarray(C_B), 2, 0)
    sigma_t = _multipeak_sigma_t(rows_A, guard)

    out, records = [], []
    for i in range(rows_A.shape[0]):
        CA_full = rows_A[i]
        CA_i = CA_full[..., guard:-guard]
        CB_i = rows_B[i][..., 0][..., None] * np.ones(CA_i.shape[-1])
        nodes, weights = _multipeak_reserve_rule(
            CA_full, guard, data, sigma_t,
            float(reserve_sigma), float(reserve_pts_per_sigma))

        CB_flat = rows_B[i][..., 0]                  # (KP,KS), time-independent

        def _reserve(CA_full=CA_full, CB_flat=CB_flat, nodes=nodes,
                     weights=weights):
            flat = jnp.asarray(CA_full).reshape((-1, CA_full.shape[-1]))
            coeff, freq, off = _aap._time_primitive_spectrum(flat, guard)
            tgt = _aap._evaluate_time_spectrum(
                coeff, freq, nodes, off).reshape(
                    CA_full.shape[:-1] + (nodes.size,))
            lnLt = coefficient_table_distphipsimarg_laplace(
                tgt, jnp.asarray(CB_flat), x_grid, log_w_grid,
                amp_sizing=amp_sizing, m_max=m_max)
            return float(np.asarray(_time_marginalize(lnLt, weights)[0]))

        res = _mp.multipeak_local_marginalize(
            CA_i, CB_i, x_min, x_max, _reserve,
            log_integral_tol=float(log_integral_tol),
            tier0=tuple(int(v) for v in tier0),
            tier1=tuple(int(v) for v in tier1),
            refine_iterations=int(refine_iterations),
            cell_sigma=float(cell_sigma),
            quadrature_order=int(quadrature_order),
            log_measure=float(log_measure), label="multipeak_row%d" % i)
        out.append(float(res.value))
        records.append(res)
    values = jnp.asarray(np.asarray(out, dtype=float))
    return (values, records) if return_record else values


def choose_angle_marg_scheme(amplitude, gh_enabled=None,
                             gh_laplace_ok=None):
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
    if gh_enabled and not gh_laplace_ok:
        # 'laplace' CAN use the adaptive distance quadrature now, but only where
        # gh_laplace_supported() says so.  Selecting it anywhere else would route
        # 'auto' into the raise inside the laplace kernel, so this branch is
        # deliberately more conservative than that kernel's own gate: when the
        # caller does not supply the predicate at all (gh_laplace_ok=None) we
        # take the safe branch rather than guess.
        return "exact", dict(reason="JAX_ILE_DISTMARG_GH set and the laplace "
                                    "psi-marginal node placement is not "
                                    "available for this data",
                             amplitude=amp,
                             crossover=ANGLE_MARG_CROSSOVER_AMPLITUDE)
    # NOTE, and it is a live limitation rather than a subtlety:
    # ANGLE_MARG_CROSSOVER_AMPLITUDE is an ACCURACY crossover (A=450, rho~30) --
    # the point above which BOTH schemes are accurate.  The measured COST
    # crossover is rho ~200-326 (A ~2e4-5e4), an order of magnitude higher, so
    # between them 'auto' picks the accurate-but-slower scheme.  Re-deriving the
    # constant from cost as well as accuracy is a separate, measured change; it
    # is deliberately NOT folded in here.
    scheme = "laplace" if amp >= ANGLE_MARG_CROSSOVER_AMPLITUDE else "exact"
    return scheme, dict(reason="measured amplitude bound %s crossover"
                               % ("above" if scheme == "laplace" else "below"),
                        amplitude=amp,
                        crossover=ANGLE_MARG_CROSSOVER_AMPLITUDE)
