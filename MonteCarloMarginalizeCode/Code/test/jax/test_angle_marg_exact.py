"""
Gate for RIFT.likelihood.jax_ile.anglemarg: the exact (phi_ref, psi)
marginalization schemes and their selector.

WHAT IS PINNED, AND WHY THESE PARTICULAR TESTS
----------------------------------------------
The module rests on one analytic fact: at fixed time and distance the factored
lnL is a bivariate trig polynomial -- phi_ref order <= 2*m_max, u = 2 psi
order <= 2.  Everything else (Nyquist sample sizing, coefficient tables,
dense reconstruction, the psi Laplace) is bookkeeping on top of that fact, and
each layer of the bookkeeping is pinned here:

  * the harmonic-content invariant itself (sampler-free, injection-free; it
    catches exactly what a rewrite gets wrong: hermiticity of U/V handling,
    mode conventions, array alignment).  CAVEAT honoured here: the invariant
    holds for the UNMARGINALIZED lnL at fixed time -- log-sum-exp over time
    destroys the polynomial structure and manufactures fake high harmonics,
    so every decomposition below fixes the time index;
  * the coefficient tables reproduce the direct likelihood OFF the sample
    grid (trig interpolation exactness);
  * both schemes against a brute-force dense reference (which converges TO
    the exact answer), and against the legacy grid path where that path is
    converged (pins the shared normalization convention);
  * the historical nphi=8 Nyquist aliasing of the n=4 phi harmonic, at the
    DFT-coefficient level and at the marginal level (the regression that
    motivated the module);
  * exact/laplace agreement in the selector's overlap region -- the runtime
    check that makes the crossover a validated constant, not a tuning knob;
  * gradients (the point of this code path) against finite differences;
  * the wrapper selector, its provenance record, and the grid default being
    byte-identical to the legacy path (no default change);
  * the driver flag actually reaching the wrapper and the resolved scheme
    being printed (this pipeline has a documented history of silently-inert
    flags), via AST over the driver source.

Synthetic packed data (no frames, no waveform generation) keeps this fast and
CI-friendly; the trig-polynomial structure is a property of the accumulation
ALGEBRA, not of any particular rholm/U/V values, so random-but-structured
tensors (U Hermitian PSD, V symmetric) exercise it fully.
"""

import ast
import os
import types

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import build_likelihood_data
from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile.core import (
    _accumulate_unit, _time_marginalize, _logsumexp_grid_blocked,
    fused_log_likelihood_distphipsimarg, phi_ref_grid, psi_grid,
    make_distance_grid)
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood

INTERP = "sinc"


# ---------------------------------------------------------------------------
# synthetic likelihood data
# ---------------------------------------------------------------------------

def make_synth(scale=1.0, seed=3, modes=((2, 2), (2, -2)), npts=32,
               deltaT=1.0 / 1024, kappa_boost=1.0):
    """Structurally-faithful synthetic packed data (cf. test_jax_likelihood).

    U is Hermitian positive definite and V complex symmetric, as the real
    precompute produces; ``scale`` sets the overall amplitude (lnL ~ scale^2),
    standing in for SNR.  ``kappa_boost`` multiplies the rholm timeseries
    ONLY (not U/V), producing a target with a large coherent (phi,psi)
    amplitude A -- the regime where an undersized dense grid measurably
    biases the marginal (used by the sizing regression tests).
    """
    rng = np.random.default_rng(seed)
    tw = npts * deltaT / 2.0
    tvals = np.linspace(-tw, tw, npts)
    tref = 1126259462.413
    K = len(modes)
    packed = {}
    for det in ("H1", "L1"):
        npts_full = 4096
        white = (rng.standard_normal((K, npts_full))
                 + 1j * rng.standard_normal((K, npts_full)))
        kx = np.arange(-40, 41)
        kern = np.exp(-0.5 * (kx / 12.0) ** 2)
        kern /= kern.sum()
        rho = np.stack([np.convolve(white[k].real, kern, "same")
                        + 1j * np.convolve(white[k].imag, kern, "same")
                        for k in range(K)]).astype(np.complex128)
        rho *= np.sqrt(len(kx)) * scale * kappa_boost
        M = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        U = (M @ M.conj().T + 3 * np.eye(K)) * scale ** 2
        B = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        V = (B @ B.T) * scale ** 2 * 0.3
        packed[det] = dict(lms=np.array(modes, dtype=int), rholmArray=rho,
                           U=U, V=V, epoch=tref - 0.5)
    return build_likelihood_data(packed, deltaT, tref, tvals)


RA, DEC, INCL = np.array([0.9]), np.array([0.4]), np.array([1.1])
S = 1


def _dist_grid(data, n=64):
    return make_distance_grid(30.0, 3000.0, n, distMpcRef=data.distMpcRef)


def brute_marginal(data, x_grid, log_w, nphi, npsi):
    """Brute-force dist+phi+psi marginal: dense product grid of DIRECT
    likelihood evaluations (no coefficient machinery shared with the schemes
    under test)."""
    ph = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    ps = np.linspace(0, np.pi, npsi, endpoint=False)
    m = jnp.full((S, data.npts), -jnp.inf)
    s = jnp.zeros((S, data.npts))
    for p in ph:
        rb = np.repeat(RA[None, :], npsi, 0).ravel()
        db = np.repeat(DEC[None, :], npsi, 0).ravel()
        ib = np.repeat(INCL[None, :], npsi, 0).ravel()
        pb = np.full(npsi * S, p)
        sb = np.repeat(ps[:, None], S, 1).ravel()
        ku, rs = _accumulate_unit(data, rb, db, sb, ib, pb, INTERP, False)
        lnL = _logsumexp_grid_blocked(ku.real, rs, x_grid,
                                      -0.5 * jnp.square(x_grid), log_w, 64)
        m, s = AM._lse_update(m, s, lnL.reshape(npsi, S, data.npts), axis=0)
    lnL_t = m + jnp.log(s) - np.log(nphi * npsi)
    return np.asarray(_time_marginalize(lnL_t, data.w_t))


def _lnL_t_fixed_time(data, phis, psis, x, t_index):
    """UNMARGINALIZED lnL at fixed time bin and fixed distance factor x."""
    n = len(phis)
    ku, rs = _accumulate_unit(data, np.full(n, RA[0]), np.full(n, DEC[0]),
                              np.asarray(psis), np.full(n, INCL[0]),
                              np.asarray(phis), INTERP, False)
    return (np.asarray(ku.real)[:, t_index] * x
            - 0.5 * np.asarray(rs)[:, t_index] * x ** 2)


# ---------------------------------------------------------------------------
# 1. the harmonic-content invariant
# ---------------------------------------------------------------------------

def test_harmonic_content_psi():
    """lnL_t(psi) at fixed (phi, t, x) has u-harmonics {0,1,2} ONLY."""
    data = make_synth(scale=2.0)
    n = 64
    psis = np.linspace(0, np.pi, n, endpoint=False)
    f = _lnL_t_fixed_time(data, np.full(n, 1.3), psis, 0.8, t_index=10)
    C = np.fft.rfft(f) / n
    power = np.abs(C)
    assert power[:3].max() > 0
    # everything above u-order 2 is numerically zero
    assert power[3:].max() < 1e-12 * power.max()


def test_harmonic_content_phi():
    """lnL_t(phi) at fixed (psi, t, x) has harmonics <= 2*m_max, odd ones zero
    for a (2,+-2)-only mode set."""
    data = make_synth(scale=2.0)
    n = 64
    phis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    f = _lnL_t_fixed_time(data, phis, np.full(n, 0.6), 0.8, t_index=10)
    C = np.fft.rfft(f) / n
    power = np.abs(C)
    m_max = 2
    assert power[: 2 * m_max + 1].max() > 0
    assert power[2 * m_max + 1:].max() < 1e-12 * power.max()
    # (2,+-2) only: odd phi harmonics identically absent
    assert power[1] < 1e-12 * power.max()
    assert power[3] < 1e-12 * power.max()


def test_time_marginalization_destroys_the_invariant():
    """The caveat that confounded a first analysis, pinned so nobody re-learns
    it: after log-sum-exp over TIME the psi decomposition has fake high
    harmonics.  (Guards the tests above against being 'simplified' onto the
    marginalized quantity.)"""
    data = make_synth(scale=6.0)
    n = 64
    psis = np.linspace(0, np.pi, n, endpoint=False)
    ku, rs = _accumulate_unit(data, np.full(n, RA[0]), np.full(n, DEC[0]),
                              psis, np.full(n, INCL[0]), np.full(n, 1.3),
                              INTERP, False)
    lnL_t = np.asarray(ku.real) * 0.8 - 0.5 * np.asarray(rs) * 0.8 ** 2
    f = np.asarray(_time_marginalize(jnp.asarray(lnL_t), data.w_t))
    C = np.abs(np.fft.rfft(f) / n)
    assert C[3:].max() > 1e-9 * C.max()


# ---------------------------------------------------------------------------
# 2. sample-grid sizing is derived and asserted, not settable
# ---------------------------------------------------------------------------

def test_sample_grid_sizes():
    assert AM.angle_sample_grid_sizes(2) == (16, 8)
    for m_max in (1, 2, 3, 4, 5):
        nphi_s, npsi_s = AM.angle_sample_grid_sizes(m_max)
        # strictly above Nyquist for the highest harmonic present
        assert nphi_s > 2 * (2 * m_max)
        assert npsi_s > 2 * 2
    with pytest.raises(ValueError):
        AM.angle_sample_grid_sizes(0)
    # the public entry points take NO sample-size argument at all
    import inspect
    for fn in (AM.fused_log_likelihood_distphipsimarg_exact,
               AM.fused_log_likelihood_distphipsimarg_laplace):
        assert not any("nphi" in p or "npsi" in p
                       for p in inspect.signature(fn).parameters)


# ---------------------------------------------------------------------------
# 3. coefficient tables reproduce the direct likelihood OFF the sample grid
# ---------------------------------------------------------------------------

def test_coefficient_tables_reconstruct_off_grid():
    data = make_synth(scale=3.0)
    C_A, C_B, meta = AM.angle_coefficient_tables(data, RA, DEC, INCL,
                                                 interp=INTERP)
    assert meta["m_max"] == 2 and meta["nphi_s"] == 16 and meta["npsi_s"] == 8
    rng = np.random.default_rng(11)
    phis = rng.uniform(0, 2 * np.pi, 7)
    psis = rng.uniform(0, np.pi, 7)
    A_rec = np.asarray(AM._reconstruct_field(C_A, jnp.asarray(phis),
                                             jnp.asarray(2 * psis)))
    B_rec = np.asarray(AM._reconstruct_field(C_B, jnp.asarray(phis),
                                             jnp.asarray(2 * psis)))
    ku, rs = _accumulate_unit(data, np.full(7, RA[0]), np.full(7, DEC[0]),
                              psis, np.full(7, INCL[0]), phis, INTERP, False)
    A_dir = np.asarray(ku.real)[:, None, :]
    B_dir = np.asarray(rs)[:, None, :]
    ref = max(np.abs(A_dir).max(), np.abs(B_dir).max())
    assert np.abs(A_rec - A_dir).max() < 1e-10 * ref
    assert np.abs(B_rec - B_dir).max() < 1e-10 * ref


# ---------------------------------------------------------------------------
# 4/5. schemes against brute force and against the converged legacy grid
# ---------------------------------------------------------------------------

def test_exact_scheme_vs_bruteforce():
    data = make_synth(scale=6.0)
    x_grid, log_w = _dist_grid(data)
    ref = brute_marginal(data, x_grid, log_w, 96, 48)
    ex = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
    assert np.abs(ex - ref).max() < 1e-10


def test_exact_matches_legacy_grid_where_converged():
    """Same normalization convention as the production grid path: at low
    amplitude the legacy 32x8 grid is converged and the two must agree."""
    data = make_synth(scale=2.0)
    x_grid, log_w = _dist_grid(data)
    leg = np.asarray(fused_log_likelihood_distphipsimarg(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, phi_ref_grid(32), psi_grid(8), interp=INTERP))
    ex = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
    # legacy 32x8 truncation at this amplitude measured 2.2e-8; the bound
    # pins the shared normalization convention, not the grid's residual
    assert np.abs(ex - leg).max() < 1e-6


def test_laplace_high_amplitude_accuracy_and_trend():
    """Laplace error is small at high amplitude and SHRINKS as amplitude
    grows (it is O(1/A)); measured on this configuration: 1.8e-2 at scale 50,
    5.6e-3 at scale 100."""
    errs = []
    for scale in (50.0, 100.0):
        data = make_synth(scale=scale)
        x_grid, log_w = _dist_grid(data)
        ref = brute_marginal(data, x_grid, log_w, 192, 96)
        lp = np.asarray(AM.fused_log_likelihood_distphipsimarg_laplace(
            data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
            x_grid, log_w, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
        errs.append(np.abs(lp - ref).max())
    # measured on this configuration: 0.055 at scale 50, 0.028 at scale 100.
    # NOTE this synthetic target is Laplace's WORST case (noise-like data, no
    # coherent peak: every bin sits near the small-b regime); the real-signal
    # injection ladder in the PR measures |laplace-exact| ~ 1e-3 at A=50 and
    # falling.  The bound here is a regression pin, not the operating error.
    assert errs[0] < 0.15
    assert errs[1] < 0.08
    assert errs[1] < errs[0]


def test_overlap_agreement_exact_vs_laplace():
    """The selector's runtime validity check: in the overlap region the two
    schemes agree, so the crossover cannot be silently mis-set -- either
    branch is accurate there."""
    data = make_synth(scale=100.0)
    x_grid, log_w = _dist_grid(data)
    args = (data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
            x_grid, log_w)
    ex = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        *args, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
    lp = np.asarray(AM.fused_log_likelihood_distphipsimarg_laplace(
        *args, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
    assert np.abs(ex - lp).max() < 0.06


# ---------------------------------------------------------------------------
# 6. the nphi=8 Nyquist aliasing regression
# ---------------------------------------------------------------------------

def test_nphi8_aliases_the_n4_harmonic_dft_level():
    """8-point phi sampling puts the n=4 harmonic AT Nyquist: the DFT bin
    collapses C4 + C-4 = 2 Re C4 and the imaginary part is unrecoverable.
    Pinned at the coefficient level, deterministically."""
    data = make_synth(scale=3.0)
    n_good = 16
    phis16 = np.linspace(0, 2 * np.pi, n_good, endpoint=False)
    f16 = _lnL_t_fixed_time(data, phis16, np.full(n_good, 0.6), 0.8, 10)
    C4_true = np.fft.fft(f16)[4] / n_good
    phis8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    f8 = _lnL_t_fixed_time(data, phis8, np.full(8, 0.6), 0.8, 10)
    C4_alias = np.fft.fft(f8)[4] / 8
    # the alias identity: the 8-point bin 4 is exactly 2*Re(C4), not C4
    assert abs(C4_alias - 2 * C4_true.real) < 1e-12 * abs(C4_true)
    # and the information it destroyed was genuinely there
    assert abs(C4_true.imag) > 1e-3 * abs(C4_true)


def test_nphi8_marginal_regression():
    """The production 8x8 grid is measurably wrong at moderate amplitude
    while the exact scheme is not."""
    data = make_synth(scale=25.0)
    x_grid, log_w = _dist_grid(data)
    ref = brute_marginal(data, x_grid, log_w, 128, 64)
    leg8 = np.asarray(fused_log_likelihood_distphipsimarg(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, phi_ref_grid(8), psi_grid(8), interp=INTERP))
    ex = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
    assert np.abs(leg8 - ref).max() > 1e-3       # the defect (measured 4.5e-2)
    assert np.abs(ex - ref).max() < 1e-9         # the fix


# ---------------------------------------------------------------------------
# 7. gradients
# ---------------------------------------------------------------------------

def test_exact_gradient_matches_finite_differences():
    data = make_synth(scale=25.0)
    x_grid, log_w = _dist_grid(data)

    def scalar(theta):
        return AM.fused_log_likelihood_distphipsimarg_exact(
            data, theta[0:1], theta[1:2], theta[2:3],
            x_grid, log_w, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)[0]

    theta0 = jnp.asarray([RA[0], DEC[0], INCL[0]])
    v, g = jax.jit(jax.value_and_grad(scalar))(theta0)
    g = np.asarray(g)
    assert np.all(np.isfinite(g)), "AD gradient is not finite: %r" % (g,)
    h = 1e-5
    for i in range(3):
        tp = theta0.at[i].add(h)
        tm = theta0.at[i].add(-h)
        fd = (float(scalar(tp)) - float(scalar(tm))) / (2 * h)
        assert abs(fd - g[i]) < 1e-4 * max(1.0, abs(fd)), \
            "param %d: fd %g vs AD %g" % (i, fd, g[i])


def test_laplace_gradient_matches_exact_scheme():
    """Laplace's lnL is piecewise-smooth (series cut, stationary-point
    rejection migrate bins as parameters move), so a finite difference on a
    noise-like synthetic target is contaminated by kink noise (measured: FD
    inconsistent between h=1e-4 and 1e-5 in one component while two others
    match AD to 1e-5).  The clean scheme-level check is AD-vs-AD against the
    exact scheme, which shares no marginalization code beyond the coefficient
    tables: measured agreement ~5e-4 relative at scale 100."""
    data = make_synth(scale=100.0)
    x_grid, log_w = _dist_grid(data)
    theta0 = jnp.asarray([RA[0], DEC[0], INCL[0]])
    grads = {}
    for name, fn in (("exact", AM.fused_log_likelihood_distphipsimarg_exact),
                     ("laplace",
                      AM.fused_log_likelihood_distphipsimarg_laplace)):
        def scalar(theta, fn=fn):
            return fn(data, theta[0:1], theta[1:2], theta[2:3],
                      x_grid, log_w, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)[0]
        v, g = jax.jit(jax.value_and_grad(scalar))(theta0)
        grads[name] = np.asarray(g)
        assert np.all(np.isfinite(grads[name])), \
            "%s AD gradient is not finite: %r" % (name, grads[name])
    scale_ref = np.abs(grads["exact"]).max()
    assert np.abs(grads["laplace"] - grads["exact"]).max() < 1e-2 * scale_ref


# ---------------------------------------------------------------------------
# 7b. the Laplace kernel in isolation: gradient exactness and the O(1/b) law
# ---------------------------------------------------------------------------

def _kernel(p):
    a = p[0]
    c1 = p[1] + 1j * p[2]
    c2 = p[3] + 1j * p[4]
    return AM._laplace_psi_lnI(a, c1, c2)


def test_laplace_kernel_gradient_finite_differences():
    """On smooth inputs (away from the branch boundaries) the kernel gradient
    is FD-exact -- both the Laplace branch and the small-amplitude series."""
    for p0 in ([0.3, 400.0, -250.0, 30.0, 15.0],  # Laplace branch, t ~ 540
               [0.1, 0.12, 0.08, 0.03, -0.02],    # tiny amplitude (quad)
               [0.3, 40.0, -25.0, 3.0, 1.5],      # moderate (quad), b ~ 47
               [0.2, 150.0, 80.0, 30.0, -20.0]):  # blend band, t ~ 242
        p0 = jnp.asarray(p0)
        g = np.asarray(jax.grad(_kernel)(p0))
        assert np.all(np.isfinite(g))
        h = 1e-6
        for i in range(5):
            fd = (float(_kernel(p0.at[i].add(h)))
                  - float(_kernel(p0.at[i].add(-h)))) / (2 * h)
            assert abs(fd - g[i]) < 1e-6 * max(1.0, abs(fd))


def test_laplace_kernel_error_law():
    """Kernel error vs a dense trapezoid truth follows ~0.1/b nats and
    SHRINKS with amplitude -- including the two-maxima regime (d ~ b/2).
    Measured: 7.7e-4 at b=200, 3.3e-5 at b=2000, 4.6e-6 at b=20000."""
    cases = ((200.0, 0.7, 12.0, -0.4, 1.0),       # blend band now (t = 224)
             (2000.0, -1.2, 80.0, 0.9, 0.0),      # pure Laplace
             (20000.0, 0.3, 700.0, -2.0, 0.0),    # pure Laplace, decade up
             (50.0, 1.0, 30.0, 2.0, 0.0))         # two-maxima; quadrature now
    errs = {}
    for b, beta, d, delta, a in cases:
        c1 = b * np.exp(-1j * beta)
        c2 = d * np.exp(-1j * delta)
        val = float(AM._laplace_psi_lnI(jnp.asarray(a), jnp.asarray(c1),
                                        jnp.asarray(c2)))
        u = np.linspace(0, 2 * np.pi, 2_000_001)
        f = a + b * np.cos(u - beta) + d * np.cos(2 * u - delta)
        fm = f.max()
        truth = fm + np.log(np.trapezoid(np.exp(f - fm), u) / (2 * np.pi))
        errs[b] = abs(val - truth)
        assert errs[b] < 0.5 / b, "b=%g: err %g exceeds the O(1/b) law" % (
            b, errs[b])
    # the decay comparison must pair two PURE-Laplace points: the smaller
    # cases now route through the (near-exact) quadrature/blend branches, so
    # their errors sit far BELOW the Laplace law rather than on it
    assert errs[20000.0] < errs[2000.0]


def test_dense_size_rule_pinned():
    """The dense-reconstruction sizing rule is a CALIBRATED constant, not a
    knob (see the derivation note in anglemarg.py): pin its values at the
    crossover amplitude and its floors, and that it can only grow with the
    amplitude it must cover."""
    assert AM._dense_grid_sizes(AM.ANGLE_MARG_CROSSOVER_AMPLITUDE) == (352, 176)
    assert AM._dense_grid_sizes(1.0) == (128, 64)         # floors bind
    n_lo = AM._dense_grid_sizes(AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)
    n_hi = AM._dense_grid_sizes(4 * AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)
    assert n_hi[0] >= 2 * n_lo[0] - 16 and n_hi[1] >= 2 * n_lo[1] - 16
    # review 3 (P1): the phi axis must additionally scale with the mode
    # content -- amplitude alone under-resolves higher modes (a pure order-8
    # term at A = 450 was phase-dependently ~0.037 nats wrong, order 16 up
    # to 1.2 nats, under the amplitude-only rule).  The u axis never scales:
    # psi enters at spin-2 for every mode.
    m4 = AM._dense_grid_sizes(AM.ANGLE_MARG_CROSSOVER_AMPLITUDE, m_max=4)
    assert m4[0] >= 2 * n_lo[0] - 16      # phi doubles at m_max = 4
    assert m4[1] == n_lo[1]               # u unchanged
    m8 = AM._dense_grid_sizes(AM.ANGLE_MARG_CROSSOVER_AMPLITUDE, m_max=8)
    assert m8 == (1360, 176)     # 16 * 4 * sqrt(450) -> 1360 after rounding;
    # (4 * the ROUNDED m_max=2 value would be 1408 -- the scaling applies to
    # the unrounded rule, so pin the exact derived number instead)


# ---------------------------------------------------------------------------
# 8. the selector
# ---------------------------------------------------------------------------

def test_choose_angle_marg_scheme():
    cross = AM.ANGLE_MARG_CROSSOVER_AMPLITUDE
    s, info = AM.choose_angle_marg_scheme(cross * 0.9, gh_enabled=False)
    assert s == "exact"
    s, info = AM.choose_angle_marg_scheme(cross * 1.1, gh_enabled=False)
    assert s == "laplace"
    assert info["crossover"] == cross
    # the selector keys on the MEASURED amplitude bound, never on an SNR
    # guess: its signature must not accept one (external-review defect 2)
    import inspect
    assert "guess_snr" not in inspect.signature(
        AM.choose_angle_marg_scheme).parameters
    # no amplitude available: exact (the conservative branch)
    s, info = AM.choose_angle_marg_scheme(None)
    assert s == "exact" and "no amplitude" in info["reason"]
    # adaptive distance quadrature forces the exact branch
    s, info = AM.choose_angle_marg_scheme(cross * 100, gh_enabled=True)
    assert s == "exact" and "DISTMARG_GH" in info["reason"]


def test_laplace_refuses_gh_env(monkeypatch):
    """JAX_ILE_DISTMARG_GH + laplace must raise, not silently ignore the env
    var (documented silently-inert-flag history)."""
    from RIFT.likelihood.jax_ile import core as core_mod
    monkeypatch.setattr(core_mod, "_DISTMARG_GH_N", 8)
    data = make_synth(scale=2.0)
    x_grid, log_w = _dist_grid(data)
    with pytest.raises(ValueError, match="DISTMARG_GH"):
        AM.fused_log_likelihood_distphipsimarg_laplace(
            data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
            x_grid, log_w, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)


def test_exact_supports_gh_env(monkeypatch):
    """The exact scheme honors JAX_ILE_DISTMARG_GH *identically to the grid
    path*: with GH active on both, exact must match a converged legacy grid
    to the angle-quadrature floor.  (GH-vs-uniform is a property of the
    distance treatment itself, deliberately NOT re-litigated here: the
    comparison holds the distance treatment fixed on both sides.)"""
    from RIFT.likelihood.jax_ile import core as core_mod
    data = make_synth(scale=6.0)
    x_grid, log_w = _dist_grid(data, n=128)
    monkeypatch.setattr(core_mod, "_DISTMARG_GH_N", 33)
    gh_exact = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
    gh_legacy = np.asarray(fused_log_likelihood_distphipsimarg(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, phi_ref_grid(64), psi_grid(32), interp=INTERP))
    assert np.abs(gh_exact - gh_legacy).max() < 1e-6


# ---------------------------------------------------------------------------
# 9. the wrapper: selection, provenance, and NO default change
# ---------------------------------------------------------------------------

def test_wrapper_default_is_grid_and_matches_legacy():
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                       n_grid=64, interp=INTERP)
    assert like.angle_marg_scheme == "grid"
    x_grid, log_w = like.x_grid, like.log_w_grid
    direct = np.asarray(fused_log_likelihood_distphipsimarg(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, phi_ref_grid(32), psi_grid(8), interp=INTERP))
    got = np.asarray(like.log_likelihood(RA, DEC, INCL))
    # jit-vs-eager can differ in the last ulp; anything beyond that would
    # mean the default path changed
    assert np.abs(got - direct).max() < 1e-12


def test_wrapper_auto_selects_and_records():
    """Auto selection keys on the DATA (measured amplitude bound), not on
    guess_snr: a quiet target selects exact regardless of a huge claimed
    SNR, a loud target selects laplace regardless of a missing one."""
    lo = JAXDistPhiPsiMargLikelihood(make_synth(scale=2.0), 30.0, 3000.0,
                                     n_grid=64, interp=INTERP,
                                     guess_snr=1000.0, angle_marg="auto")
    assert lo.angle_marg_scheme == "exact"
    hi = JAXDistPhiPsiMargLikelihood(make_synth(scale=2.0, kappa_boost=200.0),
                                     30.0, 3000.0, n_grid=64, interp=INTERP,
                                     guess_snr=None, angle_marg="auto")
    assert hi.angle_marg_scheme == "laplace"
    assert hi.angle_marg_info["amplitude"] > AM.ANGLE_MARG_CROSSOVER_AMPLITUDE
    for like in (lo, hi):
        info = like.angle_marg_info
        assert info["requested"] == "auto"
        assert info["scheme"] == like.angle_marg_scheme
        assert "reason" in info and "crossover" in info
        assert info["amp_sizing"] >= AM.ANGLE_MARG_CROSSOVER_AMPLITUDE
        assert info["sample_grid"] == (16, 8)
    with pytest.raises(ValueError):
        JAXDistPhiPsiMargLikelihood(make_synth(scale=2.0), 30.0, 3000.0,
                                    n_grid=64, angle_marg="bogus")


def test_wrapper_exact_scheme_end_to_end():
    """The wrapper's exact path produces the brute-force marginal through the
    same public interface production uses (value/value_and_grad/batched)."""
    data = make_synth(scale=6.0)
    like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, n_grid=64,
                                       interp=INTERP, guess_snr=3.0,
                                       angle_marg="auto")
    assert like.angle_marg_scheme == "exact"
    ref = brute_marginal(data, like.x_grid, like.log_w_grid, 96, 48)
    got = np.asarray(like.log_likelihood(RA, DEC, INCL))
    assert np.abs(got - ref).max() < 1e-10
    v, g = like.value_and_grad(np.array([RA[0], DEC[0], INCL[0]]))
    assert abs(v - ref[0]) < 1e-10
    assert np.all(np.isfinite(g))


# ---------------------------------------------------------------------------
# 10. the driver wiring (AST over the source: the defect this guards lives at
#     the call site, where a helper-level assertion cannot see it)
# ---------------------------------------------------------------------------

def _driver_source():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "bin",
                        "integrate_likelihood_extrinsic_jax")
    with open(path) as f:
        return f.read()


def test_driver_flag_exists_with_grid_default():
    src = _driver_source()
    tree = ast.parse(src)
    found = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_option"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--angle-marg-scheme"):
            kw = {k.arg: k.value for k in node.keywords}
            found = kw
    assert found is not None, "--angle-marg-scheme not registered"
    assert isinstance(found.get("default"), ast.Constant)
    assert found["default"].value == "grid", \
        "the DEFAULT scheme must stay 'grid'; changing it is a separate decision"


def test_driver_passes_scheme_to_wrapper_and_reports_it():
    """External review, item 3: an earlier version of this guard only checked
    that SOME angle_marg= keyword is passed, so the inert-flag mutant
    ``angle_marg="grid"`` (flag parsed, help present, print present, value
    ignored) passed the whole suite -- exactly this repo's documented
    silent-no-op pattern.  The guard now pins the keyword's VALUE node: it
    must be the local variable ``angle_marg`` (which test_driver_flag_exists
    ties to the option), not a constant."""
    src = _driver_source()
    tree = ast.parse(src)
    passed = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "JAXDistPhiPsiMargLikelihood"):
            for k in node.keywords:
                if k.arg == "angle_marg":
                    assert isinstance(k.value, ast.Name) \
                        and k.value.id == "angle_marg", \
                        "angle_marg= is passed a %r, not the angle_marg " \
                        "variable: the flag would be silently inert" \
                        % (ast.dump(k.value),)
                    passed = True
    assert passed, "driver builds JAXDistPhiPsiMargLikelihood without angle_marg="
    # and the variable itself must be read from the option, not re-hardcoded
    assert 'angle_marg = getattr(opts, "angle_marg_scheme", "grid")' in src
    assert "angle-marg scheme:" in src, \
        "driver must print the RESOLVED scheme (silently-inert-flag history)"
    # the print uses the wrapper's resolved attribute, not the raw option
    assert "angle_marg_scheme" in src and "angle_marg_info" in src


# ---------------------------------------------------------------------------
# 11. external-review defect 1: ALL maxima must be enumerated
# ---------------------------------------------------------------------------

def _kernel_truth(a, c1, c2, n=400001):
    u = np.linspace(0, 2 * np.pi, n)
    f = a + (c1 * np.exp(1j * u)).real + (c2 * np.exp(2j * u)).real
    fm = f.max()
    return fm + np.log(np.trapezoid(np.exp(f - fm), u) / (2 * np.pi))


def test_laplace_kernel_first_harmonic_cancellation():
    """The review's counterexample: c1 = 0, c2 = -d (delta = pi), d > 0.5.
    f = -d cos(2u): the extrema of the FIRST harmonic are u = 0, pi -- both
    MINIMA of f -- so the historical two-seed Newton rejected everything and
    returned -inf for a finite integral.  The maxima are at u = pi/2, 3pi/2
    and must both be found (finding only one loses ln 2)."""
    for dd in (0.7, 5.0, 50.0, 500.0):
        val = float(AM._laplace_psi_lnI(jnp.asarray(0.0),
                                        jnp.asarray(0.0 + 0.0j),
                                        jnp.asarray(-dd + 0.0j)))
        assert np.isfinite(val), "d=%g: kernel returned %r" % (dd, val)
        truth = _kernel_truth(0.0, 0.0, -dd)
        # Laplace error O(1/d) (measured: 0.16 at d=0.7 down to 2.5e-4 at
        # d=500); ln 2 = 0.69 would signal a missed second maximum
        assert abs(val - truth) < 0.5 / dd + 0.05, \
            "d=%g: err %g" % (dd, val - truth)
    # gradient is finite and FD-exact at the degenerate b=0 point
    p0 = jnp.asarray([0.0, 0.0, 0.0, -5.0, 0.0])
    g = np.asarray(jax.grad(_kernel)(p0))
    assert np.all(np.isfinite(g))
    h = 1e-6
    for i in range(5):
        fd = (float(_kernel(p0.at[i].add(h)))
              - float(_kernel(p0.at[i].add(-h)))) / (2 * h)
        assert abs(fd - g[i]) < 1e-6 * max(1.0, abs(fd))


def test_laplace_kernel_randomized_sweep():
    """Randomized (b, d, beta, delta) sweep against brute-force quadrature,
    log-uniform in d and in b/d INCLUDING b << d -- the failure region a
    hand-picked example set misses (review's explicit request) -- and with
    NO low-amplitude filter: an earlier revision skipped b + 2d < 0.6,
    which is exactly where review 2 then found a 0.5-nat window with a
    sign-inverted gradient.  Below the blend band the fixed-N u-quadrature
    is machine-exact; above it the O(1/A) Laplace law applies."""
    rng = np.random.default_rng(42)
    for _ in range(60):
        dd = 10 ** rng.uniform(-0.5, 2.5)
        b = dd * 10 ** rng.uniform(-3, 1.5)
        beta = rng.uniform(0, 2 * np.pi)
        delta = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(-1, 1)
        c1 = b * np.exp(-1j * beta)
        c2 = dd * np.exp(-1j * delta)
        val = float(AM._laplace_psi_lnI(jnp.asarray(a), jnp.asarray(c1),
                                        jnp.asarray(c2)))
        truth = _kernel_truth(a, c1, c2, n=200001)
        assert np.isfinite(val)
        if b + 2 * dd < AM._LAPLACE_BLEND_LO:
            tol = 1e-10                       # pure fixed-N quadrature
        else:
            tol = 4.0 / (b + dd) + 1e-3       # Laplace O(1/A) law
        assert abs(val - truth) < tol, \
            "b=%g d=%g beta=%g delta=%g: err %g (tol %g)" % (
                b, dd, beta, delta, val - truth, tol)


def test_laplace_kernel_branch_window():
    """Reviews 2 and 3 (P2-local), pinned WITHOUT filtering the window out.

    History: a hard series/Laplace switch at b + 2d = 0.5 gave 0.27-0.53
    nats and a SIGN-INVERTED |c2| gradient just above the cut (review 2); a
    Bessel-series fix moved the handover to ~16 but relied on a GLOBAL
    subdominance argument, and review 3 exhibited a locally-dominant bin at
    t = 18.9 with +0.251 nats.  The kernel now integrates u by fixed-N
    quadrature below the handover -- machine-exact for every t <= 220
    including that counterexample -- hands over to enumerated-maxima
    Laplace across a C^1 blend [220, 300], and its worst LOCAL error
    anywhere is bounded: measured 4e-3 on random draws above the band and
    <= ~0.45 nats at the measure-tiny degenerate-curvature alignment
    (b = 4d, phases aligned; was ~5 nats before the gated quartic-width
    correction), decaying with t.
    """
    # review 3's exact counterexample: must be machine-exact now
    b, dd, beta, delta = 11.8866, 3.5163, -1.8900, -0.6497
    c1 = b * np.exp(-1j * beta)
    c2 = dd * np.exp(-1j * delta)
    val = float(AM._laplace_psi_lnI(jnp.asarray(0.0), jnp.asarray(c1),
                                    jnp.asarray(c2)))
    assert abs(val - _kernel_truth(0.0, c1, c2)) < 1e-12

    rng = np.random.default_rng(5)
    val_tol = {0.5001: 1e-12, 2.0: 1e-12, 19.0: 1e-12, 120.0: 1e-12,
               220.0: 1e-11, 260.0: 0.02, 300.0: 0.02, 500.0: 0.03}
    for t, vtol in val_tol.items():
        for _ in range(3):
            frac = rng.uniform(0.05, 0.95)
            bb = t * frac
            d2 = t * (1 - frac) / 2
            c1 = bb * np.exp(-1j * rng.uniform(0, 2 * np.pi))
            c2 = d2 * np.exp(-1j * rng.uniform(0, 2 * np.pi))
            val = float(AM._laplace_psi_lnI(jnp.asarray(0.2),
                                            jnp.asarray(c1),
                                            jnp.asarray(c2)))
            truth = _kernel_truth(0.2, c1, c2, n=200001)
            assert abs(val - truth) < vtol, \
                "t=%g: val err %g (tol %g)" % (t, val - truth, vtol)
            # |c2|-direction gradient: machine-exact below the band,
            # bounded and sign-correct wherever resolved above it
            e2 = c2 / max(abs(c2), 1e-300)
            g_ad = float(jax.grad(
                lambda dv: AM._laplace_psi_lnI(jnp.asarray(0.2),
                                               jnp.asarray(c1),
                                               dv * e2))(jnp.asarray(d2)))
            h = 1e-5
            g_tr = (_kernel_truth(0.2, c1, (d2 + h) * e2, n=200001)
                    - _kernel_truth(0.2, c1, (d2 - h) * e2, n=200001)) / (2 * h)
            gtol = 1e-7 if t <= AM._LAPLACE_BLEND_LO else 0.1 + 0.1 * abs(g_tr)
            assert abs(g_ad - g_tr) < gtol, \
                "t=%g: grad AD %+g vs truth %+g (tol %g)" % (t, g_ad, g_tr,
                                                             gtol)
            if abs(g_tr) > 0.5:
                assert np.sign(g_ad) == np.sign(g_tr), \
                    "t=%g: gradient SIGN inverted (AD %+g, truth %+g)" % (
                        t, g_ad, g_tr)
    # degenerate-curvature alignment (b = 4d, delta = pi): the quartic-flat
    # global maximum where the floored-Gaussian Laplace was ~5 nats high;
    # the gated quartic-width correction bounds it (measured <= 0.45)
    for t in (320.0, 600.0):
        for eps in (-0.05, 0.0, 0.05):
            d2 = t / 6.0
            bb = 4 * d2 * (1 + eps)
            val = float(AM._laplace_psi_lnI(jnp.asarray(0.0),
                                            jnp.asarray(bb + 0.0j),
                                            jnp.asarray(-d2 + 0.0j)))
            truth = _kernel_truth(0.0, bb + 0.0j, -d2 + 0.0j, n=400001)
            assert abs(val - truth) < 0.6, \
                "degenerate t=%g eps=%g: err %g" % (t, eps, val - truth)
    # C^1 blend: no value jumps across the band (a hard switch stepped by
    # ~0.5 nats; measured max step 3.2e-5 at this resolution)
    prev = None
    for t in np.linspace(AM._LAPLACE_BLEND_LO - 5.0,
                         AM._LAPLACE_BLEND_HI + 5.0, 30):
        c1 = 0.6 * t * np.exp(-1j * 1.1)
        c2 = 0.2 * t * np.exp(-1j * 2.3)
        v = float(AM._laplace_psi_lnI(jnp.asarray(0.0), jnp.asarray(c1),
                                      jnp.asarray(c2))) \
            - _kernel_truth(0.0, c1, c2, n=200001)
        if prev is not None:
            assert abs(v - prev) < 5e-3
        prev = v


# ---------------------------------------------------------------------------
# 12. external-review defect 2: sizing must come from the data, not guess_snr
# ---------------------------------------------------------------------------

def test_estimate_angle_amplitude_tracks_the_data():
    from RIFT.likelihood.jax_ile.core import make_distance_grid
    quiet = make_synth(scale=2.0)
    xg, _ = make_distance_grid(30.0, 3000.0, 64, distMpcRef=quiet.distMpcRef)
    a_quiet = AM.estimate_angle_amplitude(quiet, xg)
    # quiet target: bound well below the crossover (UNfloored -- the selector
    # needs the raw value, or every quiet target would select laplace; the
    # wrapper floors the SIZING separately)
    assert 0.0 <= a_quiet < AM.ANGLE_MARG_CROSSOVER_AMPLITUDE
    loud = make_synth(scale=2.0, kappa_boost=200.0)
    a_loud = AM.estimate_angle_amplitude(loud, xg)
    louder = make_synth(scale=2.0, kappa_boost=400.0)
    a_louder = AM.estimate_angle_amplitude(louder, xg)
    assert a_loud > 20 * AM.ANGLE_MARG_CROSSOVER_AMPLITUDE
    assert a_louder > 1.5 * a_loud                            # monotone
    # The estimator must reproduce an INDEPENDENTLY-computed 512x128
    # reference to < 2% (band-limited content, default grid ~6x Nyquist).
    # The reference deliberately does NOT call estimate_angle_amplitude with
    # a finer grid: a mutant gutting the estimator's internals would gut
    # such a reference identically (common-mode blindness -- an earlier
    # revision of this pin let exactly that mutant survive).  White-box
    # coupling: the sky draws replicate the estimator's documented
    # rng(seed=0) construction.
    rng = np.random.default_rng(0)
    n_sky = AM.ANGLE_AMP_SKY_POINTS
    ra_s = rng.uniform(0.0, 2 * np.pi, n_sky)
    dec_s = np.arcsin(rng.uniform(-1.0, 1.0, n_sky))
    incl_s = np.arccos(rng.uniform(-1.0, 1.0, n_sky))
    g_ra, g_dec = np.meshgrid(np.linspace(0, 2 * np.pi, 6, endpoint=False),
                              np.array([-1.0, -0.35, 0.35, 1.0]),
                              indexing="ij")
    for i0_ in (0.0, np.pi):
        ra_s = np.concatenate([ra_s, g_ra.ravel()])
        dec_s = np.concatenate([dec_s, g_dec.ravel()])
        incl_s = np.concatenate([incl_s, np.full(g_ra.size, i0_)])
    C_A, C_B, _meta = AM.angle_coefficient_tables(loud, ra_s, dec_s, incl_s)
    C_A, C_B = np.asarray(C_A), np.asarray(C_B)
    PH, UU = np.meshgrid(np.linspace(0, 2 * np.pi, 512, endpoint=False),
                         np.linspace(0, 2 * np.pi, 128, endpoint=False),
                         indexing="ij")
    phis, us = PH.ravel(), UU.ravel()

    def _mat(C):
        kp = np.arange(C.shape[0])
        ks = np.arange(-(C.shape[1] - 1) // 2, (C.shape[1] - 1) // 2 + 1)
        E = np.exp(1j * (phis[:, None, None] * kp[None, :, None]
                         + us[:, None, None] * ks[None, None, :]))
        w = np.ones(C.shape[0])
        w[1:] = 2.0
        return (E * w[None, :, None]).reshape(len(phis), -1)

    E_A, E_B = _mat(C_A), _mat(C_B)
    xv = np.asarray(xg)
    x_min, x_max = float(xv.min()), float(xv.max())
    a_ref = 0.0
    for j in range(len(ra_s)):
        A_g = (E_A @ C_A[:, :, j].reshape(-1, C_A.shape[-1])).real
        B_g = np.maximum((E_B @ C_B[:, :, j].reshape(-1, C_B.shape[-1])).real,
                         0.0)
        x_hat = np.clip(A_g / np.maximum(B_g, 1e-300), x_min, x_max)
        a_ref = max(a_ref, float((x_hat * A_g
                                  - 0.5 * np.square(x_hat) * B_g).max()))
    a_ref *= AM.ANGLE_AMP_MARGIN
    assert a_loud > 0.98 * a_ref
    assert a_loud <= a_ref * (1 + 1e-9)      # grid max cannot EXCEED the max


def test_amp_sizing_is_required():
    """No default: a silently-undersized dense grid is the module's own
    defect class (review: 'a number that is too small, with nothing that
    notices')."""
    data = make_synth(scale=2.0)
    x_grid, log_w = _dist_grid(data)
    for fn in (AM.fused_log_likelihood_distphipsimarg_exact,
               AM.fused_log_likelihood_distphipsimarg_laplace):
        with pytest.raises(ValueError, match="amp_sizing"):
            fn(data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
               x_grid, log_w, interp=INTERP)


def test_wrapper_sizing_survives_missing_or_low_guess_snr():
    """THE defect-2 regression.  On a target whose true angular amplitude is
    ~1.7e4 (kappa_boost=200; the crossover-floor grid is off by -1.04 nats,
    measured), the wrapper must produce the correctly-sized answer whether
    guess_snr is None or underestimated 10x -- because sizing and selection
    key on the coefficient tables, not on the caller's estimate.  Both
    sub-cases FAILED against the pre-review implementation (which pinned
    amp_sizing = 450 whenever guess_snr was absent or small)."""
    data = make_synth(scale=2.0, kappa_boost=200.0)
    like0 = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, n_grid=64,
                                        interp=INTERP, guess_snr=None,
                                        angle_marg="exact")
    amp_data = like0.angle_marg_info["amp_sizing"]
    assert amp_data > 20 * AM.ANGLE_MARG_CROSSOVER_AMPLITUDE
    # reference: exact scheme sized ABOVE the wrapper's own bound
    ref = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        like0.x_grid, like0.log_w_grid, interp=INTERP,
        amp_sizing=2 * amp_data))
    # the bite: the old floor-sized grid is measurably wrong here
    floor = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        like0.x_grid, like0.log_w_grid, interp=INTERP,
        amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
    assert np.abs(floor - ref).max() > 0.1, \
        "the regression no longer bites; the target needs a larger boost"
    for guess in (None, np.sqrt(2 * amp_data) / 10.0):   # missing, 10x low
        like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, n_grid=64,
                                           interp=INTERP, guess_snr=guess,
                                           angle_marg="exact")
        got = np.asarray(like.log_likelihood(RA, DEC, INCL))
        assert np.abs(got - ref).max() < 1e-6, \
            "guess_snr=%r: wrapper answer off by %g" % (
                guess, np.abs(got - ref).max())


# ---------------------------------------------------------------------------
# 13. review 3: higher modes in the FINAL marginal, and the runtime fail-safe
# ---------------------------------------------------------------------------

MODES_M4 = ((2, 2), (2, -2), (3, 3), (3, -3), (4, 4), (4, -4))


def test_higher_mode_marginal_vs_bruteforce():
    """Review 3, P1: 'test the FINAL MARGINAL for higher-mode inputs, not
    just the estimator's reconstruction' -- the earlier coverage stopped at
    the reconstruction step, which is exactly why the amplitude-only dense
    sizing survived.  m_max = 4 end to end at cheap scale: sample grid
    (24, 8), coefficient tables with KPA = 5 / KPB = 9, and the full
    marginal against an independent brute force.  Measured 1.8e-15."""
    data = make_synth(scale=4.0, modes=MODES_M4)
    assert AM._data_m_max(data) == 4
    assert AM.angle_sample_grid_sizes(4) == (24, 8)
    x_grid, log_w = _dist_grid(data)
    ref = brute_marginal(data, x_grid, log_w, 96, 48)
    ex = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, interp=INTERP,
        amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE))
    assert np.abs(ex - ref).max() < 1e-10
    # and the harmonic-content invariant extends to n = 8, machine-zero above
    n = 64
    phis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    f = _lnL_t_fixed_time(data, phis, np.full(n, 0.6), 0.8, 10)
    C = np.abs(np.fft.rfft(f) / n)
    assert C[:9].max() > 0
    assert C[9:].max() < 1e-12 * C.max()


def test_higher_mode_dense_sizing_self_convergence():
    """Review 3, P1 at the sizing level: on an m_max = 4 target loud enough
    that the dense grid actually works (amp ~ few hundred), the marginal
    must be converged in the dense sizes -- quadrupling amp_sizing (which
    doubles every dense axis) must not move it.  Under the amplitude-only
    rule this bites at the ~0.03-nat level (review 3 measured +-0.037 for a
    pure order-8 term at A = 450); with the m_max-scaled rule it is
    ~machine (measured 0.0)."""
    data = make_synth(scale=2.0, modes=MODES_M4, kappa_boost=14.0)
    x_grid, log_w = _dist_grid(data)
    amp = AM.estimate_angle_amplitude(data, x_grid)
    assert amp > 50.0          # genuinely loud, or the test proves nothing
    args = (data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
            x_grid, log_w)
    a_sz = max(amp, AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)
    v1 = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        *args, interp=INTERP, amp_sizing=a_sz))
    v4 = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        *args, interp=INTERP, amp_sizing=4 * a_sz))
    assert np.abs(v1 - v4).max() < 1e-6


def test_runtime_amp_failsafe_warns(capfd):
    """Review 3, P2: the amplitude is an ESTIMATOR, so the fused functions
    carry a runtime fail-safe -- a call whose own coefficient tables exceed
    2x the amp_sizing the grids were built for prints a loud warning from
    inside jit (never silent), and a correctly-sized call prints nothing."""
    data = make_synth(scale=2.0, kappa_boost=200.0)
    x_grid, log_w = _dist_grid(data)
    args = (data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
            x_grid, log_w)
    # deliberately undersized: the warning must fire
    v = AM.fused_log_likelihood_distphipsimarg_exact(
        *args, interp=INTERP, amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)
    jax.block_until_ready(v)
    out = capfd.readouterr()
    assert "WARNING anglemarg/exact" in out.out + out.err
    # correctly sized: silence
    amp = AM.estimate_angle_amplitude(data, x_grid)
    v = AM.fused_log_likelihood_distphipsimarg_exact(
        *args, interp=INTERP, amp_sizing=amp)
    jax.block_until_ready(v)
    out = capfd.readouterr()
    assert "WARNING anglemarg" not in out.out + out.err
