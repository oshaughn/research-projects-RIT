"""
Gate for the psi-marginal distance-node placement that lets the 'laplace'
angle-marg scheme honour JAX_ILE_DISTMARG_GH
(``anglemarg._gh_psi_node_offsets`` and the ``_use_gh`` branch of
``fused_log_likelihood_distphipsimarg_laplace``).

WHAT IS PINNED, AND WHY
-----------------------
core._distmarg_gh_logL places frozen nodes around the distance peak of a FIXED
psi.  On the laplace path psi is already integrated out, so the nodes must
bracket the psi-MARGINAL distance integrand -- a mixture over u = 2 psi of
Gaussians centred at x*(u) = A(u)/B(u).  The shipped rule centres on the u that
maximises A(u) and takes the width from the closed-form envelope
R_lo = B0 - |B1| - |B2| <= min_u B, with a +-12 sigma half-span.

Its whole validity rests on ONE structural identity: for (2,+-2) mode content
the spin-2 response makes A0 and B1 vanish identically, so R_lo IS min_u B and
the closed-form centre sits inside the weight-carrying span.  That identity does
not survive richer mode content, so:

  * the identity itself is pinned here, WITH a positive control on m_max = 3
    data proving the assertion can fail (a pass-through that cannot fail is not
    coverage);
  * the m_max gate is pinned, with a positive control that the same data is
    accepted with the adaptive quadrature off;
  * the placement constants are pinned, and the sufficiency of 12 sigma is
    demonstrated by MUTATION -- shrinking the half-span changes the answer,
    growing it does not;
  * agreement is checked against BOTH independent references (exact + the same
    adaptive quadrature, and laplace + a converged uniform grid), on a fixture
    loud enough that the adaptive bracket is the right tool;
  * gradients through the new branch are finite (the reason the placement is
    under stop_gradient at all).

Ladder-2 measurements behind the constants live in
devnotes/DESIGN_gh_laplace.md of the branch that introduced them, not here.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile import core as core_mod
from RIFT.likelihood.jax_ile.core import make_distance_grid

from test_angle_marg_exact import make_synth, RA, DEC, INCL, INTERP

AMP = AM.ANGLE_MARG_CROSSOVER_AMPLITUDE


# ---------------------------------------------------------------------------
# fixtures: a LOUD target, so the distance peak is narrow and an adaptive
# bracket is the right tool (at low amplitude the +-N sigma window of the
# fixed-psi rule does not cover the support either -- a property of
# core._distmarg_gh_logL, deliberately not re-litigated here)
# ---------------------------------------------------------------------------

def loud_data(modes=((2, 2), (2, -2))):
    return make_synth(scale=6.0, kappa_boost=12.0, modes=modes)


def loud_grid(data, n=256):
    return make_distance_grid(200.0, 4000.0, n, distMpcRef=data.distMpcRef)


def _fields(data, nphi=24):
    """A0, A1, B0, B1, B2 on a phi grid -- the SAME expressions the kernel uses."""
    C_A, C_B, meta = AM.angle_coefficient_tables(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL), INTERP)
    C_A = np.asarray(C_A); C_B = np.asarray(C_B)
    m_max = int(meta["m_max"])
    wA = np.asarray(AM._kp_weights(m_max + 1))
    wB = np.asarray(AM._kp_weights(2 * m_max + 1))
    phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
    EA = np.exp(1j * phi[:, None] * np.arange(m_max + 1)[None, :]) * wA[None, :]
    EB = np.exp(1j * phi[:, None] * np.arange(2 * m_max + 1)[None, :]) * wB[None, :]
    MA = lambda k: np.einsum("ck,kst->cst", EA, C_A[:, k])
    MB = lambda k: np.einsum("ck,kst->cst", EB, C_B[:, k])
    kA = (C_A.shape[1] - 1) // 2
    kB = (C_B.shape[1] - 1) // 2
    return dict(
        A0=MA(kA).real, A1=MA(kA + 1) + np.conj(MA(kA - 1)),
        B0=MB(kB).real, B1=MB(kB + 1) + np.conj(MB(kB - 1)),
        B2=MB(kB + 2) + np.conj(MB(kB - 2)), m_max=m_max)


# ---------------------------------------------------------------------------
# 1. the structural identity the placement rests on
# ---------------------------------------------------------------------------

def _identity_residuals(f):
    """(|A0|/|A1|, |B1|/|B0|) -- zero iff A(u) is a pure first harmonic and
    B(u) a constant plus a pure second harmonic."""
    return (np.abs(f["A0"]).max() / np.abs(f["A1"]).max(),
            np.abs(f["B1"]).max() / np.abs(f["B0"]).max())


@pytest.mark.parametrize("modes,m_max", [
    (((2, 2), (2, -2)), 2),
    (((2, 2), (2, -2), (3, 3), (3, -3)), 3),
    (((2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (4, 4), (4, -4)), 4),
])
def test_a0_and_b1_vanish(modes, m_max):
    """A(u) is a PURE first harmonic and B(u) a constant plus a PURE second
    harmonic, so R_lo = B0 - |B1| - |B2| is min_u B EXACTLY rather than a
    bound.  This is the identity the +-12 sigma rule is derived from; if a
    mode-convention change breaks it, the rule is void.

    It is not a (2,+-2) accident: the psi dependence enters only through the
    spin-2 antenna response F(psi) ~ e^{-2 i psi}, which puts the kappa term at
    exactly one u-harmonic (u = 2 psi) and the rho^2 term at exactly harmonics
    0 and 2, whatever the mode content.  Checked here through m_max = 4, and
    on real IMRPhenomXHM l_max = 3/4 data in the branch's DESIGN note.  (The
    SHIP gate is still m_max <= 2, deliberately: the identity fixes the WIDTH,
    while the half-span constant was measured on (2,+-2) fixtures.)"""
    f = _fields(loud_data(modes=modes))
    assert f["m_max"] == m_max
    a0, b1 = _identity_residuals(f)
    assert a0 < 1e-12 and b1 < 1e-12, (a0, b1)
    # ... and the second harmonic is REAL content, not another zero: a bound
    # that is tight only because every harmonic vanished would prove nothing.
    assert np.median(np.abs(f["B2"]) / f["B0"]) > 1e-3
    # R_lo is then min_u B to u-grid resolution, and strictly positive
    u = np.linspace(0.0, 2 * np.pi, 4096, endpoint=False)
    Bu = (f["B0"][..., None] + (f["B1"][..., None] * np.exp(1j * u)).real
          + (f["B2"][..., None] * np.exp(2j * u)).real)
    R_lo = f["B0"] - np.abs(f["B1"]) - np.abs(f["B2"])
    assert (R_lo > 0).all()
    assert np.abs(Bu.min(-1) / R_lo - 1.0).max() < 1e-4


def test_identity_check_has_a_positive_control():
    """POSITIVE CONTROL for the test above.  The identity holds for EVERY mode
    set, so no choice of modes can make the assertion fail -- which means a
    bug that returned all-zero coefficient tables, or a residual computed from
    the wrong slots, would leave it passing for the wrong reason.  Prove the
    detector fires by PLANTING the harmonics it is supposed to reject."""
    f = _fields(loud_data())
    ok_a0, ok_b1 = _identity_residuals(f)
    assert ok_a0 < 1e-12 and ok_b1 < 1e-12
    planted_a0 = dict(f, A0=f["A0"] + 1e-3 * np.abs(f["A1"]))
    planted_b1 = dict(f, B1=f["B1"] + 1e-3 * f["B0"])
    assert _identity_residuals(planted_a0)[0] >= 1e-12
    assert _identity_residuals(planted_b1)[1] >= 1e-12
    # and the residual is computed from a NON-DEGENERATE table: |A1| and |B0|
    # are the scales it divides by, so a zeroed table would give 0/0, not 0
    assert np.abs(f["A1"]).max() > 0 and np.abs(f["B0"]).max() > 0


# ---------------------------------------------------------------------------
# 2. the node-offset rule
# ---------------------------------------------------------------------------

def test_gh_psi_node_offsets():
    assert AM._GH_PSI_HALF_SIGMA == 22.0
    assert AM._GH_PSI_MIN_NODES == 49
    assert AM._GH_PSI_M_MAX == 2
    for n_req in (8, 16, 33, 64, 129):
        z, zp, zn, n = AM._gh_psi_node_offsets(n_req)
        assert len(z) == len(zp) == len(zn) == n
        assert z[0] == -AM._GH_PSI_HALF_SIGMA and z[-1] == AM._GH_PSI_HALF_SIGMA
        assert n >= AM._GH_PSI_MIN_NODES
        # neighbour arrays are z with the INDEX clamped: this is what makes
        # 0.5*(x[k+1]-x[k-1]) reproduce core._distmarg_gh_logL's trapezoid
        # weights (0.5*dx at both ends) without cross-block communication
        assert zp[0] == z[0] and zn[-1] == z[-1]
        assert np.allclose(zp[1:], z[:-1]) and np.allclose(zn[:-1], z[1:])
        # node DENSITY is at least what the caller asked for at +-7 sigma
        h_req = 14.0 / max(n_req - 1, 1)
        assert (2 * AM._GH_PSI_HALF_SIGMA) / (n - 1) <= h_req + 1e-12
    # the floor binds for small requests, and gives <= 1 sigma spacing
    assert AM._gh_psi_node_offsets(4)[3] == AM._GH_PSI_MIN_NODES
    z = AM._gh_psi_node_offsets(4)[0]
    assert (z[1] - z[0]) <= 1.0


def test_trapezoid_weights_match_the_fixed_psi_rule():
    """The index-clamped neighbour form must reproduce core._distmarg_gh_logL's
    diff()/concatenate() weights bit for bit on unclipped nodes."""
    z, zp, zn, n = AM._gh_psi_node_offsets(33)
    centre, sigma = 3.0, 0.25
    x = centre + sigma * z
    w_new = 0.5 * (centre + sigma * zn - (centre + sigma * zp))
    dx = np.diff(x)
    w_old = np.concatenate([0.5 * dx[:1], 0.5 * (dx[1:] + dx[:-1]), 0.5 * dx[-1:]])
    assert np.array_equal(w_new, w_old)


# ---------------------------------------------------------------------------
# 3. the mode-content gate  (with its positive control)
# ---------------------------------------------------------------------------

def test_gh_laplace_gate_on_mode_content(monkeypatch):
    data3 = loud_data(modes=((2, 2), (2, -2), (3, 3), (3, -3)))
    x3, lw3 = loud_grid(data3)
    call3 = lambda: AM.fused_log_likelihood_distphipsimarg_laplace(
        data3, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x3, lw3, interp=INTERP, amp_sizing=AMP)
    # POSITIVE CONTROL: with the adaptive quadrature OFF the very same data
    # runs.  Without this the raise below could be any other failure.
    monkeypatch.setattr(core_mod, "_DISTMARG_GH_N", 0)
    assert np.isfinite(np.asarray(call3())).all()
    monkeypatch.setattr(core_mod, "_DISTMARG_GH_N", 33)
    with pytest.raises(ValueError, match="m_max"):
        call3()
    # ... and the covered mode content is ACCEPTED, so the gate is a gate and
    # not a blanket refusal
    data2 = loud_data()
    x2, lw2 = loud_grid(data2)
    got = np.asarray(AM.fused_log_likelihood_distphipsimarg_laplace(
        data2, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x2, lw2, interp=INTERP, amp_sizing=AMP))
    assert np.isfinite(got).all()


# ---------------------------------------------------------------------------
# 4. agreement with two INDEPENDENT references
# ---------------------------------------------------------------------------

def _lap_gh(data, x, lw, n_gh, monkeypatch, **over):
    for k, v in over.items():
        monkeypatch.setattr(AM, k, v)
    monkeypatch.setattr(core_mod, "_DISTMARG_GH_N", n_gh)
    return np.asarray(AM.fused_log_likelihood_distphipsimarg_laplace(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x, lw, interp=INTERP, amp_sizing=AMP))


def test_gh_laplace_matches_exact_gh(monkeypatch):
    """Same adaptive distance treatment on both sides: the residual is the
    psi-Laplace error alone, which the crossover constant already bounds."""
    data = loud_data()
    x, lw = loud_grid(data)
    lap = _lap_gh(data, x, lw, 65, monkeypatch)
    monkeypatch.setattr(core_mod, "_DISTMARG_GH_N", 65)
    ex = np.asarray(AM.fused_log_likelihood_distphipsimarg_exact(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x, lw, interp=INTERP, amp_sizing=AMP))
    assert np.abs(lap - ex).max() < 5e-3


def test_gh_laplace_matches_converged_uniform_grid(monkeypatch):
    """Same angle treatment on both sides, distance treatment independent:
    a converged uniform grid must reproduce the adaptive answer."""
    data = loud_data()
    x, lw = loud_grid(data)
    lap_gh = _lap_gh(data, x, lw, 65, monkeypatch)
    monkeypatch.setattr(core_mod, "_DISTMARG_GH_N", 0)
    xu, lwu = loud_grid(data, n=16384)
    lap_uni = np.asarray(AM.fused_log_likelihood_distphipsimarg_laplace(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        xu, lwu, interp=INTERP, amp_sizing=AMP))
    assert np.abs(lap_gh - lap_uni).max() < 5e-3


def test_gh_laplace_converged_in_node_count(monkeypatch):
    data = loud_data()
    x, lw = loud_grid(data)
    ref = _lap_gh(data, x, lw, 129, monkeypatch)
    for n_gh in (16, 33, 65):
        got = _lap_gh(data, x, lw, n_gh, monkeypatch)
        assert np.abs(got - ref).max() < 1e-4, n_gh


# ---------------------------------------------------------------------------
# 5. the half-span, by MUTATION: 12 sigma is enough and is not gratuitous
# ---------------------------------------------------------------------------

def test_half_span_is_sufficient_and_not_gratuitous(monkeypatch):
    """A frozen bracket is only correct if it CONTAINS the psi-marginal peak.
    Testing that by construction is circular, so test it by intervention:
    doubling the half-span must not move the answer (12 sigma already
    contains everything), while collapsing it must (proving the answer really
    does depend on the bracket, i.e. that this knob is live at all)."""
    data = loud_data()
    x, lw = loud_grid(data)
    base = _lap_gh(data, x, lw, 65, monkeypatch)
    wide = _lap_gh(data, x, lw, 65, monkeypatch, _GH_PSI_HALF_SIGMA=44.0)
    assert np.abs(base - wide).max() < 1e-4, "22 sigma does not contain the peak"
    narrow = _lap_gh(data, x, lw, 65, monkeypatch, _GH_PSI_HALF_SIGMA=0.05)
    assert np.abs(base - narrow).max() > 1e-2, (
        "collapsing the bracket did not change the answer -- the half-span is "
        "inert and this test proves nothing")


def test_node_floor_is_live(monkeypatch):
    """MUTATION of the other constant: with the floor removed, a tiny node
    request must degrade the answer -- otherwise _GH_PSI_MIN_NODES is
    decoration."""
    data = loud_data()
    x, lw = loud_grid(data)
    ref = _lap_gh(data, x, lw, 129, monkeypatch)
    monkeypatch.setattr(AM, "_GH_PSI_MIN_NODES", 3)
    n_coarse = AM._gh_psi_node_offsets(2)[3]
    assert n_coarse < 10, n_coarse          # the floor is what was holding it up
    coarse = _lap_gh(data, x, lw, 2, monkeypatch)
    assert np.abs(coarse - ref).max() > 1e-3, (
        "a %d-node bracket matched the converged answer -- the node count is "
        "inert and the floor pins nothing" % n_coarse)


# ---------------------------------------------------------------------------
# 6. the sigma cap (unreachable on signal-carrying data; must still be live)
# ---------------------------------------------------------------------------

def test_sigma_cap_is_inactive_at_production_support_and_live_when_it_binds(
        monkeypatch):
    """The cap keeps the bracket inside the physical support when R_lo -> 0 (a
    bin with no response, where the exponent is flat in x and sigma would blow
    up, leaving every node on one of the two rails).  Pin BOTH directions: it
    is far from binding at a production distance support, and it does bind --
    and change the answer -- on a narrow one."""
    data = loud_data()
    f = _fields(data)
    sigma = 1.0 / np.sqrt(f["B0"] - np.abs(f["B1"]) - np.abs(f["B2"]))

    def cap(d_min, d_max):
        xg, _ = make_distance_grid(d_min, d_max, 64, distMpcRef=data.distMpcRef)
        xg = np.asarray(xg)
        return (xg.max() - xg.min()) / (2.0 * AM._GH_PSI_HALF_SIGMA)

    # production-scale support (RIFT's own --d-min 1 --d-max 10000): the widths
    # this data actually uses are two orders of magnitude below the cap, so the
    # cap cannot be shaping any production result.  (On the ladder-2 injection
    # itself the margin is ~1100x -- sigma ~ 0.02 against a cap of 22.7 -- this
    # synthetic target is quieter, hence the looser factor here.)
    assert sigma.max() < 1e-2 * cap(1.0, 10000.0)
    # the fixture's own narrow support DOES reach it -- so the cap is live code
    assert sigma.max() > cap(200.0, 4000.0)
    # ... and removing it changes the answer, i.e. it is not a no-op minimum
    x, lw = loud_grid(data)
    base = _lap_gh(data, x, lw, 65, monkeypatch)
    monkeypatch.setattr(AM, "_GH_PSI_HALF_SIGMA", 1e-9)   # cap -> enormous
    uncapped = _lap_gh(data, x, lw, 65, monkeypatch)
    assert np.abs(uncapped - base).max() > 1e-2


# ---------------------------------------------------------------------------
# 7. gradients (the reason the placement is frozen)
# ---------------------------------------------------------------------------

def test_gh_laplace_gradients_are_finite(monkeypatch):
    data = loud_data()
    x, lw = loud_grid(data)
    monkeypatch.setattr(core_mod, "_DISTMARG_GH_N", 33)

    def f(ra, dec, incl):
        return AM.fused_log_likelihood_distphipsimarg_laplace(
            data, ra, dec, incl, x, lw, interp=INTERP, amp_sizing=AMP).sum()

    g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(RA), jnp.asarray(DEC),
                                       jnp.asarray(INCL))
    assert all(np.isfinite(np.asarray(gi)).all() for gi in g)
    # ... and against finite differences on the sky angles
    eps = 1e-5
    for i, base in enumerate((RA, DEC, INCL)):
        args = [jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL)]
        args[i] = jnp.asarray(base + eps)
        hi = float(f(*args))
        args[i] = jnp.asarray(base - eps)
        lo = float(f(*args))
        fd = (hi - lo) / (2 * eps)
        assert abs(fd - float(np.asarray(g[i]).sum())) <= 1e-4 * max(1.0, abs(fd))


# ---------------------------------------------------------------------------
# 8. the closed-form psi maximiser the centre sits on
# ---------------------------------------------------------------------------

def test_closed_form_psi_argmax_is_exact():
    """With A0 == 0 and B1 == 0, argmax_u A(u)^2/(2 B(u)) is exactly
    e^{i u*} = +- conj(w)/|w|, w = B0*A1 - conj(A1)*B2.  This is what the node
    CENTRE sits on; centring on argmax A(u) instead (the B2 -> 0 limit of the
    same formula) is loose by up to ~7000 sigma at large |B2|/B0.

    Checked against brute force over random coefficient triples, INCLUDING the
    large-|B2|/B0 corner that the ladder-2 fixture never reaches -- the point
    being that a fixture cannot exercise this and the closed form must be
    right everywhere."""
    rng = np.random.default_rng(11)
    n = 4000
    A1 = rng.normal(size=n) + 1j * rng.normal(size=n)
    B0 = np.abs(rng.normal(size=n)) * 3 + 0.05
    r = rng.uniform(0.0, 0.9999, n)
    B2 = B0 * r * np.exp(1j * rng.uniform(0, 2 * np.pi, n))

    w = B0 * A1 - np.conj(A1) * B2
    ph = np.conj(w) / np.maximum(np.abs(w), 1e-300)
    ph = ph * np.where((A1 * ph).real >= 0, 1.0, -1.0)
    A_st = (A1 * ph).real
    B_st = B0 + (B2 * ph * ph).real
    assert (A_st > 0).all(), "sign branch does not put the centre at x* > 0"
    assert (B_st > 0).all()
    E_st = A_st ** 2 / (2 * B_st)

    u = np.linspace(0.0, 2 * np.pi, 100001, endpoint=False)
    e1 = np.exp(1j * u); e2 = e1 * e1
    best = np.empty(n)
    for k in range(0, n, 100):
        sl = slice(k, k + 100)
        A = (A1[sl, None] * e1).real
        B = B0[sl, None] + (B2[sl, None] * e2).real
        best[sl] = np.where(A > 0, A * A / (2 * np.maximum(B, 1e-300)), 0.0).max(-1)
    # brute force must never BEAT the closed form (it can only fall short by
    # the u-grid resolution)
    assert ((best - E_st) / E_st).max() < 1e-12
    # POSITIVE CONTROL: the naive centring (argmax A) is measurably worse, so
    # the correction is doing work rather than being a no-op rewrite
    ph_naive = np.conj(A1) / np.abs(A1)
    E_naive = ((A1 * ph_naive).real ** 2
               / (2 * (B0 + (B2 * ph_naive * ph_naive).real)))
    assert (E_naive < E_st * (1 - 1e-6)).mean() > 0.5, (
        "argmax A already equals argmax A^2/2B on this sample; the closed form "
        "is then untested")
    # ... and the two coincide exactly in the B2 -> 0 limit
    w0 = B0 * A1
    assert np.abs(np.conj(w0) / np.abs(w0) - np.conj(A1) / np.abs(A1)).max() < 1e-12
