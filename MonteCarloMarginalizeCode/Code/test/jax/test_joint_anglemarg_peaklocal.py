"""Tests for the JAX joint (phi,psi) peak-local kernel."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from RIFT.likelihood import joint_angle_peak_local as JN
from RIFT.likelihood.jax_ile import joint_anglemarg_peaklocal as JP


def _tables(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    A = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * scale
    B = (rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))) * scale
    B[0, 2] = abs(B[0, 2].real) + 3.0 * scale
    return A, B


def test_exported_names_exist():
    assert not [n for n in JP.__all__ if not hasattr(JP, n)]


@pytest.mark.parametrize("scale", [0.5, 3.0, 12.0])
def test_inner_u_integral_is_exact(scale):
    """The cell partition is a PARTITION, so this is exact, not truncated."""
    rng = np.random.default_rng(1)
    f = jax.jit(JP.log_inner_u_integral)
    u = np.linspace(0.0, 2 * np.pi, 400000, endpoint=False)
    for _ in range(4):
        c1 = scale * (rng.normal() + 1j * rng.normal())
        c2 = scale * (rng.normal() + 1j * rng.normal())
        g = (c1 * np.exp(1j * u)).real + (c2 * np.exp(2j * u)).real
        m = g.max()
        ref = m + np.log(np.exp(g - m).mean()) + np.log(2 * np.pi)
        assert abs(float(f(0.0, complex(c1), complex(c2))) - ref) < 1e-4


def test_both_signs_of_q_enter_the_u_coefficients():
    """Both +q and -q columns are stored, so c_q = D_{+q} + conj(D_{-q}).  Using only
    the +q column is survivable where roots are seeds, but here they define the
    integration partition -- it was worth 17 nats at a single phi."""
    A, B = _tables(seed=3, scale=4.0)
    C = JN.joint_table(A, B, x=0.9)
    u = np.linspace(0.0, 2 * np.pi, 200000, endpoint=False)
    f = jax.jit(JP.log_inner_u_integral)
    for phi in np.linspace(0.0, 2 * np.pi, 5)[:4]:
        a, c1, c2 = JP._a_c1_c2(jnp.asarray(C), jnp.atleast_1d(phi))
        g = JN.eval_g(C, np.full(u.size, phi), u)
        m = g.max()
        ref = m + np.log(np.exp(g - m).mean()) + np.log(2 * np.pi)
        got = float(f(float(a[0]), complex(c1[0]), complex(c2[0])))
        assert abs(got - ref) < 1e-3, (phi, got, ref)


def test_spurious_off_circle_roots_do_not_orphan_an_arc():
    """MIDPOINT cells tile the circle for ANY four angles.  Root-bounded cells do not
    when two roots leave the unit circle as a conjugate-reciprocal pair, and the
    orphaned arc silently cost 0.23 nats."""
    # search for a genuine off-circle case rather than hand-picking one: only SOME
    # coefficient pairs push a conjugate-reciprocal pair off the unit circle.
    rng = np.random.default_rng(0)
    c1 = c2 = None
    for _ in range(2000):
        a1 = rng.normal() + 1j * rng.normal()
        a2 = rng.normal() + 1j * rng.normal()
        z = np.roots([a2, a1 / 2, 0, -np.conj(a1) / 2, -np.conj(a2)])
        if np.sum(np.abs(np.abs(z) - 1.0) > 1e-6) >= 2:
            c1, c2 = 0.3 * a1, 0.3 * a2
            break
    assert c1 is not None, "no off-circle fixture found"
    z = np.roots([c2, c1 / 2, 0, -np.conj(c1) / 2, -np.conj(c2)])
    assert np.sum(np.abs(np.abs(z) - 1.0) > 1e-6) >= 2
    u = np.linspace(0.0, 2 * np.pi, 400000, endpoint=False)
    g = (c1 * np.exp(1j * u)).real + (c2 * np.exp(2j * u)).real
    m = g.max()
    ref = m + np.log(np.exp(g - m).mean()) + np.log(2 * np.pi)
    got = float(jax.jit(JP.log_inner_u_integral)(0.0, complex(c1), complex(c2)))
    assert abs(got - ref) < 1e-4, (got, ref)


def test_matches_the_numpy_reference_kernel():
    """Two independent implementations of the same rule: the numpy one merges 2-D
    regions, this one partitions by cells.  They must agree."""
    A, B = _tables(seed=5, scale=3.0)
    x = np.linspace(0.4, 2.2, 12)
    logw = -0.5 * (x - 1.1) ** 2 * 4.0
    ref, ok, _ = JN.joint_marginalize_over_distance(A, B, x, logw, n_phi=64,
                                                    n_bound_grid=128)
    assert ok
    got = float(JP.joint_lnL_phi_dense(jnp.asarray(A), jnp.asarray(B),
                                       jnp.asarray(x), jnp.asarray(logw), n_phi=256))
    assert abs(got - ref) < 1e-5, (got, ref)


def test_phi_chunking_is_a_memory_knob_only():
    """phi_chunk bounds the transient and must not move the answer."""
    A, B = _tables(seed=8, scale=2.0)
    x = np.linspace(0.5, 1.8, 8)
    logw = np.zeros_like(x)
    out = [float(JP.joint_lnL_phi_dense(jnp.asarray(A), jnp.asarray(B), jnp.asarray(x),
                                        jnp.asarray(logw), n_phi=128, phi_chunk=c))
           for c in (8, 16, 64)]
    assert max(out) - min(out) < 1e-11, out


def test_required_n_phi_grows_like_sqrt_amplitude():
    """The phi axis is NOT localized here, so it must be SIZED, not guessed: hard-coding
    it cost 191 nats at amplitude 1.25e4 during development."""
    a, b = JP.required_n_phi(100.0), JP.required_n_phi(10000.0)
    assert b > a
    assert 5.0 < (b / a) / np.sqrt(100.0) * 10.0 < 20.0


# ------------------------------------------------------- differentiability

def test_hessian_works_and_the_gradient_is_still_correct():
    """P1 from review.  jnp.linalg.eigvals has NO second derivative in JAX, so any
    Hessian through this kernel raised -- and the caller that matters, _fisher_whitening,
    swallows that in an `except Exception` and returns None, so --fisher-precondition
    would silently degrade to raw coordinates with the flag still recorded as supplied.

    The fix cuts the tangent before the eigensolve.  That CHANGES the derivative, so the
    gradient must be re-validated, not assumed: these angles are cell boundaries of an
    exact partition, so a boundary shift adds to one cell exactly what it removes from
    its neighbour and cancels."""
    F = lambda x: JP.log_inner_u_integral(0.0, x + 1j, 0.7 - 0.3j)
    assert np.isfinite(float(jax.hessian(F)(2.0)))
    g = jax.grad(F)
    for x in (0.3, 2.0, 7.0, 25.0):
        h = 1e-5
        fd = (float(F(x + h)) - float(F(x - h))) / (2 * h)
        ad = float(g(x))
        assert abs(ad - fd) < 1e-4 * max(abs(fd), 1.0), (x, ad, fd)


def test_gradient_is_finite_as_the_quartic_leading_coefficient_vanishes():
    """P1 from review.  The old guard caught only an EXACT zero; at c2 = 1e-20 the
    companion matrix acquires ~1e20 entries and the eig JVP degenerates -- measured grad
    0.567 at c2=1, -1.2e14 at 1e-20 and nan at 1e-30, while the VALUE stayed fine.  c2 is
    the B-table q=+-2 coefficient and passes through small values at special geometries;
    one nan gradient poisons a MALA/flowMC chain."""
    g = jax.grad(lambda c: JP.log_inner_u_integral(0.0, 2.0 + 1j, c * (1.0 + 0j)))
    vals = [float(g(c2)) for c2 in (1.0, 1e-6, 1e-20, 1e-30)]
    assert all(np.isfinite(v) for v in vals), vals
    # and stable, not merely finite, across 24 orders of magnitude in c2
    assert abs(vals[1] - vals[3]) < 1e-3, vals


def test_required_u_nodes_is_derived_and_grows_like_sqrt_amplitude():
    """P1 from review: the fallback (whole-cell) branch integrates with the SAME fixed
    node count spread over the entire cell, so rejecting a stalled Newton centre makes
    the resolution worse rather than safer.  JAX cannot adapt the count -- shapes may not
    depend on traced values -- so the sizing is exposed as a caller-side helper, derived
    from the exact bound |d2g/du2| <= M2u ~ 5A.

    Production uses this count because fallback is data-dependent.  It is intentionally
    uncapped: memory is bounded by streaming the node axis, not by truncating an accuracy
    request inside a region the omitted-mass certificate cannot inspect.
    """
    lo = JP.required_u_nodes(1.0)
    mid = JP.required_u_nodes(100.0)
    hi = JP.required_u_nodes(1.0e4)
    assert lo == JP.U_NODES_PER_CELL          # never below the windowed default
    assert lo < mid < hi                       # grows with amplitude
    assert JP.u_nodes_in_use(450.0) == JP.required_u_nodes(450.0)
    assert hi > 2048                           # production does not silently cap accuracy
    # the growth is the sqrt law, not something steeper
    assert 5.0 < mid / np.sqrt(100.0) < 60.0, mid


def test_a_fallback_cell_is_resolved_when_the_caller_sizes_it():
    """The helper must actually buy resolution: a whole-cell integration at a raised node
    count must agree with a much finer one."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(6):
        sc = 10.0 ** rng.uniform(0.5, 2.0)
        c1 = sc * (rng.normal() + 1j * rng.normal())
        c2 = sc * (rng.normal() + 1j * rng.normal())
        amp = abs(c1) + 2 * abs(c2)
        n = JP.required_u_nodes(amp)
        a = float(JP.log_inner_u_integral(0.0, c1, c2, n_nodes=n))
        b = float(JP.log_inner_u_integral(0.0, c1, c2, n_nodes=min(4 * n, 4096)))
        worst = max(worst, abs(a - b))
    assert worst < 1e-4, worst


def test_large_fallback_policy_streams_a_fixed_live_node_block():
    """The accurate production count must not reappear as a materialized node axis.

    At the sizing floor the policy requests hundreds of nodes.  Observe the shape handed
    to the exponent evaluator while tracing the rolled loop: its live last axis must stay
    at the stream chunk, independent of the total quadrature count.
    """
    n = JP.u_nodes_in_use(450.0)
    assert n > JP.U_NODE_STREAM_CHUNK
    shapes = []
    real_g = JP._g_u

    def _spy_g(a, c1, c2, u, order=0):
        if order == 0 and getattr(u, "ndim", 0) == 2:
            shapes.append(tuple(u.shape))
        return real_g(a, c1, c2, u, order)

    JP._g_u = _spy_g
    try:
        out = JP.log_inner_u_integral(0.0, 2.0 + 1j, 0.7 - 0.3j, n_nodes=n)
        assert np.isfinite(float(out))
    finally:
        JP._g_u = real_g

    assert shapes, "stream body never reached the exponent evaluator"
    assert max(shape[-1] for shape in shapes) <= JP.U_NODE_STREAM_CHUNK, shapes

# --------------------------------------------- phi localization (both axes local)

def _tables_scaled(seed, scale):
    rng = np.random.default_rng(seed)
    A = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * scale
    B = (rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))) * scale
    B[0, 2] = abs(B[0, 2].real) + 3.0 * scale
    return A, B


def _joint(A, B, x=1.0):
    from RIFT.likelihood import joint_angle_peak_local as JN
    return JN.joint_table(A, B, x=x)


def _torus_ref(C, n=2048):
    from RIFT.likelihood import joint_angle_peak_local as JN
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    P, U = np.meshgrid(t, t, indexing='ij')
    g = JN.eval_g(C, P.ravel(), U.ravel())
    m = g.max()
    return m + np.log(np.exp(g - m).mean()) + 2 * np.log(2 * np.pi)


def test_u_profile_derivatives_match_the_numpy_reference():
    """F' and F'' come from differentiating under the integral, so they are exact and
    cost no extra evaluation.  Two independent implementations must agree."""
    from RIFT.likelihood import joint_angle_peak_local as JN
    A, B = _tables_scaled(3, 3.0)
    C = _joint(A, B)
    f = jax.jit(JP.u_profile)
    for phi in np.linspace(0.4, 5.6, 5):
        F, d1, d2, _, _, _ = f(jnp.asarray(C), float(phi))
        Fn, d1n, d2n = JN.u_profile(C, np.array([phi]))
        assert abs(float(F) - Fn[0]) < 1e-4, (phi, F, Fn[0])
        scale = max(1.0, abs(d1n[0]))
        assert abs(float(d1) - d1n[0]) < 1e-3 * scale, (phi, d1, d1n[0])


@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
def test_phi_local_matches_a_dense_torus_reference(scale):
    A, B = _tables_scaled(3, 1.0)
    C = _joint(A * scale, B * scale)
    got, ok, info = jax.jit(JP.phi_local_lnI)(jnp.asarray(C))
    assert abs(float(got) - _torus_ref(C)) < 1e-4, (scale, float(got))


def test_empty_merge_slots_do_not_poison_the_sum_with_nan():
    """Regression.  There are always more slots than groups, and an empty slot comes
    back from the segment reductions as (+inf, -inf).  Masking its WEIGHT is not enough:
    the node positions are still built from it, jnp.mod(inf, 2pi) is NaN, and NaN * 0 is
    NaN -- so the poison reached the sum through a term that was supposed to be switched
    off.  Every amplitude above ~400 returned NaN before the position was neutralized."""
    for scale in (10.0, 30.0, 100.0, 300.0):
        A, B = _tables_scaled(3, 1.0)
        got, _ok, _info = jax.jit(JP.phi_local_lnI)(jnp.asarray(_joint(A * scale, B * scale)))
        assert np.isfinite(float(got)), (scale, float(got))


def test_phi_local_cost_is_flat_in_amplitude():
    """The point of localizing BOTH axes.  Measured wall time is ~0.19 s at every
    amplitude from 42 to 12650; here we assert the structural property that makes that
    true -- the work is set by static shapes, so the SAME jitted callable serves every
    amplitude without recompiling."""
    f = jax.jit(JP.phi_local_lnI)
    A, B = _tables_scaled(3, 1.0)
    shapes = set()
    for scale in (1.0, 10.0, 100.0):
        C = jnp.asarray(_joint(A * scale, B * scale))
        shapes.add(C.shape)
        assert np.isfinite(float(f(C)[0]))
    assert len(shapes) == 1, shapes      # one shape => one compilation


def test_u_profile_rejects_a_clipped_newton_point_as_a_peak():
    """External-review P1 on the phi-localization branch.  ``u_profile`` classified a cell
    as peaked from ``g'' < 0`` ALONE -- the same defect ``log_inner_u_integral`` already
    gates, reintroduced because this function was written as a fresh copy of that Newton
    iteration rather than as a call to it.  The iteration is clamped to ``[lo_c, mid]``, so
    it can come to rest ON a boundary carrying a large stationary residual; curvature then
    centres a +-window on a non-stationary point and can EXCLUDE the true maximum, which
    underestimates ``F`` while the docstring calls its derivatives exact.

    Non-vacuity is the point of this test: measured over 200 random coefficient draws, the
    gate rejects 7.3% of the cells the curvature-only test accepted, the worst at
    ``|g_u|/M_1 = 0.512``.  A gate that rejected nothing would pass this file's other tests
    just as happily.
    """
    from jax import lax
    rng = np.random.default_rng(3)
    total = rejected = 0
    worst = 0.0
    for _ in range(120):
        sc = 10.0 ** rng.uniform(0.5, 2.0)
        c1 = complex(sc * rng.normal(), sc * rng.normal())
        c2 = complex(sc * rng.normal(), sc * rng.normal())
        u = jnp.sort(JP.u_stationary_roots(c1, c2))
        mid = 0.5 * (u + jnp.roll(u, -1) + jnp.where(jnp.arange(4) == 3, 2 * jnp.pi, 0.0))
        lo_c = jnp.roll(mid, 1) - jnp.where(jnp.arange(4) == 0, 2 * jnp.pi, 0.0)

        def _step(uc, _):
            g1 = JP._g_u(0.0, c1, c2, uc, 1)
            g2 = JP._g_u(0.0, c1, c2, uc, 2)
            st = jnp.where(jnp.abs(g2) > 0, -g1 / jnp.where(jnp.abs(g2) > 0, g2, 1.0), 0.0)
            return jnp.clip(uc + jnp.clip(st, -0.5, 0.5), lo_c, mid), None

        ustar, _ = lax.scan(_step, u, None, length=8)
        g1s = JP._g_u(0.0, c1, c2, ustar, 1)
        g2s = JP._g_u(0.0, c1, c2, ustar, 2)
        m1u = abs(c1) + 2.0 * abs(c2)
        edge = 1e-9 * float(jnp.max(mid - lo_c))
        curvature_only = np.asarray(g2s < 0.0)
        gated = np.asarray((g2s < 0.0)
                           & (jnp.abs(g1s) <= 1e-8 * max(m1u, 1e-300))
                           & (ustar > lo_c + edge) & (ustar < mid - edge))
        assert not (gated & ~curvature_only).any(), "gate must only ever REMOVE cells"
        dropped = curvature_only & ~gated
        total += int(curvature_only.sum())
        rejected += int(dropped.sum())
        if dropped.any():
            r = np.asarray(jnp.abs(g1s)) / max(m1u, 1e-300)
            worst = max(worst, float(r[dropped].max()))
    assert total > 0
    assert rejected > 0, "gate rejected nothing -- it is decoration, not a check"
    assert worst > 1e-3, "worst rejected residual %.3g is within tolerance of stationary" % worst


def test_phi_local_returns_a_certificate_that_actually_declines():
    """External-review P1: ``phi_local_lnI`` returned a bare float -- no bound, no validity
    result, no fallback signal -- while its docstring claimed correctness rested on "the
    caller's cover bound", a contract no caller implemented.  Fixed seeds are targeting,
    not an enumeration, so a missed maximum came back as a finite likelihood.

    It now returns ``(value, ok, info)`` with an omitted-mass bound on the phi axis:
    ``area_outside * exp(sup_outside F)``, the supremum obtained by LIFTING grid values of
    ``F`` with a true remainder from ``profile_derivative_bounds`` -- never the grid
    maximum, which is a lower bound on a supremum.

    The assertion that matters is that it DECLINES: a certificate that always accepts is
    decoration, and would have passed every other test in this file.
    """
    rng = np.random.default_rng(0)
    verdicts = []
    for scale in (0.3, 3.0, 40.0, 200.0):
        C = (rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))) * scale
        val, ok, info = JP.phi_local_lnI(jnp.asarray(C))
        assert np.isfinite(float(val))
        for key in ("margin", "area_outside", "sup_outside", "n_phi_regions",
                    "n_u_fallback"):
            assert key in info, key
        # THE CONTRACT CHANGED AND THIS TEST USED TO PIN THE DEFECT.  It asserted that ok
        # was exactly the margin test and that a full cover MUST be accepted -- which is
        # precisely the conflation test_a_full_cover_no_longer_accepts_unconditionally
        # exists to remove.  Both assertions passed only because this test's four fixtures
        # all happen to converge; adversarial review found them contradicting each other
        # across files.  ok is now the margin test AND the resolution test.
        assert bool(ok) == (float(info["margin"]) < JP.OUTSIDE_TOL_NATS
                            and bool(info["phi_resolved"]))
        if float(info["area_outside"]) == 0.0:
            # nothing omitted, so the margin is -inf; whether that ACCEPTS now depends on
            # the integration having converged, which is the whole point of the change.
            assert float(info["margin"]) == -np.inf
            assert bool(ok) == bool(info["phi_resolved"])
        verdicts.append(bool(ok))
    assert any(verdicts), "certificate declined everything -- it is unusable, not strict"
    assert not all(verdicts), "certificate accepted everything -- it is decoration"


def test_algebraic_phi_seeds_are_complete_and_agree_where_the_cover_is_partial():
    """phi HAS an algebraic warrant, through g rather than through F.  The orbital phase
    enters as e^{-i m phi}, so the table's k_max = KP-1 = 2 m_max is exact, and the
    resultant's z-roots are every phi at which a 2-D stationary point exists -- a complete
    seed set, which a uniform linspace cannot claim to be.

    Asserted here: the seed count is fixed by the MODE CONTENT (16 k_max), and where the
    cover is partial the two seedings agree on the value.
    """
    KS = 2
    for KP in (5, 9):
        rng = np.random.default_rng(101)
        C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
        C = jnp.asarray(C * (1e4 / np.sum(np.abs(C))))
        seeds = JP.phi_seeds_algebraic(C)
        assert seeds.shape[0] == (2 * (KP - 1)) * (2 * KS) * 2, (KP, seeds.shape)
        assert np.isfinite(np.asarray(seeds)).all()
        # THE SHAPE IDENTITY ABOVE IS NOT COMPLETENESS -- it holds whatever the roots are,
        # and it passed while the roots were the REFLECTION of the true ones (the FFT-sign
        # defect).  Adversarial review named it vacuous, correctly.  This is the property
        # that actually distinguishes an enumeration: every true stationary phi must be AT
        # a seed, before any Newton step.
        from RIFT.likelihood import joint_angle_peak_local as _JN
        G, _ = _JN.enumerate_modes(np.asarray(C), n_phi=256)
        if G.shape[0]:
            sd = np.asarray(seeds)
            worst = max(float(np.abs(((G[i, 0] - sd + np.pi) % (2 * np.pi)) - np.pi).min())
                        for i in range(G.shape[0]))
            assert worst < 1e-6, (KP, worst)
        vu, _, iu = JP.phi_local_lnI(C, algebraic_seeds=False)
        va, _, ia = JP.phi_local_lnI(C, algebraic_seeds=True)
        if float(ia["area_outside"]) > 0 and float(iu["area_outside"]) > 0:
            assert abs(float(vu) - float(va)) < 1e-2, (KP, float(vu), float(va))


def test_a_full_cover_no_longer_accepts_unconditionally():
    """The covering path used to conflate two different statements.  ``area_outside = 0``
    says nothing was left OUT; it says nothing about the quadrature INSIDE, yet it gave
    ``margin = -inf`` and an unconditional accept.  Measured before the fix at KP=13,
    amplitude 1e2 with algebraic seeds: full cover, accepted, value 0.196 nats wrong -- the
    same conflation that cost the numpy reference 0.36 nats on production tables.

    ``ok`` now also requires the integration to have CONVERGED, measured by halving the
    nodes -- free, because ``PHI_NODES_PER_REGION`` is odd so indices 0,2,...,n-1 span the
    same interval at double the spacing.  Two gates were tried first and rejected on
    evidence: the exact ``M2F`` bound demands 3.8e3-2.3e4 nodes and declines cases right to
    1e-4, and a local-curvature rule declines cases right to 1e-5, because a periodic
    trapezoid converges spectrally and any points-per-sigma rule is far too conservative.
    """
    KS = 2
    rng = np.random.default_rng(101)
    C = rng.normal(size=(13, 2 * KS + 1)) + 1j * rng.normal(size=(13, 2 * KS + 1))
    C = jnp.asarray(C * (1e2 / np.sum(np.abs(C))))
    v, ok, info = JP.phi_local_lnI(C, algebraic_seeds=True)
    assert float(info["area_outside"]) == 0.0            # the cover IS full
    assert not bool(info["phi_resolved"])                # but the integration is not converged
    assert not bool(ok), "a full cover must not accept an unconverged integration"
    assert float(info["phi_convergence"]) > JP.PHI_CONVERGENCE_NATS


def test_the_convergence_gate_does_not_decline_accurate_results():
    """A gate that refuses correct answers is as useless as one that accepts wrong ones, and
    the two gates tried before this one both did.  These cases are accurate to ~1e-5 against
    a converged torus reference and MUST still accept."""
    KS = 2
    accepted = 0
    for amp in (4.5, 19.0):
        rng = np.random.default_rng(101)
        C = rng.normal(size=(3, 2 * KS + 1)) + 1j * rng.normal(size=(3, 2 * KS + 1))
        C = jnp.asarray(C * (amp / np.sum(np.abs(C))))
        v, ok, info = JP.phi_local_lnI(C)
        assert abs(float(v) - _torus_ref(np.asarray(C))) < 1e-3, (amp, float(v))
        assert float(info["phi_convergence"]) < JP.PHI_CONVERGENCE_NATS, (amp,)
        accepted += bool(ok)
    assert accepted == 2, accepted


def test_algebraic_seeds_stay_off_by_default():
    """Still off: the completeness gain is real, but switching a default that changes which
    rows return a value is a separate decision from making it safe to switch."""
    import inspect
    assert inspect.signature(JP.phi_local_lnI).parameters["algebraic_seeds"].default is False


def test_the_convergence_check_is_guarded_against_its_own_blind_spot():
    """Adversarial review F3.  ``conv`` halves the nodes and compares -- but the n and n/2
    trapezoids share EVERY aliased harmonic at multiples of n, so it measures the n/2
    aliasing and infers the rest from smoothness.  Content at exactly harmonic n is
    invisible to it: review built a table with a phi ripple at n and got values 0.83-0.99
    nats wrong with ``conv`` as low as 1.3e-04 -- BELOW the 1e-3 gate, so ``conv`` alone
    accepted them.

    The guard tested here is ``n_nodes > 2 k_max``.  IT IS NECESSARY AND NOT SUFFICIENT,
    and this docstring used to claim otherwise -- that Nyquist-resolving ``k_max`` "rules
    out content at the sampling harmonic by construction".  That is a statement about
    ``g``; the outer trapezoid integrates ``exp(F)`` with ``F = log int du exp(g)``, and
    neither is band-limited because ``g`` is.  A later review supplied a ``k_max = 1``
    table that passes this guard trivially and is still 0.02 nats wrong -- see
    :func:`test_the_halving_check_is_blind_at_the_sampling_harmonic`, which covers the
    part of the family this guard does not.

    Tested through ``n_nodes`` rather than by building the degree-1552 counterexample,
    which is correct-but-unaffordable in CI.
    """
    KS = 2
    rng = np.random.default_rng(101)
    C = rng.normal(size=(9, 2 * KS + 1)) + 1j * rng.normal(size=(9, 2 * KS + 1))
    C = jnp.asarray(C * (1e2 / np.sum(np.abs(C))))
    k_max = 8                                            # KP - 1

    # under-resolved: the check cannot see harmonic n, so it must not be believed
    _, ok_bad, info_bad = JP.phi_local_lnI(C, n_nodes=2 * k_max - 1)
    assert not bool(info_bad["phi_alias_safe"])
    assert not bool(ok_bad), "an unresolvable node count must never accept"

    # comfortably resolved: the guard must not be what blocks an otherwise good case
    _, _, info_ok = JP.phi_local_lnI(C, n_nodes=JP.PHI_NODES_PER_REGION)
    assert bool(info_ok["phi_alias_safe"]), (JP.PHI_NODES_PER_REGION, k_max)

    # and the guard is load-bearing, not decoration: it must be able to veto a case whose
    # conv is below the threshold, which is exactly what the counterexample showed.
    assert JP.PHI_NODES_PER_REGION > 2 * k_max


def _separable_phi_table(kappa, shift, r=6.0, KS=2):
    """A table whose profile is EXACTLY ``F(phi) = kappa cos(phi - shift) + const``.

    Only ``C[1, q=0]`` and ``C[0, q=+2]`` are set, so ``c1 = 0`` and ``c2 = r`` are both
    phi-independent: the u integral contributes a constant and the phi dependence is the
    single harmonic.  ``k_max = KP - 1 = 1``, and the double integral is closed form,
    ``2 pi I_0(kappa) * 2 pi I_0(r)``, so the error is known rather than estimated.
    """
    C = np.zeros((2, 2 * KS + 1), dtype=complex)
    C[1, KS + 0] = 0.5 * kappa * np.exp(-1j * shift)
    C[0, KS + 2] = r
    from scipy.special import ive
    exact = (np.log(2 * np.pi) + kappa + np.log(ive(0, kappa))
             + np.log(2 * np.pi) + r + np.log(ive(0, r)))
    return jnp.asarray(C), exact


def test_the_halving_check_is_blind_at_the_sampling_harmonic():
    """Adversarial review.  ``conv`` halves the nodes -- but the n and n/2 periodic rules
    alias at multiples of n and n/2, and the second set CONTAINS the first, so the leading
    error term cancels out of the difference.  No subset of the nodes already evaluated can
    ever see it; that is Nyquist, not an implementation shortfall.

    Review's case: ``F = 1000 cos(phi - pi/96)`` on the full circle at 96 intervals.  The
    phase makes the c_48 alias vanish exactly and leaves c_96, so the 96- and 48-interval
    rules agree to 1e-13 while both are 0.02017 nats wrong.  ``k_max = 1`` here, so the
    ``n_nodes > 2 k_max`` guard reports it safe at 97 > 2 and cannot help.

    THE FIX IS THE NODE COUNT, NOT A SECOND GRID.  Because a rule's own aliases are
    invisible in its own samples, the probes can only ever certify the COARSE rule, so the
    answer has to ride a level finer than the probes.  With the nested grid at 193 the
    answer IS the fine rule and comes back right, while the probes still fire because the
    97-node rule they measure was bad -- fail-closed, and correct as well.

    Both halves are asserted, including the blind one: at 97 the probes read ~1e-13 on a
    0.02-nat error.  That is the measurement the default rests on, and it is a statement
    about Nyquist, so it will not stop being true.
    """
    C, exact = _separable_phi_table(1000.0, np.pi / 96)

    # w_sigma forces the wrapped branch: one region spanning 2 pi, which is where a
    # periodic aliasing family can exist at all.
    v, ok, info = JP.phi_local_lnI(C, w_sigma=200.0)
    assert int(info["n_phi_regions"]) == 1, int(info["n_phi_regions"])
    assert abs(float(v) - exact) < 1e-4, float(v) - exact       # the ANSWER is now right
    assert float(info["phi_convergence_shift"]) > JP.PHI_CONVERGENCE_NATS
    assert not bool(ok), "the coarse rule was bad; declining is the conservative direction"

    # why the default is 193 and not 97: at 97 BOTH probes are blind to the error, so the
    # same table would come back wrong and unflagged.
    v9, _, info9 = JP.phi_local_lnI(C, w_sigma=200.0, n_nodes=97)
    assert abs(float(v9) - exact) > 1e-2, float(v9) - exact
    assert float(info9["phi_convergence"]) < 1e-9
    assert float(info9["phi_convergence_shift"]) < 1e-9
    assert bool(info9["phi_alias_safe"])        # and the k_max guard says "safe"

    # ...and the companion is not merely a decline switch: resolved, the table accepts.
    v2, ok2, info2 = JP.phi_local_lnI(C, w_sigma=200.0, n_nodes=769)
    assert abs(float(v2) - exact) < 1e-6, float(v2) - exact
    assert float(info2["phi_convergence_shift"]) < JP.PHI_CONVERGENCE_NATS
    assert bool(ok2), dict(info2)


def test_the_phi_grid_is_nested_so_no_evaluation_is_spent_on_a_probe_alone():
    """The first version of the companion evaluated a SECOND grid of n-1 midpoints, used
    only for the probe and then discarded: 1.85x the cost for a diagnostic.  With an odd
    node count one grid already contains both sub-rules -- even indices are a trapezoid at
    half the density, odd indices are exactly its midpoints -- so both probes are free and
    the returned value is the fine rule.

    Counted at the GRID level, which is the level that costs: under ``jax.vmap`` the
    profile is traced once per grid, so the number of ``u_profile`` invocations is the
    number of distinct grids the kernel builds.  There are four -- the Newton step, the
    seed evaluation, the quadrature grid and the bound grid -- and a separate midpoint
    grid would make five.  The probes must come out of the quadrature grid by striding,
    not out of a grid of their own.
    """
    calls = []
    real = JP.u_profile

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    C, _ = _separable_phi_table(30.0, 0.3)
    JP.u_profile = counting
    try:
        _, _, info = JP.phi_local_lnI(C, n_slots=4, n_seed=4)
    finally:
        JP.u_profile = real
    assert len(calls) == 4, (len(calls), "a fifth grid means a probe is paying its own way")
    assert "phi_convergence_shift" in info

    # and the striding is exact only for an odd count: the even indices must span the same
    # interval and the odd ones must be their midpoints.
    assert JP.PHI_NODES_PER_REGION % 2 == 1


def test_the_outside_bound_gates_on_the_fallback_that_can_invert_it():
    """Adversarial review: ``Fb`` and ``d1b`` were taken from ``u_profile`` with its
    whole-cell fallback and the count was DISCARDED at that call, so a row could be
    accepted on a lift applied to an underestimated profile with no signal it had
    happened.  ``info["n_u_fallback"]`` carried only the Newton-seed evaluation.

    The remedy as stated -- decline whenever any bound-grid profile falls back -- is not
    implementable: every generic table has four u-stationary points of which two are
    minima, so the fallback count is never zero and that gate declines universally
    (measured: 0 of 2 accepted on cases accurate to 1e-5).  A minimum cell has no peak to
    window and is exponentially subdominant in F; the cells that can invert the bound are
    those with ``g'' < 0`` that failed the stationarity or interior test, because a real
    maximum may sit in one unresolved.

    Nor is "did a max-bearing cell fall back" the question: an 8-step Newton misses the
    1e-8 relative residual on plenty of ordinary maxima, and that test fired on 127 of 256
    bound-grid points for tables accurate to 1e-5.  What the bound needs is review's other
    remedy -- whether the whole-cell quadrature was ADEQUATE -- and that is exact here,
    because the u spectrum has two terms so ``|d2g/du2| <= |c1| + 4|c2|`` everywhere and a
    cell of ``width`` needs ``width sqrt(M2u) U_PTS_PER_SIGMA`` nodes.

    So this test pins BOTH directions: a case that must accept with a non-zero fallback
    count, and a case where the gate fires and is CLEARED by sizing the quadrature.
    """
    C, exact = _separable_phi_table(1000.0, np.pi / 96)
    v, ok, info = JP.phi_local_lnI(C, w_sigma=200.0, n_nodes=385)
    assert abs(float(v) - exact) < 1e-4
    assert int(info["n_u_fallback_bound"]) > 0, "the naive gate would have fired here"
    assert int(info["n_u_risky_bound"]) == 0
    assert bool(ok), "gating on the whole-cell count declines every table there is"


def test_the_bound_grid_adequacy_gate_fires_and_is_cleared_by_sizing():
    """Non-vacuity, at the source rather than through the kernel so it stays affordable.

    A gate that never fires is decoration.  This one must fire on a table sharp enough
    that 48 nodes cannot resolve a whole cell, and must CLEAR when the node count is
    raised to what the curvature bound asks for -- that is what makes it a sizing
    requirement the caller can act on rather than a wall.  ``required_u_nodes`` is the
    static helper that computes the same quantity from an amplitude proxy, and both now
    read ``U_PTS_PER_SIGMA`` so the budget and the check cannot drift apart.
    """
    KS = 2
    rng = np.random.default_rng(101)
    C = rng.normal(size=(3, 2 * KS + 1)) + 1j * rng.normal(size=(3, 2 * KS + 1))
    C = jnp.asarray(C * (1.0e4 / np.sum(np.abs(C))))

    fired = cleared = 0
    for phi in np.linspace(0.0, 2 * np.pi, 12, endpoint=False):
        _, _, _, fb_lo, risk_lo, _ = JP.u_profile(C, float(phi), n_nodes=48)
        _, _, _, fb_hi, risk_hi, _ = JP.u_profile(C, float(phi), n_nodes=1024)
        assert int(fb_lo) > 0                      # minima always fall back; that is fine
        fired += int(risk_lo) > 0
        cleared += int(risk_hi) == 0
    assert fired > 0, "an adequacy gate that never fires cannot protect the bound"
    assert cleared == 12, "sizing the quadrature must clear it, or it is not a requirement"
    assert JP.required_u_nodes(1.0e4) > 48
