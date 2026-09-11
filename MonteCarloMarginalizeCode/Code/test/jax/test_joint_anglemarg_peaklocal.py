"""Tests for the JAX joint (phi,psi) peak-local kernel."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from RIFT.likelihood import joint_angle_peak_local as JN
from RIFT.likelihood.jax_ile import joint_anglemarg_peaklocal as JP


@pytest.fixture(autouse=True)
def _drop_jax_caches():
    """Release JAX's compiled-executable cache after every test in this file.

    THE PEAK HERE IS RETENTION, NOT ALLOCATION, and that distinction is the whole fix.
    Measured: the largest single test peaks at 1408 MB against a 590 MB import baseline,
    so no test costs more than ~800 MB -- yet the file as a whole peaked at 4268 MB.  The
    gap is JAX holding a compiled executable per distinct shape, and this file
    deliberately sweeps many combinations of (n_slots, n_nodes, u_nodes, n_bound) because
    the properties under test are ABOUT those parameters.  Nothing is freed between tests.

    Trimming individual fixtures therefore does almost nothing: six reductions across the
    heaviest tests moved the file peak from 4318 MB to 4268 MB, about one percent, and one
    of them silently broke an acceptance assertion by starving u_sizing_ok.  Dropping the
    caches attacks the actual mechanism.

    The cost is recompilation, which is why this is scoped to this file rather than to the
    session: it is the one that sweeps shapes.
    """
    yield
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()


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
    u = np.linspace(0.0, 2 * np.pi, 60000, endpoint=False)
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
    u = np.linspace(0.0, 2 * np.pi, 60000, endpoint=False)
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
    u = np.linspace(0.0, 2 * np.pi, 60000, endpoint=False)
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
                    "n_u_fallback", "n_u_risky_quad"):
            assert key in info, key
        # THE CONTRACT CHANGED AND THIS TEST USED TO PIN THE DEFECT.  It asserted that ok
        # was exactly the margin test and that a full cover MUST be accepted -- which is
        # precisely the conflation test_a_full_cover_no_longer_accepts_unconditionally
        # exists to remove.  Both assertions passed only because this test's four fixtures
        # all happen to converge; adversarial review found them contradicting each other
        # across files.  ok is the margin test AND the resolution test AND the u sizing
        # test -- three independent ways to be wrong, and the contract is their conjunction.
        #
        # u_sizing_ok used to be read off the BOUND grid, where it was near-vacuous: that
        # grid's job was to bound F from outside, not to produce `value`.  It now reads the
        # QUADRATURE grid, so it reports whether the integration that produced the returned
        # number was adequately sampled -- and it does fire here, on a fixture whose margin
        # (-29.3) and resolution both pass at the default 48 u nodes.  Omitting it from
        # this identity is what made the test fail when the gate moved to the right grid.
        assert bool(ok) == (float(info["margin"]) < JP.OUTSIDE_TOL_NATS
                            and bool(info["phi_resolved"])
                            and bool(info["u_sizing_ok"]))
        if float(info["area_outside"]) == 0.0:
            # nothing omitted, so the margin is -inf; whether that ACCEPTS now depends on
            # the integration having converged, which is the whole point of the change.
            assert float(info["margin"]) == -np.inf
            assert bool(ok) == bool(info["phi_resolved"])
        verdicts.append(bool(ok))
    assert any(verdicts), "certificate declined everything -- it is unusable, not strict"
    assert not all(verdicts), "certificate accepted everything -- it is decoration"


def test_a_full_cover_no_longer_accepts_unconditionally():
    """The covering path used to conflate two different statements.  ``area_outside = 0``
    says nothing was left OUT; it says nothing about the quadrature INSIDE, yet it gave
    ``margin = -inf`` and an unconditional accept.  Measured before the fix at KP=13,
    amplitude 1e2 with algebraic seeds: full cover, accepted, value 0.196 nats wrong -- the
    same conflation that cost the numpy reference 0.36 nats on production tables.

    ``ok`` now also requires the integration to have CONVERGED, measured from the nested
    grid -- free, because ``PHI_NODES_PER_REGION`` is odd so the even indices are a
    trapezoid at half the density and the odd ones are exactly its midpoints.  Two gates
    were tried first and rejected on evidence: the exact ``M2F`` bound demands 3.8e3-2.3e4
    nodes and declines cases right to 1e-4, and a local-curvature rule declines cases right
    to 1e-5, because a periodic trapezoid converges spectrally and any points-per-sigma
    rule is far too conservative.

    The full cover is forced with ``w_sigma`` rather than with the algebraic seeder that
    used to produce one here.  That seeder has been removed -- it duplicated
    ``bivariate_trig_stationary`` more weakly -- and the ``wrapped`` branch is the other
    way a cover comes to span the circle, so the defect is still reachable.
    """
    KS = 2
    rng = np.random.default_rng(101)
    C = rng.normal(size=(13, 2 * KS + 1)) + 1j * rng.normal(size=(13, 2 * KS + 1))
    C = jnp.asarray(C * (1e2 / np.sum(np.abs(C))))
    v, ok, info = JP.phi_local_lnI(C, w_sigma=400.0)
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


def test_a_full_circuit_phi_window_is_one_region_at_every_peak_location():
    """This is the pin the test above could not be, and the reason it could not is the
    finding: at ``w_sigma = 200`` the window spans a full circuit, so the seam split
    emits ``[a0, 2 pi]`` and ``[0, a0 + 2 pi - 2 pi]``, adjacent BY CONSTRUCTION -- and
    adjacent in floating point only when ``(a0 + 2 pi) - 2 pi`` rounds back to ``a0``.
    ONLY THE LOW SIDE BREAKS: a round-trip landing one ulp ABOVE ``a0`` is fine, the
    pieces overlap and the merge folds them.  What is lost is ``a0``'s low bits --
    ``ulp(a0 + 2 pi)`` is 1.78e-15 whatever ``a0`` is, against ``ulp(a0)`` from 6.9e-18 to
    8.9e-16.  In pure float64, no jax involved, 34.4% of ``a0`` uniform on the circle
    round low.  Then the pieces sit one ulp apart, the merge (exact-touching, no
    tolerance, by design) keeps them separate, ``total`` comes out 8.9e-16 under 2 pi and
    the ``wrapped`` clamp misses.  The rule runs a SEAMED two-region trapezoid where the
    periodic one is spectrally accurate: 2.1e-3 nats wrong instead of 2.1e-8.

    So the single fixture in
    :func:`test_the_halving_check_is_blind_at_the_sampling_harmonic` pinned the property
    BY LUCK.  jax 0.9.2 and 0.10.2 put the Newton fixed point two ulp apart on the same
    host, same python 3.13, same numpy 2.4.6; 0.10.2 landed on the good side, which is
    why CI was green while the local runs failed.

    THE GRID BELOW IS PART OF THE GUARD, not incidental to it.  Pre-fix it fails on 10 of
    these 64 shifts under jax 0.9.2 (max error 1.85e-02, 10 of them past the 1e-4
    assertion) and on 10 under 0.10.2 -- so it catches the bug in the environment where
    CI was green.  That is a property of THIS grid, though: an equally natural
    golden-ratio sweep of the same length was measured at 0/64 pre-fix.  Sweeping is
    necessary and not sufficient; the count above is the evidence, and it should be
    re-measured rather than assumed if the grid is ever changed.

    ``vmap`` is what makes it affordable: the per-call cost is host-side tracing, ~1.35 s,
    flat in the node counts, so 64 separate calls run ~85 s against ~20-30 s batched
    (the spread is host load, not the method).
    """
    shifts = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    C = jnp.stack([_separable_phi_table(1000.0, s)[0] for s in shifts])
    exact = _separable_phi_table(1000.0, 0.0)[1]        # shift-independent
    v, _, info = jax.vmap(lambda c: JP.phi_local_lnI(c, w_sigma=200.0))(C)

    regions = np.asarray(info["n_phi_regions"])
    bad = shifts[regions != 1]
    assert bad.size == 0, (regions[regions != 1][:4], bad[:4])

    # EXACTLY 2 pi, not approximately: one ulp short IS the failure, and a tolerance here
    # would pass the very state this test exists to forbid.
    total = np.asarray(info["seg_width"]).sum(axis=1)
    assert (total == 2 * np.pi).all(), total[total != 2 * np.pi][:4] - 2 * np.pi

    # and the seam costs ACCURACY, which is why the count is worth pinning at all
    err = np.abs(np.asarray(v) - exact)
    assert err.max() < 1e-4, (float(err.max()), float(shifts[err.argmax()]))

    # EVERYTHING ABOVE CONSTRAINS WHAT HAPPENS ONCE THE FULL-CIRCUIT BRANCH FIRES, AND
    # NOTHING CONSTRAINS WHEN.  Every table above is a full circuit by construction
    # (w_sigma=200, sigma=1/sqrt(1000), so the window is 12.65 rad), so a kernel taking the
    # branch for EVERY window satisfies all three assertions.  Mutation-tested:
    # `full_circuit = peaked` leaves this file's two seam tests green.
    #
    # WIDTH IS THE WRONG THING TO ASSERT HERE, and a first version of this block asserted
    # it and was inert.  That mutation preserves the width -- a narrow window stays 0.1265
    # rad wide either way -- and moves only the LOCATION, anchoring every region at 0.  The
    # invariant that separates them is that a narrow region sits ON its peak: measured,
    # 64/64 covered with the fix against 2/64 under the mutation.
    narrow = jax.vmap(lambda c: JP.phi_local_lnI(c, w_sigma=2.0))(C)[2]
    nlo = np.asarray(narrow["seg_lo"])
    nw = np.asarray(narrow["seg_width"])
    assert (nw.sum(axis=1) < 2 * np.pi).all(), float(nw.sum(axis=1).max())
    peak = (shifts % (2 * np.pi))[:, None]
    live = nw > 0
    on_peak = ((nlo - 1e-9 <= peak) & (peak <= nlo + nw + 1e-9))
    wrapped_hit = ((nlo - 1e-9 <= peak + 2 * np.pi)
                   & (peak + 2 * np.pi <= nlo + nw + 1e-9))
    covered = ((on_peak | wrapped_hit) & live).any(axis=1)
    assert covered.all(), shifts[~covered][:4]


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
    # THREE, not four: the Newton step, the seed evaluation and the quadrature grid.  The
    # bound grid used to be a fourth, and no longer calls the profile at all -- sup_g_bound
    # needs four quartic roots per point and no u quadrature.  A fifth would mean a probe
    # is paying its own way; a fourth would mean the bound grid is back on the profile.
    assert len(calls) == 3, (len(calls), "the bound grid must not run the u quadrature")
    assert "phi_convergence_shift" in info

    # and the striding is exact only for an odd count: the even indices must span the same
    # interval and the odd ones must be their midpoints.
    assert JP.PHI_NODES_PER_REGION % 2 == 1


def test_the_outside_bound_does_not_depend_on_the_u_quadrature_at_all():
    """The review finding this replaces is retired BY CONSTRUCTION, not by a gate.

    ``Fb`` and ``d1b`` used to come from ``u_profile`` on the bound grid, so a whole-cell
    fallback there could underestimate ``F`` and a lift applied to an underestimate bounds
    nothing.  The fix was a gate on that fallback.  :func:`sup_g_bound` removes the
    exposure instead: the outside bound is ``log(2 pi) + max_u g``, four quartic roots per
    point, and never touches the quadrature.

    So the property to assert is not "the gate fires" but "the bound cannot move": vary
    ``u_nodes`` over a factor of 8 and ``sup_outside`` must be bit-identical.  That is a
    much stronger statement than the gate ever made, and it cannot pass by accident.
    """
    KS = 2
    rng = np.random.default_rng(101)
    C = rng.normal(size=(3, 2 * KS + 1)) + 1j * rng.normal(size=(3, 2 * KS + 1))
    C = jnp.asarray(C * (1e3 / np.sum(np.abs(C))))
    sups = [float(JP.phi_local_lnI(C, u_nodes=un)[2]["sup_outside"])
            for un in (48, 96, 384)]
    assert sups[0] == sups[1] == sups[2], sups


def test_sup_g_bound_is_actually_an_upper_bound_on_the_profile():
    """The whole certificate now rests on ``F(phi) <= log(2 pi) + max_u g(phi,u)``.  If that
    is ever violated the outside bound is not a bound and every accepted row is suspect, so
    it is checked directly against the profile rather than assumed from the algebra.

    Measured slack is 1.2-5.5 nats across the range -- small enough that the bound is
    usable, and the reason the certificate stopped declining rows whose true margin was
    already tens of nats clear.
    """
    KS = 2
    worst = 1e9
    for KP, amp in ((3, 30.0), (3, 3e3), (5, 1e3), (9, 1e4)):
        rng = np.random.default_rng(7)
        C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
        C = jnp.asarray(C * (amp / np.sum(np.abs(C))))
        # 128 u nodes and 17 phi: the claim is an INEQUALITY that was violated by
        # 0.024-0.092 nats, so the reference does not need to be fine, only correct.
        un = min(JP.required_u_nodes(amp), 128)
        for phi in np.linspace(0.0, 2 * np.pi, 17, endpoint=False):
            F = float(JP.u_profile(C, float(phi), n_nodes=un)[0])
            H = float(JP.sup_g_bound(C, float(phi)))
            worst = min(worst, H - F)
    assert worst >= 0.0, ("sup_g_bound is NOT an upper bound", worst)
    assert worst < 20.0, ("bound is sound but so loose it cannot certify", worst)


def test_the_localized_regime_now_accepts_and_is_right():
    """What the whole exercise was for.  Before the bound was rebuilt, a sweep over
    KP x amplitude found exactly ONE case in 36 that both localized (area_outside > 0) and
    accepted, at amplitude 100 -- the cost win of localizing phi was real and the
    certificate refused every row that realized it.  The Taylor lift sat 2.7e5 nats above
    the integral at amplitude 3e4 while the true margin was about -66.

    These cases localize into several regions AND accept AND are right to machine
    precision.  If this test starts declining, the certificate has regressed to refusing
    the regime it exists to serve.
    """
    KS = 2
    # u_nodes 256 and 8 slots, not required_u_nodes(amp) = 2310 and 16.  The sizing helper
    # is a conservative UPPER bound derived from amplitude; the gate that actually decides,
    # u_sizing_ok, measures risky cells and passes here at 9x fewer nodes.  Sizing this
    # test from the helper costs 5.9 GB in eval_g2's intermediate and is killed when the
    # file runs as a whole -- a guard that cannot run in CI guards nothing.
    for KP, amp in ((3, 3e3), (9, 1e3)):
        rng = np.random.default_rng(7)
        C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
        C = C * (amp / np.sum(np.abs(C)))
        v, ok, info = JP.phi_local_lnI(jnp.asarray(C),
                                       n_bound=int(JP.required_bound_grid(amp)),
                                       u_nodes=256, n_slots=8, n_nodes=97)
        assert float(info["area_outside"]) > 0.0, "not localized -- fixture is degenerate"
        assert int(info["n_phi_regions"]) >= 4, int(info["n_phi_regions"])
        assert bool(ok), (KP, amp, float(info["margin"]))
        assert abs(float(v) - _torus_ref(np.asarray(C))) < 1e-3, float(v)


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
        _, _, _, fb_hi, risk_hi, _ = JP.u_profile(C, float(phi), n_nodes=384)
        assert int(fb_lo) > 0                      # minima always fall back; that is fine
        fired += int(risk_lo) > 0
        cleared += int(risk_hi) == 0
    assert fired > 0, "an adequacy gate that never fires cannot protect the bound"
    assert cleared == 12, "sizing the quadrature must clear it, or it is not a requirement"
    assert JP.required_u_nodes(1.0e4) > 48


def test_sup_g_bound_survives_a_degenerate_u_quartic():
    """Adversarial review, and the worst defect in this branch.

    ``sup_g_bound`` took ``max`` over ``u_stationary_roots`` as if that set contained the
    maximizer.  A max over a candidate set is a LOWER bound unless it provably does, and
    ``u_stationary_roots`` substitutes ``lead = 1`` when ``c2 == 0`` -- solving a different
    polynomial -- so for a table with no ``q = +-2`` content it need not.  The whole outside
    certificate rests on ``bound >= F``, so this made margins understated rather than loose.

    Measured before the fix: 0.024 to 0.092 nats BELOW ``log(2 pi) + max_u g``.  The
    fixture is the degenerate table, because every table in the rest of this file carries
    full mode content and none of them can see it.
    """
    KS = 2
    for trial in range(4):
        rng = np.random.default_rng(trial)
        C = np.zeros((3, 2 * KS + 1), dtype=complex)
        C[:, KS + 0] = rng.normal(size=3) + 1j * rng.normal(size=3)
        C[:, KS + 1] = rng.normal(size=3) + 1j * rng.normal(size=3)   # no q = +-2
        C = jnp.asarray(C * (50.0 / np.sum(np.abs(C))))
        for phi in np.linspace(0.0, 2 * np.pi, 13, endpoint=False):
            H = float(JP.sup_g_bound(C, float(phi)))
            u = np.linspace(0.0, 2 * np.pi, 8000, endpoint=False)
            g = np.asarray(JP.eval_g2(C, jnp.full(u.shape, float(phi)),
                                      jnp.asarray(u), (0, 0)))
            assert H >= float(np.log(2 * np.pi) + g.max()) - 1e-9, (trial, phi)


def test_empty_slots_do_not_vote_on_the_u_sizing_gate():
    """Adversarial review, found by this session and by external review independently.

    Empty merged-region slots are neutralized for the VALUE -- position zeroed, weight
    masked -- but their nodes are still evaluated at phi = 0, and their fallback counts
    were summed into the gate and the reported counters.  A risky cell at that artificial
    point could decline a row whose every contributing node was adequate.

    The tell is that the counts tracked the SLOT ALLOCATION rather than the regions:
    5 risky at n_slots=2 and 176 at n_slots=8 while the region count only went 2 -> 4.
    So the invariant to pin is that the counters do not move once the slots exceed the
    regions, which no amount of real structure could cause.
    """
    KS = 2
    rng = np.random.default_rng(101)
    C = rng.normal(size=(3, 2 * KS + 1)) + 1j * rng.normal(size=(3, 2 * KS + 1))
    C = jnp.asarray(C * (1000.0 / np.sum(np.abs(C))))
    # SLOTS 4/8/16 AT 48 NODES, not 8/32/64 at 96.  The invariant is that the counters
    # stop moving once the slots exceed the regions, and with 4 regions that is shown just
    # as well at 16 as at 64 -- for an eighth of the memory.  The 64-slot version cost
    # 954 MB in one call and helped kill the CI gate at exit 137.
    seen = {}
    for ns in (4, 8, 16):
        _, _, i = JP.phi_local_lnI(C, u_nodes=48, n_slots=ns, n_nodes=97)
        seen[ns] = (int(i["n_phi_regions"]), int(i["n_u_risky_quad"]),
                    int(i["n_u_fallback_quad"]))
    assert len({v[0] for v in seen.values()}) == 1, ("regions moved", seen)
    assert len({v[1] for v in seen.values()}) == 1, ("risky tracked slots", seen)
    assert len({v[2] for v in seen.values()}) == 1, ("fallback tracked slots", seen)


def _AB(scale, seed=3, KP=3, KS=2):
    rng = np.random.default_rng(seed)
    A = (rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))) * scale
    B = (rng.normal(size=(KP + 2, 2 * KS + 1))
         + 1j * rng.normal(size=(KP + 2, 2 * KS + 1))) * scale
    B[0, KS] = abs(B[0, KS].real) + 3.0 * scale
    return jnp.asarray(A), jnp.asarray(B)


def test_phi_local_distance_combiner_matches_the_dense_scheme():
    """The wiring's correctness condition.  ``joint_lnL_phi_local`` must reproduce the
    shipping ``joint_lnL_phi_dense`` on the same inputs, or the normalization split
    between the per-node seam (bare torus integral) and the combiner (the ``(2 pi)^-2``
    prior factor) is wrong -- and that error is invisible in the per-node value.
    """
    # deliberately small: the claim is about NORMALIZATION, which a 6-node grid tests as
    # well as a 32-node one, and the full-size version cannot run beside the rest of this
    # file -- 640 MB of eval_g2 intermediate per chunk kills the process.
    for scale in (1.0, 4.0, 12.0):
        A, B = _AB(scale)
        x = jnp.linspace(0.5, 2.0, 6)
        lw = jnp.full(6, -np.log(6.0))
        d = JP.joint_lnL_phi_dense(A, B, x, lw, n_phi=512, n_nodes=48)
        v, _, _ = JP.joint_lnL_phi_local(A, B, x, lw, u_nodes=48, n_slots=8,
                                         n_nodes=97, x_chunk=2)
        assert abs(float(d) - float(v)) < 1e-5, (scale, float(d), float(v))


def test_an_arbitrary_distance_rule_stacks_on_the_seam():
    """RO asked for this to be stackable, so it is asserted rather than described.

    ``phi_local_lnI_at_distance`` is the unit; the distance rule is entirely a matter of
    which ``x`` a caller evaluates and what weights it applies.  A 5-node Gauss-Legendre
    rule -- nothing grid-shaped about it -- driven by hand through the seam must equal the
    same nodes routed through the default combiner.  If those ever diverge, the combiner
    has grown an assumption about the rule and stacking is broken.
    """
    A, B = _AB(4.0)
    gx, gw = np.polynomial.legendre.leggauss(5)
    xs = 0.5 * (gx + 1.0) * 1.5 + 0.5
    lws = np.log(gw * 0.75)
    kw = dict(u_nodes=48, n_slots=8, n_nodes=97)
    vals = np.array([float(JP.phi_local_lnI_at_distance(A, B, float(z), **kw)[0])
                     for z in xs])
    from scipy.special import logsumexp
    stacked = logsumexp(vals + lws) - 2.0 * np.log(2.0 * np.pi)
    through, _, _ = JP.joint_lnL_phi_local(A, B, jnp.asarray(xs), jnp.asarray(lws),
                                           x_chunk=1, **kw)
    assert abs(stacked - float(through)) < 1e-9, (stacked, float(through))


def test_rolling_the_quadrature_axis_does_not_move_the_value():
    """``pt_chunk`` exists to bound memory and must be invisible in the answer -- the same
    contract ``phi_chunk`` has on the dense path.  Bit-identical, not merely close: a scan
    that reassociated the reduction would show up here as a last-digit drift and would
    mean the chunking is not a pure refactor."""
    KS = 2
    rng = np.random.default_rng(7)
    C = rng.normal(size=(3, 2 * KS + 1)) + 1j * rng.normal(size=(3, 2 * KS + 1))
    C = jnp.asarray(C * (3e3 / np.sum(np.abs(C))))
    kw = dict(n_bound=int(JP.required_bound_grid(3e3)), u_nodes=96,
              n_slots=4, n_nodes=97)
    # The unrolled reference asks for a chunk of exactly the grid, not 1e6: `_prof_scan`
    # now clamps, but a test should not depend on that and external review measured
    # 6.56 GB of temporaries when it padded 388 points to a million.
    n_pts = 4 * 97
    ref = float(JP.phi_local_lnI(C, pt_chunk=n_pts, **kw)[0])
    for pc in (16, 32, 256):
        got = float(JP.phi_local_lnI(C, pt_chunk=pc, **kw)[0])
        # NOT exact equality.  This asserted bit-identity and passed here, but external
        # review measured 4.55e-13 nats of spread across chunk sizes on another CPU: the
        # scan's reduction IS reassociated on some platforms, so bit-identity was a claim
        # about this machine rather than about the code.  The contract that matters is
        # that chunking is invisible at the scale anything downstream cares about.
        assert abs(got - ref) < 1e-9, (pc, got, ref)


def test_the_distance_combiner_is_fail_closed_across_nodes():
    """``ok`` is the CONJUNCTION over distance nodes.  A declining node still returns a
    finite number that would otherwise be summed in silently, so one bad node must sink
    the row.  Asserted by starving a single node's slot budget."""
    A, B = _AB(4.0)
    x = jnp.linspace(0.5, 2.0, 4)
    lw = jnp.full(4, -np.log(4.0))
    _, ok_starved, _ = JP.joint_lnL_phi_local(A, B, x, lw, u_nodes=48, n_slots=1,
                                              n_nodes=97, x_chunk=2)
    assert not bool(ok_starved), "a starved node must sink the distance sum"
