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

    Deliberately NOT wired into the default: it reaches 2048 nodes at amplitude 1e4,
    roughly 40x the windowed cost, for an effect measured at 2.2e-04 nats in the numpy
    twin -- far below this rule's 23 nat tolerance, on a path no production run reaches.
    A caller that cares can size it; the default documents the limit instead of hiding it.
    """
    lo = JP.required_u_nodes(1.0)
    mid = JP.required_u_nodes(100.0)
    hi = JP.required_u_nodes(1.0e4)
    assert lo == JP.U_NODES_PER_CELL          # never below the windowed default
    assert lo < mid < hi                       # grows with amplitude
    assert hi <= 2048                          # and is capped
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
