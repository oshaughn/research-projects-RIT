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
