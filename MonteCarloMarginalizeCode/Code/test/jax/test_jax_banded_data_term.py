"""Regression tests for the compact banded ``<d|h>`` contraction.

The production compound bank has many response basis elements.  Expanding the
two basis/mode reductions as Python loops made the staged JAX program grow with
``A * K`` and dominated cold-start compilation.  These tests pin both the
numerical contract and the requirement that changing ``A`` does not change the
top-level JAX program size.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import core as JC


def _problem(A=5, K=3, S=2):
    rng = np.random.default_rng(1905 + A + K)
    nfull, npts, M = max(32, 8 + 5 * (S - 1) + 6 + 4), 6, 4

    def complex_normal(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    q = jnp.asarray(complex_normal((A, nfull, K)), dtype=jnp.complex128)
    conj_y = jnp.asarray(complex_normal((S, K)), dtype=jnp.complex128)
    coeff = jnp.asarray(complex_normal((A, S)), dtype=jnp.complex128)
    # Include distinct sub-sample positions for the two samples while staying
    # away from the buffer edge for every interpolation stencil.
    pos = jnp.asarray([[8.2 + 5 * i + j for j in range(npts)]
                       for i in range(S)], dtype=jnp.float64)
    u = pos - jnp.floor(pos[:, :1]) - jnp.arange(npts)[None, :]
    pp_t1 = jnp.asarray(np.arange(A) % M, dtype=jnp.int32)
    pe = jnp.asarray(np.exp(1j * rng.normal(size=(M, S))), dtype=jnp.complex128)
    pt = jnp.asarray(np.exp(1j * rng.normal(size=(M, npts))), dtype=jnp.complex128)
    return q, conj_y, coeff, pos, u, pp_t1, pe, pt


def _python_oracle(q, conj_y, coeff, gather, pos, u, pp_t1=None, pe=None, pt=None):
    """Literal pre-optimization A-by-K contraction, retained only as an oracle."""
    A, _, K = q.shape
    out = jnp.zeros(pos.shape, dtype=jnp.complex128)
    for a in range(A):
        inner = jnp.zeros(pos.shape, dtype=jnp.complex128)
        for k in range(K):
            inner = inner + conj_y[:, k, None] * gather(q[a, :, k], pos, u)
        if pp_t1 is None:
            out = out + jnp.conj(coeff[a])[:, None] * inner
        else:
            im = int(pp_t1[a])
            out = out + ((jnp.conj(coeff[a]) * pe[im])[:, None]
                         * (pt[im][None, :] * inner))
    return out


@pytest.mark.parametrize("interp", ["nearest", "linear", "cubic", "sinc"])
@pytest.mark.parametrize("post_phase", [False, True])
def test_compact_contraction_matches_literal_oracle(interp, post_phase):
    q, conj_y, coeff, pos, u, pp_t1, pe, pt = _problem()
    gather = JC._GATHERERS[interp]
    kwargs = dict(pp_t1=pp_t1, pe=pe, pt=pt) if post_phase else {}
    expected = _python_oracle(q, conj_y, coeff, gather, pos, u, **kwargs)
    got = JC._contract_banded_data_term(
        q, conj_y, coeff, gather, pos, u, **kwargs)
    got_jit = jax.jit(
        lambda qq, cc: JC._contract_banded_data_term(
            qq, conj_y, cc, gather, pos, u, **kwargs))(q, coeff)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected),
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(np.asarray(got_jit), np.asarray(expected),
                               rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_compact_contraction_retains_extrinsic_reverse_mode_ad(interp):
    q, conj_y, coeff, pos, u, pp_t1, pe, pt = _problem()
    gather = JC._GATHERERS[interp]
    coeff_phase = jnp.linspace(-0.7, 0.9, coeff.shape[0])[:, None]

    def scalar(contract, pos_shift, angle):
        shifted_pos = pos + pos_shift
        # The wired separable-u path differentiates its common fractional p0
        # offset.  We remain away from integer knots, so u + shift is identical
        # to recomputing that offset in this neighborhood.
        shifted_u = u + pos_shift
        shifted_coeff = coeff * jnp.exp(1j * angle * coeff_phase)
        value = contract(
            q, conj_y, shifted_coeff, gather, shifted_pos, shifted_u,
            pp_t1=pp_t1, pe=pe, pt=pt)
        return jnp.real(jnp.sum(value * jnp.conj(value)))

    compact = lambda shift, angle: scalar(
        JC._contract_banded_data_term, shift, angle)
    literal = lambda shift, angle: scalar(
        _python_oracle, shift, angle)
    point = (jnp.asarray(0.031), jnp.asarray(-0.23))
    compact_value = compact(*point)
    literal_value = literal(*point)
    compact_grad = jax.grad(compact, argnums=(0, 1))(*point)
    literal_grad = jax.grad(literal, argnums=(0, 1))(*point)
    assert np.isfinite(float(compact_value))
    np.testing.assert_allclose(np.asarray(compact_value), np.asarray(literal_value),
                               rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.asarray(compact_grad), np.asarray(literal_grad),
                               rtol=3e-12, atol=3e-12)
    assert np.all(np.abs(np.asarray(compact_grad)) > 1e-8)


def test_jax_program_does_not_grow_with_number_of_bands():
    """A=40 must remain a loop bound, not become forty copies of the body."""

    def trace(A):
        q, conj_y, coeff, pos, u, pp_t1, pe, pt = _problem(A=A, K=2)
        return jax.make_jaxpr(
            lambda qq: JC._contract_banded_data_term(
                qq, conj_y, coeff, JC._gather_cubic, pos, u,
                pp_t1=pp_t1, pe=pe, pt=pt))(q).jaxpr

    small = trace(2)
    production_sized = trace(40)
    assert len(small.eqns) == len(production_sized.eqns)
    assert any(eqn.primitive.name in ("scan", "while")
               for eqn in production_sized.eqns)


def test_partial_post_phase_contract_is_rejected():
    q, conj_y, coeff, pos, u, pp_t1, _, _ = _problem()
    with pytest.raises(ValueError, match="must be supplied together"):
        JC._contract_banded_data_term(
            q, conj_y, coeff, JC._gather_cubic, pos, u, pp_t1=pp_t1)


def test_chunked_rows_and_samples_preserve_padded_tail(monkeypatch):
    # Both A*K and S are nonmultiples of their tile sizes.  Repeated tail
    # indices must not leak into the returned samples or their derivatives.
    q, conj_y, coeff, pos, u, pp_t1, pe, pt = _problem(S=5)
    gather = JC._gather_cubic
    expected = _python_oracle(
        q, conj_y, coeff, gather, pos, u, pp_t1, pe, pt)
    monkeypatch.setattr(JC, "_banded_chunk_shape",
                        lambda *args: (2, 3, 0))

    def contracted(offset):
        return JC._contract_banded_data_term(
            q, conj_y, coeff, gather, pos + offset, u + offset,
            pp_t1=pp_t1, pe=pe, pt=pt)

    got = jax.jit(contracted)(0.0)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected),
                               rtol=2e-14, atol=2e-14)
    literal_grad = jax.grad(lambda x: jnp.real(jnp.sum(
        jnp.abs(_python_oracle(q, conj_y, coeff, gather, pos + x, u + x,
                              pp_t1, pe, pt)) ** 2)))(0.031)
    tiled_grad = jax.grad(lambda x: jnp.real(jnp.sum(
        jnp.abs(contracted(x)) ** 2)))(0.031)
    np.testing.assert_allclose(np.asarray(tiled_grad), np.asarray(literal_grad),
                               rtol=3e-12, atol=3e-12)


def test_chunk_shape_respects_forward_scratch_budget():
    budget = 2 * 1024**2
    rows, samples, estimated = JC._banded_chunk_shape(
        A=100, K=20, S=2000, npts=128, nfull=4096, taps=16,
        budget=budget)
    assert 1 <= rows < 2000
    assert 1 <= samples < 2000
    assert estimated <= budget
    with pytest.raises(ValueError, match="exceeds the scratch budget"):
        JC._banded_chunk_shape(100, 20, 2000, 128, 4096, 16, budget=1024)


def test_empty_sample_batch_preserves_empty_result():
    q, conj_y, coeff, pos, u, pp_t1, pe, pt = _problem()
    args = (q, conj_y[:0], coeff[:, :0], JC._gather_cubic,
            pos[:0], u[:0])
    kwargs = dict(pp_t1=pp_t1, pe=pe[:, :0], pt=pt)
    got = JC._contract_banded_data_term(*args, **kwargs)
    got_jit = jax.jit(lambda: JC._contract_banded_data_term(
        *args, **kwargs))()
    assert got.shape == (0, pos.shape[1])
    assert got_jit.shape == got.shape
    np.testing.assert_array_equal(np.asarray(got_jit), np.asarray(got))
