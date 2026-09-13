"""Adversarial array-level tests for RIFT.likelihood.gpu_precompute.

These inputs are intentionally non-Hermitian.  Positive-only or rfft-based
implementations must fail several tests here.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from RIFT.likelihood.gpu_precompute import (
    GPUPrecomputeContext, build_compound_basis, compound_precompute_arrays,
    streamed_v_matrix,
)

from conftest import to_host
from oracle import direct_log_likelihood, q_oracle, uv_oracle


def test_default_context_separates_devices_and_explicit_context_rejects_switch(monkeypatch):
    from RIFT.likelihood import gpu_precompute as gpu
    selected = [0]
    backend = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: selected[0])),
        asarray=lambda value: np.array(value, copy=True))
    monkeypatch.setattr(gpu, "_DEFAULT_CONTEXTS", {})
    zero = gpu.default_context(backend)
    first = zero.array("data", np.arange(4.))
    selected[0] = 1
    one = gpu.default_context(backend)
    assert one is not zero
    second = one.array("data", np.arange(4.))
    assert second is not first
    with pytest.raises(RuntimeError, match="belongs to CUDA device 0"):
        zero.array("data", np.arange(4.))
    with pytest.raises(RuntimeError, match="belongs to CUDA device 0"):
        zero.clear()
    selected[0] = 0
    assert gpu.default_context(backend) is zero
    assert zero.array("data", np.arange(4.)) is first


def _case(seed=2917, a=3, m=2, n=32):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(a, m, n)) + 1j * rng.normal(size=(a, m, n))
    # Deliberately independent rather than conj(basis); V must use this argument.
    basis_conj = rng.normal(size=(a, m, n)) + 1j * rng.normal(size=(a, m, n))
    data = rng.normal(size=n) + 1j * rng.normal(size=n)
    weights = rng.uniform(0.05, 2.0, size=n)
    # Asymmetric zeros ensure the implementation neither assumes Hermitian weights nor
    # derives support from one half of the spectrum.
    weights[[0, 3, n // 2 + 2, n - 1]] = 0.0
    return basis, basis_conj, data, weights


def _call(backend, case, *, delta_f=0.125, delta_t=None, n_shift=5,
          n_window=13, return_device=False, **kwargs):
    basis, basis_conj, data, weights = case
    if delta_t is None:
        delta_t = 1.0 / (basis.shape[-1] * delta_f)
    return compound_precompute_arrays(
        backend.asarray(basis), backend.asarray(basis_conj), backend.asarray(data),
        backend.asarray(weights), delta_f, delta_t, n_shift, n_window,
        backend=backend, return_device=return_device, **kwargs
    )


def _unpack(result):
    if isinstance(result, dict):
        return result["Q"], result["U"], result["V"]
    assert len(result) >= 3
    return result[:3]


def test_nonhermitian_q_uv_match_independent_oracle(backend):
    case = _case()
    q, u, v = map(to_host, _unpack(_call(backend, case, return_device=True)))
    q0 = q_oracle(case[0], case[2], case[3], 0.125, 5, 13)
    u0, v0 = uv_oracle(case[0], case[1], case[3], 0.125)
    np.testing.assert_allclose(q, q0, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(u, u0, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(v, v0, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("n_shift", [0, 1, 7, -3, 31, 35])
def test_roll_cut_and_wraparound(backend, n_shift):
    case = _case(n=32)
    q, _, _ = _unpack(_call(backend, case, n_shift=n_shift, n_window=11))
    expected = q_oracle(case[0], case[2], case[3], 0.125, n_shift, 11)
    np.testing.assert_allclose(to_host(q), expected, rtol=3e-12, atol=3e-12)


def test_centered_negative_frequency_bins_are_live(backend):
    n = 16
    basis = np.zeros((1, 1, n), complex)
    data = np.zeros(n, complex)
    weights = np.zeros(n)
    # A single negative-frequency bin in centered LAL ordering.  A positive-only/rfft
    # implementation returns zero; a missing ifftshift gives the wrong alternating phase.
    k = n // 2 + 3
    basis[0, 0, k] = 1.25 - 0.4j
    data[k] = -0.2 + 2.0j
    weights[k] = 0.7
    case = basis, basis * (0.3 + 0.8j), data, weights
    q, u, v = map(to_host, _unpack(_call(backend, case, n_shift=0, n_window=n)))
    np.testing.assert_allclose(q, q_oracle(basis, data, weights, 0.125, 0, n), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(u, uv_oracle(basis, case[1], weights, 0.125)[0], rtol=2e-13, atol=2e-13)
    assert np.max(np.abs(q)) > 0 and np.max(np.abs(v)) > 0


def test_fractional_sample_phase_and_complex_phase_are_preserved(backend):
    case = list(_case(n=64))
    dt = 1.0 / (64 * 0.125)
    tau = 0.37 * dt
    # Physical frequencies matching the centered LAL layout.
    f = np.arange(-32, 32) * 0.125
    phase = np.exp(-2j * np.pi * f * tau)
    case[0] = case[0] * phase
    case[1] = case[1] * np.conj(phase)
    case = tuple(case)
    q, u, v = map(to_host, _unpack(_call(backend, case, delta_t=dt, n_shift=-4, n_window=29)))
    q0 = q_oracle(case[0], case[2], case[3], 0.125, -4, 29)
    u0, v0 = uv_oracle(case[0], case[1], case[3], 0.125)
    np.testing.assert_allclose(q, q0, rtol=4e-12, atol=4e-12)
    np.testing.assert_allclose(u, u0, rtol=4e-12, atol=4e-12)
    np.testing.assert_allclose(v, v0, rtol=4e-12, atol=4e-12)


def test_v_uses_supplied_conjugate_family_with_correct_orientation(backend):
    case = _case(seed=8)
    q, u, v = map(to_host, _unpack(_call(backend, case)))
    _, v0 = uv_oracle(case[0], case[1], case[3], 0.125)
    wrong = uv_oracle(case[0], np.conj(case[0]), case[3], 0.125)[1]
    np.testing.assert_allclose(v, v0, rtol=3e-12, atol=3e-12)
    assert np.max(np.abs(v - wrong)) > 1e-3
    np.testing.assert_allclose(u, np.swapaxes(np.swapaxes(u.conj(), 0, 1), 2, 3), rtol=3e-12, atol=3e-12)


def test_memory_bounded_streamed_v_matches_dense_oracle(backend):
    rng = np.random.default_rng(884)
    n, m, b = 40, 2, 3
    df, dt = 0.2, 1.0 / (40 * 0.2)
    base = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
    base_c = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
    response = rng.normal(size=(b, n)) + 1j * rng.normal(size=(b, n))
    a_list = [(0, 0, -2), (1, 1, 0), (2, 0, 3), (0, 2, -1), (2, 1, 2)]
    weights = rng.uniform(0.0, 1.5, size=n)
    primary = build_compound_basis(
        backend.asarray(base), backend.asarray(response), a_list, df, dt, -1.75,
        f_sidereal=0.031, backend=backend, fft_batch=2)
    dense_c = build_compound_basis(
        backend.asarray(base_c), backend.asarray(response), a_list, df, dt, -1.75,
        f_sidereal=0.031, backend=backend, fft_batch=3)
    expected = uv_oracle(to_host(primary), to_host(dense_c), weights, df)[1]
    # Awkward block and chunk sizes force all boundary branches.
    got = streamed_v_matrix(
        backend.asarray(base_c), backend.asarray(response), a_list, primary,
        backend.asarray(weights), df, dt, -1.75, f_sidereal=0.031,
        backend=backend, a_block=2, fft_batch=1, frequency_chunk=7,
        return_device=True)
    np.testing.assert_allclose(to_host(got), expected, rtol=4e-12, atol=4e-12)


def test_downstream_full_gaussian_likelihood_parity(backend):
    case = _case(seed=192, a=4, m=3, n=48)
    q, u, _ = map(to_host, _unpack(_call(backend, case, n_shift=0, n_window=1)))
    rng = np.random.default_rng(704)
    coeff = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    # t=0 is the first LAL reverse-FFT sample.  Q(t=0)=2 df sum conj(chi)dW.
    contracted = float(np.real(np.vdot(coeff, q[..., 0])
                               - 0.5 * np.einsum("ai,abij,bj->", np.conj(coeff), u, coeff)))
    direct = direct_log_likelihood(case[0], case[2], case[3], 0.125, coeff)
    assert abs(contracted - direct) < 2e-10 * max(1.0, abs(direct))


def test_detector_specific_weights_cannot_alias(backend):
    case = list(_case(seed=99))
    weights_h = case[3].copy()
    weights_l = case[3][::-1].copy() * np.linspace(0.2, 1.8, case[3].size)
    case_h = tuple(case[:3] + [weights_h])
    case_l = tuple(case[:3] + [weights_l])
    h = tuple(map(to_host, _unpack(_call(backend, case_h))))
    l = tuple(map(to_host, _unpack(_call(backend, case_l))))
    l0 = (q_oracle(case_l[0], case_l[2], weights_l, 0.125, 5, 13),) + uv_oracle(case_l[0], case_l[1], weights_l, 0.125)
    for got, expected in zip(l, l0):
        np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-12)
    assert any(np.max(np.abs(x - y)) > 1e-4 for x, y in zip(h, l))


def test_reused_context_invalidates_data_weights_and_intrinsic_basis(backend):
    # One worker evaluates multiple intrinsic points.  Reusing plans/storage is welcome;
    # reusing any value that depends on data, PSD, or template is a correctness defect.
    context = GPUPrecomputeContext(backend)
    c1 = _case(seed=1)
    c2 = _case(seed=2)
    d1 = context.array(("H1", "data"), c1[2])
    w1 = context.array(("H1", "weights"), c1[3])
    d2 = context.array(("H1", "data"), c2[2])
    w2 = context.array(("H1", "weights"), c2[3])
    np.testing.assert_array_equal(to_host(d2), c2[2])
    np.testing.assert_array_equal(to_host(w2), c2[3])
    assert np.max(np.abs(to_host(d1) - to_host(d2))) > 1e-4
    assert np.max(np.abs(to_host(w1) - to_host(w2))) > 1e-4
    mutable = c1[2].copy()
    old = context.array(("L1", "data"), mutable)
    mutable[0] += 100 + 20j
    new = context.array(("L1", "data"), mutable)
    assert to_host(new)[0] == mutable[0]
    assert to_host(old)[0] != to_host(new)[0]
    # An exact repeat is a hit; a changed grid shape replaces the role's old allocation.
    before = context.stats()
    again = context.array(("L1", "data"), mutable)
    after_hit = context.stats()
    assert after_hit["cache_hits"] == before["cache_hits"] + 1
    assert after_hit["uploads"] == before["uploads"]
    np.testing.assert_array_equal(to_host(again), mutable)
    shorter = mutable[:-2].copy()
    context.array(("L1", "data"), shorter)
    after_grid_change = context.stats()
    assert after_grid_change["uploads"] == before["uploads"] + 1
    # Three roles remain: H1 data/weights and the replaced L1 data entry only once.
    assert after_grid_change["retained_arrays"] == 3


def test_two_intrinsics_in_sequence_match_independent_oracle(backend):
    c1, c2 = _case(seed=120), _case(seed=121)
    _call(backend, c1)
    got = tuple(map(to_host, _unpack(_call(backend, c2))))
    expected = (q_oracle(c2[0], c2[2], c2[3], 0.125, 5, 13),) + \
        uv_oracle(c2[0], c2[1], c2[3], 0.125)
    for actual, oracle in zip(got, expected):
        np.testing.assert_allclose(actual, oracle, rtol=3e-12, atol=3e-12)


def test_shape_and_grid_contracts_fail_fast(backend):
    case = _case()
    bad = list(case)
    bad[1] = bad[1][..., :-1]
    with pytest.raises((AssertionError, ValueError)):
        _call(backend, tuple(bad))
    with pytest.raises((AssertionError, ValueError)):
        _call(backend, case, delta_t=0.12345)
    with pytest.raises((AssertionError, ValueError)):
        _call(backend, case, n_window=33)
    bad = list(case)
    bad[3] = bad[3].copy()
    bad[3][2] = -1.0
    with pytest.raises((AssertionError, ValueError)):
        _call(backend, tuple(bad))


def test_return_device_contract(backend):
    q, u, v = _unpack(_call(backend, _case(), return_device=True))
    assert type(q).__module__.split(".")[0] == backend.__name__.split(".")[0]
    assert type(u).__module__.split(".")[0] == backend.__name__.split(".")[0]
    assert type(v).__module__.split(".")[0] == backend.__name__.split(".")[0]
