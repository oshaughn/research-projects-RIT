"""Tests for primitive-first time peak-local composition."""

import inspect

import numpy as np
import pytest
from scipy import special

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from RIFT.likelihood.jax_ile import time_first_peaklocal as TFP


def _log_i0(x):
    x = np.asarray(x, dtype=float)
    return np.log(special.i0e(x)) + np.abs(x)


def _cosine_samples(n, amplitude, harmonic):
    span = n - 1.0
    t = np.arange(n, dtype=float)
    return amplitude * np.cos(harmonic * np.pi * t / span)


def test_distance_and_time_known_integral_uses_fewer_nonlinear_nodes():
    """Distance nodes are lanes; each time integral is exactly an I0 integral."""
    n = 65
    span = n - 1.0
    K = 36.0
    harmonic = 3
    kappa = _cosine_samples(n, K, harmonic).astype(complex)
    rho = 4.0
    x = np.array([0.45, 0.7, 1.0, 1.25])
    logw = np.log(np.array([0.1, 0.25, 0.4, 0.25]))

    got, ok, info = TFP.time_first_distance_peak_local_marginalize(
        jnp.asarray(kappa), rho, jnp.asarray(x), jnp.asarray(logw), 1.0,
        enum_factor=8, fine_factor=32, max_nodes=8192,
        keep_nats=36.0, quadrature_tol_nats=2.0e-6)
    want = special.logsumexp(
        logw - 0.5 * rho * x * x + np.log(span) + _log_i0(K * x))

    assert bool(ok), {k: np.asarray(v) for k, v in info.items()}
    assert abs(float(got) - float(want)) < 2.0e-6
    assert int(info["n_local_hi"]) < int(info["n_dense_hi"])
    assert float(info["tail_margin"]) < -23.0


def test_symmetric_angle_reduction_adversary_reconstructs_before_logsumexp():
    """A nonlinear marginal can be constant on samples and structured between them.

    The two lanes represent symmetry-related angle states with primitive
    correlations ``+A cos(pi t)`` and ``-A cos(pi t)``.  At integer input
    samples their marginalized log integrand is the constant ``log cosh(A)``.
    Interpolating that already-marginalized row therefore converges to the wrong
    constant function.  Reconstructing both primitive lanes first recovers
    ``log cosh(A cos(pi t))`` and the known ``T I0(A)`` integral.
    """
    n, amplitude = 17, 8.0
    base = amplitude * (-1.0) ** np.arange(n)
    lanes = np.stack((base, -base)).astype(complex)
    logw = np.full(2, -np.log(2.0))
    rho = np.zeros(2)

    got, ok, info = TFP.time_first_peak_local_marginalize(
        jnp.asarray(lanes), jnp.asarray(rho), jnp.asarray(logw), 1.0,
        enum_factor=8, fine_factor=32, max_nodes=8192,
        keep_nats=20.0, quadrature_tol_nats=1.0e-7)
    want = np.log(n - 1.0) + float(_log_i0(amplitude))
    wrong = np.log(n - 1.0) + np.log(np.cosh(amplitude))

    assert bool(ok), {k: np.asarray(v) for k, v in info.items()}
    assert abs(float(got) - want) < 1.0e-7
    assert abs(wrong - want) > 1.0

    # Pin the ordering structurally as well as numerically: the evaluator has
    # explicit primitive -> gather -> downstream-reduction stages.
    source = inspect.getsource(TFP._evaluate_cover_at_factor)
    assert source.index("reconstruct_time_primitive") < source.index(
        "_lane_log_integrand")


def test_cell_upper_bound_dominates_a_much_finer_reconstruction():
    """The planner's correctness-bearing output is an upper bound, not a grid max."""
    n = 49
    a = _cosine_samples(n, 13.0, 5)
    b = (_cosine_samples(n, 7.0, 2)
         + _cosine_samples(n, 3.0, 7))
    lanes = np.stack((a, b)).astype(complex)
    rho = jnp.asarray([1.3, 0.7])
    logw = jnp.log(jnp.asarray([0.35, 0.65]))
    enum_factor, truth_factor = 4, 128

    k_enum = TFP.reconstruct_time_primitive(jnp.asarray(lanes), enum_factor)
    m1 = TFP.spectral_time_derivative_bound(jnp.asarray(lanes), 1.0)
    plan = TFP.plan_time_cover(
        k_enum, rho, logw, m1, 1.0 / enum_factor, keep_nats=12.0)
    k_truth = TFP.reconstruct_time_primitive(jnp.asarray(lanes), truth_factor)
    g_truth = np.asarray(TFP._lane_log_integrand(k_truth, rho, logw))

    sub = truth_factor // enum_factor
    upper = np.asarray(plan.cell_log_upper)
    observed = np.array([
        g_truth[i * sub:(i + 1) * sub + 1].max()
        for i in range(upper.size)
    ])
    assert np.all(observed <= upper + 2.0e-11), np.max(observed - upper)


def test_capacity_decline_is_ledgered_and_does_not_silently_widen():
    n = 33
    kappa = _cosine_samples(n, 20.0, 1)[None, :].astype(complex)
    got, ok, info = TFP.time_first_peak_local_marginalize(
        jnp.asarray(kappa), jnp.zeros(1), jnp.zeros(1), 1.0,
        enum_factor=8, fine_factor=32, max_nodes=8,
        keep_nats=30.0)
    assert np.isfinite(float(got))
    assert not bool(ok)
    assert not bool(info["capacity_ok"])
    assert bool(info["decline_capacity"])
    assert bool(info["reconciles"])
    assert sum(bool(info[k]) for k in (
        "decline_nonfinite", "decline_capacity", "decline_quadrature",
        "decline_tail")) == 1
    assert int(info["n_local_hi"]) > 8


def test_fixed_shape_kernel_jits_and_has_finite_gradient():
    n = 33
    shape = _cosine_samples(n, 1.0, 3)

    @jax.jit
    def f(amplitude):
        lanes = (amplitude * jnp.asarray(shape))[None, :].astype(jnp.complex128)
        value, ok, _ = TFP.time_first_peak_local_marginalize(
            lanes, jnp.zeros(1), jnp.zeros(1), 1.0,
            enum_factor=4, fine_factor=16, max_nodes=4096,
            keep_nats=30.0, quadrature_tol_nats=1.0e-5)
        return jnp.where(ok, value, jnp.nan)

    value = f(12.0)
    grad = jax.grad(f)(12.0)
    assert np.all(np.isfinite(np.asarray([value, grad])))


def test_api_rejects_an_already_marginalized_time_row():
    with pytest.raises(ValueError, match="already-marginalized"):
        TFP.time_first_peak_local_marginalize(
            jnp.ones(17), jnp.zeros(1), jnp.zeros(1), 1.0)
