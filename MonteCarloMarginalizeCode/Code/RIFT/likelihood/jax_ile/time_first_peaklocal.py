"""Time-first peak-local marginalization of band-limited JAX primitives.

This module is the deliberately small composition seam missing from the JAX
likelihood.  A caller supplies one *primitive correlation* row for every fixed
state of the axes that will subsequently be marginalized (distance, angle, or
their Cartesian product).  The rows are reconstructed in time before the
nonlinear log-sum-exp over those axes is formed.  There is intentionally no API
that accepts a sampled, already-marginalized ``lnL(t)``: that object is not
band-limited and interpolating it is mathematically the wrong operation.

The implementation is a fixed-shape prototype rather than production wiring.
It provides the two pieces needed to make that wiring safe:

* :func:`plan_time_cover` builds a finite cell cover and an omitted-mass bound
  from reconstructed primitive values plus a true spectral derivative bound;
* :func:`time_first_peak_local_marginalize` evaluates the nonlinear downstream
  marginal only at nodes in that cover and returns ``(value, ok, ledger)``.

``ok`` owns no fallback policy.  A production time adapter should fail closed
to the existing dense primitive reconstruction when it is false.  Keeping that
choice at the call site follows ``DESIGN_peak_local_framework.md`` and prevents
one axis's policy from leaking into another.

Scope
-----
The model norm must be time-independent.  ``kappa_t`` has shape
``(n_lanes, n_support)`` and ``rho_sq`` has shape ``(n_lanes,)``; the latter
shape makes the precondition explicit.  A lane is a fixed downstream
quadrature state with exponent

    q_l(t) = Re[kappa_l(t)] - rho_sq_l / 2.

The marginal integrand is ``sum_l exp(log_weight_l + q_l(t))``.  Consequently
one lane can represent a distance node, an angle node, or one point of their
product.  :func:`time_first_distance_peak_local_marginalize` is a convenience
adapter for the RIFT distance form ``x Re(kappa_unit) - x^2 rho_unit^2 / 2``.

The current reconstruction topology matches the existing JAX terminal path:
an endpoint-nonduplicating even extension, with optional raised-cosine support
guards.  Guard convergence is not certified here; production wiring must apply
the same two-guard comparison as ``core._time_marginalize_reflected_primitive``.

Why the cover bound is valid
----------------------------
For the finite Fourier series defining each reconstructed primitive,

    |kappa'_l(t)| <= M1_l = sum_k |K_lk| |omega_k|.

On an enumeration cell of width ``h``, either endpoint therefore bounds the
whole lane by ``q_l(endpoint) + M1_l h``.  Taking the smaller of the two
endpoint-derived log-sum-exp bounds gives a true upper bound on the downstream
marginal over that cell.  The omitted integral is then bounded by the sum of
``h * exp(cell_upper)`` over cells outside the cover.  The first-derivative
bound is intentionally conservative; a future production adapter can replace
it with the shared Hermite/M4 certificate without changing the plan contract.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import _upsample_bandlimited


__all__ = [
    "TimeCoverPlan",
    "reconstruct_time_primitive",
    "spectral_time_derivative_bound",
    "plan_time_cover",
    "time_first_peak_local_marginalize",
    "time_first_distance_peak_local_marginalize",
]


class TimeCoverPlan(NamedTuple):
    """Fixed-shape result of the time-axis planner.

    ``live_cells`` identifies complete enumeration cells included in the local
    quadrature.  ``cell_log_upper`` is a certified supremum bound for every
    cell, not a sampled maximum.  ``outside_log_bound`` bounds the integral over
    all cells not in the cover.  ``peak_lower`` is the largest reconstructed
    nodal value and is used only for targeting; correctness does not depend on
    it being the continuous maximum.
    """

    live_cells: jax.Array
    cell_log_upper: jax.Array
    outside_log_bound: jax.Array
    peak_lower: jax.Array
    enum_step: jax.Array


def _validate_primitive_shapes(kappa_t, rho_sq, log_lane_weight, guard):
    if kappa_t.ndim != 2:
        raise ValueError(
            "kappa_t must have shape (n_lanes, n_support); an already-"
            "marginalized lnL(t) is deliberately not accepted")
    if rho_sq.ndim != 1 or rho_sq.shape[0] != kappa_t.shape[0]:
        raise ValueError(
            "rho_sq must have shape (n_lanes,), making the time-independent "
            "norm precondition explicit")
    if log_lane_weight.ndim != 1 or log_lane_weight.shape[0] != kappa_t.shape[0]:
        raise ValueError("log_lane_weight must have shape (n_lanes,)")
    if kappa_t.shape[-1] - 2 * guard < 2:
        raise ValueError("guard must leave at least two integration samples")


def _tapered_support(kappa_t, guard):
    """Move the artificial reflection seam through support-only tapering."""
    guard = int(guard)
    if guard == 0:
        return kappa_t
    n_keep = kappa_t.shape[-1] - 2 * guard
    u = jnp.arange(guard + 1, dtype=jnp.float64) / float(guard)
    ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * u))
    taper = jnp.concatenate(
        (ramp[:-1], jnp.ones((n_keep,), dtype=jnp.float64),
         jnp.flip(ramp[:-1])))
    return kappa_t * taper[None, :]


def _reflected_series(kappa_t, guard):
    supported = _tapered_support(kappa_t, guard)
    return jnp.concatenate(
        (supported, jnp.flip(supported[..., 1:-1], axis=-1)), axis=-1)


def reconstruct_time_primitive(kappa_t, factor, guard=0):
    """Reconstruct raw complex correlations on a uniformly refined time grid.

    The returned interval contains the original unguarded closed window only;
    guard samples influence the Fourier reconstruction but are never integrated.
    This is the primitive operation that must precede every distance/angle
    reduction in this module.
    """
    factor = int(factor)
    guard = int(guard)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    kappa_t = jnp.asarray(kappa_t, dtype=jnp.complex128)
    if kappa_t.ndim != 2:
        raise ValueError("kappa_t must have shape (n_lanes, n_support)")
    n_keep = kappa_t.shape[-1] - 2 * guard
    if n_keep < 2:
        raise ValueError("guard must leave at least two integration samples")

    reflected = _reflected_series(kappa_t, guard)
    dense = _upsample_bandlimited(reflected, factor, axis=-1)
    # The forward half of the endpoint-nonduplicating reflection has
    # (n_support - 1) * factor + 1 points.  Crop support after refinement so
    # both integration endpoints remain exact input samples.
    forward = dense[..., :(kappa_t.shape[-1] - 1) * factor + 1]
    start = guard * factor
    return forward[..., start:start + (n_keep - 1) * factor + 1]


def spectral_time_derivative_bound(kappa_t, delta_t, guard=0, order=1):
    """True per-lane bound on ``|d^order kappa/dt^order|``.

    The coefficients are those of the exact reflected finite Fourier series
    used by :func:`reconstruct_time_primitive`.  This is a triangle-inequality
    bound, never a fit to samples.
    """
    guard = int(guard)
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")
    if not (float(delta_t) > 0.0):
        raise ValueError("delta_t must be positive")
    kappa_t = jnp.asarray(kappa_t, dtype=jnp.complex128)
    if kappa_t.ndim != 2:
        raise ValueError("kappa_t must have shape (n_lanes, n_support)")
    if kappa_t.shape[-1] - 2 * guard < 2:
        raise ValueError("guard must leave at least two integration samples")
    series = _reflected_series(kappa_t, guard)
    n = series.shape[-1]
    coeff = jnp.fft.fft(series, axis=-1) / float(n)
    omega = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=float(delta_t))
    return jnp.sum(jnp.abs(coeff) * (jnp.abs(omega)[None, :] ** order), axis=-1)


def _lane_log_integrand(kappa, rho_sq, log_lane_weight):
    """Nonlinear downstream marginal, evaluated only after reconstruction."""
    exponent = kappa.real - 0.5 * rho_sq[:, None]
    return jax.scipy.special.logsumexp(
        exponent + log_lane_weight[:, None], axis=0)


def plan_time_cover(kappa_enum, rho_sq, log_lane_weight, derivative_bound,
                    enum_step, keep_nats=40.0):
    """Plan complete time cells and certify the mass outside their union.

    ``kappa_enum`` must already be a reconstruction of the primitive.  The API
    accepts no marginalized time series.  ``derivative_bound[l]`` must be a true
    bound on ``|kappa'_l|``; use :func:`spectral_time_derivative_bound`.
    """
    kappa_enum = jnp.asarray(kappa_enum, dtype=jnp.complex128)
    rho_sq = jnp.asarray(rho_sq, dtype=jnp.float64)
    log_lane_weight = jnp.asarray(log_lane_weight, dtype=jnp.float64)
    derivative_bound = jnp.asarray(derivative_bound, dtype=jnp.float64)
    if kappa_enum.ndim != 2 or kappa_enum.shape[-1] < 2:
        raise ValueError("kappa_enum must have shape (n_lanes, n_enum >= 2)")
    n_lane = kappa_enum.shape[0]
    for name, value in (("rho_sq", rho_sq),
                        ("log_lane_weight", log_lane_weight),
                        ("derivative_bound", derivative_bound)):
        if value.ndim != 1 or value.shape[0] != n_lane:
            raise ValueError("%s must have shape (n_lanes,)" % name)
    if not (float(enum_step) > 0.0):
        raise ValueError("enum_step must be positive")
    if not (float(keep_nats) > 0.0):
        raise ValueError("keep_nats must be positive")

    node_log = _lane_log_integrand(kappa_enum, rho_sq, log_lane_weight)
    peak_lower = jnp.max(node_log)
    q = kappa_enum.real - 0.5 * rho_sq[:, None]
    lift = derivative_bound[:, None] * float(enum_step)

    # Each endpoint-derived expression bounds the ENTIRE cell.  The minimum
    # of two upper bounds is still an upper bound and is often much tighter.
    left_upper = jax.scipy.special.logsumexp(
        q[:, :-1] + lift + log_lane_weight[:, None], axis=0)
    right_upper = jax.scipy.special.logsumexp(
        q[:, 1:] + lift + log_lane_weight[:, None], axis=0)
    cell_upper = jnp.minimum(left_upper, right_upper)

    # Target from the reconstructed nodes, certify from cell_upper.  Selection
    # is intentionally stopped: changing which cells belong to a cover is a
    # discrete planner decision, not a differentiable likelihood operation.
    live = jax.lax.stop_gradient(cell_upper >= peak_lower - float(keep_nats))
    omitted = jnp.where(
        live, -jnp.inf, cell_upper + jnp.log(float(enum_step)))
    outside = jax.scipy.special.logsumexp(omitted)
    return TimeCoverPlan(live, cell_upper, outside, peak_lower,
                         jnp.asarray(enum_step, dtype=jnp.float64))


def _node_weights(live_cells, fine_factor, enum_factor, delta_t):
    """Composite-trapezoid weights for a union of complete enum cells."""
    sub = int(fine_factor) // int(enum_factor)
    fine_cells = jnp.repeat(live_cells, sub)
    h = float(delta_t) / float(fine_factor)
    # Every live fine cell contributes h/2 at each end.  Adjacent cells
    # therefore give their shared point weight h, without double counting.
    middle = 0.5 * h * (fine_cells[:-1].astype(jnp.float64)
                        + fine_cells[1:].astype(jnp.float64))
    return jnp.concatenate(
        (jnp.asarray([0.5 * h * fine_cells[0]], dtype=jnp.float64),
         middle,
         jnp.asarray([0.5 * h * fine_cells[-1]], dtype=jnp.float64)))


def _evaluate_cover_at_factor(kappa_t, rho_sq, log_lane_weight, plan,
                              delta_t, enum_factor, factor, guard, max_nodes):
    weights = jax.lax.stop_gradient(
        _node_weights(plan.live_cells, factor, enum_factor, delta_t))
    n_local = jnp.count_nonzero(weights > 0.0)
    capacity_ok = n_local <= int(max_nodes)
    index = jnp.nonzero(weights > 0.0, size=int(max_nodes), fill_value=0)[0]
    slot_live = jnp.arange(int(max_nodes)) < n_local
    index = jax.lax.stop_gradient(index)
    slot_live = jax.lax.stop_gradient(slot_live)

    # Reconstruct FIRST, gather SECOND, marginalize other axes LAST.  Keeping
    # these as three explicit operations is the load-bearing ordering contract.
    primitive_fine = reconstruct_time_primitive(kappa_t, factor, guard=guard)
    primitive_local = primitive_fine[:, index]
    log_t = _lane_log_integrand(primitive_local, rho_sq, log_lane_weight)
    local_weight = jnp.where(slot_live, weights[index], 1.0)
    terms = jnp.where(slot_live, log_t + jnp.log(local_weight), -jnp.inf)
    return jax.scipy.special.logsumexp(terms), n_local, capacity_ok, weights.shape[0]


def time_first_peak_local_marginalize(
        kappa_t, rho_sq, log_lane_weight, delta_t, *, guard=0,
        enum_factor=8, fine_factor=32, max_nodes=8192,
        keep_nats=40.0, tail_tol_nats=-23.0, quadrature_tol_nats=1.0e-5):
    """Peak-local joint marginal with time applied to primitives first.

    Parameters other than the three lane arrays are planner policy and are
    expected to be static under :func:`jax.jit`.  ``fine_factor`` is checked
    against ``2*fine_factor``; the latter value is returned.  The cover is
    planned once on ``enum_factor`` and reused by both quadratures.

    Returns
    -------
    value : scalar
        Local-cover integral at ``2*fine_factor``.  It is diagnostic only when
        ``ok`` is false.
    ok : bool scalar
        True iff the node capacity, local quadrature convergence, finite-input
        check, and certified omitted-mass threshold all pass.
    ledger : dict of JAX scalars
        Named diagnostics.  A caller owns the fail-closed fallback.
    """
    guard = int(guard)
    enum_factor = int(enum_factor)
    fine_factor = int(fine_factor)
    max_nodes = int(max_nodes)
    if enum_factor < 1:
        raise ValueError("enum_factor must be >= 1")
    if fine_factor < enum_factor or fine_factor % enum_factor:
        raise ValueError("fine_factor must be a multiple of enum_factor")
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least 2")
    if not (float(delta_t) > 0.0):
        raise ValueError("delta_t must be positive")

    kappa_t = jnp.asarray(kappa_t, dtype=jnp.complex128)
    rho_sq = jnp.asarray(rho_sq, dtype=jnp.float64)
    log_lane_weight = jnp.asarray(log_lane_weight, dtype=jnp.float64)
    _validate_primitive_shapes(kappa_t, rho_sq, log_lane_weight, guard)

    derivative_bound = spectral_time_derivative_bound(
        kappa_t, delta_t, guard=guard, order=1)
    kappa_enum = reconstruct_time_primitive(
        kappa_t, enum_factor, guard=guard)
    plan = plan_time_cover(
        kappa_enum, rho_sq, log_lane_weight, derivative_bound,
        float(delta_t) / enum_factor, keep_nats=keep_nats)

    value_lo, n_lo, cap_lo, dense_lo = _evaluate_cover_at_factor(
        kappa_t, rho_sq, log_lane_weight, plan, delta_t, enum_factor,
        fine_factor, guard, max_nodes)
    value_hi, n_hi, cap_hi, dense_hi = _evaluate_cover_at_factor(
        kappa_t, rho_sq, log_lane_weight, plan, delta_t, enum_factor,
        2 * fine_factor, guard, max_nodes)

    quadrature_error = jnp.abs(value_hi - value_lo)
    tail_margin = plan.outside_log_bound - value_hi
    finite_inputs = (jnp.all(jnp.isfinite(kappa_t.real))
                     & jnp.all(jnp.isfinite(kappa_t.imag))
                     & jnp.all(jnp.isfinite(rho_sq))
                     & jnp.all(jnp.isfinite(derivative_bound))
                     & jnp.all(jnp.isfinite(log_lane_weight)
                               | jnp.isneginf(log_lane_weight)))
    capacity_ok = cap_lo & cap_hi
    quadrature_ok = quadrature_error <= float(quadrature_tol_nats)
    tail_ok = tail_margin < float(tail_tol_nats)
    # Priority makes the decline reasons disjoint.  A caller can therefore
    # reconcile one and only one terminal state without interpreting a set of
    # overlapping diagnostic predicates.
    decline_nonfinite = ~finite_inputs
    decline_capacity = finite_inputs & (~capacity_ok)
    decline_quadrature = finite_inputs & capacity_ok & (~quadrature_ok)
    decline_tail = finite_inputs & capacity_ok & quadrature_ok & (~tail_ok)
    ok = finite_inputs & capacity_ok & quadrature_ok & tail_ok
    reconciles = (ok.astype(jnp.int32)
                  + decline_nonfinite.astype(jnp.int32)
                  + decline_capacity.astype(jnp.int32)
                  + decline_quadrature.astype(jnp.int32)
                  + decline_tail.astype(jnp.int32)) == 1

    ledger = {
        "accepted": ok,
        "decline_nonfinite": decline_nonfinite,
        "decline_capacity": decline_capacity,
        "decline_quadrature": decline_quadrature,
        "decline_tail": decline_tail,
        "reconciles": reconciles,
        "capacity_ok": capacity_ok,
        "quadrature_ok": quadrature_ok,
        "tail_ok": tail_ok,
        "finite_inputs": finite_inputs,
        "quadrature_error": quadrature_error,
        "tail_margin": tail_margin,
        "outside_log_bound": plan.outside_log_bound,
        "peak_lower": plan.peak_lower,
        "n_live_cells": jnp.count_nonzero(plan.live_cells),
        "n_cells": jnp.asarray(plan.live_cells.size),
        "n_local_lo": n_lo,
        "n_local_hi": n_hi,
        "n_dense_lo": jnp.asarray(dense_lo),
        "n_dense_hi": jnp.asarray(dense_hi),
    }
    return value_hi, ok, ledger


def time_first_distance_peak_local_marginalize(
        kappa_unit_t, rho_sq_unit, x_grid, log_weight, delta_t, **kwargs):
    """Distance adapter for :func:`time_first_peak_local_marginalize`.

    ``kappa_unit_t`` is the raw unit-distance complex correlation, including
    optional support guards.  Distance scaling is applied lane-by-lane *before*
    reconstruction; linearity then makes reconstructing the scaled lanes
    identical to scaling the reconstructed primitive.  The nonlinear distance
    log-sum-exp is formed only after reconstruction at each requested time.

    This helper handles one outer sample.  Batch it with :func:`jax.vmap`.
    """
    kappa_unit_t = jnp.asarray(kappa_unit_t, dtype=jnp.complex128)
    if kappa_unit_t.ndim != 1:
        raise ValueError("kappa_unit_t must have shape (n_support,); use vmap for batches")
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64).ravel()
    log_weight = jnp.asarray(log_weight, dtype=jnp.float64).ravel()
    if x_grid.shape != log_weight.shape:
        raise ValueError("x_grid and log_weight must have identical shape")
    rho_sq_unit = jnp.asarray(rho_sq_unit, dtype=jnp.float64)
    if rho_sq_unit.ndim != 0:
        raise ValueError("rho_sq_unit must be a scalar (time-independent norm)")
    kappa_lanes = x_grid[:, None] * kappa_unit_t[None, :]
    rho_lanes = jnp.square(x_grid) * rho_sq_unit
    return time_first_peak_local_marginalize(
        kappa_lanes, rho_lanes, log_weight, delta_t, **kwargs)
