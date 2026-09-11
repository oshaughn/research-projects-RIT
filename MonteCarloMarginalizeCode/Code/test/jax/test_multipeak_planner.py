"""Load-bearing tests for the U,V/Q multi-peak diagnostic planner."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from RIFT.likelihood.jax_ile import multipeak_planner as planner  # noqa: E402


def _synthetic_tables(n_time=9):
    """Small reflected-polynomial problem with an interior four-axis peak."""
    time = np.arange(n_time, dtype=float)
    C_A = np.zeros((3, 3, n_time), dtype=np.complex128)
    C_B = np.zeros((5, 5), dtype=np.complex128)
    # DCT-compatible time dependence, with its interior maximum at t=4.
    C_A[0, 1] = 20.0 - 2.0 * np.cos(2.0 * np.pi * time / (n_time - 1))
    C_A[2, 0] = 0.25
    C_A[2, 2] = 0.25
    C_B[0, 2] = 4.0
    C_B[2, 1] = 0.02
    C_B[2, 3] = 0.02
    return C_A, C_B


def test_uv_summary_rejects_time_dependent_norm():
    _, C_B = _synthetic_tables()
    repeated = np.repeat(C_B[..., None], 7, axis=-1)
    summary = planner.summarize_uv_norm_table(repeated)
    assert summary.time_invariant
    repeated[1, 1, 3] += 1.0e-3
    changed = planner.summarize_uv_norm_table(repeated)
    assert not changed.time_invariant
    C_A, _ = _synthetic_tables()
    with pytest.raises(ValueError, match="arrival-time-dependent"):
        planner.rank_joint_starts_from_uvq(C_A, changed, 1.0, 8.0)


def test_exact_symmetry_expansion_is_scale_invariant():
    C_A, C_B = _synthetic_tables()
    summary = planner.summarize_uv_norm_table(C_B)
    first = planner.rank_joint_starts_from_uvq(
        C_A, summary, 1.0, 8.0, max_time_starts=2, max_starts=24)

    assert first.symmetry.certified
    assert first.symmetry.group_order == 4
    np.testing.assert_allclose(
        first.symmetry.shifts,
        [[0.0, 0.0], [0.5 * np.pi, np.pi],
         [np.pi, 0.0], [1.5 * np.pi, np.pi]], atol=1.0e-13)
    assert len(first.starts) == first.symmetry.group_order * len(
        first.raw_starts)
    for raw_index in range(len(first.raw_starts)):
        actions = first.group_action[
            raw_index * first.symmetry.group_order:
            (raw_index + 1) * first.symmetry.group_order]
        np.testing.assert_array_equal(actions, np.arange(4))

    # A -> s A, B -> s^2 B, x -> x/s preserves every extrema location in
    # (time, phi, u) and changes the exponent only by the constant 4 log(s).
    scale = 4.0
    scaled = planner.rank_joint_starts_from_uvq(
        scale * C_A, planner.summarize_uv_norm_table(scale * scale * C_B),
        1.0 / scale, 8.0 / scale, max_time_starts=2, max_starts=24)
    assert len(scaled.raw_starts) == len(first.raw_starts)
    assert len(scaled.starts) == len(first.starts)
    np.testing.assert_allclose(scaled.starts[:, :3], first.starts[:, :3],
                               atol=1.0e-13)
    np.testing.assert_allclose(scaled.starts[:, 3], first.starts[:, 3] / scale,
                               rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        scaled.scores - first.scores, 4.0 * np.log(scale), atol=1.0e-12)


def test_symmetry_orbits_are_reduced_before_representative_capacity():
    """A louder orbit's four copies must not evict a lower distinct orbit."""
    n_time = 9
    time = np.arange(n_time, dtype=float)
    C_A = np.zeros((5, 3, n_time), dtype=np.complex128)
    C_B = np.zeros((5, 5), dtype=np.complex128)
    C_A[0, 1] = 20.0 - 2.0 * np.cos(
        2.0 * np.pi * time / (n_time - 1))
    C_A[2, 0] = 0.25
    C_A[2, 2] = 0.25
    C_A[4, 1] = -0.25 + 0.05j
    C_B[0, 2] = 4.0
    C_B[2, 1] = 0.02
    C_B[2, 3] = 0.02
    portfolio = planner.rank_joint_starts_from_uvq(
        C_A, planner.summarize_uv_norm_table(C_B), 1.0, 8.0,
        max_time_starts=1, max_starts=8)
    assert portfolio.symmetry.group_order == 4
    assert len(portfolio.raw_starts) == 2
    assert len(portfolio.starts) == 8
    assert not portfolio.capacity_truncated
    assert portfolio.raw_scores[1] < portfolio.raw_scores[0]
    np.testing.assert_array_equal(
        portfolio.group_action, np.tile(np.arange(4), 2))


def test_ranked_starts_optimize_distance_at_each_angular_candidate():
    C_A, C_B = _synthetic_tables()
    # A strong norm harmonic makes the analytic distance optimum follow angle.
    C_B[2, 1] = 0.35
    C_B[2, 3] = 0.35
    portfolio = planner.rank_joint_starts_from_uvq(
        C_A, planner.summarize_uv_norm_table(C_B), 1.0, 8.0,
        max_time_starts=2, max_starts=24)
    phi, u, A = planner._harmonic_lattice(
        C_A, portfolio.n_phi_lattice, portfolio.n_u_lattice)
    _, _, B = planner._harmonic_lattice(
        C_B, portfolio.n_phi_lattice, portfolio.n_u_lattice)
    for start in portfolio.raw_starts:
        it = int(start[0])
        iphi = int(np.argmin(np.abs(phi - start[1])))
        iu = int(np.argmin(np.abs(u - start[2])))
        _, expected_x = planner._distance_profile(
            np.asarray(A[iphi, iu, it]), np.asarray(B[iphi, iu]), 1.0, 8.0)
        assert start[3] == pytest.approx(float(expected_x), abs=1.0e-13)


def test_time_endpoints_require_one_sided_maximum():
    C_A, C_B = _synthetic_tables()
    C_A[0, 1] = np.linspace(12.0, 20.0, C_A.shape[-1])
    portfolio = planner.rank_joint_starts_from_uvq(
        C_A, planner.summarize_uv_norm_table(C_B), 1.0, 8.0,
        max_time_starts=3, max_starts=24)
    assert 0 not in portfolio.time_starts
    assert portfolio.time_starts.tolist() == [C_A.shape[-1] - 1]


def test_jax_refiner_reaches_strict_stationary_maximum():
    C_A, C_B = _synthetic_tables()
    starts = np.asarray([[3.3, 0.2, 0.2, 5.0],
                         [4.7, 3.0, 0.1, 5.0]])
    result = tuple(np.asarray(item) for item in planner.refine_joint_starts_jax(
        C_A, C_B, starts, 1.0, 8.0, iterations=18))
    points, values, gradients, hessians, curvatures = result
    selected, stationary = planner.select_refined_modes(
        points, values, gradients, curvatures, max_modes=2)
    assert stationary.any()
    assert len(selected) >= 1
    assert np.max(np.linalg.norm(gradients[selected], axis=1)) < 2.0e-6
    assert np.all(curvatures[selected] > 0.0)
    assert np.all(np.isfinite(hessians[selected]))


def test_two_tier_local_integral_accepts_or_returns_finite_reserve():
    C_A, C_B = _synthetic_tables()
    calls = []

    def reserve():
        calls.append("called")
        return 123.456

    accepted = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, reserve, log_integral_tol=0.1,
        tier0=(2, 2, 24), tier1=(3, 3, 48), quadrature_order=7,
        cell_sigma=4.0, chunk_size=32)
    assert accepted.accepted
    assert not accepted.used_reserve
    assert accepted.provenance == "uvq-multipeak-tier1"
    assert np.isfinite(accepted.value)
    assert accepted.delta_log_integral < 0.1
    assert calls == []

    declined = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, reserve, log_integral_tol=1.0e-12,
        tier0=(2, 2, 24), tier1=(3, 3, 48), quadrature_order=5,
        cell_sigma=4.0, chunk_size=32)
    assert not declined.accepted
    assert declined.used_reserve
    assert declined.value == 123.456
    assert declined.provenance.startswith("dense-reserve:")
    assert calls == ["called"]

    bad_calls = []

    def failing_reserve():
        bad_calls.append("called")
        raise ValueError("deliberate reserve failure")

    with pytest.raises(planner._DenseReserveError):
        planner.multipeak_local_marginalize(
            C_A, C_B, 1.0, 8.0, failing_reserve,
            log_integral_tol=1.0e-12, tier0=(2, 2, 24),
            tier1=(3, 3, 48), quadrature_order=5,
            cell_sigma=4.0, chunk_size=32)
    assert bad_calls == ["called"]


def test_affine_cell_overlap_is_partitioned_not_rejected():
    C_A, C_B = _synthetic_tables()
    refined = tuple(np.asarray(item) for item in
                    planner.refine_joint_starts_jax(
                        C_A, C_B, np.asarray([[4.0, 0.0, 0.0, 5.0]]),
                        1.0, 8.0, iterations=18))
    points, values, _, hessians, _ = refined
    one = planner.integrate_refined_modes_tensor(
        C_A, C_B, points, values, hessians, 1.0, 8.0,
        log_integral_tol=0.1, cell_sigma=4.0, quadrature_order=7,
        chunk_size=32)
    duplicate = planner.integrate_refined_modes_tensor(
        C_A, C_B, np.repeat(points, 2, axis=0),
        np.repeat(values, 2), np.repeat(hessians, 2, axis=0), 1.0, 8.0,
        log_integral_tol=0.1, cell_sigma=4.0, quadrature_order=7,
        chunk_size=32)
    assert duplicate.min_core_separation == pytest.approx(0.0)
    assert duplicate.overlap_ok
    assert duplicate.ok == one.ok
    assert duplicate.value == pytest.approx(one.value, abs=2.0e-12)


def _log_density_dense(C_A, C_B, theta):
    enclosure = planner._time_fourier_enclosure(C_A)
    C_t = planner._evaluate_spectrum_numpy(
        enclosure[0], enclosure[1], theta[0])
    A, _ = planner._field_variation(C_t, theta[1], theta[2], 0.0, 0.0)
    B, _ = planner._field_variation(C_B, theta[1], theta[2], 0.0, 0.0)
    x = theta[3]
    return x * A - 0.5 * x * x * B - 4.0 * np.log(x)


def test_local_fourier_box_bound_dominates_dense_points():
    C_A, C_B = _synthetic_tables()
    enclosure = planner._time_fourier_enclosure(C_A)
    lo = np.asarray([3.25, 0.0, 0.0, 3.0])
    hi = np.asarray([4.75, 0.6, 0.7, 6.0])
    log_integral_upper, _, point_upper = planner._box_log_upper(
        C_A, C_B, enclosure, lo, hi)
    rng = np.random.default_rng(1729)
    points = rng.uniform(lo, hi, size=(20000, 4))
    dense = np.asarray([_log_density_dense(C_A, C_B, p) for p in points])
    assert np.max(dense) <= point_upper + 2.0e-11
    assert log_integral_upper == pytest.approx(
        point_upper + np.log(np.prod(hi - lo)), abs=1.0e-13)


def test_overlap_is_owned_once_in_exact_axis_box_geometry():
    C_A, C_B = _synthetic_tables()
    summary = planner.summarize_uv_norm_table(C_B)
    center = np.asarray([[4.0, np.pi, np.pi, 4.5],
                         [4.0, np.pi, np.pi, 4.5]])
    half = np.asarray([[4.0, np.pi, np.pi, 3.5],
                       [4.0, np.pi, np.pi, 3.5]])
    report = planner.hierarchical_union_cover(
        C_A, summary, center, half, 1.0, 8.0,
        target_log_value=0.0, max_boxes=10)
    assert report.bound_certified
    assert report.budget_met
    assert report.n_owned_leaves == 1
    assert report.n_overlap_owned == 1
    assert report.n_outside_leaves == 0
    assert report.owned_mode.tolist() == [0]
    np.testing.assert_allclose(report.owned_centers, center[:1])
    np.testing.assert_allclose(report.owned_half_widths, half[:1])


def test_cover_cap_is_a_decline_not_a_failure_or_false_certificate():
    C_A, C_B = _synthetic_tables()
    summary = planner.summarize_uv_norm_table(C_B)
    report = planner.hierarchical_union_cover(
        C_A, summary, np.asarray([[4.0, 0.0, 0.0, 5.0]]),
        np.asarray([[0.1, 0.1, 0.1, 0.1]]), 1.0, 8.0,
        target_log_value=1.0e6, outside_tol_nats=-23.0, max_boxes=1)
    assert report.bound_certified
    assert report.budget_met  # A huge supplied target makes the comparison pass.
    assert not report.cap_reached  # No refinement was needed.

    declined = planner.hierarchical_union_cover(
        C_A, summary, np.asarray([[4.0, 0.0, 0.0, 5.0]]),
        np.asarray([[0.1, 0.1, 0.1, 0.1]]), 1.0, 8.0,
        target_log_value=-1.0e6, outside_tol_nats=-23.0, max_boxes=1)
    assert declined.bound_certified
    assert not declined.budget_met
    assert declined.cap_reached
    assert declined.n_outside_leaves == 1


# ---------------------------------------------------------------------------
# Refinement stall.  The 2026-09-07 ladder campaign declined EVERY row of every
# rung with "planner-exception: RuntimeError".  The cause was not degenerate
# starts and not the tier1 configuration: the bounded Newton loop had a hard
# FIXED POINT, and the single element responsible was the per-coordinate
# ``clip`` that bounded the step.
#
# The modified-Newton direction always ascends, because every ``safe_i`` in
# ``(v_i . g) / safe_i`` is positive.  Clipping each coordinate independently
# rescales them by DIFFERENT factors and does not preserve that.  At an
# indefinite Hessian the raw step is about 1e8 gradients long, so every
# coordinate saturates and only the signs survive.  Measured on the campaign's
# own rung-160 table, row 0 tier1:
#
#   eig(-H) = [-1.157e+01, 2.629e+02, 1.924e+03, 1.325e+05]
#   g       = [-54.47, +51.02, +28.92, 0]
#   clipped = [ +2.00,  -0.50,  +0.50, +0.25]      g . d = -119.9
#   lanes   = [4878.19, 11062.76, 12679.19, 13091.06, 13237.70]  <- zero wins
#
# A descent direction has no improving lane, so the value-only search took its
# zero lane, the iterate was bit-identical next step, and all eighteen steps
# ran without moving.  Scaling by ONE factor keeps every ratio and so keeps the
# sign of g . d.  all_axis_peaklocal.py, the independent four-axis
# implementation merged in #268, bounds its own Newton step the same way and
# for the same stated reason.
#
# Reporting the resulting decline under its own name is PR #277's subject, not
# this one's.
#
# Both tiers stalled, on different rows of the campaign, so the four gradient
# norms agreeing to ten digits was the certified order-4 symmetry orbit doing
# its job, not a degeneracy.  Zero spread WITHIN an orbit is correct; a test
# asserting non-zero spread would fail on healthy rows.
#
# Tried and rejected, both measured against the real rung-160 table and neither
# shipped: taking |lambda| instead of flooring at ridge (the ablation shows the
# tier converges identically with and without it), and widening the
# backtracking ladder from 1/8 to 2**-15 (converged 16/64 with, 18/64 without,
# from the same random starts).
# ---------------------------------------------------------------------------

_STALLED_SPECTRUM = (-1.157203234765e+01, 2.629300619639e+02,
                     1.924365666846e+03, 1.324806698996e+05)
_STALLED_GRADIENT = (-5.446512310805e+01, 5.102215811225e+01,
                     2.892359149328e+01, -1.818989403546e-12)
_MAX_STEP = (2.0, 0.5, 0.5, 0.25)


def _narrow_time_peak_tables(amp, n_time=65, width=0.35, centre=32.37):
    """Fixture with the production GEOMETRY, not just its amplitude.

    ``_synthetic_tables`` puts its maximum exactly on the targeting lattice
    and is nearly isotropic, so the Newton loop is barely exercised there and
    the stall is invisible.  What makes the production problem hard is that the
    arrival-time peak is much narrower than one sample -- the campaign measured
    deltaT/sigma_t between 4 and 65 -- which is what makes a fixed absolute
    step bound span many peak widths.  The centre is deliberately off-grid.
    """
    time = np.arange(n_time, dtype=float)
    bump = np.exp(-0.5 * ((time - centre) / width) ** 2)
    C_A = np.zeros((3, 3, n_time), dtype=np.complex128)
    C_B = np.zeros((5, 5), dtype=np.complex128)
    C_A[0, 1] = amp * (2.0 + 18.0 * bump)
    C_A[2, 0] = 0.25 * amp * bump
    C_A[2, 2] = 0.25 * amp * bump
    # Scaling C_B with C_A holds the distance optimum inside [x_min, x_max]:
    # x* ~ A/B.  Scaling C_A alone would drive x* through x_max and the residual
    # gradient would then be a boundary artefact rather than a stall.
    C_B[0, 2] = 4.0 * amp
    C_B[2, 1] = 0.02 * amp
    C_B[2, 3] = 0.02 * amp
    return C_A, C_B


def test_bounded_ascent_direction_ascends_and_respects_its_bound():
    """The BOUNDED step must still increase lnL to first order.

    ``g . d > 0`` is what makes a backtracking search able to succeed at all.
    The unbounded direction has it by construction; the old per-coordinate clip
    destroyed it, giving ``g . d = -119.9`` on the campaign's own stalled row.
    """
    max_step = jnp.asarray(_MAX_STEP, dtype=jnp.float64)
    gradient = jnp.asarray(_STALLED_GRADIENT, dtype=jnp.float64)
    rng = np.random.default_rng(20260907)
    checked = 0
    for trial in range(40):
        # A random orthonormal frame carrying the measured spectrum, plus
        # spectra with more than one negative eigenvalue.  The invariant is a
        # property of the construction, not of one matrix.
        basis, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        if trial < 20:
            spectrum = np.asarray(_STALLED_SPECTRUM)
        else:
            spectrum = rng.normal(size=4) * 10.0 ** rng.uniform(-1, 4, size=4)
        hessian = jnp.asarray(-(basis * spectrum) @ basis.T,
                              dtype=jnp.float64)
        this_gradient = (gradient if trial < 20 else
                         jnp.asarray(rng.normal(size=4) * 50.0))
        direction, _ = planner._bounded_ascent_direction(
            this_gradient, hessian, 1.0e-8, max_step)
        assert np.all(np.isfinite(np.asarray(direction)))
        assert np.all(np.abs(np.asarray(direction))
                      <= np.asarray(_MAX_STEP) * (1.0 + 1.0e-12))
        assert float(jnp.dot(this_gradient, direction)) > 0.0
        checked += 1
    assert checked == 40
    # A zero gradient gives a zero step, so a converged point stays put.  This
    # is not a NaN guard: max_step is validated positive, so the rescale is
    # finite for a zero direction with or without the tiny floor.
    at_rest, _ = planner._bounded_ascent_direction(
        jnp.zeros(4), jnp.asarray(-np.eye(4)), 1.0e-8, max_step)
    assert np.allclose(np.asarray(at_rest), 0.0)


def test_refinement_outcome_does_not_collapse_as_the_step_bound_widens():
    """``max_step`` is a safeguard; it must not decide whether the loop works.

    The bound is fixed in absolute coordinates while the peak width scales as
    1/rho, so widening it is the CI-affordable stand-in for raising amplitude:
    production runs at 8 to 130 arrival-sample widths per unit of the time
    bound, and a synthetic table narrow enough to reach that at the default
    bound has no interior maximum left to find.  Measured across this sweep,
    frozen interior starts and converged modes:

        old            0, 3, 13 frozen;  4, 2, 0 converged
        this commit    0, 0,  5 frozen;  3, 4, 3 converged

    The five at the widest bound were not converging under either version, so
    the assertion below is on the two bounds where standing still is a defect,
    plus the convergence count across all three.
    """
    C_A, C_B = _narrow_time_peak_tables(256.0)
    rng = np.random.default_rng(31)
    starts = np.column_stack([
        rng.uniform(28.0, 37.0, 24), rng.uniform(0.0, 2.0 * np.pi, 24),
        rng.uniform(0.0, 2.0 * np.pi, 24), rng.uniform(1.0, 8.0, 24)])
    converged, frozen_counts = [], []
    for bound in (_MAX_STEP, (8.0, 2.0, 2.0, 1.0), (32.0, 8.0, 8.0, 4.0)):
        refined = tuple(np.asarray(item) for item in
                        planner.refine_joint_starts_jax(
                            C_A, C_B, starts, 1.0, 8.0, iterations=18,
                            max_step=bound))
        points, values, gradients, hessians, curvatures = refined
        interior = ((points[:, 0] > 0.0)
                    & (points[:, 0] < C_A.shape[-1] - 1.0)
                    & (points[:, 3] > 1.0) & (points[:, 3] < 8.0))
        frozen_counts.append(
            int((np.all(points == starts, axis=1) & interior).sum()))
        selected, _ = planner.select_refined_modes(
            points, values, gradients, curvatures, max_modes=len(starts))
        converged.append(len(selected))
    assert frozen_counts[0] == 0 and frozen_counts[1] == 0, (
        "interior starts frozen in place at the production and 4x bounds: %s"
        % frozen_counts)
    # An ABSOLUTE floor.  Normalizing against converged[0] would compare the
    # code under test with itself: [0, 0, 0] satisfies any ratio, and a real
    # loss at the production bound would be absorbed into the baseline rather
    # than caught.  What the old construction did was reduce the loop to
    # finding NOTHING as the bound widened (4, 2, 0); no bound may do that.
    assert min(converged) >= 1, (
        "some step bound left the loop unable to converge any mode at all: %s"
        % converged)


def test_refine_joint_starts_rejects_a_degenerate_step_bound():
    """The rescale divides by max_step, so a zero bound must be refused.

    Under the old per-coordinate clip a zero bound merely pinned that
    coordinate; it now produces a non-finite step, so the check is required by
    the change rather than decorative.
    """
    C_A, C_B = _narrow_time_peak_tables(16.0)
    start = np.asarray([[32.0, 0.5, 0.5, 4.0]])
    for bad in ((2.0, 0.5, 0.0, 0.25), (2.0, 0.5, -0.5, 0.25),
                (2.0, 0.5, 0.5)):
        with pytest.raises(ValueError):
            planner.refine_joint_starts_jax(C_A, C_B, start, 1.0, 8.0,
                                            max_step=bad)


def test_tier_starts_are_a_distinct_orbit_with_one_shared_gradient_norm():
    """Distinct starts; equal refined gradient norms WITHIN a symmetry orbit.

    The campaign's write-up read "min equals median to ten digits" as evidence
    of degenerate starts.  It is not: every retained representative receives
    every certified group action, the log density is exactly invariant under
    them, so an orbit's members must agree to roundoff.  Pin both halves, so
    that neither the distinctness nor the invariance can regress unnoticed.
    """
    C_A, C_B = _narrow_time_peak_tables(256.0)
    summary = planner.summarize_uv_norm_table(C_B)
    portfolio = planner.rank_joint_starts_from_uvq(
        C_A, summary, 1.0, 8.0, angular_oversample=3, max_time_starts=5,
        max_starts=48)
    assert portfolio.symmetry.certified
    assert portfolio.symmetry.group_order > 1
    starts = portfolio.starts
    assert len(np.unique(np.round(starts, 12), axis=0)) == len(starts)
    refined = tuple(np.asarray(item) for item in
                    planner.refine_joint_starts_jax(
                        C_A, summary.C_B, starts, 1.0, 8.0, iterations=18))
    points, values, gradients, hessians, curvatures = refined
    order = int(portfolio.symmetry.group_order)
    norms = np.linalg.norm(gradients, axis=1)
    # group_action is emitted representative-major, so members of one orbit are
    # consecutive: [rep0 act0, rep0 act1, ..., rep1 act0, ...]
    for base in range(0, len(starts), order):
        block = values[base:base + order]
        assert np.allclose(block, block[0], rtol=0.0, atol=1.0e-6), (
            "one symmetry orbit disagreed on its log density: %s" % block)
        # The half this test is named for.  The campaign write-up read "min
        # equals median to ten digits" as evidence of degenerate starts; it is
        # the exact group invariance, and it is pinned here rather than
        # described.  Measured spread within a block is 0 to 3.1e-12.
        gradient_block = norms[base:base + order]
        assert np.max(gradient_block) - np.min(gradient_block) <= 1.0e-9, (
            "one symmetry orbit disagreed on its gradient norm: %s"
            % gradient_block)
        assert np.all(curvatures[base:base + order] > 0.0)

_HM_PACKET = "/tmp/hm51_Ctables_incl0.6.npz"
_SNR40_PACKET = ("/tmp/rift-paper-av-ladder/analyses/va_sequence_20260902/"
                 "records/angle_coeffs_rung40_n256.npz")
_SNR160_PACKET = ("/tmp/rift-paper-av-ladder/analyses/va_sequence_20260902/"
                  "records/angle_coeffs_rung160_n256.npz")


@pytest.mark.skipif(not os.path.exists(_HM_PACKET),
                    reason="external real-table validation packet is absent")
def test_hm_second_mode_survives_unsafe_proxy_gap():
    """Regression for the real Lmax=4 mode proxy that defeated PR267's line."""
    packet = np.load(_HM_PACKET)
    C_A = packet["C_A"]
    summary = planner.summarize_uv_norm_table(packet["C_B"])
    portfolio = planner.rank_joint_starts_from_uvq(
        C_A, summary, 1000.0 / 720.0, 1000.0 / 240.0,
        max_time_starts=3, max_starts=24)
    # This is load-bearing: proxy pruning at the nominal -23 nat error budget,
    # or even at -32, discards a mode whose refined contribution is relevant.
    assert len(portfolio.raw_scores) >= 15
    assert portfolio.raw_scores[14] - portfolio.raw_scores[0] == pytest.approx(
        -37.13988596, abs=2.0e-6)

    result = tuple(np.asarray(item) for item in planner.refine_joint_starts_jax(
        C_A, summary.C_B, portfolio.starts, 1000.0 / 720.0,
        1000.0 / 240.0, iterations=18))
    points, values, gradients, _, curvatures = result
    selected, _ = planner.select_refined_modes(
        points, values, gradients, curvatures, max_modes=24)
    assert len(selected) == 2
    delta = np.sort(values[selected] - np.max(values[selected]))
    np.testing.assert_allclose(delta, [-11.671141934982415, 0.0],
                               rtol=0.0, atol=2.0e-8)
    assert np.max(np.linalg.norm(gradients[selected], axis=1)) < 2.0e-6


@pytest.mark.skipif(not os.path.exists(_HM_PACKET),
                    reason="external real-table validation packet is absent")
def test_hm_two_tier_integral_matches_overcomplete_oracle():
    packet = np.load(_HM_PACKET)
    oracle = 1305.8219235157544
    result = planner.multipeak_local_marginalize(
        packet["C_A"], packet["C_B"], 1000.0 / 720.0, 1000.0 / 240.0,
        oracle, log_integral_tol=1.0e-3, quadrature_order=7,
        cell_sigma=5.0, chunk_size=64)
    assert result.accepted
    assert not result.used_reserve
    assert result.tier0.n_retained_modes == 2
    assert result.tier1.n_retained_modes == 2
    assert abs(result.value - oracle) < 1.0e-3
    assert result.modeled_peak_bytes < 32 * 1024 ** 2


@pytest.mark.skipif(not os.path.exists(_SNR40_PACKET),
                    reason="external real-table validation packet is absent")
def test_real_low_snr_declines_to_finite_reserve():
    packet = np.load(_SNR40_PACKET)
    C_A = packet["C_A"][:, :, 148, :]
    C_B = packet["C_B"][:, :, 148, :]
    oracle = 814.7510954543737
    result = planner.multipeak_local_marginalize(
        C_A, C_B, 0.2, 7.0, oracle, log_integral_tol=1.0e-3,
        quadrature_order=7, cell_sigma=5.0, chunk_size=64)
    assert not result.accepted
    assert result.used_reserve
    assert result.value == oracle
    assert result.provenance.startswith("dense-reserve:")


@pytest.mark.skipif(not os.path.exists(_SNR160_PACKET),
                    reason="external real-table validation packet is absent")
def test_real_high_snr_two_tier_path_matches_overcomplete_oracle():
    packet = np.load(_SNR160_PACKET)
    C_A = packet["C_A"][:, :, 148, :]
    C_B = packet["C_B"][:, :, 148, :]
    oracle = 13255.018541583624
    result = planner.multipeak_local_marginalize(
        C_A, C_B, 0.2, 7.0, oracle, log_integral_tol=1.0e-3,
        quadrature_order=7, cell_sigma=5.0, chunk_size=64)
    assert result.accepted
    assert not result.used_reserve
    assert result.tier0.n_retained_modes == 4
    assert result.tier1.n_retained_modes == 4
    assert result.delta_log_integral < 1.0e-3
    assert abs(result.value - oracle) < 1.0e-3
