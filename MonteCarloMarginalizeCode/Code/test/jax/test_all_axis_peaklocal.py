"""Tests for fixed-shape all-variable multi-peak marginalization."""

import types

import numpy as np
import pytest
from scipy import integrate, special

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from RIFT.likelihood.jax_ile import all_axis_peaklocal as AAP
from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile import core as JCORE


def _problem(n=129):
    span = n - 1.0
    t = np.arange(n, dtype=float)
    k0, kt, kp, ku, B = 5.0, 15.0, 8.0, 6.0, 10.0
    C_A = np.zeros((3, 3, n), dtype=np.complex128)
    C_A[0, 1] = k0 - kt * np.cos(2.0 * np.pi * t / span)
    C_A[2, 1] = 0.5 * kp
    C_A[0, 0] = 0.5 * ku
    C_A[0, 2] = 0.5 * ku
    C_B = np.zeros((5, 5), dtype=np.complex128)
    C_B[0, 2] = B
    constants = dict(span=span, k0=k0, kt=kt, kp=kp, ku=ku, B=B)
    return C_A, C_B, constants


def _joint_peak(constants):
    A = sum(constants[k] for k in ("k0", "kt", "kp", "ku"))
    B = constants["B"]
    x = (A + np.sqrt(A * A - 16.0 * B)) / (2.0 * B)
    centers = np.asarray([
        [constants["span"] / 2.0, 0.0, 0.0, x],
        [constants["span"] / 2.0, np.pi, 0.0, x],
    ])
    hessian = np.zeros((2, 4, 4))
    diagonal = np.asarray([
        -x * constants["kt"] * (2.0 * np.pi / constants["span"]) ** 2,
        -4.0 * x * constants["kp"],
        -x * constants["ku"],
        -B + 4.0 / (x * x),
    ])
    hessian[:, np.arange(4), np.arange(4)] = diagonal
    return centers, hessian


def _analytic_log_integral(constants, x_min, x_max):
    c = constants

    def log_i0(z):
        return np.log(special.i0e(z)) + abs(z)

    def log_integrand(x):
        return (-4.0 * np.log(x) - 0.5 * c["B"] * x * x + c["k0"] * x
                + log_i0(c["kt"] * x) + log_i0(c["kp"] * x)
                + log_i0(c["ku"] * x))

    probe = np.linspace(x_min, x_max, 2001)
    shift = max(log_integrand(x) for x in probe)
    value = integrate.quad(
        lambda x: np.exp(log_integrand(x) - shift), x_min, x_max,
        epsabs=1.0e-13, epsrel=1.0e-13, limit=500)[0]
    return (shift + np.log(value) + np.log(c["span"])
            + 2.0 * np.log(2.0 * np.pi))


def test_angle_tables_forward_primitive_guard_and_report_support(monkeypatch):
    data = types.SimpleNamespace(lms=np.asarray([[2, 2]]), npts=5)
    seen = []

    def fake_accumulate(data, ra, dec, psi, incl, phi, interp,
                        phase_marginalization, guard=0):
        seen.append(int(guard))
        shape = (ra.shape[0], data.npts + 2 * int(guard))
        return jnp.ones(shape, dtype=jnp.complex128), jnp.ones(shape)

    monkeypatch.setattr(AM, "_accumulate_unit", fake_accumulate)
    C_A, C_B, meta = AM.angle_coefficient_tables(
        data, jnp.asarray([0.1]), jnp.asarray([0.2]), jnp.asarray([0.3]),
        guard=3)
    assert C_A.shape == (3, 3, 1, 11)
    assert C_B.shape == (5, 5, 1, 11)
    assert meta["guard"] == 3 and meta["ntime"] == 11
    assert seen and set(seen) == {3}


def test_uv_summary_bounds_the_exact_norm_and_collapses_time():
    rng = np.random.default_rng(813)
    C_B = np.zeros((5, 5), dtype=np.complex128)
    C_B[0, 2] = 20.0
    perturbation = (rng.normal(size=(5, 5))
                    + 1j * rng.normal(size=(5, 5))) * 0.02
    perturbation[0, 2] = 0.0
    C_B += perturbation
    repeated = np.repeat(C_B[..., None], 17, axis=-1)
    summary = AAP.summarize_uv_norm_table(repeated)

    phi = rng.uniform(0.0, 2.0 * np.pi, 1000)
    u = rng.uniform(0.0, 2.0 * np.pi, 1000)
    values = np.asarray(jax.vmap(
        lambda p, q: AAP._angular_field(jnp.asarray(C_B), p, q))(
            jnp.asarray(phi), jnp.asarray(u)))
    assert summary.time_invariant
    assert values.min() >= summary.b_lower - 1.0e-12
    assert values.max() <= summary.b_upper + 1.0e-12
    assert summary.summary_build_count == 1
    assert summary.input_harmonic_coefficients == repeated.size


def test_uv_envelope_ranks_the_interior_time_mode_without_dense_starts():
    C_A, C_B, constants = _problem()
    summary = AAP.summarize_uv_norm_table(C_B)
    starts, envelope = AAP.rank_time_starts_from_uv(
        C_A, summary, 0.2, 7.0, max_starts=3, min_separation=4)
    assert starts[0] == int(constants["span"] // 2)
    assert len(starts) <= 3
    assert envelope[starts[0]] == pytest.approx(envelope.max())


def test_loud_off_lattice_refiner_enters_narrow_coupled_basin():
    """A fixed structural lattice remains useful as the peak narrows."""
    C_A, C_B, _ = _problem(33)
    scale = 64.0
    starts = np.asarray([[15.3, 0.2, 0.2, 5.0 / scale],
                         [16.7, 3.0, 0.1, 5.0 / scale]])
    result = tuple(np.asarray(item) for item in AAP.refine_all_axis_starts(
        scale * C_A, scale * scale * C_B, starts,
        0.2 / scale, 7.0 / scale, iterations=18))
    points, values, gradients, _, curvatures = result
    selected, stationary = AAP.select_refined_modes(
        points, values, gradients, curvatures, max_modes=2,
        gradient_tol=2.0e-6)
    assert stationary.any()
    assert len(selected) >= 1
    assert np.max(np.linalg.norm(gradients[selected], axis=1)) < 2.0e-6
    assert np.all(curvatures[selected] > 0.0)


def test_refiner_mask_skips_padding_without_changing_live_modes():
    C_A, C_B, constants = _problem(33)
    centers, _ = _joint_peak(constants)
    starts = np.vstack((
        centers + np.asarray([[0.4, 0.1, 0.1, -0.05],
                              [-0.4, -0.1, 0.1, 0.05]]),
        np.repeat([[0.0, 0.0, 0.0, 0.2]], 6, axis=0)))
    live = np.asarray([True, True] + [False] * 6)
    masked = jax.jit(lambda seed, mask: AAP.refine_all_axis_starts(
        C_A, C_B, seed, 0.2, 7.0, iterations=14, live=mask))(
            jnp.asarray(starts), jnp.asarray(live))
    direct = AAP.refine_all_axis_starts(
        C_A, C_B, starts[:2], 0.2, 7.0, iterations=14)
    for masked_value, direct_value in zip(masked, direct):
        np.testing.assert_allclose(
            np.asarray(masked_value)[:2], np.asarray(direct_value),
            rtol=1.0e-12, atol=1.0e-12)
    assert np.all(np.isneginf(np.asarray(masked[1])[2:]))
    assert np.all(np.asarray(masked[2])[2:] == 0.0)
    assert np.all(np.asarray(masked[4])[2:] == 1.0)


def test_uv_ranked_time_start_feeds_algebraic_angles_and_analytic_distance():
    C_A, C_B, constants = _problem(65)
    summary = AAP.summarize_uv_norm_table(C_B)
    time_starts, _ = AAP.rank_time_starts_from_uv(
        C_A, summary, 0.2, 7.0, max_starts=1, min_separation=4)
    # This deliberately sparse symmetric polynomial includes zero/infinite
    # generalized roots; the authoritative report, not NumPy's intermediate
    # negative-power warning, carries their classification.
    with np.errstate(invalid="ignore", divide="ignore"):
        starts, algebraic_ok, reports = AAP.algebraic_angle_starts_from_uv(
            C_A, summary, time_starts, 0.2, 7.0)

    # The exact enumerator supplies the two maxima without a dense seed lattice.
    # Its independent completeness result is carried separately; definite
    # maxima remain useful targeting data if a numerically marginal projection
    # makes that result false on another LAPACK implementation.
    assert starts.shape == (2, 4)
    assert algebraic_ok == bool(reports[0]["ok"])
    assert len(reports) == 1 and reports[0]["n_maxima"] == 2
    assert np.allclose(starts[:, 0], constants["span"] / 2.0)
    assert np.all((starts[:, 3] > 0.2) & (starts[:, 3] < 7.0))


def test_joint_start_lattice_is_harmonic_order_sized_not_snr_sized():
    C_A, C_B, constants = _problem(65)
    summary = AAP.summarize_uv_norm_table(C_B)
    low = AAP.rank_joint_starts_from_uvq(
        C_A, summary, 0.2, 7.0, max_time_starts=1, max_starts=8)
    high = AAP.rank_joint_starts_from_uvq(
        32.0 * C_A, summary, 0.2, 7.0,
        max_time_starts=1, max_starts=8)

    assert low.starts.shape[1] == 4
    assert low.time_starts[0] == int(constants["span"] // 2)
    assert low.n_phi_lattice == high.n_phi_lattice == 17
    assert low.n_u_lattice == high.n_u_lattice == 9
    assert low.n_lattice_evaluations == high.n_lattice_evaluations == 17 * 9 * 65
    assert low.n_exact_symmetry_shifts == high.n_exact_symmetry_shifts == 2
    assert low.capacity_ok and high.capacity_ok
    assert np.all((low.starts[:, 3] >= 0.2) & (low.starts[:, 3] <= 7.0))


def test_joint_start_guard_discards_support_before_ranking():
    C_A, C_B, constants = _problem(65)
    guard = 8
    guarded = np.full(C_A.shape[:-1] + (65 + 2 * guard,),
                      1.0e8 + 2.0e8j, dtype=np.complex128)
    guarded[..., guard:-guard] = C_A
    summary = AAP.summarize_uv_norm_table(C_B)
    plan = AAP.rank_joint_starts_from_uvq(
        guarded, summary, 0.2, 7.0, time_guard=guard,
        max_time_starts=1, max_starts=8)

    assert plan.time_starts.tolist() == [int(constants["span"] // 2)]
    assert plan.n_lattice_evaluations == 17 * 9 * 65


def test_joint_start_capacity_declines_instead_of_silent_truncation():
    C_A, C_B, _ = _problem(65)
    summary = AAP.summarize_uv_norm_table(C_B)
    plan = AAP.rank_joint_starts_from_uvq(
        C_A, summary, 0.2, 7.0, max_time_starts=1, max_starts=1)
    assert plan.starts.shape == (1, 4)
    assert plan.n_candidates_before_cap > 1
    assert not plan.capacity_ok


def test_device_joint_start_portfolio_is_fixed_shape_jittable_and_vmappable():
    C_A, C_B, _ = _problem(33)

    def rank(table):
        return AAP.rank_joint_starts_from_uvq_device(
            table, C_B, 0.2, 7.0, max_starts=32,
            angular_oversample=2)

    low = jax.jit(rank)(jnp.asarray(C_A))
    high = jax.jit(rank)(jnp.asarray(64.0 * C_A))
    assert low.starts.shape == high.starts.shape == (32, 4)
    assert int(low.n_phi_lattice) == int(high.n_phi_lattice) == 17
    assert int(low.n_u_lattice) == int(high.n_u_lattice) == 9
    assert int(low.n_lattice_evaluations) == 17 * 9 * 33
    assert int(high.n_lattice_evaluations) == 17 * 9 * 33
    assert int(low.n_time_scout_evaluations) == 4 * 4 * 33
    assert int(low.n_retained_time_samples) == 33
    assert bool(low.time_cover_certified) and bool(high.time_cover_certified)
    assert bool(low.time_capacity_ok) and bool(high.time_capacity_ok)
    assert bool(low.norm_nonnegative) and bool(high.norm_nonnegative)
    assert bool(low.capacity_ok) and bool(high.capacity_ok)
    assert np.all(np.asarray(low.starts)[np.asarray(low.live), 3] >= 0.2)
    assert np.all(np.asarray(high.starts)[np.asarray(high.live), 3] <= 7.0)

    batch = jax.jit(jax.vmap(rank))(jnp.asarray(
        np.stack((C_A, 1.01 * C_A))))
    assert batch.starts.shape == (2, 32, 4)
    np.testing.assert_array_equal(
        np.asarray(batch.n_lattice_evaluations), [17 * 9 * 33] * 2)


def test_device_time_cover_bounds_discarded_cells_and_limits_full_lattice():
    n_time = 129
    t = np.arange(n_time, dtype=float)
    C_A = np.zeros((1, 1, n_time), dtype=np.complex128)
    C_A[0, 0] = 5.0 - 15.0 * np.cos(2.0 * np.pi * t / (n_time - 1.0))
    C_B = np.asarray([[10.0 + 0.0j]])
    cover = jax.jit(lambda table: AAP._time_cell_cover_device(
        table, C_B, 0.5, 2.0, time_guard=0, keep_nats=5.0,
        scout_size=4))(jnp.asarray(C_A))
    live = np.asarray(cover["live_cells"])
    cell_upper = np.asarray(cover["cell_mass_upper"])
    assert bool(cover["certified"])
    assert 0 < np.count_nonzero(live) < live.size
    assert float(cover["outside_log_bound"]) == pytest.approx(
        special.logsumexp(cell_upper[~live]))

    coeff, frequency, offset = AAP._time_primitive_spectrum(
        jnp.asarray(C_A.reshape((1, n_time))), 0)
    x = np.linspace(0.5, 2.0, 129)
    log_volume = np.log((2.0 * np.pi) ** 2 * (2.0 - 0.5))
    for cell in np.flatnonzero(~live):
        position = np.linspace(cell, cell + 1.0, 33)
        amplitude = np.asarray(AAP._evaluate_time_spectrum(
            coeff, frequency, jnp.asarray(position), offset))[0].real
        sampled = (amplitude[:, None] * x[None, :]
                   - 5.0 * x[None, :] ** 2 - 4.0 * np.log(x[None, :]))
        assert sampled.max() + log_volume <= cell_upper[cell] + 1.0e-10

    plan = jax.jit(lambda table: AAP.rank_joint_starts_from_uvq_device(
        table, C_B, 0.5, 2.0, max_starts=32, max_time_nodes=64,
        time_keep_nats=5.0))(jnp.asarray(C_A))
    assert int(plan.n_retained_time_samples) == n_time
    assert int(plan.n_time_lattice) == 64
    assert int(plan.n_lattice_evaluations) == 9 * 9 * 64
    assert int(plan.n_time_scout_evaluations) == 4 * 4 * n_time
    assert int(plan.n_time_nodes_retained) <= 64
    assert bool(plan.time_capacity_ok)
    assert float(plan.time_cover_min_sample) == float(
        np.flatnonzero(live)[0])
    assert float(plan.time_cover_max_sample) == float(
        np.flatnonzero(live)[-1] + 1)
    # The angle-constant fixture is deliberately degenerate: every angular
    # lattice point is a maximum, so the independent start-capacity gate still
    # declines even though the time cover itself fits and is certified.
    assert int(plan.n_candidates_before_cap) > 32
    assert not bool(plan.capacity_ok)


def test_device_joint_start_portfolio_fails_closed_on_capacity_and_norm():
    C_A, C_B, _ = _problem(33)
    truncated = jax.jit(lambda table: AAP.rank_joint_starts_from_uvq_device(
        table, C_B, 0.2, 7.0, max_starts=1))(jnp.asarray(C_A))
    assert int(truncated.n_candidates_before_cap) > 1
    assert not bool(truncated.capacity_ok)

    invalid_norm = C_B.copy()
    invalid_norm[0, 2] = -10.0
    rejected = jax.jit(lambda table: AAP.rank_joint_starts_from_uvq_device(
        C_A, table, 0.2, 7.0, max_starts=8))(jnp.asarray(invalid_norm))
    assert not bool(rejected.norm_nonnegative)
    assert not bool(rejected.capacity_ok)
    assert not np.any(np.asarray(rejected.live))

    time_overflow = jax.jit(
        lambda table: AAP.rank_joint_starts_from_uvq_device(
            table, C_B, 0.2, 7.0, max_starts=8, max_time_nodes=2))(
                jnp.asarray(C_A))
    assert not bool(time_overflow.time_capacity_ok)
    assert not bool(time_overflow.capacity_ok)

    truncated = truncated._replace(
        time_cover_min_sample=jnp.asarray(2.0),
        time_cover_max_sample=jnp.asarray(5.0),
        time_outside_log_bound=jnp.asarray(-11.0))
    rejected = rejected._replace(
        time_cover_min_sample=jnp.asarray(3.0),
        time_cover_max_sample=jnp.asarray(6.0),
        time_outside_log_bound=jnp.asarray(-13.0))
    combined = AAP.combine_device_start_plans(truncated, rejected)
    assert combined.starts.shape == (9, 4)
    assert not bool(combined.capacity_ok)
    assert not bool(combined.norm_nonnegative)
    assert float(combined.time_cover_min_sample) == 2.0
    assert float(combined.time_cover_max_sample) == 6.0
    assert float(combined.time_outside_log_bound) == -13.0


def test_device_mode_plan_refines_and_deduplicates_without_host_transfer():
    C_A, C_B, _ = _problem(33)

    @jax.jit
    def build(table):
        starts = AAP.rank_joint_starts_from_uvq_device(
            table, C_B, 0.2, 7.0, max_starts=32)
        return AAP.make_all_axis_mode_plan_device(
            table, C_B, starts, 0.2, 7.0, max_modes=4,
            local_radius=3.0, iterations=14,
            time_reconstruction_certified=True)

    plan, ledger = build(jnp.asarray(C_A))
    assert plan.centers.shape == (4, 4)
    assert plan.local_transforms.shape == (4, 4, 4)
    assert int(ledger["n_optimizer_starts"]) >= 2
    assert int(ledger["n_selected_modes"]) == 2
    assert int(jnp.count_nonzero(plan.live)) == 2
    assert bool(ledger["discovery_capacity_ok"])
    assert not bool(ledger["selection_overflow"])
    assert not bool(ledger["global_completeness_certified"])
    assert not bool(ledger["derivative_warrant_certified"])
    assert bool(plan.time_reconstruction_certified)
    live_transforms = np.asarray(plan.local_transforms)[np.asarray(plan.live)]
    assert np.all(np.diagonal(live_transforms, axis1=1, axis2=2) > 0.0)

    plans, ledgers = jax.jit(jax.vmap(build))(jnp.asarray(
        np.stack((C_A, 1.01 * C_A))))
    assert plans.centers.shape == (2, 4, 4)
    assert ledgers["n_selected_modes"].shape == (2,)
    assert np.all(np.asarray(ledgers["discovery_capacity_ok"]))

    @jax.jit
    def overflow(table):
        starts = AAP.rank_joint_starts_from_uvq_device(
            table, C_B, 0.2, 7.0, max_starts=32)
        return AAP.make_all_axis_mode_plan_device(
            table, C_B, starts, 0.2, 7.0, max_modes=1,
            local_radius=3.0, iterations=14,
            time_reconstruction_certified=True)

    overflow_plan, overflow_ledger = overflow(jnp.asarray(C_A))
    assert int(jnp.count_nonzero(overflow_plan.live)) == 1
    assert bool(overflow_ledger["selection_overflow"])
    assert not bool(overflow_ledger["discovery_capacity_ok"])
    assert not bool(overflow_plan.discovery_capacity_ok)


def test_device_plans_compose_with_empirical_local_controller_under_vmap():
    C_A, C_B, _ = _problem(33)

    def evaluate(table):
        base_starts = AAP.rank_joint_starts_from_uvq_device(
            table, C_B, 0.2, 7.0, max_starts=32,
            angular_oversample=1)
        extra_starts = AAP.rank_joint_starts_from_uvq_device(
            table, C_B, 0.2, 7.0, max_starts=32,
            angular_oversample=2)
        separate_base_plan, separate_base_planning = (
            AAP.make_all_axis_mode_plan_device(
                table, C_B, base_starts, 0.2, 7.0, max_modes=4,
                local_radius=3.0, iterations=14,
                time_reconstruction_certified=True))
        (base_plan, enriched_plan, base_planning, enriched_planning,
         shared_planning) = AAP.make_all_axis_mode_plan_pair_device(
            table, C_B, base_starts, extra_starts, 0.2, 7.0,
            max_modes=4, enriched_max_modes=8,
            local_radius=3.0, iterations=14,
            time_reconstruction_certified=True)
        # Plans are row-local control data.  Until derivative parity is
        # established, outer differentiation must not interpret the discrete
        # rank/dedup decisions as a full-marginal derivative certificate.
        base_plan = jax.tree.map(jax.lax.stop_gradient, base_plan)
        enriched_plan = jax.tree.map(jax.lax.stop_gradient, enriched_plan)
        value, accepted, ledger = AAP.empirical_enrichment_marginalize(
            table, C_B, base_plan, enriched_plan, 0.2, 7.0,
            base_order=7, base_check_order=9,
            enriched_order=9, enriched_check_order=11,
            convergence_tol_nats=1.0e-3)
        return (value, accepted, ledger, base_planning, enriched_planning,
                shared_planning, separate_base_plan,
                separate_base_planning, base_plan)

    (value, accepted, ledger, base_planning, enriched_planning,
     shared_planning, separate_base_plan, separate_base_planning,
     paired_base_plan) = jax.jit(evaluate)(jnp.asarray(C_A))
    assert np.isfinite(float(value))
    assert bool(accepted)
    assert not bool(ledger["fallback_required"])
    assert bool(ledger["mode_nesting_ok"])
    assert int(base_planning["n_selected_modes"]) == 2
    assert int(enriched_planning["n_selected_modes"]) == 2
    assert int(enriched_planning["n_lattice_evaluations"]) == (
        9 * 9 * 33 + 17 * 9 * 33)
    n_base = int(base_planning["n_optimizer_starts"])
    n_enriched = int(enriched_planning["n_optimizer_starts"])
    assert int(shared_planning["n_optimizer_starts_executed"]) == n_enriched
    assert int(shared_planning["n_optimizer_starts_avoided"]) == n_base
    assert int(shared_planning["n_optimizer_starts_previous_two_pass"]) == (
        n_base + n_enriched)
    assert bool(shared_planning["start_nesting_structural"])
    assert int(separate_base_planning["n_selected_modes"]) == 2
    for separate, paired in zip(jax.tree.leaves(separate_base_plan),
                                jax.tree.leaves(paired_base_plan)):
        np.testing.assert_allclose(np.asarray(separate), np.asarray(paired),
                                   rtol=0.0, atol=1.0e-12)

    batch = jax.jit(jax.vmap(evaluate))(jnp.asarray(
        np.stack((C_A, 1.01 * C_A))))
    assert batch[0].shape == batch[1].shape == (2,)
    assert np.all(np.isfinite(np.asarray(batch[0])))
    assert np.all(np.asarray(batch[1]))


def test_exact_coefficient_symmetry_completes_quadrupole_orbit():
    C_A = np.zeros((3, 3, 9), dtype=np.complex128)
    C_A[2, 0] = 1.0
    C_A[2, 2] = 0.7
    C_B = np.zeros((5, 5), dtype=np.complex128)
    C_B[0, 2] = 2.0
    shifts = AAP._exact_angular_translation_symmetries(C_A, C_B)
    want = np.asarray([
        [0.0, 0.0], [np.pi, 0.0],
        [0.5 * np.pi, np.pi], [1.5 * np.pi, np.pi]])
    assert shifts.shape == (4, 2)
    for shift in want:
        assert np.min(np.linalg.norm(shifts - shift, axis=1)) < 1.0e-12

    device_shifts, device_live = jax.jit(
        AAP._exact_angular_translation_symmetries_device)(C_A, C_B)
    device_shifts = np.asarray(device_shifts)[np.asarray(device_live)]
    assert device_shifts.shape == (4, 2)
    for shift in want:
        assert np.min(np.linalg.norm(device_shifts - shift, axis=1)) < 1.0e-12


def test_mode_stationarity_uses_curvature_scaled_displacement_at_high_snr():
    point = np.asarray([[10.0, 1.0, 2.0, 1.5]])
    value = np.asarray([10000.0])
    gradient = np.asarray([[2.0e-4, 0.0, 0.0, 0.0]])
    curvature = np.asarray([[35.0, 200.0, 1000.0, 100000.0]])
    selected, stationary = AAP.select_refined_modes(
        point, value, gradient, curvature, max_modes=1,
        gradient_tol=2.0e-6)
    assert stationary.tolist() == [True]
    assert selected.tolist() == [0]

    curvature[0, 0] = 1.0
    selected, stationary = AAP.select_refined_modes(
        point, value, gradient, curvature, max_modes=1,
        gradient_tol=2.0e-6)
    assert stationary.tolist() == [False]
    assert selected.size == 0


def test_jax_gradient_hessian_refinement_finds_both_angular_modes():
    C_A, C_B, constants = _problem(65)
    centers, _ = _joint_peak(constants)
    starts = centers + np.asarray([
        [1.3, 0.17, -0.11, -0.13],
        [-1.1, -0.15, 0.09, 0.16],
    ])
    refined, value, gradient, hessian, curvature = AAP.refine_all_axis_starts(
        C_A, C_B, starts, 0.2, 7.0, iterations=14)
    refined = np.asarray(refined)
    angular_error = np.abs(
        (refined[:, 1:3] - centers[:, 1:3] + np.pi) % (2.0 * np.pi) - np.pi)
    assert np.max(np.abs(refined[:, [0, 3]] - centers[:, [0, 3]])) < 2.0e-7
    assert np.max(angular_error) < 2.0e-7
    assert np.max(np.linalg.norm(np.asarray(gradient), axis=1)) < 2.0e-7
    assert np.all(np.asarray(curvature) > 0.0)
    assert np.all(np.isfinite(np.asarray(value)))
    assert np.all(np.isfinite(np.asarray(hessian)))
    selected, stationary = AAP.select_refined_modes(
        refined, value, gradient, curvature, max_modes=4)
    assert np.all(stationary)
    assert selected.shape == (2,)
    with pytest.raises(ValueError, match="exceeds fixed plan capacity"):
        AAP.select_refined_modes(
            refined, value, gradient, curvature, max_modes=1)
    with pytest.raises(ValueError, match="must be positive"):
        AAP.select_refined_modes(
            refined, value, gradient, curvature, max_modes=0)


def test_multimode_local_primitive_matches_oracle_but_stays_uncertified():
    C_A, C_B, constants = _problem()
    centers, hessian = _joint_peak(constants)
    transforms, half_widths = AAP.mode_local_geometry(hessian, w_sigma=5.0)
    x_min, x_max = 0.2, 7.0
    truth = _analytic_log_integral(constants, x_min, x_max)
    plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=4,
        local_transforms=transforms, local_radius=5.0,
        outside_log_bound=np.inf,
        enumeration_complete=True, outside_bound_certified=False)
    value, ok, ledger = AAP.all_axis_peak_local_marginalize(
        C_A, C_B, plan, x_min, x_max, local_order=13, check_order=19,
        quadrature_tol_nats=1.0e-4)

    assert not bool(ok)
    assert bool(ledger["decline_incomplete"])
    assert abs(float(value) - truth) < 2.0e-4
    assert int(ledger["n_modes"]) == 2
    assert int(ledger["n_local_evaluations_hi"]) == 2 * 19 ** 4
    assert int(ledger["n_selected_time_points_hi"]) == 2 * 19
    assert int(ledger["n_time_frequency_terms_hi"]) == (
        2 * 19 * (2 * C_A.shape[-1] - 2) * np.prod(C_A.shape[:-1]))
    assert int(ledger["n_angle_harmonic_terms_hi"]) == (
        2 * 19 ** 3 * (np.prod(C_A.shape[:-1]) + C_B.size))
    assert int(ledger["workspace_bytes_hi"]) < 8_000_000
    assert bool(ledger["reconciles"])


def test_empirical_enrichment_accepts_without_claiming_global_proof():
    C_A, C_B, constants = _problem(33)
    C_A *= 0.1
    x_min, x_max = 0.5, 2.0
    centers = np.asarray([[
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_min + x_max)]])
    transforms = np.asarray([np.diag([
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_max - x_min)])])
    plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, outside_bound_certified=False,
        time_reconstruction_certified=True,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True)
    value, accepted, ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, plan, plan, x_min, x_max,
        convergence_tol_nats=1.0e-3)

    assert np.isfinite(float(value))
    assert bool(accepted)
    assert bool(ledger["acceptance_is_empirical_enrichment"])
    assert not bool(ledger["global_completeness_certified"])
    assert not bool(ledger["empirical_value_error_certified"])
    assert float(ledger["convergence_error"]) <= 1.0e-3
    assert bool(ledger["value_error_budget_complete"])
    assert bool(ledger["value_error_budget_ok"])
    assert bool(ledger["value_error_budget_is_empirical"])
    assert not bool(ledger["value_error_budget_is_formal_bound"])
    assert (float(ledger["empirical_value_error_score_nats"])
            <= float(ledger["total_value_error_budget_nats"]))
    # Paired terms are charged ONCE, as the max over the two plans: they are
    # the same quantity measured on nested plans, and summing them double
    # counted a common-mode error (refused a correct value at 10x amplitude).
    score_components = [
        "error_score_discovery_nats",
        "error_score_quadrature_nats",
        "error_score_time_guard_nats",
        "error_score_omitted_time_nats",
    ]
    components = np.asarray([float(ledger[key]) for key in score_components])
    score = float(ledger["empirical_value_error_score_nats"])
    assert score == pytest.approx(float(np.sum(components)), abs=1.0e-15)
    assert bool(ledger["error_score_pairs_charged_as_max"])
    for pair, charged in (
            (("error_score_base_quadrature_nats",
              "error_score_enriched_quadrature_nats"),
             "error_score_quadrature_nats"),
            (("error_score_base_time_guard_nats",
              "error_score_enriched_time_guard_nats"),
             "error_score_time_guard_nats"),
            (("error_score_base_omitted_time_nats",
              "error_score_enriched_omitted_time_nats"),
             "error_score_omitted_time_nats")):
        assert float(ledger[charged]) == pytest.approx(
            max(float(ledger[pair[0]]), float(ledger[pair[1]])), abs=1e-15)
    assert (float(ledger["error_score_base_time_guard_nats"])
            == float(ledger["error_score_enriched_time_guard_nats"]) == 0.0)
    assert bool(ledger["mode_nesting_ok"])
    assert not bool(ledger["fallback_required"])
    assert bool(ledger["reconciles"])

    loose_time_plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, outside_bound_certified=False,
        time_reconstruction_certified=True,
        time_outside_log_bound=1.0e6,
        time_outside_bound_certified=True)
    _, accepted, time_tail_ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, loose_time_plan, loose_time_plan, x_min, x_max,
        convergence_tol_nats=1.0e-3)
    assert not bool(accepted)
    assert bool(time_tail_ledger["time_outside_cover_used"])
    assert not bool(time_tail_ledger["time_omitted_mass_ok"])
    assert bool(time_tail_ledger["decline_time_omitted_mass"])
    assert bool(time_tail_ledger["decline_time_omitted_mass_bound"])
    assert not bool(time_tail_ledger["decline_time_cover_incomplete"])
    assert bool(time_tail_ledger["reconciles"])

    missing_time_plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, outside_bound_certified=False,
        time_reconstruction_certified=True)
    _, accepted, missing_time_ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, missing_time_plan, missing_time_plan, x_min, x_max,
        convergence_tol_nats=1.0e-3)
    assert not bool(accepted)
    assert not bool(missing_time_ledger["time_outside_cover_used"])
    assert not bool(missing_time_ledger["value_error_budget_complete"])
    assert bool(missing_time_ledger["decline_time_omitted_mass"])
    assert bool(missing_time_ledger["decline_time_cover_incomplete"])
    assert not bool(missing_time_ledger["decline_time_omitted_mass_bound"])
    assert not bool(missing_time_ledger["decline_error_budget"])
    assert bool(missing_time_ledger["reconciles"])

    _, accepted, one_time_ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, plan, missing_time_plan, x_min, x_max,
        convergence_tol_nats=1.0e-3)
    assert not bool(accepted)
    assert bool(one_time_ledger["time_outside_cover_any"])
    assert not bool(one_time_ledger["time_outside_cover_used"])
    assert bool(one_time_ledger["decline_time_cover_incomplete"])
    assert bool(one_time_ledger["reconciles"])

    # Each legacy diagnostic can clear its individual gate while their
    # cancellation-resistant sum exceeds one shared allowance.
    aggregate_budget = 0.5 * (float(np.max(components)) + score)
    assert float(np.max(components)) < aggregate_budget < score
    _, accepted, aggregate_ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, plan, plan, x_min, x_max,
        convergence_tol_nats=1.0e-3,
        total_value_error_budget_nats=aggregate_budget)
    assert not bool(accepted)
    assert bool(aggregate_ledger["base_quadrature_error"] <= 1.0e-3)
    assert bool(aggregate_ledger["enriched_quadrature_error"] <= 1.0e-3)
    assert bool(aggregate_ledger["convergence_error"] <= 1.0e-3)
    assert not bool(aggregate_ledger["value_error_budget_ok"])
    assert bool(aggregate_ledger["decline_error_budget"])
    assert bool(aggregate_ledger["reconciles"])

    _, accepted_at_budget, boundary_ledger = (
        AAP.empirical_enrichment_marginalize(
            C_A, C_B, plan, plan, x_min, x_max,
            convergence_tol_nats=1.0e-3,
            total_value_error_budget_nats=score))
    assert bool(accepted_at_budget)
    assert bool(boundary_ledger["value_error_budget_ok"])
    _, accepted_below_budget, below_ledger = (
        AAP.empirical_enrichment_marginalize(
            C_A, C_B, plan, plan, x_min, x_max,
            convergence_tol_nats=1.0e-3,
            total_value_error_budget_nats=np.nextafter(score, -np.inf)))
    assert not bool(accepted_below_budget)
    assert bool(below_ledger["decline_error_budget"])
    assert bool(below_ledger["reconciles"])

    _, accepted, ordered_ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, plan, plan, x_min, x_max,
        convergence_tol_nats=1.0e-12,
        total_value_error_budget_nats=1.0e-15)
    assert not bool(accepted)
    assert bool(ordered_ledger["decline_quadrature"])
    assert not bool(ordered_ledger["decline_error_budget"])
    assert bool(ordered_ledger["reconciles"])

    for invalid_budget in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="total_value_error_budget_nats"):
            AAP.empirical_enrichment_marginalize(
                C_A, C_B, plan, plan, x_min, x_max,
                total_value_error_budget_nats=invalid_budget)

    # A stronger discovery pass may expose a broad diagnostic basin whose
    # positive integral is negligible.  It must not invalidate the unchanged,
    # valid base cover when the enrichment delta remains inside the same budget.
    extra_centers = np.vstack((centers, [[
        centers[0, 0], 0.1, 0.2, centers[0, 3]]]))
    extra_transforms = np.concatenate((
        transforms,
        np.asarray([np.diag([1.0e-8, 4.0, 4.0, 1.0e-8])])), axis=0)
    diagnostic_plan = AAP.make_all_axis_mode_plan(
        extra_centers, max_modes=3, local_transforms=extra_transforms,
        local_radius=1.0, outside_bound_certified=False,
        time_reconstruction_certified=True,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True)
    retained, accepted, diagnostic_ledger = (
        AAP.empirical_enrichment_marginalize(
            C_A, C_B, plan, diagnostic_plan, x_min, x_max,
            convergence_tol_nats=1.0e-3))
    assert bool(accepted)
    assert bool(diagnostic_ledger["base_geometry_ok"])
    assert not bool(diagnostic_ledger["enriched_geometry_ok"])
    assert bool(diagnostic_ledger["geometry_nesting_ok"])
    assert bool(diagnostic_ledger["accepted_value_uses_base_geometry"])
    assert float(retained) == pytest.approx(float(ledger["base_value"]), abs=1e-12)

    shifted = centers.copy()
    shifted[:, 1] += 0.1
    shifted_plan = AAP.make_all_axis_mode_plan(
        shifted, max_modes=2, local_transforms=transforms,
        local_radius=1.0, outside_bound_certified=False,
        time_reconstruction_certified=True,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True)
    _, accepted, nesting_ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, plan, shifted_plan, x_min, x_max,
        convergence_tol_nats=1.0e-3)
    assert not bool(accepted)
    assert bool(nesting_ledger["decline_mode_nesting"])
    assert bool(nesting_ledger["reconciles"])

    truncated_plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, outside_bound_certified=False,
        time_reconstruction_certified=True,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True,
        discovery_capacity_ok=False)
    _, accepted, capacity_ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, truncated_plan, plan, x_min, x_max,
        convergence_tol_nats=1.0e-3)
    assert not bool(accepted)
    assert bool(capacity_ledger["decline_capacity"])
    assert bool(capacity_ledger["fallback_required"])
    assert not bool(capacity_ledger["decline_is_waveform_failure"])
    assert bool(capacity_ledger["reconciles"])

    empty_plan = AAP.make_all_axis_mode_plan(
        np.empty((0, 4)), max_modes=2,
        local_transforms=np.empty((0, 4, 4)), local_radius=1.0,
        outside_bound_certified=False,
        time_reconstruction_certified=True,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True)
    _, accepted, empty_ledger = AAP.empirical_enrichment_marginalize(
        C_A, C_B, empty_plan, empty_plan, x_min, x_max,
        convergence_tol_nats=1.0e-3)
    assert not bool(accepted)
    assert not bool(empty_ledger["base_and_enriched_values_finite"])
    assert bool(empty_ledger["decline_no_modes"])
    assert not bool(empty_ledger["decline_nonfinite"])
    assert bool(empty_ledger["fallback_required"])
    assert bool(empty_ledger["reconciles"])


def test_native_time_reserve_is_diagnostic_on_local_decline():
    C_A, C_B, constants = _problem(33)
    C_A *= 0.1
    x_min, x_max = 0.5, 2.0
    centers = np.asarray([[
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_min + x_max)]])
    transforms = np.asarray([np.diag([
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_max - x_min)])])
    declined_plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, time_reconstruction_certified=True,
        discovery_capacity_ok=False)

    x_grid = np.linspace(x_min, x_max, 129)
    dx = np.empty_like(x_grid)
    dx[1:-1] = 0.5 * (x_grid[2:] - x_grid[:-2])
    dx[0] = x_grid[1] - x_grid[0]
    dx[-1] = x_grid[-1] - x_grid[-2]
    log_w = np.log(dx * x_grid ** -4)
    time_weights = np.ones(C_A.shape[-1])
    selected, usable, ledger = AAP.empirical_enrichment_with_exact_reserve(
        C_A, C_B, declined_plan, declined_plan, x_min, x_max,
        reserve_x_grid=x_grid, reserve_log_weights=log_w,
        time_weights=time_weights, reserve_amp_sizing=30.0,
        reserve_dense_chunk=8, reserve_grid_block=16)

    lnL_t = AM.coefficient_table_distphipsimarg_exact(
        C_A, C_B, x_grid, log_w, amp_sizing=30.0,
        dense_chunk=8, grid_block=16)
    m = np.max(np.asarray(lnL_t)[0])
    expected = m + np.log(np.sum(
        time_weights * np.exp(np.asarray(lnL_t)[0] - m)))
    assert float(selected) == pytest.approx(expected, abs=2.0e-12)
    assert not bool(usable)
    assert not bool(ledger["accepted_local"])
    assert bool(ledger["decline_capacity"])
    assert bool(ledger["reserve_executed"])
    assert bool(ledger["reserve_finite"])
    assert not bool(ledger["selected_value_is_warranted_reserve"])
    assert not bool(ledger["sample_retained_after_local_decline"])
    assert not bool(ledger["decline_is_waveform_failure"])
    assert bool(ledger["local_fallback_required"])
    assert bool(ledger["fallback_required"])
    assert not bool(ledger["accepted"])
    assert bool(ledger["reconciles"])
    assert bool(ledger["disposition_reconciles"])
    assert int(ledger["reserve_distance_points"]) == x_grid.size
    assert bool(ledger["reserve_uses_native_time"])
    assert not bool(ledger["reserve_native_time_warranted"])
    assert not bool(ledger["reserve_time_warranted"])


def test_declined_controller_can_execute_guarded_bandlimited_time_reserve():
    C_A, C_B, constants = _problem(33)
    C_A *= 0.1
    guard = 8
    support_time = np.arange(-guard, C_A.shape[-1] + guard, dtype=float)
    guarded = np.zeros(C_A.shape[:-1] + (support_time.size,),
                       dtype=np.complex128)
    guarded[0, 1] = 0.1 * (
        constants["k0"] - constants["kt"]
        * np.cos(2.0 * np.pi * support_time / constants["span"]))
    guarded[2, 1] = 0.05 * constants["kp"]
    guarded[0, 0] = 0.05 * constants["ku"]
    guarded[0, 2] = 0.05 * constants["ku"]
    np.testing.assert_allclose(guarded[..., guard:-guard], C_A, atol=1e-14)

    x_min, x_max = 0.5, 2.0
    centers = np.asarray([[
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_min + x_max)]])
    transforms = np.asarray([np.diag([
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_max - x_min)])])
    declined_plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, time_reconstruction_certified=False,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True,
        discovery_capacity_ok=False)
    x_grid = np.linspace(x_min, x_max, 65)
    dx = np.empty_like(x_grid)
    dx[1:-1] = 0.5 * (x_grid[2:] - x_grid[:-2])
    dx[0] = x_grid[1] - x_grid[0]
    dx[-1] = x_grid[-1] - x_grid[-2]
    log_w = np.log(dx * x_grid ** -4)
    time_nodes = np.linspace(0.0, constants["span"], 65)
    time_weights = np.full(time_nodes.size, 0.5)
    time_weights[[0, -1]] *= 0.5

    coeff, frequency, offset = AAP._time_primitive_spectrum(
        guarded.reshape((-1, guarded.shape[-1])), guard)
    fine = AAP._evaluate_time_spectrum(
        coeff, frequency, time_nodes, offset).reshape(
            C_A.shape[:-1] + (time_nodes.size,))
    lnL_t = AM.coefficient_table_distphipsimarg_exact(
        fine, C_B, x_grid, log_w, amp_sizing=30.0,
        dense_chunk=8, grid_block=16)
    m = np.max(np.asarray(lnL_t)[0])
    expected = m + np.log(np.sum(
        time_weights * np.exp(np.asarray(lnL_t)[0] - m)))
    selected, usable, ledger = AAP.empirical_enrichment_with_exact_reserve(
        guarded, C_B, declined_plan, declined_plan, x_min, x_max,
        reserve_x_grid=x_grid, reserve_log_weights=log_w,
        time_weights=time_weights, reserve_amp_sizing=30.0,
        reserve_dense_chunk=8, reserve_grid_block=16,
        reserve_time_nodes=time_nodes,
        reserve_time_resolution_warranted=True,
        reserve_time_check_value=expected, time_guard=guard,
        time_guard_tol_nats=1.0e-3)
    assert float(selected) == pytest.approx(expected, abs=2.0e-12)
    assert bool(usable)
    assert not bool(ledger["accepted_local"])
    assert bool(ledger["reserve_executed"])
    assert bool(ledger["reserve_uses_bandlimited_time"])
    assert not bool(ledger["reserve_uses_native_time"])
    assert bool(ledger["reserve_time_resolution_warranted"])
    assert float(ledger["reserve_time_resolution_error_nats"]) == pytest.approx(
        0.0, abs=2.0e-12)
    assert bool(ledger["reserve_time_resolution_validated"])
    assert bool(ledger["reserve_time_nodes_finite"])
    assert bool(ledger["reserve_time_nodes_increasing"])
    assert bool(ledger["reserve_time_weights_valid"])
    assert bool(ledger["reserve_time_nodes_in_support"])
    assert bool(ledger["reserve_time_nodes_cover_target"])
    assert bool(ledger["reserve_time_guard_validated"])
    assert float(ledger["reserve_time_guard_error"]) <= 1.0e-3
    assert bool(ledger["reserve_time_error_budget_ok"])
    assert bool(ledger["reserve_time_warranted"])
    assert not bool(ledger["reserve_time_failed"])
    assert bool(ledger["sample_retained_after_local_decline"])
    assert bool(ledger["reconciles"])

    _, uncertified_usable, uncertified = (
        AAP.empirical_enrichment_with_exact_reserve(
            guarded, C_B, declined_plan, declined_plan, x_min, x_max,
            reserve_x_grid=x_grid, reserve_log_weights=log_w,
            time_weights=time_weights, reserve_amp_sizing=30.0,
            reserve_dense_chunk=8, reserve_grid_block=16,
            reserve_time_nodes=time_nodes,
            reserve_time_resolution_warranted=False, time_guard=guard,
            time_guard_tol_nats=1.0e-3))
    assert not bool(uncertified_usable)
    assert bool(uncertified["reserve_time_failed"])
    assert not bool(uncertified["sample_retained_after_local_decline"])
    assert bool(uncertified["fallback_required"])
    assert bool(uncertified["reconciles"])

    clipped_nodes = np.linspace(1.0, constants["span"] - 1.0, 65)
    _, clipped_usable, clipped = (
        AAP.empirical_enrichment_with_exact_reserve(
            guarded, C_B, declined_plan, declined_plan, x_min, x_max,
            reserve_x_grid=x_grid, reserve_log_weights=log_w,
            time_weights=time_weights, reserve_amp_sizing=30.0,
            reserve_dense_chunk=8, reserve_grid_block=16,
            reserve_time_nodes=clipped_nodes,
            reserve_time_resolution_warranted=True, time_guard=guard,
            time_guard_tol_nats=1.0e-3))
    assert not bool(clipped_usable)
    assert not bool(clipped["reserve_time_nodes_cover_target"])
    assert bool(clipped["reserve_time_failed"])
    assert bool(clipped["fallback_required"])
    assert bool(clipped["reconciles"])

    accepted_plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, time_reconstruction_certified=False,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True)
    batched_plans = jax.tree.map(
        lambda accepted, declined: jnp.stack((accepted, declined)),
        accepted_plan, declined_plan)
    batched = jax.jit(
        lambda tables, warrants, checks:
        AAP.empirical_enrichment_with_exact_reserve_sequential_batch(
            tables, C_B, batched_plans, batched_plans, x_min, x_max,
            reserve_x_grid=x_grid, reserve_log_weights=log_w,
            time_weights=time_weights, reserve_amp_sizing=30.0,
            reserve_dense_chunk=8, reserve_grid_block=16,
            reserve_time_nodes=time_nodes,
            reserve_time_resolution_warranted=warrants,
            reserve_time_check_value=checks,
            time_guard=guard, time_guard_tol_nats=1.0e-3))
    batch_selected, batch_usable, batch_ledger = batched(
        jnp.stack((guarded, guarded)), jnp.asarray([False, True]),
        jnp.asarray([np.nan, expected]))
    assert np.all(np.asarray(batch_usable))
    assert bool(batch_ledger["accepted_local"][0])
    assert not bool(batch_ledger["reserve_executed"][0])
    assert not bool(batch_ledger["accepted_local"][1])
    assert bool(batch_ledger["reserve_executed"][1])
    assert float(batch_selected[1]) == pytest.approx(expected, abs=2.0e-12)
    assert np.all(np.asarray(
        batch_ledger["reserve_batch_execution_sequential"]))
    assert np.all(np.asarray(batch_ledger["reconciles"]))


def test_native_time_reserve_cannot_be_warranted():
    C_A, C_B, constants = _problem(17)
    x_min, x_max = 0.5, 2.0
    centers = np.asarray([[
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_min + x_max)]])
    transforms = np.asarray([np.diag([
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_max - x_min)])])
    declined_plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, time_reconstruction_certified=False,
        discovery_capacity_ok=False)
    x_grid = np.linspace(x_min, x_max, 33)
    dx = np.empty_like(x_grid)
    dx[1:-1] = 0.5 * (x_grid[2:] - x_grid[:-2])
    dx[0] = x_grid[1] - x_grid[0]
    dx[-1] = x_grid[-1] - x_grid[-2]
    log_w = np.log(dx * x_grid ** -4)
    time_weights = np.ones(C_A.shape[-1])

    _, usable, ledger = AAP.empirical_enrichment_with_exact_reserve(
        C_A, C_B, declined_plan, declined_plan, x_min, x_max,
        reserve_x_grid=x_grid, reserve_log_weights=log_w,
        time_weights=time_weights, reserve_amp_sizing=30.0,
        reserve_dense_chunk=8, reserve_grid_block=16)

    assert not bool(usable)
    assert not bool(ledger["reserve_native_time_warranted"])
    assert not bool(ledger["reserve_time_warranted"])
    assert bool(ledger["reserve_time_failed"])
    assert bool(ledger["fallback_required"])
    assert not bool(ledger["sample_retained_after_local_decline"])
    assert not bool(ledger["selected_nonfinite_is_integration_failure"])
    assert bool(ledger["reconciles"])


def test_cropped_bandlimited_reserve_requires_certified_scout_union(monkeypatch):
    C_A, C_B, constants = _problem(9)
    guard = 2
    guarded = np.pad(C_A, ((0, 0), (0, 0), (guard, guard)), mode="edge")
    x_min, x_max = 0.5, 2.0
    centers = np.asarray([[
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_min + x_max)]])
    transforms = np.asarray([np.diag([
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_max - x_min)])])
    raw_outside_bound = np.log(4.0) - 24.0

    def plan(time_min, time_max, outside_bound=raw_outside_bound,
             certified=True):
        return AAP.make_all_axis_mode_plan(
            centers, max_modes=2, local_transforms=transforms,
            local_radius=1.0, time_reconstruction_certified=False,
            time_outside_log_bound=outside_bound,
            time_outside_bound_certified=certified,
            time_cover_min_sample=time_min,
            time_cover_max_sample=time_max,
            discovery_capacity_ok=False)

    # This contract-level fixture has unit marginalized density on the named
    # compact support.  The external bound is therefore a conservative warrant
    # for its deliberately negligible discarded cells; the test exercises how
    # that warrant is propagated, not how a production scout derives it.
    def fake_exact(table, _norm, _x_grid, _log_w_grid, **_kwargs):
        return jnp.zeros((1, table.shape[-1]), dtype=jnp.float64)

    monkeypatch.setattr(AM, "coefficient_table_distphipsimarg_exact",
                        fake_exact)
    base = plan(2.0, 5.0)
    enriched = plan(3.0, 6.0, raw_outside_bound + 1.0)
    nodes = np.linspace(2.0, 6.0, 9)
    weights = JCORE._simpson_weights(nodes.size, 0.5)
    offset = 3.0
    expected = np.log(4.0) + offset

    def evaluate(base_plan=base, enriched_plan=enriched,
                 time_nodes=nodes, check_value=expected):
        return AAP.empirical_enrichment_with_exact_reserve(
            guarded, C_B, base_plan, enriched_plan, x_min, x_max,
            reserve_x_grid=np.asarray([x_min, x_max]),
            reserve_log_weights=np.zeros(2), time_weights=weights,
            reserve_amp_sizing=30.0, reserve_dense_chunk=8,
            reserve_grid_block=16, reserve_time_nodes=time_nodes,
            reserve_time_resolution_warranted=True,
            reserve_time_check_value=check_value,
            reserve_log_offset=offset, time_guard=guard,
            time_guard_tol_nats=1.0e-3)

    value, usable, ledger = evaluate()
    assert bool(usable)
    assert float(value) == pytest.approx(expected, abs=2.0e-12)
    assert not bool(ledger["reserve_time_nodes_cover_target"])
    assert bool(ledger["reserve_time_plan_cover_finite"])
    assert bool(ledger["reserve_time_nodes_cover_plans"])
    assert bool(ledger["reserve_time_plan_cover_certified"])
    assert bool(ledger["reserve_time_cropped_cover_warranted"])
    assert bool(ledger["reserve_time_interval_warranted"])
    assert float(ledger["reserve_required_time_min_sample"]) == 2.0
    assert float(ledger["reserve_required_time_max_sample"]) == 6.0
    assert float(ledger["reserve_time_outside_log_bound"]) == pytest.approx(
        raw_outside_bound + offset)
    assert float(ledger["reserve_time_tail_margin"]) == pytest.approx(-24.0)
    assert bool(ledger["reserve_time_tail_ok"])
    assert bool(ledger["reserve_time_error_budget_ok"])
    assert bool(ledger["selected_value_is_warranted_reserve"])

    short_nodes = np.linspace(2.0, 5.5, 8)
    short_weights = JCORE._simpson_weights(short_nodes.size,
                                           short_nodes[1] - short_nodes[0])
    _, short_usable, short = AAP.empirical_enrichment_with_exact_reserve(
        guarded, C_B, base, enriched, x_min, x_max,
        reserve_x_grid=np.asarray([x_min, x_max]),
        reserve_log_weights=np.zeros(2), time_weights=short_weights,
        reserve_amp_sizing=30.0, reserve_dense_chunk=8,
        reserve_grid_block=16, reserve_time_nodes=short_nodes,
        reserve_time_resolution_warranted=True,
        reserve_time_check_value=np.log(3.5) + offset,
        reserve_log_offset=offset, time_guard=guard,
        time_guard_tol_nats=1.0e-3)
    assert not bool(short_usable)
    assert not bool(short["reserve_time_nodes_cover_plans"])
    assert not bool(short["reserve_time_interval_warranted"])

    _, uncertified_usable, uncertified = evaluate(
        enriched_plan=plan(3.0, 6.0, raw_outside_bound + 1.0, False))
    assert not bool(uncertified_usable)
    assert not bool(uncertified["reserve_time_plan_cover_certified"])
    assert not bool(uncertified["reserve_time_interval_warranted"])

    loose = raw_outside_bound + 2.0
    _, loose_usable, loose_ledger = evaluate(
        base_plan=plan(2.0, 5.0, loose),
        enriched_plan=plan(3.0, 6.0, loose + 1.0))
    assert not bool(loose_usable)
    assert not bool(loose_ledger["reserve_time_tail_ok"])
    assert bool(loose_ledger["reserve_time_failed"])
    assert bool(loose_ledger["fallback_required"])
    assert bool(loose_ledger["reconciles"])


def test_bandlimited_reserve_rejects_invalid_time_rules(monkeypatch):
    C_A, C_B, constants = _problem(9)
    guard = 2
    guarded = np.pad(C_A, ((0, 0), (0, 0), (guard, guard)), mode="edge")
    n_target = C_A.shape[-1]
    x_min, x_max = 0.5, 2.0
    centers = np.asarray([[
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_min + x_max)]])
    transforms = np.asarray([np.diag([
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_max - x_min)])])
    declined_plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2, local_transforms=transforms,
        local_radius=1.0, time_reconstruction_certified=False,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True,
        discovery_capacity_ok=False)

    def fake_exact(table, _norm, _x_grid, _log_w_grid, **_kwargs):
        return jnp.zeros((1, table.shape[-1]), dtype=jnp.float64)

    monkeypatch.setattr(AM, "coefficient_table_distphipsimarg_exact",
                        fake_exact)
    fine = np.linspace(0.0, n_target - 1.0, 2 * n_target - 1)
    valid_weights = np.ones(fine.size)
    cases = []
    cases.append((fine[::-1], valid_weights,
                  "reserve_time_nodes_increasing"))
    nan_node = fine.copy()
    nan_node[4] = np.nan
    cases.append((nan_node, valid_weights, "reserve_time_nodes_finite"))
    outside = fine.copy()
    outside[0] = -0.25
    cases.append((outside, valid_weights, "reserve_time_nodes_in_support"))
    negative = valid_weights.copy()
    negative[3] = -1.0
    cases.append((fine, negative, "reserve_time_weights_valid"))
    not_finite = valid_weights.copy()
    not_finite[3] = np.nan
    cases.append((fine, not_finite, "reserve_time_weights_valid"))
    cases.append((fine, np.zeros_like(valid_weights),
                  "reserve_time_weights_valid"))
    cases.append((np.linspace(0.25, n_target - 1.25, fine.size),
                  valid_weights, "reserve_time_nodes_cover_target"))
    cases.append((np.arange(n_target, dtype=float), np.ones(n_target),
                  "reserve_time_subsampled"))

    for nodes, weights, failed_field in cases:
        _, usable, ledger = AAP.empirical_enrichment_with_exact_reserve(
            guarded, C_B, declined_plan, declined_plan, x_min, x_max,
            reserve_x_grid=np.asarray([x_min, x_max]),
            reserve_log_weights=np.zeros(2), time_weights=weights,
            reserve_amp_sizing=30.0, reserve_dense_chunk=8,
            reserve_grid_block=16, reserve_time_nodes=nodes,
            reserve_time_resolution_warranted=True, time_guard=guard,
            time_guard_tol_nats=1.0e-3)
        assert not bool(usable)
        assert not bool(ledger[failed_field])
        assert bool(ledger["reserve_time_failed"])
        assert bool(ledger["fallback_required"])
        assert bool(ledger["reconciles"])

    # A REPEATED position is a non-decreasing rule and is accepted: the
    # peak-local time rule (peaklocal_time_reserve) repeats a position where
    # a mode slot is dead or a block is clipped, and the repeat carries no
    # trapezoid weight.  Only a decreasing rule fails the node check.
    duplicate = fine.copy()
    duplicate[4] = duplicate[3]
    _, _, ledger = AAP.empirical_enrichment_with_exact_reserve(
        guarded, C_B, declined_plan, declined_plan, x_min, x_max,
        reserve_x_grid=np.asarray([x_min, x_max]),
        reserve_log_weights=np.zeros(2), time_weights=valid_weights,
        reserve_amp_sizing=30.0, reserve_dense_chunk=8,
        reserve_grid_block=16, reserve_time_nodes=duplicate,
        reserve_time_resolution_warranted=True, time_guard=guard,
        time_guard_tol_nats=1.0e-3)
    assert bool(ledger["reserve_time_nodes_increasing"])

    delta_t = 0.25
    physical_weights = JCORE._simpson_weights(
        fine.size, delta_t / 2.0)
    physical_expected = np.log((n_target - 1) * delta_t)
    value, usable, ledger = AAP.empirical_enrichment_with_exact_reserve(
        guarded, C_B, declined_plan, declined_plan, x_min, x_max,
        reserve_x_grid=np.asarray([x_min, x_max]),
        reserve_log_weights=np.zeros(2), time_weights=physical_weights,
        reserve_amp_sizing=30.0, reserve_dense_chunk=8,
        reserve_grid_block=16, reserve_time_nodes=fine,
        reserve_time_resolution_warranted=True,
        reserve_time_check_value=physical_expected, time_guard=guard,
        time_guard_tol_nats=1.0e-3)
    assert bool(usable)
    assert float(value) == pytest.approx(physical_expected, abs=2.0e-12)
    assert bool(ledger["reserve_time_warranted"])

    _, usable, aggregate = AAP.empirical_enrichment_with_exact_reserve(
        guarded, C_B, declined_plan, declined_plan, x_min, x_max,
        reserve_x_grid=np.asarray([x_min, x_max]),
        reserve_log_weights=np.zeros(2), time_weights=physical_weights,
        reserve_amp_sizing=30.0, reserve_dense_chunk=8,
        reserve_grid_block=16, reserve_time_nodes=fine,
        reserve_time_resolution_warranted=True,
        reserve_time_check_value=physical_expected - 6.0e-4,
        reserve_time_resolution_tol_nats=1.0e-3,
        total_value_error_budget_nats=5.0e-4, time_guard=guard,
        time_guard_tol_nats=1.0e-3)
    assert not bool(usable)
    assert bool(aggregate["reserve_time_resolution_validated"])
    assert bool(aggregate["reserve_time_guard_validated"])
    assert not bool(aggregate["reserve_time_error_budget_ok"])
    assert bool(aggregate["reserve_time_failed"])
    assert bool(aggregate["fallback_required"])
    assert bool(aggregate["reconciles"])

    def table_dependent_exact(table, _norm, _x_grid, _log_w_grid,
                              **_kwargs):
        return jnp.real(table[0, 1])[None, :]

    monkeypatch.setattr(AM, "coefficient_table_distphipsimarg_exact",
                        table_dependent_exact)
    corrupt_guard = guarded.copy()
    corrupt_guard[0, 1, 0] += 100.0
    _, usable, ledger = AAP.empirical_enrichment_with_exact_reserve(
        corrupt_guard, C_B, declined_plan, declined_plan, x_min, x_max,
        reserve_x_grid=np.asarray([x_min, x_max]),
        reserve_log_weights=np.zeros(2), time_weights=physical_weights,
        reserve_amp_sizing=30.0, reserve_dense_chunk=8,
        reserve_grid_block=16, reserve_time_nodes=fine,
        reserve_time_resolution_warranted=True,
        reserve_time_check_value=physical_expected, time_guard=guard,
        time_guard_tol_nats=1.0e-3)
    assert not bool(usable)
    assert float(ledger["reserve_time_guard_error"]) > 1.0e-3
    assert not bool(ledger["reserve_time_guard_validated"])
    assert bool(ledger["reserve_time_failed"])
    assert bool(ledger["fallback_required"])
    assert bool(ledger["reconciles"])


def test_missing_completeness_declines_to_reserve_not_waveform_failure():
    C_A, C_B, constants = _problem(33)
    centers, hessian = _joint_peak(constants)
    transforms, half_widths = AAP.mode_local_geometry(hessian, w_sigma=3.0)
    plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=4, local_transforms=transforms,
        local_radius=3.0, enumeration_complete=False,
        outside_bound_certified=False)
    value, ok, ledger = AAP.all_axis_peak_local_marginalize(
        C_A, C_B, plan, 0.2, 7.0, local_order=3, check_order=5)

    assert np.isfinite(float(value))
    assert not bool(ok)
    assert bool(ledger["fallback_required"])
    assert bool(ledger["decline_incomplete"])
    assert not bool(ledger["decline_is_waveform_failure"])
    assert bool(ledger["reconciles"])


def test_two_guard_local_integral_validates_same_target_window():
    C_A, C_B, constants = _problem(33)
    guard = 32
    support_time = np.arange(-guard, C_A.shape[-1] + guard, dtype=float)
    guarded = np.zeros(C_A.shape[:-1] + (support_time.size,), dtype=np.complex128)
    guarded[0, 1] = (constants["k0"] - constants["kt"]
                     * np.cos(2.0 * np.pi * support_time / constants["span"]))
    guarded[2, 1] = 0.5 * constants["kp"]
    guarded[0, 0] = 0.5 * constants["ku"]
    guarded[0, 2] = 0.5 * constants["ku"]
    np.testing.assert_allclose(guarded[..., guard:-guard], C_A, atol=1e-14)

    centers, hessian = _joint_peak(constants)
    transforms, _ = AAP.mode_local_geometry(hessian, w_sigma=3.0)
    plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=4, local_transforms=transforms,
        local_radius=3.0, outside_bound_certified=False,
        time_reconstruction_certified=True,
        time_outside_log_bound=-np.inf,
        time_outside_bound_certified=True)
    value, ok, ledger = AAP.all_axis_peak_local_marginalize(
        guarded, C_B, plan, 0.2, 7.0,
        local_order=7, check_order=11, time_guard=guard,
        time_guard_tol_nats=1.0e-3)

    assert np.isfinite(float(value))
    assert not bool(ok)  # outside mass is deliberately still unwarranted
    assert bool(ledger["time_guard_validated"])
    assert bool(ledger["time_reconstruction_warranted"])
    assert int(ledger["time_guard"]) == 32
    assert int(ledger["time_guard_inner"]) == 16
    assert float(ledger["time_guard_error"]) <= 1.0e-3
    assert int(ledger["n_guard_local_evaluations_hi"]) == 2 * 11 ** 4
    assert int(ledger["n_total_local_evaluations_hi"]) == 4 * 11 ** 4
    assert int(ledger["n_total_time_frequency_terms_hi"]) == (
        int(ledger["n_time_frequency_terms_hi"])
        + int(ledger["n_guard_time_frequency_terms_hi"]))
    assert int(ledger["workspace_bytes_peak_bound_hi"]) == max(
        int(ledger["workspace_bytes_hi"]),
        int(ledger["workspace_bytes_guard_hi"]))
    assert bool(ledger["decline_incomplete"])
    assert bool(ledger["reconciles"])

    _, empirical_ok, empirical_ledger = (
        AAP.empirical_enrichment_marginalize(
            guarded, C_B, plan, plan, 0.2, 7.0,
            base_order=7, base_check_order=9,
            enriched_order=9, enriched_check_order=11,
            convergence_tol_nats=1.0e-3, time_guard=guard,
            time_guard_tol_nats=1.0e-3,
            total_value_error_budget_nats=1.0e-2))
    assert bool(empirical_ok)
    assert float(empirical_ledger["error_score_base_time_guard_nats"]) == (
        pytest.approx(float(empirical_ledger["base_time_guard_error"])))
    assert float(empirical_ledger[
        "error_score_enriched_time_guard_nats"]) == pytest.approx(
            float(empirical_ledger["enriched_time_guard_error"]))
    assert float(empirical_ledger["error_score_base_time_guard_nats"]) > 0.0
    assert bool(empirical_ledger["value_error_budget_ok"])
    assert bool(empirical_ledger["reconciles"])

    # Corrupt only support discarded by the inner guard.  Integer target
    # samples remain unchanged, but the outer Fourier seam rings into the local
    # nodes; the two-guard comparison must see it rather than blessing exact
    # retained-sample parity or trusting the plan's stale external warrant.
    bad = guarded.copy()
    bad[0, 1, :guard // 2] += 1.0e4
    _, _, bad_ledger = AAP.all_axis_peak_local_marginalize(
        bad, C_B, plan, 0.2, 7.0,
        local_order=7, check_order=11, time_guard=guard,
        time_guard_tol_nats=1.0e-3)
    assert not bool(bad_ledger["time_guard_validated"])
    assert not bool(bad_ledger["time_reconstruction_warranted"])
    assert float(bad_ledger["time_guard_error"]) > 1.0e-3
    _, bad_empirical_ok, bad_empirical_ledger = (
        AAP.empirical_enrichment_marginalize(
            bad, C_B, plan, plan, 0.2, 7.0,
            base_order=7, base_check_order=9,
            enriched_order=9, enriched_check_order=11,
            convergence_tol_nats=1.0e-3, time_guard=guard,
            time_guard_tol_nats=1.0e-3,
            total_value_error_budget_nats=1.0e-15))
    assert not bool(bad_empirical_ok)
    assert bool(bad_empirical_ledger["decline_time_reconstruction"])
    assert not bool(bad_empirical_ledger["decline_error_budget"])
    assert bool(bad_empirical_ledger["reconciles"])


def test_certified_omitted_mass_can_cover_an_incomplete_root_report():
    C_A, C_B, constants = _problem(33)
    scale = 0.1
    C_A *= scale
    constants = dict(constants)
    for name in ("k0", "kt", "kp", "ku"):
        constants[name] *= scale
    x_min, x_max = 0.5, 2.0
    truth = _analytic_log_integral(constants, x_min, x_max)
    # One diagonal affine region is exactly the complete t x phi x u x x
    # support.  Its complement has zero measure, so -inf is an actual outside
    # integral bound rather than a fabricated tail assertion.
    centers = np.asarray([[
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_min + x_max)]])
    transforms = np.asarray([np.diag([
        constants["span"] / 2.0, np.pi, np.pi,
        0.5 * (x_max - x_min)])])
    plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=2,
        local_transforms=transforms, local_radius=1.0,
        outside_log_bound=-np.inf,
        # This models an incomplete/degenerate root report.  The exact full-
        # support cover, not the root count, owns the science error budget.
        enumeration_complete=False, outside_bound_certified=True,
        # This fixture is itself an exact finite reflected cosine series.  Real
        # packets need the independent two-guard convergence warrant.
        time_reconstruction_certified=True)
    value, ok, ledger = AAP.all_axis_peak_local_marginalize(
        C_A, C_B, plan, x_min, x_max, local_order=13, check_order=19,
        quadrature_tol_nats=2.0e-5)

    assert bool(ok)
    assert float(value) == pytest.approx(truth, abs=2.0e-5)
    assert not bool(ledger["enumeration_complete"])
    assert bool(ledger["outside_bound_certified"])
    assert not bool(ledger["fallback_required"])
    assert bool(ledger["reconciles"])


@pytest.mark.parametrize("mutation", ("nan", "upper", "negative_diagonal"))
def test_plan_rejects_geometry_that_does_not_match_the_integrated_region(mutation):
    _, _, constants = _problem(33)
    centers, hessian = _joint_peak(constants)
    transforms, _ = AAP.mode_local_geometry(hessian, w_sigma=3.0)
    if mutation == "nan":
        transforms[0, 0, 0] = np.nan
    elif mutation == "upper":
        transforms[0, 0, 1] = 1.0e-4
    else:
        transforms[0, 0, 0] *= -1.0
    with pytest.raises(ValueError):
        AAP.make_all_axis_mode_plan(
            centers, max_modes=4, local_transforms=transforms,
            local_radius=3.0)


def test_fixed_plan_is_transform_compatible_without_claiming_derivative_accuracy():
    C_A, C_B, constants = _problem(33)
    centers, hessian = _joint_peak(constants)
    transforms, half_widths = AAP.mode_local_geometry(hessian, w_sigma=3.0)
    plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=4, local_transforms=transforms,
        local_radius=3.0)
    C_A = jnp.asarray(C_A)
    C_B = jnp.asarray(C_B)

    @jax.jit
    def value(scale):
        answer, _, _ = AAP.all_axis_peak_local_marginalize(
            scale * C_A, C_B, plan, 0.2, 7.0,
            local_order=3, check_order=5)
        return answer

    got = value(1.0)
    gradient = jax.grad(value)(1.0)
    hessian_value = jax.hessian(value)(1.0)
    assert np.all(np.isfinite(np.asarray([got, gradient, hessian_value])))
    _, _, ledger = AAP.all_axis_peak_local_marginalize(
        C_A, C_B, plan, 0.2, 7.0, local_order=3, check_order=5)
    assert bool(ledger["fixed_plan_autodiff_only"])
    assert not bool(ledger["derivative_warrant_certified"])


def test_padded_fixed_plan_survives_outer_vmap():
    C_A, C_B, constants = _problem(33)
    centers, hessian = _joint_peak(constants)
    transforms, _ = AAP.mode_local_geometry(hessian, w_sigma=3.0)
    plan = AAP.make_all_axis_mode_plan(
        centers, max_modes=4, local_transforms=transforms,
        local_radius=3.0)

    def value(table):
        return AAP.all_axis_peak_local_marginalize(
            table, C_B, plan, 0.2, 7.0,
            local_order=3, check_order=5)[0]

    batch = jnp.asarray(np.stack((C_A, 1.01 * C_A)))
    result = jax.jit(jax.vmap(value))(batch)
    assert result.shape == (2,)
    assert np.all(np.isfinite(np.asarray(result)))
