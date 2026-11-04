"""Fixed-shape multi-peak marginalization over time, polarization, phase and distance.

This module is the device-evaluation half of the all-variable peak-local design.
It deliberately separates three jobs which must not be conflated:

* the compact norm table produced by the upstream packed ``U,V`` contraction is
  summarized once and used by the host planner to rank candidate basins;
* JAX gradients and Hessians refine supplied starts and size local boxes;
* a separate acceptance layer decides whether the local integral is usable,
  either from a formal omitted-mass warrant or from an explicitly empirical
  stronger discovery/quadrature enrichment with conservative reserve.  Optimizer
  convergence alone is never treated as proof that every mode was found.

The integration kernel consumes a padded :class:`AllAxisModePlan`, so its live
workspace is ``O(local_order**4)`` and is independent of the dense time/angle/
distance resolutions.  Time is reconstructed from the reflected primitive only
at the local nodes.  The two angular axes use the exact finite Fourier tables, and
distance uses the exact quadratic likelihood with the volumetric ``x**-4``
Jacobian.  Modes are streamed through ``lax.scan``; no ``mode x local-grid`` tensor
is retained under nested JIT/AD/vmap transformations.  Values are unnormalized:
the caller-provided ``log_normalization`` must include the time-sample Jacobian
and normalized time, angle, and distance priors appropriate to its application.

This is an explicit prototype seam, not a sampler policy.  ``ok=False`` means the
caller must evaluate its dense/exact reserve and keep the sample.  It never means
waveform failure and the diagnostic local value must not be substituted silently.
The primitive ``ok`` is only a scalar-value usability gate: the outside-mass
bound may be certified, but the nested quadrature comparison is validated rather
than a formal error bound.  :func:`empirical_enrichment_marginalize` implements
the more practical one-step operational gate and labels its non-rigorous basis.
Although the fixed-shape kernel is compatible with outer JIT/AD,
differentiating it holds
the host plan and its regions fixed and therefore differentiates the truncated
local integral.  A production gradient/Hessian consumer needs a separate omitted-
derivative warrant; this prototype does not claim one.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .time_first_peaklocal import (_evaluate_time_spectrum,
                                   _time_primitive_spectrum,
                                   spectral_time_derivative_bound)


__all__ = [
    "AllAxisModePlan",
    "UVHarmonicSummary",
    "JointStartPlan",
    "DeviceJointStartPlan",
    "summarize_uv_norm_table",
    "rank_time_starts_from_uv",
    "rank_joint_starts_from_uvq",
    "rank_joint_starts_from_uvq_device",
    "combine_device_start_plans",
    "algebraic_angle_starts_from_uv",
    "refine_all_axis_starts",
    "select_refined_modes",
    "mode_local_geometry",
    "make_all_axis_mode_plan",
    "make_all_axis_mode_plan_device",
    "make_all_axis_mode_plan_pair_device",
    "all_axis_peak_local_marginalize",
    "empirical_enrichment_marginalize",
    "empirical_enrichment_with_exact_reserve",
    "empirical_enrichment_with_exact_reserve_sequential_batch",
]


class AllAxisModePlan(NamedTuple):
    """Padded host plan consumed by the fixed-shape device kernel.

    Coordinates are ``(time_sample, phi_ref, u=2*psi, x=Dref/D)``.
    ``outside_log_bound`` bounds the *unnormalized* integral outside the union
    of the exact transformed regions
    ``center + local_transform @ [-local_radius,local_radius]^4`` (with the two
    angular coordinates interpreted periodically).  A finite value is
    correctness-bearing only when ``outside_bound_certified`` is true.  The
    axis-aligned ``half_widths`` are derived conservative enclosures used only
    for support and disjointness checks; they never define the certified cover.
    ``time_reconstruction_certified`` is a separate guard/seam warrant.  Real
    unguarded captures must leave it false; an outside-mass certificate cannot
    certify the noninteger reflected-time reconstruction inside a region.
    ``time_outside_log_bound`` has a deliberately narrower meaning: it bounds
    the full angle/distance integral in time cells discarded before basin
    localization.  ``time_cover_min_sample`` and ``time_cover_max_sample``
    enclose every retained scout cell associated with that bound, permitting a
    reserve to integrate the contiguous enclosing interval without silently
    dropping a retained time basin.  Nonfinite endpoints mean that no such
    cropped reserve warrant is available.  The time bound is not a bound on
    missing angular modes inside retained
    cells and can only augment, never replace, the global outside-cover warrant.
    ``enumeration_complete`` is a separate
    diagnostic statement about the supplied root set.  It is deliberately not
    an acceptance requirement: a missed algebraic root is scientifically
    harmless when the independent bound proves that *all* mass outside the
    integrated regions fits the error budget.  Algebraic roots, optimizer
    starts, and an outside bound have different failure modes and remain
    separate here.  ``discovery_capacity_ok`` freezes whether the upstream
    bounded start portfolio fit without truncation; the empirical gate declines
    rather than trusting a caller-supplied boolean at evaluation time.
    ``boundary_maximum_pinned`` records that a live start with a competitive
    value ended on the time or distance boundary of the support after
    refinement and was rejected by the stationarity filter.  Such a start is
    a constrained maximum the local integral does not cover: on a synthetic
    window whose exact lnL(t) peaks at the first sample, the plans that
    ignored it accepted a value 22.7 nat below the exact reserve with every
    other diagnostic passing.  The gate declines on it.
    """

    centers: jax.Array
    half_widths: jax.Array
    local_transforms: jax.Array
    local_radius: jax.Array
    live: jax.Array
    outside_log_bound: jax.Array
    enumeration_complete: jax.Array
    outside_bound_certified: jax.Array
    time_reconstruction_certified: jax.Array
    time_outside_log_bound: jax.Array
    time_outside_bound_certified: jax.Array
    time_cover_min_sample: jax.Array
    time_cover_max_sample: jax.Array
    boxes_disjoint: jax.Array
    discovery_capacity_ok: jax.Array
    boundary_maximum_pinned: jax.Array


class UVHarmonicSummary(NamedTuple):
    """Compact structural summary of the norm table derived from ``U,V``.

    ``C_B`` is the exact harmonic table already produced upstream from the
    packed self terms.  This class does not claim to repeat or count that
    contraction.  The lower/upper and derivative entries are triangle-
    inequality bounds, not fits.  They are host-planning data and should be
    cached outside JIT/vmap.
    """

    C_B: np.ndarray
    b_lower: float
    b_upper: float
    phi_derivative_bound: float
    u_derivative_bound: float
    time_invariant: bool
    time_max_deviation: float
    summary_build_count: int
    input_harmonic_coefficients: int


class JointStartPlan(NamedTuple):
    """Bounded U,V/Q-informed starts for joint four-axis refinement.

    The angular lattice is sized from the exact harmonic orders and an explicit
    oversampling factor, never from SNR.  It is a targeting device rather than
    a completeness proof.  ``capacity_ok`` is false instead of silently
    discarding excess candidates; the empirical controller must then enrich or
    use its reserve.
    """

    starts: np.ndarray
    scores: np.ndarray
    time_starts: np.ndarray
    time_profile: np.ndarray
    n_phi_lattice: int
    n_u_lattice: int
    n_lattice_evaluations: int
    n_exact_symmetry_shifts: int
    n_candidates_before_cap: int
    capacity_ok: bool


class DeviceJointStartPlan(NamedTuple):
    """Fixed-capacity U,V/Q basin portfolio produced inside JAX.

    Unlike :class:`JointStartPlan`, every array has a static leading dimension
    and can therefore cross ``jit`` and ``vmap`` boundaries.  The angular
    lattice is fixed by the finite harmonic degrees, not by SNR; it is a cheap
    basin-placement device, not an integration grid or completeness proof.
    ``capacity_ok`` is false whenever more local lattice maxima exist than fit
    in ``starts``.  Such a row must enrich or execute the exact reserve.
    The time-cover endpoints enclose every retained scout cell and travel with
    the corresponding discarded-cell integral bound.
    """

    starts: jax.Array
    scores: jax.Array
    live: jax.Array
    n_lattice_candidates_before_symmetry: jax.Array
    n_candidates_before_cap: jax.Array
    capacity_ok: jax.Array
    n_phi_lattice: jax.Array
    n_u_lattice: jax.Array
    n_time_lattice: jax.Array
    n_retained_time_samples: jax.Array
    n_lattice_evaluations: jax.Array
    n_time_scout_evaluations: jax.Array
    n_time_cells_retained: jax.Array
    n_time_nodes_retained: jax.Array
    time_outside_log_bound: jax.Array
    time_cover_min_sample: jax.Array
    time_cover_max_sample: jax.Array
    time_scout_peak_lower: jax.Array
    time_cover_certified: jax.Array
    time_capacity_ok: jax.Array
    norm_nonnegative: jax.Array
    n_exact_symmetry_shifts: jax.Array


def _kp_weights_numpy(n):
    out = np.ones(int(n), dtype=float)
    out[1:] = 2.0
    return out


def summarize_uv_norm_table(C_B_t, *, invariance_atol=1.0e-10):
    """Collapse the ``U,V``-derived norm table and form exact harmonic bounds.

    ``C_B_t`` may be ``(KP,2KS+1)`` or the historical
    ``(KP,2KS+1,Ntime)`` table.  Ordinary (non-rotation) ILE has a
    time-independent norm; the latter representation repeats it at every time.
    Arrival-time-dependent input is reported in ``time_invariant`` and must make
    a peak-local plan decline rather than being averaged away.
    """
    table = np.asarray(C_B_t, dtype=np.complex128)
    if table.ndim == 2:
        base = table
        deviation = 0.0
    elif table.ndim == 3:
        base = table[..., 0]
        deviation = float(np.max(np.abs(table - base[..., None])))
    else:
        raise ValueError("C_B_t must have shape (KP,2KS+1[,Ntime])")
    scale = max(1.0, float(np.max(np.abs(base))))
    invariant = bool(np.isfinite(deviation)
                     and deviation <= float(invariance_atol) * scale)

    kp = np.arange(base.shape[0], dtype=float)[:, None]
    ks_max = (base.shape[1] - 1) // 2
    ks = np.arange(-ks_max, ks_max + 1, dtype=float)[None, :]
    weight = _kp_weights_numpy(base.shape[0])[:, None]
    magnitude = weight * np.abs(base)
    centre = float(base[0, ks_max].real)
    remainder = float(np.sum(magnitude) - abs(base[0, ks_max]))
    # B=<h|h> is non-negative.  Combining that identity with the harmonic
    # triangle inequality makes the lower bound tighter but never optimistic.
    b_lower = max(0.0, centre - remainder)
    b_upper = abs(centre) + remainder
    m_phi = float(np.sum(magnitude * np.abs(kp)))
    m_u = float(np.sum(magnitude * np.abs(ks)))
    return UVHarmonicSummary(
        np.ascontiguousarray(base), b_lower, b_upper, m_phi, m_u,
        invariant, deviation, 1, int(table.size))


def rank_time_starts_from_uv(C_A_t, uv_summary, x_min, x_max, *,
                             max_starts=16, min_separation=2):
    """Rank time basins with a true ``U,V``-informed likelihood envelope.

    For every retained time sample, ``A_upper=sum w_k |C_A|`` bounds the data
    term over both angles.  ``uv_summary.b_lower`` bounds the norm from below,
    so maximizing ``x*A_upper - B_lower*x**2/2`` on the physical distance
    interval gives an upper envelope.  The volumetric ``-4 log(x)`` term is
    separately bounded at ``x_min``.  This ranks starts cheaply; it does *not*
    certify that unselected time cells are negligible.  That remains the
    outside-mass warrant in :class:`AllAxisModePlan`.
    """
    C_A_t = np.asarray(C_A_t, dtype=np.complex128)
    if C_A_t.ndim != 3:
        raise ValueError("C_A_t must have shape (KP,2KS+1,Ntime)")
    if not isinstance(uv_summary, UVHarmonicSummary):
        raise TypeError("uv_summary must come from summarize_uv_norm_table")
    if not uv_summary.time_invariant:
        raise ValueError("arrival-time-dependent U,V norm cannot be collapsed")
    x_min, x_max = float(x_min), float(x_max)
    if not (0.0 < x_min < x_max):
        raise ValueError("need 0 < x_min < x_max")
    max_starts = int(max_starts)
    min_separation = int(min_separation)
    if max_starts < 1 or min_separation < 0:
        raise ValueError("invalid start-count policy")

    weight = _kp_weights_numpy(C_A_t.shape[0])[:, None, None]
    a_upper = np.sum(weight * np.abs(C_A_t), axis=(0, 1))
    if uv_summary.b_lower > 0.0:
        x_star = np.clip(a_upper / uv_summary.b_lower, x_min, x_max)
    else:
        x_star = np.full_like(a_upper, x_max)
    envelope = (x_star * a_upper
                - 0.5 * uv_summary.b_lower * np.square(x_star)
                - 4.0 * np.log(x_min))

    # Endpoints are legitimate boundary basins.  Interior starts are drawn only
    # from local maxima, then ranked by the structural upper envelope.
    is_peak = np.ones(envelope.size, dtype=bool)
    if envelope.size > 2:
        is_peak[1:-1] = ((envelope[1:-1] >= envelope[:-2])
                         & (envelope[1:-1] >= envelope[2:]))
    candidates = np.flatnonzero(is_peak)
    candidates = candidates[np.argsort(envelope[candidates])[::-1]]
    selected = []
    for index in candidates:
        if all(abs(int(index) - old) > min_separation for old in selected):
            selected.append(int(index))
        if len(selected) == max_starts:
            break
    if not selected:
        selected = [int(np.argmax(envelope))]
    return np.asarray(selected, dtype=np.int32), envelope


def _distance_profile_numpy(A, B, x_min, x_max):
    """Maximize ``x*A-x**2*B/2-4log(x)`` on a finite interval."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    scale = max(1.0, float(np.max(np.abs(B))))
    if np.min(B) < -1.0e-9 * scale:
        raise ValueError("U,V norm table is negative on the planning lattice")
    B = np.maximum(B, 0.0)
    x0 = np.full_like(A, float(x_min))
    x1 = np.full_like(A, float(x_max))

    def value(x):
        return x * A - 0.5 * B * x * x - 4.0 * np.log(x)

    v0, v1 = value(x0), value(x1)
    choose_hi = v1 > v0
    best_x = np.where(choose_hi, x1, x0)
    best_v = np.where(choose_hi, v1, v0)
    discriminant = A * A - 16.0 * B
    valid = (B > 0.0) & (discriminant >= 0.0)
    root = np.where(
        valid,
        (A + np.sqrt(np.maximum(discriminant, 0.0)))
        / np.where(B > 0.0, 2.0 * B, 1.0),
        x0)
    valid &= (root >= float(x_min)) & (root <= float(x_max))
    root_safe = np.where(valid, root, x0)
    root_value = value(root_safe)
    improve = valid & (root_value > best_v)
    return (np.where(improve, root_value, best_v),
            np.where(improve, root_safe, best_x))


def _harmonic_lattice(table, n_phi, n_u):
    """Evaluate a stored real Fourier half-plane on a periodic lattice."""
    table = np.asarray(table, dtype=np.complex128)
    kp = np.arange(table.shape[0], dtype=float)
    ks = np.arange(-(table.shape[1] - 1) // 2,
                   (table.shape[1] - 1) // 2 + 1, dtype=float)
    weight = _kp_weights_numpy(table.shape[0])
    phi = 2.0 * np.pi * np.arange(int(n_phi), dtype=float) / int(n_phi)
    u = 2.0 * np.pi * np.arange(int(n_u), dtype=float) / int(n_u)
    ep = weight[None, :] * np.exp(1j * phi[:, None] * kp[None, :])
    eu = np.exp(1j * u[:, None] * ks[None, :])
    if table.ndim == 2:
        result = np.einsum("pk,uq,kq->pu", ep, eu, table,
                           optimize=True).real
    elif table.ndim == 3:
        result = np.einsum("pk,uq,kqt->put", ep, eu, table,
                           optimize=True).real
    else:
        raise ValueError("harmonic table must have shape (KP,2KS+1[,Ntime])")
    return phi, u, result


def _exact_angular_translation_symmetries(C_A_t, C_B, *, rtol=1.0e-10):
    """Find common coefficient-certified translations on a degree grid."""
    C_A_t = np.asarray(C_A_t, dtype=np.complex128)
    C_B = np.asarray(C_B, dtype=np.complex128)
    k_phi = max(C_A_t.shape[0] - 1, C_B.shape[0] - 1)
    k_u = max((C_A_t.shape[1] - 1) // 2, (C_B.shape[1] - 1) // 2)
    n_phi = max(1, 2 * k_phi)
    n_u = max(1, 2 * k_u)

    def invariant(table, dphi, du):
        kp = np.arange(table.shape[0], dtype=float)[:, None]
        ks = np.arange(-(table.shape[1] - 1) // 2,
                       (table.shape[1] - 1) // 2 + 1, dtype=float)[None, :]
        phase = np.exp(1j * (kp * dphi + ks * du))
        if table.ndim == 3:
            phase = phase[..., None]
        scale = max(float(np.max(np.abs(table))), 1.0)
        return (float(np.max(np.abs(table * (phase - 1.0))))
                <= float(rtol) * scale)

    shifts = []
    for i in range(n_phi):
        dphi = 2.0 * np.pi * i / n_phi
        for j in range(n_u):
            du = 2.0 * np.pi * j / n_u
            if invariant(C_A_t, dphi, du) and invariant(C_B, dphi, du):
                shifts.append((dphi, du))
    return np.asarray(shifts, dtype=float).reshape((-1, 2))


def rank_joint_starts_from_uvq(
        C_A_t, uv_summary, x_min, x_max, *, time_guard=0,
        max_time_starts=4, max_starts=64, min_time_separation=2,
        angular_oversample=2):
    """Build a bounded distance-following start set from U,V and Q tables.

    The exact U,V norm harmonics and Q data harmonics are evaluated on a lattice
    sized by their finite polynomial degrees.  Distance is profiled analytically
    at each lattice point.  Only angular local maxima at the highest-ranked time
    basins become starts, followed by coefficient-certified translation orbits.

    This procedure performs no sampled ``delta lnL`` pruning: a legitimate peak
    becomes arbitrarily narrow with SNR and may lie between coarse nodes.  The
    lattice is for basin placement, not likelihood integration.  Increasing
    ``angular_oversample`` and the bounded capacities defines the independent
    enrichment used by the operational convergence gate.
    """
    C_A_t = np.asarray(C_A_t, dtype=np.complex128)
    if not isinstance(uv_summary, UVHarmonicSummary):
        raise TypeError("uv_summary must come from summarize_uv_norm_table")
    _validate_tables(C_A_t, uv_summary.C_B)
    if not uv_summary.time_invariant:
        raise ValueError("arrival-time-dependent U,V norm cannot be collapsed")
    if not (0.0 < float(x_min) < float(x_max)):
        raise ValueError("need 0 < x_min < x_max")
    time_guard = int(time_guard)
    n_time = C_A_t.shape[-1] - 2 * time_guard
    if time_guard < 0 or n_time < 2:
        raise ValueError("time_guard must leave at least two target samples")
    if min(int(max_time_starts), int(max_starts)) < 1:
        raise ValueError("start capacities must be positive")
    angular_oversample = int(angular_oversample)
    if angular_oversample < 1:
        raise ValueError("angular_oversample must be positive")
    target = (C_A_t if time_guard == 0
              else C_A_t[..., time_guard:-time_guard])

    k_phi = uv_summary.C_B.shape[0] - 1
    k_u = (uv_summary.C_B.shape[1] - 1) // 2
    n_phi = max(9, 2 * angular_oversample * k_phi + 1)
    n_u = max(9, 2 * angular_oversample * k_u + 1)
    phi, u, A = _harmonic_lattice(target, n_phi, n_u)
    _, _, B = _harmonic_lattice(uv_summary.C_B, n_phi, n_u)
    profile, x_best = _distance_profile_numpy(
        A, B[..., None], float(x_min), float(x_max))
    time_profile = np.max(profile, axis=(0, 1))

    peak_t = np.ones(time_profile.size, dtype=bool)
    if time_profile.size > 2:
        peak_t[1:-1] = ((time_profile[1:-1] >= time_profile[:-2])
                        & (time_profile[1:-1] >= time_profile[2:]))
    candidates_t = np.flatnonzero(peak_t)
    candidates_t = candidates_t[np.argsort(time_profile[candidates_t])[::-1]]
    time_starts = []
    for time_index in candidates_t:
        if all(abs(int(time_index) - old) > int(min_time_separation)
               for old in time_starts):
            time_starts.append(int(time_index))
        if len(time_starts) == int(max_time_starts):
            break
    if not time_starts:
        time_starts = [int(np.argmax(time_profile))]

    raw = []
    for time_index in time_starts:
        surface = profile[..., time_index]
        local = np.ones(surface.shape, dtype=bool)
        for dphi in (-1, 0, 1):
            for du in (-1, 0, 1):
                if dphi or du:
                    local &= surface >= np.roll(
                        np.roll(surface, dphi, axis=0), du, axis=1)
        angular_indices = np.argwhere(local)
        if not len(angular_indices):
            angular_indices = np.asarray([
                np.unravel_index(np.argmax(surface), surface.shape)])
        for iphi, iu in angular_indices:
            raw.append((
                float(surface[iphi, iu]),
                (float(time_index), float(phi[iphi]), float(u[iu]),
                 float(x_best[iphi, iu, time_index]))))

    shifts = _exact_angular_translation_symmetries(
        target, uv_summary.C_B)
    orbit = []
    for score, start in raw:
        for dphi, du in shifts:
            candidate = (
                start[0], (start[1] + dphi) % (2.0 * np.pi),
                (start[2] + du) % (2.0 * np.pi), start[3])
            if not any(
                    abs(candidate[0] - old[1][0]) <= 1.0e-10
                    and _periodic_distance(candidate[1], old[1][1]) <= 1.0e-10
                    and _periodic_distance(candidate[2], old[1][2]) <= 1.0e-10
                    and abs(candidate[3] - old[1][3]) <= 1.0e-10
                    for old in orbit):
                orbit.append((score, candidate))
    orbit.sort(key=lambda item: item[0], reverse=True)
    n_candidates = len(orbit)
    capacity_ok = n_candidates <= int(max_starts)
    kept = orbit[:int(max_starts)]
    return JointStartPlan(
        np.asarray([item[1] for item in kept], dtype=float).reshape((-1, 4)),
        np.asarray([item[0] for item in kept], dtype=float),
        np.asarray(time_starts, dtype=np.int32), time_profile,
        int(n_phi), int(n_u), int(n_phi * n_u * n_time),
        int(len(shifts)), int(n_candidates), bool(capacity_ok))


def _harmonic_lattice_device(table, n_phi, n_u):
    """JAX counterpart of :func:`_harmonic_lattice` for fixed-shape plans."""
    table = jnp.asarray(table, dtype=jnp.complex128)
    kp = jnp.arange(table.shape[0], dtype=jnp.float64)
    ks = jnp.arange(-(table.shape[1] - 1) // 2,
                    (table.shape[1] - 1) // 2 + 1, dtype=jnp.float64)
    weight = jnp.where(kp == 0.0, 1.0, 2.0)
    phi = (2.0 * jnp.pi / float(n_phi)
           * jnp.arange(int(n_phi), dtype=jnp.float64))
    u = (2.0 * jnp.pi / float(n_u)
         * jnp.arange(int(n_u), dtype=jnp.float64))
    ep = weight[None, :] * jnp.exp(1j * phi[:, None] * kp[None, :])
    eu = jnp.exp(1j * u[:, None] * ks[None, :])
    if table.ndim == 2:
        field = jnp.einsum("pk,uq,kq->pu", ep, eu, table).real
    elif table.ndim == 3:
        field = jnp.einsum("pk,uq,kqt->put", ep, eu, table).real
    else:
        raise ValueError("harmonic table must have shape (KP,2KS+1[,Ntime])")
    return phi, u, field


def _distance_profile_device(A, B, x_min, x_max):
    """JAX support-aware distance maximum used only to rank basins."""
    A = jnp.asarray(A, dtype=jnp.float64)
    B = jnp.asarray(B, dtype=jnp.float64)
    B_safe = jnp.maximum(B, 0.0)
    x0 = jnp.full_like(A, float(x_min))
    x1 = jnp.full_like(A, float(x_max))

    def value(x):
        return x * A - 0.5 * B_safe * x * x - 4.0 * jnp.log(x)

    v0, v1 = value(x0), value(x1)
    choose_hi = v1 > v0
    best_x = jnp.where(choose_hi, x1, x0)
    best_v = jnp.where(choose_hi, v1, v0)
    discriminant = A * A - 16.0 * B_safe
    valid = (B_safe > 0.0) & (discriminant >= 0.0)
    root = ((A + jnp.sqrt(jnp.maximum(discriminant, 0.0)))
            / jnp.where(B_safe > 0.0, 2.0 * B_safe, 1.0))
    valid &= (root >= float(x_min)) & (root <= float(x_max))
    root_safe = jnp.where(valid, root, x0)
    root_value = value(root_safe)
    improve = valid & (root_value > best_v)
    return (jnp.where(improve, root_value, best_v),
            jnp.where(improve, root_safe, best_x))


def _distance_upper_profile_device(A_upper, B_lower, x_min, x_max):
    """Maximize a likelihood upper envelope with a possibly negative B bound."""
    A_upper = jnp.asarray(A_upper, dtype=jnp.float64)
    B_lower = jnp.asarray(B_lower, dtype=jnp.float64)
    x0 = jnp.full_like(A_upper, float(x_min))
    x1 = jnp.full_like(A_upper, float(x_max))

    def value(x):
        return x * A_upper - 0.5 * B_lower * x * x - 4.0 * jnp.log(x)

    v0, v1 = value(x0), value(x1)
    best = jnp.maximum(v0, v1)
    discriminant = A_upper * A_upper - 16.0 * B_lower
    valid = (B_lower > 0.0) & (discriminant >= 0.0)
    root = ((A_upper + jnp.sqrt(jnp.maximum(discriminant, 0.0)))
            / jnp.where(B_lower > 0.0, 2.0 * B_lower, 1.0))
    valid &= (root >= float(x_min)) & (root <= float(x_max))
    return jnp.where(valid, jnp.maximum(best, value(root)), best)


def _time_cell_cover_device(
        C_A_t, C_B, x_min, x_max, *, time_guard, keep_nats,
        scout_size):
    """Certify omitted full-angle/distance mass for discarded time cells.

    A constant-size angular scout supplies only a lower reference used to choose
    cells.  Correctness comes instead from a spectral derivative bound on every
    Q coefficient, the angular triangle inequality, and a triangle lower bound
    on the U,V norm polynomial.  Thus a poor scout can retain extra cells but
    cannot make the omitted-time integral optimistic.
    """
    time_guard = int(time_guard)
    scout_size = int(scout_size)
    target = (C_A_t if time_guard == 0
              else C_A_t[..., time_guard:-time_guard])
    n_time = target.shape[-1]
    _, _, scout_A = _harmonic_lattice_device(
        target, scout_size, scout_size)
    _, _, scout_B = _harmonic_lattice_device(
        C_B, scout_size, scout_size)
    scout_profile, _ = _distance_profile_device(
        scout_A, scout_B[..., None], float(x_min), float(x_max))
    scout_peak_lower = jnp.max(scout_profile)

    kp_weight = jnp.where(
        jnp.arange(C_A_t.shape[0]) == 0, 1.0, 2.0)
    lane_derivative = spectral_time_derivative_bound(
        C_A_t.reshape((-1, C_A_t.shape[-1])), 1.0,
        guard=time_guard, order=1)
    coefficient_derivative_bound = jnp.sum(
        kp_weight[:, None]
        * lane_derivative.reshape(C_A_t.shape[:-1]))
    a_upper_node = jnp.sum(
        kp_weight[:, None, None] * jnp.abs(target), axis=(0, 1))
    a_cell_upper = jnp.minimum(
        a_upper_node[:-1] + coefficient_derivative_bound,
        a_upper_node[1:] + coefficient_derivative_bound)

    ks0 = (C_B.shape[1] - 1) // 2
    b_weight = jnp.where(
        jnp.arange(C_B.shape[0]) == 0, 1.0, 2.0)[:, None]
    b_magnitude = b_weight * jnp.abs(C_B)
    b_centre = C_B[0, ks0].real
    b_remainder = jnp.sum(b_magnitude) - jnp.abs(C_B[0, ks0])
    b_triangle_lower = b_centre - b_remainder
    cell_peak_upper = _distance_upper_profile_device(
        a_cell_upper, jnp.full_like(a_cell_upper, b_triangle_lower),
        float(x_min), float(x_max))
    cell_mass_upper = (cell_peak_upper
                       + jnp.log((2.0 * jnp.pi) ** 2
                                 * (float(x_max) - float(x_min))))
    finite = (jnp.all(jnp.isfinite(C_A_t.real))
              & jnp.all(jnp.isfinite(C_A_t.imag))
              & jnp.all(jnp.isfinite(C_B.real))
              & jnp.all(jnp.isfinite(C_B.imag))
              & jnp.isfinite(coefficient_derivative_bound)
              & jnp.isfinite(scout_peak_lower)
              & jnp.all(jnp.isfinite(cell_mass_upper)))
    live_cells = cell_mass_upper >= scout_peak_lower - float(keep_nats)
    # Invalid arithmetic retains the full time support and then fails the
    # explicit certificate/capacity gates downstream.
    live_cells = jnp.where(finite, live_cells, jnp.ones_like(live_cells))
    outside_log_bound = jax.scipy.special.logsumexp(jnp.where(
        live_cells, -jnp.inf, cell_mass_upper))
    live_nodes = jnp.concatenate((
        live_cells[:1], live_cells[:-1] | live_cells[1:], live_cells[-1:]))
    return {
        "live_cells": live_cells,
        "live_nodes": live_nodes,
        "cell_mass_upper": cell_mass_upper,
        "outside_log_bound": outside_log_bound,
        "scout_peak_lower": scout_peak_lower,
        "coefficient_derivative_bound": coefficient_derivative_bound,
        "b_triangle_lower": b_triangle_lower,
        "certified": finite,
        "n_scout_evaluations": jnp.asarray(scout_size * scout_size * n_time),
    }


def _exact_angular_translation_symmetries_device(
        C_A_t, C_B, *, rtol=1.0e-10):
    """Return a fixed grid of coefficient-certified translations and a mask."""
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    k_phi = max(C_A_t.shape[0] - 1, C_B.shape[0] - 1)
    k_u = max((C_A_t.shape[1] - 1) // 2,
              (C_B.shape[1] - 1) // 2)
    n_phi = max(1, 2 * k_phi)
    n_u = max(1, 2 * k_u)
    dphi = (2.0 * jnp.pi / float(n_phi)
            * jnp.arange(n_phi, dtype=jnp.float64))
    du = (2.0 * jnp.pi / float(n_u)
          * jnp.arange(n_u, dtype=jnp.float64))
    DPHI, DU = jnp.meshgrid(dphi, du, indexing="ij")
    shifts = jnp.stack((DPHI.reshape(-1), DU.reshape(-1)), axis=1)

    def invariant(table):
        kp = jnp.arange(table.shape[0], dtype=jnp.float64)
        ks = jnp.arange(-(table.shape[1] - 1) // 2,
                        (table.shape[1] - 1) // 2 + 1,
                        dtype=jnp.float64)
        phase = jnp.exp(1j * (
            shifts[:, 0, None, None] * kp[None, :, None]
            + shifts[:, 1, None, None] * ks[None, None, :]))
        if table.ndim == 3:
            residual = jnp.max(jnp.abs(
                table[None, ...] * (phase[..., None] - 1.0)), axis=(1, 2, 3))
        else:
            residual = jnp.max(jnp.abs(
                table[None, ...] * (phase - 1.0)), axis=(1, 2))
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(table)))
        return residual <= float(rtol) * scale

    live = invariant(C_A_t) & invariant(C_B)
    return shifts, live


def rank_joint_starts_from_uvq_device(
        C_A_t, C_B, x_min, x_max, *, time_guard=0, max_starts=32,
        angular_oversample=2, norm_rtol=1.0e-9,
        symmetry_rtol=1.0e-10, time_keep_nats=30.0,
        max_time_nodes=64, time_scout_size=4):
    """Rank a static U,V/Q basin portfolio inside ``jit``/``vmap``.

    The finite harmonic orders determine a small ``(phi_ref, 2*psi)`` lattice.
    A constant-size angular scout and a spectral coefficient derivative bound
    first retain complete time cells and certify an upper bound on all discarded
    time-cell mass.  The full harmonic lattice is then evaluated only at a
    fixed-capacity set of retained time nodes; only joint angular maxima at
    time-profile maxima become optimizer starts.
    This is deliberately the device analogue of
    :func:`rank_joint_starts_from_uvq`, with a fixed padded result rather than a
    variable host list.  It performs no likelihood-drop pruning and never claims
    completeness.  Capacity overflow or a negative reconstructed norm is
    explicit and must force enrichment/reserve downstream.
    """
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    _validate_tables(C_A_t, C_B)
    if not (0.0 < float(x_min) < float(x_max)):
        raise ValueError("need 0 < x_min < x_max")
    time_guard = int(time_guard)
    n_time = C_A_t.shape[-1] - 2 * time_guard
    if time_guard < 0 or n_time < 2:
        raise ValueError("time_guard must leave at least two target samples")
    max_starts = int(max_starts)
    angular_oversample = int(angular_oversample)
    max_time_nodes = int(max_time_nodes)
    time_scout_size = int(time_scout_size)
    if max_starts < 1 or angular_oversample < 1:
        raise ValueError("start capacity and angular oversampling must be positive")
    if max_time_nodes < 2 or time_scout_size < 1:
        raise ValueError("time capacity and scout size must be positive")
    if not np.isfinite(float(time_keep_nats)) or float(time_keep_nats) <= 0.0:
        raise ValueError("time_keep_nats must be finite and positive")
    if not np.isfinite(float(norm_rtol)) or float(norm_rtol) < 0.0:
        raise ValueError("norm_rtol must be finite and non-negative")
    if not np.isfinite(float(symmetry_rtol)) or float(symmetry_rtol) < 0.0:
        raise ValueError("symmetry_rtol must be finite and non-negative")
    target = (C_A_t if time_guard == 0
              else C_A_t[..., time_guard:-time_guard])
    k_phi = max(C_A_t.shape[0] - 1, C_B.shape[0] - 1)
    k_u = max((C_A_t.shape[1] - 1) // 2,
              (C_B.shape[1] - 1) // 2)
    n_phi = max(9, 2 * angular_oversample * k_phi + 1)
    n_u = max(9, 2 * angular_oversample * k_u + 1)
    time_capacity = min(max_time_nodes, n_time)
    n_lattice = n_phi * n_u * time_capacity
    if max_starts > n_lattice:
        raise ValueError("max_starts exceeds the structural lattice size")

    time_cover = _time_cell_cover_device(
        C_A_t, C_B, float(x_min), float(x_max), time_guard=time_guard,
        keep_nats=float(time_keep_nats), scout_size=time_scout_size)
    n_live_time_nodes = jnp.count_nonzero(time_cover["live_nodes"])
    time_capacity_ok = n_live_time_nodes <= time_capacity
    cell_upper = time_cover["cell_mass_upper"]
    node_priority = jnp.maximum(
        jnp.concatenate((jnp.asarray([-jnp.inf]), cell_upper)),
        jnp.concatenate((cell_upper, jnp.asarray([-jnp.inf]))))
    node_priority = jnp.where(
        time_cover["live_nodes"], node_priority, -jnp.inf)
    selected_priority, selected_time_index = jax.lax.top_k(
        node_priority, time_capacity)
    selected_time_live = jnp.isfinite(selected_priority)
    # Restore chronological order so adjacency in the compact vector retains
    # its time meaning.  Inactive padding sorts after every physical sample.
    sort_key = jnp.where(
        selected_time_live, selected_time_index,
        n_time + jnp.arange(time_capacity))
    chronological = jnp.argsort(sort_key)
    selected_time_index = selected_time_index[chronological]
    selected_time_live = selected_time_live[chronological]
    selected_time_index_safe = jnp.where(
        selected_time_live, selected_time_index, 0)
    selected_target = jnp.take(
        target, selected_time_index_safe, axis=-1)

    phi, u, A = _harmonic_lattice_device(selected_target, n_phi, n_u)
    _, _, B = _harmonic_lattice_device(C_B, n_phi, n_u)
    profile, x_best = _distance_profile_device(
        A, B[..., None], float(x_min), float(x_max))
    b_scale = jnp.maximum(1.0, jnp.max(jnp.abs(B)))
    norm_nonnegative = jnp.min(B) >= -float(norm_rtol) * b_scale

    # The structural time profile identifies basins without resolving their
    # SNR-narrow interior.  The continuous refiner performs that second job.
    time_profile = jnp.max(profile, axis=(0, 1))
    compact_index = jnp.arange(time_capacity)
    left_adjacent = (
        selected_time_live
        & (compact_index > 0)
        & jnp.roll(selected_time_live, 1)
        & (selected_time_index == jnp.roll(selected_time_index, 1) + 1))
    right_adjacent = (
        selected_time_live
        & (compact_index + 1 < time_capacity)
        & jnp.roll(selected_time_live, -1)
        & (jnp.roll(selected_time_index, -1) == selected_time_index + 1))
    time_left = jnp.where(left_adjacent, jnp.roll(time_profile, 1), -jnp.inf)
    time_right = jnp.where(
        right_adjacent, jnp.roll(time_profile, -1), -jnp.inf)
    time_peak = (time_profile >= time_left) & (time_profile >= time_right)
    angular_peak = jnp.ones(profile.shape, dtype=bool)
    for dphi in (-1, 0, 1):
        for du in (-1, 0, 1):
            if dphi or du:
                angular_peak &= profile >= jnp.roll(
                    jnp.roll(profile, dphi, axis=0), du, axis=1)
    candidate = (angular_peak & time_peak[None, None, :]
                 & selected_time_live[None, None, :] & norm_nonnegative)
    n_lattice_candidates = jnp.count_nonzero(candidate)
    ranked = jnp.where(candidate, profile, -jnp.inf).reshape(-1)
    scores, flat = jax.lax.top_k(ranked, max_starts)
    live = jnp.isfinite(scores)
    compact_time_index = flat % time_capacity
    angular_flat = flat // time_capacity
    u_index = angular_flat % n_u
    phi_index = angular_flat // n_u
    representative_starts = jnp.stack((
        selected_time_index_safe[compact_time_index].astype(jnp.float64),
        phi[phi_index], u[u_index],
        x_best.reshape(-1)[flat]), axis=1)
    fallback = jnp.asarray([
        0.5 * (n_time - 1.0), 0.0, 0.0,
        0.5 * (float(x_min) + float(x_max))])
    representative_starts = jnp.where(
        live[:, None], representative_starts, fallback[None, :])

    # Odd degree-sized targeting lattices need not contain an exact symmetry
    # translate of their best representative.  Complete its orbit from the
    # coefficients themselves; otherwise base and enrichment can agree on the
    # same one-quarter quadrupole cover.  The fixed expansion is small
    # (<=64 shifts through m_max=4), and overflow remains an explicit decline.
    shifts, shift_live = _exact_angular_translation_symmetries_device(
        target, C_B, rtol=float(symmetry_rtol))
    expanded = jnp.broadcast_to(
        representative_starts[:, None, :],
        (max_starts, shifts.shape[0], 4))
    expanded = expanded.at[..., 1:3].set(jnp.mod(
        expanded[..., 1:3] + shifts[None, :, :], 2.0 * jnp.pi))
    expanded_live = live[:, None] & shift_live[None, :]
    expanded_scores = jnp.where(
        expanded_live, scores[:, None], -jnp.inf).reshape(-1)
    final_scores, final_index = jax.lax.top_k(
        expanded_scores, max_starts)
    starts = expanded.reshape((-1, 4))[final_index]
    live = jnp.isfinite(final_scores)
    starts = jnp.where(live[:, None], starts, fallback[None, :])
    n_symmetry = jnp.count_nonzero(shift_live)
    n_candidates = n_lattice_candidates * n_symmetry
    live_cell_index = jnp.arange(n_time - 1, dtype=jnp.int32)
    has_live_time_cell = jnp.any(time_cover["live_cells"])
    time_cover_min = jnp.where(
        has_live_time_cell,
        jnp.min(jnp.where(
            time_cover["live_cells"], live_cell_index, n_time)),
        jnp.nan)
    time_cover_max = jnp.where(
        has_live_time_cell,
        jnp.max(jnp.where(
            time_cover["live_cells"], live_cell_index, -1)) + 1,
        jnp.nan)
    return DeviceJointStartPlan(
        starts, final_scores, live, n_lattice_candidates, n_candidates,
        (norm_nonnegative & time_cover["certified"] & time_capacity_ok
         & (n_candidates <= max_starts)),
        jnp.asarray(n_phi), jnp.asarray(n_u), jnp.asarray(time_capacity),
        jnp.asarray(n_time), jnp.asarray(n_lattice),
        time_cover["n_scout_evaluations"],
        jnp.count_nonzero(time_cover["live_cells"]), n_live_time_nodes,
        time_cover["outside_log_bound"], time_cover_min, time_cover_max,
        time_cover["scout_peak_lower"],
        time_cover["certified"], time_capacity_ok,
        norm_nonnegative, n_symmetry)


def combine_device_start_plans(base, extra):
    """Form a stronger fixed portfolio that contains every base start.

    Duplicates are intentionally retained here and removed only after continuous
    refinement, where basin identity is meaningful.  This construction makes
    the empirical controller's nesting premise structural: the stronger pass
    cannot silently omit a base optimizer start.  Either input overflow remains
    a fail-closed capacity flag.
    """
    if not isinstance(base, DeviceJointStartPlan):
        raise TypeError("base must be DeviceJointStartPlan")
    if not isinstance(extra, DeviceJointStartPlan):
        raise TypeError("extra must be DeviceJointStartPlan")
    for name, plan in (("base", base), ("extra", extra)):
        if (plan.starts.ndim != 2 or plan.starts.shape[1] != 4
                or plan.scores.shape != (plan.starts.shape[0],)
                or plan.live.shape != (plan.starts.shape[0],)):
            raise ValueError("%s device start plan has inconsistent shapes" % name)
    same_time_support = (
        base.n_retained_time_samples == extra.n_retained_time_samples)
    starts = jnp.concatenate((base.starts, extra.starts), axis=0)
    scores = jnp.concatenate((base.scores, extra.scores), axis=0)
    live = jnp.concatenate((base.live, extra.live), axis=0)
    return DeviceJointStartPlan(
        starts, scores, live,
        (base.n_lattice_candidates_before_symmetry
         + extra.n_lattice_candidates_before_symmetry),
        base.n_candidates_before_cap + extra.n_candidates_before_cap,
        base.capacity_ok & extra.capacity_ok & same_time_support,
        jnp.maximum(base.n_phi_lattice, extra.n_phi_lattice),
        jnp.maximum(base.n_u_lattice, extra.n_u_lattice),
        jnp.maximum(base.n_time_lattice, extra.n_time_lattice),
        jnp.maximum(base.n_retained_time_samples,
                    extra.n_retained_time_samples),
        base.n_lattice_evaluations + extra.n_lattice_evaluations,
        base.n_time_scout_evaluations + extra.n_time_scout_evaluations,
        base.n_time_cells_retained + extra.n_time_cells_retained,
        base.n_time_nodes_retained + extra.n_time_nodes_retained,
        jnp.minimum(base.time_outside_log_bound,
                    extra.time_outside_log_bound),
        jnp.minimum(base.time_cover_min_sample,
                    extra.time_cover_min_sample),
        jnp.maximum(base.time_cover_max_sample,
                    extra.time_cover_max_sample),
        jnp.maximum(base.time_scout_peak_lower,
                    extra.time_scout_peak_lower),
        base.time_cover_certified & extra.time_cover_certified,
        base.time_capacity_ok & extra.time_capacity_ok,
        base.norm_nonnegative & extra.norm_nonnegative,
        jnp.maximum(base.n_exact_symmetry_shifts,
                    extra.n_exact_symmetry_shifts))


def _numpy_angular_field(C, phi, u):
    kp = np.arange(C.shape[0], dtype=float)[:, None]
    ks_max = (C.shape[1] - 1) // 2
    ks = np.arange(-ks_max, ks_max + 1, dtype=float)[None, :]
    weight = _kp_weights_numpy(C.shape[0])[:, None]
    return float(np.sum(weight * C * np.exp(1j * (kp * phi + ks * u))).real)


def _distance_start(K, R, x_min, x_max):
    """Best support-aware stationary/boundary candidate for ``x^-4 L``."""
    candidates = [float(x_min), float(x_max)]
    R = max(float(R), 0.0)
    K = float(K)
    discriminant = K * K - 16.0 * R
    if R > 0.0 and discriminant >= 0.0:
        x_plus = (K + np.sqrt(discriminant)) / (2.0 * R)
        if x_min <= x_plus <= x_max:
            candidates.append(float(x_plus))
    values = [K * x - 0.5 * R * x * x - 4.0 * np.log(x)
              for x in candidates]
    return candidates[int(np.argmax(values))]


def algebraic_angle_starts_from_uv(C_A_t, uv_summary, time_starts,
                                   x_min, x_max):
    """Build sparse four-axis starts from U,V ranking and algebraic maxima.

    One U,V-informed distance probe is used per selected time basin.  At that
    probe the exact bivariate trigonometric stationary system is enumerated by
    :func:`RIFT.likelihood.bivariate_trig_stationary.enumerate_torus_maxima`.
    Every returned maximum is then assigned its support-aware analytic distance
    candidate.  No generic angle or distance seed lattice is constructed.

    The returned ``all_enumerations_ok`` covers only the angular solves at the
    probed time/distance slices.  It is deliberately *not* suitable for
    ``AllAxisModePlan.enumeration_complete``: completeness of the joint 4-D
    modes still needs the independent outside-cover warrant.
    """
    from RIFT.likelihood.bivariate_trig_stationary import enumerate_torus_maxima
    from RIFT.likelihood.joint_angle_peak_local import joint_table

    C_A_t = np.asarray(C_A_t, dtype=np.complex128)
    time_starts = np.asarray(time_starts, dtype=np.int32).ravel()
    if not isinstance(uv_summary, UVHarmonicSummary):
        raise TypeError("uv_summary must come from summarize_uv_norm_table")
    _validate_tables(C_A_t, uv_summary.C_B)
    starts = []
    reports = []
    all_ok = True
    weight = _kp_weights_numpy(C_A_t.shape[0])[:, None]
    b_scale = max(uv_summary.b_lower,
                  float(np.abs(uv_summary.C_B[0,
                      (uv_summary.C_B.shape[1] - 1) // 2])), 1.0e-30)
    for time_index in time_starts:
        if not (0 <= int(time_index) < C_A_t.shape[-1]):
            raise ValueError("time start outside retained support")
        C_A = C_A_t[..., int(time_index)]
        a_upper = float(np.sum(weight * np.abs(C_A)))
        x_probe = float(np.clip(a_upper / b_scale, x_min, x_max))
        result = enumerate_torus_maxima(
            joint_table(C_A, uv_summary.C_B, x_probe))
        report = dict(result.report)
        report.update(time_index=int(time_index), x_probe=x_probe)
        reports.append(report)
        all_ok = all_ok and bool(result.ok)
        for phi, u in result.points:
            K = _numpy_angular_field(C_A, phi, u)
            R = _numpy_angular_field(uv_summary.C_B, phi, u)
            x = _distance_start(K, R, float(x_min), float(x_max))
            starts.append((float(time_index), float(phi), float(u), x))
    return (np.asarray(starts, dtype=float).reshape((-1, 4)),
            bool(all_ok), reports)


def _validate_tables(C_A_t, C_B):
    if C_A_t.ndim != 3:
        raise ValueError("C_A_t must have shape (KP,2KS+1,Ntime)")
    if C_B.ndim != 2:
        raise ValueError("C_B must be the collapsed (KP,2KS+1) norm table")
    if C_A_t.shape[1] % 2 != 1 or C_B.shape[1] % 2 != 1:
        raise ValueError("angular harmonic axes must have odd length")
    if C_A_t.shape[0] > C_B.shape[0] or C_A_t.shape[1] > C_B.shape[1]:
        raise ValueError("C_B must contain every harmonic represented by C_A")


def _angular_field(C, phi, u):
    """Evaluate a real stored-half-plane angular Fourier table at one point."""
    kp = jnp.arange(C.shape[0], dtype=jnp.float64)
    ks_max = (C.shape[1] - 1) // 2
    ks = jnp.arange(-ks_max, ks_max + 1, dtype=jnp.float64)
    weight = jnp.where(kp == 0.0, 1.0, 2.0)
    phase = jnp.exp(1j * (kp[:, None] * phi + ks[None, :] * u))
    return jnp.sum(weight[:, None] * C * phase).real


def _scalar_log_density(theta, coeff, frequency, offset, C_A_shape, C_B,
                        x_min, x_max):
    """Unnormalized four-axis log density at one continuous coordinate."""
    t, phi, u, x = theta
    flat = _evaluate_time_spectrum(
        coeff, frequency, jnp.atleast_1d(t), offset)[:, 0]
    C_A = flat.reshape(C_A_shape[:-1])
    A = _angular_field(C_A, phi, u)
    B = _angular_field(C_B, phi, u)
    inside = ((t >= 0.0) & (t <= C_A_shape[-1] - 1.0)
              & (x >= x_min) & (x <= x_max) & (x > 0.0))
    value = x * A - 0.5 * x * x * B - 4.0 * jnp.log(jnp.maximum(x, 1e-300))
    return jnp.where(inside, value, -jnp.inf)


def refine_all_axis_starts(C_A_t, C_B, starts, x_min, x_max, *,
                           time_guard=0,
                           iterations=12, ridge=1.0e-8,
                           max_step=(2.0, 0.5, 0.5, 0.25),
                           time_localize_iterations=32, live=None):
    """Refine four-axis starts with fixed-iteration JAX gradient/Hessian steps.

    This is local optimization only.  The return values report stationarity and
    local curvature; they do not assert completeness.  Angular coordinates are
    wrapped, while time and distance remain on their physical support.  An
    optional fixed-shape ``live`` mask skips optimizer work for padded starts
    and returns finite geometry placeholders with value ``-inf`` for them.
    """
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    starts = jnp.asarray(starts, dtype=jnp.float64)
    _validate_tables(C_A_t, C_B)
    if starts.ndim != 2 or starts.shape[1] != 4:
        raise ValueError("starts must have shape (N,4)")
    if live is None:
        live = jnp.ones(starts.shape[0], dtype=bool)
    else:
        live = jnp.asarray(live, dtype=bool)
        if live.shape != (starts.shape[0],):
            raise ValueError("live must match the start capacity")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")
    if int(time_localize_iterations) < 1:
        raise ValueError("time_localize_iterations must be positive")
    time_guard = int(time_guard)
    n_time = C_A_t.shape[-1] - 2 * time_guard
    if time_guard < 0 or n_time < 2:
        raise ValueError("time_guard must leave at least two integration samples")
    coeff, frequency, offset = _time_primitive_spectrum(
        C_A_t.reshape((-1, C_A_t.shape[-1])), time_guard)
    model_shape = C_A_t.shape[:-1] + (n_time,)
    fn = lambda th: _scalar_log_density(
        th, coeff, frequency, offset, model_shape, C_B,
        float(x_min), float(x_max))
    grad_fn = jax.grad(fn)
    hess_fn = jax.hessian(fn)
    max_step = jnp.asarray(max_step, dtype=jnp.float64)

    def _project(th):
        return jnp.asarray([
            jnp.clip(th[0], 0.0, n_time - 1.0),
            jnp.mod(th[1], 2.0 * jnp.pi),
            jnp.mod(th[2], 2.0 * jnp.pi),
            jnp.clip(th[3], float(x_min), float(x_max)),
        ])

    def _one(start):
        # The U,V/Q portfolio is ranked on native time samples.  Localize the
        # continuous time maximum inside that basin before the coupled Newton
        # solve; the basin narrows with SNR while this work remains fixed.
        left = jnp.maximum(0.0, start[0] - 1.0)
        right = jnp.minimum(float(n_time - 1), start[0] + 1.0)
        golden_ratio = 0.5 * (jnp.sqrt(5.0) - 1.0)
        c = right - golden_ratio * (right - left)
        d = left + golden_ratio * (right - left)

        def _at_time(value):
            return fn(start.at[0].set(value))

        fc, fd = _at_time(c), _at_time(d)

        def _golden_step(_, state):
            lo, hi, ca, da, fca, fda = state
            choose_left = fca >= fda
            new_hi = jnp.where(choose_left, da, hi)
            new_lo = jnp.where(choose_left, lo, ca)
            new_c = jnp.where(
                choose_left,
                new_hi - golden_ratio * (new_hi - new_lo), da)
            new_d = jnp.where(
                choose_left, ca,
                new_lo + golden_ratio * (new_hi - new_lo))
            new_fc = jnp.where(choose_left, _at_time(new_c), fda)
            new_fd = jnp.where(choose_left, fca, _at_time(new_d))
            return new_lo, new_hi, new_c, new_d, new_fc, new_fd

        left, right, c, d, fc, fd = jax.lax.fori_loop(
            0, int(time_localize_iterations), _golden_step,
            (left, right, c, d, fc, fd))
        localized = start.at[0].set(jnp.where(fc >= fd, c, d))
        candidates = jnp.stack((start, localized), axis=0)
        start = candidates[jnp.argmax(jax.vmap(fn)(candidates))]

        def _step(th, _):
            g = grad_fn(th)
            H = hess_fn(th)
            eigenvalue, eigenvector = jnp.linalg.eigh(-H)
            safe = jnp.maximum(eigenvalue, float(ridge))
            step = eigenvector @ ((eigenvector.T @ g) / safe)
            # Keep the coupled Newton direction.  Component-wise clipping can
            # reverse its directional derivative for the narrow correlated
            # time/angle basins seen in the matched SNR ladder.
            ratio = max_step / jnp.maximum(
                jnp.abs(step), jnp.finfo(jnp.float64).tiny)
            step = step * jnp.minimum(1.0, jnp.min(ratio))
            proposals = jax.vmap(
                lambda scale: _project(th + scale * step))(
                    jnp.concatenate((
                        jnp.exp2(-jnp.arange(13, dtype=jnp.float64)),
                        jnp.zeros(1, dtype=jnp.float64))))
            values = jax.vmap(fn)(proposals)
            return proposals[jnp.argmax(values)], None

        point, _ = jax.lax.scan(_step, _project(start), None,
                                length=int(iterations))
        value = fn(point)
        gradient = grad_fn(point)
        hessian = hess_fn(point)
        curvature = jnp.linalg.eigvalsh(-hessian)
        return point, value, gradient, hessian, curvature

    def _inactive(start):
        point = _project(start)
        return (point, jnp.asarray(-jnp.inf, dtype=start.dtype),
                jnp.zeros(4, dtype=start.dtype),
                -jnp.eye(4, dtype=start.dtype),
                jnp.ones(4, dtype=start.dtype))

    def _mapped(args):
        start, is_live = args
        return jax.lax.cond(
            is_live, jax.checkpoint(_one), _inactive, start)

    return jax.lax.map(_mapped, (starts, live))


def select_refined_modes(points, values, gradients, curvatures, *,
                         max_modes, gradient_tol=1.0e-6,
                         coordinate_tol=(0.25, 1.0e-4, 1.0e-4, 1.0e-5),
                         scaled_step_tol=None):
    """Host-side stationarity filter, rank and periodic deduplication.

    A rejected optimizer result is not a missed-mode decision: callers retain
    the original basin in their completeness accounting and must add starts or
    decline if its mass has not independently been bounded.  In particular,
    constrained maxima on the time or distance boundary need a future one-sided
    local region; the full-gradient/positive-curvature filter here rejects them
    and relies on the outside warrant or conservative reserve.
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    gradients = np.asarray(gradients, dtype=float)
    curvatures = np.asarray(curvatures, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 4
            or gradients.shape != points.shape
            or curvatures.shape != points.shape
            or values.shape[0] != points.shape[0]):
        raise ValueError("inconsistent refined-mode arrays")
    tolerance = np.asarray(coordinate_tol, dtype=float)
    if tolerance.shape != (4,) or np.any(tolerance <= 0.0):
        raise ValueError("coordinate_tol must contain four positive values")
    if scaled_step_tol is None:
        scaled_step_tol = float(np.min(tolerance))
    if float(scaled_step_tol) <= 0.0:
        raise ValueError("scaled_step_tol must be positive")
    gradient_norm = np.linalg.norm(gradients, axis=1)
    min_curvature = np.min(curvatures, axis=1)
    # Absolute gradients scale with lnL and hence with SNR.  For a positive
    # Hessian, ||g||/lambda_min bounds the Newton displacement, providing an
    # SNR-stable stationarity criterion alongside the legacy absolute gate.
    scaled_stationary = gradient_norm <= min_curvature * float(scaled_step_tol)
    stationary = (np.all(np.isfinite(points), axis=1)
                  & np.isfinite(values)
                  & np.all(np.isfinite(gradients), axis=1)
                  & ((gradient_norm <= float(gradient_tol)) | scaled_stationary)
                  & np.all(curvatures > 0.0, axis=1))
    order = np.flatnonzero(stationary)
    order = order[np.argsort(values[order])[::-1]]
    max_modes = int(max_modes)
    if max_modes < 1:
        raise ValueError("max_modes must be positive")
    selected = []
    for index in order:
        point = points[index]
        duplicate = False
        for old in selected:
            delta = np.abs(point - points[old])
            delta[1] = _periodic_distance(point[1], points[old, 1])
            delta[2] = _periodic_distance(point[2], points[old, 2])
            if np.all(delta <= tolerance):
                duplicate = True
                break
        if not duplicate:
            selected.append(int(index))
    if len(selected) > max_modes:
        raise ValueError(
            "unique stationary mode count exceeds fixed plan capacity")
    selected = np.asarray(selected, dtype=np.int32)
    return selected, stationary


def mode_local_geometry(hessians, *, w_sigma=8.0,
                        eigenvalue_floor=1.0e-12):
    """Return Hessian-whitening transforms and conservative box enclosures.

    If ``L L.T = inv(-H)``, local coordinates are ``theta=center+L z`` with
    ``z`` in a fixed ``[-w_sigma,w_sigma]^4`` cube.  Cholesky's lower-triangular
    form is computationally important: time depends only on ``z[0]``, so exact
    selected-point time reconstruction still needs just ``local_order`` points,
    not ``local_order**4``.  The returned axis-aligned half-width encloses the
    transformed cube and is used for conservative support/overlap checks.
    """
    hessians = np.asarray(hessians, dtype=float)
    if hessians.ndim != 3 or hessians.shape[1:] != (4, 4):
        raise ValueError("hessians must have shape (N,4,4)")
    transforms = np.full_like(hessians, np.nan)
    half_widths = np.full((hessians.shape[0], 4), np.nan)
    for i, H in enumerate(hessians):
        eigenvalue = np.linalg.eigvalsh(-H)
        if (not np.all(np.isfinite(eigenvalue))
                or np.min(eigenvalue) <= float(eigenvalue_floor)):
            continue
        covariance = np.linalg.inv(-H)
        try:
            transform = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            continue
        transforms[i] = transform
        half_widths[i] = float(w_sigma) * np.sum(np.abs(transform), axis=1)
    return transforms, half_widths


def _periodic_distance(a, b):
    return abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi)


def _boxes_disjoint(centers, half_widths):
    for i in range(len(centers)):
        for j in range(i):
            separated = (
                abs(centers[i, 0] - centers[j, 0])
                >= half_widths[i, 0] + half_widths[j, 0]
                or _periodic_distance(centers[i, 1], centers[j, 1])
                >= half_widths[i, 1] + half_widths[j, 1]
                or _periodic_distance(centers[i, 2], centers[j, 2])
                >= half_widths[i, 2] + half_widths[j, 2]
                or abs(centers[i, 3] - centers[j, 3])
                >= half_widths[i, 3] + half_widths[j, 3])
            if not separated:
                return False
    return True


def make_all_axis_mode_plan(centers, *, max_modes, local_transforms,
                            local_radius=1.0,
                            outside_log_bound=np.inf,
                            enumeration_complete=False,
                            outside_bound_certified=False,
                            time_reconstruction_certified=False,
                            time_outside_log_bound=np.inf,
                            time_outside_bound_certified=False,
                            time_cover_min_sample=np.nan,
                            time_cover_max_sample=np.nan,
                            discovery_capacity_ok=True,
                            boundary_maximum_pinned=False):
    """Pad a host mode set and freeze its independent acceptance warrants."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 4:
        raise ValueError("centers must have shape (N,4)")
    local_transforms = np.asarray(local_transforms, dtype=float)
    if local_transforms.shape != (len(centers), 4, 4):
        raise ValueError("local_transforms must have shape (N,4,4)")
    if not (float(local_radius) > 0.0):
        raise ValueError("local_radius must be positive")
    # This enclosure is a theorem for an affine image of a cube, not caller
    # policy: |L z|_i <= radius * sum_j |L_ij|.  Deriving it here prevents an
    # undersized supplied box from blessing overlapping or out-of-support
    # quadrature regions.
    half_widths = (float(local_radius)
                   * np.sum(np.abs(local_transforms), axis=2))
    max_modes = int(max_modes)
    if max_modes < 1 or len(centers) > max_modes:
        raise ValueError("mode count exceeds fixed plan capacity")
    valid = (np.all(np.isfinite(centers), axis=1)
             & np.all(np.isfinite(half_widths) & (half_widths > 0.0), axis=1)
             & np.all(np.isfinite(local_transforms), axis=(1, 2)))
    if not np.all(valid):
        raise ValueError(
            "every supplied mode needs finite, nondegenerate local geometry")
    upper = np.triu(local_transforms, k=1)
    if not np.all(upper == 0.0):
        raise ValueError(
            "local_transforms must be lower triangular; the selected-point "
            "time topology does not evaluate upper-triangle entries")
    if not np.all(np.diagonal(local_transforms, axis1=1, axis2=2) > 0.0):
        raise ValueError("local_transforms must have positive diagonal")
    kept_centers = centers
    kept_widths = half_widths
    kept_transforms = local_transforms
    padded_centers = np.zeros((max_modes, 4), dtype=float)
    padded_widths = np.ones((max_modes, 4), dtype=float)
    padded_transforms = np.repeat(np.eye(4)[None, ...], max_modes, axis=0)
    # ``lax.scan`` traces/evaluates the padded lanes even though their values
    # are masked from the log-sum.  Reuse one finite live geometry so inactive
    # lanes cannot manufacture NaNs (notably log(x) at x<=0) which would leak
    # into outer gradients or Hessians through the masked branch.
    if len(kept_centers):
        padded_centers[:] = kept_centers[0]
        padded_widths[:] = kept_widths[0]
        padded_transforms[:] = kept_transforms[0]
    live = np.zeros(max_modes, dtype=bool)
    padded_centers[:len(kept_centers)] = kept_centers
    padded_widths[:len(kept_widths)] = kept_widths
    padded_transforms[:len(kept_transforms)] = kept_transforms
    live[:len(kept_centers)] = True
    disjoint = _boxes_disjoint(kept_centers, kept_widths)
    return AllAxisModePlan(
        jnp.asarray(padded_centers), jnp.asarray(padded_widths),
        jnp.asarray(padded_transforms), jnp.asarray(float(local_radius)),
        jnp.asarray(live), jnp.asarray(float(outside_log_bound)),
        jnp.asarray(bool(enumeration_complete)),
        jnp.asarray(bool(outside_bound_certified)),
        jnp.asarray(bool(time_reconstruction_certified)),
        jnp.asarray(float(time_outside_log_bound)),
        jnp.asarray(bool(time_outside_bound_certified)),
        jnp.asarray(float(time_cover_min_sample)),
        jnp.asarray(float(time_cover_max_sample)),
        jnp.asarray(disjoint),
        jnp.asarray(bool(discovery_capacity_ok)),
        jnp.asarray(bool(boundary_maximum_pinned)))


def _boxes_disjoint_device(centers, half_widths, live):
    """Fixed-shape periodic counterpart of :func:`_boxes_disjoint`."""
    delta = jnp.abs(centers[:, None, :] - centers[None, :, :])
    angular = jnp.abs(jnp.mod(
        centers[:, None, 1:3] - centers[None, :, 1:3] + jnp.pi,
        2.0 * jnp.pi) - jnp.pi)
    delta = delta.at[..., 1:3].set(angular)
    separated = jnp.any(
        delta >= half_widths[:, None, :] + half_widths[None, :, :],
        axis=-1)
    index = jnp.arange(centers.shape[0])
    pair = ((index[:, None] > index[None, :])
            & live[:, None] & live[None, :])
    return jnp.all((~pair) | separated)


def _validate_device_mode_plan_arguments(
        start_plan, max_modes, local_radius, coordinate_tol,
        scaled_step_tol, eigenvalue_floor):
    if not isinstance(start_plan, DeviceJointStartPlan):
        raise TypeError("start_plan must be DeviceJointStartPlan")
    if (start_plan.starts.ndim != 2 or start_plan.starts.shape[1] != 4
            or start_plan.scores.shape != (start_plan.starts.shape[0],)
            or start_plan.live.shape != (start_plan.starts.shape[0],)):
        raise ValueError("device start plan has inconsistent shapes")
    max_modes = int(max_modes)
    if max_modes < 1 or max_modes > start_plan.starts.shape[0]:
        raise ValueError("max_modes must fit inside the start capacity")
    if not (float(local_radius) > 0.0):
        raise ValueError("local_radius must be positive")
    tolerance = jnp.asarray(coordinate_tol, dtype=jnp.float64)
    if tolerance.shape != (4,) or np.any(np.asarray(coordinate_tol) <= 0.0):
        raise ValueError("coordinate_tol must contain four positive values")
    if scaled_step_tol is None:
        scaled_step_tol = float(np.min(np.asarray(coordinate_tol)))
    if float(scaled_step_tol) <= 0.0:
        raise ValueError("scaled_step_tol must be positive")
    if (not np.isfinite(float(eigenvalue_floor))
            or float(eigenvalue_floor) <= 0.0):
        raise ValueError("eigenvalue_floor must be finite and positive")
    return max_modes, tolerance, float(scaled_step_tol)


def _assemble_all_axis_mode_plan_device(
        C_A_t, start_plan, refined, x_min, x_max, *, max_modes,
        local_radius, time_guard, gradient_tol, tolerance,
        scaled_step_tol, eigenvalue_floor,
        time_reconstruction_certified, boundary_keep_nats=30.0):
    """Select fixed-shape local geometry from an existing device refinement."""
    points, values, gradients, hessians, curvatures = refined
    if (points.shape != start_plan.starts.shape
            or values.shape != (start_plan.starts.shape[0],)
            or gradients.shape != start_plan.starts.shape
            or hessians.shape != (start_plan.starts.shape[0], 4, 4)
            or curvatures.shape != start_plan.starts.shape):
        raise ValueError("refinement arrays do not match the start plan")

    gradient_norm = jnp.linalg.norm(gradients, axis=1)
    min_curvature = jnp.min(curvatures, axis=1)
    scaled_stationary = (
        gradient_norm <= min_curvature * float(scaled_step_tol))
    stationary = (
        start_plan.live
        & jnp.all(jnp.isfinite(points), axis=1)
        & jnp.isfinite(values)
        & jnp.all(jnp.isfinite(gradients), axis=1)
        & ((gradient_norm <= float(gradient_tol)) | scaled_stationary)
        & jnp.all(curvatures > float(eigenvalue_floor), axis=1))

    fisher = -0.5 * (hessians + jnp.swapaxes(hessians, 1, 2))
    # Never factor an invalid lane.  Padded/non-stationary starts still exist
    # in the fixed shape and an indefinite inverse can emit NaNs that leak into
    # outer AD even when that lane is later masked.  Identity is a finite
    # tracing placeholder only; such a lane remains non-stationary.
    safe_fisher = jnp.where(
        stationary[:, None, None], fisher,
        jnp.eye(4, dtype=jnp.float64)[None, :, :])
    covariance = jnp.linalg.inv(safe_fisher)
    transforms = jnp.linalg.cholesky(covariance)
    geometry_finite = jnp.all(jnp.isfinite(transforms), axis=(1, 2))
    stationary &= geometry_finite
    order = jnp.argsort(jnp.where(stationary, values, -jnp.inf))[::-1]

    n_time = C_A_t.shape[-1] - 2 * int(time_guard)
    # A live start that refinement pinned to the support boundary and the
    # stationarity filter then rejected is a constrained maximum no mode
    # covers.  If its value is within ``boundary_keep_nats`` of the best live
    # value it can carry the integral, so the plan records it and the gate
    # declines rather than integrating the interior modes alone.
    x_span = max(1.0e-300, float(x_max) - float(x_min))
    at_time_bound = ((points[:, 0] <= 1.0e-9)
                     | (points[:, 0] >= n_time - 1.0 - 1.0e-9))
    at_x_bound = ((points[:, 3] <= float(x_min) + 1.0e-9 * x_span)
                  | (points[:, 3] >= float(x_max) - 1.0e-9 * x_span))
    live_finite = (start_plan.live & jnp.all(jnp.isfinite(points), axis=1)
                   & jnp.isfinite(values))
    best_live_value = jnp.max(jnp.where(live_finite, values, -jnp.inf))
    pinned = (live_finite & (~stationary) & (at_time_bound | at_x_bound)
              & (values >= best_live_value - float(boundary_keep_nats)))
    boundary_maximum_pinned = jnp.any(pinned)
    fallback_center = jnp.asarray([
        0.5 * (n_time - 1.0), 0.0, 0.0,
        0.5 * (float(x_min) + float(x_max))])
    fallback_transform = jnp.diag(jnp.asarray([
        1.0, 0.1, 0.1,
        max(1.0e-12, 0.1 * (float(x_max) - float(x_min)))],
        dtype=jnp.float64))
    selected_centers = jnp.broadcast_to(
        fallback_center, (max_modes, 4)).copy()
    selected_transforms = jnp.broadcast_to(
        fallback_transform, (max_modes, 4, 4)).copy()
    selected_live = jnp.zeros(max_modes, dtype=bool)

    def _select(index, state):
        centers, local_transforms, live, n_unique, overflow = state
        candidate_index = order[index]
        candidate = points[candidate_index]
        candidate_transform = transforms[candidate_index]
        delta = jnp.abs(centers - candidate[None, :])
        angular = jnp.abs(jnp.mod(
            centers[:, 1:3] - candidate[None, 1:3] + jnp.pi,
            2.0 * jnp.pi) - jnp.pi)
        delta = delta.at[:, 1:3].set(angular)
        duplicate = jnp.any(live & jnp.all(delta <= tolerance[None, :], axis=1))
        unique = stationary[candidate_index] & (~duplicate)
        has_room = n_unique < max_modes
        add = unique & has_room
        slot = jnp.minimum(n_unique, max_modes - 1)

        def _write(payload):
            old_centers, old_transforms, old_live = payload
            return (old_centers.at[slot].set(candidate),
                    old_transforms.at[slot].set(candidate_transform),
                    old_live.at[slot].set(True))

        centers, local_transforms, live = jax.lax.cond(
            add, _write, lambda payload: payload,
            (centers, local_transforms, live))
        return (centers, local_transforms, live, n_unique + add,
                overflow | (unique & (~has_room)))

    (selected_centers, selected_transforms, selected_live,
     n_selected, selection_overflow) = jax.lax.fori_loop(
         0, start_plan.starts.shape[0], _select,
         (selected_centers, selected_transforms, selected_live,
          jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)))
    half_widths = (float(local_radius)
                   * jnp.sum(jnp.abs(selected_transforms), axis=2))
    disjoint = _boxes_disjoint_device(
        selected_centers, half_widths, selected_live)
    discovery_capacity_ok = start_plan.capacity_ok & (~selection_overflow)
    plan = AllAxisModePlan(
        selected_centers, half_widths, selected_transforms,
        jnp.asarray(float(local_radius)), selected_live,
        jnp.asarray(jnp.inf), jnp.asarray(False), jnp.asarray(False),
        jnp.asarray(bool(time_reconstruction_certified)),
        start_plan.time_outside_log_bound,
        start_plan.time_cover_certified,
        start_plan.time_cover_min_sample,
        start_plan.time_cover_max_sample,
        disjoint,
        discovery_capacity_ok,
        boundary_maximum_pinned)
    ledger = {
        "boundary_maximum_pinned": boundary_maximum_pinned,
        "n_boundary_pinned_starts": jnp.count_nonzero(pinned),
        "n_optimizer_starts": jnp.count_nonzero(start_plan.live),
        "n_refined_stationary": jnp.count_nonzero(stationary),
        "n_selected_modes": n_selected,
        "selection_overflow": selection_overflow,
        "start_capacity_ok": start_plan.capacity_ok,
        # start_capacity_ok is an AND of four terms (:862): the candidate count
        # against max_starts, and these three.  Exposing only the conjunction
        # makes decline_capacity a single label for four different failures,
        # three of which no cap value can repair.  Additive; nothing reads a
        # planning dict by position.
        "time_cover_certified": start_plan.time_cover_certified,
        "time_capacity_ok": start_plan.time_capacity_ok,
        "discovery_capacity_ok": discovery_capacity_ok,
        "norm_nonnegative": start_plan.norm_nonnegative,
        "n_lattice_candidates_before_symmetry":
            start_plan.n_lattice_candidates_before_symmetry,
        "n_candidates_before_cap": start_plan.n_candidates_before_cap,
        "n_exact_symmetry_shifts": start_plan.n_exact_symmetry_shifts,
        "n_lattice_evaluations": start_plan.n_lattice_evaluations,
        "n_time_scout_evaluations": start_plan.n_time_scout_evaluations,
        "n_phi_lattice": start_plan.n_phi_lattice,
        "n_u_lattice": start_plan.n_u_lattice,
        "n_time_lattice": start_plan.n_time_lattice,
        "n_retained_time_samples": start_plan.n_retained_time_samples,
        "n_time_cells_retained": start_plan.n_time_cells_retained,
        "n_time_nodes_retained": start_plan.n_time_nodes_retained,
        "time_outside_log_bound": start_plan.time_outside_log_bound,
        "time_cover_min_sample": start_plan.time_cover_min_sample,
        "time_cover_max_sample": start_plan.time_cover_max_sample,
        "time_scout_peak_lower": start_plan.time_scout_peak_lower,
        "time_cover_certified": start_plan.time_cover_certified,
        "time_capacity_ok": start_plan.time_capacity_ok,
        "max_gradient_norm": jnp.max(jnp.where(
            stationary, gradient_norm, -jnp.inf)),
        "min_selected_curvature": jnp.min(jnp.where(
            stationary, min_curvature, jnp.inf)),
        "global_completeness_certified": jnp.asarray(False),
        "derivative_warrant_certified": jnp.asarray(False),
    }
    return plan, ledger


def make_all_axis_mode_plan_device(
        C_A_t, C_B, start_plan, x_min, x_max, *, max_modes,
        local_radius=6.0, time_guard=0, iterations=12,
        time_localize_iterations=32, ridge=1.0e-8,
        max_step=(2.0, 0.5, 0.5, 0.25), gradient_tol=1.0e-6,
        coordinate_tol=(0.25, 1.0e-4, 1.0e-4, 1.0e-5),
        scaled_step_tol=None, eigenvalue_floor=1.0e-12,
        time_reconstruction_certified=False):
    """Refine and deduplicate a fixed-shape device start portfolio.

    This is the per-row planning seam needed by nested JAX callers.  It never
    transfers a tracer to NumPy: starts are refined, ranked, deduplicated, and
    converted to Hessian-whitened local regions entirely on device.  Overflow
    is recorded in ``discovery_capacity_ok`` rather than silently truncating a
    mode set.  No outside-mass or derivative certificate is manufactured here;
    callers must use empirical enrichment plus exact reserve, or supply a
    separately derived certificate through a future API.
    """
    max_modes, tolerance, scaled_step_tol = (
        _validate_device_mode_plan_arguments(
            start_plan, max_modes, local_radius, coordinate_tol,
            scaled_step_tol, eigenvalue_floor))
    refined = refine_all_axis_starts(
        C_A_t, C_B, start_plan.starts, x_min, x_max,
        time_guard=int(time_guard), iterations=int(iterations), ridge=float(ridge),
        max_step=max_step,
        time_localize_iterations=int(time_localize_iterations),
        live=start_plan.live)
    return _assemble_all_axis_mode_plan_device(
        C_A_t, start_plan, refined, x_min, x_max,
        max_modes=max_modes, local_radius=float(local_radius),
        time_guard=int(time_guard), gradient_tol=float(gradient_tol),
        tolerance=tolerance, scaled_step_tol=scaled_step_tol,
        eigenvalue_floor=float(eigenvalue_floor),
        time_reconstruction_certified=time_reconstruction_certified)


def make_all_axis_mode_plan_pair_device(
        C_A_t, C_B, base_starts, extra_starts, x_min, x_max, *, max_modes,
        enriched_max_modes=None,
        local_radius=6.0, time_guard=0, iterations=12,
        time_localize_iterations=32, ridge=1.0e-8,
        max_step=(2.0, 0.5, 0.5, 0.25), gradient_tol=1.0e-6,
        coordinate_tol=(0.25, 1.0e-4, 1.0e-4, 1.0e-5),
        scaled_step_tol=None, eigenvalue_floor=1.0e-12,
        time_reconstruction_certified=False):
    """Build nested base/enriched plans with one shared optimizer pass.

    ``extra_starts`` is combined with ``base_starts`` structurally before the
    refinement, so the enriched portfolio contains every base lane.  The
    independent per-lane optimizer means the prefix of the shared result is
    exactly the refinement that a separate base call would have produced.
    Reusing it removes the otherwise duplicated base hill-climbing work without
    changing either mode selection or any completeness warrant.
    ``enriched_max_modes`` may raise the stronger plan's output capacity; it
    defaults to the base ``max_modes``.

    The fifth return value records actual shared optimizer work and the work
    avoided relative to two separate base-plus-enriched refinements.  This is a
    cost transformation only: neither start nesting nor optimizer convergence
    supplies a global omitted-mass or derivative certificate.
    """
    base_max_modes, tolerance, scaled_step_tol = (
        _validate_device_mode_plan_arguments(
            base_starts, max_modes, local_radius, coordinate_tol,
            scaled_step_tol, eigenvalue_floor))
    combined = combine_device_start_plans(base_starts, extra_starts)
    if enriched_max_modes is None:
        enriched_max_modes = base_max_modes
    enriched_max_modes, _, _ = _validate_device_mode_plan_arguments(
        combined, enriched_max_modes, local_radius, coordinate_tol,
        scaled_step_tol, eigenvalue_floor)
    refined = refine_all_axis_starts(
        C_A_t, C_B, combined.starts, x_min, x_max,
        time_guard=int(time_guard), iterations=int(iterations), ridge=float(ridge),
        max_step=max_step,
        time_localize_iterations=int(time_localize_iterations),
        live=combined.live)
    base_capacity = base_starts.starts.shape[0]
    base_refined = tuple(value[:base_capacity] for value in refined)
    base_plan, base_ledger = _assemble_all_axis_mode_plan_device(
        C_A_t, base_starts, base_refined, x_min, x_max,
        max_modes=base_max_modes, local_radius=float(local_radius),
        time_guard=int(time_guard), gradient_tol=float(gradient_tol),
        tolerance=tolerance, scaled_step_tol=scaled_step_tol,
        eigenvalue_floor=float(eigenvalue_floor),
        time_reconstruction_certified=time_reconstruction_certified)
    enriched_plan, enriched_ledger = _assemble_all_axis_mode_plan_device(
        C_A_t, combined, refined, x_min, x_max,
        max_modes=enriched_max_modes, local_radius=float(local_radius),
        time_guard=int(time_guard), gradient_tol=float(gradient_tol),
        tolerance=tolerance, scaled_step_tol=scaled_step_tol,
        eigenvalue_floor=float(eigenvalue_floor),
        time_reconstruction_certified=time_reconstruction_certified)
    base_live = jnp.count_nonzero(base_starts.live)
    extra_live = jnp.count_nonzero(extra_starts.live)
    shared_ledger = {
        "n_optimizer_starts_executed": base_live + extra_live,
        "n_optimizer_starts_avoided": base_live,
        "n_optimizer_starts_previous_two_pass": 2 * base_live + extra_live,
        "start_nesting_structural": jnp.asarray(True),
    }
    return (base_plan, enriched_plan, base_ledger, enriched_ledger,
            shared_ledger)


def _legendre_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return (jnp.asarray(nodes, dtype=jnp.float64),
            jnp.asarray(weights, dtype=jnp.float64))


def _mode_integral(coeff, frequency, offset, C_A_shape, C_B, center,
                   transform, radius, nodes, log_weights, concentration):
    if float(concentration) > 0.0:
        scaled = (jnp.sinh(float(concentration) * nodes)
                  / jnp.sinh(float(concentration)))
        log_jacobian_shape = (
            jnp.log(float(concentration))
            + jnp.log(jnp.cosh(float(concentration) * nodes))
            - jnp.log(jnp.sinh(float(concentration))))
    else:
        scaled = nodes
        log_jacobian_shape = jnp.zeros_like(nodes)
    z = radius * scaled
    # Lower-triangular whitening preserves a separable reconstruction topology:
    # t has n points, (t,phi) n^2, (t,phi,u) n^3, and only the final exponent
    # has n^4.  This is the central memory property of the all-axis kernel.
    t = center[0] + transform[0, 0] * z
    phi = jnp.mod(
        center[1] + transform[1, 0] * z[:, None]
        + transform[1, 1] * z[None, :], 2.0 * jnp.pi)
    u = jnp.mod(
        center[2] + transform[2, 0] * z[:, None, None]
        + transform[2, 1] * z[None, :, None]
        + transform[2, 2] * z[None, None, :], 2.0 * jnp.pi)
    x = (center[3] + transform[3, 0] * z[:, None, None, None]
         + transform[3, 1] * z[None, :, None, None]
         + transform[3, 2] * z[None, None, :, None]
         + transform[3, 3] * z[None, None, None, :])

    flat = _evaluate_time_spectrum(coeff, frequency, t, offset)
    C_A = flat.reshape(C_A_shape[:-1] + (nodes.size,))

    kp_a = jnp.arange(C_A.shape[0], dtype=jnp.float64)
    ks_a = jnp.arange(-(C_A.shape[1] - 1) // 2,
                      (C_A.shape[1] - 1) // 2 + 1, dtype=jnp.float64)
    wa = jnp.where(kp_a == 0.0, 1.0, 2.0)
    EA = jnp.exp(1j * (phi[:, :, None, None, None] * kp_a[None, None, None, :, None]
                       + u[:, :, :, None, None] * ks_a[None, None, None, None, :]))
    EA = EA * wa[None, None, None, :, None]
    A = jnp.einsum("tpukq,kqt->tpu", EA, C_A).real

    kp_b = jnp.arange(C_B.shape[0], dtype=jnp.float64)
    ks_b = jnp.arange(-(C_B.shape[1] - 1) // 2,
                      (C_B.shape[1] - 1) // 2 + 1, dtype=jnp.float64)
    wb = jnp.where(kp_b == 0.0, 1.0, 2.0)
    EB = jnp.exp(1j * (phi[:, :, None, None, None] * kp_b[None, None, None, :, None]
                       + u[:, :, :, None, None] * ks_b[None, None, None, None, :]))
    EB = EB * wb[None, None, None, :, None]
    B = jnp.einsum("tpukq,kq->tpu", EB, C_B).real

    exponent = (A[..., None] * x
                - 0.5 * B[..., None] * jnp.square(x)
                - 4.0 * jnp.log(jnp.maximum(x, 1.0e-300)))
    sign, log_det = jnp.linalg.slogdet(transform)
    mapped_log_weights = (log_weights + log_jacobian_shape
                          + jnp.log(radius))
    logw = (mapped_log_weights[:, None, None, None]
            + mapped_log_weights[None, :, None, None]
            + mapped_log_weights[None, None, :, None]
            + mapped_log_weights[None, None, None, :]
            + log_det)
    return jnp.where(sign != 0.0,
                     jax.scipy.special.logsumexp(exponent + logw), -jnp.inf)


def _evaluate_plan_at_order(C_A_t, C_B, plan, order, concentration,
                            time_guard):
    nodes, weights = _legendre_rule(order)
    log_weights = jnp.log(weights)
    coeff, frequency, offset = _time_primitive_spectrum(
        C_A_t.reshape((-1, C_A_t.shape[-1])), int(time_guard))
    model_shape = C_A_t.shape[:-1] + (
        C_A_t.shape[-1] - 2 * int(time_guard),)

    def _step(total, args):
        center, transform, live = args
        def _live_mode(local_args):
            local_center, local_transform = local_args
            return _mode_integral(
                coeff, frequency, offset, model_shape, C_B,
                local_center, local_transform, plan.local_radius, nodes,
                log_weights, concentration)

        value = jax.lax.cond(
            live, _live_mode, lambda _: jnp.asarray(-jnp.inf),
            (center, transform))
        return jnp.logaddexp(total, value), None

    value, _ = jax.lax.scan(
        jax.checkpoint(_step), jnp.asarray(-jnp.inf),
        (plan.centers, plan.local_transforms, plan.live))
    n_live = jnp.count_nonzero(plan.live)
    n_eval = n_live * int(order) ** 4
    # Conservative live-array accounting for one streamed mode.  It is a
    # deterministic shape counter, not a device allocator measurement.
    nt = int(order)
    lanes = int(np.prod(C_A_t.shape[:-1]))
    n_frequency = 2 * C_A_t.shape[-1] - 2
    angle_terms_a = C_A_t.shape[0] * C_A_t.shape[1]
    angle_terms_b = C_B.shape[0] * C_B.shape[1]
    table_bytes = (((nt + lanes) * n_frequency + lanes * nt
                    + nt ** 3 * (angle_terms_a + angle_terms_b)) * 16
                   + C_B.size * 16)
    field_bytes = (n_frequency + 3 * nt ** 3 + nt ** 2
                   + 3 * nt ** 4 + 10 * nt) * 8
    workspace_bytes = jnp.asarray(table_bytes + field_bytes)
    n_selected_time_points = n_live * nt
    n_time_frequency_terms = n_live * nt * n_frequency * lanes
    n_angle_harmonic_terms = (
        n_live * nt ** 3 * (angle_terms_a + angle_terms_b))
    return (value, n_eval, workspace_bytes, n_selected_time_points,
            n_time_frequency_terms, n_angle_harmonic_terms)


def all_axis_peak_local_marginalize(
        C_A_t, C_B, plan, x_min, x_max, *, local_order=5,
        check_order=9, quadrature_tol_nats=1.0e-5,
        outside_tol_nats=-23.0, log_normalization=0.0,
        node_concentration=1.0, time_guard=0,
        time_guard_tol_nats=1.0e-3):
    """Marginalize a padded multi-mode plan with explicit fail-closed ledger.

    ``C_A_t`` is the primitive data table ``(mmax+1,3,Ntime)`` and ``C_B`` is
    the cached ``U,V`` norm table ``(2*mmax+1,5)``.  The returned value is
    diagnostic unless ``ok`` is true.  Here ``ok`` is a validated value-only
    disposition, not a certified quadrature bound or a gradient/Hessian
    certificate.  On any decline the caller must use the
    dense/exact reserve; ``fallback_required`` is provided to make that branch
    hard to omit accidentally.

    With ``time_guard >= 2``, ``C_A_t`` contains support on both sides of the
    target window.  The high-order local integral is repeated after trimming to
    ``time_guard//2`` support and acceptance requires their difference to meet
    ``time_guard_tol_nats``.  A guarded input must pass that comparison and
    cannot be rescued by the plan's external warrant; an unguarded input needs
    the external warrant.  This is an operational convergence validation, not
    a rigorous interpolation-error bound, and the ledger labels it accordingly.
    """
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    _validate_tables(C_A_t, C_B)
    if not isinstance(plan, AllAxisModePlan):
        raise TypeError("plan must be AllAxisModePlan")
    if not (0.0 < float(x_min) < float(x_max)):
        raise ValueError("need 0 < x_min < x_max")
    if int(local_order) < 2 or int(check_order) <= int(local_order):
        raise ValueError("need 2 <= local_order < check_order")
    if float(node_concentration) < 0.0:
        raise ValueError("node_concentration must be non-negative")
    time_guard = int(time_guard)
    n_time = C_A_t.shape[-1] - 2 * time_guard
    if time_guard < 0 or n_time < 2:
        raise ValueError("time_guard must leave at least two integration samples")
    if time_guard == 1:
        raise ValueError("two-guard validation requires time_guard=0 or >=2")
    if float(time_guard_tol_nats) <= 0.0:
        raise ValueError("time_guard_tol_nats must be positive")

    (value_lo, eval_lo, bytes_lo, time_lo,
     time_terms_lo, angle_terms_lo) = _evaluate_plan_at_order(
        C_A_t, C_B, plan, int(local_order), float(node_concentration),
        time_guard)
    (value_hi, eval_hi, bytes_hi, time_hi,
     time_terms_hi, angle_terms_hi) = _evaluate_plan_at_order(
        C_A_t, C_B, plan, int(check_order), float(node_concentration),
        time_guard)
    if time_guard:
        inner_guard = time_guard // 2
        trim = time_guard - inner_guard
        inner_table = C_A_t[..., trim:-trim]
        (value_guard_inner, guard_eval_hi, guard_bytes_hi, guard_time_hi,
         guard_time_terms_hi, guard_angle_terms_hi) = _evaluate_plan_at_order(
            inner_table, C_B, plan, int(check_order),
            float(node_concentration), inner_guard)
        guard_error = jnp.abs(value_hi - value_guard_inner)
        guard_validated = (jnp.isfinite(value_guard_inner)
                           & (guard_error <= float(time_guard_tol_nats)))
    else:
        inner_guard = 0
        value_guard_inner = jnp.asarray(jnp.nan)
        guard_error = jnp.asarray(jnp.inf)
        guard_validated = jnp.asarray(False)
        guard_eval_hi = jnp.asarray(0)
        guard_bytes_hi = jnp.asarray(0)
        guard_time_hi = jnp.asarray(0)
        guard_time_terms_hi = jnp.asarray(0)
        guard_angle_terms_hi = jnp.asarray(0)
    value_lo = value_lo + float(log_normalization)
    value_hi = value_hi + float(log_normalization)
    outside = plan.outside_log_bound + float(log_normalization)
    time_outside = plan.time_outside_log_bound + float(log_normalization)

    centers = plan.centers
    widths = plan.half_widths
    inside_time_distance = (
        (centers[:, 0] - widths[:, 0] >= 0.0)
        & (centers[:, 0] + widths[:, 0] <= n_time - 1.0)
        & (centers[:, 3] - widths[:, 3] >= float(x_min))
        & (centers[:, 3] + widths[:, 3] <= float(x_max)))
    angular_single_cover = jnp.all(widths[:, 1:3] <= jnp.pi, axis=1)
    support_ok = jnp.all(jnp.where(
        plan.live, inside_time_distance & angular_single_cover, True))
    finite = (jnp.all(jnp.isfinite(C_A_t.real))
              & jnp.all(jnp.isfinite(C_A_t.imag))
              & jnp.all(jnp.isfinite(C_B.real))
              & jnp.all(jnp.isfinite(C_B.imag))
              & jnp.isfinite(value_hi)
              & jnp.any(plan.live))
    quadrature_error = jnp.abs(value_hi - value_lo)
    quadrature_ok = quadrature_error <= float(quadrature_tol_nats)
    tail_margin = outside - value_hi
    tail_ok = tail_margin < float(outside_tol_nats)
    # The outside-cover certificate is the correctness-bearing completeness
    # warrant.  Requiring the algebraic root report as well would incorrectly
    # reject an otherwise bounded missed root.  Conversely, a perfect root
    # report cannot replace an integral bound outside the local regions.
    # A supplied guarded reconstruction owns its own convergence check.  Do not
    # let a stale external warrant mask a failed outer/inner comparison.
    time_warranted = (guard_validated if time_guard
                       else plan.time_reconstruction_certified)
    cover_warranted = plan.outside_bound_certified & time_warranted

    decline_nonfinite = ~finite
    decline_incomplete = finite & (~plan.outside_bound_certified)
    decline_time_reconstruction = (
        finite & plan.outside_bound_certified
        & (~time_warranted))
    decline_overlap = finite & cover_warranted & (~plan.boxes_disjoint)
    decline_support = (finite & cover_warranted & plan.boxes_disjoint
                       & (~support_ok))
    decline_quadrature = (finite & cover_warranted & plan.boxes_disjoint & support_ok
                          & (~quadrature_ok))
    decline_tail = (finite & cover_warranted & plan.boxes_disjoint & support_ok
                    & quadrature_ok & (~tail_ok))
    ok = (finite & cover_warranted & plan.boxes_disjoint & support_ok
          & quadrature_ok & tail_ok)
    reconciles = (ok.astype(jnp.int32)
                  + decline_nonfinite.astype(jnp.int32)
                  + decline_incomplete.astype(jnp.int32)
                  + decline_time_reconstruction.astype(jnp.int32)
                  + decline_overlap.astype(jnp.int32)
                  + decline_support.astype(jnp.int32)
                  + decline_quadrature.astype(jnp.int32)
                  + decline_tail.astype(jnp.int32)) == 1
    ledger = {
        "accepted": ok,
        "fallback_required": ~ok,
        "decline_is_waveform_failure": jnp.asarray(False),
        "fixed_plan_autodiff_only": jnp.asarray(True),
        "derivative_warrant_certified": jnp.asarray(False),
        "decline_nonfinite": decline_nonfinite,
        "decline_incomplete": decline_incomplete,
        "decline_time_reconstruction": decline_time_reconstruction,
        "decline_overlap": decline_overlap,
        "decline_support": decline_support,
        "decline_quadrature": decline_quadrature,
        "decline_tail": decline_tail,
        "reconciles": reconciles,
        "enumeration_complete": plan.enumeration_complete,
        "outside_bound_certified": plan.outside_bound_certified,
        "time_reconstruction_certified": plan.time_reconstruction_certified,
        "time_outside_bound_certified": plan.time_outside_bound_certified,
        "time_outside_log_bound": time_outside,
        "time_outside_tail_margin": time_outside - value_hi,
        "time_guard_validated": guard_validated,
        "time_reconstruction_warranted": time_warranted,
        "time_guard_error_certified": jnp.asarray(False),
        "time_guard": jnp.asarray(time_guard),
        "time_guard_inner": jnp.asarray(inner_guard),
        "time_guard_error": guard_error,
        "time_guard_tol_nats": jnp.asarray(float(time_guard_tol_nats)),
        "time_guard_inner_value": value_guard_inner + float(log_normalization),
        "boxes_disjoint": plan.boxes_disjoint,
        "support_ok": support_ok,
        "quadrature_ok": quadrature_ok,
        "quadrature_error_certified": jnp.asarray(False),
        "value_warrant_certified": jnp.asarray(False),
        "tail_ok": tail_ok,
        "quadrature_error": quadrature_error,
        "tail_margin": tail_margin,
        "outside_log_bound": outside,
        "n_modes": jnp.count_nonzero(plan.live),
        "n_mode_capacity": jnp.asarray(plan.live.size),
        "n_local_evaluations_lo": eval_lo,
        "n_local_evaluations_hi": eval_hi,
        "n_guard_local_evaluations_hi": guard_eval_hi,
        "n_total_local_evaluations_hi": eval_hi + guard_eval_hi,
        "n_selected_time_points_lo": time_lo,
        "n_selected_time_points_hi": time_hi,
        "n_guard_selected_time_points_hi": guard_time_hi,
        "n_total_selected_time_points_hi": time_hi + guard_time_hi,
        "n_time_frequency_terms_lo": time_terms_lo,
        "n_time_frequency_terms_hi": time_terms_hi,
        "n_guard_time_frequency_terms_hi": guard_time_terms_hi,
        "n_total_time_frequency_terms_hi": time_terms_hi + guard_time_terms_hi,
        "n_angle_harmonic_terms_lo": angle_terms_lo,
        "n_angle_harmonic_terms_hi": angle_terms_hi,
        "n_guard_angle_harmonic_terms_hi": guard_angle_terms_hi,
        "n_total_angle_harmonic_terms_hi": angle_terms_hi + guard_angle_terms_hi,
        "workspace_bytes_lo": bytes_lo,
        "workspace_bytes_hi": bytes_hi,
        "workspace_bytes_guard_hi": guard_bytes_hi,
        "workspace_bytes_peak_bound_hi": jnp.maximum(bytes_hi, guard_bytes_hi),
    }
    return value_hi, ok, ledger


def empirical_enrichment_marginalize(
        C_A_t, C_B, base_plan, enriched_plan, x_min, x_max, *,
        base_order=13, base_check_order=19,
        enriched_order=19, enriched_check_order=25,
        convergence_tol_nats=1.0e-3, time_guard=0,
        time_guard_tol_nats=1.0e-3, log_normalization=0.0,
        time_outside_tol_nats=-23.0,
        total_value_error_budget_nats=1.0e-3,
        node_concentration=1.0,
        mode_match_tol=(0.25, 1.0e-4, 1.0e-4, 1.0e-5),
        geometry_match_rtol=1.0e-3, geometry_match_atol=1.0e-8):
    """Apply the operational one-step enrichment gate to two fixed plans.

    ``enriched_plan`` must come from a strictly stronger discovery portfolio
    that includes the base starts, and uses the stronger quadrature orders
    supplied here.  Acceptance requires finite values, explicit start capacity,
    disjoint in-support regions, healthy nested quadrature and time-guard
    diagnostics, certified omitted-time bounds for both plans to clear
    ``time_outside_tol_nats``, and agreement within ``convergence_tol_nats``.
    The empirical discovery, nested-quadrature, guarded-time, and certified
    omitted-time contributions must also fit one shared
    ``total_value_error_budget_nats``.  Each of the three paired terms is
    charged once, as the maximum over the base and enriched plans; their
    cancellation-resistant sum is an operational error score, not a formal
    global bound: enrichment and quadrature differences remain empirical
    convergence diagnostics.
    Every base
    mode must recur with matching local geometry.  Additional enriched basins
    are probes, not automatically part of the accepted cover: if a probe has
    broad/overlapping geometry but changes the positive local integral by less
    than the same convergence budget, the valid base cover is retained and its
    value returned.  This prevents a manifestly negligible low-curvature HM
    basin from forcing dense reserve while still making a changed relevant
    basin or an invalid base cover decline.  It does not claim formal global
    completeness or derivative accuracy.

    Any decline returns the finite enriched diagnostic with
    ``fallback_required=True``.  The caller must execute and retain the
    dense/exact reserve; a decline is never a waveform failure.
    """
    if not (int(base_order) < int(base_check_order)
            <= int(enriched_order) < int(enriched_check_order)):
        raise ValueError(
            "need base_order < base_check_order <= enriched_order "
            "< enriched_check_order")
    if float(convergence_tol_nats) <= 0.0:
        raise ValueError("convergence_tol_nats must be positive")
    if (not np.isfinite(float(total_value_error_budget_nats))
            or float(total_value_error_budget_nats) <= 0.0):
        raise ValueError(
            "total_value_error_budget_nats must be finite and positive")
    if (not np.isfinite(float(time_outside_tol_nats))
            or float(time_outside_tol_nats) >= 0.0):
        raise ValueError("time_outside_tol_nats must be finite and negative")
    mode_match_tol = np.asarray(mode_match_tol, dtype=float)
    if mode_match_tol.shape != (4,) or np.any(mode_match_tol <= 0.0):
        raise ValueError("mode_match_tol must contain four positive values")
    if float(geometry_match_rtol) < 0.0 or float(geometry_match_atol) < 0.0:
        raise ValueError("geometry match tolerances must be non-negative")

    base_value, _, base = all_axis_peak_local_marginalize(
        C_A_t, C_B, base_plan, x_min, x_max,
        local_order=int(base_order), check_order=int(base_check_order),
        quadrature_tol_nats=float(convergence_tol_nats),
        log_normalization=float(log_normalization),
        node_concentration=float(node_concentration), time_guard=int(time_guard),
        time_guard_tol_nats=float(time_guard_tol_nats))
    enriched_value, _, enriched = all_axis_peak_local_marginalize(
        C_A_t, C_B, enriched_plan, x_min, x_max,
        local_order=int(enriched_order),
        check_order=int(enriched_check_order),
        quadrature_tol_nats=float(convergence_tol_nats),
        log_normalization=float(log_normalization),
        node_concentration=float(node_concentration), time_guard=int(time_guard),
        time_guard_tol_nats=float(time_guard_tol_nats))

    has_modes = (base["n_modes"] > 0) & (enriched["n_modes"] > 0)
    values_finite = jnp.isfinite(base_value) & jnp.isfinite(enriched_value)
    # An empty padded plan evaluates to -inf by construction.  That is a
    # discovery disposition, not numerical corruption: keep it eligible for
    # the explicit ``decline_no_modes`` branch below.  A nonfinite value from
    # a plan that does contain modes remains a numerical decline.
    finite = values_finite | (~has_modes)
    capacity_ok = (base_plan.discovery_capacity_ok
                   & enriched_plan.discovery_capacity_ok)
    boundary_ok = ~(base_plan.boundary_maximum_pinned
                    | enriched_plan.boundary_maximum_pinned)
    time_ok = (base["time_reconstruction_warranted"]
               & enriched["time_reconstruction_warranted"])
    any_time_cover = (base_plan.time_outside_bound_certified
                      | enriched_plan.time_outside_bound_certified)
    time_cover_pair = (base_plan.time_outside_bound_certified
                       & enriched_plan.time_outside_bound_certified)
    base_time_tail_margin = (base["time_outside_log_bound"] - base_value)
    enriched_time_tail_margin = (
        enriched["time_outside_log_bound"] - enriched_value)
    time_tail_bounds_ok = (
        (base_time_tail_margin < float(time_outside_tol_nats))
        & (enriched_time_tail_margin < float(time_outside_tol_nats)))
    time_omitted_ok = time_cover_pair & time_tail_bounds_ok
    base_geometry_ok = base["boxes_disjoint"] & base["support_ok"]
    enriched_geometry_ok = (enriched["boxes_disjoint"]
                            & enriched["support_ok"])
    quadrature_ok = base["quadrature_ok"] & enriched["quadrature_ok"]
    delta = jnp.abs(base_plan.centers[:, None, :]
                    - enriched_plan.centers[None, :, :])
    angular_delta = jnp.abs(jnp.mod(
        delta[..., 1:3] + jnp.pi, 2.0 * jnp.pi) - jnp.pi)
    delta = delta.at[..., 1:3].set(angular_delta)
    matches = (jnp.all(delta <= jnp.asarray(mode_match_tol), axis=-1)
               & enriched_plan.live[None, :])
    width_scale = (float(geometry_match_atol)
                   + float(geometry_match_rtol)
                   * jnp.abs(base_plan.half_widths[:, None, :]))
    width_matches = jnp.all(
        jnp.abs(base_plan.half_widths[:, None, :]
                - enriched_plan.half_widths[None, :, :]) <= width_scale,
        axis=-1)
    transform_scale = (float(geometry_match_atol)
                       + float(geometry_match_rtol)
                       * jnp.abs(base_plan.local_transforms[:, None, :, :]))
    transform_matches = jnp.all(
        jnp.abs(base_plan.local_transforms[:, None, :, :]
                - enriched_plan.local_transforms[None, :, :, :])
        <= transform_scale, axis=(-2, -1))
    geometry_matches = matches & width_matches & transform_matches
    # Padded base rows are vacuously retained.  A stronger plan may add modes,
    # but it may not silently lose one that contributed to the base value.
    mode_nesting_ok = jnp.all(
        jnp.where(base_plan.live, jnp.any(matches, axis=1), True))
    geometry_nesting_ok = jnp.all(jnp.where(
        base_plan.live, jnp.any(geometry_matches, axis=1), True))
    geometry_ok = base_geometry_ok & geometry_nesting_ok
    convergence_error = jnp.abs(enriched_value - base_value)
    converged = convergence_error <= float(convergence_tol_nats)
    # A single operational allowance prevents several individually acceptable
    # diagnostics from silently spending the full science tolerance apiece.
    # Sums, rather than maxima, are deliberate: base/enriched quadrature or
    # guard errors can cancel in their observed difference.  Unguarded plans
    # contribute zero here only after their external reconstruction warrants
    # have passed ``time_ok`` above.  The discarded-time terms are genuine
    # integral bounds, converted to maximum log-integral corrections, while
    # the other terms remain empirical diagnostics.
    base_guard_score = jnp.where(
        int(time_guard) > 0, base["time_guard_error"], 0.0)
    enriched_guard_score = jnp.where(
        int(time_guard) > 0, enriched["time_guard_error"], 0.0)
    base_time_tail_correction = jnp.logaddexp(
        0.0, base_time_tail_margin)
    enriched_time_tail_correction = jnp.logaddexp(
        0.0, enriched_time_tail_margin)
    # Each paired term is the same physical quantity measured on the two
    # nested plans (same table, same recurring modes, same G vs G/2
    # comparison).  Charging both against the budget double-counted a
    # common-mode error: on the analytic wiring fixture at ten times the
    # unit amplitude a value correct to 1.5e-4 nat was refused at a score of
    # 1.06e-3, of which 2 x 5.27e-4 was one guard discrepancy counted twice.
    # The maximum over the pair bounds whichever plan's value is selected and
    # counts it once.  The per-plan terms stay in the ledger.
    quadrature_score = jnp.maximum(
        base["quadrature_error"], enriched["quadrature_error"])
    guard_score = jnp.maximum(base_guard_score, enriched_guard_score)
    tail_score = jnp.maximum(
        base_time_tail_correction, enriched_time_tail_correction)
    empirical_value_error_score = (
        convergence_error + quadrature_score + guard_score + tail_score)
    error_budget_complete = time_cover_pair & time_ok
    error_budget_ok = (
        error_budget_complete
        & jnp.isfinite(empirical_value_error_score)
        & (empirical_value_error_score
           <= float(total_value_error_budget_nats)))

    decline_nonfinite = ~finite
    decline_capacity = finite & (~capacity_ok)
    decline_no_modes = finite & capacity_ok & (~has_modes)
    decline_boundary_maximum = (finite & capacity_ok & has_modes
                                & (~boundary_ok))
    decline_mode_nesting = (finite & capacity_ok & has_modes & boundary_ok
                            & (~mode_nesting_ok))
    decline_time = (finite & capacity_ok & has_modes & boundary_ok
                    & mode_nesting_ok & (~time_ok))
    decline_time_cover = (
        finite & capacity_ok & has_modes & boundary_ok & mode_nesting_ok
        & time_ok & (~time_cover_pair))
    decline_time_omitted_bound = (
        finite & capacity_ok & has_modes & boundary_ok & mode_nesting_ok
        & time_ok & time_cover_pair & (~time_tail_bounds_ok))
    # Umbrella science diagnostic retained for callers that need only the
    # broad reason.  The two exclusive fields above own reconciliation.
    decline_time_omitted = decline_time_cover | decline_time_omitted_bound
    decline_geometry = (finite & capacity_ok & has_modes & boundary_ok
                        & mode_nesting_ok & time_ok & time_omitted_ok
                        & (~geometry_ok))
    decline_quadrature = (finite & capacity_ok & has_modes & boundary_ok
                          & mode_nesting_ok & time_ok & time_omitted_ok
                          & geometry_ok & (~quadrature_ok))
    decline_enrichment = (finite & capacity_ok & has_modes & boundary_ok
                          & mode_nesting_ok & time_ok & time_omitted_ok
                          & geometry_ok & quadrature_ok & (~converged))
    decline_error_budget = (
        finite & capacity_ok & has_modes & boundary_ok & mode_nesting_ok
        & time_ok & time_omitted_ok & geometry_ok & quadrature_ok & converged
        & (~error_budget_ok))
    accepted = (finite & capacity_ok & has_modes & boundary_ok
                & mode_nesting_ok & time_ok & time_omitted_ok & geometry_ok
                & quadrature_ok & converged & error_budget_ok)
    accepted_value_uses_base_geometry = accepted & (~enriched_geometry_ok)
    accepted_value = jnp.where(
        accepted_value_uses_base_geometry, base_value, enriched_value)
    reconciles = (
        accepted.astype(jnp.int32)
        + decline_nonfinite.astype(jnp.int32)
        + decline_capacity.astype(jnp.int32)
        + decline_no_modes.astype(jnp.int32)
        + decline_boundary_maximum.astype(jnp.int32)
        + decline_mode_nesting.astype(jnp.int32)
        + decline_time.astype(jnp.int32)
        + decline_time_cover.astype(jnp.int32)
        + decline_time_omitted_bound.astype(jnp.int32)
        + decline_geometry.astype(jnp.int32)
        + decline_quadrature.astype(jnp.int32)
        + decline_enrichment.astype(jnp.int32)
        + decline_error_budget.astype(jnp.int32)) == 1
    ledger = {
        "accepted": accepted,
        "fallback_required": ~accepted,
        "decline_is_waveform_failure": jnp.asarray(False),
        "acceptance_is_empirical_enrichment": jnp.asarray(True),
        "global_completeness_certified": jnp.asarray(False),
        "empirical_value_error_certified": jnp.asarray(False),
        "derivative_warrant_certified": jnp.asarray(False),
        "decline_nonfinite": decline_nonfinite,
        "decline_capacity": decline_capacity,
        "decline_no_modes": decline_no_modes,
        "decline_boundary_maximum": decline_boundary_maximum,
        "boundary_maximum_ok": boundary_ok,
        "decline_mode_nesting": decline_mode_nesting,
        "decline_time_reconstruction": decline_time,
        "decline_time_cover_incomplete": decline_time_cover,
        "decline_time_omitted_mass_bound": decline_time_omitted_bound,
        "decline_time_omitted_mass": decline_time_omitted,
        "decline_geometry": decline_geometry,
        "decline_quadrature": decline_quadrature,
        "decline_enrichment": decline_enrichment,
        "decline_error_budget": decline_error_budget,
        "reconciles": reconciles,
        "base_value": base_value,
        "enriched_value": enriched_value,
        "base_and_enriched_values_finite": values_finite,
        "convergence_error": convergence_error,
        "convergence_tol_nats": jnp.asarray(float(convergence_tol_nats)),
        "empirical_value_error_score_nats": empirical_value_error_score,
        "total_value_error_budget_nats": jnp.asarray(
            float(total_value_error_budget_nats)),
        "value_error_budget_complete": error_budget_complete,
        "value_error_budget_ok": error_budget_ok,
        "value_error_budget_is_empirical": jnp.asarray(True),
        "value_error_budget_is_formal_bound": jnp.asarray(False),
        "error_score_discovery_nats": convergence_error,
        "error_score_quadrature_nats": quadrature_score,
        "error_score_time_guard_nats": guard_score,
        "error_score_omitted_time_nats": tail_score,
        "error_score_pairs_charged_as_max": jnp.asarray(True),
        "error_score_base_quadrature_nats": base["quadrature_error"],
        "error_score_enriched_quadrature_nats":
            enriched["quadrature_error"],
        "error_score_base_time_guard_nats": base_guard_score,
        "error_score_enriched_time_guard_nats": enriched_guard_score,
        "error_score_base_omitted_time_nats": base_time_tail_correction,
        "error_score_enriched_omitted_time_nats":
            enriched_time_tail_correction,
        "base_capacity_ok": base_plan.discovery_capacity_ok,
        "enriched_capacity_ok": enriched_plan.discovery_capacity_ok,
        "base_n_modes": base["n_modes"],
        "enriched_n_modes": enriched["n_modes"],
        "mode_nesting_ok": mode_nesting_ok,
        "geometry_nesting_ok": geometry_nesting_ok,
        "base_geometry_ok": base_geometry_ok,
        "enriched_geometry_ok": enriched_geometry_ok,
        "accepted_value_uses_base_geometry":
            accepted_value_uses_base_geometry,
        "base_quadrature_error": base["quadrature_error"],
        "enriched_quadrature_error": enriched["quadrature_error"],
        "base_time_guard_error": base["time_guard_error"],
        "enriched_time_guard_error": enriched["time_guard_error"],
        "time_outside_cover_used": time_cover_pair,
        "time_outside_cover_any": any_time_cover,
        "time_outside_cover_pair": time_cover_pair,
        "time_omitted_mass_ok": time_omitted_ok,
        "time_outside_tol_nats": jnp.asarray(float(time_outside_tol_nats)),
        "base_time_outside_tail_margin": base_time_tail_margin,
        "enriched_time_outside_tail_margin": enriched_time_tail_margin,
        "base_total_local_evaluations_hi":
            base["n_total_local_evaluations_hi"],
        "enriched_total_local_evaluations_hi":
            enriched["n_total_local_evaluations_hi"],
        "total_local_evaluations_hi": (
            base["n_total_local_evaluations_hi"]
            + enriched["n_total_local_evaluations_hi"]),
        "workspace_bytes_peak_bound_hi": jnp.maximum(
            base["workspace_bytes_peak_bound_hi"],
            enriched["workspace_bytes_peak_bound_hi"]),
    }
    return accepted_value, accepted, ledger


def empirical_enrichment_with_exact_reserve(
        C_A_t, C_B, base_plan, enriched_plan, x_min, x_max, *,
        reserve_x_grid, reserve_log_weights, time_weights,
        reserve_amp_sizing, reserve_m_max=None,
        reserve_dense_chunk=8, reserve_grid_block=32,
        reserve_time_nodes=None, reserve_time_resolution_warranted=False,
        reserve_time_check_value=np.nan,
        reserve_time_check_nodes=None, reserve_time_check_weights=None,
        reserve_time_resolution_tol_nats=1.0e-3,
        base_order=13, base_check_order=19,
        enriched_order=19, enriched_check_order=25,
        convergence_tol_nats=1.0e-3, time_guard=0,
        time_guard_tol_nats=1.0e-3, local_log_normalization=0.0,
        time_outside_tol_nats=-23.0,
        total_value_error_budget_nats=1.0e-3,
        reserve_log_offset=0.0, node_concentration=1.0,
        mode_match_tol=(0.25, 1.0e-4, 1.0e-4, 1.0e-5),
        reserve_angular_kernel=None, reserve_time_focus=None,
        reserve_time_cover=None):
    """Select an accepted local value or execute the exact table reserve.

    This is the first operational fixed-point composition seam.  The caller
    supplies two immutable host- or device-built mode plans; this device
    function evaluates the empirical gate and uses
    :func:`anglemarg.coefficient_table_distphipsimarg_exact` only on a decline.
    Thus accepted high-SNR rows pay fixed local work per retained mode after
    bounded discovery; broad,
    unresolved, capacity-limited, or otherwise unhealthy rows retain the sample
    through the established dense/exact coefficient reserve.

    Measures remain explicit.  ``reserve_log_weights`` owns the fixed-grid
    distance quadrature measure and ``time_weights`` owns the reserve time
    integral.  If ``reserve_time_nodes`` is supplied, it names sub-sample
    positions in the unguarded target window and the coefficient table is
    reconstructed there only inside the declined branch.  This band-limited
    reserve must cover either the complete target window or the contiguous
    interval enclosing every retained scout cell.  A cropped interval is usable
    only with the plan's discarded-time bound.  In both cases the node spacing
    must remain everywhere finer than one native sample and carry an external
    resolution warrant through ``reserve_time_resolution_warranted`` and the
    independently evaluated lower-resolution ``reserve_time_check_value``, as
    well as the same two-guard convergence comparison used by the local path.
    Resolution error, guard error, and the maximum omitted-time correction are
    charged to one ``total_value_error_budget_nats`` allowance.  Without
    nodes, the legacy reserve uses native target samples only as a diagnostic
    and is always fail-closed.  Native-sample angular exactness does not certify
    a time integral once its peak is narrower than a sample.  The external
    warrant is a scalar JAX boolean so callers may bind it to a per-row
    convergence record; the caller remains responsible for proving that it
    describes the exact nodes, weights, and full interval passed here.
    Alternatively ``reserve_time_check_nodes``/``reserve_time_check_weights``
    name a strictly coarser rule on the same target window.  The controller
    then evaluates that rule itself, inside the declined branch only, and the
    resolution warrant is structural: both rules come from the same reflected
    primitive, the check rule is coarser, and the two values must agree within
    ``reserve_time_resolution_tol_nats``.  An accepted local row never pays
    for either reserve evaluation.

    If ``JAX_ILE_DISTMARG_GH`` is active, the established reserve
    instead reads the support from ``reserve_x_grid`` and uses its normalized
    volumetric ``x**-4`` measure.  The
    local branch owns a continuous ``x**-4 dx dtime_sample dphi du`` integral,
    so ``local_log_normalization`` must convert that measure to the reserve's
    normalization.  ``reserve_log_offset`` is the reserve's separately
    recorded global log-measure conversion.  It is applied to both the
    included reserve value and a cropped reserve's discarded-time bound, so
    their omitted/included ratio is invariant to that conversion.  Neither
    conversion is inferred from a distance-prior name.  This prevents an
    unnormalized prototype value from silently replacing a production result.

    ``reserve_angular_kernel(target_table, C_B) -> lnL_t`` is the angular
    and distance kernel evaluated on the reserve's time nodes; ``None`` is
    :func:`anglemarg.coefficient_table_distphipsimarg_exact` with the
    arguments above.  ``reserve_time_focus=(centre, half_width)`` (samples)
    names where the rule is fine; the node carrying the largest evaluated
    ``lnL_t`` must then lie within ``half_width`` of ``centre`` or the rule
    is unwarranted (``reserve_time_focus_ok``).  A peak-local rule and its
    check share the block, so a misplaced block agrees with itself; this is
    the certificate that sees it (measured: 0.22 nat, warranted, on a
    rung-160 row with the block 0.17 samples off the maximum).
    ``reserve_time_cover=(lo, hi, outside_log_bound)`` replaces the plans'
    time cover in the cropped-cover warrant: the rule must span ``[lo, hi]``
    (samples) and ``outside_log_bound`` (already in the reserve's units) is
    charged as the omitted mass.  It is how a support-limited reserve rule
    certifies what it left out without leaning on the local branch's plan.  A peak-local time rule (``peaklocal_time_reserve``)
    may repeat a position; a repeated position carries zero weight, so the
    node checks below ask for a NON-DECREASING rule.

    Ledger field ``accepted_local`` is the empirical local disposition, while
    the returned ``usable`` describes the selected result after reserve
    execution.  A local
    decline is never a waveform failure.  A nonfinite reserve is reported as an
    integration failure and remains unusable; it is not relabeled as a missed
    waveform evaluation.
    """
    from . import anglemarg as _anglemarg
    from .core import _time_marginalize

    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    time_guard = int(time_guard)
    n_target = C_A_t.shape[-1] - 2 * time_guard
    time_weights = jnp.asarray(time_weights, dtype=jnp.float64)
    use_bandlimited_time = reserve_time_nodes is not None
    if use_bandlimited_time:
        if time_guard < 2:
            raise ValueError(
                "band-limited reserve requires time_guard >= 2")
        reserve_time_nodes = jnp.asarray(
            reserve_time_nodes, dtype=jnp.float64)
        if reserve_time_nodes.ndim != 1 or reserve_time_nodes.size < 2:
            raise ValueError(
                "reserve_time_nodes must be a one-dimensional rule")
        if (time_weights.ndim != 1
                or time_weights.shape[0] != reserve_time_nodes.shape[0]):
            raise ValueError(
                "time_weights must match reserve_time_nodes")
    elif time_weights.ndim != 1 or time_weights.shape[0] != n_target:
        raise ValueError("time_weights must match the unguarded target window")
    reserve_time_resolution_warranted = jnp.asarray(
        reserve_time_resolution_warranted, dtype=bool)
    if reserve_time_resolution_warranted.ndim != 0:
        raise ValueError("reserve_time_resolution_warranted must be scalar")
    reserve_time_check_value = jnp.asarray(
        reserve_time_check_value, dtype=jnp.float64)
    if reserve_time_check_value.ndim != 0:
        raise ValueError("reserve_time_check_value must be scalar")
    use_internal_check = reserve_time_check_nodes is not None
    if use_internal_check:
        if not use_bandlimited_time:
            raise ValueError(
                "reserve_time_check_nodes requires reserve_time_nodes")
        if reserve_time_check_weights is None:
            raise ValueError(
                "reserve_time_check_nodes requires reserve_time_check_weights")
        reserve_time_check_nodes = jnp.asarray(
            reserve_time_check_nodes, dtype=jnp.float64)
        reserve_time_check_weights = jnp.asarray(
            reserve_time_check_weights, dtype=jnp.float64)
        if (reserve_time_check_nodes.ndim != 1
                or reserve_time_check_nodes.size < 2
                or reserve_time_check_weights.shape
                != reserve_time_check_nodes.shape):
            raise ValueError(
                "reserve_time_check_nodes/weights must be one matching rule")
    if (not np.isfinite(float(reserve_time_resolution_tol_nats))
            or not float(reserve_time_resolution_tol_nats) > 0.0):
        raise ValueError(
            "reserve_time_resolution_tol_nats must be finite and positive")
    if reserve_m_max is None:
        reserve_m_max = int(C_A_t.shape[0] - 1)
    reserve_m_max = int(reserve_m_max)
    if reserve_angular_kernel is None:
        def reserve_angular_kernel(table, norm_table):
            return _anglemarg.coefficient_table_distphipsimarg_exact(
                table, norm_table, reserve_x_grid, reserve_log_weights,
                amp_sizing=float(reserve_amp_sizing), m_max=reserve_m_max,
                dense_chunk=int(reserve_dense_chunk),
                grid_block=int(reserve_grid_block))

    local_value, accepted_local, local_ledger = empirical_enrichment_marginalize(
        C_A_t, C_B, base_plan, enriched_plan, x_min, x_max,
        base_order=int(base_order), base_check_order=int(base_check_order),
        enriched_order=int(enriched_order),
        enriched_check_order=int(enriched_check_order),
        convergence_tol_nats=float(convergence_tol_nats),
        time_guard=time_guard,
        time_guard_tol_nats=float(time_guard_tol_nats),
        time_outside_tol_nats=float(time_outside_tol_nats),
        total_value_error_budget_nats=float(total_value_error_budget_nats),
        log_normalization=float(local_log_normalization),
        node_concentration=float(node_concentration),
        mode_match_tol=mode_match_tol)

    def _reserve(_):
        if use_bandlimited_time:
            flat_table = C_A_t.reshape((-1, C_A_t.shape[-1]))
            coeff, frequency, offset = _time_primitive_spectrum(
                flat_table, time_guard)
            target_table = _evaluate_time_spectrum(
                coeff, frequency, reserve_time_nodes, offset).reshape(
                    C_A_t.shape[:-1] + (reserve_time_nodes.size,))
        elif time_guard:
            target_table = C_A_t[..., time_guard:-time_guard]
        else:
            target_table = C_A_t
        lnL_t = reserve_angular_kernel(target_table, C_B)
        reserve_value = (_time_marginalize(lnL_t, time_weights)[0]
                         + float(reserve_log_offset))
        if use_bandlimited_time:
            inner_guard = time_guard // 2
            trim = time_guard - inner_guard
            inner_table = C_A_t[..., trim:-trim]
            inner_flat = inner_table.reshape((-1, inner_table.shape[-1]))
            coeff_inner, frequency_inner, offset_inner = (
                _time_primitive_spectrum(inner_flat, inner_guard))
            target_inner = _evaluate_time_spectrum(
                coeff_inner, frequency_inner, reserve_time_nodes,
                offset_inner).reshape(
                    C_A_t.shape[:-1] + (reserve_time_nodes.size,))
            lnL_inner = reserve_angular_kernel(target_inner, C_B)
            guard_value = (_time_marginalize(lnL_inner, time_weights)[0]
                           + float(reserve_log_offset))
        else:
            guard_value = jnp.asarray(jnp.nan, dtype=jnp.float64)
        if use_internal_check:
            check_table = _evaluate_time_spectrum(
                coeff, frequency, reserve_time_check_nodes, offset).reshape(
                    C_A_t.shape[:-1] + (reserve_time_check_nodes.size,))
            lnL_check = reserve_angular_kernel(check_table, C_B)
            check_value = (
                _time_marginalize(lnL_check, reserve_time_check_weights)[0]
                + float(reserve_log_offset))
        else:
            check_value = reserve_time_check_value
        if use_bandlimited_time:
            peak_node = reserve_time_nodes[jnp.argmax(lnL_t[0])]
        else:
            peak_node = jnp.asarray(jnp.nan, dtype=jnp.float64)
        return reserve_value, reserve_value, guard_value, check_value, peak_node

    def _accepted(_):
        nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
        return local_value, nan, nan, reserve_time_check_value, nan

    # Both branches are rematerialized.  Reverse-mode AD through ``lax.cond``
    # stores backward residuals for BOTH branches whatever the predicate, and
    # the dense reserve's residuals at production amplitude are tens of GiB
    # (a 158 GiB allocation was requested on a rho 163 production row, PR #278
    # follow-up).  With checkpointing the backward pass recomputes the taken
    # branch instead, so gradient memory is one forward evaluation.
    (selected_value, reserve_value, reserve_guard_value,
     reserve_time_check_value, reserve_time_peak_node) = jax.lax.cond(
        accepted_local, jax.checkpoint(_accepted), jax.checkpoint(_reserve),
        operand=None)
    if reserve_time_focus is None:
        reserve_time_focus_ok = jnp.asarray(True)
        reserve_time_focus_offset = jnp.asarray(jnp.nan, dtype=jnp.float64)
    else:
        focus_centre = jnp.asarray(reserve_time_focus[0], dtype=jnp.float64)
        focus_half = jnp.asarray(reserve_time_focus[1], dtype=jnp.float64)
        reserve_time_focus_offset = jnp.abs(reserve_time_peak_node - focus_centre)
        reserve_time_focus_ok = (
            (~(~accepted_local))
            | (jnp.isfinite(reserve_time_focus_offset)
               & (reserve_time_focus_offset <= focus_half)))
    if use_internal_check:
        # Structural warrant: same primitive, same window, strictly coarser
        # check rule with a valid measure.  The value comparison itself is
        # still applied below through reserve_time_resolution_validated.
        check_rule_valid = (
            jnp.all(jnp.isfinite(reserve_time_check_nodes))
            & jnp.all(reserve_time_check_nodes[1:] >= reserve_time_check_nodes[:-1])
            & (reserve_time_check_nodes[0] == reserve_time_nodes[0])
            & (reserve_time_check_nodes[-1] == reserve_time_nodes[-1])
            & (jnp.max(jnp.diff(reserve_time_check_nodes))
               > jnp.max(jnp.diff(reserve_time_nodes)))
            & jnp.all(jnp.isfinite(reserve_time_check_weights))
            & jnp.all(reserve_time_check_weights >= 0.0)
            & (jnp.sum(reserve_time_check_weights) > 0.0))
        reserve_time_resolution_warranted = (
            reserve_time_resolution_warranted | check_rule_valid)
    reserve_executed = ~accepted_local
    reserve_finite = jnp.isfinite(reserve_value)
    if use_bandlimited_time:
        reserve_time_nodes_finite = jnp.all(jnp.isfinite(reserve_time_nodes))
        # Non-decreasing: a peak-local rule repeats a position where a mode
        # slot is dead or a block is clipped, and the repeat carries no weight.
        # Compared as slices, not as ``diff >= 0``: under jit XLA fuses the
        # subtraction of two bitwise-equal products into a multiply-add that
        # rounds to -1e-15 (measured), which would fail a rule numpy calls
        # sorted.
        reserve_time_nodes_increasing = jnp.all(
            reserve_time_nodes[1:] >= reserve_time_nodes[:-1])
        reserve_time_subsampled = jnp.max(
            jnp.diff(reserve_time_nodes)) < 1.0
        reserve_time_weights_valid = (
            jnp.all(jnp.isfinite(time_weights))
            & jnp.all(time_weights >= 0.0)
            & (jnp.sum(time_weights) > 0.0))
        reserve_time_nodes_in_support = jnp.all(
            (reserve_time_nodes >= 0.0)
            & (reserve_time_nodes <= float(n_target - 1)))
        reserve_time_nodes_cover_target = (
            (reserve_time_nodes[0] == 0.0)
            & (reserve_time_nodes[-1] == float(n_target - 1)))
        if reserve_time_cover is None:
            reserve_required_time_min = jnp.minimum(
                base_plan.time_cover_min_sample,
                enriched_plan.time_cover_min_sample)
            reserve_required_time_max = jnp.maximum(
                base_plan.time_cover_max_sample,
                enriched_plan.time_cover_max_sample)
            reserve_time_plan_cover_certified = (
                base_plan.time_outside_bound_certified
                & enriched_plan.time_outside_bound_certified)
            cover_outside_log_bound = (
                jnp.minimum(base_plan.time_outside_log_bound,
                            enriched_plan.time_outside_log_bound)
                + float(local_log_normalization)
                + float(reserve_log_offset))
        else:
            reserve_required_time_min = jnp.asarray(reserve_time_cover[0], dtype=jnp.float64)
            reserve_required_time_max = jnp.asarray(reserve_time_cover[1], dtype=jnp.float64)
            reserve_time_plan_cover_certified = jnp.asarray(True)
            cover_outside_log_bound = jnp.asarray(reserve_time_cover[2], dtype=jnp.float64)
        reserve_time_plan_cover_finite = (
            jnp.isfinite(reserve_required_time_min)
            & jnp.isfinite(reserve_required_time_max)
            & (reserve_required_time_min < reserve_required_time_max))
        reserve_time_nodes_cover_plans = (
            (reserve_time_nodes[0] <= reserve_required_time_min)
            & (reserve_time_nodes[-1] >= reserve_required_time_max))
        reserve_time_cropped_cover_warranted = (
            reserve_time_plan_cover_finite
            & reserve_time_nodes_cover_plans
            & reserve_time_plan_cover_certified)
        reserve_time_interval_warranted = (
            reserve_time_nodes_cover_target
            | reserve_time_cropped_cover_warranted)
        reserve_time_outside_log_bound = jnp.where(
            reserve_time_nodes_cover_target, -jnp.inf, cover_outside_log_bound)
        reserve_time_tail_margin = (
            reserve_time_outside_log_bound - reserve_value)
        reserve_time_tail_correction = jnp.logaddexp(
            0.0, reserve_time_tail_margin)
        reserve_time_tail_ok = (
            reserve_time_tail_margin < float(time_outside_tol_nats))
        reserve_time_guard_error = jnp.abs(
            reserve_value - reserve_guard_value)
        reserve_time_guard_validated = (
            reserve_executed & reserve_finite
            & jnp.isfinite(reserve_guard_value)
            & (reserve_time_guard_error <= float(time_guard_tol_nats)))
        reserve_time_resolution_error = jnp.abs(
            reserve_value - reserve_time_check_value)
        reserve_time_resolution_validated = (
            reserve_executed & reserve_finite
            & reserve_time_resolution_warranted
            & jnp.isfinite(reserve_time_check_value)
            & (reserve_time_resolution_error
               <= float(reserve_time_resolution_tol_nats)))
        reserve_time_error_score = (
            reserve_time_guard_error + reserve_time_resolution_error
            + reserve_time_tail_correction)
        reserve_time_error_budget_ok = (
            jnp.isfinite(reserve_time_error_score)
            & (reserve_time_error_score
               <= float(total_value_error_budget_nats)))
        reserve_time_warranted = (
            reserve_time_focus_ok
            & reserve_time_nodes_finite
            & reserve_time_nodes_increasing
            & reserve_time_subsampled
            & reserve_time_weights_valid
            & reserve_time_nodes_in_support
            & reserve_time_interval_warranted
            & reserve_time_tail_ok
            & reserve_time_resolution_validated
            & reserve_time_guard_validated
            & reserve_time_error_budget_ok)
        reserve_time_points = reserve_time_nodes.size
        reserve_time_min = jnp.min(reserve_time_nodes)
        reserve_time_max = jnp.max(reserve_time_nodes)
    else:
        reserve_time_nodes_finite = jnp.asarray(True)
        reserve_time_nodes_increasing = jnp.asarray(True)
        reserve_time_subsampled = jnp.asarray(False)
        reserve_time_weights_valid = (
            jnp.all(jnp.isfinite(time_weights))
            & jnp.all(time_weights >= 0.0)
            & (jnp.sum(time_weights) > 0.0))
        reserve_time_nodes_in_support = jnp.asarray(True)
        reserve_time_nodes_cover_target = jnp.asarray(True)
        reserve_required_time_min = jnp.asarray(0.0)
        reserve_required_time_max = jnp.asarray(float(n_target - 1))
        reserve_time_plan_cover_finite = jnp.asarray(False)
        reserve_time_nodes_cover_plans = jnp.asarray(False)
        reserve_time_plan_cover_certified = jnp.asarray(False)
        reserve_time_cropped_cover_warranted = jnp.asarray(False)
        reserve_time_interval_warranted = jnp.asarray(False)
        reserve_time_outside_log_bound = jnp.asarray(jnp.inf)
        reserve_time_tail_margin = jnp.asarray(jnp.inf)
        reserve_time_tail_correction = jnp.asarray(jnp.inf)
        reserve_time_tail_ok = jnp.asarray(False)
        reserve_time_guard_error = jnp.asarray(jnp.nan)
        reserve_time_guard_validated = jnp.asarray(False)
        reserve_time_resolution_error = jnp.asarray(jnp.nan)
        reserve_time_resolution_validated = jnp.asarray(False)
        reserve_time_error_score = jnp.asarray(jnp.inf)
        reserve_time_error_budget_ok = jnp.asarray(False)
        reserve_time_warranted = jnp.asarray(False)
        reserve_time_points = n_target
        reserve_time_min = jnp.asarray(0.0)
        reserve_time_max = jnp.asarray(float(n_target - 1))
    reserve_time_failed = reserve_executed & (~reserve_time_warranted)
    reserve_failed = reserve_executed & (
        (~reserve_finite) | reserve_time_failed)
    usable = accepted_local | (
        reserve_executed & reserve_finite & reserve_time_warranted)
    nphi_reserve, nu_reserve = _anglemarg._dense_grid_sizes(
        float(reserve_amp_sizing), m_max=reserve_m_max)
    reserve_gh_nodes = int(_anglemarg._core._DISTMARG_GH_N)
    reserve_input_distance_points = int(jnp.asarray(reserve_x_grid).size)
    ledger = dict(local_ledger)
    ledger.update({
        "local_fallback_required": local_ledger["fallback_required"],
        "local_reconciles": local_ledger["reconciles"],
    })
    ledger.update({
        "accepted": usable,
        "fallback_required": reserve_failed,
        "reconciles": (
            usable.astype(jnp.int32)
            + reserve_failed.astype(jnp.int32)) == 1,
        "accepted_local": accepted_local,
        "selected_value_is_local": accepted_local,
        "selected_value_is_warranted_reserve": (
            reserve_executed & reserve_finite & reserve_time_warranted),
        "reserve_executed": reserve_executed,
        "reserve_value": reserve_value,
        "reserve_finite": reserve_finite,
        "reserve_failed": reserve_failed,
        "reserve_time_failed": reserve_time_failed,
        "reserve_uses_bandlimited_time": jnp.asarray(use_bandlimited_time),
        "reserve_uses_native_time": jnp.asarray(not use_bandlimited_time),
        "reserve_native_time_warranted": jnp.asarray(False),
        "reserve_time_resolution_warranted": (
            reserve_time_resolution_warranted),
        "reserve_time_check_rule_internal": jnp.asarray(use_internal_check),
        "reserve_time_peak_node": reserve_time_peak_node,
        "reserve_time_focus_offset_samples": reserve_time_focus_offset,
        "reserve_time_focus_ok": reserve_time_focus_ok,
        "reserve_time_check_value": reserve_time_check_value,
        "reserve_time_resolution_error_nats": (
            reserve_time_resolution_error),
        "reserve_time_resolution_tol_nats": jnp.asarray(
            float(reserve_time_resolution_tol_nats)),
        "reserve_time_resolution_validated": (
            reserve_time_resolution_validated),
        "reserve_time_nodes_finite": reserve_time_nodes_finite,
        "reserve_time_nodes_increasing": reserve_time_nodes_increasing,
        "reserve_time_subsampled": reserve_time_subsampled,
        "reserve_time_weights_valid": reserve_time_weights_valid,
        "reserve_time_nodes_in_support": reserve_time_nodes_in_support,
        "reserve_time_nodes_cover_target": reserve_time_nodes_cover_target,
        "reserve_required_time_min_sample": reserve_required_time_min,
        "reserve_required_time_max_sample": reserve_required_time_max,
        "reserve_time_plan_cover_finite": reserve_time_plan_cover_finite,
        "reserve_time_nodes_cover_plans": reserve_time_nodes_cover_plans,
        "reserve_time_plan_cover_certified": (
            reserve_time_plan_cover_certified),
        "reserve_time_cropped_cover_warranted": (
            reserve_time_cropped_cover_warranted),
        "reserve_time_interval_warranted": reserve_time_interval_warranted,
        "reserve_time_outside_log_bound": reserve_time_outside_log_bound,
        "reserve_time_tail_margin": reserve_time_tail_margin,
        "reserve_time_tail_correction_nats": (
            reserve_time_tail_correction),
        "reserve_time_tail_ok": reserve_time_tail_ok,
        "reserve_time_guard_validated": reserve_time_guard_validated,
        "reserve_time_guard_error": reserve_time_guard_error,
        "reserve_time_guard_value": reserve_guard_value,
        "reserve_time_error_score_nats": reserve_time_error_score,
        "reserve_time_error_budget_ok": reserve_time_error_budget_ok,
        "reserve_time_warranted": reserve_time_warranted,
        "reserve_time_min_sample": reserve_time_min,
        "reserve_time_max_sample": reserve_time_max,
        "usable": usable,
        "sample_retained_after_local_decline": (
            reserve_executed & reserve_finite & reserve_time_warranted),
        "decline_is_waveform_failure": jnp.asarray(False),
        "selected_nonfinite_is_integration_failure": (
            reserve_executed & (~reserve_finite)),
        "reserve_nphi": jnp.asarray(nphi_reserve),
        "reserve_nu": jnp.asarray(nu_reserve),
        "reserve_angle_points": jnp.asarray(nphi_reserve * nu_reserve),
        "reserve_distance_support_points": jnp.asarray(
            reserve_input_distance_points),
        "reserve_distance_points": jnp.asarray(
            reserve_gh_nodes if reserve_gh_nodes
            else reserve_input_distance_points),
        "reserve_uses_adaptive_distance": jnp.asarray(
            reserve_gh_nodes > 0),
        "reserve_distance_gh_nodes": jnp.asarray(reserve_gh_nodes),
        "reserve_time_points": jnp.asarray(reserve_time_points),
        "reserve_dense_chunk": jnp.asarray(int(reserve_dense_chunk)),
        "reserve_grid_block": jnp.asarray(int(reserve_grid_block)),
        "local_log_normalization": jnp.asarray(
            float(local_log_normalization)),
        "reserve_log_offset": jnp.asarray(float(reserve_log_offset)),
        "disposition_reconciles": (
            accepted_local.astype(jnp.int32)
            + reserve_executed.astype(jnp.int32)) == 1,
    })
    return selected_value, usable, ledger


def empirical_enrichment_with_exact_reserve_sequential_batch(
        C_A_t, C_B, base_plans, enriched_plans, x_min, x_max, *,
        reserve_x_grid, reserve_log_weights, time_weights,
        reserve_amp_sizing, reserve_m_max=None,
        reserve_dense_chunk=8, reserve_grid_block=32,
        reserve_time_nodes=None, reserve_time_resolution_warranted=False,
        reserve_time_check_value=np.nan,
        reserve_time_check_nodes=None, reserve_time_check_weights=None,
        reserve_time_resolution_tol_nats=1.0e-3,
        base_order=13, base_check_order=19,
        enriched_order=19, enriched_check_order=25,
        convergence_tol_nats=1.0e-3, time_guard=0,
        time_guard_tol_nats=1.0e-3, local_log_normalization=0.0,
        time_outside_tol_nats=-23.0,
        total_value_error_budget_nats=1.0e-3,
        reserve_log_offset=0.0, node_concentration=1.0,
        mode_match_tol=(0.25, 1.0e-4, 1.0e-4, 1.0e-5),
        reserve_angular_kernel=None, reserve_time_focus=None,
        reserve_time_cover=None):
    """Apply the scalar controller sequentially to a fixed-size batch.

    A direct ``vmap`` of :func:`empirical_enrichment_with_exact_reserve`
    rewrites its scalar conditional as a batched selection and can therefore
    execute the dense reserve for accepted rows.  This wrapper deliberately
    uses ``lax.map`` so each row reaches the scalar conditional independently;
    reserve workspace scales with one row rather than the batch size.  For a
    sparse, variable-size decline set, a host controller should instead vmap
    only :func:`empirical_enrichment_marginalize`, compact the declined rows,
    and invoke the reserve on that compact set.

    ``C_B`` and the time/distance rules are shared across the batch.  Every
    leaf of ``base_plans`` and ``enriched_plans`` must have a leading batch
    dimension.  ``reserve_time_resolution_warranted`` and
    ``reserve_time_check_value`` may each be one scalar policy value or one
    scalar per row.
    """
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    if C_A_t.ndim != 4:
        raise ValueError("C_A_t must have shape (batch,KP,KS,Ntime)")
    batch = C_A_t.shape[0]
    if C_B.ndim == 2:
        C_B = jnp.broadcast_to(C_B, (batch,) + C_B.shape)
    elif C_B.ndim != 3 or C_B.shape[0] != batch:
        raise ValueError("C_B must be shared or have a leading batch axis")
    for name, plans in (("base_plans", base_plans),
                        ("enriched_plans", enriched_plans)):
        for leaf in jax.tree.leaves(plans):
            if jnp.asarray(leaf).ndim < 1 or jnp.asarray(leaf).shape[0] != batch:
                raise ValueError("%s must have a leading batch axis" % name)
    warrant = jnp.asarray(reserve_time_resolution_warranted, dtype=bool)
    if warrant.ndim == 0:
        warrant = jnp.broadcast_to(warrant, (batch,))
    elif warrant.shape != (batch,):
        raise ValueError(
            "reserve_time_resolution_warranted must be scalar or per-row")
    check_value = jnp.asarray(reserve_time_check_value, dtype=jnp.float64)
    if check_value.ndim == 0:
        check_value = jnp.broadcast_to(check_value, (batch,))
    elif check_value.shape != (batch,):
        raise ValueError("reserve_time_check_value must be scalar or per-row")

    def _row(args):
        (table, norm, base_plan, enriched_plan, row_warrant,
         row_check_value) = args
        return empirical_enrichment_with_exact_reserve(
            table, norm, base_plan, enriched_plan, x_min, x_max,
            reserve_x_grid=reserve_x_grid,
            reserve_log_weights=reserve_log_weights,
            time_weights=time_weights,
            reserve_amp_sizing=reserve_amp_sizing,
            reserve_m_max=reserve_m_max,
            reserve_dense_chunk=reserve_dense_chunk,
            reserve_grid_block=reserve_grid_block,
            reserve_time_nodes=reserve_time_nodes,
            reserve_time_resolution_warranted=row_warrant,
            reserve_time_check_value=row_check_value,
            reserve_time_check_nodes=reserve_time_check_nodes,
            reserve_time_check_weights=reserve_time_check_weights,
            reserve_time_resolution_tol_nats=(
                reserve_time_resolution_tol_nats),
            base_order=base_order, base_check_order=base_check_order,
            enriched_order=enriched_order,
            enriched_check_order=enriched_check_order,
            convergence_tol_nats=convergence_tol_nats,
            time_guard=time_guard, time_guard_tol_nats=time_guard_tol_nats,
            local_log_normalization=local_log_normalization,
            time_outside_tol_nats=time_outside_tol_nats,
            total_value_error_budget_nats=total_value_error_budget_nats,
            reserve_log_offset=reserve_log_offset,
            node_concentration=node_concentration,
            mode_match_tol=mode_match_tol,
            reserve_angular_kernel=reserve_angular_kernel,
            reserve_time_focus=reserve_time_focus,
            reserve_time_cover=reserve_time_cover)

    selected, usable, ledger = jax.lax.map(
        _row, (C_A_t, C_B, base_plans, enriched_plans, warrant,
               check_value))
    ledger = dict(ledger)
    ledger["reserve_batch_execution_sequential"] = jnp.ones(
        (batch,), dtype=bool)
    return selected, usable, ledger
