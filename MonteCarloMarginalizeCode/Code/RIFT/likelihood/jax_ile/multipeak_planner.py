"""Diagnostic planner for joint peak-local JAX marginalization.

This module deliberately exposes an opt-in seam rather than changing the
production likelihood dispatch.  The fixed-shape device controller in
:mod:`all_axis_peaklocal` is the canonical four-axis path; the shared host
primitives (norm-table summary, harmonic lattice, distance profile, angular
field) are imported from there.  What remains here is the diagnostic variant:
symmetry-orbit start ranking, strict sequential Newton/polish refinement,
axis-aligned overlap partition, and the frozen hierarchical cover.  Its primary path tests whether a small,
mode-order-sized start portfolio plus empirical enrichment can replace global
time/angle/distance work while retaining a finite exact/dense reserve.

The planner keeps three statements separate:

* U,V/Q structure proposes a small set of four-dimensional optimizer starts;
* JAX hill climbing refines those starts, without claiming mode completeness;
* a richer structural tier repeats placement and the overlap-partitioned local
  integral; agreement and local diagnostics warrant the result, otherwise the
  caller-supplied finite reserve is returned.

The angular lattice resolves the finite coefficient polynomial, not
``exp(lnL)``.  It is targeting only.  The optional hierarchical cover below is
a frozen diagnostic: real 22 tables showed it remained thousands of nats too
loose after 5,000 boxes, so it is deliberately absent from the runtime
decision.  It bounds only the finite reflected coefficient model; a real ILE
caller still owns the guard-sample warrant connecting that model to physical
support.  Any local decline returns the reserve, never a waveform failure.

Returning the reserve because a diagnostic failed its budget and returning it
because the planner raised are different events, and this module reports them
separately: see ``MultiPeakResult.decline_kind`` and ``fault``.  Only the
second is logged, and only the second is made fatal by ``fail_on_fallback``.
"""

import heapq
import logging
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

# The host-side primitives shared with the device controller live in
# all_axis_peaklocal, which is canonical.  This module keeps only the
# diagnostic-seam variants that differ in semantics (symmetry-orbit start
# ranking, strict sequential Newton/polish refinement, axis-aligned overlap
# partition, and the frozen hierarchical cover).
from .all_axis_peaklocal import (  # noqa: E402
    UVHarmonicSummary as UVQSummary,
    _angular_field as _angular_field_jax,
    _distance_profile_numpy as _distance_profile,
    _harmonic_lattice,
    _kp_weights_numpy as _kp_weights,
    _periodic_distance,
    _validate_tables,
    summarize_uv_norm_table,
)


# A planner FAULT is reported here rather than through ``warnings.warn``.  A
# warning's delivery and its fatality are both governed by process-global
# filters that this module does not own: under ``-W error::RuntimeWarning`` the
# warn call itself raises, which would make the DEFAULT path
# (``fail_on_fallback=False``) fatal, and under the default filters the warnings
# registry de-duplicates per call site, so a campaign of identical faults from
# one call site reports once instead of once per call.  A logger has neither
# property.  The primary channel is still the returned record
# (``decline_kind``/``fault``), which no filter can suppress.
logger = logging.getLogger(__name__)


__all__ = [
    "UVQSummary",
    "HarmonicSymmetry",
    "StartPortfolio",
    "CoverReport",
    "LocalIntegralReport",
    "MultiPeakResult",
    "FallbackFault",
    "MultiPeakFallbackError",
    "DECLINE_DIAGNOSTIC",
    "DECLINE_FAULT",
    "summarize_uv_norm_table",
    "infer_harmonic_symmetry",
    "rank_joint_starts_from_uvq",
    "refine_joint_starts_jax",
    "select_refined_modes",
    "axis_local_geometry",
    "integrate_refined_modes_tensor",
    "multipeak_local_marginalize",
    "hierarchical_union_cover",
]


class HarmonicSymmetry(NamedTuple):
    """Finite angular symmetry group verified on the full coefficient tables."""

    shifts: np.ndarray
    group_order: int
    harmonic_lattice_index: int
    max_abs_residual: float
    relative_residual: float
    certified: bool


class StartPortfolio(NamedTuple):
    """Small distance-following start set and its explicit work counters."""

    starts: np.ndarray
    scores: np.ndarray
    raw_starts: np.ndarray
    raw_scores: np.ndarray
    group_action: np.ndarray
    symmetry: HarmonicSymmetry
    time_starts: np.ndarray
    time_profile: np.ndarray
    n_phi_lattice: int
    n_u_lattice: int
    n_lattice_evaluations: int
    n_raw_candidates: int
    capacity_truncated: bool


class CoverReport(NamedTuple):
    """Ledger for a hierarchical bound on the local-union complement.

    ``outside_log_upper`` is an absolute integral upper bound for the finite
    coefficient model when ``bound_certified`` is true.  ``budget_met`` is a
    separate comparison with the caller's diagnostic target value.
    """

    outside_log_upper: float
    tail_margin: float
    bound_certified: bool
    budget_met: bool
    cap_reached: bool
    n_boxes_evaluated: int
    n_subdivisions: int
    n_outside_leaves: int
    n_owned_leaves: int
    n_overlap_owned: int
    max_depth: np.ndarray
    initial_tail_margin: float
    best_tail_margin: float
    stalled: bool
    progress_checks: np.ndarray
    owned_centers: np.ndarray
    owned_half_widths: np.ndarray
    owned_mode: np.ndarray


class LocalIntegralReport(NamedTuple):
    """Empirical local-union integral and its bounded-work diagnostics."""

    value: float
    value_half: float
    quadrature_delta: float
    ok: bool
    finite: bool
    hessian_ok: bool
    edge_ok: bool
    local_geometry_ok: bool
    overlap_ok: bool
    contribution_ok: bool
    cell_tail_ok: bool
    n_input_modes: int
    n_retained_modes: int
    n_dropped_modes: int
    n_evaluations: int
    modeled_peak_bytes: int
    retained_indices: np.ndarray
    contribution_proxy: np.ndarray
    dropped_proxy_relative: float
    cell_tail_proxy_relative: float
    cell_axis_extents: np.ndarray
    edge_sigma: np.ndarray
    min_core_separation: float
    active_node_fraction: float


class _MultiPeakRecord(NamedTuple):
    """The 13-element tuple record.  Construct :class:`MultiPeakResult`."""

    value: float
    accepted: bool
    used_reserve: bool
    provenance: str
    delta_log_integral: float
    tier0: LocalIntegralReport
    tier1: LocalIntegralReport
    tier0_portfolio: StartPortfolio
    tier1_portfolio: StartPortfolio
    total_lattice_evaluations: int
    total_refinement_steps: int
    total_local_evaluations: int
    modeled_peak_bytes: int


class MultiPeakResult(_MultiPeakRecord):
    """Two-tier opt-in result with an explicit finite-reserve provenance.

    ``modeled_peak_bytes`` counts explicit planner/evaluator arrays.  It is a
    portable sizing model, not measured RSS or device high-water memory: JAX
    compilation caches, allocator retention, host/device duplication, and AD
    workspace must be measured separately on the production GPU.

    THE TUPLE IS 13 ELEMENTS AND STAYS 13.  ``decline_kind`` and ``fault``
    annotate the record; they are attributes only, and are not tuple
    elements.  A ``NamedTuple`` is a tuple, so appending two fields would have
    changed ``len()``, indexing, and iteration -- and any existing caller
    writing ``a, b, ..., m = result`` would get ``ValueError: too many values
    to unpack``.  Defaulted fields prevent that break on CONSTRUCTION, not on
    UNPACKING, which is the contract callers actually hold.  This is a core
    path; an existing caller must see exactly the previous behaviour.

    So ``len()``, iteration, indexing, ``_fields`` and ``_asdict()`` all cover
    the same 13 elements they did before.  The two annotations are reached by
    attribute, and are preserved across ``_replace``, ``copy`` and ``pickle``.
    """

    # decline_kind (Optional[str]) and fault (Optional[FallbackFault]) are set
    # per instance below.  There are no class-level fallbacks: __new__ is the
    # only door, since _make, _replace, copy and pickle all route through it,
    # so a fallback would be code no test could distinguish.
    def __new__(cls, *args, decline_kind=None, fault=None, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        # The tuple payload is immutable, and so are these: set them behind
        # the __setattr__ guard below.
        object.__setattr__(self, "decline_kind", decline_kind)
        object.__setattr__(self, "fault", fault)
        return self

    def __setattr__(self, name, value):
        raise AttributeError(
            "MultiPeakResult is immutable; use _replace(%s=...)" % name)

    def __delattr__(self, name):
        raise AttributeError("MultiPeakResult is immutable")

    @classmethod
    def _make(cls, iterable, *, decline_kind=None, fault=None):
        # namedtuple._make is tuple.__new__ bound as a classmethod: it skips
        # __new__, so it must be overridden or the annotations go missing.
        return cls(*iterable, decline_kind=decline_kind, fault=fault)

    # No __reduce__ is needed: this subclass has a __dict__, so the default
    # pickle/copy protocol carries the annotations as instance state on top of
    # the namedtuple's __getnewargs__ payload.  Removing an explicit __reduce__
    # changed no test, which is how it was found to be dead.  The round trip is
    # pinned by test_annotations_are_attributes_and_survive_replace_copy_pickle.
    def _replace(self, **kwargs):
        decline_kind = kwargs.pop("decline_kind", self.decline_kind)
        fault = kwargs.pop("fault", self.fault)
        values = dict(zip(_MultiPeakRecord._fields, self))
        unexpected = set(kwargs) - set(values)
        if unexpected:
            raise ValueError(
                "Got unexpected field names: %r" % sorted(unexpected))
        values.update(kwargs)
        return type(self)(
            *(values[name] for name in _MultiPeakRecord._fields),
            decline_kind=decline_kind, fault=fault)

    def __repr__(self):
        return "%s(%s, decline_kind=%r, fault=%r)" % (
            type(self).__name__,
            ", ".join("%s=%r" % (name, value)
                      for name, value in zip(_MultiPeakRecord._fields, self)),
            self.decline_kind, self.fault)


class _DenseReserveError(Exception):
    """A caller-supplied reserve failed; do not relabel it as planner decline."""


class MultiPeakFallbackError(Exception):
    """A planner fault was converted to a reserve under ``fail_on_fallback``.

    Deliberately outside the ``(RuntimeError, ValueError, LinAlgError)`` set the
    planner catches, for the same reason as :class:`_DenseReserveError`: a
    nested or repeated call must not swallow it back into a decline.
    """


class FallbackFault(NamedTuple):
    """Why a fallback was a fault rather than a diagnostic decline.

    ``stage`` names the planner step that raised, so a tier-specific defect is
    attributable without re-running.  ``error_type``/``message`` carry the
    exception itself.  ``None`` in ``MultiPeakResult.fault`` means no fault.
    """

    stage: str
    error_type: str
    message: str


#: ``MultiPeakResult.decline_kind`` values.  ``None`` means the local branch was
#: accepted.  A diagnostic decline is a normal outcome -- a diagnostic
#: legitimately failed its budget.  A fault means something raised, and the
#: reserve is standing in for a planner that could not run.  These are separate
#: values so downstream analysis never has to parse ``provenance``.
DECLINE_DIAGNOSTIC = "diagnostic"
DECLINE_FAULT = "fault"


def infer_harmonic_symmetry(C_A_t, C_B, *, support_rtol=1.0e-12,
                            invariance_rtol=1.0e-12):
    """Infer and verify the finite angular translation group of U,V and Q.

    Significant integer harmonics generate a rank-two lattice ``L``.  The
    finite symmetry group dual to ``Z^2/L`` has order equal to the gcd of the
    two-by-two minors.  Candidate translations are enumerated on that exact
    denominator and then verified against *all* coefficients, including those
    below the support threshold.  A noisy near-zero can therefore remove the
    ``certified`` label but can never silently invent a group action.
    """
    C_A_t = np.asarray(C_A_t, dtype=np.complex128)
    C_B = np.asarray(C_B, dtype=np.complex128)
    _validate_tables(C_A_t, C_B)
    scale = max(1.0, float(np.max(np.abs(C_A_t))), float(np.max(np.abs(C_B))))
    support = []
    tables = (C_A_t, C_B)
    for table in tables:
        ks_max = (table.shape[1] - 1) // 2
        for kp in range(table.shape[0]):
            for column, ks in enumerate(range(-ks_max, ks_max + 1)):
                if float(np.max(np.abs(table[kp, column]))) > support_rtol * scale:
                    support.append((int(kp), int(ks)))
    support = sorted(set(support))
    index = 0
    for i, first in enumerate(support):
        for second in support[:i]:
            determinant = abs(first[0] * second[1] - first[1] * second[0])
            index = math.gcd(index, int(determinant))
    if index == 0:
        return HarmonicSymmetry(
            np.zeros((1, 2)), 1, 0, np.inf, np.inf, False)

    actions = []
    for iphi in range(index):
        for iu in range(index):
            if all((kp * iphi + ks * iu) % index == 0
                   for kp, ks in support):
                actions.append((2.0 * np.pi * iphi / index,
                                2.0 * np.pi * iu / index))
    shifts = np.asarray(actions, dtype=float).reshape((-1, 2))
    maximum = 0.0
    for shift_phi, shift_u in shifts:
        for table in tables:
            kp = np.arange(table.shape[0], dtype=float)[:, None]
            ks = np.arange(-(table.shape[1] - 1) // 2,
                           (table.shape[1] - 1) // 2 + 1,
                           dtype=float)[None, :]
            factor = np.exp(1j * (kp * shift_phi + ks * shift_u)) - 1.0
            if table.ndim == 3:
                factor = factor[..., None]
            maximum = max(maximum, float(np.max(np.abs(table * factor))))
    relative = maximum / scale
    certified = bool(len(shifts) == index
                     and np.isfinite(relative)
                     and relative <= float(invariance_rtol))
    if not certified:
        shifts = np.zeros((1, 2))
    return HarmonicSymmetry(
        shifts, int(len(shifts)), int(index), maximum, relative, certified)


def rank_joint_starts_from_uvq(
        C_A_t, uv_summary, x_min, x_max, *, max_time_starts=4,
        max_starts=16, min_time_separation=2, keep_nats=None,
        angular_oversample=2):
    """Build sparse four-axis starts from the exact U,V/Q harmonics.

    U,V supplies the full angle-dependent norm, not only a scalar bound.  Q
    supplies the data harmonics at every retained time.  On a lattice sized by
    their exact harmonic orders, distance is optimized analytically at every
    point; the angular placement therefore follows distance rather than using a
    single frozen distance slice.  Only periodic angular maxima at a small
    number of ranked time basins become JAX starts.

    This is a mode-order-sized targeting lattice.  It is not an integration
    grid and carries no completeness semantics.
    """
    C_A_t = np.asarray(C_A_t, dtype=np.complex128)
    if not isinstance(uv_summary, UVQSummary):
        raise TypeError("uv_summary must come from summarize_uv_norm_table")
    _validate_tables(C_A_t, uv_summary.C_B)
    if not uv_summary.time_invariant:
        raise ValueError("arrival-time-dependent U,V norm cannot be collapsed")
    if not (0.0 < float(x_min) < float(x_max)):
        raise ValueError("need 0 < x_min < x_max")
    if min(int(max_time_starts), int(max_starts)) < 1:
        raise ValueError("start capacities must be positive")
    angular_oversample = int(angular_oversample)
    if angular_oversample < 1:
        raise ValueError("angular_oversample must be positive")
    symmetry = infer_harmonic_symmetry(C_A_t, uv_summary.C_B)

    k_phi = uv_summary.C_B.shape[0] - 1
    k_u = (uv_summary.C_B.shape[1] - 1) // 2
    n_phi = max(9, 2 * angular_oversample * k_phi + 1)
    n_u = max(9, 2 * angular_oversample * k_u + 1)
    phi, u, A = _harmonic_lattice(C_A_t, n_phi, n_u)
    _, _, B = _harmonic_lattice(uv_summary.C_B, n_phi, n_u)
    profile, x_best = _distance_profile(
        A, B[..., None], float(x_min), float(x_max))
    time_profile = np.max(profile, axis=(0, 1))

    time_peak = np.zeros(time_profile.size, dtype=bool)
    if time_profile.size == 1:
        time_peak[0] = True
    elif time_profile.size >= 2:
        # The reflected time support is not periodic.  Endpoints must pass the
        # available one-sided comparison before they can displace a real basin.
        time_peak[0] = time_profile[0] >= time_profile[1]
        time_peak[-1] = time_profile[-1] >= time_profile[-2]
    if time_profile.size > 2:
        time_peak[1:-1] = ((time_profile[1:-1] >= time_profile[:-2])
                           & (time_profile[1:-1] >= time_profile[2:]))
    candidate_time = np.flatnonzero(time_peak)
    candidate_time = candidate_time[
        np.argsort(time_profile[candidate_time])[::-1]]
    selected_time = []
    for time_index in candidate_time:
        if all(abs(int(time_index) - old) > int(min_time_separation)
               for old in selected_time):
            selected_time.append(int(time_index))
        if len(selected_time) == int(max_time_starts):
            break
    if not selected_time:
        selected_time = [int(np.argmax(time_profile))]

    raw = []
    for time_index in selected_time:
        surface = profile[..., time_index]
        is_peak = np.ones(surface.shape, dtype=bool)
        for dphi in (-1, 0, 1):
            for du in (-1, 0, 1):
                if dphi or du:
                    is_peak &= surface >= np.roll(
                        np.roll(surface, dphi, axis=0), du, axis=1)
        indices = np.argwhere(is_peak)
        if not len(indices):
            indices = np.asarray([
                np.unravel_index(np.argmax(surface), surface.shape)])
        for iphi, iu in indices:
            raw.append((
                float(surface[iphi, iu]),
                (float(time_index), float(phi[iphi]), float(u[iu]),
                 float(x_best[iphi, iu, time_index]))))
    raw.sort(key=lambda item: item[0], reverse=True)
    # Keep the odd, unshifted targeting grid: forcing its phase to align with
    # the group can move every seed outside a narrow high-SNR basin.  Instead,
    # reduce sampled cells modulo the *exact* group before capacity.  Copies
    # can differ by one grid cell because the exact translation generally lies
    # between lattice sites; two resolved cells are aliases only when the time
    # index agrees and some proven action brings both angular coordinates
    # within the corresponding lattice resolution.
    orbit_representatives = []
    phi_resolution = 2.0 * np.pi / n_phi
    u_resolution = 2.0 * np.pi / n_u
    for candidate in raw:
        start = np.asarray(candidate[1])
        duplicate = False
        for _, representative in orbit_representatives:
            representative = np.asarray(representative)
            if int(start[0]) != int(representative[0]):
                continue
            for shift in symmetry.shifts:
                shifted = representative[1:3] + shift
                delta = (start[1:3] - shifted + np.pi) % (
                    2.0 * np.pi) - np.pi
                if (abs(delta[0]) <= phi_resolution + 1.0e-13
                        and abs(delta[1]) <= u_resolution + 1.0e-13):
                    duplicate = True
                    break
            if duplicate:
                break
        if not duplicate:
            orbit_representatives.append(candidate)
    raw = orbit_representatives
    # The default does not prune on sampled height.  At high SNR a genuine
    # narrow basin can land tens of nats below its true peak on this deliberately
    # small lattice (measured -37 sampled versus -12 after refinement for the
    # second lmax=4 mode).  A fixed sampled-height cut would therefore become
    # *less* complete as amplitude grows even when extrema locations do not
    # change.  Capacity is the default work bound; an explicit keep_nats remains
    # available only as a diagnostic experiment.
    if keep_nats is None:
        kept_raw = raw
    else:
        best = raw[0][0]
        kept_raw = [item for item in raw
                    if item[0] >= best - float(keep_nats)]
    # Never truncate a proven orbit.  The total capacity limits the number of
    # representatives; every retained representative receives every verified
    # group action, with the action index recorded for the placement audit.
    n_representative = max(1, int(max_starts) // symmetry.group_order)
    n_raw_candidates = len(kept_raw)
    capacity_truncated = n_raw_candidates > n_representative
    kept_raw = kept_raw[:n_representative]
    expanded = []
    action = []
    for score, start in kept_raw:
        for action_index, shift in enumerate(symmetry.shifts):
            copied = np.asarray(start, dtype=float).copy()
            copied[1:3] = np.mod(copied[1:3] + shift, 2.0 * np.pi)
            expanded.append((score, copied))
            action.append(action_index)
    return StartPortfolio(
        np.asarray([item[1] for item in expanded], dtype=float).reshape((-1, 4)),
        np.asarray([item[0] for item in expanded], dtype=float),
        np.asarray([item[1] for item in kept_raw], dtype=float).reshape((-1, 4)),
        np.asarray([item[0] for item in kept_raw], dtype=float),
        np.asarray(action, dtype=np.int32), symmetry,
        np.asarray(selected_time, dtype=np.int32), time_profile,
        int(n_phi), int(n_u), int(n_phi * n_u * C_A_t.shape[-1]),
        int(n_raw_candidates), bool(capacity_truncated))


def _reflected_spectrum(C_A_t):
    reflected = jnp.concatenate(
        (C_A_t, jnp.flip(C_A_t[..., 1:-1], axis=-1)), axis=-1)
    return (jnp.fft.fft(reflected, axis=-1) / reflected.shape[-1],
            jnp.fft.fftfreq(reflected.shape[-1]))


def _evaluate_spectrum(coeff, frequency, time):
    phase = jnp.exp(2j * jnp.pi * time * frequency)
    if coeff.shape[-1] % 2 == 0:
        phase = phase.at[coeff.shape[-1] // 2].set(jnp.cos(jnp.pi * time))
    return jnp.einsum("kqn,n->kq", coeff, phase)


def _bounded_ascent_direction(gradient, hessian, ridge, max_step=None):
    """Modified-Newton direction, bounded WITHOUT losing the ascent property.

    ``eigenvector @ ((eigenvector.T @ g) / safe)`` always ascends:
    ``g . d = sum_i (v_i . g)^2 / safe_i > 0`` because every ``safe_i`` is
    positive.  Bounding it by clipping each COORDINATE independently does not
    preserve that, because the clip rescales the coordinates by different
    factors.  Measured on the row that declined the 2026-09-07 ladder
    campaign, where ``eigh(-H)`` is indefinite and the raw step is therefore
    about 1e8 gradients long, so every coordinate saturates:

        g       = [-54.47, +51.02, +28.92,  0]
        clipped = [ +2.00,  -0.50,  +0.50, +0.25]     g . d = -119.9

    A descent direction has no improving lane, the value-only search takes its
    zero lane, and the iterate is a fixed point of the whole loop.  Scaling by
    one factor instead keeps every ratio, hence the sign of ``g . d``, while
    respecting the same per-coordinate cap.

    Written as ``min(max_step / |d|)`` to match
    ``all_axis_peaklocal.refine_all_axis_starts``, which bounds its own step
    this way for the same reason.  Measured, so that the next reader does not
    have to re-derive it: the two spellings are equivalent here, and BOTH
    return a zero step once ``|d|`` reaches about 1e308, because the scale
    factor is then denormal and the product underflows.  That needs
    ``|g| >~ 1e299`` at the default ridge, which this likelihood cannot reach.
    ``tiny`` is defensive, not load-bearing: ``max_step`` is validated positive
    below, so ``max_step / 0`` is ``+inf`` rather than a NaN, and the step for
    a zero direction is zero either way.

    Returns the eigenvalues of ``-H`` alongside the direction so a caller that
    needs both does not decompose the same 4x4 twice.
    """
    eigenvalue, eigenvector = jnp.linalg.eigh(-hessian)
    safe = jnp.maximum(eigenvalue, float(ridge))
    direction = eigenvector @ ((eigenvector.T @ gradient) / safe)
    if max_step is None:
        return direction, eigenvalue
    ratio = max_step / jnp.maximum(jnp.abs(direction),
                                   jnp.finfo(jnp.float64).tiny)
    return direction * jnp.minimum(1.0, jnp.min(ratio)), eigenvalue


def refine_joint_starts_jax(
        C_A_t, C_B, starts, x_min, x_max, *, iterations=12,
        ridge=1.0e-8, max_step=(2.0, 0.5, 0.5, 0.25)):
    """Refine the small portfolio with bounded sequential JAX Newton steps.

    The reflected time primitive is evaluated only at each proposed time.  A
    ``lax.map`` over starts avoids a start-by-frequency-by-Hessian batch.  This
    is local hill climbing only; convergence never asserts completeness.
    """
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    starts = jnp.asarray(starts, dtype=jnp.float64)
    _validate_tables(C_A_t, C_B)
    if starts.ndim != 2 or starts.shape[1] != 4:
        raise ValueError("starts must have shape (N,4)")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")
    coeff, frequency = _reflected_spectrum(C_A_t)
    max_step = jnp.asarray(max_step, dtype=jnp.float64)
    # The rescale below is meaningless for a non-positive bound, and fails
    # QUIETLY rather than loudly: a zero bound scales every step to exactly
    # zero, which is the stall this function exists to prevent, and a negative
    # bound is silently exceeded (bound -0.5 returns a component of -3.5).
    # Neither is non-finite, so nothing downstream would notice.
    if max_step.shape != (4,) or not bool(jnp.all(max_step > 0.0)):
        raise ValueError("max_step must be four positive coordinate bounds")

    def log_density(theta):
        time, phi, u, x = theta
        C_A = _evaluate_spectrum(coeff, frequency, time)
        A = _angular_field_jax(C_A, phi, u)
        B = _angular_field_jax(C_B, phi, u)
        inside = ((time >= 0.0) & (time <= C_A_t.shape[-1] - 1.0)
                  & (x >= float(x_min)) & (x <= float(x_max)) & (x > 0.0))
        value = x * A - 0.5 * x * x * B - 4.0 * jnp.log(
            jnp.maximum(x, 1.0e-300))
        return jnp.where(inside, value, -jnp.inf)

    gradient_fn = jax.grad(log_density)
    hessian_fn = jax.hessian(log_density)

    def project(theta):
        return jnp.asarray([
            jnp.clip(theta[0], 0.0, C_A_t.shape[-1] - 1.0),
            jnp.mod(theta[1], 2.0 * jnp.pi),
            jnp.mod(theta[2], 2.0 * jnp.pi),
            jnp.clip(theta[3], float(x_min), float(x_max)),
        ])

    def one(start):
        def step(theta, _):
            gradient = gradient_fn(theta)
            hessian = hessian_fn(theta)
            direction, _ = _bounded_ascent_direction(
                gradient, hessian, ridge, max_step)
            proposal = jax.vmap(
                lambda scale: project(theta + scale * direction))(
                    jnp.asarray([1.0, 0.5, 0.25, 0.125, 0.0]))
            values = jax.vmap(log_density)(proposal)
            return proposal[jnp.argmax(values)], None

        point, _ = jax.lax.scan(step, project(start), None,
                                length=int(iterations))

        # At loud SNR the Newton improvement can be below one ulp of lnL while
        # the remaining absolute gradient is still visible.  A value-only line
        # search then selects its zero-step lane forever.  Two final root-polish
        # steps accept only a strict gradient-norm reduction, positive local
        # curvature, and no value loss beyond roundoff.  This tightens the
        # stationarity result; it does not relax the downstream gate.
        def polish(theta, _):
            gradient = gradient_fn(theta)
            hessian = hessian_fn(theta)
            # Unbounded, as before: the polish's own guard below accepts a
            # step only at a strict maximum with a smaller gradient.  One
            # decomposition serves both the step and that guard.
            direction, eigenvalue = _bounded_ascent_direction(
                gradient, hessian, ridge)
            proposal = project(theta + direction)
            proposal_gradient = gradient_fn(proposal)
            value = log_density(theta)
            proposal_value = log_density(proposal)
            tolerance = 32.0 * jnp.finfo(jnp.float64).eps * jnp.maximum(
                1.0, jnp.abs(value))
            use = (jnp.all(eigenvalue > 0.0)
                   & (jnp.linalg.norm(proposal_gradient)
                      < jnp.linalg.norm(gradient))
                   & (proposal_value >= value - tolerance))
            return jnp.where(use, proposal, theta), None

        point, _ = jax.lax.scan(polish, point, None, length=2)
        value = log_density(point)
        gradient = gradient_fn(point)
        hessian = hessian_fn(point)
        curvature = jnp.linalg.eigvalsh(-hessian)
        return point, value, gradient, hessian, curvature

    return jax.lax.map(jax.checkpoint(one), starts)


def select_refined_modes(points, values, gradients, curvatures, *,
                         max_modes, gradient_tol=2.0e-6,
                         coordinate_tol=(0.25, 1.0e-4, 1.0e-4, 1.0e-5)):
    """Filter, rank, and periodically deduplicate refined local maxima."""
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    gradients = np.asarray(gradients, dtype=float)
    curvatures = np.asarray(curvatures, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 4
            or gradients.shape != points.shape
            or curvatures.shape != points.shape
            or values.shape != (points.shape[0],)):
        raise ValueError("inconsistent refined-mode arrays")
    stationary = (np.all(np.isfinite(points), axis=1)
                  & np.isfinite(values)
                  & np.all(np.isfinite(gradients), axis=1)
                  & (np.linalg.norm(gradients, axis=1) <= float(gradient_tol))
                  & np.all(curvatures > 0.0, axis=1))
    order = np.flatnonzero(stationary)
    order = order[np.argsort(values[order])[::-1]]
    tolerance = np.asarray(coordinate_tol, dtype=float)
    selected = []
    for index in order:
        duplicate = False
        for old in selected:
            delta = np.abs(points[index] - points[old])
            delta[1] = _periodic_distance(points[index, 1], points[old, 1])
            delta[2] = _periodic_distance(points[index, 2], points[old, 2])
            if np.all(delta <= tolerance):
                duplicate = True
                break
        if not duplicate:
            selected.append(int(index))
    if len(selected) > int(max_modes):
        raise ValueError("unique stationary mode count exceeds capacity")
    return np.asarray(selected, dtype=np.int32), stationary


def axis_local_geometry(hessians, *, w_sigma=6.0,
                        eigenvalue_floor=1.0e-12):
    """Axis-aligned boxes enclosing ``w_sigma`` marginal Hessian widths."""
    hessians = np.asarray(hessians, dtype=float)
    if hessians.ndim != 3 or hessians.shape[1:] != (4, 4):
        raise ValueError("hessians must have shape (N,4,4)")
    widths = np.full((len(hessians), 4), np.nan)
    for index, hessian in enumerate(hessians):
        eigenvalue = np.linalg.eigvalsh(-hessian)
        if (np.all(np.isfinite(eigenvalue))
                and np.min(eigenvalue) > float(eigenvalue_floor)):
            covariance = np.linalg.inv(-hessian)
            widths[index] = float(w_sigma) * np.sqrt(
                np.maximum(np.diag(covariance), 0.0))
    return widths


def _evaluate_points_jax(C_A_t, C_B, points, x_min, x_max, chunk_size):
    """Stream the four-axis exponent without materializing point x frequency."""
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    points = jnp.asarray(points, dtype=jnp.float64)
    coeff, frequency = _reflected_spectrum(C_A_t)
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    n_point = points.shape[0]
    n_chunk = (n_point + chunk_size - 1) // chunk_size
    padding = n_chunk * chunk_size - n_point
    padded = jnp.pad(points, ((0, padding), (0, 0)))

    def one(theta):
        time, phi, u, x = theta
        C_A = _evaluate_spectrum(coeff, frequency, time)
        A = _angular_field_jax(C_A, phi, u)
        B = _angular_field_jax(C_B, phi, u)
        inside = ((time >= 0.0) & (time <= C_A_t.shape[-1] - 1.0)
                  & (x >= float(x_min)) & (x <= float(x_max)) & (x > 0.0))
        value = x * A - 0.5 * x * x * B - 4.0 * jnp.log(
            jnp.maximum(x, 1.0e-300))
        return jnp.where(inside, value, -jnp.inf)

    def step(_, block):
        return None, jax.vmap(one)(block)

    _, values = jax.lax.scan(
        jax.checkpoint(step), None,
        padded.reshape((n_chunk, chunk_size, 4)))
    return values.reshape((-1,))[:n_point]


def _periodic_delta_rows(points, center):
    delta = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    delta[..., 1] = (delta[..., 1] + np.pi) % (2.0 * np.pi) - np.pi
    delta[..., 2] = (delta[..., 2] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def _laplace_contribution_proxy(values, hessians):
    result = np.full(len(values), -np.inf)
    for index, (value, hessian) in enumerate(zip(values, hessians)):
        sign, logdet = np.linalg.slogdet(-hessian)
        if sign > 0 and np.isfinite(logdet):
            result[index] = (float(value) + 2.0 * np.log(2.0 * np.pi)
                             - 0.5 * logdet)
    return result


def integrate_refined_modes_tensor(
        C_A_t, C_B, points, values, hessians, x_min, x_max, *,
        log_integral_tol=1.0e-3, contribution_cutoff_nats=-18.0,
        cell_sigma=5.0, quadrature_order=7, chunk_size=64,
        max_condition=1.0e10, edge_guard_sigma=0.5,
        core_overlap_sigma=4.0, log_measure=0.0):
    """Integrate the union of finite full-Hessian local cells.

    Every retained maximum supplies the finite affine cell
    ``theta = mu + L z``, ``|z_i| <= cell_sigma``, with
    ``L L.T = (-H)^-1``.  Tensor Gauss--Hermite rules sample the corresponding
    Gaussian mixture, while an exact cell-union indicator zeros nodes outside
    those finite regions.  Dividing by the *full mixture density* partitions
    overlaps automatically, rather than integrating shared tails once per
    mode.  Orders ``n`` and ``n-2`` provide the empirical convergence check
    and are exact for the quadratic local limit.  The tensor is streamed in
    bounded chunks.  ``core_overlap_sigma`` is retained as a separation
    telemetry scale; overlap itself is not a rejection because the full
    mixture density owns it exactly.

    This is an empirical warrant, not a deterministic omitted-mass proof.  Its
    ``ok`` flag is therefore allowed to choose a finite dense reserve, never to
    delete the outer likelihood point.  Coordinates here are time-sample index,
    two radians, and inverse distance.  ``log_measure`` carries any constant
    physical time step and normalized-prior factors required by the caller;
    the dense reserve must use the same convention.
    """
    C_A_t = np.asarray(C_A_t, dtype=np.complex128)
    C_B = np.asarray(C_B, dtype=np.complex128)
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    hessians = np.asarray(hessians, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 4
            or values.shape != (len(points),)
            or hessians.shape != (len(points), 4, 4)
            or len(points) == 0):
        raise ValueError("points, values, and hessians must describe N>0 modes")
    if int(quadrature_order) < 4:
        raise ValueError("quadrature_order must be at least 4")
    if not (np.isfinite(float(log_integral_tol))
            and 0.0 < float(log_integral_tol)):
        raise ValueError("log_integral_tol must be positive")
    if not (np.isfinite(float(contribution_cutoff_nats))
            and float(contribution_cutoff_nats) <= 0.0):
        raise ValueError("contribution_cutoff_nats must be finite and nonpositive")
    if not (np.isfinite(float(cell_sigma)) and 0.0 < float(cell_sigma)):
        raise ValueError("cell_sigma must be positive")
    if not np.isfinite(float(log_measure)):
        raise ValueError("log_measure must be finite")

    proxy = _laplace_contribution_proxy(values, hessians)
    best_proxy = float(np.max(proxy))
    retained = np.flatnonzero(
        proxy >= best_proxy + float(contribution_cutoff_nats))
    dropped = np.setdiff1d(np.arange(len(points)), retained)
    dropped_proxy = (-np.inf if not len(dropped) else
                     float(scipy_logsumexp(proxy[dropped]) - best_proxy))
    contribution_ok = bool(
        dropped_proxy < math.log(float(log_integral_tol)) - 2.0)
    normal_mass = math.erf(float(cell_sigma) / math.sqrt(2.0))
    cell_tail_proxy = math.log(max(
        1.0 - normal_mass ** 4, np.finfo(float).tiny))
    cell_tail_ok = bool(
        cell_tail_proxy < math.log(float(log_integral_tol)) - 2.0)

    modes = points[retained]
    fishers = -hessians[retained]
    cholesky = []
    hessian_ok = bool(len(retained))
    for fisher in fishers:
        try:
            eigenvalue = np.linalg.eigvalsh(fisher)
            this_condition = float(np.max(eigenvalue) / np.min(eigenvalue))
            covariance = np.linalg.inv(fisher)
            factor = np.linalg.cholesky(covariance)
            good = (np.all(np.isfinite(eigenvalue)) and np.min(eigenvalue) > 0.0
                    and np.isfinite(this_condition)
                    and this_condition <= float(max_condition))
        except np.linalg.LinAlgError:
            factor = np.full((4, 4), np.nan)
            this_condition = np.inf
            good = False
        cholesky.append(factor)
        hessian_ok &= good
    cholesky = np.asarray(cholesky)

    # Exact row-wise extent of the affine parallelepiped.  Angular extents must
    # stay below half a period so nearest-copy mixture densities are unambiguous.
    extents = float(cell_sigma) * np.sum(np.abs(cholesky), axis=2)
    marginal_sigma = np.sqrt(np.maximum(
        np.diagonal(cholesky @ np.swapaxes(cholesky, 1, 2), axis1=1, axis2=2),
        0.0))
    edge_sigma = np.minimum(
        (modes[:, 0] - 0.0) / np.maximum(marginal_sigma[:, 0], 1.0e-300),
        (C_A_t.shape[-1] - 1.0 - modes[:, 0])
        / np.maximum(marginal_sigma[:, 0], 1.0e-300))
    edge_sigma = np.minimum(
        edge_sigma,
        np.minimum(
            (modes[:, 3] - float(x_min))
            / np.maximum(marginal_sigma[:, 3], 1.0e-300),
            (float(x_max) - modes[:, 3])
            / np.maximum(marginal_sigma[:, 3], 1.0e-300)))
    edge_ok = bool(np.all(edge_sigma >= float(edge_guard_sigma)))
    local_ok = bool(np.all(extents[:, 1:3] < np.pi)
                    and np.all(extents[:, 0] < 0.5 * (C_A_t.shape[-1] - 1.0))
                    and np.all(extents[:, 3] < 0.5 * (x_max - x_min)))

    separation = []
    for first in range(len(modes)):
        for second in range(first):
            delta = _periodic_delta_rows(modes[first:first + 1], modes[second])[0]
            # Symmetric local metric: use the smaller of the two Fisher lengths.
            separation.append(min(
                math.sqrt(max(0.0, float(delta @ fishers[first] @ delta))),
                math.sqrt(max(0.0, float(delta @ fishers[second] @ delta)))))
    min_separation = min(separation) if separation else np.inf
    # The mixture denominator partitions ordinary affine-cell overlaps.  The
    # only ambiguous case is a cell wide enough to meet more than one periodic
    # image of itself; the strict angular-locality gate rejects that geometry.
    overlap_ok = bool(np.all(extents[:, 1:3] < np.pi))
    n_high_model = len(modes) * int(quadrature_order) ** 4
    n_rule = int(quadrature_order) ** 4
    # Conservative count of the explicit simultaneous arrays in the Python
    # quadrature and streamed JAX evaluator: samples and their construction
    # copies, rule weights, mixture matrix, union mask, periodic deltas/local-z,
    # density/proposal/integrand vectors, and a device points/value payload.
    # Backend compilation, allocator retention, AD, and host/device duplication
    # outside these arrays are deliberately not represented (see result doc).
    modeled_peak_bytes = int(
        C_A_t.nbytes + C_B.nbytes
        + n_high_model * (209 + len(modes) * 8)
        + n_rule * (4 * 8 + 8)
        + int(chunk_size) * 2 * (C_A_t.shape[-1] - 1) * 16)

    finite_structure = bool(hessian_ok and np.all(np.isfinite(cholesky))
                            and np.all(np.isfinite(proxy[retained])))
    if not finite_structure:
        return LocalIntegralReport(
            np.nan, np.nan, np.inf, False, False, hessian_ok, edge_ok,
            local_ok, overlap_ok, contribution_ok, cell_tail_ok,
            len(points), len(retained), len(dropped), 0,
            modeled_peak_bytes,
            retained.astype(np.int32), proxy, dropped_proxy, cell_tail_proxy,
            extents, edge_sigma,
            float(min_separation), 0.0)

    log_normalization = 2.0 * np.log(2.0 * np.pi)

    def integrate_order(order):
        node, weight = np.polynomial.hermite.hermgauss(int(order))
        z_axis = np.sqrt(2.0) * node
        log_weight_axis = np.log(weight) - 0.5 * np.log(np.pi)
        mesh = np.meshgrid(z_axis, z_axis, z_axis, z_axis, indexing="ij")
        z = np.stack(mesh, axis=-1).reshape((-1, 4))
        weight_mesh = np.meshgrid(
            log_weight_axis, log_weight_axis, log_weight_axis,
            log_weight_axis, indexing="ij")
        log_rule_weight = np.sum(
            np.stack(weight_mesh, axis=-1), axis=-1).reshape(-1)
        samples = []
        rule_weights = []
        for mode, factor in zip(modes, cholesky):
            theta = mode[None, :] + z @ factor.T
            theta[:, 1:3] = np.mod(theta[:, 1:3], 2.0 * np.pi)
            samples.append(theta)
            rule_weights.append(log_rule_weight)
        samples = np.concatenate(samples, axis=0)
        rule_weights = np.concatenate(rule_weights) - np.log(len(modes))

        log_component = np.full((len(samples), len(modes)), -np.inf)
        inside_union = np.zeros(len(samples), dtype=bool)
        for mode_index, (mode, factor) in enumerate(zip(modes, cholesky)):
            delta = _periodic_delta_rows(samples, mode)
            local_z = np.linalg.solve(factor, delta.T).T
            inside_union |= np.all(
                np.abs(local_z) <= float(cell_sigma) + 1.0e-12, axis=1)
            logdet = float(np.sum(np.log(np.diag(factor))))
            log_component[:, mode_index] = (
                -0.5 * np.sum(np.square(local_z), axis=1)
                - log_normalization - logdet)
        log_proposal = (scipy_logsumexp(log_component, axis=1)
                        - np.log(len(modes)))
        log_density = np.asarray(_evaluate_points_jax(
            C_A_t, C_B, samples, x_min, x_max, chunk_size), dtype=float)
        log_density = np.where(inside_union, log_density, -np.inf)
        log_integrand = log_density - log_proposal + rule_weights
        return (float(scipy_logsumexp(log_integrand) + float(log_measure)),
                log_density,
                log_proposal, len(samples))

    value, log_density, log_proposal, n_high = integrate_order(
        int(quadrature_order))
    value_half, _, _, n_low = integrate_order(int(quadrature_order) - 2)
    quadrature_delta = abs(value - value_half)
    active_node_fraction = float(np.mean(np.isfinite(log_density)))
    finite = bool(np.isfinite(value) and np.isfinite(value_half)
                  and np.all(np.isfinite(log_proposal)))
    ok = bool(finite and hessian_ok and edge_ok and local_ok and overlap_ok
              and contribution_ok and cell_tail_ok
              and quadrature_delta <= float(log_integral_tol))
    return LocalIntegralReport(
        value, value_half, quadrature_delta, ok, finite, hessian_ok,
        edge_ok, local_ok, overlap_ok, contribution_ok, cell_tail_ok,
        len(points), len(retained),
        len(dropped), n_high + n_low, modeled_peak_bytes,
        retained.astype(np.int32), proxy,
        dropped_proxy, cell_tail_proxy, extents, edge_sigma,
        float(min_separation), active_node_fraction)


def _run_structural_tier(C_A_t, uv_summary, x_min, x_max, *,
                         angular_oversample, max_time_starts, max_starts,
                         refine_iterations, integral_kwargs):
    portfolio = rank_joint_starts_from_uvq(
        C_A_t, uv_summary, x_min, x_max,
        angular_oversample=angular_oversample,
        max_time_starts=max_time_starts, max_starts=max_starts)
    result = tuple(np.asarray(item) for item in refine_joint_starts_jax(
        C_A_t, uv_summary.C_B, portfolio.starts, x_min, x_max,
        iterations=refine_iterations))
    points, values, gradients, hessians, curvatures = result
    selected, _ = select_refined_modes(
        points, values, gradients, curvatures, max_modes=max_starts)
    if not len(selected):
        raise RuntimeError(
            "structural tier found no strict stationary maximum")
    integral = integrate_refined_modes_tensor(
        C_A_t, uv_summary.C_B, points[selected], values[selected],
        hessians[selected], x_min, x_max, **integral_kwargs)
    return portfolio, integral


def multipeak_local_marginalize(
        C_A_t, C_B_t, x_min, x_max, dense_reserve, *,
        log_integral_tol=1.0e-3, tier0=(2, 3, 24), tier1=(3, 5, 48),
        refine_iterations=18, contribution_cutoff_nats=-18.0,
        cell_sigma=5.0, quadrature_order=7, chunk_size=64,
        log_measure=0.0, fail_on_fallback=False, label=None):
    """Two-tier empirical four-axis marginal with a finite reserve fallback.

    The tuple for each tier is ``(angular_oversample, time_starts, start_cap)``.
    Acceptance requires both local-mixture quadratures to pass their internal
    diagnostics and agree within ``log_integral_tol``.  Otherwise the supplied
    dense/exact reserve is evaluated and returned with explicit provenance.
    ``dense_reserve`` should normally be a zero-argument callable, so an
    accepted local row never pays for the fallback.  A finite scalar is also
    accepted when a caller already has the reserve value.  ``log_measure`` is
    the caller-owned constant measure/normalization for time, angles, and the
    inverse-distance prior; both paths must use the same convention.

    Two different things return the reserve and they are reported separately.
    A *diagnostic* decline (``decline_kind == DECLINE_DIAGNOSTIC``) is a normal
    outcome: the tiers ran and one of their budgets was not met.  A *fault*
    (``decline_kind == DECLINE_FAULT``, ``fault`` populated) means the planner
    raised, so the reserve is standing in for a step that did not run.  The
    fault is reported in two places: in the returned record, which is the
    primary channel because no configuration can suppress it, and on the module
    logger (``RIFT.likelihood.jax_ile.multipeak_planner``) at WARNING, once per
    CALL -- not once per call site, which is what ``warnings.warn`` would give.
    ``fail_on_fallback=True`` raises
    :class:`MultiPeakFallbackError` instead of evaluating the reserve.  It
    defaults off: this is a core path and an existing caller must see exactly
    the previous behaviour, under any logging or warnings configuration.

    ``label`` is an opaque caller tag (a row id, an event name) echoed in the
    log line so a fault in a large campaign is attributable without re-running.
    """
    def evaluate_reserve():
        try:
            value = dense_reserve() if callable(dense_reserve) else dense_reserve
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("dense_reserve must produce a finite value")
            return value
        except Exception as error:
            # This wrapper is intentionally outside the planner exception
            # hierarchy.  A failing reserve is invoked once and propagated,
            # never retried or mislabeled as a local-planner exception.
            raise _DenseReserveError("dense reserve evaluation failed") from error

    # Names the planner step in flight, so a fault reports WHERE it happened
    # rather than only that something raised.  Read only in the handler below.
    stage = "uv-summary"
    try:
        uv_summary = summarize_uv_norm_table(C_B_t)
        if not uv_summary.time_invariant:
            raise ValueError(
                "the four-axis prototype requires time-independent U,V")
        integral_kwargs = dict(
            log_integral_tol=log_integral_tol,
            contribution_cutoff_nats=contribution_cutoff_nats,
            cell_sigma=cell_sigma, quadrature_order=quadrature_order,
            chunk_size=chunk_size, log_measure=log_measure)
        stage = "tier0"
        portfolio0, result0 = _run_structural_tier(
            C_A_t, uv_summary, x_min, x_max,
            angular_oversample=int(tier0[0]),
            max_time_starts=int(tier0[1]), max_starts=int(tier0[2]),
            refine_iterations=int(refine_iterations),
            integral_kwargs=integral_kwargs)
        stage = "tier1"
        portfolio1, result1 = _run_structural_tier(
            C_A_t, uv_summary, x_min, x_max,
            angular_oversample=int(tier1[0]),
            max_time_starts=int(tier1[1]), max_starts=int(tier1[2]),
            refine_iterations=int(refine_iterations),
            integral_kwargs=integral_kwargs)
        stage = "acceptance"
        delta = abs(result1.value - result0.value)
        accepted = bool(result0.ok and result1.ok and np.isfinite(delta)
                        and delta <= float(log_integral_tol))
        if accepted:
            value = result1.value
            provenance = "uvq-multipeak-tier1"
            decline_kind = None
        else:
            # Both tiers ran and reported.  This is a budget outcome, not a
            # fault: it is silent by design and must stay silent.
            value = evaluate_reserve()
            provenance = "dense-reserve:enrichment-or-local-diagnostic"
            decline_kind = DECLINE_DIAGNOSTIC
        input_bytes = (np.asarray(C_A_t).nbytes
                       + np.asarray(uv_summary.C_B).nbytes)

        def portfolio_bytes(portfolio):
            return int(input_bytes
                       + portfolio.n_phi_lattice * portfolio.n_u_lattice
                       * (3 * np.asarray(C_A_t).shape[-1] + 1) * 8)
        return MultiPeakResult(
            float(value), accepted, not accepted, provenance, float(delta),
            result0, result1, portfolio0, portfolio1,
            int(portfolio0.n_lattice_evaluations
                + portfolio1.n_lattice_evaluations),
            (int(refine_iterations) + 2) * (len(portfolio0.starts)
                                            + len(portfolio1.starts)),
            int(result0.n_evaluations + result1.n_evaluations),
            max(result0.modeled_peak_bytes, result1.modeled_peak_bytes,
                portfolio_bytes(portfolio0), portfolio_bytes(portfolio1)),
            decline_kind=decline_kind, fault=None)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        # Keep the return finite even when the local planner itself cannot form
        # a trustworthy report.  Re-run failures should be diagnosed upstream;
        # they must never be reclassified as waveform failures.
        #
        # This is a FAULT, not a decline: the reserve stands in for a step that
        # did not run.  A whole campaign once read as a conservative controller
        # because this path was silent, so it is reported once per call and can
        # be made fatal.  The report precedes the reserve so a fault is still
        # visible when the reserve itself then fails.
        #
        # A LOGGER, not warnings.warn.  Two properties of the warnings module
        # are wrong for this: under ``-W error::RuntimeWarning`` the warn call
        # RAISES, which would make this default (fail_on_fallback=False) path
        # fatal on a filter the caller may have set for unrelated reasons; and
        # under the default filters the registry de-duplicates per (message,
        # category, call site), so N identical faults from one call site report
        # ONCE.  That is worst in the campaign case this change exists for,
        # where label=None leaves every message identical.  Neither the
        # record below nor this logger has either property.
        fault = FallbackFault(
            stage, type(error).__name__, str(error))
        logger.warning(
            "multipeak_local_marginalize: the local planner RAISED at stage "
            "%r and fell back to the dense reserve.  This is a fault, not a "
            "budget decline: %s: %s.  label=%r, C_A_t.shape=%s, "
            "C_B_t.shape=%s, x=[%r, %r], tier0=%r, tier1=%r, "
            "refine_iterations=%r.  Diagnose it upstream; pass "
            "fail_on_fallback=True to make it fatal.",
            fault.stage, fault.error_type, fault.message, label,
            np.shape(C_A_t), np.shape(C_B_t), x_min, x_max,
            tuple(tier0), tuple(tier1), refine_iterations,
            stacklevel=2)
        if fail_on_fallback:
            raise MultiPeakFallbackError(
                "multipeak_local_marginalize declined by fault at stage %r "
                "(%s: %s), label=%r; fail_on_fallback is set"
                % (fault.stage, fault.error_type, fault.message, label)
            ) from error
        empty = LocalIntegralReport(
            np.nan, np.nan, np.inf, False, False, False, False, False, False,
            False, False, 0, 0, 0, 0, 0, np.empty(0, dtype=np.int32),
            np.empty(0), -np.inf, np.inf, np.empty((0, 4)), np.empty(0),
            np.nan, 0.0)
        empty_symmetry = HarmonicSymmetry(
            np.zeros((1, 2)), 1, 0, np.inf, np.inf, False)
        empty_portfolio = StartPortfolio(
            np.empty((0, 4)), np.empty(0), np.empty((0, 4)), np.empty(0),
            np.empty(0, dtype=np.int32), empty_symmetry,
            np.empty(0, dtype=np.int32), np.empty(0), 0, 0, 0, 0, False)
        return MultiPeakResult(
            evaluate_reserve(), False, True,
            "dense-reserve:planner-exception:%s" % type(error).__name__,
            np.inf,
            empty, empty, empty_portfolio, empty_portfolio, 0, 0, 0, 0,
            decline_kind=DECLINE_FAULT, fault=fault)


def _periodic_box_contains(box_center, box_half, mode_center, mode_half):
    if mode_half >= np.pi:
        return True
    if box_half >= np.pi:
        return False
    return (_periodic_distance(box_center, mode_center) + box_half
            <= mode_half + 1.0e-14)


def _box_owners(lo, hi, centers, half_widths):
    midpoint = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    owners = []
    for index, (center, width) in enumerate(zip(centers, half_widths)):
        linear = ((lo[0] >= center[0] - width[0])
                  and (hi[0] <= center[0] + width[0])
                  and (lo[3] >= center[3] - width[3])
                  and (hi[3] <= center[3] + width[3]))
        angular = (_periodic_box_contains(
            midpoint[1], half[1], center[1], width[1])
            and _periodic_box_contains(
                midpoint[2], half[2], center[2], width[2]))
        if linear and angular:
            owners.append(index)
    if not owners:
        return [], -1
    scores = []
    for index in owners:
        delta = midpoint - centers[index]
        delta[1] = _periodic_distance(midpoint[1], centers[index, 1])
        delta[2] = _periodic_distance(midpoint[2], centers[index, 2])
        score = float(np.sum(np.square(
            delta / np.maximum(half_widths[index], 1.0e-300))))
        scores.append((score, index))
    return owners, min(scores)[1]


def _periodic_segments(center, half_width):
    """Represent a periodic interval as closed segments on ``[0, 2 pi]``."""
    if half_width >= np.pi:
        return [(0.0, 2.0 * np.pi)]
    lower = (float(center) - float(half_width)) % (2.0 * np.pi)
    upper = (float(center) + float(half_width)) % (2.0 * np.pi)
    if lower <= upper:
        return [(lower, upper)]
    return [(0.0, upper), (lower, 2.0 * np.pi)]


def _local_boundary_splits(lo, hi, centers, half_widths, available):
    """Return exact local-union boundaries cutting an intersecting box.

    Splitting at these coordinates is what allows the ledger to remove a leaf
    only when the *same axis-aligned region* will be locally integrated.
    """
    candidates = [[] for _ in range(4)]
    epsilon = 64.0 * np.finfo(float).eps
    for center, width in zip(centers, half_widths):
        intervals = [
            [(center[0] - width[0], center[0] + width[0])],
            _periodic_segments(center[1], width[1]),
            _periodic_segments(center[2], width[2]),
            [(center[3] - width[3], center[3] + width[3])],
        ]
        intersects = True
        for axis in range(4):
            if not any(max(lo[axis], left) < min(hi[axis], right)
                       for left, right in intervals[axis]):
                intersects = False
                break
        if not intersects:
            continue
        for axis in range(4):
            if not available[axis]:
                continue
            for left, right in intervals[axis]:
                for boundary in (left, right):
                    tolerance = epsilon * max(
                        1.0, abs(float(lo[axis])), abs(float(hi[axis])))
                    if lo[axis] + tolerance < boundary < hi[axis] - tolerance:
                        candidates[axis].append(float(boundary))
    return candidates


def _time_fourier_enclosure(C_A_t, max_order=8):
    """Return exact reflected coefficients and global derivative remainders."""
    reflected = np.concatenate(
        (C_A_t, np.flip(C_A_t[..., 1:-1], axis=-1)), axis=-1)
    coefficient = np.fft.fft(reflected, axis=-1) / reflected.shape[-1]
    frequency = np.fft.fftfreq(reflected.shape[-1])
    omega = 2.0 * np.pi * frequency
    magnitude = np.abs(coefficient)
    derivative_bounds = np.stack([
        np.sum(magnitude * np.abs(omega) ** order, axis=-1)
        for order in range(1, int(max_order) + 1)
    ])
    return (coefficient, frequency, derivative_bounds,
            np.sum(magnitude, axis=-1))


def _evaluate_spectrum_numpy(coefficient, frequency, time, derivative=0):
    """Evaluate the reflected finite Fourier polynomial or its derivative."""
    omega = 2.0 * np.pi * frequency
    phase = np.exp(1j * omega * float(time))
    nyquist = None
    if coefficient.shape[-1] % 2 == 0:
        nyquist = coefficient.shape[-1] // 2
    factor = np.power(1j * omega, int(derivative)) * phase
    if nyquist is not None:
        factor[nyquist] = (np.pi ** int(derivative)
                           * np.cos(np.pi * float(time)
                                    + 0.5 * np.pi * int(derivative)))
    return np.einsum("kqn,n->kq", coefficient, factor, optimize=True)


def _field_variation(table, phi, u, half_phi, half_u):
    table = np.asarray(table, dtype=np.complex128)
    kp = np.arange(table.shape[0], dtype=float)[:, None]
    ks = np.arange(-(table.shape[1] - 1) // 2,
                   (table.shape[1] - 1) // 2 + 1, dtype=float)[None, :]
    weight = _kp_weights(table.shape[0])[:, None]
    phase = np.exp(1j * (kp * float(phi) + ks * float(u)))
    value = float(np.sum(weight * table * phase).real)
    phase_span = np.minimum(
        2.0, np.abs(kp) * float(half_phi) + np.abs(ks) * float(half_u))
    variation = float(np.sum(weight * np.abs(table) * phase_span))
    return value, variation


def _box_log_upper(C_A_t, C_B, time_enclosure, lo, hi,
                   inherited_point_upper=np.inf):
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    coefficient, frequency, derivative_bounds, magnitude_bound = time_enclosure
    dt = float(half[0])
    C_A_center = _evaluate_spectrum_numpy(
        coefficient, frequency, center[0], derivative=0)
    A0, A_angle = _field_variation(
        C_A_center, center[1], center[2], half[1], half[2])
    # Each Taylor expression is independently rigorous for the reflected
    # finite Fourier polynomial.  Their minimum is rigorous too.  Higher-order
    # local cancellation matters for a narrow band-limited time peak: a global
    # first-derivative lift did not contract fast enough on real 22 tables.
    candidates = [magnitude_bound + np.abs(C_A_center)]
    partial = np.zeros_like(magnitude_bound)
    factorial = 1.0
    power = 1.0
    for order, bound in enumerate(derivative_bounds, start=1):
        factorial *= order
        power *= dt
        if order > 1:
            previous = _evaluate_spectrum_numpy(
                coefficient, frequency, center[0], derivative=order - 1)
            partial = partial + np.abs(previous) * (
                dt ** (order - 1)) / math.factorial(order - 1)
        candidates.append(partial + bound * power / factorial)
    time_remainder = np.minimum.reduce(candidates)
    A_time = float(np.sum(
        _kp_weights(C_A_t.shape[0])[:, None] * time_remainder))
    B0, B_angle = _field_variation(
        C_B, center[1], center[2], half[1], half[2])
    A_upper = A0 + A_angle + A_time
    B_lower = max(0.0, B0 - B_angle)  # intersect with B=<h|h> >= 0
    profile, _ = _distance_profile(
        np.asarray(A_upper), np.asarray(B_lower), lo[3], hi[3])
    volume = float(np.prod(hi - lo))
    if not np.isfinite(volume) or volume <= 0.0:
        return -np.inf, np.zeros(4)
    magnitude = (abs(A0) + A_angle + A_time + abs(B0) + B_angle
                 + abs(float(profile)) + 1.0)
    upper = float(profile) + 64.0 * np.finfo(float).eps * magnitude
    # A child is a subset of its parent.  Capping its pointwise enclosure by
    # the inherited parent enclosure is exact and makes the ledger's integral
    # upper bound monotone under subdivision.
    upper = min(upper, float(inherited_point_upper))
    angle_total = max(half[1] + half[2], 1.0e-300)
    score = np.asarray([
        float(hi[3]) * A_time,
        (float(hi[3]) * A_angle + 0.5 * hi[3] ** 2 * B_angle)
        * half[1] / angle_total,
        (float(hi[3]) * A_angle + 0.5 * hi[3] ** 2 * B_angle)
        * half[2] / angle_total,
        (abs(A_upper) + hi[3] * max(B0 + B_angle, 0.0)
         + 4.0 / max(lo[3], 1.0e-300)) * half[3],
    ])
    return math.log(volume) + upper, score, upper


def _logsumexp(values):
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return -np.inf
    top = float(np.max(values))
    if not np.isfinite(top):
        return top
    return top + math.log(float(np.sum(np.exp(values - top))))


def hierarchical_union_cover(
        C_A_t, uv_summary, centers, half_widths, x_min, x_max, *,
        target_log_value, outside_tol_nats=-23.0, max_boxes=50000,
        max_depth=(14, 10, 10, 10), progress_interval=512,
        stall_checks=3, min_progress_nats=0.5):
    """Adaptively upper-bound mass outside a union of local four-axis boxes.

    The largest unresolved coefficient-space box is split first.  A box wholly
    inside multiple local regions is assigned to one canonical nearest owner;
    overlaps therefore become disjoint tiles rather than a failure.  The cap
    limits work, not safety: every unresolved leaf retains a valid upper bound.
    """
    C_A_t = np.asarray(C_A_t, dtype=np.complex128)
    if not isinstance(uv_summary, UVQSummary):
        raise TypeError("uv_summary must come from summarize_uv_norm_table")
    _validate_tables(C_A_t, uv_summary.C_B)
    if not uv_summary.time_invariant:
        raise ValueError("arrival-time-dependent U,V norm cannot be collapsed")
    centers = np.asarray(centers, dtype=float)
    half_widths = np.asarray(half_widths, dtype=float)
    if (centers.ndim != 2 or centers.shape[1] != 4 or len(centers) == 0
            or half_widths.shape != centers.shape):
        raise ValueError("centers and half_widths must have shape (N,4), N>0")
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(half_widths)):
        raise ValueError("local boxes must be finite")
    if np.any(half_widths <= 0.0) or np.any(half_widths[:, 1:3] > np.pi):
        raise ValueError("half-widths must be positive and angular widths <= pi")
    if not (0.0 < float(x_min) < float(x_max)):
        raise ValueError("need 0 < x_min < x_max")
    if not np.isfinite(float(target_log_value)):
        raise ValueError("target_log_value must be finite")
    max_boxes = int(max_boxes)
    max_depth = np.asarray(max_depth, dtype=np.int32)
    if max_boxes < 1 or max_depth.shape != (4,) or np.any(max_depth < 0):
        raise ValueError("invalid cover cap")

    time_enclosure = _time_fourier_enclosure(C_A_t)
    domain_lo = np.asarray([0.0, 0.0, 0.0, float(x_min)])
    domain_hi = np.asarray([
        float(C_A_t.shape[-1] - 1), 2.0 * np.pi, 2.0 * np.pi,
        float(x_max)])
    heap = []
    owned = []
    counter = 0
    evaluated = 0
    overlap_owned = 0
    max_seen = np.zeros(4, dtype=np.int32)
    progress = []
    best_tail = np.inf
    stalled = False

    def add_box(lo, hi, depth, inherited_point_upper=np.inf):
        nonlocal counter, evaluated, overlap_owned
        owners, owner = _box_owners(lo, hi, centers, half_widths)
        evaluated += 1
        max_seen[:] = np.maximum(max_seen, depth)
        if owners:
            owned.append((0.5 * (lo + hi), 0.5 * (hi - lo), owner))
            overlap_owned += int(len(owners) > 1)
            return
        log_upper, score, point_upper = _box_log_upper(
            C_A_t, uv_summary.C_B, time_enclosure, lo, hi,
            inherited_point_upper)
        counter += 1
        heapq.heappush(
            heap, (-log_upper, counter, lo, hi, depth, score, point_upper))

    add_box(domain_lo, domain_hi, np.zeros(4, dtype=np.int32))
    initial_outside = _logsumexp(-item[0] for item in heap)
    initial_tail = initial_outside - float(target_log_value)
    best_tail = initial_tail
    subdivisions = 0
    cap_reached = False
    while heap:
        outside = _logsumexp(-item[0] for item in heap)
        tail = outside - float(target_log_value)
        best_tail = min(best_tail, tail)
        if outside - float(target_log_value) < float(outside_tol_nats):
            break
        if evaluated + 2 > max_boxes:
            cap_reached = True
            break
        item = heapq.heappop(heap)
        _, _, lo, hi, depth, score, parent_point_upper = item
        available = depth < max_depth
        if not np.any(available):
            heapq.heappush(heap, item)
            cap_reached = True
            break
        boundary = _local_boundary_splits(
            lo, hi, centers, half_widths, available)
        boundary_axes = np.asarray([bool(values) for values in boundary])
        split_score = np.where(available, score, -np.inf)
        if np.any(boundary_axes):
            # Use the same physics sensitivity score to order exact local-union
            # cuts.  These cuts establish ownership; midpoint cuts then tighten
            # the complement enclosure.
            axis = int(np.argmax(np.where(boundary_axes, score, -np.inf)))
            midpoint = 0.5 * (lo[axis] + hi[axis])
            middle = min(boundary[axis], key=lambda value: abs(value - midpoint))
        else:
            axis = int(np.argmax(split_score))
            if not np.isfinite(split_score[axis]) or hi[axis] <= lo[axis]:
                axis = int(np.flatnonzero(available)[0])
            middle = 0.5 * (lo[axis] + hi[axis])
        child_depth = depth.copy()
        child_depth[axis] += 1
        left_hi = hi.copy()
        left_hi[axis] = middle
        right_lo = lo.copy()
        right_lo[axis] = middle
        add_box(lo.copy(), left_hi, child_depth.copy(), parent_point_upper)
        add_box(right_lo, hi.copy(), child_depth.copy(), parent_point_upper)
        subdivisions += 1

        if int(progress_interval) > 0 and evaluated >= (
                len(progress) + 1) * int(progress_interval):
            checkpoint = _logsumexp(-leaf[0] for leaf in heap)
            checkpoint_tail = checkpoint - float(target_log_value)
            progress.append(checkpoint_tail)
            best_tail = min(best_tail, checkpoint_tail)
            if (len(progress) > int(stall_checks)
                    and checkpoint_tail > float(outside_tol_nats)
                    and progress[-1 - int(stall_checks)] - checkpoint_tail
                    < float(min_progress_nats)):
                stalled = True
                break

    outside = _logsumexp(-item[0] for item in heap)
    tail_margin = outside - float(target_log_value)
    owned_centers = np.asarray(
        [item[0] for item in owned], dtype=float).reshape((-1, 4))
    owned_widths = np.asarray(
        [item[1] for item in owned], dtype=float).reshape((-1, 4))
    owned_mode = np.asarray([item[2] for item in owned], dtype=np.int32)
    finite = bool(np.isfinite(outside) or outside == -np.inf)
    return CoverReport(
        float(outside), float(tail_margin), finite,
        bool(finite and tail_margin < float(outside_tol_nats)),
        bool(cap_reached), int(evaluated), int(subdivisions), int(len(heap)),
        int(len(owned)), int(overlap_owned), max_seen,
        float(initial_tail), float(best_tail), bool(stalled),
        np.asarray(progress, dtype=float),
        owned_centers, owned_widths, owned_mode)
