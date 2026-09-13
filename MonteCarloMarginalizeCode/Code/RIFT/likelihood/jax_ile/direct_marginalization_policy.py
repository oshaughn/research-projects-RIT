"""Opt-in cross-axis direct-marginalization policy for the JAX ILE arm.

``--direct-marginalization-policy auto`` composes, per likelihood evaluation,
the four-axis peak-local controller of :mod:`all_axis_peaklocal` with the
established exact-angle reserve:

1. build the guarded coefficient tables once (the same U,V/Q contraction the
   exact scheme uses), collapse the time-independent norm table per row;
2. rank a base and an enriched U,V/Q start portfolio on device, refine both in
   one shared optimizer pass, and freeze two nested fixed-shape plans;
3. attempt the four-axis local integral over ``(t, phi_ref, u=2 psi, x)``
   under the empirical enrichment gate;
4. on any decline, execute the band-limited exact-angle reserve on a refined
   time rule, warranted by the two-guard comparison and a half-refined check
   rule evaluated in the same declined branch;
5. return the selected value per row with the complete acceptance ledger.

No SNR threshold is coded.  Which branch runs is decided by the diagnostics
listed in :func:`policy_acceptance_diagnostics`.  A local decline is never a
waveform failure.  A reserve that fails its own warrant is escalated once per
doubling of the time rule up to ``reserve_time_refine_max``; a row that is
still unwarranted, or whose norm table varies with time, returns ``nan``.
The driver refuses to publish a run containing such rows.  The ledger keeps
the finite diagnostic value under ``selected_value`` for the record; it is
never handed to the sampler.

Measures.  The local branch integrates ``x**-4 dx  dt_sample  dphi  du``.
The reserve, and the production exact scheme it must agree with, average the
two angles (``dphi/2pi``, ``dpsi/pi``), weight distance by the normalized
``log_w_grid`` (or the normalized volumetric measure under
``JAX_ILE_DISTMARG_GH``), and integrate time in seconds with Simpson weights.
:func:`policy_log_normalization` derives the constant that converts the local
measure to that convention; it is a derivation from the prior's stated form,
never an inferred number, and it refuses any prior it cannot derive.

Not claimed here: derivative accuracy.  The plans are frozen under
``stop_gradient``; differentiating the composite differentiates the truncated
fixed-plan local integral or the reserve.  Value and gradient parity through
an SNR/HM ladder is the gate before this policy can become a default.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import all_axis_peaklocal as _aap
from . import anglemarg as _anglemarg
from . import core as _core
from . import peaklocal_time_reserve as _plr

__all__ = [
    "POLICY_CHOICES",
    "POLICY_DEFAULT",
    "RESERVE_SCHEME_TEST_ONLY",
    "reserve_pair",
    "resolve_reserve_angular_kernel",
    "q_bandwidth_cycles_per_sample",
    "PolicyConfig",
    "BoundedMultipeakConfig",
    "validate_bounded_multipeak_config",
    "validate_policy_request",
    "validate_policy_config",
    "policy_log_normalization",
    "policy_time_rules",
    "probe_guarded_tables",
    "policy_acceptance_diagnostics",
    "fused_log_likelihood_four_axis_policy",
    "fused_log_likelihood_four_axis_bounded",
    "summarize_policy_ledger",
    "RESERVE_SCHEME_CHOICES",
    "RESERVE_SCHEME_DEFAULT",
    "RESERVE_SCHEME_EXECUTABLE",
    "q_effective_bandwidth_hz",
    "predict_reserve_pair",
    "format_reserve_pair",
]

POLICY_CHOICES = ("off", "auto")
POLICY_DEFAULT = "off"
_LAPLACE_TABLE_KERNEL = "coefficient_table_distphipsimarg_laplace"

RESERVE_SCHEME_CHOICES = ("auto", "exact", "laplace", "peaklocal")
RESERVE_SCHEME_DEFAULT = "exact"
# A reserve scheme names a PAIR (angular kernel, time rule).  "auto" is
# resolved by predict_reserve_pair BEFORE construction; the composite refuses
# it.  "peaklocal-exact" is the peak-local time rule with the exact angular
# kernel: the accuracy reference the tests need, accepted on PolicyConfig
# and not offered on the command line.
_RESERVE_PAIRS = {
    "exact": ("exact", "window"),
    "laplace": ("laplace", "window"),
    "peaklocal": ("laplace", "peaklocal"),
    "peaklocal-exact": ("exact", "peaklocal"),
}
RESERVE_SCHEME_TEST_ONLY = ("peaklocal-exact",)

# WHICH OF THOSE THE COMPOSITE CAN ACTUALLY EXECUTE TODAY.  Kept separate from
# the choices tuple, and checked in validate_policy_config, because a config
# field the composite never reads is worse than a missing one: it accepts the
# value, reports it, and runs something else.  The reserve is dispatched through
# empirical_enrichment_with_exact_reserve, which names its kernel -- so 'exact'
# is the whole of what is wired.
#
# 'laplace' has its table-level kernel (coefficient_table_distphipsimarg_laplace,
# extracted from the fused laplace path in this same PR) but no dispatch: the
# selector may CHOOSE it, and a run that needs it is refused with that reason
# rather than carried by exact.  'peaklocal' belongs to RIFT PR #304.
# Executable = dispatched by the composite through reserve_pair: the exact
# and psi-Laplace kernels on the whole-window rule, and the peak-local time
# rule (peaklocal_time_reserve) under the psi-Laplace kernel.
RESERVE_SCHEME_EXECUTABLE = ("exact", "laplace", "peaklocal")


# The controller's own decline reasons, in the order they gate acceptance.
# Every key is a boolean per row in the returned ledger.
_DECLINE_KEYS = (
    "decline_nonfinite",
    "decline_capacity",
    "decline_no_modes",
    "decline_boundary_maximum",
    "decline_mode_nesting",
    "decline_time_reconstruction",
    "decline_time_cover_incomplete",
    "decline_time_omitted_mass_bound",
    "decline_time_omitted_mass",
    "decline_geometry",
    "decline_quadrature",
    "decline_enrichment",
    "decline_error_budget",
)


class PolicyConfig(NamedTuple):
    """Operating point of the composite.

    These are the values PR #268's device composition test and real captures
    ran with, exposed so the validation ladder can move them.  They are not a
    measured production operating point yet.
    """

    # Guard: the ladder record (RIFT_roboto_paper analyses/va_sequence_20260902/
    # RESULTS_20260907_aap268_ladder.md) accepted identically at guard 128 and
    # 1024 on production tables; 128 is the largest the driver's default
    # 0.15 s storage window supports.  A guard past the stored buffer is
    # refused at construction (see the wrapper's probe), not read as a decline.
    time_guard: int = 128
    reserve_time_refine: int = 4
    # Bounded escalation of the reserve rule on a failed warrant: the rule is
    # doubled (and re-checked against its own half) until it is warranted or
    # this factor is reached.  Rows still unwarranted return nan.
    reserve_time_refine_max: int = 32
    # Start capacity of the local plan.  Sized with max_time_nodes, not
    # independently: on the same 64 rows the two capacities decline almost the
    # same rows, 36 failing on time nodes and 37 on starts, with only 7 failing
    # on starts alone.  Raising either one by itself buys nothing.  Measured
    # acceptance at rho 652.3, full-sky prior draws, as (max_time_nodes,
    # base_max_starts): (64, 32) 28%, (256, 32) 31%, (256, 128) 75%,
    # (512, 256) 77%.  The pair moves acceptance from 28% to 75%; the second
    # doubling buys 2 points.
    #
    # The resize is free.  Compiled workspace is 0.546 GiB -- 5% of a 24 GiB
    # card -- and is UNCHANGED across that whole 16x capacity range, because
    # peak scratch is set by the reserve's amplitude-sized grids, which are
    # compiled into the graph whether or not any row reaches them.  So there is
    # no memory argument for a small capacity here, only a compile-time one.
    base_max_starts: int = 128
    # Time-node capacity of the local plan, SET IN ADVANCE rather than
    # discovered and then declined.  rank_joint_starts_from_uvq_device
    # defaults it to 64 and the policy never passed it, so the value that
    # gated the local path was unreachable from any config field or flag.
    #
    # Measured, rho 652.3, full-sky prior draws, 64 rows: the live-node count
    # a row needs is min 18, median 72, p90 217, max 614.  A cap of 64 holds
    # 44% of rows, 128 holds 73%, 256 holds 91%, 1024 holds 100%.  The median
    # row misses the old cap by eight nodes.
    #
    # 256 with base_max_starts 128, approved by RO on 2026-09-08 (evening).
    # What shipped was 64/32 -- the value rank_joint_starts_from_uvq_device
    # used while the policy passed nothing, reachable from no field or flag.
    # Measured acceptance at rho 652 as (max_time_nodes, base_max_starts):
    # (64, 32) 28%, (256, 32) 31%, (256, 128) 75%, (512, 256) 77%.  They move
    # together: of 64 rows, 36 declines fail on time nodes and 37 on starts,
    # only 7 on starts alone, so raising either alone plateaus near 16%.
    #
    # The price, and it is not zero: base_max_starts 32 -> 128 shifts
    # already-accepted values by up to 3.5e-3 nats at rho 163, so an A/B across
    # this change may not treat its difference as noise.  Accepted because a
    # declined row runs the exact reserve at hours per row at rho 652 while an
    # accepted one costs ~2.3 s.  Workspace is unaffected -- 0.546 GiB, flat
    # from (64, 32) to (1024, 256) on a 24 GiB card -- so the cost of the
    # resize is compile time, 13 s to ~70 s, not memory.
    max_time_nodes: int = 256
    # Angular oversample 2/4 and 16 modes are the configuration that accepted
    # on production tables at rho 163 and 326 (same record as above; 8 to 12
    # candidates against 32 starts).  PR #268's test values 1/2 and 4/8
    # overflowed capacity on synthetic carrier tables.
    base_oversample: int = 2
    enriched_oversample: int = 4
    max_modes: int = 16
    enriched_max_modes: int = 16
    # 6 whitened sigmas, the library default.  PR #268's composition test used
    # 3.0; on the wiring test's analytic fixture that truncated ~1% of the
    # four-dimensional mass and read 0.015 nat LOW against an independent
    # fine-time reference while the gate accepted, because base and enriched
    # plans share the truncation.  The gate cannot see this term; the wiring
    # test pins it against the external reference instead.
    local_radius: float = 6.0
    refine_iterations: int = 14
    base_order: int = 13
    base_check_order: int = 19
    enriched_order: int = 19
    enriched_check_order: int = 25
    convergence_tol_nats: float = 1.0e-3
    time_guard_tol_nats: float = 1.0e-3
    total_value_error_budget_nats: float = 1.0e-2
    time_outside_tol_nats: float = -23.0
    # 64, not the kernel's 8: the reserve's reverse pass keeps one carry per
    # dense-angle scan step, so gradient memory falls ~7x from 8 to 64
    # (5.3 -> 0.73 GiB at refine 2, 21 -> 2.9 GiB at refine 8 on a rho 163
    # production row).  The value is unchanged; per-step forward memory grows
    # with the chunk.
    reserve_dense_chunk: int = 64
    reserve_grid_block: int = 32
    # Rows the controller executes together under one ``vmap``.  1 is the
    # row-at-a-time path (``lax.map`` with no ``batch_size``), whose reserve
    # workspace is one row's.  Above 1, ``B`` rows share a scan step, device
    # workspace grows linearly in ``B``, and the tier-escalation ``lax.cond``
    # becomes a ``select`` that evaluates EVERY tier for EVERY row in the
    # batch.  Values, branch decisions and gradients are unchanged at every
    # size.  0 means one full batch of all rows.
    #
    # The default is 1 because batching is a COST REGRESSION.  The
    # tier-escalation cond is not the only one: the accept/reserve cond in
    # ``all_axis_peaklocal`` also becomes a select, so a locally ACCEPTED row
    # executes the dense reserve it would otherwise skip.  The penalty scales
    # with the locally accepted fraction, and the docstring there claiming an
    # accepted row never pays for the reserve is false above B=1.
    #
    # WITHDRAWN: this comment previously read "measured and does not pay",
    # citing 96.3 s per row at B=1 against 99.9 at B=8 on ladder-2 tables at
    # rho 40.8.  Every row in those runs DECLINED (accepted_local 0 of 8 and 0
    # of 2), so the figures price the decline path, and 37-50% of the rows were
    # nan under reserve_time_refine_max=4.  They also ran at refine == refine
    # _max, where the escalation cond is absent from the graph, so both penalty
    # mechanisms were inert.  Do not cite them.  What stands from that work is
    # the workspace law, 0.046 + 0.385 B GiB, and the value equivalence.
    # See DESIGN_direct_marginalization_policy.md.
    reserve_batch_rows: int = 1
    # WHICH reserve the composite falls back to.  Default 'exact' is what
    # shipped, so no existing command line changes.  'laplace' is the same
    # accuracy crossover the angle selector already validates
    # (ANGLE_MARG_CROSSOVER_AMPLITUDE): above it laplace is the MORE accurate
    # scheme and costs ~sqrt(A) rather than ~A.  'auto' chooses from the
    # precomputed inputs via predict_reserve_pair and REFUSES when the analysis
    # says the signal needs a method that is not implemented, rather than
    # falling back to whole-window refinement.  A string so a third value plugs
    # in without touching the composite.
    reserve_scheme: str = RESERVE_SCHEME_DEFAULT
    norm_invariance_rtol: float = 1.0e-10
    # Peak-local time rule (reserve schemes 'peaklocal', 'peaklocal-exact'):
    # fine nodes per plan mode block (49 = 8 sigma either side at 3 nodes per
    # predicted sigma), the coarse scan refinement of the window, the number
    # of warrant escalations (each doubles the fine lattice at fixed span),
    # and an override of the predicted width (nan = predict per row from
    # rho and the Q bandwidth; an experiment knob, printed in the ledger).
    # Measured operating point: DESIGN_direct_marginalization_policy.md,
    # "Peak-local time reserve".
    reserve_peaklocal_fine_nodes: int = 73
    reserve_peaklocal_scan_refine: int = 2
    reserve_peaklocal_escalations: int = 2
    # The row's time maxima come from the primitive (locate_time_maxima):
    # this many candidates, on a search grid of this many nodes per native
    # sample, over this angular lattice.  The local branch's plan is only a
    # cross-check in the ledger.
    reserve_peaklocal_blocks: int = 4
    reserve_peaklocal_search_refine: int = 8
    reserve_peaklocal_angular_lattice: int = 8
    # The coarse scan is support-limited: this many nodes across the hull of
    # the live maxima widened by this many predicted sigma each side; the
    # mass outside is bounded from the locator's search profile plus this
    # slack and charged to the warrant.  Node count never grows with the
    # window (RO, 2026-09-09).
    reserve_peaklocal_scan_nodes: int = 65
    reserve_peaklocal_scan_margin_sigmas: float = 16.0
    # Locator search sizing (measured 2026-09-09, rung 652 row 0): the search
    # grid's phi lattice leaves a ripple of about rho^2 (pi / n_phi)^2 nat in
    # the profile, 1000 nat at rho 652 on 64 nodes against a 25-nat time
    # structure per search cell, so the search maximum landed 0.4 samples off
    # and the focus certificate refused the row.  4096 nodes hold the ripple
    # under 1 nat up to rho 1300; the count is a static shape, so it is a
    # config field rather than a per-row prediction.  The Newton polish of
    # (phi, u) at each polished node was clipped to 0.1 rad per step for 3
    # steps, a 0.3 rad reach against a 0.4 rad lattice offset: 500 nat low on
    # the same row.  Eight steps within a 1 rad trust region reach it.
    reserve_peaklocal_search_phi_nodes: int = 4096
    reserve_peaklocal_newton_steps: int = 8
    reserve_peaklocal_newton_step_max: float = 1.0
    reserve_peaklocal_outside_slack_nats: float = 5.0
    reserve_peaklocal_sigma_t_override_samples: float = float("nan")


class BoundedMultipeakConfig(NamedTuple):
    """Static resource envelope for the device-only multipeak variant.

    Every field that changes compiled work is fixed before tracing.  A row
    that needs more starts, time nodes, modes, or quadrature accuracy than
    this envelope supplies is returned as ``nan``; there is deliberately no
    amplitude-sized dense reserve.  Planning is stopped from the AD graph, so
    derivatives are those of the accepted fixed-plan integral, not a claim
    about derivatives of the discrete mode-selection map.
    """

    # Compact by default: with the corrected orders this accepts the analytic
    # reference and costs 0.36 s / two rows on A100, versus 3.70 s with the
    # policy's 128/128/256 guard/start/time caps (2026-09-12). Wider caps can
    # recover important declines; expose them instead of paying for every
    # prior draw. See DESIGN_bounded_multipeak.md for the tradeoff and profile.
    time_guard: int = 16
    base_max_starts: int = 32
    max_time_nodes: int = 64
    base_oversample: int = PolicyConfig().base_oversample
    enriched_oversample: int = PolicyConfig().enriched_oversample
    max_modes: int = 8
    enriched_max_modes: int = 8
    local_radius: float = PolicyConfig().local_radius
    refine_iterations: int = PolicyConfig().refine_iterations
    # Deliberately cheaper than the reserve-bearing policy: 11/13/13/15
    # accepts the analytic reference where 7/9/9/11 declines even at 0.03 nat.
    # 0.01-nat convergence / 0.03-nat total budget favors practical acceptance;
    # all remain explicit CLI controls. See DESIGN_bounded_multipeak.md.
    base_order: int = 11
    base_check_order: int = 13
    enriched_order: int = 13
    enriched_check_order: int = 15
    convergence_tol_nats: float = 1.0e-2
    time_guard_tol_nats: float = PolicyConfig().time_guard_tol_nats
    total_value_error_budget_nats: float = 3.0e-2
    time_outside_tol_nats: float = PolicyConfig().time_outside_tol_nats
    norm_invariance_rtol: float = PolicyConfig().norm_invariance_rtol
    batch_rows: int = 1


def validate_bounded_multipeak_config(config):
    """Validate the complete static contract before JAX traces it."""
    if not isinstance(config, BoundedMultipeakConfig):
        raise TypeError("config must be a BoundedMultipeakConfig")
    if int(config.time_guard) < 2:
        raise ValueError("time_guard must be >= 2 for two-guard validation")
    if int(config.base_max_starts) < 1 or int(config.max_time_nodes) < 2:
        raise ValueError("base_max_starts >= 1 and max_time_nodes >= 2 are required")
    if (int(config.base_oversample) < 1
            or int(config.enriched_oversample) <= int(config.base_oversample)):
        raise ValueError("enriched_oversample must exceed base_oversample >= 1")
    if (int(config.max_modes) < 1
            or int(config.max_modes) > int(config.base_max_starts)
            or int(config.enriched_max_modes) < int(config.max_modes)
            or int(config.enriched_max_modes) > 2 * int(config.base_max_starts)):
        raise ValueError(
            "mode caps must satisfy 1 <= max_modes <= base_max_starts, "
            "max_modes <= enriched_max_modes <= 2 * base_max_starts")
    orders = (int(config.base_order), int(config.base_check_order),
              int(config.enriched_order), int(config.enriched_check_order))
    if not (2 <= orders[0] < orders[1] <= orders[2] < orders[3]):
        raise ValueError("need 2 <= base_order < base_check_order <= "
                         "enriched_order < enriched_check_order")
    if not float(config.local_radius) > 0.0 or int(config.refine_iterations) < 1:
        raise ValueError("local_radius and refine_iterations must be positive")
    if not (float(config.convergence_tol_nats) > 0.0
            and float(config.time_guard_tol_nats) > 0.0
            and float(config.total_value_error_budget_nats) > 0.0):
        raise ValueError("all error tolerances must be positive")
    if not (np.isfinite(float(config.time_outside_tol_nats))
            and float(config.time_outside_tol_nats) < 0.0):
        raise ValueError("time_outside_tol_nats must be finite and negative")
    if not (np.isfinite(float(config.norm_invariance_rtol))
            and float(config.norm_invariance_rtol) >= 0.0):
        raise ValueError("norm_invariance_rtol must be finite and non-negative")
    validate_batch_rows(config.batch_rows)
    return config


def validate_batch_rows(batch_rows):
    """Refuse a row-batch size the controller cannot execute.

    Returns the integer.  Negative sizes and non-integers are refused here so
    the driver and the library agree on the same rule.
    """
    try:
        b = int(batch_rows)
    except (TypeError, ValueError):
        raise ValueError("PolicyConfig.reserve_batch_rows must be an integer, "
                         "got %r" % (batch_rows,))
    if b != batch_rows:
        raise ValueError("PolicyConfig.reserve_batch_rows must be an integer, "
                         "got %r" % (batch_rows,))
    if b < 0:
        raise ValueError("PolicyConfig.reserve_batch_rows must be >= 0 "
                         "(1 = row at a time, 0 = one full batch), got %d" % b)
    return b


def reserve_pair(scheme):
    """``(angular_kernel, time_rule)`` named by a reserve scheme.

    ``auto`` is not a pair: it is resolved by :func:`predict_reserve_pair`
    before the likelihood is built, and the composite refuses it.
    """
    if scheme not in _RESERVE_PAIRS:
        if scheme == "auto":
            raise ValueError(
                "reserve_scheme='auto' must be resolved by predict_reserve_pair "
                "before construction; the composite takes a concrete scheme")
        raise ValueError("reserve scheme must be one of %r (plus %r for tests), "
                         "got %r" % (tuple(k for k in RESERVE_SCHEME_CHOICES
                                           if k != "auto"),
                                     RESERVE_SCHEME_TEST_ONLY, scheme))
    return _RESERVE_PAIRS[scheme]


def q_bandwidth_cycles_per_sample(data):
    """:func:`q_effective_bandwidth_hz` in cycles per native sample, or nan
    when the data carries no stored Q (synthetic tables)."""
    try:
        hz = float(q_effective_bandwidth_hz(data))
    except (AttributeError, KeyError, TypeError):
        return float("nan")
    return hz * float(data.deltaT) if np.isfinite(hz) else float("nan")


def resolve_reserve_angular_kernel(name, x_grid, log_w_grid, *, amp_sizing,
                                   m_max, dense_chunk, grid_block):
    """``None`` for the kernel's own exact default, else a table callable.

    The psi-Laplace table kernel is provided by ``anglemarg`` under the name
    ``coefficient_table_distphipsimarg_laplace`` (the --direct-marginalization-
    reserve-scheme laplace work); a tree without it refuses the request here,
    at construction, rather than at trace time.
    """
    if name not in ("exact", "laplace"):
        raise ValueError("angular kernel must be exact or laplace, got %r" % (name,))
    if name == "exact":
        return None
    fn = getattr(_anglemarg, _LAPLACE_TABLE_KERNEL, None)
    if fn is None:
        raise ValueError(
            "the laplace angular kernel needs anglemarg.%s, which this tree "
            "does not provide; use reserve scheme exact or peaklocal-exact"
            % _LAPLACE_TABLE_KERNEL)
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)

    # ``dense_chunk`` and ``grid_block`` are the EXACT kernel's streaming knobs;
    # the Laplace kernel streams by ``phi_chunk``/``dist_block``/``point_block``
    # and rejects them.  They stay in this signature so the caller is uniform,
    # and are deliberately not forwarded (measured 2026-09-09: the first
    # psi-Laplace reserve evaluation at rung 652 died on the keyword).
    del dense_chunk, grid_block

    def kernel(table, norm_table):
        return fn(table, norm_table, x_grid, log_w_grid,
                  amp_sizing=float(amp_sizing), m_max=int(m_max))
    return kernel


def validate_policy_config(config):
    """Refuse a PolicyConfig the composite would only reject at trace time."""
    if not isinstance(config, PolicyConfig):
        raise TypeError("policy_config must be a PolicyConfig")
    if int(config.time_guard) < 2:
        raise ValueError("PolicyConfig.time_guard must be >= 2: the local "
                         "path and the reserve both need the two-guard "
                         "comparison")
    f, fm = int(config.reserve_time_refine), int(config.reserve_time_refine_max)
    if f < 2 or f % 2:
        raise ValueError("reserve_time_refine must be an even integer >= 2 "
                         "so the check rule is the half-refined rule")
    if fm < f or fm % 2:
        raise ValueError("reserve_time_refine_max must be an even integer >= "
                         "reserve_time_refine")
    if not (np.isfinite(float(config.total_value_error_budget_nats))
            and float(config.total_value_error_budget_nats) > 0.0):
        raise ValueError("total_value_error_budget_nats must be finite and "
                         "positive")
    if int(config.max_time_nodes) < 2:
        raise ValueError("PolicyConfig.max_time_nodes must be >= 2: the time "
                         "cover needs at least one interior node pair")
    if int(config.base_max_starts) < 1:
        raise ValueError("PolicyConfig.base_max_starts must be >= 1")
    if int(config.base_oversample) < 1 or int(config.enriched_oversample) <= int(
            config.base_oversample):
        raise ValueError("enriched_oversample must exceed base_oversample "
                         "(the enriched portfolio must be strictly stronger)")
    if int(config.max_modes) < 1 or int(config.enriched_max_modes) < int(
            config.max_modes):
        raise ValueError("enriched_max_modes must be >= max_modes >= 1")
    if not float(config.local_radius) > 0.0:
        raise ValueError("local_radius must be positive")
    validate_batch_rows(config.reserve_batch_rows)
    scheme = config.reserve_scheme
    if scheme not in RESERVE_SCHEME_CHOICES + RESERVE_SCHEME_TEST_ONLY:
        raise ValueError("PolicyConfig.reserve_scheme must be one of %r, got %r"
                         % (RESERVE_SCHEME_CHOICES, scheme))
    # 'auto' is resolved by predict_reserve_pair against the precomputed inputs,
    # before this config ever reaches the composite; it is not a value the
    # composite executes, so it is admitted here and refused there if the
    # analysis lands on something unimplemented.
    if (scheme != "auto" and scheme not in RESERVE_SCHEME_EXECUTABLE
            and scheme not in RESERVE_SCHEME_TEST_ONLY):
        raise ValueError(
            "PolicyConfig.reserve_scheme=%r is a declared choice but is NOT "
            "WIRED into the composite: the reserve is dispatched through "
            "empirical_enrichment_with_exact_reserve and only %r is executable "
            "today.  Refused rather than run as 'exact', which is what a "
            "silently ignored field would do -- the run would report the "
            "scheme you asked for and compute the other one."
            % (scheme, RESERVE_SCHEME_EXECUTABLE))
    if int(config.max_time_nodes) < 2:
        raise ValueError("max_time_nodes must be >= 2")
    if scheme == "auto":
        return config
    angular, time_rule = reserve_pair(scheme)
    if (angular == "laplace"
            and getattr(_anglemarg, _LAPLACE_TABLE_KERNEL, None) is None):
        raise ValueError(
            "reserve scheme %r needs anglemarg.%s, which this tree does not "
            "provide" % (config.reserve_scheme, _LAPLACE_TABLE_KERNEL))
    if time_rule == "peaklocal":
        _plr.validate_peaklocal_rule_arguments(
            2, config.reserve_peaklocal_fine_nodes,
            config.reserve_peaklocal_scan_refine)
        if int(config.reserve_peaklocal_escalations) < 0:
            raise ValueError("reserve_peaklocal_escalations must be >= 0")
        if (int(config.reserve_peaklocal_blocks) < 1
                or int(config.reserve_peaklocal_search_refine) < 1
                or int(config.reserve_peaklocal_angular_lattice) < 2):
            raise ValueError("reserve_peaklocal_blocks >= 1, search_refine >= 1 "
                             "and angular_lattice >= 2 are required")
        ns = int(config.reserve_peaklocal_scan_nodes)
        if ns < 3 or ns % 2 == 0:
            raise ValueError("reserve_peaklocal_scan_nodes must be an odd integer >= 3")
        if int(config.reserve_peaklocal_search_phi_nodes) < 4:
            raise ValueError("reserve_peaklocal_search_phi_nodes must be >= 4")
        if int(config.reserve_peaklocal_newton_steps) < 1:
            raise ValueError("reserve_peaklocal_newton_steps must be >= 1")
        if not (float(config.reserve_peaklocal_newton_step_max) > 0.0):
            raise ValueError("reserve_peaklocal_newton_step_max must be positive")
        if not (float(config.reserve_peaklocal_scan_margin_sigmas) > 0.0
                and float(config.reserve_peaklocal_outside_slack_nats) >= 0.0):
            raise ValueError("reserve_peaklocal_scan_margin_sigmas must be positive "
                             "and reserve_peaklocal_outside_slack_nats >= 0")
        ov = float(config.reserve_peaklocal_sigma_t_override_samples)
        if np.isfinite(ov) and ov <= 0.0:
            raise ValueError("reserve_peaklocal_sigma_t_override_samples must "
                             "be positive or nan")
    return config


# ---------------------------------------------------------------------------
# Analysis-driven reserve selection
# ---------------------------------------------------------------------------
# RO, 2026-09-08: "we are learning a hard lesson about refinement and the
# 'reserve' not protecting us; we need to rely on ANALYSIS and the known physics
# ... have a hierarchy of methods and pick the expected bounding pairs as needed
# that apply to our signal."
#
# The failure mode being named is try-then-decline-then-refine: run the local
# branch, discover it declined, escalate a whole-window refinement, and discover
# at the END that most rows were carried by a method nobody chose.  Everything
# below is computable from the PRECOMPUTED inputs before any row is evaluated,
# so the pair is chosen and PRINTED up front and the run can be read in its
# first line instead of its last.

# PROVISIONAL, AND KNOWN TO BE THE WRONG MODEL FOR THE RESERVE.
#
# The points-per-sigma budget below treats the reserve's trapezoid rule as
# ALGEBRAICALLY convergent, so the node count it demands scales as
# window / sigma_t.  A direct test at rho 40.77 on 64 rows says otherwise:
#
#     refine=4  2453 nodes   warrant 1.8e-03 .. 1.14e-02
#     refine=8  4905 nodes   warrant 5e-11   .. 7.8e-09
#
# The tolerance those are read against has MOVED (#301: 1e-3 -> 1e-2), so the
# numbers are recorded without a verdict: refine=4 fails 1e-3 and straddles
# 1e-2.  See DESIGN_direct_marginalization_policy.md.
#
# Doubling the rule improved the quadrature error by ~1e6.  An algebraic rule
# would give 4.  That is the signature of the trapezoid rule on a BAND-LIMITED
# reconstruction, which is spectrally accurate once the band is resolved:
# error ~ exp(-c R), not R^-2.  The measured 4905 nodes is 67% of what this
# budget demands at that rung and lands five orders INSIDE tolerance.
#
# So the correct criterion is band resolution -- node spacing against the
# integrand's highest frequency -- not points per sigma, and the replacement
# must be FITTED to a measured convergence law rather than assumed.  Until a
# second rung is measured (163.08 at refine 4 and 8 is the deciding test), the
# time verdict below is provisional and MUST NOT be hardened into a threshold
# anyone tunes against.  It is retained because refusing is the conservative
# direction, but a refusal it produces is "unproven", not "shown inadequate".
#
# Fraction of a peak sigma the local branch's time cover must resolve.  The
# cover keeps cells above a mass threshold, so a peak narrower than the node
# spacing puts its mass in one cell and the cover cannot localize it.
_TIME_NODES_PER_SIGMA = 3.0


def q_effective_bandwidth_hz(data, moment="raw"):
    """Bandwidth of the stored Q(t), in Hz.  TWO different quantities.

    ``moment="raw"`` (default, and THE one to size a time rule with) returns
    sqrt(<f^2>) over the full two-sided spectrum.

    ``moment="central"`` returns the RMS bandwidth about the mean over positive
    frequencies -- the ENVELOPE bandwidth.

    WHICH ONE, AND WHY IT IS PHYSICS RATHER THAN CONVENTION.  The reserve
    marginalizes phi exactly, so the field in time is |zeta| with
    zeta = alpha kappa + beta kappa*, kappa = E(t) e^{i theta}, theta ~ 2 pi f_c t,
    and alpha, beta the polarization weights.  Then

        |zeta|^2 = (|alpha|^2 + |beta|^2) E^2 + 2 Re(alpha beta* E^2 e^{2 i theta})

    Face-on (beta -> 0) the carrier term vanishes, the field is the envelope, and
    the peak width is the envelope one -- the CENTRAL moment.  Linearly polarized
    (|alpha| = |beta|) the envelope is modulated at the carrier and each sub-peak
    is far narrower -- the RAW moment.  Every real row lies between, set by its
    own psi and inclination.

    So raw is the NARROWEST peak the primitive can produce at that amplitude and
    central the WIDEST.  A rule that must not under-resolve, and a selector that
    must not call a whole-window rule adequate when it is not, must size on the
    NARROW one.  Measured on a carrier fixture (f_c = 200 Hz, Gaussian envelope):
    circular gives a peak of 11.3 Hz equivalent against a central moment of
    5.6 Hz; linear gives 309.6 Hz against a raw moment of 200.1 Hz.

    RAW IS EXACT FOR THE INTEGRAND, NOT sqrt(2) OPTIMISTIC.  An earlier revision
    of this docstring claimed the latter, from measuring the curvature of
    |zeta|^2 itself.  That is not the integrand.  The quadrature integrates
    exp(lnL) with lnL = (rho^2/2) |zeta_hat|^2, so in the linear limit
    |zeta_hat|^2 = cos^2(omega t) ~ 1 - omega^2 t^2 gives
    lnL ~ const - (rho^2/2) omega^2 t^2 and hence sigma_t = 1/(rho omega)
    = 1/(2 pi rho f_c) exactly -- the raw moment, no factor.  The sqrt(2)
    appears only if the curvature of |zeta|^2 is read as a Gaussian width
    WITHOUT the rho^2/2 prefactor; the two multiply.  Verified against the
    log-integrand on a carrier fixture over four decades of rho:
    measured/predicted = 1.0008, 1.0000, 0.9999, 0.9999 at rho 12.65, 40.77,
    163.08, 652.31.

    Both are returned by name so the two can never be silently confused.  An
    earlier revision of this function made central the default and fed it to
    sigma_t, which is the optimistic error: it predicts the envelope width for
    rows whose likelihood is actually carrier-modulated.

    Q is stored at deltaT / q_time_pregrid_factor, so the frequency axis uses the
    REFINED spacing; using deltaT would understate the bandwidth by that factor.
    """
    import numpy as _np
    if moment not in ("central", "raw"):
        raise ValueError("moment must be 'central' or 'raw', got %r" % (moment,))
    num1 = num2 = den = 0.0
    for det in data.detector_names:
        d = data.detectors[det]
        Q = _np.asarray(d["Q"])                       # (npts_full, K)
        f_ref = int(d.get("q_time_pregrid_factor", 1)) or 1
        dt = float(data.deltaT) / float(f_ref)
        n = Q.shape[0]
        if n < 4:
            continue
        # Q (rholm) is COMPLEX, so this is the full two-sided transform: a
        # real-input transform rejects it outright, and dropping the imaginary
        # part would discard half the phase structure.
        freqs = _np.fft.fftfreq(n, d=dt)
        spec = _np.abs(_np.fft.fft(Q, axis=0)) ** 2   # (n, K)
        w = spec.sum(axis=1)
        if moment == "central":
            keep = freqs > 0.0
            freqs, w = freqs[keep], w[keep]
        num1 += float((freqs * w).sum())
        num2 += float((freqs ** 2 * w).sum())
        den += float(w.sum())
    if den <= 0.0:
        return float("nan")
    m1, m2 = num1 / den, num2 / den
    if moment == "raw":
        return float(_np.sqrt(max(m2, 0.0)))
    return float(_np.sqrt(max(m2 - m1 * m1, 0.0)))


def predict_reserve_pair(data, guess_snr, *, reserve_time_refine_max,
                         crossover_amplitude, max_time_nodes,
                         requested="auto", available=("exact", "laplace")):
    """Choose the (local, reserve) pair from the precomputed inputs.

    Returns ``(scheme, info)``.  ``scheme`` is None when the analysis says the
    signal needs a method that is not implemented; the caller must REFUSE rather
    than fall back, which is the whole point of predicting.

    The four quantities, all available before any row is evaluated:

    * ``rho``  -- the network SNR guess the driver already computes from the
      detector response.
    * ``sigma_f`` -- the effective bandwidth of the stored Q (above).
    * ``A = rho^2 / 2`` against ``ANGLE_MARG_CROSSOVER_AMPLITUDE``, the
      selector's own VALIDATED accuracy crossover between exact and laplace
      angles.  Above it laplace is the more accurate scheme AND costs ~sqrt(A)
      rather than ~A.
    * ``sigma_t = 1 / (2 pi rho sigma_f)`` -- the expected width of the time
      peak, in native samples, against what the whole-window refined reserve
      can AFFORD: ``(npts - 1) * reserve_time_refine_max + 1`` nodes.  A peak
      the escalation ceiling cannot resolve is the regime where refinement
      carries the rows without resolving them, which is what this function
      exists to predict rather than discover at the end of a run.

      NOTE the comparison is deliberately against the RESERVE's node budget and
      NOT against ``max_time_nodes``.  The latter caps the LOCAL branch's time
      CELL COVER, which is a different quantity: the cover exceeds its budget
      when lnL(t) is BROAD (many live cells, far from truth), not when the peak
      is narrow.  Comparing a whole-window node count against a cover budget
      mixes the two, which an earlier draft of this function did.
    """
    import numpy as _np
    rho = float(guess_snr) if guess_snr else float("nan")
    # RAW: the narrowest peak the primitive can make, which is what a time
    # rule must resolve.  Central is reported as the envelope bandwidth.
    sigma_f = q_effective_bandwidth_hz(data, moment='raw')
    sigma_f_env = q_effective_bandwidth_hz(data, moment='central')
    A = 0.5 * rho * rho if _np.isfinite(rho) else float("nan")
    dt = float(data.deltaT)
    if _np.isfinite(rho) and _np.isfinite(sigma_f) and rho > 0 and sigma_f > 0:
        sigma_t = 1.0 / (2.0 * _np.pi * rho * sigma_f)
    else:
        sigma_t = float("nan")
    width_samples = sigma_t / dt if _np.isfinite(sigma_t) else float("nan")
    # Nodes the cover would need to put _TIME_NODES_PER_SIGMA across one sigma
    # over the whole window, which is what a whole-window rule has to do.
    if _np.isfinite(width_samples) and width_samples > 0:
        nodes_needed = _TIME_NODES_PER_SIGMA * float(data.npts) / width_samples
    else:
        nodes_needed = float("inf")
    nodes_available = (float(data.npts) - 1.0) * float(
        reserve_time_refine_max) + 1.0
    time_reserve_ok = nodes_needed <= nodes_available

    # FIRST question, and the one about the branch actually under test: can the
    # LOCAL cover hold the peak?  Its budget is max_time_nodes, and a peak
    # narrower than a native sample needs the cover to place its nodes inside
    # one sample rather than across the window.  MEASURED, not modelled: at
    # rho 163 raising the cover 64 -> 256 took acceptance 31% -> 75%, which is
    # why the start cap appeared to "plateau" -- the plateau was time capacity
    # binding, not the start cap saturating.
    if _np.isfinite(width_samples) and width_samples > 0:
        cover_nodes_needed = _TIME_NODES_PER_SIGMA / width_samples
    else:
        cover_nodes_needed = float("inf")
    local_cover_ok = cover_nodes_needed <= float(max_time_nodes)
    time_local_ok = time_reserve_ok

    info = dict(rho=rho, sigma_f_hz=sigma_f,
                sigma_f_envelope_hz=sigma_f_env, amplitude_A=A,
                crossover_amplitude=float(crossover_amplitude),
                sigma_t_s=sigma_t, peak_width_samples=width_samples,
                cover_nodes_needed=cover_nodes_needed,
                max_time_nodes=int(max_time_nodes),
                local_cover_resolves_peak=bool(local_cover_ok),
                whole_window_nodes_needed=nodes_needed,
                whole_window_nodes_available=nodes_available,
                reserve_time_refine_max=int(reserve_time_refine_max),
                time_peak_resolvable_whole_window=bool(time_local_ok),
                requested=requested, available=tuple(available))

    if requested != "auto":
        # An explicit request overrides the ANALYSIS.  It does not override the
        # ROSTER: `available` says which schemes this data and this distance
        # quadrature can support at all, and forcing a scheme whose premise is
        # absent is not an override, it is an unnoticed wrong answer.
        if requested not in available:
            info["reason"] = ("explicit request %r is not on the roster for "
                              "this run (available: %s); the roster is a "
                              "property of the data and the distance "
                              "quadrature, not of the analysis, so it is not "
                              "overridable"
                              % (requested, ", ".join(available)))
            return None, info
        info["reason"] = "explicit request, no analysis applied"
        return requested, info

    if not _np.isfinite(A) or not _np.isfinite(sigma_f):
        info["reason"] = ("cannot analyse: rho=%r sigma_f=%r; refusing rather "
                          "than guessing" % (rho, sigma_f))
        return None, info

    # Angles: the validated accuracy crossover.
    angular = "laplace" if A > float(crossover_amplitude) else "exact"

    # Time: if a whole-window rule cannot resolve the peak within the local
    # branch's node budget, the pair needs a peak-local time reserve.  Saying so
    # and refusing is the point; falling back to refinement is what RO is
    # calling the hard lesson.
    if not time_local_ok:
        if "peaklocal" in available:
            info["reason"] = (
                "local cover %s (needs %.1f nodes of %d); "
                "A=%.4g > crossover %.4g selects laplace angles; peak is %.3g "
                "native samples wide and a whole-window rule would need %.0f "
                "nodes but the escalation ceiling affords %.0f, so the time "
                "reserve must be peak-local"
                % ("holds the peak" if local_cover_ok else "CANNOT hold the peak",
                   cover_nodes_needed, max_time_nodes,
                   A, crossover_amplitude,
                   width_samples, nodes_needed, nodes_available))
            return "peaklocal", info
        info["reason"] = (
            "local cover %s (needs %.1f nodes of %d); "
            "peak is %.3g native samples wide; a whole-window time rule would "
            "need %.0f nodes but the escalation ceiling affords only %.0f, so this signal needs a "
            "peak-local-in-time reserve, which is NOT IMPLEMENTED.  Refusing "
            "rather than falling back to whole-window refinement, which would "
            "carry the rows without anyone choosing it."
            % ("holds the peak" if local_cover_ok else "CANNOT hold the peak",
               cover_nodes_needed, max_time_nodes,
               width_samples, nodes_needed, nodes_available))
        return None, info

    # The roster is checked HERE too, not only on the peak-local branch.  It was
    # not, and the effect was that a run with laplace off the roster still
    # selected laplace whenever A cleared the crossover: the caller's roster was
    # honoured for the scheme it could not have chosen anyway and ignored for
    # the one it could.
    if angular not in available:
        info["reason"] = (
            "the accuracy crossover selects %s angles (A=%.4g against %.4g), "
            "and %s is NOT on the roster for this run (available: %s).  "
            "Refusing rather than running the other scheme under the selected "
            "one's name."
            % (angular, A, crossover_amplitude, angular, ", ".join(available)))
        return None, info

    info["reason"] = (
        "local cover %s (needs %.1f nodes of %d at this width); "
        "A=%.4g against crossover %.4g selects %s angles; peak is %.3g native "
        "samples wide and a whole-window rule needs %.0f nodes within the %.0f "
        "the ceiling affords, so the refined whole-window time reserve is adequate"
        % ("holds the peak" if local_cover_ok else "CANNOT hold the peak",
           cover_nodes_needed, max_time_nodes,
           A, crossover_amplitude, angular, width_samples, nodes_needed,
           nodes_available))
    return angular, info


def format_reserve_pair(scheme, info):
    """One line for the run log, printed BEFORE any row is evaluated."""
    return ("RESERVE-PAIR local=four-axis reserve=%s rho=%.4g sigma_f=%.4gHz "
            "A=%.4g crossover=%.4g peak=%.3gsamples cover_needs=%.1f/%d "
            "reserve_needs=%.0f/%.0f :: %s"
            % (scheme if scheme else "REFUSED", info.get("rho", float("nan")),
               info.get("sigma_f_hz", float("nan")),
               info.get("amplitude_A", float("nan")),
               info.get("crossover_amplitude", float("nan")),
               info.get("peak_width_samples", float("nan")),
               info.get("cover_nodes_needed", float("nan")),
               info.get("max_time_nodes", 0),
               info.get("whole_window_nodes_needed", float("nan")),
               info.get("whole_window_nodes_available", 0),
               info.get("reason", "")))


def validate_policy_request(policy, *, angle_marg_scheme, time_quadrature,
                            d_prior, dist_grid, reserve_scheme=None):
    """Refuse every combination the composite cannot honour.

    Refusal is explicit because an ignored request on this arm has a history
    of reading as a result (see the ``--angle-marg-scheme`` notes).
    """
    if policy not in POLICY_CHOICES:
        raise ValueError("direct_marginalization_policy must be one of %r, "
                         "got %r" % (POLICY_CHOICES, policy))
    if policy == "off":
        return
    if reserve_scheme is not None and reserve_scheme not in RESERVE_SCHEME_CHOICES:
        raise ValueError(
            "direct-marginalization reserve scheme must be one of %r, got %r"
            % (RESERVE_SCHEME_CHOICES, reserve_scheme))
    if angle_marg_scheme != "exact":
        raise ValueError(
            "--direct-marginalization-policy auto needs the exact-angle "
            "reserve: the resolved --angle-marg-scheme is %r, and this policy "
            "does not compose with grid, laplace, peak-local or phi-local.  "
            "Use --angle-marg-scheme exact." % (angle_marg_scheme,))
    if time_quadrature != "simpson":
        raise ValueError(
            "--direct-marginalization-policy auto owns the time integral "
            "(local four-axis or refined band-limited reserve) and only "
            "composes with the simpson terminal rule as its check rule; "
            "got %r." % (time_quadrature,))
    if d_prior not in ("euclidean", "volumetric"):
        raise ValueError(
            "--direct-marginalization-policy auto derives its local measure "
            "from the volumetric distance prior p(d) ~ d^2 only; got %r.  "
            "The cosmological prior is out of scope for the composite."
            % (d_prior,))
    if dist_grid != "uniform":
        raise ValueError(
            "--direct-marginalization-policy auto reads the reserve's distance "
            "normalization off a uniform-in-d grid; --distance-grid-scheme %r "
            "is not supported by the composite." % (dist_grid,))


def policy_log_normalization(data, x_grid, log_w_grid, *, d_prior="euclidean",
                             gh_nodes=None):
    """Constant converting the local ``x**-4 dx dt_sample dphi du`` integral to
    the reserve convention.  Returns ``(local_log_normalization, info)``.

    * angles: the reserve averages, so ``-2 log(2 pi)``;
    * time: sample units to seconds, scaled by whatever constant the
      production Simpson weights carry (``sum w_t == (npts-1) deltaT`` for the
      plain rule; the ratio is measured rather than assumed);
    * distance, fixed grid: ``p(d) dd = d^2 dd / N`` with
      ``d = Dref/x`` gives ``Dref^3 x^-4 dx / N``; ``N`` is recovered from the
      first weight, ``N = d_0^2 |d_1 - d_0| / w_0``, which is exact for the
      uniform grid the request validator requires;
    * distance, adaptive GH: the built-in normalized volumetric measure,
      ``3 / (x_min^-3 - x_max^-3)``.
    """
    if d_prior not in ("euclidean", "volumetric"):
        raise ValueError("policy_log_normalization supports the volumetric "
                         "prior only, got %r" % (d_prior,))
    x = np.asarray(x_grid, dtype=float)
    lw = np.asarray(log_w_grid, dtype=float)
    if x.ndim != 1 or x.size < 2 or lw.shape != x.shape:
        raise ValueError("x_grid/log_w_grid must be matching 1-D grids")
    if gh_nodes is None:
        gh_nodes = int(_core._DISTMARG_GH_N)
    deltaT = float(data.deltaT)
    npts = int(data.npts)
    w_t = np.asarray(data.w_t, dtype=float)
    plain = (npts - 1) * deltaT
    time_scale = float(np.sum(w_t)) / plain
    log_time = np.log(deltaT) + np.log(time_scale)
    log_angles = -2.0 * np.log(2.0 * np.pi)
    if int(gh_nodes) > 0:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        log_dist = np.log(3.0) - np.log(x_min ** -3 - x_max ** -3)
        dist_mode = "gh-volumetric"
    else:
        dref = float(data.distMpcRef)
        d = dref / x
        dd = np.abs(d[1] - d[0])
        if not np.allclose(np.abs(np.diff(d)), dd, rtol=1.0e-8, atol=0.0):
            raise ValueError("policy_log_normalization needs a uniform-in-d "
                             "distance grid")
        norm = d[0] ** 2 * dd / np.exp(lw[0])
        log_dist = 3.0 * np.log(dref) - np.log(norm)
        dist_mode = "fixed-grid-volumetric"
    total = float(log_angles + log_time + log_dist)
    info = dict(log_angles=float(log_angles), log_time=float(log_time),
                log_distance=float(log_dist), distance_mode=dist_mode,
                time_weight_scale=float(time_scale),
                local_log_normalization=total)
    return total, info


def _refined_rule(npts, deltaT, refine, scale):
    """Trapezoid rule on the ``refine``-times finer grid, in seconds.

    Trapezoid, not Simpson: on a peak narrower than the node spacing Simpson's
    alternating weights alias at half the spacing (review of PR #278 measured
    0.03 to 1.8 nat at refine 4 for peaks of 0.05 to 0.2 native samples), while
    the trapezoid rule converges exponentially on a smooth peak as the spacing
    shrinks, so a passed half-rule check means what it says.
    """
    n_nodes = (npts - 1) * refine + 1
    h = deltaT / float(refine)
    nodes = np.arange(n_nodes, dtype=float) / float(refine)
    nodes[-1] = float(npts - 1)
    weights = np.full(n_nodes, h)
    weights[0] = weights[-1] = 0.5 * h
    return nodes, weights * scale


def probe_guarded_tables(data, interp, guard, n_ra=6, decs=(-1.0, 0.0, 1.0)):
    """Refuse a guard the stored data buffer cannot supply.

    ``core._guarded_window`` gathers from ``-guard`` to ``npts+guard-1``;
    samples the build never stored come back nonfinite with no error, and a
    nonfinite table empties the start plan and reads as a method decline
    (ladder record, aap268_ladder README).  The reachable guard depends on the
    storage window and on the per-detector arrival offsets, which vary with the
    sky position, so the probe sweeps a coarse sky grid at construction and
    raises with the remedy if any table is nonfinite.  This is a preflight,
    not a certificate: the per-row ``tables_finite`` flag still gates every
    evaluation.
    """
    ra = jnp.asarray(np.tile(np.linspace(0.0, 2.0 * np.pi, int(n_ra),
                                         endpoint=False), len(decs)))
    dec = jnp.asarray(np.repeat(np.asarray(decs, dtype=float), int(n_ra)))
    incl = jnp.full(ra.shape, 0.5 * np.pi)
    C_A, C_B, _ = _anglemarg.angle_coefficient_tables(
        data, ra, dec, incl, interp, guard=int(guard))
    finite = bool(jnp.all(jnp.isfinite(C_A)) and jnp.all(jnp.isfinite(C_B)))
    if not finite:
        raise ValueError(
            "direct-marginalization policy: the guarded coefficient tables are "
            "not finite at time_guard=%d for this build.  The guard gathers "
            "%d samples beyond each end of the %d-sample window, and the "
            "stored data buffer (--internal-data-storage-window-half, minus "
            "the per-detector arrival offsets) does not reach that far.  "
            "Lower --direct-marginalization-time-guard or widen the storage "
            "window; a nonfinite table is not a likelihood decline."
            % (int(guard), int(guard), int(data.npts)))
    return True


def policy_time_rules(data, refine):
    """Refined reserve rule and its coarser check rule on the target window.

    Positions are in native samples of the unguarded window, ``0 .. npts-1``.
    Weights are trapezoid weights in seconds, carrying the same constant as
    the production ``data.w_t`` (so the reserve lands in production units
    without a separate offset).  The reserve rule refines the native cadence ``refine``
    times; the check rule refines it ``refine/2`` times (the native production
    rule itself when ``refine == 2``).  Agreement between the two is the
    resolution warrant, so the warrant is a convergence statement about the
    refined rules and does not require the native rule to be converged.
    """
    refine = int(refine)
    if refine < 2 or refine % 2:
        raise ValueError("reserve_time_refine must be an even integer >= 2 "
                         "so the check rule is the half-refined rule")
    npts = int(data.npts)
    deltaT = float(data.deltaT)
    w_t = np.asarray(data.w_t, dtype=float)
    scale = float(np.sum(w_t)) / ((npts - 1) * deltaT)
    nodes, weights = _refined_rule(npts, deltaT, refine, scale)
    if refine == 2:
        check_nodes, check_weights = np.arange(npts, dtype=float), w_t
    else:
        check_nodes, check_weights = _refined_rule(
            npts, deltaT, refine // 2, scale)
    return (jnp.asarray(nodes), jnp.asarray(weights),
            jnp.asarray(check_nodes), jnp.asarray(check_weights))


def policy_acceptance_diagnostics():
    """Names of the per-row booleans that must all hold for local acceptance,
    then the reserve warrant flags.  Documentation and audit order only."""
    return dict(
        local=("tables_finite", "norm_time_invariant", "base_capacity_ok",
               "enriched_capacity_ok", "boundary_maximum_ok",
               "base_and_enriched_values_finite",
               "mode_nesting_ok", "geometry_nesting_ok",
               "time_omitted_mass_ok", "value_error_budget_ok",
               "accepted_local"),
        reserve=("reserve_executed", "reserve_finite",
                 "reserve_time_guard_validated",
                 "reserve_time_resolution_validated",
                 "reserve_time_error_budget_ok", "reserve_time_warranted"),
        declines=_DECLINE_KEYS)


def _strong(tree):
    """Strip weak types so both branches of a ``lax.cond`` agree."""
    return jax.tree.map(
        lambda x: jax.lax.convert_element_type(jnp.asarray(x),
                                               jnp.asarray(x).dtype), tree)


def fused_log_likelihood_four_axis_policy(
        data, ra, dec, incl, x_grid, log_w_grid, *, interp, amp_sizing,
        config=None, local_log_normalization=None, return_ledger=False):
    """Distance-, phi_ref-, psi- AND time-marginalized lnL under the policy.

    Same contract as :func:`anglemarg.fused_log_likelihood_distphipsimarg_exact`
    without ``return_lnLt``: the composite owns the time integral, so there is
    no ``lnL(t)`` to hand back.  With ``return_ledger`` the per-row ledger of
    the controller is returned alongside (every leaf shaped ``(S,)``).
    """
    if config is None:
        config = PolicyConfig()
    validate_policy_config(config)
    guard = int(config.time_guard)
    batch_rows = validate_batch_rows(config.reserve_batch_rows)
    if local_log_normalization is None:
        local_log_normalization, _ = policy_log_normalization(
            data, x_grid, log_w_grid)
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    x_min = float(np.min(np.asarray(x_grid)))
    x_max = float(np.max(np.asarray(x_grid)))
    nodes, weights, check_nodes, check_weights = policy_time_rules(
        data, config.reserve_time_refine)

    C_A, C_B, meta = _anglemarg.angle_coefficient_tables(
        data, ra, dec, incl, interp, guard=guard)
    # (KP,KS,S,Ntime) -> (S,KP,KS,Ntime): one row per extrinsic sample.
    rows_A = jnp.moveaxis(C_A, 2, 0)
    rows_B = jnp.moveaxis(C_B, 2, 0)
    # Ordinary ILE has an arrival-time-independent norm.  Collapse it per row
    # and record the deviation; a row whose norm moves with time cannot be
    # planned by this composite and is reported, never averaged away.
    norm0 = rows_B[..., 0]
    norm_dev = jnp.max(jnp.abs(rows_B - norm0[..., None]), axis=(1, 2, 3))
    norm_scale = jnp.maximum(1.0, jnp.max(jnp.abs(norm0), axis=(1, 2)))
    norm_time_invariant = norm_dev <= float(config.norm_invariance_rtol) * norm_scale
    # A guard past the stored data buffer gathers samples the build never
    # stored; they come back nonfinite with no error, empty the plan and would
    # read as a method decline.  Name it as the input error it is.
    tables_finite = (jnp.all(jnp.isfinite(rows_A), axis=(1, 2, 3))
                     & jnp.all(jnp.isfinite(rows_B), axis=(1, 2, 3)))

    def _plan_row(table, norm):
        base = _aap.rank_joint_starts_from_uvq_device(
            table, norm, x_min, x_max, time_guard=guard,
            max_starts=int(config.base_max_starts),
            max_time_nodes=int(config.max_time_nodes),
            angular_oversample=int(config.base_oversample))
        extra = _aap.rank_joint_starts_from_uvq_device(
            table, norm, x_min, x_max, time_guard=guard,
            max_starts=int(config.base_max_starts),
            max_time_nodes=int(config.max_time_nodes),
            angular_oversample=int(config.enriched_oversample))
        (base_plan, enriched_plan, base_planning, enriched_planning,
         shared_planning) = _aap.make_all_axis_mode_plan_pair_device(
            table, norm, base, extra, x_min, x_max,
            max_modes=int(config.max_modes),
            enriched_max_modes=int(config.enriched_max_modes),
            local_radius=float(config.local_radius),
            time_guard=guard, iterations=int(config.refine_iterations),
            time_reconstruction_certified=False)
        # Row-local control data (inputs are already under stop_gradient; the
        # output cut is kept so a direct caller of _plan_row gets the same
        # contract).  Until derivative parity is established the discrete
        # rank/dedup decisions are not part of the differentiated graph.
        base_plan = jax.tree.map(jax.lax.stop_gradient, base_plan)
        enriched_plan = jax.tree.map(jax.lax.stop_gradient, enriched_plan)
        planning = dict(
            base_n_selected_modes=base_planning["n_selected_modes"],
            enriched_n_selected_modes=enriched_planning["n_selected_modes"],
            base_n_optimizer_starts=base_planning["n_optimizer_starts"],
            enriched_n_optimizer_starts=enriched_planning["n_optimizer_starts"],
            optimizer_starts_executed=shared_planning[
                "n_optimizer_starts_executed"],
            base_n_lattice_evaluations=base_planning["n_lattice_evaluations"],
            enriched_n_lattice_evaluations=enriched_planning[
                "n_lattice_evaluations"],
            # How far over the cap a declining row actually was.  decline_capacity
            # says only that n_candidates exceeded base_max_starts; without the
            # count there is no way to tell a row that missed by one from a row
            # that would need ten times the cap, and therefore no way to judge
            # whether raising the cap would recover anything.
            #
            # SCOPE, because the name would otherwise mislead exactly as the
            # sibling `enriched_*` keys misled a reader on 2026-09-08: the
            # second plan is built from `combine_device_start_plans(base, extra)`
            # (all_axis_peaklocal.py:1602 onward), so its count is base PLUS
            # extra (`:901`), while `capacity_ok` ANDs the two plans' own flags,
            # each already compared against base_max_starts separately (`:902`).
            # Comparing the combined count against the cap is therefore not a
            # test of anything.  Named `combined_` so the units travel with it.
            base_n_candidates_before_cap=base_planning[
                "n_candidates_before_cap"],
            combined_n_candidates_before_cap=enriched_planning[
                "n_candidates_before_cap"],
            # decline_capacity is charged for THREE different causes and the
            # ledger named only the union.  `capacity_ok` at :2067 is
            # discovery_capacity_ok on both plans; :1514 makes that
            # start_capacity_ok & ~selection_overflow; and the combined plan's
            # start_capacity_ok at :902 is itself
            # base.capacity_ok & extra.capacity_ok & same_time_support.  So a
            # row can carry decline_capacity with every candidate count under
            # the cap, and raising the cap cannot recover it.  Without these
            # two flags a count-based estimate of what a larger cap buys is an
            # upper bound and reads as if it were the answer.
            base_start_capacity_ok=base_planning["start_capacity_ok"],
            combined_start_capacity_ok=enriched_planning["start_capacity_ok"],
            base_selection_overflow=base_planning["selection_overflow"],
            combined_selection_overflow=enriched_planning[
                "selection_overflow"],
            base_norm_nonnegative=base_planning["norm_nonnegative"],
            combined_norm_nonnegative=enriched_planning["norm_nonnegative"],
            base_time_cover_certified=base_planning["time_cover_certified"],
            combined_time_cover_certified=enriched_planning[
                "time_cover_certified"],
            base_time_capacity_ok=base_planning["time_capacity_ok"],
            combined_time_capacity_ok=enriched_planning["time_capacity_ok"])
        return base_plan, enriched_plan, planning

    # Planning is control data.  Cutting the tangents at its INPUTS, not only
    # at the plan outputs, keeps reverse-mode AD from tracing the 14-step
    # Newton refinement over ~100 starts and stacking its residuals: with the
    # cut at the outputs only, a rho 163 production row still asked for 85 GiB
    # (158 GiB before the branches were rematerialized).
    base_plans, enriched_plans, planning = jax.vmap(_plan_row)(
        jax.lax.stop_gradient(rows_A), jax.lax.stop_gradient(norm0))

    angular_name, time_rule = reserve_pair(config.reserve_scheme)
    angular_kernel = resolve_reserve_angular_kernel(
        angular_name, x_grid, log_w_grid,
        amp_sizing=float(amp_sizing), m_max=int(meta["m_max"]),
        dense_chunk=int(config.reserve_dense_chunk),
        grid_block=int(config.reserve_grid_block))
    peaklocal = time_rule == "peaklocal"
    if peaklocal:
        # Tiers refine the fine lattice at fixed span (m -> 2m, n -> 2n-1);
        # the scan is fixed.  Node counts never depend on the row.  The
        # first tier is sized from the PREDICTED width; a tier beyond it is
        # a report on the prediction, counted in reserve_escalations.
        n0 = int(config.reserve_peaklocal_fine_nodes)
        tiers = []
        n_fine, mult = n0, 1
        for _ in range(int(config.reserve_peaklocal_escalations) + 1):
            tiers.append(("peaklocal", n_fine, mult))
            n_fine, mult = 2 * n_fine - 1, 2 * mult
        # Seconds per native sample carrying the production w_t constant, as
        # policy_time_rules does.
        dt_scale = float(np.sum(np.asarray(data.w_t, dtype=float))) / (
            int(data.npts) - 1)
        # Bandwidth of the stored Q, cycles per sample (nan without a Q; the
        # row's own table then supplies it, and both are in the ledger).
        sigma_f_q = q_bandwidth_cycles_per_sample(data)
        sigma_t_override = float(config.reserve_peaklocal_sigma_t_override_samples)
    else:
        refine0 = int(config.reserve_time_refine)
        refine_max = int(config.reserve_time_refine_max)
        if refine_max < refine0:
            raise ValueError("reserve_time_refine_max must be >= reserve_time_refine")
        tiers = []
        f = refine0
        while f <= refine_max:
            tiers.append((f,) + tuple(policy_time_rules(data, f)))
            f *= 2

    def _predict_row(table, norm, rho_located):
        # rho from the located profile maximum (rho^2 = 2 P_max); the angular
        # triangle bound is the fallback and is reported beside it (it ran
        # 2.8x over on a rung-160 row, 453 against 157).
        rho_bound = _plr.row_amplitude(table, norm, guard)
        rho = jnp.where(jnp.isfinite(rho_located) & (rho_located > 0.0),
                        rho_located, rho_bound)
        sigma_f_table = _plr.table_bandwidth_cycles(table, guard)
        sigma_f = jnp.where(jnp.isfinite(sigma_f_q), sigma_f_q, sigma_f_table)
        sigma_t = _plr.predicted_width_samples(rho, sigma_f)
        if np.isfinite(sigma_t_override):
            sigma_t = jnp.asarray(sigma_t_override, dtype=jnp.float64)
        return dict(rho=rho, rho_bound=rho_bound, sigma_f_table=sigma_f_table,
                    sigma_f=sigma_f, sigma_t=sigma_t)

    def _rule_for_tier(tier, table, norm, base_plan, enriched_plan):
        if tier[0] == "peaklocal":
            found = _plr.locate_time_maxima(
                table, norm, guard, int(data.npts), x_min, x_max,
                n_candidates=int(config.reserve_peaklocal_blocks),
                search_refine=int(config.reserve_peaklocal_search_refine),
                angular_lattice=int(config.reserve_peaklocal_angular_lattice),
                n_phi=int(config.reserve_peaklocal_search_phi_nodes),
                newton_steps=int(config.reserve_peaklocal_newton_steps),
                newton_step_max=float(config.reserve_peaklocal_newton_step_max))
            pred = _predict_row(table, norm, found["rho_located"])
            sigma_for_margin = jnp.minimum(
                jnp.where(jnp.isfinite(pred["sigma_t"]), pred["sigma_t"], jnp.inf),
                jnp.min(jnp.where(found["live"] & (found["widths"] > 0.0),
                                  found["widths"], jnp.inf)))
            sigma_for_margin = jnp.where(jnp.isfinite(sigma_for_margin),
                                         sigma_for_margin, 1.0)
            margin = float(config.reserve_peaklocal_scan_margin_sigmas) * sigma_for_margin
            rule = _plr.peaklocal_time_rule(
                found["centres"], found["widths"], found["live"],
                int(data.npts), dt_scale,
                sigma_t_samples=pred["sigma_t"], n_fine=int(tier[1]),
                scan_refine=int(config.reserve_peaklocal_scan_refine),
                fine_refine_multiplier=int(tier[2]),
                n_scan=int(config.reserve_peaklocal_scan_nodes),
                margin_samples=margin,
                search_positions=found["search_positions"],
                search_profile=found["search_profile"],
                outside_slack_nats=float(config.reserve_peaklocal_outside_slack_nats))
            # The local branch's plan, as a cross-check only: its narrowest
            # live Newton width and the distance from its first live centre
            # to the locator's first block.
            plan_live = jnp.concatenate((base_plan.live, enriched_plan.live)).astype(bool)
            plan_c = jnp.concatenate((base_plan.centers[:, 0], enriched_plan.centers[:, 0]))
            plan_w = jnp.abs(jnp.concatenate((base_plan.local_transforms[:, 0, 0],
                                              enriched_plan.local_transforms[:, 0, 0])))
            plan_width = jnp.min(jnp.where(plan_live & (plan_w > 0.0), plan_w, jnp.inf))
            plan_centre = jnp.where(jnp.any(plan_live),
                                    plan_c[jnp.argmax(plan_live)], jnp.nan)
            ratio = rule["sigma_t_located_samples"] / rule["sigma_t_pred_samples"]
            extra = dict(
                reserve_time_refine_used=jnp.asarray(0),
                reserve_time_rule_peaklocal=jnp.asarray(True),
                reserve_peaklocal_fine_nodes_used=jnp.asarray(int(tier[1])),
                reserve_peaklocal_fine_refine=rule["fine_refine"],
                reserve_peaklocal_fine_spacing_samples=rule["fine_spacing_samples"],
                reserve_peaklocal_block_span_samples=rule["block_span_samples"],
                reserve_peaklocal_live_blocks=rule["n_live_blocks"],
                reserve_peaklocal_first_block_centre_samples=rule[
                    "first_block_centre_samples"],
                reserve_peaklocal_rho_pred=pred["rho"],
                reserve_peaklocal_rho_bound=pred["rho_bound"],
                reserve_peaklocal_search_phi_nodes=jnp.asarray(
                    int(config.reserve_peaklocal_search_phi_nodes), dtype=jnp.int32),
                # Where the rule is fine: the kernel's focus certificate.
                reserve_peaklocal_focus_centre_samples=rule["first_block_centre_samples"],
                reserve_peaklocal_focus_half_width_samples=0.25 * rule["block_span_samples"],
                reserve_peaklocal_scan_lo_samples=rule["scan_lo_samples"],
                reserve_peaklocal_scan_hi_samples=rule["scan_hi_samples"],
                reserve_peaklocal_scan_margin_samples=margin,
                reserve_peaklocal_outside_log_bound=rule["outside_log_bound"],
                reserve_peaklocal_outside_search_nodes=rule["n_outside_search_nodes"],
                reserve_peaklocal_sigma_f_q_cycles=jnp.asarray(
                    sigma_f_q, dtype=jnp.float64),
                reserve_peaklocal_sigma_f_table_cycles=pred["sigma_f_table"],
                reserve_peaklocal_sigma_t_pred_samples=rule["sigma_t_pred_samples"],
                reserve_peaklocal_sigma_t_located_samples=rule["sigma_t_located_samples"],
                reserve_peaklocal_sigma_t_used_samples=rule["sigma_t_used_samples"],
                reserve_peaklocal_sigma_t_plan_samples=plan_width,
                reserve_peaklocal_plan_centre_offset_samples=jnp.abs(
                    plan_centre - rule["first_block_centre_samples"]),
                reserve_peaklocal_prediction_finite=rule["prediction_finite"],
                # The located curvature width against the prediction.  The
                # prediction is the NARROWEST peak the primitive can make at
                # this amplitude (raw rms frequency), so a face-on envelope
                # is legitimately wider, by the raw-to-central moment ratio:
                # consistent means located / predicted in [0.5, 4].  Outside
                # it the prediction is reported as disagreeing; the warrant
                # decides the row either way.
                reserve_peaklocal_prediction_consistent=(
                    jnp.isfinite(ratio) & (ratio >= 0.5) & (ratio <= 4.0)))
            return (rule["nodes"], rule["weights"], rule["check_nodes"],
                    rule["check_weights"], extra)
        refine, nodes, weights, check_nodes, check_weights = tier
        nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
        extra = dict(
            reserve_time_refine_used=jnp.asarray(int(refine)),
            reserve_time_rule_peaklocal=jnp.asarray(False),
            reserve_peaklocal_fine_nodes_used=jnp.asarray(0),
            reserve_peaklocal_fine_refine=jnp.asarray(0, dtype=jnp.int32),
            reserve_peaklocal_fine_spacing_samples=nan,
            reserve_peaklocal_block_span_samples=nan,
            reserve_peaklocal_live_blocks=jnp.asarray(0, dtype=jnp.int32),
            reserve_peaklocal_first_block_centre_samples=nan,
            reserve_peaklocal_rho_pred=nan,
            reserve_peaklocal_rho_bound=nan,
            reserve_peaklocal_search_phi_nodes=jnp.asarray(0, dtype=jnp.int32),
            reserve_peaklocal_focus_centre_samples=nan,
            reserve_peaklocal_focus_half_width_samples=nan,
            reserve_peaklocal_scan_lo_samples=nan,
            reserve_peaklocal_scan_hi_samples=nan,
            reserve_peaklocal_scan_margin_samples=nan,
            reserve_peaklocal_outside_log_bound=nan,
            reserve_peaklocal_outside_search_nodes=jnp.asarray(0, dtype=jnp.int32),
            reserve_peaklocal_sigma_f_q_cycles=nan,
            reserve_peaklocal_sigma_f_table_cycles=nan,
            reserve_peaklocal_sigma_t_pred_samples=nan,
            reserve_peaklocal_sigma_t_located_samples=nan,
            reserve_peaklocal_sigma_t_used_samples=nan,
            reserve_peaklocal_sigma_t_plan_samples=nan,
            reserve_peaklocal_plan_centre_offset_samples=nan,
            reserve_peaklocal_prediction_finite=jnp.asarray(False),
            reserve_peaklocal_prediction_consistent=jnp.asarray(False))
        return nodes, weights, check_nodes, check_weights, extra

    def _controller(table, norm, base_plan, enriched_plan, tier):
        nodes, weights, check_nodes, check_weights, extra = _rule_for_tier(
            tier, table, norm, base_plan, enriched_plan)
        sel, ok, led = _aap.empirical_enrichment_with_exact_reserve(
            table, norm, base_plan, enriched_plan, x_min, x_max,
            reserve_x_grid=x_grid, reserve_log_weights=log_w_grid,
            time_weights=weights,
            reserve_amp_sizing=float(amp_sizing),
            reserve_m_max=int(meta["m_max"]),
            reserve_dense_chunk=int(config.reserve_dense_chunk),
            reserve_grid_block=int(config.reserve_grid_block),
            reserve_time_nodes=nodes,
            reserve_time_check_nodes=check_nodes,
            reserve_time_check_weights=check_weights,
            reserve_time_resolution_tol_nats=float(
                config.total_value_error_budget_nats),
            base_order=int(config.base_order),
            base_check_order=int(config.base_check_order),
            enriched_order=int(config.enriched_order),
            enriched_check_order=int(config.enriched_check_order),
            convergence_tol_nats=float(config.convergence_tol_nats),
            time_guard=guard,
            time_guard_tol_nats=float(config.time_guard_tol_nats),
            local_log_normalization=float(local_log_normalization),
            time_outside_tol_nats=float(config.time_outside_tol_nats),
            total_value_error_budget_nats=float(
                config.total_value_error_budget_nats),
            reserve_log_offset=0.0,
            reserve_angular_kernel=angular_kernel,
            reserve_time_focus=(
                None if not peaklocal else
                (extra["reserve_peaklocal_focus_centre_samples"],
                 extra["reserve_peaklocal_focus_half_width_samples"])),
            reserve_time_cover=(
                None if not peaklocal else
                (extra["reserve_peaklocal_scan_lo_samples"],
                 extra["reserve_peaklocal_scan_hi_samples"],
                 extra["reserve_peaklocal_outside_log_bound"])))
        led = dict(led)
        led.update(extra)
        return _strong((sel, ok, led))

    def _row(args):
        table, norm, base_plan, enriched_plan = args
        # Each tier is rematerialized: reverse-mode AD otherwise keeps the
        # residuals of every tier's dense reserve alive at once.
        state = jax.checkpoint(
            lambda t, nm, bp, ep: _controller(t, nm, bp, ep, tiers[0]))(
                table, norm, base_plan, enriched_plan)
        escalations = jnp.asarray(0)
        for tier in tiers[1:]:
            sel, ok, led = state
            need = (led["reserve_executed"] & led["reserve_finite"]
                    & (~led["reserve_time_warranted"]))
            run_tier = jax.checkpoint(
                lambda _, tier=tier: _controller(
                    table, norm, base_plan, enriched_plan, tier))
            state = jax.lax.cond(need, run_tier, lambda st: st, state)
            escalations = escalations + need.astype(escalations.dtype)
        sel, ok, led = state
        led = dict(led)
        led["reserve_escalations"] = escalations
        return sel, ok, led

    n_rows = int(rows_A.shape[0])
    xs = (rows_A, norm0, base_plans, enriched_plans)
    if batch_rows == 1 or n_rows == 1:
        # Row at a time.  Kept as a distinct call rather than batch_size=1 so
        # the graph is the one PR #268 measured: batch_size=1 would still wrap
        # the body in a vmap, paying the cond-to-select cost for no occupancy.
        #
        # n_rows == 1 takes this path whatever was requested.  The wrapper's
        # _scalar evaluates ONE row, so value_and_grad and hessian always land
        # here; a vmap over a single row would convert both conds to selects
        # and pay every reserve tier and both accept/reserve branches with no
        # second row to amortize them.  Requesting a batch must not make the
        # gradient path more expensive than not requesting one.
        selected, usable, ledger = jax.lax.map(_row, xs)
    elif batch_rows == 0 or batch_rows >= n_rows:
        # One full batch.  Spelled as an explicit vmap rather than delegated
        # to batch_size: jax 0.9.2 documents batch_size=0 as a full vmap, but
        # the IGWN environment's jax 0.7.1 computes n // batch_size first and
        # raises ZeroDivisionError, and a batch_size above the row count is a
        # zero-length scan plus a remainder in both.  The explicit vmap is the
        # same computation in every version.
        selected, usable, ledger = jax.vmap(_row)(xs)
    else:
        selected, usable, ledger = jax.lax.map(_row, xs,
                                               batch_size=batch_rows)
    ledger = dict(ledger)
    # Truthful, not decorative: this key read True unconditionally before the
    # batch size was a knob.  Record what EXECUTED, not what was asked for:
    # a request of 8 against 6 rows runs a 6-row vmap, and a request of 8 on
    # the single-row gradient path runs sequentially.  Reporting the request
    # on the one key whose purpose is truthfulness is how the old hardcoded
    # True happened.  lax.map's trailing remainder means the batch a given row
    # landed in is still not recoverable per row, so this is the size of the
    # scanned batch; the request is kept beside it.
    if batch_rows == 1 or n_rows == 1:
        batch_executed = 1
    elif batch_rows == 0 or batch_rows >= n_rows:
        batch_executed = n_rows
    else:
        batch_executed = batch_rows
    ledger["reserve_batch_execution_sequential"] = jnp.full(
        (n_rows,), batch_executed == 1, dtype=bool)
    ledger["reserve_batch_rows_executed"] = jnp.full(
        (n_rows,), batch_executed, dtype=jnp.int32)
    ledger["reserve_batch_rows_requested"] = jnp.full(
        (n_rows,), batch_rows, dtype=jnp.int32)
    usable = usable & norm_time_invariant & tables_finite
    # Fail closed: a value the controller could not warrant is not a
    # likelihood.  nan, never the finite diagnostic, reaches the sampler; the
    # driver refuses to publish a run that contains such rows.
    lnL = jnp.where(usable, selected, jnp.nan)
    ledger.update(planning)
    ledger["norm_time_invariant"] = norm_time_invariant
    ledger["norm_time_deviation"] = norm_dev
    ledger["tables_finite"] = tables_finite
    ledger["input_nonfinite"] = ~tables_finite
    ledger["usable"] = usable
    ledger["selected_value"] = selected
    ledger["lnL"] = lnL
    if return_ledger:
        return lnL, ledger
    return lnL


def fused_log_likelihood_four_axis_bounded(
        data, ra, dec, incl, x_grid, log_w_grid, *, interp, amp_sizing,
        config=None, local_log_normalization=None, x_bounds=None,
        return_ledger=False):
    """Device-only, statically bounded four-axis multipeak marginalization.

    Discovery, Newton refinement, retained modes, and both nested quadrature
    rules are fixed-shape functions of :class:`BoundedMultipeakConfig`.  No
    dense or amplitude-sized reserve is present.  A row that does not pass the
    existing empirical enrichment, quadrature, geometry, time-guard, and
    omitted-time gates fails closed with ``nan``.

    The function is JIT- and AD-compatible.  Mode planning consumes
    ``stop_gradient`` coefficient tables and the resulting plans are stopped
    again, bounding reverse-mode storage and defining AD as differentiation of
    the accepted fixed-plan integral.  This is not a derivative-accuracy
    warrant; value/gradient parity on production ladders remains separate.

    ``x_bounds`` and ``local_log_normalization`` may be supplied by a wrapper
    that traces this function.  When omitted they are derived eagerly from the
    concrete distance grid.
    """
    if config is None:
        config = BoundedMultipeakConfig()
    validate_bounded_multipeak_config(config)
    guard = int(config.time_guard)
    batch_rows = validate_batch_rows(config.batch_rows)
    if local_log_normalization is None:
        local_log_normalization, _ = policy_log_normalization(
            data, x_grid, log_w_grid)
    if x_bounds is None:
        x_host = np.asarray(x_grid)
        x_min, x_max = float(np.min(x_host)), float(np.max(x_host))
    else:
        x_min, x_max = (float(x_bounds[0]), float(x_bounds[1]))
    if not (0.0 < x_min < x_max):
        raise ValueError("x_bounds must satisfy 0 < x_min < x_max")

    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64)
    C_A, C_B, _ = _anglemarg.angle_coefficient_tables(
        data, ra, dec, incl, interp, guard=guard)
    # amp_sizing is retained for call compatibility, not used to size work.
    rows_A = jnp.moveaxis(C_A, 2, 0)
    rows_B = jnp.moveaxis(C_B, 2, 0)
    norm0 = rows_B[..., 0]
    norm_dev = jnp.max(jnp.abs(rows_B - norm0[..., None]), axis=(1, 2, 3))
    norm_scale = jnp.maximum(1.0, jnp.max(jnp.abs(norm0), axis=(1, 2)))
    norm_time_invariant = (
        norm_dev <= float(config.norm_invariance_rtol) * norm_scale)
    tables_finite = (jnp.all(jnp.isfinite(rows_A), axis=(1, 2, 3))
                     & jnp.all(jnp.isfinite(rows_B), axis=(1, 2, 3)))

    def _plan_row(table, norm):
        base = _aap.rank_joint_starts_from_uvq_device(
            table, norm, x_min, x_max, time_guard=guard,
            max_starts=int(config.base_max_starts),
            max_time_nodes=int(config.max_time_nodes),
            angular_oversample=int(config.base_oversample))
        extra = _aap.rank_joint_starts_from_uvq_device(
            table, norm, x_min, x_max, time_guard=guard,
            max_starts=int(config.base_max_starts),
            max_time_nodes=int(config.max_time_nodes),
            angular_oversample=int(config.enriched_oversample))
        base_plan, enriched_plan, bp, ep, shared = (
            _aap.make_all_axis_mode_plan_pair_device(
                table, norm, base, extra, x_min, x_max,
                max_modes=int(config.max_modes),
                enriched_max_modes=int(config.enriched_max_modes),
                local_radius=float(config.local_radius), time_guard=guard,
                iterations=int(config.refine_iterations),
                time_reconstruction_certified=False))
        base_plan = jax.tree.map(jax.lax.stop_gradient, base_plan)
        enriched_plan = jax.tree.map(jax.lax.stop_gradient, enriched_plan)
        planning = dict(
            base_n_selected_modes=bp["n_selected_modes"],
            enriched_n_selected_modes=ep["n_selected_modes"],
            base_n_optimizer_starts=bp["n_optimizer_starts"],
            enriched_n_optimizer_starts=ep["n_optimizer_starts"],
            optimizer_starts_executed=shared["n_optimizer_starts_executed"],
            base_n_lattice_evaluations=bp["n_lattice_evaluations"],
            enriched_n_lattice_evaluations=ep["n_lattice_evaluations"],
            base_n_candidates_before_cap=bp["n_candidates_before_cap"],
            combined_n_candidates_before_cap=ep["n_candidates_before_cap"],
            base_start_capacity_ok=bp["start_capacity_ok"],
            combined_start_capacity_ok=ep["start_capacity_ok"],
            base_time_capacity_ok=bp["time_capacity_ok"],
            combined_time_capacity_ok=ep["time_capacity_ok"])
        return base_plan, enriched_plan, planning

    base_plans, enriched_plans, planning = jax.vmap(_plan_row)(
        jax.lax.stop_gradient(rows_A), jax.lax.stop_gradient(norm0))

    def _row(args):
        table, norm, base_plan, enriched_plan = args
        return _aap.empirical_enrichment_marginalize(
            table, norm, base_plan, enriched_plan, x_min, x_max,
            base_order=int(config.base_order),
            base_check_order=int(config.base_check_order),
            enriched_order=int(config.enriched_order),
            enriched_check_order=int(config.enriched_check_order),
            convergence_tol_nats=float(config.convergence_tol_nats),
            time_guard=guard,
            time_guard_tol_nats=float(config.time_guard_tol_nats),
            log_normalization=float(local_log_normalization),
            time_outside_tol_nats=float(config.time_outside_tol_nats),
            total_value_error_budget_nats=float(
                config.total_value_error_budget_nats))

    n_rows = int(rows_A.shape[0])
    xs = (rows_A, norm0, base_plans, enriched_plans)
    if batch_rows == 1 or n_rows == 1:
        selected, accepted, ledger = jax.lax.map(_row, xs)
        batch_executed = 1
    elif batch_rows == 0 or batch_rows >= n_rows:
        selected, accepted, ledger = jax.vmap(_row)(xs)
        batch_executed = n_rows
    else:
        selected, accepted, ledger = jax.lax.map(
            _row, xs, batch_size=batch_rows)
        batch_executed = batch_rows

    usable = accepted & norm_time_invariant & tables_finite
    lnL = jnp.where(usable, selected, jnp.nan)
    ledger = dict(ledger)
    ledger.update(planning)
    ledger.update(
        bounded_cost=jnp.full((n_rows,), True, dtype=bool),
        dense_reserve_available=jnp.full((n_rows,), False, dtype=bool),
        fixed_plan_autodiff_only=jnp.full((n_rows,), True, dtype=bool),
        derivative_warrant_certified=jnp.full((n_rows,), False, dtype=bool),
        batch_rows_requested=jnp.full(
            (n_rows,), batch_rows, dtype=jnp.int32),
        batch_rows_executed=jnp.full(
            (n_rows,), batch_executed, dtype=jnp.int32),
        max_starts_cap=jnp.full(
            (n_rows,), int(config.base_max_starts), dtype=jnp.int32),
        max_time_nodes_cap=jnp.full(
            (n_rows,), int(config.max_time_nodes), dtype=jnp.int32),
        max_modes_cap=jnp.full(
            (n_rows,), int(config.enriched_max_modes), dtype=jnp.int32),
        decline_norm_time_variation=~norm_time_invariant,
        decline_input_nonfinite=~tables_finite,
        norm_time_invariant=norm_time_invariant,
        norm_time_deviation=norm_dev,
        tables_finite=tables_finite,
        input_nonfinite=~tables_finite,
        usable=usable,
        selected_value=selected,
        lnL=lnL)
    if return_ledger:
        return lnL, ledger
    return lnL


def summarize_policy_ledger(ledger):
    """Host-side counts for the run record.  ``ledger`` leaves are ``(S,)``."""
    def _count(key):
        return int(np.sum(np.asarray(ledger[key], dtype=bool)))
    n = int(np.asarray(ledger["usable"]).shape[0])
    out = dict(
        rows=n,
        accepted_local=_count("accepted_local"),
        reserve_executed=_count("reserve_executed"),
        reserve_warranted=_count("selected_value_is_warranted_reserve"),
        usable=_count("usable"),
        unusable=n - _count("usable"),
        norm_time_invariant=_count("norm_time_invariant"),
        tables_finite=_count("tables_finite"),
        reconciles=_count("reconciles"),
        disposition_reconciles=_count("disposition_reconciles"),
    )
    declines = {}
    for key in _DECLINE_KEYS:
        if key in ledger:
            c = _count(key)
            if c:
                declines[key] = c
    out["declines"] = declines
    if "reserve_escalations" in ledger:
        out["reserve_escalations"] = int(np.sum(
            np.asarray(ledger["reserve_escalations"])))
    if "reserve_batch_rows_executed" in ledger:
        ex = np.asarray(ledger["reserve_batch_rows_executed"])
        req = np.asarray(ledger["reserve_batch_rows_requested"])
        out["reserve_batch_rows"] = int(ex[0]) if ex.size else 0
        out["reserve_batch_rows_requested"] = int(req[0]) if req.size else 0
        out["reserve_batch_execution_sequential"] = bool(np.all(
            np.asarray(ledger["reserve_batch_execution_sequential"],
                       dtype=bool)))
    if "lnL" in ledger:
        out["nan_rows"] = int(np.sum(~np.isfinite(
            np.asarray(ledger["lnL"], dtype=float))))
    score = np.asarray(ledger["empirical_value_error_score_nats"], dtype=float)
    finite = score[np.isfinite(score)]
    out["max_local_error_score_nats"] = (
        float(np.max(finite)) if finite.size else float("nan"))
    return out
