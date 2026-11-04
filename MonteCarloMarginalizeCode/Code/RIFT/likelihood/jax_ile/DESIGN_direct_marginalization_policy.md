# The cross-axis direct-marginalization policy

Module `direct_marginalization_policy.py`; driver flag
`--direct-marginalization-policy {off,auto}`, default `off`.
Companion to `DESIGN_direct_marginalization_planner.md` (the generic
error/resource planner, still unwired) and to PR #268, whose controller this
policy runs.

## Status

Opt-in, value-only. Wired on 2026-09-07 for `--mode flowmc-phipsimarg`.
Nothing selects it by default. Gradient parity is not validated; see the
gate below.

## Why not widen `--angle-marg-scheme auto`

That selector chooses between `exact` and `laplace` on one amplitude
crossover, controls angles only, and excludes both peak-local kernels by a
pinned test. The composite method owns four axes at once and decides per
evaluation from diagnostics. It is a different object and gets a different
flag.

## What one evaluation does

For a batch of extrinsic rows `(ra, dec, incl)`:

1. `anglemarg.angle_coefficient_tables(..., guard=G)` builds the exact
   coefficient tables once with `G` primitive-only support samples at each
   end. The norm table is collapsed per row; its deviation over time is
   recorded and a row whose norm moves with time is marked unusable.
2. Per row, under `vmap`: `rank_joint_starts_from_uvq_device` at angular
   oversample 1 (base) and 2 (extra), then
   `make_all_axis_mode_plan_pair_device` refines both in one shared pass and
   freezes two nested plans. The plans are placed under `stop_gradient`.
3. `empirical_enrichment_with_exact_reserve_sequential_batch` applies the
   local gate per row and, on a decline only, executes the reserve.
4. The reserve is the exact-angle coefficient integral on a time rule
   refined `reserve_time_refine` times (default 4) over the native cadence.
   It is warranted by the two-guard comparison (`G` against `G/2`) and by
   the half-refined check rule evaluated in the same declined branch. The
   warrant is a convergence statement about the refined rules. The native
   Simpson rule is not the check rule: on a peak narrower than a sample it
   is the unconverged one, and its error is what the refinement removes.
   The refined rules are trapezoid, not Simpson: Simpson aliases at half the
   node spacing on a sub-sample peak (0.03 to 1.8 nat at refine 4 for peaks
   of 0.05 to 0.2 samples), the trapezoid rule converges exponentially.
   The wiring test measures the native rule's error on its analytic fixture.
5. The selected value and the ledger come back per row.

Acceptance diagnostics, all required for the local branch:

| diagnostic | ledger key |
|---|---|
| norm table time-independent | `norm_time_invariant` |
| no capacity truncation, base and enriched | `base_capacity_ok`, `enriched_capacity_ok` |
| finite stationary modes | `base_and_enriched_values_finite`, `decline_no_modes` |
| no competitive start pinned to the time or distance boundary | `boundary_maximum_ok`, `decline_boundary_maximum` |
| valid, nested local geometry | `geometry_nesting_ok`, `decline_geometry` |
| base/enriched mode agreement | `mode_nesting_ok` |
| nested quadrature convergence | `decline_quadrature`, `decline_enrichment` |
| two-guard time agreement | `decline_time_reconstruction` |
| omitted-time mass under budget | `time_omitted_mass_ok` |
| total empirical error score under budget | `value_error_budget_ok` |

Reserve warrant: `reserve_time_guard_validated`,
`reserve_time_resolution_validated`, `reserve_time_error_budget_ok`, combined
in `reserve_time_warranted`. A reserve that fails its warrant is escalated:
the rule is doubled and re-checked against its own half, up to
`reserve_time_refine_max` (default 32). A row still unwarranted, or whose
norm table varies with time, is `usable=False` and its likelihood is `nan`.
The finite diagnostic stays in the ledger under `selected_value` and never
reaches the sampler. The driver raises on the first `nan` it evaluates and
refuses to publish samples or evidence that contain one (external review of
PR #278, P1). A MALA step onto a `nan` target is rejected, so chains do not
carry such rows either.

No SNR threshold appears anywhere. The transitions reported in the paper
(reserve at 40 and 80, local at 160 and 320) emerge from these diagnostics.

## Operating point

`PolicyConfig` defaults follow the configuration that accepted on production
tables at rho 163 and 326 (RIFT_roboto_paper
`analyses/va_sequence_20260902/RESULTS_20260907_aap268_ladder.md`): angular
oversample 2 and 4, 16 modes, radius 6, guard 128, 14 refine iterations.
PR #268's test values (oversample 1 and 2, 4 and 8 modes) overflowed capacity
on synthetic carrier tables and were never a measured production point.

The guard is data, not a knob: the gather returns nonfinite samples past the
stored buffer with no error. The wrapper probes a coarse sky grid at
construction and refuses a guard the buffer cannot supply, and every row
carries `tables_finite`; a nonfinite row is `input_nonfinite`, `nan`, and
never a method decline.

## Gradient memory

Measured with XLA's compile-time memory analysis of `value_and_grad` on a
rho 163 production row (614-sample window, guard 128, 83200 dense angles,
16 GH distance nodes):

| stage | temp memory |
|---|---|
| production exact scheme | 2.6 GiB |
| tables, planning, local gate (base and enriched) | under 0.02 GiB |
| one reserve tier at refine 4, inside `lax.cond`, dense chunk 8 | 10.5 GiB |
| full policy, tiers to refine 32, dense chunk 8 | 84.9 GiB (158 before the branches were rematerialized) |
| reserve at refine 2 / 4 / 8, dense chunk 8 | 5.3 / 10.5 / 21.0 GiB |
| reserve at refine 2 / 4 / 8, dense chunk 64 | 0.73 / 1.46 / 2.92 GiB |

The local branch is essentially free to differentiate. The cost is the dense
reserve's reverse pass: one carry per dense-angle scan step, so it is
proportional to the refined time nodes and inversely to the dense chunk, and
`lax.cond` reserves memory for the largest tier whether or not it runs. The
policy's reserve chunk is therefore 64 (the kernel default is 8), which
brings the full policy at ceiling 32 to about 12 GiB per evaluated row.
The controller's branches and every tier are under `jax.checkpoint` and the
planning inputs are under `stop_gradient` (planning is control data). The
escalation ceiling is exposed as
`--direct-marginalization-reserve-time-refine-max` because it bounds gradient
memory; the value path does not depend on it below the ceiling. A cropped
reserve over the plans' certified time cover would cut both cost and memory
by the ratio of window to cover and is the follow-up.

## Measures

The local branch integrates `x**-4 dx  dt_sample  dphi  du`. The reserve and
the exact scheme average both angles, weight distance by the normalized
`log_w_grid` (fixed grid) or the normalized volumetric measure
(`JAX_ILE_DISTMARG_GH`), and integrate time in seconds by Simpson.
`policy_log_normalization` derives the conversion (the wiring test checks
it end to end against an independent fine-time reference on both distance
paths):

| term | constant |
|---|---|
| angles | `-2 log(2 pi)` |
| time | `log(deltaT) + log(sum(w_t) / ((npts-1) deltaT))` |
| distance, fixed grid | `3 log(Dref) - log(sum_i d_i^2 dd)` with `dd` read off the grid |
| distance, GH | `log 3 - log(x_min^-3 - x_max^-3)` |

## Refusals

The policy refuses, with a message, any of: a resolved angle scheme other
than `exact`; a time rule other than `simpson`; a distance prior other than
volumetric; a distance grid other than uniform-in-d; a `time_guard` below 2;
a reserve refinement that is not an even integer of at least 2; a request
for `lnL(t)`. The driver refuses at parse time a policy request in any mode
other than `flowmc-phipsimarg`, and any of the three policy knobs when the
policy is off (external review of PR #278, P1). Refusal rather than silence
is the standing rule on this arm.

## Cost

<!-- LENGTH: this file is over the 1500-word DESIGN budget in the plain-prose
gate.  RO authorised the length on 2026-09-08: the cost record matters more
than the budget here.  Do not trim it back to pass the gate. -->


Planning is vectorized. Accepted rows pay fixed local work per retained mode;
declined rows pay three exact reserve evaluations (refined rule at two guards,
plus the half-refined check). The sampler's angle-scheme chunk cap applies.

`PolicyConfig.reserve_batch_rows` sets how many rows run under one `vmap`;
`--direct-marginalization-batch-rows` exposes it. At 1 the rows run one at a
time under `lax.map`, the graph PR #268 measured. Above 1 the tier-escalation
`lax.cond` becomes a `select`, so every reserve tier runs for every row. Nothing else changes:
`test_row_batch_size_changes_cost_not_values_decisions_or_gradients` requires
`lnL` bitwise equal, every ledger key and summary count equal, and the
gradient equal to one ulp, over the full-batch, whole-multiple and remainder
paths.

### Device workspace

XLA buffer assignment, ladder-2 tables, NVIDIA RTX PRO 4000 Blackwell,
jax 0.9.2, `--n-phi 32 --n-psi 8 --distance-grid-points 256`,
`reserve_time_refine_max` 32:

| `reserve_batch_rows` | temp GiB at rho 40.8 | temp GiB at rho 652.3 |
|---|---|---|
| 1  |  0.425 |  0.432 |
| 2  |  0.812 |  0.822 |
| 4  |  1.588 |  1.592 |
| 8  |  3.132 |  3.132 |
| 16 |  6.212 |  6.212 |
| 32 | 12.371 | 12.371 |
| 64 | 24.692 | 24.692 |

The fit is `0.046 + 0.385 B` GiB, maximum residual 6 MiB, at the grid sizes in
the caption. The rung does not enter. Measured device use at B=32 was 23.3 GiB
against the 12.4 GiB analysis figure, so buffer assignment understates the card
about twofold: a 24 GiB card holds B=32 and not B=64.

The tier count multiplies it. At B=8, `reserve_time_refine_max` 32 costs
3.114 GiB and 138 s to compile; at 4 (one tier) the same batch costs
0.415 GiB and 29 s. A batched row pays every tier.

### Throughput

Batching is a REGRESSION, not a saving. It converts three `lax.cond`s to
selects, not the one the knob's first version named:

| site | becomes, under vmap |
|---|---|
| tier escalation in `_row` | every reserve tier runs for every row |
| `all_axis_peaklocal.py:2504` accept/reserve | every ACCEPTED row also runs the dense reserve |
| `all_axis_peaklocal.py:1760` per-mode live | every dead mode slot runs a 4-D quadrature |

So the penalty scales with the locally accepted fraction and the tier count.
Measured on the wiring fixture, 2 of 6 rows accepting, second timed call:

| tiers | B=1 | B=2 | B=6 |
|---|---|---|---|
| 1 (`reserve_time_refine_max` 4) | 2.50 s/row | 3.28 (+31%) | 4.08 (+63%) |
| 2 (`reserve_time_refine_max` 8) | 6.49 s/row | | 7.81 (+20%) |

An earlier production-table point read 96.3 s/row at B=1 against 99.9 at B=8
and was reported as cost-neutral. It ran at `reserve_time_refine_max` 4, where
`tiers` has length one and the escalation cond is absent, with `accepted_local`
0 of 8, so the accept/reserve cond took its cheap branch everywhere. Both
penalties were inert, making it the best case. The docstring at
`all_axis_peaklocal.py:2344` is false above B=1.

`reserve_batch_rows` defaults to 1, kept for the equivalence it pins rather
than for a saving. A single row takes the sequential path whatever is
requested: `_scalar` evaluates one row, so `value_and_grad` and `hessian`
would otherwise pay every select with nothing to amortize.

## Peak-local time reserve

The reserve schemes are pairs (angular kernel, time rule):

| scheme | angular kernel | time rule |
|---|---|---|
| exact | exact angles | whole window, refine 4 escalating to 32 |
| laplace | psi-Laplace | whole window |
| peaklocal | psi-Laplace | peak-local, fixed count |
| peaklocal-exact (config only) | exact angles | peak-local, fixed count |

The whole-window rule's node count grows with rho: the peak of exp(lnL(t))
has width sigma_t = 1/(2 pi rho sigma_f), and every node pays the angular
kernel. At rung 41 the refine-4 rule already misses its own 1e-3 warrant on
two of three rows (0.0018, 0.0040, 0.0114 nat, ladder controller
2026-09-08), and the exact kernel costs 18.6 / 72.6 / 290 / 1278 s per
evaluation at rungs 41 / 82 / 163 / 326 on 2453 nodes.

### The rule

One lattice per row.  The scan puts `scan_refine` nodes per native sample
across the window (default 2).  The fine lattice divides each scan cell by an
integer `m`, chosen so that its spacing is at most `sigma_t / 3` for the
predicted width; every node of the rule is a point of that lattice.  Around
each of the row's time maxima sits a block of `n_fine` consecutive fine nodes
(default 73, so the block spans 24 predicted sigma).  Overlapping blocks and
scan nodes inside a block coincide and carry no weight; the only
spacing changes are at block ends, where the mass is negligible.  Mixing two
lattices instead cost 0.02 nat on a 0.75-sample peak: the trapezoid rule is
spectrally accurate only on uniform spacing.

The maxima come from the primitive, not from the local branch.  The field
`max_x (x A(t, phi, u) - x^2 B(phi, u) / 2)` is evaluated on a search grid of
8 nodes per sample over an 8 x 8 angular lattice; the four highest local
maxima are polished by Newton steps on the fixed-angle field with the
reflected spectrum's exact derivatives.  On a synthetic peak 0.16 samples wide
the local branch's plan carried no live mode at all, so a reserve leaning on it
inherits that decline.  The plan's centre and width are reported beside the
locator's as a cross-check.

The scan is support-limited too: `reserve_peaklocal_scan_nodes` (65) nodes
on the scan lattice across the hull of the live maxima widened by
`reserve_peaklocal_scan_margin_sigmas` (16) predicted sigma each side, never
the window.  The mass outside the hull is bounded from the locator's own
search profile, summed over the outside search nodes, plus
`reserve_peaklocal_outside_slack_nats` (5) for what lies between nodes.  The
angle- and distance-maximized exponent is an upper bound on the marginal at
each search node.  That bound is charged through the kernel's cropped-cover
warrant (`reserve_time_cover`).  The node
count is `65 + 4 x 73 = 357` whatever the window; a count that grows with the
window is the wrong design (RO, 2026-09-09).

The check rule is the half-refined scan plus every other fine node, on the
same lattice with the same endpoints, so the kernel's structural resolution
warrant, its two-guard comparison and its tail bound apply unchanged.  A
failed warrant doubles the fine lattice at fixed span, at most
`reserve_peaklocal_escalations` times (default 2), and the ledger counts it.
The rule and its check share the block, so a block off the maximum agrees
with itself: measured 0.22 nat, warranted, on a rung-160 row whose block sat
0.17 samples (2.7 sigma) from the maximum.  The kernel therefore carries a
focus certificate (`reserve_time_focus_ok`): the node with the largest
evaluated `lnL(t)` must lie within a quarter of the block span of the block
centre, or the rule is unwarranted.

Under jit, XLA fuses the difference of two bitwise-equal products into a
multiply-add that rounds to -1e-15, so the kernel's node checks compare
slices rather than take `diff >= 0`, and repeated positions are accepted as
non-decreasing.

### Sizing from the prediction

`sigma_t = 1 / (2 pi rho sigma_f)` per row.  `rho` comes from the locator's
profile maximum, `rho^2 = 2 P_max`, which is the row's own `lnL` maximum; the
angular triangle bound over the norm's triangle lower bound is the fallback
and is reported beside it (`rho_bound`).  The bound ran 2.8x over on a
rung-160 row (453 against 157), which narrowed the block span to 4 sigma of
the true peak; the profile value does not.  `sigma_f` is the two-sided
rms frequency of the stored Q (`q_effective_bandwidth_hz`).  The ladder record
measured the same quantity at 0.01557 cycles per sample on the ladder-2 tables
at every rung.  The row's own table spectrum is the fallback and is printed
beside it.  The raw moment is deliberate.  The phi-marginalized field of a
(2, +-2) signal with two polarization weights is `|alpha kappa + beta kappa*|`,
which keeps carrier-scale sub-peaks unless the signal is circular.  The raw
moment bounds the curvature of anything the band-limited primitive can make at
that amplitude.  The central moment gives the face-on envelope, the widest
case.  A factor sqrt(2) was proposed for the linear limit from the curvature
of `|zeta|^2` at its maximum; it is not there.  The integrand is
`exp(lnL)` with `lnL = (rho^2 / 2) |zeta_hat|^2`, and for `cos^2(omega t)`
the curvature `2 omega^2` and the prefactor `rho^2 / 2` multiply to
`rho^2 omega^2`, so `sigma_t = 1 / (rho omega) = 1 / (2 pi rho f_c)`, the
raw moment, with no extra factor.  The carrier fixture agrees to 5 percent
(located 0.0585 samples against predicted 0.0616 at rho 12.65), and the
reserve-scheme session's fit of the log-integrand's curvature on a linearly
polarized carrier gave measured / predicted of 1.0008, 1.0000, 0.9999 and
0.9999 at rho 12.65, 40.77, 163.08 and 652.31 (their 59b48e3e).  The lattice
is sized from the narrower of the prediction and the located curvature width
at each maximum, so a row narrower than predicted sets its own spacing.  The
located and plan widths are reported.  `prediction_consistent` flags a
located width outside `[0.5, 4]` times the prediction.

The ledger prints, per row: `rho_pred`, both `sigma_f`, the predicted,
located, used and plan widths, the fine spacing and block span, the live
block count and first centre, the escalation count, and the consistency flag.
`predict_time_rule` gives the pair selector the same numbers from `rho` and
`sigma_f` before any row runs.

### Measured

Rung 40.77: `likedata_snr40.pkl`, rho 40.8, 614-sample window, 8 rows in the
ladder's truth-centred sky box, one RTX PRO 4000 Blackwell.  `peaklocal-exact`
is compared with the shipped `exact` scheme so only the time rule differs.
Policy defaults after #301, whole-window scan, `reserve_batch_rows` 1.  The
exact column is the earlier run of the same rows.

| row | exact reserve | peak-local minus exact, nat | nodes exact / peak-local | escalations exact / peak-local | wall s exact / peak-local | live blocks |
|---|---|---|---|---|---|---|
| 0 | 655.431925 | -2.3e-13 | 4905 / 1519 | 1 / 0 | 187 / 155 | 2 |
| 1 | accepted locally (575.772711) | | | | | |
| 2 | accepted locally (656.379930) | | | | | |
| 3 | 604.921535 | +7.5e-11 | 4905 / 1807 | 1 / 1 | 130 / 139 | 1 |
| 4 | accepted locally (433.744903) | | | | | |
| 5 | accepted locally (626.523450) | | | | | |
| 6 | accepted locally (591.821831) | | | | | |
| 7 | 647.095757 | +0.0e+00 | 2453 / 1807 | 0 / 1 | 44 / 92 | 2 |

Row 0 of each scheme carries the compile.  Rows 1, 2 and 6 accept locally
under the #301 capacities and show the local branch's value.  The two
peak-local escalations (rows 3 and 7) are the prediction one doubling short
on those rows: the first tier's check missed 1e-3 and the second met it.

Predicted against located width, same rows (samples; the locator's width is
the envelope's curvature at the maximum, the plan's is the local branch's
Newton width):

| row | rho (located) | sigma_t predicted | sigma_t located | sigma_t plan | located / predicted | escalations |
|---|---|---|---|---|---|---|
| 0 | 36.7 | 0.279 | 0.332 | 0.309 | 1.19 | 0 |
| 3 | 35.4 | 0.289 | 0.181 | 0.294 | 0.63 | 1 |
| 7 | 36.6 | 0.280 | 0.305 | 0.306 | 1.09 | 1 |

The stored Q's bandwidth is 0.01556 cycles per sample; the rows' own table
spectra give 0.0135 to 0.0150.  The located width agrees with the plan's
within 40 percent and the prediction sits at or below the located width.

Rung 163.08: `likedata_snr160.pkl`, rho 163.1, same window, rows 0 to 3 of the
seed-7 draw, one Blackwell.  Under the #301 capacities every row accepts locally
(issue #308 has the values), so the decline is forced with `max_modes` 1.  The
peak-local reserve has its support-limited scan here: 65 scan nodes plus
4 x 73 block nodes, 357 in all.  It is compared with the whole-window exact
reserve on the rows where that exists.  Row 1's is refine 32 with its refine-16
check agreeing to 6e-10; the exact rows 0, 2 and 3 were stopped on 2026-09-09 (RO:
the exact kernel is not the one that drops out at 652) and are not run:

| row | exact reserve | peak-local | difference, nat | nodes exact / peak-local | escalations exact / peak-local | wall s exact / peak-local | sigma_t predicted | located | plan | rho (located) | resolution error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | not run | 13165.892582 |  |  / 357 |  / 0 |  / 186 | 0.0629 | 0.0619 | 0.0619 | 162.5 | 1.9e-10 |
| 1 | 12335.902574 | 12335.902574 | +3.2e-07 | 19617 / 357 | 3 / 0 | 9805 / 107 | 0.0650 | 0.0649 | 0.0649 | 157.4 | 9.1e-12 |
| 2 | not run | 10083.512460 |  |  / 357 |  / 0 |  / 107 | 0.0719 | 0.0700 | 0.0700 | 142.3 | 8.9e-07 |
| 3 | not run | 8474.729892 |  |  / 357 |  / 0 |  / 108 | 0.0786 | 0.0782 | 0.0782 | 130.2 | 0.0e+00 |

Row 1 agrees with the converged whole-window value to every printed digit at
2 percent of its node count and 1 percent of its wall time.  No row escalated:
the predicted widths sit within 3 percent of the located ones, and the located
`rho` within 20 percent of the network 163.  The scan hull was 2.1 samples wide
(the maxima +-16 sigma) against the 614-sample window.

### Locator search sizing (rung 652, 2026-09-09)

The first `peaklocal` rows at rung 652 (`likedata_snr640.pkl`, S=8 draw, guard
128, 16 distance nodes, decline forced) refused 3 of 8 rows on the focus
certificate: the rule's argmax sat 0.18 to 0.33 samples from the block centre
(spans 0.22 to 0.37).  On row 0 a 1/256-sample lattice of the psi-Laplace
kernel peaks at 307.109 (lnL 212147.7, half-maximum width 0.027 samples); the
locator had centred the block at 307.334, 107 nat lower.  A sweep of the
locator on that row:

| search refine | phi nodes | Newton steps / clip, rad | centre | value | note |
|---|---|---|---|---|---|
| 8 | 64 | 3 / 0.1 | 307.334 | 211532 | shipped #304 |
| 8 | 64 | 8 / 1.0 | 307.334 | 212067 | angles reach, time cell wrong |
| 64 | 64 | 3 / 0.1 | 306.559 | 211536 | |
| 64 | 64 | 8 / 1.0 | 307.464 | 211908 | |
| 128 | 64 | 8 / 1.0 | 307.482 | 211880 | |

Refining the time search does not help: the search grid maximizes the angles on
a 64-node phi lattice, whose ripple is about `rho^2 (pi / n_phi)^2` = 1000 nat at
rho 652 against a 25-nat change of the profile across one search cell
(`(rho^2 / 2)(0.125 / tau)^2` with the envelope width tau about 7.6 samples), so
the search maximum lands on whichever cell the ripple favours, up to 0.4
samples away, beyond the polish's reach of 1.33 cells.  The Newton polish of
(phi, u) was clipped to 0.1 rad per step for 3 steps against a lattice offset
of up to 0.4 rad, 500 nat low on the same row.

Sizing, from the amplitude: the phi count must hold the ripple under the
per-cell change, so `n_phi >= pi rho` for 1 nat; it is a static shape, so the
policy carries `reserve_peaklocal_search_phi_nodes` = 4096 (under 1 nat up to
rho 1300; the grid is `(t, phi, u)` = 4905 x 4096 x 8 doubles, 1.3 GB, a fraction
of a second on the GPU) and `reserve_peaklocal_newton_steps` = 8 within
`reserve_peaklocal_newton_step_max` = 1 rad.  Test: a carrier at rho 632 with
tau 8 is located within tau / rho of its centre at the profile's true maximum.

Rung 652 with the sizing, S=8 draw, `peaklocal` (psi-Laplace), decline forced:

| row | reserve value | nodes | escalations | s per row | sigma_t predicted | located | rho located | focus offset, samples | resolution error, nat | warranted |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 212136.145450 | 357 | 0 | 152 | 0.0157 | 0.0154 | 651.4 | 0.001 | 2.9e-11 | yes |
| 1 | 211785.826592 | 357 | 0 | 28 | 0.0157 | 0.0154 | 650.9 | 0.001 | 2.9e-11 | yes |
| 2 | 205203.742743 | 357 | 0 | 28 | 0.0160 | 0.0157 | 640.7 | 0.000 | 2.9e-11 | yes |
| 3 | 211361.706073 | 357 | 0 | 28 | 0.0157 | 0.0154 | 650.2 | 0.002 | 2.9e-11 | yes |
| 4 | 139434.603116 | 357 | 0 | 28 | 0.0194 | 0.0190 | 528.2 | 0.001 | 0.0e+00 | yes |
| 5 | 145870.675689 | 357 | 0 | 28 | 0.0189 | 0.0182 | 540.2 | 0.002 | 0.0e+00 | yes |
| 6 | 203151.295219 | 357 | 0 | 28 | 0.0160 | 0.0157 | 637.5 | 0.001 | 2.9e-11 | yes |
| 7 | 141631.499287 | 357 | 0 | 28 | 0.0192 | 0.0185 | 532.3 | 0.001 | 0.0e+00 | yes |

Before the sizing rows 0, 1 and 3 were refused by the focus certificate
(offsets 0.334, 0.183, 0.183 samples after two escalations, 1221 nodes,
129 to 250 s); the other five rows keep their values to all printed digits.
Rung 163: all eight rows unchanged at 9 s per row.

## Gate before this can be a default

PR #268 warrants scalar values. Differentiating the composite differentiates
a truncated fixed-plan integral, and the JAX sampler's hill climbing and
MALA/HMC steps consume those gradients. Required before any default change:
value and gradient parity through the SNR and higher-mode ladder, on
production tables, recorded in the paper repository
(`development/OPEN_jax_direct_marginalization_policy.md`).

## Known adversarial items

- Acceptance was not completeness at a support boundary. On the 32-sample
  synthetic window the exact lnL(t) peaks at the first sample; the planner's
  boundary starts were rejected as non-stationary, the interior modes were
  accepted with every diagnostic passing, and the value was 22.86 nat
  against an exact 45.5. The plan now records a live start pinned to the
  time or distance boundary within 30 nat of the best value, and the gate
  declines on it (`decline_boundary_maximum`). A one-sided local region for
  boundary maxima is the eventual fix; the decline is the fail-closed one.
- Sampler cost (wiring review): flowMC vmaps the scalar AD target over
  chains, and under `vmap` a `lax.cond` with a batched predicate lowers to
  `select_n`, so every chain executes the local branch and every reserve
  tier whatever its own disposition. The `lax.map` batch is also sequential
  in rows, so an 8000-row pilot is hours. Neither is a correctness problem;
  both make the policy impractical as the sampler's target until a
  host-compacted path exists (vmap the local gate, run the reserve on the
  declined subset).
Items 1 and 2 below come from the adversarial review of PR #268 through this
wiring (2026-09-07) and are verified on synthetic tables only. They are the
first questions for the production-table ladder.

- On synthetic 22-only carrier tables the base portfolio at angular
  oversample 1 overflows `max_starts=32`: the 9-point phi lattice's
  max-over-angles time profile ripples with the rotating carrier phase and
  produces spurious time peaks, so every row declines on capacity and the
  local branch never runs. Oversample 2 fits but the same tables then
  decline on nested quadrature (13 vs 19 nodes, 8e-3 nat). PR #268's real
  SNR-160 capture reports 4 base candidates with no overflow, so the
  synthetic result does not transfer directly; the ladder must measure the
  acceptance rate on production tables before the paper's "local at 160 and
  320" is quoted from this code. Suggested fix if it does transfer: rank
  time peaks from the triangle envelope the time-cover step already
  computes, not from the lattice profile.
- Sub-sample peaks: at 150 Hz and 4096 Hz the time peak is about 4.35/rho
  samples wide, so above rho of a few tens the reserve needs refinement well
  beyond 4. The escalation ceiling and the trapezoid rules address the
  warrant; the cost (three dense evaluations per tier) is the ladder's to
  measure. The norm lower bound also loosens with inclination (0.46 of the
  norm edge-on), which can exhaust the 64 retained time nodes.
- Capacity at higher harmonic order: random m_max=4 tables show 5 to 12
  angular lattice maxima per time node against `max_modes=4`, and the u
  lattice does not grow with oversample, so enrichment refines phi only.
- The error score double-charged common-mode terms. Base and enriched plans
  measure the same quadrature, guard, and omitted-time discrepancies, and
  the score summed both. On the analytic fixture at 10x amplitude a value
  correct to 1.5e-4 nat was refused at a score of 1.06e-3 (two copies of one
  5.27e-4 guard discrepancy). Fixed in PR #278: each pair is charged once as
  its maximum; the per-plan terms stay in the ledger.
- A distance peak below the prior's support pins every optimizer lane to the
  boundary with an identical gradient norm growing like rho^2. PR #268
  declines such a row (`decline_no_modes`); the signature matches the one
  the PR #270 ladder reported for its enriched tier, which points at a
  distance-support problem in that harness rather than a refinement defect.
- The local box radius is a common-mode term. Base and enriched plans use
  the same whitened radius, so mass outside the box is invisible to the
  enrichment gate and to the error score. At radius 3 the accepted value on
  the wiring test's analytic fixture was 0.015 nat below an independent
  fine-time reference while every diagnostic passed. The default is the
  library's 6; the wiring test pins the accepted value against the external
  reference at 2e-3 nat, which the gate alone would not have caught.
- PR #270's host controller (`multipeak_planner`) declined on every rung of a
  production ladder because its enriched tier never refined (identical
  gradient norms across starts). This policy does not use that controller.
  The same failure class, degenerate or unrefined enriched starts, must be
  probed on this device pipeline with per-start gradient norms on production
  tables before the ladder result is trusted.
- The audit ledger is evaluated on a subsample of exported rows after
  sampling. It describes the exported cloud, not every evaluation the
  sampler made. Unwarranted rows are not a labelling matter: they are `nan`
  and stop the run.

## Choosing the (local, reserve) pair from analysis

RO, 2026-09-08: rely on analysis and the known physics to pick the pair, rather
than try-then-decline-then-refine. `predict_reserve_pair` computes, from the
precomputed inputs and before any row is evaluated:

| quantity | source | decides |
|---|---|---|
| network SNR | the driver's own response-derived guess | amplitude |
| `sigma_f` | second moment of the stored Q's spectrum | peak width |
| `A = rho^2/2` vs `ANGLE_MARG_CROSSOVER_AMPLITUDE` (450) | measured crossover in `anglemarg` | exact or laplace angles |
| `sigma_t = 1/(2 pi rho sigma_f)` vs the local cover budget | `max_time_nodes` | can the LOCAL branch hold the peak |
| the same width vs the reserve's node budget | `reserve_time_refine_max` | can the whole-window reserve resolve it |

The verdict and its reasons are printed before sampling. When no implemented
reserve is adequate the run REFUSES; it does not fall back, because a fallback
to whole-window refinement carries the rows in a method nobody chose.

### Which bandwidth, and why it is physics not convention

The reserve marginalizes phi exactly, so the field in time is `|zeta|` with
`zeta = alpha kappa + beta kappa*`. Face-on the carrier term vanishes and the
peak is the ENVELOPE; linearly polarized the envelope is modulated at the
carrier and each sub-peak is far narrower. Measured on a carrier fixture
(f_c = 200 Hz):

| polarization | measured peak | matches |
|---|---|---|
| circular | 11.3 Hz equivalent | central moment, 5.6 Hz |
| linear | 309.6 Hz equivalent | raw moment, 200.1 Hz |

So RAW is the narrowest peak the primitive can make and CENTRAL the widest. A
rule that must not under-resolve sizes on the raw one. Two errors were made
here and are recorded so they are not repeated: sizing on the central moment
(the Cramer-Rao bound is about an ESTIMATOR's variance, not how sharply the
INTEGRAND varies), and a claimed sqrt(2) correction that came from reading the
curvature of `|zeta|^2` without its `rho^2/2` prefactor. Against the actual
log-integrand, raw is exact: measured/predicted 1.0008, 1.0000, 0.9999, 0.9999
at rho 12.65, 40.77, 163.08, 652.31.

### The node budget is PROVISIONAL and known to be the wrong law

The budget is points-per-sigma, i.e. an ALGEBRAIC convergence model. Measured at
rho 40.77 on 64 rows:

| refine | nodes | warrant error |
|---|---|---|
| 4 | 2453 | 1.8e-03 .. 1.14e-02 |
| 8 | 4905 | 5e-11 .. 7.8e-09 |

Read against a tolerance, which has since moved: refine 4 fails the 1e-3 that
was shipped when this was measured, and straddles the 1e-2 RIFT PR #301 adopted
(only the top of the range exceeds it). #301 measured the other side of the same
quantity at the same rung — escalations 2 -> 0 and 321 s -> 147 s going from
1e-3 to 1e-2, lnL moving 4.8e-12 — which is what this range predicts. Two
measurements of one effect, taken independently; neither confirms the other's
method, and together they say the escalation at this rung was being driven by
the tolerance rather than by the rule.

Doubling improved the error by ~1e6 where an algebraic rule gives 4. That is the
trapezoid rule on a BAND-LIMITED reconstruction: spectrally accurate once the
band is resolved, `exp(-c R)` not `R^-2`. The 4905 nodes that succeeded are 67%
of what the budget demands at that rung and land five orders INSIDE tolerance.

Consequences, and they are limits on what may be claimed:

* A refusal produced by this budget means UNPROVEN, not shown inadequate.
* No statement about WHERE the whole-window reserve stops being adequate
  follows from it. Such a claim was made and withdrawn twice, on two different
  mechanisms; it is not restated here.
* The replacement is a band-resolution criterion fitted to a MEASURED
  convergence law. The deciding test is rung 163.08 at refine 4, 8 and 16 --
  three points, because two fit either law.

The warrant is what certifies a row. This budget only predicts which method to
reach for, and it must not be hardened into a threshold anyone tunes against.

## The roster: which reserves this run may choose from

`predict_reserve_pair` takes `available=`. That argument is not a preference
list. It says which schemes the **data and the distance quadrature** can support
at all, and it is computed before the analysis runs.

| scheme | on the roster when | why |
|---|---|---|
| `exact` | always | what the composite dispatches (`empirical_enrichment_with_exact_reserve`) |
| `laplace` | per-sample adaptive distance quadrature is ON **and** `gh_laplace_supported_for_data` holds | the placement is derived from A0 == 0 / B1 == 0 |
| `peaklocal` | never, today | RIFT PR #304 |

The laplace conditions are separate and both necessary. On a **static** distance
grid the laplace reserve is measured at 43.2 nats at rho 163 — that is the
grid's cost, not the scheme's, and it is why the reserve may not use it there.
The loguniform static grid is not admitted either: it is sized from the angle
amplitude and may well be adequate, but nothing has measured it.

Two rules follow, both of which the code got wrong first:

- The roster is checked on the **angular** branch, not only where the selector
  chooses `peaklocal`. It was checked only on the branch that could never have
  chosen `peaklocal` anyway, so a run with laplace off the roster still selected
  laplace as soon as A cleared the crossover.
- An explicit `requested=` overrides the **analysis**, not the roster. Forcing a
  scheme whose premise is absent is not an override.

## Declared, executable, and the gap between them

`RESERVE_SCHEME_CHOICES` is what may be named. `RESERVE_SCHEME_EXECUTABLE` is
what the composite dispatches, which is `("exact",)`. `validate_policy_config`
refuses the difference.

Without that refusal, `PolicyConfig(reserve_scheme="laplace")` would be
accepted, printed in the policy line, and computed as exact — a field the
composite never reads is worse than a missing one, because it answers.

The laplace table-level kernel exists (`coefficient_table_distphipsimarg_laplace`,
extracted from the fused laplace path so the two cannot drift). What is missing
is the dispatch: `empirical_enrichment_with_exact_reserve` names its kernel.
Wiring it is a change to `all_axis_peaklocal.py`, which is #304's file.
