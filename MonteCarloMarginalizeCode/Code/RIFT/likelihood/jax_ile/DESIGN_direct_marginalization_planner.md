# Error- and resource-budgeted direct-marginalization planner

## Status and verdict

The policy engine in `direct_marginalization_planner.py` is implemented and
tested, but its RIFT scheme catalog is deliberately **not wired into the JAX
driver or wrapper**.  It is an opt-in planning API, not a new production
`auto` mode.

That boundary is load-bearing.  The current angle, distance, and time schemes
do not yet expose comparable proof-carrying error bounds and measured costs on
a common unit.  Wiring a selector before those adapters exist would require the
planner to invent numbers, or to call a calibrated grid "certified".  The
framework note explicitly rules that out.  The implemented layer can make the
decision once real adapters supply those records; until then, a strict
three-axis request declines honestly.

No existing behavior changes:

- `ANGLE_MARG_DEFAULT` remains `exact`;
- `choose_angle_marg_scheme` is unmodified, including its existing
  amplitude crossover and GH compatibility behavior;
- `angle_marg="auto"`, the time default, and both distance-grid defaults keep
  their old paths;
- no new CLI choice is registered.

The only way to use this work is to import the new module, construct explicit
scheme offers, and call `plan_direct_marginalization` or
`plan_jax_direct_marginalization`.

## Inputs and units

A request has four independent inputs.

1. A positive error ceiling in **absolute marginalized log-likelihood error,
   nats**, for every requested axis.  There is no implicit sharing of a total
   budget: the caller must perform that allocation.
2. A compute ceiling and a peak-memory ceiling.  Both are mandatory.  Compute
   estimates must use one common unit within the request.  Memory is bytes.
3. One or more `SchemeOffer` objects per axis.  Every offer carries its error
   assessment, resource estimate, warrant, prerequisites, incompatibilities,
   and provenance.
4. Concrete capabilities established for this dataset, such as
   `gh-laplace-supported` after `gh_laplace_supported` has checked the actual
   coefficient tables.  Missing capabilities are refusals, not false values to
   route around.

By default the planner sums compute contributions and sums live-memory
contributions.  Direct marginalization nests axes, so a production adapter
should pass a combination-aware `resource_model` when those interactions
matter.  That callback returns the same provenance-carrying `ResourceEstimate`
type and is allowed to conservatively over-count buffers whose lifetimes do not
overlap; it may not assume reuse that it has not measured.  The default is safe
for additive evidence packets and tests, not a claim that nested kernel costs
are separable.  Either form is a hard resource guard, not a wall-time predictor.

## Warrants are not accuracy labels

The warrant union follows `DESIGN_peak_local_framework.md`:

| warrant | can support a certificate? | current use |
|---|---:|---|
| `exact-band-limit` | yes | time exponent reconstruction |
| `exact-trig-degree` | yes | finite angular stationary set |
| `bounded-stationary-set` | yes | support-aware distance candidates |
| `effective-bandwidth-with-margin` | no | amplitude-sized dense angle grids |
| `empirical-calibration` | no | validation envelopes |
| `none` | no | fixed historical grids |

"Can support" is still weaker than "implemented".  `Warrant` therefore has a
separate `certificate_available` field.  `AccuracyAssessment(CERTIFIED, ...)`
is rejected at construction unless both conditions hold.  In particular,
calling the angle scheme `exact` refers to exact coefficient reconstruction;
the subsequent quadrature over `exp(lnL)` is sized from an effective bandwidth
and remains best-effort with a runtime label.  The profile forbids relabeling it
as a proof.

## Current JAX profiles

The module records structural facts already enforced in the shipped call
sites.  It does not attach error or cost numbers to them.

| axis / scheme | recorded warrant | important compatibility fact |
|---|---|---|
| angle `grid` | none | cannot drive the amplitude-sized log-uniform distance grid |
| angle `exact` | effective bandwidth with margin | requires the data-derived amplitude estimate |
| angle `laplace` | effective bandwidth with margin for the complete angle result | GH additionally requires the measured `A0==0/B1==0` identity |
| angle `peak-local` | exact trig degree only on psi; effective bandwidth for the still-dense phi axis | requires an explicit feature warrant and refuses GH |
| distance `uniform` | none | historical fixed grid |
| distance `loguniform` | bounded stationary set, no implemented end-to-end certificate | requires full prior support, an interior peak, and a passing endpoint budget |
| distance `gh` | bounded stationary set, no implemented error certificate | currently the volumetric-prior kernel |
| time `simpson` | none | historical fixed grid |
| time `bandlimited` | exact band limit, no implemented per-request certificate | the nonlinear JAX distance/angle wrappers currently refuse this ordering |

The last row carries both kinds of caveat at once, and is why a production
three-axis error-budgeted plan is not merely waiting for an angle cost table.
The band limit is genuine structure, so the warrant kind could support a
certificate; but the shipped rule derives its refinement factor from a measured
peak width and remeasures it, and reports measured reconstruction errors rather
than a proved bound on the marginalized log likelihood, so no certificate is
advertised and `CERTIFIED` is refused at offer construction.  It is in any case
not compatible on the direct distance/angle-marginalized JAX path, while the
compatible Simpson rule has no per-request error bound either.  No shipped
profile is therefore certificate-bearing today: `cheapest-certified` is
reachable only for a future scheme that implements and validates its bound.

## Decision policy

The planner enumerates the small Cartesian product of per-axis offers and
records, for every combination:

- missing prerequisites and active conflicts;
- missing conditional warrants (for example GH plus Laplace);
- certification status and error-budget excess, per axis;
- compute and memory totals and any resource-budget excess.

Among compatible, affordable combinations certified inside every axis budget,
it chooses the least compute, then least memory, then the smaller normalized
error.  This is the `cheapest-certified` result.

If none exists, it ranks compatible affordable combinations by the worst
per-axis normalized assessed error, then total normalized error and evidence
strength.  Under the default policy this candidate is only `suggested` and the
decision action is `decline`.  `require_selection()` raises
`MarginalizationPlanDeclined`, so code cannot accidentally execute the
suggestion as if it were a selection.

Only `allow_best_effort=True` promotes that candidate to a runnable
`most-accurate-affordable` decision.  Its record says `certified=False` and
separately says whether its numerical assessments meet the requested budgets.
This explicit authority is the only fallback path.

Every result is JSON-ready through `PlanDecision.as_dict()`.  The record embeds
the complete input budgets, capabilities, offer provenance, warrant provenance,
resource provenance, selection basis, and combination decline ledger.

## Why amplitude alone is insufficient

The old angle selector is intentionally retained as a compatibility API.  Its
crossover is an accuracy crossover, while its own source records a different
and much higher measured cost crossover.  A single amplitude threshold cannot
simultaneously express:

- a caller's error tolerance;
- whether the dataset satisfies a scheme's warrant;
- distance/time compatibility;
- a device-memory ceiling;
- a measured execution-cost calibration.

The focused amplitude-ladder test therefore supplies a synthetic evidence
packet in which the Laplace error and the dense-rule cost have different
crossings.  The planner selects exact at low amplitude (Laplace misses the
error budget), exact at moderate amplitude (both are accurate but exact is
cheaper), and Laplace at high amplitude (both are accurate and Laplace is
cheaper).  Those numbers test policy only and are explicitly not RIFT kernel
measurements.

This follows the manuscript's Section IV policy at the structural level: no
single method is presumed to cover the whole amplitude range, cost and returned
quality are separate deliverables, and an approximation is not made correct by
being affordable.  Section IV concerns samplers, so none of its performance
numbers are reused as quadrature calibration.

## Production gate

Before exposing a driver option, each live adapter must provide all of the
following from the concrete data and device:

1. a per-axis quantitative accuracy assessment whose evidence class is honest;
2. an implemented certificate if the offer is to enter the strict pool;
3. compute on a common measured unit and a conservative live-memory estimate;
4. static and conditional compatibility tokens from the existing build-time
   predicates;
5. a wrapper-level application test showing that a `decline` cannot become a
   default scheme;
6. low/moderate/high-amplitude campaign measurements, including the overlap
   regions and device classes on which cost ordering changes.

Until that evidence exists, the planner should remain an explicit prototype.
Its useful production contribution today is the typed contract: it makes the
missing evidence visible and prevents the next selector from encoding it as
another unexplained crossover.
