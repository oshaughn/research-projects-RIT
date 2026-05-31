# Adaptive calibration sampling: driver plan

Status: **planning** (do not implement the multi-stage loop until we pick a path).

## Zero-cal burn-in of the extrinsic sampler (proposed; ILE-level, analyze_event)

**Problem.** Calibration marginalization makes the extrinsic integral much harder to
converge: with cal drawn from the broad PRIOR the per-time lnL has a large dynamic range,
so the adaptive sampler (AV) struggles to reach a useful `n_eff` -- and on the FIRST
iteration there is no learned cal proposal yet, so every cold-start ILE job is in this
regime.  Empirically (local DAG run) iteration-0 points come out with `sigma ~ 0.7-0.9`
and few effective samples; high-SNR sources are hard even WITHOUT cal, and cal makes it
worse.  We are effectively failing to seed the *intrinsic* grid because the *extrinsic*
sampler never gets going.

**Idea (O'Shaughnessy).** Burn the sampler in on a *different, cheaper* likelihood first
-- the ZERO-CAL (n_cal=1) baseline -- until it reaches a minimal `n_eff`, so the
extrinsic sampling proposal is "roughly right", THEN switch to the full cal-marginalized
likelihood for the production estimate.  The extrinsic posterior shape is nearly the same
with and without cal (cal mostly rescales / mildly shifts lnL), so the burned-in proposal
is an excellent warm start -- and the zero-cal evaluations are ~`n_cal`x cheaper.

**Where it lives.** `analyze_event` in `integrate_likelihood_extrinsic_batchmode`.  The
likelihood closures already read the module-scope `n_cal_for_likelihood`; the sampler also
already supports a warm start via `sampler.update_sampling_prior(..., external_rvs=...)`
(the existing `oracleRS` path, ~line 2078).  Two viable mechanisms:

  1. **Two-phase integrate (simplest).**  Set `n_cal_for_likelihood = 1`, call
     `sampler.integrate(likelihood_function, ..., n_eff=burn_in_neff)` (the closures now
     evaluate the fast zero-cal baseline), then restore `n_cal_for_likelihood` and call
     the production `sampler.integrate(...)` WITHOUT resetting -- reusing the adapted
     proposal.  Risk: AV's reset semantics across two integrate() calls in one
     analyze_event are unverified (it "always resets every iteration" between DAG
     iterations; need to confirm it does NOT reset at the start of integrate()).
  2. **Warm-start via update_sampling_prior (robust).**  Run the zero-cal burn-in,
     harvest its drawn extrinsic samples + lnL, and feed them to
     `sampler.update_sampling_prior(external_rvs=...)` exactly like the oracle path, then
     run the production integrate.  Survives regardless of integrate()'s reset behavior.

**Proposed flag.** `--calibration-burn-in-neff <float>` (0/None = off): target n_eff for
the zero-cal burn-in (capped by a fraction of n-max).  Default off; opt-in.

**Relation to the bigger seeding plan.** This is the in-job version of the same idea as
the *zero-cal pilots* that seed the intrinsic/extrinsic grids for high-SNR sources: burn
in cheaply, then pay for cal only once the sampling is on-target.  The pilot (`pilot.py`,
util_CalPilotStage) seeds the CAL proposal across iterations; this burn-in seeds the
EXTRINSIC proposal within a single job.  They compose.

Status: framed; needs implementation behind the flag + a GPU smoke test to confirm the
sampler reuses/seeds adaptation across the burn-in -> production handoff.

## Where we are

- In-loop calmarg works and is validated (loop == fused == reference ~1e-14; CPU+GPU;
  default/distmarg; phase-marg).  See `DESIGN_calmarg_in_loop.md`.
- **Phase 0** (importance weights, `cal_log_weights`) is wired end-to-end: the
  marginalizer computes `Z_cal = sum_c exp(log_w_c) integral L_c / sum_c exp(log_w_c)`.
- **Phase 1 core** (`adaptive.py`) is implemented and unit-tested standalone: a tempered
  unimodal-Gaussian proposal in cal spline-node space, importance weights
  `w_c = prior/proposal`, fit to the cal posterior, `neff` diagnostics.  It needs an
  `evaluate(nodes) -> log integral_theta L` callback to run for real.

The open question is **how to supply that callback in the driver without making the run
expensive** — i.e., how to *learn* the cal proposal during a real analysis.

## Key facts that shape the choice

1. **The extrinsic integration is slow even at Lmax=2** (toy problem).  So anything that
   multiplies the number of full integrations (brute force, multi-stage adaptive) is
   costly.  Per-likelihood-evaluation timing (`backtest_calmarg.py --scan-ncal`, GPU,
   3 IFO, distmarg, 4096 extrinsic samples, ms/eval):

   | n_cal | reference (brute) | loop (Option B) | fused (Option C) |
   |------:|------------------:|----------------:|-----------------:|
   |     1 |              57.6 |            57.2 |             57.2 |
   |    10 |             571   |           266   |             66.9 |
   |    50 |            2854   |          1191   |            198   |
   |   100 |            5702   |          2347   |            362   |
   |   200 |           11427   |          4662   |            704   |

   The marginal cost of one extra cal realization is ~57 ms (brute), ~23 ms (loop), and
   ~3.3 ms (fused) -- fused amortizes the cal axis ~18x better than brute force.  A
   brute-force reference at n_cal=200 is ~11 s **per likelihood evaluation**; a full
   integration is thousands of evaluations, i.e. hours-to-infeasible -> reference only.
   Fused at n_cal=200 is ~0.7 s/eval -> a production integration is feasible.  Net:
   **the production path must be fused**, and we still want to keep n_cal modest via a
   learned proposal (the pilot below).
2. **Calibration is boring**: the cal posterior is smooth, unimodal, and — crucially —
   nearly **independent of the extrinsic parameters** across the high-likelihood region
   (it is set by the data + best-fit template, not by sky/inclination/etc).  So we do
   NOT need to relearn cal per extrinsic sample, nor iterate many times.
3. The calmarg lnL sitting ABOVE the no-cal baseline is **expected physics, not a bug**:
   for cal-on-data, at fixed theta  lnL_c = lnL_baseline + (delta.h|h)  with mean 0, so
   Z_cal(theta) = E_C[L] = L_baseline * exp(+ shift); logmeanexp(lnL_c) > mean(lnL_c)
   (dominated by the best-fitting cal draws).  Confirmed at the injection point on real
   data: mean(lnL_c) ~ baseline, logmeanexp(lnL_c) above it by a positive margin.  Small
   cal variance -> shift ~ 0.5*Var_c[lnL_c]; high SNR -> larger (best-draw dominated).
   This is ALSO why neff_cal collapses and adaptive sampling is needed.

## Options (and recommendation)

### A. Brute-force reference (DO — as validation, not production)
Marginalize cal "the hard way": draw a large prior cal set and run the full extrinsic
integral, or sample (theta, cal) jointly with no proposal learning.  Cleanest and
unambiguous; the **ground truth** to validate B/C and to settle the baseline-vs-calmarg
question.  Slow (cost ~ `n_cal` x single integration), so it is a *reference harness*,
not the production path.  Implement as a mode that runs the existing integration with a
large prior cal set and high neff, and compare to the lazy/seeded result.

### B. Expanding scope: portable (extrinsic + cal) distribution / normalizing flow (LONG TERM)
Capture the learned joint (extrinsic + cal) posterior in a **portable object** to pass
downstream — historically a normalizing flow.  This is the decade-old "breadcrumbs"
goal; it has failed before partly because it was bolted on per-integrator and never
standardized.  The cal framework sits deep in the core and faces the *same* challenge,
so the right move now is **not** to build a full NF, but to define a clean,
integrator-agnostic **breadcrumb interface**: a small object that can `save`/`load` a
learned proposal (start: Gaussian mean/cov over cal nodes + the importance weights;
later: an NF), with a stable schema.  Build the hook; defer the NF.

### C. Lazy pilot (RECOMMENDED first production path)
Because cal is boring and ~extrinsic-independent, learn it ONCE from a cheap pilot:

1. Get a handful (K ~ tens) of **high-likelihood extrinsic test points** — e.g. the top-K
   by lnL from the first ILE iteration / the proposed grid, or just the best-fit point.
2. At those K points, evaluate the per-cal-realization likelihood **fully** (this is K
   cheap evaluations, embarrassingly parallel — "spam in parallel").  Average the
   responsibilities over the K points (they agree, since cal is extrinsic-independent).
3. Fit the Gaussian proposal (`adaptive.fit_proposal`, tempered) -> seed the cal nodes.
4. Redraw the run's cal realizations from the proposal; set `cal_log_weights =
   prior/proposal` (Phase 0).  Run the main integration once with the seeded set.
5. (Optional) one refine pass if `neff_cal` is still low.

This is a single extra pilot (not a multi-stage loop), exploits cal's boringness, reuses
Phase 0 + Phase 1, and degrades gracefully (if the pilot is poor, importance weights
keep it unbiased — just less efficient).

## AGREED architecture and priority (do all of A, C, B to prep for the future)

Priority order **A -> C -> B**:
- **A is the critical benchmark** -- the *only* validation.  Build first.
- **C is production** (the parallel-pilot DAG below).
- **B is the future** (portable extrinsic+cal distribution / normalizing flow).  Lay
  breadcrumbs + stub code now so the plan is remembered.

This is a deliberate "long jump": more structure than calmarg strictly needs, because
the same machinery generalizes to saving the **extrinsic** distribution (the decade-old
goal).  Longer path, but richer payoff and easy to exploit later.

### Source of pilot points: harvest from the previous iteration's `*.composite`
Every RIFT iteration already produces a `*.composite` of evaluated (intrinsic+extrinsic)
points with their lnL -- plenty of trials, no need for a dedicated pilot integration.
The pilot **harvests the top fraction by lnL (~top 5%)** from iteration N-1's composite
and does full cal there.  (This same harvest generalizes to learning the extrinsic
proposal.)

### Parallel-pilot DAG (nothing serial)
Per iteration N, run in parallel:
- **wide_N**: the normal ILE iteration, with `n_cal` modest, its cal realizations SEEDED
  from the consolidated proposal produced after iteration N-1 (importance-weighted,
  Phase 0).  This is the production likelihood.
- **pilot_N**: harvest top-5% lnL points from iteration N-1's composite; do FULL cal at
  those points (large prior `n_cal`, embarrassingly parallel -- "spam in parallel");
  emit a breadcrumb (per-point cal responsibilities / a fitted Gaussian).

Then a **consolidation_N** job (the barrier between N and N+1) collects the pilot
breadcrumbs into a single consolidated cal proposal (Gaussian mean/cov over cal nodes +
importance-weight bookkeeping).  **pilot_N informs wide_{N+1}** through that consolidated
proposal.  A **cap** limits how many iterations keep pilot jobs active (once cal is
learned -- it is boring -- freeze the proposal and drop the pilots).

```
  iter N-1.composite ──► pilot_N ──┐
                                   ├─► consolidation_N ──► wide_{N+1}  (seeded)
       (wide_N runs in parallel) ──┘
  (pilots run for the first ~K iterations, then frozen)
```

### B (breadcrumbs / future): portable distribution object
The consolidated proposal is a **portable save/load object** with a stable,
integrator-agnostic schema.  Start: a Gaussian over cal spline nodes (mean, cov) + the
prior + importance-weight metadata.  Designed from the start to ALSO carry an extrinsic
proposal (same harvest->fit->consolidate->seed structure).  NF is a later drop-in behind
the same interface.  Stub the schema + the consolidation/seed hooks now.

## Build order (this branch)
1. **Timing data** -- done (`--scan-ncal`).
2. **A: brute-force reference** -- prior-only large-`n_cal`, converged; the ground truth.
   Testable now in the backtest: brute-force (large prior set) vs adaptive-seeded must
   agree on Z_cal while the seeded run has far higher `neff_cal`.
3. **B-lite breadcrumb I/O** -- `save/load` the cal proposal (Gaussian; schema with an
   `extrinsic` slot reserved).  Used by C.
4. **C core** -- harvest top-fraction from a `*.composite`; fit (adaptive.fit_proposal);
   write/consolidate breadcrumbs; seed the next run's cal realizations.
5. **C DAG wiring** (pilot || wide || consolidation, the cap) in the pipeline builder --
   DONE (opt-in; default DAG byte-identical).  See "DAG wiring" below.  NEEDS a condor
   smoke test on a real cluster run (cannot be exercised off-cluster), like the main-path
   GPU end-to-end test.

## DAG wiring (implemented; opt-in via `--calmarg-pilot`)

A single per-iteration **calpilot** condor job collapses harvest -> dump -> fit ->
consolidate into one process (`bin/util_CalPilotStage.py`), so the pipeline-builder
surgery is minimal and the steps (which are serial anyway) stay in one place:

```
  iteration N composite ──► CALPILOT_N (util_CalPilotStage.py):
       1. util_CalHarvestGrid.py   top-frac high-lnL pts -> cal_pilot_grid_N.xml.gz
       2. ILE --calibration-dump-responsibilities (cheap: skips the extrinsic sampler)
          [+ --calibration-proposal-breadcrumb cal_consolidated_{N-1}.npz  -> refine]
       3. util_CalPilotFit.py       -> cal_proposal_N.npz   (auto-tempered)
       4. util_CalConsolidate.py    -> cal_consolidated_N.npz
                                   │
       (CALPILOT_N runs ∥ CIP_N/puff_N; parent = unify_N, does NOT gate them)
                                   ▼
  wide ILE jobs of iteration N+1  --calibration-proposal-breadcrumb cal_consolidated_N.npz
       (depend on CALPILOT_N; a missing breadcrumb at early N falls back to the prior)
```

- `dag_utils.write_calpilot_sub` defines the job; `create_event_parameter_pipeline_BasicIteration`
  instantiates `calpilot_node` per active iteration (parent `unify_node`), records it, and
  makes iteration N+1's wide ILE nodes depend on `calpilot_node[N]`.  ILE nodes carry a new
  `macroiterationprev` macro so the per-iteration breadcrumb path resolves.
- **Cap & cadence**: `--calmarg-pilot-max-it` (default 3), `--calmarg-pilot-cadence`
  (default 1) -- pilots stop once cal is learned (cal is boring), freezing the proposal.
- `util_RIFT_pseudo_pipe.py`: `--calmarg-pilot[-cadence|-max-it|-top-fraction|-max-points]`
  add the CEPP flags and append the `--calibration-proposal-breadcrumb
  .../cal_consolidated_$(macroiterationprev).npz` to the wide ILE args (args_ile.txt).

Run: add `--calmarg-pilot` to a `util_RIFT_pseudo_pipe.py` invocation that already uses
`--calmarg-envelope-directory ...`.  Everything is opt-in; without `--calmarg-pilot` the
DAG and ILE behavior are unchanged.

NOTE (subdag/exploded-ILE): the seed dependency is wired for the standard ILE batch path;
the `--ile-group-subdag` grouped path would need the dependency placed on the subdag node
(left as a follow-up; uncommon for calmarg runs).

## Implemented executable decomposition (this branch)

The pilot/seed loop is realized with two ILE flags + two thin CLIs, all opt-in (the
default DAG and likelihood are byte-identical when unused):

- `generate_realizations.py` (refactored, prior draws byte-identical):
  - `build_realizations_from_nodes(...)` -- spline construction, shared by prior & proposal.
  - `node_prior(...)` -- the diagonal-Gaussian cal prior per detector.
  - `draw_prior_realizations_with_nodes(...)` -- prior draws that KEEP the node vectors
    (cold pilot, N=0).
  - `seed_realizations_from_breadcrumb(...) -> (factors, cal_log_weights, nodes)` -- draw
    cal realizations from a learned proposal + Phase-0 weights log(prior/proposal).
- `factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(..., return_cal_components=True)`
  returns the RAW per-realization time-integrated log L `(npts_extrinsic, n_cal)` before
  the cal collapse (loop method).  Validated: `logsumexp_c(components) - log(n_cal)` ==
  the cal-marg lnL to ~1e-16.
- ILE `--calibration-proposal-breadcrumb <bc>`: seed the run's cal realizations from the
  proposal (+ thread `cal_log_weights` to all likelihood call sites & resample).  This is
  what **wide_{N+1}** uses.
- ILE `--calibration-dump-responsibilities <out>` (+ `--calibration-pilot-extrinsic`):
  the **pilot**.  Keeps the cal node draws; at each analyzed (harvested) intrinsic point,
  evaluates `return_cal_components` over a uniform-prior extrinsic batch and accumulates
  `int dOmega L_c` per realization; writes `(nodes, log_resp=log_w+log int L_c, prior...)`.
  If `--calibration-proposal-breadcrumb` is ALSO given, the pilot draws FROM that proposal
  (refinement: pilot_N seeded by consolidation_{N-1}) and folds `log_w` into `log_resp`.
- `bin/util_CalPilotFit.py`: pool dumps -> `adaptive.fit_proposal` (AUTO-TEMPERED: pick the
  largest beta<=1 whose tempered neff >= `target_neff_frac*n_cal`, so a low-neff cold draw
  cannot collapse the proposal) -> breadcrumb.
- `bin/util_CalConsolidate.py`: precision-weighted combine of pilot breadcrumbs (or a
  single-input pass-through) -> the consolidated proposal that seeds wide_{N+1}.

The across-DAG-iteration loop (pilot_N seeded by consolidation_{N-1}, refit, ...) is
exactly `adaptive.adaptive_cal` UNROLLED over RIFT iterations -- no extra serial cost.

## Convergence characterization (measured)

The cal node space is high-dimensional (2 * spline_count * n_det; e.g. 60 for 10 nodes x
3 IFOs).  A single Gaussian proposal learned from one prior shot in this space converges
SLOWLY when the cal posterior is strongly displaced/narrowed vs the prior: in a stress
test (12 of 60 nodes offset 1 sigma, tightened to 0.5 sigma) the responsibility neff sits
~1-3 and `|mean-true|` only falls to ~0.5 sigma over many rounds -- and the reference
`adaptive.adaptive_cal` behaves the SAME (this is intrinsic to broad-prior importance
sampling in high-D, not a wiring defect).  Two things make this acceptable:
1. **Correctness is independent of pilot quality.** The Phase-0 importance weights make
   the marginalization UNBIASED for any proposal; a poor pilot only lowers `neff_cal`.
2. **Real cal is boring.** Posteriors are small, smooth, near-prior displacements; in a
   benign regime (offset ~0.3 sigma) the prior is already a decent proposal and the pilot
   gives a modest neff gain.  The big wins are when cal is genuinely informative, where
   the across-iteration climb accumulates.
For a sharp high-D posterior the right long-term tool is **B (normalizing flow)** behind
the same breadcrumb interface -- a single Gaussian is the deliberate first cut.
