# Adaptive calibration sampling: driver plan

Status: **planning** (do not implement the multi-stage loop until we pick a path).

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
   the largest, condor-DAG piece; stub with TODOs referencing this doc, build last.
