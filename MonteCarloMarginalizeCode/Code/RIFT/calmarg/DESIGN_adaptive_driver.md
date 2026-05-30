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
3. The residual baseline-vs-calmarg lnL gap observed in full runs is dominated by
   **extrinsic-sampler under-convergence** (low neff), not a calmarg error — the
   brute-force reference (below) is the way to confirm this.

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

## Recommended sequencing

1. **Timing data** (done via `--scan-ncal`) — quantify the per-eval and brute-force cost.
2. **Brute-force reference harness (A)** — ground truth; settle baseline-vs-calmarg.
3. **Lazy pilot (C)** — the production path; validate against A on a boring-cal case.
4. **Breadcrumb interface (B-lite)** — `save/load` the learned cal proposal (Gaussian
   now), integrator-agnostic; NF is a separate, later project.

Open design questions for discussion before coding:
- Where exactly to source the K pilot points (CIP grid output? a dedicated short ILE
  pilot? the maxpt?).  Cleanest is probably a short low-`n_max` ILE pilot at the best
  intrinsic point.
- Whether the pilot runs inline (one process) or as separate parallel jobs.
- The breadcrumb file schema (so it is useful beyond calmarg).
