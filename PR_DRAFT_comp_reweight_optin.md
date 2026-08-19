# DRAFT PR: opt-in gated composition reweight for the low-mass transverse-spin width deficit

**This PR is not proposed for merge yet.** It is staged as a DRAFT while the paper-scale
demonstration (in preparation) independently establishes the problem and the fix; the merge
decision waits on that demonstration. Please do not review or mark ready until then.

(This file mirrors the PR description; delete it at merge.)

## What this is

An opt-in (default OFF) severe-deficit repair for CIP. Across the O4 low-mass catalogue,
RIFT's chi1_perp/a1 posteriors are systematically narrow (median width ratio 0.800 vs bilby
over 25 events; high-mass events sit at 1.007). Root cause, established by intervention:
training-set COMPOSITION — the near-peak fraction of `all.net` rows degrades with chi1_perp,
and the RF fit (a local average) regresses the transverse tail toward its junk-diluted
neighbours. Density-equalising THINNING of the far-from-peak rows (never truncation, never
invented points; lnL span preserved exactly) repairs it: on 5 deficit exemplars, paired CIP
on identical inputs moved chi1_perp toward the reference by +0.042..+0.115 with mc/q
undamaged.

An UNGATED version of that thinning was tested two-sided and REJECTED, twice and
independently: on real data it overshot three bad-composition events whose widths are
already correct to 1.18–1.54x the reference, and on the known-truth toybench it recovered
**2.56x / 3.32x / 2.23x truth** on genuinely-narrow transverse posteriors (T5/T5b/T5c) —
a specificity veto at every tail depth, so the true worst case of unconditional thinning is
up to **3.32x**, not the 1.54x real data happened to show. The two results reconcile
exactly: the toybench predicted the high-mass control group should widen past the reference
under unconditional reweighting, and the real-data test measured precisely that pattern —
good-composition high-mass controls were exact no-ops (+0.001, +0.013) while
bad-composition healthy events blew up. The driver is COMPOSITION, not mass: unconditional
thinning harms bad-composition events regardless of whether they are deficient, because on
a genuinely narrow posterior it deletes the far rows that teach the fit the tail is
unsupported. The T5 family is the known-truth instance of exactly the S240930aa class.
This PR ships the gated version: thin ONLY on a detected severe deficit.

## The gate (`util_CIPTailDeficitGate.py`)

R = delivered/implied transverse-tail posterior mass, computed from RIFT's own products
only (all.net + a CIP posterior trained on it; no external reference):
implied = per-chi1_perp-bin analytic prior volume x mean exp(lnL-peak) over the bin's real
ILE rows; boundary = the training set's own cp q80. FIRE iff R < 0.32 AND the MANDATORY
sample-resolution validity floor (implied >= 50/n_post) is satisfied; otherwise a loudly
logged no-op.

* **The validity floor cannot be bypassed** (no flag exists; `--floor-counts < 1` is
  refused; `decide()` applies it before the threshold; property-tested). Without it the
  detector returns R = 0 — maximum apparent deficit — exactly where it can resolve nothing:
  on known-truth healthy-narrow benchmarks (toybench T5/T5b/T5c) every chain read R = 0 and
  only the floor prevented false fires.
* **The toybench's own remedy is this design**: its veto of the ungated arm concludes a
  deployable version must gate on tail *support*, not row counts — R is exactly a
  tail-support statistic (implied tail mass from measured lnL x prior volume), the validity
  floor covers the unresolvable-support case, and on the T5 family the shipped gate was
  measured to ABSTAIN via that floor. The toys independently veto the ungated tool and
  independently endorse the gated one's design principle; no more than that is claimed.
* **The threshold is channel-calibrated at 0.32**, in the fresh-CIP measurement channel the
  wrapper actually uses. The population calibration (102 production events, perfect
  12-vs-77 separation, gap 0.379–0.457) was measured on production consolidated posteriors;
  the fresh-CIP channel reads mid-band healthy events lower by up to ~0.07, so the
  population threshold 0.42 does NOT transfer — **the regression fixture caught this**
  (control S240930aa: deployment-channel R 0.386 < 0.42). Deployment-channel gap: severe
  deficits <= 0.270, healthy controls >= 0.386; 0.32 is the geometric midpoint.

## The wrapper (`util_CIPCompositionReweightWrapper.sh`, drop-in `--cip-exe`)

Detect -> repair: N detect passes (default N=5, `CIP_REWEIGHT_GATE_REPS`) run the REAL CIP
on the ORIGINAL data (exports bumped to >= 20000, temp outputs; real output paths
untouched); the gate decides ONCE on the MEAN R over the reps; the final CIP runs the
original argv verbatim, with `--fname` swapped to the thinned set only on a FIRE. Every
no-op is logged with the R value, the per-rep spread, and the deciding condition.
Fail-safe throughout: any tool failure -> final CIP on the original argv, loudly. Costs N
extra CIP passes per invocation when enabled (CIP is the cheap CPU stage).
RandomizeOverlapOrder-style modularity: nothing in CIP or the merge step changes; flag-off
is byte-identical to today.

Opt in via `util_RIFT_pseudo_pipe.py --internal-cip-composition-reweight` (conflict-guarded
vs `--internal-use-amr`).

## SCOPE — read before enabling

* Repairs SEVERE deficits only (production low-mass class, width ratios ~0.62–0.69;
  deployment-channel R <= 0.27). **Roughly half the affected low-mass events; the mild rest
  are loudly left alone**: mild deficit and healthy width are NOT separable above the
  threshold (out-of-sample, truth-deficient toybench T3 chains at R 0.532–0.634 abut
  truth-healthy T2 chains at 0.633–0.673).
* Safety is the strongly supported side: zero false fires across 77 in-sample healthy
  events (95% bound 3.8%) and on known-truth healthy-narrow toys. The fire side is
  in-sample-validated (12/12 severe-deficit events, 95% miss-rate bound 22.1%) plus the
  causal 13-event paired-CIP repair.
* What the known-truth evidence does and does not certify: the toybench could not
  reproduce the pathological regime — its gate (a) FAILED (a1 = 1.381 vs required <= 0.65:
  from a fresh bootstrap the loop EXPANDS 38% at iteration 1 rather than collapsing; third
  independent harness to fail this, and the first running the real cepp_BasicIteration
  DAG) — so it cannot certify the fix where the deficit actually develops. The known-truth
  evidence certifies the SAFETY side (abstain/no-fire behaviour) only; the REPAIR side
  rests entirely on the real-data paired-CIP result.
* Margin, MEASURED (16 independent detect reps on the closest healthy control S240930aa;
  R_dispersion.json + the live N=5 run in the study record): deployment-channel mean
  R = 0.371, single-rep sigma = 0.024, and **1 of the 16 single reps actually read below
  the threshold (0.325)** -- a single-pass gate has an observed ~6%/run false-fire rate on
  this event class, each fire costing up to 3.32x truth (known-truth toybench bound;
  1.54x was the real-data instance). The shipped
  default N=5 mean-of-reps retires this: effective margin ~4.6 sigma (sem 0.011), and in
  the live N=5 verification the sub-threshold rep occurred and the mean correctly decided
  NO-FIRE. Fire side: measured exemplar sigma 0.014 (S240629by, 11 sigma single-rep);
  the nearest exemplar to the threshold (S241109bn, R = 0.27) keeps ~5 sigma at N=5.
  The two S240930aa readings quoted earlier (0.386 pooled, 0.357 live) decompose into a
  REAL channel offset (production 0.457 vs deployment 0.371, -0.086 >> sem) plus this
  single-rep noise -- both handled: channel-calibrated threshold, averaged decision.

## Verification

* `test/test_tail_deficit_gate.py` (named in ci.yml; 24 tests, all passing): measured-value
  regressions pinning the shipped gate's decisions on the 5 exemplars (FIRE), the 3
  controls the ungated fix broke (NO-FIRE), and 6 known-truth healthy-narrow toybench
  chains (ABSTAIN **via the floor** — each has R < threshold, so the threshold alone would
  false-fire; the reason is asserted, not just the outcome); property scans proving no
  (implied, R, n_post) below the floor can ever FIRE; CLI bypass refusal; synthetic
  end-to-end FIRE/NO-FIRE/ABSTAIN; wrapper e2e (FIRE swaps fname for the final pass only;
  broken gate falls back loudly to the original).
* Live end-to-end on production data (real CIP, branch as a unit): S240629by -> FIRE
  (R=0.171), final CIP trained on the thinned set; S240930aa -> NO-FIRE (R=0.357), final
  CIP verbatim; both rc=0 with the decision and record path in the log.

## Evidence record

rift_transverse_highSNR_study `results_triage/`: MULTIEVENT_REWEIGHT_2026-08-19.md (ungated
two-sided test, rejected), GATED_REWEIGHT_2026-08-19.md (13-event gate-0),
R_POPULATION_CALIBRATION_2026-08-19.md (102-event calibration; comp-contamination hypothesis
refuted), R_TOYBENCH_VALIDATION_2026-08-19.md (known-truth out-of-sample; floor load-bearing),
TOYBENCH_RESULT_2026-08-19.md (independent campaign: ungated arm VETOED at 2.56-3.32x
truth; gate (a) failure bounding what the bench can certify), BRANCH_LANDING_2026-08-19.md
(channel systematic, live e2e, measured R dispersion).

Not applicable to the LISA fork (CIP-side change; the LISA fork diverges only in the ILE
driver).
