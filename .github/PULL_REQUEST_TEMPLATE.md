<!--
Delete any section that does not apply.  The checklist below is the CHEAP pass; the
integrator-specific invariants and the blast-radius rule live in
MonteCarloMarginalizeCode/Code/RIFT/integrators/REVIEW_CHECKLIST.md
-->

## What this changes, and why

<!-- The defect or the goal.  If a number changed, say which measurement says so. -->

## Evidence

<!-- What was measured, in WHICH REGIME, over how many replicates.  A single run that
     completed and looked sane is not evidence about a degenerate-regime fix. -->

## Checklist

- [ ] New/changed tests are **named in `.github/workflows/ci.yml`** (a test file existing does
      not mean CI runs it — grep the workflow)
- [ ] Tests assert the precondition they are about, so they cannot pass vacuously
- [ ] `git show --stat` contains only my hunks (shared checkouts; `git add -A` has swept in
      other branches' work)
- [ ] Any changed default is justified by a measurement, and the previous behaviour is still
      reachable by flag

### If this touches the integrators, the ILE, the likelihood, or any evidence/weight/gate

- [ ] Read `RIFT/integrators/REVIEW_CHECKLIST.md` §2–3 (population-vs-sample; side effects
      when the feature is off)
- [ ] **Requested a full code review** — high blast radius: a wrong answer here is plausible,
      not loud
- [ ] Shape-recovery merge gate run per `RIFT/integrators/TESTING.md` (the fast CI integral
      test does not subsume it)
