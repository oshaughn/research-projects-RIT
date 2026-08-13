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
- [ ] **Read `git show` (the diff, not just `--stat`)** and confirmed every hunk is mine.
      `--stat` gives filenames and line counts, which catches an unexpected *file* but not an
      unrelated edit inside a file I also touched — the case that actually occurs on these
      shared checkouts, where `git add -A` has twice swept another branch's uncommitted work
      into a commit. Use `--stat` for the file list, the diff for the contents.
- [ ] Any changed default is justified by a measurement, and the previous behaviour is still
      reachable by flag

### If this touches the integrators, the ILE, the likelihood, any evidence/weight/gate, **or changes a default that production inherits**

- [ ] Read `RIFT/integrators/REVIEW_CHECKLIST.md` §2–3 (population-vs-sample; side effects
      when the feature is off)
- [ ] **Requested a full code review** — high blast radius: a wrong answer here is plausible,
      not loud
- [ ] Shape-recovery merge gate run per `RIFT/integrators/TESTING.md` — required whenever the
      change can move a posterior, which includes a changed default that production inherits,
      not only edits under `RIFT/integrators/` (the fast CI integral test does not subsume it)
