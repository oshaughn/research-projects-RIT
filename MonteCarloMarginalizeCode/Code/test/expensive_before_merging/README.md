# Expensive pre-merge validation suites

Tests in this tree are **NOT** run in per-commit CI.  They are the strong,
slow checks run **before confirming a merge** into a production line
(`rift_O4d`, future `rift_O4e`, ...), and when back-checking a production line
against its predecessor (e.g. `rift_O4c` -> `rift_O4d`).

Rationale: the fast CI gate (`.travis/test-integrate.sh`) validates the
*integral* on a single 3-D Gaussian.  Integrals are easy — importance-sampling
estimates of Z are unbiased under weak conditions — while the recovered
*posterior shape* (the weighted sample cloud consumed by CIP and the fairdraw
machinery) can be subtly wrong: clipped tails, wrong widths, missing mixture
components, distorted correlations.  Production merges must pass the shape
test, not just the integral test.

## Suites

* `integrators/` — posterior shape-recovery gate for the MC integrators
  (AV, GMM, NF, portfolio; optionally AC/default).  Random seeded Gaussian
  mixtures across dimensions, following RIFT-FinerNet
  `demos/integrators/multigauss_direct` (Wagner et al).  See
  `integrators/shape_recovery.py` docstring for method and thresholds, and
  `integrators/run_shape_recovery.sh` for the standard invocation.

## Merge workflow

1. Run the suite on the **base** branch: `--json base.json`.
2. Run the suite on the **candidate** branch (same preset/seeds): `--json pr.json`.
3. `python integrators/compare_shape_results.py base.json pr.json`
   - Merge-blocking: any strict-sampler run that regresses PASS -> FAIL, or a
     metric regression beyond tolerance (see script).
   - Pre-existing failures (FAIL on both) do not block, but should be ticketed.
4. Attach both JSON files + the comparison output to the PR before confirming.

Policy: AV is the gold-standard production sampler and is always strict.
GMM is strict by default.  NF and portfolio are warn-only by default (known
weaker in older lines, e.g. rift_O4c); tighten with `--strict-samplers` as
they harden.
