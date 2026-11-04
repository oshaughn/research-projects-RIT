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

### Which RIFT gets measured

Every entry point needs `<checkout>/MonteCarloMarginalizeCode/Code` ahead of the installed RIFT on
`sys.path`; without it these suites measure whatever the environment has installed (every IGWN
conda env has a RIFT) and report pass/fail exactly as if they had gated the branch.
`run_shape_recovery.sh` exports it for you.  Under `pytest`, export it yourself:

```
export PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code:$PYTHONPATH
RIFT_RUN_EXPENSIVE=1 pytest -v integrators/test_shape_recovery.py
```

Both paths now refuse to run against a foreign RIFT rather than quietly measuring it, and every
`shape_recovery.py` run prints the RIFT it resolved (`# RIFT under test:`) so an attached gate JSON
can be traced to a checkout.  `RIFT_SHAPE_CHECKOUT=<dir>` names a checkout other than the enclosing
one, for base-vs-candidate work.

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
