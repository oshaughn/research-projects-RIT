# Integrator gate / sampler follow-ups

Captured here because the junior fork has issues disabled. Each entry is self-contained: the
evidence is included so none of it has to be re-derived.

---

## 1. The flag-ON probe can fail a PR on a coin flip -- it needs confirm-on-fail

**Status:** DONE -- confirm-on-fail added to the probe (`--confirm-repeats`, `--confirm-seeds`,
`--confirm-min-valid`, `--no-confirm`).  Verified live: the `adaptive_alloc ON / d4_n1_s303` row
cleared at 3 fresh seeds with the two arms bit-identical (36/36, 84/84, 131/131), probe exit
1 -> 0.  Tests in `test_probe_confirm.py`.

`probe_portfolio_optin_flags.py` reuses the shape gate's thresholds and its `evaluate()`, so it
inherits the same near-threshold realization sensitivity -- but unlike `compare_shape_results.py`
it has **no confirmation step**, so one noisy row fails a PR.

**Evidence.** `adaptive_alloc ON / d4_n1_s303` reported `base=PASS flag=FAIL` on four consecutive
gate runs during PR #51. Running the probe against that branch's exact base reproduced it
identically (flags-off n_eff 131, flag-on 434, same FAIL), so it was never caused by the branch.
On identical code that cell has read n_eff **422 (PASS)** in one run and **46 (STARVED)** in
another.

**Fix.** Apply the #49 policy: on a flagged row, re-run that cell at several fresh run seeds with
the flag off and on, and report a regression only if the flag arm is worse in a majority. Reuse
`confirm_regressions.py` (valid-pair requirement, candidate-failure counts against the candidate,
INCONCLUSIVE exits non-zero) and `compare_shape_results.classify()` / `is_blocking()` -- do not
write a second definition of "regression".

**Acceptance.** Equivalent-at-fresh-seeds clears; a genuinely worse flag arm still blocks; too few
valid pairs is INCONCLUSIVE, not a pass. Tests in the style of `test_confirm_regressions.py`, aimed
at the direction that matters -- a false clear ships a bug, a false block costs a rerun.

**Do not** "fix" this by seeding the samplers deterministically. Independent copies that localize
differently are the working detector for support / mode-collapse failures; pinning every fit
silences it and makes N production copies no better than one.

---

## 2. Strict gate row `GMM mix_d6_n3_s303` is mis-budgeted (starves 4 of 5 seeds)

**Status:** needs a decision, not a code fix.

The cell sits on the `n_eff = 100` starvation floor, so as a **strict** (merge-blocking) row it is
close to a coin flip on every branch. From the confirm-on-fail run added in #49 (5 fresh seeds,
both arms bit-identical):

```
seed 988654: base=STARVED cand=STARVED (n_eff 93 vs 93)
seed 989654: base=STARVED cand=STARVED (n_eff 80 vs 80)
seed 990654: base=PASS    cand=PASS    (n_eff 119 vs 119)
seed 991654: base=STARVED cand=STARVED (n_eff 95 vs 95)
seed 992654: base=STARVED cand=STARVED (n_eff 96 vs 96)
```

4 of 5 starve, so the PASS at the default run seed is the lucky draw. It was reported as a blocking
`REGRESSION(pass->starved)` in two consecutive full gate runs during PR #47 before confirm-on-fail
cleared it.

**Decision:** raise the budget for this cell so it clears the floor reliably, or drop it from
`--strict-samplers`. Deliberately not changed unilaterally -- the strict list and per-cell budgets
are shared with other people's work. Confirm-on-fail now stops it blocking spuriously, so this is
cleanup rather than an outage: it costs a 5-seed rerun each time it fires.

---

## 3. Audit `_rvs` consumers that prefer a cached column over the canonical components

**Status:** not started.

PR #51 fixed a case where exported science products disagreed with the reported evidence:
`_pool_replica_rvs` rewrote `log_joint_s_prior` to carry the corrected replica weights, but the
`.dgrid` and calibration-posterior exporters read a **cached** `log_weights` column first, falling
back to `log_integrand + log_joint_prior - log_joint_s_prior` only when it is absent.
`mcsamplerPortfolio` writes that column (`mcsamplerPortfolio.py:1531`), so the stale cache was live
in exactly the portfolio-replica case.

**Task.** Find every other consumer of `sampler._rvs` that prefers a cached/derived column over the
canonical components, and either make it derive, or make whatever rewrites the components also
rewrite the cache. Start from the two sites fixed in #51 plus `mcsamplerGPU.py:766/771/847`, which
maintains its own `log_weights`.

**Why as a sweep.** Every finding in the #47/#51 review series was the same shape -- two
representations of one quantity, secondary copy goes stale, both plausible in isolation so the
failure is silent: components vs cached `log_weights`; a `defensive_frac` marker vs the component
actually installed; help text vs behaviour; remembered `setup()` kwargs vs the rebuilt integrator.
Where deriving is cheap it beats caching; where a cache is required for cost, whatever invalidates
the source must invalidate it too.

**Acceptance.** A list of consumers with a verdict each (derives / kept in sync / fixed), plus a
regression test for any defect found -- asserting the cached value both matches the components
**and differs from the stale value**, so it fails on the buggy code rather than passing vacuously.
