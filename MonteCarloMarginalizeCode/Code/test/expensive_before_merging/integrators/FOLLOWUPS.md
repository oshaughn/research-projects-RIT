# Integrator gate / sampler follow-ups

Captured here because the junior fork has issues disabled. Each entry is self-contained: the
evidence is included so none of it has to be re-derived.

---

## 1. The flag-ON probe can fail a PR on a coin flip -- it needs confirm-on-fail

**Status:** DONE -- confirm-on-fail added to the probe (`--confirm-repeats`, `--confirm-seeds`,
`--confirm-min-valid`, `--no-confirm`).  Tests in `test_probe_confirm.py`.

**RE-MEASURED ON THE FIXED PROBE: the row is a CONFIRMED opt-in regression, not noise.**
5 fresh seeds, flag arm FAILS at every one:

```
seed 988654: base=PASS(213)   flag=FAIL(124)
seed 989654: base=PASS(252)   flag=FAIL(125)
seed 990654: base=PASS(244)   flag=FAIL(292)
seed 991654: base=STARVED(67) flag=FAIL(177)
seed 992654: base=PASS(136)   flag=FAIL(140)
-> CONFIRMED (4 worse / 1 not-worse)
```

Note the failure mode: the flag arm often has HIGHER n_eff (292 vs 244, 177 vs 67, 140 vs 136) and
still fails, so it is failing on SHAPE metrics -- more effective samples, worse recovered posterior.
See item 4.  Everything below this line was written before that measurement and is retained for
the record.

**The first live verification was VOID.**  That run reported the row
cleared at 3 fresh seeds "with the two arms bit-identical (36/36, 84/84, 131/131)".  Bit-identical
arms is not a clear -- it is the signature of the patching bug found in review: `patched_build()`
wrapped `SR.build_sampler` as it then stood rather than the pristine factory, and nothing restored
it, so the default arm ran through the flag arm's wrapper.  The confirmation was comparing the flag
against itself, which cannot report "worse" no matter what the flag does.  Fixed here
(`_ORIG_BUILD_SAMPLER` + the `flag_patch` context manager, which refuses to nest); re-run the
confirmation on the fixed probe before treating that row as cleared.

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

**Status:** DONE -- measured and resolved. The cell carries a x4 `CELL_BUDGET_MULT` entry in
`shape_recovery.py`, applied through `cell_budget()` so both entry points agree. The measurement
table and the reasoning for x4 over x2 are below.

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

**MEASURED AND RESOLVED** (8 fresh seeds per budget):

| budget | n_eff min / med / max | clears 100 | median \|bias\| |
|---|---|---|---|
| x1 (matrix default) | 59 / 105 / 159 | **5/8** | 0.014 |
| x2 | 105 / 169 / 214 | 8/8 | 0.010 |
| x4 | 209 / 293 / 473 | 8/8 | 0.012 |

Bias is flat across budgets, so this is threshold margin, not a defect.

**Correction to the options above:** neither was expressible. The matrix budget is `nmax_per_dim*d`
for EVERY cell, and `--strict-samplers` is per-SAMPLER, so "re-budget this cell" and "drop this row
from strict" both needed a mechanism that did not exist. Added `CELL_BUDGET_MULT` in
`shape_recovery.py` (same shape as the existing per-case `WARM_CASES` budgets) and set this cell to
**x4**: x2 clears 8/8 but its minimum, 105, is 5% above the floor, which is not a margin worth
trusting for a row that has read 66 and 119 on unchanged code. x4 gives min 209 (2.1x) and costs
~4% of the gate's evaluations, being one cell of ~96.

**The override goes through `cell_budget()`, not an inline multiply.** The matrix has TWO entry
points -- `main()` (what `run_shape_recovery.sh` drives) and the pytest parametrization in
`test_shape_recovery.py` -- and the first cut applied the table only in `main()`, so the cell this
entry exists to fix stayed starved under `RIFT_SHAPE_PRESET=standard pytest`, which the suite
documents as an equivalent way to run it. Both now call `cell_budget(...)`; verified they agree
(4800000 == 4800000 for this cell).

An **explicit** `--nmax-per-dim` disables the table (`apply_overrides=False`) and says so on
stdout. The CLI documents `nmax = this * ndim`, and silently scaling a caller-named budget by 4
would have corrupted precisely the controlled x1/x2/x4 comparison the table above was derived
from -- the study script passes `--nmax-per-dim`, so it would have been measuring x4/x8/x16 while
labelling the columns x1/x2/x4. `--no-cell-budget-mult` disables it at preset defaults too.

---

## 3. Audit `_rvs` consumers that prefer a cached column over the canonical components

**Status:** DONE -- PR #55 (merged). Found and fixed two defects: the `.dgrid` and
calibration-posterior exporters preferred a cached `log_weights` that means DIFFERENT things in
different samplers (`mcsamplerGPU` stores the tempering-weighted adaptation weight, not the
importance weight), and `mcsamplerGPU.py:1194` appended weights onto `joint_s_prior`, a fix
`mcsampler.py:571` had carried for years. Verdict table in `RVS_CACHE_AUDIT.md`. The third
divergence it flagged (`_rvs['weights']` sorted as a side effect) is resolved in PR #57 as a
documented constraint.

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


---

## 4. `--portfolio-adaptive-alloc` degrades posterior SHAPE on `d4_n1_s303` (confirmed)

**Status:** open. Opt-in, default OFF, and the pipeline never sets it, so nothing in production is
affected -- but the flag is not safe to promote, and this was invisible for the whole #47/#51/#55
series because the probe was comparing the flag against itself.

Confirmed on the fixed probe at 5 fresh seeds (per-seed numbers in item 1). The flag arm fails at
**every** seed, and often with HIGHER n_eff than the default arm -- so it is failing on JS / pull /
width, not on starvation. More effective samples, worse recovered posterior: the "confidently
wrong" signature this project has documented elsewhere (n_eff measures weight CONCENTRATION, not
coverage), now showing up in one of our own opt-in features.

**Do not** treat the higher n_eff as evidence the flag helps. That is precisely the reading that
made the estimator-clip experiment look like a success while it was biasing lnZ by -11.5 nats.

**Interim mitigation: the two adaptive_alloc rows are EXCLUDED from the probe** (commented out in
`FLAG_CONFIGS`, `probe_portfolio_optin_flags.py`), so the probe stays a working regression detector
for the flags that do pass instead of being a standing red row everyone learns to ignore. This is
containment, not a fix -- it is recorded here, greppable in the source, and asserted by
`test_adaptive_alloc_is_excluded_from_the_probe_configs` so the exclusion cannot be quietly lost.
Reinstating the two commented lines is the first step of any fix; expect them to fail until the
flag is actually repaired.

**Next steps.** Establish scope before touching the policy: is this specific to d=4 / ncomp=1, or
does the allocation signal systematically over-concentrate on whichever member reports the best
per-chunk n_ess? Sweep the flag across the full matrix at several seeds, and record JS / pull /
width alongside n_eff so the shape degradation is visible rather than inferred. If it generalizes,
the allocation signal needs a shape-aware guard or the flag should be documented as unsafe.

---

## 5. The `quick` preset cannot clear its own shape floor on `GMM d4_n2_s101`

**Status:** RESOLVED as a preset question -- accept the skip, keep the cell, and do NOT give it a
`CELL_BUDGET_MULT` entry. But the premise below was wrong: this is not a budget shortfall. The
measurement turned up a real, budget-resistant GMM shape defect, now tracked as item 7.

`quick` budgets `nmax_per_dim=50000`, so `d=4` runs at 200k evaluations and that cell reads
**n_eff = 42** against the `MIN_NEFF_FOR_SHAPE = 100` floor. The other three quick cells pass. So
the default `RIFT_RUN_EXPENSIVE=1 pytest test_shape_recovery.py` is 3 passed, 1 skipped, and one
quarter of the quick matrix tests nothing. That much still stands.

**MEASURED (8 fresh run seeds per budget: 987654 + 988654..994654, CPU, branch on `PYTHONPATH`).**
The paragraph that used to sit here read "Not a defect in the sampler -- the same cell passes at
the standard preset's budget ... this cell needs roughly 2.5x". Both halves are false. It was
extrapolated from the single default-seed reading, which is exactly the one-realization reasoning
this file warns about everywhere else.

| budget | nmax | n_eff min / med / max | clears 100 | PASS | median `width_ratio[1]` |
|---|---|---|---|---|---|
| x1 (`quick`) | 200k | 11 / 50 / 101 | 1/8 | 0/8 | 0.989 |
| x2 | 400k | 16 / 32 / 122 | 1/8 | 1/8 | 0.976 |
| x4 (= `standard`'s per-dim budget) | 800k | 28 / 44 / 179 | 2/8 | 1/8 | 0.919 |
| x8 | 1.6M | 41 / 81 / 133 | 2/8 | 1/8 | 0.906 |
| x16 | 3.2M | 33 / 96 / 148 | 4/8 | 1/8 | 0.900 |
| x32 | 6.4M | 28 / 139 / 217 | 6/8 | 1/8 | 0.889 |

Three readings, and none of them supports raising the budget:

* **Budget does not clear the floor.** At x32 -- 6.4M evaluations, 8x the *standard* preset's own
  d=4 budget -- the cell still starves in 2 of 8 seeds, minimum 28. There is no "quick" budget that
  fixes this, and no expensive one either.
* **Clearing the floor does not produce a pass.** PASS is 1/8 at every budget from x2 up. At x32,
  6/8 clear the floor and 5 of those 6 FAIL. Raising `quick`'s budget would turn the pytest smoke
  row from a skip into a hard assertion failure on an unfixed defect -- correctly, but that is
  item 7's decision to make, not a preset-tuning side effect.
* **It fails because it converges to the WRONG answer.** `width_ratio[1]` degrades monotonically
  with budget (0.989 -> 0.889) while the other three dims sit at 1.00, and `mean_pull[1]` grows
  +0.023 -> +0.069. More samples, worse recovered posterior: the same signature as item 4.

**Decision.** Accept the skip.

* *Raise the budget* -- rejected, measured above. It does not work at any budget, and where it does
  become testable the cell fails.
* *Drop the cell from `quick`* -- rejected, but on cost, not on coverage. The matrix is
  `dims x ncomps x seeds x samplers` with no per-cell exclusion (the same "not expressible" wall
  item 2 hit: `--strict-samplers` is per-SAMPLER, `CELL_BUDGET_MULT` is budget-only). The only
  expressible drop is removing `d=4` from `quick["dims"]`, which also removes **AV** d=4 -- a row
  that clears the floor and passes 8/8 at every budget measured. Building a per-cell exclusion
  mechanism to hide one row we have now fully characterized is the wrong trade; the row costs ~11 s.
* *Accept the skip* -- taken. `quick` stays quick, the skip stays visible in the pytest summary,
  and the reason it skips is now recorded here and at `CELL_BUDGET_MULT` instead of being
  re-derived by the next person.

**What the retained row does and does not buy.** It is a *reproducer*, not a detector. `evaluate()`
returns `STARVED` at the `n_eff` floor **before** it looks at JS, pull, width or correlation, and
the pytest wrapper skips that result -- so at `quick`'s budget this row asserts nothing about shape.
Under the canonical driver it is no better: `classify()` maps STARVED/STARVED to `BOTH-STARVED` and
returns before the metric-regression branch, and STARVED->FAIL to `NEWLY-TESTABLE-FAIL`, which is
explicitly flag-don't-block. The only thing a permanently-starved row still catches is
`REGRESSION(missing-in-candidate)` -- a candidate that crashes and emits no record at all.

So `RIFT_RUN_EXPENSIVE=1 pytest test_shape_recovery.py` is 3 passed / 1 skipped by design, but be
clear about what that means: **the skipped quarter is an untested corner, and the defect in item 7
is currently ungated by every preset.** Keeping the row preserves the reproducer and the crash
canary; it does not preserve coverage, because there was never any to lose.

**Do not "fix" this with a `CELL_BUDGET_MULT` entry.** The mechanism added in #59 makes it a
one-line edit -- `("GMM", 4, 2, 101): 3` -- that looks exactly like the fix item 2 landed for
`mix_d6_n3_s303`. It is not the same situation: that cell's bias was flat across budgets, so more
budget bought real margin; this one's width deficit *grows* with budget. The table above is
reproduced in a comment at `CELL_BUDGET_MULT` in `shape_recovery.py`, at the line someone would
edit.

**How it was invisible.** `test_shape_recovery.py` unpacked `evaluate()` as `ok, reasons` and
asserted `ok`. Commit 6467ac91 (2026-07-22) changed `evaluate()`'s contract from
`return len(reasons) == 0, reasons` to `return ("FAIL" if reasons else "PASS"), reasons`, updating
`compare_shape_results.py` and `shape_recovery.py` but not this caller -- which had been written
hours earlier the same day. Every status string is truthy, so from that commit onward the pytest
gate passed on FAIL, STARVED and ERROR alike. Same family as the #47/#51/#55 findings: a contract
with two callers, one updated, both plausible in isolation, failure silent.

---

## 7. GMM under-covers a broad mixture component on `mix_d4_n2_s101`, and worsens with budget

**Status:** open, confirmed, and **ungated -- no preset detects it.** Found while measuring item 5,
which had recorded the cell as merely under-budgeted. Not known to affect production.

**Nothing currently catches this, and nothing did before.** `standard` runs `ncomps=[1, 3]`, so the
ncomp=2 geometry never appears in the merge gate at all. `quick` does run it, but at a budget where
`evaluate()` short-circuits to `STARVED` at the `n_eff` floor before examining width, pull or
correlation -- and the pytest wrapper skips STARVED, while `compare_shape_results.classify()` maps
STARVED/STARVED to the non-blocking `BOTH-STARVED` and never reaches its metric-regression branch.
Adding coverage therefore means deciding to gate a defect that is not yet fixed; see **Next steps**.

**Reproducer** (~20 s, exits 1, prints the width failure directly). Run it from anywhere inside the
checkout; `CHECKOUT` is derived, and the script is invoked by absolute path so no `cd` is needed:

```
CHECKOUT=$(git rev-parse --show-toplevel)
export PYTHONPATH="${CHECKOUT}/MonteCarloMarginalizeCode/Code:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
python3 "${CHECKOUT}/MonteCarloMarginalizeCode/Code/test/expensive_before_merging/integrators/shape_recovery.py" \
    --samplers GMM --dims 4 --ncomps 2 --target-seeds 101 \
    --nmax-per-dim 800000 --neff 2000 --warm-cases off --run-seed 989654
```
```
sampler    target               n_eff     n_ESS   JSmax  |pull| widthdev lnZbias  verdict
GMM        mix_d4_n2_s101         148      4181  0.0082   0.061    0.146  -0.058  FAIL  [width_ratio[1]=0.854 (tol 0.055)]
```

`python3`, not `python`: several IGWN/conda environments provide only the former, the same reason
`run_shape_recovery.sh` says so at its `exec` line.

The `PYTHONPATH` line is load-bearing -- without it you measure whichever RIFT is **installed**, not
the branch. On this cell at `quick`'s budget and run seed 987654 that is the difference between
n_eff 42.3 (branch) and 4.6 (CVMFS igwn), which is a different experiment wearing the same verdict
column. `run_shape_recovery.sh` exports it; the pytest entry point it advertises as equivalent does
not. Branch `claude/shape-recovery-pytest-checkout-guard` closes that hole -- it makes the mismatch
raise instead of measuring silently -- and lands its own FOLLOWUPS entry as **item 6**, which is why
this one is numbered 7.

On `MixtureTarget(4, 2, 101)` the GMM sampler recovers dimension 1's marginal **too narrow, and
progressively more so the longer it runs**, while the other three dimensions stay exact. Medians
over 8 fresh run seeds per budget:

```
budget   med n_eff   median width_ratio per dim (x0 x1 x2 x3)   median mean_pull[1]
 x1  200k     50     1.005  0.989  1.002  1.011                  +0.023
 x2  400k     32     1.001  0.976  0.981  0.993                  -0.003
 x3  600k     37     1.001  0.923  0.985  0.998                  +0.039
 x4  800k     44     1.000  0.919  0.993  0.998                  +0.039
 x8  1.6M     81     1.003  0.906  0.992  1.003                  +0.052
x16  3.2M     96     1.003  0.900  0.993  0.994                  +0.046
x32  6.4M    139     1.008  0.889  0.989  1.004                  +0.069
```

n_eff grows roughly as `nmax**0.3` instead of linearly, and every FAIL in the sweep names
`width_ratio[1]` (0.843-0.915); at x32 `mean_pull[1]` and `corr_diff_max` join it as the tolerances
tighten with n_eff. The evidence bias stays small (median |bias| 0.03-0.07 nats) throughout, so
**the integral looks fine while the posterior does not** -- which is the failure mode this whole
suite exists to catch, and is invisible to `.travis/test-integrate.sh`.

**The target is not pathological, and the thresholds are not too tight.** AV on the identical
target, same seeds:

```
budget   n_eff min/med/max   clears 100   PASS   median width_ratio per dim
 x1     157 /  188 /  203       8/8       8/8    1.001  1.004  1.000  0.996
 x4     721 /  770 /  802       8/8       8/8    1.000  1.004  1.002  1.000
x16    2002 / 2008 / 2013       8/8       8/8    1.000  1.000  0.999  0.999
```

AV converges TO 1.000 on the dimension GMM diverges from, and clears the floor at every seed and
every budget including `quick`'s.

**It is target-specific, not general GMM.** Sweeping GMM over the six `standard` d=4 cells at x1
and x4 (8 seeds each): every one scales n_eff ~4x for a 4x budget, holds `width_ratio` within 0.99
-- 1.01 in all dims, and reaches 7-8/8 PASS at x4. Only `n2_s101` fails to scale (50 -> 44).

```
cell           x1 med n_eff -> x4      x4 PASS   x4 width_ratio per dim
d4 n1 s101         69 ->  279            7/8     1.001 1.003 1.002 1.001
d4 n1 s202         63 ->  252            8/8     1.004 1.005 1.004 1.003
d4 n1 s303         89 ->  357            8/8     1.002 0.999 0.999 1.000
d4 n3 s101         74 ->  297            8/8     1.002 0.994 0.994 0.992
d4 n3 s202        182 ->  719            8/8     1.000 0.998 0.996 1.000
d4 n3 s303         62 ->  222            7/8     1.008 1.007 1.003 1.000
d4 n2 s101         50 ->   44            1/8     1.000 0.919 0.993 0.998   <-- this item
```

Note the ncomp=3 cells are fine, so plain multimodality is not the trigger.

**Geometry of the one target that breaks it.** Two near-equal-weight components (0.479 / 0.521),
separated 1.83 sigma along x0, whose widths along x1 differ by 2.5x (component sd 1.302 vs 0.520)
with only 0.95 sigma of separation in that dimension. The recovered pull is positive, i.e. toward
the NARROW component's x1 mean (0.062) and away from the broad one's (-0.985). So the proposal is
progressively abandoning the broad component's tail and concentrating on the sharper, higher-density
one -- and `n_eff = sum(w)/max(w)` cannot see it, because a proposal that has collapsed onto part
of the support still reports concentrated weights.

**Next steps.** Establish whether the trigger is the width ratio between overlapping components or
the near-equal weights, by scanning `sigma_1d` / weight ratio on a synthetic two-component target
rather than hunting more random seeds. Then check whether the fit is degenerate at the source: this
runs `n_comp=2` on a genuinely 2-component target, so the EM fit is correctly specified and should
not need to collapse -- if both fitted components land on the narrow mode, the defect is in the
initialization or in the tempered-refit path (`GMM refit skipped: ESS too low even untempered`
fires on some seeds here), not in the component count.

**On adding coverage.** Two shapes are available once the scan above says what the trigger is, and
they answer different questions:

* *Gate it* -- put a ncomp=2 row into `standard` at a budget above the floor. This is the honest
  end state, but it makes the merge gate red on every branch until the defect is fixed, so it
  belongs with the fix, not before it.
* *Characterize it* -- a non-blocking regression test pinning `width_ratio[1]` to the measured band
  at a stated budget and seed, so that a fix, or a worsening, is visible instead of silent. Cheaper,
  and it does not hold merges hostage -- but it freezes current behaviour into the suite, so it
  needs an explicit expiry: it exists to be deleted by whoever fixes the defect.

**Do not** reach for a budget increase, and do not add this cell to `standard` to "make it gated"
until the defect is understood -- that turns the quick smoke row red without telling anyone anything
the table above has not already established. See item 5 for what the retained row does and does not
buy.
