# PROPOSAL: warm-start / bootstrap cases for the shape-recovery merge gate

Status: **proposal only** — nothing in this directory has been modified.  Every number below was
measured on this branch (`rift_O4d_portfolio_freeze_tuning`) on CPU
(`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`,
`/cvmfs/software.igwn.org/conda/envs/igwn-py310`, numpy 1.24.4), using the gate's own
`MixtureTarget`, `build_sampler`, `shape_metrics` and `evaluate`.

(Note on location: the task brief called this directory `demos/<gate dir>`.  The gate actually lives
at `MonteCarloMarginalizeCode/Code/test/expensive_before_merging/integrators/`; `demos/` only holds
`integrator_snr_lottery`.  This file is placed with the gate it modifies.)

---

## 1. How the existing gate specifies and scores a case

### Case specification
`shape_recovery.py` builds a pure product matrix.  A "case" is a 4-tuple

    (sampler kind, ndim, ncomp, target_seed)

* `PRESETS[preset]` gives `dims x ncomps x seeds x nmax_per_dim x neff`
  (`quick`: dims 2,4 / ncomps 2 / seeds 101 / nmax_per_dim 50000 / neff 2000;
  `standard`: dims 2,4,6,8 / ncomps 1,3 / seeds 101,202,303 / nmax_per_dim 200000 / neff 3000).
* `main()` expands the product over `--samplers` into jobs
  `(kind, (d, nc, ts), nmax = nmax_per_dim*d, neff, run_seed)` and dispatches them through
  `_worker` (spawn `multiprocessing.Pool`, so anything monkey-patched in the parent is lost —
  see `probe_portfolio_optin_flags.py`).
* `MixtureTarget(ndim, ncomp, seed)` is fully determined by the seed: weights `U(0.1,1.1)`
  normalised, means `U(-3/sqrt(d), 3/sqrt(d))`, Wishart covariances around `sigma_1d=0.7`, all on
  the box `[-5,5]^d`.  `true_lnZ = LNL_OFFSET + ln(in-box mass) - sum ln(width)` from a 10^6-point
  rejection-sampled truth pool.
* `run_one` drives the sampler through its production API with
  `n=n_chunk, n_adapt=100, floor_level=0, tempering_exp=0.1, neff, nmax, save_intg=True`,
  reads the weighted cloud back out of `sampler._rvs`, and **never raises** — exceptions land in
  `record["error"]`.

### Scoring — `evaluate(record)` returns one of `PASS / FAIL / STARVED / ERROR`
1. `error` set                       -> `ERROR`
2. `n_eff < MIN_NEFF_FOR_SHAPE (100)` -> `STARVED` **and nothing else is checked**
3. otherwise all of, per dimension `d`:
   * `JS[d] < JS_MULT*floor[d] + JS_ABS_MIN` (3.0 x self-calibrated floor + 0.004), where the floor
     is JS(truth subsample at this run's own n_ESS, truth pool), mean + 2 sd over 5 subsamples;
   * `|mean_pull[d]| <= max(5/sqrt(n_ESS), 0.05)`;
   * `|width_ratio[d] - 1| <= max(5/sqrt(2 n_ESS), 0.05)`;
   * `corr_diff_max <= max(8/sqrt(n_ESS), 0.08)`;
   * `|lnZ_hat - true_lnZ| <= max(4*rel_err, 0.10)`.
4. `main()` exits 1 iff a **strict** sampler (`--strict-samplers`, default `AV,GMM`) scored `FAIL`.
   `STARVED` never sets the exit code; `WARN` (non-strict FAIL) never sets it.

### Base-vs-candidate — `compare_shape_results.py`
Records are paired on `(kind, target)`.  Blocking verdicts, for strict kinds only:
* `PASS` on base -> anything else on candidate = `REGRESSION(pass->...)` — this *includes*
  `PASS -> STARVED`;
* both `PASS` but a summary metric worsens by more than `TOL_WORSE`
  (`js .005, mean_pull .05, width_dev .05, corr .05, bias_ln .10`) or
  `n_eff_cand < 0.5 * n_eff_base` = `REGRESSION(metrics)`.
`STARVED -> STARVED` is `BOTH-STARVED` and **never blocks**; `STARVED -> PASS` is `IMPROVED`.
The script needs no code change to accept new kinds — it keys on `record["kind"]` generically.

---

## 2. What the gate does not exercise, and what actually catches it

Two independent defects, both of which produce *no exception and no obviously bad diagnostic*:

**(a) warm seed stored but never installed on the draw path.**  `bootstrap_from_*` only writes
`self._warm`; historically only `mcsamplerAdaptiveVolume.integrate_log` consumed it.  A PORTFOLIO
calls `member.draw_simplified()` directly and never runs the member's `integrate_log`, so the AV
member drew from the cold single-bin grid.  `mcsamplerAdaptiveVolume._apply_warm_state()` (called
from `draw_simplified`) is the fix.

**(b) stale contracted live volume leaking between sequential points.**
`mcsamplerPortfolio.integrate_log` has `self.setup()` **commented out** (mcsamplerPortfolio.py
~L874), so member state survives the call.  Point 2 therefore inherits point 1's contracted AV
live volume.  `mcsamplerPortfolio.clear_warm_state()` (and the driver's `_clear_warm_state()`
helper, `bin/integrate_likelihood_extrinsic_batchmode` L1943) is the fix.
Standalone AV is immune: `AV.integrate_log` calls `self.setup()` on entry.

### Measurement 1 — bug (a) is bit-for-bit invisible
Emulating the pre-fix code (`member._warm_applied = True` right after seeding, so
`_apply_warm_state()` is a no-op) and seeding **only the AV member**, the run reproduces the cold
run to the last printed digit.  15/15 pairs identical, e.g. d4 nc1 ts101 rs987654:

| mode | n_eff | lnZ bias | n_eval |
|---|---|---|---|
| cold           | 46.3 | -0.0209 | 200000 |
| warm_av_INERT  | 46.3 | -0.0209 | 200000 |
| warm_av (fixed)| 45.3 | -0.1398 | 200000 |

### Measurement 2 — a black-box "warm beats cold" assertion does NOT test the AV path
The gate's portfolio is AV + GMM, and `mcsamplerEnsemble` *also* has `bootstrap_from_samples`
(added in 8c29876b).  `portfolio.bootstrap_from_samples` seeds both, and the GMM member captures
essentially the whole win.  d=6, ncomp=1, `nmax=6e5`, `neff=3000`, 3 target seeds x 4 run seeds,
scored with the gate's own `evaluate()`:

| target seed | fixed n_eff | INERT n_eff (bug (a) present) | verdict, both |
|---|---|---|---|
| 101 | 5504 – 5941 | 5078 – 5972 | PASS |
| 202 | 3240 – 3439 | 5310 – 5623 | PASS |
| 303 | 5360 – 5787 | 4692 – 5540 | PASS |
| **all 12** | **3240 – 5941** | **4692 – 5972** | **12/12 PASS both** |

The buggy configuration is sometimes *better*.  **No n_eff / lnZ / shape assertion can separate
these.**  What does separate them, exactly and with zero RNG dependence, is the AV member's live
volume after a single `portfolio.draw()`:

| | AV member `V` | AV member live bins |
|---|---|---|
| fixed  | 0.0313 – 0.129 (d6), 0.0432 (d2), 0.0862 – 0.159 (d4) | 47 – 656 |
| INERT  | **1.000 exactly** | **1 exactly** |

Behavioural form of the same statement — fraction of AV-member draws landing inside the seed
cloud's bounding box, versus the uniform-box expectation `V_unif` (10000 draws, 2 target seeds x
2 run seeds each):

| d | V_unif | fixed fraction | ratio | INERT ratio |
|---|---|---|---|---|
| 2 | 0.336 / 0.187 | 0.729 – 0.764 | 2.17 / 4.09 | 0.93 – 1.01 |
| 4 | 0.0489 / 0.0311 | 0.240 – 0.289 | 4.9 / 9.2 | 0.95 – 1.06 |
| 6 | 0.00350 / 0.0210 | 0.072 – 0.139 | 6.1 – 22.3 | 0.86 – 1.09 |

Run-seed scatter of the fraction is < 0.005 absolute (binomial, n=10^4).

### Measurement 3 — bug (b) is enormous, and does not even need a warm start
Two displaced targets A (`offset=-2`) and B (`offset=+2`), both `scale_x0=1.0`, integrated
sequentially on ONE portfolio, d=2, ncomp=1, `nmax=1e5`, `neff=2000`, 3 target seeds x 5 run seeds:

| between-point handling | B n_eff (min–max) | \|B lnZ bias\| (median, max) |
|---|---|---|
| `sampler._warm = None` only (pre-fix driver) | **0.0 – 3.9** | 16.8, **337** |
| `clear_warm_state()` (post-fix)               | 799.8 – 2089.1 | 0.0038, 0.0248 |
| fresh sampler for B (reference)               | 819.1 – 2082.3 | 0.0054, 0.0124 |

With **no bootstrap at all** (pure sequential reuse, same displaced pair): leak B n_eff 0.0 – 9.9,
|bias| up to 24.2; `clear` 1587 – 2094, |bias| <= 0.019.  The leak is the run-contracted grid, not
the seed — so this case guards `--n-events-to-analyze > 1` even for users who never warm-start.
Standalone `AV` shows leak == cold bit-for-bit, confirming the defect is portfolio-only.

Scored with the gate's own `evaluate()` (4 run seeds x 3 target seeds):

| mode (12 runs each) | verdict | n_eff | \|bias\| | JSmax | width dev | max pull |
|---|---|---|---|---|---|---|
| clear | 12/12 PASS    | 815.8 – 2103.4 | 0.0004 – 0.0188 | <= 0.0003 | <= 0.010 | <= 0.017 |
| LEAK  | 12/12 STARVED | 0.0 – 12.8     | 0.20 – 318.5    | <= 0.69   | <= 1.00  | <= 6.78  |

---

## 3. Proposed cases

Two new sampler kinds, so `compare_shape_results.py` pairs them automatically.

### Case W — `portfolio_warm` (guards bug (a) + "the warm-start feature went inert")
* target: `MixtureTarget(ndim=6, ncomp=1, seed in {101, 202, 303})`, unmodified.
* seed cloud: 3000 fair draws from the target's own truth pool, `RandomState(target.seed+13)`.
* budget: `nmax = 6*100000`, `neff = 3000`, `n_chunk = 10000`.
* procedure:
  1. probe sampler: build portfolio, `setup()`, `bootstrap_from_samples(cloud, cover_frac=0.0)`,
     one `draw(n_chunk)`; record `warm_V`, `warm_bins` from `portfolio_realizations[0]`, and
     `warm_box_frac` / `warm_box_frac_uniform`.
  2. fresh sampler, same seeding, `integrate_log` -> the record scored by the normal metrics.
* assertions (all hard; this kind is STRICT and STARVED is promoted to FAIL):
  * **A1 (install)** `warm_V < 0.9` and `warm_bins > 1`.  Measured over 12 runs: fixed
    `V in [0.0313, 0.1292]`, `bins in [489, 656]`; buggy `V = 1.000` and `bins = 1` in 12/12.
    No RNG enters `V` or `binunique` — the margin is categorical, not statistical.
  * **A2 (behavioural install)** `warm_box_frac >= 3 * warm_box_frac_uniform`.  Measured at d6:
    6.1x – 22.3x; buggy: 0.86 – 1.09.  Margin >= 2x on the worst measured seed.
  * **A3 (feature-level)** `n_eff >= 1000`.  Measured warm 2426 – 5941 (5 target seeds x 5 run
    seeds at neff=2000, plus 3 x 4 at neff=3000); cold at the same budget 3.8 – 91.7.  2.4x margin
    below the warm minimum, 11x above the cold maximum.
  * normal JS / pull / width / corr / lnZ checks (measured warm: JSmax <= 0.0008,
    width dev <= 0.026, |bias| <= 0.0038; 12/12 PASS).

  A3 does **not** isolate the AV path (Measurement 2); it catches "all warm-start channels went
  inert", which is a real and separate regression.  A1/A2 are what catch bug (a).  A1 is
  deliberately white-box: bug (a) has no statistical signature, so a purely black-box gate cannot
  see it, and pretending otherwise would give a case that never fires.

### Case S — `portfolio_seq` (guards bug (b))
* targets: `A = MixtureTarget(2, 1, ts, offset=-2.0, scale_x0=1.0)`,
  `B = MixtureTarget(2, 1, ts, offset=+2.0, scale_x0=1.0)`, `ts in {101, 202, 303}`.
  Mean separation 4.0 with `sigma_1d = 0.7` -> B's mass is far outside A's contracted volume, and
  both stay inside the `[-5,5]^2` box (`scale_x0=1.0` keeps the random means within +-1).
* budget: `nmax = 1e5`, `neff = 2000`, `n_chunk = 10000`, one portfolio reused.
* procedure: seed from A's truth pool, integrate A, `sampler._rvs = {}`, then
  `sampler.clear_warm_state()` **if present else `sampler._warm = None`** (so the case also RUNS on
  a base branch that lacks the API and correctly fails there), then integrate B.  The record is
  point B, scored against `B.true_lnZ` and B's truth pool.
* assertions (hard; STRICT, STARVED promoted to FAIL):
  * **B1** `n_eff >= 100`.  Measured clear 799.8 – 2103.4 over 27 runs (>=8x margin); leak
    0.0 – 12.8.  No overlap.
  * **B2** `|lnZ_hat - true_lnZ_B| <= 0.10` (the gate's existing floor).  Measured clear <= 0.0248
    (4x margin); leak 0.20 – 318.5 in the 12 scored runs (median 16.8 over the wider 15-run set).
  * normal shape checks (measured clear: JSmax <= 0.0003, width dev <= 0.010, pull <= 0.017).
* optional companion row `portfolio_seq_nobs` — identical but with no bootstrap at all, which
  isolates the run-contraction leak from the seed.  Measured clear 1587 – 2094; leak 0.0 – 9.9.
  Cheap (same cost) and strictly more informative; recommended.

### Negative control (recommended, ~free)
`AV_seq`: the same sequential construction with the standalone AV sampler at d=2, which must be
unaffected.  Measured leak == cold **bit-for-bit** (n_eff 2616 – 3028, |bias| 0.0075 – 0.043 in
both).  Keep it warn-only: it documents that the defect is portfolio-specific.
Do **not** extend it to d=4 — AV alone on the displaced pair has |bias| 0.27 – 0.47 there and would
fail its own lnZ tolerance for reasons unrelated to this feature.

---

## 4. Cases considered and REJECTED as too flaky

**R1. "warm-from-the-correct-target beats cold" as the test for bug (a), on the AV member only.**
Seeding only the AV member does isolate the path (the inert variant reproduces cold bit-for-bit),
but the effect is not reliably positive.  15 runs per cell (3 target seeds x 5 run seeds):

| d | cold n_eff (med, min–max) | warm_av n_eff | cold \|bias\| med | warm_av \|bias\| med |
|---|---|---|---|---|
| 2 | 1659 (817 – 2062) | 2250 (2021 – 2334) | 0.0024 | 0.0035 |
| 4 | **59.5** (19.8 – 117) | **47.6** (16.7 – 111) | 0.022 | **0.043** |
| 6 | 21.6 (3.8 – 91.7) | 35.8 (9.2 – 86.9) | 0.041 | **0.098** |

At d=4 the warm run is *worse* on both n_eff and bias; at d=6 n_eff improves but bias degrades.
Any threshold that passes d=2 fails d=4 on some seeds.  Rejected.

**R2. Case S at d=4.**  `clear` gives B n_eff 23.7 – 125.2, straddling the `STARVED` floor of 100,
so the verdict flips on the target seed alone.  Worse, the discriminant collapses: at ts101 the
*leak* run gives n_eff 16.5 – 25.1 with |bias| <= 0.16, which overlaps the *clear* run at ts202
(23.7 – 38.6, |bias| <= 0.09).  Rejected.

**R3. Case S at d=6.**  `clear` gives B n_eff 6.1 – 91.6 — always STARVED even when the code is
correct — while leak gives 1.0 – 16.6 with |bias| as low as 0.058.  The case would report
`BOTH-STARVED` (non-blocking) on every branch.  Rejected.

**R4. A pure lnZ-bias assertion for case S.**  The leak's bias is heavy-tailed, not uniformly
large: at d=2 ts101 rs987654 the leaked run had |bias| = 0.025, inside the gate's 0.10 tolerance,
while n_eff was 2.1.  `n_eff` is the reliable discriminant; bias is the corroborating one.  Keep
both, but do not rely on bias alone.

**R5. Warm-vs-cold n_eff ratio (instead of the absolute floor A3).**  Cold n_eff at d6 ncomp=1
ranges 3.8 – 91.7 across seeds, a factor 24 — a ratio threshold inherits that scatter and doubles
the runtime by requiring a paired cold run.  The absolute floor (`n_eff >= 1000`, warm min 2426,
cold max 91.7) is both cheaper and tighter.

---

## 5. Diff-sized changes

### 5.1 `shape_recovery.py`

**(i) displaced targets — the only change to `MixtureTarget` (6 lines).**  `shape_recovery.py`
cannot currently express a displaced pair: `MixtureTarget` exposes `sigma_1d` and `scale_x0` but no
translation, and two different seeds give random, typically overlapping mean offsets (|mean| <=
3/sqrt(d) with sigma ~ 0.9 at d=2), so mode displacement cannot be guaranteed from seeds alone.

```python
    def __init__(self, ndim, ncomp, seed, sigma_1d=0.7, scale_x0=3.0, offset=0.0):
        ...
        self.offset = np.zeros(ndim) + np.asarray(offset, dtype=float)
        if np.any(self.offset):
            self.name += "_o{:+.2f}".format(float(np.mean(self.offset)))
        ...
        for k in range(ncomp):
            x0 = rng.uniform(-scale_x0/np.sqrt(ndim), scale_x0/np.sqrt(ndim), ndim) + self.offset
```

Adding `offset` AFTER the `rng.uniform` draw keeps the RNG stream identical, so `offset=+a` and
`offset=-a` are the same mixture translated — exactly the "same shape, displaced support"
construction the case needs.  `pool` and `true_lnZ` follow automatically (they are derived from
`means`/`covs`).  Default `0.0` leaves every existing target and every existing `name` bit-identical.

**(ii) new kinds + explicit case list (~90 lines, additive).**

```python
WARM_KINDS = ("portfolio_warm", "portfolio_seq", "portfolio_seq_nobs", "AV_seq")
STARVE_IS_FAIL = ("portfolio_warm", "portfolio_seq", "portfolio_seq_nobs")
WARM_NEFF_FLOOR = 1000.0      # case W A3; measured warm 2426-5941, cold 3.8-91.7
WARM_V_MAX      = 0.9         # case W A1; measured fixed 0.031-0.129, buggy exactly 1.0
WARM_BOX_MULT   = 3.0         # case W A2; measured 6.1-22.3x at d6, buggy ~1.0x

WARM_CASES = [   # (kind, ndim, ncomp, tseed, nmax, neff, extra)
    ("portfolio_warm",      6, 1, 101, 600000, 3000, {}),
    ("portfolio_warm",      6, 1, 202, 600000, 3000, {}),
    ("portfolio_warm",      6, 1, 303, 600000, 3000, {}),
    ("portfolio_seq",       2, 1, 101, 100000, 2000, dict(offset=2.0, scale_x0=1.0)),
    ("portfolio_seq",       2, 1, 202, 100000, 2000, dict(offset=2.0, scale_x0=1.0)),
    ("portfolio_seq",       2, 1, 303, 100000, 2000, dict(offset=2.0, scale_x0=1.0)),
    ("portfolio_seq_nobs",  2, 1, 101, 100000, 2000, dict(offset=2.0, scale_x0=1.0)),
    ("AV_seq",              2, 1, 101, 100000, 2000, dict(offset=2.0, scale_x0=1.0)),
]
```

* `_warm_seed_cloud(target, n=3000)` -> `target.pool[RandomState(target.seed+13).choice(...)]`.
* `run_warm_case(...)`: probe sampler (build / `setup` / `bootstrap_from_samples(cloud,
  cover_frac=0.0)` / one `draw(n_chunk)`) recording `warm_V`, `warm_bins`, `warm_box_frac`,
  `warm_box_frac_uniform`; then a fresh sampler, same seeding, `integrate_log`, then the existing
  `shape_metrics` + record assembly.
* `run_seq_case(...)`: build `A` (offset `-o`) and `B` (offset `+o`); integrate A;
  `sampler._rvs = {}`; `sampler.clear_warm_state()` if present else `sampler._warm = None`;
  integrate B; record B.  `AV_seq` uses `build_sampler("AV", ...)`; `portfolio_seq_nobs` skips the
  bootstrap.
* `run_one` dispatches on `kind in WARM_KINDS` before its existing branch chain; both helpers keep
  the never-raise contract (wrap in the same `try/except` that fills `record["error"]`).
* `main()`: `--warm-cases {auto,on,off}` (default `auto` = on for `--preset standard`, off for
  `quick`), appending `WARM_CASES` jobs to `jobs` after the product expansion.

**(iii) `evaluate()` — 3 additive blocks, no change to existing behaviour.**

```python
    if r["kind"] in STARVE_IS_FAIL and r["n_eff"] < MIN_NEFF_FOR_SHAPE:
        return "FAIL", ["n_eff={:.0f} < {:.0f}: warm/sequential case must not starve"
                        .format(r["n_eff"], MIN_NEFF_FOR_SHAPE)]
    if r["n_eff"] < MIN_NEFF_FOR_SHAPE:          # unchanged
        return "STARVED", [...]
    ...
    if r["kind"] == "portfolio_warm":
        if not (r.get("warm_V", 1.0) < WARM_V_MAX and r.get("warm_bins", 1) > 1):
            reasons.append("warm seed NOT installed on the draw path: AV member V={:.3f}, "
                           "live bins={} (cold state)".format(r.get("warm_V"), r.get("warm_bins")))
        if r.get("warm_box_frac", 0) < WARM_BOX_MULT * r.get("warm_box_frac_uniform", 1.0):
            reasons.append("warm draws not concentrated in the seed box: {:.3f} < {:.1f}x{:.4f}"
                           .format(r["warm_box_frac"], WARM_BOX_MULT, r["warm_box_frac_uniform"]))
        if r["n_eff"] < WARM_NEFF_FLOOR:
            reasons.append("warm n_eff {:.0f} < {:.0f}".format(r["n_eff"], WARM_NEFF_FLOOR))
```

### 5.2 `compare_shape_results.py`
No code change required (it keys on `record["kind"]`).  One default change:

```python
-    ap.add_argument("--strict-samplers", default="AV,GMM")
+    ap.add_argument("--strict-samplers",
+                    default="AV,GMM,portfolio_warm,portfolio_seq,portfolio_seq_nobs")
```

Note the intended base-vs-candidate behaviour: `portfolio_seq` FAILs on a base branch without
`clear_warm_state` and PASSes here, i.e. `IMPROVED(fail->pass)` — non-blocking, as intended.  Its
value is forward-looking: once this branch is the base, any change that re-breaks the clearing
gives `REGRESSION(pass->fail)` and blocks.

### 5.3 `run_shape_recovery.sh`
```sh
-exec python "${HERE}/shape_recovery.py" --preset standard --jobs "${SHAPE_JOBS:-8}" \
-    --json "${OUT}" "$@"
+exec "${PYTHON:-python3}" "${HERE}/shape_recovery.py" --preset standard \
+    --jobs "${SHAPE_JOBS:-8}" --warm-cases auto --json "${OUT}" "$@"
```
The `python` -> `python3` change is unrelated to this proposal but is a live foot-gun: this host has
no `python` on PATH, so the wrapper fails immediately.

### 5.4 `test_shape_recovery.py`
Optional: add a second parametrisation over `WARM_CASES` so the new cases also appear under pytest.
Keep them out of the existing `_MATRIX` (which is a strict-sampler x preset product).

---

## 6. Runtime

Measured single-threaded on this host (`OMP_NUM_THREADS=1`):

| item | cost |
|---|---|
| truth pool (10^6 draws) | 5.0 s at d=2, 2.3 s at d=6 |
| case W (probe draw + warm integrate, terminates in 1-2 chunks) | 3.0 – 5.3 s per (seed) after pool |
| case S (2 pools + 2 integrations) | 7.1 – 7.7 s per (seed) after the first |

Full proposed set (3 W + 3 S + 1 S_nobs + 1 AV_seq = 8 cases): **~70 s of serial CPU**, ~15 s wall
at `--jobs 8`.  The `standard` preset is 96 runs, many at 200k – 1.6M evaluations, so the added
cost is well under 2% of the gate.  Nothing here needs a GPU; both cases are pure-CPU deterministic
in the same sense as the rest of the suite.

---

## 7. Honest summary

* Bug (b) is cheap, deterministic and worth gating: a 3-orders-of-magnitude n_eff separation with
  no seed overlap in 15/15 runs at d=2.  Take case S.
* Bug (a) **cannot** be caught by any statistical assertion on the gate's AV+GMM portfolio, because
  the GMM member's warm start supplies nearly the whole win and the buggy configuration sometimes
  scores better.  The only reliable detector is the direct one: after seeding, the AV member's live
  volume must actually be contracted on the draw path (A1/A2).  Take that, and do not dress it up
  as a statistical test.
* The obvious-looking "warm must beat cold" assertion is fit for A3 only (all-channels-inert), not
  for bug (a), and is unusable in the AV-isolated form (R1).
* Case S must be pinned at d=2.  At d=4 and d=6 the correct behaviour is itself starved and the
  verdict becomes seed lottery (R2, R3).

---

## 8. Corrections found when implementing this (2026-08-05)

The proposal above was implemented as written and then each bug was **reintroduced** to check the
cases actually fire. Two of the proposed assertions did not survive that check. Both are corrected
in `shape_recovery.py`; this section records what was measured, not what was expected.

**C1. `portfolio_seq` does NOT catch the leak — `portfolio_seq_nobs` does.**
With `clear_warm_state()` no-op'd, `portfolio_seq` (which re-bootstraps on point B) measured
**PASS, n_eff 5979, bias -0.008**: the fresh B seed simply overwrites the stale contracted grid, so
the leak never manifests. `portfolio_seq_nobs` with the same injection:

| target seed | leak n_eff | leak lnZ bias | correct n_eff | correct bias |
|---|---|---|---|---|
| 101 | 9.9 | -0.559 | 2094 | +0.019 |
| 202 | 1.0 | -22.811 | 1593 | -0.000 |
| 303 | 1.0 | -59.667 |  822 | +0.002 |

So the case list now runs `portfolio_seq_nobs` at **all three** target seeds and keeps a single
`portfolio_seq` row, whose only job is to cover the reseed-after-reset path.

**C2. A2 measured on the portfolio mixture is not a discriminant; it must be measured on the AV
member's own draws.** With the AV install disabled, the *mixture* still concentrated **28x** in the
seed box, because the GMM member is warm-started through a separate channel. Re-measuring A2 from
`av.draw_simplified(n_chunk)` gives a clean separation:

| | ts101 | ts202 | ts303 |
|---|---|---|---|
| installed (V, bins, box ratio) | 0.042, 656, **21.0x** | 0.032, 498, **27.2x** | 0.129, 529, **6.3x** |
| inert (V, bins, box ratio) | 1.000, 1, **1.0x** | 1.000, 1, **0.9x** | 1.000, 1, **0.9x** |
| n_eff installed / inert | 3159 / **4857** | 3368 / **5718** | 5707 / **5532** |

The threshold `WARM_BOX_MULT = 3.0` sits between 1.0x and 6.3x. The n_eff row is the important
one: **the broken code scores HIGHER n_eff in 2 of 3 seeds**, which is the direct confirmation of
the proposal's Measurement 2 — no statistical assertion can catch this bug, only A1/A2.

**Measured added runtime:** all 8 warm cases complete inside a 26 s wall-clock run at `--jobs 4`
(quick preset, single-threaded BLAS).
