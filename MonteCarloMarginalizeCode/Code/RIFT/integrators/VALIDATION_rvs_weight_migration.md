# Validation plan: migrating the weight path to `rec.log_weights()`

The remaining step of option A is to move consumers off
`ln_weights_from_rvs(rvs, use_lnL=...)` and onto `rec.log_weights()`. That is what finally
removes the reason for `use_lnL` to exist, and lets `return_lnI` become historical.

It is also the step that touches the number every science product is built from, so this is the
plan **before** the change, not after. Nothing here has been started.

## The key measured fact

**At a fixed `--run-seed`, the shape gate is deterministic to the bit.** Two runs of
`shape_recovery.py --preset quick --samplers AV,GMM --dims 2 --ncomps 2 --target-seeds 101
--run-seed 987654` on identical code differ in exactly one field:

| field | run A | run B |
|---|---|---|
| `wallclock` | 8.455 | 3.753 |
| `js`, `n_eff`, `lnI`, `mean_pull`, `width_ratio`, `corr_diff_max`, `bias_ln`, `rel_err`, `n_ess`, `n_eval` | **identical** | **identical** |

That was worth checking rather than assuming: the first comparison reported "DIFFER" and looked
like it had killed this whole approach, until the diff turned out to be the timer.

**So the acceptance criterion for a pure refactor is BIT-IDENTITY, not "within tolerance."**
That is far more sensitive than the gate's own thresholds (`TOL_WORSE` js 0.005, pull 0.05,
width 0.05) and it removes the stochastic-flip problem `run_shape_recovery.sh` warns about at
length -- its `--confirm-repeats` machinery exists for cells sitting on the `n_eff >= 100` floor,
and a refactor should never produce a differing cell at all. **Any** non-`wallclock` difference
is a signal, and should be treated as one rather than compared against a tolerance.

## Tiers, cheapest first

### 0. Bit-identity on the shape gate (the main event)

```
run_shape_recovery.sh <base checkout>  base.json
run_shape_recovery.sh <cand checkout>  cand.json
# then compare ALL metric fields for exact equality, ignoring wallclock
```

`compare_shape_results.py` applies tolerances, which is right for a behaviour change and too
weak here. For a refactor, compare exactly. Fall back to `compare_shape_results.py
--confirm-base-checkout ... --confirm-cand-checkout ... --confirm-repeats 5` only if a cell does
differ and the question becomes whether the difference is real.

### 1. Independent-route cross-check (the falsification)

`shape_recovery.py` carries **its own** `log_weights_from_rvs()` -- a third implementation,
independent of both `ln_weights_from_rvs` and `RvsRecord.log_weights()`, written to be "tolerant
of the heterogeneous `_rvs` conventions". Assert the three agree on the same records, per
backend, including `mcsamplerEnsemble` in **both** `use_lnL` modes.

This is the check that can actually falsify the migration, as opposed to testing it against
itself.

### 2. Fast integrator CI

- `.travis/test-integrate.sh` -> `test/test_mcsamplerEnsemble_extended.py` (AC/GMM/AV recover
  a known integral to ~1.0)
- `test_fairdraw_double_weighting.py`, `test_seq_warmstart_seed.py`, `test_l0_rescue_seed.py`,
  `test_av_empty_live_volume.py`, `test_portfolio_fairdraw_backend.py`, `test_rvs_record.py`
- both audit gates (`audit_rvs_fairdraw.py --check`, `audit_backend_contracts.py --check`)

### 3. Full ILE run

`.travis/test-run.sh` and `test-run-alts.sh` clone `ILE-GPU-Paper` and run
`make test_workflow_batch_gpu_lowlatency`, plus `test-coord.sh` / `test-posterior.sh`. Needs
network access and is GPU-shaped; on CIT it must run on a **different host from the session**,
one campaign per host, with `OMP_NUM_THREADS=1`.

## Sequencing, and the one trap in it

**`shape_recovery.py` is itself an `_rvs` consumer** -- it reads `s._rvs` directly and derives
weights with its own helper. So it is both the ruler and a migration target.

**Migrate the ILE weight path first; validate with the shape gate UNCHANGED; migrate the gate
only afterwards, as its own step with its own before/after.** Changing the ruler and the thing
being measured in one commit destroys exactly the independence that makes tier 1 worth anything.

## What would make me stop

* a cell differs and `--confirm-repeats 5` says the difference is real -> the migration is not a
  refactor, and the change is wrong until that is explained;
* the three weight implementations disagree anywhere, in either Ensemble mode;
* tier 3 cannot be run at all -> say so plainly and mark the migration provisional rather than
  shipping on tiers 0-2 and calling it validated.

---

# RESULTS (2026-08-14), migration of `ln_weights_for_posterior`

Base `1dcabd27`. Command, both arms identical:

```
shape_recovery.py --preset quick --samplers AV,GMM,AC,portfolio \
    --dims 2,4 --ncomps 1,2 --target-seeds 101,202 --run-seed 987654 --jobs 1
```

| tier | result |
|---|---|
| 0. shape gate bit-identity | **PASS** -- 32 cells, 19 metrics each, byte-identical apart from `wallclock` |
| 1. independent third implementation | **PASS** -- AV, Ensemble (both `use_lnL` modes), mcsampler |
| 2. fast CI + both audit gates | **PASS** -- 276 passed, 4 skipped; AC/GMM/AV recover the known integral to 0.939 / 0.997 / 0.933 |
| 3. full ILE run | **NOT RUN** -- needs network + GPU; see below |

## What the validation caught

**A real defect in the new API, before it shipped.** `log_weights()` was first written as
`log_likelihood() + log_prior() - log_sampling_prior()`. That is wrong on the linear column
family: `ln_weights_from_rvs` applies a **conjunctive** keep-mask (`ig>0 & jp>0 & js>0`, whole
row to `-inf`), while evaluating the three terms independently gives `-inf - (-inf) = NaN`. A
NaN weight poisons every downstream sum; `-inf` is a real zero. Found by fuzzing the two
implementations against each other **before** switching -- 1200 randomized records, systematic
divergence on both linear families.

**And a test that could not fail.** `test_three_independent_weight_implementations_agree` --
tier 1, the end-to-end check -- **passes with that defect reintroduced**, because real sampler
records have positive priors and never reach the masked rows. It is a decoration for this
defect. The test with teeth is the randomized one
(`test_log_weights_matches_the_canonical_form_including_out_of_support_rows`), which was
revert-checked: bug in -> FAIL, bug out -> PASS. Both are kept; only the second is load-bearing.

## Tier 3 is NOT discharged

`.travis/test-run.sh` / `test-run-alts.sh` clone `ILE-GPU-Paper` and run
`make test_workflow_batch_gpu_lowlatency`. That needs network egress and is GPU-shaped, and on
CIT must run on a different host from the session. **It has not been run.** Per the plan's own
stopping rule, this migration is therefore **provisional** until it has: tiers 0-2 are strong
evidence that the change is a pure refactor, but they do not exercise a real waveform, a real
PSD, or the GPU code path.

---

# ADVERSARIAL REVIEW (2026-08-14), and what it found

A self-review of the full branch diff produced **6 findings, one of which was a production
regression this change had introduced**. All six are fixed, each with a regression test.

**HIGH -- replicas on a linear backend would have DROPPED THE EVENT.** `_pool_replica_rvs`
keeps only the *intersection* of replica keys, so pooling `adaptive_cartesian` (or Ensemble
without `use_lnL`) replicas yields a bare `integrand` column. The ILE built the pooled record
with no `integrand_is_log`, so `log_weights()` correctly refused to guess -- and that
`ValueError` escaped the **unwrapped** `.dgrid` exporter, out of `analyze_event`, into the
per-event handler, which skips the event and writes an empty `.dat`. The convention is now
taken from the pre-pool record, falling back to `rvs_integrand_is_lnL`.

Note what this says about the tiers: **tier 0 was bit-identical before and after the fix**,
because the shape gate does not run the ILE at all, and neither the fast tests nor tier 1
exercise replica pooling on a linear backend. Bit-identity is a strong check of the code path
it covers and says nothing about the paths it does not.

The other five: the caller's `convert` was silently dropped on the record path; the pooled
record's block provenance was built from *unfiltered* replica lists while `_pool_replica_rvs`
filters in lockstep; `_sampler_keeps_records` tested "has a record right now" while being named
and documented as "participates at all"; the record restored after an L0 reject could never
match its columns and was therefore inert rather than belt-and-braces; and an orphaned comment
fragment sat at all seven rebind sites.

Tier 0 was re-run after the fixes: still **bit-identical** to base across all 32 cells.

---

# TIER 3 (2026-08-18): the full ILE run, DISCHARGED

Run on `ldas-pcdev12` (4x A100-SXM4-80GB), IGWN CVMFS python 3.11.14, cupy 12.0.0, lal 7.7.0.
Base = `364a22fd` (merge-base), candidate = `63b50062`, both commit-gated against the remote.
Raw data, scripts and the analysis are committed under
`test/expensive_before_merging/integrators/tier3/`.

## The thing that had to be established first: this run is NOT deterministic

Tier 0's acceptance criterion was bit-identity. **That criterion does not transfer here.** Two
runs of the *base* code, same `--seed 4242`, same host, same GPU:

| | run 1 | run 2 |
|---|---|---|
| `lnL` | 66.2279 | 66.5573 |
| `neff` | 22.46 | 4.19 |
| `sigma_lnL` | 0.1376 | 0.2637 |

The first base-vs-candidate comparison showed `dlnL` 0.11 and looked like a regression. It is
**smaller than the spread base shows against itself** (0.33). Had I stopped at the first diff I
would have reported a regression that does not exist; had I stopped at "it differs, GPU runs
differ, fine" I would have had no argument at all. So tier 3 is a comparison of DISTRIBUTIONS
with a MEASURED null, not a diff.

## Getting a real run at all: three dead ends, all pre-existing

The first four attempts failed, and none of the failures was mine -- **every one reproduces on
the unmodified base checkout**, which is the only reason they are not blockers:

1. `TypeError: ... argument 1 of type 'REAL8'` in `ComputeYlms`. **My option set was wrong**, not
   the code: without `--vectorized` the ILE takes the scalar loop at line ~3130 and hands an
   ARRAY of inclinations to a scalar `lal.SpinWeightedSphericalHarmonic`. `--force-xpy` does not
   help. `--vectorized --gpu` is the fix.
2. `--internal-use-lnL` + `adaptive_cartesian_gpu` (the DEFAULT sampler) dies in
   `mcsamplerGPU.integrate_log` mixing a numpy `maxval` into a cupy expression.
3. `--internal-use-lnL` + `adaptive_cartesian` (CPU) dies with `'MCSampler' object has no
   attribute 'identity_convert'`.
4. `portfolio` + `--internal-use-lnL` + replicas + `.dgrid` exits 1 with `'NoneType' object is
   not iterable`.

**(2) and (3) together mean the `.dgrid` export is currently UNREACHABLE on both
linear-integrand backends**, because the exporter is gated on `opts.internal_use_lnL`. That is
worth knowing independently of this branch: it bounds how much of the HIGH finding's blast radius
is reachable in production today. Only AV and GMM can emit a `.dgrid`. Filed separately; not
fixed here, because fixing them is not this branch's job and would have made the arms differ.

## The NoLoop path was PROVEN, not assumed

`noloop_probe.py` wraps the likelihood entry points and counts calls in a real run:

```
NOLOOP-PROBE: first call to DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop  time_interp='nearest' xpy=cupy
NOLOOP-PROBE COUNTS: {'...NoLoop': 20, '...NoLoopOrig': 0, 'FactoredLogLikelihoodTimeMarginalized': 0}
```

`xpy=cupy`, 20 calls, scalar path 0. Config D re-runs the whole ensemble with
`--interpolate-time True` for the cubic time interpolation as well.

## Design: 5 configs x 2 arms x 30 replicates = 300 runs, 300 clean

Arms are **interleaved within each replicate** so any drift in machine state hits both equally,
and `CUDA_VISIBLE_DEVICES` is pinned to an idle card (index 0 was at 100% from another user).

| cfg | what it exercises |
|---|---|
| A | GPU linear backend (`integrand` = L), plain |
| B | linear backend + **replica pooling** |
| D | cubic NoLoop time interpolation |
| AV | AV (lnL family) + pooling + **`.dgrid` export** |
| GMM | GMM (lnL family) + pooling + **`.dgrid` export** |

B/AV/GMM are the point: **tiers 0-2 were structurally blind to replica pooling** -- that is
exactly how the adversarial review's HIGH finding survived a bit-identical tier 0.

## Result

19 metric comparisons (lnL, sigma_lnL, neff, and the `.dgrid` grid statistics), two-sided
**permutation test** on the arm labels (20000 shuffles, no normality assumption):

**0 of 19 comparisons reach p<0.05. Expected by chance at alpha=0.05: ~1.** Smallest p is 0.262.

And the null was measured rather than trusted: an **A/A control** that splits the base runs into
two pseudo-arms of identical code and runs the same test gives **0 of 19** as well -- so the test
is not simply insensitive to everything.

## What this does NOT establish

"No significant difference" is only as strong as the sensitivity behind it. Minimum detectable
shift in `lnL` at 80% power, n=30/arm:

| cfg | MDE (nats) | observed abs(d) |
|---|---|---|
| A | 0.150 | 0.049 (32%) |
| B | 0.087 | 0.019 (22%) |
| D | 0.141 | 0.021 (15%) |
| AV | 0.045 | 0.018 (41%) |
| GMM | 0.354 | 0.073 (21%) |

So this ensemble rules out a systematic bias larger than **~0.05 nats on AV** and **~0.15 nats on
the noisy GPU-linear config** -- not a bias below that. Every observed difference sits well
inside its own detection floor. It is also ONE event, ONE waveform (SEOBNRv4), `l-max 2`, zero
noise.

**Tier 3 is discharged and the migration is no longer provisional.** The stopping rule in the
plan above -- "tier 3 cannot be run at all -> mark the migration provisional" -- no longer
applies.
