# Audit: every `_rvs` consumer, against the fair-draw rebind

Mandate: *"Every other consumer of `_rvs` rows or lengths in the ILE is unaudited. Everything
found here was in code this change happened to touch."*  This is that sweep, done
mechanically so it can be re-run rather than repeated by hand.

Different axis from `RVS_CACHE_AUDIT.md` in this directory.  That one asks *"cached column,
or canonical components?"*; this one asks *"before or after the rebind?"*.  A site can be
wrong on either axis independently, and the two sweeps agree on nothing except the file list.

## How to re-run it

```
python3 audit_rvs_fairdraw.py --summary        # counts per file and phase
python3 audit_rvs_fairdraw.py --post-rebind    # only the sites that see the resample
python3 audit_rvs_fairdraw.py --check          # CI gate: every such site has a verdict
python3 make_rvs_fairdraw_ledger.py            # regenerate the verdicts from the rules
```

`--check` runs in CI.  It does **not** assert that post-rebind reads are bugs -- most are
legitimately per-row.  It asserts that each one carries a recorded human verdict, keyed by a
whitespace-normalized hash of the read itself.  Moving or reindenting a line keeps its
verdict; changing what it reads does not, and a new consumer fails the build.  Function-level
keying was rejected deliberately: `analyze_event` alone holds 40 sites, and every one of the
five known defects was added *next to* correct code.

## The census

306 reads of `_rvs` across the seven integrators, the three ILE scripts, the two CIP scripts
and `RIFT/misc/distance_slices.py`.  131 of them see the resample.  (Counts move as the code
is edited -- `--summary` is the authority, not this table.)

| | sites |
|---|---|
| `BEFORE` the rebind, inside the rebinding function | 160 |
| `AFTER`, inside it | 26 |
| `POST_INTEGRATE` (bin/ scripts, after `integrate` returned) | 104 |
| post-rebind reads carrying a recorded verdict | 131 / 131 |

Those 131 reads collapse to 123 distinct ledger entries (identical reads share a key):
`PER_ROW` 54, `BENIGN` 61, `FIXED` 3, `NO_FAIRDRAW` 4, **`BROKEN` 1** -- down from 8 when the
sweep started.  The one remaining is #79's cross-source fallback, described under Finding 0.

**Inside the integrators, nothing post-rebind is broken.**  All 26 lexically-`AFTER` sites are
the rebind's own right-hand side or the element-wise cupy→numpy conversion loop that follows
it.  That is a real result and it narrows the search: the hazard lives entirely in the
consumers, not in the samplers.

`--fairdraw-extrinsic-output` is not exotic.  `create_event_parameter_pipeline_BasicIteration`,
`cepp_basic_htcondor` and `create_event_nr_pipeline_with_cip` all append it to the extrinsic
stage unconditionally, so every one of these paths runs on the resample in production.

## Measured, not asserted

`verify_skew.py` drives the ILE's own `_lnZ_of_rvs` / `_kish_neff_of_rvs`:

| claim | result |
|---|---|
| `_lnZ_of_rvs(..., already_pooled=False)` on a fair-drawn record is high by `log(n/n_eff)` | excess `+4.509` vs predicted `+4.503` at n_eff 44; `+7.33` vs `+7.47` at n_eff 2.3 |
| the error does not cancel between two passes at different n_eff | two passes with **identical true lnZ**, n_eff 1.8 vs 53.3: the gate reads `+3.48` nats |
| ...and therefore rejects a good warm pass | **100%** of the time at the 0.5 default; 94% at 2.0 |
| re-weighting an already fair-drawn record shifts a posterior mean | **13%** of the true value on a weight-correlated coordinate |

## Finding 0 -- PR #79 was not in the development line (RESOLVED by #86)

> **Resolved 2026-08-13.** Re-landed as PR #86 (merge `5a2965ba`), cherry-picking `23be21ab`
> and `9fc806f8` onto `0ae3f48f` -- both clean, and the resulting diff byte-identical to the
> original, so #79's approval carried over rather than being re-reviewed. A sweep of all 74
> merged PRs found **no other orphans**: the only other non-ancestors are four `master`-based
> PRs touching solely `.github/workflows/private-review-dispatch.yml`, which correctly do not
> belong in `rift_O4d`.
>
> Two lessons worth keeping, since ancestry alone answers neither:
> * `git merge-base --is-ancestor` gives FALSE ALARMS when content re-lands by cherry-pick
>   (#79 still fails it; its patch-ids are present). Confirm with `git cherry` / `patch-id`.
> * `git cherry` gives false alarms of its own when a change lands FOLDED into another commit
>   -- `7c986d63` reports `+` while its content is in `rift_O4d`, verified by diffing the
>   function. Neither tool is sufficient alone; the file is the authority.
>
> One residual survives #79 and is recorded as `BROKEN` in the ledger: its cross-source
> fallback (`_cold_src != _warm_src`) re-reads both sides from the fair-draw record, which is
> self-consistent but not unbiased, since the two passes sit at different `n_eff`. Bounded to
> the mismatch case and documented at the site. Closing it needs a reserve for the samplers
> that keep none.

The original finding, kept because it explains how this happens and how to detect it.  Past
tense throughout: this described the tree before #86.

The reject-gate fix was **not an ancestor of `junior/rift_O4d`**, so the gate in the tree we
developed from was the pre-#79 code, and the measurement above described production.

```
7ca5c8df  warm-seed reserve: record the exact pre-cap weight total   (PR #84 branch tip)
68987b26  Merge PR #79 into that branch                              <- has the fix
bc210833  Merge PR #84 ... parents are ad426d9f and 7ca5c8df         <- took the branch BEFORE 68987b26
```

`junior/claude/l0-rescue-exact-total` pointed at `68987b26`, but `bc210833` merged `7ca5c8df`.
So #79 was merged into the #84 branch *after* #84 had already gone to `rift_O4d`, and never
followed.  Verified three ways: `merge-base --is-ancestor` fails for `23be21ab`, `7efdd229`,
`9fc806f8` and `68987b26`; `git show 0ae3f48f:...batchmode | grep _lnZ_of_reserve_or_rvs`
returns nothing; and `lnZ_from_reserve` is absent from `mcsamplerAdaptiveVolume.py`.

Note `rift_O4d_wt_ralph`, a **pinned measurement tree**, sat at `7efdd229` -- a copy of the #79
work -- so measurements had been running against code the development line did not have.

The `--sampler-l0-rescue-reject-dlnZ` re-tune was blocked on this, since re-tuning against the
`log(n/n_eff)` artifact would have calibrated to the bug.  With #86 in it was measurable, and
was measured: see Finding 4.

## Finding 1 -- the sequential warm-start seed (FIXED here)

`--sampler-sequential-warmstart` captures a seed for the next intrinsic point.  It read
`sampler._rvs` and guarded with `_lnv.size >= 2` / `np.sum(_keep) >= 2`.  That is PR #78's
defect exactly, one code path away: the resample, and a **count** where a **rank** is needed.

Both failure directions are real and neither is loud.  On the measured pass the fair draw
keeps **one** row (`Fairdraw size : 1`), so the count declines and the feature is *silently
inert* -- the user asked for a warm start and got none.  At the rho_net 102.8 regime it keeps
~5 and the count *accepts* a rank-2-of-6 seed, so the next point warm-starts into a sliver and
reports a healthy n_eff over truncated support.

Fixed by routing through `_warm_seed_reserve_for` (new shared lookup) and `build_warm_seed`
(rank test + puff), mirroring the L0 rescue.  The L0 rescue's inline reserve lookup was
replaced by the shared one, so the pair cannot drift.  13 tests in
`test/test_seq_warmstart_seed.py`, added to CI; `test_l0_rescue_seed.py` +
`test_av_empty_live_volume.py` still pass (100 passed, 2 skipped).

## Finding 2 -- three double-weighting sites (FIXED)

All three took rows that are already a `w`-proportional draw and weighted them by `w` again.
`_pool_replica_rvs` has guarded against exactly this since the replica work, via
`already_resampled`; none of these had an equivalent.

The predicate matters, and it is not the CLI flag: the samplers skip the draw when it would not
shrink the record, and those rows still carry real importance weights. So each sampler now marks
the rebind ITSELF (`self._rvs_is_fairdraw`, set at all seven rebind sites, reset per pass because
samplers are reused across events), and the ILE reads it through `_rvs_is_export_resample`.

1. **`--extrinsic-proposal-output` breadcrumb** -- fitted the GMM proposal with `w` on top of a
   `w`-proportional draw, so the proposal came out shaped like `w^2` and was then handed to the
   NEXT iteration via `--extrinsic-proposal-breadcrumb`. The worst of the three, because the
   truncation compounded across iterations. Now takes `ln_weights_for_posterior`.
2. **`.dgrid`** -- same shape, one product. Now takes `ln_weights_for_posterior`.
3. **`.dslice` reweight core** -- a *different* shape: it double-counts `pi_Omega/q_Omega` and
   takes `N` from the resample, and cannot be corrected after the fact because the pre-draw
   record is gone. Routed instead to the exact all-fresh path (K independent fixed-d
   integrations), loudly, since that changes cost.

`ln_weights_for_posterior` is the new single answer to "how should these rows be weighted to
represent the posterior", as distinct from `ln_weights_from_rvs`, which answers "what is the
importance weight of this record" and was always right about that.

## Finding 3 -- pooled n_eff (FIXED)

`_kish_neff_of_rvs` on the pooled record, which overwrites the reported `neff`. When the export
is fair-drawn `_pool_replica_rvs` deliberately flattens each block, and the Kish n_eff of
piecewise-constant weights is just the row count -- `K*min(n_max, 1.5*eff_samp, 1.5*neff)`, the
size of the EXPORT. At the default `--fairdraw-extrinsic-output-n-max 5` that reports `n_eff = 5K`.

Now computed one level up where the quantities are still meaningful -- Kish over the BLOCKS,

    neff_pooled = (sum_k Z_k)^2 / sum_k (Z_k^2 / neff_k)

which is exactly the property the original comment asked for: it reduces to `sum_k neff_k` when
the replicas agree and falls below it when they disagree. The pooled Kish is still used when the
export was NOT fair-drawn, where per-row weights are real and finer-grained.

## Finding 4 -- the reject knob (MEASURED; default raised)

See `L0_REJECT_DLNZ_MEASUREMENT.md`. Across 160 known-lnZ passes the gate caught **0 of 55**
truncated warm passes at every threshold, while at the old 0.5 default it binned **25%** of good
portfolio warm passes. Default raised to 3.0 -- strictly better, since there was no detection to
trade away. The gate is documented as not being a truncation detector; support containment is
the recommended replacement and is deliberately left as follow-up.

## Recommendation: make the rebind unable to do this again

Five defects -- now six -- of one shape in one attribute is an API problem.  Ranked by
benefit per unit of blast radius:

1. **Keep the retained set under its own name (recommended).**  `_warm_seed_reserve` already
   is this, for two callers.  Generalize it: have `integrate_log` always leave
   `self._retained` (or keep `_rvs` and expose `self._export`), and migrate consumers one at
   a time.  Cheap, incremental, and each migration is independently testable.  It does not
   *prevent* the mistake, but it gives every future author a correct thing to reach for.
2. **Have the fair draw return a new object instead of mutating in place.**  The correct fix
   in principle and the one that makes the error unrepresentable.  Blast radius is large:
   `_rvs` is read at 307 sites, and every `bin/` consumer would need to be told which object
   it wants. Worth doing behind the naming change above, not instead of it.
3. **Keep `--check` in CI regardless.**  It is the only one of the three that catches the
   *next* consumer rather than fixing the current ones, and it is already green.

A cheaper partial: set a flag on the record (`_rvs['__fairdrawn__'] = True`) and have
`ln_weights_from_rvs` warn when a caller weights a flagged record.  That would have caught all
three Finding-2 sites at runtime, and it is a dozen lines.
