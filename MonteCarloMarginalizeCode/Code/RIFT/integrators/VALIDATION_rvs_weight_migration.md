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
