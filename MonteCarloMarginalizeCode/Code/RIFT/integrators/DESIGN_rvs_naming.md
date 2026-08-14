# DESIGN (DRAFT): give the retained set and the export resample separate names

**Status: draft for discussion. Not proposed for merge.** The code here is one worked example
of the proposal, wired into a single sampler, so the shape can be argued about against something
concrete rather than against prose.

## The problem, stated once

`sampler._rvs` means two different things at two different times in one function:

```python
# ... integrate_log accumulates draws ...
self._rvs[key]  ->  the RETAINED SET: every draw the pass kept, with real importance weights

if bFairdraw and n_extr < len(self._rvs[...]):
    self._rvs[key] = self._rvs[key][indx_list]      # WITH REPLACEMENT, proportional to weight

# ... every consumer from here on ...
self._rvs[key]  ->  an EXPORT RESAMPLE: ~1.5*eff_samp equal-weight rows, built for writing out
```

Nothing in the name changes. Nothing in the type changes. A consumer written against the first
meaning keeps working, silently, against the second.

## The evidence that this is a design problem and not a run of bad luck

**Nine defects of this one shape.** Five before the audit (CIP posterior export; L0 rescue seed,
#78; rescue reject gate, #79; warm-seed reserve cap and its logarithm, #84), three found by the
mechanical sweep (#87: sequential warm-start seed; three double-weighting exporters; pooled
`n_eff`), and then **four more in review of the fix itself** — every one in the boolean
bookkeeping introduced to paper over the naming, not in the physics:

| round | defect |
|---|---|
| 1 | a fix correct in isolation, wrong once pooling ran after it |
| 2 | one flag answering two questions (`rows resampled` vs `globally equal-weight`) |
| 2 | the CLI option used where "what this pass actually did" was needed |
| 3 | a marker cleared only on the normal return, surviving a raised event |

Each was a second source of truth about `_rvs` that some site touching `_rvs` failed to
maintain. **That is what a naming problem looks like once you refuse to rename anything.**

## Blast radius, measured

From `test/expensive_before_merging/integrators/audit_rvs_fairdraw.py --summary`:

- **306** reads of `_rvs` across 7 integrators, 3 ILE scripts, 2 CIP scripts, `distance_slices`
- **131** of them run after the rebind
- **7** rebind sites, one per sampler `integrate`/`integrate_log`

So a flag-day rename is not on. Any proposal has to be incremental and has to leave every
unconverted consumer working unchanged.

## Options

### A. Two names, `_rvs` keeps its current meaning (recommended)

`integrate_log` leaves **both**:

```python
self._rvs           # unchanged: the export resample when a fair draw fired, else the retained set
self.rvs_record     # NEW: an RvsRecord carrying rows + provenance, and both views
```

`RvsRecord` answers the questions the four review rounds kept getting wrong, as *methods with
names*, rather than as booleans a caller has to combine correctly:

```python
rec.rows_are_resampled()     # were rows drawn proportional to w?      (per block)
rec.is_equal_weight()        # is the whole record uniform?            (whole record)
rec.posterior_log_weights()  # what to weight rows by to get the posterior
```

* **Pro:** no consumer breaks; migration is one call site at a time; the two questions can never
  be conflated again because they are two methods with two names; provenance travels *with* the
  rows instead of beside them, so it cannot be left stale by an exception.
* **Con:** two objects during the migration, and a rule that they stay in sync.

### B. The fair draw returns a new object; `_rvs` stays the retained set

The correct end state, and the only one that makes the error unrepresentable.

* **Pro:** the bug becomes impossible rather than merely detectable.
* **Con:** every one of the 131 post-rebind reads must be told which object it wants, in one
  change. The export path (`copy.deepcopy(sampler._rvs)`, the `.dat` writers, the LISA twin)
  wants the resample; the seed and diagnostic paths want the retained set; and the two CIP
  scripts want neither because they never fair-draw at all. That is a large, untestable-in-one-go
  change to code that writes science products.

### C. Keep the booleans, keep the CI gate, write nothing new

Where #87 leaves things. The gate (`--check`) does catch new consumers, which is worth having
regardless.

* **Pro:** no further risk today.
* **Con:** four review rounds say the booleans are hard to maintain *even for someone whose
  whole task is maintaining them*. The next person edits one site and the invariant breaks
  somewhere they were not looking.

## Recommendation

**A now, B later, C regardless.** A is incremental and each step is independently testable; it
also subsumes the flags, which is the specific thing that keeps going wrong. B stays the target
and becomes cheap once most consumers already ask a record rather than a dict. C's CI gate stays
either way — it is the only mechanism that catches a *new* consumer rather than fixing the
current ones.

## What is in this draft

* `RIFT/integrators/rvs_record.py` — `RvsRecord`, the provenance object and the two views.
* `mcsamplerAdaptiveVolume` populates `self.rvs_record` at the rebind, alongside the existing
  `_rvs` and its flags. **Nothing reads it yet**, so this branch is a no-op on every output.
* `test/test_rvs_record.py` — the contract, including the four failure shapes from review, each
  written as a test that would have caught its round.

## What is deliberately NOT in it

* No consumer migrated. That is the next step and wants its own review.
* No change to any sampler except AV. If the shape is agreed, the other six follow mechanically
  — the rebind sites are already enumerated by the audit script.
* No removal of `_rvs_is_fairdraw` / `_rvs_is_pooled`. They stay until the last consumer that
  reads them is migrated, and the record is built to reproduce them exactly in the meantime.

## Open questions for review

1. **`rvs_record` vs `_rvs_record`.** Public reads better for something consumers are meant to
   use, but every other sampler attribute of this kind is underscored.
2. **Should the record hold the RETAINED rows too?** It would close the remaining `BROKEN` entry
   (#79's cross-source lnZ fallback) and let `.dslice` reweight properly instead of falling back
   to all-fresh. It also costs memory on a portfolio, whose `_rvs` holds every draw. The
   `_warm_seed_reserve` precedent says "a bounded copy, stratified by finite-ness" is affordable;
   whether the full set is, is a real question and I have not measured it.
3. **Does the LISA twin follow, or diverge on purpose?** It carries 36 of the 131 post-rebind
   reads and none of the helpers this work added.
