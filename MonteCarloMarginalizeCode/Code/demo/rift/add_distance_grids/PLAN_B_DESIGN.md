# Plan B: distance-as-parameter ILE export

## Goal

After a normal RIFT extrinsic run, for each intrinsic point produce a usable
estimate of

    L_pure(d) = integral L(d, Omega) pi_Omega(Omega) dOmega

as a function of luminosity distance, populated densely enough that downstream
CIP can fit (intrinsic, d) jointly.  Plan A (density-histogram export, the
existing `.dgrid` pathway) reconstructs the marginal but not the curve at
the n_eff RIFT typically runs at; Plan B instead does K independent
fixed-distance integrals per intrinsic point so each slice is its own
honest extrinsic-marginalized lnL.

Deliverable target: <~10x the size of `.composite` files.  K = 10 slices
per intrinsic point and ~20 columns per row hits that budget.

## Two estimators (cross-check)

### B2-reweight (implemented)

After the main `sampler.integrate(...)` call in `analyze_event`:

1. Choose K slice centers `d_1, ..., d_K` from the posterior in d
   (equi-probable quantiles; uniform-in-log-d fallback for degenerate
   posteriors).
2. For each `d_k`, *re-evaluate* the existing `like_to_integrate` at
   `(Omega_i, d_k)` for every sample i, using the already-precomputed
   `rholms_intp` / `cross_terms`.  Cost: K * N likelihood evaluations of
   the cheap cached function; no waveform regeneration, no PSD reload.
3. Importance reweight to estimate the slice marginal:

       L(d_k) ~= (1/N) sum_i L(d_k, Omega_i) * pi_Omega(Omega_i) / q_Omega(Omega_i)

   The Omega-only IW factor `pi_Omega/q_Omega` is extracted from the
   stored joint prior/proposal ratio with the distance piece divided out.

Why this works well: for typical CBC events the Omega posterior is nearly
d-independent (d enters mostly through amplitude), so the Omega samples
drawn during the main run are good importance samples at every slice
distance.  When that assumption breaks (e.g. precessing systems where
sky/inclination couples strongly to d), `neff` at the slice drops and the
slice's `sigmaL` blows up -- that's the signal to fall back to B2-fresh.

Code:
- `MonteCarloMarginalizeCode/Code/RIFT/misc/distance_slices.py` --
  estimators and file I/O.
- `bin/integrate_likelihood_extrinsic_batchmode` --
  `--export-distance-slices K` flag; emits one `.dslice` per ILE job.

### B2-fresh (designed, not yet implemented)

For each `d_k`, build a fresh basic mcsampler over Omega only with
distance pinned to `d_k`, then call `sampler.integrate(like_to_integrate,
*omega_args, distance=d_k, ...)`.  More expensive (K independent
adaptive integrations) but doesn't rely on the Omega-quasi-independence
assumption.

Why deferred: the existing sampler-construction code in batchmode (lines
~945-1199) is straight-line, not factored.  Cleanly implementing B2-fresh
needs that block extracted into a helper, which is a larger refactor than
the user asked for in the prototype scope.  Skeleton in the same module.

When to implement B2-fresh:
- B2-reweight's per-slice `neff` drops below O(10) on real events.
- Cross-check disagreement between B2-reweight and B2-fresh exceeds the
  reported `sigmaL`.

## Output format: `.dslice`

One file per ILE job, K rows per intrinsic point.  See
`DISTANCE_SLICE_FIELDS` in `distance_slices.py`.  Key columns:

| column | meaning |
| --- | --- |
| `lnL` | extrinsic-marginalized lnL at this `dist`, pure (no distance prior baked in) |
| `sigmaL` | per-slice MC standard error of lnL |
| `neff` | effective sample count contributing to this slice |
| `ntotal` | total samples consumed by the slice estimator |
| `method` | 0 = reweight, 1 = fresh |
| `dist` | slice center (Mpc) |
| `ln_prior_d_sampling` | log of the distance prior at `dist` under the ILE sampling prior |
| intrinsic columns | m1, m2, s1x..s2z, lambda1, lambda2, eccentricity, meanPerAno, eos_index |

Re-marginalization:
```python
from RIFT.misc.distance_slices import load_distance_slice_table, reconstruct_marginal_lnL
table = load_distance_slice_table("CME_0_.dslice")
# Reproduce ILE's reported log_res (using stored sampling prior):
reconstruct_marginal_lnL(table)
# Or re-marginalize against any other prior:
reconstruct_marginal_lnL(table, ln_prior_d=lambda d: 2*np.log(d) - C)
```

## Workflow integration (non-destructive)

The user's preferred path is to add a follow-on stage after a normal RIFT
run, **without** making `create_event_parameter_pipeline_BasicIteration`
("CEPP_basic") more baroque.  Recommendation:

### Recommended (no DAG changes)

1. Enable `--export-distance-slices K` on the **last iteration only** of an
   otherwise-normal RIFT run.  `util_RIFT_pseudo_pipe.py` already threads
   `--last-iteration-export-marginal-distance-grid` for the Plan-A flag;
   add a sibling `--last-iteration-export-distance-slices K` that follows
   the same path.  This is the cheapest change to `util_RIFT_pseudo_pipe`
   (one extra option) and zero change to CEPP_basic.

2. Add a consolidation step that concatenates `.dslice` files into a
   single table per iteration -- mirror what `util_CleanILE.py` does for
   `.composite`.  The natural drop-in is a tiny wrapper script
   `util_CleanILE_dslice.py` that just concatenates with header dedup.
   Add one extra `unify_dslice.sub`/`unify_dslice.sh` pair to the existing
   subdag emitted by CEPP_basic; it depends on the same `ile` job set
   that produced the `.dslice` files.

3. (Optional, deferred) Teach CIP to ingest the `.dslice` table jointly:
   either fit `lnL(intrinsic, dist)` directly, or marginalize over `dist`
   per intrinsic point with a configurable prior and feed the marginal
   into the existing intrinsic-only fitter.  No change to RIFT structure
   needed for the export deliverable -- the `.dslice` file *is* the
   deliverable.

### Why not "another CEPP_basic" call

A second CEPP_basic invocation with an expanded sim_inspiral table
(K * N_intrinsic events, each pinned via `--pin-distance-to-sim`) was
considered.  Costs:
- K x more ILE workers spun up (K x worker startup, PSD load, frame
  read, waveform setup).  RIFT already pays a fixed worker cost
  dominated by setup at low n_eff; a Kx blowup is wasteful.
- Doubles the bookkeeping (two CEPP_basic invocations, two DAG roots,
  two output trees to keep aligned with intrinsic ids).
- Requires generating the expanded sim_inspiral, which is itself
  brittle (`xml_to_ChooseWaveformParams_array` is the very code path
  that issue #136 has been biting).

The B2-reweight pathway in-process pays only the K extra likelihood
evaluations per existing ILE job -- the expensive setup is reused.
Estimated cost: ~10% on top of a normal ILE job at K=10 (the main
integration uses ~50,000 likelihood evals; K=10 slice integrations on
the same sample set are ~10 * N_samples = ~5000 evaluations of the
*already cached* factored likelihood, dwarfed by the original
integration cost).

### Lower-level pipeline extension (only if needed)

If a `.dslice`-only iteration is needed (run the slice pass but skip a
fresh extrinsic integration -- say, because a prior run has good
adapted GMMs that we want to reuse), then we extend the low-level
pipeline with one new job class.  The least baroque approach:

- A new helper `util_RIFT_distance_slice_pass.py` that:
  1. Reads an existing `.composite` (intrinsic + ILE state)
  2. Builds a small sub-DAG of ILE jobs, each re-running on one
     intrinsic point with `--export-distance-slices K --n-max <small>`.
     (Or, if we keep adapted-state pickles per intrinsic from the
     parent run, restore those and skip the main integration.)
  3. Has its own unify step to concatenate the resulting `.dslice` files.
- This is a peer of `util_RIFT_pseudo_pipe.py` and shares its subdag
  machinery, not embedded into CEPP_basic.

Land that only if the recommended path can't carry the workflow.

## Sampler choice: use AV (or any sampler with high main n_eff)

B2-reweight relies on the existing Omega samples being a good importance
sample at every slice distance.  The synthetic stress-test
(`validate_distance_slices.py`) confirms this works to <0.1 nat even with
strong d-Omega coupling, **provided the main run's n_eff is well above
~50**.

Empirical findings on the fake-data demo (single ILE call,
`--n-max 50000`, n_eff target 100):

| sampler | main n_eff | B2 reconstruct vs log_res | per-slice n_eff |
| --- | --- | --- | --- |
| GMM (`--sampler-method GMM`) | 1-2 | +2.7 nat bias  | 16-25 (looks fine but isn't) |
| AV (`--sampler-method AV`)   | 2.6-6.5 | within `sigmaL` (-0.3 to -1.0 nat) | 7-28 |

GMM at default settings on this event ran out at n_eff=2 and B2-reweight
returned biased slice integrals without warning -- the per-slice n_eff
looked healthy because the same handful of high-weight samples dominate
every slice.  **AV is the recommended default for runs that enable
distance slices.**  GMM still works at high main n_eff (the synthetic
test confirms it); the issue is that GMM is unreliable at the n_eff
ranges RIFT routinely terminates at.

## Validation strategy

For a single ILE call (already wired into the demo):

1. Run with `--export-distance-slices K --export-marginal-distance-grid`
   so both Plan A and Plan B outputs exist side by side.
2. Check `reconstruct_marginal_lnL(slice_table)` agrees with `log_res`
   from the `.dat` row within `sigmaL`.  If yes, the slice
   re-marginalization is unbiased.
3. Plot `lnL` from `.dslice` (K honest fixed-d integrals) vs
   `lnL` reconstructed from `.dgrid` (the density histogram).  The
   slice version should be smoother and tighter at the same K.
4. If the main run's n_eff is below ~50 with GMM, fall back to AV (or
   raise n-max) before trusting B2 output.

`validate_distance_slices.py` provides a synthetic stress-test with a
known closed-form answer and an adjustable d-Omega coupling, so we can
keep the math honest as the prototype evolves.

## Known limitations / follow-ups

* **Slice center placement**: quantile centers of the posterior cluster
  where the likelihood already lives.  For re-marginalization against a
  prior with significant weight *outside* the posterior support (e.g.
  cosmologically-motivated priors that look very different from the
  volumetric default), additional slices in the prior's mass region may
  be needed.  Configurable via a future `--distance-slice-centers
  {quantile,log-uniform,custom}` flag.
* **B2-fresh** (independent integrations per slice) remains a useful
  cross-check, especially for events with extreme d-Omega coupling that
  the synthetic harness might miss.  The cleanest implementation needs
  the per-event sampler setup factored out of batchmode into a helper,
  which is the right next refactor when B2-fresh becomes a priority.
* **GMM warning**: consider emitting a runtime warning if
  `--export-distance-slices` is set together with `--sampler-method GMM`
  and the main n_eff falls below a threshold, pointing the user at AV.
