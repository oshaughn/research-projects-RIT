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

## Hybrid core+wings architecture

A single ILE job emits two kinds of slice rows in one `.dslice` file,
distinguished by the `method` column:

* **Core (method = 0, reweight)** at the heart of the posterior, where
  reweighting the main run's Omega samples is cheap and accurate.
* **Wings (method = 1, fresh)** in the low-probability tails, where
  reweighting fails because the main Omega samples don't cover the
  optimal Omega at far-from-peak distances.  Each wing is its own fresh
  AdaptiveVolume integration over Omega with distance pinned.

### Core: B2-reweight

After the main `sampler.integrate(...)` call:

1. Choose `K_core` slice centers from equi-probable quantiles of the
   posterior in d (uniform-in-log-d fallback for degenerate posteriors).
2. For each `d_k`, re-evaluate the existing `like_to_integrate` at
   `(Omega_i, d_k)` for every sample i, reusing the already-precomputed
   `rholms_intp` / `cross_terms`.  Cost: `K_core * N` likelihood
   evaluations on cached data; no waveform regeneration, no PSD reload.
3. Importance reweight:

       L(d_k) ~= (1/N) sum_i L(d_k, Omega_i) * pi_Omega(Omega_i) / q_Omega(Omega_i)

   The Omega-only IW factor `pi_Omega/q_Omega` is extracted from the
   stored joint prior/proposal ratio with the distance piece divided out.

This works well inside the posterior: Omega samples there are good
importance samples at every nearby slice distance.  Falls apart in the
tails -- which is exactly where the wings step in.

### Wings: B2-fresh

For each wing slice `d_k`:

1. Construct a fresh `mcsamplerAdaptiveVolume.MCSampler` over only the
   Omega parameters by cloning the main sampler's per-parameter
   `(pdf, prior, llim, rlim)` config.  No distance dimension.
2. Wrap `like_to_integrate` so distance is fixed to `d_k`; Omega values
   are clipped inward by ~1e-12 of their range to dodge boundary
   `arccos(1+eps) = NaN` failures.
3. Call `sampler.integrate_log(...)` with a modest budget
   (`--distance-slice-wing-nmax`, default 20k; `--distance-slice-wing-neff`,
   default 30).  AV is the canonical choice here -- it gives a real
   adapted proposal in the wings without relying on the main run's
   Omega samples.

Wing centers are placed by fitting the core `(lnL, 1/d)` points to a
parabola in `1/d` (the natural form of the marginalized lnL near peak)
and spanning each side from the core edge out to where the model drops
`--distance-slice-wing-delta-lnL` nats below peak (default 7, i.e.
prior weight < 10^{-3} outside).  This concentrates wing budget where
the likelihood actually has support.  When the parabolic fit is
degenerate (fewer than 3 core points, no lnL variation, or a
non-downward fit) it falls back to log-uniform placement across the
full `[d_min, d_core_lo]` and `[d_core_hi, d_max]` spans.

### Skip non-informative events

`--distance-slice-skip-threshold` (default 1.0 nat) is an **absolute**
lnL cut: lnL is a likelihood ratio against the noise hypothesis, so if
the *peak* lnL across the core slices is below the threshold the event
is effectively undetected and wing integrations have nothing to learn
-- they are skipped and only the core rows are written.  This is a
detectability cut, not a relative-spread test: a high-SNR event with a
flat distance profile (well-constrained inclination, unconstrained
distance) has a small spread but a large peak lnL and *does* get wings.
This guards the user's directive to "not waste time on noninformative
likelihoods".

Code:
- `MonteCarloMarginalizeCode/Code/RIFT/misc/distance_slices.py`:
  `importance_reweight_slices`, `fresh_sample_slices`,
  `quantile_slice_centers`, `pick_wing_centers`, `is_uninformative`.
- `bin/integrate_likelihood_extrinsic_batchmode`: new flags
  `--export-distance-slices K`, `--n-distance-slice-core`,
  `--n-distance-slice-wing`, `--distance-slice-wing-nmax`,
  `--distance-slice-wing-neff`, `--distance-slice-skip-threshold`,
  `--distance-slice-wing-delta-lnL`.

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

### Recommended (no DAG changes) -- step 1 DONE

1. **DONE.** Enable `--export-distance-slices K` on the **last iteration
   only** of an otherwise-normal RIFT run.  `util_RIFT_pseudo_pipe.py` now
   exposes `--export-distance-slices K` (plus
   `--export-distance-slices-{n-core,n-wing,wing-delta-lnL,skip-threshold}`),
   sibling to `--export-marginal-distance-grid`.  When set it forces ILE
   lnL mode (whole run) and routes
   `--last-iteration-export-distance-slices K ...` to the pipeline builder
   (`create_event_parameter_pipeline_{Basic,Alternate,BasicMultiApprox}Iteration`),
   which appends the ILE-level export flags to the **extrinsic** stage
   (`ILE_extr.sub`) only.  Distance marginalization is **not** disabled
   globally -- the intrinsic iterations keep it (a speedup); the pipeline
   builder strips `--distance-marginalization` from the extrinsic stage only.
   Requires `--add-extrinsic`.

   While landing this we also fixed a latent bug: the Plan-A grid flag had
   been appended to `args_ile.txt` (the ILE argument string) instead of the
   CEPP command, so it would have been handed to the ILE executable and
   rejected; both grid and slice flags now go to the CEPP command.

   End-to-end coverage: `demo/pipeline/` (Makefile + README) builds
   baseline/grid/slices pipelines and asserts the flags land in
   `ILE_extr.sub` (not the intrinsic `ILE.sub`) with no distance
   marginalization; `.travis/test-build.sh` runs the same checks in CI.

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

## Breadcrumbs for the next session

These are deliberate next changes, not unknowns.  Each one has a clear
spec; pick up here when work on `.dslice` resumes.

### 1. Skip threshold should be an absolute lnL scale -- DONE

**Status**: landed.  `--distance-slice-skip-threshold` is now an
absolute cut.  `is_uninformative(lnL_core, threshold)` returns True iff
the *peak* core lnL is below the threshold (default 1.0 nat); the old
`max - min` spread test is gone.  Help text and the ILE skip message
("peak core lnL < ... (effectively undetected)") were updated to match.

This skips undetected low-SNR events (low peak, flat profile) while
correctly *keeping* high-SNR events with a flat distance profile
(small spread but large peak lnL) -- exactly the case the relative
test got wrong.

### 2. Wing-center placement via lnL ~ parabola in 1/dist -- DONE

**Status**: landed.  `pick_wing_centers` now accepts
`(d_min, d_max, d_core, n_wing, lnL_core=None, lnL_peak=None,
delta_lnL_target=7.0)`.  When `lnL_core` is supplied it fits the core
`(lnL, 1/d)` points to a parabola in `1/d`

    lnL(d) ~= lnL_peak - 0.5 * A^2 * (1/d - 1/d_peak)^2

(`fit_lnL_parabola_in_inv_d`), solves for the two `1/d` where lnL drops
`delta_lnL_target` nats below peak (`_parabolic_wing_bounds`), and
spaces wings log-uniformly between the core edge and that boundary on
each side.  The ILE binary passes `lnL_core`, the observed peak, and
`--distance-slice-wing-delta-lnL` (default 7.0).

Robustness implemented:

* Boundaries are clamped to `[d_min, d_max]` (the sampler's distance
  support), honoring the distance-inclination-ridge caveat: the fresh
  integration will honestly report low neff on any wing that catches an
  unanticipated ridge.
* When the fit is degenerate (fewer than 3 finite core points, no lnL
  variation, non-downward fit) or leaves no room outside the core, it
  falls back to the original log-uniform full-range placement
  (`_log_uniform_wings`).
* If the requested target lnL sits above the fitted vertex (observed
  peak exceeds the fit), it uses the vertex-symmetric half-width
  `sqrt(-delta/a)`, which always yields real roots for a downward
  parabola.

**Verified**: synthetic parabola recovers `A^2` exactly and places
wings inside the solved `[d_small, d_large]` bounds rather than spread
across the full prior range; degenerate inputs fall back cleanly.

**Side benefit still open (deferred)**: `fit_lnL_parabola_in_inv_d`
exposes `A^2 = -2a` (the effective Fisher in `1/d`).  Recording it in
`.dslice` metadata for downstream CIP would require a schema/header
addition to `DISTANCE_SLICE_FIELDS`; not done yet since it touches the
load/save/reconstruct path.

## Older limitations (not high priority)

* **GMM at low main n_eff** silently biases the reweight estimator.
  The runtime warning is in place; the long-term fix is to default
  RIFT to AV for any run that turns on `--export-distance-slices`.
