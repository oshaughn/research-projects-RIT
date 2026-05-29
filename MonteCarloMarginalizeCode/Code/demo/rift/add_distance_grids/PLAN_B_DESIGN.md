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

Wing centers are placed log-uniformly in `[d_min, d_core_lo]` and
`[d_core_hi, d_max]`, half-and-half, so coverage extends from the
sampler's support boundary all the way back to the core.  Empirically
this reaches at least ~30 nats below peak on the demo event (vs the
user's ~7 nat target for 10^{-3} prior weight outside).

### Skip non-informative events

If the lnL spread across the core slices is below
`--distance-slice-skip-threshold` (default 1.0 nat), the distance
posterior is essentially flat and wing integrations have nothing to
learn -- they are skipped and only the core rows are written.  This
guards the user's directive to "not waste time on noninformative
likelihoods".

Code:
- `MonteCarloMarginalizeCode/Code/RIFT/misc/distance_slices.py`:
  `importance_reweight_slices`, `fresh_sample_slices`,
  `quantile_slice_centers`, `pick_wing_centers`, `is_uninformative`.
- `bin/integrate_likelihood_extrinsic_batchmode`: new flags
  `--export-distance-slices K`, `--n-distance-slice-core`,
  `--n-distance-slice-wing`, `--distance-slice-wing-nmax`,
  `--distance-slice-wing-neff`, `--distance-slice-skip-threshold`.

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

## Breadcrumbs for the next session

These are deliberate next changes, not unknowns.  Each one has a clear
spec; pick up here when work on `.dslice` resumes.

### 1. Skip threshold should be an absolute lnL scale

**Status**: bug.  `--distance-slice-skip-threshold` currently compares
`max(core_lnL) - min(core_lnL)` against the threshold (a *relative*
measure of how peaked the distance profile is).  But in RIFT's framing
lnL is already a likelihood ratio relative to the noise hypothesis, so
it has an absolute scale.

**What to change**:

* `is_uninformative(lnL_core, threshold)` in
  `RIFT/misc/distance_slices.py` should test whether the *peak* lnL
  across core slices exceeds the threshold, not whether the spread does.
* Default threshold should be the lnL value below which we consider an
  event undetected -- something like 1.0 to start, tunable per
  search-tier convention.
* `--distance-slice-skip-threshold` help text needs updating to reflect
  the absolute interpretation.

**Why this matters**: events with low SNR have low peak lnL *and* flat
distance posteriors; spending fresh-wing budget on them is wasteful.
Events with high SNR but a flat distance profile (well-constrained
inclination, distance unconstrained: a rare regime that does occur)
*do* need wings -- the current relative threshold would incorrectly
skip them.

### 2. Wing-center placement via lnL ~ parabola in 1/dist

**Status**: improvement.  Wings are currently placed log-uniformly in
`[d_min, d_core_lo] union [d_core_hi, d_max]` -- agnostic of the
likelihood shape.  We can do much better.

**The model**: near the peak, the extrinsic-marginalized lnL is well
approximated by a parabola in `1/dist`:

    lnL(d) ~= lnL_peak - 0.5 * A^2 * (1/d - 1/d_peak)^2

where `A` is the effective SNR amplitude.  This follows from the linear
amplitude scaling of the inner product with distance.  Fit `(lnL_core,
1/d_core)` from the core to a quadratic in `1/d`, then solve for the
two `1/d` values where `lnL = lnL_peak - delta_lnL_target`.  Set
`delta_lnL_target` to ~7 (probability < 10^{-3} outside) by default.
Place wing centers spaced log-uniformly between the core edge and the
solved boundary, on each side.

**Caveat the user flagged**: this is the *marginalized* lnL.  At the
full-likelihood level there are degeneracies where very nearby sources
fit reasonably well via fine-tuning of inclination + polarization +
phase (the so-called distance-inclination ridge).  The parabolic
extrapolation can under-estimate how far the likelihood ridge extends
toward small `d`.  Mitigation: clamp the extrapolated boundary to no
closer than `d_min_prior` and no further than `d_max_prior`, and let
the fresh integration honestly report low neff on any wing that
catches an unanticipated ridge.

**Where to land it**: replace `pick_wing_centers` in
`RIFT/misc/distance_slices.py` with a version that takes
`(d_core, lnL_core, lnL_peak, delta_lnL_target, d_min, d_max,
n_wing)`.  The current log-uniform version becomes the fallback when
the parabolic fit is degenerate (fewer than 3 core points, or all core
lnL equal, etc.).

**Side benefit**: the parabolic fit also gives a direct estimate of
`A^2` (the effective Fisher in `1/d`), which is itself worth recording
in `.dslice` metadata for downstream CIP to use as a sanity check on
its own distance-distance covariance.

## Older limitations (not high priority)

* **GMM at low main n_eff** silently biases the reweight estimator.
  The runtime warning is in place; the long-term fix is to default
  RIFT to AV for any run that turns on `--export-distance-slices`.
