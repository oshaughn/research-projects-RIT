# Extrinsic handoff: carry the extrinsic posterior between iterations

Status: **proof-of-concept implemented (GMM)**; AV partial-reset is future work (task #30).

## The decade-old goal

RIFT re-solves the *same* extrinsic integral (sky, distance, inclination, polarization,
orbital phase, time) on every intrinsic-grid point, every iteration.  But the extrinsic
posterior is set by the data + best-fit template, not by the small intrinsic-grid moves --
so it barely changes from iteration to iteration.  Today each ILE job starts its extrinsic
sampler **cold** (a wide prior proposal) and re-discovers the same sky modes / distance
blob from scratch.  This is the long-standing "save the extrinsic distribution to inform
the next iteration" idea: learn the extrinsic proposal once, hand it forward, and let the
next run start *on the answer*.

This generalizes the calibration pilot's breadcrumb (`RIFT.calmarg.breadcrumbs`,
`RIFT.calmarg.generate_realizations.seed_realizations_from_breadcrumb`): the cal handoff
carries a Gaussian over spline-node parameters; the extrinsic handoff carries a learned
proposal over the extrinsic parameters in the SAME breadcrumb file.

## GMM-first (implemented)

RIFT's ensemble sampler (`mcsamplerEnsemble`) is **already seedable**.  Its `gmm_dict`
maps parameter GROUPS -- tuples of indices into `params_ordered` -- to a fitted
`gaussian_mixture_model.gmm`; a non-`None` entry is used as the starting proposal and keeps
adapting.  The standard groups (set in `analyze_event` ~line 1410) are:

    (right_ascension, declination)   # sky
    (distance, inclination)          # distance/orientation
    (phi_orb, psi)                   # phase/polarization

So the GMM is the trivially-seedable model to prove the handoff with -- no new sampler
machinery, just pre-fill `gmm_dict`.  A normalizing flow (or a seedable AV, below) can drop
in later behind the same fit/seed interface (`extrinsic['kind'] != 'gmm'`).

### Pieces

- **`RIFT.calmarg.extrinsic_handoff`**
  - `fit_extrinsic_proposal(samples, log_weights, groups=STANDARD_GROUPS, bounds, n_comp=4)`
    -- per group, fit RIFT's OWN `gmm.fit` (the exact fitter the sampler uses in
    `update_sampling_prior`), using importance weights `lnL + ln prior - ln sampling_prior`.
    Returns the portable breadcrumb `extrinsic` dict (per-group means/covs/weights/bounds +
    parameter NAMES).  GMMs may run on cupy -- inputs are moved via
    `model.identity_convert_togpu` before `.fit`.
  - `reconstruct_gmm(group, adapt=True)` -- rebuild a `gmm` from a stored group (means/
    covariances/weights restored in the model's internal normalized frame; `adapt=True`
    lets the seeded components keep adapting, since the extrinsics drift slightly).
  - `gmm_dict_from_breadcrumb(extrinsic, params_ordered, adapt=True)` -- build the
    `{dim_group_tuple: gmm}` to seed the next sampler.  Dim-groups are looked up by
    parameter NAME against this run's `params_ordered`, so the handoff is robust to a
    different parameter ordering between runs; groups whose params aren't all present are
    skipped silently.

  Using RIFT's own fitter means the stored means/covariances are in exactly the model's
  internal (normalized) frame and restore to a byte-identical model -- no coordinate
  guesswork.

- **`RIFT.calmarg.breadcrumbs` (schema v2)** -- the `save`/`load` object gained an
  `extrinsic` slot alongside the existing `cal` slot.  A breadcrumb can carry cal, extrinsic,
  or both.  Per group it stores `params`/`means`/`covariances`/`weights`/`bounds`.  Schema
  is additive (v1 cal-only breadcrumbs still load); bump `SCHEMA_VERSION` on incompatible
  changes.

- **ILE wiring** (`integrate_likelihood_extrinsic_batchmode`, execute-point -- needs a
  container rebuild):
  - `--extrinsic-proposal-output PATH` -- after `sampler.integrate`, harvest the run's
    extrinsic posterior samples + importance weights from `sampler._rvs` (same weight recipe
    as the distance-grid export, including the GMM sampler's raw-integrand storage), fit per
    group, and `breadcrumbs.save(PATH, extrinsic=...)`.  Wrapped in try/except so a
    harvest/fit failure can never break a production integration.
  - `--extrinsic-proposal-breadcrumb PATH` -- before integration, load the breadcrumb and
    pre-fill `gmm_dict` for the matched dim-groups (`gmm_adapt=True`).  Missing/unreadable
    breadcrumb -> warn and fall back to the cold default.

### Proof of concept

`python -m RIFT.calmarg.extrinsic_handoff` builds a synthetic **bimodal** sky posterior +
unimodal distance/inclination blob, fits it, round-trips through a breadcrumb, seeds a fresh
GMM against a *shuffled* `params_ordered`, and confirms the seeded sky GMM reproduces BOTH
sky modes with ~the right mode fractions.  `python -m RIFT.calmarg.breadcrumbs` confirms the
cal-Gaussian + extrinsic-GMM coexist and round-trip.  Both PASS.

## Pipeline wiring (implemented)

The handoff is wired end-to-end through the pipeline, gated by `--extrinsic-handoff` and
**standalone** (it does NOT require the cal pilot -- it works on a plain fused / vanilla run):

- **`util_RIFT_pseudo_pipe.py --extrinsic-handoff`** adds to `args_ile.txt`:
  - `--extrinsic-proposal-output extr_proposal_$(macroiteration)_$(macroevent).npz` -- each
    wide ILE job writes its own per-event proposal ($(macroevent) is the per-node macro);
  - `--extrinsic-proposal-breadcrumb .../extr_consolidated_$(macroiterationprev).npz` -- the
    seed from the previous iteration (OSG: basename + auto-added to the ILE transfer list +
    an `extr_consolidated_-1.npz` placeholder for iteration 0; shared FS: absolute path).
  It warns if `--ile-sampler-method` is not GMM (the seed is a no-op for other samplers).

- **`util_ExtrinsicConsolidate.py`** (new) picks the single most representative per-event
  proposal (default by lnL -- nearest the peak; `--select neff|n_samples` also available) and
  writes `extr_consolidated_<it>.npz`.  It ALWAYS writes output (empty if nothing valid), so
  the next iteration's seed/transfer never fails; unreadable/placeholder inputs are skipped.

- **`dag_utils_generic.write_extrconsolidate_sub`** builds the consolidation job in the
  **local universe** on the submit node: it is pure-python file selection (no GPU/ILE/
  container/frames), and on OSG the per-event ILE outputs are transferred back to
  `<wd>/iteration_<it>_ile` (ILE's default output transfer), so a local-universe job reads
  them from the shared FS with no per-event input transfer (which condor cannot glob).

- **`create_event_parameter_pipeline_BasicIteration`** creates one consolidation node per
  iteration, gated behind that iteration's `unify` node (all ILE done -> per-event proposals
  present), and makes iteration N+1's wide ILE jobs depend on the iteration-N consolidation:
      unify_{it}  ->  EXTRCONSOLIDATE_{it}  ->  wide ILE_{it+1}
  (the consolidate barrier and the seed barrier), exactly mirroring the cal-pilot wiring.

`make extr-build` (demo/rift/calmarg) builds a pipeline with `--extrinsic-handoff
--ile-sampler-method GMM` and validates the whole thread offline (args_ile.txt flags,
EXTRCONSOLIDATE.sub, and the unify->consolidate->next-ILE DAG edges).

Because cal and extrinsic live in ONE breadcrumb object, a future refinement could ride the
extrinsic proposal on the cal pilot's existing consolidation/transfer instead of a separate
node; the standalone path was chosen first so the handoff works without the (heavier) cal
pilot.  The convergence-subdag extension (`--first-iteration-jumpstart`) does not yet carry
`--extrinsic-handoff` -- same limitation as `--calmarg-pilot`.

## Real-GPU validation (cardassia, NVS 510) and what it taught us

Ran the full loop interactively on one intrinsic point, GMM sampler + calmarg-fused, on the
CI data:  iteration-0 writes `extr_proposal_0_0.npz` -> `util_ExtrinsicConsolidate` picks it
-> iteration-1 ILE loads it and prints `Extrinsic GMM SEEDED ... for dim-groups
[(4,5),(3,2),(0,1)]` (all three standard groups) -> integrates -> writes
`extr_proposal_1_0.npz`.  End-to-end the plumbing works on real hardware.  Two bugs only the
GPU run surfaced, now fixed:

1. **bounds left on the host.** `reconstruct_gmm` set means/covs/weights onto the GPU but
   left `self.bounds` as numpy.  The sampler's `score()`/`_normalize` write into an
   `xpy.empty` (cupy) array, so a numpy `self.bounds` raised
   `ValueError: non-scalar numpy.ndarray cannot be used for fill`.  Fix: `model.bounds =
   identity_convert_togpu(bounds)`.
2. **within-group parameter ORDER.** The sampler keys the phase/pol group as
   `(psi, phi_orb)=(0,1)` but the breadcrumb stored `(phi_orb, psi)=(1,0)`, so that seed was
   silently dropped (key mismatch).  Fix: `gmm_dict_from_breadcrumb(existing_keys=...)` matches
   each breadcrumb group to the sampler's actual gmm_dict key by dim-SET and permutes the
   stored means/covariances/bounds columns into that key's order.

**Seed quality depends on the SOURCE iteration's convergence.**  When the ensemble sampler
hits a bad batch it calls `_reset()`, which sets every `gmm_dict[k]=None` -- i.e. it
**discards the seed and continues cold**.  This is the correct safety net: a bad seed is
thrown away, never corrupting the result.  In a deliberately tiny smoke (`--n-max 40000` on
the NVS 510 -> iteration-0 `n_eff ~ 1`), the iteration-0 proposal is near-degenerate, so the
seeded first batch produces zero/NaN effective weights and the sampler resets to cold.  The
handoff is then correct-but-cosmetic.  To see the seed actually ACCELERATE convergence you
need a source iteration that converged reasonably (`n_eff` in the hundreds) -- i.e. a real
`--n-max` (millions) and/or a larger GPU.  A modest `cov_inflate` (default 2.0, ~1.4x width)
broadens the seed so the sampler can contract it -- good practice for a warm start, but it
mitigates rather than rescues a genuinely degenerate source.

## Measured blocker: the GMM sampler does not converge on real sharp ILE peaks

Trying to demonstrate the seed ACCELERATING convergence on the CI point (network SNR ~17.5,
lnLmax ~ 90-115) surfaced a hard limit of the *seedable* sampler itself, independent of the
handoff and of calibration:

| config (single CI point, GMM sampler)        | n_eff at ~200k samples |
|----------------------------------------------|------------------------|
| GMM + calmarg (n_cal=20)                      | ~1.0   (256k)          |
| GMM, vanilla (no calmarg)                     | 1.00007 (196k, 50 it)  |

The ensemble (GMM) sampler collapses its mixture onto the single dominant sample at a sharp,
high-SNR peak and then stops improving -- n_eff is pinned at 1 with or without calmarg.
(The AV sampler, by contrast, reached n_eff in the hundreds at a few x10^6 samples in the
earlier calmarg tune runs -- AV's adaptive tessellation handles these peaks; GMM does not.)

Consequence for the handoff: the GMM->GMM extrinsic handoff is correct and safe, but on real
high-SNR ILE likelihoods the GMM SOURCE iteration never converges to a good proposal, so there
is nothing useful to hand off, and the cold GMM baseline is equally stuck -- there is no
acceleration to measure.  The handoff's value is therefore gated on a *seedable sampler that
actually converges*:
  - **seedable / partial-reset AV (task #30, #25)** -- the real unlock: AV converges on these
    peaks but resets every integrate() and has no seed path.  This is now the critical-path
    item for making the extrinsic handoff pay off on production data.
  - or a **cross-sampler handoff**: converge with AV, fit the GMM to AV's posterior samples
    (fit_extrinsic_proposal already does exactly this from any sampler's weighted samples),
    and seed a GMM/flow refinement.  The save side already accepts arbitrary samples+weights;
    only the "harvest AV's _rvs and fit" wiring would be new.

The handoff plumbing (save -> consolidate -> seed, all groups, GPU-correct) is done and is the
right substrate; the demonstration of speed-up waits on one of the above.

## Seed adaptation: FREEZE by default (`--extrinsic-proposal-adapt`)

Re-fitting a seeded GMM group on the first batch is fragile on these likelihoods: with
`adapt=True` the sampler's `_train` calls the GMM fit, whose `_initialize` does
`random.choice(p=weights)` and dies on the pathological first-batch weights
("probabilities are not non-negative") -> `_reset()` -> the seed is discarded.  `_train`
already skips groups whose `gmm_adapt[group]` is False, so the ILE seed path now FREEZES the
seeded groups by default (`gmm_adapt=False`); `--extrinsic-proposal-adapt` opts back into
adaptation.  Freezing is also the right semantics for a handed-off (especially cross-sampler)
proposal: trust it as-is rather than let GMM's adaptation degrade it.  Result: with freeze the
seeded run completes with **0 resets** and the seed actually drives sampling.

## Cross-sampler AV->GMM seed: partial result, integral still wrong (open)

Per the chosen plan, converged iteration-0 with **AV** (which does make progress on this
point: n_eff ~7 at 400k, lnLmax ~143), fit the GMM to AV's posterior samples
(`fit_extrinsic_proposal` reads any sampler's `_rvs`), consolidated, and seeded a **frozen**
GMM run:

- the seed lands cleanly (all 3 groups), **0 resets**, and n_eff rises from the cold ~1 to
  **~5-10** -- the seed mechanism is injecting structure.
- BUT the seeded GMM's INTEGRAL is wrong: `sqrt(2 lnLmax)` prints `nan` and Z comes out
  ~1e-4 (vs the cold GMM's valid ~1e43 and AV's lnLmax~143).  High n_eff in the WRONG region
  is worse than honest low n_eff: the frozen proposal is importance-sampling a region that is
  consistent-but-displaced from the true posterior.

Two suspects, not yet isolated (needs a focused audit, no more blind GPU time):
1. **coordinate convention** -- AV vs GMM may store extrinsic samples in `_rvs` under
   different conventions (e.g. angle vs cosine for inclination/declination; the sampler adds
   `inclination`/`declination` on `[-1,1]` = cosine when `--*-cosine-sampler` is set, but it
   is not obvious AV's `_rvs` uses the same).  A mismatch would place the fitted GMM in the
   wrong frame.  Same-sampler GMM->GMM has no such mismatch and round-trips cleanly.
2. **`cov_inflate` out of bounds** -- inflating the seed covariance (x2) can push a sampled
   `distance` outside `[1,1000]` (or other hard edges) where the likelihood returns NaN,
   contaminating lnLmax.  Worth testing `cov_inflate=1` and clipping proposed samples.

Net: the handoff machinery, the freeze, and the AV-source convergence all work; the
cross-sampler numeric correctness is one debugging session away (audit `_rvs` conventions +
inflation/bounds).  The same-sampler GMM->GMM path is already numerically clean -- it just
needs a sampler that converges as a source, i.e. seedable AV (below).

## Cross-sampler AV->GMM: numerics RESOLVED; benefit gated by GMM convergence

Debugging the wrong-integral above (per user's steer) found and fixed FOUR real issues in the
save/seed path; the cross-sampler seed is now numerically correct:

1. **tempered weights (save side).** The GPU/AV sampler (mcsamplerGPU) stores
   `_rvs['log_weights'] = tempering_exp*lnL + ln(prior) - ln(s_prior)` -- the adapt-weight-
   exponent (e.g. 0.1) baked in.  Fitting the GMM to those flattened weights displaces the
   proposal.  Fix: build the weight from the raw, UNTEMPERED components
   (`log_integrand + log_joint_prior - log_joint_s_prior`) and prefer them over `log_weights`.
   (GMM's own `_rvs` has no tempering -> GMM->GMM was already fine.)  This alone took the
   seeded n_eff from ~5 to ~26.
2. **cov_inflate.** Inflating a FROZEN seed only widens it out of bounds; default is now 1.0
   (freeze handles robustness; inflation was for the adapt=True path).
3. **starved fit -> NaN component.** A low-ESS source over-parameterized (n_comp=4 vs few
   effective samples) collapses a mixture component to a singular/NaN covariance, and one NaN
   component poisons the whole seeded proposal.  Fix: cap n_comp by the weight ESS
   (`k <= ESS/(d+2)`) and drop any non-finite component (renormalize; skip the group if none
   survive).
4. **distance sampled against a hard bound.** The real source of the persistent `nan` lnLmax:
   with distance SAMPLED on `[1,1000]`, a seeded distance Gaussian spills past the bound ->
   NaN likelihood.  Distance marginalization (`--distance-marginalization` + a lookup table
   from `util_InitMargTable`) removes distance from the extrinsic sampler entirely; with it on,
   the seeded run's lnLmax is finite and the integral is valid.  Distmarg is OPTIONAL with the
   fused kernel, not required -- the fused kernel has both a non-distmarg kernel
   (`Q_fused_calmarg_cupy`) and a distmarg kernel (`Q_fused_calmarg_distmarg_cupy`), and the
   ILE binary wires whichever applies.  In the pipeline it is `--internal-marginalize-distance`
   (which composes cleanly with `--calmarg-fused-kernel`); in the demo it is the `PP_DMARG=1`
   toggle.  RECOMMENDED with `--extrinsic-handoff` precisely because it removes the distance
   dimension + its hard bound from the seeded GMM proposal.

Measured, distmarg on, single CI point (SNR~17.5), all fixes in:
- AV source converges to n_eff~4.7 (lnLmax~152), writes a clean 2-group (sky, phase/pol) proposal.
- seeded GMM: 0 resets, FINITE lnLmax, VALID integral -- but n_eff ~1.0, ~the same as the
  cold GMM (~1.0-1.3).  The seed neither helps nor hurts.

**Conclusion.** The handoff (save -> consolidate -> seed) is now numerically correct and safe
end-to-end on real GPU data.  But it does not ACCELERATE on this point because the seedable
sampler (GMM) does not converge here (n_eff~1 cold AND seeded), and the AV source (n_eff~5) is
too under-converged to provide a strongly-informative seed.  GMM is seedable but weak; AV
converges but is not seedable.  This is now hard evidence that the payoff requires a
**seedable / partial-reset AV (task #30, #25)** -- or a converged source (lower SNR / much
larger sample budget / better GPU) so the GMM seed has real information to carry.  The numeric
substrate is done; the win is one of those two regimes away.

## Why GMM first, and the AV limitation (task #30)

The adaptive Voronoi sampler (AV, `mcsampler`) is the default extrinsic sampler and is more
efficient, but it **completely resets** between `integrate()` calls -- there is no seed path,
and re-seeding is dangerous because AV can only *contract* its boundaries, never expand or
shift them.  So a naive AV warm-start could lock the sampler onto a stale region.  The GMM
(and portfolio) samplers reuse sampling models cleanly and are trivially seedable, so they
are the right vehicle for the first working handoff.

Future work (task #30): a **seedable / partial-reset AV** -- reset only some parameters, or
seed a proposal that AV is allowed to *expand* from, so the more-efficient sampler can also
benefit from the handoff.  The breadcrumb `kind` field already leaves room for a non-GMM
model behind the same `save`/`load`/seed interface.
