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

## How it plugs into the pilot DAG (next)

The cal pilot already establishes the handoff plumbing: a stage harvests one iteration's
output, fits a proposal, consolidates, and seeds iteration N+1 via a breadcrumb file that is
transferred on OSG.  The extrinsic handoff reuses that exact path -- the wide ILE jobs write
`--extrinsic-proposal-output`, a consolidation picks/merges the best, and the next iteration's
ILE jobs read `--extrinsic-proposal-breadcrumb`.  Because cal and extrinsic live in ONE
breadcrumb object, this can ride on the same file the cal pilot already transfers.  (Pipeline
wiring not yet added -- the module + ILE hooks are the proof-of-concept.)

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
