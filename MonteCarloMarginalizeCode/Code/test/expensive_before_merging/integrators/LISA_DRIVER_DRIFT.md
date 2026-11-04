# The LISA ILE driver, against the main one

The two drivers are a **deliberate fork** (RO, 2026-08-13: *"It is super annoying we have to
have two of them, but the overhead of one ring to rule them all is too high."*).  Nothing here
argues for merging them.  The purpose is to make the *consequence* of the fork -- drift --
mechanically visible, so it stays a choice.

    bin/integrate_likelihood_extrinsic_batchmode        4,883 lines   moves fast
    bin/integrate_likelihood_extrinsic_batchmode_lisa   2,526 lines   lags

Both import the SAME integrators and expose the SAME `ok_lnL_methods`
(`GMM, adaptive_cartesian, adaptive_cartesian_gpu, AV, portfolio` -- verified identical), so
anything landed in `RIFT/integrators/` already reaches LISA.  **All measured drift is in the
driver.**

## How to regenerate this

Do not trust the numbers below; they are a snapshot.  The tooling is the authority.

```
python3 audit_lisa_driver_drift.py --summary     # counts per category and decision
python3 audit_lisa_driver_drift.py --undecided   # what nobody has classified yet
python3 audit_lisa_driver_drift.py --check       # the CI gate
python3 make_lisa_drift_ledger.py                # regenerate lisa_drift_ledger.json
```

`audit_lisa_driver_drift.py` extracts four categories from both drivers by AST and diffs them:
`FUNC` (def names, qualified by enclosing function), `OPTION` (`--foo` literals given to
`add_option`/`add_argument`), `CONST` (module-level `UPPER_CASE`), `ATTR` (sampler provenance
markers -- `_rvs_is_*`, `_warm_seed*`, including the `getattr(obj, 'name', default)` form,
which is how the readers actually access them).

The judgements live in `make_lisa_drift_ledger.py` as ordered
(pattern -> decision + reason) rules, first match wins, so a whole family is decided once.
An item matching no rule is reported and left out, which fails `--check`.  That is the
intended path for newly-drifted code: **a person has to classify it.**

## What this audit CANNOT see

Stated plainly, because an adversarial review defeated the gate with four realistic drifts
and the honest answer is that some of them are out of scope by construction rather than by
oversight.

**It is a NAME-PRESENCE set difference.**  It answers "does the LISA driver have a thing
called X".  It does not compare behaviour.  So all of these produce **zero** gap items:

* a **changed default** on an option present in both drivers (`--adapt-floor-level` going
  0.1 -> 0.9 is invisible here);
* **changed help text**;
* a **changed body** of a same-named function -- the anti-drift tests in
  `test/test_lisa_*.py` cover this for the specific helpers that were ported, and nothing
  covers it for anything else;
* a **missing `if` branch or `pinned_params` key**, which is not a FUNC/OPTION/CONST/ATTR at
  all.  A real example is below.

**Option names built at runtime evade the extractor.**  `add_option(_name_var, ...)`,
options added in a `for` loop, and `"--evade-" + "concat"` are all missed, because the
extractor reads string LITERALS out of the AST.  Since `OPTION` is the large majority of the
gap, this is the biggest hole.  Neither driver does any of this today.

**`ATTR` is presence-anywhere.**  A marker READ but never WRITTEN counts as present, so a
reader-ported/writer-missing port looks closed.  `_rvs_is_pooled` is exactly that today, and
its ledger entry says so.

The gate is worth having anyway -- it catches the ordinary case, which is a helper or an
option appearing in the main driver and nobody asking the LISA question.  It is not a proof
of equivalence, and it should not be described as one.

## The gate

`test/test_lisa_driver_drift.py`, wired into the `lisa-check` CI job via
`.travis/test-lisa.sh`.  It does **not** assert the gap is empty or that anything was ported.
It asserts that every gap item carries one of `PORT` / `PORTED` / `NA` / `PHYSICS` **with a
reason**, that no item claims `PORTED` while still absent, and that the ledger holds no
entries for items that have left the gap.

*"Does not apply to LISA" is a fine answer; silence is not.*

## Snapshot, 2026-08-15 (junior/rift_O4d @ 364a22fd)

132 items before this pass; 8 ported here, leaving 124.

| decision | n | meaning |
|---|---|---|
| `PORT` | 70 | belongs in LISA, not there yet -- open work |
| `NA` | 43 | does not apply, with the reason |
| `PHYSICS` | 11 | blocked on a physics decision, with the question |
| `PORTED` | 8 | carried across in this pass |

### Ported in this pass -- the fair-draw correctness family (PR #87)

`ln_weights_from_rvs`, `ln_weights_for_posterior`, `_rvs_is_export_resample`,
`_rvs_is_equal_weight`, `_rvs_len`, `_rvs_lnL_convention`, and reads of the `_rvs_is_fairdraw`
/ `_rvs_is_pooled` markers.

The three consumers whose double-weighting PR #87 actually fixed -- the
`--extrinsic-proposal-output` breadcrumb, the `.dgrid` exporter, the `.dslice` reweight core
-- **do not exist in the LISA driver**, so there was no live `w^2` bug there.  What existed was
the hazard: the LISA driver sets `igrand_fairdraw_samples` from `--fairdraw-extrinsic-output`,
so its `_rvs` can be a fair draw, and all seven shared rebind sites already set
`_rvs_is_fairdraw`.  **The marker was arriving and nothing read it.**  This port is preventive,
and it is the "correct thing to reach for" that the audit's Recommendation 1 asks for.

Tests: `test/test_lisa_fairdraw_weights.py` (29), revert-checked -- each fix broken in turn,
the named test confirmed failing, the file restored and verified byte-identical.

Two things deliberately NOT done:

* `ln_weights_for_posterior` passes `use_lnL` **through unresolved**, exactly as the main
  driver does, so a caller that omits it gets the linear reading rather than the run's
  convention.  That is a latent trap **in both drivers**; reproducing it beats having a
  same-named helper behave differently in the two forks.  Worth fixing in both, together.
* `_truthy_option` was initially classified with this family and moved out: its only caller in
  the main driver is the `--interpolate-time` normalizer, so porting it here would have added
  dead code.

### `NA` -- does not apply to LISA (43)

| family | n | why |
|---|---|---|
| `--calibration-*` + 4 helpers | 19 | LIGO/Virgo **spline calibration envelopes**. The LISA driver models no instrument calibration: no envelope directory, no cal nodes, response applied analytically by `factored_likelihood_LISA`. |
| `.dslice` / `.dgrid` distance export | 11 | Data products for a downstream LIGO CIP distance workflow the LISA pipeline does not run. No consumer. |
| `--freqresponse*` | 3 | Finite light-travel-time across the arms for **3G ground** detectors, on `lalsimulation` geometry with an arm length in metres. LISA's finite-size response is not an add-on -- it is the TDI response the driver already applies. |
| `--rotation-*` | 3 | Sidereal time-dependence of an **Earth-based** antenna pattern. The constellation's motion is already in the LISA response; this would apply Earth rotation to a heliocentric detector. |
| data/waveform io | 6 | LISA has its own equivalents under different names -- `--data-integration-window-half` for the storage window, `--internal-waveform-*` fd/L-frame passthroughs, h5 frames instead of gwpy, rate from the frame rather than `--srate-internal`. |
| `--e-freq`, `--save-meanPerAno` | 2 | Ground-based eccentric-waveform path (TEOBResumS); LISA's own export is `--save-eccentricity`. |

### `PHYSICS` -- needs a decision before it can be answered (11)

These are the ones that need you, not more code reading.

1. **`--d-prior-redshift`, `dLofz`, `dVdz`** (4 items incl. constants) — *which cosmology and
   which redshift range should a LISA distance prior use?*  Arguably **more** important for
   LISA than for ground-based work, since MBHB sit at z~1-20 where a Euclidean `d^2` prior is
   badly wrong -- but the main driver's helpers were built and gridded for the ground-based
   range.
2. **`--internal-reparam-dl-incl`, `_reparam_A_of_incl`, `_REPARAM_*`** (5 items) — *does the
   quadrupole amplitude `A(iota)=sqrt(((1+cos^2 i)/2)^2+cos^2 i)` remain the right axis to
   reparameterize distance against under the LISA TDI response?*  It is a pure l=|m|=2
   statement; LISA MBHB are strongly higher-mode and TDI mixes the polarizations differently,
   so the degeneracy it straightens may not be the degeneracy LISA has.
3. **`--limit-right-ascension`, `--limit-declination`** — *what should a sky zoom box mean for
   LISA?*  The driver reuses the key names `right_ascension`/`declination` for its sampled sky
   pair, but the values are ecliptic and may be further rotated by
   `--internal-sky-network-coordinates`.  LISA already has
   `--ecliptic-latitude`/`--ecliptic-longitude`/`--lisa-fixed-sky`, which may be the intended
   mechanism.  (`--limit-psi`/`--limit-inclination` have no such ambiguity and are `PORT`.)
4. **`--sampler-warmstart-samples`** — *what frame are the named columns of a LISA pilot file
   in?*  Same key-names-different-meaning problem; needs a stated convention, or a pilot
   written by the LISA driver itself.

### `PORT` -- open work, highest value first (70)

Nothing here is blocked on physics; all of it is sampler-agnostic or pure plumbing.

| family | n | note |
|---|---|---|
| L0 rescue + warm-start state | 15 | **Highest value.** Triggers on low `n_eff`; LISA MBHB are high-SNR, the regime that stalls. `_snapshot_pass_state`/`_restore_pass_state` must port **as a set** -- Finding 5 was a rejected warm pass restoring `_rvs` but not the reserve. |
| portfolio tuning | 12 | Reachable today via LISA's `--sampler-portfolio-args` eval-dict; porting is pipeline parity. |
| GMM tuning | 7 | Pure pass-through to `mcsamplerEnsemble`. |
| MC-error replicas + pooling | 7 | Includes `_pool_replica_rvs`; port the **per-replica sequence** form, not the boolean (Finding 6). |
| extrinsic proposal handoff | 6 | `--extrinsic-proposal-output` is a Finding-2 site: port it **on top of** `ln_weights_for_posterior`, never with a bare `w`. |
| lnZ / n_eff helpers | 3 | `_lnZ_of_rvs`, `_kish_neff_of_rvs`, `_lnZ_of_reserve_or_rvs` -- needed by the two families above. |
| AV state + binning | 3 | `--sampler-save/load-state`, `--sampler-anisotropic-bins`. |
| misc plumbing | 17 | `--limit-psi`/`--limit-inclination` (port the **post-#58** form, incl. the `cos(iota)` endpoint swap), `--check-good-enough`, `--random-event`, `--fairdraw-extrinsic-output-n-max`, interpolate-time normalizer, etc. |

**One trap recorded against `--fairdraw-extrinsic-output-n-max`:** the LISA driver currently
hardcodes the cap to `opts.n_eff`, while main's default for the flag is **5**.  Adopting main's
default verbatim would silently shrink every LISA export by orders of magnitude.  Port the flag
with LISA's present behaviour as its default.

## Note on CI

The LISA driver is **not** uncovered -- the `lisa-check` job runs nine test files.  But all nine
are import / contract / smoke level: they check the driver loads, exposes its CLI surface and
runs a synthetic demo.  None asserts anything about integrator weighting or fair-draw
correctness, which is how 2,357 lines of drift accumulated with CI green.  That is the gap the
drift gate closes -- not by testing the physics, but by refusing to let a new item through
without a recorded human decision.
