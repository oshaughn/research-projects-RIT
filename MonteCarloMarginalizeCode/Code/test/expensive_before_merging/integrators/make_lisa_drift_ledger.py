#!/usr/bin/env python3
"""
Regenerate ``lisa_drift_ledger.json`` -- the recorded decision for every item the main
ILE driver has and the LISA ILE driver does not.

    python3 make_lisa_drift_ledger.py            # rewrite the ledger
    python3 make_lisa_drift_ledger.py --dry-run  # show what would change
    python3 audit_lisa_driver_drift.py --check   # CI gate over the result

The gap itself is computed by ``audit_lisa_driver_drift.py``; this file holds only the
JUDGEMENTS, as ordered (pattern -> decision + reason) rules so a whole family is decided
once.  First match wins, so put specific items above their family.

DECISIONS
    PORT     belongs in LISA, not there yet.  An open work item.
    PORTED   carried across.  The audit re-checks these: a PORTED item still missing from
             the LISA driver fails the build.
    NA       does not apply to LISA, with the reason.
    PHYSICS  cannot be answered without a physics decision, with the question.

An item matching NO rule is reported and left out of the ledger, so ``--check`` fails on
it.  That is the intended path for newly-drifted code: it must be classified by a person.

WHY THESE DECISIONS LOOK THE WAY THEY DO
The two drivers import the SAME integrators and expose the SAME ``ok_lnL_methods``
(``GMM, adaptive_cartesian, adaptive_cartesian_gpu, AV, portfolio``, verified identical
2026-08-15).  So anything that is pure sampler plumbing applies to LISA by construction and
is PORT; the NA items are the ones tied to a ground-based detector, to LIGO/Virgo
calibration envelopes, or to a downstream pipeline stage LISA does not run.
"""
import argparse
import json
import os
import re
import sys

import audit_lisa_driver_drift as audit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lisa_drift_ledger.json")

# ---------------------------------------------------------------------------------------
# Ordered rules.  (regex over the audit key "CATEGORY:name", decision, reason)
# First match wins.
# ---------------------------------------------------------------------------------------
RULES = [

    # ---------------------------------------------------------------- the fair-draw family
    # PORTED in this pass.  These are the PR #87 correctness helpers.  They are pure
    # functions of the _rvs record plus the sampler's own provenance markers, and the
    # markers are already set by the shared integrators at all seven rebind sites, so they
    # already arrive on LISA's sampler objects at runtime -- only the driver-side readers
    # were missing.
    (r"^FUNC:ln_weights_from_rvs$", "PORTED",
     "Importance weight of an _rvs record. Pure function of the record; no extrinsic "
     "coordinate assumptions. LISA sets igrand_fairdraw_samples, so its records can be "
     "fair draws and need the same answer."),
    (r"^FUNC:ln_weights_for_posterior$", "PORTED",
     "How rows should be weighted to REPRESENT THE POSTERIOR, as distinct from their "
     "importance weight. Returns zeros on an equal-weight record. This is the helper "
     "that makes the w^2 double-weighting defect unrepresentable."),
    (r"^FUNC:_rvs_is_export_resample$", "PORTED",
     "Predicate: rows were drawn proportional to w (survives pooling). Reads the shared "
     "marker the integrators already set."),
    (r"^FUNC:_rvs_is_equal_weight$", "PORTED",
     "Predicate: record is globally equal-weight (fairdraw and not pooled). Finding 6 "
     "split this from _rvs_is_export_resample; porting one without the other rebuilds "
     "the flag-answering-two-questions bug."),
    (r"^FUNC:_rvs_len$", "PORTED",
     "Row count of an _rvs record, tolerant of the tuple-keyed sky column. Support "
     "helper for the above."),
    (r"^ATTR:_rvs_is_fairdraw$", "PORTED",
     "Set by all seven shared rebind sites in RIFT/integrators/, so it already reaches "
     "LISA at runtime; the LISA driver simply never read it."),
    (r"^ATTR:_rvs_is_pooled$", "PORTED",
     "READER ONLY, deliberately. The marker is read by _rvs_is_equal_weight and carried "
     "by the pass snapshot/restore; nothing in this driver ever SETS it, because there "
     "is no replica pooling here yet. Main's reset-on-entry (Finding 7: the marker "
     "outliving a FAILED event) is therefore NOT ported and MUST come with "
     "--mc-error-replicas -- without it the first pooled record would leave the marker "
     "set on the next event. Note this is also the ATTR category's blind spot: a name "
     "read anywhere counts as present, so reader-ported/writer-missing looks closed."),

    # ---------------------------------------------------------------------- the _rvs record
    # PORT decision, not NA: these are PREREQUISITES of helpers already marked PORTED.
    # ln_weights_for_posterior / _snapshot_pass_state / _restore_pass_state call them by
    # name, so leaving them out of the LISA driver does not keep the fork simpler -- it
    # breaks the ported copies outright.  The INTEGRATORS are shared between the two
    # drivers, so the samplers already carry SamplerOutputMixin and populate a record;
    # only the driver-side accessors had to come across.
    (r"^FUNC:(_rvs_record_for|_sampler_keeps_records)$", "PORTED",
     "Driver-side accessors for the sampler's RvsRecord: the identity-guarded lookup, "
     "and the 'does this backend keep records at all' test. Prerequisites of the "
     "already-PORTED ln_weights_for_posterior and the pass-state snapshot/restore."),
    (r"^FUNC:(_internal_record_of|_rebound_record|_lw_of)$", "PORTED",
     "The rest of the record accessor set: the INTERNAL record (handed back only so the "
     "driver can thread it, never as user-facing API), the post-rebind rebuild, and the "
     "weight helper. Ported as a SET with the above -- the callers reference them "
     "directly, so a partial port is a NameError at runtime, not a smaller fork."),

    # ---------------------------------------------------------------------- lnZ / n_eff
    (r"^FUNC:_lnZ_of_rvs$", "PORTED",
     "Evidence of an _rvs record with the already_pooled/fairdraw correction. Landed "
     "with the L0 rescue gate, which is its first consumer here."),
    (r"^FUNC:_kish_neff_of_rvs$", "PORTED",
     "Kish n_eff of a record. Landed with _lnZ_of_rvs; its own consumer (replica "
     "pooling) arrives in the MC-error pass."),
    (r"^FUNC:_lnZ_of_reserve_or_rvs$", "PORTED",
     "Reads a pass's lnZ from the points it RETAINED where available, so the reject "
     "gate is not comparing two differently-sized fair-draw artifacts."),
    (r"^FUNC:(_snapshot_pass_state|_restore_pass_state)$", "PORTED",
     "Snapshot/restore of everything that must travel with a put-back pass -- the "
     "reserve and the fair-draw marker included (Finding 5). Ported as a SET with the "
     "rescue; either one alone rebuilds the defect."),
    (r"^FUNC:(_warm_seed_reserve_for|_warm_seed_geometry|_clear_warm_state)$", "PORTED",
     "Shared reserve lookup (with the column-order guard), adaptive-axis geometry for "
     "the rank test, and the warm-state clear that reaches portfolio MEMBERS."),
    (r"^ATTR:_warm_seed_reserve$", "PORTED",
     "The retained-sample reserve the rescue seeds from and the snapshot carries."),
    (r"^OPTION:--sampler-warmstart-retry-neff$", "PORTED",
     "The L0 rescue trigger. High value for LISA: MBHB are high-SNR, which is the "
     "regime that stalls at n_eff~1."),
    (r"^OPTION:--sampler-l0-rescue-", "PORTED",
     "L0 rescue tuning, defaults and help text kept identical to the main driver "
     "(including reject-dlnZ 3.0, the measured value -- see "
     "L0_REJECT_DLNZ_MEASUREMENT.md). Pinned by test_lisa_l0_rescue.py."),
    (r"^OPTION:--sampler-sequential-warmstart-deltalnL$", "PORTED",
     "The lnL window build_warm_seed keeps. Consumed by the L0 rescue, so it landed "
     "with that pass rather than with the sequential warm start it is named for."),

    # --------------------------------------------------------------- L0 rescue / warm start
    (r"^OPTION:--reject-collapsed-live-volume$", "PORTED",
     "AV live-volume collapse rejection. AV is wired in the LISA driver identically. "
     "NOTE the main driver calls its gate TWICE -- first run and replica pool -- and only "
     "the first call exists here, because there is no pooling yet; the second MUST be "
     "added with --mc-error-replicas or the flag is bypassed for the case pooling creates."),
    (r"^FUNC:analyze_event\._reject_if_collapsed$", "PORTED",
     "Hoisted to module level rather than nested, because this driver has TWO "
     "analyze_event variants. The audit matches FUNC items on the bare name for exactly "
     "this reason."),
    (r"^OPTION:--sampler-sequential-warmstart$", "PORT",
     "Warm-start each intrinsic point from the previous one's cloud. Applies whenever "
     "--n-events-to-analyze>1, which LISA supports. Its snapshot/restore prerequisites "
     "(Finding 5) already landed with the L0 rescue, so this is now capture + the "
     "event-loop wiring only."),
    (r"^OPTION:--sampler-sequential-warmstart-cover-frac$", "PORT",
     "Coverage floor for the above; meaningless without it, so they travel together."),
    (r"^OPTION:--sampler-anisotropic-bins$", "PORTED",
     "AV per-axis bin counts during contraction. AV is wired in LISA, and the argument "
     "for it is if anything stronger there: the LISA extrinsic axes are no more "
     "isotropic than the ground-based ones, and a sky pair that localizes tightly "
     "while distance stays broad is the exact case this exists for."),
    (r"^OPTION:--sampler-(save|load)-state$", "PORTED",
     "AV live-volume state serialization. AV is wired in LISA; the state is the "
     "sampler's own internal grid, so it carries no LIGO-specific convention."),
    (r"^OPTION:--sampler-warmstart-(cover-frac|inflate)$", "PORT",
     "Coverage floor and inflation for a handed-off seed. Pure geometry on the "
     "sampled unit cube."),
    (r"^OPTION:--sampler-warmstart-samples$", "PORT",
     "RESOLVED (RO 2026-08-16). The convention does not matter: the seed is points in the "
     "sampler's OWN coordinate space, read positionally against params_ordered, so any "
     "self-consistent choice works and the ecliptic sky answer already determines it. The "
     "hazard is only that a mismatch is UNDETECTABLE -- ecliptic lambda and RA share "
     "[0,2pi), beta and dec share [-pi/2,pi/2], so no range check separates them and a "
     "wrong-frame seed silently contracts the live volume around the wrong region. SCOPE "
     "(RO): these files are used INTERNALLY within a homogeneous run -- we are talking to "
     "ourselves, not to heterogeneous tooling -- so keep it simple: a one-line frame stamp "
     "in the file header written by the producer, warn if it is absent or disagrees. Do NOT "
     "build a validation framework for it."),

    # --------------------------------------------------------------------- MC error replicas
    (r"^OPTION:--mc-error-(replicas|sigma-trigger|ess-trigger|khat-trigger)$", "PORTED",
     "Replica-based lnL error stabilization. Triggers on weight-tail diagnostics of the "
     "run's own weights; nothing detector-specific. Valuable for LISA for the same "
     "reason as for high-SNR ground events: the reported sigma is the thing downstream "
     "CIP trusts."),
    (r"^FUNC:_pool_replica_rvs(\._block_resampled|\._block_record)?$", "PORTED",
     "Pools replica records by evidence, verbatim -- including the PER-REPLICA "
     "already_resampled sequence (Finding 6). A single global boolean is wrong near the "
     "n_extr boundary, where a run produces a MIXTURE of raw and resampled replicas."),
    (r"^FUNC:(analyze_event\.)?_extract_mc_diag$", "PORTED",
     "Diagnostics for the replica triggers. Hoisted to module level (two analyze_event "
     "variants); the audit matches FUNC on the bare name for exactly this reason."),

    # ------------------------------------------------------------------------ GMM plumbing
    (r"^OPTION:--internal-gmm-", "PORT",
     "mcsamplerEnsemble (GMM) tuning. LISA wires that sampler and exposes the same "
     "'GMM' method string, so these knobs are reachable physics-wise but simply not "
     "plumbed. Pure pass-through. Caveat for whoever ports --internal-gmm-sky-components: "
     "the default grouping is (sky)(distance,inclination)(psi,phi), and 'sky' for LISA "
     "is the ecliptic pair -- the grouping still makes sense, the docstring does not."),

    # ------------------------------------------------------------------ portfolio plumbing
    (r"^OPTION:--portfolio-", "PORTED",
     "mcsamplerPortfolio freeze/allocation policy. Definitions copied verbatim and the "
     "_freeze_policy_kwargs assembly is textually identical to the main driver's, so "
     "unset options (None) stay out of the dict and the sampler keeps its own defaults. "
     "--portfolio-varaha-can-freeze wins over --portfolio-varaha-never-freeze, as there."),

    # ------------------------------------------------------------------- NF flow plumbing
    (r"^OPTION:--nf-flow-(load|save)$", "PORT",
     "Normalizing-flow persistence is detector-agnostic, but the LISA portfolio factory "
     "currently constructs only AV, GMM, and adaptive_cartesian_gpu members. Port the NF "
     "member construction and route load/save to that member before exposing these flags; "
     "hooks on the portfolio aggregate are a silent no-op because it has no flow API."),

    # --------------------------------------------------------- extrinsic proposal handoff
    (r"^OPTION:--extrinsic-proposal-output$", "PORT",
     "Fits the run's extrinsic posterior to a GMM and writes it as a breadcrumb. This "
     "is one of the three Finding-2 double-weighting sites, so it MUST be ported on top "
     "of ln_weights_for_posterior (done here) and never with a bare w."),
    (r"^OPTION:--extrinsic-proposal-(breadcrumb|adapt)$", "PORT",
     "Consumes the breadcrumb above. Ports with it."),
    (r"^OPTION:--extrinsic-proposal-field(-cover-frac|-inflate)?$", "PORT",
     "AV proposal-field handoff, built by util_BuildProposalField.py from a previous "
     "ILE iteration. Sampler-agnostic; blocked only on the LISA pipeline growing that "
     "stage, so it is a work item rather than an exclusion."),

    # ------------------------------------------------------------------------ fair-draw size
    (r"^OPTION:--fairdraw-extrinsic-output-n-max$", "PORT",
     "Caps rows per fair-draw export. LISA currently hardcodes this to opts.n_eff at "
     "the igrand_fairdraw_samples_max call site. WARNING for the port: main's default "
     "is 5, so adopting main's default verbatim would silently shrink every LISA "
     "export by orders of magnitude. Port the flag with LISA's present behaviour as "
     "its default."),
    (r"^FUNC:_equal_weight_fairdraw_for_serialization$", "PORT",
     "Completes a sampler-side fair draw that intentionally did not fire because it "
     "would not shrink a tiny retained record, but only on the copy written to XML. "
     "LISA has the same skip-on-no-shrink and serialization boundary, so port this "
     "with --fairdraw-extrinsic-output-n-max while preserving its larger LISA default."),

    # ------------------------------------------------------- LIGO/Virgo calibration envelopes
    (r"^OPTION:--calibration-", "NA",
     "LIGO/Virgo spline calibration-envelope marginalization. The LISA driver models no "
     "instrument calibration: it takes no envelope directory, has no cal nodes, and its "
     "response is applied analytically by factored_likelihood_LISA. LISA calibration, if "
     "it is ever modelled, will not have this data product or this spline parameterization, "
     "so porting the LIGO machinery would be actively misleading."),
    (r"^FUNC:(_cal_setup_prior_with_nodes|_draw_more_calibration_draws)$", "NA",
     "Calibration-envelope internals; see the --calibration-* reason."),
    (r"^FUNC:_cal_rng$", "NA",
     "Per-stream RNG for the calibration-side auxiliary draws (the error probe and the "
     "adaptive growth of the cal draw set), so those stay reproducible under --seed instead "
     "of taking fresh OS entropy. Calibration-envelope internals; see the --calibration-* "
     "reason. NOT a seeding gap on the LISA side: this is a thin per-stream counter over "
     "RIFT.integrators.seeding.derived_rng, which is a shared module both drivers already "
     "import, and the LISA driver calls seed_everything on the same footing as the main "
     "one. If LISA ever models calibration, it wants derived_rng directly, not this wrapper."),
    (r"^FUNC:analyze_event\._cal_error_probe(\._draw_dist)?$", "NA",
     "Calibration Monte-Carlo error probe; see the --calibration-* reason."),

    # ------------------------------------------------------- ground-based detector geometry
    (r"^OPTION:--rotation-(slow|n-harmonics|p-max)$", "NA",
     "Sidereal time-dependence of an EARTH-BASED antenna pattern F(t). The LISA "
     "constellation's motion is already carried by the LISA response itself "
     "(factored_likelihood_LISA + the h5/TDI frames), so this correction is both "
     "unnecessary and wrong there -- it would apply Earth rotation to a heliocentric "
     "detector."),
    (r"^OPTION:--freqresponse(-arm-length|-qmax)?$", "NA",
     "Finite light-travel-time transfer across the arms for 3G ground detectors "
     "(CE/ET), built on lalsimulation detector geometry and an arm-length override in "
     "metres. LISA's finite-size response is not an add-on: it is the whole point of "
     "the TDI response the LISA driver already applies."),
    (r"^OPTION:--e-freq$", "NA",
     "TEOBResumS eccentric-frequency convention. Tied to a ground-based eccentric "
     "waveform path the LISA driver does not offer (it takes --modes / h5 frames)."),

    # ---------------------------------------------------------- distance slice / grid export
    (r"^OPTION:--(export-distance-slices|distance-slice-|n-distance-slice-)", "NA",
     "The .dslice export and its placement/tuning knobs. This is a data product for a "
     "downstream LIGO CIP distance workflow that the LISA pipeline does not run; there "
     "is no consumer. If a LISA distance workflow is ever built, note that the .dslice "
     "reweight core was the third Finding-2 site and must not be revived in its "
     "pre-#87 form."),
    (r"^OPTION:--export-marginal-distance-grid$", "NA",
     "The .dgrid export. Same absent consumer as .dslice, and the second Finding-2 "
     "double-weighting site."),

    # ----------------------------------------------------------------- cosmology / d prior
    (r"^OPTION:--d-prior-redshift$", "PORT",
     "ANSWERED (RO 2026-08-16): Planck15 via the framework helper, "
     "RIFT.likelihood.priors_utils.get_astropy_cosmology('Planck15'). RESOLVED AT SOURCE -- "
     "the MAIN driver has been moved to that helper too (it previously built its own "
     "FlatLambdaCDM from lal.H0_SI/lal.OMEGA_M = 67.900/0.3065 with a hardcoded fallback), "
     "so there is no divergence to port around: both codes now ask the same helper and a "
     "change is made in one place. Pinned by test_cosmology_single_source.py."),
    (r"^FUNC:(dLofz|dVdz)$", "PORT",
     "Cosmology helpers behind --d-prior-redshift. Planck15 via the framework helper; the "
     "interpolation grid still needs a z ceiling that covers MBHB (z~20), which is a "
     "gridding choice rather than a physics decision."),

    # -------------------------------------------------------------- distance/incl reparam
    (r"^OPTION:--internal-reparam-dl-incl$", "PORT",
     "ANSWERED (RO 2026-08-16): 'should be good enough; it is a testable axis though -- "
     "do not guess, measure.' So: port it, but do NOT enable by default until measured. The "
     "test is cheap and direct -- compare n_eff / lnZ scatter with and without the "
     "reparameterization on a fixed LISA MBHB intrinsic point, since if the axis is wrong "
     "for TDI it shows up as no improvement or worse conditioning, not as a bias."),
    (r"^FUNC:_reparam_A_of_incl$", "PORT",
     "Implementation of --internal-reparam-dl-incl; ports with it, measured before default-on."),
    (r"^CONST:_REPARAM_", "PORT",
     "Tuning constants for --internal-reparam-dl-incl; port verbatim, re-tune only if the "
     "measurement says the axis helps."),

    # ---------------------------------------------------------------------- extrinsic boxes
    (r"^OPTION:--limit-(psi|inclination)$", "PORT",
     "Zoom-box limits on psi and inclination. These parameters mean the same thing in "
     "both drivers and LISA exposes --inclination-cosine-sampler, which is exactly the "
     "case junior PR #58 found silently ignored -- so port the POST-#58 form, including "
     "the cos(iota) endpoint swap."),
    (r"^OPTION:--limit-(right-ascension|declination)$", "PORT",
     "ANSWERED (RO 2026-08-16): LISA and LIGO are never overlapping use cases, so follow "
     "the convention already in this driver, document it in the help string, and DO NOT "
     "rename the options. VERIFIED that convention is ECLIPTIC: the sampled "
     "right_ascension/declination columns flow to P.phi/P.theta and then to "
     "lisa_sky_lamda/lisa_sky_beta, i.e. ecliptic longitude/latitude, under the historical "
     "key names. So --limit-right-ascension bounds lambda and --limit-declination bounds "
     "beta; say exactly that in the help text. Port the post-PR#58 form including the "
     "cos(iota)/cos(dec) endpoint swap under the cosine samplers."),

    (r"^OPTION:--limit-distance$", "PORT",
     "Sampling-only distance box: narrows what distance is DRAWN from while the prior keeps "
     "its full [--d-min,--d-max] normalization, so lnZ stays on the full-range scale. Port it, "
     "and port the SPLIT rather than the option alone. VERIFIED 2026-09-02 that the LISA "
     "driver still carries the one-range form the main driver was just moved off "
     "(integrate_likelihood_extrinsic_batchmode_lisa:1021-1024: dist_sampler and "
     "dist_prior_pdf are both built from param_limits['distance']), which normalizes the "
     "Euclidean density over whatever the sampler happens to draw from -- so narrowing for "
     "cost there would silently rescale the evidence. mcsampler.distance_sampler_kwargs() "
     "already takes the sampling range and the prior range as two arguments and is shared "
     "code, so the port is a call-site change, not a reimplementation. The motivation is "
     "STRONGER on LISA than on ground-based data: measured on real LIGO data at rho ~ 82, a "
     "box tracking the posterior removes 0.37 +- 0.11 nats of sampling bias the full-range "
     "run was carrying (4.16 nats with --no-adapt-distance), and MBHB SNRs are one to two "
     "orders of magnitude higher, where the posterior is narrower still relative to the same "
     "prior (RIFT_roboto_paper analyses/limit_distance_e2e/). CARRY THE REFUSALS, and note "
     "only one of the three transfers today: LISA HAS --distance-marginalization (:245), so "
     "refuse there for the same reason -- no distance sampler exists to narrow. It has "
     "neither --d-prior-redshift nor --internal-reparam-dl-incl, so those two refusals have "
     "nothing to attach to yet; --internal-reparam-dl-incl is itself a PORT item above, so "
     "whichever of the two lands second owes the refusal. LISA's --d-prior set is also "
     "different (Euclidean|uniform|pseudo_cosmo, no cosmo/cosmo_sourceframe), so the cosmo "
     "branch of the main driver's narrowing block has no counterpart to port."),

    # --------------------------------------------------------------------- data / waveform io
    (r"^OPTION:--internal-data-storage-window-half$", "NA",
     "Half-width of the main driver's internal precompute storage window. The LISA "
     "driver has its own equivalent under a different name, --data-integration-window-half, "
     "which it passes straight into PrecomputeAlignedSpinLISA. Same role, already present."),
    (r"^OPTION:--internal-use-gwpy$", "NA",
     "gwpy low-level frame io. The LISA driver reads its data from h5 frames "
     "(--h5-frame/--h5-frame-FD), not from GWF via gwpy."),
    (r"^OPTION:--internal-waveform-(taper|extra-kwargs)$", "NA",
     "lalsimulation taper / extra-kwargs passthrough for the ground-based waveform "
     "path. The LISA driver has its own passthroughs for the generator it uses "
     "(--internal-waveform-extra-lalsuite-args, --internal-waveform-fd-L-frame, "
     "--internal-waveform-fd-no-condition)."),
    (r"^OPTION:--srate-internal$", "NA",
     "Separate internal sampling rate for the ground-based precompute. LISA's "
     "precompute takes its rate from the h5 frame and P.deltaT; there is no second "
     "internal rate to set."),
    (r"^OPTION:--srate-resample-time-marginalization$", "PORT",
     "Interpolate the lnL time series onto a finer grid before time resampling. LISA "
     "already has --resample-time-marginalization and its own time-resampling block, "
     "so this is the matching resolution knob and applies directly."),
    (r"^CONST:_TI_LEGACY_BOOLEAN$", "PORT",
     "Legacy-boolean vocabulary for --interpolate-time. Main (PR #97) now accepts STENCIL "
     "NAMES there -- nearest/cubic/sinc -- normalizing into opts._noloop_time_interp, with "
     "this tuple for back-compat and an explicit typo guard so a misspelling is not "
     "absorbed as falsey. LISA still passes the raw --interpolate-time value straight to "
     "the likelihood, so porting means normalizing it AND teaching the LISA time path the "
     "stencil name; it travels with _normalize_interpolate_time_argv and _truthy_option."),
    (r"^FUNC:_normalize_interpolate_time_argv$", "PORT",
     "Normalizes --interpolate-time argv forms. LISA exposes --interpolate-time, so "
     "the same normalization applies."),
    (r"^OPTION:--time-marginalization-quadrature$", "PORT",
     "Selects the rule for the TIME integral of the marginalized likelihood "
     "(simpson, the unchanged default, or the opt-in band-limited refinement). LISA "
     "carries the SAME defect this addresses: factored_likelihood_LISA.py integrates "
     "exp(lnL(t)) with Simpson at the fixed data spacing, while the integrand's width "
     "sigma_t = 1/(2 pi rho sigma_f) is set by the signal and shrinks as 1/rho -- so it "
     "under-resolves its own integrand, worse at higher SNR. PORT, not NA. But porting "
     "is NOT just wiring the flag through, and the prerequisite is the whole question: "
     "the band-limited argument needs kappa band-limited below Nyquist AND rho_sq "
     "time-INDEPENDENT. The main driver refuses --rotation-slow and --freqresponse for "
     "exactly that second condition, and a response that varies across the observation "
     "is the normal case for LISA, not an exotic one. So the LISA port must first "
     "establish whether its self-term is time-independent over the integration window; "
     "if it is not, the honest outcome is a documented refusal on that path rather than "
     "a flag that silently integrates the wrong thing. Note also that the LISA site "
     "integrates on axis=0, not the last axis."),
    (r"^OPTION:--internal-precompute-ignore-threshold$", "PORT",
     "Drops negligible modes during precompute. LISA is mode-heavy (--modes, "
     "--restricted-mode-list-file) and pays more per mode than a ground-based run, so "
     "if anything this matters more there. No LIGO-specific assumption."),

    # ------------------------------------------------------------------------------- misc
    (r"^OPTION:--check-good-enough$", "PORT",
     "Early-exit when the pipeline has written an 'ile_good_enough' sentinel. Pipeline "
     "plumbing, detector-agnostic."),
    (r"^OPTION:--random-event$", "PORT",
     "Pick a random event from the input file. Detector-agnostic; flagged dangerous in "
     "its own help text for oversampling reasons that apply equally to LISA."),
    (r"^OPTION:--save-samples-process-params$", "PORT",
     "Retain the process_params table in the XML output. Pure output plumbing."),
    (r"^OPTION:--save-meanPerAno$", "NA",
     "Exports the eccentric mean anomaly. Tied to the ground-based eccentric waveform "
     "path (see --e-freq); the LISA driver's own eccentricity export is "
     "--save-eccentricity."),
    (r"^OPTION:--(force-hyperbolic-22|save-EOB-parameters|save-hyperbolic)$", "NA",
     "Controls or exports the ground-based external-TEOBResumS advanced-physics path. "
     "The LISA driver does not call that waveform path or write its a6c/E0/p_phi0 "
     "composite layout."),
    (r"^OPTION:--calibration-spline-count$", "NA", "See the --calibration-* reason."),
    (r"^CONST:_SEQ_WS_PENDING$", "PORT",
     "Sentinel for the deferred sequential warm-start capture; ports with "
     "--sampler-sequential-warmstart."),
    (r"^FUNC:_truthy_option$", "PORT",
     "Tolerant truthiness for optparse values that may arrive as strings from the pipe. "
     "Belongs with _normalize_interpolate_time_argv, its ONLY caller in the main driver "
     "(opts._noloop_time_interp), not with the fair-draw family -- porting it alongside "
     "those helpers would have added dead code to the LISA driver."),
    (r"^FUNC:_rvs_lnL_convention$", "PORTED",
     "Resolves the stored-integrand convention from the run's rvs_integrand_is_lnL. "
     "Ported alongside the weight helpers because it is how a caller is SUPPOSED to "
     "obtain use_lnL: ln_weights_for_posterior passes the argument through unresolved "
     "in both drivers, so omitting it silently yields the linear reading."),
]


def classify(key):
    for pat, decision, reason in RULES:
        if re.search(pat, key):
            return decision, reason
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gap, _extras = audit.compute_gap()
    entries, unmatched = {}, []
    for item in gap:
        decision, reason = classify(item["key"])
        if decision is None:
            unmatched.append(item)
            continue
        entries[item["key"]] = {"decision": decision, "reason": reason}

    counts = {}
    for e in entries.values():
        counts[e["decision"]] = counts.get(e["decision"], 0) + 1
    print("classified %d/%d gap items: %s" % (
        len(entries), len(gap), "  ".join("%s=%d" % kv for kv in sorted(counts.items()))))

    if unmatched:
        print("\n%d item(s) match NO rule -- add one, or they fail --check:" % len(unmatched))
        for item in unmatched:
            print("   %-58s main:%d" % (item["key"], item["main_line"]))

    if args.dry_run:
        return 1 if unmatched else 0

    payload = {
        "_comment": "GENERATED by make_lisa_drift_ledger.py -- edit the RULES there, not this file.",
        "entries": entries,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s" % os.path.relpath(OUT, HERE))
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
