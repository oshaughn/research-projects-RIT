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
     "Written by the ILE around _pool_replica_rvs. Ported as the reset-on-entry "
     "discipline plus the reader, so _rvs_is_equal_weight is correct even though LISA "
     "does not pool yet (Finding 7: the marker outliving a FAILED event is what made "
     "this dangerous, and entry-reset is what fixes it)."),

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
    (r"^OPTION:--reject-collapsed-live-volume$", "PORT",
     "AV live-volume collapse rejection. AV is wired in the LISA driver identically."),
    (r"^FUNC:analyze_event\._reject_if_collapsed$", "PORT",
     "Implementation of --reject-collapsed-live-volume."),
    (r"^OPTION:--sampler-sequential-warmstart$", "PORT",
     "Warm-start each intrinsic point from the previous one's cloud. Applies whenever "
     "--n-events-to-analyze>1, which LISA supports. Its snapshot/restore prerequisites "
     "(Finding 5) already landed with the L0 rescue, so this is now capture + the "
     "event-loop wiring only."),
    (r"^OPTION:--sampler-sequential-warmstart-cover-frac$", "PORT",
     "Coverage floor for the above; meaningless without it, so they travel together."),
    (r"^OPTION:--sampler-anisotropic-bins$", "PORT",
     "AV per-axis bin counts during contraction. AV is wired in LISA, and the argument "
     "for it is if anything stronger there: the LISA extrinsic axes are no more "
     "isotropic than the ground-based ones, and a sky pair that localizes tightly "
     "while distance stays broad is the exact case this exists for."),
    (r"^OPTION:--sampler-(save|load)-state$", "PORT",
     "AV live-volume state serialization. AV is wired in LISA; the state is the "
     "sampler's own internal grid, so it carries no LIGO-specific convention."),
    (r"^OPTION:--sampler-warmstart-(cover-frac|inflate)$", "PORT",
     "Coverage floor and inflation for a handed-off seed. Pure geometry on the "
     "sampled unit cube."),
    (r"^OPTION:--sampler-warmstart-samples$", "PHYSICS",
     "QUESTION: what frame are the named columns of a LISA pilot file in? The reader "
     "expects right_ascension/declination/inclination/psi/phi_orb/distance, and the "
     "LISA driver does use those KEY NAMES internally -- but they carry ecliptic "
     "(and, with --internal-sky-network-coordinates, rotated) values, so a file is only "
     "meaningful if the writer and reader agree on the convention. Needs a stated "
     "convention before it can be ported, or a pilot written by the LISA driver itself."),

    # --------------------------------------------------------------------- MC error replicas
    (r"^OPTION:--mc-error-(replicas|sigma-trigger|ess-trigger|khat-trigger)$", "PORT",
     "Replica-based lnL error stabilization. Triggers on weight-tail diagnostics of the "
     "run's own weights; nothing detector-specific. Valuable for LISA for the same "
     "reason as for high-SNR ground events: the reported sigma is the thing downstream "
     "CIP trusts."),
    (r"^FUNC:_pool_replica_rvs(\._block_resampled)?$", "PORT",
     "Pools replica records by evidence. Ports with --mc-error-replicas. NOTE its "
     "per-replica already_resampled sequence (Finding 6): a single global boolean is "
     "wrong near the n_extr boundary, so port the sequence form, not the boolean."),
    (r"^FUNC:analyze_event\._extract_mc_diag$", "PORT", "Diagnostics for the replica triggers."),

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
    (r"^OPTION:--nf-flow-(load|save)$", "PORTED",
     "Normalizing-flow persistence. Neither driver lists an NF method in ok_lnL_methods "
     "(identical lists), so NF is reached only as a portfolio member -- equally "
     "available to LISA. Both hooks are hasattr-guarded, so they are a no-op for every "
     "other sampler."),

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

    # ------------------------------------------------------- LIGO/Virgo calibration envelopes
    (r"^OPTION:--calibration-", "NA",
     "LIGO/Virgo spline calibration-envelope marginalization. The LISA driver models no "
     "instrument calibration: it takes no envelope directory, has no cal nodes, and its "
     "response is applied analytically by factored_likelihood_LISA. LISA calibration, if "
     "it is ever modelled, will not have this data product or this spline parameterization, "
     "so porting the LIGO machinery would be actively misleading."),
    (r"^FUNC:(_cal_setup_prior_with_nodes|_draw_more_calibration_draws)$", "NA",
     "Calibration-envelope internals; see the --calibration-* reason."),
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
    (r"^OPTION:--d-prior-redshift$", "PHYSICS",
     "QUESTION: which cosmology and which redshift range should a LISA distance prior "
     "use? This is arguably MORE important for LISA than for ground-based work -- MBHB "
     "sit at z~1-20 where a Euclidean d^2 prior is badly wrong -- but the main driver's "
     "helper was built and gridded for the ground-based range. Needs a stated "
     "cosmology and a z ceiling before porting."),
    (r"^FUNC:(dLofz|dVdz)$", "PHYSICS",
     "Cosmology helpers behind --d-prior-redshift. Same question: the interpolation "
     "range has to be re-chosen for MBHB redshifts."),

    # -------------------------------------------------------------- distance/incl reparam
    (r"^OPTION:--internal-reparam-dl-incl$", "PHYSICS",
     "QUESTION: does the quadrupole amplitude A(iota)=sqrt(((1+cos^2 i)/2)^2+cos^2 i) "
     "remain the right axis to reparameterize distance against under the LISA TDI "
     "response? The reparameterization is a pure l=|m|=2 statement; LISA MBHB are "
     "strongly higher-mode and the TDI channels mix the two polarizations differently, "
     "so the degeneracy it straightens may not be the degeneracy LISA has."),
    (r"^FUNC:_reparam_A_of_incl$", "PHYSICS", "Implementation of --internal-reparam-dl-incl."),
    (r"^CONST:_REPARAM_", "PHYSICS", "Tuning constants for --internal-reparam-dl-incl."),

    # ---------------------------------------------------------------------- extrinsic boxes
    (r"^OPTION:--limit-(psi|inclination)$", "PORT",
     "Zoom-box limits on psi and inclination. These parameters mean the same thing in "
     "both drivers and LISA exposes --inclination-cosine-sampler, which is exactly the "
     "case junior PR #58 found silently ignored -- so port the POST-#58 form, including "
     "the cos(iota) endpoint swap."),
    (r"^OPTION:--limit-(right-ascension|declination)$", "PHYSICS",
     "QUESTION: what should a sky zoom box mean for LISA? The LISA driver reuses the "
     "KEY NAMES right_ascension/declination for its sampled sky pair, but the values "
     "are ecliptic (lambda,beta) and may be further rotated by "
     "--internal-sky-network-coordinates. A box is therefore well-defined only once it "
     "is stated which frame the user is quoting -- and LISA already has "
     "--ecliptic-latitude/--ecliptic-longitude/--lisa-fixed-sky, which may already be "
     "the intended mechanism."),

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
    (r"^FUNC:_normalize_interpolate_time_argv$", "PORT",
     "Normalizes --interpolate-time argv forms. LISA exposes --interpolate-time, so "
     "the same normalization applies."),
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
