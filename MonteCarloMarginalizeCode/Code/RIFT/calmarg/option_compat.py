"""Option-compatibility gate for in-loop calibration marginalization (ILE batchmode).

WHY THIS EXISTS.  ``integrate_likelihood_extrinsic_batchmode`` turns calibration
marginalization on from ONE option -- ``--calibration-envelope-directory`` -- but the
``n_cal>1`` reduction it enables lives at exactly one family of call sites: the
``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`` calls inside the time-marginalized,
vectorized, xpy (``--gpu`` / ``--force-xpy``) branch.  Every other dispatch in that driver
reaches a likelihood function that takes no ``n_cal`` argument at all.  The calibration
realizations are still drawn, still precomputed, and then never enter the likelihood --
so the run costs more, prints ``calibration`` in its banner and its submit file, and
reports the ZERO-CALIBRATION answer.  Nothing raises.

That is the failure mode this module refuses.  It is the same discipline the sub-sample
stencil gate and the ``--time-marginalization-quadrature`` gate already apply in the same
driver, and for the same reason: an option that silently does nothing is worse than one
that is unavailable, because a campaign can be run against it and believed.

TWO KINDS OF REFUSAL, deliberately distinguished (they mean different things to whoever
reads the message):

``KIND_INERT``
    The requested option cannot take effect in this configuration.  The request is
    self-contradictory: what was asked for is unreachable from the flags given.  The
    remedy is always local -- add the missing flag, or drop the one that does nothing.

``KIND_UNIMPLEMENTED``
    The combination is not implemented, and running it would silently evaluate a
    DIFFERENT likelihood than the one requested.  The remedy is to drop one side.  These
    refusals additionally carry ``enable_requires``: what would have to be built and
    validated before the combination could be allowed.  R. O'Shaughnessy's decision
    (2026-09) is that it is fine for these paths not to be implemented, BECAUSE
    calibration marginalization is not tested against the third-generation machinery at
    all -- so a silent degradation there produces plausible numbers from a path nothing
    has ever checked.  That is a scope boundary, not an oversight; ``enable_requires``
    is where the boundary is written down.

A THIRD KIND -- runnable, accepted, but unvalidated -- IS NOT REPRESENTED HERE, and that
is a finding rather than an omission.  Tracing every calibration x 3G combination in the
driver, each one fails the STRONGER test first: the dispatch does not exist, so there is
nothing to leave untested.  See DESIGN_calmarg_in_loop.md ("Option-compatibility gate").

This module only REFUSES.  It changes no accepted configuration's arithmetic.
"""
from __future__ import print_function

import collections

__all__ = ["KIND_INERT", "KIND_UNIMPLEMENTED", "CAL_OPT_IN_FLAGS", "Refusal",
           "calibration_refusals", "calibration_notices", "refusals_from_opts",
           "notices_from_opts", "refuse_incompatible_calibration_options"]

KIND_INERT = "cannot take effect"
KIND_UNIMPLEMENTED = "not implemented"

Refusal = collections.namedtuple("Refusal", "kind options message enable_requires")


# Calibration opt-ins whose ONLY effect is inside in-loop calibration marginalization,
# i.e. inside the `if opts.calibration_envelope_directory:` block or gated on the module
# flag it sets.  Restricted on purpose to options whose default is False/None: an option
# with a numeric default (--calibration-n-realizations, --calibration-spline-count,
# --calibration-pilot-extrinsic, --calibration-mc-error-extrinsic,
# --calibration-neff-cal-target, --calibration-n-realizations-max) is ALWAYS "set", so a
# rule over those would refuse every run that merely left the defaults in place.
#
# --calibration-burn-in-nmax is listed here like the rest (it too does nothing without the
# envelope), but it needs a SECOND rule below: the envelope alone does not make it live,
# because the driver reads it only under --calibration-burn-in-neff.
#
# (cli_flag, opts_attribute).
CAL_OPT_IN_FLAGS = (
    ("--calibration-fused-kernel", "calibration_fused_kernel"),
    ("--calibration-conjugate-phase", "calibration_conjugate_phase"),
    ("--calibration-global-norm", "calibration_global_norm"),
    ("--calibration-proposal-breadcrumb", "calibration_proposal_breadcrumb"),
    ("--calibration-dump-responsibilities", "calibration_dump_responsibilities"),
    ("--calibration-export-posterior", "calibration_export_posterior"),
    ("--calibration-burn-in-neff", "calibration_burn_in_neff"),
    ("--calibration-burn-in-nmax", "calibration_burn_in_nmax"),
)

_ENVELOPE = "--calibration-envelope-directory"

# What would have to exist before calibration marginalization could be allowed on a
# third-generation likelihood path.  ONE definition, interpolated into both 3G refusals,
# so the two messages cannot drift apart.
_3G_ENABLE_REQUIRES = (
    "calibration marginalization has NO test coverage against the third-generation "
    "machinery -- not a weak test, none -- so this is a scope boundary and not an "
    "oversight.  Enabling it needs, at minimum: (a) {precompute} to return the "
    "per-realization calibration cross terms that PrecomputeLikelihoodTerms returns "
    "for the baseline likelihood; (b) an n_cal>1 reduction in {noloop}, with the same "
    "per-realization self-term <C_c h|C_c h> the baseline reduction uses; and (c) a "
    "brute-force agreement check of that reduction, of the kind "
    "RIFT/calmarg/test_selfterm_reduction.py already applies to the baseline "
    "likelihood, wired into the calmarg-check CI gate."
)


def _inert(options, message):
    return Refusal(KIND_INERT, tuple(options), message, None)


def _unimplemented(options, message, enable_requires):
    return Refusal(KIND_UNIMPLEMENTED, tuple(options), message, enable_requires)


def calibration_refusals(calibration_envelope_directory=None,
                         opt_in_flags=(),
                         time_marginalization=False,
                         vectorized=False,
                         xpy_evaluator=False,
                         rotation_slow=False,
                         freqresponse=False,
                         dump_responsibilities=False,
                         burn_in_neff=None,
                         burn_in_nmax=None,
                         n_realizations=None,
                         fused_kernel=False):
    """Return the list of Refusals for one resolved ILE configuration (possibly empty).

    Pure: booleans in, Refusals out.  No option namespace, no I/O, no raising.

    Parameters
    ----------
    calibration_envelope_directory : str or None
        The value of --calibration-envelope-directory.  This ALONE decides whether
        in-loop calibration marginalization is active in the driver.
    opt_in_flags : sequence of str
        CLI spellings of the calibration opt-ins the user actually set (see
        CAL_OPT_IN_FLAGS).  Order is preserved in the output.
    time_marginalization, vectorized, rotation_slow, freqresponse : bool
        The corresponding CLI booleans.
    xpy_evaluator : bool
        ``opts.gpu`` AFTER the driver has resolved it, i.e. a real CUDA device OR
        --force-xpy.  It must be the resolved value: `--gpu` is silently downgraded to
        False when cupy is unavailable, and the downgraded configuration is precisely
        one of the ones that drops calibration on the floor.
    dump_responsibilities : bool
        Whether --calibration-dump-responsibilities is set.  The cal PILOT is a
        legitimate diagnostic mode with DIFFERENT prerequisites: it runs inside the
        `if opts.vectorized:` precompute block, uses cal_method='loop' on whatever xpy
        is available, and returns before the driver ever selects a production
        likelihood_function -- so it needs neither --time-marginalization nor an xpy
        evaluator.  Refusing it on those grounds would break the shipped adaptive
        pipeline (util_CalPilotStage.py).
    burn_in_neff, burn_in_nmax : float/int or None
        --calibration-burn-in-neff and --calibration-burn-in-nmax.  The cap is a
        DEPENDENT option: the driver reads it only inside `if
        opts.calibration_burn_in_neff:`, so on its own it is silently ignored.
    n_realizations : int or None
        --calibration-n-realizations.  NOT an opt-in (it has a non-None default, so a rule
        over its mere presence would refuse every run), but it is a PREREQUISITE for the
        two options whose consumers sit behind `n_cal > 1`.  None means "not supplied";
        the caller passes the resolved value.
    """
    out = []

    if not calibration_envelope_directory:
        # Calibration marginalization is OFF.  Every calibration opt-in is then a silent
        # no-op: `calibration_marginalization` is set by the envelope directory and by
        # nothing else, and each of these flags is read only under it.  Nothing else in
        # this function applies -- --rotation-slow and --freqresponse are perfectly fine
        # on their own, and must stay fine.
        for flag in opt_in_flags:
            out.append(_inert(
                (flag, _ENVELOPE),
                "%s was requested without %s, so in-loop calibration marginalization is "
                "never switched on and %s is read by nothing: it is silently ignored.  "
                "Add %s (the per-IFO <IFO>.txt envelope directory is what activates "
                "calibration marginalization), or drop %s."
                % (flag, _ENVELOPE, flag, _ENVELOPE, flag)))
        return out

    if rotation_slow:
        out.append(_unimplemented(
            (_ENVELOPE, "--rotation-slow"),
            "%s (in-loop calibration marginalization) with --rotation-slow is not "
            "implemented.  --rotation-slow dispatches to "
            "DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation, which takes "
            "no n_cal/cal_method argument, and PrecomputeLikelihoodTermsWithRotation "
            "builds no calibration cross terms.  The realizations would be drawn and "
            "paid for, then never entered: the run would advertise a "
            "calibration-marginalized analysis and report the zero-calibration one.  "
            "Drop --rotation-slow to marginalize over calibration on the baseline "
            "likelihood, or drop %s to run the slow-rotation likelihood."
            % (_ENVELOPE, _ENVELOPE),
            _3G_ENABLE_REQUIRES.format(
                precompute="PrecomputeLikelihoodTermsWithRotation",
                noloop="DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation")))

    if freqresponse:
        out.append(_unimplemented(
            (_ENVELOPE, "--freqresponse"),
            "%s (in-loop calibration marginalization) with --freqresponse is not "
            "implemented.  --freqresponse dispatches to "
            "DiscreteFactoredLogLikelihoodFreqResponseNoLoop, which takes no "
            "n_cal/cal_method argument, and PrecomputeLikelihoodTermsFreqResponse "
            "builds no calibration cross terms.  The realizations would be drawn and "
            "paid for, then never entered: the run would advertise a "
            "calibration-marginalized analysis and report the zero-calibration one.  "
            "Drop --freqresponse to marginalize over calibration on the baseline "
            "likelihood, or drop %s to run the finite-size-response likelihood."
            % (_ENVELOPE, _ENVELOPE),
            _3G_ENABLE_REQUIRES.format(
                precompute="PrecomputeLikelihoodTermsFreqResponse",
                noloop="DiscreteFactoredLogLikelihoodFreqResponseNoLoop")))

    if burn_in_nmax and not burn_in_neff:
        # A DEPENDENT opt-in: unlike the others, the envelope directory is not enough to
        # make it live.  The driver reads opts.calibration_burn_in_nmax only inside
        # `if opts.calibration_burn_in_neff ...`, which is the option that switches the
        # zero-cal burn-in on; with no burn-in there is nothing for the cap to cap.
        out.append(_inert(
            ("--calibration-burn-in-nmax", "--calibration-burn-in-neff"),
            "--calibration-burn-in-nmax caps the zero-cal burn-in, but the burn-in is "
            "switched on by --calibration-burn-in-neff and by nothing else -- the driver "
            "reads the cap only inside that option's branch.  Set on its own the cap is "
            "silently ignored, and the run goes straight to the full cal-marginalized "
            "integration at the production sample budget, which is exactly the number of "
            "samples the cap was meant to hold down.  Add --calibration-burn-in-neff "
            "(the burn-in target), or drop --calibration-burn-in-nmax."))

    # ---- the n_cal > 1 prerequisites -------------------------------------------------
    # Two opt-ins have consumers guarded by n_cal > 1, so ONE realization makes them inert
    # even on an otherwise perfect configuration.  Traced to the source, not inferred from
    # the flag names:
    #   --calibration-fused-kernel  factored_likelihood's `if n_cal == 1:` branch RETURNS
    #                               before `cal_method == 'fused'` is ever read, so the
    #                               fused reduction is unreachable and the run silently
    #                               uses the ordinary one.
    #   --calibration-burn-in-neff  the driver's burn-in block is
    #                               `if opts.calibration_burn_in_neff and
    #                                calibration_marginalization and
    #                                n_cal_for_likelihood > 1:` -- with one realization the
    #                               zero-cal burn-in never runs.
    # NOT listed, deliberately: --calibration-conjugate-phase and --calibration-global-norm
    # reach the likelihood through extra_kwargs on the PRECOMPUTE and are honoured at
    # n_cal == 1 too (that branch uses rho_sq_cal[0]), so refusing them here would be a
    # false positive.
    if n_realizations is not None and int(n_realizations) <= 1:
        if fused_kernel:
            out.append(_inert(
                ("--calibration-fused-kernel", "--calibration-n-realizations"),
                "--calibration-fused-kernel with --calibration-n-realizations %d: the "
                "fused reduction is selected by cal_method='fused', which "
                "factored_likelihood reads only AFTER its `n_cal == 1` branch has already "
                "returned.  With one realization the flag is therefore silently ignored "
                "and the ordinary reduction runs -- the answer is right, but the run "
                "advertises a kernel it did not use.  Raise "
                "--calibration-n-realizations above 1, or drop the flag."
                % (int(n_realizations),)))
        if burn_in_neff:
            out.append(_inert(
                ("--calibration-burn-in-neff", "--calibration-n-realizations"),
                "--calibration-burn-in-neff with --calibration-n-realizations %d: the "
                "zero-cal burn-in runs only under `n_cal_for_likelihood > 1`, so with one "
                "realization there is no burn-in and the target is silently ignored.  "
                "Raise --calibration-n-realizations above 1, or drop the flag."
                % (int(n_realizations),)))

    if not vectorized:
        out.append(_inert(
            (_ENVELOPE, "--vectorized"),
            "%s requires --vectorized.  Without it the driver never builds the packed "
            "rholm/cross-term arrays the calibration reduction indexes, and the "
            "likelihood it calls is the scalar FactoredLogLikelihoodTimeMarginalized "
            "(or FactoredLogLikelihood without --time-marginalization), neither of "
            "which takes n_cal -- so the calibration realizations are drawn and then "
            "ignored.  Add --vectorized, or drop %s." % (_ENVELOPE, _ENVELOPE)))

    if not dump_responsibilities:
        # The cal PILOT (--calibration-dump-responsibilities) is exempt from both of
        # these: it returns 0.0 from inside the precompute block, before the driver
        # picks a production likelihood_function at all.  Its own prerequisite
        # (--vectorized) is checked above and is NOT exempted.
        if not time_marginalization:
            out.append(_inert(
                (_ENVELOPE, "--time-marginalization"),
                "%s requires --time-marginalization.  Without it the driver takes the "
                "`if not opts.time_marginalization` branch and calls "
                "FactoredLogLikelihood, which takes no n_cal, so the calibration "
                "realizations are drawn and then ignored.  Add --time-marginalization, "
                "or drop %s.  (The one configuration that legitimately runs "
                "calibration without it is the pilot, "
                "--calibration-dump-responsibilities, which returns before this branch "
                "is reached.)" % (_ENVELOPE, _ENVELOPE)))
        if not xpy_evaluator:
            out.append(_inert(
                (_ENVELOPE, "--gpu"),
                "%s requires the maintained NoLoop evaluator, selected by --gpu.  "
                "Plain --vectorized without it calls "
                "DiscreteFactoredLogLikelihoodViaArrayVector, which takes no n_cal, so "
                "the calibration realizations are drawn and then ignored.  NOTE that "
                "--gpu is SILENTLY DOWNGRADED when cupy is unavailable, so `--gpu` on a "
                "host with no CUDA device lands here: add --force-xpy, which keeps the "
                "identical NoLoop code path on numpy.  Otherwise drop %s.  (The pilot, "
                "--calibration-dump-responsibilities, is exempt: it evaluates on "
                "whatever xpy is present and returns before a production likelihood is "
                "selected.)" % (_ENVELOPE, _ENVELOPE)))

    return out


def calibration_notices(calibration_envelope_directory=None,
                        fused_kernel=False,
                        dump_responsibilities=False):
    """Non-fatal notices: accepted configurations where an option has no effect BY DESIGN.

    These are deliberately NOT refusals.  Each is emitted by the shipped pipeline on a
    stage where it is a documented no-op, so refusing them would break production:

      * util_CalPilotStage.py inherits the WIDE args_ile.txt (which may carry
        --calibration-fused-kernel) and appends --calibration-dump-responsibilities.
        The pilot deliberately uses cal_method='loop'.
      * util_RIFT_pseudo_pipe.py emits --calibration-export-posterior on the wide stage
        too, where it is documented as harmless (it only fires at the fairdraw stage).

    A notice says so out loud instead of leaving the reader to infer it from a banner.
    """
    out = []
    if calibration_envelope_directory and dump_responsibilities and fused_kernel:
        out.append(
            "[calmarg] --calibration-dump-responsibilities: this is the cal PILOT.  It "
            "uses the loop reduction (cal_method='loop') and returns before production "
            "integration, so --calibration-fused-kernel has no effect on this run.  "
            "Accepted, not refused: util_CalPilotStage.py inherits the wide ILE "
            "arguments verbatim, so the pilot legitimately carries this flag.")
    return out


def _opt_ins_set_on(opts):
    return tuple(flag for flag, attr in CAL_OPT_IN_FLAGS if getattr(opts, attr, None))


def refusals_from_opts(opts):
    """Adapter: read a driver option namespace and return calibration_refusals(...).

    THE SEAM.  The predicate above can be perfectly correct and still be wired to the
    wrong attribute; that is exactly how the two `getattr(opts,
    'calibration_marginalization', False)` guards this gate replaces came to be inert
    (there is no such option, so both always evaluated False).  So this adapter is
    covered by subprocess tests through the real CLI, not only by unit calls.

    ``opts.gpu`` MUST already be resolved (see calibration_refusals).
    """
    return calibration_refusals(
        calibration_envelope_directory=getattr(opts, "calibration_envelope_directory", None),
        opt_in_flags=_opt_ins_set_on(opts),
        time_marginalization=bool(getattr(opts, "time_marginalization", False)),
        vectorized=bool(getattr(opts, "vectorized", False)),
        xpy_evaluator=bool(getattr(opts, "gpu", False)),
        rotation_slow=bool(getattr(opts, "rotation_slow", False)),
        freqresponse=bool(getattr(opts, "freqresponse", False)),
        dump_responsibilities=bool(getattr(opts, "calibration_dump_responsibilities", None)),
        burn_in_neff=getattr(opts, "calibration_burn_in_neff", None),
        burn_in_nmax=getattr(opts, "calibration_burn_in_nmax", None),
        n_realizations=getattr(opts, "calibration_n_realizations", None),
        fused_kernel=bool(getattr(opts, "calibration_fused_kernel", False)),
    )


def notices_from_opts(opts):
    return calibration_notices(
        calibration_envelope_directory=getattr(opts, "calibration_envelope_directory", None),
        fused_kernel=bool(getattr(opts, "calibration_fused_kernel", False)),
        dump_responsibilities=bool(getattr(opts, "calibration_dump_responsibilities", None)),
    )


def format_refusals(refusals):
    """One message for the whole configuration, so a user fixes it in one pass."""
    lines = ["Refusing this calibration-marginalization configuration rather than "
             "silently degrading it to the zero-calibration likelihood:"]
    for i, r in enumerate(refusals, 1):
        lines.append("  (%d) [%s] %s" % (i, r.kind, r.message))
        if r.enable_requires:
            lines.append("      To enable it: %s" % r.enable_requires)
    return "\n".join(lines)


def refuse_incompatible_calibration_options(opts, printer=print):
    """Raise ValueError if this configuration cannot honour its calibration options.

    Emits the non-fatal notices first, so an accepted-but-inert-by-design flag is still
    visible in the log.  Returns the notice list (for tests).
    """
    notices = notices_from_opts(opts)
    for n in notices:
        printer(n)
    refusals = refusals_from_opts(opts)
    if refusals:
        raise ValueError(format_refusals(refusals))
    return notices
