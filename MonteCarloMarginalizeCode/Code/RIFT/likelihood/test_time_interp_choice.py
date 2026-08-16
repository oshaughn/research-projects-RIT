#!/usr/bin/env python3
"""test_time_interp_choice -- the pipeline's Q_lm stencil handling.

Automatic selection was REMOVED after measurement (see time_interp_choice's docstring for the
table). What is left to guard:

  1. The retired "choose for me" spellings raise, with a pointer to the guidance -- they must not
     silently resolve to some default, because a run whose stencil was picked by a rule that no
     longer exists should not start.
  2. An explicit stencil name is validated while the workflow is BUILT, not once per job after
     submission.
  3. 'off' spellings disable rather than raise: the flag takes a value, so '...=False' arrives as
     the truthy STRING 'False'.
  4. The stencil name list agrees with factored_likelihood's, since this leaf module duplicates it
     to stay import-cheap.

Self-contained: numpy only, runs instantly.

    python3 test_time_interp_choice.py
"""
from __future__ import print_function

from RIFT.likelihood.time_interp_choice import (
    BARE_FLAG_SENTINEL,
    OFF_REQUEST_TOKENS,
    RETIRED_AUTO_TOKENS,
    TIME_INTERP_CHOICES,
    is_off_request,
    is_retired_auto_request,
    resolve_interpolate_time_request,
    validate_stencil_name,
)


def test_retired_auto_spellings_raise_with_guidance():
    """'True' used to mean "choose the stencil for me".  That rule was measured to mis-select at
    2 of 8 total masses, and the correct stencil additionally depends on fmin -- at M = 5 Msun the
    winner flips between fmin 30 and 150 with srate, fmax and mass identical, which no
    (srate, fmax, mass) rule can represent.

    So these must RAISE rather than resolve to a default.  Silently substituting one would
    reintroduce exactly the failure the removal exists to prevent, and the message has to say
    what to do instead.
    """
    for v in RETIRED_AUTO_TOKENS + ('True', 'AUTO', ' true '):
        assert is_retired_auto_request(v), "%r must be recognised as a retired auto request" % v
        try:
            validate_stencil_name(v)
        except ValueError as e:
            msg = str(e)
            assert 'REMOVED' in msg, "the error must say the feature was removed: %r" % msg
            assert 'cubic' in msg and 'sinc' in msg, \
                "the error must name the alternatives: %r" % msg
            continue
        raise AssertionError("validate_stencil_name(%r) must raise" % v)
    print("retired auto spellings raise with guidance: OK")


def test_explicit_stencil_names_are_validated():
    """A typo must fail while the workflow is BUILT, not once per job after submission."""
    for good in TIME_INTERP_CHOICES + (' SINC ', 'Cubic', 'NEAREST'):
        assert validate_stencil_name(good) in TIME_INTERP_CHOICES
    for bad in ('sinK', 'lanczos', 'Sinc8', '', 'nearest,cubic', 'linear'):
        try:
            validate_stencil_name(bad)
        except ValueError:
            continue
        raise AssertionError("validate_stencil_name(%r) must raise" % bad)
    print("explicit stencil names validated, typos rejected: OK")


def test_off_spellings_disable_rather_than_raise():
    """'--internal-ile-interpolate-time False' must mean OFF, not "unknown stencil".

    The flag takes a value, so 'False' arrives as the STRING 'False' -- truthy in Python.  Without
    an explicit off-check it sails past the pipeline's `if opts...:` guard and is then rejected as
    a bad stencil name, i.e. the most natural way to spell "turn this off" becomes a hard error.
    Every value must fall into exactly one of off / retired-auto / stencil-name / invalid.
    """
    for v in OFF_REQUEST_TOKENS + ('False', 'FALSE', ' off '):
        assert is_off_request(v), "%r must mean disabled" % v
        assert not is_retired_auto_request(v), "%r must not also be a retired auto request" % v
    for v in RETIRED_AUTO_TOKENS:
        assert not is_off_request(v), "%r must not mean disabled" % v
    for v in TIME_INTERP_CHOICES:
        assert not is_off_request(v) and not is_retired_auto_request(v), \
            "%r is a stencil name, neither off nor auto" % v
    print("off / retired-auto / stencil-name are disjoint: OK")


def test_bare_flag_is_rejected_not_silently_ignored():
    """A BARE '--internal-ile-interpolate-time' must not be indistinguishable from omitting it.

    argparse's nargs='?' stores `const` for a bare flag.  With const=None a bare flag looks exactly
    like an absent flag, so the pipeline's truthiness guard skipped the block and emitted no
    --interpolate-time at all -- silently turning the feature OFF for anyone using the old
    store_true spelling, while the help text claimed an explicit stencil was required.  Both
    entry points now store BARE_FLAG_SENTINEL, which must raise with actionable text.
    """
    assert BARE_FLAG_SENTINEL is not None and BARE_FLAG_SENTINEL != '', \
        "the bare-flag sentinel must be distinguishable from an absent flag"
    assert resolve_interpolate_time_request(None) is None, "absent flag means disabled"
    try:
        resolve_interpolate_time_request(BARE_FLAG_SENTINEL)
    except ValueError as e:
        msg = str(e)
        assert 'no value' in msg and 'nearest|cubic|sinc' in msg, \
            "the bare-flag error must say what to do instead: %r" % msg
    else:
        raise AssertionError("a bare flag must raise, not resolve")
    print("bare flag raises rather than silently disabling: OK")


def test_resolver_covers_every_flag_spelling():
    """off / bare / retired-auto / stencil / typo -- one resolver, exhaustive."""
    assert resolve_interpolate_time_request(None) is None
    for off in OFF_REQUEST_TOKENS + ('False', ' OFF '):
        assert resolve_interpolate_time_request(off) is None, off
    for good in TIME_INTERP_CHOICES + (' SINC ', 'Cubic'):
        assert resolve_interpolate_time_request(good) in TIME_INTERP_CHOICES, good
    for bad in (BARE_FLAG_SENTINEL,) + RETIRED_AUTO_TOKENS + ('sinK', 'lanczos', ''):
        try:
            resolve_interpolate_time_request(bad)
        except ValueError:
            continue
        raise AssertionError("resolve_interpolate_time_request(%r) must raise" % bad)
    print("resolver covers off / bare / retired-auto / stencil / typo: OK")


def test_choices_agree_with_the_likelihood_module():
    """This leaf module duplicates TIME_INTERP_CHOICES to stay import-cheap; keep them in step."""
    from RIFT.likelihood.factored_likelihood import TIME_INTERP_CHOICES as FL_CHOICES
    assert tuple(TIME_INTERP_CHOICES) == tuple(FL_CHOICES), (
        "time_interp_choice.TIME_INTERP_CHOICES %r disagrees with factored_likelihood's %r"
        % (TIME_INTERP_CHOICES, FL_CHOICES))
    print("stencil name lists agree with factored_likelihood: OK")


def test_no_automatic_selection_api_survives():
    """Nothing may reintroduce an automatic selector without also updating this file.

    The removal is a measured conclusion, not a simplification: if a future change adds a chooser
    back, it must land with new measurements, and this guard has to be revisited deliberately
    rather than silently satisfied.
    """
    import RIFT.likelihood.time_interp_choice as tic
    for gone in ('choose_time_interp_stencil', 'interp_time_threshold',
                 'INTERP_TIME_OVERSAMPLING_THRESHOLD',
                 'INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU',
                 'INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU'):
        assert not hasattr(tic, gone), (
            "%s is back. Automatic selection was removed because a (srate, fmax, mass) rule "
            "cannot be correct -- the answer depends on fmin too. If you are reintroducing "
            "selection, do it with a real bandwidth estimator and new measurements, and rewrite "
            "this test on purpose." % gone)
    print("no automatic-selection API present: OK")


if __name__ == "__main__":
    test_retired_auto_spellings_raise_with_guidance()
    test_explicit_stencil_names_are_validated()
    test_off_spellings_disable_rather_than_raise()
    test_bare_flag_is_rejected_not_silently_ignored()
    test_resolver_covers_every_flag_spelling()
    test_choices_agree_with_the_likelihood_module()
    test_no_automatic_selection_api_survives()
    print("\nPASS")
