#!/usr/bin/env python
"""Unit tests for RIFT.likelihood.q_time_pregrid.

Companion to test_q_time_pregrid_pipeline.py, which covers the pipeline WIRING (pseudo_pipe /
helper_LDG_Events.py -> args_ile.txt).  This file covers the library alone: the choice set,
the driver-mirroring prerequisite check (--vectorized required; --rotation-slow /
--freqresponse / --calibration-envelope-directory excluded; factor 8 forces cubic and
refuses a conflicting explicit stencil), and the two-stage refuse discipline shared with
RIFT.likelihood.time_marginalization_quadrature.
"""
import pytest

from RIFT.likelihood.q_time_pregrid import (
    Q_TIME_PREGRID_CHOICES, STENCIL_CONFLICT_MESSAGE,
    validate_q_time_pregrid_factor, q_time_pregrid_pipeline_prereqs,
    refuse_unhonourable_q_time_pregrid, refuse_unless_q_time_pregrid_emitted,
    find_q_time_pregrid_in_ile_args, find_interpolate_time_in_ile_args)

GOOD_ILE_ARGS = "integrate_likelihood_extrinsic_batchmode --vectorized --gpu --srate 4096"


# --------------------------------------------------------------------------- validate

def test_choices_are_1_and_8():
    assert Q_TIME_PREGRID_CHOICES == (1, 8)


@pytest.mark.parametrize("value,expected", [(1, 1), ("1", 1), (8, 8), ("8", 8)])
def test_validate_accepts_legal_values(value, expected):
    assert validate_q_time_pregrid_factor(value) == expected


@pytest.mark.parametrize("value", [0, 2, 4, 16, "bandlimited", None, "eight"])
def test_validate_rejects_illegal_values(value):
    with pytest.raises(ValueError):
        validate_q_time_pregrid_factor(value)


# --------------------------------------------------------------------- prerequisites

def test_factor_1_is_never_refused():
    """The default must never be able to fail a workflow build, even in a configuration
    that excludes factor 8 entirely -- factor 1 is what ILE does anyway."""
    assert q_time_pregrid_pipeline_prereqs(1, "--rotation-slow --freqresponse") == []
    assert q_time_pregrid_pipeline_prereqs(1, "") == []


def test_honourable_configuration_passes():
    assert q_time_pregrid_pipeline_prereqs(8, GOOD_ILE_ARGS) == []
    # optparse abbreviation: shortest unique spelling.
    assert q_time_pregrid_pipeline_prereqs(8, "X --vec") == []


def test_required_flag_is_reported_when_missing():
    missing = q_time_pregrid_pipeline_prereqs(8, "X --gpu --srate 4096")
    assert any("--vectorized" in m for m in missing), missing


@pytest.mark.parametrize("flag,value", [
    ("--rotation-slow", ""),
    ("--freqresponse", ""),
    ("--calibration-envelope-directory", " /tmp/cal"),
])
def test_each_excluding_flag_is_reported_when_present(flag, value):
    missing = q_time_pregrid_pipeline_prereqs(8, GOOD_ILE_ARGS + " " + flag + value)
    assert any(flag in m for m in missing), missing


@pytest.mark.parametrize("innocent", [
    "--rotation-slow-foo", "--calibration-n-realizations 100", "--freqresponse-scale 2",
])
def test_exclusions_do_not_fire_on_lookalike_flags(innocent):
    assert q_time_pregrid_pipeline_prereqs(8, GOOD_ILE_ARGS + " " + innocent) == []


# ------------------------------------------------------- the forced-cubic-stencil conflict

def test_no_explicit_stencil_is_fine():
    """Factor 8 silently forces cubic when the caller never named a stencil -- exactly
    the driver's own `opts._interp_time_from_default` branch."""
    assert q_time_pregrid_pipeline_prereqs(8, GOOD_ILE_ARGS) == []


def test_explicit_cubic_is_fine():
    assert q_time_pregrid_pipeline_prereqs(8, GOOD_ILE_ARGS + " --interpolate-time cubic") == []


@pytest.mark.parametrize("stencil", ["nearest", "sinc"])
def test_explicit_conflicting_stencil_is_refused_with_the_drivers_own_wording(stencil):
    missing = q_time_pregrid_pipeline_prereqs(
        8, GOOD_ILE_ARGS + " --interpolate-time " + stencil)
    assert STENCIL_CONFLICT_MESSAGE in missing


def test_legacy_boolean_stencil_spellings_are_understood():
    """The driver accepts legacy truthy/falsy --interpolate-time spellings too (truthy meant
    'cubic', falsy meant 'nearest').  A falsy hand-passed value must still conflict."""
    assert q_time_pregrid_pipeline_prereqs(8, GOOD_ILE_ARGS + " --interpolate-time true") == []
    missing = q_time_pregrid_pipeline_prereqs(8, GOOD_ILE_ARGS + " --interpolate-time false")
    assert STENCIL_CONFLICT_MESSAGE in missing


def test_optparse_takes_the_last_stencil_occurrence():
    args = GOOD_ILE_ARGS + " --interpolate-time nearest --interpolate-time cubic"
    assert q_time_pregrid_pipeline_prereqs(8, args) == []
    args = GOOD_ILE_ARGS + " --interpolate-time cubic --interpolate-time sinc"
    missing = q_time_pregrid_pipeline_prereqs(8, args)
    assert STENCIL_CONFLICT_MESSAGE in missing


# ---------------------------------------------------- refuse_unhonourable_q_time_pregrid

def test_refusal_actually_raises():
    """Executable coverage of the raise itself, kept separately from the prereqs check so a
    guard turned into a print is a code change a test can see."""
    with pytest.raises(ValueError):
        refuse_unhonourable_q_time_pregrid(8, "X --gpu", "somewhere")
    refuse_unhonourable_q_time_pregrid(8, GOOD_ILE_ARGS, "somewhere")
    refuse_unhonourable_q_time_pregrid(1, "X --rotation-slow", "somewhere")


# --------------------------------------------------------- the guard reads the BYTES

def test_prereq_check_alone_approves_args_that_never_got_the_flag():
    """Documents WHY refuse_unless_q_time_pregrid_emitted exists: the prerequisite check
    reads prerequisites from the argument string but the INTENT from the caller, so on its
    own it approves an args_ile.txt that never received the flag."""
    assert q_time_pregrid_pipeline_prereqs(8, GOOD_ILE_ARGS) == []


def test_emission_guard_refuses_args_that_never_got_the_flag():
    with pytest.raises(ValueError) as e:
        refuse_unless_q_time_pregrid_emitted(8, GOOD_ILE_ARGS, "args_ile.txt")
    assert "contains no --q-time-pregrid-factor" in str(e.value)


def test_emission_guard_refuses_a_duplicate_because_optparse_takes_the_last():
    args = GOOD_ILE_ARGS + " --q-time-pregrid-factor 8 --q-time-pregrid-factor 1"
    with pytest.raises(ValueError) as e:
        refuse_unless_q_time_pregrid_emitted(8, args, "args_ile.txt")
    assert "occurrences" in str(e.value)


def test_emission_guard_refuses_a_value_that_does_not_match_the_request():
    args = GOOD_ILE_ARGS + " --q-time-pregrid-factor 1"
    with pytest.raises(ValueError):
        refuse_unless_q_time_pregrid_emitted(8, args, "args_ile.txt")


def test_emission_guard_holds_a_hand_passed_factor_to_the_same_standard():
    """The manual route (--manual-extra-ile-args / an ini) got no protection at all while the
    guard was keyed on the pipeline option being set."""
    args = "X --gpu --rotation-slow --q-time-pregrid-factor 8"
    with pytest.raises(ValueError) as e:
        refuse_unless_q_time_pregrid_emitted(None, args, "args_ile.txt")
    assert "--rotation-slow" in str(e.value)


def test_emission_guard_is_silent_on_the_default_path():
    refuse_unless_q_time_pregrid_emitted(None, GOOD_ILE_ARGS, "args_ile.txt")


def test_emission_guard_is_silent_when_factor_is_explicitly_one():
    refuse_unless_q_time_pregrid_emitted(1, GOOD_ILE_ARGS, "args_ile.txt")


def test_emission_guard_accepts_the_honoured_case():
    refuse_unless_q_time_pregrid_emitted(
        8, GOOD_ILE_ARGS + " --q-time-pregrid-factor 8", "args_ile.txt")


def test_emission_guard_catches_a_hand_passed_stencil_conflict():
    args = GOOD_ILE_ARGS + " --interpolate-time nearest --q-time-pregrid-factor 8"
    with pytest.raises(ValueError) as e:
        refuse_unless_q_time_pregrid_emitted(8, args, "args_ile.txt")
    assert STENCIL_CONFLICT_MESSAGE in str(e.value)


# ------------------------------------------------------------------------- find_* helpers

def test_find_q_time_pregrid_handles_the_equals_form():
    assert find_q_time_pregrid_in_ile_args(
        "X --q-time-pregrid-factor=8") == ['8']
    assert find_q_time_pregrid_in_ile_args(
        "X --q-time-pregrid-f=8") == ['8']
    assert find_q_time_pregrid_in_ile_args("X --vectorized --gpu") == []


def test_find_interpolate_time_handles_the_equals_form():
    assert find_interpolate_time_in_ile_args(
        "X --interpolate-time=cubic") == ['cubic']
    assert find_interpolate_time_in_ile_args(
        "X --interpolate-t=cubic") == ['cubic']
    assert find_interpolate_time_in_ile_args("X --vectorized --gpu") == []
