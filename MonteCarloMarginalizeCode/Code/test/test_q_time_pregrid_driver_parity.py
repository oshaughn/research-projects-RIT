"""Builder/driver parity for --q-time-pregrid-factor (PR #281 review, MAJOR #2).

RIFT/likelihood/q_time_pregrid.py's docstring claims the pipeline-side prerequisite check
and bin/integrate_likelihood_extrinsic_batchmode's own first-job guard "cannot silently
drift apart," but neither this module's test file nor test_q_time_pregrid_pipeline.py ever
ran the driver: the claim was checked by hand once during review, not by CI.  This file
executes the real driver as a subprocess -- no data files needed, since the guard runs
right after option parsing, before any PSD/frame load -- for every prerequisite the builder
checks, and asserts the two sides agree on refuse-or-not.  Modelled on
test_psi_marginalization.py's `_REFUSAL_CASES` / `test_refuses_incompatible_combinations`,
which uses the same "minimal --event-time invocation reaches the refusal" technique.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from RIFT.likelihood.q_time_pregrid import (
    STENCIL_CONFLICT_MESSAGE, q_time_pregrid_pipeline_prereqs,
    validate_q_time_pregrid_factor)

CODE = Path(__file__).resolve().parents[1]
BIN = CODE / "bin" / "integrate_likelihood_extrinsic_batchmode"

# The driver's own combined wording for the vectorized/rotation-slow/freqresponse/
# calibration exclusion group (bin/integrate_likelihood_extrinsic_batchmode, the
# `if not opts.vectorized or opts.rotation_slow or opts.freqresponse or
# opts.calibration_envelope_directory:` branch) -- ONE message for all four
# prerequisites, unlike the builder's itemised list, so parity is checked by
# refuse-or-not rather than by matching this string verbatim.
_DRIVER_EXCLUSION_MESSAGE = "is currently restricted to ordinary vectorized NoLoop"


def _driver_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CODE) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env


def _run_driver(*extra_args, factor=8):
    cmd = [sys.executable, str(BIN), "--event-time", "1000000000.0",
           "--q-time-pregrid-factor", str(factor)] + list(extra_args)
    return subprocess.run(cmd, env=_driver_env(), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)


# (label, driver flags, matching ILE-argument-string tokens for the builder check)
_GUARD_CASES = [
    ("honourable", ["--vectorized"], "--vectorized"),
    ("missing_vectorized", [], ""),
    ("rotation_slow", ["--vectorized", "--rotation-slow"], "--vectorized --rotation-slow"),
    ("freqresponse", ["--vectorized", "--freqresponse"], "--vectorized --freqresponse"),
    ("calibration_nonempty",
     ["--vectorized", "--calibration-envelope-directory", "/tmp/cal"],
     "--vectorized --calibration-envelope-directory /tmp/cal"),
    # The MINOR finding this PR fixes: an EMPTY value is falsy, so neither side should
    # refuse.  Regression-tests both the driver's pre-existing truthiness check and the
    # builder's now-matching one (RIFT/likelihood/q_time_pregrid.py,
    # _PIPELINE_EXCLUDING_VALUE_ILE_FLAGS).
    ("calibration_empty",
     ["--vectorized", "--calibration-envelope-directory", ""],
     "--vectorized --calibration-envelope-directory "),
]


@pytest.mark.parametrize("label,driver_args,builder_ile_args", _GUARD_CASES,
                          ids=[c[0] for c in _GUARD_CASES])
def test_builder_refuses_exactly_when_driver_refuses(label, driver_args, builder_ile_args):
    proc = _run_driver(*driver_args)
    driver_refuses = proc.returncode != 0 and _DRIVER_EXCLUSION_MESSAGE in proc.stdout
    missing = q_time_pregrid_pipeline_prereqs(8, "X " + builder_ile_args)
    builder_refuses = bool(missing)
    assert builder_refuses == driver_refuses, (
        label, "builder missing=%r" % (missing,), proc.stdout[-2000:])


def test_stencil_conflict_wording_matches_the_shared_constant():
    """The one place the two sides' wording IS asserted verbatim: STENCIL_CONFLICT_MESSAGE
    is a module-level constant reproduced from the driver's own raise, not retyped at the
    call site, so a future edit to either side that lets them drift is caught here."""
    proc = _run_driver("--vectorized", "--interpolate-time", "nearest")
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert STENCIL_CONFLICT_MESSAGE in proc.stdout, proc.stdout[-2000:]
    missing = q_time_pregrid_pipeline_prereqs(8, "X --vectorized --interpolate-time nearest")
    assert any(STENCIL_CONFLICT_MESSAGE in m for m in missing), missing


def test_explicit_cubic_stencil_is_accepted_by_both():
    proc = _run_driver("--vectorized", "--interpolate-time", "cubic")
    # The driver proceeds past this guard entirely and dies later on an unrelated missing
    # physical parameter; what is under test is only that OUR guard did not fire.
    assert STENCIL_CONFLICT_MESSAGE not in proc.stdout, proc.stdout[-2000:]
    assert _DRIVER_EXCLUSION_MESSAGE not in proc.stdout, proc.stdout[-2000:]
    missing = q_time_pregrid_pipeline_prereqs(8, "X --vectorized --interpolate-time cubic")
    assert missing == [], missing


# The single case the shipped test used (factor=3) is invisible to a driver that silently
# starts accepting a NEARBY factor: widening the driver's legal set to (1, 4, 8) left this
# test at 3 with nothing to say about 4 (PR #291 review, MAJOR #2, mutation-tested).
# Parametrized over several illegal values on both sides of the legal set.
@pytest.mark.parametrize("factor", [2, 3, 4, 16])
def test_illegal_factor_is_refused_by_both(factor):
    proc = _run_driver("--vectorized", factor=factor)
    assert proc.returncode != 0, proc.stdout[-2000:]
    # Compare against the SHARED validator's own message rather than a retyped literal, so a
    # wording change on either side is caught here rather than silently drifting (same
    # discipline as test_stencil_conflict_wording_matches_the_shared_constant above): the
    # driver now calls this exact function (bin/integrate_likelihood_extrinsic_batchmode),
    # not an independent tuple+message, so this is a WIRING check, not a tautology.
    with pytest.raises(ValueError) as exc:
        validate_q_time_pregrid_factor(factor)
    assert str(exc.value) in proc.stdout, (factor, proc.stdout[-2000:])


# (label, driver extra args, builder ILE-argument-string tokens, candidate flags the token is
# ambiguous between).  Measured on pcdev11 against the real driver (PR #291 review, MAJOR #1):
# neither token names an actual driver option, but each is an unambiguous-looking PREFIX of
# more than one -- optparse itself refuses these with 'ambiguous option: ...' before reaching
# ANY of the guards in _GUARD_CASES above, and the old prefix-of-one-flag _matches() attributed
# the refusal to whichever guard flag it happened to be checking instead.
_AMBIGUOUS_CASES = [
    ("rotation",
     ["--vectorized", "--rotation", "0.1"], "--vectorized --rotation 0.1",
     ["--rotation-n-harmonics", "--rotation-p-max", "--rotation-slow"]),
    ("calibration_e",
     ["--vectorized", "--calibration-e", "/tmp/cal"], "--vectorized --calibration-e /tmp/cal",
     ["--calibration-envelope-directory", "--calibration-export-posterior"]),
]


@pytest.mark.parametrize("label,driver_args,builder_ile_args,candidates", _AMBIGUOUS_CASES,
                          ids=[c[0] for c in _AMBIGUOUS_CASES])
def test_ambiguous_abbreviation_refuses_on_both_sides_with_matching_classification(
        label, driver_args, builder_ile_args, candidates):
    proc = _run_driver(*driver_args)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "ambiguous option:" in proc.stdout, proc.stdout[-2000:]
    # The driver's optparse refuses BEFORE reaching the vectorized/rotation-slow/freqresponse/
    # calibration exclusion branch -- confirms this refusal is the ambiguity, not a coincidental
    # hit of the OTHER guard.
    assert _DRIVER_EXCLUSION_MESSAGE not in proc.stdout, proc.stdout[-2000:]
    for candidate in candidates:
        assert candidate in proc.stdout, (candidate, proc.stdout[-2000:])

    missing = q_time_pregrid_pipeline_prereqs(8, "X " + builder_ile_args)
    assert missing, (label, "builder approved an ambiguous token")
    assert any("ambiguous option:" in m for m in missing), missing
    # And NOT misattributed to one of the ordinary exclusion guards (the bug this test is for):
    assert not any(m.startswith("incompatible") for m in missing), missing
    for candidate in candidates:
        assert any(candidate in m for m in missing), (candidate, missing)
