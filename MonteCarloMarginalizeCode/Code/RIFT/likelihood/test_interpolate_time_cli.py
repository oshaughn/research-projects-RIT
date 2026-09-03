#!/usr/bin/env python3
# RIFT-CI-GATE: q-window-stencil
# ^ registers this file with .travis/test-q-window-stencil.sh, run by ci.yml's
#   q-window-stencil-check job.  Membership lives here, in the test file, so that
#   adding a test needs no edit to any shared list.  Do not reword the line above.
# ---------------------------------------------------------------------------------
# WHY THIS FILE IS IN q-window-stencil-check.  Moved verbatim from the comment block
# above that job's hand-maintained file list in .github/workflows/ci.yml; it lives
# here now so that registering a test needs no edit to a shared file.
#
# test_interpolate_time_cli runs the three scripts as real SUBPROCESSES (~30 s). That
# cost is the point: the unit tests exercise the resolver and the gate predicate, but
# neither can see whether the SCRIPTS are still wired to them. Reverting a parser to
# const=None, or deleting a script's resolver call, leaves every unit test green while
# restoring a bare flag that silently does nothing. All three mutations were checked to
# fail these tests before they landed.
# ---------------------------------------------------------------------------------
"""test_interpolate_time_cli -- the stencil flag AT THE COMMAND LINE, in real subprocesses.

WHY SUBPROCESSES AND NOT UNIT CALLS.  test_time_interp_choice exercises
resolve_interpolate_time_request directly, which proves the resolver is right but proves NOTHING
about how the two pipeline scripts are wired to it.  Reverting either parser to `const=None`, or
deleting either script's call to the resolver, leaves every one of those unit tests green while
restoring the original defect: a bare `--internal-ile-interpolate-time` that silently does
nothing.  The wiring is the thing that broke, so the wiring is what has to be tested.

Same argument for the driver's honoured-path gate: the predicate can be correct in isolation and
still be unreachable, or reachable and mis-wired.

These run the real scripts with the real interpreter.  Each invocation costs a few seconds of
lal/numba import, which is why the case list is kept to the ones that DISTINGUISH behaviours
rather than every combination.  No data files are needed -- all three scripts reach the relevant
validation before touching frames or PSDs.

    python3 test_interpolate_time_cli.py       # or: pytest test_interpolate_time_cli.py
"""
from __future__ import print_function

import os
import re
import subprocess
import sys

from RIFT.likelihood.time_interp_choice import CROSSOVER_GUIDANCE

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
BIN = os.path.join(CODE_ROOT, 'bin')

HELPER = os.path.join(BIN, 'helper_LDG_Events.py')
PSEUDO = os.path.join(BIN, 'util_RIFT_pseudo_pipe.py')
DRIVER = os.path.join(BIN, 'integrate_likelihood_extrinsic_batchmode')

PIPELINE_ENTRY_POINTS = [('helper_LDG_Events.py', HELPER),
                         ('util_RIFT_pseudo_pipe.py', PSEUDO)]

# EVERY surface that hands a user a stencil recommendation.  The driver's --help was omitted from
# the original guidance test even though it interpolates the same constant, so that third copy
# could drift silently -- which is the exact failure this whole test file exists to prevent.
ADVICE_SURFACES = PIPELINE_ENTRY_POINTS + [
    ('integrate_likelihood_extrinsic_batchmode', DRIVER)]

# Values the guidance constant has previously held and that must never reappear in user-facing
# text.  A positive assertion ("the current constant is present") cannot catch a SUPERSEDED claim
# left standing beside it -- that is how a rendered error message came to read "the crossover is
# between 20 and 35 the crossover rises with fmin ...", splicing the retracted rule onto its
# replacement, while every test passed.  Add each retired value here when the constant changes.
RETIRED_GUIDANCE_FRAGMENTS = (
    # NB: no trailing ' Msun' -- the splice that actually shipped read "...between 20 and 35 "
    # immediately followed by the NEW constant, so a fragment ending in 'Msun' could not match it.
    # Keep retired fragments as short as is still unambiguous.
    'the crossover is between 20 and 35',
    'unless the total mass is below',
    'prefer sinc at any mass',
    # v3 of the constant, retired by the #109 review commit.  Fragment chosen to be absent from
    # the current value: 'measured over 9-55 Msun only' was v3's scope clause and v4 words it
    # differently.  COPY RETIRED TEXT FROM THE DIFF, never retype it -- fragment [0] was
    # originally written with a trailing ' Msun' the real splice did not have, and could not fire.
    '(measured over 9-55 Msun only)',
)


def _run(script, args, timeout=300):
    """Run a script and return its combined output.  Never raises on non-zero exit."""
    env = dict(os.environ)
    env['PYTHONPATH'] = CODE_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['OMP_NUM_THREADS'] = '1'
    env.setdefault('CUDA_VISIBLE_DEVICES', '')      # keep these CPU-only and deterministic
    proc = subprocess.Popen([sys.executable, script] + args, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    if not isinstance(out, str):
        out = out.decode('utf-8', 'replace')
    return out


def _squash(text):
    """Collapse whitespace, so a match is not defeated by argparse's line wrapping.

    argparse rewraps help text to the terminal width, so a phrase that is present can still fail a
    naive line-oriented grep.  This bit us while writing the test.
    """
    return re.sub(r'\s+', ' ', text)


def test_bare_flag_is_rejected_by_both_entry_points():
    """The defect this exists for: `const=None` makes a bare flag == an absent flag.

    A unit test on the resolver cannot see this -- the bug lives in the parser declaration.
    """
    for name, script in PIPELINE_ENTRY_POINTS:
        out = _squash(_run(script, ['--internal-ile-interpolate-time']))
        assert 'given with no value' in out, (
            "%s accepted a BARE --internal-ile-interpolate-time. If the parser has gone back to "
            "const=None, a bare flag is indistinguishable from omitting it and the feature is "
            "silently disabled. Output was: %s" % (name, out[-400:]))
        print("%-26s bare flag rejected: OK" % name)


def test_typo_and_retired_auto_are_rejected_by_both_entry_points():
    for name, script in PIPELINE_ENTRY_POINTS:
        out = _squash(_run(script, ['--internal-ile-interpolate-time', 'sinK']))
        assert 'unrecognised Q_lm time-interpolation stencil' in out, \
            "%s accepted a typo'd stencil: %s" % (name, out[-400:])
        out = _squash(_run(script, ['--internal-ile-interpolate-time', 'True']))
        assert 'REMOVED' in out, \
            "%s did not reject the retired 'True' spelling: %s" % (name, out[-400:])
        print("%-26s typo and retired 'True' rejected: OK" % name)


def test_valid_and_off_spellings_pass_the_resolver_in_both_entry_points():
    """These must NOT trip the stencil validation.  They will fail later for unrelated reasons
    (no event, no data) -- what matters is that the failure is not ours."""
    ours = re.compile(r'unrecognised Q_lm|given with no value|REMOVED')
    for name, script in PIPELINE_ENTRY_POINTS:
        for value in ('sinc', 'cubic', 'nearest', 'False'):
            out = _squash(_run(script, ['--internal-ile-interpolate-time', value]))
            assert not ours.search(out), \
                "%s wrongly rejected --internal-ile-interpolate-time %s: %s" % (
                    name, value, out[-400:])
        print("%-26s valid stencils and 'False' accepted: OK" % name)


def test_help_text_carries_the_same_crossover_guidance_in_both_entry_points():
    """Pin the DUPLICATED guidance, which has already drifted once.

    util_RIFT_pseudo_pipe.py was left recommending the pre-IMR "cubic unless below ~4 Msun" -- the
    measurably WORSE stencil across roughly 4-20 Msun -- while the other copies had been updated.
    Both helps must carry the canonical phrase from time_interp_choice, and neither may carry the
    old recommendation.
    """
    for name, script in ADVICE_SURFACES:
        out = _squash(_run(script, ['--help']))
        assert CROSSOVER_GUIDANCE in out, (
            "%s --help does not contain the canonical crossover guidance %r. If the measurement "
            "changed, update CROSSOVER_GUIDANCE in time_interp_choice and every help string "
            "together -- that is what this test is for." % (name, CROSSOVER_GUIDANCE))
        for retired in RETIRED_GUIDANCE_FRAGMENTS:
            assert retired not in out, (
                "%s --help still carries retired guidance %r. A superseded recommendation left "
                "standing beside the current one reads as authoritative." % (name, retired))
        print("%-40s help carries canonical guidance, no retired text: OK" % name)


def test_error_messages_carry_the_canonical_guidance_too():
    """The error paths advise users as much as --help does, and were never checked.

    A bare flag and a retired 'True' both print guidance. One of them shipped rendering the
    RETIRED constant spliced onto the current one -- ungrammatical, and advising the superseded
    rule -- while every test passed, because nothing asserted on those strings at all.
    """
    from RIFT.likelihood.time_interp_choice import (
        BARE_FLAG_SENTINEL, resolve_interpolate_time_request)
    for value in (BARE_FLAG_SENTINEL, 'True'):
        try:
            resolve_interpolate_time_request(value)
        except ValueError as e:
            msg = _squash(str(e))
        else:
            raise AssertionError("%r must raise" % value)
        assert CROSSOVER_GUIDANCE in msg, (
            "the error for %r does not carry the canonical guidance: %r" % (value, msg))
        for retired in RETIRED_GUIDANCE_FRAGMENTS:
            assert retired not in msg, (
                "the error for %r still carries retired guidance %r: %r" % (value, retired, msg))
        print("error path for %-12s carries canonical guidance, no retired text: OK" % repr(value))


def test_driver_refuses_configurations_that_cannot_honour_the_stencil():
    """The conjunctive gate, exercised through the real CLI.

    Each case names a prerequisite that, if missing, means the likelihood actually executed takes
    no sub-sample stencil -- so accepting the flag would run a different likelihood than the one
    the user asked for, silently.
    """
    cases = [
        (['--interpolate-time', 'sinc', '--gpu', '--force-xpy', '--time-marginalization'],
         '--vectorized',
         "GPU without --vectorized reaches DiscreteFactoredLogLikelihoodViaArrayVector"),
        (['--interpolate-time', 'sinc', '--vectorized', '--gpu', '--force-xpy'],
         '--time-marginalization',
         "no time marginalization reaches FactoredLogLikelihood, which has no stencil argument"),
        (['--interpolate-time', 'sinc', '--vectorized', '--force-xpy', '--time-marginalization'],
         'one of --gpu',
         "plain --vectorized reaches the array-vector likelihood, which has no stencil argument"),
    ]
    for args, expect_missing, why in cases:
        out = _squash(_run(DRIVER, args))
        assert 'cannot honour it' in out and expect_missing in out, (
            "driver accepted a configuration that cannot honour the stencil (%s); expected it to "
            "report missing %r. Output: %s" % (why, expect_missing, out[-500:]))
        print("driver rejects, missing %-22s : OK" % expect_missing)


def test_driver_does_not_gate_the_default_stencil():
    """'nearest' is the historical behaviour and must never be refused.

    Without this, the gate could be tightened into breaking every run that does not ask for
    interpolation at all -- a far worse regression than the one it prevents.
    """
    out = _squash(_run(DRIVER, ['--interpolate-time', 'nearest', '--vectorized']))
    assert 'cannot honour it' not in out, \
        "the gate must not fire for the default 'nearest' stencil: %s" % out[-400:]
    print("driver does not gate 'nearest': OK")


if __name__ == "__main__":
    test_bare_flag_is_rejected_by_both_entry_points()
    test_typo_and_retired_auto_are_rejected_by_both_entry_points()
    test_valid_and_off_spellings_pass_the_resolver_in_both_entry_points()
    test_help_text_carries_the_same_crossover_guidance_in_both_entry_points()
    test_error_messages_carry_the_canonical_guidance_too()
    test_driver_refuses_configurations_that_cannot_honour_the_stencil()
    test_driver_does_not_gate_the_default_stencil()
    print("\nPASS")
