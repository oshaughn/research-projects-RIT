#!/usr/bin/env python3
# RIFT-CI-GATE: q-window-stencil
# ^ registers this file with .travis/test-q-window-stencil.sh, run by ci.yml's
#   q-window-stencil-check job.  Added with the test rather than to a shared list in
#   ci.yml: that list is what this PR conflicted with when #242 replaced it.
"""test_batchmode_stencil_default -- what an ILE run gets when it says NOTHING.

WHY THIS FILE EXISTS.  Every other stencil test passes an EXPLICIT --interpolate-time, so the
whole suite was blind to the value the flag takes when it is absent -- which is the value
essentially every production run uses, because neither pipeline entry point emits the flag unless
asked (helper_LDG_Events.py only appends '--interpolate-time <name>' inside
`if time_interp_choice is not None`).  That blindness is exactly how the two ILE drivers came to
ship OPPOSITE defaults for the same physical choice, batchmode 'nearest' against jax 'sinc'
(issue #233), with the disagreement growing as SNR^2 so it reads as an amplitude-dependent bug in
one of the codes rather than as a configuration difference.

The default moved to time_interp_choice.TIME_INTERP_DEFAULT on 2026-09-02.  A default change is
the most reachable kind of result change there is, so it is pinned three ways here:

  1. the CONSTANT -- one definition, shared with the jax driver, so they cannot drift again;
  2. the WIRING -- the driver's add_option really reads that constant, checked with `ast` rather
     than by trusting a re-typed literal (this is the check test_jax_stencil_parity already has
     for --interp and the batchmode driver did not have for --interpolate-time);
  3. the BEHAVIOUR -- real subprocesses, because the interesting part of this change is not the
     new value but the FOUR places where a DEFAULT must behave differently from a REQUEST.

ON (3), STATED PLAINLY, because it is the part a reviewer should attack.  The driver refuses an
explicit --interpolate-time it cannot honour.  As a DEFAULT that same refusal would convert every
configuration lacking --time-marginalization/--vectorized/--gpu from working to a startup
ValueError, so the default is downgraded to 'nearest' instead, and the same distinction keeps the
default out of the time-posterior export mode, off the fused calibration kernel, and off the
LEGACY scalar path's `interpolate` boolean (the fourth, found by adversarial review on 2026-09-03:
correct in the code, pinned by nothing).  Each of those four is a separate test below; deleting the
distinction makes at least one of them fail.

TWO FURTHER THINGS THE SAME REVIEW FOUND, and the reason they belong in this file.  A guard can be
wrong by being TOO BROAD as easily as by being absent, and a NOTICE can be wrong by describing
behaviour the code does not have -- neither shows up as a failure anywhere, because a needless
downgrade looks exactly like the historical behaviour and a wrong remedy string still prints.  So
this file now also pins the OTHER edge of each: that a bare --calibration-fused-kernel with no
calibration envelope does NOT downgrade (nothing to protect), that each downgrade's stated remedy
matches what the driver actually does on that command line, and that the band-limited quadrature
notice fires under the new default and stays silent for 'simpson' and for an explicit 'nearest'.

Subprocess cases cost a few seconds of lal/numba import each, so the list is kept to the ones
that DISTINGUISH behaviours.

    python3 test_batchmode_stencil_default.py     # or: pytest test_batchmode_stencil_default.py
"""
from __future__ import print_function

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

from RIFT.likelihood.time_interp_choice import (TIME_INTERP_CHOICES,
                                                TIME_INTERP_DEFAULT)

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
BIN = os.path.join(CODE_ROOT, 'bin')
DRIVER = os.path.join(BIN, 'integrate_likelihood_extrinsic_batchmode')
HELPER = os.path.join(BIN, 'helper_LDG_Events.py')
PSEUDO = os.path.join(BIN, 'util_RIFT_pseudo_pipe.py')

# The smallest command line the driver accepts that satisfies all three stencil prerequisites.
# --force-xpy keeps the identical NoLoop code path on numpy, so this runs on a CI box with no GPU.
HONOURED = ['--time-marginalization', '--vectorized', '--gpu', '--force-xpy']

# --calibration-fused-kernel can only LOSE a fused kernel if one would have run, and that needs a
# calibration envelope: use_fused_calmarg (batchmode:3261) is
# `calibration_marginalization and opts.calibration_fused_kernel`, and calibration_marginalization
# (:1317) is exactly bool(opts.calibration_envelope_directory).  The path is never opened this
# early -- it is first read around :1300, long after the banner these tests parse -- so a
# non-existent directory is the cheapest way to put the driver in the calmarg configuration.  Using
# the flag WITHOUT this was the P2 review finding these tests missed: the guard fired on runs where
# no kernel could run at all.
CALMARG = ['--calibration-envelope-directory', '/nonexistent-calibration-envelope-for-tests']
FUSED = ['--calibration-fused-kernel'] + CALMARG
# NOTE the default --calibration-n-realizations is 100, i.e. > 1, so FUSED alone is a genuine
# fused configuration.  test_no_fused_configuration_that_cannot_fuse_downgrades_the_default turns
# that knob off explicitly as one of its cases.


def _run(script, args, timeout=300, in_tmpdir=False):
    """Run a script and return its combined output.  Never raises on non-zero exit.

    Same idiom as test_interpolate_time_cli._run, deliberately: sys.executable rather than a
    hard-coded interpreter, CODE_ROOT prepended to PYTHONPATH, and CUDA hidden so the cases are
    CPU-only and deterministic.  The driver exits non-zero on every case here (there are no data
    files) -- what is under test is the text it prints BEFORE it gets that far.
    """
    env = dict(os.environ)
    env['PYTHONPATH'] = CODE_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['OMP_NUM_THREADS'] = '1'
    env.setdefault('CUDA_VISIBLE_DEVICES', '')
    # in_tmpdir: helper_LDG_Events writes helper_*_args.txt and local.cache into its CWD, so
    # running it from the source tree leaves untracked files behind.  Give it a scratch cwd.
    tmp = tempfile.mkdtemp(prefix='stencil_default_') if in_tmpdir else None
    try:
        proc = subprocess.Popen([sys.executable, script] + args, env=env, cwd=tmp,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    if not isinstance(out, str):
        out = out.decode('utf-8', 'replace')
    return out


def _squash(text):
    return re.sub(r'\s+', ' ', text)


def _legacy_scalar_flag(out):
    """The `legacy scalar path interpolate=` field of the same banner.

    Separate from _stencil_banner because it pins a DIFFERENT variable with a different
    consumer: opts._legacy_interpolate_time, which is what
    FactoredLogLikelihoodTimeMarginalized(..., interpolate=...) is handed on the NON-VECTORIZED
    path.  The stencil name and this boolean can disagree, and one mutation makes them.
    """
    m = re.search(r'legacy scalar path interpolate=(\w+)', _squash(out))
    assert m, "driver printed no legacy-scalar field; output was: %s" % out[-1500:]
    return m.group(1)


def _stencil_banner(out):
    """The resolved stencil, read off the driver's own startup line.

    Reading the banner rather than re-deriving the value is the point: the banner is what a
    configuration audit reads off a completed run's log, so if it and the code disagree the test
    should fail.
    """
    # Anchored on '(from --interpolate-time' so it cannot match the DOWNGRADE line, which is a
    # different statement about the same subject.  An earlier version of this helper was not
    # anchored and read the downgrade notice as the banner, reporting the wrong stencil.
    m = re.search(r'Q_lm sub-sample time stencil: (\S+) \(from --interpolate-time', _squash(out))
    assert m, "driver printed no stencil banner; output was: %s" % out[-1500:]
    return m.group(1)


# ---------------------------------------------------------------------------
# 1. the constant
# ---------------------------------------------------------------------------
def test_default_is_a_real_stencil_and_is_sinc():
    assert TIME_INTERP_DEFAULT in TIME_INTERP_CHOICES, (
        "TIME_INTERP_DEFAULT %r is not a stencil this tree implements (%r)"
        % (TIME_INTERP_DEFAULT, TIME_INTERP_CHOICES))
    assert TIME_INTERP_DEFAULT == 'sinc', (
        "default stencil changed to %r -- intentional? It changes results for every ILE run that "
        "does not pass --interpolate-time, and the error grows as SNR^2, so update "
        "DESIGN_q_window_stencil.md 9.6 and the --interpolate-time help text in the same commit."
        % (TIME_INTERP_DEFAULT,))


def test_the_two_ile_drivers_ship_the_same_default():
    """Issue #233 in one assertion, checked WITHOUT importing jax.

    This used to `from RIFT.likelihood.jax_ile.core import JAX_INTERP_DEFAULT` and skip when the
    import failed.  That skip fired in the very job this file belongs to -- q-window-stencil-check
    runs on a numpy+lal image with no jaxlib by design -- so the one assertion tying the two
    drivers together was the one assertion CI never evaluated.  A gate that is skipped exactly
    where it is needed is not a gate.

    What actually forbids drift is that core.py BINDS the name rather than copying the value, so
    this reads the binding out of the source with ast and never imports the module.  That is a
    stronger check than the old equality as well as a runnable one: two literals that happen to
    read 'sinc' today would have satisfied the import version and would fail here.
    """
    core = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jax_ile', 'core.py')
    assert os.path.exists(core), core
    with open(core) as handle:
        tree = ast.parse(handle.read())
    bound = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'JAX_INTERP_DEFAULT' for t in node.targets):
            bound = node.value
    assert bound is not None, "no module-level JAX_INTERP_DEFAULT assignment in %s" % core
    assert isinstance(bound, ast.Name) and bound.id == 'TIME_INTERP_DEFAULT', (
        "jax_ile.core must ALIAS the shared default (JAX_INTERP_DEFAULT = TIME_INTERP_DEFAULT), "
        "not re-type it: found %r. Two independently written literals are how the drivers came to "
        "ship opposite defaults in the first place, and a comparison run at defaults then measures "
        "a flag, with the difference growing as SNR^2. See issue #233."
        % (ast.dump(bound),))


# ---------------------------------------------------------------------------
# 2. the wiring: the driver reads the constant, it does not re-type the value
# ---------------------------------------------------------------------------
def _driver_option_default_expr(option):
    with open(DRIVER) as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'add_option'):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == option):
            continue
        for kw in node.keywords:
            if kw.arg == 'default':
                return kw.value
        return None
    raise AssertionError("no add_option(%r) found in %s" % (option, DRIVER))


def test_driver_default_is_the_absent_sentinel_not_a_stencil_literal():
    """`default=None` is load-bearing, not a stylistic choice.

    Every guard downstream keys on "was this asked for, or inherited?", and that question is
    decided by `opts.interpolate_time is None`.  Baking the stencil name straight into
    `default=` would make an inherited default indistinguishable from an explicit request, which
    silently re-arms the refusal, the export flip and the fused-kernel downgrade for runs that
    passed no flag at all.
    """
    default = _driver_option_default_expr('--interpolate-time')
    assert isinstance(default, ast.Constant) and default.value is None, (
        "--interpolate-time default is %s; it must be None so the driver can tell an omitted "
        "flag from an explicit one (see _interp_time_from_default)."
        % ast.dump(default) if default is not None else "--interpolate-time has no default=")


def test_driver_resolves_the_absent_flag_through_the_shared_constant():
    """The name TIME_INTERP_DEFAULT must actually appear in the resolution, not just be imported.

    Mutation that this kills: replacing `opts._noloop_time_interp = TIME_INTERP_DEFAULT` with a
    re-typed `'sinc'`.  That passes every value-level test in this file and re-creates the exact
    condition of issue #233 -- two literals in two files that agree today.
    """
    with open(DRIVER) as handle:
        source = handle.read()
    assert 'from RIFT.likelihood.time_interp_choice import TIME_INTERP_DEFAULT' in source
    assert 'opts._noloop_time_interp = TIME_INTERP_DEFAULT' in source, (
        "the driver no longer resolves an absent --interpolate-time through the shared constant")


# ---------------------------------------------------------------------------
# 3. the behaviour: a DEFAULT is downgraded exactly where a REQUEST is refused
# ---------------------------------------------------------------------------
def test_honoured_configuration_gets_the_new_default():
    out = _run(DRIVER, HONOURED)
    assert _stencil_banner(out) == TIME_INTERP_DEFAULT, (
        "a configuration that CAN honour a stencil did not get the default %r: %s"
        % (TIME_INTERP_DEFAULT, out[-1500:]))


def test_unhonourable_configuration_downgrades_the_default_instead_of_refusing():
    """The regression this change most plausibly introduces, and the reason for the whole design.

    '--vectorized' alone runs a likelihood with no time_interp argument at all.  Under the old
    'nearest' default that configuration started normally; a naive default flip turns it into a
    startup ValueError with no command line changed anywhere.
    """
    out = _run(DRIVER, ['--vectorized'])
    squashed = _squash(out)
    assert 'cannot honour it' not in squashed, (
        "the honoured-path gate REFUSED a run that passed no --interpolate-time at all. A "
        "default must never do that -- it breaks working configurations with no flag changed. "
        "Output: %s" % out[-1500:])
    assert _stencil_banner(out) == 'nearest', (
        "an unhonourable configuration must fall back to 'nearest' (the pre-2026-09-02 default), "
        "so the run is unchanged: %s" % out[-1500:])
    assert 'Q_lm stencil DEFAULT' in squashed and 'NOT APPLIED' in squashed, (
        "the fallback must be ANNOUNCED. A stencil that is not running is the one thing the log "
        "has to say: %s" % out[-1500:])


def test_the_downgrade_also_takes_the_legacy_scalar_path_down_with_it():
    """The FIFTH thing a default had to be prevented from doing, found by adversarial review.

    opts._legacy_interpolate_time is derived from the PROVISIONAL default a hundred lines before
    the downgrade runs, and it is not the stencil: it is the plain boolean handed to
    FactoredLogLikelihoodTimeMarginalized(..., interpolate=...) at batchmode:3645 and :4002, which
    is the likelihood that ACTUALLY RUNS whenever --vectorized is absent.  A non-'nearest'
    provisional default makes it True, i.e. an omitted --interpolate-time would silently switch
    every non-vectorized run onto the legacy path's unrelated cubic interpolation -- a result
    change of the same class as the three in 9.6.3, on a path where no sub-sample stencil exists
    at all.  The downgrade block resets it; nothing pinned that until this test.

    '--time-marginalization' alone is deliberate: it is unhonourable (no --vectorized, no
    --gpu/--rotation-slow/--freqresponse), so the downgrade fires, AND it is the branch where the
    legacy scalar call site is reachable.  Base behaviour is 'False' and must stay 'False'.
    """
    out = _run(DRIVER, ['--time-marginalization'])
    assert _stencil_banner(out) == 'nearest', out[-1500:]
    assert _legacy_scalar_flag(out) == 'False', (
        "the downgrade left opts._legacy_interpolate_time True. On this configuration the "
        "likelihood that runs is the LEGACY scalar path, whose `interpolate` argument this flag "
        "is -- so the default would silently turn on the legacy cubic interpolation for every "
        "non-vectorized run: %s" % out[-1500:])


def test_an_explicit_request_is_still_refused_on_the_same_configuration():
    """The downgrade must not disarm the refusal.  Same command line as the test above, one flag
    added, opposite required outcome -- which is why they are separate tests and not one."""
    out = _squash(_run(DRIVER, ['--interpolate-time', TIME_INTERP_DEFAULT, '--vectorized']))
    assert 'cannot honour it' in out, (
        "an EXPLICIT --interpolate-time %r was not refused on a configuration that cannot honour "
        "it. That refusal is what stops a comparison campaign being run against an inert flag."
        % TIME_INTERP_DEFAULT)


def test_default_does_not_change_the_time_posterior_export_mode():
    """resolve_time_posterior_export_mode maps `auto` to 'continuous' for any non-'nearest'
    stencil, so the same one-line default change would otherwise have flipped the fair-draw time
    export of every --resample-time-marginalization run: a denser re-evaluation of the whole
    likelihood, a different draw algorithm, two extra output columns, and a new reachable
    MemoryError.  The export must key on an EXPLICIT stencil only."""
    out = _squash(_run(DRIVER, HONOURED + ['--resample-time-marginalization', '--fairdraw-extrinsic-output']))
    assert 'Time-posterior export: grid' in out, (
        "an inherited default changed the time-posterior export mode; it must stay 'grid' unless "
        "the stencil or the export was asked for. Output: %s" % out[-1500:])


def test_an_explicit_stencil_still_opts_into_the_continuous_export():
    """The other half of the previous test: asking for a stencil is still an opt-in to the better
    export, so this change narrows the trigger rather than removing the feature."""
    out = _squash(_run(DRIVER, HONOURED + ['--resample-time-marginalization',
                                           '--fairdraw-extrinsic-output',
                                           '--interpolate-time', 'sinc']))
    assert 'Time-posterior export: continuous' in out, (
        "an EXPLICIT --interpolate-time sinc no longer resolves `auto` to the continuous export: "
        "%s" % out[-1500:])


def test_default_stays_off_the_fused_calibration_kernel():
    """The fused calmarg kernels implement 'nearest' only, and the driver's three call sites fall
    back to cal_method='loop' (and drop cal_distmarg) for any other stencil, silently.  A default
    must not spend someone else's --calibration-fused-kernel that way."""
    out = _run(DRIVER, HONOURED + FUSED)
    assert _stencil_banner(out) == 'nearest', (
        "the default stencil was applied on top of --calibration-fused-kernel, which silently "
        "moves the run off the fused kernel it explicitly asked for: %s" % out[-1500:])


def test_an_explicit_stencil_with_the_fused_kernel_says_so():
    """Unchanged behaviour (the user named both flags), but it used to be silent at all three
    call sites, which contradicts this option's own 'REFUSED, not ignored' promise."""
    out = _squash(_run(DRIVER, HONOURED + FUSED + ['--interpolate-time', 'sinc']))
    assert '--calibration-fused-kernel: NOT USED' in out, (
        "losing the fused kernel to an explicit stencil is still silent: %s" % out[-1500:])


def test_the_fused_predicate_has_ONE_definition_shared_by_BOTH_call_sites():
    """The fourth P2 finding on PR #237, and the reason it is the last one of its kind.

    The condition "will a fused calibration kernel actually run?" is needed twice: at startup, to
    decide whether the inherited stencil is downgraded, and at dispatch, to pick cal_method.
    Written as two expressions it drifted THREE TIMES IN ONE DAY, always in the same direction --
    flag -> flag+envelope -> flag+envelope+path -- because an over-broad predicate downgrades to
    'nearest', which is indistinguishable from the historical behaviour, so nothing ever fails.

    A value test cannot catch that; the next conjunct someone forgets will be a fourth expression
    that agrees with these on every case anyone thought to write down.  So this pins the STRUCTURE:
    one function, called at both sites, with no second expression left behind.
    """
    with open(DRIVER) as handle:
        source = handle.read()
    tree = ast.parse(source)
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == 'fused_calmarg_in_use']
    assert len(funcs) == 1, (
        "the shared fused-calibration-kernel predicate is gone; if it was inlined again, the "
        "startup guard and use_fused_calmarg are two expressions once more and will drift")
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'fused_calmarg_in_use']
    assert len(calls) >= 2, (
        "fused_calmarg_in_use is called %d time(s); BOTH the startup stencil guard and "
        "use_fused_calmarg must go through it, or one of them is a second copy again"
        % len(calls))
    # and no site re-derives it: the old shapes, in either order, must not reappear
    for shape in ("bool(calibration_marginalization and opts.calibration_fused_kernel)",
                  "bool(opts.calibration_fused_kernel) and bool(\n    opts.calibration_envelope_directory)"):
        assert shape not in source, (
            "a hand-written copy of the fused-kernel predicate is back in the driver: %r" % shape)
    # THE TRIPWIRE, pinned STRUCTURALLY and deliberately not claimed as behaviour coverage.
    #
    # The one term the early call cannot read is `calibration_marginalization`, for which it
    # substitutes bool(opts.calibration_envelope_directory) -- that IS the condition the later
    # assignment uses, so the two agree for every command line that exists, and removing the
    # comparison changes nothing testable (verified by mutation, 2026-09-03: deleting it leaves
    # all 23 tests green).  Its whole purpose is to fail if someone later derives
    # calibration_marginalization differently, and no test can reach that without editing the
    # driver.  So this asserts the tripwire is PRESENT rather than that it fires: an unexercisable
    # guard that is silently deleted is worse than one that is honestly labelled.
    compares = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
                and isinstance(n.left, ast.Name) and n.left.id == 'use_fused_calmarg'
                and any(isinstance(c, ast.Name) and c.id == '_fused_calmarg_would_run'
                        for c in n.comparators)]
    assert compares, (
        "the startup/dispatch agreement check on the fused-kernel predicate is gone. It is the "
        "only thing that makes a future change to how calibration_marginalization is derived "
        "fail loudly instead of silently re-opening the drift this function exists to end.")


def test_no_fused_configuration_that_cannot_fuse_downgrades_the_default():
    """The OTHER edge, enumerated -- a guard must not be broader than the thing it protects.

    Each of these passes --calibration-fused-kernel, and in each the fused kernel CANNOT run, so
    downgrading the inherited stencil to 'nearest' buys nothing and silently costs the accuracy
    the new default exists to provide.  They are listed one per reason rather than folded into a
    single case because they fail for DIFFERENT reasons and a single expression has already been
    wrong about three of them.
    """
    cases = [
        ("no calibration envelope: nothing to marginalize over",
         HONOURED + ['--calibration-fused-kernel']),
        ("--calibration-n-realizations 1: the library returns from its n_cal==1 branch "
         "before cal_method is read",
         HONOURED + FUSED + ['--calibration-n-realizations', '1']),
        ("--rotation-slow REPLACES the likelihood; the fused call site is in the else branch",
         ['--time-marginalization', '--vectorized', '--rotation-slow',
          '--calibration-fused-kernel'] + CALMARG),
        ("--freqresponse REPLACES the likelihood, same dispatch",
         ['--time-marginalization', '--vectorized', '--freqresponse',
          '--calibration-fused-kernel'] + CALMARG),
        ("--calibration-dump-responsibilities is a pilot: it evaluates with cal_method='loop' "
         "and returns before the production integration exists",
         HONOURED + FUSED + ['--calibration-dump-responsibilities', 'pilot_resp.npz']),
    ]
    for why, argv in cases:
        out = _run(DRIVER, argv)
        assert _stencil_banner(out) == TIME_INTERP_DEFAULT, (
            "the default stencil was downgraded to protect a fused kernel that cannot run (%s). "
            "A needless downgrade is INVISIBLE -- it looks exactly like the historical behaviour "
            "-- which is why this edge is pinned case by case: %s" % (why, out[-1500:]))
        assert 'NOT APPLIED' not in _squash(out), (
            "the driver announced a downgrade it did not need to make (%s): %s" % (why, out[-1200:]))


def test_a_bare_fused_kernel_flag_no_longer_downgrades_the_default():
    """A guard must not be BROADER than the thing it protects.  (P2 review finding, PR #237.)

    --calibration-fused-kernel with no --calibration-envelope-directory cannot run a fused kernel
    under ANY stencil: use_fused_calmarg is `calibration_marginalization and the flag`.  Keying the
    downgrade on the flag alone therefore protected nothing on this command line and silently cost
    the accuracy the new default exists to provide -- the failure mode is invisible, because a
    needless downgrade looks exactly like the historical behaviour.

    Deliberately the SAME command line as test_default_stays_off_the_fused_calibration_kernel minus
    the envelope, with the opposite required outcome, so the pair pins both edges of the condition.
    """
    out = _run(DRIVER, HONOURED + ['--calibration-fused-kernel'])
    assert _stencil_banner(out) == TIME_INTERP_DEFAULT, (
        "a --calibration-fused-kernel flag with no calibration envelope still downgraded the "
        "default stencil, though no fused kernel can run: %s" % out[-1500:])
    squashed = _squash(out)
    assert 'NOT APPLIED' not in squashed, (
        "the driver announced a downgrade it did not need to make: %s" % out[-1500:])
    assert 'NOT USED' not in squashed, (
        "the driver reported losing a fused kernel that was never going to run: %s" % out[-1500:])


def test_each_downgrade_states_the_remedy_that_actually_applies():
    """A startup notice must not promise behaviour the driver does not have.  (P2, PR #237.)

    The two downgrades have OPPOSITE remedies and one shared sentence lied about one of them.
    Naming the stencil explicitly on a prerequisite downgrade gets you REFUSED; naming it on the
    fused-kernel downgrade gets you ACCEPTED, with the kernel dropped and a notice.  The single
    message said "pass ... explicitly to be refused instead" in both cases.

    Asserted against the driver's real behaviour on the same two command lines, not just against
    the strings: the refusal claim is checked by actually adding the flag and seeing a refusal, and
    the acceptance claim by adding it and seeing the run continue.
    """
    prereq = _squash(_run(DRIVER, ['--vectorized']))
    assert 'REFUSED rather than downgraded' in prereq, (
        "the prerequisite downgrade no longer states its remedy: %s" % prereq[-1200:])
    # ... and that claim is true:
    assert 'cannot honour it' in _squash(
        _run(DRIVER, ['--interpolate-time', TIME_INTERP_DEFAULT, '--vectorized'])), \
        "the prerequisite message promises a refusal the driver does not perform"

    fused = _squash(_run(DRIVER, HONOURED + FUSED))
    assert 'ACCEPTED, not refused' in fused, (
        "the fused-kernel downgrade still borrows the prerequisite downgrade's remedy: %s"
        % fused[-1200:])
    assert 'REFUSED rather than downgraded' not in fused, (
        "the fused-kernel downgrade tells the user they will be refused; they will not: %s"
        % fused[-1200:])
    # ... and THAT claim is true: same command line plus the explicit stencil is accepted.
    accepted = _squash(_run(DRIVER, HONOURED + FUSED + ['--interpolate-time', TIME_INTERP_DEFAULT]))
    assert 'cannot honour it' not in accepted, (
        "the fused-kernel message says an explicit stencil is accepted, but it was refused: %s"
        % accepted[-1200:])
    assert '--calibration-fused-kernel: NOT USED' in accepted, (
        "the fused-kernel message says the driver will say so; it did not: %s" % accepted[-1200:])


# ---------------------------------------------------------------------------
# 4. the band-limited quadratures were measured against a stencil that is no
#    longer the default, and the pairing is now universal rather than rare
# ---------------------------------------------------------------------------
def test_bandlimited_says_its_advantage_is_unestablished_under_the_new_default():
    """The third P2 review finding on PR #237, and the reason it is not a rare corner.

    --time-marginalization-quadrature's prerequisites (--time-marginalization --vectorized --gpu)
    are a strict SUBSET of the stencil's honoured set, so EVERY run that opts into a band-limited
    quadrature without naming a stencil now gets the default one -- the regime where that module's
    own docstring measures -2.29 nats against Simpson's +1.28.  Startup said nothing.
    """
    for quadrature in ('bandlimited', 'peak-local'):
        out = _squash(_run(DRIVER, HONOURED + ['--time-marginalization-quadrature', quadrature]))
        assert 'ADVANTAGE NOT ESTABLISHED' in out, (
            "--time-marginalization-quadrature %s ran under the DEFAULT stencil with no notice "
            "that its measured advantage is for 'nearest': %s" % (quadrature, out[-1500:]))
        assert 'DEFAULT Q_lm stencil' in out, (
            "the notice does not say the stencil was inherited rather than chosen: %s"
            % out[-1500:])


def test_the_quadrature_notice_does_not_fire_where_the_numbers_still_hold():
    """The other edge, without which the notice is unfalsifiable.

    A notice that always prints carries no information.  It must be silent for 'simpson' (the
    quadrature default, which these measurements are not about) and silent for an explicit
    'nearest' (the regime the numbers WERE measured in, and the reproduce instruction the notice
    itself gives -- so if it still fired there the advice would be self-contradicting).
    """
    assert 'ADVANTAGE NOT ESTABLISHED' not in _squash(_run(DRIVER, HONOURED)), \
        "the quadrature notice fired for the historical simpson quadrature"
    out = _squash(_run(DRIVER, HONOURED + ['--time-marginalization-quadrature', 'bandlimited',
                                           '--interpolate-time', 'nearest']))
    assert 'ADVANTAGE NOT ESTABLISHED' not in out, (
        "the notice fired for the very configuration it tells the user to switch to: %s"
        % out[-1500:])


# ---------------------------------------------------------------------------
# 5. spellings that must keep meaning what they meant
# ---------------------------------------------------------------------------
def test_explicit_nearest_and_explicit_off_still_mean_nearest():
    for value in ('nearest', 'none', 'False'):
        out = _run(DRIVER, HONOURED + ['--interpolate-time', value])
        assert _stencil_banner(out) == 'nearest', (
            "--interpolate-time %r no longer resolves to 'nearest'; that is the only way to "
            "reproduce a pre-2026-09-02 run: %s" % (value, out[-1500:]))


def test_legacy_truthy_still_means_cubic():
    out = _run(DRIVER, HONOURED + ['--interpolate-time', 'True'])
    assert _stencil_banner(out) == 'cubic', (
        "the legacy truthy spelling must still map to 'cubic', not to the new default: %s"
        % out[-1500:])


def test_a_typo_is_still_loud():
    out = _squash(_run(DRIVER, HONOURED + ['--interpolate-time', 'lanczos']))
    assert 'unrecognised value' in out, (
        "a misspelled stencil was absorbed. Before this check it fell through to 'nearest'; "
        "after the default change it would fall through to the new default, which is a different "
        "wrong answer but still a silent one: %s" % out[-1500:])


def test_pipeline_off_request_still_means_off():
    """"off" must still mean off, now that emitting nothing means the new default.

    resolve_interpolate_time_request collapses "flag absent" and an explicit off-request to the
    same None.  While the driver default was 'nearest' those were the same answer; they are now
    opposites, so helper_LDG_Events has to re-express an off-request as an explicit
    '--interpolate-time nearest' rather than emitting nothing.
    """
    # --fmin is needed only to get the helper PAST its PSD/parameter setup and as far as the
    # stencil log line; it then dies on missing frames, which _run tolerates.
    out = _squash(_run(HELPER, ['--internal-ile-interpolate-time', 'False',
                                '--event-time', '1000000000', '--fmin', '20'],
                       in_tmpdir=True))
    assert '--interpolate-time nearest' in out or "stencil 'nearest'" in out, (
        "helper_LDG_Events did not turn an explicit off-request into an explicit "
        "'--interpolate-time nearest'; omitting the flag now means the NEW default, so 'off' "
        "would silently mean 'on': %s" % out[-2000:])


def test_pseudo_pipe_forwards_an_off_request_instead_of_swallowing_it():
    """The other half of the off-request repair, and the half a subprocess test cannot reach.

    util_RIFT_pseudo_pipe forwards --internal-ile-interpolate-time to the helper only when
    resolve_interpolate_time_request returns non-None -- which it does NOT for an off-request. So
    the helper's repair above is unreachable through the pipeline entry point unless this
    condition also stops swallowing it. Pinned in the source because driving the pseudo pipe far
    enough to emit a helper command line needs a whole event configuration; the condition is one
    line and this is the cheap check that it is still there.
    """
    with open(PSEUDO) as handle:
        source = handle.read()
    assert "or opts.internal_ile_interpolate_time is not None" in source, (
        "util_RIFT_pseudo_pipe no longer forwards an explicit off-request to the helper. Since "
        "the ILE default stopped being 'nearest', dropping the flag means the NEW default, so "
        "'--internal-ile-interpolate-time False' would silently turn interpolation ON.")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print("PASS %s" % name)
