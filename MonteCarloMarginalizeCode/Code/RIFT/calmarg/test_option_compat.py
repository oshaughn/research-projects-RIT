#!/usr/bin/env python3
"""Calibration option-compatibility gate: the refusals AND their nearest legal neighbours.

Every guard in RIFT/calmarg/option_compat.py is an OVER-BROAD-CONDITION risk: a rule that
refuses too much is invisible to a test that only checks the refusal fires.  So each
refusal here is paired with the closest configuration that must still be ACCEPTED -- the
one flag away.  Those accepting cases are the load-bearing half.

Three of them are not hypothetical.  They are configurations the SHIPPED pipeline emits:

  * --calibration-dump-responsibilities without --time-marginalization / without an xpy
    evaluator.  The cal pilot returns from inside the precompute block; refusing it on
    the production likelihood's prerequisites would break util_CalPilotStage.py.
  * --calibration-fused-kernel together with --calibration-dump-responsibilities.
    util_CalPilotStage.py inherits the WIDE args_ile.txt verbatim, so a run configured
    with --calmarg-fused-kernel gives its pilot that flag.  The pilot uses the loop
    reduction; the flag is inert, not wrong.
  * --calibration-export-posterior on the wide stage.  util_RIFT_pseudo_pipe.py emits it
    there deliberately, documented as harmless (it only fires at the fairdraw stage).

    python3 test_option_compat.py        # or: pytest test_option_compat.py
"""
from __future__ import print_function

import os
import re
import subprocess
import sys

import RIFT.calmarg.option_compat as oc

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
DRIVER = os.path.join(CODE_ROOT, 'bin', 'integrate_likelihood_extrinsic_batchmode')

ENV_DIR = '/tmp/cal_env_that_need_not_exist'


class _Opts(object):
    """A stand-in for the driver's optparse namespace, with the driver's own defaults."""

    def __init__(self, **kw):
        self.calibration_envelope_directory = None
        self.time_marginalization = False
        self.vectorized = False
        self.gpu = False
        self.rotation_slow = False
        self.freqresponse = False
        for _flag, attr in oc.CAL_OPT_IN_FLAGS:
            setattr(self, attr, None)
        self.calibration_fused_kernel = False
        self.calibration_conjugate_phase = False
        self.calibration_global_norm = False
        self.calibration_export_posterior = False
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError("no such driver option: %s" % k)
            setattr(self, k, v)


def _honoured(**kw):
    """The configuration in-loop calmarg actually runs on (the demo's own)."""
    base = dict(calibration_envelope_directory=ENV_DIR, time_marginalization=True,
                vectorized=True, gpu=True)
    base.update(kw)
    return _Opts(**base)


def _kinds(refusals):
    return sorted(set(r.kind for r in refusals))


def _flags(refusals):
    return set(f for r in refusals for f in r.options)


# ------------------------------------------------------------------ ACCEPTING EDGE

def test_the_honoured_configuration_is_accepted():
    """The configuration every calmarg run and the calmarg demo use.  If this ever
    starts failing, the gate has been tightened into breaking production."""
    assert oc.refusals_from_opts(_honoured()) == []


def test_every_calibration_opt_in_is_accepted_on_the_honoured_configuration():
    """Each opt-in, one at a time, on the good configuration.  A rule keyed on the flag
    rather than on the missing prerequisite would fail here.

    --calibration-burn-in-nmax is the one opt-in the envelope does not suffice for: it is
    a cap on a burn-in that only --calibration-burn-in-neff switches on, so it is carried
    with its dependency here and refused on its own below."""
    for flag, attr in oc.CAL_OPT_IN_FLAGS:
        value = 'x' if attr.endswith(('breadcrumb', 'responsibilities')) else True
        if attr.startswith('calibration_burn_in'):
            value = 10
        extra = {'calibration_burn_in_neff': 100.} if attr.endswith('burn_in_nmax') else {}
        opts = _honoured(**dict(extra, **{attr: value}))
        assert oc.refusals_from_opts(opts) == [], (flag, oc.refusals_from_opts(opts))


def test_calibration_off_never_refuses_anything():
    """No envelope directory and no opt-ins: every other flag combination must pass,
    including the two 3G paths, which are perfectly legal on their own."""
    for kw in (dict(), dict(rotation_slow=True, vectorized=True),
               dict(freqresponse=True, vectorized=True),
               dict(time_marginalization=True, vectorized=True, gpu=True)):
        assert oc.refusals_from_opts(_Opts(**kw)) == [], kw


def test_export_posterior_on_the_wide_stage_is_accepted():
    """util_RIFT_pseudo_pipe.py emits --calibration-export-posterior on the WIDE stage,
    where it is a documented no-op (it fires only at the fairdraw stage).  Refusing it
    would break every calmarg campaign built with --calmarg-export-posterior."""
    assert oc.refusals_from_opts(_honoured(calibration_export_posterior=True)) == []


# --------------------------------------------------------- R1/R2: the 3G likelihoods

def test_calmarg_with_rotation_slow_is_refused():
    r = oc.refusals_from_opts(_honoured(rotation_slow=True))
    assert len(r) == 1 and r[0].kind == oc.KIND_UNIMPLEMENTED
    assert r[0].options == ('--calibration-envelope-directory', '--rotation-slow')
    assert 'PrecomputeLikelihoodTermsWithRotation' in r[0].message
    assert 'NO test coverage against the third-generation machinery' in r[0].enable_requires


def test_calmarg_with_freqresponse_is_refused():
    r = oc.refusals_from_opts(_honoured(freqresponse=True))
    assert len(r) == 1 and r[0].kind == oc.KIND_UNIMPLEMENTED
    assert r[0].options == ('--calibration-envelope-directory', '--freqresponse')
    assert 'PrecomputeLikelihoodTermsFreqResponse' in r[0].message
    assert 'NO test coverage against the third-generation machinery' in r[0].enable_requires


def test_the_3g_refusals_do_not_depend_on_the_backend():
    """The refusals must fire on the CPU-vectorized path too, not only under --gpu.

    This is the regression the guards this gate replaced actually had.  They read
    `opts.gpu and getattr(opts, 'calibration_marginalization', False)` -- two defects in
    one line: the attribute does not exist (so they never fired at all), and even had it
    existed the `opts.gpu and` would have left the CPU-vectorized replacement likelihood
    unprotected.  Both --rotation-slow and --freqresponse are wired into the CPU
    branch as well as the xpy one, and neither PrecomputeLikelihoodTermsWithRotation nor
    PrecomputeLikelihoodTermsFreqResponse builds calibration cross terms on EITHER
    backend, so the backend has nothing to do with it.

    Asserted by MEMBERSHIP, not by `len(r) == 1`: at gpu=False the configuration also
    earns the separate --gpu refusal (in-loop calmarg needs the xpy evaluator), so a
    length assertion here would pass for the wrong reason.  Re-adding an `opts.gpu and`
    condition to either rule must fail this test.
    """
    for kw, flag in ((dict(rotation_slow=True), '--rotation-slow'),
                     (dict(freqresponse=True), '--freqresponse')):
        r = oc.refusals_from_opts(_honoured(gpu=False, **kw))
        match = [x for x in r
                 if x.options == ('--calibration-envelope-directory', flag)]
        assert len(match) == 1, (flag, r)
        assert match[0].kind == oc.KIND_UNIMPLEMENTED, (flag, match)


def test_the_3g_refusals_survive_the_pilot_exemption():
    """--calibration-dump-responsibilities exempts a configuration from the PRODUCTION
    likelihood's prerequisites.  It must NOT exempt it from the 3G refusals: the pilot
    evaluates the BASELINE packed arrays, so under --rotation-slow it would report
    calibration responsibilities for a likelihood the user did not ask for."""
    for kw in (dict(rotation_slow=True), dict(freqresponse=True)):
        r = oc.refusals_from_opts(_honoured(calibration_dump_responsibilities='d.npz', **kw))
        assert [x.kind for x in r] == [oc.KIND_UNIMPLEMENTED], (kw, r)


def test_3g_without_calibration_is_accepted():
    """The nearest legal neighbour of R1/R2: the same 3G run with no calibration."""
    assert oc.refusals_from_opts(_Opts(rotation_slow=True, vectorized=True)) == []
    assert oc.refusals_from_opts(_Opts(freqresponse=True, vectorized=True)) == []


# ------------------------------------------------- R3/R4/R5: the honoured-path prereqs

def test_calmarg_without_vectorized_is_refused():
    r = oc.refusals_from_opts(_honoured(vectorized=False))
    assert any(x.options == ('--calibration-envelope-directory', '--vectorized') for x in r), r
    assert _kinds(r) == [oc.KIND_INERT]


def test_calmarg_without_time_marginalization_is_refused():
    r = oc.refusals_from_opts(_honoured(time_marginalization=False))
    assert any(x.options == ('--calibration-envelope-directory', '--time-marginalization')
               for x in r), r


def test_calmarg_without_an_xpy_evaluator_is_refused():
    r = oc.refusals_from_opts(_honoured(gpu=False))
    assert any(x.options == ('--calibration-envelope-directory', '--gpu') for x in r), r
    assert '--force-xpy' in [x for x in r if x.options[-1] == '--gpu'][0].message


def test_all_missing_prerequisites_are_reported_at_once():
    """A gate that reports one missing flag per run costs a queue cycle per flag."""
    r = oc.refusals_from_opts(_Opts(calibration_envelope_directory=ENV_DIR))
    assert _flags(r) >= {'--vectorized', '--time-marginalization', '--gpu'}, r


# ---------------------------------------------------------- the pilot's own exemptions

def test_pilot_is_exempt_from_time_marginalization_and_the_xpy_evaluator():
    """The cal PILOT returns 0.0 from inside the `if opts.vectorized:` precompute block,
    before the driver selects a production likelihood_function at all.  Its prerequisites
    are genuinely different, and util_CalPilotStage.py depends on that."""
    opts = _Opts(calibration_envelope_directory=ENV_DIR, vectorized=True,
                 calibration_dump_responsibilities='resp.npz')
    assert oc.refusals_from_opts(opts) == [], oc.refusals_from_opts(opts)


def test_pilot_is_NOT_exempt_from_vectorized():
    """The pilot block lives inside `if opts.vectorized:`; without it the pilot never
    runs and never writes its dump, and util_CalPilotFit.py then fails on a missing
    file.  The exemption must be scoped to the two prerequisites it actually earns."""
    opts = _Opts(calibration_envelope_directory=ENV_DIR,
                 calibration_dump_responsibilities='resp.npz')
    r = oc.refusals_from_opts(opts)
    assert any(x.options == ('--calibration-envelope-directory', '--vectorized') for x in r), r


def test_pilot_with_the_fused_kernel_is_accepted_with_a_notice():
    """util_CalPilotStage.py inherits the wide args verbatim, so a --calmarg-fused-kernel
    campaign hands its pilot --calibration-fused-kernel.  The pilot uses cal_method='loop'
    and returns early: the flag is inert here, which is a NOTICE, not a refusal."""
    opts = _honoured(calibration_dump_responsibilities='resp.npz',
                     calibration_fused_kernel=True)
    assert oc.refusals_from_opts(opts) == []
    notices = oc.notices_from_opts(opts)
    assert len(notices) == 1 and '--calibration-fused-kernel' in notices[0]
    # ... and no notice when the pilot is not involved
    assert oc.notices_from_opts(_honoured(calibration_fused_kernel=True)) == []


# --------------------------------------------- R7: the burn-in cap needs the burn-in

def test_burn_in_nmax_without_neff_is_refused():
    """The envelope is NOT enough to make --calibration-burn-in-nmax live.  The driver
    reads opts.calibration_burn_in_nmax only inside the `if opts.calibration_burn_in_neff`
    branch, so on the otherwise-honoured configuration the cap is silently ignored and the
    burn-in it was sizing never happens: the exact failure mode this gate exists for."""
    r = oc.refusals_from_opts(_honoured(calibration_burn_in_nmax=4000))
    assert len(r) == 1 and r[0].kind == oc.KIND_INERT, r
    assert r[0].options == ('--calibration-burn-in-nmax', '--calibration-burn-in-neff'), r


def test_burn_in_nmax_with_neff_is_accepted():
    """The nearest legal neighbours: the cap together with the option that reads it, and
    the burn-in target on its own (no cap needed -- it falls back to the run's --n-max)."""
    assert oc.refusals_from_opts(
        _honoured(calibration_burn_in_neff=100., calibration_burn_in_nmax=4000)) == []
    assert oc.refusals_from_opts(_honoured(calibration_burn_in_neff=100.)) == []


# ------------------------------------------------------ R6: opt-ins without the envelope

def test_each_opt_in_without_the_envelope_directory_is_refused():
    for flag, attr in oc.CAL_OPT_IN_FLAGS:
        value = 'x' if attr.endswith(('breadcrumb', 'responsibilities')) else True
        if attr.startswith('calibration_burn_in'):
            value = 10
        opts = _Opts(time_marginalization=True, vectorized=True, gpu=True, **{attr: value})
        r = oc.refusals_from_opts(opts)
        assert len(r) == 1 and r[0].kind == oc.KIND_INERT, (flag, r)
        assert r[0].options == (flag, '--calibration-envelope-directory'), (flag, r)


def test_numeric_default_options_are_not_treated_as_opt_ins():
    """--calibration-n-realizations and friends have non-None DEFAULTS, so a rule over
    them would refuse every run that merely left the defaults alone.  They are
    deliberately absent from CAL_OPT_IN_FLAGS; this pins that."""
    attrs = set(a for _f, a in oc.CAL_OPT_IN_FLAGS)
    for forbidden in ('calibration_n_realizations', 'calibration_spline_count',
                      'calibration_pilot_extrinsic', 'calibration_mc_error_extrinsic',
                      'calibration_neff_cal_target', 'calibration_n_realizations_max'):
        assert forbidden not in attrs, forbidden


def test_every_opt_in_flag_exists_in_the_driver():
    """CAL_OPT_IN_FLAGS is a hand-maintained list of CLI spellings.  A renamed or removed
    option would leave a rule that can never fire -- an inert guard."""
    src = open(DRIVER).read()
    for flag, attr in oc.CAL_OPT_IN_FLAGS:
        assert '"%s"' % flag in src, "%s is not an option of the driver" % flag
        assert re.search(r'\bopts\.%s\b' % attr, src), \
            "%s is never read by the driver" % attr


# ------------------------------------------------------------------- the wiring itself

def _driver_code_lines():
    """The driver with comment-only lines removed.

    Needed because the replaced guard is QUOTED in the comment that explains why it went,
    and a naive substring test over the whole file would match that comment -- i.e. it
    would pass only while nobody documented the change, and fail the moment somebody did.
    """
    out = []
    for line in open(DRIVER):
        if line.lstrip().startswith('#'):
            continue
        out.append(line)
    return ''.join(out)


def test_the_inert_getattr_guards_are_gone():
    """The two guards this gate replaces read
    `getattr(opts, 'calibration_marginalization', False)`.  There is no such option and
    nothing sets that attribute, so both were ALWAYS FALSE.  Re-adding one would restore
    a guard that looks like coverage and is not."""
    code = _driver_code_lines()
    assert "getattr(opts, 'calibration_marginalization'" not in code
    assert 'getattr(opts, "calibration_marginalization"' not in code
    # ... and the pointer to the replacement must survive, or the next reader re-adds the
    # guard where it used to be rather than extending the gate.
    src = open(DRIVER).read()
    assert src.count('RIFT/calmarg/option_compat.py') >= 2, \
        'the removed guards no longer point at the gate that replaced them'


def test_the_gate_runs_after_the_gpu_downgrade():
    """opts.gpu is silently downgraded to False when cupy is absent.  The gate reads it,
    so it must run AFTER that -- a call placed above the downgrade would accept exactly
    the CPU-node configuration that drops calibration on the floor."""
    src = open(DRIVER).read()
    downgrade = src.index('Override --gpu  (not available)')
    call = src.index('refuse_incompatible_calibration_options(opts)')
    precompute = src.index('factored_likelihood.PrecomputeLikelihoodTerms(')
    assert downgrade < call < precompute, (downgrade, call, precompute)


# ------------------------------------------------------- the CLI seam, in subprocesses

def _run(args, timeout=300):
    env = dict(os.environ)
    env['PYTHONPATH'] = CODE_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['OMP_NUM_THREADS'] = '1'
    env.setdefault('CUDA_VISIBLE_DEVICES', '')
    proc = subprocess.Popen([sys.executable, DRIVER] + args, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    if not isinstance(out, str):
        out = out.decode('utf-8', 'replace')
    return re.sub(r'\s+', ' ', out)


_HONOURED_CLI = ['--vectorized', '--gpu', '--force-xpy', '--time-marginalization']
_REFUSED = 'Refusing this calibration-marginalization configuration'


def test_driver_refuses_through_the_real_cli():
    """The predicate can be right and the wiring wrong; that is how the guards this
    replaces became inert.  Each case names the source evidence for its category."""
    cases = [
        (['--calibration-envelope-directory', ENV_DIR, '--rotation-slow'] + _HONOURED_CLI,
         '--rotation-slow', 'rotation dispatch takes no n_cal'),
        (['--calibration-envelope-directory', ENV_DIR, '--freqresponse'] + _HONOURED_CLI,
         '--freqresponse', 'freqresponse dispatch takes no n_cal'),
        (['--calibration-envelope-directory', ENV_DIR, '--vectorized', '--gpu', '--force-xpy'],
         '--time-marginalization', 'FactoredLogLikelihood takes no n_cal'),
        (['--calibration-envelope-directory', ENV_DIR, '--vectorized', '--time-marginalization'],
         '--force-xpy', 'ViaArrayVector (no NoLoop) takes no n_cal'),
        (['--calibration-fused-kernel'] + _HONOURED_CLI,
         '--calibration-envelope-directory', 'nothing to fuse'),
        (['--calibration-envelope-directory', ENV_DIR,
          '--calibration-burn-in-nmax', '4000'] + _HONOURED_CLI,
         '--calibration-burn-in-neff', 'no burn-in for the cap to cap'),
    ]
    for args, expect, why in cases:
        out = _run(args)
        assert _REFUSED in out and expect in out, \
            'driver accepted %s (%s); output tail: %s' % (args, why, out[-600:])
        print('driver refuses %-34s : OK' % why)


def test_driver_accepts_the_honoured_calmarg_configuration():
    """THE regression test for over-broadness at the CLI.  This is the calmarg demo's own
    BACKEND=cpu command line; it must get past the gate (and fail later, on the data it
    was not given)."""
    out = _run(['--calibration-envelope-directory', ENV_DIR] + _HONOURED_CLI)
    assert _REFUSED not in out, out[-600:]
    print('driver accepts the honoured calmarg configuration : OK')


def test_driver_accepts_the_burn_in_cap_with_its_target():
    """The over-broadness half of R7 at the CLI: the cap IS legal alongside the burn-in
    target, and a rule wired to the wrong attribute would refuse it here."""
    out = _run(['--calibration-envelope-directory', ENV_DIR,
                '--calibration-burn-in-neff', '100',
                '--calibration-burn-in-nmax', '4000'] + _HONOURED_CLI)
    assert _REFUSED not in out, out[-600:]
    print('driver accepts the burn-in cap with its target : OK')


def test_driver_accepts_the_pilot_without_time_marginalization():
    """The pilot's exemption, through the CLI.  util_CalPilotStage.py depends on it."""
    out = _run(['--calibration-envelope-directory', ENV_DIR, '--vectorized',
                '--calibration-dump-responsibilities', '/tmp/resp_unused.npz'])
    assert _REFUSED not in out, out[-600:]
    print('driver accepts the cal pilot on --vectorized alone : OK')


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except Exception as e:                       # noqa: BLE001
                fails += 1
                print('FAIL %s: %s' % (name, e))
            else:
                print('ok   %s' % name)
    print('\n%s' % ('FAILED' if fails else 'PASS'))
    sys.exit(1 if fails else 0)
