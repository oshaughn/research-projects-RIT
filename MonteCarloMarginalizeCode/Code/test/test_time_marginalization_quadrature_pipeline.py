#!/usr/bin/env python
"""Pipeline passthrough for --time-marginalization-quadrature.

Companion to test_time_marginalization_quadrature.py, which covers the quadrature
itself.  This file covers the WIRING: that a campaign can select the quadrature
without --manual-extra-ile-args, that the selection reaches every ILE submit file
INCLUDING ILE_extr.sub, that the default path emits nothing, and that a
configuration which cannot honour the request is REFUSED at DAG-build time rather
than at first-job time.

Why the wiring needs its own tests.  The option is inert unless it survives four
hand-offs -- util_RIFT_pseudo_pipe.py -> helper_LDG_Events.py -> args_ile.txt ->
create_event_parameter_pipeline_BasicIteration -> ILE*.sub -- and the last of
those is an INHERITANCE (`ile_args_extr = ile_args + ...`), not an explicit
forward, so it is exactly the kind of link that a refactor breaks silently.  A
test of the quadrature helper alone would stay green through all of it.
"""
import ast
import gzip
import os
import subprocess
import sys

import pytest

from RIFT.likelihood.time_marginalization_quadrature import (
    TIME_QUADRATURE_CHOICES, time_quadrature_pipeline_prereqs,
    find_time_quadrature_in_ile_args, refuse_unhonourable_time_quadrature,
    refuse_unless_time_quadrature_emitted)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "MonteCarloMarginalizeCode", "Code")
BIN_DIR = os.path.join(CODE_DIR, "bin")
PSEUDO_PIPE = os.path.join(BIN_DIR, "util_RIFT_pseudo_pipe.py")
HELPER = os.path.join(BIN_DIR, "helper_LDG_Events.py")
CEPP = os.path.join(BIN_DIR, "create_event_parameter_pipeline_BasicIteration")
ILE_EXE = os.path.join(BIN_DIR, "integrate_likelihood_extrinsic_batchmode")

GOOD_ILE_ARGS = ("integrate_likelihood_extrinsic_batchmode --time-marginalization "
                 "--vectorized --gpu --srate 4096 --n-eff 50")


# ---------------------------------------------------------------- prerequisites

def test_simpson_is_never_refused():
    """The default must never be able to fail a workflow build.  Even a
    configuration that excludes 'bandlimited' entirely is fine for 'simpson',
    because 'simpson' is what ILE does anyway."""
    assert time_quadrature_pipeline_prereqs('simpson', "--rotation-slow --freqresponse") == []
    assert time_quadrature_pipeline_prereqs('simpson', "") == []


def test_honourable_configuration_passes():
    assert time_quadrature_pipeline_prereqs('bandlimited', GOOD_ILE_ARGS) == []
    # optparse has no six-character floor: these are the shortest unique
    # spellings in ILE's current option set.
    assert time_quadrature_pipeline_prereqs(
        'bandlimited', "X --time-marginalization --vec --g") == []


@pytest.mark.parametrize("flag", ["--time-marginalization", "--vectorized", "--gpu"])
def test_each_required_flag_is_reported_when_missing(flag):
    args = " ".join(t for t in GOOD_ILE_ARGS.split() if t != flag)
    missing = time_quadrature_pipeline_prereqs('bandlimited', args)
    assert any(flag in m for m in missing), missing


@pytest.mark.parametrize("flag,value", [
    ("--rotation-slow", ""),
    ("--freqresponse", ""),
    ("--calibration-envelope-directory", " /tmp/cal"),
])
def test_each_excluding_flag_is_reported_when_present(flag, value):
    missing = time_quadrature_pipeline_prereqs('bandlimited', GOOD_ILE_ARGS + " " + flag + value)
    assert any(flag in m for m in missing), missing


@pytest.mark.parametrize("spelling", [
    "--calibration-envelope-directory=/tmp/cal",   # optparse accepts the equals form
    "--rotation-s",                                # shortest unique prefix today
    "--calibration-en=/tmp/cal",                   # shortest unique prefix today
    "'--rotation-slow'",                           # an ini leaves the quotes on
])
def test_legal_optparse_spellings_do_not_evade_the_exclusions(spelling):
    """Three spellings that set the excluded option and that a naive whitespace
    split does not see.  Each costs a queue-slot cycle if it reaches the driver,
    which is the cost this DAG-build guard exists to avoid."""
    missing = time_quadrature_pipeline_prereqs('bandlimited', GOOD_ILE_ARGS + " " + spelling)
    assert missing, spelling


@pytest.mark.parametrize("innocent", [
    "--no-gpu", "--gpu-fanout 2", "--rotation-slow-foo", "--calibration-n-realizations 100",
])
def test_exclusions_do_not_fire_on_lookalike_flags(innocent):
    """The exclusion side must be a token/abbreviation match, not a substring one:
    a substring test would falsely refuse any future option whose name contains
    one of these.  This is the half of the matching rule that had no test."""
    assert time_quadrature_pipeline_prereqs('bandlimited', GOOD_ILE_ARGS + " " + innocent) == []


def test_match_is_by_token_not_substring():
    """'--no-gpu' must not satisfy '--gpu', and the quadrature flag itself must not
    satisfy '--time-marginalization'.  A substring test passes both and would
    declare an unhonourable configuration fine."""
    args = ("integrate_likelihood_extrinsic_batchmode --time-marginalization-quadrature "
            "bandlimited --vectorized --no-gpu")
    missing = time_quadrature_pipeline_prereqs('bandlimited', args)
    assert any("--gpu" in m and "missing" in m for m in missing), missing
    assert any("--time-marginalization " in m or m.startswith("missing --time-marginalization (")
               for m in missing), missing


def test_bad_value_is_refused():
    with pytest.raises(ValueError):
        time_quadrature_pipeline_prereqs('bandlimted', GOOD_ILE_ARGS)
    with pytest.raises(ValueError):
        time_quadrature_pipeline_prereqs('True', GOOD_ILE_ARGS)


# ------------------------------------------------------------- static wiring

def _source(path):
    with open(path) as f:
        return f.read()


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_option_is_defined_with_a_none_default(path):
    """Default None means "pass nothing", so the default workflow is byte-identical
    to one built before this option existed."""
    src = _source(path)
    assert '"--internal-ile-time-marginalization-quadrature"' in src
    line = [l for l in src.splitlines()
            if '"--internal-ile-time-marginalization-quadrature"' in l][0]
    assert "default=None" in line, line
    assert "type=str" in line, line


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_choices_are_imported_not_retyped(path):
    """A second hand-typed copy of the choice tuple is how a typo becomes a
    silently different likelihood: the pipeline would accept it, forward it, and
    the mistake would surface only when the first ILE job died."""
    src = _source(path)
    assert "TIME_QUADRATURE_CHOICES" in src
    for literal in ("('simpson', 'bandlimited')", '("simpson", "bandlimited")',
                    "'simpson','bandlimited'", '"simpson","bandlimited"'):
        assert literal not in src, "choice tuple re-typed in %s: %r" % (path, literal)


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_ini_override_is_recorded_in_the_help(path):
    """The RIFT ini parser OVERRIDES the command line for non-boolean options, and
    this is a string option, so an ini that also sets it wins silently."""
    line = [l for l in _source(path).splitlines()
            if '"--internal-ile-time-marginalization-quadrature"' in l][0]
    assert "ini" in line.lower() and "override" in line.lower(), line


def _assign_targets_containing(path, needle):
    """Names assigned (`=` or `+=`) a string containing `needle`.  Asserting the
    TARGET, not just the presence of the literal, is what catches the refactor that
    appends the flag to a variable nothing writes out.  Both assignment forms are
    accepted: the helper's emission is a plain `=` with an explicit rstrip(), so a
    test that looked only at `+=` broke on that change -- and, because the mutation
    harness runs this same file, silently turned every mutation into a false kill."""
    tree = ast.parse(_source(path), filename=path)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not names:
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and needle in sub.value:
                out.update(names)
    return out


def test_pseudo_pipe_forwards_to_the_helper():
    """It must land on the helper_LDG_Events.py command line -- the helper owns ILE
    argument construction, so a flag appended anywhere else never reaches ILE."""
    targets = _assign_targets_containing(
        PSEUDO_PIPE, "--internal-ile-time-marginalization-quadrature")
    assert "cmd" in targets, targets


def test_helper_emits_the_ile_flag():
    """It must land on helper_ile_args, which is what becomes args_ile.txt."""
    targets = _assign_targets_containing(HELPER, "--time-marginalization-quadrature")
    assert "helper_ile_args" in targets, targets


def _dead_nodes(tree):
    """Every node inside an `if <constant falsy>:` body -- i.e. unreachable code."""
    dead = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant) \
                and not node.test.value:
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    dead.add(sub)
    return dead


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_the_refusal_call_site_is_reachable(path):
    """util_RIFT_pseudo_pipe.py cannot be run to completion in a test -- its late
    guard sits after an os.system() helper call that needs real data -- so that one
    call site has no executable coverage and a plain "is it called" ast walk passes
    even when the call is wrapped in `if False:`.  This asserts the call is in LIVE
    code.  The helper's guard is covered executably as well, below."""
    tree = ast.parse(_source(path), filename=path)
    dead = _dead_nodes(tree)
    live = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "refuse_unless_time_quadrature_emitted" and n not in dead]
    assert live, "no reachable call to refuse_unless_time_quadrature_emitted in %s" % path


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_both_layers_call_the_refusal(path):
    """Structural companion to the EXECUTED refusal tests below.  On its own this
    is presence, not effect -- a guard turned into a print still passes it -- which
    is why the executed tests exist; it is kept only to catch the call site being
    deleted outright, which is cheaper to diagnose here."""
    tree = ast.parse(_source(path), filename=path)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "refuse_unless_time_quadrature_emitted" in called, called


def test_choices_argparse_surface_matches_the_library():
    """--help must offer exactly the library's choices, not a subset frozen at the
    time this option was written."""
    env = dict(os.environ, PYTHONPATH=CODE_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, PSEUDO_PIPE, "--help"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True).stdout
    assert "--internal-ile-time-marginalization-quadrature" in out
    for choice in TIME_QUADRATURE_CHOICES:
        assert choice in out


def test_a_bad_value_is_rejected_by_the_pipeline_command_line():
    env = dict(os.environ, PYTHONPATH=CODE_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, PSEUDO_PIPE, "--internal-ile-time-marginalization-quadrature", "bandlimted"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert proc.returncode != 0
    assert "bandlimted" in proc.stdout


# --------------------------------------------------- end-to-end: args_ile -> .sub

def _write_grid(tmp_path):
    import lal
    import RIFT.lalsimutils as lsu
    P = lsu.ChooseWaveformParams()
    P.m1 = 35 * lal.MSUN_SI
    P.m2 = 30 * lal.MSUN_SI
    here = os.getcwd()
    os.chdir(os.fspath(tmp_path))
    try:
        lsu.ChooseWaveformParams_array_to_xml([P, P], "proposed-grid")
    finally:
        os.chdir(here)
    return os.fspath(tmp_path / "proposed-grid.xml.gz")


def _build_dag(tmp_path, ile_args_line, extra_cepp=()):
    """Run the real DAG builder on a hand-written args_ile.txt.  Returns the
    working directory holding the generated .sub files."""
    (tmp_path / "args_ile.txt").write_text(ile_args_line + "\n")
    (tmp_path / "args_cip_list.txt").write_text(
        "2 --parameter mc --parameter delta_mc --n-output-samples 5000\n")
    (tmp_path / "args_test.txt").write_text("X --always-succeed\n")
    grid = _write_grid(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = BIN_DIR + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [sys.executable, CEPP,
         "--ile-n-events-to-analyze", "1",
         "--input-grid", grid,
         "--ile-exe", ILE_EXE,
         "--ile-args", os.fspath(tmp_path / "args_ile.txt"),
         "--cip-args-list", os.fspath(tmp_path / "args_cip_list.txt"),
         "--test-args", os.fspath(tmp_path / "args_test.txt"),
         "--working-directory", os.fspath(tmp_path),
         "--n-iterations", "2",
         "--n-samples-per-job", "500",
         "--last-iteration-extrinsic",
         "--last-iteration-extrinsic-samples-per-ile", "200"] + list(extra_cepp),
        cwd=os.fspath(tmp_path), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert proc.returncode == 0, proc.stdout[-4000:]
    return tmp_path


def test_quadrature_reaches_both_ile_and_ile_extr_sub(tmp_path):
    """The load-bearing one.  ILE_extr.sub is built by INHERITING the whole
    main-iteration argument string (`ile_args_extr = ile_args + ...`), not by an
    explicit forward -- so the extrinsic stage silently keeping the historical
    quadrature is a live failure mode and this is what rules it out."""
    wd = _build_dag(tmp_path, GOOD_ILE_ARGS + " --time-marginalization-quadrature bandlimited")
    for name in ("ILE.sub", "ILE_extr.sub"):
        text = (wd / name).read_text()
        assert "--time-marginalization-quadrature bandlimited" in text, name


def test_default_path_emits_no_quadrature_flag(tmp_path):
    """With the pipeline option unset the helper writes nothing, so no ILE submit
    file mentions the quadrature at all and the default run is unchanged."""
    wd = _build_dag(tmp_path, GOOD_ILE_ARGS)
    for name in ("ILE.sub", "ILE_extr.sub"):
        assert "--time-marginalization-quadrature" not in (wd / name).read_text(), name


# ------------------------------------------------- the guard reads the BYTES

def test_prereq_check_alone_approves_args_that_never_got_the_flag():
    """Documents WHY refuse_unless_time_quadrature_emitted exists.  The prerequisite
    check reads the prerequisites from the argument string but the INTENT from the
    caller, so on its own it approves an args_ile.txt that never received the flag."""
    assert time_quadrature_pipeline_prereqs('bandlimited', GOOD_ILE_ARGS) == []


def test_emission_guard_refuses_args_that_never_got_the_flag():
    """The stale-helper_ile_args.txt / dropped-emission case: prerequisites all
    satisfied, request recorded, flag absent -> every ILE job would run Simpson."""
    with pytest.raises(ValueError) as e:
        refuse_unless_time_quadrature_emitted('bandlimited', GOOD_ILE_ARGS, "args_ile.txt")
    assert "contains no --time-marginalization-quadrature" in str(e.value)


def test_emission_guard_refuses_a_duplicate_because_optparse_takes_the_last():
    """--manual-extra-ile-args is appended AFTER the helper's arguments, so a
    hand-passed 'simpson' silently wins while the .sub file shows both -- which is
    exactly the case that falsifies "readable off the .sub file"."""
    args = (GOOD_ILE_ARGS + " --time-marginalization-quadrature bandlimited"
            " --time-marginalization-quadrature simpson")
    with pytest.raises(ValueError) as e:
        refuse_unless_time_quadrature_emitted('bandlimited', args, "args_ile.txt")
    assert "occurrences" in str(e.value)


def test_emission_guard_refuses_a_value_that_does_not_match_the_request():
    args = GOOD_ILE_ARGS + " --time-marginalization-quadrature simpson"
    with pytest.raises(ValueError):
        refuse_unless_time_quadrature_emitted('bandlimited', args, "args_ile.txt")


def test_emission_guard_holds_a_hand_passed_quadrature_to_the_same_standard():
    """The manual route (--manual-extra-ile-args / an ini) got no protection at all
    while the guard was keyed on the pipeline option being set."""
    args = ("X --time-marginalization --vectorized --gpu --rotation-slow"
            " --time-marginalization-quadrature bandlimited")
    with pytest.raises(ValueError) as e:
        refuse_unless_time_quadrature_emitted(None, args, "args_ile.txt")
    assert "--rotation-slow" in str(e.value)


def test_emission_guard_is_silent_on_the_default_path():
    """Nothing requested, nothing present: the default workflow must not raise."""
    refuse_unless_time_quadrature_emitted(None, GOOD_ILE_ARGS, "args_ile.txt")


def test_emission_guard_accepts_the_honoured_case():
    refuse_unless_time_quadrature_emitted(
        'bandlimited', GOOD_ILE_ARGS + " --time-marginalization-quadrature bandlimited",
        "args_ile.txt")


def test_find_handles_the_equals_form():
    assert find_time_quadrature_in_ile_args(
        "X --time-marginalization-quadrature=bandlimited") == ['bandlimited']
    # ILE uses optparse, which accepts these unique-prefix forms.  The guard must
    # see the same option or a manual abbreviation bypasses all prerequisites.
    assert find_time_quadrature_in_ile_args(
        "X --time-marginalization-q=bandlimited") == ['bandlimited']
    assert find_time_quadrature_in_ile_args(
        "X --time-marginalization- bandlimited") == ['bandlimited']
    # The exact boolean flag wins as an exact optparse match; it is not an
    # abbreviation of the quadrature option.
    assert find_time_quadrature_in_ile_args(
        "X --time-marginalization --vectorized --gpu") == []


def test_abbreviated_hand_passed_quadrature_cannot_evade_prerequisites():
    args = ("X --time-marginalization --vectorized --rotation-slow "
            "--time-marginalization-q=bandlimited")
    with pytest.raises(ValueError) as e:
        refuse_unless_time_quadrature_emitted(None, args, "args_ile.txt")
    assert "--gpu" in str(e.value)
    assert "--rotation-slow" in str(e.value)


def test_refusal_actually_raises():
    """Executable coverage of the raise itself.  It lives in the library precisely
    so that turning it into a print is a code change a test can see -- an ast walk
    for a call by name cannot."""
    with pytest.raises(ValueError):
        refuse_unhonourable_time_quadrature('bandlimited', "X --vectorized", "somewhere")
    refuse_unhonourable_time_quadrature('bandlimited', GOOD_ILE_ARGS, "somewhere")
    refuse_unhonourable_time_quadrature('simpson', "X --rotation-slow", "somewhere")


# ------------------------------------ executable: the real scripts, real bytes
#
# The four hops are pseudo_pipe -> helper -> args_ile.txt -> ILE*.sub.  The DAG
# tests above cover the last hop only, because they hand-write args_ile.txt.  These
# run the real scripts.  helper_LDG_Events.py needs no data if given --fake-data
# and a manual IFO list, which is what makes hops 1->2 and 2->3 testable at all.

HELPER_BASE = [
    "--event-time", "1240000000", "--fmin", "20", "--fmin-template", "20",
    "--manual-ifo-list", "['H1','L1']", "--fake-data", "--assume-fiducial-psd-files",
    "--data-start-time", "1239999996", "--data-end-time", "1240000004",
    "--force-notune-initial-grid", "--propose-fit-strategy",
]


def _run_helper(tmp_path, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = BIN_DIR + os.pathsep + env.get("PATH", "")
    cmd = [sys.executable, HELPER, "--working-directory", os.fspath(tmp_path)] \
        + HELPER_BASE + list(extra)
    return subprocess.run(cmd, cwd=os.fspath(tmp_path), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)


def test_helper_emits_the_requested_value_not_a_hardcoded_one(tmp_path):
    """Hop 2->3, executed.  The static test asserts only that the flag NAME is
    appended somewhere; a helper that emitted a hardcoded 'simpson' regardless of
    the request passed it.  That is the exact failure this whole effort is about:
    the user asks for bandlimited, the pipeline prints bandlimited, the .sub says
    simpson."""
    proc = _run_helper(tmp_path, "--propose-ile-convergence-options",
                       "--internal-ile-time-marginalization-quadrature", "bandlimited")
    assert proc.returncode == 0, proc.stdout[-3000:]
    args = (tmp_path / "helper_ile_args.txt").read_text()
    assert find_time_quadrature_in_ile_args(args) == ["bandlimited"], args[-400:]
    # and the emission must not have been concatenated onto its neighbour
    assert " --time-marginalization-quadrature bandlimited " in args + " "


def test_helper_default_emits_nothing(tmp_path):
    proc = _run_helper(tmp_path, "--propose-ile-convergence-options")
    assert proc.returncode == 0, proc.stdout[-3000:]
    args = (tmp_path / "helper_ile_args.txt").read_text()
    assert find_time_quadrature_in_ile_args(args) == []


def test_helper_refuses_a_configuration_it_cannot_honour(tmp_path):
    """Executed refusal.  Without --propose-ile-convergence-options the helper never
    adds --time-marginalization/--vectorized/--gpu, so the request cannot be
    honoured.  A guard turned into a print, or disabled with `if False`, passes the
    ast test and fails this one."""
    proc = _run_helper(tmp_path,
                       "--internal-ile-time-marginalization-quadrature", "bandlimited")
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert "--time-marginalization" in proc.stdout
    assert not (tmp_path / "helper_ile_args.txt").exists()


def test_helper_warns_that_the_extrinsic_t_ref_is_not_refined(tmp_path):
    """The PR offers "it reaches ILE_extr.sub" as the assurance for the extrinsic
    stage, but --resample-time-marginalization asks for lnL(t) on the ORIGINAL grid
    (return_lnLt), which never reaches this quadrature.  Say so at build time."""
    proc = _run_helper(tmp_path, "--propose-ile-convergence-options",
                       "--internal-ile-time-marginalization-quadrature", "bandlimited")
    assert proc.returncode == 0
    assert "t_ref is still quantised" in proc.stdout


def _run_pseudo_pipe(tmp_path, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    # deliberately WITHOUT BIN_DIR on PATH for the forward test: the helper is
    # invoked by name, so it fails, and we read the command line it printed.
    cmd = [sys.executable, PSEUDO_PIPE, "--approx", "SEOBNRv4",
           "--use-rundir", os.fspath(tmp_path / "run")] + list(extra)
    return subprocess.run(cmd, cwd=os.fspath(tmp_path), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)


def test_pseudo_pipe_forwards_the_requested_value_to_the_helper(tmp_path):
    """Hop 1->2, executed.  pseudo_pipe prints the helper command line it is about
    to run; a forward that hardcoded 'simpson' passed the static test."""
    proc = _run_pseudo_pipe(
        tmp_path, "--internal-ile-time-marginalization-quadrature", "bandlimited")
    assert "--internal-ile-time-marginalization-quadrature bandlimited" in proc.stdout, \
        proc.stdout[-3000:]


def test_pseudo_pipe_checks_the_helper_exit_status():
    """A same-value stale helper_ile_args.txt can satisfy the byte guard.  The
    helper's status therefore has to be checked independently, before that file
    is read."""
    src = _source(PSEUDO_PIPE)
    call = src.index("_helper_rc = os.system(cmd)")
    check = src.index("if _helper_rc != 0:", call)
    read = src.index('np.loadtxt("helper_ile_args.txt"', check)
    assert call < check < read
    assert "os.unlink('helper_ile_args.txt')" in src[:call]


def test_pseudo_pipe_refuses_calmarg_before_it_runs_anything(tmp_path):
    """Executed refusal, and it must fire EARLY -- calibration marginalization is
    added by this script, not by the helper, so the helper can never see it."""
    proc = _run_pseudo_pipe(
        tmp_path, "--internal-ile-time-marginalization-quadrature", "bandlimited",
        "--calmarg-envelope-directory", os.fspath(tmp_path))
    assert proc.returncode != 0
    assert "--calibration-envelope-directory" in proc.stdout
    assert "helper_LDG_Events.py --force-notune" not in proc.stdout, \
        "refusal must precede the helper invocation"


def test_pseudo_pipe_refuses_an_excluded_manual_extra_ile_arg(tmp_path):
    proc = _run_pseudo_pipe(
        tmp_path, "--internal-ile-time-marginalization-quadrature", "bandlimited",
        "--manual-extra-ile-args=--rotation-slow")
    assert proc.returncode != 0
    assert "--rotation-slow" in proc.stdout


def test_pseudo_pipe_refuses_on_the_lisa_known_sky_path(tmp_path):
    """--lisa-known-sky exits before both the forward and the args_ile.txt guard and
    builds its own ILE arguments through helper_LISA_Events.py, which does not carry
    the option -- so without this the request is silently dropped."""
    proc = _run_pseudo_pipe(
        tmp_path, "--internal-ile-time-marginalization-quadrature", "bandlimited",
        "--lisa-known-sky", "--event-time", "1234.5",
        "--ecliptic-longitude", "1.25", "--ecliptic-latitude", "-0.4")
    assert proc.returncode != 0
    assert "lisa-known-sky" in proc.stdout


def test_pseudo_pipe_rejects_a_bad_value_after_the_ini_block(tmp_path):
    """The validate must run AFTER --use-ini, which overrides the command line for
    non-boolean options; above it, it checks a value the ini is about to replace."""
    src = _source(PSEUDO_PIPE)
    i_ini = src.index("if (opts.use_ini):")
    i_val = src.index("validate_time_quadrature(opts.internal_ile_time_marginalization_quadrature)")
    assert i_val > i_ini, "validate_time_quadrature runs before the ini block can override it"


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_argparse_choices_are_pinned_to_the_library_tuple(path):
    """A grep for the identifier survives deleting `choices=list(...)`, because the
    help string still interpolates the tuple."""
    line = [l for l in _source(path).splitlines()
            if '"--internal-ile-time-marginalization-quadrature"' in l][0]
    assert "choices=list(TIME_QUADRATURE_CHOICES)" in line, line


def test_quadrature_also_reaches_puff_and_fetch_subs(tmp_path):
    """ILE_puff.sub and ILE_fetch.sub inherit the same argument string
    (ile_args_forpuff / ile_args_forfetch = ile_args_orig + ...).  Puffball ILE is
    standard in production, so pin it rather than relying on it happening to work."""
    (tmp_path / "args_puff.txt").write_text(
        "--parameter mc --parameter delta_mc --downselect-parameter m2 "
        "--downselect-parameter-range [1,1000]\n")
    wd = _build_dag(tmp_path, GOOD_ILE_ARGS + " --time-marginalization-quadrature bandlimited",
                    extra_cepp=["--puff-args", os.fspath(tmp_path / "args_puff.txt")])
    made = [n for n in ("ILE_puff.sub", "ILE_fetch.sub") if (wd / n).exists()]
    assert made, "neither puff nor fetch sub was generated; the inheritance claim is untested"
    for name in made:
        assert "--time-marginalization-quadrature bandlimited" in (wd / name).read_text(), name
