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
    TIME_QUADRATURE_CHOICES, time_quadrature_pipeline_prereqs)

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


def _augassign_targets_containing(path, needle):
    """Names that are `+=`-ed a string containing `needle`.  Asserting the TARGET,
    not just the presence of the literal, is what catches the refactor that appends
    the flag to a variable nothing writes out."""
    tree = ast.parse(_source(path), filename=path)
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AugAssign) or not isinstance(node.target, ast.Name):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and needle in sub.value:
                out.add(node.target.id)
    return out


def test_pseudo_pipe_forwards_to_the_helper():
    """It must land on the helper_LDG_Events.py command line -- the helper owns ILE
    argument construction, so a flag appended anywhere else never reaches ILE."""
    targets = _augassign_targets_containing(
        PSEUDO_PIPE, "--internal-ile-time-marginalization-quadrature")
    assert "cmd" in targets, targets


def test_helper_emits_the_ile_flag():
    """It must land on helper_ile_args, which is what becomes args_ile.txt."""
    targets = _augassign_targets_containing(HELPER, "--time-marginalization-quadrature")
    assert "helper_ile_args" in targets, targets


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_both_layers_call_the_refusal(path):
    """Refuse, never silently ignore.  The helper sees the ILE strategy flags; only
    util_RIFT_pseudo_pipe.py sees calibration marginalization and
    --manual-extra-ile-args, so both layers must check."""
    tree = ast.parse(_source(path), filename=path)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "time_quadrature_pipeline_prereqs" in called


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


def _build_dag(tmp_path, ile_args_line):
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
         "--last-iteration-extrinsic-samples-per-ile", "200"],
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
