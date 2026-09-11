#!/usr/bin/env python
"""Pipeline passthrough for --internal-ile-q-time-pregrid-factor.

Companion to test_q_time_pregrid.py, which covers the library alone.  This file covers the
WIRING: that a campaign can select the certified Q_lm pregrid (PR #261) without
--manual-extra-ile-args, that the default path emits nothing, and that a configuration which
cannot honour the request -- including the forced-cubic-stencil conflict -- is REFUSED at
DAG-build time rather than at first-job time.

Modelled directly on test_time_marginalization_quadrature_pipeline.py: the option is inert
unless it survives util_RIFT_pseudo_pipe.py -> helper_LDG_Events.py -> helper_ile_args.txt /
args_ile.txt, so this exercises the real scripts rather than only the library.
"""
import ast
import os
import subprocess
import sys

import pytest

from RIFT.likelihood.q_time_pregrid import (
    Q_TIME_PREGRID_CHOICES, STENCIL_CONFLICT_MESSAGE,
    q_time_pregrid_pipeline_prereqs, find_q_time_pregrid_in_ile_args,
    refuse_unless_q_time_pregrid_emitted)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "MonteCarloMarginalizeCode", "Code")
BIN_DIR = os.path.join(CODE_DIR, "bin")
PSEUDO_PIPE = os.path.join(BIN_DIR, "util_RIFT_pseudo_pipe.py")
HELPER = os.path.join(BIN_DIR, "helper_LDG_Events.py")


def _source(path):
    with open(path) as f:
        return f.read()


# ------------------------------------------------------------- static wiring

@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_option_is_defined_with_a_none_default(path):
    """Default None means "pass nothing", so the default workflow is byte-identical to one
    built before this option existed."""
    src = _source(path)
    assert '"--internal-ile-q-time-pregrid-factor"' in src
    line = [l for l in src.splitlines()
            if '"--internal-ile-q-time-pregrid-factor"' in l][0]
    assert "default=None" in line, line
    assert "type=int" in line, line


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_choices_are_imported_not_retyped(path):
    """A second hand-typed copy of the choice tuple is how a typo becomes a silently
    different behaviour: the pipeline would accept it, forward it, and the mistake would
    surface only when the first ILE job died."""
    src = _source(path)
    assert "Q_TIME_PREGRID_CHOICES" in src
    for literal in ("(1, 8)", "[1, 8]", "1,8"):
        assert literal not in src, "choice tuple re-typed in %s: %r" % (path, literal)


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_ini_override_is_recorded_in_the_help(path):
    line = [l for l in _source(path).splitlines()
            if '"--internal-ile-q-time-pregrid-factor"' in l][0]
    assert "ini" in line.lower() and "override" in line.lower(), line


def _assign_targets_containing(path, needle):
    """Names assigned (`=` or `+=`) a string containing `needle`.  Asserting the TARGET, not
    just the presence of the literal, catches a refactor that appends the flag to a variable
    nothing writes out."""
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
    targets = _assign_targets_containing(PSEUDO_PIPE, "--internal-ile-q-time-pregrid-factor")
    assert "cmd" in targets, targets


def test_helper_emits_the_ile_flag():
    targets = _assign_targets_containing(HELPER, "--q-time-pregrid-factor")
    assert "helper_ile_args" in targets, targets


def _dead_nodes(tree):
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
    tree = ast.parse(_source(path), filename=path)
    dead = _dead_nodes(tree)
    live = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "refuse_unless_q_time_pregrid_emitted" and n not in dead]
    assert live, "no reachable call to refuse_unless_q_time_pregrid_emitted in %s" % path


def test_choices_argparse_surface_matches_the_library():
    env = dict(os.environ, PYTHONPATH=CODE_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, PSEUDO_PIPE, "--help"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True).stdout
    assert "--internal-ile-q-time-pregrid-factor" in out
    for choice in Q_TIME_PREGRID_CHOICES:
        assert str(choice) in out


def test_a_bad_value_is_rejected_by_the_pipeline_command_line():
    env = dict(os.environ, PYTHONPATH=CODE_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, PSEUDO_PIPE, "--internal-ile-q-time-pregrid-factor", "4"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert proc.returncode != 0


# ------------------------------------ executable: the real scripts, real bytes

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
    """Hop 2->3, executed.  The static test asserts only that the flag NAME is appended
    somewhere; a helper that emitted a hardcoded factor regardless of the request would pass
    it."""
    proc = _run_helper(tmp_path, "--propose-ile-convergence-options",
                       "--internal-ile-q-time-pregrid-factor", "8")
    assert proc.returncode == 0, proc.stdout[-3000:]
    args = (tmp_path / "helper_ile_args.txt").read_text()
    assert find_q_time_pregrid_in_ile_args(args) == ["8"], args[-400:]
    assert " --q-time-pregrid-factor 8 " in args + " "


def test_helper_default_emits_nothing(tmp_path):
    proc = _run_helper(tmp_path, "--propose-ile-convergence-options")
    assert proc.returncode == 0, proc.stdout[-3000:]
    args = (tmp_path / "helper_ile_args.txt").read_text()
    assert find_q_time_pregrid_in_ile_args(args) == []


def test_helper_refuses_a_configuration_it_cannot_honour(tmp_path):
    """Executed refusal.  Without --propose-ile-convergence-options the helper never adds
    --vectorized, so the request cannot be honoured.  A guard turned into a print, or
    disabled with `if False`, passes the ast test and fails this one."""
    proc = _run_helper(tmp_path, "--internal-ile-q-time-pregrid-factor", "8")
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert "--vectorized" in proc.stdout
    assert not (tmp_path / "helper_ile_args.txt").exists()


def test_helper_refuses_the_stencil_conflict(tmp_path):
    """Executed refusal of the forced-cubic-stencil conflict: an explicit non-cubic stencil
    plus factor 8 must fail at build time with the driver's own wording."""
    proc = _run_helper(tmp_path, "--propose-ile-convergence-options",
                       "--internal-ile-interpolate-time", "nearest",
                       "--internal-ile-q-time-pregrid-factor", "8")
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert STENCIL_CONFLICT_MESSAGE in proc.stdout
    assert not (tmp_path / "helper_ile_args.txt").exists()


def test_helper_accepts_an_explicit_cubic_stencil_with_factor_8(tmp_path):
    proc = _run_helper(tmp_path, "--propose-ile-convergence-options",
                       "--internal-ile-interpolate-time", "cubic",
                       "--internal-ile-q-time-pregrid-factor", "8")
    assert proc.returncode == 0, proc.stdout[-3000:]
    args = (tmp_path / "helper_ile_args.txt").read_text()
    assert find_q_time_pregrid_in_ile_args(args) == ["8"], args[-400:]


def _run_pseudo_pipe(tmp_path, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    # deliberately WITHOUT BIN_DIR on PATH for the forward test: the helper is invoked by
    # name, so it fails, and we read the command line it printed.
    cmd = [sys.executable, PSEUDO_PIPE, "--approx", "SEOBNRv4",
           "--use-rundir", os.fspath(tmp_path / "run")] + list(extra)
    return subprocess.run(cmd, cwd=os.fspath(tmp_path), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)


def test_pseudo_pipe_forwards_the_requested_value_to_the_helper(tmp_path):
    """Hop 1->2, executed.  pseudo_pipe prints the helper command line it is about to run; a
    forward that dropped the value, or hardcoded 1, would pass the static test."""
    proc = _run_pseudo_pipe(
        tmp_path, "--internal-ile-q-time-pregrid-factor", "8")
    assert "--internal-ile-q-time-pregrid-factor 8" in proc.stdout, proc.stdout[-3000:]


def test_pseudo_pipe_refuses_calmarg_before_it_runs_anything(tmp_path):
    """Executed refusal, and it must fire EARLY -- calibration marginalization is added by
    this script, not by the helper, so the helper can never see it."""
    proc = _run_pseudo_pipe(
        tmp_path, "--internal-ile-q-time-pregrid-factor", "8",
        "--calmarg-envelope-directory", os.fspath(tmp_path))
    assert proc.returncode != 0
    assert "--calibration-envelope-directory" in proc.stdout
    assert "helper_LDG_Events.py --force-notune" not in proc.stdout, \
        "refusal must precede the helper invocation"


def test_pseudo_pipe_refuses_an_excluded_manual_extra_ile_arg(tmp_path):
    proc = _run_pseudo_pipe(
        tmp_path, "--internal-ile-q-time-pregrid-factor", "8",
        "--manual-extra-ile-args=--rotation-slow")
    assert proc.returncode != 0
    assert "--rotation-slow" in proc.stdout


def test_pseudo_pipe_refuses_on_the_lisa_known_sky_path(tmp_path):
    proc = _run_pseudo_pipe(
        tmp_path, "--internal-ile-q-time-pregrid-factor", "8",
        "--lisa-known-sky", "--event-time", "1234.5",
        "--ecliptic-longitude", "1.25", "--ecliptic-latitude", "-0.4")
    assert proc.returncode != 0
    assert "lisa-known-sky" in proc.stdout


@pytest.mark.parametrize("path", [PSEUDO_PIPE, HELPER])
def test_argparse_choices_are_pinned_to_the_library_tuple(path):
    line = [l for l in _source(path).splitlines()
            if '"--internal-ile-q-time-pregrid-factor"' in l][0]
    assert "choices=list(Q_TIME_PREGRID_CHOICES)" in line, line


# ------------------------------------------ end to end: helper_ile_args -> refusal library

def test_prereqs_agree_with_the_library_directly():
    """Sanity link between this file's process-level tests and test_q_time_pregrid.py's
    library-level ones: the same GOOD_ILE_ARGS-shaped line should behave identically."""
    good = "X --vectorized --gpu"
    assert q_time_pregrid_pipeline_prereqs(8, good) == []
    refuse_unless_q_time_pregrid_emitted(8, good + " --q-time-pregrid-factor 8", "args_ile.txt")
