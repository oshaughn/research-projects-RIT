"""Calmarg waveform-kwargs transport, end to end through the real DAG builder.

Chain: util_RIFT_pseudo_pipe.py -> helper -> create_event_parameter_pipeline_BasicIteration
-> Calib_reweight.sub `arguments = "..."` -> HTCondor argv -> calibration_reweighting.py's
own parser and waveform-argument code -> the waveform function in RIFT.calmarg.rift_source.

Stages exercised for real: the pseudo_pipe/CEPP build and the .sub file it writes; the
calibration script's parser and --extra-waveform-kwargs block (sliced out of the script by AST
so its data loading never runs); the real rift_source waveform function, with the hlmoft call
replaced by a recorder.  The HTCondor argv split is reproduced by _condor_argv (documented
"new syntax" rules, checked against the condor manual's examples).  Where condor_submit is on
PATH, every split is also compared with the job ad from `condor_submit -dry-run`, i.e. condor's
own parse; CI has no condor, so there the parser alone stands in for it.
"""

import argparse
import ast
import re
import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from test_q_time_pregrid_dag import BIN, COINC, REF_INI, REPO, _build, _shim_path_dir, _sub_text

CAL_EXE = BIN / "calibration_reweighting.py"

pytestmark = pytest.mark.skipif(
    not (REF_INI.exists() and COINC.exists()),
    reason="reference ini/coinc fixtures not present in this checkout")

XPHM = "{'fd_alignment_postevent_time': None, 'fd_centering_factor': 0.75}"


# ---------------------------------------------------------------- condor argv split

def _split_v2(raw, doubled_dq):
    """Split an HTCondor new-syntax (V2) argument string into argv.

    Whitespace separates arguments, '...' groups, and '' inside a group is a literal '.
    In a submit file (doubled_dq=True) a literal " is written "".
    """
    argv, cur, have, in_sq, i = [], [], False, False, 0
    while i < len(raw):
        c, nxt = raw[i], raw[i + 1:i + 2]
        if c == '"' and doubled_dq:
            assert nxt == '"', "lone double quote inside new-syntax arguments: " + raw
            cur.append('"'); have = True; i += 2; continue
        if c == "'":
            if in_sq and nxt == "'":
                cur.append("'"); i += 2; continue
            in_sq = not in_sq; have = True; i += 1; continue
        if c.isspace() and not in_sq:
            if have:
                argv.append("".join(cur))
            cur, have = [], False
            i += 1; continue
        cur.append(c); have = True; i += 1
    assert not in_sq, "unbalanced single quote in " + raw
    if have:
        argv.append("".join(cur))
    return argv


def _condor_argv(sub_text, macros=None):
    """argv from a submit file's 'arguments = "..."' line, with $(macro) values substituted."""
    line = [l for l in sub_text.splitlines() if l.strip().lower().startswith("arguments")]
    assert len(line) == 1, line
    raw = line[0].split("=", 1)[1].strip()
    assert raw.startswith('"') and raw.endswith('"'), raw
    raw = raw[1:-1]
    # condor expands $(name) anywhere in the line, inside quoted values too (as for ILE.sub)
    for k, v in (macros or {}).items():
        raw = raw.replace("$({})".format(k), v)
    left = re.findall(r"\$\((\w+)\)", raw)
    assert not left, "unsubstituted submit macros {} in {}".format(left, raw)
    return _split_v2(raw, doubled_dq=True)


def _dry_run_argv(sub_path, macros):
    """argv from the job ad that `condor_submit -dry-run` writes (condor's own parse)."""
    cmd = ["condor_submit", "-dry-run", "-", sub_path.name] + [
        "{}={}".format(k, v) for k, v in macros.items()]
    out = subprocess.run(cmd, cwd=str(sub_path.parent), text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert out.returncode == 0, out.stdout
    ad = [l for l in out.stdout.splitlines() if l.startswith("Arguments=")]
    assert len(ad) == 1, out.stdout
    lit = ad[0][len("Arguments="):]
    assert lit.startswith('"') and lit.endswith('"'), lit
    # old-ClassAd string: only " is escaped (\"); a backslash is stored literally (a\b)
    value = lit[1:-1].replace('\\"', '"')
    return _split_v2(value, doubled_dq=False)


@pytest.mark.parametrize("line,argv", [
    # examples from the condor_submit manual, "arguments" (new syntax)
    ('arguments = "3 simple arguments"', ["3", "simple", "arguments"]),
    ('arguments = "one ""two"" \'spacey \'\'quoted\'\' argument\'"',
     ["one", '"two"', "spacey 'quoted' argument"]),
    ('arguments = "one \'two with spaces\' 3"', ["one", "two with spaces", "3"]),
    ("arguments = \"''\"", [""]),
    # a backslash is literal, in the submit file and in the job ad
    ('arguments = "a\\b ""q"""', ["a\\b", '"q"']),
    # $(macro) is expanded by condor even inside a quoted value
    ('arguments = "x$(foo)y \'v=$(foo)\'"', ["xbary", "v=bar"]),
])
def test_condor_argv_split_matches_manual(tmp_path, line, argv):
    macros = {"foo": "bar"} if "$(foo)" in line else {}
    assert _condor_argv(line, macros) == argv
    if shutil.which("condor_submit"):
        sub = tmp_path / "x.sub"
        sub.write_text("universe = vanilla\nexecutable = /bin/true\n{}\nqueue 1\n".format(line))
        assert _dry_run_argv(sub, macros) == argv


# ---------------------------------------------------------------- calibration script

def _calibration_waveform_arguments(argv):
    """Run calibration_reweighting.py's parser and waveform-argument block on argv."""
    import RIFT.calmarg.rift_source as rift_source
    tree = ast.parse(CAL_EXE.read_text())
    srcs = [ast.unparse(n) for n in tree.body]
    parser_nodes = [n for n, s in zip(tree.body, srcs)
                    if s.startswith("parser = argparse.ArgumentParser") or s.startswith("parser.add(")]
    i0 = next(i for i, s in enumerate(srcs) if s.startswith("if args.use_gwsignal_lmax_nyquist"))
    i1 = next(i for i, s in enumerate(srcs) if s.startswith("waveform_generator ="))
    # The script calls parser.add(...), which exists only because importing bilby_pipe imports
    # configargparse, and that aliases ArgumentParser.add = add_argument.  Same alias, local.
    class _ArgumentParser(argparse.ArgumentParser):
        add = argparse.ArgumentParser.add_argument
    ns = {"argparse": types.SimpleNamespace(ArgumentParser=_ArgumentParser), "ast": ast,
          "rift_source": rift_source}
    exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), str(CAL_EXE), "exec"), ns)
    ns["args"] = ns["parser"].parse_args(argv)
    ns["data"] = types.SimpleNamespace(meta_data={"command_line_args": {
        "reference_frequency": 20.0, "waveform_approximant": "IMRPhenomXPHM",
        "frequency_domain_source_model": "lal_binary_black_hole"}})
    ns["ifos"] = types.SimpleNamespace(sampling_frequency=4096.0)
    exec(compile(ast.Module(body=tree.body[i0:i1], type_ignores=[]), str(CAL_EXE), "exec"), ns)
    return ns["args"], ns["waveform_arguments"], ns["wf_func"]


class _Captured(Exception):
    pass


def _waveform_call(monkeypatch, wf_func, waveform_arguments):
    """Call the real rift_source waveform function as bilby would; record the hlmoft call."""
    import RIFT.calmarg.rift_source as rift_source
    seen = {}

    def recorder(P, **kw):
        seen.update(kw)
        raise _Captured

    monkeypatch.setattr(rift_source.lalsimutils, "hlmoft", recorder)
    if getattr(rift_source, "has_GWS", False):
        monkeypatch.setattr(rift_source.rgws, "hlmoft", recorder)
    freqs = np.arange(0, 1024.0, 0.25)
    with pytest.raises(_Captured):
        wf_func(freqs, 36.0, 29.0, 400.0, 0.1, 0.0, 0.2, 0.0, 0.0, -0.1, 0.4, 0.3,
                **waveform_arguments)
    return seen


# ---------------------------------------------------------------- the build

def _ini(tmp_path, ini_lines):
    """Reference ini, OSG off, tiny grid, calmarg on; ini_lines appended to [rift-pseudo-pipe]."""
    text = REF_INI.read_text()
    for flag in ("use_osg", "use_osg_file_transfer", "use_osg_cip"):
        text = text.replace("{}=True".format(flag), "{}=False".format(flag))
    text = re.sub(r"force-initial-grid-size=\d+", "force-initial-grid-size=4", text)
    text = text.replace("calibration-reweighting=False", "calibration-reweighting=True")
    text = text.replace('bilby-ini-file=".travis/ref_ini/bilby_GW150914.ini"',
                        'bilby-ini-file="{}"'.format(REPO / ".travis/ref_ini/bilby_GW150914.ini"))
    text = re.sub(r"^manual-extra-ile-args=.*\n", "", text, flags=re.M)
    lines = text.splitlines()
    last_section = max(i for i, l in enumerate(lines) if l.startswith("["))
    assert lines[last_section] == "[rift-pseudo-pipe]", lines[last_section]
    out = tmp_path / "ref_calmarg.ini"
    out.write_text(text.rstrip("\n") + "\n" + "".join(l + "\n" for l in ini_lines))
    return out


def _failure_text(text):
    """The tail of a failed build, plus every error block (child output is not in order)."""
    lines = text.splitlines()
    marks = [i for i, l in enumerate(lines)
             if "Traceback" in l or "Error" in l or "error:" in l or "FAIL" in l]
    blocks = ["\n".join(lines[max(0, i - 2):i + 25]) for i in marks[:6]]
    return "\n----\n".join(blocks + [text[-2000:]])


def _calibration_argv(tmp_path, name, ini_lines, cli=()):
    # CEPP is handed `which bilby_pipe_generation`; the pickle job is never run here, so a
    # stub on PATH keeps the build independent of whether bilby_pipe is installed.
    stub = _shim_path_dir(tmp_path) / "bilby_pipe_generation"  # _build puts this dir on PATH
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    out, rundir = _build(tmp_path, name, list(cli), ini_fn=lambda p: _ini(p, ini_lines))
    assert out.returncode == 0, _failure_text(out.stdout)
    # DAG VARS of the first calibration node fill $(macrostartidx) etc., as DAGMan does
    dag = next(rundir.glob("*.dag")).read_text()
    node = re.search(r"^JOB (\S+) Calib_reweight\.sub$", dag, re.M).group(1)
    vars_line = re.search(r"^VARS {} (.*)$".format(re.escape(node)), dag, re.M).group(1)
    macros = dict(re.findall(r'(\w+)="([^"]*)"', vars_line))
    argv = _condor_argv(_sub_text(rundir, "Calib_reweight.sub"), macros)
    if shutil.which("condor_submit"):
        assert _dry_run_argv(rundir / "Calib_reweight.sub", macros) == argv
    return argv


def _calarg(s):
    return "calibration-reweighting-initial-extra-args=" + repr(s)   # ini values are eval'd


def _manual(d):
    return 'manual-extra-ile-args=--internal-waveform-extra-kwargs "{}"'.format(d)  # passed raw


STRV = "{'approx_tag': 'foo', 'fd_centering_factor': 0.75}"
SPACE = "{'note': 'a  b', 'fd_centering_factor': 0.75}"
L = {"fd_L_frame": True}   # the reference ini sets internal-mitigate-fd-J-frame=L_frame
XPHM_D = {"fd_alignment_postevent_time": None, "fd_centering_factor": 0.75}

# id, ini lines, pseudo_pipe CLI, expected extra_waveform_kwargs, other expected args
CASES = [
    ("A_none", [], (), L, {}),
    ("B_plain", [_calarg("--internal-waveform-fd-no-condition --internal-use-normal-reweight 0.5 --l-max 3")],
     (), dict(L, no_condition=True), {"l_max": 3, "internal_use_normal_reweight": 0.5}),
    ("C_xphm_ini", [_calarg('--extra-waveform-kwargs "{}"'.format(XPHM))], (), dict(L, **XPHM_D), {}),
    ("C2_xphm_cli", [], ('--calibration-reweighting-initial-extra-args=--extra-waveform-kwargs "{}"'.format(XPHM),),
     dict(L, **XPHM_D), {}),
    ("D_xphm_manual", [_manual(XPHM)], (), dict(L, **XPHM_D), {}),
    ("E_string_value", [_manual(STRV)], (), dict(L, approx_tag="foo", fd_centering_factor=0.75), {}),
    ("F_space_manual", [_manual(SPACE)], (), dict(L, note="a  b", fd_centering_factor=0.75), {}),
    ("G_space_ini", [_calarg('--extra-waveform-kwargs "{}"'.format(SPACE))], (),
     dict(L, note="a  b", fd_centering_factor=0.75), {}),
    # lmax_nyquist is split out to --use-gwsignal-lmax-nyquist, which selects h_method gws_hlmoft
    ("H_gwsignal", [_manual("{'lmax_nyquist': 2, 'fd_centering_factor': 0.75}")], (),
     dict(L, lmax_nyquist=2, fd_centering_factor=0.75), {"use_gwsignal": True}),
]


@pytest.mark.parametrize("name,ini_lines,cli,expected,other", CASES, ids=[c[0] for c in CASES])
def test_calmarg_waveform_kwargs_reach_waveform_call(tmp_path, monkeypatch, name, ini_lines,
                                                      cli, expected, other):
    argv = _calibration_argv(tmp_path, name, ini_lines, cli)
    args, waveform_arguments, wf_func = _calibration_waveform_arguments(argv)
    assert args.fref == 20.0
    for k, v in other.items():
        assert getattr(args, k) == v, (k, getattr(args, k))
    got = waveform_arguments["extra_waveform_kwargs"]
    assert got == expected and all(type(got[k]) is type(v) for k, v in expected.items()), got
    import RIFT.calmarg.rift_source as rift_source
    if args.h_method == "gws_hlmoft" and not rift_source.has_GWS:
        pytest.skip("gwsignal interface not importable here")
    seen = _waveform_call(monkeypatch, wf_func, waveform_arguments)
    assert _hlmoft_kwargs(args, seen) == expected, seen


def _hlmoft_kwargs(args, seen):
    """The waveform options the recorded hlmoft call carries."""
    if args.h_method == "gws_hlmoft":
        # rgws.hlmoft(P, Lmax=..., approx_string=..., **extra_waveform_kwargs)
        return {k: v for k, v in seen.items() if k not in ("Lmax", "approx_string")}
    # lalsimutils.hlmoft(P, Lmax=..., **extra_waveform_kwargs), the form ILE uses
    return {k: v for k, v in seen.items() if k != "Lmax"}


@pytest.mark.xfail(strict=True, reason=(
    "pseudo_pipe writes --extra-waveform-kwargs \"{repr}\"; a repr containing a double quote "
    "makes the builder fail loudly (base: safely_quote_arg_str raises; this PR: shlex.split "
    "'No closing quotation').  Not silent, and not addressed here."))
def test_nested_quotes_in_waveform_kwargs(tmp_path):
    """A string value that itself contains a quote character."""
    argv = _calibration_argv(tmp_path, "nested", [_manual("{'mode_label': \"it's\"}")])
    _, waveform_arguments, _ = _calibration_waveform_arguments(argv)
    assert waveform_arguments["extra_waveform_kwargs"] == dict(L, mode_label="it's")


def test_hlmoft_path_applies_waveform_kwargs(monkeypatch):
    """The kwargs bind to hlmoft's own parameters (before the O4c port they sat unused in **kwargs)."""
    import inspect
    import RIFT.lalsimutils as lalsimutils
    real = lalsimutils.hlmoft
    argv = ["--use_rift_samples=True", "--fmin", "20", "--extra-waveform-kwargs", XPHM]
    _, waveform_arguments, wf_func = _calibration_waveform_arguments(argv)
    seen = _waveform_call(monkeypatch, wf_func, waveform_arguments)
    bound = inspect.signature(real).bind(None, **seen)
    bound.apply_defaults()
    assert bound.arguments["fd_centering_factor"] == 0.75
