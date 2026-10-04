"""Calmarg must build the waveform ILE builds: same hlmoft options, same modes.

For each configuration one real DAG is built (pseudo_pipe -> helper -> CEPP).  Two argvs are
read from it, as condor splits them: Calib_reweight.sub's and ILE_extr.sub's.

- calmarg side: calibration_reweighting.py's own parser and waveform-argument block, then
  the real RIFT.calmarg.rift_source waveform function; the generator call is recorded.
- ILE side: ILE's own option parser, the waveform-options block of analyze_event and the
  keywords it hands PrecomputeLikelihoodTerms, then the real
  factored_likelihood.internal_hlm_generator; the generator call is recorded.

The recorded options must be equal.  ILE adds fd_alignment_postevent_time=2 (user
--internal-waveform-extra-kwargs wins), nests --internal-waveform-extra-lalsuite-args under
'extra_waveform_args', and on its lalsuite route adds fd_standoff_factor=0.9 unless set.

The waveform tests then generate both sides' modes for precessing XPHM and XO4a and require
identical h+.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from test_q_time_pregrid_dag import BIN, CODE, COINC, REF_INI, _sub_text
from test_calmarg_extra_args_transport import (
    _build, _calibration_waveform_arguments, _condor_argv, _failure_text, _ini,
    _shim_path_dir, _waveform_call)

ILE_EXE = BIN / "integrate_likelihood_extrinsic_batchmode"

pytestmark = pytest.mark.skipif(
    not (REF_INI.exists() and COINC.exists()),
    reason="reference ini/coinc fixtures not present in this checkout")

# Keys ILE passes that no generator reads (asserted below), and call-form arguments.
INERT = ("e_freq",)
CALL_FORM = ("Lmax", "approx_string")


def test_inert_keys_are_inert():
    for path in (CODE / "RIFT" / "lalsimutils.py", CODE / "RIFT" / "physics" / "GWSignal.py"):
        text = path.read_text()
        for key in INERT:
            assert not re.search(r"\b{}\b".format(key), text), \
                "{} now reads {}: compare it too".format(path.name, key)


# ---------------------------------------------------------------- ILE side

def _definitions_of_free_names(earlier, nodes):
    """The earlier top-level imports/assignments that bind names `nodes` read but do not set."""
    import builtins
    used, bound = set(), set()
    for n in nodes:
        if isinstance(n, ast.FunctionDef):
            bound.add(n.name)
        for x in ast.walk(n):
            if isinstance(x, ast.Name):
                (bound if isinstance(x.ctx, ast.Store) else used).add(x.id)
            elif isinstance(x, ast.arg):
                bound.add(x.arg)
    free = {u for u in used - bound if not hasattr(builtins, u)}
    out = []
    for n in earlier:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            names = {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.Assign):
            names = {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
        else:
            continue
        if names & free:
            out.append(n)
    return out


def _ile_generator_kwargs(argv):
    """ILE's options for internal_hlm_generator, from ILE's own source run on argv.

    Runs ILE's option parser, then analyze_event's waveform-options block (from
    `extra_waveform_kwargs = {}` to `t_window = ...`), then evaluates the keyword arguments
    of analyze_event's first PrecomputeLikelihoodTerms call that internal_hlm_generator takes.
    """
    import inspect
    from RIFT.likelihood import factored_likelihood as fl
    tree = ast.parse(ILE_EXE.read_text())
    srcs = [ast.unparse(n) for n in tree.body]
    i0 = next(i for i, s in enumerate(srcs) if s.startswith("optp = OptionParser("))
    # python < 3.11 unparses a tuple target with parentheses
    i1 = next(i for i, s in enumerate(srcs)
              if s.replace("(opts, args)", "opts, args").startswith("opts, args = optp.parse_args("))
    ns = {}
    body = _definitions_of_free_names(tree.body[:i0], tree.body[i0:i1]) + tree.body[i0:i1]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ILE_EXE), "exec"), ns)
    opts, _ = ns["optp"].parse_args(ns["_normalize_interpolate_time_argv"](argv))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "analyze_event")
    fsrc = [ast.unparse(n) for n in fn.body]
    j0 = next(i for i, s in enumerate(fsrc) if s.startswith("extra_waveform_kwargs = {}"))
    j1 = next(i for i, s in enumerate(fsrc) if s.startswith("t_window ="))
    loc = {"opts": opts, "NR_template_group": None, "NR_template_param": None}
    exec(compile(ast.Module(body=fn.body[j0:j1], type_ignores=[]), str(ILE_EXE), "exec"), loc)
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and ast.unparse(n.func).endswith("PrecomputeLikelihoodTerms"))
    takes = set(inspect.signature(fl.internal_hlm_generator).parameters)
    out = {}
    for kw in call.keywords:
        expr = compile(ast.Expression(body=kw.value), str(ILE_EXE), "eval")
        if kw.arg is None:        # **mapping: only the mappings the block above built
            if isinstance(kw.value, ast.Name) and kw.value.id in loc:
                out.update(eval(expr, loc))
        elif kw.arg in takes and kw.arg not in ("verbose", "quiet", "skip_interpolation"):
            out[kw.arg] = eval(expr, loc)
    assert out["extra_waveform_kwargs"] is loc["extra_waveform_kwargs"]
    return opts, out


class _Captured(Exception):
    pass


def _ile_generator_call(monkeypatch, wgk, P, Lmax=2, generate=False):
    """Run internal_hlm_generator as PrecomputeLikelihoodTerms does; record the generator call."""
    import RIFT.lalsimutils as lsu
    from RIFT.likelihood import factored_likelihood as fl
    seen = {}
    real = lsu.hlmoft

    def recorder(P, Lmax, **kw):
        seen["kwargs"] = dict(kw)
        if generate:
            seen["hlm"] = real(P, Lmax, **kw)
        raise _Captured

    monkeypatch.setattr(fl.lsu, "std_and_conj_hlmoff", recorder)
    if getattr(fl, "has_GWS", False):
        monkeypatch.setattr(fl.rgws, "std_and_conj_hlmoff", recorder)
    with pytest.raises(_Captured):
        fl.internal_hlm_generator(P, Lmax, verbose=False, quiet=True, **wgk)
    monkeypatch.undo()
    return seen


def _ile_options(seen):
    kw = {k: v for k, v in seen["kwargs"].items() if k not in INERT + CALL_FORM}
    if kw.get("force_22_mode") is False:   # explicit default
        del kw["force_22_mode"]
    return kw


def _calmarg_options(seen):
    return {k: v for k, v in seen.items() if k not in CALL_FORM}


def _P():
    import lal
    import RIFT.lalsimutils as lsu
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 36.0 * lal.MSUN_SI, 29.0 * lal.MSUN_SI
    P.s1x, P.s1z, P.s2y = 0.4, 0.2, -0.3
    P.fmin = P.fref = 20.0
    P.deltaT, P.deltaF = 1.0 / 2048, 0.125
    P.dist = 400e6 * lal.PC_SI
    P.approx = lsu.lalsim.GetApproximantFromString("IMRPhenomXPHM")
    return P


# ---------------------------------------------------------------- end to end: options

PREC = "{'PhenomXPrecVersion': 102}"
ALIGN_NONE = "{'fd_alignment_postevent_time': None}"
ALIGN_1 = "{'fd_alignment_postevent_time': 1.5, 'fd_standoff_factor': 0.95}"


def _manual(s):
    return "manual-extra-ile-args=" + s


# id, [rift-pseudo-pipe] ini lines, expected calmarg/ILE options on the lalsuite route
CASES = [
    ("no_L_frame", ["internal-mitigate-fd-J-frame='rotate'"],
     {"fd_alignment_postevent_time": 2, "fd_standoff_factor": 0.9}),
    ("L_frame_default", [],
     {"fd_alignment_postevent_time": 2, "fd_standoff_factor": 0.9, "fd_L_frame": True}),
    ("xphm_lalsuite_args", [_manual('--internal-waveform-extra-lalsuite-args "{}"'.format(PREC))],
     {"fd_alignment_postevent_time": 2, "fd_standoff_factor": 0.9, "fd_L_frame": True,
      "extra_waveform_args": {"PhenomXPrecVersion": 102}}),
    ("user_alignment_none", [_manual('--internal-waveform-extra-kwargs "{}"'.format(ALIGN_NONE))],
     {"fd_alignment_postevent_time": None, "fd_standoff_factor": 0.9, "fd_L_frame": True}),
    # (ILE's sub writer refuses two quoted blocks in manual-extra-ile-args, so one case each)
    ("user_alignment_and_standoff", [_manual('--internal-waveform-extra-kwargs "{}"'.format(ALIGN_1))],
     {"fd_alignment_postevent_time": 1.5, "fd_standoff_factor": 0.95, "fd_L_frame": True}),
]


def _argvs(tmp_path, name, ini_lines):
    stub = _shim_path_dir(tmp_path) / "bilby_pipe_generation"
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    out, rundir = _build(tmp_path, name, [], ini_fn=lambda p: _ini(p, ini_lines))
    assert out.returncode == 0, _failure_text(out.stdout)

    def argv(sub):
        text = _sub_text(rundir, sub)
        line = [l for l in text.splitlines() if l.strip().lower().startswith("arguments")][0]
        macros = {m: "0" for m in re.findall(r"\$\((\w+)\)", line)}
        return _condor_argv(text, macros)
    return argv("Calib_reweight.sub"), argv("ILE_extr.sub")


@pytest.mark.parametrize("name,ini_lines,expected", CASES, ids=[c[0] for c in CASES])
def test_calmarg_passes_ile_waveform_options(tmp_path, monkeypatch, name, ini_lines, expected):
    cal_argv, ile_argv = _argvs(tmp_path, name, ini_lines)
    args, waveform_arguments, wf_func = _calibration_waveform_arguments(cal_argv)
    assert args.h_method == "hlmoft"
    cal = _calmarg_options(_waveform_call(monkeypatch, wf_func, waveform_arguments))
    monkeypatch.undo()
    _, wgk = _ile_generator_kwargs(ile_argv)
    ile = _ile_options(_ile_generator_call(monkeypatch, wgk, _P()))
    assert ile == expected, ile          # the test's model of ILE is right
    assert cal == ile, (cal, ile)


# ---------------------------------------------------------------- waveforms

def _calmarg_hlm(monkeypatch, approx, cal_argv):
    """calmarg's modes and P for argv, from the real block and rift_source (stops after hlmoft)."""
    import RIFT.calmarg.rift_source as rift_source
    _, waveform_arguments, wf_func = _calibration_waveform_arguments(cal_argv)
    waveform_arguments = dict(waveform_arguments, waveform_approximant=approx, Lmax=4)
    real = rift_source.lalsimutils.hlmoft
    seen = {}

    def spy(P, **kw):
        seen["P"] = P.manual_copy()
        seen["kwargs"] = dict(kw)
        seen["hlm"] = real(P, **kw)
        raise _Captured

    monkeypatch.setattr(rift_source.lalsimutils, "hlmoft", spy)
    with pytest.raises(_Captured):
        wf_func(np.arange(0, 1024.0 + 0.125, 0.125), 36.0, 29.0, 400.0,
                0.4, 0.0, 0.2, 0.0, -0.3, 0.0, 0.4, 0.3, **waveform_arguments)
    monkeypatch.undo()
    return seen


def _hplus(hlm, incl=0.4, phiref=0.3):
    import lal
    out = 0
    for (l, m), h in hlm.items():
        out = out + h.data.data * lal.SpinWeightedSphericalHarmonic(incl, -phiref, -2, l, m)
    return np.real(out)


@pytest.mark.parametrize("approx", ["IMRPhenomXPHM", "IMRPhenomXO4a"])
@pytest.mark.parametrize("extra", [[], ["--extra-waveform-kwargs",
                                        "{'extra_waveform_args': {'PhenomXPrecVersion': 102}}"]],
                         ids=["default", "prec_version"])
def test_calmarg_hplus_equals_ile_hplus(monkeypatch, approx, extra):
    import lalsimulation
    try:
        lalsimulation.GetApproximantFromString(approx)
    except Exception:
        pytest.skip(approx + " not in this lalsimulation")
    cal_argv = ["--use_rift_samples=True", "--fmin", "20", "--internal-waveform-fd-L-frame"] + extra
    cal = _calmarg_hlm(monkeypatch, approx, cal_argv)
    ile_argv = ["--approximant", approx, "--internal-waveform-fd-L-frame"]
    if extra:
        ile_argv += ["--internal-waveform-extra-lalsuite-args", "{'PhenomXPrecVersion': 102}"]
    _, wgk = _ile_generator_kwargs(ile_argv)
    P = cal["P"]
    ile = _ile_generator_call(monkeypatch, wgk, P.manual_copy(), Lmax=4, generate=True)
    assert set(cal["hlm"]) == set(ile["hlm"])
    for mode in cal["hlm"]:
        assert float(cal["hlm"][mode].epoch) == float(ile["hlm"][mode].epoch), mode
    a, b = _hplus(cal["hlm"]), _hplus(ile["hlm"])
    assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(b)), np.max(np.abs(a - b)) / np.max(np.abs(b))


def test_nested_lalsuite_args_reach_the_waveform(monkeypatch):
    """Control for the prec_version case above: the nested option does change XPHM."""
    base = ["--use_rift_samples=True", "--fmin", "20", "--internal-waveform-fd-L-frame"]
    a = _calmarg_hlm(monkeypatch, "IMRPhenomXPHM", base)
    b = _calmarg_hlm(monkeypatch, "IMRPhenomXPHM", base + [
        "--extra-waveform-kwargs", "{'extra_waveform_args': {'PhenomXPrecVersion': 102}}"])
    ha, hb = _hplus(a["hlm"]), _hplus(b["hlm"])
    assert np.max(np.abs(ha - hb)) > 1e-3 * np.max(np.abs(ha))
