#!/usr/bin/env python
"""
Tests for the AV live-volume state, per-axis bin allocation and collapse gate ported into
the LISA ILE driver (bin/integrate_likelihood_extrinsic_batchmode_lisa).

Four options, all sampler-agnostic: --sampler-save-state / --sampler-load-state (the AV
grid, which carries no detector convention), --sampler-anisotropic-bins, and
--reject-collapsed-live-volume.

THE ONE THING TO KNOW.  The main driver calls its collapse gate TWICE -- once on the first
run, and again on the replica POOL, because replication can turn a healthy first run into a
collapsed pool.  This driver has no replica pooling yet, so only the first call exists here.
When --mc-error-replicas is ported the second call MUST come with it, or the flag is
silently bypassed for exactly the case pooling introduces.  That is recorded at the helper,
in the drift ledger, and asserted below.
"""

import ast
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LISA = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')
_MAIN = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

OPTS = ["--sampler-save-state", "--sampler-load-state",
        "--sampler-anisotropic-bins", "--reject-collapsed-live-volume"]

HELPERS = ['_maybe_load_av_state', '_maybe_save_av_state',
           '_maybe_enable_anisotropic_bins', '_reject_if_collapsed',
           '_report_and_gate_collapse']


def _src(path):
    with open(path) as fh:
        return fh.read()


def _option_nodes(path):
    out = {}
    for n in ast.walk(ast.parse(_src(path), filename=path)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("add_option", "add_argument")):
            names = [a.value for a in n.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if names and names[0].startswith("--"):
                out[names[0]] = n
    return out


def _kwargs_of(node):
    out = {}
    for kw in node.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            out[kw.arg] = ast.dump(kw.value)
    return out


class _Collapse(Exception):
    pass


class _AVModule(object):
    LiveVolumeCollapse = _Collapse


def _load(**optkw):
    base = {"sampler_load_state": None, "sampler_save_state": None,
            "sampler_anisotropic_bins": False, "reject_collapsed_live_volume": False,
            "sampler_method": "AV"}
    base.update(optkw)
    defs = {n.name: n for n in ast.parse(_src(_LISA)).body
            if isinstance(n, ast.FunctionDef) and n.name in HELPERS}
    missing = sorted(set(HELPERS) - set(defs))
    assert not missing, "LISA driver is missing: %s" % missing
    mod = ast.Module(body=[defs[n] for n in HELPERS], type_ignores=[])
    ns = {"opts": type("O", (), base)(),
          "mcsamplerAdaptiveVolume": _AVModule, "mcsampler_AV_ok": True}
    exec(compile(ast.fix_missing_locations(mod), "av_state", "exec"), ns)
    return ns


# ------------------------------------------------------------------------------- options
@pytest.mark.parametrize("opt", OPTS)
def test_option_present_and_matches_the_main_driver(opt):
    a, b = _kwargs_of(_option_nodes(_LISA)[opt]), _kwargs_of(_option_nodes(_MAIN)[opt])
    for key in ("default", "type", "action", "choices"):
        assert a.get(key) == b.get(key), "%s: %s differs" % (opt, key)


# ---------------------------------------------------------------------------- load / save
class _AV(object):
    def __init__(self):
        self.loaded = self.saved = None

    def load_state(self, p):
        self.loaded = p

    def save_state(self, p):
        self.saved = p


class _NoState(object):
    pass


def test_state_hooks_are_noops_when_unset():
    ns = _load()
    s = _AV()
    ns['_maybe_load_av_state'](s)
    ns['_maybe_save_av_state'](s)
    assert s.loaded is None and s.saved is None


def test_state_round_trip_reaches_the_sampler():
    ns = _load(sampler_load_state="/in.npz", sampler_save_state="/out.npz")
    s = _AV()
    ns['_maybe_load_av_state'](s)
    ns['_maybe_save_av_state'](s)
    assert s.loaded == "/in.npz" and s.saved == "/out.npz"


def test_save_state_is_restricted_to_the_AV_method():
    """Main gates the save on sampler_method == 'AV'; a portfolio's aggregate has no such grid."""
    ns = _load(sampler_save_state="/out.npz", sampler_method="portfolio")
    s = _AV()
    ns['_maybe_save_av_state'](s)
    assert s.saved is None


def test_state_hooks_tolerate_a_sampler_without_state_support():
    ns = _load(sampler_load_state="/in.npz", sampler_save_state="/out.npz")
    ns['_maybe_load_av_state'](_NoState())
    ns['_maybe_save_av_state'](_NoState())


def test_a_bad_state_file_degrades_to_a_cold_run():
    """A missing/corrupt state must not kill the point."""
    ns = _load(sampler_load_state="/in.npz", sampler_save_state="/out.npz")

    class _Boom(object):
        def load_state(self, p):
            raise IOError("nope")

        def save_state(self, p):
            raise IOError("read-only")

    ns['_maybe_load_av_state'](_Boom())
    ns['_maybe_save_av_state'](_Boom())


# --------------------------------------------------------------------------- anisotropic
class _Binned(object):
    anisotropic_bins = False


def test_anisotropic_bins_is_opt_in():
    ns = _load()
    s = _Binned()
    ns['_maybe_enable_anisotropic_bins'](s)
    assert s.anisotropic_bins is False


def test_anisotropic_bins_reaches_portfolio_members_too():
    """The grid lives on the MEMBERS; setting it only on the aggregate would do nothing."""
    ns = _load(sampler_anisotropic_bins=True)
    m1, m2 = _Binned(), _Binned()
    s = _Binned()
    s.portfolio_realizations = [m1, m2]
    ns['_maybe_enable_anisotropic_bins'](s)
    assert s.anisotropic_bins and m1.anisotropic_bins and m2.anisotropic_bins


def test_anisotropic_bins_skips_members_that_do_not_support_it():
    ns = _load(sampler_anisotropic_bins=True)
    s = _Binned()
    s.portfolio_realizations = [_NoState()]
    ns['_maybe_enable_anisotropic_bins'](s)          # must not raise
    assert s.anisotropic_bins is True


# -------------------------------------------------------------------------- collapse gate
COLLAPSED = {'live_volume_collapsed': True, 'collapse_reason': 'zero volume'}
HEALTHY = {'live_volume_collapsed': False}


def test_gate_is_inert_when_the_flag_is_off():
    _load()['_reject_if_collapsed'](COLLAPSED, "first run")


def test_gate_is_inert_on_a_healthy_run():
    _load(reject_collapsed_live_volume=True)['_reject_if_collapsed'](HEALTHY, "first run")


def test_gate_raises_when_flag_set_and_run_collapsed():
    with pytest.raises(_Collapse):
        _load(reject_collapsed_live_volume=True)['_reject_if_collapsed'](COLLAPSED, "first run")


def test_gate_message_names_the_stage_and_reason():
    """The stage is in the message because the main driver calls this at two stages."""
    with pytest.raises(_Collapse) as e:
        _load(reject_collapsed_live_volume=True)['_reject_if_collapsed'](COLLAPSED, "pooled")
    assert "pooled" in str(e.value) and "zero volume" in str(e.value)


@pytest.mark.parametrize("dd", [None, "not a dict", {}])
def test_gate_tolerates_a_missing_or_malformed_dict_return(dd):
    _load(reject_collapsed_live_volume=True)['_reject_if_collapsed'](dd, "first run")


def test_report_announces_a_collapse_even_when_the_gate_is_off(capsys):
    """Not rejecting is not the same as not telling anyone."""
    ns = _load()
    capsys.readouterr()
    ns['_report_and_gate_collapse'](COLLAPSED)
    out = capsys.readouterr().out
    assert "LIVE VOLUME COLLAPSED" in out and "NOT a fair draw" in out


def test_report_says_nothing_on_a_healthy_run(capsys):
    ns = _load()
    capsys.readouterr()
    ns['_report_and_gate_collapse'](HEALTHY)
    assert "COLLAPSED" not in capsys.readouterr().out


def test_report_still_raises_when_gated():
    with pytest.raises(_Collapse):
        _load(reject_collapsed_live_volume=True)['_report_and_gate_collapse'](COLLAPSED)


def test_gate_falls_back_to_RuntimeError_without_AV():
    """mcsampler_AV_ok False -> the AV exception class is unavailable."""
    defs = {n.name: n for n in ast.parse(_src(_LISA)).body
            if isinstance(n, ast.FunctionDef) and n.name == '_reject_if_collapsed'}
    mod = ast.Module(body=[defs['_reject_if_collapsed']], type_ignores=[])
    ns = {"opts": type("O", (), {"reject_collapsed_live_volume": True})(),
          "mcsamplerAdaptiveVolume": None, "mcsampler_AV_ok": False}
    exec(compile(ast.fix_missing_locations(mod), "av_state", "exec"), ns)
    with pytest.raises(RuntimeError):
        ns['_reject_if_collapsed'](COLLAPSED, "first run")


# ------------------------------------------------------------------------- call-site wiring
def test_both_analyze_event_variants_get_every_hook():
    tree = ast.parse(_src(_LISA))
    fns = {n.name: n for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name in ('analyze_event', 'analyze_event_LISA')}
    assert set(fns) == {'analyze_event', 'analyze_event_LISA'}
    for name, node in fns.items():
        called = {c.func.id for c in ast.walk(node)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        for hook in ('_maybe_load_av_state', '_maybe_save_av_state',
                     '_maybe_enable_anisotropic_bins', '_report_and_gate_collapse'):
            assert hook in called, "%s does not call %s" % (name, hook)


def test_hook_ordering_at_both_call_sites():
    """load/aniso before the integration, save after it, the gate after the result check."""
    src = _src(_LISA)
    pos = 0
    for _ in range(2):
        load = src.index("_maybe_load_av_state(sampler)", pos)
        aniso = src.index("_maybe_enable_anisotropic_bins(sampler)", load)
        integ = src.index("sampler.integrate(like_to_integrate", aniso)
        save = src.index("_maybe_save_av_state(sampler)", integ)
        guard = src.index("if not(res): # no resut", save)
        gate = src.index("_report_and_gate_collapse(dict_return", guard)
        assert load < aniso < integ < save < guard < gate
        pos = gate + 1


def test_the_second_gate_call_site_is_recorded_as_missing():
    """Main gates twice; this driver gates once because it has no replica pooling yet.

    If someone ports --mc-error-replicas without adding the second call, the flag is
    silently bypassed for the case pooling creates.  This asserts the warning is still
    written down where that person will be working.
    """
    src = _src(_LISA)
    fn = src[src.index("def _reject_if_collapsed"):]
    fn = fn[:fn.index("\ndef ")]
    assert "mc-error-replicas" in fn and "TWICE" in fn
    # Count CALLS, not textual occurrences: the `def` line matches the same substring.
    calls = [c for c in ast.walk(ast.parse(src))
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == '_report_and_gate_collapse']
    assert len(calls) == 2, \
        "expected exactly one gate call per analyze_event variant, found %d" % len(calls)


# ---------------------------------------------------------------- anti-drift vs the main driver
def _named(path, name):
    for n in ast.walk(ast.parse(_src(path))):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError("%s not found in %s" % (name, os.path.basename(path)))


def _normalized(fn):
    node = ast.parse(ast.unparse(fn)).body[0] if hasattr(ast, "unparse") else fn
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return ast.dump(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])))


def test_reject_if_collapsed_body_is_identical_to_the_main_drivers():
    """Hoisted out of analyze_event here, but the body must not have changed with it."""
    assert (_normalized(_named(_LISA, '_reject_if_collapsed'))
            == _normalized(_named(_MAIN, '_reject_if_collapsed'))), \
        "_reject_if_collapsed has drifted between the two drivers (docstrings excluded)"
