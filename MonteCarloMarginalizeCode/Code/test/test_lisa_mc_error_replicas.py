#!/usr/bin/env python
"""
MC-error replicas and replica pooling in the LISA ILE driver.

`--mc-error-replicas` re-runs the extrinsic integration as independent cold replicas when the
reported error is untrustworthy, then POOLS every replica's samples rather than picking one.
Pooling, not selection, because lnZ is the linear mean over K replicas so the exported
posterior must represent that same mixture -- and n_eff is the wrong selector anyway, since it
measures weight CONCENTRATION, not coverage, so a mode-collapsed replica scores highest.

THE THREE THINGS THAT MUST BE RIGHT, each a defect the main driver already paid for:

1. `already_resampled` is a PER-REPLICA SEQUENCE, not one boolean.  Each pass decides
   independently whether to fair-draw (the draw is skipped when it would not shrink that
   pass's record), so near the n_extr boundary a run produces a MIXTURE.  One global flag
   either flattens a replica whose importance weights are genuine, or leaves a resampled
   replica double-weighted (audit Finding 6).
2. The empty-record filter runs in LOCKSTEP with rep_lnZ and the flags.  Filtering rep_rvs
   alone shifts every later block against the wrong evidence.
3. The collapse gate fires on the POOLED verdict as well as the first run, or
   --reject-collapsed-live-volume is silently bypassed for the case pooling creates.

Pooling maths: block k gets weights summing to Z_k/K -- equal WITHIN a block when that block
was fair-drawn (it is already an equal-weight posterior draw), scaled otherwise.  That is the
importance weight against the real pooled proposal q'_ki = q_ki * K * n_k.
"""

import ast
import os
import textwrap

import numpy as np
from RIFT.integrators.rvs_record import SamplerOutputMixin as _SamplerOutputMixin
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LISA = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')
_MAIN = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

# The record accessors are REQUIRED here even though no test calls them directly:
# _lnZ_of_rvs / _kish_neff_of_rvs resolve their weights through _lw_of.  Leaving one out
# does NOT raise -- _lnZ_of_rvs catches broadly and returns None, so a NameError becomes
# "no evidence for this block" and the pooled weights silently collapse to 1/K.  That is
# an assertion failure three layers away from its cause; see the guard in H() below.
HELPERS = ['_rvs_lnL_convention', 'ln_weights_from_rvs', '_rvs_len',
           '_rvs_record_for', '_sampler_keeps_records', '_internal_record_of',
           '_rebound_record', '_lw_of', '_lnZ_of_rvs',
           '_kish_neff_of_rvs', '_extract_mc_diag', '_pool_replica_rvs']


def _src(path):
    with open(path) as fh:
        return fh.read()


def _driver_def_names(path):
    """Every top-level name the driver BINDS: functions and imports alike.

    Imports are in here because of a real miss: the guard originally covered only defs, so
    `SamplerOutputMixin` -- imported by the driver, referenced by _sampler_keeps_records --
    slipped straight through it and surfaced as a NameError inside an exec'd helper.
    """
    with open(path) as fh:          # read directly: _src() differs between these harnesses
        src = fh.read()
    names = set()
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name != '*':
                    names.add(a.asname or a.name.split('.')[0])
    return names


def _assert_helper_set_is_closed(ns, names, path=_LISA):
    """Fail LOUDLY if an exec'd helper calls a driver helper that was not exec'd with it.

    Without this, a name missing from the list above is not a NameError anyone sees:
    _lnZ_of_rvs catches broadly and returns None, so the omission reads as "this block
    has no evidence" and the pooled weights collapse to 1/K.  The test then fails on a
    weight assertion far from the cause.  Checked against the DRIVER's own def names, so
    ordinary attribute names and locals cannot trip it.
    """
    driver = _driver_def_names(path)
    missing = {}
    for name in names:
        fn = ns.get(name)
        code = getattr(fn, "__code__", None)
        if code is None:
            continue
        stack, seen = [code], set()
        while stack:
            c = stack.pop()
            if id(c) in seen:
                continue
            seen.add(id(c))
            for used in c.co_names:
                if used in driver and used not in ns:
                    missing.setdefault(name, set()).add(used)
            stack.extend(k for k in c.co_consts if hasattr(k, "co_names"))
    assert not missing, (
        "exec'd helper set is not closed -- add these to the name list:\n  "
        + "\n  ".join("%s needs %s" % (k, sorted(v)) for k, v in sorted(missing.items())))


def _defs(path, names):
    found = {n.name: n for n in ast.parse(_src(path)).body
             if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = sorted(set(names) - set(found))
    assert not missing, "%s missing: %s" % (os.path.basename(path), missing)
    return found


@pytest.fixture(scope="module")
def H():
    defs = _defs(_LISA, HELPERS)
    mod = ast.Module(body=[defs[n] for n in HELPERS], type_ignores=[])
    ns = {"numpy": np, "np": np, "SamplerOutputMixin": _SamplerOutputMixin}
    exec(compile(ast.fix_missing_locations(mod), "mcerr", "exec"), ns)
    _assert_helper_set_is_closed(ns, HELPERS)
    return ns


class _S(object):
    """Minimal sampler: pooling only needs identity_convert."""
    def identity_convert(self, x):
        return x


def _rec(lnL, n=None):
    lnL = np.asarray(lnL, dtype=float)
    n = len(lnL) if n is None else n
    return {'log_integrand': lnL.copy(),
            'log_joint_prior': np.zeros(n),
            'log_joint_s_prior': np.zeros(n),
            'x': np.linspace(0.0, 1.0, n)}


def _lw(H, rec):
    return H['ln_weights_from_rvs'](rec)


# ------------------------------------------------------------------------ _extract_mc_diag
def test_extract_mc_diag_pulls_the_four_diagnostics(H):
    dd = {'pareto_khat': 0.9, 'sigma_lnZ_block': 0.3, 'n_ESS': 12.0, 'lnZ_ci90': [1, 2, 3]}
    assert H['_extract_mc_diag'](dd) == (0.9, 0.3, 12.0, [1, 2, 3])


@pytest.mark.parametrize("dd", [None, "not a dict", {}, 7])
def test_extract_mc_diag_tolerates_anything(H, dd):
    assert H['_extract_mc_diag'](dd) == (None, None, None, None)


# --------------------------------------------------------------- pooling: the basic contract
def test_single_replica_is_returned_unchanged(H):
    r = _rec([0.0, 0.0])
    assert H['_pool_replica_rvs']([r], _S()) is r


def test_no_replicas_gives_an_empty_record(H):
    assert H['_pool_replica_rvs']([], _S()) == {}


def test_pooled_record_concatenates_every_replica(H):
    reps = [_rec([0.0] * 3), _rec([0.0] * 4)]
    out = H['_pool_replica_rvs'](reps, _S(), rep_lnZ=[0.0, 0.0])
    assert H['_rvs_len'](out) == 7, "pooling dropped or duplicated rows"


def test_each_block_contributes_its_own_evidence_over_K(H):
    """Block k's weights must sum to Z_k/K -- that is what makes the pool match lnZ."""
    reps = [_rec([0.0] * 4), _rec([0.0] * 6)]
    rep_lnZ = [0.0, np.log(3.0)]                      # Z = 1 and 3
    out = H['_pool_replica_rvs'](reps, _S(), rep_lnZ=rep_lnZ)
    w = np.exp(_lw(H, out))
    K = 2
    b0, b1 = w[:4].sum(), w[4:].sum()
    assert np.isclose(b0, 1.0 / K, rtol=1e-6), b0
    assert np.isclose(b1, 3.0 / K, rtol=1e-6), b1
    assert np.isclose(w.sum(), (1.0 + 3.0) / K, rtol=1e-6), "pooled Z is not the linear mean"


# ------------------------------------------------- constraint (c): the per-replica sequence
def test_a_resampled_block_is_flattened_and_a_raw_one_is_not(H):
    """The MIXTURE case, which one global boolean cannot express.

    Replica 0 was fair-drawn -- its rows are already an equal-weight posterior draw, so it
    must contribute CONSTANT weights.  Replica 1 was not, so its genuine importance weights
    must survive.
    """
    reps = [_rec([0.0, 3.0, 6.0]), _rec([0.0, 3.0, 6.0])]
    out = H['_pool_replica_rvs'](reps, _S(), rep_lnZ=[0.0, 0.0],
                                already_resampled=[True, False])
    w = np.exp(_lw(H, out))
    b0, b1 = w[:3], w[3:]
    assert np.allclose(b0, b0[0]), "the fair-drawn block was not flattened (w^2 double-weighting)"
    assert not np.allclose(b1, b1[0]), "the raw block was flattened, discarding real weights"


def test_a_global_boolean_would_get_the_mixture_wrong(H):
    """Pins that the sequence and the boolean genuinely differ, so the test above has teeth."""
    reps = [_rec([0.0, 3.0, 6.0]), _rec([0.0, 3.0, 6.0])]
    seq = np.exp(_lw(H, H['_pool_replica_rvs'](reps, _S(), rep_lnZ=[0.0, 0.0],
                                               already_resampled=[True, False])))
    allT = np.exp(_lw(H, H['_pool_replica_rvs'](reps, _S(), rep_lnZ=[0.0, 0.0],
                                                already_resampled=True)))
    allF = np.exp(_lw(H, H['_pool_replica_rvs'](reps, _S(), rep_lnZ=[0.0, 0.0],
                                                already_resampled=False)))
    assert not np.allclose(seq, allT) and not np.allclose(seq, allF)


@pytest.mark.parametrize("flags", [True, False, [True, True], [False, False]])
def test_uniform_flags_still_work_in_either_form(H, flags):
    out = H['_pool_replica_rvs']([_rec([0.0, 1.0]), _rec([0.0, 1.0])], _S(),
                                 rep_lnZ=[0.0, 0.0], already_resampled=flags)
    assert H['_rvs_len'](out) == 4


# ------------------------------------------------------- constraint (b): lockstep filtering
def test_empty_replicas_are_dropped_in_lockstep_with_their_metadata(H):
    """An empty record in the middle must not shift later blocks onto the wrong lnZ.

    Replica 1 is empty. If the filter ran on rep_rvs alone, block 2 would be weighted with
    replica 1's evidence.
    """
    reps = [_rec([0.0] * 3), {}, _rec([0.0] * 3)]
    rep_lnZ = [0.0, -99.0, np.log(3.0)]
    out = H['_pool_replica_rvs'](reps, _S(), rep_lnZ=rep_lnZ)
    w = np.exp(_lw(H, out))
    assert H['_rvs_len'](out) == 6
    K = 2                                   # the empty replica is gone, so K is 2 not 3
    assert np.isclose(w[:3].sum(), 1.0 / K, rtol=1e-6)
    assert np.isclose(w[3:].sum(), 3.0 / K, rtol=1e-6), \
        "the surviving block was weighted with the dropped replica's evidence"


def test_lockstep_applies_to_the_resampled_flags_too(H):
    reps = [{}, _rec([0.0, 3.0, 6.0]), _rec([0.0, 3.0, 6.0])]
    out = H['_pool_replica_rvs'](reps, _S(), rep_lnZ=[-99.0, 0.0, 0.0],
                                 already_resampled=[False, True, False])
    w = np.exp(_lw(H, out))
    assert np.allclose(w[:3], w[0]), "the flags did not shift with the records"
    assert not np.allclose(w[3:], w[3])


# ------------------------------------------------------------------------ fallbacks
def test_a_record_without_a_sampling_prior_column_falls_back_to_the_first_replica(H):
    reps = [{'x': np.zeros(3)}, {'x': np.zeros(3)}]
    out = H['_pool_replica_rvs'](reps, _S(), rep_lnZ=[0.0, 0.0])
    assert out is reps[0], "fallback must return an INPUT record, so _did_pool is False"


def test_fallback_identity_is_what_the_driver_keys_on():
    """`_did_pool = not any(_pooled_rvs is _r for _r in _rep_rvs)` -- identity, not length."""
    src = _src(_LISA)
    assert "_did_pool = not any(_pooled_rvs is _r for _r in _rep_rvs)" in src


# ------------------------------------------------------------------- cached weights
def test_cached_weights_are_recomputed_from_the_canonical_columns(H):
    """Consumers PREFER a cached log_weights column; a stale one silently undoes the pooling."""
    reps = [_rec([0.0, 1.0]), _rec([0.0, 1.0])]
    for r in reps:
        r['log_weights'] = np.full(2, 999.0)
    out = H['_pool_replica_rvs'](reps, _S(), rep_lnZ=[0.0, 0.0])
    assert not np.allclose(out['log_weights'], 999.0), "stale cached weights survived pooling"
    assert np.allclose(out['log_weights'], _lw(H, out))


# ------------------------------------------------------------------ source-level wiring
def _helper_src(name):
    src = _src(_LISA)
    a = src.index("def %s(" % name)
    return src[a:src.index("\ndef ", a + 1)]


def test_the_driver_passes_the_per_replica_sequence_not_the_cli_flag():
    """The CLI flag is not the question: the draw is skipped per pass when it would not shrink."""
    fn = _helper_src("_maybe_replicate_for_mc_error")
    assert "already_resampled=_rep_fairdraw" in fn
    assert "_rep_fairdraw = [bool(getattr(sampler, '_rvs_is_fairdraw', False))]" in fn
    assert "already_resampled=opts.fairdraw_extrinsic_output" not in fn, \
        "the pooler was handed the CLI flag instead of what each pass actually did"


def test_the_pooled_marker_is_set_only_when_pooling_happened():
    fn = _helper_src("_maybe_replicate_for_mc_error")
    assert "sampler._rvs_is_pooled = True" in fn
    assert "if _did_pool:" in fn


def test_rvs_is_pooled_is_reset_on_entry_of_both_analyze_event_variants():
    """Cleared only on the happy path, it survives the pooled gate's raise (Finding 7).

    POSITION-AWARE, not merely presence-aware.  An adversarial review pointed out that an
    earlier version used `"... = False" in ast.unparse(fn)`, which passes just as happily if
    the reset is MOVED to after the replica call -- reintroducing exactly the bug, since the
    pooled gate raises and the caller's `except` swallows it.  So: the reset must be the first
    statement region of the function and must precede the replica call.
    """
    tree = ast.parse(_src(_LISA))
    for n in tree.body:
        if not (isinstance(n, ast.FunctionDef)
                and n.name in ("analyze_event", "analyze_event_LISA")):
            continue
        resets = [st.lineno for st in ast.walk(n)
                  if isinstance(st, ast.Assign)
                  and any(isinstance(t, ast.Attribute) and t.attr == "_rvs_is_pooled"
                          and isinstance(t.value, ast.Name) and t.value.id == "sampler"
                          for t in st.targets)]
        assert resets, "%s never resets the pooled marker" % n.name
        calls = [c.lineno for c in ast.walk(n)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                 and c.func.id == "_maybe_replicate_for_mc_error"]
        assert calls, "%s never runs the replica helper" % n.name
        assert min(resets) < min(calls), (
            "%s resets _rvs_is_pooled at line %d, AFTER the replica helper at %d; the pooled "
            "gate raises, the caller swallows it, and the marker survives into the next event"
            % (n.name, min(resets), min(calls)))
        # and it must be one of the FIRST STATEMENTS, not buried behind work that can fail.
        # Counted in statements, not lines: the reset carries a long explanatory comment, so a
        # line-distance rule fails on the correct code (it did, on the first attempt).
        top = [st for st in n.body[:4]]
        assert any(isinstance(st, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == "_rvs_is_pooled"
                           for t in st.targets)
                   for st in top), (
            "%s does not reset the pooled marker within its first 4 statements; it must "
            "happen on entry, before anything can raise" % n.name)

def test_block_kish_neff_is_used_when_blocks_were_flattened():
    """Kish over a flattened pooled record just reports the EXPORT SIZE (5K by default)."""
    fn = _helper_src("_maybe_replicate_for_mc_error")
    assert "_blocks_flattened" in fn
    assert "numpy.sum(_Zk[_ok]) ** 2 / numpy.sum(_Zk[_ok] ** 2 / _nk[_ok])" in fn


def test_collapse_status_is_the_OR_over_pooled_replicas():
    fn = _helper_src("_maybe_replicate_for_mc_error")
    assert "_any_collapsed = any(_rep_collapsed)" in fn
    assert "n_replicas_pooled" in fn and "n_replicas_collapsed" in fn


# ------------------------------------------------------- anti-drift vs the main driver
def _normalized(fn):
    node = ast.parse(ast.unparse(fn)).body[0] if hasattr(ast, "unparse") else fn
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return ast.dump(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])))


@pytest.mark.parametrize("name", ["_pool_replica_rvs", "_extract_mc_diag"])
def test_ported_helper_is_identical_to_the_main_driver(name):
    lisa = _defs(_LISA, [name])[name]
    main = [n for n in ast.walk(ast.parse(_src(_MAIN)))
            if isinstance(n, ast.FunctionDef) and n.name == name][0]
    assert _normalized(lisa) == _normalized(main), \
        "%s has drifted between the two drivers (docstrings excluded)" % name


# ==========================================================================================
# BEHAVIOURAL coverage of _maybe_replicate_for_mc_error.
#
# Everything above this line tests _pool_replica_rvs (a pure function) behaviourally and the
# ORCHESTRATION only at source level.  An adversarial review planted five bugs in the
# orchestration -- moving the _rvs_is_pooled reset off entry, dedenting the pooled-marker
# assignment out of `if _did_pool`, inverting `if _blocks_flattened`, neutering the POOLED
# collapse gate, and deleting the collapse-status OR -- and all five passed 220/220 tests.
# Substring and AST-name checks cannot see any of that.  So: execute the helper.
# ==========================================================================================

ORCH = ['_rvs_lnL_convention', 'ln_weights_from_rvs', '_rvs_len',
        '_rvs_record_for', '_sampler_keeps_records', '_internal_record_of',
        '_rebound_record', '_lw_of', '_lnZ_of_rvs',
        '_kish_neff_of_rvs', '_extract_mc_diag', '_pool_replica_rvs',
        '_maybe_save_av_state', '_reject_if_collapsed', '_report_and_gate_collapse',
        '_maybe_replicate_for_mc_error']


class _Collapse(Exception):
    pass


class _AVmod(object):
    LiveVolumeCollapse = _Collapse


class _RepSampler(object):
    """Sampler whose integrate() returns a scripted list of replica results."""

    def __init__(self, first_rvs, replicas, fairdraw_first=False):
        self._rvs = first_rvs
        self._rvs_is_fairdraw = fairdraw_first
        self._rvs_is_pooled = False
        self._warm_seed_reserve = None
        self.params_ordered = ['x']
        self._queue = list(replicas)          # [(res,var,neff,dd,rvs,fairdraw), ...]
        self.saved = None

    def identity_convert(self, x):
        return x

    def save_state(self, path):
        self.saved = path

    def integrate(self, fn, *a, **kw):
        res, var, neff, dd, rvs, fd = self._queue.pop(0)
        self._rvs = rvs
        self._rvs_is_fairdraw = fd
        return res, var, neff, dd


def _load_orch(**optkw):
    base = dict(mc_error_replicas=0, mc_error_sigma_trigger=1e9,
                mc_error_khat_trigger=0.7, mc_error_ess_trigger=0.0,
                reject_collapsed_live_volume=False, internal_use_lnL=True,
                sampler_method='AV', sampler_save_state=None)
    base.update(optkw)
    defs = _defs(_LISA, ORCH)
    mod = ast.Module(body=[defs[n] for n in ORCH], type_ignores=[])
    ns = {"numpy": np, "np": np, "SamplerOutputMixin": _SamplerOutputMixin, "mcsamplerAdaptiveVolume": _AVmod,
          "mcsampler_AV_ok": True, "rvs_integrand_is_lnL": False,
          "opts": type("O", (), base)()}
    exec(compile(ast.fix_missing_locations(mod), "orch", "exec"), ns)
    _assert_helper_set_is_closed(ns, ORCH)
    return ns


def _run_orch(ns, sampler, dict_return, log_res=0.0, sigma=5.0, neff=1.0):
    return ns['_maybe_replicate_for_mc_error'](
        sampler, np.exp(log_res), 1.0, neff, dict_return, log_res, sigma,
        lambda *a, **k: None, (), {'neff': 100.0})


def test_no_trigger_means_no_replicas_and_nothing_changed():
    ns = _load_orch(mc_error_replicas=0)
    s = _RepSampler(_rec([0.0, 0.0]), [])
    out = _run_orch(ns, s, {}, sigma=0.001)
    assert out[0:3] == (1.0, 1.0, 1.0) or out[2] == 1.0
    assert s._rvs_is_pooled is False


def test_sigma_trigger_runs_replicas_and_pools_them():
    ns = _load_orch(mc_error_replicas=2, mc_error_sigma_trigger=0.1)
    s = _RepSampler(_rec([0.0] * 4), [
        (1.0, 1.0, 5.0, {}, _rec([0.0] * 4), False),
        (1.0, 1.0, 5.0, {}, _rec([0.0] * 4), False)])
    out = _run_orch(ns, s, {}, sigma=5.0)
    assert not s._queue, "the replica loop did not run the requested replicas"
    assert s._rvs_is_pooled is True, "the pooled marker was not set"
    assert _rvs_len(s._rvs) == 12, "the pooled record is not the concatenation"
    assert out[2] > 0


class _RecordingRepSampler(_RepSampler):
    """A _RepSampler that PARTICIPATES in the record scheme (samples/set_samples)."""

    def __init__(self, *a, **kw):
        _RepSampler.__init__(self, *a, **kw)
        self._rvs_record = None

    def samples(self):
        return self._rvs_record

    def set_samples(self, record):
        self._rvs_record = record
        return record


def test_pooling_clears_a_stale_record_on_a_record_keeping_sampler():
    """The LISA replica path publishes NO pooled record, so it must publish none at all.

    The main driver builds an _RvsRecord.pooled() here; this driver does not collect the
    per-replica records to build one from, so the weight route falls back to the flags.
    That fallback is correct -- but the record left on the sampler describes the PRE-POOL
    columns, and it is otherwise declined only because _rvs_record_for compares by
    identity and `_rvs` happens to become a new dict.  Reading a per-pass record as if it
    described the mixture would mix the replicas by row count instead of by evidence,
    which is the exact defect the pooled weights exist to prevent.
    """
    ns = _load_orch(mc_error_replicas=2, mc_error_sigma_trigger=0.1)
    s = _RecordingRepSampler(_rec([0.0] * 4), [
        (1.0, 1.0, 5.0, {}, _rec([0.0] * 4), False),
        (1.0, 1.0, 5.0, {}, _rec([0.0] * 4), False)])

    class _StaleRecord(object):
        internal = False
        columns = s._rvs                      # describes the PRE-POOL columns
    s.set_samples(_StaleRecord())

    _run_orch(ns, s, {}, sigma=5.0)
    assert s._rvs_is_pooled is True, "precondition: this test only means anything if it pooled"
    assert s.samples() is None, \
        "a pre-pool record survived the pooling step: it would be read as the mixture"


def _rvs_len(rec):
    for v in rec.values():
        return len(np.atleast_1d(np.asarray(v)).ravel())
    return 0


def test_the_pooled_collapse_gate_fires_on_a_collapsed_REPLICA():
    """Mutation D: a healthy first run plus a collapsed replica must still be rejected.

    This is the whole reason the gate is called twice.  The first-run gate sees nothing wrong.
    """
    ns = _load_orch(mc_error_replicas=1, mc_error_sigma_trigger=0.1,
                    reject_collapsed_live_volume=True)
    s = _RepSampler(_rec([0.0] * 4), [
        (1.0, 1.0, 5.0, {'live_volume_collapsed': True, 'collapse_reason': 'replica died'},
         _rec([0.0] * 4), False)])
    with pytest.raises(_Collapse) as e:
        _run_orch(ns, s, {'live_volume_collapsed': False}, sigma=5.0)
    assert "pooled over" in str(e.value)


def test_collapse_status_is_folded_back_as_the_OR():
    """Mutation E: the sidecar must not record collapsed=false for a tainted pool."""
    ns = _load_orch(mc_error_replicas=1, mc_error_sigma_trigger=0.1)
    dd = {'live_volume_collapsed': False}
    s = _RepSampler(_rec([0.0] * 4), [
        (1.0, 1.0, 5.0, {'live_volume_collapsed': True, 'collapse_reason': 'replica died'},
         _rec([0.0] * 4), False)])
    out = _run_orch(ns, s, dd, sigma=5.0)
    got = out[5]
    assert got['live_volume_collapsed'] is True, "a collapsed replica was not folded in"
    assert got['n_replicas_pooled'] == 2 and got['n_replicas_collapsed'] == 1
    assert 'replica died' in got['collapse_reason']


def test_the_pooled_marker_is_not_set_when_pooling_fell_back():
    """Mutation B: a fallback returns an INPUT record, which is not a pooled mixture.

    Records with no sampling-prior column make _pool_replica_rvs return replica 0 unchanged.
    """
    bad = {'x': np.zeros(3), 'log_integrand': np.zeros(3)}       # no *_joint_s_prior
    ns = _load_orch(mc_error_replicas=1, mc_error_sigma_trigger=0.1)
    s = _RepSampler(dict(bad), [(1.0, 1.0, 5.0, {}, dict(bad), False)])
    _run_orch(ns, s, {}, sigma=5.0)
    assert s._rvs_is_pooled is False, \
        "the pooled marker was set even though pooling fell back to an input record"


def test_flattened_blocks_report_block_kish_not_the_export_row_count():
    """Mutation C: with fair-drawn replicas the pooled Kish is just the row count.

    Two agreeing replicas of n_eff 5 should give a pooled n_eff near their sum (10), not the
    12 rows of the export.
    """
    ns = _load_orch(mc_error_replicas=1, mc_error_sigma_trigger=0.1)
    # AGREEING replicas: with --internal-use-lnL the replica's lnZ IS its `res`, so the first
    # run's log_res must match it or the two disagree and block-Kish correctly falls below the
    # sum.  (An earlier version of this test used 0.0 vs 1.0 and measured 8.24 -- the code was
    # right and the setup was wrong, which is itself evidence the assertion is sensitive.)
    s = _RepSampler(_rec([0.0] * 6), [(1.0, 1.0, 5.0, {}, _rec([0.0] * 6), True)],
                    fairdraw_first=True)
    out = _run_orch(ns, s, {}, log_res=1.0, sigma=5.0, neff=5.0)
    neff_out = float(out[2])
    assert 9.0 < neff_out < 11.0, (
        "expected block-Kish ~sum(neff)=10 for agreeing replicas, got %r (12 would be the "
        "exported row count)" % neff_out)


def test_disagreeing_replicas_report_less_than_the_sum():
    """The property block-Kish exists for: disagreement must SHOW UP as lower n_eff."""
    ns = _load_orch(mc_error_replicas=1, mc_error_sigma_trigger=0.1)
    s = _RepSampler(_rec([0.0] * 6), [(1.0, 1.0, 5.0, {}, _rec([0.0] * 6), True)],
                    fairdraw_first=True)
    # replica lnZ far below the first run -> Z_k wildly unequal -> pooled neff -> ~5
    out = _run_orch(ns, s, {}, log_res=20.0, sigma=5.0, neff=5.0)
    assert float(out[2]) < 9.0, "disagreeing replicas still reported the full sum"


# ==========================================================================================
# The XML export of a pooled record.
#
# The pool is deliberately weighted BETWEEN blocks (Z_k/K), and the SimInspiral export keeps no
# column carrying that: xmlutils maps joint_prior/joint_s_prior onto alpha2/alpha3, which the
# ILE export overwrites with zeros, and the log_joint_* columns the pool uses map to nothing.
# So the rows must be re-drawn to equal weight first, or downstream mixes the replicas by ROW
# COUNT instead of by evidence -- discarding the disagreement the replicas were run to measure.
# ==========================================================================================

EXPORT = ['_rvs_lnL_convention', 'ln_weights_from_rvs', '_rvs_len', '_rvs_is_equal_weight',
          '_rvs_record_for', '_sampler_keeps_records', '_internal_record_of',
          '_rebound_record', '_lw_of',
          'ln_weights_for_posterior', '_export_rvs_equal_weight']


@pytest.fixture(scope="module")
def EW():
    defs = _defs(_LISA, EXPORT)
    mod = ast.Module(body=[defs[n] for n in EXPORT], type_ignores=[])
    ns = {"numpy": np, "np": np, "SamplerOutputMixin": _SamplerOutputMixin}
    exec(compile(ast.fix_missing_locations(mod), "export", "exec"), ns)
    _assert_helper_set_is_closed(ns, EXPORT)
    return ns


class _ES(_S):
    """Minimal sampler carrying the provenance markers the export helper keys on."""
    def __init__(self, pooled=True, fairdraw=True):
        self._rvs_is_pooled = pooled
        self._rvs_is_fairdraw = fairdraw


def test_a_record_that_was_never_pooled_is_exported_untouched(EW):
    """Identity, not equality: no non-replica run may change shape because of this path."""
    r = _rec([0.0, 1.0, 2.0])
    assert EW['_export_rvs_equal_weight'](r, _ES(pooled=False)) is r


def test_the_pooled_export_mixes_replicas_by_EVIDENCE_not_by_row_count(EW):
    """Block 1 has 3x the evidence of block 0 at equal row counts, so it must dominate.

    Both blocks are flat (each is its own equal-weight draw), which is exactly the case where
    the row count carries no evidence information at all: unconverted, the XML would report the
    two replicas as an even mixture.
    """
    rec = {'log_integrand': np.zeros(8),
           'log_joint_prior': np.zeros(8),
           # weights e^0 in block 0, e^log(3)=3 in block 1
           'log_joint_s_prior': np.concatenate([np.zeros(4), -np.log(3.0) * np.ones(4)]),
           'x': np.concatenate([np.zeros(4), np.ones(4)])}
    np.random.seed(7)
    out = EW['_export_rvs_equal_weight'](rec, _ES())
    frac = float(np.mean(out['x']))                       # share of rows from block 1
    assert 0.6 < frac < 0.9, (
        "pooled export mixed the replicas at %.2f; 0.5 is mixing by row count, 0.75 is the "
        "evidence share" % frac)
    assert _rvs_len(out) <= 8, "the export claims more rows than the pool held"


def test_an_unusable_pooled_record_is_returned_rather_than_mangled(EW):
    bad = {'x': np.zeros(4)}                              # no weight components at all
    assert EW['_export_rvs_equal_weight'](bad, _ES()) is bad


def test_both_xml_export_paths_convert_before_consuming_the_pool():
    """Source-level: the conversion must sit on the deepcopy, ahead of every consumer.

    Including resample_samples*, which picks a time per row and so assumes the rows already are
    the posterior -- converting after it would leave that draw made from the wrong mixture.
    """
    src = _src(_LISA)
    copies = [i for i in range(len(src)) if src.startswith("copy.deepcopy(sampler._rvs)", i)]
    assert len(copies) == 2, "expected two --save-samples export blocks, found %d" % len(copies)
    for i in copies:
        end = src.index("append_samples_to_xmldoc", i)      # the block this deepcopy feeds
        block = src[i:end]
        assert "_export_rvs_equal_weight(samples, sampler" in block, \
            "an XML export path consumes the pooled record without converting it"
        assert block.index("_export_rvs_equal_weight(samples, sampler") \
            < block.index("resample_time_marginalization"), \
            "the conversion happens after the time resampler has already drawn from the rows"


# --------------------------------------------- cold replicas on the standalone GMM sampler
class _GMMSampler(_RepSampler):
    """mcsamplerEnsemble's shape: an `integrator` attribute and NONE of the reset methods.

    Warmth reaches a replica by two routes there, so a "cold" replica needs both cut:
    integrate() transfers the previous integrator's fitted models into the new one, and the
    gmm_dict it is handed is the caller's object, which the fit writes its models back into.
    """

    def __init__(self, first_rvs, replicas):
        _RepSampler.__init__(self, first_rvs, replicas)
        self.integrator = object()            # the first run's fitted integrator
        self.seen_integrator = "unset"
        self.seen_gmm = None

    def integrate(self, fn, *a, **kw):
        self.seen_integrator = self.integrator
        self.seen_gmm = dict(kw.get('gmm_dict') or {})
        return _RepSampler.integrate(self, fn, *a, **kw)


def test_a_standalone_GMM_replica_is_cold_in_both_warm_start_channels():
    """Sharing the first run's proposal keeps the same mode missed in every replica.

    The between-replica scatter is then a measure of the draws alone, understating exactly the
    MC error the replicas were run to expose.
    """
    ns = _load_orch(mc_error_replicas=1, mc_error_sigma_trigger=0.1, sampler_method='GMM')
    fitted, seeded = object(), object()
    sky, phase = ('right_ascension', 'declination'), ('psi', 'phi_orb')
    gmm_dict = {sky: fitted, phase: seeded}
    gmm_adapt = {sky: True, phase: False}
    s = _GMMSampler(_rec([0.0] * 4), [(1.0, 1.0, 5.0, {}, _rec([0.0] * 4), False)])
    ns['_maybe_replicate_for_mc_error'](
        s, 1.0, 1.0, 1.0, {}, 0.0, 5.0, lambda *a, **k: None, (),
        {'neff': 100.0, 'gmm_dict': gmm_dict, 'gmm_adapt': gmm_adapt})
    assert s.seen_integrator is None, \
        "the replica ran with the previous integrator, whose fitted models integrate() transfers"
    assert s.seen_gmm[sky] is None, \
        "the replica inherited the first run's fit through the aliased gmm_dict"
    assert s.seen_gmm[phase] is seeded, (
        "the fixed non-adapting proposal was blanked; _train skips that group, so it would "
        "have no model at all and the group would degrade to uniform sampling")


def test_the_GMM_cold_reset_does_not_touch_samplers_that_have_their_own():
    """A portfolio owns clear_warm_state/reset_adaptation and no `integrator`: unchanged."""
    class _Portfolio(_RepSampler):
        def __init__(self, *a, **kw):
            _RepSampler.__init__(self, *a, **kw)
            self.cleared = 0
            self.seen_gmm = None

        def reset_adaptation(self):
            self.cleared += 1

        def integrate(self, fn, *a, **kw):
            self.seen_gmm = dict(kw.get('gmm_dict') or {})
            return _RepSampler.integrate(self, fn, *a, **kw)

    ns = _load_orch(mc_error_replicas=1, mc_error_sigma_trigger=0.1)
    sky = ('right_ascension', 'declination')
    fitted = object()
    s = _Portfolio(_rec([0.0] * 4), [(1.0, 1.0, 5.0, {}, _rec([0.0] * 4), False)])
    ns['_maybe_replicate_for_mc_error'](
        s, 1.0, 1.0, 1.0, {}, 0.0, 5.0, lambda *a, **k: None, (),
        {'neff': 100.0, 'gmm_dict': {sky: fitted}, 'gmm_adapt': {sky: True}})
    assert s.cleared == 1, "the portfolio's own reset stopped being called"
    assert s.seen_gmm[sky] is fitted, \
        "the GMM branch reached a sampler that rebuilds its members from their setup arguments"


def test_a_failing_replica_is_skipped_not_fatal():
    class _Boom(_RepSampler):
        def integrate(self, fn, *a, **kw):
            raise RuntimeError("replica exploded")

    ns = _load_orch(mc_error_replicas=1, mc_error_sigma_trigger=0.1)
    s = _Boom(_rec([0.0] * 4), [])
    out = _run_orch(ns, s, {}, sigma=5.0)
    assert out is not None and out[2] == 1.0
