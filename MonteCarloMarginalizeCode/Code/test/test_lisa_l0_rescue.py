#!/usr/bin/env python
"""
Tests for the L0 auto-rescue ported into the LISA ILE driver
(bin/integrate_likelihood_extrinsic_batchmode_lisa) from the main driver.

WHY IT BELONGS IN LISA.  The rescue targets the high-SNR n_eff LOTTERY: a large fraction of
independent AV/portfolio runs collapse to n_eff ~ 1 by contracting onto the wrong spot, and
the rescue re-seeds such a run from the peak it did find.  LISA MBHB are high-SNR by
construction, so this is the regime, not an edge case.  The sampler-side machinery
(`build_warm_seed`, `lnZ_from_reserve`, the reserve itself) already lives in
RIFT/integrators/ and therefore already reached LISA; only the driver-side wiring was missing.

ONE DELIBERATE STRUCTURAL DIVERGENCE.  The main driver inlines the rescue in its single
`analyze_event`.  This driver has TWO -- `analyze_event_LISA` (with --LISA) and
`analyze_event` (the fallback) -- so the block was lifted into `_maybe_l0_rescue` and both
call it.  That is a divergence in SHAPE, not behaviour, and it buys something main does not
have: the reject gate becomes unit-testable.  The audit notes that in main these call sites
"cannot be exercised from a unit test" because analyze_event needs data, PSDs and a waveform.
Here the gate is a function of its arguments, so the tests below drive it directly.

ORDERING IS LOAD-BEARING (see test_rescue_runs_before_the_no_result_guard).  In main the
`if not(res): raise` guard sits ~200 lines below the integrate call and the ordering is
implicit.  In this driver it is immediately after, so the rescue had to be inserted BETWEEN
them: a degenerate early termination returns (None,None,None,None) and is the STRONGEST
rescue trigger, so raising on it first would skip exactly the case the rescue exists for.
"""

import ast
import os

import numpy as np
from RIFT.integrators.rvs_record import SamplerOutputMixin as _SamplerOutputMixin
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LISA = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')
_MAIN = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

# Helpers ported verbatim from the main driver.  _maybe_l0_rescue is NOT in this list: it is
# the LISA-only wrapper, and has no counterpart to be identical to.
PORTED = ['_lnZ_of_rvs', '_kish_neff_of_rvs', '_lnZ_of_reserve_or_rvs',
          '_snapshot_pass_state', '_restore_pass_state',
          '_warm_seed_reserve_for', '_warm_seed_geometry', '_clear_warm_state']

# Everything the exec'd namespace needs, in dependency order.
# The record accessors are here because the PORTED helpers call them by name:
# _snapshot_pass_state/_restore_pass_state thread the sampler's RvsRecord, and
# ln_weights_for_posterior reads it.  Leaving one out is a NameError at exec time,
# not a missing assertion -- which is exactly how this list is meant to fail.
_DEPS = ['_rvs_lnL_convention', 'ln_weights_from_rvs', '_rvs_len',
         '_rvs_is_export_resample', '_rvs_is_equal_weight',
         '_rvs_record_for', '_sampler_keeps_records', '_internal_record_of',
         '_rebound_record', '_lw_of', 'ln_weights_for_posterior']



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


def _assert_helper_set_is_closed(ns, names, path):
    """Fail LOUDLY if an exec'd helper calls a driver helper that was not exec'd with it.

    The same omission in test_lisa_mc_error_replicas.py did NOT raise: _lnZ_of_rvs catches
    broadly and returns None, so a missing name read as "no evidence" and the pooled
    weights silently collapsed to 1/K.  Kept in all three LISA harnesses so the next
    ported helper cannot reintroduce it here instead.
    """
    driver = _driver_def_names(path)
    missing = {}
    for name in names:
        code = getattr(ns.get(name), "__code__", None)
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
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    found = {n.name: n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = sorted(set(names) - set(found))
    assert not missing, "%s is missing: %s" % (os.path.basename(path), missing)
    return found


class _FakeAV(object):
    """Stand-in for RIFT.integrators.mcsamplerAdaptiveVolume inside the helpers."""
    lnZ_value = None
    seed_info = {'puffed': False, 'n_core': 3, 'rank_core': 3, 'dim': 3,
                 'rank_final': 3, 'n_puff': 0, 'puff_scale': 'auto'}

    @classmethod
    def lnZ_from_reserve(cls, reserve):
        return cls.lnZ_value

    @classmethod
    def build_warm_seed(cls, cols, lnL, lo, hi, axes, **kw):
        return np.asarray(cols, dtype=float), dict(cls.seed_info)


class _Opts(object):
    sampler_method = 'AV'
    sampler_warmstart_retry_neff = 5.0
    sampler_l0_rescue_reject_dlnZ = 3.0
    sampler_l0_rescue_accept_truncated = False
    sampler_l0_rescue_puff_scale = 'auto'
    sampler_l0_rescue_puff_width_frac = 0.005
    sampler_l0_rescue_puff_factor = 2.0
    sampler_sequential_warmstart_deltalnL = 15.0

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _load(opts=None, av=None):
    """Exec the rescue helpers out of the LISA driver with injected globals."""
    names = _DEPS + PORTED + ['_maybe_l0_rescue']
    defs = _defs(_LISA, names)
    mod = ast.Module(body=[defs[n] for n in names], type_ignores=[])
    ns = {"numpy": np, "np": np, "SamplerOutputMixin": _SamplerOutputMixin,
          "opts": opts if opts is not None else _Opts(),
          "mcsamplerAdaptiveVolume": av if av is not None else _FakeAV}
    exec(compile(ast.fix_missing_locations(mod), "lisa_l0_helpers", "exec"), ns)
    _assert_helper_set_is_closed(ns, names, _LISA)
    return ns


@pytest.fixture
def H():
    return _load()


# ------------------------------------------------------------------------------ fake sampler
class _Sampler(object):
    def __init__(self, rvs=None, reserve=None, params=('a', 'b'), members=None,
                 integrate_result=None, raise_in_integrate=False):
        self._rvs = rvs if rvs is not None else {}
        self._warm_seed_reserve = reserve
        self.params_ordered = list(params)
        self.llim = {p: 0.0 for p in self.params_ordered}
        self.rlim = {p: 1.0 for p in self.params_ordered}
        self.portfolio_realizations = members or []
        self._warm = "stale"
        self._warm_applied = True
        self._integrate_result = integrate_result
        self._raise_in_integrate = raise_in_integrate
        self.bootstrapped = None
        self.warm_rvs = None

    def identity_convert(self, x):
        return x

    def bootstrap_from_samples(self, seed, cover_frac=0.0):
        self.bootstrapped = (np.asarray(seed), cover_frac)

    def integrate(self, fn, *a, **kw):
        if self._raise_in_integrate:
            # Repopulate _rvs IN PLACE first, then raise: this is the dangerous shape --
            # the assignment at the call site never completes, so res/var/neff still hold
            # the COLD pass while _rvs holds the WARM samples.
            self._rvs = dict(self.warm_rvs or {})
            raise RuntimeError("warm pass exploded")
        if self.warm_rvs is not None:
            self._rvs = dict(self.warm_rvs)
        return self._integrate_result


def _rec(lnL, n=None):
    lnL = np.asarray(lnL, dtype=float)
    n = len(lnL) if n is None else n
    return {'log_integrand': lnL,
            'log_joint_prior': np.zeros(n),
            'log_joint_s_prior': np.zeros(n),
            'a': np.linspace(0.1, 0.9, n), 'b': np.linspace(0.2, 0.8, n)}


# ------------------------------------------------------------------------------- lnZ helpers
def test_lnZ_pooled_is_the_sum_and_unpooled_is_the_mean(H):
    r = _rec([0.0, 0.0, 0.0, 0.0])
    pooled = H['_lnZ_of_rvs'](r, already_pooled=True)
    single = H['_lnZ_of_rvs'](r, already_pooled=False)
    assert np.isclose(pooled, np.log(4.0))
    assert np.isclose(single, 0.0)
    assert np.isclose(pooled - single, np.log(4.0))


def test_lnZ_returns_none_when_weights_cannot_be_rebuilt(H):
    assert H['_lnZ_of_rvs']({'a': np.zeros(3)}) is None


def test_lnZ_ignores_non_finite_rows(H):
    r = _rec([0.0, -np.inf, 0.0])
    assert np.isclose(H['_lnZ_of_rvs'](r, already_pooled=True), np.log(2.0))


def test_kish_neff_of_equal_weights_is_the_row_count(H):
    assert np.isclose(H['_kish_neff_of_rvs'](_rec(np.zeros(7))), 7.0)


def test_kish_neff_collapses_on_one_dominant_row(H):
    neff = H['_kish_neff_of_rvs'](_rec([0.0, -50.0, -50.0, -50.0]))
    assert 1.0 <= neff < 1.01


# -------------------------------------------------------------- reserve-vs-fairdraw provenance
def test_lnZ_prefers_the_retained_reserve_and_says_so(H):
    _FakeAV.lnZ_value = -1.25
    s = _Sampler(reserve={'log_joint_prior': np.zeros(3), 'log_joint_s_prior': np.zeros(3)})
    val, src = H['_lnZ_of_reserve_or_rvs'](s, _rec([0.0, 0.0]))
    assert src == 'retained' and np.isclose(val, -1.25)


def test_lnZ_falls_back_to_the_fairdraw_record_when_no_reserve(H):
    s = _Sampler(reserve=None)
    val, src = H['_lnZ_of_reserve_or_rvs'](s, _rec([0.0, 0.0]))
    assert src == 'fairdraw' and np.isclose(val, 0.0)


def test_lnZ_falls_back_when_lnZ_from_reserve_is_not_finite(H):
    """Degrade to the previous behaviour, not to no gate at all."""
    _FakeAV.lnZ_value = np.nan
    s = _Sampler(reserve={'log_joint_prior': np.zeros(3), 'log_joint_s_prior': np.zeros(3)})
    _val, src = H['_lnZ_of_reserve_or_rvs'](s, _rec([0.0, 0.0]))
    assert src == 'fairdraw'


# ----------------------------------------------------------------- snapshot / restore (Finding 5)
def test_snapshot_restore_round_trips_the_whole_pass(H):
    member = _Sampler(params=('a', 'b'))
    member._warm_seed_reserve = {'tag': 'cold-member'}
    s = _Sampler(rvs=_rec([1.0, 2.0]), reserve={'tag': 'cold'}, members=[member])
    s._rvs_is_fairdraw, s._rvs_is_pooled = True, False

    snap = H['_snapshot_pass_state'](s, 'RES', 'VAR', 'NEFF', {'d': 1})

    # the warm pass overwrites everything
    s._rvs = _rec([9.0])
    s._warm_seed_reserve = {'tag': 'WARM'}
    s._rvs_is_fairdraw, s._rvs_is_pooled = False, True
    member._warm_seed_reserve = {'tag': 'WARM-member'}

    out = H['_restore_pass_state'](s, snap)
    assert out == ('RES', 'VAR', 'NEFF', {'d': 1})
    assert s._warm_seed_reserve == {'tag': 'cold'}, "the RESERVE did not come back (Finding 5)"
    assert member._warm_seed_reserve == {'tag': 'cold-member'}, "per-member reserve did not come back"
    assert s._rvs_is_fairdraw is True and s._rvs_is_pooled is False
    assert np.allclose(s._rvs['log_integrand'], [1.0, 2.0])


def test_snapshot_takes_a_copy_not_an_alias(H):
    """integrate_log repopulates _rvs IN PLACE, so an alias would hold the warm samples."""
    s = _Sampler(rvs=_rec([1.0, 2.0]))
    snap = H['_snapshot_pass_state'](s, 1, 2, 3, {})
    s._rvs['log_integrand'] = np.array([99.0, 99.0])
    assert snap['rvs'] is not s._rvs


# -------------------------------------------------------------------- the reserve lookup guard
def test_reserve_lookup_declines_a_column_order_mismatch(H):
    """A silent mismatch produces a seed in the wrong coordinates, so decline it."""
    s = _Sampler(reserve={'params_ordered': ['b', 'a']}, params=('a', 'b'))
    assert H['_warm_seed_reserve_for'](s) is None


def test_reserve_lookup_accepts_matching_column_order(H):
    res = {'params_ordered': ['a', 'b']}
    assert H['_warm_seed_reserve_for'](_Sampler(reserve=res, params=('a', 'b'))) is res


def test_reserve_lookup_falls_through_to_a_portfolio_member(H):
    member = _Sampler(params=('a', 'b'))
    member._warm_seed_reserve = {'params_ordered': ['a', 'b'], 'tag': 'member'}
    s = _Sampler(reserve=None, params=('a', 'b'), members=[member])
    assert H['_warm_seed_reserve_for'](s)['tag'] == 'member'


# ------------------------------------------------------------------------------- seed geometry
def test_geometry_uses_the_samplers_adaptive_axes_when_it_has_them(H):
    s = _Sampler(params=('a', 'b'))
    s.warm_seed_axes = lambda: [1]
    axes, lo, hi = H['_warm_seed_geometry'](s)
    assert axes == [1] and np.allclose(lo, [0, 0]) and np.allclose(hi, [1, 1])


def test_geometry_defaults_to_every_column(H):
    axes, _lo, _hi = H['_warm_seed_geometry'](_Sampler(params=('a', 'b', 'c')))
    assert axes == [0, 1, 2]


def test_geometry_falls_through_to_a_portfolio_member(H):
    member = _Sampler(params=('a', 'b'))
    member.warm_seed_axes = lambda: [0]
    s = _Sampler(params=('a', 'b'), members=[member])
    assert H['_warm_seed_geometry'](s)[0] == [0]


# ---------------------------------------------------------------------------- clearing warm state
def test_clear_warm_state_prefers_the_portfolio_hook(H):
    s = _Sampler()
    calls = []
    s.clear_warm_state = lambda: calls.append(1)
    H['_clear_warm_state'](s)
    assert calls == [1], "portfolio members would keep the previous point's contracted grid"


def test_clear_warm_state_falls_back_to_the_attributes(H):
    s = _Sampler()
    H['_clear_warm_state'](s)
    assert s._warm is None and s._warm_applied is False


def test_clear_warm_state_does_not_swallow_failures(H):
    """A reset that quietly did not happen is the silent bias this guards against."""
    s = _Sampler()

    def _boom():
        raise RuntimeError("no")
    s.clear_warm_state = _boom
    with pytest.raises(RuntimeError):
        H['_clear_warm_state'](s)


# ------------------------------------------------------------------------- the rescue itself
def _run(H, sampler, res=1.0, var=0.1, neff=1.0, dict_return=None):
    return H['_maybe_l0_rescue'](sampler, res, var, neff, dict_return or {'cold': True},
                                 lambda *a, **k: None, (), {})


def _assert_declined(H, sampler, capsys, **runkw):
    """The rescue must DECLINE silently -- not run and get rescued by its own except.

    Asserting only the return value is not enough, and an earlier version of these tests
    made exactly that mistake: with a guard removed the rescue starts, throws somewhere
    inside, and `except Exception` returns the inputs unchanged -- so the return value is
    identical either way.  The observable difference is that a declining rescue says
    NOTHING and never touches the sampler.
    """
    capsys.readouterr()
    out_vals = _run(H, sampler, **runkw)
    printed = capsys.readouterr().out
    assert "[L0 auto-rescue]" not in printed, \
        "the rescue engaged when it should have declined: %r" % printed
    assert getattr(sampler, 'bootstrapped', None) is None
    return out_vals


def test_rescue_is_a_noop_when_the_option_is_off(capsys):
    """Uses a DEGENERATE neff, so the option guard is the only thing declining.

    With neff=None, `_needs_l0_rescue` is True on its own; only the
    `opts.sampler_warmstart_retry_neff` conjunct can stop the rescue here.  A healthy neff
    would make this test pass with that conjunct deleted.
    """
    H = _load(opts=_Opts(sampler_warmstart_retry_neff=None))
    s = _Sampler(rvs=_rec([1.0, 2.0, 3.0]), integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    assert _assert_declined(H, s, capsys, neff=None) == (1.0, 0.1, None, {'cold': True})


def test_rescue_is_a_noop_for_a_sampler_method_it_does_not_apply_to(capsys):
    """AV/portfolio only.  Every other conjunct is satisfied here."""
    H = _load(opts=_Opts(sampler_method='GMM'))
    s = _Sampler(rvs=_rec([1.0, 2.0, 3.0]), integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    _assert_declined(H, s, capsys, neff=1.0)


def test_rescue_is_a_noop_for_a_sampler_that_cannot_warm_start(capsys):
    """mcsampler/GMM have no bootstrap_from_samples; the rescue must decline, not crash."""
    H = _load()

    class _NoBootstrap(object):
        def __init__(self):
            self._rvs = _rec([1.0])
            self.params_ordered = ['a', 'b']

        def identity_convert(self, x):
            return x

    s = _NoBootstrap()
    assert not hasattr(s, 'bootstrap_from_samples')
    _assert_declined(H, s, capsys, neff=1.0)


def test_rescue_does_not_touch_identity_convert_before_deciding_it_applies(capsys):
    """Regression: RIFT.integrators.mcsampler.MCSampler has NO identity_convert.

    That is the object this driver keeps for --sampler-method adaptive_cartesian.  The main
    driver evaluates `sampler.identity_convert(neff)` BEFORE its guard, so porting it
    verbatim made every adaptive_cartesian event die with AttributeError at the end of a
    completed integration, before --output-file was written.  The applicability guards must
    run first.
    """
    H = _load(opts=_Opts(sampler_method='adaptive_cartesian'))

    class _NoConvert(object):
        """Exactly mcsampler.MCSampler's relevant shape: no identity_convert."""
        def __init__(self):
            self._rvs = _rec([1.0])
            self.params_ordered = ['a', 'b']

    s = _NoConvert()
    assert not hasattr(s, 'identity_convert')
    capsys.readouterr()
    assert _run(H, s, neff=1.0) == (1.0, 0.1, 1.0, {'cold': True})


def test_rescue_is_a_noop_when_neff_is_healthy(capsys):
    H = _load()
    s = _Sampler(rvs=_rec([1.0]), integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    assert _assert_declined(H, s, capsys, neff=500.0)[3] == {'cold': True}


def test_degenerate_early_termination_triggers_the_rescue():
    """neff=None is the STRONGEST trigger, not a reason to skip."""
    H = _load()
    s = _Sampler(rvs=_rec([1.0, 2.0, 3.0]), integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    s.warm_rvs = _rec([5.0, 5.0, 5.0])
    out = _run(H, s, neff=None)
    assert s.bootstrapped is not None, "a degenerate pass did not trigger the rescue"
    assert out[2] == 42.0


def test_accepted_warm_pass_replaces_the_cold_result():
    H = _load()
    s = _Sampler(rvs=_rec([0.0, 0.0]), integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    s.warm_rvs = _rec([0.0, 0.0])         # same lnZ -> no evidence of loss
    out = _run(H, s)
    assert out == ('R2', 'V2', 42.0, {'warm': True})
    assert s._av_state_reuse_safe is True


def test_warm_pass_far_below_cold_is_rejected_and_cold_is_restored():
    """The gate: positive evidence of lost mass keeps the full-support cold pass."""
    H = _load()
    cold = _rec([0.0, 0.0, 0.0, 0.0])     # lnZ = 0
    s = _Sampler(rvs=cold, reserve={'tag': 'cold'},
                 integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    s._rvs_is_fairdraw = True
    s.warm_rvs = _rec([-20.0, -20.0, -20.0, -20.0])   # lnZ = -20, far below
    out = _run(H, s, res='R1', var='V1', neff=1.0, dict_return={'cold': True})
    assert out == ('R1', 'V1', 1.0, {'cold': True}), "the warm pass was not rejected"
    assert s._warm_seed_reserve == {'tag': 'cold'}, "the reserve did not come back (Finding 5)"
    assert np.allclose(s._rvs['log_integrand'], cold['log_integrand'])
    assert s._av_state_reuse_safe is False, "the rejected warm grid could be persisted"


def test_a_later_healthy_event_resets_the_state_save_veto():
    """Sampler objects are reused; an earlier rejection must not poison later state saves."""
    H = _load()
    s = _Sampler(rvs=_rec([0.0, 0.0]), integrate_result=('R2', 'V2', 42.0, {}))
    s.warm_rvs = _rec([-20.0, -20.0])
    _run(H, s)  # rejected warm pass
    assert s._av_state_reuse_safe is False
    _run(H, s, neff=42.0)  # healthy next event; returns before attempting a rescue
    assert s._av_state_reuse_safe is True


def test_reject_message_reports_lnZ_on_the_events_offset_scale(capsys):
    """lnL_offset is this event's manual_avoid_overflow_logarithm.

    It exists so the *** REJECTING *** line quotes absolute lnZ rather than the internally
    offset value.  Nothing else reads it, so dropping it at the call sites is invisible
    unless a test drives it at a NON-ZERO value -- which is what made it possible to delete
    `lnL_offset=manual_avoid_overflow_logarithm` from both call sites with 81 tests green.
    """
    H = _load()
    s = _Sampler(rvs=_rec([0.0, 0.0]), integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    s.warm_rvs = _rec([-20.0, -20.0])
    capsys.readouterr()
    H['_maybe_l0_rescue'](s, 'R1', 'V1', 1.0, {'cold': True},
                          lambda *a, **k: None, (), {}, lnL_offset=1000.0)
    out = capsys.readouterr().out
    assert "REJECTING" in out
    # cold lnZ 0.0 and warm lnZ -20.0, both shifted by +1000 in the report
    assert "1000.000" in out and "980.000" in out, \
        "the reject message did not quote lnZ on the event's offset scale: %r" % out


def test_both_call_sites_pass_the_events_offset():
    """Source-level, because the value comes from a local of each analyze_event."""
    src = _src()
    # Count PER HELPER, not globally: more than one helper now takes lnL_offset (the L0
    # rescue and the MC-error replica block), so a global count silently absorbs a call
    # site that dropped it as long as some other helper still passes it.
    tree = ast.parse(src)
    for helper, want in (("_maybe_l0_rescue", 2), ("_maybe_replicate_for_mc_error", 2)):
        passing = [c for c in ast.walk(tree)
                   if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == helper
                   and any(k.arg == "lnL_offset"
                           and isinstance(k.value, ast.Name)
                           and k.value.id == "manual_avoid_overflow_logarithm"
                           for k in c.keywords)]
        assert len(passing) == want, (
            "%s: %d of %d call sites pass lnL_offset=manual_avoid_overflow_logarithm; a "
            "site that dropped it would quote the internally-offset lnZ, not the absolute one"
            % (helper, len(passing), want))


def test_accept_truncated_reports_the_warm_pass_anyway():
    H = _load(opts=_Opts(sampler_l0_rescue_accept_truncated=True))
    s = _Sampler(rvs=_rec([0.0, 0.0]), integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    s.warm_rvs = _rec([-20.0, -20.0])
    assert _run(H, s)[0] == 'R2'


def test_reject_threshold_is_respected():
    """A shortfall smaller than the threshold is not evidence of loss."""
    H = _load(opts=_Opts(sampler_l0_rescue_reject_dlnZ=50.0))
    s = _Sampler(rvs=_rec([0.0, 0.0]), integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    s.warm_rvs = _rec([-20.0, -20.0])
    assert _run(H, s)[0] == 'R2', "a 20-nat drop was rejected against a 50-nat threshold"


def test_a_raising_warm_pass_restores_the_cold_state():
    """The silent-for-a-campaign shape: _rvs holds warm samples, res/neff still hold cold."""
    H = _load()
    cold = _rec([0.0, 0.0])
    s = _Sampler(rvs=cold, reserve={'tag': 'cold'}, raise_in_integrate=True)
    s.warm_rvs = _rec([7.0, 7.0])
    out = _run(H, s, res='R1', var='V1', neff=1.0, dict_return={'cold': True})
    assert out == ('R1', 'V1', 1.0, {'cold': True})
    assert np.allclose(s._rvs['log_integrand'], cold['log_integrand']), \
        "cold diagnostics were reported beside a warm export"
    assert s._warm_seed_reserve == {'tag': 'cold'}
    assert s._av_state_reuse_safe is False, "the failed warm grid could be persisted"


def test_rescue_clears_warm_state_afterwards():
    H = _load()
    s = _Sampler(rvs=_rec([0.0, 0.0]), integrate_result=('R2', 'V2', 42.0, {}))
    s.warm_rvs = _rec([0.0, 0.0])
    _run(H, s)
    assert s._warm is None, "the next point would draw from this point's contracted grid"


def test_mixed_lnZ_provenance_falls_back_to_a_like_for_like_comparison():
    """Cold read from the reserve, warm from the fair draw, is not a difference.

    The two readings differ by ~log(n_retained/eff_samp), so a mixed comparison manufactures
    a gap of several nats out of nothing.  The numbers here are chosen so the two paths
    DISAGREE about the outcome -- an earlier version of this test used values where both
    accepted, and it passed with the guard disabled.

        mixed (broken):  cold 'retained' +10.0  vs warm 'fairdraw' 0.0  -> 10 nats -> REJECT
        like-for-like :  both re-read from _rvs, 0.0 vs 0.0            ->  0 nats -> ACCEPT
    """
    class _AV(_FakeAV):
        calls = {'n': 0}

        @classmethod
        def lnZ_from_reserve(cls, reserve):
            # available for the cold read, gone for the warm one
            cls.calls['n'] += 1
            return 10.0 if cls.calls['n'] == 1 else None
    _AV.calls['n'] = 0
    H = _load(av=_AV)
    s = _Sampler(rvs=_rec([0.0, 0.0]),
                 reserve={'log_joint_prior': np.zeros(2), 'log_joint_s_prior': np.zeros(2)},
                 integrate_result=('R2', 'V2', 42.0, {'warm': True}))
    s.warm_rvs = _rec([0.0, 0.0])
    out = _run(H, s, res='R1', var='V1', neff=1.0, dict_return={'cold': True})
    assert out[0] == 'R2', ("a like-for-like lnZ comparison found no evidence of loss, so the "
                            "warm pass must stand; rejecting it means the gate compared a "
                            "'retained' reading against a 'fairdraw' one")


# ---------------------------------------------------------------------- source-level wiring
def _src():
    with open(_LISA) as fh:
        return fh.read()


def test_both_analyze_event_variants_call_the_rescue():
    """This driver has two; a rescue wired into only one is a silent half-port."""
    tree = ast.parse(_src())
    fns = {n.name: n for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name in ('analyze_event', 'analyze_event_LISA')}
    assert set(fns) == {'analyze_event', 'analyze_event_LISA'}
    for name, node in fns.items():
        called = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                     and c.func.id == '_maybe_l0_rescue' for c in ast.walk(node))
        assert called, "%s does not call _maybe_l0_rescue" % name


def test_rescue_runs_before_the_no_result_guard():
    """Ordering is load-bearing.

    A degenerate early termination returns (None,None,None,None); `if not(res): raise` would
    abort on it, skipping the strongest rescue trigger.  In the main driver that guard sits
    ~200 lines below the integrate call so the ordering is implicit -- here it is adjacent,
    so it is pinned.
    """
    src = _src()
    guard = "if not(res): # no resut"
    assert src.count(guard) == 2, "expected the guard in both analyze_event variants"
    pos = 0
    for _ in range(2):
        g = src.index(guard, pos)
        call = src.rindex("_maybe_l0_rescue(", 0, g)
        integ = src.rindex("sampler.integrate(like_to_integrate", 0, call)
        assert integ < call < g, "the rescue must sit between integrate and the not(res) guard"
        pos = g + 1


def test_rescue_is_not_hidden_behind_the_LISA_flag():
    """Both variants get it; nothing keys the rescue off opts.LISA."""
    tree = ast.parse(_src())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_maybe_l0_rescue'][0]
    body = ast.dump(fn)
    assert "'LISA'" not in body and 'attr=\'LISA\'' not in body


@pytest.mark.parametrize("opt,default", [
    ("--sampler-l0-rescue-reject-dlnZ", "default=3.0"),
    ("--sampler-l0-rescue-puff-width-frac", "default=0.005"),
    ("--sampler-l0-rescue-puff-factor", "default=2.0"),
    ("--sampler-sequential-warmstart-deltalnL", "default=15.0"),
])
def test_option_defaults_match_the_main_driver(opt, default):
    """A knob that means something different in the two drivers is worse than a missing one.

    reject-dlnZ 3.0 in particular is a MEASURED value (L0_REJECT_DLNZ_MEASUREMENT.md); the
    old 0.5 binned 25% of good portfolio warm passes while catching 0 of 55 truncated ones.
    """
    for path in (_LISA, _MAIN):
        with open(path) as fh:
            src = fh.read()
        i = src.index('"%s"' % opt)
        line = src[i:src.index("\n", i)]
        assert default.replace(" ", "") in line.replace(" ", ""), \
            "%s: %s does not carry %s" % (os.path.basename(path), opt, default)


# ------------------------------------------------------------------ anti-drift vs the main driver
def _normalized(fn):
    node = ast.parse(ast.unparse(fn)).body[0] if hasattr(ast, "unparse") else fn
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return ast.dump(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])))


@pytest.mark.parametrize("name", PORTED)
def test_ported_helper_is_identical_to_the_main_driver(name):
    """Deliberate COPIES in a deliberate fork.  Change one, change both."""
    assert _normalized(_defs(_LISA, [name])[name]) == _normalized(_defs(_MAIN, [name])[name]), \
        "%s has drifted between the two drivers (docstrings excluded)" % name
