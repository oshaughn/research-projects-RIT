#!/usr/bin/env python
"""
Tests for the fair-draw weighting helpers ported into the LISA ILE driver
(bin/integrate_likelihood_extrinsic_batchmode_lisa) from the main driver, PR #87.

WHY THE LISA DRIVER NEEDS THEM AT ALL.  The three consumers whose double-weighting PR #87
fixed -- the `--extrinsic-proposal-output` breadcrumb, the `.dgrid` exporter and the
`.dslice` reweight core -- do not exist in the LISA driver, so there is no live w^2 bug
there today.  What DOES exist is the hazard: the LISA driver sets
`igrand_fairdraw_samples` from `--fairdraw-extrinsic-output`, so its `_rvs` can be a fair
draw, and every shared sampler in RIFT/integrators/ already sets `_rvs_is_fairdraw` at its
rebind.  The marker was arriving and nothing read it.  These tests pin the readers.

TWO DISTINCT PROPERTIES, deliberately not one flag (audit Finding 6):

    rows resampled  -- each row drawn proportional to w   (per-BLOCK property)
    equal weight    -- the record as a whole is uniform   (property of the WHOLE record)

and the anti-drift test at the bottom pins the LISA copies to the main driver's, because
these are deliberate COPIES in a deliberate fork, not an import.

Conventions follow test_fairdraw_double_weighting.py and test_l0_rescue_seed.py: the driver
scripts are not importable (they parse argv at import), so the helpers are exec'd out.
"""

import ast
import os

import numpy as np
from RIFT.integrators.rvs_record import SamplerOutputMixin as _SamplerOutputMixin
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LISA = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')
_MAIN = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

# The helpers ported in this pass.  Named explicitly: if a future edit drops one, the
# extraction below fails loudly rather than silently testing a smaller surface.
# The record accessors are in this list DELIBERATELY: it is both the exec set and the
# anti-drift set, so naming them here fixes the namespace AND puts them under the
# change-one-change-both gate, which is where a shared-by-copy helper belongs.
PORTED = ['_rvs_lnL_convention', 'ln_weights_from_rvs', '_rvs_len',
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

def _extract(path, names):
    """Return {name: ast.FunctionDef} for top-level defs, by name."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    found = {n.name: n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = sorted(set(names) - set(found))
    assert not missing, "%s is missing ported helper(s): %s" % (os.path.basename(path), missing)
    return found


def _load(path, names=PORTED):
    """Exec the named helpers out of a driver script into a namespace."""
    defs = _extract(path, names)
    mod = ast.Module(body=[defs[n] for n in names], type_ignores=[])
    ns = {"numpy": np, "np": np, "SamplerOutputMixin": _SamplerOutputMixin}
    exec(compile(ast.fix_missing_locations(mod), "lisa_weight_helpers", "exec"), ns)
    _assert_helper_set_is_closed(ns, names, _LISA)
    return ns


@pytest.fixture(scope="module")
def H():
    return _load(_LISA)


# --------------------------------------------------------------------------- record builders
def _log_record(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return {'log_integrand': rng.normal(size=n) * 3.0,
            'log_joint_prior': rng.normal(size=n),
            'log_joint_s_prior': rng.normal(size=n),
            'right_ascension': rng.uniform(0, 2 * np.pi, size=n)}


def _linear_record(n=6, seed=1, lnL=False):
    rng = np.random.default_rng(seed)
    ig = (rng.normal(size=n) * 3.0) if lnL else rng.uniform(0.1, 5.0, size=n)
    return {'integrand': ig,
            'joint_prior': rng.uniform(0.1, 2.0, size=n),
            'joint_s_prior': rng.uniform(0.1, 2.0, size=n),
            'psi': rng.uniform(0, np.pi, size=n)}


class _FakeSampler(object):
    def __init__(self, fairdraw=None, pooled=None):
        if fairdraw is not None:
            self._rvs_is_fairdraw = fairdraw
        if pooled is not None:
            self._rvs_is_pooled = pooled


# ------------------------------------------------------------------- ln_weights_from_rvs
def test_log_form_is_the_canonical_combination(H):
    r = _log_record()
    got = H['ln_weights_from_rvs'](r)
    want = r['log_integrand'] + r['log_joint_prior'] - r['log_joint_s_prior']
    assert np.allclose(got, want)


def test_log_form_preferred_over_linear_when_both_present(H):
    """The log columns win.  A record carrying both must not be read the linear way."""
    r = _log_record()
    r.update({'integrand': np.full(len(r['log_integrand']), 1.0),
              'joint_prior': np.full(len(r['log_integrand']), 1.0),
              'joint_s_prior': np.full(len(r['log_integrand']), 1.0)})
    got = H['ln_weights_from_rvs'](r)
    want = r['log_integrand'] + r['log_joint_prior'] - r['log_joint_s_prior']
    assert np.allclose(got, want), "linear columns shadowed the canonical log ones"


def test_linear_form_linear_convention(H):
    r = _linear_record(lnL=False)
    got = H['ln_weights_from_rvs'](r, use_lnL=False)
    want = np.log(r['integrand']) + np.log(r['joint_prior']) - np.log(r['joint_s_prior'])
    assert np.allclose(got, want)


def test_linear_form_out_of_support_rows_are_minus_inf(H):
    r = _linear_record(lnL=False)
    r['joint_prior'][2] = 0.0          # zero prior  -> out of support
    r['integrand'][4] = 0.0            # zero L      -> out of support
    got = H['ln_weights_from_rvs'](r, use_lnL=False)
    assert got[2] == -np.inf and got[4] == -np.inf
    assert np.isfinite(got[[0, 1, 3, 5]]).all()


def test_lnL_convention_does_not_log_twice_and_keeps_negative_lnL(H):
    """The bug this argument exists for.

    mcsamplerEnsemble reuses 'integrand' for BOTH conventions.  Under return_lnI it holds
    lnL, so the linear reading would (a) take log() of it, compressing tens of nats into
    log(tens), and (b) apply `ig > 0`, silently discarding every sample with lnL <= 0.
    """
    r = _linear_record(lnL=True)
    r['integrand'][0] = -12.5          # a perfectly good low-likelihood point
    got = H['ln_weights_from_rvs'](r, use_lnL=True)
    want = r['integrand'] + np.log(r['joint_prior']) - np.log(r['joint_s_prior'])
    assert np.allclose(got, want)
    assert np.isfinite(got[0]), "a negative lnL row was discarded as out-of-support"

    wrong = H['ln_weights_from_rvs'](r, use_lnL=False)
    assert not np.allclose(np.nan_to_num(wrong, neginf=-1e9), got), \
        "the two conventions agree, so this test cannot detect reading lnL as L"


def test_raises_when_neither_component_set_is_present(H):
    """An explicit failure beats a plausible wrong number."""
    with pytest.raises(Exception):
        H['ln_weights_from_rvs']({'psi': np.zeros(4), 'log_weights': np.zeros(4)})


def test_cached_log_weights_column_is_never_read(H):
    """mcsamplerGPU stores the ADAPTATION weight there, with adapt-weight-exponent baked in."""
    r = _log_record()
    r['log_weights'] = np.full(len(r['log_integrand']), 999.0)
    got = H['ln_weights_from_rvs'](r)
    assert not np.allclose(got, 999.0)


# ------------------------------------------------------------------------ the two predicates
@pytest.mark.parametrize("fairdraw,pooled,resample,equal", [
    (None,  None,  False, False),   # markers absent entirely -> both False, no AttributeError
    (False, False, False, False),
    (True,  False, True,  True),    # a plain fair draw has BOTH properties
    (True,  True,  True,  False),   # pooled: rows resampled, record NOT globally uniform
    (False, True,  False, False),
])
def test_predicate_truth_table(H, fairdraw, pooled, resample, equal):
    s = _FakeSampler(fairdraw, pooled)
    assert H['_rvs_is_export_resample'](s) is resample
    assert H['_rvs_is_equal_weight'](s) is equal


def test_predicates_differ_on_a_pooled_record(H):
    """The Finding-6 property: one flag cannot answer both questions."""
    s = _FakeSampler(fairdraw=True, pooled=True)
    assert H['_rvs_is_export_resample'](s) != H['_rvs_is_equal_weight'](s)


# --------------------------------------------------------------- ln_weights_for_posterior
def test_fair_drawn_record_gets_uniform_posterior_weights(H):
    """The anti-double-weighting property: rows already ~w must not be weighted by w again."""
    r = _log_record()
    w = H['ln_weights_for_posterior'](r, _FakeSampler(fairdraw=True, pooled=False))
    assert w.shape == (len(r['log_integrand']),)
    assert np.allclose(w, 0.0)


def test_non_fairdrawn_record_gets_the_importance_weights(H):
    r = _log_record()
    s = _FakeSampler(fairdraw=False, pooled=False)
    assert np.allclose(H['ln_weights_for_posterior'](r, s), H['ln_weights_from_rvs'](r))


def test_pooled_record_keeps_its_between_block_weights(H):
    """Pooling weights block k by the replica evidence: uniform here would discard that."""
    r = _log_record()
    w = H['ln_weights_for_posterior'](r, _FakeSampler(fairdraw=True, pooled=True))
    assert not np.allclose(w, 0.0)
    assert np.allclose(w, H['ln_weights_from_rvs'](r))


def test_double_weighting_would_shift_a_posterior_mean(H):
    """Why it matters, not just that it differs.

    Build a record whose weight correlates with a coordinate, fair-draw it, then compare the
    mean under the correct (uniform) weights against the mean under a second application of
    w.  The second application concentrates toward high-w rows and moves the answer.
    """
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.uniform(0.0, 1.0, size=n)
    lnw = 4.0 * x                                    # weight correlated with the coordinate
    w = np.exp(lnw - lnw.max())
    idx = rng.choice(n, size=n, replace=True, p=w / w.sum())   # the fair draw
    rec = {'log_integrand': lnw[idx], 'log_joint_prior': np.zeros(n),
           'log_joint_s_prior': np.zeros(n), 'x': x[idx]}

    correct = H['ln_weights_for_posterior'](rec, _FakeSampler(fairdraw=True, pooled=False))
    assert np.allclose(correct, 0.0)
    mean_correct = np.average(rec['x'], weights=np.exp(correct - correct.max()))

    doubled = H['ln_weights_from_rvs'](rec)          # what the pre-fix consumers did
    mean_doubled = np.average(rec['x'], weights=np.exp(doubled - doubled.max()))

    shift = abs(mean_doubled - mean_correct) / abs(mean_correct)
    assert shift > 0.05, ("double weighting should move the posterior mean materially; "
                          "got %.3f%%" % (100 * shift))


# ----------------------------------------------------------------------------- _rvs_len
def test_rvs_len_counts_rows(H):
    assert H['_rvs_len'](_log_record(n=9)) == 9


def test_rvs_len_survives_an_unsized_entry(H):
    r = _log_record(n=5)
    r['not_an_array'] = None
    assert H['_rvs_len'](r) == 5


def _record_with_a_combined_parameter(n=6):
    """Columns in the order a sampler seeds them: PARAMETERS FIRST, then the weight columns.

    The order is the whole point.  A parameter registered under a TUPLE key is a combined
    parameter stored (ndim, N) -- the convention every sampler indexes by, `col[:, idx]` for a
    tuple key against `col[idx]` otherwise -- and it is seeded before the weight columns, so
    "whichever column came first" lands on it in the ordinary case rather than a corner.
    """
    r = {('mc', 'delta_mc'): np.zeros((2, n))}
    r.update(_log_record(n=n))
    return r


def test_rvs_len_counts_ROWS_not_entries_for_a_combined_parameter():
    """ndim*N is not a row count, and it is not a cosmetic one either.

    Both drivers: the LISA copy checks the pooled export's weight vector against this number,
    so an inflated count made the check fail and shipped the pooled record weight-mixed; the
    main copy hands back a uniform vector OF THIS LENGTH for a fair draw and records it as the
    pooled `block_sizes`.
    """
    r = _record_with_a_combined_parameter(n=6)
    for path in (_LISA, _MAIN):
        assert _load(path)['_rvs_len'](r) == 6, os.path.basename(path)


def test_rvs_len_reads_the_row_axis_from_the_key_when_no_weight_column_is_present():
    """No canonical per-row column to settle it -> the key's own layout decides."""
    r = {('mc', 'delta_mc'): np.zeros((2, 7)), 'psi': np.zeros(7)}
    for path in (_LISA, _MAIN):
        assert _load(path)['_rvs_len'](r) == 7, os.path.basename(path)


def test_fair_draw_uniform_weights_are_one_per_row_with_a_combined_parameter(H):
    """The consumer-visible failure: a weight vector ndim times longer than the record."""
    r = _record_with_a_combined_parameter(n=6)
    w = H['ln_weights_for_posterior'](r, _FakeSampler(fairdraw=True, pooled=False))
    assert w.shape == (6,)


# ------------------------------------------------------------------ the convention resolver
def test_lnL_convention_prefers_the_explicit_argument(H):
    assert H['_rvs_lnL_convention'](True) is True
    assert H['_rvs_lnL_convention'](False) is False


def test_lnL_convention_falls_back_to_linear_outside_the_driver(H):
    """No `rvs_integrand_is_lnL` in scope (which is the case in these tests) -> False."""
    assert H['_rvs_lnL_convention'](None) is False


# ------------------------------------------------------------- source-level wiring in LISA
def _lisa_src():
    with open(_LISA) as fh:
        return fh.read()


def test_lisa_derives_the_convention_from_pinned_params_not_the_cli_option():
    """The trap this port had to avoid.

    --internal-use-lnL is ALSO accepted for adaptive_cartesian_gpu and portfolio, and those
    branches set use_lnL WITHOUT return_lnI -- they still store linear L.  Deriving the
    stored convention from the option would read those records as lnL.
    """
    src = _lisa_src()
    assert 'rvs_integrand_is_lnL = bool(pinned_params.get("return_lnI", False))' in src, \
        "the stored-integrand convention is not derived from pinned_params['return_lnI']"
    assert 'rvs_integrand_is_lnL = bool(opts.internal_use_lnL' not in src, \
        "the convention is keyed off the CLI option, which is a different predicate"


def test_lisa_still_requests_the_fair_draw():
    """If this ever stops being set, the helpers become dead code and should be revisited."""
    assert '"igrand_fairdraw_samples": opts.fairdraw_extrinsic_output' in _lisa_src()


# ------------------------------------------------------------------ anti-drift vs the main driver
def _normalized(fn):
    """AST dump of a function with its docstring stripped.

    Docstrings are deliberately allowed to differ -- the LISA copies carry LISA-specific
    notes.  Everything the interpreter runs must match.
    """
    node = ast.parse(ast.unparse(fn)).body[0] if hasattr(ast, "unparse") else fn
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    stripped = ast.Module(body=body, type_ignores=[])
    return ast.dump(ast.fix_missing_locations(stripped))


@pytest.mark.parametrize("name", PORTED)
def test_ported_helper_is_identical_to_the_main_driver(name):
    """These are COPIES in a deliberate fork.  A copy that quietly changes is the whole risk.

    If you intend to change one, change both -- or record the divergence explicitly.
    """
    lisa = _extract(_LISA, [name])[name]
    main = _extract(_MAIN, [name])[name]
    assert _normalized(lisa) == _normalized(main), (
        "%s has drifted between the two drivers (docstrings excluded)" % name)
