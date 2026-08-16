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
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LISA = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')
_MAIN = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

HELPERS = ['_rvs_lnL_convention', 'ln_weights_from_rvs', '_rvs_len', '_lnZ_of_rvs',
           '_kish_neff_of_rvs', '_extract_mc_diag', '_pool_replica_rvs']


def _src(path):
    with open(path) as fh:
        return fh.read()


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
    ns = {"numpy": np, "np": np}
    exec(compile(ast.fix_missing_locations(mod), "mcerr", "exec"), ns)
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
    """Cleared only on the happy path, it survives the pooled gate's raise (Finding 7)."""
    tree = ast.parse(_src(_LISA))
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("analyze_event", "analyze_event_LISA"):
            body = textwrap.dedent(ast.unparse(n)) if hasattr(ast, "unparse") else ""
            assert "sampler._rvs_is_pooled = False" in body, \
                "%s does not reset the pooled marker on entry" % n.name


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
