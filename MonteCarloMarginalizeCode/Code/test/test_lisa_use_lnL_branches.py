#!/usr/bin/env python
"""
The per-sampler `use_lnL` / `return_lnI` branches in the LISA ILE driver.

FOUND BY ADVERSARIAL AUDIT, NOT BY THE DRIFT GATE.  The main driver has

    if opts.sampler_method == "AV" and opts.internal_use_lnL:
        return_lnL = True
        pinned_params.update({"use_lnL": True})

with the comment: *"without this, --internal-use-lnL --sampler-method AV passed the
ok_lnL_methods check but silently did nothing, so exp(lnL) overflowed at high SNR when no
logarithm offset was set."*  The LISA driver had branches for GMM, adaptive_cartesian_gpu
and portfolio -- and none for AV.

High SNR is the LISA MBHB regime, so this is the case, not an edge.

WHY THE DRIFT AUDIT MISSED IT, and why this file exists.  A missing `if` branch is not a
FUNC, OPTION, CONST or ATTR, so it produces zero gap items.  The audit is a name-presence
set difference; behaviour behind a shared name is invisible to it.  These tests close that
specific hole by pinning the branch TABLE in both drivers against each other.
"""

import ast
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LISA = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')
_MAIN = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

# Samplers both drivers accept.  Verified identical in both ok_lnL_methods lists.
METHODS = ['GMM', 'adaptive_cartesian', 'adaptive_cartesian_gpu', 'AV', 'portfolio']


def _src(path):
    with open(path) as fh:
        return fh.read()


def _pinned_updates(path):
    """{method: {key: value}} for every `pinned_params.update({...})` guarded by a method test.

    Walks module-level `if` statements, works out which sampler method each one is about
    from the string constants in its test, and records the pinned_params keys it sets.
    """
    tree = ast.parse(_src(path), filename=path)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        methods = {c.value for c in ast.walk(node.test)
                   if isinstance(c, ast.Constant) and c.value in METHODS}
        if not methods:
            continue
        uses_lnL_opt = any(isinstance(a, ast.Attribute) and a.attr == 'internal_use_lnL'
                           for a in ast.walk(node.test))
        keys = {}
        for call in ast.walk(node):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'update'
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == 'pinned_params'):
                for arg in call.args:
                    if isinstance(arg, ast.Dict):
                        for k, v in zip(arg.keys, arg.values):
                            if isinstance(k, ast.Constant):
                                try:
                                    keys[k.value] = ast.literal_eval(v)
                                except Exception:
                                    keys[k.value] = '<expr>'
        if keys:
            for m in methods:
                rec = out.setdefault(m, {"keys": {}, "gated_on_internal_use_lnL": False})
                rec["keys"].update(keys)
                rec["gated_on_internal_use_lnL"] |= uses_lnL_opt
    return out


@pytest.fixture(scope="module")
def lisa():
    return _pinned_updates(_LISA)


@pytest.fixture(scope="module")
def main():
    return _pinned_updates(_MAIN)


def test_AV_sets_use_lnL_under_internal_use_lnL(lisa):
    """The regression this file exists for.

    Without it, --sampler-method AV --internal-use-lnL is a SILENT no-op: the option passes
    the ok_lnL_methods check and changes nothing, so the integrand stays linear and exp(lnL)
    overflows at high SNR unless a manual logarithm offset happens to be set.
    """
    assert 'AV' in lisa, "no AV branch sets pinned_params at all"
    assert lisa['AV']['keys'].get('use_lnL') is True, \
        "--sampler-method AV --internal-use-lnL does not set use_lnL: silent no-op"
    assert lisa['AV']['gated_on_internal_use_lnL'], \
        "the AV branch must be gated on --internal-use-lnL, not unconditional"


@pytest.mark.parametrize("method", ['GMM', 'adaptive_cartesian_gpu', 'AV'])
def test_branch_table_matches_the_main_driver(method, lisa, main):
    """Same method -> same pinned_params keys in both drivers.

    This is the check that would have caught the missing AV branch, and it is the shape the
    name-based drift audit cannot express.
    """
    assert method in main, "the main driver has no %s branch to compare against" % method
    assert method in lisa, "the LISA driver has no %s branch" % method
    assert lisa[method]['keys'] == main[method]['keys'], (
        "%s: pinned_params differ (lisa=%r, main=%r)"
        % (method, lisa[method]['keys'], main[method]['keys']))


def test_portfolio_differs_from_main_only_by_the_deferred_GMM_forwarding(lisa, main):
    """portfolio is the ONE branch still divergent, and only in a known, recorded way.

    The main driver's portfolio branch also forwards the --internal-gmm-* knobs to its GMM
    member (gmm_adaptive / gmm_defensive_frac / gmm_inflate).  Those options are deliberately
    deferred: main wires them through its group-pairing setup, and this driver's GMM block is
    structured differently, so they need their own pass.

    Asserting the delta EXACTLY -- rather than skipping portfolio -- means any OTHER
    divergence in this branch still fails, and this test tightens on its own once the GMM
    pass lands.
    """
    deferred = {'gmm_adaptive', 'gmm_defensive_frac', 'gmm_inflate'}
    lk, mk = lisa['portfolio']['keys'], main['portfolio']['keys']
    assert set(mk) - set(lk) == deferred, (
        "portfolio branch diverges beyond the deferred GMM forwarding: missing here = %s"
        % sorted(set(mk) - set(lk)))
    assert not set(lk) - set(mk), "the LISA portfolio branch sets keys main does not: %s" \
        % sorted(set(lk) - set(mk))
    for k in set(lk) & set(mk):
        assert lk[k] == mk[k], "portfolio: %s differs (lisa=%r, main=%r)" % (k, lk[k], mk[k])


def test_only_GMM_requests_return_lnI(lisa):
    """return_lnI is what makes 'integrand' hold lnL, and it drives rvs_integrand_is_lnL.

    If another sampler gains it, the stored-convention derivation has to be revisited --
    ln_weights_from_rvs reads that convention to decide whether to log the integrand.
    """
    with_lnI = {m for m, rec in lisa.items() if rec['keys'].get('return_lnI') is True}
    assert with_lnI == {'GMM'}, "unexpected return_lnI set: %s" % sorted(with_lnI)


def test_adaptive_cartesian_has_no_use_lnL_branch(lisa):
    """Plain adaptive_cartesian (RIFT.integrators.mcsampler) has no use_lnL handling at all.

    It always stores linear L.  A branch here would make ln_weights_from_rvs read its
    records as lnL, which is the failure the helper's docstring warns about.
    """
    assert 'adaptive_cartesian' not in lisa or \
        'use_lnL' not in lisa['adaptive_cartesian']['keys']


def test_the_convention_is_still_derived_from_pinned_params():
    """Adding a branch must not tempt anyone back to the CLI option."""
    src = _src(_LISA)
    assert 'rvs_integrand_is_lnL = bool(pinned_params.get("return_lnI", False))' in src


def test_the_convention_is_derived_after_every_branch_that_could_set_return_lnI():
    """Ordering: pinned_params must be final where the convention is read off it.

    The main driver derives it "where pinned_params is final".  If a later update ever
    carried return_lnI, deriving it early would silently pick the wrong convention.
    """
    src = _src(_LISA)
    tree = ast.parse(src)
    derive_line = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', None) == 'rvs_integrand_is_lnL'):
            derive_line = node.lineno
    assert derive_line is not None, "rvs_integrand_is_lnL is never assigned"
    # AST, not a text search: 'return_lnI' also appears in docstrings that DESCRIBE the
    # convention, and an earlier version of this test matched those and failed on prose.
    later = [c.lineno for c in ast.walk(tree)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == 'update'
             and isinstance(c.func.value, ast.Name) and c.func.value.id == 'pinned_params'
             and c.lineno > derive_line
             and any(isinstance(k, ast.Constant) and k.value == 'return_lnI'
                     for a in c.args if isinstance(a, ast.Dict) for k in a.keys)]
    assert not later, \
        "pinned_params gains return_lnI at line(s) %s, AFTER the stored convention is " \
        "derived from it at line %d" % (later, derive_line)
