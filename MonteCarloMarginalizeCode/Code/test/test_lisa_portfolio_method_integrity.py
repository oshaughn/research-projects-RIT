#!/usr/bin/env python
"""
`opts.sampler_method` must survive portfolio construction.

THE DEFECT.  Building a portfolio that carries a GMM member used to CLOBBER
`opts.sampler_method = 'GMM'`, so the GMM-specific argument blocks further down would run
and forward that member's config.  It worked for that, and silently broke everything else
that asks "what sampler is this run using", because by then the honest answer -- 'portfolio'
-- had been overwritten.

The consequence that matters here: the **L0 auto-rescue never fired for a portfolio**.  Its
guard is

    opts.sampler_method in ('AV', 'portfolio')

so for the single most common portfolio configuration -- one carrying a GMM member -- the
rescue silently declined, on a driver where the rescue had just been ported specifically
because LISA MBHB are high-SNR and that is the regime that stalls.  No error, no log line;
the feature was simply absent.

A portfolio also took GMM-only branches, `return_lnI` among them, which feeds
`rvs_integrand_is_lnL` and therefore how `ln_weights_from_rvs` reads the record.

THE FIX, ported from the main driver, which had already made it: flag the member
non-destructively.  `opts.sampler_method` stays 'portfolio'; the GMM blocks key off
`use_gmm_args = (sampler_method == "GMM") or use_gmm_member`.

WHY THIS FILE IS SHAPED AS AN INVARIANT.  A mutation of a shared option is not a FUNC,
OPTION, CONST or ATTR, so the drift audit produces zero gap items for it -- the same blind
spot that hid the missing AV/`use_lnL` branch.  The first test below is therefore the
general rule ("nothing assigns opts.sampler_method") rather than a check on this one site,
because the next such clobber will be somewhere else.
"""

import ast
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LISA = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')
_MAIN = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')


def _src(path):
    with open(path) as fh:
        return fh.read()


def _assignments_to(path, attr):
    """Line numbers where `opts.<attr>` is assigned (=, augmented, or walrus-ish)."""
    tree = ast.parse(_src(path), filename=path)
    hits = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for t in targets:
            if (isinstance(t, ast.Attribute) and t.attr == attr
                    and isinstance(t.value, ast.Name) and t.value.id == 'opts'):
                hits.append(node.lineno)
    return hits


# ------------------------------------------------------------------------- the invariant
@pytest.mark.parametrize("path,label", [(_LISA, 'lisa'), (_MAIN, 'main')])
def test_nothing_assigns_opts_sampler_method(path, label):
    """The general rule, in BOTH drivers.

    `opts.sampler_method` is read by the L0 rescue gate, the AV state save, the use_lnL
    branch table and several per-event resets.  Any code that reassigns it makes every one
    of those answer a question about a sampler the run is not using.
    """
    hits = _assignments_to(path, 'sampler_method')
    assert not hits, (
        "%s driver assigns opts.sampler_method at line(s) %s. Flag the condition "
        "non-destructively (see use_gmm_member) instead of overwriting the run's identity."
        % (label, hits))


def test_the_rescue_guard_still_reads_sampler_method():
    """If the guard stops reading it, the invariant above protects nothing.

    Pins the two together so neither can be quietly relaxed on its own.
    """
    assert "opts.sampler_method in ('AV', 'portfolio')" in _src(_LISA)


# ------------------------------------------------------------------- the replacement flag
def test_portfolio_loop_flags_a_GMM_member_without_clobbering():
    src = _src(_LISA)
    assert 'use_gmm_member = True' in src, "the GMM member is not flagged at all"
    assert "opts.sampler_method = 'GMM'" not in src, "the clobber is back"
    assert 'use_gmm_member=False' in src, "the flag is never initialised"


def test_use_gmm_args_is_standalone_GMM_or_a_portfolio_member():
    assert 'use_gmm_args = (opts.sampler_method == "GMM") or use_gmm_member' in _src(_LISA)


def test_use_gmm_args_is_defined_before_every_use():
    src = _src(_LISA)
    define = src.index('use_gmm_args = (opts.sampler_method')
    first_use = src.index('if use_gmm_args:')
    assert define < first_use
    tree = ast.parse(src)
    define_line = min(n.lineno for n in ast.walk(tree)
                      if isinstance(n, ast.Assign) and len(n.targets) == 1
                      and getattr(n.targets[0], 'id', None) == 'use_gmm_args')
    module_level_uses = [n.lineno for n in ast.walk(tree)
                         if isinstance(n, ast.Name) and n.id == 'use_gmm_args'
                         and isinstance(n.ctx, ast.Load)]
    # uses inside analyze_event run later regardless; only module-level order can break.
    assert min(module_level_uses) >= define_line


def test_the_GMM_setup_block_runs_for_a_portfolio_member():
    """This is what the clobber existed to achieve, now achieved honestly."""
    src = _src(_LISA)
    i = src.index('use_gmm_args = (opts.sampler_method')
    block = src[i:i + 400]
    assert 'if use_gmm_args:' in block, "the GMM setup block no longer runs for a portfolio member"


def test_per_event_gmm_resets_key_off_use_gmm_args():
    """gmm_dict exists for a portfolio-with-GMM too, so the resets must reach it.

    Two analyze_event variants plus the --force-reset-all block: three sites.
    """
    src = _src(_LISA)
    assert src.count('elif use_gmm_args:') == 3, \
        "expected the two per-event resets and --force-reset-all to key off use_gmm_args"


def test_return_lnI_still_keys_on_the_method_not_the_member():
    """A portfolio must NOT take the GMM lnL branch.

    This is the other half of the clobber's damage: with sampler_method overwritten, a
    portfolio run set return_lnI, which flips rvs_integrand_is_lnL and changes how
    ln_weights_from_rvs reads the record.
    """
    src = _src(_LISA)
    assert 'if opts.sampler_method=="GMM"  and opts.internal_use_lnL:' in src
    i = src.index('if opts.sampler_method=="GMM"  and opts.internal_use_lnL:')
    assert 'return_lnI' in src[i:i + 300]
    # and it must not have been widened to the member flag
    assert 'use_gmm_args' not in src[i:i + 300], \
        "the return_lnI branch was widened to portfolios carrying a GMM member"


def _driver_def_names(path):
    """Every top-level function the driver defines."""
    return {n.name for n in ast.parse(_src(path)).body if isinstance(n, ast.FunctionDef)}


def _assert_helper_set_is_closed(ns, names, path):
    """Fail LOUDLY if an exec'd helper calls a driver helper that was not exec'd with it.

    Same guard as the other LISA harnesses.  The failure it prevents is silent: the
    callers here catch broadly, so a missing name turns into "the rescue did not fire"
    rather than a NameError naming the helper.
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


# ------------------------------------------------------------- the rescue actually fires
def _load_rescue(sampler_method):
    """Exec the rescue with a chosen opts.sampler_method."""
    # The record accessors ride along because the ported helpers resolve their weights
    # through _lw_of / _rvs_record_for.  Omitting one is NOT a visible NameError here --
    # _maybe_l0_rescue catches it and the rescue simply never fires, which shows up as
    # "the rescue did not fire", three layers from the cause.  Guarded below.
    names = ['_rvs_lnL_convention', 'ln_weights_from_rvs', '_rvs_len',
             '_rvs_is_export_resample', '_rvs_is_equal_weight',
             '_rvs_record_for', '_sampler_keeps_records', '_internal_record_of',
             '_rebound_record', '_lw_of', 'ln_weights_for_posterior',
             '_lnZ_of_rvs', '_kish_neff_of_rvs', '_lnZ_of_reserve_or_rvs',
             '_snapshot_pass_state', '_restore_pass_state', '_warm_seed_reserve_for',
             '_warm_seed_geometry', '_clear_warm_state', '_maybe_l0_rescue']
    import numpy as np
    defs = {n.name: n for n in ast.parse(_src(_LISA)).body
            if isinstance(n, ast.FunctionDef) and n.name in names}
    mod = ast.Module(body=[defs[n] for n in names], type_ignores=[])

    class _AV(object):
        @staticmethod
        def lnZ_from_reserve(r):
            return None

        @staticmethod
        def build_warm_seed(cols, lnL, lo, hi, axes, **kw):
            return np.asarray(cols, dtype=float), {'puffed': False, 'n_core': 3,
                                                   'rank_core': 3, 'dim': 3,
                                                   'rank_final': 3, 'n_puff': 0,
                                                   'puff_scale': 'auto'}

    opts = type('O', (), {
        'sampler_method': sampler_method, 'sampler_warmstart_retry_neff': 5.0,
        'sampler_l0_rescue_reject_dlnZ': 3.0, 'sampler_l0_rescue_accept_truncated': False,
        'sampler_l0_rescue_puff_scale': 'auto', 'sampler_l0_rescue_puff_width_frac': 0.005,
        'sampler_l0_rescue_puff_factor': 2.0,
        'sampler_sequential_warmstart_deltalnL': 15.0})()
    ns = {"numpy": np, "np": np, "opts": opts, "mcsamplerAdaptiveVolume": _AV}
    exec(compile(ast.fix_missing_locations(mod), "rescue", "exec"), ns)
    _assert_helper_set_is_closed(ns, names, _LISA)
    return ns


class _Sampler(object):
    def __init__(self):
        import numpy as np
        n = 3
        self._rvs = {'log_integrand': np.zeros(n), 'log_joint_prior': np.zeros(n),
                     'log_joint_s_prior': np.zeros(n),
                     'a': np.linspace(0.1, 0.9, n), 'b': np.linspace(0.2, 0.8, n)}
        self._warm_seed_reserve = None
        self.params_ordered = ['a', 'b']
        self.llim = {'a': 0.0, 'b': 0.0}
        self.rlim = {'a': 1.0, 'b': 1.0}
        self.portfolio_realizations = []
        self._warm = None
        self._warm_applied = False
        self.bootstrapped = None

    def identity_convert(self, x):
        return x

    def bootstrap_from_samples(self, seed, cover_frac=0.0):
        self.bootstrapped = seed

    def integrate(self, fn, *a, **k):
        return ('R2', 'V2', 42.0, {'warm': True})


@pytest.mark.parametrize("method", ['portfolio', 'AV'])
def test_rescue_fires_for_both_eligible_methods(method):
    """The end the whole fix serves.

    With the clobber, a portfolio carrying a GMM member arrived here as 'GMM' and this
    returned untouched -- no bootstrap, no warm pass, no message.
    """
    ns = _load_rescue(method)
    s = _Sampler()
    out = ns['_maybe_l0_rescue'](s, 'R1', 'V1', 1.0, {'cold': True},
                                 lambda *a, **k: None, (), {})
    assert s.bootstrapped is not None, "the rescue did not fire for %s" % method
    assert out[2] == 42.0


def test_rescue_declines_for_a_clobbered_method():
    """The failure mode itself, pinned: if the method ever reads 'GMM', the rescue is off.

    Not an argument that declining for standalone GMM is wrong -- it is correct, GMM has no
    bootstrap_from_samples in practice.  It documents that the guard is exactly what the
    clobber defeated, so the invariant above is what protects it.
    """
    ns = _load_rescue('GMM')
    s = _Sampler()
    ns['_maybe_l0_rescue'](s, 'R1', 'V1', 1.0, {'cold': True}, lambda *a, **k: None, (), {})
    assert s.bootstrapped is None


# --------------------------------------------------- the member-dispatch chain itself
def _member_loop(path):
    """The `for name in sampler_types:` loop body, as AST."""
    for node in ast.walk(ast.parse(_src(path), filename=path)):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == 'name'
                and isinstance(node.iter, ast.Name) and node.iter.id == 'sampler_types'):
            return node
    raise AssertionError("no `for name in sampler_types` loop in %s" % os.path.basename(path))


@pytest.mark.parametrize("path,label", [(_LISA, 'lisa'), (_MAIN, 'main')])
def test_member_dispatch_is_a_single_elif_chain(path, label):
    """A chain of separate `if`s reuses the previous member on an unmatched name.

    With `if name == 'AV': ... ; if name == 'GMM': ...` a name matching NOTHING falls
    through every test and leaves `sampler` bound to whatever it last held -- the plain
    MCSampler built before the chain, or on later iterations the PREVIOUS member -- which is
    then appended.  A typo in --sampler-portfolio silently produced a DUPLICATE member
    rather than an error.
    """
    loop = _member_loop(path)
    # Only the statements that DISPATCH ON THE MEMBER NAME.  The loop body also holds an
    # `if hasattr(sampler, 'xpy')` after the chain in both drivers, which is not part of it.
    dispatch = [st for st in loop.body
                if isinstance(st, ast.If)
                and any(isinstance(n, ast.Name) and n.id == 'name'
                        for n in ast.walk(st.test))]
    assert len(dispatch) == 1, (
        "%s: member dispatch is %d separate `if` statements, not one elif chain; an "
        "unmatched name reuses the previous member" % (label, len(dispatch)))


@pytest.mark.parametrize("path,label", [(_LISA, 'lisa'), (_MAIN, 'main')])
def test_an_unknown_member_name_raises(path, label):
    """The chain must END in an else that raises, not fall off silently."""
    node = [st for st in _member_loop(path).body
            if isinstance(st, ast.If)
            and any(isinstance(n, ast.Name) and n.id == 'name'
                    for n in ast.walk(st.test))][0]
    while isinstance(node, ast.If):
        tail = node.orelse
        if len(tail) == 1 and isinstance(tail[0], ast.If):
            node = tail[0]
            continue
        break
    assert tail, "%s: the member dispatch chain has no else clause" % label
    assert any(isinstance(st, ast.Raise) for st in tail), (
        "%s: the else clause does not raise, so an unknown --sampler-portfolio member is "
        "accepted silently" % label)


def test_plugin_pipelines_are_dispatched_before_the_error():
    """A plugin member (nflow, ...) must CONSTRUCT, not fall through to the raise.

    Checking that the string "known_pipelines" merely appears is not enough: it also appears
    in the error message, so deleting the whole dispatch branch left that check green.  Walk
    the chain and require a branch that both TESTS and SUBSCRIPTS known_pipelines.
    """
    node = [st for st in _member_loop(_LISA).body
            if isinstance(st, ast.If)
            and any(isinstance(n, ast.Name) and n.id == 'name'
                    for n in ast.walk(st.test))][0]
    found = False
    while isinstance(node, ast.If):
        tests_it = any(isinstance(a, ast.Attribute) and a.attr == 'known_pipelines'
                       for a in ast.walk(node.test))
        builds_it = any(isinstance(sub, ast.Subscript)
                        and any(isinstance(a, ast.Attribute) and a.attr == 'known_pipelines'
                                for a in ast.walk(sub.value))
                        for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])))
        if tests_it and builds_it:
            found = True
            break
        node = node.orelse[0] if (len(node.orelse) == 1
                                  and isinstance(node.orelse[0], ast.If)) else None
        if node is None:
            break
    assert found, ("no branch dispatches to mcsamplerPortfolio.known_pipelines, so a plugin "
                   "member falls through to the unknown-member error")


def test_the_unknown_member_error_names_what_is_known():
    assert "--sampler-portfolio: unknown member" in _src(_LISA)


def test_AC_is_accepted_as_an_alias():
    """main accepts 'AC' alongside 'adaptive_cartesian_gpu'; a portfolio spec is shared."""
    assert "name == 'AC'" in _src(_LISA)
