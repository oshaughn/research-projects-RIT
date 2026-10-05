"""`indx_ok` must keep one meaning in the posterior drivers.

Both drivers build one master post-integration mask and apply it to the whole sample set:

    indx_ok = np.logical_and(dat_logL > lnLmax - opts.lnL_offset, samples["joint_s_prior"] > 0)
    ...
    dat_logL = dat_logL[indx_ok]

Downstream blocks used to rebind that same name for unrelated throwaway cuts -- spin-prior
range cuts, and the "significant points" subset in the corner plots.  Nothing read the master
mask after its four applications, so this was never a live bug; it is a trap.  A later block
that indexes with `indx_ok` when the nearest rebinding is a few lines further up gets the
master mask instead, with the right length and the wrong meaning, and no error.

These tests are STATIC (ast-based).  The drivers are top-level scripts, not importable
modules, so the block cannot be exercised without a full inference job.
"""
import ast
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.abspath(os.path.join(HERE, "..", "bin"))
DRIVERS = [
    "util_ConstructIntrinsicPosterior_GenericCoordinates.py",
    "util_ConstructEOSPosterior.py",
]
NAME = "indx_ok"

SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _tree(fname):
    path = os.path.join(BIN, fname)
    # NOT pytest.skip: a missing driver disarms every assertion below, and a skip exits 0.
    assert os.path.exists(path), (
        "%s: driver not found at %s.  This guard cannot run, which is a failure, not a pass "
        "-- if the driver moved, update BIN/DRIVERS." % (fname, path))
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


def _refs(node, _top=True):
    """Module-level Name nodes for NAME under `node`.

    Nested scopes have their own `indx_ok` locals (the fit wrappers near the top of the CIP
    driver), which are fine and are skipped -- INCLUDING when the statement handed in is
    itself a def/class, which an earlier version of this descended into.  A nested scope that
    declares `global indx_ok` is not local, so those are reported (see _global_rebinds).
    """
    if _top and isinstance(node, SCOPES):
        return
    for ch in ast.iter_child_nodes(node):
        if isinstance(ch, SCOPES):
            continue
        if isinstance(ch, ast.Name) and ch.id == NAME:
            yield ch
        for sub in _refs(ch, _top=False):
            yield sub


def _global_rebinds(tree):
    """Nested scopes that declare `global indx_ok` AND assign it -- a module-level rebind."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, SCOPES):
            continue
        declares = any(isinstance(n, ast.Global) and NAME in n.names for n in ast.walk(node))
        if not declares:
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id == NAME and isinstance(n.ctx, ast.Store):
                out.append(n.lineno)
    return sorted(set(out))


def _anchor(tree):
    """Index of the module statement `dat_logL = dat_logL[indx_ok]`, the mask's defining use."""
    for i, stmt in enumerate(tree.body):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        tgt, val = stmt.targets[0], stmt.value
        if (isinstance(tgt, ast.Name) and tgt.id == "dat_logL"
                and isinstance(val, ast.Subscript)
                and isinstance(val.value, ast.Name) and val.value.id == "dat_logL"
                and any(_refs(val))):
            return i
    return None


def _is_sample_remask(stmt):
    """`samples[<key>] = samples[<key>][indx_ok]` -- the only shape allowed to follow the anchor.

    The object being re-masked must be `samples`.  Accepting any `Subscript = Subscript[Name]`
    let `prior_range_map['mc'] = weights[indx_ok]` through, which is exactly the consumer this
    file exists to catch.
    """
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
        return False
    tgt, val = stmt.targets[0], stmt.value
    if not (isinstance(tgt, ast.Subscript)
            and isinstance(tgt.value, ast.Name) and tgt.value.id == "samples"):
        return False
    return (isinstance(val, ast.Subscript)
            and isinstance(val.value, ast.Subscript)
            and isinstance(val.value.value, ast.Name) and val.value.value.id == "samples"
            and isinstance(val.slice, ast.Name) and val.slice.id == NAME
            and len(list(_refs(stmt))) == 1)


@pytest.mark.parametrize("fname", DRIVERS)
def test_master_mask_is_present(fname):
    """The anchor must exist -- otherwise every other assertion here passes vacuously."""
    tree = _tree(fname)
    assert _anchor(tree) is not None, (
        "%s: no module-level `dat_logL = dat_logL[%s]`; the master mask was renamed or "
        "removed, and this guard no longer guards anything" % (fname, NAME))


@pytest.mark.parametrize("fname", DRIVERS)
def test_master_mask_is_never_rebound(fname):
    """No module-level rebinding of `indx_ok` after the master mask has been applied."""
    tree = _tree(fname)
    i = _anchor(tree)
    stores = [(n.lineno, ast.unparse(stmt).splitlines()[0][:90])
              for stmt in tree.body[i + 1:]
              for n in _refs(stmt) if isinstance(n.ctx, ast.Store)]
    assert not stores, (
        "%s: `%s` is rebound after the master mask is applied, at %s.  Give the local cut its "
        "own name (e.g. indx_in_range, indx_significant)." % (fname, NAME, stores))


@pytest.mark.parametrize("fname", DRIVERS)
def test_only_remasking_reads_the_master_mask(fname):
    """After the anchor, `indx_ok` may only re-mask another `samples[...]` entry."""
    tree = _tree(fname)
    i = _anchor(tree)
    bad = [(next(_refs(stmt)).lineno, ast.unparse(stmt).splitlines()[0][:90])
           for stmt in tree.body[i + 1:]
           if any(_refs(stmt)) and not _is_sample_remask(stmt)]
    assert not bad, (
        "%s: `%s` is read after the master mask is applied, at %s.  A later block that indexes "
        "with this name gets the master mask, not a local cut." % (fname, NAME, bad))


@pytest.mark.parametrize("fname", DRIVERS)
def test_no_nested_scope_rebinds_the_master_mask(fname):
    """A helper may keep a local `indx_ok`; it may not `global indx_ok` and assign it.

    The three tests above read module-level statements only, so a `global` rebind inside a
    function is invisible to them while still changing the mask everything downstream uses.
    """
    tree = _tree(fname)
    lines = _global_rebinds(tree)
    assert not lines, (
        "%s: a nested scope declares `global %s` and assigns it, at line(s) %s.  That rebinds "
        "the master mask from inside a helper." % (fname, NAME, lines))
