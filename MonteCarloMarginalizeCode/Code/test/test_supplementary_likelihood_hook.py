"""Guard the --supplementary-likelihood-factor plugin hook in CIP and EOSPosterior.

Motivation: both drivers carried an identical typo for months --

    supplemental_ln_likelhood_prep       = getattr(module, name_prep)   # assigned (misspelt)
    supplemental_ln_likelhood_parsed_ini = config                       # assigned (misspelt)
    supplemental_ln_likelihood_prep(config=supplemental_ln_likelihood_parsed_ini, ...)   # called

so the CALLED names were never the ASSIGNED ones and remained None from their initialisation.
Anyone supplying --supplementary-likelihood-factor-ini together with a plugin that defines a
prepare_<function> hook got `TypeError: 'NoneType' object is not callable`.  The plain hook (no
ini, or no prepare_) worked, which is why it survived: the hasattr() guard skips the whole block.

These tests are STATIC (ast-based).  The drivers are top-level scripts, not importable modules, so
we cannot exercise the block directly without running a full inference job; parsing catches the
whole class of defect at negligible cost.
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


def _tree(fname):
    path = os.path.join(BIN, fname)
    if not os.path.exists(path):
        pytest.skip("driver not present: %s" % fname)
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


def _assigned_names(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and \
                isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for n in ast.walk(node.optional_vars):
                if isinstance(n, ast.Name):
                    out.add(n.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, ast.comprehension):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    out.add(n.id)
    return out


@pytest.mark.parametrize("fname", DRIVERS)
def test_supplementary_names_are_assigned_before_use(fname):
    """Every `supplemental_*` identifier that is READ must also be ASSIGNED in the same file.

    This is the general form of the bug: a misspelt assignment leaves the read name bound to its
    initial None, which fails only on the rarely-exercised ini path.
    """
    tree = _tree(fname)
    assigned = _assigned_names(tree)
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id.startswith("supplemental_")}
    missing = sorted(used - assigned)
    assert not missing, (
        "%s reads supplementary-hook names it never assigns: %s -- almost certainly a typo in "
        "the assignment, which leaves the name None and breaks "
        "--supplementary-likelihood-factor-ini" % (fname, missing))


@pytest.mark.parametrize("fname", DRIVERS)
def test_no_dead_supplementary_assignments(fname):
    """No `supplemental_*` name may be ASSIGNED and then never READ.

    This is the general signature of the historical bug and the one that matters: the misspelt
    names WERE assigned, so a "used but never assigned" check does not see them -- the
    correctly-spelt names are assigned at initialisation.  What is anomalous is that the misspelt
    assignments are dead: nothing ever reads them.  A spelling blocklist would only catch this one
    typo; this catches any future variant.
    """
    tree = _tree(fname)
    loaded = {n.id for n in ast.walk(tree)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    stored = {n.id for n in ast.walk(tree)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
              and n.id.startswith("supplemental_")}
    dead = sorted(stored - loaded)
    assert not dead, (
        "%s assigns supplementary-hook name(s) that are never read: %s -- a dead assignment here "
        "means the value silently never reaches the code that uses it" % (fname, dead))


@pytest.mark.parametrize("fname", DRIVERS)
def test_no_known_likelihood_misspelling(fname):
    """Direct guard on the specific historical typo, in identifiers and attributes alike."""
    tree = _tree(fname)
    bad = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and "likelhood" in n.id}
    bad |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and "likelhood" in n.attr}
    assert not bad, "%s contains misspelt 'likelhood' identifier(s): %s" % (fname, sorted(bad))


@pytest.mark.parametrize("fname", DRIVERS)
def test_prepare_hook_is_actually_invoked(fname):
    """The prepare_<function> hook must still be called with config= and coords=.

    Guards against 'fixing' the typo by deleting the call rather than repairing the name.
    """
    tree = _tree(fname)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "supplemental_ln_likelihood_prep"]
    assert calls, "%s never calls supplemental_ln_likelihood_prep" % fname
    kw = {k.arg for c in calls for k in c.keywords}
    assert {"config", "coords"} <= kw, (
        "%s calls the prepare hook without config=/coords=; plugins rely on that signature "
        "(got %s)" % (fname, sorted(kw)))
