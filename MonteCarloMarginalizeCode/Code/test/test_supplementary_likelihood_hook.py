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


@pytest.mark.parametrize("fname", DRIVERS)
def test_prepare_hook_is_invoked_without_an_ini(fname):
    """The prepare call must NOT sit inside `if opts.supplementary_likelihood_factor_ini:`.

    A plugin configured entirely by environment gets no ini, so gating preparation on one leaves it
    never told the sampling basis -- and a plugin that then has to guess what its input arrays are
    called can guess wrong with the right array count and no error.  `config=None` is a perfectly
    good argument; the basis is the part that cannot be reconstructed later.
    """
    tree = _tree(fname)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        gated_on_ini = any(isinstance(a, ast.Attribute) and
                           a.attr == "supplementary_likelihood_factor_ini"
                           for a in ast.walk(node.test))
        if not gated_on_ini:
            continue
        for inner in node.body:
            for c in ast.walk(inner):
                assert not (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                            and c.func.id == "supplemental_ln_likelihood_prep"), (
                    "%s only prepares the supplementary-likelihood plugin when an ini is supplied; "
                    "it must also be prepared (with config=None) so the plugin is always told "
                    "coords=" % fname)


@pytest.mark.parametrize("fname", DRIVERS)
def test_prepare_hook_is_told_the_sampling_basis(fname):
    """`coords=` must name the SAME list the driver splats into `sampler.integrate`.

    The sampler is handed one dimension per name in that starred list, so it calls
    `supplemental_ln_likelihood(*x)` with one array per SAMPLING coordinate, in that order.
    Declaring any other list -- notably the FIT basis `coord_names`, which differs from
    `low_level_coord_names` as soon as --parameter-implied or --parameter-nofit is used -- makes
    the plugin attach the wrong name to each array.  Nothing raises: the plugin simply evaluates
    at coordinates it has mislabelled.  Comparing the two identifiers is the whole invariant, and
    it is checkable statically; the runtime consequence of getting it wrong is exercised in
    test_nal_io.py::test_wrong_basis_from_the_driver_would_evaluate_at_the_wrong_point.
    """
    tree = _tree(fname)
    sampled = {n.value.id for c in ast.walk(tree)
               if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and c.func.attr == "integrate"
               for n in c.args
               if isinstance(n, ast.Starred) and isinstance(n.value, ast.Name)}
    assert sampled, (
        "%s never splats a coordinate-name list into sampler.integrate(); the basis the "
        "supplementary hook must be told can no longer be identified" % fname)
    declared = [k.value for c in ast.walk(tree)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                and c.func.id == "supplemental_ln_likelihood_prep"
                for k in c.keywords if k.arg == "coords"]
    assert declared, "%s never passes coords= to the prepare hook" % fname
    for node in declared:
        assert isinstance(node, ast.Name) and node.id in sampled, (
            "%s tells the prepare hook coords=%s, but integrates over %s -- the hook must be "
            "given the SAMPLING basis, since that is the order the plugin is called with"
            % (fname, ast.dump(node) if not isinstance(node, ast.Name) else node.id,
               sorted(sampled)))


# List methods that change the contents in place.  `+=` on a list is an AugAssign whose target is a
# Store of the same Name, so it is caught by the rebind check rather than this one.
_LIST_MUTATORS = ("append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse")


@pytest.mark.parametrize("fname", DRIVERS)
def test_prepare_hook_runs_after_the_sampling_basis_is_final(fname):
    """Nothing may change the declared coordinate list after the prepare hook has been told it.

    The plugin RECORDS the basis it is handed (nal_io copies it into module state); the sampler
    then calls the plugin with one array per coordinate in the FINAL list.  CIP builds that list
    in stages -- `--parameter-implied`/`--parameter-nofit` early, then `ordering` appended for a
    tabular-EOS run much later -- so preparing next to the plugin import snapshots a list that is
    one entry short of what the sampler supplies.  The plugin then either raises on the array
    count (nal_io does) or, worse, mislabels every array after the missing one.

    Checked by line number rather than by running the driver: these are top-level scripts, and the
    ordering is a property of the file, not of any particular run's flags.
    """
    tree = _tree(fname)
    calls = [c for c in ast.walk(tree)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == "supplemental_ln_likelihood_prep"]
    assert calls, "%s never calls supplemental_ln_likelihood_prep" % fname
    prepared_at = min(c.lineno for c in calls)
    declared = {k.value.id for c in calls for k in c.keywords
                if k.arg == "coords" and isinstance(k.value, ast.Name)}
    assert declared, "%s never passes a named coordinate list as coords=" % fname
    for name in sorted(declared):
        rebound = [n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and n.id == name
                   and isinstance(n.ctx, ast.Store)]
        mutated = [c.lineno for c in ast.walk(tree)
                   if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and c.func.attr in _LIST_MUTATORS
                   and isinstance(c.func.value, ast.Name) and c.func.value.id == name]
        late = sorted(ln for ln in rebound + mutated if ln > prepared_at)
        assert not late, (
            "%s prepares the supplementary-likelihood plugin at line %d with coords=%s, but that "
            "list is still changed afterwards at line(s) %s -- the plugin would be told a basis "
            "that is not the one the sampler integrates over. Prepare after the last change."
            % (fname, prepared_at, name, late))
