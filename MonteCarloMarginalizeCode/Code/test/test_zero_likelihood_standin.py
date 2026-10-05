"""The --zero-likelihood stand-in, exercised directly out of the ILE driver's own source.

Its companion, test_e2e_analytic_pipeline.py, runs ILE to completion and checks ln Z against a
closed form.  That cannot see everything.  An end-to-end marginal CANNOT distinguish a
permutation among right_ascension, phi_orb and psi, because all three are sampled uniformly on
[0, 2pi) and are independent: any factor's marginal is identical under the swap, so handing the
factor kwargs['psi'] where it wanted kwargs['phi_orb'] passes every lane.  That is exactly the
defect class the stand-in can have.

So this file checks the WIRING instead of the answer: the argument order against the driver's
own real call sites, the generated signature against the live likelihood_function signatures,
and that the values handed to the factor are the RAW sampled ones.  That part is pure AST +
exec, runs in a few seconds, and needs no data, no network and no GPU.

It also covers the device path, twice and deliberately.  Section 3 INJECTS a fake array module
that records every call, which runs everywhere.  Section 4 runs the same paths against REAL
cupy, because a fake can be more permissive than the thing it stands for, and a device claim
checked only against a fake is a claim nobody ran.  Section 4 skips where there is no usable
GPU -- a skip is not a pass; see _cupy_or_skip for how to make it run.

test_e2e_analytic_pipeline.py has device lanes of its own now (its section 4), so this file is
no longer the only place that sees a device.  What is still only here is the PER-SAMPLE view:
that gate reads a marginal, so it cannot distinguish a permutation of iid parameters however it
is run, on a device or off one.
"""
import ast
import inspect
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.abspath(os.path.join(HERE, "..", "bin"))
_ILE = os.path.join(BIN, "integrate_likelihood_extrinsic_batchmode")

pytestmark = pytest.mark.skipif(not os.path.exists(_ILE),
                                reason="ILE executable not in this tree")

# P attribute -> the sampled parameter it carries, for reading the vectorized call sites.
_P_ATTR = {"phi": "right_ascension", "theta": "declination", "phiref": "phi_orb",
           "incl": "inclination", "psi": "psi", "dist": "distance"}

# How a literal argument is reported.  Only the distance slot may hold one: the
# distance-marginalized sites pass 0 because distance has been integrated away.
_LITERAL = "<literal %r>"
_LITERAL_OK_IN = "distance"


def _driver_source():
    with open(_ILE) as f:
        return f.read()


def _driver_ns():
    """exec just the stand-in factory and its constant, out of the driver's real source."""
    src = _driver_source()
    tree = ast.parse(src)
    wanted = ("make_zero_likelihood_standin", "_SUPPLEMENT_ARG_ORDER",
              "_SUPPLEMENT_OPTIONAL_KWARGS")
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in wanted)
             or (isinstance(n, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in wanted for t in n.targets))]
    assert len(nodes) == 3, "expected the factory and its two constants, found %d" % len(nodes)
    # The factory is lifted out of the driver, so the module-level names its body uses have to
    # be supplied here.  Keep this list MINIMAL and explicit: anything added to it is a name the
    # driver gets from its own imports and this harness is standing in for.
    ns = {"inspect": inspect}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), _ILE, "exec"), ns)
    return ns


# How many times the driver calls the factor.  Pinned EXACTLY, not as a lower bound: with a
# `>=` the extractor could lose a site, or fail to see one, and still report "no mismatch".
# If this number changes, look at the new site and check its argument order by hand before
# updating it.
_EXPECTED_CALL_SITES = 5

# How many likelihood_function signatures the driver defines.  Pinned exactly, for the same
# reason as the call-site count: a lower bound lets the extractor lose one silently.
_EXPECTED_SIGNATURES = 8


def _textual_call_site_count():
    """Count the factor's call sites WITHOUT the AST, as a cross-check on the walker.

    The failure mode this exists for: a call the AST pass cannot see, but the SOURCE TEXT still
    shows -- e.g. one the walker's func/Name filter rejects -- contributes nothing to the
    comparison below, so the argument-order test would pass on a driver it had only partly read.

    It does NOT cover aliasing (`f = supplemental_ln_likelihood; f(...)`): that removes the token
    from the text as well, so both counts fall together and the equality still holds.  What
    catches aliasing is _EXPECTED_CALL_SITES being pinned exactly.  Verified by mutation, which
    is how this paragraph got corrected."""
    return len(re.findall(r"(?<![\w.])supplemental_ln_likelihood\s*\(", _driver_source()))


def _likelihood_signatures():
    """Every `def likelihood_function(...)` signature in the driver, as tuples of names."""
    out = []
    for n in ast.walk(ast.parse(_driver_source())):
        if isinstance(n, ast.FunctionDef) and n.name == "likelihood_function":
            out.append(tuple(a.arg for a in n.args.args))
    assert out, "no likelihood_function definitions found"
    return out


def _real_supplement_call_orders():
    """The positional argument order of every real supplemental_ln_likelihood(...) call.

    Returns (parameter-name tuple, keyword-name frozenset, source line) per site.  A site that
    passes a literal for distance -- the distance-marginalized ones pass 0 -- reports 'distance'
    for it, which is what that slot means.

    KEYWORDS ARE READ TOO, and that is not decoration.  Reading n.args alone checked the ORDER
    thoroughly and the ARITY not at all, which is how a documented six-argument contract sat
    next to three call sites passing xpy=xpy_default without anything noticing."""
    out = []
    for n in ast.walk(ast.parse(_driver_source())):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "supplemental_ln_likelihood"):
            continue
        names = []
        for i, a in enumerate(n.args):
            if isinstance(a, ast.Name):
                names.append(a.id)
            elif isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name):
                assert a.attr in _P_ATTR, "unmapped attribute %s.%s at line %d" % (
                    a.value.id, a.attr, n.lineno)
                names.append(_P_ATTR[a.attr])
            elif isinstance(a, ast.Constant):
                # Reported as a literal, NOT resolved to a parameter name.  Naming it "distance"
                # outright made the comparison agree with itself wherever a literal appeared;
                # which slot may carry one is checked positionally below.
                names.append(_LITERAL % (a.value,))
            else:
                pytest.fail("un-readable argument %d at line %d of the ILE" % (i, n.lineno))
        kwargs = frozenset(k.arg for k in n.keywords if k.arg is not None)
        out.append((tuple(names), kwargs, n.lineno))
    assert out, "no supplemental_ln_likelihood call sites found in the ILE"
    return out


# ---------------------------------------------------------------------------------------
# 1. the argument order, against the driver's own call sites

def test_the_stand_in_uses_the_same_argument_order_as_every_real_call_site():
    """THE CHECK THE END-TO-END GATE CANNOT MAKE.  phi_orb, psi and right_ascension are iid
    uniform on [0, 2pi), so no marginal can tell a permutation of them apart; only the wiring
    can be read.  Every real site is read, not one of them, so a site that disagrees with the
    others is a failure here rather than a silent inconsistency."""
    order = _driver_ns()["_SUPPLEMENT_ARG_ORDER"]
    sites = _real_supplement_call_orders()
    assert len(sites) == _EXPECTED_CALL_SITES, (
        "the AST pass found %d call sites, expected %d.  If a site was added, read its argument "
        "order and update _EXPECTED_CALL_SITES; if one vanished, the extractor broke."
        % (len(sites), _EXPECTED_CALL_SITES))
    for names, _kwargs, lineno in sites:
        resolved = []
        for idx, nm in enumerate(names):
            if not nm.startswith("<literal"):
                resolved.append(nm)
                continue
            slot = order[idx] if idx < len(order) else "(beyond the contract)"
            assert slot == _LITERAL_OK_IN, (
                "the call site at line %d passes %s in the %r slot.  Only %r may be a literal, "
                "because it is the one argument the driver marginalizes away; a literal "
                "anywhere else is a value the factor cannot distinguish from a sampled one."
                % (lineno, nm, slot, _LITERAL_OK_IN))
            resolved.append(slot)
        assert tuple(resolved) == tuple(order), (
            "the --zero-likelihood stand-in passes the factor %r, but the real call site at "
            "line %d passes %r" % (tuple(order), lineno, tuple(resolved)))


def test_the_extractor_reads_every_call_site_the_source_has():
    """The guard on the guard.  A site the AST pass cannot see would make the order comparison
    above pass on a driver it had only partly read, which is the quiet way this whole file stops
    being worth anything."""
    ast_count = len(_real_supplement_call_orders())
    textual = _textual_call_site_count()
    assert ast_count == textual, (
        "the AST pass sees %d call sites but the source text has %d: at least one call is in a "
        "shape the walker does not recognise, and its argument order is NOT being checked."
        % (ast_count, textual))
    assert textual == _EXPECTED_CALL_SITES, (
        "the driver has %d call sites, expected %d; read the new one before updating the number."
        % (textual, _EXPECTED_CALL_SITES))


def test_every_keyword_the_call_sites_pass_is_declared():
    """THE ARITY AXIS.  The positional order was checked here from the start; the keywords were
    not, so a factor written to the driver's own stated contract raised TypeError on three of
    the five call sites and worked on the other two.  A factor has to ACCEPT every keyword any
    site passes, with a default, because the sites disagree about passing them."""
    declared = frozenset(_driver_ns()["_SUPPLEMENT_OPTIONAL_KWARGS"])
    sites = _real_supplement_call_orders()
    used = frozenset().union(*[kw for _n, kw, _l in sites])
    undeclared = sorted(used - declared)
    assert not undeclared, (
        "call sites pass keyword(s) %s that _SUPPLEMENT_OPTIONAL_KWARGS does not declare.  A "
        "factor written to the documented contract will raise TypeError there.  Declare them, "
        "document them in --help, and give the shipped example a default for them."
        % undeclared)
    unused = sorted(declared - used)
    assert not unused, (
        "_SUPPLEMENT_OPTIONAL_KWARGS declares %s, which no call site passes any more.  Drop it "
        "rather than leaving the contract describing a keyword nothing sends." % unused)


def test_the_documented_contract_names_every_optional_keyword():
    """--help is where a plugin author reads the contract, so the keyword has to reach it."""
    src = _driver_source()
    i = src.index('"--supplementary-likelihood-factor-function"')
    text = src[i:src.index("\n", i)]
    for kw in _driver_ns()["_SUPPLEMENT_OPTIONAL_KWARGS"]:
        assert kw in text, (
            "--supplementary-likelihood-factor-function's help does not mention the %r keyword "
            "that the call sites pass" % kw)


def test_the_shipped_example_factor_accepts_the_real_call_shapes():
    """The example is what someone copies.  Ship one that works on ALL five sites, not on the
    two the gate happens to exercise."""
    import numpy as _np
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import analytic_supplement_for_e2e as ex
    a = _np.zeros(3)
    order = _driver_ns()["_SUPPLEMENT_ARG_ORDER"]
    ex.ln_analytic_factor(*([a] * len(order)))                       # the two bare sites
    for kw in _driver_ns()["_SUPPLEMENT_OPTIONAL_KWARGS"]:           # and the three that do not
        ex.ln_analytic_factor(*([a] * len(order)), **{kw: _np})


def test_the_shipped_example_survives_object_dtype_draws():
    """The example's cast, pinned on the keyword that is easy to lose.

    mcsampler (--sampler-method adaptive_cartesian) hands its integrand OBJECT-dtype draws, on
    which np.cos raises "loop of ufunc does not support argument 0 of type float".  What makes
    the example work on them is the EXPLICIT `dtype=float` on its asarray -- and that same
    explicit dtype is what lets cupy.asarray accept an object array at all, since
    cupy.asarray(obj) without one raises ValueError: Unsupported dtype object (measured, cupy
    12.0.0; see the comment on the line itself).  One keyword carries the host path and the
    device path together.

    WHAT THIS ADDS.  The end-to-end gate's adaptive_cartesian lane runs with B = 2, so it does
    reach both casts and does redden if either loses its dtype -- but at ~3 minutes and a full
    ILE run, reported as a ln Z failure rather than as a cast.  This is the same check in ~5 s,
    in the file that already reads this contract, naming the cause.  It runs on a CPU-only
    runner, where every device lane in this file and in the e2e gate skips.

    Verified by mutation: deleting `dtype=float` from either asarray reddens this test."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import analytic_supplement_for_e2e as ex
    obj = np.array([0.1, 0.2, 0.3], dtype=object)
    flt = np.asarray(obj, dtype=float)
    b_was = ex.B_COEFF
    try:
        # B = 0 exercises only the phi_orb cast; B != 0 also reaches the inclination one, which
        # the gate's default lane never does.  Restored, because this is module state.
        for b in (0.0, 2.0):
            ex.B_COEFF = b
            want = ex.ln_analytic_factor(*([flt] * 6))
            got = ex.ln_analytic_factor(*([obj] * 6))
            assert np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float)), (
                "B_COEFF=%r: the example gives a different answer on object-dtype draws (%r) "
                "than on the same values as float64 (%r)" % (b, got, want))
    finally:
        ex.B_COEFF = b_was


def test_the_stand_in_passes_a_keyword_only_to_a_factor_that_takes_it():
    """A factor without xpy must not be handed one, or fixing the contract would break every
    plugin written for the two sites that never passed it."""
    make = _driver_ns()["make_zero_likelihood_standin"]
    sig = ("right_ascension", "declination", "phi_orb", "inclination", "psi", "distance")
    kw = _driver_ns()["_SUPPLEMENT_OPTIONAL_KWARGS"][0]
    seen = {}

    def takes_it(right_ascension, declination, phi_orb, inclination, psi, distance, **k):
        seen["with"] = dict(k)
        return np.zeros(len(right_ascension))

    # NAMED explicitly, which is the shape --help tells plugin authors to write and the shape
    # the shipped example uses.  Only the **kwargs case was driven before, so deleting the
    # `"xpy" in _params` clause from the driver left every test green.
    def names_it(right_ascension, declination, phi_orb, inclination, psi, distance, xpy=None):
        seen["named"] = xpy
        return np.zeros(len(right_ascension))

    def takes_it_not(right_ascension, declination, phi_orb, inclination, psi, distance):
        seen["without"] = True
        return np.zeros(len(right_ascension))

    xpy = _FakeXpy()
    args = {nm: np.zeros(3) for nm in sig}
    make(sig, True, takes_it, _DEFAULTS, xpy)(**args)
    assert kw in seen["with"] and seen["with"][kw] is xpy, \
        "a factor that accepts %r was not given it: %r" % (kw, seen.get("with"))
    make(sig, True, names_it, _DEFAULTS, xpy)(**args)
    assert seen.get("named") is xpy, \
        "a factor NAMING %r in its signature was not given it: %r" % (kw, seen.get("named"))
    make(sig, True, takes_it_not, _DEFAULTS, np)(**args)    # must not raise TypeError
    assert seen.get("without") is True


# Signatures whose BODY contains no supplemental_ln_likelihood(...) call at all, keyed by the
# parameter tuple so the pin survives line numbers moving.  A factor is silently ignored on
# these paths.  Pre-existing, pinned here so it cannot widen, and so FIXING it reddens this test
# and forces someone to decide deliberately.
_SIGNATURES_THAT_DROP_THE_FACTOR = {
    # no --time-marginalization, --psi-marginalization
    ("right_ascension", "declination", "t_ref", "phi_orb", "inclination", "distance"),
    # no --time-marginalization, the driver's plain default path
    ("right_ascension", "declination", "t_ref", "phi_orb", "inclination", "psi", "distance"),
    # --rom-integrate-intrinsic
    ("right_ascension", "declination", "phi_orb", "inclination", "psi", "distance", "q"),
}


def _signatures_by_whether_they_call_the_factor():
    calls, drops = set(), set()
    for n in ast.walk(ast.parse(_driver_source())):
        if not (isinstance(n, ast.FunctionDef) and n.name == "likelihood_function"):
            continue
        sig = tuple(a.arg for a in n.args.args)
        used = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == "supplemental_ln_likelihood" for c in ast.walk(n))
        (calls if used else drops).add(sig)
    return calls, drops


def test_the_paths_that_silently_drop_the_factor_are_the_known_ones():
    """THE TRAP THE STAND-IN CREATES.  make_zero_likelihood_standin applies the supplementary
    factor for EVERY signature, but only five of the eight real bodies call it.  So a plugin
    validated under --zero-likelihood and then run in production WITHOUT --time-marginalization
    is silently ignored, with the startup banner still announcing it.

    The hole is pre-existing; this pins its shape.  WHAT IT PINS IS A PARTITION OF PARAMETER
    TUPLES, NOT OF DEFINITIONS.  The driver's eight likelihood_function defs carry only six
    distinct tuples -- three of them share (right_ascension, declination, phi_orb, inclination,
    psi, distance) -- so a NEW dropping definition whose tuple is already recorded here is
    invisible to this test.  What catches that one is _EXPECTED_SIGNATURES, pinned exactly in
    test_the_stand_in_reproduces_every_live_likelihood_signature.

    AND THE PAIR IS STILL NOT A GENERAL GUARANTEE.  Both tests reason about DEFINITIONS.  A new
    silently-ignored configuration that lives INSIDE an existing calling body -- wrapping that
    body's supplemental_ln_likelihood(...) in a new `if` -- changes no signature and moves no
    tuple between the two sets, so neither test sees it.  Demonstrated by a review, which did
    exactly that to the call at driver line 4395 and got a fully green file.  So do not write
    here that the two together catch "a new silently-ignored configuration"; what they catch is
    a new silently-ignored DEFINITION.

    If this set widens, that is a new silently-ignored definition.  If it shrinks, someone fixed
    a path and should delete the entry rather than let this file keep describing a hole that
    closed."""
    calls, drops = _signatures_by_whether_they_call_the_factor()
    # A tuple in BOTH sets means two definitions share a signature and disagree about calling
    # the factor.  That is the one shape the partition below cannot express -- "this signature
    # drops the factor" stops being a well-formed statement -- and it would otherwise surface
    # only as a confusing widening of `drops`.
    assert calls.isdisjoint(drops), (
        "signature(s) %r have one likelihood_function definition that calls the supplementary "
        "factor and another that does not, so whether the factor applies no longer follows from "
        "the signature.  The recorded set below cannot describe that; give the two definitions "
        "distinguishable signatures, or record the hole some other way."
        % sorted(calls & drops))
    assert drops == _SIGNATURES_THAT_DROP_THE_FACTOR, (
        "the set of likelihood_function signatures that never call the supplementary factor "
        "changed.\n  now dropping: %r\n  recorded:    %r\nIf a path was fixed, remove it here "
        "and from --help.  If one was added, a new configuration now ignores the factor."
        % (sorted(drops), sorted(_SIGNATURES_THAT_DROP_THE_FACTOR)))


def test_the_help_warns_that_some_paths_drop_the_factor():
    """A plugin author reads --help, not this file."""
    src = _driver_source()
    i = src.index('"--supplementary-likelihood-factor-function"')
    text = src[i:src.index("\n", i)]
    for phrase in ("time-marginalization", "silently ignored"):
        assert phrase in text, (
            "--supplementary-likelihood-factor-function's help no longer warns that some paths "
            "drop the factor (missing %r)" % phrase)


# What the DRIVER's own call sites pass as supplement_defaults.  Read out of the source, not
# reproduced here from memory: the values below were unpinned entirely until a review mutated
# them to nonsense and every test still passed.  Written as ordinary source text; both sides of
# the comparison are normalized through ast.unparse, so a multi-token value can be spelled here
# the way it would be written in the driver.
_EXPECTED_DEFAULT_SOURCES = {
    "right_ascension": "P.phi",
    "declination": "P.theta",
    "phi_orb": "P.phiref",
    "inclination": "P.incl",
    "psi": "P.psi",
    "distance": "0.0",
}

# How many times the driver CONSTRUCTS the stand-in.  Pinned EXACTLY, for the same reason as
# _EXPECTED_CALL_SITES and _EXPECTED_SIGNATURES: the check below reads every site it is given,
# but with the count unpinned a SECOND site could appear and simply never be looked at.
# Demonstrated by mutation: the reader used to return on the first ast.Dict it walked into, so
# splitting the one site into two branches and corrupting the second branch's defaults left
# every test in this file green (all 23 of them, as it then stood) -- coverage decided by walk
# order rather than by the driver.
# If this number changes, read the new site's defaults by hand before updating it.
_EXPECTED_STANDIN_CONSTRUCTIONS = 1


def _normalize_expr(text):
    """Canonical source text for an expression, for comparing a hand-written value to an
    unparsed one.

    `ast.unparse(v).replace(" ", "")` normalized only the driver's side, and did it by deleting
    every space -- which collapses 'a b' with 'ab' and f"{x} {y}" with f"{x}{y}", and means a
    multi-token expected value written naturally above ("P.phi if opts.q else P.psi") could
    never match anything.  Round-tripping BOTH sides through ast.unparse normalizes spacing
    without reaching inside a string literal."""
    return ast.unparse(ast.parse(text, mode="eval").body)


def _driver_supplement_defaults():
    """Every supplement_defaults dict the driver hands make_zero_likelihood_standin.

    Returns [(line, {name: source text}), ...] -- EVERY construction site, not whichever one
    ast.walk reached first."""
    out = []
    for n in ast.walk(ast.parse(_driver_source())):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "make_zero_likelihood_standin"):
            continue
        dicts = [a for a in n.args if isinstance(a, ast.Dict)]
        assert len(dicts) == 1, (
            "the make_zero_likelihood_standin call at line %d passes %d dict literals; this "
            "reader cannot say which one is supplement_defaults." % (n.lineno, len(dicts)))
        got = {}
        for k, v in zip(dicts[0].keys, dicts[0].values):
            # k is None for `**spread`, and a computed key is an expression this reader cannot
            # resolve.  Both used to be an AttributeError on k.value, which reads like a broken
            # test rather than like a contract the test can no longer check.
            assert isinstance(k, ast.Constant), (
                "the supplement_defaults dict at line %d has a key this reader cannot resolve "
                "(%s).  Spell the defaults out with literal keys, or teach this reader the new "
                "shape -- do not leave them unchecked."
                % (n.lineno, "**spread" if k is None else ast.dump(k)))
            got[k.value] = ast.unparse(v)
        out.append((n.lineno, got))
    assert out, "no make_zero_likelihood_standin call site found in the ILE"
    return out


def test_the_driver_supplies_the_defaults_it_documents():
    """These are the values a factor sees for an argument the live signature does not sample, so
    a wrong one is a silent wrong answer for any plugin that reads it.  Nothing tested them: the
    wiring tests below use their own _DEFAULTS, and the only gate lane that reaches a default is
    the distance-marginalized one, whose factor ignores distance.  Mutating the driver's literal
    to P.dist (SI, ~1e24) and P.phiref left all twenty tests green."""
    sites = _driver_supplement_defaults()
    assert len(sites) == _EXPECTED_STANDIN_CONSTRUCTIONS, (
        "the driver constructs the stand-in at %d sites (lines %r), expected %d.  Every site is "
        "checked below, so this is not a formality: read the new one's defaults by hand before "
        "updating _EXPECTED_STANDIN_CONSTRUCTIONS."
        % (len(sites), [ln for ln, _ in sites], _EXPECTED_STANDIN_CONSTRUCTIONS))
    want = {k: _normalize_expr(v) for k, v in _EXPECTED_DEFAULT_SOURCES.items()}
    order = set(_driver_ns()["_SUPPLEMENT_ARG_ORDER"])
    for lineno, got in sites:
        assert got == want, (
            "the driver's supplement_defaults at line %d changed.\n  now: %r\n  was: %r\n"
            "distance must stay 0.0, which is what the two distance-marginalized call sites pass "
            "the factor; the angles must stay the template's own value for that parameter."
            % (lineno, got, want))
        assert set(got) == order, (
            "the defaults dict at line %d no longer covers exactly the factor's arguments: %r"
            % (lineno, sorted(got)))


def test_the_argument_order_is_the_documented_one():
    """A second, independent statement of the same fact, so that changing BOTH the stand-in and
    the call sites together still trips something.  This is the order in --help."""
    assert _driver_ns()["_SUPPLEMENT_ARG_ORDER"] == (
        "right_ascension", "declination", "phi_orb", "inclination", "psi", "distance")


# ---------------------------------------------------------------------------------------
# 2. the generated signature

def test_the_stand_in_reproduces_every_live_likelihood_signature():
    """Consumers decide what to pass by reading func.__code__.co_varnames[:co_argcount] --
    mcsampler does exactly that.  A `def zero_like(*args, **kwargs)` stand-in reports ZERO
    arguments there, which killed --zero-likelihood --sampler-method adaptive_cartesian."""
    make = _driver_ns()["make_zero_likelihood_standin"]
    sigs = _likelihood_signatures()
    assert len(sigs) == _EXPECTED_SIGNATURES, (
        "found %d likelihood_function signatures, expected %d.  A lower bound here would let the "
        "extractor lose one and still report every signature reproduced."
        % (len(sigs), _EXPECTED_SIGNATURES))
    for sig in sigs:
        f = make(sig, True, None, _DEFAULTS, np)
        got = f.__code__.co_varnames[:f.__code__.co_argcount]
        assert got == sig, "stand-in signature %r does not match likelihood_function's %r" % (
            got, sig)


_DEFAULTS = {"right_ascension": 0.1, "declination": 0.2, "phi_orb": 0.3,
             "inclination": 0.4, "psi": 0.5, "distance": 0.0}


def _spy():
    seen = {}

    def supplement(*args):
        seen["args"] = args
        return np.zeros(len(args[0]))
    return supplement, seen


@pytest.mark.parametrize("sig", _likelihood_signatures())
def test_the_factor_gets_the_raw_sampled_value_for_every_argument(sig):
    """Positionally AND by keyword, for every signature the driver can build -- because the
    integrand is called both ways (mcsampler by keyword, AV's selfish update positionally), and
    a hand-written unpack was correct for exactly one of the eight signatures."""
    make = _driver_ns()["make_zero_likelihood_standin"]
    order = _driver_ns()["_SUPPLEMENT_ARG_ORDER"]
    # a distinct, recognisable value per sampled parameter
    vals = {nm: np.arange(4, dtype=float) + 100.0 * (i + 1) for i, nm in enumerate(sig)}
    for how in ("positional", "keyword"):
        supplement, seen = _spy()
        f = make(sig, True, supplement, _DEFAULTS, np)
        if how == "positional":
            f(*[vals[nm] for nm in sig])
        else:
            f(**vals)
        got = seen["args"]
        assert len(got) == len(order)
        for nm, g in zip(order, got):
            want = vals[nm] if nm in sig else _DEFAULTS[nm]
            assert np.all(np.asarray(g) == np.asarray(want)), (
                "%s call, signature %r: the factor's %r argument got %r, expected %r "
                "(raw sampled value, or the default when not sampled)" % (how, sig, nm, g, want))


def test_a_signature_missing_a_factor_argument_with_no_default_is_refused():
    """Silently substituting something would be a wrong answer with no diff line on it."""
    make = _driver_ns()["make_zero_likelihood_standin"]
    supplement, _ = _spy()
    with pytest.raises(ValueError) as exc:
        make(("right_ascension", "declination"), True, supplement,
             {"right_ascension": 0.0}, np)
    assert "no value available" in str(exc.value)


# ---------------------------------------------------------------------------------------
# 3. the convention, and the array module

class _FakeXpy(object):
    """Stands in for cupy: every array it makes is tagged, so a stand-in that reached for
    numpy instead produces an untagged array and is caught.

    Its value is that it runs EVERYWHERE, including on a runner with no cupy, where section 4
    below and the e2e gate's device lanes both skip.  It is not a substitute for them: a fake
    accepts things cupy refuses, so the two are paired on purpose."""

    def __init__(self):
        self.calls = []

    def _tag(self, a):
        a = np.asarray(a)
        out = a.view(_TaggedArray)
        out.from_fake = True
        return out

    def zeros(self, n):
        self.calls.append(("zeros", n))
        return self._tag(np.zeros(n))

    def ones(self, n):
        self.calls.append(("ones", n))
        return self._tag(np.ones(n))

    def exp(self, x):
        self.calls.append(("exp", None))
        return self._tag(np.exp(np.asarray(x, dtype=float)))


class _TaggedArray(np.ndarray):
    """The tag has to survive arithmetic, or the check below passes on the base array alone and
    says nothing about `base + supp`.  __array_finalize__ is what carries it through a ufunc."""
    from_fake = False

    def __array_finalize__(self, obj):
        if obj is not None:
            self.from_fake = getattr(obj, "from_fake", False)


@pytest.mark.parametrize("return_lnL,expect_maker", [(True, "zeros"), (False, "ones")])
def test_the_base_array_comes_from_the_injected_module_not_numpy(return_lnL, expect_maker):
    make = _driver_ns()["make_zero_likelihood_standin"]
    sig = ("right_ascension", "declination", "phi_orb", "inclination", "psi", "distance")
    xpy = _FakeXpy()
    supplement, _ = _spy()
    f = make(sig, return_lnL, supplement, _DEFAULTS, xpy)
    out = f(**{nm: np.zeros(3) for nm in sig})
    assert [c[0] for c in xpy.calls][0] == expect_maker, \
        "the stand-in built its base with %r, expected %r" % (xpy.calls, expect_maker)
    assert getattr(out, "from_fake", False), \
        "the result did not come from the injected array module: %r" % type(out)


def test_the_two_conventions_give_the_right_arithmetic():
    """return_lnL says whether the INTEGRAND is ln L or L, which is not the same question as
    opts.internal_use_lnL; they disagree for adaptive_cartesian, which is the one lane that
    exercises the linear branch below."""
    make = _driver_ns()["make_zero_likelihood_standin"]
    sig = ("right_ascension", "declination", "phi_orb", "inclination", "psi", "distance")
    supp_vals = np.array([0.5, -1.25, 2.0])

    def supplement(*args):
        return supp_vals

    kw = {nm: np.zeros(3) for nm in sig}
    log_out = make(sig, True, supplement, _DEFAULTS, np)(**kw)
    lin_out = make(sig, False, supplement, _DEFAULTS, np)(**kw)
    assert np.allclose(log_out, supp_vals), "ln-convention: expected 0 + ln f"
    assert np.allclose(lin_out, np.exp(supp_vals)), "linear convention: expected 1 * f"
    # and with no factor at all, the exact zero / one the option promises
    assert np.allclose(make(sig, True, None, _DEFAULTS, np)(**kw), 0.0)
    assert np.allclose(make(sig, False, None, _DEFAULTS, np)(**kw), 1.0)


# ---------------------------------------------------------------------------------------
# 4. the device path, on a REAL GPU

def _no_gpu(reason):
    """Skip, or FAIL when the environment promised a device.

    The shell gate applies this rule too (RIFT_CI_REQUIRE_GPU=1 makes any skip fatal there),
    but living only in the shell means `pytest <this file>` on the GPU runner -- what someone
    does to reproduce a CI failure -- reports green with every device lane skipped.  The rule
    belongs with the tests.  test_time_marginalization_quadrature.py converts skip to failure
    the same way, for the same reason.

    RIFT_CI_REQUIRE_GPU is read from the ambient environment on purpose: _child_env in the e2e
    gate strips RIFT_* from ILE CHILDREN, which is a different question from what this pytest
    process was promised."""
    if os.environ.get("RIFT_CI_REQUIRE_GPU", "0") == "1":
        pytest.fail("RIFT_CI_REQUIRE_GPU=1 promised a usable device and there is none: %s.  "
                    "On this runner a skipped device lane is a failure, not a pass." % reason)
    pytest.skip("%s  A skip is NOT a pass: pin CUDA_VISIBLE_DEVICES to a slot the installed "
                "cupy supports and rerun." % reason)


# Which visible device this cupy can actually build a kernel for, decided ONCE, in a
# SUBPROCESS.  Cached because three tests ask -- INCLUDING a failure verdict, so a probe that
# times out costs one timeout and not one per test.
_USABLE_SLOT = None

_SLOT_PROBE = r"""
import numpy as np
def _flat(e):
    return ("%s: %s" % (type(e).__name__, e)).replace("\n", " ")[:150]
try:
    import cupy
except Exception as e:
    print("VERDICT NOCUPY %s" % _flat(e)); raise SystemExit(0)
bad = []
try:
    n = cupy.cuda.runtime.getDeviceCount()
except Exception as e:
    print("VERDICT NOQUERY %s" % _flat(e)); raise SystemExit(0)
for d in range(n):
    try:
        with cupy.cuda.Device(d):
            cupy.asnumpy(cupy.cos(cupy.asarray(np.zeros(2), dtype=float)))
    except Exception as e:
        bad.append("%d:%s" % (d, type(e).__name__)); continue
    print("VERDICT SLOT %d" % d); raise SystemExit(0)
print("VERDICT NOSLOT %s" % (",".join(bad) or "no devices visible"))
"""


def _cupy_or_skip():
    """Real cupy, pinned to a device this cupy can actually build a kernel for, or a skip
    saying why.

    TWO distinct failures, and conflating them is how a device claim goes unchecked.  `import
    cupy` fails outright on a host with no CUDA runtime (ldas-grid, and every CI runner).  It
    SUCCEEDS on the CIT GPU head nodes while the visible device is one this cupy cannot compile
    for: ldas-pcdev13 slots 0-2 and all of ldas-pcdev11 are Blackwell cc 12.0, and cupy 12.0.0
    answers `nvrtc: error: invalid value for --gpu-architecture`.  So the probe RUNS a kernel
    rather than trusting the import, and it probes at dispatch, because the slot map moves.

    THE PROBE RUNS IN A SUBPROCESS, and that is not tidiness.  Finding out whether a device
    works means creating a CUDA context on it, and a context is not undone by finishing with the
    device: `with cupy.cuda.Device(d):` restores the CURRENT DEVICE and destroys nothing.
    Measured on ldas-pcdev2 with two devices visible, the first unusable -- probing in-process
    with `.use()` and probing in-process with `with` BOTH leave two contexts on one pid, 252 and
    446 MB, one of them on a card this cupy cannot even build for.  On pcdev11/13 that is three,
    on a shared node.  Probing out of process leaves this process holding exactly one, on the
    device it actually uses; the rejected contexts die with the child.  (An earlier version of
    this comment claimed `with` fixed the leak.  It does not; that was measured afterwards.)

    A SKIP HERE IS NOT A PASS.  Pin CUDA_VISIBLE_DEVICES to a slot this cupy supports and run
    the file again; as of 2026-09-17 ldas-pcdev2 slot 0 (A100, cc 8.0) and ldas-pcdev13 slot 3
    (RTX 2080 Ti, cc 7.5) work.  The reason string says which failure happened."""
    global _USABLE_SLOT
    if _USABLE_SLOT is None:
        try:
            proc = subprocess.run([sys.executable, "-c", _SLOT_PROBE], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=600)
        except subprocess.TimeoutExpired:
            # Cached, or each of the three device tests pays its own 600 s.
            _USABLE_SLOT = "TIMEOUT the device probe did not finish in 600 s"
        else:
            verdicts = [l for l in proc.stdout.decode().splitlines()
                        if l.startswith("VERDICT ")]
            # Flattened: this probe exists because a multi-line message ruins a verdict line,
            # and the fallback must not reintroduce one.
            _USABLE_SLOT = (verdicts[-1][len("VERDICT "):] if verdicts
                            else "NOVERDICT rc=%d %s"
                            % (proc.returncode,
                               " ".join(proc.stdout.decode().split())[-200:]))
    if not _USABLE_SLOT.startswith("SLOT "):
        _no_gpu("no usable GPU here -- %s." % _USABLE_SLOT)
    import cupy
    # The only device this process ever touches, so the only context it holds.  Anything failing
    # BETWEEN a good verdict and here is a test ERROR rather than a skip, deliberately: the
    # probe just proved this device works, so a failure now is a real change, not an absence.
    cupy.cuda.Device(int(_USABLE_SLOT.split()[1])).use()
    return cupy


@pytest.mark.parametrize("b_coeff", [0.0, 2.0])
def test_the_shipped_example_really_runs_on_a_gpu(b_coeff):
    """THE DEVICE HALF, RUN RATHER THAN ARGUED.

    The example's one cast has to do two jobs: object dtype to float, for the draws mcsampler
    hands --sampler-method adaptive_cartesian, and host to device, because the three call sites
    that pass xpy do `lnL += factor(...)` with lnL on the device.  Nothing that runs on a
    CPU-only runner can see the second job: the e2e gate's own device lanes skip there too, and
    _FakeXpy above is a stand-in that can be more permissive than the thing it stands for.

    So this runs the real module with xpy=cupy, on BOTH input kinds, and checks the result is
    on the device and equals the numpy arm.  It is also the test that refuses the rewrite the
    comment on that line warns about: xpy.asarray(np.asarray(x, dtype=float)) raises TypeError
    on a cupy argument, so the device-float64 case below goes red.  Verified by mutation on
    ldas-pcdev2.

    b_coeff = 0 exercises only the phi_orb cast; b_coeff != 0 also reaches the inclination one,
    which the default B = 0 never does."""
    cupy = _cupy_or_skip()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import analytic_supplement_for_e2e as ex
    vals = np.array([0.1, 1.2, 2.3, 3.4])
    b_was = ex.B_COEFF
    try:
        ex.B_COEFF = b_coeff
        want = np.asarray(ex.ln_analytic_factor(*([vals] * 6), xpy=np), dtype=float)
        cases = (("device float64", cupy.asarray(vals, dtype=float)),
                 ("host object dtype", np.asarray(vals, dtype=object)))
        for label, arg in cases:
            got = ex.ln_analytic_factor(*([arg] * 6), xpy=cupy)
            assert isinstance(got, cupy.ndarray), (
                "B=%r, %s input: the factor returned %r, not a device array.  The three call "
                "sites that pass xpy add this to a device lnL and would raise."
                % (b_coeff, label, type(got)))
            assert np.allclose(cupy.asnumpy(got), want), (
                "B=%r, %s input: the device answer %r disagrees with the numpy arm %r"
                % (b_coeff, label, cupy.asnumpy(got), want))
    finally:
        ex.B_COEFF = b_was


def test_the_stand_in_builds_its_array_on_a_real_device():
    """The same question as test_the_base_array_comes_from_the_injected_module_not_numpy, asked
    of real cupy instead of _FakeXpy.

    _FakeXpy implements zeros/ones/exp over numpy and tags the result.  It therefore accepts
    things cupy refuses -- an object-dtype array, a numpy array handed to a ufunc -- so it can
    report a device path working that would raise on a GPU.  This runs the driver's own factory
    with xpy=cupy and the shipped factor, on a fully-sampled signature and on one that has to
    fall back to a scalar default, and checks the result is on the device and matches numpy."""
    cupy = _cupy_or_skip()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import analytic_supplement_for_e2e as ex
    make = _driver_ns()["make_zero_likelihood_standin"]
    full = ("right_ascension", "declination", "phi_orb", "inclination", "psi", "distance")
    # distance is not sampled here, so the factory has to supply _DEFAULTS["distance"] = 0.0 --
    # a PYTHON SCALAR reaching the factor's cast alongside device arrays.
    partial = ("right_ascension", "declination", "phi_orb", "inclination", "psi")
    b_was = ex.B_COEFF
    try:
        ex.B_COEFF = 2.0
        for sig in (full, partial):
            host = {nm: np.linspace(0.1, 1.0, 4) for nm in sig}
            dev = {nm: cupy.asarray(v, dtype=float) for nm, v in host.items()}
            for return_lnL in (True, False):
                want = make(sig, return_lnL, ex.ln_analytic_factor, _DEFAULTS, np)(**host)
                got = make(sig, return_lnL, ex.ln_analytic_factor, _DEFAULTS, cupy)(**dev)
                assert isinstance(got, cupy.ndarray), (
                    "signature %r, return_lnL=%r: the stand-in returned %r, not a device array"
                    % (sig, return_lnL, type(got)))
                assert np.allclose(cupy.asnumpy(got), np.asarray(want, dtype=float)), (
                    "signature %r, return_lnL=%r: device %r != numpy %r"
                    % (sig, return_lnL, cupy.asnumpy(got), want))
        # and with no factor at all, the exact zero / one the option promises, on the device
        kw = {nm: cupy.asarray(np.zeros(4), dtype=float) for nm in full}
        assert np.allclose(cupy.asnumpy(make(full, True, None, _DEFAULTS, cupy)(**kw)), 0.0)
        assert np.allclose(cupy.asnumpy(make(full, False, None, _DEFAULTS, cupy)(**kw)), 1.0)
    finally:
        ex.B_COEFF = b_was
