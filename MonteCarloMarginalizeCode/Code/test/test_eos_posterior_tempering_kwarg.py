"""The EOS driver must hand the sampler ``tempering_exp``, not ``adapt_weight_exponent``.

``adapt_weight_exponent`` is an argparse *dest* in the ILE drivers, which translate it
to ``tempering_exp`` before calling a sampler.  No integrator reads the argparse spelling,
so passing it to ``integrate()`` is silently swallowed by ``**kwargs`` and the sampler
falls back to its module default.  util_ConstructIntrinsicPosterior_GenericCoordinates.py
had this bug and it was fixed in f9f4456c0 ("cip: bug with arg name passing"); the EOS
driver, forked from the same code, kept it.

The first test is a call-site spy: it executes the driver's real ``sampler.integrate(...)``
expression against a recording stand-in and asserts on the keyword dict the sampler
receives, so it fails if the kwarg is renamed back, dropped, or set from something other
than ``my_exp``.  It binds its own ``extra_args``, so it cannot see a key smuggled in
through the driver's real ``extra_args``; ``test_extra_args_does_not_collide...`` below
covers that, and neither test can see the call being made unreachable.
"""
import ast
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)                      # .../Code
BIN_DIR = os.path.join(CODE_ROOT, "bin")
EOS_DRIVER = os.path.join(BIN_DIR, "util_ConstructEOSPosterior.py")

BAD_KWARG = "adapt_weight_exponent"
GOOD_KWARG = "tempering_exp"
INTEGRATE_METHODS = ("integrate", "integrate_log")


def _integrate_calls(tree):
    """Every ``<something>.integrate(...)`` / ``.integrate_log(...)`` Call in a module."""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in INTEGRATE_METHODS]


class _SamplerSpy(object):
    """Stands in for the sampler object at the driver's call site."""

    def __init__(self):
        self.args = None
        self.kwargs = None

    def integrate(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        # The driver unpacks four values from the return.
        return 1.0, 0.0, 1.0, {}

    integrate_log = integrate


class _Opts(object):
    n_max = 1000
    n_eff = 100
    force_no_adapt = False
    tripwire_fraction = 2


def test_eos_driver_hands_the_sampler_tempering_exp():
    """Execute the driver's own integrate() expression and inspect the kwargs it sends."""
    with open(EOS_DRIVER) as f:
        source = f.read()
    tree = ast.parse(source, filename=EOS_DRIVER)
    calls = [c for c in _integrate_calls(tree)
             if isinstance(c.func.value, ast.Name) and c.func.value.id == "sampler"]
    assert len(calls) == 1, (
        "expected exactly one sampler.integrate* call site in %s, found %d; "
        "this test pins the call site and must be updated if the driver grows another"
        % (EOS_DRIVER, len(calls)))

    my_exp_sentinel = object()   # identity, so we also prove the VALUE is my_exp
    spy = _SamplerSpy()
    namespace = {
        "sampler": spy,
        "fn_passed": lambda *a: 1.0,
        "low_level_coord_names": ["x0", "x1"],
        "opts": _Opts(),
        "n_step": 10,
        "test_converged": None,
        "my_exp": my_exp_sentinel,
        "extra_args": {"n_adapt": 100, "history_mult": 10,
                       "force_no_adapt": False, "tripwire_fraction": 2},
    }
    expr = ast.Expression(body=calls[0])
    ast.fix_missing_locations(expr)
    try:
        eval(compile(expr, EOS_DRIVER, "eval"), namespace)
    except NameError as e:
        raise AssertionError(
            "the driver's integrate() call uses a name this test does not bind (%s); "
            "add it to `namespace` above so the call site stays under test" % e)

    assert spy.kwargs is not None, "the spy was never called"
    assert BAD_KWARG not in spy.kwargs, (
        "%s passes %s=... to the sampler.  No integrator reads that name -- it is the "
        "argparse dest used by the ILE drivers, which rename it to %s before calling a "
        "sampler.  Passing it here is swallowed by **kwargs and the sampler silently "
        "uses its module default instead of my_exp."
        % (os.path.basename(EOS_DRIVER), BAD_KWARG, GOOD_KWARG))
    assert GOOD_KWARG in spy.kwargs, (
        "%s no longer passes %s to the sampler, so the weight exponent it computes "
        "(my_exp) never reaches the integrator."
        % (os.path.basename(EOS_DRIVER), GOOD_KWARG))
    assert spy.kwargs[GOOD_KWARG] is my_exp_sentinel, (
        "%s is set from something other than my_exp (got %r); the driver computes and "
        "prints my_exp as the weight exponent, so that is the value that must be sent."
        % (GOOD_KWARG, spy.kwargs[GOOD_KWARG]))


def _my_exp_block(tree):
    """The driver's my_exp assignment and the guards that follow it, as a Module."""
    body = tree.body
    start = None
    for i, node in enumerate(body):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "my_exp" for t in node.targets)):
            start = i
            break
    assert start is not None, "no top-level my_exp assignment in %s" % EOS_DRIVER
    end = start + 1
    while end < len(body) and isinstance(body[end], ast.If):
        end += 1
    block = ast.Module(body=body[start:end], type_ignores=[])
    ast.fix_missing_locations(block)
    return block, end - start


@pytest.mark.parametrize("y_orig_max,shift", [
    (50.0, 0.0),      # ordinary run
    (50.0, 100.0),    # --lnL-shift-prevent-overflow above the peak: max(Y) < 0
    (50.0, 50.0),     # shifted exactly to zero: the ratio is +inf
    (-50.0, -50.0),   # all-negative lnL, driver's own auto-shift
])
def test_my_exp_is_positive_whatever_the_shift(y_orig_max, shift):
    """my_exp reaches the sampler now, so a negative value is no longer inert.

    The ratio divides by max(Y), which carries lnL_shift; the historical guard tests the
    unshifted Y_orig.  A negative exponent makes the sampler adapt away from the peak.
    """
    import numpy as np
    with open(EOS_DRIVER) as f:
        tree = ast.parse(f.read(), filename=EOS_DRIVER)
    block, n_stmt = _my_exp_block(tree)
    assert n_stmt >= 2, (
        "the my_exp block in %s has no guard after the assignment; a negative or "
        "non-finite exponent would reach the sampler" % os.path.basename(EOS_DRIVER))

    y_orig = np.array([y_orig_max - 10.0, y_orig_max])
    namespace = {"np": np, "n_step": 2000, "Y": y_orig - shift, "Y_orig": y_orig}
    exec(compile(block, EOS_DRIVER, "exec"), namespace)
    my_exp = namespace["my_exp"]
    assert np.isfinite(my_exp) and my_exp > 0, (
        "my_exp = %r for max(Y_orig)=%s, lnL_shift=%s; the sampler receives this as "
        "tempering_exp" % (my_exp, y_orig_max, shift))
    assert my_exp <= 1, "my_exp = %r exceeds the driver's own cap of 1" % my_exp


def test_extra_args_does_not_collide_with_the_tempering_kwarg():
    """The driver also sends **extra_args, which the spy above binds itself.

    A key there named `adapt_weight_exponent` reaches the sampler as the dead name even
    though the call site is right; a key named `tempering_exp` makes the call raise
    TypeError ("multiple values for keyword argument") at run time.  Neither is visible
    to a spy that supplies its own extra_args, so check the driver's literals.
    """
    with open(EOS_DRIVER) as f:
        tree = ast.parse(f.read(), filename=EOS_DRIVER)

    dicts = []
    for node in ast.walk(tree):
        # extra_args = {...}
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(isinstance(t, ast.Name) and t.id == "extra_args" for t in node.targets):
                dicts.append(node.value)
        # extra_args.update({...})
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "extra_args"):
            dicts.extend([a for a in node.args if isinstance(a, ast.Dict)])

    assert dicts, (
        "found no extra_args dict literal in %s; the driver stopped building it the way "
        "this test reads it" % os.path.basename(EOS_DRIVER))

    keys = {}
    for d in dicts:
        for k in d.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.setdefault(k.value, k.lineno)

    assert BAD_KWARG not in keys, (
        "extra_args carries %s (line %s); it reaches the sampler as the dead name even "
        "though the call site spells %s" % (BAD_KWARG, keys.get(BAD_KWARG), GOOD_KWARG))
    assert GOOD_KWARG not in keys, (
        "extra_args carries %s (line %s) and the call site passes it too; **extra_args "
        "then raises TypeError, multiple values for keyword argument"
        % (GOOD_KWARG, keys.get(GOOD_KWARG)))


def _driver_sources():
    for name in sorted(os.listdir(BIN_DIR)):
        path = os.path.join(BIN_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                head = f.read(200)
                f.seek(0)
                source = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        if not (name.endswith(".py") or (head.startswith("#!") and "python" in head)):
            continue
        yield path, source


def test_no_driver_passes_the_argparse_spelling_to_a_sampler():
    """Fleet-wide guard: the same rename bug in any other bin/ driver."""
    offenders = []
    for path, source in _driver_sources():
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue                      # non-python executable in bin/
        for call in _integrate_calls(tree):
            for kw in call.keywords:
                if kw.arg == BAD_KWARG:
                    offenders.append("%s:%d" % (os.path.relpath(path, CODE_ROOT),
                                                call.lineno))
    assert not offenders, (
        "these integrate() call sites pass %s, which no integrator reads; the samplers "
        "take %s: %s" % (BAD_KWARG, GOOD_KWARG, ", ".join(offenders)))


def _delegates_to_sibling(func_node):
    """True if this method UNCONDITIONALLY forwards its kwargs bundle to the sibling.

    AV.integrate, Portfolio.integrate and Ensemble.integrate_log are thin wrappers, so
    their read lives in the sibling, once.  The delegating call must be a direct child
    statement of the function body: mcsamplerGPU.integrate delegates too, but only
    inside `if kwargs["use_lnL"]`, and then falls through to its own implementation --
    walking the whole body would exempt a method that does read the kwarg itself.
    """
    for stmt in func_node.body:
        if isinstance(stmt, ast.Return):
            call = stmt.value
        elif isinstance(stmt, ast.Assign):
            call = stmt.value
        else:
            continue
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr in INTEGRATE_METHODS
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
                and any(kw.arg is None for kw in call.keywords)):
            return True
    return False


def _reads_from_kwargs(func_node, name):
    """True if this function body reads kwargs["<name>"] or kwargs.get("<name>")."""
    for n in ast.walk(func_node):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "kwargs"
                and isinstance(n.slice, ast.Constant) and n.slice.value == name):
            return True
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "kwargs" and n.args
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == name):
            return True
    return False


@pytest.mark.parametrize("module_name", [
    "RIFT.integrators.mcsampler",
    "RIFT.integrators.mcsamplerGPU",
    "RIFT.integrators.mcsamplerAdaptiveVolume",
    "RIFT.integrators.mcsamplerEnsemble",
    "RIFT.integrators.mcsamplerPortfolio",
    "RIFT.integrators.mcsamplerNFlow",
])
def test_integrators_read_tempering_exp_from_kwargs(module_name):
    """The premise of the rename, checked per entry point rather than per file.

    A module-wide substring scan passes while one of the two entry points has had its
    read replaced by a hardcoded value, because the other still spells the name.
    """
    mod = pytest.importorskip(module_name)
    with open(mod.__file__) as f:
        tree = ast.parse(f.read(), filename=mod.__file__)
    checked = []
    for cls in [n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "MCSampler"]:
        for fn in [n for n in cls.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name in INTEGRATE_METHODS]:
            checked.append(fn.name)
            if _delegates_to_sibling(fn):
                continue
            assert _reads_from_kwargs(fn, GOOD_KWARG), (
                "%s.MCSampler.%s neither reads %s from kwargs nor forwards ** to the "
                "sibling entry point, so the value the EOS driver sends is dropped there"
                % (module_name, fn.name, GOOD_KWARG))
    assert checked, (
        "found no MCSampler.integrate/integrate_log in %s to check; the class or method "
        "names moved and this test stopped testing anything" % module_name)


def test_the_argparse_spelling_is_inert_at_the_sampler():
    """Behavioural proof that the misspelling does nothing: it must not change a result.

    Forces the numpy backend, so it is skipped where cupy is importable (cupy's RNG is
    not seeded by numpy.random.seed, and the run would not be bit-reproducible).
    """
    try:
        import cupy                                    # noqa: F401
        pytest.skip("cupy importable: run is not bit-reproducible under numpy seeding")
    except ImportError:
        pass
    import numpy as np
    mcsamplerGPU = pytest.importorskip("RIFT.integrators.mcsamplerGPU")

    def build():
        s = mcsamplerGPU.MCSampler()
        s.xpy = np
        s.identity_convert = lambda x: x
        for p in ("x0", "x1"):
            s.add_parameter(p, pdf=np.vectorize(lambda x: 1),
                            prior_pdf=np.vectorize(lambda x: 0.1),
                            left_limit=-5.0, right_limit=5.0, adaptive_sampling=True)
        return s

    def integrand(*args):
        x = np.atleast_2d(np.array([*args], dtype=np.float64).T)
        return np.exp(30.0 - 0.5 * ((x - 1.0) ** 2).sum(axis=1) / 0.35 ** 2)

    def go(**extra):
        np.random.seed(11)
        s = build()
        res, _, neff, _ = s.integrate(
            integrand, "x0", "x1", verbose=False, nmax=3000, n=500, neff=1e9,
            save_intg=True, tempering_adapt=True, floor_level=1e-3,
            igrand_threshold_p=1e-3, convergence_tests=None, no_protect_names=True,
            n_adapt=100, history_mult=10, **extra)
        return float(np.log(res)), float(neff)

    default = go()
    misspelled = go(**{BAD_KWARG: 1.0})
    correct = go(**{GOOD_KWARG: 1.0})

    assert misspelled == default, (
        "%s=1.0 changed the result, so some integrator now consumes that name; the "
        "driver-side rename in this PR needs rechecking" % BAD_KWARG)
    assert correct != default, (
        "%s=1.0 did not change the result, so this toy no longer exercises the "
        "adaptation weights and the inertness check above proves nothing" % GOOD_KWARG)
