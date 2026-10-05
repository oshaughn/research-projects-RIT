#!/usr/bin/env python3
"""
Unit tests for the prior densities defined inside
``bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py`` (CIP).

Why this file looks the way it does
-----------------------------------
CIP is a script, not an importable module: importing it parses argv, reads the
input grid and runs several thousand lines of module-level setup.  So the
priors -- which are otherwise ordinary pure functions of one array -- have
never been reachable from a test, and have never had one.

That is how ``--eccentricity-prior log_uniform`` shipped in 0.0.17.12 calling
``np.ln``, which does not exist in numpy.  The option raised AttributeError the
first time the prior was evaluated, and its normalization was independently
wrong: it used ``log(ECC_MAX-ECC_MIN)``, the *uniform* prior's normalization,
where a density uniform in ln(e) needs ``log(ECC_MAX/ECC_MIN)``.

Rather than transcribe the priors here -- which lets the test silently drift
away from the shipped code, the usual failure mode of a copied reference
implementation -- this module extracts the actual ``def`` blocks from the CIP
source with ``ast`` and execs them in a namespace holding numpy and the handful
of module-level constants they close over.  The functions under test are
therefore byte-identical to the ones CIP runs.

Three layers of coverage:

``test_prior_evaluates``
    Every extracted prior must evaluate on a valid array and return finite,
    non-negative, correctly-shaped values.  This is the cheap generic guard: it
    catches the ``np.ln`` class of defect (a name that does not exist) for any
    prior, including ones added later, without anyone having to write a new
    test.

``test_prior_is_normalized``
    The subset whose docstring or comment claims a normalized density is
    integrated numerically over its stated support and must come to 1.  This is
    what catches a wrong normalization constant, which evaluates perfectly
    happily and silently reweights a posterior.  Priors documented in-source as
    unnormalized are listed in UNNORMALIZED below and deliberately excluded.

the ``_eccentricity_setup`` tests
    A correct density is worth nothing if the option does not install it for the
    coordinate the run actually samples.  These execute CIP's own
    ``--eccentricity-prior`` block against its own default prior_map /
    prior_range_map entries, and check every eccentricity coordinate --
    including eccentricity_squared, which is what an eccentric pseudo_pipe run
    samples in iteration 0.  They also run the eccentricity_ln coordinate at
    CIP's *shipped* ``--ecc-min`` default, read out of the argparse call rather
    than assumed here: that coordinate is logarithmic under every prior, so the
    default of 0.0 gives it a [-inf, ...] range and a prior that divides by
    zero, independently of --eccentricity-prior.

``test_eccentricity_prior_option_rejects_unknown_values``
    The option value is forwarded verbatim from pseudo_pipe to CIP and only the
    exact string 'log_uniform' is branched on, so both parsers must reject
    anything else rather than fall through to the uniform prior.
"""

import ast
import functools
import os
import re
import sys
import types

import numpy as np
import pytest

scipy_stats = pytest.importorskip("scipy.stats")
scipy_special = pytest.importorskip("scipy.special")
from scipy import integrate

CIP_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "bin",
    "util_ConstructIntrinsicPosterior_GenericCoordinates.py",
)

# The pipeline driver that forwards --eccentricity-prior to CIP.  Only its argparse
# spec is inspected (by ast, like everything else here); the script is never imported.
PSEUDO_PIPE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "bin",
    "util_RIFT_pseudo_pipe.py",
)

# Values the priors close over.  Chosen to be ordinary production-shaped
# numbers rather than 1.0 everywhere, so a normalization that happens to be
# right only for the unit interval does not pass by accident.
CHI_MAX = 0.9
ECC_MIN = 0.001          # CIP's own auto-correction when --ecc-min is 0
ECC_MAX = 0.4
LAMBDA_MIN = 0.0
LAMBDA_MAX = 4000.0
LAMBDA_SMALL_MAX = 2000.0
MC_MIN = 5.0
MC_MAX = 60.0
A6C_MIN = -80.0
A6C_MAX = -20.0
E0_MIN = 1.0
E0_MAX = 1.2
PPHI0_MIN = 0.0
PPHI0_MAX = 5.4

# CIP sets p_Rbar = lalsimutils.p_R.  Read out of the lalsimutils SOURCE rather
# than imported: importing lalsimutils pulls in LAL, whose default error handler
# calls abort(), which turns any unrelated numerical complaint raised inside
# scipy.integrate.quad below into a hard core dump instead of a test failure.
# These priors are pure numpy, so the test stays free of that whole stack.
LALSIMUTILS_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "RIFT", "lalsimutils.py")


def _p_rbar(default=0.25):
    """lalsimutils.p_R, parsed from source; `default` matches the shipped value."""
    try:
        with open(LALSIMUTILS_SOURCE) as handle:
            for line in handle:
                match = re.match(r"^p_R\s*=\s*([0-9.eE+-]+)\s*(#.*)?$", line)
                if match:
                    return float(match.group(1))
    except OSError:
        pass
    return default


# Functions with 'prior' in the name that are NOT one-dimensional densities of
# a parameter, and so are not subject to either test below.
NOT_A_DENSITY = {
    # a CDF helper: takes a scalar eta_min, not an array of samples
    "unscaled_eta_prior_cdf",
    # operate on a whole parameter vector during the fit, not on one coordinate
    "my_prior_scale",
    "my_log_prior_scale",
}


def _parse_script(path):
    with open(path) as handle:
        return ast.parse(handle.read())


CIP_TREE = _parse_script(CIP_SCRIPT)
PSEUDO_PIPE_TREE = _parse_script(PSEUDO_PIPE_SCRIPT)


def _add_argument_kwargs(tree, option):
    """The keyword arguments of a shipped ``parser.add_argument(option, ...)`` call.

    Lets a test assert against the CLI as actually shipped -- the real default, the
    real `choices` -- instead of a value transcribed into the test, which is the same
    drift problem the prior extraction above avoids.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        try:
            if ast.literal_eval(node.args[0]) != option:
                continue
        except (ValueError, SyntaxError):
            continue
        found = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                found[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                # e.g. type=float, or a default built from an expression; the tests
                # here only read literal defaults and choices
                found[keyword.arg] = None
        return found
    raise AssertionError("no add_argument({!r}) call found".format(option))


# argparse's own default for --ecc-min: what a run gets when the user says nothing.
CIP_ECC_MIN_DEFAULT = _add_argument_kwargs(CIP_TREE, "--ecc-min")["default"]
CIP_ECC_PRIOR_DEFAULT = _add_argument_kwargs(
    CIP_TREE, "--eccentricity-prior")["default"]


def _exec_in(namespace, *nodes):
    """Compile and run the given top-level CIP nodes in `namespace`."""
    module = ast.Module(body=list(nodes), type_ignores=[])
    exec(compile(module, CIP_SCRIPT, "exec"), namespace)


def _make_namespace(ecc_min=ECC_MIN, ecc_max=ECC_MAX, eccentricity_prior="uniform",
                    coords=()):
    """The module-level constants the priors and the option wiring close over.

    `coords` stands in for CIP's low_level_coord_names, the coordinates the Monte
    Carlo actually samples in; the eccentricity block consults it because the ln
    coordinate needs a positive floor whatever the prior is.
    """
    return {
        "low_level_coord_names": list(coords),
        # the eccentricity block exits on an unusable logarithmic range
        "sys": sys,
        "np": np,
        "numpy": np,
        "scipy": types.SimpleNamespace(stats=scipy_stats),
        "chi_max": CHI_MAX,
        "chi_small_max": CHI_MAX,
        "ECC_MIN": ecc_min,
        "ECC_MAX": ecc_max,
        "MEANPERANO_MIN": 0.0,
        "MEANPERANO_MAX": 2 * np.pi,
        "A6C_MIN": A6C_MIN,
        "A6C_MAX": A6C_MAX,
        "E0_MIN": E0_MIN,
        "E0_MAX": E0_MAX,
        "PPHI0_MIN": PPHI0_MIN,
        "PPHI0_MAX": PPHI0_MAX,
        "lambda_min": LAMBDA_MIN,
        "lambda_max": LAMBDA_MAX,
        "lambda_small_max": LAMBDA_SMALL_MAX,
        "mc_min": MC_MIN,
        "mc_max": MC_MAX,
        "p_Rbar": _p_rbar(),
        # lambda_tilde_prior reads opts directly, as does the --eccentricity-prior block
        "opts": types.SimpleNamespace(lambda_max=LAMBDA_MAX,
                                      eccentricity_prior=eccentricity_prior),
    }


def _load_priors(namespace):
    """Exec the prior ``def`` blocks out of the CIP source, verbatim.

    Only top-level FunctionDef nodes are taken, so the surrounding script
    (argparse, I/O, the fitting machinery) never runs.  Selection is on
    'prior' appearing anywhere in the name, NOT a '_prior' suffix: the suffix
    rule silently skips s_component_zprior, s_component_zprior_positive and
    the two *volumetricprior densities, which is most of the spin sector.
    """
    found = {}
    for node in CIP_TREE.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if "prior" not in node.name.lower() or node.name in NOT_A_DENSITY:
            continue
        _exec_in(namespace, node)
        found[node.name] = namespace[node.name]
    return found


PRIORS = _load_priors(_make_namespace())

# Support on which each prior may be evaluated.  Only used to feed the smoke
# test valid inputs; priors with an integrable singularity at an endpoint are
# sampled strictly inside.
SUPPORT = {
    # masses: the mass priors are normalized against the mc window, and M_prior
    # / mc_prior go negative for x < 0, so they must not be fed the default
    # spin-shaped interval
    "M_prior": (MC_MIN, MC_MAX),
    "mc_prior": (MC_MIN, MC_MAX),
    "m1_prior": (1.0, 200.0),
    "m2_prior": (1.0, 200.0),
    "m_prior": (1.0, 1000.0),
    "q_prior": (0.0, 1.0),
    # eta in (0, 1/4]; both endpoints are singular, and the linspace below
    # drops them
    "eta_prior": (0.0, 0.25),
    # delta_mc = sqrt(1-4 eta) in [0,1); eta -> 0 at the upper end
    "delta_mc_prior": (0.0, 1.0),
    "gaussian_mass_prior": (-4.0, 4.0),
    "eccentricity_prior": (ECC_MIN, ECC_MAX),
    "log_eccentricity_prior": (ECC_MIN, ECC_MAX),
    "uniform_eccentricity_ln_prior": (ECC_MIN, ECC_MAX),
    "eccentricity_squared_prior": (ECC_MIN, ECC_MAX),
    # a density in e^2, so it is evaluated on the squared interval
    "log_eccentricity_squared_prior": (ECC_MIN ** 2, ECC_MAX ** 2),
    "meanPerAno_prior": (0.0, 2 * np.pi),
    "a6c_prior": (A6C_MIN, A6C_MAX),
    "initial_energy_prior": (E0_MIN, E0_MAX),
    "initial_angmom_prior": (PPHI0_MIN, PPHI0_MAX),
    "precession_prior": (0.0, 2.0),
    "lambda_prior": (LAMBDA_MIN, LAMBDA_MAX),
    "lambda_small_prior": (LAMBDA_MIN, LAMBDA_SMALL_MAX),
    "lambda_tilde_prior": (0.0, LAMBDA_MAX),
    "delta_lambda_tilde_prior": (-500.0, 500.0),
    "unnormalized_log_prior": (0.1, 10.0),
    "normalized_Rbar_prior": (0.0, 1.0),
    "normalized_Rbar_singular_prior": (1e-6, 1.0),
    "normalized_Rbar_taper_prior": (0.0, 1.0),
    # integrable 1/sqrt singularity at Rbar=0, sampled strictly inside it
    "normalized_Rbar_sqrt_prior": (1e-6, 1.0),
    "normalized_zbar_prior": (-1.0, 1.0),
    "s_component_volumetricprior": (0.0, 1.0),
    "s_component_aligned_volumetricprior": (-1.0, 1.0),
    "s_magnitude_uniform_prior": (0.0, CHI_MAX),
    "s_component_sqrt_prior": (1e-6, CHI_MAX),
    "s_component_zprior": (-CHI_MAX, CHI_MAX),
    "s_component_zprior_positive": (0.0, CHI_MAX),
}
DEFAULT_SUPPORT = (-CHI_MAX, CHI_MAX)

# Documented in-source as not normalized (or normalized only up to a factor the
# caller supplies).  Excluded from the normalization test on purpose, not by
# oversight -- see the comments on each in CIP.
UNNORMALIZED = {
    "unnormalized_uniform_prior",
    "unnormalized_log_prior",
    "xi_uniform_prior",
    "M_prior",
    "m_prior",
    "m1_prior",
    "m2_prior",
    "mc_prior",
    "q_prior",
    "eta_prior",
    "delta_mc_prior",
    "s1z_prior",
    "s2z_prior",
    "lambda_tilde_prior",
    "delta_lambda_tilde_prior",
    "tapered_magnitude_prior",
    "tapered_magnitude_prior_alt",
    # p(a) for a volumetric spin MAGNITUDE prior; carries the 1/3 of the
    # 3-d measure, so it is not a normalized 1-d density on its own.
    "s_component_volumetricprior",
}

# (prior, lower, upper, change of variable, interior singular points) for every
# prior that claims a normalized density.  The 4th entry names the measure the
# density is defined against: 'x' integrates dx directly, 'log' integrates
# d(ln x), 'square' integrates d(x^2).  Getting this wrong is exactly the bug
# being tested for, so each is spelled out rather than inferred.
#
# The 5th entry lists interior points where the integrand is singular; they are
# handed to quad's `points` so QUADPACK subdivides there.  Without it an
# integrand that returns inf at a node aborts the process rather than raising.
NORMALIZED = [
    ("eccentricity_prior", ECC_MIN, ECC_MAX, "x", ()),
    # The regression target: log-uniform in e over [ECC_MIN, ECC_MAX].
    ("log_eccentricity_prior", ECC_MIN, ECC_MAX, "x", ()),
    # Density against d(ln e), so it must integrate to 1 over ln-space.
    ("uniform_eccentricity_ln_prior", ECC_MIN, ECC_MAX, "log", ()),
    # Density against d(e^2); see the INCONSISTENT note in CIP.
    ("eccentricity_squared_prior", ECC_MIN, ECC_MAX, "square", ()),
    # Already written as a function of u=e^2, so it integrates du over the squared
    # interval directly rather than through the 'square' substitution.
    ("log_eccentricity_squared_prior", ECC_MIN ** 2, ECC_MAX ** 2, "x", ()),
    ("meanPerAno_prior", 0.0, 2 * np.pi, "x", ()),
    ("a6c_prior", A6C_MIN, A6C_MAX, "x", ()),
    ("initial_energy_prior", E0_MIN, E0_MAX, "x", ()),
    ("initial_angmom_prior", PPHI0_MIN, PPHI0_MAX, "x", ()),
    ("precession_prior", 0.0, 2.0, "x", ()),
    ("triangle_prior", -CHI_MAX, CHI_MAX, "x", ()),
    ("s_component_uniform_prior", -CHI_MAX, CHI_MAX, "x", ()),
    ("s_magnitude_uniform_prior", 0.0, CHI_MAX, "x", ()),
    # 1/sqrt(|x|) singularity at the origin, integrable
    ("s_component_sqrt_prior", -CHI_MAX, CHI_MAX, "x", (0.0,)),
    ("s_component_zprior", -CHI_MAX, CHI_MAX, "x", (0.0,)),
    ("s_component_zprior_positive", 0.0, CHI_MAX, "x", ()),
    ("s_component_gaussian_prior", -CHI_MAX, CHI_MAX, "x", ()),
    ("s_component_aligned_volumetricprior", -1.0, 1.0, "x", ()),
    ("normalized_Rbar_prior", 0.0, 1.0, "x", ()),
    ("normalized_Rbar_singular_prior", 0.0, 1.0, "x", ()),
    # the [0,1] counterparts of triangle_prior and s_component_sqrt_prior, for the
    # normalized radial coordinate; a [-R,R] density reused here integrates to 1/2
    ("normalized_Rbar_taper_prior", 0.0, 1.0, "x", ()),
    ("normalized_Rbar_sqrt_prior", 0.0, 1.0, "x", ()),
    ("normalized_zbar_prior", -1.0, 1.0, "x", ()),
    ("lambda_prior", LAMBDA_MIN, LAMBDA_MAX, "x", ()),
    ("lambda_small_prior", LAMBDA_MIN, LAMBDA_SMALL_MAX, "x", ()),
]


def test_priors_were_actually_extracted():
    """Guard against the extraction silently finding nothing.

    If CIP is refactored so the priors are no longer top-level '*_prior'
    functions, every parametrized test below would collect zero cases and the
    suite would go green while testing nothing.  Fail loudly instead.
    """
    assert len(PRIORS) > 25, "only found {} priors in CIP: {}".format(
        len(PRIORS), sorted(PRIORS))
    for name in ("eccentricity_prior", "log_eccentricity_prior",
                 "uniform_eccentricity_ln_prior", "eccentricity_squared_prior",
                 "log_eccentricity_squared_prior"):
        assert name in PRIORS, "{} not extracted from CIP".format(name)


@pytest.mark.parametrize("name", sorted(PRIORS))
def test_prior_evaluates(name):
    """Every prior evaluates on its support without raising, and returns
    finite non-negative densities of the input shape.

    This is the check that would have caught np.ln at the point it was written:
    the call raises AttributeError rather than returning a number.
    """
    lo, hi = SUPPORT.get(name, DEFAULT_SUPPORT)
    # strictly interior, so an integrable endpoint singularity is not the thing
    # under test here
    x = np.linspace(lo, hi, 17)[1:-1]

    value = np.asarray(PRIORS[name](x), dtype=float)

    # A constant prior may legitimately return a bare scalar rather than an
    # array (m1_prior, m2_prior, m_prior, s1z_prior, s2z_prior all do), and
    # callers rely on numpy broadcasting it.  Require broadcastability, not an
    # exact shape match.
    try:
        broadcast = np.broadcast_to(value, x.shape)
    except ValueError:
        pytest.fail("{}: returned shape {} does not broadcast to input {}".format(
            name, value.shape, x.shape))

    assert np.all(np.isfinite(broadcast)), "{}: non-finite densities".format(name)
    assert np.all(broadcast >= 0), "{}: negative density".format(name)


@pytest.mark.parametrize("name,lo,hi,measure,singular",
                         NORMALIZED, ids=[row[0] for row in NORMALIZED])
def test_prior_is_normalized(name, lo, hi, measure, singular):
    """Priors that claim a normalized density must integrate to 1.

    Catches a wrong normalization constant, which -- unlike a wrong function
    name -- raises nothing and merely reweights the posterior.  With the
    0.0.17.12 log(ECC_MAX-ECC_MIN) constant this integrates to about -0.13
    rather than 1.
    """
    prior = PRIORS[name]

    if measure == "log":
        # density against d(ln x): substitute u = ln x
        integrand = lambda u: float(prior(np.array([np.exp(u)]))[0])
        lo_t, hi_t = np.log(lo), np.log(hi)
    elif measure == "square":
        # density against d(x^2): substitute u = x^2
        integrand = lambda u: float(prior(np.array([np.sqrt(u)]))[0])
        lo_t, hi_t = lo ** 2, hi ** 2
    else:
        integrand = lambda u: float(prior(np.array([u]))[0])
        lo_t, hi_t = lo, hi

    if singular:
        total, err = integrate.quad(integrand, lo_t, hi_t, limit=200,
                                    points=list(singular))
    else:
        total, err = integrate.quad(integrand, lo_t, hi_t, limit=200)

    assert err < 1e-4, "{}: quadrature did not converge (err={})".format(name, err)
    assert total == pytest.approx(1.0, rel=2e-3), (
        "{} integrates to {:.6f} over [{}, {}] d{}, not 1".format(
            name, total, lo, hi, measure))


def test_log_eccentricity_prior_is_log_uniform():
    """The shape check behind the normalization: e*p(e) is constant.

    A density uniform in ln(e) is p(e) = 1/(e * ln(emax/emin)), so e*p(e) does
    not depend on e.  This pins the 1/e, independently of the constant, and
    distinguishes it from the flat eccentricity_prior.
    """
    prior = PRIORS["log_eccentricity_prior"]
    e = np.geomspace(ECC_MIN, ECC_MAX, 25)

    scaled = e * np.asarray(prior(e), dtype=float)

    assert np.allclose(scaled, scaled[0], rtol=1e-10), (
        "e*p(e) is not constant, so p is not log-uniform: {}".format(scaled))
    assert scaled[0] == pytest.approx(1.0 / np.log(ECC_MAX / ECC_MIN), rel=1e-10)


def test_uniform_and_log_eccentricity_priors_differ():
    """The two eccentricity priors must not be the same function.

    --eccentricity-prior selects between them; if a refactor collapsed one onto
    the other the option would silently stop doing anything.
    """
    e = np.linspace(ECC_MIN, ECC_MAX, 11)

    flat = np.asarray(PRIORS["eccentricity_prior"](e), dtype=float)
    log_uniform = np.asarray(PRIORS["log_eccentricity_prior"](e), dtype=float)

    assert not np.allclose(flat, log_uniform)
    # log-uniform puts more weight at small e, which is the entire point
    assert log_uniform[0] > flat[0]
    assert log_uniform[-1] < flat[-1]


###
### End-to-end coordinate selection: which density --eccentricity-prior actually
### installs for the coordinate a run samples in.
###
### CIP can sample eccentricity in three coordinates, and pseudo_pipe chooses among
### them: --parameter eccentricity, --parameter eccentricity_squared (what
### --use-eccentricity-squared asks for, and what iteration 0 of an eccentric run uses),
### and eccentricity_ln.  The prior is looked up by coordinate name -- prior_map[p] with
### the range prior_range_map[p] -- so an option that rewrites only one entry silently
### leaves the other coordinates on their default density.
###

ECC_COORDS = ("eccentricity", "eccentricity_ln", "eccentricity_squared")


def _eccentricity_dict_entries(name, namespace):
    """Exec only the eccentricity entries of a shipped top-level dict literal.

    CIP's prior_map / prior_range_map also hold mcsampler callables, functools partials
    and mass/spin/matter constants that this test has no business constructing.
    Rebuilding the literal with just the eccentricity keys keeps the entries under test
    identical to the shipped ones, while leaving the rest of the script out and not
    breaking when an unrelated sector gains an entry.
    """
    for node in CIP_TREE.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            continue
        keys, values = [], []
        for key, value in zip(node.value.keys, node.value.values):
            # IGWN production hosts still provide Python 3.6, where parsed
            # string literals are ast.Str rather than ast.Constant.
            key_value = key.s if isinstance(key, ast.Str) else getattr(key, "value", None)
            if key_value in ECC_COORDS:
                keys.append(key)
                values.append(value)
        assert keys, "no eccentricity entries in CIP's {}".format(name)
        trimmed = ast.Assign(targets=node.targets,
                             value=ast.Dict(keys=keys, values=values))
        _exec_in(namespace, ast.fix_missing_locations(
            ast.copy_location(trimmed, node)))
        return namespace[name]
    raise AssertionError("could not find the {} dict in CIP".format(name))


def _selects_eccentricity(test):
    """Does this `if` test steer the eccentricity setup?

    Matched on `opts.eccentricity_prior` or `ECC_MIN` appearing anywhere in the test,
    rather than on one exact comparison: the prior selection and the zero-floor
    correction are separate top-level conditions with different triggers, and a test
    that recognised only the first would silently stop running the second.
    """
    for node in ast.walk(test):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "opts" and node.attr == "eccentricity_prior"):
            return True
        if isinstance(node, ast.Name) and node.id == "ECC_MIN":
            return True
    return False


def _exec_eccentricity_option_block(namespace):
    """Run CIP's top-level eccentricity `if` blocks, verbatim and in source order."""
    found = 0
    for node in CIP_TREE.body:
        if isinstance(node, ast.If) and _selects_eccentricity(node.test):
            _exec_in(namespace, node)
            found += 1
    assert found, "could not find the --eccentricity-prior block in CIP"


def _eccentricity_setup(ecc_min=ECC_MIN, ecc_max=ECC_MAX, eccentricity_prior="uniform",
                        coords=()):
    """Reproduce CIP's eccentricity prior selection: defaults, then the option block."""
    namespace = _make_namespace(ecc_min=ecc_min,
                                ecc_max=ecc_max,
                                eccentricity_prior=eccentricity_prior,
                                coords=coords)
    _load_priors(namespace)
    # ln(ECC_MIN) with --ecc-min 0 is -inf here exactly as it is in CIP (and nan for a
    # negative one); the option block is what repairs or rejects it, and that is the
    # thing under test
    with np.errstate(divide="ignore", invalid="ignore"):
        prior_map = _eccentricity_dict_entries("prior_map", namespace)
        prior_range_map = _eccentricity_dict_entries("prior_range_map", namespace)
        _exec_eccentricity_option_block(namespace)
    return namespace, prior_map, prior_range_map


def _integral_over_range(density, bounds):
    """Integrate a coordinate's density over that coordinate's sampling range."""
    lo, hi = bounds
    integrand = lambda u: float(np.asarray(density(np.array([u])), dtype=float)[0])
    return integrate.quad(integrand, lo, hi, limit=200)


def test_uniform_eccentricity_prior_leaves_the_shipped_defaults():
    """--eccentricity-prior uniform (the default) must not touch any entry."""
    namespace, prior_map, _ = _eccentricity_setup(eccentricity_prior="uniform")

    assert prior_map["eccentricity"] is namespace["eccentricity_prior"]
    assert prior_map["eccentricity_squared"] is namespace["eccentricity_squared_prior"]
    assert prior_map["eccentricity_ln"] is namespace["uniform_eccentricity_ln_prior"]


def test_log_uniform_selects_a_log_uniform_density_for_every_coordinate():
    """--eccentricity-prior log_uniform must reach the coordinate actually sampled.

    Setting only prior_map['eccentricity'] left a --parameter eccentricity_squared run
    on the flat-in-e^2 default: no error, no warning, a different posterior than the
    one requested.
    """
    namespace, prior_map, prior_range_map = _eccentricity_setup(
        eccentricity_prior="log_uniform")

    assert prior_map["eccentricity"] is namespace["log_eccentricity_prior"]
    assert prior_map["eccentricity_squared"] is namespace["log_eccentricity_squared_prior"]
    # uniform in ln(e) already IS this distribution written in that coordinate, so the
    # default entry is correct and deliberately left alone
    assert prior_map["eccentricity_ln"] is namespace["uniform_eccentricity_ln_prior"]

    for coord in ECC_COORDS:
        total, err = _integral_over_range(prior_map[coord], prior_range_map[coord])
        assert err < 1e-4, "{}: quadrature did not converge".format(coord)
        assert total == pytest.approx(1.0, rel=2e-3), (
            "{}: selected density integrates to {:.6f} over its sampling range {}, "
            "not 1".format(coord, total, prior_range_map[coord]))


def test_log_uniform_is_one_distribution_in_e_and_in_e_squared():
    """The e and e^2 coordinates must describe the SAME distribution.

    Equal densities are not the requirement -- equal probability is.  P(e < E) computed
    in the e coordinate must equal P(e^2 < E^2) computed in the e^2 coordinate, which is
    what fails if the e^2 entry keeps a density of a different family.
    """
    _, prior_map, _ = _eccentricity_setup(eccentricity_prior="log_uniform")

    for cut in np.geomspace(1.5 * ECC_MIN, 0.9 * ECC_MAX, 5):
        cdf_e, _ = _integral_over_range(prior_map["eccentricity"], (ECC_MIN, cut))
        cdf_u, _ = _integral_over_range(prior_map["eccentricity_squared"],
                                        (ECC_MIN ** 2, cut ** 2))
        assert cdf_u == pytest.approx(cdf_e, rel=1e-6), (
            "P(e<{:.4f}) is {:.6f} sampling in e but {:.6f} sampling in e^2".format(
                cut, cdf_e, cdf_u))


def test_ecc_min_zero_correction_reaches_every_coordinate_range():
    """--ecc-min 0 with log_uniform: the 0.001 floor must reach every range.

    A log-uniform density is not integrable down to zero in ANY of these coordinates, so
    a range whose lower edge is left at 0 gives a divergent normalization rather than a
    prior.
    """
    namespace, prior_map, prior_range_map = _eccentricity_setup(
        ecc_min=0.0, eccentricity_prior="log_uniform")

    assert namespace["ECC_MIN"] == 0.001

    for coord in ECC_COORDS:
        bounds = prior_range_map[coord]
        assert np.all(np.isfinite(bounds)), (
            "{}: sampling range {} still has a zero-eccentricity edge".format(
                coord, bounds))
        total, err = _integral_over_range(prior_map[coord], bounds)
        assert err < 1e-4, "{}: quadrature did not converge".format(coord)
        assert total == pytest.approx(1.0, rel=2e-3), (
            "{}: selected density integrates to {:.6f} over its sampling range {}, "
            "not 1".format(coord, total, bounds))


def test_ln_coordinate_is_usable_at_the_shipped_cli_defaults():
    """--parameter eccentricity_ln with no --ecc-min and no --eccentricity-prior.

    eccentricity_ln is a logarithmic coordinate under EVERY prior, so the shipped
    --ecc-min default hits it whatever --eccentricity-prior says: the range is
    [log(0), log(ECC_MAX)] and uniform_eccentricity_ln_prior divides by log(ECC_MAX/0).
    The floor therefore has to be keyed on the coordinate as well as on the prior.

    Both defaults are read out of CIP's own argparse calls rather than written here, so
    this exercises the real default invocation and keeps following it if it changes.
    """
    namespace, prior_map, prior_range_map = _eccentricity_setup(
        ecc_min=CIP_ECC_MIN_DEFAULT,
        eccentricity_prior=CIP_ECC_PRIOR_DEFAULT,
        coords=("mc", "eta", "eccentricity_ln"))

    assert namespace["ECC_MIN"] > 0, (
        "ecc-min is still {} for a run sampling ln(e)".format(namespace["ECC_MIN"]))

    bounds = prior_range_map["eccentricity_ln"]
    assert np.all(np.isfinite(bounds)), (
        "eccentricity_ln sampling range {} still has a log(0) edge".format(bounds))

    # evaluating at all is the point: with ECC_MIN left at 0.0 this raises
    # ZeroDivisionError inside the prior rather than returning a density
    density = prior_map["eccentricity_ln"]
    values = np.asarray(density(np.linspace(bounds[0], bounds[1], 9)), dtype=float)
    assert np.all(np.isfinite(values)) and np.all(values > 0)

    total, err = _integral_over_range(density, bounds)
    assert err < 1e-4, "eccentricity_ln: quadrature did not converge"
    assert total == pytest.approx(1.0, rel=2e-3), (
        "eccentricity_ln: density integrates to {:.6f} over its sampling range {}, "
        "not 1".format(total, bounds))


def test_ecc_min_zero_is_left_alone_without_a_log_prior_or_log_coordinate():
    """The floor is a repair, not a policy: a linear-in-e run keeps the ecc-min given.

    --parameter eccentricity under the uniform prior is perfectly well defined down to
    e=0, so raising its lower edge would move a boundary the user set.
    """
    namespace, _, prior_range_map = _eccentricity_setup(
        ecc_min=0.0, eccentricity_prior="uniform", coords=("mc", "eccentricity"))

    assert namespace["ECC_MIN"] == 0.0
    assert prior_range_map["eccentricity"][0] == 0.0


# Every way a logarithmic eccentricity range can be unusable other than the zero floor,
# which is repaired rather than rejected.  All of these reach np.log of a non-positive
# number or a zero/negative log(ECC_MAX/ECC_MIN), i.e. nan or inf bounds and densities.
INVALID_LOG_RANGES = [
    (-0.1, ECC_MAX, "negative ecc-min"),
    (0.0, -0.1, "zero ecc-min floored, negative ecc-max"),
    (0.1, 0.0, "zero ecc-max"),
    (0.3, 0.2, "ecc-max below ecc-min"),
    (0.2, 0.2, "empty interval"),
]

# The two independent ways a run becomes logarithmic in e; both must validate.
LOG_TRIGGERS = [
    ({"eccentricity_prior": "log_uniform", "coords": ("mc", "eccentricity")},
     "log_uniform prior"),
    ({"eccentricity_prior": "uniform", "coords": ("mc", "eccentricity_ln")},
     "ln coordinate"),
]


@pytest.mark.parametrize("trigger,trigger_id", LOG_TRIGGERS,
                         ids=[row[1] for row in LOG_TRIGGERS])
@pytest.mark.parametrize("ecc_min,ecc_max,case", INVALID_LOG_RANGES,
                         ids=[row[2] for row in INVALID_LOG_RANGES])
def test_logarithmic_eccentricity_rejects_an_unusable_range(ecc_min, ecc_max, case,
                                                            trigger, trigger_id):
    """An invalid logarithmic range must fail the run, not produce nan priors.

    Only an exactly-zero ecc-min was ever checked, so e.g. --ecc-min -0.1 walked past the
    floor correction into np.log of a negative number: the sampling bounds and the prior
    densities come out nan, nothing raises, and the run reports a prior it does not have.
    """
    with pytest.raises(SystemExit) as excinfo:
        _eccentricity_setup(ecc_min=ecc_min, ecc_max=ecc_max, **trigger)

    assert excinfo.value.code not in (0, None), (
        "{} with {}: exited {}, which reads as success".format(
            case, trigger_id, excinfo.value.code))


@pytest.mark.parametrize("trigger,trigger_id", LOG_TRIGGERS,
                         ids=[row[1] for row in LOG_TRIGGERS])
def test_valid_logarithmic_eccentricity_range_is_accepted_untouched(trigger, trigger_id):
    """The rejection above must not catch an ordinary 0 < ecc-min < ecc-max run.

    A validity check that also refuses good input would take out every eccentric run.
    """
    namespace, prior_map, prior_range_map = _eccentricity_setup(
        ecc_min=0.01, ecc_max=ECC_MAX, **trigger)

    assert namespace["ECC_MIN"] == 0.01, "a valid ecc-min was moved"
    for coord in ECC_COORDS:
        bounds = prior_range_map[coord]
        assert np.all(np.isfinite(bounds)), "{}: non-finite range {}".format(coord, bounds)
        assert bounds[0] < bounds[1], "{}: inverted range {}".format(coord, bounds)
        values = np.asarray(prior_map[coord](np.linspace(bounds[0], bounds[1], 9)[1:-1]),
                            dtype=float)
        assert np.all(np.isfinite(values)) and np.all(values > 0), (
            "{}: non-finite or non-positive densities".format(coord))


@pytest.mark.parametrize("tree,script", [(CIP_TREE, "CIP"),
                                         (PSEUDO_PIPE_TREE, "pseudo_pipe")],
                         ids=["CIP", "pseudo_pipe"])
def test_eccentricity_prior_option_rejects_unknown_values(tree, script):
    """Both parsers must constrain --eccentricity-prior to the values CIP implements.

    pseudo_pipe forwards the string verbatim and CIP branches on exactly 'log_uniform',
    so an unconstrained option turns a typo -- or an unimplemented value -- into a run
    that silently uses the uniform prior and reports the requested one.
    """
    kwargs = _add_argument_kwargs(tree, "--eccentricity-prior")
    choices = kwargs.get("choices")

    assert choices is not None, (
        "{}: --eccentricity-prior accepts any string".format(script))
    assert sorted(choices) == sorted(["uniform", "log_uniform"]), (
        "{}: --eccentricity-prior choices are {}".format(script, choices))
    assert kwargs.get("default") in choices, (
        "{}: default {!r} is not one of the accepted values".format(
            script, kwargs.get("default")))


# ---------------------------------------------------------------------------
# Behaviour ON the support boundary.
#
# test_prior_evaluates above samples np.linspace(lo, hi, 17)[1:-1] -- strictly
# interior, deliberately, so that an integrable endpoint singularity is not what
# is under test.  That also makes its "negative density" assertion blind to a
# density that is only wrong AT the endpoint, which is the shape of the defect
# these tests pin: s_component_zprior was written as -log(|x|/R + 1e-7)/(2R),
# and the OFFSET pushes the log argument above 1 -- so the density negative --
# for |x| > R*(1-1e-7).  One CIP sample in ~450k landed in that shell, its
# importance weight went negative, and the whole posterior export aborted in
# RIFT.misc.cip_pipeline with "weights must be finite and nonnegative" after the
# run had already converged and written integral_result.dat.
# ---------------------------------------------------------------------------

# Fractions of the support width to step in from each endpoint.  0.0 is the
# endpoint itself; the small ones straddle the 1e-7 shell the offset corrupted.
ENDPOINT_OFFSETS = (0.0, 1e-12, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5)


@pytest.mark.parametrize("name", sorted(PRIORS))
def test_prior_nonnegative_at_support_endpoints(name):
    """No prior may return a negative (or NaN) density at or beside an endpoint.

    +inf is allowed: several of these densities have an integrable endpoint
    singularity, which is why test_prior_evaluates stays inside.  A negative
    value is different in kind -- it is not a density at all, and downstream it
    becomes a negative importance weight.
    """
    lo, hi = SUPPORT.get(name, DEFAULT_SUPPORT)
    span = hi - lo
    x = np.array(sorted(
        {lo + fraction * span for fraction in ENDPOINT_OFFSETS}
        | {hi - fraction * span for fraction in ENDPOINT_OFFSETS}))

    value = np.broadcast_to(np.asarray(PRIORS[name](x), dtype=float), x.shape)

    # ~(v >= 0) catches NaN as well: NaN >= 0 is False.
    bad = ~(value >= 0)
    assert not np.any(bad), "{}: density {} at x = {}".format(
        name, value[bad], x[bad])


@pytest.mark.parametrize("name,scale",
                         [("s_component_zprior", 1.0),
                          ("s_component_zprior_positive", 2.0)])
def test_s_component_zprior_boundary_shell(name, scale):
    """The zprior is zero at |x| = R and non-negative throughout the shell below it.

    The second assertion is what makes the first one worth having: clamping the
    density flat to zero everywhere would satisfy non-negativity while destroying
    the prior, so the interior is pinned to the analytic -log(|x|/R)/(2R).
    """
    prior = PRIORS[name]
    R = CHI_MAX

    # the endpoint, then a log-spaced approach to it from inside, crossing 1e-7
    shell = R * (1.0 - np.concatenate([[0.0], np.logspace(-16, -5, 45)]))
    value = np.asarray(prior(shell), dtype=float)
    bad = ~(value >= 0)
    assert not np.any(bad), "{}: density {} at |x|/R = {}".format(
        name, value[bad], shell[bad] / R)

    assert float(np.asarray(prior(np.array([R])))[0]) == 0.0, (
        "{}: density at the support boundary is not zero".format(name))

    interior = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]) * R
    expected = scale * -np.log(interior / R) / (2 * R)
    assert np.allclose(np.asarray(prior(interior), dtype=float), expected,
                       rtol=1e-6), (
        "{} no longer has the zprior shape on its interior".format(name))


# ---------------------------------------------------------------------------
# The whole path, from one boundary sample to the export validator.
# ---------------------------------------------------------------------------

CIP_PIPELINE_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "RIFT", "misc", "cip_pipeline.py")


def _load_export_validator():
    """``unique_draw_bound`` and its helper, by ast, out of RIFT.misc.cip_pipeline.

    Imported the ordinary way this would pull RIFT's package __init__ -- lal,
    lalsimulation, h5py, igwn_ligolw, precession -- into a suite whose whole
    premise is that it needs numpy and scipy only.  The function itself is pure
    numpy; take just it, the same way the priors above are taken from CIP.
    """
    tree = _parse_script(CIP_PIPELINE_SOURCE)
    wanted = ("_validated_scaled_weights", "unique_draw_bound")
    namespace = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         CIP_PIPELINE_SOURCE, "exec"), namespace)
    for name in wanted:
        assert name in namespace, (
            "{} is no longer a top-level function in cip_pipeline.py; this test "
            "is pointing at nothing".format(name))
    return namespace["unique_draw_bound"]


def _aligned_zprior_reweight_node():
    """CIP's own ``--aligned-prior alignedspin-zprior`` reweight block.

    Identified by its guard rather than by line number: the only top-level `if`
    whose condition names both the option value and the chiz_plus coordinate.
    """
    found = [node for node in CIP_TREE.body
             if isinstance(node, ast.If)
             and "alignedspin-zprior" in ast.dump(node.test)
             and "chiz_plus" in ast.dump(node.test)]
    assert len(found) == 1, (
        "expected exactly one top-level alignedspin-zprior/chiz_plus reweight "
        "block in CIP, found {}".format(len(found)))
    return found[0]


def _aligned_prior_map_node():
    """CIP's own ``if opts.aligned_prior == 'alignedspin-zprior':`` prior_map block.

    Exec'ing this rather than transcribing it is what stops the test agreeing
    with itself: the sampling density the block divides out is whatever CIP
    installs, including the --spin-prior-chizplusminus-alternate-sampling branch.
    """
    found = [node for node in CIP_TREE.body
             if isinstance(node, ast.If)
             and "alignedspin-zprior" in ast.dump(node.test)
             and "chiz_plus" not in ast.dump(node.test)]
    assert len(found) == 1, (
        "expected exactly one top-level aligned_prior prior_map block in CIP, "
        "found {}".format(len(found)))
    return found[0]


def _run_aligned_zprior_reweight(chiz_plus, chiz_minus, weights,
                                 alternate_sampling="alignedspin_zprior"):
    """Run CIP's reweight block verbatim on a hand-built sample set."""
    import functools

    namespace = _make_namespace()
    namespace.update({
        "functools": functools,
        "prior_map": {},
        "prior_range_map": {},
        "low_level_coord_names": ["mc", "delta_mc", "chiz_plus", "chiz_minus"],
        "internal_dtype": np.float64,
        "opts": types.SimpleNamespace(
            aligned_prior="alignedspin-zprior",
            spin_prior_chizplusminus_alternate_sampling=alternate_sampling),
    })
    _exec_in(namespace, *[node for node in CIP_TREE.body
                          if isinstance(node, ast.FunctionDef)
                          and ("prior" in node.name.lower()
                               or node.name == "divisible_sampling_density")
                          and node.name not in NOT_A_DENSITY])
    _exec_in(namespace, _aligned_prior_map_node())

    namespace["samples"] = {"chiz_plus": np.asarray(chiz_plus, dtype=float),
                            "chiz_minus": np.asarray(chiz_minus, dtype=float)}
    namespace["weights"] = np.asarray(weights, dtype=float).copy()
    _exec_in(namespace, _aligned_zprior_reweight_node())
    return namespace["weights"]


# (label, s1z, s2z) in units of chi_max.  The first three rows are the defect;
# the rest are ordinary interior samples that must survive it.
BOUNDARY_ROWS = [
    ("s1z exactly on the boundary", 1.0, 0.25),
    ("s2z inside the 1e-7 shell", 0.4, 1.0 - 5e-8),
    ("both on the boundary: sampling density is zero too", 1.0, 1.0),
    ("interior", 0.3, -0.6),
    ("interior", -0.8, 0.1),
    ("interior", 0.55, 0.45),
]


@pytest.mark.parametrize("alternate_sampling",
                         ["alignedspin_zprior", "gaussian"])
def test_aligned_zprior_reweight_survives_boundary_samples(alternate_sampling):
    """A sample on the spin boundary must not produce a negative export weight.

    This drives CIP's own reweight block and then CIP's own export validator, so
    it fails with the production ValueError -- not a proxy for it -- when the
    prior goes negative.  Deterministic by construction: reaching the offending
    shell in a real run takes a sample within 1e-7 of chi_max, which happens
    roughly once per few hundred thousand draws.
    """
    unique_draw_bound = _load_export_validator()
    s1z = CHI_MAX * np.array([row[1] for row in BOUNDARY_ROWS])
    s2z = CHI_MAX * np.array([row[2] for row in BOUNDARY_ROWS])

    weights = _run_aligned_zprior_reweight(
        0.5 * (s1z + s2z), 0.5 * (s1z - s2z), np.ones(len(BOUNDARY_ROWS)),
        alternate_sampling=alternate_sampling)

    # CIP's own validator first, so the failure a regression produces here is the
    # production one, not a proxy for it.
    assert unique_draw_bound(weights) >= 1

    bad = ~(weights >= 0)
    assert not np.any(bad), "negative export weight from {}".format(
        [BOUNDARY_ROWS[i][0] for i in np.flatnonzero(bad)])
    assert np.all(np.isfinite(weights)), "non-finite export weight from {}".format(
        [BOUNDARY_ROWS[i][0] for i in np.flatnonzero(~np.isfinite(weights))])

    # A sample ON the boundary is zeroed: the density there is zero, so the
    # sample carries no posterior mass.
    assert weights[0] == 0.0, BOUNDARY_ROWS[0][0]
    assert weights[2] == 0.0, BOUNDARY_ROWS[2][0]
    # A sample just INSIDE the 1e-7 shell is not on the boundary and must keep a
    # real weight -- tiny, because the density is nearly zero there, but positive.
    # This is the row whose weight used to come out tiny and NEGATIVE.
    assert 0 < weights[1] < 1e-5, "{}: weight {}".format(
        BOUNDARY_ROWS[1][0], weights[1])
    # ... and the interior rows are untouched, so the block still does its job
    assert np.all(weights[3:] > 1e-3), (
        "interior samples were zeroed too: {}".format(weights[3:]))


# ---------------------------------------------------------------------------
# OUTSIDE the support.
#
# The tests above only reach the boundary, which pins the inner clamp and leaves
# the outer one (np.maximum(val, 0)) doing nothing a test can see.  It is not
# decoration: CIP installs several priors over a sampling range WIDER than the R
# they were built with.  prior_range_map['s1z_bar'] is [-1,1] while
# prior_map['s1z_bar'] is s_component_zprior with R=chi_max, so a run with
# --chi-max 0.8 draws |x| > R over a fifth of that coordinate's range -- six
# orders of magnitude more often than the 1e-7 boundary shell.  Without the outer
# clamp the density is negative across all of it.
# ---------------------------------------------------------------------------

# How far outside R these densities are asked for, as a fraction of R.  1.25 is
# 1/0.8: the ratio a --chi-max 0.8 run actually produces for a *_bar coordinate.
OUTSIDE_SUPPORT_FACTORS = (1.0 + 1e-9, 1.0 + 1e-6, 1.01, 1.25, 2.0)


@pytest.mark.parametrize("name", ["s_component_zprior", "s_component_zprior_positive"])
def test_s_component_zprior_nonnegative_outside_support(name):
    """Zero, not negative, beyond |x| = R.

    A density evaluated outside its own support should be zero.  The old
    -log(|x|/R + 1e-7) form went negative there and stayed negative, without
    limit.
    """
    prior = PRIORS[name]
    R = CHI_MAX
    x = R * np.array(OUTSIDE_SUPPORT_FACTORS)
    x = np.concatenate([x, -x])

    value = np.asarray(prior(x), dtype=float)

    bad = ~(value >= 0)
    assert not np.any(bad), "{}: density {} at |x|/R = {}".format(
        name, value[bad], np.abs(x[bad]) / R)
    assert np.all(value == 0), (
        "{}: density outside [-R,R] is {}, expected 0".format(name, value))


def test_zprior_sampling_range_can_exceed_its_R():
    """The condition that makes the test above load-bearing, read out of CIP.

    If CIP ever stops installing s_component_zprior over a range wider than its
    R, the outside-support test becomes hypothetical and should be retired --
    rather than sitting there looking like coverage.  Fail loudly at that point
    instead.  Checked on s1z_bar, whose prior_range_map entry is fixed at +-1
    while --aligned-prior alignedspin-zprior gives it a prior with R=chi_max.
    """
    # Read the one entry out of CIP's dict LITERAL.  Exec'ing the whole assignment
    # would drag in every name its other entries close over (lambda_plus_max, ...),
    # which is a lot of setup for one pair of numbers.
    entry = None
    for node in CIP_TREE.body:
        if not (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "prior_range_map"
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            try:
                if ast.literal_eval(key) == "s1z_bar":
                    entry = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
        break
    assert entry is not None, (
        "prior_range_map no longer carries a literal s1z_bar range; re-derive which "
        "coordinates are sampled outside their prior's R before trusting the test above")

    lo, hi = entry
    assert max(abs(lo), abs(hi)) > CHI_MAX, (
        "s1z_bar is now sampled inside chi_max ({} vs {}); the outside-support "
        "clamp may no longer be reachable from a real run".format((lo, hi), CHI_MAX))

    # ... and the prior installed on it really is the zprior with R=chi_max
    aligned = ast.dump(_aligned_prior_map_node())
    assert "s1z_bar" in aligned and "s_component_zprior" in aligned, (
        "CIP's alignedspin-zprior block no longer installs s_component_zprior on "
        "s1z_bar; this test's premise has moved")


# ---------------------------------------------------------------------------
# The other copies of the same function.
#
# This module extracts priors from CIP only, so a fix applied to CIP and missed
# in one of the three sibling copies would pass everything above.  Compare the
# source of each shared definition instead of re-testing each one: the defect
# class here is a copy left behind, not a copy that drifts subtly.
# ---------------------------------------------------------------------------

SHARED_PRIOR_SOURCES = {
    "rift_priors": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "RIFT", "likelihood", "rift_priors.py"),
    "GaussianResampling": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "bin",
                                       "util_ConstructIntrinsicPosterior_GaussianResampling.py"),
    "IntermediateEOSIntegralFromFit": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "bin",
        "util_IntermediateEOSIntegralFromFit.py"),
}

# Present in CIP and in every copy listed above.
SHARED_PRIOR_NAMES = ["s_component_zprior"]
# Present in CIP and in rift_priors only.
SHARED_PRIOR_NAMES_LIBRARY_ONLY = ["s_component_zprior_positive"]
# Present in CIP, rift_priors and GaussianResampling, but not in the EOS driver.
SHARED_PRIOR_NAMES_NO_EOS = ["triangle_prior"]


def _function_source(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node, annotate_fields=True)
    return None


@pytest.mark.parametrize("where", sorted(SHARED_PRIOR_SOURCES))
def test_duplicated_priors_match_cip(where):
    """Every copy of a duplicated prior must be the same function as CIP's.

    Compared as a normalized AST, so comments and whitespace do not matter and a
    changed expression does.  If a copy legitimately has to differ, this test is
    the place to record why -- silence here means the copies drifted.
    """
    tree = _parse_script(SHARED_PRIOR_SOURCES[where])
    names = list(SHARED_PRIOR_NAMES)
    if where == "rift_priors":
        names += SHARED_PRIOR_NAMES_LIBRARY_ONLY
    if where != "IntermediateEOSIntegralFromFit":
        names += SHARED_PRIOR_NAMES_NO_EOS

    for name in names:
        theirs = _function_source(tree, name)
        assert theirs is not None, "{} no longer defines {}".format(where, name)
        mine = _function_source(CIP_TREE, name)
        assert mine is not None, "CIP no longer defines {}".format(name)
        assert theirs == mine, (
            "{}.{} has drifted from CIP's copy of the same function".format(where, name))


# ---------------------------------------------------------------------------
# Device arrays.
#
# prior_map entries are handed to the sampler as prior_pdf, and mcsamplerGPU
# calls them on cupy arrays without copying to the host first (unlike
# mcsamplerAdaptiveVolume.prior_prod, which converts).  cupy refuses
# np.asarray(), so coercing the ARGUMENT rather than casting the RESULT turns
# every GPU-backend aligned-spin run into a TypeError.  CI has no cupy, so the
# contract is checked against a stand-in.
# ---------------------------------------------------------------------------

class _DeviceLike(object):
    """The parts of cupy's ndarray contract this prior has to respect.

    Dispatches ufuncs and stays itself, supports scalar arithmetic from both
    sides, has .astype -- and REFUSES implicit conversion to numpy, which is the
    behaviour that catches np.asarray(x).
    """
    __array_priority__ = 100

    def __init__(self, a):
        self._a = np.asarray(a)

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        unwrapped = [i._a if isinstance(i, _DeviceLike) else i for i in inputs]
        return _DeviceLike(getattr(ufunc, method)(*unwrapped, **kwargs))

    def astype(self, dtype):
        return _DeviceLike(self._a.astype(dtype))

    def _binary(self, other, op, reflected=False):
        value = other._a if isinstance(other, _DeviceLike) else other
        return _DeviceLike(op(value, self._a) if reflected else op(self._a, value))

    def __truediv__(self, o):
        import operator
        return self._binary(o, operator.truediv)

    def __rtruediv__(self, o):
        import operator
        return self._binary(o, operator.truediv, True)

    def __mul__(self, o):
        import operator
        return self._binary(o, operator.mul)

    def __rmul__(self, o):
        import operator
        return self._binary(o, operator.mul, True)

    def __add__(self, o):
        import operator
        return self._binary(o, operator.add)

    def __radd__(self, o):
        import operator
        return self._binary(o, operator.add, True)


def test_device_like_standin_refuses_numpy_conversion():
    """The stand-in must actually be able to fail the test below.

    A permissive fake passes the defect it exists to catch.
    """
    with pytest.raises(TypeError):
        np.asarray(_DeviceLike([0.1, 0.5]), dtype=float)
    # ... while still supporting everything the prior legitimately does
    probe = _DeviceLike([0.1, 0.5])
    assert isinstance(np.maximum(np.abs(probe) / 2.0, 1e-7).astype(float), _DeviceLike)
    assert isinstance(-1.0 * np.log(probe), _DeviceLike)


@pytest.mark.parametrize("name", ["s_component_zprior", "s_component_zprior_positive"])
def test_zprior_stays_on_the_device(name):
    """Evaluating the prior on a device array must not force a host conversion."""
    prior = PRIORS[name]
    device = _DeviceLike([0.1, 0.5, 0.9 * CHI_MAX])

    value = prior(device)

    assert isinstance(value, _DeviceLike), (
        "{} returned {}, so it left the device".format(name, type(value).__name__))
    host = np.asarray(prior(np.array([0.1, 0.5, 0.9 * CHI_MAX])), dtype=float)
    assert np.allclose(value._a, host, rtol=1e-12), (
        "{}: device and host results disagree".format(name))


# ---------------------------------------------------------------------------
# The other three sampling-density divisions.
#
# CIP divides out a sampling density in four places.  All four can be handed a
# zero -- under --transverse-prior alignedspin-zprior, prior_map['s1x'..'s2y']
# ARE s_component_zprior with R=chi_max, and prior_range_map['s1x'] is
# [-chi_max, chi_max], so the sampling bound IS the support bound.  Guarding only
# the block that happened to produce the reported traceback would leave the same
# abort reachable from the other three, and with the density now clamped to a
# hard zero the symptom there is inf rather than a negative.
# ---------------------------------------------------------------------------

def _pseudo_uniform_magnitude_node():
    """CIP's own `--pseudo-uniform-magnitude-prior` reweight chain."""
    # Two top-level chains mention the option: the prior_map one (line ~1253) and
    # the reweight one.  Only the reweight chain consults `samples`.
    found = [node for node in CIP_TREE.body
             if isinstance(node, ast.If)
             and "pseudo_uniform_magnitude_prior" in ast.dump(node.test)
             and "samples" in ast.dump(node.test)]
    assert len(found) == 1, (
        "expected exactly one top-level pseudo_uniform_magnitude_prior REWEIGHT "
        "chain in CIP, found {}".format(len(found)))
    return found[0]


def _transverse_prior_map_node():
    """CIP's own `--transverse-prior` prior_map chain."""
    found = [node for node in CIP_TREE.body
             if isinstance(node, ast.If) and "transverse_prior" in ast.dump(node.test)]
    assert len(found) == 1, (
        "expected exactly one top-level transverse_prior chain in CIP, found "
        "{}".format(len(found)))
    return found[0]


def _run_pseudo_uniform_magnitude_reweight(spins, weights):
    """Run CIP's pseudo-uniform-magnitude reweight verbatim on hand-built samples.

    prior_map is built by exec'ing CIP's own --transverse-prior alignedspin-zprior
    and --aligned-prior alignedspin-zprior blocks, so the density divided out is
    the one CIP installs, not one this test chose.
    """
    import functools

    namespace = _make_namespace()
    namespace.update({
        "functools": functools,
        "prior_map": {},
        "prior_range_map": {},
        "low_level_coord_names": ["s1x", "s1y", "s1z", "s2x", "s2y", "s2z"],
        "internal_dtype": np.float64,
        "opts": types.SimpleNamespace(
            aligned_prior="alignedspin-zprior",
            transverse_prior="alignedspin-zprior",
            pseudo_uniform_magnitude_prior=True,
            pseudo_uniform_magnitude_prior_alternate_sampling=False,
            spin_prior_chizplusminus_alternate_sampling="alignedspin_zprior"),
    })
    _exec_in(namespace, *[node for node in CIP_TREE.body
                          if isinstance(node, ast.FunctionDef)
                          and ("prior" in node.name.lower()
                               or node.name == "divisible_sampling_density")
                          and node.name not in NOT_A_DENSITY])
    _exec_in(namespace, _aligned_prior_map_node())
    _exec_in(namespace, _transverse_prior_map_node())
    assert namespace["prior_map"]["s1x"] is namespace["s_component_zprior"], (
        "CIP no longer installs s_component_zprior on s1x under --transverse-prior "
        "alignedspin-zprior; this test's premise has moved")

    namespace["samples"] = {k: np.asarray(v, dtype=float) for k, v in spins.items()}
    namespace["weights"] = np.asarray(weights, dtype=float).copy()
    _exec_in(namespace, _pseudo_uniform_magnitude_node())
    return namespace["weights"]


def test_pseudo_uniform_magnitude_survives_a_transverse_boundary_sample():
    """A component exactly on the spin boundary must not make an inf weight.

    Under --transverse-prior alignedspin-zprior the transverse sampling density is
    the zprior, which is zero at |s1x| = chi_max, so this block divides by zero
    unless the same guard applies here as in the aligned-spin block.

    The boundary rows put the ENTIRE spin on one transverse component, so that
    chi1 (or chi2) equals chi_max exactly and the block's own `chi1 > chi_max` cut
    -- which is strict -- does NOT fire.  With any other placement that cut zeroes
    the row first and the guard is untestable: both of the first two attempts at
    this test passed with the guard removed for exactly that reason.
    """
    unique_draw_bound = _load_export_validator()
    R = CHI_MAX          # _make_namespace sets chi_small_max = chi_max = CHI_MAX
    spins = {
        # row 0: all of spin 1 in s1x, exactly at chi_max -> p(s1x) = 0, chi1 == chi_max
        # row 1: the same for body 2
        # rows 2-4: ordinary interior samples that must survive
        "s1x": [R,   0.2 * R, -0.3 * R, 0.1 * R,  0.25 * R],
        "s1y": [0.0, 0.1 * R,  0.2 * R, -0.2 * R, 0.1 * R],
        "s1z": [0.0, 0.3 * R,  0.1 * R, 0.4 * R, -0.2 * R],
        "s2x": [0.1 * R, R,   0.1 * R,  0.2 * R, -0.1 * R],
        "s2y": [0.1 * R, 0.0, -0.1 * R, 0.1 * R,  0.2 * R],
        "s2z": [0.2 * R, 0.0,  0.3 * R, -0.1 * R, 0.1 * R],
    }
    n = len(spins["s1x"])

    # the premise: the boundary rows must survive the block's own range cuts, or
    # this test cannot see the guard at all
    chi1 = np.sqrt(np.sum([np.asarray(spins[k], dtype=float) ** 2
                           for k in ("s1x", "s1y", "s1z")], axis=0))
    chi2 = np.sqrt(np.sum([np.asarray(spins[k], dtype=float) ** 2
                           for k in ("s2x", "s2y", "s2z")], axis=0))
    assert chi1[0] == pytest.approx(R) and not chi1[0] > R
    assert chi2[1] == pytest.approx(R) and not chi2[1] > R

    weights = _run_pseudo_uniform_magnitude_reweight(spins, np.ones(n))

    assert unique_draw_bound(weights) >= 1
    assert np.all(np.isfinite(weights)), "non-finite export weight: {}".format(weights)
    assert np.all(weights >= 0), "negative export weight: {}".format(weights)
    assert weights[0] == 0.0, "the s1 boundary sample was not zeroed: {}".format(weights[0])
    assert weights[1] == 0.0, "the s2 boundary sample was not zeroed: {}".format(weights[1])
    assert np.all(weights[2:] > 0), (
        "interior samples were zeroed too: {}".format(weights[2:]))


def test_divisible_sampling_density_leaves_nan_loud():
    """NaN must reach the export validator, not be silently zeroed.

    A zero sampling density is a sample with no posterior mass -- a normal thing
    to drop.  A NaN one is a bug somewhere upstream, and turning it into a
    dropped sample would hide it.
    """
    namespace = {"np": np}
    _exec_in(namespace, *[node for node in CIP_TREE.body
                          if isinstance(node, ast.FunctionDef)
                          and node.name == "divisible_sampling_density"])
    fn = namespace["divisible_sampling_density"]

    mask = fn(np.array([1.0, 0.0, np.nan, -1.0]), 4)

    assert list(mask) == [True, False, True, True], (
        "expected only the exact zero to be masked, got {}".format(mask))
    # scalar prior_weight must still broadcast to a per-sample mask
    assert fn(1.0, 3).shape == (3,)
    assert fn(0.0, 3).shape == (3,) and not np.any(fn(0.0, 3))


def test_third_pseudo_uniform_branch_is_unreachable_as_written():
    """The chain's 2nd and 3rd branches have IDENTICAL conditions.

    So the third -- the only one of the four that divides by a chiz_plus/chiz_minus
    sampling density -- can never run, which is why no test drives it.  This is
    pre-existing and is NOT fixed here: the two branches differ in body, and which
    one was meant to be taken (presumably the third under
    --pseudo-uniform-magnitude-prior-alternate-sampling) is a physics call.

    Pinned because repairing that condition would make an untested block live.
    When this test fails, the branch has become reachable: write a case that
    drives it, the way test_pseudo_uniform_magnitude_survives_a_transverse_boundary_sample
    drives the first two.
    """
    node = _pseudo_uniform_magnitude_node()
    conditions = []
    while isinstance(node, ast.If):
        conditions.append(ast.dump(node.test))
        node = (node.orelse[0] if len(node.orelse) == 1
                and isinstance(node.orelse[0], ast.If) else None)

    assert len(conditions) == 3, (
        "the pseudo-uniform reweight chain now has {} branches, not 3; re-derive "
        "which are reachable".format(len(conditions)))
    assert conditions[1] == conditions[2], (
        "the 3rd branch's condition now differs from the 2nd, so it is REACHABLE. "
        "It divides by prior_map['chiz_plus']*prior_map['chiz_minus'], which is "
        "zero on the spin boundary -- give it a test.")
###
### The prior a coordinate actually gets, over the range that coordinate is actually
### sampled on.
###
### Everything above tests a prior function against the support it is *documented* on.
### That cannot see the other half of the contract: prior_map[c] and prior_range_map[c]
### are chosen independently, by different `if opts...` blocks, and a density evaluated
### outside its own support is not merely inaccurate -- it goes NEGATIVE, and a negative
### prior weight is nan under mcsampler's fractional tempering exponent, nan under
### mcsamplerAdaptiveVolume's np.log (dropped silently, exit 0), and a hard
### "weights must be finite and nonnegative" from the export reweight.
###
### The defect this catches: --transverse-prior taper-down installed triangle_prior, a
### density on [-chi_max, chi_max], for chi1_perp_bar, whose range is [0,1].  At
### --chi-max 0.8 that is negative over 18% of the sampled range, reaching -0.312.
###

# Every value of the two spin-sector options that CIP actually BRANCHES ON, read out of
# the `opts.<dest> == '...'` comparisons in its source.
#
# Not out of the --help text, which was the first thing tried and is wrong twice over:
# --aligned-prior implements 'alignedspin-zprior-positive' without listing it, so the one
# branch this change deliberately leaves alone would never have been swept, and
# --transverse-prior writes its default as "(default)" without quotes, so the DEFAULT
# configuration would never have been swept either.  Reading the branches cannot drift
# from what the script does, because it is what the script does.
def _implemented_choices(dest):
    found = set()
    for node in ast.walk(CIP_TREE):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr == dest
                and isinstance(left.value, ast.Name) and left.value.id == "opts"):
            continue
        try:
            value = ast.literal_eval(node.comparators[0])
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str):
            found.add(value)
    assert found, "no `opts.{} == ...` branches found in CIP".format(dest)
    return sorted(found)


def _swept_choices(option, dest):
    """Branch values, plus the shipped default.

    --aligned-prior uniform has no branch at all -- it means "leave every default alone"
    -- so it has to come from argparse or the configuration almost every run uses would
    never be swept.
    """
    found = set(_implemented_choices(dest))
    found.add(_add_argument_kwargs(CIP_TREE, option)["default"])
    return sorted(found)


ALIGNED_PRIORS = _swept_choices("--aligned-prior", "aligned_prior")
TRANSVERSE_PRIORS = _swept_choices("--transverse-prior", "transverse_prior")

# Guard against the extraction above silently matching nothing, or matching less than it
# did: without this the sweeps would collect fewer cases and still go green.  A rename in
# CIP should fail here, loudly, rather than quietly shrink the sweep.
EXPECTED_ALIGNED = {"uniform", "alignedspin-zprior", "alignedspin-zprior-positive",
                    "volumetric"}
EXPECTED_TRANSVERSE = {"uniform", "uniform-mag", "taper-down", "sqrt-prior",
                       "Rbar-singular", "alignedspin-zprior"}


@pytest.mark.parametrize("option,swept,expected",
                         [("--aligned-prior", ALIGNED_PRIORS, EXPECTED_ALIGNED),
                          ("--transverse-prior", TRANSVERSE_PRIORS, EXPECTED_TRANSVERSE)],
                         ids=["aligned", "transverse"])
def test_every_known_option_value_is_swept(option, swept, expected):
    missing = expected - set(swept)
    assert not missing, "{}: {} implemented but not swept".format(option, sorted(missing))
    assert _add_argument_kwargs(CIP_TREE, option)["default"] in swept

# (chi_max, chi_small_max).  chi_max enters the spin priors, so a sweep at one value --
# and especially at 1.0, where R=chi_max and R=1 coincide -- proves nothing about the
# rest.  The last two rows are the NSBH shape, chi_small_max != chi_max: with the two
# pinned equal, an s2z_bar/chi2_perp_bar entry written against the wrong one of the two
# is indistinguishable from a correct one, and the suite stayed green under exactly that
# mutation.
CHI_RANGES = [(0.5, 0.5), (0.8, 0.8), (0.99, 0.99), (1.0, 1.0), (0.99, 0.05), (0.8, 0.2)]
# kept for the tests that vary only the large-body bound
CHI_MAX_VALUES = [0.5, 0.8, 0.99, 1.0]

# What the run samples in.  Includes both spin coordinate systems so the cylindrical
# (s1z_bar, chi1_perp_bar, phi1) block and the chiz_plus branch of --aligned-prior both
# execute.
SWEEP_COORDS = ["mc", "delta_mc", "s1z", "s2z", "chiz_plus", "chiz_minus",
                "s1z_bar", "s2z_bar", "chi1_perp_bar", "chi2_perp_bar"]

MCSAMPLER_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "RIFT", "integrators",
    "mcsampler.py")

# prior_map holds three mcsampler callables for the angular coordinates.  They are taken
# from the mcsampler SOURCE, by the same ast route as everything else here: importing
# mcsampler pulls in lalsimutils and LAL, whose error handler calls abort().
MCSAMPLER_NAMES = ("uniform_samp_phase", "uniform_samp_theta", "uniform_samp_cos_theta")


def _mcsampler_stub():
    namespace = {"numpy": np, "np": np}
    tree = _parse_script(MCSAMPLER_SOURCE)
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in MCSAMPLER_NAMES):
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         MCSAMPLER_SOURCE, "exec"), namespace)
    missing = set(MCSAMPLER_NAMES) - set(namespace)
    assert not missing, "not found in mcsampler: {}".format(sorted(missing))
    return types.SimpleNamespace(**{name: namespace[name] for name in MCSAMPLER_NAMES})


def _cip_opt_defaults():
    """Every CIP option at its shipped argparse default, as an opts namespace.

    The prior_map rewrites are guarded by a dozen unrelated `if opts.*` conditions, so
    the sweep has to supply all of them.  Reading them off the add_argument calls means
    the sweep runs the default configuration CIP itself would, and keeps doing so.
    """
    defaults = {}
    for node in ast.walk(CIP_TREE):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        flags = []
        for arg in node.args:
            try:
                value = ast.literal_eval(arg)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, str) and value.startswith("--"):
                flags.append(value)
        if not flags:
            continue
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                kwargs[keyword.arg] = None
        action = kwargs.get("action")
        if "default" in kwargs:
            default = kwargs["default"]
        elif action == "store_true":
            default = False
        elif action == "store_false":
            default = True
        else:
            default = None
        dest = kwargs.get("dest") or flags[0][2:].replace("-", "_")
        defaults[dest] = default
    assert "aligned_prior" in defaults and "transverse_prior" in defaults
    return defaults


def _assigns_into(node, names):
    """Does this subtree assign into any of `names` by subscript, e.g. prior_map[c]=..?"""
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id in names):
                return True
    return False


def _installed_priors(aligned_prior, transverse_prior, chi_max,
                      coords=tuple(SWEEP_COORDS), chi_small_max=None):
    """CIP's prior_map / prior_range_map as a run with these options would hold them.

    Builds the two shipped dict literals whole -- not trimmed, as the eccentricity tests
    above do -- then runs every top-level `if` block that rewrites either of them, in
    source order.  Selection is on "does this block assign prior_map[...] or
    prior_range_map[...]", so a future option that rewrites one of them is swept without
    anyone editing this list.
    """
    namespace = _make_namespace(coords=coords)
    namespace["functools"] = functools
    namespace["mcsampler"] = _mcsampler_stub()
    namespace["chi_max"] = chi_max
    namespace["chi_small_max"] = chi_max if chi_small_max is None else chi_small_max
    namespace["lambda_plus_max"] = LAMBDA_MAX
    namespace["p_Rbar"] = _p_rbar()
    namespace["scipy"] = types.SimpleNamespace(stats=scipy_stats, special=scipy_special)
    opts = _cip_opt_defaults()
    opts.update(aligned_prior=aligned_prior, transverse_prior=transverse_prior,
                lambda_max=LAMBDA_MAX)
    namespace["opts"] = types.SimpleNamespace(**opts)

    _load_priors(namespace)

    maps = ("prior_map", "prior_range_map")
    for name in maps:
        for node in CIP_TREE.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == name):
                _exec_in(namespace, node)
                break
        else:
            raise AssertionError("could not find the {} dict in CIP".format(name))

    rewrote = 0
    for node in CIP_TREE.body:
        if isinstance(node, ast.If) and _assigns_into(node, maps):
            _exec_in(namespace, node)
            rewrote += 1
    assert rewrote >= 4, (
        "only {} option blocks rewrite prior_map/prior_range_map; the selection above "
        "has stopped matching CIP".format(rewrote))

    return namespace["prior_map"], namespace["prior_range_map"]


@pytest.mark.parametrize("chi_max,chi_small_max", CHI_RANGES,
                         ids=["{}-{}".format(*row) for row in CHI_RANGES])
@pytest.mark.parametrize("transverse_prior", TRANSVERSE_PRIORS)
@pytest.mark.parametrize("aligned_prior", ALIGNED_PRIORS)
def test_installed_prior_is_never_negative_on_its_sampling_range(
        aligned_prior, transverse_prior, chi_max, chi_small_max):
    """No coordinate may be given a density that goes negative where it is sampled.

    Evaluated strictly inside the range, on a fine grid: the boundary cell of the spin
    z-prior is a separate, chi_max-INDEPENDENT defect (the `+1e-7` offset) and is not
    what this sweep is for.  +inf is likewise allowed through -- s_component_sqrt_prior
    and normalized_Rbar_singular_prior have an integrable singularity at 0 by design --
    so the assertions are "not nan" and ">= 0", which is exactly the class that turns
    into nan weights downstream.
    """
    prior_map, prior_range_map = _installed_priors(
        aligned_prior, transverse_prior, chi_max, chi_small_max=chi_small_max)

    checked = 0
    for coord in sorted(prior_map):
        if coord not in prior_range_map:
            continue
        lo, hi = [float(v) for v in prior_range_map[coord]]
        assert np.isfinite([lo, hi]).all() and lo < hi, (
            "{}: sampling range {} is not usable".format(coord, prior_range_map[coord]))
        x = np.linspace(lo, hi, 401)[1:-1]

        value = np.asarray(prior_map[coord](x), dtype=float)
        value = np.broadcast_to(value, x.shape)
        checked += 1

        assert not np.any(np.isnan(value)), "{}: nan density".format(coord)
        worst = int(np.argmin(value))
        assert value[worst] >= 0, (
            "{}: density {:.4g} at {:.4g}, negative over {:.1f}% of its sampling range "
            "{}".format(coord, value[worst], x[worst], 100 * np.mean(value < 0),
                        [lo, hi]))

    assert checked > 20, "only swept {} coordinates".format(checked)


# The cylindrical spin coordinates, chi{1,2}_perp_bar and s{1,2}z_bar.  The two halves are
# NOT the same kind of quantity, which is the whole point of this group:
#
#   chi1_perp_bar = chi1_perp/sqrt(1-s1z^2)  -- already divided by the UNIT sphere's
#       radius at that height, so it has no chi-max scale.  Range [0,1] with no
#       chi_small_max variant, and every density on it must be normalized there.
#   s1z_bar       = s1z                      -- an unscaled spin component.  It carries
#       chi-max directly, and --aligned-prior sets its range alongside its prior.
#
# What both must satisfy is the same, and is what the sweeps below check: the installed
# density has to be a normalized, positive density over the range actually drawn.
BAR_COORDS = ("chi1_perp_bar", "chi2_perp_bar", "s1z_bar", "s2z_bar")


@pytest.mark.parametrize("chi_max,chi_small_max", CHI_RANGES,
                         ids=["{}-{}".format(*row) for row in CHI_RANGES])
@pytest.mark.parametrize("transverse_prior", TRANSVERSE_PRIORS)
@pytest.mark.parametrize("aligned_prior", ALIGNED_PRIORS)
def test_bar_coordinate_prior_is_normalized_on_its_sampling_range(
        aligned_prior, transverse_prior, chi_max, chi_small_max):
    """Whatever option installs it, a _bar density integrates to 1 over its own range.

    Stronger than the sign check and independent of chi_max: reusing a [-R,R] cartesian
    density on the half-range [0,1] loses half its mass whatever R is.  On the base
    commit --transverse-prior taper-down integrated to 0.500 at --chi-max 1 and 0.469 at
    --chi-max 0.8, i.e. a 1.4-1.5 nat lnZ offset across the two spins, on top of the
    negative-density region.
    """
    prior_map, prior_range_map = _installed_priors(
        aligned_prior, transverse_prior, chi_max, chi_small_max=chi_small_max)

    for coord in BAR_COORDS:
        density = prior_map[coord]
        lo, hi = [float(v) for v in prior_range_map[coord]]
        # 0 is where every singular member of this family (sqrt, Rbar-singular, zprior)
        # has its integrable pole; hand it to QUADPACK so it subdivides rather than
        # sampling the pole and aborting.
        points = [0.0] if lo <= 0.0 <= hi else None
        integrand = lambda u: float(np.asarray(density(np.array([u])), dtype=float)[0])
        total, err = integrate.quad(integrand, lo, hi, limit=400, points=points)

        assert err < 1e-3, "{}: quadrature did not converge (err={})".format(coord, err)
        assert total == pytest.approx(1.0, rel=2e-3), (
            "{} under --aligned-prior {} --transverse-prior {} --chi-max {} "
            "--chi-small-max {}: the "
            "installed density integrates to {:.6f} over its sampling range {}, not "
            "1".format(coord, aligned_prior, transverse_prior, chi_max, chi_small_max,
                       total, [lo, hi]))


def test_triangle_prior_is_zero_outside_its_support():
    """triangle_prior is a density, so it is zero past |x|=R, not negative.

    The clamp is what stops the defect above from being a nan the moment any caller --
    now or later -- hands this function an argument past its own R.
    """
    prior = PRIORS["triangle_prior"]
    R = CHI_MAX

    outside = np.linspace(R, 4 * R, 25)
    assert np.all(np.asarray(prior(outside), dtype=float) == 0.0)

    # and the interior is untouched by the clamp
    inside = np.linspace(-0.999 * R, 0.999 * R, 25)
    assert np.allclose(np.asarray(prior(inside), dtype=float),
                       (1 - np.abs(inside / R)) / R)


@pytest.mark.parametrize("chi_max,chi_small_max", CHI_RANGES,
                         ids=["{}-{}".format(*row) for row in CHI_RANGES])
@pytest.mark.parametrize("transverse_prior", TRANSVERSE_PRIORS)
@pytest.mark.parametrize("aligned_prior", ALIGNED_PRIORS)
def test_bar_coordinate_prior_covers_its_sampling_range(
        aligned_prior, transverse_prior, chi_max, chi_small_max):
    """A _bar density must be positive everywhere its coordinate is sampled.

    The third failure mode of a prior paired with the wrong range, and the one the
    other two cannot see.  A density that is identically ZERO over part of the range
    it is drawn on integrates to 1 and is never negative, so it passes both checks
    above -- but log(0) is -inf, the sample is dropped by the sampler's own isfinite
    admission filter, and the run quietly explores a smaller volume than the range it
    reports.

    This is what --aligned-prior alignedspin-zprior did to s1z_bar: the chi_max-width
    z-prior on a [-1,1] range, zero over 20% of it at --chi-max 0.8.  Before the clamp
    in #361 that region was NEGATIVE and the sign check caught it; the clamp turned it
    into zero and silently took the coverage with it.  Hence a separate predicate.

    Evaluated strictly inside the range: every correct density in this family is zero
    at one endpoint or the other (2*Rbar at 0, the taper at 1, the z-prior at +-1),
    which is the support boundary, not a gap.
    """
    prior_map, prior_range_map = _installed_priors(
        aligned_prior, transverse_prior, chi_max, chi_small_max=chi_small_max)

    for coord in BAR_COORDS:
        lo, hi = [float(v) for v in prior_range_map[coord]]
        x = np.linspace(lo, hi, 401)[1:-1]
        value = np.broadcast_to(
            np.asarray(prior_map[coord](x), dtype=float), x.shape)

        dead = value <= 0
        assert not np.any(dead), (
            "{} under --aligned-prior {} --transverse-prior {} --chi-max {} "
            "--chi-small-max {}: the "
            "installed density is zero over {:.1f}% of its sampling range {}, out to "
            "|x| = {:.4g}".format(coord, aligned_prior, transverse_prior, chi_max,
                                  chi_small_max, 100 * np.mean(dead), [lo, hi],
                                  np.max(np.abs(x[dead])) if np.any(dead) else 0.0))


###
### Wiring.
###
### The three sweeps above check that whatever density is installed is a usable one.
### They cannot see whether the OPTION installed it: CIP's defaults (2*Rbar on [0,1],
### 3(1-z^2)/4 on [-1,1]) satisfy all three predicates, so deleting the whole elif
### chain leaves the suite green.  Measured: with the four prior_map assignments this
### change adds removed, the sweeps still reported 0 failures.  These tests assert the
### identity of what lands in prior_map instead.
###

def _installed(coord, aligned_prior="uniform", transverse_prior="uniform", chi_max=0.8):
    prior_map, prior_range_map = _installed_priors(
        aligned_prior, transverse_prior, chi_max)
    return prior_map[coord], prior_range_map[coord]


def _density_name(fn):
    """The underlying function name, through functools.partial."""
    return getattr(getattr(fn, "func", fn), "__name__", None)


# (option value, coordinate, the density that value must install)
TRANSVERSE_WIRING = [
    ("taper-down", "chi1_perp_bar", "normalized_Rbar_taper_prior"),
    ("taper-down", "chi2_perp_bar", "normalized_Rbar_taper_prior"),
    ("sqrt-prior", "chi1_perp_bar", "normalized_Rbar_sqrt_prior"),
    ("sqrt-prior", "chi2_perp_bar", "normalized_Rbar_sqrt_prior"),
    # unchanged by this commit, pinned so the [-R,R] cartesian densities cannot drift
    # back onto the normalized radial coordinate
    ("uniform-mag", "chi1_perp_bar", "unnormalized_uniform_prior"),
    ("Rbar-singular", "chi1_perp_bar", "normalized_Rbar_singular_prior"),
    ("taper-down", "s1x", "triangle_prior"),
    ("sqrt-prior", "s1x", "s_component_sqrt_prior"),
]


@pytest.mark.parametrize("value,coord,expected", TRANSVERSE_WIRING,
                         ids=["{}-{}".format(row[0], row[1]) for row in TRANSVERSE_WIRING])
def test_transverse_prior_installs_the_named_density(value, coord, expected):
    density, _ = _installed(coord, transverse_prior=value)
    assert _density_name(density) == expected, (
        "--transverse-prior {} gave {} the density {}, not {}".format(
            value, coord, _density_name(density), expected))


@pytest.mark.parametrize("coord", ["chi1_perp_bar", "chi2_perp_bar"])
@pytest.mark.parametrize("value", ["taper-down", "sqrt-prior", "uniform-mag",
                                   "Rbar-singular"])
def test_transverse_prior_actually_changes_the_radial_density(value, coord):
    """Each --transverse-prior value must move chi_perp_bar off the default.

    The shape checks cannot tell "the option installed its density" from "the option did
    nothing and the default is fine", because the default is fine.  This can: it compares
    the installed density against the one the same coordinate gets under
    --transverse-prior uniform, pointwise.
    """
    chosen, (lo, hi) = _installed(coord, transverse_prior=value)
    default, _ = _installed(coord, transverse_prior="uniform")
    x = np.linspace(lo, hi, 51)[1:-1]

    assert not np.allclose(np.asarray(chosen(x), dtype=float),
                           np.asarray(default(x), dtype=float)), (
        "--transverse-prior {} leaves {} on the default density".format(value, coord))


ZPRIOR_BAR_WIRING = [
    ("alignedspin-zprior", "s1z_bar", "s_component_zprior"),
    ("alignedspin-zprior", "s2z_bar", "s_component_zprior"),
    ("alignedspin-zprior-positive", "s1z_bar", "s_component_zprior_positive"),
    ("alignedspin-zprior-positive", "s2z_bar", "s_component_zprior_positive"),
]


@pytest.mark.parametrize("value,coord,expected", ZPRIOR_BAR_WIRING,
                         ids=["{}-{}".format(row[0], row[1]) for row in ZPRIOR_BAR_WIRING])
def test_aligned_prior_reaches_the_bar_coordinate(value, coord, expected):
    density, _ = _installed(coord, aligned_prior=value)
    assert _density_name(density) == expected, (
        "--aligned-prior {} gave {} the density {}, not {}".format(
            value, coord, _density_name(density), expected))


@pytest.mark.parametrize("chi_max", CHI_MAX_VALUES)
@pytest.mark.parametrize("coord,cartesian", [("s1z_bar", "s1z"), ("s2z_bar", "s2z")])
def test_bar_z_prior_matches_its_cartesian_twin(coord, cartesian, chi_max):
    """s1z_bar IS s1z, so --aligned-prior must give the two the same density.

    The predicate behind keeping R=chi_max here rather than setting R=1: the z-prior's R
    is a maximum spin MAGNITUDE, and s1z_bar is an unscaled spin component (lalsimutils
    extract_param returns self.s1z), so under R=1 the two names would carry different
    priors for one physical quantity.

    For s1z the two map entries are the same object, so this pair is a tautology and the
    s2z pair (distinct partials) is what has teeth here.  The non-vacuous check on both
    is test_bar_z_prior_support_edge_is_the_spin_bound below, which locates R rather than
    comparing two lookups.
    """
    bar, bar_range = _installed(coord, aligned_prior="alignedspin-zprior",
                                chi_max=chi_max)
    twin, twin_range = _installed(cartesian, aligned_prior="alignedspin-zprior",
                                  chi_max=chi_max)

    # both entries carry the same 0.999 endpoint factor now, so this is the same
    # interval twice; kept as a min/max so the test still samples inside BOTH if one of
    # the two range entries is ever changed on its own
    lo = max(float(bar_range[0]), float(twin_range[0]))
    hi = min(float(bar_range[1]), float(twin_range[1]))
    x = np.linspace(lo, hi, 101)[1:-1]

    assert np.allclose(np.asarray(bar(x), dtype=float),
                       np.asarray(twin(x), dtype=float)), (
        "{} and {} have different densities under --aligned-prior alignedspin-zprior "
        "at --chi-max {}".format(coord, cartesian, chi_max))


@pytest.mark.parametrize("chi_max,chi_small_max", CHI_RANGES,
                         ids=["{}-{}".format(*row) for row in CHI_RANGES])
@pytest.mark.parametrize("coord,scale", [("s1z_bar", "chi_max"),
                                         ("s2z_bar", "chi_small_max")])
def test_bar_z_prior_support_edge_is_the_spin_bound(coord, scale, chi_max,
                                                    chi_small_max):
    """--aligned-prior alignedspin-zprior must put s{1,2}z_bar's support edge at chi-max.

    The behavioural form of "R is a maximum spin MAGNITUDE here".  Asserting the
    function's NAME cannot see this: s_component_zprior at R=1, at R=chi_max and at
    R=chi_small_max are all the same function under a different partial, and
    functools.partial hides the keyword from __name__.  The z-prior is exactly zero at
    |x| = R and positive just inside, so the edge locates R without reading it.

    Pins the decision to keep R=chi_max rather than raise it to 1: under R=1 the edge
    would sit at 1 for every chi-max, and s{1,2}z_bar -- which IS s1z -- would describe a
    different prior than its own cartesian name.  It also separates chi_max from
    chi_small_max, which a run with --chi-small-max set does not hold equal.
    """
    bound = chi_max if scale == "chi_max" else chi_small_max
    prior_map, _ = _installed_priors("alignedspin-zprior", "uniform", chi_max,
                                     chi_small_max=chi_small_max)
    density = prior_map[coord]

    at_edge = float(np.asarray(density(np.array([bound])), dtype=float)[0])
    inside = float(np.asarray(density(np.array([0.99 * bound])), dtype=float)[0])
    outside = float(np.asarray(density(np.array([1.01 * bound])), dtype=float)[0])

    assert at_edge == 0.0, (
        "{}: density is {:.6g} at |x| = {} (the {} edge), so its R is not {}".format(
            coord, at_edge, bound, scale, scale))
    assert inside > 0, "{}: density is {:.6g} just inside the edge".format(coord, inside)
    assert outside == 0.0, (
        "{}: density is {:.6g} outside |x| = {}".format(coord, outside, bound))
