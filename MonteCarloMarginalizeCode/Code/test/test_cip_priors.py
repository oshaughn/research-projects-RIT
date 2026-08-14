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
import os
import re
import types

import numpy as np
import pytest

scipy_stats = pytest.importorskip("scipy.stats")
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


def _make_namespace(ecc_min=ECC_MIN, eccentricity_prior="uniform", coords=()):
    """The module-level constants the priors and the option wiring close over.

    `coords` stands in for CIP's low_level_coord_names, the coordinates the Monte
    Carlo actually samples in; the eccentricity block consults it because the ln
    coordinate needs a positive floor whatever the prior is.
    """
    return {
        "low_level_coord_names": list(coords),
        "np": np,
        "numpy": np,
        "scipy": types.SimpleNamespace(stats=scipy_stats),
        "chi_max": CHI_MAX,
        "chi_small_max": CHI_MAX,
        "ECC_MIN": ecc_min,
        "ECC_MAX": ECC_MAX,
        "MEANPERANO_MIN": 0.0,
        "MEANPERANO_MAX": 2 * np.pi,
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
    "precession_prior": (0.0, 2.0),
    "lambda_prior": (LAMBDA_MIN, LAMBDA_MAX),
    "lambda_small_prior": (LAMBDA_MIN, LAMBDA_SMALL_MAX),
    "lambda_tilde_prior": (0.0, LAMBDA_MAX),
    "delta_lambda_tilde_prior": (-500.0, 500.0),
    "unnormalized_log_prior": (0.1, 10.0),
    "normalized_Rbar_prior": (0.0, 1.0),
    "normalized_Rbar_singular_prior": (1e-6, 1.0),
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


def _eccentricity_setup(ecc_min=ECC_MIN, eccentricity_prior="uniform", coords=()):
    """Reproduce CIP's eccentricity prior selection: defaults, then the option block."""
    namespace = _make_namespace(ecc_min=ecc_min,
                                eccentricity_prior=eccentricity_prior,
                                coords=coords)
    _load_priors(namespace)
    # ln(ECC_MIN) with --ecc-min 0 is -inf here exactly as it is in CIP; the option
    # block is what repairs it, and that repair is the thing under test
    with np.errstate(divide="ignore"):
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
