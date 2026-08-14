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
where a density uniform in ln(e) needs ``log(ECC_MAX/ECC_MIN)`` -- which is
negative for the shipped ecc-min/ecc-max defaults, so the density would have
been negative everywhere had it evaluated at all.

Both were fixed in 0.0.17.13.  This file is the missing half: the fix went in
without a test, so nothing stops the next edit to these functions from
reintroducing either defect.  Checked against the 0.0.17.12 source, the tests
below fail 4/60 -- all four naming log_eccentricity_prior -- and pass 60/60
against 0.0.17.13.

Rather than transcribe the priors here -- which lets the test silently drift
away from the shipped code, the usual failure mode of a copied reference
implementation -- this module extracts the actual ``def`` blocks from the CIP
source with ``ast`` and execs them in a namespace holding numpy and the handful
of module-level constants they close over.  The functions under test are
therefore byte-identical to the ones CIP runs.

Two layers of coverage:

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


def _load_priors():
    """Exec the prior ``def`` blocks out of the CIP source, verbatim.

    Only top-level FunctionDef nodes are taken, so the surrounding script
    (argparse, I/O, the fitting machinery) never runs.  Selection is on
    'prior' appearing anywhere in the name, NOT a '_prior' suffix: the suffix
    rule silently skips s_component_zprior, s_component_zprior_positive and
    the two *volumetricprior densities, which is most of the spin sector.
    """
    with open(CIP_SCRIPT) as handle:
        tree = ast.parse(handle.read())

    namespace = {
        "np": np,
        "numpy": np,
        "scipy": types.SimpleNamespace(stats=scipy_stats),
        "chi_max": CHI_MAX,
        "chi_small_max": CHI_MAX,
        "ECC_MIN": ECC_MIN,
        "ECC_MAX": ECC_MAX,
        "MEANPERANO_MIN": 0.0,
        "MEANPERANO_MAX": 2 * np.pi,
        "lambda_min": LAMBDA_MIN,
        "lambda_max": LAMBDA_MAX,
        "lambda_small_max": LAMBDA_SMALL_MAX,
        "mc_min": MC_MIN,
        "mc_max": MC_MAX,
        "p_Rbar": _p_rbar(),
        # lambda_tilde_prior reads opts directly
        "opts": types.SimpleNamespace(lambda_max=LAMBDA_MAX),
    }

    found = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if "prior" not in node.name.lower() or node.name in NOT_A_DENSITY:
            continue
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, CIP_SCRIPT, "exec"), namespace)
        found[node.name] = namespace[node.name]
    return found


PRIORS = _load_priors()

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
                 "uniform_eccentricity_ln_prior", "eccentricity_squared_prior"):
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
