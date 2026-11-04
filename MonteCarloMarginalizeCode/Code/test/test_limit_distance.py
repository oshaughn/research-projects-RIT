#!/usr/bin/env python
"""
Regression tests for --limit-distance: a SAMPLING-only restriction of the distance
range that must leave the evidence normalization alone.

WHY THIS IS NOT THE SAME AS THE ANGULAR --limit-* BOXES.  Those narrow the prior
SUPPORT: the angle priors are never renormalized, so a box costs exactly the prior
mass outside it and lnZ moves by an analytic amount (a whole-sky box reproduces the
unboxed lnZ to +0.000000; a narrow one returns ln(dOmega/4pi)).  --limit-distance
instead keeps the prior AND its normalization over the full [--d-min,--d-max] and
narrows only what the sampler draws from, so the reported lnZ is the SAME NUMBER a
full-range run reports -- no correction -- to the extent the likelihood is negligible
outside the box.  That is what makes lnZ comparable across runs and across samplers
while the quadrature gets to stay cheap at high amplitude, where the distance
posterior has narrowed as 1/rho.

THE DEFECT THESE TESTS LOCK DOWN.  The obvious implementation narrows
param_limits["distance"], which the Euclidean prior lambda ALSO read for its own
normalization:

    dist_prior_pdf = lambda x: x**2/(param_limits["distance"][1]**3/3.
                                     - param_limits["distance"][0]**3/3.)

so the density silently renormalizes to the box.  The failure is invisible in the
usual acceptance check -- with the likelihood inside the box, lnZ still comes back
"unchanged" -- because the renormalization exactly cancels the missing prior mass.
The signature that separates the two is a CONSTANT likelihood: with the prior
correctly left alone, narrowing must cost exactly the prior mass outside the box
(test_constant_likelihood_*), and it is the renormalizing implementation that
returns "unchanged" there.  Both directions are tested.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import RIFT.integrators.mcsampler as mcsampler
from RIFT.integrators.mcsampler import distance_limit_range, distance_sampler_kwargs

# np.trapz was REMOVED in numpy 2.x (renamed np.trapezoid); CI runs both lanes.
_trapz = getattr(np, 'trapezoid', None) or np.trapz

# The physical prior range: what --d-min/--d-max set, and what the prior must stay
# normalized over no matter how the sampling is restricted.
D_MIN, D_MAX = 1.0, 10000.0

# A synthetic distance posterior: a Gaussian peak, and a box at +-6 sigma around it.
# The analytic prior mass of the likelihood OUTSIDE the box is ~2e-9 of the total, so
# the "unchanged lnZ" claim has an exact reference to be checked against and is not a
# statement about the tolerance of a sampler.
D_PEAK, D_SIGMA = 2000.0, 200.0
BOX = (D_PEAK - 6 * D_SIGMA, D_PEAK + 6 * D_SIGMA)      # (800, 3200)


def _like(d):
    d = np.asarray(d, dtype=float)
    return np.exp(-0.5 * ((d - D_PEAK) / D_SIGMA) ** 2)


def _reference_integral(lo, hi, d_prior='Euclidean', like=_like, n=2000001):
    """int_lo^hi L(d) pi(d) dd with pi normalized over the FULL [D_MIN, D_MAX].

    This is the expectation of the estimator the sampler forms, computed without
    Monte Carlo noise, so a disagreement is a normalization disagreement.
    """
    kw = distance_sampler_kwargs(mcsampler, (lo, hi), (D_MIN, D_MAX), d_prior=d_prior)
    grid = np.linspace(lo, hi, n)
    return _trapz(like(grid) * np.asarray(kw['prior_pdf'](grid), dtype=float), grid)


###
### 1. The option parser: loud on anything ambiguous
###

def test_limit_range_parses_and_validates():
    assert distance_limit_range('800,3200', D_MIN, D_MAX) == (800.0, 3200.0)
    assert distance_limit_range(' 800 , 3200 ', D_MIN, D_MAX) == (800.0, 3200.0)


@pytest.mark.parametrize('bad', ['3200,800',      # inverted
                                 '800,800',       # empty
                                 '800',           # not a pair
                                 '800,900,1000',  # too many
                                 'a,b',           # unparseable
                                 'nan,3200',      # non-finite
                                 ])
def test_limit_range_rejects_bad_requests(bad):
    with pytest.raises(ValueError):
        distance_limit_range(bad, D_MIN, D_MAX)


@pytest.mark.parametrize('bad', ['0.5,3200',      # below --d-min
                                 '800,20000',     # above --d-max
                                 ])
def test_limit_range_must_lie_inside_the_prior_range(bad):
    """A box outside [d_min,d_max] asks for distances the prior gives no mass.
    Clipping it silently would turn a configuration error into a quiet answer."""
    with pytest.raises(ValueError):
        distance_limit_range(bad, D_MIN, D_MAX)


###
### 2. The prior is NOT renormalized to the box
###

@pytest.mark.parametrize('d_prior', ['Euclidean', 'pseudo_cosmo'])
def test_prior_pdf_is_normalized_over_the_full_range_not_the_box(d_prior):
    kw_full = distance_sampler_kwargs(mcsampler, (D_MIN, D_MAX), (D_MIN, D_MAX), d_prior=d_prior)
    kw_box = distance_sampler_kwargs(mcsampler, BOX, (D_MIN, D_MAX), d_prior=d_prior)
    grid = np.linspace(D_MIN, D_MAX, 200001)
    p_full = np.asarray(kw_full['prior_pdf'](grid), dtype=float)
    p_box = np.asarray(kw_box['prior_pdf'](grid), dtype=float)
    # identical densities: the box changed the SAMPLER, not the prior
    assert np.allclose(p_full, p_box, rtol=0, atol=0)
    assert _trapz(p_box, grid) == pytest.approx(1.0, rel=1e-6)
    # and the box-renormalized density -- the defect -- is a DIFFERENT function
    kw_wrong = distance_sampler_kwargs(mcsampler, BOX, BOX, d_prior=d_prior)
    inbox = (grid >= BOX[0]) & (grid <= BOX[1])
    p_wrong = np.asarray(kw_wrong['prior_pdf'](grid), dtype=float)
    assert not np.allclose(p_wrong[inbox], p_box[inbox])
    # (its own grid, so the endpoints land exactly on the box)
    gbox = np.linspace(BOX[0], BOX[1], 200001)
    assert _trapz(np.asarray(kw_wrong['prior_pdf'](gbox), dtype=float), gbox) == pytest.approx(1.0, rel=1e-6)


def test_euclidean_prior_normalization_is_the_full_range_analytic_one():
    kw = distance_sampler_kwargs(mcsampler, BOX, (D_MIN, D_MAX), d_prior='Euclidean')
    d = np.array([500.0, 2000.0, 9000.0])
    expected = d ** 2 / (D_MAX ** 3 / 3. - D_MIN ** 3 / 3.)
    assert np.allclose(np.asarray(kw['prior_pdf'](d), dtype=float), expected)


def test_unknown_prior_is_refused():
    with pytest.raises(ValueError):
        distance_sampler_kwargs(mcsampler, BOX, (D_MIN, D_MAX), d_prior='cosmo')


def test_default_path_reproduces_the_historical_expressions_bitwise():
    """No existing default may move.  With sampling_range == prior_range the helper
    must reproduce, BIT FOR BIT, the expressions the ILE driver used before
    --limit-distance existed (the pre-change source is quoted in each comment)."""
    import functools
    import RIFT.likelihood.priors_utils as priors_utils
    x = np.linspace(D_MIN, D_MAX, 100001)
    u = np.linspace(0.0, 1.0, 100001)

    # dist_prior_pdf = lambda x: x**2/(param_limits["distance"][1]**3/3.
    #                                  - param_limits["distance"][0]**3/3.)
    kw = distance_sampler_kwargs(mcsampler, (D_MIN, D_MAX), (D_MIN, D_MAX), d_prior='Euclidean')
    old = x ** 2 / (D_MAX ** 3 / 3. - D_MIN ** 3 / 3.)
    assert np.array_equal(np.asarray(kw['prior_pdf'](x), dtype=float), old)

    # nm = priors_utils.dist_prior_pseudo_cosmo_eval_norm(lo, hi)
    # dist_prior_pdf = functools.partial(priors_utils.dist_prior_pseudo_cosmo, nm=nm, xpy=...)
    kwp = distance_sampler_kwargs(mcsampler, (D_MIN, D_MAX), (D_MIN, D_MAX), d_prior='pseudo_cosmo')
    nm = priors_utils.dist_prior_pseudo_cosmo_eval_norm(D_MIN, D_MAX)
    old_p = priors_utils.dist_prior_pseudo_cosmo(x, nm=nm, xpy=np)
    assert np.array_equal(np.asarray(kwp['prior_pdf'](x), dtype=float), np.asarray(old_p, dtype=float))

    # dist_sampler = mcsampler.ret_uniform_samp_vector_alt(lo, hi)
    # dist_sampler_cdf_inv = functools.partial(mcsampler.uniform_samp_cdf_inv_vector, lo, hi)
    assert np.array_equal(np.asarray(kw['pdf'](x)),
                          np.asarray(mcsampler.ret_uniform_samp_vector_alt(D_MIN, D_MAX)(x)))
    assert np.array_equal(np.asarray(kw['cdf_inv'](u)),
                          np.asarray(functools.partial(mcsampler.uniform_samp_cdf_inv_vector,
                                                       D_MIN, D_MAX)(u)))
    # left_limit = param_limits["distance"][0], right_limit = param_limits["distance"][1]
    assert kw['left_limit'] == D_MIN and kw['right_limit'] == D_MAX


###
### 3. THE ACCEPTANCE TEST: narrowed vs full-range evidence
###

@pytest.mark.parametrize('d_prior', ['Euclidean', 'pseudo_cosmo'])
def test_ACCEPTANCE_narrowed_and_full_range_evidence_agree(d_prior):
    """The reported evidence must not move when the sampling range is narrowed.

    Evaluated as the estimator's EXPECTATION (deterministic quadrature of
    L*pi over [left_limit,right_limit]) so the number is a statement about the
    normalization and not about a sampler's variance.  The residual is the
    likelihood mass outside +-6 sigma, which is real physics, not an error: it
    is why the option's help says to keep the box wide vs the posterior.
    """
    z_full = _reference_integral(D_MIN, D_MAX, d_prior)
    z_box = _reference_integral(BOX[0], BOX[1], d_prior)
    dlnZ = abs(np.log(z_box) - np.log(z_full))
    assert dlnZ < 1e-6, "narrowing moved lnZ by {:.3e} nats".format(dlnZ)


@pytest.mark.parametrize('d_prior', ['Euclidean', 'pseudo_cosmo'])
def test_ACCEPTANCE_the_renormalizing_implementation_would_fail_it(d_prior):
    """Power check for the test above: if the prior were renormalized to the box
    (prior_range narrowed along with sampling_range -- the defect), lnZ would move
    by ln of the prior-mass fraction, which for this box is several nats."""
    z_full = _reference_integral(D_MIN, D_MAX, d_prior)
    kw = distance_sampler_kwargs(mcsampler, BOX, BOX, d_prior=d_prior)   # the defect
    grid = np.linspace(BOX[0], BOX[1], 2000001)
    z_wrong = _trapz(_like(grid) * np.asarray(kw['prior_pdf'](grid), dtype=float), grid)
    assert abs(np.log(z_wrong) - np.log(z_full)) > 1.0


@pytest.mark.parametrize('d_prior', ['Euclidean', 'pseudo_cosmo'])
def test_constant_likelihood_narrowing_costs_exactly_the_prior_mass(d_prior):
    """With L == 1 there is no 'negligible outside' region, so the CORRECT answer is
    that narrowing reduces the integral to the prior mass of the box.  An
    implementation that renormalizes the prior returns 1.0 here -- i.e. 'unchanged',
    which is the signature of the defect, not of success."""
    one = lambda d: np.ones(np.shape(d))
    z_full = _reference_integral(D_MIN, D_MAX, d_prior, like=one)
    z_box = _reference_integral(BOX[0], BOX[1], d_prior, like=one)
    assert z_full == pytest.approx(1.0, rel=1e-6)          # prior integrates to one
    assert z_box < 0.5 * z_full                            # narrowing DID cost mass
    if d_prior == 'Euclidean':
        expected = ((BOX[1] ** 3 - BOX[0] ** 3) / (D_MAX ** 3 - D_MIN ** 3))
        assert z_box == pytest.approx(expected, rel=1e-6)


###
### 3b. The PHYSICAL likelihood shape, and where the claim stops holding
###
# The Gaussian above is a convenient stand-in, but it is not the shape a real
# extrinsic likelihood has in distance: exp(K x - R x^2/2) with x = d_ref/d tends to
# ONE as d -> infinity, not to zero.  So there is always a far-field contribution
# ~ (prior mass outside the box) x 1 that the box throws away, and "narrowing costs
# no evidence" is a statement about AMPLITUDE, not a theorem.  These tests MEASURE
# the crossover rather than assume it -- which is what the option's help warns about,
# and the reason it is opt-in per run.
#
# Everything here is done in log space: lnL at the peak is rho^2/2, which overflows
# a float at rho ~ 38.

D_REF = 1000.0


def _physical_log_like(rho, d_star):
    """lnL(d) = K x - R x^2/2, x = D_REF/d; peak lnL = rho^2/2 at d = d_star."""
    x_star = D_REF / d_star
    R = rho ** 2 / x_star ** 2
    K = R * x_star
    def log_like(d):
        x = D_REF / np.asarray(d, dtype=float)
        return K * x - 0.5 * R * x ** 2
    return log_like


def _reference_lnZ(lo, hi, log_like, d_prior='Euclidean', n=400001):
    """ln int_lo^hi L(d) pi(d) dd, with pi normalized over the FULL [D_MIN,D_MAX]."""
    from scipy.special import logsumexp
    kw = distance_sampler_kwargs(mcsampler, (lo, hi), (D_MIN, D_MAX), d_prior=d_prior)
    grid = np.linspace(lo, hi, n)
    dd = np.full(n, grid[1] - grid[0])
    dd[0] = dd[-1] = 0.5 * (grid[1] - grid[0])          # trapezoid
    log_pi = np.log(np.asarray(kw['prior_pdf'](grid), dtype=float))
    return float(logsumexp(log_like(grid) + log_pi + np.log(dd)))


# MEASURED on this configuration (d_star=2000 Mpc, +-6/rho fractional box, prior
# [1,10000] Mpc): |dlnZ| = 6.2e-4 at rho=10, 5.8e-6 at rho=20, 1.6e-7 at rho=40
# (and 1.0e-1 at rho=5, which is why the crossover gets its own test below).
# The tolerances below are those numbers rounded up, not aspirations.
@pytest.mark.parametrize('rho,tol', [(10.0, 1e-3), (20.0, 1e-5), (40.0, 1e-6)])
def test_ACCEPTANCE_physical_likelihood_shape(rho, tol):
    d_star = 2000.0
    log_like = _physical_log_like(rho, d_star)
    half = 6.0 / rho
    lo, hi = d_star * (1 - half), d_star * (1 + half)
    lnz_full = _reference_lnZ(D_MIN, D_MAX, log_like)
    lnz_box = _reference_lnZ(lo, hi, log_like)
    dlnZ = abs(lnz_box - lnz_full)
    assert dlnZ < tol, "rho={}: narrowing moved lnZ by {:.3e} nats".format(rho, dlnZ)


def test_the_far_field_is_what_breaks_it_at_low_amplitude():
    """The honest negative, recorded so the crossover is documented and not assumed:
    at rho=2 the distance posterior is not localized -- the prior's own d^2 mass at
    large d carries the integral -- and a +-30% box costs 3.3 nats (MEASURED).
    A run that narrows the box at low amplitude gets a wrong lnZ, quietly."""
    d_star = 2000.0
    log_like = _physical_log_like(2.0, d_star)
    lo, hi = d_star * 0.7, d_star * 1.3
    lnz_full = _reference_lnZ(D_MIN, D_MAX, log_like)
    lnz_box = _reference_lnZ(lo, hi, log_like)
    assert lnz_full - lnz_box > 0.5


###
### 4. Through the real MCSampler construction path
###
# distance_sampler_kwargs() output is fed to add_parameter() verbatim by the ILE
# driver, so building a real sampler from it is the construction path.  These assert
# on the sampler's ACTUAL bounds: a --limit-distance that parsed and then did nothing
# (this codebase's documented failure mode) fails here.

@pytest.mark.parametrize('module_name', ['RIFT.integrators.mcsampler',
                                         'RIFT.integrators.mcsamplerGPU',
                                         'RIFT.integrators.mcsamplerAdaptiveVolume'])
def test_sampler_bounds_are_the_narrowed_ones(module_name):
    mod = pytest.importorskip(module_name)
    kw = distance_sampler_kwargs(mod, BOX, (D_MIN, D_MAX), d_prior='Euclidean')
    s = mod.MCSampler()
    s.add_parameter("distance", **kw)
    assert s.llim["distance"] == pytest.approx(BOX[0])
    assert s.rlim["distance"] == pytest.approx(BOX[1])
    # ... and NOT the full range: the disconnected-flag failure mode
    assert s.rlim["distance"] < D_MAX
    # the prior the sampler will weight with is still the full-range one
    d = np.array([2000.0])
    assert float(np.asarray(s.prior_pdf["distance"](d), dtype=float)[0]) == pytest.approx(
        2000.0 ** 2 / (D_MAX ** 3 / 3. - D_MIN ** 3 / 3.))


def test_sampler_bounds_are_the_full_range_without_the_option():
    kw = distance_sampler_kwargs(mcsampler, (D_MIN, D_MAX), (D_MIN, D_MAX))
    s = mcsampler.MCSampler()
    s.add_parameter("distance", **kw)
    assert s.llim["distance"] == pytest.approx(D_MIN)
    assert s.rlim["distance"] == pytest.approx(D_MAX)


def test_gpu_sampler_draws_inside_the_box():
    """The actual mcsamplerGPU draw path (CPU fallback when cupy is absent): the
    narrowed pdf/cdf_inv must place every draw inside the box, and fill it."""
    mcsamplerGPU = pytest.importorskip('RIFT.integrators.mcsamplerGPU')
    kw = distance_sampler_kwargs(mcsamplerGPU, BOX, (D_MIN, D_MAX), d_prior='Euclidean')
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("distance", **kw)
    rv = s.draw_simplified(4000, "distance")[-1]
    drawn = np.asarray(mcsamplerGPU.identity_convert(rv)).reshape(-1)
    assert drawn.min() >= BOX[0] - 1e-9
    assert drawn.max() <= BOX[1] + 1e-9
    assert drawn.max() - drawn.min() > 0.8 * (BOX[1] - BOX[0])
    # the whole point: the draws are NOT spread over the physical prior range
    assert drawn.max() < 0.5 * D_MAX


def test_end_to_end_MCSampler_integral_is_unchanged_by_the_box():
    """Same claim as the acceptance test, but paid for with real Monte Carlo:
    mcsampler.MCSampler weights each draw by prior_pdf/pdf, so this exercises the
    sampling density as well as the prior.  Tolerance is set by MC noise of the
    FULL-range run (the box is ~24% of the prior mass here), not by the physics."""
    np.random.seed(20260901)
    ref = _reference_integral(D_MIN, D_MAX)

    def _run(rng_seed, sampling_range):
        np.random.seed(rng_seed)
        kw = distance_sampler_kwargs(mcsampler, sampling_range, (D_MIN, D_MAX))
        s = mcsampler.MCSampler()
        s.add_parameter("distance", **kw)
        res = s.integrate(lambda distance: _like(distance), "distance",
                          nmax=400000, n=4000, no_protect_names=True, verbose=False)
        return float(res[0])

    z_full = _run(1234, (D_MIN, D_MAX))
    z_box = _run(1234, BOX)
    assert z_full == pytest.approx(ref, rel=0.05)
    assert z_box == pytest.approx(ref, rel=0.01)
    assert abs(np.log(z_box) - np.log(z_full)) < 0.05


###
### 5. Driver wiring
###
# The library helper can be perfect and the driver still normalize over the narrowed
# range: that is one edit away, in a file with no cheap end-to-end test (the distance
# block sits ~1300 lines in, after data loading).  So: assert the driver hands the
# helper the UNNARROWED range, and that the old conflated normalization is gone.

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_passes_the_unnarrowed_range_as_the_prior_normalization():
    with open(_ILE) as f:
        src = f.read()
    assert '--limit-distance' in src
    # captured BEFORE the narrowing, and handed to the helper as prior_range
    assert 'dist_prior_range = (param_limits["distance"][0], param_limits["distance"][1])' in src
    assert 'distance_sampler_kwargs(\n                        mcsampler, param_limits["distance"], dist_prior_range,' in src
    # the conflated normalization must not come back
    assert 'param_limits["distance"][1]**3/3.' not in src


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_refuses_the_incompatible_distance_modes():
    with open(_ILE) as f:
        src = f.read()
    for needle in ('--limit-distance is not compatible with --distance-marginalization',
                   '--limit-distance is not compatible with --d-prior-redshift',
                   '--limit-distance is not compatible with --internal-reparam-dl-incl'):
        assert needle in src


def _run_ile_narrowing_block(limit_distance, **optkw):
    """Execute the DRIVER'S OWN narrowing block, verbatim from the file, against a
    stub `opts`/`param_limits`.

    The block sits ~1375 lines into a monolithic script, after data loading, so a
    subprocess test of it would need frames and PSDs.  Exec'ing the real source text
    is the next best thing: it is the code that ships, not a paraphrase, so an edit
    that reconnects the prior normalization to the narrowed range fails here.
    """
    with open(_ILE) as f:
        src = f.read()
    start = src.index('dist_prior_range = (param_limits["distance"][0]')
    end = src.index('#\n# Parameter integral sampling strategy', start)
    block = src[start:end]

    class _O(object):
        distance_marginalization = False
        d_prior_redshift = False
        internal_reparam_dl_incl = False
        pin_distance_to_sim = False
        limit_distance = None
    o = _O()
    o.limit_distance = limit_distance
    for k, v in optkw.items():
        setattr(o, k, v)
    ns = {'opts': o, 'param_limits': {"distance": (1.0, 10000.0)},
          'distance_limit_range': distance_limit_range, 'print': lambda *a, **k: None}
    exec(compile(block, _ILE, 'exec'), ns)                      # noqa: S102
    return ns


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_narrowing_block_narrows_sampling_and_keeps_the_prior_range():
    ns = _run_ile_narrowing_block('800,3200')
    assert ns['param_limits']["distance"] == (800.0, 3200.0)     # SAMPLING narrowed
    assert ns['dist_prior_range'] == (1.0, 10000.0)              # PRIOR untouched
    assert ns['limit_distance_active'] is True
    # and the kwargs the driver then builds from those two ranges
    kw = distance_sampler_kwargs(mcsampler, ns['param_limits']["distance"],
                                 ns['dist_prior_range'])
    assert (kw['left_limit'], kw['right_limit']) == (800.0, 3200.0)
    assert float(np.asarray(kw['prior_pdf'](np.array([2000.0])))[0]) == pytest.approx(
        2000.0 ** 2 / (10000.0 ** 3 / 3. - 1.0 ** 3 / 3.))


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_narrowing_block_is_a_no_op_without_the_option():
    ns = _run_ile_narrowing_block(None)
    assert ns['param_limits']["distance"] == (1.0, 10000.0)
    assert ns['dist_prior_range'] == (1.0, 10000.0)
    assert ns['limit_distance_active'] is False


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('flag', ['distance_marginalization', 'd_prior_redshift',
                                  'internal_reparam_dl_incl', 'pin_distance_to_sim'])
def test_ile_narrowing_block_refuses_the_incompatible_modes(flag):
    """These four have no d_L sampler to narrow.  Refusing beats reinterpreting.

    pin_distance_to_sim was MISSED by the first three and added after an
    adversarial audit that enumerated every distance-touching option in the driver
    rather than trusting the declared list: it pins distance to the injection value
    inside analyze_event, so the box was accepted and silently did nothing.  That
    is the same class as --distance-marginalization, and the same failure mode this
    option's whole design is meant to avoid."""
    with pytest.raises(SystemExit):
        _run_ile_narrowing_block('800,3200', **{flag: True})
    # ... and they are NOT refused when the option is absent
    _run_ile_narrowing_block(None, **{flag: True})


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('bad', ['3200,800', '0.5,3200', '800,20000', 'a,b'])
def test_ile_narrowing_block_exits_loudly_on_a_bad_range(bad):
    with pytest.raises(SystemExit):
        _run_ile_narrowing_block(bad)


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_advertises_the_option():
    """--help goes through optparse, so this catches an option that was written into
    the source but never registered."""
    out = subprocess.run([sys.executable, _ILE, '--help'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         timeout=900).stdout.decode('utf-8', 'replace')
    assert '--limit-distance' in out
