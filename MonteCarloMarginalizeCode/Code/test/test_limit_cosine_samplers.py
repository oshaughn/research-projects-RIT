#!/usr/bin/env python
"""
Regression tests for the extrinsic "zoom box" options
(--limit-declination / --limit-inclination / --limit-right-ascension / --limit-psi)
under the COSINE samplers (--declination-cosine-sampler / --inclination-cosine-sampler).

Background (the bug these tests lock down): the cosine branches of
bin/integrate_likelihood_extrinsic_batchmode used to hardcode left_limit=-1,
right_limit=1 and never consulted param_limits, so --limit-declination and
--limit-inclination were SILENTLY IGNORED whenever the cosine samplers were on
(which is the production default).  No error, no warning, no narrowing.

Coordinate conventions, read off the likelihood closures in that script
(`dec = pi/2 - arccos(z)`, `iota = arccos(z)`):

  declination:  sampled variable z = sin(dec),  sin INCREASING on [-pi/2,pi/2]
                => [lo,hi] -> [sin(lo), sin(hi)]     (order preserved)
  inclination:  sampled variable z = cos(iota), cos DECREASING on [0,pi]
                => [lo,hi] -> [cos(hi), cos(lo)]     (order SWAPS)

The second one is the easy thing to get backwards, so it gets its own test.
"""

import os

import numpy as np
import pytest

import RIFT.integrators.mcsampler as mcsampler
from RIFT.integrators.mcsampler import (
    clip_angle_limits,
    cosine_sampler_limits,
    ret_cos_samp_cdf_inv_vector,
    ret_cos_samp_vector,
    ret_dec_samp_cdf_inv_vector,
    ret_dec_samp_vector,
)

# The conversions applied inside the ILE likelihood closures, verbatim
# (the .astype mirrors the numpy.copy(...).astype(numpy.float64) those closures do,
# because mcsampler hands back object arrays).
_dec_from_z = lambda z: np.pi / 2 - np.arccos(np.asarray(z).astype(np.float64))
_incl_from_z = lambda z: np.arccos(np.asarray(z).astype(np.float64))


###
### 1. Coordinate transform
###

def test_declination_limits_map_to_sin_and_preserve_order():
    lo, hi = -0.62, -0.41
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'declination')
    assert z_lo == pytest.approx(np.sin(lo))
    assert z_hi == pytest.approx(np.sin(hi))
    assert z_lo < z_hi
    # round trip through the conversion the likelihood actually applies
    assert _dec_from_z(z_lo) == pytest.approx(lo)
    assert _dec_from_z(z_hi) == pytest.approx(hi)


def test_inclination_limits_map_to_cos_and_SWAP_order():
    lo, hi = 0.30, 1.20
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'inclination')
    # this is the assertion that fails if someone writes [cos(lo), cos(hi)]
    assert z_lo == pytest.approx(np.cos(hi))
    assert z_hi == pytest.approx(np.cos(lo))
    assert z_lo < z_hi
    # and the round trip: the LOWER cosine limit is the UPPER angle
    assert _incl_from_z(z_lo) == pytest.approx(hi)
    assert _incl_from_z(z_hi) == pytest.approx(lo)
    # explicit guard against the naive (unswapped) answer
    assert (z_lo, z_hi) != pytest.approx((np.cos(lo), np.cos(hi)))


def test_inclination_swap_would_produce_empty_or_inverted_interval():
    """A [cos(lo), cos(hi)] implementation is not merely mislabeled: it is inverted."""
    lo, hi = 0.30, 1.20
    naive_lo, naive_hi = np.cos(lo), np.cos(hi)
    assert naive_lo > naive_hi          # inverted -> would silently give a negative volume
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'inclination')
    assert z_hi - z_lo == pytest.approx(naive_lo - naive_hi)   # same width, correct sign


def test_full_range_is_a_no_op():
    assert cosine_sampler_limits(-np.pi / 2, np.pi / 2, 'declination') == pytest.approx((-1.0, 1.0))
    assert cosine_sampler_limits(0.0, np.pi, 'inclination') == pytest.approx((-1.0, 1.0))


def test_limits_are_clipped_to_the_physical_domain():
    assert clip_angle_limits(-10.0, 0.1, 'declination') == pytest.approx((-np.pi / 2, 0.1))
    assert clip_angle_limits(0.1, 10.0, 'inclination') == pytest.approx((0.1, np.pi))
    z_lo, z_hi = cosine_sampler_limits(-10.0, 10.0, 'declination')
    assert (z_lo, z_hi) == pytest.approx((-1.0, 1.0))


@pytest.mark.parametrize('kind', ['declination', 'inclination'])
def test_empty_or_inverted_range_raises(kind):
    with pytest.raises(ValueError):
        cosine_sampler_limits(0.5, 0.5, kind)          # empty
    with pytest.raises(ValueError):
        cosine_sampler_limits(0.9, 0.2, kind)          # inverted
    with pytest.raises(ValueError):
        cosine_sampler_limits(np.nan, 0.2, kind)       # non-finite


def test_range_outside_physical_domain_raises():
    with pytest.raises(ValueError):
        cosine_sampler_limits(2.0, 3.0, 'declination')   # entirely north of the pole
    with pytest.raises(ValueError):
        cosine_sampler_limits(-2.0, -1.0, 'inclination')  # entirely below iota=0


def test_unknown_angle_raises():
    with pytest.raises(ValueError):
        cosine_sampler_limits(0.1, 0.2, 'right_ascension')


###
### 2. Support: the box actually restricts, in both samplers
###

def test_declination_box_restricts_support_in_both_samplers():
    lo, hi = -0.62, -0.41
    p = np.linspace(0.0, 1.0, 4001)

    # plain (angle) sampler, truncated
    dec_plain = ret_dec_samp_cdf_inv_vector(lo, hi)(p)
    assert dec_plain.min() == pytest.approx(lo)
    assert dec_plain.max() == pytest.approx(hi)

    # cosine sampler: uniform in z over the transformed box, then converted back
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'declination')
    dec_cos = _dec_from_z(z_lo + p * (z_hi - z_lo))
    assert dec_cos.min() == pytest.approx(lo)
    assert dec_cos.max() == pytest.approx(hi)

    # ... and the two draws are the SAME map (both are uniform-in-sin(dec) on the box)
    assert np.allclose(np.sort(dec_plain), np.sort(dec_cos))


def test_inclination_box_restricts_support_in_both_samplers():
    lo, hi = 0.30, 1.20
    p = np.linspace(0.0, 1.0, 4001)

    incl_plain = ret_cos_samp_cdf_inv_vector(lo, hi)(p)
    assert incl_plain.min() == pytest.approx(lo)
    assert incl_plain.max() == pytest.approx(hi)

    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'inclination')
    incl_cos = _incl_from_z(z_lo + p * (z_hi - z_lo))
    assert incl_cos.min() == pytest.approx(lo)
    assert incl_cos.max() == pytest.approx(hi)

    assert np.allclose(np.sort(incl_plain), np.sort(incl_cos))


def test_truncated_samplers_reduce_to_the_untruncated_ones():
    """Full-range truncated samplers must reproduce the legacy distributions."""
    p = np.linspace(1e-9, 1 - 1e-9, 501)
    dec_new = np.sort(ret_dec_samp_cdf_inv_vector(-np.pi / 2, np.pi / 2)(p))
    dec_old = np.sort(mcsampler.dec_samp_cdf_inv_vector(p))
    assert np.allclose(dec_new, dec_old, atol=1e-10)

    incl_new = np.sort(ret_cos_samp_cdf_inv_vector(0.0, np.pi)(p))
    incl_old = np.sort(mcsampler.cos_samp_cdf_inv_vector(p))
    assert np.allclose(incl_new, incl_old, atol=1e-10)

    x = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 257)
    assert np.allclose(ret_dec_samp_vector(-np.pi / 2, np.pi / 2)(x),
                       mcsampler.dec_samp_vector(x))
    y = np.linspace(1e-6, np.pi - 1e-6, 257)
    assert np.allclose(ret_cos_samp_vector(0.0, np.pi)(y), mcsampler.cos_samp_vector(y))


###
### 3. Normalization: identical prior mass / lnZ in both samplers
###
# mcsampler / mcsamplerGPU weight each draw by prior_pdf(x)/pdf(x); the expectation of
# that weight over the sampling pdf is the prior MASS inside the box.  The two branches
# must agree, otherwise the same physical box would give different lnZ.

def _weight_plain_dec(lo, hi, dec):
    pdf = ret_dec_samp_vector(lo, hi)(dec)
    prior = mcsampler.uniform_samp_dec(dec)          # 0.5*cos(dec), NOT renormalized
    return prior / pdf


def _weight_cosine(z_lo, z_hi, z):
    pdf = mcsampler.ret_uniform_samp_vector_alt(z_lo, z_hi)(z)
    prior = mcsampler.ret_uniform_samp_vector_alt(-1.0, 1.0)(z)   # constant 1/2 in z
    return prior / pdf


def test_declination_box_same_prior_mass_in_both_samplers():
    lo, hi = -0.62, -0.41
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'declination')
    expected = 0.5 * (np.sin(hi) - np.sin(lo))     # isotropic prior mass in the box

    dec = np.linspace(lo + 1e-9, hi - 1e-9, 2001)
    w_plain = _weight_plain_dec(lo, hi, dec)
    assert np.allclose(w_plain, expected)          # constant weight

    z = np.linspace(z_lo + 1e-12, z_hi - 1e-12, 2001)
    w_cos = _weight_cosine(z_lo, z_hi, z)
    assert np.allclose(w_cos, expected)


def test_inclination_box_same_prior_mass_in_both_samplers():
    lo, hi = 0.30, 1.20
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'inclination')
    expected = 0.5 * (np.cos(lo) - np.cos(hi))

    incl = np.linspace(lo + 1e-9, hi - 1e-9, 2001)
    w_plain = mcsampler.uniform_samp_theta(incl) / ret_cos_samp_vector(lo, hi)(incl)
    assert np.allclose(w_plain, expected)

    z = np.linspace(z_lo + 1e-12, z_hi - 1e-12, 2001)
    assert np.allclose(_weight_cosine(z_lo, z_hi, z), expected)


def test_prior_mass_scales_like_the_box_not_the_full_prior():
    """Sanity: narrowing must actually cost prior mass (that is the whole point)."""
    full = 0.5 * (np.sin(np.pi / 2) - np.sin(-np.pi / 2))
    lo, hi = -0.62, -0.41
    narrow = 0.5 * (np.sin(hi) - np.sin(lo))
    assert narrow < full
    assert full / narrow == pytest.approx(1.0 / (0.5 * (np.sin(hi) - np.sin(lo))))


###
### 4. AV-style estimator equivalence (the production sampler)
###
# mcsamplerAdaptiveVolume draws uniformly in [llim,rlim] and multiplies by the sampling
# volume V_s = prod(rlim-llim), weighting by prior_pdf.  Same box => same answer.

def _av_prior_integral(llim, rlim, prior_pdf, n=200001):
    x = np.linspace(llim, rlim, n)
    return (rlim - llim) * np.mean(prior_pdf(x))


def test_AV_style_prior_integral_matches_between_samplers_dec():
    lo, hi = -0.62, -0.41
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'declination')
    plain = _av_prior_integral(lo, hi, mcsampler.uniform_samp_dec)
    cosine = _av_prior_integral(z_lo, z_hi, lambda x: np.full_like(x, 0.5))
    assert plain == pytest.approx(cosine, rel=1e-6)
    assert plain == pytest.approx(0.5 * (np.sin(hi) - np.sin(lo)), rel=1e-6)


def test_AV_style_prior_integral_matches_between_samplers_incl():
    lo, hi = 0.30, 1.20
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'inclination')
    plain = _av_prior_integral(lo, hi, mcsampler.uniform_samp_theta)
    cosine = _av_prior_integral(z_lo, z_hi, lambda x: np.full_like(x, 0.5))
    assert plain == pytest.approx(cosine, rel=1e-6)
    assert plain == pytest.approx(0.5 * (np.cos(lo) - np.cos(hi)), rel=1e-6)


def test_AV_style_posterior_shape_matches_between_samplers_dec():
    """Posterior *shape* in declination must be identical (isotropic prior preserved)."""
    lo, hi = -0.62, -0.41
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'declination')
    dec_grid = np.linspace(lo, hi, 4001)

    # plain branch: uniform in dec, weight 0.5*cos(dec)
    q_plain = mcsampler.uniform_samp_dec(dec_grid)
    q_plain = q_plain / np.trapz(q_plain, dec_grid)

    # cosine branch: uniform in z=sin(dec), weight 1/2 -> push forward to dec
    q_cos = 0.5 * np.cos(dec_grid)      # Jacobian dz/ddec = cos(dec)
    q_cos = q_cos / np.trapz(q_cos, dec_grid)

    assert np.allclose(q_plain, q_cos)
    assert z_lo < z_hi


###
### 5. End-to-end through MCSampler: same box => same integral
###

def _integrate_1d(name, pdf, cdf_inv, llim, rlim, prior_pdf, fn, nmax=20000):
    s = mcsampler.MCSampler()
    s.add_parameter(name, pdf=pdf, cdf_inv=cdf_inv, left_limit=llim, right_limit=rlim,
                    prior_pdf=prior_pdf)
    res = s.integrate(fn, name, nmax=nmax, n=1000, no_protect_names=True, verbose=False)
    return res[0]


def test_end_to_end_declination_box_gives_same_integral():
    lo, hi = -0.62, -0.41
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'declination')
    expected = 0.5 * (np.sin(hi) - np.sin(lo))

    # integrand == 1 -> the integral IS the prior mass in the box
    unit = lambda declination: np.ones(np.shape(declination))

    plain = _integrate_1d('declination',
                          ret_dec_samp_vector(lo, hi), ret_dec_samp_cdf_inv_vector(lo, hi),
                          lo, hi, mcsampler.uniform_samp_dec, unit)
    cosine = _integrate_1d('declination',
                           mcsampler.ret_uniform_samp_vector_alt(z_lo, z_hi),
                           lambda x, _a=z_lo, _b=z_hi: _a + x * (_b - _a),
                           z_lo, z_hi, mcsampler.ret_uniform_samp_vector_alt(-1.0, 1.0), unit)

    assert float(plain) == pytest.approx(expected, rel=1e-6)
    assert float(cosine) == pytest.approx(expected, rel=1e-6)
    assert float(plain) == pytest.approx(float(cosine), rel=1e-6)


def test_end_to_end_inclination_box_gives_same_integral():
    lo, hi = 0.30, 1.20
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'inclination')
    expected = 0.5 * (np.cos(lo) - np.cos(hi))

    unit = lambda inclination: np.ones(np.shape(inclination))

    plain = _integrate_1d('inclination',
                          ret_cos_samp_vector(lo, hi), ret_cos_samp_cdf_inv_vector(lo, hi),
                          lo, hi, mcsampler.uniform_samp_theta, unit)
    cosine = _integrate_1d('inclination',
                           mcsampler.ret_uniform_samp_vector_alt(z_lo, z_hi),
                           lambda x, _a=z_lo, _b=z_hi: _a + x * (_b - _a),
                           z_lo, z_hi, mcsampler.ret_uniform_samp_vector_alt(-1.0, 1.0), unit)

    assert float(plain) == pytest.approx(expected, rel=1e-6)
    assert float(cosine) == pytest.approx(expected, rel=1e-6)


def test_end_to_end_declination_box_same_integral_for_a_peaked_likelihood():
    """Non-constant integrand: the cosine branch must convert z -> dec exactly as ILE does."""
    lo, hi = -0.62, -0.41
    z_lo, z_hi = cosine_sampler_limits(lo, hi, 'declination')
    mu, sig = -0.50, 0.03
    like_dec = lambda d: np.exp(-0.5 * ((np.asarray(d, dtype=float) - mu) / sig) ** 2)

    plain = _integrate_1d('declination',
                          ret_dec_samp_vector(lo, hi), ret_dec_samp_cdf_inv_vector(lo, hi),
                          lo, hi, mcsampler.uniform_samp_dec,
                          lambda declination: like_dec(declination), nmax=200000)
    cosine = _integrate_1d('declination',
                           mcsampler.ret_uniform_samp_vector_alt(z_lo, z_hi),
                           lambda x, _a=z_lo, _b=z_hi: _a + x * (_b - _a),
                           z_lo, z_hi, mcsampler.ret_uniform_samp_vector_alt(-1.0, 1.0),
                           lambda declination: like_dec(_dec_from_z(declination)), nmax=200000)

    # analytic reference: int 0.5*cos(dec) * L(dec) ddec over the box
    grid = np.linspace(lo, hi, 200001)
    ref = np.trapz(0.5 * np.cos(grid) * like_dec(grid), grid)

    assert float(plain) == pytest.approx(ref, rel=2e-2)
    assert float(cosine) == pytest.approx(ref, rel=2e-2)


###
### 6. Wiring: the bin script must not reintroduce the hardcoded [-1,1] range
###

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_cosine_branches_consult_param_limits():
    with open(_ILE) as f:
        src = f.read()
    for angle in ('declination', 'inclination'):
        needle = 'cosine_sampler_limits(param_limits["{}"][0], param_limits["{}"][1], \'{}\')'.format(angle, angle, angle)
        assert needle in src, \
            "cosine {} branch no longer transforms param_limits -- --limit-{} would be silently ignored".format(angle, angle)
    # the old hardcoded literals must be gone from the sampler setup
    assert 'left_limit = -1,' not in src
    assert 'right_limit = 1,' not in src
