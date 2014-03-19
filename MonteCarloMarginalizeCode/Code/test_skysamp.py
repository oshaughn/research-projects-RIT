#!/usr/bin/env python
import sys
import os
import functools
import itertools

import healpy

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

import numpy

from lalinference.bayestar import fits as bfits
from lalinference.bayestar import plot as bplot

import mcsampler


__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

#
# 2-D "sky sampler" uses a custom sampling class wrapped around a fits file
# skymap
#

#
# Read FITS data
#
smap, smap_meta = bfits.read_sky_map(sys.argv[1])

#
# Integrating a constant (1)
#
def integrand(dec, ra):
    return numpy.ones(ra.shape)

print "Test 1, prior is isotropic (unnormalized). Should get the normalization factor for the prior (1/len(skymap)) from this test."
smap_isotropic = numpy.ones(len(smap))/len(smap)
skysampler = mcsampler.HealPixSampler(smap_isotropic)

integrator = mcsampler.MCSampler()
integrator.add_parameter(params=("dec", "ra"), pdf=skysampler.pseudo_pdf, cdf_inv=skysampler.pseudo_cdf_inverse, prior_pdf=lambda d, r: 1, left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))

v = integrator.integrate(integrand, (("ra", "dec"),), verbose=True, nmax=100000)
print v[0], len(smap)

print "Test 2, prior is isotropic (normalized). Should get 1.0 for this test"
iso_bstar_prior = numpy.vectorize(lambda d, r: 1.0/len(skysampler.skymap))
integrator = mcsampler.MCSampler()
integrator.add_parameter(params=("dec", "ra"), pdf=skysampler.pseudo_pdf, cdf_inv=skysampler.pseudo_cdf_inverse, prior_pdf=iso_bstar_prior, left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))

v = integrator.integrate(integrand, (("ra", "dec"),), verbose=True, nmax=100000)
print v[0]

print "Test 3, prior is isotropic (normalized). BAYESTAR map is not isotropic, but has support everywhere. Should still get 1.0 for this test. The integral will accuracy increase as the mixing factor becomes larger (and thus more effective in coverage)."
for mixing_factor in [1e-3, 5e-3, 1e-2, 1e-1]:
    print "Mixing factor %g" % mixing_factor
    smap_full_support = (1-mixing_factor)*smap + (mixing_factor)*numpy.ones(len(smap))/len(smap)
    skysampler = mcsampler.HealPixSampler(smap_full_support)
    integrator = mcsampler.MCSampler()
    integrator.add_parameter(params=("dec", "ra"), pdf=skysampler.pseudo_pdf, cdf_inv=skysampler.pseudo_cdf_inverse, prior_pdf=iso_bstar_prior, left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))

    v = integrator.integrate(integrand, (("ra", "dec"),), verbose=True, nmax=100000)
    print v[0]

print "Test 4, prior is isotropic (normalized). BAYESTAR map does not have full support over the entire sky, so the skymap values are (internally) renormalized to account for the missing area (in the ratio of used pixels to total pixels). Answer should still be 1.0"
skysampler = mcsampler.HealPixSampler(smap)
iso_bstar_prior = numpy.vectorize(lambda d, r: 1.0/len(skysampler.skymap))
integrator = mcsampler.MCSampler()
integrator.add_parameter(params=("dec", "ra"), pdf=skysampler.pseudo_pdf, cdf_inv=skysampler.pseudo_cdf_inverse, prior_pdf=iso_bstar_prior, left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))

v = integrator.integrate(integrand, (("ra", "dec"),), verbose=True, nmax=1000000)
print v[0]

print "Test 5, prior is BAYESTAR map, which does not have full support over the sky, but does over itself, thus the integral over itself should still be 1.0. Note that we change the prior, and not the integrand because the assumptions about the discretization of p and p_s appear in a ratio, and this preserves that ratio."

integrator = mcsampler.MCSampler()
integrator.add_parameter(params=("dec", "ra"), pdf=skysampler.pseudo_pdf, cdf_inv=skysampler.pseudo_cdf_inverse, prior_pdf=skysampler.pseudo_pdf, left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))
v = integrator.integrate(integrand, (("ra", "dec"),), verbose=True, nmax=100000)
print v[0]

#
# Check sample distribution plots
#
skysampler = mcsampler.HealPixSampler(smap)
dec, ra = skysampler.pseudo_cdf_inverse(ndraws=10000)
bins, x, y = numpy.histogram2d(dec, ra, bins=32)
pyplot.figure()
pyplot.imshow(bins, extent=(y[0], y[-1], x[0], x[-1]), interpolation="nearest")
pyplot.xlabel("Right Ascension")
pyplot.ylabel("Declination")
pyplot.colorbar()
pyplot.savefig("2d_ss_hist.png")

pyplot.figure()
pyplot.subplot(211)
pyplot.hist(dec, bins=20)
pyplot.xlabel("Declination")
pyplot.subplot(212)
pyplot.hist(ra, bins=20)
pyplot.xlabel("Right Ascension")
pyplot.savefig("1d_ss_hists.png")

np = healpy.nside2npix(32)
ss_test = mcsampler.HealPixSampler(numpy.ones(np)/float(np))

#
# Processing time tests
#
import timeit
setup = """
from lalinference.bayestar import fits as bfits
import mcsampler
smap, smap_meta = bfits.read_sky_map(sys.argv[1])
skysampler = mcsampler.HealPixSampler(smap)
"""
ncalls = 1000
# FIXME: Disabled for now
#print "Checking time for %d cdf_inv calls for 1000 pts each" % ncalls
#res = timeit.Timer("skysampler.pseudo_cdf_inverse(ndraws=1000)", setup=setup).repeat(1, number=ncalls)
#print "min val %f" % min(res)

