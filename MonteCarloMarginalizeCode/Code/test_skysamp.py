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
# Read FITS data
#
smap, smap_meta = bfits.read_sky_map(sys.argv[1])

skysampler = mcsampler.HealPixSampler(smap)

#def integrand(dec, ra, test):
def integrand(dec, ra):
    #return numpy.ones(ra.shape)/(2.0*numpy.pi**2)
    #return numpy.sin(dec.astype(numpy.float64))/(4*numpy.pi)
    # Function... integrate thyself...
    return skysampler.pseudo_pdf(dec.astype(numpy.float64), ra.astype(numpy.float64))


#
# Check sample distribution plots
#
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

integrator = mcsampler.MCSampler()

#
# 2-D "sky sampler" uses a custom sampling class wrapped around a fits file
# skymap
#
#integrator.add_parameter(params=("dec", "ra"), pdf=ss_test.pseudo_pdf, cdf_inv=ss_test.pseudo_cdf_inverse, prior_pdf=lambda a,b: numpy.ones(a.shape), left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))
integrator.add_parameter(params=("dec", "ra"), pdf=skysampler.pseudo_pdf, cdf_inv=skysampler.pseudo_cdf_inverse, prior_pdf=lambda a,b: numpy.ones(a.shape), left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))

#
# "test" dimension is just here to ensure we don't break the mixed 1-D and N-D 
# pdf cases
#
#test_samp = functools.partial(mcsampler.uniform_samp_vector, 0, 1)
#integrator.add_parameter(params="test", pdf=test_samp, cdf_inv=None, prior_pdf=lambda t: numpy.ones(t.shape), left_limit=0, right_limit=1)

#print integrator.integrate(integrand, (("ra", "dec"), "test"), verbose=True, nmax=10000)
print integrator.integrate(integrand, (("ra", "dec"),), verbose=True, nmax=30000)
