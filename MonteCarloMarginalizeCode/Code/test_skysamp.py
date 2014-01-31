#!/usr/bin/env python
import sys
import os
import functools
import itertools

import healpy

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap

import numpy

from lalinference.bayestar import fits as bfits
from lalinference.bayestar import plot as bplot

import mcsampler


__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"



#
# Read FITS data
#
smap, smap_meta = bfits.read_sky_map(sys.argv[1])

#
# Integrand: three dimensional cube
#
def integrand_2d(dec, ra, test):
    return numpy.ones(ra.shape)/(2.0*numpy.pi**2)**2

skysampler = mcsampler.HealPixSampler(smap)

#dec, ra = zip(skysampler.pseudo_cdf_inverse(ndraws=1000))
#hist, x, y = numpy.histogram2d(ra[0], dec[0], bins=40)
#pyplot.imshow(hist, extent=(x[0], x[-1], y[0], y[-1]), interpolation='nearest')
#pyplot.savefig("2d_hist.png")

integrator = mcsampler.MCSampler()

#
# Not a Bayesian calculation... prior shouldn't need to be defined
#
def blah_blah(dec, ra):
    return 1.0/2.0/numpy.pi**2
vect_blah_blah = numpy.vectorize(blah_blah)

def blah_blah_cdf_inv(dec, ra):
    np = healpy.nside2npix(32)
    ss = mcsampler.HealPixSampler(numpy.ones(np)/float(np))
    return ss.pseudo_cdf_inverse(ndraws=len(dec))
def blah_blah_blah(test):
    return 1.0

#
# 2-D "sky sampler" uses a custom sampling class wrapped around a fits file
# skymap
#
integrator.add_parameter(params=("dec", "ra"), pdf=skysampler.pseudo_pdf, cdf_inv=skysampler.pseudo_cdf_inverse, prior_pdf=lambda a,b: numpy.ones(a.shape), left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))
#integrator.add_parameter(params=("dec", "ra"), pdf=vect_blah_blah, cdf_inv=blah_blah_cdf_inv, prior_pdf=lambda a,b: numpy.ones(a.shape), left_limit=(0, 0), right_limit=(numpy.pi, 2*numpy.pi))

#
# "test" dimension is just here to ensure we don't break the mixed 1-D and N-D 
# pdf cases
#
#test_samp = functools.partial(mcsampler.uniform_samp_vector, 0, 2*numpy.pi)
test_samp = functools.partial(mcsampler.uniform_samp_vector, 0, 1)
vect_blah_blah_blah = numpy.vectorize(blah_blah_blah)
integrator.add_parameter(params="test", pdf=test_samp, cdf_inv=None, prior_pdf=lambda t: numpy.ones(t.shape), left_limit=0, right_limit=1)

print integrator.integrate(integrand_2d, (("ra", "dec"), "test"), verbose=True, nmax=100000)
