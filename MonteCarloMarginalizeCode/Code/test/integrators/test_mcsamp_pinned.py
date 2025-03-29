#!/usr/bin/env python
import sys
import os
import functools
import itertools
import bisect

import healpy

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap

import numpy
import scipy.integrate
import scipy.special

from lalinference.bayestar import fits as bfits
from lalinference.bayestar import plot as bplot

import RIFT.integrators.mcsampler as mcsampler
from statutils import cumvar


__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

#
# set up bounds on parameters
#

# Polarization angle
#psi_min, psi_max = 0, 2*numpy.pi
psi_min, psi_max = -2*numpy.pi, 2*numpy.pi
psi_val, psi_width = numpy.pi/2, numpy.pi/6
# RA and dec
ra_min, ra_max = 0, 2*numpy.pi
#ra_min, ra_max = -2*numpy.pi, 2*numpy.pi
ra_val, ra_width = numpy.pi/4, 5*numpy.pi/180
dec_min, dec_max = -numpy.pi/2, numpy.pi/2
dec_val, dec_width = -numpy.pi/4, 5*numpy.pi/180
# Inclination angle
inc_min, inc_max = -numpy.pi/2, numpy.pi/2
inc_val, inc_width = -numpy.pi/3, 10*numpy.pi/180
# orbital phi
phi_min, phi_max = 0, 2*numpy.pi
phi_val, phi_width = numpy.pi/5, 10*numpy.pi/180
# distance
dist_min, dist_max = 0.0, 100.0
dist_val, dist_width = 25.0, 25.0


samp = mcsampler.MCSampler()

#
# Test 6: sampling, pin parameters
#
# Uniform sampling, auto-cdf inverse
samp.add_parameter("psi", functools.partial(mcsampler.gauss_samp, psi_val, 2*psi_width), None, psi_min, psi_max, prior_pdf=lambda x: 1)
samp.add_parameter("ra", functools.partial(mcsampler.uniform_samp_vector, ra_min, ra_max), None, ra_min, ra_max, prior_pdf=lambda x: 1)
samp.add_parameter("dec", functools.partial(mcsampler.uniform_samp_vector, dec_min, dec_max), None, dec_min, dec_max, prior_pdf=lambda x: 1)
samp.add_parameter("phi", functools.partial(mcsampler.uniform_samp_vector, phi_min, phi_max), None, phi_min, phi_max, prior_pdf=lambda x: 1)
samp.add_parameter("inc", functools.partial(mcsampler.gauss_samp, inc_val, 2*inc_width), None, inc_min, inc_max, prior_pdf=lambda x: 1)
samp.add_parameter("dist", functools.partial(mcsampler.uniform_samp_vector, dist_min, dist_max), None, dist_min, dist_max, prior_pdf=lambda x: 1)

#
# Full 6-D test
#

a, b, c, d, e, f = 2*psi_width**2, 2*ra_width**2, 2*dec_width**2, 2*inc_width**2, 2*phi_width**2, 2*dist_width**2
norm = 1.0/numpy.sqrt((numpy.pi)**len(samp.params)*a*b*c*d*e*f)
numpy.seterr(invalid="raise")
def integrand(p, r, dec, ph, i, di):
	exponent = -(p-psi_val)**2/a-(r-ra_val)**2/b-(dec-dec_val)**2/c-(i-inc_val)**2/d-(ph-phi_val)**2/e-(di-dist_val)**2/f
	return norm * numpy.exp(exponent)

res, var = samp.integrate(integrand, "psi", "phi", "inc", "dist", neff=100, nmax=int(1e6), ra=ra_val, dec=dec_val)
print("Integral value: %f, stddev %f" % (res, numpy.sqrt(var)))
