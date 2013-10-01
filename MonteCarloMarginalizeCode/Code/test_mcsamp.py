#!/usr/bin/env python
import sys
import functools
import itertools

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap

import numpy
import scipy.integrate
import scipy.special

from mcsampler import MCSampler



__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

#
# Plotting utilities
#

def plot_integrand(fcn, x1, x2):
	"""
	Plot integrand (fcn) from x1 to x2
	"""
	pyplot.figure()
	x_i = numpy.linspace(x1, x2, 1000)
	pyplot.plot(x_i, fcn(x_i))
	pyplot.savefig("integrand.pdf")


def plot_pdf(samp):
	"""
	Plot PDFs of sampling distributions
	"""
	pyplot.figure()
	for param in samp.params:
		x_i = numpy.array([numpy.linspace(samp.llim[param], samp.rlim[param], 100)])
		pyplot.plot(x_i[0], map(samp.pdf[param], x_i)[0], '-', label=param)

	pyplot.grid()
	pyplot.legend()
	pyplot.savefig("pdf.pdf")

def plot_cdf_inv(samp):
	"""
	Plot inverse CDFs
	"""

	pyplot.figure()
	x_i = numpy.linspace(0, 1, 1000)
	for param, cdfinv in samp.cdf_inv.iteritems():
		pyplot.plot(x_i, map(cdfinv, x_i), '-', label=param)

	pyplot.grid()
	pyplot.legend()
	pyplot.savefig("cdf_inv.pdf")


def plot_one_d_hist(samp):
	"""
	Plot one-d histograms of all parameters
	"""
	fig = pyplot.figure(figsize=(10,4))
	samples = samp._rvs
	np = len(samp.params)
	i=1
	for p in samp.params:
		subp = pyplot.subplot(1, np, i)
		i += 1
		pyplot.hist(samples[p], bins=20)
		pyplot.grid()
		pyplot.xlabel(p)
	pyplot.savefig("samples_1d.pdf")

def plot_two_d_hist(samp):
	"""
	Plot two-d histograms of all parameters
	"""
	samples = samp._rvs
	fig = pyplot.figure(figsize=(10,10))
	np = len(samp.params)-1
	i, j = 1, 1
	label_next = True
	for (p1, p2) in itertools.combinations(samp.params, 2):
		subp = pyplot.subplot(np, np, i)
		s1, s2 = samples[p1], samples[p2]
		hist, yedge, xedge = numpy.histogram2d(s2, s1, bins=(100, 100))
	
		pyplot.imshow(hist.T, extent=(yedge[0], yedge[-1], xedge[-1], xedge[0]), interpolation='nearest')
		im = subp.get_images()
		ex = im[0].get_extent()
		subp.set_aspect(abs((ex[1]-ex[0])/(ex[3]-ex[2])))
		pyplot.colorbar()
		if label_next:
			pyplot.ylabel(p1)
			pyplot.xlabel(p2)
			label_next = False
	
		if i % np == 0:
			i += j
			j += 1
			label_next = True
		i += 1
	
	pyplot.savefig("samples_2d.pdf", figsize=(10,3))

	
def plot_ra_dec(samp):
	"""
	Plot the RA and dec distribution in a Mollewide projection.
	"""
	samples = samp._rvs
	fig = pyplot.figure(figsize=(10,10))
	hist, xedge, yedge = numpy.histogram2d(samples["ra"], samples["dec"], bins=(100, 100))
	x, y = numpy.meshgrid(xedge, yedge)
	x *= 180/numpy.pi
	y *= 180/numpy.pi
	m = Basemap(projection='moll', lon_0=0, resolution='c')
	m.contourf(x[:-1,:-1], y[:-1,:-1], hist, 100, cmap=matplotlib.cm.jet, latlon=True)
	pyplot.colorbar()
	m.drawparallels(numpy.arange(-90, 120, 30))
	m.drawmeridians(numpy.arange(0, 420, 60))
	pyplot.savefig("radec_proj.pdf")

if len(sys.argv)<2:
	print "Usage: mcsamp_test npoints psi"
	exit(-1)

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

# TODO: Make a class function
def uniform_samp(a, b, x):
	if type(x) is float:
#                if x>b and x< a:
                return 1/(b-a)
#                else:
#                        return 0
	else:
		return numpy.ones(x.shape[0])/(b-a)

# TODO: Make a class function
def inv_uniform_cdf(a, b, x):
	return (b-a)*x+a

def gauss_samp(mu, std, x):
	return 1.0/numpy.sqrt(2*numpy.pi*std**2)*numpy.exp(-(x-mu)**2/2/std**2)

def inv_gauss_cdf(mu, std, x):
	return mu + std*numpy.sqrt(2) * scipy.special.erfinv(2*x-1)

samp = MCSampler()

def integrand_1d(p):
	return 1.0/numpy.sqrt(2*numpy.pi*psi_width**2)*numpy.exp(-(p-psi_val)**2/2.0/psi_width**2)
plot_integrand(integrand_1d, psi_min, psi_max)

#
# Test 1: What happens when we know the exact right answer
#

samp.add_parameter("psi", functools.partial(gauss_samp, psi_val, psi_width), None, psi_min, psi_max)
integralViaSampler = samp.integrate(integrand_1d, 1, "psi")
integral = scipy.integrate.quad(integrand_1d, psi_min, psi_max)[0]
print "scipy answer vs our answer:",  integral, integralViaSampler
plot_pdf(samp)
plot_cdf_inv(samp)
plot_one_d_hist(samp)
#plot_two_d_hist(samp)
#plot_ra_dec(samp)
samp.clear()

#
# Test 2: Same integrand, uniform sampling
#

samp.add_parameter("psi", functools.partial(uniform_samp, psi_min, psi_max), None, psi_min, psi_max)
print samp.integrate(integrand_1d, int(sys.argv[1]), "psi")
plot_pdf(samp)
plot_cdf_inv(samp)
plot_one_d_hist(samp)
#plot_two_d_hist(samp)
#plot_ra_dec(samp)
samp.clear()

#
# Test 3: Same integrand, gaussian sampling -- various width and offsets
#
widths = [0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 1.01, 1.1, 1.5, 2, 4, 5, 10]
offsets = [-2.0, -1.0, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0]

variances = []
wo = []
for w, o in itertools.product(widths, offsets):
	samp.add_parameter("psi", functools.partial(gauss_samp, psi_val-o*psi_width, w*psi_width), None, psi_min, psi_max)
	print "Width of Gaussian sampler (in units of the width of the integrand: %f" % w
	print "offset of Gaussian sampler (in units of the width of the integrand: %f" % o
	res, var = samp.integrate(integrand_1d, int(sys.argv[1]), "psi")
	variances.append(var)
	wo.append((w, o))
	print res, var

pyplot.figure()
widths, offsets = zip(*wo)
pyplot.grid()
pyplot.xlabel("width in units of $\\sigma_{\\psi}$")
pyplot.ylabel("offset in units of $\\sigma_{\\psi}$")
pyplot.scatter(widths, offsets, c=numpy.log10(numpy.array(variances)))
cbar = pyplot.colorbar()
cbar.set_label("log10 variance")
pyplot.semilogx()
pyplot.savefig("gsamp_variances.pdf")

exit()

#
# Testing convergence: Loop over samples and test sigma relation for desired error
# 
pyplot.figure()
for n in 10**(numpy.arange(1,6)):
	ans = []
	for i in range(1000):
		res, var = samp.integrate(integrand_1d, int(sys.argv[1]), "psi")
		ans.append( (res-1.0)/numpy.sqrt(var) )
	pyplot.hist(ans, bins=20, label="%d" % n)
pyplot.title("$(I-\\bar{I})/\\bar\\sigma$")
pyplot.grid()
pyplot.legend()
pyplot.savefig("integral_hist.pdf")

samp.clear()
exit()

# Uniform sampling, cdf provided
#samp.add_parameter("psi", functools.partial(uniform_samp, psi_min, psi_max), functools.partial(inv_uniform_cdf, psi_min, psi_max), psi_min, psi_max)
#samp.add_parameter("ra", functools.partial(uniform_samp, ra_min, ra_max), functools.partial(inv_uniform_cdf, ra_min, ra_max), ra_min, ra_max)
#samp.add_parameter("dec", functools.partial(uniform_samp, dec_min, dec_max), functools.partial(inv_uniform_cdf, dec_min, dec_max), dec_min, dec_max)

# Uniform sampling, auto-cdf inverse
#samp.add_parameter("psi", functools.partial(uniform_samp, psi_min, psi_max), None, psi_min, psi_max)
samp.add_parameter("psi", functools.partial(gauss_samp, psi_val, 2*psi_width), None, psi_min, psi_max)
samp.add_parameter("ra", functools.partial(uniform_samp, ra_min, ra_max), None, ra_min, ra_max)
samp.add_parameter("dec", functools.partial(uniform_samp, dec_min, dec_max), None, dec_min, dec_max)
samp.add_parameter("phi", functools.partial(uniform_samp, phi_min, phi_max), None, phi_min, phi_max)
#samp.add_parameter("inc", functools.partial(uniform_samp, inc_min, inc_max), None, inc_min, inc_max)
samp.add_parameter("inc", functools.partial(gauss_samp, inc_val, 2*inc_width), None, inc_min, inc_max)
samp.add_parameter("dist", functools.partial(uniform_samp, dist_min, dist_max), None, dist_min, dist_max)

# Gaussian sampling, auto-cdf inverse -- Doesn't work yet
#samp.add_parameter("psi", functools.partial(gauss_samp, 0, (psi_max-psi_min)/3.0), None, psi_min, psi_max)
#samp.add_parameter("ra", functools.partial(gauss_samp, 0, (ra_max-ra_min)/10.0), None, ra_min, ra_max)
#samp.add_parameter("dec", functools.partial(gauss_samp, (dec_max+dec_min)/2, (dec_max-dec_min)/10.0), None, dec_min, dec_max)

#
# Full 6-D test
#

a, b, c, d, e, f = 2*psi_width**2, 2*ra_width**2, 2*dec_width**2, 2*inc_width**2, 2*phi_width**2, 2*dist_width**2
norm = 1.0/numpy.sqrt((numpy.pi)**len(samp.params)*a*b*c*d*e*f)
numpy.seterr(invalid="raise")
from numpy import ma
def integrand(p, r, dec, ph, i, di):
	# FIXME: Don't hardcode this, we need to deal with underflows in a more graceful manner
	# Thoughts: Store the value of the exponent, but send a masked array back.
	# If we find there's a problem with this solution, then, starting from the
	# smallest exponent, deal with each sample independently until we've
	# resolved it
	# NOTE: Really, this will only matter if the number of non zero points 
	# required for convergence is of the order of 10^100 : probably not gonna 
	# happen
	"""
	# FIXME: Debug plots, remove when not needed.
	exponent = -(p-psi_val)**2/a-(r-ra_val)**2/b-(dec-dec_val)**2/c-(i-inc_val)**2/d-(ph-phi_val)**2/e-(di-dist_val)**2/f
	from matplotlib import pyplot
	pyplot.clf()
	decimate = int(len(exponent)/1e5)
	if decimate < 1: decimate = 1
	pyplot.plot(numpy.linspace(1,len(exponent), len(exponent))[::decimate], exponent[::decimate], 'b-')
	#pyplot.plot(exponent)
	pyplot.grid()
	#pyplot.clf()
	pyplot.savefig("exponent.pdf")
	"""
	exponent = ma.masked_less_equal(-(p-psi_val)**2/a-(r-ra_val)**2/b-(dec-dec_val)**2/c-(i-inc_val)**2/d-(ph-phi_val)**2/e-(di-dist_val)**2/f, -700)
	exponent.fill_value = 0
	#print ma.count(exponent)
	return norm * numpy.exp(exponent)

#integral = scipy.integrate.tplquad(integrand, dec_min, dec_max, lambda x: ra_min, lambda x: ra_max, lambda x, y: psi_min, lambda x, y: psi_max)[0]
#print "scipy says: %f" % integral

#res, var = samp.integrate(integrand_1d, int(sys.argv[1]), "psi")
res, var = samp.integrate(integrand, int(sys.argv[1]), "psi", "ra", "dec", "phi", "inc", "dist")
print "Integral value: %f, stddev %f" % (res, numpy.sqrt(var))

plot_pdf(samp)
plot_cdf_inv(samp)
plot_one_d_hist(samp)
plot_two_d_hist(samp)
plot_ra_dec(samp)
