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
# FITS utilities
#
def make_skymap(samp, res=32, fname=None):
	"""
	Plot the RA and dec distribution in a Mollewide projection.
	"""
	samples = samp._rvs
	smap = numpy.zeros(healpy.nside2npix(res))
	phi, thetar = samples[("ra", "dec")]
	# FIXME: Why is this flipped?
	for ind in healpy.ang2pix(res, numpy.pi - (thetar + numpy.pi/2), numpy.pi*2 - phi):
		smap[ind] += 1
	bfits.write_sky_map(fname or "pe_skymap.fits", smap)


#
# Plotting utilities
#
def plot_integral(fcn, samp, fargs=None, fname=None):
	"""
	Effectively redoes the process which happens in MCSampler.integrate, but keeps track of the values for plotting.
	"""
	args = []
	p_s = numpy.array([])
	for p in fargs or samp.params:
		if isinstance(p, tuple) and len(p) > 1:
			# FIXME: Will this preserve order?
			for a in samp._rvs[p]:
				args.append(a)
			p_s = numpy.hstack( (p_s, samp.pdf[p](*samp._rvs[p])/samp._pdf_norm[p]) )
		else:
			p_s = numpy.hstack( (p_s, samp.pdf[p](samp._rvs[p])/samp._pdf_norm[p]) )
			args.append(samp._rvs[p])

	p_s.shape = (len(samp.params), len(p_s)/len(samp.params))

	fval = fcn(*args)
	n = len(fval)
	#p_s = numpy.array([ samp.pdf[p](*x)/samp._pdf_norm[p] for p, x in samp._rvs.items() ])
	joint_p_s = numpy.prod(p_s, axis=0)
	int_val = fval/joint_p_s
	maxval = [fval[0]/joint_p_s[0] or -float("Inf")]
	for v in int_val[1:]:
			maxval.append( v if v > maxval[-1] and v != 0 else maxval[-1] )
	eff_samp = int_val.cumsum()/maxval

	pyplot.figure()
	pyplot.subplot(311)
	pyplot.title("Integral estimate")
	n_itr = range(1, len(int_val)+1)
	pyplot.semilogx()
	pyplot.plot(n_itr, int_val.cumsum()/numpy.linspace(1,n,n), 'k-')
	pyplot.ylabel("integral val")
	pyplot.twinx()
	pyplot.ylabel("integral std")
	pyplot.plot(n_itr, numpy.sqrt(cumvar(int_val)/n_itr), 'b-')
	pyplot.grid()
	pyplot.subplot(312)
	pyplot.title("Point integral estimate and maximum over iterations")
	pyplot.plot(n_itr, maxval, 'k-')
	pyplot.plot(n_itr, int_val, 'b-')
	pyplot.grid()
	pyplot.semilogx()
	pyplot.subplot(313)
	pyplot.title("Effective samples")
	pyplot.loglog()
	pyplot.ylabel("N_eff")
	pyplot.plot(range(1, len(int_val)+1), eff_samp, 'k-')
	pyplot.twinx()
	pyplot.loglog()
	pyplot.ylabel("ratio N_eff/N")
	pyplot.plot(range(1, len(int_val)+1), eff_samp/numpy.arange(1,len(int_val)+1), 'b-')
	pyplot.grid()
	pyplot.subplots_adjust(hspace=0.5)
	pyplot.savefig(fname or "integral.pdf")
	pyplot.close()

def plot_integrand(fcn, x1, x2, fname=None):
	"""
	Plot 1D integrand (fcn) from x1 to x2
	"""
	pyplot.figure()
	x_i = numpy.linspace(x1, x2, 1000)
	pyplot.plot(x_i, fcn(x_i))
	pyplot.savefig(fname or "integrand.pdf")
	pyplot.close()


def plot_pdf(samp, fname=None):
	"""
	Plot PDFs of sampling distributions if they are independent
	"""
	pyplot.figure()
	for param in samp.params:
		x_i = numpy.array([numpy.linspace(samp.llim[param], samp.rlim[param], 100)])
		pyplot.plot(x_i[0], map(samp.pdf[param], x_i)[0], '-', label=param)

	pyplot.grid()
	pyplot.legend()
	pyplot.savefig(fname or "pdf.pdf")
	pyplot.close()

def plot_cdf_inv(samp, fname=None):
	"""
	Plot inverse CDFs
	"""

	pyplot.figure()
	x_i = numpy.linspace(0, 1, 1000)
	for param, cdfinv in samp.cdf_inv.items():
		pyplot.plot(x_i, map(cdfinv, x_i), '-', label=param)

	pyplot.grid()
	pyplot.legend()
	pyplot.savefig(fname or "cdf_inv.pdf")
	pyplot.close()


def plot_one_d_hist(samp, fname=None):
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
	pyplot.savefig(fname or "samples_1d.pdf")
	pyplot.close()

def plot_two_d_hist(samp, fname=None):
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
	
	pyplot.savefig(fname or "samples_2d.pdf", figsize=(10,3))
	pyplot.close()
	
def plot_ra_dec(samp, use_pdf=False, fname=None, key=("ra", "dec")):
	"""
	Plot the RA and dec distribution in a Mollewide projection.
	"""
	samples = samp._rvs
	if isinstance(key, tuple):
		ra_samp, dec_samp = samples[key]
		ra_min, dec_min = samp.llim[key]
		ra_max, dec_max = samp.rlim[key]
		pdf = samp.pdf[key]
	else:
		ra_samp, dec_samp = samples[key[0]], samples[key[1]]
		ra_min, ra_max = samp.llim[key[0]], samp.rlim[key[0]]
		dec_min, dec_max = samp.llim[key[1]], samp.rlim[key[1]]
		def radec(ra, dec):
			return (samp.pdf[key[0]], samp.pdf[key[1]])
		pdf = radec

	fig = pyplot.figure(figsize=(10,10))
	m = Basemap(projection='moll', lon_0=0, resolution='c')
	if not use_pdf:
		hist, xedge, yedge = numpy.histogram2d(ra_samp, dec_samp, bins=(100, 100), normed=True)
		x, y = numpy.meshgrid(xedge, yedge)
		x *= 180/numpy.pi
		y *= 180/numpy.pi
		m.contourf(x[:-1,:-1], y[:-1,:-1], hist.T, 100, cmap=matplotlib.cm.jet, latlon=True)
	else:
		xedge = numpy.linspace(ra_min, ra_max, 100)
		yedge = numpy.linspace(dec_min, dec_max, 100)
		x, y = numpy.meshgrid(xedge, yedge)
		hist = pdf(x, y)
		x *= 180/numpy.pi
		y *= 180/numpy.pi
		m.contourf(x, y, hist, cmap=matplotlib.cm.jet, latlon=True)

	pyplot.colorbar()
	m.drawparallels(numpy.arange(-90, 120, 30))
	m.drawmeridians(numpy.arange(0, 420, 60))
	pyplot.savefig(fname or "radec_proj.pdf")
	pyplot.close()
	return hist

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
# Test some other 1-D integrals
#
"""
test_min, test_max = -10, 10
def integrand_1d(x):
	return numpy.sinc(10*x)
plot_integrand(integrand_1d, test_min, test_max)

samp.add_parameter("p1", functools.partial(mcsampler.uniform_samp_vector, test_min, test_max), None, test_min, test_max)
plot_cdf_inv(samp)
res, var = samp.integrate(integrand_1d, int(sys.argv[1]), "p1")
print res, var
integral = scipy.integrate.quad(integrand_1d, psi_min, psi_max)[0]
print integral
samp.clear()
"""

def integrand_1d(p):
	return 1.0/numpy.sqrt(2*numpy.pi*psi_width**2)*numpy.exp(-(p-psi_val)**2/2.0/psi_width**2)
plot_integrand(integrand_1d, psi_min, psi_max)

#
# Test 1: What happens when we know the exact right answer
#

d = "test1/"
if not os.path.exists(d): os.makedirs(d)
os.chdir(d)

samp.add_parameter("psi", functools.partial(mcsampler.gauss_samp, psi_val, psi_width), None, psi_min, psi_max)
sampler_integral = samp.integrate(integrand_1d, "psi", nmax=1)
integral = scipy.integrate.quad(integrand_1d, psi_min, psi_max)[0]
print("scipy answer vs our answer:",  integral, sampler_integral)
plot_pdf(samp)
plot_cdf_inv(samp)
plot_one_d_hist(samp)
samp.clear()

os.chdir("../")

#
# Test 2: Same integrand, uniform sampling
#

d = "test2/"
if not os.path.exists(d): os.makedirs(d)
os.chdir(d)

samp.add_parameter("psi", functools.partial(mcsampler.uniform_samp_vector, psi_min, psi_max), None, psi_min, psi_max)
#print samp.integrate(integrand_1d, "psi", neff=1000, nmax=int(sys.argv[1]))
res, var = samp.integrate(integrand_1d, "psi", neff=1000)
print("Integral: %f, stddev: %f" % (res, numpy.sqrt(var)))
plot_integral(integrand_1d, samp)
plot_pdf(samp)
plot_cdf_inv(samp)
plot_one_d_hist(samp)
samp.clear()

os.chdir("../")

#
# Test 3: Same integrand, gaussian sampling -- various width and offsets
#
widths = [0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 1.01, 1.1, 1.5, 2, 4, 5, 10]
offsets = [-2.0, -1.0, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0]

variances = []
wo = []
d = "test3/"
if not os.path.exists(d): os.makedirs(d)
for w, o in itertools.product(widths, offsets):
	d1 = "test3/" + "w_%2.2f_o_%2.2f/" % (w,o)
	if not os.path.exists(d1): os.makedirs(d1)
	os.chdir(d1)
	samp.add_parameter("psi", functools.partial(mcsampler.gauss_samp, psi_val-o*psi_width, w*psi_width), None, psi_min, psi_max)
	plot_pdf(samp)
	plot_cdf_inv(samp)
	print("Width of Gaussian sampler (in units of the width of the integrand: %f" % w)
	print("offset of Gaussian sampler (in units of the width of the integrand: %f" % o)
	res, var = samp.integrate(integrand_1d, "psi", neff=100, nmax=int(1e5))
	plot_integral(integrand_1d, samp)
	variances.append(var)
	wo.append((w, o))
	print("Integral: %f, stddev: %f" % (res, numpy.sqrt(var)))
	samp.clear()
	os.chdir("../../")

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

#
# Testing variance under convergence: Loop over samples and test sigma relation 
# for desired error
# 
pyplot.figure()

d = "test3b/"
if not os.path.exists(d): os.makedirs(d)
os.chdir(d)

samp.add_parameter("psi", functools.partial(mcsampler.uniform_samp_vector, psi_min, psi_max), None, psi_min, psi_max)
for n in 10**(numpy.arange(1,6)):
	ans = []
	print("Number of samples in integral %d" % n)
	for i in range(1000):
		res, var = samp.integrate(integrand_1d, "psi", nmax=n)
		ans.append( (res-1.0)/numpy.sqrt(var) )
	pyplot.hist(ans, bins=20, label="%d" % n)
pyplot.title("$(I-\\bar{I})/\\bar\\sigma$")
pyplot.grid()
pyplot.legend()
pyplot.savefig("integral_hist.pdf")
samp.clear()
exit()

os.chdir("../")

#
# Test 4: 2D PDFs, rejection sampling, and BAYESTAR
#

#
# Read FITS data
#
smap, smap_meta = bfits.read_sky_map("data/30602.toa.fits.gz")
sides = healpy.npix2nside(len(smap))

d = "test4/"
if not os.path.exists(d): os.makedirs(d)
os.chdir(d)

#
# For sampling PDF, have the option to apply a temperature argument
#
def bayestar_temp(temp, skymap, lon, lat):
	return bplot._healpix_lookup(skymap, lon, lat)**(1.0/temp)
def bayestar_norm(skymap, lon, lat):
	return bplot._healpix_lookup(skymap, lon, lat)*numpy.cos(lat)

sky_pdf = functools.partial(bayestar_temp, 1, smap)
norm_sky_pdf = functools.partial(bayestar_norm, smap)

#
# Renormalize for the integral
#
#norm = scipy.integrate.dblquad(sky_pdf, dec_min, dec_max, lambda x: ra_min, lambda x: ra_max, epsabs=1e-3, epsrel=1e-3)[0]
norm = scipy.integrate.dblquad(norm_sky_pdf, dec_min, dec_max, lambda x: ra_min, lambda x: ra_max, epsabs=1e-3, epsrel=1e-3)[0]
# FIXME: This shouldn't address variables that are "hidden"
samp._pdf_norm[("ra", "dec")] = norm

#
# Try to integrate the sky PDF itself
#
pix_radsq = len(smap)/4.0/numpy.pi
def integrand_2d(ra, dec):
	#return bayestar_temp(1, smap, ra, dec) * pix_radsq
	return numpy.ones(ra.shape)

generate_sky_points = functools.partial(mcsampler.sky_rejection, smap)
samp.add_parameter(("ra", "dec"), sky_pdf, generate_sky_points, (ra_min, dec_min), (ra_max, dec_max))

res, var = samp.integrate(integrand_2d, ("ra", "dec"), neff=1000)
print("Integral: %f, stddev: %f" % (res, numpy.sqrt(var)))
plot_ra_dec(samp)
plot_ra_dec(samp, use_pdf=True, fname="pdf.pdf")
plot_integral(integrand_2d, samp)
make_skymap(samp)

# FIXME: Never converges
#print "scipy says %g" % scipy.integrate.dblquad(integrand_2d, dec_min, dec_max, lambda x: ra_min, lambda x: ra_max, epsabs=1e-3, epsrel=1e-3)[0]

samp.clear()
os.chdir("../")

#
# Test 5a: Uniform sampling, cdf provided
#
d = "test5a/"
if not os.path.exists(d): os.makedirs(d)
os.chdir(d)

#samp.add_parameter("psi", functools.partial(mcsampler.uniform_samp_vector, psi_min, psi_max), functools.partial(mcsampler.inv_uniform_cdf, psi_min, psi_max), psi_min, psi_max)
#samp.add_parameter("ra", functools.partial(mcsampler.uniform_samp_vector, ra_min, ra_max), functools.partial(mcsampler.inv_uniform_cdf, ra_min, ra_max), ra_min, ra_max)
#samp.add_parameter("dec", functools.partial(mcsampler.uniform_samp_vector, dec_min, dec_max), functools.partial(mcsampler.inv_uniform_cdf, dec_min, dec_max), dec_min, dec_max)

#
# Test 5b: Uniform sampling, cdf provided
#
# Uniform sampling, auto-cdf inverse
#samp.add_parameter("psi", functools.partial(mcsampler.uniform_samp_vector, psi_min, psi_max), None, psi_min, psi_max)
samp.add_parameter("psi", functools.partial(mcsampler.gauss_samp, psi_val, 2*psi_width), None, psi_min, psi_max)
samp.add_parameter("ra", functools.partial(mcsampler.uniform_samp_vector, ra_min, ra_max), None, ra_min, ra_max)
samp.add_parameter("dec", functools.partial(mcsampler.uniform_samp_vector, dec_min, dec_max), None, dec_min, dec_max)
samp.add_parameter("phi", functools.partial(mcsampler.uniform_samp_vector, phi_min, phi_max), None, phi_min, phi_max)
#samp.add_parameter("inc", functools.partial(mcsampler.uniform_samp_vector, inc_min, inc_max), None, inc_min, inc_max)
samp.add_parameter("inc", functools.partial(mcsampler.gauss_samp, inc_val, 2*inc_width), None, inc_min, inc_max)
samp.add_parameter("dist", functools.partial(mcsampler.uniform_samp_vector, dist_min, dist_max), None, dist_min, dist_max)

#
# Test 5c: Gaussian sampling, auto-cdf inverse -- Doesn't work yet
#
#samp.add_parameter("psi", functools.partial(mcsampler.gauss_samp, 0, (psi_max-psi_min)/3.0), None, psi_min, psi_max)
#samp.add_parameter("ra", functools.partial(mcsampler.gauss_samp, 0, (ra_max-ra_min)/10.0), None, ra_min, ra_max)
#samp.add_parameter("dec", functools.partial(mcsampler.gauss_samp, (dec_max+dec_min)/2, (dec_max-dec_min)/10.0), None, dec_min, dec_max)

#
# Full 6-D test
#

a, b, c, d, e, f = 2*psi_width**2, 2*ra_width**2, 2*dec_width**2, 2*inc_width**2, 2*phi_width**2, 2*dist_width**2
norm = 1.0/numpy.sqrt((numpy.pi)**len(samp.params)*a*b*c*d*e*f)
numpy.seterr(invalid="raise")
def integrand(p, r, dec, ph, i, di):
	exponent = -(p-psi_val)**2/a-(r-ra_val)**2/b-(dec-dec_val)**2/c-(i-inc_val)**2/d-(ph-phi_val)**2/e-(di-dist_val)**2/f
	return norm * numpy.exp(exponent)

res, var = samp.integrate(integrand, "psi", "ra", "dec", "phi", "inc", "dist", neff=100, nmax=int(1e6))
print("Integral value: %f, stddev %f" % (res, numpy.sqrt(var)))

plot_integral(integrand, samp, ("psi", "ra", "dec", "phi", "inc", "dist"))
plot_pdf(samp)
plot_cdf_inv(samp)
plot_one_d_hist(samp)
plot_two_d_hist(samp)
plot_ra_dec(samp, key=["ra", "dec"])

os.chdir("../")
