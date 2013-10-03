import bisect
from collections import defaultdict

import numpy
from scipy import integrate, interpolate

__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

# TODO: Move other sampling routines here

class MCSampler(object):
	"""
	Class to define a set of parameter names, limits, and probability densities.
	"""
	def __init__(self):
		# Parameter names
		self.params = set()
		# parameter -> pdf function object
		self.pdf = {}
		# If the pdfs aren't normalized, this will hold the normalization 
		# constant
		self._pdf_norm = defaultdict(lambda: 1)
		# Cache for the sampling points
		self._rvs = None
		# Sample point cache
		self._cache = []
		# parameter -> cdf^{-1} function object
		self.cdf = {}
		self.cdf_inv = {}
		# params for left and right limits
		self.llim, self.rlim = {}, {}

	def clear(self):
		"""
		Clear out the parameters and their settings, as well as clear the sample cache.
		"""
		self.params = set()
		self.pdf = {}
		self._pdf_norm = defaultdict(lambda: 1)
		self._rvs = None
		self._cache = []
		self.cdf = {}
		self.cdf_inv = {}
		self.llim = {}
		self.rlim = {}

	def add_parameter(self, params, pdf, cdf_inv=None, left_limit=None, right_limit=None):
		"""
		Add one (or more) parameters to sample dimensions. params is either a string describing the parameter, or a tuple of strings. The tuple will indicate to the sampler that these parameters must be sampled together. left_limit and right_limit are on the infinite interval by default, but can and probably should be specified. If several params are given, left_limit, and right_limit must be a set of tuples with corresponding length. Sampling PDF is required, and if not provided, the cdf inverse function will be determined numerically from the sampling PDF.
		"""
		self.params.add(params)
		if isinstance(params, tuple):
			if left_limit is None:
				self.llim[params] = list(float("-inf"))*len(params)
			else:
				self.llim[params] = left_limit
			if right_limit is None:
				self.rlim[params] = list(float("+inf"))*len(params)
			else:
				self.rlim[params] = right_limit
		else:
			if left_limit is None:
				self.llim[params] = float("-inf")
			else:
				self.llim[params] = left_limit
			if right_limit is None:
				self.rlim[params] = float("+inf")
			else:
				self.rlim[params] = right_limit
		self.pdf[params] = pdf
		# FIXME: This only works automagically for the 1d case currently
		self.cdf_inv[params] = cdf_inv or self.cdf_inverse(params)
		if not isinstance(params, tuple):
			self.cdf[params] =  self.cdf_function(params)

	def cdf_function(self, param):
		"""
		Numerically determine the  CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF.
		"""
		# Solve P'(x) == p(x), with P[lower_boun] == 0
		def dP_cdf(p, x):
			return self.pdf[param](x)
		x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000)
		# Integrator needs to have a step size which doesn't step over the
		# probability mass
		# TODO: Determine h_max.
		cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.1*(self.rlim[param]-self.llim[param])).T[0]
		if cdf[-1] != 1.0: # original pdf wasn't normalized
			self._pdf_norm[param] = cdf[-1]
			cdf /= cdf[-1]
		# Interpolate the inverse
		return interpolate.interp1d( x_i,cdf)

	def cdf_inverse(self, param):
		"""
		Numerically determine the inverse CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF inverse.
		"""
		# Solve P'(x) == p(x), with P[lower_boun] == 0
		def dP_cdf(p, x):
			return self.pdf[param](x)
		x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000)
		# Integrator needs to have a step size which doesn't step over the
		# probability mass
		# TODO: Determine h_max.
		cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.1*(self.rlim[param]-self.llim[param])).T[0]
		if cdf[-1] != 1.0: # original pdf wasn't normalized
			self._pdf_norm[param] = cdf[-1]
			cdf /= cdf[-1]
		# Interpolate the inverse
		return interpolate.interp1d(cdf, x_i)

	def draw(self, rvs, *args, **kwargs):
		"""
		Draw a set of random variates for parameter(s) args. Left and right limits are handed to the function. If args is None, then draw *all* parameters. 'rdict' parameter is a boolean. If true, returns a dict matched to param name rather than list. rvs must be either a list of uniform random variates to transform for sampling, or an integer number of samples to draw.
		"""
		if len(args) == 0 :
			args = self.params

		if isinstance(rvs, int) or isinstance(rvs, float):
			#
			# Convert all arguments to tuples
			#
			# FIXME: UGH! Really? This was the most elegant thing you could come
			# up with?
			self._rvs = [numpy.random.uniform(0,1,(len(p), rvs)) for p in map(lambda i: (i,) if not isinstance(i, tuple) else i, args)]
			self._rvs = numpy.array([self.cdf_inv[param](*rv) for (rv, param) in zip(self._rvs, args)])
		else:
			self._rvs = numpy.array(rvs)


		# FIXME: ELegance; get some of that...
		# This is mainly to ensure that the array can be "splatted", e.g.
		# separated out into its components for matching with args. The case of
		# one argument has to be handled specially.
		for (cdf_rv, param) in zip(self._rvs, args):
			if len(cdf_rv.shape) == 1:
				res = [(self.pdf[param](cdf_rv)/self._pdf_norm[param], cdf_rv) for (cdf_rv, param) in zip(self._rvs, args)]
			else:
				res = [(self.pdf[param](*cdf_rv)/self._pdf_norm[param], cdf_rv) for (cdf_rv, param) in zip(self._rvs, args)]

		#
		# Cache the samples we chose
		#
		self._rvs = dict(zip(args, self._rvs))

		#
		# Pack up the result if the user wants a dictonary instead
		#
		if kwargs.has_key("rdict"):
			return dict(zip(args, res))
		return zip(*res)

	def save_points(self, intg, prior):
		# NOTE: Will save points from other integrations before this if used more than once.
		self._cache.extend( [ rvs for rvs, ratio, rnd in zip(numpy.array(self._rvs).T, intg/prior, numpy.random.uniform(0, 1, len(prior))) if ratio < 1 or 1.0/ratio < rnd ] )

	# TODO: Idea: have args and kwargs, and let the user pin values via
	# kwargs and integrate through args
	def integrate(self, func, n, *args):
		"""
		Integrate func, by using n sample points. Right now, all params defined must be passed to args must be provided, but this will change soon.
		"""
		p_s, rv = self.draw(n, *args)
		joint_p_s = numpy.prod(p_s, axis=0)
		if len(rv[0].shape) != 1:
			rv = rv[0]
		fval = func(*rv)
		# sum_i f(x_i)/p_s(x_i)
		int_val = fval/joint_p_s
		maxval = [fval[0]/joint_p_s[0] or -float("Inf")]
		for v in int_val[1:]:
			maxval.append( v if v > maxval[-1] and v != 0 else maxval[-1] )
		eff_samp = int_val.cumsum()/maxval
		std = int_val.std()
		#self.save_points(int_val, joint_p_s)
		print "%d samples saved" % len(self._cache)
		int_val1 = int_val.sum()/n
		return int_val1, std**2/n

### UTILITIES: Predefined distributions
def uniform_samp(a, b, x):   # I prefer to vectorize with the same call for all functions, rather than hardcode vectorization
        if  x>a and x<b:
                return 1/(b-a)
        else:
                return 0
uniform_samp_vector = numpy.vectorize(uniform_samp,excluded=['a','b'],otypes=[numpy.float])

def inv_uniform_cdf(a, b, x):
	return (b-a)*x+a

def gauss_samp(mu, std, x):
	return 1.0/numpy.sqrt(2*numpy.pi*std**2)*numpy.exp(-(x-mu)**2/2/std**2)

def gauss_samp_withfloor(mu, std, myfloor, x):
	return 1.0/numpy.sqrt(2*numpy.pi*std**2)*numpy.exp(-(x-mu)**2/2/std**2) + myfloor

gauss_samp_withfloor_vector = numpy.vectorize(gauss_samp_withfloor,excluded=['mu','std','myfloor'],otypes=[numpy.float])


def cos_samp(x):
        return numpy.sin(x)/2   # x from 0, pi

def dec_samp(x):
        return numpy.sin(x+numpy.pi/2)/2   # x from 0, pi

cos_samp_vector = numpy.vectorize(cos_samp,otypes=[numpy.float])
dec_samp_vector = numpy.vectorize(dec_samp,otypes=[numpy.float])

def pseudo_dist_samp(r0,r):
        return r*r*numpy.exp( - (r0/r)*(r0/r)/2. + r0/r)+0.01  # put a floor on probability, so we converge

pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,excluded=['r0'],otypes=[numpy.float])
