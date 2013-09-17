from collections import defaultdict

import numpy
from scipy import integrate, interpolate

__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

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
		self._pdf_norm = defaultdict(lambda x: 1)
		# Cache for the sampling points
		self._rvs = None
		# Sample point cache
		self._cache = []
		# parameter -> cdf^{-1} function object
		self.cdf_inv = {}
		# params for left and right limits
		self.llim, self.rlim = {}, {}

	def clear(self):
		"""
		Clear out the parameters and their settings, as well as clear the sample cache.
		"""
		self.params = set()
		self.pdf = {}
		self._pdf_norm = defaultdict(lambda x: 1)
		self._rvs = None
		self._cache = []
		self.cdf_inv = {}
		self.llim = {}
		self.rlim = {}

	def add_parameter(self, params, pdf, cdf_inv=None, left_limit=None, right_limit=None):
		"""
		Add one (or more) parameters to sample dimensions. params is either a string describing the parameter, or a tuple of strings. The tuple will indicate to the sampler that these parameters must be sampled together. left_limit and right_limit are on the infinite interval by default, but can and probably should be specified. If several params are given, left_limit, and right_limit must be a set of tuples with corresponding length. Sampling PDF is required, and if not provided, the cdf inverse function will be determined numerically from the sampling PDF.
		"""
		self.params.add(params)
		if type(params) is tuple:
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
		self.cdf_inv[params] = cdf_inv or self.cdf_inverse(params)

	def cdf_inverse(self, param):
		"""
		Numerically determine the inverse CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF inverse.
		"""
		# Solve P'(x) == p(x), with P[lower_boun] == 0
		def dP_cdf(p, x):
			return self.pdf[param](x)
		x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000)
		# FIXME: Why does this return only a nearly complete (0,1) interval
		cdf = integrate.odeint(dP_cdf, [0], x_i).T[0]
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

		if type(rvs) is int or type(rvs) is float:
			self._rvs = [numpy.random.uniform(0,1,rvs) for (a,b) in [(self.llim[p], self.rlim[p]) for p in args]]
			#cdf_rvs = [self.cdf_inv[param](rv) for (rv, param) in zip(self._rvs, args)]
			self._rvs = [self.cdf_inv[param](rv) for (rv, param) in zip(self._rvs, args)]
		else:
			self._rvs = rvs

		res = [(self.pdf[param](cdf_rv)/self._pdf_norm[param], cdf_rv) for (cdf_rv, param) in zip(self._rvs, args)]

		if kwargs.has_key("rdict"):
			return dict(zip(args, res))
		return zip(*res)

	def save_points(self, intg, prior):
		# NOTE: Will save points from other integrations before this if used more than once.
		self._cache.extend( [ rvs for rvs, ratio, rnd in zip(numpy.array(self._rvs).T, intg/prior, numpy.random.uniform(0, 1, len(prior))) if ratio < 1 or 1.0/ratio < rnd ] )

	# TODO: Remove args
	def integrate(self, func, n, *args):
		#self._rvs = numpy.random.uniform(0, 1, (len(self.params), n))
		p_s, rv = self.draw(n, *args)
		joint_p_s = numpy.prod(p_s, axis=0)
		fval = func(*rv)
		int_val = fval/joint_p_s
		maxval = [fval[0] or 1e-300]
		for v in fval[1:]:
			maxval.append( v if v > maxval[-1] else maxval[-1] )
		eff_samp = maxval/(int_val.cumsum()/numpy.linspace(1,n,n))
		"""
		FIXME: Debug plots. Get rid of them when done
		import matplotlib
		matplotlib.use("Agg")
		from matplotlib import pyplot
		pyplot.clf()
		pyplot.subplot(311)
		pyplot.plot(range(len(int_val)), int_val.cumsum()/numpy.linspace(1,n,n), 'k-')
		pyplot.grid()
		pyplot.subplot(312)
		pyplot.plot(range(len(int_val)), fval, 'b-')
		pyplot.subplot(313)
		pyplot.plot(range(len(int_val)), maxval/(int_val.cumsum()/numpy.linspace(1,n,n)), 'r-')
		pyplot.semilogy()
		pyplot.ylim([1e-5, 1e1])
		pyplot.grid()
		pyplot.savefig("integral.png")
		pyplot.clf()
		"""
		std = int_val.std()
		#self.save_points(int_val, joint_p_s)
		print "%d samples saved" % len(self._cache)
		int_val1 = int_val.sum()/n
		# FIXME: Running stddev
		return int_val1, std**2/n
