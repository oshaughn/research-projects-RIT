import bisect
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
		self._pdf_norm = defaultdict(lambda: 1)
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
		self._pdf_norm = defaultdict(lambda: 1)
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
		# FIXME: This only works for the 2d case currently
		if isinstance(params, tuple):
			self.cdf_inv[params] = cdf_inv or self.cdf_inverse_2d(params)
		else:
			self.cdf_inv[params] = cdf_inv or self.cdf_inverse(params)

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
		cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.1).T[0]
		if cdf[-1] != 1.0: # original pdf wasn't normalized
			self._pdf_norm[param] = cdf[-1]
			cdf /= cdf[-1]
		# Interpolate the inverse
		return interpolate.interp1d(cdf, x_i)

	# NOTE: Could just be a user function that's passed in
	def cdf_inverse_2d(self, param):
		"""
		Numerically determine the inverse CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF inverse.
		"""
		def reject_samp(xin, yin):
			i = 0
			while i < len(xin):
				x_i = numpy.random.uniform(self.llim[param][0], self.rlim[param][0])
				y_i = numpy.random.uniform(self.llim[param][1], self.rlim[param][1])

				trial = numpy.random.uniform(0,1)
				print i, trial, self.pdf[param](x_i, y_i) 
				if trial < self.pdf[param](x_i, y_i):
					xin[i], yin[i] = [x_i, y_i]
					i += 1
			return numpy.array([xin, yin])
		return reject_samp

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
			# TODO: *rv
			self._rvs = numpy.array([self.cdf_inv[param](*rv) for (rv, param) in zip(self._rvs, args)])
		else:
			self._rvs = numpy.array(rvs)


		# FIXME: ELegance; get some of that...
		for (cdf_rv, param) in zip(self._rvs, args):
			if len(cdf_rv.shape) == 1:
				res = [(self.pdf[param](cdf_rv)/self._pdf_norm[param], cdf_rv) for (cdf_rv, param) in zip(self._rvs, args)]
			else:
				res = [(self.pdf[param](*cdf_rv)/self._pdf_norm[param], cdf_rv) for (cdf_rv, param) in zip(self._rvs, args)]

		self._rvs = dict(zip(args, self._rvs))

		if kwargs.has_key("rdict"):
			return dict(zip(args, res))
		return zip(*res)

	def save_points(self, intg, prior):
		# NOTE: Will save points from other integrations before this if used more than once.
		self._cache.extend( [ rvs for rvs, ratio, rnd in zip(numpy.array(self._rvs).T, intg/prior, numpy.random.uniform(0, 1, len(prior))) if ratio < 1 or 1.0/ratio < rnd ] )

	# TODO: Remove args
	# NOTE: Better idea: have args and kwargs, and let the user pin values via
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
		maxval = [fval[0] or 0]
		for v in fval[1:]:
			maxval.append( v if v > maxval[-1] else maxval[-1] )
		eff_samp = int_val.cumsum()/maxval
		indx = bisect.bisect_right(eff_samp, 100)
		#FIXME: Debug plots. Get rid of them when done
		import matplotlib
		matplotlib.use("Agg")
		from matplotlib import pyplot
		pyplot.clf()
		pyplot.subplot(311)
		pyplot.plot(range(1, len(int_val)+1), int_val.cumsum()/numpy.linspace(1,n,n), 'k-')
		pyplot.semilogx()
		pyplot.grid()
		pyplot.subplot(312)
		pyplot.plot(range(1, len(int_val)+1), maxval, 'k-')
		pyplot.plot(range(1, len(int_val)+1), fval, 'b-')
		pyplot.semilogx()
		pyplot.subplot(313)
		pyplot.plot(range(1, len(int_val)+1), eff_samp, 'b-')
		#pyplot.plot(range(len(int_val)), eff_samp, 'r-')
		pyplot.loglog()
		#pyplot.ylim([1e-1, 1e1])
		pyplot.grid()
		pyplot.savefig("integral.png")
		pyplot.clf()
		if indx == len(int_val):
			std = 0
		else:
			std = int_val[indx:].std()
		#self.save_points(int_val, joint_p_s)
		print "%d samples saved" % len(self._cache)
		int_val1 = int_val.sum()/n
		# FIXME: Running stddev
		# TODO: Wrong n in variance
		return int_val1, std**2/max(1, (n-indx))
