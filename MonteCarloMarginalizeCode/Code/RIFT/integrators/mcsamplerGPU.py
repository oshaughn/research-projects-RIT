# mcsamplerGPU.py
#   From Dan Wysocki based on code by C. Pankow
#   From fork https://github.com/oshaughn/research-projects-RIT/blob/GPU-danw-merge/MonteCarloMarginalizeCode/Code/mcsampler.py



import sys
import math
#import bisect
from collections import defaultdict

import numpy
np=numpy #import numpy as np
from scipy import integrate, interpolate, special
import itertools
import functools

import os

try:
  import cupy
  import cupyx   # needed for logsumexp
  xpy_default=cupy
  try:
    xpy_special_default = cupyx.scipy.special
    if not(hasattr(xpy_special_default,'logsumexp')):
          print(" mcsamplerGPU: no cupyx.scipy.special.logsumexp, fallback mode ...")
          xpy_special_default= special
  except:
    print(" mcsamplerGPU: no cupyx.scipy.special, fallback mode ...")
    xpy_special_default= special
  identity_convert = cupy.asnumpy
  identity_convert_togpu = cupy.asarray
  junk_to_check_installed = cupy.array(5)  # this will fail if GPU not installed correctly
  cupy_ok = True
  cupy_pi = cupy.array(np.pi)

  from RIFT.interpolators.interp_gpu import interp

#  from logging import info as log
#  import inspect
#  def verbose_cupy_asarray(*args, **kwargs):
#     print("Transferring data to VRAM", *args, **kwargs)
#     return cupy.asarray(*args, **kwargs)
#  def verbose_cupy_asnumpy(*args, **kwargs):
#     curframe = inspect.currentframe()
#     calframe = inspect.getouterframes(curframe, 2)
#     log("Transferring data to RAM",calframe[1][3]) #,args[0].__name__) #, *args, **kwargs)
#     return cupy.ndarray.asnumpy(*args, **kwargs)
#  cupy.asarray = verbose_cupy_asarray  
#  cupy.ndarray.asnumpy = verbose_cupy_asnumpy

except:
  print(' no cupy (mcsamplerGPU)')
#  import numpy as cupy  # will automatically replace cupy calls with numpy!
  xpy_default=numpy  # just in case, to make replacement clear and to enable override
  xpy_special_default = special
  identity_convert = lambda x: x  # trivial return itself
  identity_convert_togpu = lambda x: x
  cupy_ok = False
  cupy_pi = np.pi

def set_xpy_to_numpy():
   xpy_default=numpy
   identity_convert = lambda x: x  # trivial return itself
   identity_convert_togpu = lambda x: x
   cupy_ok = False
   

if 'PROFILE' not in os.environ:
   def profile(fn):
        return fn

if not( 'RIFT_LOWLATENCY'  in os.environ):
    # Dont support selected external packages in low latency
 try:
    import healpy
 except:
    print(" - No healpy - ")

from ..integrators.statutils import  update,finalize, init_log,update_log,finalize_log

#from multiprocessing import Pool

from RIFT.likelihood import vectorized_general_tools

__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>, Dan Wysocki, R. O'Shaughnessy"

rosDebugMessages = True

class NanOrInf(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MCSampler(object):
    """
    Class to define a set of parameter names, limits, and probability densities.
    """

    @staticmethod
    def match_params_from_args(args, params):
        """
        Given two unordered sets of parameters, one a set of all "basic" elements (strings) possible, and one a set of elements both "basic" strings and "combined" (basic strings in tuples), determine whether the sets are equivalent if no basic element is repeated.
        e.g. set A ?= set B
        ("a", "b", "c") ?= ("a", "b", "c") ==> True
        (("a", "b", "c")) ?= ("a", "b", "c") ==> True
        (("a", "b"), "d")) ?= ("a", "b", "c") ==> False  # basic element 'd' not in set B
        (("a", "b"), "d")) ?= ("a", "b", "d", "c") ==> False  # not all elements in set B represented in set A
        """
        not_common = set(args) ^ set(params)
        if len(not_common) == 0:
            # All params match
            return True
        if all([not isinstance(i, tuple) for i in not_common]):
            # The only way this is possible is if there are
            # no extraneous params in args
            return False

        to_match, against = [i for i in not_common if not isinstance(i, tuple)], [i for i in not_common if isinstance(i, tuple)]

        matched = []
        import itertools
        for i in range(2, max(list(map(len, against)))+1):
            matched.extend([t for t in itertools.permutations(to_match, i) if t in against])
        return (set(matched) ^ set(against)) == set()


    def __init__(self):
        # Total number of samples drawn
        self.ntotal = 0
        # Parameter names
        self.params = set()
        self.params_ordered = []  # keep them in order. Important to break likelihood function need for names
        # parameter -> pdf function object
        self.pdf = {}
        self.pdf_initial = {}
        # If the pdfs aren't normalized, this will hold the normalization 
        # constant
        self._pdf_norm = defaultdict(lambda: 1)
        # Cache for the sampling points
        self._rvs = {}
        # parameter -> cdf^{-1} function object
        self.cdf = {}
        self.cdf_inv = {}
        self.cdf_inv_initial = {}
        # params for left and right limits
        self.llim, self.rlim = {}, {}
        # Keep track of the adaptive parameters
        self.adaptive = []

        # Keep track of the adaptive parameter 1-D marginalizations
        self._hist = {}

        # MEASURES (=priors): ROS needs these at the sampler level, to clearly separate their effects
        # ASSUMES the user insures they are normalized
        self.prior_pdf = {}

        # histogram setup
        self.setup_hist()
        self.xpy = numpy
        self.identity_convert = lambda x: x  # if needed, convert to numpy format  (e.g, cupy.asnumpy)
        self.identity_convert_togpu = lambda x: x

    def clear(self):
        """
        Clear out the parameters and their settings, as well as clear the sample cache.
        """
        self.params = set()
        self.params_ordered = []
        self.pdf = {}
        self._pdf_norm = defaultdict(lambda: 1.0)
        self._rvs = {}
        self._hist = {}
        self.cdf = {}
        self.cdf_inv = {}
        self.llim = {}
        self.rlim = {}
        self.adaptive = []

    def add_parameter(self, params, pdf,  cdf_inv=None, left_limit=None, right_limit=None, prior_pdf=None, adaptive_sampling=False):
        """
        Add one (or more) parameters to sample dimensions. params is either a string describing the parameter, or a tuple of strings. The tuple will indicate to the sampler that these parameters must be sampled together. left_limit and right_limit are on the infinite interval by default, but can and probably should be specified. If several params are given, left_limit, and right_limit must be a set of tuples with corresponding length. Sampling PDF is required, and if not provided, the cdf inverse function will be determined numerically from the sampling PDF.
        """
        self.params.add(params) # does NOT preserve order in which parameters are provided
        self.params_ordered.append(params)
        if rosDebugMessages: 
            print(" Adding parameter ", params, " with limits ", [left_limit, right_limit])
        if isinstance(params, tuple):
            assert all([lim[0] < lim[1] for lim in zip(left_limit, right_limit)])
            if left_limit is None:
                self.llim[params] = list(float("-inf"))*len(params)
            else:
                self.llim[params] = left_limit
            if right_limit is None:
                self.rlim[params] = list(float("+inf"))*len(params)
            else:
                self.rlim[params] = right_limit
        else:
            assert left_limit < right_limit
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
        if cdf_inv:
          self.cdf_inv[params] = cdf_inv
        else:
          self.cdf_inv[params] =  self.cdf_inverse(params)
        self.pdf_initial[params] = pdf
        self.cdf_inv_initial[params] = self.cdf_inv[params]
        if not isinstance(params, tuple):
#            self.cdf[params] =  self.cdf_function(params)
            if prior_pdf is None:
                self.prior_pdf[params] = lambda x:1
            else:
                self.prior_pdf[params] = prior_pdf
        self.prior_pdf[params] = prior_pdf

        if adaptive_sampling:
            print("   Adapting ", params)
            self.adaptive.append(params)

    def reset_sampling(self,param):
      self.pdf[param] = self.pdf_initial[param]
      self.cdf_inv[param] = self.cdf_inv_initial[param]

    def setup_hist(self):
        """
        Initializes dictionaries for all of the info that needs to be stored for
        the histograms, across every parameter.
        """
        self.x_min = {}
        self.x_max = {}
        self.x_max_minus_min = {}
        self.dx = {}
        self.n_bins = {}

        self.histogram_edges = {}
        self.histogram_values = {}
        self.histogram_cdf = {}


    def setup_hist_single_param(self, x_min, x_max, n_bins, param):
        # Compute the range of allowed values.
        x_max_minus_min = x_max - x_min
        # Compute the points at which the histogram will be evaluated, and store
        # the spacing used.
        histogram_edges, dx = self.xpy.linspace(
            0.0, 1.0, n_bins+1,
            retstep=True,
        )

        # Initialize output array for CDF.
        # Reminder:   histogram_cdf[0] ==0 (hardcoded) and we *should* have histogram_cdf[-1 == n_bins+1] == 1
        histogram_cdf = self.xpy.empty(n_bins+1, dtype=numpy.float64)

        # Store basic setup parameters
        self.x_min[param] = x_min
        self.x_max[param] = x_max
        self.x_max_minus_min[param] = x_max_minus_min
        self.dx[param] = dx
        self.n_bins[param] = n_bins

        self.histogram_edges[param] = histogram_edges
        self.histogram_cdf[param] = histogram_cdf


    def compute_hist(self, x_samples, param,weights=None,floor_level=0):
        # Rescale the samples to [0, 1]
        y_samples = (
            (x_samples - self.x_min[param]) / self.x_max_minus_min[param]
        )
        # Evaluate the histogram at each of the bins.
        histogram_values = vectorized_general_tools.histogram(
            y_samples, self.n_bins[param],
            xpy=self.xpy,
            weights=weights
        )
        # *renormalize* histogram
        #   - ideally this isn't necessary, BUT mainly for GPUs there can be some points lost to the nbins+1 region due to truncation
        #   - this slightly breaks normalization, and produces an overabundance in the highest bin when we build the CDF. Which (I think) cascades.
        #    - because we use a histogram CDF later
        histogram_values *= 1./self.xpy.sum(histogram_values)

        # Smooth the histogram
#        kernel_size =3
#        histogram_values = self.xpy.convolve( histogram_values, self.xpy.ones(kernel_size)/kernel_size,mode='same')
        # Mix with a uniform sampling
        histogram_values =    histogram_values*(1-floor_level)+floor_level*self.xpy.ones(len(histogram_values))/len(histogram_values)

        # Evaluate the CDF by taking a cumulative sum of the histogram.
        n_bins = len(self.histogram_cdf[param]) 
        self.xpy.cumsum(histogram_values[:n_bins], out=self.histogram_cdf[param][1:])  # redundant cast, since length also enforced at a lower level to be correct
        self.histogram_cdf[param][0] = 0.0
        #print(param, 'cdf', self.histogram_cdf[param])

        # Renormalize histogram (count per unit delta x, assumed constant)
        histogram_values /= self.x_max_minus_min[param] /self.n_bins[param]

        # Store histogram values.
        self.histogram_values[param] = histogram_values


    def cdf_inverse_from_hist(self, P, param,old_style=False):
        # Compute the value of the inverse CDF, but scaled to [0, 1].
       """
        cdf_inverse_from_hist
           - for now, do on the CPU, since this is done rarely and involves fairly small arrays
           - this is very wasteful, since we are casting back to the CPU for ALL our sampling points
       """
       if old_style or not(cupy_ok):
         dat_cdf = identity_convert(self.histogram_cdf[param])
         dat_edges = identity_convert(self.histogram_edges[param])
         y = np.interp(
           identity_convert(P), dat_cdf,
           dat_edges,
           )
         # Return the value in the original scaling.
         return identity_convert_togpu(y)*self.x_max_minus_min[param] + self.x_min[param]
       dat_cdf = self.histogram_cdf[param]
       dat_edges =self.histogram_edges[param]
       y = interp(P,dat_cdf,dat_edges)
       return y*self.x_max_minus_min[param] + self.x_min[param]

    def pdf_from_hist(self, x, param):
        # Rescale `x` to [0, 1].
        y = (x - self.x_min[param]) / self.x_max_minus_min[param]
        # Compute the indices of the histogram bins that `x` falls into.
        indices = self.xpy.trunc(y / self.dx[param], out=y).astype(np.int32)
        indices = self.xpy.minimum(indices,self.n_bins[param])  # prevent being out of range due to rounding !
        # Return the value of the histogram.
        return self.histogram_values[param][indices]


    def cdf_function(self, param):
        """
        Numerically determine the  CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF.
        NOT USED IN THIS ROUTINE, SHOULD NOT BE CALLED
        """
        print(" Do not call this routine! ")
        raise Exception(" mcsamplerGPU: cdf_function not to be used ")
        # Solve P'(x) == p(x), with P[lower_boun] == 0
        def dP_cdf(p, x):
            if x > self.rlim[param] or x < self.llim[param]:
                return 0
            return self.pdf[param](x)
        x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000)
        # Integrator needs to have a step size which doesn't step over the
        # probability mass
        # TODO: Determine h_max.
        cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.01*(self.rlim[param]-self.llim[param])).T[0]
        if cdf[-1] != 1.0: # original pdf wasn't normalized
            self._pdf_norm[param] = cdf[-1]
            cdf /= cdf[-1]
        # Interpolate the inverse
        return interpolate.interp1d( x_i,cdf)

    def cdf_function_from_histogram(self, x):
        """
        Computes the CDF from a histogram at the points `x`.
        Params
        ------
        x : array_like, shape = sample_shape
        Returns
        -------
        P(x) : array_like, shape = sample_shape
        """
        float_indices = (x - self.x0) / self.dx
        indices, fractions = self.xpy.modf(float_indices)

        cdf_before = self.partial_cdfs[indices]
        cdf_after = self.dx * fractions * (
            self.bin_heights[indices] +
            fractions * self.bin_deltas[indices]
        )

        return self.xpy.add(cdf_before, cdf_after, out=cdf_before)


    def cdf_inverse(self, param):
        """
        Numerically determine the inverse CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF inverse.
        """
        # Solve P'(x) == p(x), with P[lower_boun] == 0
        def dP_cdf(p, x):
            if x > self.rlim[param] or x < self.llim[param]:
                return 0
            return self.pdf[param](x)
        x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000)
        # Integrator needs to have a step size which doesn't step over the
        # probability mass
        # TODO: Determine h_max.
        cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.01*(self.rlim[param]-self.llim[param])).T[0]
        if cdf[-1] != 1.0: # original pdf wasn't normalized
            self._pdf_norm[param] = cdf[-1]
            cdf /= cdf[-1]
        # Interpolate the inverse
        return interpolate.interp1d(cdf, x_i)

    @profile
    def draw_simplified(self, n_samples, *args, **kwargs):
        if len(args) == 0:
            args = self.params

        n_samples = int(n_samples) 
        n_params = len(args)
        
        save_no_samples = kwargs.get("save_no_samples", False)

        # Allocate memory.
        rv = self.xpy.empty((n_params, n_samples), dtype=numpy.float64)
        joint_p_s = self.xpy.ones(n_samples, dtype=numpy.float64)
        joint_p_prior = self.xpy.ones(n_samples, dtype=numpy.float64)

        # Iterate over the parameters.
        for i, param in enumerate(args):
            # Do inverse CDF sampling for the parameter.
            unif_samples = self.xpy.random.uniform(0.0, 1.0, n_samples)
            param_samples = self.cdf_inv[param](unif_samples)

            # Store the random samples, and multiply on the contribution to the
            # joint PDF and joint prior at those samples.
            rv[i] = param_samples
            joint_p_s *= self.pdf[param](param_samples)
            #val= self.pdf[param](param_samples); print(type(val),param,xpy_default)
            # portfolio compatibility: prior_pdf is not always returning nice things
            prior_vals = self.prior_pdf[param](param_samples)
            if isinstance(prior_vals, self.xpy.ndarray):
              joint_p_prior *= prior_vals
            else:
              joint_p_prior *= identity_convert_togpu(prior_vals)
            #val=self.prior_pdf[param](param_samples); print(type(val),param)

        #
        # Cache the samples we chose
        #
        if not save_no_samples:
            if len(self._rvs) == 0:
               self._rvs = dict(list(zip(args, rv)))
            else:
               rvs_tmp = dict(list(zip(args, rv)))
               #for p, ar in self._rvs.items():
               for p in self.params_ordered:
                   self._rvs[p] = self.xpy.hstack((self._rvs[p], rvs_tmp[p]))


        return joint_p_s, joint_p_prior, rv

    #@profile
    def draw(self, rvs, *args,**kwargs):
        """
        Draw a set of random variates for parameter(s) args. Left and right limits are handed to the function. If args is None, then draw *all* parameters. 'rdict' parameter is a boolean. If true, returns a dict matched to param name rather than list. rvs must be either a list of uniform random variates to transform for sampling, or an integer number of samples to draw.
        """
        if len(args) == 0:
            args = self.params

        save_no_samples= False
        if 'save_no_samples' in list(kwargs.keys()):
            save_no_samples = kwargs['save_no_samples']


        if isinstance(rvs, int) or isinstance(rvs, float):
            #
            # Convert all arguments to tuples
            #
            # FIXME: UGH! Really? This was the most elegant thing you could come
            # up with?
            rvs_tmp = [
                numpy.random.uniform(0,1,(len(p), int(rvs)))
                for p in
                [(i,) if not isinstance(i, tuple) else i for i in args]
            ]
            rvs_tmp = numpy.array([
                self.cdf_inv[param](*rv)
                for (rv, param)
                in zip(rvs_tmp, args)
            ], dtype=numpy.float64)

        # FIXME: ELegance; get some of that...
        # This is mainly to ensure that the array can be "splatted", e.g.
        # separated out into its components for matching with args. The case of
        # one argument has to be handled specially.
        res = []
        for (cdf_rv, param) in zip(rvs_tmp, args):
            if len(cdf_rv.shape) == 1:
                res.append((self.pdf[param](numpy.float64(cdf_rv)).astype(numpy.float64)/self._pdf_norm[param], self.prior_pdf[param](cdf_rv), cdf_rv))
            else:
                # NOTE: the "astype" is employed here because the arrays can be
                # irregular and thus assigned the 'object' type. Since object
                # arrays can't be splatted, we have to force the conversion
                res.append((self.pdf[param](*cdf_rv.astype(numpy.float64))/self._pdf_norm[param], self.prior_pdf[param](*cdf_rv.astype(numpy.float64)), cdf_rv))

        #
        # Cache the samples we chose
        #
        if not save_no_samples:
         if len(self._rvs) == 0:
            self._rvs = dict(list(zip(args, rvs_tmp)))
         else:
            rvs_tmp = dict(list(zip(args, rvs_tmp)))
            #for p, ar in self._rvs.items():
            for p in self.params_ordered:
                self._rvs[p] = numpy.hstack( (self._rvs[p], rvs_tmp[p]) )

        #
        # Pack up the result if the user wants a dictonary instead
        #
        if "rdict" in kwargs:
            return dict(list(zip(args, res)))
        return list(zip(*res))

    def setup(self,n_bins=100,**kwargs):
        # Setup histogram data.  Used in portfolio
        for p in self.params_ordered:
            self.setup_hist_single_param(self.llim[p], self.rlim[p], n_bins, p)

    def update_sampling_prior(self,ln_weights, n_history,tempering_exp=1,log_scale_weights=True,floor_integrated_probability=0,external_rvs=None,**kwargs):
      """
      update_sampling_prior

      Default setting is for lnL. The choices adopted are fairly arbitrary. For simplicity, we are using 'log scale weights' (derived from lnL, not L)
      which lets us sample well away from the posterior at the expense of not being as strongly sampling near the peak.

      NOTE: Currently deployed for mcsamplerPortfolio, NOT yet part of core code here
      """
      # Allow updates provided from outside sources,
      rvs_here = self._rvs
      if external_rvs:
        rvs_here = external_rvs

      # apply tempering exponent (structurally slightly different than in low-level code - not just to likelihood)
      ln_weights  = self.xpy.array(ln_weights) # force copy
      ln_weights *= tempering_exp

      n_history_to_use = np.min([n_history, len(ln_weights), len(rvs_here[self.params_ordered[0]])] )

      # default is to use logarithmic (!) weights, relying on them being positive.
      weights_alt = ln_weights[-n_history_to_use:]  - self.xpy.max(ln_weights) + 100  
      weights_alt = self.xpy.maximum(weights_alt, 1e-5)    # prevent negative weights, in case integrating function with lnL < 0
      # now treat as sum
      weights_alt = weights_alt/(weights_alt.sum())
      if weights_alt.dtype == numpy.float128:
        weights_alt = weights_alt.astype(numpy.float64,copy=False)

      def function_wrapper(f, p):
          def inner(arg):
            return f(arg, p)
          return inner


      for itr, p in enumerate(self.params_ordered):
                # # FIXME: The second part of this condition should be made more
                # # specific to pinned parameters
                if p not in self.adaptive or p in list(kwargs.keys()):
                    continue

                points = rvs_here[p][-n_history_to_use:]
                self.compute_hist(points, p,weights=weights_alt,floor_level=floor_integrated_probability)
                self.pdf[p] = function_wrapper(self.pdf_from_hist, p)
                self.cdf_inv[p] = function_wrapper(self.cdf_inverse_from_hist, p)



    @profile
    def integrate_log(self, lnF, *args, xpy=xpy_default,**kwargs):
        """
        Integrate exp(lnF) returning lnI, by using n sample points, assuming integrand is lnF
        Does NOT allow for tuples of arguments, an unused feature in mcsampler

        tempering is done with lnF, suitably modified.

        kwargs:
        nmax -- total allowed number of sample points, will throw a warning if this number is reached before neff.
        neff -- Effective samples to collect before terminating. If not given, assume infinity
        n -- Number of samples to integrate in a 'chunk' -- default is 1000
        save_integrand -- Save the evaluated value of the integrand at the sample points with the sample point
        history_mult -- Number of chunks (of size n) to use in the adaptive histogramming: only useful if there are parameters with adaptation enabled
        tempering_exp -- Exponent to raise the weights of the 1-D marginalized histograms for adaptive sampling prior generation, by default it is 0 which will turn off adaptive sampling regardless of other settings
        temper_log -- Adapt in min(ln L, 10^(-5))^tempering_exp
        tempering_adapt -- Gradually evolve the tempering_exp based on previous history.
        floor_level -- *total probability* of a uniform distribution, averaged with the weighted sampled distribution, to generate a new sampled distribution
        n_adapt -- number of chunks over which to allow the pdf to adapt. Default is zero, which will turn off adaptive sampling regardless of other settings
        convergence_tests - dictionary of function pointers, each accepting self._rvs and self.params as arguments. CURRENTLY ONLY USED FOR REPORTING
        Pinning a value: By specifying a kwarg with the same of an existing parameter, it is possible to "pin" it. The sample draws will always be that value, and the sampling prior will use a delta function at that value.
        """

        # Setup histogram data
        n_bins = kwargs['n_bins'] if 'n_bins' in kwargs else 100
        for p in self.params_ordered:
            self.setup_hist_single_param(self.llim[p], self.rlim[p], n_bins, p)

        xpy_here = self.xpy

        #
        # Pin values
        #
        tempcdfdict, temppdfdict, temppriordict, temppdfnormdict = {}, {}, {}, {}
        temppdfnormdict = defaultdict(lambda: 1.0)
        for p, val in list(kwargs.items()):
            if p in self.params_ordered:
                # Store the previous pdf/cdf in case it's already defined
                tempcdfdict[p] = self.cdf_inv[p]
                temppdfdict[p] = self.pdf[p]
                temppdfnormdict[p] = self._pdf_norm[p]
                temppriordict[p] = self.prior_pdf[p]
                # Set a new one to always return the same value
                self.pdf[p] = functools.partial(delta_func_pdf_vector, val)
                self._pdf_norm[p] = 1.0
                self.prior_pdf[p] = functools.partial(delta_func_pdf_vector, val)
                self.cdf_inv[p] = functools.partial(delta_func_samp_vector, val)

        
        #
        # Determine stopping conditions
        #
        nmax = kwargs["nmax"] if "nmax" in kwargs else float("inf")
        neff = kwargs["neff"] if "neff" in kwargs else numpy.float128("inf")
        n = int(kwargs["n"] if "n" in kwargs else min(1000, nmax))
        convergence_tests = kwargs["convergence_tests"] if "convergence_tests" in kwargs else None
        save_no_samples = kwargs["save_no_samples"] if "save_no_samples" in kwargs else None


        #
        # Adaptive sampling parameters
        #
        n_history = int(kwargs["history_mult"]*n) if "history_mult" in kwargs else 2*n
        if n_history<=0:
            print("  Note: cannot adapt, no history ")

        tempering_exp = kwargs["tempering_exp"] if "tempering_exp" in kwargs else 0.0
        n_adapt = int(kwargs["n_adapt"]*n) if "n_adapt" in kwargs else 1000  # default to adapt to 1000 chunks, then freeze
        floor_integrated_probability = kwargs["floor_level"] if "floor_level" in kwargs else 0
        temper_log = kwargs["tempering_log"] if "tempering_log" in kwargs else False
        tempering_adapt = kwargs["tempering_adapt"] if "tempering_adapt" in kwargs else False
            

        save_intg = kwargs["save_intg"] if "save_intg" in kwargs else False
        # FIXME: The adaptive step relies on the _rvs cache, so this has to be
        # on in order to work
        if n_adapt > 0 and tempering_exp > 0.0:
            save_intg = True

        deltalnL = kwargs['igrand_threshold_deltalnL'] if 'igrand_threshold_deltalnL' in kwargs else float("Inf") # default is to return all
        deltaP    = kwargs["igrand_threshold_p"] if 'igrand_threshold_p' in kwargs else 0 # default is to omit 1e-7 of probability
        bFairdraw  = kwargs["igrand_fairdraw_samples"] if "igrand_fairdraw_samples" in kwargs else False
        n_extr = kwargs["igrand_fairdraw_samples_max"] if "igrand_fairdraw_samples_max" in kwargs else None

        bShowEvaluationLog = kwargs['verbose'] if 'verbose' in kwargs else False
        bShowEveryEvaluation = kwargs['extremely_verbose'] if 'extremely_verbose' in kwargs else False

        if bShowEvaluationLog:
            print(" .... mcsampler : providing verbose output ..... ")

        current_log_aggregate = None
        eff_samp = 0  # ratio of max weight to sum of weights
        maxlnL = -np.inf  # max lnL
        maxval=0   # max weight
        outvals=None  # define in top level scope
        self.ntotal = 0
        if bShowEvaluationLog:
            print("iteration Neff  sqrt(2*lnLmax) sqrt(2*lnLmarg) ln(Z/Lmax) int_var")

        if convergence_tests:
            bConvergenceTests = False   # start out not converged, if tests are on
            last_convergence_test = defaultdict(lambda: False)   # initialize record of tests
        else:
            bConvergenceTests = False    # if tests are not available, assume not converged. The other criteria will stop it
            last_convergence_test = defaultdict(lambda: False)   # need record of tests to be returned always
        n_zero_prior =0
        while (eff_samp < neff and self.ntotal < nmax): #  and (not bConvergenceTests):

            # Draw our sample points
            # Non-log draw
            joint_p_s, joint_p_prior, rv = self.draw_simplified(
                n, *self.params_ordered
            )

            #
            # Prevent zeroes in the sampling prior
            #
            # FIXME: If we get too many of these, we should bail
            if not(cupy_ok):  # don't do this if using cupy! 
              if any(joint_p_s <= 0):
                for p in self.params_ordered:
                    self._rvs[p] = identity_convert_togpu(numpy.resize(identity_convert(self._rvs[p]), len(self._rvs[p])-n))
                    self.cdf_inv = self.cdf_inv_initial
                    self.pdf = self.pdf_initial
                print("Zero prior value detected, skipping.", file=sys.stderr)
                print("Resetting sampling priors to initial values.", file=sys.stderr)
                n_zero_prior +=1
                if n_zero_prior > 5:
                  raise Exception('Zero prior failure', 'fail')
                continue

            #
            # Unpack rvs and evaluate integrand
            #
            if len(rv[0].shape) != 1:
                rv = rv[0]

            params = []
            for item in self.params_ordered:  # USE IN ORDER
                if isinstance(item, tuple):
                    params.extend(item)
                else:
                    params.append(item)
            unpacked = unpacked0 = rv #numpy.hstack([r.flatten() for r in rv]).reshape(len(args), -1)
            unpacked = dict(list(zip(params, unpacked)))

            # Evaluate function, protecting argument order
            if 'no_protect_names' in kwargs:
                lnL = lnF(*unpacked0)  # do not protect order
            else:
                lnL= lnF(**unpacked)  # protect order using dictionary
            # take log if we are NOT using lnL
            if cupy_ok:
              if not(isinstance(lnL,cupy.ndarray)):
                lnL = identity_convert_togpu(lnL)  # send to GPU, if not already there

            log_integrand =lnL + self.xpy.log(joint_p_prior) - self.xpy.log(joint_p_s)
            log_weights = tempering_exp*lnL + self.xpy.log(joint_p_prior) - self.xpy.log(joint_p_s)
            
            # append to cumulative values (assume all can be added)

            if save_intg:
                # FIXME: See warning at beginning of function. The prior values
                # need to be moved out of this, as they are not part of MC
                # integration
                if "log_integrand" in self._rvs:
                    self._rvs["log_integrand"] = xpy_here.hstack( (self._rvs["log_integrand"], lnL) )
                    self._rvs["log_joint_prior"] = xpy_here.hstack( (self._rvs["log_joint_prior"], self.xpy.log(joint_p_prior)) )
                    self._rvs["log_joint_s_prior"] = xpy_here.hstack( (self._rvs["log_joint_s_prior"], self.xpy.log(joint_p_s)))
                    self._rvs["log_weights"] = xpy_here.hstack( (self._rvs["log_weights"], log_weights ))
                else:
                    self._rvs["log_integrand"] = lnL
                    self._rvs["log_joint_prior"] = self.xpy.log(joint_p_prior)
                    self._rvs["log_joint_s_prior"] = self.xpy.log(joint_p_s)
                    self._rvs["log_weights"] = log_weights
            # maxlnL
            maxlnL_now = identity_convert(xpy.max(lnL))
            maxlnL = identity_convert(maxlnL)
            if np.isinf(maxlnL ):
              maxlnL = maxlnL_now
            else:
              maxlnL = np.max([maxlnL, maxlnL_now,-100])


            # n, Mean, error tracked by statutils structure
            if current_log_aggregate is None:
              current_log_aggregate = init_log(log_integrand,xpy=xpy,special=xpy_special_default)
            else:
              current_log_aggregate = update_log(current_log_aggregate, log_integrand,xpy=xpy,special=xpy_special_default)
            outvals = finalize_log(current_log_aggregate,xpy=xpy)
            self.ntotal = current_log_aggregate[0]
            # effective samples
            maxval = max(maxval, identity_convert(self.xpy.max(log_integrand) ))

            # sum of weights is the integral * the number of points
            eff_samp = xpy.exp(  outvals[0]+np.log(self.ntotal) - maxval)   # integral value minus floating point, which is maximum


            # Throw exception if we get infinity or nan
            if math.isnan(eff_samp):
                raise NanOrInf("Effective samples = nan")

            if bShowEvaluationLog:
                print(" :",  self.ntotal, eff_samp, numpy.sqrt(2*maxlnL), numpy.sqrt(2*outvals[0]), outvals[0]-maxlnL, np.exp(outvals[1]/2  - outvals[0]  - np.log(self.ntotal)/2 ))

            if (not convergence_tests) and self.ntotal >= nmax and neff != float("inf"):
                print("WARNING: User requested maximum number of samples reached... bailing.", file=sys.stderr)

            # Convergence tests:
            if convergence_tests:
                bConvergedThisIteration = True  # start out optimistic
                for key in list(convergence_tests.keys()):
                    last_convergence_test[key] =  convergence_tests[key](self._rvs, self.params_ordered)
                    bConvergedThisIteration = bConvergedThisIteration  and                      last_convergence_test[key]
                bConvergenceTests = bConvergedThisIteration

            if convergence_tests  and bShowEvaluationLog:  # Print status of each test
                for key in convergence_tests:
                    print("   -- Convergence test status : ", key, last_convergence_test[key])

            #
            # The total number of adaptive steps is reached
            #
            # FIXME: We need a better stopping condition here
            if self.ntotal > n_adapt*n:
                print(n_adapt,self.ntotal)
                continue

            #
            # Iterate through each of the parameters, updating the sampling
            # prior PDF according to the 1-D marginalization
            #
            def function_wrapper(f, p):
                def inner(arg):
                    return f(arg, p)
                return inner

            weights_alt = self._rvs["log_integrand"][-n_history:]+np.max([maxlnL, 200])  # try to make sure we have some dynamic range here
            weights_alt = self.xpy.maximum(weights_alt, 1e-5)  # prevent negative weights. NOTE THIS IS IMPORTANT: if you are integrating a function with lnL<0, use an offset!
            weights_alt = weights_alt/(weights_alt.sum())
            if weights_alt.dtype == numpy.float128:
              weights_alt = weights_alt.astype(numpy.float64,copy=False)

            for itr, p in enumerate(self.params_ordered):
                # # FIXME: The second part of this condition should be made more
                # # specific to pinned parameters
                if p not in self.adaptive or p in list(kwargs.keys()):
                    continue

                points = self._rvs[p][-n_history:]
                self.compute_hist(points, p,weights=weights_alt,floor_level=floor_integrated_probability)
                self.pdf[p] = function_wrapper(self.pdf_from_hist, p)
                self.cdf_inv[p] = function_wrapper(self.cdf_inverse_from_hist, p)

        # If we were pinning any values, undo the changes we did before
        self.cdf_inv.update(tempcdfdict)
        self.pdf.update(temppdfdict)
        self._pdf_norm.update(temppdfnormdict)
        self.prior_pdf.update(temppriordict)

        # Clean out the _rvs arrays for 'irrelevant' points
        #   - find and remove samples with  lnL less than maxlnL - deltalnL (latter user-specified)
        #   - create the cumulative weights
        #   - find and remove samples which contribute too little to the cumulative weights
        if (not save_no_samples) and ( "log_integrand" in self._rvs):
            self._rvs["sample_n"] = numpy.arange(len(self._rvs["log_integrand"]))  # create 'iteration number'        
            # Step 1: Cut out any sample with lnL belw threshold
            indx_list = [k for k, value in enumerate( (self._rvs["log_integrand"] > maxlnL - deltalnL)) if value] # threshold number 1
            # FIXME: This is an unncessary initial copy, the second step (cum i
            # prob) can be accomplished with indexing first then only pare at
            # the end
            for key in list(self._rvs.keys()):
                if isinstance(key, tuple):
                    self._rvs[key] = self._rvs[key][:,indx_list]
                else:
                    self._rvs[key] = self._rvs[key][indx_list]
            # Step 2: Create and sort the cumulative weights, among the remaining points, then use that as a threshold
            ln_wt = self._rvs["log_integrand"] + self._rvs["log_joint_prior"] - self._rvs["log_joint_s_prior"]
            # Convert to CPU as needed
            ln_wt = identity_convert(ln_wt)
            ln_wt += - np.max(ln_wt)  # remove maximum value, irrelevant
            wt = np.exp(ln_wt) # exponentiate.  Danger underflow
            idx_sorted_index = numpy.lexsort((numpy.arange(len(wt)), wt))  # Sort the array of weights, recovering index values
            indx_list = numpy.array( [[k, ln_wt[k]] for k in idx_sorted_index])     # pair up with the weights again. NOTE NOT INTEGER TYPE ANY MORE
            cum_sum = numpy.cumsum(indx_list[:,1])  # find the cumulative sum
            cum_sum = cum_sum/cum_sum[-1]          # normalize the cumulative sum
            indx_list = [int(indx_list[k, 0]) for k, value in enumerate(cum_sum > deltaP) if value]  # find the indices that preserve > 1e-7 of total probability. RECAST TO INTEGER
            # FIXME: See previous FIXME
            for key in list(self._rvs.keys()):
                if isinstance(key, tuple):
                    self._rvs[key] = self._rvs[key][:,indx_list]
                else:
                    self._rvs[key] = self._rvs[key][indx_list]

        # Do a fair draw of points, if option is set. CAST POINTS BACK TO NUMPY, IDEALLY
        if bFairdraw and not(n_extr is None):
           n_extr = int(numpy.min([n_extr,1.5*identity_convert(eff_samp),1.5*neff]))
           print(" Fairdraw size : ", n_extr)
           ln_wt = self.xpy.array(self._rvs["log_integrand"] + self._rvs["log_joint_prior"] - self._rvs["log_joint_s_prior"] ,dtype=float)
           ln_wt = identity_convert(ln_wt)  # send to CPU
           ln_wt += - special.logsumexp(ln_wt)
           wt = xpy.exp(identity_convert_togpu(ln_wt))
           if n_extr < len(self._rvs["log_integrand"]):
               indx_list = self.xpy.random.choice(self.xpy.arange(len(wt)), size=n_extr,replace=True,p=wt) # fair draw
               # FIXME: See previous FIXME
               for key in list(self._rvs.keys()):
                   if isinstance(key, tuple):
                       self._rvs[key] = identity_convert(self._rvs[key][:,indx_list])
                   else:
                       self._rvs[key] = identity_convert(self._rvs[key][indx_list])


        # Create extra dictionary to return things
        dict_return ={}
        if convergence_tests is not None:
            dict_return["convergence_test_results"] = last_convergence_test

        # perform type conversion of all stored variables
        if cupy_ok:
          for name in self._rvs:
            if isinstance(self._rvs[name],xpy_default.ndarray):
              self._rvs[name] = identity_convert(self._rvs[name])   # this is trivial if xpy_default is numpy, and a conversion otherwise

        # Return.  Take care of typing
        if outvals:
          out0 = outvals[0]; out1 = outvals[1]
          if not(isinstance(outvals[0], np.float64)):
            # type convert everything as needed
            out0 = identity_convert(out0)
          if not(isinstance(outvals[1], np.float64)):
            out1 = identity_convert(out1)
            eff_samp = identity_convert(eff_samp)
          return out0, out1 - np.log(self.ntotal), eff_samp, dict_return
        else: # very strange case where we terminate early
          return None, None, None, None



    #
    # FIXME: The priors are not strictly part of the MC integral, and so any
    # internal reference to them needs to be moved to a subclass which handles
    # the incovnenient part os doing the \int p/p_s L d\theta integral.
    #
    @profile
    def integrate(self, func, *args, **kwargs):
        """
        Integrate func, by using n sample points. Right now, all params defined must be passed to args must be provided, but this will change soon.
        Does NOT allow for tuples of arguments, an unused feature in mcsampler

        kwargs:
        nmax -- total allowed number of sample points, will throw a warning if this number is reached before neff.
        neff -- Effective samples to collect before terminating. If not given, assume infinity
        n -- Number of samples to integrate in a 'chunk' -- default is 1000
        save_integrand -- Save the evaluated value of the integrand at the sample points with the sample point
        history_mult -- Number of chunks (of size n) to use in the adaptive histogramming: only useful if there are parameters with adaptation enabled
        tempering_exp -- Exponent to raise the weights of the 1-D marginalized histograms for adaptive sampling prior generation, by default it is 0 which will turn off adaptive sampling regardless of other settings
        temper_log -- Adapt in min(ln L, 10^(-5))^tempering_exp
        tempering_adapt -- Gradually evolve the tempering_exp based on previous history.
        floor_level -- *total probability* of a uniform distribution, averaged with the weighted sampled distribution, to generate a new sampled distribution
        n_adapt -- number of chunks over which to allow the pdf to adapt. Default is zero, which will turn off adaptive sampling regardless of other settings
        convergence_tests - dictionary of function pointers, each accepting self._rvs and self.params as arguments. CURRENTLY ONLY USED FOR REPORTING
        Pinning a value: By specifying a kwarg with the same of an existing parameter, it is possible to "pin" it. The sample draws will always be that value, and the sampling prior will use a delta function at that value.
        """
        if "use_lnL" in kwargs:
          if kwargs["use_lnL"]:
            return self.integrate_log(func, **kwargs)  # pass it on, easier than mixed coding

        # Setup histogram data
        n_bins = 100
        for p in self.params_ordered:
            self.setup_hist_single_param(self.llim[p], self.rlim[p], n_bins, p)

        xpy_here = self.xpy

        #
        # Pin values
        #
        tempcdfdict, temppdfdict, temppriordict, temppdfnormdict = {}, {}, {}, {}
        temppdfnormdict = defaultdict(lambda: 1.0)
        for p, val in list(kwargs.items()):
            if p in self.params_ordered:
                # Store the previous pdf/cdf in case it's already defined
                tempcdfdict[p] = self.cdf_inv[p]
                temppdfdict[p] = self.pdf[p]
                temppdfnormdict[p] = self._pdf_norm[p]
                temppriordict[p] = self.prior_pdf[p]
                # Set a new one to always return the same value
                self.pdf[p] = functools.partial(delta_func_pdf_vector, val)
                self._pdf_norm[p] = 1.0
                self.prior_pdf[p] = functools.partial(delta_func_pdf_vector, val)
                self.cdf_inv[p] = functools.partial(delta_func_samp_vector, val)

        # put it back in the args
#        if 'args' in kwargs.keys():
#            args = kwargs['args']
        #args = tuple(list(args) + filter(lambda p: p in self.params, kwargs.keys()))
        # This is a semi-hack to ensure that the integrand is called with
        # the arguments in the right order
        # FIXME: How dangerous is this?
#        else:
#            args = func.func_code.co_varnames[:func.func_code.co_argcount]


        #if set(args) & set(params) != set(args):
        # DISABLE THIS CHECK
#        if not MCSampler.match_params_from_args(args, self.params):
#            raise ValueError("All integrand variables must be represented by integral parameters.")
        
        #
        # Determine stopping conditions
        #
        nmax = kwargs["nmax"] if "nmax" in kwargs else float("inf")
        neff = kwargs["neff"] if "neff" in kwargs else numpy.float128("inf")
        n = int(kwargs["n"] if "n" in kwargs else min(1000, nmax))
        convergence_tests = kwargs["convergence_tests"] if "convergence_tests" in kwargs else None
        save_no_samples = kwargs["save_no_samples"] if "save_no_samples" in kwargs else None


        #
        # Adaptive sampling parameters
        #
        n_history = int(kwargs["history_mult"]*n) if "history_mult" in kwargs else 2*n
        if n_history<=0:
            print("  Note: cannot adapt, no history ")

        tempering_exp = kwargs["tempering_exp"] if "tempering_exp" in kwargs else 0.0
        n_adapt = int(kwargs["n_adapt"]*n) if "n_adapt" in kwargs else 1000*n
        floor_integrated_probability = kwargs["floor_level"] if "floor_level" in kwargs else 0
        temper_log = kwargs["tempering_log"] if "tempering_log" in kwargs else False
        tempering_adapt = kwargs["tempering_adapt"] if "tempering_adapt" in kwargs else False
        if not tempering_adapt:
            tempering_exp_running=tempering_exp
        else:
            print(" Adaptive tempering ")
            #tempering_exp_running=0.01  # decent place to start for the first step. Note starting at zero KEEPS it at zero.
            tempering_exp_running=tempering_exp
            

        save_intg = kwargs["save_intg"] if "save_intg" in kwargs else False
        # FIXME: The adaptive step relies on the _rvs cache, so this has to be
        # on in order to work
        if n_adapt > 0 and tempering_exp > 0.0:
            save_intg = True

        deltalnL = kwargs['igrand_threshold_deltalnL'] if 'igrand_threshold_deltalnL' in kwargs else float("Inf") # default is to return all
        deltaP    = kwargs["igrand_threshold_p"] if 'igrand_threshold_p' in kwargs else 0 # default is to omit 1e-7 of probability
        bFairdraw  = kwargs["igrand_fairdraw_samples"] if "igrand_fairdraw_samples" in kwargs else False
        n_extr = kwargs["igrand_fairdraw_samples_max"] if "igrand_fairdraw_samples_max" in kwargs else None

        bShowEvaluationLog = kwargs['verbose'] if 'verbose' in kwargs else False
        bShowEveryEvaluation = kwargs['extremely_verbose'] if 'extremely_verbose' in kwargs else False

        if bShowEvaluationLog:
            print(" .... mcsampler : providing verbose output ..... ")

        int_val1 = numpy.float128(0)
        self.ntotal = 0
        maxval = -float("Inf")
        maxlnL = -float("Inf")
        eff_samp = 0
        mean, var = None, numpy.float128(0)    # to prevent infinite variance due to overflow
        if cupy_ok:
          var = xpy_default.float64(0)   # cupy doesn't have float128

        if bShowEvaluationLog:
            print("iteration Neff  sqrt(2*lnLmax) sqrt(2*lnLmarg) ln(Z/Lmax) int_var")

        if convergence_tests:
            bConvergenceTests = False   # start out not converged, if tests are on
            last_convergence_test = defaultdict(lambda: False)   # initialize record of tests
        else:
            bConvergenceTests = False    # if tests are not available, assume not converged. The other criteria will stop it
            last_convergence_test = defaultdict(lambda: False)   # need record of tests to be returned always
        n_zero_prior =0
        while (eff_samp < neff and self.ntotal < nmax): #  and (not bConvergenceTests):
            # Draw our sample points
            joint_p_s, joint_p_prior, rv = self.draw_simplified(
                n, *self.params_ordered
            )

            #
            # Prevent zeroes in the sampling prior
            #
            # FIXME: If we get too many of these, we should bail
            if not(cupy_ok):  # don't do this if using cupy! 
              if any(joint_p_s <= 0):
                for p in self.params_ordered:
                    self._rvs[p] = identity_convert_togpu(numpy.resize(identity_convert(self._rvs[p]), len(self._rvs[p])-n))
                    self.cdf_inv = self.cdf_inv_initial
                    self.pdf = self.pdf_initial
                print("Zero prior value detected, skipping.", file=sys.stderr)
                print("Resetting sampling priors to initial values.", file=sys.stderr)
                n_zero_prior +=1
                if n_zero_prior > 5:
                  raise Exception('Zero prior failure', 'fail')
                continue

            #
            # Unpack rvs and evaluate integrand
            #
            if len(rv[0].shape) != 1:
                rv = rv[0]

            params = []
            for item in self.params_ordered:  # USE IN ORDER
                if isinstance(item, tuple):
                    params.extend(item)
                else:
                    params.append(item)
            unpacked = unpacked0 = rv #numpy.hstack([r.flatten() for r in rv]).reshape(len(args), -1)
            unpacked = dict(list(zip(params, unpacked)))
            if 'no_protect_names' in kwargs:
                fval = func(*unpacked0)  # do not protect order
            else:
                fval = func(**unpacked) # Chris' original plan: note this insures the function arguments are tied to the parameters, using a dictionary. 

            if cupy_ok:
              if not(isinstance(fval,cupy.ndarray)):
                fval = identity_convert_togpu(fval)  # send to GPU, if not already there

            #
            # Check if there is any practical contribution to the integral
            #
            # FIXME: While not technically a fatal error, this will kill the 
            # adaptive sampling
            if not(cupy_ok): # only do this check if not on GPU
              if fval.sum() == 0:
                for p in self.params_ordered:
                    self._rvs[p] = numpy.resize(self._rvs[p], len(self._rvs[p])-n)
                print("No contribution to integral, skipping.", file=sys.stderr)
                continue

            if save_intg:
                # FIXME: The joint_prior, if not specified is set to one and
                # will come out as a scalar here, hence the hack
                if not isinstance(joint_p_prior, xpy_here.ndarray):
                    joint_p_prior = xpy_here.ones(fval.shape)*joint_p_prior


#                print "      Prior", type(joint_p_prior), joint_p_prior.dtype

                # FIXME: See warning at beginning of function. The prior values
                # need to be moved out of this, as they are not part of MC
                # integration
                if "integrand" in self._rvs:
                    self._rvs["integrand"] = xpy_here.hstack( (self._rvs["integrand"], fval) )
                    self._rvs["joint_prior"] = xpy_here.hstack( (self._rvs["joint_prior"], joint_p_prior) )
                    self._rvs["joint_s_prior"] = xpy_here.hstack( (self._rvs["joint_s_prior"], joint_p_s) )
                    self._rvs["weights"] = xpy_here.hstack( (self._rvs["joint_s_prior"], fval*joint_p_prior/joint_p_s) )
                else:
                    self._rvs["integrand"] = fval
                    self._rvs["joint_prior"] = joint_p_prior
                    self._rvs["joint_s_prior"] = joint_p_s
                    self._rvs["weights"] = fval*joint_p_prior/joint_p_s

            # Calculate the integral over this chunk
            int_val = fval * joint_p_prior / joint_p_s

            if bShowEveryEvaluation:
                for i in range(n):
                    print(" Evaluation details: p,ps, L = ", joint_p_prior[i], joint_p_s[i], fval[i])

            # Calculate max L (a useful convergence feature) for debug 
            # reporting.  Not used for integration
            # Try to avoid nan's
            maxlnL = numpy.log(numpy.max([numpy.exp(maxlnL), identity_convert(xpy_here.max(fval)),numpy.exp(-100)]))   # note if f<0, this will return nearly 0

            # Calculate the effective samples via max over the current 
            # evaluations
            maxval = max(maxval, identity_convert(int_val[0])) if int_val[0] != 0 else maxval
            maxval = identity_convert(max(maxval,xpy_here.amax(int_val)))
            #for v in int_val[1:]:
            #    maxval.append( v if v > maxval[-1] and v != 0 else maxval[-1] )

            # running variance
#            var = cumvar(identity_convert(int_val), mean, var, int(self.ntotal))[-1]
            if var is None:
              var=0
            if mean is None:
              mean=identity_convert_togpu(0.0)
            current_aggregate = [int(self.ntotal),mean, (self.ntotal-1)*var]
            current_aggregate = update(current_aggregate, int_val,xpy=xpy_default)
            outvals = finalize(current_aggregate)
#            print(var, outvals[-1])
            var = outvals[-1]

            # running integral
            int_val1 += identity_convert(int_val.sum())
            # running number of evaluations
            self.ntotal += n
            # FIXME: Likely redundant with int_val1
            mean = identity_convert_togpu(xpy_default.float64(int_val1/self.ntotal))

            # this test should not be required (!), but ...
            if not(np.isinf(maxval)):
              eff_samp = int_val1/maxval
            else:
              eff_samp = 1 # fallback in case of insanity

            # Throw exception if we get infinity or nan
            if math.isnan(eff_samp):
                raise NanOrInf("Effective samples = nan")
            if maxlnL is float("Inf"):
                raise NanOrInf("maxlnL = inf")

            if bShowEvaluationLog:
                var_print = identity_convert(var)
                print(" :",  self.ntotal, eff_samp, numpy.sqrt(2*maxlnL), numpy.sqrt(2*numpy.log(int_val1/self.ntotal)), numpy.log(int_val1/self.ntotal)-maxlnL, numpy.sqrt(var_print*self.ntotal)/int_val1)

            if (not convergence_tests) and self.ntotal >= nmax and neff != float("inf"):
                print("WARNING: User requested maximum number of samples reached... bailing.", file=sys.stderr)

            # Convergence tests:
            if convergence_tests:
                bConvergedThisIteration = True  # start out optimistic
                for key in list(convergence_tests.keys()):
                    last_convergence_test[key] =  convergence_tests[key](self._rvs, self.params_ordered)
                    bConvergedThisIteration = bConvergedThisIteration  and                      last_convergence_test[key]
                bConvergenceTests = bConvergedThisIteration

            if convergence_tests  and bShowEvaluationLog:  # Print status of each test
                for key in convergence_tests:
                    print("   -- Convergence test status : ", key, last_convergence_test[key])

            #
            # The total number of adaptive steps is reached
            #
            # FIXME: We need a better stopping condition here
            if self.ntotal > n_adapt:
#                print(self.ntotal, n_adapt)
                print(" ... skipping adaptation in late iterations .. ")
                continue

            #
            # Iterate through each of the parameters, updating the sampling
            # prior PDF according to the 1-D marginalization
            #
            def function_wrapper(f, p):
                def inner(arg):
                    return f(arg, p)
                return inner

            if not(save_intg):
                print("Direct access ")
                weights_alt = int_vals**tempering_exp
            elif not(tempering_exp):  # zero value, should not happen but just in case, fall back to using integrand for adaptation
                weights_alt = self._rvs["integrand"][-n_history:]
            else:
                #print(fval, self._rvs["integrand"][-n_history:])
                if temper_log:
#                  weights_alt =self.xpy.power(self.xpy.log((self._rvs["integrand"][-n_history:]/self._rvs["joint_s_prior"][-n_history:]*self._rvs["joint_prior"][-n_history:])), tempering_exp)
                  weights_alt =self.xpy.log(self._rvs["integrand"][-n_history:])
#                  weights_alt = self.xpy.nan_to_num(weights_alt,copy=False,posinf=1e3) # remove infinity, which oddly can occur?
                  weights_alt = self.xpy.maximum(weights_alt, 1e-5)  # prevent negative weights
                else:
                  weights_alt =((self._rvs["integrand"][-n_history:]/self._rvs["joint_s_prior"][-n_history:]*self._rvs["joint_prior"][-n_history:])**tempering_exp )
                  weights_alt = self.xpy.maximum(weights_alt,10)   # preventing too little dynamic range

            weights_alt = weights_alt/(weights_alt.sum())
            # Type convert as needed: if weights are float128, convert to float64; otherwise we hit a typing error later with bincount
            if weights_alt.dtype == numpy.float128:
              weights_alt = weights_alt.astype(numpy.float64,copy=False)
#            weights_alt = floor_integrated_probability*xpy_default.ones(len(weights_alt))/len(weights_alt) + (1-floor_integrated_probability)*weights_alt

            for itr, p in enumerate(self.params_ordered):
                # # FIXME: The second part of this condition should be made more
                # # specific to pinned parameters
                if p not in self.adaptive or p in list(kwargs.keys()):
                    continue

                points = self._rvs[p][-n_history:]
                self.compute_hist(points, p,weights=weights_alt,floor_level=floor_integrated_probability)
            #    if p == 'declination':
            #          vals = identity_convert(self.histogram_values[p])
            #          print(vals)
            #          print(np.mean(vals),np.std(vals))
                self.pdf[p] = function_wrapper(self.pdf_from_hist, p)
                self.cdf_inv[p] = function_wrapper(self.cdf_inverse_from_hist, p)

        # If we were pinning any values, undo the changes we did before
        self.cdf_inv.update(tempcdfdict)
        self.pdf.update(temppdfdict)
        self._pdf_norm.update(temppdfnormdict)
        self.prior_pdf.update(temppriordict)

        # Clean out the _rvs arrays for 'irrelevant' points
        #   - find and remove samples with  lnL less than maxlnL - deltalnL (latter user-specified)
        #   - create the cumulative weights
        #   - find and remove samples which contribute too little to the cumulative weights
        if (not save_no_samples) and ( "integrand" in self._rvs):
            self._rvs["sample_n"] = numpy.arange(len(self._rvs["integrand"]))  # create 'iteration number'        
            if deltalnL < 1e10:
              # Step 1: Cut out any sample with lnL belw threshold
              indx_list = [k for k, value in enumerate( (self._rvs["integrand"] > maxlnL - deltalnL)) if value] # threshold number 1
              # FIXME: This is an unncessary initial copy, the second step (cum i
              # prob) can be accomplished with indexing first then only pare at
              # the end
              for key in list(self._rvs.keys()):
                if isinstance(key, tuple):
                    self._rvs[key] = self._rvs[key][:,indx_list]
                else:
                    self._rvs[key] = self._rvs[key][indx_list]
            # Step 2: Create and sort the cumulative weights, among the remaining points, then use that as a threshold
            wt = self._rvs["integrand"]*self._rvs["joint_prior"]/self._rvs["joint_s_prior"]
            idx_sorted_index = numpy.lexsort((numpy.arange(len(wt)), wt))  # Sort the array of weights, recovering index values
            indx_list = numpy.array( [[k, wt[k]] for k in idx_sorted_index])     # pair up with the weights again. NOTE NOT INTEGER TYPE ANY MORE
            cum_sum = numpy.cumsum(indx_list[:,1])  # find the cumulative sum
            cum_sum = cum_sum/cum_sum[-1]          # normalize the cumulative sum
            indx_list = [int(indx_list[k, 0]) for k, value in enumerate(cum_sum > deltaP) if value]  # find the indices that preserve > 1e-7 of total probability. RECAST TO INTEGER
            # FIXME: See previous FIXME
            for key in list(self._rvs.keys()):
                if isinstance(key, tuple):
                    self._rvs[key] = self._rvs[key][:,indx_list]
                else:
                    self._rvs[key] = self._rvs[key][indx_list]

        # Do a fair draw of points, if option is set. CAST POINTS BACK TO NUMPY, IDEALLY
        if bFairdraw and not(n_extr is None):
           n_extr = int(numpy.min([n_extr,1.5*eff_samp,1.5*neff]))
           print(" Fairdraw size : ", n_extr)
           wt = self.xpy.array(self._rvs["integrand"]*self._rvs["joint_prior"]/self._rvs["joint_s_prior"]/self.xpy.max(self._rvs["integrand"]),dtype=float)
           wt *= 1.0/self.xpy.sum(wt)
           if n_extr < len(self._rvs["integrand"]):
               indx_list = self.xpy.random.choice(self.xpy.arange(len(wt)), size=n_extr,replace=True,p=wt) # fair draw
               # FIXME: See previous FIXME
               for key in list(self._rvs.keys()):
                   if isinstance(key, tuple):
                       self._rvs[key] = identity_convert(self._rvs[key][:,indx_list])
                   else:
                       self._rvs[key] = identity_convert(self._rvs[key][indx_list])

        # Create extra dictionary to return things
        dict_return ={}
        if convergence_tests is not None:
            dict_return["convergence_test_results"] = last_convergence_test

        # perform type conversion of all stored variables
        if cupy_ok:
          for name in self._rvs:
            if isinstance(self._rvs[name],xpy_default.ndarray):
              self._rvs[name] = identity_convert(self._rvs[name])   # this is trivial if xpy_default is numpy, and a conversion otherwise

        return int_val1/self.ntotal, identity_convert(var)/self.ntotal, eff_samp, dict_return

### UTILITIES: Predefined distributions
#  Be careful: vectorization is not always implemented consistently in new versions of numpy
def uniform_samp(a, b, x):   # I prefer to vectorize with the same call for all functions, rather than hardcode vectorization
        if  x>a and x<b:
                return 1.0/(b-a)
        else:
                return 0
def uniform_samp_cdf_inv_vector(a,b,p):
    # relies on input being a numpy array, with range from 0 to 1
    out= p.copy()
    out = p*(b-a) + a
    return out
#uniform_samp_vector = numpy.vectorize(uniform_samp,excluded=['a','b'],otypes=[numpy.float])
#uniform_samp_vector = numpy.vectorize(uniform_samp,otypes=[numpy.float])
# def uniform_samp_vector(a,b,x):
#    """
#    uniform_samp_vector:
#       Implement uniform sampling with np primitives, not np.vectorize !
#    Note NO cupy implementation yet
#    """
#    return numpy.heaviside(x-a,0)*numpy.heaviside(b-x,0)/(b-a)
def uniform_samp_vector(a,b,x,xpy=xpy_default):
   """
   uniform_samp_vector_lazy:
      Implement uniform sampling as multiplication by a constant.
      Much faster and lighter weight. We never use the cutoffs anyways, because the limits are hardcoded elsewhere.
   """
   return xpy.ones(len(x))/(b-a)  # requires the variable in range.  Needed because there is no cupy implementation of np.heavyside
# if cupy_ok:
#    uniform_samp_vector = uniform_samp_vector_lazy  

def ret_uniform_samp_vector_alt(a,b):
    return lambda x: xpy_default.ones(len(x))/(b-a)
#    return lambda x: 1./(b-a)


def uniform_samp_withfloor_vector(rmaxQuad,rmaxFlat,pFlat,x,xpy=xpy_default):
    if isinstance(x, float):
        ret =0.
        if x<rmaxQuad:
            ret+= (1-pFlat)/rmaxQuad
        if x<rmaxFlat:
            ret +=pFlat/rmaxFlat
        return  ret
    ret = xpy.zeros(x.shape,dtype=numpy.float64)
    ret += xpy.select([x<rmaxQuad],[(1.-pFlat)/rmaxQuad])
    ret += xpy.select([x<rmaxFlat],[pFlat/rmaxFlat])
    return ret



# syntatic sugar : predefine the most common distributions
def uniform_samp_phase(x,xpy=xpy_default):
   """
   Assume range known as 0,2pi
   """
   return xpy.ones(len(x))/(2*cupy_pi) 
def uniform_samp_psi(x,xpy=xpy_default):
   """
   Assume range known as 0,pi
   """
   return xpy.ones(len(x))/(cupy_pi) 
def uniform_samp_theta(x,xpy=xpy_default):
   """
   Assume range known as 
   """
   return xpy.sin(x)/(2.) 
def uniform_samp_dec(x,xpy=xpy_default):
   """
   Assume range known as 
   """
   return xpy.cos(x)/(2.) 


def cos_samp(x,xpy=xpy_default):
        return xpy.sin(x)/2   # x from 0, pi

def dec_samp(x,xpy=xpy_default):
        return xpy.sin(x+cupy_pi/2)/2   # x from 0, pi

cos_samp_vector = cos_samp
dec_samp_vector = dec_samp
def cos_samp_cdf_inv_vector(p,xpy=xpy_default):
    return xpy.arccos( 2*p-1)   # returns from 0 to pi
def dec_samp_cdf_inv_vector(p,xpy=xpy_default):
    return xpy.arccos(2*p-1) - xpy.pi/2  # target from -pi/2 to pi/2




# Mass ratio. PDF propto 1/(1+q)^2.  Defined so mass ratio is < 1
# expr = Integrate[1/(1 + q)^2, q]
# scale = (expr /. q -> qmax )  - (expr /. q -> qmin)
# (expr - (expr /. q -> qmin))/scale == x // Simplify
# q /. Solve[%, q][[1]] // Simplify
# % // CForm
def q_samp_vector(qmin,qmax,x):
    scale = 1./(1+qmin) - 1./(1+qmax)
    return 1/numpy.power((1+x),2)/scale
def q_cdf_inv_vector(qmin,qmax,x,xpy=xpy_default):
    return np.array((qmin + qmax*qmin + qmax*x - qmin*x)/(1 + qmax - qmax*x + qmin*x),dtype=np.float128)

# total mass. Assumed used with q.  2M/Mmax^2-Mmin^2
def M_samp_vector(Mmin,Mmax,x):
    scale = 2./(Mmax**2 - Mmin**2)
    return x*scale



def pseudo_dist_samp(r0,r):
        return r*r*numpy.exp( - (r0/r)*(r0/r)/2. + r0/r)+0.01  # put a floor on probability, so we converge. Note this floor only cuts out NEARBY distances

#pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,excluded=['r0'],otypes=[numpy.float64])
pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,otypes=[numpy.float64])

if xpy_default == numpy:
  def delta_func_pdf(x_0, x):
     return 1.0 if x == x_0 else 0.0
  delta_func_pdf_vector = xpy_default.vectorize(delta_func_pdf, otypes=[xpy_default.float64])
else:
   # only called on data which already satisfies the constraint, so return 1
   delta_func_pdf_vector = lambda x,y: xpy_default.ones(y.shape)

if xpy_default==numpy:
 def delta_func_samp(x_0, x):
     return x_0
 delta_func_samp_vector = xpy_default.vectorize(delta_func_samp, otypes=[xpy_default.float64])
else:
 # return value equal to the requested floating point constant
 delta_func_samp_vector = lambda x,y: x*xpy_default.ones(y.shape)


def sanityCheckSamplerIntegrateUnity(sampler,*args,**kwargs):
        return sampler.integrate(lambda *args: 1,*args,**kwargs)

###
### CONVERGENCE TESTS
###


# neff by another name:
#    - value: tests for 'smooth' 1-d cumulative distributions
#    - require  require the most significant-weighted point be less than p of all cumulative probability
#    - this test is *equivalent* to neff > 1/p
#    - provided to illustrate the interface
def convergence_test_MostSignificantPoint(pcut, rvs, params):
    weights = rvs["weights"] #rvs["integrand"]* rvs["joint_prior"]/rvs["joint_s_prior"]
    indxmax = numpy.argmax(weights)
    wtSum = numpy.sum(weights)
    return  weights[indxmax]/wtSum < pcut


# normality test: is the MC integral normally distributed, with a small standard deviation?
#    - value: tests for converged integral
#    - arguments: 
#         - ncopies:               # of sub-integrals
#         - pcutNormalTest     Threshold p-value for normality test
#         - sigmaCutErrorThreshold   Threshold relative error in the integral
#    - implement normality test on **log(integral)** since the log should also be normally distributed if well converged
#           - this helps us handle large orders-of-magnitude differences
#           - compatible with a *relative* error threshold on integral
#           - only works for *positive-definite* integrands
#    - other python normality tests:  
#          scipy.stats.shapiro
#          scipy.stats.anderson
#  WARNING:
#    - this test assumes *unsorted* past history: the 'ncopies' segments are assumed independent.
import scipy.stats as stats
def convergence_test_NormalSubIntegrals(ncopies, pcutNormalTest, sigmaCutRelativeErrorThreshold, rvs, params):
    weights = rvs["integrand"]* rvs["joint_prior"]/rvs["joint_s_prior"]  # rvs["weights"] # rvs["weights"] is *sorted* (side effect?), breaking test. Recalculated weights are not.  Use explicitly calculated weights until sorting effect identified
#    weights = weights /numpy.sum(weights)    # Keep original normalization, so the integral values printed to stdout have meaning relative to the overall integral value.  No change in code logic : this factor scales out (from the log, below)
    igrandValues = numpy.zeros(ncopies)
    len_part = int(len(weights)/ncopies)  # deprecated: np.floor->np.int
    for indx in numpy.arange(ncopies):
        igrandValues[indx] = numpy.log(numpy.mean(weights[indx*len_part:(indx+1)*len_part]))  # change to mean rather than sum, so sub-integrals have meaning
    igrandValues= numpy.sort(igrandValues)#[2:]                            # Sort.  Useful in reports 
    valTest = stats.normaltest(igrandValues)[1]                              # small value is implausible
    igrandSigma = (numpy.std(igrandValues))/numpy.sqrt(ncopies)   # variance in *overall* integral, estimated from variance of sub-integrals
    print(" Test values on distribution of log evidence:  (gaussianity p-value; standard deviation of ln evidence) ", valTest, igrandSigma)
    print(" Ln(evidence) sub-integral values, as used in tests  : ", igrandValues)
    return valTest> pcutNormalTest and igrandSigma < sigmaCutRelativeErrorThreshold   # Test on left returns a small value if implausible. Hence pcut ->0 becomes increasingly difficult (and requires statistical accidents). Test on right requires relative error in integral also to be small when pcut is small.   FIXME: Give these variables two different names
