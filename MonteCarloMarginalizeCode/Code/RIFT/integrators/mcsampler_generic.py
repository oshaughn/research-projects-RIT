import sys
import math
#import bisect
from collections import defaultdict
from types import ModuleType

import numpy
np=numpy #import numpy as np
from scipy import integrate, interpolate, special
import itertools
import functools

xpy_default= numpy


class MCSamplerGeneric(object):
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




    def __init__(self,**kwargs):

        # Total number of samples drawn
        self.ntotal = 0
        # Parameter names
        self.params = set()
        self.params_ordered = []  # keep them in order. Important to break likelihood function need for names
        # If the pdfs aren't normalized, this will hold the normalization 
        # Cache for the sampling points
        self._rvs = {}
        # parameter -> cdf^{-1} function object
        # params for left and right limits
        self.llim, self.rlim = {}, {}


        self.adaptive =[]

        # sampling distribution
        self.pdf = {}
        self.cdf_inv = {}
        # prior
        self.prior_pdf = {}


        # histogram setup
        self.xpy = numpy
        self.identity_convert = lambda x: x  # if needed, convert to numpy format  (e.g, cupy.asnumpy)
        self.identity_convert_togpu = lambda x: x  # if needed, convert to numpy format  (e.g, cupy.asnumpy)

        # extra args, created during setup
        self.extra_args = {}

    def add_parameter(self, params, pdf,   left_limit=None, right_limit=None, prior_pdf=None, adaptive_sampling=False,**kwargs):
        """
        Add one (or more) parameters to sample dimensions. params is either a string describing the parameter, or a tuple of strings. The tuple will indicate to the sampler that these parameters must be sampled together. left_limit and right_limit are on the infinite interval by default, but can and probably should be specified. If several params are given, left_limit, and right_limit must be a set of tuples with corresponding length. Sampling PDF is required, and if not provided, the cdf inverse function will be determined numerically from the sampling PDF.
        """
        self.params.add(params) # does NOT preserve order in which parameters are provided
        self.params_ordered.append(params)
        # update dictionary limits
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
        self.prior_pdf[params] = prior_pdf

        if adaptive_sampling:
            print("   Adapting ", params)
            self.adaptive.append(params)


    def setup(self,  **kwargs):
        self.extra_args =kwargs  # may need to pass/use during the 'update' step

    def update_sampling_prior(self,ln_weights, n_history,tempering_exp=1,log_scale_weights=True,floor_integrated_probability=0,external_rvs=None,**kwargs):
      """
      update_sampling_prior

      Default setting is for lnL. The choices adopted are fairly arbitrary. For simplicity, we are using 'log scale weights' (derived from lnL, not L)
      which lets us sample well away from the posterior at the expense of not being as strongly sampling near the peak.

      NOTE: Currently deployed for mcsamplerPortfolio, NOT yet part of core code here
      """
      raise Exception(" - mcsampler_generic : not implemented: update_sampling_prior ")

    def draw_simplified(self,n_samples, *args, **kwargs):
        """
        draw n_samples

        Should not be implemented in generic class
        """
        #return joint_p_s, joint_p_prior, rv
        raise Exception(" - mcsampler_generic : not implemented: draw")

    def integrate(self, lnF, *args, xpy=xpy_default,**kwargs):
        use_lnL = kwargs['use_lnL'] if 'use_lnL' in kwargs else False
        raise Exception(" - mcsampler_generic : not implemented: integrate")
        

    def integrate_log(self, lnF, *args, xpy=xpy_default,**kwargs):
        raise Exception(" - mcsampler_generic : not implemented: integrate_log")
        xpy_here = self.xpy
        
