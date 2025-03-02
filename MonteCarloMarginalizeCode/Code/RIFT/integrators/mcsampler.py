

import sys
import math
#import bisect
from collections import defaultdict

import numpy
from scipy import integrate, interpolate
from ..integrators.statutils import cumvar, welford, update, finalize
import itertools
import functools


import os
if not( 'RIFT_LOWLATENCY'  in os.environ):
    # Dont support selected external packages in low latency
 try:
    import healpy
 except:
    print(" - No healpy - ")
 try:
    import vegas
 except:
    print(" - No vegas - ")


from multiprocessing import Pool


__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

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
        # If the pdfs aren't normalized, this will hold the normalization 
        # constant
        self._pdf_norm = defaultdict(lambda: 1)
        # Cache for the sampling points
        self._rvs = {}
        # parameter -> cdf^{-1} function object
        self.cdf = {}
        self.cdf_inv = {}
        # params for left and right limits
        self.llim, self.rlim = {}, {}
        # Keep track of the adaptive parameters
        self.adaptive = []

        # Keep track of the adaptive parameter 1-D marginalizations
        self._hist = {}

        # MEASURES (=priors): ROS needs these at the sampler level, to clearly separate their effects
        # ASSUMES the user insures they are normalized
        self.prior_pdf = {}

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
        self.cdf_inv[params] = cdf_inv or self.cdf_inverse(params)
        if not isinstance(params, tuple):
            self.cdf[params] =  self.cdf_function(params)
            if prior_pdf is None:
                self.prior_pdf[params] = lambda x:1
            else:
                self.prior_pdf[params] = prior_pdf
        else:
            self.prior_pdf[params] = prior_pdf

        if adaptive_sampling:
            print("   Adapting ", params)
            self.adaptive.append(params)


    def cdf_function(self, param):
        """
        Numerically determine the  CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF.
        """
        # Solve P'(x) == p(x), with P[lower_boun] == 0
        def dP_cdf(p, x):
            if x > self.rlim[param] or x < self.llim[param]:
                return 0
            return numpy.float64(self.pdf[param](x))
        x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000,dtype=numpy.float64)
        # Integrator needs to have a step size which doesn't step over the
        # probability mass
        # TODO: Determine h_max.
        cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.01*numpy.float64(self.rlim[param]-self.llim[param])).T[0]  # issue of type limits for odeint
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
            if x > self.rlim[param] or x < self.llim[param]:
                return 0
            return numpy.float64(self.pdf[param](x))
        x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000,dtype=numpy.float64)
        # Integrator needs to have a step size which doesn't step over the
        # probability mass
        # TODO: Determine h_max.
        cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.01*numpy.float64(self.rlim[param]-self.llim[param])).T[0]
        if cdf[-1] != 1.0: # original pdf wasn't normalized
            self._pdf_norm[param] = cdf[-1]
            cdf /= cdf[-1]
        # Interpolate the inverse
        return interpolate.interp1d(cdf, x_i)

    def draw(self, rvs, *args, **kwargs):
        """
        Draw a set of random variates for parameter(s) args. Left and right limits are handed to the function. If args is None, then draw *all* parameters. 'rdict' parameter is a boolean. If true, returns a dict matched to param name rather than list. rvs must be either a list of uniform random variates to transform for sampling, or an integer number of samples to draw.
        """
        if len(args) == 0:
            args = self.params

        no_cache_samples = kwargs["no_cache_samples"] if "no_cache_samples" in kwargs else False


        if isinstance(rvs, int) or isinstance(rvs, float):
            #
            # Convert all arguments to tuples
            #
            # FIXME: UGH! Really? This was the most elegant thing you could come
            # up with?
            rvs_tmp = [numpy.random.uniform(0,1,(len(p), int(rvs))) for p in [(i,) if not isinstance(i, tuple) else i for i in args]]
            rvs_tmp = numpy.array([self.cdf_inv[param](*rv) for (rv, param) in zip(rvs_tmp, args)], dtype=object)
        else:
            rvs_tmp = numpy.array(rvs)

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
        if not no_cache_samples:  # more efficient memory usage. Note adaptation will not wor
          if len(self._rvs) == 0:
            self._rvs = dict(list(zip(args, rvs_tmp)))
          else:
            rvs_tmp = dict(list(zip(args, rvs_tmp)))
            #for p, ar in self._rvs.items():
            for p in self.params_ordered:
                self._rvs[p] = numpy.hstack( (self._rvs[p], rvs_tmp[p]) )
        else:  
            # if we are not caching samples, DELETE the sample record.  Saves memory!
            if len(self._rvs) >0:
                for p in self.params_ordered:
                    del self._rvs[p]
            self._rvs = {}

        #
        # Pack up the result if the user wants a dictonary instead
        #
        if "rdict" in kwargs:
            return dict(list(zip(args, res)))
        return list(zip(*res))


    def integrate_vegas(self, func, *args, **kwargs):
        """
        Uses vegas to do the integral.  Does not return sample points
        Remember:   pdf, cdf_inv refer to the *sampling* prior, so I need to multiply the integrand by a PDF ratio product!
        """
        # Method: use loop over params (=args), 
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
        nmax = kwargs["nmax"] if "nmax" in kwargs else 1e6
        neff = kwargs["neff"] if "neff" in kwargs else 1000
        n = int(kwargs["n"]) if "n" in kwargs else min(1000, nmax)  # chunk size
        nBlocks = 10
        n_itr = numpy.max([10,numpy.min([20,int(nmax/nBlocks/n)])])  # largest number to use


        # What I am actually doing: an n-dimensional integral with vegas, using the CDF function to generate the parameter
        # range.
        paramListDefault = kwargs['param_order'] # do not try to get it from the function
        strToEval = "lambda x: func("
        for indx in numpy.arange(len(paramListDefault)):
            strToEval+= 'self.cdf_inv["'+str(paramListDefault[indx])+'"](x['+str(indx)+']),'
        strToEval=strToEval[:len(strToEval)-1]  # drop last comma
        strToEval += ')'
        # multiply by ratio of p/ps
        for indx in numpy.arange(len(paramListDefault)):
            strToEval+= '*( self.prior_pdf["'+str(paramListDefault[indx])+'"](x['+str(indx)+'])/(self.pdf["'+str(paramListDefault[indx])+'"](x['+str(indx)+'])/self._pdf_norm["'  +str(paramListDefault[indx])+ '"] ))'
        print(strToEval)
        fnToUse = eval(strToEval,{'func':func, 'self':self})  # evaluate in context
#        fnToUse =vegas.batchintegrand(fnToUse)   # batch mode
#        print fnToUse
#        grid = numpy.zeros((len(paramListDefault), 2))
#        grid[:,1] = numpy.ones(len(paramListDefault))
        integ = vegas.Integrator( len(paramListDefault)*[[0,1]]) # generate the grid
        # quick and dirty training
        print('Start training')
        result = integ(fnToUse,nitn=10, neval=1000)
        print(result.summary())
        # result -- problem of very highly peaked function, not clear if vanilla vegas is smart enough.
        # Loop over blocks of 1000 evaluations, and check that final chi2/dof  is within 0.05 of 1
        bDone = False
        alphaRunning = numpy.min([0.1, 8/numpy.log(result.mean)])   # constrain dynamic range to a reaonsable range
        print('Start full  : WARNING VEGAS TENDS TO OVERADAPT given huge dynamic range')
        while (not bDone and nBlocks):  # this is basically training
            print(" Block run " , n_itr, n)
            result=integ(fnToUse,nitn=n_itr, neval=n)
            alphaRunning = numpy.min([0.1, 8/numpy.log(result.mean)])   # constrain dynamic range to a reaonsable range
            print(nBlocks, numpy.sqrt(2*numpy.log(result.mean)), result.sdev/result.mean, result.chi2/result.dof, alphaRunning)
            print(result.summary())
            nBlocks+= -1
            if numpy.abs(result.chi2/result.dof - 1) < 0.05:
                bDone =True
        print(result.summary())
        return result

    #
    # FIXME: The priors are not strictly part of the MC integral, and so any
    # internal reference to them needs to be moved to a subclass which handles
    # the incovnenient part os doing the \int p/p_s L d\theta integral.
    #
    def integrate(self, func, *args, **kwargs):
        """
        Integrate func, by using n sample points. Right now, all params defined must be passed to args must be provided, but this will change soon.

        Limitations:
            func's signature must contain all parameters currently defined by the sampler, and with the same names. This is required so that the sample values can be passed consistently.

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

        tripwire_fraction - fraction of nmax of iterations after which n_eff needs to be greater than 1+epsilon for epsilon a small number
        tripwire_epsilon - small number used in tripwire test

        Pinning a value: By specifying a kwarg with the same of an existing parameter, it is possible to "pin" it. The sample draws will always be that value, and the sampling prior will use a delta function at that value.
        """

        #
        # Pin values
        #
        n_horrible = 0
        n_horrible_max = 10

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
        #args = tuple(list(args) + filter(lambda p: p in self.params, kwargs.keys()))
        # This is a semi-hack to ensure that the integrand is called with
        # the arguments in the right order
        # FIXME: How dangerous is this?
        no_protect_names = False
        if 'no_protect_names' in kwargs:
           no_protect_names = kwargs['no_protect_names']
        if hasattr(func,'__code__') and not(no_protect_names):
            args = func.__code__.co_varnames[:func.__code__.co_argcount]
        else:
            args=list(args[:len(args)])
            no_protect_params=True   # not code dictionry, so just use arguments in order

        #if set(args) & set(params) != set(args):
        # DISABLE THIS CHECK
#        if not MCSampler.match_params_from_args(args, self.params):
#            raise ValueError("All integrand variables must be represented by integral parameters.")
        
        #
        # Determine stopping conditions
        #
        nmax = int(kwargs["nmax"]) if "nmax" in kwargs else float("inf")
        neff = kwargs["neff"] if "neff" in kwargs else numpy.float128("inf")
        n = int(kwargs["n"]) if "n" in kwargs else min(1000, nmax)
        convergence_tests = kwargs["convergence_tests"] if "convergence_tests" in kwargs else None


        #
        # Adaptive sampling parameters
        #
        n_history = int(kwargs["history_mult"]*n) if "history_mult" in kwargs else n
        tempering_exp = kwargs["tempering_exp"] if "tempering_exp" in kwargs else 0.0
        n_adapt = int(kwargs["n_adapt"]*n) if "n_adapt" in kwargs else 0
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
        force_no_adapt = kwargs["force_no_adapt"] if "force_no_adapt" in kwargs else False
        save_no_samples = kwargs["save_no_samples"] if "save_no_samples" in kwargs else False
        if save_no_samples:   # can't adapt without saved samples
            force_no_adapt = True
        # FIXME: The adaptive step relies on the _rvs cache, so this has to be
        # on in order to work
        if n_adapt > 0 and tempering_exp > 0.0:
            save_intg = True

        deltalnL = kwargs['igrand_threshold_deltalnL'] if 'igrand_threshold_deltalnL' in kwargs else float("Inf") # default is to return all
        deltaP    = kwargs["igrand_threshold_p"] if 'igrand_threshold_p' in kwargs else 0 # default is to omit 1e-7 of probability
        bFairdraw  = kwargs["igrand_fairdraw_samples"] if "igrand_fairdraw_samples" in kwargs else False
        n_extr = kwargs["igrand_fairdraw_samples_max"] if "igrand_fairdraw_samples_max" in kwargs else None

        tripwire_fraction = kwargs["tripwire_fraction"] if "tripwire_fraction" in kwargs else 2  # make it impossible to trigger
        tripwire_epsilon = kwargs["tripwire_epsilon"] if "tripwire_epsilon" in kwargs else 0.001 # if we are not reasonably far away from unity, fail!

        bUseMultiprocessing = kwargs['use_multiprocessing'] if 'use_multiprocessing' in kwargs else False
        nProcesses = kwargs['nprocesses'] if 'nprocesses' in kwargs else 2
        bShowEvaluationLog = kwargs['verbose'] if 'verbose' in kwargs else False
        bShowEveryEvaluation = kwargs['extremely_verbose'] if 'extremely_verbose' in kwargs else False

        if bShowEvaluationLog:
            print(" .... mcsampler : providing verbose output ..... ")
        if bUseMultiprocessing:
            if rosDebugMessages:
                print(" Initiating multiprocessor pool : ", nProcesses)
            p = Pool(nProcesses)

        int_val1 = numpy.float128(0)
        self.ntotal = 0
        maxval = -float("Inf")
        maxlnL = -float("Inf")
        eff_samp = 0
        mean, var = None, numpy.float128(0)    # to prevent infinite variance due to overflow

        if bShowEvaluationLog:
            print("iteration Neff  sqrt(2*lnLmax) sqrt(2*lnLmarg) ln(Z/Lmax) int_var")

        if convergence_tests:
            bConvergenceTests = False   # start out not converged, if tests are on
            last_convergence_test = defaultdict(lambda: False)   # initialize record of tests
        else:
            bConvergenceTests = False    # if tests are not available, assume not converged. The other criteria will stop it
            last_convergence_test = defaultdict(lambda: False)   # need record of tests to be returned always
        while (eff_samp < neff and self.ntotal < nmax): #  and (not bConvergenceTests):
            if (self.ntotal > nmax*tripwire_fraction) and (eff_samp < 1+tripwire_epsilon):
                print(" Tripwire: n_eff too low ")
                raise Exception("Tripwire on n_eff")

            if n_horrible >= n_horrible_max:
                raise Exception("mcsampler: Too many iterations with no "
                                "contribution to integral, hard fail")

            # Draw our sample points
            args_draw ={}
            if force_no_adapt or save_no_samples:  # don't save permanent sample history if not needed
                args_draw.update({"no_cache_samples":True})
            p_s, p_prior, rv = self.draw(n, *self.params_ordered,**args_draw)  # keep in order

#            print "Prior ",  type(p_prior[0]), p_prior[0]
#            print "rv ",  type(rv[0]), rv[0]    # rv generally has dtype = object, to enable joint sampling with multid variables
#            print "Sampling prior ", type(p_s[0]), p_s[0]  
                        
            # Calculate the overall p_s assuming each pdf is independent
            joint_p_s = numpy.prod(p_s, axis=0)
            joint_p_prior = numpy.prod(p_prior, axis=0)
            joint_p_prior = numpy.array(joint_p_prior,dtype=numpy.float128)  # Force type. Some type issues have arisen (dtype=object returns by accident)

#            print "Joint prior ",  type(joint_p_prior), joint_p_prior.dtype, joint_p_prior
#            print "Joint sampling prior ", type(joint_p_s), joint_p_s.dtype

            #
            # Prevent zeroes in the sampling prior
            #
            # FIXME: If we get too many of these, we should bail
            if any(joint_p_s <= 0):
                for p in self.params_ordered:
                    self._rvs[p] = numpy.resize(self._rvs[p], len(self._rvs[p])-n)
                print("Zero prior value detected, skipping.", file=sys.stderr)
                n_horrible += 1
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
            unpacked = unpacked0 = numpy.hstack([r.flatten() for r in rv]).reshape(len(args), -1)
            unpacked = dict(list(zip(params, unpacked)))
            if 'no_protect_names' in kwargs:
                fval = func(*unpacked0)  # do not protect order
            else:
                fval = func(**unpacked) # Chris' original plan: note this insures the function arguments are tied to the parameters, using a dictionary. 

            #
            # Check if there is any practical contribution to the integral
            #
            # FIXME: While not technically a fatal error, this will kill the
            # adaptive sampling
            if fval.sum() == 0:
                for p in self.params_ordered:
                    self._rvs[p] = numpy.resize(self._rvs[p], len(self._rvs[p])-n)
                print("No contribution to integral, skipping.", file=sys.stderr)
                n_horrible += 1
                continue

            if save_intg and not force_no_adapt:
                # FIXME: The joint_prior, if not specified is set to one and
                # will come out as a scalar here, hence the hack
                if not isinstance(joint_p_prior, numpy.ndarray):
                    joint_p_prior = numpy.ones(fval.shape)*joint_p_prior


#                print "      Prior", type(joint_p_prior), joint_p_prior.dtype

                # FIXME: See warning at beginning of function. The prior values
                # need to be moved out of this, as they are not part of MC
                # integration
                if "integrand" in self._rvs:
                    self._rvs["integrand"] = numpy.hstack( (self._rvs["integrand"], fval) )
                    self._rvs["joint_prior"] = numpy.hstack( (self._rvs["joint_prior"], joint_p_prior) )
                    self._rvs["joint_s_prior"] = numpy.hstack( (self._rvs["joint_s_prior"], joint_p_s) )
                    self._rvs["weights"] = numpy.hstack( (self._rvs["joint_s_prior"], fval*joint_p_prior/joint_p_s) )
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
                    print(self.params_ordered)
                    for indx in numpy.arange(len(args)):
                        print( self.params_ordered[indx],p_s[indx],p_prior[indx],rv[indx])

            # Calculate max L (a useful convergence feature) for debug 
            # reporting.  Not used for integration
            # Try to avoid nan's
            maxlnL = numpy.log(numpy.max([numpy.exp(maxlnL), numpy.max(fval),numpy.exp(-100)]))   # note if f<0, this will return nearly 0

            # Calculate the effective samples via max over the current 
            # evaluations
            maxval = [max(maxval, int_val[0]) if int_val[0] != 0 else maxval]
            for v in int_val[1:]:
                maxval.append( v if v > maxval[-1] and v != 0 else maxval[-1] )

            # running variance
#            var = cumvar(int_val, mean, var, int(self.ntotal))[-1]
#            var = welford(int_val, mean, var, int(self.ntotal))
            if var is None:
                var=0
            if mean is None:
                mean=0
            current_aggregate = [int(self.ntotal),mean, (self.ntotal-1)*var]
            current_aggregate = update(current_aggregate, int_val)
            outvals = finalize(current_aggregate)
#            print(var, outvals[-1])
            var = outvals[-1]
            # running integral (note also in current_aggregate)
            int_val1 += int_val.sum()
            # running number of evaluations
            self.ntotal += n
            # FIXME: Likely redundant with int_val1
            mean = int_val1/self.ntotal
            maxval = maxval[-1]

            eff_samp = int_val1/maxval

            # Throw exception if we get infinity or nan
            if math.isnan(eff_samp):
                raise NanOrInf("Effective samples = nan")
            if maxlnL is float("Inf"):
                raise NanOrInf("maxlnL = inf")

            if bShowEvaluationLog:
                print(" :",  self.ntotal, eff_samp, numpy.sqrt(2*maxlnL), numpy.sqrt(2*numpy.log(int_val1/self.ntotal)), numpy.log(int_val1/self.ntotal)-maxlnL, numpy.sqrt(var*self.ntotal)/int_val1)

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
               if bShowEvaluationLog:
                  print(" ... skipping adaptation in late iterations .. ")
               continue

            if force_no_adapt:
               continue

            # FIXME: Hardcoding
            #mixing_floor = 10**(-numpy.sqrt(ntotal))
            #mixing_floor = 10**-50

            #
            # Iterate through each of the parameters, updating the sampling
            # prior PDF according to the 1-D marginalization
            #
            for itr, p in enumerate(self.params_ordered):
                # FIXME: The second part of this condition should be made more
                # specific to pinned parameters
                if p not in self.adaptive or p in list(kwargs.keys()):
                    continue
                points = self._rvs[p][-n_history:]
#                print "      Points", p, type(points),points.dtype
                # use log weights or weights
                if not temper_log:
                    weights = (self._rvs["integrand"][-n_history:]/self._rvs["joint_s_prior"][-n_history:]*self._rvs["joint_prior"][-n_history:])**tempering_exp_running
                else:
                    weights = numpy.maximum(1e-5,numpy.log(self._rvs["integrand"][-n_history:] )) #**tempering_exp_running

                if tempering_adapt:
                    # have the adaptive exponent converge to 2/ln(w_max), ln (w)*alpha <= 2. This helps dynamic range
                    # almost always dominated by the parameters we care about
                    tempering_exp_running = 0.8 *tempering_exp_running + 0.2*(3./numpy.max([1,numpy.log(numpy.max(weights))]))
                    if rosDebugMessages:
                        print("     -  New adaptive exponent  ", tempering_exp_running, " based on max 1d weight ", numpy.max(weights), " based on parameter ", p)

#                print "      Weights",  type(weights),weights.dtype
                self._hist[p], edges = numpy.histogram( points,
                    bins = 100,
                    range = (self.llim[p], self.rlim[p]),
                    weights = weights
                )
                # FIXME: numpy.hist can't normalize worth a damn
                self._hist[p] /= self._hist[p].sum()

                # Mix with uniform distribution
                self._hist[p] = (1-floor_integrated_probability)*self._hist[p] + numpy.ones(len(self._hist[p]))*floor_integrated_probability/len(self._hist[p])
                if rosDebugMessages and bShowEvaluationLog:
                    print("         Weight entropy (after histogram) ", numpy.sum(-1*self._hist[p]*numpy.log(self._hist[p])), p)

                edges = [ (e0+e1)/2.0 for e0, e1 in zip(edges[:-1], edges[1:]) ]
                edges.append( edges[-1] + (edges[-1] - edges[-2]) )
                edges.insert( 0, edges[0] - (edges[-1] - edges[-2]) )

                # FIXME: KS test probably has a place here. Not sure yet where.
                #from scipy.stats import kstest
                #d, pval = kstest(self._rvs[p][-n_history:], self.cdf[p])
                #print p, d, pval

                self.pdf[p] = interpolate.interp1d(edges, [0] + list(self._hist[p]) + [0])
                self.cdf[p] = self.cdf_function(p)
                self.cdf_inv[p] = self.cdf_inverse(p)

        # If we were pinning any values, undo the changes we did before
        self.cdf_inv.update(tempcdfdict)
        self.pdf.update(temppdfdict)
        self._pdf_norm.update(temppdfnormdict)
        self.prior_pdf.update(temppriordict)

        # Clean out the _rvs arrays for 'irrelevant' points
        #   - find and remove samples with  lnL less than maxlnL - deltalnL (latter user-specified)
        #   - create the cumulative weights
        #   - find and remove samples which contribute too little to the cumulative weights
        if "integrand" in self._rvs:
            self._rvs["sample_n"] = numpy.arange(len(self._rvs["integrand"]))  # create 'iteration number'        
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

        # Do a fair draw of points, if option is set
        if bFairdraw and not(n_extr is None):
           n_extr = int(numpy.min([n_extr,1.5*eff_samp,1.5*neff]))
           print(" Fairdraw size : ", n_extr)
           wt = numpy.array(self._rvs["integrand"]*self._rvs["joint_prior"]/self._rvs["joint_s_prior"]/numpy.max(self._rvs["integrand"]),dtype=float)
           wt *= 1.0/numpy.sum(wt)
           if n_extr < len(self._rvs["integrand"]):
               indx_list = numpy.random.choice(numpy.arange(len(wt)), size=n_extr,replace=True,p=wt) # fair draw
               # FIXME: See previous FIXME
               for key in list(self._rvs.keys()):
                   if isinstance(key, tuple):
                       self._rvs[key] = self._rvs[key][:,indx_list]
                   else:
                       self._rvs[key] = self._rvs[key][indx_list]

        # Create extra dictionary to return things
        dict_return ={}
        if convergence_tests is not None:
            dict_return["convergence_test_results"] = last_convergence_test

        return int_val1/self.ntotal, var/self.ntotal, eff_samp, dict_return

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
uniform_samp_vector = numpy.vectorize(uniform_samp,otypes=[numpy.float64])

def ret_uniform_samp_vector_alt(a,b):
    return lambda x: numpy.where( (x>a) & (x<b), numpy.reciprocal(b-a),0.0)

# def uniform_samp_withfloor_vector(rmaxQuad,rmaxFlat,pFlat,x):
#     ret =0.
#     if x<rmaxQuad:
#         ret+= (1-pFlat)/rmaxQuad
#     if x<rmaxFlat:
#         ret +=pFlat/rmaxFlat
#     return  ret
# uniform_samp_withfloor_vector = numpy.vectorize(uniform_samp_withfloor, otypes=[numpy.float])
def uniform_samp_withfloor_vector(rmaxQuad,rmaxFlat,pFlat,x):
    if isinstance(x, float):
        ret =0.
        if x<rmaxQuad:
            ret+= (1-pFlat)/rmaxQuad
        if x<rmaxFlat:
            ret +=pFlat/rmaxFlat
        return  ret
    ret = numpy.zeros(x.shape,dtype=numpy.float64)
    ret += numpy.select([x<rmaxQuad],[(1.-pFlat)/rmaxQuad])
    ret += numpy.select([x<rmaxFlat],[pFlat/rmaxFlat])
    return ret



# syntatic sugar : predefine the most common distributions
uniform_samp_phase = lambda x,numpy=numpy: numpy.broadcast_to(0.5/numpy.pi, numpy.shape(x))
uniform_samp_psi = lambda x,numpy=numpy: numpy.broadcast_to(1.0/numpy.pi, numpy.shape(x))
uniform_samp_theta = lambda x,numpy=numpy: 0.5*numpy.sin(x.astype(float))
uniform_samp_dec = lambda x,numpy=numpy: 0.5*numpy.cos(x.astype(float))

uniform_samp_cos_theta = lambda x: numpy.broadcast_to(0.5, numpy.shape(x))
# uniform_samp_phase = numpy.vectorize(lambda x: 1/(2*numpy.pi))
# uniform_samp_psi = numpy.vectorize(lambda x: 1/(numpy.pi))
# uniform_samp_theta = numpy.vectorize(lambda x: numpy.sin(x)/(2))
# uniform_samp_dec = numpy.vectorize(lambda x: numpy.cos(x)/(2))

# uniform_samp_cos_theta = numpy.vectorize(lambda x: 1./2.)  #dumbest-possible implementation

def quadratic_samp(rmax,x):
        if x<rmax:
                return x**2/(3*rmax**3)
        else:
                return 0

quadratic_samp_vector = numpy.vectorize(quadratic_samp, otypes=[numpy.float64])

def inv_uniform_cdf(a, b, x):
    return (b-a)*x+a

def gauss_samp(mu, std, x):
    return 1.0/numpy.sqrt(2*numpy.pi*std**2)*numpy.exp(-(x-mu)**2/2/std**2)

def gauss_samp_withfloor(mu, std, myfloor, x):
    return 1.0/numpy.sqrt(2*numpy.pi*std**2)*numpy.exp(-(x-mu)**2/2/std**2) + myfloor

#gauss_samp_withfloor_vector = numpy.vectorize(gauss_samp_withfloor,excluded=['mu','std','myfloor'],otypes=[numpy.float])
gauss_samp_withfloor_vector = numpy.vectorize(gauss_samp_withfloor,otypes=[numpy.float64])


# Mass ratio. PDF propto 1/(1+q)^2.  Defined so mass ratio is < 1
# expr = Integrate[1/(1 + q)^2, q]
# scale = (expr /. q -> qmax )  - (expr /. q -> qmin)
# (expr - (expr /. q -> qmin))/scale == x // Simplify
# q /. Solve[%, q][[1]] // Simplify
# % // CForm
def q_samp_vector(qmin,qmax,x):
    scale = 1./(1+qmin) - 1./(1+qmax)
    return 1/numpy.power((1+x),2)/scale
def q_cdf_inv_vector(qmin,qmax,x):
    return numpy.array((qmin + qmax*qmin + qmax*x - qmin*x)/(1 + qmax - qmax*x + qmin*x),dtype=np.float128)

# total mass. Assumed used with q.  2M/Mmax^2-Mmin^2
def M_samp_vector(Mmin,Mmax,x):
    scale = 2./(Mmax**2 - Mmin**2)
    return x*scale


def cos_samp(x):
        return numpy.sin(x)/2   # x from 0, pi

def dec_samp(x):
        return numpy.sin(x+numpy.pi/2)/2   # x from 0, pi


cos_samp_vector = lambda x: cos_samp(numpy.array(x,dtype=numpy.float64))
dec_samp_vector = lambda x: dec_samp(numpy.array(x,dtype=numpy.float64))

#cos_samp_vector = numpy.vectorize(cos_samp,otypes=[numpy.float])
#dec_samp_vector = numpy.vectorize(dec_samp,otypes=[numpy.float])
def cos_samp_cdf_inv_vector(p):
    return numpy.arccos( 2*p-1)   # returns from 0 to pi
def dec_samp_cdf_inv_vector(p):
    return numpy.arccos(2*p-1) - numpy.pi/2  # target from -pi/2 to pi/2


def pseudo_dist_samp(r0,r):
        return r*r*numpy.exp( - (r0/r)*(r0/r)/2. + r0/r)+0.01  # put a floor on probability, so we converge. Note this floor only cuts out NEARBY distances

#pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,excluded=['r0'],otypes=[numpy.float])
pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,otypes=[numpy.float64])

def delta_func_pdf(x_0, x):
    return 1.0 if x == x_0 else 0.0

delta_func_pdf_vector = numpy.vectorize(delta_func_pdf, otypes=[numpy.float64])

def delta_func_samp(x_0, x):
    return x_0

delta_func_samp_vector = numpy.vectorize(delta_func_samp, otypes=[numpy.float64])


def linear_down_samp(x,xmin=0,xmax=1):
    """
    distribution p(x) \propto (xmax-x) 
    """
    return 2./(xmax-xmin)**2 * numpy.power(xmax-x,1)

def linear_down_samp_cdf(x,xmin=0,xmax=1):
    r"""
    CDF of distribution p(x) \propto (xmax-x) 
    """
    return 1.-1./(xmax-xmin)**2 * numpy.power(xmax-x,2)


def power_down_samp(x,xmin=0,xmax=1,alpha=3):
    r"""
    distribution p(x) \propto (xmax-x) 
    """
    return alpha/(xmax-xmin)**alpha * numpy.power(xmax-x,alpha-1)

def power_down_samp_cdf(x,xmin=0,xmax=1,alpha=3):
    r"""
    CDF of distribution p(x) \propto (xmax-x) 
    """
    return 1.-1./(xmax-xmin)**alpha * numpy.power(xmax-x,alpha)


class HealPixSampler(object):
    """
    Class to sample the sky using a FITS healpix map. Equivalent to a joint 2-D pdf in RA and dec.
    """

    @staticmethod
    def thph2decra(th, ph):
        """
        theta/phi to RA/dec
        theta (north to south) (0, pi)
        phi (east to west) (0, 2*pi)
        declination: north pole = pi/2, south pole = -pi/2
        right ascension: (0, 2*pi)
        
        dec = pi/2 - theta
        ra = phi
        """
        return numpy.pi/2-th, ph

    @staticmethod
    def decra2thph(dec, ra):
        """
        theta/phi to RA/dec
        theta (north to south) (0, pi)
        phi (east to west) (0, 2*pi)
        declination: north pole = pi/2, south pole = -pi/2
        right ascension: (0, 2*pi)
        
        theta = pi/2 - dec
        ra = phi
        """
        return numpy.pi/2-dec, ra

    def __init__(self, skymap, massp=1.0):
        self.skymap = skymap
        self._massp = massp
        self.renormalize()

    @property
    def massp(self):
        return self._massp

    @massp.setter
    def massp(self, value):
        assert 0 <= value <= 1
        self._massp = value
        norm = self.renormalize()

    def renormalize(self):
        """
        Identify the points contributing to the overall cumulative probability distribution, and set the proper normalization.
        """
        res = healpy.npix2nside(len(self.skymap))
        self.pdf_sorted = sorted([(p, i) for i, p in enumerate(self.skymap)], reverse=True)
        self.valid_points_decra = []
        cdf, np = 0, 0
        for p, i in self.pdf_sorted:
            if p == 0:
                continue # Can't have a zero prior
            self.valid_points_decra.append(HealPixSampler.thph2decra(*healpy.pix2ang(res, i)))
            cdf += p
            if cdf > self._massp:
                break
        self._renorm = cdf
        # reset to indicate we'd need to recalculate this
        self.valid_points_hist = None
        return self._renorm

    def __expand_valid(self, min_p=1e-7):
        #
        # Determine what the 'quanta' of probabilty is
        #
        if self._massp == 1.0:
            # This is to ensure we don't blow away everything because the map
            # is very spread out
            min_p = min(min_p, max(self.skymap))
        else:
            # NOTE: Only valid if CDF descending order is kept
            min_p = self.pseudo_pdf(*self.valid_points_decra[-1])

        self.valid_points_hist = []
        ns = healpy.npix2nside(len(self.skymap))

        # Renormalize first so that the vector histogram is properly normalized
        self._renorm = 0
        # Account for probability lost due to cut off
        for i, v in enumerate(self.skymap >= min_p):
            self._renorm += self.skymap[i] if v else 0

        for pt in self.valid_points_decra:
            th, ph = HealPixSampler.decra2thph(pt[0], pt[1])
            pix = healpy.ang2pix(ns, th, ph)
            if self.skymap[pix] < min_p:
                continue
            self.valid_points_hist.extend([pt]*int(round(self.pseudo_pdf(*pt)/min_p)))
        self.valid_points_hist = numpy.array(self.valid_points_hist).T

    def pseudo_pdf(self, dec_in, ra_in):
        """
        Return pixel probability for a given dec_in and ra_in. Note, uses healpy functions to identify correct pixel.
        """
        th, ph = HealPixSampler.decra2thph(dec_in, ra_in)
        res = healpy.npix2nside(len(self.skymap))
        return self.skymap[healpy.ang2pix(res, th, ph)]/self._renorm

    def pseudo_cdf_inverse(self, dec_in=None, ra_in=None, ndraws=1, stype='vecthist'):
        """
        Select points from the skymap with a distribution following its corresponding pixel probability. If dec_in, ra_in are suupplied, they are ignored except that their shape is reproduced. If ndraws is supplied, that will set the shape. Will return a 2xN numpy array of the (dec, ra) values.
        stype controls the type of sampling done to retrieve points. Valid choices are
        'rejsamp': Rejection sampling: accurate but slow
        'vecthist': Expands a set of points into a larger vector with the multiplicity of the points in the vector corresponding roughly to the probability of drawing that point. Because this is not an exact representation of the proability, some points may not be represented at all (less than quantum of minimum probability) or inaccurately (a significant fraction of the fundamental quantum).
        """

        if ra_in is not None:
            ndraws = len(ra_in)
        if ra_in is None:
            ra_in, dec_in = numpy.zeros((2, ndraws))

        if stype == 'rejsamp':
            # FIXME: This is only valid under descending ordered CDF summation
            ceiling = max(self.skymap)
            i, np = 0, len(self.valid_points_decra)
            while i < len(ra_in):
                rnd_n = numpy.random.randint(0, np)
                trial = numpy.random.uniform(0, ceiling)
                if trial <= self.pseudo_pdf(*self.valid_points_decra[rnd_n]):
                    dec_in[i], ra_in[i] = self.valid_points_decra[rnd_n]
                    i += 1
            return numpy.array([dec_in, ra_in])
        elif stype == 'vecthist':
            if self.valid_points_hist is None:
                self.__expand_valid()
            np = self.valid_points_hist.shape[1]
            rnd_n = numpy.random.randint(0, np, len(ra_in))
            dec_in, ra_in = self.valid_points_hist[:,rnd_n]
            return numpy.array([dec_in, ra_in])
        else:
            raise ValueError("%s is not a recgonized sampling type" % stype)

#pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,excluded=['r0'],otypes=[numpy.float])
pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,otypes=[numpy.float64])


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
    
