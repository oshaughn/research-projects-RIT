# mcsamplerAdaptiveVolume
#
# Algorithm: based on Tiwari VARAHA https://arxiv.org/pdf/2303.01463.pdf
# Based strongly on 'varaha_example.ipynb' email from 2023/03/07


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
          print(" mcsamplerAV: no cupyx.scipy.special.logsumexp, fallback mode ...")
          xpy_special_default= special
  except:
    print(" mcsamplerAV: no cupyx.scipy.special, fallback mode ...")
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
  print(' no cupy (mcsamplerAV)')
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

from RIFT.integrators.statutils import  update,finalize, init_log,update_log,finalize_log

#from multiprocessing import Pool

from RIFT.likelihood import vectorized_general_tools

__author__ = "R. O'Shaughnessy, V. Tiwari"

rosDebugMessages = True

class NanOrInf(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

### V. Tiwari routines

def get_likelihood_threshold(lkl, lkl_thr, nsel, discard_prob,xpy_here=xpy_default):
    """
    Find the likelihood threshold that encolses a probability
    lkl  : array of likelihoods (on bins)
    lkl_thr: scalar cutoff
    nsel : integer, has to do with size of array of likelihoods used to evaluate for next array.
    discard_prob: threshold on CDF to throw away an entire bin.  Should be very small
    """
    
    w = xpy_here.exp(lkl - np.max(lkl))
    npoints = len(w)
    sumw = xpy_here.sum(w)
    prob = w/sumw
    idx = xpy_here.argsort(prob)
    ecdf = xpy_here.cumsum(prob[idx])
    F = xpy_here.linspace(np.min(ecdf), 1., npoints)
    prob_stop_thr = lkl[idx][ecdf >= discard_prob][0]
    
    lkl_stop_thr = xpy_here.flip(np.sort(lkl))
    if len(lkl_stop_thr)>nsel:
        lkl_stop_thr = lkl_stop_thr[nsel]
    else:
        lkl_stop_thr = lkl_stop_thr[-1]
    lkl_thr = min(lkl_stop_thr, prob_stop_thr)

    truncp = xpy_here.sum(w[lkl < lkl_thr]) / sumw
            
    return identity_convert(lkl_thr), identity_convert(truncp)  # send both to CPU as needed

def sample_from_bins(xrange, dx, bu, ninbin, reject_out_of_range=False):
    
        ndim = xrange.shape[0]
        xlo, xhi = xrange.T[0] + dx * bu, xrange.T[0] + dx * (bu+1)
        x = np.vstack([np.random.uniform(xlo[kk], xhi[kk], size = (npb, ndim)) for kk, npb in enumerate(ninbin)])
        # remove points that are out of range.  Due to rounding issues etc, the sampler above can generate points out of range!
        # Note this rejection will bias the integral, because volumes are calculated assuming a regular grid. We *should* fix the grid sizes to integers
        if reject_out_of_range:
          for indx in np.arange(len(xrange)):
            indx_ok = np.where(np.logical_and(x[:,indx] >= xrange[indx,0], x[:,indx] <= xrange[indx,1], ))
            x = x[indx_ok]
        return x


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


    def __init__(self,n_chunk=400000,**kwargs):
        # Total number of samples drawn
        self.ntotal = 0
        # Parameter names
        self.params = set()
        self.params_ordered = []  # keep them in order. Important to break likelihood function need for names
        self.params_pinned_vals = {}
        # If the pdfs aren't normalized, this will hold the normalization 
        # Cache for the sampling points
        self._rvs = {}
        # parameter -> cdf^{-1} function object
        # params for left and right limits
        self.llim, self.rlim = {}, {}


        self.n_chunk = n_chunk
        self.nbins = None
        self.ninbin = None
        self.adaptive =[]

        self.pdf = {} # not used

        # MEASURES (=priors): ROS needs these at the sampler level, to clearly separate their effects
        # ASSUMES the user insures they are normalized
        self.prior_pdf = {}

        # histogram setup
        self.xpy = numpy
        self.identity_convert = lambda x: x  # if needed, convert to numpy format  (e.g, cupy.asnumpy)

        # sampling tool
        self.V=None  # fractional volume
        self.delta_V=None  # fractional volume
        

    def setup(self, **kwargs):
        ndim = len(self.params)
        self.nbins = np.ones(ndim)
        self.d_adaptive = len(self.adaptive)
        self.indx_adaptive = [self.params_ordered.index(name) for name in self.adaptive]
        self.indx_not_adaptive = list(set(list( range(ndim))) -set( self.indx_adaptive))
        self.binunique = np.array([ndim* [0]])
        self.ninbin   = [self.n_chunk]
        self.my_ranges =  np.array([[self.llim[x],self.rlim[x]] for x in self.params_ordered])
        self.dx = np.diff(self.my_ranges, axis = 1).flatten()  # weird way to code this
        self.dx0  = np.array(self.dx)   # Save initial prior widths (used for initial prior ragne at end/volume)
        self.cycle = 1

        self.V=1
        self.V_s = np.prod([ self.rlim[x] - self.llim[x] for x in self.llim])  # global sampling volume
        self.lnL_thresh = -np.inf
        self.enc_prob = 0.999

        self.is_varaha=True

    def clear(self):
        """
        Clear out the parameters and their settings, as well as clear the sample cache.
        """
        self.params = set()
        self.params_ordered = []
        self.pdf = {}
        self._pdf_norm = defaultdict(lambda: 1.0)
        self._rvs = {}
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
        self.prior_pdf[params] = prior_pdf

        if adaptive_sampling:
            print("   Adapting ", params)
            self.adaptive.append(params)

    def prior_prod(self, x):
        """
        Evaluates prior_pdf(x), multiplying together all factors
        """
        p_out = np.ones(len(x))
        indx = 0
        for param in self.params_ordered:
            p_out *= self.prior_pdf[param](x[:,indx])
            indx +=1
        return p_out


    def draw_simplified(self,n_to_get, *args, **kwargs):
        rv, log_p = self.draw_simple()
        p = np.exp(log_p)[:n_to_get]
        ps = self.xpy.ones(len(p))*self.V_s/self.V   # sampling prior, full hypercube normalized to 1
        ps = ps[:n_to_get]
        rv = rv[:n_to_get].T
        return ps, p, rv

    def draw_simple(self):
        # Draws
        x =  sample_from_bins(self.my_ranges, self.dx, self.binunique, self.ninbin)
        # if pinning, assign hard values. Note this means prior probabilities are still propagated as arbitrary scales
        if self.params_pinned_vals:
            for p in self.params_pinned_vals:
               indx_p = self.params_ordered.index(p)
               x[:,indx_p] = self.params_pinned_vals[p]
          
        # probabilities at these points.  
        log_p = np.log(self.prior_prod(x))
        # Not including any sampling prior factors, since it is de facto uniform right now (just discarding 'irrelevant' regions)
        return x, log_p

    def update_sampling_prior_selfish(self, lnF, *args, xpy=xpy_default,no_protect_names=True,**kwargs):
        """
      update_sampling_prior

      Update VARAHA sampling hypercubes/

      Note that external samples are NOT uniform.
      VARAHA should only be trained on its own samples, not others!

      We therefore do a single pure step of VARAHA, including *independent* draws.  We will keep state about 'V' etc from previous iterations.
        We therefore also have to know about the function we are integrating. However, we do not keep track of the integral result here -- the top -level routine does this.
       """
        xpy_here = self.xpy
        enforce_bounds=True

        # VT specific items
        loglkl_thr = -1e15
        enc_prob = 0.999 #The approximate upper limit on the final probability enclosed by histograms.
        V = self.V  # nominal scale factor for hypercube volume
        ndim = len(self.params_ordered)
        allx, allloglkl = np.transpose([[]] * ndim), []
        allp = []
        trunc_p = 1e-10 #How much probability analysis removes with evolution
        nsel = 1000# number of largest log-likelihood samples selected to estimate lkl_thr for the next cycle.
        if cupy_ok:
          alldx = identity_convert_togpu(allx)
          allloglkl = identity_convert_togpu(allloglkl)

        ntotal_true = 0
        if True: # while (eff_samp < neff and ntotal_true < nmax ): #  and (not bConvergenceTests):
            # Draw samples. Note state variables binunique, ninbin -- so we can re-use the sampler later outside the loop
            rv, log_joint_p_prior = self.draw_simple()  # Beware reversed order of rv
            ntotal_true += len(rv)
            if cupy_ok:
              rv = identity_convert_togpu(rv) # send random numbers to GPU : ugh
              log_joint_p_prior = identity_convert_togpu(log_joint_p_prior)    # send to GPU if required. Don't waste memory reassignment otherwise

            # Evaluate function, protecting argument order
            if True: #'no_protect_names' in kwargs:
                unpacked0 = rv.T
                lnL = lnF(*unpacked0)  # do not protect order
            # else:
            #     unpacked = dict(list(zip(self.params_ordered,rv.T)))
            #     lnL= lnF(**unpacked)  # protect order using dictionary
            # take log if we are NOT using lnL
            if cupy_ok:
              if not(isinstance(lnL,cupy.ndarray)):
                lnL = identity_convert_togpu(lnL)  # send to GPU, if not already there


            # For now: no prior, just duplicate VT algorithm
            log_integrand =lnL  + log_joint_p_prior
            
            loglkl = log_integrand # note we are putting the prior in here

            idxsel = xpy_here.where(loglkl > loglkl_thr)
            #only admit samples that lie inside the live volume, i.e. one that cross likelihood threshold
            allx = xpy_here.append(allx, rv[idxsel], axis = 0)
            allloglkl = xpy_here.append(allloglkl, loglkl[idxsel])
            allp = xpy_here.append(allp, log_joint_p_prior[idxsel])
            ninj = len(allloglkl)


            #just some test to verify if we dont discard more than 1 - Pthr probability
            at_final_threshold = np.round(enc_prob/trunc_p) - np.round(enc_prob/(1 - enc_prob)) == 0
            #Estimate likelihood threshold
            if not(at_final_threshold):
                loglkl_thr, truncp = get_likelihood_threshold(allloglkl, loglkl_thr, nsel, 1 - enc_prob - trunc_p,xpy_here=xpy_here)
                trunc_p += truncp
    
            # Select with threshold
            idxsel = xpy_here.where(allloglkl > loglkl_thr)
            allloglkl = allloglkl[idxsel]
            allp = allp[idxsel]
            allx = allx[idxsel]
            nrec = len(allloglkl)   # recovered size of active volume at present, after selection

            # Weights
            lw = allloglkl - xpy_here.max(allloglkl)
            w = xpy_here.exp(lw)
            neff_varaha = identity_convert(xpy_here.sum(w) ** 2 / xpy_here.sum(w ** 2))
            eff_samp = identity_convert(xpy_here.sum(w)/xpy_here.max(w))  # to CPU as needed
 
            #New live volume based on new likelihood threshold
            V *= (nrec / ninj)
            delta_V = V / np.sqrt(nrec) 
 
            # Redefine bin sizes, reassign points to redefined hypercube set. [Asymptotically this becomes stationary]
            # Note hypercube calculation is on CPU at present, always
            if self.d_adaptive > 0:
              self.nbins = np.ones(ndim)*(1/delta_V) ** (1/self.d_adaptive)  # uniform split in each dimension is normal, but we have array - can be irregular
              self.nbins[self.indx_not_adaptive] = 1  # reset to 1 bin for non-adaptive dimensions
            else:
              self.nbins = np.ones(ndim) # why are we even doing this!

            # bin sizes integers?  May slow us down
            if enforce_bounds:
              self.nbins = np.floor(self.nbins)

            self.dx = np.diff(self.my_ranges, axis = 1).flatten() / self.nbins   # update bin widths
            binidx = ( (( identity_convert(allx) - self.my_ranges.T[0]) / self.dx.T).astype(int)  ) #bin indexs of the samples ... sent back to CPU as needed

            self.binunique = np.unique(binidx, axis = 0)
            self.ninbin = ((self.n_chunk // self.binunique.shape[0] + 1) * np.ones(self.binunique.shape[0])).astype(int)

            self.cycle += 1

        self.V = V
        self.delta_V  = delta_V
        

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


        xpy_here = self.xpy

        #
        # Pin values
        #
        for p, val in list(kwargs.items()):
            reset_indexes = False
            if p in self.params_ordered:
              reset_indexes = True
              # add to list of pinned values
              self.params_pinned_vals[p] = val
              # disable adaptivity in this parameter, if present
              if p in self.adaptive:
                self.adaptive.remove(p)
            if reset_indexes:
              ndim = len(self.params)
              self.indx_adaptive = [self.params_ordered.index(name) for name in self.adaptive]
              self.indx_not_adaptive = list(set(list( range(ndim))) -set( self.indx_adaptive))

        
        #
        # Determine stopping conditions
        #
        nmax = kwargs["nmax"] if "nmax" in kwargs else float("inf")
        neff = kwargs["neff"] if "neff" in kwargs else numpy.float128("inf")
        n = int(kwargs["n"] if "n" in kwargs else min(100000, nmax))
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


        verbose = kwargs["verbose"] if "verbose" in kwargs else False  # default
        super_verbose = kwargs["super_verbose"] if "super_verbose" in kwargs else False  # default
        dict_return_q = kwargs["dict_return"] if "dict_return" in kwargs else False  # default.  Method for passing back rich data structures for debugging

        # use integer numbers of bins always
        enforce_bounds = kwargs["enforce_bounds"] if "enforce_bounds" in kwargs else False

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

        self.n_chunk = n
        self.setup()  # sets up self.my_ranges, self.dx initially

        cycle =1

        # VT specific items
        loglkl_thr = -1e15
        enc_prob = 0.999 #The approximate upper limit on the final probability enclosed by histograms.
        V = 1  # nominal scale factor for hypercube volume
        ndim = len(self.params_ordered)
        allx, allloglkl = np.transpose([[]] * ndim), []
        allp = []
        trunc_p = 1e-10 #How much probability analysis removes with evolution
        nsel = 1000# number of largest log-likelihood samples selected to estimate lkl_thr for the next cycle.
        nsel = np.min([nsel, int(0.1*self.n_chunk)]) #  if chunk size is small, don't pick too many points
        if cupy_ok:
          alldx = identity_convert_togpu(allx)
          allloglkl = identity_convert_togpu(allloglkl)

        ntotal_true = 0
        while (eff_samp < neff and ntotal_true < nmax ): #  and (not bConvergenceTests):
            # Draw samples. Note state variables binunique, ninbin -- so we can re-use the sampler later outside the loop
            rv, log_joint_p_prior = self.draw_simple()  # Beware reversed order of rv
            ntotal_true += len(rv)
            if cupy_ok:
              rv = identity_convert_togpu(rv) # send random numbers to GPU : ugh
              log_joint_p_prior = identity_convert_togpu(log_joint_p_prior)    # send to GPU if required. Don't waste memory reassignment otherwise

            # Evaluate function, protecting argument order
            if 'no_protect_names' in kwargs:
                unpacked0 = rv.T
                lnL = lnF(*unpacked0)  # do not protect order
            else:
                unpacked = dict(list(zip(self.params_ordered,rv.T)))
                lnL= lnF(**unpacked)  # protect order using dictionary
            # take log if we are NOT using lnL
            if cupy_ok:
              if not(isinstance(lnL,cupy.ndarray)):
                lnL = identity_convert_togpu(lnL)  # send to GPU, if not already there


            # For now: no prior, just duplicate VT algorithm
            log_integrand =lnL  + log_joint_p_prior
#            log_weights = tempering_exp*lnL + log_joint_p_prior
            # log aggregate: NOT USED at present, remember the threshold is floating
            if current_log_aggregate is None:
              current_log_aggregate = init_log(log_integrand,xpy=xpy,special=xpy_special_default)
            else:
              current_log_aggregate = update_log(current_log_aggregate, log_integrand,xpy=xpy,special=xpy_special_default)
            
            loglkl = log_integrand # note we are putting the prior in here

            idxsel = xpy_here.where(loglkl > loglkl_thr)
            #only admit samples that lie inside the live volume, i.e. one that cross likelihood threshold
            allx = xpy_here.append(allx, rv[idxsel], axis = 0)
            allloglkl = xpy_here.append(allloglkl, loglkl[idxsel])
            allp = xpy_here.append(allp, log_joint_p_prior[idxsel])
            ninj = len(allloglkl)


            #just some test to verify if we dont discard more than 1 - Pthr probability
            at_final_threshold = np.round(enc_prob/trunc_p) - np.round(enc_prob/(1 - enc_prob)) == 0
            #Estimate likelihood threshold
            if not(at_final_threshold):
                loglkl_thr, truncp = get_likelihood_threshold(allloglkl, loglkl_thr, nsel, 1 - enc_prob - trunc_p,xpy_here=xpy_here)
                trunc_p += truncp
    
            # Select with threshold
            idxsel = xpy_here.where(allloglkl > loglkl_thr)
            allloglkl = allloglkl[idxsel]
            allp = allp[idxsel]
            allx = allx[idxsel]
            nrec = len(allloglkl)   # recovered size of active volume at present, after selection

            # Weights
            lw = allloglkl - xpy_here.max(allloglkl)
            w = xpy_here.exp(lw)
            neff_varaha = identity_convert(xpy_here.sum(w) ** 2 / xpy_here.sum(w ** 2))
            eff_samp = identity_convert(xpy_here.sum(w)/xpy_here.max(w))  # to CPU as needed
 
            #New live volume based on new likelihood threshold
            V *= (nrec / ninj)
            delta_V = V / np.sqrt(nrec) 
 
            # Redefine bin sizes, reassign points to redefined hypercube set. [Asymptotically this becomes stationary]
            # Note hypercube calculation is on CPU at present, always
            if self.d_adaptive > 0:
              self.nbins = np.ones(ndim)*(1/delta_V) ** (1/self.d_adaptive)  # uniform split in each dimension is normal, but we have array - can be irregular
              self.nbins[self.indx_not_adaptive] = 1  # reset to 1 bin for non-adaptive dimensions
            else:
              self.nbins = np.ones(ndim) # why are we even doing this!

            # bin sizes integers?  May slow us down
            if enforce_bounds:
              self.nbins = np.floor(self.nbins)

            self.dx = np.diff(self.my_ranges, axis = 1).flatten() / self.nbins   # update bin widths
            binidx = ( (( identity_convert(allx) - self.my_ranges.T[0]) / self.dx.T).astype(int)  ) #bin indexs of the samples ... sent back to CPU as needed

            self.binunique = np.unique(binidx, axis = 0)
            self.ninbin = ((self.n_chunk // self.binunique.shape[0] + 1) * np.ones(self.binunique.shape[0])).astype(int)
            self.ntotal = current_log_aggregate[0]

            if super_verbose:
              print(ntotal_true,eff_samp, np.round(neff_varaha), np.round(np.max(allloglkl), 1), len(allloglkl), np.mean(self.nbins), V,  len(self.binunique),  np.round(loglkl_thr, 1), trunc_p)
            else:
              print(ntotal_true,eff_samp, np.sqrt(2*xpy_here.max(allloglkl - allp)), '-', np.log(V), np.sqrt(xpy_here.var(w/xpy_here.mean(w))/len(w) ))

            cycle += 1
            if cycle > 1000:
                break

        # VT approach was to accumulate samples, but then prune them.  So we have all the lnL and x draws

        # write in variables requested in the standard format
        for indx in np.arange(len(self.params_ordered)):
            self._rvs[self.params_ordered[indx]] = allx[:,indx]  # pull out variable
        # write out log integrand
        self._rvs['log_integrand']  = allloglkl - allp  # remember 'allloglkl' really is Lp -- despite the misleading name! --  so we are *undoing* that
        self._rvs['log_joint_prior'] = allp
        self._rvs['log_joint_s_prior'] = xpy_here.ones(len(allloglkl))*(np.log(1/V) - np.sum(np.log(self.dx0)))  # effective uniform sampling on this volume

        # Manual estimate of integrand, done transparently (no 'log aggregate' or running calculation -- so memory hog
        log_wt = self._rvs["log_integrand"] + self._rvs["log_joint_prior"] - self._rvs["log_joint_s_prior"]
        log_wt = identity_convert(log_wt)  # convert to CPU
        log_int = special.logsumexp( log_wt) - np.log(len(log_wt))  # mean value
        rel_var = np.var( np.exp(log_wt - log_int))/len(log_wt)   # error in integral, estimated: just taking int = <w> , so error is V(w_k)/N (sample mean/variance)
        eff_samp = np.sum(np.exp(log_wt - np.max(log_wt)))
        maxval = np.max(allloglkl)  # max of log

        # Integral value: NOT RELIABLE b/c not just using samples in 
#        outvals = finalize_log(current_log_aggregate,xpy=xpy)
#        log_wt_tmp = allloglkl[np.isfinite(allloglkl)]  # remove infinite entries
#        outvals = init_log(log_wt_tmp)
#        print(outvals, log_int, maxval, current_log_aggregate)
#        eff_samp = xpy.exp(  outvals[0]+np.log(len(allloglkl)) - maxval)   # integral value minus floating point, which is maximum
#        rel_var = np.exp(outvals[1]/2  - outvals[0]  - np.log(self.ntotal)/2 )

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


        # perform type conversion of all stored variables.  VERY LARGE -- should only do this if we need it!
        if cupy_ok:
          for name in self._rvs:
            if isinstance(self._rvs[name],xpy_default.ndarray):
              self._rvs[name] = identity_convert(self._rvs[name])   # this is trivial if xpy_default is numpy, and a conversion otherwise

        dict_return = {}
        return log_int, np.log(rel_var)  +2*log_int, eff_samp, dict_return

        # if outvals:
        #   out0 = outvals[0]; out1 = outvals[1]
        #   if not(isinstance(outvals[0], np.float64)):
        #     # type convert everything as needed
        #     out0 = identity_convert(out0)
        #   if not(isinstance(outvals[1], np.float64)):
        #     out1 = identity_convert(out1)
        #     eff_samp = identity_convert(eff_samp)
        #   return out0, out1 - np.log(self.ntotal), eff_samp, dict_return
        # else: # very strange case where we terminate early
        #   return None, None, None, None


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
        def ln_func(*args):
          return np.log(func(*args))
        infunc = ln_func
        use_lnL=False
        if 'use_lnL' in kwargs:   # should always be positive
          if kwargs['use_lnL']:
            infunc = func
            use_lnL=True
        log_int_val, log_var, eff_samp, dict_return =  self.integrate_log(func, **kwargs)  # pass it on, easier than mixed coding
        if use_lnL:
          self._rvs['integrand'] = self._rvs["log_integrand"]

        return log_int_val, log_var, eff_samp, dict_return
