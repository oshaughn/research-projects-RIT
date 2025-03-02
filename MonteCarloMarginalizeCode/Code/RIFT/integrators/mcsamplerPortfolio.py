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

from copy import deepcopy


import os

try:
  import cupy
  import cupyx   # needed for logsumexp
  xpy_default=cupy
  try:
    xpy_special_default = cupyx.scipy.special
    if not(hasattr(xpy_special_default,'logsumexp')):
          print(" mcsamplerPortfolio no cupyx.scipy.special.logsumexp, fallback mode ...")
          xpy_special_default= special
  except:
    print(" mcsamplerPortfolio no cupyx.scipy.special, fallback mode ...")
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
  print(' no cupy (mcsamplerPortfolio)')
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


from RIFT.integrators.statutils import  update,finalize, init_log,update_log,finalize_log

#from multiprocessing import Pool

from RIFT.likelihood import vectorized_general_tools

# import matching integrators registered through plutings
#  https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
discovered_plugins = entry_points(group='RIFT.integrator_plugins')
known_pipelines = {}
for pipeline in discovered_plugins:
  print(" Portfolio discovery: loading ", pipeline.name)
  known_pipelines[pipeline.name] = pipeline.load()
print('RIFT portfolio plugins:', [ep.name for ep in discovered_plugins])


class NanOrInf(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


###
### scheduling functions : return probability array, given 
###

def portfolio_default_weights(n_ess_list, wt_previous, portfolio_probability_floor=0.01, history_factor=0.5, xpy=xpy_default, identity_convert=lambda x:x, **kwargs):
  assert len(n_ess_list) == len(wt_previous)

  vals = n_ess_list
  rewt = vals - 1   # will be non-negative
  # don't update if we have insane answers
  if any(np.isnan(rewt)):
    return wt_previous
  rewt = np.ones(len(rewt))*portfolio_probability_floor + (rewt/np.sum(rewt)) * (1-portfolio_probability_floor)
  net = (rewt * history_factor + wt_previous*(1-history_factor))
  return net/np.sum(net) # make SURE normalized correctly


###
### PORTFOLIO CLASS
###


class MCSampler(object):
    """
    Class to define a set of parameter names, limits, and probability densities.
    """



    def __init__(self,portfolio=None,portfolio_weights=None,oracle_realizations =None,n_chunk=400000, portfolio_freeze_wt=0.05,**kwargs):
        if portfolio is None:
            raise Exception("mcsamplerPortfolio: must provide portfolio on init")
        self.portfolio=portfolio
        self.portfolio_realizations = []
        self.oracle_realizations= oracle_realizations if oracle_realizations else []
        self.portfolio_member_varaha = {} # these members ONLY train from their own data (i.e., VARAHA), and use likelihood contours (VARAHA)
        self.portfolio_draw_iteration = 0  #  counter, used fo
        self.portfolio_breakpoints = None # breakpoints, at which we activate the other samplers for (a) drawing and (b) training
        for member in self.portfolio:
            if isinstance(member, ModuleType):
              # can pass it a top-level routine, OR
              sampler = member.MCSampler()
              self.portfolio_realizations.append(sampler)
            else:
              # can pass low-level sampler object itself
              self.portfolio_realizations.append(member)

        self.portfolio_weights =portfolio_weights   # cpu-type data structure !
        if not(self.portfolio_weights ):
            self.portfolio_weights = np.ones(len(self.portfolio))/(1.0*len(self.portfolio))

        self.portfolio_adapt = np.ones(len(self.portfolio),dtype=bool) # default : everything adapts.  
        self.portfolio_freeze_wt =portfolio_freeze_wt  # if weight is below this number, the portfolio member's distribution will NOT update. SCALAR

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


        self.n_chunk = n_chunk
        self.nbins = None
        self.adaptive =[]


        # histogram setup
        self.xpy = numpy
        self.identity_convert = lambda x: x  # if needed, convert to numpy format  (e.g, cupy.asnumpy)
        self.identity_convert_togpu = lambda x: x  # if needed, convert to numpy format  (e.g, cupy.asnumpy)

        # extra args, created during setup
        self.extra_args = {}

    def add_parameter(self, params, pdf,  **kwargs):
        """
        Add one (or more) parameters to sample dimensions. params is either a string describing the parameter, or a tuple of strings. The tuple will indicate to the sampler that these parameters must be sampled together. left_limit and right_limit are on the infinite interval by default, but can and probably should be specified. If several params are given, left_limit, and right_limit must be a set of tuples with corresponding length. Sampling PDF is required, and if not provided, the cdf inverse function will be determined numerically from the sampling PDF.
        """
        self.params.add(params) # does NOT preserve order in which parameters are provided
        self.params_ordered.append(params)
        for member in self.portfolio_realizations  + self.oracle_realizations:
            member.add_parameter(params, pdf, **kwargs)
        # update dictionary limits
        self.llim.update( member.llim)
        self.rlim.update(member.rlim)
        # set master list of adaptive parameters 
        self.adaptive = member.adaptive  # top level list of adaptive coordinates


    def setup(self,  **kwargs):
        self.extra_args =kwargs  # may need to pass/use during the 'update' step
        if 'oracle_realizations' in kwargs:
          if kwargs['oracle_realizations']: 
            self.oracle_realizations = kwargs['oracle_realizations']  # might not have been initialized earlier
        if (not('portfolio_breakpoints') in kwargs) or not(self.portfolio_breakpoints):
          self.portfolio_breakpoints = np.zeros(len(self.portfolio)) # always use all of them
        if 'portfolio_breakpoints' in kwargs:
            if kwargs['portfolio_breakpoints']:
              self.portfolio_breakpoints =np.array( kwargs['portfolio_breakpoints']     )
        assert len(self.portfolio_breakpoints) == len(self.portfolio_realizations)  # must match


        portfolio_extra_args = [{} for x in self.portfolio_realizations] # empty list
        if 'portfolio_args' in kwargs:
          if not(kwargs['portfolio_args'] is None):
            if len(kwargs['portfolio_args']) == len(self.portfolio_realizations): # Only pass args if valid
              portfolio_extra_args = kwargs['portfolio_args']
            else:
              print(" PORTFOLIO - format ERROR ", kwargs['portfolio_args'])
        for indx, member in enumerate(self.portfolio):
            if hasattr(member, 'setup'):
              print(" PORTFOLIO setup ", member, portfolio_extra_args[indx])
              args_here = {}
              args_here.update(kwargs)
              args_here.update(portfolio_extra_args[indx])
              member.setup(**args_here)
        for indx, member in enumerate(self.oracle_realizations):
            if hasattr(member, 'setup'):
              print(" PORTFOLIO ORACLE setup ", member, portfolio_extra_args[indx])
              args_here = {}
              args_here.update(kwargs)
              args_here.update(portfolio_extra_args[indx])
              member.setup(**args_here)
              member.params_ordered = list(self.params_ordered)  # enforce parameters for oracle being sane

    def draw(self,n_samples, *args, **kwargs):
        """
        draw n_samples

        Draw from portfolio.
        Uses portfolio weights to calculate desired outcomes.
        Restricts to ACTIVE members.
        """
        if len(args) == 0:
            args = self.params
        n_params = len(args)
        n_samples = int(n_samples)

        self.portfolio_draw_iteration += 1


        # Allocate memory.
        #    - initialize with zeros so we will hard fail /nan if error
        rv = self.xpy.empty((n_params, n_samples), dtype=numpy.float64)
        joint_p_s = self.xpy.zeros(n_samples, dtype=numpy.float64)
        joint_p_prior = self.xpy.zeros(n_samples, dtype=numpy.float64)

        indx_active = np.argwhere(self.portfolio_breakpoints <= self.portfolio_draw_iteration).flatten() # provide indexes
        weights_active = np.array([self.portfolio_weights[x] for x in indx_active]) # only provide desired ones
        weights_active *= 1./np.sum(weights_active)  # renormalize
        portfolio_active = [self.portfolio_realizations[x] for x in indx_active] # get the active portfolio members
#        print(" \t ",indx_active, self.portfolio_breakpoints, self.portfolio_draw_iteration)

        # if only one method is active, just call the low-level function
        if len(indx_active) == 1:
           joint_p_s, joint_p_prior, rv = self.portfolio[indx_active[0]].draw_simplified(n_samples, *self.params_ordered, **kwargs)
        else:
          # Identify number of samples per member of the portfolio. Can be zero.
          n_samples_per_member = ((np.array(weights_active))*n_samples).astype(int)

          # logic to block cases where we zero out a number of samples per member.
          # Note this motivates keeping portfolio adaptive weights frozen and not too small, to avoid accidental negative counts.
          if np.sum(n_samples_per_member[0:-1]) < n_samples:
            n_samples_per_member[-1] = n_samples - np.sum(n_samples_per_member[0:-1])
          elif np.sum(n_samples_per_member[0:-2]) < n_samples:
            n_samples_per_member[-1] = 0
            n_samples_per_member[-2] = n_samples - np.sum(n_samples_per_member[0:-2])

          n_index_start_per_member = np.zeros(len(portfolio_active),dtype=int)
          n_index_start_per_member[1:] = np.cumsum(n_samples_per_member)[:-1]

          # Draw in blocks, and copy in place
          # only draw from ACTIVE members
          for indx_member, member in enumerate(portfolio_active):
            joint_p_s_here, joint_p_prior_here, rv_here = member.draw_simplified(
                n_samples_per_member[indx_member], *self.params_ordered, **kwargs
                )
            # type convert as needed, to GPU
            if not(isinstance( type(joint_p_s_here), type(joint_p_s))):
              joint_p_s_here = self.identity_convert_togpu(joint_p_s_here)
              joint_p_prior_here = self.identity_convert_togpu(joint_p_prior_here)
              rv_here = self.identity_convert_togpu(rv_here)
            indx_start = int(n_index_start_per_member[indx_member])
            indx_end = indx_start + int(n_samples_per_member[indx_member])
            joint_p_s[indx_start:indx_end] = joint_p_s_here
            joint_p_prior[indx_start:indx_end] = joint_p_prior_here
            rv[:,indx_start:indx_end] = rv_here
            
        #
        # Cache the samples we chose.  REQUIRED
        #
        if True:
         if len(self._rvs) == 0:
            self._rvs = dict(list(zip(args, rv)))
         else:
            rvs_tmp = dict(list(zip(args, rv)))
            #for p, ar in self._rvs.items():
            for p in self.params_ordered:
                self._rvs[p] = numpy.hstack( (self._rvs[p], rvs_tmp[p]) )


        return joint_p_s, joint_p_prior, rv


    def integrate(self, lnF, *args, xpy=xpy_default,**kwargs):
        use_lnL = kwargs['use_lnL'] if 'use_lnL' in kwargs else False
        if not(use_lnL):
          raise Exception("mcsamplerPortfolio: must integrate lnL")
        return self.integrate_log(lnF, *args, xpy=xpy, **kwargs)
        

    def integrate_log(self, lnF, *args, xpy=xpy_default,**kwargs):
        xpy_here = self.xpy
        
        #
        # Determine stopping conditions
        #
        nmax = kwargs["nmax"] if "nmax" in kwargs else float("inf")
        neff = kwargs["neff"] if "neff" in kwargs else numpy.float128("inf")
        n = int(kwargs["n"] if "n" in kwargs else min(100000, nmax))
        convergence_tests = kwargs["convergence_tests"] if "convergence_tests" in kwargs else None
        save_no_samples = kwargs["save_no_samples"] if "save_no_samples" in kwargs else None
        portfolio_wt_func = kwargs['portfolio_schedule'] if 'portfolio_schedule' in kwargs else portfolio_default_weights

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
#        self.setup()  # sets up self.my_ranges, self.dx initially


        n_zero_prior =0
        it_max_oracle = 7
        it_now  =0
        if 'integrand' in self._rvs:
          # remove conflict
          del self._rvs['integrand']
        while (eff_samp < neff and self.ntotal < nmax): #  and (not bConvergenceTests):
            

            ###
            ### COMMON INTEGRATION BLOCK
            ###

            # Draw our sample points
            # non-log draw
            joint_p_s, joint_p_prior, rv = self.draw(
                n, *self.params_ordered
            )
            it_now +=1 

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
            # tempering_exp done inside the update proposal, NOT here
            log_weights = lnL + self.xpy.log(joint_p_prior) - self.xpy.log(joint_p_s)

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


            if self.ntotal > n_adapt*n:
                print(n_adapt,self.n_total)
                continue

            ###
            ### PORTFOLIO REPORT BLOCK (and reweighting of member priority)
            ###
            #  n_ess for each portfolio member
            #  computed with TRUE weights, not 
            #  Use this to reassess which portfolio members are being refined.
            n_samples = len(log_weights)
            n_samples_per_member = ((self.portfolio_weights)*len(log_weights)).astype(int)
            if np.sum(n_samples_per_member[0:-1]) < n_samples:
              n_samples_per_member[-1] = n_samples - np.sum(n_samples_per_member[0:-1])
            elif np.sum(n_samples_per_member[0:-2]) < n_samples:
              n_samples_per_member[-1] = 0
              n_samples_per_member[-2] = n_samples - np.sum(n_samples_per_member[0:-2])

            n_index_start_per_member = np.zeros(len(self.portfolio_realizations),dtype=int)
            n_index_start_per_member[1:] = np.cumsum(n_samples_per_member)[:-1]

            portfolio_report = {}
            for indx_member, member in enumerate(self.portfolio):
              indx_start = int(n_index_start_per_member[indx_member])
              indx_end = indx_start + int(n_samples_per_member[indx_member])    
              ln_wt_here =  log_weights[indx_start:indx_end]
              ln_wt_here += - np.max(ln_wt_here)
              # evaluate  n_ess, n_eff for this set of samples in batch specifically,
              portfolio_report[indx_member] = [ self.portfolio_weights[indx_member], self.identity_convert(self.xpy.sum(self.xpy.exp(ln_wt_here))**2/self.xpy.sum(self.xpy.exp(ln_wt_here*2))), identity_convert(self.xpy.sum(self.xpy.exp(ln_wt_here)))]
            print("\t",portfolio_report)
            # Weight based on n_ESS from batch.  remember these are >=1, so no negatives or 0 will happen
            dat =np.array([ portfolio_report[k][1] for k in range(len(self.portfolio))])
            self.portfolio_weights = portfolio_wt_func(dat, self.portfolio_weights, xpy=self.xpy, identity_convert=self.identity_convert) # call weighting function

              
            ###
            ### ORACLE BLOCK
            ###
            rvs_train = self._rvs
            if it_now < it_max_oracle and len(self.oracle_realizations )>0:
              rvs_train = deepcopy(self._rvs)  # duplicate deeply, since we will append to it
              n_samples_per_oracle = int(n*0.1/len(self.oracle_realizations)) # try to minimize oracle effort
              print(" ORACLE: attempting updates ")
              # update each oracle
              for member in self.oracle_realizations:
                member.update_sampling_prior(log_weights, n_history, external_rvs=rvs_train, log_scale_weights=True)
              # generate samples from oracles
              rv_oracle = self.xpy.empty((n_samples_per_oracle*len(self.oracle_realizations), len(self.params_ordered)))
              base_now = 0
              for member in self.oracle_realizations:
                _, _, rv_here = member.draw_simplified(n_samples_per_oracle)
                rv_oracle[base_now:base_now+n_samples_per_oracle] = rv_here
                base_now += n_samples_per_oracle
              # evaluate lnL for each,
              lnL_oracles = lnF(*rv_oracle.T)
              # put into weights and rvs, for use in training other samples
              self.xpy.append(log_weights, lnL_oracles)
              for indx, p in enumerate(self.params_ordered):
                self.xpy.append(rvs_train[p],  rv_oracle[:,indx])


            ###
            ### WEIGHT UPDATE BLOCK (improve by adding all portfolio options - default vanilla independent updates now)
            ###
            update_dict = {}
            update_dict.update(self.extra_args)
            update_dict['tempering_exp'] =tempering_exp
            for indx, member in enumerate(self.portfolio_realizations):
                # update sampling prior, using ALL past data
                # Don't update samples which are not being drawn
                # always update if we have an oracle  - don't freeze out out oracle, UNLESS we have explicitly frozen it with a breakpoint
                if self.portfolio_draw_iteration < self.portfolio_breakpoints[indx]:
                  print("  - before activation breakpoint for member {} ".format( indx))
                  pass
                elif (len(self.oracle_realizations) > 0 and it_now <it_max_oracle) or (   self.portfolio_weights[indx] > self.portfolio_freeze_wt):
                  if not(hasattr(member, 'is_varaha')):
                    member.update_sampling_prior(log_weights, n_history,external_rvs=rvs_train,log_scale_weights=True, **update_dict)
                  else:
                    # just do a single VARAHA step, independent of others
                    member.update_sampling_prior_selfish(lnF)
                else:
                  if self.portfolio_draw_iteration > self.portfolio_breakpoints[indx]:  
                    print("   - frozen sampling for member {} {}".format(indx, self.portfolio_weights[indx]))
                  else:
                    print("  - before activation breakpoint for member {} ".format( indx))

        # If we were pinning any values, undo the changes we did before
        # self.pdf.update(temppdfdict)
        # self._pdf_norm.update(temppdfnormdict)
        # self.prior_pdf.update(temppriordict)

        # Clean out the _rvs arrays for 'irrelevant' points
        #   - find and remove samples with  lnL less than maxlnL - deltalnL (latter user-specified)
        #   - create the cumulative weights
        #   - find and remove samples which contribute too little to the cumulative weights
        if (not save_no_samples) and ( "log_integrand" in self._rvs):
            self._rvs["sample_n"] = self.identity_convert_togpu(numpy.arange(len(self._rvs["log_integrand"])))  # create 'iteration number'        
            # Step 1: Cut out any sample with lnL belw threshold
            if deltalnL < 1e10: # not infinity, so we are truncating the sample list
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
        # if convergence_tests is not None:
        #     dict_return["convergence_test_results"] = None # last_convergence_test

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
          self._rvs['integrand'] = self._rvs['log_integrand'] # always integrating log function.  Match behavior of other routines
          return out0, out1 - np.log(self.ntotal), eff_samp, dict_return
        else: # very strange case where we terminate early
          return None, None, None, None
