# mcsamplerNFlows
#    Based on A.H. Fernando code
#
# Refs
#   https://github.com/bayesiains/nflows/blob/master/examples/conditional_moons.ipynb
#   https://dfdazac.github.io/02-flows.html


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


from RIFT.integrators.mcsampler_generic import MCSamplerGeneric

from typing import List, Tuple

import torch
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedPiecewiseLinearAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedPiecewiseQuadraticAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedPiecewiseCubicAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
# -------------------------------------
from nflows.transforms.base import InverseTransform
from nflows.transforms.base import MultiscaleCompositeTransform
from nflows.transforms.standard import IdentityTransform
#from nflows.transforms.standard import AffineScalarTransform
#from nflows.transforms.standard import AffineTransform
from nflows.transforms.standard import PointwiseAffineTransform
from nflows.transforms.linear import NaiveLinear

try:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except:
  True


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

__author__ = "R. O'Shaughnessy, A. H. Fernando"

rosDebugMessages = True

class NanOrInf(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def check_nonzero_deriv(my_seq,deriv_crit =0.05):
  """
  Computes  dy/dx/y \simeq (<yx> - <x><y>)/(<x><y>)
  Tries to require more than a certain percentage change per test block
  """
  mean_y = np.mean(my_seq)  # <y>
  mean_x  = len(my_seq)/2
  mean_yx = np.mean(my_seq*np.arange(len(my_seq))) # <xy>
  dydx_scale = mean_yx/(mean_y*mean_x) - 1  # relative change
  std_y_scale = np.std(my_seq)/mean_y    # relative noise
#  print(" deriv check ", dydx_scale, np.mean(my_seq), std_y_scale)  # should be PER ITERATION check
  if dydx_scale < -deriv_crit:
    return True


class NFlowsNFS_Trainer:
    def __init__(self, bounds: List[Tuple[float, float]], 
                 n_samples=100,
                 target_distribution= StandardNormal,
                 base_distribution=StandardNormal,
                 transform=CompositeTransform):
        self.bounds              = bounds
        self.n_samples           = n_samples
        self.target_distribution = target_distribution
        self.base_distribution   = base_distribution
        self.transform           = transform
        self.flow                = None
        self.plotting            = False
        self.loss_history = []
    
    def set_plotting(self, plotting: bool = False) -> None:
        self.plotting            = plotting
    
    def train_flow(self, samples_in: List[List[float]],
                   out_n_samples: int, max_epochs: int, bound_offset: float,n_print=50):
        if self.flow is None:
          self.flow                = Flow(self.transform, self.base_distribution(shape=[len(self.bounds)]))

        optimizer                = optim.Adam(self.flow.parameters())
        prev_efficiency          = float('inf')
        losses                   = self.loss_history
        # Fixed training input
        samples              = torch.tensor(samples_in, dtype=torch.float32)
        for epoch in range(1, int(max_epochs)):       
            # Set gradients equals to zero
            optimizer.zero_grad()
            
            # Compute the loss
            loss                 = -self.flow.log_prob(samples).mean()
            
            # Backpropergate the loss
            loss.backward()
            
            # Store loss value for monitoring
            losses.append(loss.item()) 
            if (epoch%n_print)==0:
              print(epoch, loss.item())
              n_hist = np.min([len(losses), 100])
#              if check_nonzero_deriv(losses[-n_hist:]):
#                print("    -- stationary ---")
            
            optimizer.step()
            
            # Display Results Every 500 Iterations 
            if (epoch + 1) % 500 == 0:
                # with torch.no_grad():
                #     min_bounds   = [bound[0] - bound_offset for bound in self.bounds]
                #     max_bounds   = [bound[1] + bound_offset for bound in self.bounds]
                #     xgrid        = [torch.linspace(min_bound, 
                #                                    max_bound, 
                #                                    out_n_samples) for min_bound, 
                #                                                       max_bound in zip(min_bounds, 
                #                                                                        max_bounds)]
                #     meshgrids    = torch.meshgrid(*xgrid)
                #     xyinput      = torch.cat([meshgrid.reshape(-1, 1) for meshgrid in meshgrids], dim=1)

                #     zgrid        = self.flow.log_prob(xyinput).exp().reshape([out_n_samples] * len(self.bounds))
                    
                # if len(self.bounds) == 2 and self.plotting == True:
                #     self.plot_if_2d(epoch, meshgrids, zgrid, loss)
              #else:
              print(f"Iteration: {epoch + 1}   Training Loss: {loss}")

        return losses
    
    
    def plot_if_2d(self, epoch, meshgrids, zgrid, loss) -> None:
        plt.figure()
        dummy_variable = plt.contourf(*meshgrids, zgrid.numpy())
        plt.title(f"Iteration: {epoch + 1}   Training Loss: {loss}")
        plt.savefig("fig_{}.png".format(epoch))




class MCSampler(MCSamplerGeneric):
    """
    Class to define a set of parameter names, limits, and probability densities.
    """


    def __init__(self,n_chunk=400000,**kwargs):
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
        self.nf_model = None
        self.nf_trainer = None
        self.nf_flow = None


    def setup(self, nf_cov=None, nf_mean=None,nf_method=None,**kwargs):

        self._rvs={}
        self.lnL_thresh = -np.inf
        self.enc_prob = 0.999
        d_nf = len(self.params_ordered)

        bounds = np.array([ [self.llim[p],self.rlim[p]] for p in self.params_ordered])

        # https://github.com/bayesiains/nflows/blob/master/examples/moons.ipynb
        self.num_layers = int(len(bounds)/2)  # autoregressive
        transforms = []
        # trivial scale layer first, to get in the right place
#        transforms.append(PointwiseAffineTransform())
#        transforms.append(NaiveLinear(features=len(bounds)))
        for _ in range(self.num_layers):
          transforms.append(ReversePermutation(features = len(bounds)))
          transforms.append(MaskedAffineAutoregressiveTransform(features = len(bounds),
                                                          hidden_features = 2 * len(bounds)))    
        transform  = CompositeTransform(transforms)
        self.nf_model = transform

        trainer    = NFlowsNFS_Trainer(bounds              = bounds, 
                               transform           = transform)
        self.nf_trainer = trainer
        self.nf_flow = trainer.flow
        


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
        verbose = kwargs["verbose"] if "verbose" in kwargs else False  # default
        super_verbose = kwargs["super_verbose"] if "super_verbose" in kwargs else False  # default
        save_no_samples = kwargs.get("save_no_samples", False)
        enforce_bounds = kwargs["enforce_bounds"] if "enforce_bounds" in kwargs else False

        args = self.params_ordered # by default draw all

        if self.nf_flow is None:
          if super_verbose:
            print(" Using uniform ")
          bounds = np.array([ [self.llim[p],self.rlim[p]] for p in self.params_ordered])
          V = np.prod((bounds[:,1] - bounds[:,0]))
          # rv = np.random.uniform(low=bounds[:,0], high=bounds[:,1], size = n_to_get)
          rv = np.empty(( len(self.params_ordered),n_to_get))
          for indx, p in enumerate(self.params_ordered):
            rv[indx] = np.random.uniform(self.llim[p], self.rlim[p], size=n_to_get)
          log_ps = -np.log(V)*np.ones(n_to_get)  # de facto constaint on hypercube
          log_p = np.log(self.prior_prod(rv.T))
#          print('ps shape', log_p.shape)
        else:
          if super_verbose:
            print(" Using actual flow ")
          flow = self.nf_flow
          flow_samples = flow.sample(n_to_get)
          rv = flow_samples.detach().numpy().T
          log_ps = flow.log_prob(flow_samples).detach().numpy()
          log_p  =  np.log(self.prior_prod(rv.T))
          # remove nan values
          # enforce boundaries: don't trust flow
          if enforce_bounds:
            indx_valid = np.ones(len(log_p), dtype=bool)
            bounds = np.array([ [self.llim[p],self.rlim[p]] for p in self.params_ordered])
            for indx, p in enumerate(self.params_ordered):
              indx_valid = np.logical_and(indx_valid, rv[indx] <= self.rlim[p])
              indx_valid = np.logical_and(indx_valid, rv[indx] >= self.llim[p])
            if super_verbose:
              print(" Valid ", np.sum(indx_valid))
            rv = rv[:,indx_valid]
            log_ps = log_ps[indx_valid]
            log_p = log_p[indx_valid]

        # Cache the samples we chose
        #
        if not save_no_samples:
#            print(" ===== RECORDING SAMPLES ====")
            if len(self._rvs) == 0:
               self._rvs = dict(list(zip(args, rv)))
#               print(self._rvs)
            else:
               rvs_tmp = dict(list(zip(args, rv)))
#               print(rvs_tmp)
               #for p, ar in self._rvs.items():
               for p in self.params_ordered:
                   self._rvs[p] = self.xpy.hstack((self._rvs[p], rvs_tmp[p]))


        return  rv, np.exp(log_ps), np.exp(log_p)

    def update_sampling_prior(self, lnw, *args, xpy=xpy_default,no_protect_names=True,external_rvs=None,tempering_exp=1,n_history=1000,**kwargs):
      """
      update_sampling_prior

      """
      verbose = kwargs["verbose"] if "verbose" in kwargs else False  # default
      super_verbose = kwargs["super_verbose"] if "super_verbose" in kwargs else False  # default

      xpy_here = self.xpy
      enforce_bounds=True

      # Allow updates provided from outside sources,
      rvs_here = self._rvs
      if external_rvs:
        rvs_here = external_rvs

      # skip update if lnw is too flat
      if np.mean(lnw) > np.max(lnw)-1.5*len(self.params_ordered):
        if super_verbose:
          print(" Skipping update ")
        return 

      # apply tempering exponent (structurally slightly different than in low-level code - not just to likelihood)
      ln_weights  = self.xpy.array(lnw) # force copy
      ln_weights *= tempering_exp

      n_history_to_use = np.min([n_history, len(ln_weights), len(rvs_here[self.params_ordered[0]])] )
      if (n_history_to_use) < 10:
        print("  Skipping update: no history ")
        return

      # default is to use logarithmic (!) weights, relying on them being positive.
      weights_alt = ln_weights[-n_history_to_use:]  - self.xpy.max(ln_weights) + 100  
#      weights_alt = self.xpy.maximum(weights_alt, 1e-5)    # prevent negative weights, in case integrating function with lnL < 0
      # now treat as sum
      weights_alt = weights_alt/(weights_alt.sum())
      if weights_alt.dtype == numpy.float128:
        weights_alt = weights_alt.astype(numpy.float64,copy=False)


      # copy into training array, truncating size
      samples_train = np.zeros((len(self.params_ordered),n_history_to_use))
      for itr, p in enumerate(self.params_ordered):
                samples_train[itr] = rvs_here[p][-n_history_to_use:]

      # Eliminate points with low weight.  Drop bottom half, right now
      indx_ok = weights_alt >  np.mean(weights_alt) +0.5*np.std(weights_alt)     
      if np.sum(indx_ok) < 10:
        if super_verbose:
          print(" Skipping update: too few valid ") 
        return
#      print(indx_ok.shape, samples_train.shape)
      samples_train = samples_train[:,indx_ok]
      if samples_train.shape[1] < 10:
        if super_verbose:
          print(" Skipping update: too few valid ") 
        return        
      if super_verbose:
        print("    NF: Training data shape ", samples_train.shape)
        print("    NF: training data mean ", np.mean(samples_train,axis=-1))

      # Train
      trainer = self.nf_trainer
      max_epochs = 300  # be short, don't need precision/long training
      if max_epochs < int(10*self.num_layers): max_epochs = int(10*self.num_layers)

      losses     = trainer.train_flow(samples_in= samples_train.T,
                                out_n_samples = 100, 
                                max_epochs    = max_epochs, 
                                bound_offset  = 0.5)

      self.nf_flow = trainer.flow


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
        n_adapt = int(kwargs["n_adapt"]) if "n_adapt" in kwargs else 10  # default to adapt to 10 chunks, then freeze
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

        bShowEvaluationLog = kwargs['verbose'] if 'verbose' in kwargs else True
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

        ntotal_true = 0
        while (eff_samp < neff and ntotal_true < nmax ): #  and (not bConvergenceTests):
            # Draw samples. Note state variables binunique, ninbin -- so we can re-use the sampler later outside the loop
            rv, joint_p_s, joint_p_prior = self.draw_simplified(self.n_chunk, save_no_samples=False)  # Beware reversed order of rv
            if super_verbose:
              print(" Drawn ", np.mean(rv, axis=-1))
#              print(" Drawn ", np.cov(rv))
#            print(" Drawn rv ", rv)
#            print(self._rvs)
            log_joint_p_s = np.log(joint_p_s)
            log_joint_p_prior = np.log(joint_p_prior)
            ntotal_true += len(joint_p_s)
            if cupy_ok:
              rv = identity_convert_togpu(rv) # send random numbers to GPU : ugh
              log_joint_p_prior = identity_convert_togpu(log_joint_p_prior)    # send to GPU if required. Don't waste memory reassignment otherwise

            # Evaluate function, protecting argument order
            params = []
            for item in self.params_ordered:  # USE IN ORDER
                if isinstance(item, tuple):
                    params.extend(item)
                else:
                    params.append(item)
            unpacked = unpacked0 = rv #numpy.hstack([r.flatten() for r in rv]).reshape(len(args), -1)
            unpacked = dict(list(zip(params, unpacked)))
            if 'no_protect_names' in kwargs:
                lnL = lnF(*unpacked0)  # do not protect order
            else:
                unpacked = dict(list(zip(self.params_ordered,rv.T)))
                lnL= lnF(**unpacked)  # protect order using dictionary
            # take log if we are NOT using lnL
            if cupy_ok:
              if not(isinstance(lnL,cupy.ndarray)):
                lnL = identity_convert_togpu(lnL)  # send to GPU, if not already there


            # For now: no prior, just duplicate VT algorithm
#            print(rv.shape, lnL.shape, log_joint_p_prior.shape,log_joint_p_s.shape)
            log_integrand =lnL  + log_joint_p_prior - log_joint_p_s
#            log_weights = tempering_exp*lnL + log_joint_p_prior
            # log aggregate: NOT USED at present, remember the threshold is floating
            if current_log_aggregate is None:
              current_log_aggregate = init_log(log_integrand,xpy=xpy,special=xpy_special_default)
            else:
              current_log_aggregate = update_log(current_log_aggregate, log_integrand,xpy=xpy,special=xpy_special_default)

            # Adapt if needed; decrement adaptation counter
            if n_adapt > 0:
              if super_verbose:
                print("    -- n_adapt {} ".format(n_adapt))
              self.update_sampling_prior(lnL,**kwargs)
              n_adapt += -1  # decrement
            else:
              if super_verbose:
                print("  ... skipping adaptation (NF) ")

            # Monitoring for i/o
            outvals = finalize_log(current_log_aggregate,xpy=xpy)
            self.ntotal = current_log_aggregate[0]
            # effective samples
            maxval = max(maxval, identity_convert(self.xpy.max(log_integrand) ))

            # sum of weights is the integral * the number of points
            eff_samp = xpy.exp(  outvals[0]+np.log(self.ntotal) - maxval)   # integral value minus floating point, which is maximum
            if bShowEvaluationLog:
                print(" :",  self.ntotal, eff_samp, numpy.sqrt(2*maxlnL), numpy.sqrt(2*outvals[0]), outvals[0]-maxlnL, np.exp(outvals[1]/2  - outvals[0]  - np.log(self.ntotal)/2 ))

            if save_intg:
                # FIXME: See warning at beginning of function. The prior values
                # need to be moved out of this, as they are not part of MC
                # integration
                if "log_integrand" in self._rvs:
                    self._rvs["log_integrand"] = xpy_here.hstack( (self._rvs["log_integrand"], lnL) )
                    self._rvs["log_joint_prior"] = xpy_here.hstack( (self._rvs["log_joint_prior"], self.xpy.log(joint_p_prior)) )
                    self._rvs["log_joint_s_prior"] = xpy_here.hstack( (self._rvs["log_joint_s_prior"], self.xpy.log(joint_p_s)))
                else:
                    self._rvs["log_integrand"] = lnL
                    self._rvs["log_joint_prior"] = self.xpy.log(joint_p_prior)
                    self._rvs["log_joint_s_prior"] = self.xpy.log(joint_p_s)



        # Manual estimate of integrand, done transparently (no 'log aggregate' or running calculation -- so memory hog
        log_wt = self._rvs["log_integrand"] + self._rvs["log_joint_prior"] - self._rvs["log_joint_s_prior"]
        log_wt = identity_convert(log_wt)  # convert to CPU
        log_int = special.logsumexp( log_wt) - np.log(len(log_wt))  # mean value
        rel_var = np.var( np.exp(log_wt - log_int))/len(log_wt)   # error in integral, estimated: just taking int = <w> , so error is V(w_k)/N (sample mean/variance)
        eff_samp = np.sum(np.exp(log_wt - np.max(log_wt)))

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
