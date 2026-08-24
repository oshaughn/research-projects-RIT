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
from RIFT.precision import RiftFloat  # platform-portable replacement for np.float128
from scipy import integrate, interpolate, special
import itertools
import functools

import os


from RIFT.integrators.mcsampler_generic import MCSamplerGeneric

from typing import List, Tuple

import torch
from torch import optim

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split


from nflows.flows.base import Flow
from nflows.utils import torchutils
from nflows.distributions.normal import StandardNormal
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.base import CompositeTransform, Transform
from nflows.transforms import AffineCouplingTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedPiecewiseLinearAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedPiecewiseQuadraticAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedPiecewiseCubicAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.nn.nets import ResidualNet
# -------------------------------------
from nflows.transforms.base import InverseTransform
from nflows.transforms.base import MultiscaleCompositeTransform
from nflows.transforms.standard import IdentityTransform
#from nflows.transforms.standard import AffineScalarTransform
#from nflows.transforms.standard import AffineTransform
from nflows.transforms.standard import PointwiseAffineTransform
from nflows.transforms.lu import LULinear

try:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except:
  True


try:
  import cupy
  import cupyx.scipy.special   # needed for logsumexp
  xpy_default=cupy
  try:
    xpy_special_default = cupyx.scipy.special
    if not(hasattr(xpy_special_default,'logsumexp')):
          print(" mcsamplerNF: no cupyx.scipy.special.logsumexp, fallback mode ...")
          xpy_special_default= special
  except:
    print(" mcsamplerNF: no cupyx.scipy.special, fallback mode ...")
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
  print(' no cupy (mcsamplerNF)')
#  import numpy as cupy  # will automatically replace cupy calls with numpy!
  xpy_default=numpy  # just in case, to make replacement clear and to enable override
  xpy_special_default = special
  identity_convert = lambda x: x  # trivial return itself
  identity_convert_togpu = lambda x: x
  cupy_ok = False
  cupy_pi = np.pi

from RIFT.integrators.rvs_record import RvsRecord, SamplerOutputMixin   # see DESIGN_rvs_naming.md

def set_xpy_to_numpy():
   xpy_default=numpy
   identity_convert = lambda x: x  # trivial return itself
   identity_convert_togpu = lambda x: x
   cupy_ok = False
   

if 'PROFILE' not in os.environ:
   def profile(fn):
        return fn

# if not( 'RIFT_LOWLATENCY'  in os.environ):
#     # Dont support selected external packages in low latency
#  try:
#     import healpy
#  except:
#     print(" - No healpy - ")

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


# From mbank: https://github.com/stefanoschmidt1995/mbank/blob/master/mbank/flow/flowmodel.py
#   Should modify to be untrainable, since we want to enforce boundaries.
class TanhTransform(Transform):
	"""
	Implements the Tanh transformation. This maps a Rectangle [low, high] into R^D.
	It is *very* recommended to use this as the last layer of every flow you will ever train on GW data.
	"""
	def __init__(self, D, low = None, high = None):
		"""
		Initialize the transformation.
		
		Parameters
		----------
			D: int
				Dimensionality of the space
		"""
		super().__init__()
			#Placeholders for the true values
			#They will be fitted as a first thing in the training procedure
		if low is None:
			self.low = torch.nn.Parameter(torch.randn([D], dtype=torch.float32), requires_grad = False)
		else:
			self.low = torch.nn.Parameter(torch.tensor(low, dtype=torch.float32), requires_grad = False)
		if high is None:
			self.high = torch.nn.Parameter(torch.randn([D], dtype=torch.float32), requires_grad = False)
		else:
			self.high = torch.nn.Parameter(torch.tensor(high, dtype=torch.float32), requires_grad = False)
	
	def inverse(self, inputs, context=None):
		th_inputs = torch.tanh(inputs)
		outputs = (th_inputs*(self.high-self.low)+self.high+self.low)/2
		logabsdet = torch.log((1 - th_inputs ** 2)*(self.high-self.low)*0.5)
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet

	def forward(self, inputs, context=None):
		inside = torch.logical_and(torch.prod(inputs>self.low, dim = -1), torch.prod(inputs<self.high, dim = -1))
		inputs = inputs.mul(2)
		inputs = inputs.add(-self.high-self.low)
		inputs = inputs.div(self.high-self.low)

		if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
			raise InputOutsideDomain()
		outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
		logabsdet = -torch.log((1 - inputs ** 2)*0.5*(self.high-self.low))
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet


class TanhTransformFrozen(Transform):
	"""
	Implements the Tanh transformation. This maps a Rectangle [low, high] into R^D.
	It is *very* recommended to use this as the last layer of every flow you will ever train on GW data.
	"""
	def __init__(self, D, low = None, high = None):
                """
		Initialize the transformation.
		
		Parameters
		----------
			D: int
				Dimensionality of the space
                """
                super().__init__()
                self.register_buffer("low", torch.Tensor(low))
                self.register_buffer("high", torch.Tensor(high))
	
	def inverse(self, inputs, context=None):
		th_inputs = torch.tanh(inputs)
		outputs = (th_inputs*(self.high-self.low)+self.high+self.low)/2
		logabsdet = torch.log((1 - th_inputs ** 2)*(self.high-self.low)*0.5)
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet

	def forward(self, inputs, context=None):
		inside = torch.logical_and(torch.prod(inputs>self.low, dim = -1), torch.prod(inputs<self.high, dim = -1))
		inputs = inputs.mul(2)
		inputs = inputs.add(-self.high-self.low)
		inputs = inputs.div(self.high-self.low)

		if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
			raise InputOutsideDomain()
		outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
		logabsdet = -torch.log((1 - inputs ** 2)*0.5*(self.high-self.low))
		logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
		return outputs, logabsdet


class IterativeSnapshot_Trainer:
    """
     Transform sequence provided F0, F1, F2, ... Usually F0 will be *fixed* (eg Tanh to get to correct range)
    """
    def __init__(self, bounds: List[Tuple[float, float]], 
                 base_distribution=StandardNormal,
                 transform_list=None):
        self.bounds              = bounds
        self.base_distribution   = base_distribution
        self.transform_list           = transform_list
        self.loss_history = []
        self.flow = None
        
    def train_flow(self, samples_in: List[List[float]],
                   weights: List[float],
                   max_epochs: int, bound_offset: float,n_print=20, n_transforms=0, n_transforms_delta=2,batch_size=1000,val_split=0.2):
        """
            train_flow
               - we build the flow object HERE, to allow it to use flexible transform sets
        """
        # Transform samples by first n_transforms-1 transforms
        assert n_transforms >= 0
        samples= torch.tensor(samples_in, dtype=torch.float32)
        if n_transforms >0:
          transform_before = CompositeTransform(self.transform_list[:n_transforms])
          # just do the transform
          with torch.no_grad():
            samples = transform_before.forward(samples)[0]  # forwards, just need to evaluate -
          print(samples, samples_in)
        # Validation/training split
        full_data = samples_in
        val_size = int(len(samples_in)*val_split)
        train_size = num_samples - val_size

        indices                          = torch.randperm(num_samples)
        train_indices                    = indices[:train_size]
        val_indices                      = indices[train_size:train_size+val_size]
        
        train_data                       = TensorDataset(full_data[train_indices])
        val_data                         = TensorDataset(full_data[val_indices])
        
        train_loader                     = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader                       = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        if True: # n_transforms+n_transforms_delta < len(self.transform_list):
          # try LOCAL optimization of part of flow
          flow_here = Flow(CompositeTransform(self.transform_list[n_transforms:n_transforms+n_transforms_delta]), self.base_distribution(shape=[len(self.bounds)]))
        else:
          # try GLOBAL optimization at end
          flow_here = Flow(CompositeTransform(self.transform_list), self.base_distribution(shape=[len(self.bounds)]))

        optimizer                = optim.Adam(flow_here.parameters(),lr=1e-3,weight_decay=1e-5)
        scheduler                        = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)
        
        prev_efficiency          = float('inf')
        losses                   = self.loss_history
        # Fixed training input
#        lnL_weighted_func = torch.nn.NLLLoss(weight=torch.tensor(weights))
        for epoch in range(1, int(max_epochs)):
            train_loss = 0
            for batch in train_loader:
              # Set gradients equals to zero
              optimizer.zero_grad()
            
              # Compute the loss
              loss                 = -flow_here.log_prob(batch).mean()
            
              # Create necessary derivatives
              loss.backward()
              # Optimize on batch
              optimizer.step()
              train_loss += loss.item()

            # Store loss value for monitoring
            losses.append(loss.item())

            # Compute validation loss
            flow_here.eval()
            val_loss                     = 0
            val_accuracy                 = 0
            val_batch_count              = 0
            with torch.no_grad():
                for batch, in val_loader:
                    val_loss += -flow_here.log_prob(batch).mean()
                    val_batch_count+=1
            val_loss /= val_batch_count
            # Increment scheduler, based on validation loss
            scheduler.step(val_loss)
            if (epoch%n_print)==0:
                print("    Flow training ", epoch, loss.item(), val_loss.item())
                n_hist = np.min([len(losses), 100])
            

        # define flow, so we can draw from it
        self.flow = Flow(CompositeTransform(self.transform_list[:n_transforms+n_transforms_delta]),  self.base_distribution(shape=[len(self.bounds)]) )
        return losses

              
class NFlowsNFS_Trainer:
    def __init__(self, bounds: List[Tuple[float, float]], 
                 target_distribution= StandardNormal,
                 base_distribution=StandardNormal,
                 transform=CompositeTransform):
        self.bounds              = bounds
#        self.target_distribution = target_distribution
        self.base_distribution   = base_distribution
        self.transform           = transform
        self.flow                = None
        self.plotting            = False
        self.loss_history = []
    
    def set_plotting(self, plotting: bool = False) -> None:
        self.plotting            = plotting
    
    def train_flow(self, samples_in: List[List[float]],
                   weights: List[float],
                   max_epochs: int, bound_offset: float,n_print=20,batch_size=1000, **kwargs):
        if self.flow is None:
          self.flow                = Flow(self.transform, self.base_distribution(shape=[len(self.bounds)]))

        optimizer                = optim.Adam(self.flow.parameters(),lr=1e-3, weight_decay=1e-5)
        scheduler                        = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)
                
        prev_efficiency          = float('inf')
        losses                   = self.loss_history
        # Fixed training input. Default use WHOLE data set
        samples              = torch.tensor(samples_in, dtype=torch.float32)
        train_loader                     = DataLoader(samples, batch_size=batch_size, shuffle=True)
#        lnL_weighted_func = torch.nn.NLLLoss(weight=torch.tensor(weights))
        for epoch in range(1, int(max_epochs)):
            n_batch = 0
            train_loss=0
            for batch in train_loader:
              # Set gradients equals to zero
              optimizer.zero_grad()
            
              # Compute the loss
              loss                 = -self.flow.log_prob(batch).mean()
            
              # Create necessary derivatives
              loss.backward()
              optimizer.step()
              train_loss += loss.item()
              n_batch +=1
            
            # Store loss value for monitoring
            loss_now = train_loss/n_batch
            losses.append(loss_now)
            scheduler.step(loss_now)
            if (epoch%n_print)==0:
              print("    Flow training ", epoch, loss_now)

        return losses


class MCSampler(SamplerOutputMixin, MCSamplerGeneric):
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
        self.nf_epoch = 0

        self.mean_affine_set = False
        # pre-trained flow to warm-load (see load_flow); applied inside integrate_log
        self._preloaded_state = None


    def setup(self, nf_method='default',**kwargs):

        self._rvs={}
        self.lnL_thresh = -np.inf
        self.nf_method=nf_method
        d_nf = len(self.params_ordered)

        bounds = np.array([ [self.llim[p],self.rlim[p]] for p in self.params_ordered])

        # https://github.com/bayesiains/nflows/blob/master/examples/moons.ipynb
        self.num_layers = int(len(bounds)/2)  # autoregressive
        n_features = len(bounds)
        transforms = []
        # trivial scale layer first, to get the boundaries in the right place. Use TanhTransform to scale
        if nf_method=='iterative':
          transforms.append(TanhTransformFrozen(len(bounds), low=bounds[:,0], high=bounds[:,1]))
        else:
          transforms.append(PointwiseAffineTransform())
#        transforms.append(LULinear(features=len(bounds)))

        for _ in range(self.num_layers):
            transforms.append(RandomPermutation(features = len(bounds)))
            if self.nf_method =='iterative':
              transforms.append(AffineCouplingTransform(
                mask                    = np.arange(n_features) % 2,
                transform_net_create_fn = lambda in_features, out_features: ResidualNet(in_features, out_features, hidden_features=64, num_blocks=2)))
            else:
              # https://homepages.inf.ed.ac.uk/imurray2/pub/17maf/maf.pdf
              transforms.append(MaskedAffineAutoregressiveTransform(features = len(bounds),
                                                                hidden_features = 2 * len(bounds)) )
        transform  = CompositeTransform(transforms)
        self.nf_model = transform

        if nf_method=='iterative':
          trainer    = IterativeSnapshot_Trainer(bounds              = bounds, 
                                               transform_list           = transforms)
        else:
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
        enforce_bounds = kwargs["enforce_bounds"] if "enforce_bounds" in kwargs else True

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
          flow_samples, flow_log_prob = flow.sample_and_log_prob(n_to_get) # alternate function call
          rv = flow_samples.detach().numpy().T
          log_ps = flow.log_prob(flow_samples).detach().numpy()  # should replace with above and detach
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

    def update_sampling_prior(self, lnw, *args, xpy=xpy_default,no_protect_names=True,external_rvs=None,tempering_exp=1,max_epochs_requested=300,n_history=1000,**kwargs):
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
      print(" Tempering exp " , tempering_exp)
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
      if weights_alt.dtype == RiftFloat:
        weights_alt = weights_alt.astype(numpy.float64,copy=False)


      # copy into training array, truncating size
      samples_train = np.zeros((len(self.params_ordered),n_history_to_use))
      for itr, p in enumerate(self.params_ordered):
                samples_train[itr] = rvs_here[p][-n_history_to_use:]

      # Eliminate points with low weight.  DANGEROUS, adjust
      indx_ok = weights_alt >  np.mean(weights_alt) -1.5*np.std(weights_alt)     
      if np.sum(indx_ok) < 10:
        if super_verbose:
          print(" Skipping update: too few valid ") 
        return
#      print(indx_ok.shape, samples_train.shape)
      samples_train = samples_train[:,indx_ok]
      weights_alt = weights_alt[indx_ok]
      if samples_train.shape[1] < 10:
        if super_verbose:
          print(" Skipping update: too few valid ") 
        return        
      if super_verbose:
        print("    NF: Training data shape ", samples_train.shape)
        print("    NF: training data mean ", np.mean(samples_train,axis=-1))

      if not(self.mean_affine_set) and self.nf_method != 'iterative':
          # pvals = weights_alt / np.sum(weights_alt)   # not going to use proper weighted sample mean, just fix to original hotspot to get close.
          my_mean = torch.as_tensor(np.mean(samples_train.T , axis=0),dtype=torch.float32)
          my_scale = torch.Tensor(np.diag(np.cov(samples_train)) )  # cov is trickier, do NOT use weights since catastrophe possible
          #print(my_mean,np.diag(np.cov(samples_train)).shape)
          tf = PointwiseAffineTransform(scale=1./my_scale,shift=-my_mean/my_scale)
          self.nf_trainer.transform._transforms[0]=tf # affine transform. MUST BE POSITION OF PointwiseAffine ! 
          #print(samples_train.T)
          print("  Training step 0: Set mean of affine transform ", tf._shift, tf._scale) # remember applying in reverse
          self.mean_affine_set=True
        
      # Train
      trainer = self.nf_trainer
      max_epochs = max_epochs_requested  # be short, don't need precision/long training
      if max_epochs < int(10*self.num_layers): max_epochs = int(10*self.num_layers)
      # for iterative trainer: start with fixed first transform, train next wo
      n_transforms=1 + 2*self.nf_epoch
      n_transforms_delta = 2
      
      losses     = trainer.train_flow(samples_in= samples_train.T,
                                      weights=weights_alt,
                                      max_epochs    = max_epochs, 
                                      bound_offset  = 0.5, n_transforms=n_transforms, n_transforms_delta=n_transforms_delta)

      self.nf_flow = trainer.flow
      self.nf_epoch +=1


    ###
    ### FLOW STORAGE / REUSE
    ###
    # Training a normalizing flow is slow and expensive, and only pays off if the
    # trained flow can be RE-USED across the many ILE instances that share similar
    # posterior structure.  These helpers persist a trained flow to a small file
    # and warm-load it into a fresh sampler, which can then either sample from it
    # directly (n_adapt=0) or do a few cheap 'polish' epochs (small n_adapt) to
    # adapt it to the new instance -- amortizing the training cost.

    def save_flow(self, path):
        """Serialize the trained flow (torch state_dict + the architecture
        metadata needed to rebuild it) to `path`."""
        if self.nf_flow is None:
            raise Exception("mcsamplerNFlow.save_flow: no trained flow to save (run integrate first)")
        payload = dict(
            state_dict=self.nf_flow.state_dict(),
            params_ordered=[str(p) for p in self.params_ordered],
            bounds=[[float(self.llim[p]), float(self.rlim[p])] for p in self.params_ordered],
            nf_method=getattr(self, 'nf_method', 'default'),
            num_layers=int(getattr(self, 'num_layers', max(1, len(self.params_ordered) // 2))),
            nf_epoch=int(self.nf_epoch),
        )
        torch.save(payload, path)
        return path

    def load_flow(self, path):
        """Stage a pre-trained flow saved by save_flow().  The weights are applied
        inside integrate_log() (after the architecture is rebuilt), so this must be
        called before integrate/integrate_log.  Verifies the parameters and box
        match this sampler."""
        payload = torch.load(path, map_location='cpu')
        if [str(p) for p in payload['params_ordered']] != [str(p) for p in self.params_ordered]:
            raise ValueError("saved flow params {} != sampler params {}".format(
                payload['params_ordered'], [str(p) for p in self.params_ordered]))
        saved_bounds = np.array(payload['bounds'], dtype=float)
        my_bounds = np.array([[self.llim[p], self.rlim[p]] for p in self.params_ordered], dtype=float)
        if not np.allclose(saved_bounds, my_bounds):
            raise ValueError("saved flow box does not match sampler box")
        self._preloaded_state = payload
        return payload

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
        # The normalizing flow (nflows/torch) samples and scores on the host, and
        # the integrand is a host function, so the NF integrator AGGREGATES on the
        # host.  Force the running estimate onto numpy/scipy regardless of the
        # xpy=cupy default -- mixing a cupy xpy with the flow's host arrays is what
        # crashed the GPU path (cupy rv handed to a numpy integrand).
        xpy = self.xpy            # = numpy
        special_here = special    # scipy.special (host)
        #
        # Determine stopping conditions
        #
        nmax = kwargs["nmax"] if "nmax" in kwargs else float("inf")
        neff = kwargs["neff"] if "neff" in kwargs else RiftFloat("inf")
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
        n_adapt = int(kwargs["n_adapt"]) if "n_adapt" in kwargs else 20  # default to adapt to 10 chunks, then freeze
        floor_integrated_probability = kwargs["floor_level"] if "floor_level" in kwargs else 0
        temper_log = kwargs["tempering_log"] if "tempering_log" in kwargs else False
        tempering_adapt = kwargs["tempering_adapt"] if "tempering_adapt" in kwargs else False
            

        save_intg = kwargs["save_intg"] if "save_intg" in kwargs else False
        # The NF's final integral estimate reads log_integrand/log_joint_prior/
        # log_joint_s_prior back out of self._rvs, so those MUST be accumulated
        # regardless of adaptation.  (Previously save_intg was only turned on when
        # n_adapt>0 and tempering_exp>0, so pure flow REUSE with n_adapt=0 hit a
        # KeyError('log_integrand') at the end.)
        save_intg = True

        deltalnL = kwargs['igrand_threshold_deltalnL'] if 'igrand_threshold_deltalnL' in kwargs else float("Inf") # default is to return all
        deltaP    = kwargs["igrand_threshold_p"] if 'igrand_threshold_p' in kwargs else 0 # default is to omit 1e-7 of probability
        bFairdraw  = kwargs["igrand_fairdraw_samples"] if "igrand_fairdraw_samples" in kwargs else False
        # The fair draw below REPLACES _rvs with an export resample; a consumer that then
        # weights those rows applies w twice.  Record whether it actually FIRED -- the CLI
        # flag is not the same predicate, since the draw is skipped when it would not
        # shrink the record.  Reset per pass: samplers are reused across events.
        self._rvs_is_fairdraw = False
        # The record describes THIS pass only.  Cleared with the flag above and set
        # below, so it can never survive into a pass it does not describe.
        self._rvs_record = None
        n_extr = kwargs["igrand_fairdraw_samples_max"] if "igrand_fairdraw_samples_max" in kwargs else None

        bShowEvaluationLog = kwargs['verbose'] if 'verbose' in kwargs else True
        bShowEveryEvaluation = kwargs['extremely_verbose'] if 'extremely_verbose' in kwargs else False

        nf_method = kwargs['nf_method'] if 'nf_method' in kwargs else 'default'

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
        # a warm-loaded flow dictates the architecture; use its method so the
        # rebuilt transform matches the saved weights
        if self._preloaded_state is not None:
            nf_method = self._preloaded_state.get('nf_method', nf_method)
        self.setup(nf_method=nf_method)

        # WARM-LOAD: rebuild the flow object on the freshly-created architecture
        # and load the pre-trained weights, so this run starts from a trained flow
        # (n_adapt=0 -> pure reuse; small n_adapt -> a few polish epochs).
        if self._preloaded_state is not None:
            flow = Flow(self.nf_model, StandardNormal(shape=[len(self.params_ordered)]))
            flow.load_state_dict(self._preloaded_state['state_dict'])
            self.nf_flow = flow
            self.nf_trainer.flow = flow
            self.mean_affine_set = True   # affine layer is part of the loaded weights
            self.nf_epoch = int(self._preloaded_state.get('nf_epoch', 0))
            if bShowEvaluationLog:
                print("  [NF warm-load] restored pre-trained flow ({} layers)".format(self.num_layers))

        ntotal_true = 0
        max_epochs_requested =300
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
            # rv is a host array from the flow (nflows/torch samples on the host).
            rv = identity_convert(rv)
            params = []
            for item in self.params_ordered:  # USE IN ORDER
                if isinstance(item, tuple):
                    params.extend(item)
                else:
                    params.append(item)
            # Evaluate the integrand.  The real GPU ILE likelihood is DEVICE-native
            # (wants cupy); synthetic/CI integrands are host-native.  Feed
            # device-first, fall back to host on a type error, and remember the
            # choice (same contract as AV / the portfolio).  The flow's own math
            # stays on the host, so lnL is brought back to host afterwards.
            def _eval_integrand(cols):
                if 'no_protect_names' in kwargs:
                    return lnF(*cols)
                return lnF(**dict(list(zip(params, cols))))
            if getattr(self, '_integrand_wants_host', False) or not cupy_ok:
                lnL = _eval_integrand(rv)
            else:
                try:
                    lnL = _eval_integrand(identity_convert_togpu(rv))
                except (TypeError, ValueError):
                    self._integrand_wants_host = True
                    lnL = _eval_integrand(rv)
            # bring lnL back to the host for the flow-side / aggregation math
            lnL = identity_convert(lnL)


            # For now: no prior, just duplicate VT algorithm
#            print(rv.shape, lnL.shape, log_joint_p_prior.shape,log_joint_p_s.shape)
            log_integrand =lnL  + log_joint_p_prior - log_joint_p_s
#            log_weights = tempering_exp*lnL + log_joint_p_prior
            # log aggregate: NOT USED at present, remember the threshold is floating
            if current_log_aggregate is None:
              current_log_aggregate = init_log(log_integrand,xpy=xpy,special=special_here)
            else:
              current_log_aggregate = update_log(current_log_aggregate, log_integrand,xpy=xpy,special=special_here)

            # Monitoring for i/o
            outvals = finalize_log(current_log_aggregate,xpy=xpy)
            self.ntotal = current_log_aggregate[0]
            # effective samples
            maxval = max(maxval, identity_convert(self.xpy.max(log_integrand) ))

            # sum of weights is the integral * the number of points
            eff_samp = xpy.exp(  outvals[0]+np.log(self.ntotal) - maxval)   # integral value minus floating point, which is maximum
            if bShowEvaluationLog:
                print(" :",  self.ntotal, eff_samp, numpy.sqrt(2*maxlnL), numpy.sqrt(2*outvals[0]), outvals[0]-maxlnL, np.exp(outvals[1]/2  - outvals[0]  - np.log(self.ntotal)/2 ))

            # Adapt if needed; decrement adaptation counter
            if n_adapt > 0:
              if super_verbose:
                print("    -- n_adapt {} ".format(n_adapt))
              self.update_sampling_prior(lnL,max_epochs_requested=max_epochs_requested,**kwargs)
              max_epochs_requested +=- 20 # reduce!  Don't constantly overtune
              max_epochs_requested = np.max([max_epochs_requested,20]) # Don't under-tune
              n_adapt += -1  # decrement
            else:
              if super_verbose:
                print("  ... skipping adaptation (NF) ")

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
        # (DESIGN_rvs_naming.md) _rvs is the RETAINED set at this point -- pruned,
        # perhaps, but never resampled.  Record that before the draw below can change what it
        # means, so "not resampled" is a statement the record makes rather than the absence of
        # one.  The reserve rides along BY REFERENCE where the sampler keeps one (AV and the
        # portfolio); None elsewhere is the honest answer, not a gap.
        self._rvs_record = RvsRecord.retained(
            self._rvs, reserve=getattr(self, '_warm_seed_reserve', None))
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


               self._rvs_is_fairdraw = True   # _rvs is now an EXPORT resample, rows already ~ w
               # ...and now it is an export resample.  n_retained comes from that record's
                # PROVENANCE, which captured the count eagerly -- NOT from len(), which reads
                # self._rvs and would return the POST-draw length: the retained record holds a
                # REFERENCE to the live dict this block has just replaced in place.  That is
                # this project's own bug class, so it is spelled out rather than assumed.
               self._rvs_record = RvsRecord.fair_draw(
                   self._rvs, n_retained=self._rvs_record.n_retained(),
                   reserve=getattr(self, '_warm_seed_reserve', None))
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
