import sys
import math
#import bisect
from collections import defaultdict
from types import ModuleType

import numpy
np=numpy #import numpy as np
from RIFT.precision import RiftFloat  # platform-portable replacement for np.float128
from scipy import integrate, interpolate, special
import itertools
import functools

from copy import deepcopy


import os

try:
  import cupy
  import cupyx.scipy.special   # needed for logsumexp
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
  try:
    known_pipelines[pipeline.name] = pipeline.load()
  except Exception as e:
    # optional plugins (e.g. the NF pipeline needing torch) must not make
    # importing mcsamplerPortfolio itself fail on torch-free installations
    print(" Portfolio discovery: SKIPPING {} (unavailable: {})".format(pipeline.name, e))
print('RIFT portfolio plugins:', sorted(known_pipelines))


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
  # if every member is degenerate (n_ess ~ 1, e.g. a very hard target's first
  # chunk found nothing), the normalization below would divide by zero and
  # produce nan weights -> negative per-member sample counts downstream.  Keep
  # the previous (typically uniform) weights instead.
  if np.sum(rewt) <= 0:
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
        # Freeze protection.  A member (esp. a VARAHA/AV workhorse) contributes little on its
        # first chunks -- before it has contracted -- so a plain weight<freeze_wt freeze starves
        # it from chunk 1 and it never gets going.  Instead of freezing permanently:
        #  * GRACE: never freeze during the first `grace_iters` iterations (let members contract);
        #  * REVIVE: every `revive_period` iterations, update the frozen members anyway (one step)
        #    so a starved-but-recoverable member gets periodic chances instead of wasting cycles.
        # Both are overridable via setup(**kwargs).
        self.portfolio_grace_iters = kwargs.get('portfolio_grace_iters', 25)
        self.portfolio_revive_period = kwargs.get('portfolio_revive_period', 8)
        # VARAHA/AV members are a special case.  A VARAHA member ONLY contracts its live
        # volume on the chunk where it is UPDATED (update_sampling_prior_selfish); grace/revive
        # update it only intermittently, so it contracts ~revive_period x slower than a
        # standalone AV that updates every chunk -- it never becomes the workhorse.  Because the
        # balance-heuristic mixture density (q_mix) makes the estimate unbiased for ANY member
        # weights, a continuously-updated VARAHA can only ever COST efficiency (extra selfish
        # draws), never bias the integral.  So by default we make VARAHA members freeze-EXEMPT:
        # once past their activation breakpoint they update every chunk, exactly like standalone
        # AV.  Set portfolio_varaha_never_freeze=False to fall back to the grace/revive schedule
        # (e.g. to save eval cycles on a VARAHA member you know is a genuinely bad fit).
        self.portfolio_varaha_never_freeze = kwargs.get('portfolio_varaha_never_freeze', True)
        # Diagnostic: per-member n_ess history (one list per portfolio member), appended each
        # chunk in the report block.  Enables plateau-aware policies and post-hoc analysis.
        self.portfolio_member_ness_history = [[] for _ in range(len(self.portfolio))]

        # ADAPTIVE-PROBE DRAW ALLOCATION -- OPT-IN (default OFF).  Idea: keep a per-member QUALITY
        # estimate updated only from fair-allocation chunks, allocate draws by quality^exponent, and
        # round-robin PROBE each member at a raised share so a suppressed member can prove itself.
        # q_mix keeps this unbiased for ANY allocation.  On strongly-correlated SYNTHETIC targets it
        # works well (a full-covariance GMM wins the probe and the portfolio beats AV --
        # test_portfolio_adaptive_alloc.py).
        #
        # BUT it is NOT a safe default: the quality signal is each member's per-chunk Kish n_ess,
        # which rewards SELF-CONSISTENCY, not integral coverage.  A warm GMM is instantly
        # self-consistent (per-chunk n_ess ~120) while a warm VARAHA/AV member's per-chunk n_ess is
        # genuinely ~1 during its slow, CUMULATIVE contraction (its value emerges over ~70 chunks).
        # So on a real high-SNR AV-favorable event (S250114ax) adaptive drives the true AV workhorse
        # to the floor and rides the self-consistent-but-worse GMM: measured n_eff 8 vs 53 for the
        # legacy allocation -- a regression.  The probe cannot rescue AV because AV still looks bad
        # at high allocation until fully contracted.  A correct default needs a GLOBAL-impact quality
        # signal (how much a member improves the pooled q_mix n_eff), not per-member self-n_ess; that
        # is future work.  Until then the DEFAULT keeps the legacy n_ess reweighting.
        self.portfolio_adaptive_alloc = kwargs.get('portfolio_adaptive_alloc', False)
        # QUALITY SIGNAL for the allocation:
        #  'global' (default) -- each member's MARGINAL GAIN IN POOLED n_eff PER SAMPLE,
        #      g_m = 2*mean_w_m/S - mean_w2_m/Q  (S=sum w, Q=sum w^2 over ALL samples; see the
        #      derivation where it is computed).  This directly optimizes the quantity we care
        #      about: it credits a member for the weight MASS it contributes and debits it for the
        #      weight VARIANCE it injects.  The two simpler candidates both fail:
        #        * Kish n_ess is SCALE-INVARIANT ((sum w)^2/sum w^2 is unchanged if all w are
        #          scaled), so it cannot see whether a member carries any integral mass at all -- a
        #          self-consistent member sitting off-peak scores as well as one covering the peak.
        #        * mean weight alone REWARDS badly-matched proposals: a well-matched contracted AV
        #          correctly has small uniform weights, while a broad GMM's rare huge-weight outlier
        #          sets the max (measured on S250114ax: AV 1e-40 vs GMM 2e-4 -- backwards).
        #      g_m is also self-correcting at low allocation: a starved peak-covering member sees
        #      inflated weights (q_mix is small there) so it earns share, and as its share grows
        #      q_mix rises and the weights fall -- an equilibrium, with no under-observation trap.
        #  'ness' -- legacy per-member Kish n_ess (kept for comparison; see the S250114ax regression).
        self.portfolio_quality_signal = kwargs.get('portfolio_quality_signal', 'global')
        self.portfolio_alloc_exponent = kwargs.get('portfolio_alloc_exponent', 1.0)  # weights ~ quality^p
        self.portfolio_alloc_floor    = kwargs.get('portfolio_alloc_floor', 0.05)     # min share (coverage+probe)
        self.portfolio_quality_decay  = kwargs.get('portfolio_quality_decay', 0.5)    # EMA alpha for quality
        self.portfolio_probe_period   = kwargs.get('portfolio_probe_period', 4)       # probe one member every N chunks
        self.portfolio_probe_frac     = kwargs.get('portfolio_probe_frac', 0.6)       # raise probed member to >= this
        # WEIGHT CLIPPING (truncated importance sampling) -- OPT-IN, default off, PROPOSAL-FIT INPUT
        # ONLY (see the clipping block in integrate_log).  Cap w at
        #     tau = portfolio_weight_clip * sqrt(n) * mean(w)          (Ionides 2008, truncated IS)
        # Clipping is BIASED and distorts n_ess, so the clipped copy feeds ONLY
        # member.update_sampling_prior (the GMM covariance fit) -- one enormous weight can't make
        # that fit degenerate.  The estimator (ln Z, n_eff), the n_ess report, and the allocation
        # signal all use the TRUE weights, so they stay exactly unbiased/undistorted.  The withheld
        # mass is accumulated in log space and reported as a TAIL DIAGNOSTIC, not a bias.
        self.portfolio_weight_clip = kwargs.get('portfolio_weight_clip', 0.0)  # 0 = off
        self.portfolio_clip_log_removed = -np.inf
        self.portfolio_clip_log_total = -np.inf
        self.portfolio_clip_n = 0
        # diagnostic: samples whose mixture density UNDERFLOWED to 0 and hit the 1e-300 floor (those
        # produce spurious ~1/1e-300 weights -- a numerical artifact, not real tail mass)
        self.portfolio_qmix_underflow = 0
        self.portfolio_quality = np.ones(len(self.portfolio))   # per-member quality (EMA of the signal)
        self.portfolio_quality_nobs = np.zeros(len(self.portfolio), dtype=int)  # #updates per member
        self.portfolio_probe_ptr = 0                            # round-robin probe pointer

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
            # update dictionary limits, yes this is super-redundant, but we have a scoping issue and this is easier to code
            self.llim.update( member.llim)
            self.rlim.update(member.rlim)
            # set master list of adaptive parameters 
            self.adaptive = member.adaptive  # top level list of adaptive coordinates


    def bootstrap_from_samples(self, samples, params=None, **kwargs):
        """Warm-start: forward a seed cloud to every member that supports it (e.g. the
        AV/VARAHA member's live volume).  Members without bootstrap_from_samples are left
        cold.  This is safe: a warm start only ever shapes a member's proposal, and the
        portfolio combines members with the balance-heuristic mixture density (q_mix), so a
        cold or mis-seeded member can only cost efficiency, never bias the estimate.  Column
        order matches self.params_ordered, which every member shares (add_parameter forwards
        to all members in the same order), so no per-member remapping is needed.

        Only VARAHA/AV-style members (those exposing bootstrap_from_samples) are seeded
        directly here.  GMM / adaptive-Gaussian members cannot be seeded pre-integration
        (their internal integrator, hence gmm_dict, does not exist until the first
        integrate() call), but they still warm up STRUCTURALLY during the run: the portfolio
        lets the cold GMM member adapt its proposal from the warm AV member's high-likelihood
        draws, which is what gives the mixture a faster early n_eff than warm-AV alone.  An
        explicit GMM pre-seed (build an initial gmm_dict from the sample cloud) is a future
        enhancement.  q_mix keeps any cold/mis-seeded member from biasing the estimate.

        Returns the number of members warm-started (0 is fine; the portfolio still runs)."""
        samples = np.asarray(samples)
        n_warmed = 0
        for indx, member in enumerate(self.portfolio_realizations):
            if not hasattr(member, 'bootstrap_from_samples'):
                continue
            try:
                member.bootstrap_from_samples(samples, params=params, **kwargs)
                n_warmed += 1
            except Exception as e:
                print("  [portfolio] member {} warm-start skipped ( {} )".format(indx, e))
        print("  [portfolio] warm-started {}/{} members directly (others warm structurally)".format(
            n_warmed, len(self.portfolio_realizations)))
        return n_warmed

    def setup(self,  **kwargs):
        self.extra_args =kwargs  # may need to pass/use during the 'update' step
        # allow the driver/CLI to tune the freeze-protection knobs.  A None means "not set on
        # the CLI" (optparse default), so keep the current value rather than clobbering it.
        def _kw_keep(name):
            v = kwargs.get(name, None)
            if v is not None:
                setattr(self, name, v)
        _kw_keep('portfolio_freeze_wt')
        _kw_keep('portfolio_grace_iters')
        _kw_keep('portfolio_revive_period')
        _kw_keep('portfolio_varaha_never_freeze')
        _kw_keep('portfolio_adaptive_alloc')
        _kw_keep('portfolio_quality_signal')
        _kw_keep('portfolio_alloc_exponent')
        _kw_keep('portfolio_alloc_floor')
        _kw_keep('portfolio_quality_decay')
        _kw_keep('portfolio_probe_period')
        _kw_keep('portfolio_probe_frac')
        _kw_keep('portfolio_weight_clip')
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
        # Iterate the INSTANTIATED samplers (portfolio_realizations), NOT self.portfolio: the
        # latter may hold modules/names (see __init__), which lack .setup(), so member setup was
        # silently skipped -> a cold member's internal state (AV my_ranges, GMM integrator) was
        # never built and draw_simplified failed.  Setting up the realizations fixes AV+GMM cold.
        for indx, member in enumerate(self.portfolio_realizations):
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

    def _adaptive_allocation(self, ness_now, frac_now, iteration):
        """Adaptive-probe draw allocation (see __init__).  Returns the next chunk's per-member
        draw weights.  Decouples a QUALITY estimate (EMA of n_ess, updated only from chunks where
        the member had a fair allocation) from the ALLOCATION (quality^exponent, floored), and
        round-robin PROBES one member per `probe_period` chunks at a raised share so a suppressed
        member can prove itself.  Unbiased for any allocation (q_mix handles correctness)."""
        m = len(self.portfolio)
        if m <= 1:
            return np.ones(m)
        _global = (self.portfolio_quality_signal == 'global')
        _floor_obs = 0.0 if _global else 1.0   # contribution is zero-based; Kish n_ess is >= 1
        obs = np.asarray(ness_now, dtype=float)
        obs = np.where(np.isfinite(obs), obs, _floor_obs)
        frac = np.asarray(frac_now, dtype=float)
        # 1) update QUALITY.  With the 'ness' signal a member drawn at the floor has too few/too
        #    noisy samples to trust, so only fair-allocation chunks count.  The 'global'
        #    contribution signal is SELF-CORRECTING at low allocation (a starved peak-covering
        #    member shows inflated weights), so every chunk is informative -- no gating needed.
        fair = 0.0 if _global else 0.9 / m
        a = self.portfolio_quality_decay
        for k in range(m):
            if frac[k] > 0 and frac[k] >= fair:
                _o = max(obs[k], _floor_obs)
                if self.portfolio_quality_nobs[k] == 0:
                    # first real observation: adopt it outright.  The 'global' contribution signal
                    # has an arbitrary scale, so EMA-ing from the placeholder 1.0 would bias it.
                    self.portfolio_quality[k] = _o
                else:
                    self.portfolio_quality[k] = (1 - a) * self.portfolio_quality[k] + a * _o
                self.portfolio_quality_nobs[k] += 1
        # 2) base allocation ~ quality^exponent above a floor.  For 'ness' we use the EXCESS over the
        #    degenerate n_ess=1 (a member at n_ess 1 contributes nothing); the 'global' contribution
        #    is already zero-based.
        q = np.maximum(self.portfolio_quality - _floor_obs, 0.0)
        if np.sum(q) <= 0:
            base = np.ones(m) / m
        else:
            w = (q / np.sum(q)) ** self.portfolio_alloc_exponent
            base = w / np.sum(w)
        base = self.portfolio_alloc_floor + base * (1.0 - m * self.portfolio_alloc_floor)
        base = base / np.sum(base)
        # 3) round-robin probe: raise ONE member to >= probe_frac every probe_period chunks so an
        #    under-observed member gets a fair look next chunk (breaks the under-observation trap).
        if self.portfolio_probe_period > 0 and (iteration % self.portfolio_probe_period == 0):
            k = self.portfolio_probe_ptr % m
            self.portfolio_probe_ptr += 1
            if base[k] < self.portfolio_probe_frac:
                base = base * (1.0 - self.portfolio_probe_frac) / max(1e-12, 1.0 - base[k])
                base[k] = self.portfolio_probe_frac
                base = base / np.sum(base)
        return base

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
           only_member = self.portfolio_realizations[indx_active[0]]
           joint_p_s, joint_p_prior, rv = only_member.draw_simplified(n_samples, *self.params_ordered, **kwargs)
           # The portfolio aggregates on the host (self.xpy is numpy); members
           # may be GPU-backed (cupy), so bring their draws to the host.
           joint_p_s = identity_convert(joint_p_s); joint_p_prior = identity_convert(joint_p_prior); rv = identity_convert(rv)
           # Record which members produced this chunk, and each member's SAMPLING
           # FRACTION (n_from_member / n_total).  integrate_log uses these to form
           # the balance-heuristic mixture density q_mix = sum_m frac_m * q_m.
           self._chunk_members = [only_member]
           self._chunk_fractions = np.array([1.0])
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
            # Bring member draws to the host backend the portfolio aggregates in
            # (self.xpy is numpy).  identity_convert is cupy.asnumpy when a member
            # is GPU-backed, else a no-op.  (The previous isinstance(type(x),..)
            # guard never fired, leaving cupy arrays to collide with numpy ones.)
            joint_p_s_here = identity_convert(joint_p_s_here)
            joint_p_prior_here = identity_convert(joint_p_prior_here)
            rv_here = identity_convert(rv_here)
            indx_start = int(n_index_start_per_member[indx_member])
            indx_end = indx_start + int(n_samples_per_member[indx_member])
            joint_p_s[indx_start:indx_end] = joint_p_s_here
            joint_p_prior[indx_start:indx_end] = joint_p_prior_here
            rv[:,indx_start:indx_end] = rv_here

          # Record the ACTUAL per-member sampling fractions for this chunk (the
          # counts actually drawn, not the raw portfolio_weights).  These are the
          # w_m in the balance-heuristic mixture density q_mix = sum_m w_m q_m
          # that integrate_log builds; using the true drawn fractions is what
          # keeps the deterministic-mixture estimator exactly unbiased.
          self._chunk_members = list(portfolio_active)
          self._chunk_fractions = np.array(n_samples_per_member, dtype=float) / float(n_samples)

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
        # The portfolio AGGREGATES on the host: draw() brings member draws to the
        # host, so force the running-estimate math onto numpy/scipy regardless of
        # what the driver set self.xpy to (it sets cupy for GPU members).  Members
        # still do their own heavy sampling/adaptation on their own backend.  The
        # INTEGRAND, however, may be device-native (the real vectorized GPU ILE
        # likelihood) or host-native (synthetic/CI): _eval_integrand() below feeds
        # it device-first and falls back to host, so both work.
        self.xpy = numpy
        xpy_here = numpy
        xpy = numpy
        special_here = special    # scipy.special (host); statutils uses this

        #
        # Determine stopping conditions
        #
        nmax = kwargs["nmax"] if "nmax" in kwargs else float("inf")
        neff = kwargs["neff"] if "neff" in kwargs else RiftFloat("inf")
        n = int(kwargs["n"] if "n" in kwargs else min(100000, nmax))
        convergence_tests = kwargs["convergence_tests"] if "convergence_tests" in kwargs else None
        save_no_samples = kwargs["save_no_samples"] if "save_no_samples" in kwargs else None
        portfolio_wt_func = kwargs['portfolio_schedule'] if 'portfolio_schedule' in kwargs else portfolio_default_weights
        # allow a per-integration override of the adaptive-probe allocation (else use the instance
        # default set at init/setup); falling back to the legacy n_ess reweighting when off.
        use_adaptive_alloc = kwargs.get('portfolio_adaptive_alloc', self.portfolio_adaptive_alloc)
        # per-integration override of the weight clip (else the instance default from init/setup)
        if 'portfolio_weight_clip' in kwargs:
            self.portfolio_weight_clip = kwargs['portfolio_weight_clip']

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
            # Evaluate the integrand.  rv is on the host (see draw()).  The real
            # GPU ILE likelihood is DEVICE-native (wants cupy); synthetic/CI
            # integrands are host-native.  Feed device-first, fall back to host on
            # a type error, and remember the choice (same contract as the AV
            # integrator).  lnL is brought back to the host for aggregation.
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
            # bring lnL back to the host for the host-side aggregation
            lnL = identity_convert(lnL)

            # ---- BALANCE-HEURISTIC (deterministic-mixture) sampling density ----
            # The pooled draw is, by construction, a sample from the MIXTURE
            #     q_mix(theta) = sum_m frac_m * q_m(theta),
            # where frac_m = (# samples member m contributed)/n for THIS chunk and
            # q_m is member m's own sampling density evaluated at theta.  Using
            # q_mix (rather than each sample's own member density -- the previous
            # STRATIFIED estimator) makes the estimate unbiased for ANY member
            # weights, provided the mixture covers the peak (Veach & Guibas MIS
            # balance heuristic).  A broad member with even a small weight then
            # guarantees coverage, so a wrongly-contracted member can no longer
            # drive the integral low.  See portfolio_default_weights: n_ess-based
            # weighting made the old stratified denominator UNSAFE.
            #
            # We require EVERY active member (that drew >0 samples) to expose a
            # sampling_density; if any does not (e.g. an AC-histogram member with
            # no pointwise density yet), we fall back to the legacy per-member
            # joint_p_s so those portfolios keep running unchanged.
            use_mixture = kwargs['portfolio_use_mixture_density'] if 'portfolio_use_mixture_density' in kwargs else True
            q_mix = None
            if use_mixture:
                X_all = numpy.asarray(identity_convert(rv), dtype=float)
                if X_all.shape[0] == len(self.params_ordered):
                    X_all = X_all.T   # -> (N, ndim), columns in params_ordered order
                members_here = getattr(self, '_chunk_members', [])
                fracs_here = getattr(self, '_chunk_fractions', None)
                if len(members_here) > 0 and fracs_here is not None:
                    acc = numpy.zeros(X_all.shape[0], dtype=float)
                    all_ok = True
                    any_active = False
                    for frac_m, member_m in zip(fracs_here, members_here):
                        if frac_m <= 0:
                            continue   # member drew nothing this chunk
                        dens_fn = getattr(member_m, 'sampling_density', None)
                        q_m = dens_fn(X_all) if dens_fn is not None else None
                        if q_m is None:
                            all_ok = False
                            break
                        acc = acc + float(frac_m) * numpy.asarray(identity_convert(q_m), dtype=float)
                        any_active = True
                    if all_ok and any_active:
                        # every pooled sample was drawn by some active member, so
                        # q_mix >= frac*q_m(own) > 0 there; floor only guards FP.
                        # UNDERFLOW DIAGNOSTIC: mathematically acc>0 for every drawn sample, so any
                        # acc==0 is a floating-point UNDERFLOW of the linear-space density sum.  The
                        # 1e-300 floor then turns it into a spurious ~target/1e-300 weight, which can
                        # single-handedly dominate the pooled estimator.  Count these so a numerical
                        # artifact can be told apart from genuine heavy-tailed weights (the former
                        # wants a log-space q_mix / member fix, the latter wants weight clipping).
                        _n_uf = int(numpy.sum(acc <= 0))
                        if _n_uf > 0:
                            self.portfolio_qmix_underflow += _n_uf
                            print("  PORTFOLIO: q_mix UNDERFLOW on {}/{} samples this chunk"
                                  " (density summed to 0 -> floored 1e-300 -> spurious huge weight;"
                                  " cumulative {})".format(_n_uf, len(acc), self.portfolio_qmix_underflow))
                        q_mix = numpy.maximum(acc, 1e-300)
            if q_mix is not None:
                joint_p_s = q_mix   # deterministic-mixture denominator
            else:
                if use_mixture and getattr(self, '_warned_no_mixture', False) is False:
                    print(" PORTFOLIO: some active member lacks sampling_density; "
                          "falling back to legacy stratified per-member density.")
                    self._warned_no_mixture = True

            log_integrand =lnL + self.xpy.log(joint_p_prior) - self.xpy.log(joint_p_s)
            # tempering_exp done inside the update proposal, NOT here
            log_weights = lnL + self.xpy.log(joint_p_prior) - self.xpy.log(joint_p_s)
            # NaN guard: a frozen/degenerate member (e.g. an un-contracted VARAHA) can emit NaN
            # samples -> NaN lnL / joint_p_s -> NaN weights, which otherwise propagate into the
            # aggregation and the reported crash ("boolean index did not match ... 10000 vs 9882"
            # when a NaN mask is applied downstream).  Map any non-finite weight to -inf (zero
            # weight) IN PLACE, keeping the array length fixed so no mask-size mismatch can arise.
            _bad = ~self.xpy.isfinite(log_integrand)
            if bool(self.identity_convert(self.xpy.any(_bad))):
                log_integrand = self.xpy.where(_bad, -self.xpy.inf, log_integrand)
                log_weights   = self.xpy.where(_bad, -self.xpy.inf, log_weights)

            # WEIGHT CLIPPING (truncated IS; OPT-IN) -- PROPOSAL-TRAINING INPUT ONLY.
            # Clipping is BIASED and also distorts n_ess, so its scope is deliberately narrow: it
            # produces log_weights_adapt, which is fed ONLY to member.update_sampling_prior (the GMM
            # covariance fit), so that a single enormous weight cannot make that fit degenerate.
            # Everything else uses the TRUE weights:
            #   * log_integrand -> the ESTIMATE (ln Z, eff_samp): UNCLIPPED -> exactly unbiased.
            #   * the per-member n_ess REPORT and the ALLOCATION signal: UNCLIPPED -> undistorted
            #     (clipping flattens weights and would INFLATE the clipped member's Kish n_ess,
            #      perversely rewarding the very member whose weights had to be clipped).
            # This is also strictly better than DROPPING a chunk that clipped: dropping conditional
            # on "a big weight appeared" is data-dependent selection and would bias ln Z low.  The
            # tracked withheld mass is a TAIL DIAGNOSTIC (how much weight the proposal fit ignored).
            log_weights_adapt = log_weights
            if self.portfolio_weight_clip and self.portfolio_weight_clip > 0:
                _lw = numpy.asarray(self.identity_convert(log_weights), dtype=float)
                _fin = numpy.isfinite(_lw)
                if bool(numpy.any(_fin)):
                    _mx = float(numpy.max(_lw[_fin]))
                    _u = numpy.where(_fin, numpy.exp(_lw - _mx), 0.0)
                    _n_here = max(1, len(_u))
                    _total = float(numpy.sum(_u))
                    _tau = self.portfolio_weight_clip * numpy.sqrt(_n_here) * (_total / _n_here)
                    if _total > 0:
                        self.portfolio_clip_log_total = numpy.logaddexp(
                            self.portfolio_clip_log_total, numpy.log(_total) + _mx)
                    _over = _u > _tau
                    _n_over = int(numpy.sum(_over))
                    if _n_over > 0 and _tau > 0:
                        _removed = float(numpy.sum(_u[_over] - _tau))
                        if _removed > 0:
                            self.portfolio_clip_log_removed = numpy.logaddexp(
                                self.portfolio_clip_log_removed, numpy.log(_removed) + _mx)
                        self.portfolio_clip_n += _n_over
                        _u = numpy.minimum(_u, _tau)
                        # ADAPTATION copy only -- log_integrand (the estimator) is deliberately
                        # untouched, so ln Z and n_eff remain exactly unbiased.
                        log_weights_adapt = numpy.where(_u > 0,
                                                        numpy.log(numpy.maximum(_u, 1e-300)) + _mx,
                                                        -numpy.inf)
                        _frac = float(numpy.exp(self.portfolio_clip_log_removed
                                                - self.portfolio_clip_log_total)) \
                            if numpy.isfinite(self.portfolio_clip_log_removed) else 0.0
                        _frac = min(max(_frac, 0.0), 1.0 - 1e-15)
                        print("  PORTFOLIO: proposal-fit weight-clip tau={:.3e}(rel max) clipped {} "
                              "this chunk ({} total); cumulative tail mass withheld from PROPOSAL FIT"
                              " ={:.3e} (estimator + n_ess report + allocation all unclipped)"
                              .format(_tau, _n_over, self.portfolio_clip_n, _frac))

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
              current_log_aggregate = init_log(log_integrand,xpy=xpy,special=special_here)
            else:
              current_log_aggregate = update_log(current_log_aggregate, log_integrand,xpy=xpy,special=special_here)
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

            # GLOBAL-IMPACT signal: each member's MARGINAL GAIN IN POOLED n_eff PER SAMPLE.
            # Pooled Kish n_eff = S^2/Q with S = sum(w), Q = sum(w^2) over ALL members' samples.
            # One extra sample from member m adds (in expectation) mean_w_m to S and mean_w2_m to Q,
            # so d(n_eff)/dn_m divided by n_eff gives the relative per-sample gain
            #     g_m = 2*mean_w_m/S  -  mean_w2_m/Q .
            # This is the quantity the allocation should maximize: it credits a member for the
            # weight MASS it supplies but debits it for the weight VARIANCE it injects, so an
            # outlier-heavy broad member (a few enormous weights) scores LOW or negative -- those
            # outliers are precisely what destroys pooled n_eff.  Note both simpler candidates fail:
            # Kish n_ess is SCALE-INVARIANT (blind to whether a member carries any integral mass),
            # and mean weight alone REWARDS badly-matched proposals (a well-matched, contracted AV
            # correctly has small uniform weights, while a broad GMM's rare huge-weight outlier sets
            # the maximum) -- measured on S250114ax, mean weight ranked AV at 1e-40 vs GMM 2e-4.
            # A single global normalization (the chunk's max log-weight) keeps members comparable.
            # NB: the report n_ess and the allocation signal use the TRUE (unclipped) weights.
            # Feeding them the clipped copy is WRONG: clipping flattens weights, which INFLATES a
            # member's Kish n_ess, so the allocation would perversely favor exactly the member whose
            # weights had to be clipped (measured: on S250114ax it starved the AV workhorse to the
            # 1% floor and collapsed n_eff to ~1).  Clipping's ONLY job is to protect proposal FITS.
            _lw_all = numpy.asarray(self.identity_convert(log_weights), dtype=float)
            _finite = numpy.isfinite(_lw_all)
            _lw_max = float(numpy.max(_lw_all[_finite])) if bool(numpy.any(_finite)) else 0.0
            _u_all = numpy.where(_finite, numpy.exp(_lw_all - _lw_max), 0.0)
            _S_tot = float(numpy.sum(_u_all)); _Q_tot = float(numpy.sum(_u_all * _u_all))
            contrib_per_sample = numpy.zeros(len(self.portfolio))

            portfolio_report = {}
            for indx_member, member in enumerate(self.portfolio):
              indx_start = int(n_index_start_per_member[indx_member])
              indx_end = indx_start + int(n_samples_per_member[indx_member])
              _n_here = max(1, indx_end - indx_start)
              _u_here = _u_all[indx_start:indx_end]
              if _S_tot > 0 and _Q_tot > 0 and indx_end > indx_start:
                _mean_w = float(numpy.sum(_u_here)) / _n_here
                _mean_w2 = float(numpy.sum(_u_here * _u_here)) / _n_here
                contrib_per_sample[indx_member] = 2.0 * _mean_w / _S_tot - _mean_w2 / _Q_tot
              ln_wt_here =  log_weights[indx_start:indx_end]  # TRUE weights (see note above); not the clipped copy
              ln_wt_here += - np.max(ln_wt_here)
              # evaluate  n_ess, n_eff for this set of samples in batch specifically,
              portfolio_report[indx_member] = [ self.portfolio_weights[indx_member], self.identity_convert(self.xpy.sum(self.xpy.exp(ln_wt_here))**2/self.xpy.sum(self.xpy.exp(ln_wt_here*2))), identity_convert(self.xpy.sum(self.xpy.exp(ln_wt_here)))]
            print("\t",portfolio_report)
            if use_adaptive_alloc and len(self.portfolio) > 1:
              print("\t contrib/sample (global-impact signal):", numpy.array2string(contrib_per_sample, precision=3),
                    " quality:", numpy.array2string(np.asarray(self.portfolio_quality, dtype=float), precision=3))
            # Record each member's per-chunk n_ess so freeze policies (and post-hoc analysis)
            # can tell a member that is still CLIMBING from one that has PLATEAUED.
            for indx_member in range(len(self.portfolio)):
              self.portfolio_member_ness_history[indx_member].append(float(portfolio_report[indx_member][1]))
            # Weight based on n_ESS from batch.  remember these are >=1, so no negatives or 0 will happen
            dat =np.array([ portfolio_report[k][1] for k in range(len(self.portfolio))])
            if use_adaptive_alloc and len(self.portfolio) > 1:
              # adaptive-probe allocation: quality-EMA + round-robin probe (see _adaptive_allocation).
              # frac_now = the fraction each member actually drew THIS chunk (n_samples_per_member is
              # derived from self.portfolio_weights just above).  The quality OBSERVABLE is either the
              # global-impact contribution (default) or the legacy per-member Kish n_ess.
              frac_now = np.array(n_samples_per_member, dtype=float) / float(max(1, n_samples))
              _obs = contrib_per_sample if self.portfolio_quality_signal == 'global' else dat
              self.portfolio_weights = self._adaptive_allocation(_obs, frac_now, self.portfolio_draw_iteration)
            else:
              self.portfolio_weights = portfolio_wt_func(dat, self.portfolio_weights, xpy=self.xpy, identity_convert=self.identity_convert) # call weighting function

              
            ###
            ### ORACLE BLOCK
            ###
            # Oracles PROPOSE points (hill-climb hotspots, a Fisher/Gaussian, a
            # previous posterior).  We evaluate the true likelihood there and
            # APPEND those (point, weight) pairs to the training data the other
            # portfolio members adapt from, so they learn about regions the plain
            # sampling missed.  Oracles never enter the integral estimate itself,
            # so they cannot bias it -- at worst they cost a few evaluations.
            rvs_train = self._rvs
            log_weights_train = log_weights_adapt   # weights aligned with rvs_train tail (clipped copy)
            if it_now < it_max_oracle and len(self.oracle_realizations )>0:
              rvs_train = deepcopy(self._rvs)  # duplicate deeply, since we will append to it
              n_samples_per_oracle = int(n*0.1/len(self.oracle_realizations)) # try to minimize oracle effort
              if n_samples_per_oracle > 0:
                print(" ORACLE: attempting updates ")
                # update each oracle from the current (host) history
                for member in self.oracle_realizations:
                  member.update_sampling_prior(log_weights_adapt, n_history, external_rvs=rvs_train, log_scale_weights=True)
                # generate proposals from oracles (oracles are host/numpy)
                rv_list = []
                for member in self.oracle_realizations:
                  _, _, rv_here = member.draw_simplified(n_samples_per_oracle)
                  rv_list.append(numpy.asarray(identity_convert(rv_here)))  # (n, ndim) host
                rv_oracle = numpy.vstack(rv_list)  # host, (n_oracle_total, ndim)
                # evaluate the true integrand at the proposals; feed it device-first
                # (real GPU ILE likelihood) with host fallback, mirroring the main loop
                _cols = rv_oracle.T
                if getattr(self, '_integrand_wants_host', False) or not cupy_ok:
                  _lnLo = lnF(*_cols) if 'no_protect_names' in kwargs else lnF(**dict(zip(self.params_ordered, _cols)))
                else:
                  try:
                    _colsg = identity_convert_togpu(_cols)
                    _lnLo = lnF(*_colsg) if 'no_protect_names' in kwargs else lnF(**dict(zip(self.params_ordered, _colsg)))
                  except (TypeError, ValueError):
                    self._integrand_wants_host = True
                    _lnLo = lnF(*_cols) if 'no_protect_names' in kwargs else lnF(**dict(zip(self.params_ordered, _cols)))
                lnL_oracles = numpy.asarray(identity_convert(_lnLo))
                # training weight for a proposal = its lnL (same log scale as
                # log_weights up to the shared normalization the members remove)
                log_w_oracle = lnL_oracles
                # ACTUALLY append (numpy.append is not in-place -- must reassign)
                for indx, p in enumerate(self.params_ordered):
                  base = identity_convert(rvs_train[p])
                  rvs_train[p] = numpy.append(base, rv_oracle[:, indx])
                log_weights_train = numpy.append(identity_convert(log_weights_adapt), log_w_oracle)


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
                _is_varaha = hasattr(member, 'is_varaha')
                # VARAHA EXEMPTION: a VARAHA/AV member contracts its live volume ONLY on the chunk
                # it is updated, so it must update EVERY chunk (like standalone AV) to become the
                # workhorse.  q_mix keeps this unbiased regardless of weight, so exempt it from the
                # freeze schedule entirely by default (see portfolio_varaha_never_freeze).
                _varaha_exempt = _is_varaha and self.portfolio_varaha_never_freeze
                # GRACE: don't freeze anyone during the first grace_iters iterations (let a slow
                # starter like a VARAHA member contract before its weight is judged).
                _in_grace = (self.portfolio_draw_iteration <= self.portfolio_grace_iters)
                # REVIVE: periodically update even a frozen member so it gets a chance to recover
                # instead of being starved forever.
                _revive = (self.portfolio_revive_period > 0
                           and (self.portfolio_draw_iteration % self.portfolio_revive_period == 0))
                # PLATEAU-AWARE revive: also update a low-weight member while its OWN per-chunk
                # n_ess is still climbing (it is still learning); only let the freeze schedule
                # govern a member that has plateaued.  Uses the n_ess history recorded above.
                _climbing = False
                _hist = self.portfolio_member_ness_history[indx]
                if len(_hist) >= 3:
                  _recent = _hist[-1]; _older = np.median(_hist[-3:-1])
                  _climbing = (_recent > 1.05*max(_older, 1.0))
                if self.portfolio_draw_iteration < self.portfolio_breakpoints[indx]:
                  print("  - before activation breakpoint for member {} ".format( indx))
                  pass
                elif (len(self.oracle_realizations) > 0 and it_now <it_max_oracle) or (self.portfolio_weights[indx] > self.portfolio_freeze_wt) or _in_grace or _revive or _varaha_exempt or _climbing:
                  if not(_is_varaha):
                    # log_weights_train / rvs_train include any oracle proposals appended above
                    member.update_sampling_prior(log_weights_train, n_history,external_rvs=rvs_train,log_scale_weights=True, **update_dict)
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
