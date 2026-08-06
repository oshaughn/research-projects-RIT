import sys
import math
#import bisect
from collections import defaultdict
from types import ModuleType, FunctionType, BuiltinFunctionType

import numpy
np=numpy #import numpy as np
from RIFT.precision import RiftFloat  # platform-portable replacement for np.float128
from scipy import integrate, interpolate, special
import itertools
import functools

from copy import deepcopy, copy as shallow_copy


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
        # PLATEAU-AWARE REVIVE (OPT-IN, default OFF): also update a low-weight member while its
        # own per-chunk n_ess is still climbing.  It sounded strictly helpful, but the shape
        # merge gate showed it SYSTEMATICALLY HALVES portfolio n_eff: forcing updates of members
        # the freeze schedule would have parked makes their proposals worse, not better.
        # Isolated on the gate's own targets (plateau ON -> OFF, vs gate base):
        #   d4_n1_s101 25.9 -> 53.5 (base 53.5)   d4_n3_s202 29.1 -> 83.8 (base 83.8)
        #   d6_n1_s202 64.0 -> 102.1 (base 102.1)  d6_n3_s202  7.2 -> 31.4 (base 31.4)
        #   d8_n1_s101 37.3 -> 61.9 (base 61.9)
        # i.e. OFF reproduces base EXACTLY, so this was the sole default-path regression in
        # PR #28 (never-freeze was measured to be a no-op on these targets, ratio 1.00).
        # Kept available for experimentation; DO NOT default it on without re-running the gate.
        self.portfolio_plateau_revive = kwargs.get('portfolio_plateau_revive', False)
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
        # VARAHA DRAW FLOOR (opt-in, default 0 = off).  never-freeze guarantees a VARAHA/AV member
        # keeps UPDATING (contracting), but nothing guarantees it keeps DRAWING: both allocation
        # rules score members by per-chunk n_ess, and a VARAHA member's per-chunk n_ess sits at ~1
        # during its slow CUMULATIVE contraction, so a member that looks instantly good (a live GMM)
        # can take almost the whole budget.  Measured on S250114ax after the PR #33 fixes made the
        # GMM member genuinely live: the allocation gave GMM ~0.84 and the portfolio collapsed to
        # n_eff ~2 at 4M, versus ~100 for standalone AV.  Setting this to f reserves a combined
        # fraction f of the draws for VARAHA members (applied AFTER whichever allocation rule runs,
        # so it protects the legacy and adaptive paths alike).  q_mix keeps any allocation unbiased,
        # so this only trades efficiency.
        self.portfolio_varaha_min_frac = kwargs.get('portfolio_varaha_min_frac', 0.0)
        self.portfolio_varaha_max_frac = kwargs.get('portfolio_varaha_max_frac', 0.0)  # 0 = no cap
        # Range restriction (see setup(): portfolio_restrict_ranges).  Default OFF, so every code
        # path guarded by these is inert unless a member is explicitly narrowed.
        self._has_restricted_member = False
        self._full_support_members = []
        self._pending_range_overrides = set()  # (member, param) awaiting add_parameter; see setup()
        # Keep the full-support backstop COLD when warm-starting.  Seeding every member removes
        # the mixture's coverage of the prior box (cover_frac cannot restore it -- see
        # bootstrap_from_samples), which turns a mis-placed seed from an efficiency cost into a
        # silent low bias.  Set False to restore the old seed-everything behaviour.
        self.portfolio_warmstart_backstop_cold = True
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

        # PER-MEMBER RANGE OVERRIDES for interval narrowing: {member_index: {param: (lo, hi)}}.
        # Populate with restrict_member_range() BEFORE add_parameter()/setup().  Member 0 is the
        # designated full-support backstop and must never be narrowed.
        self.member_range_overrides = {}

    def restrict_member_range(self, member_index, param, lo, hi):
        """Narrow ONE portfolio member's sampling range for `param` to [lo, hi] (interval narrowing).

        This is a PROPOSAL-only change: the member's prior callables are untouched, so it keeps
        reporting the true global prior, and the balance-heuristic mixture density q_mix keeps the
        estimate unbiased with no renormalization -- PROVIDED at least one member retains full
        support.  Member 0 is that backstop by convention and may not be narrowed.

        Why the backstop is not optional: measured on a truth-known ladder, a WRONG sub-box costs a
        STANDALONE sampler up to -1949 nats (while still reporting a healthy n_eff of 220-840, i.e.
        confidently wrong), but only ~1 nat inside a portfolio whose full-box member keeps q_mix
        covering the complement.  Restriction without a backstop converts a rare pathology into a
        systematic one.

        Call before add_parameter(); narrowing is applied there, and setup() then builds every
        derived quantity from the narrowed range.
        """
        n_members = len(self.portfolio_realizations)
        member_index = int(member_index)
        # Validate STRICTLY: a negative index or a misspelled parameter used to be accepted here and
        # then silently fail to match the positive enumerate()/params checks in add_parameter, so the
        # call "succeeded" while applying no restriction at all.
        if member_index == 0:
            raise ValueError(
                "mcsamplerPortfolio.restrict_member_range: member 0 is the full-support backstop and "
                "must not be narrowed -- q_mix would then have no component covering the complement, "
                "and a mode outside every sub-box becomes uncoverable rather than merely under-covered.")
        if not (1 <= member_index < n_members):
            raise ValueError("restrict_member_range: member_index must satisfy 1 <= i < {} (got {}); "
                             "negative indices are NOT accepted -- they never match the positive "
                             "enumerate() in add_parameter and would silently be a no-op.".format(
                                 n_members, member_index))
        if not (hi > lo):
            raise ValueError("restrict_member_range: need hi > lo, got [{}, {}]".format(lo, hi))
        self.member_range_overrides.setdefault(member_index, {})[param] = (float(lo), float(hi))
        # Track for the consumed-check in setup(): a parameter name that never arrives via
        # add_parameter must be an ERROR, not a silent no-op.
        self._pending_range_overrides = getattr(self, '_pending_range_overrides', set())
        self._pending_range_overrides.add((member_index, param))

        # CENTRALISE the coverage invariants: these are the SAME flags the
        # setup(portfolio_restrict_ranges=...) path establishes.  Setting them only there meant this
        # public API disabled the full-support draw floor, the restricted-only active-member guard and
        # the q_mix fallback guard -- so member 0 could be allocated zero draws and the mixture could
        # silently lose full support, which is precisely the failure restriction is supposed to avoid.
        restricted = set(self.member_range_overrides)
        if len(restricted) >= n_members:
            raise ValueError(
                "restrict_member_range: that would restrict EVERY member, leaving no component with "
                "full support.  The mixture would not cover L*p outside the sub-boxes and the integral "
                "would be biased low with no diagnostic.  Leave at least one member unrestricted.")
        self._has_restricted_member = True
        self._full_support_members = [i for i in range(n_members) if i not in restricted]

    def add_parameter(self, params, pdf,  **kwargs):
        """
        Add one (or more) parameters to sample dimensions. params is either a string describing the parameter, or a tuple of strings. The tuple will indicate to the sampler that these parameters must be sampled together. left_limit and right_limit are on the infinite interval by default, but can and probably should be specified. If several params are given, left_limit, and right_limit must be a set of tuples with corresponding length. Sampling PDF is required, and if not provided, the cdf inverse function will be determined numerically from the sampling PDF.
        """
        self.params.add(params) # does NOT preserve order in which parameters are provided
        self.params_ordered.append(params)
        _all_members = self.portfolio_realizations + self.oracle_realizations
        for indx, member in enumerate(_all_members):
            member.add_parameter(params, pdf, **kwargs)
            # The PORTFOLIO's own limits must always describe the FULL prior range, never a
            # restricted member's sub-box: they are the reference range used downstream (L0-rescue
            # puff width, breadcrumb bounds, distance-marginalization bounds).  Take them from
            # member 0, which is the designated FULL-SUPPORT member by convention (see
            # restrict_member_range), and take them BEFORE any narrowing is applied below.
            if indx == 0:
                self.llim.update( member.llim)
                self.rlim.update(member.rlim)
            # set master list of adaptive parameters
            self.adaptive = member.adaptive  # top level list of adaptive coordinates

        # PER-MEMBER RANGE RESTRICTION (interval narrowing).
        # At high SNR the posterior can occupy a vanishing fraction of the prior box, so a member
        # confined to a well-chosen sub-box resolves it far better (measured: n_eff 2495 vs 1629, and
        # SNR-INDEPENDENT, on the truth-known ladder).  We do this by narrowing ONE member's limits
        # rather than clipping the prior, which is what makes it safe:
        #   * The estimator weight is L*p_prior/q_mix with p_prior the TRUE prior.  A member's range
        #     is purely a PROPOSAL choice -- proposals need not cover the prior, only the MIXTURE
        #     must cover the support of L*p.  So NO prior renormalization and NO clipped-volume
        #     correction are required, PROVIDED a full-support member remains (see _full_support_members).
        #   * We must NOT rebuild the prior callables for the narrowed member: `prior_prod` evaluates
        #     the callables handed to add_parameter, and those are absolute densities normalized over
        #     the ORIGINAL range.  Sharing them is what keeps every member reporting the SAME true
        #     prior -- the portfolio takes joint_p_prior from whichever member drew each sample, so
        #     a member that renormalized its prior over its sub-box would silently bias the integral.
        #     Hence we only overwrite llim/rlim here, and only AFTER add_parameter has installed the
        #     shared callables.
        # Narrowing happens before setup(), so every derived AV quantity (my_ranges, dx, dx0, V,
        # binunique, ninbin) is built from the narrowed range and nothing is left stale.
        for indx, member in enumerate(self.portfolio_realizations):
            _ov = self.member_range_overrides.get(indx)
            if not _ov or params not in _ov:
                continue
            lo, hi = _ov[params]
            # NARROW ONLY.  Widening past the member's own limits would sample where the SHARED
            # prior callables (normalized over the ORIGINAL range) are not normalized, so the
            # member would report a prior density that is wrong outside the original box -- a
            # biased integral, silently.  The name says "restrict"; refuse rather than quietly
            # clip, so a caller who meant to widen finds out instead of getting a no-op.
            if lo < member.llim[params] or hi > member.rlim[params]:
                raise ValueError(
                    "restrict_member_range: requested [{}, {}] for {!r} on member {} is NOT contained "
                    "in that member's range [{}, {}].  This API can only narrow: the prior callables "
                    "are absolute densities normalized over the original range, so sampling outside "
                    "it would bias the integral.".format(lo, hi, params, indx,
                                                         member.llim[params], member.rlim[params]))
            member.llim[params] = lo
            member.rlim[params] = hi
            getattr(self, '_pending_range_overrides', set()).discard((indx, params))
            print("  [portfolio] member {} range for {} narrowed to [{}, {}] (proposal only; "
                  "prior callables untouched)".format(indx, params, lo, hi))


    @staticmethod
    def _snapshot_setup_args(args):
        """Copy the mutable containers in a setup-argument dict; pass everything else by reference.

        REQUIRED for correctness, not tidiness.  Setup arguments are not inert: production supplies
        `gmm_dict` as a grouping spec ({(0,1,2): None, ...}), mcsamplerEnsemble hands that very
        object to monte_carlo.integrator, which stores it WITHOUT copying (MonteCarloEnsemble.py:110)
        and then writes trained models into it (`self.gmm_dict[dim_group] = model`, :403).  Keeping a
        reference and replaying it would hand the next point the PREVIOUS point's trained proposal --
        reintroducing, through the reset itself, exactly the state leak the reset exists to remove.

        Objects NESTED INSIDE a spec container are cloned too, not just the container.  A seeded
        GMM model supplied via `--extrinsic-proposal-breadcrumb` lives as a VALUE in gmm_dict, and
        with `--extrinsic-proposal-adapt` it keeps adapting: `model.update()` mutates it in place
        (gaussian_mixture_model.py:548).  Copying only the dict would leave the stored "baseline"
        pointing at the live model, so it would drift during point 1 and be replayed into point 2 --
        the same leak one level down.  (With adapt OFF, the default, `_train` skips seeded groups
        and nothing mutates, so this path was previously harmless.)

        TOP-LEVEL non-container arguments are still passed by reference: those are callables,
        modules and sampler objects, which a deepcopy would try to clone and can fail on or spend
        real time duplicating.  If a nested clone fails, the original is kept and the failure is
        REPORTED -- a silently shared object is how this class of bug survives.
        """
        _unclonable = []

        def _cp(v, depth=0):
            if depth > 6:      # runaway guard; setup specs are shallow
                return v
            if isinstance(v, dict):
                return dict((k, _cp(x, depth + 1)) for k, x in v.items())
            if isinstance(v, list):
                return [_cp(x, depth + 1) for x in v]
            if isinstance(v, tuple):
                return tuple(_cp(x, depth + 1) for x in v)
            if isinstance(v, set):
                return set(v)
            if isinstance(v, np.ndarray):
                return v.copy()
            if depth == 0 or v is None or isinstance(v, (bool, int, float, complex, str, bytes)):
                return v
            if isinstance(v, (ModuleType, FunctionType, BuiltinFunctionType, type)):
                return v      # stateless: sharing these is safe and cloning them is not
            # A nested object with mutable state -- e.g. a seeded GMM `estimator`.  deepcopy is
            # NOT usable: a real estimator holds a module reference (`xpy`) and deepcopy raises
            # "cannot pickle 'module' object", which would send us down the fallback and leave the
            # model SHARED -- i.e. not fixed at all.  Shallow-copy the object (which never
            # pickles) and then clone its mutable attributes, leaving module/function refs shared.
            try:
                new_obj = shallow_copy(v)
                d = getattr(new_obj, '__dict__', None)
                if d is None:
                    raise TypeError("no __dict__ (__slots__?), cannot clone attribute state")
                for k, val in list(d.items()):
                    d[k] = _cp(val, depth + 1)
                return new_obj
            except Exception as e:
                _unclonable.append("{} ({})".format(type(v).__name__, e))
                return v

        out = dict((k, _cp(v)) for k, v in args.items())
        if _unclonable:
            print("  [portfolio] WARNING: could not clone {} nested setup object(s): {}.  These are "
                  "SHARED with the live sampler, so if anything mutates them in place their state "
                  "will persist across a reset.".format(len(_unclonable), "; ".join(_unclonable)))
        return out

    def clear_warm_state(self):
        """Clear any warm-start seed AND the installed active grid on every member.

        Setting `portfolio._warm = None` does NOT do this: `_warm` and the contracted AV grid live on
        the MEMBERS, not on the portfolio object.  Now that a seed is actually installed on the draw
        path (AV._apply_warm_state), failing to clear it between points would let the next point reuse
        the PREVIOUS point's contracted live volume -- which can exclude the new point's support and
        bias it low with no diagnostic.  Called by the driver wherever it used to do
        `sampler._warm = None`, including the seed-capture failure and exception paths.

        Failures PROPAGATE.  A reset that quietly did not happen leaves the next point drawing from
        the previous point's grid, which is the exact silent-wrong-answer this method exists to
        prevent -- so it must not be reducible to a log line.
        """
        self._warm = None
        _groups = [(list(getattr(self, 'portfolio_realizations', [])),
                    getattr(self, '_member_setup_args', None)),
                   (list(getattr(self, 'oracle_realizations', [])),
                    getattr(self, '_oracle_setup_args', None))]
        for members, saved_args in _groups:
            for indx, member in enumerate(members):
                member._warm = None
                member._warm_applied = False
                if not hasattr(member, 'setup'):
                    continue
                # REPLAY the member's original setup arguments.  A bare setup() restores the cold
                # grid but DISCARDS the configuration: mcsamplerEnsemble.setup() rebuilds its
                # dimension grouping and re-reads n_comp / gmm_adapt / correlate_all_dims from
                # kwargs, so a configured (0,1) GMM with n_comp=3 and adaptation off comes back as
                # separate (0,), (1,) groups with n_comp defaulted and gmm_adapt=None -- a quietly
                # different sampler for every point after the first.
                args_here = None
                if saved_args is not None and indx < len(saved_args):
                    args_here = saved_args[indx]
                if args_here is None:
                    # setup() was never run through the portfolio: nothing to replay, nothing to
                    # lose.  AV.setup() ignores kwargs and rebuilds from the member's own llim/rlim
                    # (so a narrowed member stays narrowed across the reset).
                    member.setup()
                else:
                    # a FRESH copy per replay: passing the stored dict itself would let the
                    # rebuilt integrator train into our snapshot, so the reset after next would
                    # replay a polluted spec and the leak would return one point later.
                    member.setup(**self._snapshot_setup_args(args_here))

    def reset_adaptation(self):
        """FULL reset: member proposals AND the portfolio's own adaptive bookkeeping.

        clear_warm_state() rebuilds the MEMBERS, but the portfolio itself also learns during a run
        -- draw allocation, per-member quality EMAs and their observation counts, the round-robin
        probe pointer, the iteration counter, breakpoint progression and the per-member n_ess
        histories.  MC-error replicas that inherit those start with scheduling learned from the
        earlier replicas, so they are not adaptation-independent and the between-replica scatter
        still understates the true error -- which is the entire quantity the replicas exist to
        measure.  Restores every field to its post-setup value.
        """
        self.clear_warm_state()
        n = len(self.portfolio)
        # draw allocation: back to uniform (or the caller's explicit initial weights)
        w0 = getattr(self, '_portfolio_weights_initial', None)
        self.portfolio_weights = np.array(w0) if w0 is not None else np.ones(n) / (1.0 * n)
        self.portfolio_quality = np.ones(n)
        self.portfolio_quality_nobs = np.zeros(n, dtype=int)
        self.portfolio_probe_ptr = 0
        self.portfolio_draw_iteration = 0
        self.portfolio_member_ness_history = [[] for _ in range(n)]
        # breakpoints are a SCHEDULE (set at setup), not learned state: restore the schedule that
        # setup() installed rather than zeroing it, or a replica would activate members on a
        # different iteration than the first run did.
        bp0 = getattr(self, '_portfolio_breakpoints_initial', None)
        if bp0 is not None:
            self.portfolio_breakpoints = np.array(bp0)
        for _attr in ('portfolio_frozen', 'portfolio_grace_left', 'portfolio_last_revive'):
            _v0 = getattr(self, '_' + _attr + '_initial', None)
            if _v0 is not None:
                setattr(self, _attr, np.array(_v0) if hasattr(_v0, '__len__') else _v0)

    def bootstrap_from_samples(self, samples, params=None, keep_backstop_cold=None, **kwargs):
        """Warm-start: forward a seed cloud to every member that supports it (e.g. the
        AV/VARAHA member's live volume), EXCEPT the full-support backstop (see below).
        Members without bootstrap_from_samples are left cold.  A warm start only shapes a
        member's proposal, and the portfolio combines members with the balance-heuristic
        mixture density (q_mix), so a mis-seeded member costs efficiency rather than bias --
        BUT ONLY WHILE SOME MEMBER STILL COVERS THE SUPPORT.  Column
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

        THE BACKSTOP MUST STAY COLD.  Seeding EVERY member destroys the coverage invariant the
        whole design rests on.  `cover_frac` does not save it: it mixes a FINITE number of uniform
        points into the seed, and a finite point set occupies only the bins it lands in, so the
        seeded live volume is NOT a superset of a cold start.  Measured on a tight seed in the
        [-5,5]^d box -- fraction of the prior box covered, against 1.0 cold:

            d=2:  cover_frac 0.0 / 0.2 / 0.5 / 0.9  ->  0.027 / 0.634 / 0.982 / 1.000
            d=4:                                    ->  0.0015 / 0.028 / 0.104 / 0.620
            d=6:                                    ->  6.3e-05 / 0.00087 / 0.0033 / 0.0287

        so at d=6 even cover_frac=0.9 leaves 97% of the box unsampled.  Before this change a
        portfolio warm start narrowed member 0 to V=0.0033 along with everyone else.  Keeping
        member 0 (the designated backstop, which restrict_member_range also refuses to narrow)
        cold makes the mixture-level guarantee TRUE instead of merely asserted.

        BE PRECISE ABOUT WHAT THIS BUYS, because the measurements are not what one expects:

          * With a CORRECT seed it is nearly free -- n_eff 3630/3466/5695 cold-backstop vs
            3876/3461/5621 seeded at d=4, and 5078/5689/5163 vs 5857/5387/5266 at d=6.  Within
            run-to-run scatter, so it is cheap insurance.  That is the case for the default.
          * It is NOT what protects the DEFAULT AV+GMM portfolio.  The GMM member is a Gaussian
            mixture with nonzero density over the whole box, so q_mix never vanishes there
            regardless of this setting.  With a deliberately displaced seed, |lnZ bias| stayed
            <= 0.05 in every d=4 and d=6 run in BOTH arms.  That unbounded support is the real
            (previously undocumented) reason production has not been biased by warm starts.
          * It does NOT rescue a badly mismatched seed.  In an ALL-AV portfolio (every component
            a hard-edged box) with a displaced seed at d=6, the cold backstop still gave lnZ bias
            -1.1 to -4.2 nats with n_eff 3-9, versus -1.0 to -6.8 seeded.  A uniform member finds
            a sharp 6-D peak too rarely to carry the integral within budget.  Coverage in
            principle is necessary, not sufficient -- for a mismatched seed the mitigations are
            detection (L0 rescue) and not warm-starting across dissimilar points.

        `keep_backstop_cold`: None (default) uses self.portfolio_warmstart_backstop_cold, itself
        True by default; pass False to restore the old seed-everything behaviour.

        Returns the number of members warm-started (0 is fine; the portfolio still runs)."""
        samples = np.asarray(samples)
        if keep_backstop_cold is None:
            keep_backstop_cold = bool(getattr(self, 'portfolio_warmstart_backstop_cold', True))
        # Which members must retain full support?  If range restriction is in play it already
        # computed them; otherwise it is member 0 by the same convention.
        # The invariant is NOT "member 0 must be cold" -- it is "SOME member must have support
        # everywhere".  A GMM/ensemble member satisfies that inherently: it carries an explicit
        # uniform defensive component (gmm_defensive_frac, default 0.05) plus Gaussian tails, so
        # q_mix never vanishes however it is seeded.  Measured: in the default [AV, GMM] portfolio
        # a deliberately displaced seed left |lnZ bias| <= 0.05 whether or not a member was held
        # cold, while an ALL-AV portfolio gave -1.0 to -6.8 nats.
        # So hold a member cold ONLY when EVERY member has compact support.  Doing it
        # unconditionally disables the AV warm start in [AV, GMM] -- member 0 IS the AV member --
        # to buy a guarantee the GMM member already provides.  The merge gate caught exactly that.
        # Default FALSE: a sampler must DECLARE full support to be counted.  Defaulting to True
        # meant any member that simply had not been annotated was treated as the coverage
        # guarantee -- the safe default is to assume compact and keep a cold backstop.
        # And a nominally broad member does NOT count if it has been RANGE-RESTRICTED: its
        # proposal is confined to a sub-box, so it no longer covers the prior.  Without this,
        # [unrestricted AV, restricted GMM] reported _full_support_members == [0] and then
        # warm-started and contracted member 0 anyway, leaving nothing covering the prior box --
        # the silent low bias this whole mechanism exists to prevent.
        if getattr(self, '_has_restricted_member', False):
            _unrestricted = set(getattr(self, '_full_support_members', []) or [])
        else:
            _unrestricted = set(range(len(self.portfolio_realizations)))
        _has_broad = any(getattr(m, 'has_unbounded_support', False) and (i in _unrestricted)
                         for i, m in enumerate(self.portfolio_realizations))
        if keep_backstop_cold and _has_broad:
            keep_backstop_cold = False
            print("  [portfolio] warm-starting all members: a full-support member is present "
                  "(defensive mixture), so no cold backstop is needed")
        _backstop = set(getattr(self, '_full_support_members', None) or [0]) if keep_backstop_cold else set()
        if len(_backstop) >= len(self.portfolio_realizations):
            _backstop = set([0])   # never refuse to warm-start EVERY member
        self._warmstart_backstop_cold = sorted(_backstop)
        n_warmed = 0
        for indx, member in enumerate(self.portfolio_realizations):
            if indx in _backstop:
                print("  [portfolio] member {} kept COLD as the full-support backstop "
                      "(cover_frac cannot make a seeded grid cover the prior box; see "
                      "bootstrap_from_samples docstring)".format(indx))
                continue
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
        _kw_keep('portfolio_plateau_revive')
        _kw_keep('portfolio_adaptive_alloc')
        _kw_keep('portfolio_quality_signal')
        _kw_keep('portfolio_alloc_exponent')
        _kw_keep('portfolio_alloc_floor')
        _kw_keep('portfolio_quality_decay')
        _kw_keep('portfolio_probe_period')
        _kw_keep('portfolio_probe_frac')
        _kw_keep('portfolio_weight_clip')
        _kw_keep('portfolio_varaha_min_frac')
        _kw_keep('portfolio_varaha_max_frac')
        _kw_keep('portfolio_warmstart_backstop_cold')
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
        # RANGE RESTRICTION (opt-in): narrow ONE OR MORE members to a sub-box, so a member can put
        # its fixed bin budget where the posterior actually is.  Applied HERE -- after add_parameter
        # (which forwards identical limits AND the shared prior callables to every member) and
        # BEFORE member.setup() (which rebuilds my_ranges/dx/dx0/V_s/binunique/ninbin/V from
        # llim/rlim) -- so no derived state can go stale.
        #
        # WHY THIS NEEDS NO NORMALIZATION CORRECTION.  The estimator weights are
        # lnL + log(joint_p_prior) - log(q_mix).  `joint_p_prior` comes from each member's STORED
        # prior callables (AV.prior_prod), which are absolute densities over the ORIGINAL physical
        # ranges and never consult llim/rlim -- so a narrowed member still reports the TRUE global
        # prior.  A member's range is therefore purely a PROPOSAL choice: proposals need not cover
        # the prior, only the MIXTURE must cover the support of L*p.  Hence no prior renormalization
        # and no ln(V_sub/V_full) term.  The invariant that makes this true is enforced below.
        _restrict = kwargs.get('portfolio_restrict_ranges', None)
        if _restrict:
            if len(_restrict) != len(self.portfolio_realizations):
                raise Exception("portfolio_restrict_ranges must align with the member list "
                                "({} entries for {} members)".format(len(_restrict),
                                                                     len(self.portfolio_realizations)))
            _n_restricted = 0
            for indx, member in enumerate(self.portfolio_realizations):
                spec = _restrict[indx]
                if not spec:
                    continue
                for p, (lo, hi) in dict(spec).items():
                    if p not in member.llim:
                        raise Exception("portfolio_restrict_ranges: member {} has no parameter {!r}".format(indx, p))
                    # NARROW ONLY -- clip into the existing range.  Widening a member beyond the
                    # prior's support would sample where the prior callable is not normalized.
                    lo_new = max(float(lo), float(member.llim[p]))
                    hi_new = min(float(hi), float(member.rlim[p]))
                    if not (hi_new > lo_new):
                        raise Exception("portfolio_restrict_ranges: empty sub-range for {!r} on member {}".format(p, indx))
                    member.llim[p] = lo_new
                    member.rlim[p] = hi_new
                _n_restricted += 1
                print("  PORTFOLIO: member {} RESTRICTED to sub-box {}".format(indx, dict(spec)))
            # COVERAGE INVARIANT: at least one member must keep FULL support, otherwise the mixture
            # no longer covers L*p outside the union of sub-boxes and the integral is biased low by
            # the missing mass (silently -- n_eff can even look BETTER).  Refuse rather than bias.
            # UNION with any restrictions registered through the public restrict_member_range() API.
            # Both entry points must feed the SAME bookkeeping: computing _full_support_members from
            # _restrict alone would declare an API-restricted member "full support" and hand it the
            # draw floor that is meant to protect a genuinely unrestricted component.
            _restricted_set = set(i for i in range(len(self.portfolio_realizations)) if _restrict[i])
            _restricted_set |= set(getattr(self, 'member_range_overrides', {}))
            if len(_restricted_set) >= len(self.portfolio_realizations):
                raise Exception(
                    "portfolio_restrict_ranges: every member is restricted, so no member retains "
                    "full support.  The mixture would not cover L*p outside the sub-boxes and the "
                    "integral would be biased low with no diagnostic.  Leave at least one member "
                    "unrestricted (it is the defensive component).")
            self._has_restricted_member = bool(_restricted_set)
            # index of a full-support member: the per-member draw floor below protects it
            self._full_support_members = [i for i in range(len(self.portfolio_realizations))
                                          if i not in _restricted_set]

        # Snapshot the post-setup values of everything reset_adaptation() restores, so a replica
        # returns to THIS state rather than to a hard-coded guess.
        self._portfolio_weights_initial = np.array(self.portfolio_weights)
        self._portfolio_breakpoints_initial = np.array(self.portfolio_breakpoints)
        for _attr in ('portfolio_frozen', 'portfolio_grace_left', 'portfolio_last_revive'):
            if hasattr(self, _attr):
                _v = getattr(self, _attr)
                setattr(self, '_' + _attr + '_initial',
                        np.array(_v) if hasattr(_v, '__len__') else _v)

        # CONSUMED CHECK.  restrict_member_range() only takes effect if it was called BEFORE
        # add_parameter forwarded that parameter to the members.  A restriction naming a parameter
        # that never arrives (typo, or the call came too late) used to be a SILENT no-op: the caller
        # believes a member is focused on the posterior while it still samples the full box.  Fail
        # loudly instead -- a narrowing that quietly did nothing is a wasted member, not a safe one.
        _pending = getattr(self, '_pending_range_overrides', set())
        if _pending:
            raise Exception(
                "restrict_member_range: {} restriction(s) were never applied: {}.  Either the "
                "parameter name does not exist on that member, or restrict_member_range() was "
                "called AFTER add_parameter() -- it must be called before.".format(
                    len(_pending), sorted(_pending)))

        # Iterate the INSTANTIATED samplers (portfolio_realizations), NOT self.portfolio: the
        # latter may hold modules/names (see __init__), which lack .setup(), so member setup was
        # silently skipped -> a cold member's internal state (AV my_ranges, GMM integrator) was
        # never built and draw_simplified failed.  Setting up the realizations fixes AV+GMM cold.
        # REMEMBER each member's setup arguments.  clear_warm_state() has to re-run setup() to
        # restore a member's cold grid, and calling it bare would silently DISCARD the member's
        # configuration: mcsamplerEnsemble.setup() rebuilds its dimension grouping and re-reads
        # n_comp / gmm_adapt / correlate_all_dims etc from kwargs, so a configured (0,1) GMM with
        # n_comp=3 and adaptation off comes back as separate (0,), (1,) groups with n_comp
        # defaulted and gmm_adapt=None.  Store the exact args and replay them.
        self._member_setup_args = [None] * len(self.portfolio_realizations)
        self._oracle_setup_args = [None] * len(self.oracle_realizations)
        for indx, member in enumerate(self.portfolio_realizations):
            if hasattr(member, 'setup'):
              print(" PORTFOLIO setup ", member, portfolio_extra_args[indx])
              args_here = {}
              args_here.update(kwargs)
              args_here.update(portfolio_extra_args[indx])
              # snapshot BEFORE setup: the member (or its integrator) may mutate these in place
              # A portfolio member may be the mixture's ONLY full-support component, so ask for
              # the defensive component on every fit path for OUR members.  Standalone users of
              # the same sampler are unaffected -- measured, it costs real n_eff at d>=6.
              args_here.setdefault('gmm_defensive_all_paths', True)
              self._member_setup_args[indx] = self._snapshot_setup_args(args_here)
              member.setup(**args_here)
        for indx, member in enumerate(self.oracle_realizations):
            if hasattr(member, 'setup'):
              print(" PORTFOLIO ORACLE setup ", member, portfolio_extra_args[indx])
              args_here = {}
              args_here.update(kwargs)
              args_here.update(portfolio_extra_args[indx])
              self._oracle_setup_args[indx] = self._snapshot_setup_args(args_here)
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
        # 'global' (marginal pooled n_eff) and 'credit' (MIS credit) are both ZERO-based
        # contribution measures; only the legacy Kish n_ess signal is floored at 1.
        _global = self.portfolio_quality_signal in ('global', 'credit')
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
           # Single-member fast path: q_mix degenerates to this member's own density.  If that lone
           # member is a RESTRICTED one, nothing covers L*p outside its sub-box for this chunk and
           # the estimate is biased low with no diagnostic.  (Reachable when activation breakpoints
           # delay the full-support member.)  Refuse rather than silently bias.
           if (getattr(self, '_has_restricted_member', False)
                   and int(indx_active[0]) not in set(getattr(self, '_full_support_members', []))):
               raise Exception(
                   "mcsamplerPortfolio: the only ACTIVE member (realization {}) has a RESTRICTED "
                   "range, so this chunk has no full-support component and the integral would be "
                   "biased low.  Give the full-support member an activation breakpoint of 0."
                   .format(int(indx_active[0])))
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

          # PER-MEMBER DRAW FLOOR for full-support members when some member is RESTRICTED.
          # A member that draws 0 samples this chunk contributes NOTHING to q_mix (the mixture loop
          # skips frac_m <= 0), so its coverage vanishes for that chunk.  That is harmless when all
          # members share a support, but FATAL once a member has been narrowed: the full-support
          # member is the only thing covering L*p outside the sub-box, and a chunk where it is
          # rounded to zero draws is a chunk with an uncovered region -- the silent low-bias failure
          # this whole design exists to avoid.  The VARAHA share band is a GROUP constraint over all
          # is_varaha members, so it does NOT protect an individual full-box AV against a restricted
          # sibling absorbing the group's share.  Enforce a per-member minimum of 1 draw here.
          if getattr(self, '_has_restricted_member', False):
            # CAREFUL: n_samples_per_member is indexed by POSITION WITHIN portfolio_active (the
            # breakpoint-filtered subset), while _full_support_members holds REALIZATION indices.
            # Map one to the other; indexing directly by realization index protects the wrong member
            # as soon as any member is still behind its activation breakpoint.
            _full_set = set(getattr(self, '_full_support_members', []))
            _full = [pos for pos, ridx in enumerate(indx_active) if int(ridx) in _full_set]
            for i in _full:
              if n_samples_per_member[i] < 1:
                # take the deficit from the largest member so the total is preserved exactly
                j = int(np.argmax(n_samples_per_member))
                if j != i and n_samples_per_member[j] > 1:
                  n_samples_per_member[j] -= 1
                  n_samples_per_member[i] = 1

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
                    _mix_parts = {}
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
                        _contrib_m = float(frac_m) * numpy.asarray(identity_convert(q_m), dtype=float)
                        acc = acc + _contrib_m
                        # retain frac_m*q_m per member: the balance-heuristic credit
                        # frac_m q_m / q_mix is the MIS share of each sample owed to member m
                        _mix_parts[id(member_m)] = _contrib_m
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
                        self._chunk_mix_parts = _mix_parts
            if q_mix is not None:
                joint_p_s = q_mix   # deterministic-mixture denominator
            else:
                # The legacy stratified per-member density is only valid when every member shares the
                # SAME support.  If any member has been given a RESTRICTED range (see
                # portfolio_restricted_members) the stratified estimator is silently WRONG -- it does
                # not form the true mixture denominator -- so refuse rather than return a biased
                # number.  Unequal supports are exactly the configuration the fallback cannot handle.
                if getattr(self, '_has_restricted_member', False):
                    raise Exception(
                        "mcsamplerPortfolio: a member has a RESTRICTED sampling range, but the "
                        "balance-heuristic q_mix could not be formed (some active member lacks "
                        "sampling_density).  The legacy stratified density is invalid for members "
                        "with unequal support and would bias the integral; refusing to continue.")
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
            # MIS CREDIT ASSIGNMENT (q_mix-native, the 'credit' quality signal).  Under the
            # balance heuristic each sample's contribution is owed to members in proportion to
            # their share of the mixture density there, so member m's credit is
            #     credit_m = sum_i [ frac_m q_m(x_i) / q_mix(x_i) ] * w_i .
            # Unlike Kish n_ess (scale-invariant, hence blind to whether a member carries any
            # integral mass) this credits a member for COVERING WHERE THE INTEGRAND IS, even if
            # it drew few samples there -- exactly the signal a slow-contracting VARAHA member
            # needs.  Normalized per drawn sample so members are comparable at unequal shares.
            credit_per_sample = numpy.zeros(len(self.portfolio))
            _parts = getattr(self, '_chunk_mix_parts', None)
            if _parts and q_mix is not None:
                _qm = numpy.asarray(self.identity_convert(q_mix), dtype=float)
                _w_all = numpy.where(numpy.isfinite(_lw_all), numpy.exp(_lw_all - _lw_max), 0.0)
                for _im, _mem in enumerate(self.portfolio_realizations):
                    _pc = _parts.get(id(_mem))
                    if _pc is None:
                        continue
                    _share = numpy.where(_qm > 0, _pc / _qm, 0.0)
                    # DIVIDE OUT THE MEMBER'S OWN ALLOCATION.  The raw share frac_m*q_m/q_mix scales
                    # with frac_m, so a member accrues credit simply BECAUSE it is dominant -- the
                    # same circularity the n_ess signal has (measured: a 0.95-share GMM scored 6e-4
                    # vs AV's 9e-10 and starved AV to the floor).  Normalizing by frac_m turns this
                    # into "integral explained PER UNIT ALLOCATION", which is allocation-invariant
                    # and is what an allocation rule must compare.
                    _frac_m = float(n_samples_per_member[_im]) / float(max(1, n_samples))
                    if _frac_m > 0:
                        credit_per_sample[_im] = float(numpy.sum(_share * _w_all)) / (_frac_m * max(1, n_samples))

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
              print("\t credit/sample (MIS credit):", numpy.array2string(credit_per_sample, precision=3))
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
              if self.portfolio_quality_signal == 'credit':
                _obs = credit_per_sample
              elif self.portfolio_quality_signal == 'global':
                _obs = contrib_per_sample
              else:
                _obs = dat
              self.portfolio_weights = self._adaptive_allocation(_obs, frac_now, self.portfolio_draw_iteration)
            else:
              self.portfolio_weights = portfolio_wt_func(dat, self.portfolio_weights, xpy=self.xpy, identity_convert=self.identity_convert) # call weighting function
            # VARAHA DRAW FLOOR (see __init__): reserve a combined fraction for VARAHA members, so a
            # slow-contracting workhorse cannot be starved of DRAWS by a member that merely looks
            # good per-chunk.  Applied after either allocation rule; unbiased (q_mix).
            # BANDED: a floor alone is not enough.  Measured on a loud-event best-fit point: with no
            # floor the mixture degenerates to GMM-alone (VARAHA share -> 0.0099), q_mix loses its
            # broad backstop, and a mode the peaked member misses is uncovered -> lnZ silently low
            # while n_eff looks GOOD (the confidently-wrong failure).  With a floor but no cap, one
            # seed ran away the OTHER way (VARAHA -> 0.99) and was the outlier of its arm.  Both are
            # mixture degeneration.  Constraining the VARAHA share to a BAND keeps q_mix genuinely
            # mixed -- a broad backstop AND a peaked component -- by construction.  Unbiased either
            # way (q_mix balance heuristic), so this costs at most draws, never correctness.
            _vmin = float(self.portfolio_varaha_min_frac)
            _vmax = float(self.portfolio_varaha_max_frac)   # <=0 or >=1 => no cap (back-compatible)
            _cap_on = (0.0 < _vmax < 1.0)
            if (_vmin > 0 or _cap_on) and len(self.portfolio) > 1:
              _is_v = np.array([hasattr(m, 'is_varaha') for m in self.portfolio_realizations])
              if _is_v.any() and not _is_v.all():
                _w = np.asarray(self.portfolio_weights, dtype=float)
                _w = np.where(np.isfinite(_w) & (_w > 0), _w, 0.0)
                _sv = _w[_is_v].sum(); _so = _w[~_is_v].sum()
                _target = None
                if _vmin > 0 and _sv < _vmin:
                  _target = _vmin
                elif _cap_on and _sv > _vmax:
                  _target = _vmax
                if _target is not None and (_sv > 0 or _so > 0):
                  # put the VARAHA group at _target and the rest at (1-_target), each preserving its
                  # own internal split; if a group is all-zero, spread its share evenly within it.
                  if _sv > 0:
                    _w[_is_v] *= _target / _sv
                  else:
                    _w[_is_v] = _target / max(int(_is_v.sum()), 1)
                  if _so > 0:
                    _w[~_is_v] *= (1.0 - _target) / _so
                  else:
                    _w[~_is_v] = (1.0 - _target) / max(int((~_is_v).sum()), 1)
                  _tot = _w.sum()
                  if _tot > 0:
                    self.portfolio_weights = _w / _tot

              
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
                if self.portfolio_plateau_revive and len(_hist) >= 3:
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
            ln_wt = numpy.asarray(ln_wt, dtype=float)
            # Guard: rejected/underflowed samples (-inf) or bad priors (nan) must not poison the
            # cumulative sum below.  Map any non-finite log-weight to -inf so it exponentiates to a
            # zero linear weight instead of corrupting cumsum/normalization (which would drop ALL rows).
            ln_wt[~numpy.isfinite(ln_wt)] = -numpy.inf
            ln_wt_max = numpy.max(ln_wt)
            if numpy.isfinite(ln_wt_max):
                ln_wt = ln_wt - ln_wt_max  # remove maximum value, irrelevant to the normalized cumulative prob
                wt = numpy.exp(ln_wt)      # exponentiate to LINEAR weights (max-subtracted).  Underflow -> 0, which is fine
            else:
                # degenerate: no finite-weight sample survived Step 1 -- keep everything rather than drop all rows
                wt = numpy.ones(len(ln_wt))
            idx_sorted_index = numpy.lexsort((numpy.arange(len(wt)), wt))  # Sort the array of weights, recovering index values
            # Pair the sorted index with the LINEAR weight wt[k] (NOT the log-weight ln_wt[k]): the cumulative
            # sum below must be a cumulative PROBABILITY, matching mcsampler/mcsamplerEnsemble.  Cumsumming the
            # log-weights (<=0, and -inf for rejects) is not a probability threshold and kept 0 rows for peaked runs.
            indx_list = numpy.array( [[k, wt[k]] for k in idx_sorted_index])     # pair up with the LINEAR weights again. NOTE NOT INTEGER TYPE ANY MORE
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
