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
from RIFT.precision import RiftFloat  # platform-portable replacement for np.float128
from scipy import integrate, interpolate, special
import itertools
import functools
import inspect


@functools.lru_cache(maxsize=None)
def _prior_pdf_accepts_xpy(fn):
    """True if a prior_pdf callable takes an `xpy` kwarg.  Many of the mcsamplerGPU prior
    helpers default xpy=cupy, so evaluating them on the host CPU copy (as prior_prod does)
    would feed a numpy array to cupy and raise; we pass xpy=numpy to those that accept it."""
    try:
        return 'xpy' in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False

import os

try:
  import cupy
  import cupyx.scipy.special   # needed for logsumexp
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

from RIFT.integrators.statutils import  update,finalize, init_log,update_log,finalize_log, pareto_khat_from_log, ess_from_log_weights, bootstrap_lnZ_quantiles

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
        # Draw uniformly within each occupied hypercube bin.  VECTORIZED: the old
        # implementation looped over bins in Python (a list comprehension + vstack
        # over one entry per bin), which is O(n_bins) per chunk and becomes the
        # bottleneck once the live volume is finely resolved -- e.g. a concentrated
        # warm start from a full PE posterior can seed thousands of bins.  Here we
        # instead repeat each bin's lower corner ninbin[k] times and add a single
        # (N, ndim) uniform draw, so cost is O(N) with no Python-level bin loop.
        ndim = xrange.shape[0]
        # per-bin lower corners + the point->bin expansion are done on the HOST
        # (np.repeat with an int-array of counts is reliable everywhere; cupy.repeat
        # with array repeats is version-fragile), then the single uniform draw is on
        # the active backend so the output matches the previous cupy behaviour.
        bu_h = identity_convert(bu); dx_h = np.asarray(identity_convert(dx))
        xlo_h = np.asarray(identity_convert(xrange)).T[0] + dx_h * np.asarray(bu_h)  # (n_bins, ndim)
        reps = np.asarray(identity_convert(ninbin)).astype(int)
        lo_per_point = np.repeat(xlo_h, reps, axis=0)      # host (N, ndim)
        N = lo_per_point.shape[0]
        x = xpy_default.asarray(lo_per_point) + xpy_default.asarray(dx_h) * xpy_default.random.uniform(0.0, 1.0, size=(N, ndim))
        # remove points that are out of range.  Due to rounding issues etc, the sampler above can generate points out of range!
        # Note this rejection will bias the integral, because volumes are calculated assuming a regular grid. We *should* fix the grid sizes to integers
        if reject_out_of_range:
          for indx in np.arange(len(xrange)):
            indx_ok = xpy_default.where(xpy_default.logical_and(x[:,indx] >= xrange[indx,0], x[:,indx] <= xrange[indx,1], ))
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
        self._warm=None  # bootstrap/warm-start live-volume state (see bootstrap_from_*)
        self._warm_applied=False  # has _warm been installed into the ACTIVE grid? (see _apply_warm_state)
        # Opt-in ANISOTROPIC bin allocation: give each axis a different number of bins
        # (fine where the live points cluster tightly -- phase/pol/sky; coarse where they are
        # broad -- distance/inclination), instead of the default equal split.  Keeps the same
        # total bin budget (prod(nbins)=1/delta_V) so the estimator is unchanged.  Default off.
        self.anisotropic_bins = False


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

        prior_pdf are host (numpy) functions in general, so evaluate them on a
        CPU copy of the samples and convert the product back to the active
        backend. identity_convert / identity_convert_togpu are no-ops when cupy
        is not in use.
        """
        p_out = xpy_default.ones(len(x))
        x_cpu = identity_convert(x)
        indx = 0
        for param in self.params_ordered:
            fn = self.prior_pdf[param]
            xc = x_cpu[:, indx]
            # Force host evaluation: several mcsamplerGPU prior helpers default xpy=cupy, which
            # would raise on the numpy host copy when cupy is importable (e.g. GPU container runs).
            val = fn(xc, xpy=numpy) if _prior_pdf_accepts_xpy(fn) else fn(xc)
            p_out *= identity_convert_togpu(val)
            indx += 1
        return p_out


    def _apply_warm_state(self):
        """Install a seeded live-volume grid (self._warm) into the ACTIVE sampling state.

        `bootstrap_from_samples` / `bootstrap_from_oracle` / `load_state` only STORE the seeded grid
        in `self._warm`; historically it was installed only inside `integrate_log`.  Anything that
        drives this sampler WITHOUT calling its own integrate_log -- above all a PORTFOLIO, which
        calls draw_simplified()/update_sampling_prior() directly, and the driver's L0 auto-rescue,
        which re-seeds a portfolio then re-runs -- therefore kept drawing from the COLD grid while
        reporting that it had been warm-started.  Idempotent; safe to call on every draw.
        """
        warm = getattr(self, '_warm', None)
        if warm is None or getattr(self, '_warm_applied', False):
            return
        try:
            self.binunique = np.array(warm['binunique'])
            self.dx = np.array(warm['dx'])
            self.nbins = np.array(warm['nbins'])
            self.ninbin = ((self.n_chunk // self.binunique.shape[0] + 1)
                           * np.ones(self.binunique.shape[0])).astype(int)
            if 'V' in warm:
                self.V = float(warm['V'])
            if 'loglkl_thr' in warm:
                self.lnL_thresh = float(warm['loglkl_thr'])
            self._warm_applied = True
            print("  [AV warm-start] seeded grid APPLIED to the active draw path: "
                  "live bins={}".format(self.binunique.shape[0]))
        except Exception as e:
            # never let a malformed seed break sampling: fall back to the cold grid
            print("  [AV warm-start] could not apply seeded grid ({}); continuing cold".format(e))
            self._warm_applied = True

    def draw_simplified(self,n_to_get, *args, **kwargs):
        # Self-contained cold start.  A PORTFOLIO (mcsamplerPortfolio) drives draw_simplified on
        # its members directly, WITHOUT running each member's own integrate()/setup(), so a cold
        # AV member may not have its live-volume grid (my_ranges/dx/binunique/ninbin) built yet
        # -> AttributeError on self.my_ranges.
        # Build the cold full-box grid on first use so AV works as a portfolio member cold or warm.
        if getattr(self, 'my_ranges', None) is None:
            self.setup()
        # ... and if a seed was supplied, INSTALL it: setup() above (and the driver, which calls
        # setup BEFORE bootstrap_from_samples) leaves the active grid cold, so without this a
        # warm-started portfolio member draws from the cold grid.
        self._apply_warm_state()
        rv, log_p = self.draw_simple()
        # Subsample RANDOMLY, never a head slice: sample_from_bins emits points
        # grouped in lexicographic bin order (binunique from np.unique), so
        # rv[:n_to_get] returns only the first ~n_to_get/ninbin bins of the live
        # volume while sampling_density (hence the portfolio's q_mix) claims
        # uniform coverage of ALL occupied bins.  In any multi-member portfolio
        # that mismatch systematically biased the recovered shape (pulls up to
        # 0.6 sigma at d2; random subsample collapses them to ~1e-3).
        n_have = len(rv)
        if n_to_get < n_have:
            keep = np.random.choice(n_have, size=int(n_to_get), replace=False)
            rv = rv[keep]
            log_p = log_p[keep]
        p = np.exp(log_p)
        ps = self.xpy.ones(len(p))*self.V_s/self.V   # sampling prior, full hypercube normalized to 1
        rv = rv.T
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

    def sampling_density(self, X):
        """Pointwise sampling density q(theta) of THIS member, evaluated at
        ARBITRARY points X (shape (N, ndim), columns in self.params_ordered
        order).  Returns a host (numpy) array of length N, or None if the
        live-volume state has not been set up yet.

        VARAHA draws uniformly over its live volume -- the union of the
        currently-occupied hypercubes (self.binunique), each of width self.dx.
        The density is therefore the SAME constant this sampler reports in
        integrate_log's log_joint_s_prior,

            q_live = 1 / (n_occupied_bins * prod(dx))   (== 1/(V*prod(dx0))
                                                          for VARAHA's geometric V),

        inside the live volume and 0 outside it.  We use the geometric form
        1/(n_bins*prod(dx)) directly: it is *exactly* the density of the points
        draw_simple() produces (equal draws per occupied bin, uniform within a
        bin), so a multiple-importance-sampling denominator built from it is
        unbiased regardless of any drift between the tracked scalar V and the
        actual bin grid.

        This method is READ-ONLY -- it does not touch any sampler state and does
        not affect this sampler's own integrate_log.  It exists so the portfolio
        can form the balance-heuristic mixture density q_mix = sum_m w_m q_m.
        """
        binunique = getattr(self, 'binunique', None)
        dx = getattr(self, 'dx', None)
        if binunique is None or dx is None or not hasattr(self, 'my_ranges'):
            return None
        X = np.atleast_2d(np.asarray(identity_convert(X), dtype=float))
        ndim = len(self.params_ordered)
        if X.shape[1] != ndim and X.shape[0] == ndim:
            X = X.T  # tolerate (ndim, N)
        box_lo = self.my_ranges.T[0]
        box_hi = self.my_ranges.T[1]
        dx = np.asarray(identity_convert(dx), dtype=float)
        bins = np.asarray(identity_convert(binunique)).astype(np.int64)
        n_bins = bins.shape[0]
        if n_bins == 0:
            return np.zeros(X.shape[0], dtype=float)
        # bin index of each point (same floor((x-lo)/dx) mapping the sampler uses
        # in integrate_log to build binidx), then test membership in the occupied
        # set.  Points outside the box floor out of range and are excluded below.
        binidx = np.floor((X - box_lo) / dx).astype(np.int64)
        binset = set(map(tuple, bins.tolist()))
        inside = np.array([tuple(row) in binset for row in binidx], dtype=bool)
        inside &= np.all((X >= box_lo) & (X <= box_hi), axis=1)
        q_live = 1.0 / (float(n_bins) * float(np.prod(dx)))
        q = np.zeros(X.shape[0], dtype=float)
        q[inside] = q_live
        return q

    def _allocate_nbins(self, live_pts, delta_V, ndim):
        """Per-axis bin counts whose product over adaptive dims equals 1/delta_V (the same
        total resolution the isotropic split uses, so the volume V=n_bins*prod(dx) and hence
        the estimator are unchanged).

        Default (self.anisotropic_bins False): equal split -- nbins_i = (1/delta_V)**(1/d).
        Anisotropic: redistribute that SAME total bin budget by each axis's *compressibility*
        c_i = log(range_i / spread_i), where spread_i is the std of the live points on axis i.
        Axes whose points fill only a small fraction of their range (tight: phase, polarization,
        sky) get many bins (fine); broad axes (distance, inclination) get few (coarse).  This
        lets the live hypercube wrap a correlated/degenerate posterior far more tightly than an
        isotropic grid, which must use one resolution for both the narrow and the broad axes."""
        nbins = np.ones(ndim)
        if self.d_adaptive <= 0:
            return nbins
        adaptive = np.ones(ndim, dtype=bool)
        if len(self.indx_not_adaptive):
            adaptive[np.array(self.indx_not_adaptive, dtype=int)] = False
        total_log = -np.log(max(float(delta_V), 1e-300))   # log(1/delta_V): total log-bins to spread
        n_live = 0 if live_pts is None else len(live_pts)
        if (not self.anisotropic_bins) or n_live < max(8, 2 * self.d_adaptive):
            nbins[adaptive] = np.exp(total_log / self.d_adaptive)      # isotropic fallback
            return nbins
        lp = np.asarray(identity_convert(live_pts))
        rng = np.diff(self.my_ranges, axis=1).flatten()               # range per axis
        spread = lp.std(axis=0)
        spread = np.maximum(spread, 1e-6 * np.maximum(rng, 1e-30))
        c = np.clip(np.log(np.maximum(rng, 1e-30) / spread), 0.0, None)  # compressibility
        c[~adaptive] = 0.0
        csum = c[adaptive].sum()
        if csum <= 0:
            nbins[adaptive] = np.exp(total_log / self.d_adaptive)     # degenerate -> isotropic
        else:
            nbins[adaptive] = np.exp(c[adaptive] / csum * total_log)  # prod(adaptive)=1/delta_V
        return nbins

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
          allx = identity_convert_togpu(allx)
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

            # admit only FINITE samples above threshold: a cold portfolio member can draw points
            # whose loglkl is -inf/NaN (out-of-support / degenerate extrinsic config).  With the
            # initial threshold -1e15 the plain "> thr" test then passes NaN (breaking later maxes)
            # or, if ALL are non-finite, yields an empty set -> the reported crash chain
            # (get_likelihood_threshold max of empty array; then this method's max at line ~532).
            idxsel = xpy_here.where(xpy_here.logical_and(loglkl > loglkl_thr, xpy_here.isfinite(loglkl)))
            #only admit samples that lie inside the live volume, i.e. one that cross likelihood threshold
            allx = xpy_here.append(allx, rv[idxsel], axis = 0)
            allloglkl = xpy_here.append(allloglkl, loglkl[idxsel])
            allp = xpy_here.append(allp, log_joint_p_prior[idxsel])
            ninj = len(allloglkl)
            if ninj == 0:
                # Nothing finite in the live volume this step (cold portfolio member / degenerate
                # draw).  Leave V and the grid UNCHANGED rather than crashing on empty-array
                # reductions downstream.  The portfolio's other members carry this step; a later
                # draw with finite samples lets AV resume training.  (This method is a SINGLE
                # selfish step, so an early return is correct -- unlike integrate_log's loop.)
                print("  [AV selfish-update] no finite in-volume samples this step; live volume unchanged")
                self.V = V
                return


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
            if nrec == 0:
                # threshold selected nothing (degenerate all-equal finite draws): leave the
                # live volume unchanged instead of crashing on max()/divide-by-zero below.
                self.V = V
                return

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
              # per-axis (anisotropic) or equal (default) split; same total bin budget either way
              self.nbins = self._allocate_nbins(allx, delta_V, ndim)
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


    ###
    ### BOOTSTRAP / WARM-START SUPPORT
    ###
    # The VARAHA algorithm normally starts every integrate_log() call cold: one
    # bin spanning the whole box, threshold -1e15, fractional volume V=1, and
    # spends its first several chunks carving the live volume down from the full
    # prior.  In production (repeated ILE instances, successive CIP iterations,
    # or events with a known Fisher matrix) we already know roughly where the
    # posterior lives, so that carving is wasted work -- worst in high dimension.
    #
    # These methods seed the live-volume state (`self._warm`) from prior
    # information; integrate_log() then starts from that concentrated grid.  The
    # seeded fractional volume is set GEOMETRICALLY (n_occupied_bins / prod(nbins))
    # so the final integral normalization (log_joint_s_prior = log(1/V) - sum log dx0)
    # stays unbiased regardless of how the state was produced.

    def _order_columns(self, samples, params=None):
        """Return samples as an (M, ndim) array whose columns are in
        self.params_ordered order.  `params` names the columns of `samples`;
        if None the caller guarantees they are already in order."""
        X = np.atleast_2d(np.asarray(samples, dtype=float))
        if X.shape[1] != len(self.params_ordered) and X.shape[0] == len(self.params_ordered):
            X = X.T  # tolerate (ndim, M)
        if params is None:
            return X
        out = np.empty((X.shape[0], len(self.params_ordered)))
        for j, p in enumerate(self.params_ordered):
            out[:, j] = X[:, list(params).index(p)]
        return out

    def _build_grid_from_points(self, pts, loglkl=None, enc_prob=0.999, dilate=1,
                                resolution_pts=None):
        """Build a VARAHA live-volume grid (binunique, dx, nbins) and a
        geometrically-consistent fractional volume V from points that populate
        the high-likelihood region.  Mirrors the bin-refinement block of
        integrate_log() so a warm start lands on the same kind of grid the cold
        algorithm would have converged to.

        `dilate` (>=0): grow the occupied-bin set by this many axis-neighbor
        layers along the adaptive dimensions.  This is a SAFETY margin: VARAHA's
        live volume only ever *contracts*, so a warm start that seeded a grid
        tighter than the true support could never recover the missing region and
        would bias the integral low.  Dilating guarantees the seed is a superset
        of the sampled support (at a small efficiency cost the first few chunks
        then trim away)."""
        ndim = len(self.params_ordered)
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        box_lo = self.my_ranges.T[0]
        box_hi = self.my_ranges.T[1]
        inside = np.all((pts >= box_lo) & (pts <= box_hi), axis=1)
        pts = pts[inside]
        if loglkl is not None:
            loglkl = np.asarray(loglkl, dtype=float)[inside]
        nrec = len(pts)
        if nrec < 2:
            raise ValueError("AV bootstrap needs >=2 in-box reference points (got {})".format(nrec))
        box = box_hi - box_lo
        # Bin RESOLUTION (nbins) is set from the CONCENTRATED core, not the full
        # cloud: when a coverage floor (cover_frac) adds uniform full-box points,
        # they must not coarsen the grid to a single bin per dim (which collapses
        # V to 1 and throws away the seed's concentration).  resolution_pts is the
        # core (the actual proposal, without the uniform floor); coverage points
        # then land in scattered fine bins that still guarantee coverage.
        res_pts = pts if resolution_pts is None else np.atleast_2d(np.asarray(resolution_pts, dtype=float))
        n_res = max(len(res_pts), 2)
        lo = np.quantile(res_pts, 0.5 * (1 - enc_prob), axis=0)
        hi = np.quantile(res_pts, 1 - 0.5 * (1 - enc_prob), axis=0)
        ext = np.clip(hi - lo, box * 1e-6, None)
        V_extent = float(np.prod(ext / box))
        # VARAHA bin count: nbins = (1/delta_V)^(1/d_adaptive), delta_V = V/sqrt(nrec)
        delta_V = V_extent / np.sqrt(n_res)
        if self.d_adaptive > 0:
            # per-axis (anisotropic) or equal (default) split of the warm-seed grid
            nbins = self._allocate_nbins(res_pts, delta_V, ndim)
            nbins[self.indx_not_adaptive] = 1
        else:
            nbins = np.ones(ndim)
        nbins = np.maximum(np.floor(nbins), 1)
        dx = box / nbins
        # CLIP bin indices to [0, nbins-1]: a point exactly on the upper box edge
        # maps to binidx == nbins (out of range), which would put out-of-range bins
        # in binunique -> V = n_bins/prod(nbins) can exceed 1 (an invalid fractional
        # volume) and draw_simple would sample outside the box.  This bites hardest
        # for a WIDE seed (e.g. a full PE posterior + cover_frac spanning the box).
        nb_int = np.maximum(nbins.astype(np.int64), 1)
        binidx = np.clip(((pts - box_lo) / dx).astype(np.int64), 0, nb_int - 1)
        binunique = np.unique(binidx, axis=0)
        # SAFETY dilation: grow occupied bins by axis-neighbor layers along the
        # adaptive dims, clipped to [0, nbins-1].  Uses a bounded 2*d_adaptive
        # neighborhood per layer (not the full 3^d) so the volume grows linearly.
        if dilate and self.d_adaptive > 0:
            bins = set(map(tuple, binunique.tolist()))
            nb_max = nbins.astype(int)
            for _ in range(int(dilate)):
                grown = set(bins)
                for b in bins:
                    for ax in self.indx_adaptive:
                        for step in (-1, 1):
                            nb = list(b); nb[ax] += step
                            if 0 <= nb[ax] < nb_max[ax]:
                                grown.add(tuple(nb))
                bins = grown
            binunique = np.array(sorted(bins))
        # fractional volume ACTUALLY sampled = occupied bins / total bins
        V = float(binunique.shape[0] / np.prod(nbins))
        # seed the threshold just below the reference support so the first chunk
        # keeps the seeded region; if no lnL given, let integrate_log recompute it
        # (the concentrated grid already delivers the efficiency win).
        loglkl_thr = -1e15 if loglkl is None else float(np.min(loglkl))
        return dict(binunique=binunique, dx=dx, nbins=nbins, V=V,
                    loglkl_thr=loglkl_thr, trunc_p=1e-10)

    def bootstrap_from_samples(self, samples, params=None, loglkl=None, enc_prob=0.999,
                               cover_frac=0.0, dilate=1, inflate=1.0, seed=None):
        """Warm-start from an explicit set of reference points populating the
        high-likelihood region (e.g. a previous run's posterior draws, a puff of
        an earlier MAP point, or fair-draw samples from a prior ILE instance).
        `loglkl` (optional) is L*prior at those points, used to seed the threshold.

        `cover_frac` (0..1) is the SAFETY FLOOR for reuse across DIFFERENT problems
        (a neighbouring intrinsic point, a stale breadcrumb): it mixes this fraction
        of uniform full-box points into the seed cloud, so the seeded live volume is
        a superset of a cold (uniform) start.  Then a mis-placed proposal can only
        cost efficiency -- warm coverage always contains cold coverage, so the
        warm-started integral can never be MORE biased than a cold one.  Leave 0
        when reusing a proposal for the SAME problem (e.g. an in-run second pass).

        `inflate` (>=1) is the HANDOFF SAFETY MARGIN: widen the seed cloud by this
        factor about its own mean before building the grid.  When importing a proposal
        that came from a *neighbouring* intrinsic point, the true peak is shifted (and
        often slightly broader) at this point, so an un-inflated seed may sit just off
        it; inflate>1 (e.g. 1.5-2) gives margin for that shift while staying far tighter
        than a cold start.  cover_frac is the coarse safety net for gross mismatch;
        inflate is the fine margin for a modest shift."""
        if not hasattr(self, 'my_ranges'):
            self.setup()
        X = self._order_columns(samples, params)
        inflate = float(max(inflate, 1.0))
        if inflate > 1.0 and len(X) >= 2:
            _m = np.mean(X, axis=0)
            X = _m + inflate * (X - _m)
            X = np.clip(X, self.my_ranges.T[0], self.my_ranges.T[1])
            loglkl = None   # inflated points no longer carry their original lnL
        cover_frac = float(np.clip(cover_frac, 0.0, 1.0))
        _core = X   # the concentrated proposal; sets the grid RESOLUTION
        if cover_frac > 0:
            rng = np.random.RandomState(seed)
            n_cover = max(int(cover_frac / (1.0 - cover_frac) * len(X)), 1)
            Xc = rng.uniform(self.my_ranges.T[0], self.my_ranges.T[1],
                             size=(n_cover, len(self.params_ordered)))
            X = np.vstack([X, Xc])
            # the cover points are not part of the high-L region, so drop the
            # lnL-threshold seed (let integrate_log recompute it from the data)
            loglkl = None
        # resolution from the core (not the uniform cover), so the coverage floor
        # cannot coarsen away the proposal's concentration (a wide PE + cover_frac
        # would otherwise collapse the grid to one bin per dim, V->1)
        self._warm = self._build_grid_from_points(X, loglkl=loglkl, enc_prob=enc_prob,
                                                  dilate=dilate, resolution_pts=_core)
        self._warm_applied = False   # a NEW seed must be re-installed (L0 rescue re-seeds mid-run)
        return self._warm

    def bootstrap_from_gaussian(self, mean, cov, n=None, params=None, enc_prob=0.999,
                                seed=None, cover_frac=0.0, dilate=1):
        """Warm-start from a single Gaussian proposal N(mean, cov) -- the
        Fisher-oracle entry point.  Draws `n` points from the (box-clipped)
        Gaussian and builds the live-volume grid from them.

        IMPORTANT -- this is a UNIMODAL seed.  A single Gaussian covers only one
        mode, and because VARAHA's live volume only ever contracts, any mode the
        seed misses is lost forever and biases the integral low.  Use this only
        when the target is (locally) unimodal -- e.g. a Fisher matrix at the MAP.
        For known multimodal structure use bootstrap_from_gaussian_mixture(); for
        an empirical proposal use bootstrap_from_samples() (both cover the full
        support and stay unbiased).

        `cover_frac` (0..1): optional safety valve -- fraction of the seed cloud
        drawn uniformly from the full box, trading efficiency for coverage on a
        possibly-misspecified seed.  Default 0."""
        if not hasattr(self, 'my_ranges'):
            self.setup()
        rng = np.random.RandomState(seed)
        mean = np.asarray(mean, dtype=float)
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        if params is not None:
            order = [list(params).index(p) for p in self.params_ordered]
            mean = mean[order]
            cov = cov[np.ix_(order, order)]
        n = int(n or self.n_chunk)
        n_cover = int(np.clip(cover_frac, 0.0, 1.0) * n)
        n_gauss = n - n_cover
        X = rng.multivariate_normal(mean, cov, size=n_gauss)
        X = np.clip(X, self.my_ranges.T[0], self.my_ranges.T[1])
        if n_cover > 0:
            Xc = rng.uniform(self.my_ranges.T[0], self.my_ranges.T[1],
                             size=(n_cover, len(self.params_ordered)))
            X = np.vstack([X, Xc])
        self._warm = self._build_grid_from_points(X, enc_prob=enc_prob, dilate=dilate)
        self._warm_applied = False   # a NEW seed must be re-installed (L0 rescue re-seeds mid-run)
        return self._warm

    def bootstrap_from_fisher(self, mean, fisher, **kwargs):
        """Warm-start from a Fisher matrix (mean, Gamma): cov = Gamma^{-1}.
        This is the 'Fisher-matrix oracle' -- an essentially free substitute for
        an expensively-trained flow, giving the integrator a correct-to-2nd-order
        starting proposal."""
        cov = np.linalg.inv(np.atleast_2d(np.asarray(fisher, dtype=float)))
        return self.bootstrap_from_gaussian(mean, cov, **kwargs)

    def bootstrap_from_gaussian_mixture(self, means, covs, weights=None, n=None,
                                        params=None, enc_prob=0.999, seed=None, dilate=1):
        """Warm-start from a MIXTURE of Gaussians -- the general oracle seed for
        multimodal targets.  This is what a flow oracle, a GMM fit of a previous
        posterior, or a set of known degenerate modes (e.g. sky reflections)
        provides.  Because the seed cloud covers every component, the resulting
        live volume is a superset of the support and the integral stays
        unbiased."""
        if not hasattr(self, 'my_ranges'):
            self.setup()
        rng = np.random.RandomState(seed)
        means = [np.asarray(m, dtype=float) for m in means]
        covs = [np.atleast_2d(np.asarray(c, dtype=float)) for c in covs]
        k = len(means)
        weights = np.ones(k) / k if weights is None else np.asarray(weights, float) / np.sum(weights)
        if params is not None:
            order = [list(params).index(p) for p in self.params_ordered]
            means = [m[order] for m in means]
            covs = [c[np.ix_(order, order)] for c in covs]
        n = int(n or self.n_chunk)
        counts = rng.multinomial(n, weights)
        chunks = []
        for c, m, cov in zip(counts, means, covs):
            if c > 0:
                chunks.append(rng.multivariate_normal(m, cov, size=c))
        X = np.vstack(chunks)
        X = np.clip(X, self.my_ranges.T[0], self.my_ranges.T[1])
        self._warm = self._build_grid_from_points(X, enc_prob=enc_prob, dilate=dilate)
        self._warm_applied = False   # a NEW seed must be re-installed (L0 rescue re-seeds mid-run)
        return self._warm

    def save_state(self, path):
        """Serialize the compact live-volume state (occupied bins + widths +
        volume + threshold) to a lightweight .npz.  This is RIFT's cheap
        alternative to persisting a trained flow: the entire adapted proposal is
        just an integer bin-index array plus a few scalars."""
        warm = getattr(self, '_warm', None)
        if warm is None:
            thr = float(self.lnL_thresh) if np.isfinite(self.lnL_thresh) else -1e15
            warm = dict(binunique=self.binunique, dx=self.dx, nbins=self.nbins,
                        V=float(self.V), loglkl_thr=thr, trunc_p=1e-10)
        np.savez(path,
                 params=np.array([str(p) for p in self.params_ordered]),
                 llim=self.my_ranges.T[0], rlim=self.my_ranges.T[1],
                 binunique=warm['binunique'], dx=warm['dx'], nbins=warm['nbins'],
                 V=warm['V'], loglkl_thr=warm['loglkl_thr'],
                 trunc_p=warm.get('trunc_p', 1e-10))
        return path

    def load_state(self, path):
        """Restore a live-volume state saved by save_state().  Verifies the
        parameter names and box match this sampler before warm-starting."""
        if not hasattr(self, 'my_ranges'):
            self.setup()
        d = np.load(path, allow_pickle=True)
        saved_params = [str(p) for p in d['params']]
        if saved_params != [str(p) for p in self.params_ordered]:
            raise ValueError("saved state params {} != sampler params {}".format(
                saved_params, [str(p) for p in self.params_ordered]))
        if not (np.allclose(d['llim'], self.my_ranges.T[0]) and
                np.allclose(d['rlim'], self.my_ranges.T[1])):
            raise ValueError("saved state box does not match sampler box")
        self._warm_applied = False   # new seed -> must be re-installed
        self._warm = dict(binunique=np.array(d['binunique']), dx=np.array(d['dx']),
                          nbins=np.array(d['nbins']), V=float(d['V']),
                          loglkl_thr=float(d['loglkl_thr']), trunc_p=float(d['trunc_p']))
        return self._warm


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
        n_adapt = int(kwargs["n_adapt"]*n) if "n_adapt" in kwargs else 1000  # default to adapt to 1000 chunks, then freeze
        floor_integrated_probability = kwargs["floor_level"] if "floor_level" in kwargs else 0
        temper_log = kwargs["tempering_log"] if "tempering_log" in kwargs else False
        tempering_adapt = kwargs["tempering_adapt"] if "tempering_adapt" in kwargs else False
            

        save_intg = kwargs["save_intg"] if "save_intg" in kwargs else False
        # opt-in anisotropic (per-axis) bin allocation; also settable as a sampler attribute
        if "anisotropic_bins" in kwargs:
            self.anisotropic_bins = bool(kwargs["anisotropic_bins"])
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

        # WARM START: if this sampler was bootstrapped (bootstrap_from_* /
        # load_state), override the cold single-bin grid, fractional volume and
        # threshold with the seeded live-volume state.  self.setup() above has
        # already reset these to cold defaults, so we re-apply the seed here.
        warm = getattr(self, '_warm', None)
        if warm is not None:
            self.binunique = np.array(warm['binunique'])
            self.dx = np.array(warm['dx'])
            self.nbins = np.array(warm['nbins'])
            self.ninbin = ((self.n_chunk // self.binunique.shape[0] + 1) * np.ones(self.binunique.shape[0])).astype(int)
            V = float(warm['V'])
            loglkl_thr = float(warm['loglkl_thr'])
            trunc_p = float(warm.get('trunc_p', 1e-10))
            if bShowEvaluationLog:
                print("  [AV warm-start] live bins={} V={:.3e} loglkl_thr={:.3g}".format(
                    self.binunique.shape[0], V, loglkl_thr))

        var_lnV = 0.0  # accumulated variance of ln(V): V is a stochastic product of per-cycle
                       # binomial survival fractions, and Z ~ V*mean(w), so Var(lnV) is a
                       # component of the lnZ error the weight variance is structurally blind to
        if cupy_ok:
          allx = identity_convert_togpu(allx)
          allloglkl = identity_convert_togpu(allloglkl)

        ntotal_true = 0
        while (eff_samp < neff and ntotal_true < nmax ): #  and (not bConvergenceTests):
            # Draw samples. Note state variables binunique, ninbin -- so we can re-use the sampler later outside the loop
            rv, log_joint_p_prior = self.draw_simple()  # Beware reversed order of rv
            ntotal_true += len(rv)
            if cupy_ok:
              rv = identity_convert_togpu(rv) # send random numbers to GPU : ugh
              log_joint_p_prior = identity_convert_togpu(log_joint_p_prior)    # send to GPU if required. Don't waste memory reassignment otherwise

            # Evaluate the integrand.  Two contracts exist: the production
            # GPU/vectorized ILE likelihood is DEVICE-native (wants cupy arrays),
            # while synthetic/host integrands (CI tests, benchmarks) want numpy.
            # Feed the native (device) array -- matching production -- but if a
            # host-only integrand chokes on a cupy array, fall back to a host copy
            # and remember the choice for the rest of the run.  (The previous
            # version always fed a host copy, which silently broke the real GPU
            # ILE likelihood with 'Unsupported type numpy.ndarray'.)
            def _eval_integrand(samples):
                if 'no_protect_names' in kwargs:
                    return lnF(*samples.T)
                return lnF(**dict(list(zip(self.params_ordered, samples.T))))
            if getattr(self, '_integrand_wants_host', False):
                lnL = _eval_integrand(identity_convert(rv))
            else:
                try:
                    lnL = _eval_integrand(rv)
                except (TypeError, ValueError):
                    self._integrand_wants_host = True
                    lnL = _eval_integrand(identity_convert(rv))
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
              # per-axis (anisotropic) or equal (default) split; same total bin budget either way
              self.nbins = self._allocate_nbins(allx, delta_V, ndim)
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
            # accumulate the binomial variance of this cycle's ln(V) update:
            # Var(ln p_hat) ~= (1-p_hat)/(n p_hat) = (1-nrec/ninj)/nrec.  Cycles reuse
            # surviving samples, so this is an approximate (disclosed) budget rather
            # than a rigorous iid propagation; it vanishes as the volume stabilizes.
            if nrec > 0 and ninj > 0:
                var_lnV += (1.0 - nrec/ninj)/nrec

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
        # ones_like(allloglkl) follows allloglkl's backend (cupy via numpy's
        # __array_function__ dispatch when on GPU); xpy_here.ones(len) would
        # instead create a host array, leaving this term on a different backend
        # than log_integrand / log_joint_prior and breaking the arithmetic below.
        self._rvs['log_joint_s_prior'] = xpy_here.ones_like(allloglkl)*(np.log(1/V) - np.sum(np.log(self.dx0)))  # effective uniform sampling on this volume

        # Manual estimate of integrand, done transparently (no 'log aggregate' or running calculation -- so memory hog
        log_wt = self._rvs["log_integrand"] + self._rvs["log_joint_prior"] - self._rvs["log_joint_s_prior"]
        log_wt = identity_convert(log_wt)  # convert to CPU
        log_int = special.logsumexp( log_wt) - np.log(len(log_wt))  # mean value
        rel_var_mc = np.var( np.exp(log_wt - log_int))/len(log_wt)   # error in integral, estimated: just taking int = <w> , so error is V(w_k)/N (sample mean/variance)
        # Total DISCLOSED relative variance: the naive weight-variance term above is
        # structurally blind to (a) the stochasticity of the live volume V itself
        # (Z ~ V*mean(w); var_lnV accumulated per cycle) and (b) the probability
        # deliberately truncated by the likelihood threshold (trunc_p, a one-sided
        # systematic entered here as a variance in quadrature).  Add them.
        rel_var = rel_var_mc + var_lnV + trunc_p**2
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
        # MC-error diagnostics: disclose the components and the weight-tail state.
        # NOTE the AV estimator assigns the surviving (threshold-selected) samples a
        # pretend-uniform density on the final live volume, so the naive term is if
        # anything MORE optimistic than for the other samplers -- k-hat matters here.
        try:
            mc_diag = {'sigma_lnZ_mc': float(np.sqrt(rel_var_mc)),
                       'sigma_lnV': float(np.sqrt(var_lnV)),
                       'trunc_p': float(trunc_p)}
            _kh = pareto_khat_from_log(log_wt)
            if _kh is not None:
                mc_diag['pareto_khat'] = _kh
            mc_diag['n_ESS'] = ess_from_log_weights(log_wt)
            if np.sqrt(rel_var) > 0.3:
                _q = bootstrap_lnZ_quantiles(log_wt, n_total=len(log_wt))
                if _q is not None:
                    mc_diag['lnZ_ci90'] = _q
            dict_return.update(mc_diag)
            print(" [AV mc diag] sigma_mc={:.4f} sigma_lnV={:.4f} trunc_p={:.2e} khat={} ESS={}".format(
                mc_diag['sigma_lnZ_mc'], mc_diag['sigma_lnV'], mc_diag['trunc_p'],
                round(mc_diag['pareto_khat'],3) if 'pareto_khat' in mc_diag else None,
                round(mc_diag['n_ESS'],1) if 'n_ESS' in mc_diag else None))
        except Exception as _e_diag:
            print(" mcsamplerAdaptiveVolume: MC-error diagnostics failed ({}); continuing.".format(_e_diag))
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


# PROVIDE CROSS-CODE DEPLOY, without rewrite
from RIFT.integrators.mcsamplerGPU import  uniform_samp_cdf_inv_vector,     ret_uniform_samp_vector_alt,uniform_samp_phase, cos_samp_vector, cos_samp_cdf_inv_vector,ret_uniform_samp_vector_alt, uniform_samp_theta,uniform_samp_psi, dec_samp_vector, dec_samp_cdf_inv_vector, uniform_samp_dec
from RIFT.integrators.mcsamplerGPU import q_samp_vector, M_samp_vector,q_cdf_inv_vector
