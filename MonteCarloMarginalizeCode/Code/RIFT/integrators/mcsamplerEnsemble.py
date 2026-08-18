import sys
import math
import bisect
from collections import defaultdict

import numpy as np
from RIFT.precision import RiftFloat  # platform-portable replacement for np.float128

try:
    import cupy
    import cupyx.scipy.special
    # Probe for an actual device: cupy imports cleanly on GPU-less nodes but
    # every kernel launch then dies with cudaErrorNoDevice.  getDeviceCount
    # raises CUDARuntimeError (not ImportError), hence the broad except.
    if cupy.cuda.runtime.getDeviceCount() == 0:
        raise ImportError("cupy installed but no CUDA device available")
    xpy_default = cupy
    xpy_special_default = cupyx.scipy.special
    identity_convert = cupy.asnumpy
    identity_convert_togpu = cupy.asarray
    cupy_ok = True
except Exception:
    xpy_default = np
    xpy_special_default = None
    identity_convert = lambda x: x
    identity_convert_togpu = lambda x: x
    cupy_ok = False

import itertools
import functools

import scipy.special
import scipy.stats as stats

#from statutils import cumvar

from multiprocessing import Pool


# Mirror healpy stuff
from RIFT.integrators.mcsampler import HealPixSampler

from . import MonteCarloEnsemble as monte_carlo

__author__ = "Ben Champion"

rosDebugMessages = True

from RIFT.integrators.rvs_record import RvsRecord, SamplerOutputMixin   # see DESIGN_rvs_naming.md

class NanOrInf(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MCSampler(SamplerOutputMixin, object):

    @property
    def has_unbounded_support(self):
        """Does this member's proposal genuinely have support across the WHOLE prior box?

        mcsamplerPortfolio uses this to decide whether it must hold a member cold on a warm start.
        Getting it wrong costs coverage silently, so it is answered from the INSTALLED MODELS, not
        from a configuration value.

        Two traps this avoids:
          * `gmm_defensive_frac > 0` is only a REQUEST.  add_defensive_component() is called by
            fit_gmm_adaptive, but the fixed-component fit paths did not call it, and gmm_adaptive
            defaults to None (off) -- so the default configuration asked for a defensive component
            and never installed one.
          * gmm.score() FLOORS its return at 1e-300, so a member always looks like it has nonzero
            density everywhere.  That is a numerical guard against log(0), not coverage: a sample
            landing there carries weight L*p/q ~ 1e300 and would wreck the estimate rather than
            support it.  Measured, a fixed-component fit to a tight cloud returns exactly that
            floor at the far corner for every d >= 4.

        Reports False whenever it cannot be verified -- before the integrator exists, before any
        group has been trained, or if ANY trained group lacks the component.
        """
        integ = getattr(self, 'integrator', None)
        if integ is None:
            return False
        if not (getattr(integ, 'gmm_defensive_frac', 0.0) or 0.0) > 0:
            return False
        models = [m for m in getattr(integ, 'gmm_dict', {}).values() if m is not None]
        if not models:
            if not getattr(integ, 'gmm_defensive_all_paths', False) and not getattr(
                    integ, 'gmm_adaptive', None):
                # Neither path that installs the component is active: the request in
                # gmm_defensive_frac will not be honoured, so do not promise coverage.
                return False
            # UNTRAINED.  The portfolio has to decide about warm-starting before any group is
            # fitted, so there is nothing to inspect yet.  Trusting the config is justified only
            # because EVERY fit path now installs the component (fit_gmm_adaptive did already;
            # the fixed-component paths in this file and in MonteCarloEnsemble were fixed at the
            # same time as this check).  test_every_fit_path_installs_the_defensive_component
            # pins that invariant -- if a new fit path is added without it, that test fails rather
            # than this property silently over-promising again.
            return True
        return all((getattr(m, 'defensive_frac', 0.0) or 0.0) > 0 for m in models)

    """
    Class to define a set of parameter names, limits, and probability densities.
    """

    @staticmethod
    def match_params_from_args(args, params):
        not_common = set(args) ^ set(params)
        if len(not_common) == 0:
            return True
        if all([not isinstance(i, tuple) for i in not_common]):
            return False

        to_match = [i for i in not_common if not isinstance(i, tuple)]
        against = [i for i in not_common if isinstance(i, tuple)]
        
        matched = []
        import itertools
        for i in range(2, max(list(map(len, against)))+1):
            matched.extend([t for t in itertools.permutations(to_match, i) if t in against])
        return (set(matched) ^ set(against)) == set()


    def __init__(self):
        self.ntotal = 0
        self.n = 0
        self.params = set()
        self.params_ordered = []
        self.pdf = {}
        self._pdf_norm = defaultdict(lambda: 1)
        self._rvs = {}
        self.cdf = {}
        self.cdf_inv = {}
        self.llim, self.rlim = {}, {}
        self.adaptive = []
        self._hist = {}
        self.prior_pdf = {}
        self.func = None
        self.sample_format = None
        self.curr_args = None
        self.gmm_dict ={} 
        self.integrator = None 

        self.xpy = xpy_default
        self.identity_convert = identity_convert
        self.identity_convert_togpu = identity_convert_togpu


    def clear(self):
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
        self.integrator=None

    def add_parameter(self, params, pdf=None,  cdf_inv=None, left_limit=None, right_limit=None, 
                        prior_pdf=None, adaptive_sampling=False):
        self.params.add(params)
        self.params_ordered.append(params)
        if rosDebugMessages:
            print(" mcsampler: Adding parameter ", params, " with limits ", [left_limit, right_limit])
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
            if prior_pdf is not None:
                for p in params:
                    self.prior_pdf[p] = prior_pdf
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
            if prior_pdf is not None:
                self.prior_pdf[params] = prior_pdf

    def evaluate(self, samples):
        # The user integrand is a host (numpy/scipy) function in general, so move
        # samples to the CPU before calling it and push the result back to the
        # active backend (cupy on GPU). This is a no-op when xpy is numpy.
        samples = self.identity_convert(samples)
        temp = []
        for index in range(len(self.curr_args)):
            temp.append(samples[:,index])
        temp_ret = self.identity_convert_togpu(self.func(*temp))
        # column vector (n,1); cupy.rot90 does not accept array-likes/lists, and
        # reshape is backend-agnostic and order-preserving (equiv. to the old
        # np.rot90([temp_ret], -1)).
        return temp_ret.reshape((-1, 1))


    def calc_pdf(self, samples):
        n, _ = samples.shape
        temp_ret = self.xpy.ones((n, 1))
        # Prior pdfs are host functions in general; evaluate them on CPU samples
        # and convert the result back to the active backend.
        samples_cpu = self.identity_convert(samples)
        for index in range(len(self.curr_args)):
            if self.curr_args[index] in self.prior_pdf:
                pdf_func = self.prior_pdf[self.curr_args[index]]
                temp_samples = samples_cpu[:,index]
                pdf_vals = self.identity_convert_togpu(pdf_func(temp_samples))
                temp_ret *= pdf_vals.reshape( temp_ret.shape)
        return temp_ret

    def setup(self, n_comp=None, **kwargs):
        """Build the integrator.  REMEMBERS its arguments and re-applies them on later calls.

        setup() is not called once: bootstrap_from_samples() re-runs it to rebuild the proposal as
        a single full-dim group, and mcsamplerPortfolio replays it to reset a member between
        points.  Each of those rebuilt the integrator from ONLY the kwargs of that call, so every
        option the caller set originally was silently dropped.  That has now bitten three separate
        settings -- gmm_dict (the dimension grouping and any seeded models), gmm_defensive_frac,
        and gmm_defensive_all_paths -- each found as its own P1, each patched individually, and a
        fourth would have followed.

        So: merge this call's kwargs OVER the remembered ones, and remember the result.  An
        explicit argument still wins; an omitted one keeps whatever it was configured to be
        instead of reverting to a library default.  Pass `setup_forget=True` to start clean.
        """
        _prev = dict(getattr(self, '_setup_kwargs_seen', {}) or {})
        if kwargs.pop('setup_forget', False):
            _prev = {}
        if n_comp is None:
            n_comp = _prev.get('n_comp', None)
        merged = dict(_prev)
        merged.update(kwargs)
        merged['n_comp'] = n_comp
        self._setup_kwargs_seen = dict(merged)
        kwargs = dict(merged)
        kwargs.pop('n_comp', None)
        return self._setup_impl(n_comp=n_comp, **kwargs)

    def _setup_impl(self, n_comp=None, **kwargs):
      # n_comp=None silently disabled ALL training downstream: the integrator
      # stores it verbatim and update_sampling_prior only builds a model for
      # int!=0 or dict n_comp, so every gmm_dict entry stayed None forever.  In
      # default portfolio wiring (setup() forwarded without GMM args) the GMM
      # member therefore never trained and "portfolio" ran as AV-only, with no
      # error (2026-07-22 shape-gate probe).  Default to a single component and
      # say so; n_comp=0 remains the explicit off-switch.
      if n_comp is None:
          print(" mcsamplerEnsemble: setup() called without n_comp; defaulting n_comp=1 "
                "(n_comp=None previously disabled GMM training silently; pass n_comp=0 to disable adaptation)")
          n_comp = 1
      integrator_func  = kwargs['integrator_func'] if "integrator_func" in kwargs  else None
      mcsamp_func  = kwargs['mcsamp_func'] if "mcsamp_func" in kwargs  else None
      proc_count = kwargs['proc_count'] if "proc_count" in kwargs else None
      direct_eval = kwargs['direct_eval'] if "direct_eval" in kwargs else False
      min_iter = kwargs['min_iter'] if "min_iter" in kwargs else 10
      max_iter = kwargs['max_iter'] if "max_iter" in kwargs else 20
      var_thresh = kwargs['var_thres'] if "var_thresh" in kwargs else 0.05
      write_to_file = kwargs['write_to_file'] if "write_to_file" in kwargs else False
      correlate_all_dims = kwargs['correlate_all_dims'] if  "correlate_all_dims" in kwargs else False
      gmm_adapt = kwargs['gmm_adapt'] if "gmm_adapt" in kwargs else None
      gmm_adaptive = kwargs['gmm_adaptive'] if "gmm_adaptive" in kwargs else None
      gmm_defensive_frac = kwargs['gmm_defensive_frac'] if "gmm_defensive_frac" in kwargs else 0.05
      _defensive_all = kwargs['gmm_defensive_all_paths'] if "gmm_defensive_all_paths" in kwargs else False
      gmm_inflate = kwargs['gmm_inflate'] if "gmm_inflate" in kwargs else 1.0
      gmm_epsilon = kwargs['gmm_epsilon'] if "gmm_epsilon" in kwargs else None
      L_cutoff = kwargs["L_cutoff"] if "L_cutoff" in kwargs else None
      tempering_exp = kwargs["tempering_exp"] if "tempering_exp" in kwargs else 1.0
      tempering_adapt = kwargs["tempering_adapt"] if "tempering_adapt" in kwargs else False
      ess_target = kwargs["ess_target"] if "ess_target" in kwargs else None
      ess_floor = kwargs["ess_floor"] if "ess_floor" in kwargs else None
      lnw_failure_cut = kwargs["lnw_failure_cut"] if "lnw_failure_cut" in kwargs else None
      nmax = kwargs["nmax"] if "nmax" in kwargs else 1e6
      neff = kwargs["neff"] if "neff" in kwargs else 1000
      n = kwargs["n"] if "n" in kwargs else min(1000, nmax)

      self.n = n
      self.curr_args = self.params_ordered

      if 'gmm_dict' in list(kwargs.keys()):
          gmm_dict = kwargs['gmm_dict']
      else:
          gmm_dict = None
      dim = len(self.params_ordered)
      bounds=[]
      for param in self.params_ordered:
            bounds.append([self.llim[param], self.rlim[param]])
      raw_bounds = self.xpy.array(bounds)
          
      if gmm_dict is None:
            # See note in integrate(): dict keys must be host ints, not 0-d
            # cupy arrays (which are unhashable).
            bounds = {}
            for indx in np.arange(len(raw_bounds)):
                bounds[(indx,)] = raw_bounds[indx]
            bounds=raw_bounds
            if correlate_all_dims:
                gmm_dict = {tuple(range(dim)):None}
                bounds = {tuple(range(dim)): raw_bounds}
            else:
                gmm_dict = {}
                for i in range(dim):
                    gmm_dict[(i,)] = None
      else:
            bounds ={}
            for dims in gmm_dict:
                n_dims = len(dims)
                bounds_here = self.xpy.empty((n_dims,2))
                for indx in range(n_dims):
                    bounds_here[indx] = raw_bounds[dims[indx]]
                bounds[dims]=bounds_here

      self.integrator = monte_carlo.integrator(dim, bounds, gmm_dict, n_comp, n=self.n, prior=self.calc_pdf,
                         user_func=integrator_func, proc_count=proc_count,L_cutoff=L_cutoff,gmm_adapt=gmm_adapt,gmm_epsilon=gmm_epsilon,tempering_exp=tempering_exp,
                         tempering_adapt=tempering_adapt, ess_target=ess_target, ess_floor=ess_floor, gmm_adaptive=gmm_adaptive,
                         gmm_defensive_frac=gmm_defensive_frac, gmm_inflate=gmm_inflate)
      self.integrator.gmm_defensive_all_paths = bool(_defensive_all)

    def update_sampling_prior(self,ln_weights, n_history,tempering_exp=1,log_scale_weights=True,floor_integrated_probability=0,external_rvs=None,**kwargs):
      rvs_here = self._rvs
      if external_rvs:
        rvs_here = external_rvs

      ln_weights  = self.xpy.array(self.identity_convert(ln_weights))
      ln_weights *= tempering_exp

      gmm_dict = self.integrator.gmm_dict

      # These are all host ints; use the Python builtin min (self.xpy.min([list])
      # crashes on cupy -- "'list' object has no attribute 'min'" -- the same
      # backend-min-of-a-list bug fixed in integrate()'s fairdraw block).  A host
      # int is also required for the [-n_history_to_use:] slices just below.
      n_history_to_use = int(min(n_history, len(ln_weights), len(rvs_here[self.params_ordered[0]])))

      # external_rvs (e.g. the portfolio's host history) may be host numpy while
      # sample_array lives on the active backend (cupy on GPU); assigning a host
      # slice into a cupy row raises "non-scalar numpy.ndarray cannot be used for
      # fill".  Convert each slice to the backend first so this method is
      # backend-consistent (previously it only worked on CPU).
      sample_array = self.xpy.empty( (len(self.params_ordered), n_history_to_use))
      for indx, p in enumerate(self.params_ordered):
          sample_array[indx] = self.identity_convert_togpu(rvs_here[p][-n_history_to_use:])
      sample_array = sample_array.T

      for dim_group in gmm_dict:
            if self.integrator.gmm_adapt:
                if (dim_group in self.integrator.gmm_adapt):
                    if not(self.integrator.gmm_adapt[dim_group]):
                        continue
            new_bounds = self.xpy.empty((len(dim_group), 2))
            new_bounds = self.integrator.bounds[dim_group]
            # per-dimension (uncorrelated) groups: setup() hands the integrator raw
            # (dim,2) array bounds, so bounds[(i,)] is a bare (2,) row; GMM.fit
            # needs (n_dims,2).  Same up-shape guard as _sample()/q-scoring.
            # (Latent until now: the n_comp=None bug meant this line was never
            # reached in the default portfolio configuration.)
            if len(new_bounds.shape) < 2:
                new_bounds = self.xpy.array([new_bounds])
            model = self.integrator.gmm_dict[dim_group]
            temp_samples = self.xpy.empty((n_history_to_use, len(dim_group)))
            index = 0
            for dim in dim_group:
                # keep on the active backend: temp_samples and sample_array are
                # both self.xpy arrays, and the GMM model.fit/update below runs on
                # self.xpy.  (The old identity_convert here forced a host array
                # into a cupy column -> the same fill error as above on GPU.)
                temp_samples[:,index] = sample_array[:,dim]
                index += 1

            # Drop NaN-weight samples before fitting.  NOTE: filter into LOOP-LOCAL names.  This used
            # to reassign `ln_weights` itself, which is loop-INVARIANT (built once, before the loop
            # over dim_groups): the first group with any NaN shrank it (e.g. 10000 -> 8686), and every
            # LATER group then rebuilt temp_samples at full n_history_to_use but reused the stale,
            # shorter weights -> "boolean index did not match indexed array" inside GMM.update /
            # GMM.fit.  Only reachable when weights actually contain NaN, i.e. a degenerate/cold pass,
            # which is why warm runs never hit it and cold portfolio starts died on chunk ~8.
            if self.xpy.any(self.xpy.isnan(ln_weights)):
                ok_indx = ~self.xpy.isnan(ln_weights)
                temp_samples = temp_samples[ok_indx]      # rebuilt each iteration: safe to filter
                ln_weights_group = ln_weights[ok_indx]    # loop-LOCAL: never touch ln_weights itself
            else:
                ln_weights_group = ln_weights
            
            # Data-driven component count (matches integrator._train): scalar or
            # per-group gmm_adaptive picks k by BIC at init, floored at the
            # stress-tested n_comp, then the merge path below adapts.  This is the
            # path the PORTFOLIO drives its GMM member through (update_sampling_prior).
            adaptive_kmax = None
            _ga = getattr(self.integrator, 'gmm_adaptive', None)
            if _ga:
                if isinstance(_ga, dict):
                    adaptive_kmax = _ga.get(dim_group)
                elif isinstance(_ga, bool):
                    adaptive_kmax = 8
                else:
                    adaptive_kmax = int(_ga)
            if model is None:
                if adaptive_kmax:
                    if isinstance(self.integrator.n_comp, dict):
                        k_floor = self.integrator.n_comp.get(dim_group, 1)
                    else:
                        k_floor = self.integrator.n_comp
                    k_floor = int(k_floor) if isinstance(k_floor, int) and k_floor > 0 else 1
                    model = GMM.fit_gmm_adaptive(temp_samples, new_bounds,
                                                 log_sample_weights=ln_weights_group,
                                                 k_max=max(int(adaptive_kmax), k_floor),
                                                 k_min=k_floor,
                                                 epsilon=self.integrator.gmm_epsilon,
                                                 defensive_frac=getattr(self.integrator,'gmm_defensive_frac',0.0),
                                                 inflate=getattr(self.integrator,'gmm_inflate',1.0))
                elif isinstance(self.integrator.n_comp, int) and self.integrator.n_comp != 0:
                    model = GMM.gmm(self.integrator.n_comp, new_bounds,epsilon=self.integrator.gmm_epsilon)
                    model.fit(temp_samples, log_sample_weights=ln_weights_group)
                    # The defensive component is the ONLY thing that actually guarantees this member
                    # has support across the box -- gmm.score() merely FLOORS at 1e-300, which is a
                    # numerical guard, not coverage (a sample there would carry weight ~1e300).
                    # fit_gmm_adaptive adds it; the fixed-component path did not.  OPT-IN, because
                    # measured on the shape gate a 5% broad component costs real n_eff in
                    # higher dimensions (d6_n3_s303 119->75, d8_n1_s303 448->210): it spends
                    # 5% of draws where the likelihood is negligible.  Only a consumer that
                    # NEEDS this member as its coverage guarantee should pay -- so a
                    # portfolio sets gmm_defensive_all_paths on its members, and a standalone
                    # GMM user is unaffected.
                    GMM.add_defensive_component(model, defensive_frac=(
                        getattr(self.integrator,'gmm_defensive_frac',0.0)
                        if getattr(self.integrator,'gmm_defensive_all_paths',False) else 0.0))
                elif isinstance(self.integrator.n_comp, dict) and self.integrator.n_comp[dim_group] != 0:
                    model = GMM.gmm(self.integrator.n_comp[dim_group], new_bounds,epsilon=self.integrator.gmm_epsilon)
                    model.fit(temp_samples, log_sample_weights=ln_weights_group)
                    # The defensive component is the ONLY thing that actually guarantees this member
                    # has support across the box -- gmm.score() merely FLOORS at 1e-300, which is a
                    # numerical guard, not coverage (a sample there would carry weight ~1e300).
                    # fit_gmm_adaptive adds it; the fixed-component path did not.  OPT-IN, because
                    # measured on the shape gate a 5% broad component costs real n_eff in
                    # higher dimensions (d6_n3_s303 119->75, d8_n1_s303 448->210): it spends
                    # 5% of draws where the likelihood is negligible.  Only a consumer that
                    # NEEDS this member as its coverage guarantee should pay -- so a
                    # portfolio sets gmm_defensive_all_paths on its members, and a standalone
                    # GMM user is unaffected.
                    GMM.add_defensive_component(model, defensive_frac=(
                        getattr(self.integrator,'gmm_defensive_frac',0.0)
                        if getattr(self.integrator,'gmm_defensive_all_paths',False) else 0.0))
                elif not (self.integrator.n_comp == 0 or
                          (isinstance(self.integrator.n_comp, dict) and self.integrator.n_comp.get(dim_group) == 0)):
                    # invalid n_comp (e.g. None from an integrator built outside
                    # setup()): never no-op silently -- that hid a dead GMM
                    # portfolio member in production.  n_comp==0 is the only
                    # sanctioned way to skip training.
                    if not getattr(self, '_warned_invalid_n_comp', False):
                        self._warned_invalid_n_comp = True
                        print(" mcsamplerEnsemble: update_sampling_prior SKIPPING training for dim_group {}: "
                              "invalid n_comp {!r} (use n_comp=0 to disable adaptation intentionally)".format(
                                  dim_group, self.integrator.n_comp))
            else:
                model.update(temp_samples, log_sample_weights=ln_weights_group)
            self.integrator.gmm_dict[dim_group] = model

    def bootstrap_from_samples(self, samples, params=None, n_comp_warm=2, **kwargs):
        """Warm-start: fit this GMM's proposal to a seed cloud so it samples AT the peak from
        the first draw, instead of having to discover the peak location from scratch.  For a
        needle-in-a-haystack extrinsic posterior (peak ~ 10^-11 of the prior box) this is the
        difference between converging and never finding the peak by cold draws.

        Builds the integrator if setup() has not run yet, then (re)fits one GMM per dim-group
        from the seed with EQUAL weights -- this only shapes the proposal, carries no lnL
        information, so it cannot bias the estimate (the importance weights still use the true
        likelihood).  AV-only kwargs (cover_frac / inflate) are accepted and ignored, so a
        portfolio can forward a single seed to every member uniformly.

        `samples`: (N, ndim) array, columns in self.params_ordered order."""
        samples = np.asarray(self.identity_convert(samples))
        ndim = len(self.params_ordered)
        if samples.ndim != 2 or samples.shape[1] != ndim:
            raise ValueError("GMM warm-start expects (N,{}) samples in params_ordered order".format(ndim))
        # (Re)build the integrator as a SINGLE full-dimensional Gaussian group.  Two reasons:
        #  * modeling -- a localized high-SNR peak has strong cross-parameter correlations
        #    (sky<->phase<->distance); one full-dim mixture captures them, whereas the default
        #    per-dimension factored proposal cannot and "can stall at the prior".
        #  * robustness -- the default gmm_dict=None path leaves integrator.bounds as a raw
        #    array (not a per-group dict), which breaks the per-group fit; the correlate-all
        #    path builds proper dict bounds.
        n_comp = n_comp_warm
        if (self.integrator is not None and isinstance(self.integrator.n_comp, int)
                and self.integrator.n_comp > 0):
            n_comp = self.integrator.n_comp
        # CARRY THE COVERAGE CONFIG THROUGH.  setup() rebuilds the integrator from its kwargs,
        # so calling it bare here reset gmm_defensive_all_paths to False and refitted the warm
        # GMM with NO defensive component -- after the portfolio had already decided, on the
        # strength of that flag, that it was safe to contract its AV member.  The guarantee has
        # to survive the sampler's own lifecycle, not just its initial setup.
        _prev = getattr(self, 'integrator', None)
        # gmm_dict=None EXPLICITLY.  setup() now remembers its kwargs, so a remembered explicit
        # grouping would survive this call and correlate_all_dims=True would have no effect --
        # defeating the single full-dimensional group this path exists to build (the whole point
        # is to capture sky<->phase<->distance correlations at high SNR).  Passing None overrides
        # the remembered value; the coverage settings below are still carried forward.
        self.setup(n_comp=int(n_comp), correlate_all_dims=True, gmm_dict=None,
                   gmm_defensive_frac=getattr(_prev, 'gmm_defensive_frac', 0.05),
                   gmm_defensive_all_paths=getattr(_prev, 'gmm_defensive_all_paths', False))
        rvs = {p: samples[:, j] for j, p in enumerate(self.params_ordered)}
        # equal weights == "put proposal mass at these seed locations" (no lnL info)
        self.update_sampling_prior(self.xpy.zeros(len(samples)), len(samples),
                                   external_rvs=rvs, log_scale_weights=True)
        print("  [GMM warm-start] fitted full-dim proposal to {} seed samples (n_comp={})".format(
            len(samples), n_comp))
        return True

    def draw_simplified(self,n,*args,**kwargs):
        n_samples = int(n)
        self.integrator.n = n

        if len(args) == 0:
            args = self.params
        n_params = int(len(args))

        save_no_samples= False
        if 'save_no_samples' in list(kwargs.keys()):
            save_no_samples = kwargs['save_no_samples']

        rv = self.xpy.empty((n_params, n_samples), dtype=np.float64)
        joint_p_s = self.xpy.ones(n_samples, dtype=np.float64)
        joint_p_prior = self.xpy.ones(n_samples, dtype=np.float64)

        self.integrator._sample()
        for indx, p in enumerate(self.params_ordered):
            if isinstance(type(rv), type(self.integrator.sample_array)):
                rv[indx,:]  = self.integrator.sample_array[:,indx]
            else:
                rv[indx,:]  = self.identity_convert_togpu(self.integrator.sample_array[:,indx])
        joint_p_s = self.integrator.sampling_prior_array
        joint_p_prior = self.calc_pdf(rv.T).flatten()

        return joint_p_s, joint_p_prior, rv

    def sampling_density(self, X):
        """Pointwise sampling density q(theta) of THIS GMM member, evaluated at
        ARBITRARY points X (shape (N, ndim), columns in self.params_ordered
        order).  Returns a host (numpy) array of length N, or None if the
        integrator/GMM has not been built yet.

        This is exactly the per-sample product MonteCarloEnsemble._sample stores
        as sampling_prior_array, but evaluated at supplied points rather than at
        the member's own draws: for each grouped set of dimensions it is the
        fitted mixture density gmm.score(...) (already normalized to integrate to
        1 over the box, in ORIGINAL coordinates), or the uniform density 1/vol
        for a not-yet-fitted (None) group.  READ-ONLY; does not affect this
        sampler's own integrate().  Used by the portfolio balance heuristic.
        """
        integrator = getattr(self, 'integrator', None)
        if integrator is None:
            return None
        Xc = np.atleast_2d(np.asarray(self.identity_convert(X), dtype=float))
        ndim = len(self.params_ordered)
        if Xc.shape[1] != ndim and Xc.shape[0] == ndim:
            Xc = Xc.T  # tolerate (ndim, N)
        Xg = self.identity_convert_togpu(Xc)
        q = self.xpy.ones(Xg.shape[0])
        for dim_group in integrator.gmm_dict:
            new_bounds = integrator.bounds[dim_group]
            if len(new_bounds.shape) < 2:
                new_bounds = self.xpy.array([new_bounds])
            model = integrator.gmm_dict[dim_group]
            cols = self.xpy.empty((Xg.shape[0], len(dim_group)))
            for index, dim in enumerate(dim_group):
                cols[:, index] = Xg[:, dim]
            if model is None:
                llim = new_bounds[:, 0]
                rlim = new_bounds[:, 1]
                vol = self.xpy.prod(rlim - llim)
                q *= 1.0 / vol
            else:
                q *= model.score(cols)
        return self.identity_convert(q)


    def integrate_log(self, func, *args,**kwargs):
        args_passed = {}
        args_passed.update(kwargs)
        args_passed['use_lnL']=True
        args_passed['return_lnI']=True
        return self.integrate(func, *args, **args_passed)

    def integrate(self, func, *args,**kwargs):
        nmax = kwargs["nmax"] if "nmax" in kwargs else 1e6
        neff = kwargs["neff"] if "neff" in kwargs else 1000
        n = kwargs["n"] if "n" in kwargs else min(1000, nmax)
        n_comp = kwargs["n_comp"] if "n_comp" in kwargs else 1
        if 'gmm_dict' in list(kwargs.keys()):
            gmm_dict = kwargs['gmm_dict']
        else:
            gmm_dict = None
        reflect = kwargs['reflect'] if "reflect" in kwargs else False
        integrator_func  = kwargs['integrator_func'] if "integrator_func" in kwargs  else None
        mcsamp_func  = kwargs['mcsamp_func'] if "mcsamp_func" in kwargs  else None
        proc_count = kwargs['proc_count'] if "proc_count" in kwargs else None
        direct_eval = kwargs['direct_eval'] if "direct_eval" in kwargs else False
        min_iter = kwargs['min_iter'] if "min_iter" in kwargs else 10
        max_iter = kwargs['max_iter'] if "max_iter" in kwargs else 20
        var_thresh = kwargs['var_thres'] if "var_thresh" in kwargs else 0.05
        write_to_file = kwargs['write_to_file'] if "write_to_file" in kwargs else False
        correlate_all_dims = kwargs['correlate_all_dims'] if  "correlate_all_dims" in kwargs else False
        gmm_adapt = kwargs['gmm_adapt'] if "gmm_adapt" in kwargs else None
        gmm_adaptive = kwargs['gmm_adaptive'] if "gmm_adaptive" in kwargs else None
        gmm_defensive_frac = kwargs['gmm_defensive_frac'] if "gmm_defensive_frac" in kwargs else 0.05
        gmm_inflate = kwargs['gmm_inflate'] if "gmm_inflate" in kwargs else 1.0
        gmm_epsilon = kwargs['gmm_epsilon'] if "gmm_epsilon" in kwargs else None
        L_cutoff = kwargs["L_cutoff"] if "L_cutoff" in kwargs else None
        tempering_exp = kwargs["tempering_exp"] if "tempering_exp" in kwargs else 1.0
        # --adapt-adapt: ESS-self-tuned refit exponent (previously silently
        # dropped by this sampler; only mcsampler/mcsamplerGPU honored it)
        tempering_adapt = kwargs["tempering_adapt"] if "tempering_adapt" in kwargs else False
        ess_target = kwargs["ess_target"] if "ess_target" in kwargs else None
        ess_floor = kwargs["ess_floor"] if "ess_floor" in kwargs else None
        lnw_failure_cut = kwargs["lnw_failure_cut"] if "lnw_failure_cut" in kwargs else None

        max_err = kwargs["max_err"] if "max_err" in kwargs else 10

        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        super_verbose = kwargs["super_verbose"] if "super_verbose" in kwargs else False
        dict_return_q = kwargs["dict_return"] if "dict_return" in kwargs else False

        tripwire_fraction = kwargs["tripwire_fraction"] if "tripwire_fraction" in kwargs else 2
        tripwire_epsilon = kwargs["tripwire_epsilon"] if "tripwire_epsilon" in kwargs else 0.001

        use_lnL = kwargs["use_lnL"] if "use_lnL" in kwargs else False 
        return_lnI = kwargs["return_lnI"] if "return_lnI" in kwargs else False

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

        self.func = func
        self.curr_args = args
        if n_comp is None:
            print('No n_comp given, assuming 1 component per dimension')
            n_comp = 1
        dim = len(args)
        bounds=[]
        for param in args:
            bounds.append([self.llim[param], self.rlim[param]])
        raw_bounds = self.xpy.array(bounds)

        bounds=None
        if gmm_dict is None:
            # NOTE: dim-group / bounds dict keys must be *host* integers. Building
            # them with self.xpy.arange would produce unhashable 0-d cupy arrays
            # on GPU; keep this bookkeeping on the CPU with range/np.arange.
            bounds = {}
            for indx in np.arange(len(raw_bounds)):
                bounds[(indx,)] = raw_bounds[indx]
            bounds=raw_bounds
            if correlate_all_dims:
                gmm_dict = {tuple(range(dim)):None}
                bounds = {tuple(range(dim)): raw_bounds}
            else:
                gmm_dict = {}
                for i in range(dim):
                    gmm_dict[(i,)] = None
        else:
            bounds ={}
            for dims in gmm_dict:
                n_dims = len(dims)
                bounds_here = self.xpy.empty((n_dims,2))
                for indx in range(n_dims):
                    bounds_here[indx] = raw_bounds[dims[indx]]
                bounds[dims]=bounds_here

        integrator = monte_carlo.integrator(dim, bounds, gmm_dict, n_comp, n=n, prior=self.calc_pdf,
                         user_func=integrator_func, proc_count=proc_count,L_cutoff=L_cutoff,gmm_adapt=gmm_adapt,gmm_epsilon=gmm_epsilon,tempering_exp=tempering_exp,
                         tempering_adapt=tempering_adapt, ess_target=ess_target, ess_floor=ess_floor, gmm_adaptive=gmm_adaptive,
                         gmm_defensive_frac=gmm_defensive_frac, gmm_inflate=gmm_inflate)
        # Warm-start survival: a prior setup()/bootstrap_from_samples fits proposal
        # models and stores them on self.integrator, but integrate() rebuilds a fresh
        # integrator from the passed gmm_dict (values None) -- so without this the
        # bootstrapped fit is SILENTLY DISCARDED and a "warm" run starts cold
        # (measured: warm correlate-all began at n_eff=1.0, climbed to only ~7 @4M).
        # Transfer any fitted model whose dim-group key matches; a key mismatch
        # (e.g. bootstrap built correlate-all but the run uses the factored pairing)
        # simply falls back to cold, so this can never bias or crash.
        prev = getattr(self, 'integrator', None)
        if prev is not None and prev is not integrator and getattr(prev, 'gmm_dict', None):
            n_xfer = 0
            for key, model in prev.gmm_dict.items():
                if model is not None and key in integrator.gmm_dict and integrator.gmm_dict[key] is None:
                    integrator.gmm_dict[key] = model
                    n_xfer += 1
            if n_xfer:
                print("  [GMM warm-start] transferred {} fitted proposal group(s) into the integrator".format(n_xfer))
        self.integrator = integrator
        if not direct_eval:
            func = self.evaluate
        if use_lnL:
            print(" ==> input assumed as lnL ")
        if return_lnI:
            print(" ==> internal calculations and return values are lnI ")
        integrator.integrate(func, min_iter=min_iter, max_iter=max_iter, var_thresh=var_thresh, neff=neff, nmax=nmax,max_err=max_err,verbose=verbose,progress=super_verbose,tripwire_fraction=tripwire_fraction,tripwire_epsilon=tripwire_epsilon,use_lnL=use_lnL,return_lnI=return_lnI,lnw_failure_cut=lnw_failure_cut)

        self.n = int(integrator.n)
        self.ntotal = int(integrator.ntotal)
        integral = integrator.integral
        print("Result ",integrator.scaled_error_squared, integrator.integral)
        if not(return_lnI):
            error_squared = integrator.scaled_error_squared * self.xpy.exp(integrator.log_error_scale_factor)/ (self.ntotal/self.n)
        else:
            error_squared = integrator.scaled_error_squared  - self.xpy.log(self.ntotal/self.n)
        eff_samp = integrator.eff_samp
        sample_array = integrator.cumulative_samples
        if not(return_lnI):
            value_array = self.xpy.exp(integrator.cumulative_values)
        else:
            value_array = integrator.cumulative_values
        p_array = integrator.cumulative_p_s
        prior_array = integrator.cumulative_p

        if mcsamp_func is not None:
            mcsamp_func(self, integrator)

        # Store sample history on the host so downstream (CPU) consumers --
        # weights, CDFs, posterior plots -- work regardless of backend.
        # A sampler can be reused across log- and linear-space integrations.
        # Remove mode-specific results from the previous run before exporting
        # this one so convergence checks cannot consume stale log weights.
        for key in (
            'log_integrand',
            'log_joint_prior',
            'log_joint_s_prior',
            'log_weights',
        ):
            self._rvs.pop(key, None)
        index = 0
        for param in args:
            self._rvs[param] = self.identity_convert(sample_array[:,index])
            index += 1
        self._rvs['joint_prior'] = self.identity_convert(prior_array)
        self._rvs['joint_s_prior'] = self.identity_convert(p_array)
        self._rvs['integrand'] = self.identity_convert(value_array)
        if use_lnL:
            # Preserve the historical ``integrand`` alias for direct callers,
            # while exposing an unambiguous log-space contract to downstream
            # consumers.  In log mode ``value_array`` is already ln(L).
            self._rvs['log_integrand'] = self.identity_convert(integrator.cumulative_values)
            self._rvs['log_joint_prior'] = self.identity_convert(self.xpy.log(prior_array))
            self._rvs['log_joint_s_prior'] = self.identity_convert(self.xpy.log(p_array))
            self._rvs['log_weights'] = self.identity_convert(
                integrator.cumulative_values
                + self.xpy.log(prior_array)
                - self.xpy.log(p_array)
            )

        # (DESIGN_rvs_naming.md) _rvs is the RETAINED set at this point -- pruned,
        # perhaps, but never resampled.  Record that before the draw below can change what it
        # means, so "not resampled" is a statement the record makes rather than the absence of
        # one.  The reserve rides along BY REFERENCE where the sampler keeps one (AV and the
        # portfolio); None elsewhere is the honest answer, not a gap.
        # THE return_lnI CASE.  What `integrand` holds is decided by return_lnI and by
        # NOTHING ELSE: value_array above is `cumulative_values` (always lnL, whichever
        # convention the CALLABLE used) when return_lnI, and exp() of it when not.  use_lnL
        # governs a different question -- whether the log columns were written alongside --
        # so recording it here would mislabel the supported return_lnI=True, use_lnL=False
        # pass as linear, sending its negative-lnL rows to zero weight and taking log() of a
        # log on the rest.  Recording the convention HERE, once, where it is known, is what
        # lets every consumer stop caring -- and lets return_lnI become historical material
        # rather than something a caller must thread through.
        self._rvs_record = RvsRecord.retained(
            self._rvs, reserve=getattr(self, '_warm_seed_reserve', None),
                   integrand_is_log=bool(return_lnI))
        if bFairdraw and not(n_extr is None):
           # scalars: use Python min on floats.  self.xpy.min([list]) fails on cupy
           # (cupy.min has no list overload -> "'list' object has no attribute 'min'"),
           # which crashed the GMM sampler's fairdraw export on GPU.
           n_extr = int(min(float(n_extr), 1.5*float(eff_samp), 1.5*float(neff)))
           print(" Fairdraw size : ", n_extr)
           if return_lnI:
               ln_wt =  integrator.cumulative_values
           else:
               ln_wt = self.xpy.log(value_array)
           ln_wt += self.xpy.log(prior_array/p_array)
           ln_wt += - scipy.special.logsumexp(self.identity_convert(ln_wt))
           wt = self.xpy.exp(ln_wt)
           if n_extr < len(value_array):
               indx_list = self.identity_convert(self.xpy.random.choice(self.xpy.arange(len(wt)), size=n_extr,replace=True,p=wt))
               for key in list(self._rvs.keys()):
                   if isinstance(key, tuple):
                       self._rvs[key] = self._rvs[key][:,indx_list]
                   else:
                       self._rvs[key] = self._rvs[key][indx_list]

               self._rvs_is_fairdraw = True   # _rvs is now an EXPORT resample, rows already ~ w
               # ...and now it is an export resample.  n_retained comes from that record's
                # PROVENANCE, which captured the count eagerly -- NOT from len(), which reads
                # self._rvs and would return the POST-draw length: the retained record holds a
                # REFERENCE to the live dict this block has just replaced in place.  That is
                # this project's own bug class, so it is spelled out rather than assumed.
               self._rvs_record = RvsRecord.fair_draw(
                   self._rvs, n_retained=self._rvs_record.n_retained(),
                   reserve=getattr(self, '_warm_seed_reserve', None),
                   integrand_is_log=bool(return_lnI))
        dict_return = {}
        if dict_return_q:
            dict_return["integrator"] = integrator

        if write_to_file:
            dat_out = self.xpy.c_[sample_array, value_array, p_array]
            np.savetxt('mcsampler_data.txt', self.identity_convert(dat_out),
                        header=" ".join(['sample_array', 'value_array', 'p_array']))

        # Return scalars on the host so callers can do plain numpy arithmetic
        # (np.sqrt, np.log, np.array([...])) on the results.
        return self.identity_convert(integral), self.identity_convert(error_squared), self.identity_convert(eff_samp), dict_return


def inv_uniform_cdf(a, b, x):
    return (b-a)*x+a

def gauss_samp(mu, std, x):
    return 1.0/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mu)**2/2/std**2)

def gauss_samp_withfloor(mu, std, myfloor, x):
    return 1.0/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mu)**2/2/std**2) + myfloor

gauss_samp_withfloor_vector = np.vectorize(gauss_samp_withfloor,otypes=[np.float64])


def q_samp_vector(qmin,qmax,x):
    scale = 1./(1+qmin) - 1./(1+qmax)
    return 1/np.power((1+x),2)/scale
def q_cdf_inv_vector(qmin,qmax,x):
    return np.array((qmin + qmax*qmin + qmax*x - qmin*x)/(1 + qmax - qmax*x + qmin*x),dtype=RiftFloat)

def M_samp_vector(Mmin,Mmax,x):
    scale = 2./(Mmax**2 - Mmin**2)
    return x*scale


def cos_samp(x):
        return np.sin(x)/2

def dec_samp(x):
        return np.sin(x+np.pi/2)/2

cos_samp_vector = np.vectorize(cos_samp,otypes=[np.float64])
dec_samp_vector = np.vectorize(dec_samp,otypes=[np.float64])
def cos_samp_cdf_inv_vector(p):
    return np.arccos( 2*p-1)
def dec_samp_cdf_inv_vector(p):
    return np.arccos(2*p-1) - np.pi/2


def pseudo_dist_samp(r0,r):
        return r*r*np.exp( - (r0/r)*(r0/r)/2. + r0/r)+0.01

pseudo_dist_samp_vector = np.vectorize(pseudo_dist_samp,otypes=[np.float64])

def delta_func_pdf(x_0, x):
    return 1.0 if x == x_0 else 0.0

delta_func_pdf_vector = np.vectorize(delta_func_pdf, otypes=[np.float64])

def delta_func_samp(x_0, x):
    return x_0

delta_func_samp_vector = np.vectorize(delta_func_samp, otypes=[np.float64])

class HealPixSampler(object):
    @staticmethod
    def thph2decra(th, ph):
        return np.pi/2-th, ph

    @staticmethod
    def decra2thph(dec, ra):
        return np.pi/2-dec, ra

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
        res = healpy.npix2nside(len(self.skymap))
        self.pdf_sorted = sorted([(p, i) for i, p in enumerate(self.skymap)], reverse=True)
        self.valid_points_decra = []
        cdf, np_count = 0, 0
        for p, i in self.pdf_sorted:
            if p == 0:
                continue
            self.valid_points_decra.append(HealPixSampler.thph2decra(*healpy.pix2ang(res, i)))
            cdf += p
            if cdf > self._massp:
                break
        self._renorm = cdf
        self.valid_points_hist = None
        return self._renorm

    def __expand_valid(self, min_p=1e-7):
        if self._massp == 1.0:
            min_p = min(min_p, max(self.skymap))
        else:
            min_p = self.pseudo_pdf(*self.valid_points_decra[-1])

        self.valid_points_hist = []
        ns = healpy.npix2nside(len(self.skymap))

        self._renorm = 0
        for i, v in enumerate(self.skymap >= min_p):
            self._renorm += self.skymap[i] if v else 0

        for pt in self.valid_points_decra:
            th, ph = HealPixSampler.decra2thph(pt[0], pt[1])
            pix = healpy.ang2pix(ns, th, ph)
            if self.skymap[pix] < min_p:
                continue
            self.valid_points_hist.extend([pt]*int(round(self.pseudo_pdf(*pt)/min_p)))
        self.valid_points_hist = np.array(self.valid_points_hist).T

    def pseudo_pdf(self, dec_in, ra_in):
        th, ph = HealPixSampler.decra2thph(dec_in, ra_in)
        res = healpy.npix2nside(len(self.skymap))
        return self.skymap[healpy.ang2pix(res, th, ph)]/self._renorm

    def pseudo_cdf_inverse(self, dec_in=None, ra_in=None, ndraws=1, stype='vecthist'):
        if ra_in is not None:
            ndraws = len(ra_in)
        if ra_in is None:
            ra_in, dec_in = np.zeros((2, ndraws))

        if stype == 'rejsamp':
            ceiling = max(self.skymap)
            i, np_count = 0, len(self.valid_points_decra)
            while i < len(ra_in):
                rnd_n = np.random.randint(0, np_count)
                trial = np.random.uniform(0, ceiling)
                if trial <= self.pseudo_pdf(*self.valid_points_decra[rnd_n]):
                    dec_in[i], ra_in[i] = self.valid_points_decra[rnd_n]
                    i += 1
            return np.array([dec_in, ra_in])
        elif stype == 'vecthist':
            if self.valid_points_hist is None:
                self.__expand_valid()
            np_count = self.valid_points_hist.shape[1]
            rnd_n = np.random.randint(0, np_count, len(ra_in))
            dec_in, ra_in = self.valid_points_hist[:,rnd_n]
            return np.array([dec_in, ra_in])
        else:
            raise ValueError("%s is not a recgonized sampling type" % stype)

pseudo_dist_samp_vector = np.vectorize(pseudo_dist_samp,otypes=[np.float64])


def sanityCheckSamplerIntegrateUnity(sampler,*args,**kwargs):
        return sampler.integrate(lambda *args: 1,*args,**kwargs)

def convergence_test_MostSignificantPoint(pcut, rvs, params):
    if "log_weights" in rvs:
        log_weights = np.asarray(rvs["log_weights"])
        return np.exp(np.max(log_weights) - scipy.special.logsumexp(log_weights)) < pcut
    weights = rvs.get("weights")
    if weights is None:
        weights = rvs["integrand"]*rvs["joint_prior"]/rvs["joint_s_prior"]
    indxmax = np.argmax(weights)
    wtSum = np.sum(weights)
    return  weights[indxmax]/wtSum < pcut

def convergence_test_NormalSubIntegrals(ncopies, pcutNormalTest, sigmaCutRelativeErrorThreshold, rvs, params):
    igrandValues = np.zeros(ncopies)
    if "log_weights" in rvs:
        log_weights = np.asarray(rvs["log_weights"])
        len_part = int(len(log_weights)/ncopies)
        for indx in np.arange(ncopies):
            log_weights_here = log_weights[indx*len_part:(indx+1)*len_part]
            igrandValues[indx] = scipy.special.logsumexp(log_weights_here) - np.log(len(log_weights_here))
    else:
        weights = rvs["integrand"]*rvs["joint_prior"]/rvs["joint_s_prior"]
        len_part = int(len(weights)/ncopies)
        for indx in np.arange(ncopies):
            igrandValues[indx] = np.log(np.mean(weights[indx*len_part:(indx+1)*len_part]))
    igrandValues= np.sort(igrandValues)
    valTest = stats.normaltest(igrandValues)[1]
    igrandSigma = (np.std(igrandValues))/np.sqrt(ncopies)
    print(" Test values on distribution of log evidence:  (gaussianity p-value; standard deviation of ln evidence) ", valTest, igrandSigma)
    print(" Ln(evidence) sub-integral values, as used in tests  : ", igrandValues)
    return valTest> pcutNormalTest and igrandSigma < sigmaCutRelativeErrorThreshold

from . import gaussian_mixture_model as GMM
def create_wide_single_component_prior(bounds, epsilon=None):
    model = GMM.gmm(1, bounds, epsilon=epsilon)
    widths = np.array([ bounds[k][1] - bounds[k][0] for k in np.arange(len(bounds))])  
    model.means = [np.array([np.mean(bounds[k]) for k in np.arange(len(bounds))]) ]
    model.covariances = [np.diag( widths**2)]
    model.weights = [1]
    model.adapt = [False]
    model.d = len(bounds)
