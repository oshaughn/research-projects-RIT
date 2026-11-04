# -*- coding: utf-8 -*-
'''
Monte Carlo Integrator
----------------------
Perform an adaptive monte carlo integral.
'''
from __future__ import print_function
import warnings
import numpy as np
from . import gaussian_mixture_model as GMM
import traceback
import time
from scipy.special import logsumexp

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

regularize_log_scale = 1e-64  # before taking np.log, add this, so we don't propagate infinities


def _xpy_logsumexp(a, axis=None):
    """Portable logsumexp (mirror of gaussian_mixture_model._xpy_logsumexp).

    cupyx.scipy.special.logsumexp is absent in the CUDA 10.2 cupy build needed
    for older (sm_30) GPUs, so implement the reduction with cupy primitives and
    fall back to scipy on CPU.
    """
    if cupy_ok:
        a = cupy.asarray(a)
        a_max = cupy.amax(a, axis=axis, keepdims=True)
        a_max = cupy.where(cupy.isfinite(a_max), a_max, cupy.zeros_like(a_max))
        out = cupy.log(cupy.sum(cupy.exp(a - a_max), axis=axis, keepdims=True)) + a_max
        if axis is None:
            return out.reshape(())
        return cupy.squeeze(out, axis=axis)
    return logsumexp(a, axis=axis)


try:
    from multiprocess import Pool
except:
    print('no multiprocess')


def validate_gmm_dict(bounds, gmm_dict, where="gmm_dict", param_names=None):
    """Reject a seeded proposal that does not describe the dim-group it is installed against.

    _sample() draws from model.sample() and divides the weights by model.score(), the model's
    own normalized density over its OWN box.  Two things therefore have to agree with the
    group, and neither is checked anywhere else:

      * DIMENSION.  A model with more axes than its group contributes a density carrying the
        extra axes, so lnZ comes out high by their log-volume -- +1.84 nats for a 2-D
        (psi, phi_orb) model against the 1-element (psi,) group the ILE drivers use when
        phase is marginalized.  Fewer axes raises IndexError on the first _sample().
      * BOUNDS.  A model normalized over a SMALLER box than the group's is never wrong about
        any sample it draws, and never draws outside its own box -- so the run silently
        explores a fraction of the group and lnZ is low by ln(area ratio).  Replaying an
        --extrinsic-proposal-breadcrumb recorded without --internal-rotate-phase into a run
        with it does exactly this: the option doubles psi and phi_orb to (0, 4pi), and the
        stored seed keeps (0, 2pi).  Measured: -1.39 nats, a quarter of the box reachable.

    Neither failure disturbs the recovered marginals, which keep their shape while only the
    evidence moves, so it has to be caught here or not at all.

    This is an equality check on the box, NOT an ordering check: two axes sharing a box
    (psi and phi_orb do) can be permuted without tripping it.  Axis order is
    gmm_dict_from_breadcrumb._permute_group's job.
    """
    if not gmm_dict:
        return

    def _label(grp):
        """Name the parameters, not just the dim indices: an operator reading a held job's
        log cannot map (4,) to psi without reading the driver."""
        if param_names:
            try:
                return "{} {}".format(tuple(param_names[i] for i in grp), tuple(grp))
            except Exception:
                pass
        return str(tuple(grp))

    for grp, model in gmm_dict.items():
        if model is None:
            continue
        md = getattr(model, 'd', None)
        if md is not None:
            # compare the VALUE: int() would silently accept 2.4 or '2'
            try:
                bad_dim = (md != len(grp))
            except Exception:
                bad_dim = True
            if bad_dim:
                raise ValueError(
                    "{}[{}]: seeded model has d={!r} for a {}-dimensional dim-group. "
                    "A seed must span exactly its group's axes: with more axes its score() "
                    "carries the extra ones and lnZ is biased HIGH by their log-volume; with "
                    "fewer, _sample() raises IndexError. Neither shows up in a marginal."
                    .format(where, _label(grp), md, len(grp)))
        mb = getattr(model, 'bounds', None)
        # `bounds` is a dict keyed by dim-group when the caller passed an explicit grouping,
        # and a flat (d,2) array otherwise -- which is what a bare setup() leaves behind, and
        # what the portfolio's GMM member runs on.  Reading only the dict form made this half
        # of the check a silent no-op on exactly that path.
        if isinstance(bounds, dict):
            gb = bounds.get(grp)
        elif bounds is not None:
            try:
                # _to_host_bounds FIRST: the flat form is built with self.xpy, which is cupy on
                # a GPU host, and np.asarray() of a device array raises.  Without the hop this
                # branch lands in the except on exactly the hosts the portfolio's GMM member
                # runs on, and the skip below is silent -- inert where it was written to work.
                gb = np.asarray(_to_host_bounds(bounds))[list(grp)]
            except Exception:
                gb = None
        else:
            gb = None
        if mb is None or gb is None:
            if mb is not None:
                warnings.warn(
                    "{}[{}]: no comparable bounds for this dim-group; dimension was checked, "
                    "the box was not.".format(where, _label(grp)), RuntimeWarning)
            continue
        try:
            mb_h = np.asarray(_to_host_bounds(mb), dtype=float).reshape(-1, 2)
            gb_h = np.asarray(_to_host_bounds(gb), dtype=float).reshape(-1, 2)
        except Exception:
            # Unrecognized bounds shape.  The dimension check above still applies, but say so:
            # the same swallow would otherwise hide a real device-conversion failure, and only
            # on the hosts that have a device.
            warnings.warn(
                "{}[{}]: could not compare bounds ({!r} vs {!r}); dimension was checked, the "
                "box was not.".format(where, _label(grp), type(mb).__name__, type(gb).__name__),
                RuntimeWarning)
            continue
        # rtol is deliberately loose: a box mismatch worth catching is a factor of two or a
        # narrowing, never 1e-7.  A tight rtol only turns a float32 round-trip into a
        # failure an operator has to count digits to understand.
        if mb_h.shape != gb_h.shape or not np.allclose(mb_h, gb_h, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "{}[{}]: seeded model is normalized over {} but the dim-group's box is {}. "
                "A seed drawn and scored on a different box explores only the overlap and "
                "shifts lnZ by ln(area ratio), with every marginal still the right shape. "
                "Re-fit the seed on this run's bounds, or drop it."
                .format(where, _label(grp), mb_h.tolist(), gb_h.tolist()))


def _to_host_bounds(b):
    """numpy view of a bounds array that may be a list, an ndarray, or on a device."""
    get = getattr(b, 'get', None)
    return get() if (get is not None and not isinstance(b, (list, tuple, dict))) else b


def _validate_dim_group_cover(gmm_dict, d):
    """Every integration dimension must belong to exactly one dim-group key of gmm_dict.

    _sample() allocates sample_array with xpy.empty and writes only the columns named by a
    dim-group.  A dimension missing from gmm_dict therefore carries UNINITIALIZED MEMORY into
    the integrand and into the prior, with no matching factor in sampling_prior_array -- an
    unnormalized importance-sampling estimator whose ln Z is arbitrarily wrong while its
    reported sigma stays small.  A repeated dimension is double-counted the same way.  Neither
    is recoverable downstream, so refuse the integrator instead of returning a confident wrong
    number.
    """
    if gmm_dict is None:
        raise ValueError("gmm_dict is None: mcsamplerEnsemble needs a dim-group mapping covering all {} dimensions".format(d))
    seen = {}
    empty_keys = []
    for key in list(gmm_dict):
        if len(tuple(key)) == 0:
            empty_keys.append(key)
            continue
        for i in tuple(key):
            i = int(i)
            if not (0 <= i < d):
                raise ValueError(
                    "gmm_dict dim-group {} names dimension {}, outside the {} integration dimensions. "
                    "Dim-group keys index the sampler's POSITIONAL ARGUMENT order, not params_ordered.".format(
                        tuple(key), i, d))
            if i in seen:
                raise ValueError(
                    "gmm_dict dimension {} appears in both {} and {}; each dimension must be in exactly one group".format(
                        i, seen[i], tuple(key)))
            seen[i] = tuple(key)
    # An empty key is NOT inert, despite naming no dimension.  _sample() tolerates it (it draws
    # an (n, 0) block and multiplies the sampling density by prod([]) == 1), but _train() also
    # iterates gmm_dict and tries to FIT a mixture to that zero-width block: LAPACK raises, the
    # whole proposal is _reset(), and after enough consecutive chunks integrate() gives up with
    # "GMM proposal refit failed 5 consecutive times".  Measured on a sharp 3-d target with CIP's
    # call shape (int n_comp, no gmm_adapt): 4 proposal resets, on this base AND before it.
    #
    # CIP manufactures one from a typo: parse_corr_params swallows an unknown parameter name with
    # a bare `except: continue`, so an --internal-correlate-parameters block naming only unknown
    # parameters collapses to ().  Refusing the run is too harsh (the remaining groups still cover
    # every dimension) and warning alone leaves the training failures in place, so DROP the key
    # and say so.  This mutates the caller's dict on purpose: it is the only way to keep _train
    # from seeing it, and every consumer reads the group list from here.
    for key in empty_keys:
        del gmm_dict[key]
        warnings.warn(
            "gmm_dict had an empty dim-group key; it names no dimension and has been dropped. "
            "It would otherwise make the proposal refit fail and reset. This usually means a "
            "correlate-parameters block named only parameters that are not being sampled.",
            RuntimeWarning)
    missing = sorted(set(range(d)) - set(seen))
    if missing:
        raise ValueError(
            "gmm_dict covers dimensions {} but the integral has {} dimensions; {} are in no dim-group. "
            "Uncovered dimensions are never sampled (left as uninitialized memory) and make ln Z "
            "meaningless.  Dim-group keys index the sampler's POSITIONAL ARGUMENT order.".format(
                sorted(seen), d, missing))


class integrator:
    '''
    Class to iteratively perform an adaptive Monte Carlo integral where the integrand
    is a combination of one or more Gaussian curves, in one or more dimensions.

    Parameters
    ----------
    d : int
        Total number of dimensions.

    bounds : dictionary with array bounds, with keys matching gmm_dict
        Limits of integration, where each row represents [left_lim, right_lim]
        for its corresponding dimension.

    gmm_dict : dict
        Dictionary where each key is a tuple of one or more dimensions
        that are to be modeled together. If the integrand has strong correlations between
        two or more dimensions, they should be grouped. Each value is by default initialized
        to None, and is replaced with the GMM object for its dimension(s).

    n_comp : int or {tuple:int}
        The number of Gaussian components per group of dimensions. If its type is int,
        this number of components is used for all dimensions. If it is a dict, it maps
        each key in gmm_dict to an integer number of mixture model components.

    n : int
        Number of samples per iteration

    prior : function
        Function to evaluate prior for samples

    user_func : function
        Function to run each iteration

    L_cutoff : float
        Likelihood cutoff for samples to store

    use_lnL : bool
        Whether or not lnL or L will be returned by the integrand
    '''

    def __init__(self, d, bounds, gmm_dict, n_comp, n=None, prior=None,
                user_func=None, proc_count=None, L_cutoff=None, use_lnL=False,return_lnI=False,gmm_adapt=None,gmm_epsilon=None,tempering_exp=1,temper_log=False,lnw_failure_cut=None,
                tempering_adapt=False, ess_target=None, ess_floor=None, gmm_adaptive=None,
                gmm_defensive_frac=0.05, gmm_inflate=1.0, param_names=None):
        # if 'return_lnI' is active, 'integral' holds the *logarithm* of the integral.
        # user-specified parameters
        self.d = d
        self.bounds = bounds
        # Order matters: validate_gmm_dict reports a SEED that does not match its group, which is
        # the more specific complaint, and a caller may hand a deliberately partial gmm_dict while
        # exercising it.  The cover check is the blunter one, so it runs second.
        validate_gmm_dict(bounds, gmm_dict, param_names=param_names)
        _validate_dim_group_cover(gmm_dict, d)
        self.gmm_dict = gmm_dict
        self.gmm_adapt = gmm_adapt
        # gmm_adaptive: {dim_group: k_max}.  Groups listed here choose their
        # component count from the data by BIC (GMM.fit_gmm_adaptive) at
        # initialization, then adapt via the stable merge path, instead of using
        # a fixed n_comp -- see _train.
        self.gmm_adaptive = gmm_adaptive
        # defensive tail coverage + covariance inflation for adaptive groups
        self.gmm_defensive_frac = gmm_defensive_frac
        # Opt-in: install the defensive component on the FIXED-COMPONENT fit paths too.
        # Off by default because it costs n_eff at d>=6; a portfolio that relies on this
        # member for coverage turns it on (mcsamplerPortfolio.setup).
        self.gmm_defensive_all_paths = False
        self.gmm_inflate = gmm_inflate
        self.gmm_epsilon= gmm_epsilon
        self.n_comp = n_comp
        self.user_func=user_func
        self.prior = prior
        self.proc_count = proc_count
        self.use_lnL = use_lnL
        self.return_lnI = return_lnI
        
        self.xpy = xpy_default
        self.identity_convert = identity_convert
        self.identity_convert_togpu = identity_convert_togpu
        
        # constants
        self.t = 0.02 # percent estimated error threshold
        if n is None:
            self.n = int(5000 * self.d) # number of samples per batch
        else:
            self.n = int(n)
        self.ntotal = 0
        # integrator object parameters
        self.sample_array = None
        self.value_array = None
        self.sampling_prior_array = None
        self.prior_array = None
        self.scaled_error_squared = 0.
        if self.return_lnI:
            self.scaled_error_squared = None
        if not(lnw_failure_cut):
            self.terrible_lnw_threshold = -1000
        else:
            self.terrible_lnw_threshold = lnw_failure_cut
        self.log_error_scale_factor = 0.
        self.integral = 0
        if self.return_lnI:
            self.integral=None
        self.eff_samp = 0
        self.iterations = 0 # for weighted averages and count
        self.log_scale_factor = 0  # to handle very large answers
        self.max_value = float('-inf') # for calculating eff_samp
        self.total_value = 0 # for calculating eff_samp
        if self.return_lnI:
            self.total_value = None
        self.n_max = float('inf')
        # set to a descriptive string when integrate() exits abnormally (error
        # budget exhausted); None means a clean run.  Callers that cannot catch
        # the consecutive-refit-failure RuntimeError can inspect this instead.
        self.integration_error = None
        # saved values
        self.cumulative_samples = self.xpy.empty((0, d))
        self.cumulative_values = self.xpy.empty(0)
        self.cumulative_p = self.xpy.empty(0)
        self.cumulative_p_s = self.xpy.empty(0)
        self.tempering_exp=tempering_exp
        self.temper_log=temper_log
        # --- ESS-based tempering self-protection / self-tuning -------------
        # tempering_adapt: choose the refit exponent each chunk so the
        #   effective sample size of the refit weights hits ess_target
        #   (the user exponent becomes a cap, not a requirement).
        # Always on (any settings): if the user exponent would leave the
        #   refit with ESS < ess_floor, the exponent is clamped down for that
        #   refit only. The evidence integral never uses these weights.
        self.tempering_adapt = tempering_adapt
        # largest number of mixture components in any group (for the floor)
        if isinstance(n_comp, dict):
            _k_max = max([v for v in n_comp.values()]) if len(n_comp)>0 else 1
        else:
            _k_max = n_comp if n_comp else 1
        self.ess_floor = ess_floor if ess_floor else max(10.0, 2.0*_k_max)
        self.ess_target = ess_target if ess_target else max(50.0, 0.05*self.n)
        self.tempering_exp_running = tempering_exp  # last exponent actually used
        if L_cutoff is None:
            self.L_cutoff = -1
        else:
            self.L_cutoff = L_cutoff
        
    def _calculate_prior(self):
        if self.prior is None:
            self.prior_array = self.xpy.ones(self.n)
        else:
            self.prior_array = self.prior(self.sample_array).flatten()

    def _sample(self):
        self.sampling_prior_array = self.xpy.ones(self.n)
        self.sample_array = self.xpy.empty((self.n, self.d))
        for dim_group in self.gmm_dict: # iterate over grouped dimensions
            # create a matrix of the left and right limits for this set of dimensions
            new_bounds = self.xpy.empty((len(dim_group), 2))
            new_bounds = self.bounds[dim_group]
            if len(new_bounds.shape) < 2:
                new_bounds = self.xpy.array([new_bounds])
            model = self.gmm_dict[dim_group]
            if model is None:
                # sample uniformly for this group of dimensions
                llim = new_bounds[:,0]
                rlim = new_bounds[:,1]
                temp_samples = self.xpy.random.uniform(llim, rlim, (self.n, len(dim_group)))
                # update responsibilities
                vol = self.xpy.prod(rlim - llim)
                self.sampling_prior_array *= 1.0 / vol
            else:
                # sample from the gmm
                temp_samples = model.sample(self.n)
                # update responsibilities
                self.sampling_prior_array *= model.score(temp_samples)
            index = 0
            for dim in dim_group:
                self.sample_array[:,dim] = temp_samples[:,index]
                index += 1

    def _log_ess(self, log_w):
        """log of the Kish effective sample size of log-weights log_w."""
        return 2.*_xpy_logsumexp(log_w) - _xpy_logsumexp(2.*log_w)

    def _solve_tempering_exp(self, lnL, log_pq):
        """
        Choose the tempering exponent beta for THIS refit from the effective
        sample size of  beta*lnL + log_pq  (log_pq = ln p - ln p_s).

        - tempering_adapt: bisect so ESS(beta) ~ self.ess_target, with
          beta <= beta_max = max(1, tempering_exp).  As the proposal converges
          the lnL spread across the cloud shrinks and beta rises automatically.
        - otherwise: keep the user exponent unless ESS(user) < self.ess_floor,
          in which case bisect down to the floor (pure safety net; cannot
          crash the fit regardless of settings).
        Returns (beta, log_ess_at_beta).
        """
        ln_floor = self.xpy.log(self.ess_floor)
        ln_target = self.xpy.log(self.ess_target)
        beta_user = self.tempering_exp
        if self.tempering_adapt:
            beta_hi = max(1.0, beta_user)
            ln_goal = ln_target
        else:
            beta_hi = beta_user
            ln_goal = ln_floor
            if self._log_ess(beta_user*lnL + log_pq) >= ln_floor:
                return beta_user, self._log_ess(beta_user*lnL + log_pq)
        # ESS is (near-)monotone decreasing in beta for peaked lnL; bisect.
        if self._log_ess(beta_hi*lnL + log_pq) >= ln_goal:
            return beta_hi, self._log_ess(beta_hi*lnL + log_pq)
        lo, hi = 0.0, beta_hi
        for _ in range(40):
            mid = 0.5*(lo+hi)
            if self._log_ess(mid*lnL + log_pq) >= ln_goal:
                lo = mid
            else:
                hi = mid
        return lo, self._log_ess(lo*lnL + log_pq)

    def _train(self):
        sample_array, value_array, sampling_prior_array = self.xpy.copy(self.sample_array), self.xpy.copy(self.value_array), self.xpy.copy(self.sampling_prior_array)
        if self.use_lnL:
            lnL = value_array
        else:
            lnL = self.xpy.log(value_array+regularize_log_scale)

        # drop NaN evaluations up front (a NaN poisons every logsumexp below);
        # -inf is fine (zero weight) and is kept.
        prior_array = self.prior_array
        mask_ok = ~self.xpy.isnan(lnL)
        if not bool(self.xpy.all(mask_ok)):
            sample_array = sample_array[mask_ok]
            lnL = lnL[mask_ok]
            sampling_prior_array = sampling_prior_array[mask_ok]
            prior_array = prior_array[mask_ok]

        # replace -inf lnL (zero likelihood) by a finite very-low value:
        # beta=0 would otherwise produce 0*(-inf)=NaN in the tempered weights
        if not bool(self.xpy.all(self.xpy.isfinite(lnL))):
            lnL_min = self.xpy.min(self.xpy.where(self.xpy.isfinite(lnL), lnL, self.xpy.inf))
            if not bool(self.xpy.isfinite(lnL_min)):
                lnL_min = 0.0
            lnL = self.xpy.where(self.xpy.isfinite(lnL), lnL, lnL_min - 1000.)

        # ln p - ln p_s  (NOTE: log of the sampling prior. The legacy code
        # subtracted the *raw* sampling_prior_array from a log-quantity.)
        log_pq = self.xpy.log(self.xpy.maximum(prior_array, 1e-300)) \
                 - self.xpy.log(self.xpy.maximum(sampling_prior_array, 1e-300))

        # ESS-protected/self-tuned tempering exponent for this refit
        beta, log_ess = self._solve_tempering_exp(lnL, log_pq)
        self.tempering_exp_running = beta
        log_weights = beta*lnL + log_pq
        adapt_mode = 'beta'
        # Honest ESS of the FULL posterior weights (beta=1): how reachable the
        # posterior is from the current proposal cloud.
        log_ess1 = self._log_ess(lnL + log_pq)
        if self.temper_log:
            log_weights = self.xpy.log(self.xpy.maximum(lnL,1e-5))
        elif self.tempering_adapt and bool(log_ess1 < self.xpy.log(self.ess_floor)):
            # BOOTSTRAP (rank-elite refit, cross-entropy-method style): while
            # the posterior is out of reach of the cloud, beta-tempered
            # honest weights equilibrate the proposal near the PRIOR (the
            # solver keeps beta tiny while the cloud is broad, and the
            # -ln p_s correction cancels incremental concentration) -- the
            # proposal never localizes, exactly the eff_samp~1 stall seen in
            # the ILE SNR sequence.  Rank weights are lnL-scale-free and
            # compound: fit the top-k samples BY lnL (prior/proposal
            # corrected within the elite set), so the threshold ratchets up
            # every chunk like the AV sampler's volume shrinking.  Hand back
            # to the honest beta-solver once ESS(beta=1) clears the floor,
            # after which the refit target smoothly becomes L*p (beta->1).
            k_elite = int(min(max(self.ess_target, self.ess_floor), len(lnL)//2))
            if k_elite >= 2:
                gamma = self.xpy.sort(lnL)[-k_elite]
                neg_inf = -self.xpy.inf*self.xpy.ones(lnL.shape)
                log_weights = self.xpy.where(lnL >= gamma, log_pq, neg_inf)
                adapt_mode = 'elite'
        else:
            # If even beta=0 leaves too few effective samples (proposal/prior
            # pathologies), skip this refit: keep the current proposal rather
            # than fit garbage (breadcrumb item 2).
            if self.xpy.exp(log_ess) < min(self.ess_floor, self.d + 2):
                print(" GMM refit skipped: ESS {:.1f} too low even untempered ".format(float(self.xpy.exp(log_ess))))
                return
        if getattr(self, '_verbose_diag', False):
            print(" GMM adapt[{}]: mode={} beta={:.3g} ESS_beta={:.1f} ESS_1={:.3g} max_lnL={:.1f}".format(
                self.iterations, adapt_mode, float(beta),
                float(self.xpy.exp(log_ess)), float(self.xpy.exp(log_ess1)),
                float(self.xpy.max(lnL))))

        for dim_group in self.gmm_dict: # iterate over grouped dimensions
            if self.gmm_adapt:
                if (dim_group in self.gmm_adapt):
                    if not(self.gmm_adapt[dim_group]):
                        continue
            new_bounds = self.xpy.empty((len(dim_group), 2))
            new_bounds = self.bounds[dim_group]
            if len(new_bounds.shape) < 2:
                # 1-d group with flat bounds (per-dim default): GMM expects (d,2)
                new_bounds = self.xpy.array([new_bounds])
            model = self.gmm_dict[dim_group]
            temp_samples = self.xpy.empty((len(sample_array), len(dim_group)))
            index = 0
            for dim in dim_group:
                temp_samples[:,index] = sample_array[:,dim]
                index += 1
            # gmm_adaptive may be a dict {group:k_max} (per-group opt-in) or a
            # scalar/bool (apply to every adapting group -- used by the portfolio,
            # whose GMM member's grouping is not known here).
            adaptive_kmax = None
            if self.gmm_adaptive:
                if isinstance(self.gmm_adaptive, dict):
                    adaptive_kmax = self.gmm_adaptive.get(dim_group)
                elif isinstance(self.gmm_adaptive, bool):
                    adaptive_kmax = 8   # default cap when enabled globally
                else:
                    adaptive_kmax = int(self.gmm_adaptive)
            if model is None:
                if adaptive_kmax:
                    # FLEXIBLE allocation: choose this group's component count
                    # from the data by BIC at INITIALIZATION, then hand off to the
                    # proven-stable merge adaptation below (model.update()).  We
                    # deliberately do NOT re-fit fresh every chunk: a per-chunk
                    # BIC refit makes the proposal wander (measured: n_eff peaks
                    # then collapses) because each fit sees a different elite
                    # cloud; the incremental merge smooths that out.
                    # SAFETY FLOOR: never fewer components than the stress-tested
                    # hard-coded count for this group (self.n_comp) -- adaptive is
                    # a REFINEMENT that only adds capacity, e.g. a broad multi-modal
                    # sky keeps its default components.
                    if isinstance(self.n_comp, dict):
                        k_floor = self.n_comp.get(dim_group, 1)
                    else:
                        k_floor = self.n_comp
                    k_floor = int(k_floor) if isinstance(k_floor, int) and k_floor > 0 else 1
                    model = GMM.fit_gmm_adaptive(temp_samples, new_bounds,
                                                 log_sample_weights=log_weights,
                                                 k_max=max(int(adaptive_kmax), k_floor),
                                                 k_min=k_floor,
                                                 epsilon=self.gmm_epsilon,
                                                 defensive_frac=self.gmm_defensive_frac,
                                                 inflate=self.gmm_inflate)
                elif isinstance(self.n_comp, int) and self.n_comp != 0:
                    model = GMM.gmm(self.n_comp, new_bounds,epsilon=self.gmm_epsilon)
                    model.fit(temp_samples, log_sample_weights=log_weights)
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
                        getattr(self,'gmm_defensive_frac',0.0)
                        if getattr(self,'gmm_defensive_all_paths',False) else 0.0))
                elif isinstance(self.n_comp, dict) and self.n_comp[dim_group] != 0:
                    model = GMM.gmm(self.n_comp[dim_group], new_bounds,epsilon=self.gmm_epsilon)
                    model.fit(temp_samples, log_sample_weights=log_weights)
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
                        getattr(self,'gmm_defensive_frac',0.0)
                        if getattr(self,'gmm_defensive_all_paths',False) else 0.0))
            else:
                model.update(temp_samples, log_sample_weights=log_weights)
            try:
                model.score(temp_samples[:5])
                self.gmm_dict[dim_group] = model
            except:
                print(" Failed to update ", dim_group)


    def _calculate_results(self):
        if self.use_lnL:
            lnL = self.xpy.copy(self.value_array)
        else:
            lnL = self.xpy.log(self.value_array+regularize_log_scale)
        mask = self.xpy.ones(lnL.shape,dtype=bool)
        if not(self.L_cutoff is None):
            if not(self.xpy.isinf(self.L_cutoff)):
                mask = lnL > (self.xpy.log(self.L_cutoff) if self.L_cutoff > 0 else -self.xpy.inf)
        lnL = lnL[mask]
        prior = self.prior_array[mask]
        sampling_prior = self.sampling_prior_array[mask]
        
        self.cumulative_samples = self.xpy.append(self.cumulative_samples, self.sample_array[mask], axis=0)
        self.cumulative_values = self.xpy.append(self.cumulative_values, lnL, axis=0)
        self.cumulative_p = self.xpy.append(self.cumulative_p, prior, axis=0)
        self.cumulative_p_s = self.xpy.append(self.cumulative_p_s, sampling_prior, axis=0)
        
        log_weights = lnL + self.xpy.log(prior) - self.xpy.log(sampling_prior)
        if self.xpy.any(self.xpy.isnan(log_weights)):
            print(" NAN weight ")
            raise ValueError
        if self.terrible_lnw_threshold: 
            if self.xpy.max(log_weights) < self.terrible_lnw_threshold:
                print(" TERRIBLE FIT ")
                raise ValueError

        log_scale_factor = self.xpy.max(log_weights)
        if not(self.return_lnI):
            scale_factor = self.xpy.exp(log_scale_factor)
            log_weights -= log_scale_factor
            summed_vals = scale_factor * self.xpy.sum(self.xpy.exp(log_weights))
            integral_value = summed_vals / self.n
        
            scaled_error_squared = self.xpy.var(self.xpy.exp(log_weights)) / self.n
            log_error_scale_factor = 2. * log_scale_factor
        
            self.integral = (self.iterations * self.integral + integral_value) / (self.iterations + 1)
            self.scaled_error_squared = (self.iterations * self.xpy.exp(self.log_error_scale_factor - log_error_scale_factor) * self.scaled_error_squared + scaled_error_squared) / (self.iterations + 1)
            self.log_error_scale_factor = log_error_scale_factor
        
            self.total_value += summed_vals
            self.max_value = self.xpy.maximum(scale_factor, self.max_value)
            self.eff_samp = self.total_value / self.max_value
        else:
            log_sum_weights = _xpy_logsumexp(log_weights)

            log_integral_here = log_sum_weights - self.xpy.log(self.n)
            if not(self.integral ):
                self.integral = log_integral_here
                self.total_value = log_sum_weights
                self.max_value = log_scale_factor
            else:
                self.integral = _xpy_logsumexp([ self.integral + self.xpy.log(self.iterations), log_integral_here]) - self.xpy.log(self.iterations+1)
                self.total_value = _xpy_logsumexp([self.total_value, log_sum_weights])
                self.max_value = self.xpy.maximum(self.max_value, self.xpy.max(log_weights))
            self.eff_samp = self.xpy.exp(self.total_value - (self.max_value  ))

            tmp_max = self.xpy.max(log_weights)
            log_scaled_error_squared = self.xpy.log(self.xpy.var(self.xpy.exp(log_weights - tmp_max))) + 2*tmp_max - self.xpy.log(self.n)
            if not(self.scaled_error_squared):
                self.scaled_error_squared = log_scaled_error_squared
            else:
                self.scaled_error_squared = _xpy_logsumexp([ self.scaled_error_squared + self.xpy.log(self.iterations), log_scaled_error_squared]) - self.xpy.log(self.iterations+1)

    def _reset(self):
        for k in self.gmm_dict:
            self.gmm_dict[k] = None
        

    def integrate(self, func, min_iter=10, max_iter=20, var_thresh=0.0, max_err=10,
            neff=float('inf'), nmax=None, progress=False, epoch=None,verbose=True,force_no_adapt=False,use_lnL=False,return_lnI=False,**kwargs):
        n_adapt = int(kwargs["n_adapt"]) if "n_adapt" in kwargs else 100
        tripwire_fraction = kwargs["tripwire_fraction"] if "tripwire_fraction" in kwargs else 2
        tripwire_epsilon = kwargs["tripwire_epsilon"] if "tripwire_epsilon" in kwargs else 0.001
        self.use_lnL = use_lnL
        self.return_lnI = return_lnI
        self._verbose_diag = verbose   # per-chunk adaptation diagnostics in _train

        err_count = 0
        # Consecutive-refit-failure budget: if the proposal refit fails this
        # many chunks IN A ROW the proposal has never adapted and the returned
        # integral/eff_samp are meaningless (the cupy-without-GPU regression
        # produced exactly this: every refit raised, 'Error training,
        # resetting...' each chunk, and integrate() returned eff_samp~1 with no
        # error signal).  Fail loudly instead.
        max_train_fail = int(kwargs["max_consecutive_train_failures"]) if "max_consecutive_train_failures" in kwargs else 5
        consec_train_fail = 0
        cumulative_eval_time = 0
        adapting=True
        if nmax is None:
            nmax = max_iter * self.n
        while self.iterations < max_iter and self.ntotal < nmax and self.eff_samp < neff:
            if (self.ntotal > nmax*tripwire_fraction) and (self.eff_samp < 1+tripwire_epsilon):
                print(" Tripwire: n_eff too low ")
                raise Exception("Tripwire on n_eff")

            if force_no_adapt or self.iterations >= n_adapt:
                adapting=False
            if err_count >= max_err:
                print('Exiting due to errors...')
                self.integration_error = 'exited after {} sampling/results/training errors'.format(err_count)
                break
            try:
                self._sample()
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                break
            except Exception as e:
                print(traceback.format_exc())
                print('Error sampling, resetting...')
                err_count += 1
                self._reset()
                continue
            t1 = time.time()
            if self.proc_count is None:
                # Ensure input to func is numpy for CPU-based user functions if needed, 
                # but the user is encouraged to support xpy.
                self.value_array = func(self.xpy.copy(self.sample_array)).flatten()
            else:
                split_samples = self.xpy.array_split(self.sample_array, self.proc_count)
                p = Pool(self.proc_count)
                self.value_array = self.xpy.concatenate(p.map(func, split_samples), axis=0)
                p.close()
            cumulative_eval_time += time.time() - t1
            self._calculate_prior()
            try:
                self._calculate_results()
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                break
            except Exception as e:
                print(traceback.format_exc())
                print('Error calculating results, resetting...')
                err_count += 1
                self._reset()
                continue
            self.iterations += 1
            self.ntotal += self.n
            testval = self.scaled_error_squared
            if not(self.return_lnI):
                testval = self.xpy.log(self.scaled_error_squared) + self.log_error_scale_factor
            if self.iterations >= min_iter and testval < self.xpy.log(var_thresh):
                break
            try:
                if adapting:
                    self._train()
                    consec_train_fail = 0
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                break
            except Exception as e:
                print(traceback.format_exc())
                print('Error training, resetting...')
                err_count += 1
                consec_train_fail += 1
                self._reset()
                if consec_train_fail >= max_train_fail:
                    self.integration_error = 'proposal refit failed {} consecutive times; proposal never adapted'.format(consec_train_fail)
                    raise RuntimeError('GMM ' + self.integration_error) from e
            if self.user_func is not None:
                self.user_func(self)
            if progress:
                for k in self.gmm_dict:
                    if self.gmm_dict[k] is not None:
                        self.gmm_dict[k].print_params()
            if epoch is not None and self.iterations % epoch == 0:
                self._reset()
            if verbose:
                print(self.scaled_error_squared)
                if not(self.return_lnI):
                    print(" : {} {} {} {} {} ".format((self.iterations-1)*self.n, self.eff_samp, self.xpy.sqrt(2*self.xpy.max(self.cumulative_values)), self.xpy.sqrt(2*(self.xpy.log(self.integral))),  self.xpy.sqrt(self.scaled_error_squared )/self.integral/self.xpy.sqrt(self.iterations ) ) )
                else:
                    print(" : {} {} {} {} {} ".format((self.iterations-1)*self.n, self.eff_samp, self.xpy.sqrt(2*self.xpy.max(self.cumulative_values)), self.xpy.sqrt(2*self.integral), self.xpy.exp(0.5*(self.scaled_error_squared - self.integral*2) )/self.xpy.sqrt(self.iterations)))
        print('cumulative eval time: ', cumulative_eval_time)
        print('integrator iterations: ', self.iterations)
