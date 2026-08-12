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

# Opt-in per-cycle trace of the live-volume contraction (RIFT_AV_TRACE=1).  Diagnosing
# a contraction failure needs the *sequence* of (n_finite, ninj, thr, nrec), which no
# other output exposes; it is far too chatty for production, hence the env gate.
_AV_TRACE = bool(os.environ.get('RIFT_AV_TRACE', ''))

def _av_trace(msg):
    if _AV_TRACE:
        print("  [AV trace] " + msg)
        sys.stdout.flush()

class NanOrInf(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class LiveVolumeCollapse(Exception):
    """The adaptive-volume live set is empty (or carries no usable information).

    Raised INSTEAD of the bare numpy/cupy "zero-size array to reduction operation
    ... which has no identity" that the empty-live-volume path used to produce, so
    callers and logs can tell a degenerate contraction apart from a waveform
    generation failure.  See the collapse discussion in get_likelihood_threshold.
    """
    pass

def live_volume_collapse_verdict(n_live, ndim, ess=None, khat=None,
                                 n_empty_cycles=0, n_live_collapses=0,
                                 n_warm_seed=None, n_warm_seed_rank=None,
                                 n_warm_seed_dim=None):
    """Has the adaptive-volume live set degenerated?  -> (collapsed, [reasons])

    A degenerate contraction must be REPORTED rather than silently exported: the run
    still returns a lnZ and a sample cloud, but both describe a single mode and the
    cloud is not a fair posterior draw.

    Thresholds, against the separation measured on zero-noise injections at a fixed
    intrinsic point (rho_net 51 -> 147):
        healthy    ESS 16.8-20.2,  k-hat 1.03-1.50   (rho 51.4, converges cold)
        collapsed  ESS 1.0-1.7,    k-hat 21-202      (rho 103-147)
      * n_live <= ndim -- a live set no larger than the dimension cannot span the
        space, let alone describe a posterior in it.  Geometric, not tuned.
      * ESS < 2 -- fewer than two effective samples IS one sample.
      * ESS < 5 with k-hat > 10 -- near-degenerate AND a pathological weight tail.
    The gap between the regimes is an order of magnitude wide, so these sit far from
    both sides of it.  k-hat is deliberately NOT a gate on its own: it exceeds its
    nominal 0.70 "unresolved tail" threshold even in the healthy runs on this problem,
    and it is not always computable once the live set is tiny.

    A WARM start fails the OTHER way, and none of the rules above see it: seeded from too
    few points the grid contracts onto a sliver of the support, the integrand is then flat
    across it, and the pass terminates in one cycle looking excellent -- large n_live, ESS
    ~ n, small k-hat -- while lnZ is short by the mass outside the sliver.  Measured on 12
    rho_net=146.8 rescue replicates: the eleven seeded from 2000 puffed points warm-started
    at V = 7.5e-9 to 1.5e-8 (351-684 live bins) and returned ln(Z/Lmax) = -27.0 to -30.6;
    the one seeded from 2 points warm-started at V = 9.2e-36 (13 bins) and returned -80.7,
    i.e. ~50 nats low, with eff_samp 9789 of 10010 samples.  So:
      * n_warm_seed_rank < n_warm_seed_dim -- the seed cloud's AFFINE RANK (rank of the
        mean-centred points, per-axis scaled by the box) is below the dimension it must
        span, so it lies in a lower-dimensional subspace and cannot define a volume there.
    RANK, not row count, is the invariant.  Rows are neither necessary nor sufficient:
    thousands of duplicated or collinear points span the same degenerate subspace two
    points do and fail identically, while d+1 affinely independent points are exactly
    enough to define a volume in d dimensions and must NOT be flagged.  Rank subsumes the
    count anyway, since n points span at most n-1 affine dimensions.

    n_warm_seed* are None on a cold pass -> the rule is skipped.  If rank is unavailable
    (a grid restored by load_state from a run that predates it) we fall back to the count,
    at the correct simplex boundary: fewer than dim+1 points, i.e. n <= dim.
    """
    reasons = []
    _seed_dim = n_warm_seed_dim if n_warm_seed_dim else ndim
    if n_warm_seed_rank is not None and n_warm_seed_dim:
        if n_warm_seed_rank < n_warm_seed_dim:
            reasons.append(
                "warm-started from a seed of affine rank {} in {} adaptive dimension(s)"
                "{}".format(n_warm_seed_rank, n_warm_seed_dim,
                            "" if not n_warm_seed else " ({} point(s))".format(n_warm_seed)))
    elif n_warm_seed and n_warm_seed <= _seed_dim:   # None (cold) / 0 (unknown) -> skip
        reasons.append("warm-started from only {} seed point(s) in {} dimensions".format(
            n_warm_seed, _seed_dim))
    if n_empty_cycles:
        reasons.append("{} cycle(s) with no finite in-volume sample".format(n_empty_cycles))
    if n_live_collapses:
        reasons.append("{} cycle(s) whose threshold emptied the live set".format(n_live_collapses))
    if n_live <= ndim:
        reasons.append("final live volume holds {} sample(s) in {} dimensions".format(n_live, ndim))
    if ess is not None and (ess < 2.0 or (khat is not None and ess < 5.0 and khat > 10.0)):
        reasons.append("ESS={:.2f}".format(ess)
                       + ("" if khat is None else " with k-hat={:.1f}".format(khat)))
    return bool(reasons), reasons


### V. Tiwari routines

def get_likelihood_threshold(lkl, lkl_thr, nsel, discard_prob,xpy_here=xpy_default):
    """
    Find the likelihood threshold that encolses a probability
    lkl  : array of likelihoods (on bins)
    lkl_thr: scalar cutoff
    nsel : integer, has to do with size of array of likelihoods used to evaluate for next array.
    discard_prob: threshold on CDF to throw away an entire bin.  Should be very small
    """
    if len(lkl) == 0:
        # Caller must not ask for a threshold on an empty live volume: every reduction
        # below (max, argsort, [0]) is undefined.  Named error, so the caller can tell
        # this apart from a waveform/likelihood failure.
        raise LiveVolumeCollapse("no samples in the live volume: cannot set a likelihood threshold")

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

    # CLAMP: the threshold is applied downstream as a STRICT `lkl > thr`, so a threshold
    # at or above max(lkl) discards the entire live volume -- which encloses zero
    # probability and so contradicts the enc_prob=0.999 this function exists to maintain.
    # It happens whenever the live set is small AND one weight dominates: then
    # prob_stop_thr saturates at max(lkl) (every other weight underflows to 0, so the
    # discard_prob quantile IS the top sample) while lkl_stop_thr falls back to the
    # array MINIMUM (the len<=nsel branch above).  At high SNR both conditions hold from
    # the first cycle -- ~1e5 cold extrinsic draws yield a handful of finite lnL -- and
    # the live volume ratchets down one sample per cycle to 1, then to 0, and every
    # reduction over it raises "zero-size array to reduction operation ... no identity".
    # Back the threshold off to the largest value strictly below the maximum so at least
    # the peak always survives.  In a healthy run len(lkl) >> nsel and lkl_stop_thr is
    # the nsel-th largest, far below the max, so this clamp never engages.
    # Reduce on the ACTIVE backend and move only the scalar: identity_convert(lkl) here
    # would copy the whole live set device->host every cycle, on the healthy path too.
    lkl_max = float(identity_convert(xpy_here.max(lkl)))
    if not (float(identity_convert(lkl_thr)) < lkl_max):
        lkl_host = identity_convert(lkl)   # rare branch: the live set is tiny by construction
        below = lkl_host[lkl_host < lkl_max]
        if len(below):
            lkl_thr = np.max(below)          # keep only the maximum: maximal (but safe) contraction
        else:
            # every surviving sample has the SAME lnL: no contraction is possible, so
            # take a threshold below all of them and leave the live volume intact.
            lkl_thr = np.nextafter(lkl_max, -np.inf)
        _av_trace("threshold CLAMPED to {:.10g} (would have discarded the whole live volume of {})".format(
            float(lkl_thr), npoints))

    if _AV_TRACE:
        _av_trace("threshold: n={} nsel={} lkl_stop_thr={:.6g} prob_stop_thr={:.6g} -> thr={:.6g} (max={:.6g})".format(
            npoints, nsel, float(identity_convert(lkl_stop_thr)), float(identity_convert(prob_stop_thr)),
            float(identity_convert(lkl_thr)), lkl_max))

    truncp = xpy_here.sum(w[lkl < lkl_thr]) / sumw

    return identity_convert(lkl_thr), identity_convert(truncp)  # send both to CPU as needed

def seed_affine_rank(pts, box_lo, box_hi, axes=None, tol=1e-9):
    """Affine rank of a warm-seed cloud over `axes`, i.e. the dimension of the subspace
    the seed actually spans -> (rank, n_in_box).

    THE ONE PLACE this is defined, because two callers must agree exactly: the grid
    builder records it for the collapse diagnostic, and the ILE's L0 rescue tests it to
    decide whether the seed needs puffing.  A rescue that puffed on a rank the diagnostic
    then measured differently would either puff a healthy seed or ship a flagged one.

    Measured the way _build_grid_from_points must see it:
      * IN-BOX ROWS ONLY.  The grid only ever spans the box, so out-of-box rows describe
        nothing it will build -- and left in they inflate the rank, so a seed that is
        degenerate where it matters could be recorded full-rank.
      * mean-centred (AFFINE rank: n points span at most n-1 affine dimensions, so this
        subsumes the row-count test that used to stand in for it), and
      * per-axis scaled by the box, so the tolerance is unit-free -- a distance in Mpc and
        an angle in radians must not get different tolerances.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    box_lo = np.asarray(box_lo, dtype=float)
    box_hi = np.asarray(box_hi, dtype=float)
    if pts.size == 0:
        return 0, 0
    inside = np.all((pts >= box_lo) & (pts <= box_hi), axis=1)
    core = pts[inside]
    if len(core) < 2:
        return 0, len(core)
    ax = list(range(pts.shape[1])) if axes is None else list(axes)
    core = core[:, ax]
    scaled = (core - core.mean(axis=0)) / np.clip((box_hi - box_lo)[ax], 1e-300, None)
    return int(np.linalg.matrix_rank(scaled, tol=tol)), len(core)


def make_warm_seed_reserve(X, lnL, params_ordered, n_max=20000,
                           log_joint_prior=None, log_joint_s_prior=None, rng=None):
    """A bounded copy of the points a pass RETAINED, for a later warm start -> dict.

    THE ONE BUILDER, because every sampler that can be L0-rescued needs the identical
    record and they reach this point by different routes: mcsamplerAdaptiveVolume from its
    own accumulated draws, mcsamplerPortfolio from its aggregated _rvs (it drives members
    through draw_simplified(), never their integrate_log(), so a member never builds one).

    WHY IT HAS TO BE TAKEN EARLY.  Both samplers then prune _rvs and fair-draw it down to
    ~1.5*n_eff rows resampled WITH REPLACEMENT -- a resample built for EXPORT.  Anything
    that reads _rvs afterwards and treats it as the sample set sees, on the collapsed pass
    a rescue exists for, a handful of rows several of which are the same point twice.

    Bounded by a uniform subsample WITHOUT replacement, because the full array is the one
    the surrounding code calls a memory hog.  The PEAK row is appended unconditionally: the
    seed is defined relative to it and a subsample can drop it.

    The two prior components ride along when given, so a consumer can rebuild the importance
    weight -- and therefore lnZ -- from the retained set rather than from the fair draw.
    """
    X = np.atleast_2d(np.asarray(identity_convert(X), dtype=float))
    lnL = np.asarray(identity_convert(lnL), dtype=float).ravel()
    n_ret = len(X)
    extra = {}
    if log_joint_prior is not None:
        extra['log_joint_prior'] = np.asarray(identity_convert(log_joint_prior), dtype=float).ravel()
    if log_joint_s_prior is not None:
        extra['log_joint_s_prior'] = np.asarray(identity_convert(log_joint_s_prior), dtype=float).ravel()
    n_max = int(n_max)
    if n_max > 0 and n_ret > n_max:
        rng = rng if rng is not None else np.random
        idx = rng.choice(n_ret, size=n_max, replace=False)
        idx = np.unique(np.append(idx, int(np.nanargmax(lnL))))
        X, lnL = X[idx], lnL[idx]
        extra = {k: v[idx] for k, v in extra.items()}
    out = dict(X=X, lnL=lnL, n_retained=int(n_ret), params_ordered=list(params_ordered))
    out.update(extra)
    return out


def warm_seed_scale_from_finite_points(points, lnL, box_lo, box_hi, axes,
                                       eig_lo=1e-5, eig_hi=0.5):
    """Estimate the POSTERIOR scale (a box-scaled covariance over `axes`) from the finite
    log-likelihoods a collapsed pass already drew -> cov, or None if it cannot be measured.

    Why this exists.  The L0 rescue's fallback puff used a hardcoded 1/200 of each
    parameter's prior range, which is a property of the PRIOR and knows nothing about the
    posterior -- but the posterior narrows as 1/rho, so one fixed fraction cannot be right
    across the amplitude range, and a puff narrower than the posterior truncates real mass
    (VARAHA's live volume only ever contracts, so the seed is a ceiling on the support).

    The information needed is already in hand and was being thrown away.  A cold pass at
    high amplitude draws (near enough) uniformly from the prior box, and returns a finite
    lnL only inside the region where exp() has not underflowed -- i.e. the level set
    lnL > lnL_max - D.  For a locally Gaussian peak that level set is the ellipsoid
    u^T A u < 2D (u box-scaled about the peak), and points uniform in an ellipsoid have
    covariance (2D/(d+2)) A^{-1}.  So the posterior covariance A^{-1} is recovered as

        cov_post  =  cov(finite points) * (d + 2) / (2 D),     D = lnL_max - min(finite lnL)

    which is one sample covariance, no fit and nothing to fail to converge.  It also
    delivers the CORRELATIONS -- sky position, time and distance are strongly correlated at
    high amplitude, and an isotropic puff wastes almost all of its points off the ridge.

    Deliberately approximate -- the draws are only uniform-in-prior until AV starts
    contracting, and the peak is only locally Gaussian.  Measured against a known lnZ on a
    correlated 6-D peak with the same 745-nat underflow (sigma recovered / true, per axis,
    6 replicates): 1.01 - 1.23.  So it is good to ~20%, which is what matters, because the
    error that was being made is a factor of ~5-10.

    NEITHER DIRECTION IS FREE, so do not treat "wide is safe" as a licence.  Too narrow
    silently truncates: on that same target a puff at the historical 1/200 of the prior
    range came in 0.8 - 8.3 nats below the truth, with a healthy-looking ESS.  Too wide is
    not merely inefficient, which is what the surrounding code used to assume.  Scanning a
    multiplier on this estimate, mean (worst) lnZ error over 6 replicates, mean ESS:

        x0.5   -8.52 (-19.0) nats, ESS 71      truncated
        x1     -1.61  (-3.5) nats, ESS 53
        x2     +0.08  (-0.2) nats, ESS 52      <-- the default
        x3     +1.13  (+0.6) nats, ESS 26
        x6     +3.03  (-1.6) nats, ESS 10      biased HIGH, and efficiency is going
        x12   -29.99 (-71.9) nats, ESS  1      a cold start in all but name: re-collapses

    Both tails are wrong, and the useful range is under a decade wide, so inflate by a small
    factor and not by an order of magnitude.

    Eigenvalues are floored/capped in box-scaled units (`eig_lo`, `eig_hi` are standard
    deviations as a fraction of the box) so no direction can come back degenerate -- a
    zero-width direction would re-create the rank deficiency this is being used to repair.
    """
    box_lo = np.asarray(box_lo, dtype=float)
    box_hi = np.asarray(box_hi, dtype=float)
    box = np.clip(box_hi - box_lo, 1e-300, None)
    ax = list(axes)
    d = len(ax)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    lnL = np.asarray(lnL, dtype=float).ravel()
    good = np.isfinite(lnL) & np.all(np.isfinite(pts), axis=1) \
        & np.all((pts >= box_lo) & (pts <= box_hi), axis=1)
    if int(np.sum(good)) < max(2 * d, d + 2):
        return None                     # too few finite points to estimate a d-dim covariance
    u = (pts[good][:, ax] - box_lo[ax]) / box[ax]
    depth = float(np.max(lnL[good]) - np.min(lnL[good]))
    if not np.isfinite(depth) or depth <= 0:
        return None
    cov = np.cov(u, rowvar=False) * (d + 2.0) / (2.0 * depth)
    cov = np.atleast_2d(cov)
    if not np.all(np.isfinite(cov)):
        return None
    w, Q = np.linalg.eigh(0.5 * (cov + cov.T))
    w = np.clip(w, eig_lo ** 2, eig_hi ** 2)
    return Q @ np.diag(w) @ Q.T


def build_warm_seed(points, lnL, box_lo, box_hi, axes, deltalnL=15.0,
                    puff_width_frac=1.0 / 200, puff_scale='auto', puff_factor=2.0,
                    n_puff=2000, seed=0):
    """Build the L0 rescue's warm seed from a pass's own samples -> (seed, info).

    `points` (n, ndim) and `lnL` (n,) are the completed pass's draws.  The seed is the
    points within `deltalnL` of the peak, PUFFED to full rank if they do not span `axes`.

    RANK, NOT COUNT, is the guard.  The rule this replaces was `len(seed) < 2`, and a count
    cannot see the failure: measured on zero-noise injections, a 5-point seed at rho_net
    102.8 had affine rank 2-4 of 6 and a 2-point seed at rho_net 146.8 had rank 0 of 6.
    Both passed the count test, and both then warm-started a live volume that had collapsed
    onto a degenerate subspace (V ~ 3e-06 and ~9e-36 against a healthy ~1e-08), which
    reports a fine n_eff while lnZ is a lower bound.  n points span at most n-1 affine
    dimensions, so the rank test subsumes the count it replaces.

    AUGMENT, DO NOT REPLACE.  The handful of real points are the only direct evidence of
    where the peak is and how wide it is, so they are kept and the puff is added alongside
    them.  This can only help: the grid is built from the union's extent, so a real point
    lying outside the puff widens the seeded volume to include it, and VARAHA can only
    contract afterwards -- whereas replacing them throws that information away and pins the
    support to a guessed width about a single point.

    `puff_scale`:
      'fixed'  -- isotropic, `puff_width_frac` of each parameter's prior range (the
                  historical behaviour; `puff_width_frac` = 1/200 reproduces it exactly).
      'auto'   -- the measured posterior scale and correlations from every finite lnL the
                  pass drew (warm_seed_scale_from_finite_points), falling back to 'fixed'
                  when there are too few finite points to estimate one.
    `puff_factor` multiplies the resulting width (variance scales as its square).  2 is the
    measured optimum and both tails are wrong -- see warm_seed_scale_from_finite_points.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    lnL = np.asarray(lnL, dtype=float).ravel()
    box_lo = np.asarray(box_lo, dtype=float)
    box_hi = np.asarray(box_hi, dtype=float)
    box = np.clip(box_hi - box_lo, 1e-300, None)
    ax = list(axes)
    ndim = pts.shape[1]
    best = pts[int(np.nanargmax(lnL))]
    core = pts[lnL > (np.nanmax(lnL) - float(deltalnL))]
    rank, n_in_box = seed_affine_rank(core, box_lo, box_hi, axes=ax)
    info = dict(n_core=int(len(core)), n_core_in_box=int(n_in_box), rank_core=int(rank),
                dim=len(ax), puffed=False, puff_scale=None, n_puff=0,
                rank_final=int(rank), n_seed=int(len(core)))
    if rank >= len(ax):
        return core, info

    # --- the seed is rank-deficient: puff to full rank about the best point
    cov_u = None
    if puff_scale == 'auto':
        cov_u = warm_seed_scale_from_finite_points(pts, lnL, box_lo, box_hi, ax)
    used = 'auto'
    if cov_u is None:
        used = 'fixed'
        cov_u = np.diag(np.full(len(ax), float(puff_width_frac) ** 2))
    cov_u = cov_u * (float(puff_factor) ** 2)
    rng = np.random.RandomState(seed)
    n_puff = int(n_puff)
    # scaled draws on the adaptive axes; the remaining axes get the isotropic width (the
    # grid puts one bin on them, so their only job is to not be a single repeated value)
    u = rng.multivariate_normal(np.zeros(len(ax)), cov_u, size=n_puff)
    pad = np.tile(best, (n_puff, 1)).astype(float)
    pad[:, ax] += u * box[ax]
    _other = [i for i in range(ndim) if i not in set(ax)]
    if _other:
        pad[:, _other] += rng.normal(
            0.0, float(puff_width_frac) * float(puff_factor), size=(n_puff, len(_other))) * box[_other]
    # CLIP to the box.  The grid builder discards out-of-box rows, so an unclipped puff
    # silently loses points (and, at a peak near an edge, most of them) -- and a seed the
    # sampler never sees is not the seed that was measured for rank here.
    pad = np.clip(pad, box_lo, box_hi)
    out = np.vstack([core, pad]) if len(core) else pad
    rank_final, _ = seed_affine_rank(out, box_lo, box_hi, axes=ax)
    info.update(puffed=True, puff_scale=used, n_puff=n_puff,
                rank_final=int(rank_final), n_seed=int(len(out)),
                puff_sigma_scaled=np.sqrt(np.clip(np.diag(cov_u), 0, None)))
    return out, info


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
    # COMPACT SUPPORT: this sampler's density is EXACTLY ZERO outside its contracted live volume,
    # so once seeded or contracted it cannot serve as the mixture's coverage guarantee.
    # mcsamplerPortfolio reads this to decide whether it must hold one member cold.
    has_unbounded_support = False

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

    def warm_seed_axes(self):
        """Column indices a warm seed must span: the ADAPTIVE axes (all of them when
        nothing is adaptive, since then the grid is one bin per dim and the seed's only
        job is to be well-defined).  Exposed so a caller building a seed -- the ILE's L0
        rescue -- can ask the sampler which dimensions its seed will be judged on instead
        of guessing.  A portfolio has no such axes of its own; ask a member."""
        if getattr(self, 'd_adaptive', 0) > 0:
            return list(self.indx_adaptive)
        return list(range(len(self.params_ordered)))

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
        # The core is filtered to the box by the SAME rule as `pts` above.  It describes the
        # grid that gets built, and the grid only ever spans the box, so points outside it
        # are not part of that description: left in, they widen `ext` (under-resolving the
        # grid, since V_extent can even exceed 1) and they inflate the recorded affine rank,
        # so a seed that is degenerate in-box could be recorded full-rank.  Callers do not
        # always clip -- bootstrap_from_samples clips only when inflating, and the ILE's
        # puffed fallback seed is an unclipped Gaussian about the peak.
        if resolution_pts is None:
            res_pts = pts
        else:
            res_pts = np.atleast_2d(np.asarray(resolution_pts, dtype=float))
            _res_in = np.all((res_pts >= box_lo) & (res_pts <= box_hi), axis=1)
            # Fall back to the (already filtered, guaranteed non-empty) full cloud if the
            # core lies entirely outside: a resolution set of zero points defines nothing.
            res_pts = res_pts[_res_in] if np.any(_res_in) else pts
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
        # SEED PROVENANCE, carried so integrate_log can report a seed that cannot define a
        # volume -- see live_volume_collapse_verdict.  Not sampling parameters.
        #
        # The row COUNT alone is the wrong test.  Many rows that are duplicated or
        # collinear span the same degenerate subspace two points do and produce the
        # identical near-zero-volume failure; conversely d+1 affinely independent points
        # are exactly enough to define a volume in d dimensions and are fine.  So record
        # the AFFINE RANK of the seed cloud -- the rank of the mean-centred points -- over
        # the adaptive axes, scaled by the box so the test is unit-free (a distance in Mpc
        # and an angle in radians must not get different tolerances).  n points span at
        # most n-1 affine dimensions, so rank subsumes the count test.
        _ax = self.warm_seed_axes()
        n_seed_rank, _ = seed_affine_rank(res_pts, box_lo, box_hi, axes=_ax)
        return dict(binunique=binunique, dx=dx, nbins=nbins, V=V,
                    loglkl_thr=loglkl_thr, trunc_p=1e-10, n_seed=nrec,
                    n_seed_rank=n_seed_rank, n_seed_dim=len(_ax))

    def bootstrap_from_samples(self, samples, params=None, loglkl=None, enc_prob=0.999,
                               cover_frac=0.0, dilate=1, inflate=1.0, seed=None):
        """Warm-start from an explicit set of reference points populating the
        high-likelihood region (e.g. a previous run's posterior draws, a puff of
        an earlier MAP point, or fair-draw samples from a prior ILE instance).
        `loglkl` (optional) is L*prior at those points, used to seed the threshold.

        `cover_frac` (0..1) mixes this fraction of uniform full-box points into the
        seed cloud, widening the seeded live volume.  Leave 0 when reusing a proposal
        for the SAME problem (e.g. an in-run second pass).

        IT IS NOT A COVERAGE GUARANTEE, despite what this docstring claimed until
        2026-08.  A FINITE set of uniform points occupies only the bins it lands in,
        so the seeded grid is NOT a superset of a cold (uniform) start -- and the
        shortfall grows fast with dimension.  Measured, fraction of the [-5,5]^d prior
        box covered by the seeded grid (a cold start is 1.0 by construction):

            cover_frac:      0.0        0.2       0.5      0.9
            d=2           0.027      0.634     0.982     1.000
            d=4          0.0015      0.028     0.104     0.620
            d=6         6.3e-05    0.00087    0.0033    0.0287

        At d=6 even cover_frac=0.9 leaves 97% of the box unsampled.  So the claim that
        "a warm-started integral can never be MORE biased than a cold one" was false;
        do not rely on it.

        WHAT ACTUALLY PROTECTS YOU is having a component with support everywhere.  In
        the default AV+GMM portfolio that is the GMM member: a Gaussian mixture has
        nonzero density over the whole box, so q_mix never vanishes and a badly-seeded
        AV member costs efficiency rather than bias (measured with a deliberately
        displaced seed at d=4 and d=6: |lnZ bias| <= 0.05 in every run).  An ALL-AV
        portfolio has no such member -- every component is a hard-edged box -- and the
        same displaced seed gave lnZ bias -1.0 to -6.8 nats.

        And note the limit of any coverage fix: in that all-AV test, keeping one member
        fully cold (V=1) still gave -1.1 to -4.2 nats, because a uniform member at d=6
        finds a sharp peak too rarely to carry the integral within the budget (n_eff
        3-9).  Coverage in principle is necessary, not sufficient.  A badly mismatched
        seed is an efficiency catastrophe no knob repairs -- detect it (the L0 rescue)
        or do not warm-start across dissimilar points.

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
                 trunc_p=warm.get('trunc_p', 1e-10),
                 n_seed=warm.get('n_seed', 0),   # 0 = unknown provenance (grid taken from the live state)
                 n_seed_rank=warm.get('n_seed_rank', -1),   # -1 = not recorded by the producing run
                 n_seed_dim=warm.get('n_seed_dim', 0))
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
                          loglkl_thr=float(d['loglkl_thr']), trunc_p=float(d['trunc_p']),
                          n_seed=int(d['n_seed']) if 'n_seed' in d else 0,
                          # -1 / 0 = a state file written before the rank was recorded; the
                          # verdict then falls back to the count rule.
                          n_seed_rank=int(d['n_seed_rank']) if 'n_seed_rank' in d else -1,
                          n_seed_dim=int(d['n_seed_dim']) if 'n_seed_dim' in d else 0)
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
        n_adapt -- IGNORED as an adaptation schedule by this sampler; accepted only for API
            compatibility with mcsampler/mcsamplerGPU, where it does gate update_sampling_prior.
            AV has no update_sampling_prior: its volume adaptation is intrinsic to the algorithm
            and runs every cycle regardless of this value. (Verified: output is bit-identical for
            n_adapt in {10,100,1000}.) The ONLY thing it still does here is participate in the
            save_intg gate, so n_adapt=0 can still suppress the _rvs cache. Do not reach for this
            expecting an "adapt then freeze" control -- there isn't one.
        convergence_tests - dictionary of function pointers, each accepting self._rvs and self.params as arguments. CURRENTLY ONLY USED FOR REPORTING
        Pinning a value: By specifying a kwarg with the same of an existing parameter, it is possible to "pin" it. The sample draws will always be that value, and the sampling prior will use a delta function at that value.
        """


        xpy_here = self.xpy

        # A SECOND integral on the same sampler object (the ILE L0 warm-start rescue) must not
        # inherit the first one's 'integrand'.  integrate() writes that key AFTER integrate_log
        # returns -- i.e. after the block below has already moved every array to the host and, if
        # a fair draw ran, truncated it to the fair-draw length.  It is therefore stale on entry
        # here in BOTH size and backend, and integrate_log repopulates every other key but not it.
        # The fair-draw loop then indexes it (host, cold length) with a device index array and
        # raises "Implicit conversion to a NumPy array is not allowed", aborting the pass.
        # mcsamplerPortfolio.integrate_log already drops it on entry for the same reason.
        if 'integrand' in self._rvs:
          del self._rvs['integrand']
        # Same hazard for the warm-seed reserve, and a worse consequence: a pass that raises
        # part-way leaves the PREVIOUS point's retained samples sitting here, and an L0 rescue
        # would then seed this point's live volume from a different point's peak.  Drop it on
        # entry, so "present" always means "this pass wrote it".
        self._warm_seed_reserve = None

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
        # NOTE: n_adapt does NOT schedule adaptation in this sampler -- AV has no
        # update_sampling_prior, and the volume adaptation below runs every cycle
        # unconditionally.  It survives only as a way to force save_intg off (n_adapt=0),
        # which is how --no-adapt reaches this code.  It is deliberately NOT wired up to
        # gate adaptation: doing so would change the numerics of every production AV run.
        # Do not read the value below as "adapt to 1000 chunks, then freeze" -- it never
        # freezes.  (Empirically bit-identical output for n_adapt in {10,100,1000}.)
        n_adapt = int(kwargs["n_adapt"]*n) if "n_adapt" in kwargs else 1000
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
        n_warm_seed = None   # in-box points the seeded grid was built from (None: cold pass)
        n_warm_seed_rank = None  # affine rank of that seed cloud (None: cold, or not recorded)
        n_warm_seed_dim = None   # dimensions it had to span
        V_warm = None        # the seeded fractional volume, for the collapse report
        if warm is not None:
            self.binunique = np.array(warm['binunique'])
            self.dx = np.array(warm['dx'])
            self.nbins = np.array(warm['nbins'])
            self.ninbin = ((self.n_chunk // self.binunique.shape[0] + 1) * np.ones(self.binunique.shape[0])).astype(int)
            V = float(warm['V'])
            loglkl_thr = float(warm['loglkl_thr'])
            trunc_p = float(warm.get('trunc_p', 1e-10))
            # 0 (or absent) = provenance unknown, e.g. a grid restored by load_state from a run
            # that predates this field; the seed-size check below then does not fire.
            n_warm_seed = int(warm.get('n_seed', 0)) or None
            # Affine rank of the seed cloud, which is the invariant the verdict tests; -1 or
            # absent means the producing run predates it and the count rule is used instead.
            _rk = int(warm.get('n_seed_rank', -1))
            n_warm_seed_rank = _rk if _rk >= 0 else None
            n_warm_seed_dim = int(warm.get('n_seed_dim', 0)) or None
            V_warm = V
            if bShowEvaluationLog:
                print("  [AV warm-start] live bins={} V={:.3e} loglkl_thr={:.3g} from {} seed pt(s), affine rank {}/{}".format(
                    self.binunique.shape[0], V, loglkl_thr,
                    "?" if n_warm_seed is None else n_warm_seed,
                    "?" if n_warm_seed_rank is None else n_warm_seed_rank,
                    "?" if n_warm_seed_dim is None else n_warm_seed_dim))

        var_lnV = 0.0  # accumulated variance of ln(V): V is a stochastic product of per-cycle
                       # binomial survival fractions, and Z ~ V*mean(w), so Var(lnV) is a
                       # component of the lnZ error the weight variance is structurally blind to
        if cupy_ok:
          allx = identity_convert_togpu(allx)
          allloglkl = identity_convert_togpu(allloglkl)

        # live-volume health bookkeeping (reported at the end and in dict_return)
        n_empty_cycles = 0      # cycles in which NO finite sample fell inside the live volume
        n_live_collapses = 0    # cycles in which the threshold would have emptied the live set
        collapse_reported = False
        nrec = 0
        allloglkl_prev, allp_prev, allx_prev = allloglkl, allp, allx
        loglkl_thr_prev = loglkl_thr

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

            # Admit only FINITE samples above threshold.  A +inf lnL (overflow in a
            # degenerate extrinsic configuration) passes the plain `> thr` test and then
            # poisons every downstream max()/exp(); NaN silently fails it.  Screening here
            # matches update_sampling_prior_selfish, which already does this.
            idxsel = xpy_here.where(xpy_here.logical_and(loglkl > loglkl_thr, xpy_here.isfinite(loglkl)))
            # How many samples did THIS chunk contribute?  Count before the append: `ninj`
            # below is the CUMULATIVE live-set size, so testing that instead would only ever
            # detect LEADING empty chunks.  Once a single sample has survived, a later chunk
            # contributing nothing would sail past such a test and re-threshold the recycled
            # live set -- shedding a point and shrinking V every cycle on no new evidence at
            # all, which biases lnZ (Z ~ V*mean(w)).  Measured before this guard: 20 live
            # points and ln V decreasing monotonically -0.05, -0.11, -0.16, -0.22, ... over
            # chunks that each returned zero finite samples.
            n_new = len(idxsel[0])
            #only admit samples that lie inside the live volume, i.e. one that cross likelihood threshold
            allx = xpy_here.append(allx, rv[idxsel], axis = 0)
            allloglkl = xpy_here.append(allloglkl, loglkl[idxsel])
            allp = xpy_here.append(allp, log_joint_p_prior[idxsel])
            ninj = len(allloglkl)

            if _AV_TRACE:
                _lk = identity_convert(loglkl)
                _av_trace("cycle {}: drawn={} finite={} neginf={} posinf={} nan={} new={} ninj={} thr_in={:.6g}".format(
                    cycle, len(_lk), int(np.sum(np.isfinite(_lk))), int(np.sum(np.isneginf(_lk))),
                    int(np.sum(np.isposinf(_lk))), int(np.sum(np.isnan(_lk))), n_new, ninj, loglkl_thr))

            if n_new == 0:
                # This chunk contributed NO finite in-volume sample.  At high SNR the
                # production likelihood underflows to -inf more than ~745 nats below its
                # peak, so a chunk can hold nothing usable.  Contraction is an inference
                # FROM the chunk, so an empty chunk supports none: leave the threshold, V
                # and the grid untouched and draw again.  This is recoverable; the old code
                # instead either fell into get_likelihood_threshold and died on max() of an
                # empty array (no survivors yet) or contracted on recycled samples.
                n_empty_cycles += 1
                if not collapse_reported:
                    print("  [AV collapse] cycle {}: no finite in-volume samples ({} drawn, all -inf/NaN;"
                          " live set holds {}).".format(cycle, len(rv), ninj))
                    print("                Threshold and live volume left unchanged; continuing to draw.")
                    print("                This is the high-SNR likelihood-underflow regime --")
                    print("                see --sampler-warmstart-retry-neff.")
                    collapse_reported = True
                cycle += 1
                if cycle > 1000:
                    break
                continue

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

            _av_trace("cycle {}: after selection at thr={:.6g}: nrec={} (was ninj={})".format(
                cycle, loglkl_thr, nrec, ninj))

            if nrec == 0:
                # Defensive: get_likelihood_threshold now clamps the threshold below max(lkl),
                # so this is unreachable by the route that produced the reported crash.  Keep
                # the guard anyway -- an empty live set must never reach the reductions below.
                n_live_collapses += 1
                print("  [AV collapse] cycle {}: threshold {:.6g} emptied a live volume of {}; ".format(cycle, loglkl_thr, ninj)
                      + "restoring it and stopping contraction.")
                allloglkl, allp, allx = allloglkl_prev, allp_prev, allx_prev
                loglkl_thr = loglkl_thr_prev
                nrec = len(allloglkl)
                if nrec == 0:
                    raise LiveVolumeCollapse(
                        "adaptive-volume live set is empty after {} cycles: the likelihood returned no "
                        "finite value inside the sampled volume (high-SNR underflow, or a likelihood/"
                        "waveform failure).  This is NOT a waveform Nyquist/duration problem.".format(cycle))
                break

            # remember the last GOOD state, so a degenerate contraction can be undone
            allloglkl_prev, allp_prev, allx_prev, loglkl_thr_prev = allloglkl, allp, allx, loglkl_thr

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

        if len(allloglkl) == 0:
            # Every cycle came back empty: the integrand never returned a finite value inside
            # the sampled volume.  There is no integral to report, so fail with a message that
            # names the actual cause instead of an anonymous empty-array reduction.
            raise LiveVolumeCollapse(
                "adaptive-volume live set is empty after {} cycles ({} draws): the likelihood "
                "returned no finite value anywhere in the sampled volume.  At high network SNR "
                "this is likelihood UNDERFLOW (exp() of a lnL more than ~745 nats below the peak "
                "returns 0), not a waveform Nyquist/start-frequency/duration problem; narrow the "
                "extrinsic prior or seed the sampler (--sampler-warmstart-retry-neff).".format(
                    cycle, ntotal_true))

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

        # WARM-SEED RESERVE: keep a bounded copy of the points this pass actually RETAINED,
        # before the fair draw below overwrites self._rvs in place.
        #
        # That overwrite is why a warm start seeded from _rvs was starving.  The fair draw
        # takes n_extr = min(n_extr, 1.5*eff_samp, 1.5*neff) rows WITH REPLACEMENT and
        # REBINDS every _rvs key to that subset, so on the collapsed high-amplitude pass the
        # rescue is meant to fix -- eff_samp ~ 1 -- everything downstream sees ONE row, no
        # matter that the live set held a thousand.  Measured at rho_net 146.8: "Fairdraw
        # size : 1", and the rescue then reported a 1-point seed; at rho_net 102.8, 5 rows,
        # several of them the same point drawn twice, which is how a "5-point" seed came back
        # with affine rank 2 (and a "2-point" seed with rank 0 -- two copies of one point).
        # So the earlier reading of this failure, that "a collapsed cold pass never sampled
        # more than a handful of finite-likelihood points", was wrong: the points were drawn
        # and retained, then discarded by a resample meant for EXPORT, not for provenance.
        # It also explains why widening --sampler-sequential-warmstart-deltalnL could not
        # help -- there were only n_extr rows left to admit at any window.
        #
        # Bounded, because this is the array the surrounding code calls a memory hog: a
        # uniform subsample without replacement (plus the peak row, which the seed needs and
        # a subsample can drop) is all a seed or a scale estimate can use.
        try:
            self._warm_seed_reserve = make_warm_seed_reserve(
                allx, allloglkl - allp, self.params_ordered,
                n_max=getattr(self, 'n_warm_seed_reserve', 20000),
                log_joint_prior=allp,
                log_joint_s_prior=self._rvs['log_joint_s_prior'])
        except Exception as _e_res:
            # Provenance for a rescue, never a reason to lose a completed integral.
            self._warm_seed_reserve = None
            print("  [AV] warm-seed reserve not kept (", _e_res, ")")

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
           # Build the weights on the SAMPLER's backend (self.xpy), which is what draws below.
           # The module-global `xpy` kwarg defaults to cupy independently of self.xpy, so using it
           # here handed a device array to numpy.random.choice whenever the two disagreed.
           wt = self.xpy.exp(self.xpy.asarray(ln_wt))
           if n_extr < len(self._rvs["log_integrand"]):
               indx_list = self.xpy.random.choice(self.xpy.arange(len(wt)), size=n_extr,replace=True,p=wt) # fair draw
               # FIXME: See previous FIXME
               # Gather on the HOST.  _rvs entries are not guaranteed to sit on the same backend as
               # indx_list (a caller may set self.xpy independently of the module-level backend, and
               # keys written outside integrate_log arrive host-typed), and a numpy array indexed by
               # a cupy array raises "Implicit conversion to a NumPy array is not allowed" -- which
               # aborted this pass mid-way, leaving the caller's result tuple unassigned.  Converting
               # first is free: the block just below moves every array to the host anyway.
               indx_host = np.asarray(identity_convert(indx_list))
               for key in list(self._rvs.keys()):
                   arr = identity_convert(self._rvs[key])
                   if isinstance(key, tuple):
                       self._rvs[key] = arr[:,indx_host]
                   else:
                       self._rvs[key] = arr[indx_host]


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

        # ------------------------------------------------------------------
        # LIVE-VOLUME COLLAPSE VERDICT: a degenerate contraction must be REPORTED, not
        # silently exported.  Thresholds and the measured regimes they separate are
        # documented on live_volume_collapse_verdict.
        n_live_final = int(len(log_wt))
        _ess = dict_return.get('n_ESS', None)
        _khat_v = dict_return.get('pareto_khat', None)
        collapsed, _reasons = live_volume_collapse_verdict(
            n_live_final, ndim, ess=_ess, khat=_khat_v,
            n_empty_cycles=n_empty_cycles, n_live_collapses=n_live_collapses,
            n_warm_seed=n_warm_seed, n_warm_seed_rank=n_warm_seed_rank,
            n_warm_seed_dim=n_warm_seed_dim)
        dict_return['live_volume_collapsed'] = collapsed
        dict_return['n_live_final'] = n_live_final
        dict_return['n_empty_cycles'] = int(n_empty_cycles)
        dict_return['n_live_collapses'] = int(n_live_collapses)
        if n_warm_seed is not None:
            dict_return['n_warm_seed'] = int(n_warm_seed)
            if n_warm_seed_rank is not None:
                dict_return['n_warm_seed_rank'] = int(n_warm_seed_rank)
                dict_return['n_warm_seed_dim'] = int(n_warm_seed_dim or 0)
            dict_return['V_warm_start'] = float(V_warm)
        if collapsed:
            dict_return['collapse_reason'] = "; ".join(_reasons)
            print(" [AV COLLAPSE] the live volume degenerated: " + dict_return['collapse_reason'] + ".")
            print(" [AV COLLAPSE] lnZ and the exported samples describe a SINGLE mode of the integrand and are")
            print(" [AV COLLAPSE] NOT a fair draw from the posterior.  Do not use this export unweighted.")
            # The same test the verdict used -- rank when recorded, count at the simplex
            # boundary otherwise -- so the advice always matches the reason just given.
            _seed_degenerate = (
                (n_warm_seed_rank < n_warm_seed_dim)
                if (n_warm_seed_rank is not None and n_warm_seed_dim)
                else (n_warm_seed is not None and n_warm_seed <= (n_warm_seed_dim or ndim)))
            if _seed_degenerate:
                # OPPOSITE failure to the cold one below, so it needs the opposite advice: the
                # numbers look GOOD (n_eff at target in one cycle) precisely because the seeded
                # volume is too small for the integrand to vary across it.  lnZ is a lower bound.
                print(" [AV COLLAPSE] this is an OVER-CONTRACTED WARM START (V={:.3e}), not underflow:".format(V_warm))
                print(" [AV COLLAPSE] a healthy n_eff here is an artifact of a live volume too small to")
                print(" [AV COLLAPSE] resolve the peak, and lnZ is a LOWER BOUND missing the mass outside it.")
                print(" [AV COLLAPSE] Seed from more points (widen --sampler-sequential-warmstart-deltalnL,")
                print(" [AV COLLAPSE] or let the caller puff a thin peak) rather than trusting this pass.")
            else:
                print(" [AV COLLAPSE] At high network SNR this is likelihood underflow over a cold extrinsic prior;")
                print(" [AV COLLAPSE] narrow the prior or seed the sampler (--sampler-warmstart-retry-neff).")

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
        n_adapt -- IGNORED as an adaptation schedule by this sampler; accepted only for API
            compatibility with mcsampler/mcsamplerGPU, where it does gate update_sampling_prior.
            AV has no update_sampling_prior: its volume adaptation is intrinsic to the algorithm
            and runs every cycle regardless of this value. (Verified: output is bit-identical for
            n_adapt in {10,100,1000}.) The ONLY thing it still does here is participate in the
            save_intg gate, so n_adapt=0 can still suppress the _rvs cache. Do not reach for this
            expecting an "adapt then freeze" control -- there isn't one.
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
