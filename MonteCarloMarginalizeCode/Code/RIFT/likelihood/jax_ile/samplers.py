"""Robust *multimodal* samplers for the JAX distance-marginalized extrinsic
likelihood.

The extrinsic posterior over the 5 angular parameters
``theta5 = (ra, dec, psi, incl, phiref)`` is strongly multimodal: the detector
time-delay sky ring produces several discrete sky "blobs", and there are
additional phase (``phiref``) + polarization (``psi``) degeneracies.  A single
gradient-based chain seeded from one start finds only one mode, so the global
evidence integral has poor effective sample size.

This module provides two strategies that use the AD-compatible likelihood
(``RIFT.likelihood.jax_ile.wrapper.JAXDistanceMarginalizedLikelihood``) and its
exact gradients to cover *all* the modes:

* :func:`multistart_nuts` -- a pilot prior scan picks several well-separated,
  high-lnL seeds (one per mode), runs a gradient-based NUTS chain from each, and
  pools the draws.  A Gaussian-*mixture* importance proposal (one component per
  seed) moment-matched to the pooled draws then yields a high-``neff`` evidence
  estimate.
* :func:`flowmc_sample` -- a normalizing-flow sampler (flowMC) that trains a flow
  to the full multimodal geometry in a single run.

The priors are the standard physical ones (uniform sky/orientation):
``ra ~ U(0, 2pi)``, ``sin(dec) ~ U(-1, 1)``, ``psi ~ U(0, pi)``,
``cos(incl) ~ U(-1, 1)``, ``phiref ~ U(0, 2pi)``; the target posterior is
``propto exp(lnL) * prior``.

Run the self-test (builds the standard synthetic injection, no frames needed)::

    PYTHONPATH=<...>/MonteCarloMarginalizeCode/Code \
        python RIFT/likelihood/jax_ile/samplers.py
"""

import os

import numpy as np

import jax
import jax.numpy as jnp

# Default chunk for the batched lnL evals.  The per-sample distance quadrature
# (JAX_ILE_DISTMARG_GH=G) materialises a (chunk, npts, G) array, ~G/ (grid_block)
# more device memory than the legacy grid, so a 4000-row chunk OOMs the 11GB
# 2080Ti.  Shrink the chunk when per-sample is active so the per-sample path fits
# small-VRAM GPUs too (don't force every high-SNR job onto a 24GB node).
_GH_NODES = int(os.environ.get("JAX_ILE_DISTMARG_GH", "0"))
_EVAL_CHUNK = max(500, 4000 * 16 // max(16, _GH_NODES)) if _GH_NODES > 0 else 4000

# Parameter order used everywhere in this module.
ANG_NAMES = ("ra", "dec", "psi", "incl", "phiref")
# 4-D order when phi_ref is marginalised out.
ANG_NAMES_NOPHI = ("ra", "dec", "psi", "incl")
_TWO_PI = float(2 * np.pi)
_PI = float(np.pi)


# ---------------------------------------------------------------------------
# Likelihood tempering: what --adapt-weight-exponent costs on THIS path
# ---------------------------------------------------------------------------
# Decision + provenance: DESIGN_jax_tempering.md, beside this module (2026-08-23).
# The short version, because it is the thing that gets ported wrong:
#
#   non-JAX RIFT  beta shapes only the adaptive sampling PRIOR
#                 (mcsamplerGPU: log_weights = beta*lnL + ln p - ln p_s, while the
#                 estimator stays log_integrand = lnL + ln p - ln p_s).  Unbiased
#                 at any beta; beta costs nothing in exported samples.
#   JAX flowMC    beta = inv_T is the exponent of the target the MCMC SAMPLES.
#                 The draws are the deliverable, so the export must be reweighted
#                 by L^(1-beta) (post_weight) -- and that reweight has an ESS cost
#                 the non-JAX path never pays.
#
# helper_LDG_Events.py keys its beta on SNR.  That is right there and wrong here:
# the cost below is set by the SAMPLED DIMENSION and is independent of lnLmax.
# Measured ratio (measured ESS/N) / (Gaussian law) at dim=4, as a CONSERVATIVE
# piecewise-linear envelope: every knot sits at or below every measured point of
# the sweep in DESIGN_jax_tempering.md.  The law is optimistic and increasingly so
# at small beta, so a single flat factor is the wrong shape -- a flat 0.79 would
# also make any target above 0.79 unachievable, since it never reaches 1.
#
# MEASURED AT dim=4 ONLY.  Applying the same ratio at other dimensions is an
# assumption, not a measurement; it is the conservative direction (the law is
# optimistic in dim too, since the exponent grows), but it is not verified.
_ESS_CAL_BETA = (0.05, 0.20, 0.40, 0.60, 0.80, 1.00)
_ESS_CAL_RATIO = (0.79, 0.79, 0.83, 0.91, 0.97, 1.00)
_TEMPER_ESS_LAW_CAL = _ESS_CAL_RATIO[0]   # worst case, retained for reference


def export_ess_fraction(beta, n_dim):
    """Fraction of a beta-tempered cloud that survives the post_weight reweight.

    For a locally Gaussian peak ``ln Z_g = g lnLmax - (n_dim/2) ln g + const``,
    so the self-normalised reweight ``L^(1-beta)`` from the tempered target
    ``L^beta pi`` back to the posterior has

        ESS/N = Z_1^2 / (Z_beta Z_{2-beta}) = [beta (2 - beta)]^(n_dim/2)

    **It depends on n_dim and NOT on lnLmax** -- i.e. not on SNR.  That is the
    whole reason the non-JAX helper's SNR-keyed exponent must not be carried
    over to this path.

    Measured against the real phi-marginalised BNS likelihood (SNR 23.8, 4-D),
    the law holds to a ratio 0.79-1.00 over beta in [0.05, 1]; it is therefore a
    slightly OPTIMISTIC closed form.  Callers sizing a budget should apply
    ``_TEMPER_ESS_LAW_CAL``.  Provenance and the full sweep: DESIGN_jax_tempering.md.
    """
    beta = float(beta)
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must be in (0, 1]; got %r" % (beta,))
    return float((beta * (2.0 - beta)) ** (0.5 * int(n_dim)))


def _ess_law_calibration(beta):
    """Conservative lower-bound ratio (measured/law) at this beta, from the sweep."""
    beta = float(beta)
    if beta <= _ESS_CAL_BETA[0]:
        return _ESS_CAL_RATIO[0]
    if beta >= _ESS_CAL_BETA[-1]:
        return _ESS_CAL_RATIO[-1]
    return float(np.interp(beta, _ESS_CAL_BETA, _ESS_CAL_RATIO))


def export_ess_estimate(beta, n_dim):
    """Calibrated ESTIMATE of the surviving export fraction.  NOT a bound.

    ``export_ess_fraction`` is the Gaussian-peak law; the SNR-23.8 sweep shows it
    optimistic by up to 21% (ratio 0.79 at beta=0.05, rising to 1.00 at beta=1),
    and this applies that ratio.

    IT IS NOT A GUARANTEE, AND THIS FUNCTION WAS ONCE NAMED AS IF IT WERE.  The
    calibration is fitted at SNR ~= 23.8 only, and the shortfall grows with SNR:
    the SNR ladder in DESIGN_jax_tempering.md measures ESS/N = 0.00823 at
    beta=0.1, d=4, SNR ~= 67, against 0.0285 from this estimate -- a factor 3.5
    the wrong way.  A real bound needs a calibration in (beta, SNR), and this
    driver has no trustworthy SNR at the point the choice is made (`guess_snr` is
    an explicit guesstimate: 10.32 against a true network 23.78 on the study
    event).  So callers must treat the result as advisory and must not refuse a
    run on it.  Reported by review on #186.
    """
    return _ess_law_calibration(beta) * export_ess_fraction(beta, n_dim)


def beta_for_export_ess(target_frac, n_dim):
    """Inverse of :func:`export_ess_fraction`: the SMALLEST beta meeting a target.

    Solves ``[beta(2-beta)]^(n_dim/2) = target_frac`` on ``beta in (0, 1]``:

        beta = 1 - sqrt(1 - target_frac^(2/n_dim))

    Smallest is the useful root: beta is a breadth knob, so among exponents that
    meet the export budget the broadest target is the one that explores most.

    Solves against :func:`export_ess_estimate`, so the returned beta meets
    ``target_frac`` **on the SNR ~= 23.8 calibration**.  That is not a guarantee
    at other SNRs -- the shortfall grows with SNR (see that function) -- which is
    why ``--auto-adapt-weight-exponent`` is documented as experimental and why
    nothing refuses a run on this number.
    """
    t = float(target_frac)
    if not (0.0 < t <= 1.0):
        raise ValueError("target_frac must be in (0, 1]; got %r" % (t,))
    # Solve on the CALIBRATED lower bound, not the bare law.  Both factors are
    # non-decreasing in beta, so the product is monotone and bisection is safe.
    # There is no closed form once the piecewise-linear calibration is included,
    # and the previous closed-form inverse of the optimistic law returned betas
    # that retained less than the caller asked for.
    if export_ess_estimate(1.0, n_dim) < t:
        raise ValueError(
            "target_frac %g is unreachable in %d-D even at beta=1 (lower bound "
            "%.4f)" % (t, int(n_dim), export_ess_estimate(1.0, n_dim)))
    lo, hi = 1e-6, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if export_ess_estimate(mid, n_dim) >= t:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ---------------------------------------------------------------------------
# Prior (physical, uniform sky + orientation), numpy + JAX flavors
# ---------------------------------------------------------------------------
def sample_prior(n, rng):
    """Draw ``n`` prior samples of ``theta5`` (uniform sky/orientation).

    Returns an ``(n, 5)`` array ``[ra, dec, psi, incl, phiref]``.
    """
    ra = rng.uniform(0.0, _TWO_PI, n)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, n))
    psi = rng.uniform(0.0, _PI, n)
    incl = np.arccos(rng.uniform(-1.0, 1.0, n))
    phiref = rng.uniform(0.0, _TWO_PI, n)
    return np.stack([ra, dec, psi, incl, phiref], axis=-1)


def log_prior(theta):
    """log of the physical prior density on ``theta5`` (numpy, batched).

    ``-inf`` outside the support.  Includes the ``cos(dec)`` and ``sin(incl)``
    Jacobians from the uniform-on-sphere / uniform-cos parameterizations.
    """
    theta = np.atleast_2d(theta)
    ra, dec, psi, incl, phiref = [theta[..., i] for i in range(5)]
    inb = ((ra >= 0) & (ra <= _TWO_PI) & (dec >= -_PI / 2) & (dec <= _PI / 2)
           & (psi >= 0) & (psi <= _PI) & (incl >= 0) & (incl <= _PI)
           & (phiref >= 0) & (phiref <= _TWO_PI))
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = (np.log(np.cos(dec)) - np.log(2.0)
                + np.log(np.sin(incl)) - np.log(2.0)
                - np.log(_TWO_PI) - np.log(_PI) - np.log(_TWO_PI))
    return np.where(inb, logp, -np.inf)


def _log_prior_jax(theta5):
    """JAX-traceable log-prior of a single length-5 ``theta5`` vector.

    Used inside the flowMC target.  Returns ``-inf`` (large negative) outside
    the support so the sampler is repelled from it.
    """
    ra, dec, psi, incl, phiref = (theta5[0], theta5[1], theta5[2],
                                  theta5[3], theta5[4])
    inb = ((ra >= 0) & (ra <= _TWO_PI) & (dec >= -_PI / 2) & (dec <= _PI / 2)
           & (psi >= 0) & (psi <= _PI) & (incl >= 0) & (incl <= _PI)
           & (phiref >= 0) & (phiref <= _TWO_PI))
    logp = (jnp.log(jnp.cos(dec)) - jnp.log(2.0)
            + jnp.log(jnp.sin(incl)) - jnp.log(2.0)
            - jnp.log(_TWO_PI) - jnp.log(_PI) - jnp.log(_TWO_PI))
    return jnp.where(inb, logp, -1e30)


# ---------------------------------------------------------------------------
# Batched lnL evaluation (chunked to bound memory)
# ---------------------------------------------------------------------------
# Largest single XLA buffer of the anglemarg laplace path, per sample per
# time point: the (quad_chunk=16, dist_block=4, phi_chunk=16) stacked
# quadrature block, 16*4*16*8 = 8192 bytes.  Measured 2026-08-28: at the
# default chunk 4000 with npts=1193 XLA requested exactly 36.41 GiB for that
# buffer and the SNR-40 acceptance run died RESOURCE_EXHAUSTED on a 25 GiB
# cgroup -- the pre-fix code never got past COMPILATION at production size,
# so this execution-side wall was previously unreachable.  The exact scheme's
# dense reconstruction has the same batch-multiplied structure (smaller
# constant); the laplace constant is used for both as the worst case.
_ANGLE_MARG_BYTES_PER_SAMPLE_PT = 8192
_ANGLE_MARG_BUFFER_TARGET = 4 << 30      # ~4 GiB largest single buffer


def angle_marg_eval_chunk(like, chunk):
    """Cap the batched-eval chunk when ``like`` runs an anglemarg scheme.

    Slices of the batched eval are INDEPENDENT (lnL is elementwise in the
    sample axis), so this changes peak memory and nothing else -- same
    pattern as the _GH_NODES shrink above.  Grid-scheme and 4/5-param
    likelihoods pass through unchanged.
    """
    if getattr(like, "angle_marg_scheme", "grid") not in ("exact", "laplace"):
        return chunk
    npts = int(getattr(getattr(like, "data", None), "npts", 0) or 0)
    if npts <= 0:
        return chunk
    cap = max(64, _ANGLE_MARG_BUFFER_TARGET
              // (_ANGLE_MARG_BYTES_PER_SAMPLE_PT * npts))
    return min(chunk, cap)


def eval_lnL(like, theta, chunk=_EVAL_CHUNK):
    """Evaluate the distance-marginalized lnL on an ``(N, 5)`` array in chunks.

    Chunking bounds peak device memory (the distance grid multiplies the batch
    dimension inside the likelihood).
    """
    theta = np.atleast_2d(theta)
    chunk = angle_marg_eval_chunk(like, chunk)
    N = theta.shape[0]
    out = np.empty(N)
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        cols = [theta[sl, j] for j in range(5)]
        out[sl] = np.asarray(like.log_likelihood(*cols))
    return out


# ---------------------------------------------------------------------------
# Angular distance (for well-separated seed selection / mode clustering)
# ---------------------------------------------------------------------------
def _sky_angle_distance(t1, t2):
    """A distance between two ``theta5`` points capturing mode separation.

    Combines the great-circle sky separation (the dominant mode structure, from
    the time-delay ring) with wrapped distances in the orientation angles.
    All terms are in radians; the sky term is weighted up because the discrete
    modes are primarily a sky-location phenomenon.
    """
    ra1, dec1 = t1[0], t1[1]
    ra2, dec2 = t2[0], t2[1]
    # great-circle (haversine) sky separation
    dlon = ra1 - ra2
    sky = np.arccos(np.clip(
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(dlon), -1.0, 1.0))

    def wrap(d, period):
        d = np.abs(d) % period
        return np.minimum(d, period - d)

    dpsi = wrap(t1[2] - t2[2], _PI)
    dincl = np.abs(t1[3] - t2[3])
    dphi = wrap(t1[4] - t2[4], _TWO_PI)
    return np.sqrt((2.0 * sky) ** 2 + dpsi ** 2 + dincl ** 2 + dphi ** 2)


def _pick_well_separated(theta, lnL, n_starts, min_sep=0.3):
    """Greedy farthest-point selection of high-lnL, well-separated seeds.

    Start from the global-best draw, then repeatedly add the highest-lnL
    remaining draw whose angular distance to *every* already-chosen seed exceeds
    ``min_sep``.  If fewer than ``n_starts`` such draws exist (few resolvable
    modes), relax by appending the next-highest-lnL draws regardless of
    separation, so we always return ``n_starts`` seeds.

    Returns ``(seeds (k,5), seed_lnL (k,))`` with ``k == n_starts``.
    """
    order = np.argsort(lnL)[::-1]
    chosen = [order[0]]
    for idx in order[1:]:
        if len(chosen) >= n_starts:
            break
        far = all(_sky_angle_distance(theta[idx], theta[c]) > min_sep
                  for c in chosen)
        if far:
            chosen.append(idx)
    # top-up if too few well-separated modes were found
    if len(chosen) < n_starts:
        chosen_set = set(int(c) for c in chosen)
        for idx in order:
            if len(chosen) >= n_starts:
                break
            if int(idx) not in chosen_set:
                chosen.append(idx)
                chosen_set.add(int(idx))
    chosen = np.array(chosen[:n_starts])
    return theta[chosen], lnL[chosen]


def cluster_modes(theta, min_sep=0.5):
    """Greedily cluster ``theta5`` rows into distinct modes by angular distance.

    Returns a list of ``(representative_theta5, member_indices)`` tuples, sorted
    by cluster size (largest first).  Useful for reporting the distinct sky
    modes found by a sampler.
    """
    n = theta.shape[0]
    reps = []          # representative theta of each cluster
    members = []       # list of index lists
    for i in range(n):
        placed = False
        for c, rep in enumerate(reps):
            if _sky_angle_distance(theta[i], rep) <= min_sep:
                members[c].append(i)
                placed = True
                break
        if not placed:
            reps.append(theta[i])
            members.append([i])
    order = np.argsort([-len(m) for m in members])
    return [(reps[c], np.array(members[c])) for c in order]


# ---------------------------------------------------------------------------
# Evidence helpers (same math as the bin/ driver)
# ---------------------------------------------------------------------------
def evidence_from_logweights(logw):
    """``(logZ, sigma/Z, neff)`` for ``Z = E[w]`` from log importance weights."""
    logw = np.asarray(logw)
    fin = np.isfinite(logw)
    logw = logw[fin]
    if logw.size == 0:
        return -np.inf, np.inf, 0.0
    m = np.max(logw)
    w = np.exp(logw - m)
    n = len(w)
    Zhat = np.mean(w)
    logZ = m + np.log(Zhat)
    sigma_over_Z = np.sqrt(np.var(w) / n) / Zhat
    neff = (np.sum(w) ** 2) / np.sum(w ** 2)
    return logZ, sigma_over_Z, neff


def _gaussian_logq(theta, mu, cov):
    """log density of a multivariate Gaussian ``N(mu, cov)`` at rows of theta."""
    diff = theta - mu[None, :]
    sol = np.linalg.solve(cov, diff.T).T
    return (-0.5 * np.einsum("ij,ij->i", diff, sol)
            - 0.5 * theta.shape[1] * np.log(2 * np.pi)
            - 0.5 * np.linalg.slogdet(cov)[1])


def _moment_match(theta, logL):
    """Weighted mean/cov of ``theta`` with weights ``propto exp(logL)``.

    The covariance eigenvalues are floored at a small *relative* level so the
    proposal stays invertible/well-conditioned even when the posterior is
    extremely narrow (high SNR) or degenerate in some direction -- otherwise the
    Gaussian log-density's ``slogdet`` term blows up.
    """
    w = np.exp(logL - np.max(logL))
    w = w / np.sum(w)
    mu = np.sum(w[:, None] * theta, axis=0)
    d = theta - mu[None, :]
    cov = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(axis=0)
    cov = 0.5 * (cov + cov.T)
    evals, V = np.linalg.eigh(cov)
    emax = float(np.max(evals)) if evals.size else 0.0
    floor = max(1e-10 * emax, 1e-300)
    evals = np.clip(evals, floor, None)
    cov = (V * evals) @ V.T
    return mu, cov


def _finalize_evidence(logZ, sigma_over_Z, neff, max_lnL):
    """Flag an importance-evidence estimate as unreliable (nan) when it cannot
    be trusted: log Z must satisfy ``log Z <= lnL_max`` for a normalized prior
    (Z = E_prior[L] <= L_max), and a low ``neff`` means the proposal failed to
    bracket the (possibly sub-resolution-narrow) peak."""
    if (not np.isfinite(logZ)) or (np.isfinite(max_lnL) and logZ > max_lnL + 5.0) \
            or (np.isfinite(neff) and neff < 1.5):
        return np.nan, np.nan, neff
    return logZ, sigma_over_Z, neff


def _mixture_logq(theta, mus, covs, weights):
    """log density of a Gaussian mixture at rows of ``theta``."""
    logws = np.log(np.asarray(weights))
    comps = np.stack([_gaussian_logq(theta, mu, cov) + lw
                      for mu, cov, lw in zip(mus, covs, logws)], axis=0)
    m = np.max(comps, axis=0)
    return m + np.log(np.sum(np.exp(comps - m[None, :]), axis=0))


# ---------------------------------------------------------------------------
# 1. multistart NUTS
# ---------------------------------------------------------------------------
def multistart_nuts(like, d_min, d_max, n_starts=8, num_warmup=300,
                    num_samples=500, n_prior_pilot=8000, seed=0,
                    target_accept=0.8, min_sep=0.3, proposal_inflate=2.0,
                    n_is=40000, sky_coords="equatorial",
                    dense_mass=True, max_tree_depth=10, rotate_phase=False,
                    polish_seeds=True, extra_seeds=None,
                    verbose=False, chain_progress_bar=False):
    """Multimodal posterior sampling by multi-start gradient-based NUTS.

    A pilot prior scan locates several well-separated, high-lnL seeds (one per
    resolvable mode of the multimodal extrinsic posterior); a numpyro NUTS chain
    is run from each (each climbing/sampling its own mode via the exact JAX
    gradient of the distance-marginalized lnL); the draws are pooled.  A
    Gaussian-mixture importance proposal (one component per seed, moment-matched
    to that seed's chain) then gives a high-``neff`` evidence estimate.

    Parameters
    ----------
    like : JAXDistanceMarginalizedLikelihood
        Must expose ``log_likelihood`` (batched) and ``_scalar`` (JAX-traceable
        scalar lnL of a length-5 jnp array).
    d_min, d_max : float
        Distance bounds (accepted for interface symmetry; the distance
        marginalization is already baked into ``like``).
    n_starts : int
        Number of NUTS chains / candidate modes.
    num_warmup, num_samples : int
        Per-chain NUTS warmup and posterior sample counts.
    n_prior_pilot : int
        Prior draws used to find the seeds.
    seed : int
        Base PRNG seed (numpy + JAX).
    target_accept : float
        NUTS target acceptance probability.
    dense_mass : bool
        Adapt a FULL (dense) mass matrix during warmup instead of the default
        diagonal one.  The distance-marginalized angular posterior is strongly
        correlated -- at high SNR it is a thin, curved sky ring entangled with
        the psi/incl/phiref degeneracies -- so a diagonal mass matrix leaves the
        Hamiltonian geometry wildly anisotropic and NUTS hits ``max_tree_depth``
        (~2^depth leapfrog steps) on essentially every sample, stalling the run.
        A dense mass matrix ≈ the inverse posterior covariance whitens the
        geometry so trajectories are short and acceptance is high.  ``True`` is
        the sane production default for this problem; only set ``False`` for a
        deliberately cheap low-SNR run where the posterior is broad and round.
    max_tree_depth : int or (int, int)
        NUTS maximum tree depth (numpyro passthrough).  Bounds the worst-case
        leapfrog steps per sample (``2^depth``) so an ill-conditioned *early*
        warmup window -- before the mass matrix has adapted -- cannot blow up
        wall-clock.  A ``(warmup_depth, sampling_depth)`` tuple caps warmup more
        tightly than sampling; a scalar applies to both.
    rotate_phase : bool
        Sample the rotated "polarization-phase" coordinates
        ``phase_p = phiref + psi`` and ``phase_m = phiref - psi`` (each over
        ``[0, 4pi)``) instead of ``(psi, phiref)`` directly, then map back
        ``psi = (phase_p - phase_m)/2``, ``phiref = (phase_p + phase_m)/2``.
        This is the JAX mirror of production RIFT's ``--internal-rotate-phase``:
        the quadrupole-dominated likelihood depends on ``2psi +/- 2phiref``, so
        the curved psi/phiref degeneracy ridge becomes AXIS-ALIGNED in
        ``(phase_p, phase_m)`` -- the sampler's (dense) mass matrix is then
        near-diagonal and NUTS keeps a healthy step at high SNR.  The map is a
        constant-Jacobian rotation, so the flat prior is preserved (exactly, in
        the enlarged periodic domain).  Combine with ``sky_coords="network"``
        (which similarly straightens the sky time-delay ring) for the full
        high-SNR reparameterization.  Exact for the (2,+/-2) quadrupole; still a
        valid (just less-perfectly-decorrelating) reparameterization with higher
        modes.
    polish_seeds : bool
        Gradient-ascend (+ Newton) each pilot seed to its local MAP before
        running NUTS.  At very high SNR the sky posterior is a ~1/SNR-thin ring
        that a finite prior pilot cannot land ON -- the best raw pilot draw sits
        many nats below the peak (e.g. SNR=1000: seed lnL ~24500 below
        0.5<d|d>), so a chain started there samples the wrong arc (MAP degrees
        off truth).  A few hundred AD-gradient steps + Fisher-inverse Newton
        steps climb from the broad basin onto the true peak, so NUTS starts AT
        the needle -- the whole point of having exact gradients.  Cheap
        (a few hundred grad evals per seed); default on.
    min_sep : float
        Minimum angular separation (radians, in the combined sky+angle metric)
        between seeds.
    proposal_inflate : float
        Covariance inflation for the importance proposal components.
    n_is : int
        Total importance-sampling draws for the evidence estimate.
    verbose : bool
        Print high-level progress (pilot, per-chain, seeds).
    chain_progress_bar : bool
        Show the numpyro per-chain tqdm progress bar (noisy; off by default).

    Returns
    -------
    dict with keys
        ``theta``     : ``(N, 5)`` pooled posterior draws ``[ra,dec,psi,incl,phiref]``.
        ``lnL``       : ``(N,)`` lnL recomputed at the pooled draws.
        ``seeds``     : ``(n_starts, 5)`` seed thetas.
        ``seed_lnL``  : ``(n_starts,)`` lnL at the seeds.
        ``logZ``      : importance-sampling log-evidence estimate.
        ``sigma_over_Z`` : relative statistical error of ``Z``.
        ``neff``      : effective sample size of the evidence estimator.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value

    rng = np.random.default_rng(seed)

    # -- 1. pilot prior scan -> well-separated high-lnL seeds ---------------
    pilot = sample_prior(n_prior_pilot, rng)
    pilot_lnL = eval_lnL(like, pilot)
    seeds, seed_lnL = _pick_well_separated(pilot, pilot_lnL, n_starts,
                                           min_sep=min_sep)
    if verbose:
        print("  pilot %d prior draws; max lnL=%.2f" %
              (n_prior_pilot, pilot_lnL.max()))
        print("  chose %d seeds (lnL): %s" %
              (len(seeds), np.array2string(seed_lnL, precision=1)))

    # Optional caller-supplied seeds (each a length-5 (ra,dec,psi,incl,phiref)),
    # PREPENDED to the pilot seeds before the polish.  At very high SNR the true
    # peak is thinner than 1 pilot draw can resolve (~(1/SNR)^2 of the sky), so a
    # blind pilot + gradient polish can settle on a secondary mode nats below the
    # global peak; a known seed near the true basin (in production: the intrinsic
    # grid + coarse extrinsic pass; here: the injected truth) guarantees one chain
    # characterizes the injected mode.  Still polished, so it snaps to the exact MAP.
    if extra_seeds is not None:
        ex = np.atleast_2d(np.asarray(extra_seeds, dtype=float))
        seeds = np.vstack([ex, seeds])
        seed_lnL = np.concatenate([eval_lnL(like, ex), seed_lnL])
        if verbose:
            print("  + %d caller seed(s) (lnL): %s" %
                  (len(ex), np.array2string(eval_lnL(like, ex), precision=1)))

    # Gradient MAP-polish: climb each raw pilot seed onto the true (1/SNR-thin)
    # peak so NUTS starts AT the needle rather than degrees off it on the wrong
    # arc.  Uses the exact JAX gradient (+ Fisher-inverse Newton); cheap.  Keeps
    # each seed at its OWN local MAP (preserves the multi-start mode coverage).
    if polish_seeds:
        _, _, _pol = _map_polish_4(like, seeds, bounds=_BOUNDS5)
        seeds = np.array([p[0] for p in _pol])
        seed_lnL = np.array([p[1] for p in _pol])
        if verbose:
            print("  polished seeds to MAP (lnL): %s"
                  % np.array2string(seed_lnL, precision=1))

    # Optional: sample the sky in NETWORK-frame coordinates (polar axis = the
    # baseline of the first two detectors), which folds the time-delay ring onto
    # a constant-polar-angle line.  Falls back to equatorial if <2 detectors.
    net = None
    if sky_coords == "network":
        names = like.data.detector_names
        if len(names) >= 2:
            from . import coordinates as _C
            loc1 = np.asarray(like.data.detectors[names[0]]["location"])
            loc2 = np.asarray(like.data.detectors[names[1]]["location"])
            R = _C.build_network_frame(loc1, loc2, like.data.gmst)
            net = (_C, R, float(like.data.gmst))
            if verbose:
                print("  sky sampled in network frame (baseline %s-%s)"
                      % (names[0], names[1]))
        elif verbose:
            print("  network sky coords requested but <2 detectors; equatorial")

    # numpyro model + per-seed init/extract helpers.  sin_dec/cos_incl (or
    # cos_theta_n/phi_n in the network frame) keep the sampler in a
    # well-conditioned space with the prior Jacobians handled automatically;
    # the uniform sky prior is uniform in (cos_theta_n, phi_n) too, since the
    # rotation preserves the sphere measure.
    # Shared phase parameterization: either sample (psi, phiref) directly, or
    # the rotated (phase_p, phase_m) = (phiref+psi, phiref-psi) that decorrelate
    # the 2psi+/-2phiref degeneracy (production --internal-rotate-phase).
    _4PI = 4.0 * _PI

    def _sample_phase():
        if rotate_phase:
            pp = numpyro.sample("phase_p", dist.Uniform(0.0, _4PI))
            pm = numpyro.sample("phase_m", dist.Uniform(0.0, _4PI))
            return (pp - pm) * 0.5, (pp + pm) * 0.5      # psi, phiref
        psi = numpyro.sample("psi", dist.Uniform(0.0, _PI))
        phiref = numpyro.sample("phiref", dist.Uniform(0.0, _TWO_PI))
        return psi, phiref

    def _init_phase(th0):
        psi0, phi0 = float(th0[2]), float(th0[4])
        if rotate_phase:
            return {"phase_p": (phi0 + psi0) % _4PI,
                    "phase_m": (phi0 - psi0) % _4PI}
        return {"psi": psi0, "phiref": phi0}

    def _extract_phase(s):
        if rotate_phase:
            pp = np.asarray(s["phase_p"]); pm = np.asarray(s["phase_m"])
            return np.mod((pp - pm) * 0.5, _PI), np.mod((pp + pm) * 0.5, _TWO_PI)
        return np.mod(np.asarray(s["psi"]), _PI), np.mod(np.asarray(s["phiref"]), _TWO_PI)

    if net is None:
        def model():
            ra = numpyro.sample("ra", dist.Uniform(0.0, _TWO_PI))
            sin_dec = numpyro.sample("sin_dec", dist.Uniform(-1.0, 1.0))
            cos_incl = numpyro.sample("cos_incl", dist.Uniform(-1.0, 1.0))
            psi, phiref = _sample_phase()
            lnL = like._scalar(jnp.stack(
                [ra, jnp.arcsin(sin_dec), psi, jnp.arccos(cos_incl), phiref]))
            numpyro.factor("loglike", lnL)

        def make_init(th0):
            d = {"ra": float(th0[0]), "sin_dec": float(np.sin(th0[1])),
                 "cos_incl": float(np.cos(th0[3]))}
            d.update(_init_phase(th0))
            return d

        def extract(s):
            psi, phiref = _extract_phase(s)
            return np.stack([np.asarray(s["ra"]), np.arcsin(np.asarray(s["sin_dec"])),
                             psi, np.arccos(np.asarray(s["cos_incl"])),
                             phiref], axis=-1)
    else:
        _C, R, gmst = net

        def model():
            cos_tn = numpyro.sample("cos_theta_n", dist.Uniform(-1.0, 1.0))
            phi_n = numpyro.sample("phi_n", dist.Uniform(0.0, _TWO_PI))
            cos_incl = numpyro.sample("cos_incl", dist.Uniform(-1.0, 1.0))
            psi, phiref = _sample_phase()
            ra, dec = _C.network_to_equatorial(jnp.arccos(cos_tn), phi_n, R, gmst)
            lnL = like._scalar(jnp.stack([ra, dec, psi, jnp.arccos(cos_incl), phiref]))
            numpyro.factor("loglike", lnL)

        def make_init(th0):
            tn, pn = _C.equatorial_to_network(float(th0[0]), float(th0[1]), R, gmst)
            d = {"cos_theta_n": float(np.cos(float(tn))),
                 "phi_n": float(float(pn) % _TWO_PI),
                 "cos_incl": float(np.cos(th0[3]))}
            d.update(_init_phase(th0))
            return d

        def extract(s):
            tn = np.arccos(np.asarray(s["cos_theta_n"]))
            ra, dec = _C.network_to_equatorial(tn, np.asarray(s["phi_n"]), R, gmst)
            psi, phiref = _extract_phase(s)
            return np.stack([np.asarray(ra), np.asarray(dec), psi,
                             np.arccos(np.asarray(s["cos_incl"])),
                             phiref], axis=-1)

    # -- 2. one NUTS chain per seed, pooled --------------------------------
    # (len(seeds) may exceed n_starts when the caller supplies extra_seeds)
    per_chain = []     # (num_samples, 5) per seed, in equatorial theta5
    n_chains = len(seeds)
    for k in range(n_chains):
        kernel = NUTS(model, target_accept_prob=target_accept,
                      dense_mass=dense_mass, max_tree_depth=max_tree_depth,
                      init_strategy=init_to_value(values=make_init(seeds[k])))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=chain_progress_bar)
        mcmc.run(jax.random.PRNGKey(seed + 1 + k))
        per_chain.append(extract(mcmc.get_samples()))
        if verbose:
            print("  chain %d/%d done (seed lnL=%.2f)" %
                  (k + 1, n_chains, seed_lnL[k]))

    theta = np.concatenate(per_chain, axis=0)
    lnL = eval_lnL(like, theta)

    # Evidence-weighted pooling.  Multi-start places one chain per mode, but the
    # modes carry vastly different posterior mass -- at high SNR the sub-dominant
    # time-delay-ring images and amplitude-degeneracy branches sit many nats below
    # the true peak.  Pooling the chains with EQUAL weight over-represents those
    # negligible modes (they dominate a naive credible-region or corner plot).
    # Weight each chain by a LAPLACE estimate of its mode evidence,
    #   log Z_k ~= peak_lnL_k + 1/2 log det Sigma^sky_k ,
    # i.e. the mode's peak likelihood times its (sky) width.  Peak alone is wrong:
    # at LOW SNR the chains sample one broad, overlapping posterior with similar
    # peaks, and a tiny peak difference would spuriously collapse it -- the width
    # term keeps broad modes comparable there, while at HIGH SNR the sub-dominant
    # modes are suppressed by their far-lower peak regardless of width.  ``theta``
    # stays the raw pooled draws; ``post_weight`` is the per-sample posterior weight
    # callers use for credible regions, sky areas, and corner plots.
    n_per = [len(c) for c in per_chain]
    _off = np.cumsum([0] + n_per)
    logev = np.full(len(per_chain), -np.inf)
    for k in range(len(per_chain)):
        if n_per[k] < 3:
            continue
        thk = per_chain[k]
        ra_k, dec_k = thk[:, 0], thk[:, 1]
        ra0 = np.angle(np.mean(np.exp(1j * ra_k)))
        x = ((ra_k - ra0 + np.pi) % (2 * np.pi) - np.pi) * np.cos(np.median(dec_k))
        y = dec_k - np.median(dec_k)
        cov = np.cov(np.vstack([x, y])) + 1e-8 * np.eye(2)
        logdet = float(np.log(max(np.linalg.det(cov), 1e-30)))
        peak_k = float(lnL[_off[k]:_off[k + 1]].max())
        logev[k] = peak_k + 0.5 * logdet
    cw = np.exp(logev - np.max(logev))
    post_weight = np.concatenate([
        np.full(n_per[k], cw[k] / max(n_per[k], 1)) for k in range(len(per_chain))])
    sw = post_weight.sum()
    post_weight = post_weight / sw if sw > 0 else np.full(len(theta), 1.0 / max(len(theta), 1))

    # -- 3. Gaussian-mixture importance evidence (one comp per seed) -------
    mus, covs = [], []
    for th in per_chain:
        if len(th) >= 6:
            mu, cov = _moment_match(th, np.zeros(len(th)))  # posterior moments
            mus.append(mu)
            covs.append(cov * proposal_inflate)
    if not mus:        # degenerate fallback: single component over the pool
        mu, cov = _moment_match(theta, np.zeros(len(theta)))
        mus, covs = [mu], [cov * proposal_inflate]
    n_comp = len(mus)
    weights = np.full(n_comp, 1.0 / n_comp)

    # draw from the mixture
    counts = rng.multinomial(n_is, weights)
    draws = []
    for c in range(n_comp):
        if counts[c] == 0:
            continue
        Lc = np.linalg.cholesky(covs[c] + 1e-12 * np.eye(5))
        z = rng.standard_normal((counts[c], 5))
        draws.append(mus[c][None, :] + z @ Lc.T)
    th_is = np.concatenate(draws, axis=0)

    logq = _mixture_logq(th_is, mus, covs, weights)
    logp = log_prior(th_is)
    valid = np.isfinite(logp)
    lnL_is = np.full(len(th_is), -np.inf)
    if valid.any():
        lnL_is[valid] = eval_lnL(like, th_is[valid])
    logw = np.where(valid, lnL_is + logp - logq, -np.inf)
    logZ, sigma_over_Z, neff = evidence_from_logweights(logw)
    logZ, sigma_over_Z, neff = _finalize_evidence(
        logZ, sigma_over_Z, neff, float(np.max(lnL)) if len(lnL) else np.nan)

    # Per-chain posterior draws (stacked) so callers can compute a POSTERIOR
    # effective-sample-size / R-hat -- the right "did we resolve the posterior"
    # diagnostic, distinct from the importance-sampling evidence ``neff`` above
    # (which is limited by the Gaussian-mixture proposal's fit to the target).
    theta_per_chain = np.stack(per_chain, axis=0)   # (n_starts, num_samples, 5)
    return dict(theta=theta, lnL=lnL, seeds=seeds, seed_lnL=seed_lnL,
                logZ=logZ, sigma_over_Z=sigma_over_Z, neff=neff,
                theta_per_chain=theta_per_chain, post_weight=post_weight)


# ---------------------------------------------------------------------------
# 2. flowMC
# ---------------------------------------------------------------------------
def flowmc_sample(like, d_min, d_max, n_chains=20, n_local_steps=20,
                  n_global_steps=20, n_training_loops=4, n_production_loops=4,
                  n_epochs=10, n_prior_pilot=8000, seed=0, mala_step_size=0.01,
                  reuse_state=None, temper=1.0, verbose=False):
    """Sample the multimodal extrinsic posterior with flowMC (normalizing flow).

    flowMC interleaves a local MALA kernel with a global normalizing-flow
    proposal; the flow learns the full multimodal geometry (the discrete sky
    blobs + phase/polarization structure) in one run, so it can hop between
    modes that a single local chain could not.

    Target ``logpdf(theta5, data) = lnL(theta5) + log_prior(theta5)`` built from
    ``like._scalar`` (JAX-traceable, so flowMC's gradient-based MALA works) and
    :func:`_log_prior_jax`.

    Flow re-use across evaluation points
    ------------------------------------
    ``reuse_state`` (the dict returned in ``result['flow_state']`` from a
    previous call) bootstraps this run from a previously-trained flow:

      * its trained normalizing-flow ``model`` is swapped into the bundle's
        ``model`` resource (warm-starting the flow weights), and
      * its ``positions`` (shape ``(n_chains, 5)``) initialize the chains.

    For a *batch* of nearby intrinsic templates (``--n-events-to-analyze``) the
    posterior geometry changes only slowly, so the re-used flow is a strong
    initialization.  Re-use degrades gracefully: a model-shape/version mismatch
    falls back to a fresh flow, mismatched ``positions`` fall back to a high-lnL
    prior draw.

    **RE-USE IS OFF BY DEFAULT AND SHOULD STAY OFF FOR SAMPLE-PRODUCING RUNS.**
    It CONTRACTS the extrinsic posterior in later slots -- psi to ~40% of its
    no-re-use width by slot 7 of an 8-event batch, on both of two seeds, with
    slot 0 as a control at ~1.0 -- and the efficiency argument that used to sit
    here does not survive production settings: 1589 s mean wall with re-use
    against 1567 s without, a difference smaller than the seed-to-seed spread and
    of flipping sign.  The ~2x speed-up reported in this module's README is
    specific to the small-budget SNR-sequence benchmark.  Enable it only where
    the EVIDENCE, not the samples, is the product.

    Returns
    -------
    dict with keys ``theta`` ``(N, 5)``, ``lnL`` ``(N,)``, ``logZ``,
    ``sigma_over_Z``, ``neff`` (moment-matched-Gaussian importance estimate),
    and ``flow_state`` = ``{'model': <trained flow>, 'positions': (n_chains,5)}``
    to pass as ``reuse_state`` to the next evaluation point.
    """
    from flowMC.Sampler import Sampler
    from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

    n_dim = 5

    # Likelihood tempering (a la old-RIFT's adapt-weight-exponent / GMM sampler):
    # sample the broadened target L^{1/T} pi (T = temper >= 1) so the flow learns
    # the full support and does not collapse onto the (possibly sub-resolution)
    # MAP at high SNR.  The true posterior (T=1) is recovered by reweighting the
    # tempered draws with w propto L^{1 - 1/T} (returned as ``post_weight``).
    inv_T = 1.0 / float(temper)

    def logpdf(theta5, data):
        # JAX-traceable tempered target: lnL/T + log-prior (both finite-on-support).
        return inv_T * like._scalar(theta5) + _log_prior_jax(theta5)

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    bkey, skey = jax.random.split(key)

    bundle = RQSpline_MALA_Bundle(
        rng_key=bkey, n_chains=n_chains, n_dims=n_dim, logpdf=logpdf,
        n_local_steps=n_local_steps, n_global_steps=n_global_steps,
        n_training_loops=n_training_loops, n_production_loops=n_production_loops,
        n_epochs=n_epochs, mala_step_size=mala_step_size, verbose=verbose)

    # --- flow re-use: warm-start the flow weights from a previous run ---
    if reuse_state is not None and reuse_state.get("model") is not None:
        try:
            bundle.resources["model"] = reuse_state["model"]
            if verbose:
                print("  flowMC: re-using trained flow from previous point")
        except Exception as e:   # version / shape mismatch -> fresh flow
            if verbose:
                print("  flowMC: flow re-use failed (%r); fresh flow" % e)

    sampler = Sampler(n_dim=n_dim, n_chains=n_chains, rng_key=skey,
                      resource_strategy_bundles=bundle)

    # --- chain initialization: re-used positions, else high-lnL prior draws ---
    init = None
    if reuse_state is not None and reuse_state.get("positions") is not None:
        pos = np.asarray(reuse_state["positions"])
        if pos.shape == (n_chains, n_dim) and np.all(np.isfinite(pos)) \
                and np.all(np.isfinite(log_prior(pos))):
            init = jnp.asarray(pos)
    if init is None:
        pilot = sample_prior(max(n_prior_pilot, n_chains), rng)
        pilot_lnL = eval_lnL(like, pilot)
        top = np.argsort(pilot_lnL)[::-1][:n_chains]
        init = jnp.asarray(pilot[top])

    sampler.sample(init, {})
    prod = np.asarray(sampler.resources["positions_production"].data)
    theta = prod.reshape(-1, n_dim)

    # drop any non-finite / out-of-support rows
    finite = np.all(np.isfinite(theta), axis=1) & np.isfinite(log_prior(theta))
    theta = theta[finite]
    lnL = eval_lnL(like, theta) if len(theta) else np.array([])

    # --- state to bootstrap the next evaluation point ---
    try:
        trained_model = sampler.resources["model"]
    except Exception:
        trained_model = None
    if len(theta) >= n_chains:                       # warm-start positions:
        top = np.argsort(lnL)[::-1][:n_chains]        # the n_chains best draws
        next_positions = theta[top]
    else:
        next_positions = None
    flow_state = {"model": trained_model, "positions": next_positions}

    # posterior reweighting: tempered draws ~ L^{1/T} pi -> true posterior L pi
    # by w propto L^{1-1/T}.  (Uniform when temper==1.)  Used for the skymap /
    # any posterior summary built from the (broadened, collapse-free) draws.
    if len(lnL):
        lw = (1.0 - inv_T) * lnL
        post_weight = np.exp(lw - lw.max()); post_weight /= post_weight.sum()
    else:
        post_weight = np.array([])

    # importance evidence from the flowMC draws (moment-matched Gaussian)
    logZ = sigma_over_Z = neff = np.nan
    if len(theta) >= 6:
        mu, cov = _moment_match(theta, np.zeros(len(theta)))
        cov = cov * 2.0
        Lc = np.linalg.cholesky(cov + 1e-12 * np.eye(n_dim))
        n_is = 40000
        z = rng.standard_normal((n_is, n_dim))
        th_is = mu[None, :] + z @ Lc.T
        logq = _gaussian_logq(th_is, mu, cov)
        logp = log_prior(th_is)
        valid = np.isfinite(logp)
        lnL_is = np.full(n_is, -np.inf)
        if valid.any():
            lnL_is[valid] = eval_lnL(like, th_is[valid])
        logw = np.where(valid, lnL_is + logp - logq, -np.inf)
        logZ, sigma_over_Z, neff = evidence_from_logweights(logw)
        logZ, sigma_over_Z, neff = _finalize_evidence(
            logZ, sigma_over_Z, neff, float(np.max(lnL)) if len(lnL) else np.nan)

    return dict(theta=theta, lnL=lnL, logZ=logZ, sigma_over_Z=sigma_over_Z,
                neff=neff, flow_state=flow_state, post_weight=post_weight,
                temper=float(temper))


# ---------------------------------------------------------------------------
# 2b. flowMC with φ_ref marginalised (4-D sampler)
# ---------------------------------------------------------------------------

_BOUNDS4 = [(0.0, _TWO_PI), (-_PI / 2 + 1e-3, _PI / 2 - 1e-3),
            (0.0, _PI), (1e-3, _PI - 1e-3)]
# 5-D support (ra, dec, psi, incl, phiref) for MAP-polishing the 5-D
# distance-marginalized seeds in multistart_nuts.
_BOUNDS5 = _BOUNDS4 + [(0.0, _TWO_PI)]


def sample_prior_4(n, rng):
    """Draw ``n`` prior samples of ``theta4 = (ra, dec, psi, incl)``."""
    ra = rng.uniform(0.0, _TWO_PI, n)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, n))
    psi = rng.uniform(0.0, _PI, n)
    incl = np.arccos(rng.uniform(-1.0, 1.0, n))
    return np.stack([ra, dec, psi, incl], axis=-1)


def log_prior_4(theta):
    """log prior density for ``theta4`` (numpy, batched).

    ``-inf`` outside the support.  Includes ``cos(dec)`` and ``sin(incl)``
    Jacobians from the uniform-sphere / uniform-cos parameterisations.
    """
    theta = np.atleast_2d(theta)
    ra, dec, psi, incl = [theta[..., i] for i in range(4)]
    inb = ((ra >= 0) & (ra <= _TWO_PI) & (dec >= -_PI / 2) & (dec <= _PI / 2)
           & (psi >= 0) & (psi <= _PI) & (incl >= 0) & (incl <= _PI))
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = (np.log(np.cos(dec)) - np.log(2.0)
                + np.log(np.sin(incl)) - np.log(2.0)
                - np.log(_TWO_PI) - np.log(_PI))
    return np.where(inb, logp, -np.inf)


def _log_prior_4_jax(theta4):
    """JAX-traceable log-prior for a single length-4 ``theta4`` vector."""
    ra, dec, psi, incl = theta4[0], theta4[1], theta4[2], theta4[3]
    inb = ((ra >= 0) & (ra <= _TWO_PI) & (dec >= -_PI / 2) & (dec <= _PI / 2)
           & (psi >= 0) & (psi <= _PI) & (incl >= 0) & (incl <= _PI))
    logp = (jnp.log(jnp.cos(dec)) - jnp.log(2.0)
            + jnp.log(jnp.sin(incl)) - jnp.log(2.0)
            - jnp.log(_TWO_PI) - jnp.log(_PI))
    return jnp.where(inb, logp, -1e30)


def eval_lnL_4(like, theta, chunk=_EVAL_CHUNK, desc="lnL"):
    """Evaluate the 4-param (phi-marginalised) lnL on an ``(N, 4)`` array."""
    theta = np.atleast_2d(theta)
    chunk = angle_marg_eval_chunk(like, chunk)
    N = theta.shape[0]
    out = np.empty(N)
    try:
        from tqdm.auto import tqdm
        chunks = list(range(0, N, chunk))
        it = tqdm(chunks, desc=desc, unit="chunk", leave=False)
    except ImportError:
        it = range(0, N, chunk)
    for i in it:
        sl = slice(i, min(i + chunk, N))
        out[sl] = np.asarray(like.log_likelihood(*[theta[sl, j] for j in range(4)]))
    return out


# d+psi 4-D order: (ra, dec, phiref, incl) -- distance+psi marginalized, phi_ref sampled.
_BOUNDS4PHI = [(0.0, _TWO_PI), (-_PI / 2 + 1e-3, _PI / 2 - 1e-3),
               (0.0, _TWO_PI), (1e-3, _PI - 1e-3)]


def sample_prior_4phi(n, rng):
    """Draw ``n`` prior samples of ``theta4 = (ra, dec, phiref, incl)`` (d+psi)."""
    ra = rng.uniform(0.0, _TWO_PI, n)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, n))
    phiref = rng.uniform(0.0, _TWO_PI, n)
    incl = np.arccos(rng.uniform(-1.0, 1.0, n))
    return np.stack([ra, dec, phiref, incl], axis=-1)


def log_prior_4phi(theta):
    """log prior density for ``theta4 = (ra, dec, phiref, incl)`` (numpy, batched).

    ``-inf`` outside the support.  phiref ~ U(0, 2pi) (density 1/(2pi)); includes
    ``cos(dec)`` and ``sin(incl)`` Jacobians from the uniform-sphere / uniform-cos
    parameterisations.
    """
    theta = np.atleast_2d(theta)
    ra, dec, phiref, incl = [theta[..., i] for i in range(4)]
    inb = ((ra >= 0) & (ra <= _TWO_PI) & (dec >= -_PI / 2) & (dec <= _PI / 2)
           & (phiref >= 0) & (phiref <= _TWO_PI)
           & (incl >= 0) & (incl <= _PI))
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = (np.log(np.cos(dec)) - np.log(2.0)
                + np.log(np.sin(incl)) - np.log(2.0)
                - np.log(_TWO_PI) - np.log(_TWO_PI))
    return np.where(inb, logp, -np.inf)


def _log_prior_4phi_jax(theta4):
    """JAX-traceable log-prior for a single length-4 ``theta4 = (ra,dec,phiref,incl)``."""
    ra, dec, phiref, incl = theta4[0], theta4[1], theta4[2], theta4[3]
    inb = ((ra >= 0) & (ra <= _TWO_PI) & (dec >= -_PI / 2) & (dec <= _PI / 2)
           & (phiref >= 0) & (phiref <= _TWO_PI)
           & (incl >= 0) & (incl <= _PI))
    logp = (jnp.log(jnp.cos(dec)) - jnp.log(2.0)
            + jnp.log(jnp.sin(incl)) - jnp.log(2.0)
            - jnp.log(_TWO_PI) - jnp.log(_TWO_PI))
    return jnp.where(inb, logp, -1e30)


_BOUNDS3 = [(0.0, _TWO_PI), (-_PI / 2 + 1e-3, _PI / 2 - 1e-3),
            (1e-3, _PI - 1e-3)]   # (ra, dec, incl) -- psi marginalized out


def sample_prior_3(n, rng):
    """Draw ``n`` prior samples of ``theta3 = (ra, dec, incl)`` (psi marginalized)."""
    ra = rng.uniform(0.0, _TWO_PI, n)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, n))
    incl = np.arccos(rng.uniform(-1.0, 1.0, n))
    return np.stack([ra, dec, incl], axis=-1)


def log_prior_3(theta):
    """log prior density for ``theta3 = (ra, dec, incl)`` (numpy, batched)."""
    theta = np.atleast_2d(theta)
    ra, dec, incl = [theta[..., i] for i in range(3)]
    inb = ((ra >= 0) & (ra <= _TWO_PI) & (dec >= -_PI / 2) & (dec <= _PI / 2)
           & (incl >= 0) & (incl <= _PI))
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = (np.log(np.cos(dec)) - np.log(2.0)
                + np.log(np.sin(incl)) - np.log(2.0) - np.log(_TWO_PI))
    return np.where(inb, logp, -np.inf)


def _log_prior_3_jax(theta3):
    """JAX-traceable log-prior for a single length-3 ``theta3`` vector."""
    ra, dec, incl = theta3[0], theta3[1], theta3[2]
    inb = ((ra >= 0) & (ra <= _TWO_PI) & (dec >= -_PI / 2) & (dec <= _PI / 2)
           & (incl >= 0) & (incl <= _PI))
    logp = (jnp.log(jnp.cos(dec)) - jnp.log(2.0)
            + jnp.log(jnp.sin(incl)) - jnp.log(2.0) - jnp.log(_TWO_PI))
    return jnp.where(inb, logp, -1e30)


def eval_lnL_3(like, theta, chunk=_EVAL_CHUNK, desc="lnL"):
    """Evaluate the 3-param (phi+psi-marginalised) lnL on an ``(N, 3)`` array."""
    theta = np.atleast_2d(theta)
    chunk = angle_marg_eval_chunk(like, chunk)
    N = theta.shape[0]
    out = np.empty(N)
    try:
        from tqdm.auto import tqdm
        it = tqdm(list(range(0, N, chunk)), desc=desc, unit="chunk", leave=False)
    except ImportError:
        it = range(0, N, chunk)
    for i in it:
        sl = slice(i, min(i + chunk, N))
        out[sl] = np.asarray(like.log_likelihood(*[theta[sl, j] for j in range(3)]))
    return out


def _warmup_compile(like, verbose=True):
    """Trigger JIT compilation on a 2-sample dummy batch before the main work.

    The first JAX call traces and compiles the XLA graph, which for a
    phi-marginalised likelihood with lax.scan can take 30–120 s.  Doing it
    explicitly with a message prevents the silent hang.
    """
    import time as _time
    if verbose:
        print("  [jax_ile] compiling JAX kernel … ", end="", flush=True)
    t0 = _time.perf_counter()
    n_dim = int(getattr(like, "_n_dim", 0)) or len(getattr(like, "ANGULAR_PARAM_ORDER", (0, 0, 0, 0)))
    dummy = np.zeros((2, n_dim))
    if n_dim >= 4:
        dummy[:, 2] = 0.5   # psi in (0, pi)
        dummy[:, 3] = 1.0   # incl in (0, pi)
    elif n_dim == 3:
        dummy[:, 2] = 1.0   # incl in (0, pi)  (psi marginalized out)
    _ = np.asarray(like.log_likelihood(*[dummy[:, j] for j in range(n_dim)]))
    if verbose:
        print("done (%.1f s)" % (_time.perf_counter() - t0), flush=True)


def _map_polish_4(like, seeds, n_steps=300, lr=3e-3, bounds=None, n_newton=30):
    """AD gradient-ascent polish of 4-D seeds to the local MAP.

    The flow under-reaches the true peak at high SNR (its best draw sits a few
    nats below the MAP); a few hundred projected-gradient steps on the
    JAX-traceable ``like._scalar`` climb the rest of the way.  Bounds are enforced
    by clipping into the 4-D support each step (``bounds`` defaults to the phi-marg
    (ra,dec,psi,incl) support; pass ``_BOUNDS4PHI`` for the d+psi order).  Returns
    (best_theta, best_lnL, [(theta_i, lnL_i), ...]) over the distinct seeds.
    """
    bounds = bounds if bounds is not None else _BOUNDS4
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    grad_f = jax.jit(jax.grad(lambda t: like._scalar(t)))
    hess_f = jax.jit(jax.hessian(lambda t: like._scalar(t)))
    polished = []
    best_t, best_v = None, -np.inf
    for s in np.atleast_2d(np.asarray(seeds, float)):
        t = jnp.asarray(np.clip(s, lo, hi))
        step = lr
        v_prev = float(like._scalar(t))
        for _ in range(n_steps):
            g = np.asarray(grad_f(t))
            if not np.all(np.isfinite(g)):
                break
            t_new = np.clip(np.asarray(t) + step * g, lo, hi)
            v_new = float(like._scalar(jnp.asarray(t_new)))
            if v_new >= v_prev:            # accept + grow step
                t = jnp.asarray(t_new); v_prev = v_new; step *= 1.2
            else:                           # reject + shrink (backtracking)
                step *= 0.5
                if step < 1e-8:
                    break
        # Newton refinement: gradient-ascent stalls on a sharp (1/SNR)-narrow peak
        # (huge gradient -> tiny backtracked steps).  A curvature-normalized step
        # dx = F^{-1} g (F=-Hessian, eig-floored pos-def) lands on the peak in ~1
        # step, so the Fisher whitening that follows uses the TRUE peak curvature.
        for _ in range(n_newton):
            g = np.asarray(grad_f(t))
            H = np.asarray(hess_f(t)); H = 0.5 * (H + H.T)
            if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
                break
            evals, V = np.linalg.eigh(-H)               # F = -H (pos-def at a max)
            emax = float(np.max(np.abs(evals)))
            if not np.isfinite(emax) or emax <= 0:
                break
            evals = np.clip(evals, 1e-6 * max(emax, 1.0), None)
            dx = V @ ((V.T @ g) / evals)                # F^{-1} g  (ascent direction)
            alpha, improved = 1.0, False
            for _ls in range(40):                       # backtracking along Newton dir
                cand = np.clip(np.asarray(t) + alpha * dx, lo, hi)
                vc = float(like._scalar(jnp.asarray(cand)))
                if vc >= v_prev:
                    t = jnp.asarray(cand); v_prev = vc; improved = True; break
                alpha *= 0.5
            if not improved:
                break
        v = float(like._scalar(t))
        polished.append((np.asarray(t), v))
        if v > best_v:
            best_v, best_t = v, np.asarray(t)
    return best_t, best_v, polished


def _bounds_for_order(param_order):
    """Support box matching the angular parameter order (3-D / 4-D phi / 4-D psi)."""
    n = len(param_order)
    if n == 3:
        return _BOUNDS3
    if "phiref" in param_order:
        return _BOUNDS4PHI
    return _BOUNDS4


def _fisher_whitening(like, seeds, bounds, eig_floor_rel=1e-6, verbose=False):
    """Build a Fisher whitening map ``theta = theta_MAP + A @ y`` (y ~ unit scale).

    MAP-polishes ``seeds`` to the local maximum, takes the observed Fisher
    ``F = -Hessian(lnL)`` there, and returns the symmetric sqrt of the covariance
    ``A = F^{-1/2}`` (so a unit-isotropic ``y`` maps to the local posterior
    ellipsoid).  At high SNR the posterior is ~Gaussian, so sampling in ``y`` makes
    the (1/SNR)-narrow target O(1)-scaled -> the flow/MALA no longer collapse.

    Eigenvalues of F are floored at ``eig_floor_rel * max|eig|`` to tame the
    indefinite/near-flat directions (e.g. a residual degeneracy ridge), bounding A.
    Returns ``(theta_MAP, A, A_inv, map_lnL)`` (all numpy) or ``None`` on failure.
    """
    try:
        map_theta, map_lnL, _ = _map_polish_4(like, seeds, bounds=bounds)
        H = np.asarray(jax.hessian(lambda t: like._scalar(t))(jnp.asarray(map_theta)))
        F = -0.5 * (H + H.T)                               # observed Fisher, symmetric
        evals, V = np.linalg.eigh(F)
        emax = float(np.max(np.abs(evals)))
        if not np.isfinite(emax) or emax <= 0:
            return None
        evals = np.clip(evals, eig_floor_rel * emax, None)  # pos-def, bounded width
        A = (V * (1.0 / np.sqrt(evals))) @ V.T              # F^{-1/2} (symmetric)
        A_inv = (V * np.sqrt(evals)) @ V.T                  # F^{1/2}
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(A_inv))):
            return None
        return np.asarray(map_theta, float), A, A_inv, float(map_lnL)
    except Exception as e:                                  # noqa: BLE001
        if verbose:
            print("  [fisher] whitening failed (%r); raw coords" % e)
        return None


def flowmc_sample_phimarg(like, d_min, d_max, n_chains=20, n_local_steps=20,
                           n_global_steps=20, n_training_loops=4, n_production_loops=4,
                           n_epochs=10, n_prior_pilot=8000, seed=0, mala_step_size=0.01,
                           reuse_state=None, temper=1.0, temper_adapt=False,
                           temper_init=0.02, temper_ess_frac=0.5, temper_max_stages=16,
                           temper_max_dbeta=0.15, fisher_precondition=False,
                           fisher_inv_T_min=0.5, fisher_is_samples=0,
                           fisher_is_inflate=1.3, verbose=False):
    """flowMC with φ_ref marginalised — 4-D sampler over (ra, dec, psi, incl).

    Identical in structure to :func:`flowmc_sample` except φ_ref is removed from
    the sample space; the target is the φ_ref- and distance-marginalised posterior.
    ``like`` must be a :class:`~RIFT.likelihood.jax_ile.wrapper.JAXDistPhiMargLikelihood`.

    The degenerate φ_ref–psi ridge is absent after marginalisation, so the
    sampler converges reliably at high SNR.  Draw φ_ref from its conditional
    posterior after the sampling step via ``like.sample_phi_ref``.

    Returns the same dict as :func:`flowmc_sample` (``theta`` is (N,4)).
    """
    from flowMC.Sampler import Sampler
    from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

    # Dimension-agnostic: 4-D (ra,dec,psi,incl) phi-marg, or 3-D (ra,dec,incl)
    # phi+psi-marg.  Helpers selected from the likelihood's parameter count so the
    # 4-D path is unchanged (defaults) and the 3-D psi-marg path reuses this code.
    _param_order = getattr(like, "ANGULAR_PARAM_ORDER", ("ra", "dec", "psi", "incl"))
    n_dim = len(_param_order)
    if n_dim == 3:
        _sample_prior, _log_prior = sample_prior_3, log_prior_3
        _log_prior_jax, _eval_lnL = _log_prior_3_jax, eval_lnL_3
    elif "phiref" in _param_order:
        # d+psi 4-D: slot-2 is phi_ref in [0, 2pi) (sampled), psi marginalized.
        _sample_prior, _log_prior = sample_prior_4phi, log_prior_4phi
        _log_prior_jax, _eval_lnL = _log_prior_4phi_jax, eval_lnL_4
    else:
        # phi-marg 4-D: slot-2 is psi in [0, pi), phi_ref marginalized.
        _sample_prior, _log_prior = sample_prior_4, log_prior_4
        _log_prior_jax, _eval_lnL = _log_prior_4_jax, eval_lnL_4
    rng = np.random.default_rng(seed)

    # Fisher whitening map (set after the pilot, below, if fisher_precondition).
    # When active the flowMC chains live in whitened coords y (theta = map + A@y),
    # so the (1/SNR)-narrow high-SNR posterior is O(1)-scaled for the flow/MALA.
    # init_positions passed to _one_pass are ALWAYS theta-space; the y<->theta
    # transform is localized here.  The constant Jacobian |det A| does not enter
    # the (theta-space) prior/likelihood or the downstream evidence.
    _W = None

    # One flow training+production pass at a fixed tempering exponent inv_T,
    # optionally warm-started from a previous flow model + chain positions.
    # Returns (theta, lnL, trained_model).
    def _one_pass(inv_T_pass, init_positions, init_model, pass_seed):
        def logpdf(coords, data):
            theta4 = (_W["map"] + _W["A"] @ coords) if _W is not None else coords
            return inv_T_pass * like._scalar(theta4) + _log_prior_jax(theta4)
        if _W is not None:                       # theta-space init -> whitened y
            init_positions = jnp.asarray(
                (np.asarray(init_positions) - np.asarray(_W["map"]))
                @ np.asarray(_W["A_inv"]).T)
        key = jax.random.PRNGKey(pass_seed)
        bkey, skey = jax.random.split(key)
        bundle = RQSpline_MALA_Bundle(
            rng_key=bkey, n_chains=n_chains, n_dims=n_dim, logpdf=logpdf,
            n_local_steps=n_local_steps, n_global_steps=n_global_steps,
            n_training_loops=n_training_loops, n_production_loops=n_production_loops,
            n_epochs=n_epochs, mala_step_size=mala_step_size, verbose=verbose)
        if init_model is not None:
            try:
                bundle.resources["model"] = init_model
            except Exception as e:
                if verbose:
                    print("  flowMC (phimarg): flow re-use failed (%r); fresh flow" % e)
        sampler = Sampler(n_dim=n_dim, n_chains=n_chains, rng_key=skey,
                          resource_strategy_bundles=bundle)
        if verbose:
            print("  [flowMC] sampling (inv_T=%.4g) …" % inv_T_pass, flush=True)
        sampler.sample(init_positions, {})
        prod = np.asarray(sampler.resources["positions_production"].data)
        th = prod.reshape(-1, n_dim)
        if _W is not None:                       # whitened y -> theta
            th = np.asarray(_W["map"])[None, :] + th @ np.asarray(_W["A"]).T
        ok = np.all(np.isfinite(th), axis=1) & np.isfinite(_log_prior(th))
        th = th[ok]
        ll = _eval_lnL(like, th, desc="reweight") if len(th) else np.array([])
        try:
            mdl = sampler.resources["model"]
        except Exception:
            mdl = None
        return th, ll, mdl

    # ---- initial chain positions: re-used, else high-lnL prior draws ----
    init = None
    prior_lnL_mean = prior_lnL_sem = None   # beta=0 anchor for thermodynamic integration
    if reuse_state is not None and reuse_state.get("positions") is not None:
        pos = np.asarray(reuse_state["positions"])
        if pos.shape == (n_chains, n_dim) and np.all(np.isfinite(pos)) \
                and np.all(np.isfinite(_log_prior(pos))):
            init = jnp.asarray(pos)
    if init is None:
        # Compile the JIT kernel before the pilot scan so the progress bar
        # reflects actual likelihood evaluations, not silent compilation time.
        _warmup_compile(like, verbose=verbose)
        pilot = _sample_prior(max(n_prior_pilot, n_chains), rng)
        pilot_lnL = _eval_lnL(like, pilot, desc="pilot")
        prior_lnL_mean = float(np.mean(pilot_lnL))
        prior_lnL_sem = float(np.std(pilot_lnL) / max(1.0, np.sqrt(len(pilot_lnL))))
        top = np.argsort(pilot_lnL)[::-1][:n_chains]
        init = jnp.asarray(pilot[top])
    else:
        _warmup_compile(like, verbose=verbose)

    # ---- Fisher preconditioning -------------------------------------------
    # Whiten the sample space around the MAP so the narrow posterior is O(1) for
    # the flow.  For the SINGLE-stage path we whiten up front from a cold MAP.
    # For the ADAPT-ADAPT path we do NOT whiten cold: at very high SNR a cold
    # gradient polish from the prior pilot cannot reach the (1/SNR)-narrow peak
    # (its MAP sits ~thousands of nats low -> the Fisher there is far too broad ->
    # the whitening fails and SNR>=640 still collapses).  Instead we RE-whiten
    # inside the anneal loop from the tempering-tracked best sample once inv_T is
    # high enough (see below), where the MAP polish starts near-peak and converges.
    helper_bounds = _bounds_for_order(_param_order)
    if fisher_precondition and not temper_adapt:
        fw = _fisher_whitening(like, np.asarray(init), helper_bounds, verbose=verbose)
        if fw is not None:
            map_theta, A_w, A_inv_w, map_lnL = fw
            _W = dict(map=jnp.asarray(map_theta), A=jnp.asarray(A_w),
                      A_inv=jnp.asarray(A_inv_w))
            if verbose:
                widths = np.sqrt(np.clip(np.diag(A_w @ A_w.T), 0.0, None))
                print("  [fisher] whitening ON  MAP lnL=%.2f  theta-widths=%s"
                      % (map_lnL, np.array2string(widths, precision=4)))
            yj = rng.normal(size=(n_chains, n_dim)) * 0.5     # near-MAP, unit scale
            init = jnp.asarray(np.asarray(map_theta)[None, :] + yj @ np.asarray(A_w).T)
        elif verbose:
            print("  [fisher] whitening requested but unavailable; raw coords")

    # thermodynamic integration needs a prior (beta=0) anchor even when warm-started
    if temper_adapt and prior_lnL_mean is None:
        anchor = _sample_prior(max(n_prior_pilot, n_chains), rng)
        anchor_lnL = _eval_lnL(like, anchor, desc="prior-anchor")
        prior_lnL_mean = float(np.mean(anchor_lnL))
        prior_lnL_sem = float(np.std(anchor_lnL) / max(1.0, np.sqrt(len(anchor_lnL))))

    init_model = reuse_state.get("model") if reuse_state is not None else None
    if init_model is not None and verbose:
        print("  flowMC (phimarg): re-using trained flow from previous point")

    if not temper_adapt:
        # ---- single static tempering stage (default; temper=1 -> exact) ----
        inv_T = 1.0 / float(temper)
        theta, lnL, trained_model = _one_pass(inv_T, init, init_model, seed)
    else:
        # ---- adaptive likelihood tempering (a la mcsamplerGPU adapt-adapt) ----
        # Anneal inv_T from temper_init up to 1.0; each step is the largest value
        # whose tempered-reweight ESS from the current stage stays >= the target
        # fraction (adaptive SMC tempering).  Flow + chain positions warm-start
        # across stages, so the final inv_T=1 flow is tight on the true peak and
        # the evidence step below no longer collapses to neff=1.  When ESS is
        # poor we keep adapting on the tempered draws instead of giving up.
        inv_T = min(max(float(temper_init), 1e-3), 1.0)
        positions, trained_model = init, init_model
        theta = np.empty((0, n_dim)); lnL = np.array([])
        ti_beta = []; ti_mean = []; ti_sem = []   # ladder for thermodynamic integration
        ti_min_step_ess = np.inf                   # bottleneck inter-stage ESS
        stage = 0
        _whitened_once = False                      # Fisher re-whitening done?
        while True:
            theta, lnL, trained_model = _one_pass(inv_T, positions, trained_model,
                                                  seed + stage)
            if len(theta) >= n_chains:
                positions = jnp.asarray(theta[np.argsort(lnL)[::-1][:n_chains]])
            if len(lnL):
                # <lnL>_{inv_T} estimated from this stage's tempered draws
                ti_beta.append(float(inv_T))
                ti_mean.append(float(np.mean(lnL)))
                ti_sem.append(float(np.std(lnL) / max(1.0, np.sqrt(len(lnL)))))
            if verbose:
                print("  [temper-adapt] stage %d  inv_T=%.4g  ndraw=%d  maxlnL=%.2f"
                      % (stage, inv_T, len(theta),
                         float(np.max(lnL)) if len(lnL) else np.nan), flush=True)
            stage += 1
            if inv_T >= 1.0 or len(lnL) < 2:   # just sampled the exact target
                break
            # next inv_T: largest value in (inv_T, 1] with tempered ESS >= target
            lnL_c = lnL - np.max(lnL)
            def _ess(next_invT):
                lw = (next_invT - inv_T) * lnL_c
                w = np.exp(lw - lw.max())
                return (w.sum() ** 2) / np.sum(w ** 2)
            target = float(temper_ess_frac) * len(lnL)
            if stage >= int(temper_max_stages):
                inv_T_new = 1.0                 # safety stop: force the exact pass
            elif _ess(1.0) >= target:
                inv_T_new = 1.0
            else:
                lo, hi = inv_T, 1.0
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if _ess(mid) >= target:
                        lo = mid
                    else:
                        hi = mid
                inv_T_new = lo
            # Cap the step so the thermodynamic-integration trapezoid stays
            # accurate: the ESS criterion controls sampling quality but allows
            # large dbeta jumps that under-resolve the (concave) <lnL>(beta) curve.
            if temper_max_dbeta and temper_max_dbeta > 0:
                inv_T_new = min(inv_T_new, inv_T + float(temper_max_dbeta))
            ti_min_step_ess = min(ti_min_step_ess, float(_ess(inv_T_new)))
            inv_T = inv_T_new

            # ---- adaptive Fisher re-whitening (the high-SNR fix) ----
            # Once the anneal has tightened enough that the best tracked sample is
            # near the true peak, re-estimate the whitening from THAT sample (the
            # MAP polish now starts near-peak and converges, unlike a cold polish).
            # The Fisher uses the full inv_T=1 lnL curvature, so one re-whitening is
            # correct for all remaining (sharper) stages.  Reset the flow so it
            # retrains in the new O(1) whitened coordinates.
            if (fisher_precondition and not _whitened_once
                    and inv_T >= float(fisher_inv_T_min) and len(lnL)):
                best = np.asarray(positions[0])[None, :]   # highest-lnL tracked draw
                fw = _fisher_whitening(like, best, helper_bounds, verbose=verbose)
                if fw is not None:
                    map_theta, A_w, A_inv_w, map_lnL = fw
                    # accept only if the polish reached at least the tracked best
                    if map_lnL >= float(np.max(lnL)) - 1.0:
                        _W = dict(map=jnp.asarray(map_theta), A=jnp.asarray(A_w),
                                  A_inv=jnp.asarray(A_inv_w))
                        trained_model = None           # retrain flow in whitened coords
                        positions = jnp.asarray(        # recentre chains at the MAP
                            np.asarray(map_theta)[None, :]
                            + (rng.normal(size=(n_chains, n_dim)) * 0.5) @ np.asarray(A_w).T)
                        _whitened_once = True
                        if verbose:
                            widths = np.sqrt(np.clip(np.diag(A_w @ A_w.T), 0.0, None))
                            print("  [fisher] re-whitened at inv_T=%.3g  MAP lnL=%.2f "
                                  " theta-widths=%s" % (inv_T, map_lnL,
                                  np.array2string(widths, precision=5)), flush=True)
                    elif verbose:
                        print("  [fisher] re-whiten skipped (polish %.1f < best %.1f)"
                              % (map_lnL, float(np.max(lnL))), flush=True)

    next_positions = (theta[np.argsort(lnL)[::-1][:n_chains]]
                      if len(theta) >= n_chains else None)
    flow_state = {"model": trained_model, "positions": next_positions}

    # AD MAP-polish (high-SNR diagnostic): the flow under-reaches the sharp peak
    # (its best draw is a few nats low), so gradient-ascend distinct high-lnL
    # draws to the true local MAP; seeds spread over ring modes for multimodality.
    map_theta = map_lnL = None
    logZ_laplace = np.nan
    if temper_adapt and len(theta) >= 1 and n_dim == 4:   # polish is 4-D-specific
        order = np.argsort(lnL)[::-1]
        uniq = []
        for i in order:
            if all(np.linalg.norm(theta[i] - theta[j]) > 1e-2 for j in uniq):
                uniq.append(i)
            if len(uniq) >= 8:
                break
        seeds = theta[uniq] if uniq else theta[order[:1]]
        _polish_bounds = _BOUNDS4PHI if "phiref" in _param_order else _BOUNDS4
        try:
            map_theta, map_lnL, _ = _map_polish_4(like, seeds, bounds=_polish_bounds)
            if verbose:
                print("  [map-polish] flow lnLmax=%.3f -> polished MAP=%.3f (gain %.3f)"
                      % (float(np.max(lnL)), map_lnL,
                         map_lnL - float(np.max(lnL))), flush=True)
        except Exception as e:
            if verbose:
                print("  [map-polish] failed: %r" % e)

    if len(lnL):
        lw = (1.0 - inv_T) * lnL
        post_weight = np.exp(lw - lw.max()); post_weight /= post_weight.sum()
    else:
        post_weight = np.array([])

    logZ = sigma_over_Z = neff = np.nan
    if temper_adapt and len(ti_beta) >= 1 and prior_lnL_mean is not None:
        # ---- thermodynamic integration over the anneal ladder ----
        #   ln Z = \int_0^1 <lnL>_{inv_T} d(inv_T),  with Z(inv_T=0) = \int pi = 1.
        # Makes no single-peak/Gaussian assumption -> robust to the curved
        # (psi, phi_ref) degeneracy ridge that collapses the moment-matched IS.
        #
        # KNOWN BIAS (deferred; not now): logZ here is ~1% LOW vs a converged AV
        # reference.  This is NOT a peak-finding problem: an AD gradient MAP-polish
        # of the flow draws gains ~0 (the flow already sits at the jax likelihood's
        # phi-marginalized MAP, ~779.9 at SNR40).  The apparent "flow lnLmax 779.9
        # vs AV 783.4" gap is just the phi marginalization (AV samples phi_orb;
        # this likelihood marginalizes phi_ref) and CANCELS in the evidence.  So
        # the residual ~1% is a genuine jax-TI-vs-AV *evidence* discrepancy
        # (normalization / method), not under-reach -- needs a likelihood-norm
        # audit (wrapper.py distance/phi-marg constants vs AV's distmarg table).
        # It is consistent across the sequence, which is what matters for
        # relative-lnL inference; absolute |dlnL|<1 (tolerance ~SNR^2) is
        # overwhelmed by neglected physical/discretization systematics. Defer.
        betas = np.array([0.0] + ti_beta, dtype=float)
        means = np.array([prior_lnL_mean] + ti_mean, dtype=float)
        sems = np.array([prior_lnL_sem or 0.0] + ti_sem, dtype=float)
        order = np.argsort(betas)
        betas, means, sems = betas[order], means[order], sems[order]
        keep = np.concatenate([np.diff(betas) > 1e-12, [True]])  # drop duplicate betas
        betas, means, sems = betas[keep], means[keep], sems[keep]
        if len(betas) >= 2:
            db = np.diff(betas)
            logZ = float(np.sum(0.5 * (means[1:] + means[:-1]) * db))  # trapezoid
            w = np.zeros_like(betas)                                   # variance weights
            w[0] = 0.5 * db[0]; w[-1] = 0.5 * db[-1]
            if len(betas) > 2:
                w[1:-1] = 0.5 * (betas[2:] - betas[:-2])
            sigma_over_Z = float(np.sqrt(np.sum((w * sems) ** 2)))
            neff = (float(ti_min_step_ess) if np.isfinite(ti_min_step_ess)
                    else float(len(theta)))
        if map_theta is not None:
            # Laplace-at-(polished)-MAP evidence diagnostic.  Regularize the
            # Hessian (|eigvals|) to tolerate the indefinite ridge; single-mode,
            # so it under-counts multimodal time-delay-ring contributions --
            # reported for comparison against TI, not as the primary estimate.
            try:
                H = np.asarray(jax.hessian(lambda t: like._scalar(t))(jnp.asarray(map_theta)))
                wv = np.linalg.eigvalsh(-0.5 * (H + H.T))
                wreg = np.maximum(np.abs(wv), 1e-6)
                dim = map_theta.shape[0]
                logZ_laplace = (map_lnL + float(_log_prior(map_theta[None, :])[0])
                                + 0.5 * dim * np.log(2 * np.pi)
                                - 0.5 * float(np.sum(np.log(wreg))))
                if verbose:
                    print("  [evidence] TI logZ=%.3f  Laplace@MAP logZ=%.3f"
                          % (logZ, logZ_laplace), flush=True)
            except Exception as e:
                if verbose:
                    print("  [evidence] Laplace diag failed: %r" % e)
    elif len(theta) >= 6:
        mu, cov = _moment_match(theta, np.zeros(len(theta)))
        cov = cov * 2.0
        Lc = np.linalg.cholesky(cov + 1e-12 * np.eye(n_dim))
        n_is = 40000
        z = rng.standard_normal((n_is, n_dim))
        th_is = mu[None, :] + z @ Lc.T
        logq = _gaussian_logq(th_is, mu, cov)
        logp = _log_prior(th_is)
        valid = np.isfinite(logp)
        lnL_is = np.full(n_is, -np.inf)
        if valid.any():
            lnL_is[valid] = _eval_lnL(like, th_is[valid])
        logw = np.where(valid, lnL_is + logp - logq, -np.inf)
        logZ, sigma_over_Z, neff = evidence_from_logweights(logw)
        logZ, sigma_over_Z, neff = _finalize_evidence(
            logZ, sigma_over_Z, neff, float(np.max(lnL)) if len(lnL) else np.nan)

    # ---- High-SNR FALLBACK: Fisher-whitened importance sampling -------------
    # The flow can collapse at extreme SNR (NF training nan -> a few unique sky
    # points) even when the posterior is fine.  This draws sky samples DIRECTLY
    # from the Fisher-whitened Gaussian about the (Newton-polished) MAP and
    # importance-reweights by the true lnL -- no flow training, so it is immune to
    # that collapse.  Overrides the sample set (theta,lnL); the TI logZ above stays
    # the primary evidence (the IS logZ is reported as a cross-check).
    if fisher_is_samples and fisher_is_samples > 0 and len(theta) >= 1:
        if _W is not None:
            mapT = np.asarray(_W["map"]); A_is = np.asarray(_W["A"])
        else:
            seed_best = (np.asarray(map_theta)[None, :] if map_theta is not None
                         else theta[np.argmax(lnL)][None, :])
            fw = _fisher_whitening(like, seed_best, helper_bounds, verbose=verbose)
            mapT, A_is = (fw[0], fw[1]) if fw is not None else (None, None)
        if mapT is not None:
            cov_is = (float(fisher_is_inflate) ** 2) * (A_is @ A_is.T)
            cov_is = 0.5 * (cov_is + cov_is.T)
            try:
                Lc = np.linalg.cholesky(cov_is + 1e-12 * np.eye(n_dim))
                N = int(fisher_is_samples)
                z = rng.standard_normal((N, n_dim))
                th_is = mapT[None, :] + z @ Lc.T
                logp = _log_prior(th_is)
                valid = np.isfinite(logp)
                lnL_is = np.full(N, -np.inf)
                if valid.any():
                    lnL_is[valid] = _eval_lnL(like, th_is[valid], desc="fisher-IS")
                logq = _gaussian_logq(th_is, mapT, cov_is)
                logw = np.where(valid & np.isfinite(lnL_is), lnL_is + logp - logq, -np.inf)
                logZ_is, sigZ_is, neff_is = evidence_from_logweights(logw)
                fin = np.isfinite(logw)
                if fin.sum() >= 2:
                    lw = logw[fin] - np.max(logw[fin])
                    w = np.exp(lw); w = w / w.sum()
                    nout = min(N, 4800)
                    idx = rng.choice(np.where(fin)[0], size=nout, p=w)
                    theta = th_is[idx]; lnL = lnL_is[idx]
                    post_weight = np.ones(nout) / nout
                    if verbose:
                        print("  [fisher-IS] N=%d ESS=%.0f neff=%.0f unique=%d "
                              "logZ_IS=%.2f (TI logZ=%.2f)"
                              % (N, 1.0 / np.sum(w ** 2), neff_is,
                                 len(np.unique(idx)), logZ_is, logZ), flush=True)
            except Exception as e:                                # noqa: BLE001
                if verbose:
                    print("  [fisher-IS] failed (%r); keeping flow samples" % e)

    return dict(theta=theta, lnL=lnL, logZ=logZ, sigma_over_Z=sigma_over_Z,
                neff=neff, flow_state=flow_state, post_weight=post_weight,
                temper=float(1.0 / inv_T) if inv_T > 0 else float(temper),
                logZ_laplace=float(logZ_laplace),
                lnL_map=(float(map_lnL) if map_lnL is not None else np.nan))


# ---------------------------------------------------------------------------
# 2b. Adaptive SMC with a "puffball" random-walk move (robust at any SNR)
# ---------------------------------------------------------------------------
def smc_puffball_sample(like, d_min, d_max, n_walkers=2000, seed=0,
                        ess_frac=0.5, max_stages=80, n_move=10, puff_scale=1.0,
                        max_dbeta=0.25, is_evidence=True, is_samples=60000,
                        is_inflate=1.5, verbose=False, **_ignore):
    """Tempered adaptive SMC over the dist+phi(+psi) marginalized angular target.

    The flow collapses on sharp high-SNR peaks because it trusts one learned/Hessian
    geometry.  This instead carries a CLOUD of walkers up an adaptive temperature
    ladder (inv_T: 0 -> 1) and, at each rung, (1) tempered-resamples toward higher
    lnL then (2) applies K "puffball" random-walk Metropolis moves whose proposal
    covariance is ESTIMATED FROM THE CLOUD (so it shrinks to match the posterior as
    the cloud concentrates -- never from the Hessian, so slivers/non-Gaussianity
    can't fool it, and the move re-broadens the cloud every rung so it cannot
    collapse).  This is the SMC analogue of RIFT-AV's "sample -> puffball -> sample"
    and of nested sampling's hill-climb; robust at LISA-loud SNR.

    Returns the same dict shape as :func:`flowmc_sample_phimarg`, plus ``inv_T``:
    the tempering exponent the ladder ACTUALLY reached (< 1 when it stopped at
    ``max_stages``), with ``post_weight`` the matching ``L**(1-inv_T)`` correction
    to the posterior.  Evidence is the standard SMC normalizing-constant estimator
    logZ = sum_t logmeanexp(dbeta_t lnL) -- which is log Z(inv_T), not log Z, on a
    ladder that stopped short.
    """
    _param_order = getattr(like, "ANGULAR_PARAM_ORDER", ("ra", "dec", "psi", "incl"))
    n_dim = len(_param_order)
    if n_dim == 3:
        _sample_prior, _log_prior, _eval = sample_prior_3, log_prior_3, eval_lnL_3
    elif "phiref" in _param_order:
        _sample_prior, _log_prior, _eval = sample_prior_4phi, log_prior_4phi, eval_lnL_4
    else:
        _sample_prior, _log_prior, _eval = sample_prior_4, log_prior_4, eval_lnL_4
    rng = np.random.default_rng(seed)
    bounds = _bounds_for_order(_param_order)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    # periodic angles (ra, psi, phi_ref) wrap; dec/incl reflect-clip via the prior.
    period = np.array([(hi[i] - lo[i]) if _param_order[i] in ("ra", "psi", "phiref")
                       else 0.0 for i in range(n_dim)])

    def _fix(x):                                   # wrap periodic dims into [lo,hi)
        out = np.array(x, float)
        for i in range(n_dim):
            if period[i] > 0:
                out[:, i] = lo[i] + np.mod(out[:, i] - lo[i], period[i])
        return out

    W = int(n_walkers)
    cloud = _sample_prior(W, rng)
    lnL = _eval(like, cloud, desc="smc-prior")
    inv_T, logZ, stage = 0.0, 0.0, 0
    ti_min_ess = float(W)
    while inv_T < 1.0 and stage < int(max_stages):
        finite = np.isfinite(lnL)
        if finite.sum() < 2:
            break
        lnLc = lnL - np.max(lnL[finite])

        def _ess(db):
            lw = db * lnLc
            lw = np.where(np.isfinite(lw), lw, -np.inf)
            m = np.max(lw)
            w = np.exp(lw - m)
            s = w.sum()
            return (s * s) / np.sum(w * w) if s > 0 else 0.0

        target = float(ess_frac) * W
        # Largest rung allowed here: never past inv_T == 1, and never past the
        # per-stage cap (max_dbeta <= 0 disables that cap, as on the flowMC path).
        hi_db = 1.0 - inv_T
        if float(max_dbeta) > 0:
            hi_db = min(hi_db, float(max_dbeta))
        if _ess(hi_db) >= target:
            db = hi_db
        else:
            a, b = 0.0, hi_db
            for _ in range(40):
                mid = 0.5 * (a + b)
                if _ess(mid) >= target:
                    a = mid
                else:
                    b = mid
            # The floor keeps a stalled bisection moving, but it must never
            # carry the rung PAST the rung cap: db > hi_db advances inv_T beyond
            # 1 (or beyond max_dbeta), and the resample/Metropolis moves below
            # then target L**inv_T with inv_T > 1 -- an OVER-tempered cloud that
            # the final min(inv_T, 1) would report as temper=1 with uniform
            # post_weight, i.e. exactly the mislabelling the tail guards against.
            db = min(max(a, 1e-4), hi_db)
        # SMC evidence increment: logZ += logmeanexp(db * lnL)
        z = db * lnL
        z = z[np.isfinite(z)]
        mz = np.max(z)
        logZ += float(mz + np.log(np.mean(np.exp(z - mz))))
        inv_T += db
        # tempered resample (multinomial)
        lw = db * lnLc
        w = np.where(np.isfinite(lw), np.exp(lw - np.max(lw)), 0.0)
        ti_min_ess = min(ti_min_ess, float((w.sum() ** 2) / np.sum(w * w)))
        w = w / w.sum()
        idx = rng.choice(W, size=W, p=w)
        cloud, lnL = cloud[idx], lnL[idx]
        # puffball random-walk Metropolis moves at the current inv_T
        C = np.atleast_2d(np.cov(cloud.T)) + 1e-12 * np.eye(n_dim)
        try:
            L = np.linalg.cholesky((puff_scale ** 2) * C)
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.maximum(np.diag((puff_scale ** 2) * C), 1e-14)))
        acc = 0.0
        lp_cur = _log_prior(cloud)
        for _ in range(int(n_move)):
            prop = _fix(cloud + rng.standard_normal((W, n_dim)) @ L.T)
            lp_prop = _log_prior(prop)
            ok = np.isfinite(lp_prop)
            lnL_prop = np.full(W, -np.inf)
            if ok.any():
                lnL_prop[ok] = _eval(like, prop[ok], desc="smc-move")
            logA = inv_T * (lnL_prop - lnL) + lp_prop - lp_cur
            take = ok & np.isfinite(lnL_prop) & (np.log(rng.random(W)) < logA)
            cloud = np.where(take[:, None], prop, cloud)
            lnL = np.where(take, lnL_prop, lnL)
            lp_cur = np.where(take, lp_prop, lp_cur)
            acc += float(take.mean())
        stage += 1
        if verbose:
            uniq = len(np.unique(cloud[:, 0]))
            print("  [smc] stage %d inv_T=%.4g dbeta=%.4g maxlnL=%.2f acc=%.2f uniq=%d"
                  % (stage, inv_T, db, float(np.max(lnL)), acc / max(1, n_move), uniq),
                  flush=True)

    logZ_smc = float(logZ)
    sigma_over_Z = float(1.0 / np.sqrt(ti_min_ess)) if ti_min_ess > 0 else np.nan
    neff = float(ti_min_ess)

    # ---- Cloud-fitted importance-sampling evidence (the accurate normalization) -
    # The converged cloud now MAPS the posterior directly, so a moment-matched
    # ("Fisher-like") Gaussian fit to the cloud -- inflated for fat tails -- is a
    # GOOD IS proposal (unlike the earlier Hessian/MAP Fisher-IS, whose Gaussian
    # came from an off-peak curvature).  IS reweighting then gives a well-conditioned
    # normalization constant logZ = logmeanexp(lnL + log_prior - logq).  Forward-only
    # (no AD).  This is reported as the primary logZ; the raw SMC logZ is kept too.
    logZ_is = sigma_is = neff_is = np.nan
    if is_evidence and len(cloud) >= n_dim + 2:
        try:
            mu = cloud.mean(axis=0)
            C = np.atleast_2d(np.cov(cloud.T))
            Cq = (float(is_inflate) ** 2) * C + 1e-10 * np.eye(n_dim)
            Lq = np.linalg.cholesky(Cq)
            N = int(is_samples)
            th = _fix(mu[None, :] + rng.standard_normal((N, n_dim)) @ Lq.T)
            logp = _log_prior(th)
            good = np.isfinite(logp)
            lnL_is = np.full(N, -np.inf)
            if good.any():
                lnL_is[good] = _eval(like, th[good], desc="smc-IS-Z")
            logq = _gaussian_logq(th, mu, Cq)
            logw = np.where(good & np.isfinite(lnL_is), lnL_is + logp - logq, -np.inf)
            logZ_is, sigma_is, neff_is = evidence_from_logweights(logw)
            # Only TRUST the cloud-IS evidence when the proposal actually covers the
            # posterior (high ESS): excellent at high SNR (tight ~Gaussian cloud,
            # ESS~1e4) but a single Gaussian is too crude for the broad multimodal
            # low-SNR sky (ESS~30) -- there, keep the raw SMC logZ.  ESS>=2% of N is
            # the gate (a single-Gaussian over a unimodal-ish posterior gives tens of %).
            is_ok = np.isfinite(logZ_is) and neff_is >= max(500.0, 0.02 * N)
            if verbose:
                print("  [smc-IS-Z] N=%d ESS=%.0f logZ_IS=%.3f (+/- %.3g) | logZ_SMC=%.3f"
                      "  -> %s" % (N, neff_is, logZ_is, sigma_is, logZ_smc,
                                   "USE IS-Z" if is_ok else "low-ESS, keep SMC"), flush=True)
            if is_ok:
                logZ, sigma_over_Z, neff = float(logZ_is), float(sigma_is), float(neff_is)
        except Exception as e:                                  # noqa: BLE001
            if verbose:
                print("  [smc-IS-Z] failed (%r); keeping SMC logZ" % e)

    # THE LADDER CAN STOP SHORT OF THE POSTERIOR.  The loop above also exits on
    # ``stage == max_stages`` (and on a cloud with fewer than two finite lnL), and
    # the cloud then still targets ``L**inv_T * prior`` with ``inv_T < 1``.
    # Reporting ``temper=1.0`` with uniform ``post_weight`` in that case handed the
    # caller a TEMPERED cloud labelled as a posterior draw.  Report the exponent
    # actually reached, plus the correction weight ``L**(1-inv_T)`` that carries
    # the cloud to the posterior -- the same contract flowmc_sample_phimarg uses.
    # The weight is identically uniform once inv_T == 1, so the converged path is
    # unchanged.  Each rung is capped at the distance left to 1, so the clip
    # below only absorbs the rounding of the accumulated sum -- it must never be
    # covering for a ladder that genuinely ran past 1 (see the ``db`` cap above).
    lnL = np.asarray(lnL, dtype=float)
    inv_T_final = float(min(inv_T, 1.0))
    if len(lnL) and inv_T_final < 1.0:
        lw = (1.0 - inv_T_final) * lnL
        lw = np.where(np.isfinite(lw), lw, -np.inf)
        mx = np.max(lw)
        # All -inf stays all-zero: a cloud whose correction cannot be normalised
        # must be REFUSED by the caller, not quietly restored to uniform weights,
        # which is the mislabelling this block exists to prevent.
        post_weight = np.exp(lw - mx) if np.isfinite(mx) else np.zeros(len(lnL))
        s = post_weight.sum()
        if s > 0:
            post_weight = post_weight / s
    else:
        post_weight = np.ones(W) / W
    return dict(theta=cloud, lnL=lnL, logZ=float(logZ),
                sigma_over_Z=float(sigma_over_Z), neff=float(neff),
                flow_state=None, post_weight=post_weight, inv_T=inv_T_final,
                temper=(float(1.0 / inv_T_final) if inv_T_final > 0
                        else float("inf")),
                logZ_laplace=float(logZ_smc),
                lnL_map=float(np.max(lnL)) if len(lnL) else np.nan)


# ---------------------------------------------------------------------------
# 3. Fisher-preconditioned importance sampling (high-SNR)
# ---------------------------------------------------------------------------
_BOUNDS5 = [(0.0, _TWO_PI), (-_PI / 2 + 1e-3, _PI / 2 - 1e-3),
            (0.0, _PI), (1e-3, _PI - 1e-3), (0.0, _TWO_PI)]


def _multistart_map(like, n_starts, n_prior_pilot, rng):
    """Best MAP over the 5 angles from several high-lnL prior seeds."""
    from scipy.optimize import minimize
    pilot = sample_prior(n_prior_pilot, rng)
    plnL = eval_lnL(like, pilot)
    seeds = pilot[np.argsort(plnL)[::-1][:n_starts]]

    def negf(x):
        v, g = like.value_and_grad(x)
        return -float(v), -np.asarray(g)

    best = None
    for s in seeds:
        try:
            r = minimize(negf, s, jac=True, method="L-BFGS-B",
                         bounds=_BOUNDS5, options={"maxiter": 300})
        except Exception:
            continue
        if best is None or -r.fun > best[1]:
            best = (r.x, -r.fun)
    if best is None:
        i = int(np.argmax(plnL)); return pilot[i], float(plnL[i])
    return best


def _wrap_angles(theta):
    """Wrap the periodic angles (ra, phiref mod 2pi; psi mod pi) into range.
    dec, incl are non-periodic and left for the prior bounds to reject."""
    t = np.array(theta, dtype=float)
    t[:, 0] = np.mod(t[:, 0], _TWO_PI)      # ra
    t[:, 2] = np.mod(t[:, 2], _PI)          # psi
    t[:, 4] = np.mod(t[:, 4], _TWO_PI)      # phiref
    return t


def fisher_is_sample(like, n_samples=20000, n_starts=16, n_prior_pilot=20000,
                     max_std=2.5, inflate=1.6, seed=0, chunk=8000,
                     verbose=False):
    """Fisher-preconditioned importance sampling of the angular posterior.

    The differentiable likelihood makes the MAP and the local curvature
    (observed Fisher = -Hessian of lnL) directly available, so at high SNR --
    where the posterior is sharply peaked and well approximated by its Laplace
    expansion -- we can build an importance proposal that *matches the posterior
    shape* rather than hunting for it:

      1. multi-start gradient ascent -> the MAP ``theta*`` (dominant mode);
      2. observed Fisher ``F = -Hessian(lnL)`` at ``theta*`` (AD Hessian),
         symmetrized and eigenvalue-floored (the psi/phi_ref degeneracy gives
         near-flat directions; the floor caps their proposal width);
      3. propose ``theta ~ N(theta*, inflate * F^{-1})`` -- whitened to the local
         curvature, so it is narrow in well-constrained directions (sky at high
         SNR) and broad in degenerate ones;
      4. importance weights ``w propto L(theta) pi(theta) / q(theta)``.

    Domain of validity (important).  This works cleanly when the posterior is
    locally Gaussian -- i.e. when the constrained directions dominate.  It is
    LIMITED by genuine non-Gaussian degeneracies: the polarization / orbital-
    phase (psi, phi_ref) degeneracy of the quadrupole likelihood is a *curved*
    near-flat ridge that a single Gaussian proposal cannot follow, so the
    importance weights degrade (low neff) at high SNR even though the MAP and the
    sky curvature are correct.  The returned ``neff`` self-diagnoses this: where
    it is small, fold the known phase/polarization degeneracy first (see
    :func:`polarization_phase_fold`) or use the gradient-MCMC variant
    :func:`fisher_nuts_sample`, which follows the curved ridge.  Periodic angles
    are wrapped; ``inflate`` / ``max_std`` trade coverage against efficiency.

    Returns dict: ``theta`` (N,5), ``lnL`` (N,), ``post_weight`` (N,), ``logZ``,
    ``sigma_over_Z``, ``neff``, ``theta_map`` (5,), ``cov`` (5,5).
    """
    rng = np.random.default_rng(seed)
    th0, lnL0 = _multistart_map(like, n_starts, n_prior_pilot, rng)
    if verbose:
        print("  MAP lnL=%.3f at sky (RA,DEC)=(%.4f,%.4f)" % (lnL0, th0[0], th0[1]))

    # Proposal covariance = inflate * F^{-1}, but with the per-direction VARIANCE
    # *capped* at max_std^2.  The psi/phi_ref degeneracy gives near-flat (tiny or
    # slightly-negative) Fisher eigenvalues whose F^{-1} variance would be
    # enormous/ill-defined; capping at the prior scale (~max_std rad) makes the
    # proposal broad-but-bounded along the ridge (covering the degenerate
    # posterior) while staying tight (~1/SNR) in the well-constrained directions.
    F = np.asarray(like.fisher(th0)); F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)
    var = inflate / np.clip(w, inflate / max_std ** 2, None)   # cap variance
    cov = (V * var) @ V.T
    cov = 0.5 * (cov + cov.T)
    Lc = np.linalg.cholesky(cov + 1e-12 * np.eye(5) * np.trace(cov) / 5)

    z = rng.standard_normal((n_samples, 5))
    theta = _wrap_angles(th0[None, :] + z @ Lc.T)
    logq = _gaussian_logq(th0[None, :] + z @ Lc.T, th0, cov)  # q on the raw draw
    logp = log_prior(theta)
    valid = np.isfinite(logp)
    lnL = np.full(n_samples, -np.inf)
    for i in range(0, n_samples, chunk):
        sl = slice(i, i + chunk)
        v = valid[sl]
        if v.any():
            idx = np.where(v)[0] + i
            lnL[idx] = eval_lnL(like, theta[idx])
    logw = np.where(valid, lnL + logp - logq, -np.inf)
    logZ, sigma_over_Z, neff = evidence_from_logweights(logw)
    logZ, sigma_over_Z, neff = _finalize_evidence(
        logZ, sigma_over_Z, neff, float(lnL0))
    fin = np.isfinite(logw)
    pw = np.zeros(n_samples)
    if fin.any():
        m = np.max(logw[fin])
        pw[fin] = np.exp(logw[fin] - m); pw /= pw.sum()
    if verbose:
        print("  Fisher-IS: neff=%.1f / %d  logZ=%.3f" % (neff, n_samples, logZ))
    return dict(theta=theta, lnL=lnL, post_weight=pw, logZ=logZ,
                sigma_over_Z=sigma_over_Z, neff=neff, theta_map=th0, cov=cov)


def fisher_nuts_sample(like, num_warmup=300, num_samples=1500, num_chains=4,
                       n_starts=12, n_prior_pilot=20000, max_std=2.5,
                       target_accept=0.85, seed=0, verbose=False):
    """Fisher-WHITENED NUTS -- the high-SNR "superb sampling" path.

    A single Gaussian (importance) proposal cannot follow the *curved*
    polarization/orbital-phase degeneracy ridge, so its weights collapse at high
    SNR.  Instead we use the AD Fisher only to *precondition* the geometry and
    let gradient MCMC do the sampling:

      1. multi-start MAP -> theta*, observed Fisher F = -Hessian(lnL) there;
      2. build a whitening map ``theta = theta* + A y`` with ``A = V diag(sqrt(v))``,
         ``v = clip(1/eig(F), max_std^2)`` -- so each direction is O(1) in ``y``
         (tight, ~1/SNR, directions and capped-broad degenerate directions alike);
      3. run NUTS on ``y`` (target = lnL(theta(y)) + log pi(theta(y))).  In the
         whitened frame the posterior has unit scale, so NUTS keeps a healthy
         step size *at every SNR* (no vanishing-step slowdown), and -- being
         gradient MCMC, not importance sampling -- it follows the curved ridge
         and resolves the narrow sky without collapse.

    Returns dict: ``theta`` (N,5), ``lnL`` (N,), ``post_weight`` (uniform),
    ``theta_map``, ``cov``, ``neff`` (== N; MCMC draws are posterior samples).

    Cost caveat: NUTS makes many gradient evaluations per sample, and each
    distance-marginalized evaluation is ~milliseconds on CPU, so this is
    GPU-territory for production high-SNR use -- on CPU it is correct but slow.
    Whitening keeps the *conditioning* (step size) healthy at any SNR; it does
    not reduce the per-evaluation cost.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    th0, lnL0 = _multistart_map(like, n_starts, n_prior_pilot, rng)
    F = np.asarray(like.fisher(th0)); F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)
    v = np.clip(1.0 / np.clip(w, 1.0 / max_std ** 2, None), 0.0, max_std ** 2)
    A = jnp.asarray((V * np.sqrt(v)))          # theta = th0 + A @ y
    th0j = jnp.asarray(th0)
    _ra_lo, _ra_hi = 0.0, _TWO_PI

    def model():
        y = numpyro.sample("y", dist.Normal(0.0, 4.0).expand([5]))
        th = th0j + A @ y
        # physical log-prior (uniform sphere/orientation) as a factor; the
        # vague N(0,4) on y is ~flat over the O(1) whitened posterior.
        dec, incl = th[1], th[3]
        in_dec = (dec > -_PI / 2) & (dec < _PI / 2)
        in_incl = (incl > 0.0) & (incl < _PI)
        logpri = jnp.where(in_dec & in_incl,
                           jnp.log(jnp.clip(jnp.cos(dec), 1e-30, None))
                           + jnp.log(jnp.clip(jnp.sin(incl), 1e-30, None)),
                           -1e10)
        lnL = like._scalar(jnp.stack([th[0], dec, th[2], incl, th[4]]))
        numpyro.factor("post", lnL + logpri + 0.5 * jnp.sum((y / 4.0) ** 2))

    kernel = NUTS(model, target_accept_prob=target_accept,
                  init_strategy=init_to_value(values={"y": np.zeros(5)}))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="sequential",
                progress_bar=verbose)
    mcmc.run(jax.random.PRNGKey(seed))
    y = np.asarray(mcmc.get_samples()["y"])               # (N,5)
    theta = _wrap_angles(np.asarray(th0)[None, :] + y @ np.asarray(A).T)
    lnL = eval_lnL(like, theta)
    if verbose:
        print("  Fisher-NUTS: %d samples, MAP lnL=%.2f" % (len(theta), lnL0))
    return dict(theta=theta, lnL=lnL,
                post_weight=np.full(len(theta), 1.0 / max(len(theta), 1)),
                logZ=np.nan, sigma_over_Z=np.nan, neff=float(len(theta)),
                theta_map=th0, cov=(V * v) @ V.T)


def _pad_phi(theta4):
    """Pad ``(N, 4)`` theta4 with a zero phiref column so the 5-D angular-
    distance helpers (:func:`_pick_well_separated`, :func:`cluster_modes`)
    apply unchanged (the dphi term is then identically zero)."""
    theta4 = np.atleast_2d(theta4)
    return np.concatenate([theta4, np.zeros((len(theta4), 1))], axis=1)


def _multistart_map_4(like, n_starts, n_prior_pilot, rng, min_sep=0.3,
                      n_modes=4, verbose=False):
    """Distinct local maxima of the 4-D (dist+phi-marginalised) lnL.

    Pilot prior scan -> well-separated high-lnL seeds -> L-BFGS-B (AD
    gradient) from each -> cluster the optimized endpoints into distinct
    modes.  Returns ``(modes (k,4), mode_lnL (k,))`` with ``k <= n_modes``,
    sorted by lnL (best first).
    """
    from scipy.optimize import minimize
    pilot = sample_prior_4(max(n_prior_pilot, n_starts), rng)
    plnL = eval_lnL_4(like, pilot, desc="pilot")
    seeds5, _ = _pick_well_separated(_pad_phi(pilot), plnL, n_starts,
                                     min_sep=min_sep)
    seeds = seeds5[:, :4]

    def negf(x):
        v, g = like.value_and_grad(x)
        return -float(v), -np.asarray(g)

    ends, ends_lnL = [], []
    for s in seeds:
        try:
            r = minimize(negf, s, jac=True, method="L-BFGS-B",
                         bounds=_BOUNDS4, options={"maxiter": 300})
            ends.append(r.x); ends_lnL.append(-r.fun)
        except Exception:
            continue
    if not ends:
        i = int(np.argmax(plnL))
        return pilot[i:i + 1], plnL[i:i + 1]
    ends = np.asarray(ends); ends_lnL = np.asarray(ends_lnL)

    # cluster optimizer endpoints -> distinct modes, best-first
    modes, mode_lnL = [], []
    for rep, idx in cluster_modes(_pad_phi(ends), min_sep=min_sep):
        j = idx[np.argmax(ends_lnL[idx])]
        modes.append(ends[j]); mode_lnL.append(ends_lnL[j])
    order = np.argsort(mode_lnL)[::-1][:n_modes]
    modes = np.asarray(modes)[order]; mode_lnL = np.asarray(mode_lnL)[order]
    # drop modes overwhelmingly below the best (no posterior mass)
    keep = mode_lnL > mode_lnL[0] - 30.0
    if verbose:
        print("  multistart MAP: %d distinct modes (lnL: %s); keeping %d"
              % (len(modes), np.array2string(mode_lnL, precision=1),
                 int(keep.sum())))
    return modes[keep], mode_lnL[keep]


def fisher_nuts_sample_phimarg(like, num_warmup=300, num_samples=1000,
                               n_starts=12, n_modes=4, n_prior_pilot=20000,
                               max_std=2.5, target_accept=0.85, n_is=40000,
                               seed=0, verbose=False):
    """Fisher-WHITENED NUTS on the 4-D (distance+phi_ref)-marginalised posterior.

    The principled high-SNR path: no tempering, no importance-reweighting of a
    broadened target.  Combines the two existing ingredients:

      * :class:`~RIFT.likelihood.jax_ile.wrapper.JAXDistPhiMargLikelihood`
        removes the curved psi/phi_ref ridge (the failure mode of the 5-D
        Fisher proposal), leaving a 4-D posterior whose modes are locally
        Gaussian at high SNR;
      * Fisher whitening (as :func:`fisher_nuts_sample`) maps each mode to
        O(1) scale so NUTS keeps a healthy step size at ANY SNR -- the sky
        ring narrowing as 1/SNR no longer shrinks the step.

    Multimodality (the discrete time-delay sky modes) is handled by
    multi-start: distinct AD-gradient MAP modes are found first
    (:func:`_multistart_map_4`), one whitened NUTS chain runs per mode, and
    chains are pooled with per-mode posterior-mass weights.

    Evidence: a Gaussian-mixture importance estimate (one moment-matched,
    inflated component per mode chain), as :func:`multistart_nuts` -- NOT the
    single-Gaussian estimator of :func:`flowmc_sample`, which collapses on a
    multimodal ring.

    Returns dict: ``theta`` (N,4), ``lnL`` (N,), ``post_weight`` (N,; uniform
    within a chain, proportional to the mode's mass across chains), ``logZ``,
    ``sigma_over_Z``, ``neff``, ``theta_map`` (4,), ``modes`` (k,4),
    ``mode_lnL`` (k,), ``mode_logZ`` (k,).
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value

    rng = np.random.default_rng(seed)
    _warmup_compile(like, verbose=verbose)
    modes, mode_lnL = _multistart_map_4(like, n_starts, n_prior_pilot, rng,
                                        n_modes=n_modes, verbose=verbose)
    K = len(modes)

    per_chain = []
    for k in range(K):
        th0 = modes[k]
        F = np.asarray(like.fisher(th0)); F = 0.5 * (F + F.T)
        w, V = np.linalg.eigh(F)
        v = np.clip(1.0 / np.clip(w, 1.0 / max_std ** 2, None),
                    0.0, max_std ** 2)
        A = jnp.asarray(V * np.sqrt(v))         # theta = th0 + A @ y
        th0j = jnp.asarray(th0)

        def model():
            y = numpyro.sample("y", dist.Normal(0.0, 4.0).expand([4]))
            th = th0j + A @ y
            dec, incl = th[1], th[3]
            in_dec = (dec > -_PI / 2) & (dec < _PI / 2)
            in_incl = (incl > 0.0) & (incl < _PI)
            logpri = jnp.where(
                in_dec & in_incl,
                jnp.log(jnp.clip(jnp.cos(dec), 1e-30, None))
                + jnp.log(jnp.clip(jnp.sin(incl), 1e-30, None)),
                -1e10)
            lnL = like._scalar(jnp.stack([th[0], dec, th[2], incl]))
            # undo the N(0,4) pseudo-prior so the target is exactly lnL+logpri
            numpyro.factor("post", lnL + logpri
                           + 0.5 * jnp.sum((y / 4.0) ** 2))

        kernel = NUTS(model, target_accept_prob=target_accept,
                      init_strategy=init_to_value(values={"y": np.zeros(4)}))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=verbose)
        mcmc.run(jax.random.PRNGKey(seed + 1 + k))
        y = np.asarray(mcmc.get_samples()["y"])
        th = th0[None, :] + y @ np.asarray(A).T
        th[:, 0] = np.mod(th[:, 0], _TWO_PI)     # wrap ra
        th[:, 2] = np.mod(th[:, 2], _PI)         # wrap psi
        per_chain.append(th)
        if verbose:
            print("  mode %d/%d: NUTS done (MAP lnL=%.2f)" %
                  (k + 1, K, mode_lnL[k]))

    theta = np.concatenate(per_chain, axis=0)
    lnL = eval_lnL_4(like, theta, desc="reweight")

    # -- evidence: Gaussian-mixture IS, one component per mode chain --------
    mus, covs = [], []
    for th in per_chain:
        if len(th) >= 6:
            mu, cov = _moment_match(th, np.zeros(len(th)))
            mus.append(mu); covs.append(cov * 2.0)
    if not mus:
        mu, cov = _moment_match(theta, np.zeros(len(theta)))
        mus, covs = [mu], [cov * 2.0]
    weights = np.full(len(mus), 1.0 / len(mus))

    counts = rng.multinomial(n_is, weights)
    draws, comp_of_draw = [], []
    for c in range(len(mus)):
        if counts[c] == 0:
            continue
        Lc = np.linalg.cholesky(covs[c] + 1e-12 * np.eye(4))
        z = rng.standard_normal((counts[c], 4))
        draws.append(mus[c][None, :] + z @ Lc.T)
        comp_of_draw.append(np.full(counts[c], c))
    th_is = np.concatenate(draws, axis=0)
    comp_of_draw = np.concatenate(comp_of_draw)

    logq = _mixture_logq(th_is, mus, covs, weights)
    logp = log_prior_4(th_is)
    valid = np.isfinite(logp)
    lnL_is = np.full(len(th_is), -np.inf)
    if valid.any():
        lnL_is[valid] = eval_lnL_4(like, th_is[valid], desc="evidence")
    logw = np.where(valid, lnL_is + logp - logq, -np.inf)
    logZ, sigma_over_Z, neff = evidence_from_logweights(logw)
    logZ, sigma_over_Z, neff = _finalize_evidence(
        logZ, sigma_over_Z, neff, float(np.max(lnL)) if len(lnL) else np.nan)

    # -- per-mode mass -> chain weights (pooled chains are equal-length, so
    # subdominant modes are over-represented; reweight by mode evidence).
    # Mass of mode k ~ sum of IS weights from draws nearest to component k
    # (components are well-separated, so component label ~ mode label).
    mode_logZ = np.full(len(mus), -np.inf)
    fin = np.isfinite(logw)
    if fin.any():
        m = np.max(logw[fin])
        for c in range(len(mus)):
            sel = fin & (comp_of_draw == c)
            if sel.any():
                mode_logZ[c] = m + np.log(np.sum(np.exp(logw[sel] - m))
                                          / max(counts[c], 1)) \
                               + np.log(weights[c])
    if np.isfinite(mode_logZ).any():
        lw = mode_logZ - np.max(mode_logZ[np.isfinite(mode_logZ)])
        mass = np.where(np.isfinite(lw), np.exp(lw), 0.0)
    else:
        mass = np.ones(len(mus))
    mass = mass / mass.sum()
    post_weight = np.concatenate([
        np.full(len(per_chain[c]), mass[c] / max(len(per_chain[c]), 1))
        for c in range(len(per_chain))])
    post_weight = post_weight / post_weight.sum()

    if verbose:
        print("  Fisher-NUTS(phimarg): %d draws over %d modes; "
              "logZ=%.3f  neff(IS)=%.1f  mode mass=%s"
              % (len(theta), K, logZ, neff,
                 np.array2string(mass, precision=3)))
    theta_per_chain = per_chain    # list of (num_samples, 4) arrays, one per mode
    return dict(theta=theta, lnL=lnL, post_weight=post_weight, logZ=logZ,
                sigma_over_Z=sigma_over_Z, neff=neff,
                theta_map=modes[0], modes=modes, mode_lnL=mode_lnL,
                mode_logZ=mode_logZ, theta_per_chain=theta_per_chain,
                flow_state=None)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _build_standard_injection_likelihood(verbose=False):
    """Build JAXDistanceMarginalizedLikelihood for the standard test injection.

    Mirrors ``test/jax/test_jax_endtoend.py``: m1=35, m2=30, s1z=0.1, s2z=-0.2,
    H1/L1/V1, IMRPhenomD, fiducial_epoch=1126259462.0, truth sky (1.2, -0.4).
    """
    import lal
    import lalsimulation as lalsim
    import RIFT.lalsimutils as lalsimutils
    from RIFT.likelihood.jax_ile import build_data_from_precompute
    from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood

    MSUN = lal.MSUN_SI
    PC = lal.PC_SI

    fiducial_epoch = 1126259462.0
    detectors = ["H1", "L1", "V1"]
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = 35.0 * MSUN
    P.m2 = 30.0 * MSUN
    P.s1z = 0.1
    P.s2z = -0.2
    P.fmin = 30.0
    P.fref = 30.0
    P.deltaT = 1.0 / 4096
    P.deltaF = 1.0 / 4
    P.dist = 600.0 * 1e6 * PC
    P.fmax = 0.0
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = fiducial_epoch
    P.phi = 1.2       # RA truth
    P.theta = -0.4    # DEC truth
    P.psi = 0.7
    P.incl = 0.9
    P.phiref = 2.1

    data_dict, psd_dict = {}, {}
    for det in detectors:
        Pdet = P.copy()
        Pdet.detector = det
        data_dict[det] = lalsimutils.non_herm_hoff(Pdet)
        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower

    data, _ = build_data_from_precompute(
        P.copy(), data_dict, psd_dict, fiducial_epoch,
        storage_window_half=0.15, integration_window_half=0.075,
        Lmax=2, fMax=1000.0, analyticPSD_Q=True, verbose=verbose)

    like = JAXDistanceMarginalizedLikelihood(data, d_min=1.0, d_max=5000.0,
                                             n_grid=128)
    return like


def _selftest():
    import time
    jax.config.update("jax_enable_x64", True)

    print("Building standard injection likelihood (truth sky RA,DEC=(1.20,-0.40)) ...")
    like = _build_standard_injection_likelihood(verbose=False)

    d_min, d_max = 1.0, 5000.0

    # Each NUTS leapfrog step evaluates the (distance-grid) likelihood + its
    # gradient and is fairly slow on CPU, and warmup spends many steps adapting
    # the step size from a tiny initial value; keep the per-chain budget modest
    # so the whole self-test (pilot + n_starts chains + IS evidence) stays well
    # under ~10 min.  Chain progress bars are off (printed per-chain instead) to
    # avoid their overhead dominating the wall time.
    print("\n=== multistart_nuts (n_starts=3, warmup=120, samples=200) ===")
    t0 = time.time()
    res = multistart_nuts(like, d_min, d_max, n_starts=3, num_warmup=120,
                          num_samples=200, n_prior_pilot=8000, seed=0,
                          verbose=True)
    dt = time.time() - t0

    theta, lnL = res["theta"], res["lnL"]
    print("\n-- results --")
    print("  pooled samples: %d   max lnL=%.3f" % (len(theta), lnL.max()))
    best = theta[np.argmax(lnL)]
    print("  peak at: " + ", ".join("%s=%.4f" % (nm, v)
                                     for nm, v in zip(ANG_NAMES, best)))

    # distinct sky modes among the seeds and pooled samples
    print("\n  seeds (RA, DEC, lnL):")
    for s, sl in zip(res["seeds"], res["seed_lnL"]):
        print("     (%.3f, %.3f)  lnL=%.2f" % (s[0], s[1], sl))

    clusters = cluster_modes(theta, min_sep=0.5)
    print("\n  distinct modes among pooled samples (%d found):" % len(clusters))
    for rep, idx in clusters[:8]:
        sub = lnL[idx]
        print("     (RA,DEC)=(%.3f,%.3f)  n=%5d  max lnL=%.2f" %
              (rep[0], rep[1], len(idx), sub.max()))

    # how close is the best mode to the truth SKY (1.2, -0.4)?  Use a sky-only
    # great-circle distance: the orientation angles (psi,incl,phiref) have their
    # own (phase/polarization) degeneracies, so the highest-lnL point legitimately
    # sits at a different orientation -- what matters for "recovery" is the sky.
    def _sky_only(t, truth=np.array([1.2, -0.4])):
        dlon = t[0] - truth[0]
        return float(np.arccos(np.clip(
            np.sin(t[1]) * np.sin(truth[1])
            + np.cos(t[1]) * np.cos(truth[1]) * np.cos(dlon), -1.0, 1.0)))
    dsky = _sky_only(best)
    near = any(_sky_only(rep) < 0.1 for rep, _ in clusters)
    print("\n  best-mode SKY distance to truth=%.4f rad   "
          "truth-sky mode recovered: %s" % (dsky, near))

    print("\n  evidence: logZ=%.4f   sigma/Z=%.4f   neff=%.1f" %
          (res["logZ"], res["sigma_over_Z"], res["neff"]))
    print("  multistart_nuts wall time: %.1f s" % dt)

    # flowMC (best-effort; report status, do not fail the self-test on it)
    print("\n=== flowmc_sample (short run) ===")
    try:
        t0 = time.time()
        fres = flowmc_sample(like, d_min, d_max, n_chains=20, n_local_steps=20,
                             n_global_steps=20, n_training_loops=3,
                             n_production_loops=3, n_epochs=5, seed=1,
                             verbose=False)
        dt = time.time() - t0
        fclusters = cluster_modes(fres["theta"], min_sep=0.5)
        print("  flowMC: %d production samples; max lnL=%.3f; %d modes; "
              "logZ=%.4f neff=%.1f  (%.1f s)" %
              (len(fres["theta"]), float(np.max(fres["lnL"])) if len(fres["lnL"]) else float("nan"),
               len(fclusters), fres["logZ"], fres["neff"], dt))
    except Exception as e:
        import traceback
        print("  flowMC run raised: %r" % e)
        traceback.print_exc()

    print("\nSELF-TEST COMPLETE")


if __name__ == "__main__":
    _selftest()
