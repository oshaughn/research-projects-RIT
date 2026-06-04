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

import numpy as np

import jax
import jax.numpy as jnp

# Parameter order used everywhere in this module.
ANG_NAMES = ("ra", "dec", "psi", "incl", "phiref")
_TWO_PI = float(2 * np.pi)
_PI = float(np.pi)


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
def eval_lnL(like, theta, chunk=4000):
    """Evaluate the distance-marginalized lnL on an ``(N, 5)`` array in chunks.

    Chunking bounds peak device memory (the distance grid multiplies the batch
    dimension inside the likelihood).
    """
    theta = np.atleast_2d(theta)
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
    """Weighted mean/cov of ``theta`` with weights ``propto exp(logL)``."""
    w = np.exp(logL - np.max(logL))
    w = w / np.sum(w)
    mu = np.sum(w[:, None] * theta, axis=0)
    d = theta - mu[None, :]
    cov = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(axis=0)
    cov += 1e-9 * np.eye(theta.shape[1]) * (np.trace(cov) / theta.shape[1] + 1e-12)
    return mu, cov


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
    if net is None:
        def model():
            ra = numpyro.sample("ra", dist.Uniform(0.0, _TWO_PI))
            sin_dec = numpyro.sample("sin_dec", dist.Uniform(-1.0, 1.0))
            psi = numpyro.sample("psi", dist.Uniform(0.0, _PI))
            cos_incl = numpyro.sample("cos_incl", dist.Uniform(-1.0, 1.0))
            phiref = numpyro.sample("phiref", dist.Uniform(0.0, _TWO_PI))
            lnL = like._scalar(jnp.stack(
                [ra, jnp.arcsin(sin_dec), psi, jnp.arccos(cos_incl), phiref]))
            numpyro.factor("loglike", lnL)

        def make_init(th0):
            return {"ra": float(th0[0]), "sin_dec": float(np.sin(th0[1])),
                    "psi": float(th0[2]), "cos_incl": float(np.cos(th0[3])),
                    "phiref": float(th0[4])}

        def extract(s):
            return np.stack([np.asarray(s["ra"]), np.arcsin(np.asarray(s["sin_dec"])),
                             np.asarray(s["psi"]), np.arccos(np.asarray(s["cos_incl"])),
                             np.asarray(s["phiref"])], axis=-1)
    else:
        _C, R, gmst = net

        def model():
            cos_tn = numpyro.sample("cos_theta_n", dist.Uniform(-1.0, 1.0))
            phi_n = numpyro.sample("phi_n", dist.Uniform(0.0, _TWO_PI))
            psi = numpyro.sample("psi", dist.Uniform(0.0, _PI))
            cos_incl = numpyro.sample("cos_incl", dist.Uniform(-1.0, 1.0))
            phiref = numpyro.sample("phiref", dist.Uniform(0.0, _TWO_PI))
            ra, dec = _C.network_to_equatorial(jnp.arccos(cos_tn), phi_n, R, gmst)
            lnL = like._scalar(jnp.stack([ra, dec, psi, jnp.arccos(cos_incl), phiref]))
            numpyro.factor("loglike", lnL)

        def make_init(th0):
            tn, pn = _C.equatorial_to_network(float(th0[0]), float(th0[1]), R, gmst)
            return {"cos_theta_n": float(np.cos(float(tn))),
                    "phi_n": float(float(pn) % _TWO_PI),
                    "psi": float(th0[2]), "cos_incl": float(np.cos(th0[3])),
                    "phiref": float(th0[4])}

        def extract(s):
            tn = np.arccos(np.asarray(s["cos_theta_n"]))
            ra, dec = _C.network_to_equatorial(tn, np.asarray(s["phi_n"]), R, gmst)
            return np.stack([np.asarray(ra), np.asarray(dec), np.asarray(s["psi"]),
                             np.arccos(np.asarray(s["cos_incl"])),
                             np.asarray(s["phiref"])], axis=-1)

    # -- 2. one NUTS chain per seed, pooled --------------------------------
    per_chain = []     # (num_samples, 5) per seed, in equatorial theta5
    for k in range(n_starts):
        kernel = NUTS(model, target_accept_prob=target_accept,
                      init_strategy=init_to_value(values=make_init(seeds[k])))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=chain_progress_bar)
        mcmc.run(jax.random.PRNGKey(seed + 1 + k))
        per_chain.append(extract(mcmc.get_samples()))
        if verbose:
            print("  chain %d/%d done (seed lnL=%.2f)" %
                  (k + 1, n_starts, seed_lnL[k]))

    theta = np.concatenate(per_chain, axis=0)
    lnL = eval_lnL(like, theta)

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

    return dict(theta=theta, lnL=lnL, seeds=seeds, seed_lnL=seed_lnL,
                logZ=logZ, sigma_over_Z=sigma_over_Z, neff=neff)


# ---------------------------------------------------------------------------
# 2. flowMC
# ---------------------------------------------------------------------------
def flowmc_sample(like, d_min, d_max, n_chains=20, n_local_steps=20,
                  n_global_steps=20, n_training_loops=4, n_production_loops=4,
                  n_epochs=10, n_prior_pilot=8000, seed=0, mala_step_size=0.01,
                  reuse_state=None, verbose=False):
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
    initialization -- the partial-flow-reuse efficiency gain (most visible at
    scale, and at high SNR where peaks are narrow).  Re-use degrades gracefully:
    a model-shape/version mismatch falls back to a fresh flow, mismatched
    ``positions`` fall back to a high-lnL prior draw.

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

    def logpdf(theta5, data):
        # JAX-traceable target: lnL + log-prior (both finite-on-support).
        return like._scalar(theta5) + _log_prior_jax(theta5)

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

    return dict(theta=theta, lnL=lnL, logZ=logZ, sigma_over_Z=sigma_over_Z,
                neff=neff, flow_state=flow_state)


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
