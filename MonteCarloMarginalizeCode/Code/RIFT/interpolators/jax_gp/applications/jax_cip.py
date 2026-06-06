"""
jax_cip -- a clean, standalone pure-JAX posterior-construction path.

This is the deliberate alternative to shoehorning JAX into the 2000-line legacy
CIP: a small, self-contained tool that does the whole intrinsic-posterior job in
JAX so we can **sample in real (physical) coordinates** with a derivative-aware
sampler. It does NOT touch the legacy CIP code path.

Pipeline:
  1. load a RIFT ILE .net file (with per-point MC errors),
  2. map to good fit coordinates (BNS: mu1,mu2,delta_mc,LambdaTilde,DeltaLambdaTilde),
  3. fit a differentiable RFF lnL surrogate (using the MC errors),
  4. run numpyro NUTS in the DECORRELATED FIT coordinates (gradient-based),
  5. expose a differentiable lnL in *physical* params (m1,m2,s1z,s2z,lambda1,lambda2)
     via coordinates.physical_lnL -- the hook for AD population inference.

Lesson baked in: sample in the fit coordinates, not raw (m1,m2). The mu1/mu2
coordinates exist precisely to remove the sharp curved chirp-mass degeneracy; a
gradient sampler with a diagonal metric mixes well there but chokes on that curved
ridge in physical coordinates (use ``--physical-sampling`` to see the poor ESS).
The physical-parameter gradient is still produced -- it is what a population
analysis needs, evaluated at in-support points -- just not used as the sampling
geometry.

The legacy CIP remains the production path; this is the JAX-native track for the
AD use cases (differentiable sampling, population inference), which are
qualitatively different and cleaner to manage on their own.

CLI: this tool accepts the legacy CIP argument surface *inclusively*
(parse_known_args swallows args it does not use), reads ``--fname``, honours
``--parameter`` / ``--parameter-implied`` / ``--parameter-nofit`` / ``--lnL-offset``
/ ``--cap-points`` / ``--sigma-cut`` / ``--n-output-samples``, and writes the same
output the pipeline consumes (``--fname-output-samples`` -> ChooseWaveformParams
XML + ``_lnL.dat``). So the SAME pipeline command line can hot-swap legacy CIP for
this path.

STATUS: the I/O / coordinate / CLI contract is the deliverable and is exercised
end-to-end. Posterior-sample *quality* is still WIP -- NUTS mixing (ESS) on the
sharp real-BNS RFF surrogate is poor (the surrogate is not perfectly smooth and the
posterior is multiscale); it localizes the right region (mc recovered to ~1e-4) but
the samples are autocorrelated. Improving this (smoother surrogate; flow sampler;
better priors/Jacobians) is the science-grade follow-up tracked in DESIGN.md.

Note: performance numbers here are on an old CPU box; production hardware is far
faster -- treat timings as relative, not absolute.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import RIFT.interpolators.jax_gp as jax_gp  # enables float64
from RIFT.interpolators.jax_gp import get_interpolator, coordinates
from RIFT.interpolators.jax_gp.benchmark.datasets import (
    load_ile_net, to_fit_coordinates, mc_delta_from_m1m2, BNS_FIT_COORDS)

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import effective_sample_size

PHYS = ["m1", "m2", "s1z", "s2z", "lambda1", "lambda2"]

# Default BNS coordinate system (matches the recommended CIP flags):
#   --parameter delta_mc --parameter-implied mu1 mu2 LambdaTilde DeltaLambdaTilde
#   --parameter-nofit mc s1z s2z lambda1 lambda2
DEFAULT_FIT_COORDS = list(BNS_FIT_COORDS)                       # what the GP fits
DEFAULT_LOW_LEVEL = ["mc", "delta_mc", "s1z", "s2z", "lambda1", "lambda2"]  # what we sample


def _fit_surrogate(net_path, sigma_cut, lnL_offset, cap_points,
                   n_features, n_opt_steps, seed):
    """Load + good-coord transform + RFF fit. Returns (model, Xphys_kept, y_kept)."""
    X6, y, yerr, _ = load_ile_net(net_path, sigma_cut=sigma_cut, return_errors=True)
    m1, m2, s1z, s2z, l1, l2 = X6.T
    mc, dmc = mc_delta_from_m1m2(m1, m2)
    Xfit = to_fit_coordinates(
        np.column_stack([mc, dmc, s1z, s2z, l1, l2]),
        ["mc", "delta_mc", "s1z", "s2z", "lambda1", "lambda2"], BNS_FIT_COORDS)

    # Tree-ring (lnL-band stratified) downselection (demo path; default rings).
    sel = _tree_ring_select(y, cap_points if cap_points > 0 else len(y), seed=seed)
    X6, Xfit, y, yerr = X6[sel], Xfit[sel], y[sel], yerr[sel]

    model = get_interpolator("rff")(n_features=n_features, n_opt_steps=n_opt_steps,
                                    seed=seed).fit(Xfit, y, y_errors=yerr)
    model.coord_names = list(BNS_FIT_COORDS)   # needed by coordinates.physical_lnL
    return model, X6, Xfit, y


def sample_lnL(lnL_fn, loc, scale, bounds=None, num_warmup=500, num_samples=2000,
               seed=0):
    """numpyro NUTS over an arbitrary lnL callable.

    Prior: a Uniform over ``bounds=(low, high)`` if given (keeps the sampler within
    the surrogate's training support -- essential for weakly-constrained directions
    where the posterior is prior-dominated and would otherwise drift into GP
    extrapolation), else a broad ``Normal(loc, scale)``.
    """
    loc = jnp.asarray(loc); scale = jnp.asarray(scale)

    if bounds is not None:
        low = jnp.asarray(bounds[0]); high = jnp.asarray(bounds[1])
        prior = dist.Uniform(low, high).to_event(1)
    else:
        prior = dist.Normal(loc, scale).to_event(1)

    def numpyro_model():
        theta = numpyro.sample("theta", prior)
        numpyro.factor("lnL", lnL_fn(theta))

    # dense_mass: the intrinsic posterior is multiscale + correlated (mc razor-sharp,
    # tides broad); a dense mass matrix adapts to that geometry, where a diagonal one
    # mixes poorly. Higher target_accept stabilizes the sharp directions.
    kernel = NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed))
    samples = np.asarray(mcmc.get_samples()["theta"])
    wall = time.time() - t0
    ess = np.asarray(effective_sample_size(samples[None, ...]))
    return {"samples": samples, "wall_clock": wall, "ess_min": float(np.min(ess)),
            "mean": samples.mean(0), "std": samples.std(0)}


def sample_nuts_muframe(lnL_fn, gmean, gcov, lo, hi, num_warmup=1000,
                        num_samples=4000, num_chains=2, target_accept=0.95,
                        adapt_mass=True, seed=0):
    """NUTS in low-level coords, *preconditioned* with the mu-frame covariance.

    Plain NUTS stalled here (HANDOFF): the low-level posterior is a razor-thin,
    ill-conditioned ridge in mc, so warmup never found a workable step size / mass
    matrix.  ``_muframe_proposal`` gives a well-conditioned covariance ``gcov`` in
    *low-level* coordinates -- the sharp mc direction and its correlations are
    resolved in the decorrelated Morisaki (fit) frame, then pulled back -- so we
    seed NUTS's dense mass matrix with it.  The dynamics then see an approximately
    isotropic geometry from the first step, and, unlike importance sampling, NUTS
    explores the weakly-constrained directions (delta_mc, tides) by construction
    rather than being proposal-limited (the diagnosed IS failure mode).

    The prior is Uniform over the CLI box.  numpyro samples in the unconstrained
    reparameterization ``theta = lo + (hi-lo) sigmoid(u)``, so the seeded mass
    matrix -- specified in *u*-space -- is the low-level ``gcov`` mapped through the
    local Jacobian ``dtheta/du = (hi-lo) s (1-s)`` of that sigmoid at the peak.
    With ``adapt_mass=True`` numpyro re-adapts from this seed during warmup; the
    seed's job is to bootstrap the step-size search that previously collapsed.
    """
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    gmean = np.asarray(gmean, float); gcov = np.asarray(gcov, float)
    d = len(gmean)
    span = hi - lo
    frac = np.clip((gmean - lo) / span, 1e-4, 1 - 1e-4)
    S = span * frac * (1.0 - frac)            # d(theta)/d(u) at the peak (per coord)
    Sinv = 1.0 / S
    imm = Sinv[:, None] * gcov * Sinv[None, :]    # low-level cov -> unconstrained u-space
    imm = 0.5 * (imm + imm.T) + 1e-12 * np.eye(d)

    prior = dist.Uniform(jnp.asarray(lo), jnp.asarray(hi)).to_event(1)

    def numpyro_model():
        theta = numpyro.sample("theta", prior)
        numpyro.factor("lnL", lnL_fn(theta))

    kernel = NUTS(numpyro_model, dense_mass=True, adapt_mass_matrix=adapt_mass,
                  inverse_mass_matrix=jnp.asarray(imm),
                  target_accept_prob=target_accept,
                  init_strategy=init_to_value(values={"theta": jnp.asarray(gmean)}))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="sequential",
                progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    wall = time.time() - t0

    chains = np.asarray(mcmc.get_samples(group_by_chain=True)["theta"])   # [C, N, d]
    ess = np.asarray(effective_sample_size(chains))                       # [d]
    samples = chains.reshape(-1, d)
    n_div = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    return {"samples": samples, "wall_clock": wall,
            "ess": float(np.min(ess)), "ess_min": float(np.min(ess)),
            "ess_frac": float(np.min(ess) / samples.shape[0]),
            "ess_per_dim": ess, "n_divergences": n_div, "frac_in_box": 1.0,
            "logZ": None, "mean": samples.mean(0), "std": samples.std(0)}


def _train_box_flow(lnL_fn, lo, hi, init_theta=None, n_chains=30, n_local=40,
                    n_global=40, n_train_loops=8, n_prod_loops=2, n_epochs=12,
                    seed=0, rq_n_bins=12, rq_n_layers=6):
    """Train a flowMC normalizing flow over the prior box (sigmoid latent).

    Returns (flow, theta_of_u, log_jac, d) where theta_of_u/log_jac are pure-JAX.
    """
    import jax
    import jax.numpy as jnp
    from flowMC.Sampler import Sampler
    from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
    lo = jnp.asarray(lo); hi = jnp.asarray(hi); span = hi - lo
    d = int(lo.shape[0])

    def theta_of_u(u):
        return lo + span * jax.nn.sigmoid(u)

    def log_jac(u):
        return jnp.sum(jnp.log(span) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))

    def target(u, data=None):
        return lnL_fn(theta_of_u(u)) + log_jac(u)

    key = jax.random.PRNGKey(seed)
    key, kb, ks, ki = jax.random.split(key, 4)
    bundle = RQSpline_MALA_Bundle(
        rng_key=kb, n_chains=n_chains, n_dims=d, logpdf=target, n_local_steps=n_local,
        n_global_steps=n_global, n_training_loops=n_train_loops,
        n_production_loops=n_prod_loops, n_epochs=n_epochs,
        rq_spline_n_bins=rq_n_bins, rq_spline_n_layers=rq_n_layers)
    sampler = Sampler(d, n_chains, ks, resource_strategy_bundles=bundle)
    if init_theta is not None:
        frac = np.clip((np.asarray(init_theta, float) - np.asarray(lo))
                       / np.asarray(span), 1e-3, 1 - 1e-3)
        u0 = np.log(frac / (1 - frac))
        init = jnp.asarray(u0)[None, :] + 0.3 * jax.random.normal(ki, (n_chains, d))
    else:
        init = 0.3 * jax.random.normal(ki, (n_chains, d))
    sampler.sample(init, {})
    return sampler.resources["model"], theta_of_u, log_jac, d


def sample_mixture_is(lnL_fn, lo, hi, gmean, gcov, init_theta=None, alpha=0.5,
                      inflate=1.2, n_samples=40000, n_train_loops=8, seed=0):
    """Defensive mixture importance sampling: q = alpha N(peak, Fisher) + (1-alpha) flow.

    The Gaussian core covers the razor-sharp peak (where the flow can't learn the thin
    feature); the trained flow covers the non-Gaussian wings (where the Gaussian
    under-covers). The mixture keeps weights bounded -- no region with high target but
    ~zero proposal. Densities are combined in physical (low-level) theta space.
    """
    import jax
    import jax.numpy as jnp
    flow, theta_of_u, log_jac, d = _train_box_flow(
        lnL_fn, lo, hi, init_theta=init_theta, n_train_loops=n_train_loops, seed=seed)
    train_wall_marker = time.time()
    lo = np.asarray(lo); hi = np.asarray(hi); span = hi - lo
    gcov = inflate ** 2 * (np.asarray(gcov) + 1e-12 * np.eye(d))
    Lg = np.linalg.cholesky(gcov)

    def log_qg(th):
        dx = th - np.asarray(gmean)
        sol = np.linalg.solve(Lg, dx.T).T
        return -0.5 * np.sum(sol ** 2, 1) - np.sum(np.log(np.diag(Lg))) \
            - 0.5 * d * np.log(2 * np.pi)

    def log_qf(th):
        frac = np.clip((th - lo) / span, 1e-7, 1 - 1e-7)
        u = np.log(frac / (1 - frac))
        lqu = np.asarray(jax.vmap(flow.log_prob)(jnp.asarray(u)))
        lj = np.asarray(jax.vmap(log_jac)(jnp.asarray(u)))
        return lqu - lj

    rng = np.random.default_rng(seed)
    ng = n_samples // 2
    th_g = rng.multivariate_normal(np.asarray(gmean), gcov, ng)
    kf = jax.random.PRNGKey(seed + 7)
    u_f = flow.sample(kf, n_samples - ng)
    th_f = np.asarray(jax.vmap(theta_of_u)(u_f))
    th = np.vstack([th_g, th_f])
    in_box = np.all((th >= lo) & (th <= hi), axis=1)
    th = th[in_box]

    log_qmix = np.logaddexp(np.log(alpha) + log_qg(th),
                            np.log(1.0 - alpha) + log_qf(th))
    lnL = np.asarray(jax.jit(jax.vmap(lnL_fn))(jnp.asarray(th)))
    log_w = np.array(lnL - log_qmix, dtype=np.float64)
    logZ = float(_logsumexp(log_w) - np.log(n_samples))
    log_w -= log_w.max()
    w = np.exp(log_w); w = w / w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    idx = rng.choice(len(th), size=min(8000, len(th)), replace=True, p=w)
    samples = th[idx]
    return {"samples": samples, "ess": ess, "ess_frac": ess / n_samples,
            "train_wall": 0.0, "logZ": logZ, "frac_in_box": float(in_box.mean()),
            "mean": samples.mean(0), "std": samples.std(0)}


def _muframe_proposal(low_level, fit_coords, Xlow_prior, y_prior, box_lo, box_hi):
    """Proposal (mean, cov) for low-level IS built in the Morisaki (fit) frame.

    The posterior is razor-sharp + decorrelated in the mu coords, so its covariance
    there is well-conditioned and near-diagonal (the physical low-level covariance is
    near-singular in the mc direction). We compute the lnL-weighted covariance in fit
    coords and pull it back to low-level via the JAX Jacobian J = d(fit)/d(low) at the
    peak: precision P_low = J^T C_fit^-1 J + diag(1/prior_var). The prior term fills
    the unconstrained direction (the anti-symmetric spin the 5 fit coords drop) at its
    prior width and keeps P_low invertible. No inverse transform, no Rube-Goldberg.
    """
    import jax
    import jax.numpy as jnp
    tf = coordinates.make_transform(low_level, fit_coords)
    Xfit = np.asarray(jax.vmap(tf)(jnp.asarray(Xlow_prior)))
    wp = np.exp(y_prior - y_prior.max()); wp /= wp.sum()
    C_fit = np.atleast_2d(np.cov(Xfit.T, aweights=wp))
    peak_low = Xlow_prior[np.argmax(y_prior)]
    J = np.asarray(jax.jacobian(tf)(jnp.asarray(peak_low)))           # [n_fit, n_low]
    prior_var = ((np.asarray(box_hi) - np.asarray(box_lo)) ** 2) / 12.0
    P_low = J.T @ np.linalg.inv(C_fit) @ J + np.diag(1.0 / prior_var)
    cov_low = np.linalg.inv(P_low)
    gmean = (Xlow_prior * wp[:, None]).sum(0)
    return gmean, 0.5 * (cov_low + cov_low.T)


def sample_gaussian_is(lnL_fn, mean, cov, lo, hi, n_samples=40000, inflate=1.3,
                       seed=0):
    """Importance sampling with a Gaussian proposal matched to the posterior.

    For a SHARP surrogate (quadgp), a normalizing flow struggles to learn the razor-
    thin peak, but a Gaussian proposal matched to the (data lnL-weighted) posterior
    covariance nails it: draw theta ~ N(mean, inflate^2 cov), clip to the prior box,
    and weight by exp(lnL(theta) - log N). The proposal already sits on the peak, so
    ESS is high and the surrogate's non-Gaussian structure is corrected by the weights.
    """
    import jax
    import jax.numpy as jnp
    d = len(mean)
    cov = inflate ** 2 * (np.asarray(cov) + 1e-12 * np.eye(d))
    rng = np.random.default_rng(seed)
    th = rng.multivariate_normal(np.asarray(mean), cov, size=n_samples)
    in_box = np.all((th >= np.asarray(lo)) & (th <= np.asarray(hi)), axis=1)
    th = th[in_box]
    L = np.linalg.cholesky(cov)
    dx = th - np.asarray(mean)
    sol = np.linalg.solve(L, dx.T).T
    log_q = -0.5 * np.sum(sol ** 2, axis=1) - np.sum(np.log(np.diag(L))) \
        - 0.5 * d * np.log(2 * np.pi)
    lnL = np.asarray(jax.jit(jax.vmap(lnL_fn))(jnp.asarray(th)))
    log_w = np.array(lnL - log_q, dtype=np.float64)
    logZ = float(_logsumexp(log_w) - np.log(n_samples))
    log_w -= log_w.max()
    w = np.exp(log_w); w = w / w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    idx = rng.choice(len(th), size=min(8000, len(th)), replace=True, p=w)
    samples = th[idx]
    return {"samples": samples, "ess": ess, "ess_frac": ess / n_samples,
            "train_wall": 0.0, "logZ": logZ, "frac_in_box": float(in_box.mean()),
            "mean": samples.mean(0), "std": samples.std(0)}


def sample_flow_is(lnL_fn, lo, hi, scale_hint=None, init_theta=None,
                   n_samples=8000, n_chains=30, n_local=60, n_global=60,
                   n_train_loops=8, n_prod_loops=2, n_epochs=15, seed=0,
                   rq_n_bins=12, rq_n_layers=6):
    """Flow-based importance sampling on the CLI prior box (sigmoid-into-box).

    A flowMC run (gradient MALA + normalizing-flow training) learns a flow q over an
    unconstrained latent u; a per-coordinate sigmoid maps u into the prior box
    ``theta = lo + (hi-lo)*sigmoid(u)``, so EVERY draw is inside the prior support by
    construction. The CLI ranges (--mc-range / --chi-max / --lambda-* / --eta-range)
    ARE the prior and are trusted exactly -- the grid may extend past them on purpose,
    but we sample only the prior. The flow then absorbs the box geometry (including
    sharp directions like mc within a narrow --mc-range) into its learned shape.

    i.i.d. flow draws + weights exp(lnL + log_jac - log_q) give near-i.i.d. samples and
    the evidence Z. (Efficiency improves with a narrower CLI box and more flow training.)
    """
    import jax
    import jax.numpy as jnp
    from flowMC.Sampler import Sampler
    from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

    lo = jnp.asarray(lo); hi = jnp.asarray(hi); span = hi - lo
    d = int(lo.shape[0])

    def theta_of_u(u):
        return lo + span * jax.nn.sigmoid(u)

    def log_jac(u):
        return jnp.sum(jnp.log(span) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))

    def target(u, data=None):
        return lnL_fn(theta_of_u(u)) + log_jac(u)

    key = jax.random.PRNGKey(seed)
    key, kb, ks, ki, kd = jax.random.split(key, 5)
    bundle = RQSpline_MALA_Bundle(
        rng_key=kb, n_chains=n_chains, n_dims=d, logpdf=target,
        n_local_steps=n_local, n_global_steps=n_global,
        n_training_loops=n_train_loops, n_production_loops=n_prod_loops,
        n_epochs=n_epochs, rq_spline_n_bins=rq_n_bins, rq_spline_n_layers=rq_n_layers)
    sampler = Sampler(d, n_chains, ks, resource_strategy_bundles=bundle)
    if init_theta is not None:
        frac = np.clip((np.asarray(init_theta, float) - np.asarray(lo))
                       / np.asarray(span), 1e-3, 1 - 1e-3)
        u0 = np.log(frac / (1 - frac))
        init = jnp.asarray(u0)[None, :] + 0.3 * jax.random.normal(ki, (n_chains, d))
    else:
        init = 0.3 * jax.random.normal(ki, (n_chains, d))
    t0 = time.time()
    sampler.sample(init, {})
    train_wall = time.time() - t0

    flow = sampler.resources["model"]
    u = flow.sample(kd, n_samples)
    theta = np.asarray(jax.vmap(theta_of_u)(u))
    log_q = np.asarray(jax.vmap(flow.log_prob)(u))
    log_p = np.asarray(jax.vmap(target)(u))
    log_w = np.array(log_p - log_q, dtype=np.float64)
    logZ = float(_logsumexp(log_w) - np.log(n_samples))
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w); w = w / w.sum()
    ess = float(1.0 / np.sum(w ** 2))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_samples, size=n_samples, replace=True, p=w)
    samples = theta[idx]
    return {"samples": samples, "ess": ess, "ess_frac": ess / n_samples,
            "train_wall": train_wall, "logZ": logZ, "frac_in_box": 1.0,
            "mean": samples.mean(0), "std": samples.std(0)}


def _logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def run(net_path, lnL_offset=20.0, cap_points=6000, n_features=512,
        n_opt_steps=300, sigma_cut=0.6, num_warmup=500, num_samples=2000,
        seed=0, physical_sampling=False):
    print("=" * 72)
    print("jax_cip: pure-JAX intrinsic posterior (sample in fit coords)")
    print("=" * 72)
    t0 = time.time()
    model, Xphys, Xfit, y = _fit_surrogate(net_path, sigma_cut, lnL_offset,
                                           cap_points, n_features, n_opt_steps, seed)
    print("fit RFF surrogate on {} pts; fit coords = {} (in {:.1f}s)".format(
        len(y), model.coord_names, time.time() - t0))

    # --- (1) sample in the DECORRELATED FIT coordinates -------------------- #
    # This is the right space: mu1,mu2 remove the curved chirp-mass degeneracy,
    # so NUTS mixes well. (Sampling in raw m1,m2 reintroduces that sharp curved
    # ridge -- see the --physical-sampling contrast below.)
    # Bound the prior to the training support so weakly-constrained directions
    # (mass ratio, tides) stay where the surrogate is valid, not in extrapolation.
    lo, hi = Xfit.min(0), Xfit.max(0)
    pad = 0.02 * (hi - lo)
    bounds = (lo - pad, hi + pad)
    loc = np.asarray(model.x_mean); scale = 3.0 * np.asarray(model.x_std)
    print("\n[fit-coord NUTS] ({} dims, prior bounded to training support) ...".format(
        len(model.coord_names)))
    res = sample_lnL(model.lnL_physical, loc, scale, bounds=bounds,
                     num_warmup=num_warmup, num_samples=num_samples, seed=seed)
    print("  wall-clock {:.1f}s   ESS(min) {:.0f}".format(
        res["wall_clock"], res["ess_min"]))
    print("  posterior (fit coordinates):")
    for i, name in enumerate(model.coord_names):
        print("    {:18s} {:12.5g} +/- {:.3g}".format(
            name, res["mean"][i], res["std"][i]))

    # --- (2) differentiable PHYSICAL lnL -- the population-inference hook --- #
    # Gradient wrt physical params (m1,m2,s1z,s2z,lambda1,lambda2), for AD/numpyro
    # population analyses that need per-event lnL derivatives.
    lnL_phys = coordinates.physical_lnL(model, PHYS)
    theta0 = jnp.asarray(Xphys[np.argmax(y)])
    v, g = float(lnL_phys(theta0)), np.asarray(jax.grad(lnL_phys)(theta0))
    print("\n[physical gradient] lnL at data-MAP = {:.2f}".format(v))
    print("  d(lnL)/d{} =\n   {}".format(PHYS, np.array2string(g, precision=3)))

    out = {"model": model, "fit_result": res, "lnL_phys": lnL_phys}

    # --- (3) optional: the cautionary contrast -- NUTS in physical coords -- #
    if physical_sampling:
        himask = y > (y.max() - lnL_offset)
        ploc = Xphys[himask].mean(0); pscale = 1.0 * Xphys[himask].std(0) + 1e-9
        print("\n[physical-coord NUTS] (cautionary contrast) ...")
        pres = sample_lnL(lnL_phys, ploc, pscale, num_warmup=num_warmup,
                          num_samples=num_samples, seed=seed)
        print("  ESS(min) {:.0f}  -- expect POOR mixing: physical (m1,m2) is the "
              "curved\n  degeneracy the mu coords remove.".format(pres["ess_min"]))
        out["physical_result"] = pres

    return out


def _tree_ring_select(y, n_keep, ring_edges=(2.0, 5.0, 10.0, 20.0, 40.0),
                      min_per_ring=120, seed=0):
    """Stratified downselection by lnL band ("tree rings").

    Random capping concentrates training points near the peak, leaving the GP with
    no anchors on the falloff -- so the fitted peak drifts (e.g. to the mc-range edge)
    and the surrogate is locally wrong where it matters. Instead, partition points into
    rings by how far below the peak they sit (delta = lnLmax - lnL), keep dense coverage
    in the inner rings and a FEW anchor points in each outer (low-lnL) ring. Those
    far-field anchors regularize the falloff and stop the peak from running away.

    Returns indices to keep (<= ~n_keep, plus the per-ring floors).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    delta = y.max() - y                      # >= 0, distance below the peak
    # Cap at the outermost edge -- do NOT include the extreme low-lnL tail: a wide
    # dynamic range makes the RFF surrogate ring/overshoot near the peak. The rings
    # add a few anchors on the *near* falloff to keep the peak from drifting.
    edges = [0.0] + list(ring_edges)
    weights = np.array([0.5 ** i for i in range(len(edges) - 1)])  # inner rings denser
    weights = weights / weights.sum()
    keep = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        idx = np.where((delta >= a) & (delta < b))[0]
        if len(idx) == 0:
            continue
        k = min(len(idx), max(min_per_ring, int(weights[i] * n_keep)))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.concatenate(keep)


def _parse_list(s, default):
    """Parse a '[a,b,c]' string -> tuple of floats, else the default."""
    if not s:
        return tuple(default)
    try:
        import ast
        return tuple(float(x) for x in ast.literal_eval(s))
    except Exception:
        return tuple(default)


def _parse_pair(s):
    """Parse a legacy '[a,b]' range string -> [float, float] (or None)."""
    if not s:
        return None
    try:
        import ast
        v = ast.literal_eval(s)
        return [float(v[0]), float(v[1])]
    except Exception:
        return None


def _coord_box(low_level, opts, Xlow):
    """Per-coord uniform-prior box [lo, hi] honouring the legacy range args
    (--mc-range / --eta-range / --mtot-range / --chi-max / --lambda-min/max),
    falling back to the legacy prior_range_map defaults, then the data range.

    This is the authoritative support: the pipeline passes the correct narrow
    --mc-range for the event, so mc is NOT a wide default and we do not waste the
    sampler outside the measured region.
    """
    import math
    chi_max = float(getattr(opts, "chi_max", 1.0) or 1.0)
    lam_min = float(getattr(opts, "lambda_min", 0.01) or 0.0)
    lam_max = float(getattr(opts, "lambda_max", 4000.0) or 4000.0)
    mc_r = _parse_pair(getattr(opts, "mc_range", None))
    eta_r = _parse_pair(getattr(opts, "eta_range", None))
    mtot_r = _parse_pair(getattr(opts, "mtot_range", None))
    dmc_r = None
    if eta_r:   # eta = 0.25(1 - delta_mc^2)  ->  delta_mc = sqrt(1 - 4 eta)
        d_hi = math.sqrt(max(0.0, 1.0 - 4.0 * eta_r[0]))
        d_lo = math.sqrt(max(0.0, 1.0 - 4.0 * min(eta_r[1], 0.25)))
        dmc_r = [max(0.0, d_lo), min(0.999, d_hi)]
    explicit = {"mc": mc_r, "eta": eta_r, "mtot": mtot_r, "delta_mc": dmc_r}
    defaults = {
        "mc": [0.9, 250.0], "eta": [0.01, 0.2499999], "mtot": [1.0, 300.0],
        "delta_mc": [0.0, 0.9], "q": [0.01, 1.0],
        "s1z": [-0.999 * chi_max, 0.999 * chi_max],
        "s2z": [-0.999 * chi_max, 0.999 * chi_max],
        "lambda1": [lam_min, lam_max], "lambda2": [lam_min, lam_max],
    }
    lo, hi = [], []
    for i, n in enumerate(low_level):
        r = explicit.get(n) or defaults.get(n)
        if r is None:
            r = [float(Xlow[:, i].min()), float(Xlow[:, i].max())]
        lo.append(r[0]); hi.append(r[1])
    return np.array(lo), np.array(hi)


def _physical_to_lowlevel(X6, low_level_names):
    """Input physical columns (m1,m2,s1z,s2z,lambda1,lambda2) -> low-level coords."""
    m1, m2, s1z, s2z, l1, l2 = X6.T
    src = {
        "m1": m1, "m2": m2, "s1z": s1z, "s2z": s2z, "lambda1": l1, "lambda2": l2,
        "mc": (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2,
        "delta_mc": (m1 - m2) / (m1 + m2),
        "eta": m1 * m2 / (m1 + m2) ** 2, "q": m2 / m1,
    }
    return np.column_stack([src[n] for n in low_level_names])


def _write_output(samples, low_level_names, lnL_vals, opts, logZ=None):
    """Write posterior samples in the legacy CIP format: an XML of
    ChooseWaveformParams (physical params) + a ``<fname>_lnL.dat``, plus a minimal
    integral-result file -- so the pipeline's downstream stages find what they
    expect when jax_cip is hot-swapped for the legacy CIP."""
    import lal
    import lalsimulation as lalsim
    import RIFT.lalsimutils as lalsimutils

    P0 = lalsimutils.ChooseWaveformParams()
    try:
        P0.approx = lalsim.GetApproximantFromString(opts.approx_output)
    except Exception:
        pass
    P0.fref, P0.fmin = opts.fref, opts.fmin

    n = min(len(samples), opts.n_output_samples)
    P_list = []
    for s in samples[:n]:
        d = {low_level_names[i]: float(s[i]) for i in range(len(low_level_names))}
        if "m1" in d and "m2" in d:
            m1, m2 = d["m1"], d["m2"]
        else:  # from (mc, delta_mc)
            dmc = d.get("delta_mc", 0.0)
            eta = 0.25 * (1.0 - dmc ** 2)
            mtot = d["mc"] * eta ** (-0.6)
            m1, m2 = 0.5 * mtot * (1.0 + dmc), 0.5 * mtot * (1.0 - dmc)
        P = P0.manual_copy()
        P.m1, P.m2 = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
        P.s1z, P.s2z = d.get("s1z", 0.0), d.get("s2z", 0.0)
        P.lambda1, P.lambda2 = d.get("lambda1", 0.0), d.get("lambda2", 0.0)
        if P.m2 > P.m1:
            P.swap_components()   # enforce m2 <= m1, as the legacy CIP does
        P_list.append(P)

    lalsimutils.ChooseWaveformParams_array_to_xml(
        P_list, fname=opts.fname_output_samples, fref=P0.fref)
    np.savetxt(opts.fname_output_samples + "_lnL.dat", np.asarray(lnL_vals[:n]))
    # evidence: the flow-IS logZ if available, else a max-lnL placeholder
    np.savetxt(opts.fname_output_integral + ".dat",
               [float(logZ) if logZ is not None else float(np.max(lnL_vals))])
    print("  wrote {}.xml.gz  +  {}_lnL.dat  ({} samples)".format(
        opts.fname_output_samples, opts.fname_output_samples, n))


def run_pipeline(opts, ignored):
    """Legacy-CIP-compatible flow: --fname in, posterior-sample XML out.

    Builds fit coordinates from --parameter/--parameter-implied, samples the
    low-level coordinates (--parameter + --parameter-nofit) with NUTS, and writes
    the same output files the pipeline consumes.
    """
    if ignored:
        print("[jax_cip] accepted but unused legacy args: "
              + " ".join(sorted(set(a for a in ignored if a.startswith("--")))))
    fit_coords = list((opts.parameter or []) + (opts.parameter_implied or []))
    low_level = list((opts.parameter or []) + (opts.parameter_nofit or []))
    if not fit_coords:
        fit_coords, low_level = DEFAULT_FIT_COORDS, DEFAULT_LOW_LEVEL
        print("[jax_cip] no --parameter* given; default BNS coords:", fit_coords)
    print("[jax_cip] fit coords = {} ; sampling low-level = {}".format(
        fit_coords, low_level))

    X6, y, yerr, _ = load_ile_net(opts.fname, sigma_cut=opts.sigma_cut,
                                  return_errors=True)
    tf_phys_to_fit = jax.vmap(coordinates.make_transform(PHYS, fit_coords))
    Xfit = np.asarray(tf_phys_to_fit(X6))

    # Tree-ring (lnL-band stratified) downselection: dense near the peak + a few
    # far-field anchors per ring, so the GP sees the falloff and the peak stays put.
    ring_edges = tuple(_parse_list(opts.downselect_rings, (2.0, 5.0, 10.0, 20.0, 40.0)))
    sel = _tree_ring_select(y, opts.cap_points if opts.cap_points > 0 else len(y),
                            ring_edges=ring_edges, seed=opts.seed)
    X6, Xfit, y, yerr = X6[sel], Xfit[sel], y[sel], yerr[sel]
    print("[jax_cip] tree-ring downselect: {} pts across rings (lnL bands {})".format(
        len(y), ring_edges))

    t0 = time.time()
    cls = get_interpolator(opts.jax_fit_method)
    meth = opts.jax_fit_method
    # RFF rings/overshoots on a sharp peak (IS ESS collapse); SVGP/exact are smooth but
    # over-smooth unless constrained; quadgp (quadratic Fisher core + GP residual) is
    # the PE-grade choice for the razor-sharp mc direction.
    if meth in ("rff", "gp-jax-rff"):
        model = cls(n_features=opts.n_features, n_opt_steps=opts.n_opt_steps, seed=opts.seed)
    elif meth in ("svgp", "gp-jax-svgp"):
        model = cls(n_inducing=opts.n_features, n_opt_steps=opts.n_opt_steps, seed=opts.seed)
    elif meth == "quadgp":
        if opts.quadgp_residual == "svgp":
            model = cls(gp_method="svgp", n_opt_steps=opts.n_opt_steps,
                        n_inducing=opts.n_features, seed=opts.seed)
        else:
            model = cls(gp_method="exact", n_opt_steps=opts.n_opt_steps)
    else:  # exact (no seed arg)
        model = cls(n_opt_steps=opts.n_opt_steps)
    model = model.fit(Xfit, y, y_errors=yerr)
    model.coord_names = list(fit_coords)
    fit_wall = time.time() - t0
    print("[jax_cip] {} fit on {} pts in {:.1f}s".format(
        opts.jax_fit_method, len(y), fit_wall))

    Xlow = _physical_to_lowlevel(X6, low_level)
    # Uniform-prior box from the pipeline's range args (authoritative support).
    box_lo, box_hi = _coord_box(low_level, opts, Xlow)
    if "mc" in low_level and _parse_pair(getattr(opts, "mc_range", None)) is None:
        print("[jax_cip] WARNING: no --mc-range given -> wide default mc box; pass "
              "the event's --mc-range so the sampler isn't wasted off-peak.")
    # TRUST the CLI ranges as the prior box (the grid may deliberately extend past
    # them; we sample only the prior). Use the in-prior MAP to seed the flow chains.
    in_prior = np.all((Xlow >= box_lo) & (Xlow <= box_hi), axis=1)
    if int(in_prior.sum()) < 10:
        raise SystemExit("jax_cip: fewer than 10 evaluations inside the prior box; "
                         "check the --mc-range / --chi-max / --lambda-* ranges.")
    init_theta = Xlow[in_prior][np.argmax(y[in_prior])]
    print("[jax_cip] {} / {} evals in prior box; sampling the prior support "
          "{}".format(int(in_prior.sum()), len(y),
                      {n: [round(float(box_lo[i]), 4), round(float(box_hi[i]), 4)]
                       for i, n in enumerate(low_level)}))

    tf_low_to_fit = coordinates.make_transform(low_level, fit_coords)

    def lnL_low(theta):
        return model.lnL_physical(tf_low_to_fit(theta))

    logZ = None
    if opts.sampler in ("gaussian", "mixture", "nuts-mu"):
        # Proposal/preconditioner built in the Morisaki (fit) frame -> well-conditioned cov.
        gmean, gcov = _muframe_proposal(low_level, fit_coords, Xlow[in_prior],
                                        y[in_prior], box_lo, box_hi)
        t1 = time.time()
        if opts.sampler == "nuts-mu":
            # Gradient-based NUTS preconditioned with the mu-frame covariance. Unlike
            # importance sampling it is NOT proposal-limited -> it explores the weakly-
            # constrained directions (delta_mc, tides) the IS proposal under-covered.
            res = sample_nuts_muframe(lnL_low, gmean, gcov, box_lo, box_hi,
                                      num_warmup=opts.num_warmup,
                                      num_samples=opts.num_samples,
                                      num_chains=opts.num_chains, seed=opts.seed)
            sample_wall = time.time() - t1
            total = fit_wall + sample_wall
            print("[jax_cip] NUTS (mu-frame preconditioned): {:.1f}s, ESS(min) {:.0f} "
                  "({:.1%}), {} divergences".format(
                      sample_wall, res["ess"], res["ess_frac"], res["n_divergences"]))
            print("[jax_cip] runtime/effective-sample = {:.3f}s (total {:.0f}s)".format(
                total / max(res["ess"], 1e-9), total))
            for i, name in enumerate(low_level):
                print("    {:9s} {:12.5g} +/- {:.3g}".format(
                    name, res["mean"][i], res["std"][i]))
            samples = res["samples"]
            lnL_at = np.asarray(jax.vmap(lnL_low)(jnp.asarray(samples)))
            _write_output(samples, low_level, lnL_at, opts, logZ=None)
            return {"model": model, "result": res}
        if opts.sampler == "mixture":
            # Defensive: Gaussian core (sharp peak) + flow wings (non-Gaussian tails).
            res = sample_mixture_is(lnL_low, box_lo, box_hi, gmean, gcov,
                                    init_theta=init_theta,
                                    n_samples=max(40000, 8 * opts.num_samples),
                                    n_train_loops=opts.flow_train_loops, seed=opts.seed)
            tag = "mixture-IS (Gauss core + flow wings)"
        else:
            res = sample_gaussian_is(lnL_low, gmean, gcov, box_lo, box_hi,
                                     n_samples=max(40000, 8 * opts.num_samples),
                                     inflate=1.1, seed=opts.seed)
            tag = "gaussian-IS (mu-frame proposal)"
        logZ = res["logZ"]; sample_wall = time.time() - t1
        total = fit_wall + sample_wall
        print("[jax_cip] {}: {:.1f}s, ESS {:.0f} ({:.1%}), {:.0%} in-box, "
              "logZ={:.2f}".format(tag, sample_wall, res["ess"], res["ess_frac"],
                                   res["frac_in_box"], logZ))
        print("[jax_cip] runtime/effective-sample = {:.3f}s (total {:.0f}s)".format(
            total / max(res["ess"], 1e-9), total))
        for i, name in enumerate(low_level):
            print("    {:9s} {:12.5g} +/- {:.3g}".format(
                name, res["mean"][i], res["std"][i]))
        samples = res["samples"]
        lnL_at = np.asarray(jax.vmap(lnL_low)(jnp.asarray(samples)))
        _write_output(samples, low_level, lnL_at, opts, logZ=logZ)
        return {"model": model, "result": res}
    if opts.sampler == "_gaussian_unused":
        # Gaussian-IS: proposal matched to the DATA lnL-weighted posterior covariance.
        # The right sampler for a SHARP surrogate (quadgp) -- a flow can't learn the
        # razor-thin peak, but a peak-matched Gaussian proposal nails it (high ESS).
        Xp, yp = Xlow[in_prior], y[in_prior]
        wp = np.exp(yp - yp.max()); wp /= wp.sum()
        gmean = (Xp * wp[:, None]).sum(0)
        gcov = np.cov(Xp.T, aweights=wp)
        t1 = time.time()
        res = sample_gaussian_is(lnL_low, gmean, gcov, box_lo, box_hi,
                                 n_samples=max(40000, 8 * opts.num_samples),
                                 seed=opts.seed)
        logZ = res["logZ"]; sample_wall = time.time() - t1
        total = fit_wall + sample_wall
        print("[jax_cip] gaussian-IS: {:.1f}s, ESS {:.0f} ({:.1%}), {:.0%} in-box, "
              "logZ={:.2f}".format(sample_wall, res["ess"], res["ess_frac"],
                                   res["frac_in_box"], logZ))
        print("[jax_cip] runtime/effective-sample = {:.3f}s (total {:.0f}s)".format(
            total / max(res["ess"], 1e-9), total))
        for i, name in enumerate(low_level):
            print("    {:9s} {:12.5g} +/- {:.3g}".format(
                name, res["mean"][i], res["std"][i]))
        samples = res["samples"]
        lnL_at = np.asarray(jax.vmap(lnL_low)(jnp.asarray(samples)))
        _write_output(samples, low_level, lnL_at, opts, logZ=logZ)
        return {"model": model, "result": res}
    if opts.sampler == "flow":
        # Train a normalizing flow (flowMC) and use it as the sampling model: i.i.d.
        # draws + importance weights -> efficiency decoupled from MCMC mixing, plus an
        # evidence estimate. Affine map scaled to posterior width; the CLI range box is
        # a hard support clip on the weights (lambda>=0, mc-range, chi-max respected).
        res = sample_flow_is(lnL_low, box_lo, box_hi, init_theta=init_theta,
                             n_samples=max(opts.num_samples, 8000),
                             n_train_loops=opts.flow_train_loops, seed=opts.seed)
        logZ = res["logZ"]
        sample_wall = res["train_wall"]
        total = fit_wall + sample_wall
        print("[jax_cip] flow-IS: train {:.1f}s, ESS {:.0f} ({:.1%} of draws), "
              "{:.0%} in-box, logZ={:.2f}".format(
                  sample_wall, res["ess"], res["ess_frac"],
                  res["frac_in_box"], logZ))
        print("[jax_cip] runtime/effective-sample = {:.3f}s "
              "(total {:.0f}s: fit {:.0f}s + flow {:.0f}s, ESS {:.0f})".format(
                  total / max(res["ess"], 1e-9), total, fit_wall, sample_wall,
                  res["ess"]))
    else:
        res = sample_lnL(lnL_low, Xlow.mean(0), 3.0 * Xlow.std(0),
                         bounds=(box_lo, box_hi), num_warmup=opts.num_warmup,
                         num_samples=opts.num_samples, seed=opts.seed)
        print("[jax_cip] NUTS in low-level coords: {:.1f}s, ESS(min) {:.0f}".format(
            res["wall_clock"], res["ess_min"]))
    for i, name in enumerate(low_level):
        print("    {:9s} {:12.5g} +/- {:.3g}".format(
            name, res["mean"][i], res["std"][i]))

    samples = res["samples"]
    lnL_at = np.asarray(jax.vmap(lnL_low)(jnp.asarray(samples)))
    _write_output(samples, low_level, lnL_at, opts, logZ=logZ)
    return {"model": model, "result": res}


def _build_parser():
    """Parser mirroring the legacy CIP names we act on; everything else is
    swallowed by parse_known_args so the SAME command line hot-swaps."""
    p = argparse.ArgumentParser(description=__doc__)
    # I/O (legacy names)
    p.add_argument("--fname", default=None, help="input ILE .dat/.net file")
    p.add_argument("--net", default=None, help="alias for --fname")
    p.add_argument("--fname-output-samples", default="output-ILE-samples")
    p.add_argument("--fname-output-integral", default="integral_result")
    p.add_argument("--n-output-samples", type=int, default=3000)
    # coordinates (legacy names; append)
    p.add_argument("--parameter", action="append")
    p.add_argument("--parameter-implied", action="append")
    p.add_argument("--parameter-nofit", action="append")
    # cuts (legacy names)
    p.add_argument("--lnL-offset", type=float, default=20.0)
    p.add_argument("--cap-points", type=int, default=8000)
    p.add_argument("--sigma-cut", type=float, default=0.6)
    p.add_argument("--downselect-rings", default=None,
                   help="tree-ring lnL-band edges '[2,5,10,20,40]' (delta below peak) "
                        "for stratified downselection that keeps far-field anchors")
    # prior-support ranges (legacy names) -> the uniform-prior box per coordinate
    p.add_argument("--mc-range", default=None, help="chirp-mass range '[mc1,mc2]'")
    p.add_argument("--eta-range", default=None, help="eta range '[e1,e2]'")
    p.add_argument("--mtot-range", default=None, help="total-mass range '[m1,m2]'")
    p.add_argument("--chi-max", type=float, default=1.0)
    p.add_argument("--lambda-min", type=float, default=0.01)
    p.add_argument("--lambda-max", type=float, default=4000.0)
    # waveform metadata for the output XML (legacy names)
    p.add_argument("--approx-output", default="IMRPhenomD_NRTidalv2")
    p.add_argument("--fref", type=float, default=20.0)
    p.add_argument("--fmin", type=float, default=20.0)
    # JAX-specific knobs (harmless extras the pipeline won't pass)
    p.add_argument("--sampler",
                   choices=["flow", "nuts", "nuts-mu", "gaussian", "mixture"],
                   default="flow",
                   help="flow (default): train a normalizing flow (flowMC) and use "
                        "it as the sampling model with importance weights; nuts-mu: "
                        "NUTS preconditioned with the mu-frame covariance (explores "
                        "weakly-constrained dirs, not proposal-limited); gaussian: "
                        "mu-frame Gaussian importance sampling; nuts: plain numpyro NUTS")
    p.add_argument("--num-chains", type=int, default=2,
                   help="NUTS chains for --sampler nuts-mu (default: 2)")
    p.add_argument("--flow-train-loops", type=int, default=5)
    p.add_argument("--quadgp-residual", choices=["exact", "svgp"], default="exact",
                   help="residual GP backend for quadgp: exact (O(N^3), <~8k) or svgp "
                        "(scalable -> far more data; residual is smooth so no oversmooth)")
    p.add_argument("--jax-fit-method", default="svgp",
                   choices=["rff", "svgp", "exact", "quadgp"],
                   help="surrogate: svgp (default; smooth inducing-point GP, no "
                        "overshoot -> high IS efficiency) | rff (fast, but rings on "
                        "sharp peaks -> IS ESS collapse) | exact (smooth, small N only)")
    p.add_argument("--n-features", type=int, default=512,
                   help="RFF features / SVGP inducing points")
    p.add_argument("--n-opt-steps", type=int, default=300)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo", action="store_true",
                   help="run the console analysis demo instead of writing output")
    p.add_argument("--physical-sampling", action="store_true")
    return p


def main(argv=None):
    opts, ignored = _build_parser().parse_known_args(argv)
    if opts.fname is None and opts.net is not None:
        opts.fname = opts.net
    if opts.demo:
        if not opts.fname:
            raise SystemExit("--fname/--net required for --demo")
        run(opts.fname, opts.lnL_offset, opts.cap_points, opts.n_features,
            opts.n_opt_steps, opts.sigma_cut, opts.num_warmup, opts.num_samples,
            opts.seed, physical_sampling=opts.physical_sampling)
        return
    if not opts.fname:
        raise SystemExit("--fname (or --net) is required")
    run_pipeline(opts, ignored)


if __name__ == "__main__":
    main()
