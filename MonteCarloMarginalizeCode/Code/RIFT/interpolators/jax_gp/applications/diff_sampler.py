"""
Differentiable sampling of a fitted jax_gp likelihood surrogate.

The thesis (see ``../DESIGN.md``): once the GP gives us a *differentiable*
``lnL(theta)``, we can sample the (unnormalized) posterior with a
**gradient-based** sampler (HMC / NUTS) instead of brute-force Monte Carlo.
This pays off enormously when the posterior is **sharp** (high SNR), where a
random-walk explorer wastes almost every proposal but a gradient-informed
sampler walks straight to and around the peak.

This module provides

* :func:`sample_nuts` -- treat ``model.lnL_physical(theta)`` as an unnormalized
  log-density, put a broad Normal prior around the fit centre, and run numpyro
  NUTS.  Returns samples, gradient-evaluation count, ESS and wall-clock.
* :func:`sample_rwm` -- a gradient-free random-walk Metropolis baseline that
  spends the *same number of lnL evaluations*, for an apples-to-apples
  efficiency comparison.
* :func:`sample_flowMC` -- best-effort flowMC wrapper (skips gracefully if the
  flowMC API does not cooperate; NUTS is the required path).
* :func:`demo_synthetic` -- fit an RFF surrogate to a known sharp correlated
  Gaussian lnL in d=5, sample it both ways, and report posterior-recovery
  accuracy and ESS-per-lnL-evaluation for NUTS vs the gradient-free baseline.

Everything samples the *fitted surrogate* in *fit coordinates* -- a stand-in for
the eventual CIP integrator swap, not (yet) a real physical-parameter sampler.
See ``SAMPLER_NOTES.md``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

# Importing the package enables jax float64 *before* numpyro draws any arrays.
import RIFT.interpolators.jax_gp as jax_gp
from RIFT.interpolators.jax_gp import export, get_interpolator

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import effective_sample_size


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _prior_loc_scale(model, bounds, prior_scale):
    """Return ``(loc, scale)`` jnp vectors [d] for the broad Normal prior.

    If ``bounds`` (a ``(lo, hi)`` pair of length-d arrays) is given, the prior
    is centred on the box midpoint with a scale spanning it; otherwise it is
    built from the fit's ``x_mean`` / ``x_std`` (the region the GP was trained
    on, which is exactly the relevant region).
    """
    if bounds is not None:
        lo, hi = (jnp.asarray(np.asarray(b, dtype=np.float64)) for b in bounds)
        loc = 0.5 * (lo + hi)
        scale = 0.5 * (hi - lo)
    else:
        loc = jnp.asarray(model.x_mean)
        scale = float(prior_scale) * jnp.asarray(model.x_std)
    return loc, scale


def _make_numpyro_model(model, loc, scale):
    """Build a numpyro model: broad Normal prior + ``factor`` of the GP lnL.

    A broad *Normal* prior (rather than a hard Uniform) keeps NUTS in an
    unconstrained space and well-behaved while still localizing the relevant
    region; the GP ``factor`` then sculpts the actual posterior.
    """
    lnL = model.lnL_physical

    def numpyro_model():
        theta = numpyro.sample("theta", dist.Normal(loc, scale).to_event(1))
        numpyro.factor("lnL", lnL(theta))

    return numpyro_model


# --------------------------------------------------------------------------- #
# 1. NUTS                                                                      #
# --------------------------------------------------------------------------- #
def sample_nuts(model, bounds=None, num_warmup=500, num_samples=2000,
                prior_scale=3.0, seed=0):
    """Sample ``model.lnL_physical`` as a log-density with numpyro NUTS.

    Parameters
    ----------
    model : interpolator
        A fitted (or exported/loaded) jax_gp model exposing ``lnL_physical``,
        ``x_mean`` and ``x_std``.
    bounds : tuple(array, array), optional
        ``(lo, hi)`` length-d arrays.  If given, the broad prior spans this box;
        otherwise it is ``Normal(x_mean, prior_scale * x_std)``.
    num_warmup, num_samples : int
        NUTS warmup / sampling iterations.
    prior_scale : float
        Prior width in units of ``x_std`` (only used when ``bounds`` is None).
    seed : int
        PRNG seed.

    Returns
    -------
    dict
        ``samples`` [num_samples, d], ``n_grad_evals`` (leapfrog steps, the
        gradient-evaluation count), ``ess`` [d], ``ess_min`` (float),
        ``wall_clock`` (s), ``mean`` [d], ``cov`` [d, d].
    """
    loc, scale = _prior_loc_scale(model, bounds, prior_scale)
    numpyro_model = _make_numpyro_model(model, loc, scale)

    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    rng = jax.random.PRNGKey(seed)

    t0 = time.time()
    # ``num_steps`` per iteration = leapfrog steps = gradient evaluations.
    mcmc.run(rng, extra_fields=("num_steps",))
    samples = np.asarray(mcmc.get_samples()["theta"])
    wall = time.time() - t0

    # Gradient-eval count = total leapfrog steps (each step = one grad of lnL).
    extra = mcmc.get_extra_fields(group_by_chain=False)
    if "num_steps" in extra:
        n_grad = int(np.sum(np.asarray(extra["num_steps"])))
    else:
        n_grad = num_samples  # conservative placeholder if field unavailable
    ess = np.asarray(effective_sample_size(samples[None, ...]))

    return {
        "samples": samples,
        "n_grad_evals": n_grad,
        "ess": ess,
        "ess_min": float(np.min(ess)),
        "wall_clock": wall,
        "mean": samples.mean(axis=0),
        "cov": np.cov(samples, rowvar=False),
        "n_lnL_evals": n_grad,   # NUTS: one lnL+grad per leapfrog step
    }


# --------------------------------------------------------------------------- #
# 2. gradient-free baseline: random-walk Metropolis                           #
# --------------------------------------------------------------------------- #
def sample_rwm(model, bounds=None, n_evals=None, prior_scale=3.0,
               step_scale=0.5, seed=0):
    """Gradient-free random-walk Metropolis on the same log-density.

    Uses the *same* ``lnL_physical + broad-Normal-prior`` target as
    :func:`sample_nuts` and is budgeted to spend ``n_evals`` likelihood
    evaluations (one per proposal), so ESS/eval is directly comparable.

    Parameters
    ----------
    n_evals : int
        Number of lnL evaluations (= proposals) to spend.
    step_scale : float
        Proposal std in units of the prior ``scale``.  Tuned loosely toward a
        reasonable acceptance rate for the sharp-posterior regime.

    Returns
    -------
    dict
        ``samples``, ``n_lnL_evals``, ``ess``, ``ess_min``, ``wall_clock``,
        ``mean``, ``cov``, ``accept_rate``.
    """
    loc, scale = _prior_loc_scale(model, bounds, prior_scale)
    loc = np.asarray(loc)
    scale = np.asarray(scale)
    d = loc.shape[0]
    if n_evals is None:
        n_evals = 4000

    lnL_fn = jax.jit(model.lnL_physical)

    def log_target(theta):
        # broad Normal prior (matching the numpyro model) + GP factor
        lp = -0.5 * np.sum(((theta - loc) / scale) ** 2)
        return float(lnL_fn(jnp.asarray(theta))) + lp

    rng = np.random.default_rng(seed)
    cur = loc.copy()
    cur_lp = log_target(cur)
    prop_std = step_scale * scale

    samples = np.empty((n_evals, d))
    n_accept = 0
    t0 = time.time()
    for i in range(n_evals):
        prop = cur + prop_std * rng.standard_normal(d)
        prop_lp = log_target(prop)
        if np.log(rng.random()) < (prop_lp - cur_lp):
            cur, cur_lp = prop, prop_lp
            n_accept += 1
        samples[i] = cur
    wall = time.time() - t0

    # discard first 25% as burn-in for ESS / posterior estimates
    burn = n_evals // 4
    post = samples[burn:]
    ess = np.asarray(effective_sample_size(post[None, ...]))
    return {
        "samples": post,
        "n_lnL_evals": n_evals,
        "ess": ess,
        "ess_min": float(np.min(ess)),
        "wall_clock": wall,
        "mean": post.mean(axis=0),
        "cov": np.cov(post, rowvar=False),
        "accept_rate": n_accept / n_evals,
    }


# --------------------------------------------------------------------------- #
# 3. flowMC (best-effort)                                                      #
# --------------------------------------------------------------------------- #
def sample_flowMC(model, bounds=None, num_samples=2000, prior_scale=3.0,
                  seed=0):
    """Best-effort flowMC sampler using the same gradient.

    flowMC's public API has shifted across releases; this wrapper attempts a
    minimal MALA + normalizing-flow run and **skips gracefully** (returning
    ``None`` with a logged note) if anything in its constructor signature does
    not line up.  NUTS is the supported path; this is a bonus.
    """
    try:
        loc, scale = _prior_loc_scale(model, bounds, prior_scale)
        loc_j = jnp.asarray(loc)
        scale_j = jnp.asarray(scale)
        d = int(loc_j.shape[0])

        def log_target(theta, data=None):
            lp = -0.5 * jnp.sum(((theta - loc_j) / scale_j) ** 2)
            return model.lnL_physical(theta) + lp

        from flowMC.Sampler import Sampler
        from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

        rng = jax.random.PRNGKey(seed)
        n_chains = 20
        rng, sub = jax.random.split(rng)
        init = loc_j + scale_j * jax.random.normal(sub, (n_chains, d))

        # flowMC >= 0.4 bundle-style API.
        rng, kb = jax.random.split(rng)
        bundle = RQSpline_MALA_Bundle(
            rng_key=kb,
            n_chains=n_chains,
            n_dims=d,
            logpdf=log_target,
            n_local_steps=50,
            n_global_steps=50,
            n_training_loops=3,
            n_production_loops=3,
            n_epochs=5,
        )
        sampler = Sampler(d, n_chains, rng,
                          resource_strategy_bundles=bundle)
        t0 = time.time()
        sampler.sample(init, {})
        wall = time.time() - t0
        prod = sampler.resources["positions_production"]
        chains = np.asarray(getattr(prod, "data", prod))
        samples = chains.reshape(-1, d)
        return {
            "samples": samples,
            "wall_clock": wall,
            "mean": samples.mean(axis=0),
            "cov": np.cov(samples, rowvar=False),
        }
    except Exception as exc:  # noqa: BLE001 -- best-effort by design
        print("[sample_flowMC] skipped (best-effort): {}: {}".format(
            type(exc).__name__, exc))
        return None


# --------------------------------------------------------------------------- #
# synthetic ground truth                                                       #
# --------------------------------------------------------------------------- #
def _make_sharp_gaussian(d=5, seed=0):
    """Construct a known sharp correlated-Gaussian lnL in ``d`` dims.

    Returns ``(lnL_fn, mu, Sigma)`` where ``lnL_fn(x)`` = log N(x; mu, Sigma) up
    to a constant, ``mu`` is the true mean and ``Sigma`` the true covariance.
    Because the posterior we sample is ``lnL + broad-Normal-prior``, the
    analytic posterior is a Gaussian we can compare against (the prior is so
    broad relative to ``Sigma`` that the posterior ~ the lnL Gaussian; we still
    compute the exact prior-corrected analytic posterior below).
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-2.0, 2.0, size=d)
    # sharp + correlated: small eigenvalues, random rotation
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    # lengthscales ~0.05..0.2 in each rotated direction => sharp peak
    evals = rng.uniform(0.05, 0.2, size=d) ** 2
    Sigma = (Q * evals) @ Q.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    Prec = np.linalg.inv(Sigma)

    def lnL_fn(X):
        X = np.atleast_2d(X)
        delta = X - mu
        quad = np.einsum("ni,ij,nj->n", delta, Prec, delta)
        return -0.5 * quad

    return lnL_fn, mu, Sigma


def _analytic_posterior(mu_L, Sigma_L, prior_loc, prior_scale_vec):
    """Exact Gaussian posterior of ``N(mu_L, Sigma_L) * N(prior_loc, diag)``."""
    Prec_L = np.linalg.inv(Sigma_L)
    Prec_p = np.diag(1.0 / prior_scale_vec ** 2)
    Prec_post = Prec_L + Prec_p
    Sigma_post = np.linalg.inv(Prec_post)
    mu_post = Sigma_post @ (Prec_L @ mu_L + Prec_p @ prior_loc)
    return mu_post, Sigma_post


# --------------------------------------------------------------------------- #
# demo                                                                         #
# --------------------------------------------------------------------------- #
def demo_synthetic(d=5, n_train=3000, n_features=512, n_opt_steps=300,
                   num_warmup=500, num_samples=2000, seed=0):
    """Fit an RFF surrogate to a sharp Gaussian lnL and sample it two ways.

    Demonstrates that gradient-based NUTS recovers the known posterior and does
    so at far higher ESS-per-lnL-evaluation than a gradient-free random-walk
    baseline given the *same* evaluation budget.
    """
    print("=" * 72)
    print("SYNTHETIC DEMO: sharp correlated Gaussian lnL, d = {}".format(d))
    print("=" * 72)

    # --- ground truth + training data --------------------------------------
    lnL_true, mu_true, Sigma_true = _make_sharp_gaussian(d=d, seed=seed)
    rng = np.random.default_rng(seed + 1)
    # Sample training points around the peak (covering the sharp region well)
    # plus a broader cloud so the RFF sees the falloff.
    L = np.linalg.cholesky(Sigma_true)
    n_near = int(0.7 * n_train)
    n_far = n_train - n_near
    X_near = mu_true + (rng.standard_normal((n_near, d)) * 2.0) @ L.T
    X_far = mu_true + rng.standard_normal((n_far, d)) * 1.0
    X = np.vstack([X_near, X_far])
    y = lnL_true(X)
    # small per-point "MC error" so the heteroscedastic path is exercised
    yerr = np.full_like(y, 0.05)
    y = y + rng.normal(0.0, 0.05, size=y.shape)

    print("fitting RFF surrogate ({} pts, {} features, {} steps) ...".format(
        n_train, n_features, n_opt_steps))
    t0 = time.time()
    model = get_interpolator("rff")(
        n_features=n_features, n_opt_steps=n_opt_steps, seed=seed
    ).fit(X, y, y_errors=yerr)
    print("  fit done in {:.1f}s".format(time.time() - t0))

    # surrogate accuracy at the truth peak / held-out check
    Xte = mu_true + (rng.standard_normal((500, d)) * 2.0) @ L.T
    pred = model.predict(Xte)
    rmse = float(np.sqrt(np.mean((pred - lnL_true(Xte)) ** 2)))
    print("  surrogate held-out RMSE (lnL units): {:.3f}".format(rmse))

    # --- NUTS ---------------------------------------------------------------
    print("\n[NUTS] gradient-based sampling ...")
    nuts = sample_nuts(model, num_warmup=num_warmup, num_samples=num_samples,
                       prior_scale=3.0, seed=seed)
    print("  wall-clock     : {:.2f}s".format(nuts["wall_clock"]))
    print("  lnL/grad evals : {}".format(nuts["n_grad_evals"]))
    print("  ESS (min/dim)  : {:.0f}".format(nuts["ess_min"]))
    print("  ESS per eval   : {:.4f}".format(
        nuts["ess_min"] / max(nuts["n_grad_evals"], 1)))

    # --- gradient-free baseline (same eval budget) --------------------------
    budget = nuts["n_grad_evals"]
    print("\n[RWM] gradient-free baseline, same eval budget = {} ...".format(budget))
    rwm = sample_rwm(model, n_evals=budget, prior_scale=3.0,
                     step_scale=0.4, seed=seed)
    print("  wall-clock     : {:.2f}s".format(rwm["wall_clock"]))
    print("  lnL evals      : {}".format(rwm["n_lnL_evals"]))
    print("  accept rate    : {:.3f}".format(rwm["accept_rate"]))
    print("  ESS (min/dim)  : {:.0f}".format(rwm["ess_min"]))
    print("  ESS per eval   : {:.4f}".format(
        rwm["ess_min"] / max(rwm["n_lnL_evals"], 1)))

    # --- analytic posterior to compare against ------------------------------
    prior_loc = np.asarray(model.x_mean)
    prior_scale_vec = 3.0 * np.asarray(model.x_std)
    mu_post, Sigma_post = _analytic_posterior(
        mu_true, Sigma_true, prior_loc, prior_scale_vec)

    def _report_recovery(tag, res):
        dmu = res["mean"] - mu_post
        # whiten the mean error by the posterior std for a scale-free number
        sd_post = np.sqrt(np.diag(Sigma_post))
        z = np.abs(dmu) / sd_post
        # cov fractional error (Frobenius)
        cov_err = (np.linalg.norm(res["cov"] - Sigma_post)
                   / np.linalg.norm(Sigma_post))
        print("  [{}] mean |z| max={:.2f} mean={:.2f}  | cov rel-err={:.2f}".format(
            tag, float(z.max()), float(z.mean()), float(cov_err)))
        return float(z.max()), float(cov_err)

    print("\n--- posterior recovery vs analytic Gaussian posterior ---")
    nuts_z, nuts_cov = _report_recovery("NUTS", nuts)
    rwm_z, rwm_cov = _report_recovery("RWM ", rwm)

    # --- headline efficiency ratio -----------------------------------------
    nuts_eff = nuts["ess_min"] / max(nuts["n_grad_evals"], 1)
    rwm_eff = rwm["ess_min"] / max(rwm["n_lnL_evals"], 1)
    ratio = nuts_eff / max(rwm_eff, 1e-12)
    print("\n" + "=" * 72)
    print("HEADLINE: NUTS ESS/eval = {:.4f}, RWM ESS/eval = {:.4f}".format(
        nuts_eff, rwm_eff))
    print("          NUTS is {:.1f}x more efficient per lnL-eval (sharp posterior)".format(
        ratio))
    print("=" * 72)

    # --- optional flowMC ----------------------------------------------------
    fmc = sample_flowMC(model, num_samples=num_samples, seed=seed)
    if fmc is not None:
        _report_recovery("flowMC", fmc)

    return {
        "model": model,
        "nuts": nuts,
        "rwm": rwm,
        "flowMC": fmc,
        "mu_post": mu_post,
        "Sigma_post": Sigma_post,
        "rmse": rmse,
        "efficiency_ratio": ratio,
    }


def demo_artifact(base, num_warmup=500, num_samples=2000, seed=0):
    """Load an exported real lnL artifact and sample it with NUTS."""
    if not export.exists(base):
        print("[demo_artifact] no artifact at {!r}; skipping.".format(base))
        return None
    print("loading exported artifact: {}".format(base))
    model = export.load(base)
    print("  coord_names: {}".format(getattr(model, "coord_names", None)))
    print("  d = {}".format(np.asarray(model.x_mean).shape[0]))
    nuts = sample_nuts(model, num_warmup=num_warmup, num_samples=num_samples,
                       prior_scale=3.0, seed=seed)
    print("  NUTS: {:.2f}s, {} grad-evals, ESS(min)={:.0f}".format(
        nuts["wall_clock"], nuts["n_grad_evals"], nuts["ess_min"]))
    print("  posterior mean ({}):".format(getattr(model, "coord_names", "dims")))
    print("   ", np.array2string(nuts["mean"], precision=4))
    return {"model": model, "nuts": nuts}


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", choices=["synthetic"], default="synthetic",
                   help="which built-in demo to run (default: synthetic)")
    p.add_argument("--artifact", default=None,
                   help="base path of an exported lnL bundle to sample "
                        "(e.g. /tmp/gw170817_rff); skipped if absent")
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.demo == "synthetic":
        demo_synthetic(d=args.dim, n_train=args.n_train,
                       num_warmup=args.num_warmup,
                       num_samples=args.num_samples, seed=args.seed)

    if args.artifact is not None:
        demo_artifact(args.artifact, num_warmup=args.num_warmup,
                      num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()
