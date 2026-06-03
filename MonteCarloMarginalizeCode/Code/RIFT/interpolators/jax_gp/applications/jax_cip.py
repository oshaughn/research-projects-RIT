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
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import effective_sample_size

PHYS = ["m1", "m2", "s1z", "s2z", "lambda1", "lambda2"]


def _fit_surrogate(net_path, sigma_cut, lnL_offset, cap_points,
                   n_features, n_opt_steps, seed):
    """Load + good-coord transform + RFF fit. Returns (model, Xphys_kept, y_kept)."""
    X6, y, yerr, _ = load_ile_net(net_path, sigma_cut=sigma_cut, return_errors=True)
    m1, m2, s1z, s2z, l1, l2 = X6.T
    mc, dmc = mc_delta_from_m1m2(m1, m2)
    Xfit = to_fit_coordinates(
        np.column_stack([mc, dmc, s1z, s2z, l1, l2]),
        ["mc", "delta_mc", "s1z", "s2z", "lambda1", "lambda2"], BNS_FIT_COORDS)

    keep = y > (y.max() - lnL_offset)
    X6, Xfit, y, yerr = X6[keep], Xfit[keep], y[keep], yerr[keep]
    rng = np.random.default_rng(seed)
    if 0 < cap_points < len(y):
        sel = rng.choice(len(y), size=cap_points, replace=False)
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

    mcmc = MCMC(NUTS(numpyro_model), num_warmup=num_warmup,
                num_samples=num_samples, progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed))
    samples = np.asarray(mcmc.get_samples()["theta"])
    wall = time.time() - t0
    ess = np.asarray(effective_sample_size(samples[None, ...]))
    return {"samples": samples, "wall_clock": wall, "ess_min": float(np.min(ess)),
            "mean": samples.mean(0), "std": samples.std(0)}


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", required=True, help="RIFT ILE .net file")
    p.add_argument("--lnL-offset", type=float, default=20.0)
    p.add_argument("--cap-points", type=int, default=6000)
    p.add_argument("--n-features", type=int, default=512)
    p.add_argument("--n-opt-steps", type=int, default=300)
    p.add_argument("--sigma-cut", type=float, default=0.6)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--physical-sampling", action="store_true",
                   help="also run the cautionary NUTS-in-physical-coords contrast")
    a = p.parse_args(argv)
    run(a.net, a.lnL_offset, a.cap_points, a.n_features, a.n_opt_steps,
        a.sigma_cut, a.num_warmup, a.num_samples, a.seed,
        physical_sampling=a.physical_sampling)


if __name__ == "__main__":
    main()
