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
from numpyro.infer import MCMC, NUTS
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

    keep = y > (y.max() - opts.lnL_offset)
    X6, Xfit, y, yerr = X6[keep], Xfit[keep], y[keep], yerr[keep]
    rng = np.random.default_rng(opts.seed)
    if 0 < opts.cap_points < len(y):
        sel = rng.choice(len(y), size=opts.cap_points, replace=False)
        X6, Xfit, y, yerr = X6[sel], Xfit[sel], y[sel], yerr[sel]

    t0 = time.time()
    model = get_interpolator("rff")(n_features=opts.n_features,
                                    n_opt_steps=opts.n_opt_steps,
                                    seed=opts.seed).fit(Xfit, y, y_errors=yerr)
    model.coord_names = list(fit_coords)
    fit_wall = time.time() - t0
    print("[jax_cip] RFF fit on {} pts in {:.1f}s".format(len(y), fit_wall))

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
    p.add_argument("--sampler", choices=["flow", "nuts"], default="flow",
                   help="flow (default): train a normalizing flow (flowMC) and use "
                        "it as the sampling model with importance weights; nuts: "
                        "plain numpyro NUTS")
    p.add_argument("--flow-train-loops", type=int, default=5)
    p.add_argument("--n-features", type=int, default=512)
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
