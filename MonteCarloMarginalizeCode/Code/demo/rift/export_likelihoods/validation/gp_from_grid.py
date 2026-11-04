"""
Generic GP-surrogate posterior from a RIFT lnL grid (validation harness).

Case-driven, NOT BNS-specialized: given a consolidated lnL grid (the named-column
``.dat`` a RIFT pipeline run produces, e.g. ``all_dgrid.dat`` with columns
``lnL sigmaL m1 m2 dist ...``) and a list of parameters, this fits the ``quadgp``
surrogate (quadratic Fisher core + GP residual) and draws a posterior with
mu-frame-preconditioned NUTS (``sample_nuts_muframe``).  The output is a
named-column ``.dat`` posterior over the chosen parameters, directly comparable to
the standard RIFT posterior via ``compare_marginals.py``.

This is deliberately the *generic* path used for the systematic head-to-head ladder
(mc,q,dL -> +aligned spin -> ...).  We fit in per-dimension whitened coordinates and
let the quadratic core absorb the Fisher correlations; the NUTS mass matrix is seeded
from the lnL-weighted data covariance of the in-prior grid points.  (The bespoke
Morisaki/mu-frame coordinate machinery in ``applications/jax_cip.py`` is reserved for
the razor-sharp BNS mass+spin+tides case, where the geometry is genuinely
ill-conditioned; the easy rungs of the ladder do not need it.)

Usage::

    python gp_from_grid.py --grid all_dgrid.dat \\
        --param m1 --param m2 --param dist \\
        --range 'm1:[24,34]' --range 'm2:[24,34]' --range 'dist:[10,500]' \\
        --lnL-offset 30 --out gp_posterior.dat
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _load_grid(path, params, lnL_col="lnL", err_col="sigmaL"):
    """Read a named-column .dat grid -> (X[n,d], y[n], yerr[n] or None)."""
    data = np.genfromtxt(path, names=True, comments="#")
    names = data.dtype.names
    for p in list(params) + [lnL_col]:
        if p not in names:
            raise SystemExit(
                "column {!r} not in grid (have: {}). For all_dgrid.dat the header is "
                "'# lnL sigmaL m1 m2 ...'".format(p, ", ".join(names)))
    X = np.column_stack([np.asarray(data[p], float) for p in params])
    y = np.asarray(data[lnL_col], float)
    yerr = np.asarray(data[err_col], float) if err_col in names else None
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if yerr is not None:
        ok &= np.isfinite(yerr)
    return X[ok], y[ok], (yerr[ok] if yerr is not None else None)


def _parse_range(s):
    name, rng = s.split(":", 1)
    lo, hi = json.loads(rng)
    return name.strip(), float(lo), float(hi)


def run(grid, params, ranges, out, lnL_offset=30.0, fit_method="quadgp",
        quadgp_residual="svgp", n_features=400, n_opt_steps=200,
        num_warmup=800, num_samples=4000, num_chains=2, inflate=1.3, seed=0,
        auto_range=False):
    import jax
    import jax.numpy as jnp
    from RIFT.interpolators.jax_gp import get_interpolator
    from RIFT.interpolators.jax_gp.applications.jax_cip import sample_nuts_muframe

    X, y, yerr = _load_grid(grid, params)

    # Keep the informative high-lnL region (the surrogate should not waste capacity
    # on the deep tail) -- mirrors CIP's --lnL-offset.
    keep = y > y.max() - lnL_offset
    X, y = X[keep], y[keep]
    yerr = yerr[keep] if yerr is not None else None

    # Prior box: explicit --range where given; otherwise (--auto-range) the extent
    # of the run's grid in the high-lnL region.  Deriving the box from the grid keeps
    # the harness run-agnostic, but note it is the GRID's coverage, NOT necessarily
    # the run's prior -- pass explicit --range to match the run's prior exactly.
    box_lo = np.empty(len(params)); box_hi = np.empty(len(params))
    for i, p in enumerate(params):
        if p in ranges:
            box_lo[i], box_hi[i] = ranges[p]
        elif auto_range:
            box_lo[i], box_hi[i] = float(X[:, i].min()), float(X[:, i].max())
        else:
            raise SystemExit("no --range for {!r} and --auto-range not set".format(p))
    print("[gp_from_grid] {} grid pts after lnL-offset {} cut; params {}".format(
        len(y), lnL_offset, list(params)))

    # Fit the surrogate (whitened internally by BaseInterpolator; lnL_physical takes
    # raw physical params -> pure-JAX + differentiable, exactly the export contract).
    cls = get_interpolator(fit_method)
    if fit_method == "quadgp":
        kw = dict(gp_method=quadgp_residual, n_opt_steps=n_opt_steps)
        if quadgp_residual == "svgp":
            kw["n_inducing"] = n_features
        model = cls(**kw)
    elif fit_method in ("svgp",):
        model = cls(n_inducing=n_features, n_opt_steps=n_opt_steps)
    elif fit_method in ("rff",):
        model = cls(n_features=n_features, n_opt_steps=n_opt_steps)
    else:
        model = cls(n_opt_steps=n_opt_steps)
    model = model.fit(X, y, y_errors=yerr)
    model.coord_names = list(params)

    # NUTS preconditioner: lnL-weighted mean/cov of the in-prior grid points (the
    # well-conditioned, low-dim analogue of the BNS mu-frame pull-back).
    in_prior = np.all((X >= box_lo) & (X <= box_hi), axis=1)
    if int(in_prior.sum()) < 10:
        raise SystemExit("fewer than 10 grid points inside the prior box; check --range")
    Xp, yp = X[in_prior], y[in_prior]
    wp = np.exp(yp - yp.max()); wp /= wp.sum()
    gmean = (Xp * wp[:, None]).sum(0)
    gcov = np.atleast_2d(np.cov(Xp.T, aweights=wp))
    gcov = inflate ** 2 * (gcov + 1e-12 * np.eye(len(params)))

    def lnL_fn(theta):
        return model.lnL_physical(theta)

    res = sample_nuts_muframe(lnL_fn, gmean, gcov, box_lo, box_hi,
                              num_warmup=num_warmup, num_samples=num_samples,
                              num_chains=num_chains, seed=seed)
    print("[gp_from_grid] NUTS: ESS(min) {:.0f} ({:.1%}), {} divergences".format(
        res["ess"], res["ess_frac"], res["n_divergences"]))
    for i, p in enumerate(params):
        print("    {:6s} {:12.5g} +/- {:.3g}".format(p, res["mean"][i], res["std"][i]))

    samples = res["samples"]
    lnL_at = np.asarray(jax.jit(jax.vmap(lnL_fn))(jnp.asarray(samples)))
    header = " ".join(list(params) + ["lnL"])
    np.savetxt(out, np.column_stack([samples, lnL_at]), header=header, comments="")
    print("[gp_from_grid] wrote {} ({} samples)".format(out, len(samples)))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", required=True, help="consolidated lnL grid (.dat, named columns)")
    p.add_argument("--param", action="append", required=True, dest="params",
                   help="parameter column to fit/sample (repeatable)")
    p.add_argument("--range", action="append", default=[], dest="ranges",
                   help="prior range 'name:[lo,hi]' (repeatable)")
    p.add_argument("--auto-range", action="store_true",
                   help="derive any unspecified prior range from the grid extent "
                        "(run-agnostic; prefer explicit --range to match the run prior)")
    p.add_argument("--lnL-offset", type=float, default=30.0)
    p.add_argument("--fit-method", default="quadgp", choices=["quadgp", "svgp", "rff", "exact"])
    p.add_argument("--quadgp-residual", default="svgp", choices=["svgp", "exact", "rff"])
    p.add_argument("--n-features", type=int, default=400)
    p.add_argument("--n-opt-steps", type=int, default=200)
    p.add_argument("--num-warmup", type=int, default=800)
    p.add_argument("--num-samples", type=int, default=4000)
    p.add_argument("--num-chains", type=int, default=2)
    p.add_argument("--inflate", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output posterior .dat (named columns)")
    a = p.parse_args(argv)

    ranges = {}
    for s in a.ranges:
        name, lo, hi = _parse_range(s)
        ranges[name] = (lo, hi)
    if not a.auto_range:
        for p_ in a.params:
            if p_ not in ranges:
                raise SystemExit(
                    "missing --range for {!r} (or pass --auto-range)".format(p_))

    run(a.grid, a.params, ranges, a.out, lnL_offset=a.lnL_offset,
        fit_method=a.fit_method, quadgp_residual=a.quadgp_residual,
        n_features=a.n_features, n_opt_steps=a.n_opt_steps,
        num_warmup=a.num_warmup, num_samples=a.num_samples,
        num_chains=a.num_chains, inflate=a.inflate, seed=a.seed,
        auto_range=a.auto_range)


if __name__ == "__main__":
    main()
