"""
Package a RIFT ILE ``.net`` file into a self-contained, differentiable lnL artifact.

This is the "ship it" end of the jax_gp pipeline: it takes raw ILE output (the
per-point Monte-Carlo lnL evaluations CIP normally consumes) and produces a small,
portable bundle (``<base>.npz`` + ``<base>.meta.json``) that reconstructs a pure-JAX,
``jax.grad``-able ``lnL(theta)`` -- with no RIFT/lalsimutils dependency at load time.

The surrogate is fit in *fit coordinates* (the decorrelated space CIP itself fits in),
so the exported lnL is differentiable in those coordinates -- recorded as
``coord_names`` in the meta -- not in the raw physical parameters.  See ``ARTIFACT.md``.

Example
-------
::

    python export_artifact.py --net /home/oshaughn/all.net \\
        --out /tmp/gw170817_rff --coords bns
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from RIFT.interpolators.jax_gp import get_interpolator, export
from RIFT.interpolators.jax_gp.benchmark.datasets import (
    BNS_FIT_COORDS,
    load_ile_net,
    mc_delta_from_m1m2,
    to_fit_coordinates,
)

#: raw 6-parameter intrinsic columns as loaded from an ILE ``.net`` file
RAW_COORDS = ("m1", "m2", "s1z", "s2z", "lambda1", "lambda2")


def _build_fit_coordinates(X6, coords):
    """Map the raw 6-column ILE intrinsic block ``X6`` to fit coordinates.

    Parameters
    ----------
    X6 : ndarray [n, 6]
        Columns ``(m1, m2, s1z, s2z, lambda1, lambda2)`` as returned by
        :func:`load_ile_net`.
    coords : {"bns", "raw"}
        ``"bns"`` applies the decorrelated BNS transform
        (:data:`~RIFT.interpolators.jax_gp.benchmark.datasets.BNS_FIT_COORDS`);
        ``"raw"`` keeps the 6 raw physical parameters.

    Returns
    -------
    X_fit : ndarray [n, d]
    coord_names : list[str]
    """
    if coords == "raw":
        return np.asarray(X6, dtype=np.float64), list(RAW_COORDS)
    if coords == "bns":
        m1, m2, s1z, s2z, l1, l2 = X6.T
        mc, dmc = mc_delta_from_m1m2(m1, m2)
        X_low = np.column_stack([mc, dmc, s1z, s2z, l1, l2])
        low_names = ["mc", "delta_mc", "s1z", "s2z", "lambda1", "lambda2"]
        X_fit = to_fit_coordinates(X_low, low_names, BNS_FIT_COORDS)
        return np.asarray(X_fit, dtype=np.float64), list(BNS_FIT_COORDS)
    raise ValueError("coords must be 'bns' or 'raw', got {!r}".format(coords))


def build_artifact(net_path, out_base, coords="bns", method="rff",
                   sigma_cut=0.6, lnL_offset=40.0, cap_points=8000,
                   n_features=512, n_opt_steps=300, seed=0):
    """Build and persist a differentiable lnL artifact from an ILE ``.net`` file.

    The pipeline mirrors CIP's fit preparation: load + de-dupe + ``sigma_cut`` the
    ILE points, transform to fit coordinates, keep only the high-lnL region
    (``lnL > max - lnL_offset``), optionally subsample to ``cap_points``, then fit
    the chosen interpolator *using the per-point Monte-Carlo errors*.  The model is
    exported via :func:`RIFT.interpolators.jax_gp.export.save`, reloaded, and checked
    for predict agreement and finite ``jax.grad``.

    Parameters
    ----------
    net_path : str
        Path to the RIFT ILE ``.net`` file.
    out_base : str
        Output base path; writes ``<out_base>.npz`` and ``<out_base>.meta.json``.
    coords : {"bns", "raw"}, optional
        Fit-coordinate system (see :func:`_build_fit_coordinates`).
    method : str, optional
        jax_gp interpolator name (``"rff"``, ``"exact"``, ``"svgp"``).
    sigma_cut : float, optional
        Drop ILE points whose reported ``sigma_lnL`` exceeds this (CIP default 0.6).
    lnL_offset : float, optional
        Keep only points with ``lnL > max(lnL) - lnL_offset``.
    cap_points : int or None, optional
        If set and there are more surviving points, random-subsample down to this
        many (like CIP ``--cap-points``).
    n_features, n_opt_steps, seed : optional
        Passed to the interpolator constructor (where applicable).

    Returns
    -------
    dict
        Metadata: ``n_train``, ``coord_names``, ``lnL_max``, ``holdout_rmse``,
        plus the resolved build settings.
    """
    rng = np.random.default_rng(seed)

    # 1. load ILE points (with per-point MC errors, sigma-cut + dedupe)
    X6, y, yerr, _ = load_ile_net(
        net_path, sigma_cut=sigma_cut, return_errors=True)

    # 2. transform to fit coordinates
    X_fit, coord_names = _build_fit_coordinates(X6, coords)

    # drop any rows the coordinate transform made non-finite
    ok = np.all(np.isfinite(X_fit), axis=1) & np.isfinite(y) & np.isfinite(yerr)
    X_fit, y, yerr = X_fit[ok], y[ok], yerr[ok]

    # 3. lnL peak cut: keep the informative high-likelihood region
    lnL_max = float(np.max(y))
    keep = y > lnL_max - lnL_offset
    X_fit, y, yerr = X_fit[keep], y[keep], yerr[keep]

    # 4. optional random subsample to bound fit cost
    if cap_points is not None and len(y) > cap_points:
        sel = rng.choice(len(y), size=cap_points, replace=False)
        X_fit, y, yerr = X_fit[sel], y[sel], yerr[sel]

    # 5. 15% holdout for an honest generalization estimate
    n = len(y)
    perm = rng.permutation(n)
    n_hold = max(1, int(round(0.15 * n)))
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    Xtr, ytr, etr = X_fit[train_idx], y[train_idx], yerr[train_idx]
    Xho, yho = X_fit[hold_idx], y[hold_idx]

    # 6. fit the chosen interpolator WITH the per-point MC errors
    cls = get_interpolator(method)
    kwargs = {}
    for k, v in (("n_features", n_features), ("n_opt_steps", n_opt_steps),
                 ("seed", seed)):
        if k in cls.__init__.__code__.co_varnames:
            kwargs[k] = v
    model = cls(**kwargs).fit(Xtr, ytr, y_errors=etr)

    # 7. export + reload, and verify the round-trip is faithful + differentiable
    export.save(model, out_base, coord_names=coord_names)
    reloaded = export.load(out_base)

    p_orig = model.predict(Xho)
    p_reload = reloaded.predict(Xho)
    if not np.allclose(p_orig, p_reload, rtol=1e-5, atol=1e-5):
        raise AssertionError("reloaded predict() disagrees with original model")

    import jax
    import jax.numpy as jnp
    theta0 = jnp.asarray(Xtr[0], dtype=jnp.float64)
    grad = np.asarray(jax.grad(reloaded.lnL_physical)(theta0))
    if not np.all(np.isfinite(grad)):
        raise AssertionError("jax.grad of reloaded lnL_physical is not finite")

    # held-out RMSE (on the reloaded model, the thing users actually get)
    holdout_rmse = float(np.sqrt(np.mean((p_reload - yho) ** 2)))

    return {
        "net_path": net_path,
        "out_base": out_base,
        "coords": coords,
        "method": method,
        "coord_names": coord_names,
        "n_train": int(len(ytr)),
        "n_holdout": int(len(yho)),
        "lnL_max": lnL_max,
        "holdout_rmse": holdout_rmse,
        "grad_finite": True,
        "sigma_cut": sigma_cut,
        "lnL_offset": lnL_offset,
        "cap_points": cap_points,
        "n_features": n_features,
        "n_opt_steps": n_opt_steps,
    }


def main(argv=None):
    """argparse CLI: build an artifact and print its metadata as JSON."""
    p = argparse.ArgumentParser(
        description="Export a differentiable lnL artifact from a RIFT ILE .net file.")
    p.add_argument("--net", required=True, help="path to the ILE .net file")
    p.add_argument("--out", required=True,
                   help="output base path (writes <out>.npz + <out>.meta.json)")
    p.add_argument("--coords", choices=("bns", "raw"), default="bns",
                   help="fit coordinate system (default: bns)")
    p.add_argument("--method", default="rff",
                   help="jax_gp interpolator: rff|exact|svgp (default: rff)")
    p.add_argument("--sigma-cut", type=float, default=0.6,
                   help="drop ILE points with sigma_lnL above this (default: 0.6)")
    p.add_argument("--lnL-offset", type=float, default=40.0,
                   help="keep lnL > max - lnL_offset (default: 40.0)")
    p.add_argument("--cap-points", type=int, default=8000,
                   help="random-subsample to at most this many points (default: 8000)")
    p.add_argument("--n-features", type=int, default=512,
                   help="number of random Fourier features (RFF) (default: 512)")
    p.add_argument("--n-opt-steps", type=int, default=300,
                   help="optimizer steps for the fit (default: 300)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    args = p.parse_args(argv)

    meta = build_artifact(
        net_path=args.net, out_base=args.out, coords=args.coords,
        method=args.method, sigma_cut=args.sigma_cut, lnL_offset=args.lnL_offset,
        cap_points=args.cap_points, n_features=args.n_features,
        n_opt_steps=args.n_opt_steps, seed=args.seed)
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
