"""
Accuracy and cost metrics for interpolator benchmarking.

The metrics deliberately separate three things downstream users care about:

  * value fidelity everywhere       -> lnL_rmse
  * value fidelity *where it matters* -> peak_weighted_rmse (weighted by the
    posterior weight exp(lnL - lnLmax), so the peak dominates)
  * gradient fidelity               -> grad_cosine / grad_relerr  (the property
    trees / KDE cannot provide, and the whole point of AD-compatible export)

plus wall-clock fit/predict cost, so "robust and not onerous" is measurable.
"""
from __future__ import annotations

import time

import numpy as np


def lnL_rmse(pred, truth):
    pred = np.asarray(pred); truth = np.asarray(truth)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def peak_weighted_rmse(pred, truth):
    """RMSE weighted by posterior weight exp(lnL - max lnL)."""
    truth = np.asarray(truth)
    w = np.exp(truth - np.max(truth))
    w = w / np.sum(w)
    return float(np.sqrt(np.sum(w * (np.asarray(pred) - truth) ** 2)))


def gradient_metrics(model, truth, Xt, n_points=64, rng=None):
    """Compare model gradients to analytic truth gradients at sampled test points.

    Returns (mean_cosine, median_relerr).  Uses the model's jitted ``grad_fn``
    (compiled once, then mapped over points) so cost is one compilation plus
    cheap evaluations, not one compilation per point.
    """
    import jax
    rng = np.random.default_rng(0) if rng is None else rng
    Xt = np.asarray(Xt)
    idx = rng.choice(len(Xt), size=min(n_points, len(Xt)), replace=False)
    gt_all = truth.grad(Xt[idx])
    vg = jax.vmap(model.grad_fn())                       # (theta) -> (lnL, grad)
    _, g_all = vg(np.asarray(Xt[idx]))
    g_all = np.asarray(g_all)
    cosines, relerrs = [], []
    for g, gt in zip(g_all, gt_all):
        ng, ngt = np.linalg.norm(g), np.linalg.norm(gt)
        if ng > 0 and ngt > 0:
            cosines.append(float(g @ gt / (ng * ngt)))
            relerrs.append(float(np.linalg.norm(g - gt) / ngt))
    return (float(np.mean(cosines)) if cosines else np.nan,
            float(np.median(relerrs)) if relerrs else np.nan)


def timed_fit(model, X, y, y_errors=None):
    t0 = time.perf_counter()
    model.fit(X, y, y_errors=y_errors)
    return model, time.perf_counter() - t0


def timed_predict(model, X):
    # one warm-up call (triggers JAX compilation) then time the steady-state call
    _ = model.predict(X[:8])
    t0 = time.perf_counter()
    pred = model.predict(X)
    return pred, time.perf_counter() - t0


def evaluate(model, truth, Xt, yt, fit_time):
    """Bundle all metrics for a fitted model into one row dict."""
    pred, pred_time = timed_predict(model, Xt)
    cos, relerr = gradient_metrics(model, truth, Xt)
    return {
        "rmse": lnL_rmse(pred, yt),
        "peak_rmse": peak_weighted_rmse(pred, yt),
        "grad_cos": cos,
        "grad_relerr": relerr,
        "fit_s": fit_time,
        "pred_s": pred_time,
    }
