"""Overdamped Langevin on surrogate + kNN birth-death corrector.

Engine version: takes a Fit. surrogate_prev is ignored (BD doesn't bridge).
"""
import numpy as np
from ._knn import knn_dist


def _langevin_step(X, surrogate, eps, prior_box, rng):
    g = surrogate.grad(X)
    Xp = X + eps * g + np.sqrt(2 * eps) * rng.normal(size=X.shape)
    lo = prior_box[:, 0]; hi = prior_box[:, 1]
    Xp = np.where(Xp > hi, 2 * hi - Xp, Xp)
    Xp = np.where(Xp < lo, 2 * lo - Xp, Xp)
    return np.clip(Xp, lo, hi)


def _knn_log_density(X, k=5):
    from math import lgamma, log, pi
    n, d = X.shape
    k = min(k, n - 1)
    r = knn_dist(X, X, k + 1) + 1e-12
    log_Vd = (d / 2) * log(pi) - lgamma(d / 2 + 1)
    return np.log(k) - np.log(n) - log_Vd - d * np.log(r)


def iterate(particles, *, surrogate, surrogate_prev=None,
            prior_box, rng, state=None,
            n_langevin_steps=20, n_bd_passes=3, eps_factor=0.05,
            birth_death_rate=1.0, **_):
    state = dict(state or {})
    X = np.asarray(particles, dtype=float).copy()
    n = len(X)
    scale = float(np.sqrt(np.diag(np.cov(X.T)) + 1e-8).mean()) if X.shape[1] > 1 \
            else float(X.std() + 1e-8)
    eps = eps_factor * scale**2

    for _ in range(n_langevin_steps):
        X = _langevin_step(X, surrogate, eps, prior_box, rng)

    for _ in range(n_bd_passes):
        log_rho = _knn_log_density(X, k=min(5, n - 1))
        log_pi = surrogate.predict(X)
        log_pi -= log_pi.max(); log_rho -= log_rho.max()
        score = log_rho - log_pi
        order = np.argsort(score)
        n_swap = max(1, int(round(birth_death_rate * n / 10.0)))
        kill = order[-n_swap:]
        birth = order[:n_swap]
        X[kill] = X[rng.choice(birth, size=n_swap, replace=True)] \
                  + 0.05 * scale * rng.normal(size=(n_swap, X.shape[1]))
        X = np.clip(X, prior_box[:, 0], prior_box[:, 1])
        for _ in range(3):
            X = _langevin_step(X, surrogate, eps, prior_box, rng)

    info = {"state": state, "eps": eps}
    return X, info
