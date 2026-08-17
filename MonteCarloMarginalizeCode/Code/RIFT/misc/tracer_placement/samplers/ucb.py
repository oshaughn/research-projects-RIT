"""UCB (Upper Confidence Bound) placement.

Target regime: expensive ILE, few generations, ~50-500 points per generation.
Each evaluation point should be "earned" -- either we expect a large mean lnL
*or* we have large uncertainty about lnL there. UCB picks both: rank
candidates by

    score(lambda) = mu(lambda) + kappa * sigma(lambda)

where (mu, sigma) come from the surrogate's predict_with_std. The N chosen
points are taken by greedy descending-score selection with a built-in
self-avoidance constraint (population-covariance Mahalanobis distance).

Local polish: each selected point is optionally hill-climbed for a few steps
against `score`. For smooth-gradient fits (RBF, quadratic, GP) we use
gradient ascent on `score`; for piecewise-constant fits (RF) we use a
coordinate-hop polish that perturbs each coordinate independently and keeps
the move only if `score` improved. The fit signals which polish strategy to
use via FitBase.smooth_gradient.

Sampler signature matches the rest of the tracer engine:
    ucb_place(particles, *, surrogate, surrogate_prev=None, prior_box, rng,
              kappa=2.0, n_candidates=20000, polish_steps=20,
              min_separation_factor=0.0, state=None, **_) -> (X_out, info)

`particles` only sets the output count (len(particles)) and the candidate
seed pool. surrogate_prev is ignored (UCB doesn't bridge).
"""

import numpy as np


# ----------------- candidate generation ---------------------------------- #

def _candidates(rng, particles, prior_box, n_candidates):
    """Mix candidates from current particles (jittered) and uniform draws
    from the prior box. Returns array (n_candidates, d)."""
    d = prior_box.shape[0]
    n_seed = max(1, n_candidates // 4)
    # uniform fill (exploration)
    n_unif = n_candidates - n_seed
    lo = prior_box[:, 0]; hi = prior_box[:, 1]
    unif = lo + (hi - lo) * rng.uniform(size=(n_unif, d))
    # jittered current particles (exploitation seed)
    if len(particles) > 0:
        idx = rng.integers(0, len(particles), size=n_seed)
        if particles.shape[1] > 1:
            scale = np.std(particles, axis=0) + 1e-9
        else:
            scale = np.array([particles.std() + 1e-9])
        jitter = 0.25 * scale * rng.normal(size=(n_seed, d))
        seed = particles[idx] + jitter
        seed = np.clip(seed, lo, hi)
    else:
        seed = unif[:n_seed]
    return np.vstack([seed, unif])


# ----------------- greedy selection with Mahalanobis spacing ------------- #

def _mahalanobis_greedy(candidates, scores, n_keep, icov, min_mah_dist):
    """Pick n_keep candidates by descending score, rejecting any candidate
    within `min_mah_dist` Mahalanobis distance of an already-kept point."""
    order = np.argsort(-scores)
    kept = []
    if min_mah_dist <= 0:
        # No self-avoidance: just take top-n.
        return candidates[order[:n_keep]]
    for i in order:
        x = candidates[i]
        if not kept:
            kept.append(x)
            continue
        diffs = np.asarray(kept) - x
        d2 = np.einsum("ij,jk,ik->i", diffs, icov, diffs)
        if (d2 >= min_mah_dist ** 2).all():
            kept.append(x)
        if len(kept) >= n_keep:
            break
    # If under-filled (very tight separation), fall back to top-n without
    # self-avoidance for the remainder.
    if len(kept) < n_keep:
        fill = [candidates[i] for i in order if not any(
            np.array_equal(candidates[i], k) for k in kept)]
        kept += fill[: n_keep - len(kept)]
    return np.asarray(kept)


# ----------------- local polish: gradient vs coord-hop ------------------- #

def _polish_gradient(x, surrogate, kappa, prior_box, n_steps, scale, rng):
    """Gradient ascent on mu + kappa*sigma. Uses FitBase.grad for mu;
    treats sigma as locally piecewise-smooth and falls back to finite
    differences."""
    eps = 1e-3 * scale
    step = 0.1 * scale
    lo = prior_box[:, 0]; hi = prior_box[:, 1]
    x = x.copy()
    for _ in range(n_steps):
        gmu = surrogate.grad(x[None])[0]
        # FD sigma gradient (small extra cost; sigma is cheap for RF/quadratic)
        gsig = np.zeros_like(x)
        for k in range(len(x)):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            _, sp = surrogate.predict_with_std(xp[None])
            _, sm = surrogate.predict_with_std(xm[None])
            gsig[k] = (sp[0] - sm[0]) / (2 * eps)
        g = gmu + kappa * gsig
        norm = np.linalg.norm(g) + 1e-12
        x = np.clip(x + step * g / norm, lo, hi)
    return x


def _polish_coord_hop(x, surrogate, kappa, prior_box, n_steps, scale, rng):
    """For piecewise-constant fits (RF). Each step: pick a random coordinate,
    propose a Gaussian hop, accept if mu + kappa*sigma went up. Greedy."""
    lo = prior_box[:, 0]; hi = prior_box[:, 1]
    x = x.copy()
    def _score(xi):
        m, s = surrogate.predict_with_std(xi[None])
        return float(m[0] + kappa * s[0])
    best = _score(x)
    for _ in range(n_steps):
        k = rng.integers(0, len(x))
        delta = 0.5 * scale[k] * rng.normal()
        x_prop = x.copy(); x_prop[k] = float(np.clip(x[k] + delta, lo[k], hi[k]))
        sc = _score(x_prop)
        if sc > best:
            x = x_prop
            best = sc
    return x


def _polish(x, surrogate, kappa, prior_box, n_steps, scale, rng):
    if n_steps <= 0:
        return x
    if getattr(surrogate, "smooth_gradient", True):
        return _polish_gradient(x, surrogate, kappa, prior_box, n_steps, float(np.mean(scale)), rng)
    return _polish_coord_hop(x, surrogate, kappa, prior_box, n_steps, scale, rng)


# ----------------- driver ------------------------------------------------ #

def iterate(particles, *, surrogate, surrogate_prev=None,
            prior_box, rng, state=None,
            kappa=2.0, n_candidates=20000,
            polish_steps=20, min_separation_factor=0.25,
            **_):
    state = dict(state or {})
    X_in = np.asarray(particles, dtype=float)
    n_out = len(X_in)
    if n_out == 0:
        return X_in.copy(), {"state": state, "note": "no input particles"}

    if not getattr(surrogate, "has_uncertainty", False):
        # UCB without uncertainty degenerates to greedy mean-maximization,
        # which is not what the user wants. Warn but proceed (kappa effectively 0).
        import sys
        sys.stderr.write(
            "samplers.ucb: surrogate has no uncertainty estimate "
            "(predict_with_std returns zeros); UCB will degenerate to greedy "
            "mean-maximization. Use --tracer-fit-method gp_linmean (calibrated "
            "GP posterior variance) or rf (tree disagreement).\n")

    # 1. Build candidate pool
    cand = _candidates(rng, X_in, prior_box, n_candidates)
    mu, sigma = surrogate.predict_with_std(cand)
    score = mu + kappa * sigma

    # 2. Greedy select with Mahalanobis self-avoidance
    if X_in.shape[1] > 1:
        cov_in = np.cov(X_in.T)
    else:
        cov_in = np.array([[float(X_in.std() ** 2) + 1e-12]])
    cov_in = np.atleast_2d(cov_in) + 1e-10 * np.eye(prior_box.shape[0])
    icov = np.linalg.inv(cov_in)
    # min_separation_factor is in *Mahalanobis units* (so 0.25 = one-quarter std)
    X_pick = _mahalanobis_greedy(cand, score, n_out, icov, min_separation_factor)

    # 3. Local polish per point
    if X_in.shape[1] > 1:
        scale = np.sqrt(np.diag(cov_in))
    else:
        scale = np.array([float(np.sqrt(cov_in[0, 0]))])
    X_out = np.array([
        _polish(x, surrogate, kappa, prior_box, polish_steps, scale, rng)
        for x in X_pick
    ])

    info = {
        "state": state,
        "kappa": kappa,
        "n_candidates": n_candidates,
        "min_separation_factor": min_separation_factor,
        "polish_steps": polish_steps,
        "polish_strategy": "gradient" if getattr(surrogate, "smooth_gradient", True) else "coord_hop",
    }
    return X_out, info
