"""Tempered SMC with MALA moves on a pre-built surrogate.

Engine version: takes Fit objects from RIFT.misc.tracer_placement.fits.
Stateless across calls — caller owns the `state` dict (with `mala_eps`,
optionally other adaptive choices).
"""
import numpy as np


def _ess(logw):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return 1.0 / np.sum(w * w)


def _systematic_resample(logw, rng):
    w = np.exp(logw - logw.max()); w /= w.sum()
    n = len(w)
    u = (rng.uniform() + np.arange(n)) / n
    return np.clip(np.searchsorted(np.cumsum(w), u), 0, n - 1)


def _mala_step(X, f_eval, f_grad, eps, prior_box, rng):
    g = f_grad(X)
    prop = X + 0.5 * eps**2 * g + eps * rng.normal(size=X.shape)
    lo = prior_box[:, 0]; hi = prior_box[:, 1]
    prop = np.where(prop > hi, 2 * hi - prop, prop)
    prop = np.where(prop < lo, 2 * lo - prop, prop)
    prop = np.clip(prop, lo, hi)
    g_prop = f_grad(prop)
    log_qf = -np.sum((prop - X - 0.5 * eps**2 * g)**2, axis=1) / (2 * eps**2)
    log_qb = -np.sum((X - prop - 0.5 * eps**2 * g_prop)**2, axis=1) / (2 * eps**2)
    log_alpha = f_eval(prop) - f_eval(X) + log_qb - log_qf
    acc = np.log(rng.uniform(size=len(X))) < log_alpha
    return np.where(acc[:, None], prop, X), float(acc.mean())


def iterate(particles, *, surrogate, surrogate_prev=None,
            prior_box, rng, state=None,
            n_mala_steps=8, target_ess_frac=0.5,
            birth_death_rate=0.0, **_):
    state = dict(state or {})
    X = np.asarray(particles, dtype=float).copy()
    n, _d = X.shape
    if state.get("mala_eps") is None:
        cov_diag = np.diag(np.cov(X.T)) if X.shape[1] > 1 else np.array([X.var()])
        state["mala_eps"] = 0.3 * float(np.sqrt(cov_diag + 1e-8).mean())

    accept_log = []

    if surrogate_prev is not None:
        beta = 0.0
        logw = np.zeros(n)
        delta_f = surrogate.predict(X) - surrogate_prev.predict(X)
        while beta < 1.0:
            lo, hi = 0.0, 1.0 - beta
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                if _ess(logw + mid * delta_f) > target_ess_frac * n:
                    lo = mid
                else:
                    hi = mid
            d_beta = lo if lo > 1e-6 else (1.0 - beta)
            logw = logw + d_beta * delta_f
            beta += d_beta
            if _ess(logw) < target_ess_frac * n or beta >= 1.0:
                idx = _systematic_resample(logw, rng)
                X = X[idx]
                logw = np.zeros(n)
                delta_f = surrogate.predict(X) - surrogate_prev.predict(X)

                def fe(z, beta=beta, s=surrogate, sp=surrogate_prev):
                    return beta * s.predict(z) + (1 - beta) * sp.predict(z)

                def fg(z, beta=beta, s=surrogate, sp=surrogate_prev):
                    return beta * s.grad(z) + (1 - beta) * sp.grad(z)

                for _ in range(n_mala_steps):
                    X, acc = _mala_step(X, fe, fg, state["mala_eps"], prior_box, rng)
                    accept_log.append(acc)
                    state["mala_eps"] *= np.exp(0.1 * (acc - 0.574))
                    state["mala_eps"] = float(np.clip(state["mala_eps"], 1e-4, 5.0))
    else:
        for _ in range(n_mala_steps):
            X, acc = _mala_step(X, surrogate.predict, surrogate.grad,
                                state["mala_eps"], prior_box, rng)
            accept_log.append(acc)
            state["mala_eps"] *= np.exp(0.1 * (acc - 0.574))
            state["mala_eps"] = float(np.clip(state["mala_eps"], 1e-4, 5.0))

    info = {"state": state,
            "mean_accept": float(np.mean(accept_log)) if accept_log else float("nan"),
            "n_steps": len(accept_log)}
    return X, info
