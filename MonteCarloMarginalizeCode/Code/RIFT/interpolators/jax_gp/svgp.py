"""
Sparse variational GP interpolator -- Titsias (2009) collapsed bound (SGPR).

This is the scalable, calibrated-uncertainty GP for the production regime
(N ~ 2e4-5e4, d ~ 8-12).  It approximates the full GP with ``M`` inducing points
Z and maximizes the collapsed variational lower bound (ELBO), giving O(N M^2)
cost -- linear in N.  Unlike stochastic (Hensman) SVI, the collapsed bound is
deterministic and needs no minibatching for our batch-fit setting, and the
predictive mean is a closed form in a handful of cached M-sized factors, so the
exported artifact stays tiny and differentiable.

We hand-roll it in pure JAX (rather than depend on gpjax) so that:
  * there is no version coupling to a fast-moving external GP library, and
  * the predictive mean is a transparent, exportable closed form.

Kernel: scaled RBF (ExpSquared), matching ``exact.py`` so the two are directly
comparable in the benchmark.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import optax

from .interface import BaseInterpolator


def _sqdist(A, B):
    """Squared Euclidean distances, [n,d] x [m,d] -> [n,m], clipped at 0."""
    a2 = jnp.sum(A * A, axis=1)[:, None]
    b2 = jnp.sum(B * B, axis=1)[None, :]
    return jnp.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)


class SVGPInterpolator(BaseInterpolator):
    name = "svgp"

    def __init__(self, n_inducing=256, n_opt_steps=400, lr=0.02, seed=0, jitter=1e-6):
        super().__init__(jitter=jitter)
        self.n_inducing = int(n_inducing)
        self.n_opt_steps = int(n_opt_steps)
        self.lr = float(lr)
        self.seed = int(seed)

    # --- ARD RBF kernel (per-dimension lengthscales) -------------------- #
    def _kernel(self, X1, X2, log_amp, log_scale):
        amp2 = jnp.exp(2.0 * log_amp)
        scale = jnp.exp(log_scale)               # [d] -> ARD; per-dim lengthscale
        return amp2 * jnp.exp(-0.5 * _sqdist(X1 / scale, X2 / scale))

    def _kdiag(self, log_amp):
        return jnp.exp(2.0 * log_amp)

    # --- inducing-point initialization ---------------------------------- #
    def _init_inducing(self, Xw, M):
        """k-means centroids (better coverage than a random subset); fall back to
        a random subset if scikit-learn is unavailable."""
        if M >= len(Xw):
            return Xw
        try:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=M, random_state=self.seed,
                                 n_init=3, batch_size=max(256, 3 * M))
            return km.fit(Xw).cluster_centers_
        except Exception:
            rng = np.random.default_rng(self.seed)
            return Xw[rng.choice(len(Xw), size=M, replace=False)]

    # --- collapsed ELBO (Titsias) -------------------------------------- #
    def _neg_elbo(self, params, Xw, yw):
        Z = params["Z"]
        la, ls, lsn = params["log_amp"], params["log_scale"], params["log_sn"]
        n = Xw.shape[0]
        M = Z.shape[0]
        sn2 = jnp.exp(2.0 * lsn)
        eyeM = jnp.eye(M)

        Kuu = self._kernel(Z, Z, la, ls) + self.jitter * eyeM
        Kuf = self._kernel(Z, Xw, la, ls)                       # [M, n]
        Luu = jnp.linalg.cholesky(Kuu)
        V = jsla.solve_triangular(Luu, Kuf, lower=True)         # [M, n]
        G = eyeM + (V @ V.T) / sn2                              # [M, M]
        Lg = jnp.linalg.cholesky(G)
        Vy = V @ yw                                            # [M]
        c = jsla.solve_triangular(Lg, Vy / sn2, lower=True)    # [M]

        # log N(y | 0, sn2 I + Qff)
        logdet = n * jnp.log(sn2) + 2.0 * jnp.sum(jnp.log(jnp.diag(Lg)))
        quad = (yw @ yw - sn2 * (c @ c)) / sn2  # = (y.y - (Vy/sn).G^{-1}.(Vy/sn))/sn2
        log_marg = -0.5 * (n * jnp.log(2.0 * jnp.pi) + logdet + quad)
        # Titsias trace penalty: -(1/2 sn2)(tr Kff - tr Qff)
        trace_term = (n * self._kdiag(la) - jnp.sum(V * V)) / (2.0 * sn2)
        return -(log_marg - trace_term)

    # --- fit ------------------------------------------------------------ #
    def _fit_whitened(self, Xw, yw, yerr_w):
        n, d = Xw.shape
        M = min(self.n_inducing, max(2, n // 2))
        Z0 = self._init_inducing(np.asarray(Xw), M)

        if yerr_w is not None:
            base_noise = float(jnp.maximum(jnp.mean(yerr_w ** 2), 1e-4))
        else:
            base_noise = 0.01

        params = {
            "Z": jnp.asarray(Z0),
            "log_amp": jnp.asarray(0.0),
            "log_scale": jnp.zeros(d),                # ARD: one lengthscale per dim
            "log_sn": jnp.asarray(0.5 * np.log(base_noise)),
        }
        opt = optax.adam(self.lr)
        state = opt.init(params)

        @jax.jit
        def step(params, state):
            loss, g = jax.value_and_grad(self._neg_elbo)(params, Xw, yw)
            updates, state = opt.update(g, state)
            return optax.apply_updates(params, updates), state, loss

        for _ in range(self.n_opt_steps):
            params, state, _loss = step(params, state)
        self.params = {k: jax.lax.stop_gradient(v) for k, v in params.items()}

        # Cache the closed-form predictive-mean factors.
        Z = self.params["Z"]
        la, ls, lsn = self.params["log_amp"], self.params["log_scale"], self.params["log_sn"]
        sn2 = jnp.exp(2.0 * lsn)
        eyeM = jnp.eye(Z.shape[0])
        Kuu = self._kernel(Z, Z, la, ls) + self.jitter * eyeM
        Kuf = self._kernel(Z, Xw, la, ls)
        self._Luu = jnp.linalg.cholesky(Kuu)
        V = jsla.solve_triangular(self._Luu, Kuf, lower=True)
        self._Lg = jnp.linalg.cholesky(eyeM + (V @ V.T) / sn2)
        self._c = jsla.solve_triangular(self._Lg, (V @ yw) / sn2, lower=True)  # [M]
        self._Z, self._la, self._ls = Z, la, ls

    # --- prediction ----------------------------------------------------- #
    def _lnL_whitened(self, xw):
        Kus = self._kernel(self._Z, jnp.atleast_2d(xw), self._la, self._ls)  # [M,1]
        ws = jsla.solve_triangular(self._Luu, Kus, lower=True)               # [M,1]
        t = jsla.solve_triangular(self._Lg, ws, lower=True)[:, 0]            # [M]
        return t @ self._c

    # --- serialization -------------------------------------------------- #
    def _export_params(self):
        # Only the M-sized closed-form factors are needed -- not the training set.
        return {
            "Z": self._Z,
            "Luu": self._Luu,
            "Lg": self._Lg,
            "c": self._c,
            "log_amp": self._la,
            "log_scale": self._ls,
        }

    def _import_params(self, p):
        self._Z = jnp.asarray(p["Z"])
        self._Luu = jnp.asarray(p["Luu"])
        self._Lg = jnp.asarray(p["Lg"])
        self._c = jnp.asarray(p["c"])
        self._la = jnp.asarray(p["log_amp"])
        self._ls = jnp.asarray(p["log_scale"])
