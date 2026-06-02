"""
Random Fourier Features (RFF) GP interpolator.

A parametric approximation to a stationary (RBF) GP using Rahimi & Recht random
features: the kernel ``k(x,x') = sf^2 exp(-||x-x'||^2 / 2 ell^2)`` is approximated
by an inner product of ``M`` random cosine features.  Fitting then reduces to
*Bayesian linear regression* in feature space, which is

  * cheap: O(N M^2), linear in the number of training points N, and
  * trivially AD-compatible and exportable -- the model is just (omega, b, w),
    and the predictive mean ``phi(x) . w`` is a smooth closed form.

This is the cheapest-to-export method and the natural first scalable baseline.
It is less faithful for very sharp or strongly non-stationary lnL peaks (a known
RFF limitation); the benchmark harness quantifies that against the exact GP.

Frequencies ``omega`` are sampled once from N(0, I) and the lengthscale enters
smoothly as ``omega / ell`` inside the features, so all three hyperparameters
(lengthscale, signal amplitude, noise) are optimized by gradient descent on the
exact feature-space marginal likelihood.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .interface import BaseInterpolator


class RFFInterpolator(BaseInterpolator):
    name = "rff"

    def __init__(self, n_features=512, n_opt_steps=300, lr=0.05, seed=0, jitter=1e-6):
        super().__init__(jitter=jitter)
        self.n_features = int(n_features)
        self.n_opt_steps = int(n_opt_steps)
        self.lr = float(lr)
        self.seed = int(seed)

    # --- feature map ---------------------------------------------------- #
    def _features(self, X, log_ell, log_sf):
        """phi(X): [n,d] -> [n,M].  Single point [d] is promoted to [1,M]."""
        X = jnp.atleast_2d(X)
        ell = jnp.exp(log_ell)
        sf2 = jnp.exp(2.0 * log_sf)
        proj = (X / ell) @ self.omega.T + self.b          # [n, M]
        return jnp.sqrt(2.0 * sf2 / self.n_features) * jnp.cos(proj)

    # --- fit ------------------------------------------------------------ #
    def _fit_whitened(self, Xw, yw, yerr_w):
        n, d = Xw.shape
        M = self.n_features
        key = jax.random.PRNGKey(self.seed)
        k1, k2 = jax.random.split(key)
        self.omega = jax.random.normal(k1, (M, d))        # fixed frequencies
        self.b = jax.random.uniform(k2, (M,), minval=0.0, maxval=2.0 * jnp.pi)

        if yerr_w is not None:
            base_noise = float(jnp.maximum(jnp.mean(yerr_w ** 2), 1e-4))
        else:
            base_noise = 0.01
        eyeM = jnp.eye(M)

        def nlml(params):
            # Negative log marginal likelihood of y ~ N(0, Phi Phi^T + sn2 I_n),
            # evaluated via the M-dimensional system A = Phi^T Phi + sn2 I_M and
            # the Sylvester determinant identity
            #   |Phi Phi^T + sn2 I_n| = sn2^(n-M) |A|.
            Phi = self._features(Xw, params["log_ell"], params["log_sf"])
            sn2 = jnp.exp(2.0 * params["log_sn"])
            A = Phi.T @ Phi + sn2 * eyeM + self.jitter * eyeM
            L = jnp.linalg.cholesky(A)
            Phiy = Phi.T @ yw
            alpha = jax.scipy.linalg.cho_solve((L, True), Phiy)
            quad = (yw @ yw - Phiy @ alpha) / sn2
            logdet = (n - M) * jnp.log(sn2) + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            return 0.5 * (quad + logdet + n * jnp.log(2.0 * jnp.pi))

        params = {
            "log_ell": jnp.asarray(0.0),
            "log_sf": jnp.asarray(0.0),
            "log_sn": jnp.asarray(0.5 * np.log(base_noise)),
        }
        opt = optax.adam(self.lr)
        state = opt.init(params)

        @jax.jit
        def step(params, state):
            loss, g = jax.value_and_grad(nlml)(params)
            updates, state = opt.update(g, state)
            return optax.apply_updates(params, updates), state, loss

        for _ in range(self.n_opt_steps):
            params, state, _loss = step(params, state)
        self.params = {k: jax.lax.stop_gradient(v) for k, v in params.items()}

        # Posterior weight mean: w = A^{-1} Phi^T y, mean prediction = phi(x*) . w
        Phi = self._features(Xw, self.params["log_ell"], self.params["log_sf"])
        sn2 = jnp.exp(2.0 * self.params["log_sn"])
        A = Phi.T @ Phi + sn2 * eyeM + self.jitter * eyeM
        L = jnp.linalg.cholesky(A)
        self.w_mean = jax.scipy.linalg.cho_solve((L, True), Phi.T @ yw)

    # --- prediction ----------------------------------------------------- #
    def _lnL_whitened(self, xw):
        phi = self._features(xw, self.params["log_ell"], self.params["log_sf"])[0]
        return phi @ self.w_mean

    # --- serialization -------------------------------------------------- #
    def _export_params(self):
        return {
            "omega": self.omega,
            "b": self.b,
            "w_mean": self.w_mean,
            "log_ell": self.params["log_ell"],
            "log_sf": self.params["log_sf"],
        }

    def _import_params(self, p):
        self.omega = jnp.asarray(p["omega"])
        self.b = jnp.asarray(p["b"])
        self.w_mean = jnp.asarray(p["w_mean"])
        self.n_features = int(self.omega.shape[0])
        self.params = {"log_ell": jnp.asarray(p["log_ell"]),
                       "log_sf": jnp.asarray(p["log_sf"])}
