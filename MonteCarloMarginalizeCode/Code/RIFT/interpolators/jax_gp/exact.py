"""
Exact GP interpolator (tinygp).

This is O(N^3) and intended as the *accuracy reference baseline* at small N: the
benchmark harness measures the scalable approximations (RFF, SVGP) against it,
and it is the source of "ground truth" GP predictions when no analytic truth is
available.  Do not use it at production N -- that is exactly what the scalable
methods exist to replace.

Hyperparameters (amplitude, lengthscale, noise) are optimized by maximizing the
exact log marginal likelihood with optax.  Everything is pure JAX, so the
resulting ``lnL_physical`` is differentiable like the other backends.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
from tinygp import kernels, GaussianProcess

from .interface import BaseInterpolator


class ExactGPInterpolator(BaseInterpolator):
    name = "exact"

    def __init__(self, n_opt_steps=200, lr=0.05, jitter=1e-6):
        super().__init__(jitter=jitter)
        self.n_opt_steps = int(n_opt_steps)
        self.lr = float(lr)

    def _build_gp(self, params, Xw):
        amp2 = jnp.exp(2.0 * params["log_amp"])
        scale = jnp.exp(params["log_scale"])
        sn2 = jnp.exp(2.0 * params["log_sn"])
        kernel = amp2 * kernels.ExpSquared(scale)
        return GaussianProcess(kernel, Xw, diag=sn2 + self.jitter, mean=0.0)

    def _fit_whitened(self, Xw, yw, yerr_w):
        self.Xw = Xw
        self.yw = yw
        if yerr_w is not None:
            base_noise = float(jnp.maximum(jnp.mean(yerr_w ** 2), 1e-4))
        else:
            base_noise = 0.01

        params = {
            "log_amp": jnp.asarray(0.0),
            "log_scale": jnp.asarray(0.0),
            "log_sn": jnp.asarray(0.5 * np.log(base_noise)),
        }

        def nll(params):
            gp = self._build_gp(params, Xw)
            return -gp.log_probability(yw)

        opt = optax.adam(self.lr)
        state = opt.init(params)

        @jax.jit
        def step(params, state):
            loss, g = jax.value_and_grad(nll)(params)
            updates, state = opt.update(g, state)
            return optax.apply_updates(params, updates), state, loss

        for _ in range(self.n_opt_steps):
            params, state, _loss = step(params, state)
        self.params = {k: jax.lax.stop_gradient(v) for k, v in params.items()}
        self.gp = self._build_gp(self.params, Xw)

    def _lnL_whitened(self, xw):
        _, cond_gp = self.gp.condition(self.yw, jnp.atleast_2d(xw))
        return cond_gp.loc[0]

    # --- serialization -------------------------------------------------- #
    def _export_params(self):
        # Exact GP must carry its full training set -- that is the price of exact.
        return {
            "Xw": self.Xw,
            "yw": self.yw,
            "log_amp": self.params["log_amp"],
            "log_scale": self.params["log_scale"],
            "log_sn": self.params["log_sn"],
        }

    def _import_params(self, p):
        self.Xw = jnp.asarray(p["Xw"])
        self.yw = jnp.asarray(p["yw"])
        self.params = {"log_amp": jnp.asarray(p["log_amp"]),
                       "log_scale": jnp.asarray(p["log_scale"]),
                       "log_sn": jnp.asarray(p["log_sn"])}
        self.gp = self._build_gp(self.params, self.Xw)
