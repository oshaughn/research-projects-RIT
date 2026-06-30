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
from tinygp import kernels, transforms, GaussianProcess

from .interface import BaseInterpolator


class ExactGPInterpolator(BaseInterpolator):
    name = "exact"

    def __init__(self, n_opt_steps=200, lr=0.05, jitter=1e-6):
        super().__init__(jitter=jitter)
        self.n_opt_steps = int(n_opt_steps)
        self.lr = float(lr)

    def _build_gp(self, params, Xw, yvar):
        amp2 = jnp.exp(2.0 * params["log_amp"])
        inv_scale = jnp.exp(-params["log_scale"])     # [d] -> ARD (per-dim lengthscale)
        # Heteroscedastic noise: reported per-point MC variance + a learnable floor.
        diag = yvar + jnp.exp(2.0 * params["log_sn"]) + self.jitter
        kernel = amp2 * transforms.Linear(inv_scale, kernels.ExpSquared())
        return GaussianProcess(kernel, Xw, diag=diag, mean=0.0)

    def _fit_whitened(self, Xw, yw, yerr_w):
        self.Xw = Xw
        self.yw = yw
        d = Xw.shape[1]
        # Per-point MC variance on lnL (0 if not provided -> homoscedastic).
        self.yvar = (yerr_w ** 2) if yerr_w is not None else jnp.zeros(Xw.shape[0])

        # CONSTRAIN the lengthscale to ~the peak width (high-lnL spread). Free
        # marginal-likelihood fitting drives it long (global trend / far anchors) ->
        # under-curved peak -> posterior too broad. Bounding it forces the sharp
        # curvature to be captured (cf. sklearn length_scale_bounds in legacy CIP).
        yw_np = np.asarray(yw)
        dlnL = (yw_np.max() - yw_np) * self.y_std        # raw lnL below the peak
        peak_mask = dlnL < 2.0                           # ~near-peak (curvature) region
        if int(peak_mask.sum()) < max(20, 3 * d):
            peak_mask = dlnL < 10.0
        peak_std = np.clip(np.std(np.asarray(Xw)[peak_mask], axis=0), 1e-3, 3.0)
        log_ls_lo = jnp.asarray(np.log(0.2 * peak_std))
        log_ls_hi = jnp.asarray(np.log(1.0 * peak_std))

        params = {
            "log_amp": jnp.asarray(0.0),
            "log_scale": jnp.asarray(np.log(peak_std)),   # ARD: peak-matched init
            "log_sn": jnp.asarray(0.5 * np.log(0.01)),
        }

        def nll(params):
            gp = self._build_gp(params, Xw, self.yvar)
            return -gp.log_probability(yw)

        opt = optax.adam(self.lr)
        state = opt.init(params)

        @jax.jit
        def step(params, state):
            loss, g = jax.value_and_grad(nll)(params)
            updates, state = opt.update(g, state)
            params = optax.apply_updates(params, updates)
            params["log_scale"] = jnp.clip(params["log_scale"], log_ls_lo, log_ls_hi)
            return params, state, loss

        for _ in range(self.n_opt_steps):
            params, state, _loss = step(params, state)
        self.params = {k: jax.lax.stop_gradient(v) for k, v in params.items()}
        self.gp = self._build_gp(self.params, Xw, self.yvar)

    def _lnL_whitened(self, xw):
        _, cond_gp = self.gp.condition(self.yw, jnp.atleast_2d(xw))
        return cond_gp.loc[0]

    # --- serialization -------------------------------------------------- #
    def _export_params(self):
        # Exact GP must carry its full training set -- that is the price of exact.
        return {
            "Xw": self.Xw,
            "yw": self.yw,
            "yvar": self.yvar,
            "log_amp": self.params["log_amp"],
            "log_scale": self.params["log_scale"],
            "log_sn": self.params["log_sn"],
        }

    def _import_params(self, p):
        self.Xw = jnp.asarray(p["Xw"])
        self.yw = jnp.asarray(p["yw"])
        self.yvar = jnp.asarray(p["yvar"])
        self.params = {"log_amp": jnp.asarray(p["log_amp"]),
                       "log_scale": jnp.asarray(p["log_scale"]),
                       "log_sn": jnp.asarray(p["log_sn"])}
        self.gp = self._build_gp(self.params, self.Xw, self.yvar)
