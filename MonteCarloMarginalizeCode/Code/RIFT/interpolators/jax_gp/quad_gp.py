"""
Quadratic-core + GP-residual interpolator (PE-grade accuracy on sharp peaks).

A single global GP kernel cannot match a razor-sharp, near-Gaussian lnL peak to the
few-percent width precision PE demands (its curvature is bounded by amplitude/length-
scale). So we split:

    lnL(x) ~= Q(x) + GP_residual(x)

  * Q is a quadratic whose Hessian is the *exact* local Fisher curvature -- fit by
    posterior-weighted (exp(lnL)) least squares, so the sharp directions (mc/mu1) get
    their true width by construction. H is projected to negative-semidefinite so Q is
    a proper peak.
  * GP_residual fits lnL - Q, which is smooth and broad (the quadratic removed the
    sharp structure), exactly the regime GPs handle well.

Both terms are pure JAX, so lnL_physical stays differentiable for the AD export and
gradient sampling.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .interface import BaseInterpolator


def _poly_design(X):
    """Degree-2 design matrix [1, x_i, x_i x_j (i<=j)] and the (i,j) index list."""
    n, d = X.shape
    cols = [np.ones(n)] + [X[:, i] for i in range(d)]
    idx = []
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            idx.append((i, j))
    return np.column_stack(cols), idx


class QuadraticPlusGPInterpolator(BaseInterpolator):
    name = "quadgp"

    def __init__(self, gp_method="exact", n_opt_steps=200, reg=1e-6,
                 keep_curv_frac=0.05, **gp_kwargs):
        super().__init__()
        self.gp_method = gp_method
        self.n_opt_steps = int(n_opt_steps)
        self.reg = float(reg)
        # Keep only the SHARP eigen-directions of the Fisher Hessian in Q (curvature
        # within this fraction of the sharpest); the gentle/broad directions are left
        # to the GP residual, which captures their true (wider) width. Imposing the
        # quadratic on broad directions over-curves them -> posterior too narrow.
        self.keep_curv_frac = float(keep_curv_frac)
        self.gp_kwargs = gp_kwargs

    def _fit_whitened(self, Xw, yw, yerr_w):
        Xw_np = np.asarray(Xw, dtype=np.float64)
        yw_np = np.asarray(yw, dtype=np.float64)
        n, d = Xw_np.shape

        # Posterior weights (RAW lnL) -> weighted-quadratic regression recovers the
        # Fisher curvature at the peak (the sharp directions' true width).
        w = np.exp((yw_np - yw_np.max()) * self.y_std)
        w = np.clip(w, 1e-10, None)
        Phi, idx = _poly_design(Xw_np)
        WPhi = Phi * w[:, None]
        A = Phi.T @ WPhi + self.reg * np.eye(Phi.shape[1])
        beta = np.linalg.solve(A, WPhi.T @ yw_np)

        c0 = float(beta[0])
        c1 = beta[1:1 + d].copy()
        H = np.zeros((d, d))
        for k, (i, j) in enumerate(idx):
            v = beta[1 + d + k]
            if i == j:
                H[i, i] = 2.0 * v
            else:
                H[i, j] = v
                H[j, i] = v
        # Project H to negative-semidefinite, AND keep only the sharp eigen-directions
        # (curvature within keep_curv_frac of the sharpest); zero the gentle ones so
        # the GP residual -- not the quadratic -- sets the broad-direction widths.
        ev, U = np.linalg.eigh(H)
        ev = np.minimum(ev, 0.0)
        if (-ev).max() > 0:
            ev = np.where(-ev >= self.keep_curv_frac * (-ev).max(), ev, 0.0)
        H = (U * ev) @ U.T

        self._c0 = jnp.asarray(c0)
        self._c1 = jnp.asarray(c1)
        self._H = jnp.asarray(H)

        # Residual = lnL - Q, fit by a (smooth) GP in the same whitened X space. We
        # treat Xw as the sub-GP's 'physical' input; its own (near-identity) whitening
        # is harmless and its lnL_physical returns the residual in lnL units.
        Q = c0 + Xw_np @ c1 + 0.5 * np.einsum("ni,ij,nj->n", Xw_np, H, Xw_np)
        r = yw_np - Q
        from . import get_interpolator
        cls = get_interpolator(self.gp_method)
        kw = dict(self.gp_kwargs)
        kw.setdefault("n_opt_steps", self.n_opt_steps)
        ye = np.asarray(yerr_w) if yerr_w is not None else None
        self._resid = cls(**kw).fit(Xw_np, r, y_errors=ye)

    def _eval_Q(self, xw):
        return self._c0 + self._c1 @ xw + 0.5 * xw @ (self._H @ xw)

    def _lnL_whitened(self, xw):
        return self._eval_Q(xw) + self._resid.lnL_physical(xw)

    # --- serialization -------------------------------------------------- #
    # quadgp is the one method that nests a *full* sub-interpolator (the
    # residual GP) inside itself, so it cannot round-trip through the base
    # array-only export.  _export_params/_import_params carry just the
    # quadratic core; export_state/from_state are overridden below to embed
    # the residual model -- which round-trips through its own export_state --
    # alongside it.  The on-disk contract (export.save/load -> export_state/
    # from_state) is unchanged: the residual's arrays are namespaced under a
    # "_resid_" prefix and its meta is nested under meta["resid_meta"].
    _RESID_PREFIX = "_resid_"

    def _export_params(self):
        return {
            "c0": np.asarray(self._c0),
            "c1": np.asarray(self._c1),
            "H": np.asarray(self._H),
        }

    def _import_params(self, p):
        self._c0 = jnp.asarray(p["c0"])
        self._c1 = jnp.asarray(p["c1"])
        self._H = jnp.asarray(p["H"])

    def export_state(self):
        meta, arrays = super().export_state()
        resid_meta, resid_arrays = self._resid.export_state()
        meta["resid_meta"] = resid_meta
        arrays.update({self._RESID_PREFIX + k: v for k, v in resid_arrays.items()})
        return meta, arrays

    @classmethod
    def from_state(cls, meta, arrays):
        from . import get_interpolator
        pre = cls._RESID_PREFIX
        resid_arrays = {k[len(pre):]: v for k, v in arrays.items() if k.startswith(pre)}
        core_arrays = {k: v for k, v in arrays.items() if not k.startswith(pre)}
        obj = super().from_state(meta, core_arrays)   # restores whitening + quadratic core
        resid_meta = meta["resid_meta"]
        rcls = get_interpolator(resid_meta["method"])
        obj._resid = rcls.from_state(resid_meta, resid_arrays)
        obj.gp_method = resid_meta["method"]          # keep a re-export self-consistent
        return obj
