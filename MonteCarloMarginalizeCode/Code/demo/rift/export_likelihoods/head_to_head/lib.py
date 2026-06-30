"""
Head-to-head test library: RF oracle, on-support design, domain boundaries, and the
GP fit, for comparing the differentiable GP surrogate to RIFT's production RF fit.

Coordinates: the BNS Morisaki/tidal fit coords (mu1, mu2, delta_mc, LambdaTilde,
DeltaLambdaTilde); low-level sampling coords (mc, delta_mc, s1z, s2z, lambda1,
lambda2).  The data path defaults to the committed ``data/all.net`` (see PROVENANCE.md).

This is the cleaned, repo version of the scratch-workspace exploration; see the paper /
docs for the methodology and what each piece is for.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import jax.numpy as jnp

from RIFT.interpolators.jax_gp.benchmark.datasets import (
    BNS_FIT_COORDS, load_ile_net, mc_delta_from_m1m2, to_fit_coordinates)
from RIFT.interpolators.jax_gp.benchmark.baselines import RFBaseline

LOW = ["mc", "delta_mc", "s1z", "s2z", "lambda1", "lambda2"]
FIT = list(BNS_FIT_COORDS)
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NET = os.path.join(_HERE, "data", "all.net")


# --- general boundary / special-locus framework ----------------------------- #
@dataclass
class Coord:
    name: str
    lo: Optional[float] = None
    hi: Optional[float] = None


@dataclass
class Domain:
    coords: List[Coord]
    loci: List = field(default_factory=list)   # (name, distance_fn) tuples

    @property
    def names(self):
        return [c.name for c in self.coords]

    def clip(self, X):
        X = np.array(X, float, copy=True)
        for i, c in enumerate(self.coords):
            if c.lo is not None:
                X[:, i] = np.maximum(X[:, i], c.lo)
            if c.hi is not None:
                X[:, i] = np.minimum(X[:, i], c.hi)
        return X


# --- oracle ------------------------------------------------------------------ #
def load_oracle(net=None, sigma_cut=0.6):
    """Return (Xfit, Xlow, y, yerr, rf, domain) for the BNS test."""
    net = net or os.environ.get("NET", DEFAULT_NET)
    X6, y, yerr, _ = load_ile_net(net, sigma_cut=sigma_cut, return_errors=True)
    m1, m2, s1z, s2z, l1, l2 = X6.T
    mc, dmc = mc_delta_from_m1m2(m1, m2)
    Xlow = np.column_stack([mc, dmc, s1z, s2z, l1, l2])
    Xfit = np.asarray(to_fit_coordinates(Xlow, LOW, BNS_FIT_COORDS))
    ok = np.all(np.isfinite(Xfit), axis=1) & np.isfinite(y) & np.isfinite(yerr)
    Xfit, Xlow, y, yerr = Xfit[ok], Xlow[ok], y[ok], yerr[ok]
    rf = RFBaseline().fit(Xfit, y, y_errors=yerr)
    coords = [Coord(n) for n in FIT]
    coords[FIT.index("delta_mc")].lo = 0.0       # equal-mass boundary
    coords[FIT.index("LambdaTilde")].lo = 0.0    # tidal floor
    dom = Domain(coords, loci=[("equal_mass",
                 lambda X: np.abs(np.atleast_2d(X)[:, FIT.index("delta_mc")]))])
    return Xfit, Xlow, y, yerr, rf, dom


def peak_metric(Xfit, y, depth=20.0):
    band = y > y.max() - depth
    Xb, yb = Xfit[band], y[band]
    w = np.exp(yb - yb.max()); w /= w.sum()
    C = np.atleast_2d(np.cov(Xb.T, aweights=w))
    C = 0.5 * (C + C.T) + 1e-9 * np.eye(C.shape[0]) * np.trace(C) / C.shape[0]
    return Xfit[np.argmax(y)], C, np.linalg.cholesky(C)


def backbone(Xfit, y, yerr, depth=20.0, cap=9000, seed=0):
    """On-support high-lnL backbone (the points that inform lnL, peak-outward)."""
    idx = np.where(y > y.max() - depth)[0]
    if len(idx) > cap:
        idx = np.random.default_rng(seed).choice(idx, cap, replace=False)
    return idx


def fit_gp(X, y, yerr=None, n_inducing=600, n_opt_steps=200):
    from RIFT.interpolators.jax_gp import get_interpolator
    m = get_interpolator("quadgp")(gp_method="svgp", n_inducing=n_inducing,
                                   n_opt_steps=n_opt_steps)
    return m.fit(X, y, y_errors=yerr)
