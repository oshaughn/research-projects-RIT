"""
Validate the pure-JAX coordinate transforms against the legacy NumPy path
(``lalsimutils.convert_waveform_coordinates``), and check differentiability.

Run: ``python -m RIFT.interpolators.jax_gp.test_coordinates`` (needs RIFT on the
path, i.e. PYTHONPATH=<.../Code>).
"""
from __future__ import annotations

import numpy as np
import jax

from . import coordinates as C

BNS = ("mu1", "mu2", "delta_mc", "LambdaTilde", "DeltaLambdaTilde")


def _random_bns(n, seed=0):
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(1.2, 2.0, n)
    m2 = rng.uniform(1.0, 1.0, n) * 0  # placeholder
    m2 = np.minimum(m1, rng.uniform(1.0, 2.0, n))   # ensure m2 <= m1
    m2 = np.where(m2 > m1, m1, m2)
    s1z = rng.uniform(-0.05, 0.05, n)
    s2z = rng.uniform(-0.05, 0.05, n)
    l1 = rng.uniform(0.0, 3000.0, n)
    l2 = rng.uniform(0.0, 3000.0, n)
    return np.column_stack([m1, m2, s1z, s2z, l1, l2])


def _legacy(Xphys):
    import RIFT.lalsimutils as lsu
    m1, m2, s1z, s2z, l1, l2 = Xphys.T
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    dmc = (m1 - m2) / (m1 + m2)
    xin = np.column_stack([mc, dmc, s1z, s2z, l1, l2])
    return lsu.convert_waveform_coordinates(
        xin, coord_names=list(BNS),
        low_level_coord_names=["mc", "delta_mc", "s1z", "s2z", "lambda1", "lambda2"])


def test_matches_legacy():
    Xphys = _random_bns(200, seed=1)
    low = ["m1", "m2", "s1z", "s2z", "lambda1", "lambda2"]
    tf = jax.vmap(C.make_transform(low, BNS))
    mine = np.asarray(tf(Xphys))
    ref = _legacy(Xphys)
    # column-wise relative agreement (these span very different scales)
    for j, name in enumerate(BNS):
        scale = np.maximum(np.abs(ref[:, j]).max(), 1e-8)
        rel = np.abs(mine[:, j] - ref[:, j]) / scale
        assert rel.max() < 1e-5, "{}: max rel err {:.2e}".format(name, rel.max())
    return mine, ref


def test_differentiable():
    low = ["m1", "m2", "s1z", "s2z", "lambda1", "lambda2"]
    tf = C.make_transform(low, BNS)
    theta = np.array([1.5, 1.3, 0.01, -0.01, 400.0, 600.0])
    # jacobian of fit coords wrt physical params must be finite
    J = np.asarray(jax.jacobian(tf)(theta))
    assert J.shape == (len(BNS), len(low))
    assert np.all(np.isfinite(J)), "non-finite jacobian"
    return J


if __name__ == "__main__":
    mine, ref = test_matches_legacy()
    print("legacy-vs-JAX coordinate agreement OK (200 random BNS points)")
    for j, name in enumerate(BNS):
        print("  {:18s} JAX[0]={:12.5g}  legacy[0]={:12.5g}".format(
            name, mine[0, j], ref[0, j]))
    J = test_differentiable()
    print("jacobian d(fit)/d(physical) finite, shape", J.shape)
