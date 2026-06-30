#!/usr/bin/env python3
"""Unit test of fisher_nuts_sample_phimarg against an ANALYTIC 4-D target.

Needs jax + numpyro + scipy only (NO lal): samplers.py is loaded standalone
(its top-level imports are numpy/jax) and given a mock likelihood exposing
the wrapper interface (log_likelihood, _scalar, fisher, value_and_grad).

Target: TWO narrow Gaussian sky modes (mimicking discrete time-delay-ring
solutions at HIGH SNR, sigma_sky = 0.004 rad) with mild psi/incl structure.
The analytic evidence is known, so the test checks:
  * both modes found (multi-start MAP + clustering);
  * whitened NUTS samples both modes without step-size collapse;
  * mixture-IS logZ matches the analytic value (|d| < 0.15);
  * post_weight reproduces the known mode-mass ratio.

Run:  python test_nuts_phimarg.py     (exit 0 on PASS)
"""
import importlib.util
import os
import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLERS = os.path.join(_HERE, "..", "..", "RIFT", "likelihood", "jax_ile",
                        "samplers.py")
spec = importlib.util.spec_from_file_location("samplers", SAMPLERS)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# ---------------------------------------------------------------------------
# Analytic 4-D target: sum of two Gaussian bumps in (ra, dec, psi, incl)
# ---------------------------------------------------------------------------
MU1 = jnp.array([1.20, -0.40, 0.70, 0.90])
MU2 = jnp.array([2.60,  0.35, 1.10, 1.60])
SIG = jnp.array([0.004, 0.004, 0.05, 0.05])   # narrow sky = high SNR
AMP1, AMP2 = 200.0, 197.0                     # lnL peak heights (mode 2 weaker)


def _lnL_one(th):
    g1 = AMP1 - 0.5 * jnp.sum(((th - MU1) / SIG) ** 2)
    g2 = AMP2 - 0.5 * jnp.sum(((th - MU2) / SIG) ** 2)
    m = jnp.maximum(g1, g2)
    return m + jnp.log(jnp.exp(g1 - m) + jnp.exp(g2 - m))


class MockLike:
    def __init__(self):
        self._scalar = _lnL_one
        self._b = jax.jit(jax.vmap(_lnL_one))
        self._vg = jax.jit(jax.value_and_grad(_lnL_one))
        self._h = jax.jit(jax.hessian(_lnL_one))

    def log_likelihood(self, ra, dec, psi, incl):
        th = jnp.stack([jnp.asarray(ra), jnp.asarray(dec),
                        jnp.asarray(psi), jnp.asarray(incl)], axis=-1)
        return self._b(jnp.atleast_2d(th))

    def value_and_grad(self, th):
        v, g = self._vg(jnp.asarray(th, dtype=jnp.float64))
        return float(v), np.asarray(g)

    def fisher(self, th):
        return -np.asarray(self._h(jnp.asarray(th, dtype=jnp.float64)))


# Analytic evidence: Z = E_prior[L], prior = cos(dec)/2 * sin(incl)/2
#                    / (2pi * pi), Gaussian bumps narrow vs prior variation.
def _analytic_logZ():
    det = float(np.prod(np.asarray(SIG)))
    vol = (2 * np.pi) ** 2 * det                  # 4-D Gaussian integral
    pri = []
    for MU in (np.asarray(MU1), np.asarray(MU2)):
        pri.append(np.cos(MU[1]) / 2 * np.sin(MU[3]) / 2
                   / (2 * np.pi) / np.pi)
    Z = vol * (np.exp(AMP1) * pri[0] + np.exp(AMP2) * pri[1])
    return np.log(Z)


def main():
    LOGZ_TRUE = _analytic_logZ()
    m1 = np.exp(AMP1) * np.cos(-0.40) * np.sin(0.90)
    m2 = np.exp(AMP2) * np.cos(0.35) * np.sin(1.60)
    MASS2_TRUE = m2 / (m1 + m2)
    print("analytic logZ = %.4f   true mass(mode2) = %.4f"
          % (LOGZ_TRUE, MASS2_TRUE))

    like = MockLike()
    res = S.fisher_nuts_sample_phimarg(
        like, num_warmup=150, num_samples=400, n_starts=10, n_modes=3,
        n_prior_pilot=20000, n_is=20000, seed=7, verbose=True)

    th, pw = res["theta"], res["post_weight"]
    print("\nmodes found:")
    for m, l in zip(res["modes"], res["mode_lnL"]):
        print("   (%.4f, %.4f, %.3f, %.3f)  lnL=%.2f" % (*m, l))

    found1 = any(np.linalg.norm(m[:2] - np.array([1.20, -0.40])) < 0.02
                 for m in res["modes"])
    found2 = any(np.linalg.norm(m[:2] - np.array([2.60, 0.35])) < 0.02
                 for m in res["modes"])
    dlogZ = abs(res["logZ"] - LOGZ_TRUE)
    d2 = np.linalg.norm(th[:, :2] - np.array([2.60, 0.35]), axis=1)
    d1 = np.linalg.norm(th[:, :2] - np.array([1.20, -0.40]), axis=1)
    mass2 = float(pw[d2 < d1].sum())

    print("\nlogZ = %.4f (true %.4f, |d|=%.3f)   neff=%.1f"
          % (res["logZ"], LOGZ_TRUE, dlogZ, res["neff"]))
    print("mass(mode2) = %.4f (true %.4f)" % (mass2, MASS2_TRUE))

    ok = (found1 and found2 and dlogZ < 0.15 and res["neff"] > 50
          and abs(mass2 - MASS2_TRUE) < 0.1
          and np.isclose(pw.sum(), 1.0))
    print("\nTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
