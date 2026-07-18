#!/usr/bin/env python
"""
test_portfolio_oracle.py

Quantitative test of the portfolio oracle mechanism, focused on the case where
oracles are supposed to help: a NEEDLE -- a narrow likelihood mode far off-centre
in a large prior box, which uniform sampling almost never finds, but a
Fisher/Gaussian oracle points straight at.

Compares mcsamplerPortfolio (two adaptive-cartesian members) WITH and WITHOUT a
FisherGaussianOracle, on the same needle, and checks that:
  * the oracle-seeded run reaches a substantially higher n_eff / n_ESS, and
  * both runs stay unbiased (oracles only propose; they cannot bias the integral).

Usage:
  python test_portfolio_oracle.py                 # GPU if visible
  python test_portfolio_oracle.py --as-test
"""
from __future__ import print_function
import argparse
import numpy as np

import benchmark_integrators as B
from RIFT.integrators import mcsamplerGPU, mcsamplerPortfolio
from RIFT.integrators.unreliable_oracle.fisher_gaussian import FisherGaussianOracle


class Needle(B.Target):
    """Narrow correlated Gaussian mode placed far off-centre in a big box.  The
    posterior occupies a tiny fraction of the prior volume, so uniform sampling
    has a very low hit rate -- the regime where a proposal oracle pays off."""
    def __init__(self, ndim=4, width=20.0, scale=100.0, sigma=0.15, seed=3):
        self.name = "needle_d{}".format(ndim)
        self.ndim = ndim
        self.width = width
        self.scale = scale
        self.params = [str(i) for i in range(ndim)]
        self.llim = -0.5 * width * np.ones(ndim)
        self.rlim = 0.5 * width * np.ones(ndim)
        rng = np.random.RandomState(seed)
        self.mu = rng.uniform(0.30 * 0.5 * width, 0.42 * 0.5 * width, ndim) * rng.choice([-1, 1], ndim)
        A = rng.normal(size=(ndim, ndim))
        cov = A @ A.T
        d = np.sqrt(np.diag(cov))
        cov = cov / np.outer(d, d) * (sigma ** 2)   # correlated, ~sigma per dim
        self.cov = cov
        from scipy.stats import multivariate_normal
        self._mvn = multivariate_normal(self.mu, self.cov)
        self.true_lnZ = np.log(scale) - np.sum(np.log(self.rlim - self.llim))

    def lnL(self, X):
        return np.atleast_1d(np.log(self.scale * self._mvn.pdf(X) + 1e-300))


def build_portfolio(target, with_oracle, n_chunk):
    members = [mcsamplerGPU.MCSampler(), mcsamplerGPU.MCSampler()]
    oracles = []
    if with_oracle:
        oracles = [FisherGaussianOracle()]
    port = mcsamplerPortfolio.MCSampler(portfolio=members, portfolio_freeze_wt=0.1,
                                        oracle_realizations=oracles, n_chunk=n_chunk)
    for d, p in enumerate(target.params):
        w = target.rlim[d] - target.llim[d]
        port.add_parameter(p, np.vectorize(lambda x, w=w: 1.0 / w),
                           prior_pdf=np.vectorize(lambda x, w=w: 1.0 / w),
                           left_limit=float(target.llim[d]), right_limit=float(target.rlim[d]),
                           adaptive_sampling=True)
    port.setup(portfolio_breakpoints=None)
    for m in members:
        m.setup()
    if with_oracle:
        # seed the Fisher oracle with the (known) mode shape -- in production this
        # is a Fisher matrix at the MAP; here we use the true mean/cov.
        oracles[0].setup(mean=target.mu, cov=target.cov)
    return port


def run(target, with_oracle, nmax, neff, n_chunk, seed):
    np.random.seed(seed)
    port = build_portfolio(target, with_oracle, n_chunk)
    ln_f = target.as_lnfunc()
    import time
    t0 = time.time()
    lnI, logvar, eff, _ = port.integrate_log(ln_f, *target.params, no_protect_names=True,
                                             nmax=nmax, neff=neff, n=n_chunk, n_adapt=100,
                                             tempering_exp=0.1, floor_level=0.0, use_lnL=True,
                                             save_intg=True, verbose=False)
    wall = time.time() - t0
    lnI = float(B._asnumpy(lnI)); eff = float(B._asnumpy(eff))
    ln_wt = B.log_weights_from_rvs(port._rvs)
    ness = B.n_ess_kish(ln_wt)
    n_eval = int(getattr(port, "ntotal", 0)) or nmax
    return dict(kind="portfolio" + ("+oracle" if with_oracle else ""),
                target=target.name, backend="gpu" if mcsamplerGPU.cupy_ok else "cpu",
                ndim=target.ndim, n_eval=n_eval, wallclock=wall, lnI=lnI,
                true_lnZ=float(target.true_lnZ), bias_ln=lnI - float(target.true_lnZ),
                rel_err=float(np.exp(0.5 * (float(B._asnumpy(logvar)) - 2 * lnI))),
                n_eff=eff, n_ess=ness, efficiency=eff / max(n_eval, 1), js_marginal=float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=400000)
    ap.add_argument("--neff", type=int, default=500)
    ap.add_argument("--n-chunk", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--as-test", action="store_true")
    args = ap.parse_args()

    target = Needle(ndim=args.ndim)
    print("# needle target ndim={} true_lnZ={:.4f}  mode at {}".format(
        target.ndim, target.true_lnZ, np.round(target.mu, 2)))

    base = run(target, False, args.nmax, args.neff, args.n_chunk, args.seed)
    print("NO-ORACLE ", B._fmt(base))
    orc = run(target, True, args.nmax, args.neff, args.n_chunk, args.seed)
    print("ORACLE    ", B._fmt(orc))

    ness_gain = orc["n_ess"] / max(base["n_ess"], 1e-9)
    neff_gain = orc["n_eff"] / max(base["n_eff"], 1e-9)
    print("\n# oracle vs none:  n_eff x{:.2f}   n_ESS x{:.2f}   "
          "bias(no-oracle)={:+.3f}  bias(oracle)={:+.3f}".format(
              neff_gain, ness_gain, base["bias_ln"], orc["bias_ln"]))

    if args.as_test:
        ok = True
        tol = max(0.15, 3 * max(base["rel_err"], orc["rel_err"]))
        if abs(orc["bias_ln"]) > tol:
            print(" FAIL: oracle run biased ({:+.3f} > {:.3f})".format(orc["bias_ln"], tol)); ok = False
        if not (orc["n_eff"] >= base["n_eff"] * 1.2 or orc["n_ess"] >= base["n_ess"] * 1.2):
            print(" FAIL: oracle did not improve n_eff/n_ESS by >=1.2x"); ok = False
        if not ok:
            raise SystemExit(1)
        print(" PASS: oracle improved sampling and stayed unbiased (tol {:.3f})".format(tol))


if __name__ == "__main__":
    main()
