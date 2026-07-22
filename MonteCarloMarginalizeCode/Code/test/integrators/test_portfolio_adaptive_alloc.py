#!/usr/bin/env python
"""
test_portfolio_adaptive_alloc.py

Validates mcsamplerPortfolio's ADAPTIVE-PROBE draw allocation: the portfolio should
automatically concentrate its draw budget on whichever member is actually winning, so it
TRACKS the best single member on both a weakly- and a strongly-correlated target -- and
therefore BEATS standalone AV on the correlated one (where a full-covariance GMM wraps the
degeneracy that AV's axis-aligned bins cannot).

Two targets, both scaled Gaussians with known true_lnZ:
  * UNCORRELATED : axis-aligned anisotropic Gaussian.
  * CORRELATED   : a COMPOUND-SYMMETRIC Gaussian (every coordinate pair correlated) -- its narrow
                   eigen-directions are OFF the coordinate axes, so AV's axis-aligned bins cannot
                   wrap the tilted ridge, while a single full-covariance GMM component captures it.

For each target we run standalone AV, standalone GMM, and the AV+GMM portfolio to a FIXED sample
budget and compare n_eff (efficiency) and bias = lnI - true_lnZ (correctness).  The GMM member is
broad-seeded (a wide peak-covering proposal) so the test deterministically exercises the ALLOCATION
policy given a member that CAN model the correlation, rather than gambling on cold GMM finding a
thin ridge; AV starts cold (axis-aligned bins cannot wrap the correlation, seed or not).  Observed:
  * On the CORRELATED target GMM's n_eff is several-fold AV's, and adaptive allocation concentrates
    on GMM, so the portfolio BEATS standalone AV (the whole point of a portfolio on a correlated
    problem).  On the uncorrelated target the portfolio matches/beats the better single member too.
  * A cold VARAHA/AV only CONTRACTS, so it under-covers the Gaussian tails and is BIASED LOW on
    both targets; the portfolio stays UNBIASED because the covering GMM member enters q_mix -- a
    second reason to prefer the portfolio over AV alone here.  (This is the opposite regime from a
    warm, cover-frac'd AV on a real ILE likelihood, where AV is the unbiased workhorse; the point
    tested here is that adaptive allocation follows whichever member is actually winning.)

Usage:
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
     python test_portfolio_adaptive_alloc.py --as-test
"""
from __future__ import print_function
import argparse
import numpy as np
from scipy.stats import multivariate_normal

import benchmark_integrators as B
from RIFT.integrators import mcsamplerAdaptiveVolume as AVmod
from RIFT.integrators import mcsamplerEnsemble as Emod
from RIFT.integrators import mcsamplerPortfolio as Pmod


class CompoundCorrelatedGaussian(B.CorrelatedGaussian):
    """A COMPOUND-SYMMETRIC correlated Gaussian: every pair of coordinates is correlated (cov_ij=c
    for i!=j).  Its eigen-structure is one wide direction along (1,1,...,1) and (ndim-1) narrow
    directions that are OFF the coordinate axes -- exactly the correlated/degenerate geometry that
    a full-covariance GMM captures in one component but that AV's axis-aligned bins cannot wrap
    (they must staircase the tilted ridge, wasting most of their bounding box).  `base` scales the
    whole covariance so the wide direction is comfortably contained in the box AND the narrow
    directions stay findable cold (std ~0.3, not a needle)."""
    def __init__(self, ndim=5, c=0.85, base=0.5, width=10.0, scale=100.0, seed=7):
        super(CompoundCorrelatedGaussian, self).__init__(ndim=ndim, width=width, scale=scale,
                                                         seed=seed, rho=0.0)
        self.cov = base * ((1.0 - c) * np.eye(ndim) + c * np.ones((ndim, ndim)))
        self.mu = np.zeros(ndim)  # centered -> contained in the box
        self._mvn = multivariate_normal(self.mu, self.cov)
        self.name = "compound_d{}".format(ndim)
        self.true_lnZ = np.log(scale) - np.sum(np.log(self.rlim - self.llim))


def _host_lnfunc(target):
    """cupy-tolerant wrapper: AV's selfish self-update evaluates on device-native draws, but the
    synthetic integrand is host/numpy -- move any device args to the host first."""
    base = target.as_lnfunc()
    def ln_f(*cols):
        cols = [Emod.identity_convert(c) for c in cols]
        return base(*cols)
    return ln_f


def _seed_gmm_broad(gmm, target, broad=3.0, n=8000, seed=7):
    """Give the GMM member a BROAD but peak-covering full-covariance proposal (fit to a wide cloud
    N(mu, broad^2 cov) around the mode).  This removes the cold-start LOTTERY -- cold GMM only
    sometimes finds a thin correlated ridge from a uniform start -- so the test deterministically
    exercises the ALLOCATION policy given a member that *can* model the correlation (AV cannot,
    seed or not).  The member still adapts/tightens during the run."""
    rng = np.random.RandomState(seed)
    cloud = rng.multivariate_normal(target.mu, broad ** 2 * np.atleast_2d(target.cov), n)
    cloud = np.clip(cloud, target.llim + 1e-3, target.rlim - 1e-3)
    gmm.update_sampling_prior(np.zeros(len(cloud)), 2 * len(cloud),
                              external_rvs={p: cloud[:, i] for i, p in enumerate(gmm.params_ordered)},
                              log_scale_weights=True)


def build(target, members, n_chunk):
    """Build a portfolio of the requested members ('AV', 'GMM' or both).  GMM members are seeded
    with a broad peak-covering proposal (see _seed_gmm_broad); AV members start cold."""
    objs, gmms = [], []
    for name in members:
        if name == 'AV':
            objs.append(AVmod.MCSampler(n_chunk=n_chunk))
        else:
            g = Emod.MCSampler(); objs.append(g); gmms.append(g)
    port = Pmod.MCSampler(portfolio=objs, n_chunk=n_chunk)
    for d, p in enumerate(target.params):
        w = target.rlim[d] - target.llim[d]
        port.add_parameter(p, np.vectorize(lambda x, w=w: 1.0 / w),
                           prior_pdf=np.vectorize(lambda x, w=w: 1.0 / w),
                           left_limit=float(target.llim[d]), right_limit=float(target.rlim[d]),
                           adaptive_sampling=True)
    # GMM: single full-covariance component (captures a correlated ridge in one component)
    port.setup(portfolio_breakpoints=None, n_comp=1, correlate_all_dims=True, n=n_chunk)
    for g in gmms:
        _seed_gmm_broad(g, target)
    return port


def run(target, members, n_chunk, nmax, seed=1234):
    np.random.seed(seed)
    port = build(target, members, n_chunk)
    lnI, _, eff, _ = port.integrate_log(
        _host_lnfunc(target), *target.params, no_protect_names=True,
        nmax=nmax, neff=10**9, n=n_chunk, n_adapt=100, tempering_exp=0.3,
        floor_level=0.0, use_lnL=True, save_intg=True, verbose=False,
        portfolio_adaptive_alloc=True)   # opt-in: this test exercises adaptive allocation
    lnI = float(B._asnumpy(lnI))
    # use the integrator's OWN reported effective-sample count (the q_mix-based pooled eff_samp),
    # the quantity it actually targets -- comparable across standalone AV/GMM and the portfolio.
    n_eff = float(B._asnumpy(eff))
    return dict(lnI=lnI, bias=lnI - float(target.true_lnZ), n_eff=n_eff,
                wts=np.round(np.array(port.portfolio_weights), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=5)
    ap.add_argument("--nmax", type=int, default=400000)
    ap.add_argument("--n-chunk", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--as-test", action="store_true")
    args = ap.parse_args()

    uncorr = B.CorrelatedGaussian(ndim=args.ndim, rho=0.0, narrow=0.1)
    corr = CompoundCorrelatedGaussian(ndim=args.ndim)
    kw = dict(n_chunk=args.n_chunk, nmax=args.nmax, seed=args.seed)

    print("# fixed budget nmax={} n_chunk={} ndim={}\n".format(args.nmax, args.n_chunk, args.ndim))
    rows = {}
    for label, tgt in [("UNCORRELATED (axis-aligned)", uncorr), ("CORRELATED (compound-symmetric)", corr)]:
        print("== {}   true_lnZ={:.3f} ==".format(label, tgt.true_lnZ))
        av = run(tgt, ['AV'], **kw)
        gm = run(tgt, ['GMM'], **kw)
        pf = run(tgt, ['AV', 'GMM'], **kw)
        rows[label] = (av, gm, pf)
        for nm, r in [("AV   ", av), ("GMM  ", gm), ("PORT ", pf)]:
            extra = "  final wts(AV,GMM)={}".format(r["wts"]) if nm == "PORT " else ""
            print("  {}: n_eff={:9.1f}   bias={:+.3f}{}".format(nm, r["n_eff"], r["bias"], extra))
        print()

    if args.as_test:
        # Only the ROBUST claims are gated (cold GMM's absolute n_eff on the correlated target is
        # stochastic run-to-run; the portfolio also legitimately carries the biased AV member, so it
        # is not always >= GMM-alone).  The durable, seed-insensitive facts are: the portfolio is
        # UNBIASED, and on the CORRELATED target adaptive allocation concentrates on the full-cov GMM
        # and the portfolio clearly BEATS standalone AV (the correlated-problem win).
        ok = True
        for label, (av, gm, pf) in rows.items():
            for nm, r in [("GMM", gm), ("PORT", pf)]:
                if abs(r["bias"]) > 0.2:
                    print(" FAIL[{}]: {} biased ({:+.3f})".format(label, nm, r["bias"])); ok = False
        av_c, gm_c, pf_c = rows["CORRELATED (compound-symmetric)"]
        if not (pf_c["n_eff"] > 1.5 * av_c["n_eff"]):
            print(" FAIL: portfolio did not clearly beat standalone AV on the correlated target "
                  "(PORT {:.1f} vs AV {:.1f})".format(pf_c["n_eff"], av_c["n_eff"])); ok = False
        if not (pf_c["wts"][1] > 0.6):
            print(" FAIL: adaptive allocation did not concentrate on GMM on the correlated target "
                  "(GMM weight {:.2f})".format(pf_c["wts"][1])); ok = False
        if not ok:
            raise SystemExit(1)
        print("\n PASS: portfolio unbiased on both targets, and on the correlated target adaptive "
              "allocation concentrates on the full-cov GMM so the portfolio beats standalone AV "
              "({:.0f} vs {:.0f} n_eff).".format(pf_c["n_eff"], av_c["n_eff"]))


if __name__ == "__main__":
    main()
