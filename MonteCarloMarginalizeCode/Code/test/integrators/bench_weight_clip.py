#!/usr/bin/env python
"""
bench_weight_clip.py -- quantify the BIAS vs n_eff trade of portfolio weight clipping
(truncated importance sampling) against targets with an ANALYTIC true ln Z.

Weight clipping caps each importance weight at tau = C*sqrt(n)*mean(w) (Ionides 2008).  A single
enormous weight crushes pooled n_eff = (sum w)^2/sum w^2, so clipping can buy a large variance
reduction -- but it is a BIASED estimator (it discards the clipped mass), and the whole question is
whether the bias is small enough to be worth it.  Because these targets have a known true ln Z we
can measure BOTH sides directly, and the sampler also reports the removed-mass fraction, so the
predicted bias log1p(-frac) can be checked against the measured bias.

Sweeps C over several values (C=0 is clipping OFF, the unbiased reference) on the correlated and
uncorrelated Gaussians from test_portfolio_adaptive_alloc.py, over several seeds.

Usage:
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONPATH=<Code> python bench_weight_clip.py
  options: --seeds 3 --nmax 400000 --n-chunk 10000 --ndim 5
"""
from __future__ import print_function
import argparse
import numpy as np

import benchmark_integrators as B
import test_portfolio_adaptive_alloc as T


def run_clip(target, clip, n_chunk, nmax, seed, adaptive=False):
    np.random.seed(seed)
    port = T.build(target, ['AV', 'GMM'], n_chunk)
    lnI, _, eff, _ = port.integrate_log(
        T._host_lnfunc(target), *target.params, no_protect_names=True,
        nmax=nmax, neff=10 ** 9, n=n_chunk, n_adapt=100, tempering_exp=0.3,
        floor_level=0.0, use_lnL=True, save_intg=True, verbose=False,
        portfolio_adaptive_alloc=adaptive, portfolio_weight_clip=clip)
    lnI = float(B._asnumpy(lnI))
    # removed-mass fraction the sampler tracked (0 if nothing clipped)
    frac = 0.0
    if np.isfinite(port.portfolio_clip_log_removed) and np.isfinite(port.portfolio_clip_log_total):
        frac = float(np.exp(port.portfolio_clip_log_removed - port.portfolio_clip_log_total))
        frac = min(max(frac, 0.0), 1.0 - 1e-15)
    return dict(lnI=lnI, bias=lnI - float(target.true_lnZ), n_eff=float(B._asnumpy(eff)),
                clip_frac=frac, predicted_bias=float(np.log1p(-frac)) if frac > 0 else 0.0,
                n_clipped=int(port.portfolio_clip_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=5)
    ap.add_argument("--nmax", type=int, default=400000)
    ap.add_argument("--n-chunk", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--clips", type=str, default="0,0.5,1,2,5,20")
    args = ap.parse_args()

    clips = [float(c) for c in args.clips.split(',')]
    targets = [("uncorrelated", B.CorrelatedGaussian(ndim=args.ndim, rho=0.0, narrow=0.1)),
               ("correlated", T.CompoundCorrelatedGaussian(ndim=args.ndim))]
    seeds = [1234 + 101 * i for i in range(args.seeds)]

    print("# weight-clip sweep: nmax={} n_chunk={} ndim={} seeds={}".format(
        args.nmax, args.n_chunk, args.ndim, seeds))
    print("# clip C=0 is OFF (unbiased reference).  bias = lnI - true_lnZ (mean +/- std over seeds)")
    for name, tgt in targets:
        print("\n== {}  true_lnZ={:.4f} ==".format(name, tgt.true_lnZ))
        print("{:>6} {:>12} {:>18} {:>12} {:>12}".format(
            "C", "n_eff", "bias", "clip_frac", "pred_bias"))
        for c in clips:
            rows = [run_clip(tgt, c, args.n_chunk, args.nmax, s) for s in seeds]
            ne = np.array([r["n_eff"] for r in rows])
            bi = np.array([r["bias"] for r in rows])
            cf = np.mean([r["clip_frac"] for r in rows])
            pb = np.mean([r["predicted_bias"] for r in rows])
            print("{:>6.2f} {:>6.0f}+/-{:<5.0f} {:>+8.3f}+/-{:<7.3f} {:>12.2e} {:>+12.3f}".format(
                c, ne.mean(), ne.std(), bi.mean(), bi.std(), cf, pb))


if __name__ == "__main__":
    main()
