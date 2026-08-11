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
import json
import os
import subprocess
import numpy as np

import benchmark_integrators as B
import test_portfolio_adaptive_alloc as T


def _provenance():
    """Record WHICH code produced the numbers.  The installed venv RIFT is routinely stale, so the
    resolved module path is the only trustworthy statement of what was measured; the backend matters
    because clipping interacts with the pooled n_eff the sampler reports."""
    import RIFT.integrators.mcsamplerPortfolio as P
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(P.__file__))))
    def _git(*a):
        try:
            return subprocess.check_output(('git', '-C', src) + a, stderr=subprocess.DEVNULL
                                           ).decode().strip()
        except Exception:
            return None
    # Ask the sampler what it CHOSE, not whether cupy imports: cupy imports fine with no visible
    # device and RIFT then falls back to numpy, so "import cupy worked" would misreport the backend.
    backend = 'cupy' if getattr(P, 'cupy_ok', False) else 'numpy'
    return dict(module=os.path.abspath(P.__file__), backend=backend,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                git_sha=_git('rev-parse', 'HEAD'), git_branch=_git('rev-parse', '--abbrev-ref', 'HEAD'),
                git_describes=_git('log', '-1', '--format=%h %s'))


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
    ap.add_argument("--json", type=str, default=None,
                    help="persist per-seed rows + provenance here (the printed table is a summary "
                         "of this file, so downstream macros never re-type a number)")
    args = ap.parse_args()

    clips = [float(c) for c in args.clips.split(',')]
    targets = [("uncorrelated", B.CorrelatedGaussian(ndim=args.ndim, rho=0.0, narrow=0.1)),
               ("correlated", T.CompoundCorrelatedGaussian(ndim=args.ndim))]
    seeds = [1234 + 101 * i for i in range(args.seeds)]

    print("# weight-clip sweep: nmax={} n_chunk={} ndim={} seeds={}".format(
        args.nmax, args.n_chunk, args.ndim, seeds))
    print("# clip C=0 is OFF (unbiased reference).  bias = lnI - true_lnZ (mean +/- std over seeds)")
    cells = []
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
            # Per-seed rows travel with the summary: the standard error over seeds is what decides
            # whether a C-to-C difference is a measurement or noise, and n_engaged says whether the
            # clip did anything at all -- at production chunk sizes tau is loose enough that it may
            # never bind, and a flat n_eff then means "inactive", not "harmless".
            cells.append(dict(
                target=name, true_lnZ=float(tgt.true_lnZ), clip=float(c),
                n_chunk=int(args.n_chunk), nmax=int(args.nmax), ndim=int(args.ndim),
                seeds=list(map(int, seeds)), rows=rows,
                n_eff_mean=float(ne.mean()), n_eff_std=float(ne.std(ddof=1)) if len(ne) > 1 else 0.0,
                n_eff_sem=float(ne.std(ddof=1) / np.sqrt(len(ne))) if len(ne) > 1 else 0.0,
                bias_mean=float(bi.mean()), bias_std=float(bi.std(ddof=1)) if len(bi) > 1 else 0.0,
                bias_sem=float(bi.std(ddof=1) / np.sqrt(len(bi))) if len(bi) > 1 else 0.0,
                clip_frac_mean=float(cf), predicted_bias_mean=float(pb),
                n_engaged=int(sum(1 for r in rows if r["n_clipped"] > 0))))

    if args.json:
        json.dump(dict(provenance=_provenance(), cells=cells), open(args.json, 'w'), indent=2)
        print("\nwrote {}".format(args.json))


if __name__ == "__main__":
    main()
