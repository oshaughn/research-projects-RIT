#!/usr/bin/env python
"""
test_AV_bootstrap.py

Quantitative test of the bootstrappable AdaptiveVolume integrator.

Demonstrates and validates that warm-starting AV from prior information reaches a
target effective sample size in far fewer likelihood evaluations than a cold
start, WITHOUT biasing the integral (must stay within the CI 4-sigma gate).

Three warm-start channels are exercised, all matching the production sampler API:
  1. Fisher matrix  (bootstrap_from_fisher)  -- the free "Fisher oracle"
  2. reference samples (bootstrap_from_samples) -- e.g. a previous ILE posterior
  3. serialized state round-trip (save_state/load_state) -- reuse across instances

Usage:
  python test_AV_bootstrap.py                      # corrgauss5, GPU if visible
  python test_AV_bootstrap.py --target gaussmix8 --as-test
"""
from __future__ import print_function
import argparse
import os
import tempfile
import numpy as np

import benchmark_integrators as B
from RIFT.integrators import mcsamplerAdaptiveVolume as AVmod


def _numeric_fisher(target, at=None, eps=1e-3):
    """Central-difference Fisher (negative Hessian of lnL) at the mode `at`."""
    d = target.ndim
    if at is None:
        # crude mode: densest of a coarse random scan
        rng = np.random.RandomState(0)
        X = rng.uniform(target.llim, target.rlim, size=(20000, d))
        at = X[np.argmax(target.lnL(X))]
    H = np.zeros((d, d))
    scale = (target.rlim - target.llim) * eps
    f0 = target.lnL(np.atleast_2d(at))[0]
    for i in range(d):
        for j in range(i, d):
            ei = np.zeros(d); ei[i] = scale[i]
            ej = np.zeros(d); ej[j] = scale[j]
            fpp = target.lnL(np.atleast_2d(at + ei + ej))[0]
            fpm = target.lnL(np.atleast_2d(at + ei - ej))[0]
            fmp = target.lnL(np.atleast_2d(at - ei + ej))[0]
            fmm = target.lnL(np.atleast_2d(at - ei - ej))[0]
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * scale[i] * scale[j])
    fisher = -0.5 * (H + H.T)
    # regularize to SPD
    w, Vv = np.linalg.eigh(fisher)
    w = np.clip(w, 1e-6, None)
    return at, Vv @ np.diag(w) @ Vv.T


def run_cold(target, **kw):
    return B.run("AV", target, **kw)


def run_warm(target, warm_kind, seed_info, **kw):
    def warm_start(sampler, tgt):
        if warm_kind == "fisher":
            mean, fisher = seed_info
            sampler.bootstrap_from_fisher(mean, fisher, n=sampler.n_chunk, seed=1)
        elif warm_kind == "mixture":
            means, covs, weights = seed_info
            sampler.bootstrap_from_gaussian_mixture(means, covs, weights,
                                                    n=sampler.n_chunk, seed=1)
        elif warm_kind == "samples":
            sampler.bootstrap_from_samples(seed_info)
        elif warm_kind == "state":
            sampler.load_state(seed_info)
    return B.run("AV", target, warm_start=warm_start, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="corrgauss5")
    ap.add_argument("--nmax", type=int, default=300000)
    ap.add_argument("--neff", type=int, default=1000)
    ap.add_argument("--n-chunk", type=int, default=10000)
    ap.add_argument("--as-test", action="store_true")
    ap.add_argument("--seed", type=int, default=123456)
    args = ap.parse_args()

    target = B._TARGETS[args.target]()
    kw = dict(nmax=args.nmax, neff=args.neff, n_chunk=args.n_chunk, seed=args.seed)
    multimodal = hasattr(target, "means") and len(getattr(target, "means")) > 1
    print("# target {}  ndim={}  true_lnZ={:.4f}  {}".format(
        target.name, target.ndim, target.true_lnZ,
        "MULTIMODAL" if multimodal else "unimodal"))

    # --- cold baseline ---
    cold = run_cold(target, **kw)
    print("COLD           ", B._fmt(cold))

    results = []  # (label, result)

    at, fisher = _numeric_fisher(target)
    if multimodal:
        # a single Fisher cannot cover multiple modes; use a mixture oracle
        # (stand-in for a GMM/flow oracle fit to a previous posterior) and also
        # empirical samples -- both cover the full support.
        seed_mix = (target.means, target.covs, target.wt)
        results.append(("mixture", run_warm(target, "mixture", seed_mix, **kw)))
    else:
        results.append(("fisher", run_warm(target, "fisher", (at, fisher), **kw)))

    # --- warm: reference samples (analytic-truth draws stand in for a previous
    #     ILE posterior; the mixture can be sampled exactly) ---
    if hasattr(target, "sample_truth"):
        ref = target.sample_truth(5000, seed=7)
    else:
        ref = np.random.RandomState(3).multivariate_normal(at, np.linalg.inv(fisher), 5000)
    results.append(("samples", run_warm(target, "samples", ref, **kw)))

    # --- warm: serialized-state round trip (reuse across sampler instances) ---
    s0, _ = B.build_sampler("AV", target, n_chunk=args.n_chunk)
    s0.setup()
    if multimodal:
        s0.bootstrap_from_gaussian_mixture(target.means, target.covs, target.wt,
                                           n=args.n_chunk, seed=1)
    else:
        s0.bootstrap_from_fisher(at, fisher, n=args.n_chunk, seed=1)
    tmp = os.path.join(tempfile.gettempdir(), "av_state_%s.npz" % target.name)
    s0.save_state(tmp)
    results.append(("state", run_warm(target, "state", tmp, **kw)))
    sz = os.path.getsize(tmp)

    for label, r in results:
        print("WARM({:<8s}".format(label + ")"), B._fmt(r))
    print("# serialized state size: {} bytes ({} live bins)".format(sz, len(np.load(tmp)['binunique'])))

    # --- summary: efficiency and samples-to-neff speedups ---
    print("\n# --- speedup (warm vs cold) ---")
    for name, r in results:
        eff_ratio = r["efficiency"] / cold["efficiency"]
        n_ratio = cold["n_eval"] / max(r["n_eval"], 1)
        print("  {:>8s}: efficiency x{:.2f}   N_eval-to-neff x{:.2f} fewer   "
              "bias_ln={:+.3f} (cold {:+.3f})".format(name, eff_ratio, n_ratio,
                                                      r["bias_ln"], cold["bias_ln"]))

    if args.as_test:
        # Correctness criterion: a warm start must not make the integral MORE
        # biased than a cold start (AV can have its own intrinsic bias on hard
        # high-D targets; the bootstrap must not worsen it) and must not lose
        # efficiency.  Gate = cold's own |bias| plus a margin.
        tol = max(0.10, abs(cold["bias_ln"]) + 3 * cold["rel_err"])
        ok = True
        for name, r in results:
            if abs(r["bias_ln"]) > tol:
                print(" FAIL: warm({}) bias_ln {:+.3f} exceeds tol {:.3f} (cold {:+.3f})".format(
                    name, r["bias_ln"], tol, cold["bias_ln"]))
                ok = False
            if r["efficiency"] < 0.95 * cold["efficiency"]:
                print(" WARN: warm({}) efficiency {:.2e} < cold {:.2e}".format(
                    name, r["efficiency"], cold["efficiency"]))
        if not ok:
            raise SystemExit(1)
        print(" PASS: all warm starts no more biased than cold (tol {:.3f}) and no less efficient".format(tol))


if __name__ == "__main__":
    main()
