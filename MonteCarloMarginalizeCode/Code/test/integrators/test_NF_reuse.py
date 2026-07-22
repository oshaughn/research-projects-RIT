#!/usr/bin/env python
"""
test_NF_reuse.py

Demonstrates + validates the normalizing-flow storage/reuse framework.

NF training is slow and only pays off if the trained flow is re-used across the
many ILE instances that share similar posterior structure.  This test:
  1. trains a flow once on a target and saves it (save_flow);
  2. reuses it in FRESH samplers via load_flow, either
       - pure reuse (n_adapt=0: sample straight from the trained flow), or
       - a few 'polish' epochs (small n_adapt) to adapt to the instance;
  3. checks the reused runs match the cold-trained integral (unbiased) while
     spending far less wallclock on training.

Usage:
  python test_NF_reuse.py --as-test
Run with thread caps, e.g. OMP_NUM_THREADS=2, to avoid torch oversubscription.
"""
from __future__ import print_function
import argparse
import os
import tempfile
import time
import numpy as np

import benchmark_integrators as B
from RIFT.integrators import mcsamplerNFlow


def _build(target, n_chunk=10000):
    s = mcsamplerNFlow.MCSampler(n_chunk=n_chunk)
    for d, p in enumerate(target.params):
        w = target.rlim[d] - target.llim[d]
        s.add_parameter(p, np.vectorize(lambda x, w=w: 1.0 / w),
                        prior_pdf=np.vectorize(lambda x, w=w: 1.0 / w),
                        left_limit=float(target.llim[d]), right_limit=float(target.rlim[d]),
                        adaptive_sampling=True)
    return s


def _integrate(s, target, nmax, neff, n_chunk, n_adapt, load_path=None):
    if load_path is not None:
        s.load_flow(load_path)
    t0 = time.time()
    lnI, logvar, eff, _ = s.integrate_log(target.as_lnfunc(), *target.params,
                                          no_protect_names=True, nmax=nmax, neff=neff,
                                          n=n_chunk, n_adapt=n_adapt, tempering_exp=1.0,
                                          verbose=False)
    wall = time.time() - t0
    lnI = float(B._asnumpy(lnI)); eff = float(B._asnumpy(eff))
    ln_wt = B.log_weights_from_rvs(s._rvs)
    return dict(bias_ln=lnI - float(target.true_lnZ), n_eff=eff,
                n_ess=B.n_ess_kish(ln_wt), n_eval=int(getattr(s, "ntotal", 0)) or nmax,
                wall=wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=3)
    ap.add_argument("--n-chunk", type=int, default=10000)
    ap.add_argument("--train-nmax", type=int, default=80000)
    ap.add_argument("--nmax", type=int, default=40000)
    ap.add_argument("--neff", type=int, default=300)
    ap.add_argument("--as-test", action="store_true")
    args = ap.parse_args()

    target = B.CorrelatedGaussian(ndim=args.ndim)
    print("# NF reuse on {}  true_lnZ={:.4f}".format(target.name, target.true_lnZ))

    # --- phase 1: train once and save (the expensive, one-time cost) ---
    trainer = _build(target, args.n_chunk)
    r_train = _integrate(trainer, target, args.train_nmax, args.neff, args.n_chunk, n_adapt=8)
    path = os.path.join(tempfile.gettempdir(), "nf_flow_%s.pt" % target.name)
    trainer.save_flow(path)
    sz = os.path.getsize(path)
    print("TRAIN+SAVE  bias={bias_ln:+.3f} neff={n_eff:.1f} nESS={n_ess:.1f} "
          "t={wall:.1f}s  -> saved {sz} bytes".format(sz=sz, **r_train))

    # --- phase 2a: cold (fresh sampler, train from scratch) ---
    cold = _integrate(_build(target, args.n_chunk), target, args.nmax, args.neff, args.n_chunk, n_adapt=8)
    print("COLD        bias={bias_ln:+.3f} neff={n_eff:.1f} nESS={n_ess:.1f} t={wall:.1f}s".format(**cold))

    # --- phase 2b: warm reuse (load flow, NO training) ---
    reuse = _integrate(_build(target, args.n_chunk), target, args.nmax, args.neff, args.n_chunk,
                       n_adapt=0, load_path=path)
    print("WARM(reuse) bias={bias_ln:+.3f} neff={n_eff:.1f} nESS={n_ess:.1f} t={wall:.1f}s".format(**reuse))

    # --- phase 2c: warm + polish (load flow, a couple of epochs) ---
    polish = _integrate(_build(target, args.n_chunk), target, args.nmax, args.neff, args.n_chunk,
                        n_adapt=2, load_path=path)
    print("WARM(polish)bias={bias_ln:+.3f} neff={n_eff:.1f} nESS={n_ess:.1f} t={wall:.1f}s".format(**polish))

    print("\n# training-cost saving: cold {:.1f}s vs warm-reuse {:.1f}s  (x{:.1f} faster)".format(
        cold["wall"], reuse["wall"], cold["wall"] / max(reuse["wall"], 1e-3)))

    if args.as_test:
        ok = True
        tol = max(0.20, 4 * abs(cold["bias_ln"]) + 0.1)
        for name, r in [("reuse", reuse), ("polish", polish)]:
            if abs(r["bias_ln"]) > tol:
                print(" FAIL: warm({}) biased {:+.3f} > {:.3f}".format(name, r["bias_ln"], tol)); ok = False
        # reuse must be materially cheaper than cold (no training loop)
        if not (reuse["wall"] < 0.6 * cold["wall"]):
            print(" FAIL: warm reuse not faster than cold ({:.1f}s vs {:.1f}s)".format(reuse["wall"], cold["wall"])); ok = False
        # reused flow must actually sample the mode (not degenerate)
        if not (reuse["n_ess"] > 20):
            print(" FAIL: reused flow n_ESS too low ({:.1f})".format(reuse["n_ess"])); ok = False
        if not ok:
            raise SystemExit(1)
        print(" PASS: flow reuse is unbiased and skips training (tol {:.3f})".format(tol))


if __name__ == "__main__":
    main()
