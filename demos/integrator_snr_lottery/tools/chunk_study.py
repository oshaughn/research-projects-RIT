#!/usr/bin/env python
"""
chunk_study.py -- is CHUNK SIZE a stability knob for high-SNR extrinsic integration?

HYPOTHESIS (reviewer): part of the high-SNR collapse is that the chunk is too small.  At high SNR the
posterior occupies a vanishing fraction of the prior volume, so a small chunk contains almost no
informative samples per adaptation step and the sampler adapts on noise.  A larger chunk may raise
the SNR at which extrinsic integration collapses.

COUNTERWEIGHT: GPU memory scales with chunk size.  Bigger chunks restrict which resources a job can
run on and would force per-job memory tuning that production does not normally do (held jobs, idle
capacity).  So the deliverable is not "bigger is better" but WHERE the trade sits.

WHY THIS HARNESS RATHER THAN A REAL EVENT
  * TRUTH IS KNOWN.  MixtureTarget exposes `true_lnZ`, so we measure real BIAS (lnI - true_lnZ), not
    scatter about an unknown answer.  The real-event study could only ever measure scatter, and that
    is what made it so easy to fool ourselves with small samples.
  * It is CPU-ONLY, so copies are cheap.  The real-event work was GPU-bound and ran 4-9 copies per
    cell; at that size a variance estimate is nearly worthless (we retracted a result for exactly
    this reason).  Here we can afford tens of copies per cell.

SNR LADDER.  Posterior width scales as 1/SNR, and MixtureTarget's `sigma_1d` IS the peak width on a
fixed box [-5,5]^d.  So we set

    sigma_1d = SIGMA_REF * (SNR_REF / SNR),     SIGMA_REF=0.7 at SNR_REF=20

i.e. SNR 20 -> sigma 0.7 (the gate's own default), SNR 160 -> sigma 0.0875.  The peak's volume
fraction falls like (sigma/10)^d, which is the mechanism we care about.

FAIR COMPARISON.  Total budget `nmax` is held FIXED across chunk sizes, so this is a same-cost
comparison.  Note the coupling that makes it interesting: n_steps = nmax/n_chunk, so a larger chunk
buys better per-step statistics at the price of FEWER adaptation steps.  That is the real trade.

Usage:
    export PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
    python chunk_study.py --copies 24 --jobs 32 --json results/chunk_study.json
"""
from __future__ import print_function
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "MonteCarloMarginalizeCode", "Code",
    "test", "expensive_before_merging", "integrators"))
sys.path.insert(0, GATE)
import shape_recovery as SR   # reuse the gate's targets, runner, metrics and verdicts

SIGMA_REF, SNR_REF = 0.7, 20.0


def sigma_for_snr(snr):
    """Peak width for an 'SNR': posterior width ~ 1/SNR, anchored at the gate's default."""
    return SIGMA_REF * (SNR_REF / float(snr))


def _one(job):
    (snr, n_chunk, ndim, ncomp, nmax, neff, seed, kind) = job
    # nmax is resolved by the caller: FIXED-BUDGET mode passes the same nmax to every chunk size
    # (same cost, but steps = nmax/n_chunk falls as the chunk grows), while FIXED-STEPS mode passes
    # nmax = n_chunk*steps (equal adaptation opportunities, but cost grows with the chunk).
    sigma = sigma_for_snr(snr)
    t0 = time.time()
    try:
        target = SR.MixtureTarget(ndim, ncomp, seed, sigma_1d=sigma)
        rec = SR.run_one(kind, target, nmax, neff, n_chunk=n_chunk, seed=seed)
        verdict = SR.evaluate(rec)
        status = verdict if isinstance(verdict, str) else verdict[0]
    except Exception as e:                      # never let one cell kill the sweep
        rec, status = {"error": str(e)[:200]}, "ERROR"
    return dict(snr=snr, sigma=sigma, n_chunk=n_chunk, ndim=ndim, ncomp=ncomp,
                nmax=nmax, seed=seed, kind=kind, status=status,
                n_steps=int(nmax // n_chunk),
                bias_ln=float(rec.get("bias_ln", np.nan)),
                n_eff=float(rec.get("n_eff", np.nan)),
                wall=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", default="20,40,80,160")
    ap.add_argument("--chunks", default="10000,40000,160000")
    ap.add_argument("--ndim", type=int, default=4)
    ap.add_argument("--ncomp", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=1000000, help="FIXED total budget (same cost)")
    ap.add_argument("--steps", type=int, default=None,
                    help="FIXED-STEPS mode: set nmax = n_chunk*steps per cell instead of --nmax. "
                         "Equal adaptation opportunities across chunk sizes, so it isolates whether "
                         "richer per-step statistics help; NOT a same-cost comparison.")
    ap.add_argument("--neff", type=int, default=3000)
    ap.add_argument("--copies", type=int, default=24)
    ap.add_argument("--kinds", default="AV,portfolio")
    ap.add_argument("--seed0", type=int, default=4000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    snrs = [float(x) for x in a.snrs.split(",")]
    chunks = [int(x) for x in a.chunks.split(",")]
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]

    jobs = []
    for kind in kinds:
        for snr in snrs:
            for nc in chunks:
                nmax_here = nc * a.steps if a.steps else a.nmax
                for c in range(a.copies):
                    jobs.append((snr, nc, a.ndim, a.ncomp, nmax_here, a.neff,
                                 a.seed0 + c, kind))
    print("# chunk study: {} kinds x {} SNR x {} chunks x {} copies = {} runs"
          .format(len(kinds), len(snrs), len(chunks), a.copies, len(jobs)))
    if a.steps:
        print("# FIXED-STEPS mode: steps={} so nmax = n_chunk*steps (cost GROWS with chunk); "
              "isolates per-step statistics".format(a.steps))
    else:
        print("# FIXED-BUDGET mode: nmax={} for every chunk (same cost; steps = nmax/n_chunk "
              "FALLS as the chunk grows)".format(a.nmax))
    print("# ndim={} ncomp={} neff={}".format(a.ndim, a.ncomp, a.neff))
    print("# sigma ladder: " + ", ".join("SNR%g->%.4f" % (s, sigma_for_snr(s)) for s in snrs))
    sys.stdout.flush()

    t0 = time.time()
    pool = Pool(a.jobs)
    recs = pool.map(_one, jobs)
    pool.close(); pool.join()
    print("# done in {:.1f} min".format((time.time() - t0) / 60.0))

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(recs, open(a.json, "w"), indent=1)
        print("# wrote", a.json)

    summarize(recs)


def summarize(recs):
    """Per cell: collapse fraction (status != PASS) and median |bias| among PASSing copies."""
    import collections
    cells = collections.OrderedDict()
    for r in recs:
        cells.setdefault((r["kind"], r["snr"], r["n_chunk"]), []).append(r)
    print("\n%-10s %6s %9s %7s %8s %12s %10s" %
          ("kind", "SNR", "n_chunk", "steps", "collapse", "med|bias|", "med n_eff"))
    for (kind, snr, nc), rs in cells.items():
        bad = [r for r in rs if r["status"] != "PASS"]
        ok = [r for r in rs if r["status"] == "PASS"]
        mb = np.median([abs(r["bias_ln"]) for r in ok]) if ok else float("nan")
        mn = np.median([r["n_eff"] for r in rs if np.isfinite(r["n_eff"])]) if rs else float("nan")
        print("%-10s %6g %9d %7d %7.0f%% %12.4f %10.0f" %
              (kind, snr, nc, rs[0]["n_steps"], 100.0 * len(bad) / len(rs), mb, mn))


if __name__ == "__main__":
    main()
