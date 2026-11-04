#!/usr/bin/env python
"""escaped_mass_study.py -- ROC / sensitivity study for the portfolio's warm-start
SUPPORT-MISMATCH detector (mcsamplerPortfolio.support_diagnostics).

QUESTION.  A RIFT extrinsic point is often warm-started from a proposal built at a DIFFERENT
point (a neighbouring intrinsic grid point, a stale breadcrumb, a recovered posterior).  If the
seed is misplaced, an AV/VARAHA member's live volume -- whose density is EXACTLY ZERO outside it
-- can exclude the true peak.  n_eff and the Pareto k-hat cannot see this: both are functions
only of the weights actually drawn (k-hat has been measured at 0.435 on a -1949-nat run).  The
proposed detector is

    escaped_mass[m] = sum_{i : q_m(x_i)==0} w_i / sum_i w_i

-- the fraction of total posterior weight carried by samples member m could not have drawn -- and
its cheaper comparator, weight_share[m] (fraction of weight from samples m DREW).

METHOD.  Truth-known testbed (shape_recovery.MixtureTarget).  Warm-start from the DISPLACED
target's own truth pool while integrating the TRUE (offset=0) target; sweep the displacement.
Three portfolio arms, because the answer depends entirely on what ELSE is in the mixture:
    avgmm_cold : [AV warm-seeded, GMM COLD]        -- an independent broad member
    avgmm_warm : [AV warm-seeded, GMM warm-seeded from the SAME displaced cloud] -- partly blind
    avav       : [AV warm-seeded, AV warm-seeded]  -- no soft component at all

Usage (CPU, deterministic):
    export PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
    python escaped_mass_study.py --dims 4,6 --offsets 0,0.5,1,1.5,2,2.5,3,4 --copies 20 \
        --arms avgmm_cold,avgmm_warm,avav --jobs 8 --json out.json
    python escaped_mass_report.py out.json

MEASURED VERDICT (960 early-stopping runs + 480 fixed-budget runs; 20 independent target seeds
per cell, d=4 and d=6, ncomp=1, nmax 120000, n_chunk 5000).  Read this before using the number.

 1. THE CUMULATIVE STATISTIC HAS NO USABLE FLOOR.  The hypothesis "matched seed -> escaped_mass
    ~0" is FALSE.  With a PERFECT seed (offset=0) the cumulative escaped mass of the warm AV
    member is median 0.51 at d=4 and 0.80 at d=6 (20 seeds; max 0.84 / 0.88).  A correctly-placed
    VARAHA member contracts to a likelihood contour that legitimately excludes half to four-fifths
    of the posterior WEIGHT, and the statistic cannot tell that from a misplaced seed: AUC vs the
    offset=0 control is only 0.48-0.63 out to offset 2.

 2. THE FIRST-CHUNK STATISTIC IS SHARP -- IN ONE ARM ONLY.  Measured at the seed's own live
    volume, before any contraction, the offset=0 floor is 6e-6 (max 2.3e-4) at d=4 and 2.5e-7
    (max 6.8e-3) at d=6, four to six decades below the signal.  In the avgmm_cold arm, at a
    threshold equal to the measured offset=0 MAXIMUM (0/20 false positives):
        d=4 TPR  0.15 / 0.70 / 0.95 / 1.00 / 1.00 / 1.00 / 1.00  at offset 0.5/1/1.5/2/2.5/3/4
        d=6 TPR  0.00 / 0.10 / 0.50 / 0.80 / 0.75 / 1.00 / 1.00
    The d=6 degradation is the predicted STARVATION false negative: a cold uniform member has to
    land on the true peak for the escape to be observable at all.
    It beats the soft comparator decisively: 1-weight_share at the same 0-FP threshold reaches
    only TPR 0.20 (d=4) / 0.05 (d=6) at offset 2, where escaped_mass_early is at 1.00 / 0.80.

 3. IT IS BLIND IN BOTH OTHER ARMS -- EXACTLY 0.000 IN 320/320 RUNS.  If every member is seeded
    from the same displaced cloud (avgmm_warm -- which is what portfolio.bootstrap_from_samples
    does by DEFAULT, since mcsamplerEnsemble also implements bootstrap_from_samples), or if the
    portfolio is all-AV, then in the first chunk NOTHING is drawn outside the seeded volume and
    there is no escaping weight to measure.  AUC 0.500 / 0.45 at every offset, both dims.

 4. THE APPARENT PERFECT DETECTOR IN avgmm_warm IS A RUN-LENGTH ARTIFACT.  Under production-style
    early stopping the cumulative statistic separates offset 0 from every offset >= 0.5 at
    AUC 1.000 -- because the matched run reaches neff in ~3 chunks and its AV never contracts,
    while a mismatched run burns 24.  With the budget FIXED (neff disabled) the offset=0 floor
    moves from 4.9e-5 to 0.524 (d=4) and 1.5e-4 to 0.783 (d=6) and the AUC collapses to 0.47/0.58
    at offset 1/2.  1000/n_eff scores AUC 1.000 in the same cells: the statistic was measuring
    n_eff, not support.  ALWAYS compare at matched budget.

 5. STRUCTURAL LIMIT (the reason 2 and 3 cannot both be fixed).  escaped_mass is a reduction over
    samples that were DRAWN.  Weight can only be seen escaping member m if some OTHER member
    covers the complement of m's support -- which is precisely the configuration in which the
    balance heuristic already keeps lnZ unbiased (measured: |bias| <= 0.31 over all 160 avgmm_cold
    runs at d=4 and <= 1.29 at d=6, at every displacement out to 4).  In the arms where
    displacement DOES bias lnZ (all-AV: median -0.40 nat at d=4 / -1.27 at d=6 even at offset 0,
    reaching -22.6 median and -26000 worst at offset 4) no variant reaches a usable threshold:
    esc_cum has TPR 0.85 (d=4) / 0.60 (d=6) at offset 3, where the median bias is already -4.2 /
    -2.0 nats, and at d=6 the runs it MISSES are the worse ones (median |bias| 2.27 vs 1.42 for
    the hits).  The detector fires when it does not matter and is quiet when it does.
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_recovery as sr   # noqa: E402


# The truth pool only supplies (a) the in-box mass for true_lnZ and (b) the warm seed cloud.
# 2e5 draws give ~0.2% on the box mass = 0.002 nat, far below the 0.05-nat bias scale we resolve,
# and each study run builds TWO pools (true + displaced), so this is the dominant fixed cost.
DEFAULT_POOL_N = 200000


def _build_portfolio(arm, target, n_chunk):
    """Assemble the portfolio for one arm and return (sampler, members)."""
    if not sr._gpu_available():
        sr._force_cpu_modules()
    from RIFT.integrators import (mcsamplerPortfolio, mcsamplerAdaptiveVolume,
                                  mcsamplerEnsemble)

    def _av():
        try:
            return mcsamplerAdaptiveVolume.MCSampler(n_chunk=n_chunk)
        except TypeError:
            return mcsamplerAdaptiveVolume.MCSampler()

    if arm == "avav":
        members = [_av(), _av()]
    else:
        members = [_av(), mcsamplerEnsemble.MCSampler()]
    if not sr._gpu_available():
        for m in members:
            sr._force_cpu(m)
    s = mcsamplerPortfolio.MCSampler(portfolio=list(members))
    if not sr._gpu_available():
        sr._force_cpu(s)

    def uniform_pdf(d):
        wdt = target.rlim[d] - target.llim[d]
        return np.vectorize(lambda x, wdt=wdt: 1.0 / wdt)

    for d, p in enumerate(target.params):
        s.add_parameter(p, uniform_pdf(d), prior_pdf=uniform_pdf(d),
                        left_limit=float(target.llim[d]), right_limit=float(target.rlim[d]),
                        adaptive_sampling=True)
    return s, members


def run_case(ndim, ncomp, target_seed, offset, arm, run_seed,
             nmax, neff, n_chunk, ncomp_gmm=2, verbose=False):
    """One (target, displacement, arm) run.  Never raises: errors are recorded."""
    t0 = time.time()
    out = dict(ndim=ndim, ncomp=ncomp, target_seed=target_seed, offset=float(offset),
               arm=arm, run_seed=run_seed, nmax=int(nmax), n_chunk=int(n_chunk))
    try:
        np.random.seed(run_seed)
        true_t = sr.MixtureTarget(ndim, ncomp, target_seed)                 # what we integrate
        seed_t = sr.MixtureTarget(ndim, ncomp, target_seed, offset=offset)  # where the seed came from
        cloud = sr._warm_seed_cloud(seed_t)

        s, members = _build_portfolio(arm, true_t, n_chunk)
        _dims = tuple(range(ndim))
        setup_kw = {}
        if arm != "avav":
            # PRODUCTION SHAPE: a real run always supplies a grouping spec, and that is the
            # configuration in which the GMM member is a genuine full-dim mixture.
            setup_kw = dict(n_comp={_dims: ncomp_gmm}, gmm_dict={_dims: None},
                            correlate_all_dims=True)
        try:
            s.setup(**setup_kw)
        except TypeError:
            s.setup()

        # SEEDING.  avgmm_cold seeds ONLY the AV member; the other two arms seed every member that
        # exposes bootstrap_from_samples (which is what portfolio.bootstrap_from_samples does, and
        # what production therefore does by default).
        if arm == "avgmm_cold":
            members[0].bootstrap_from_samples(cloud, cover_frac=0.0)
            out["n_warmed"] = 1
        else:
            out["n_warmed"] = int(s.bootstrap_from_samples(cloud, cover_frac=0.0))

        extra = dict(n=n_chunk, n_adapt=100, floor_level=0.0, tempering_exp=0.1,
                     neff=neff, nmax=int(nmax), save_intg=True, verbose=verbose)
        lnI, logvar, eff, dret = s.integrate_log(true_t.as_lnfunc(), *true_t.params,
                                                 no_protect_names=True, **extra)
        lnI = float(sr._asnumpy(lnI))
        logvar = float(sr._asnumpy(logvar))
        ln_wt = sr.log_weights_from_rvs(s._rvs)

        esc = np.asarray(dret.get("portfolio_escaped_mass", []), dtype=float)
        early = np.asarray(dret.get("portfolio_escaped_mass_early", []), dtype=float)
        share = np.asarray(dret.get("portfolio_member_weight_share", []), dtype=float)
        hard = np.asarray(dret.get("portfolio_member_hard_edged", []), dtype=bool)
        hist = np.asarray(dret.get("portfolio_escaped_mass_history", []), dtype=float)
        out.update(
            lnI=lnI, true_lnZ=float(true_t.true_lnZ), bias_ln=lnI - float(true_t.true_lnZ),
            n_eff=float(sr._asnumpy(eff)), n_ess=float(sr.n_ess_kish(ln_wt)),
            n_eval=int(getattr(s, "ntotal", 0)),
            rel_err=float(np.exp(0.5 * logvar - lnI)) if np.isfinite(logvar) else float("nan"),
            escaped_mass=[float(x) for x in esc],
            escaped_mass_early=[float(x) for x in early],
            weight_share=[float(x) for x in share],
            hard_edged=[bool(x) for x in hard],
            # HEADLINE STATISTICS, as a monitor would read them:
            #  esc_warm  -- the warm-started member (index 0), the one under test;
            #  esc_max   -- worst over members observed to be hard-edged (what production reads,
            #               since it does not know which member was warm-started);
            #  share_warm-- the soft comparator for the same member.
            esc_warm=float(esc[0]) if len(esc) else float("nan"),
            esc_early_warm=float(early[0]) if len(early) else float("nan"),
            esc_max=float(dret.get("portfolio_escaped_mass_max", np.nan)),
            esc_early_max=float(dret.get("portfolio_escaped_mass_early_max", np.nan)),
            share_warm=float(share[0]) if len(share) else float("nan"),
            n_chunks=int(hist.shape[0]) if hist.ndim == 2 else 0,
            # full per-chunk history (n_chunks x n_members) so any "first K chunks" variant of the
            # statistic can be evaluated post hoc without re-running: the cumulative and the
            # first-chunk numbers are two points on this curve, and which one is the detector is
            # exactly what the study has to decide.
            esc_hist=hist.tolist() if hist.ndim == 2 else [],
            wallclock=time.time() - t0, error=None)
    except Exception as e:
        import traceback
        out.update(error="{}: {}".format(type(e).__name__, e),
                   traceback=traceback.format_exc(), wallclock=time.time() - t0)
    return out


def _worker(job):
    sr.TRUTH_POOL_N = job.pop("pool_n", DEFAULT_POOL_N)
    return run_case(**job)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dims", default="4,6")
    ap.add_argument("--ncomp", type=int, default=1)
    ap.add_argument("--ncomp-gmm", type=int, default=2)
    ap.add_argument("--offsets", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--arms", default="avgmm_cold,avgmm_warm,avav")
    ap.add_argument("--copies", type=int, default=20,
                    help="independent (target seed, run seed) pairs per cell")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--nmax", type=int, default=120000)
    ap.add_argument("--neff", type=int, default=3000)
    ap.add_argument("--n-chunk", type=int, default=5000)
    ap.add_argument("--pool-n", type=int, default=DEFAULT_POOL_N)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--verbose", action="store_true")
    opts = ap.parse_args(argv)

    dims = [int(x) for x in opts.dims.split(",") if x.strip()]
    offsets = [float(x) for x in opts.offsets.split(",") if x.strip()]
    arms = [x.strip() for x in opts.arms.split(",") if x.strip()]

    jobs = []
    for d in dims:
        for off in offsets:
            for arm in arms:
                for c in range(opts.copies):
                    ts = opts.seed0 + c
                    jobs.append(dict(ndim=d, ncomp=opts.ncomp, target_seed=ts, offset=off,
                                     arm=arm, run_seed=900000 + 37 * ts + 11 * d,
                                     nmax=opts.nmax, neff=opts.neff, n_chunk=opts.n_chunk,
                                     ncomp_gmm=opts.ncomp_gmm, verbose=opts.verbose,
                                     pool_n=opts.pool_n))
    print("# escaped_mass_study: {} runs ({} dims x {} offsets x {} arms x {} copies)".format(
        len(jobs), len(dims), len(offsets), len(arms), opts.copies))
    sys.stdout.flush()

    t0 = time.time()
    if opts.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(opts.jobs) as pool:
            results = pool.map(_worker, jobs, chunksize=1)
    else:
        results = [_worker(j) for j in jobs]
    print("# wallclock {:.1f}s".format(time.time() - t0))

    if opts.json:
        with open(opts.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print("# wrote", opts.json)
    n_err = sum(1 for r in results if r.get("error"))
    print("# errors:", n_err)
    for r in results[:3]:
        if r.get("error"):
            print(r.get("traceback", r["error"]))
    return 1 if n_err == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
