#!/usr/bin/env python
"""
khat_validation.py -- does the Pareto-k tail diagnostic actually CATCH the confidently-wrong runs?

MOTIVATION.  On a sharp high-SNR target we measured runs that were confidently wrong: the copy with
the HIGHEST n_eff in its arm was the MOST biased in lnZ (n_eff 58 -> 11 nats low; n_eff 123.6 -- the
highest of the whole study -- -> 10 nats low).  n_eff (Kish) measures weight CONCENTRATION, not
COVERAGE, so a proposal that has missed mass looks confident.  Any error estimate keyed on n_eff
inherits that, and it fails in the dangerous direction.

`RIFT.integrators.statutils.pareto_khat_from_log` (added with the MC-error-estimate work) is the
proposed instrument: the generalized-Pareto tail index of the importance weights, with
  k < 0.5  variance finite, naive sigma meaningful
  0.5-0.7  variance marginal, naive sigma optimistic
  k > 0.7  tail unresolved -- naive sigma is a LOWER BOUND and the integral may be dominated by
           unseen tail mass.
That is exactly the failure above, so the question is empirical: on REAL failures (not synthetic GPD
draws), does k-hat fire when n_eff does not?

WHY THIS HARNESS CAN ANSWER IT.  We reuse the merge gate's MixtureTarget, which exposes `true_lnZ`.
So every run has a KNOWN bias, and we can score k-hat as a binary classifier of "this run is wrong"
-- sensitivity, specificity, and a head-to-head against n_eff.  Sampling on a real event could never
do this: there is no truth to score against.

k-hat is computed from the SAME importance log-weights the gate uses for its shape metrics, captured
by wrapping `shape_recovery.shape_metrics` (which receives ln_wt).  We do not modify the gate: today
`run_one` discards `dict_return`, so the sampler-emitted `mc_diag['pareto_khat']` never reaches the
record.

Usage (CPU; keep jobs small -- RLIMIT_NPROC counts THREADS on this cluster):
    OMP_NUM_THREADS=1 python khat_validation.py --snrs 80,160 --copies 40 --jobs 6
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
import shape_recovery as SR
from RIFT.integrators.statutils import pareto_khat_from_log

SIGMA_REF, SNR_REF = 0.7, 20.0
_CAP = {}


def _install_capture():
    """Wrap the gate's shape_metrics to stash the importance log-weights it is handed.
    Idempotent, and applied inside the worker so it survives fork/spawn."""
    if getattr(SR, "_khat_capture_installed", False):
        return
    _orig = SR.shape_metrics

    def _wrapped(target, X, ln_wt, rng, *a, **kw):
        try:
            _CAP["ln_wt"] = np.asarray(ln_wt, dtype=float).ravel().copy()
        except Exception:
            _CAP["ln_wt"] = None
        return _orig(target, X, ln_wt, rng, *a, **kw)
    SR.shape_metrics = _wrapped
    SR._khat_capture_installed = True


def _one(job):
    (snr, ndim, ncomp, nmax, neff, n_chunk, seed, kind) = job
    _install_capture()
    _CAP.pop("ln_wt", None)
    sigma = SIGMA_REF * (SNR_REF / float(snr))
    try:
        target = SR.MixtureTarget(ndim, ncomp, seed, sigma_1d=sigma)
        rec = SR.run_one(kind, target, nmax, neff, n_chunk=n_chunk, seed=seed)
        status = SR.evaluate(rec)
        status = status if isinstance(status, str) else status[0]
    except Exception as e:
        return dict(snr=snr, seed=seed, kind=kind, status="ERROR", err=str(e)[:120],
                    bias=float("nan"), n_eff=float("nan"), khat=None)
    lw = _CAP.get("ln_wt")
    khat = None
    if lw is not None and len(lw):
        try:
            khat = pareto_khat_from_log(lw)
        except Exception:
            khat = None
    return dict(snr=snr, seed=seed, kind=kind, status=status,
                bias=float(rec.get("bias_ln", float("nan"))),
                n_eff=float(rec.get("n_eff", float("nan"))),
                khat=(float(khat) if khat is not None else None))


def score(recs, bias_tol, khat_cut, neff_cut):
    """Score k-hat and n_eff as binary detectors of 'this run is materially wrong'."""
    use = [r for r in recs if np.isfinite(r["bias"]) and r["khat"] is not None]
    wrong = [r for r in use if abs(r["bias"]) > bias_tol]
    right = [r for r in use if abs(r["bias"]) <= bias_tol]
    def rate(rs, pred):
        return (100.0 * sum(1 for r in rs if pred(r)) / len(rs)) if rs else float("nan")
    k_flag = lambda r: r["khat"] > khat_cut
    n_flag = lambda r: r["n_eff"] < neff_cut
    print("\nscored on %d runs with finite bias and a k-hat (%d wrong, %d accurate; |bias|>%.2f = wrong)"
          % (len(use), len(wrong), len(right), bias_tol))
    print("  %-28s %12s %12s" % ("detector", "sensitivity", "false alarm"))
    print("  %-28s %11.0f%% %11.0f%%" % ("k-hat > %.2f" % khat_cut, rate(wrong, k_flag), rate(right, k_flag)))
    print("  %-28s %11.0f%% %11.0f%%" % ("n_eff < %g" % neff_cut, rate(wrong, n_flag), rate(right, n_flag)))
    # the decisive subset: runs n_eff would have PASSED but that are actually wrong
    sneaky = [r for r in wrong if r["n_eff"] >= neff_cut]
    if sneaky:
        caught = sum(1 for r in sneaky if k_flag(r))
        print("  CONFIDENTLY WRONG (n_eff>=%g yet |bias|>%.2f): %d runs, k-hat catches %d (%.0f%%)"
              % (neff_cut, bias_tol, len(sneaky), caught, 100.0 * caught / len(sneaky)))
        print("     their k-hat: %s" % " ".join("%.2f" % r["khat"] for r in sorted(sneaky, key=lambda r: -abs(r["bias"]))[:12]))
        print("     their bias : %s" % " ".join("%+.2f" % r["bias"] for r in sorted(sneaky, key=lambda r: -abs(r["bias"]))[:12]))
    else:
        print("  (no confidently-wrong runs in this sample -- raise SNR or copies)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", default="80,160")
    ap.add_argument("--ndim", type=int, default=4)
    ap.add_argument("--ncomp", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=2000000)
    ap.add_argument("--neff", type=int, default=3000)
    ap.add_argument("--n-chunk", type=int, default=10000)
    ap.add_argument("--copies", type=int, default=40)
    ap.add_argument("--kinds", default="AV")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--bias-tol", type=float, default=0.10, help="gate's lnZ tolerance")
    ap.add_argument("--khat-cut", type=float, default=0.7)
    ap.add_argument("--neff-cut", type=float, default=100.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    jobs = [(float(s), a.ndim, a.ncomp, a.nmax, a.neff, a.n_chunk, a.seed0 + c, k)
            for k in a.kinds.split(",") for s in a.snrs.split(",") for c in range(a.copies)]
    print("# k-hat validation: %d runs (%s, SNR %s, %d copies, nmax=%d, chunk=%d)"
          % (len(jobs), a.kinds, a.snrs, a.copies, a.nmax, a.n_chunk))
    sys.stdout.flush()
    t0 = time.time()
    pool = Pool(a.jobs); recs = pool.map(_one, jobs); pool.close(); pool.join()
    print("# done in %.1f min" % ((time.time() - t0) / 60.0))
    if a.json:
        json.dump(recs, open(a.json, "w"), indent=1); print("# wrote", a.json)
    ok = [r for r in recs if r["khat"] is not None]
    print("# k-hat available for %d/%d runs" % (len(ok), len(recs)))
    score(recs, a.bias_tol, a.khat_cut, a.neff_cut)


if __name__ == "__main__":
    main()
