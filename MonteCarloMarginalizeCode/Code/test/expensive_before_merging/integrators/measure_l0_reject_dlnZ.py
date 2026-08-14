#!/usr/bin/env python3
"""Measure `--sampler-l0-rescue-reject-dlnZ` against the POST-#79 gate.

WHY THIS HAD TO WAIT FOR #79
----------------------------
The 0.5-nat default was chosen when both sides of the comparison were read out of the
fair-drawn `_rvs`, which carries a `log(n_retained/eff_samp)` offset that does NOT cancel
between two passes at different `n_eff` (measured at +3.48 nats, `verify_skew.py`).  Tuning
against that would have been calibrating to the bug.  With #79 in (re-landed as #86) both
sides come from the retained reserve via `lnZ_from_reserve`, so the number now means what the
knob's help text says it means, and it can be measured.

WHAT THE KNOB DECIDES
---------------------
The L0 rescue re-runs a collapsed pass warm, seeded from that pass's own peak.  The warm pass
is precise but confined to the seeded region, so it is biased low by any mode the seed missed.
The gate keeps the COLD result when

    cold_lnZ - warm_lnZ > dlnZ

The trade is asymmetric and both directions are real:
  * TOO SMALL -> a good warm pass is binned in favour of a collapsed cold one.  LOUD: the run
    reports the collapse.
  * TOO LARGE -> a genuinely truncated warm pass is reported.  QUIET: it has an excellent ESS
    over a sliver of the support, and nothing says so.

So this measures two distributions on known-lnZ targets:
  NULL    unimodal.  The seed cannot miss a mode because there is only one.  Any gap is noise,
          and its spread is the floor below which no threshold can go.
  SIGNAL  bimodal, mass fraction `f` in the mode the cold pass found.  A seed confined to that
          mode measures f*Z, so the true deficit is -log(f).

Usage:
    python3 measure_l0_reject_dlnZ.py                 # default: 40 replicates per condition
    python3 measure_l0_reject_dlnZ.py --reps 100
Runs single-threaded on CPU; set OMP_NUM_THREADS=1.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE)

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV  # noqa: E402
from RIFT.integrators.mcsamplerAdaptiveVolume import build_warm_seed  # noqa: E402

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
NDIM = len(NAMES)
LO = np.zeros(NDIM)
HI = np.ones(NDIM)
AX = list(range(NDIM))
_ILE = os.path.join(CODE, "bin", "integrate_likelihood_extrinsic_batchmode")


def _load_ile_helpers():
    """Exec the ILE's lnZ helpers so this measures THE gate, not a copy of it."""
    src = open(_ILE).read()
    start = src.index("def ln_weights_from_rvs")
    end = src.index("def _warm_seed_geometry")
    ns = {"numpy": np, "np": np, "mcsamplerAdaptiveVolume": mcsamplerAV,
          "_rvs_lnL_convention": lambda x=None: bool(x)}
    exec(compile(src[start:end], "ile_lnZ_helpers", "exec"), ns)
    assert "_lnZ_of_reserve_or_rvs" in ns, "PR #79 helper absent -- is #86 merged in this tree?"
    return ns


H = _load_ile_helpers()
_lnZ_of_reserve_or_rvs = H["_lnZ_of_reserve_or_rvs"]


SAMPLER_KIND = 'AV'


def _sampler(n_chunk=20000):
    if SAMPLER_KIND == 'portfolio':
        return _portfolio(n_chunk)
    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = mcsamplerAV.xpy_default
    s.identity_convert = mcsamplerAV.identity_convert
    for name in NAMES:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    return s


def _portfolio(n_chunk=20000):
    """AV + GMM portfolio, built the way the ILE builds one (see test_l0_rescue_seed).

    The interesting difference for this measurement: the GMM member carries a DEFENSIVE
    component, so the portfolio is the configuration the L0 rescue's own docstring says
    "avoids this trade entirely".  Whether its cold pass is a usable lnZ reference -- which is
    what the gate needs and what AV cannot supply -- is exactly the open question.
    """
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPF
    import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
    members = [mcsamplerAV.MCSampler(n_chunk=n_chunk), mcsamplerEnsemble.MCSampler()]
    s = mcsamplerPF.MCSampler(portfolio=members)
    pdf = np.vectorize(lambda x: 1.0)
    for name in NAMES:
        s.add_parameter(name, pdf, prior_pdf=pdf, left_limit=0.0, right_limit=1.0,
                        adaptive_sampling=True)
    s.setup()
    return s


class Target(object):
    """Sum of Gaussians on the unit cube, with the float64 underflow of the real code.

    Uniform prior of density 1, so Z = sum_k A_k * prod_i (w_ki * sqrt(2 pi)) when every mode
    sits well inside the cube -- which is checked at construction.
    """

    def __init__(self, centers, widths, log_amps):
        self.c = np.atleast_2d(np.asarray(centers, dtype=float))
        self.w = np.atleast_2d(np.asarray(widths, dtype=float))
        self.a = np.asarray(log_amps, dtype=float).ravel()
        # every mode must be >5 sigma from every face, or the analytic Z is wrong
        assert np.all(self.c - 5 * self.w > 0) and np.all(self.c + 5 * self.w < 1), \
            "a mode is too close to the boundary for the analytic normalization"
        self.ln_mode_Z = self.a + np.sum(np.log(self.w * np.sqrt(2 * np.pi)), axis=1)
        m = self.ln_mode_Z.max()
        self.lnZ_true = float(m + np.log(np.sum(np.exp(self.ln_mode_Z - m))))
        self.mass_frac = np.exp(self.ln_mode_Z - self.lnZ_true)
        self.lnLmax = float(self.a.max())
        # modes carrying a non-negligible share; a 1e-6 mode is not a miss worth flagging
        self.n_modes_expected = int(np.sum(self.mass_frac > 0.02))

    def __call__(self, *args, **kwargs):
        x = np.array([np.asarray(v, dtype=float).ravel() for v in args]).T
        terms = np.stack([
            self.a[k] - 0.5 * np.sum(((x - self.c[k]) / self.w[k]) ** 2, axis=-1)
            for k in range(len(self.a))], axis=0)
        m = terms.max(axis=0)
        out = m + np.log(np.sum(np.exp(terms - m), axis=0))
        return np.where(out > self.lnLmax - 745.0, out, -np.inf)

    def nearest_mode(self, X):
        d = np.stack([np.sum(((X - self.c[k]) / self.w[k]) ** 2, axis=-1)
                      for k in range(len(self.a))], axis=0)
        return np.argmin(d, axis=0), np.sqrt(d.min(axis=0))


def one_replicate(target, seed, neff_target=8, nmax=400000, n=20000, fairdraw_max=200):
    """Run the REAL rescue sequence: cold pass, reserve, rank-tested seed, warm pass.

    Returns None when the cold pass did not actually collapse -- a replicate that lands on a
    healthy pass is not evidence about a knob that only fires on collapsed ones.
    """
    np.random.seed(seed)
    s = _sampler()
    kw = dict(no_protect_names=True, verbose=False,
              igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=fairdraw_max)
    if SAMPLER_KIND == 'portfolio':
        kw['save_intg'] = True
    try:
        cold = s.integrate_log(target, *NAMES, nmax=nmax, neff=neff_target, n=n, **kw)
    except Exception:
        return None
    cold_neff = None if cold is None or cold[2] is None else float(
        mcsamplerAV.identity_convert(cold[2]))
    reserve = getattr(s, '_warm_seed_reserve', None)
    if reserve is None:
        return None
    cold_rvs = dict(s._rvs)
    cold_lnZ, cold_src = _lnZ_of_reserve_or_rvs(s, cold_rvs, reserve=reserve)

    X = np.asarray(reserve['X'], dtype=float)
    lnv = np.asarray(reserve['lnL'], dtype=float).ravel()
    if lnv.size < 1 or not np.any(np.isfinite(lnv)):
        return None
    seed_pts, info = build_warm_seed(X, lnv, LO, HI, AX, deltalnL=15.0)
    seed_mode = int(target.nearest_mode(np.atleast_2d(seed_pts))[0][0])

    try:
        s.bootstrap_from_samples(seed_pts, cover_frac=0.0)
        warm = s.integrate_log(target, *NAMES, nmax=nmax, neff=neff_target, n=n, **kw)
    except Exception:
        return None
    warm_neff = None if warm is None or warm[2] is None else float(
        mcsamplerAV.identity_convert(warm[2]))
    warm_lnZ, warm_src = _lnZ_of_reserve_or_rvs(s, s._rvs)
    if cold_lnZ is None or warm_lnZ is None:
        return None
    if not (np.isfinite(cold_lnZ) and np.isfinite(warm_lnZ)):
        return None

    # did the warm pass actually reach every mode?  (its own retained set is the evidence)
    wres = getattr(s, '_warm_seed_reserve', None)
    modes_seen = set()
    if wres is not None:
        idx, _ = target.nearest_mode(np.asarray(wres['X'], dtype=float))
        modes_seen = set(int(v) for v in np.unique(idx))
    _rep = lambda r: (None if (r is None or r[0] is None)
                      else float(mcsamplerAV.identity_convert(r[0])))
    return dict(cold_reported=_rep(cold), warm_reported=_rep(warm),
                gap=float(cold_lnZ - warm_lnZ), cold_lnZ=float(cold_lnZ),
                warm_lnZ=float(warm_lnZ), cold_src=cold_src, warm_src=warm_src,
                cold_neff=cold_neff, warm_neff=warm_neff, seed_mode=seed_mode,
                modes_seen=modes_seen, puffed=bool(info['puffed']))


def run(label, target, reps, seed0):
    rows = []
    for i in range(reps):
        r = one_replicate(target, seed0 + i)
        if r is not None:
            rows.append(r)
    print("\n### {}   ({} usable of {} replicates)".format(label, len(rows), reps))
    if not rows:
        print("    no usable replicates")
        return rows
    g = np.array([r['gap'] for r in rows])
    srcs = set((r['cold_src'], r['warm_src']) for r in rows)
    print("    true lnZ {:.4f}   mass fractions {}".format(
        target.lnZ_true, np.array2string(target.mass_frac, precision=3)))
    print("    lnZ sources (cold,warm): {}".format(sorted(srcs)))
    print("    cold n_eff median {:.2f}   warm n_eff median {:.2f}".format(
        float(np.median([r['cold_neff'] or np.nan for r in rows])),
        float(np.median([r['warm_neff'] or np.nan for r in rows]))))
    if len(target.a) > 1:
        conf = sum(1 for r in rows if len(r['modes_seen']) < len(target.a))
        print("    warm pass reached only ONE mode in {}/{} replicates".format(conf, len(rows)))
    dc = np.array([r['cold_lnZ'] for r in rows]) - target.lnZ_true
    dw = np.array([r['warm_lnZ'] for r in rows]) - target.lnZ_true
    print("    cold_lnZ - TRUE: median {:+.3f}  sd {:.3f}    <- the gate's reference".format(
        float(np.median(dc)), float(np.std(dc, ddof=1))))
    print("    warm_lnZ - TRUE: median {:+.3f}  sd {:.3f}".format(
        float(np.median(dw)), float(np.std(dw, ddof=1))))
    cr = np.array([r['cold_reported'] for r in rows if r['cold_reported'] is not None])
    wr = np.array([r['warm_reported'] for r in rows if r['warm_reported'] is not None])
    if cr.size:
        print("    cold lnZ REPORTED BY integrate_log - TRUE: median {:+.3f}  sd {:.3f}".format(
            float(np.median(cr - target.lnZ_true)), float(np.std(cr - target.lnZ_true, ddof=1))))
    if wr.size:
        print("    warm lnZ REPORTED BY integrate_log - TRUE: median {:+.3f}  sd {:.3f}".format(
            float(np.median(wr - target.lnZ_true)), float(np.std(wr - target.lnZ_true, ddof=1))))
    print("    gap = cold_lnZ - warm_lnZ:  median {:+.3f}   mean {:+.3f}   sd {:.3f}"
          "   [p05 {:+.3f}, p95 {:+.3f}]".format(
              float(np.median(g)), float(np.mean(g)), float(np.std(g, ddof=1)),
              float(np.percentile(g, 5)), float(np.percentile(g, 95))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--rho", type=float, default=100.0)
    ap.add_argument("--sampler", choices=("AV", "portfolio"), default="AV")
    ap.add_argument("--out-json", default=None,
                    help="dump per-replicate records so the analysis can be redone without "
                         "re-measuring (the threshold table is cheap; the passes are not)")
    args = ap.parse_args()
    global SAMPLER_KIND
    SAMPLER_KIND = args.sampler

    rho = args.rho
    w = (0.5 / rho) * np.ones(NDIM)
    lnA = 0.5 * rho ** 2

    print("=" * 78)
    print("Measuring --sampler-l0-rescue-reject-dlnZ on the POST-#79 gate")
    print("sampler = {}, rho = {}, {} replicates per condition".format(
        args.sampler, rho, args.reps))
    print("=" * 78)

    conditions = []

    # NULL: one mode.  Any gap here is noise.
    uni = Target([0.5 * np.ones(NDIM)], [w], [lnA])
    conditions.append(("NULL (unimodal -- a good warm pass)", uni,
                       run("NULL (unimodal -- a good warm pass)", uni, args.reps, 1000)))

    # SIGNAL: two modes, varying how much mass sits in the one the cold pass finds.
    for f in (0.5, 0.75, 0.9):
        c2 = np.copy(0.5 * np.ones(NDIM)); c2[0] = 0.25
        c1 = np.copy(0.5 * np.ones(NDIM)); c1[0] = 0.75
        # equal widths -> amplitude ratio sets the mass ratio
        tgt = Target([c1, c2], [w, w], [lnA, lnA + np.log((1 - f) / f)])
        lbl = "SIGNAL (bimodal, seeded mode holds f={:.2f}; true deficit -log f = {:+.3f})".format(
            f, -np.log(f))
        conditions.append((lbl, tgt, run(lbl, tgt, args.reps, 2000 + int(1000 * f))))

    if args.out_json:
        import json
        with open(args.out_json, "w") as fh:
            json.dump([dict(condition=lbl, lnZ_true=t.lnZ_true,
                            mass_frac=list(map(float, t.mass_frac)), n_modes=len(t.a),
                            **{k: (sorted(v) if isinstance(v, set) else v)
                               for k, v in r.items()})
                       for lbl, t, rows in conditions for r in rows], fh, indent=1)
        print("\nper-replicate records -> {}".format(args.out_json))

    # ---- rejection rate CONDITIONED ON WHAT ACTUALLY HAPPENED
    # The condition label is the INTENT, not the outcome: on a portfolio the defensive GMM
    # component means a seeded warm pass usually still reaches every mode, so most "SIGNAL"
    # replicates are not truncated and a rejection there is a FALSE positive.  Split on the
    # warm pass's own retained set instead of on the label.
    good = [r for _, t, rows in conditions for r in rows
            if len(r['modes_seen']) >= t.n_modes_expected]
    trunc = [r for _, t, rows in conditions for r in rows
             if len(r['modes_seen']) < t.n_modes_expected]
    print("\n" + "=" * 78)
    print("REJECTION RATE BY WHAT THE WARM PASS ACTUALLY DID")
    print("  good  = warm pass reached every mode ({} replicates)".format(len(good)))
    print("  trunc = warm pass reached fewer     ({} replicates)".format(len(trunc)))
    print("=" * 78)
    gg = np.array([r['gap'] for r in good]) if good else np.array([])
    tg = np.array([r['gap'] for r in trunc]) if trunc else np.array([])
    print("{:>8}  {:>26}  {:>26}".format("dlnZ", "FPR (good pass binned)", "TPR (truncation caught)"))
    for thr in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        f = "{:>25.0f}%".format(100 * np.mean(gg > thr)) if gg.size else "{:>26}".format("n/a")
        t_ = "{:>25.0f}%".format(100 * np.mean(tg > thr)) if tg.size else "{:>26}".format("n/a")
        print("{:>8.2f}  {}  {}".format(thr, f, t_))

    # ---- the decision table
    print("\n" + "=" * 78)
    print("THRESHOLD TABLE")
    print("  FPR = a GOOD warm pass rejected (loud failure: collapsed cold result reported)")
    print("  TPR = a TRUNCATED warm pass rejected (the catch we want)")
    print("=" * 78)
    null = np.array([r['gap'] for r in conditions[0][2]]) if conditions[0][2] else np.array([])
    sig = [(lbl, np.array([r['gap'] for r in rows])) for lbl, _, rows in conditions[1:] if rows]
    hdr = "{:>8}  {:>8}".format("dlnZ", "FPR")
    for lbl, _ in sig:
        f = lbl.split("f=")[1][:4]
        hdr += "  {:>10}".format("TPR f=" + f)
    print(hdr)
    for thr in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        line = "{:>8.2f}  {:>7.0f}%".format(thr, 100 * np.mean(null > thr) if null.size else np.nan)
        for _, gg in sig:
            line += "  {:>9.0f}%".format(100 * np.mean(gg > thr))
        print(line)

    if null.size:
        print("\nNULL spread sets the floor: sd {:.3f} nats, p95 {:+.3f}.".format(
            float(np.std(null, ddof=1)), float(np.percentile(null, 95))))
    print("STRUCTURAL BLIND SPOT: a threshold T can never catch a missed mode carrying less")
    print("than 1-exp(-T) of the mass, however well tuned --")
    for thr in (0.5, 1.0, 2.0):
        print("    T={:.1f} -> blind to any mode holding < {:.0f}% of the total mass".format(
            thr, 100 * (1 - np.exp(-thr))))


if __name__ == "__main__":
    sys.exit(main())
