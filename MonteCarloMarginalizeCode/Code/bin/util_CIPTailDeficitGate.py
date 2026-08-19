#!/usr/bin/env python3
#
# util_CIPTailDeficitGate.py
#
# SEVERE-deficit gate for the CIP composition reweight (used by
# util_CIPCompositionReweightWrapper.sh).  From RIFT's own products alone -- a CIP training
# set (composite/all.net) and a CIP posterior produced FROM it -- it decides whether the
# posterior returns materially less transverse-tail mass than the training likelihoods
# license:
#
#     implied tail mass   = sum over chi1_perp bins of prior_vol(bin ∩ tail) x <L>_bin,
#                           <L>_bin = mean exp(lnL - peak) over the bin's ILE rows;
#                           prior volume by MC of the analytic spin prior
#                           (a1 ~ U(0, chi-max), isotropic tilt), fixed seed
#     delivered tail mass = fraction of the posterior's samples in the tail
#     tail boundary       = the training set's own chi1_perp q80 (self-quantile)
#     R = delivered / implied
#
#     FIRE (severe deficit)  <=>  implied >= floor_counts / n_post   AND   R < threshold
#     ABSTAIN-FLOOR          <=>  implied <  floor_counts / n_post
#     NO-FIRE                otherwise
#
# THE VALIDITY FLOOR IS MANDATORY AND CANNOT BE BYPASSED.  When the implied tail mass is
# below what n_post samples can resolve, a perfectly correct posterior is expected to put
# ~zero samples there and R degenerates to 0 -- maximum apparent deficit exactly when the
# detector can resolve nothing, the worst possible failure direction.  On known-truth
# healthy-narrow benchmarks (toybench T5/T5b/T5c, transverse confinement 8/15/20 nats) every
# chain read R = 0 and only the floor prevented a false FIRE.  There is deliberately no
# option to disable it; decide() applies it before the threshold is ever consulted.
#
# THE THRESHOLD IS CHANNEL-CALIBRATED.  R depends on which posterior supplies "delivered":
# a fresh single-CIP posterior (what this gate's pass-1 measures) reads mid-band healthy
# events LOWER than the production consolidated posterior by up to ~0.07, while severe-
# deficit events read the same in both.  The population threshold 0.42 (production-channel
# gap 0.379-0.457 over 102 events) therefore does NOT transfer: in the deployment channel
# the healthy-control floor is 0.386 and the severe-deficit ceiling is 0.270, giving
# THRESHOLD = 0.32 (geometric midpoint; margins ~1.2x each side, outside the training-row
# bootstrap bands of the edge events).  This was caught by the regression fixture, not by
# design -- see test/test_tail_deficit_gate.py.
#
# SCOPE (measured, 2026-08-19 study record: results_triage/R_TOYBENCH_VALIDATION_2026-08-19.md
# and R_POPULATION_CALIBRATION_2026-08-19.md of rift_transverse_highSNR_study):
#   * This detects SEVERE deficits only (production low-mass class, chi1_perp width ratios
#     ~0.62-0.69 vs reference; deployment-channel R <= 0.270).  Mild deficits and healthy-
#     width posteriors are NOT separable above the threshold: out-of-sample, truth-deficient
#     toybench T3 chains (R 0.532-0.634) abut truth-healthy T2 chains (R 0.633-0.673), and
#     two borderline production events (dead-tail S241011k at 0.358, ratio-0.794 S240512r)
#     sit just above the deployment gap.  Expect roughly half the affected low-mass events
#     to be repaired and the mild rest to be (loudly) left alone.
#   * Safety is the strongly supported side: zero false fires in 77 in-sample healthy events
#     (95% bound: false-fire rate <= 3.8%) and zero on known-truth healthy-narrow toys.
#     The fire side is in-sample-validated only (12 production events at 100%, 95% bound on
#     the miss rate 22.1%, plus the causal 13-event paired-CIP repair).
#
# Exit codes: 0 = evaluated (decision on the LAST stdout line, machine-readable);
#             2 = could not evaluate (caller must fail safe: proceed with the original data).

import argparse
import json
import sys

import numpy as np

THRESHOLD = 0.32       # calibrated IN THE DEPLOYMENT CHANNEL (fresh-CIP posterior; see SCOPE)
FLOOR_COUNTS = 50.0    # validity floor: implied must be >= FLOOR_COUNTS / n_post (pre-registered)
CHI_MAX = 0.99
EDGES = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99])
MIN_BIN = 20
SELFQ = 0.80
LNL_COL = 9
N_PRIOR = 4_000_000
PRIOR_SEED = 11
MIN_POST = 100         # a posterior this small cannot support any decision -> error (exit 2)


def decide(implied, R, n_post, threshold=THRESHOLD, floor_counts=FLOOR_COUNTS):
    """The one and only decision function.  The validity floor is applied FIRST and
    unconditionally; no caller can reach the threshold comparison without passing it."""
    floor = floor_counts / float(n_post)
    if not np.isfinite(implied) or implied < floor:
        return "ABSTAIN-FLOOR", floor
    decision = "FIRE" if (np.isfinite(R) and R < threshold) else "NO-FIRE"
    # structural guarantee, not a debug aid: a FIRE below the floor is a contract violation
    assert decision != "FIRE" or implied >= floor
    return decision, floor


def load_posterior_cp(path):
    hdr = open(path).readline().lstrip('#').split()
    d = np.genfromtxt(path, names=hdr, skip_header=1, invalid_raise=False)
    cp = np.hypot(np.asarray(d['a1x'], float), np.asarray(d['a1y'], float))
    return cp[np.isfinite(cp)]


def compute_R(train_path, post_paths, chi_max=CHI_MAX):
    if isinstance(post_paths, str):
        post_paths = [post_paths]
    a = np.loadtxt(train_path, ndmin=2)
    if a.shape[1] <= LNL_COL:
        raise ValueError(f"training file has {a.shape[1]} cols, need lnL at 0-based col {LNL_COL}")
    lnl = a[:, LNL_COL]
    cp = np.hypot(a[:, 3], a[:, 4])
    ok = np.isfinite(lnl) & np.isfinite(cp)
    lnl, cp = lnl[ok], cp[ok]
    if len(lnl) < 1000:
        raise ValueError(f"only {len(lnl)} usable training rows")
    peak = lnl.max()
    b = float(np.quantile(cp, SELFQ))
    rng = np.random.default_rng(PRIOR_SEED)
    am = rng.uniform(0, chi_max, N_PRIOR)
    c_ = rng.uniform(-1, 1, N_PRIOR)
    cpp = am * np.sqrt(1 - c_ ** 2)
    imp_tail = imp_tot = 0.0
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (cp >= lo) & (cp < hi)
        if m.sum() < MIN_BIN:
            continue
        meanL = float(np.exp(lnl[m] - peak).mean())
        ib = (cpp >= lo) & (cpp < hi)
        imp_tot += float(ib.mean()) * meanL
        imp_tail += float((ib & (cpp > b)).mean()) * meanL
    implied = imp_tail / imp_tot if imp_tot > 0 else float("nan")
    dl, np_list = [], []
    for pp in post_paths:
        cpo = load_posterior_cp(pp)
        if len(cpo) < MIN_POST:
            raise ValueError(f"{pp}: only {len(cpo)} posterior samples (< {MIN_POST})")
        dl.append(float(np.mean(cpo > b)))
        np_list.append(int(len(cpo)))
    delivered = float(np.mean(dl))              # equal-weight mean over detect reps
    R = delivered / implied if implied > 0 else float("nan")
    R_reps = [d / implied if implied > 0 else float("nan") for d in dl]
    # floor uses the MIN per-rep sample count: conservative, and unchanged for N=1
    return dict(boundary=b, implied=implied, delivered=delivered, R=R,
                n_post=int(min(np_list)), n_train=int(len(lnl)),
                n_reps=len(dl), R_per_rep=R_reps,
                R_rep_sd=float(np.std(R_reps, ddof=1)) if len(dl) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser(
        description="Severe-tail-deficit gate from RIFT's own products (training set + its CIP "
                    "posterior).  Detects SEVERE transverse deficits only; mild deficit and "
                    "healthy width are not separable in the mid-R band (see module docstring "
                    "for the measured scope and bounds).  The sample-resolution validity floor "
                    "is mandatory and has no disable option: without it the detector returns "
                    "maximum deficit exactly where it can resolve nothing.")
    ap.add_argument("training", help="composite/all.net (col 9 = lnL)")
    ap.add_argument("posterior", nargs="+",
                    help="CIP posterior samples .dat produced from that training set (header "
                         "with a1x a1y columns).  Give SEVERAL independent detect-pass "
                         "posteriors to decide on the MEAN R: single-rep R noise is "
                         "shot-dominated (~0.02 at 20k samples on mid-band events, measured), "
                         "so N reps shrink it by 1/sqrt(N).")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="R below this (and above the validity floor) fires; default %(default)s, "
                         "calibrated in the DEPLOYMENT channel (fresh-CIP posterior); changing "
                         "it voids the recorded validation")
    ap.add_argument("--floor-counts", type=float, default=FLOOR_COUNTS,
                    help="validity floor = this / n_post expected tail samples; default "
                         "%(default)s (pre-registered).  May be raised (stricter); values below "
                         "1 are refused -- the floor cannot be turned off.")
    ap.add_argument("--chi-max", type=float, default=CHI_MAX)
    ap.add_argument("--json", default=None, help="write full evaluation record here")
    args = ap.parse_args()
    if args.floor_counts < 1.0:
        sys.stderr.write("util_CIPTailDeficitGate: --floor-counts < 1 refused: the validity "
                         "floor is mandatory and cannot be effectively disabled\n")
        sys.exit(2)
    try:
        r = compute_R(args.training, args.posterior, chi_max=args.chi_max)
        if r["n_reps"] > 1:
            sys.stderr.write(f"util_CIPTailDeficitGate: {r['n_reps']} detect reps: R per rep = "
                             + " ".join(f"{x:.4f}" for x in r["R_per_rep"])
                             + f"  (sd {r['R_rep_sd']:.4f}; deciding on the mean)\n")
    except Exception as e:
        sys.stderr.write(f"util_CIPTailDeficitGate: cannot evaluate: {type(e).__name__}: {e}\n")
        sys.exit(2)
    decision, floor = decide(r["implied"], r["R"], r["n_post"],
                             threshold=args.threshold, floor_counts=args.floor_counts)
    r.update(decision=decision, floor=floor, threshold=args.threshold,
             floor_counts=args.floor_counts)
    if args.json:
        try:
            json.dump(r, open(args.json, "w"), indent=1, default=float)
        except OSError as e:
            sys.stderr.write(f"util_CIPTailDeficitGate: cannot write json: {e}\n")
    reason = {"FIRE": f"R={r['R']:.4f} < threshold={args.threshold}",
              "NO-FIRE": f"R={r['R']:.4f} >= threshold={args.threshold}",
              "ABSTAIN-FLOOR": f"implied={r['implied']:.3e} < floor={floor:.3e} "
                               f"({args.floor_counts:g}/{r['n_post']} samples): tail unresolvable"
              }[decision]
    sys.stderr.write(f"util_CIPTailDeficitGate: {decision}: {reason} "
                     f"(boundary cp>{r['boundary']:.3f}, delivered={r['delivered']:.4e}, "
                     f"implied={r['implied']:.4e}, n_post={r['n_post']})\n")
    print(f"GATE DECISION={decision} R={r['R']:.6f} implied={r['implied']:.6e} "
          f"delivered={r['delivered']:.6e} n_post={r['n_post']} floor={floor:.6e} "
          f"threshold={args.threshold}")
    sys.exit(0)


if __name__ == "__main__":
    main()
