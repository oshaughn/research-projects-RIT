#!/usr/bin/env python
"""escaped_mass_report.py -- tabulate / ROC the output of escaped_mass_study.py.

    python escaped_mass_report.py run1.json [run2.json ...]

Candidate statistics compared (all for the WARM-STARTED member, index 0):
    esc_cum    cumulative escaped_mass over the whole run
    esc_early  escaped_mass in the FIRST chunk the member had a density in
    esc_first3 max over the first 3 chunks
    lo_share   1 - weight_share  (the cheap comparator: a useless warm member stops
               attracting weight, so a LOW share is the alarm)
    inv_neff   1000/n_eff        (null comparator: an efficiency signal, not a support signal)

Two questions are scored separately, because they have different answers:
  * SEED-MISMATCH ROC  -- can the statistic tell offset>0 from offset=0?
  * BIAS ROC           -- can it tell |lnZ bias| > BIAS_MATERIAL from |bias| <= BIAS_MATERIAL?
    This is the one that matters operationally: a detector that fires only on runs that were
    going to be fine is a nuisance alarm, and one that stays quiet on biased runs is worthless.
"""
from __future__ import print_function

import json
import sys

import numpy as np

BIAS_MATERIAL = 0.5    # nats; the scale at which a lnZ error starts to matter downstream


def _firstk(r, k, m=0):
    h = r.get("esc_hist") or []
    col = [row[m] for row in h[:k] if len(row) > m and np.isfinite(row[m])]
    return float(np.max(col)) if col else np.nan


STATS = {
    "esc_cum":    lambda r: r["esc_warm"],
    "esc_early":  lambda r: r["esc_early_warm"],
    "esc_first3": lambda r: _firstk(r, 3),
    "lo_share":   lambda r: 1.0 - r["share_warm"],
    "inv_neff":   lambda r: 1000.0 / max(r["n_eff"], 1e-9),
}


def auc(pos, neg):
    """Mann-Whitney AUC: P(stat_pos > stat_neg), ties counted as 1/2."""
    pos = np.asarray([x for x in pos if np.isfinite(x)], dtype=float)
    neg = np.asarray([x for x in neg if np.isfinite(x)], dtype=float)
    if not len(pos) or not len(neg):
        return np.nan
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def qs(a):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    if not len(a):
        return (np.nan,) * 4
    return (float(np.median(a)), float(np.percentile(a, 10)),
            float(np.percentile(a, 90)), float(np.max(a)))


def main(paths):
    rs = []
    for p in paths:
        rs += json.load(open(p))
    n_err = sum(1 for r in rs if r.get("error"))
    rs = [r for r in rs if not r.get("error")]
    dims = sorted(set(r["ndim"] for r in rs))
    arms = sorted(set(r["arm"] for r in rs))
    offs = sorted(set(r["offset"] for r in rs))
    print("# {} records ({} errors dropped); dims {} arms {} offsets {}".format(
        len(rs), n_err, dims, arms, offs))

    # ---------------- 1. sensitivity table ----------------
    print("\n== SENSITIVITY: statistic and |lnZ bias| vs seed displacement ==")
    hdr = ("%-11s %2s %5s %3s | %8s %8s %8s | %9s %9s | %9s %9s | %9s | %7s" %
           ("arm", "d", "off", "N", "med bias", "|b|max", "med neff",
            "esc_cum", "(p10)", "esc_early", "(p10)", "esc_f3", "share0"))
    print(hdr); print("-" * len(hdr))
    for d in dims:
        for arm in arms:
            for off in offs:
                sel = [r for r in rs if r["ndim"] == d and r["arm"] == arm and r["offset"] == off]
                if not sel:
                    continue
                b = np.asarray([r["bias_ln"] for r in sel])
                print("%-11s %2d %5.1f %3d | %8.3f %8.2f %8.0f | %9.2e %9.2e | %9.2e %9.2e | %9.2e | %7.3f" % (
                    arm, d, off, len(sel), np.median(b), np.max(np.abs(b)),
                    np.median([r["n_eff"] for r in sel]),
                    qs([r["esc_warm"] for r in sel])[0], qs([r["esc_warm"] for r in sel])[1],
                    qs([r["esc_early_warm"] for r in sel])[0], qs([r["esc_early_warm"] for r in sel])[1],
                    qs([_firstk(r, 3) for r in sel])[0],
                    np.median([r["share_warm"] for r in sel])))
            print()

    # ---------------- 2. false-positive floor at offset 0 ----------------
    print("== FALSE-POSITIVE FLOOR at offset=0 (across independent target seeds) ==")
    hdr = "%-11s %2s %3s | %-10s %10s %10s %10s %10s" % (
        "arm", "d", "N", "stat", "median", "p90", "max", "frac>1e-3")
    print(hdr); print("-" * len(hdr))
    floors = {}
    for d in dims:
        for arm in arms:
            sel = [r for r in rs if r["ndim"] == d and r["arm"] == arm and r["offset"] == 0.0]
            if not sel:
                continue
            for name in ("esc_cum", "esc_early", "esc_first3", "lo_share"):
                v = np.asarray([STATS[name](r) for r in sel], dtype=float)
                v = v[np.isfinite(v)]
                m, p10, p90, mx = qs(v)
                floors[(d, arm, name)] = mx
                print("%-11s %2d %3d | %-10s %10.3e %10.3e %10.3e %10.2f" % (
                    arm, d, len(v), name, m, p90, mx, float(np.mean(v > 1e-3))))
            print()

    # ---------------- 3. seed-mismatch ROC (AUC vs the offset=0 control) ----------------
    print("== SEED-MISMATCH AUC: P(stat[offset] > stat[offset=0]), 0.5 = useless ==")
    names = ["esc_cum", "esc_early", "esc_first3", "lo_share", "inv_neff"]
    hdr = "%-11s %2s %5s | " % ("arm", "d", "off") + " ".join("%10s" % n for n in names)
    print(hdr); print("-" * len(hdr))
    for d in dims:
        for arm in arms:
            ctl = [r for r in rs if r["ndim"] == d and r["arm"] == arm and r["offset"] == 0.0]
            for off in offs:
                if off == 0.0:
                    continue
                sel = [r for r in rs if r["ndim"] == d and r["arm"] == arm and r["offset"] == off]
                if not sel or not ctl:
                    continue
                row = [auc([STATS[n](r) for r in sel], [STATS[n](r) for r in ctl]) for n in names]
                print("%-11s %2d %5.1f | " % (arm, d, off)
                      + " ".join("%10.3f" % x for x in row))
            print()

    # ---------------- 4. BIAS ROC: does it flag the runs that are actually WRONG? ----------
    print("== BIAS DETECTION: positives = |lnZ bias| > {} nat, pooled over offsets ==".format(
        BIAS_MATERIAL))
    hdr = "%-11s %2s | %5s %5s | " % ("arm", "d", "Npos", "Nneg") + " ".join("%10s" % n for n in names)
    print(hdr); print("-" * len(hdr))
    for d in dims:
        for arm in arms:
            sel = [r for r in rs if r["ndim"] == d and r["arm"] == arm]
            pos = [r for r in sel if abs(r["bias_ln"]) > BIAS_MATERIAL]
            neg = [r for r in sel if abs(r["bias_ln"]) <= BIAS_MATERIAL]
            if not pos or not neg:
                print("%-11s %2d | %5d %5d |  (degenerate: one class empty)" % (
                    arm, d, len(pos), len(neg)))
                continue
            row = [auc([STATS[n](r) for r in pos], [STATS[n](r) for r in neg]) for n in names]
            print("%-11s %2d | %5d %5d | " % (arm, d, len(pos), len(neg))
                  + " ".join("%10.3f" % x for x in row))
    print()

    # ---------------- 5. operating point: threshold = the measured offset=0 max ----------
    print("== OPERATING POINT: threshold = max over the 20 offset=0 controls of the SAME cell ==")
    hdr = "%-11s %2s %-10s | %10s | " % ("arm", "d", "stat", "thresh") + \
          " ".join("%6.1f" % o for o in offs if o > 0)
    print(hdr); print("-" * len(hdr))
    for d in dims:
        for arm in arms:
            for name in ("esc_cum", "esc_early", "esc_first3", "lo_share"):
                thr = floors.get((d, arm, name))
                if thr is None or not np.isfinite(thr):
                    continue
                cells = []
                for off in offs:
                    if off == 0.0:
                        continue
                    sel = [r for r in rs if r["ndim"] == d and r["arm"] == arm
                           and r["offset"] == off]
                    v = np.asarray([STATS[name](r) for r in sel], dtype=float)
                    v = v[np.isfinite(v)]
                    cells.append(float(np.mean(v > thr)) if len(v) else np.nan)
                print("%-11s %2d %-10s | %10.3e | " % (arm, d, name, thr)
                      + " ".join("%6.2f" % c for c in cells))
            print()
    print("# (numbers are TRUE-POSITIVE RATE at a threshold with 0/20 false positives by "
          "construction; 1-sided 95% upper bound on that FP rate is ~0.14)")

    # ---------------- 6. BIAS operating point: the question that actually matters ----------
    # Threshold set on the BENIGN runs (|bias| <= BIAS_MATERIAL) at a 10% false-alarm rate; then
    # report what fraction of the genuinely-wrong runs it catches, and how wrong they were.
    print("\n== BIAS OPERATING POINT: threshold = p90 of the statistic over runs with "
          "|bias| <= {} nat ==".format(BIAS_MATERIAL))
    hdr = ("%-11s %2s %-10s | %10s | %5s %5s | %6s | %10s %10s" %
           ("arm", "d", "stat", "thr(FP=.10)", "Npos", "Nneg", "TPR", "med|b| hit", "med|b| miss"))
    print(hdr); print("-" * len(hdr))
    for d in dims:
        for arm in arms:
            sel = [r for r in rs if r["ndim"] == d and r["arm"] == arm]
            pos = [r for r in sel if abs(r["bias_ln"]) > BIAS_MATERIAL]
            neg = [r for r in sel if abs(r["bias_ln"]) <= BIAS_MATERIAL]
            if len(pos) < 5 or len(neg) < 5:
                print("%-11s %2d %-10s |  (too few in one class: %d pos / %d neg)" % (
                    arm, d, "-", len(pos), len(neg)))
                continue
            for name in ("esc_cum", "esc_early", "esc_first3", "lo_share", "inv_neff"):
                vneg = np.asarray([STATS[name](r) for r in neg], dtype=float)
                vneg = vneg[np.isfinite(vneg)]
                thr = float(np.percentile(vneg, 90))
                vpos = np.asarray([STATS[name](r) for r in pos], dtype=float)
                bpos = np.abs([r["bias_ln"] for r in pos])
                hit = vpos > thr
                print("%-11s %2d %-10s | %11.3e | %5d %5d | %6.2f | %10.2f %10.2f" % (
                    arm, d, name, thr, len(pos), len(neg), float(np.mean(hit)),
                    float(np.median(bpos[hit])) if np.any(hit) else float("nan"),
                    float(np.median(bpos[~hit])) if np.any(~hit) else float("nan")))
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
