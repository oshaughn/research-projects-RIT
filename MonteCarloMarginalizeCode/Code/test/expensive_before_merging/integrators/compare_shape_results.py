#!/usr/bin/env python
"""
compare_shape_results.py BASE.json CANDIDATE.json [--strict-samplers AV,GMM]

Compare two shape_recovery.py --json outputs (same preset/seeds!) run on a
base branch and a candidate branch.  Exit 1 iff a strict-sampler run
REGRESSES: PASS on base -> FAIL on candidate, or a shape metric worsens
beyond tolerance.  Pre-existing failures (FAIL on both) are reported but do
not block; improvements are celebrated.
"""
from __future__ import print_function

import argparse
import json
import sys

import numpy as np

from shape_recovery import evaluate

# metric-worsening tolerances (candidate - base), applied only when both pass
TOL_WORSE = dict(js=0.005, mean_pull=0.05, width_dev=0.05, corr=0.05,
                 bias_ln=0.10, neff_frac=0.5)


def _key(r):
    return (r["kind"], r["target"])


def _summ(r):
    if r.get("error"):
        return None
    return dict(js=max(r["js"]),
                mean_pull=max(abs(p) for p in r["mean_pull"]),
                width_dev=max(abs(w - 1.0) for w in r["width_ratio"]),
                corr=r["corr_diff_max"],
                bias_ln=abs(r["bias_ln"]),
                neff=r["n_eff"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("candidate")
    ap.add_argument("--strict-samplers", default="AV,GMM")
    opts = ap.parse_args()
    strict = set(x.strip() for x in opts.strict_samplers.split(","))

    with open(opts.base) as fh:
        base = {_key(r): r for r in json.load(fh)}
    with open(opts.candidate) as fh:
        cand = {_key(r): r for r in json.load(fh)}

    n_block = 0
    rows = []
    for k in sorted(set(base) | set(cand)):
        b, c = base.get(k), cand.get(k)
        if b is None or c is None:
            rows.append((k, "ONLY-IN-" + ("CANDIDATE" if b is None else "BASE"), ""))
            continue
        ok_b, _ = evaluate(b)
        ok_c, why_c = evaluate(c)
        sb, sc = _summ(b), _summ(c)
        verdict, note = "OK", ""
        if ok_b and not ok_c:
            verdict = "REGRESSION(pass->fail)"
            note = "; ".join(why_c)
        elif not ok_b and ok_c:
            verdict = "IMPROVED(fail->pass)"
        elif not ok_b and not ok_c:
            verdict = "PREEXISTING-FAIL"
        elif sb and sc:
            worse = []
            for m, tol in TOL_WORSE.items():
                if m == "neff_frac":
                    if sc["neff"] < TOL_WORSE["neff_frac"] * sb["neff"]:
                        worse.append("n_eff {:.0f}->{:.0f}".format(sb["neff"], sc["neff"]))
                elif sc[m] - sb[m] > tol:
                    worse.append("{} {:.3f}->{:.3f}".format(m, sb[m], sc[m]))
            if worse:
                verdict = "REGRESSION(metrics)"
                note = "; ".join(worse)
        blocking = verdict.startswith("REGRESSION") and k[0] in strict
        if blocking:
            n_block += 1
        rows.append((k, verdict + ("  <-- BLOCKS MERGE" if blocking else ""), note))

    for (kind, tgt), verdict, note in rows:
        print("{:<10s} {:<16s} {} {}".format(kind, tgt, verdict,
                                             ("[" + note + "]") if note else ""))
    print("# blocking regressions (strict={}): {}".format(sorted(strict), n_block))
    return 1 if n_block else 0


if __name__ == "__main__":
    sys.exit(main())
