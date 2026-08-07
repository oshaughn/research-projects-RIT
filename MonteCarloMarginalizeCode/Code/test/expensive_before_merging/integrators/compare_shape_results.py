#!/usr/bin/env python
"""
compare_shape_results.py BASE.json CANDIDATE.json [--strict-samplers ...]

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


def classify(b, c):
    """Return (verdict, note) for one base/candidate record pair.

    SINGLE SOURCE OF TRUTH for what counts as a regression.  confirm_regressions.py
    imports this: it previously reimplemented only the PASS->non-PASS case and was blind to
    REGRESSION(metrics), so a real metric regression (measured: n_eff 448->210) produced
    "no blocking regressions to confirm" and exited 0.  Two copies of this logic will always
    drift; there is now one.
    """
    if c is None and b is not None:
        # The candidate produced NO record for a row the base did.  That is a regression, not a
        # bookkeeping curiosity: a candidate that crashes before emitting a result would otherwise
        # be classified ONLY-IN-BASE, never reach confirmation, and exit the gate successfully --
        # bypassing the fail-closed rerun logic entirely.
        return "REGRESSION(missing-in-candidate)", "candidate produced no record for this row"
    if b is None:
        return "ONLY-IN-CANDIDATE", ""
    st_b, _ = evaluate(b)
    st_c, why_c = evaluate(c)
    sb, sc = _summ(b), _summ(c)
    verdict, note = "OK", ""
    if st_b == "PASS" and st_c != "PASS":
        # includes healthy->STARVED: candidate lost the efficiency the
        # base had on this target -> regression
        verdict = "REGRESSION(pass->{})".format(st_c.lower())
        note = "; ".join(why_c)
    elif st_b != "PASS" and st_c == "PASS":
        verdict = "IMPROVED({}->pass)".format(st_b.lower())
    elif st_b == "STARVED" and st_c == "STARVED":
        verdict = "BOTH-STARVED"
    elif st_b == "STARVED" and st_c in ("FAIL", "ERROR"):
        # base gave no shape information here; candidate at least reaches
        # testability (or crashes) -- flag, don't block
        verdict = "NEWLY-TESTABLE-" + st_c
        note = "; ".join(why_c)
    elif st_b in ("FAIL", "ERROR") and st_c != "PASS":
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
    return verdict, note


def is_blocking(verdict, kind, strict):
    return verdict.startswith("REGRESSION") and kind in strict


def blocking_keys(base, cand, strict):
    """Every (kind, target) the gate would BLOCK on -- both regression flavours."""
    out = []
    for k in sorted(set(base) | set(cand)):
        v, _ = classify(base.get(k), cand.get(k))
        if is_blocking(v, k[0], strict):
            out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("candidate")
    # The warm/sequential kinds are STRICT: they exist to catch silent wrong answers, so a
    # regression there must block.  Note the intended asymmetry on first merge -- portfolio_seq
    # FAILs on a base without clear_warm_state and PASSes here, i.e. IMPROVED (non-blocking).
    # Its value is forward-looking: once this is the base, re-breaking the reset blocks.
    # ENFORCEMENT.  Given both checkouts, a blocking regression is re-tested at fresh seeds
    # before it is allowed to fail the gate, and THIS script's exit code reflects the confirmed
    # verdict.  Without these the script only reports, and confirmation is advisory -- which is
    # how the first version shipped: documented in the runner but never actually invoked.
    ap.add_argument("--confirm-base-checkout", default=None,
                    help="with --confirm-cand-checkout: re-test blocking rows at fresh seeds")
    ap.add_argument("--confirm-cand-checkout", default=None)
    ap.add_argument("--confirm-repeats", type=int, default=5)
    ap.add_argument("--confirm-jobs", type=int, default=4)
    ap.add_argument("--strict-samplers",
                    default="AV,GMM,portfolio_warm,portfolio_seq,portfolio_seq_nobs")
    opts = ap.parse_args()
    strict = set(x.strip() for x in opts.strict_samplers.split(","))

    with open(opts.base) as fh:
        base = {_key(r): r for r in json.load(fh)}
    with open(opts.candidate) as fh:
        cand = {_key(r): r for r in json.load(fh)}

    n_block = 0
    rows = []
    for k in sorted(set(base) | set(cand)):
        verdict, note = classify(base.get(k), cand.get(k))
        blocking = is_blocking(verdict, k[0], strict)
        if blocking:
            n_block += 1
        rows.append((k, verdict + ("  <-- BLOCKS MERGE" if blocking else ""), note))

    for (kind, tgt), verdict, note in rows:
        print("{:<10s} {:<16s} {} {}".format(kind, tgt, verdict,
                                             ("[" + note + "]") if note else ""))
    print("# blocking regressions (strict={}): {}".format(sorted(strict), n_block))
    if not n_block:
        return 0
    if not (opts.confirm_base_checkout and opts.confirm_cand_checkout):
        print("# NOT CONFIRMED AT FRESH SEEDS: pass --confirm-base-checkout/--confirm-cand-checkout\n"
              "#   to re-test these rows before treating them as real.  Every threshold here is a\n"
              "#   hard cut on a stochastic quantity, so a single blocking row is a hypothesis.")
        return 1
    import confirm_regressions
    print("\n# re-testing {} blocking row(s) at {} fresh seeds per arm".format(
        n_block, opts.confirm_repeats))
    return confirm_regressions.main([
        opts.base, opts.candidate,
        "--base-checkout", opts.confirm_base_checkout,
        "--cand-checkout", opts.confirm_cand_checkout,
        "--repeats", str(opts.confirm_repeats),
        "--jobs", str(opts.confirm_jobs),
        "--strict-samplers", opts.strict_samplers])


if __name__ == "__main__":
    sys.exit(main())
