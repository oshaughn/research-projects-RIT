#!/usr/bin/env python
"""
probe_portfolio_optin_flags.py -- shape-gate probe for the portfolio OPT-IN flags.

`RIFT/integrators/TESTING.md` requires that a change behind an opt-in flag ALSO be probed with the
flag ON: the default-path merge gate necessarily shows bitwise-identical results for opt-in code, so
it proves nothing about that code.  This probe covers the two opt-in portfolio features:

    portfolio_adaptive_alloc  (adaptive-probe draw allocation)
    portfolio_weight_clip     (truncated IS on the GMM proposal-fit input)

Method: reuse the merge-gate suite as a library (per shape_recovery.py's docstring) so the targets,
truth pools, metrics and pass thresholds are IDENTICAL to the gate.  We monkey-patch
`build_sampler` to switch the flags on for portfolio runs, and run IN-PROCESS (jobs=1): the gate's
multiprocessing path uses spawn, which would not carry the patch into workers.

Each configuration is scored with the gate's own `evaluate()`, so a row that PASSes here passes by
exactly the gate's criteria.

Usage (CPU, like the gate):
    export PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
    python probe_portfolio_optin_flags.py [--dims 2,4] [--ncomps 1,3] [--seeds 303]
"""
from __future__ import print_function
import argparse
import os
import sys

# The merge-gate WRAPPER (run_shape_recovery.sh) exports these; library mode does NOT.  Without
# them you silently import the INSTALLED RIFT (not the checkout under test) and/or hit the
# cupy-without-a-device path -- both yield confident, meaningless numbers.  A valid probe
# reproduces the gate's ABSOLUTE values row-for-row.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '1')
_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import shape_recovery as SR


def patched_build(flags):
    """Return a build_sampler that switches the opt-in flags on for portfolio samplers."""
    orig = SR.build_sampler

    def build(kind, target, n_chunk):
        s = orig(kind, target, n_chunk)
        if kind == "portfolio":
            for k, v in flags.items():
                setattr(s, k, v)
        return s
    return build


def run_config(label, flags, jobs_spec, nmax_per_dim, neff, run_seed):
    """Run the portfolio rows for one flag configuration; return list of (job, record)."""
    SR.build_sampler = patched_build(flags)
    out = []
    for (d, nc, ts) in jobs_spec:
        target = SR.MixtureTarget(d, nc, ts)
        rec = SR.run_one("portfolio", target, nmax_per_dim * d, neff, seed=run_seed)
        verdict = SR.evaluate(rec)
        out.append(((d, nc, ts), rec, verdict))
        print("  {:22s} d{}_n{}_s{}  n_eff={:8.0f}  lnI-lnZ={:+.4f}  {}".format(
            label, d, nc, ts,
            float(rec.get("n_eff", float("nan"))),
            float(rec.get("bias_ln", float("nan"))),
            verdict if isinstance(verdict, str) else verdict[0]))
        sys.stdout.flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="2,4")
    ap.add_argument("--ncomps", default="1,3")
    ap.add_argument("--seeds", default="303")
    ap.add_argument("--nmax-per-dim", type=int, default=None)
    ap.add_argument("--neff", type=int, default=None)
    ap.add_argument("--run-seed", type=int, default=987654)
    args = ap.parse_args()

    cfg = dict(SR.PRESETS["standard"])
    nmax_per_dim = args.nmax_per_dim or cfg["nmax_per_dim"]
    neff = args.neff or cfg["neff"]
    jobs_spec = [(int(d), int(nc), int(ts))
                 for d in args.dims.split(",")
                 for nc in args.ncomps.split(",")
                 for ts in args.seeds.split(",")]

    configs = [
        ("flags OFF (default)", {}),
        ("adaptive_alloc ON", {"portfolio_adaptive_alloc": True}),
        ("weight_clip ON", {"portfolio_weight_clip": 1.0}),
        ("adaptive+clip ON", {"portfolio_adaptive_alloc": True, "portfolio_weight_clip": 1.0}),
    ]
    print("# portfolio opt-in flag probe: {} targets x {} configs "
          "(nmax_per_dim={}, neff={})".format(len(jobs_spec), len(configs), nmax_per_dim, neff))

    results = {}
    for label, flags in configs:
        print("== {} ==".format(label))
        results[label] = run_config(label, flags, jobs_spec, nmax_per_dim, neff, args.run_seed)

    # Summary: the opt-in paths must not be WORSE than the default path on the gate's own verdict.
    print("\n# SUMMARY (verdict per target; opt-in must not regress vs flags OFF)")
    base = {k: (v, d) for k, v, d in results["flags OFF (default)"]}
    bad = 0
    for label, _ in configs[1:]:
        for key, rec, verdict in results[label]:
            b_rec, b_verdict = base[key]
            vs = verdict if isinstance(verdict, str) else verdict[0]
            bs = b_verdict if isinstance(b_verdict, str) else b_verdict[0]
            flag = ""
            if bs == "PASS" and vs not in ("PASS", "STARVED"):
                flag = "  <-- REGRESSION (base PASS -> {})".format(vs); bad += 1
            print("  {:22s} d{}_n{}_s{}  base={:8s} flag={:8s}{}".format(
                label, key[0], key[1], key[2], bs, vs, flag))
    print("\n# opt-in regressions: {}".format(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
