"""
Benchmark harness: sweep {method} x {N} x {truth} at a fixed dimension and emit
a results table.

Usage (standalone)::

    python -m RIFT.interpolators.jax_gp.benchmark.harness --d 8 --N 2000 8000 \
        --methods rff exact

The default method set is ``rff`` and ``exact`` (Phase 1).  ``svgp`` is added in
Phase 2 and is picked up automatically once available.
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import get_interpolator
from . import truth_functions as tf
from . import metrics as mt


def _make_model(method, opt_steps=None):
    cls = get_interpolator(method)
    if opt_steps is not None:
        return cls(n_opt_steps=opt_steps)
    return cls()


def run_benchmark(methods, d, Ns, n_test=2000, seed=0, truths=None,
                  opt_steps=None, truth_names=None):
    """Return a list of result rows (one per method x N x truth).

    ``opt_steps`` overrides each method's optimization-step count (useful for
    quick smoke runs).  ``truth_names`` restricts to a subset of truth functions.
    """
    rng = np.random.default_rng(seed)
    truths = tf.all_truths(d, seed=seed) if truths is None else truths
    if truth_names is not None:
        truths = [t for t in truths if t.name in truth_names]
    rows = []
    for truth in truths:
        Xt = truth.sample_domain(n_test, rng)
        yt = truth.lnL(Xt)
        for N in Ns:
            Xtr = truth.sample_domain(N, rng)
            ytr = truth.lnL(Xtr)
            for method in methods:
                try:
                    model = _make_model(method, opt_steps=opt_steps)
                    model, fit_s = mt.timed_fit(model, Xtr, ytr)
                    m = mt.evaluate(model, truth, Xt, yt, fit_s)
                    m.update({"method": method, "truth": truth.name, "N": N, "d": d})
                    rows.append(m)
                except Exception as e:  # keep the sweep going; record the failure
                    rows.append({"method": method, "truth": truth.name, "N": N,
                                 "d": d, "error": "{}: {}".format(type(e).__name__, e)})
    return rows


def format_table(rows):
    cols = ["method", "truth", "N", "rmse", "peak_rmse", "grad_cos",
            "grad_relerr", "fit_s", "pred_s"]
    head = "{:<7} {:<20} {:>7} {:>9} {:>9} {:>8} {:>10} {:>8} {:>8}".format(*cols)
    lines = [head, "-" * len(head)]
    for r in rows:
        if "error" in r:
            lines.append("{:<7} {:<20} {:>7}  ERROR {}".format(
                r["method"], r["truth"], r["N"], r["error"]))
            continue
        lines.append("{:<7} {:<20} {:>7} {:>9.4f} {:>9.4f} {:>8.4f} {:>10.4f} "
                     "{:>8.2f} {:>8.3f}".format(
                         r["method"], r["truth"], r["N"], r["rmse"], r["peak_rmse"],
                         r["grad_cos"], r["grad_relerr"], r["fit_s"], r["pred_s"]))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--N", type=int, nargs="+", default=[2000, 8000])
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--methods", nargs="+", default=["rff", "exact"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opt-steps", type=int, default=None,
                   help="override per-method optimization steps (quick runs)")
    p.add_argument("--truths", nargs="+", default=None,
                   help="restrict to named truth functions")
    args = p.parse_args(argv)
    rows = run_benchmark(args.methods, args.d, args.N, n_test=args.n_test,
                         seed=args.seed, opt_steps=args.opt_steps,
                         truth_names=args.truths)
    print(format_table(rows))
    return rows


if __name__ == "__main__":
    main()
