"""
Scaling study: how the scalable interpolators (svgp, rff) behave as N grows
toward the production regime, across dimension and lnL-surface shape.

Writes one JSON line per (truth, d, N, method) cell *as it completes*, so partial
results survive an interrupt and progress is visible live::

    python -m RIFT.interpolators.jax_gp.benchmark.scaling_study --out study.jsonl

Defaults are chosen to finish in a bounded wall-clock on CPU while still reaching
production-scale N. Edit the grid via CLI flags.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .. import get_interpolator
from . import truth_functions as tf
from . import metrics as mt


def _model(method, opt_steps):
    cls = get_interpolator(method)
    if method == "svgp":
        return cls(n_opt_steps=opt_steps, n_inducing=512)
    return cls(n_opt_steps=opt_steps)


def run(out_path, dims, Ns, methods, truth_names, opt_steps, n_test, seed):
    rng = np.random.default_rng(seed)
    with open(out_path, "w") as fh:
        for d in dims:
            truths = [t for t in tf.all_truths(d, seed=seed)
                      if truth_names is None or t.name in truth_names]
            for truth in truths:
                Xt = truth.sample_domain(n_test, rng)
                yt = truth.lnL(Xt)
                for N in Ns:
                    Xtr = truth.sample_domain(N, rng)
                    ytr = truth.lnL(Xtr)
                    for method in methods:
                        row = {"truth": truth.name, "d": int(d), "N": int(N),
                               "method": method}
                        try:
                            model = _model(method, opt_steps)
                            t0 = time.perf_counter()
                            model.fit(Xtr, ytr)
                            fit_s = time.perf_counter() - t0
                            row.update(mt.evaluate(model, truth, Xt, yt, fit_s))
                        except Exception as e:
                            row["error"] = "{}: {}".format(type(e).__name__, e)
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        msg = row.get("error") or (
                            "rmse=%.3f peak=%.3f gcos=%.3f fit=%.1fs"
                            % (row["rmse"], row["peak_rmse"], row["grad_cos"], row["fit_s"]))
                        print("[done] %-18s d=%2d N=%6d %-5s | %s"
                              % (truth.name, d, N, method, msg), flush=True)
    print("WROTE", out_path, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="scaling_study.jsonl")
    p.add_argument("--dims", type=int, nargs="+", default=[8, 12])
    p.add_argument("--N", type=int, nargs="+", default=[2000, 20000])
    p.add_argument("--methods", nargs="+", default=["svgp", "rff"])
    p.add_argument("--truths", nargs="+",
                   default=["correlated_gaussian", "banana_ridge", "sharp_peak"])
    p.add_argument("--opt-steps", type=int, default=300)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    run(a.out, a.dims, a.N, a.methods, a.truths, a.opt_steps, a.n_test, a.seed)


if __name__ == "__main__":
    main()
