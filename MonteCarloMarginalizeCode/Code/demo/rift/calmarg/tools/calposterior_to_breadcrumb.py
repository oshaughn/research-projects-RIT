#!/usr/bin/env python3
"""Warm-start a cal PILOT from a finished run's recovered calibration posterior.

A normal pilot starts COLD: its iteration-0 seed `cal_consolidated_-1.npz` is the prior
placeholder (proposal == prior, zero weights), so wide_0 draws cal realizations blindly.
But a run with `--calibration-export-posterior` writes, at the final fairdraw, a recovered
cal posterior `<output>_<event>_cal.dat` whose columns `cal_<IFO>_amp_<k>` / `cal_<IFO>_phase_<k>`
are SAMPLES of exactly the spline-node vector the breadcrumb is a Gaussian over.

So we can fit a Gaussian (mean, cov) to those samples and write it as the pilot's
iteration-0 breadcrumb -> the pilot starts WARM, seeded from where the brute-force run
already learned the cal posterior sits.  Drop the output in as
`rundir_pp_pilot/cal_consolidated_-1.npz` before submitting the pilot.

We reuse the placeholder breadcrumb for prior_mean/prior_sigma/node_log_f/n_nodes_amp/dets
(the exact node structure) and only overwrite proposal_mean/proposal_cov -- so the node
ordering is guaranteed to match.  Node-vector order (breadcrumbs.py): per det, [amp_0..amp_{Na-1},
phase_0..phase_{Na-1}], concatenated over dets in order.

Usage:
    calposterior_to_breadcrumb.py --cal-dat <run>_<event>_cal.dat \\
        --placeholder rundir_pp_pilot/cal_consolidated_-1.npz \\
        --output rundir_pp_pilot/cal_consolidated_-1.npz [--cov-inflate 1.5]

NOTE: validate against a real _cal.dat the first time (the column-header parsing below
assumes a whitespace table whose header names columns `cal_<IFO>_amp_<k>` etc.; adjust the
`read_cal_dat` parser if the actual format differs).
"""
import argparse, re, sys
import numpy as np
import RIFT.calmarg.breadcrumbs as breadcrumbs


def read_cal_dat(path):
    """Return (colnames, data[N, ncol]).  Handles a leading '#'-commented header naming
    the columns, else a first non-comment header row of names."""
    names, rows = None, []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            toks = s.lstrip("#").split()
            if names is None and any(re.search(r"cal_.*_(amp|phase)_\d", t) for t in toks):
                names = toks                      # header line (commented or not)
                continue
            if s.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                continue
    if names is None:
        raise SystemExit("ERROR: no header naming cal_<IFO>_(amp|phase)_<k> columns found in " + path)
    return names, np.asarray(rows, dtype=float)


def build_node_matrix(names, data, dets, n_nodes_amp):
    """Assemble (Nsamples, dim) in breadcrumb order: per det [amp_0..,phase_0..], over dets."""
    idx = {n: i for i, n in enumerate(names)}
    cols = []
    for det in dets:
        for kind in ("amp", "phase"):
            for k in range(n_nodes_amp):
                key = "cal_%s_%s_%d" % (det, kind, k)
                if key not in idx:
                    raise SystemExit("ERROR: column %s missing from cal.dat header" % key)
                cols.append(idx[key])
    return data[:, cols]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cal-dat", required=True, help="recovered cal posterior <out>_<event>_cal.dat")
    p.add_argument("--placeholder", required=True, help="cold cal_consolidated_-1.npz (for prior+structure)")
    p.add_argument("--output", required=True, help="warm breadcrumb to write (overwrite the pilot's -1.npz)")
    p.add_argument("--cov-inflate", type=float, default=1.0, help="multiply fitted covariance (broaden proposal)")
    opts = p.parse_args(argv)

    bc = breadcrumbs.load(opts.placeholder)
    cal = bc["cal"]
    dets, na = list(cal["dets"]), int(cal["n_nodes_amp"])
    dim = 2 * na * len(dets)

    names, data = read_cal_dat(opts.cal_dat)
    X = build_node_matrix(names, data, dets, na)
    if X.shape[1] != dim:
        raise SystemExit("ERROR: assembled %d node cols, breadcrumb dim is %d" % (X.shape[1], dim))

    cal["proposal_mean"] = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cal["proposal_cov"] = cov * opts.cov_inflate + np.eye(dim) * 1e-12   # tiny ridge for PD
    breadcrumbs.save(opts.output, cal=cal,
                     meta=dict(warm_start=True, source=opts.cal_dat, n_samples=int(X.shape[0]),
                               cov_inflate=opts.cov_inflate, iteration=-1))
    sd = np.sqrt(np.diag(cov))
    print("WARM breadcrumb -> %s" % opts.output)
    print("  fit from %d samples, dim=%d (%d dets x %d amp+phase nodes)" % (X.shape[0], dim, len(dets), na))
    print("  proposal sigma vs prior sigma (median ratio): %.3f"
          % np.median(sd / np.asarray(cal["prior_sigma"])))


if __name__ == "__main__":
    main()
