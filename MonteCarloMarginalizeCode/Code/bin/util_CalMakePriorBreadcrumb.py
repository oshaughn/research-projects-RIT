#! /usr/bin/env python
"""
util_CalMakePriorBreadcrumb.py

Write a VALID "prior" calibration breadcrumb (RIFT.calmarg.breadcrumbs) whose learned proposal
IS the broad prior (proposal == prior).  Seeding an ILE run from it
(--calibration-proposal-breadcrumb) therefore draws cal realizations from the prior with ZERO
importance weights -- i.e. it is exactly equivalent to the cold prior draws, but as a file
that LOADS cleanly.

Use it as the iteration-0 placeholder `cal_consolidated_-1.npz` so that an ILE binary which
does NOT guard against an empty (0-byte) placeholder will not crash on it (EOFError).  Newer
util_RIFT_pseudo_pipe.py already writes this automatically; this script lets you (re)generate
the placeholder for an ALREADY-built run directory in place -- no rebuild, no re-run:

    util_CalMakePriorBreadcrumb.py --calibration-envelope-directory rundir/cal_env \\
        --ifo H1 --ifo L1 --ifo V1 --fmin 10 --fmax 2047 --calibration-spline-count 10 \\
        --output rundir/cal_consolidated_-1.npz

fmin/fmax/spline-count must match the wide-ILE cal settings (so the node dimension lines up:
dim = 2 * spline-count * n_ifo).  Exact frequency values are not critical -- iteration 0 is a
cold start refined later -- but the IFO list and spline count MUST match.
"""
from __future__ import print_function

import sys
import argparse

import RIFT.calmarg.generate_realizations as genr
import RIFT.calmarg.breadcrumbs as breadcrumbs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration-envelope-directory", required=True,
                   help="Directory with per-IFO envelope files <IFO>.txt (the cal prior).")
    p.add_argument("--ifo", action="append", required=True,
                   help="Detector (repeatable), IN THE NODE-BLOCK ORDER the wide ILE uses "
                        "(== the IFO order of the analysis).")
    p.add_argument("--fmin", type=float, default=20.0, help="Spline fmin (all IFOs unless --fmin-ifo).")
    p.add_argument("--fmin-ifo", action="append", default=[],
                   help="Per-detector fmin override, IFO=fmin (repeatable).")
    p.add_argument("--fmax", type=float, required=True, help="Spline fmax (~ srate/2 - 1).")
    p.add_argument("--calibration-spline-count", type=int, default=10,
                   help="Spline nodes per detector (MUST match the wide ILE --calibration-spline-count).")
    p.add_argument("--output", required=True, help="Output breadcrumb path (e.g. cal_consolidated_-1.npz).")
    opts = p.parse_args(argv)

    fmin_ifo = {}
    for s in opts.fmin_ifo:
        k, v = s.split("=")
        fmin_ifo[k] = float(v)

    cal = genr.prior_cal_breadcrumb_dict(
        opts.calibration_envelope_directory, list(opts.ifo), opts.fmin, opts.fmax,
        opts.calibration_spline_count, fmin_ifo=(fmin_ifo or None))
    breadcrumbs.save(opts.output, cal=cal, meta=dict(placeholder=True, iteration=-1,
                                                     dets=list(opts.ifo)))
    print("util_CalMakePriorBreadcrumb: wrote prior placeholder {} (dim {}, dets {})".format(
        opts.output, cal["proposal_mean"].shape[0], list(opts.ifo)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
