#!/usr/bin/env python
"""
util_PickPilotPoint.py

Cherry-pick the best intrinsic grid point(s) after iteration 0, to run as warm-start
PILOT(s).  This is the disk- and fit-safe way to seed later ILE evaluations:

  * We must NOT --save-samples the whole grid (disk blows up fast), and
  * we must NOT pick pilots at random (a poor point makes a poor extrinsic proposal, and
    -- more importantly -- the CIP fit that builds the next grid needs good coverage, so
    random subsetting is dangerous).

So: after a cheap iteration-0 ILE (per-point marginal lnL only, no sample dump), pick the
TOP-k points by lnL and emit them as a small sub-grid.  A follow-up ILE runs only those
few points WITH --save-samples (~tens of KB each) to produce the pilot extrinsic
proposal, which then warm-starts the rest of the grid (via
integrate_likelihood_extrinsic_batchmode --sampler-warmstart-samples).

Usage:
  util_PickPilotPoint.py --grid overlap-grid-0.xml.gz --output-prefix ile_0 \
      --top-k 1 --out pilot-grid.xml.gz
  # then run ILE on pilot-grid.xml.gz with --save-samples, and warm-start iteration 1+
  # from the pilot's saved extrinsic samples.

lnL is read from the per-point ILE .dat outputs (<prefix>_<i>_.dat, column 10 = marginal
lnL), or from a single --net composite file with an explicit --lnL-column.
"""
from __future__ import print_function
import argparse
import os
import numpy as np


def _lnL_from_dat(prefix, i):
    path = "{}_{}_.dat".format(prefix, i)
    if not os.path.exists(path):
        return None
    try:
        row = np.atleast_2d(np.loadtxt(path))
        # ILE .dat layout: idx m1 m2 s1x s1y s1z s2x s2y s2z lnL sigma ntotal neff
        return float(row[0, 9])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="iteration-0 intrinsic grid xml")
    ap.add_argument("--output-prefix", default=None, help="ILE --output-file prefix; reads <prefix>_<i>_.dat for lnL")
    ap.add_argument("--net", default=None, help="alternative: a composite/net file with one row per grid point")
    ap.add_argument("--lnL-column", type=int, default=9, help="0-based lnL column in --net (default 9)")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--out", required=True, help="output pilot sub-grid xml")
    args = ap.parse_args()

    import RIFT.lalsimutils as lalsimutils
    P_list = lalsimutils.xml_to_ChooseWaveformParams_array(args.grid)

    lnL = np.full(len(P_list), -np.inf)
    if args.net:
        dat = np.atleast_2d(np.loadtxt(args.net))
        m = min(len(P_list), dat.shape[0])
        lnL[:m] = dat[:m, args.lnL_column]
    elif args.output_prefix:
        for i in range(len(P_list)):
            v = _lnL_from_dat(args.output_prefix, i)
            if v is not None:
                lnL[i] = v
    else:
        raise SystemExit("provide --output-prefix or --net for lnL values")

    finite = np.isfinite(lnL)
    if not np.any(finite):
        raise SystemExit("no finite lnL found; cannot pick a pilot")
    order = np.argsort(lnL)[::-1]
    order = [i for i in order if np.isfinite(lnL[i])][:max(1, args.top_k)]
    print("Picked {} pilot point(s) by lnL: {}".format(
        len(order), [(int(i), round(float(lnL[i]), 1)) for i in order]))

    P_out = [P_list[i] for i in order]
    lalsimutils.ChooseWaveformParams_array_to_xml(P_out, fname=args.out.replace('.xml.gz', '').replace('.xml', ''))
    print("Wrote pilot sub-grid ->", args.out)


if __name__ == "__main__":
    main()
