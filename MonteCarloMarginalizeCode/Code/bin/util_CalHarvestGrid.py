#! /usr/bin/env python
"""
util_CalHarvestGrid.py

Harvest the top-fraction (by lnL) intrinsic points from a RIFT *.composite file into a
small sim-xml grid, for the calibration PILOT to analyze (Option C; see
RIFT/calmarg/DESIGN_adaptive_driver.md).

The cal posterior is ~extrinsic- and ~intrinsic-independent across the high-likelihood
region, so a handful of the best-fit points suffice to learn the cal proposal.  This
writes those points as an xml grid that integrate_likelihood_extrinsic_batchmode can
analyze with --calibration-dump-responsibilities.

Composite column convention (RIFT standard, headerless whitespace):
   0:indx 1:m1 2:m2 3:a1x 4:a1y 5:a1z 6:a2x 7:a2y 8:a2z 9:lnL 10:sigma/lnL 11:ntot ...
(masses in Msun).  Override with --lnL-col / --mass-col if your composite differs.

  util_CalHarvestGrid.py --composite consolidated_3.composite \
      --output-grid cal_pilot_grid_3.xml.gz --top-fraction 0.05 --max-points 32
"""
from __future__ import division

import sys
import argparse
import numpy as np

import lal
import RIFT.lalsimutils as lalsimutils


def harvest_top_rows(dat, lnL_col=9, top_fraction=0.05, max_points=32):
    """Return the rows of `dat` (2D array) with the highest lnL: the top `top_fraction`,
    capped at `max_points` (and at least 1)."""
    dat = np.atleast_2d(dat)
    lnL = dat[:, lnL_col]
    n_keep = max(1, int(np.ceil(len(lnL) * top_fraction)))
    if max_points:
        n_keep = min(n_keep, max_points)
    order = np.argsort(lnL)[::-1][:n_keep]
    return dat[order]


def rows_to_P_list(rows, mass_col=1):
    """Build ChooseWaveformParams from composite rows (masses Msun, spins dimensionless)."""
    P_list = []
    for r in rows:
        P = lalsimutils.ChooseWaveformParams()
        P.m1 = float(r[mass_col]) * lal.MSUN_SI
        P.m2 = float(r[mass_col + 1]) * lal.MSUN_SI
        P.s1x, P.s1y, P.s1z = float(r[mass_col + 2]), float(r[mass_col + 3]), float(r[mass_col + 4])
        P.s2x, P.s2y, P.s2z = float(r[mass_col + 5]), float(r[mass_col + 6]), float(r[mass_col + 7])
        P.fmin = 0.0
        P_list.append(P)
    return P_list


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--composite", required=True, help="input *.composite file")
    p.add_argument("--output-grid", required=True, help="output sim-xml(.gz) grid")
    p.add_argument("--top-fraction", type=float, default=0.05)
    p.add_argument("--max-points", type=int, default=32,
                   help="cap on harvested points (the cal posterior is ~the same across "
                        "the high-L region, so a handful suffice)")
    p.add_argument("--lnL-col", type=int, default=9)
    p.add_argument("--mass-col", type=int, default=1,
                   help="column index of m1 (m2..spins follow)")
    opts = p.parse_args(argv)

    dat = np.loadtxt(opts.composite)
    rows = harvest_top_rows(dat, lnL_col=opts.lnL_col, top_fraction=opts.top_fraction,
                            max_points=opts.max_points)
    P_list = rows_to_P_list(rows, mass_col=opts.mass_col)
    fname = opts.output_grid
    if fname.endswith(".xml.gz"):
        fname = fname[:-len(".xml.gz")]
    elif fname.endswith(".xml"):
        fname = fname[:-len(".xml")]
    lalsimutils.ChooseWaveformParams_array_to_xml(P_list, fname=fname)
    print(" Harvested %d/%d points (top %.1f%%, lnL in [%.2f, %.2f]) -> %s.xml.gz"
          % (len(P_list), len(np.atleast_2d(dat)), 100 * opts.top_fraction,
             rows[:, opts.lnL_col].min(), rows[:, opts.lnL_col].max(), fname))
    return 0


if __name__ == "__main__":
    sys.exit(main())
