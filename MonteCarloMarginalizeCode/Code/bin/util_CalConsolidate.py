#! /usr/bin/env python
"""
util_CalConsolidate.py

Consolidation step of the adaptive calibration driver (Option C; see
RIFT/calmarg/DESIGN_adaptive_driver.md).  This is the per-iteration BARRIER between
the pilot jobs of iteration N and the wide ILE jobs of iteration N+1.

Combines several pilot proposal breadcrumbs (one per harvested high-L point / pilot
job) into a single consolidated cal proposal breadcrumb, via a precision-weighted
(moment-matched) combination of the per-pilot Gaussians.  The consolidated breadcrumb
is what wide_{N+1} consumes through --calibration-proposal-breadcrumb.

  util_CalConsolidate.py --breadcrumb pilot_a.npz --breadcrumb pilot_b.npz \
      --output-breadcrumb cal_consolidated.npz

With a single input breadcrumb this is a pass-through (copy), which is the common case
once cal is learned and only one pilot remains active.
"""
from __future__ import division

import sys
import argparse

from RIFT.calmarg import pilot, breadcrumbs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--breadcrumb", action="append", required=True,
                   help="input pilot proposal breadcrumb .npz (repeatable)")
    p.add_argument("--output-breadcrumb", required=True,
                   help="output consolidated proposal breadcrumb .npz")
    p.add_argument("--iteration", type=int, default=None,
                   help="iteration number (recorded in breadcrumb meta)")
    opts = p.parse_args(argv)

    if len(opts.breadcrumb) == 1:
        # pass-through: re-save with consolidated meta so downstream is uniform
        g = breadcrumbs.load(opts.breadcrumb[0])
        meta = dict(g.get("meta", {})); meta["consolidated_from"] = 1
        if opts.iteration is not None:
            meta["iteration"] = int(opts.iteration)
        breadcrumbs.save(opts.output_breadcrumb, cal=g["cal"], meta=meta)
        print(" Consolidation: single pilot -> pass-through to %s" % opts.output_breadcrumb)
        return 0

    out = pilot.consolidate(opts.breadcrumb, out_path=None)
    meta = dict(consolidated_from=len(opts.breadcrumb))
    if opts.iteration is not None:
        meta["iteration"] = int(opts.iteration)
    breadcrumbs.save(opts.output_breadcrumb, cal=out, meta=meta)
    print(" Consolidated %d pilot proposals -> %s" % (len(opts.breadcrumb), opts.output_breadcrumb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
