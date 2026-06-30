#! /usr/bin/env python
"""
util_CalPilotFit.py

Pilot fit step of the adaptive calibration driver (Option C; see
RIFT/calmarg/DESIGN_adaptive_driver.md).

Reads one or more calibration-responsibility dumps produced by
`integrate_likelihood_extrinsic_batchmode --calibration-dump-responsibilities`
(each an .npz with: nodes (n_cal, dim), log_resp (n_cal,), prior_mean, prior_sigma,
node_log_f, n_nodes_amp, dets) and fits a (tempered) Gaussian proposal over the cal
spline nodes, writing it as a breadcrumb (RIFT.calmarg.breadcrumbs) that seeds the
next iteration's wide ILE jobs via --calibration-proposal-breadcrumb.

Multiple dumps (e.g. several harvested high-L intrinsic points run as separate pilot
jobs) are combined by pooling their responsibilities -- the node draws are identical
across them (the dump uses a fixed seed), so log_resp simply adds in log space.

  util_CalPilotFit.py --dump pilot_0.npz [--dump pilot_1.npz ...] \
      --output-breadcrumb cal_proposal.npz --beta 1.0
"""
from __future__ import division

import sys
import argparse
import numpy as np
from scipy.special import logsumexp

from RIFT.calmarg import adaptive, breadcrumbs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dump", action="append", required=True,
                   help="responsibility dump .npz (repeatable; pooled)")
    p.add_argument("--output-breadcrumb", required=True,
                   help="output proposal breadcrumb .npz")
    p.add_argument("--beta", type=float, default=None,
                   help="tempering exponent for the responsibility weights (0<beta<=1); "
                        "lower = broader proposal.  Default: auto (choose beta so the "
                        "tempered neff hits --target-neff-frac, preventing collapse).")
    p.add_argument("--target-neff-frac", type=float, default=0.3,
                   help="auto-temper target: pick the LARGEST beta<=1 whose tempered neff "
                        ">= this fraction of n_cal.  Guards the high-dim collapse where a "
                        "single prior draw dominates (neff->1).  Default 0.3.")
    p.add_argument("--cov-inflate", type=float, default=1.0,
                   help="multiply the fitted covariance by this factor (safety margin)")
    p.add_argument("--iteration", type=int, default=None,
                   help="iteration number (recorded in breadcrumb meta)")
    opts = p.parse_args(argv)

    dumps = [np.load(f, allow_pickle=True) for f in opts.dump]
    nodes = dumps[0]["nodes"]
    # pool responsibilities across dumps (identical node draws -> add in log space)
    log_resp = dumps[0]["log_resp"].astype(float)
    for z in dumps[1:]:
        assert z["nodes"].shape == nodes.shape, "dump node grids differ; cannot pool"
        assert np.allclose(z["nodes"], nodes), "dump node draws differ; cannot pool (re-run pilots with the same --seed)"
        log_resp = np.logaddexp(log_resp, z["log_resp"].astype(float))

    n_cal = nodes.shape[0]
    neff = adaptive.neff_from_logweights(log_resp)
    print(" Pilot fit: %d realizations, %d dump(s); neff(responsibility)=%.1f/%d"
          % (n_cal, len(dumps), neff, n_cal))

    # Choose the tempering exponent.  In high dimensions a cold prior draw can collapse
    # to neff~1 (one realization dominates); fitting at beta=1 then gives a degenerate
    # (near-singular) proposal.  Auto-temper: take the LARGEST beta<=1 whose tempered
    # neff reaches target_neff_frac*n_cal, so the fit always uses a healthy sample.
    beta = opts.beta
    if beta is None:
        target = max(1.0, opts.target_neff_frac * n_cal)
        if adaptive.neff_from_logweights(log_resp) >= target:
            beta = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if adaptive.neff_from_logweights(mid * log_resp) >= target:
                    lo = mid          # mid is healthy -> can we go higher (sharper)?
                else:
                    hi = mid
            beta = lo
        print(" Auto-tempering: beta=%.3f (target neff>=%.0f; tempered neff=%.1f)"
              % (beta, target, adaptive.neff_from_logweights(beta * log_resp)))

    # shrink toward the prior diagonal: in ~60-D cal node space a pilot with neff~tens
    # cannot constrain the full covariance, so uninformed directions must default to
    # ~prior width (else the proposal collapses to a near-delta and seeded weights blow up)
    mean, cov = adaptive.fit_proposal(nodes, log_resp, beta, cov_inflate=opts.cov_inflate,
                                      prior_sigma=dumps[0]["prior_sigma"])

    cal = dict(proposal_mean=mean, proposal_cov=cov,
               prior_mean=dumps[0]["prior_mean"], prior_sigma=dumps[0]["prior_sigma"],
               node_log_f=dumps[0]["node_log_f"], n_nodes_amp=int(dumps[0]["n_nodes_amp"]),
               dets=[str(x) for x in dumps[0]["dets"]])
    meta = dict(kind="pilot_fit", n_dumps=len(opts.dump), n_cal=int(n_cal),
                beta=float(beta), neff_responsibility=float(neff))
    if opts.iteration is not None:
        meta["iteration"] = int(opts.iteration)
    breadcrumbs.save(opts.output_breadcrumb, cal=cal, meta=meta)
    print(" Wrote cal proposal breadcrumb -> %s" % opts.output_breadcrumb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
