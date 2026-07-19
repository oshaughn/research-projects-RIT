#!/usr/bin/env python
"""
util_BuildProposalField.py

L3 producer (MULTI-pilot generalization): aggregate the extrinsic outputs of a FEW
cherry-picked points into a ProposalField keyed by intrinsic parameters, consumed by
integrate_likelihood_extrinsic_batchmode --extrinsic-proposal-field.

IMPORTANT -- do NOT point this at a whole grid analyzed with --save-samples: per-point
sample dumps blow up disk fast, and a full field is rarely needed.  The standard,
disk-safe path is a SINGLE cherry-picked pilot (util_PickPilotPoint.py picks the best
point by lnL after a cheap iteration 0; run ONLY that point with --save-samples;
warm-start the rest with --sampler-warmstart-samples).  Use this multi-pilot field only
when a few (2-4) well-separated pilots are genuinely warranted (e.g. a known multimodal
source) -- run save-samples on THOSE few points only.

For each provided point it reads that point's saved extrinsic samples (--save-samples
.xml.gz), keeps the high-likelihood subset, converts to the sampler's coordinate
convention (matching --declination-cosine-sampler / --inclination-cosine-sampler), and
records it against the point's intrinsic lambda = [m1, m2, s1x..s2z].

A proposal only ever shapes p_s, so a stale/partial field can only cost efficiency, never
bias -- missing points are simply skipped.

Usage:
  util_BuildProposalField.py --grid overlap-grid-5.xml.gz --output-prefix ile_5 \
      --out proposal_field_5.npz [--deltalnL 15] [--max-per-point 4000] \
      [--no-cosine-dec] [--no-cosine-incl]
"""
from __future__ import print_function
import argparse
import glob
import os
import numpy as np

# sampler extrinsic coordinate order used by the AV extrinsic integrator
EXTRINSIC_PARAMS = ["right_ascension", "declination", "phi_orb", "inclination", "psi", "distance"]


def _read_extrinsic_xml(path):
    """Return (samples Nx6 in sampler coords, lnL N) from an ILE --save-samples xml.gz,
    or (None, None).  Converts physical (dec, incl) to the cosine-sampler variables."""
    try:
        from igwn_ligolw import ligolw, lsctables, utils
        xd = utils.load_filename(path, contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
        t = lsctables.SimInspiralTable.get_table(xd)
    except Exception as e:
        print("   (could not read {}: {})".format(path, e))
        return None, None
    if len(t) == 0:
        return None, None
    ra = np.array([r.longitude for r in t]); lat = np.array([r.latitude for r in t])
    dist = np.array([r.distance for r in t]); incl = np.array([r.inclination for r in t])
    psi = np.array([r.polarization for r in t]); phi = np.array([r.coa_phase for r in t])
    lnL = np.array([getattr(r, 'alpha1', 0.0) for r in t])
    return ra, lat, dist, incl, psi, phi, lnL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="sim xml with the iteration's intrinsic grid")
    ap.add_argument("--output-prefix", required=True, help="ILE --output-file prefix; reads <prefix>_<i>_.xml.gz")
    ap.add_argument("--out", required=True, help="output ProposalField .npz")
    ap.add_argument("--deltalnL", type=float, default=15.0, help="keep samples within this lnL of each point's max")
    ap.add_argument("--max-per-point", type=int, default=4000)
    ap.add_argument("--no-cosine-dec", action="store_true", help="grid did NOT use --declination-cosine-sampler")
    ap.add_argument("--no-cosine-incl", action="store_true", help="grid did NOT use --inclination-cosine-sampler")
    args = ap.parse_args()

    from RIFT.integrators.proposal_field import ProposalField, lambda_from_P, LAMBDA_INTRINSIC_PARAMS
    import RIFT.lalsimutils as lalsimutils

    P_list = lalsimutils.xml_to_ChooseWaveformParams_array(args.grid)
    pf = ProposalField(intrinsic_params=LAMBDA_INTRINSIC_PARAMS, extrinsic_params=EXTRINSIC_PARAMS)

    n_added = 0
    for i, P in enumerate(P_list):
        path = "{}_{}_.xml.gz".format(args.output_prefix, i)
        if not os.path.exists(path):
            continue
        got = _read_extrinsic_xml(path)
        if got[0] is None:
            continue
        ra, lat, dist, incl, psi, phi, lnL = got
        dec_s = np.sin(lat) if not args.no_cosine_dec else lat
        incl_s = np.cos(incl) if not args.no_cosine_incl else incl
        cols = np.vstack([ra, dec_s, phi, incl_s, psi, dist]).T
        keep = lnL > (np.nanmax(lnL) - args.deltalnL)
        if np.sum(keep) < 2:
            keep = np.ones(len(cols), dtype=bool)   # fall back to all saved samples
        sub = cols[keep]
        if len(sub) > args.max_per_point:
            sub = sub[np.random.RandomState(0).choice(len(sub), args.max_per_point, replace=False)]
        pf.add(lambda_from_P(P), sub)
        n_added += 1

    if len(pf) == 0:
        print("WARNING: no usable per-point outputs found; writing nothing.")
        return
    pf.save(args.out)
    print("Built ProposalField with {} entries (of {} grid points) -> {} ({} bytes)".format(
        n_added, len(P_list), args.out, os.path.getsize(args.out)))


if __name__ == "__main__":
    main()
