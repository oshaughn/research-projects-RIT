#!/usr/bin/env python3
"""
shape_extrinsic.py -- weighted-sample SHAPE check on an ILE extrinsic export.

n_eff alone can hide a wrong posterior SHAPE: a run can report a healthy n_eff while a handful of
samples own the estimate (over-broad GMM proposal at too-high a component cap) or the recovered
marginals are degenerate.  This reads the extrinsic sample cloud saved by `--save-samples` and
reports, per extrinsic parameter, the WEIGHTED marginal (mean, std) plus global diagnostics:
  * n_eff (Kish) of the weighted cloud,
  * max single-sample weight FRACTION -- the outlier-dominance signal (the cap-too-high failure),
  * effective vs raw sample count.

Usage: shape_extrinsic.py <ILE .xml or _0_.dat output> [<more> ...]
Auto-detects the format.  Compare several caps side by side to see the failure mode emerge.
"""
from __future__ import print_function
import sys
import numpy as np

# extrinsic columns we summarize (name -> unit label)
PARAMS = [("distance", "Mpc"), ("inclination", "rad"), ("right_ascension", "rad"),
          ("declination", "rad"), ("psi", "rad"), ("phi_orb", "rad")]
# aliases as written by --save-samples (longitude/latitude/polarization/coa_phase)
ALIAS = {"right_ascension": ["longitude", "ra"], "declination": ["latitude", "dec"],
         "psi": ["polarization"], "phi_orb": ["coa_phase"]}


def _from_xml(path):
    """Read the sim_inspiral-style table --save-samples writes; return dict of arrays incl weights."""
    from igwn_ligolw import ligolw, lsctables, utils as ligolw_utils
    xmldoc = ligolw_utils.load_filename(path, contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
    tbl = lsctables.SimInspiralTable.get_table(xmldoc)
    cols = {}
    # standard extrinsic mapping (see the save-samples block in the driver)
    getters = {"distance": "distance", "inclination": "inclination",
               "right_ascension": "longitude", "declination": "latitude",
               "psi": "polarization", "phi_orb": "coa_phase", "loglikelihood": "alpha1"}
    for k in ("distance", "inclination", "longitude", "latitude", "polarization", "coa_phase"):
        try:
            cols[k] = np.array([getattr(r, k) for r in tbl], dtype=float)
        except Exception:
            pass
    # weight: RIFT stores the sampling info in alpha columns; loglikelihood via alpha1 typically.
    for wk in ("alpha1", "alpha", "snr"):
        try:
            cols["loglikelihood"] = np.array([getattr(r, wk) for r in tbl], dtype=float)
            break
        except Exception:
            continue
    return cols


def _from_dat(path):
    """The .dat ASCII form: RIFT writes extrinsic columns + weights when --save-samples is on.
    Column order is the ILE convention; we detect it by width and pull the log-weight column."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    # A single-row .dat is the marginalized point (no cloud) -- not a shape export.
    if arr.shape[0] < 5:
        return None
    return arr


def summarize(path):
    name = path.split("/")[-1]
    try:
        if path.endswith(".xml") or path.endswith(".xml.gz"):
            cols = _from_xml(path)
            ll = cols.get("loglikelihood")
            if ll is None:
                print("%-28s  (no weight column found in XML)" % name); return
            w = np.exp(ll - np.max(ll))
            data = {"distance": cols.get("distance"), "inclination": cols.get("inclination"),
                    "right_ascension": cols.get("longitude"), "declination": cols.get("latitude"),
                    "psi": cols.get("polarization"), "phi_orb": cols.get("coa_phase")}
        else:
            arr = _from_dat(path)
            if arr is None:
                print("%-28s  (single-row .dat: marginalized point, not a cloud)" % name); return
            # heuristic: last col = neff-ish, cols mapped by the standard ILE .dat layout is
            # fragile, so require the XML path for full shape; here just report n_eff proxy.
            print("%-28s  (.dat cloud reader needs the XML; run made %d rows)" % (name, arr.shape[0])); return
    except Exception as e:
        print("%-28s  ERROR %s" % (name, str(e)[:70])); return

    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    W = w.sum()
    if W <= 0:
        print("%-28s  (all-zero weights)" % name); return
    neff = W * W / np.sum(w * w)
    maxfrac = float(np.max(w) / W)
    print("== %s ==" % name)
    print("   n_eff(Kish)=%.1f  raw=%d  max-weight-frac=%.3e%s" % (
        neff, len(w), maxfrac, "   <-- ONE SAMPLE DOMINATES" if maxfrac > 0.05 else ""))
    for p, unit in PARAMS:
        x = data.get(p)
        if x is None or len(x) != len(w):
            continue
        m = np.sum(w * x) / W
        s = np.sqrt(max(0.0, np.sum(w * (x - m) ** 2) / W))
        print("   %-16s mean=%9.3f  std=%9.3f  %s" % (p, m, s, unit))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
        print()
