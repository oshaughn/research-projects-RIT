#!/usr/bin/env python3
"""Warm-start a cal pilot from a PRODUCTION pesummary calibration posterior.

Just as asimov bootstraps an intrinsic grid for a RIFT run, a finished PE analysis of an
event already constrains that event's *calibration* -- and a pilot can be seeded with it
instead of starting cold from the broad prior.  Production pesummary/bilby posteriors carry
the recovered cal posterior as columns `recalib_<IFO>_amplitude_<k>` (fractional dA/A) and
`recalib_<IFO>_phase_<k>` (radians) -- the SAME spline-node vector a RIFT cal breadcrumb is
a Gaussian over.  This fits those samples into proposal_mean/proposal_cov and writes them as
the pilot's iteration-0 seed `cal_consolidated_-1.npz` (reusing a placeholder breadcrumb's
prior + node structure), so wide_0 draws cal realizations where PE already pinned them.

COMPATIBILITY (the warm seed is only valid if the cal SETUP matches):
  * same detectors (the placeholder's `dets` must all be present as recalib_<IFO>_*),
  * same spline count (n_nodes_amp), and
  * ideally the same spline node FREQUENCIES (we warn if the pesummary nodes differ; a node
    at f1 is not interchangeable with one at f2).
So this seeds the SAME (or a cal-identical) event's pilot -- e.g. warm-start a RIFT-calmarg
rerun of an event from its bilby cal posterior.  It is NOT a cross-event transfer.

Reads via pesummary.io.read when available (handles all pesummary layouts), else h5py on the
`<label>/posterior_samples` structured dataset.

Usage:
    pesummary_cal_to_breadcrumb.py --pesummary results/<ev>/bilby-*/posterior_samples.h5 \\
        --placeholder rundir/cal_consolidated_-1.npz \\
        --output      rundir/cal_consolidated_-1.npz [--label <run>] [--cov-inflate 1.5]
"""
import argparse, numpy as np
import RIFT.calmarg.breadcrumbs as breadcrumbs


def read_recalib(path, label=None):
    """Return {recalib_<IFO>_(amplitude|phase)_<k>: samples} from a pesummary file."""
    try:
        from pesummary.io import read
        sd = read(path).samples_dict
        lab = label or list(sd.keys())[0]
        cols = {k: np.asarray(sd[lab][k]) for k in sd[lab].keys() if str(k).startswith("recalib_")}
        if cols:
            return lab, cols
    except Exception:
        pass
    import h5py                                   # fallback: structured posterior_samples ds
    f = h5py.File(path, "r")
    labels = [k for k in f if hasattr(f[k], "keys") and "posterior_samples" in f[k]]
    lab = label or labels[0]
    ps = f[lab]["posterior_samples"]
    names = ps.dtype.names or []
    return lab, {n: np.asarray(ps[n]) for n in names if n.startswith("recalib_")}


def build_matrix(cols, dets, na):
    """(Nsamples, 2*na*len(dets)) in breadcrumb order: per det [amp_0..,phase_0..], over dets."""
    out, missing = [], []
    for det in dets:
        for kind in ("amplitude", "phase"):
            for k in range(na):
                key = "recalib_%s_%s_%d" % (det, kind, k)
                (out.append(cols[key]) if key in cols else missing.append(key))
    if missing:
        raise SystemExit("ERROR: pesummary file lacks %d expected cal columns, e.g. %s\n"
                         "  (detector/spline mismatch -- this posterior's cal setup differs "
                         "from the breadcrumb's dets=%s, n_nodes_amp=%d)" % (len(missing), missing[0], dets, na))
    return np.column_stack(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pesummary", required=True, help="pesummary/bilby posterior_samples.h5 with recalib_* cols")
    p.add_argument("--placeholder", required=True, help="cold cal_consolidated_-1.npz (prior + node structure)")
    p.add_argument("--output", required=True, help="warm breadcrumb to write")
    p.add_argument("--label", default=None, help="pesummary run label (default: first)")
    p.add_argument("--cov-inflate", type=float, default=1.0)
    opts = p.parse_args(argv)

    bc = breadcrumbs.load(opts.placeholder)
    cal = bc["cal"]
    dets, na = list(cal["dets"]), int(cal["n_nodes_amp"])
    dim = 2 * na * len(dets)

    lab, cols = read_recalib(opts.pesummary, opts.label)
    X = build_matrix(cols, dets, na)
    if X.shape[1] != dim:
        raise SystemExit("ERROR: assembled %d cols, breadcrumb dim %d" % (X.shape[1], dim))

    cal["proposal_mean"] = X.mean(axis=0)
    cal["proposal_cov"] = np.cov(X, rowvar=False) * opts.cov_inflate + np.eye(dim) * 1e-12
    breadcrumbs.save(opts.output, cal=cal,
                     meta=dict(warm_start=True, source=opts.pesummary, pesummary_label=lab,
                               n_samples=int(X.shape[0]), cov_inflate=opts.cov_inflate, iteration=-1))
    sd = np.sqrt(np.diag(np.cov(X, rowvar=False)))
    print("WARM breadcrumb -> %s" % opts.output)
    print("  from pesummary label '%s', %d samples, dim=%d (%s x %d amp+phase nodes)"
          % (lab, X.shape[0], dim, dets, na))
    print("  proposal sigma vs prior sigma (median ratio): %.3f"
          % np.median(sd / np.asarray(cal["prior_sigma"])))


if __name__ == "__main__":
    main()
