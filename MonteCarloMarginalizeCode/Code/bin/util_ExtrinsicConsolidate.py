#! /usr/bin/env python
"""
util_ExtrinsicConsolidate.py

Consolidate the per-event EXTRINSIC proposal breadcrumbs written by one iteration's wide
ILE jobs (each ILE job that ran with --extrinsic-proposal-output drops a small .npz holding
its run's extrinsic GMM proposal, see RIFT.calmarg.extrinsic_handoff) into ONE breadcrumb
that seeds the NEXT iteration's ILE jobs via --extrinsic-proposal-breadcrumb.

The extrinsic posterior is nearly the same across intrinsic-grid points (it is set by the
data + best-fit template), so we do not need to merge mixtures across points -- we just pick
the single MOST REPRESENTATIVE per-event proposal and hand it forward.  "Most representative"
defaults to the highest-lnL (near the peak) job, with effective-sample-count / sample-count
as tie-breaks; --select lets you change the key.

Robust by design: unreadable / empty (placeholder) / extrinsic-less inputs are skipped, and
the output breadcrumb is ALWAYS written (empty if nothing valid was found) so that the
next iteration's OSG file-transfer for extr_consolidated_<it>.npz never fails -- a downstream
empty/missing breadcrumb simply makes ILE fall back to its cold default proposal.
"""
from __future__ import print_function

import sys
import glob
import argparse

import numpy as np

import RIFT.calmarg.breadcrumbs as breadcrumbs


def _load_one(path):
    """Return (meta, extrinsic) from a breadcrumb, or None if it is unusable
    (missing/empty placeholder/corrupt/no extrinsic payload)."""
    try:
        g = breadcrumbs.load(path)
    except Exception as e:
        print("  skip {} ({})".format(path, e))
        return None
    ext = g.get("extrinsic")
    if ext is None or not ext.get("groups"):
        print("  skip {} (no extrinsic payload)".format(path))
        return None
    return g.get("meta", {}) or {}, ext


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-glob", default=None,
                   help="Glob for the per-event proposal breadcrumbs (e.g. 'extr_proposal_3_*.npz'). "
                        "Matched in the current working directory; on OSG the inputs are transferred "
                        "flat into the job scratch dir, so the same glob works there.")
    p.add_argument("--input", action="append", default=[],
                   help="Explicit per-event proposal breadcrumb path (repeatable). Combined with --input-glob.")
    p.add_argument("--output", required=True,
                   help="Output consolidated breadcrumb path (e.g. extr_consolidated_<it>.npz).")
    p.add_argument("--select", default="lnL", choices=["lnL", "neff", "n_samples"],
                   help="Per-event metric to rank by when picking the representative proposal. Default lnL.")
    p.add_argument("--iteration", default=None,
                   help="Iteration index (recorded in the output meta; informational).")
    opts = p.parse_args(argv)

    paths = list(opts.input)
    if opts.input_glob:
        paths += sorted(glob.glob(opts.input_glob))
    # de-duplicate, preserve order
    seen = set(); paths = [x for x in paths if not (x in seen or seen.add(x))]
    print("util_ExtrinsicConsolidate: {} candidate breadcrumb(s); ranking by '{}'".format(len(paths), opts.select))

    candidates = []
    for path in paths:
        loaded = _load_one(path)
        if loaded is None:
            continue
        meta, ext = loaded
        score = float(meta.get(opts.select, -np.inf))
        candidates.append((score, path, meta, ext))

    if not candidates:
        # Always emit an (empty) breadcrumb so the next iteration's transfer/seed never fails.
        breadcrumbs.save(opts.output, extrinsic=None,
                         meta=dict(iteration=opts.iteration, n_candidates=0, source=None))
        print("util_ExtrinsicConsolidate: no usable extrinsic proposals; wrote EMPTY {} "
              "(next iteration falls back to the cold default).".format(opts.output))
        return 0

    # tie-break: primary --select metric, then neff, then n_samples
    def _key(c):
        _, _, m, _e = c
        return (c[0], float(m.get("neff", -np.inf)), float(m.get("n_samples", -np.inf)))
    best = max(candidates, key=_key)
    best_score, best_path, best_meta, best_ext = best

    breadcrumbs.save(opts.output, extrinsic=best_ext,
                     meta=dict(iteration=opts.iteration, n_candidates=len(candidates),
                               source=best_path, select=opts.select, select_value=best_score,
                               source_event=best_meta.get("event"),
                               source_lnL=best_meta.get("lnL"),
                               source_neff=best_meta.get("neff"),
                               source_n_samples=best_meta.get("n_samples"),
                               groups=[g["params"] for g in best_ext["groups"]]))
    print("util_ExtrinsicConsolidate: picked {} ({}={}, event={}, lnL={}) -> {} ({} groups)".format(
        best_path, opts.select, best_score, best_meta.get("event"), best_meta.get("lnL"),
        opts.output, len(best_ext["groups"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
