#!/usr/bin/env python3
"""
compare_extrinsic_breadcrumbs.py -- cross-copy EXTRINSIC-POSTERIOR stability check.

Reads the per-run breadcrumbs written by `--extrinsic-proposal-output` (a weight-correct GMM fit of
each run's TRUE-weighted extrinsic posterior; see RIFT/calmarg/extrinsic_handoff.py).  For a seed
ensemble of one config, it answers the question that n_eff alone cannot:

  Across independent copies, is the recovered extrinsic posterior STABLE, and does it preserve the
  real degeneracy structure -- sky ring (ra,dec), dL-inclination arc, psi-phi -- or does some copy
  silently COLLAPSE a group (fewer modes / a shifted blob)?  A collapse is a failure mode even when
  n_eff looks acceptable.

Per group we report, per copy: the number of effective mixture modes (weight > MODE_WT) and the
mixture mean + spread in the model's NORMALIZED frame (all copies share the same bounds, so the
normalized frame is directly comparable -- we un-normalize the summary to physical units too).  Then
per group across the ensemble: the cross-copy scatter of the mode count and of the group mean.  A
stable config has consistent mode counts and small cross-copy mean scatter.

Usage: compare_extrinsic_breadcrumbs.py <ext_*.npz> [<ext_*.npz> ...]
Groups files by config prefix (ext_<cfg>_s<seed>.npz).
"""
from __future__ import print_function
import sys, os, re
import numpy as np

# repo import: RIFT/calmarg/breadcrumbs.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "RIFT", ".."))
try:
    from RIFT.calmarg import breadcrumbs
except Exception as e:
    print("cannot import RIFT.calmarg.breadcrumbs (%s); set PYTHONPATH to the repo Code dir" % e)
    sys.exit(2)

MODE_WT = 0.10   # a mixture component counts as a 'mode' if its weight exceeds this
NAME_RE = re.compile(r"ext_(?P<cfg>.+?)_s(?P<seed>\d+)\.npz$")


def _phys_mean(g):
    """Mixture mean per param, un-normalized to physical units via stored bounds.
    RIFT GMM works in a [0,1]-normalized frame per dimension: x_phys = lo + x_norm*(hi-lo)."""
    means = np.asarray(g["means"], dtype=float)          # (K,d) normalized
    w = np.asarray(g["weights"], dtype=float); w = w / w.sum()
    b = np.asarray(g["bounds"], dtype=float)             # (d,2)
    lo, hi = b[:, 0], b[:, 1]
    mu_norm = (w[:, None] * means).sum(axis=0)           # (d,)
    return lo + mu_norm * (hi - lo)                       # (d,) physical


def load_group_summaries(paths):
    """-> dict cfg -> list of per-copy dicts {seed, neff, nsamp, groups:{gname:{K,modes,mean_phys}}}"""
    out = {}
    for p in paths:
        m = NAME_RE.search(os.path.basename(p))
        cfg = m.group("cfg") if m else "?"
        seed = int(m.group("seed")) if m else -1
        try:
            bc = breadcrumbs.load(p)
        except Exception as e:
            print("  skip %s (%s)" % (p, str(e)[:60])); continue
        meta = bc.get("meta", {})
        ext = bc.get("extrinsic")
        rec = dict(seed=seed, neff=float(meta.get("neff", np.nan)),
                   nsamp=int(meta.get("n_samples", 0)), groups={})
        if ext:
            for g in ext["groups"]:
                gname = ",".join(g["params"])
                w = np.asarray(g["weights"], dtype=float)
                rec["groups"][gname] = dict(K=len(w), modes=int((w > MODE_WT).sum()),
                                            mean_phys=_phys_mean(g),
                                            bounds=np.asarray(g["bounds"], dtype=float))
        out.setdefault(cfg, []).append(rec)
    for cfg in out:
        out[cfg].sort(key=lambda r: r["seed"])
    return out


GOOD_NEFF = 5.0   # a copy counts as 'landed' (usable posterior) above this n_eff


def _in_bounds(rec):
    """True if the group's phys-mean lies inside its bounds -- a collapsed run's degenerate GMM
    fit drifts a component out of range, so out-of-bounds is a collapse signature."""
    b = rec.get("bounds")
    m = rec.get("mean_phys")
    if b is None or m is None:
        return True
    lo, hi = np.asarray(b)[:, 0], np.asarray(b)[:, 1]
    return bool(np.all(m >= lo - 1e-6) and np.all(m <= hi + 1e-6))


def report(summaries):
    for cfg in sorted(summaries):
        copies = summaries[cfg]
        neffs = np.array([c["neff"] for c in copies], dtype=float)
        ngood = int(np.sum(neffs >= GOOD_NEFF))
        # Kish effective #copies over the reliability weights: how many copies the pool really rests on
        w = np.where(np.isfinite(neffs), neffs, 0.0)
        kish_copies = (w.sum() ** 2 / np.sum(w * w)) if np.sum(w * w) > 0 else 0.0
        print("=" * 78)
        print("CONFIG %s  (%d copies, %d landed n_eff>=%.0f)   n_eff: %s" % (
            cfg, len(copies), ngood, GOOD_NEFF, " ".join("%.1f" % x for x in neffs)))
        print("  reliability-weighted effective #copies (Kish over n_eff) = %.1f" % kish_copies)

        gnames = []
        for c in copies:
            for gn in c["groups"]:
                if gn not in gnames:
                    gnames.append(gn)

        # per-copy: n_eff + per-group mode count + in-bounds (collapse detector)
        print("  per-copy structure (seed: n_eff | group->modes,inbounds):")
        for c in copies:
            parts = []
            for gn in gnames:
                r = c["groups"].get(gn)
                if r is None:
                    parts.append("%s:-" % gn.split(",")[0]); continue
                ib = "ok" if _in_bounds(r) else "OOB"
                parts.append("%s:m%d/%s" % (gn.split(",")[0], r["modes"], ib))
            tag = "" if c["neff"] >= GOOD_NEFF else "  (collapsed)"
            print("    s%-3d n_eff=%6.1f | %s%s" % (c["seed"], c["neff"], "  ".join(parts), tag))

        # POOLING: reliability-weighted vs naive vs good-only, per group mean
        print("  POOLED group mean  [reliability-weighted (all) | naive-unweighted | good-only]:")
        for gn in gnames:
            recs = [(c["neff"], c["groups"][gn]) for c in copies if gn in c["groups"]]
            if not recs:
                continue
            means = np.array([r["mean_phys"] for _, r in recs])          # (n,d)
            ne = np.array([max(0.0, n) for n, _ in recs])
            wpool = ne / ne.sum() if ne.sum() > 0 else np.ones(len(ne)) / len(ne)
            m_relw = (wpool[:, None] * means).sum(axis=0)                 # reliability-weighted
            m_naive = means.mean(axis=0)                                  # naive (corrupted by collapses)
            good = ne >= GOOD_NEFF
            m_good = means[good].mean(axis=0) if good.any() else np.full(means.shape[1], np.nan)
            # do the good copies AGREE? scatter of good-only means
            good_scatter = np.nanstd(means[good], axis=0) if good.sum() >= 2 else np.full(means.shape[1], np.nan)
            print("    [%s]" % gn)
            print("        relw =%s   naive=%s   good =%s   good-scatter=%s" % (
                np.array2string(m_relw, precision=2, suppress_small=True),
                np.array2string(m_naive, precision=2, suppress_small=True),
                np.array2string(m_good, precision=2, suppress_small=True),
                np.array2string(good_scatter, precision=2, suppress_small=True)))
        print()


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print(__doc__); sys.exit(1)
    report(load_group_summaries(paths))
