#!/usr/bin/env python3
#
# util_CompositionReweightNet.py
#
# Composition-equalising THINNING of a CIP training set (composite/all.net: 13 whitespace
# cols; 1=m1, 2=m2, 3-5=s1xyz, 6-8=s2xyz, 9=lnL).  In accumulated low-mass training sets the
# fraction of near-peak rows degrades with chi1_perp, so an RF fit -- a local average --
# regresses the transverse tail toward its many low-lnL neighbours and the recovered
# chi1_perp/a1 posterior is too narrow.  This tool thins the FAR-from-peak rows per
# chi1_perp bin so every bin's near-peak fraction matches the best bin's:
#   - per bin, KEEP every near-peak row, thin the far rows at random (density reweighting;
#     NEVER a global lnL truncation, NEVER duplication, NEVER invented points -- the lnL
#     span and peak are preserved exactly);
#   - SELF-QUANTILE definitions only (no external reference, no absolute thresholds):
#       near-peak:          lnL > lnL.max() - NAT   (NAT = 5 nats below the set's own peak)
#       tail/core boundary: chi1_perp > its own q80
#   - FAIL-SAFE: on any internal error or degenerate input (too few rows, flat lnL, dead
#     tail, ...) it copies input -> output UNCHANGED, warns loudly, exits 0 -- it must never
#     break the pipeline.  --strict makes such failures fatal (bench use).
#
# Rows are written back as the ORIGINAL INPUT LINES (values untouched); only line SELECTION
# happens here, deterministically under --seed.  Opt in from util_RIFT_pseudo_pipe.py via
# --internal-cip-composition-reweight (runs through util_CIPCompositionReweightWrapper.sh).

import argparse
import json
import shutil
import sys

import numpy as np

NAT = 5.0          # near-peak window, nats below the set's own peak
SELFQ = 0.80       # self-quantile tail/core boundary
LNL_COL = 9
MIN_ROWS = 1000    # below this, thinning statistics are meaningless -> fallback
MIN_BIN = 10       # bins with fewer rows are skipped
MIN_NEAR = 100     # too few near-peak rows overall -> fallback
MIN_BINS_USED = 2  # need at least two usable bins to equalise anything
# validated bin edges (default --nbins 10 keeps these)
VERIFY_EDGES = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99, 10.])


def comp_stats(cp, lnl):
    """Self-quantile composition statistic (fully self-contained definitions)."""
    peak = lnl.max()
    near = lnl > peak - NAT
    q80 = float(np.quantile(cp, SELFQ))
    tail_s = cp > q80
    nf_tail_s = float(near[tail_s].mean()) if tail_s.sum() >= 50 else float("nan")
    nf_core_s = float(near[~tail_s].mean()) if (~tail_s).sum() else float("nan")
    comp_self = nf_tail_s / nf_core_s if nf_core_s and nf_core_s > 0 else float("nan")
    m05 = cp > 0.5
    nf_gt05 = float(near[m05].mean()) if m05.sum() else float("nan")
    return dict(n=int(len(lnl)), peak=float(peak), n_near=int(near.sum()),
                near_frac=float(near.mean()), q80=q80,
                nf_tail=nf_tail_s, nf_core=nf_core_s, comp_self=comp_self,
                nearfrac_cp_gt_0p5=nf_gt05,
                lnl_span=float(lnl.max() - lnl.min()))


def fallback(args, reason, stats):
    sys.stderr.write("*" * 78 + "\n")
    sys.stderr.write(f"util_CompositionReweightNet WARNING: {reason}\n")
    if args.strict:
        sys.stderr.write("  --strict: FATAL, no output written.\n")
        sys.stderr.write("*" * 78 + "\n")
        _write_stats(args, dict(stats, fallback=reason, strict_fatal=True))
        sys.exit(2)
    sys.stderr.write(f"  FALLING BACK: copying input -> output unchanged ({args.output})\n")
    sys.stderr.write("*" * 78 + "\n")
    shutil.copyfile(args.input, args.output)
    _write_stats(args, dict(stats, fallback=reason, strict_fatal=False))
    sys.exit(0)


def _write_stats(args, d):
    if not args.stats_json:
        return
    try:
        with open(args.stats_json, "w") as f:
            json.dump(d, f, indent=1, default=float)
    except OSError as e:
        sys.stderr.write(f"util_CompositionReweightNet WARNING: cannot write stats json: {e}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="composite/all.net (13 whitespace cols, col 9 = lnL)")
    ap.add_argument("--output", required=True, help="thinned output file")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for the deterministic far-row thinning")
    ap.add_argument("--exploration-parameter", default="chi1_perp",
                    help="binning coordinate; only chi1_perp implemented")
    ap.add_argument("--nbins", type=int, default=10,
                    help="number of chi1_perp bins; default 10 uses the validated "
                         "edges [0,.1,...,.8,.99,10]; any other value uses uniform bins over the data range")
    ap.add_argument("--strict", action="store_true",
                    help="make degenerate input / internal errors fatal (bench use); default is fail-safe copy-through")
    ap.add_argument("--stats-json", default=None, help="write before/after composition stats as json")
    args = ap.parse_args()

    base_stats = dict(tool="util_CompositionReweightNet", input=args.input, output=args.output,
                      seed=args.seed, nat=NAT, self_quantile=SELFQ,
                      exploration_parameter=args.exploration_parameter, nbins=args.nbins,
                      definitions=dict(
                          near_peak="lnL > lnL.max() - 5.0  (self peak, NAT=5 nats)",
                          tail_core_boundary="chi1_perp > quantile(chi1_perp, 0.80) of the input itself",
                          comp="nearfrac(tail)/nearfrac(core)",
                          construction="per chi1_perp bin keep ALL near-peak rows; thin far rows so "
                                       "per-bin near-peak fraction equals the best bin's (thinning only)"))

    try:
        if args.exploration_parameter != "chi1_perp":
            fallback(args, f"exploration parameter '{args.exploration_parameter}' not implemented", base_stats)

        # keep original lines so selected rows are written back verbatim
        lines, rows = [], []
        with open(args.input) as f:
            for ln in f:
                if not ln.strip() or ln.lstrip().startswith("#"):
                    continue
                lines.append(ln)
                rows.append(ln.split())
        if len(rows) < MIN_ROWS:
            fallback(args, f"too few rows ({len(rows)} < {MIN_ROWS}) for composition thinning", base_stats)
        ncol = len(rows[0])
        if ncol <= LNL_COL or any(len(r) != ncol for r in rows):
            fallback(args, "inconsistent or too-few columns; need lnL at 0-based col 9", base_stats)
        a = np.asarray(rows, dtype=float)
        lnl = a[:, LNL_COL]
        cp = np.hypot(a[:, 3], a[:, 4])  # chi1_perp = |s1_xy|
        if not (np.isfinite(lnl).all() and np.isfinite(cp).all()):
            fallback(args, "non-finite lnL or spin components in input", base_stats)

        before = comp_stats(cp, lnl)
        base_stats["rows_in"] = before["n"]
        base_stats["before"] = before
        peak = lnl.max()
        near = lnl > peak - NAT

        if before["lnl_span"] <= NAT:
            fallback(args, f"lnL span {before['lnl_span']:.3f} <= NAT={NAT}: every row is 'near-peak', "
                           "nothing to classify (flat/degenerate likelihoods)", base_stats)
        if near.sum() < MIN_NEAR:
            fallback(args, f"only {near.sum()} near-peak rows (< {MIN_NEAR})", base_stats)

        # --- the composition-equalising construction ---
        if args.nbins == 10:
            edges = VERIFY_EDGES
        else:
            if args.nbins < 2:
                fallback(args, f"--nbins {args.nbins} < 2", base_stats)
            hi = max(cp.max() * (1 + 1e-9), 1e-6)
            edges = np.linspace(0.0, hi, args.nbins + 1)
            edges[-1] = max(edges[-1], 10.0)  # last bin catches everything
        binfo = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (cp >= lo) & (cp < hi)
            if m.sum() < MIN_BIN:
                continue
            binfo.append((float(lo), float(hi), m, float(near[m].mean())))
        if len(binfo) < MIN_BINS_USED:
            fallback(args, f"only {len(binfo)} usable chi1_perp bins (need >= {MIN_BINS_USED})", base_stats)

        # Equalise UP to the BEST bin: keep every near-peak point everywhere, thin the FAR
        # points in the poorer bins until their near-peak fraction matches.  Thinning only.
        target = max(b[3] for b in binfo)
        if not (0 < target <= 1):
            fallback(args, f"degenerate target near-peak fraction {target}", base_stats)
        rng = np.random.default_rng(args.seed)
        keep = np.zeros(len(a), bool)
        bins_out = []
        for lo, hi, m, f in binfo:
            idx = np.flatnonzero(m)
            nr = idx[near[idx]]
            fr = idx[~near[idx]]
            n_far_needed = int(len(nr) * (1 - target) / target)
            keep[nr] = True
            if len(fr):
                keep[rng.choice(fr, min(len(fr), n_far_needed), replace=False)] = True
            bins_out.append(dict(lo=lo, hi=hi, n_before=int(m.sum()), near=int(len(nr)),
                                 nearfrac_before=f))

        kept_idx = np.flatnonzero(keep)
        after = comp_stats(cp[keep], lnl[keep])
        for b in bins_out:
            m = (cp[keep] >= b["lo"]) & (cp[keep] < b["hi"])
            nn = near[keep][m]
            b["n_after"] = int(m.sum())
            b["nearfrac_after"] = float(nn.mean()) if m.sum() else float("nan")

        # sanity: this is a thinning -- never grow, never lose the peak
        assert len(kept_idx) <= len(a) and near[keep].sum() == near.sum()

        with open(args.output, "w") as f:
            for i in kept_idx:      # original file order, original bytes
                f.write(lines[i])

        base_stats.update(rows_out=int(len(kept_idx)), target_nearfrac=float(target),
                          edges=[float(e) for e in edges], after=after,
                          bins=bins_out, fallback=None)
        _write_stats(args, base_stats)

        print(f"util_CompositionReweightNet: {len(a)} -> {len(kept_idx)} rows "
              f"(target per-bin nearfrac {target:.4f})")
        print(f"  lnL span before/after: {before['lnl_span']:.4f} / {after['lnl_span']:.4f}   "
              f"(peak preserved: {before['peak']:.4f})")
        print(f"  comp (self-q80 tail/core) before/after: {before['comp_self']:.4f} / {after['comp_self']:.4f}")
        print(f"  nearfrac|cp>0.5 before/after: {before['nearfrac_cp_gt_0p5']:.4f} / {after['nearfrac_cp_gt_0p5']:.4f}")
        print(f"  {'bin':<14}{'n_in':>8}{'nf_in':>9}{'n_out':>8}{'nf_out':>9}")
        for b in bins_out:
            print(f"  [{b['lo']:.2f},{b['hi']:.2f}){'':<3}{b['n_before']:>8d}{b['nearfrac_before']:>9.4f}"
                  f"{b['n_after']:>8d}{b['nearfrac_after']:>9.4f}")
    except SystemExit:
        raise
    except Exception as e:
        fallback(args, f"internal error: {type(e).__name__}: {e}", base_stats)


if __name__ == "__main__":
    main()
