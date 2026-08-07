#!/usr/bin/env python
"""Re-test the blocking regressions a merge-gate comparison reported, at NEW random seeds.

WHY THIS EXISTS.  Every gate verdict is a hard threshold (n_eff >= 100, JS < 3*floor + 0.004, ...)
applied to a stochastic quantity, so a cell sitting near a threshold flips on realization alone.
Observed: `GMM mix_d6_n3_s303` read n_eff 66 / 119 / 104 across runs of the SAME unchanged
checkout -- straddling the 100 floor -- purely from where it landed in the worker pool.  Reported
as a REGRESSION once, it would have blocked a merge that changed nothing about that sampler.

The fix is NOT to make the samplers deterministic.  Independent copies that localize differently
are our main detector for support/mode-collapse failures; pinning every fit to one seed would
silence it, and would make N copies of a production run no better than one.  The fix is to ask the
question again, properly: re-run the disputed cell in BOTH arms at several fresh run seeds and see
whether the candidate is really worse.

Usage:
  confirm_regressions.py base.json cand.json --base-checkout DIR --cand-checkout DIR \\
      [--repeats 3] [--jobs 4] [--seeds 11,22,33]

Exit 0 if no regression is CONFIRMED; 1 if any is.  A regression is confirmed when the candidate
is worse than the base in a MAJORITY of the fresh seeds (ties count as not-worse: the burden of
proof is on the claim that the candidate broke something).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# The comparator OWNS the definition of "blocking".  Importing it -- rather than reimplementing
# the PASS->non-PASS case, as this script first did -- is what keeps the two in step: the local
# copy was blind to REGRESSION(metrics), so a real metric regression (measured: n_eff 448->210)
# reported "no blocking regressions to confirm" and exited 0.
from compare_shape_results import classify, is_blocking, blocking_keys   # noqa: E402


def _key(r):
    return (r["kind"], r["target"])


def _blocking(base_path, cand_path, strict):
    with open(base_path) as fh:
        base = {_key(r): r for r in json.load(fh)}
    with open(cand_path) as fh:
        cand = {_key(r): r for r in json.load(fh)}
    return [(k, base.get(k), cand.get(k)) for k in blocking_keys(base, cand, strict)]


def _rerun(checkout, rec, seed, jobs, tag):
    """Re-run ONE cell of the matrix at a given run seed; return its record or None."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="confirm_%s_" % tag)
    os.close(fd)
    cmd = [os.environ.get("PYTHON", "python3"), os.path.join(HERE, "shape_recovery.py"),
           "--preset", "standard", "--json", path, "--jobs", str(jobs),
           "--samplers", rec["kind"] if not rec["kind"].startswith(("portfolio_warm",
                                                                   "portfolio_seq", "AV_seq"))
           else "AV",
           "--dims", str(rec["ndim"]), "--ncomps", str(rec["ncomp"]),
           "--target-seeds", str(rec["target_seed"]), "--run-seed", str(seed),
           "--warm-cases", "on" if rec["kind"] in ("portfolio_warm", "portfolio_seq",
                                                   "portfolio_seq_nobs", "AV_seq") else "off"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(checkout, "MonteCarloMarginalizeCode", "Code") + \
        os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        with open(path) as fh:
            for r in json.load(fh):
                if _key(r) == _key(rec):
                    return r
    except Exception:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("base")
    ap.add_argument("candidate")
    ap.add_argument("--base-checkout", required=True)
    ap.add_argument("--cand-checkout", required=True)
    ap.add_argument("--repeats", type=int, default=3,
                    help="fresh run seeds per arm (default 3; use more for a near-threshold cell)")
    ap.add_argument("--seeds", default=None, help="explicit comma list, overrides --repeats")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--min-valid", type=int, default=None,
                    help="usable base/candidate pairs required for a verdict (default: all "
                         "seeds). Fewer -> INCONCLUSIVE and exit 1, never a silent clear.")
    ap.add_argument("--strict-samplers",
                    default="AV,GMM,portfolio_warm,portfolio_seq,portfolio_seq_nobs")
    opts = ap.parse_args(argv)

    strict = set(x.strip() for x in opts.strict_samplers.split(",") if x.strip())
    seeds = ([int(x) for x in opts.seeds.split(",")] if opts.seeds
             else [987654 + 1000 * (i + 1) for i in range(opts.repeats)])
    if opts.min_valid is None:
        opts.min_valid = len(seeds)

    disputed = _blocking(opts.base, opts.candidate, strict)
    if not disputed:
        print("# no blocking regressions to confirm")
        return 0
    print("# confirming {} blocking regression(s) at {} fresh seed(s): {}".format(
        len(disputed), len(seeds), seeds))

    n_confirmed = 0
    n_inconclusive = 0
    for k, brec, crec in disputed:
        worse = same = 0
        detail = []
        for s in seeds:
            # The cell to re-run is defined by whichever record exists -- for a
            # REGRESSION(missing-in-candidate) row the candidate has no record, but the base
            # record still tells us which (kind, dim, ncomp, seed) to run, so the candidate CAN
            # and must be re-tested rather than written off.
            spec = brec if brec is not None else crec
            rb = _rerun(opts.base_checkout, spec, s, opts.jobs, "base")
            rc = _rerun(opts.cand_checkout, spec, s, opts.jobs, "cand")
            if rc is None and rb is None:
                detail.append("seed {}: BOTH reruns produced no record (no evidence either way)"
                              .format(s))
                continue
            if rc is None:
                # The CANDIDATE failed where the base did not.  That is not missing evidence, it
                # IS the regression: crashing or emitting no record is worse than passing.
                # Discarding it -- as this script first did -- let a candidate that failed on
                # every seed be declared "not confirmed".
                worse += 1
                detail.append("seed {}: CANDIDATE PRODUCED NO RECORD (counts against candidate)"
                              .format(s))
                continue
            if rb is None:
                detail.append("seed {}: base rerun produced no record; pair unusable".format(s))
                continue
            # the SAME classifier the gate uses, so a metrics-only regression is judged here
            # exactly as it was there
            verdict, note = classify(rb, rc)
            if is_blocking(verdict, k[0], strict):
                worse += 1
            else:
                same += 1
            detail.append("seed {}: {} (n_eff {:.0f} vs {:.0f}){}".format(
                s, verdict, rb.get("n_eff", float("nan")), rc.get("n_eff", float("nan")),
                "  [" + note + "]" if note else ""))

        valid = worse + same
        if valid < opts.min_valid:
            status = ("INCONCLUSIVE -- {}/{} valid pairs, need {}: NOT cleared"
                      .format(valid, len(seeds), opts.min_valid))
            n_inconclusive += 1
        elif worse > same:
            status = "CONFIRMED REGRESSION -- BLOCKS ({} worse / {} not-worse)".format(worse, same)
            n_confirmed += 1
        else:
            status = ("NOT CONFIRMED (realization noise; does not block) ({} worse / {} not-worse)"
                      .format(worse, same))
        print("\n{} {}".format(k[0], k[1]))
        for d in detail:
            print("   " + d)
        print("   -> " + status)

    print("\n# confirmed blocking regressions: {}".format(n_confirmed))
    if n_inconclusive:
        print("# INCONCLUSIVE rows (too few valid reruns): {}".format(n_inconclusive))
    # Inconclusive must NOT read as success: we failed to obtain the evidence that would clear
    # the row, so the gate stays red until a human looks.
    return 1 if (n_confirmed or n_inconclusive) else 0


if __name__ == "__main__":
    sys.exit(main())
