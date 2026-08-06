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
from shape_recovery import evaluate            # noqa: E402


def _key(r):
    return (r["kind"], r["target"])


def _blocking(base_path, cand_path, strict):
    with open(base_path) as fh:
        base = {_key(r): r for r in json.load(fh)}
    with open(cand_path) as fh:
        cand = {_key(r): r for r in json.load(fh)}
    out = []
    for k in sorted(set(base) & set(cand)):
        if k[0] not in strict:
            continue
        st_b, _ = evaluate(base[k])
        st_c, _ = evaluate(cand[k])
        if st_b == "PASS" and st_c != "PASS":
            out.append((k, base[k], cand[k]))
    return out


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
    ap.add_argument("--strict-samplers",
                    default="AV,GMM,portfolio_warm,portfolio_seq,portfolio_seq_nobs")
    opts = ap.parse_args(argv)

    strict = set(x.strip() for x in opts.strict_samplers.split(",") if x.strip())
    seeds = ([int(x) for x in opts.seeds.split(",")] if opts.seeds
             else [987654 + 1000 * (i + 1) for i in range(opts.repeats)])

    disputed = _blocking(opts.base, opts.candidate, strict)
    if not disputed:
        print("# no blocking regressions to confirm")
        return 0
    print("# confirming {} blocking regression(s) at {} fresh seed(s): {}".format(
        len(disputed), len(seeds), seeds))

    n_confirmed = 0
    for k, brec, crec in disputed:
        worse = same = 0
        detail = []
        for s in seeds:
            rb = _rerun(opts.base_checkout, brec, s, opts.jobs, "base")
            rc = _rerun(opts.cand_checkout, crec, s, opts.jobs, "cand")
            if rb is None or rc is None:
                detail.append("seed {}: RERUN FAILED".format(s))
                continue
            sb = evaluate(rb)[0]
            sc = evaluate(rc)[0]
            if sb == "PASS" and sc != "PASS":
                worse += 1
            else:
                same += 1
            detail.append("seed {}: base={} cand={} (n_eff {:.0f} vs {:.0f})".format(
                s, sb, sc, rb.get("n_eff", float("nan")), rc.get("n_eff", float("nan"))))
        confirmed = worse > same
        n_confirmed += int(confirmed)
        print("\n{} {}".format(k[0], k[1]))
        for d in detail:
            print("   " + d)
        print("   -> {} ({} worse / {} not-worse across fresh seeds)".format(
            "CONFIRMED REGRESSION -- BLOCKS" if confirmed
            else "NOT CONFIRMED (realization noise; does not block)", worse, same))

    print("\n# confirmed blocking regressions: {}".format(n_confirmed))
    return 1 if n_confirmed else 0


if __name__ == "__main__":
    sys.exit(main())
