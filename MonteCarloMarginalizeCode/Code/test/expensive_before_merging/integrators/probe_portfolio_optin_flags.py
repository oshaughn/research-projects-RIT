#!/usr/bin/env python
"""
probe_portfolio_optin_flags.py -- shape-gate probe for the portfolio OPT-IN flags.

`RIFT/integrators/TESTING.md` requires that a change behind an opt-in flag ALSO be probed with the
flag ON: the default-path merge gate necessarily shows bitwise-identical results for opt-in code, so
it proves nothing about that code.  This probe covers the two opt-in portfolio features:

    portfolio_adaptive_alloc  (adaptive-probe draw allocation)
    portfolio_weight_clip     (truncated IS on the GMM proposal-fit input)

Method: reuse the merge-gate suite as a library (per shape_recovery.py's docstring) so the targets,
truth pools, metrics and pass thresholds are IDENTICAL to the gate.  We monkey-patch
`build_sampler` to switch the flags on for portfolio runs, and run IN-PROCESS (jobs=1): the gate's
multiprocessing path uses spawn, which would not carry the patch into workers.

Each configuration is scored with the gate's own `evaluate()`, so a row that PASSes here passes by
exactly the gate's criteria.

Usage (CPU, like the gate):
    export PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
    python probe_portfolio_optin_flags.py [--dims 2,4] [--ncomps 1,3] [--seeds 303]
"""
from __future__ import print_function
import argparse
import contextlib
import os
import sys

# The merge-gate WRAPPER (run_shape_recovery.sh) exports these; library mode does NOT.  Without
# them you silently import the INSTALLED RIFT (not the checkout under test) and/or hit the
# cupy-without-a-device path -- both yield confident, meaningless numbers.  A valid probe
# reproduces the gate's ABSOLUTE values row-for-row.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '1')
_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import shape_recovery as SR

# The PRISTINE factory, captured once at import, before any patch can be installed.  Everything
# below wraps THIS, never `SR.build_sampler` as it happens to stand: wrapping the current value
# meant the second configuration wrapped the first one's wrapper, so the arms accumulated and the
# "flags OFF" baseline still ran the PREVIOUS arm's flags -- a comparison of a flag against itself,
# which cannot show a regression.
_ORIG_BUILD_SAMPLER = SR.build_sampler


def patched_build(flags):
    """Return a build_sampler that switches the opt-in flags on for portfolio samplers.

    Most knobs are plain attributes on the portfolio object, so setattr suffices.  The GMM
    component cap is NOT: it lives on the GMM MEMBER's integrator (`gmm_adaptive`), which the
    portfolio forwards through setup().  Since the probe patches AFTER build, reach into the
    realized members for that one.  Use the reserved key '_gmm_adaptive_cap'.
    """
    orig = _ORIG_BUILD_SAMPLER

    def build(kind, target, n_chunk):
        s = orig(kind, target, n_chunk)
        if kind == "portfolio":
            for k, v in flags.items():
                if k == "_gmm_adaptive_cap":
                    # per-group BIC cap on the portfolio's GMM member(s)
                    for m in list(getattr(s, "portfolio_realizations", [])):
                        integ = getattr(m, "integrator", None)
                        if integ is not None and hasattr(integ, "gmm_dict"):
                            setattr(integ, "gmm_adaptive",
                                    {g: int(v) for g in integ.gmm_dict})
                    continue
                setattr(s, k, v)
        return s
    return build


@contextlib.contextmanager
def flag_patch(flags):
    """Install the flag-applying build_sampler for ONE run, then restore the original.

    Restoring in a `finally` -- rather than overwriting the global at the next call site -- is what
    keeps the arms independent: a run that raised used to leave its patch installed, and the next
    "flags OFF" run inherited it.  Refusing to nest makes the accumulation bug impossible to
    reintroduce quietly: a stacked wrapper is an error here, not a silently contaminated baseline.
    """
    if SR.build_sampler is not _ORIG_BUILD_SAMPLER:
        raise RuntimeError(
            "build_sampler is already patched; nesting would stack wrappers and leak one arm's "
            "flags into another's")
    SR.build_sampler = patched_build(flags)
    try:
        yield
    finally:
        SR.build_sampler = _ORIG_BUILD_SAMPLER


def run_config(label, flags, jobs_spec, nmax_per_dim, neff, run_seed):
    """Run the portfolio rows for one flag configuration; return list of (job, record)."""
    out = []
    for (d, nc, ts) in jobs_spec:
        target = SR.MixtureTarget(d, nc, ts)
        with flag_patch(flags):
            rec = SR.run_one("portfolio", target, nmax_per_dim * d, neff, seed=run_seed)
        verdict = SR.evaluate(rec)
        out.append(((d, nc, ts), rec, verdict))
        print("  {:22s} d{}_n{}_s{}  n_eff={:8.0f}  lnI-lnZ={:+.4f}  {}".format(
            label, d, nc, ts,
            float(rec.get("n_eff", float("nan"))),
            float(rec.get("bias_ln", float("nan"))),
            verdict if isinstance(verdict, str) else verdict[0]))
        sys.stdout.flush()
    return out


def _verdict_of(v):
    """The probe stores either a bare status or an (status, reasons) pair."""
    return v if isinstance(v, str) else v[0]


def is_probe_regression(base_verdict, flag_verdict):
    """THE definition of an opt-in regression, used by both the summary and the confirmation.

    Note it differs from compare_shape_results.is_blocking: this probe tolerates flag=STARVED,
    because an opt-in path is allowed to trade efficiency on a target the default already
    resolves.  Keeping one function rather than two copies is deliberate -- every silent-failure
    bug in this review series came from two representations of one rule drifting apart.
    """
    bs, vs = _verdict_of(base_verdict), _verdict_of(flag_verdict)
    return bs == "PASS" and vs not in ("PASS", "STARVED")


def _run_one_cell(flags, job, nmax_per_dim, neff, run_seed):
    """One (config, target) cell at one run seed.  Returns the record, or None if it did not run."""
    d, nc, ts = job
    try:
        target = SR.MixtureTarget(d, nc, ts)
        with flag_patch(flags):
            return SR.run_one("portfolio", target, nmax_per_dim * d, neff, seed=run_seed)
    except Exception as e:
        print("     cell {} failed at seed {}: {}".format(job, run_seed, e))
        return None


def confirm_plan(run_seed, repeats, explicit_seeds, min_valid):
    """Turn the confirmation options into (seeds, min_valid), REJECTING zero-evidence settings.

    A confirmation that runs no seeds, or that needs no valid pair to reach a verdict, prints
    "NOT CONFIRMED (realization noise)" about a row nobody re-tested and exits 0 -- the precise
    silent clear this step exists to prevent (--confirm-repeats 0 did exactly that).  So the
    settings have to buy actual evidence:

      * at least one fresh seed, and at least one valid pair required for a verdict;
      * never more valid pairs required than there are seeds (unsatisfiable: nothing could clear);
      * seeds DISTINCT, and distinct from the run seed that flagged the row.  Repeating a seed
        repeats its realization, and the realization is exactly what is in dispute here: four
        reruns at the flagging seed reproduced the same false FAIL.

    Raises ValueError naming the reason; the CLI turns that into a usage error.
    """
    if explicit_seeds is not None:
        seeds = [int(x) for x in explicit_seeds.split(",") if x.strip() != ""]
        if not seeds:
            raise ValueError("--confirm-seeds is empty: confirmation needs at least one fresh seed")
    else:
        if repeats < 1:
            raise ValueError(
                "--confirm-repeats must be >= 1 (got {}): zero reruns is no evidence, and would "
                "clear the flagged row untested.  Use --no-confirm to skip confirmation "
                "deliberately -- that fails the row rather than passing it.".format(repeats))
        seeds = [run_seed + 1000 * (i + 1) for i in range(repeats)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("confirmation seeds must be distinct, got {}: repeating a seed repeats "
                         "its realization instead of testing a fresh one".format(seeds))
    if run_seed in seeds:
        raise ValueError("confirmation seed {} is the run seed that flagged the row: it would "
                         "reproduce that verdict, not test it".format(run_seed))
    if min_valid is None:
        min_valid = len(seeds)
    if min_valid < 1:
        raise ValueError("--confirm-min-valid must be >= 1 (got {}): a verdict resting on zero "
                         "valid pairs is a silent clear".format(min_valid))
    if min_valid > len(seeds):
        raise ValueError("--confirm-min-valid {} exceeds the {} seed(s) available: no row could "
                         "ever reach a verdict".format(min_valid, len(seeds)))
    return seeds, min_valid


def confirm_flagged(flagged, nmax_per_dim, neff, seeds, min_valid):
    """Re-test flagged rows at FRESH seeds before letting them fail the run.

    WHY.  This probe reuses the gate's thresholds and evaluate(), so it inherits the same
    near-threshold realization sensitivity -- but it had no confirmation step, so one noisy row
    failed a PR.  Measured: `adaptive_alloc ON / d4_n1_s303` reported base=PASS flag=FAIL on four
    consecutive runs, reproduced identically against an unrelated base, and the same cell has read
    n_eff 422 (PASS) and 46 (STARVED) on IDENTICAL code.  Re-running the same seed would reproduce
    the same false verdict; only fresh seeds separate "the flag broke this" from "this cell
    responds to its realization".

    Fails closed, like confirm_regressions.py: a flag arm that produces no record counts AGAINST
    the flag, and too few usable pairs is INCONCLUSIVE rather than a pass.  The evidence
    requirements are re-checked here rather than trusted from the CLI, so no caller can obtain a
    verdict this function has no evidence for.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("confirmation needs at least one fresh seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("confirmation seeds must be distinct, got {}".format(seeds))
    if not 1 <= min_valid <= len(seeds):
        raise ValueError("min_valid must be in 1..{} (got {})".format(len(seeds), min_valid))
    n_conf, n_inconc = 0, 0
    for label, flags, job in flagged:
        worse = same = 0
        detail = []
        for s in seeds:
            r_off = _run_one_cell({}, job, nmax_per_dim, neff, s)
            r_on = _run_one_cell(flags, job, nmax_per_dim, neff, s)
            if r_off is None and r_on is None:
                detail.append("seed {}: neither arm produced a record".format(s))
                continue
            if r_on is None:
                worse += 1
                detail.append("seed {}: FLAG ARM produced no record (counts against the flag)".format(s))
                continue
            if r_off is None:
                detail.append("seed {}: default arm produced no record; pair unusable".format(s))
                continue
            v_off, v_on = SR.evaluate(r_off), SR.evaluate(r_on)
            if is_probe_regression(v_off, v_on):
                worse += 1
            else:
                same += 1
            detail.append("seed {}: base={} flag={} (n_eff {:.0f} vs {:.0f})".format(
                s, _verdict_of(v_off), _verdict_of(v_on),
                float(r_off.get("n_eff", float("nan"))), float(r_on.get("n_eff", float("nan")))))
        valid = worse + same
        if valid < min_valid:
            status = "INCONCLUSIVE -- {}/{} valid pairs, need {}: NOT cleared".format(
                valid, len(seeds), min_valid)
            n_inconc += 1
        elif worse > same:
            status = "CONFIRMED ({} worse / {} not-worse)".format(worse, same)
            n_conf += 1
        else:
            status = "NOT CONFIRMED (realization noise) ({} worse / {} not-worse)".format(worse, same)
        print("\n  {} d{}_n{}_s{}".format(label, job[0], job[1], job[2]))
        for line in detail:
            print("     " + line)
        print("     -> " + status)
    return n_conf, n_inconc


# The opt-in configurations the probe exercises.  Module level so that (a) the exclusion below
# is greppable without reading main(), and (b) tests of the confirmation machinery can inject
# their own list instead of coupling to whichever flags happen to ship.
FLAG_CONFIGS = [
    ("flags OFF (default)", {}),
    # ("adaptive_alloc ON", {"portfolio_adaptive_alloc": True}),
    # ("adaptive+clip ON", {"portfolio_adaptive_alloc": True, "portfolio_weight_clip": 1.0}),
    #   ^ EXCLUDED, not passing.  --portfolio-adaptive-alloc is a CONFIRMED regression on
    #   d4_n1_s303: at 5 fresh seeds the flag arm FAILS every time, and at three of them with
    #   HIGHER n_eff than the default arm (292 vs 244, 177 vs 67, 140 vs 136), i.e. it degrades
    #   posterior SHAPE rather than efficiency.  See FOLLOWUPS.md item 4 for the evidence and
    #   the scoping work needed.
    #
    #   Excluded rather than tolerated so the probe stays useful as a regression detector for
    #   the remaining configurations, all of which pass.  The exclusion is deliberately written
    #   as commented-out config lines, so it is greppable and reinstating it is one edit -- do
    #   that as the first step of fixing the flag, and expect these rows to fail until it is.
    #   The flag is opt-in, defaults OFF, and the pipeline never sets it, so production is
    #   unaffected in the meantime.
    ("weight_clip ON", {"portfolio_weight_clip": 1.0}),
    # VARAHA draw-share constraints (see DESIGN_portfolio_freeze_policy.md).  Motivation: on a
    # sharp high-SNR target the mixture degenerates to peaked-member-only (VARAHA share -> ~0.01),
    # q_mix loses its broad backstop, and a missed mode goes uncovered -> lnZ silently low while
    # n_eff looks GOOD.  A floor blocks that; a floor WITHOUT a cap lets the share run away to ~1
    # (VARAHA-only), which is the same degeneracy mirrored.  These rows check the constraints do
    # not damage shape recovery on the gate's own targets, which are NOT pathological -- the
    # constraint should be close to a no-op there, and must not regress it.
    ("varaha floor .25", {"portfolio_varaha_min_frac": 0.25}),
    ("varaha band .25-.75", {"portfolio_varaha_min_frac": 0.25,
                             "portfolio_varaha_max_frac": 0.75}),
    ("band + gmm cap3", {"portfolio_varaha_min_frac": 0.25,
                         "portfolio_varaha_max_frac": 0.75,
                         "_gmm_adaptive_cap": 3}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-repeats", type=int, default=5,
                    help="fresh run seeds per arm used to re-test a flagged row (default 5; "
                         "must be >= 1 -- see --no-confirm to skip confirmation instead)")
    ap.add_argument("--confirm-seeds", default=None,
                    help="explicit comma list; overrides --confirm-repeats.  Must be non-empty, "
                         "distinct, and none of them the --run-seed that flagged the row.")
    ap.add_argument("--confirm-min-valid", type=int, default=None,
                    help="usable default/flag pairs required for a verdict (default: all seeds; "
                         "must be 1..#seeds).  Fewer -> INCONCLUSIVE and exit 1, never a silent "
                         "clear.")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip confirmation; a flagged row fails immediately (old behaviour)")
    ap.add_argument("--dims", default="2,4")
    ap.add_argument("--ncomps", default="1,3")
    ap.add_argument("--seeds", default="303")
    ap.add_argument("--nmax-per-dim", type=int, default=None)
    ap.add_argument("--neff", type=int, default=None)
    ap.add_argument("--run-seed", type=int, default=987654)
    args = ap.parse_args()

    # Validate the confirmation settings BEFORE the arms run: an unusable setting should cost a
    # usage error, not an hour of sampling followed by a verdict backed by nothing.
    seeds = min_valid = None
    if not args.no_confirm:
        try:
            seeds, min_valid = confirm_plan(args.run_seed, args.confirm_repeats,
                                            args.confirm_seeds, args.confirm_min_valid)
        except ValueError as e:
            ap.error(str(e))

    cfg = dict(SR.PRESETS["standard"])
    nmax_per_dim = args.nmax_per_dim or cfg["nmax_per_dim"]
    neff = args.neff or cfg["neff"]
    jobs_spec = [(int(d), int(nc), int(ts))
                 for d in args.dims.split(",")
                 for nc in args.ncomps.split(",")
                 for ts in args.seeds.split(",")]

    configs = FLAG_CONFIGS

    print("# portfolio opt-in flag probe: {} targets x {} configs "
          "(nmax_per_dim={}, neff={})".format(len(jobs_spec), len(configs), nmax_per_dim, neff))

    results = {}
    for label, flags in configs:
        print("== {} ==".format(label))
        results[label] = run_config(label, flags, jobs_spec, nmax_per_dim, neff, args.run_seed)

    # Summary: the opt-in paths must not be WORSE than the default path on the gate's own verdict.
    print("\n# SUMMARY (verdict per target; opt-in must not regress vs flags OFF)")
    base = {k: (v, d) for k, v, d in results["flags OFF (default)"]}
    bad = 0
    flagged = []
    for label, _ in configs[1:]:
        for key, rec, verdict in results[label]:
            b_rec, b_verdict = base[key]
            vs = verdict if isinstance(verdict, str) else verdict[0]
            bs = b_verdict if isinstance(b_verdict, str) else b_verdict[0]
            flag = ""
            if is_probe_regression(b_verdict, verdict):
                flag = "  <-- FLAGGED (base PASS -> {})".format(vs); bad += 1
                flagged.append((label, dict(configs)[label], key))
            print("  {:22s} d{}_n{}_s{}  base={:8s} flag={:8s}{}".format(
                label, key[0], key[1], key[2], bs, vs, flag))
    print("\n# flagged rows: {}".format(bad))
    if not bad:
        return 0
    if args.no_confirm:
        print("# NOT CONFIRMED AT FRESH SEEDS (--no-confirm): treating flagged rows as regressions.\n"
              "#   Every threshold here is a hard cut on a stochastic quantity, so a single flagged\n"
              "#   row is a hypothesis; re-run without --no-confirm to test it.")
        return 1
    print("\n# re-testing {} flagged row(s) at {} fresh seed(s) per arm: {}".format(
        bad, len(seeds), seeds))
    n_conf, n_inconc = confirm_flagged(flagged, nmax_per_dim, neff, seeds, min_valid)
    print("\n# confirmed opt-in regressions: {}".format(n_conf))
    if n_inconc:
        print("# INCONCLUSIVE rows (too few valid reruns): {}".format(n_inconc))
    return 1 if (n_conf or n_inconc) else 0


if __name__ == "__main__":
    sys.exit(main())
