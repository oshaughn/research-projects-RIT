#!/usr/bin/env python
"""Unit tests for the confirm-on-fail accounting.

These cover the ways a confirmation step can WRONGLY CLEAR a real regression, which is the only
dangerous direction: a false block costs a rerun, a false clear ships a bug.

Run:  python test_confirm_regressions.py
"""
import sys

import confirm_regressions as CR
from compare_shape_results import classify, is_blocking

STRICT = {"GMM", "AV"}


def _rec(kind="GMM", target="t", n_eff=3000.0, js=0.0001, bias=0.001):
    return dict(kind=kind, target=target, ndim=4, ncomp=1, target_seed=101, n_eff=n_eff,
                n_ess=n_eff * 3, js=[js, js], js_floor=[0.0005, 0.0005],
                mean_pull=[0.005, 0.005], width_ratio=[1.001, 1.001], corr_diff_max=0.005,
                rel_err=0.01, bias_ln=bias, error=None)


def test_metrics_only_regression_is_recognised():
    """The comparator blocks on REGRESSION(metrics) too.  A confirm step that only knew about
    PASS->non-PASS reported 'nothing to confirm' and exited 0 on a real n_eff collapse."""
    b, c = _rec(n_eff=4000.0), _rec(n_eff=400.0)      # 10x n_eff drop, both still PASS
    verdict, _ = classify(b, c)
    assert verdict == "REGRESSION(metrics)", verdict
    assert is_blocking(verdict, "GMM", STRICT)


def _run_with(monkey_results, seeds=(1, 2, 3), min_valid=None):
    """Drive main() with _rerun stubbed to a scripted sequence of (base, cand) records."""
    calls = {"i": 0}

    def fake_rerun(checkout, rec, seed, jobs, tag):
        pair = monkey_results[calls["i"] // 2]
        out = pair[0] if tag == "base" else pair[1]
        calls["i"] += 1
        return out

    orig = CR._rerun
    CR._rerun = fake_rerun
    try:
        import json, tempfile, os
        b, c = _rec(n_eff=4000.0), _rec(n_eff=400.0)
        paths = []
        for recs in ([b], [c]):
            fd, p = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            json.dump(recs, open(p, "w"))
            paths.append(p)
        argv = [paths[0], paths[1], "--base-checkout", "/b", "--cand-checkout", "/c",
                "--seeds", ",".join(str(s) for s in seeds)]
        if min_valid is not None:
            argv += ["--min-valid", str(min_valid)]
        return CR.main(argv)
    finally:
        CR._rerun = orig


def test_candidate_crash_counts_against_the_candidate():
    """If the candidate produces no record where the base does, that IS the regression.
    Discarding those pairs let a candidate that failed on every seed be 'not confirmed'."""
    good = _rec(n_eff=4000.0)
    rc = _run_with([(good, None), (good, None), (good, None)])
    assert rc == 1, "candidate produced no record on every seed but was cleared (rc={})".format(rc)


def test_insufficient_valid_pairs_is_inconclusive_not_a_pass():
    """Missing evidence must not read as 'cleared'."""
    good = _rec(n_eff=4000.0)
    rc = _run_with([(None, None), (None, None), (None, None)])
    assert rc == 1, "zero valid pairs was reported as success (rc={})".format(rc)


def test_genuine_noise_clears():
    """A row that is equivalent at fresh seeds must clear, or the step is useless."""
    good = _rec(n_eff=4000.0)
    rc = _run_with([(good, good), (good, good), (good, good)])
    assert rc == 0, "equivalent arms were reported as a confirmed regression (rc={})".format(rc)


def test_real_regression_is_confirmed():
    good, bad = _rec(n_eff=4000.0), _rec(n_eff=200.0)
    rc = _run_with([(good, bad), (good, bad), (good, bad)])
    assert rc == 1, "a reproducible 20x n_eff drop was not confirmed (rc={})".format(rc)




def test_missing_candidate_record_is_a_blocking_regression():
    """A candidate that emits no record for a strict row must BLOCK.

    Classified as ONLY-IN-BASE it was not a regression, so it never reached confirmation and the
    gate exited 0 -- a candidate crashing before its first result would bypass the fail-closed
    rerun logic entirely."""
    b = _rec()
    verdict, note = classify(b, None)
    assert verdict.startswith("REGRESSION"), verdict
    assert is_blocking(verdict, "GMM", STRICT), "missing candidate record did not block"
    # and it must be picked up as a row to confirm
    from compare_shape_results import blocking_keys
    keys = blocking_keys({("GMM", "t"): b}, {}, STRICT)
    assert keys == [("GMM", "t")], keys


def test_extra_candidate_record_is_not_a_regression():
    """The reverse direction is not a defect: a NEW row in the candidate must not block."""
    verdict, _ = classify(None, _rec())
    assert not verdict.startswith("REGRESSION"), verdict


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("confirm-on-fail accounting holds")
