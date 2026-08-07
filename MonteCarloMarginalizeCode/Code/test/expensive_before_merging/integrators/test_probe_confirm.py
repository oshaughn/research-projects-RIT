#!/usr/bin/env python
"""Unit tests for the flag-ON probe's confirm-on-fail accounting.

Same principle as test_confirm_regressions.py: every check targets the direction that can ship a
bug -- a false CLEAR of a real opt-in regression. A false block only costs a rerun.

Run:  python test_probe_confirm.py
"""
import sys

import probe_portfolio_optin_flags as PB


def _rec(n_eff=3000.0, js=0.0001, bias=0.001):
    return dict(kind="portfolio", target="t", ndim=4, ncomp=1, target_seed=101, n_eff=n_eff,
                n_ess=n_eff * 3, js=[js, js], js_floor=[0.0005, 0.0005],
                mean_pull=[0.005, 0.005], width_ratio=[1.001, 1.001], corr_diff_max=0.005,
                rel_err=0.01, bias_ln=bias, error=None)


def test_regression_rule_tolerates_starved_but_not_fail():
    """The probe's rule differs from the comparator's on purpose: an opt-in path may trade
    efficiency on a target the default already resolves, but must not break it."""
    assert PB.is_probe_regression("PASS", "FAIL")
    assert not PB.is_probe_regression("PASS", "STARVED")
    assert not PB.is_probe_regression("PASS", "PASS")
    assert not PB.is_probe_regression("STARVED", "FAIL")     # base was not passing either
    # and it accepts the (status, reasons) pair form the probe actually stores
    assert PB.is_probe_regression(("PASS", []), ("FAIL", ["js too big"]))


def _drive(seq, seeds=(1, 2, 3), min_valid=None):
    """Run confirm_flagged with _run_one_cell stubbed to a scripted (default, flag) sequence."""
    calls = {"i": 0}

    def fake(flags, job, nmax_per_dim, neff, run_seed):
        pair = seq[calls["i"] // 2]
        out = pair[0] if (calls["i"] % 2 == 0) else pair[1]
        calls["i"] += 1
        return out

    orig = PB._run_one_cell
    PB._run_one_cell = fake
    try:
        return PB.confirm_flagged([("adaptive_alloc ON", {"portfolio_adaptive_alloc": True},
                                    (4, 1, 303))],
                                  200000, 3000, list(seeds),
                                  len(seeds) if min_valid is None else min_valid)
    finally:
        PB._run_one_cell = orig


def test_noise_clears():
    good = _rec()
    n_conf, n_inconc = _drive([(good, good)] * 3)
    assert (n_conf, n_inconc) == (0, 0), (n_conf, n_inconc)


def test_real_regression_confirms():
    good, bad = _rec(), _rec(js=0.05, bias=0.9)      # flag arm genuinely FAILs the metrics
    n_conf, n_inconc = _drive([(good, bad)] * 3)
    assert n_conf == 1, "a reproducible flag-arm failure was not confirmed"


def test_flag_arm_producing_no_record_counts_against_the_flag():
    """Crashing is worse than passing; discarding those pairs would clear a flag that always dies."""
    good = _rec()
    n_conf, n_inconc = _drive([(good, None)] * 3)
    assert n_conf == 1, "flag arm produced no record on every seed but was cleared"


def test_no_valid_pairs_is_inconclusive_not_a_pass():
    n_conf, n_inconc = _drive([(None, None)] * 3)
    assert n_inconc == 1 and n_conf == 0, (n_conf, n_inconc)


def test_minority_worse_does_not_confirm():
    """One bad seed out of three is the realization sensitivity this exists to absorb."""
    good, bad = _rec(), _rec(js=0.05, bias=0.9)
    n_conf, n_inconc = _drive([(good, bad), (good, good), (good, good)])
    assert (n_conf, n_inconc) == (0, 0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("probe confirm-on-fail accounting holds")
