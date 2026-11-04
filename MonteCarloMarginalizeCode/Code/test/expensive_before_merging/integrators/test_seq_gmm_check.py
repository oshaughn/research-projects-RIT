#!/usr/bin/env python
"""The sequential-case GMM clearing check must not pass VACUOUSLY.

A check keyed only on `cleared is False` treats three distinct failures as success: the member
never trained (so "cleared" is trivially true and proves nothing), the inspection raised (cleared
is None), or the fields are absent entirely.  A check that could not run is a failed check.

Run:  python test_seq_gmm_check.py
"""
import sys

import shape_recovery as SR


def _rec(**kw):
    """A record that passes every ordinary metric, so only the GMM check can move the verdict."""
    r = dict(kind="portfolio_seq_nobs", n_eff=2000.0, n_ess=9000.0,
             js=[0.0001, 0.0001], js_floor=[0.0005, 0.0005],
             mean_pull=[0.005, 0.005], width_ratio=[1.001, 1.001],
             corr_diff_max=0.005, rel_err=0.01, bias_ln=0.002, error=None)
    r.update(kw)
    return r


def test_healthy_case_passes():
    st, why = SR.evaluate(_rec(seq_gmm_trained_before_reset=True, seq_gmm_cleared=True))
    assert st == "PASS", (st, why)


def test_leak_fails():
    st, why = SR.evaluate(_rec(seq_gmm_trained_before_reset=True, seq_gmm_cleared=False))
    assert st == "FAIL", (st, why)
    assert any("did NOT clear" in w for w in why), why


def test_never_trained_is_not_a_pass():
    """"Cleared" is trivially true if nothing was ever trained -- that must not read as success."""
    st, why = SR.evaluate(_rec(seq_gmm_trained_before_reset=False, seq_gmm_cleared=True))
    assert st == "FAIL", "a vacuous 'cleared' verdict passed: {} {}".format(st, why)
    assert any("vacuous" in w for w in why), why


def test_inspection_failure_is_not_a_pass():
    st, why = SR.evaluate(_rec(seq_gmm_trained_before_reset=True, seq_gmm_cleared=None,
                               seq_gmm_error="AttributeError: no integrator"))
    assert st == "FAIL", "an unreadable GMM state passed: {} {}".format(st, why)
    assert any("AttributeError" in w for w in why), "the underlying error was not surfaced: {}".format(why)


def test_missing_fields_are_not_a_pass():
    st, why = SR.evaluate(_rec())
    assert st == "FAIL", "absent instrumentation passed: {} {}".format(st, why)


def test_av_seq_is_exempt():
    """AV_seq has no GMM member, so the check must not fire on it."""
    st, why = SR.evaluate(_rec(kind="AV_seq"))
    assert st == "PASS", (st, why)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("sequential GMM clearing check cannot pass vacuously")
