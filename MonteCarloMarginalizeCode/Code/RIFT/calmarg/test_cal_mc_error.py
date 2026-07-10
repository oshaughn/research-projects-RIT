"""Unit test for adaptive.cal_mc_error_from_components.

Validates the delta-method calibration MC error budget two ways, no GPU/lal needed:

1. LOGNORMAL CLOSED FORM: for lnL_c ~ N(mu, s^2) iid, Var(ln Zhat) = (e^{s^2}-1)/n_cal
   (to leading order).  The estimator must reproduce this from a single draw set.

2. BRUTE FORCE: redraw the cal set many times, measure the true scatter of
   ln Zhat = logsumexp_c(lnL_c) - log n_cal directly, compare.

Also checks the importance-weighted (cal_log_weights) path is unbiased, and that the
extrinsic batch weighting reduces to the same answer when responsibilities are
extrinsic-independent.

Run:  python -m RIFT.calmarg.test_cal_mc_error
"""
import numpy as np
from scipy.special import logsumexp

from RIFT.calmarg.adaptive import cal_mc_error_from_components


def _true_scatter(sig, n_cal, trials, rng):
    x = rng.normal(0.0, sig, size=(trials, n_cal))
    return np.std(logsumexp(x, axis=1) - np.log(n_cal))


def test_lognormal_closed_form():
    rng = np.random.default_rng(7)
    for sig in [0.3, 0.7, 1.0]:
        n_cal = 400
        analytic = np.sqrt((np.exp(sig ** 2) - 1.0) / n_cal)
        # average the estimator over independent draw sets (the estimator is itself
        # a one-draw-set statistic, so compare in expectation)
        est = []
        for _ in range(200):
            comp = rng.normal(0.0, sig, size=(1, n_cal))   # 1 extrinsic sample suffices
            s, neff, a = cal_mc_error_from_components(comp)
            est.append(s)
            assert abs(a.sum() - 1.0) < 1e-12
        est = np.mean(est)
        assert abs(est - analytic) / analytic < 0.15, (sig, est, analytic)
    print("test_lognormal_closed_form: OK")


def test_brute_force_scatter():
    rng = np.random.default_rng(11)
    sig, n_cal = 1.2, 100
    truth = _true_scatter(sig, n_cal, 20000, rng)
    est = np.mean([cal_mc_error_from_components(rng.normal(0, sig, (1, n_cal)))[0]
                   for _ in range(300)])
    # delta method degrades as neff_cal drops; require agreement within 30%
    assert abs(est - truth) / truth < 0.30, (est, truth)
    print("test_brute_force_scatter: OK (true {:.3f}, est {:.3f})".format(truth, est))


def test_neff_dominated():
    # one realization dominating -> neff ~ 1 and a loud (lower-bound) sigma
    comp = np.full((4, 50), -100.0)
    comp[:, 3] = 0.0
    s, neff, a = cal_mc_error_from_components(comp)
    assert neff < 1.5
    assert np.argmax(a) == 3
    print("test_neff_dominated: OK (neff {:.2f})".format(neff))


def test_importance_weights_consistency():
    # drawing from a proposal with weights must agree with prior draws in expectation
    rng = np.random.default_rng(3)
    n_cal, sig = 800, 0.8
    # prior draws
    s_prior = np.mean([cal_mc_error_from_components(rng.normal(0, sig, (1, n_cal)))[0]
                       for _ in range(100)])
    # 'proposal' = prior here, with identically zero log-weights: must match exactly in law
    s_w = np.mean([cal_mc_error_from_components(rng.normal(0, sig, (1, n_cal)),
                                                cal_log_weights=np.zeros(n_cal))[0]
                   for _ in range(100)])
    assert abs(s_prior - s_w) / s_prior < 0.2
    print("test_importance_weights_consistency: OK")


def test_extrinsic_batch_weighting():
    # responsibilities ~extrinsic-independent: a batch with a common per-sample offset
    # (the extrinsic-dependent part) must give the same answer as a single sample.
    rng = np.random.default_rng(5)
    n_cal = 200
    base = rng.normal(0, 1.0, n_cal)
    offsets = rng.normal(0, 5.0, 64)                  # huge extrinsic spread
    comp = offsets[:, None] + base[None, :]
    s_batch, neff_b, _ = cal_mc_error_from_components(comp)
    s_one, neff_1, _ = cal_mc_error_from_components(base[None, :])
    assert abs(s_batch - s_one) < 1e-10
    assert abs(neff_b - neff_1) < 1e-8
    print("test_extrinsic_batch_weighting: OK")


def test_total_underflow_guard():
    # A sample with -inf lnL across ALL cal draws (e.g. a zero-response extrinsic point)
    # must not poison the batch with NaN; it carries zero weight, so the answer must equal
    # the finite-samples-only answer.
    rng = np.random.default_rng(17)
    n_cal = 200
    good = rng.normal(0.0, 0.8, size=(8, n_cal))
    s_good, neff_good, a_good = cal_mc_error_from_components(good)
    comp = np.vstack([good, np.full((1, n_cal), -np.inf)])   # append a dead sample
    s, neff, a = cal_mc_error_from_components(comp)
    assert np.isfinite(s) and np.isfinite(neff), (s, neff)
    assert abs(a.sum() - 1.0) < 1e-12 and np.all(np.isfinite(a))
    assert abs(s - s_good) < 1e-9 and abs(neff - neff_good) < 1e-7, (s, s_good, neff, neff_good)
    # fully-degenerate batch (every sample dead) returns a finite sentinel, not NaN
    dead = np.full((3, n_cal), -np.inf)
    s0, neff0, a0 = cal_mc_error_from_components(dead)
    assert np.isinf(s0) and neff0 == 1.0 and abs(a0.sum() - 1.0) < 1e-12
    print("test_total_underflow_guard: OK")


if __name__ == "__main__":
    test_lognormal_closed_form()
    test_brute_force_scatter()
    test_neff_dominated()
    test_importance_weights_consistency()
    test_extrinsic_batch_weighting()
    test_total_underflow_guard()
    print("ALL OK")
