"""Known-answer tests for the averaged-amplitude distance model."""
import pytest
import numpy as np
from scipy.integrate import quad

from RIFT.interpolators.dslice_amplitude_model import DistanceAmplitudeModel, fit_point, log_model


def test_closed_form_matches_quadrature_including_deep_tails():
    R, us, fm = 40.0, 1 / 3000.0, 0.4
    for d in [300, 500, 1500, 3000, 6000, 9000, 1e5, 1e7]:
        x = (1 / d) / us
        num = np.log(quad(lambda f: np.exp(R * (1 - (1 - f * x) ** 2)), fm, 1, epsabs=0, epsrel=1e-12)[0] / (1 - fm))
        assert abs(log_model(1 / d, R, us, fm, 0.0) - num) < 1e-6, d


def test_far_limit_is_zero_and_flat_top_is_R():
    R, us, fm = 30.0, 1 / 2000.0, 0.5
    assert abs(log_model(1e-9, R, us, fm, 0.0)) < 1e-4
    top = log_model(np.linspace(us * 1.1, us / fm * 0.9, 20), R, us, fm, 0.0)
    assert np.all(np.abs(top - R) < 1.5)      # approximately flat across the degenerate range


def test_point_fit_reproduces_the_conditional_distance_posterior():
    """Slices placed over the central part of the conditional (as the export does); the fitted model's
    distance posterior with a d^2 prior must match the true one, tails included."""
    R, us, fm = 40.0, 1 / 3000.0, 0.4
    rng = np.random.default_rng(0)
    d = np.linspace(1400, 3200, 50)
    y = log_model(1 / d, R, us, fm, 0.0) + rng.normal(0, 0.15, 50)
    p, rms = fit_point(1 / d, y, np.ones(50) / 0.15 ** 2, fix_C=True)
    assert rms < 0.2 and p[3] == 0.0
    grid = np.linspace(300, 10000, 20000)

    def quantiles(lnl):
        w = np.exp(lnl - lnl.max()) * grid ** 2
        c = np.cumsum(w) / w.sum()
        return np.interp([0.01, 0.05, 0.5, 0.95, 0.99], c, grid)

    qt, qf = quantiles(log_model(1 / grid, R, us, fm, 0.0)), quantiles(log_model(1 / grid, *p))
    err = np.abs(qf / qt - 1)
    # one point, 50 noisy slices: the 1/99% tails carry more fit noise than the 5-95% body
    assert np.max(err[1:4]) < 0.03 and np.max(err[[0, 4]]) < 0.06, (qt, qf)


def test_model_interpolates_fields_over_intrinsic_coordinates():
    rng = np.random.default_rng(1)
    rows = []
    for x0 in rng.uniform(-1, 1, 200):
        R, us = 30 + 5 * x0, 1 / (3000 * (1 + 0.2 * x0))
        d = rng.uniform(1500, 9000, 30)
        rows.append(np.column_stack([np.full(30, x0), d, log_model(1 / d, R, us, 0.4, 0.0) + rng.normal(0, 0.1, 30)]))
    g = np.vstack(rows)
    m = DistanceAmplitudeModel(dist_index=1, n_jobs=1).fit(g[:, :2], g[:, 2], 0.1 * np.ones(len(g)))
    assert m.report["points_fit"] == 200
    q = np.column_stack([np.zeros(5), np.array([2000, 3000, 4000, 6000, 8000.0])])
    truth = log_model(1 / q[:, 1], 30.0, 1 / 3000.0, 0.4, 0.0)
    assert np.max(np.abs(m.predict(q) - truth)) < 1.0


def test_decomposition_makes_the_intrinsic_marginal_the_interpolated_M():
    import pickle
    rng = np.random.default_rng(2)
    rows = []
    for x0 in rng.uniform(-1, 1, 150):
        R, us = 30 + 5 * x0, 1 / (3000 * (1 + 0.2 * x0))
        d = rng.uniform(1500, 6000, 30)
        rows.append(np.column_stack([np.full(30, x0), d, log_model(1 / d, R, us, 0.4, 0.0) + rng.normal(0, 0.1, 30)]))
    g = np.vstack(rows)
    prior = lambda d: np.asarray(d) ** 2
    m = DistanceAmplitudeModel(dist_index=1, n_jobs=1, prior=prior, d_range=(500.0, 10000.0)).fit(
        g[:, :2], g[:, 2], 0.1 * np.ones(len(g)))
    m = pickle.loads(pickle.dumps(m))                   # must survive save/reload
    x0 = 0.1
    dg = np.linspace(500, 10000, 4000)
    lnl = m.predict(np.column_stack([np.full(len(dg), x0), dg]))
    marg = np.log(np.trapz(np.exp(lnl) * dg ** 2, dg))
    M = m.rf.predict(np.array([[x0]]))[0, 3]
    assert abs(marg - M) < 0.02, (marg, M)


# ---- batched per-point fits and the array-module model (GPU path) ------------------------------
from RIFT.interpolators.dslice_amplitude_model import (fit_all_points, fit_all_points_batched,
                                                        fit_points_batched, log_model_xp)


def _synthetic_points(n_pts=300, seed=5):
    """n_pts intrinsic points with known (R, u*, f_min), 20-60 noisy slices each; returns key, u, y, sig, truth."""
    rng = np.random.default_rng(seed)
    key, u, y, sig, truth = [], [], [], [], []
    for k in range(n_pts):
        R, us, fm = rng.uniform(15, 60), 1 / rng.uniform(1500, 5000), rng.uniform(0.2, 0.7)
        n = int(rng.integers(20, 61))
        d = rng.uniform(0.4, 1.6, n) / us * (0.5 + fm)        # around the flat top
        s = rng.uniform(0.05, 0.3, n)
        key.append(np.full((n, 1), float(k))); u.append(1 / d); sig.append(s)
        y.append(log_model(1 / d, R, us, fm, 0.0) + rng.normal(0, s))
        truth.append([R, us, fm])
    return np.vstack(key), np.concatenate(u), np.concatenate(y), np.concatenate(sig), np.array(truth)


def _costs(key, u, y, sig, P):
    w = 1 / np.maximum(sig, 1e-3) ** 2
    g = key[:, 0].astype(int)
    r2 = w * (log_model(u, P[g, 0], P[g, 1], P[g, 2], P[g, 3]) - y) ** 2
    return np.bincount(g, weights=r2)


def test_log_model_xp_numpy_is_log_model():
    u = np.geomspace(1e-5, 1e-2, 500)
    assert np.array_equal(log_model_xp(u, 30.0, 1 / 3000.0, 0.4, 0.0), log_model(u, 30.0, 1 / 3000.0, 0.4, 0.0))


def test_batched_fit_reaches_the_scipy_optimum():
    key, u, y, sig, truth = _synthetic_points()
    _, Ps, _, _ = fit_all_points(key, u, y, sig, fix_C=True)
    _, Pb, rb, _ = fit_all_points_batched(key, u, y, sig, fix_C=True, block=128)   # several blocks
    cs, cb = _costs(key, u, y, sig, Ps), _costs(key, u, y, sig, Pb)
    assert np.all(np.isfinite(Pb)) and np.all(Pb[:, 3] == 0.0)
    assert np.all(cb <= cs * (1 + 1e-5) + 1e-8), np.max((cb - cs) / cs)
    # same optimum, so the same curve over each point's slices
    g = key[:, 0].astype(int)
    dcurve = np.abs(log_model(u, *(Pb[g, j] for j in range(4))) - log_model(u, *(Ps[g, j] for j in range(4))))
    assert np.max(dcurve) < 0.05
    assert np.median(np.abs(np.log(Pb[:, 0] / truth[:, 0]))) < 0.1


def test_unconverged_batched_fit_fails_the_same_check():
    """The criterion above discriminates: stopping the optimizer at its start point is caught."""
    key, u, y, sig, _ = _synthetic_points(n_pts=50)
    _, Ps, _, _ = fit_all_points(key, u, y, sig, fix_C=True)
    import RIFT.interpolators.dslice_amplitude_model as D
    real = D.fit_points_batched
    try:
        D.fit_points_batched = lambda *a, **k: real(*a, **dict(k, max_iter=0))
        _, P0, _, _ = D.fit_all_points_batched(key, u, y, sig, fix_C=True)
    finally:
        D.fit_points_batched = real
    cs, c0 = _costs(key, u, y, sig, Ps), _costs(key, u, y, sig, P0)
    assert not np.all(c0 <= cs * (1 + 1e-5) + 1e-8)


def test_batched_model_matches_scipy_model_on_cupy():
    import pytest
    try:
        import cupy as cp
        cp.zeros(1)
    except Exception as err:
        pytest.skip("needs cupy and a CUDA device: %s" % type(err).__name__)
    key, u, y, sig, _ = _synthetic_points(n_pts=200)
    _, Pn, _, _ = fit_all_points_batched(key, u, y, sig, fix_C=True)
    _, Pc, _, _ = fit_all_points_batched(key, u, y, sig, fix_C=True, xp=cp)
    cn, cc = _costs(key, u, y, sig, Pn), _costs(key, u, y, sig, Pc)
    assert np.allclose(cc, cn, rtol=1e-6, atol=1e-8)
    # device predict == host predict for a fitted, decomposed model
    rng = np.random.default_rng(2)
    rows = []
    for x0 in rng.uniform(-1, 1, 150):
        R, us = 30 + 5 * x0, 1 / (3000 * (1 + 0.2 * x0))
        d = rng.uniform(1500, 6000, 30)
        rows.append(np.column_stack([np.full(30, x0), d, log_model(1 / d, R, us, 0.4, 0.0) + rng.normal(0, 0.1, 30)]))
    g = np.vstack(rows)
    m = DistanceAmplitudeModel(dist_index=1, n_jobs=1, prior=lambda d: np.asarray(d) ** 2, d_range=(500.0, 10000.0),
                               point_fit="batched", xp=cp).fit(g[:, :2], g[:, 2], 0.1 * np.ones(len(g)))
    q = np.column_stack([rng.uniform(-1, 1, 2000), rng.uniform(500, 10000, 2000)])
    assert np.max(np.abs(cp.asnumpy(m.predict_device(q)) - m.predict(q))) < 1e-9


def test_device_check_tolerates_degenerate_fits_and_catches_a_bad_forest():
    """check_device: a runaway fit (R ~ 1e10, f_min ~ 0) differs between paths by rounding of order
    100 eps R and must pass; a corrupted device leaf must fail."""
    import pytest
    try:
        import cupy as cp
        cp.zeros(1)
    except Exception as err:
        pytest.skip("needs cupy and a CUDA device: %s" % type(err).__name__)
    rng = np.random.default_rng(3)
    rows = []
    for x0 in rng.uniform(-1, 1, 150):
        R, us = 30 + 5 * x0, 1 / (3000 * (1 + 0.2 * x0))
        d = rng.uniform(1500, 6000, 30)
        rows.append(np.column_stack([np.full(30, x0), d, log_model(1 / d, R, us, 0.4, 0.0) + rng.normal(0, 0.1, 30)]))
    g = np.vstack(rows)
    m = DistanceAmplitudeModel(dist_index=1, n_jobs=1, prior=lambda d: np.asarray(d) ** 2,
                               d_range=(500.0, 30000.0)).fit(g[:, :2], g[:, 2], 0.1 * np.ones(len(g)))
    for e in m.rf.estimators_:          # make one region degenerate, as on S241201ac
        t = e.tree_
        t.value[:, 0, 0] = np.where(t.value[:, 0, 0] > np.log(33), np.log(4.5e10), t.value[:, 0, 0])
        t.value[:, 1, 0] = np.where(t.value[:, 0, 0] > 20, np.log(6.9e8), t.value[:, 1, 0])
        t.value[:, 2, 0] = np.where(t.value[:, 0, 0] > 20, -45.0, t.value[:, 2, 0])
    q = np.column_stack([rng.uniform(-1, 1, 4000), rng.uniform(500, 30000, 4000)])
    ok, rep = m.check_device(q)
    assert ok, rep
    assert rep["lnl_max_abs"] > 1e-8, rep     # the flat 1e-8 check would fail here
    m._device["forest"]._val[:, 0] += 0.5
    ok, rep = m.check_device(q)
    assert not ok and rep["field_max_rel"] > 1e-3, rep


def test_parallel_scipy_point_fits_equal_serial():
    key, u, y, sig, _ = _synthetic_points(n_pts=120)
    a = fit_all_points(key, u, y, sig, fix_C=True)
    b = fit_all_points(key, u, y, sig, fix_C=True, n_jobs=2)
    for x1, x2 in zip(a, b):
        assert np.array_equal(x1, x2, equal_nan=True)


# ---- review additions: mass scaling, overshoot guard, fixed f_min, normalization refinement ------
def _grid(n_pts, d_lo, d_hi, R_of, us_of, fm=0.4, noise=0.1, seed=7, mass=None):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_pts):
        x0 = rng.uniform(-1, 1)
        M = 1.0 if mass is None else mass(x0)
        d = rng.uniform(d_lo, d_hi, 30)
        lnl = log_model(1 / d, R_of(x0), us_of(x0, M), fm, 0.0) + rng.normal(0, noise, 30)
        rows.append(np.column_stack([np.full(30, x0), np.full(30, M), d, lnl]))
    return np.vstack(rows)


def test_mass_scaled_distance_field_predicts_the_truth():
    """u* = c / M exactly: interpolating log(u* M) is constant; a sign error in the mass term breaks it."""
    g = _grid(200, 1500, 9000, lambda x: 30.0, lambda x, M: 1 / (3000.0 * M), mass=lambda x: 0.6 + 0.8 * (x + 1))
    m = DistanceAmplitudeModel(dist_index=2, mass_index=1, n_jobs=1).fit(g[:, :3], g[:, 3], 0.1 * np.ones(len(g)))
    q = np.column_stack([np.zeros(4), np.array([0.8, 1.2, 1.6, 2.0]), np.full(4, 3500.0)])
    truth = log_model(1 / q[:, 2], 30.0, 1 / (3000.0 * q[:, 1]), 0.4, 0.0)
    assert np.max(np.abs(m.predict(q) - truth)) < 0.5, (m.predict(q), truth)


def test_overshoot_guard_drops_unresolved_points_at_any_distance_scale():
    """Points whose slices sit on the far side of the peak alone (no flat top seen) can fit a peak far
    above what they measured; the guard must drop them, including for a nearby source (d* ~ 40 Mpc)."""
    rng = np.random.default_rng(3)
    for dstar in (40.0, 3000.0):
        R, us = 1000.0, 1 / dstar
        rows = []
        for k in range(60):
            d = rng.uniform(3.0, 3.6, 25) * dstar      # far tail only: lnL well below the peak
            rows.append(np.column_stack([np.full(25, float(k)), d, log_model(1 / d, R, us, 0.5, 0.0) + rng.normal(0, 0.3, 25)]))
        g = np.vstack(rows)
        m = DistanceAmplitudeModel(dist_index=1, n_jobs=1, max_overshoot=1.0)
        try:
            m.fit(g[:, :2], g[:, 2], 0.3 * np.ones(len(g)))
        except ValueError:
            pass                                       # every point dropped: nothing left to interpolate
        assert m.report["dropped_overshoot"] > 0, (dstar, m.report)


def test_fixed_fmin_is_respected_and_costs_no_less_than_free():
    R, us, fm = 40.0, 1 / 3000.0, 0.4
    rng = np.random.default_rng(1)
    d = np.linspace(1400, 3200, 50)
    y = log_model(1 / d, R, us, fm, 0.0) + rng.normal(0, 0.15, 50)
    w = np.ones(50) / 0.15 ** 2
    pf, _ = fit_point(1 / d, y, w, fix_C=True)
    px, _ = fit_point(1 / d, y, w, fix_C=True, fmin_fixed=0.25)
    assert px[2] == pytest.approx(0.25, abs=1e-12)
    cost = lambda p: np.sum(w * (log_model(1 / d, *p) - y) ** 2)
    assert cost(px) >= cost(pf) - 1e-9


def test_normalization_grid_refines_for_narrow_distance_peaks():
    """A loud, nearby source has a distance peak narrower than a 256-node grid over [1, 2000] Mpc can
    integrate; the fit must refine the grid until M(x) is the marginal of the predicted lnL."""
    g = _grid(120, 25, 70, lambda x: 1000.0, lambda x, M: 1 / (40.0 * (1 + 0.1 * x)), noise=0.3, seed=11)
    g = g[:, [0, 2, 3]]
    prior = lambda d: np.asarray(d) ** 2
    m = DistanceAmplitudeModel(dist_index=1, n_jobs=1, prior=prior, d_range=(1.0, 2000.0)).fit(
        g[:, :2], g[:, 2], 0.3 * np.ones(len(g)))
    assert m.report["dgrid_nodes"] > 256 and m.report["dgrid_check_nats"] <= 0.01, m.report
    for x0 in (-0.5, 0.0, 0.5):
        dg = np.linspace(1.0, 2000.0, 400000)
        lnl = m.predict(np.column_stack([np.full(len(dg), x0), dg]))
        top = lnl.max()
        marg = top + np.log(np.trapz(np.exp(lnl - top) * dg ** 2, dg))
        assert abs(marg - m.rf.predict(np.array([[x0]]))[0, 3]) < 0.05, (x0, marg)


def test_lognorm_chunks_bound_memory_and_keep_values():
    # _lognorm sizes its row chunks from the grid length: a refined grid (1e5 nodes) must not build a
    # (4096 x 1e5) temporary per chunk, and chunking must not change any value.
    import RIFT.interpolators.dslice_amplitude_model as dam
    rng = np.random.default_rng(2)
    n = 3000
    R, us, fm = rng.uniform(20, 400, n), 1 / rng.uniform(800, 3000, n), rng.uniform(0.1, 0.9, n)
    seen = []
    orig = dam.log_model
    def spy(u, *a):
        size = np.broadcast(u, *a[:3]).size          # checked before evaluating, so a regression fails fast
        assert size <= 2 ** 24, size
        seen.append(size)
        return orig(u, *a)
    g = np.linspace(100.0, 8000.0, 131072)
    w = np.log(g ** 2) + np.log(np.gradient(g))
    dam.log_model = spy
    try:
        big = dam.DistanceAmplitudeModel._lognorm(None, R, us, fm, dgrid=g, lw=w)
    finally:
        dam.log_model = orig
    assert max(seen) <= 2 ** 24 and np.all(np.isfinite(big))
    # same values as one row at a time on a short grid
    gs, ws = g[::64], w[::64]
    whole = dam.DistanceAmplitudeModel._lognorm(None, R[:50], us[:50], fm[:50], dgrid=gs, lw=ws)
    rows = np.array([dam.DistanceAmplitudeModel._lognorm(None, R[i:i + 1], us[i:i + 1], fm[i:i + 1], dgrid=gs, lw=ws)[0]
                     for i in range(50)])
    assert np.array_equal(whole, rows)
