"""RFDistanceTails: base fit kept inside each point's slices, physical continuation outside."""
import numpy as np

from RIFT.interpolators.dslice_amplitude_model import log_model
from RIFT.interpolators.dslice_rf_tails import RFDistanceTails


def _grid(seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for x0 in rng.uniform(-1, 1, 120):
        R, us = 30 + 5 * x0, 1 / (3000 * (1 + 0.2 * x0))
        d = np.linspace(1800, 4000, 30)
        rows.append(np.column_stack([np.full(30, x0), d, log_model(1 / d, R, us, 0.4, 0.0) + rng.normal(0, 0.05, 30)]))
    return np.vstack(rows)


def test_inside_is_base_outside_is_continuation():
    g = _grid()
    flat = lambda x: np.full(len(x), 7.0)          # a 'tree' that is flat everywhere
    m = RFDistanceTails(flat, dist_index=1, sides="both").fit(g[:, :2], g[:, 2], 0.05 * np.ones(len(g)))
    x0 = g[0, 0]
    inside = m(np.array([[x0, 2500.0]]))
    assert inside[0] == 7.0
    near, far = m(np.array([[x0, 600.0]])), m(np.array([[x0, 9000.0]]))
    assert near[0] < 7.0 - 1 and far[0] < 7.0 - 1      # both tails fall away from the flat edge value
    near_only = RFDistanceTails(flat, 1, sides="near").fit(g[:, :2], g[:, 2], 0.05 * np.ones(len(g)))
    assert near_only(np.array([[x0, 9000.0]]))[0] == 7.0 and near_only(np.array([[x0, 600.0]]))[0] < 6.0


def _grid_with_inv_dist(seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for x0 in rng.uniform(-1, 1, 120):
        R, us = 30 + 5 * x0, 1 / (3000 * (1 + 0.2 * x0))
        d = np.linspace(1800, 4000, 30)
        rows.append(np.column_stack([np.full(30, x0), d, 1.0 / d,
                                     log_model(1 / d, R, us, 0.4, 0.0) + rng.normal(0, 0.05, 30)]))
    return np.vstack(rows)


def test_inv_dist_does_not_split_the_intrinsic_points():
    # 'inv_dist' is the same axis as 'dist', so it varies across one point's slices: left in the
    # intrinsic key, each of the 30 slices becomes its own one-row point, none reaches the fitter's
    # five-slice minimum, and the continuation silently reverts to the base fit's flat extrapolation
    g = _grid_with_inv_dist()
    base = lambda x: 1000.0 * np.asarray(x, dtype=float)[:, 2]     # reads 1/d: it must see 1/d_edge
    m = RFDistanceTails(base, 1, sides="near", coord_names=["x0", "dist", "inv_dist"]) \
        .fit(g[:, :3], g[:, 3], 0.05 * np.ones(len(g)))
    assert m.report["intrinsic_dim"] == 1
    assert m.report["points"] == 120 and m.report["points_fit"] == 120
    x0 = g[:, 0].min()                                             # unique keys are sorted: this is P[0]
    assert m.dmin[0] == 1800.0
    near = m(np.array([[x0, 600.0, 1.0 / 600.0]]))[0]
    P = m.P[0]
    step = log_model(1 / 600.0, P[0], P[1], P[2], 0.0) - log_model(1 / 1800.0, P[0], P[1], P[2], 0.0)
    # the edge value is the base fit at d_edge in BOTH distance columns (1000/1800, not 1000/600)
    assert np.isclose(near, 1000.0 / 1800.0 + min(step, 0.0))


def test_distance_only_fit_needs_no_tree():
    # --parameter dist on its own: no intrinsic coordinate is left, so there is no tree to build and the
    # whole grid is one fixed point
    rng = np.random.default_rng(7)
    d = np.linspace(1800.0, 4000.0, 30)
    y = log_model(1 / d, 30.0, 1 / 3000.0, 0.4, 0.0) + rng.normal(0, 0.05, 30)
    m = RFDistanceTails(lambda x: np.full(len(x), 7.0), 0, sides="both", coord_names=["dist"]) \
        .fit(d[:, None], y, 0.05 * np.ones(30))
    assert m.tree is None and m.report["intrinsic_dim"] == 0
    assert m.report["points"] == 1 and m.report["points_fit"] == 1
    assert m(np.array([[2500.0]]))[0] == 7.0                       # inside the slices: base unchanged
    assert m(np.array([[600.0]]))[0] < 6.0 and m(np.array([[9000.0]]))[0] < 6.0


def test_shared_fmin_is_reported():
    g = _grid()
    m = RFDistanceTails(lambda x: np.zeros(len(x)), 1).fit(g[:, :2], g[:, 2], 0.05 * np.ones(len(g)))
    assert abs(m.report["fmin_shared"] - 0.4) < 0.1


def test_continuation_never_rises_above_the_edge():
    # slices only on the far side of the peak: the fitted shape rises toward small d beyond the data
    rng = np.random.default_rng(5)
    rows = []
    for x0 in rng.uniform(-1, 1, 80):
        d = np.linspace(4000, 7000, 30)
        rows.append(np.column_stack([np.full(30, x0), d, log_model(1 / d, 30.0, 1 / 2000.0, 0.5, 0.0) + rng.normal(0, 0.05, 30)]))
    g = np.vstack(rows)
    m = RFDistanceTails(lambda x: np.full(len(x), 5.0), 1, sides="both").fit(g[:, :2], g[:, 2], 0.05 * np.ones(len(g)))
    assert np.all(m(np.column_stack([np.full(5, g[0, 0]), [500.0, 1000.0, 2000.0, 9000.0, 20000.0]])) <= 5.0)


def test_point_fit_scipy_default_and_batched_option():
    g = _grid()
    m = RFDistanceTails(lambda x: np.zeros(len(x)), 1).fit(g[:, :2], g[:, 2], 0.05 * np.ones(len(g)))
    b = RFDistanceTails(lambda x: np.zeros(len(x)), 1, point_fit="batched").fit(g[:, :2], g[:, 2], 0.05 * np.ones(len(g)))
    assert m.report["point_fit"] == "scipy" and b.report["point_fit"] == "batched"
    _, inv = np.unique(np.round(g[:, :1], 10), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cost = lambda P: np.sum((log_model(1 / g[:, 1], *P[inv].T[:3], 0.0) - g[:, 2]) ** 2)
    # two fitters (each with its own shared f_min): scipy reaches at most the batched cost
    assert cost(m.P) <= cost(b.P) * (1 + 1e-6)


def test_driver_refuses_tail_options_outside_rf(tmp_path):
    # parse-time refusals: tails need rf; point-fit options on rf need the tails
    import os
    import subprocess
    import sys
    code = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    driver = os.path.join(code, "bin", "util_ConstructEOSPosterior.py")
    env = dict(os.environ, PYTHONPATH=code, OMP_NUM_THREADS="1", MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="",
               MPLCONFIGDIR=str(tmp_path))
    base = [sys.executable, driver, "--fname", str(tmp_path / "missing.dat"), "--parameter", "xx"]
    for extra, msg in ((["--fit-method", "dslice-amp", "--rf-dslice-tails", "near"], "apply to --fit-method rf only"),
                       (["--fit-method", "rf", "--dslice-amp-point-fit-jobs", "2"], "apply to --fit-method dslice-amp only"),
                       (["--fit-method", "rf", "--rf-dslice-tails-fmin", "free"], "needs --rf-dslice-tails"),
                       (["--fit-method", "rf", "--rf-dslice-tails", "near", "--ignore-errors-in-data"], "per-row sigma"),
                       (["--fit-method", "rf", "--rf-dslice-tails", "near"], "'dist' as a fit coordinate")):
        p = subprocess.run(base + extra, env=env, cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=300)
        assert p.returncode == 2 and msg in p.stdout, p.stdout[-2000:]
    # a valid configuration passes the parse-time checks: 'dist' given via --parameter-implied (it then
    # fails later only because the data file does not exist)
    p = subprocess.run(base + ["--fit-method", "rf", "--rf-dslice-tails", "near", "--parameter-implied", "dist"],
                       env=env, cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=300)
    assert p.returncode != 2 and "fit coordinate" not in p.stdout, p.stdout[-2000:]


def _truth_grid(seed=12, n=150, shift=-37.0):
    # two intrinsic coordinates on very different scales; the distance shape depends strongly on x1
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(1000, 3000, n)])
    par = lambda p: (30 + 10 * p[..., 0], 1 / (2500 * (1 + 0.3 * p[..., 0])))
    d = np.linspace(1800, 4000, 30)
    rows = [np.column_stack([np.full(30, p[0]), np.full(30, p[1]), d,
                             log_model(1 / d, *par(p), 0.4, 0.0) + rng.normal(0, 0.01, 30)]) for p in pts]
    truth = lambda x: log_model(1 / x[:, 2], *par(x[:, :2]), 0.4, 0.0)
    return pts, np.vstack(rows), truth, shift


def test_known_answer_nonflat_base_off_grid_queries():
    # base = truth - shift, as the driver's RF sees it; tails are fitted on truth (= Y + shift).
    pts, g, truth, shift = _truth_grid()
    base = lambda x: truth(np.asarray(x)) - shift
    m = RFDistanceTails(base, 2, sides="near").fit(g[:, :3], g[:, 3], 0.01 * np.ones(len(g)))
    rng = np.random.default_rng(3)
    q = pts[:40] + np.column_stack([rng.normal(0, 0.002, 40), rng.normal(0, 1.0, 40)])   # near each point
    for dq in (700.0, 1200.0):
        X = np.column_stack([q, np.full(len(q), dq)])
        err = m(X) - (truth(X) - shift)
        # correct code errs by <= 0.42 here (the tail borrows the nearest point's shape); anchoring at
        # d_max, reading the base at the query distance, or an unstandardized lookup err by 6-30 nats
        assert np.max(np.abs(err)) < 1.0, (dq, np.max(np.abs(err)))
    X = np.column_stack([q, np.full(len(q), 2500.0)])
    assert np.allclose(m(X), base(X))                            # inside the slices: base unchanged
    assert all(np.isclose(m.P[:, 2], m.report["fmin_shared"]))   # shared f_min was refit, not left free
    assert abs(m.report["fmin_shared"] - 0.4) < 0.05


def test_failed_point_nan_rows_and_nonpositive_distance():
    pts, g, truth, shift = _truth_grid(n=40)
    few = np.array([[0.5, 2000.0, 2500.0, 10.0], [0.5, 2000.0, 3000.0, 9.0], [0.5, 2000.0, 3500.0, 8.0]])
    bad = np.array([[np.nan, 1500.0, 2500.0, 5.0], [0.1, 1500.0, 2500.0, np.nan]])
    # a d <= 0 row and a NaN-sigma row on real grid points: each must be dropped, not sink that point's fit
    p0, p1 = g[0, :2], g[30, :2]
    extra = np.array([[p0[0], p0[1], -10.0, 1.0], [p1[0], p1[1], 2600.0, 1.0]])
    G = np.vstack([g, few, bad, extra])
    sig = 0.01 * np.ones(len(G))
    sig[-1] = np.nan
    base = lambda x: np.full(len(x), 3.0)
    m = RFDistanceTails(base, 2, sides="near").fit(G[:, :3], G[:, 3], sig)
    assert m.report["rows_dropped"] == 4 and m.report["points_fit"] == m.report["points"] - 1
    for p in (p0, p1):                                 # both points keep their tails
        assert m(np.array([[p[0], p[1], 900.0]]))[0] < 3.0 - 1
    out = m(np.array([[0.5, 2000.0, 1000.0]]))        # the 3-slice point: no fit, so the base value
    assert out[0] == 3.0
    v = m(np.column_stack([pts[:3], [0.0, -5.0, 900.0]]))
    assert np.all(v[:2] == -np.inf) and np.isfinite(v[2]) and v[2] <= 3.0


def test_nonfinite_rows_go_through_the_base_guard():
    # a base like the driver's CPU RF: it fills non-finite rows itself and, like sklearn, rejects a batch
    # with nothing finite in it; the wrapper must never hand it the non-finite rows on their own
    g = _grid()
    def base(x):
        x = np.asarray(x, dtype=float)
        ok = np.all(np.isfinite(x), axis=1)
        if not np.any(ok):
            raise ValueError("Found array with 0 sample(s)")
        out = np.full(len(x), -500.0)
        out[ok] = 7.0
        return out
    m = RFDistanceTails(base, 1, sides="near").fit(g[:, :2], g[:, 2], 0.05 * np.ones(len(g)))
    v = m(np.array([[g[0, 0], np.nan], [g[0, 0], 2500.0], [np.inf, 900.0], [g[0, 0], 900.0]]))
    assert v[0] == -500.0 and v[2] == -500.0 and v[1] == 7.0 and v[3] < 7.0
