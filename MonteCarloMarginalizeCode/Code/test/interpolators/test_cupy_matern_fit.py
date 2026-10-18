"""cupy_matern_fit reproduces the sklearn Matérn recipe of matern_gp.

Run with xp=numpy so the identity is checked on any host; the cupy leg runs only with a GPU.
"""
import numpy as np
import pytest

from RIFT.interpolators.cupy_matern_fit import _nlml_and_grad, fit_matern_gp_cupy
from RIFT.interpolators.matern_gp import fit_matern_gp


def _data(n=400, d=3, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, d)) * np.array([1.0, 3.0, 0.2])[:d]
    y = 30 - 0.5 * np.sum((x / np.array([0.4, 1.5, 0.1])[:d]) ** 2, 1) + rng.normal(0, 0.1, n)
    return x, y, 0.1 * np.ones(n)


def test_gradient_matches_finite_differences():
    x, y, e = _data(n=120)
    xs = (x - x.mean(0)) / x.std(0)
    yn = (y - y.mean()) / y.std()
    alpha = e ** 2 / y.std() ** 2
    theta = np.log(np.r_[1.3, 0.7, 2.0, 0.4, 3e-3])
    f0, g = _nlml_and_grad(theta, xs, yn, alpha, np)
    h = 1e-6
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        fd = (_nlml_and_grad(tp, xs, yn, alpha, np)[0] - _nlml_and_grad(tm, xs, yn, alpha, np)[0]) / (2 * h)
        assert abs(fd - g[k]) < 1e-4 * max(1.0, abs(fd)), (k, fd, g[k])


def _interior_data(n=400, seed=5):
    """Smooth target whose reported errors understate the scatter, so the fitted constant and white
    noise land inside their bounds and every term of the final weights matters."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 3))
    y = 2 * np.sin(2 * x[:, 0]) + np.cos(3 * x[:, 1]) + 0.5 * np.sin(4 * x[:, 2]) + rng.normal(0, 0.15, n)
    return x, y, 0.05 * np.ones(n)


def test_matches_sklearn_recipe_interior():
    x, y, e = _interior_data()
    ref, rec_ref = fit_matern_gp(x, y, e, max_train_points=400)
    mine, rec = fit_matern_gp_cupy(x, y, e, max_train_points=400, xp=np)
    assert rec["optimizer"]["at_bounds"] == [], rec["theta"]
    assert np.allclose(rec["theta"], rec_ref["theta"], atol=1e-4)
    q = np.random.default_rng(9).uniform(-1, 1, (500, 3))
    assert np.max(np.abs(mine.predict(q) - ref.predict(q))) < 1e-6


def test_matches_sklearn_recipe():
    x, y, e = _data()
    ref, rec_ref = fit_matern_gp(x, y, e, max_train_points=300)
    mine, rec = fit_matern_gp_cupy(x, y, e, max_train_points=300, xp=np)
    assert rec["selected_indices_sha256"] == rec_ref["selected_indices_sha256"]
    q = np.random.default_rng(9).uniform(-1, 1, (500, 3)) * np.array([1.0, 3.0, 0.2])
    diff = np.abs(mine.predict(q) - ref.predict(q))
    # same objective, start, bounds and optimizer; residual differences are optimizer round-off
    assert diff.max() < 1e-3, diff.max()
    assert np.isclose(rec["optimizer"]["nlml"], rec_ref["optimization"][0]["negative_log_marginal_likelihood"],
                      rtol=1e-6, atol=1e-6)


def test_cupy_backend_agrees_with_numpy():
    try:
        import cupy as cp
        cp.zeros(1) + 1
    except Exception as exc:  # driverless or unsupported card
        pytest.skip("no usable CUDA device: %s" % exc)
    x, y, e = _data()
    a, _ = fit_matern_gp_cupy(x, y, e, max_train_points=300, xp=np)
    b, _ = fit_matern_gp_cupy(x, y, e, max_train_points=300)
    q = np.random.default_rng(1).uniform(-1, 1, (300, 3))
    assert np.max(np.abs(a.predict(q) - b.predict(q))) < 1e-6
