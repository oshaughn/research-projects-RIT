"""
Self-contained regression tests for the jax_gp interpolators.

Run directly (``python -m RIFT.interpolators.jax_gp.test_interpolators``) or under
pytest.  They are deliberately small/fast: the heavy accuracy-vs-scale study
lives in ``benchmark/harness.py``.

What they pin down:
  * each backend recovers a smooth analytic target to a sane tolerance,
  * the differentiable contract works -- ``lnL_and_grad`` agrees with a finite
    difference of ``predict`` (this is the property downstream users rely on),
  * the exported predictive mean and the AD value agree.
"""
from __future__ import annotations

import os

import numpy as np


def _target(X):
    # smooth, anisotropic quadratic bowl -- exactly representable-ish, known grad
    A = np.diag([1.0, 0.5, 2.0, 0.3])
    return -0.5 * np.einsum("ni,ij,nj->n", X, A, X)


def _make(method):
    from . import get_interpolator
    cls = get_interpolator(method)
    if method == "rff":
        return cls(n_features=400, n_opt_steps=150)
    if method == "svgp":
        return cls(n_inducing=128, n_opt_steps=250)
    return cls(n_opt_steps=150)


def _check_method(method, rmse_tol, fd_tol):
    rng = np.random.default_rng(0)
    d, n = 4, 600
    X = rng.normal(size=(n, d))
    y = _target(X) + 0.01 * rng.normal(size=n)
    model = _make(method).fit(X, y)

    Xt = rng.normal(size=(150, d))
    rmse = float(np.sqrt(np.mean((model.predict(Xt) - _target(Xt)) ** 2)))
    assert rmse < rmse_tol, "{}: rmse {:.4f} >= {}".format(method, rmse, rmse_tol)

    # AD gradient vs finite difference of predict, at a near-peak point
    x0 = np.array([0.3, -0.2, 0.1, 0.0])
    v, g = model.lnL_and_grad(x0)
    assert np.isclose(v, model.predict(x0[None])[0], atol=1e-5), method
    eps = 1e-4
    fd = np.zeros(d)
    for i in range(d):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps; xm[i] -= eps
        fd[i] = (model.predict(xp[None])[0] - model.predict(xm[None])[0]) / (2 * eps)
    rel = np.linalg.norm(g - fd) / np.linalg.norm(fd)
    assert rel < fd_tol, "{}: AD vs FD rel err {:.4f} >= {}".format(method, rel, fd_tol)
    return rmse, rel


def _check_export_roundtrip(method):
    import tempfile
    import jax
    from . import export
    rng = np.random.default_rng(1)
    d, n = 4, 500
    X = rng.normal(size=(n, d))
    y = _target(X)
    model = _make(method).fit(X, y)

    Xt = rng.normal(size=(50, d))
    pred_before = model.predict(Xt)
    x0 = Xt[0]
    v0, g0 = model.lnL_and_grad(x0)

    with tempfile.TemporaryDirectory() as tmp:
        base = export.save(model, os.path.join(tmp, "fit"),
                           coord_names=["a", "b", "c", "e"])
        assert export.exists(base)
        loaded = export.load(base)

    pred_after = loaded.predict(Xt)
    assert np.allclose(pred_before, pred_after, atol=1e-5), method
    v1, g1 = loaded.lnL_and_grad(x0)
    assert np.isclose(v0, v1, atol=1e-5), method
    assert np.allclose(g0, g1, atol=1e-5), method
    # and the reloaded model is still differentiable via jax directly
    gj = jax.grad(loaded.lnL_physical)(np.asarray(x0))
    assert np.allclose(np.asarray(gj), g1, atol=1e-5), method
    assert loaded.coord_names == ["a", "b", "c", "e"]


def _check_heteroscedastic(method):
    """Per-point errors should down-weight noisy labels: fitting WITH the reported
    errors should recover the truth better than ignoring them."""
    rng = np.random.default_rng(3)
    d, n = 3, 700
    X = rng.normal(size=(n, d))
    A = np.diag([1.0, 0.6, 1.4])
    truth = -0.5 * np.einsum("ni,ij,nj->n", X, A, X)
    # heteroscedastic noise: half the points are very noisy
    sigma = np.where(rng.random(n) < 0.5, 1.5, 0.05)
    y = truth + sigma * rng.normal(size=n)

    Xt = rng.normal(size=(300, d))
    tt = -0.5 * np.einsum("ni,ij,nj->n", Xt, A, Xt)

    def rmse(m):
        return float(np.sqrt(np.mean((m.predict(Xt) - tt) ** 2)))

    with_err = rmse(_make(method).fit(X, y, y_errors=sigma))
    no_err = rmse(_make(method).fit(X, y))
    assert with_err <= no_err + 1e-3, \
        "{}: using errors did not help ({:.3f} vs {:.3f})".format(method, with_err, no_err)
    return with_err, no_err


def test_exact():
    _check_method("exact", rmse_tol=0.05, fd_tol=1e-3)


def test_heteroscedastic_svgp():
    _check_heteroscedastic("svgp")


def test_heteroscedastic_exact():
    _check_heteroscedastic("exact")


def test_export_roundtrip_rff():
    _check_export_roundtrip("rff")


def test_export_roundtrip_svgp():
    _check_export_roundtrip("svgp")


def test_export_roundtrip_exact():
    _check_export_roundtrip("exact")


def test_rff():
    _check_method("rff", rmse_tol=0.30, fd_tol=1e-2)


def test_svgp():
    _check_method("svgp", rmse_tol=0.30, fd_tol=1e-2)


if __name__ == "__main__":
    for m in ("exact", "rff", "svgp"):
        rmse, rel = _check_method(m, rmse_tol=0.30 if m != "exact" else 0.05,
                                  fd_tol=1e-2 if m != "exact" else 1e-3)
        _check_export_roundtrip(m)
        we, ne = _check_heteroscedastic(m)
        print("{:6s} OK  rmse={:.4f}  AD-vs-FD relerr={:.2e}  export OK  "
              "hetero rmse(with/without err)={:.3f}/{:.3f}".format(m, rmse, rel, we, ne))
