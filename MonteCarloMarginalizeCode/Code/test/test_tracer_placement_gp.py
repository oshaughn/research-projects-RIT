"""
Tests for the linear-mean GP fit and the optional lnL floor in
RIFT/misc/tracer_placement/fits/.

Headline test: `test_gp_extrapolates_where_rf_goes_flat` builds a synthetic lnL
surface whose peak lies OUTSIDE the training hull -- the clipped-peak failure
that motivated the port -- and checks that gp_linmean keeps rising toward the
peak where the random forest is exactly flat.

These intentionally avoid importing the RIFT package proper (RIFT/__init__.py
pulls in lalsimutils + lalsuite, which the placement engine does not need), by
putting RIFT/misc on sys.path and importing `tracer_placement` directly. That
is the same fallback import path the two tracer CLI tools use for local dev::

    python test/test_tracer_placement_gp.py
    pytest test/test_tracer_placement_gp.py

or, with a self-contained environment that needs no lalsuite::

    cd test/tracer_placement && pixi run test

sklearn is needed only for the `rf` half of the comparison and scipy only for
`rbf`; those checks skip cleanly without them. Everything about the GP itself is
numpy-only.
"""

import ast
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_MISC = os.path.normpath(os.path.join(HERE, "..", "RIFT", "misc"))
_BIN = os.path.normpath(os.path.join(HERE, "..", "bin"))
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)

from tracer_placement import fits, samplers          # noqa: E402
from tracer_placement.fits._gp_linmean import LinearMeanGPFit   # noqa: E402

try:
    import sklearn                                   # noqa: F401
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

try:
    import scipy                                     # noqa: F401
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

try:
    import pytest
    _skip_no_sklearn = pytest.mark.skipif(
        not _HAVE_SKLEARN, reason="sklearn not installed; rf fit unavailable")
except ImportError:                                  # pytest-free execution
    pytest = None

    def _skip_no_sklearn(fn):
        return fn


# --------------------------------------------------------------------------- #
# Synthetic surfaces
# --------------------------------------------------------------------------- #

# The clipped-peak geometry: a Gaussian lnL ridge peaked at x = X_PEAK, but the
# grid we are allowed to train on only reaches x = X_EDGE. Inside the training
# box lnL rises monotonically with x and simply runs off the edge -- exactly
# R3's batch-0 situation at the v_outer wall.
X_PEAK, Y_PEAK = 3.0, 0.5
X_EDGE = 1.0


def _true_lnL(Z):
    Z = np.atleast_2d(Z)
    return -0.5 * ((Z[:, 0] - X_PEAK) ** 2 / 0.8 ** 2
                   + (Z[:, 1] - Y_PEAK) ** 2 / 0.5 ** 2)


def _clipped_training_set(n=200, seed=0, noise=0.0):
    """Draw a training grid confined to x in [0, X_EDGE] (peak is outside)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(0.0, X_EDGE, n),
                         rng.uniform(0.0, 1.0, n)])
    Y = _true_lnL(X)
    if noise:
        Y = Y + noise * rng.normal(size=n)
    sigma = np.full(n, max(noise, 1e-2))
    return X, Y, sigma


def _ray_toward_peak(x_values):
    """Points marching from inside the hull out toward the true peak."""
    return np.column_stack([np.asarray(x_values, dtype=float),
                            np.full(len(x_values), Y_PEAK)])


# --------------------------------------------------------------------------- #
# The headline argument: extrapolation past the training hull
# --------------------------------------------------------------------------- #

@_skip_no_sklearn
def test_gp_extrapolates_where_rf_goes_flat():
    """gp_linmean chases a peak outside the training hull; rf cannot.

    This is the whole argument for adding the fit. The random forest is
    piecewise-constant, so every point beyond the training hull falls in the
    same boundary leaf and gets the same prediction -- placement sees zero
    gradient and no reason to leave the box. The linear-mean GP carries the
    fitted trend outward and keeps rising toward the true peak.
    """
    X, Y, sigma = _clipped_training_set()
    gp = fits.build("gp_linmean", X, Y, sigma=sigma)
    rf = fits.build("rf", X, Y, sigma=sigma)

    inside = _ray_toward_peak([0.9])
    outside = _ray_toward_peak([1.5, 2.0, 2.5, 3.0])

    rf_in = rf.predict(inside)[0]
    rf_out = rf.predict(outside)
    gp_in = gp.predict(inside)[0]
    gp_out = gp.predict(outside)

    lnL_scale = float(np.ptp(Y))

    # 1. rf is flat outside the hull: identical predictions and, what actually
    #    matters for placement, exactly zero gradient to climb.
    assert np.ptp(rf_out) < 1e-9 * max(lnL_scale, 1.0), (
        "rf should be piecewise-constant outside the training hull, "
        f"got spread {np.ptp(rf_out)}")
    assert np.allclose(rf.grad(outside), 0.0), (
        "rf should offer no gradient outside the hull, got "
        f"{rf.grad(outside)}")
    assert np.all(np.abs(gp.grad(outside)[:, 0]) > 1e-3), (
        "gp_linmean should still have a gradient to climb outside the hull")

    # 2. gp_linmean keeps rising toward the peak, monotonically.
    assert np.all(np.diff(gp_out) > 0), (
        f"gp_linmean should rise toward the peak outside the hull, got {gp_out}")
    assert gp_out[-1] - gp_in > 0.5 * lnL_scale, (
        "gp_linmean extrapolation should gain a substantial fraction of the "
        f"in-hull lnL range; got {gp_out[-1] - gp_in:g} vs range {lnL_scale:g}")

    # 3. Stated as placement sees it: maximizing the surrogate over a box that
    #    extends past the old edge moves the GP's argmax outside, while rf's
    #    surface is flat there so it offers no improvement at all.
    grid = np.column_stack([np.linspace(0.0, 3.5, 141),
                            np.full(141, Y_PEAK)])
    outside_mask = grid[:, 0] > X_EDGE
    gp_grid = gp.predict(grid)
    rf_grid = rf.predict(grid)
    assert grid[np.argmax(gp_grid), 0] > X_EDGE, (
        "gp_linmean's best point should lie outside the sampled region")
    rf_gain = rf_grid[outside_mask].max() - rf_grid[~outside_mask].max()
    assert rf_gain <= 1e-9, (
        f"rf should see no improvement outside the hull, got gain {rf_gain}")


def test_linear_mean_extrapolates_where_const_mean_reverts():
    """The mean function, not the kernel, is what buys extrapolation.

    Same GP, same kernel, same data: with mean="const" the surrogate relaxes
    back toward a flat prior away from the data (the zero-mean-GP failure CIP's
    --lnL-shift-prevent-overflow help text warns about); with mean="linear" it
    follows the trend. Runs without sklearn.
    """
    X, Y, sigma = _clipped_training_set()
    gp_lin = LinearMeanGPFit(X, Y, sigma=sigma, mean="linear")
    gp_const = LinearMeanGPFit(X, Y, sigma=sigma, mean="const")

    ray = _ray_toward_peak([0.9, 1.5, 2.0, 2.5, 3.0])
    lin = gp_lin.predict(ray)
    const = gp_const.predict(ray)

    assert np.all(np.diff(lin) > 0), f"linear mean should keep rising: {lin}"
    # The constant-mean fit decays back to the training mean, i.e. it gives up
    # the gain it had at the hull edge.
    assert const[-1] < const[0], f"const mean should revert away from data: {const}"
    assert lin[-1] > const[-1] + 0.5 * float(np.ptp(Y))


def test_uncertainty_grows_outside_the_hull():
    """predict_with_std is the calibrated sigma samplers.ucb asks for."""
    X, Y, sigma = _clipped_training_set(noise=0.05, seed=3)
    gp = fits.build("gp_linmean", X, Y, sigma=sigma)
    assert gp.has_uncertainty is True
    assert gp.smooth_gradient is True

    _, s_train = gp.predict_with_std(X)
    _, s_far = gp.predict_with_std(_ray_toward_peak([2.5, 3.0]))
    assert np.median(s_train) < np.min(s_far), (
        "GP sigma must be smaller on training points than in the unsampled "
        f"frontier; got median {np.median(s_train):g} vs far {s_far}")
    # Far from any data the posterior std saturates at the signal amplitude.
    assert np.all(s_far <= np.sqrt(gp.sf2) * (1 + 1e-8))


# --------------------------------------------------------------------------- #
# GP mechanics
# --------------------------------------------------------------------------- #

def test_gp_interpolates_training_data():
    """With small observation noise the fit reproduces its training values."""
    X, Y, _ = _clipped_training_set(n=60, seed=1)
    gp = LinearMeanGPFit(X, Y, sigma=np.full(len(Y), 1e-3), sigma_floor=1e-3)
    assert gp.train_rms < 0.02 * float(np.ptp(Y)), gp.train_rms
    assert np.allclose(gp.predict(X), Y, atol=0.05 * float(np.ptp(Y)))


def test_analytic_grad_matches_finite_difference():
    X, Y, sigma = _clipped_training_set(n=80, seed=2)
    gp = LinearMeanGPFit(X, Y, sigma=sigma)
    Z = np.array([[0.4, 0.6], [0.9, 0.2], [2.0, 0.5]])
    g = gp.grad(Z)
    eps = 1e-5
    fd = np.zeros_like(Z)
    for k in range(Z.shape[1]):
        zp = Z.copy(); zp[:, k] += eps
        zm = Z.copy(); zm[:, k] -= eps
        fd[:, k] = (gp.predict(zp) - gp.predict(zm)) / (2 * eps)
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-5), (g, fd)


def test_shapes_and_one_dimensional_input():
    X, Y, sigma = _clipped_training_set(n=40, seed=4)
    gp = LinearMeanGPFit(X, Y, sigma=sigma)
    for Z, n_expected in ((np.array([0.5, 0.5]), 1), (X[:7], 7)):
        mu = gp.predict(Z)
        m2, s2 = gp.predict_with_std(Z)
        assert mu.shape == (n_expected,)
        assert m2.shape == (n_expected,) and s2.shape == (n_expected,)
        assert np.allclose(mu, m2)
        assert np.all(np.isfinite(s2)) and np.all(s2 >= 0)
        assert gp.grad(Z).shape == (n_expected, 2)

    # A genuinely 1-D parameter space must work too (RIFT runs those).
    X1 = np.linspace(0, 1, 30)[:, None]
    gp1 = LinearMeanGPFit(X1, np.sin(3 * X1[:, 0]))
    assert gp1.predict(np.array([[0.5]])).shape == (1,)


def test_prediction_chunking_is_seamless():
    """predict_with_std chunks internally; results must not depend on that."""
    X, Y, sigma = _clipped_training_set(n=50, seed=5)
    gp = LinearMeanGPFit(X, Y, sigma=sigma)
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 4, size=(5000, 2))     # > the internal 2048 chunk
    mu, sd = gp.predict_with_std(Z)
    mu_ref = gp.predict(Z)
    assert np.allclose(mu, mu_ref)
    assert np.all(np.isfinite(sd))


def test_bad_inputs_are_rejected_loudly():
    X, Y, _ = _clipped_training_set(n=20, seed=6)
    try:
        LinearMeanGPFit(X, Y, mean="cubic")
        raise AssertionError("expected ValueError for unknown mean")
    except ValueError:
        pass
    try:
        LinearMeanGPFit(X, Y[:-1])
        raise AssertionError("expected ValueError for mismatched lengths")
    except ValueError:
        pass
    Y_bad = Y.copy(); Y_bad[3] = -np.inf
    try:
        LinearMeanGPFit(X, Y_bad)
        raise AssertionError("expected ValueError for non-finite lnL")
    except ValueError as e:
        assert "lnl_floor_delta" in str(e)      # points at the supported remedy


def test_nonpositive_length_scale_is_refused():
    """ls=0 silently produced all-NaN predictions and ls<0 silently gave a
    DIFFERENT fit than asked for (the sign is squared away). Both are
    silent-wrong, so the constructor must refuse them."""
    X, Y, _ = _clipped_training_set(n=30, seed=14)
    for bad in (0.0, -1.0, np.nan, np.inf):
        try:
            LinearMeanGPFit(X, Y, length_scale=bad)
            raise AssertionError(f"expected ValueError for length_scale={bad}")
        except ValueError as e:
            assert "length_scale" in str(e)
    gp = LinearMeanGPFit(X, Y, length_scale=0.5)
    assert np.all(np.isfinite(gp.predict(X)))


def test_large_candidate_pools_stay_chunked():
    """Every public evaluator must chunk. An unchunked (m, n) kernel block at
    UCB's pool size is hundreds of MB on its own -- enough to blow a modest
    Condor memory request."""
    import tracemalloc
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1500, 3))
    gp = LinearMeanGPFit(X, X.sum(axis=1))
    Z = rng.uniform(0, 1, (20000, 3))
    n_block = 1500 * 20000 * 8            # what one unchunked block would cost
    for name in ("predict", "predict_with_std", "grad"):
        tracemalloc.start()
        getattr(gp, name)(Z)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < 0.5 * n_block, (
            f"{name} peaked at {peak/1e6:.0f} MB; an unchunked block would be "
            f"{n_block/1e6:.0f} MB, so this is not chunking")


def test_warns_when_the_linear_mean_is_underdetermined():
    """With fewer points than mean coefficients, lstsq returns the min-norm
    hyperplane -- an arbitrary pick among infinitely many. This fit exists to
    extrapolate along that hyperplane, so it must not do so quietly."""
    import io
    import contextlib
    rng = np.random.default_rng(1)
    d = 5
    for n, expect_warning in ((3, True), (d + 1, True), (40, False)):
        X = rng.uniform(0, 1, (n, d))
        Y = X[:, 0] * 3.0
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            LinearMeanGPFit(X, Y)
        got = "mean function is" in err.getvalue()
        assert got is expect_warning, (n, err.getvalue())
        if expect_warning:
            assert "extrapolation" in err.getvalue().lower()
    # mean="const" has one coefficient, so it is not subject to this at all.
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        LinearMeanGPFit(rng.uniform(0, 1, (3, d)), rng.normal(size=3), mean="const")
    assert "mean function is" not in err.getvalue()


def test_duplicate_points_do_not_break_the_cholesky():
    """Repeated grid rows are common in RIFT unions; jitter must absorb them."""
    X, Y, sigma = _clipped_training_set(n=30, seed=7)
    X = np.vstack([X, X[:5]])
    Y = np.concatenate([Y, Y[:5]])
    sigma = np.concatenate([sigma, sigma[:5]])
    gp = LinearMeanGPFit(X, Y, sigma=sigma)
    assert np.all(np.isfinite(gp.predict(X)))


# --------------------------------------------------------------------------- #
# Dispatch registration
# --------------------------------------------------------------------------- #

def test_dispatch_registers_gp_linmean():
    X, Y, sigma = _clipped_training_set(n=30, seed=8)
    for name in ("gp_linmean", "GP_LINMEAN", "gp-linmean"):
        assert isinstance(fits.build(name, X, Y, sigma=sigma), LinearMeanGPFit)
    try:
        fits.build("no_such_fit", X, Y)
        raise AssertionError("expected ValueError for unknown method")
    except ValueError:
        pass


def test_gp_kwargs_pass_through_dispatch():
    X, Y, sigma = _clipped_training_set(n=30, seed=9)
    gp = fits.build("gp_linmean", X, Y, sigma=sigma,
                    mean="const", length_scale=0.7)
    assert gp.mean_kind == "const"
    assert gp.length_scale == 0.7


# --------------------------------------------------------------------------- #
# Task 2: the optional lnL floor
# --------------------------------------------------------------------------- #

def test_lnl_floor_off_by_default_is_a_pass_through():
    """Legacy behaviour must be bit-for-bit unchanged: same object, untouched."""
    Y = np.array([1.0, -1e9, 3.0])
    assert fits.apply_lnl_floor(Y, None) is Y


def test_lnl_floor_clamps_without_dropping_points():
    Y = np.array([10.0, 9.0, -1e9, 8.0, -np.inf])
    out = fits.apply_lnl_floor(Y, 100.0)
    assert len(out) == len(Y), "the floor clamps, it does not cut"
    assert out.min() == -90.0                     # max(Y)=10 -> floor 10-100
    assert np.array_equal(out[:2], Y[:2])         # good points untouched
    assert np.all(np.isfinite(out))

    for bad in (0.0, -5.0, np.inf):
        try:
            fits.apply_lnl_floor(Y, bad)
            raise AssertionError(f"expected ValueError for delta={bad}")
        except ValueError:
            pass

    # NaN is a failed evaluation: same kind of anchor as a catastrophic one.
    assert fits.apply_lnl_floor(np.array([1.0, np.nan, 3.0]), 10.0)[1] == -7.0

    # +inf is not something a floor can rescue. Letting it through used to
    # fail downstream with a message telling the user to apply the floor they
    # had just applied.
    try:
        fits.apply_lnl_floor(np.array([1.0, np.inf, 3.0]), 10.0)
        raise AssertionError("expected ValueError for +inf lnL")
    except ValueError as e:
        assert "+inf" in str(e)


def test_lnl_floor_rescues_a_gp_fit_wrecked_by_an_outlier():
    """The reason to floor rather than cut: a single -1e9 point otherwise
    inflates the residual scatter so much that the kernel term is numerically
    irrelevant and the surrogate degenerates to its mean function."""
    X, Y, sigma = _clipped_training_set(n=60, seed=10)
    Y_bad = Y.copy()
    Y_bad[0] = -1e9                               # catastrophic model failure

    gp_raw = fits.build("gp_linmean", X, Y_bad, sigma=sigma)
    gp_floored = fits.build("gp_linmean", X, Y_bad, sigma=sigma,
                            lnl_floor_delta=50.0)

    good = np.ones(len(Y), dtype=bool); good[0] = False
    err_raw = np.sqrt(np.mean((gp_raw.predict(X[good]) - Y[good]) ** 2))
    err_floored = np.sqrt(np.mean((gp_floored.predict(X[good]) - Y[good]) ** 2))
    assert err_floored < 0.05 * err_raw, (err_floored, err_raw)

    # The floored point is still in the fit as an anchor: the surrogate knows
    # that corner of the space is bad rather than never having heard of it.
    assert gp_floored.predict(X[:1])[0] < Y[good].min()


def test_lnl_floor_applies_to_every_fit_method():
    X, Y, sigma = _clipped_training_set(n=40, seed=11)
    Y_bad = Y.copy(); Y_bad[0] = -1e9
    methods = ["quadratic", "polynomial", "gp_linmean"]
    if _HAVE_SKLEARN:
        methods.append("rf")
    if _HAVE_SCIPY:
        methods.append("rbf")
    for m in methods:
        f = fits.build(m, X, Y_bad, sigma=sigma, lnl_floor_delta=50.0)
        assert np.all(np.isfinite(f.predict(X))), m


# --------------------------------------------------------------------------- #
# Integration: UCB placement, and the two CLI wrappers
# --------------------------------------------------------------------------- #

def test_ucb_placement_with_gp_surrogate_leaves_the_sampled_region():
    """End-to-end through samplers.ucb: with a GP surrogate whose trend points
    out of the box, UCB should place points beyond the old edge."""
    X, Y, sigma = _clipped_training_set(n=120, seed=12)
    gp = fits.build("gp_linmean", X, Y, sigma=sigma)
    prior_box = np.array([[0.0, 3.5], [0.0, 1.0]])     # extended in x
    X_out, info = samplers.ucb_place(
        X[:40], surrogate=gp, prior_box=prior_box,
        rng=np.random.default_rng(0), kappa=2.0,
        n_candidates=4000, polish_steps=5)
    assert X_out.shape == (40, 2)
    assert np.all(np.isfinite(X_out))
    assert np.all(X_out[:, 0] >= prior_box[0, 0] - 1e-9)
    assert np.all(X_out[:, 0] <= prior_box[0, 1] + 1e-9)
    assert info["polish_strategy"] == "gradient"
    assert np.mean(X_out[:, 0] > X_EDGE) > 0.5, (
        "UCB on a linear-mean GP should mostly place outside the old hull")


def _parser_choices_via_ast(path, flag):
    """Read an argparse `choices=` tuple out of a source file without importing
    it. util_ParameterTracerUpdate.py imports lalsimutils/lalsuite at module
    scope, which this test deliberately does not require."""
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                return [ast.literal_eval(e) for e in kw.value.elts]
        return []
    return None


def test_both_cli_tools_offer_gp_linmean_and_the_floor():
    for tool in ("util_HyperparameterTracerUpdate.py", "util_ParameterTracerUpdate.py"):
        path = os.path.join(_BIN, tool)
        choices = _parser_choices_via_ast(path, "--tracer-fit-method")
        assert choices is not None, f"{tool}: no --tracer-fit-method"
        assert "gp_linmean" in choices, (tool, choices)
        assert _parser_choices_via_ast(path, "--tracer-lnl-floor-delta") is not None, (
            f"{tool}: --tracer-lnl-floor-delta not defined")


def test_hyperpipe_passes_the_floor_flag_through():
    """The hyperpipe drives the tracer via a yaml-key -> CLI-flag table; a new
    flag is unreachable from a config unless it is listed there. Read the table
    statically (util_RIFT_hyperpipe.py needs hydra to import)."""
    with open(os.path.join(_BIN, "util_RIFT_hyperpipe.py")) as f:
        tree = ast.parse(f.read())
    # Several stages define a `setting_flags` table; take the puff one.
    table = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "setting_flags"):
            candidate = dict(ast.literal_eval(node.value))
            if "tracer-fit-method" in candidate:
                table = candidate
    assert table is not None, "puff setting_flags table not found"
    assert table.get("tracer-lnl-floor-delta") == "--tracer-lnl-floor-delta"
    assert table.get("tracer-fit-method") == "--tracer-fit-method"


def test_hyperparameter_tool_end_to_end_with_gp():
    """Run the hyperpipe CLI wrapper for real on a small .dat grid.

    (The event-level twin needs lalsuite for its XML I/O, so it is covered by
    the parser check above rather than an end-to-end run.)"""
    sys.path.insert(0, _BIN)
    try:
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "util_HyperparameterTracerUpdate",
            os.path.join(_BIN, "util_HyperparameterTracerUpdate.py"))
        tool = ilu.module_from_spec(spec)
        spec.loader.exec_module(tool)
    finally:
        sys.path.remove(_BIN)
    assert tool._TRACER_OK, "tracer engine not importable from the CLI tool"

    X, Y, sigma = _clipped_training_set(n=60, seed=13)
    Y[0] = -1e9                                    # exercise the floor too
    rows = np.column_stack([Y, sigma, X])
    tmpdir = tempfile.mkdtemp()
    try:
        fin = os.path.join(tmpdir, "grid.dat")
        fout = os.path.join(tmpdir, "grid_out.dat")
        np.savetxt(fin, rows, header="lnL sigma_lnL p1 p2")

        tool.main(["--inj-file", fin, "--inj-file-out", fout,
                   "--parameter", "p1", "--parameter", "p2",
                   "--update-method", "ucb", "--tracer-fit-method", "gp_linmean",
                   "--tracer-lnl-floor-delta", "50",
                   "--ucb-n-candidates", "2000", "--rng-seed", "0"])

        out = np.loadtxt(fout)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    assert out.shape[1] == rows.shape[1]
    assert len(out) == len(rows)
    assert np.all(np.isfinite(out))
    assert np.all(out[:, 0] == 0) and np.all(out[:, 1] == 0)   # puffball convention


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        if not _HAVE_SKLEARN and name == "test_gp_extrapolates_where_rf_goes_flat":
            print(f"SKIP {name} (no sklearn)")
            continue
        try:
            fn()
            print(f"ok   {name}")
        except Exception:
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    # Print the headline numbers so the argument is visible, not just asserted.
    if _HAVE_SKLEARN:
        X, Y, sigma = _clipped_training_set()
        gp = fits.build("gp_linmean", X, Y, sigma=sigma)
        rf = fits.build("rf", X, Y, sigma=sigma)
        ray = _ray_toward_peak([0.9, 1.5, 2.0, 2.5, 3.0])
        print("\n x (peak at %.1f, training hull ends at %.1f)" % (X_PEAK, X_EDGE))
        print("      x:   " + "  ".join(f"{v:8.3f}" for v in ray[:, 0]))
        print(" true lnL: " + "  ".join(f"{v:8.3f}" for v in _true_lnL(ray)))
        print(" gp_linmean:" + " ".join(f"{v:8.3f}" for v in gp.predict(ray)))
        print(" rf:        " + " ".join(f"{v:8.3f}" for v in rf.predict(ray)))
    sys.exit(1 if fails else 0)
